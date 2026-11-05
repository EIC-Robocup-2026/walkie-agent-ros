"""The Runner: one asyncio loop thread that runs the agent's tasks one at a time.

Whatever owns ROS later calls `hear`, `submit`, `stop` and `reset` from its own
threads. They only queue work and return, so an executor thread never waits for an
LLM. All state is in the checkpoint, and only the loop thread touches it, so a stop
or a reset is serialized with the active run by coming later in the queue.
"""

import asyncio
import logging
import secrets
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from walkie_agent.capabilities import Ctx
from walkie_agent.runtime.middleware import ToolLog
from walkie_agent.runtime.todos import parse_seed

logger = logging.getLogger(__name__)

STOP_LINE = "Okay, stopping."
FAILURE_LINE = "The task failed with an error, so I stopped working on it."
_STOP_WORDS = frozenset({"stop", "never mind", "nevermind"})


def is_stop_utterance(text: str) -> bool:
    """True for a bare stop only. A compound such as "stop and come here" goes to the LLM."""
    return " ".join(text.lower().strip(" \t\r\n.,!?").split()) in _STOP_WORDS


def _new_thread_id() -> str:
    return f"walkie-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


@dataclass(frozen=True)
class Session:
    """One conversation: one checkpointer thread, and one Langfuse session."""

    thread_id: str = field(default_factory=_new_thread_id)

    def config(self, callbacks: Iterable[BaseCallbackHandler] = ()) -> dict[str, Any]:
        """The config for ainvoke. max_concurrency=1 runs one message's calls in order.

        That serial order is observed behaviour of the installed versions, not
        documented, so tests/test_runtime.py pins it. Rerun it on any dependency update.
        """
        return {
            "configurable": {"thread_id": self.thread_id},
            "max_concurrency": 1,
            "metadata": {"langfuse_session_id": self.thread_id},
            "callbacks": list(callbacks),
        }


@dataclass(frozen=True)
class _Task:
    text: str
    seed: list[dict[str, Any]] | None  # parse_seed's new items, which carry their source


@dataclass(frozen=True)
class _Stop:
    text: str


class _Reset:
    pass


class _Shutdown:
    pass


class Runner:
    """Queues tasks for one agent and runs them on its own thread, in order.

    A stop sets the shared `ctx.cancel` at once, which ends a drive through the verb
    and makes StopLatch refuse the rest of the run. It holds until the next task
    starts. A stop or a reset also calls `ctx.base.stop()`, so a goal that outlived its
    verb, such as one whose first cancel Nav2 refused, is asked again. `on_reply(text)` gets the final text of a task, and `on_todos(items,
    changed)`, the same listener as build_agent's, gets the empty list at a start or a
    reset. They run on the loop thread and must not block. `tool_log` logs the stop line.
    """

    def __init__(
        self,
        agent: CompiledStateGraph,
        ctx: Ctx,
        *,
        on_todos: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
        on_reply: Callable[[str], None] | None = None,
        callbacks: Iterable[BaseCallbackHandler] = (),
        tool_log: ToolLog | None = None,
    ) -> None:
        self.agent = agent
        self.ctx = ctx
        self.session = Session()
        self._on_todos = on_todos
        self._on_reply = on_reply
        self._callbacks = list(callbacks)
        self._tool_log = tool_log
        # The lock makes "take a task and clear the token" one step against "set the
        # token and drop the queue". Otherwise a stop in between would be lost.
        self._lock = threading.Lock()
        self._queue: deque[_Task | _Stop | _Reset | _Shutdown] = deque()
        self._closed = False
        self._idle = threading.Event()
        self._idle.set()
        self._loop = asyncio.new_event_loop()
        self._wake = asyncio.Event()
        self._speaking: set[asyncio.Task[None]] = set()
        self._thread = threading.Thread(target=self._main, name="walkie-runtime", daemon=True)

    # -- Public: safe from any thread ------------------------------------------

    def start(self) -> None:
        """Start the loop thread and publish the empty list."""
        self._loop.call_soon_threadsafe(self._call, self._on_todos, [], [])
        self._thread.start()

    def hear(self, text: str) -> None:
        """Take an utterance: a bare stop stops, and anything else is a task."""
        if is_stop_utterance(text):
            self.stop(text)
        else:
            self.submit(text)

    def submit(
        self, text: str, todos: Iterable[Any] | None = None, source: str = "seed"
    ) -> None:
        """Queue a task. A bad seed raises here in the caller, before anything is queued.

        `todos=None` is no seed, and `todos=[]` is an empty one. The queued seed is a
        copy, so the caller may change its own list afterwards.
        """
        seed = None if todos is None else parse_seed(todos, source)
        with self._lock:
            self._require_open()
            self._enqueue(_Task(text, seed))

    def stop(self, text: str = "stop") -> None:
        """Stop now: cancel, drop the queued tasks, speak the stop line once, update the list.

        The line does not wait for the active run, which may be a hung LLM call. The
        list changes after that run, in a graph pass that makes no model call.
        """
        self._stop_base()
        with self._lock:
            self._require_open()
            self.ctx.cancel.cancel()
            self._queue = deque(item for item in self._queue if isinstance(item, _Reset))
            self._enqueue(_Stop(text))
            # The session is read now: a queued reset can replace it before the line ends.
            self._loop.call_soon_threadsafe(self._speak_stop_line, text, self.session.thread_id)

    def reset(self) -> None:
        """Forget everything: cancel, drop the queue, then a new session after the active run.

        The old thread is deleted only once the active run and its checkpoint writes
        have finished, which a hung LLM call can delay by up to the deadline.
        """
        self._stop_base()
        with self._lock:
            self._require_open()
            self.ctx.cancel.cancel()
            self._queue.clear()
            self._enqueue(_Reset())

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until the queue is empty and nothing runs. False if the time ran out."""
        return self._idle.wait(timeout)

    def shutdown(self, timeout: float | None = None) -> bool:
        """Cancel, drop the queue, and wait for the active run and the stop line to end.

        Returns False if the thread is still running after `timeout`. It is a daemon
        thread, so it cannot keep the process alive. Using the Runner later raises, and
        calling this again waits again.
        """
        with self._lock:
            if not self._closed:
                self._closed = True
                self.ctx.cancel.cancel()
                self._queue.clear()
                if self._thread.ident:
                    self._enqueue(_Shutdown())
                else:
                    self._loop.close()
        if self._thread.ident:
            self._thread.join(timeout)
        return not self._thread.is_alive()

    # -- The loop thread -------------------------------------------------------

    def _stop_base(self) -> None:
        """Abandon the base's goal. Call it before taking the lock.

        Only a verb that is still waiting reads the token. A goal left behind by a
        refused or failed cancel therefore needs base.stop() to ask again. base.stop()
        also sets the token. Run after the lock, it could set the token of a task that
        started in between. Run first, it can only meet a task that this stop cancels
        anyway, with the token it sets under the lock.
        """
        try:
            self.ctx.base.stop()
        except Exception:  # The protocol says it never raises, and a stop must not fail.
            logger.exception("base.stop failed.")

    def _require_open(self) -> None:
        """Call with the lock held, before any change: the queue holds the shutdown's item."""
        if self._closed:
            raise RuntimeError("The Runner has shut down.")

    def _enqueue(self, item: Any) -> None:
        """Add to the queue and wake the worker. Call with the lock held."""
        self._queue.append(item)
        self._idle.clear()
        self._loop.call_soon_threadsafe(self._wake.set)

    def _next(self) -> Any:
        with self._lock:
            if not self._queue:
                return None
            item = self._queue.popleft()
            if isinstance(item, _Task):
                self.ctx.cancel.reset()  # the next non-stop input releases the latch
            return item

    def _main(self) -> None:
        try:
            self._loop.run_until_complete(self._work())
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
        finally:
            self._loop.close()
            self._idle.set()

    async def _work(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while (item := self._next()) is not None:
                if isinstance(item, _Shutdown):
                    await self._speech_done()
                    return
                try:
                    await self._handle(item)
                except Exception:
                    # A failed task, a timeout included, must not end the worker. The next
                    # task runs on the same checkpoint, and RepairHistory tidies it.
                    logger.exception("The task %r failed.", getattr(item, "text", item))
                    self._call(self._on_reply, FAILURE_LINE)
            await self._speech_done()
            with self._lock:
                if not self._queue:
                    self._idle.set()

    async def _speech_done(self) -> None:
        await asyncio.sleep(0)  # a stop line scheduled just before this starts first
        await asyncio.gather(*self._speaking)

    async def _handle(self, item: Any) -> None:
        config = self.session.config(self._callbacks)
        if isinstance(item, _Reset):
            await self.agent.checkpointer.adelete_thread(self.session.thread_id)
            self.session = Session()
            self._call(self._on_todos, [], [])
        elif isinstance(item, _Stop):
            # The history shows the stop and its answer, and the list shows what was
            # stopped. before_agent ends this pass, so it cannot call the model or drive.
            messages = [HumanMessage(item.text), AIMessage(STOP_LINE)]
            await self.agent.ainvoke({"messages": messages, "stopping": True}, config)
        else:
            payload = {"messages": [HumanMessage(item.text)], "todo_seed": item.seed}
            result = await self.agent.ainvoke(payload, config)
            last = result["messages"][-1]
            if isinstance(last, AIMessage) and not last.tool_calls and last.content:
                self._call(self._on_reply, last.text)

    def _speak_stop_line(self, text: str, session: str) -> None:
        task = self._loop.create_task(self._say_stop_line(text, session))
        self._speaking.add(task)
        task.add_done_callback(self._speaking.discard)

    async def _say_stop_line(self, text: str, session: str) -> None:
        """Speak in a thread, since it blocks, and log it: no tool call covers this line."""
        status = "success"
        try:
            await asyncio.to_thread(self.ctx.speech.say, STOP_LINE)
        except Exception:
            logger.exception("Could not speak the stop line.")
            status = "error"
        logger.info("Heard %r, so the robot stopped and said %r (%s).", text, STOP_LINE, status)
        if self._tool_log is not None:
            self._tool_log.write(
                session=session,
                tool="stop_line",
                args={"heard": text},
                status=status,
                result=STOP_LINE,
            )

    def _call(self, listener: Callable[..., None] | None, *args: Any) -> None:
        if listener is None:
            return
        try:
            listener(*args)
        except Exception:
            logger.exception("A Runner listener failed.")
