"""Builds the Walkie agent: RAI's LangChainAgent running Walkie's own graph."""

import copy
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from rai.agents.langchain import LangChainAgent
from rai.communication.hri_connector import HRIConnector, HRIMessage
from rai.communication.ros2 import ROS2Connector
from rai.messages import SystemMultimodalMessage
from std_msgs.msg import Empty

from walkie_agent.graph import ToolSet, build_walkie_graph
from walkie_agent.locations import Location
from walkie_agent.todo import TodoList
from walkie_agent.tools import build_walkie_tools
from walkie_agent.whoami import build_system_prompt

SYSTEM_PROMPT = """\
You are Walkie, a home service robot competing in RoboCup@Home.
You have a mobile base and can drive between named places.

Rules:
- To talk to a person in the room, use the `speak` tool. Your plain-text replies
  go to the operator console only, never to the speaker.
- Use `list_locations` to see where you can go, then `navigate_to_location`.
- For short moves from where you are, such as "move forward one meter" or
  "turn around", use `move_relative`.
- To find where you last saw an object, such as "where is my cup?", use
  `find_objects`. It only knows objects you have already seen.
- If a tool reports a failure, tell the person what went wrong with `speak`
  and do not pretend the action succeeded.
- For a task with more than one step, write the steps with `update_todos`
  before you start. Mark a step in_progress when you start it, and completed
  only after its tool succeeded.
- If a step cannot be done, mark it failed with a note and tell the person
  with `speak`.
- The <todos> block at the end of the messages is your current list. The
  system writes it, not the person.
- When you need an answer from the person to go on, for example which object
  they mean, ask with `wait_for_person` instead of `speak`. Your turn ends,
  and the answer arrives as the next message.
- If the person says stop or never mind, stop. Mark the open steps cancelled
  with a note, and say so briefly with `speak`.
- Keep spoken sentences short and friendly. They are read out by a
  text-to-speech voice, so never use markdown, lists or code in them.
"""

# LangGraph stops a run after this many steps and raises. Its default, 25, is
# about 12 tool rounds, too few for a seeded task with several steps.
RECURSION_LIMIT = 100

_STOP_UTTERANCES = frozenset({"stop", "never mind", "nevermind"})


def is_stop_utterance(text: str) -> bool:
    """True when the utterance is just "stop" or "never mind".

    Only these interrupt a running task. The whole utterance must match, so
    "stop the music" or "Walkie, stop" waits in the queue like any other
    message: at a noisy venue, overheard speech must not cancel a step
    (ADR-0038).
    """
    normalized = " ".join(text.lower().strip(" \t\r\n.,!?").split())
    return normalized in _STOP_UTTERANCES


@dataclass(frozen=True)
class _Seed:
    """The todo list to start a submitted task with."""

    items: list[str | dict[str, Any]]
    source: Literal["seed", "operator"]


class WalkieAgent(LangChainAgent):
    """RAI's LangChainAgent running Walkie's own graph (ADR-0038).

    The graph (walkie_agent.graph) binds the agent's tools on every LLM call,
    sends the todo list as a reminder, repairs the history and runs the
    completion gate. This class adds what happens between runs:

    - Messages queue. An utterance that arrives during a task waits until
      the task ends. Only reset(), or an utterance that is just "stop" or
      "never mind", interrupts a task. RAI checks the interrupt between graph
      steps, so a running tool call finishes first.
    - submit() sends a task with a seeded todo list (ADR-0041).
    - reset() forgets the conversation and the todo list without restarting
      the agent.
    - An error from the LLM or the graph is logged, and the agent keeps
      serving the next message.

    Each conversation is also a Langfuse session (`session_id`), so traces
    from the same conversation are grouped together in the Langfuse UI.

    Relies on LangChainAgent internals of rai-core 2.12: `_run_agent` and
    `__call__`, both overridden in full from RAI's code, `stop()`, which sets
    its events under the buffer lock before RAI's runs, the message buffer
    and its lock, the stop, interrupt and ready events, and the ToolRunner
    internals that WalkieToolRunner uses. Recheck all of them on every
    rai-core upgrade (ADR-0038).
    """

    def __init__(
        self,
        target_connectors: dict[str, HRIConnector[HRIMessage]],
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        system_prompt: str | SystemMultimodalMessage | None = None,
        stream_response: bool = True,
        todos: TodoList | None = None,
    ):
        self.todos = todos if todos is not None else TodoList()
        self.tools = ToolSet(tools or [])
        graph = build_walkie_graph(llm, self.tools, self.todos, system_prompt)
        super().__init__(
            target_connectors=target_connectors,
            runnable=graph,
            stream_response=stream_response,
            new_message_behavior="queue",
        )
        self._graph = self.agent
        self._reset_requested = threading.Event()
        # The seed of each queued message that has one, by id(message). The
        # entry also holds the message, so its id cannot be reused by another
        # object while the seed waits. Guarded by _buffer_lock.
        self._seeds: dict[int, tuple[HRIMessage, _Seed]] = {}
        self.session_id = ""
        self._start_session()

    def __call__(self, msg: HRIMessage) -> None:
        """Queues a message. RAI calls this for every message on a subscribed source."""
        self._enqueue(msg, None)

    def submit(
        self,
        text: str,
        todos: list[str | dict[str, Any]] | None = None,
        source: Literal["seed", "operator"] = "seed",
    ) -> None:
        """Sends a task, optionally with the todo list to start it with.

        When the task starts, `todos` replaces the current list; [] starts it
        with an empty list, and None keeps the list as any other message does.
        The seed is checked here, so a bad seed raises TodoRejected in the
        caller rather than on the agent thread.
        """
        seed = None
        if todos is not None:
            # A scratch list runs the same checks that the real seed will.
            TodoList().seed(todos, source)
            seed = _Seed(items=copy.deepcopy(list(todos)), source=source)
        self._enqueue(HRIMessage(text=text, message_author="human"), seed)

    def reset(self) -> None:
        """Drop queued tasks, stop the current task, and clear the conversation.

        Returns immediately; the agent thread applies the reset. A running task
        stops at its next step, so a tool call already in progress (such as a
        navigation goal) finishes first. Safe to call from any thread.
        """
        with self._buffer_lock:
            self._received_messages.clear()
            self._seeds.clear()
            self._reset_requested.set()
            self._interrupt_event.set()

    def stop(self) -> None:
        """Stop the agent thread. A running task ends after its current step."""
        # RAI's stop() sets both events without the buffer lock. A run that
        # starts clears the interrupt under that lock, so an interrupt set in
        # between would be lost and the whole task would run before the
        # thread stops. Under the lock, a starting run either sees the stop
        # and does not start, or starts first and is interrupted.
        with self._buffer_lock:
            self._stop_event.set()
            self._interrupt_event.set()
        super().stop()

    def subscribe_reset(self, connector: ROS2Connector, topic: str) -> None:
        """Call reset() whenever a std_msgs/Empty message arrives on `topic`."""
        connector.node.create_subscription(Empty, topic, lambda _msg: self.reset(), 10)

    def _enqueue(self, msg: HRIMessage, seed: _Seed | None) -> None:
        # Replaces RAI's __call__. The seed is stored under the same lock as
        # the message, so a reset can never separate the two.
        with self._buffer_lock:
            if self.max_size is not None and len(self._received_messages) >= self.max_size:
                self.logger.warning("Buffer overflow. Dropping the oldest message.")
                dropped = self._received_messages.popleft()
                self._seeds.pop(id(dropped), None)
            self.logger.info(f"Received message: {msg}, {type(msg)}")
            if seed is not None:
                self._seeds[id(msg)] = (msg, seed)
            self._received_messages.append(msg)

            # _run_agent takes a message and clears the ready event under this
            # lock, so this check cannot race with the start of a run: a stop
            # either interrupts the run or waits as an ordinary message.
            running = self._thread is not None and not self._agent_ready_event.is_set()
            if running and is_stop_utterance(msg.text):
                for queued in list(self._received_messages)[:-1]:
                    self._seeds.pop(id(queued), None)
                self._received_messages.clear()
                # The stop itself stays queued, so the LLM sees it next and
                # can cancel the open steps.
                self._received_messages.append(msg)
                self._interrupt_event.set()
                self.logger.info("Stop heard: interrupting the task, dropping older messages.")

    def _run_agent(self) -> None:
        # Replaces RAI's _run_agent. Runs on the agent thread between tasks,
        # so the message list is never cleared while the graph is using it.
        if self._reset_requested.is_set():
            self._reset_requested.clear()
            self._interrupt_event.clear()
            self.state["messages"].clear()
            self.todos.clear()
            self._start_session()
            self.logger.info(f"Conversation reset. Langfuse session: {self.session_id}")

        msg: HRIMessage | None = None
        seed: _Seed | None = None
        with self._buffer_lock:
            # A reset that came after the check above is applied on the next
            # pass, so a message sent after it never runs on the old history.
            # Once stop() is called, no new run starts, because a new run
            # clears the interrupt that stop() set. stop() sets both events
            # under this lock, so it cannot land between this check and the
            # clear below.
            if (
                self._received_messages
                and not self._reset_requested.is_set()
                and not self._stop_event.is_set()
            ):
                # "queue": one message per task, oldest first.
                msg = self._received_messages.popleft()
                entry = self._seeds.pop(id(msg), None)
                seed = entry[1] if entry is not None else None
                # An interrupt meant for the previous run must not end this one.
                self._interrupt_event.clear()
                self._agent_ready_event.clear()

        if msg is None:
            self._agent_ready_event.set()
            self.logger.debug("Waiting for messages...")
            time.sleep(0.1)
            return

        try:
            self.logger.info("Running agent...")
            self._prepare_todos(seed)
            # The trailing newline keeps the format of RAI's _reduce_messages.
            message = HRIMessage(
                text=f"{msg.text}\n",
                images=msg.images,
                audios=msg.audios,
                message_author="human",
            )
            self.state["messages"].append(message.to_langchain())
            for _ in self.agent.stream(
                self.state,
                config={"callbacks": [self._langchain_callback, *self.tracing_callbacks]},
            ):
                if self._interrupt_event.is_set():
                    break
        except Exception:
            # RAI lets such an error end the agent thread. Here only the task
            # ends; the history repair fixes a half-finished turn on the next
            # run.
            self.logger.exception("The task failed with an error; waiting for the next message.")
        finally:
            self._agent_ready_event.set()

    def _prepare_todos(self, seed: _Seed | None) -> None:
        if seed is not None:
            self.todos.seed(seed.items, seed.source)
            return
        # A message without a seed may be the answer to a wait_for_person
        # question.
        self.todos.resume_waiting()
        # A finished list does not carry into the next task.
        if not self.todos.is_empty and not self.todos.open_ids():
            self.todos.clear()

    def _start_session(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_id = f"walkie-{stamp}-{secrets.token_hex(2)}"
        # Langfuse reads the session ID from the metadata of each run.
        self.agent = self._graph.with_config(
            metadata={"langfuse_session_id": self.session_id},
            recursion_limit=RECURSION_LIMIT,
        )


def create_walkie_agent(
    llm: BaseChatModel,
    connector: ROS2Connector,
    locations: dict[str, Location],
    target_connectors: dict[str, HRIConnector[HRIMessage]],
    stream_response: bool = True,
    whoami_dir: Path | None = None,
) -> WalkieAgent:
    """Build the agent. `whoami_dir` adds the robot description and documentation
    search from `pixi run whoami-build`, when they have been built (see whoami.py)."""
    # One list, shared by the todo tools and the graph.
    todos = TodoList()
    return WalkieAgent(
        target_connectors=target_connectors,
        llm=llm,
        tools=build_walkie_tools(connector, locations, whoami_dir, todos=todos),
        system_prompt=build_system_prompt(SYSTEM_PROMPT, whoami_dir),
        stream_response=stream_response,
        todos=todos,
    )
