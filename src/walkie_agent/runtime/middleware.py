"""The small create_agent middlewares of the runtime, each guarding one failure.

The order of the list matters and is pinned in `build_middleware`, in
walkie_agent.runtime.agent. Tool wrappers need `awrap_tool_call` and model
wrappers `awrap_model_call`, because the Runner only uses `ainvoke`. Node hooks
may stay sync: langgraph runs them inline on the loop, so they must not block.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from walkie_agent.capabilities import CancelToken

logger = logging.getLogger(__name__)

ToolResult = ToolMessage | Command
ToolHandler = Callable[[ToolCallRequest], Awaitable[ToolResult]]

INTERRUPTED = "Interrupted before it ran."
UNREADABLE_ARGUMENTS = "The arguments were not a valid JSON object, so the call did not run."
ONE_UPDATE_TODOS = (
    "Only one update_todos call per message is accepted, so this one was not applied. "
    "Put all ops in one call."
)
AFTER_QUESTION = (
    "Not run: you asked the person a question, so the task stopped to wait for the answer. "
    "Call it again after the answer if it is still needed."
)
AFTER_STOP = "Not run: the person said stop, so the task ended."


def tool_result(result: ToolResult, call_id: str) -> ToolMessage | None:
    """The ToolMessage that answers `call_id`, from what a tool wrapper's handler returned."""
    if isinstance(result, ToolMessage):
        return result
    update = result.update if isinstance(result.update, dict) else {}
    answers = (m for m in update.get("messages", []) if isinstance(m, ToolMessage))
    return next((m for m in answers if m.tool_call_id == call_id), None)


def refusal(request: ToolCallRequest, text: str) -> ToolMessage:
    """The answer to a call that did not run. Every call needs one, or the next LLM call fails."""
    call = request.tool_call
    return ToolMessage(text, name=call["name"], tool_call_id=call["id"], status="error")


def owning_message(state: dict[str, Any]) -> AIMessage:
    """The AI message whose calls are running. Tools see the state from before their superstep."""
    return next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))


class ToolLog(AgentMiddleware):
    """Appends one JSON line per tool call, refused calls included. Goes outermost.

    A log failure is logged and never costs the tool result, because the robot
    has already acted on it. The session id comes from the run's config, since a
    reset changes it.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)

    def write(self, **record: Any) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"ts": time.time(), **record}, default=str, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except (OSError, TypeError, ValueError):
            logger.exception("Could not write to the tool log %s.", self.path)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolResult:
        call, started = request.tool_call, time.monotonic()
        result = await handler(request)
        message = tool_result(result, call["id"])
        self.write(
            session=request.runtime.config.get("configurable", {}).get("thread_id"),
            tool=call["name"],
            call_id=call["id"],
            args=call["args"],
            status=message.status if message else "unknown",
            result=message.content if message else "",
            seconds=round(time.monotonic() - started, 3),
        )
        return result


class StopLatch(AgentMiddleware):
    """Ends the run once a stop has set the token: no more model calls, and no more tools.

    It refuses every call, not only motion, so the calls of an LLM reply that
    was on its way when the stop came cannot say or record anything false. A stop can
    still land after this look and before the tool's body runs, so the adapter and the
    todo tools look at the token again there.
    """

    def __init__(self, cancel: CancelToken) -> None:
        super().__init__()
        self.cancel = cancel

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return {"jump_to": "end"} if self.cancel.cancelled else None

    async def awrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolResult:
        if self.cancel.cancelled:
            return refusal(request, AFTER_STOP)
        return await handler(request)


class SerialGuard(AgentMiddleware):
    """Enforces the rules that tie together the calls of one AI message.

    A failed call skips the later ones, because they were planned on it working.
    A question skips them too, because they must wait for the answer. And only
    the first update_todos runs: a refused copy is a failed call like any other. These
    answers go out of this wrapper, so they never reach the todo ledger.
    """

    def __init__(self) -> None:
        super().__init__()
        self._blocked: tuple[str | None, str] | None = None  # the AI message's id, and the reason

    async def awrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolResult:
        call, owner = request.tool_call, owning_message(request.state)
        if self._blocked and self._blocked[0] == owner.id:
            return refusal(request, self._blocked[1])
        updates = [c["id"] for c in owner.tool_calls if c["name"] == "update_todos"]
        if call["name"] == "update_todos" and updates[0] != call["id"]:
            result = refusal(request, ONE_UPDATE_TODOS)
        else:
            result = await handler(request)
        message = tool_result(result, call["id"])
        if message is not None and message.status == "error":
            reason = f"Not run: {call['name']} failed earlier in this message. Decide again."
            self._blocked = (owner.id, reason)
        elif call["name"] == "wait_for_person":
            self._blocked = (owner.id, AFTER_QUESTION)
        return result


class EndAfterQuestion(AgentMiddleware):
    """Ends the run after a successful wait_for_person, so its answer starts the next run.

    WalkieTodos consumes `question` when the next run starts, so a newer human
    message always reaches the model. wait_for_person has no return_direct, because
    the factory ends on that whatever the status, and a failed question must go
    back to the model.
    """

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return {"jump_to": "end"} if state.get("question") else None


class RepairHistory(AgentMiddleware):
    """Makes sure every tool call in the history has an answer, which providers require."""

    def before_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Answer the calls a failed or cancelled run left open."""
        messages = state["messages"]
        fixed = _with_every_call_answered(messages)
        if len(fixed) == len(messages):
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *fixed]}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Answer the calls whose arguments did not parse, which the tools node never sees.

        The valid calls of a mixed message still run through the tools node. Only
        a message with none goes straight back to the model. The limit middleware
        comes later in the list, so its after_model runs before this one and
        counts the call even when this hook jumps.
        """
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.invalid_tool_calls:
            return None
        answers = [_answer(call, UNREADABLE_ARGUMENTS) for call in last.invalid_tool_calls]
        update: dict[str, Any] = {"messages": [a for a in answers if a.tool_call_id]}
        return update if last.tool_calls else {**update, "jump_to": "model"}


class Deadline(AgentMiddleware):
    """Bounds one model attempt with asyncio.timeout around the awaited call.

    This is cooperative, not a hard wall-clock bound: blocking code on the loop
    delays it. It also ends the client's wait, which does not prove the provider
    stopped computing. Goes inside ModelFallbackMiddleware, so each attempt has
    its own budget.
    """

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        async with asyncio.timeout(self.seconds):
            return await handler(request)


def _answer(call: Any, text: str) -> ToolMessage:
    return ToolMessage(text, name=call.get("name"), tool_call_id=call["id"] or "", status="error")


def _with_every_call_answered(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The messages, with an answer at the end of each AI message's block of tool results."""
    fixed: list[AnyMessage] = []
    open_calls: list[Any] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            open_calls = [c for c in open_calls if c["id"] != message.tool_call_id]
        else:
            fixed += [_answer(call, INTERRUPTED) for call in open_calls]
            open_calls = []
        fixed.append(message)
        if isinstance(message, AIMessage):
            calls = [*message.tool_calls, *message.invalid_tool_calls]
            open_calls = [c for c in calls if c.get("id")]
    return fixed + [_answer(call, INTERRUPTED) for call in open_calls]
