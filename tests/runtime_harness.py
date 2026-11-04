"""What the runtime tests share: a scripted async model and the real stack over FakeRobot.

The model behaves like a provider: it raises on a history with an unanswered tool
call. No network, no ROS.
"""

import asyncio
import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.runtime.agent import build_agent
from walkie_agent.runtime.todos import parse_seed


class ScriptedModel(BaseChatModel):
    """Plays `script` in order, then `then`. An entry is a message, a callable or an exception.

    A callable gets the messages sent and returns the reply. Each copy from
    bind_tools shares the call log, as a real model's would. `_generate` raises, to
    prove nothing takes the sync path.
    """

    script: list[Any] = Field(default_factory=list)
    then: Any = None
    delay_s: float = 0.0
    strict: bool = True
    received: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[list[str]] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedModel":
        return self.model_copy(update={"tools": [t.name for t in tools]})

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("The runtime must only use the async path.")

    async def _agenerate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.received.append(list(messages))
        self.bound.append(list(self.tools))
        index = len(self.received) - 1
        if self.strict:
            assert_every_call_answered(messages)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        entry = self.script[index] if index < len(self.script) else self.then
        if isinstance(entry, Exception):
            raise entry
        reply = entry(messages) if callable(entry) else entry
        if reply is None:
            raise AssertionError(f"Model call {index + 1} was not in the script.")
        # add_messages replaces a message whose id it already holds, so a repeated
        # entry needs a fresh one, as a real model's reply has.
        reply = reply.model_copy(update={"id": None})
        return ChatResult(generations=[ChatGeneration(message=reply)])


def assert_every_call_answered(messages: Sequence[BaseMessage]) -> None:
    """Raises as an OpenAI-compatible endpoint does: answers must directly follow the calls."""
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage):
            continue
        block = set()
        for later in messages[index + 1 :]:
            if not isinstance(later, ToolMessage):
                break
            block.add(later.tool_call_id)
        calls = [*message.tool_calls, *message.invalid_tool_calls]
        missing = [c["id"] for c in calls if c["id"] not in block]
        if missing:
            raise ValueError(f"400: tool calls without an answer right after them: {missing}")


def call(
    name: str, args: dict[str, Any] | None = None, call_id: str | None = None
) -> dict[str, Any]:
    return {"name": name, "args": args or {}, "id": call_id or f"call_{name}", "type": "tool_call"}


def bad_call(call_id: str, name: str = "go_to") -> dict[str, Any]:
    """A call whose arguments did not parse, as a provider's parser reports it."""
    return {
        "name": name,
        "args": '{"place": ',
        "id": call_id,
        "error": "Bad JSON",
        "type": "invalid_tool_call",
    }


def ai(*calls: dict[str, Any], text: str = "", invalid: Sequence[dict[str, Any]] = ()) -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls), invalid_tool_calls=list(invalid))


def todo_call(*ops: dict[str, Any], call_id: str = "u1") -> dict[str, Any]:
    return call("update_todos", {"ops": list(ops)}, call_id)


def add(content: str, **fields: Any) -> dict[str, Any]:
    return {"op": "add", "content": content, **fields}


def update(item_id: str, **fields: Any) -> dict[str, Any]:
    return {"op": "update", "id": item_id, **fields}


def remove(item_id: str) -> dict[str, Any]:
    return {"op": "remove", "id": item_id}


def drive(call_id: str, place: str = "sofa") -> dict[str, Any]:
    return call("go_to", {"place": place}, call_id)


@dataclass
class Stack:
    """The real agent over a FakeRobot and a scripted model, driven through ainvoke."""

    model: ScriptedModel
    robot: FakeRobot
    agent: Any
    published: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = field(default_factory=list)
    thread_id: str = "test-thread"

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}, "max_concurrency": 1}

    def run(
        self, text: str = "Do it.", todos: Any = None, source: str = "seed", **inputs: Any
    ) -> dict[str, Any]:
        """One ainvoke on a new loop, as one task. Returns the agent's result."""
        seed = None if todos is None else parse_seed(todos, source)
        payload = {"messages": [HumanMessage(text)], "todo_seed": seed, **inputs}
        return asyncio.run(self.agent.ainvoke(payload, self.config))

    @property
    def state(self) -> dict[str, Any]:
        return self.agent.get_state(self.config).values

    @property
    def messages(self) -> list[BaseMessage]:
        return self.state["messages"]

    def items(self) -> list[dict[str, Any]]:
        return self.state.get("todos") or []

    def statuses(self) -> list[str]:
        return [item["status"] for item in self.items()]

    def answer(self, call_id: str) -> ToolMessage:
        """The one answer to `call_id` in the history."""
        answers = [
            m for m in self.messages if isinstance(m, ToolMessage) and m.tool_call_id == call_id
        ]
        assert len(answers) == 1, f"{call_id} has {len(answers)} answers"
        return answers[0]

    def verbs(self) -> list[str]:
        return [entry["verb"] for entry in self.robot.calls]

    def reminder(self, model_call: int = -1) -> str:
        """The reminder sent on one model call, or "" if none."""
        last = self.model.received[model_call][-1]
        text = str(last.content)
        return text if text.startswith("<todos>") else ""


def make_stack(
    *script: Any,
    then: Any = None,
    delay_s: float = 0.0,
    robot: FakeRobot | None = None,
    **build: Any,
) -> Stack:
    robot = FakeRobot() if robot is None else robot
    model = ScriptedModel(script=list(script), then=then, delay_s=delay_s)
    stack = Stack(model=model, robot=robot, agent=None)
    build.setdefault("on_todos", lambda items, changed: stack.published.append((items, changed)))
    stack.agent = build_agent(model, robot.ctx, **build)
    return stack


def wait_until(done: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not done():
        assert time.monotonic() < deadline, "Timed out waiting for a condition."
        time.sleep(0.005)


def contents(messages: Sequence[BaseMessage]) -> list[Any]:
    return [copy.deepcopy(m.content) for m in messages]
