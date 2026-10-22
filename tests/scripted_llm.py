"""A chat model that replays a fixed list of responses, for testing agents without an LLM."""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr


def tool_call(name: str, args: dict[str, Any] | None = None, call_id: str = "") -> AIMessage:
    """An AI message that asks for a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id or f"call_{name}"}],
    )


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    # The full message list the model was given on each call, for assertions.
    # It includes the todo reminder when one was sent.
    received: list[list[BaseMessage]] = Field(default_factory=list)
    # The tool names bound for each call: those given to bind_tools since the
    # previous call, or [] when nothing was bound.
    bound: list[list[str]] = Field(default_factory=list)
    # The todo reminder sent last on each call, or "" when there was none.
    reminders: list[str] = Field(default_factory=list)
    # Code to run during a call, by call index from 0, before the response is
    # returned. A hook that raises makes that call fail without using up a
    # response.
    hooks: dict[int, Callable[[], Any]] = Field(default_factory=dict)
    _next: int = PrivateAttr(default=0)
    _bound_names: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        self._bound_names = [tool.name for tool in tools]
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        call = len(self.received)
        self.received.append(list(messages))
        self.bound.append(self._bound_names)
        self._bound_names = []
        self.reminders.append(_reminder(messages))
        hook = self.hooks.get(call)
        if hook is not None:
            hook()
        if self._next >= len(self.responses):
            raise AssertionError(f"Scripted model ran out of responses after {self._next} calls")
        message = self.responses[self._next]
        self._next += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def _reminder(messages: list[BaseMessage]) -> str:
    last = messages[-1] if messages else None
    if (
        isinstance(last, HumanMessage)
        and isinstance(last.content, str)
        and last.content.startswith("<todos>")
    ):
        return last.content
    return ""
