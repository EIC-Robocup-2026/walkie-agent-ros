"""A chat model that replays a fixed list of responses, for testing agents without an LLM."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
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
    received: list[list[BaseMessage]] = Field(default_factory=list)
    _next: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        if self._next >= len(self.responses):
            raise AssertionError(f"Scripted model ran out of responses after {self._next} calls")
        message = self.responses[self._next]
        self._next += 1
        return ChatResult(generations=[ChatGeneration(message=message)])
