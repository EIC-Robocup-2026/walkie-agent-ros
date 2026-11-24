"""Compaction: between tasks, the older part of a long history becomes a short summary.

The Runner calls it once its queue is empty, so a person never waits for it (PK-74).
The messages before a person's turn become one HumanMessage that starts with
SUMMARY_PREFIX, and the newest turns stay as they were. The summary is a human
message because not every provider accepts a second system message in the middle of
a history. The system prompt tells the model how to read it.
"""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig

SUMMARY_PREFIX = "[summary]"
# A tool's answer is cut to this many characters in the transcript the summarizer reads.
ANSWER_CHARS = 300

SUMMARY_PROMPT = """\
You keep the memory of Walkie, a robot that helps people in a home. Below is the \
older part of its conversation, as a transcript. Write the summary that Walkie will \
read in its place from now on.

Keep the names of people and what each one asked for, what Walkie did and how each \
request ended, failures included, anything Walkie promised and has not done yet, \
where objects were left or who holds them, and what people told Walkie about \
themselves or the home.

Leave out small talk, coordinates and the ids that tools return, since Walkie can \
ask its tools for them again. If the transcript starts with an earlier summary, \
merge it in and keep what still matters.

Write plain sentences in the past tense, at most 150 words, with no heading or list."""


def is_summary(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) and message.text.startswith(SUMMARY_PREFIX)


def starts_turn(message: BaseMessage) -> bool:
    """A person's message. The runtime's own notes, such as "[todo] ...", start with "["."""
    return isinstance(message, HumanMessage) and not message.text.startswith("[")


def transcript(messages: Sequence[BaseMessage]) -> str:
    """The plain transcript the summarizer reads: who said what, and what each tool did.

    The runtime's notes are left out, and so are the successful answers of say and
    wait_for_person, which only repeat the line. A line whose call failed, was not run
    after an earlier failure, or has no answer at all reads as tried, so the summary
    never claims it.
    """
    ran = {m.tool_call_id for m in messages if isinstance(m, ToolMessage) and m.status != "error"}
    lines = []
    for message in messages:
        if is_summary(message):
            lines.append(f"Earlier summary: {message.text.removeprefix(SUMMARY_PREFIX).strip()}")
        elif starts_turn(message):
            lines.append(f"Person: {message.text.strip()}")
        elif isinstance(message, AIMessage):
            if message.text.strip():
                lines.append(f"Walkie said: {message.text.strip()}")
            lines.extend(_call_line(call, call.get("id") in ran) for call in message.tool_calls)
        elif isinstance(message, ToolMessage):
            if message.status == "error":
                lines.append(f"  {message.name} failed: {_clip(message.text)}")
            elif message.name not in _SPOKEN:
                lines.append(f"  {message.name} answered: {_clip(message.text)}")
    return "\n".join(lines)


# A speaking tool's argument, and how its line reads when it ran and when it failed.
_SPOKEN = {
    "say": ("text", "said", "tried to say"),
    "wait_for_person": ("question", "asked", "tried to ask"),
}


def _call_line(call: dict[str, Any], ran: bool) -> str:
    args = call["args"]
    if call["name"] in _SPOKEN:
        key, done, tried = _SPOKEN[call["name"]]
        return f"Walkie {done if ran else tried}: {args.get(key, '')}"
    return f"Walkie called {call['name']} with {json.dumps(args, ensure_ascii=False, default=str)}"


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= ANSWER_CHARS else text[:ANSWER_CHARS] + " [cut]"


@dataclass(frozen=True)
class Compactor:
    """Decides where to split a history, and summarizes the older part.

    `model` is the summarizer, called without tools. A history of `after_tokens` or
    more is split, and the newest turns, at least `keep_tokens` of them, stay. Tokens
    are langchain_core's approximate count, about four characters each.
    """

    model: BaseChatModel
    after_tokens: int
    keep_tokens: int
    timeout_s: float = 30.0

    def split(self, messages: Sequence[BaseMessage]) -> int:
        """Where the kept turns start, or 0 to keep the whole history.

        The kept part starts at a person's message, so no tool answer is parted from
        its call. A split that would summarize only an earlier summary is no split.
        """
        if count_tokens_approximately(messages) < self.after_tokens:
            return 0
        start, kept = len(messages), 0
        while start > 0 and kept < self.keep_tokens:
            start -= 1
            kept += count_tokens_approximately([messages[start]])
        while start > 0 and not starts_turn(messages[start]):
            start -= 1
        if start == 1 and is_summary(messages[0]):
            return 0
        return start

    async def summarize(
        self, older: Sequence[BaseMessage], config: RunnableConfig | None = None
    ) -> HumanMessage:
        """The message that replaces `older`. Raises TimeoutError after timeout_s.

        An empty answer raises ValueError, so a failed summary never replaces a history.
        """
        prompt = [SystemMessage(SUMMARY_PROMPT), HumanMessage(transcript(older))]
        async with asyncio.timeout(self.timeout_s):
            reply = await self.model.ainvoke(prompt, config)
        text = " ".join(reply.text.split())
        if not text:
            raise ValueError("The summarizer answered with no text.")
        return HumanMessage(f"{SUMMARY_PREFIX} {text}")
