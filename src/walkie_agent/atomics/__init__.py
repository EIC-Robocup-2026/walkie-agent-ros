"""Atomics: plain typed functions over ctx, and the tool set each test uses.

An atomic takes ctx first, returns a JSON-able value, and raises StepError on a
failure. Its Google-style docstring is what the LLM reads, so the return value
is described in the summary: LangChain drops a Returns section from a tool's
description. Tool messages and Monty scripts call the same functions, through
walkie_agent.adapter.
"""

from collections.abc import Callable
from typing import Any

from walkie_agent.atomics.manipulation import hand_over, pick, place_on
from walkie_agent.atomics.navigation import go_to, list_places, move_relative
from walkie_agent.atomics.perception import find_objects, find_people, object_info
from walkie_agent.atomics.speech import say

__all__ = [
    "DEFAULT",
    "GPSR",
    "HRI",
    "find_objects",
    "find_people",
    "go_to",
    "hand_over",
    "list_places",
    "move_relative",
    "object_info",
    "pick",
    "place_on",
    "say",
]

# The tool sets are plain lists in code (decision 3). Commands name their
# places, so GPSR has no list_places, as in the spike it was measured with.
GPSR: list[Callable[..., Any]] = [
    go_to,
    move_relative,
    say,
    find_objects,
    object_info,
    find_people,
    pick,
    place_on,
    hand_over,
]
HRI: list[Callable[..., Any]] = [list_places, go_to, move_relative, say, find_people]
# Every atomic that needs no arms, which the robot does not have yet.
DEFAULT: list[Callable[..., Any]] = [
    list_places,
    go_to,
    move_relative,
    say,
    find_objects,
    object_info,
    find_people,
]
