"""Atomics: plain typed functions over ctx, and the tool set each test uses.

An atomic takes ctx first, returns a JSON-able value, and raises StepError on a
failure. Its Google-style docstring is what the LLM reads, so the return value
is described in the summary: LangChain drops a Returns section from a tool's
description. Tool messages and Monty scripts call the same functions, through
walkie_agent.adapter.
"""

from collections.abc import Callable
from typing import Any

from walkie_agent.tools.atomic.arms import get_arm_state, list_arm_poses, move_arm_to_pose, set_gripper
from walkie_agent.tools.atomic.facts import get_facts
from walkie_agent.tools.atomic.manipulation import hand_over, pick, place_on
from walkie_agent.tools.atomic.memory import recognize_person, remember_person
from walkie_agent.tools.atomic.navigation import face_person, go_to, list_places, move_relative
from walkie_agent.tools.atomic.perception import find_objects, find_people, object_info
from walkie_agent.tools.atomic.speech import say

__all__ = [
    "ARMS",
    "ARM_TOOLS",
    "DEFAULT",
    "GPSR",
    "HRI",
    "face_person",
    "find_objects",
    "find_people",
    "get_arm_state",
    "get_facts",
    "go_to",
    "hand_over",
    "list_arm_poses",
    "list_places",
    "move_arm_to_pose",
    "move_relative",
    "object_info",
    "pick",
    "place_on",
    "recognize_person",
    "remember_person",
    "say",
    "set_gripper",
]

# The atomics that move or read the arms through the commander. A set may hold them
# whether or not the robot has arms connected: main.py drops them by these names
# unless it runs with --arms.
_ARM_ATOMICS: list[Callable[..., Any]] = [
    list_arm_poses,
    move_arm_to_pose,
    set_gripper,
    get_arm_state,
]
ARM_TOOLS: frozenset[str] = frozenset(fn.__name__ for fn in _ARM_ATOMICS)

# The tool sets are plain lists in code (decision 3). Commands name their
# places, so GPSR has no list_places, as in the spike it was measured with.
# get_facts came after the spike, for the talk commands (decision M35).
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
    get_facts,
]
# face_person is not in GPSR, whose tool text the Mahoraga runs measured (decision S2-6).
# The two memory atomics are in HRI alone, so DEFAULT and GPSR stay as they were:
# no effect covers a memory write yet (ADR-0070). The arm atomics come last, so HRI
# without them is the set it was before them. find_objects lets HRI see the seats
# (decision S4-5); object_info stays out, since nothing measures a live object.
HRI: list[Callable[..., Any]] = [
    list_places,
    go_to,
    move_relative,
    say,
    find_people,
    find_objects,
    face_person,
    remember_person,
    recognize_person,
    *_ARM_ATOMICS,
]
# The arms on their own: the base verbs the drive check guards, speech to say what the
# arms will do, and the arm atomics. GPSR and DEFAULT have no arm atomics, so their
# tool text stays the one the Mahoraga runs measured.
ARMS: list[Callable[..., Any]] = [
    list_places,
    go_to,
    move_relative,
    say,
    *_ARM_ATOMICS,
]
# Every atomic that needs no arms, which the robot runs without unless --arms connects
# them, and writes no people memory, which no effect covers yet (ADR-0070).
DEFAULT: list[Callable[..., Any]] = [
    list_places,
    go_to,
    move_relative,
    say,
    find_objects,
    object_info,
    find_people,
    face_person,
    get_facts,
]
