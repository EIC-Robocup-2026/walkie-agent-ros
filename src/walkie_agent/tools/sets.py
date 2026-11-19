"""Which atomics each run binds: the fixed tool sets, and the atomics a flag adds or drops.

The sets are plain lists of the atomics in walkie_agent.tools.atomic, in the order
the LLM sees them. main.py picks one with --tool-set, and bound_tools drops the arm
and object memory atomics unless --arms and --object-memory connect what they use,
and pick_object unless both do. --web-search adds search_web at the end of any set.
"""

from collections.abc import Callable
from typing import Any

from walkie_agent.tools.atomic import (
    face_person,
    find_objects,
    find_people,
    get_arm_state,
    get_facts,
    go_to,
    go_to_object,
    hand_over,
    list_arm_poses,
    list_places,
    list_room_objects,
    move_arm_to_pose,
    move_relative,
    object_info,
    pick,
    pick_object,
    place_on,
    recall_object,
    recognize_person,
    remember_person,
    say,
    search_web,
    set_gripper,
)

# The atomics that move or read the arms through the commander. A set may hold them
# whether or not the robot has arms connected: bound_tools drops them by these names
# unless the run has --arms.
_ARM_ATOMICS: list[Callable[..., Any]] = [
    list_arm_poses,
    move_arm_to_pose,
    set_gripper,
    get_arm_state,
]
ARM_TOOLS: frozenset[str] = frozenset(fn.__name__ for fn in _ARM_ATOMICS)

# The atomics that read the object memory, and the one that drives to what it
# remembers. bound_tools drops them by these names unless the run has --object-memory,
# so a set without the flag keeps the tool text it had before them (decision SG-16).
_OBJECT_MEMORY_ATOMICS: list[Callable[..., Any]] = [
    recall_object,
    list_room_objects,
    go_to_object,
]
OBJECT_MEMORY_TOOLS: frozenset[str] = frozenset(fn.__name__ for fn in _OBJECT_MEMORY_ATOMICS)

# The atomic that picks up a remembered object: it finds the object in the object
# memory and reaches with the arms, so bound_tools drops it unless the run has both
# --arms and --object-memory (decision PK-4).
_PICK_ATOMICS: list[Callable[..., Any]] = [pick_object]
PICK_TOOLS: frozenset[str] = frozenset(fn.__name__ for fn in _PICK_ATOMICS)

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
# (decision S4-5); object_info stays out, since nothing measures a live object. The
# object memory atomics come before the arm atomics, and both are dropped without
# their flags. pick_object comes last, and needs both flags.
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
    *_OBJECT_MEMORY_ATOMICS,
    *_ARM_ATOMICS,
    *_PICK_ATOMICS,
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
# them, and writes no people memory, which no effect covers yet (ADR-0070). The object
# memory atomics only read the memory, and are dropped without --object-memory.
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
    *_OBJECT_MEMORY_ATOMICS,
]

TOOL_SETS = {"default": DEFAULT, "gpsr": GPSR, "hri": HRI, "arms": ARMS}

# The atomic that --web-search adds to every set, last, so a set's tool text without
# the flag stays the one measured before it (decision PK-6).
WEB_SEARCH_TOOL = search_web.__name__


def bound_tools(
    tool_set: str,
    *,
    arms: bool = False,
    object_memory: bool = False,
    web_search: bool = False,
) -> list[Callable[..., Any]]:
    """The atomics a run binds: the set's, in its order, less the atomics of a missing flag.

    Without --arms ctx.arms has no commander, and without --object-memory there is no
    object memory, so their atomics would only report UNAVAILABLE: the arm atomics are
    dropped unless `arms`, the object memory atomics unless `object_memory`, and
    pick_object unless both. `web_search` adds search_web after them, whatever the
    set. Raises ValueError for an unknown set.
    """
    if tool_set not in TOOL_SETS:
        known = ", ".join(TOOL_SETS)
        raise ValueError(f"There is no tool set {tool_set!r}. Known tool sets: {known}.")
    dropped = (set() if arms else ARM_TOOLS) | (set() if object_memory else OBJECT_MEMORY_TOOLS)
    if not (arms and object_memory):
        dropped |= PICK_TOOLS
    kept = [fn for fn in TOOL_SETS[tool_set] if fn.__name__ not in dropped]
    return [*kept, search_web] if web_search else kept


def bound_tool_names(
    tool_set: str,
    *,
    arms: bool = False,
    object_memory: bool = False,
    web_search: bool = False,
) -> set[str]:
    """The names of the atomics a run binds (bound_tools)."""
    return {
        fn.__name__
        for fn in bound_tools(
            tool_set, arms=arms, object_memory=object_memory, web_search=web_search
        )
    }
