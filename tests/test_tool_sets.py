"""The per-test tool sets and what every atomic in them must offer."""

import inspect

import pytest

from walkie_agent.tools import atomic
from walkie_agent.tools.atomic import (
    describe_objects,
    get_arm_state,
    go_to_object,
    hand_over,
    list_arm_poses,
    list_room_objects,
    move_arm_to_pose,
    pick,
    pick_object,
    place_on,
    recall_object,
    recognize_person,
    remember_person,
    search_web,
    set_gripper,
)
from walkie_agent.tools.sets import (
    ARM_TOOLS,
    ARMS,
    DEFAULT,
    GPSR,
    HRI,
    OBJECT_MEMORY_TOOLS,
    PICK_TOOLS,
    TOOL_SETS,
    WEB_SEARCH_TOOL,
    bound_tool_names,
    bound_tools,
)

# The fake's object model: pick, place and hand over one held object.
MANIPULATION = {pick, place_on, hand_over}
# Named poses and the grippers, through the commander.
ARM_ATOMICS = {list_arm_poses, move_arm_to_pose, set_gripper, get_arm_state}
MEMORY = {remember_person, recognize_person}
# Read the object memory, and drive to what it remembers (decision SG-16).
OBJECT_MEMORY = {recall_object, list_room_objects, go_to_object}
# Picks up a remembered object, with both of the above connected (decision PK-4).
PICK = {pick_object}
SETS = {"GPSR": GPSR, "HRI": HRI, "ARMS": ARMS, "DEFAULT": DEFAULT}
# search_web is in no set's list: --web-search adds it to any set (bound_tools).
ALL = sorted(set().union(*SETS.values()) | {search_web}, key=lambda fn: fn.__name__)


def names(tool_set) -> list[str]:
    return [fn.__name__ for fn in tool_set]


def test_gpsr_is_the_set_the_spike_used_and_get_facts():
    # get_facts came after the spike, for the talk commands (decision M35).
    assert names(GPSR) == [
        "go_to",
        "move_relative",
        "say",
        "find_objects",
        "object_info",
        "find_people",
        "pick",
        "place_on",
        "hand_over",
        "get_facts",
    ]


def test_hri_finds_objects_but_cannot_measure_or_move_them_and_ends_with_memories_then_arms():
    assert names(HRI) == [
        "list_places",
        "go_to",
        "move_relative",
        "say",
        "find_people",
        "find_objects",
        "describe_objects",
        "face_person",
        "remember_person",
        "recognize_person",
        "recall_object",
        "list_room_objects",
        "go_to_object",
        "list_arm_poses",
        "move_arm_to_pose",
        "set_gripper",
        "get_arm_state",
        "pick_object",
    ]


def test_arms_is_the_base_and_speech_then_the_arm_tools():
    assert names(ARMS) == [
        "list_places",
        "go_to",
        "move_relative",
        "say",
        "list_arm_poses",
        "move_arm_to_pose",
        "set_gripper",
        "get_arm_state",
    ]


def test_arm_tools_names_the_arm_atomics():
    assert ARM_TOOLS == frozenset(fn.__name__ for fn in ARM_ATOMICS)
    assert isinstance(ARM_TOOLS, frozenset)


def test_object_memory_tools_names_the_object_memory_atomics():
    assert OBJECT_MEMORY_TOOLS == frozenset(fn.__name__ for fn in OBJECT_MEMORY)
    assert isinstance(OBJECT_MEMORY_TOOLS, frozenset)


def test_hri_without_the_arm_and_object_memory_tools_is_the_set_it_was_before_them():
    # main.py drops ARM_TOOLS without --arms, OBJECT_MEMORY_TOOLS without
    # --object-memory and PICK_TOOLS without either, which must leave HRI as it was.
    dropped = ARM_TOOLS | OBJECT_MEMORY_TOOLS | PICK_TOOLS
    assert [name for name in names(HRI) if name not in dropped] == [
        "list_places",
        "go_to",
        "move_relative",
        "say",
        "find_people",
        "find_objects",
        "describe_objects",
        "face_person",
        "remember_person",
        "recognize_person",
    ]


def test_the_arm_tools_are_in_hri_and_arms_alone():
    # GPSR and DEFAULT keep the tool text the Mahoraga runs measured.
    assert ARM_ATOMICS <= set(HRI)
    assert ARM_ATOMICS <= set(ARMS)
    assert ARM_ATOMICS.isdisjoint(GPSR)
    assert ARM_ATOMICS.isdisjoint(DEFAULT)


def test_default_is_every_atomic_that_needs_no_arms_and_no_people_memory():
    assert set(DEFAULT) == set(ALL) - MANIPULATION - ARM_ATOMICS - MEMORY - PICK - {search_web}


def test_pick_tools_names_pick_object_alone():
    assert PICK_TOOLS == frozenset({"pick_object"})


def test_pick_object_is_in_hri_alone():
    # It needs the arms and the object memory, and HRI is the one set with both.
    assert PICK <= set(HRI)
    assert PICK.isdisjoint(GPSR)
    assert PICK.isdisjoint(DEFAULT)
    assert PICK.isdisjoint(ARMS)


@pytest.mark.parametrize(
    ("arms", "object_memory", "bound"),
    [(False, False, False), (True, False, False), (False, True, False), (True, True, True)],
)
def test_pick_object_is_bound_only_with_both_the_arms_and_the_object_memory(
    arms, object_memory, bound
):
    names_bound = bound_tool_names("hri", arms=arms, object_memory=object_memory)
    assert ("pick_object" in names_bound) is bound
    assert [fn.__name__ for fn in bound_tools("hri", arms=True, object_memory=True)][-1] == (
        "pick_object"
    )


@pytest.mark.parametrize("tool_set", TOOL_SETS)
def test_search_web_is_bound_last_in_every_set_with_web_search_alone(tool_set):
    without = bound_tools(tool_set, arms=True, object_memory=True)
    with_search = bound_tools(tool_set, arms=True, object_memory=True, web_search=True)
    assert search_web not in without
    assert with_search == [*without, search_web]
    assert WEB_SEARCH_TOOL == "search_web"
    assert WEB_SEARCH_TOOL in bound_tool_names(tool_set, web_search=True)
    assert WEB_SEARCH_TOOL not in bound_tool_names(tool_set)


def test_search_web_is_in_no_set_list():
    # The lists stay the tool text measured before it.
    assert all(search_web not in tool_set for tool_set in SETS.values())


def test_default_keeps_its_order_without_the_memory_tools():
    assert names(DEFAULT) == [
        "list_places",
        "go_to",
        "move_relative",
        "say",
        "find_objects",
        "describe_objects",
        "object_info",
        "find_people",
        "face_person",
        "get_facts",
        "recall_object",
        "list_room_objects",
        "go_to_object",
    ]


def test_default_without_the_object_memory_tools_is_the_set_it_was_before_them():
    assert [name for name in names(DEFAULT) if name not in OBJECT_MEMORY_TOOLS] == [
        "list_places",
        "go_to",
        "move_relative",
        "say",
        "find_objects",
        "describe_objects",
        "object_info",
        "find_people",
        "face_person",
        "get_facts",
    ]


def test_the_object_memory_tools_are_in_default_and_hri_alone():
    # GPSR keeps the tool text the Mahoraga runs measured, and ARMS is the arms alone.
    assert OBJECT_MEMORY <= set(DEFAULT)
    assert OBJECT_MEMORY <= set(HRI)
    assert OBJECT_MEMORY.isdisjoint(GPSR)
    assert OBJECT_MEMORY.isdisjoint(ARMS)


@pytest.mark.parametrize("tool_set", [HRI, DEFAULT], ids=["HRI", "DEFAULT"])
def test_describe_objects_comes_right_after_find_objects_in_hri_and_default(tool_set):
    assert names(tool_set)[names(tool_set).index("find_objects") + 1] == "describe_objects"


def test_describe_objects_is_in_no_other_set():
    # GPSR keeps the tool text the Mahoraga runs measured, and ARMS is the arms alone.
    assert describe_objects not in GPSR
    assert describe_objects not in ARMS
    assert all("describe_objects" in bound_tool_names(name) for name in ("hri", "default"))


def test_the_memory_tools_are_in_hri_alone():
    # No effect covers a memory write yet, so GPSR and Mahoraga never see them (ADR-0070).
    assert MEMORY <= set(HRI)
    assert MEMORY.isdisjoint(GPSR)
    assert MEMORY.isdisjoint(DEFAULT)
    assert MEMORY.isdisjoint(ARMS)


def test_every_public_atomic_is_in_a_set():
    public = {
        getattr(atomic, name)
        for name in atomic.__all__
        if inspect.isfunction(getattr(atomic, name))
    }
    assert public == set(ALL)


@pytest.mark.parametrize("tool_set", SETS.values(), ids=SETS.keys())
def test_a_set_has_no_duplicate(tool_set):
    assert len(set(tool_set)) == len(tool_set)


@pytest.mark.parametrize("fn", ALL, ids=lambda fn: fn.__name__)
def test_an_atomic_takes_ctx_first_and_is_fully_typed(fn):
    signature = inspect.signature(fn)
    parameters = list(signature.parameters.values())
    assert parameters[0].name == "ctx"
    assert all(parameter.annotation is not inspect.Parameter.empty for parameter in parameters)
    assert signature.return_annotation is not inspect.Signature.empty


@pytest.mark.parametrize("fn", ALL, ids=lambda fn: fn.__name__)
def test_a_docstring_says_what_comes_back_and_names_every_parameter_but_ctx(fn):
    doc = inspect.getdoc(fn)
    assert "Returns" in doc.partition("Args:")[0]
    assert "Returns:" not in doc
    parameters = [name for name in inspect.signature(fn).parameters if name != "ctx"]
    for parameter in parameters:
        assert f"{parameter}: " in doc.partition("Args:")[2]
    assert "ctx" not in doc
