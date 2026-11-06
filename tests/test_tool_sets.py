"""The per-test tool sets and what every atomic in them must offer."""

import inspect

import pytest

from walkie_agent import atomics
from walkie_agent.atomics import (
    DEFAULT,
    GPSR,
    HRI,
    hand_over,
    pick,
    place_on,
    recognize_person,
    remember_person,
)

ARMS = {pick, place_on, hand_over}
MEMORY = {remember_person, recognize_person}
ALL = sorted(set(GPSR) | set(HRI) | set(DEFAULT), key=lambda fn: fn.__name__)


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


def test_hri_has_no_object_or_arm_tools_and_ends_with_the_memory_tools():
    assert names(HRI) == [
        "list_places",
        "go_to",
        "move_relative",
        "say",
        "find_people",
        "face_person",
        "remember_person",
        "recognize_person",
    ]


def test_default_is_every_atomic_that_needs_no_arms_and_no_people_memory():
    assert set(DEFAULT) == set(ALL) - ARMS - MEMORY


def test_default_keeps_its_order_without_the_memory_tools():
    assert names(DEFAULT) == [
        "list_places",
        "go_to",
        "move_relative",
        "say",
        "find_objects",
        "object_info",
        "find_people",
        "face_person",
        "get_facts",
    ]


def test_the_memory_tools_are_in_hri_alone():
    # No effect covers a memory write yet, so GPSR and Mahoraga never see them (ADR-0070).
    assert MEMORY <= set(HRI)
    assert MEMORY.isdisjoint(GPSR)
    assert MEMORY.isdisjoint(DEFAULT)


def test_every_public_atomic_is_in_a_set():
    public = {
        getattr(atomics, name)
        for name in atomics.__all__
        if inspect.isfunction(getattr(atomics, name))
    }
    assert public == set(ALL)


@pytest.mark.parametrize("tool_set", [GPSR, HRI, DEFAULT], ids=["GPSR", "HRI", "DEFAULT"])
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
