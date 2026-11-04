"""The per-test tool sets and what every atomic in them must offer."""

import inspect

import pytest

import walkie_agent.atomics as atomics
from walkie_agent.atomics import DEFAULT, GPSR, HRI, hand_over, pick, place_on

ARMS = {pick, place_on, hand_over}
ALL = sorted(set(GPSR) | set(HRI) | set(DEFAULT), key=lambda fn: fn.__name__)


def names(tool_set) -> list[str]:
    return [fn.__name__ for fn in tool_set]


def test_gpsr_is_the_set_the_spike_used():
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
    ]


def test_hri_has_no_object_or_arm_tools():
    assert names(HRI) == ["list_places", "go_to", "move_relative", "say", "find_people"]


def test_default_is_every_atomic_that_needs_no_arms():
    assert set(DEFAULT) == set(ALL) - ARMS


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
