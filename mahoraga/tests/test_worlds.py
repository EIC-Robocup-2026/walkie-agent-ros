"""Every generated command has a world in which it can be done, over many seeds."""

import json

import pytest

from mahoraga.commands import commands
from mahoraga.worlds import OPERATOR, build_world, world_seed
from walkie_agent.capabilities.fake import FakeRobot


def _go(robot, place):
    robot.go_to(place)


def _assert_feasible(spec, world):
    """Walk the fake robot to what the command needs, through its own verbs."""
    robot = FakeRobot(world)
    kind = spec["kind"]
    if kind == "move_object":
        _go(robot, spec["source"])
        assert robot.objects(spec["want"]), spec
        dest = spec["dest"]
        if dest["to"] == "place":
            assert dest["place"] in world["places"]
        elif dest["to"] == "person":
            _go(robot, dest["room"])
            found = robot.people(dest.get("name") or dest["description"])
            assert [seen for seen in found if seen.id != OPERATOR], spec
    elif kind == "count_objects":
        _go(robot, spec["place"])
        assert robot.objects(spec["category"]), spec
    elif kind == "best_object":
        _go(robot, spec["place"])
        names = {seen.name for seen in robot.objects(spec["category"] or "")}
        assert len(names) >= 2, spec
    elif kind == "count_people":
        _go(robot, spec["room"])
        assert robot.people(spec["description"]), spec
    elif kind in ("person_info", "relay_info"):
        _go(robot, spec["room"])
        assert len([seen for seen in robot.people() if seen.id != OPERATOR]) == 1, spec
        if kind == "relay_info":
            _go(robot, spec["to_room"])
            assert [seen for seen in robot.people() if seen.id != OPERATOR], spec
    else:
        _go(robot, spec["place"])
        person = spec["person"]
        found = robot.people(person.get("name") or person["description"])
        assert [seen for seen in found if seen.id != OPERATOR], spec
        if kind == "escort" and spec["dest"]:
            _go(robot, spec["dest"])


@pytest.mark.parametrize("category", ["mixed", "objects", "people"])
def test_every_command_is_feasible_over_many_seeds(category):
    records = commands("trial", 400, category)

    for record in records:
        for epoch in (1, 2, 3):
            _assert_feasible(record["spec"], build_world(record["spec"], world_seed(record["seed"], epoch)))


def test_a_world_is_json_and_repeats_for_its_seed():
    spec = commands("trial", 1)[0]["spec"]

    assert json.dumps(build_world(spec, 5), sort_keys=True) == json.dumps(build_world(spec, 5), sort_keys=True)


def test_epochs_of_one_command_get_different_worlds():
    spec = commands("trial", 1)[0]["spec"]

    assert build_world(spec, world_seed(0, 1)) != build_world(spec, world_seed(0, 2))


def test_the_operator_never_fits_a_commands_selector():
    for record in commands("trial", 200):
        world = build_world(record["spec"], world_seed(record["seed"], 1))
        operator = world["people"][OPERATOR]
        assert operator["name"] == OPERATOR
        assert operator["gesture"] == "no gesture"
        assert operator["room"] == "living room"


def test_names_stay_unique_when_the_builder_renames_someone():
    for record in commands("trial", 300, "people"):
        world = build_world(record["spec"], world_seed(record["seed"], 1))
        names = [who["name"] for who in world["people"].values()]
        assert len(set(names)) == len(names), record["spec"]


def test_a_description_that_is_no_pose_or_gesture_is_a_bug():
    spec = {"kind": "talk", "person": {"room": "living room", "description": "juggling person"}, "place": "sofa", "topic": "x", "greet": False}

    with pytest.raises(ValueError, match="juggling"):
        build_world(spec, 1)
