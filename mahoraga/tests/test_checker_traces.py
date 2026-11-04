"""Hand-written traces that judge the checkers from outside.

The oracle and the checkers share an author, so the gate's 100% only shows that they
agree. These traces come from the review's checker probe. Each one and its expected
result was written from the command's sentence alone. Expected results stay literal
here, and nothing in this file may import or call the oracle.
"""

from functools import lru_cache

from mahoraga.checkers import check
from mahoraga.commands import commands
from mahoraga.driver import allow_list
from mahoraga.worlds import build_world, world_seed
from walkie_agent.atomics import GPSR
from walkie_agent.capabilities.fake import FakeRobot


@lru_cache
def _records():
    return {record["seed"]: record for record in commands("trial", 300)}


def _run(seed, trace):
    """Run `trace(f, start, spec)` on command `seed`, in its epoch 1 world, and check the end."""
    spec = _records()[seed]["spec"]
    robot = FakeRobot(build_world(spec, world_seed(seed, 1)))
    start = robot.snapshot()
    trace(allow_list(GPSR, robot.ctx), start, spec)
    return check(spec, start, robot.snapshot())


# Relative moves that change room (M11).

SEATED_IN_LIVING_ROOM = 81  # "tell me how many sitting persons are in the living room": one
STANDING_IN_KITCHEN = 163  # "tell me how many standing persons are in the kitchen room": one


def _look_after_moving_to_the_kitchen(f, _start, _spec):
    f["go_to"]("living room")
    f["move_relative"](0.0, -3.0)
    seen = f["find_people"]()
    assert seen and all(person["room"] == "kitchen room" for person in seen)
    f["say"]("1")


def test_a_look_from_the_wrong_room_after_a_relative_move_does_not_count():
    ok, why = _run(SEATED_IN_LIVING_ROOM, _look_after_moving_to_the_kitchen)

    assert not ok, why


def test_a_look_from_the_right_room_after_a_relative_move_counts():
    ok, why = _run(STANDING_IN_KITCHEN, _look_after_moving_to_the_kitchen)

    assert ok, why


def test_a_look_that_a_relative_move_left_the_room_before_still_counts():
    def looks_then_leaves(f, _start, _spec):
        f["go_to"]("living room")
        f["find_people"]()
        f["move_relative"](0.0, -3.0)
        f["say"]("1")

    ok, why = _run(SEATED_IN_LIVING_ROOM, looks_then_leaves)

    assert ok, why


def test_a_look_from_a_room_the_robot_walked_back_into_counts():
    def out_and_back(f, _start, _spec):
        f["go_to"]("living room")
        f["move_relative"](0.0, -3.0)
        f["move_relative"](0.0, 3.0)
        assert all(person["room"] == "living room" for person in f["find_people"]())
        f["say"]("1")

    ok, why = _run(SEATED_IN_LIVING_ROOM, out_and_back)

    assert ok, why
