"""The atomics over FakeRobot: what they return and how they fail."""

import copy
import json
import logging
import math

import pytest

from walkie_agent.atomics import (
    face_person,
    find_objects,
    find_people,
    go_to,
    hand_over,
    list_places,
    move_relative,
    object_info,
    pick,
    place_on,
    say,
)
from walkie_agent.capabilities import CancelToken, Ctx, PersonSeen
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.capabilities.guards import MAX_MOVE_M
from walkie_agent.errors import ErrorCode, GoalFailed, StepError

WORLD = {
    "places": {
        "kitchen room": {"x": 0.0, "y": 0.0, "yaw": 0.0, "room": "kitchen room"},
        "shelf": {"x": 1.0, "y": 0.0, "yaw": 0.0, "room": "kitchen room"},
        "table": {"x": 0.0, "y": 1.0, "yaw": 0.0, "room": "kitchen room"},
    },
    "objects": {
        "apple_1": {
            "name": "apple",
            "category": "fruit",
            "size_cm": 8,
            "weight_g": 180,
            "thickness_cm": 8,
            "at": "shelf",
        },
        "banana_1": {
            "name": "banana",
            "category": "fruit",
            "size_cm": 19,
            "weight_g": 120,
            "thickness_cm": 4,
            "at": "shelf",
        },
    },
    "people": {
        "jane": {
            "name": "Jane",
            "room": "kitchen room",
            "pose": "sitting person",
            "gesture": "waving person",
        },
    },
    "start": "kitchen room",
}


@pytest.fixture
def robot() -> FakeRobot:
    return FakeRobot(WORLD)


def code_of(call, *args, **kwargs) -> ErrorCode:
    with pytest.raises(StepError) as caught:
        call(*args, **kwargs)
    return caught.value.code


def test_list_places_gives_names_and_descriptions(robot):
    places = list_places(robot.ctx)
    assert {"name": "shelf", "description": "In the kitchen room."} in places
    assert {"name": "kitchen room", "description": "A room."} in places


def test_go_to_says_where_it_arrived(robot):
    assert go_to(robot.ctx, "shelf") == "Arrived at shelf."
    assert robot.snapshot()["at"] == "shelf"


def test_go_to_raises_the_verb_s_error(robot):
    assert code_of(go_to, robot.ctx, "moon") is ErrorCode.NOT_FOUND


def test_move_relative_describes_the_move_and_the_pose(robot):
    reply = move_relative(robot.ctx, 1.0, -0.5, 90.0)
    assert reply == (
        "Moved 1 m forward, 0.5 m right, turning 90 degrees left. "
        "Now at x=1.00 m, y=-0.50 m, heading 90 degrees."
    )


def test_move_relative_never_prints_negative_zero(robot):
    robot.go_to("shelf")
    # y ends at -0.001, which rounds to -0.0 and would print as "-0.00".
    assert "x=2.00 m, y=0.00 m, heading 0 degrees" in move_relative(robot.ctx, 1.0, -0.001)


@pytest.mark.parametrize(
    "args", [(math.nan,), (MAX_MOVE_M + 1,), (0.0,), (1.0, math.inf), (1.0, 0.0, -math.inf)]
)
def test_move_relative_raises_the_guards_refusal(robot, args):
    assert code_of(move_relative, robot.ctx, *args) is ErrorCode.REFUSED


def test_move_relative_tells_the_llm_the_cap():
    assert f"at most {MAX_MOVE_M:g} m" in move_relative.__doc__


def test_say_repeats_the_line(robot):
    assert say(robot.ctx, "Hello.") == "Said: Hello."
    assert robot.said == ["Hello."]


def test_say_raises_on_blank_text(robot):
    assert code_of(say, robot.ctx, " ") is ErrorCode.REFUSED


def test_find_objects_lists_what_is_in_view_without_its_measures(robot):
    robot.go_to("shelf")
    assert find_objects(robot.ctx, "apple") == [
        {"id": "apple_1", "name": "apple", "category": "fruit", "place": "shelf"}
    ]
    assert [seen["id"] for seen in find_objects(robot.ctx)] == ["apple_1", "banana_1"]


def test_find_objects_is_empty_when_nothing_matches(robot):
    assert find_objects(robot.ctx, "cola") == []


def test_object_info_adds_the_measures(robot):
    robot.go_to("shelf")
    assert object_info(robot.ctx, "banana_1") == {
        "id": "banana_1",
        "name": "banana",
        "category": "fruit",
        "place": "shelf",
        "size_cm": 19,
        "weight_g": 120,
        "thickness_cm": 4,
    }


def test_object_info_refuses_an_object_out_of_view(robot):
    robot.go_to("table")
    assert code_of(object_info, robot.ctx, "apple_1") is ErrorCode.NOT_FOUND


def test_find_people_lists_who_is_in_the_room(robot):
    assert find_people(robot.ctx, "waving person") == [
        {
            "id": "jane",
            "name": "Jane",
            "pose": "sitting person",
            "gesture": "waving person",
            "room": "kitchen room",
        }
    ]
    assert find_people(robot.ctx, "lying person") == []


def facing(heading_deg: float, start_deg: float = 0.0) -> FakeRobot:
    """WORLD with jane at `heading_deg` in the map and the robot starting at `start_deg`."""
    world = copy.deepcopy(WORLD)
    world["people"]["jane"]["heading_deg"] = heading_deg
    world["places"]["kitchen room"]["yaw"] = math.radians(start_deg)
    return FakeRobot(world)


def moves(robot: FakeRobot) -> list[dict]:
    return [call["args"] for call in robot.calls if call["verb"] == "base.move_relative"]


def aborted() -> GoalFailed:
    """What RosBase raises when Nav2 aborts the goal."""
    return GoalFailed(ErrorCode.SKILL_FAILED, "Navigation failed for the move (ABORTED, error code 105).")


def unknown() -> StepError:
    """What RosBase raises when the result request fails: the goal's state is unknown."""
    return StepError(
        ErrorCode.SKILL_FAILED, "Navigation failed for the move: the request for its result failed."
    )


class ShortTurns:
    """A base whose turns end `short_deg` before their goal, or past it when negative,
    and then raise `error`.

    As Nav2 does when its progress checker aborts a turn on the spot that is still
    settling: the robot has turned, and the action reports SKILL_FAILED. Both such
    aborts in the Chula simulation ended about 3 degrees past the goal.
    """

    def __init__(self, robot: FakeRobot, short_deg: float, error: StepError) -> None:
        self.robot, self.short_deg, self.error = robot, short_deg, error

    def move_relative(self, forward_m=0.0, left_m=0.0, turn_deg=0.0, *, timeout_s=60.0):
        ended = turn_deg - self.short_deg * math.copysign(1.0, turn_deg)
        self.robot.move_relative(forward_m, left_m, ended)
        raise self.error

    def __getattr__(self, name):
        return getattr(self.robot, name)


def short_turns(robot: FakeRobot, short_deg: float, error: StepError | None = None) -> Ctx:
    base = ShortTurns(robot, short_deg, error or aborted())
    return Ctx(cancel=robot.cancel, base=base, speech=robot, vision=robot)


def test_find_people_leaves_the_bearing_out():
    assert "bearing_deg" not in find_people(facing(40.0).ctx)[0]


@pytest.mark.parametrize(
    ("heading", "start", "reply", "turn"),
    [
        (40.0, 0.0, "Turned 40 degrees left to face jane.", 40.0),
        (-75.5, 0.0, "Turned 76 degrees right to face jane.", -75.5),
        # The short way round: -150 from 170 is 40 to the left, not 320 to the right.
        (-150.0, 170.0, "Turned 40 degrees left to face jane.", 40.0),
    ],
)
def test_face_person_turns_on_the_spot_by_the_bearing(heading, start, reply, turn):
    robot = facing(heading, start)
    assert face_person(robot.ctx, "jane") == reply
    assert moves(robot) == [{"forward_m": 0.0, "left_m": 0.0, "turn_deg": pytest.approx(turn)}]
    assert robot.people()[0].bearing_deg == pytest.approx(0.0, abs=0.06)
    assert robot.snapshot()["at"] == "kitchen room"


@pytest.mark.parametrize(
    ("heading", "reply"),
    [
        (10.0, "Already facing jane, 10 degrees to the left."),
        (-3.4, "Already facing jane, 3 degrees to the right."),
        (0.4, "Already facing jane, straight ahead."),
    ],
)
def test_a_person_within_10_degrees_gets_no_turn(heading, reply):
    robot = facing(heading)
    assert face_person(robot.ctx, "jane") == reply
    assert moves(robot) == []


def test_a_person_not_in_view_is_not_found_and_nothing_moves(robot):
    with pytest.raises(StepError) as raised:
        face_person(robot.ctx, "bob")
    assert raised.value.code is ErrorCode.NOT_FOUND
    assert "Call find_people first" in str(raised.value)
    assert moves(robot) == []


def test_a_person_whose_direction_is_unknown_is_unavailable(robot):
    class NoBearings:
        def people(self, query="", *, timeout_s=5.0):
            return [PersonSeen("person_1", None, "standing person", "", "living room")]

    ctx = Ctx(cancel=robot.cancel, base=robot, speech=robot, vision=NoBearings())
    with pytest.raises(StepError) as raised:
        face_person(ctx, "person_1")
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert moves(robot) == []


@pytest.mark.parametrize(
    ("short", "reply"),
    [
        (3.0, "Turned 37 degrees left to face jane."),
        (-2.9, "Turned 43 degrees left to face jane."),
    ],
)
def test_a_turn_reported_failed_that_ends_within_10_degrees_still_faces_the_person(caplog, short, reply):
    robot = facing(40.0)
    with caplog.at_level(logging.INFO, logger="walkie_agent.atomics.navigation"):
        assert face_person(short_turns(robot, short), "jane") == reply
    assert f"counts the turn to jane as done, {abs(short):.1f} degrees off" in caplog.text


def test_a_turn_reported_failed_that_ends_further_off_raises_the_failure():
    robot = facing(40.0)
    error = aborted()
    with pytest.raises(StepError) as raised:
        face_person(short_turns(robot, 15.0, error), "jane")
    assert raised.value is error


def test_a_turn_whose_end_is_unknown_raises_its_failure_even_within_10_degrees():
    robot = facing(40.0)
    reads = []
    error = unknown()
    ctx = short_turns(robot, 3.0, error)
    ctx.base.pose = lambda **kwargs: reads.append(kwargs) or robot.pose(**kwargs)
    with pytest.raises(StepError) as raised:
        face_person(ctx, "jane")
    assert raised.value is error
    assert len(reads) == 1  # The start only: the base may still be turning.


@pytest.mark.parametrize("code", [ErrorCode.CANCELLED, ErrorCode.TIMEOUT, ErrorCode.UNAVAILABLE])
def test_any_other_failure_of_the_turn_is_raised_as_it_is(code):
    robot = facing(40.0)
    robot.faults["base.move_relative"] = StepError(code, "The turn did not happen.")
    with pytest.raises(StepError) as raised:
        face_person(robot.ctx, "jane")
    assert raised.value.code is code


def test_every_return_value_is_json_able(robot):
    ctx = robot.ctx
    replies = [
        list_places(ctx),
        go_to(ctx, "shelf"),
        move_relative(ctx, 0.1),
        say(ctx, "Hi."),
        find_objects(ctx),
        object_info(ctx, "apple_1"),
        find_people(ctx),
        face_person(ctx, "jane"),
        pick(ctx, "apple_1"),
        place_on(ctx, "shelf"),
        pick(ctx, "apple_1"),
        hand_over(ctx, "jane"),
    ]
    assert json.loads(json.dumps(replies, allow_nan=False)) == replies


def test_a_pick_place_and_hand_over_task_ends_in_the_right_state(robot):
    ctx = robot.ctx
    go_to(ctx, "shelf")
    fruits = find_objects(ctx, "fruits")
    measured = [object_info(ctx, seen["id"]) for seen in fruits]
    lightest = min(measured, key=lambda item: item["weight_g"])["id"]
    assert pick(ctx, lightest) == "Holding banana_1."
    go_to(ctx, "table")
    assert place_on(ctx, "table") == "Placed banana_1 on table."
    assert pick(ctx, "banana_1") == "Holding banana_1."
    go_to(ctx, "kitchen room")
    person = find_people(ctx, "waving person")[0]["id"]
    assert hand_over(ctx, person) == "Handed banana_1 to jane."
    say(ctx, "Here is your banana.")

    snapshot = robot.snapshot()
    assert snapshot["objects"]["banana_1"]["at"] == "jane"
    assert snapshot["objects"]["apple_1"]["at"] == "shelf"
    assert snapshot["holding"] is None
    assert snapshot["said"] == ["Here is your banana."]


@pytest.mark.parametrize(
    "call",
    [
        lambda ctx: pick(ctx, "apple_1"),
        lambda ctx: place_on(ctx, "shelf"),
        lambda ctx: hand_over(ctx, "jane"),
    ],
)
def test_the_arm_atomics_raise_on_a_robot_with_no_arms(robot, call):
    ctx = Ctx(cancel=CancelToken(), base=robot, speech=robot, vision=robot)
    assert code_of(call, ctx) is ErrorCode.UNAVAILABLE


def test_a_set_token_stops_the_atomics_but_not_say(robot):
    robot.cancel.cancel()
    assert code_of(go_to, robot.ctx, "shelf") is ErrorCode.CANCELLED
    assert code_of(find_objects, robot.ctx) is ErrorCode.CANCELLED
    assert say(robot.ctx, "Stopping.") == "Said: Stopping."


def test_a_verb_fault_reaches_the_caller(robot):
    robot.faults["vision.people"] = StepError(ErrorCode.TIMEOUT, "The camera went quiet.")
    assert code_of(find_people, robot.ctx) is ErrorCode.TIMEOUT
