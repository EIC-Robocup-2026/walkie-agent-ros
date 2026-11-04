"""The atomics over FakeRobot: what they return and how they fail."""

import json
import math

import pytest

from walkie_agent.atomics import (
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
from walkie_agent.capabilities import CancelToken, Ctx
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.capabilities.guards import MAX_MOVE_M
from walkie_agent.errors import ErrorCode, StepError

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
