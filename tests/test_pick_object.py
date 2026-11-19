"""pick_object over FakeRobot with arm poses and a stand-in object memory (decision PK-4)."""

import math
import re

import pytest

from conftest import PROJECT_ROOT
from test_object_memory_atomics import WORLD, Remembers
from walkie_agent.arm.pick import heading_from, load_pick_settings
from walkie_agent.arm.poses import load_arm_poses
from walkie_agent.arm.state import NOT_SAFE_SENTENCE, SAFE_SENTENCE
from walkie_agent.capabilities import Ctx, RecalledObject
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.errors import ErrorCode, GoalFailed, StepError
from walkie_agent.tools.atomic import pick_object

ABORTED = "Nav2 aborted the goal."
READS = frozenset(("base.pose", "base.places"))


@pytest.fixture(autouse=True)
def project_dir(monkeypatch):
    # pick_object reads config/pick.toml from the working directory, as main.py does.
    monkeypatch.chdir(PROJECT_ROOT)


@pytest.fixture(scope="module")
def arm_poses():
    return load_arm_poses(
        PROJECT_ROOT / "config" / "arm_poses.toml", PROJECT_ROOT / "config" / "robot_model.toml"
    )


def remembered(
    name: str = "water bottle",
    *,
    object_id: str = "7",
    x: float | None = 1.0,
    y: float | None = 0.0,
    z: float | None = 0.75,
    room: str | None = "kitchen room",
) -> RecalledObject:
    return RecalledObject(
        id=object_id,
        name=name,
        label="bottle",
        category="drink",
        room=room,
        on="kitchen table 3",
        state="present",
        seen_s_ago=12.0,
        x=x,
        y=y,
        z=z,
    )


def picking(arm_poses, *objects: RecalledObject, holds: bool = True):
    """A FakeRobot at the kitchen room's (0, 0), facing +x, and a ctx with every capability."""
    robot = FakeRobot(WORLD, arm_poses=arm_poses)
    if holds:
        robot.grasping.add("right")
    ctx = Ctx(
        cancel=robot.cancel,
        base=robot,
        speech=robot,
        vision=robot,
        arms=robot,
        object_memory=Remembers(*objects),
    )
    return robot, ctx


def verbs(robot: FakeRobot) -> list[str]:
    """The verbs that move or command something, without the base's reads."""
    return [call["verb"] for call in robot.calls if call["verb"] not in READS]


def reaches(robot: FakeRobot) -> list[tuple[bool, bool, bool, float]]:
    return [
        (
            call["args"]["whole_body"],
            call["args"]["cartesian"],
            call["args"]["touch"],
            call["args"]["timeout_s"],
        )
        for call in robot.calls
        if call["verb"] == "arms.reach"
    ]


def at_stand_off(robot: FakeRobot, x: float, y: float) -> None:
    """The base stands at a stand-off pose of the object at (x, y), facing its heading."""
    here = robot.pose()
    settings = load_pick_settings()
    distance = math.hypot(settings.standoff_m, settings.standoff_left_m)
    assert math.hypot(x - here.x, y - here.y) == pytest.approx(distance)
    assert heading_from(settings, here.x, here.y, x, y) == pytest.approx(here.yaw)


def fail_on_call(monkeypatch, robot: FakeRobot, verb: str, nth: int, error: StepError) -> None:
    """Make the nth call of robot.<verb> raise `error`; every other call runs as before."""
    original = getattr(robot, verb)
    count = 0

    def flaky(*args, **kwargs):
        nonlocal count
        count += 1
        if count == nth:
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(robot, verb, flaky)


def fails(ctx: Ctx, name: str = "water bottle") -> StepError:
    with pytest.raises(StepError) as raised:
        pick_object(ctx, name)
    return raised.value


# The pick


def test_a_pick_drives_to_the_stand_off_reaches_closes_and_lifts(arm_poses):
    robot, ctx = picking(arm_poses, remembered())

    answer = pick_object(ctx, "water bottle")

    assert answer == (
        "Picked up the water bottle with the right hand and lifted it. Before you drive, "
        f"move the right arm to a pose that is safe for driving. {NOT_SAFE_SENTENCE}"
    )
    assert verbs(robot) == [
        "base.move_relative",
        "arms.gripper",
        "arms.reach",
        "arms.reach",
        "arms.gripper",
        "arms.reach",
    ]
    # 0.65 m back from the object and 0.20 m to the left of the right hand's line to
    # it, which is the line the robot stood on.
    move = next(call for call in robot.calls if call["verb"] == "base.move_relative")
    heading = math.asin(0.20)
    assert move["args"]["turn_deg"] == pytest.approx(math.degrees(heading))
    at_stand_off(robot, 1.0, 0.0)
    grippers = [call["args"]["state"] for call in robot.calls if call["verb"] == "arms.gripper"]
    assert grippers == ["open", "close"]
    # The whole-body reach to the pre-grasp pose, then the straight approach and lift,
    # on which the fingers may touch the octomap.
    assert reaches(robot) == [
        (True, False, False, 90.0),
        (False, True, True, 30.0),
        (False, True, True, 30.0),
    ]
    arms = {call["args"]["arm"] for call in robot.calls if call["verb"].startswith("arms.")}
    assert arms == {"right"}


def test_the_answer_holds_no_number_of_the_geometry(arm_poses):
    _, ctx = picking(arm_poses, remembered(x=2.37, y=-1.21, z=0.83))

    assert not re.search(r"\d", pick_object(ctx, "water bottle"))


def test_the_first_match_with_a_full_position_is_picked(arm_poses):
    robot, ctx = picking(
        arm_poses,
        remembered(object_id="3", z=None),
        remembered(object_id="7", x=0.0, y=1.0),
    )

    pick_object(ctx, "water bottle")

    # The second one, to the left: the base turns a quarter turn and a little more, to
    # face it with the right hand.
    move = next(call for call in robot.calls if call["verb"] == "base.move_relative")
    assert move["args"]["turn_deg"] == pytest.approx(90.0 + math.degrees(math.asin(0.20)))
    at_stand_off(robot, 0.0, 1.0)
    assert ctx.object_memory.calls == [("search", "water bottle", 10)]


def test_a_robot_already_at_the_stand_off_does_not_drive(arm_poses):
    robot, ctx = picking(arm_poses, remembered(x=0.65, y=-0.20))

    pick_object(ctx, "water bottle")

    assert "base.move_relative" not in verbs(robot)
    assert verbs(robot)[0] == "arms.gripper"


def test_a_far_object_is_driven_to_by_its_nearest_place_first(arm_poses):
    # The sofa is at (10, 2) facing +x: the bottle is 1 m behind it.
    robot, ctx = picking(arm_poses, remembered(x=9.0, y=2.0, room="living room"))

    pick_object(ctx, "water bottle")

    assert verbs(robot)[:2] == ["base.go_to", "base.move_relative"]
    assert robot.calls[[c["verb"] for c in robot.calls].index("base.go_to")]["args"]["place"] == (
        "sofa"
    )
    # It turns round to face the bottle from the sofa's side.
    move = next(call for call in robot.calls if call["verb"] == "base.move_relative")
    assert abs(move["args"]["turn_deg"]) > 150.0
    at_stand_off(robot, 9.0, 2.0)
    assert robot.pose().x > 9.0


def test_a_far_object_with_no_place_near_it_is_not_driven_to(arm_poses):
    robot, ctx = picking(arm_poses, remembered(x=20.0, y=20.0, room=None))

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        "The water bottle is far away, and Walkie knows no named place within 2 m of it "
        "to drive to first."
    )
    assert verbs(robot) == []


# Driving to the stand-off


def test_a_failed_drive_tries_the_next_heading(arm_poses, monkeypatch):
    robot, ctx = picking(arm_poses, remembered())
    aborted = GoalFailed(ErrorCode.SKILL_FAILED, ABORTED)
    fail_on_call(monkeypatch, robot, "move_relative", 1, aborted)

    pick_object(ctx, "water bottle")

    # The first heading's drive raised before the fake logged it; the second is 30
    # degrees round, and the base ends at its stand-off pose.
    move = next(call for call in robot.calls if call["verb"] == "base.move_relative")
    first = math.degrees(math.asin(0.20))
    assert move["args"]["turn_deg"] == pytest.approx(first + 30.0)
    assert math.degrees(robot.pose().yaw) == pytest.approx(first + 30.0)
    at_stand_off(robot, 1.0, 0.0)


def test_a_pick_that_reaches_no_stand_off_says_how_many_tries(arm_poses):
    robot, ctx = picking(arm_poses, remembered())
    robot.faults["base.move_relative"] = GoalFailed(ErrorCode.SKILL_FAILED, ABORTED)

    error = fails(ctx)

    assert error.code is ErrorCode.SKILL_FAILED
    assert error.message == (
        "Walkie could not reach a spot to pick up the water bottle from, after 5 tries. "
        "The last one: Nav2 aborted the goal."
    )
    assert verbs(robot) == ["base.move_relative"] * 5


def test_an_arm_out_stops_the_drive_without_another_try(arm_poses):
    robot, ctx = picking(arm_poses, remembered())
    robot.move_to("hands_up")

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert verbs(robot) == ["arms.move_to", "base.move_relative"]


def test_a_stop_ends_the_pick_without_another_try(arm_poses):
    robot, ctx = picking(arm_poses, remembered())
    robot.cancel.cancel()

    error = fails(ctx)

    assert error.code is ErrorCode.CANCELLED
    assert verbs(robot) == ["base.move_relative"]


# The grasp


def test_a_close_on_nothing_does_not_lift(arm_poses):
    robot, ctx = picking(arm_poses, remembered(), holds=False)

    error = fails(ctx)

    assert error.code is ErrorCode.SKILL_FAILED
    assert error.message == (
        "Walkie reached the water bottle, but the right hand closed on nothing, so it did "
        "not pick it up. The object may have moved: call find_objects to look again. "
        f"{NOT_SAFE_SENTENCE}"
    )
    assert len(reaches(robot)) == 2


def test_a_failed_reach_says_which_step_failed_and_keeps_its_code(arm_poses):
    robot, ctx = picking(arm_poses, remembered())
    robot.faults["arms.reach"] = StepError(
        ErrorCode.SKILL_FAILED, f"MoveIt found no path. {SAFE_SENTENCE}"
    )

    error = fails(ctx)

    assert error.code is ErrorCode.SKILL_FAILED
    assert error.message == (
        f"The reach toward the water bottle failed: MoveIt found no path. {SAFE_SENTENCE}"
    )
    assert "close" not in [c["args"].get("state") for c in robot.calls]


def test_a_failed_lift_says_the_hand_holds_the_object(arm_poses, monkeypatch):
    robot, ctx = picking(arm_poses, remembered())
    stopped = GoalFailed(ErrorCode.SKILL_FAILED, f"The commander aborted it. {NOT_SAFE_SENTENCE}")
    fail_on_call(monkeypatch, robot, "reach", 3, stopped)

    error = fails(ctx)

    assert isinstance(error, GoalFailed)
    assert error.message == (
        "The right hand holds the water bottle, but the lift failed: The commander aborted "
        f"it. {NOT_SAFE_SENTENCE}"
    )


# What it cannot pick


def test_an_object_it_does_not_remember_is_not_found(arm_poses):
    robot, ctx = picking(arm_poses, remembered(name="cup"))

    error = fails(ctx, " water bottle ")

    assert error.code is ErrorCode.NOT_FOUND
    assert error.message == (
        "Walkie remembers no water bottle. Call recall_object to see what it remembers."
    )
    assert robot.calls == []


def test_an_object_with_no_height_cannot_be_reached(arm_poses):
    robot, ctx = picking(arm_poses, remembered(z=None), remembered(object_id="8", x=None))

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert "not exactly where it is" in error.message
    assert robot.calls == []


def test_an_empty_name_is_refused(arm_poses):
    _, ctx = picking(arm_poses, remembered())

    assert fails(ctx, "  ").code is ErrorCode.REFUSED


def test_unreadable_settings_stop_the_pick_before_anything_moves(arm_poses, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    robot, ctx = picking(arm_poses, remembered())

    error = fails(ctx)

    assert error.code is ErrorCode.INTERNAL
    assert error.message == "Walkie's pick settings cannot be read, so it cannot pick."
    assert robot.calls == []
