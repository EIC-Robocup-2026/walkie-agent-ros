"""pick_object over FakeRobot with arm poses and a stand-in object memory (decision PK-4)."""

import math
import re
from dataclasses import replace

import pytest

from conftest import PROJECT_ROOT
from test_object_memory_atomics import WORLD, Remembers
from walkie_agent.arm.pick import HandPose, heading_from, load_pick_settings, plan_pick
from walkie_agent.arm.poses import load_arm_poses
from walkie_agent.arm.state import NOT_SAFE_SENTENCE, SAFE_SENTENCE
from walkie_agent.capabilities import Ctx, RecalledObject
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.errors import ErrorCode, GoalFailed, StepError
from walkie_agent.tools.atomic import grasp, pick_object

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
        # Seen a second ago: a sighting fresh enough to reach for after the drive.
        seen_s_ago=1.0,
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


def refusals(robot: FakeRobot) -> list[str | None]:
    """Each reach's error code; FakeRobot sends no goal for a reach whose check refused."""
    return [call["error"] for call in robot.calls if call["verb"] == "arms.reach"]


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


def hand_targets(monkeypatch, robot: FakeRobot) -> list[HandPose]:
    """The pose of every arms.reach call from now on, in order."""
    original = robot.reach
    poses: list[HandPose] = []

    def recording(arm, pose, **kwargs):
        poses.append(pose)
        return original(arm, pose, **kwargs)

    monkeypatch.setattr(robot, "reach", recording)
    return poses


def fails(ctx: Ctx, name: str = "water bottle") -> StepError:
    with pytest.raises(StepError) as raised:
        pick_object(ctx, name)
    return raised.value


def read_again(monkeypatch, ctx: Ctx, *answers: RecalledObject | StepError) -> list[str]:
    """Make each read of an object by its id give the next of `answers`, the last for good.

    A StepError is raised. Returns the ids read, in order.
    """
    reads: list[str] = []

    def again(object_id, **kwargs):
        reads.append(object_id)
        answer = answers[min(len(reads), len(answers)) - 1]
        if isinstance(answer, StepError):
            raise answer
        return answer

    monkeypatch.setattr(ctx.object_memory, "get", again)
    return reads


def after_call(monkeypatch, robot: FakeRobot, verb: str, nth: int, then) -> None:
    """Run then() right after the nth call of robot.<verb> returns, as if during that motion."""
    original = getattr(robot, verb)
    count = 0

    def changing(*args, **kwargs):
        nonlocal count
        count += 1
        result = original(*args, **kwargs)
        if count == nth:
            then()
        return result

    monkeypatch.setattr(robot, verb, changing)


def pick_settings(monkeypatch, **changes) -> None:
    """Pick with config/pick.toml's numbers but `changes`, and read the memory often."""
    settings = replace(load_pick_settings(PROJECT_ROOT / "config" / "pick.toml"), **changes)
    monkeypatch.setattr(grasp, "load_pick_settings", lambda: settings)
    monkeypatch.setattr(grasp, "SIGHTING_POLL_S", 0.01)


# The pick


def test_a_pick_drives_to_the_stand_off_reaches_closes_and_lifts(arm_poses):
    robot, ctx = picking(arm_poses, remembered())

    answer = pick_object(ctx, "water bottle")

    assert answer == (
        "Picked up the water bottle with the right hand, lifted it and brought both arms "
        f"home. {SAFE_SENTENCE}"
    )
    assert verbs(robot) == [
        "base.move_relative",
        "arms.move_to",
        "arms.gripper",
        "arms.reach",
        "arms.reach",
        "arms.gripper",
        "arms.reach",
        "arms.reach",
        "arms.move_to",
        "arms.move_to",
        "arms.restore_touch",
    ]
    assert robot.touch is False
    # 0.65 m back from the object and 0.20 m to the left of the right hand's line to
    # it, which is the line the robot stood on.
    move = next(call for call in robot.calls if call["verb"] == "base.move_relative")
    heading = math.asin(0.20)
    assert move["args"]["turn_deg"] == pytest.approx(math.degrees(heading))
    at_stand_off(robot, 1.0, 0.0)
    grippers = [call["args"]["state"] for call in robot.calls if call["verb"] == "arms.gripper"]
    assert grippers == ["open", "close"]
    # The whole-body reach to the pre-grasp pose, then the straight approach, lift and
    # move clear of the surface, on which the fingers may touch the octomap.
    assert reaches(robot) == [
        (True, False, False, 90.0),
        (False, True, True, 30.0),
        (False, True, True, 30.0),
        (False, True, True, 30.0),
    ]
    # The left arm goes up out of the way once the base stands still; the right picks.
    # Then the right arm goes home with the lift raised to its home height, keeping
    # the fingers' allowance, and the left arm goes home, which ends it.
    moves = [call["args"] for call in robot.calls if call["verb"] == "arms.move_to"]
    assert moves == [
        {"pose": "standby", "arm": "left"},
        {"pose": "home", "arm": "right", "lift_m": 0.7435, "touch": True},
        {"pose": "home", "arm": "left"},
    ]
    arms = {
        call["args"]["arm"]
        for call in robot.calls
        if call["verb"].startswith("arms.")
        and call["verb"] not in ("arms.move_to", "arms.restore_touch")
    }
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
    # Read again after the drive, before the reach and before the approach.
    assert ctx.object_memory.calls == [("search", "water bottle", 10)] + [("get", "7")] * 3


def test_the_reach_goes_to_where_the_memory_holds_the_object_after_the_drive(
    arm_poses, monkeypatch
):
    robot, ctx = picking(arm_poses, remembered())
    original = ctx.object_memory.get

    def moved(object_id, **kwargs):
        # Seen again from the stand-off pose: 3 cm further and 5 cm lower.
        found = original(object_id, **kwargs)
        return replace(found, x=found.x + 0.03, z=found.z - 0.05)

    monkeypatch.setattr(ctx.object_memory, "get", moved)
    targets = hand_targets(monkeypatch, robot)

    pick_object(ctx, "water bottle")

    here = robot.pose()
    plan = plan_pick(
        load_pick_settings(), 1.03, 0.0, 0.70, here.yaw, math.atan2(-here.y, 1.03 - here.x)
    )
    assert targets[1] == plan.grasp


# Seeing the object again after the drive


@pytest.mark.parametrize(
    ("answer", "code", "words"),
    [
        (
            StepError(ErrorCode.NOT_FOUND, "Walkie remembers no object 7."),
            ErrorCode.NOT_FOUND,
            "Walkie's memory no longer holds the water bottle, so it did not reach for it. "
            "Call find_objects to look for it again.",
        ),
        (
            replace(remembered(), state="held"),
            ErrorCode.PRECONDITION_FAILED,
            "Walkie's memory now holds the water bottle as held, so it did not reach for it. "
            "Call find_objects to look again.",
        ),
        (
            replace(remembered(), state="missing"),
            ErrorCode.PRECONDITION_FAILED,
            "Walkie's memory now holds the water bottle as missing, so it did not reach for "
            "it. Call find_objects to look again.",
        ),
        (
            replace(remembered(), z=None),
            ErrorCode.PRECONDITION_FAILED,
            "Walkie's memory no longer holds exactly where the water bottle is, so it did not "
            "reach for it. Call find_objects to look again.",
        ),
    ],
    ids=["forgotten", "held", "missing", "no-position"],
)
def test_an_object_gone_held_or_without_a_position_after_the_drive_is_not_reached_for(
    arm_poses, monkeypatch, answer, code, words
):
    robot, ctx = picking(arm_poses, remembered())
    reads = read_again(monkeypatch, ctx, answer)

    error = fails(ctx)

    assert error.code is code
    assert error.message == f"{words} {SAFE_SENTENCE}"
    assert reads == ["7"]
    assert verbs(robot) == ["base.move_relative"]
    assert robot.touch is False


@pytest.mark.parametrize("code", [ErrorCode.UNAVAILABLE, ErrorCode.TIMEOUT, ErrorCode.INTERNAL])
def test_a_memory_that_cannot_be_read_after_the_drive_sends_no_reach_and_may_be_retried(
    arm_poses, monkeypatch, code
):
    robot, ctx = picking(arm_poses, remembered())
    reads = read_again(monkeypatch, ctx, StepError(code, "The object memory did not answer."))

    error = fails(ctx)

    # Never NOT_FOUND, which would say the object is gone, and never a reach at the
    # position read before the drive.
    assert error.code is ErrorCode.UNAVAILABLE
    assert error.message == (
        "Walkie could not read its object memory after driving up to the water bottle, so "
        f"it did not reach for it. Call pick_object again. {SAFE_SENTENCE}"
    )
    assert reads == ["7"]
    assert verbs(robot) == ["base.move_relative"]


def test_a_stop_during_the_read_after_the_drive_is_raised_as_it_is(arm_poses, monkeypatch):
    robot, ctx = picking(arm_poses, remembered())
    stopped = StepError(ErrorCode.CANCELLED, "Cancelled the read of object 7: a stop is in effect.")
    read_again(monkeypatch, ctx, stopped)

    assert fails(ctx) is stopped
    assert verbs(robot) == ["base.move_relative"]


@pytest.mark.parametrize("seen_s_ago", [30.0, None], ids=["old", "unknown"])
def test_an_object_not_seen_lately_is_waited_for_then_refused_without_a_reach(
    arm_poses, monkeypatch, seen_s_ago
):
    pick_settings(monkeypatch, observation_wait_s=0.2)
    robot, ctx = picking(arm_poses, remembered())
    reads = read_again(monkeypatch, ctx, replace(remembered(), seen_s_ago=seen_s_ago))

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        "Walkie stands by the water bottle but has not seen it in the last few seconds, so "
        "it did not reach for it: it may have moved or be out of view. Call find_objects to "
        f"look again. {SAFE_SENTENCE}"
    )
    assert len(reads) > 1, "the memory is read again while the atomic waits"
    assert verbs(robot) == ["base.move_relative"]


def test_a_new_sighting_during_the_wait_is_the_one_reached_for(arm_poses, monkeypatch):
    pick_settings(monkeypatch, observation_wait_s=5.0)
    robot, ctx = picking(arm_poses, remembered())
    reads = read_again(
        monkeypatch,
        ctx,
        replace(remembered(), seen_s_ago=8.0),
        replace(remembered(), seen_s_ago=8.5),
        # Exactly max_observation_age_s old still counts as fresh.
        replace(remembered(), x=1.02, seen_s_ago=5.0),
    )
    targets = hand_targets(monkeypatch, robot)

    pick_object(ctx, "water bottle")

    # Three reads after the drive, then one before the reach and one before the approach.
    assert len(reads) == 5
    here = robot.pose()
    plan = plan_pick(
        load_pick_settings(), 1.02, 0.0, 0.75, here.yaw, math.atan2(-here.y, 1.02 - here.x)
    )
    assert targets[1] == plan.grasp


def test_an_object_that_moved_too_far_during_the_drive_asks_for_a_new_pick(
    arm_poses, monkeypatch
):
    robot, ctx = picking(arm_poses, remembered())
    # 0.16 m to the side of where the drive aimed, past max_target_shift_m.
    read_again(monkeypatch, ctx, replace(remembered(), y=-0.16))

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        "The water bottle moved while Walkie drove up to it, so it did not reach for it. "
        f"Call pick_object again to approach it where it is now. {SAFE_SENTENCE}"
    )
    assert verbs(robot) == ["base.move_relative"]


# Seeing the object again before the reach and before the approach

HOME_FIRST = "Bring both arms home with move_arm_to_pose, then"


def changed(ctx: Ctx, **changes) -> None:
    """The memory now holds object 7 with `changes`, or no longer holds it with none."""
    memory = ctx.object_memory
    if changes:
        memory.objects["7"] = replace(memory.objects["7"], **changes)
    else:
        del memory.objects["7"]


@pytest.mark.parametrize(
    ("changes", "code", "words"),
    [
        (
            {},
            ErrorCode.NOT_FOUND,
            "Walkie's memory no longer holds the water bottle, so it did not reach for it. "
            f"{HOME_FIRST} call find_objects to look for it again.",
        ),
        (
            {"state": "held"},
            ErrorCode.PRECONDITION_FAILED,
            "Walkie's memory now holds the water bottle as held, so it did not reach for it. "
            f"{HOME_FIRST} call find_objects to look again.",
        ),
        (
            {"y": -0.16},
            ErrorCode.PRECONDITION_FAILED,
            "The water bottle moved while Walkie readied its arms, so it did not reach for it. "
            f"{HOME_FIRST} call pick_object again to approach it where it is now.",
        ),
    ],
    ids=["vanished", "held", "moved"],
)
def test_an_object_gone_or_moved_while_the_other_arm_moves_gets_no_reach(
    arm_poses, monkeypatch, changes, code, words
):
    robot, ctx = picking(arm_poses, remembered())
    # Taken or moved while the left arm goes to standby.
    after_call(monkeypatch, robot, "move_to", 1, lambda: changed(ctx, **changes))

    error = fails(ctx)

    assert error.code is code
    assert error.message == f"{words} {NOT_SAFE_SENTENCE}"
    # No pre-grasp goal, nor any other motion after the open.
    assert reaches(robot) == []
    assert verbs(robot) == [
        "base.move_relative",
        "arms.move_to",
        "arms.gripper",
        "arms.restore_touch",
    ]
    assert robot.touch is False


def test_a_sighting_that_ages_past_the_limit_before_the_reach_gets_no_reach(
    arm_poses, monkeypatch
):
    pick_settings(monkeypatch, observation_wait_s=0.2)
    robot, ctx = picking(arm_poses, remembered())
    # Fresh after the drive; while the left arm moves, nothing sees the object again.
    after_call(monkeypatch, robot, "move_to", 1, lambda: changed(ctx, seen_s_ago=5.5))

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        "Walkie stands by the water bottle but has not seen it in the last few seconds, so "
        f"it did not reach for it: it may have moved or be out of view. {HOME_FIRST} call "
        f"find_objects to look again. {NOT_SAFE_SENTENCE}"
    )
    # After the drive once, then again and again while the atomic waited.
    assert ctx.object_memory.calls.count(("get", "7")) > 2
    assert reaches(robot) == []


def test_a_sighting_during_the_wait_before_the_reach_is_reached_for(arm_poses, monkeypatch):
    pick_settings(monkeypatch, observation_wait_s=5.0)
    robot, ctx = picking(arm_poses, remembered())
    reads = read_again(
        monkeypatch,
        ctx,
        remembered(),
        replace(remembered(), seen_s_ago=6.0),
        # A new sighting, 4 cm further than after the drive.
        replace(remembered(), x=1.04, seen_s_ago=0.5),
    )
    targets = hand_targets(monkeypatch, robot)

    pick_object(ctx, "water bottle")

    # One read after the drive, two before the reach and one before the approach.
    assert len(reads) == 4
    here = robot.pose()
    plan = plan_pick(
        load_pick_settings(), 1.04, 0.0, 0.75, here.yaw, math.atan2(-here.y, 1.04 - here.x)
    )
    assert targets[:2] == [plan.pregrasp, plan.grasp]


def test_an_object_lowered_out_of_the_envelope_before_the_reach_gets_no_reach(
    arm_poses, monkeypatch
):
    # 0.60 m high after the drive, then seen 0.50 m high, under the envelope's 0.55 m.
    robot, ctx = picking(arm_poses, remembered(z=0.60))
    after_call(monkeypatch, robot, "move_to", 1, lambda: changed(ctx, z=0.50))

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        "The water bottle is too low for the right hand to pick up, so Walkie did not reach "
        f"for it. Bring both arms home with move_arm_to_pose. {NOT_SAFE_SENTENCE}"
    )
    assert reaches(robot) == []


@pytest.mark.parametrize(
    ("changes", "code", "words"),
    [
        (
            {},
            ErrorCode.NOT_FOUND,
            "Walkie's memory no longer holds the water bottle, so it did not move the hand in "
            f"to grasp it. {HOME_FIRST} call find_objects to look for it again.",
        ),
        (
            {"state": "held"},
            ErrorCode.PRECONDITION_FAILED,
            "Walkie's memory now holds the water bottle as held, so it did not move the hand "
            f"in to grasp it. {HOME_FIRST} call find_objects to look again.",
        ),
        (
            {"state": "missing"},
            ErrorCode.PRECONDITION_FAILED,
            "Walkie's memory now holds the water bottle as missing, so it did not move the "
            f"hand in to grasp it. {HOME_FIRST} call find_objects to look again.",
        ),
        (
            {"x": 1.16},
            ErrorCode.PRECONDITION_FAILED,
            "The water bottle moved while Walkie reached toward it, so it did not move the "
            f"hand in to grasp it. {HOME_FIRST} call pick_object again to approach it where "
            "it is now.",
        ),
    ],
    ids=["vanished", "held", "missing", "moved"],
)
def test_an_object_gone_or_moved_during_the_reach_gets_no_approach(
    arm_poses, monkeypatch, changes, code, words
):
    robot, ctx = picking(arm_poses, remembered())
    # Taken or moved while the whole-body reach brings the hand to the pre-grasp pose.
    after_call(monkeypatch, robot, "reach", 1, lambda: changed(ctx, **changes))

    error = fails(ctx)

    assert error.code is code
    assert error.message == f"{words} {NOT_SAFE_SENTENCE}"
    # The pre-grasp reach, then the approach refused by its check before its goal: no
    # close or any other motion after it.
    assert reaches(robot) == [(True, False, False, 90.0), (False, True, True, 30.0)]
    assert refusals(robot) == [None, str(code)]
    assert verbs(robot)[-2:] == ["arms.reach", "arms.restore_touch"]
    assert robot.touch is False


def test_a_sighting_that_ages_past_the_limit_during_the_reach_gets_no_approach(
    arm_poses, monkeypatch
):
    pick_settings(monkeypatch, observation_wait_s=0.2)
    robot, ctx = picking(arm_poses, remembered())
    after_call(monkeypatch, robot, "reach", 1, lambda: changed(ctx, seen_s_ago=30.0))

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.message.startswith(
        "Walkie stands by the water bottle but has not seen it in the last few seconds, so "
        "it did not move the hand in to grasp it"
    )
    assert reaches(robot) == [(True, False, False, 90.0), (False, True, True, 30.0)]
    assert refusals(robot) == [None, str(ErrorCode.PRECONDITION_FAILED)]


def test_the_approach_keeps_the_plan_of_the_reach_when_the_object_shifts_a_little(
    arm_poses, monkeypatch
):
    robot, ctx = picking(arm_poses, remembered())
    # Seen 5 cm further while the hand reached: within max_target_shift_m.
    after_call(monkeypatch, robot, "reach", 1, lambda: changed(ctx, x=1.05))
    targets = hand_targets(monkeypatch, robot)

    pick_object(ctx, "water bottle")

    # The straight approach goes from the pre-grasp pose along the reach's own line.
    here = robot.pose()
    plan = plan_pick(
        load_pick_settings(), 1.0, 0.0, 0.75, here.yaw, math.atan2(-here.y, 1.0 - here.x)
    )
    assert targets[:2] == [plan.pregrasp, plan.grasp]


# The reach envelope


def test_an_object_the_memory_lowered_below_the_envelope_is_not_reached_for(
    arm_poses, monkeypatch
):
    # The drive aimed at a centre 0.60 m high; seen from close by, it is 0.48 m.
    robot, ctx = picking(arm_poses, remembered(z=0.60))
    read_again(monkeypatch, ctx, replace(remembered(), z=0.48))

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        "The water bottle is too low for the right hand to pick up, so Walkie did not reach "
        f"for it. {SAFE_SENTENCE}"
    )
    assert verbs(robot) == ["base.move_relative"]


def test_an_object_too_high_from_the_start_is_driven_to_but_not_reached_for(arm_poses):
    robot, ctx = picking(arm_poses, remembered(z=1.40))

    error = fails(ctx)

    assert error.message.startswith("The water bottle is too high for the right hand")
    assert verbs(robot) == ["base.move_relative"]


def test_an_object_the_memory_moved_out_of_reach_asks_for_a_new_pick(arm_poses, monkeypatch):
    # An envelope that ends 0.75 m from the base; the object, 0.68 m from the stand-off
    # pose, is seen 0.10 m further away after the drive.
    pick_settings(monkeypatch, max_reach_m=0.75)
    robot, ctx = picking(arm_poses, remembered())
    read_again(monkeypatch, ctx, replace(remembered(), x=1.10))

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        "The water bottle is out of the right hand's reach from where Walkie stopped, so it "
        f"did not reach for it. Call pick_object again to approach it afresh. {SAFE_SENTENCE}"
    )
    assert verbs(robot) == ["base.move_relative"]


def test_a_robot_already_at_the_stand_off_does_not_drive(arm_poses):
    robot, ctx = picking(arm_poses, remembered(x=0.65, y=-0.20))

    pick_object(ctx, "water bottle")

    assert "base.move_relative" not in verbs(robot)
    assert verbs(robot)[:2] == ["arms.move_to", "arms.gripper"]


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


@pytest.mark.parametrize(("arrives", "goes_on"), [(True, True), (False, False)])
def test_a_failed_drive_to_the_place_goes_on_only_when_it_ended_near_enough(
    arm_poses, monkeypatch, arrives, goes_on
):
    robot, ctx = picking(arm_poses, remembered(x=9.0, y=2.0, room="living room"))
    original = robot.go_to
    aborted = GoalFailed(ErrorCode.SKILL_FAILED, ABORTED)

    def short(place, **kwargs):
        # Nav2 aborts the drive, either at the sofa or before the base moved.
        if arrives:
            original(place, **kwargs)
        raise aborted

    monkeypatch.setattr(robot, "go_to", short)

    if goes_on:
        pick_object(ctx, "water bottle")
        at_stand_off(robot, 9.0, 2.0)
    else:
        assert fails(ctx) is aborted
        # The stand-in drive is not logged, and nothing else moved.
        assert verbs(robot) == []


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


def test_the_hand_moves_in_along_the_line_from_where_the_base_stopped(arm_poses, monkeypatch):
    robot, ctx = picking(arm_poses, remembered())
    drive, reach = robot.move_relative, robot.reach
    targets = []

    def overshoot(forward_m=0.0, left_m=0.0, turn_deg=0.0, **kwargs):
        # The base stops 0.10 m past its goal, as Nav2 did in the Chula simulation.
        return drive(forward_m + 0.10, left_m, turn_deg, **kwargs)

    def record(arm, pose, **kwargs):
        targets.append(pose)
        return reach(arm, pose, **kwargs)

    monkeypatch.setattr(robot, "move_relative", overshoot)
    monkeypatch.setattr(robot, "reach", record)

    pick_object(ctx, "water bottle")

    here = robot.pose()
    approach = math.atan2(0.0 - here.y, 1.0 - here.x)
    pregrasp, grasp, lift, clear = targets
    assert grasp.x == pytest.approx(1.0 - 0.06 * math.cos(approach))
    assert grasp.y == pytest.approx(-0.06 * math.sin(approach))
    assert pregrasp.x == pytest.approx(1.0 - 0.16 * math.cos(approach))
    assert pregrasp.y == pytest.approx(-0.16 * math.sin(approach))
    for pose in targets:
        assert pose.yaw == pytest.approx(approach + math.pi)
    assert (lift.x, lift.y) == (grasp.x, grasp.y)
    # 0.30 m ahead of where the base stopped and 0.35 m to its right.
    cos, sin = math.cos(here.yaw), math.sin(here.yaw)
    assert clear.x == pytest.approx(here.x + 0.30 * cos + 0.35 * sin)
    assert clear.y == pytest.approx(here.y + 0.30 * sin - 0.35 * cos)
    assert clear.z == lift.z


@pytest.mark.parametrize(("short_m", "tries"), [(0.04, 1), (0.30, 2)], ids=["near", "far"])
def test_a_failed_drive_that_stopped_near_its_stand_off_goes_on_from_there(
    arm_poses, monkeypatch, short_m, tries
):
    robot, ctx = picking(arm_poses, remembered())
    drive = robot.move_relative
    calls = []

    def stops_short(forward_m=0.0, left_m=0.0, turn_deg=0.0, **kwargs):
        # The first drive stops short of its goal and Nav2 aborts it, as beside the
        # Chula bar; any later one arrives.
        calls.append(forward_m)
        if len(calls) > 1:
            return drive(forward_m, left_m, turn_deg, **kwargs)
        scale = 1.0 - short_m / math.hypot(forward_m, left_m)
        drive(forward_m * scale, left_m * scale, turn_deg, **kwargs)
        raise GoalFailed(ErrorCode.SKILL_FAILED, ABORTED)

    monkeypatch.setattr(robot, "move_relative", stops_short)

    answer = pick_object(ctx, "water bottle")

    assert answer.startswith("Picked up the water bottle")
    assert len(calls) == tries


def test_a_pick_that_reaches_no_stand_off_says_how_many_tries(arm_poses):
    robot, ctx = picking(arm_poses, remembered())
    robot.faults["base.move_relative"] = GoalFailed(ErrorCode.SKILL_FAILED, ABORTED)

    error = fails(ctx)

    assert error.code is ErrorCode.SKILL_FAILED
    assert error.message == (
        "Walkie could not reach a spot to pick up the water bottle from, after 8 tries. "
        "The last one: Nav2 aborted the goal."
    )
    assert verbs(robot) == ["base.move_relative"] * 8


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


def test_a_close_on_nothing_does_not_lift_and_brings_the_arms_home(arm_poses):
    robot, ctx = picking(arm_poses, remembered(), holds=False)

    error = fails(ctx)

    assert error.code is ErrorCode.SKILL_FAILED
    assert error.message == (
        "Walkie reached the water bottle, but the right hand closed on nothing, so it did "
        "not pick it up. It brought both arms home. The object may have moved: call "
        f"find_objects to look again. {SAFE_SENTENCE}"
    )
    assert len(reaches(robot)) == 2
    assert verbs(robot)[-3:] == ["arms.move_to", "arms.move_to", "arms.restore_touch"]


def test_a_close_on_nothing_whose_move_home_fails_says_both(arm_poses, monkeypatch):
    robot, ctx = picking(arm_poses, remembered(), holds=False)
    blocked = GoalFailed(ErrorCode.SKILL_FAILED, f"MoveIt found no path. {NOT_SAFE_SENTENCE}")
    fail_on_call(monkeypatch, robot, "move_to", 2, blocked)

    error = fails(ctx)

    assert isinstance(error, GoalFailed)
    assert error.message == (
        "Walkie reached the water bottle, but the right hand closed on nothing, so it did "
        "not pick it up, and the move of the arms home failed: MoveIt found no path. "
        f"{NOT_SAFE_SENTENCE}"
    )


def test_a_failed_move_home_after_the_lift_says_the_hand_holds_the_object(
    arm_poses, monkeypatch
):
    robot, ctx = picking(arm_poses, remembered())
    blocked = GoalFailed(ErrorCode.SKILL_FAILED, f"MoveIt found no path. {NOT_SAFE_SENTENCE}")
    fail_on_call(monkeypatch, robot, "move_to", 3, blocked)

    error = fails(ctx)

    assert isinstance(error, GoalFailed)
    assert error.message == (
        "The right hand holds the water bottle, but the move of the arms home failed: "
        f"MoveIt found no path. {NOT_SAFE_SENTENCE}"
    )


def test_a_left_arm_that_cannot_move_out_of_the_way_stops_the_pick(arm_poses):
    robot, ctx = picking(arm_poses, remembered())
    robot.faults["arms.move_to"] = StepError(
        ErrorCode.SKILL_FAILED, f"MoveIt found no path. {SAFE_SENTENCE}"
    )

    error = fails(ctx)

    assert error.code is ErrorCode.SKILL_FAILED
    assert error.message == (
        "Walkie did not reach for the water bottle: it could not move the left arm out of "
        f"the way first. MoveIt found no path. {SAFE_SENTENCE}"
    )
    assert reaches(robot) == []
    assert "arms.gripper" not in verbs(robot)


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


def test_a_failed_move_clear_of_the_surface_says_the_hand_holds_the_object_and_stays(
    arm_poses, monkeypatch
):
    robot, ctx = picking(arm_poses, remembered())
    stopped = GoalFailed(ErrorCode.SKILL_FAILED, f"The commander aborted it. {NOT_SAFE_SENTENCE}")
    fail_on_call(monkeypatch, robot, "reach", 4, stopped)

    error = fails(ctx)

    assert isinstance(error, GoalFailed)
    assert error.message == (
        "The right hand holds the water bottle, but the move clear of the surface failed: The "
        f"commander aborted it. {NOT_SAFE_SENTENCE}"
    )
    # The move home would swing the held object over the surface, so it is not tried.
    moves = [call["args"] for call in robot.calls if call["verb"] == "arms.move_to"]
    assert moves == [{"pose": "standby", "arm": "left"}]


# The finger allowance after the pick


@pytest.mark.parametrize(("nth", "step"), [(2, "approach"), (3, "lift"), (4, "clear")])
def test_a_failed_step_with_the_allowance_on_turns_it_off_and_moves_nothing_after(
    arm_poses, monkeypatch, nth, step
):
    robot, ctx = picking(arm_poses, remembered())
    original = robot.reach

    def fails_with_the_allowance_on(arm, pose, **kwargs):
        if len(reaches(robot)) == nth - 1:
            # The commander had turned the allowance on for this straight move.
            robot.touch = True
            raise GoalFailed(ErrorCode.SKILL_FAILED, f"It aborted. {NOT_SAFE_SENTENCE}")
        return original(arm, pose, **kwargs)

    monkeypatch.setattr(robot, "reach", fails_with_the_allowance_on)

    error = fails(ctx)

    assert isinstance(error, GoalFailed), step
    assert robot.touch is False
    assert verbs(robot)[-1] == "arms.restore_touch"
    assert verbs(robot).count("arms.restore_touch") == 1


def test_a_stop_mid_grasp_turns_the_allowance_off_with_no_motion_after(arm_poses, monkeypatch):
    robot, ctx = picking(arm_poses, remembered())
    original = robot.reach

    def stopped_during_the_lift(arm, pose, **kwargs):
        if len(reaches(robot)) == 2:
            robot.cancel.cancel()
        return original(arm, pose, **kwargs)

    monkeypatch.setattr(robot, "reach", stopped_during_the_lift)

    error = fails(ctx)

    assert error.code is ErrorCode.CANCELLED
    # The approach left the allowance on; the reset goes through the stop.
    assert robot.touch is False
    # No move home or any other motion after the stopped lift.
    assert verbs(robot) == [
        "base.move_relative",
        "arms.move_to",
        "arms.gripper",
        "arms.reach",
        "arms.reach",
        "arms.gripper",
        "arms.reach",
        "arms.restore_touch",
    ]


def test_an_unexpected_error_mid_grasp_turns_the_allowance_off_and_is_raised(
    arm_poses, monkeypatch
):
    robot, ctx = picking(arm_poses, remembered())
    fail_on_call(monkeypatch, robot, "gripper", 2, RuntimeError("the fake broke"))
    robot.touch = True

    with pytest.raises(RuntimeError, match="the fake broke"):
        pick_object(ctx, "water bottle")

    assert robot.touch is False
    assert verbs(robot)[-1] == "arms.restore_touch"


RESET_FAILED = StepError(
    ErrorCode.TIMEOUT, "The commander's /bimanual_commander/set_parameters did not answer."
)
RESET_NOTE = (
    "Walkie could not turn off its fingers' allowance to touch what its cameras sense: The "
    "commander's /bimanual_commander/set_parameters did not answer. Its next motion of an arm "
    "or a gripper sets the allowance again first, and moves only if that works."
)


def test_a_failed_reset_after_a_pick_fails_the_pick_with_the_answer_kept(arm_poses):
    robot, ctx = picking(arm_poses, remembered())
    robot.faults["arms.restore_touch"] = RESET_FAILED

    error = fails(ctx)

    assert type(error) is StepError and error.code is ErrorCode.TIMEOUT
    assert error.message == (
        f"{RESET_NOTE} Picked up the water bottle with the right hand, lifted it and brought "
        f"both arms home. {SAFE_SENTENCE}"
    )
    assert verbs(robot)[-1] == "arms.restore_touch"


def test_a_failed_reset_after_a_failed_step_leads_its_message_and_keeps_its_code(
    arm_poses, monkeypatch
):
    robot, ctx = picking(arm_poses, remembered())
    stopped = GoalFailed(ErrorCode.SKILL_FAILED, f"The commander aborted it. {NOT_SAFE_SENTENCE}")
    fail_on_call(monkeypatch, robot, "reach", 3, stopped)
    robot.faults["arms.restore_touch"] = RESET_FAILED

    error = fails(ctx)

    assert isinstance(error, GoalFailed) and error.code is ErrorCode.SKILL_FAILED
    assert error.message == (
        f"{RESET_NOTE} The right hand holds the water bottle, but the lift failed: The "
        f"commander aborted it. {NOT_SAFE_SENTENCE}"
    )
    assert verbs(robot)[-1] == "arms.restore_touch"


def test_a_failed_reset_after_an_unexpected_error_is_a_note_on_it(arm_poses, monkeypatch):
    robot, ctx = picking(arm_poses, remembered())
    fail_on_call(monkeypatch, robot, "gripper", 2, RuntimeError("the fake broke"))
    robot.faults["arms.restore_touch"] = RESET_FAILED

    with pytest.raises(RuntimeError, match="the fake broke") as raised:
        pick_object(ctx, "water bottle")

    assert raised.value.__notes__ == [RESET_NOTE]


def test_a_refusal_before_the_grasp_sends_no_reset(arm_poses, monkeypatch):
    robot, ctx = picking(arm_poses, remembered())
    read_again(monkeypatch, ctx, replace(remembered(), state="held"))

    fails(ctx)

    # Nothing of this pick turned the allowance on.
    assert "arms.restore_touch" not in verbs(robot)


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
