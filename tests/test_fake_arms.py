"""FakeRobot's arms: the named poses, the grippers and the drive check, with no ROS."""

import dataclasses
import inspect
import json
import math
import time

import pytest

from conftest import PROJECT_ROOT
from walkie_agent.arm import answers
from walkie_agent.arm.pick import HandPose
from walkie_agent.arm.poses import load_arm_poses
from walkie_agent.arm.results import GripperResult
from walkie_agent.arm.state import (
    ARMS_OUT,
    NOT_SAFE_SENTENCE,
    SAFE_SENTENCE,
    UNKNOWN_SENTENCE,
    ArmsState,
    ArmState,
)
from walkie_agent.capabilities import (
    NO_ARMS,
    Arms,
    ArmsMoved,
    ArmsReport,
    GripperMoved,
    GripperReading,
)
from walkie_agent.capabilities.fake import ARMS_START_POSE, FakeRobot
from walkie_agent.errors import ErrorCode, StepError

HOME = ArmState("home", True)
HANDS_UP = ArmState("hands_up", False)
AT_HOME = ArmsReport(ArmsState(left=HOME, right=HOME))
ARM_VERBS = ("poses", "state", "read", "move_to", "gripper", "reach", "restore_touch")
# A hand target as pick_object computes it, in the map.
HAND = HandPose(0.6, -0.2, 0.8, 0.0, math.pi / 2, 0.3, "map")


def world() -> dict:
    return {
        "places": {
            "lobby": {"x": 0.0, "y": 0.0, "yaw": 0.0, "room": "lobby"},
            "desk": {"x": 1.0, "y": 0.0, "yaw": 0.0, "room": "lobby"},
            "den": {"x": 10.0, "y": 0.0, "yaw": 0.0, "room": "den"},
            "shelf": {"x": 11.0, "y": 0.0, "yaw": 1.5, "room": "den"},
        },
        "objects": {"apple_1": {"name": "apple", "category": "fruit", "at": "shelf"}},
        "people": {
            "ann": {
                "name": "Ann",
                "room": "lobby",
                "pose": "sitting person",
                "gesture": "waving person",
            },
        },
        "start": "lobby",
    }


@pytest.fixture(scope="module")
def arm_poses():
    return load_arm_poses(
        PROJECT_ROOT / "config" / "arm_poses.toml", PROJECT_ROOT / "config" / "robot_model.toml"
    )


@pytest.fixture
def robot(arm_poses) -> FakeRobot:
    return FakeRobot(world(), arm_poses=arm_poses)


@pytest.fixture
def armless() -> FakeRobot:
    return FakeRobot(world())


def raised(call, *args, **kwargs) -> StepError:
    with pytest.raises(StepError) as caught:
        call(*args, **kwargs)
    return caught.value


def verbs(robot: FakeRobot) -> list[str]:
    return [call["verb"] for call in robot.calls]


def refusal(what: str, problem: str) -> str:
    return (
        f"Not starting {what}: {problem} with move_arm_to_pose first, for example "
        f"{answers.SAFE_EXAMPLE!r}."
    )


# Without arm poses, as in Mahoraga


@pytest.mark.parametrize(
    ("verb", "args"),
    [
        ("poses", ()),
        ("read", ()),
        ("move_to", ("home",)),
        ("gripper", ("left", "open")),
        ("reach", ("right", HAND)),
    ],
)
def test_without_arm_poses_every_arm_verb_but_state_is_unavailable(armless, verb, args):
    error = raised(getattr(armless, verb), *args)

    assert (error.code, error.message) == (ErrorCode.UNAVAILABLE, NO_ARMS)
    call = armless.calls[-1]
    assert (call["verb"], call["error"]) == (f"arms.{verb}", "UNAVAILABLE")


def test_without_arm_poses_the_arms_are_unknown_and_no_drive_is_checked(armless):
    assert armless.state() == ArmsReport(ArmsState())
    assert armless.state(read=True).sentence() == UNKNOWN_SENTENCE

    armless.go_to("shelf")
    armless.move_relative(forward_m=0.5)

    assert verbs(armless) == ["base.go_to", "base.move_relative"]
    assert [call["error"] for call in armless.calls] == [None, None]


def test_without_arm_poses_unavailable_wins_over_a_stop(armless):
    armless.cancel.cancel()

    assert raised(armless.move_to, "home").code is ErrorCode.UNAVAILABLE


# Poses and state


def test_the_simulation_poses_are_offered_in_file_order(robot, arm_poses):
    offered = robot.poses()

    assert offered == arm_poses.offered_poses(sim=True)
    assert list(offered) == ["home", "hands_up", "standby", "tray", "receive", "carry", "point"]
    assert robot.calls[-1]["verb"] == "arms.poses"


def test_both_arms_start_home_with_both_grippers_open(robot, arm_poses):
    open_m = arm_poses.grippers.open_position

    assert ARMS_START_POSE == answers.SAFE_EXAMPLE == "home"
    assert robot.state() == AT_HOME
    assert robot.state().travel_safe is True
    assert robot.state().sentence() == SAFE_SENTENCE
    reading = robot.read()
    assert reading.before == reading.after == AT_HOME
    assert reading.grippers == (
        GripperReading("left", open_m, "open", in_grasp_band=False),
        GripperReading("right", open_m, "open", in_grasp_band=False),
    )


def test_its_ctx_holds_the_robot_as_the_arms(robot):
    assert robot.ctx.arms is robot


def test_the_arm_verbs_take_the_protocol_s_parameters():
    for name in ARM_VERBS:
        assert inspect.signature(getattr(FakeRobot, name)) == inspect.signature(
            getattr(Arms, name)
        ), name


# Moves


def test_a_move_of_both_arms_lands_exactly_and_reports_the_arms_after(robot):
    moved = robot.move_to("Hands up")

    assert moved == ArmsMoved(
        pose="hands_up",
        arm="both",
        status="Success",
        max_error_rad=0.0,
        report=ArmsReport(ArmsState(left=HANDS_UP, right=HANDS_UP)),
    )
    assert moved.report.sentence() == NOT_SAFE_SENTENCE
    assert robot.state() == moved.report
    assert robot.move_to("home").report == AT_HOME


def test_a_move_of_one_arm_leaves_the_other_where_it_was(robot):
    moved = robot.move_to("receive", "left")

    assert moved.arm == "left"
    assert moved.report.arms == ArmsState(left=ArmState("receive", False), right=HOME)
    assert moved.report.arms.unsafe_arms == ("left",)
    assert robot.move_to("carry", "right").report.arms.right == ArmState("carry", True)
    # Each arm in carry, one at a time: both travel-safe, though not one both_arms goal.
    assert robot.move_to("carry", "left").report.travel_safe is True


@pytest.mark.parametrize("pose", ["wave", "pre_place", "basket"])
def test_an_unknown_or_hidden_pose_is_not_found_and_ends_with_the_sentence(robot, pose):
    error = raised(robot.move_to, pose)

    assert error.code is ErrorCode.NOT_FOUND
    assert error.message == (
        f"No pose named {pose!r}. The poses are: home, hands_up, standby, tray, receive, "
        f"carry, point. {SAFE_SENTENCE}"
    )
    assert robot.state() == AT_HOME
    assert robot.calls[-1]["error"] == "NOT_FOUND"


def test_an_arm_the_pose_does_not_take_is_refused_with_the_sentence(robot):
    error = raised(robot.move_to, "receive", "right")
    assert (error.code, error.message) == (
        ErrorCode.REFUSED,
        f"'receive' is for the left arm only. {SAFE_SENTENCE}",
    )
    robot.move_to("hands_up")

    error = raised(robot.move_to, "carry", "both")

    assert (error.code, error.message) == (
        ErrorCode.REFUSED,
        f"'carry' is for one arm at a time: left or right. {NOT_SAFE_SENTENCE}",
    )
    assert robot.state().arms == ArmsState(left=HANDS_UP, right=HANDS_UP)


# Grippers


@pytest.mark.parametrize("state", ["open", "half"])
def test_an_open_or_half_reports_only_that_it_moved(robot, state):
    moved = robot.gripper("left", state)

    assert moved == GripperMoved(
        arm="left",
        state=state,
        outcome="moved",
        result=GripperResult(
            success=True, status="Success", grasped=None, gripper_gap=None, command=1
        ),
    )
    assert (moved.holding, moved.finger_m, moved.read_error) == (None, None, None)


def test_a_close_closes_empty_at_the_closed_position(robot, arm_poses):
    closed_m = arm_poses.grippers.closed_position

    moved = robot.gripper("right", "close")

    assert moved == GripperMoved(
        arm="right",
        state="close",
        outcome="empty",
        result=GripperResult(
            success=True, status="Success", grasped=False, gripper_gap=closed_m, command=1
        ),
        finger_m=closed_m,
    )
    assert moved.holding is False


def test_a_close_of_a_grasping_gripper_holds_inside_the_grasp_band(robot):
    robot.grasping.add("right")

    moved = robot.gripper("right", "close")

    assert (moved.outcome, moved.holding) == ("holding", True)
    assert answers.reads_as(robot._require_arms().grippers, moved.finger_m) == "close"
    assert 0.005 <= moved.finger_m <= 0.044
    # The other gripper still closes empty, and an open of the grasping one opens.
    assert robot.gripper("left", "close").outcome == "empty"
    assert robot.gripper("right", "open").outcome == "moved"


# Reaches


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [({"whole_body": True}, "Success (base fixed)"), ({"cartesian": True}, "Success")],
)
def test_a_reach_leaves_its_arm_in_no_named_pose(robot, kwargs, status):
    assert robot.reach("right", HAND, **kwargs) == status

    assert robot.state() == ArmsReport(ArmsState(left=HOME, right=ArmState(None, False)))
    assert robot.state().sentence() == NOT_SAFE_SENTENCE
    with pytest.raises(StepError) as refused:
        robot.go_to("desk")
    assert refused.value.code is ErrorCode.PRECONDITION_FAILED
    assert ARMS_OUT in refused.value.message


def test_a_reach_no_backend_sends_is_refused_in_the_robots_words(robot):
    error = raised(robot.reach, "right", HAND, whole_body=True, cartesian=True)

    assert error.code is ErrorCode.REFUSED
    assert "whole-body reach cannot follow a straight line" in error.message
    assert raised(robot.reach, "right", dataclasses.replace(HAND, z=math.inf)).message == (
        "Refused: z must be a finite number."
    )
    assert robot.state() == AT_HOME


def test_a_stop_refuses_a_reach_with_the_sentence(robot):
    robot.cancel.cancel()

    error = raised(robot.reach, "right", HAND)

    assert error.code is ErrorCode.CANCELLED
    assert error.message.endswith(SAFE_SENTENCE)
    assert robot.state() == AT_HOME


def test_each_gripper_counts_its_own_commands(robot):
    commands = [
        robot.gripper("left", "open").result.command,
        robot.gripper("left", "half").result.command,
        robot.gripper("right", "close").result.command,
        robot.gripper("left", "close").result.command,
    ]

    assert commands == [1, 2, 1, 3]


def test_a_read_finds_each_gripper_at_its_last_position(robot, arm_poses):
    grippers = arm_poses.grippers
    robot.gripper("left", "half")
    robot.gripper("right", "close")
    robot.move_to("standby", "right")

    reading = robot.read()

    assert reading.grippers == (
        # The half position lies inside the default grasp band: the band says what a
        # close left, not what a half open holds.
        GripperReading("left", grippers.half_position, "half", in_grasp_band=True),
        GripperReading("right", grippers.closed_position, "close", in_grasp_band=False),
    )
    assert reading.before == reading.after == robot.state()
    assert reading.after.arms == ArmsState(left=HOME, right=ArmState("standby", False))


def test_a_read_finds_a_held_object_inside_the_grasp_band_and_an_empty_close_below_it(robot):
    robot.grasping.add("right")
    robot.gripper("right", "close")
    robot.gripper("left", "close")

    left, right = robot.read().grippers

    assert (left.reads_as, left.in_grasp_band) == ("close", False)
    assert (right.reads_as, right.in_grasp_band) == ("close", True)


def test_the_grippers_and_the_held_object_are_apart(robot, arm_poses):
    robot.go_to("shelf")

    assert robot.pick("apple_1") == "apple_1"

    assert robot.snapshot()["holding"] == "apple_1"
    assert robot.state() == AT_HOME
    assert [gripper.reads_as for gripper in robot.read().grippers] == ["open", "open"]
    assert robot.gripper("left", "close").outcome == "empty"


def test_the_finger_allowance_is_the_last_motion_s_and_restore_turns_it_off_in_a_stop(robot):
    assert robot.touch is False
    robot.reach("right", HAND, cartesian=True, touch=True)
    assert robot.touch is True
    robot.gripper("right", "open")
    assert robot.touch is False
    robot.move_to("home", "right", touch=True)
    assert robot.touch is True
    robot.cancel.cancel()

    robot.restore_touch()

    assert robot.touch is False
    assert robot.calls[-1]["verb"] == "arms.restore_touch"


# The drive check


@pytest.mark.parametrize(
    ("move", "problem"),
    [
        (("hands_up",), f"the arms are {ARMS_OUT}. Move them"),
        (("receive", "left"), f"the left arm is {ARMS_OUT}. Move it"),
        (("point", "right"), f"the right arm is {ARMS_OUT}. Move it"),
    ],
)
def test_a_drive_with_an_arm_out_is_refused_before_anything_moves(robot, move, problem):
    robot.move_to(*move)
    before = robot.snapshot()

    error = raised(robot.go_to, "shelf")

    assert (error.code, error.message) == (
        ErrorCode.PRECONDITION_FAILED,
        refusal("the drive to shelf", problem),
    )
    assert type(error) is StepError
    after = robot.snapshot()
    assert (after["at"], after["pose"]) == (before["at"], before["pose"])
    # The check logs nothing of its own: the drive's record holds its error.
    assert verbs(robot) == ["arms.move_to", "base.go_to"]
    assert robot.calls[-1]["error"] == "PRECONDITION_FAILED"


def test_a_relative_move_with_an_arm_out_is_refused_in_the_same_words(robot):
    robot.move_to("point", "right")
    before = robot.pose()

    error = raised(robot.move_relative, forward_m=0.5)

    assert (error.code, error.message) == (
        ErrorCode.PRECONDITION_FAILED,
        refusal("the move", f"the right arm is {ARMS_OUT}. Move it"),
    )
    assert robot.pose() == before
    assert robot.calls[-2]["error"] == "PRECONDITION_FAILED"


def test_a_drive_goes_once_the_arms_are_back_in_travel_safe_poses(robot):
    robot.move_to("hands_up")
    assert raised(robot.go_to, "shelf").code is ErrorCode.PRECONDITION_FAILED

    robot.move_to("home")
    robot.go_to("shelf")
    robot.move_to("carry", "left")
    robot.move_to("carry", "right")
    robot.move_relative(turn_deg=90.0)

    assert robot.snapshot()["at"] == "shelf"
    assert [call["error"] for call in robot.calls if call["verb"].startswith("base.")] == [
        "PRECONDITION_FAILED",
        None,
        None,
    ]


def test_a_refused_drive_does_not_wait_out_its_drive_time(arm_poses):
    robot = FakeRobot(world(), drive_s=30.0, arm_poses=arm_poses)
    robot.move_to("standby")
    started = time.monotonic()

    assert raised(robot.go_to, "shelf").code is ErrorCode.PRECONDITION_FAILED

    assert time.monotonic() - started < 1.0


def test_the_robot_s_earlier_refusals_come_before_the_arm_check(robot):
    robot.move_to("hands_up")

    assert raised(robot.go_to, "moon").code is ErrorCode.NOT_FOUND
    assert raised(robot.go_to, "shelf", timeout_s=0.0).code is ErrorCode.REFUSED
    assert raised(robot.move_relative).code is ErrorCode.REFUSED
    assert raised(robot.move_relative, forward_m=0.5, timeout_s=-1.0).code is ErrorCode.REFUSED
    robot.cancel.cancel()
    assert raised(robot.go_to, "shelf").code is ErrorCode.CANCELLED
    assert raised(robot.move_relative, forward_m=0.5).code is ErrorCode.CANCELLED


# Stops, faults and the log


def test_a_stop_refuses_every_arm_goal_with_the_sentence_and_changes_nothing(robot):
    robot.move_to("hands_up", "both")
    robot.cancel.cancel()

    move = raised(robot.move_to, "home")
    close = raised(robot.gripper, "left", "close")
    read = raised(robot.read)

    assert (move.code, move.message) == (
        ErrorCode.CANCELLED,
        f"Cancelled the move to 'home': a stop is in effect. {NOT_SAFE_SENTENCE}",
    )
    assert (close.code, close.message) == (
        ErrorCode.CANCELLED,
        f"Cancelled the left gripper command: a stop is in effect. {NOT_SAFE_SENTENCE}",
    )
    assert (read.code, read.message) == (
        ErrorCode.CANCELLED,
        "Cancelled the read of the arms: a stop is in effect.",
    )
    robot.cancel.reset()
    assert robot.state().arms == ArmsState(left=HANDS_UP, right=HANDS_UP)
    assert [gripper.reads_as for gripper in robot.read().grippers] == ["open", "open"]
    assert robot.gripper("left", "half").result.command == 1


def test_a_stop_still_lists_the_poses_and_names_an_unknown_one(robot):
    robot.cancel.cancel()

    assert "home" in robot.poses()
    assert raised(robot.move_to, "wave").code is ErrorCode.NOT_FOUND
    assert raised(robot.move_to, "receive", "right").code is ErrorCode.REFUSED


def test_a_cancel_after_the_checks_and_before_the_commit_moves_nothing(robot, monkeypatch):
    offered_pose = answers.offered_pose

    def then_cancel(*args, **kwargs):
        found = offered_pose(*args, **kwargs)
        robot.cancel.cancel()
        return found

    monkeypatch.setattr(answers, "offered_pose", then_cancel)

    assert raised(robot.move_to, "hands_up").code is ErrorCode.CANCELLED
    assert robot.state() == AT_HOME
    assert robot.calls[-1]["error"] == "CANCELLED"


@pytest.mark.parametrize(
    ("verb", "args"),
    [("poses", ()), ("read", ()), ("move_to", ("hands_up",)), ("gripper", ("left", "close"))],
)
def test_a_fault_fails_the_arm_verb_and_moves_nothing(robot, verb, args):
    robot.faults[f"arms.{verb}"] = StepError(ErrorCode.SKILL_FAILED, "Planning Failed.")

    assert raised(getattr(robot, verb), *args).code is ErrorCode.SKILL_FAILED

    assert robot.calls[-1]["error"] == "SKILL_FAILED"
    del robot.faults[f"arms.{verb}"]
    assert robot.state() == AT_HOME
    assert [gripper.reads_as for gripper in robot.read().grippers] == ["open", "open"]


def test_every_arm_verb_but_state_is_logged_with_its_arguments(robot):
    robot.poses()
    robot.state()
    robot.state(read=True)
    robot.read()
    robot.move_to("home", "left")
    robot.gripper("right", "half")

    assert [(call["verb"], call["args"], call["error"]) for call in robot.calls] == [
        ("arms.poses", {}, None),
        ("arms.read", {}, None),
        ("arms.move_to", {"pose": "home", "arm": "left"}, None),
        ("arms.gripper", {"arm": "right", "state": "half"}, None),
    ]
    assert json.loads(json.dumps(robot.snapshot()["calls"])) == robot.calls


def test_the_arm_state_stays_out_of_the_snapshot(robot, armless):
    def without_calls(snapshot: dict) -> dict:
        return {key: value for key, value in snapshot.items() if key != "calls"}

    before = without_calls(robot.snapshot())
    robot.move_to("hands_up")
    robot.gripper("left", "close")

    assert without_calls(robot.snapshot()) == before
    assert robot.snapshot().keys() == armless.snapshot().keys()


# Construction


def test_arm_poses_without_a_home_for_each_arm_are_refused(arm_poses):
    home = arm_poses.poses["home"]
    no_home = {name: pose for name, pose in arm_poses.poses.items() if name != "home"}
    left_only = dataclasses.replace(home, tables={"left": home.tables["left"]})
    one_armed = {**arm_poses.poses, "home": left_only}

    for poses in (no_home, one_armed):
        with pytest.raises(ValueError, match="the arms start in 'home'"):
            FakeRobot(world(), arm_poses=dataclasses.replace(arm_poses, poses=poses))
