"""The shared arm wording and rules (walkie_agent.arm.answers) and ArmsReport.

Plain logic, no ROS: the poses come from config/arm_poses.toml.
"""

import logging

import pytest

from conftest import PROJECT_ROOT
from walkie_agent.arm import answers
from walkie_agent.arm.poses import load_arm_poses
from walkie_agent.arm.results import NO_ANSWER, NO_READING, NOT_FOUND, DriveCheck
from walkie_agent.arm.state import ARMS_OUT, ArmsState, ArmState
from walkie_agent.capabilities import ArmsReport
from walkie_agent.errors import ErrorCode, GoalFailed, StepError

HOME = ArmState("home", True)
HANDS_UP = ArmState("hands_up", False)
SENTENCE = "Arms: state unknown."


@pytest.fixture(scope="module")
def arm_poses():
    return load_arm_poses(
        PROJECT_ROOT / "config" / "arm_poses.toml", PROJECT_ROOT / "config" / "robot_model.toml"
    )


def raised(call, *args):
    with pytest.raises(StepError) as caught:
        call(*args)
    return caught.value


@pytest.mark.parametrize("text", ["Hands up", "hands-up", "  HANDS_UP "])
def test_pose_name_spells_a_pose_as_the_file_does(text):
    assert answers.pose_name(text) == "hands_up"


def test_an_offered_pose_is_found_by_any_spelling(arm_poses):
    offered = arm_poses.offered_poses(sim=True)
    assert answers.offered_pose(offered, "Hands up", "both", lambda: SENTENCE).name == "hands_up"


def test_a_hidden_pose_gets_the_answer_of_an_unknown_one(arm_poses):
    robot = arm_poses.offered_poses(sim=False)
    assert "receive" not in robot
    error = raised(answers.offered_pose, robot, "receive", "left", lambda: SENTENCE)
    assert error.code is ErrorCode.NOT_FOUND
    assert error.message == (
        f"No pose named 'receive'. The poses are: {', '.join(robot)}. {SENTENCE}"
    )


def test_an_arm_the_pose_does_not_take_is_refused(arm_poses):
    offered = arm_poses.offered_poses(sim=True)
    error = raised(answers.offered_pose, offered, "receive", "right", lambda: SENTENCE)
    assert error.code is ErrorCode.REFUSED
    assert error.message == f"'receive' is for the left arm only. {SENTENCE}"


def test_with_sentence_keeps_the_code_and_the_class():
    error = answers.with_sentence(
        GoalFailed(ErrorCode.SKILL_FAILED, "The motion stopped"), "Could not move: ", SENTENCE
    )
    assert isinstance(error, GoalFailed)
    assert str(error) == f"SKILL_FAILED: Could not move: The motion stopped. {SENTENCE}"


def test_a_move_that_fails_its_check_is_a_plain_skill_failure():
    error = answers.unchecked("Success", "joint1 is 0.30 rad from 'home'.", SENTENCE)
    assert type(error) is StepError
    assert error.code is ErrorCode.SKILL_FAILED
    assert "Treat this as a failure." in error.message
    planned = answers.unchecked(answers.PLAN_ONLY, "joint1 is 0.30 rad from 'home'.", SENTENCE)
    assert answers.PLAN_ONLY_NEXT in planned.message


@pytest.mark.parametrize(
    ("finger_m", "state"), [(0.0, "close"), (0.03, "close"), (0.035, "half"), (0.044, "open")]
)
def test_a_gripper_reads_as_the_nearest_named_position(arm_poses, finger_m, state):
    assert answers.reads_as(arm_poses.grippers, finger_m) == state


def test_the_last_command_is_named_only_when_it_is_exact(arm_poses):
    grippers = arm_poses.grippers
    assert answers.commanded_state(grippers, grippers.half_position) == "half"
    assert answers.commanded_state(grippers, 0.01) is None
    assert answers.commanded_state(grippers, None) is None


def test_arms_report_never_says_safe_while_unverified():
    safe = ArmsState(HOME, HOME)
    assert ArmsReport(safe).travel_safe is True
    assert ArmsReport(safe).sentence() == "Arms: safe for driving."
    unverified = ArmsReport(safe, unverified="a goal was given up on")
    assert unverified.travel_safe is None
    assert unverified.sentence() == SENTENCE
    out = ArmsReport(ArmsState(HANDS_UP, HOME), unverified="a goal was given up on")
    assert out.travel_safe is False
    assert out.sentence() == "Arms: not safe for driving."


@pytest.mark.parametrize(
    ("state", "words"),
    [
        (ArmsState(HANDS_UP, HOME), f"the left arm is {ARMS_OUT}. Move it"),
        (ArmsState(HOME, HANDS_UP), f"the right arm is {ARMS_OUT}. Move it"),
        (ArmsState(HANDS_UP, HANDS_UP), f"the arms are {ARMS_OUT}. Move them"),
        (ArmsState(HANDS_UP, None), f"the left arm is {ARMS_OUT}. Move it"),
    ],
)
def test_the_drive_is_refused_with_an_arm_out(state, words):
    error = raised(answers.drive_verdict, lambda: DriveCheck(state), "the drive to kitchen room")
    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        f"Not starting the drive to kitchen room: {words} with move_arm_to_pose first, "
        "for example 'home'."
    )


def test_the_drive_goes_ahead_with_both_arms_travel_safe():
    answers.drive_verdict(lambda: DriveCheck(ArmsState(HOME, ArmState("carry", True))), "the move")


@pytest.mark.parametrize("unread", [NOT_FOUND, NO_ANSWER, NO_READING])
def test_the_drive_fails_open_with_a_warning_when_the_arms_were_not_read(unread, caplog):
    with caplog.at_level(logging.WARNING, logger="walkie_agent.arm.answers"):
        answers.drive_verdict(lambda: DriveCheck(None, unread), "the move")
    assert "Starting the move with the arm state unknown" in caplog.text


def test_a_reading_without_both_arms_refuses_the_drive():
    error = raised(answers.drive_verdict, lambda: DriveCheck(ArmsState(HOME, None)), "the move")
    assert error.code is ErrorCode.INTERNAL
    assert error.message.startswith("Not starting the move: the commander's reading")


@pytest.mark.parametrize(
    ("code", "ending"),
    [
        (ErrorCode.PRECONDITION_FAILED, "An arm may still be moving."),
        (ErrorCode.TIMEOUT, "An arm may still be moving. Try again in a few seconds."),
        (ErrorCode.INTERNAL, "An arm may still be moving."),
    ],
)
def test_a_failed_check_refuses_the_drive_as_a_plain_error(code, ending):
    def check():
        raise GoalFailed(code, "An arm may still be moving")

    error = raised(answers.drive_verdict, check, "the move")
    assert type(error) is StepError
    assert error.code is code
    assert error.message == f"Not starting the move. {ending}"


def test_a_stop_during_the_check_stays_cancelled():
    def check():
        raise StepError(ErrorCode.CANCELLED, "Cancelled the check: a stop is in effect.")

    error = raised(answers.drive_verdict, check, "the move")
    assert error.code is ErrorCode.CANCELLED
    assert error.message == "Cancelled the check: a stop is in effect."
