"""The arm atomics: their schemas, their answers over FakeRobot, and the ends only a stub gives.

FakeRobot's arms always land exactly and its grippers never meet an object, so the
other ends of ctx.arms (plan-only, every gripper outcome, an unread gripper, a
recovery, arms that became unverified during a read, a failed read) come from
StubArms, which answers each verb as a test sets it.
"""

import dataclasses
import inspect
import threading
import time

import pydantic
import pytest

from conftest import PROJECT_ROOT
from test_object_memory_atomics import Remembers
from walkie_agent.arm import answers
from walkie_agent.arm.poses import load_arm_poses
from walkie_agent.arm.results import GripperResult, Recovery
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
    ArmsMoved,
    ArmsReading,
    ArmsReport,
    Ctx,
    GripperMoved,
    GripperReading,
    RecalledObject,
)
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.errors import ErrorCode, GoalFailed, StepError
from walkie_agent.tools.adapter import as_tool
from walkie_agent.tools.atomic import (
    get_arm_state,
    go_to,
    list_arm_poses,
    move_arm_to_pose,
    set_gripper,
)
from walkie_agent.tools.atomic import arms as arm_atomics
from walkie_agent.tools.sets import ARM_TOOLS, ARMS

HOME = ArmState("home", True)
HANDS_UP = ArmState("hands_up", False)
NO_POSE = ArmState(None, False)
AT_HOME = ArmsReport(ArmsState(left=HOME, right=HOME))
# Causes as the commander client words them (CommanderClient.unverified).
ABANDONED = "the move of left_arm from an earlier call was given up on before its end was known"
ACCEPTED_LATE = "the commander accepted the move of left_arm from an earlier call late"
UNREAD = "The commander's joint state service is not available"


def world() -> dict:
    return {
        "places": {
            "lobby": {"x": 0.0, "y": 0.0, "yaw": 0.0, "room": "lobby"},
            "desk": {"x": 1.0, "y": 0.0, "yaw": 0.0, "room": "lobby"},
        },
        "objects": {},
        "people": {},
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


def raised(call, *args, **kwargs) -> StepError:
    with pytest.raises(StepError) as caught:
        call(*args, **kwargs)
    return caught.value


def arm_calls(robot: FakeRobot) -> list[tuple[str, dict]]:
    return [
        (call["verb"], call["args"]) for call in robot.calls if call["verb"].startswith("arms.")
    ]


def pose_line(arm_poses, name: str, arms: str, safe: bool) -> str:
    description = arm_poses.poses[name].description
    return f"{name}: {arms}; {'safe' if safe else 'not safe'} for driving. {description}"


def moved(arm, state, outcome, *, success=True, status="Success", finger_m=None, read_error=None):
    """A GripperMoved as RosArms.gripper gives it for `outcome`."""
    closed = state == "close" and outcome != "unmeasured"
    result = GripperResult(
        success=success,
        status=status,
        grasped=(outcome == "holding") if closed else None,
        gripper_gap=finger_m if closed else None,
        command=1,
    )
    if outcome == "unmeasured":
        result = GripperResult(False, "Failed", False, 0.0, 1)
    return GripperMoved(
        arm=arm,
        state=state,
        outcome=outcome,
        result=result,
        finger_m=finger_m,
        read_error=read_error,
    )


class StubArms:
    """ctx.arms that answers each verb as the test sets it, and records what was asked."""

    def __init__(self, report: ArmsReport = AT_HOME) -> None:
        # What state() returns: one report, or a callable of `read`.
        self.report = report
        self.offered: dict = {}
        self.reading: ArmsReading | None = None
        self.read_error: StepError | None = None
        self.moved: ArmsMoved | None = None
        self.move_error: StepError | None = None
        # By gripper: a GripperMoved, or a StepError to raise.
        self.grippers: dict = {}
        self.states: list[bool | None] = []
        self.asked: list[tuple] = []

    def poses(self):
        return self.offered

    def state(self, *, read: bool | None = None) -> ArmsReport:
        self.states.append(read)
        return self.report(read) if callable(self.report) else self.report

    def read(self) -> ArmsReading:
        self.asked.append(("read",))
        if self.read_error is not None:
            raise self.read_error
        return self.reading

    def move_to(self, pose, arm="both", *, lift_m=None, touch=False) -> ArmsMoved:
        self.asked.append(("move_to", pose, arm))
        if self.move_error is not None:
            raise self.move_error
        return self.moved

    def gripper(self, arm, state) -> GripperMoved:
        self.asked.append(("gripper", arm, state))
        answer = self.grippers[arm]
        if isinstance(answer, StepError):
            raise answer
        return answer

    def pick(self, object_id, *, timeout_s=60.0):
        raise AssertionError("not an arm atomic's verb")

    place_on = hand_over = pick


@pytest.fixture
def stub() -> StubArms:
    return StubArms()


@pytest.fixture
def ctx(stub) -> Ctx:
    return dataclasses.replace(FakeRobot(world()).ctx, arms=stub)


def reading(before=AT_HOME, after=AT_HOME, grippers=None) -> ArmsReading:
    if grippers is None:
        grippers = (GripperReading("left", 0.044, "open"), GripperReading("right", 0.044, "open"))
    return ArmsReading(before=before, after=after, grippers=grippers)


def cleared(recovery: Recovery) -> ArmsReport:
    """Both arms home and verified, as `recovery` left them: its count is the latest."""
    return dataclasses.replace(AT_HOME, unverified_count=recovery.count, recovery=recovery)


# Schemas


def test_the_arm_tools_build_and_take_their_choices_as_enums(robot):
    tools = {fn.__name__: as_tool(fn, robot.ctx) for fn in ARMS if fn.__name__ in ARM_TOOLS}
    assert set(tools) == ARM_TOOLS
    assert tools["list_arm_poses"].args == {}
    assert tools["get_arm_state"].args == {}
    move = tools["move_arm_to_pose"].tool_call_schema.model_json_schema()
    assert move["required"] == ["pose"]
    assert move["properties"]["pose"]["type"] == "string"
    assert move["properties"]["arm"]["enum"] == ["left", "right", "both"]
    assert move["properties"]["arm"]["default"] == "both"
    grip = tools["set_gripper"].tool_call_schema.model_json_schema()
    assert grip["required"] == ["arm", "state"]
    assert grip["properties"]["arm"]["enum"] == ["left", "right", "both"]
    assert grip["properties"]["state"]["enum"] == ["open", "half", "close"]


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (move_arm_to_pose, {"pose": "home", "arm": "up"}),
        (move_arm_to_pose, {"pose": "home", "arm": "Left"}),
        (move_arm_to_pose, {"pose": 3}),
        (set_gripper, {"arm": "left", "state": "closed"}),
        (set_gripper, {"arm": "middle", "state": "open"}),
        (set_gripper, {"arm": "left"}),
    ],
)
def test_a_tool_refuses_a_value_outside_its_choices_and_moves_nothing(robot, fn, args):
    with pytest.raises(pydantic.ValidationError):
        as_tool(fn, robot.ctx).invoke(args)
    assert robot.calls == []


def test_the_receive_delay_is_the_one_set_gripper_s_docstring_gives():
    doc = " ".join(inspect.getdoc(set_gripper).split())
    assert f"starts {arm_atomics.RECEIVE_CLOSE_DELAY_S:g} s after the call" in doc


# list_arm_poses


def test_list_arm_poses_gives_each_offered_pose_and_ends_with_the_sentence(robot, arm_poses):
    lines = list_arm_poses(robot.ctx).splitlines()

    assert lines == [
        pose_line(arm_poses, "home", "left, right or both", True),
        pose_line(arm_poses, "hands_up", "left, right or both", False),
        pose_line(arm_poses, "standby", "left, right or both", False),
        pose_line(arm_poses, "tray", "left, right or both", False),
        pose_line(arm_poses, "receive", "left arm only", False),
        pose_line(arm_poses, "carry", "left or right, one arm at a time", True),
        pose_line(arm_poses, "point", "right arm only", False),
        SAFE_SENTENCE,
    ]
    assert arm_calls(robot) == [("arms.poses", {})]


def test_list_arm_poses_ends_with_where_the_arms_are_now(robot):
    move_arm_to_pose(robot.ctx, "hands_up")

    assert list_arm_poses(robot.ctx).splitlines()[-1] == NOT_SAFE_SENTENCE


def test_list_arm_poses_says_when_no_pose_is_offered(ctx, stub):
    stub.report = ArmsReport(ArmsState())

    assert list_arm_poses(ctx).splitlines() == ["No arm poses are offered.", UNKNOWN_SENTENCE]
    # It lets the backend decide whether to read the arms.
    assert stub.states == [None]


# move_arm_to_pose


@pytest.mark.parametrize(
    ("pose", "arm", "answer"),
    [
        ("Hands up", "both", f"Moved to 'hands_up'. {NOT_SAFE_SENTENCE}"),
        ("home", "both", f"Moved to 'home'. {SAFE_SENTENCE}"),
        ("receive", "left", f"Moved the left arm to 'receive'. {NOT_SAFE_SENTENCE}"),
        # carry is travel-safe, and the right arm stays in home.
        ("carry", "left", f"Moved the left arm to 'carry'. {SAFE_SENTENCE}"),
    ],
)
def test_a_move_says_where_the_arms_went_and_ends_with_the_sentence(robot, pose, arm, answer):
    assert move_arm_to_pose(robot.ctx, pose, arm) == answer
    assert arm_calls(robot) == [("arms.move_to", {"pose": pose, "arm": arm})]


def test_a_move_takes_both_arms_by_default(robot):
    as_tool(move_arm_to_pose, robot.ctx).invoke({"pose": "standby"})

    assert arm_calls(robot) == [("arms.move_to", {"pose": "standby", "arm": "both"})]
    assert robot.state().arms == ArmsState(
        left=ArmState("standby", False), right=ArmState("standby", False)
    )


@pytest.mark.parametrize(
    ("pose", "arm", "code"),
    [("wave", "both", ErrorCode.NOT_FOUND), ("receive", "right", ErrorCode.REFUSED)],
)
def test_a_refused_move_reaches_the_llm_as_the_arms_raised_it(robot, pose, arm, code):
    error = raised(move_arm_to_pose, robot.ctx, pose, arm)

    assert error.code is code
    assert error.message.endswith(f" {SAFE_SENTENCE}")
    assert error.message == raised(robot.move_to, pose, arm).message


@pytest.mark.parametrize(
    ("arm", "answer"),
    [
        ("both", "Already at 'home'."),
        ("left", "The left arm is already at 'home'."),
    ],
)
def test_plan_only_never_says_the_arms_moved(ctx, stub, arm, answer):
    stub.moved = ArmsMoved("home", arm, answers.PLAN_ONLY, 0.01, AT_HOME)

    assert move_arm_to_pose(ctx, "home", arm) == (
        f"{answer} The commander runs in plan-only mode, so it moved nothing. {SAFE_SENTENCE}"
    )
    assert stub.asked == [("move_to", "home", arm)]


def test_a_move_ends_with_the_sentence_of_the_report_after_the_check(ctx, stub):
    # The report after the check is unverified, as when a goal from an earlier call
    # was accepted late, so the sentence is not the one the arms alone would give.
    report = dataclasses.replace(AT_HOME, unverified=ACCEPTED_LATE, unverified_count=1)
    stub.moved = ArmsMoved("home", "both", "Success", 0.02, report)

    assert move_arm_to_pose(ctx, "home") == f"Moved to 'home'. {UNKNOWN_SENTENCE}"
    # No other report was asked for.
    assert stub.states == []


@pytest.mark.parametrize(
    "error",
    [
        GoalFailed(
            ErrorCode.SKILL_FAILED,
            f"Could not move to 'home': MoveIt found no path. {SAFE_SENTENCE}",
        ),
        StepError(ErrorCode.CANCELLED, f"Stopped before the arms moved. {UNKNOWN_SENTENCE}"),
    ],
)
def test_a_failed_move_reaches_the_llm_unchanged(ctx, stub, error):
    stub.move_error = error

    assert raised(move_arm_to_pose, ctx, "home") is error


def test_the_robot_drives_only_once_the_arms_are_back_in_a_travel_safe_pose(robot):
    move_arm_to_pose(robot.ctx, "hands_up")

    refusal = raised(go_to, robot.ctx, "desk")

    assert refusal.code is ErrorCode.PRECONDITION_FAILED
    assert ARMS_OUT in refusal.message and "move_arm_to_pose" in refusal.message
    assert robot.snapshot()["at"] == "lobby"
    assert move_arm_to_pose(robot.ctx, "home") == f"Moved to 'home'. {SAFE_SENTENCE}"
    assert go_to(robot.ctx, "desk") == "Arrived at desk."


# set_gripper over the fake


@pytest.mark.parametrize(
    ("arm", "state", "answer"),
    [
        ("left", "open", "Opened the left gripper."),
        ("right", "half", "Set the right gripper half open."),
        ("both", "open", "Opened both grippers."),
        ("both", "half", "Set both grippers half open."),
        ("right", "close", "Closed the right gripper. It is holding nothing."),
        (
            "both",
            "close",
            (
                "Closed the left gripper. It is holding nothing. Closed the right gripper. It "
                "is holding nothing."
            ),
        ),
    ],
)
def test_a_gripper_command_says_what_each_gripper_did(robot, arm, state, answer):
    assert set_gripper(robot.ctx, arm, state) == f"{answer} {SAFE_SENTENCE}"
    sides = ["left", "right"] if arm == "both" else [arm]
    assert arm_calls(robot) == [("arms.gripper", {"arm": side, "state": state}) for side in sides]


def test_a_gripper_answer_ends_with_where_the_arms_are(robot):
    move_arm_to_pose(robot.ctx, "hands_up")

    assert set_gripper(robot.ctx, "left", "open") == f"Opened the left gripper. {NOT_SAFE_SENTENCE}"


# set_gripper: what the object memory records when a hand opens (decision PK-70)

RIGHT_HAND = "openarm_right_hand_tcp"
HELD_CUP = RecalledObject(
    id="9",
    name="cup",
    label="cup",
    category="dish",
    room=None,
    on=None,
    state="held",
    seen_s_ago=0.5,
    x=None,
    y=None,
)


def remembering_a_cup_in_the_right_hand(robot: FakeRobot) -> tuple[Ctx, Remembers]:
    memory = Remembers(HELD_CUP, holders={"9": RIGHT_HAND})
    return dataclasses.replace(robot.ctx, object_memory=memory), memory


@pytest.mark.parametrize(
    ("pose", "operation", "after"),
    [("standby", "hand_over_succeeded", "handed_over"), ("home", "place_succeeded", "missing")],
    ids=["handover", "place"],
)
@pytest.mark.parametrize(
    ("state", "said"),
    [("open", "Opened the right gripper."), ("half", "Set the right gripper half open.")],
)
def test_opening_a_hand_records_that_it_let_go_of_what_it_held(
    robot, pose, operation, after, state, said
):
    ctx, memory = remembering_a_cup_in_the_right_hand(robot)
    move_arm_to_pose(ctx, pose, "right")

    answer = set_gripper(ctx, "right", state)

    # The answer is the one without a memory: the record is bookkeeping only.
    assert answer == f"{said} {ctx.arms.state().sentence()}"
    assert memory.calls[-1] == ("set_state", "9", operation, RIGHT_HAND)
    assert memory.objects["9"].state == after


def test_opening_both_hands_records_the_release_only_for_the_hand_that_held_it(robot):
    ctx, memory = remembering_a_cup_in_the_right_hand(robot)
    move_arm_to_pose(ctx, "standby", "right")

    set_gripper(ctx, "both", "open")

    # The left arm is home, so its open reads as a place, and the memory refuses that
    # record, since the right hand held the cup.
    records = [call for call in memory.calls if call[0] == "set_state"]
    assert records == [
        ("set_state", "9", "place_succeeded", "openarm_left_hand_tcp"),
        ("set_state", "9", "hand_over_succeeded", RIGHT_HAND),
    ]
    assert memory.objects["9"].state == "handed_over"


def test_a_close_records_nothing(robot):
    ctx, memory = remembering_a_cup_in_the_right_hand(robot)

    set_gripper(ctx, "right", "close")

    assert memory.calls == []
    assert memory.objects["9"].state == "held"


def test_a_memory_that_cannot_record_the_release_changes_no_answer(robot):
    ctx, memory = remembering_a_cup_in_the_right_hand(robot)
    memory.set_error = StepError(ErrorCode.TIMEOUT, "The object memory did not answer in time.")

    assert set_gripper(ctx, "right", "open") == f"Opened the right gripper. {SAFE_SENTENCE}"
    assert memory.objects["9"].state == "held"


def holding_the_cup(ctx: Ctx) -> tuple[Ctx, Remembers]:
    memory = Remembers(HELD_CUP, holders={"9": RIGHT_HAND})
    return dataclasses.replace(ctx, object_memory=memory), memory


@pytest.mark.parametrize(
    ("known", "operation"),
    [(ArmState("standby", False), "hand_over_succeeded"), (HOME, "place_succeeded")],
    ids=["standby", "home"],
)
def test_a_release_before_the_arm_was_read_reads_it_for_its_pose(ctx, stub, known, operation):
    # As after a restart with the cup still in the hand: the backend knows no pose yet.
    stub.report = lambda read: ArmsReport(ArmsState(left=HOME, right=known if read else None))
    stub.grippers["right"] = moved("right", "open", "moved")
    ctx, memory = holding_the_cup(ctx)

    set_gripper(ctx, "right", "open")

    assert memory.calls[-1] == ("set_state", "9", operation, RIGHT_HAND)
    # The release reads the arms once; the answer's own state() comes after it.
    assert stub.states == [False, True, None]


def test_a_release_while_the_arms_are_unverified_reads_nothing_and_records_a_place(ctx, stub):
    stub.report = ArmsReport(ArmsState(), unverified=ABANDONED, unverified_count=1)
    stub.grippers["right"] = moved("right", "open", "moved")
    ctx, memory = holding_the_cup(ctx)

    set_gripper(ctx, "right", "open")

    # A read would recover the arms, which only get_arm_state and the drive check do.
    assert True not in stub.states
    assert memory.calls[-1] == ("set_state", "9", "place_succeeded", RIGHT_HAND)


@pytest.mark.parametrize("state", ["open", "half"])
def test_a_hand_that_did_not_open_records_nothing(ctx, stub, state):
    stub.grippers["right"] = moved("right", state, "not_reached", success=False, status="Failed")
    ctx, memory = holding_the_cup(ctx)

    raised(set_gripper, ctx, "right", state)

    assert memory.calls == []
    assert memory.objects["9"].state == "held"


# set_gripper: each outcome, through the stub


@pytest.mark.parametrize(
    ("result", "answer"),
    [
        (
            moved("left", "close", "holding", success=False, status="Failed", finger_m=0.02),
            "Closed the left gripper. It is holding something.",
        ),
        (
            moved("left", "close", "empty", finger_m=0.0),
            "Closed the left gripper. It is holding nothing.",
        ),
        (
            moved("left", "close", "unmeasured"),
            (
                "The commander reported that the left gripper's close failed and gave no "
                "measurement, as when another command replaced it, so whether the gripper "
                "holds something is unknown. Call get_arm_state to see how far it is open."
            ),
        ),
        (
            moved("left", "close", "superseded"),
            (
                "Another command for the left gripper came after the close, so whether the "
                "close held something is unknown. Call get_arm_state to see how far the "
                "gripper is open."
            ),
        ),
        (
            # The read error has no full stop of its own, so the answer adds one.
            moved("left", "close", "unread", read_error=UNREAD),
            (
                "The left gripper's fingers could not be read after the close, so whether it "
                f"closed on something is unknown: {UNREAD}. Call get_arm_state to see how far "
                "it closed."
            ),
        ),
    ],
    ids=["holding", "empty", "unmeasured", "superseded", "unread"],
)
def test_a_close_that_ended_as_the_commander_allows_is_answered(ctx, stub, result, answer):
    stub.grippers["left"] = result

    assert set_gripper(ctx, "left", "close") == f"{answer} {SAFE_SENTENCE}"


@pytest.mark.parametrize(
    ("state", "outcome", "answer"),
    [
        (
            "open",
            "not_reached",
            "The left gripper did not open fully (the commander reported 'Failed').",
        ),
        (
            "half",
            "not_reached",
            "The left gripper did not reach half open (the commander reported 'Failed').",
        ),
        ("close", "not_closed", "The left gripper did not close (gap 0.043 m)."),
    ],
)
def test_a_gripper_that_did_not_get_there_is_a_failure(ctx, stub, state, outcome, answer):
    finger = 0.043 if outcome == "not_closed" else None
    stub.grippers["left"] = moved(
        "left", state, outcome, success=False, status="Failed", finger_m=finger
    )

    error = raised(set_gripper, ctx, "left", state)

    assert type(error) is StepError
    assert (error.code, error.message) == (ErrorCode.SKILL_FAILED, f"{answer} {SAFE_SENTENCE}")


def test_when_one_gripper_fails_the_other_still_moves_and_the_answer_says_what_each_did(ctx, stub):
    stub.grippers = {
        "left": moved("left", "half", "moved"),
        "right": moved("right", "half", "not_reached", success=False, status="Failed"),
    }

    error = raised(set_gripper, ctx, "both", "half")

    assert (error.code, error.message) == (
        ErrorCode.SKILL_FAILED,
        (
            "Set the left gripper half open. The right gripper did not reach half open (the "
            f"commander reported 'Failed'). {SAFE_SENTENCE}"
        ),
    )
    assert stub.asked == [("gripper", "left", "half"), ("gripper", "right", "half")]


def test_a_stuck_gripper_fails_and_the_other_still_closes(ctx, stub):
    stub.grippers = {
        "left": moved("left", "close", "not_closed", success=False, finger_m=0.0425),
        "right": moved("right", "close", "holding", success=False, finger_m=0.02),
    }

    error = raised(set_gripper, ctx, "both", "close")

    assert error.message == (
        "The left gripper did not close (gap 0.043 m). Closed the right gripper. It is holding "
        f"something. {SAFE_SENTENCE}"
    )


def test_an_error_on_the_second_gripper_keeps_what_the_first_did(ctx, stub):
    rejected = StepError(
        ErrorCode.INTERNAL,
        "Could not move the right gripper: The commander rejected the goal for right_gripper. "
        f"{SAFE_SENTENCE}",
    )
    stub.grippers = {"left": moved("left", "close", "empty", finger_m=0.0), "right": rejected}

    error = raised(set_gripper, ctx, "both", "close")

    assert (error.code, error.message) == (
        ErrorCode.INTERNAL,
        (
            "Closed the left gripper. It is holding nothing. Could not move the right "
            f"gripper: The commander rejected the goal for right_gripper. {SAFE_SENTENCE}"
        ),
    )


def test_an_error_on_the_first_gripper_sends_nothing_more_and_keeps_its_class(ctx, stub):
    failed = GoalFailed(
        ErrorCode.SKILL_FAILED, f"Could not move the left gripper: no. {SAFE_SENTENCE}"
    )
    stub.grippers = {"left": failed, "right": moved("right", "open", "moved")}

    error = raised(set_gripper, ctx, "both", "open")

    assert type(error) is GoalFailed
    assert (error.code, error.message) == (failed.code, failed.message)
    assert stub.asked == [("gripper", "left", "open")]


# set_gripper: the close in the receive pose


@pytest.fixture
def short_delay(monkeypatch) -> float:
    monkeypatch.setattr(arm_atomics, "RECEIVE_CLOSE_DELAY_S", 0.5)
    return 0.5


def timed(call, *args):
    started = time.monotonic()
    answer = call(*args)
    return answer, time.monotonic() - started


def test_a_close_in_receive_waits_before_its_goal_is_sent(robot, short_delay):
    move_arm_to_pose(robot.ctx, "receive", "left")

    answer, took = timed(set_gripper, robot.ctx, "left", "close")

    assert answer == f"Closed the left gripper. It is holding nothing. {NOT_SAFE_SENTENCE}"
    assert took >= short_delay


def test_a_close_of_both_grippers_waits_once_when_one_arm_is_in_receive(robot, short_delay):
    move_arm_to_pose(robot.ctx, "receive", "left")

    _, took = timed(set_gripper, robot.ctx, "both", "close")

    assert short_delay <= took < 2 * short_delay


@pytest.mark.parametrize(
    ("pose", "arm", "gripper", "state"),
    [
        ("receive", "left", "left", "open"),
        ("receive", "left", "left", "half"),
        # The right arm is still in home.
        ("receive", "left", "right", "close"),
        ("hands_up", "both", "left", "close"),
    ],
)
def test_an_open_or_a_close_outside_receive_is_sent_at_once(robot, pose, arm, gripper, state):
    # With the delay the robot uses, a wait here would take 5 s.
    move_arm_to_pose(robot.ctx, pose, arm)

    _, took = timed(set_gripper, robot.ctx, gripper, state)

    assert took < 1.0


@pytest.mark.parametrize(
    ("arm", "which"), [("left", "the left gripper"), ("both", "both grippers")]
)
def test_a_stop_during_the_close_delay_sends_no_goal(robot, monkeypatch, arm, which):
    move_arm_to_pose(robot.ctx, "receive", "left")
    wait_for = robot.cancel.wait_for
    waited = []

    def stopped_while_waiting(*args, **kwargs):
        waited.append(args[1])
        robot.cancel.cancel()
        return wait_for(*args, **kwargs)

    monkeypatch.setattr(robot.cancel, "wait_for", stopped_while_waiting)

    error = raised(set_gripper, robot.ctx, arm, "close")

    assert waited == [arm_atomics.RECEIVE_CLOSE_DELAY_S]
    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        f"Stopped before {which} moved: a stop or a reset is in effect. {NOT_SAFE_SENTENCE}",
    )
    assert [verb for verb, _ in arm_calls(robot)] == ["arms.move_to"]


def test_a_stop_from_another_thread_ends_the_close_delay_at_once(robot):
    move_arm_to_pose(robot.ctx, "receive", "left")
    stop = threading.Timer(0.2, robot.cancel.cancel)
    stop.start()
    try:
        error, took = timed(raised, set_gripper, robot.ctx, "left", "close")
    finally:
        stop.cancel()

    assert error.code is ErrorCode.CANCELLED
    assert took < 1.0


def test_a_close_reads_the_arms_only_when_the_memory_does_not_know_them(ctx, stub):
    in_receive = ArmsReport(ArmsState(left=ArmState("receive", False), right=HOME))
    stub.report = lambda read: in_receive if read else ArmsReport(ArmsState())
    stub.grippers["right"] = moved("right", "close", "empty", finger_m=0.0)

    set_gripper(ctx, "right", "close")

    # Unknown and verified: read once for the pose, then the final sentence.
    assert stub.states == [False, True, None]


def test_a_close_while_the_arms_are_unverified_takes_the_memory_alone(ctx, stub):
    # A read would recover the arms; the close goes to the backend, which refuses it.
    stub.report = ArmsReport(ArmsState(), unverified=ABANDONED, unverified_count=1)
    refusal = StepError(
        ErrorCode.PRECONDITION_FAILED, f"Cannot move left_gripper. {UNKNOWN_SENTENCE}"
    )
    stub.grippers["left"] = refusal

    assert raised(set_gripper, ctx, "left", "close").message == refusal.message
    assert stub.states == [False]


# get_arm_state over the fake


def test_get_arm_state_reads_the_arms_and_the_grippers(robot):
    move_arm_to_pose(robot.ctx, "hands_up")
    set_gripper(robot.ctx, "left", "close")
    set_gripper(robot.ctx, "right", "half")

    assert get_arm_state(robot.ctx).splitlines() == [
        "Left arm: hands_up, not safe for driving.",
        "Right arm: hands_up, not safe for driving.",
        "Left gripper: closed, empty (gap 0.000 m).",
        "Right gripper: half open (gap 0.036 m).",
        NOT_SAFE_SENTENCE,
    ]
    assert arm_calls(robot)[-1] == ("arms.read", {})


def test_get_arm_state_at_the_start_finds_both_arms_home_and_open(robot):
    assert get_arm_state(robot.ctx).splitlines() == [
        "Left arm: home, safe for driving.",
        "Right arm: home, safe for driving.",
        "Left gripper: open (gap 0.044 m).",
        "Right gripper: open (gap 0.044 m).",
        SAFE_SENTENCE,
    ]


# get_arm_state: what only the stub gives


def test_get_arm_state_names_no_pose_and_an_unknown_arm(ctx, stub):
    after = ArmsReport(ArmsState(left=None, right=NO_POSE))
    stub.reading = reading(after=after)

    lines = get_arm_state(ctx).splitlines()

    assert lines[:2] == [
        "Left arm: state unknown.",
        "Right arm: no named pose, not safe for driving.",
    ]
    assert lines[-1] == NOT_SAFE_SENTENCE


@pytest.mark.parametrize(("commanded", "said"), [("close", "close"), (None, "unknown")])
def test_a_gripper_that_cannot_be_read_falls_back_to_its_last_command(ctx, stub, commanded, said):
    unread = GripperReading("left", None, None, read_error="no answer", commanded=commanded)
    stub.reading = reading(grippers=(unread, GripperReading("right", 0.044, "open")))

    lines = get_arm_state(ctx).splitlines()

    assert lines[2:] == [
        f"Left gripper: could not be read; last command {said}.",
        "Right gripper: open (gap 0.044 m).",
        SAFE_SENTENCE,
    ]


@pytest.mark.parametrize(
    ("finger", "reads_as", "in_band", "said"),
    [
        (0.013, "close", True, "closed, holding something (gap 0.013 m)"),
        (0.001, "close", False, "closed, empty (gap 0.001 m)"),
        (0.036, "half", True, "half open (gap 0.036 m)"),
        (0.013, "close", None, "closed (gap 0.013 m)"),
    ],
)
def test_get_arm_state_says_whether_a_closed_gripper_holds_something(
    ctx, stub, finger, reads_as, in_band, said
):
    held = GripperReading("right", finger, reads_as, in_grasp_band=in_band)
    stub.reading = reading(grippers=(GripperReading("left", 0.044, "open"), held))

    assert get_arm_state(ctx).splitlines()[3] == f"Right gripper: {said}."


@pytest.mark.parametrize(
    ("how", "resolved"),
    [
        ("result", "The commander confirmed that the move of left_arm from an earlier call ended"),
        ("ended", "The commander confirmed that the move of left_arm from an earlier call ended"),
        ("rejected", "The commander rejected the move of left_arm from an earlier call"),
        (
            "unknown",
            (
                "The commander does not know the move of left_arm from an earlier call, as "
                "after a restart"
            ),
        ),
    ],
)
def test_get_arm_state_says_how_a_recovery_showed_an_earlier_goal_over(ctx, stub, how, resolved):
    before = ArmsReport(ArmsState(), unverified=ABANDONED, unverified_count=1)
    recovery = Recovery(count=1, goals=(("the move of left_arm", how),), grippers=())
    stub.reading = reading(before, cleared(recovery))

    lines = get_arm_state(ctx).splitlines()

    assert lines[0] == (
        f"The arms had to be checked, as {ABANDONED}. {resolved}, and two reads of both arms a "
        "moment apart agree. Arm and gripper tools can be used again."
    )
    assert lines[1:3] == ["Left arm: home, safe for driving.", "Right arm: home, safe for driving."]
    assert lines[-1] == SAFE_SENTENCE


@pytest.mark.parametrize(
    ("grippers", "checked"),
    [
        (("left_gripper",), "Two reads of both arms and the left gripper a moment apart agree."),
        (
            ("left_gripper", "right_gripper"),
            "Two reads of both arms and both grippers a moment apart agree.",
        ),
        ((), "Two reads of both arms a moment apart agree."),
    ],
)
def test_get_arm_state_says_which_groups_the_settled_read_checked(ctx, stub, grippers, checked):
    before = ArmsReport(ArmsState(), unverified=ABANDONED, unverified_count=3)
    recovery = Recovery(count=3, goals=(), grippers=grippers)
    stub.reading = reading(before, cleared(recovery))

    assert get_arm_state(ctx).splitlines()[0] == (
        f"The arms had to be checked, as {ABANDONED}. {checked} Arm and gripper tools can be used "
        "again."
    )


def test_get_arm_state_names_every_goal_a_recovery_resolved(ctx, stub):
    before = ArmsReport(ArmsState(), unverified=ABANDONED, unverified_count=2)
    goals = (("the move of left_arm", "unknown"), ("the left_gripper command", "result"))
    recovery = Recovery(count=2, goals=goals, grippers=("left_gripper",))
    stub.reading = reading(before, cleared(recovery))

    assert get_arm_state(ctx).splitlines()[0] == (
        f"The arms had to be checked, as {ABANDONED}. The commander does not know the move of "
        "left_arm from an earlier call, as after a restart. The commander confirmed that the "
        "left_gripper command from an earlier call ended, and two reads of both arms and the left "
        "gripper a moment apart agree. Arm and gripper tools can be used again."
    )


def test_a_recovery_before_the_call_is_not_reported_again(ctx, stub):
    # Verified at the start: the last recovery is an old one.
    old = Recovery(count=1, goals=(("the move of left_arm", "result"),), grippers=())
    report = dataclasses.replace(AT_HOME, unverified_count=1, recovery=old)
    stub.reading = reading(report, report)

    assert get_arm_state(ctx).splitlines()[0] == "Left arm: home, safe for driving."


def test_get_arm_state_says_when_the_arms_became_unverified_during_its_reads(ctx, stub):
    before = ArmsReport(ArmsState(), unverified=ABANDONED, unverified_count=1)
    after = ArmsReport(
        ArmsState(left=None, right=HOME), unverified=ACCEPTED_LATE, unverified_count=2
    )
    stub.reading = reading(before, after)

    lines = get_arm_state(ctx).splitlines()

    assert lines[:3] == [
        (
            f"While the arms were being read, {ACCEPTED_LATE}, so the arms have to be checked "
            "again. Call get_arm_state again in a few seconds."
        ),
        "Left arm: state unknown.",
        "Right arm: home, safe for driving.",
    ]
    assert lines[-1] == UNKNOWN_SENTENCE
    assert "can be used again" not in "\n".join(lines)


def test_get_arm_state_says_to_check_again_when_the_arms_were_verified_again_during_its_reads(
    ctx, stub
):
    # Another call recovered the arms again before the end of this one: the answer does
    # not take over what that other recovery checked, and never says the robot may drive.
    before = ArmsReport(ArmsState(), unverified=ABANDONED, unverified_count=1)
    recovery = Recovery(count=2, goals=(("the move of left_arm", "result"),), grippers=())
    stub.reading = reading(before, cleared(recovery))

    assert get_arm_state(ctx).splitlines() == [
        (
            "While the arms were being read, a goal from an earlier call was given up on or "
            "accepted late, so the arms have to be checked again. Call get_arm_state again in "
            "a few seconds."
        ),
        "Left arm: home, safe for driving.",
        "Right arm: home, safe for driving.",
        "Left gripper: open (gap 0.044 m).",
        "Right gripper: open (gap 0.044 m).",
        UNKNOWN_SENTENCE,
    ]


def test_get_arm_state_never_says_the_robot_may_drive_while_a_gripper_goal_is_unverified(
    ctx, stub
):
    # A gripper goal leaves the arms known, so the arms alone would say they are safe.
    cause = "the left_gripper command from an earlier call was given up on before its end was known"
    after = dataclasses.replace(AT_HOME, unverified=cause, unverified_count=1)
    stub.reading = reading(after, after)

    lines = get_arm_state(ctx).splitlines()

    assert lines[0].startswith(f"While the arms were being read, {cause}, so")
    assert lines[-1] == UNKNOWN_SENTENCE
    assert SAFE_SENTENCE not in lines


# get_arm_state: a read that fails


def unverified_now(stub: StubArms, cause: str | None = ABANDONED) -> None:
    stub.report = ArmsReport(ArmsState(left=HOME), unverified=cause, unverified_count=1)


def test_a_failed_recovery_says_why_the_arms_had_to_be_checked(ctx, stub):
    unverified_now(stub)
    still = (
        "An arm is still moving: openarm_left_joint1 moved 0.020 rad between two reads 0.3 s "
        "apart. Wait a few seconds, then try again."
    )
    stub.read_error = StepError(ErrorCode.PRECONDITION_FAILED, still)

    error = raised(get_arm_state, ctx)

    assert (error.code, error.message) == (
        ErrorCode.PRECONDITION_FAILED,
        f"The arms had to be checked, as {ABANDONED}. {still} {UNKNOWN_SENTENCE}",
    )
    # The cause and the sentence come from one report, read without asking the commander.
    assert stub.states == [False]


def test_a_failed_recovery_that_says_the_arms_may_still_be_moving_still_says_why(ctx, stub):
    unverified_now(stub)
    refused = (
        "The commander refused the cancel of the move of left_arm from an earlier call, so an "
        "arm may still be moving. Wait a few seconds, then try again."
    )
    stub.read_error = StepError(ErrorCode.PRECONDITION_FAILED, refused)

    assert raised(get_arm_state, ctx).message == (
        f"The arms had to be checked, as {ABANDONED}. {refused} {UNKNOWN_SENTENCE}"
    )


def test_a_read_that_names_the_cause_itself_gets_no_second_one(ctx, stub):
    unverified_now(stub, ACCEPTED_LATE)
    again = (
        f"While the arms were being checked, {ACCEPTED_LATE}. Wait a few seconds, then try again."
    )
    stub.read_error = StepError(ErrorCode.PRECONDITION_FAILED, again)

    assert raised(get_arm_state, ctx).message == f"{again} {UNKNOWN_SENTENCE}"


def test_a_stop_during_the_read_gets_no_cause(ctx, stub):
    unverified_now(stub)
    stub.read_error = StepError(ErrorCode.CANCELLED, "A stop or a reset interrupted the read")

    error = raised(get_arm_state, ctx)

    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        f"A stop or a reset interrupted the read. {UNKNOWN_SENTENCE}",
    )


def test_a_failed_read_of_verified_arms_ends_with_the_sentence(ctx, stub):
    stub.report = ArmsReport(ArmsState(left=HANDS_UP))
    nan = "The commander's reading of both_arms is not usable: openarm_left_joint1 = nan."
    stub.read_error = StepError(ErrorCode.INTERNAL, nan)

    error = raised(get_arm_state, ctx)

    assert type(error) is StepError
    assert (error.code, error.message) == (ErrorCode.INTERNAL, f"{nan} {NOT_SAFE_SENTENCE}")


# Without arms


def test_without_arm_poses_every_arm_atomic_is_unavailable():
    ctx = FakeRobot(world()).ctx
    calls = [
        lambda: list_arm_poses(ctx),
        lambda: move_arm_to_pose(ctx, "home"),
        lambda: set_gripper(ctx, "left", "open"),
        lambda: set_gripper(ctx, "both", "close"),
    ]
    for call in calls:
        error = raised(call)
        assert (error.code, error.message) == (ErrorCode.UNAVAILABLE, NO_ARMS)
    error = raised(get_arm_state, ctx)
    assert (error.code, error.message) == (ErrorCode.UNAVAILABLE, f"{NO_ARMS} {UNKNOWN_SENTENCE}")
