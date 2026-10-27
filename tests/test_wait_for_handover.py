"""wait_for_handover on fake_ctx: the refusals, the hold on the FakeClock, and the re-read.

The atomic holds the hand still and open in the receive pose for HOLD_S
(ADR-0059). It sends no goal and says nothing, so every case also checks
that no arm or gripper goal was sent.
"""

from collections.abc import Callable

import pytest

from walkie_agent.arm.state import NOT_SAFE_SENTENCE, SAFE_SENTENCE
from walkie_agent.testing import OUT, FakeRobot, fake_ctx
from walkie_agent.toolbox import Effect, StepError, ToolContext
from walkie_agent.tools.atomic import _arm as shared
from walkie_agent.tools.atomic import wait_for_handover
from walkie_agent.tools.atomic.wait_for_handover import HOLD_S

HELD = (
    "The left hand stayed open in receive for 5 s. Say the close line, then close the hand. "
    f"{NOT_SAFE_SENTENCE}"
)


def in_receive() -> tuple[ToolContext, FakeRobot]:
    """A ctx with the left arm in receive, the right at home and both hands open."""
    return fake_ctx(arms={"left": "receive", "right": "home"}, in_simulation=True)


def error_of(ctx: ToolContext, **args: str) -> StepError:
    with pytest.raises(StepError) as error:
        wait_for_handover.run(ctx, wait_for_handover.Args(**args))
    return error.value


def during_the_hold(robot: FakeRobot, at_s: float, act: Callable[[], None]) -> None:
    """Call `act` once, on the first wait that starts `at_s` or more into the hold."""
    clock = robot.clock
    wait = clock.wait
    # The hold sums 0.05 s waits, so allow for the float error of the sum.
    due = clock.monotonic() + at_s - 1e-9
    done = []

    def waiting(event, timeout_s):
        if not done and clock.monotonic() >= due:
            done.append(True)
            act()
        return wait(event, timeout_s)

    clock.wait = waiting  # type: ignore[method-assign]


def assert_no_goal(robot: FakeRobot) -> None:
    assert robot.arms.goals == [] and robot.arms.gripper_goals == []
    assert robot.gaze.stops == []


def test_the_meta_is_read_only_and_takes_no_hold():
    meta = wait_for_handover.META

    assert meta.effects == frozenset({Effect.R})
    assert meta.idempotent is True
    assert meta.requires == (shared.ARM,)
    assert 3 <= len(meta.questions) <= 5
    assert meta.examples == ({"arm": "left"},)
    # The hold is a set time, never the LLM's choice.
    assert set(wait_for_handover.Args.model_fields) == {"arm"}
    assert wait_for_handover.Args().arm == "left"


def test_an_open_hand_in_receive_is_held_for_hold_s_on_the_clock():
    ctx, robot = in_receive()
    start = robot.clock.monotonic()

    out = wait_for_handover.run(ctx, wait_for_handover.Args())

    assert out.summary == HELD
    assert (out.arm, out.waited_s) == ("left", HOLD_S)
    assert robot.clock.monotonic() - start == pytest.approx(HOLD_S)
    # Read before the hold and again after it.
    assert robot.calls == ["arms.state", "arms.finger", "arms.state", "arms.finger"]
    assert_no_goal(robot)


@pytest.mark.parametrize(
    ("arms", "arm", "where", "sentence"),
    [
        ("home", "left", "it is in 'home'", SAFE_SENTENCE),
        ({"left": "receive", "right": "home"}, "right", "it is in 'home'", NOT_SAFE_SENTENCE),
        ({"left": OUT, "right": "home"}, "left", "it is in no named pose", NOT_SAFE_SENTENCE),
    ],
)
def test_an_arm_outside_receive_is_refused_before_the_hold(arms, arm, where, sentence):
    ctx, robot = fake_ctx(arms=arms, in_simulation=True)
    start = robot.clock.monotonic()

    error = error_of(ctx, arm=arm)

    assert error.code == "PRECONDITION_FAILED"
    assert error.message == (
        f"The {arm} arm is not in the receive pose: {where}. Move the {arm} arm to receive "
        f"first. {sentence}"
    )
    assert error.data["arm"] == arm
    assert robot.clock.monotonic() == start
    # The hand is not read while the arm is out of the pose.
    assert "arms.finger" not in robot.calls
    assert_no_goal(robot)


@pytest.mark.parametrize(("state", "gap"), [("close", "0.000"), ("half", "0.036")])
def test_a_hand_that_is_not_open_is_refused_before_the_hold(state, gap):
    ctx, robot = in_receive()
    ctx.arms.gripper("left", state)
    robot.arms.gripper_goals.clear()
    start = robot.clock.monotonic()

    error = error_of(ctx)

    assert error.code == "PRECONDITION_FAILED"
    assert error.message == (
        f"The left hand is not open (gap {gap} m). Open the left hand first. {NOT_SAFE_SENTENCE}"
    )
    assert error.data == {"arm": "left", "finger_m": robot.arms.finger_m["left"]}
    assert robot.clock.monotonic() == start
    assert_no_goal(robot)


def test_a_hand_opened_again_after_a_close_is_held():
    # The retry: an empty close, then an open, then the second hold.
    ctx, robot = in_receive()
    ctx.arms.gripper("left", "close")
    ctx.arms.gripper("left", "open")

    out = wait_for_handover.run(ctx, wait_for_handover.Args())

    assert out.summary == HELD
    assert robot.arms.gripper_goals == [("left", "close"), ("left", "open")]


def test_a_stop_during_the_hold_cancels_it_and_reads_nothing_more():
    ctx, robot = in_receive()
    start = robot.clock.monotonic()
    during_the_hold(robot, 2.0, ctx.cancel.set)

    error = error_of(ctx)

    assert error.code == "CANCELLED"
    assert error.message == (
        "Stopped by reset during the wait for the hand-over. The left gripper did not move. "
        f"{NOT_SAFE_SENTENCE}"
    )
    assert robot.clock.monotonic() - start == pytest.approx(2.0)
    # The answer's sentence is the client's memory: no read after the stop.
    assert robot.calls == ["arms.state", "arms.finger", "arms.state"]
    assert_no_goal(robot)


def test_a_stop_before_the_call_cancels_it_before_the_hand_is_read():
    ctx, robot = in_receive()
    ctx.cancel.set()

    error = error_of(ctx)

    assert error.code == "CANCELLED"
    assert error.message == f"Stopped by reset before the left hand was read. {NOT_SAFE_SENTENCE}"
    assert_no_goal(robot)


def test_an_arm_that_leaves_receive_during_the_hold_fails_the_call():
    ctx, robot = in_receive()
    during_the_hold(robot, 2.0, lambda: robot.arms.pose.__setitem__("left", "home"))

    error = error_of(ctx)

    assert error.code == "PRECONDITION_FAILED"
    assert error.message == (
        "The left arm left the receive pose during the wait: it is in 'home'. Move the left "
        f"arm to receive first. {SAFE_SENTENCE}"
    )
    assert error.data == {"arm": "left", "pose": "home"}
    assert_no_goal(robot)


def test_a_hand_that_closes_during_the_hold_fails_the_call():
    ctx, robot = in_receive()
    during_the_hold(robot, 2.0, lambda: robot.arms.finger_m.__setitem__("left", 0.0))

    error = error_of(ctx)

    assert error.code == "PRECONDITION_FAILED"
    assert error.message == (
        "The left hand closed during the wait (gap 0.000 m). Open the left hand first. "
        f"{NOT_SAFE_SENTENCE}"
    )
    assert error.data == {"arm": "left", "finger_m": 0.0}
    assert_no_goal(robot)


def test_a_hand_that_cannot_be_read_fails_the_call_before_the_hold():
    ctx, robot = in_receive()
    robot.arms.read_failure = "MoveIt has no robot state"
    start = robot.clock.monotonic()

    error = error_of(ctx)

    assert error.code == "UNAVAILABLE"
    assert error.message == (
        "Could not read the left hand: The commander could not read the joints of "
        f"left_gripper: MoveIt has no robot state. {NOT_SAFE_SENTENCE}"
    )
    assert robot.clock.monotonic() == start
    assert_no_goal(robot)
