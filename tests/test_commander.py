"""CommanderClient against FakeCommander over ROS.

The fake serves the commander's names and types on the `connector` fixture's Fast DDS
domain, with the groups of config/robot_model.toml, so these tests cover the client's
message handling, its waits and its cancels the way the real commander would see them.

A deadline test puts the client on a test clock (module_clock), so a deadline passes only
when the test moves the clock on, once the fake has shown that the call waits where the
test expects. Real time is left for the waits the test clock does not cover: the slices
in which every wait watches the CancelToken, the wait for an answer to the call's own
cancel (the client's cancel_answer_s), which a test raises to 10 s when the answer comes
at once, the settled read's pause (SETTLE_S), and close()'s wait for a goal that has not
ended.

The helpers below (wait_until, Call, Spinner, ModuleClock) stand in for those of
experiment/hri-agent's conftest, which this tree does not have.
"""

import logging
import math
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from action_msgs.msg import GoalInfo, GoalStatus
from action_msgs.srv import CancelGoal
from rclpy.action.client import ClientGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.task import Future
from unique_identifier_msgs.msg import UUID

from walkie_agent.arm.pick import HandPose
from walkie_agent.arm.poses import RIGHT_JOINT2, load_arm_poses, load_robot_model
from walkie_agent.arm.results import Recovery
from walkie_agent.arm.state import (
    NOT_SAFE_SENTENCE,
    SAFE_SENTENCE,
    UNKNOWN_SENTENCE,
    ArmsState,
    ArmState,
    worst_joint,
)
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import GoalFailed, StepError
from walkie_agent.ros import commander as commander_module
from walkie_agent.ros.commander import CommanderClient
from walkie_agent.ros.policy import FORBIDDEN_NAMES
from walkie_agent.sim.fake_commander import (
    CONTROL_GRIPPER,
    SET_JOINT_POSITION,
    WHOLE_BODY_SUCCESS,
    FakeCommander,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "config" / "robot_model.toml"
POSES_PATH = PROJECT_ROOT / "config" / "arm_poses.toml"

GROUPS = ("left_arm", "right_arm", "both_arms", "left_gripper", "right_gripper")
# A hand target in front of the robot, as pick_object sends one.
HAND = HandPose(0.6, -0.2, 0.8, 0.0, math.pi / 2, 0.3, "map")
# The executor threads of the fake's executor. A held goal, a held cancel and a slow
# read each keep one, so a test can need several at once.
FAKE_THREADS = 8


# -- Test helpers -------------------------------------------------------------------


def _stack_of(thread: threading.Thread) -> str:
    """The stack of `thread` right now, for a failure message."""
    frame = sys._current_frames().get(thread.ident)
    if frame is None:
        return f"{thread.name}: no longer running\n"
    return f"{thread.name}:\n{''.join(traceback.format_stack(frame))}"


def wait_until(
    predicate: Callable[[], Any], timeout_s: float = 10.0, poll_s: float = 0.01
) -> bool:
    """Whether `predicate` came true within `timeout_s`. For positive waits only.

    It returns as soon as the predicate holds, so the bound can stay far above a stalled
    runner's delays. Do not use it to show that something stays false: that would wait
    out the whole bound.
    """
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)
    return True


class Call:
    """Run `fn(*args, **kwargs)` on a daemon thread, for a call a test has to watch."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._value: Any = None
        self._error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, args=(fn, args, kwargs), daemon=True)
        self.thread.start()

    def _run(self, fn: Callable[..., Any], args: tuple, kwargs: dict) -> None:
        try:
            self._value = fn(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - raised again by result(), in the test
            self._error = error

    def result(self, timeout_s: float = 10) -> Any:
        """What the call returned, or its exception raised here; fails if it runs on."""
        self.thread.join(timeout_s)
        assert not self.thread.is_alive(), (
            f"The call did not return within {timeout_s} s.\n{_stack_of(self.thread)}"
        )
        if self._error is not None:
            raise self._error
        return self._value


class Spinner:
    """Spin nodes on a MultiThreadedExecutor of the test's own, on a daemon thread.

    close() ends the spin before it destroys the nodes, then waits for the callbacks they
    held. rclpy's Executor.shutdown() destroys the executor's guard condition without
    waiting for the spin thread, which can then wait forever, and FakeCommander releases
    the goals and requests it holds only in destroy_node().
    """

    def __init__(self, *nodes: Any, num_threads: int = FAKE_THREADS) -> None:
        self.nodes = nodes
        self.executor = MultiThreadedExecutor(num_threads=num_threads)
        for node in nodes:
            self.executor.add_node(node)
        # No executor, so its set_result() wakes the spin inline.
        self._stop = Future()
        self._thread = threading.Thread(
            target=self.executor.spin_until_future_complete, args=(self._stop,), daemon=True
        )
        self._thread.start()

    def close(self, timeout_s: float = 10) -> None:
        """Stop the spin, destroy the nodes, and wait for the handlers still running."""
        self._stop.set_result(True)
        self._thread.join(timeout_s)
        assert not self._thread.is_alive(), (
            f"A Spinner's thread did not stop within {timeout_s} s.\n{_stack_of(self._thread)}"
        )
        for node in self.nodes:
            node.destroy_node()
        handlers = list(self.executor._futures)
        try:
            assert wait_until(
                lambda: all(handler.done() or handler.cancelled() for handler in handlers),
                timeout_s,
            ), f"A Spinner's callback did not end within {timeout_s} s of its nodes closing."
        finally:
            self.executor.shutdown()
            # Otherwise the idle pool workers of every executor stay until exit.
            self.executor._executor.shutdown(wait=False)


class ModuleClock:
    """A manual monotonic() for a module's `time` global, which a test moves on.

    Every other attribute is the real time module's.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._now = 1000.0

    def monotonic(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


@pytest.fixture
def module_clock(monkeypatch):
    """A function that sets `module.time` to a new ModuleClock and returns it.

    At teardown every clock moves on by 1e6 s, so no call stays parked on one.
    """
    clocks = []

    def patch(module) -> ModuleClock:
        clock = ModuleClock()
        monkeypatch.setattr(module, "time", clock)
        clocks.append(clock)
        return clock

    yield patch
    for clock in clocks:
        clock.advance(1e6)


# -- Fixtures -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return load_robot_model(MODEL_PATH)


@pytest.fixture(scope="module")
def poses():
    return load_arm_poses(POSES_PATH, MODEL_PATH)


@pytest.fixture
def fake_commander(ros, model):
    node = FakeCommander(model, move_time_s=0.2, gripper_time_s=0.05)
    spinner = Spinner(node)
    yield node
    spinner.close()


@pytest.fixture
def token():
    return CancelToken()


@pytest.fixture
def commander(connector, token, poses):
    client = CommanderClient(connector, token, poses)
    yield client
    client.destroy()


@pytest.fixture
def client(fake_commander, commander):
    """A CommanderClient that has discovered the fake commander."""
    assert wait_until(commander.available), "The client never discovered the fake commander."
    return commander


def table(poses, pose: str, arm: str) -> dict[str, float]:
    return dict(poses.poses[pose].tables[arm])


def move(client, group: str, targets: dict[str, float], timeout_s=5.0):
    """Send `targets` in the commander's joint order, as move_arm_to_pose does."""
    order = client.joint_names(group)
    return client.set_joint_position(group, [targets[j] for j in order], timeout_s)


def accepted_handle(client):
    """The goal handle of the client's running goal once the commander accepted it, else None.

    Not None only once the client asked for the result, so its call then waits for the
    result, and a deadline that passes ends that wait.
    """
    goal = client._goal
    if goal is None or goal.result is None:
        return None
    return goal.sent.result()


def goal_status(client) -> int | None:
    """The status the client last heard for its running goal, None before the acceptance.

    GoalStatus.STATUS_SUCCEEDED once the commander ended the goal, whether or not its
    result was sent.
    """
    handle = accepted_handle(client)
    return None if handle is None else handle.status


def results_asked(goal) -> int:
    """How many times the client asked the commander for the result of `goal`, its record.

    Recovery asks again once the commander answered its cancel accepted or ended
    (_await_end), so a count that grows shows that recovery had that answer and waits
    for the result.
    """
    return (goal.result is not None) + len(goal.more_results)


def readiness_checked(monkeypatch, entity, name: str) -> threading.Event:
    """An event set by the next call of `entity`'s discovery check `name`, which still answers.

    _wait_ready() takes its time limit before its first check, so once the event is set,
    a test clock moved on past the client's ready_wait_s ends that wait.
    """
    checked = threading.Event()
    check = getattr(entity, name)

    def check_and_tell() -> bool:
        checked.set()
        return check()

    monkeypatch.setattr(entity, name, check_and_tell)
    return checked


def cancel_answer(error: StepError) -> str:
    """The answer to the call's own cancel that the message of `error` ends with.

    A key of CANCEL_OUTCOMES, or "pending" when the commander had not answered the send,
    so no cancel was sent.
    """
    for answer, words in commander_module.CANCEL_OUTCOMES.items():
        if any(error.message.endswith(f"; {phrase}.") for phrase in words):
            return answer
    if any(error.message.endswith(f"; {phrase}.") for phrase in commander_module.PENDING_OUTCOMES):
        return "pending"
    raise AssertionError(f"No answer to a cancel in {error.message!r}.")


def stop_matched(client) -> bool:
    """True once the client's stop publisher is matched with a subscription, the fake's.

    The stop is published once, on a volatile topic, so a stop sent before discovery
    matched the two would be lost.
    """
    return client._stop_publisher.get_subscription_count() > 0


# -- Reads ------------------------------------------------------------------------


def test_joint_states_lists_every_group_in_the_commanders_order(client, model, poses):
    for group in GROUPS:
        assert list(client.joint_states(group)) == list(model.groups[group])

    assert client.joint_states("left_arm") == pytest.approx(table(poses, "home", "left"))
    # The fingers start open.
    assert client.joint_states("right_gripper") == {
        "openarm_right_finger_joint1": pytest.approx(0.044)
    }


def test_joint_names_are_read_once_per_group(client, fake_commander):
    names = client.joint_names("right_arm")

    assert client.joint_names("right_arm") == names
    assert fake_commander.joint_state_requests == ["right_arm"]


def test_read_arms_and_last_state(client, fake_commander, poses):
    assert client.last_state == ArmsState()
    assert client.last_state.sentence() == UNKNOWN_SENTENCE

    arms = client.read_arms()

    assert arms == ArmsState(left=ArmState("home", True), right=ArmState("home", True))
    assert arms.sentence() == SAFE_SENTENCE
    assert client.last_state == arms
    assert fake_commander.joint_state_requests == ["both_arms"]

    # A read of one arm updates that arm and keeps the other's last state.
    fake_commander.teleport(table(poses, "standby", "left"))
    client.joint_states("left_arm")
    assert client.last_state == ArmsState(left=ArmState("standby", False), right=arms.right)
    assert client.last_state.sentence() == NOT_SAFE_SENTENCE
    # A gripper read says nothing about the arms.
    client.joint_states("left_gripper")
    assert client.last_state.left == ArmState("standby", False)


def test_a_slow_read_times_out(client, fake_commander, module_clock, caplog):
    clock = module_clock(commander_module)
    # The commander holds its answer until the test lets it go.
    held = threading.Event()
    fake_commander.joint_states_gate = held

    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        reading = Call(client.joint_states, "both_arms", timeout_s=0.2)
        assert wait_until(lambda: fake_commander.joint_state_requests == ["both_arms"])
        clock.advance(0.3)
        with pytest.raises(StepError) as error:
            reading.result()

    # It returned while the commander still held the answer.
    assert not held.is_set()
    assert error.value.code == "TIMEOUT"
    assert "/get_joint_states did not answer within 0.2 s" in error.value.message
    # An unanswered call is what a stale my_robot_interfaces pin looks like
    # under rmw_zenoh (ADR-0035); the hint is for the log, not the LLM.
    assert "my_robot_interfaces" not in error.value.message
    assert commander_module.STALE_PIN_HINT in caplog.text
    held.set()
    fake_commander.joint_states_gate = None
    assert client.read_arms().travel_safe is True


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_a_reading_that_is_not_a_number_is_internal_and_forgets_the_arms(
    client, fake_commander, value
):
    client.read_arms()
    # Every other joint at home but joint4 far off: a NaN passes every
    # tolerance check, so a match that let it through would call the arm home.
    fake_commander.teleport({"openarm_left_joint1": value, "openarm_left_joint4": 2.0})

    with pytest.raises(StepError) as error:
        client.read_arms()

    assert error.value.code == "INTERNAL"
    assert error.value.message == (
        f"The commander's reading of both_arms is not usable: openarm_left_joint1 = {value}."
    )
    assert client.last_state == ArmsState()
    # A read of the other arm alone says nothing about this one.
    client.joint_states("right_arm")
    assert client.last_state == ArmsState(right=ArmState("home", True))


def test_a_reading_that_lacks_an_arms_joints_is_internal_and_forgets_that_arm(
    client, fake_commander, model
):
    home = client.read_arms()
    # Only the left arm's joints, as in a malformed answer.
    fake_commander.missing_joints = set(model.groups["right_arm"])
    # openarm_right_joint2 is not missed: right_joint2_fixed leaves it out.
    missing = sorted(set(model.groups["right_arm"]) - {RIGHT_JOINT2})

    with pytest.raises(StepError) as error:
        client.read_arms()

    assert error.value.code == "INTERNAL"
    assert error.value.message == (
        f"The commander's reading of both_arms lacks {', '.join(missing)}, so it cannot say "
        "where the right arm is."
    )
    assert client.last_state == ArmsState(left=home.left, right=None)

    # A read of one arm that lacks a joint forgets that arm, and keeps the other.
    fake_commander.missing_joints = {"openarm_left_joint3"}
    assert "openarm_left_joint3" not in client.joint_states("left_arm")
    assert client.last_state == ArmsState()
    client.joint_states("right_arm")
    assert client.last_state == ArmsState(right=home.right)

    fake_commander.missing_joints = {"openarm_left_joint3", "openarm_right_joint5"}
    with pytest.raises(StepError) as error:
        client.read_arms()
    assert error.value.message == (
        "The commander's reading of both_arms lacks openarm_left_joint3, openarm_right_joint5, "
        "so it cannot say where either arm is."
    )
    assert client.last_state == ArmsState()


def test_the_joint_order_of_a_reading_that_lacks_a_joint_is_not_kept(
    client, fake_commander, model
):
    fake_commander.missing_joints = {"openarm_left_joint3"}
    client.joint_states("left_arm")
    fake_commander.missing_joints = set()
    requests = len(fake_commander.joint_state_requests)

    # Read again, not taken from the incomplete reading.
    assert set(client.joint_names("left_arm")) == set(model.groups["left_arm"])
    assert len(fake_commander.joint_state_requests) == requests + 1
    # A complete reading is kept.
    client.joint_names("left_arm")
    assert len(fake_commander.joint_state_requests) == requests + 1


# -- Joint goals ------------------------------------------------------------------


def test_a_move_reaches_the_pose_and_the_read_matches_it(client, fake_commander, poses):
    standby = table(poses, "standby", "left")

    assert move(client, "left_arm", standby) == "Success"

    group, sent = fake_commander.joint_goals[-1]
    assert (group, sent) == ("left_arm", tuple(standby.values()))
    reading = client.joint_states("left_arm")
    assert worst_joint(standby, reading)[1] < 1e-9
    arms = client.read_arms()
    assert arms == ArmsState(left=ArmState("standby", False), right=ArmState("home", True))
    assert client.last_state == arms


def test_a_move_follows_the_order_the_commander_reports(client, fake_commander, poses):
    fake_commander.reverse_joint_order = True
    hands_up = table(poses, "hands_up", "right")

    assert client.joint_names("right_arm")[0] == "openarm_right_joint7"
    assert move(client, "right_arm", hands_up) == "Success"

    assert client.read_arms().right == ArmState("hands_up", False)


def test_a_planning_failure_moves_nothing(client, fake_commander, poses):
    fake_commander.plan_failure = True
    before = fake_commander.joints()

    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"))

    # The commander saw the goal end, so the arms are not moving under it.
    assert isinstance(error.value, GoalFailed)
    assert error.value.code == "SKILL_FAILED"
    assert error.value.message == (
        "MoveIt found no path for left_arm (Planning Failed). The arms did not move."
    )
    assert fake_commander.joints() == before


def test_an_execution_failure_stops_part_way(client, fake_commander, poses):
    fake_commander.execution_failure = True
    standby = table(poses, "standby", "left")

    with pytest.raises(StepError) as error:
        move(client, "left_arm", standby)

    assert isinstance(error.value, GoalFailed)
    assert error.value.code == "SKILL_FAILED"
    assert error.value.message == (
        "The motion of left_arm stopped before the target (Execution Failed). The arms may "
        "have stopped part way."
    )
    assert client.read_arms().left == ArmState(None, False)


def test_plan_only_reports_success_without_moving(client, fake_commander, poses):
    fake_commander.plan_only = True

    assert move(client, "left_arm", table(poses, "standby", "left")) == "Plan Only"

    # The read after the goal shows that nothing moved (ADR-0024).
    assert client.read_arms().left == ArmState("home", True)


def test_a_joint_left_off_shows_in_the_read(client, fake_commander, poses):
    fake_commander.joint_errors = {"openarm_left_joint4": 0.31}
    standby = table(poses, "standby", "left")

    assert move(client, "left_arm", standby) == "Success"

    joint, error = worst_joint(standby, client.joint_states("left_arm"))
    assert (joint, error) == ("openarm_left_joint4", pytest.approx(0.31))
    assert client.read_arms().left == ArmState(None, False)


def test_a_rejected_goal_is_internal_and_drops_the_joint_order(client, fake_commander, poses):
    fake_commander.reject_goals = True

    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "home", "left"))

    assert error.value.code == "INTERNAL"
    assert "rejected the goal for left_arm with 7 positions" in error.value.message
    # The order is read again, in case walkie-ros restarted with other groups.
    requests = len(fake_commander.joint_state_requests)
    client.joint_names("left_arm")
    assert len(fake_commander.joint_state_requests) == requests + 1

    # The commander's own rule: a count that differs from the group's joints.
    fake_commander.reject_goals = False
    with pytest.raises(StepError) as error:
        client.set_joint_position("left_arm", [0.0] * 6, 5.0)
    assert error.value.code == "INTERNAL"
    assert "rejected the goal for left_arm with 6 positions" in error.value.message


def test_a_goal_never_answered_times_out_and_is_cancelled(
    client, fake_commander, poses, monkeypatch, module_clock
):
    clock = module_clock(commander_module)
    # The commander answers the cancel at once; the call waits for that
    # answer in real time.
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.never_answer = True
    client.joint_names("left_arm")

    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.5)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.6)
    with pytest.raises(StepError) as error:
        moving.result()

    assert error.value.code == "TIMEOUT"
    assert error.value.message == (
        "The move of left_arm did not finish within 0.5 s; the arms were told to stop."
    )
    assert cancel_answer(error.value) == "accepted"
    assert fake_commander.cancel_requests == 1
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])
    assert wait_until(lambda: client.busy() is None)


def refused(call) -> StepError:
    with pytest.raises(StepError) as error:
        call()
    assert error.value.code == "PRECONDITION_FAILED"
    return error.value


def test_a_goal_accepted_after_its_call_gave_up_holds_off_every_call_until_it_ends(
    client, fake_commander, poses, module_clock, caplog
):
    client.joint_names("left_arm")
    before = fake_commander.joints()
    clock = module_clock(commander_module)
    # The commander answers the send only once the test lets it. Accepted,
    # the goal plans until its cancel is requested, and its result is held
    # until the test lets it go.
    accept = threading.Event()
    fake_commander.accept_gate = accept
    fake_commander.plan_until_cancelled = True
    result = threading.Event()
    fake_commander.result_gate = result
    calls = (
        lambda: client.read_arms(),
        lambda: client.control_gripper("left_gripper", 0.0),
    )

    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.2)
        assert wait_until(lambda: fake_commander.joint_goals)
        clock.advance(0.3)
        with pytest.raises(StepError) as error:
            moving.result()

        assert error.value.code == "TIMEOUT"
        assert error.value.message == (
            "The move of left_arm did not finish within 0.2 s; if the commander accepts it "
            "later, it is cancelled at once."
        )
        # No cancel was sent: the commander had not answered the send.
        assert cancel_answer(error.value) == "pending"
        assert commander_module.STALE_PIN_HINT in caplog.text
        # Not answered yet: the commander may still accept it and move the arm.
        assert client.busy() == "the move of left_arm from an earlier call"
        for call in calls:
            assert refused(call).message.endswith(
                ": the move of left_arm from an earlier call has not ended yet. "
                "Try again in a few seconds."
            )
        # Accepted, and cancelled at once; still refused until its result.
        accept.set()
        assert wait_until(lambda: fake_commander.cancel_requests == 1)
        for call in calls:
            refused(call)
        result.set()
        assert wait_until(lambda: client.busy() is None)

    assert "accepted the move of left_arm after its call gave up" in caplog.text
    # Cancelled while it planned, so the arm never moved.
    assert fake_commander.results == [("left_arm", "Canceled")]
    assert fake_commander.joints() == before
    assert fake_commander.gripper_goals == []
    # It was outstanding, not abandoned, when the commander accepted it, so
    # the arms stay verified.
    assert client.unverified is None
    assert client.read_arms().left == ArmState("home", True)


def test_a_gripper_goal_accepted_after_its_call_gave_up_still_moves_the_fingers(
    client, fake_commander, module_clock
):
    clock = module_clock(commander_module)
    # The commander answers the send only once the test lets it, and the
    # fingers then hold part way until the test lets them finish.
    accept = threading.Event()
    fake_commander.accept_gate = accept
    ramp = threading.Event()
    fake_commander.ramp_gate = ramp

    closing = Call(client.control_gripper, "left_gripper", 0.0, timeout_s=0.2)
    assert wait_until(lambda: fake_commander.gripper_goals)
    clock.advance(0.3)
    with pytest.raises(StepError) as error:
        closing.result()

    assert error.value.code == "TIMEOUT"
    assert error.value.message == (
        "The left_gripper command did not finish within 0.2 s; if the commander accepts it "
        "later, the gripper still makes that motion."
    )
    assert client.busy() == "the left_gripper command from an earlier call"
    record = client._goal
    # Accepted and cancelled, but the fingers finish their ramp first. The
    # answer to the cancel reaching the client shows that the commander
    # has it, so the ramp ends with the cancel requested.
    accept.set()
    assert wait_until(lambda: record.cancels and record.cancels[-1].done())
    assert client.busy() == "the left_gripper command from an earlier call"
    ramp.set()
    assert wait_until(lambda: client.busy() is None)
    assert fake_commander.cancel_requests == 1
    assert fake_commander.results == [("left_gripper", "Canceled")]
    assert fake_commander.joints()["openarm_left_finger_joint1"] == 0.0


def test_a_goal_accepted_after_the_client_was_destroyed_sends_nothing(
    connector, fake_commander, poses, monkeypatch, module_clock, caplog
):
    clock = module_clock(commander_module)
    client = CommanderClient(connector, CancelToken(), poses)
    assert wait_until(client.available), "The client never discovered the fake commander."
    # The commander's answer reaches the client, and the test holds it on its
    # way to the client's callbacks, as when the executor has taken an answer
    # as destroy() runs. destroy() takes the action client off the node, so an
    # answer still held by the fake (accept_gate) would never arrive at all.
    send_goal_async = client._control_gripper.send_goal_async
    sent = []
    held = Future()

    def send_and_hold_the_answer(goal, **kwargs):
        sent.append(send_goal_async(goal, **kwargs))
        return held

    monkeypatch.setattr(client._control_gripper, "send_goal_async", send_and_hold_the_answer)
    closing = Call(client.control_gripper, "left_gripper", 0.0, timeout_s=0.2)
    assert wait_until(lambda: sent and sent[0].done())
    clock.advance(0.3)
    with pytest.raises(StepError) as error:
        closing.result()
    assert cancel_answer(error.value) == "pending"
    record = client._goal

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=commander_module.__name__):
        client.destroy()
        # This future has no executor, so its callbacks run here, inline.
        held.set_result(sent[0].result())

    assert [
        (log.levelno, log.getMessage())
        for log in caplog.records
        if log.name == commander_module.__name__
    ] == [
        (
            logging.DEBUG,
            (
                "The commander accepted the left_gripper command after its call gave up, and "
                "the client was already destroyed, so nothing was sent."
            ),
        )
    ]
    assert record.cancels == []
    assert record.result is None
    assert fake_commander.cancel_requests == 0


# -- Abandoned goals and recovery ---------------------------------------------------

ABANDONED = "the move of left_arm from an earlier call was given up on before its end was known"
ACCEPTED_LATE = (
    "the commander accepted the move of left_arm from an earlier call late and was asked to "
    "cancel it"
)


def unverified_refusal(action: str, cause: str) -> str:
    return (
        f"Cannot {action}: an arm or a gripper may still be moving, as {cause}. Call "
        "get_arm_state to check the arms, then try again."
    )


def arm_moved(fake_commander, poses, pose: str, arm: str):
    """A check that is true once `arm` left its current pose for `pose`, by its largest change."""
    start = fake_commander.joints()
    target = table(poses, pose, arm)
    joint = max(target, key=lambda name: abs(target[name] - start[name]))
    return lambda: abs(fake_commander.joints()[joint] - start[joint]) > 1e-3


def restart(client, fake_commander, name: str = SET_JOINT_POSITION) -> None:
    """Restart the fake's action server `name`, as when the commander restarts.

    The new server does not know the goals of the old one, which go on as the fake's
    docstring says.
    """
    fake_commander.vanish(name)
    assert wait_until(lambda: not client.available())
    fake_commander.come_back()
    assert wait_until(client.available)


def test_a_goal_whose_result_never_comes_is_abandoned_after_the_limit(
    client, fake_commander, poses, monkeypatch, module_clock, caplog
):
    # As when the commander restarted: its servers are still discovered,
    # but the old goal's result never comes.
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.move_time_s = 5.0
    fake_commander.result_delay_s = math.inf
    client.read_arms()
    client.joint_names("left_arm")
    client.joint_names("right_arm")

    def in_no_pose() -> bool:
        joints = fake_commander.joints()
        return ArmsState.from_joints(joints, poses.poses).left == ArmState(None, False)

    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    # Given up on part way, out of every named pose.
    assert wait_until(lambda: accepted_handle(client) is not None and in_no_pose())
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()

    assert error.value.message == (
        "The move of left_arm did not finish within 0.3 s; the arms were told to stop."
    )
    record = client._goal
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])
    refused(lambda: client.read_arms())
    # Kept until abandon_after_s has passed since its call gave up.
    clock.advance(client._abandon_after_s - 1)
    assert client.busy() == "the move of left_arm from an earlier call"
    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        clock.advance(2)
        assert client.busy() is None
    assert (
        "Abandoned the move of left_arm: no result came within 60 s after its call gave up"
        in caplog.text
    )
    # Calls are taken again, but the goal's end is unknown: the arms are
    # unverified, and every goal is refused until a settled read.
    assert client.unverified == ABANDONED
    assert client.last_state.left is None
    for call, action in (
        (lambda: move(client, "right_arm", table(poses, "standby", "right")), "move right_arm"),
        (lambda: client.control_gripper("left_gripper", 0.0), "move left_gripper"),
    ):
        assert refused(call).message == unverified_refusal(action, ABANDONED)
    assert len(fake_commander.joint_goals) == 1
    assert fake_commander.gripper_goals == []
    # One read does not verify the arms.
    client.joint_states("left_arm")
    assert client.unverified == ABANDONED
    # read_arms recovers. The commander answers its cancel that the goal
    # ended, but it accepted the call's cancel before, and then only the
    # result shows that the goal ended. Recovery asks for it, and its wait
    # ends at the read's deadline.
    recovering = Call(client.read_arms)
    assert wait_until(lambda: results_asked(record) == 2)
    clock.advance(client._read_timeout_s + 1)
    error = refused(recovering.result)
    assert error.message == (
        "The commander accepted the cancel of the move of left_arm from an earlier call, and "
        "its end has not come yet. Wait a few seconds, then try again."
    )
    assert client.unverified == ABANDONED
    # A restarted commander does not know the goal, which resolves it, and
    # the settled read shows the arm stopped part way.
    restart(client, fake_commander)
    count = client.unverified_count
    requests = len(fake_commander.joint_state_requests)
    assert client.read_arms().left == ArmState(None, False)
    assert fake_commander.joint_state_requests[requests:] == ["both_arms", "both_arms"]
    assert client.unverified is None
    assert client.last_recovery == Recovery(count, (("the move of left_arm", "unknown"),), ())
    fake_commander.move_time_s = 0.2
    fake_commander.result_delay_s = 0.0
    assert move(client, "left_arm", table(poses, "home", "left")) == "Success"


def test_a_goal_whose_server_stays_away_is_abandoned(
    client, fake_commander, poses, monkeypatch, module_clock, caplog
):
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.never_answer = True
    fake_commander.reject_cancels = True
    client.read_arms()
    client.joint_names("left_arm")

    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()

    assert error.value.code == "TIMEOUT"
    assert cancel_answer(error.value) == "refused"
    record = client._goal
    refused(lambda: client.read_arms())
    assert client.busy() == "the move of left_arm from an earlier call"

    fake_commander.vanish()
    assert wait_until(lambda: not client._set_joint_position.server_is_ready())
    # One check that finds the server missing does not abandon the goal:
    # discovery can drop for a moment while the commander still runs it.
    assert client.busy() == "the move of left_arm from an earlier call"
    clock.advance(client._absent_abandon_s - 1)
    assert client.busy() == "the move of left_arm from an earlier call"
    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        clock.advance(2)
        assert client.busy() is None
    assert (
        "Abandoned the move of left_arm: its action server has not been discovered for 5 s"
        in caplog.text
    )
    assert client.unverified == ABANDONED
    assert client.last_state.left is None
    error = refused(lambda: move(client, "left_arm", table(poses, "home", "left")))
    assert error.message == unverified_refusal("move left_arm", ABANDONED)
    # Reads are taken again and fail as without a commander, once the wait
    # for the service has passed ready_wait_s. The service can leave
    # discovery later than the action server, and a read sent while it is
    # still discovered is never answered.
    assert wait_until(lambda: not client._joint_states.service_is_ready())
    asked = readiness_checked(monkeypatch, client._joint_states, "service_is_ready")
    reading = Call(client.joint_states, "left_arm")
    assert asked.wait(10)
    clock.advance(client._ready_wait_s + 1)
    with pytest.raises(StepError) as error:
        reading.result()
    assert error.value.code == "UNAVAILABLE"
    # Nothing answers recovery's cancel, so the arms stay unverified once
    # cancel_answer_s has passed since it was sent.
    monkeypatch.setattr(client, "_cancel_answer_s", 1.0)
    sent = record.cancel_sent_at
    recovering = Call(client.read_arms)
    assert wait_until(lambda: record.cancel_sent_at != sent)
    clock.advance(client._cancel_answer_s + 1)
    error = refused(recovering.result)
    assert error.message == (
        "The commander did not answer a cancel of the move of left_arm from an earlier call in "
        "time, so an arm may still be moving. Wait a few seconds, then try again; if it fails "
        "again, tell the person that the arms cannot be checked."
    )
    assert client.unverified == ABANDONED


def test_a_goal_whose_server_comes_back_is_kept_until_the_time_limit(
    client, fake_commander, poses, monkeypatch, module_clock, caplog
):
    # As when the commander restarts: its servers go for a moment, and the
    # new ones do not know the old goal, whose result never comes.
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.never_answer = True
    fake_commander.reject_cancels = True
    client.joint_names("left_arm")
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "refused"

    fake_commander.vanish()
    assert wait_until(lambda: not client._set_joint_position.server_is_ready())
    assert client.busy() == "the move of left_arm from an earlier call"
    fake_commander.come_back()
    # Checked while the servers are away and once they are back: an absence
    # shorter than absent_abandon_s abandons nothing.
    assert wait_until(lambda: client.busy() is not None and client.available())
    # Discovered again, so the absence no longer counts, however short the
    # limit is now and however long ago the servers went.
    monkeypatch.setattr(client, "_absent_abandon_s", 0.01)
    clock.advance(1.0)
    assert client.busy() == "the move of left_arm from an earlier call"
    assert client.unverified is None

    # The new servers never send the old goal's result, so only the time
    # limit ends the wait.
    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        clock.advance(client._abandon_after_s)
        assert client.busy() is None
    assert (
        "Abandoned the move of left_arm: no result came within 60 s after its call gave up"
        in caplog.text
    )
    assert client.unverified == ABANDONED
    # The old server's goal never ended.
    assert fake_commander.results == []


def test_a_settled_read_refuses_while_an_arm_still_moves(
    client, fake_commander, poses, monkeypatch, module_clock
):
    # The commander refuses the call's cancel and restarts while the goal
    # plans. A joint then moves between the settled read's two reads, as a
    # trajectory that move_group still executes would move it, which the
    # new commander does not know.
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    plan = threading.Event()
    fake_commander.plan_gate = plan
    fake_commander.reject_cancels = True
    client.read_arms()
    client.joint_names("left_arm")
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "refused"
    restart(client, fake_commander)
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    assert client.unverified == ABANDONED
    start = len(fake_commander.joint_state_requests)

    def move_joint(_group: str, index: int) -> None:
        # Before the answer to the settled read's second read.
        if index == start + 1:
            joint = "openarm_left_joint4"
            fake_commander.teleport({joint: fake_commander.joints()[joint] + 0.05})

    fake_commander.joint_states_hook = move_joint

    error = refused(lambda: client.read_arms())

    assert error.message == (
        "An arm is still moving: openarm_left_joint4 moved 0.050 rad between two reads 0.3 s "
        "apart. Wait a few seconds, then try again."
    )
    assert fake_commander.joint_state_requests[start:] == ["both_arms", "both_arms"]
    assert client.unverified == ABANDONED
    assert client.last_state == ArmsState()
    # Once the move ends, the next settled read shows the arms still.
    fake_commander.joint_states_hook = None
    plan.set()
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Success")])
    assert client.read_arms().left == ArmState("standby", False)
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "unknown"),)


@pytest.mark.parametrize("call_cancel", ["accepted", "refused"])
def test_recovery_waits_for_the_result_of_a_goal_that_still_plans(
    client, fake_commander, poses, monkeypatch, module_clock, call_cancel
):
    # The goal plans until the test lets it finish, so the arms are still
    # meanwhile. The commander accepts the call's cancel, or refuses it and
    # accepts recovery's.
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    plan = threading.Event()
    fake_commander.plan_gate = plan
    fake_commander.reject_cancels = call_cancel == "refused"
    client.read_arms()
    client.joint_names("left_arm")
    before = fake_commander.joints()
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == call_cancel
    record = client._goal
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    assert client.unverified == ABANDONED
    fake_commander.reject_cancels = False

    # Recovery has the answer to its cancel once it asks for the result
    # again, and waits for the result until the read's deadline.
    recovering = Call(client.read_arms)
    assert wait_until(lambda: results_asked(record) == 2)
    clock.advance(client._read_timeout_s + 1)
    error = refused(recovering.result)

    # After an accepted cancel, the commander answers another one as if the
    # goal had ended, without asking its cancel callback. Only the result
    # shows that the goal ended.
    assert error.message == (
        "The commander accepted the cancel of the move of left_arm from an earlier call, and "
        "its end has not come yet. Wait a few seconds, then try again."
    )
    assert client.unverified == ABANDONED
    assert fake_commander.cancel_requests == (1 if call_cancel == "accepted" else 2)
    # Once the plan is done, the goal sees the cancel and ends without moving.
    plan.set()
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Canceled")])
    count = client.unverified_count
    assert client.read_arms().left == ArmState("home", True)
    assert client.unverified is None
    assert client.last_recovery == Recovery(count, (("the move of left_arm", "result"),), ())
    assert fake_commander.joints() == before


@pytest.mark.parametrize("answer", ["refused", "no answer"])
def test_recovery_refuses_while_the_cancel_of_a_goal_is_refused_or_unanswered(
    client, fake_commander, poses, monkeypatch, module_clock, caplog, answer
):
    clock = module_clock(commander_module)
    # The call waits cancel_answer_s for the answer to its own cancel, which
    # a refusal ends at once.
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0 if answer == "refused" else 0.2)
    # Held until a cancel of it is accepted.
    fake_commander.never_answer = True
    held = threading.Event()
    if answer == "refused":
        fake_commander.reject_cancels = True
    else:
        fake_commander.cancel_gate = held
    client.joint_names("left_arm")
    client.joint_names("right_arm")
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == answer
    record = client._goal
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None

    sent = record.cancel_sent_at
    recovering = Call(client.read_arms)
    if answer == "no answer":
        # Recovery's cancel is held too, and the wait for its answer ends
        # once cancel_answer_s has passed since it was sent.
        assert wait_until(lambda: record.cancel_sent_at != sent)
        clock.advance(client._cancel_answer_s + 1)
    error = refused(recovering.result)

    reply = (
        "refused the cancel of the move of left_arm from an earlier call"
        if answer == "refused"
        else "did not answer a cancel of the move of left_arm from an earlier call in time"
    )
    assert error.message == (
        f"The commander {reply}, so an arm may still be moving. Wait a few seconds, then try "
        "again; if it fails again, tell the person that the arms cannot be checked."
    )
    assert client.unverified == ABANDONED
    # No newer goal starts while the old one may still be active.
    for call, action in (
        (lambda: move(client, "right_arm", table(poses, "standby", "right")), "move right_arm"),
        (lambda: client.control_gripper("left_gripper", 0.0), "move left_gripper"),
    ):
        assert refused(call).message == unverified_refusal(action, ABANDONED)
    assert len(fake_commander.joint_goals) == 1
    assert fake_commander.gripper_goals == []
    # Once the commander accepts a cancel, the goal ends, and its result
    # resolves it.
    fake_commander.reject_cancels = False
    with caplog.at_level(logging.INFO, logger=commander_module.__name__):
        held.set()
        if answer == "no answer":
            # The held cancels are answered now.
            assert wait_until(lambda: "is no longer active: its result came" in caplog.text)
        count = client.unverified_count
        client.read_arms()
    assert (
        "The move of left_arm from an earlier call is no longer active: its result came."
        in caplog.text
    )
    assert fake_commander.results == [("left_arm", "Execution Failed")]
    assert client.unverified is None
    assert client.last_recovery == Recovery(count, (("the move of left_arm", "result"),), ())


def test_a_newer_goal_waits_until_the_commander_confirms_that_an_older_one_ended(
    client, fake_commander, poses, monkeypatch, caplog
):
    # The commander accepts an abandoned goal late and holds the cancel of
    # it unanswered while recovery runs. A cancel that it accepts stops both
    # arms, so no newer goal may start until it was answered and the goal
    # ended.
    monkeypatch.setattr(client, "_abandon_after_s", 0.3)
    # Longer than the read's timeout, so recovery waits for the cancel that
    # is on its way instead of sending another.
    monkeypatch.setattr(client, "_cancel_answer_s", 5.0)
    client.read_arms()
    client.joint_names("left_arm")
    client.joint_names("right_arm")
    before = fake_commander.joints()
    accept = threading.Event()
    fake_commander.accept_gate = accept
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert cancel_answer(error.value) == "pending"
    assert wait_until(lambda: client.busy() is None)
    assert client.unverified == ABANDONED
    # Accepted late, the goal plans until the test lets it finish, and the
    # cancel that follows the acceptance is held.
    plan = threading.Event()
    answer = threading.Event()
    fake_commander.plan_gate = plan
    fake_commander.cancel_gate = answer
    accept.set()
    assert wait_until(lambda: fake_commander.cancel_requests == 1)
    assert client.unverified == ACCEPTED_LATE

    error = refused(lambda: client.read_arms(timeout_s=0.5))

    assert error.message == (
        "The commander did not answer a cancel of the move of left_arm from an earlier call in "
        "time, so an arm may still be moving. Wait a few seconds, then try again; if it fails "
        "again, tell the person that the arms cannot be checked."
    )
    assert fake_commander.cancel_requests == 1
    error = refused(lambda: move(client, "right_arm", table(poses, "standby", "right")))
    assert error.message == unverified_refusal("move right_arm", ACCEPTED_LATE)
    assert "right_arm" not in [goal[0] for goal in fake_commander.joint_goals]
    # The commander accepts the cancel, and the goal ends without moving once
    # its plan is done.
    with caplog.at_level(logging.INFO, logger=commander_module.__name__):
        answer.set()
        assert wait_until(
            lambda: "Cancel of the move of left_arm, which the commander accepted late: the "
            "arms were told to stop." in caplog.text
        )
        plan.set()
        assert wait_until(
            lambda: "The move of left_arm from an earlier call is no longer active: its result "
            "came." in caplog.text
        )
    assert fake_commander.results == [("left_arm", "Canceled")]
    assert fake_commander.joints() == before
    assert client.read_arms().left == ArmState("home", True)
    assert client.last_recovery.goals == (("the move of left_arm", "result"),)
    # Nothing is left that could stop the newer goal.
    fake_commander.move_time_s = 0.5
    assert move(client, "right_arm", table(poses, "standby", "right")) == "Success"
    assert fake_commander.results[-1] == ("right_arm", "Success")


def test_recovery_waits_for_the_answer_to_a_cancel_already_on_its_way(
    client, fake_commander, poses, monkeypatch, module_clock
):
    # As above, but the commander answers the held cancel while recovery
    # waits, so recovery uses that answer instead of sending another cancel.
    clock = module_clock(commander_module)
    client.read_arms()
    client.joint_names("left_arm")
    before = fake_commander.joints()
    accept = threading.Event()
    fake_commander.accept_gate = accept
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: fake_commander.joint_goals)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "pending"
    record = client._goal
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    plan = threading.Event()
    answer = threading.Event()
    fake_commander.plan_gate = plan
    fake_commander.cancel_gate = answer
    accept.set()
    assert wait_until(lambda: fake_commander.cancel_requests == 1)
    assert client.unverified == ACCEPTED_LATE
    # Recovery decides whether to send a cancel before it waits for any
    # answer (_await_end), so the cancel is answered only once it waits. The
    # plan is done once the commander accepted the cancel, so the goal sees
    # the cancel and ends without moving.
    waiting = threading.Event()
    await_end = client._await_end

    def await_end_and_tell(goal, deadline):
        waiting.set()
        return await_end(goal, deadline)

    monkeypatch.setattr(client, "_await_end", await_end_and_tell)

    def release() -> None:
        assert waiting.wait(10)
        answer.set()
        assert wait_until(lambda: record.cancels[-1].done())
        plan.set()

    reading = Call(client.read_arms)
    releaser = Call(release)
    releaser.result()
    state = reading.result()

    assert state.left == ArmState("home", True)
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "result"),)
    assert fake_commander.cancel_requests == 1
    assert len(record.cancels) == 1
    assert fake_commander.results == [("left_arm", "Canceled")]
    assert fake_commander.joints() == before


def test_a_goal_accepted_late_after_recovery_resolved_it_is_unresolved_again(
    client, fake_commander, poses, monkeypatch, module_clock
):
    # The commander answers the send only after the call gave up, so
    # recovery's cancel finds no goal and resolves it as unknown.
    clock = module_clock(commander_module)
    client.read_arms()
    client.joint_names("left_arm")
    client.joint_names("right_arm")
    before = fake_commander.joints()
    accept = threading.Event()
    fake_commander.accept_gate = accept
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: fake_commander.joint_goals)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "pending"
    record = client._goal
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    client.read_arms()
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "unknown"),)
    # Accepted after all, the goal plans until the test lets it finish. The
    # commander accepts the cancel that follows the acceptance.
    plan = threading.Event()
    fake_commander.plan_gate = plan
    accept.set()
    assert wait_until(lambda: client.unverified == ACCEPTED_LATE)
    # Recovery asks for the result again once it has the answer to the
    # cancel, and waits for it until the read's deadline.
    asked = results_asked(record)
    recovering = Call(client.read_arms)
    assert wait_until(lambda: results_asked(record) > asked)
    clock.advance(client._read_timeout_s + 1)
    error = refused(recovering.result)

    # The arms are still, but the goal is not over until its result comes.
    assert error.message == (
        "The commander accepted the cancel of the move of left_arm from an earlier call, and "
        "its end has not come yet. Wait a few seconds, then try again."
    )
    error = refused(lambda: move(client, "right_arm", table(poses, "standby", "right")))
    assert error.message == unverified_refusal("move right_arm", ACCEPTED_LATE)
    assert "right_arm" not in [goal[0] for goal in fake_commander.joint_goals]
    # Once the plan is done, the goal sees the cancel and ends without moving.
    plan.set()
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Canceled")])
    assert client.read_arms().left == ArmState("home", True)
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "result"),)
    assert fake_commander.joints() == before


def test_recovery_cancels_a_goal_whose_send_was_never_answered_by_its_id(
    client, fake_commander, poses, monkeypatch, module_clock
):
    # The commander accepts the goal, which then plans, but its answer to
    # the send never reaches the client, which so has no goal handle.
    clock = module_clock(commander_module)
    plan = threading.Event()
    fake_commander.plan_gate = plan
    client.read_arms()
    client.joint_names("left_arm")
    before = fake_commander.joints()
    send_goal_async = client._set_joint_position.send_goal_async

    def send_and_lose_the_answer(goal, **kwargs):
        send_goal_async(goal, **kwargs)
        return Future()

    monkeypatch.setattr(client._set_joint_position, "send_goal_async", send_and_lose_the_answer)
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: fake_commander.joint_goals)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "pending"
    record = client._goal
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    assert client.unverified == ABANDONED

    # Recovery asks for the result once the commander accepted its cancel,
    # and waits for it until the read's deadline.
    recovering = Call(client.read_arms)
    assert wait_until(lambda: record.result is not None)
    clock.advance(client._read_timeout_s + 1)
    error = refused(recovering.result)

    # Cancelled by the id the client made, which the commander accepted.
    assert fake_commander.cancel_requests == 1
    assert error.message == (
        "The commander accepted the cancel of the move of left_arm from an earlier call, and "
        "its end has not come yet. Wait a few seconds, then try again."
    )
    # Its result, asked for by the same id, comes once the plan is done.
    plan.set()
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Canceled")])
    assert client.read_arms().left == ArmState("home", True)
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "result"),)
    assert fake_commander.joints() == before


def test_recovery_asks_again_for_a_result_whose_answer_was_lost(
    client, fake_commander, poses, monkeypatch, module_clock
):
    # The commander accepts the call's cancel and ends the goal, but its
    # answer to the call's request for the result never reaches the client,
    # as when discovery dropped meanwhile. The commander then answers
    # recovery's cancel that the goal ended, which after an accepted cancel
    # only the result can confirm.
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    # Held until its cancel.
    fake_commander.never_answer = True
    client.read_arms()
    client.joint_names("left_arm")
    get_result_async = ClientGoalHandle.get_result_async
    requests = []

    def lose_the_first_answer(handle):
        future = get_result_async(handle)
        requests.append(future)
        return Future() if len(requests) == 1 else future

    monkeypatch.setattr(ClientGoalHandle, "get_result_async", lose_the_first_answer)
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "accepted"
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    assert client.unverified == ABANDONED
    count = client.unverified_count

    # On a thread: recovery that waits for the lost answer would wait until
    # the test clock moves on.
    Call(client.read_arms).result()

    assert len(requests) == 2
    assert client.unverified is None
    assert client.last_recovery == Recovery(count, (("the move of left_arm", "result"),), ())


GRIPPER_ABANDONED = (
    "the left_gripper command from an earlier call was given up on before its end was known"
)


def test_recovery_waits_for_an_abandoned_gripper_command_and_reads_its_finger(
    client, fake_commander, monkeypatch, module_clock
):
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    # The commander accepts the cancel, but the fingers finish their ramp,
    # which holds part way until a newer command supersedes it.
    fake_commander.ramp_gate = threading.Event()
    client.read_arms()
    closing = Call(client.control_gripper, "left_gripper", 0.0, timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        closing.result()
    assert cancel_answer(error.value) == "accepted"
    record = client._goal
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    assert client.unverified == GRIPPER_ABANDONED

    # Recovery asks for the result again once the commander answered its
    # cancel, and waits for it until the read's deadline.
    recovering = Call(client.read_arms)
    assert wait_until(lambda: results_asked(record) == 2)
    clock.advance(client._read_timeout_s + 1)
    error = refused(recovering.result)

    assert error.message == (
        "The commander accepted the cancel of the left_gripper command from an earlier call, "
        "and its end has not come yet. Wait a few seconds, then try again."
    )
    # The ramp ends, and its result comes. The settled read then reads the
    # finger too, so a reading of the gripper that lacks it refuses.
    fake_commander.supersede("left_gripper", 0.0)
    assert wait_until(lambda: fake_commander.results == [("left_gripper", "Canceled")])
    fake_commander.missing_joints = {"openarm_left_finger_joint1"}
    with pytest.raises(StepError) as error:
        client.read_arms()
    assert (error.value.code, error.value.message) == (
        "INTERNAL",
        "The commander answered 0 joint names and 0 positions for left_gripper.",
    )
    assert client.unverified == GRIPPER_ABANDONED
    fake_commander.missing_joints = set()
    count = client.unverified_count
    requests = len(fake_commander.joint_state_requests)
    client.read_arms()
    assert fake_commander.joint_state_requests[requests:] == [
        "both_arms", "left_gripper", "both_arms", "left_gripper"
    ]
    assert client.unverified is None
    assert client.last_recovery == Recovery(
        count, (("the left_gripper command", "result"),), ("left_gripper",)
    )


def test_a_settled_read_refuses_while_a_gripper_still_moves(
    client, fake_commander, monkeypatch, module_clock
):
    # The commander restarts during a close. The new one does not know the
    # command, whose ramp goes on: it holds part way, and the finger moves
    # between the settled read's two reads.
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.ramp_gate = threading.Event()
    client.read_arms()
    finger = "openarm_left_finger_joint1"
    opened = fake_commander.joints()[finger]
    closing = Call(client.control_gripper, "left_gripper", 0.0, timeout_s=0.3)
    # Accepted, and past the ramp's first step: the finger then holds where
    # it is, so the hook's move is the only change the settled read sees.
    assert wait_until(
        lambda: accepted_handle(client) is not None and fake_commander.joints()[finger] < opened
    )
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        closing.result()
    assert cancel_answer(error.value) == "accepted"
    restart(client, fake_commander, CONTROL_GRIPPER)
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    start = len(fake_commander.joint_state_requests)

    def move_finger(_group: str, index: int) -> None:
        # Before the answer to the second round's read of the finger.
        if index == start + 3:
            fake_commander.teleport({finger: fake_commander.joints()[finger] - 0.002})

    fake_commander.joint_states_hook = move_finger

    error = refused(lambda: client.read_arms())

    assert error.message == (
        "A gripper is still moving: openarm_left_finger_joint1 moved 0.0020 m between two reads "
        "0.3 s apart. Wait a few seconds, then try again."
    )
    assert fake_commander.joint_state_requests[start:] == [
        "both_arms", "left_gripper", "both_arms", "left_gripper"
    ]
    assert client.unverified == GRIPPER_ABANDONED
    # Once the ramp ends, the next settled read shows the finger still.
    fake_commander.joint_states_hook = None
    fake_commander.supersede("left_gripper", 0.0)
    assert wait_until(lambda: fake_commander.results)
    count = client.unverified_count
    client.read_arms()
    assert client.unverified is None
    assert client.last_recovery == Recovery(
        count, (("the left_gripper command", "unknown"),), ("left_gripper",)
    )


@pytest.mark.parametrize("during", ["first read", "settle"])
def test_a_goal_accepted_late_during_a_settled_read_keeps_the_arms_unverified(
    client, fake_commander, poses, monkeypatch, module_clock, caplog, during
):
    clock = module_clock(commander_module)
    client.read_arms()
    client.joint_names("left_arm")
    before = fake_commander.joints()
    gate = threading.Event()
    fake_commander.accept_gate = gate
    # Cancelled while it plans, so it never moves.
    fake_commander.plan_until_cancelled = True
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: fake_commander.joint_goals)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "pending"
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    assert client.unverified == ABANDONED

    # The commander holds back the answer to one of the settled read's two
    # reads, and accepts the goal meanwhile.
    first = len(fake_commander.joint_state_requests)
    held = threading.Event()
    if during == "first read":
        # The first read: the mark must be taken before that read.
        fake_commander.joint_states_gate = held
        held_request = first
    else:
        # The second read, after the first one was answered.
        def hold_the_second_read(_group: str, index: int) -> None:
            if index == first:
                fake_commander.joint_states_gate = held

        fake_commander.joint_states_hook = hold_the_second_read
        held_request = first + 1

    def accept_during_the_read() -> None:
        assert wait_until(lambda: len(fake_commander.joint_state_requests) > held_request)
        gate.set()
        assert wait_until(lambda: client.unverified == ACCEPTED_LATE)
        held.set()

    with caplog.at_level(logging.INFO, logger=commander_module.__name__):
        reading = Call(client.read_arms)
        releaser = Call(accept_during_the_read)
        releaser.result()
        error = refused(reading.result)
        fake_commander.joint_states_gate = None
        fake_commander.joint_states_hook = None

        assert error.message == (
            f"While the arms were being checked, {ACCEPTED_LATE}. Wait a few seconds, then try "
            "again."
        )
        assert client.unverified == ACCEPTED_LATE
        assert client.last_state == ArmsState()
        assert wait_until(
            lambda: "Cancel of the move of left_arm, which the commander accepted late: the "
            "arms were told to stop." in caplog.text
        )
    assert "The move of left_arm had been abandoned, so the arms are unverified" in caplog.text
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Canceled")])
    assert fake_commander.joints() == before
    assert client.read_arms().left == ArmState("home", True)
    assert client.unverified is None


@pytest.mark.parametrize("reject_cancels", [False, True], ids=["accepted", "refused"])
def test_a_goal_accepted_late_fails_a_newer_move_that_its_cancel_may_stop(
    client, fake_commander, poses, monkeypatch, module_clock, caplog, reject_cancels
):
    clock = module_clock(commander_module)
    client.joint_names("left_arm")
    client.joint_names("right_arm")
    gate = threading.Event()
    fake_commander.accept_gate = gate
    # Given up on before the commander answered it, then abandoned.
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: fake_commander.joint_goals)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "pending"
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    # A settled read verifies the arms, so goals are taken again.
    home = ArmState("home", True)
    assert client.read_arms() == ArmsState(left=home, right=home)

    fake_commander.accept_gate = None
    fake_commander.reject_cancels = reject_cancels
    # The goals plan until the commander has the old goal's cancel, and that
    # cancel is answered only once the newer goal moves. A refused cancel
    # leaves the newer move to reach its end; an accepted one stops it long
    # before its end.
    fake_commander.move_time_s = 0.5 if reject_cancels else 30.0
    plan = threading.Event()
    answer_gate = threading.Event()
    fake_commander.plan_gate = plan
    fake_commander.cancel_gate = answer_gate
    moved = arm_moved(fake_commander, poses, "standby", "right")

    with caplog.at_level(logging.INFO, logger=commander_module.__name__):
        moving = Call(move, client, "right_arm", table(poses, "standby", "right"))
        assert wait_until(lambda: accepted_handle(client) is not None)
        # The commander accepts the old goal while the newer one runs.
        gate.set()
        assert wait_until(lambda: fake_commander.cancel_requests == 1)
        plan.set()
        assert wait_until(moved)
        answer_gate.set()
        with pytest.raises(StepError) as error:
            moving.result()
        answer = (
            "the commander refused the cancel, so an arm may still be moving"
            if reject_cancels
            else "the arms were told to stop"
        )
        assert wait_until(
            lambda: f"Cancel of the move of left_arm, which the commander accepted late: {answer}."
            in caplog.text
        )
        # A refused cancel is a warning; an accepted one only information.
        [logged] = [
            record
            for record in caplog.records
            if record.getMessage().startswith("Cancel of the move of left_arm, which")
        ]
        assert logged.levelno == (logging.WARNING if reject_cancels else logging.INFO)

    disturbed = f"{ACCEPTED_LATE}, and a cancel it accepts stops both arms"
    assert (error.value.code, error.value.message) == (
        "SKILL_FAILED",
        (
            f"The move of right_arm may have been stopped part way: {disturbed}. Call "
            "get_arm_state to see where the arms are."
        ),
    )
    # Its end is not the commander's report, so the arm may still move.
    assert not isinstance(error.value, GoalFailed)
    assert (
        "The move of right_arm may be stopped part way by the cancel of the move of left_arm."
        in caplog.text
    )
    assert client.unverified == ACCEPTED_LATE
    assert client.last_state == ArmsState()
    # An accepted cancel of the old goal stopped the newer one too.
    status = "Success" if reject_cancels else "Execution Failed"
    assert ("right_arm", status) in fake_commander.results


def test_a_goal_accepted_late_just_before_a_newer_send_refuses_that_send(
    client, fake_commander, poses, monkeypatch
):
    monkeypatch.setattr(client, "_abandon_after_s", 0.3)
    # Recovery waits in real time for the answer to its cancel, which the
    # commander sends at once, since it does not know the goal yet.
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    client.joint_names("left_arm")
    client.joint_names("right_arm")
    gate = threading.Event()
    fake_commander.accept_gate = gate
    with pytest.raises(StepError):
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: client.busy() is None)
    client.read_arms()
    assert client.unverified is None
    fake_commander.accept_gate = None
    server_is_ready = client._set_joint_position.server_is_ready

    def accepted_after_the_check() -> bool:
        # The newer call has passed the checks of _exclusive(), and the
        # commander accepts the old goal before the newer one is sent.
        gate.set()
        assert wait_until(lambda: client.unverified == ACCEPTED_LATE)
        return server_is_ready()

    monkeypatch.setattr(client._set_joint_position, "server_is_ready", accepted_after_the_check)

    error = refused(lambda: move(client, "right_arm", table(poses, "standby", "right")))

    assert error.message == unverified_refusal("move right_arm", ACCEPTED_LATE)
    # The newer goal was never sent, so the old goal's cancel cannot stop it.
    assert "right_arm" not in [goal[0] for goal in fake_commander.joint_goals]


def test_a_late_rejection_of_an_abandoned_goal_changes_nothing(
    client, fake_commander, poses, monkeypatch, module_clock, caplog
):
    clock = module_clock(commander_module)
    client.joint_names("left_arm")
    client.joint_names("right_arm")
    gate = threading.Event()
    fake_commander.accept_gate = gate
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: fake_commander.joint_goals)
    clock.advance(0.4)
    with pytest.raises(StepError):
        moving.result()
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    client.read_arms()
    fake_commander.accept_gate = None
    # The newer goal plans until the old one's rejection was handled.
    plan = threading.Event()
    fake_commander.plan_gate = plan

    with caplog.at_level(logging.DEBUG, logger=commander_module.__name__):
        moving = Call(move, client, "right_arm", table(poses, "standby", "right"))
        assert wait_until(lambda: accepted_handle(client) is not None)
        fake_commander.reject_goals = True
        gate.set()
        assert wait_until(
            lambda: "The commander did not accept the move of left_arm, which was abandoned."
            in caplog.text
        )
        # The newer goal's record is untouched.
        record = client._goal
        assert record is not None
        assert (record.what, record.disturbed) == ("the move of right_arm", None)
        assert client.unverified is None
        plan.set()
        assert moving.result() == "Success"

    assert fake_commander.cancel_requests == 0
    assert fake_commander.results == [("right_arm", "Success")]


def test_a_late_result_of_an_abandoned_goal_does_not_end_a_newer_one(
    client, fake_commander, poses, token, monkeypatch, module_clock
):
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    # The commander refuses the call's cancel while the move plans, and the
    # move's result comes only once the test lets it.
    fake_commander.reject_cancels = True
    plan = threading.Event()
    fake_commander.plan_gate = plan
    result = threading.Event()
    fake_commander.result_gate = result
    client.joint_names("left_arm")
    client.joint_names("right_arm")
    # Every end of a goal that the client handles, once handled.
    ended = []
    end_goal = client._end_goal

    def end_goal_and_note(goal, future=None):
        end_goal(goal, future)
        ended.append(goal)

    monkeypatch.setattr(client, "_end_goal", end_goal_and_note)
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "refused"
    older = client._goal
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    # The move has ended, so the commander answers recovery's cancel that it
    # ended, which resolves it before its result comes.
    plan.set()
    assert wait_until(lambda: older.sent.result().status == GoalStatus.STATUS_SUCCEEDED)
    assert fake_commander.results == [("left_arm", "Success")]
    client.read_arms()
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "ended"),)
    fake_commander.reject_cancels = False
    # Held until its call cancels it.
    fake_commander.never_answer = True

    moving = Call(move, client, "right_arm", table(poses, "standby", "right"))
    assert wait_until(lambda: accepted_handle(client) is not None)
    newer = client._goal
    assert newer is not None and newer is not older
    assert not older.result.done()

    # The old goal's result comes while the newer one runs.
    result.set()
    assert wait_until(lambda: older in ended)
    assert client._goal is newer
    token.cancel()
    with pytest.raises(StepError) as error:
        moving.result()
    assert error.value.code == "CANCELLED"


class _LockThatAnotherCallTakesFirst:
    """The call lock, taken once by another call while the first acquire() waits.

    That call leaves its goal outstanding, so a call that passed the check before it
    waited for the lock must check again once it holds it.
    """

    def __init__(self, client, other_call):
        self._client = client
        self._lock = client._lock
        self._other_call = other_call

    def acquire(self, timeout=-1):
        self._client._lock = self._lock
        with pytest.raises(StepError) as error:
            self._other_call()
        assert error.value.code == "TIMEOUT"
        return self._lock.acquire(timeout=timeout)


def test_a_call_that_waited_for_the_lock_is_refused_once_it_holds_it(
    client, fake_commander, poses, monkeypatch
):
    fake_commander.never_answer = True
    fake_commander.reject_cancels = True
    client.joint_names("left_arm")

    def other_move():
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)

    monkeypatch.setattr(client, "_lock", _LockThatAnotherCallTakesFirst(client, other_move))

    error = refused(lambda: client.read_arms())

    assert error.message == (
        "Cannot read the joints of both_arms: the move of left_arm from an earlier call has not "
        "ended yet. Try again in a few seconds."
    )
    assert fake_commander.joint_state_requests == ["left_arm"]


# -- Cancels ----------------------------------------------------------------------


def test_a_refused_cancel_says_the_arm_may_still_move(
    client, fake_commander, poses, monkeypatch, module_clock, caplog
):
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    # The move plans until the test lets it go on.
    plan = threading.Event()
    fake_commander.plan_gate = plan
    fake_commander.reject_cancels = True
    client.joint_names("left_arm")

    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
        assert wait_until(lambda: accepted_handle(client) is not None)
        clock.advance(0.4)
        with pytest.raises(StepError) as error:
            moving.result()

    assert error.value.message == (
        "The move of left_arm did not finish within 0.3 s; the commander refused the cancel, "
        "so an arm may still be moving."
    )
    assert cancel_answer(error.value) == "refused"
    assert "Cancel of the move of left_arm: the commander refused the cancel" in caplog.text
    assert fake_commander.cancel_requests == 1
    assert client.busy() == "the move of left_arm from an earlier call"
    # The move goes on to its end.
    plan.set()
    assert wait_until(lambda: client.busy() is None)
    assert fake_commander.results == [("left_arm", "Success")]
    assert client.read_arms().left == ArmState("standby", False)


def test_a_cancel_after_the_goal_ended_says_so(
    client, fake_commander, poses, monkeypatch, module_clock
):
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    # The move ends at once, but its result is sent only once the test lets
    # it.
    fake_commander.move_time_s = 0.05
    result = threading.Event()
    fake_commander.result_gate = result
    client.joint_names("left_arm")

    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.5)
    assert wait_until(lambda: goal_status(client) == GoalStatus.STATUS_SUCCEEDED)
    clock.advance(0.6)
    with pytest.raises(StepError) as error:
        moving.result()

    assert error.value.message == (
        "The move of left_arm did not finish within 0.5 s; the goal had already ended."
    )
    assert cancel_answer(error.value) == "ended"
    # The commander answers ERROR_GOAL_TERMINATED without calling its
    # cancel callback.
    assert fake_commander.cancel_requests == 0
    assert client.busy() == "the move of left_arm from an earlier call"
    result.set()
    assert wait_until(lambda: client.busy() is None)
    assert fake_commander.results == [("left_arm", "Success")]


def test_an_unanswered_cancel_says_the_arm_may_still_move(
    client, fake_commander, poses, monkeypatch, module_clock
):
    clock = module_clock(commander_module)
    # The call waits this long, in real time, for the answer to its cancel.
    monkeypatch.setattr(client, "_cancel_answer_s", 0.2)
    # Held until its cancel, which the commander answers only once the test
    # lets it.
    fake_commander.never_answer = True
    held = threading.Event()
    fake_commander.cancel_gate = held
    client.joint_names("left_arm")

    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()

    assert error.value.message == (
        "The move of left_arm did not finish within 0.3 s; the commander did not answer the "
        "cancel, so an arm may still be moving."
    )
    assert cancel_answer(error.value) == "no answer"
    # It returned while the commander still held the answer.
    assert not held.is_set()
    assert wait_until(lambda: fake_commander.cancel_requests == 1)
    assert client.busy() == "the move of left_arm from an earlier call"
    # The late answer stops the arm.
    held.set()
    assert wait_until(lambda: client.busy() is None)
    assert fake_commander.results == [("left_arm", "Execution Failed")]


GOAL_ID = UUID(uuid=list(range(16)))
OTHER_ID = UUID(uuid=list(range(1, 17)))


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        (CancelGoal.Response(goals_canceling=[GoalInfo(goal_id=GOAL_ID)]), "accepted"),
        # An rclpy server's refusal, and an rclcpp server's.
        (CancelGoal.Response(), "refused"),
        (CancelGoal.Response(goals_canceling=[GoalInfo(goal_id=OTHER_ID)]), "refused"),
        (CancelGoal.Response(return_code=CancelGoal.Response.ERROR_REJECTED), "refused"),
        (CancelGoal.Response(return_code=CancelGoal.Response.ERROR_UNKNOWN_GOAL_ID), "unknown"),
        (CancelGoal.Response(return_code=CancelGoal.Response.ERROR_GOAL_TERMINATED), "ended"),
        (None, "no answer"),
    ],
)
def test_the_cancel_outcome_follows_the_commanders_answer(answer, outcome):
    assert commander_module._cancel_outcome(answer, GOAL_ID) == outcome


def test_a_cancel_mid_move_ends_the_goal_promptly(
    client, fake_commander, poses, token, monkeypatch
):
    # The commander answers the cancel at once; the call waits for that
    # answer in real time, whatever the token.
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.move_time_s = 5.0
    # The goal's result is sent only once the test lets it, so the call
    # returns without waiting for the result.
    result = threading.Event()
    fake_commander.result_gate = result
    home = client.read_arms()
    client.joint_names("left_arm")
    standby = table(poses, "standby", "left")
    moved = arm_moved(fake_commander, poses, "standby", "left")

    moving = Call(move, client, "left_arm", standby)
    # The call has the acceptance and waits for the result, and the arm has
    # left home: a token set before the call has the acceptance would end
    # its wait for the send instead, and send no cancel.
    assert wait_until(lambda: accepted_handle(client) is not None and moved())
    token.cancel()
    with pytest.raises(StepError) as error:
        moving.result()

    assert error.value.code == "CANCELLED"
    assert error.value.message == (
        "A stop or a reset interrupted the move of left_arm; the arms were told to stop."
    )
    assert cancel_answer(error.value) == "accepted"
    assert fake_commander.cancel_requests == 1
    # The commander stops the arms and ends the goal as failed, not cancelled.
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])
    # The arm stopped part way, and stays unknown until it is read again.
    joint1 = fake_commander.joints()["openarm_left_joint1"]
    assert table(poses, "home", "left")["openarm_left_joint1"] < joint1
    assert joint1 < standby["openarm_left_joint1"]
    assert client.last_state == ArmsState(left=None, right=home.right)
    assert client.last_state.sentence() == UNKNOWN_SENTENCE
    # A stop is the only thing that sends "stop" on the execution event topic.
    assert fake_commander.stops_heard == 0
    result.set()
    assert wait_until(lambda: client.busy() is None)


def test_a_set_token_sends_nothing(client, fake_commander, token):
    token.cancel()

    for call, action in (
        (lambda: client.read_arms(), "read the joints of both_arms"),
        (lambda: client.joint_states("left_gripper"), "read the joints of left_gripper"),
        (lambda: client.set_joint_position("left_arm", [0.0] * 7, 5.0), "move left_arm"),
        (lambda: client.control_gripper("left_gripper", 0.0), "move left_gripper"),
        (lambda: client.check_for_driving(), "check the arms for driving"),
    ):
        with pytest.raises(StepError) as error:
            call()
        assert (error.value.code, error.value.message) == (
            "CANCELLED",
            f"Cannot {action}: a stop or a reset is in effect.",
        )

    assert fake_commander.joint_state_requests == []
    assert fake_commander.joint_goals == []
    assert fake_commander.gripper_goals == []
    # Calls are taken again once the runtime resets the token.
    token.reset()
    assert client.read_arms().travel_safe is True


@pytest.mark.parametrize("gripper", [False, True], ids=["arm", "gripper"])
def test_a_stop_once_the_action_server_is_found_sends_no_goal(
    client, fake_commander, poses, monkeypatch, gripper
):
    # The stop comes from another thread after the call found the action server and
    # before it sent the goal. Once stop() has returned, no goal may go out, since
    # "stop" does not reach a gripper and the goal would run with nobody waiting.
    client.joint_names("left_arm")
    wait_ready = client._wait_ready

    def found_then_stopped(ready, name, *args, **kwargs):
        wait_ready(ready, name, *args, **kwargs)
        if name.endswith(" action"):
            client.stop()

    monkeypatch.setattr(client, "_wait_ready", found_then_stopped)

    with pytest.raises(StepError) as error:
        if gripper:
            client.control_gripper("left_gripper", 0.0)
        else:
            move(client, "left_arm", table(poses, "standby", "left"))

    assert error.value.code == "CANCELLED"
    assert fake_commander.joint_goals == []
    assert fake_commander.gripper_goals == []
    assert client._goal is None


def test_a_read_or_a_second_goal_during_a_move_is_refused(client, fake_commander, poses):
    # The move plans until the test lets it go on, so it holds the call lock
    # meanwhile, and a call that waited for that lock would not return.
    plan = threading.Event()
    fake_commander.plan_gate = plan
    client.read_arms()
    client.joint_names("left_arm")
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"))
    assert wait_until(lambda: fake_commander.joint_goals)

    for call in (
        lambda: client.read_arms(),
        lambda: client.control_gripper("left_gripper", 0.0),
    ):
        with pytest.raises(StepError) as error:
            Call(call).result()
        assert error.value.code == "PRECONDITION_FAILED"
        assert "while the move of left_arm is running" in error.value.message

    assert not plan.is_set()
    assert client.last_state.left is None
    plan.set()
    assert moving.result() == "Success"
    assert fake_commander.gripper_goals == []


def test_right_joint2_fixed(client, fake_commander, poses):
    fake_commander.right_joint2_fixed = True

    right = client.joint_names("right_arm")
    assert RIGHT_JOINT2 not in right
    assert len(right) == 6
    assert len(client.joint_names("both_arms")) == 13

    # Every preset has joint2 at 0, which the fixed joint holds.
    assert move(client, "right_arm", table(poses, "standby", "right")) == "Success"
    assert client.read_arms().right == ArmState("standby", False)
    with pytest.raises(StepError) as error:
        client.set_joint_position("right_arm", [0.0] * 7, 5.0)
    assert error.value.code == "INTERNAL"
    assert "rejected the goal for right_arm with 7 positions" in error.value.message


# -- The stop -----------------------------------------------------------------------


def test_stop_ends_a_move_and_stops_the_arms_through_moveit(
    client, fake_commander, poses, token, monkeypatch
):
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.move_time_s = 5.0
    home = client.read_arms()
    client.joint_names("left_arm")
    assert wait_until(lambda: stop_matched(client))

    def in_no_pose() -> bool:
        joints = fake_commander.joints()
        return ArmsState.from_joints(joints, poses.poses).left == ArmState(None, False)

    moving = Call(move, client, "left_arm", table(poses, "standby", "left"))
    # Stopped out of every named pose, so the read after the stop shows it.
    assert wait_until(lambda: accepted_handle(client) is not None and in_no_pose())
    client.stop()

    assert token.cancelled
    with pytest.raises(StepError) as error:
        moving.result()
    assert error.value.code == "CANCELLED"
    assert error.value.message.startswith("A stop or a reset interrupted the move of left_arm; ")
    # The stop event and the call's cancel race to end the goal, and either
    # stops the arms.
    assert cancel_answer(error.value) in ("accepted", "ended")
    assert wait_until(lambda: fake_commander.stops_heard == 1)
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])
    # Every call is refused while the token is set, and sends nothing.
    requests = len(fake_commander.joint_state_requests)
    with pytest.raises(StepError) as error:
        client.read_arms()
    assert (error.value.code, error.value.message) == (
        "CANCELLED",
        "Cannot read the joints of both_arms: a stop or a reset is in effect.",
    )
    assert len(fake_commander.joint_state_requests) == requests
    # Once the runtime resets the token, the arm reads as stopped part way.
    token.reset()
    assert wait_until(lambda: client.busy() is None)
    assert client.read_arms() == ArmsState(left=ArmState(None, False), right=home.right)
    assert fake_commander.stops_heard == 1


def test_stop_also_stops_a_move_whose_call_gave_up(
    client, fake_commander, poses, monkeypatch, module_clock
):
    # The commander refuses the call's cancel, so the move goes on with no
    # call waiting for it. MoveIt's stop reaches every execution.
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.move_time_s = 5.0
    fake_commander.reject_cancels = True
    client.joint_names("left_arm")
    moved = arm_moved(fake_commander, poses, "standby", "left")
    assert wait_until(lambda: stop_matched(client))
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None and moved())
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "refused"
    assert client.busy() == "the move of left_arm from an earlier call"

    client.stop()

    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])
    assert fake_commander.stops_heard == 1
    assert wait_until(lambda: client.busy() is None)


def test_stop_after_destroy_sets_the_token_and_publishes_nothing(connector, poses, monkeypatch):
    token = CancelToken()
    client = CommanderClient(connector, token, poses)
    published = []
    monkeypatch.setattr(client._stop_publisher, "publish", published.append)
    client.destroy()

    client.stop()

    assert token.cancelled
    assert published == []


def test_stop_never_raises(commander, token, monkeypatch, caplog):
    def fail(_message) -> None:
        raise RuntimeError("publisher gone")

    monkeypatch.setattr(commander._stop_publisher, "publish", fail)

    with caplog.at_level(logging.ERROR, logger=commander_module.__name__):
        commander.stop()

    assert token.cancelled
    assert "arms.stop failed." in caplog.text


# -- Close --------------------------------------------------------------------------


def count_destroys(monkeypatch, client) -> list[str]:
    """A list that gets one entry each time `client` destroys its joint action client."""
    destroys = []
    action_client = client._set_joint_position
    destroy = action_client.destroy

    def destroy_and_count() -> None:
        destroys.append("destroyed")
        destroy()

    monkeypatch.setattr(action_client, "destroy", destroy_and_count)
    return destroys


@pytest.mark.parametrize("timeout_s", [30.0, 0])
def test_a_close_with_nothing_outstanding_destroys_the_client_at_once(
    client, monkeypatch, timeout_s
):
    client.read_arms()
    destroys = count_destroys(monkeypatch, client)
    assert client.unresolved() == []

    started = time.monotonic()
    assert client.close(timeout_s)

    assert time.monotonic() - started < 5.0
    assert destroys == ["destroyed"]
    assert client.unresolved() == []
    # A repeated close, and a destroy after it, do nothing.
    assert client.close(30.0)
    client.destroy()
    assert destroys == ["destroyed"]


def test_a_close_waits_for_a_goal_that_its_call_still_waits_for(
    client, fake_commander, poses, monkeypatch
):
    client.joint_names("left_arm")
    # The goal's result is held until the test lets it go, so its call
    # keeps the call lock and has not given up on it.
    result = threading.Event()
    fake_commander.result_gate = result
    destroys = count_destroys(monkeypatch, client)
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"))
    assert wait_until(lambda: accepted_handle(client) is not None)

    assert client.close(0.2) is False
    # A close with no time checks once and does not wait.
    started = time.monotonic()
    assert client.close(0) is False
    assert time.monotonic() - started < 0.1
    assert destroys == []
    assert client.unresolved() == ["The move of left_arm has not ended: no result has come."]

    result.set()
    assert moving.result() == "Success"
    assert client.close(10.0)
    assert destroys == ["destroyed"]


def test_a_close_waits_for_a_read_that_has_not_returned(client, fake_commander, monkeypatch):
    # The commander holds its answer until the test lets it go, so the read
    # keeps the call lock with no goal recorded.
    held = threading.Event()
    fake_commander.joint_states_gate = held
    destroys = count_destroys(monkeypatch, client)
    reading = Call(client.joint_states, "both_arms", timeout_s=30.0)
    assert wait_until(lambda: fake_commander.joint_state_requests == ["both_arms"])

    assert client.close(0.2) is False
    assert destroys == []
    assert client.unresolved() == ["A call to the commander has not returned."]

    held.set()
    reading.result()
    assert client.close(10.0)
    assert destroys == ["destroyed"]
    assert client.unresolved() == []


def test_a_goal_accepted_after_the_stop_holds_off_close_until_its_cancel_ends_it(
    client, fake_commander, poses, monkeypatch, caplog
):
    client.joint_names("left_arm")
    before = fake_commander.joints()
    assert wait_until(lambda: stop_matched(client))
    # The commander answers the send only once the test lets it. Accepted,
    # the goal plans until its cancel is requested, which "stop" does not
    # do, and its result is held until the test lets it go.
    accept = threading.Event()
    fake_commander.accept_gate = accept
    fake_commander.plan_until_cancelled = True
    result = threading.Event()
    fake_commander.result_gate = result
    destroys = count_destroys(monkeypatch, client)
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"))
    assert wait_until(lambda: fake_commander.joint_goals)

    # As at shutdown: the stop ends the call's wait before the commander
    # answers the send, and "stop" reaches MoveIt before the goal runs.
    client.stop()
    with pytest.raises(StepError) as error:
        moving.result()
    assert error.value.code == "CANCELLED"
    assert cancel_answer(error.value) == "pending"
    assert wait_until(lambda: fake_commander.stops_heard == 1)

    assert client.close(0.2) is False
    assert destroys == []
    assert client.available()
    assert client.unresolved() == [
        "The move of left_arm has not ended: the commander has not answered its send."
    ]

    # The client is still alive, so the late acceptance is cancelled, not
    # only logged.
    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        accept.set()
        assert wait_until(
            lambda: "The commander accepted the move of left_arm after its call gave up; "
            "cancelling it." in caplog.text
        )
    assert wait_until(lambda: fake_commander.cancel_requests == 1)
    assert wait_until(
        lambda: client.unresolved()
        == ["The move of left_arm has not ended: a cancel was sent, but no result has come."]
    )
    assert client.close(0.2) is False
    assert destroys == []

    result.set()
    assert client.close(10.0)
    assert destroys == ["destroyed"]
    assert client.unresolved() == []
    # Cancelled while it planned, so the arm never moved.
    assert fake_commander.results == [("left_arm", "Canceled")]
    assert fake_commander.joints() == before
    # A repeated close does nothing.
    assert client.close(10.0)
    assert destroys == ["destroyed"]


@pytest.mark.parametrize(
    ("bound", "reason"),
    [
        ("abandon_after_s", "no result came within 0.3 s after its call gave up"),
        ("absent_abandon_s", "its action server has not been discovered for 0.3 s"),
    ],
)
def test_a_close_destroys_the_client_once_the_goal_holding_it_off_is_abandoned(
    client, fake_commander, poses, token, monkeypatch, caplog, bound, reason
):
    # The goal is held until it is cancelled, and the commander refuses the
    # call's cancel, so its result never comes.
    fake_commander.never_answer = True
    fake_commander.reject_cancels = True
    monkeypatch.setattr(client, f"_{bound}", 0.3)
    client.joint_names("left_arm")
    destroys = count_destroys(monkeypatch, client)
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"))
    assert wait_until(lambda: accepted_handle(client) is not None)
    # The token alone: "stop" would end the held goal.
    token.cancel()
    with pytest.raises(StepError) as error:
        moving.result()
    assert cancel_answer(error.value) == "refused"
    if bound == "absent_abandon_s":
        fake_commander.vanish()

    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        assert client.close(10.0)

    assert f"Abandoned the move of left_arm: {reason}" in caplog.text
    assert destroys == ["destroyed"]
    assert client.unverified == ABANDONED
    assert client.unresolved() == []
    # Still unresolved: close() does not wait for an abandoned goal.
    assert [goal.what for goal in client._unresolved] == ["the move of left_arm"]


@pytest.mark.parametrize("timeout_s", [-1.0, -0.001, math.inf, math.nan, True, "1"])
def test_a_close_bound_that_is_not_a_finite_number_of_seconds_is_a_bug(commander, timeout_s):
    with pytest.raises(ValueError, match="timeout_s must be a finite number of seconds, 0 or more"):
        commander.close(timeout_s)
    assert commander.unresolved() == []


# -- Gripper goals ----------------------------------------------------------------


def test_a_close_on_an_object_holds_it(client, fake_commander):
    fake_commander.object_widths["left_gripper"] = 0.02

    result = client.control_gripper("left_gripper", 0.0)

    # The fingers stall on the object, so the commander reports a failure
    # and the grasp at once.
    assert (result.success, result.status) == (False, "Failed")
    assert result.grasped is True
    assert result.gripper_gap == pytest.approx(0.02)
    assert client.last_gripper_position("left_gripper") == 0.0
    assert client.last_gripper_position("right_gripper") is None


def test_a_close_on_nothing_holds_nothing(client):
    result = client.control_gripper("right_gripper", 0.0)

    assert (result.success, result.status) == (True, "Success")
    assert result.grasped is False
    assert result.gripper_gap == 0.0


def test_grasp_detect_min_below_zero_reads_an_empty_close_as_held(client, fake_commander):
    fake_commander.set_parameters([Parameter("grasp_detect_min", value=-1.0)])

    result = client.control_gripper("left_gripper", 0.0)

    assert result.grasped is True
    assert result.gripper_gap == 0.0


def test_open_and_half_ignore_the_commanders_gap(client, fake_commander):
    fake_commander.object_widths["left_gripper"] = 0.02
    client.control_gripper("left_gripper", 0.0)

    for position in (0.044, 0.036):
        result = client.control_gripper("left_gripper", position)
        # The commander answers grasped false and a 0.0 gap, which says
        # nothing about the fingers.
        assert (result.success, result.grasped, result.gripper_gap) == (True, None, None)
        assert fake_commander.joints()["openarm_left_finger_joint1"] == pytest.approx(position)
        assert client.last_gripper_position("left_gripper") == position


def test_the_grasp_band_is_the_one_the_commander_holds_now(client, fake_commander):
    assert client.grasp_band() == pytest.approx((0.005, 0.044))

    # As the HRI scenario hangs a bag.
    fake_commander.set_parameters([Parameter("grasp_detect_min", value=-1.0)])

    assert client.grasp_band() == pytest.approx((-1.0, 0.044))


def redeclare(fake_commander, value) -> None:
    fake_commander.undeclare_parameter("grasp_detect_max")
    if value is not None:
        fake_commander.declare_parameter("grasp_detect_max", value)


NOT_A_NUMBER = "The commander's grasp_detect_max is not a finite number (parameter type {})."


@pytest.mark.parametrize(
    ("value", "message"),
    [
        # A parameter service answers no values when it lacks any of the names.
        (None, "The commander answered 0 values for grasp_detect_min, grasp_detect_max."),
        ("wide", NOT_A_NUMBER.format(4)),
        (math.nan, NOT_A_NUMBER.format(3)),
    ],
    ids=["missing", "text", "nan"],
)
def test_a_grasp_band_that_is_not_two_numbers_is_unavailable(
    client, fake_commander, value, message
):
    redeclare(fake_commander, value)

    with pytest.raises(StepError) as error:
        client.grasp_band()

    assert (error.value.code, error.value.message) == ("UNAVAILABLE", message)


def test_a_set_token_ends_the_grasp_band_read(client, token):
    token.cancel()

    with pytest.raises(StepError) as error:
        client.grasp_band()

    assert (error.value.code, error.value.message) == (
        "CANCELLED",
        "A stop or a reset interrupted the read of the commander's grasp detection band.",
    )


def test_the_finger_allowance_is_set_on_the_commander_and_then_settles(
    client, fake_commander, monkeypatch
):
    waits = []
    real = client._wait_slice

    def recording(done, seconds, what):
        waits.append((seconds, what))
        return real(done, min(seconds, 0.01), what)

    monkeypatch.setattr(client, "_wait_slice", recording)

    client.set_gripper_touch(True)
    assert fake_commander.get_parameter("allow_gripper_vs_octomap").value is True
    client.set_gripper_touch(False)
    assert fake_commander.get_parameter("allow_gripper_vs_octomap").value is False

    settles = [seconds for seconds, what in waits if what.endswith("allow_gripper_vs_octomap")]
    assert settles.count(commander_module.TOUCH_SETTLE_S) == 2


def test_a_finger_allowance_the_commander_refuses_is_a_skill_failure(
    client, fake_commander, monkeypatch
):
    monkeypatch.setattr(commander_module, "TOUCH_SETTLE_S", 0.0)
    fake_commander.undeclare_parameter("allow_gripper_vs_octomap")

    with pytest.raises(StepError) as error:
        client.set_gripper_touch(True)

    assert error.value.code == "SKILL_FAILED"
    assert error.value.message.startswith(
        "The commander refused allow_gripper_vs_octomap=True: "
    )


def test_the_finger_allowance_waits_out_a_running_goal(client, fake_commander, monkeypatch):
    monkeypatch.setattr(commander_module, "TOUCH_SETTLE_S", 0.0)
    fake_commander.move_time_s = 2.0
    hand = HandPose(0.6, -0.2, 0.8, 0.0, math.pi / 2, 0.3, "map")

    reaching = Call(client.go_to_pose, "right_arm", hand, 10.0)
    assert wait_until(lambda: accepted_handle(client) is not None)
    with pytest.raises(StepError) as error:
        client.set_gripper_touch(True)
    reaching.result()

    assert error.value.code == "PRECONDITION_FAILED"
    assert "allow_gripper_vs_octomap" in error.value.message
    assert fake_commander.get_parameter("allow_gripper_vs_octomap").value is False


def test_a_finger_allowance_that_is_not_a_bool_is_a_bug(client):
    with pytest.raises(ValueError):
        client.set_gripper_touch(1)


def test_a_set_token_ends_the_finger_allowance_before_it_is_sent(client, fake_commander, token):
    token.cancel()

    with pytest.raises(StepError) as error:
        client.set_gripper_touch(True)

    assert error.value.code == "CANCELLED"
    assert fake_commander.get_parameter("allow_gripper_vs_octomap").value is False


def allowance_on(fake_commander) -> None:
    """The commander as a pick, or a run killed mid-pick, leaves it."""
    fake_commander.set_parameters([Parameter("allow_gripper_vs_octomap", value=True)])


def test_the_finger_allowance_is_read_as_the_commander_holds_it_now(client, fake_commander):
    assert client.gripper_touch() is False

    allowance_on(fake_commander)

    assert client.gripper_touch() is True


def test_a_finger_allowance_the_commander_lacks_is_unavailable(client, fake_commander):
    fake_commander.undeclare_parameter("allow_gripper_vs_octomap")

    with pytest.raises(StepError) as error:
        client.gripper_touch()

    assert (error.value.code, error.value.message) == (
        "UNAVAILABLE",
        "The commander holds no bool allow_gripper_vs_octomap (parameter types []).",
    )


def test_a_set_token_ends_the_finger_allowance_read(client, token):
    token.cancel()

    with pytest.raises(StepError) as error:
        client.gripper_touch()

    assert error.value.code == "CANCELLED"


def test_the_reset_of_the_finger_allowance_goes_through_a_stop_and_settles(
    client, fake_commander, token, monkeypatch
):
    monkeypatch.setattr(commander_module, "TOUCH_SETTLE_S", 0.3)
    allowance_on(fake_commander)
    token.cancel()
    start = time.monotonic()

    client.restore_gripper_touch()

    assert fake_commander.get_parameter("allow_gripper_vs_octomap").value is False
    # The stop does not cut the settle short, so no later plan races the change.
    assert time.monotonic() - start >= 0.3


def test_the_reset_of_the_finger_allowance_goes_through_a_running_goal(
    client, fake_commander, monkeypatch
):
    monkeypatch.setattr(commander_module, "TOUCH_SETTLE_S", 0.0)
    client.set_gripper_touch(True)
    fake_commander.move_time_s = 2.0
    hand = HandPose(0.6, -0.2, 0.8, 0.0, math.pi / 2, 0.3, "map")

    reaching = Call(client.go_to_pose, "right_arm", hand, 10.0)
    assert wait_until(lambda: accepted_handle(client) is not None)
    client.restore_gripper_touch()
    assert fake_commander.get_parameter("allow_gripper_vs_octomap").value is False
    reaching.result()


def test_a_reset_of_the_finger_allowance_with_no_service_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(client, "_ready_wait_s", 0.1)
    monkeypatch.setattr(client._set_parameters, "service_is_ready", lambda: False)

    with pytest.raises(StepError) as error:
        client.restore_gripper_touch()

    assert (error.value.code, error.value.message) == (
        "UNAVAILABLE",
        "The commander's /bimanual_commander/set_parameters service is not available. "
        "Is bimanual_commander running?",
    )


def test_a_reset_of_the_finger_allowance_never_answered_times_out(client, monkeypatch):
    never = Future()
    monkeypatch.setattr(client._set_parameters, "call_async", lambda request: never)
    monkeypatch.setattr(client._set_parameters, "remove_pending_request", lambda future: None)

    with pytest.raises(StepError) as error:
        client.restore_gripper_touch(timeout_s=0.2)

    assert (error.value.code, error.value.message) == (
        "TIMEOUT",
        "The commander's /bimanual_commander/set_parameters did not answer within 0.2 s.",
    )


def test_a_reset_the_commander_refuses_is_a_skill_failure(client, fake_commander, monkeypatch):
    monkeypatch.setattr(commander_module, "TOUCH_SETTLE_S", 0.0)
    fake_commander.undeclare_parameter("allow_gripper_vs_octomap")

    with pytest.raises(StepError) as error:
        client.restore_gripper_touch()

    assert error.value.code == "SKILL_FAILED"
    assert error.value.message.startswith(
        "The commander refused allow_gripper_vs_octomap=False: "
    )


def test_a_gripper_goal_never_answered_times_out(
    client, fake_commander, monkeypatch, module_clock
):
    client.control_gripper("left_gripper", 0.044)
    clock = module_clock(commander_module)
    # The commander answers the cancel at once; the call waits for that
    # answer in real time.
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.never_answer = True

    closing = Call(client.control_gripper, "left_gripper", 0.0, timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        closing.result()

    assert error.value.code == "TIMEOUT"
    assert error.value.message == (
        "The left_gripper command did not finish within 0.3 s; the commander accepted the "
        "cancel, but the gripper finishes its current motion."
    )
    assert cancel_answer(error.value) == "accepted"
    # Stopped somewhere unknown.
    assert client.last_gripper_position("left_gripper") is None
    assert wait_until(lambda: fake_commander.results[-1:] == [("left_gripper", "Canceled")])


def test_an_older_gripper_call_cannot_overwrite_the_position_of_a_newer_one(client, monkeypatch):
    remember = client._remember_gripper
    paused = threading.Event()
    resume = threading.Event()

    def remember_slowly(group, position):
        # The close's result, kept back until the main thread has tried
        # its own command.
        if position == 0.0:
            paused.set()
            assert resume.wait(5.0)
        remember(group, position)

    monkeypatch.setattr(client, "_remember_gripper", remember_slowly)
    older = threading.Thread(target=lambda: client.control_gripper("left_gripper", 0.0))
    older.start()
    assert paused.wait(5.0)

    # The older call still holds the call lock while it records the close.
    try:
        with pytest.raises(StepError) as error:
            client.control_gripper("left_gripper", 0.044)
    finally:
        resume.set()
        older.join(timeout=5)

    assert error.value.code == "PRECONDITION_FAILED"
    assert client.last_gripper_position("left_gripper") == 0.0
    client.control_gripper("left_gripper", 0.044)
    assert client.last_gripper_position("left_gripper") == 0.044


def test_a_gripper_command_superseded_from_outside_ends_failed_and_unmeasured(
    client, fake_commander
):
    fake_commander.gripper_time_s = 1.0
    results = []
    closing = threading.Thread(
        target=lambda: results.append(client.control_gripper("left_gripper", 0.0))
    )
    closing.start()
    assert wait_until(lambda: fake_commander.joints()["openarm_left_finger_joint1"] < 0.04)

    # Another node commands the gripper while the close ramps.
    fake_commander.supersede("left_gripper", 0.044)
    closing.join(timeout=5)

    # The commander's answer looks like a measured empty close.
    [result] = results
    assert (result.success, result.status, result.grasped, result.gripper_gap) == (
        False,
        "Failed",
        False,
        0.0,
    )
    assert fake_commander.joints()["openarm_left_finger_joint1"] == 0.044
    assert fake_commander.results == [("left_gripper", "Failed")]


def test_a_read_since_a_gripper_command_is_refused_once_a_newer_one_came(client, fake_commander):
    close = client.control_gripper("left_gripper", 0.0)
    assert client.joint_states_since("left_gripper", close.command) == pytest.approx(
        {"openarm_left_finger_joint1": 0.0}
    )
    opened = client.control_gripper("left_gripper", 0.044)
    assert opened.command == close.command + 1
    # The other gripper counts its own commands.
    client.control_gripper("right_gripper", 0.0)
    requests = len(fake_commander.joint_state_requests)

    error = refused(lambda: client.joint_states_since("left_gripper", close.command))

    assert error.message == (
        f"Another left_gripper command came after command {close.command}, so a read now would "
        "not show where that one left the fingers."
    )
    assert len(fake_commander.joint_state_requests) == requests
    assert client.joint_states_since("left_gripper", opened.command) == pytest.approx(
        {"openarm_left_finger_joint1": 0.044}
    )


def test_a_gripper_call_that_sends_nothing_does_not_count_as_a_command(
    client, fake_commander, monkeypatch
):
    monkeypatch.setattr(client, "_ready_wait_s", 0.2)
    close = client.control_gripper("left_gripper", 0.0)
    fake_commander.vanish(CONTROL_GRIPPER)
    assert wait_until(lambda: not client._control_gripper.server_is_ready())

    with pytest.raises(StepError) as error:
        client.control_gripper("left_gripper", 0.044)

    assert error.value.code == "UNAVAILABLE"
    # Nothing was sent, so the close is still the gripper's latest command.
    assert client.joint_states_since("left_gripper", close.command) == pytest.approx(
        {"openarm_left_finger_joint1": 0.0}
    )


def test_a_gripper_goal_accepted_late_refuses_the_read_after_a_newer_close(
    client, fake_commander, monkeypatch, module_clock, caplog
):
    clock = module_clock(commander_module)
    gate = threading.Event()
    fake_commander.accept_gate = gate
    opening = Call(client.control_gripper, "left_gripper", 0.044, timeout_s=0.3)
    assert wait_until(lambda: fake_commander.gripper_goals)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        opening.result()
    assert cancel_answer(error.value) == "pending"
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    assert client.unverified == GRIPPER_ABANDONED
    client.read_arms()
    fake_commander.accept_gate = None
    close = client.control_gripper("left_gripper", 0.0)
    assert client.last_gripper_position("left_gripper") == 0.0
    late = (
        "the commander accepted the left_gripper command from an earlier call late, and the "
        "gripper makes that motion although it was asked to cancel it"
    )
    # The old command's ramp holds part way until the commander has its
    # cancel, which the commander accepts but the ramp does not heed.
    ramp = threading.Event()
    fake_commander.ramp_gate = ramp

    with caplog.at_level(logging.INFO, logger=commander_module.__name__):
        gate.set()
        assert wait_until(lambda: client.unverified == late)
        error = refused(lambda: client.joint_states_since("left_gripper", close.command))
        assert error.message == (
            f"Another left_gripper command came after command {close.command}, so a read now "
            "would not show where that one left the fingers."
        )
        assert wait_until(
            lambda: "Cancel of the left_gripper command, which the commander accepted late: the "
            "commander accepted the cancel, but the gripper finishes its current motion."
            in caplog.text
        )

    # The fingers make the old command's motion after the close.
    ramp.set()
    assert wait_until(lambda: fake_commander.results[-1:] == [("left_gripper", "Canceled")])
    assert fake_commander.joints()["openarm_left_finger_joint1"] == pytest.approx(0.044)
    assert client.last_gripper_position("left_gripper") is None
    error = refused(lambda: client.control_gripper("left_gripper", 0.0))
    assert error.message == unverified_refusal("move left_gripper", late)


# -- Availability, endpoints and arguments ------------------------------------------


def test_available_answers_at_once(client, monkeypatch):
    # A wait for discovery would take ready_wait_s, far past the bound.
    monkeypatch.setattr(client, "_ready_wait_s", 30.0)
    start = time.monotonic()

    assert client.available()
    assert time.monotonic() - start < 5.0


def test_without_a_commander_available_is_false_at_once_and_calls_fail(
    commander, monkeypatch, module_clock
):
    monkeypatch.setattr(commander, "_ready_wait_s", 30.0)
    start = time.monotonic()

    assert not commander.available()
    assert time.monotonic() - start < 5.0

    # Each call fails once ready_wait_s has passed on the test clock since
    # it began to wait for the commander, well before its deadline.
    clock = module_clock(commander_module)
    monkeypatch.setattr(commander, "_ready_wait_s", 0.2)

    def fails_at_ready_wait(entity, check: str, call) -> StepError:
        asked = readiness_checked(monkeypatch, entity, check)
        calling = Call(call)
        assert asked.wait(10)
        clock.advance(0.3)
        with pytest.raises(StepError) as error:
            calling.result()
        return error.value

    for entity, check, call, name in (
        (commander._joint_states, "service_is_ready",
         lambda: commander.read_arms(), "/get_joint_states service"),
        (commander._set_joint_position, "server_is_ready",
         lambda: commander.set_joint_position("left_arm", [0.0] * 7, 5.0),
         "/set_joint_position action"),
        (commander._control_gripper, "server_is_ready",
         lambda: commander.control_gripper("left_gripper", 0.0),
         "/control_gripper action"),
        (commander._parameters, "service_is_ready",
         lambda: commander.grasp_band(), "/bimanual_commander/get_parameters service"),
    ):
        error = fails_at_ready_wait(entity, check, call)
        assert (error.code, error.message) == (
            "UNAVAILABLE",
            f"The commander's {name} is not available. Is bimanual_commander running?",
        )


@pytest.mark.parametrize(
    ("entity", "check", "call", "what"),
    [
        ("_joint_states", "service_is_ready", lambda c: c.read_arms(), "the read of both_arms"),
        (
            "_set_joint_position",
            "server_is_ready",
            lambda c: c.set_joint_position("left_arm", [0.0] * 7, 60.0),
            "the move of left_arm",
        ),
    ],
    ids=["read", "move"],
)
def test_a_set_token_ends_the_wait_for_the_commander(
    commander, token, monkeypatch, entity, check, call, what
):
    # Far past the bound below, so only the token ends the wait.
    monkeypatch.setattr(commander, "_ready_wait_s", 30.0)
    asked = readiness_checked(monkeypatch, getattr(commander, entity), check)
    calling = Call(call, commander)
    assert asked.wait(10)
    start = time.monotonic()

    token.cancel()

    with pytest.raises(StepError) as error:
        calling.result()
    assert time.monotonic() - start < 5.0
    assert (error.value.code, error.value.message) == (
        "CANCELLED",
        f"A stop or a reset interrupted {what}.",
    )


@pytest.mark.parametrize(
    "name",
    [
        commander_module.SET_JOINT_POSITION_ACTION,
        commander_module.CONTROL_GRIPPER_ACTION,
        commander_module.GET_JOINT_STATES_SERVICE,
        commander_module.GET_PARAMETERS_SERVICE,
        commander_module.SET_PARAMETERS_SERVICE,
        commander_module.STOP_TOPIC,
    ],
)
def test_a_forbidden_name_refuses_the_client_before_it_creates_anything(
    connector, poses, name
):
    node = connector.node
    before = (len(list(node.clients)), len(list(node.publishers)), len(list(node.waitables)))

    with pytest.raises(StepError) as error:
        CommanderClient(connector, CancelToken(), poses, forbidden=(*FORBIDDEN_NAMES, name))

    assert (error.value.code, error.value.message) == (
        "REFUSED",
        f"Refused: a capability may not use {name}.",
    )
    assert (
        len(list(node.clients)), len(list(node.publishers)), len(list(node.waitables))
    ) == before


def test_the_bounds_given_to_the_client_are_its_defaults(
    connector, fake_commander, poses, monkeypatch, module_clock
):
    client = CommanderClient(
        connector, CancelToken(), poses, read_timeout_s=0.2, gripper_timeout_s=0.3
    )
    held = threading.Event()
    try:
        assert wait_until(client.available), "The client never discovered the fake commander."
        clock = module_clock(commander_module)
        # The commander answers the cancel at once; the call waits for that
        # answer in real time.
        monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
        fake_commander.joint_states_gate = held
        reading = Call(client.joint_states, "left_arm")
        assert wait_until(lambda: fake_commander.joint_state_requests == ["left_arm"])
        clock.advance(0.25)
        with pytest.raises(StepError) as error:
            reading.result()
        assert error.value.code == "TIMEOUT"
        assert "/get_joint_states did not answer within 0.2 s" in error.value.message
        held.set()
        fake_commander.joint_states_gate = None

        fake_commander.never_answer = True
        closing = Call(client.control_gripper, "left_gripper", 0.0)
        assert wait_until(lambda: accepted_handle(client) is not None)
        clock.advance(0.35)
        with pytest.raises(StepError) as error:
            closing.result()
        assert error.value.code == "TIMEOUT"
        assert error.value.message.startswith(
            "The left_gripper command did not finish within 0.3 s; "
        )
        assert wait_until(lambda: client.busy() is None)
    finally:
        held.set()
        client.destroy()


@pytest.mark.parametrize("value", [0, -1.0, math.nan, math.inf, True, "1"])
def test_a_bound_that_is_not_a_positive_number_of_seconds_is_a_bug(poses, value):
    for name in (
        "read_timeout_s",
        "gripper_timeout_s",
        "cancel_answer_s",
        "abandon_after_s",
        "absent_abandon_s",
        "ready_wait_s",
    ):
        # Checked before the client touches the connector.
        with pytest.raises(ValueError, match=name):
            CommanderClient(None, CancelToken(), poses, **{name: value})


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.set_joint_position("both_arms_lift", [0.0] * 15, 5.0),
        lambda c: c.set_joint_position("left_gripper", [0.0], 5.0),
        lambda c: c.set_joint_position("left_arm", [float("nan")] * 7, 5.0),
        lambda c: c.set_joint_position("left_arm", [0.0] * 7, math.inf),
        lambda c: c.control_gripper("left_arm", 0.0),
        lambda c: c.control_gripper("left_gripper", math.inf),
        lambda c: c.joint_states("both_arms_lift"),
        lambda c: c.joint_states("left_arm", timeout_s=0.0),
        lambda c: c.read_arms(timeout_s=math.nan),
        lambda c: c.grasp_band(timeout_s=-1.0),
        lambda c: c.last_gripper_position("left_arm"),
        lambda c: c.go_to_pose("right_arm_lift", HAND, 5.0),
        lambda c: c.go_to_pose("both_arms", HAND, 5.0),
        lambda c: c.go_to_pose("whole_body_right", HAND, 5.0, cartesian=True),
        lambda c: c.go_to_pose("right_arm", HandPose(1.0, 0.0, 0.8, 0.0, 0.0, 0.0, ""), 5.0),
        lambda c: c.go_to_pose("right_arm", HandPose(math.nan, 0, 0.8, 0, 0, 0, "map"), 5.0),
    ],
)
def test_a_group_or_value_the_agent_may_not_send_is_a_bug(commander, call):
    with pytest.raises(ValueError):
        call(commander)


def test_integer_positions_arrive_as_their_values(client, fake_commander):
    # rclpy accepts an int in a float64 field but sends it wrong, so the
    # client converts every value.
    client.set_joint_position("left_arm", [0, 0, 0, 1, 0, 0, 0], 5.0)

    assert fake_commander.joint_goals[-1] == ("left_arm", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    assert fake_commander.joints()["openarm_left_joint4"] == 1.0


# -- Pose goals (pick_object's reaches) ----------------------------------------------


def sent_pose(goal) -> tuple:
    return (
        goal.group_name,
        (goal.x, goal.y, goal.z, goal.roll, goal.pitch, goal.yaw),
        goal.cartesian_path,
        goal.frame_id,
    )


def test_a_cartesian_reach_of_one_arm_sends_the_hand_pose_and_forgets_the_arm(
    client, fake_commander
):
    home = client.read_arms()
    fake_commander.pose_joints = {"openarm_right_joint1": 0.4}

    assert client.go_to_pose("right_arm", HAND, 5.0, cartesian=True) == "Success"

    [goal] = fake_commander.pose_goals
    assert sent_pose(goal) == (
        "right_arm",
        (0.6, -0.2, 0.8, 0.0, pytest.approx(math.pi / 2), 0.3),
        True,
        "map",
    )
    # A pose goal has no joint target, so the arm is unknown until it is read.
    assert client.last_state == ArmsState(left=home.left, right=None)
    assert client.read_arms().right == ArmState(None, False)


def test_a_cartesian_reach_of_an_arm_with_the_lift_forgets_the_arm(client, fake_commander):
    home = client.read_arms()

    assert client.go_to_pose("right_arm_lift", HAND, 5.0, cartesian=True) == "Success"

    assert sent_pose(fake_commander.pose_goals[-1])[0::2] == ("right_arm_lift", True)
    assert client.last_state == ArmsState(left=home.left, right=None)


def test_a_joint_goal_of_an_arm_with_the_lift_moves_the_lift_and_the_arm(client, fake_commander):
    order = client.joint_names("right_arm_lift")
    assert order[0] == "lift_joint"
    targets = [0.7435, *([0.1] * (len(order) - 1))]

    assert client.set_joint_position("right_arm_lift", targets, 5.0) == "Success"

    assert fake_commander.joint_goals[-1] == ("right_arm_lift", tuple(targets))
    reading = client.joint_states("right_arm_lift")
    assert list(reading) == list(order)
    assert list(reading.values()) == pytest.approx(targets)


def test_a_whole_body_reach_returns_the_commanders_status(client, fake_commander):
    status = client.go_to_pose("whole_body_right", HAND, 5.0)

    assert status == WHOLE_BODY_SUCCESS
    assert sent_pose(fake_commander.pose_goals[-1])[0::2] == ("whole_body_right", False)
    assert client.last_state.right is None


@pytest.mark.parametrize(
    ("knob", "value", "message"),
    [
        (
            "plan_failure",
            True,
            "MoveIt found no path for whole_body_right (Planning Failed). The arms did not "
            "move.",
        ),
        (
            "pose_failure",
            "Refused: no plan reaches the target",
            "The commander reported 'Refused: no plan reaches the target' (ABORTED) for the "
            "move of whole_body_right.",
        ),
    ],
)
def test_a_reach_the_commander_fails_is_a_goal_failure(
    client, fake_commander, knob, value, message
):
    setattr(fake_commander, knob, value)

    with pytest.raises(GoalFailed) as error:
        client.go_to_pose("whole_body_right", HAND, 5.0)

    assert (error.value.code, error.value.message) == ("SKILL_FAILED", message)


def test_plan_only_reports_its_status_for_a_reach(client, fake_commander):
    fake_commander.plan_only = True

    assert client.go_to_pose("whole_body_right", HAND, 5.0) == "Plan Only (base fixed)"


def test_a_rejected_reach_is_internal(client, fake_commander):
    fake_commander.reject_goals = True

    with pytest.raises(StepError) as error:
        client.go_to_pose("right_arm", HAND, 5.0, cartesian=True)

    assert (error.value.code, error.value.message) == (
        "INTERNAL",
        "The commander rejected the pose goal for right_arm.",
    )


def test_a_stop_mid_reach_cancels_it_and_stops_the_arms(
    client, fake_commander, token, monkeypatch
):
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.move_time_s = 5.0
    fake_commander.pose_joints = {"openarm_right_joint1": 0.4}
    result = threading.Event()
    fake_commander.result_gate = result

    reaching = Call(client.go_to_pose, "whole_body_right", HAND, 10.0)
    assert wait_until(
        lambda: accepted_handle(client) is not None
        and fake_commander.joints()["openarm_right_joint1"] > 0.0
    )
    token.cancel()
    with pytest.raises(StepError) as error:
        reaching.result()

    assert error.value.code == "CANCELLED"
    assert error.value.message == (
        "A stop or a reset interrupted the reach of whole_body_right; the arms were told to "
        "stop."
    )
    assert fake_commander.cancel_requests == 1
    assert wait_until(lambda: fake_commander.results == [("whole_body_right", "Execution Failed")])
    result.set()
    assert wait_until(lambda: client.busy() is None)


def test_recovery_cancels_a_reach_whose_send_was_never_answered_by_its_id(
    client, fake_commander, monkeypatch, module_clock
):
    # As for a joint goal: the handle made from the id needs the pose action's type.
    clock = module_clock(commander_module)
    plan = threading.Event()
    fake_commander.plan_gate = plan
    client.read_arms()
    send_goal_async = client._go_to_pose.send_goal_async

    def send_and_lose_the_answer(goal, **kwargs):
        send_goal_async(goal, **kwargs)
        return Future()

    monkeypatch.setattr(client._go_to_pose, "send_goal_async", send_and_lose_the_answer)
    reaching = Call(client.go_to_pose, "right_arm", HAND, 0.3, cartesian=True)
    assert wait_until(lambda: fake_commander.pose_goals)
    clock.advance(0.4)
    with pytest.raises(StepError) as error:
        reaching.result()
    assert cancel_answer(error.value) == "pending"
    clock.advance(client._abandon_after_s + 1)
    assert client.busy() is None
    assert client.unverified == (
        "the reach of right_arm from an earlier call was given up on before its end was known"
    )

    recovering = Call(client.read_arms)
    assert wait_until(lambda: fake_commander.cancel_requests == 1)
    plan.set()
    assert wait_until(lambda: fake_commander.results == [("right_arm", "Canceled")])
    recovering.result()
    assert client.unverified is None
    assert client.last_recovery.goals == (("the reach of right_arm", "result"),)
