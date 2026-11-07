"""RosArms against FakeCommander over ROS, through a real CommanderClient.

The fake serves the commander's names and types on the `connector` fixture's Fast DDS
domain, with the groups of config/robot_model.toml, so these tests cover the answers
ctx.arms gives for every outcome the commander can give. The client's own waits,
cancels and recovery are covered by tests/test_commander.py; these tests cover what
RosArms makes of the client's answers.

A deadline test puts the client on a test clock (module_clock), so a deadline passes
only when the test moves the clock on. The helpers below are copies of those in
tests/test_commander.py.
"""

import inspect
import logging
import math
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.task import Future

from walkie_agent.arm.poses import (
    GRIPPER_GROUPS,
    RIGHT_JOINT2,
    ArmPoses,
    load_arm_poses,
    load_robot_model,
)
from walkie_agent.arm.state import (
    NOT_SAFE_SENTENCE,
    SAFE_SENTENCE,
    UNKNOWN_SENTENCE,
    ArmsState,
)
from walkie_agent.capabilities import Arms, ArmsReport
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import ErrorCode, GoalFailed, StepError
from walkie_agent.ros import arms as arms_module
from walkie_agent.ros import commander as commander_module
from walkie_agent.ros.arms import FINGER_JOINTS, NO_GRASP_SKILL, RosArms, interrupted
from walkie_agent.ros.commander import CommanderClient
from walkie_agent.sim.fake_commander import CONTROL_GRIPPER, FakeCommander

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "config" / "robot_model.toml"
POSES_PATH = PROJECT_ROOT / "config" / "arm_poses.toml"

# The executor threads of the fake's executor. A held goal, a held cancel and a slow
# read each keep one, so a test can need several at once.
FAKE_THREADS = 8

ROBOT_POSES = ["home", "hands_up", "standby"]
SIM_POSES = ["home", "hands_up", "standby", "tray", "receive", "carry", "point"]
# What follows a stop's message when the commander may still be moving an arm or the
# gripper.
WAIT_THEN_CHECK = "Wait a few seconds, then call get_arm_state."
# What follows a stop's message when the commander accepted the cancel of an arm goal,
# and of a gripper goal.
ARMS_TOLD_TO_STOP = (
    "They may take a moment to stop, so call get_arm_state before relying on where they are."
)
GRIPPER_FINISHES = "Wait a few seconds for it to finish."
STOP_IN_EFFECT = "a stop or a reset is in effect"
# A stop after a close of the left gripper ended, before its fingers were judged; the
# close leaves the arms as the earlier read found them, in home.
STOPPED_CHECKING_LEFT = (
    "Stopped before checking whether the left gripper holds something: "
    f"{STOP_IN_EFFECT}. {SAFE_SENTENCE}"
)
ABANDONED = "the move of left_arm from an earlier call was given up on before its end was known"
DRIVE = "the drive to kitchen"


# -- Test helpers (copies of tests/test_commander.py's) -----------------------------


def _stack_of(thread: threading.Thread) -> str:
    """The stack of `thread` right now, for a failure message."""
    frame = sys._current_frames().get(thread.ident)
    if frame is None:
        return f"{thread.name}: no longer running\n"
    return f"{thread.name}:\n{''.join(traceback.format_stack(frame))}"


def wait_until(
    predicate: Callable[[], Any], timeout_s: float = 10.0, poll_s: float = 0.01
) -> bool:
    """Whether `predicate` came true within `timeout_s`. For positive waits only."""
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
    held, since FakeCommander releases the goals and requests it holds only in
    destroy_node().
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
    """A manual monotonic() for a module's `time` global, which a test moves on."""

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


@pytest.fixture
def arms(client, token):
    """ctx.arms on the robot: only the poses tried on it."""
    return RosArms(client, token, sim=False)


@pytest.fixture
def sim_arms(client, token):
    """ctx.arms in the simulation: every offered pose."""
    return RosArms(client, token, sim=True)


def table(poses, pose: str, arm: str) -> dict[str, float]:
    return dict(poses.poses[pose].tables[arm])


def targets(poses, pose: str, *arms: str) -> dict[str, float]:
    return {joint: value for arm in arms for joint, value in poses.poses[pose].tables[arm].items()}


def in_pose(fake: FakeCommander, wanted: dict[str, float], tolerance: float = 1e-6) -> bool:
    joints = fake.joints()
    return all(abs(joints[joint] - value) <= tolerance for joint, value in wanted.items())


def move(client, group: str, targets: dict[str, float], timeout_s=5.0):
    """Send `targets` in the commander's joint order, past RosArms."""
    order = client.joint_names(group)
    return client.set_joint_position(group, [targets[j] for j in order], timeout_s)


def accepted_handle(client):
    """The goal handle of the client's running goal once the commander accepted it, else None.

    Not None only once the client asked for the result, so its call then waits for the
    result.
    """
    goal = client._goal
    if goal is None or goal.result is None:
        return None
    return goal.sent.result()


def error_of(call: Callable[[], Any]) -> StepError:
    with pytest.raises(StepError) as error:
        call()
    return error.value


def cancel_when(fake: FakeCommander, token: CancelToken, group: str) -> None:
    """Set the token when the fake gets the next read of `group`, before it answers.

    So a stop comes while the client waits for that read, and the read's answer comes
    only after it.
    """

    def hook(read: str, index: int) -> None:
        if read == group:
            token.cancel()

    fake.joint_states_hook = hook


# -- The Arms protocol --------------------------------------------------------------


def test_ros_arms_has_every_verb_of_the_arms_protocol_with_its_parameters():
    for name in ("poses", "state", "read", "move_to", "gripper", "pick", "place_on", "hand_over"):
        ours = inspect.signature(getattr(RosArms, name)).parameters
        theirs = inspect.signature(getattr(Arms, name)).parameters
        assert [(p.name, p.kind, p.default) for p in ours.values()] == [
            (p.name, p.kind, p.default) for p in theirs.values()
        ], name


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, True])
def test_a_floor_on_the_move_timeout_that_is_not_a_number_of_seconds_is_a_bug(
    commander, token, value
):
    with pytest.raises(ValueError, match="min_move_timeout_s"):
        RosArms(commander, token, sim=False, min_move_timeout_s=value)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_each_gripper_is_read_by_the_one_joint_of_its_group(model, arm):
    assert model.groups[GRIPPER_GROUPS[arm]] == (FINGER_JOINTS[arm],)


@pytest.mark.parametrize(
    "verb",
    [
        lambda arms: arms.pick("cup_1"),
        lambda arms: arms.place_on("table"),
        lambda arms: arms.hand_over("person_1", timeout_s=5.0),
    ],
    ids=["pick", "place_on", "hand_over"],
)
def test_there_is_no_grasp_skill_yet(commander, token, verb):
    error = error_of(lambda: verb(RosArms(commander, token, sim=True)))

    assert (error.code, error.message) == (ErrorCode.UNAVAILABLE, NO_GRASP_SKILL)
    assert NO_GRASP_SKILL == (
        "Walkie has no grasp skill yet, so it cannot pick, place or hand over objects. "
        "move_arm_to_pose and set_gripper still work."
    )


# -- poses and state ------------------------------------------------------------------


def test_the_robot_offers_only_the_poses_tried_on_it_and_the_simulation_every_one(
    arms, sim_arms
):
    assert list(arms.poses()) == ROBOT_POSES
    assert list(sim_arms.poses()) == SIM_POSES


def test_state_reads_the_arms_only_when_the_memory_cannot_say_whether_to_drive(
    arms, client, fake_commander
):
    assert arms.state(read=False) == ArmsReport(ArmsState())
    assert fake_commander.joint_state_requests == []

    report = arms.state()
    assert report.arms.left.pose == report.arms.right.pose == "home"
    assert (report.travel_safe, report.sentence()) == (True, SAFE_SENTENCE)
    assert (report.unverified, report.unverified_count, report.recovery) == (None, 0, None)
    # Known now, so the memory answers.
    assert arms.state() == report
    assert fake_commander.joint_state_requests == ["both_arms"]
    assert arms.state(read=True) == report
    assert fake_commander.joint_state_requests == ["both_arms", "both_arms"]


def test_state_never_raises_and_keeps_the_memory_when_its_read_fails(
    arms, client, fake_commander, token
):
    known = arms.state()
    fake_commander.read_failure = "Failed to get current robot state"

    # The read failed, and the memory still holds the last read.
    assert arms.state(read=True) == known
    assert len(fake_commander.joint_state_requests) == 2
    fake_commander.read_failure = None
    assert arms.state() == known

    token.cancel()
    assert arms.state(read=True) == known


def test_without_a_commander_state_and_the_drive_check_answer_at_once(
    commander, token, module_clock, caplog
):
    # The client's clock stands still, so a call that waited for the commander would
    # never return.
    module_clock(commander_module)
    arms = RosArms(commander, token, sim=False)

    assert Call(arms.state, read=True).result() == ArmsReport(ArmsState())
    with caplog.at_level(logging.WARNING):
        assert Call(arms.check_for_driving, DRIVE, 5.0).result() is None
    assert (
        f"Starting {DRIVE} with the arm state unknown: the commander's joint state service "
        "was not found."
    ) in caplog.text


# -- move_to ------------------------------------------------------------------------------


def test_a_move_of_both_arms_sends_the_pose_in_the_commanders_order(
    arms, fake_commander, poses
):
    # The LLM may write the name as it would say it.
    moved = arms.move_to("Hands up")

    assert (moved.pose, moved.arm, moved.status) == ("hands_up", "both", "Success")
    assert moved.max_error_rad < 1e-6
    assert (moved.report.arms.left.pose, moved.report.arms.right.pose) == ("hands_up", "hands_up")
    assert (moved.report.travel_safe, moved.report.sentence()) == (False, NOT_SAFE_SENTENCE)
    wanted = targets(poses, "hands_up", "left", "right")
    order = fake_commander.group_joints("both_arms")
    assert fake_commander.joint_goals == [("both_arms", tuple(wanted[joint] for joint in order))]
    assert in_pose(fake_commander, wanted)

    back = arms.move_to("home")
    assert (back.pose, back.status, back.report.travel_safe) == ("home", "Success", True)
    assert back.report.sentence() == SAFE_SENTENCE
    assert in_pose(fake_commander, targets(poses, "home", "left", "right"))


def test_a_move_follows_a_joint_order_unlike_the_srdfs(arms, fake_commander, poses):
    fake_commander.reverse_joint_order = True

    moved = arms.move_to("standby", "left")

    assert (moved.pose, moved.arm) == ("standby", "left")
    assert in_pose(fake_commander, targets(poses, "standby", "left"))


def test_a_one_arm_pose_moves_only_that_arm(sim_arms, fake_commander, poses):
    moved = sim_arms.move_to("receive", "left")

    [(group, positions)] = fake_commander.joint_goals
    assert (group, len(positions)) == ("left_arm", 7)
    assert in_pose(fake_commander, targets(poses, "receive", "left"))
    assert in_pose(fake_commander, targets(poses, "home", "right"))
    assert (moved.report.arms.left.pose, moved.report.arms.right.pose) == ("receive", "home")
    assert moved.report.sentence() == NOT_SAFE_SENTENCE

    # carry is travel-safe, and the right arm is still in home.
    carry = sim_arms.move_to("carry", "left")
    assert (carry.report.travel_safe, carry.report.sentence()) == (True, SAFE_SENTENCE)


@pytest.mark.parametrize(
    ("pose", "sim"),
    [("wave", False), ("receive", False), ("carry", False), ("pre_place", True)],
)
def test_an_unknown_or_hidden_pose_is_not_found(arms, sim_arms, fake_commander, pose, sim):
    offered = ", ".join(SIM_POSES if sim else ROBOT_POSES)

    error = error_of(lambda: (sim_arms if sim else arms).move_to(pose))

    # A hidden pose gets the same answer as a pose that does not exist, and the arms
    # are read for the sentence.
    assert (error.code, error.message) == (
        ErrorCode.NOT_FOUND,
        f"No pose named {pose!r}. The poses are: {offered}. {SAFE_SENTENCE}",
    )
    assert fake_commander.joint_goals == []


@pytest.mark.parametrize(
    ("pose", "arm", "message"),
    [
        ("receive", "right", "'receive' is for the left arm only."),
        ("receive", "both", "'receive' is for the left arm only."),
        ("carry", "both", "'carry' is for one arm at a time: left or right."),
    ],
)
def test_an_arm_the_pose_does_not_take_is_refused(sim_arms, fake_commander, pose, arm, message):
    error = error_of(lambda: sim_arms.move_to(pose, arm))

    assert (error.code, error.message) == (ErrorCode.REFUSED, f"{message} {SAFE_SENTENCE}")
    assert fake_commander.joint_goals == []


def test_right_joint2_fixed_drops_the_joint_at_zero_and_refuses_it_elsewhere(
    sim_arms, fake_commander, poses
):
    fake_commander.right_joint2_fixed = True

    moved = sim_arms.move_to("hands_up")

    assert moved.pose == "hands_up"
    [(group, positions)] = fake_commander.joint_goals
    assert (group, len(positions)) == ("both_arms", 13)
    wanted = targets(poses, "hands_up", "left", "right")
    del wanted[RIGHT_JOINT2]
    assert in_pose(fake_commander, wanted)

    # carry's right table needs the joint at 0.5 rad, which the fixed joint cannot reach.
    error = error_of(lambda: sim_arms.move_to("carry", "right"))
    assert error.code == ErrorCode.REFUSED
    assert error.message == (
        "'carry' cannot move the right arm while walkie-ros runs with right_joint2_fixed: "
        f"the pose needs {RIGHT_JOINT2} at 0.50 rad, and that joint is held at 0. Use the "
        f"left arm or another pose. {NOT_SAFE_SENTENCE}"
    )
    assert len(fake_commander.joint_goals) == 1
    # The left table has no such joint.
    assert sim_arms.move_to("carry", "left").pose == "carry"


def test_joints_that_do_not_match_the_robot_model_are_internal(
    arms, client, fake_commander, monkeypatch
):
    order = client.joint_names("both_arms")
    monkeypatch.setattr(client, "joint_names", lambda group: (*order, "openarm_left_joint9"))

    error = error_of(lambda: arms.move_to("standby"))

    assert error.code == ErrorCode.INTERNAL
    assert error.message == (
        "The commander's joints for both do not match config/robot_model.toml (missing: none; "
        "unknown: openarm_left_joint9). Regenerate the snapshot for the running walkie-ros. "
        f"{SAFE_SENTENCE}"
    )
    assert fake_commander.joint_goals == []


def test_a_planning_failure_says_the_arms_did_not_move_after_reading_them(
    arms, fake_commander
):
    fake_commander.plan_failure = True

    error = error_of(lambda: arms.move_to("standby"))

    # The goal ended, so it stays a GoalFailed.
    assert isinstance(error, GoalFailed)
    assert error.code == ErrorCode.SKILL_FAILED
    # The goal made the arms unknown, and the read after the failure shows them in home.
    assert error.message == (
        "Could not move to 'standby': MoveIt found no path for both_arms (Planning Failed). "
        f"The arms did not move. {SAFE_SENTENCE}"
    )
    assert fake_commander.joint_state_requests[-1] == "both_arms"


def test_an_execution_failure_reads_where_the_arms_stopped(arms, fake_commander):
    fake_commander.execution_failure = True

    error = error_of(lambda: arms.move_to("hands_up"))

    assert isinstance(error, GoalFailed)
    # Half way to hands_up is in no named pose.
    assert error.code == ErrorCode.SKILL_FAILED
    assert error.message == (
        "Could not move to 'hands_up': The motion of both_arms stopped before the target "
        f"(Execution Failed). The arms may have stopped part way. {NOT_SAFE_SENTENCE}"
    )


def test_a_rejected_goal_is_internal_with_the_group_and_count(arms, fake_commander):
    fake_commander.reject_goals = True

    error = error_of(lambda: arms.move_to("hands_up", "right"))

    assert error.code == ErrorCode.INTERNAL
    assert error.message == (
        "Could not move to 'hands_up': The commander rejected the goal for right_arm with 7 "
        "positions. It rejects a count that differs from the group's joints; the joint order "
        f"is read again on the next call. {SAFE_SENTENCE}"
    )


def test_a_joint_left_off_its_target_fails_the_check_after_a_reported_success(
    arms, fake_commander
):
    fake_commander.joint_errors = {"openarm_left_joint4": 0.31}

    error = error_of(lambda: arms.move_to("standby"))

    # The goal ended, but the arms are not where it said.
    assert type(error) is StepError
    assert error.code == ErrorCode.SKILL_FAILED
    assert error.message == (
        "The commander reported success, but openarm_left_joint4 is 0.31 rad from 'standby'. "
        f"Treat this as a failure. {NOT_SAFE_SENTENCE}"
    )


def test_a_move_whose_reading_afterwards_fails_cannot_be_checked(
    arms, client, fake_commander, poses
):
    # The joint order is known from an earlier read, so the move is sent.
    client.joint_names("both_arms")
    fake_commander.read_failure = "Failed to get current robot state"

    error = error_of(lambda: arms.move_to("standby"))

    assert error.code == ErrorCode.SKILL_FAILED
    assert error.message == (
        "The commander reported success, but the arms could not be checked against 'standby': "
        "The commander could not read the joints of both_arms: Failed to get current robot "
        f"state. Treat this as a failure. {UNKNOWN_SENTENCE}"
    )
    assert in_pose(fake_commander, targets(poses, "standby", "left", "right"))


def test_a_move_whose_reading_afterwards_lacks_a_joint_cannot_be_checked(
    arms, client, fake_commander
):
    client.joint_names("both_arms")
    fake_commander.missing_joints = {"openarm_left_joint4"}

    error = error_of(lambda: arms.move_to("standby"))

    # The left arm is unknown, and the right one is read in standby.
    assert error.code == ErrorCode.SKILL_FAILED
    assert error.message == (
        "The commander reported success, but its reading lacks a usable position for "
        "openarm_left_joint4, so the arms cannot be checked against 'standby'. Treat this as a "
        f"failure. {NOT_SAFE_SENTENCE}"
    )


def test_a_joint_read_as_not_a_number_after_a_move_fails_the_check(arms, fake_commander):
    # A NaN passes every tolerance check, since every comparison with it is false, so
    # the move must not count as arrived.
    fake_commander.joint_errors = {"openarm_left_joint1": math.nan}

    error = error_of(lambda: arms.move_to("standby"))

    assert error.code == ErrorCode.SKILL_FAILED
    assert error.message == (
        "The commander reported success, but the arms could not be checked against 'standby': "
        "The commander's reading of both_arms is not usable: openarm_left_joint1 = nan. Treat "
        f"this as a failure. {UNKNOWN_SENTENCE}"
    )


def test_plan_only_never_says_the_arms_moved(arms, fake_commander):
    fake_commander.plan_only = True

    error = error_of(lambda: arms.move_to("standby"))

    # The joint check catches it, and the answer says why nothing moved.
    assert error.code == ErrorCode.SKILL_FAILED
    assert error.message == (
        "The commander runs in plan-only mode, so it moved nothing, and openarm_left_joint4 is "
        "1.82 rad from 'standby'. Treat this as a failure. No arm can move until the commander "
        f"restarts without plan_only, so tell the person instead of trying again. {SAFE_SENTENCE}"
    )

    # Already in the pose, the check passes, and the status says that nothing moved.
    moved = arms.move_to("home")
    assert (moved.pose, moved.status, moved.report.travel_safe) == ("home", "Plan Only", True)
    assert fake_commander.results == [("both_arms", "Plan Only"), ("both_arms", "Plan Only")]


@pytest.mark.parametrize(("floor_s", "bound"), [(0.0, "0.5"), (2.0, "2")])
def test_a_move_waits_for_the_longer_of_its_pose_timeout_and_the_floor(
    connector, fake_commander, token, poses, monkeypatch, module_clock, floor_s, bound
):
    quick = ArmPoses(
        poses={**poses.poses, "hands_up": replace(poses.poses["hands_up"], timeout_s=0.5)},
        grippers=poses.grippers,
    )
    client = CommanderClient(connector, token, quick)
    try:
        assert wait_until(client.available)
        monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
        client.read_arms()
        clock = module_clock(commander_module)
        fake_commander.never_answer = True
        arms = RosArms(client, token, sim=False, min_move_timeout_s=floor_s)

        moving = Call(error_of, lambda: arms.move_to("hands_up"))
        assert wait_until(lambda: accepted_handle(client) is not None)
        clock.advance(max(0.5, floor_s) + 0.1)
        error = moving.result()

        # The commander did not finish, so the sentence comes from the memory.
        assert error.code == ErrorCode.TIMEOUT
        assert error.message == (
            "Could not move to 'hands_up': The move of both_arms did not finish within "
            f"{bound} s; the arms were told to stop. {UNKNOWN_SENTENCE}"
        )
        assert fake_commander.cancel_requests == 1
        assert wait_until(lambda: client.busy() is None)
    finally:
        client.destroy()


def test_a_move_names_the_action_when_its_server_is_not_ready(arms, client, monkeypatch):
    monkeypatch.setattr(client, "_ready_wait_s", 0.2)
    # The service answers, so only the action is missing.
    monkeypatch.setattr(client._set_joint_position, "server_is_ready", lambda: False)

    error = error_of(lambda: arms.move_to("hands_up"))

    assert error.code == ErrorCode.UNAVAILABLE
    assert error.message == (
        "Could not move to 'hands_up': The commander's /set_joint_position action is not "
        f"available. Is bimanual_commander running? {UNKNOWN_SENTENCE}"
    )


def test_a_move_the_commander_cancels_itself_reads_the_arms(arms, client, fake_commander):
    # The goal plans until another client of the commander cancels it.
    fake_commander.plan_until_cancelled = True

    moving = Call(error_of, lambda: arms.move_to("hands_up"))
    assert wait_until(lambda: accepted_handle(client) is not None)
    accepted_handle(client).cancel_goal_async()
    error = moving.result()

    # Not a stop: the token is not set, so the arms are read for the sentence.
    assert error.code == ErrorCode.CANCELLED
    assert error.message == (
        "Could not move to 'hands_up': The commander cancelled the move of both_arms. "
        f"{SAFE_SENTENCE}"
    )
    assert fake_commander.results == [("both_arms", "Canceled")]


# -- A stop or a reset -----------------------------------------------------------------


def test_a_stop_mid_move_cancels_the_goal_and_says_to_check_the_arms(
    arms, client, fake_commander, poses, token, monkeypatch
):
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.move_time_s = 5.0

    moving = Call(error_of, lambda: arms.move_to("hands_up"))
    assert wait_until(lambda: accepted_handle(client) is not None)
    token.cancel()
    error = moving.result()

    # The client's message, what to do about the cancel it accepted, and the sentence
    # from the memory, which the move made unknown.
    assert error.code == ErrorCode.CANCELLED
    assert error.message == (
        "A stop or a reset interrupted the move of both_arms; the arms were told to stop. "
        f"{ARMS_TOLD_TO_STOP} {UNKNOWN_SENTENCE}"
    )
    assert fake_commander.cancel_requests == 1
    assert wait_until(lambda: fake_commander.results == [("both_arms", "Execution Failed")])
    assert not in_pose(fake_commander, targets(poses, "hands_up", "left", "right"), 0.05)


@pytest.mark.parametrize("read_first", [False, True], ids=["unread", "read"])
def test_a_stop_before_the_goal_is_sent_says_the_arms_did_not_move(
    arms, client, fake_commander, token, read_first
):
    # A read leaves the joint order known, so the move gets as far as its send.
    if read_first:
        arms.state()
    token.cancel()

    error = error_of(lambda: arms.move_to("hands_up"))

    sentence = SAFE_SENTENCE if read_first else UNKNOWN_SENTENCE
    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        f"Stopped before the arms moved: {STOP_IN_EFFECT}. {sentence}",
    )
    assert fake_commander.joint_goals == []


def test_a_stop_during_the_check_after_a_move_says_the_arms_were_not_checked(
    arms, client, fake_commander, token
):
    client.joint_names("both_arms")
    cancel_when(fake_commander, token, "both_arms")

    error = error_of(lambda: arms.move_to("hands_up"))

    assert error.code == ErrorCode.CANCELLED
    assert error.message == (
        f"Stopped before the arms were checked against 'hands_up': {STOP_IN_EFFECT}. "
        f"{UNKNOWN_SENTENCE}"
    )
    assert fake_commander.results == [("both_arms", "Success")]


def test_a_stop_mid_gripper_command_says_the_gripper_finishes_its_motion(
    arms, client, fake_commander, token, monkeypatch
):
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    arms.state()
    # The ramp holds after its first step, until the test lets it finish.
    ramp = threading.Event()
    fake_commander.ramp_gate = ramp

    closing = Call(error_of, lambda: arms.gripper("left", "close"))
    assert wait_until(lambda: accepted_handle(client) is not None)
    token.cancel()
    error = closing.result()

    # A gripper goal leaves the arms known.
    assert error.code == ErrorCode.CANCELLED
    assert error.message == (
        "A stop or a reset interrupted the left_gripper command; the commander accepted the "
        f"cancel, but the gripper finishes its current motion. {GRIPPER_FINISHES} "
        f"{SAFE_SENTENCE}"
    )
    assert fake_commander.cancel_requests == 1
    ramp.set()
    assert wait_until(lambda: client.busy() is None)
    assert fake_commander.results == [("left_gripper", "Canceled")]


def test_a_stop_before_a_gripper_command_is_sent_says_it_did_not_move(
    arms, fake_commander, token
):
    token.cancel()

    error = error_of(lambda: arms.gripper("left", "close"))

    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        f"Stopped before the left gripper moved: {STOP_IN_EFFECT}. {UNKNOWN_SENTENCE}",
    )
    assert fake_commander.gripper_goals == []


def test_a_stop_while_the_fingers_are_read_after_a_close_stops_the_call(
    arms, fake_commander, token
):
    arms.state()
    cancel_when(fake_commander, token, "left_gripper")

    error = error_of(lambda: arms.gripper("left", "close"))

    assert error.code == ErrorCode.CANCELLED
    assert error.message == STOPPED_CHECKING_LEFT
    assert fake_commander.gripper_goals == [("left_gripper", 0.0)]


@pytest.mark.parametrize("gripper", [False, True], ids=["arm", "gripper"])
@pytest.mark.parametrize("outcome", list(commander_module.CANCEL_OUTCOMES) + ["pending"])
def test_interrupted_says_what_to_do_after_each_answer_to_the_cancel(outcome, gripper):
    side = 1 if gripper else 0
    if outcome == "pending":
        words = commander_module.PENDING_OUTCOMES[side]
    else:
        words = commander_module.CANCEL_OUTCOMES[outcome][side]
    message = f"A stop or a reset interrupted the move of both_arms; {words}."
    after = {
        "accepted": GRIPPER_FINISHES if gripper else ARMS_TOLD_TO_STOP,
        # Nothing to wait for.
        "ended": None,
    }.get(outcome, WAIT_THEN_CHECK)

    text = interrupted(StepError(ErrorCode.CANCELLED, message), gripper)

    assert text == (message if after is None else f"{message} {after}")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # The commander's own report of a cancelled goal: the goal was sent and ended.
        (
            "The commander cancelled the move of both_arms.",
            "The commander cancelled the move of both_arms.",
        ),
        (
            "The commander cancelled the left_gripper command (status 'Canceled').",
            "The commander cancelled the left_gripper command (status 'Canceled').",
        ),
        # No goal was sent.
        ("Cannot move both_arms: a stop or a reset is in effect.", None),
        ("A stop or a reset interrupted the move of both_arms.", None),
        ("Cancelled the move of both_arms: a stop is in effect.", None),
    ],
)
def test_interrupted_tells_a_goal_that_was_sent_from_one_that_was_not(message, expected):
    assert interrupted(StepError(ErrorCode.CANCELLED, message), False) == expected


# -- gripper --------------------------------------------------------------------------


@pytest.mark.parametrize(("state", "position"), [("open", 0.044), ("half", 0.036)])
def test_an_open_or_half_reports_only_that_it_moved(arms, fake_commander, state, position):
    # An object between the fingers: on an open the commander still reports grasped
    # false and a 0.0 gap, which must not read as "holding nothing".
    fake_commander.object_widths["left_gripper"] = 0.02

    moved = arms.gripper("left", state)

    assert (moved.arm, moved.state, moved.outcome, moved.holding) == (
        "left",
        state,
        "moved",
        None,
    )
    assert (moved.result.success, moved.result.grasped, moved.result.gripper_gap) == (
        True,
        None,
        None,
    )
    assert (moved.finger_m, moved.read_error) == (None, None)
    assert fake_commander.gripper_goals == [("left_gripper", position)]


def test_a_half_that_an_object_stops_short_did_not_reach(arms, fake_commander):
    fake_commander.object_widths["right_gripper"] = 0.04

    moved = arms.gripper("right", "half")

    assert (moved.outcome, moved.result.success, moved.result.status) == (
        "not_reached",
        False,
        "Failed",
    )


def test_a_close_on_an_object_holds_it(arms, fake_commander):
    fake_commander.object_widths["left_gripper"] = 0.02

    moved = arms.gripper("left", "close")

    # The fingers stall on the object, so the goal fails; grasped is the answer.
    assert (moved.outcome, moved.holding, moved.result.success) == ("holding", True, False)
    assert moved.finger_m == pytest.approx(0.02)
    assert fake_commander.gripper_goals == [("left_gripper", 0.0)]
    assert fake_commander.joint_state_requests[-1] == "left_gripper"


def test_a_close_on_nothing_holds_nothing(arms):
    moved = arms.gripper("right", "close")

    assert (moved.outcome, moved.holding) == ("empty", False)
    assert moved.finger_m == pytest.approx(0.0)


def test_a_stuck_gripper_did_not_close(arms, fake_commander):
    # The fingers stop near the open position, inside the commander's grasp band:
    # grasped is true, but nothing that thin is held that wide.
    fake_commander.object_widths["left_gripper"] = 0.043

    moved = arms.gripper("left", "close")

    assert (moved.outcome, moved.holding, moved.result.grasped) == ("not_closed", None, True)
    assert moved.finger_m == pytest.approx(0.043)


def test_grasp_detect_min_below_zero_reads_an_empty_close_as_held(
    arms, client, fake_commander, monkeypatch
):
    # A band that includes a closed gripper reads every close as a grasp.
    fake_commander.set_parameters([Parameter("grasp_detect_min", value=-1.0)])
    band_reads = []
    monkeypatch.setattr(client, "grasp_band", lambda: band_reads.append(1))

    moved = arms.gripper("left", "close")

    assert (moved.outcome, moved.holding) == ("holding", True)
    # The commander's gap agrees with the read, so its grasped decides, and the band
    # is not read.
    assert band_reads == []


@pytest.mark.parametrize(
    ("width", "outcome"), [(0.02, "holding"), (None, "empty")], ids=["object", "empty"]
)
def test_a_close_reported_reached_without_a_measurement_is_judged_from_a_read(
    arms, fake_commander, monkeypatch, width, outcome
):
    # A threshold below every goal: the commander runs no grasp detection and reports
    # grasped false with a gap of 0.0, as with grasp_detect_enable off.
    fake_commander.set_parameters([Parameter("grasp_close_threshold", value=-1.0)])
    fake_commander.object_widths["left_gripper"] = width
    finish = fake_commander._finish

    def reached(goal_handle, result, group, success, status):
        # A controller that allows stalling reports fingers stopped on an object as
        # having reached the position.
        if group == "left_gripper":
            success, status = True, "Success"
        return finish(goal_handle, result, group, success, status)

    monkeypatch.setattr(fake_commander, "_finish", reached)

    moved = arms.gripper("left", "close")

    # With an object, the gap of 0.0 does not agree with the read, so the band decides.
    assert moved.outcome == outcome


def measure_before_settling(fake: FakeCommander, monkeypatch, gap: float = 0.0077) -> None:
    """Make the fake measure an empty close of the left gripper at `gap`, before it settles.

    As the commander can in a loaded simulation: it reads the finger once the
    controller answers, and then judges that gap by its band.
    """
    finish = fake._finish

    def early(goal_handle, result, group, success, status):
        if group == "left_gripper":
            low = fake.get_parameter("grasp_detect_min").value
            high = fake.get_parameter("grasp_detect_max").value
            result.gripper_gap = gap
            result.grasped = low <= gap <= high
        return finish(goal_handle, result, group, success, status)

    monkeypatch.setattr(fake, "_finish", early)


@pytest.mark.parametrize(
    ("detect_min", "outcome"), [(-1.0, "holding"), (0.005, "empty")], ids=["hung", "default"]
)
def test_a_close_measured_before_the_fingers_settle_is_judged_by_the_commanders_band(
    arms, fake_commander, monkeypatch, detect_min, outcome
):
    # The HRI scenario hangs a bag by setting grasp_detect_min to -1. Either way the
    # commander reports grasped true, from a gap 7.7 mm from the read.
    fake_commander.set_parameters([Parameter("grasp_detect_min", value=detect_min)])
    measure_before_settling(fake_commander, monkeypatch)

    moved = arms.gripper("left", "close")

    # The read, about 0.0, is judged by the band the commander holds now.
    assert moved.result.grasped is True
    assert moved.outcome == outcome


def test_a_close_whose_band_cannot_be_read_is_judged_by_the_default_band(
    arms, client, fake_commander, monkeypatch, caplog
):
    fake_commander.set_parameters([Parameter("grasp_detect_min", value=-1.0)])
    measure_before_settling(fake_commander, monkeypatch)

    def no_answer():
        raise StepError(ErrorCode.TIMEOUT, "The commander's band did not come.")

    monkeypatch.setattr(client, "grasp_band", no_answer)

    with caplog.at_level(logging.WARNING, logger=arms_module.__name__):
        moved = arms.gripper("left", "close")

    # The default band, 0.005 to 0.044 m, does not hold a read of about 0.0.
    assert moved.outcome == "empty"
    assert (
        "Judging a close by the commander's default grasp detection band: The commander's band "
        "did not come."
    ) in caplog.text


def test_a_stop_while_the_band_is_read_stops_the_call(
    arms, client, fake_commander, token, monkeypatch
):
    arms.state()
    measure_before_settling(fake_commander, monkeypatch)
    grasp_band = client.grasp_band

    def stopped_first():
        token.cancel()
        return grasp_band()

    monkeypatch.setattr(client, "grasp_band", stopped_first)

    error = error_of(lambda: arms.gripper("left", "close"))

    assert error.code == ErrorCode.CANCELLED
    assert error.message == STOPPED_CHECKING_LEFT


@pytest.mark.parametrize("width", [0.02, 0.043], ids=["object", "barely moved"])
def test_a_close_reported_failed_without_a_measurement_is_unmeasured(
    arms, fake_commander, width
):
    # No grasp detection, and the fingers stall on the object, so the commander
    # reports the close failed with grasped false and a gap of 0.0, as it does for a
    # close that a newer command superseded.
    fake_commander.set_parameters([Parameter("grasp_close_threshold", value=-1.0)])
    fake_commander.object_widths["left_gripper"] = width

    moved = arms.gripper("left", "close")

    assert (moved.outcome, moved.holding, moved.finger_m) == ("unmeasured", None, None)
    # The fingers may show a newer command, so they are not read.
    assert "left_gripper" not in fake_commander.joint_state_requests


def test_a_close_that_another_node_supersedes_is_unmeasured(arms, fake_commander):
    fake_commander.gripper_time_s = 1.0
    # The ramp holds after its first step, so the other node's command always comes
    # mid-ramp.
    fake_commander.ramp_gate = threading.Event()
    finger = FINGER_JOINTS["left"]
    opened = fake_commander.joints()[finger]

    def supersede_mid_ramp() -> None:
        assert wait_until(lambda: fake_commander.joints()[finger] < opened)
        # Half open: a read would find the fingers inside the band.
        fake_commander.supersede("left_gripper", 0.036)

    superseding = Call(supersede_mid_ramp)
    moved = arms.gripper("left", "close")
    superseding.result()

    assert fake_commander.results == [("left_gripper", "Failed")]
    assert moved.outcome == "unmeasured"
    assert "left_gripper" not in fake_commander.joint_state_requests


def test_a_close_followed_by_another_command_before_its_read_is_superseded(
    arms, client, fake_commander, monkeypatch
):
    # The commander measures the close on the object. Another call then sets the
    # gripper half open before the close's read, which would find the fingers inside
    # the band.
    fake_commander.object_widths["left_gripper"] = 0.02
    joint_states_since = client.joint_states_since

    def another_command_first(group, command, timeout_s=None):
        client.control_gripper("left_gripper", 0.036)
        return joint_states_since(group, command, timeout_s)

    monkeypatch.setattr(client, "joint_states_since", another_command_first)

    moved = arms.gripper("left", "close")

    assert (moved.outcome, moved.holding, moved.finger_m) == ("superseded", None, None)
    assert fake_commander.gripper_goals == [("left_gripper", 0.0), ("left_gripper", 0.036)]
    # The read is refused before it is made.
    assert "left_gripper" not in fake_commander.joint_state_requests


def test_a_close_whose_fingers_cannot_be_read_is_unread(arms, fake_commander):
    fake_commander.missing_joints = {"openarm_right_finger_joint1"}

    moved = arms.gripper("right", "close")

    # The close ended as the commander allows, so the call does not fail.
    assert (moved.outcome, moved.holding, moved.finger_m) == ("unread", None, None)
    assert moved.read_error == (
        "The commander answered 0 joint names and 0 positions for right_gripper."
    )


def test_a_close_whose_reading_lacks_the_finger_joint_is_unread(arms, client, monkeypatch):
    monkeypatch.setattr(
        client, "joint_states_since", lambda group, command: {"openarm_left_joint1": 0.0}
    )

    moved = arms.gripper("left", "close")

    assert (moved.outcome, moved.read_error) == (
        "unread",
        "The commander's reading of left_gripper lacks openarm_left_finger_joint1.",
    )


def test_a_gripper_command_the_commander_cannot_take_ends_with_the_sentence(
    arms, client, fake_commander, monkeypatch
):
    arms.state()
    monkeypatch.setattr(client, "_ready_wait_s", 0.2)
    fake_commander.vanish(CONTROL_GRIPPER)
    assert wait_until(lambda: not client._control_gripper.server_is_ready())

    error = error_of(lambda: arms.gripper("left", "open"))

    assert error.code == ErrorCode.UNAVAILABLE
    assert error.message == (
        "Could not move the left gripper: The commander's /control_gripper action is not "
        f"available. Is bimanual_commander running? {SAFE_SENTENCE}"
    )


# -- read ---------------------------------------------------------------------------------


def test_read_reads_the_arms_and_the_grippers(arms, fake_commander):
    arms.state()
    arms.gripper("left", "close")

    reading = arms.read()

    assert fake_commander.joint_state_requests[-3:] == [
        "both_arms",
        "left_gripper",
        "right_gripper",
    ]
    assert reading.before == reading.after
    assert (reading.after.arms.left.pose, reading.after.sentence()) == ("home", SAFE_SENTENCE)
    left, right = reading.grippers
    assert (left.arm, left.reads_as, left.read_error, left.commanded) == (
        "left",
        "close",
        None,
        None,
    )
    assert left.finger_m == pytest.approx(0.0)
    # Never commanded, but read: the fingers start open.
    assert (right.arm, right.reads_as) == ("right", "open")
    assert right.finger_m == pytest.approx(0.044)


def test_read_gives_the_before_report_from_before_its_read(arms, fake_commander, poses):
    fake_commander.teleport(targets(poses, "standby", "left"))

    reading = arms.read()

    assert reading.before == ArmsReport(ArmsState())
    assert (reading.after.arms.left.pose, reading.after.arms.right.pose) == ("standby", "home")


def test_a_gripper_that_cannot_be_read_falls_back_to_its_last_command(arms, fake_commander):
    arms.gripper("left", "close")
    # A finger read as not a number, and one the reading lacks.
    fake_commander.teleport({"openarm_left_finger_joint1": math.nan})
    fake_commander.missing_joints = {"openarm_right_finger_joint1"}

    reading = arms.read()

    left, right = reading.grippers
    assert (left.finger_m, left.reads_as, left.commanded) == (None, None, "close")
    assert left.read_error == (
        "The commander's reading of left_gripper is not usable: "
        "openarm_left_finger_joint1 = nan."
    )
    assert (right.finger_m, right.reads_as, right.commanded) == (None, None, None)
    assert right.read_error == (
        "The commander answered 0 joint names and 0 positions for right_gripper."
    )
    assert reading.after.sentence() == SAFE_SENTENCE


def test_a_gripper_reading_that_lacks_the_finger_joint_is_not_read(
    arms, client, monkeypatch
):
    arms.gripper("right", "half")
    joint_states = client.joint_states

    def without_fingers(group, timeout_s=None):
        if group in GRIPPER_GROUPS.values():
            return {"openarm_left_joint1": 0.0}
        return joint_states(group, timeout_s)

    monkeypatch.setattr(client, "joint_states", without_fingers)

    left, right = arms.read().grippers

    assert left.read_error == (
        "The commander's reading of left_gripper lacks openarm_left_finger_joint1."
    )
    assert (right.read_error, right.commanded) == (
        "The commander's reading of right_gripper lacks openarm_right_finger_joint1.",
        "half",
    )


def test_read_raises_the_errors_of_the_read_of_the_arms(arms, fake_commander, token):
    fake_commander.read_failure = "Failed to get current robot state"

    error = error_of(arms.read)
    assert error.code == ErrorCode.UNAVAILABLE
    assert error.message == (
        "The commander could not read the joints of both_arms: Failed to get current robot state"
    )

    fake_commander.read_failure = None
    token.cancel()
    error = error_of(arms.read)
    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        f"Cannot read the joints of both_arms: {STOP_IN_EFFECT}.",
    )


def test_a_stop_while_a_gripper_is_read_ends_the_read(arms, fake_commander, token):
    cancel_when(fake_commander, token, "left_gripper")

    error = error_of(arms.read)

    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        "A stop or a reset interrupted the read of left_gripper.",
    )
    assert "right_gripper" not in fake_commander.joint_state_requests


# -- The drive check (ADR-0025) ---------------------------------------------------------


def test_the_drive_check_passes_with_both_arms_travel_safe(arms, fake_commander, poses):
    assert arms.check_for_driving(DRIVE, 5.0) is None

    # carry is travel-safe too: the robot guides the guest holding the bag.
    fake_commander.teleport(targets(poses, "carry", "left"))
    assert arms.check_for_driving(DRIVE, 5.0) is None
    assert fake_commander.joint_state_requests == ["both_arms", "both_arms"]


def test_the_drive_check_refuses_with_an_arm_out(arms, fake_commander, poses):
    fake_commander.teleport(targets(poses, "receive", "left"))

    error = error_of(lambda: arms.check_for_driving(DRIVE, 5.0))
    assert error.code == ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        f"Not starting {DRIVE}: the left arm is not in a pose that is safe for driving. Move "
        "it with move_arm_to_pose first, for example 'home'."
    )

    fake_commander.teleport(targets(poses, "hands_up", "right"))
    error = error_of(lambda: arms.check_for_driving(DRIVE, 5.0))
    assert error.message == (
        f"Not starting {DRIVE}: the arms are not in a pose that is safe for driving. Move "
        "them with move_arm_to_pose first, for example 'home'."
    )


def test_the_drive_check_reads_within_the_shorter_of_its_budget_and_the_read_bound(
    arms, client, monkeypatch
):
    bounds = []
    check = client.check_for_driving

    def noted(timeout_s=None):
        bounds.append(timeout_s)
        return check(timeout_s)

    monkeypatch.setattr(client, "check_for_driving", noted)

    arms.check_for_driving(DRIVE, 100.0)
    arms.check_for_driving(DRIVE, 0.5)

    assert client.read_timeout_s == commander_module.READ_TIMEOUT_S
    assert bounds == [client.read_timeout_s, 0.5]


def test_the_drive_check_fails_open_when_the_commander_vanishes(
    arms, client, fake_commander, caplog
):
    fake_commander.vanish()
    assert wait_until(lambda: not client._joint_states.service_is_ready())

    with caplog.at_level(logging.WARNING):
        assert arms.check_for_driving(DRIVE, 5.0) is None

    assert (
        f"Starting {DRIVE} with the arm state unknown: the commander's joint state service "
        "was not found."
    ) in caplog.text


def test_the_drive_check_refuses_while_the_arms_are_unverified(
    arms, client, fake_commander, poses, monkeypatch, module_clock
):
    clock = module_clock(commander_module)
    monkeypatch.setattr(client, "_cancel_answer_s", 10.0)
    fake_commander.never_answer = True
    fake_commander.reject_cancels = True
    client.read_arms()
    client.joint_names("left_arm")

    # A goal whose cancel the commander refuses, and whose server then goes away.
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: accepted_handle(client) is not None)
    clock.advance(0.4)
    assert error_of(moving.result).code == ErrorCode.TIMEOUT
    fake_commander.vanish()
    assert wait_until(lambda: not client._set_joint_position.server_is_ready())
    assert client.busy() == "the move of left_arm from an earlier call"
    clock.advance(client._absent_abandon_s + 1)
    assert client.busy() is None
    assert client.unverified == ABANDONED
    report = arms.state()
    assert (report.unverified, report.travel_safe) == (ABANDONED, None)
    assert report.sentence() == UNKNOWN_SENTENCE
    assert wait_until(lambda: not client._joint_states.service_is_ready())

    # The arms cannot be read, but the check does not fail open while they are
    # unverified.
    error = error_of(lambda: arms.check_for_driving(DRIVE, 5.0))

    assert type(error) is StepError
    assert error.code == ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        f"Not starting {DRIVE}. An arm or a gripper may still be moving, as {ABANDONED}, and "
        "the arms cannot be checked now. The commander's /get_joint_states service is not "
        "discovered. Is bimanual_commander running? Try again in a few seconds; if it fails "
        "again, tell the person that the arms cannot be checked."
    )


def test_the_drive_check_refuses_while_an_arm_moves(arms, client, fake_commander, poses):
    # The goal plans until the check is done, so the move is still running.
    plan = threading.Event()
    fake_commander.plan_gate = plan
    moving = Call(move, client, "left_arm", table(poses, "standby", "left"))
    assert wait_until(lambda: accepted_handle(client) is not None)

    error = error_of(lambda: arms.check_for_driving(DRIVE, 5.0))

    assert error.code == ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        f"Not starting {DRIVE}. Cannot check the arms for driving while the move of left_arm "
        "is running. Wait for it to finish."
    )
    plan.set()
    assert moving.result() == "Success"


def test_the_drive_check_raises_a_stop_as_it_came(arms, token):
    token.cancel()

    error = error_of(lambda: arms.check_for_driving(DRIVE, 5.0))

    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        f"Cannot check the arms for driving: {STOP_IN_EFFECT}.",
    )
