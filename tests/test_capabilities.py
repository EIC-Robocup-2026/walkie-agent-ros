"""The typed capabilities on ctx: their records, the fakes and the contract between them.

capabilities.md and ADR-0052. The contract cases run once on each fake of
walkie_agent.testing and once on its ROS implementation: ctx.pose on a TF
buffer, ctx.arms on a CommanderClient over FakeCommander, ctx.gaze on a
GazeLoop that turns test_gaze's FakeBase, and ctx.vision, also the gaze
loop's, on a VisionFeed that test_people_atomics' publisher feeds, on Fast
DDS and a random domain (ADR-0019). Each side logs the safety steps that
both can observe, so a fake that drifts from the real code fails here. The
fake-only cases cover the call log, the scripted faults, the FakeClock and
the atomics that run on ctx.
"""

import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import Buffer

from conftest import PROJECT_ROOT
from test_arm_atomics import LEFT_OUT, MODEL_PATH, POSES_PATH, wait_until
from test_gaze import DEADBAND, Recorder, assert_rotation_only, place_base, transform
from test_gaze import FakeBase as TurningBase
from test_gaze_atomics import column, pose_known
from test_people_atomics import FakeVision as VisionPublisher
from test_people_atomics import person, thing
from walkie_agent import capabilities, gaze
from walkie_agent.arm import commander
from walkie_agent.arm.commander import CommanderClient
from walkie_agent.arm.poses import GRIPPER_GROUPS, load_arm_poses, load_robot_model
from walkie_agent.arm.state import ArmsState, ArmState
from walkie_agent.capabilities import (
    ArmsReport,
    GripperMoved,
    GripperResult,
    Looking,
    Moved,
    Pose2D,
    Recovery,
)
from walkie_agent.gaze import INTERRUPT, LOST, NAVIGATION, STOP_LOOKING, GazeLoop, wrap_angle
from walkie_agent.perception.vision_feed import (
    OBJECTS_TOPIC,
    PEOPLE_TOPIC,
    TrackKey,
    VisionFeed,
    stale_message,
)
from walkie_agent.sim.fake_commander import FakeCommander
from walkie_agent.testing import OUT, FakeClock, FakeObject, FakePerson, fake_ctx
from walkie_agent.toolbox import StepError, ToolContext
from walkie_agent.toolbox import handles as handles_module
from walkie_agent.tools.atomic import (
    _arm,
    _perception,
    look_at,
    move_arm_to_pose,
    set_gripper,
    stop_looking,
)
from walkie_agent.tools.atomic._gaze import TURN_TIMEOUT
from walkie_agent.tools.navigation import MAP_FRAME, ROBOT_FRAME, check_arms, tf_publisher_hint

SRC = PROJECT_ROOT / "src"

POSITION = f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME})"
# What tf2 says when its buffer has no map frame at all.
NO_MAP = f'"{MAP_FRAME}" passed to lookupTransform argument target_frame does not exist. '
STOPPED = "A stop or a reset interrupted the call."

HOME = ArmsState(left=ArmState("home", True), right=ArmState("home", True))
OFFERED = ["home", "hands_up", "standby"]
OFFERED_IN_SIM = [*OFFERED, "tray", "receive", "carry"]

# How far apart the image's columns and boxes are for each third, on the
# 640 px image of test_people_atomics.
COLUMNS = {"left": 120.0, "centre": 320.0, "right": 520.0}
BOXES = {
    "left": (60.0, 200.0, 180.0, 400.0),
    "centre": (260.0, 200.0, 380.0, 400.0),
    "right": (460.0, 200.0, 580.0, 400.0),
}


def error_of(fn: Callable[..., object], *args, **kwargs) -> StepError:
    with pytest.raises(StepError) as error:
        fn(*args, **kwargs)
    return error.value


def xyz(pose: Pose2D) -> tuple[float, float, float]:
    return (pose.x, pose.y, pose.yaw)


# -- The module and the records ------------------------------------------------------


def test_the_capabilities_import_without_ros():
    # In a fresh interpreter, since this one has rclpy loaded already: a
    # teammate types against the Protocols without ROS installed.
    code = (
        "import json, sys\n"
        "import walkie_agent.capabilities as module\n"
        "roots = ('rclpy', 'tf2_ros', 'rai', 'walkie_agent.toolbox')\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if any(name == root or name.startswith(root + '.') for root in roots)\n"
        ")\n"
        "print(json.dumps({'file': module.__file__, 'loaded': loaded}))\n"
    )
    path = os.pathsep.join(part for part in (str(SRC), os.environ.get("PYTHONPATH")) if part)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    found = json.loads(result.stdout.strip().splitlines()[-1])

    assert found["loaded"] == []
    assert Path(found["file"]).resolve().is_relative_to(SRC.resolve())


def test_the_moved_records_keep_their_old_import_paths():
    assert handles_module.Pose2D is capabilities.Pose2D
    assert gaze.GazeStatus is capabilities.GazeStatus
    assert commander.GripperResult is capabilities.GripperResult
    assert commander.Recovery is capabilities.Recovery


# -- ctx.pose --------------------------------------------------------------------------


class LiveTf:
    """Writes map -> base_footprint into `buffer` every 20 ms, stamped now, as a localizer would."""

    def __init__(self, buffer: Buffer, clock) -> None:
        self._buffer = buffer
        self._clock = clock
        self._lock = threading.Lock()
        # None while nothing is sent.
        self._pose: Pose2D | None = None
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def put(self, pose: Pose2D) -> int:
        """Send `pose` from now on; the stamp of its first transform."""
        with self._lock:
            self._pose = pose
            return self._write(pose, self._clock.now().nanoseconds)

    def freeze(self) -> None:
        """Send nothing newer: the buffer keeps the last transform."""
        with self._lock:
            self._pose = None

    def write(self, pose: Pose2D, stamp_ns: int) -> None:
        with self._lock:
            self._write(pose, stamp_ns)

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._closed.wait(0.02):
            with self._lock:
                if self._pose is not None:
                    self._write(self._pose, self._clock.now().nanoseconds)

    def _write(self, pose: Pose2D, stamp_ns: int) -> int:
        stamp = Time(nanoseconds=stamp_ns).to_msg()
        self._buffer.set_transform(
            transform(MAP_FRAME, ROBOT_FRAME, (pose.x, pose.y, 0.0), (0.0, 0.0, pose.yaw), stamp),
            "test",
        )
        return stamp_ns


class FakePoseSide:
    """ctx.pose on FakePose, with no TF until put()."""

    def __init__(self) -> None:
        self.ctx, self.robot = fake_ctx()
        self.robot.pose.tf = False

    def put(self, x: float, y: float, yaw: float) -> int:
        """The base at this pose from now on; the stamp it begins at."""
        pose = self.robot.pose
        pose.tf = True
        pose.updating = True
        pose.teleport(x, y, yaw)
        return self.robot.clock.now_ns()

    def freeze(self) -> None:
        self.robot.pose.updating = False

    def stale(self, age_s: float) -> None:
        """One transform, `age_s` old, and none newer."""
        self.robot.pose.tf = True
        self.robot.pose.age_s = age_s
        self.freeze()

    def pass_time(self, seconds: float) -> None:
        self.robot.clock.advance(seconds)

    def close(self) -> None:
        pass


class RosPoseSide:
    """ctx.pose as RosPose, on a buffer of its own that only LiveTf writes."""

    def __init__(self, connector) -> None:
        # A stand-in connector: the node's clock and graph, and a buffer no
        # listener writes, so the test alone decides what TF holds.
        self._buffer = Buffer()
        self.ctx = ToolContext(SimpleNamespace(node=connector.node, _tf_buffer=self._buffer))
        self._clock = connector.node.get_clock()
        self._tf = LiveTf(self._buffer, self._clock)

    def put(self, x: float, y: float, yaw: float) -> int:
        return self._tf.put(Pose2D(x, y, yaw))

    def freeze(self) -> None:
        self._tf.freeze()

    def stale(self, age_s: float) -> None:
        self._tf.freeze()
        now_ns = self._clock.now().nanoseconds
        self._tf.write(Pose2D(0.0, 0.0, 0.0), now_ns - int(age_s * 1e9))

    def pass_time(self, seconds: float) -> None:
        time.sleep(seconds)

    def close(self) -> None:
        self._tf.close()


@pytest.fixture(params=["fake", "ros"])
def pose_side(request) -> Iterator[FakePoseSide | RosPoseSide]:
    if request.param == "fake":
        side: FakePoseSide | RosPoseSide = FakePoseSide()
    else:
        side = RosPoseSide(request.getfixturevalue("connector"))
    yield side
    side.close()


def test_pose_latest_is_the_newest_transform(pose_side):
    pose_side.put(1.0, 2.0, 0.5)

    assert xyz(pose_side.ctx.pose.latest(max_age_s=1.0)) == pytest.approx((1.0, 2.0, 0.5))


def test_pose_latest_older_than_its_limit_is_stale(pose_side):
    pose_side.stale(5.0)

    err = error_of(pose_side.ctx.pose.latest, max_age_s=1.0)

    assert (err.code, err.message) == ("STALE", f"{POSITION} is 5.0 s old, more than 1 s.")
    assert err.data["age_s"] == pytest.approx(5.0, abs=0.5)


def test_pose_without_tf_is_unavailable_with_the_hint(pose_side):
    ctx = pose_side.ctx
    expected = f"{POSITION} is unknown. {NO_MAP}{tf_publisher_hint(ctx.node)}"

    for read in (lambda: ctx.pose.latest(max_age_s=1.0), lambda: ctx.pose.fresh(timeout_s=0.2)):
        err = error_of(read)
        assert (err.code, err.message) == ("UNAVAILABLE", expected)


def test_pose_fresh_takes_a_transform_stamped_after_the_call(pose_side):
    pose_side.put(1.0, 0.0, -0.5)

    assert xyz(pose_side.ctx.pose.fresh(timeout_s=1.0)) == pytest.approx((1.0, 0.0, -0.5))


def test_pose_fresh_is_out_of_date_when_tf_stops_updating(pose_side):
    ctx = pose_side.ctx
    pose_side.put(1.0, 0.0, 0.0)
    pose_side.freeze()

    err = error_of(ctx.pose.fresh, timeout_s=0.3)

    assert (err.code, err.message) == (
        "UNAVAILABLE",
        f"{POSITION} is out of date: no transform stamped after the call came within 0.3 s. "
        "TF is not updating, or the agent and the robot disagree on sim time (see "
        f"--use-sim-time).{tf_publisher_hint(ctx.node)}",
    )


def test_pose_at_a_stamp_is_the_pose_then_after_a_teleport(pose_side):
    ctx = pose_side.ctx
    before = pose_side.put(1.0, 0.0, 0.0)
    pose_side.pass_time(0.1)
    after = pose_side.put(2.0, 0.0, 0.5)

    assert xyz(ctx.pose.at(before, timeout_s=0.5)) == pytest.approx((1.0, 0.0, 0.0))
    assert xyz(ctx.pose.at(after, timeout_s=0.5)) == pytest.approx((2.0, 0.0, 0.5))
    err = error_of(ctx.pose.at, 0, timeout_s=0.5)
    assert (err.code, err.message) == (
        "UNAVAILABLE",
        f"{POSITION} at that stamp is unknown: 0 is not a stamp.",
    )


def test_a_stop_ends_the_wait_for_a_fresh_pose(pose_side):
    ctx = pose_side.ctx
    pose_side.put(1.0, 0.0, 0.0)
    pose_side.freeze()
    ctx.cancel.set()

    err = error_of(ctx.pose.fresh, timeout_s=5.0)

    assert (err.code, err.message) == ("CANCELLED", STOPPED)


# -- ctx.arms --------------------------------------------------------------------------


class FakeArmsSide:
    """ctx.arms on FakeArms, with both arms at home and not read yet."""

    def __init__(self, in_simulation: bool) -> None:
        self.ctx, self.robot = fake_ctx(arms=None, in_simulation=in_simulation)
        self.steps = self.robot.arms.steps
        # What navigation.check_arms takes for the drive check.
        self.drive_check_arm = self.robot.arms

    def fault(self, name: str, value: object = True) -> None:
        setattr(self.robot.arms, name, value)

    def hold(self, arm: str, width: float = 0.02) -> None:
        """Put an object `width` wide, in metres, between `arm`'s fingers."""
        self.robot.arms.holding[arm] = True
        self.robot.arms.object_width_m[arm] = width

    def never_accept(self) -> None:
        """Have the commander neither accept nor reject any goal from now on."""
        self.robot.arms.never_answer = True

    def close(self) -> None:
        pass


class GazeStub:
    """The ROS side's gaze backend: it logs each stop an arm move makes, and runs nothing."""

    def __init__(self, steps: list[str]) -> None:
        self._steps = steps

    def stop(self, reason: str, *, wait_s: float = 0.0) -> str | None:
        self._steps.append("gaze.stop")
        return None


class RosArmsSide:
    """ctx.arms as RosArms, on a CommanderClient that has found FakeCommander.

    A goal step is logged only when FakeCommander received the goal, so the
    log is what the commander saw, not what the client meant to send.
    """

    def __init__(self, connector, node: FakeCommander, in_simulation: bool) -> None:
        self.commander = node
        self.steps: list[str] = []
        self.client = CommanderClient(connector, load_arm_poses(POSES_PATH, MODEL_PATH))
        assert wait_until(self.client.available, 5.0), "The client never discovered the fake."
        self._log_goals("set_joint_position", "arms.goal", lambda: node.joint_goals)
        self._log_goals("control_gripper", "gripper.goal", lambda: node.gripper_goals)
        self.ctx = ToolContext(
            connector,
            {"arm": self.client, "gaze": GazeStub(self.steps)},
            in_simulation=in_simulation,
        )
        self.drive_check_arm = self.client

    def fault(self, name: str, value: object = True) -> None:
        setattr(self.commander, name, value)

    def hold(self, arm: str, width: float = 0.02) -> None:
        self.commander.object_widths[GRIPPER_GROUPS[arm]] = width

    def never_accept(self) -> None:
        self.commander.accept_gate = threading.Event()

    def close(self) -> None:
        gate = self.commander.accept_gate
        if gate is not None:
            # A goal the gate holds is rejected once let through, so nothing
            # moves after the test.
            self.commander.reject_goals = True
            gate.set()
        self.client.destroy()

    def _log_goals(self, method: str, step: str, received: Callable[[], list]) -> None:
        send = getattr(self.client, method)

        def logged(*args, **kwargs):
            count = len(received())
            try:
                return send(*args, **kwargs)
            finally:
                if len(received()) > count:
                    self.steps.append(step)

        # An instance attribute, which RosArms looks up on each call.
        setattr(self.client, method, logged)


@pytest.fixture
def commander_node() -> Iterator[FakeCommander]:
    node = FakeCommander(load_robot_model(MODEL_PATH), move_time_s=0.2, gripper_time_s=0.05)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    executor.shutdown()
    node.destroy_node()
    thread.join(timeout=2)


@pytest.fixture(params=["fake", "ros"])
def make_arms(request) -> Iterator[Callable[..., FakeArmsSide | RosArmsSide]]:
    """A factory of arm sides; `in_simulation` fixes which poses are offered."""
    made: list[FakeArmsSide | RosArmsSide] = []
    if request.param == "ros":
        connector = request.getfixturevalue("connector")
        node = request.getfixturevalue("commander_node")

    def make(in_simulation: bool = False) -> FakeArmsSide | RosArmsSide:
        if request.param == "fake":
            side: FakeArmsSide | RosArmsSide = FakeArmsSide(in_simulation)
        else:
            side = RosArmsSide(connector, node, in_simulation)
        made.append(side)
        return side

    yield make
    for side in made:
        side.close()


def test_arms_offer_the_tried_poses_outside_sim_and_every_pose_in_sim(make_arms):
    assert list(make_arms().ctx.arms.poses()) == OFFERED
    assert list(make_arms(in_simulation=True).ctx.arms.poses()) == OFFERED_IN_SIM


def test_arms_state_reads_the_arms_only_when_the_memory_is_unknown(make_arms):
    side = make_arms()
    arms = side.ctx.arms

    assert arms.state(read=False) == ArmsReport(ArmsState(), None, 0, None)
    assert arms.state() == ArmsReport(HOME, None, 0, None)
    assert side.steps == []


def test_a_move_stops_the_gaze_loop_then_sends_one_goal(make_arms):
    side = make_arms()

    moved = side.ctx.arms.move_to("home")

    assert (moved.pose, moved.arm, moved.status) == ("home", "both", "Success")
    assert moved.max_error_rad == pytest.approx(0.0, abs=1e-6)
    assert moved.report == ArmsReport(HOME, None, 0, None)
    assert side.steps == ["gaze.stop", "arms.goal"]

    moved = side.ctx.arms.move_to("Hands up")

    assert moved.pose == "hands_up"
    assert (moved.report.arms.left.pose, moved.report.arms.right.pose) == ("hands_up", "hands_up")
    assert side.steps == ["gaze.stop", "arms.goal"] * 2


def test_an_unknown_or_refused_pose_sends_nothing(make_arms):
    side = make_arms(in_simulation=True)

    err = error_of(side.ctx.arms.move_to, "dance")
    assert (err.code, err.message) == (
        "NOT_FOUND",
        f"No pose named 'dance'. The poses are: {', '.join(OFFERED_IN_SIM)}. "
        "Arms: safe for driving.",
    )
    err = error_of(side.ctx.arms.move_to, "carry", "both")
    assert (err.code, err.message) == (
        "REFUSED",
        "'carry' is for one arm at a time: left or right. Arms: safe for driving.",
    )
    assert side.steps == []


def test_a_plan_failure_is_a_skill_failure_with_the_arms_read_again(make_arms):
    side = make_arms()
    side.fault("plan_failure")

    err = error_of(side.ctx.arms.move_to, "hands_up")

    assert (err.code, err.message) == (
        "SKILL_FAILED",
        "Could not move to 'hands_up': MoveIt found no path (Planning Failed). The arms did "
        "not move. Arms: safe for driving.",
    )
    assert err.data["status"] == "Planning Failed"
    assert side.steps == ["gaze.stop", "arms.goal"]


def test_plan_only_passes_the_check_only_where_the_arms_already_are(make_arms):
    side = make_arms()
    side.fault("plan_only")

    moved = side.ctx.arms.move_to("home")
    assert (moved.status, moved.report.arms) == ("Plan Only", HOME)

    err = error_of(side.ctx.arms.move_to, "hands_up")
    assert err.code == "POSTCONDITION"
    assert err.message.startswith(
        "The commander runs in plan-only mode, so it moved nothing, and openarm_"
    )
    assert err.message.endswith(
        "from 'hands_up'. Treat this as a failure. No arm can move until the commander "
        "restarts without plan_only, so tell the person instead of trying again. Arms: safe "
        "for driving."
    )
    assert side.steps == ["gaze.stop", "arms.goal"] * 2


def test_a_stop_before_the_goal_stops_the_gaze_loop_and_sends_nothing(make_arms):
    side = make_arms()
    side.ctx.cancel.set()

    err = error_of(side.ctx.arms.move_to, "home")

    assert (err.code, err.message) == (
        "CANCELLED",
        "Stopped by reset before the arms moved. Arms: state unknown.",
    )
    assert side.steps == ["gaze.stop"]


def test_a_gripper_goal_does_not_stop_the_gaze_loop(make_arms):
    side = make_arms()

    moved = side.ctx.arms.gripper("left", "open")

    assert moved == GripperMoved(
        arm="left",
        state="open",
        outcome="moved",
        result=GripperResult(True, "Success", None, None, 1),
    )
    assert side.steps == ["gripper.goal"]


def test_a_close_holds_something_or_closes_empty(make_arms):
    side = make_arms()
    side.hold("right")

    empty = side.ctx.arms.gripper("left", "close")
    held = side.ctx.arms.gripper("right", "close")

    assert (empty.outcome, empty.holding) == ("empty", False)
    assert empty.finger_m == pytest.approx(0.0, abs=1e-6)
    result = empty.result
    assert (result.success, result.status, result.grasped, result.command) == (
        True,
        "Success",
        False,
        1,
    )
    assert (held.outcome, held.holding) == ("holding", True)
    assert held.finger_m == pytest.approx(0.02, abs=1e-6)
    result = held.result
    assert (result.success, result.status, result.grasped, result.command) == (
        False,
        "Failed",
        True,
        1,
    )
    assert result.gripper_gap == pytest.approx(0.02, abs=1e-6)
    assert side.steps == ["gripper.goal"] * 2


def test_a_close_is_judged_by_where_the_fingers_settle(make_arms):
    side = make_arms()
    # Below the grasp band, and near the open position.
    side.hold("left", width=0.003)
    side.hold("right", width=0.042)

    thin = side.ctx.arms.gripper("left", "close")
    wide = side.ctx.arms.gripper("right", "close")

    assert (thin.outcome, thin.holding) == ("empty", False)
    assert thin.finger_m == pytest.approx(0.003, abs=1e-6)
    assert (thin.result.success, thin.result.grasped) == (False, False)
    assert (wide.outcome, wide.holding) == ("not_closed", None)
    assert wide.finger_m == pytest.approx(0.042, abs=1e-6)
    assert (wide.result.success, wide.result.grasped) == (False, True)


def test_a_goal_given_up_on_refuses_every_call_until_its_end_is_known(make_arms, monkeypatch):
    monkeypatch.setattr(_arm, "CONFIG", dataclasses.replace(_arm.CONFIG, gripper_timeout_s=0.5))
    side = make_arms()
    side.never_accept()

    err = error_of(side.ctx.arms.gripper, "left", "close")
    assert (err.code, err.message, err.data) == (
        "TIMEOUT",
        "Could not move the left gripper: The left_gripper command did not finish within "
        "0.5 s; if the commander accepts it later, the gripper still makes that motion. Arms: "
        "state unknown.",
        {"cancel": "pending"},
    )
    outstanding = (
        "the left_gripper command from an earlier call has not ended yet. Try again in a few "
        "seconds."
    )
    err = error_of(side.ctx.arms.gripper, "left", "open")
    assert (err.code, err.message, err.data) == (
        "PRECONDITION_FAILED",
        f"Could not move the left gripper: Cannot move left_gripper: {outstanding} Arms: state "
        "unknown.",
        {"outstanding": "the left_gripper command"},
    )
    err = error_of(check_arms, side.drive_check_arm, "driving", "'home'", side.ctx.cancel)
    assert (err.code, err.message) == (
        "PRECONDITION_FAILED",
        f"Not driving. Cannot check the arms for driving: {outstanding}",
    )
    # Only one goal reached the commander, and the arms are not unverified.
    assert side.steps == ["gripper.goal"]
    assert side.ctx.arms.state() == ArmsReport(ArmsState(), None, 0, None)


def test_a_move_that_cannot_be_checked_after_it_is_a_postcondition(make_arms):
    side = make_arms()
    side.ctx.arms.move_to("home")
    side.fault("read_failure", "MoveIt has no robot state")

    err = error_of(side.ctx.arms.move_to, "hands_up")

    assert (err.code, err.message, err.data) == (
        "POSTCONDITION",
        "The commander reported success, but the arms could not be checked against "
        "'hands_up': The commander could not read the joints of both_arms: MoveIt has no robot "
        "state. Treat this as a failure. Arms: state unknown.",
        {"pose": "hands_up", "status": "Success", "read_error": "UNAVAILABLE"},
    )
    assert side.steps == ["gaze.stop", "arms.goal"] * 2


def test_a_rejected_goal_is_internal(make_arms):
    side = make_arms()
    side.fault("reject_goals")
    home = load_arm_poses(POSES_PATH, MODEL_PATH).poses["home"]
    count = sum(len(table) for table in home.tables.values())

    err = error_of(side.ctx.arms.move_to, "home")
    assert (err.code, err.message) == (
        "INTERNAL",
        f"Could not move to 'home': The commander rejected the goal for both_arms with {count} "
        "positions. It rejects a count that differs from the group's joints; the joint order "
        "is read again on the next call. Arms: safe for driving.",
    )
    err = error_of(side.ctx.arms.gripper, "left", "open")
    assert (err.code, err.message) == (
        "INTERNAL",
        "Could not move the left gripper: The commander rejected the goal for left_gripper. "
        "Arms: safe for driving.",
    )
    assert side.steps == ["gaze.stop", "arms.goal", "gripper.goal"]


# -- ctx.gaze --------------------------------------------------------------------------


class FakeGazeSide:
    """ctx.gaze on FakeGaze, with no arm capability, as the ROS side has no arm backend."""

    def __init__(self, turns: bool) -> None:
        self.ctx, self.robot = fake_ctx(without=("arms",))
        self.robot.gaze.faces = turns

    def yaw(self) -> float:
        return self.robot.pose.yaw

    def show_person(self, heading: float, track: int) -> None:
        """One image with one person at map `heading`, on track id `track`."""
        self.robot.vision.show(people=[FakePerson(heading_map=heading, track=track)])

    def show_nobody(self) -> None:
        self.robot.vision.show(people=[])

    def close(self) -> None:
        pass


class RosGazeSide:
    """ctx.gaze as RosGaze, on a GazeLoop that turns a base, or a base that never turns.

    The loop and ctx.vision share one VisionFeed, which test_people_atomics'
    publisher feeds; the base, with the camera ahead, starts at yaw 0.
    """

    def __init__(self, connector, turns: bool) -> None:
        self.recorder = Recorder()
        feed = VisionFeed(connector)
        self.loop = GazeLoop(connector, vision=feed)
        self.ctx = ToolContext(connector, {"gaze": self.loop, "vision": feed})
        self._publisher = VisionPublisher(connector, feed)
        self.base = None
        try:
            assert wait_until(
                lambda: self.loop._publisher._publisher.get_subscription_count() > 0
                and self.recorder.nav_status.get_subscription_count() > 0
            ), "The recorder and the loop did not discover each other."
            self._publisher.wait_for_subscribers()
            if turns:
                self.base = TurningBase(connector, self.recorder, 0.0)
            else:
                place_base(connector, 0.0)
            assert wait_until(lambda: pose_known(self.ctx))
        except BaseException:
            self.close()
            raise

    def yaw(self) -> float:
        return self.ctx.robot_pose().yaw

    def show_person(self, heading: float, track: int) -> None:
        self._publisher.show(people=[person(column(heading, 0.0), track_id=track)])

    def show_nobody(self) -> None:
        self._publisher.show(people=[])

    def close(self) -> None:
        self.loop.shutdown()
        self._publisher.close()
        if self.base is not None:
            self.base.close()
        try:
            assert_rotation_only(self.recorder.messages())
        finally:
            self.recorder.close()


@pytest.fixture(params=["fake", "ros"])
def make_gaze(request) -> Iterator[Callable[..., FakeGazeSide | RosGazeSide]]:
    """A factory of gaze sides; `turns` False is a base that never turns."""
    made: list[FakeGazeSide | RosGazeSide] = []
    if request.param == "ros":
        connector = request.getfixturevalue("connector")

    def make(turns: bool = True) -> FakeGazeSide | RosGazeSide:
        if request.param == "fake":
            side: FakeGazeSide | RosGazeSide = FakeGazeSide(turns)
        else:
            side = RosGazeSide(connector, turns)
        made.append(side)
        return side

    yield make
    for side in made:
        side.close()


def put_seat(ctx: ToolContext, heading_map: float | None) -> str:
    """One chair handle, as find_seats stores it, seen at map `heading_map`; its id."""
    header = Header(stamp=Time(nanoseconds=ctx.now_ns()).to_msg(), frame_id="camera")
    payload = _perception.SeatPayload(
        position="centre",
        box=BOXES["centre"],
        label="chair",
        part=None,
        occupied=False,
        header=header,
        image_width=640,
        image_height=480,
        heading_map=heading_map,
    )
    return ctx.handles.put_set("seat", [payload], pose=ctx.robot_pose()).member_ids[0]


def put_person(ctx: ToolContext, track: TrackKey | None, age_s: float = 0.0) -> str:
    """One person handle seen `age_s` ago, straight ahead; its id.

    `track` None is a person without a track, as find_people stores track id 0.
    """
    stamp_ns = ctx.now_ns() - int(age_s * 1e9)
    payload = _perception.PersonPayload(
        track=track,
        position="centre",
        bbox=BOXES["centre"],
        face_bbox=None,
        keypoints=(),
        keypoint_scores=(),
        gestures=frozenset(),
        face=None,
        appearance=None,
        header=Header(stamp=Time(nanoseconds=stamp_ns).to_msg(), frame_id="camera"),
        stamp_ns=stamp_ns,
        image_width=640,
        image_height=480,
        heading_map=0.0,
    )
    return ctx.handles.put_set("person", [payload], pose=ctx.robot_pose()).member_ids[0]


def test_look_at_an_unknown_handle_is_not_found_with_the_same_hint(make_gaze):
    ctx = make_gaze(turns=False).ctx

    err = error_of(ctx.gaze.look_at, "per_0000.0")

    assert (err.code, err.message) == (
        "NOT_FOUND",
        "per_0000.0 is not a known handle. look_at takes one person handle from find_people, "
        "such as per_51c0.1, or one seat handle from find_seats.",
    )


def test_look_at_an_untracked_person_is_refused(make_gaze):
    ctx = make_gaze(turns=False).ctx
    handle = put_person(ctx, None)

    err = error_of(ctx.gaze.look_at, handle)

    assert (err.code, err.message) == (
        "REFUSED",
        f"{handle} is not tracked, cannot follow. Call find_people again for a handle the "
        "robot can follow.",
    )
    assert ctx.gaze.status().target is None


def test_look_at_a_person_in_view_starts_from_the_newest_sighting(make_gaze):
    side = make_gaze(turns=False)
    ctx = side.ctx
    side.show_person(0.4, track=7)
    [someone] = ctx.vision.people(max_age_s=1.0)

    looking = ctx.gaze.look_at(someone.handle)

    assert looking == Looking(
        target=someone.handle,
        kind="person",
        notes=(),
        lost_after_s=gaze.DEFAULT_CONFIG.lost_after_s,
        seat_label=None,
        seat_part=None,
    )
    status = ctx.gaze.status()
    assert (status.target, status.kind) == (someone.handle, gaze.PERSON)
    assert status.setpoint == pytest.approx(0.4, abs=1e-3)
    assert ctx.gaze.stop() == someone.handle


def test_look_at_an_old_handle_of_a_person_out_of_view_is_stale(make_gaze):
    side = make_gaze(turns=False)
    ctx = side.ctx
    handle = put_person(ctx, TrackKey(1, 1), age_s=5.0)
    side.show_nobody()

    err = error_of(ctx.gaze.look_at, handle)

    assert err.code == "STALE"
    assert re.fullmatch(
        rf"Not turning toward {re.escape(handle)}: the handle saw them \d+\.\d s ago, and the "
        r"newest people array does not show them\. The robot stops following a person 1 s "
        r"after the last sighting\. Call find_people for a current handle\.",
        err.message,
    ), err.message
    assert err.data == {"handle": handle}
    assert ctx.gaze.status().target is None


def test_look_at_a_seat_without_a_heading_is_unavailable(make_gaze):
    ctx = make_gaze(turns=False).ctx
    seat = put_seat(ctx, None)

    err = error_of(ctx.gaze.look_at, seat)

    assert (err.code, err.message) == (
        "UNAVAILABLE",
        f"No heading toward {seat}: TF had no camera position when it was seen. Call find_seats "
        "again.",
    )


def test_look_at_a_seat_then_face_it(make_gaze):
    side = make_gaze()
    ctx = side.ctx
    seat = put_seat(ctx, 0.5)

    looking = ctx.gaze.look_at(seat)

    assert looking == Looking(
        target=seat, kind="seat", notes=(), lost_after_s=None, seat_label="chair", seat_part=None
    )
    assert ctx.gaze.face(looking, 10.0) is True
    status = ctx.gaze.status()
    assert (status.target, status.kind, status.running, status.setpoint) == (
        seat,
        gaze.HEADING,
        True,
        0.5,
    )
    assert abs(wrap_angle(side.yaw() - 0.5)) <= DEADBAND + 0.02


def test_stop_returns_the_target_then_none(make_gaze):
    ctx = make_gaze(turns=False).ctx
    seat = put_seat(ctx, 0.5)
    ctx.gaze.look_at(seat)

    assert ctx.gaze.stop() == seat
    assert ctx.gaze.stop() is None
    status = ctx.gaze.status()
    assert (status.target, status.running, status.stop_reason) == (seat, False, STOP_LOOKING)
    assert status.stopped_at is not None


def test_a_base_that_never_turns_does_not_face_the_seat(make_gaze):
    ctx = make_gaze(turns=False).ctx
    looking = ctx.gaze.look_at(put_seat(ctx, 1.0))

    assert ctx.gaze.face(looking, 0.5) is False
    # The loop goes on: the caller decides what a late turn means.
    assert ctx.gaze.status().running


def test_a_stop_cancels_look_at_and_face(make_gaze):
    ctx = make_gaze(turns=False).ctx
    seat = put_seat(ctx, 1.0)
    ctx.cancel.set()

    err = error_of(ctx.gaze.look_at, seat)

    assert (err.code, err.message) == ("CANCELLED", STOPPED)
    assert ctx.gaze.status().target is None

    ctx.cancel.clear()
    looking = ctx.gaze.look_at(seat)
    ctx.cancel.set()
    err = error_of(ctx.gaze.face, looking, 5.0)
    assert (err.code, err.message) == ("CANCELLED", STOPPED)
    status = ctx.gaze.status()
    assert (status.running, status.stop_reason) == (False, INTERRUPT)


def test_stop_takes_look_ats_turn_timeout_reason(make_gaze):
    ctx = make_gaze(turns=False).ctx
    seat = put_seat(ctx, 1.0)
    ctx.gaze.look_at(seat)

    assert ctx.gaze.stop(TURN_TIMEOUT) == seat
    assert ctx.gaze.status().stop_reason == TURN_TIMEOUT


# -- ctx.vision ------------------------------------------------------------------------


class FakeVisionSide:
    """ctx.vision on FakeVision."""

    def __init__(self) -> None:
        self.ctx, self.robot = fake_ctx()

    def show_people(self, people: list[tuple[str, int]]) -> None:
        """One image with a person at each (third, track id)."""
        self.robot.vision.show(
            people=[FakePerson(position=where, track=track) for where, track in people]
        )

    def show_objects(self, objects: list[tuple[str, str]]) -> None:
        """One image with a detection for each (label, third)."""
        self.robot.vision.show(
            objects=[FakeObject(label, score=0.8, position=where) for label, where in objects]
        )

    def close(self) -> None:
        pass


class RosVisionSide:
    """ctx.vision as RosVision, on a VisionFeed that a second node feeds."""

    def __init__(self, connector) -> None:
        feed = VisionFeed(connector)
        self.ctx = ToolContext(connector, {"vision": feed})
        self._publisher = VisionPublisher(connector, feed)
        try:
            self._publisher.wait_for_subscribers()
            self._publisher.place_base(self.ctx)
        except BaseException:
            self.close()
            raise

    def show_people(self, people: list[tuple[str, int]]) -> None:
        self._publisher.show(
            people=[
                person(COLUMNS[where], f"person_{track}", track_id=track) for where, track in people
            ]
        )

    def show_objects(self, objects: list[tuple[str, str]]) -> None:
        self._publisher.show(objects=[thing(label, *BOXES[where]) for label, where in objects])

    def close(self) -> None:
        self._publisher.close()


@pytest.fixture(params=["fake", "ros"])
def vision_side(request) -> Iterator[FakeVisionSide | RosVisionSide]:
    if request.param == "fake":
        side: FakeVisionSide | RosVisionSide = FakeVisionSide()
    else:
        side = RosVisionSide(request.getfixturevalue("connector"))
    yield side
    side.close()


def stored_kinds(ctx: ToolContext) -> list[str]:
    """The kind of every handle set stored from now on, in order."""
    kinds: list[str] = []
    put_set = ctx.handles.put_set

    def spy(kind, payloads, pose=None):
        kinds.append(kind)
        return put_set(kind, payloads, pose=pose)

    ctx.handles.put_set = spy
    return kinds


def test_vision_gives_the_people_left_to_right_as_one_handle_set(vision_side):
    ctx = vision_side.ctx
    vision_side.show_people([("right", 2), ("left", 1)])

    seen = ctx.vision.people(max_age_s=1.0)

    assert [p.position for p in seen] == ["left", "right"]
    assert all(p.face_seen and p.heading_map is not None and p.stamp_ns > 0 for p in seen)
    handles = [ctx.handles.get(p.handle) for p in seen]
    assert [h.kind for h in handles] == ["person", "person"]
    assert [h.id.rpartition(".")[2] for h in handles] == ["0", "1"]
    assert len({h.id.rpartition(".")[0] for h in handles}) == 1
    assert [h.payload.track.track_id for h in handles] == [1, 2]


def test_vision_with_nobody_in_view_stores_nothing(vision_side):
    ctx = vision_side.ctx
    vision_side.show_people([])
    stored = stored_kinds(ctx)

    assert ctx.vision.people(max_age_s=1.0) == []
    assert stored == []


def test_vision_objects_keep_only_the_labels_asked_for(vision_side):
    ctx = vision_side.ctx
    vision_side.show_objects([("cup", "left"), ("chair", "right")])

    [chair] = ctx.vision.objects(["chair"], max_age_s=1.0)

    assert (chair.label, chair.position) == ("chair", "right")
    assert chair.score == pytest.approx(0.8)
    assert ctx.handles.get(chair.handle).kind == "object"
    assert [o.label for o in ctx.vision.objects(max_age_s=1.0)] == ["cup", "chair"]
    assert ctx.vision.objects(["sofa"], max_age_s=1.0) == []


def test_vision_without_an_array_is_unavailable(vision_side):
    err = error_of(vision_side.ctx.vision.objects, max_age_s=1.0)

    assert (err.code, err.message) == (
        "UNAVAILABLE",
        stale_message(OBJECTS_TOPIC, None, 1.0, _perception.CONFIG.wait_s),
    )


def test_a_stop_cancels_vision_and_stores_no_handle(vision_side):
    ctx = vision_side.ctx
    vision_side.show_people([("centre", 1)])
    stored = stored_kinds(ctx)
    ctx.cancel.set()

    err = error_of(ctx.vision.people, max_age_s=1.0)

    assert (err.code, err.message) == ("CANCELLED", STOPPED)
    assert stored == []


# -- The fakes on their own -------------------------------------------------------------


KITCHEN = {"kitchen": (2.0, 1.0, 0.0)}


def test_the_call_log_names_each_call_and_the_step_log_the_safety_order():
    ctx, robot = fake_ctx(locations=KITCHEN)

    moved = ctx.base.go_to("Kitchen")
    ctx.base.move_relative(forward_m=1.0)

    assert moved == Moved(goal=Pose2D(2.0, 1.0, 0.0), pose=Pose2D(2.0, 1.0, 0.0), notes=())
    assert robot.calls == ["base.go_to", "base.move_relative"]
    assert robot.base.steps == [
        "gaze.stop",
        "base.cancel_others",
        "arms.check",
        "base.goal",
        "gaze.stop",
        "base.cancel_others",
        "arms.check",
        "pose.fresh",
        "base.goal",
    ]
    assert robot.base.goals == [Pose2D(2.0, 1.0, 0.0), Pose2D(3.0, 1.0, 0.0)]
    assert robot.gaze.stops == [NAVIGATION, NAVIGATION]
    assert xyz(robot.pose.current()) == (3.0, 1.0, 0.0)


def test_fake_ctx_refuses_an_unknown_capability_or_pose():
    with pytest.raises(ValueError, match="Unknown capabilities"):
        fake_ctx(without=("legs",))
    with pytest.raises(ValueError, match="'wave' is not a pose of the left arm"):
        fake_ctx(arms="wave")


@pytest.mark.parametrize(
    ("fault", "code", "message", "data"),
    [
        ("reject", "REFUSED", "Nav2 rejected the drive to kitchen.", {"status": "REJECTED"}),
        (
            "abort",
            "SKILL_FAILED",
            "Nav2 aborted the drive to kitchen.",
            {"status": "ABORTED", "error_code": 0, "error_msg": ""},
        ),
        (
            "slow",
            "TIMEOUT",
            "The drive to kitchen did not finish within 120 s; Nav2 accepted the cancel.",
            {"cancel": "accepted"},
        ),
        (
            "cancel_refused",
            "CANCELLED",
            "A stop or a reset interrupted the drive to kitchen; Nav2 refused the cancel, so the "
            "base may still be moving.",
            {"cancel": "refused"},
        ),
        (
            "cancel_unanswered",
            "CANCELLED",
            "A stop or a reset interrupted the drive to kitchen; Nav2 did not answer the cancel, "
            "so the base may still be moving.",
            {"cancel": "no answer"},
        ),
        ("stop", "CANCELLED", "Nav2 cancelled the drive to kitchen.", {"status": "CANCELED"}),
    ],
)
def test_each_base_fault_gives_its_error(fault, code, message, data):
    ctx, robot = fake_ctx(locations=KITCHEN)
    base = robot.base
    if fault == "slow":
        base.drive_s = 200.0
    elif fault == "stop":
        base.on_goal(ctx.base.stop)
    elif fault in ("cancel_refused", "cancel_unanswered"):
        setattr(base, fault, True)
        base.on_goal(ctx.cancel.set)
    else:
        setattr(base, fault, True)
    started = time.monotonic()

    err = error_of(ctx.base.go_to, "kitchen")

    assert (err.code, err.message, err.data) == (code, message, data)
    assert base.goals == [Pose2D(2.0, 1.0, 0.0)]
    # The FakeClock: a 120 s timeout costs no real time.
    assert time.monotonic() - started < 1.0
    if fault == "slow":
        assert robot.clock.monotonic() == pytest.approx(1120.0)


def test_an_abort_gives_nav2s_error_code_and_message():
    ctx, robot = fake_ctx(locations=KITCHEN)
    robot.base.abort = True
    robot.base.abort_error_code = 203
    robot.base.abort_error_msg = "No valid path found."

    err = error_of(ctx.base.go_to, "kitchen")

    assert (err.code, err.message, err.data) == (
        "SKILL_FAILED",
        "Nav2 aborted the drive to kitchen. No valid path found.",
        {"status": "ABORTED", "error_code": 203, "error_msg": "No valid path found."},
    )


def test_a_second_base_goal_is_refused_while_the_first_runs():
    ctx, robot = fake_ctx(locations={**KITCHEN, "door": (0.0, 2.0, 0.0)})
    base = robot.base
    refusals: list[StepError] = []
    base.on_goal(lambda: refusals.append(error_of(ctx.base.go_to, "door")))

    ctx.base.go_to("kitchen")

    [err] = refusals
    assert (err.code, err.message, err.data) == (
        "PRECONDITION_FAILED",
        "Cannot start the drive to door while the drive to kitchen is running. Wait for it to "
        "finish.",
        {},
    )
    # The refusal comes after the gaze stop, and cancels nothing.
    assert base.steps == ["gaze.stop", "base.cancel_others", "arms.check", "base.goal", "gaze.stop"]
    assert base.goals == [Pose2D(2.0, 1.0, 0.0)]
    assert base.cancels == 0
    assert not base.busy()


def test_the_base_cancels_other_goals_before_its_own():
    ctx, robot = fake_ctx(locations=KITCHEN)
    robot.base.other_goals = 2

    ctx.base.go_to("kitchen")

    assert (robot.base.others_cancelled, robot.base.other_goals) == (2, 0)
    assert robot.base.steps.index("base.cancel_others") < robot.base.steps.index("base.goal")


def test_the_base_refuses_a_bad_move_before_any_step():
    ctx, robot = fake_ctx(locations=KITCHEN)

    cases = [
        (lambda: ctx.base.go_to("nowhere"), "NOT_FOUND", "Unknown location 'nowhere'. Known "
         "locations: kitchen."),
        (lambda: ctx.base.move_relative(forward_m=4.0), "REFUSED", "Refused: a move of 4.00 m "
         "is more than the 3 m allowed in one move."),
        (lambda: ctx.base.move_relative(), "REFUSED", "Nothing to do: the move is zero."),
        (lambda: ctx.base.turn_by(math.nan), "REFUSED", "Refused: every value must be a finite "
         "number."),
        (lambda: ctx.base.turn_by(0.0), "REFUSED", "Nothing to do: the turn is zero."),
    ]
    for call, code, message in cases:
        err = error_of(call)
        assert (err.code, err.message) == (code, message)
    assert robot.base.steps == [] and robot.base.goals == []


def test_an_arm_out_or_unread_arms_decide_the_drive():
    ctx, robot = fake_ctx(locations=KITCHEN, arms={"left": OUT})
    err = error_of(ctx.base.go_to, "kitchen")
    assert (err.code, err.message) == ("PRECONDITION_FAILED", LEFT_OUT)
    assert robot.base.goals == []

    ctx, robot = fake_ctx(locations=KITCHEN)
    robot.arms.unread = commander.NO_ANSWER
    moved = ctx.base.go_to("kitchen")
    assert moved.notes == ("Arm state unknown: the commander did not answer.",)


def test_unverified_arms_refuse_the_drive_until_a_read_recovers():
    ctx, robot = fake_ctx(locations=KITCHEN)
    robot.arms.never_answer = True

    err = error_of(ctx.arms.move_to, "hands_up")
    assert (err.code, err.message) == (
        "TIMEOUT",
        "Could not move to 'hands_up': The move of both_arms did not finish within 30 s; if "
        "the commander accepts it later, it is cancelled at once. Arms: state unknown.",
    )
    # Outstanding first, as the client keeps the goal until its end is known.
    err = error_of(ctx.base.go_to, "kitchen")
    assert (err.code, err.message, err.data) == (
        "PRECONDITION_FAILED",
        "Not driving. Cannot check the arms for driving: the move of both_arms from an earlier "
        "call has not ended yet. Try again in a few seconds.",
        {"outstanding": "the move of both_arms"},
    )
    assert robot.arms.unverified is None

    # Abandoned once ABANDON_AFTER_S passes, and the arms are then unverified.
    robot.clock.advance(commander.ABANDON_AFTER_S)
    err = error_of(ctx.base.go_to, "kitchen")
    assert err.code == "PRECONDITION_FAILED"
    assert err.message.startswith(
        "Not driving. An arm or a gripper may still be moving, as the move of both_arms from an "
        "earlier call was given up on before its end was known"
    )
    assert robot.base.goals == []

    robot.arms.never_answer = False
    ctx.base.go_to("kitchen")
    assert robot.arms.recovery == Recovery(
        count=1, goals=(("the move of both_arms", "result"),), grippers=()
    )
    assert ctx.arms.state(read=False) == ArmsReport(
        HOME, None, 1, robot.arms.recovery
    )


def test_each_arm_fault_gives_the_atomics_answer():
    ctx, robot = fake_ctx()
    robot.arms.execution_failure = True
    err = error_of(ctx.arms.move_to, "hands_up")
    assert (err.code, err.message) == (
        "SKILL_FAILED",
        "Could not move to 'hands_up': The motion stopped before the target (Execution "
        "Failed). The arms may have stopped part way. Arms: not safe for driving.",
    )

    ctx, robot = fake_ctx()
    robot.arms.read_failure = "MoveIt has no robot state"
    # The left arm's joint order is not known yet, so the move reads it first.
    err = error_of(ctx.arms.move_to, "home", "left")
    assert (err.code, err.message) == (
        "UNAVAILABLE",
        "Could not move to 'home': The commander could not read the joints of left_arm: "
        "MoveIt has no robot state. Arms: safe for driving.",
    )
    assert robot.arms.goals == []
    closed = ctx.arms.gripper("left", "close")
    assert (closed.outcome, closed.holding) == ("unread", None)
    assert closed.read_error == (
        "The commander could not read the joints of left_gripper: MoveIt has no robot state"
    )

    ctx, robot = fake_ctx()
    robot.arms.unmeasured = True
    closed = ctx.arms.gripper("left", "close")
    assert (closed.outcome, closed.result) == (
        "unmeasured",
        GripperResult(False, "Failed", False, 0.0, 1),
    )
    robot.arms.not_reached = True
    opened = ctx.arms.gripper("left", "open")
    assert (opened.outcome, opened.result) == (
        "not_reached",
        GripperResult(False, "Failed", None, None, 2),
    )


def test_each_speech_vision_and_gaze_fault_gives_its_error():
    ctx, robot = fake_ctx()
    assert error_of(ctx.speech.say, "  ").code == "REFUSED"
    robot.speech.available = False
    err = error_of(ctx.speech.say, "Hello.")
    assert (err.code, err.message) == (
        "UNAVAILABLE",
        "Nothing was said: no speech node is listening on /to_human.",
    )

    robot.vision.show(people=[FakePerson()])
    robot.vision.age_s = 2.0
    err = error_of(ctx.vision.people, max_age_s=1.0)
    assert (err.code, err.message) == (
        "UNAVAILABLE",
        stale_message(PEOPLE_TOPIC, 2.0, 1.0, _perception.CONFIG.wait_s),
    )
    robot.vision.age_s = 0.0

    seat = robot.vision.seat(0.5)
    robot.gaze.navigation_active = True
    err = error_of(ctx.gaze.look_at, seat)
    assert (err.code, err.message) == (
        "REFUSED",
        f"Not turning toward {seat}: a navigation goal is active.",
    )
    robot.gaze.navigation_active = False
    robot.gaze.closed = True
    assert error_of(ctx.gaze.look_at, seat).code == "UNAVAILABLE"
    robot.gaze.closed = False

    [someone] = ctx.vision.people(max_age_s=1.0)
    robot.gaze.lost = True
    err = error_of(ctx.gaze.look_at, someone.handle)
    assert (err.code, err.message) == (
        "STALE",
        f"Not turning toward {someone.handle}: they were last seen 2.0 s ago. The robot stops "
        "following a person 1 s after the last sighting. Call find_people for a current handle.",
    )
    assert ctx.gaze.status().stop_reason == LOST


@pytest.mark.parametrize("kind", ["person", "seat"])
def test_a_stop_during_the_arm_check_of_look_at_cancels_it(kind, monkeypatch):
    ctx, robot = fake_ctx()
    if kind == "person":
        robot.vision.show(people=[FakePerson()])
        [someone] = ctx.vision.people(max_age_s=1.0)
        target = someone.handle
    else:
        target = robot.vision.seat(0.5)
    check_for_driving = robot.arms.check_for_driving

    def stopped_first(*args, **kwargs):
        # stop_looking while the arm check reads the commander.
        ctx.gaze.stop()
        return check_for_driving(*args, **kwargs)

    monkeypatch.setattr(robot.arms, "check_for_driving", stopped_first)

    err = error_of(ctx.gaze.look_at, target)

    assert (err.code, err.message, err.data) == (
        "CANCELLED",
        f"Not turning toward {target}: the gaze loop was stopped ({STOP_LOOKING}) during the "
        f"call. Call look_at again if the robot should still look at {target}.",
        {"handle": target, "reason": STOP_LOOKING},
    )
    assert robot.gaze.steps == ["arms.check"]
    assert ctx.gaze.status().target is None


# The example atomic of capabilities.md ("An atomic on ctx"), as its run() body.
def guide_to(ctx: ToolContext, location: str, line: str) -> str:
    moved = ctx.base.go_to(location)
    ctx.speech.say(line)
    notes = "".join(f" {note}" for note in moved.notes)
    return f"At {location}. Said: {line}{notes}"


def test_the_guide_to_example_drives_then_speaks():
    ctx, robot = fake_ctx(locations=KITCHEN)

    assert guide_to(ctx, "kitchen", "Here it is.") == "At kitchen. Said: Here it is."
    assert robot.calls == ["base.go_to", "speech.say"]


def test_the_guide_to_example_refuses_with_an_arm_out_and_stays_quiet():
    ctx, robot = fake_ctx(locations=KITCHEN, arms="hands_up")

    err = error_of(guide_to, ctx, "kitchen", "Here it is.")

    assert err.code == "PRECONDITION_FAILED"
    assert robot.base.goals == [] and robot.speech.said == []


def test_the_guide_to_example_cancels_the_goal_on_a_reset_mid_drive():
    ctx, robot = fake_ctx(locations=KITCHEN)
    robot.base.on_goal(ctx.cancel.set)

    err = error_of(guide_to, ctx, "kitchen", "Here it is.")

    assert err.code == "CANCELLED" and robot.base.cancels == 1
    assert robot.speech.said == []


def test_move_arm_to_pose_runs_on_fake_ctx():
    ctx, robot = fake_ctx(arms="hands_up")

    out = move_arm_to_pose.run(ctx, move_arm_to_pose.Args(pose="home"))

    assert out.summary == "Moved to 'home'. Arms: safe for driving."
    assert (out.status, out.travel_safe) == ("Success", True)
    assert robot.calls == ["arms.move_to"]
    assert robot.arms.steps == ["gaze.stop", "arms.goal"]


def test_set_gripper_close_runs_on_fake_ctx():
    ctx, robot = fake_ctx()
    robot.arms.holding["left"] = True

    out = set_gripper.run(ctx, set_gripper.Args(arm="left", state="close"))

    assert out.summary == (
        "Closed the left gripper. It is holding something. Arms: safe for driving."
    )
    assert out.holding == {"left": True}
    assert robot.calls == ["arms.gripper", "arms.state"]
    assert robot.gaze.stops == []


def test_look_at_a_seat_runs_on_fake_ctx():
    ctx, robot = fake_ctx()
    seat = robot.vision.seat(0.5, label="sofa", part="left")

    out = look_at.run(ctx, look_at.Args(target=seat))

    assert out.summary == f"Turned toward {seat} (sofa, left)."
    assert (out.kind, out.heading, out.facing, out.arms_unknown) == ("seat", 0.5, True, False)
    assert robot.calls == ["gaze.look_at", "gaze.face", "gaze.status"]
    assert robot.gaze.steps == ["arms.check", "gaze.start"]
    assert robot.pose.yaw == 0.5

    out = stop_looking.run(ctx, stop_looking.Args())
    assert out.summary == f"Stopped looking at {seat}."


def test_look_at_a_seat_the_base_never_faces_times_out_on_the_fake_clock():
    ctx, robot = fake_ctx()
    robot.gaze.faces = False
    seat = robot.vision.seat(0.5)
    started = time.monotonic()

    err = error_of(look_at.run, ctx, look_at.Args(target=seat))

    assert (err.code, err.message) == (
        "TIMEOUT",
        f"The robot did not face {seat} within 10 s, so it stopped turning.",
    )
    assert ctx.gaze.status().stop_reason == TURN_TIMEOUT
    assert robot.clock.monotonic() == pytest.approx(1010.0)
    assert time.monotonic() - started < 1.0


def test_the_fake_clock_moves_only_on_a_timeout_or_advance():
    clock = FakeClock(start_s=5.0)
    event = threading.Event()

    assert clock.wait(event, 2.0) is False
    assert clock.monotonic() == 7.0
    event.set()
    assert clock.wait(event, 2.0) is True
    clock.advance(0.5)
    assert (clock.monotonic(), clock.now_ns()) == (7.5, 7_500_000_000)
    with pytest.raises(ValueError):
        clock.advance(-1.0)
