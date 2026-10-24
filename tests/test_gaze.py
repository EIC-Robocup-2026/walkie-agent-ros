"""The gaze loop (ADR-0051, docs/architecture/hri.md "Gaze").

The control law is tested on its own, with fake clocks and the yaw passed
in. The GazeLoop tests run on the `connector` fixture's node, on Fast DDS and
a random domain (ADR-0019), so the /cmd_vel they publish on reaches no
robot: a second node records it, and stands in for Nav2's goal status. TF is
written straight into the connector's buffer, either fixed or by a fake base
that turns at the last recorded command. People arrays are handed to a real
VisionFeed's callback, as its executor would. A spy on the loop's publish
records when each command went out, and can block it as a stuck transport
would.
"""

import logging
import math
import random
import re
import threading
import time
from types import SimpleNamespace

import pytest
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped, TwistStamped
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_action_status_default
from sensor_msgs.msg import CameraInfo
from tf_transformations import quaternion_from_euler
from vision_msgs.msg import BoundingBox2D
from walkie_vision_msgs.msg import PersonDetection, PersonDetectionArray

from walkie_agent import gaze as gaze_module
from walkie_agent.arm.state import ArmState, ArmsState
from walkie_agent.gaze import (
    ARM_MOTION,
    ARM_OUT,
    DEFAULT_CONFIG,
    ERROR,
    HEADING,
    INTERRUPT,
    LOST,
    NAV_STATUS_TOPIC,
    NAVIGATION_ACTIVE,
    NEW_TASK,
    PERSON,
    SHUTDOWN,
    STOP_HEARD,
    STOP_LOOKING,
    GazeConfig,
    GazeControl,
    GazeLoop,
    RotationPublisher,
    limit_change,
    proportional_command,
    wrap_angle,
)
from walkie_agent.perception import vision_feed as vision_feed_module
from walkie_agent.perception.vision_feed import TrackKey, VisionFeed
from walkie_agent.toolbox import StepError, ToolContext, Toolbox

NO_CANCEL = threading.Event()
# ADR-0051's hard ceilings, which no config can raise.
MAX_SPEED, MAX_ACCEL = 0.6, 3.2
# The time from one tick to the next, at 15 Hz.
PERIOD = 1 / 15
# The largest change of command per tick: velocity_smoother's 3.2 rad/s2 at 15 Hz.
STEP = 3.2 / 15
DEADBAND = math.radians(6.0)

CAMERA_FRAME = "head_camera_optical_frame"
WIDTH, HEIGHT = 640, 480
# The focal length and the principal point's column, so the ray through
# column u is atan((CX - u) / FX) to the left of the camera's axis.
FX, CX = 320.0, 320.0
# The camera's optical frame in the base's, as roll, pitch and yaw, looking ahead.
CAMERA_AHEAD = (-math.pi / 2, 0.0, -math.pi / 2)
# Where a person's body box is centred, in pixels right of their face: far
# enough that a loop turning toward the body instead would end outside the
# deadband.
BODY_OFFSET_PX = 60.0


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def paced_control(clock: FakeClock | None = None) -> tuple[GazeControl, FakeClock]:
    """A GazeControl and the fake monotonic clock that times its acceleration cap."""
    monotonic = FakeClock(100.0)
    return GazeControl(DEFAULT_CONFIG, clock or FakeClock(), monotonic), monotonic


def ticks(control: GazeControl, monotonic: FakeClock, yaw, count: int, dt=PERIOD) -> list[float]:
    """The commands of `count` ticks at `yaw`, `dt` apart on the monotonic clock."""
    commands = []
    for _ in range(count):
        monotonic.now += dt
        commands.append(control.tick(yaw)[0])
    return commands


# -- The control law ---------------------------------------------------------------


def test_the_command_is_proportional_inside_the_clamp():
    assert proportional_command(0.3) == pytest.approx(0.36)
    assert proportional_command(-0.3) == pytest.approx(-0.36)


def test_the_command_has_a_floor_outside_the_deadband():
    # 0.11 rad is 6.3 degrees: outside the deadband, and 1.2 * 0.11 = 0.132
    # would not turn the base.
    assert proportional_command(0.11) == pytest.approx(0.15)
    assert proportional_command(-0.11) == pytest.approx(-0.15)
    assert proportional_command(DEADBAND + 1e-6) == pytest.approx(0.15)


def test_the_command_is_clamped_at_max_speed():
    assert proportional_command(1.0) == pytest.approx(0.6)
    assert proportional_command(-2.5) == pytest.approx(-0.6)


def test_no_command_inside_the_deadband():
    assert proportional_command(0.0) == 0.0
    assert proportional_command(0.1) == 0.0
    assert proportional_command(-0.1) == 0.0
    assert proportional_command(DEADBAND) == 0.0


def test_the_acceleration_cap_ramps_from_rest():
    control, monotonic = paced_control()
    control.start(1.0)

    commands = ticks(control, monotonic, 0.0, 4)

    assert commands == pytest.approx([STEP, 2 * STEP, 0.6, 0.6])


def test_the_acceleration_cap_holds_on_a_reversal():
    control, monotonic = paced_control()
    control.start(1.0)
    ticks(control, monotonic, 0.0, 3)

    # The base is now past the setpoint by a radian: the full command is -0.6.
    commands = ticks(control, monotonic, 2.0, 7)

    assert commands == pytest.approx(
        [0.6 - STEP, 0.6 - 2 * STEP, 0.6 - 3 * STEP, 0.6 - 4 * STEP, 0.6 - 5 * STEP, -0.6, -0.6]
    )


def test_the_cap_counts_the_time_since_the_last_command_up_to_one_tick():
    control, monotonic = paced_control()
    control.start(1.0)

    # The first command has no earlier one, and gets one tick's step.
    assert ticks(control, monotonic, 0.0, 3, dt=PERIOD / 2) == pytest.approx(
        [STEP, 1.5 * STEP, 2 * STEP]
    )
    # After a pause, no more than one tick's step either.
    control.start(-1.0)
    assert ticks(control, monotonic, 0.0, 1, dt=5.0) == pytest.approx([STEP])
    # A clock that did not move allows no change.
    assert ticks(control, monotonic, 0.0, 1, dt=0.0) == pytest.approx([STEP])


def test_a_new_target_5_ms_after_a_command_moves_it_only_as_far_as_5_ms_allow():
    control, monotonic = paced_control()
    control.start(1.0)
    [*_, command] = ticks(control, monotonic, 0.0, 4)
    assert command == pytest.approx(0.6)

    monotonic.now += 0.005
    control.start(-1.0)
    [replaced] = control.tick(0.0)

    assert abs(replaced - command) <= MAX_ACCEL * 0.005 + 1e-12
    assert replaced == pytest.approx(0.6 - MAX_ACCEL * 0.005)


def test_the_cap_returns_the_command_itself_once_within_a_step():
    # previous + change could round past max_speed; the command itself cannot.
    assert limit_change(0.6, 0.6 - STEP / 3) == 0.6
    assert limit_change(0.0, 0.1) == 0.0
    assert limit_change(0.6, 0.0) == pytest.approx(STEP)
    # By the time since the previous command, at most one tick.
    assert limit_change(0.6, 0.0, DEFAULT_CONFIG, 0.005) == pytest.approx(MAX_ACCEL * 0.005)
    assert limit_change(0.6, 0.0, DEFAULT_CONFIG, 10.0) == pytest.approx(STEP)
    assert limit_change(0.6, 0.0, DEFAULT_CONFIG, -1.0) == 0.0


def test_the_error_wraps_at_pi():
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    control = GazeControl(DEFAULT_CONFIG, FakeClock())
    control.start(2 * math.pi + 0.5)
    assert control.setpoint == pytest.approx(0.5)

    # 3.0 from -3.0 is 0.28 rad clockwise, not 6 rad counterclockwise.
    control.start(3.0)
    assert control.tick(-3.0) == pytest.approx([-STEP])

    # -3.1 from 3.1 is 0.08 rad, inside the deadband.
    other = GazeControl(DEFAULT_CONFIG, FakeClock())
    other.start(-3.1)
    assert other.tick(3.1) == [0.0]


def test_a_person_is_lost_after_one_second_unseen():
    clock = FakeClock(10.0)
    control = GazeControl(DEFAULT_CONFIG, clock)
    control.start(0.5, last_seen=10.0)

    clock.now = 10.99
    assert len(control.tick(0.0)) == 1
    clock.now = 11.01
    assert control.tick(0.0) == [0.0, 0.0, 0.0]
    assert not control.running
    assert control.tick(0.0) == []


def test_a_sighting_restarts_the_lost_timer():
    clock = FakeClock(10.0)
    control = GazeControl(DEFAULT_CONFIG, clock)
    control.start(0.5, last_seen=10.0)

    clock.now = 10.5
    assert control.sight(0.4, 10.5)
    clock.now = 11.4
    assert len(control.tick(0.0)) == 1
    assert control.setpoint == pytest.approx(0.4)
    clock.now = 11.51
    assert control.tick(0.0) == [0.0, 0.0, 0.0]


def test_a_person_is_lost_one_second_after_the_sighting_not_after_the_start():
    clock = FakeClock(10.0)
    control = GazeControl(DEFAULT_CONFIG, clock)
    # Seen 0.5 s before the target was set.
    control.start(0.5, last_seen=9.5)

    assert len(control.tick(0.0)) == 1
    clock.now = 10.49
    assert len(control.tick(0.0)) == 1
    clock.now = 10.51
    assert control.tick(0.0) == [0.0, 0.0, 0.0]

    # A sighting already older than that is lost at the first tick.
    stale = GazeControl(DEFAULT_CONFIG, FakeClock(10.0))
    stale.start(0.5, last_seen=5.0)
    assert stale.tick(0.0) == [0.0, 0.0, 0.0]


def test_a_clock_that_jumps_back_loses_the_person():
    clock = FakeClock(100.0)
    control = GazeControl(DEFAULT_CONFIG, clock)
    control.start(0.5, last_seen=100.0)

    # A simulation restart: the clock is far behind the last sighting.
    clock.now = 3.0
    assert control.tick(0.0) == [0.0, 0.0, 0.0]
    assert not control.running


def test_a_fixed_heading_is_never_lost():
    clock = FakeClock(10.0)
    control = GazeControl(DEFAULT_CONFIG, clock)
    control.start(0.5)

    clock.now = 1000.0
    assert len(control.tick(0.0)) == 1
    assert control.running
    # A sighting is a person's alone.
    assert not control.sight(0.9, 1000.0)
    assert control.setpoint == pytest.approx(0.5)


def test_a_stop_is_three_zeros_then_nothing():
    control = GazeControl(DEFAULT_CONFIG, FakeClock())
    control.start(1.0)
    control.tick(0.0)

    assert control.stop() == [0.0, 0.0, 0.0]
    assert control.stop() == []
    assert control.tick(0.0) == []
    assert not control.running


def test_an_unknown_yaw_commands_zero_and_restarts_the_ramp():
    control, monotonic = paced_control()
    control.start(1.0)
    ticks(control, monotonic, 0.0, 2)

    assert ticks(control, monotonic, None, 1) == [0.0]
    assert ticks(control, monotonic, 0.0, 1) == pytest.approx([STEP])
    assert ticks(control, monotonic, math.nan, 1) == [0.0]
    assert control.running


def test_an_older_sighting_does_not_move_the_setpoint():
    control = GazeControl(DEFAULT_CONFIG, FakeClock(10.0))
    control.start(0.5, last_seen=10.0)

    assert not control.sight(0.9, 9.0)
    assert not control.sight(0.9, 10.0)
    assert control.setpoint == pytest.approx(0.5)
    assert control.sight(0.9, 10.1)
    assert (control.setpoint, control.last_seen) == (pytest.approx(0.9), 10.1)


def test_a_new_target_continues_from_the_current_command():
    control, monotonic = paced_control()
    control.start(1.0)
    ticks(control, monotonic, 0.0, 3)

    control.start(-1.0)
    assert ticks(control, monotonic, 0.0, 1) == pytest.approx([0.6 - STEP])

    # After a stop, the next target starts from rest.
    control.stop()
    control.start(1.0)
    assert ticks(control, monotonic, 0.0, 1) == pytest.approx([STEP])


def test_the_config_keeps_cmd_vel_and_sane_values():
    with pytest.raises(ValueError, match="ADR-0051"):
        GazeConfig(cmd_vel_topic="/cmd_vel_nav")
    with pytest.raises(ValueError, match="min_speed"):
        GazeConfig(min_speed=0.7)
    with pytest.raises(ValueError, match="rate_hz"):
        GazeConfig(rate_hz=0.0)
    with pytest.raises(ValueError, match="deadband_deg"):
        GazeConfig(deadband_deg=-1.0)
    config = GazeConfig()
    assert (config.rate_hz, config.gain, config.min_speed, config.max_speed) == (
        15.0,
        1.2,
        0.15,
        0.6,
    )
    assert (config.deadband_deg, config.lost_after_s, config.accel_cap) == (6.0, 1.0, 3.2)


def test_the_config_refuses_more_than_the_hard_ceilings():
    with pytest.raises(ValueError, match=r"max_speed must not be above 0\.6 rad/s.*ADR-0051"):
        GazeConfig(max_speed=0.61)
    with pytest.raises(ValueError, match=r"accel_cap must not be above 3\.2 rad/s2.*ADR-0051"):
        GazeConfig(accel_cap=3.3)
    with pytest.raises(ValueError, match="max_speed"):
        GazeConfig(max_speed=2.0, accel_cap=100.0)
    # Lower is allowed: the ceilings are limits, not settings.
    slow = GazeConfig(max_speed=0.4, accel_cap=1.0)
    assert (slow.max_speed, slow.accel_cap) == (0.4, 1.0)


def test_the_control_law_keeps_to_the_ceilings_whatever_its_config_says():
    # A frozen config whose checks were bypassed.
    wide = GazeConfig()
    object.__setattr__(wide, "max_speed", 2.0)
    object.__setattr__(wide, "accel_cap", 100.0)
    monotonic = FakeClock(100.0)
    control = GazeControl(wide, FakeClock(), monotonic)
    control.start(3.0)

    commands = ticks(control, monotonic, 0.0, 5)

    assert commands == pytest.approx([STEP, 2 * STEP, MAX_SPEED, MAX_SPEED, MAX_SPEED])
    assert proportional_command(3.0, wide) == MAX_SPEED


# -- The loop over ROS -----------------------------------------------------------------


class Recorder:
    """A second node that hears /cmd_vel and publishes Nav2's goal status, on its own thread."""

    def __init__(self) -> None:
        self.node = rclpy.create_node(f"gaze_recorder_{random.randint(0, 1_000_000)}")
        self._lock = threading.Lock()
        self._messages: list[tuple[float, TwistStamped]] = []
        self.listeners = []
        self.node.create_subscription(TwistStamped, "/cmd_vel", self._heard, 100)
        self.nav_status = self.node.create_publisher(
            GoalStatusArray, NAV_STATUS_TOPIC, qos_profile_action_status_default
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _heard(self, msg: TwistStamped) -> None:
        with self._lock:
            self._messages.append((time.monotonic(), msg))
        for listener in self.listeners:
            listener(msg.twist.angular.z)

    def messages(self) -> list[TwistStamped]:
        with self._lock:
            return [msg for _, msg in self._messages]

    def commands(self, since: float = 0.0) -> list[float]:
        """The angular z of every command heard, or of those heard after `since` (monotonic)."""
        with self._lock:
            return [msg.twist.angular.z for t, msg in self._messages if t >= since]

    def count(self) -> int:
        with self._lock:
            return len(self._messages)

    def publish_goal_status(self, status: int) -> None:
        """One navigate_to_pose goal in `status`, as Nav2 reports a goal from any client."""
        goal = GoalStatus(status=status)
        goal.goal_info.goal_id.uuid = [random.randint(0, 255) for _ in range(16)]
        self.nav_status.publish(GoalStatusArray(status_list=[goal]))

    def close(self) -> None:
        self._executor.shutdown()
        self.node.destroy_node()
        self._thread.join(timeout=2)


def assert_rotation_only(messages) -> None:
    for msg in messages:
        twist = msg.twist
        assert (twist.linear.x, twist.linear.y, twist.linear.z) == (0.0, 0.0, 0.0)
        assert (twist.angular.x, twist.angular.y) == (0.0, 0.0)
        assert msg.header.frame_id == "base_footprint"
        assert abs(twist.angular.z) <= DEFAULT_CONFIG.max_speed


@pytest.fixture
def recorder():
    node = Recorder()
    yield node
    try:
        # Whatever a test did, the loop never sent anything but a turn in place.
        assert_rotation_only(node.messages())
    finally:
        node.close()


@pytest.fixture
def make_loop(connector, recorder):
    """GazeLoop(connector, ...), once its /cmd_vel and goal status reach the recorder."""
    loops: list[GazeLoop] = []

    def make(**kwargs) -> GazeLoop:
        loop = GazeLoop(connector, **kwargs)
        loops.append(loop)
        assert wait_until(
            lambda: loop._publisher._publisher.get_subscription_count() > 0
            and recorder.nav_status.get_subscription_count() > 0
        ), "The recorder and the loop did not discover each other."
        return loop

    yield make
    for loop in loops:
        loop.shutdown()


def transform(parent: str, child: str, xyz, rpy, stamp: TimeMsg | None = None) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp = TimeMsg() if stamp is None else stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
    q = quaternion_from_euler(*rpy)
    t.transform.rotation.x, t.transform.rotation.y = q[0], q[1]
    t.transform.rotation.z, t.transform.rotation.w = q[2], q[3]
    return t


def mount_camera(connector) -> None:
    connector._tf_buffer.set_transform_static(
        transform("base_footprint", CAMERA_FRAME, (0.1, 0.0, 1.2), CAMERA_AHEAD), "test"
    )


def place_base(connector, yaw: float) -> None:
    """A base that stays at `yaw` in the map, as static transforms that never go stale."""
    connector._tf_buffer.set_transform_static(
        transform("map", "base_footprint", (0.0, 0.0, 0.0), (0.0, 0.0, yaw)), "test"
    )
    mount_camera(connector)


class FakeBase:
    """A base that turns at the last command heard, as twist_mux and the drive would.

    It writes its pose into the agent's TF buffer at 50 Hz, stamped with the
    node clock, as the robot's odometry and localisation would.
    """

    def __init__(self, connector, recorder: Recorder, yaw: float = 0.0) -> None:
        self._buffer = connector._tf_buffer
        self._clock = connector.node.get_clock()
        self._lock = threading.Lock()
        self._yaw = yaw
        self._rate = 0.0
        mount_camera(connector)
        self._publish()
        recorder.listeners.append(self._command)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def yaw(self) -> float:
        with self._lock:
            return self._yaw

    def _command(self, angular_z: float) -> None:
        with self._lock:
            self._rate = angular_z

    def _run(self) -> None:
        last = time.monotonic()
        while not self._closed.wait(0.02):
            now = time.monotonic()
            with self._lock:
                self._yaw = wrap_angle(self._yaw + self._rate * (now - last))
            last = now
            self._publish()

    def _publish(self) -> None:
        stamp = self._clock.now().to_msg()
        self._buffer.set_transform(
            transform("map", "base_footprint", (0.0, 0.0, 0.0), (0.0, 0.0, self.yaw), stamp),
            "fake_base",
        )

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)


@pytest.fixture
def make_base(connector, recorder):
    bases: list[FakeBase] = []

    def make(yaw: float = 0.0) -> FakeBase:
        base = FakeBase(connector, recorder, yaw)
        bases.append(base)
        return base

    yield make
    for base in bases:
        base.close()


def camera_info() -> CameraInfo:
    info = CameraInfo(width=WIDTH, height=HEIGHT)
    info.k = [FX, 0.0, CX, 0.0, FX, HEIGHT / 2, 0.0, 0.0, 1.0]
    return info


def person(track_id: int, heading: float, base_yaw: float) -> PersonDetection:
    """A person on track `track_id` whose face is at `heading` in the map, seen from `base_yaw`.

    The body box is centred BODY_OFFSET_PX to the face's right, so the face
    alone gives `heading`.
    """
    u = CX - FX * math.tan(wrap_angle(heading - base_yaw))
    msg = PersonDetection(track_id=track_id, score=0.9)
    msg.bbox.center.position.x, msg.bbox.center.position.y = u + BODY_OFFSET_PX, 240.0
    msg.bbox.size_x, msg.bbox.size_y = 160.0, 400.0
    msg.face_bbox.center.position.x, msg.face_bbox.center.position.y = u, 80.0
    msg.face_bbox.size_x = msg.face_bbox.size_y = 60.0
    msg.face_det_score = 0.9
    return msg


def people_array(people, stamp: TimeMsg) -> PersonDetectionArray:
    msg = PersonDetectionArray(people=list(people))
    msg.header.stamp = stamp
    msg.header.frame_id = CAMERA_FRAME
    msg.camera_info = camera_info()
    return msg


def show(connector, feed: VisionFeed, people: dict[int, float], base_yaw: float):
    """Hand the feed one array with a person per track id at each map heading; its PeopleFrame."""
    stamp = connector.node.get_clock().now().to_msg()
    msg = people_array(
        [person(track_id, heading, base_yaw) for track_id, heading in people.items()], stamp
    )
    feed._on_people(msg)
    return feed.people_at(stamp, 0.0, NO_CANCEL)


class Camera:
    """Hands the feed an array at 15 Hz with the people of `people`, seen from the base's yaw."""

    def __init__(self, connector, feed: VisionFeed, base: FakeBase) -> None:
        self.people: dict[int, float] = {}
        self._connector, self._feed, self._base = connector, feed, base
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._closed.wait(1 / 15):
            show(self._connector, self._feed, dict(self.people), self._base.yaw)

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)


def stamp_ns(stamp: TimeMsg) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def assert_silent_after_stop(recorder: Recorder, quiet_s: float = 0.3) -> None:
    """The last three commands are the stop's zeros, and none comes after them.

    Called once the stop has happened. The gaze thread sends the zeros after
    stop() returns, so this first waits until no command has come for three
    ticks, then checks that none comes for `quiet_s` more.
    """
    settle_s = 3 / DEFAULT_CONFIG.rate_hz
    deadline = time.monotonic() + 3.0
    count, changed = recorder.count(), time.monotonic()
    while time.monotonic() - changed < settle_s:
        assert time.monotonic() < deadline, "The loop kept publishing after the stop."
        time.sleep(0.01)
        if recorder.count() != count:
            count, changed = recorder.count(), time.monotonic()
    time.sleep(quiet_s)
    assert recorder.count() == count, "A command came after the stop's zeros."
    assert recorder.commands()[-3:] == [0.0, 0.0, 0.0]


class PublishSpy:
    """Wraps the loop's RotationPublisher.publish, records each command, and can block it.

    Each record is (time.monotonic() when publish was called, the angular z,
    the calling thread's name), made once the real publish has returned.
    While `gate` is clear, a publish first waits for it, as on a transport
    that blocks, and sets `blocked`.
    """

    def __init__(self, loop: GazeLoop, monkeypatch) -> None:
        self._publish = loop._publisher.publish
        self._changed = threading.Condition()
        self._read = 0
        self._records: list[tuple[float, float, str]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.blocked = threading.Event()
        monkeypatch.setattr(loop._publisher, "publish", self._spy)

    def _spy(self, angular_z: float) -> None:
        called = time.monotonic()
        if not self.gate.is_set():
            self.blocked.set()
            # Bounded, so a test that fails with the gate shut cannot hang.
            self.gate.wait(5.0)
        self._publish(angular_z)
        with self._changed:
            self._records.append((called, angular_z, threading.current_thread().name))
            self._changed.notify_all()

    def next(self, timeout: float = 2.0) -> float:
        """The command published after the one next() returned last, once it is out."""
        with self._changed:
            assert self._changed.wait_for(lambda: len(self._records) > self._read, timeout), (
                "No command was published."
            )
            self._read += 1
            return self._records[self._read - 1][1]

    def records(self) -> list[tuple[float, float, str]]:
        with self._changed:
            return list(self._records)

    def commands(self) -> list[float]:
        return [command for _, command, _ in self.records()]


def assert_capped(records) -> None:
    """Each command differs from the one before by no more than 3.2 rad/s2 allows in between.

    The time between two commands counts as at most one tick, and the first
    command starts from rest. Records as PublishSpy's, without a stop's
    zeros, which are not capped.
    """
    previous_at, previous = -math.inf, 0.0
    for at, command, _ in records:
        assert abs(command) <= MAX_SPEED
        allowed = MAX_ACCEL * min(at - previous_at, PERIOD)
        assert abs(command - previous) <= allowed + 1e-9, (
            f"{previous} to {command} in {at - previous_at:.4f} s"
        )
        previous_at, previous = at, command


def timed(call, limit: float = 1.0) -> float:
    """How long `call()` took, on a thread of its own; math.inf when it had not returned in `limit`.

    Re-raises what the call raised. A call that does not return is left on
    its daemon thread.
    """
    result: dict = {}

    def run() -> None:
        started = time.monotonic()
        try:
            call()
        except BaseException as e:
            result["error"] = e
        result["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(limit)
    if "error" in result:
        raise result["error"]
    return result.get("elapsed", math.inf)


class Call:
    """`call()` on a thread of its own, started at once; join() returns or raises its result."""

    def __init__(self, call) -> None:
        self._result: dict = {}
        self._thread = threading.Thread(target=self._run, args=(call,), daemon=True)
        self._thread.start()

    def _run(self, call) -> None:
        try:
            self._result["value"] = call()
        except BaseException as e:
            self._result["error"] = e

    def join(self, timeout: float = 3.0):
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "The call did not return."
        if "error" in self._result:
            raise self._result["error"]
        return self._result.get("value")


@pytest.fixture
def held_tf_wait(monkeypatch):
    """Holds follow_person's TF wait, the one camera_transform with a timeout, until released."""
    waiting, release = threading.Event(), threading.Event()
    real = gaze_module.camera_transform

    def held(header, tf_buffer, timeout_s=0.2):
        if timeout_s > 0:
            waiting.set()
            release.wait(5.0)
        return real(header, tf_buffer, timeout_s)

    monkeypatch.setattr(gaze_module, "camera_transform", held)
    yield SimpleNamespace(waiting=waiting, release=release)
    release.set()


def test_the_loop_publishes_on_cmd_vel_through_the_forbidden_names_check(
    connector, make_loop, monkeypatch
):
    made = []
    helper = ToolContext.publisher

    def spy(ctx, msg_type, topic, qos, **kwargs):
        publisher = helper(ctx, msg_type, topic, qos, **kwargs)
        made.append((msg_type, topic, kwargs, publisher))
        return publisher

    monkeypatch.setattr(ToolContext, "publisher", spy)

    loop = make_loop()

    # Made by the ctx helper, which checks FORBIDDEN_NAMES, with /cmd_vel
    # alone as the exception.
    [(msg_type, topic, kwargs, publisher)] = made
    assert (msg_type, topic, kwargs) == (
        TwistStamped,
        "/cmd_vel",
        {"allow_forbidden": ("/cmd_vel",)},
    )
    assert loop._publisher._publisher is publisher
    assert loop._publisher.topic_name == "/cmd_vel"
    infos = [
        info
        for info in connector.node.get_publishers_info_by_topic("/cmd_vel")
        if info.node_name == connector.node.get_name()
    ]
    assert [info.topic_type for info in infos] == ["geometry_msgs/msg/TwistStamped"]
    # The same helper refuses /cmd_vel to anyone who does not name it.
    with pytest.raises(ValueError, match="/cmd_vel"):
        ToolContext(connector).publisher(TwistStamped, "/cmd_vel", 10)


def test_a_remap_of_cmd_vel_onto_another_input_is_refused(connector):
    # A remap onto cmd_vel_nav would rank the loop's commands with Nav2's.
    # The ctx allows it, since /cmd_vel_nav is not forbidden; the publisher
    # does not.
    node = rclpy.create_node(
        f"gaze_remapped_{random.randint(0, 1_000_000)}",
        cli_args=["--ros-args", "-r", "/cmd_vel:=/cmd_vel_nav"],
        use_global_arguments=False,
    )
    remapped = SimpleNamespace(node=node, _tf_buffer=connector._tf_buffer)
    try:
        with pytest.raises(
            ValueError,
            match=r"^The gaze loop's publisher resolves to /cmd_vel_nav, not /cmd_vel "
            r"\(ADR-0051\)\. Remove the remap\.$",
        ):
            RotationPublisher(ToolContext(remapped))
        with pytest.raises(ValueError, match="resolves to /cmd_vel_nav"):
            GazeLoop(remapped)
        # A remap of another name onto /cmd_vel is refused by the ctx itself.
        onto = rclpy.create_node(
            f"gaze_remapped_onto_{random.randint(0, 1_000_000)}",
            cli_args=["--ros-args", "-r", "/cmd_vel_nav:=/cmd_vel"],
            use_global_arguments=False,
        )
        try:
            with pytest.raises(ValueError, match="/cmd_vel is in FORBIDDEN_NAMES"):
                ToolContext(SimpleNamespace(node=onto)).publisher(TwistStamped, "/cmd_vel_nav", 10)
        finally:
            onto.destroy_node()
    finally:
        node.destroy_node()


def test_the_publisher_clamps_to_the_ceiling_and_refuses_what_is_not_finite(
    recorder, make_loop, caplog
):
    loop = make_loop()

    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="must be a finite number"):
            loop._publisher.publish(bad)
    with caplog.at_level(logging.WARNING, logger="walkie_agent.gaze"):
        for command in (0.25, 0.61, -1.0, 5.0):
            loop._publisher.publish(command)

    assert wait_until(lambda: recorder.count() == 4)
    assert recorder.commands() == [0.25, MAX_SPEED, -MAX_SPEED, MAX_SPEED]
    assert "A gaze command of 0.61 rad/s was clamped to 0.6 rad/s." in caplog.text
    assert_rotation_only(recorder.messages())


def test_no_config_widens_the_publisher_or_the_loop(connector, recorder, make_loop, monkeypatch):
    # A config whose checks were bypassed, as a frozen dataclass allows.
    wide = GazeConfig()
    object.__setattr__(wide, "max_speed", 2.0)
    object.__setattr__(wide, "accel_cap", 100.0)
    publisher = RotationPublisher(
        ToolContext(connector), SimpleNamespace(max_speed=2.0, cmd_vel_topic="/cmd_vel")
    )
    assert wait_until(lambda: publisher._publisher.get_subscription_count() > 0)

    publisher.publish(2.0)

    assert wait_until(lambda: recorder.count() == 1)
    assert recorder.commands() == [MAX_SPEED]

    loop = make_loop(config=wide, yaw_source=lambda: 0.0)
    spy = PublishSpy(loop, monkeypatch)
    loop.hold_heading("seat_1", 3.0, cancel=NO_CANCEL)
    for _ in range(5):
        spy.next()
    loop.stop(STOP_LOOKING)

    # Up to 0.6 rad/s, at no more than 3.2 rad/s2.
    assert max(abs(command) for command in spy.commands()) == MAX_SPEED
    assert_capped(spy.records()[:5])


def test_a_fixed_heading_is_reached_and_held(recorder, make_loop, make_base, monkeypatch):
    base = make_base(yaw=-0.4)
    loop = make_loop()
    spy = PublishSpy(loop, monkeypatch)

    loop.hold_heading("seat_2", 1.0, cancel=NO_CANCEL)

    assert wait_until(lambda: abs(wrap_angle(1.0 - base.yaw)) <= DEADBAND, timeout=6.0)
    # Held for longer than a person would be kept unseen.
    time.sleep(DEFAULT_CONFIG.lost_after_s + 0.3)
    status = loop.status()
    assert (status.running, status.target, status.kind) == (True, "seat_2", HEADING)
    assert status.setpoint == pytest.approx(1.0)
    assert abs(wrap_angle(1.0 - base.yaw)) <= DEADBAND + 0.02
    # Inside the deadband the loop commands nothing, so the base stays put.
    assert set(recorder.commands(since=time.monotonic() - 0.5)) == {0.0}
    commands = recorder.commands()
    assert max(commands) == pytest.approx(0.6)
    # The acceleration cap held all the way, by the time between the
    # commands. A little over a step allows for the float sums.
    assert all(abs(b - a) <= STEP + 1e-9 for a, b in zip([0.0, *commands], commands))
    assert_capped(spy.records())

    assert loop.stop(STOP_LOOKING) == "seat_2"
    assert_silent_after_stop(recorder)
    assert loop.stop(STOP_LOOKING) is None
    status = loop.status()
    assert (status.running, status.stop_reason) == (False, STOP_LOOKING)
    assert status.stopped_at == pytest.approx(status.now, abs=1.0)
    assert_rotation_only(recorder.messages())


def test_a_new_target_waits_for_the_next_tick_and_keeps_the_cap(recorder, make_loop, monkeypatch):
    loop = make_loop(yaw_source=lambda: 0.0)
    spy = PublishSpy(loop, monkeypatch)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    read = 1
    while spy.next() != MAX_SPEED:
        read += 1

    # Just after a command went out, a target the other way.
    loop.hold_heading("seat_2", -1.0, cancel=NO_CANCEL)
    reversal = [spy.next() for _ in range(4)]
    loop.stop(STOP_LOOKING)

    records = spy.records()[: read + 4]
    assert_capped(records)
    # The new target did not bring its first command forward: it went out at
    # the next scheduled tick, about a period after the last, as every other did.
    gaps = [b[0] - a[0] for a, b in zip(records, records[1:])]
    assert min(gaps) >= PERIOD / 2, gaps
    # Down from 0.6, no faster than the cap allows.
    assert reversal[-1] < reversal[0] <= MAX_SPEED
    assert reversal == sorted(reversal, reverse=True)
    assert_silent_after_stop(recorder)


def test_a_slow_tick_does_not_bring_the_next_command_forward(recorder, make_loop, monkeypatch):
    calls = 0

    def slow_second_yaw() -> float:
        # The second tick takes over two ticks' time, as on a TF lock held elsewhere.
        nonlocal calls
        calls += 1
        if calls == 2:
            time.sleep(0.15)
        return 0.0

    loop = make_loop(yaw_source=slow_second_yaw)
    spy = PublishSpy(loop, monkeypatch)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    for _ in range(4):
        spy.next()
    loop.stop(STOP_LOOKING)

    records = spy.records()[:4]
    # The slow tick's command went out late; the next one came a full tick after it.
    assert records[2][0] - records[1][0] >= PERIOD
    assert_capped(records)
    assert_silent_after_stop(recorder)


def test_a_person_is_followed_by_their_track(connector, recorder, make_loop, make_base):
    feed = VisionFeed(connector)
    base = make_base(yaw=0.0)
    loop = make_loop(vision=feed)
    # Within the camera's 45 degrees either side, at FX on a 640 px image.
    first = show(connector, feed, {4: 0.6}, base.yaw)
    [key] = first.tracks
    camera = Camera(connector, feed, base)
    try:
        camera.people = {4: 0.6, 9: -0.6}

        loop.follow_person("per_4", key, 0.6, stamp_ns(first.msg.header.stamp), cancel=NO_CANCEL)

        assert wait_until(lambda: abs(wrap_angle(0.6 - base.yaw)) <= DEADBAND, timeout=5.0)
        status = loop.status()
        assert (status.running, status.target, status.kind) == (True, "per_4", PERSON)
        # The person walks round to the other side of the robot.
        camera.people = {4: -0.3, 9: -0.6}
        assert wait_until(lambda: abs(wrap_angle(-0.3 - base.yaw)) <= DEADBAND, timeout=5.0)
        assert loop.status().setpoint == pytest.approx(-0.3, abs=0.05)
        assert loop.status().last_seen > status.last_seen

        # They leave the image: lost after a second, then silence.
        camera.people = {9: -0.6}
        left = time.monotonic()
        assert wait_until(lambda: not loop.status().running, timeout=3.0)
        assert time.monotonic() - left == pytest.approx(DEFAULT_CONFIG.lost_after_s, abs=0.3)
        assert loop.status().stop_reason == LOST
        assert_silent_after_stop(recorder)
    finally:
        camera.close()


def test_the_starting_heading_comes_from_the_newest_array_when_the_handle_has_none(
    connector, make_loop
):
    feed = VisionFeed(connector)
    place_base(connector, yaw=0.2)
    loop = make_loop(vision=feed)
    [key] = show(connector, feed, {3: 0.7}, 0.2).tracks

    loop.follow_person("per_3", key, None, 0, cancel=NO_CANCEL)

    assert loop.status().setpoint == pytest.approx(0.7, abs=1e-6)
    loop.stop(STOP_LOOKING)
    # Out of view now: there is nothing to start from.
    show(connector, feed, {5: 0.1}, 0.2)
    with pytest.raises(StepError) as error:
        loop.follow_person("per_3", key, None, 0, cancel=NO_CANCEL)
    assert error.value.code == "UNAVAILABLE"
    assert "find_people" in error.value.message


STALE_HANDLE = (
    r"Not turning toward per_3: the handle saw them 5\.\d s ago, and {unseen}\. The robot "
    r"stops following a person 1 s after the last sighting\. Call find_people for a current "
    r"handle\."
)


def test_a_handle_seen_5_s_ago_starts_nothing_unless_the_newest_array_shows_them(
    connector, recorder, make_loop
):
    feed = VisionFeed(connector)
    place_base(connector, yaw=0.2)
    loop = make_loop(vision=feed)
    seen_ns = connector.node.get_clock().now().nanoseconds - 5_000_000_000

    # No array has come at all.
    with pytest.raises(StepError) as error:
        loop.follow_person("per_3", TrackKey(3, 0), 0.7, seen_ns, cancel=NO_CANCEL)
    assert (error.value.code, error.value.data) == ("STALE", {"handle": "per_3"})
    assert re.fullmatch(
        STALE_HANDLE.format(unseen=r"no people array came in the last 1 s"), error.value.message
    )
    # The newest array shows someone else.
    show(connector, feed, {5: 0.1}, 0.2)
    with pytest.raises(StepError) as error:
        loop.follow_person("per_3", TrackKey(3, 0), 0.7, seen_ns, cancel=NO_CANCEL)
    assert error.value.code == "STALE"
    assert re.fullmatch(
        STALE_HANDLE.format(unseen=r"the newest people array does not show them"),
        error.value.message,
    )
    time.sleep(0.2)
    assert recorder.count() == 0
    assert loop.status().target is None

    # The newest array shows them: the loop starts from that sighting, not the handle's.
    frame = show(connector, feed, {3: -0.4}, 0.2)
    [key] = frame.tracks
    loop.follow_person("per_3", key, 0.7, seen_ns, cancel=NO_CANCEL)

    status = loop.status()
    assert (status.running, status.target) == (True, "per_3")
    assert status.setpoint == pytest.approx(-0.4, abs=1e-6)
    assert status.last_seen == pytest.approx(stamp_ns(frame.msg.header.stamp) / 1e9)
    loop.stop(STOP_LOOKING)


def test_a_handle_from_the_newest_array_starts_from_that_array_whatever_its_stamp(
    connector, recorder, make_loop
):
    feed = VisionFeed(connector)
    place_base(connector, yaw=0.0)
    loop = make_loop(vision=feed, yaw_source=lambda: 0.0)
    # A stamp whose sec + nanosec / 1e9 rounds below its nanoseconds / 1e9,
    # as about one stamp in eight does.
    now_ns = connector.node.get_clock().now().nanoseconds
    seen_ns = next(
        ns for ns in range(now_ns, now_ns - 10_000, -1) if ns // 10**9 + ns % 10**9 / 1e9 < ns / 1e9
    )
    stamp = TimeMsg(sec=seen_ns // 10**9, nanosec=seen_ns % 10**9)
    feed._on_people(people_array([person(3, 0.4, 0.0)], stamp))
    [key] = feed.people_at(stamp, 0.0, NO_CANCEL).tracks

    # The handle's sighting is that array's: the same stamp. Its heading
    # differs here only to show which one the loop started from.
    loop.follow_person("per_3", key, 0.9, seen_ns, cancel=NO_CANCEL)

    status = loop.status()
    assert status.setpoint == pytest.approx(0.4, abs=1e-6)
    assert status.last_seen == seen_ns / 1e9
    loop.stop(STOP_LOOKING)


@pytest.fixture
def moved_clock(monkeypatch):
    """A clock that can be moved on, for the feed's track ages only: set [0]."""
    offset = [0.0]
    monkeypatch.setattr(
        vision_feed_module, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0])
    )
    return offset


def test_a_recycled_track_id_is_not_followed(connector, recorder, make_loop, moved_clock):
    feed = VisionFeed(connector)
    place_base(connector, yaw=0.0)
    loop = make_loop(vision=feed)
    sam_frame = show(connector, feed, {7: 0.5}, 0.0)
    [sam] = sam_frame.tracks
    loop.follow_person("per_sam", sam, 0.5, stamp_ns(sam_frame.msg.header.stamp), cancel=NO_CANCEL)

    # Sam leaves; once the feed has dropped the track, Bob gets track 7.
    moved_clock[0] = 61.0
    bob = None
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline and loop.status().running:
        [bob] = show(connector, feed, {7: -0.5}, 0.0).tracks
        time.sleep(1 / 15)

    assert bob.track_id == sam.track_id and bob != sam
    status = loop.status()
    assert (status.running, status.stop_reason) == (False, LOST)
    assert status.setpoint == pytest.approx(0.5)
    # The base never turned toward Bob, who is to the right.
    assert all(command >= 0.0 for command in recorder.commands())
    assert_silent_after_stop(recorder)


def base_at(connector, stamp: TimeMsg, yaw: float = 0.0) -> None:
    """The base at `yaw` in the map at `stamp`, as one localisation transform that goes stale."""
    connector._tf_buffer.set_transform(
        transform("map", "base_footprint", (0.0, 0.0, 0.0), (0.0, 0.0, yaw), stamp), "test"
    )


def test_a_frozen_tf_commands_zero_until_it_moves_again(
    connector, recorder, make_loop, monkeypatch
):
    clock = connector.node.get_clock()
    mount_camera(connector)
    # The newest base transform is 2 s old, past yaw_max_age_s: TF has stopped.
    base_at(connector, (clock.now() - Duration(seconds=2.0)).to_msg())
    loop = make_loop()
    spy = PublishSpy(loop, monkeypatch)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)

    # The heading is unknown: the loop holds the base still and keeps its target.
    assert [spy.next() for _ in range(4)] == [0.0, 0.0, 0.0, 0.0]
    assert loop.status().running

    # TF moves again, fresh for the next 0.8 s: the base turns, ramping from rest.
    base_at(connector, (clock.now() + Duration(seconds=0.3)).to_msg())
    turning = [command for command in (spy.next() for _ in range(4)) if command != 0.0]
    loop.stop(STOP_LOOKING)

    assert turning, "The loop did not turn once TF moved again."
    assert 0.0 < turning[0] <= STEP + 1e-9
    assert_silent_after_stop(recorder)


def test_a_clock_reset_loses_a_person_and_keeps_a_fixed_heading(connector, recorder, make_loop):
    clock = FakeClock(1000.0)
    loop = make_loop(vision=VisionFeed(connector), clock=clock, yaw_source=lambda: 0.0)
    # Seen now by the loop's clock; no array has come, so the handle's sighting counts.
    loop.follow_person("per_3", TrackKey(3, 0), 0.7, 1000 * 1_000_000_000, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)
    assert loop.status().running

    # The simulation restarts: the clock is far behind the last sighting.
    clock.now = 5.0

    assert wait_until(lambda: not loop.status().running)
    status = loop.status()
    assert (status.stop_reason, status.stopped_at) == (LOST, 5.0)
    assert_silent_after_stop(recorder)

    # A fixed heading has no sighting to lose.
    before = recorder.count()
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= before + 2)
    clock.now = 1.0
    time.sleep(0.3)
    assert loop.status().running
    loop.stop(STOP_LOOKING)
    assert_silent_after_stop(recorder)


def faceless(track_id: int, heading: float, base_yaw: float) -> PersonDetection:
    """person(), with no face found: only the body box, BODY_OFFSET_PX right of the face."""
    msg = person(track_id, heading, base_yaw)
    msg.face_det_score = 0.0
    msg.face_bbox = BoundingBox2D()
    return msg


def body_heading(heading: float) -> float:
    """The map heading of faceless(...)'s body box, from a base at yaw 0."""
    return math.atan(math.tan(heading) - BODY_OFFSET_PX / FX)


def test_a_person_without_a_face_is_followed_by_their_body_box(
    connector, recorder, make_loop
):
    feed = VisionFeed(connector)
    place_base(connector, yaw=0.0)
    loop = make_loop(vision=feed, yaw_source=lambda: 0.0)
    stamp = connector.node.get_clock().now().to_msg()
    feed._on_people(people_array([faceless(3, 0.4, 0.0)], stamp))
    [key] = feed.people_at(stamp, 0.0, NO_CANCEL).tracks

    # The start, from the newest array.
    loop.follow_person("per_3", key, None, 0, cancel=NO_CANCEL)
    assert loop.status().setpoint == pytest.approx(body_heading(0.4), abs=1e-6)

    # A sighting on the gaze thread.
    feed._on_people(
        people_array([faceless(3, -0.2, 0.0)], connector.node.get_clock().now().to_msg())
    )
    assert wait_until(lambda: abs(loop.status().setpoint - body_heading(-0.2)) < 1e-6)
    loop.stop(STOP_LOOKING)
    assert_silent_after_stop(recorder)


def test_an_array_waits_for_tf_at_its_stamp_for_tf_wait_s_and_no_longer(
    connector, recorder, make_loop
):
    feed = VisionFeed(connector)
    clock = connector.node.get_clock()
    mount_camera(connector)
    config = GazeConfig(tf_wait_s=0.5, lost_after_s=3.0)
    loop = make_loop(vision=feed, config=config, yaw_source=lambda: 0.0)
    first = clock.now().to_msg()
    base_at(connector, first)
    feed._on_people(people_array([person(3, 0.2, 0.0)], first))
    [key] = feed.people_at(first, 0.0, NO_CANCEL).tracks
    loop.follow_person("per_3", key, None, 0, cancel=NO_CANCEL)
    assert loop.status().setpoint == pytest.approx(0.2, abs=1e-6)

    # An array stamped after TF's newest transform: no sighting yet.
    image = clock.now().to_msg()
    feed._on_people(people_array([person(3, 0.5, 0.0)], image))
    time.sleep(0.15)
    assert loop.status().setpoint == pytest.approx(0.2, abs=1e-6)
    # TF reaches its stamp within tf_wait_s: the array is a sighting.
    base_at(connector, (clock.now() + Duration(seconds=0.05)).to_msg())
    assert wait_until(lambda: abs(loop.status().setpoint - 0.5) < 1e-6, timeout=0.5)
    assert loop.status().last_seen == pytest.approx(stamp_ns(image) / 1e9)

    # One that TF does not reach within tf_wait_s is dropped, and a TF that
    # comes later does not bring it back.
    time.sleep(0.1)
    late = clock.now().to_msg()
    feed._on_people(people_array([person(3, -0.4, 0.0)], late))
    time.sleep(config.tf_wait_s + 0.2)
    base_at(connector, (clock.now() + Duration(seconds=0.05)).to_msg())
    time.sleep(0.3)
    status = loop.status()
    assert status.running
    assert status.setpoint == pytest.approx(0.5, abs=1e-6)
    assert status.last_seen == pytest.approx(stamp_ns(image) / 1e9)
    loop.stop(STOP_LOOKING)
    assert_silent_after_stop(recorder)


def test_an_interrupt_returns_at_once_while_the_gaze_thread_is_mid_tick(recorder, make_loop):
    calls = 0
    mid_tick = threading.Event()
    release = threading.Event()

    def slow_yaw() -> float:
        # Three quick ticks, then one that is stuck, as on a TF lock.
        nonlocal calls
        calls += 1
        if calls > 3:
            mid_tick.set()
            release.wait(0.5)
        return 0.0

    toolbox = Toolbox(None)
    loop = make_loop(yaw_source=slow_yaw)
    loop.attach(toolbox)
    loop.hold_heading("seat_1", 1.0, cancel=toolbox.ctx.cancel)
    assert mid_tick.wait(2.0)

    started = time.monotonic()
    toolbox.interrupt()
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 0.05
    assert (loop.status().running, loop.status().stop_reason) == (False, INTERRUPT)
    # The stuck tick's command is dropped: the zeros are the last word.
    assert wait_until(lambda: recorder.count() == 6)
    assert_silent_after_stop(recorder)
    ramp, zeros = recorder.commands()[:3], recorder.commands()[3:]
    assert 0.0 < ramp[0] < ramp[1] < ramp[2] <= MAX_SPEED
    assert zeros == [0.0, 0.0, 0.0]
    # The run's cancel is set, so a look_at in the interrupted run is refused.
    with pytest.raises(StepError) as error:
        loop.hold_heading("seat_1", 1.0, cancel=toolbox.ctx.cancel)
    assert error.value.code == "CANCELLED"


def test_a_blocked_publish_holds_up_no_stop_and_no_callback(
    connector, recorder, make_loop, monkeypatch
):
    feed = VisionFeed(connector)
    toolbox = Toolbox(None)
    loop = make_loop(vision=feed, yaw_source=lambda: 0.0)
    loop.attach(toolbox)
    spy = PublishSpy(loop, monkeypatch)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    spy.next()
    # The next publish blocks in the transport, with the gaze thread inside it.
    spy.gate.clear()
    assert spy.blocked.wait(2.0)
    people = people_array([person(3, 0.5, 0.0)], connector.node.get_clock().now().to_msg())
    try:
        took = {
            "people callback": timed(lambda: feed._on_people(people)),
            "navigation status callback": timed(
                lambda: loop._on_navigation_status(GoalStatusArray())
            ),
            "status": timed(loop.status),
            "interrupt": timed(toolbox.interrupt),
            "new task": timed(toolbox.new_task),
            "stop": timed(lambda: loop.stop(STOP_LOOKING)),
        }
    finally:
        spy.gate.set()

    assert all(elapsed < 0.05 for elapsed in took.values()), took
    assert (loop.status().running, loop.status().stop_reason) == (False, INTERRUPT)
    # Once the publish returns: the command it held, then the stop's zeros,
    # all from the gaze thread, and then nothing.
    assert wait_until(lambda: spy.commands()[-3:] == [0.0, 0.0, 0.0])
    assert_silent_after_stop(recorder)
    records = spy.records()
    commands = [command for _, command, _ in records]
    assert len(commands) >= 5
    assert all(command != 0.0 for command in commands[:-3])
    assert {thread for *_, thread in records} == {"gaze_loop"}
    assert recorder.commands() == commands


def test_a_stop_that_waits_returns_once_its_zeros_are_out_and_raises_when_they_are_not(
    recorder, make_loop, monkeypatch
):
    loop = make_loop(yaw_source=lambda: 0.0)
    spy = PublishSpy(loop, monkeypatch)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    spy.next()

    assert loop.stop(STOP_LOOKING, wait_s=0.5) == "seat_1"

    # Published before stop() returned, not only queued.
    assert spy.commands()[-3:] == [0.0, 0.0, 0.0]
    assert_silent_after_stop(recorder)

    # With the publish blocked, the wait ends at its bound with TIMEOUT.
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    spy.next()
    spy.gate.clear()
    assert spy.blocked.wait(2.0)
    try:
        started = time.monotonic()
        with pytest.raises(StepError) as error:
            loop.stop(STOP_LOOKING, wait_s=0.2)
        took = time.monotonic() - started
        # A stop without a wait, as every listener makes, still returns at
        # once and does not raise.
        assert timed(lambda: loop.stop(STOP_LOOKING)) < 0.05
    finally:
        spy.gate.set()

    assert 0.2 <= took < 0.5
    assert (error.value.code, error.value.message, error.value.data) == (
        "TIMEOUT",
        "The gaze loop's stop was not confirmed: its zero commands on /cmd_vel were not "
        "published within 0.2 s, so the base may still be turning.",
        {"gaze": "stop not confirmed"},
    )
    # The stop stands, and its zeros go out once the publish returns.
    assert (loop.status().running, loop.status().stop_reason) == (False, STOP_LOOKING)
    assert wait_until(lambda: spy.commands()[-3:] == [0.0, 0.0, 0.0])
    assert_silent_after_stop(recorder)
    # Once they are out, a stop that waits returns at once.
    assert timed(lambda: loop.stop(STOP_LOOKING, wait_s=0.5)) < 0.05


def test_a_publisher_that_always_raises_is_retried_each_period_logged_once_and_shut_down(
    recorder, make_loop, monkeypatch, caplog
):
    loop = make_loop(yaw_source=lambda: 0.0)
    attempts: list[tuple[float, float]] = []

    def failing(angular_z: float) -> None:
        attempts.append((time.monotonic(), angular_z))
        raise RuntimeError("The publisher's context is not valid.")

    monkeypatch.setattr(loop._publisher, "publish", failing)
    with caplog.at_level(logging.INFO, logger="walkie_agent.gaze"):
        loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
        # The first tick fails and stops the loop; its zeros fail too, and
        # are tried again every period.
        assert wait_until(lambda: len(attempts) >= 5, 2.0)
        # A stop that waits is not confirmed while no batch of zeros went
        # out, and says the publish failed.
        with pytest.raises(StepError) as error:
            loop.stop(STOP_LOOKING, wait_s=0.2)
        thread = loop._thread
        took = timed(loop.shutdown, 3.0)

    assert (error.value.code, error.value.data) == ("TIMEOUT", {"gaze": "stop not confirmed"})
    assert error.value.message == (
        "The gaze loop's stop was not confirmed: publishing its zero commands on /cmd_vel "
        "failed, so the base may still be turning."
    )
    assert took < gaze_module._JOIN_TIMEOUT_S
    assert not thread.is_alive()
    assert loop.status().stop_reason == ERROR
    first, *zeros = attempts
    assert first[1] != 0.0
    assert all(z == 0.0 for _, z in zeros)
    gaps = [later - earlier for (earlier, _), (later, _) in zip(zeros, zeros[1:-1])]
    assert all(gap >= 0.95 * PERIOD for gap in gaps), gaps
    # One exception for the tick, and one for the whole run of failed zeros.
    assert [r.getMessage() for r in caplog.records if r.exc_info] == [
        "A gaze tick failed; the loop stops.",
        "Could not publish the gaze loop's stop. Trying again every 0.067 s; the next failures "
        "are not logged.",
    ]
    assert "did not end" not in caplog.text
    assert recorder.count() == 0


def test_zeros_that_fail_at_shutdown_are_tried_once_and_not_said_to_be_retried(
    make_loop, monkeypatch, caplog
):
    loop = make_loop(yaw_source=lambda: 0.0)
    publish = loop._publisher.publish
    commands: list[float] = []
    zeros: list[float] = []

    def zeros_fail(angular_z: float) -> None:
        if angular_z == 0.0:
            zeros.append(angular_z)
            raise RuntimeError("The publisher's context is not valid.")
        publish(angular_z)
        commands.append(angular_z)

    monkeypatch.setattr(loop._publisher, "publish", zeros_fail)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: commands)
    thread = loop._thread
    with caplog.at_level(logging.INFO, logger="walkie_agent.gaze"):
        took = timed(loop.shutdown, 3.0)

    assert took < gaze_module._JOIN_TIMEOUT_S
    assert not thread.is_alive()
    # One attempt, which stops at the first zero that raises.
    assert zeros == [0.0]
    assert [r.getMessage() for r in caplog.records if r.exc_info] == [
        "Could not publish the gaze loop's stop at shutdown."
    ]


def test_shutdown_with_a_blocked_publish_returns_after_its_join_timeout(
    recorder, make_loop, monkeypatch, caplog
):
    monkeypatch.setattr(gaze_module, "_JOIN_TIMEOUT_S", 0.2)
    loop = make_loop(yaw_source=lambda: 0.0)
    spy = PublishSpy(loop, monkeypatch)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    spy.next()
    spy.gate.clear()
    assert spy.blocked.wait(2.0)
    thread = loop._thread
    try:
        with caplog.at_level(logging.WARNING, logger="walkie_agent.gaze"):
            took = timed(loop.shutdown)
    finally:
        spy.gate.set()

    assert 0.2 <= took < 0.6
    assert "The gaze thread did not end within 0.2 s." in caplog.text
    # Once the publish returns, the thread sends the zeros and ends.
    thread.join(2.0)
    assert not thread.is_alive()
    assert spy.commands()[-3:] == [0.0, 0.0, 0.0]
    assert loop.status().stop_reason == SHUTDOWN
    assert_silent_after_stop(recorder)


def test_a_new_task_stops_the_loop(recorder, make_loop):
    toolbox = Toolbox(None)
    loop = make_loop(yaw_source=lambda: 0.0)
    loop.attach(toolbox)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)

    toolbox.new_task()

    assert (loop.status().running, loop.status().stop_reason) == (False, NEW_TASK)
    assert_silent_after_stop(recorder)


def test_a_stop_heard_stops_the_loop_at_once_with_the_publish_blocked(
    recorder, make_loop, monkeypatch
):
    toolbox = Toolbox(None)
    loop = make_loop(yaw_source=lambda: 0.0)
    loop.attach(toolbox)
    spy = PublishSpy(loop, monkeypatch)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    spy.next()
    spy.gate.clear()
    assert spy.blocked.wait(2.0)
    try:
        took = timed(toolbox.stop_heard)
    finally:
        spy.gate.set()

    assert took < 0.05
    assert (loop.status().running, loop.status().stop_reason) == (False, STOP_HEARD)
    # Not an interrupt: no cancel, and no run held back.
    assert not toolbox.ctx.cancel.is_set()
    assert not toolbox.interrupt_pending()
    assert wait_until(lambda: spy.commands()[-3:] == [0.0, 0.0, 0.0])
    assert_silent_after_stop(recorder)


def test_a_cancel_set_before_the_call_starts_nothing(recorder, make_loop):
    loop = make_loop(yaw_source=lambda: 0.0)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(StepError) as error:
        loop.hold_heading("seat_1", 1.0, cancel=cancel)

    assert error.value.code == "CANCELLED"
    time.sleep(0.2)
    assert recorder.count() == 0
    assert not loop.status().running


def test_a_stop_while_follow_person_waits_for_tf_cancels_it(
    connector, recorder, make_loop, held_tf_wait
):
    feed = VisionFeed(connector)
    place_base(connector, yaw=0.2)
    loop = make_loop(vision=feed, yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)
    [key] = show(connector, feed, {3: 0.7}, 0.2).tracks

    # Without a stored heading, follow_person reads the newest array and waits for TF.
    call = Call(lambda: loop.follow_person("per_3", key, None, 0, cancel=NO_CANCEL))
    assert held_tf_wait.waiting.wait(2.0)
    assert loop.stop(STOP_LOOKING) == "seat_1"
    held_tf_wait.release.set()

    with pytest.raises(StepError) as error:
        call.join()
    assert (error.value.code, error.value.message) == (
        "CANCELLED",
        "Not turning toward per_3: the gaze loop was stopped (stop_looking) during the call. "
        "Call look_at again if the robot should still look at per_3.",
    )
    assert error.value.data == {"handle": "per_3", "reason": STOP_LOOKING}
    status = loop.status()
    assert (status.running, status.target, status.stop_reason) == (False, "seat_1", STOP_LOOKING)
    assert_silent_after_stop(recorder)


def test_a_stop_while_hold_heading_prepares_cancels_it(recorder, make_loop, monkeypatch):
    loop = make_loop(yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)
    arms_known = loop._arms_known

    def stop_first():
        # A stop_looking that lands after the call began and before its target is set.
        loop.stop(STOP_LOOKING)
        return arms_known()

    monkeypatch.setattr(loop, "_arms_known", stop_first)

    with pytest.raises(StepError) as error:
        loop.hold_heading("seat_2", -1.0, cancel=NO_CANCEL)

    assert error.value.code == "CANCELLED"
    assert error.value.data == {"handle": "seat_2", "reason": STOP_LOOKING}
    status = loop.status()
    assert (status.running, status.target, status.stop_reason) == (False, "seat_1", STOP_LOOKING)
    assert_silent_after_stop(recorder)


def test_an_idle_stop_during_the_tf_wait_cancels_the_call_too(
    connector, recorder, make_loop, held_tf_wait
):
    feed = VisionFeed(connector)
    place_base(connector, yaw=0.2)
    toolbox = Toolbox(None)
    loop = make_loop(vision=feed, yaw_source=lambda: 0.0)
    loop.attach(toolbox)
    [key] = show(connector, feed, {3: 0.7}, 0.2).tracks

    call = Call(lambda: loop.follow_person("per_3", key, None, 0, cancel=NO_CANCEL))
    assert held_tf_wait.waiting.wait(2.0)
    # Nothing runs, so the new task's stop stops nothing; it still wins over the call.
    toolbox.new_task()
    held_tf_wait.release.set()

    with pytest.raises(StepError) as error:
        call.join()
    assert error.value.code == "CANCELLED"
    assert error.value.data == {"handle": "per_3", "reason": NEW_TASK}
    time.sleep(0.2)
    assert recorder.count() == 0
    status = loop.status()
    assert (status.running, status.target, status.stop_reason) == (False, None, None)


def test_shutdown_sends_zeros_and_joins_the_thread(connector, recorder, make_loop):
    feed = VisionFeed(connector)
    loop = make_loop(vision=feed, yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 3)
    thread = loop._thread

    loop.shutdown()

    assert not thread.is_alive()
    assert loop.status().stop_reason == SHUTDOWN
    assert_silent_after_stop(recorder)
    with pytest.raises(StepError) as error:
        loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert error.value.code == "UNAVAILABLE"
    with pytest.raises(StepError) as error:
        loop.follow_person("per_1", TrackKey(1, 0), 0.5, 0, cancel=NO_CANCEL)
    assert error.value.code == "UNAVAILABLE"
    # The feed no longer calls the loop.
    assert feed._people_listeners == ()
    loop.shutdown()


def test_shutdown_before_any_target_publishes_nothing(recorder, make_loop):
    loop = make_loop()

    loop.shutdown()

    time.sleep(0.2)
    assert recorder.count() == 0
    assert loop._thread is None


@pytest.mark.parametrize(
    "goal_status",
    # CANCELING too: Nav2's controller may still drive until the cancel completes.
    [GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING],
)
def test_a_navigation_goal_stops_the_loop_and_keeps_it_silent(recorder, make_loop, goal_status):
    loop = make_loop(yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 3)

    # A goal from another client: the agent sent none.
    recorder.publish_goal_status(goal_status)

    assert wait_until(lambda: not loop.status().running)
    status = loop.status()
    assert (status.stop_reason, status.navigation_active) == (NAVIGATION_ACTIVE, True)
    assert_silent_after_stop(recorder)
    with pytest.raises(StepError) as error:
        loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert error.value.code == "REFUSED"

    # Once the goal is over, a new target runs.
    recorder.publish_goal_status(GoalStatus.STATUS_SUCCEEDED)
    assert wait_until(lambda: not loop.status().navigation_active)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert loop.status().running


def test_a_goal_whose_server_is_gone_no_longer_refuses_a_target(recorder, make_loop):
    loop = make_loop(yaw_source=lambda: 0.0)
    recorder.publish_goal_status(GoalStatus.STATUS_EXECUTING)
    assert wait_until(lambda: loop.status().navigation_active)
    with pytest.raises(StepError) as error:
        loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert error.value.code == "REFUSED"

    # Nav2's action server goes away during the goal, so no status ends it.
    recorder.node.destroy_publisher(recorder.nav_status)

    assert wait_until(lambda: not loop.status().navigation_active)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert loop.status().running


def test_a_goal_reported_before_the_loop_subscribed_refuses_a_target(recorder, make_loop):
    # Nav2 reported the goal before the agent started: the status's
    # transient-local durability still delivers it to the loop, which joins late.
    recorder.publish_goal_status(GoalStatus.STATUS_EXECUTING)
    loop = make_loop(yaw_source=lambda: 0.0)

    assert wait_until(lambda: loop.status().navigation_active)
    with pytest.raises(StepError) as error:
        loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert error.value.code == "REFUSED"
    time.sleep(0.2)
    assert recorder.count() == 0


def test_without_nav2_the_loop_turns(connector, recorder):
    # Nav2 never ran: nothing publishes the goal status.
    recorder.node.destroy_publisher(recorder.nav_status)
    loop = GazeLoop(connector, yaw_source=lambda: 0.0)
    try:
        assert wait_until(lambda: loop._publisher._publisher.get_subscription_count() > 0)
        # Once an earlier test's status publisher, if any, is undiscovered.
        assert wait_until(lambda: loop._navigation_status.get_publisher_count() == 0)

        loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)

        assert wait_until(lambda: recorder.count() >= 3)
        status = loop.status()
        assert (status.running, status.navigation_active) == (True, False)
        assert all(command > 0.0 for command in recorder.commands())
        loop.stop(STOP_LOOKING)
        assert_silent_after_stop(recorder)
    finally:
        loop.shutdown()


HOME = ArmState(pose="home", travel_safe=True)


def fake_commander(last_state: ArmsState) -> SimpleNamespace:
    """What the loop reads of a CommanderClient: `last_state`, `unverified` and busy().

    busy() returns `goal`. `unverified` is None until a test sets it, as the
    client's is until it abandons a goal.
    """
    arm = SimpleNamespace(last_state=last_state, unverified=None, goal=None)
    arm.busy = lambda: arm.goal
    return arm


def test_an_arm_that_starts_moving_stops_the_loop(recorder, make_loop):
    arm = fake_commander(ArmsState(left=HOME, right=HOME))
    loop = make_loop(arm=arm, yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)

    # The commander forgets an arm's state when it sends that arm a goal.
    arm.last_state = ArmsState(left=None, right=HOME)

    assert wait_until(lambda: not loop.status().running)
    assert loop.status().stop_reason == ARM_MOTION
    assert_silent_after_stop(recorder)


def test_an_arm_out_stops_the_loop(recorder, make_loop):
    arm = fake_commander(ArmsState(left=HOME, right=HOME))
    loop = make_loop(arm=arm, yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)

    arm.last_state = ArmsState(left=HOME, right=ArmState(pose=None, travel_safe=False))

    assert wait_until(lambda: not loop.status().running)
    assert loop.status().stop_reason == ARM_OUT
    assert_silent_after_stop(recorder)


def test_arms_unknown_from_the_start_do_not_stop_the_loop(recorder, make_loop):
    arm = fake_commander(ArmsState())
    loop = make_loop(arm=arm, yaw_source=lambda: 0.0)

    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)

    time.sleep(0.3)
    assert loop.status().running
    assert recorder.count() >= 3


def test_a_commander_goal_while_an_arm_is_unknown_stops_the_loop(recorder, make_loop):
    arm = fake_commander(ArmsState())
    loop = make_loop(arm=arm, yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)

    # A goal that no arm atomic sent, for an arm the loop never knew: the
    # commander's last reading cannot show it moving.
    arm.goal = "the move of left_arm"

    assert wait_until(lambda: not loop.status().running)
    assert loop.status().stop_reason == ARM_MOTION
    assert_silent_after_stop(recorder)


def test_a_gripper_goal_with_both_arms_known_does_not_stop_the_loop(recorder, make_loop):
    arm = fake_commander(ArmsState(left=HOME, right=HOME))
    loop = make_loop(arm=arm, yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)

    # The grippers do not move the arms, and both arms are known.
    arm.goal = "the left_gripper command"

    time.sleep(0.3)
    assert loop.status().running


# The client's cause once it abandoned a gripper command: the goal is no
# longer in busy(), and a gripper command leaves both arms known.
GRIPPER_ABANDONED = (
    "the left_gripper command from an earlier call was given up on before its end was known"
)


def test_unverified_arms_stop_a_person_the_loop_follows(connector, recorder, make_loop):
    arm = fake_commander(ArmsState(left=HOME, right=HOME))
    # A clock that stays at the handle's sighting, so the person is never lost.
    clock = FakeClock(1000.0)
    loop = make_loop(vision=VisionFeed(connector), arm=arm, clock=clock, yaw_source=lambda: 0.0)
    loop.follow_person("per_3", TrackKey(3, 0), 0.7, 1000 * 1_000_000_000, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)
    # Still turning, so the stop below is unverified's.
    assert loop.status().running

    arm.unverified = GRIPPER_ABANDONED

    # Nothing else shows it: no goal is busy, and both arms are known and safe.
    assert (arm.busy(), arm.last_state.travel_safe) == (None, True)
    assert wait_until(lambda: not loop.status().running)
    status = loop.status()
    assert (status.target, status.kind, status.stop_reason) == ("per_3", PERSON, ARM_MOTION)
    assert_silent_after_stop(recorder)


def test_unverified_arms_stop_a_held_heading(recorder, make_loop):
    arm = fake_commander(ArmsState(left=HOME, right=HOME))
    loop = make_loop(arm=arm, yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)
    assert loop.status().running

    arm.unverified = GRIPPER_ABANDONED

    assert (arm.busy(), arm.last_state.travel_safe) == (None, True)
    assert wait_until(lambda: not loop.status().running)
    status = loop.status()
    assert (status.target, status.kind, status.stop_reason) == ("seat_1", HEADING, ARM_MOTION)
    assert_silent_after_stop(recorder)


def test_an_arm_out_stops_the_loop_as_arm_out_while_the_arms_are_unverified(recorder, make_loop):
    # Both hold before the first tick, so no tick sees only one of them. The
    # loop itself does not check the arms when a target is set; look_at does.
    arm = fake_commander(ArmsState(left=HOME, right=ArmState(pose=None, travel_safe=False)))
    arm.unverified = GRIPPER_ABANDONED
    loop = make_loop(arm=arm, yaw_source=lambda: 0.0)

    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)

    assert wait_until(lambda: not loop.status().running)
    assert loop.status().stop_reason == ARM_OUT
    assert_silent_after_stop(recorder)


def test_no_track_no_heading_or_no_vision_starts_nothing(connector, recorder, make_loop):
    loop = make_loop()
    with pytest.raises(StepError) as error:
        loop.follow_person("per_1", TrackKey(1, 0), 0.5, 0, cancel=NO_CANCEL)
    assert error.value.code == "UNAVAILABLE"
    # A seat seen while TF had no camera transform has no heading.
    with pytest.raises(StepError) as error:
        loop.hold_heading("seat_1", None, cancel=NO_CANCEL)
    assert error.value.code == "UNAVAILABLE"
    with pytest.raises(ValueError):
        loop.hold_heading("seat_1", math.nan, cancel=NO_CANCEL)

    with_vision = make_loop(vision=VisionFeed(connector))
    with pytest.raises(StepError) as error:
        with_vision.follow_person("per_2", None, 0.5, 0, cancel=NO_CANCEL)
    assert error.value.code == "REFUSED"
    assert "not tracked" in error.value.message

    time.sleep(0.2)
    assert recorder.count() == 0
