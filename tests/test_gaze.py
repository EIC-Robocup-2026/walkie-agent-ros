"""The gaze loop (ADR-0051, docs/architecture/hri.md "Gaze").

The control law is tested on its own, with a fake clock and the yaw passed
in. The GazeLoop tests run on the `connector` fixture's node, on Fast DDS and
a random domain (ADR-0019), so the /cmd_vel they publish on reaches no
robot: a second node records it, and stands in for Nav2's goal status. TF is
written straight into the connector's buffer, either fixed or by a fake base
that turns at the last recorded command. People arrays are handed to a real
VisionFeed's callback, as its executor would.
"""

import math
import random
import threading
import time
from types import SimpleNamespace

import pytest
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped, TwistStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_action_status_default
from sensor_msgs.msg import CameraInfo
from tf_transformations import quaternion_from_euler
from walkie_vision_msgs.msg import PersonDetection, PersonDetectionArray

from walkie_agent.arm.state import ArmState, ArmsState
from walkie_agent.gaze import (
    ARM_MOTION,
    ARM_OUT,
    DEFAULT_CONFIG,
    HEADING,
    INTERRUPT,
    LOST,
    NAV_STATUS_TOPIC,
    NAVIGATION_ACTIVE,
    NEW_TASK,
    PERSON,
    SHUTDOWN,
    STOP_LOOKING,
    GazeConfig,
    GazeControl,
    GazeLoop,
    limit_change,
    proportional_command,
    wrap_angle,
)
from walkie_agent.perception import vision_feed as vision_feed_module
from walkie_agent.perception.vision_feed import TrackKey, VisionFeed
from walkie_agent.toolbox import StepError, ToolContext, Toolbox

NO_CANCEL = threading.Event()
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
    control = GazeControl(DEFAULT_CONFIG, FakeClock())
    control.start(1.0)

    commands = [control.tick(0.0)[0] for _ in range(4)]

    assert commands == pytest.approx([STEP, 2 * STEP, 0.6, 0.6])


def test_the_acceleration_cap_holds_on_a_reversal():
    control = GazeControl(DEFAULT_CONFIG, FakeClock())
    control.start(1.0)
    for _ in range(3):
        control.tick(0.0)

    # The base is now past the setpoint by a radian: the full command is -0.6.
    commands = [control.tick(2.0)[0] for _ in range(7)]

    assert commands == pytest.approx(
        [0.6 - STEP, 0.6 - 2 * STEP, 0.6 - 3 * STEP, 0.6 - 4 * STEP, 0.6 - 5 * STEP, -0.6, -0.6]
    )


def test_the_cap_returns_the_command_itself_once_within_a_step():
    # previous + change could round past max_speed; the command itself cannot.
    assert limit_change(0.6, 0.6 - STEP / 3) == 0.6
    assert limit_change(0.0, 0.1) == 0.0
    assert limit_change(0.6, 0.0) == pytest.approx(STEP)


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


def test_an_old_handle_gets_a_full_second_from_the_start():
    clock = FakeClock(10.0)
    control = GazeControl(DEFAULT_CONFIG, clock)
    # Seen 5 s before look_at was called.
    control.start(0.5, last_seen=5.0)

    clock.now = 10.99
    assert len(control.tick(0.0)) == 1
    clock.now = 11.01
    assert control.tick(0.0) == [0.0, 0.0, 0.0]


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
    control = GazeControl(DEFAULT_CONFIG, FakeClock())
    control.start(1.0)
    control.tick(0.0)
    control.tick(0.0)

    assert control.tick(None) == [0.0]
    assert control.tick(0.0) == pytest.approx([STEP])
    assert control.tick(math.nan) == [0.0]
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
    control = GazeControl(DEFAULT_CONFIG, FakeClock())
    control.start(1.0)
    for _ in range(3):
        control.tick(0.0)

    control.start(-1.0)
    assert control.tick(0.0) == pytest.approx([0.6 - STEP])

    # After a stop, the next target starts from rest.
    control.stop()
    control.start(1.0)
    assert control.tick(0.0) == pytest.approx([STEP])


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
    """The last three commands are the stop's zeros, and none is sent after the stop.

    Called once the stop has happened. The commands are told apart by their
    stamps, since the last ones may still be on their way.
    """
    stopped_ns = recorder.node.get_clock().now().nanoseconds
    time.sleep(quiet_s)
    messages = recorder.messages()
    assert [msg for msg in messages if stamp_ns(msg.header.stamp) > stopped_ns] == []
    assert [msg.twist.angular.z for msg in messages[-3:]] == [0.0, 0.0, 0.0]


def test_the_loop_publishes_on_cmd_vel_through_the_forbidden_names_check(connector, make_loop):
    loop = make_loop()

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


def test_the_publisher_takes_only_a_bounded_angular_z(recorder, make_loop):
    loop = make_loop()

    for bad in (0.61, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            loop._publisher.publish(bad)
    loop._publisher.publish(0.25)

    assert wait_until(lambda: recorder.count() == 1)
    [msg] = recorder.messages()
    assert msg.twist.angular.z == 0.25
    assert_rotation_only([msg])


def test_a_fixed_heading_is_reached_and_held(recorder, make_loop, make_base):
    base = make_base(yaw=-0.4)
    loop = make_loop()

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
    # The acceleration cap held all the way. A little over a step allows
    # for the float sums.
    assert all(abs(b - a) <= STEP + 1e-9 for a, b in zip([0.0, *commands], commands))

    assert loop.stop(STOP_LOOKING) == "seat_2"
    assert_silent_after_stop(recorder)
    assert loop.stop(STOP_LOOKING) is None
    status = loop.status()
    assert (status.running, status.stop_reason) == (False, STOP_LOOKING)
    assert status.stopped_at == pytest.approx(status.now, abs=1.0)
    assert_rotation_only(recorder.messages())


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
    assert recorder.commands() == pytest.approx([STEP, 2 * STEP, 0.6, 0.0, 0.0, 0.0])
    # The run's cancel is set, so a look_at in the interrupted run is refused.
    with pytest.raises(StepError) as error:
        loop.hold_heading("seat_1", 1.0, cancel=toolbox.ctx.cancel)
    assert error.value.code == "CANCELLED"


def test_a_new_task_stops_the_loop(recorder, make_loop):
    toolbox = Toolbox(None)
    loop = make_loop(yaw_source=lambda: 0.0)
    loop.attach(toolbox)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)

    toolbox.new_task()

    assert (loop.status().running, loop.status().stop_reason) == (False, NEW_TASK)
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
    "goal_status", [GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING]
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


HOME = ArmState(pose="home", travel_safe=True)


def test_an_arm_that_starts_moving_stops_the_loop(recorder, make_loop):
    arm = SimpleNamespace(last_state=ArmsState(left=HOME, right=HOME))
    loop = make_loop(arm=arm, yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)

    # The commander forgets an arm's state when it sends that arm a goal.
    arm.last_state = ArmsState(left=None, right=HOME)

    assert wait_until(lambda: not loop.status().running)
    assert loop.status().stop_reason == ARM_MOTION
    assert_silent_after_stop(recorder)


def test_an_arm_out_stops_the_loop(recorder, make_loop):
    arm = SimpleNamespace(last_state=ArmsState(left=HOME, right=HOME))
    loop = make_loop(arm=arm, yaw_source=lambda: 0.0)
    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)
    assert wait_until(lambda: recorder.count() >= 2)

    arm.last_state = ArmsState(left=HOME, right=ArmState(pose=None, travel_safe=False))

    assert wait_until(lambda: not loop.status().running)
    assert loop.status().stop_reason == ARM_OUT
    assert_silent_after_stop(recorder)


def test_arms_unknown_from_the_start_do_not_stop_the_loop(recorder, make_loop):
    arm = SimpleNamespace(last_state=ArmsState())
    loop = make_loop(arm=arm, yaw_source=lambda: 0.0)

    loop.hold_heading("seat_1", 1.0, cancel=NO_CANCEL)

    time.sleep(0.3)
    assert loop.status().running
    assert recorder.count() >= 3


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
