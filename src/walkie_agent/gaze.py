"""GazeLoop: turn the base toward a person or a fixed heading (ADR-0051).

The HRI task scores looking at the guest who talks, and the head servo only
tilts, so the robot looks at someone by turning its base. look_at starts the
loop on a person handle, followed by its track, or on a seat handle, held as
a fixed map-frame heading; stop_looking, a stop or a reset, a new task, a
navigation goal and an arm motion end it (hri.md, Gaze).

The loop is walkie-agent-v2's FaceTracker: a setpoint and a proportional
controller on the live heading. The setpoint is a map-frame heading, from the
column of the target's face in each people array that shows it, the array's
camera_info and the camera's transform at the array's stamp
(perception/geometry.py), so a turn during detection does not skew it. The
controller runs at 15 Hz on the heading from TF now, so the error shrinks as
the base turns, and a stale pixel offset cannot make it overshoot.

It publishes geometry_msgs/TwistStamped on /cmd_vel, twist_mux's
lowest-priority input and the one exception to FORBIDDEN_NAMES. That is safe
only because the loop can do no more than ADR-0051 judged safe:
  - rotation only: RotationPublisher takes an angular z and nothing else,
    and clamps it to SPEED_CEILING, which no config can raise;
  - Nav2 and the operator win: twist_mux ranks /cmd_vel below them, and the
    loop stops, then stays silent, while any navigate_to_pose goal is active;
  - arms in: look_at checks the arms, every arm atomic stops the loop before
    it moves an arm, and the loop stops itself when the commander's last
    reading shows an arm out or moving, or a goal runs while an arm is
    unknown;
  - it stops by itself 1 s after its person was last seen.
The gaze thread is the only writer of /cmd_vel, and publishes outside the
loop's lock. Every stop queues three zero commands for it rather than
publishing them; the thread sends them, after any command it computed before
the stop, and then nothing, so twist_mux falls back to its other inputs. If
the agent dies, or a publish blocks in the transport, twist_mux drops the
input 0.5 s after its last message, and the drive controller stops the base
0.5 s after its last command.

The control law, GazeControl, is plain logic with an injected clock that
takes the yaw as an argument, so it is tested without ROS.
"""

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from action_msgs.msg import GoalStatus, GoalStatusArray
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TwistStamped
from rai.communication.ros2 import ROS2Connector
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_action_status_default,
)
from rclpy.time import Time
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion
from vision_msgs.msg import BoundingBox2D
from walkie_vision_msgs.msg import PersonDetection, PersonDetectionArray

from walkie_agent.arm.poses import ARMS
from walkie_agent.perception.geometry import camera_transform, ray_heading
from walkie_agent.perception.vision_feed import PeopleFrame, TrackKey, VisionFeed
from walkie_agent.toolbox import StepError, ToolContext
from walkie_agent.tools.navigation import MAP_FRAME, ROBOT_FRAME

if TYPE_CHECKING:
    from walkie_agent.arm.commander import CommanderClient
    from walkie_agent.toolbox import Toolbox

logger = logging.getLogger(__name__)

# The one forbidden name the loop may use (ADR-0051). A constant rather than
# the config's value, so that no config change can widen the exception.
CMD_VEL_TOPIC = "/cmd_vel"
# Nav2's goal status for navigate_to_pose, which lists the goals of every
# client, an operator's included.
NAV_STATUS_TOPIC = "/navigate_to_pose/_action/status"

# Reliable, which matches a reliable and a best-effort subscription.
# VOLATILE, so that an input that joins late never receives an old command.
# KEEP_LAST, so that a slow subscriber costs old samples rather than a queue
# that grows. ROS does not promise that publish() never blocks, so only the
# gaze thread publishes, outside the loop's lock, and no stop waits for it
# unless its caller asks to (GazeLoop.stop).
CMD_VEL_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

# Zero commands sent by every stop, as walkie-agent-v2 did: one may be lost,
# and the last command the drive receives must be a zero.
STOP_ZEROS = 3

# The hard limits of ADR-0051, separate from GazeConfig's adjustable values:
# GazeConfig refuses values above them, the control law never exceeds them,
# and RotationPublisher clamps every command to SPEED_CEILING whatever its
# config says. 0.6 rad/s is the speed v2 tuned on the robot; 3.2 rad/s2 is
# walkie-ros's velocity_smoother limit for Nav2, which this input does not
# pass.
SPEED_CEILING = 0.6
ACCEL_CEILING = 3.2

# How long the callers that move the robot next, a drive and an arm
# atomic, wait in stop() for the zeros to be published. The gaze thread
# sends them as soon as it is awake; a wait that runs this long means it
# is held up, as by a publish that blocks, and twist_mux drops the input
# 0.5 s after its last message either way.
STOP_WAIT_S = 0.5

# The goal states in which Nav2 drives, or may still drive, the base.
# CANCELING too: until the cancel completes, Nav2's controller may still
# publish, and the goal is not over.
_ACTIVE_GOAL = frozenset(
    (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING)
)

# How many people arrays that show the target wait for TF at once. The gaze
# thread looks each one up without waiting and tries again next tick, so a
# few are kept: with only the newest, arrays that always arrive just before
# TF reaches their stamp would never give a sighting.
_PENDING_KEPT = 4

# How long shutdown() waits for the gaze thread to send the stop's zeros and
# end. A tick never waits, so it ends well within this, unless a publish
# blocks in the transport; shutdown() then logs and returns.
_JOIN_TIMEOUT_S = 2.0

# Never set: the cancel of the feed read in follow_person, which does not wait.
_NEVER_SET = threading.Event()

# Why a target stopped, in GazeStatus.stop_reason, as passed to stop() or
# recorded by the loop itself. stop() takes any string: look_at adds its own
# reason for a turn that takes too long (tools/atomic/_gaze.py).
LOST = "lost"
STOP_LOOKING = "stop_looking"
INTERRUPT = "interrupt"
# "stop" or "never mind", which stops the loop between tasks too, where it
# interrupts nothing (Toolbox.on_stop_heard).
STOP_HEARD = "stop heard"
NEW_TASK = "new task"
NAVIGATION = "navigation"
NAVIGATION_ACTIVE = "navigation goal active"
ARM_MOTION = "arm motion"
ARM_OUT = "arm out"
SHUTDOWN = "shutdown"
ERROR = "error"

PERSON = "person"
HEADING = "heading"


@dataclass(frozen=True)
class GazeConfig:
    """The loop's settings.

    The control values are walkie-agent-v2's FaceTracker's, tuned on the
    robot; the acceleration cap is walkie-ros's velocity_smoother's.
    """

    # Control ticks per second. v2's HRI_FACE_TRACK_HZ, tuned on the robot.
    # Well inside twist_mux's 0.5 s input timeout.
    rate_hz: float = 15.0
    # rad/s of command per radian of heading error. v2's HRI_FACE_TRACK_KP,
    # tuned on the robot.
    gain: float = 1.2
    # The smallest command outside the deadband, in rad/s: below it the
    # base's stiction holds it still, and a small error would never close.
    # v2's HRI_FACE_TRACK_MIN_WZ, tuned on the robot.
    min_speed: float = 0.15
    # The largest command, in rad/s, below the 1.0 rad/s velocity_smoother
    # allows Nav2. v2's HRI_FACE_TRACK_MAX_WZ, tuned on the robot. At most
    # SPEED_CEILING.
    max_speed: float = 0.6
    # No command while the error is within this many degrees, so the base
    # does not hunt around the setpoint. v2's HRI_FACE_TRACK_DEADBAND_DEG,
    # tuned on the robot.
    deadband_deg: float = 6.0
    # A person unseen for this many seconds, by the arrays' stamps against
    # the node clock, is lost. v2's HRI_FACE_TRACK_LOST_SEC, tuned on the
    # robot.
    lost_after_s: float = 1.0
    # The largest change of command per second, in rad/s2: walkie-ros's
    # velocity_smoother limit for Nav2. This input does not pass the
    # smoother, so nothing else limits it. At most ACCEL_CEILING.
    accel_cap: float = 3.2
    # twist_mux's lowest-priority input. Any other name is refused: another
    # input would rank the loop above Nav2 or the operator (ADR-0051).
    cmd_vel_topic: str = CMD_VEL_TOPIC
    # How long a people array may wait for TF to reach its stamp before it
    # is dropped, as perception's tf_timeout_s. Not v2's: v2 read the
    # heading when it captured the image.
    tf_wait_s: float = 0.2
    # The live heading counts as unknown, and the command is 0, once the
    # newest map -> base_footprint is this much older than the node clock.
    # Not v2's: a TF that stopped would otherwise turn the base toward a
    # fixed heading for ever.
    yaw_max_age_s: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "rate_hz",
            "gain",
            "min_speed",
            "max_speed",
            "lost_after_s",
            "accel_cap",
            "tf_wait_s",
            "yaw_max_age_s",
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"GazeConfig.{name} must be a positive number, not {value}.")
        if not (math.isfinite(self.deadband_deg) and 0 <= self.deadband_deg < 180):
            raise ValueError(
                f"GazeConfig.deadband_deg must be from 0 to under 180, not {self.deadband_deg}."
            )
        if self.min_speed > self.max_speed:
            raise ValueError("GazeConfig.min_speed must not be above max_speed.")
        if self.max_speed > SPEED_CEILING:
            raise ValueError(
                f"GazeConfig.max_speed must not be above {SPEED_CEILING:g} rad/s, not "
                f"{self.max_speed} (ADR-0051)."
            )
        if self.accel_cap > ACCEL_CEILING:
            raise ValueError(
                f"GazeConfig.accel_cap must not be above {ACCEL_CEILING:g} rad/s2, not "
                f"{self.accel_cap} (ADR-0051)."
            )
        if self.cmd_vel_topic != CMD_VEL_TOPIC:
            raise ValueError(
                f"GazeConfig.cmd_vel_topic must be {CMD_VEL_TOPIC}, twist_mux's lowest-priority "
                f"input, not {self.cmd_vel_topic!r} (ADR-0051)."
            )

    @property
    def deadband_rad(self) -> float:
        return math.radians(self.deadband_deg)

    @property
    def period(self) -> float:
        """The time from one tick to the next, in seconds."""
        return 1.0 / self.rate_hz


DEFAULT_CONFIG = GazeConfig()


# -- The control law, without ROS -------------------------------------------------


def wrap_angle(angle: float) -> float:
    """`angle` in radians, wrapped to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def proportional_command(error: float, config: GazeConfig = DEFAULT_CONFIG) -> float:
    """v2's command for a heading error in radians, before the acceleration cap.

    0 inside the deadband; else the gain times the error, at least
    `min_speed` and at most `max_speed`, in the error's direction. Never
    above SPEED_CEILING, even for a config that bypassed its checks.
    """
    if abs(error) <= config.deadband_rad:
        return 0.0
    top = min(config.max_speed, SPEED_CEILING)
    speed = min(top, max(config.min_speed, config.gain * abs(error)))
    return math.copysign(speed, error)


def limit_change(
    command: float,
    previous: float,
    config: GazeConfig = DEFAULT_CONFIG,
    elapsed: float | None = None,
) -> float:
    """`command`, moved no further from `previous` than the acceleration cap allows in `elapsed`.

    `elapsed` is the time in seconds since `previous` was published, and
    counts as at most one tick: a longer pause allows no bigger step, so the
    command after it still ramps. None, as before the first command, is one
    tick. The cap is never above ACCEL_CEILING.
    """
    period = config.period
    elapsed = period if elapsed is None else min(max(elapsed, 0.0), period)
    max_step = min(config.accel_cap, ACCEL_CEILING) * elapsed
    change = command - previous
    if abs(change) <= max_step:
        # `command` itself, not previous + change, which can round past it
        # and so past max_speed.
        return command
    return previous + math.copysign(max_step, change)


class GazeControl:
    """The control law for one target at a time: setpoint, sightings, loss and the cap.

    Plain logic: `clock` returns the time in seconds on the clock of the
    sightings' stamps, the node clock in GazeLoop, and the base's yaw is
    passed to tick(). `monotonic`, time.monotonic by default, times the
    acceleration cap: a simulation clock that resets must not make the time
    between two commands negative or huge. Not thread-safe; GazeLoop calls
    it under its lock.
    """

    def __init__(
        self,
        config: GazeConfig,
        clock: Callable[[], float],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._monotonic = monotonic
        self._running = False
        self._setpoint: float | None = None
        self._last_seen: float | None = None
        # The last command returned, which the acceleration cap starts from,
        # and when it went out, on `monotonic`; None before the first.
        self._previous = 0.0
        self._previous_at: float | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def setpoint(self) -> float | None:
        """The map-frame yaw it turns toward, in radians; kept after a stop."""
        return self._setpoint

    @property
    def last_seen(self) -> float | None:
        """The stamp of the newest sighting of a person target; None for a fixed heading."""
        return self._last_seen

    def start(self, setpoint: float, last_seen: float | None = None) -> None:
        """Turn toward `setpoint`, a map-frame yaw in radians, replacing any target.

        With `last_seen`, the stamp of the sighting the setpoint came from,
        the target is a person and is lost `lost_after_s` after its last
        sighting, this one or a newer one, with no allowance for the start:
        a sighting already older than that is lost at the first tick.
        Without it, the target is a fixed heading and never lost. A target
        that replaces a running one starts from the current command, so the
        base does not jerk, and its first command is capped by the time
        since the last one went out.
        """
        if not math.isfinite(setpoint):
            raise ValueError(f"The setpoint must be a finite angle, not {setpoint}.")
        if not self._running:
            self._previous = 0.0
        self._running = True
        self._setpoint = wrap_angle(setpoint)
        self._last_seen = last_seen

    def sight(self, heading: float, stamp: float) -> bool:
        """A sighting of a person target: its heading in the map and its array's stamp.

        Only a sighting newer than the last moves the setpoint, so an array
        that arrives late cannot turn the base back to an older position.
        Returns whether it did.
        """
        if not self._running or self._last_seen is None or stamp <= self._last_seen:
            return False
        if not math.isfinite(heading):
            return False
        self._setpoint = wrap_angle(heading)
        self._last_seen = stamp
        return True

    def tick(self, yaw: float | None) -> list[float]:
        """The commands to publish now: one while running, the stop's zeros when lost, else none.

        `yaw` is the base's heading in the map now, None when TF has none:
        the command is then 0, and the next one starts from rest. A command
        differs from the last by at most the acceleration cap times the time
        since the last went out, at most one tick (limit_change).
        """
        if not self._running:
            return []
        if self._last_seen is not None and self._lost():
            return self.stop()
        now = self._monotonic()
        if yaw is None or not math.isfinite(yaw):
            command = 0.0
        else:
            assert self._setpoint is not None
            error = wrap_angle(self._setpoint - yaw)
            elapsed = None if self._previous_at is None else now - self._previous_at
            command = limit_change(
                proportional_command(error, self.config), self._previous, self.config, elapsed
            )
        self._previous = command
        self._previous_at = now
        return [command]

    def stop(self) -> list[float]:
        """Stop: the zero commands to publish, or none when nothing was running.

        The zeros are not capped: a stop takes effect at once (ADR-0051).
        """
        if not self._running:
            return []
        self._running = False
        self._previous = 0.0
        self._previous_at = self._monotonic()
        return [0.0] * STOP_ZEROS

    def published(self) -> None:
        """The commands tick() or stop() returned last have just gone out.

        GazeLoop publishes after it computes, so the time between two
        commands counts from when the previous one was sent.
        """
        self._previous_at = self._monotonic()

    def _lost(self) -> bool:
        assert self._last_seen is not None
        # From the last sighting alone: a person seen longer ago than
        # lost_after_s is lost, however recently the target was set. abs(),
        # so a clock that jumped back, as at a simulation restart, loses the
        # target rather than keeping it for the length of the jump.
        return abs(self._clock() - self._last_seen) > self.config.lost_after_s


# -- ROS ------------------------------------------------------------------------------


class RotationPublisher:
    """The loop's only way onto /cmd_vel: TwistStamped commands that turn the base in place.

    publish() takes an angular z and nothing else, and leaves every linear
    field and angular x and y at 0, so no caller can drive the base
    (ADR-0051, "Rotation only"). It clamps the angular z to SPEED_CEILING,
    not to the config's max_speed, so no config can widen it. The publisher
    is made with the ctx helper, whose FORBIDDEN_NAMES check lets /cmd_vel
    through only because it is named here.
    """

    def __init__(self, ctx: ToolContext, config: GazeConfig = DEFAULT_CONFIG) -> None:
        self._clock = ctx.node.get_clock()
        self._publisher = ctx.publisher(
            TwistStamped, config.cmd_vel_topic, CMD_VEL_QOS, allow_forbidden=(CMD_VEL_TOPIC,)
        )
        # The name as the node resolved it, remaps included: a remap onto
        # another twist_mux input would rank the loop above Nav2 or the
        # operator, and only /cmd_vel is forbidden, so the ctx would allow it.
        if self._publisher.topic_name != CMD_VEL_TOPIC:
            raise ValueError(
                f"The gaze loop's publisher resolves to {self._publisher.topic_name}, not "
                f"{CMD_VEL_TOPIC} (ADR-0051). Remove the remap."
            )

    @property
    def topic_name(self) -> str:
        return self._publisher.topic_name

    def publish(self, angular_z: float) -> None:
        """Command the base to turn at `angular_z` rad/s, counterclockwise positive.

        A value above SPEED_CEILING either way is sent as SPEED_CEILING, and
        logged: the control law never asks for more. Raises ValueError for a
        value that is not finite, which is never sent to the base.
        """
        if not math.isfinite(angular_z):
            raise ValueError(f"A gaze command must be a finite number, not {angular_z}.")
        if abs(angular_z) > SPEED_CEILING:
            logger.warning(
                f"A gaze command of {angular_z:g} rad/s was clamped to {SPEED_CEILING:g} rad/s."
            )
            angular_z = math.copysign(SPEED_CEILING, angular_z)
        msg = TwistStamped()
        msg.header.stamp = self._clock.now().to_msg()
        msg.header.frame_id = ROBOT_FRAME
        # A new message has every field at 0: only the yaw rate is set.
        msg.twist.angular.z = float(angular_z)
        self._publisher.publish(msg)


@dataclass(frozen=True)
class GazeStatus:
    """What the loop is doing, for look_at, stop_looking and the logs."""

    # The current target's name, such as per_51c0.1; once stopped, the last
    # target's. None before the first target.
    target: str | None
    # PERSON or HEADING; None before the first target.
    kind: str | None
    running: bool
    # The map-frame yaw it turns, or last turned, toward, in radians.
    setpoint: float | None
    # The stamp of the newest sighting of a person target, in seconds on the
    # node clock; None for a fixed heading.
    last_seen: float | None
    # Why the last target stopped, and when, in seconds on the node clock.
    # None while a target runs and before the first stop.
    stop_reason: str | None
    stopped_at: float | None
    # True while a navigate_to_pose goal is active: the loop then refuses
    # new targets.
    navigation_active: bool
    # The node clock when the status was taken, so that now - stopped_at is
    # how long ago the loop stopped.
    now: float


class _Target(NamedTuple):
    name: str
    kind: str
    # The followed track; None for a fixed heading.
    track: TrackKey | None
    # The arms the commander's last reading knew when the target was set.
    arms_known: frozenset[str]


class _Pending(NamedTuple):
    """A people array that shows the target, waiting for the gaze thread."""

    msg: PersonDetectionArray
    # The target's index in msg.people.
    index: int
    # header.stamp in seconds.
    stamp: float
    # time.monotonic() when it arrived, for tf_wait_s.
    arrived: float


class GazeLoop:
    """The `gaze` backend: turns the base toward one target at a time, on a thread of its own.

    Created once per agent, before the Toolbox: look_at and stop_looking
    load only when the backend is present when the Toolbox is built.
    `vision` is the agent's VisionFeed, without which no person can be
    followed; `arm` its CommanderClient, or None without --arms, of which
    the loop only reads `last_state` and busy(), neither of which waits for
    the commander. `clock`, the node clock in seconds by default, and
    `yaw_source`, the base's yaw in the map from TF by default, None when
    unknown, are there for tests.

    The setters, stop() and status() are safe from any thread. The loop
    publishes nothing until a target is set, a command every tick while one
    runs, three zeros when it stops, and then nothing. Only the gaze thread
    publishes: the setters and stop() only change the loop's state, and
    stop() waits for the zeros only when its caller asks it to.
    """

    def __init__(
        self,
        connector: ROS2Connector,
        vision: VisionFeed | None = None,
        arm: "CommanderClient | None" = None,
        config: GazeConfig = DEFAULT_CONFIG,
        *,
        clock: Callable[[], float] | None = None,
        yaw_source: Callable[[], float | None] | None = None,
    ) -> None:
        self.config = config
        self._vision = vision
        self._arm = arm
        # The Toolbox's ctx does not exist yet (see the class docstring), so
        # the loop has a ctx of its own for its two entities: the same
        # helpers, and the same FORBIDDEN_NAMES check on the resolved names.
        # Building it touches no ROS; its handle store stays empty.
        self._ctx = ToolContext(connector, logger=logger)
        self._node = self._ctx.node
        self._tf_buffer = self._ctx.tf_buffer
        self._clock = clock if clock is not None else self._node_seconds
        self._yaw_source = yaw_source if yaw_source is not None else self._tf_yaw

        # Guards everything below. Every holder only reads and updates
        # state: none publishes or waits while holding it, so a stop from
        # any thread takes it within a tick's bookkeeping, whatever the gaze
        # thread is doing, a blocked publish included.
        self._lock = threading.Lock()
        self._control = GazeControl(config, self._clock)
        self._target: _Target | None = None
        # Changes with every new target and every stop, so a tick that ran
        # while either happened drops its result instead of publishing it.
        self._run_id = 0
        self._pending: list[_Pending] = []
        self._navigation_active = False
        self._stop_reason: str | None = None
        self._stopped_at: float | None = None
        self._closed = False
        # The zero commands a stop queued for the gaze thread, the only
        # writer of /cmd_vel; 0 once sent.
        self._zeros_due = 0
        # How many stops queued their zeros, and how many of those the gaze
        # thread has sent, for stop(wait_s=...), which waits on `_sent`.
        self._stops_queued = 0
        self._stops_sent = 0
        self._sent = threading.Condition(self._lock)
        # Changes with every stop() and shutdown() call, even one that
        # stops nothing, and the reason of the last: a follow_person or a
        # hold_heading that saw another value on entry was overtaken by a
        # stop while it prepared, and starts nothing. The loop's own stops,
        # such as LOST, do not change it.
        self._stop_calls = 0
        self._last_stop_call: str | None = None
        # A dedicated thread that waits on an Event between ticks, as
        # half_duplex's speaking thread does: the agent's own code adds no
        # timers to the connector's executor, whose callbacks deliver the TF
        # and the people arrays a tick reads, and must not wait behind a
        # tick. It starts with the first target.
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

        self._publisher = RotationPublisher(self._ctx, config)
        self._navigation_status = self._ctx.subscription(
            GoalStatusArray,
            NAV_STATUS_TOPIC,
            self._on_navigation_status,
            qos_profile_action_status_default,
        )
        if vision is not None:
            vision.on_people(self._on_people)

    # -- Targets -----------------------------------------------------------------

    def follow_person(
        self,
        name: str,
        track: TrackKey | None,
        heading_map: float | None,
        stamp_ns: int,
        *,
        cancel: threading.Event,
    ) -> None:
        """Follow a person handle's track, replacing any target.

        `track`, `heading_map` and `stamp_ns` are the handle's
        (PersonPayload). The loop starts from the newer of two sightings:
        the handle's, `heading_map` at `stamp_ns`, and the newest array's,
        when that is no older than `lost_after_s` and shows `track`. It then
        turns toward each new sighting, and the person is lost `lost_after_s`
        after the last one, with no allowance for the start. It follows the
        TrackKey, not the bare track id: once VisionFeed has dropped a
        track, its id may name someone else. `name` is the handle's id, for
        status().

        Raises StepError:
          - REFUSED when `track` is None, an untracked person, and while a
            navigate_to_pose goal is active;
          - UNAVAILABLE without the vision backend, after shutdown(), and
            when there is no `heading_map` and the newest array gives none;
          - STALE when the newest array gives no sighting and the handle's is
            more than `lost_after_s` from the node clock: the loop would
            lose the person at once, so it starts nothing;
          - CANCELLED when `cancel`, the caller's ctx.cancel, is set, or when
            stop() was called during the call: a stop or a reset that came
            during the call wins.
        """
        if track is None:
            raise StepError(
                "REFUSED", f"{name} is not tracked, cannot follow.", {"handle": name}
            )
        if self._vision is None:
            raise StepError("UNAVAILABLE", "The vision backend is not available.")
        # Before the TF wait below, so a refusal does not wait first, and a
        # stop during the wait is seen in _start.
        with self._lock:
            self._check_can_start(name)
            stop_calls = self._stop_calls
        stored = None if heading_map is None else (heading_map, stamp_ns / 1e9)
        current, unseen = self._current_sighting(track)
        if current is not None and (stored is None or current[1] >= stored[1]):
            # No older than lost_after_s: latest_people checked it.
            heading, seen = current
        elif stored is not None:
            heading, seen = stored
            self._check_fresh(name, seen, unseen)
        else:
            raise StepError(
                "UNAVAILABLE",
                f"No heading toward {name}: none was stored when they were seen, and they are "
                "not in view now. Call find_people again.",
                {"handle": name},
            )
        target = _Target(name, PERSON, track, self._arms_known())
        self._start(target, heading, seen, cancel, stop_calls)

    def hold_heading(
        self, name: str, heading_map: float | None, *, cancel: threading.Event
    ) -> None:
        """Turn toward `heading_map`, a map-frame yaw in radians, and hold it, replacing any target.

        For a seat handle, whose heading was fixed when it was seen. Never
        lost. `name` is the handle's id, for status(). Raises StepError
        UNAVAILABLE when `heading_map` is None, as for a seat seen while TF
        had no camera transform, and after shutdown(); REFUSED while a
        navigate_to_pose goal is active; CANCELLED as follow_person() does.
        """
        if heading_map is None:
            raise StepError(
                "UNAVAILABLE",
                f"No heading toward {name}: TF had no camera position when it was seen.",
                {"handle": name},
            )
        if not math.isfinite(heading_map):
            raise ValueError(f"heading_map must be a finite angle, not {heading_map}.")
        with self._lock:
            stop_calls = self._stop_calls
        target = _Target(name, HEADING, None, self._arms_known())
        self._start(target, heading_map, None, cancel, stop_calls)

    # -- Stops -------------------------------------------------------------------

    def stop(self, reason: str, *, wait_s: float = 0.0) -> str | None:
        """Stop turning: three zero commands, then nothing. Returns the stopped target's name.

        Takes the lock only to change state: it marks the target stopped and
        queues the zeros for the gaze thread, which sends them when it next
        wakes, after any command it computed before the stop, and sends
        nothing after them. It returns at once, unless `wait_s` is given:
        then it waits up to that long for the zeros to be published, for a
        caller that moves the robot next, as a drive and an arm atomic do
        with STOP_WAIT_S. The Toolbox listeners, and every executor
        callback, never wait. A wait that runs out is logged and
        the stop stands: if a publish blocks in the transport, twist_mux
        drops the input 0.5 s after its last message, and the drive
        controller stops the base 0.5 s after its last command.

        Returns None, and changes nothing else, when no target runs, so that
        the reason of the last real stop, such as LOST, stays in status().
        Every call, even one that stops nothing, makes a follow_person() or
        a hold_heading() still preparing its target raise CANCELLED. Safe
        from any thread.
        """
        with self._lock:
            self._stop_calls += 1
            self._last_stop_call = reason
            name = self._stop_locked(reason)
            if wait_s > 0:
                self._wait_sent_locked(wait_s)
        return name

    def handle_interrupt(self) -> None:
        """The Toolbox interrupt listener: a stop or a reset ends the loop. Returns at once."""
        self.stop(INTERRUPT)

    def handle_new_task(self) -> None:
        """The Toolbox new-task listener: the handles the loop follows are gone."""
        self.stop(NEW_TASK)

    def handle_stop_heard(self) -> None:
        """The Toolbox stop-heard listener: "stop" or "never mind" ends the loop. Returns at once.

        Also between tasks, where the stop interrupts nothing and the loop
        would otherwise keep turning the base.
        """
        self.stop(STOP_HEARD)

    def attach(self, toolbox: "Toolbox") -> None:
        """Stop on every interrupt, every new task and every "stop" heard by `toolbox`."""
        toolbox.on_interrupt(self.handle_interrupt)
        toolbox.on_new_task(self.handle_new_task)
        toolbox.on_stop_heard(self.handle_stop_heard)

    def shutdown(self) -> None:
        """Stop for SHUTDOWN and end the gaze thread, which sends the stop's zeros first.

        Later targets are UNAVAILABLE. Waits up to _JOIN_TIMEOUT_S for the
        thread, and logs when it has not ended, as when a publish blocks in
        the transport; twist_mux and the drive controller then stop the base
        0.5 s after the last command that reached them.
        """
        with self._lock:
            self._stop_calls += 1
            self._last_stop_call = SHUTDOWN
            self._stop_locked(SHUTDOWN)
            self._closed = True
            thread = self._thread
            self._wake.set()
        if self._vision is not None:
            self._vision.remove_people_listener(self._on_people)
        if thread is not None and thread is not threading.current_thread():
            thread.join(_JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.warning(f"The gaze thread did not end within {_JOIN_TIMEOUT_S:g} s.")

    # -- Status ------------------------------------------------------------------

    def status(self) -> GazeStatus:
        with self._lock:
            target = self._target
            return GazeStatus(
                target=None if target is None else target.name,
                kind=None if target is None else target.kind,
                running=self._control.running,
                setpoint=self._control.setpoint,
                last_seen=self._control.last_seen,
                stop_reason=self._stop_reason,
                stopped_at=self._stopped_at,
                navigation_active=self._goal_active_locked(),
                now=self._clock(),
            )

    # -- Starting and stopping, under the lock ------------------------------------

    def _start(
        self,
        target: _Target,
        setpoint: float,
        last_seen: float | None,
        cancel: threading.Event,
        stop_calls: int,
    ) -> None:
        """Set `target`, unless a stop came since the caller read `stop_calls` on entry."""
        with self._lock:
            self._check_can_start(target.name)
            # Here, under the lock: Toolbox.interrupt() sets the cancel before
            # it calls the listeners, so either this sees it, or the target is
            # set first and the interrupt listener then stops it.
            if cancel.is_set():
                raise StepError("CANCELLED", "A stop or a reset interrupted the call.")
            # A stop() that came while the caller prepared, such as during
            # follow_person's TF wait, which ctx.cancel misses unless it was
            # an interrupt: stop_looking, a new task, a drive or an arm
            # motion. Starting now would turn the base after the stop's zeros.
            if self._stop_calls != stop_calls:
                reason = self._last_stop_call
                raise StepError(
                    "CANCELLED",
                    f"Not turning toward {target.name}: the gaze loop was stopped ({reason}) "
                    f"during the call. Call look_at again if the robot should still look at "
                    f"{target.name}.",
                    {"handle": target.name, "reason": reason},
                )
            self._control.start(setpoint, last_seen)
            self._target = target
            self._run_id += 1
            self._pending = []
            self._stop_reason = self._stopped_at = None
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="gaze_loop", daemon=True)
                self._thread.start()
            self._wake.set()
        logger.info(f"Gaze: turning toward {target.name} ({target.kind}).")

    def _check_can_start(self, name: str) -> None:
        if self._closed:
            raise StepError("UNAVAILABLE", "The gaze loop has shut down.")
        if self._goal_active_locked():
            raise StepError(
                "REFUSED",
                f"Not turning toward {name}: a navigation goal is active.",
                {"handle": name},
            )

    def _goal_active_locked(self) -> bool:
        """Whether a navigate_to_pose goal is active, by the newest goal status.

        An active goal counts only while the status's publisher, Nav2's
        action server, is still matched: a server that dies during a goal
        never reports its end, and every look_at would be refused until Nav2
        came back. The count is DDS's matching, which precedes any message,
        so a status that just arrived always has its publisher counted.
        """
        if self._navigation_active and self._navigation_status.get_publisher_count() == 0:
            self._navigation_active = False
        return self._navigation_active

    def _stop_locked(self, reason: str) -> str | None:
        """Stop the target for `reason` and queue its zeros. Its name, or None when none ran."""
        zeros = self._control.stop()
        if not zeros:
            return None
        name = self._ended(reason)
        self._queue_zeros(len(zeros))
        return name

    def _queue_zeros(self, count: int) -> None:
        """Have the gaze thread send `count` zero commands, at once and before anything else."""
        # Set, not added: a second stop before the first one's zeros went
        # out has nothing more to stop.
        self._zeros_due = count
        self._stops_queued += 1
        self._wake.set()

    def _wait_sent_locked(self, wait_s: float) -> None:
        """Wait up to `wait_s`, under the lock, until every zero queued so far is published."""
        if threading.current_thread() is self._thread:
            # The gaze thread sends them itself, after this returns.
            return
        queued = self._stops_queued
        deadline = time.monotonic() + wait_s
        while self._stops_sent < queued:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    f"The gaze loop's stop was not published within {wait_s:g} s. twist_mux "
                    "drops /cmd_vel 0.5 s after its last message, and the drive controller "
                    "stops the base 0.5 s after its last command."
                )
                return
            # Releases the lock while it waits.
            self._sent.wait(remaining)

    def _check_fresh(self, name: str, seen: float, unseen: str | None) -> None:
        """Raises StepError STALE when the handle's sighting at `seen` is already lost.

        `unseen` says why the newest array gave no sighting. Only the
        handle's sighting is checked: one from the newest array is no older
        than lost_after_s, as latest_people() checks.
        """
        age = self._clock() - seen
        lost_after_s = self.config.lost_after_s
        if age > lost_after_s:
            raise StepError(
                "STALE",
                f"Not turning toward {name}: the handle saw them {age:.1f} s ago, and "
                f"{unseen or 'the newest people array gives no newer sighting'}. The robot "
                f"stops following a person {lost_after_s:g} s after the last sighting. Call "
                "find_people for a current handle.",
                {"handle": name},
            )
        if age < -lost_after_s:
            raise StepError(
                "STALE",
                f"Not turning toward {name}: the handle's sighting is stamped {-age:.1f} s "
                "ahead of the robot's clock, which has moved back, as at a simulation restart. "
                "Call find_people for a current handle.",
                {"handle": name},
            )

    def _ended(self, reason: str) -> str:
        """Record that the target stopped, for `reason`. Returns its name."""
        assert self._target is not None
        self._run_id += 1
        self._pending = []
        self._stop_reason = reason
        self._stopped_at = self._clock()
        logger.info(f"Gaze: stopped turning toward {self._target.name} ({reason}).")
        return self._target.name

    # -- The gaze thread ---------------------------------------------------------

    def _run(self) -> None:
        """The only writer of /cmd_vel: computes under the lock, publishes outside it.

        One writer keeps the order: a command computed before a stop goes
        out before that stop's zeros, and none is computed after it. The
        ticks keep a fixed schedule: a wakeup, as for a new target, does not
        bring the next command forward, and after a slow tick the schedule
        restarts a period later rather than catching up. A stop's zeros go
        out as soon as the thread is awake.
        """
        period = self.config.period
        next_tick = time.monotonic()
        while True:
            with self._lock:
                zeros, self._zeros_due = self._zeros_due, 0
                queued = self._stops_queued
                closed = self._closed
                running = self._control.running
                # Cleared under the lock, where every setter and stop sets
                # it, so one that comes after this check ends the wait below.
                self._wake.clear()
            if zeros:
                self._send_zeros(zeros)
                with self._lock:
                    self._control.published()
                    self._stops_sent = queued
                    self._sent.notify_all()
                # As after a command: one for a target set since waits a
                # full period.
                next_tick = max(next_tick, time.monotonic() + period)
                continue
            if closed:
                return
            if not running:
                self._wake.wait()
                continue
            delay = next_tick - time.monotonic()
            if delay > 0:
                # Woken early, as by a new target, or not yet time: the
                # target takes effect at the scheduled tick.
                self._wake.wait(delay)
                continue
            try:
                commands = self._tick()
                for command in commands:
                    self._publisher.publish(command)
            except Exception:
                logger.exception("A gaze tick failed; the loop stops.")
                with self._lock:
                    self._stop_locked(ERROR)
            else:
                if commands:
                    with self._lock:
                        self._control.published()
            next_tick += period
            now = time.monotonic()
            if next_tick <= now:
                # Behind, as after a slow tick: the schedule restarts a
                # period from now rather than catching up, so two commands
                # never go out back to back.
                next_tick = now + period

    def _send_zeros(self, count: int) -> None:
        try:
            for _ in range(count):
                self._publisher.publish(0.0)
        except Exception:
            # The loop is stopped either way; a publisher that fails, as
            # while rclpy shuts down, must not end the gaze thread.
            logger.exception("Could not publish the gaze loop's stop.")

    def _tick(self) -> list[float]:
        """One control tick: the command to publish, computed under the lock, or none.

        A stop it makes, for the arms or a lost person, queues its zeros as
        stop() does and returns no command.
        """
        with self._lock:
            if not self._control.running:
                return []
            run_id = self._run_id
            target = self._target
            pending, self._pending = self._pending, []
        assert target is not None
        # Outside the lock: TF and the arm state. None of it waits, since
        # every lookup has a zero timeout and busy() takes only the client's
        # state lock, but it takes other locks.
        sighting, waiting = self._sighting(pending)
        yaw = self._yaw_source()
        arm_reason = self._arm_stop_reason(target.arms_known)
        with self._lock:
            if run_id != self._run_id:
                # A stop or a new target came meanwhile, and this tick's work
                # is for a target that is gone.
                return []
            if arm_reason is not None:
                self._stop_locked(arm_reason)
                return []
            # Before the arrays that came during the tick, which are newer.
            self._pending = (waiting + self._pending)[-_PENDING_KEPT:]
            if sighting is not None:
                self._control.sight(*sighting)
            commands = self._control.tick(yaw)
            if not self._control.running:
                # Lost: the commands are the stop's zeros, queued as any
                # stop's are.
                self._ended(LOST)
                self._queue_zeros(len(commands))
                return []
            return commands

    def _sighting(
        self, pending: list[_Pending]
    ) -> tuple[tuple[float, float] | None, list[_Pending]]:
        """The newest sighting TF can place, as (heading, stamp), and the newer arrays that wait.

        One camera_transform per array per tick, without waiting: an array
        whose stamp TF has not reached yet is tried again next tick, until
        `tf_wait_s` after it arrived. An array without the transform then,
        or whose ray has no yaw, gives no setpoint and is no sighting.
        """
        waiting: list[_Pending] = []
        now = time.monotonic()
        for frame in sorted(pending, key=_by_stamp, reverse=True):
            transform = camera_transform(frame.msg.header, self._tf_buffer, 0.0)
            if transform is None:
                if now - frame.arrived < self.config.tf_wait_s:
                    waiting.append(frame)
                continue
            heading = ray_heading(
                _target_box(frame.msg.people[frame.index]), frame.msg.camera_info, transform
            )
            # The older arrays go: a sighting from them would be older still.
            waiting.sort(key=_by_stamp)
            return (None if heading is None else (heading, frame.stamp)), waiting
        waiting.sort(key=_by_stamp)
        return None, waiting

    def _tf_yaw(self) -> float | None:
        """The base's yaw in the map from the newest map -> base_footprint, or None.

        None when TF has no such transform, or when the newest is more than
        `yaw_max_age_s` from the node clock: a TF that stopped would leave a
        heading that never changes, and the loop would turn for ever.
        """
        try:
            stamped = self._tf_buffer.lookup_transform(MAP_FRAME, ROBOT_FRAME, Time())
        except TransformException:
            return None
        stamp_ns = Time.from_msg(stamped.header.stamp).nanoseconds
        # Stamp 0 is a chain of static transforms, as in a test, which never
        # goes stale.
        if stamp_ns and abs(self._clock() - stamp_ns / 1e9) > self.config.yaw_max_age_s:
            return None
        q = stamped.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    # -- Arms ----------------------------------------------------------------------

    def _arm_state(self) -> Any:
        # last_state and busy() only, never a call: the commander's call
        # lock may be held by a goal for many seconds.
        return None if self._arm is None else self._arm.last_state

    def _arms_known(self) -> frozenset[str]:
        state = self._arm_state()
        if state is None:
            return frozenset()
        return frozenset(arm for arm in ARMS if state.arm(arm) is not None)

    def _arm_stop_reason(self, arms_known: frozenset[str]) -> str | None:
        """ARM_OUT, ARM_MOTION or None, from the commander's last reading and busy().

        ARM_OUT when an arm is known to be out of every travel-safe pose.
        ARM_MOTION when an arm known when the target was set is unknown now,
        which happens when a goal for it is sent, and when a commander goal
        runs, or may still run (busy()), while an arm is unknown: an arm
        goal makes its arm unknown before it is sent, so this also catches a
        motion of an arm that was unknown from the start. Every arm atomic
        stops the loop first, so these only catch a motion that did not. A
        gripper goal with both arms known does not stop it: the grippers do
        not move the arms. Arms unknown from the start with no goal running,
        as without a commander, do not stop it: the check fails open, as
        navigation's does (ADR-0025).
        """
        state = self._arm_state()
        if state is None:
            return None
        if state.unsafe_arms:
            return ARM_OUT
        unknown = {arm for arm in ARMS if state.arm(arm) is None}
        if unknown & arms_known:
            return ARM_MOTION
        # busy() never waits: it reads the client's state under its short
        # state lock, never the call lock or the executor.
        if unknown and self._arm.busy() is not None:
            return ARM_MOTION
        return None

    # -- Callbacks, on the executor thread -------------------------------------------

    def _on_people(self, frame: PeopleFrame) -> None:
        # Keeps the array and nothing more. The TF lookup is the gaze
        # thread's: a TF wait here would hold up the TF messages it waits for.
        stamp = _stamp_seconds(frame.msg.header.stamp)
        with self._lock:
            target = self._target
            if not self._control.running or target is None or target.track is None:
                return
            if target.track not in frame.tracks:
                return
            last_seen = self._control.last_seen
            if last_seen is not None and stamp <= last_seen:
                return
            index = frame.tracks.index(target.track)
            self._pending.append(_Pending(frame.msg, index, stamp, time.monotonic()))
            del self._pending[:-_PENDING_KEPT]

    def _on_navigation_status(self, msg: GoalStatusArray) -> None:
        # Every message lists all the goals the server still keeps, so the
        # newest one alone says whether any is active.
        try:
            active = any(status.status in _ACTIVE_GOAL for status in msg.status_list)
            with self._lock:
                self._navigation_active = active
                if active:
                    self._stop_locked(NAVIGATION_ACTIVE)
        except Exception:
            # Never raised into the executor, where it would end the
            # executor's thread and every subscription of the node.
            logger.exception("The gaze loop could not read a navigation status.")

    # -- Helpers -------------------------------------------------------------------------

    def _current_sighting(
        self, track: TrackKey
    ) -> tuple[tuple[float, float] | None, str | None]:
        """`track`'s (heading, stamp) in the newest array and None, or None and why there is none.

        The newest array counts only when it is no older than lost_after_s
        by the node clock. On the caller's thread, so it may wait for TF as
        find_people does.
        """
        assert self._vision is not None
        lost_after_s = self.config.lost_after_s
        try:
            frame = self._vision.latest_people(lost_after_s, 0.0, _NEVER_SET)
        except StepError:
            return None, f"no people array came in the last {lost_after_s:g} s"
        if track not in frame.tracks:
            return None, "the newest people array does not show them"
        msg = frame.msg
        transform = camera_transform(msg.header, self._tf_buffer, self.config.tf_wait_s)
        if transform is not None:
            person = msg.people[frame.tracks.index(track)]
            heading = ray_heading(_target_box(person), msg.camera_info, transform)
            if heading is not None:
                return (heading, _stamp_seconds(msg.header.stamp)), None
        return None, "the newest people array gives no heading toward them"

    def _node_seconds(self) -> float:
        return self._node.get_clock().now().nanoseconds / 1e9


def _target_box(person: PersonDetection) -> BoundingBox2D:
    # The face when there is one, as find_people's heading_map: the loop
    # looks at the face. face_det_score is 0 when no face was found.
    return person.face_bbox if person.face_det_score > 0 else person.bbox


def _stamp_seconds(stamp: TimeMsg) -> float:
    # From the nanoseconds, as a handle's stamp_ns / 1e9: sec + nanosec / 1e9
    # can round one bit lower, and the same sighting would then look older
    # than itself.
    return (stamp.sec * 1_000_000_000 + stamp.nanosec) / 1e9


def _by_stamp(frame: _Pending) -> float:
    return frame.stamp
