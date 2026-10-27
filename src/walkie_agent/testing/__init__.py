"""In-memory fakes of the capabilities on ctx, for atomic tests (capabilities.md, ADR-0052).

fake_ctx() returns a real ToolContext whose six capabilities are fakes, and
a FakeRobot to script and inspect them. The fakes share the real code where
it decides an answer: the arm check is navigation.check_arms, a pose is
found by _arm.offered_pose, a gaze target by _gaze.look_target, and the
handles are stored by _perception.put_handles, so a fake refuses what the
ROS side refuses, with the same message. robot.calls logs every capability
call an atomic makes, such as "base.go_to"; each fake's `steps` logs the
safety steps that the ROS side can also observe, such as "gaze.stop" and
"arms.goal". Every wait is on a FakeClock, so a timeout costs no real time.

Test code only: create_walkie_agent never registers it. It imports rclpy,
since ToolContext does, but creates no ROS entity: the ctx's node is a
FakeNode, which refuses to make one.
"""

import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geometry_msgs.msg import TransformStamped
from rclpy.clock import ClockType
from rclpy.exceptions import ParameterNotDeclaredException
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import LookupException
from tf_transformations import quaternion_from_euler

from walkie_agent.arm.commander import (
    ABANDON_AFTER_S,
    NO_READING,
    DriveCheck,
    unverified_refusal,
)
from walkie_agent.arm.poses import (
    ARM_GROUPS,
    ARMS,
    BOTH,
    BOTH_ARMS_GROUP,
    GRASP_CLOSE_THRESHOLD,
    GRASP_DETECT_MAX,
    GRASP_DETECT_MIN,
    GRIPPER_GROUPS,
    ArmPose,
    ArmPoses,
    load_arm_poses,
)
from walkie_agent.arm.state import MATCH_TOLERANCE_RAD, ArmsState, worst_joint
from walkie_agent.capabilities import (
    Arm,
    ArmsMoved,
    ArmsReport,
    GazeStatus,
    Gripper,
    GripperMoved,
    GripperResult,
    GripperState,
    Looking,
    Moved,
    ObjectSeen,
    PersonSeen,
    Pose2D,
    Recovery,
)
from walkie_agent.gaze import (
    ARM_MOTION,
    DEFAULT_CONFIG,
    HEADING,
    INTERRUPT,
    LOST,
    NAVIGATION,
    PERSON,
    STOP_LOOKING,
    wrap_angle,
)
from walkie_agent.locations import Location
from walkie_agent.perception.vision_feed import (
    OBJECTS_TOPIC,
    PEOPLE_TOPIC,
    TrackKey,
    model_vector,
    stale_message,
)
from walkie_agent.toolbox import HandleSet, StepError, ToolContext
from walkie_agent.toolbox.context import CAPABILITIES
from walkie_agent.tools.atomic import _arm, _gaze, _perception
from walkie_agent.tools.navigation import MAP_FRAME, ROBOT_FRAME, check_arms, tf_publisher_hint

# The repository's config/, for the default arm poses: load_arm_poses's own
# default is relative to the working directory.
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

# fake_ctx's `arms` value for both arms in no named pose.
OUT = "out"

# What tf2 says when its buffer has no map frame, as with no TF at all.
_NO_TF = f'"{MAP_FRAME}" passed to lookupTransform argument target_frame does not exist. '
_POSITION = f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME})"

# The base's limit for one move_relative, as the move tools have it.
MAX_MOVE_M = 3.0

# The fake camera: its frame, its image and a box for each third of it.
CAMERA_FRAME = "fake_camera_optical_frame"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
_BOXES = {
    "left": (40.0, 80.0, 180.0, 460.0),
    "centre": (250.0, 80.0, 390.0, 460.0),
    "right": (460.0, 80.0, 600.0, 460.0),
}
_FACES = {
    "left": (80.0, 90.0, 140.0, 150.0),
    "centre": (290.0, 90.0, 350.0, 150.0),
    "right": (500.0, 90.0, 560.0, 150.0),
}
_RANK = {"left": 0, "centre": 1, "right": 2}

# What a base goal's cancel answer means, by FakeBase's outcome.
_BASE_CANCEL_WORDS = {
    "accepted": "Nav2 accepted the cancel",
    "refused": "Nav2 refused the cancel, so the base may still be moving",
    "no answer": "Nav2 did not answer the cancel, so the base may still be moving",
}


class FakeClock:
    """A clock that moves only when a wait times out or a test calls advance().

    wait() returns at once: True when the event is set, else after moving
    the clock by the timeout, so ctx.sleep(60) takes no real time and leaves
    the clock 60 s later. The ctx's node clock reads it too. Safe from any
    thread.
    """

    def __init__(self, start_s: float = 1000.0) -> None:
        self._lock = threading.Lock()
        self._now = float(start_s)

    def monotonic(self) -> float:
        with self._lock:
            return self._now

    def now_ns(self) -> int:
        """monotonic() in nanoseconds, the node clock's time."""
        return int(round(self.monotonic() * 1e9))

    def advance(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"seconds must be a finite number of seconds, not {seconds}.")
        with self._lock:
            self._now += seconds

    def wait(self, event: threading.Event, timeout_s: float) -> bool:
        if event.is_set():
            return True
        self.advance(max(0.0, timeout_s))
        return event.is_set()


class _NodeClock:
    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    def now(self) -> Time:
        return Time(nanoseconds=self._clock.now_ns(), clock_type=ClockType.ROS_TIME)


class FakeNode:
    """The node of a fake_ctx: its clock, use_sim_time and TF publishers; no entities."""

    def __init__(self, robot: "FakeRobot") -> None:
        self._robot = robot
        self._clock = _NodeClock(robot.clock)

    def get_name(self) -> str:
        return "fake_ctx"

    def get_clock(self) -> _NodeClock:
        return self._clock

    def get_parameter(self, name: str) -> Parameter:
        if name != "use_sim_time":
            raise ParameterNotDeclaredException(name)
        return Parameter("use_sim_time", value=self._robot.in_simulation)

    def count_publishers(self, topic: str) -> int:
        """1 for /tf and /tf_static while the fake TF is up, so tf_publisher_hint() works."""
        return int(topic in ("/tf", "/tf_static") and self._robot.pose.tf)

    def resolve_topic_name(self, name: str, *, only_expand: bool = False) -> str:
        return name

    def resolve_service_name(self, name: str, *, only_expand: bool = False) -> str:
        return name

    def create_publisher(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fake_ctx has no ROS entities.")

    def create_subscription(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fake_ctx has no ROS entities.")

    def create_client(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fake_ctx has no ROS entities.")

    def create_service(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fake_ctx has no ROS entities.")

    def create_timer(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fake_ctx has no ROS entities.")


class FakeConnector:
    """The connector of a fake_ctx: a FakeNode, and the base's pose as TF gives it.

    It has no TF buffer, so ctx.tf_buffer is None: an atomic on fake_ctx
    reads TF through ctx.pose and ctx.robot_pose() only.
    """

    def __init__(self, robot: "FakeRobot") -> None:
        self.node = FakeNode(robot)
        self._robot = robot
        self._tf_buffer = None

    def get_transform(
        self, target_frame: str, source_frame: str, timeout_sec: float = 5.0
    ) -> TransformStamped:
        """map -> base_footprint from FakePose; raises LookupException while TF is down."""
        return self._robot.pose._transform()


class FakePose:
    """ctx.pose: the base's pose as a test sets it, with RosPose's errors.

    `tf` False is a TF without map -> base_footprint, `age_s` the age of
    the newest transform, `updating` False a TF that sends no newer one, and
    `fresh_delay_s` how long fresh() waits for one stamped after the call.
    `history` holds every pose with the stamp it began at, for at().
    """

    def __init__(self, robot: "FakeRobot", x: float, y: float, yaw: float) -> None:
        self._robot = robot
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.age_s = 0.0
        self.tf = True
        # One TF period at 50 Hz: as on the robot, a transform stamped after
        # the call is never there at once, so a set cancel ends fresh().
        self.fresh_delay_s = 0.02
        # The first entry is from stamp 0, so at() knows the pose at every
        # stamp before the first teleport.
        self.history: list[tuple[int, Pose2D]] = [(0, self.current())]
        # The newest transform's stamp once updating was set False.
        self._frozen_ns: int | None = None

    @property
    def updating(self) -> bool:
        return self._frozen_ns is None

    @updating.setter
    def updating(self, value: bool) -> None:
        if value:
            self._frozen_ns = None
        elif self._frozen_ns is None:
            self._frozen_ns = self._newest_ns()

    def current(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    def teleport(
        self, x: float | None = None, y: float | None = None, yaw: float | None = None
    ) -> None:
        """Put the base at a new pose now; a value left out keeps its value."""
        self.x = self.x if x is None else float(x)
        self.y = self.y if y is None else float(y)
        self.yaw = self.yaw if yaw is None else float(yaw)
        self.history.append((self._robot.clock.now_ns(), self.current()))

    def latest(self, *, max_age_s: float) -> Pose2D:
        self._robot._call("pose.latest")
        if not self.tf:
            raise self._unknown()
        age = (self._robot.clock.now_ns() - self._newest_ns()) / 1e9
        if age > max_age_s:
            raise StepError(
                "STALE",
                f"{_POSITION} is {age:.1f} s old, more than {max_age_s:g} s.",
                {"age_s": age},
            )
        return self.current()

    def fresh(self, *, timeout_s: float = 1.0) -> Pose2D:
        self._robot._call("pose.fresh")
        return self._fresh(timeout_s)

    def at(self, stamp_ns: int, *, timeout_s: float) -> Pose2D:
        self._robot._call("pose.at")
        ctx = self._robot.ctx
        if stamp_ns <= 0:
            raise StepError(
                "UNAVAILABLE",
                f"{_POSITION} at that stamp is unknown: {stamp_ns} is not a stamp.",
            )
        try:
            ctx.wait_until(lambda: self.tf and self._newest_ns() >= stamp_ns, timeout_s)
        except StepError as e:
            if e.code != "TIMEOUT":
                raise
        hint = tf_publisher_hint(ctx.node)
        if not self.tf:
            raise StepError("UNAVAILABLE", f"{_POSITION} at that stamp is unknown. {_NO_TF}{hint}")
        newest = self._newest_ns()
        if stamp_ns > newest:
            raise StepError(
                "UNAVAILABLE",
                f"{_POSITION} at that stamp is unknown. Lookup would require extrapolation "
                f"into the future. Requested time {stamp_ns / 1e9:.6f} but the latest data is "
                f"at time {newest / 1e9:.6f}, when looking up transform from frame "
                f"[{ROBOT_FRAME}] to frame [{MAP_FRAME}]{hint}",
            )
        return [pose for stamp, pose in self.history if stamp <= stamp_ns][-1]

    def _fresh(self, timeout_s: float) -> Pose2D:
        """fresh() without the call log, for the base's relative moves."""
        robot = self._robot
        ctx = robot.ctx
        ready_at = robot.clock.monotonic() + self.fresh_delay_s
        try:
            ctx.wait_until(
                lambda: self.tf and self.updating and robot.clock.monotonic() >= ready_at,
                timeout_s,
            )
        except StepError as e:
            if e.code != "TIMEOUT":
                raise
            if not self.tf:
                raise self._unknown() from None
            raise StepError(
                "UNAVAILABLE",
                f"{_POSITION} is out of date: no transform stamped after the call came within "
                f"{timeout_s:g} s. TF is not updating, or the agent and the robot disagree on "
                f"sim time (see --use-sim-time).{tf_publisher_hint(ctx.node)}",
            ) from None
        return self.current()

    def _unknown(self) -> StepError:
        hint = tf_publisher_hint(self._robot.ctx.node)
        return StepError("UNAVAILABLE", f"{_POSITION} is unknown. {_NO_TF}{hint}")

    def _newest_ns(self) -> int:
        if self._frozen_ns is not None:
            return self._frozen_ns
        return self._robot.clock.now_ns() - int(round(self.age_s * 1e9))

    def _transform(self) -> TransformStamped:
        if not self.tf:
            raise LookupException(_NO_TF)
        stamped = TransformStamped()
        stamped.header.stamp = Time(nanoseconds=self._newest_ns()).to_msg()
        stamped.header.frame_id = MAP_FRAME
        stamped.child_frame_id = ROBOT_FRAME
        stamped.transform.translation.x = self.x
        stamped.transform.translation.y = self.y
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.yaw)
        rotation = stamped.transform.rotation
        rotation.x, rotation.y, rotation.z, rotation.w = float(qx), float(qy), float(qz), float(qw)
        return stamped


class FakeBase:
    """ctx.base: drives that end on the FakeClock, in capabilities.md's safety order.

    Each verb validates, stops the gaze loop ("gaze.stop"), refuses while
    a goal of its own runs (PRECONDITION_FAILED), cancels the goals of
    other clients ("base.cancel_others"; `other_goals` counts them, and
    `others_cancelled` the ones it cancelled), checks the arms with
    navigation.check_arms ("arms.check"), reads a fresh pose for a
    relative move or a turn ("pose.fresh"), then sends one goal
    ("base.goal"), which ends after `drive_s`. Faults: `reject` (REFUSED),
    `abort` (SKILL_FAILED, with `abort_error_code` and `abort_error_msg`
    as Nav2's result gives them), a `drive_s` past the timeout (TIMEOUT),
    and `cancel_refused` and `cancel_unanswered` for the cancel's answer.
    on_goal() runs a callback once a goal is sent, as a stop mid-drive.
    """

    def __init__(self, robot: "FakeRobot", locations: Mapping[str, Location]) -> None:
        self._robot = robot
        self._locations = dict(locations)
        self.steps: list[str] = []
        # Every goal sent, rejected ones included.
        self.goals: list[Pose2D] = []
        # The cancels of the agent's own goal.
        self.cancels = 0
        # Other clients' active goals, which each verb cancels.
        self.other_goals = 0
        self.others_cancelled = 0
        self.reject = False
        self.abort = False
        self.abort_error_code = 0
        self.abort_error_msg = ""
        self.drive_s = 1.0
        self.cancel_refused = False
        self.cancel_unanswered = False
        self._on_goal: list[Callable[[], Any]] = []
        self._active = False
        # The drive that `_active` runs, such as "the drive to kitchen".
        self._active_what: str | None = None
        # Counts the goals that stop() cancelled, so a drive can tell.
        self._stopped = 0

    def on_goal(self, callback: Callable[[], Any]) -> None:
        """Call `callback()` each time a goal is sent, before the drive waits."""
        self._on_goal.append(callback)

    def locations(self) -> Mapping[str, Location]:
        self._robot._call("base.locations")
        return dict(self._locations)

    def go_to(self, location: str, *, timeout_s: float = 120.0) -> Moved:
        self._robot._call("base.go_to")
        found = self._locations.get(location.strip().lower())
        if found is None:
            known = ", ".join(self._locations) or "none"
            raise StepError(
                "NOT_FOUND",
                f"Unknown location '{location}'. Known locations: {known}.",
                {"locations": list(self._locations)},
            )
        goal = Pose2D(found.x, found.y, found.yaw)
        return self._drive(f"the drive to {found.name}", timeout_s, lambda _: goal, fresh=False)

    def move_relative(
        self,
        forward_m: float = 0.0,
        left_m: float = 0.0,
        turn_deg: float = 0.0,
        *,
        timeout_s: float = 120.0,
    ) -> Moved:
        self._robot._call("base.move_relative")
        _check_finite(forward_m, left_m, turn_deg)
        if forward_m == 0 and left_m == 0 and turn_deg == 0:
            raise StepError("REFUSED", "Nothing to do: the move is zero.")
        distance = math.hypot(forward_m, left_m)
        if distance > MAX_MOVE_M:
            raise StepError(
                "REFUSED",
                f"Refused: a move of {distance:.2f} m is more than the {MAX_MOVE_M:g} m allowed "
                "in one move.",
            )

        def target(start: Pose2D | None) -> Pose2D:
            assert start is not None
            cos, sin = math.cos(start.yaw), math.sin(start.yaw)
            return Pose2D(
                start.x + forward_m * cos - left_m * sin,
                start.y + forward_m * sin + left_m * cos,
                wrap_angle(start.yaw + math.radians(turn_deg)),
            )

        return self._drive("the move", timeout_s, target, fresh=True)

    def turn_by(self, degrees: float, *, timeout_s: float = 30.0) -> Moved:
        self._robot._call("base.turn_by")
        _check_finite(degrees)
        if degrees == 0:
            raise StepError("REFUSED", "Nothing to do: the turn is zero.")
        return self._drive(
            "the turn",
            timeout_s,
            lambda start: Pose2D(start.x, start.y, wrap_angle(start.yaw + math.radians(degrees))),
            fresh=True,
        )

    def turn_to(self, heading_rad: float, *, timeout_s: float = 30.0) -> Moved:
        self._robot._call("base.turn_to")
        _check_finite(heading_rad)
        return self._drive(
            "the turn",
            timeout_s,
            lambda start: Pose2D(start.x, start.y, wrap_angle(heading_rad)),
            fresh=True,
        )

    def stop(self) -> None:
        """Stop the gaze loop and cancel the active goal. Never raises."""
        self._robot._call("base.stop")
        if "gaze" not in self._robot.without:
            self._robot.gaze._stop(INTERRUPT)
        if self._active:
            self._stopped += 1
            self._cancel()

    def busy(self) -> bool:
        self._robot._call("base.busy")
        return self._active

    def _drive(
        self,
        what: str,
        timeout_s: float,
        target: Callable[[Pose2D | None], Pose2D],
        *,
        fresh: bool,
    ) -> Moved:
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError(f"timeout_s must be a positive number of seconds, not {timeout_s}.")
        robot = self._robot
        ctx = robot.ctx
        if "gaze" not in robot.without:
            self.steps.append("gaze.stop")
            robot.gaze._stop(NAVIGATION)
        if self._active:
            raise StepError(
                "PRECONDITION_FAILED",
                f"Cannot start {what} while {self._active_what} is running. Wait for it to "
                "finish.",
            )
        self.steps.append("base.cancel_others")
        self.others_cancelled += self.other_goals
        self.other_goals = 0
        arm = None if "arms" in robot.without else robot.arms
        if arm is not None:
            self.steps.append("arms.check")
        note = check_arms(arm, "driving", "'home'", ctx.cancel)
        notes = (note.removeprefix(" "),) if note else ()
        start = None
        if fresh:
            self.steps.append("pose.fresh")
            start = robot.pose._fresh(1.0)
        goal = target(start)
        ctx.check_cancelled()
        self.steps.append("base.goal")
        self.goals.append(goal)
        if self.reject:
            raise StepError("REFUSED", f"Nav2 rejected {what}.", {"status": "REJECTED"})
        self._active, self._active_what = True, what
        try:
            self._wait(what, timeout_s)
        finally:
            self._active, self._active_what = False, None
        if self.abort:
            detail = f" {self.abort_error_msg}" if self.abort_error_msg else ""
            raise StepError(
                "SKILL_FAILED",
                f"Nav2 aborted {what}.{detail}",
                {
                    "status": "ABORTED",
                    "error_code": self.abort_error_code,
                    "error_msg": self.abort_error_msg,
                },
            )
        robot.pose.teleport(goal.x, goal.y, goal.yaw)
        pose = robot.pose.current() if robot.pose.tf and robot.pose.updating else None
        return Moved(goal=goal, pose=pose, notes=notes)

    def _wait(self, what: str, timeout_s: float) -> None:
        """Wait out the goal just sent, which ends after `drive_s`, as a stop or a timeout allow."""
        ctx = self._robot.ctx
        stopped = self._stopped
        for callback in list(self._on_goal):
            callback()
        try:
            ctx.sleep(min(self.drive_s, timeout_s))
        except StepError as e:
            if e.code != "CANCELLED":
                raise
            outcome = self._cancel()
            raise StepError(
                "CANCELLED",
                f"A stop or a reset interrupted {what}; {_BASE_CANCEL_WORDS[outcome]}.",
                {"cancel": outcome},
            ) from None
        if self._stopped != stopped:
            raise StepError("CANCELLED", f"Nav2 cancelled {what}.", {"status": "CANCELED"})
        if self.drive_s > timeout_s:
            outcome = self._cancel()
            raise StepError(
                "TIMEOUT",
                f"{what[0].upper()}{what[1:]} did not finish within {timeout_s:g} s; "
                f"{_BASE_CANCEL_WORDS[outcome]}.",
                {"cancel": outcome},
            )

    def _cancel(self) -> str:
        """Cancel the active goal; the answer, a key of _BASE_CANCEL_WORDS.

        A goal whose cancel Nav2 did not accept stays active once the verb
        returns, as another goal, which the next verb cancels.
        """
        self.cancels += 1
        if self.cancel_unanswered or self.cancel_refused:
            self.other_goals += 1
            return "no answer" if self.cancel_unanswered else "refused"
        return "accepted"


class FakeSpeech:
    """ctx.speech: `said` holds every line; `available` False is no node on /to_human."""

    def __init__(self, robot: "FakeRobot") -> None:
        self._robot = robot
        self.said: list[str] = []
        self.available = True

    def say(self, text: str) -> None:
        self._robot._call("speech.say")
        if not text.strip():
            raise StepError("REFUSED", "Nothing was said: the text is blank.")
        # As the speech client's discovery wait, which ctx.cancel ends.
        self._robot.ctx.check_cancelled()
        if not self.available:
            raise StepError(
                "UNAVAILABLE", "Nothing was said: no speech node is listening on /to_human."
            )
        self.said.append(text)


class FakeArms:
    """ctx.arms: the arms' true poses, the client's memory of them, and scripted faults.

    `pose` is each arm's true pose name, or OUT for no named pose; `memory`
    is what the client last read, as CommanderClient.last_state. A move
    runs _arm.offered_pose, stops the gaze loop ("gaze.stop"), reads the
    group's joint order once, as CommanderClient.joint_names, and sends one
    goal ("arms.goal"); a gripper sends one goal ("gripper.goal") and never
    stops the loop. Faults, each read on every call: `plan_failure`,
    `execution_failure`, `reject_goals`, `never_answer` (TIMEOUT; the goal
    is then outstanding, and refuses every call, until it is cleared or
    ABANDON_AFTER_S passes on the FakeClock, when the arms are unverified
    until a read after it is cleared), `plan_only`, `read_failure` (the
    commander's status for a failed read), `unread` (NOT_FOUND, NO_ANSWER
    or NO_READING: the drive check cannot read the arms), `unmeasured` (a
    close the commander did not measure), `not_reached` (an open or "half"
    the gripper did not reach) and `holding`, by arm, for an object
    between the fingers, `object_width_m` wide, also by arm. A close is
    judged by where the fingers settle, as FakeCommander and RosArms judge
    it: "holding" or "empty" by the grasp band, and "not_closed" near the
    open position. `finger_m`, by arm, is each gripper's finger travel in
    metres: it starts at the open position, as FakeCommander's fingers do,
    a gripper goal moves it to where the fingers settle, except an open or
    "half" that is `not_reached`, and finger() reads it ("arms.finger"),
    failing as a read does.
    """

    def __init__(
        self,
        robot: "FakeRobot",
        poses: ArmPoses,
        arms: str | Mapping[str, str] | None,
    ) -> None:
        self._robot = robot
        self._poses = poses
        self.steps: list[str] = []
        # Every goal sent: (group, pose name) and (arm, gripper state).
        self.goals: list[tuple[str, str]] = []
        self.gripper_goals: list[tuple[str, str]] = []
        if arms is None or isinstance(arms, str):
            truth = {side: "home" if arms is None else arms for side in ARMS}
        else:
            truth = {side: arms.get(side, "home") for side in ARMS}
        for side, name in truth.items():
            if name != OUT and side not in getattr(poses.poses.get(name), "tables", {}):
                raise ValueError(
                    f"{name!r} is not a pose of the {side} arm in arm_poses.toml; use one of "
                    f"{', '.join(n for n, p in poses.poses.items() if side in p.tables)} or "
                    f"{OUT!r}."
                )
        self.pose: dict[str, str] = truth
        # None for arms that are unknown until the first read.
        self.memory = ArmsState() if arms is None else self._truth()
        self.unverified: str | None = None
        self.unverified_count = 0
        self.recovery: Recovery | None = None
        self.plan_failure = False
        self.execution_failure = False
        self.reject_goals = False
        self.never_answer = False
        self.plan_only = False
        self.read_failure: str | None = None
        self.unread: str | None = None
        self.unmeasured = False
        self.not_reached = False
        self.holding = {"left": False, "right": False}
        self.object_width_m = {"left": 0.02, "right": 0.02}
        self.finger_m = {side: poses.grippers.open_position for side in ARMS}
        # By arm, as the client counts each gripper group's commands.
        self._commands = {"left": 0, "right": 0}
        # The groups whose joint order the client has read; known arms were
        # read as both_arms.
        self._orders: set[str] = set() if arms is None else {BOTH_ARMS_GROUP}
        # The goal a call gave up on, and when, until its end is known.
        self._outstanding: tuple[str, float] | None = None
        self._abandoned: str | None = None

    # -- Arms ------------------------------------------------------------------

    def poses(self) -> Mapping[str, ArmPose]:
        self._robot._call("arms.poses")
        return self._offered()

    def state(self, *, read: bool | None = None) -> ArmsReport:
        self._robot._call("arms.state")
        return self._state(read)

    def move_to(self, pose: str, arm: Arm = "both") -> ArmsMoved:
        self._robot._call("arms.move_to")
        robot = self._robot
        ctx = robot.ctx
        name = _arm.pose_name(pose)
        found = _arm.offered_pose(
            self._offered(), pose, arm, lambda: _arm.state_sentence(self._state(None))
        )
        sides = ARMS if arm == BOTH else (arm,)
        group = BOTH_ARMS_GROUP if arm == BOTH else ARM_GROUPS[arm]
        prefix = f"Could not move to {name!r}: "
        if "gaze" not in robot.without:
            self.steps.append("gaze.stop")
            robot.gaze._stop(ARM_MOTION)
        if ctx.cancel.is_set():
            raise StepError(
                "CANCELLED", f"Stopped by reset before the arms moved. {self._memory_sentence()}"
            )
        try:
            if group not in self._orders:
                # As CommanderClient.joint_names: a read of the group, which
                # keeps its order and its arms' state.
                self._check_free(f"read the joints of {group}", sends=False)
                if self.read_failure is not None:
                    raise self._read_error(group)
                self._orders.add(group)
                truth = self._truth()
                self.memory = ArmsState(
                    left=truth.left if "left" in sides else self.memory.left,
                    right=truth.right if "right" in sides else self.memory.right,
                )
            self._check_free(f"move {group}", sends=True)
        except StepError as e:
            raise _arm.with_sentence(e, prefix, self._memory_sentence()) from None
        targets = {joint: value for side in sides for joint, value in found.tables[side].items()}
        self.steps.append("arms.goal")
        self.goals.append((group, name))
        if self.reject_goals:
            error = StepError(
                "INTERNAL",
                f"The commander rejected the goal for {group} with {len(targets)} positions. It "
                "rejects a count that differs from the group's joints; the joint order is read "
                "again on the next call.",
                {"group": group, "count": len(targets)},
            )
            raise self._move_error(name, error)
        # A goal sent leaves the moved arms unknown until the read after it.
        self.memory = ArmsState(
            left=None if "left" in sides else self.memory.left,
            right=None if "right" in sides else self.memory.right,
        )
        if self.never_answer:
            error = self._pending(f"the move of {group}", found.timeout_s, gripper=False)
            raise self._move_error(name, error)
        if self.plan_failure:
            error = StepError(
                "SKILL_FAILED",
                "MoveIt found no path (Planning Failed). The arms did not move.",
                {"group": group, "status": "Planning Failed"},
            )
            raise self._move_error(name, error)
        if self.execution_failure:
            for side in sides:
                self.pose[side] = OUT
            error = StepError(
                "SKILL_FAILED",
                "The motion stopped before the target (Execution Failed). The arms may have "
                "stopped part way.",
                {"group": group, "status": "Execution Failed"},
            )
            raise self._move_error(name, error)
        status = _arm.PLAN_ONLY if self.plan_only else "Success"
        if not self.plan_only:
            for side in sides:
                self.pose[side] = name
        if self.read_failure is not None:
            # A move that cannot be checked counts as failed (ADR-0024); the
            # moved arms stay unknown.
            read_error = self._read_error(BOTH_ARMS_GROUP)
            raise _arm.unchecked(
                status,
                f"the arms could not be checked against {name!r}: "
                f"{_arm.full_stop(read_error.message)}",
                self._memory_sentence(),
                {"pose": name, "status": status, "read_error": read_error.code},
            )
        self._orders.add(BOTH_ARMS_GROUP)
        self.memory = self._truth()
        report = self._report()
        worst = worst_joint(targets, self._reading())
        assert worst is not None
        joint, error_rad = worst
        if error_rad > MATCH_TOLERANCE_RAD:
            raise _arm.unchecked(
                status,
                f"{joint} is {error_rad:.2f} rad from {name!r}.",
                _arm.state_sentence(report),
                {"pose": name, "status": status, "joint": joint, "error_rad": error_rad},
            )
        return ArmsMoved(pose=name, arm=arm, status=status, max_error_rad=error_rad, report=report)

    def gripper(self, arm: Gripper, state: GripperState) -> GripperMoved:
        self._robot._call("arms.gripper")
        ctx = self._robot.ctx
        group = GRIPPER_GROUPS[arm]
        prefix = f"Could not move the {arm} gripper: "
        position = self._poses.grippers.position(state)
        if ctx.cancel.is_set():
            raise StepError(
                "CANCELLED",
                f"Stopped by reset before the {arm} gripper moved. {self._memory_sentence()}",
            )
        try:
            self._check_free(f"move {group}", sends=True)
        except StepError as e:
            raise _arm.with_sentence(e, prefix, self._memory_sentence()) from None
        self.steps.append("gripper.goal")
        self.gripper_goals.append((arm, state))
        self._commands[arm] += 1
        command = self._commands[arm]
        if self.reject_goals:
            error = StepError(
                "INTERNAL", f"The commander rejected the goal for {group}.", {"group": group}
            )
            raise _arm.with_sentence(error, prefix, self._memory_sentence())
        if self.never_answer:
            error = self._pending(
                f"the {group} command", _arm.CONFIG.gripper_timeout_s, gripper=True
            )
            if error.code == "CANCELLED":
                stopped = _arm.interrupted(error, gripper=True)
                raise StepError(
                    "CANCELLED", f"{stopped} {self._memory_sentence()}", error.data
                )
            raise _arm.with_sentence(error, prefix, self._memory_sentence())
        # As FakeCommander: the fingers stop on an object wider than the
        # position, and the goal then fails.
        settled = max(position, self.object_width_m[arm]) if self.holding[arm] else position
        closed = position <= GRASP_CLOSE_THRESHOLD
        if state == "close" or not self.not_reached:
            # An open or "half" that the gripper did not reach leaves the
            # fingers where they were.
            self.finger_m[arm] = settled
        if state != "close":
            reached = settled == position and not self.not_reached
            result = GripperResult(reached, "Success" if reached else "Failed", None, None, command)
            outcome = "moved" if reached else "not_reached"
            return GripperMoved(arm=arm, state=state, outcome=outcome, result=result)
        if self.unmeasured:
            # The commander's answer to a close it did not measure, or one a
            # newer command superseded.
            result = GripperResult(False, "Failed", False, 0.0, command)
            return GripperMoved(arm=arm, state=state, outcome="unmeasured", result=result)
        reached = settled == position
        result = GripperResult(
            success=reached,
            status="Success" if reached else "Failed",
            grasped=(GRASP_DETECT_MIN <= settled <= GRASP_DETECT_MAX) if closed else None,
            gripper_gap=settled if closed else None,
            command=command,
        )
        if self.read_failure is not None:
            return GripperMoved(
                arm=arm,
                state=state,
                outcome="unread",
                result=result,
                read_error=self._read_error(group).message,
            )
        # The read finds the fingers where they settled, so the gap always
        # agrees with it, and `grasped` is the answer (_arm._holding).
        finger = settled
        if finger >= self._poses.grippers.open_position - _arm.CONFIG.holding_margin_m:
            return GripperMoved(
                arm=arm, state=state, outcome="not_closed", result=result, finger_m=finger
            )
        return GripperMoved(
            arm=arm,
            state=state,
            outcome="holding" if result.grasped else "empty",
            result=result,
            finger_m=finger,
        )

    def finger(self, arm: Gripper) -> float:
        self._robot._call("arms.finger")
        group = GRIPPER_GROUPS[arm]
        if self._robot.ctx.cancel.is_set():
            raise StepError("CANCELLED", "A stop or a reset interrupted the call.")
        self._check_free(f"read the joints of {group}", sends=False)
        if self.read_failure is not None:
            raise self._read_error(group)
        return self.finger_m[arm]

    # -- The arm client's drive check, which navigation.check_arms calls --------

    def check_for_driving(self, cancel: threading.Event, timeout_s: float = 3.0) -> DriveCheck:
        """As CommanderClient.check_for_driving(): the arms, or why they were not read."""
        if cancel.is_set():
            raise StepError("CANCELLED", "A stop or a reset interrupted the call.")
        self._check_free("check the arms for driving", sends=False)
        if self.unread is not None or self.read_failure is not None:
            if self.unverified is not None:
                raise self._refuse_unverified("The commander cannot read the arms.")
            return DriveCheck(None, self.unread or NO_READING)
        if not self._read():
            assert self.unverified is not None
            raise self._refuse_unverified(
                "The commander has not shown that the goal is no longer active."
            )
        return DriveCheck(self.memory)

    # -- Internals: no call log ---------------------------------------------------

    def _offered(self) -> dict[str, ArmPose]:
        return self._poses.offered_poses(sim=self._robot.ctx.in_simulation)

    def _state(self, read: bool | None) -> ArmsReport:
        if read is not False and (read or self.memory.travel_safe is None):
            self._read()
        return self._report()

    def _report(self) -> ArmsReport:
        return ArmsReport(
            arms=self.memory,
            unverified=self.unverified,
            unverified_count=self.unverified_count,
            recovery=self.recovery,
        )

    def _memory_sentence(self) -> str:
        return _arm.state_sentence(self._state(False))

    def _read(self) -> bool:
        """Read both arms into `memory`, recovering first when unverified; False on failure.

        Refused while a goal is outstanding. Recovery succeeds once
        `never_answer` is cleared: the commander then shows the abandoned
        goal to be no longer active.
        """
        if self._check_outstanding() is not None:
            return False
        if self.unread is not None or self.read_failure is not None:
            return False
        if self.unverified is not None:
            if self.never_answer:
                return False
            goals = () if self._abandoned is None else ((self._abandoned, "result"),)
            self.recovery = Recovery(count=self.unverified_count, goals=goals, grippers=())
            self.unverified = None
        self._orders.add(BOTH_ARMS_GROUP)
        self.memory = self._truth()
        return True

    def _reading(self) -> dict[str, float]:
        """Both arms' joints by the true poses; an arm OUT is 0.5 rad off its first pose."""
        reading = {}
        for side in ARMS:
            name = self.pose[side]
            if name == OUT:
                table = next(p.tables[side] for p in self._poses.poses.values() if side in p.tables)
                reading.update({joint: value + 0.5 for joint, value in table.items()})
            else:
                reading.update(self._poses.poses[name].tables[side])
        return reading

    def _truth(self) -> ArmsState:
        return ArmsState.from_joints(self._reading(), self._poses.poses)

    def _read_error(self, group: str) -> StepError:
        return StepError(
            "UNAVAILABLE",
            f"The commander could not read the joints of {group}: {self.read_failure}",
        )

    def _pending(self, what: str, timeout_s: float, *, gripper: bool) -> StepError:
        """Wait out a goal the commander never answers; it is then outstanding."""
        ctx = self._robot.ctx
        outcome = (
            "if the commander accepts it later, the gripper still makes that motion"
            if gripper
            else "if the commander accepts it later, it is cancelled at once"
        )
        try:
            ctx.sleep(timeout_s)
            error = StepError(
                "TIMEOUT",
                f"{what[0].upper()}{what[1:]} did not finish within {timeout_s:g} s; {outcome}.",
                {"cancel": "pending"},
            )
        except StepError as e:
            if e.code != "CANCELLED":
                raise
            error = StepError(
                "CANCELLED",
                f"A stop or a reset interrupted {what}; {outcome}.",
                {"cancel": "pending"},
            )
        self._outstanding = (what, self._robot.clock.monotonic())
        return error

    def _check_outstanding(self) -> str | None:
        """As CommanderClient._outstanding: the goal a call gave up on, until its end is known.

        The goal ends once `never_answer` is cleared. Once ABANDON_AFTER_S
        has passed on the FakeClock since its call gave up, it is abandoned
        instead, and the arms are unverified. The client's ABSENT_ABANDON_S
        is not modelled: the commander's action servers never go away here.
        """
        if self._outstanding is None:
            return None
        what, gave_up_at = self._outstanding
        if not self.never_answer:
            self._outstanding = None
            return None
        if self._robot.clock.monotonic() - gave_up_at < ABANDON_AFTER_S:
            return what
        self._outstanding = None
        self._abandoned = what
        self.unverified = f"{what} from an earlier call was given up on before its end was known"
        self.unverified_count += 1
        return None

    def _check_free(self, action: str, *, sends: bool) -> None:
        """As CommanderClient._check_free: refuse while a goal is outstanding.

        With `sends`, for a call that sends a goal, also while the arms are
        unverified.
        """
        what = self._check_outstanding()
        if what is not None:
            raise StepError(
                "PRECONDITION_FAILED",
                f"Cannot {action}: {what} from an earlier call has not ended yet. Try again in "
                "a few seconds.",
                {"outstanding": what},
            )
        if sends and self.unverified is not None:
            raise unverified_refusal(action, self.unverified)

    def _move_error(self, name: str, error: StepError) -> StepError:
        """As _arm._move_error: the answer to a move the commander did not finish as asked."""
        if error.code == "CANCELLED" and self._robot.ctx.cancel.is_set():
            stopped = (
                _arm.interrupted(error, gripper=False) or "Stopped by reset before the arms moved."
            )
            return StepError("CANCELLED", f"{stopped} {self._memory_sentence()}", error.data)
        if error.code in ("SKILL_FAILED", "INTERNAL", "CANCELLED"):
            sentence = _arm.state_sentence(self._state(True))
        else:
            sentence = self._memory_sentence()
        return _arm.with_sentence(error, f"Could not move to {name!r}: ", sentence)

    def _refuse_unverified(self, why: str) -> StepError:
        return StepError(
            "PRECONDITION_FAILED",
            f"An arm or a gripper may still be moving, as {self.unverified}, and the arms "
            f"cannot be checked now. {why} Try again in a few seconds; if it fails again, tell "
            "the person that the arms cannot be checked.",
            {"unverified": self.unverified},
        )


class FakeGaze:
    """ctx.gaze: a loop that takes a person or a seat handle and turns on the FakeClock.

    look_at() reads the stop generation, runs _gaze.look_target and
    navigation.check_arms ("arms.check"), then starts ("gaze.start"). A
    person starts from the newer of the handle's sighting and the one in
    FakeVision's people array, which counts only when no older than
    `lost_after_s` and showing the handle's track; a handle's sighting
    older than that is STALE, as GazeLoop.follow_person. Every stop call,
    from any caller, changes the generation, so a stop during the arm check
    makes look_at CANCELLED, as on the ROS side. face() waits `turn_s` and
    then faces the target, turning FakePose's yaw to it, unless `faces` is
    False. Faults: `navigation_active` (REFUSED), `lost` (STALE, the person
    last seen `lost_age_s` ago) and `closed` (UNAVAILABLE). `stops` holds
    every stop's reason, from any caller.
    """

    def __init__(self, robot: "FakeRobot") -> None:
        self._robot = robot
        self.steps: list[str] = []
        self.stops: list[str] = []
        # The stop generation and the reason of the last stop call, as
        # GazeLoop.stop_calls().
        self._stop_calls = 0
        self._last_stop_call: str | None = None
        self.faces = True
        self.turn_s = 1.0
        self.navigation_active = False
        self.lost = False
        self.lost_age_s = 2.0
        self.closed = False
        self.lost_after_s = DEFAULT_CONFIG.lost_after_s
        self.target: str | None = None
        self.kind: str | None = None
        self.running = False
        self.setpoint: float | None = None
        self.last_seen: float | None = None
        self.stop_reason: str | None = None
        self.stopped_at: float | None = None

    def look_at(self, handle: str) -> Looking:
        self._robot._call("gaze.look_at")
        robot = self._robot
        ctx = robot.ctx
        # First, as RosGaze: a stop during the arm check then wins.
        stop_calls = self._stop_calls
        target = _gaze.look_target(ctx, handle)
        payload = target.payload
        arm = None if "arms" in robot.without else robot.arms
        if arm is not None:
            self.steps.append("arms.check")
        note = check_arms(
            arm, f"starting a turn toward {target.id}", "'carry' or 'home'", ctx.cancel
        )
        notes = (note.removeprefix(" "),) if note else ()
        person = target.kind == "person"
        if person and "vision" in robot.without:
            raise StepError("UNAVAILABLE", "The vision backend is not available.")
        if self.closed:
            raise StepError("UNAVAILABLE", "The gaze loop has shut down.")
        if self.navigation_active:
            raise StepError(
                "REFUSED",
                f"Not turning toward {target.id}: a navigation goal is active.",
                {"handle": target.id},
            )
        if person:
            setpoint, seen = self._sighting(target.id, payload)
        else:
            setpoint, seen = payload.heading_map, None
        ctx.check_cancelled()
        if self._stop_calls != stop_calls:
            reason = self._last_stop_call
            raise StepError(
                "CANCELLED",
                f"Not turning toward {target.id}: the gaze loop was stopped ({reason}) during "
                f"the call. Call look_at again if the robot should still look at {target.id}.",
                {"handle": target.id, "reason": reason},
            )
        self.steps.append("gaze.start")
        self.target = target.id
        self.kind = PERSON if person else HEADING
        self.running = True
        self.setpoint = setpoint
        self.last_seen = seen
        self.stop_reason = self.stopped_at = None
        if not person:
            return Looking(
                target=target.id,
                kind="seat",
                notes=notes,
                lost_after_s=None,
                seat_label=payload.label,
                seat_part=payload.part,
            )
        if self.lost:
            self._stop(LOST)
            raise _gaze.lost_at_start(target.id, self.lost_age_s, self.lost_after_s)
        return Looking(
            target=target.id,
            kind="person",
            notes=notes,
            lost_after_s=self.lost_after_s,
            seat_label=None,
            seat_part=None,
        )

    def face(self, looking: Looking, timeout_s: float) -> bool:
        self._robot._call("gaze.face")
        robot = self._robot
        ctx = robot.ctx
        self._check_turning(looking)
        turn_s = self.turn_s if self.faces else math.inf
        try:
            ctx.sleep(min(turn_s, timeout_s))
        except StepError:
            self._stop(INTERRUPT)
            raise
        self._check_turning(looking)
        if turn_s > timeout_s:
            return False
        assert self.setpoint is not None
        robot.pose.teleport(yaw=self.setpoint)
        return True

    def stop(self, reason: str | None = None) -> str | None:
        self._robot._call("gaze.stop")
        return self._stop(reason or STOP_LOOKING)

    def status(self) -> GazeStatus:
        self._robot._call("gaze.status")
        return self._status()

    def _stop(self, reason: str, *, call: bool = True) -> str | None:
        """Stop for `reason`, without the call log; the target stopped, or None.

        `call` False is the loop's own stop, as GazeLoop's tick makes for a
        person lost, which leaves the stop generation as it is.
        """
        self.stops.append(reason)
        if call:
            self._stop_calls += 1
            self._last_stop_call = reason
        if not self.running:
            return None
        self.running = False
        self.stop_reason = reason
        self.stopped_at = self._robot.clock.monotonic()
        return self.target

    def _status(self) -> GazeStatus:
        return GazeStatus(
            target=self.target,
            kind=self.kind,
            running=self.running,
            setpoint=self.setpoint,
            last_seen=self.last_seen,
            stop_reason=self.stop_reason,
            stopped_at=self.stopped_at,
            navigation_active=self.navigation_active,
            now=self._robot.clock.monotonic(),
        )

    def _check_turning(self, looking: Looking) -> None:
        """As RosGaze.face's checks: CANCELLED, STALE for a person lost, or the loop's end."""
        ctx = self._robot.ctx
        name = looking.target
        if ctx.cancel.is_set():
            self._stop(INTERRUPT)
            ctx.check_cancelled()
        if looking.kind == "person" and self.running and self.target == name and self.lost:
            self.last_seen = self._robot.clock.monotonic() - self.lost_age_s
            self._stop(LOST, call=False)
        status = self._status()
        if status.running and status.target == name:
            return
        if (
            looking.kind == "person"
            and status.target == name
            and status.stop_reason == LOST
            and status.last_seen is not None
        ):
            raise _gaze.lost_at_start(name, status.now - status.last_seen, self.lost_after_s)
        raise _gaze.turn_ended(name, status)

    def _sighting(self, name: str, payload: Any) -> tuple[float, float]:
        """As GazeLoop.follow_person: the (heading, stamp) to start from, the newer one.

        The newest array is FakeVision's people, stamped `age_s` ago, which
        counts only when no older than `lost_after_s` and showing the
        handle's track.
        """
        robot = self._robot
        vision = robot.vision
        lost_after_s = self.lost_after_s
        current: tuple[float, float] | None = None
        shown = vision._people
        if shown is None or not vision.available or vision.age_s > lost_after_s:
            unseen: str | None = f"no people array came in the last {lost_after_s:g} s"
        else:
            match = next(
                (
                    seen
                    for seen in shown
                    if seen.track is not None and TrackKey(seen.track, 1) == payload.track
                ),
                None,
            )
            if match is None:
                unseen = "the newest people array does not show them"
            elif match.heading_map is None:
                unseen = "the newest people array gives no heading toward them"
            else:
                current = (match.heading_map, robot.clock.monotonic() - vision.age_s)
                unseen = None
        stored = (
            None if payload.heading_map is None else (payload.heading_map, payload.stamp_ns / 1e9)
        )
        if current is not None and (stored is None or current[1] >= stored[1]):
            return current
        if stored is not None:
            self._check_fresh(name, stored[1], unseen)
            return stored
        raise StepError(
            "UNAVAILABLE",
            f"No heading toward {name}: none was stored when they were seen, and they are not "
            "in view now. Call find_people again.",
            {"handle": name},
        )

    def _check_fresh(self, name: str, seen: float, unseen: str | None) -> None:
        """As GazeLoop._check_fresh: STALE when the handle's sighting at `seen` is already lost."""
        age = self._robot.clock.monotonic() - seen
        lost_after_s = self.lost_after_s
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


@dataclass(frozen=True)
class FakePerson:
    """One person in the fake camera's view, for FakeVision.show()."""

    position: str = "centre"
    heading_map: float | None = 0.0
    face_seen: bool = True
    gestures: frozenset[str] = frozenset()
    # The track id; None for an untracked person.
    track: int | None = 1


@dataclass(frozen=True)
class FakeObject:
    """One detection in the fake camera's view, for FakeVision.show()."""

    label: str
    score: float = 0.9
    position: str = "centre"
    heading_map: float | None = 0.0


class FakeVision:
    """ctx.vision: the people and objects a test shows, stored as real payloads and handles.

    Each array is stamped `age_s` before the call. Before show() gives an
    array, or with `available` False, a call waits CONFIG.wait_s on the
    FakeClock and raises UNAVAILABLE with the feed's message.
    """

    def __init__(self, robot: "FakeRobot") -> None:
        self._robot = robot
        self.age_s = 0.0
        self.available = True
        self._people: tuple[FakePerson, ...] | None = None
        self._objects: tuple[FakeObject, ...] | None = None

    def show(
        self,
        people: Sequence[FakePerson] | None = None,
        objects: Sequence[FakeObject] | None = None,
    ) -> None:
        """Set what the camera sees; an array left out stays as it was."""
        if people is not None:
            self._people = tuple(people)
        if objects is not None:
            self._objects = tuple(objects)

    def seat(
        self,
        heading_map: float | None = 0.0,
        label: str = "chair",
        part: str | None = None,
        occupied: bool = False,
        position: str = "centre",
    ) -> str:
        """Store one seat handle, as find_seats would; its id, for look_at."""
        header = self._header()
        payload = _perception.SeatPayload(
            position=position,
            box=_BOXES[position],
            label=label,
            part=part,
            occupied=occupied,
            header=header,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            heading_map=heading_map,
        )
        handle_set = self._put("seat", [payload], header)
        return handle_set.member_ids[0]

    def people(self, *, max_age_s: float) -> list[PersonSeen]:
        self._robot._call("vision.people")
        shown = self._newest(self._people, PEOPLE_TOPIC, max_age_s)
        if not shown:
            return []
        order = sorted(range(len(shown)), key=lambda i: (_RANK[shown[i].position], i))
        header = self._header()
        stamp_ns = Time.from_msg(header.stamp).nanoseconds
        payloads = [_person_payload(shown[i], header, stamp_ns) for i in order]
        handle_set = self._put("person", payloads, header)
        return [
            PersonSeen(
                handle=handle_id,
                position=payload.position,
                heading_map=payload.heading_map,
                face_seen=payload.face is not None,
                gestures=payload.gestures,
                stamp_ns=stamp_ns,
            )
            for handle_id, payload in zip(handle_set.member_ids, payloads, strict=True)
        ]

    def objects(self, labels: Sequence[str] = (), *, max_age_s: float) -> list[ObjectSeen]:
        self._robot._call("vision.objects")
        shown = self._newest(self._objects, OBJECTS_TOPIC, max_age_s)
        wanted = set(labels)
        chosen = [i for i, thing in enumerate(shown) if not wanted or thing.label in wanted]
        if not chosen:
            return []
        chosen.sort(key=lambda i: (_RANK[shown[i].position], i))
        header = self._header()
        payloads = [
            _perception.ObjectPayload(
                label=thing.label,
                score=thing.score,
                track_id=0,
                position=thing.position,
                box=_BOXES[thing.position],
                header=header,
                image_width=IMAGE_WIDTH,
                image_height=IMAGE_HEIGHT,
                heading_map=thing.heading_map,
            )
            for thing in (shown[i] for i in chosen)
        ]
        handle_set = self._put("object", payloads, header)
        stamp_ns = Time.from_msg(header.stamp).nanoseconds
        return [
            ObjectSeen(
                handle=handle_id,
                label=payload.label,
                score=payload.score,
                position=payload.position,
                heading_map=payload.heading_map,
                stamp_ns=stamp_ns,
            )
            for handle_id, payload in zip(handle_set.member_ids, payloads, strict=True)
        ]

    def _newest(self, shown: tuple[Any, ...] | None, topic: str, max_age_s: float) -> tuple:
        """The array shown, or UNAVAILABLE after the feed's wait when none is fresh enough."""
        ctx = self._robot.ctx
        ctx.check_cancelled()
        if shown is not None and self.available and self.age_s <= max_age_s:
            return shown
        wait_s = _perception.CONFIG.wait_s
        ctx.sleep(wait_s)
        age = self.age_s if shown is not None and self.available else None
        raise StepError("UNAVAILABLE", stale_message(topic, age, max_age_s, wait_s))

    def _header(self) -> Header:
        stamp_ns = self._robot.clock.now_ns() - int(round(self.age_s * 1e9))
        return Header(stamp=Time(nanoseconds=stamp_ns).to_msg(), frame_id=CAMERA_FRAME)

    def _put(self, kind: str, payloads: Sequence[object], header: Header) -> HandleSet:
        """put_handles with the base's pose, or capture_pose's UNAVAILABLE while TF is down."""
        robot = self._robot
        ctx = robot.ctx
        if not robot.pose.tf:
            ctx.check_cancelled()
            raise StepError(
                "UNAVAILABLE",
                f"{_POSITION} when the camera saw this is unknown, so nothing was stored. "
                f"{_NO_TF}{tf_publisher_hint(ctx.node)}",
            )
        return _perception.put_handles(ctx, kind, payloads, header, pose=robot.pose.current())


class FakeRobot:
    """What fake_ctx() returns beside the ctx: the fakes, their call log and their clock."""

    def __init__(
        self,
        *,
        locations: Mapping[str, Location],
        arms: str | Mapping[str, str] | None,
        arm_poses: ArmPoses,
        pose: tuple[float, float, float],
        in_simulation: bool,
        without: frozenset[str],
    ) -> None:
        self._lock = threading.Lock()
        # Every capability call, in order, such as "base.go_to".
        self.calls: list[str] = []
        self.clock = FakeClock()
        self.in_simulation = in_simulation
        # The capabilities the ctx lacks. The others skip their steps on
        # them, as the ROS side does without that backend.
        self.without = without
        self.pose = FakePose(self, *pose)
        self.base = FakeBase(self, locations)
        self.speech = FakeSpeech(self)
        self.arms = FakeArms(self, arm_poses, arms)
        self.gaze = FakeGaze(self)
        self.vision = FakeVision(self)
        # Set by fake_ctx() once the ctx exists.
        self.ctx: ToolContext = None  # type: ignore[assignment]

    def _call(self, name: str) -> None:
        with self._lock:
            self.calls.append(name)


def fake_ctx(
    locations: Mapping[str, tuple[float, float, float] | Location] | None = None,
    arms: str | Mapping[str, str] | None = "home",
    *,
    in_simulation: bool = False,
    without: Iterable[str] = (),
    arm_poses: ArmPoses | None = None,
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[ToolContext, FakeRobot]:
    """A real ToolContext on in-memory fakes of the six capabilities, and the FakeRobot.

    `locations` maps a name to (x, y, yaw) in the map, yaw in radians, or
    to a Location. `arms` is both arms' pose, as named in arm_poses.toml,
    a mapping by arm, OUT for no named pose, or None for arms the client has
    not read yet, which are then at "home". `without` names capabilities
    the ctx lacks: using one raises UNAVAILABLE. `arm_poses` defaults to the
    repository's config/arm_poses.toml.
    """
    without = frozenset(without)
    unknown = sorted(without - set(CAPABILITIES))
    if unknown:
        raise ValueError(f"Unknown capabilities {unknown}; a ctx has {', '.join(CAPABILITIES)}.")
    places = {
        name.strip().lower(): (
            place if isinstance(place, Location) else Location(name.strip().lower(), *place)
        )
        for name, place in (locations or {}).items()
    }
    if arm_poses is None:
        arm_poses = load_arm_poses(
            _CONFIG_DIR / "arm_poses.toml", _CONFIG_DIR / "robot_model.toml"
        )
    robot = FakeRobot(
        locations=places,
        arms=arms,
        arm_poses=arm_poses,
        pose=pose,
        in_simulation=in_simulation,
        without=without,
    )
    fakes = {
        "base": robot.base,
        "speech": robot.speech,
        "pose": robot.pose,
        "arms": robot.arms,
        "gaze": robot.gaze,
        "vision": robot.vision,
    }
    ctx = ToolContext(
        FakeConnector(robot),  # type: ignore[arg-type]
        capabilities={
            name: (lambda _ctx, fake=(None if name in without else fake): fake)
            for name, fake in fakes.items()
        },
        clock=robot.clock,
        in_simulation=in_simulation,
    )
    robot.ctx = ctx
    return ctx, robot


def _check_finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise StepError("REFUSED", "Refused: every value must be a finite number.")


def _person_payload(person: FakePerson, header: Header, stamp_ns: int) -> Any:
    return _perception.PersonPayload(
        track=None if person.track is None else TrackKey(person.track, 1),
        position=person.position,
        bbox=_BOXES[person.position],
        face_bbox=_FACES[person.position] if person.face_seen else None,
        keypoints=(0.0,) * 34,
        keypoint_scores=(0.0,) * 17,
        gestures=frozenset(person.gestures),
        face=model_vector("fake_face", [1.0, 0.0]) if person.face_seen else None,
        appearance=None,
        header=header,
        stamp_ns=stamp_ns,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        heading_map=person.heading_map,
    )
