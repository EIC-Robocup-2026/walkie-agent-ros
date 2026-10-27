"""The robot's capabilities on ctx: ctx.base, ctx.speech, ctx.pose, ctx.arms, ctx.gaze, ctx.vision.

Each capability is a Protocol here, with the records it returns
(capabilities.md, ADR-0052). An atomic calls them as ctx.arms.move_to("home")
and so on, and every verb is bound to ctx.cancel: a stop or a reset ends a
wait with StepError CANCELLED. A failure is a StepError, whose code each
Protocol names, with the goal status and the cancel answer in `data`.

This module imports no rclpy and nothing from walkie_agent.toolbox, whose
package imports rclpy, so a teammate can read and type against it without
ROS. The ROS implementations live beside the atomics they came from
(walkie_agent.toolbox.context.RosPose, tools.atomic._arm.RosArms,
_gaze.RosGaze and _perception.RosVision), and the fakes in
walkie_agent.testing. ctx.base and ctx.speech have only their Protocols and
fakes until PR B.
"""

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from walkie_agent.arm.poses import ArmPose
from walkie_agent.arm.state import ArmsState
from walkie_agent.locations import Location

# Which arm a pose move takes, and which gripper a gripper goal moves.
Arm = Literal["left", "right", "both"]
Gripper = Literal["left", "right"]
GripperState = Literal["open", "half", "close"]


@dataclass(frozen=True)
class Pose2D:
    """The base's pose in the map frame: metres, and yaw in radians."""

    x: float
    y: float
    yaw: float


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


@dataclass(frozen=True)
class GripperResult:
    """The commander's answer to one gripper goal."""

    # True when the controller reached the commanded position. A real grasp
    # usually ends false, since the fingers stall on the object; on a close,
    # `grasped` and the gap are the answer.
    success: bool
    status: str
    # Only for a close, a position at or below GRASP_CLOSE_THRESHOLD. On an
    # open or "half" the commander reports false and 0.0 whatever the
    # fingers did, so they are None here and cannot be misread. A close the
    # commander did not measure, with grasp_detect_enable off, superseded,
    # or with no finger reading, reports false and 0.0 too, which looks like
    # a measured empty close.
    grasped: bool | None
    # The settled travel of one finger, in metres.
    gripper_gap: float | None
    # The number of this command among the client's commands for its group,
    # for joint_states_since().
    command: int


@dataclass(frozen=True)
class Recovery:
    """What the recovery that last cleared the unverified state checked (last_recovery)."""

    # unverified_count when it cleared the state, which is also the count
    # when it began: a change in between keeps the state.
    count: int
    # (what, how) for each goal resolved since the arms became unverified,
    # such as ("the move of left_arm", "unknown"). `how` is "result" (its
    # result came), "rejected", "ended" (it answered a cancel that the goal
    # had already ended) or "unknown" (it does not know the goal, as after
    # a restart).
    goals: tuple[tuple[str, str], ...]
    # The gripper groups whose finger the settled read checked, besides
    # both arms.
    grippers: tuple[str, ...]


@dataclass(frozen=True)
class Moved:
    """What a base verb reached."""

    # The goal Nav2 reached.
    goal: Pose2D
    # From TF stamped after the result; None when none came within 1 s.
    pose: Pose2D | None
    # The arm check's notes, such as an arm state that was not read.
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ArmsReport:
    """What the arm client knows at one moment, from CommanderClient.snapshot()."""

    # The arms as of the last read.
    arms: ArmsState
    # Why the arms are unverified, or None.
    unverified: str | None
    # How many times the arms became unverified so far.
    unverified_count: int
    # What the recovery that last cleared the unverified state checked, else
    # None.
    recovery: Recovery | None


@dataclass(frozen=True)
class ArmsMoved:
    """A pose move the commander finished and the joint check confirmed."""

    # The pose's name as poses() spells it, such as "hands_up".
    pose: str
    arm: Arm
    # The commander's status, such as "Success" or "Plan Only".
    status: str
    # The largest joint error from the pose, in radians, as measured by the
    # read after the move.
    max_error_rad: float
    # The client's snapshot after the joint check.
    report: ArmsReport


@dataclass(frozen=True)
class GripperMoved:
    """One gripper goal and what the fingers did."""

    arm: Gripper
    state: GripperState
    # "moved" or "not_reached" for an open or "half". For a close:
    # "holding" or "empty" by the grasp band, "not_closed" when the fingers
    # did not close past the band, "unmeasured" when the commander did not
    # measure the close, "superseded" when a newer command took the gripper
    # before the read, and "unread" when the finger could not be read.
    outcome: Literal[
        "moved",
        "not_reached",
        "holding",
        "empty",
        "not_closed",
        "unmeasured",
        "superseded",
        "unread",
    ]
    result: GripperResult
    # The finger's settled travel from the read after a close, in metres;
    # None when it was not read.
    finger_m: float | None = None
    # Why the finger could not be read, for "unread".
    read_error: str | None = None

    @property
    def holding(self) -> bool | None:
        """True when the close holds something, False when it closed empty, else None."""
        if self.outcome == "holding":
            return True
        if self.outcome == "empty":
            return False
        return None


@dataclass(frozen=True)
class Looking:
    """A target the gaze loop has taken: a person it follows or a seat's heading it holds."""

    # The handle's id, such as per_51c0.1.
    target: str
    kind: Literal["person", "seat"]
    # The arm check's notes, such as an arm state that was not read: the
    # base turns with the arms unchecked.
    notes: tuple[str, ...]
    # For a person, how long after the last sighting the loop stops
    # following them, in seconds; None for a seat.
    lost_after_s: float | None
    # For a seat, the detector's class, such as "sofa", and the place on a
    # sofa, such as "left"; None for a person, and the part None for a
    # chair.
    seat_label: str | None
    seat_part: str | None


@dataclass(frozen=True)
class PersonSeen:
    """One person in the newest people array, stored as a person handle."""

    handle: str
    # The third of the image that holds the box's centre: "left", "centre"
    # or "right", as the robot sees it.
    position: str
    # The map-frame yaw of the ray toward the face, else the body, in
    # radians; None when TF had no camera transform at the stamp.
    heading_map: float | None
    # True when a face was found in the box and passed the feed's gate, so
    # the handle can be recognized.
    face_seen: bool
    # The gesture and posture names of walkie_agent.perception.gestures.
    gestures: frozenset[str]
    # The image's stamp, in nanoseconds.
    stamp_ns: int


@dataclass(frozen=True)
class ObjectSeen:
    """One detection in the newest objects array, stored as an object handle."""

    handle: str
    # The detector's class, such as "chair".
    label: str
    score: float
    position: str
    heading_map: float | None
    stamp_ns: int


class Base(Protocol):
    """ctx.base: drives to configured places and bounded moves, as one Nav2 goal each (effect N).

    Every verb runs the safety order of capabilities.md: validate, stop the
    agent's own loops, refuse while one of its own base goals runs, cancel
    other goals, check the arms, read a fresh pose for a relative move,
    then send one goal. Raises StepError NOT_FOUND for an unknown location;
    REFUSED for a value that is not finite, an empty move, more than 3 m,
    or a goal Nav2 rejects; PRECONDITION_FAILED while a base goal of its
    own runs, and PRECONDITION_FAILED, TIMEOUT or INTERNAL from the arm
    check; UNAVAILABLE without Nav2 or a fresh pose; SKILL_FAILED when Nav2
    aborts; TIMEOUT past `timeout_s`; CANCELLED on a stop or a reset.
    `data` holds the goal status, Nav2's error code and message on an
    abort, and the cancel's answer.
    """

    def locations(self) -> Mapping[str, Location]: ...

    def go_to(self, location: str, *, timeout_s: float = 120.0) -> Moved: ...

    def move_relative(
        self,
        forward_m: float = 0.0,
        left_m: float = 0.0,
        turn_deg: float = 0.0,
        *,
        timeout_s: float = 120.0,
    ) -> Moved: ...

    def turn_by(self, degrees: float, *, timeout_s: float = 30.0) -> Moved: ...

    def turn_to(self, heading_rad: float, *, timeout_s: float = 30.0) -> Moved: ...

    def stop(self) -> None:
        """Stop the agent's loops and cancel the base's goal. Never raises."""
        ...

    def busy(self) -> bool:
        """True while a base goal of the agent's is active."""
        ...


class Speech(Protocol):
    """ctx.speech: says one line on /to_human (effect S).

    say() returns once the line is published, not once it is spoken. Raises
    StepError REFUSED for blank text, UNAVAILABLE when nothing listens, and
    CANCELLED on a stop or a reset before it publishes.
    """

    def say(self, text: str) -> None: ...


class Pose(Protocol):
    """ctx.pose: the base's pose in the map, by a named staleness rule.

    latest() takes the newest transform and does not wait; fresh() waits
    for one stamped after the call; at() takes the one at an image's stamp.
    Raises StepError UNAVAILABLE, with the TF hint, when there is no such
    transform; STALE from latest() for one older than `max_age_s`; CANCELLED
    on a stop or a reset during a wait.
    """

    def latest(self, *, max_age_s: float) -> Pose2D: ...

    def fresh(self, *, timeout_s: float = 1.0) -> Pose2D: ...

    def at(self, stamp_ns: int, *, timeout_s: float) -> Pose2D: ...


class Arms(Protocol):
    """ctx.arms: named arm poses and the grippers, through the commander (effect M).

    poses() is the poses offered, in order: outside sim only those tried on
    the robot. state() never raises: `read` None reads the arms only when
    their state is unknown, True always reads them, False never does.

    move_to() stops the gaze loop first. It raises StepError NOT_FOUND for
    a pose that is unknown or not offered here; REFUSED for an arm the pose
    does not take, or a right-arm pose that needs openarm_right_joint2
    while walkie-ros holds it fixed; PRECONDITION_FAILED while another
    call's goal runs or is outstanding, or while the arms are unverified;
    SKILL_FAILED when the commander fails the goal; POSTCONDITION when the
    joints do not end in the pose or cannot be checked after a reported
    success; TIMEOUT when the gaze loop's stop is not confirmed or the goal
    does not finish in time; UNAVAILABLE without the commander or when it
    cannot read the joints;
    INTERNAL for a rejected goal or joints that do not match the robot
    model; CANCELLED on a stop or a reset.

    gripper() does not stop the gaze loop. It raises StepError
    PRECONDITION_FAILED while another call's goal runs or is outstanding,
    or while the arms are unverified; TIMEOUT; UNAVAILABLE; INTERNAL;
    CANCELLED. Every other end is an outcome, not an error: "moved",
    "not_reached", "holding", "empty", "not_closed", "unmeasured",
    "superseded" or "unread" (GripperMoved). The messages end with the arm
    sentence, as the arm atomics give it.

    finger() reads how far one gripper is open now, one finger's travel in
    metres, once, and sends no goal. It raises StepError as a read of the
    joints does: PRECONDITION_FAILED while another call's goal runs or is
    outstanding; TIMEOUT; UNAVAILABLE without the commander or when it
    cannot read the joints; INTERNAL for a reading without the finger
    joint; CANCELLED on a stop or a reset.
    """

    def poses(self) -> Mapping[str, ArmPose]: ...

    def state(self, *, read: bool | None = None) -> ArmsReport: ...

    def move_to(self, pose: str, arm: Arm = "both") -> ArmsMoved: ...

    def gripper(self, arm: Gripper, state: GripperState) -> GripperMoved: ...

    def finger(self, arm: Gripper) -> float: ...


class Gaze(Protocol):
    """ctx.gaze: turns the base toward a person or a seat handle, on the gaze loop (effect N).

    look_at() starts the loop and returns once it has the target; face()
    waits until the base faces it. look_at() raises StepError NOT_FOUND for
    a handle that is not a person or a seat; STALE for a person already
    lost; REFUSED for an untracked person or while Nav2 drives;
    UNAVAILABLE for a seat without a heading or a closed loop;
    PRECONDITION_FAILED, TIMEOUT or INTERNAL from the arm check; CANCELLED
    on a stop or a reset. face() raises CANCELLED, STALE for a person lost
    during the wait, and the loop's reason when it ended the turn. stop()
    and status() never raise.
    """

    def look_at(self, handle: str) -> Looking: ...

    def face(self, looking: Looking, timeout_s: float) -> bool: ...

    def stop(self, reason: str | None = None) -> str | None: ...

    def status(self) -> GazeStatus: ...


class Vision(Protocol):
    """ctx.vision: the newest vision arrays as records with handles (effect P).

    Each call stores what it returns as one handle set, people as "person"
    handles and detections as "object" handles, left to right. Nothing in
    view, or no detection with a label in `labels`, returns [] and stores
    nothing; empty `labels` takes every detection. Raises StepError
    UNAVAILABLE when no array as fresh as `max_age_s` comes, or TF has no
    base pose at the image's stamp; CANCELLED on a stop or a reset, which
    stores nothing.
    """

    def people(self, *, max_age_s: float) -> list[PersonSeen]: ...

    def objects(self, labels: Sequence[str] = (), *, max_age_s: float) -> list[ObjectSeen]: ...


class Clock(Protocol):
    """The clock ctx's waits use: ctx.wait(), sleep(), wait_until(), ctx.pose and handle ages."""

    def monotonic(self) -> float: ...

    def wait(self, event: threading.Event, timeout_s: float) -> bool:
        """Wait up to `timeout_s` for `event`; True when it is set."""
        ...


class SystemClock:
    """The real clock: time.monotonic() and Event.wait()."""

    def monotonic(self) -> float:
        # Looked up on every call, so a test that patches time.monotonic
        # applies.
        return time.monotonic()

    def wait(self, event: threading.Event, timeout_s: float) -> bool:
        return event.wait(timeout_s)
