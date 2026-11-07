"""The small frozen records that capability verbs return."""

from dataclasses import dataclass
from typing import Literal

from walkie_agent.arm.poses import GripperState
from walkie_agent.arm.results import GripperResult, Recovery
from walkie_agent.arm.state import UNKNOWN_SENTENCE, ArmsState

# Which arm a pose move takes, and which gripper a gripper verb takes.
Arm = Literal["left", "right", "both"]
Gripper = Literal["left", "right"]
# How one gripper goal ended (GripperMoved.outcome).
GripperOutcome = Literal[
    "moved",
    "not_reached",
    "holding",
    "empty",
    "not_closed",
    "unmeasured",
    "superseded",
    "unread",
]


@dataclass(frozen=True)
class Pose2D:
    """The base's pose in the map frame: meters, and yaw in radians."""

    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class ObjectSeen:
    """One object in view. A plain id that later calls can name (decision 4)."""

    id: str
    name: str
    category: str
    # The named place the object is at.
    place: str
    # None when the backend cannot measure it.
    size_cm: float | None = None
    weight_g: float | None = None
    thickness_cm: float | None = None


@dataclass(frozen=True)
class PersonSeen:
    """One person in view. `pose` is a posture, such as "sitting person".

    `bearing_deg` is where the person is from straight ahead of the base, in degrees,
    positive to the left as a turn is, and None when the backend cannot tell.
    """

    id: str
    # None until the person has been identified.
    name: str | None
    pose: str
    gesture: str
    room: str
    bearing_deg: float | None = None


@dataclass(frozen=True)
class Remembered:
    """What ctx.memory.remember_person stored for one person in view.

    `person_id` is the live id the call named, as find_people gives it; the memory's
    own id for the stored person never leaves the backend. `faces` and `appearances`
    count the samples stored. `unstored_models` names the models whose samples the
    memory has no table for and left out, so those models can never recognize them.
    """

    person_id: str
    name: str
    facts: dict[str, str]
    faces: int
    appearances: int
    unstored_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class Recognized:
    """Who one person in view is, from ctx.memory.recognize.

    `name` is None for someone Walkie has not remembered, or cannot recognize; `facts`
    is then empty and `matched_by` None. Otherwise `matched_by` is "face" or
    "appearance", whichever named them.
    """

    person_id: str
    name: str | None
    facts: dict[str, str]
    matched_by: str | None


@dataclass(frozen=True)
class Team:
    """The team that built the robot, as the robot says it."""

    name: str
    country: str
    affiliation: str


@dataclass(frozen=True)
class ArmsReport:
    """What the arm backend knows about the arms at one moment, without asking the commander.

    The fields come from one snapshot, so they agree with each other.
    """

    # The arms as of the last read.
    arms: ArmsState
    # Why the arms are unverified, or None. Unverified means a goal whose end is
    # unknown may still move an arm or a gripper, so the robot may not drive on the
    # strength of `arms`.
    unverified: str | None = None
    # How many times the arms became unverified so far.
    unverified_count: int = 0
    # What the recovery that last cleared the unverified state checked, else None.
    recovery: Recovery | None = None

    @property
    def travel_safe(self) -> bool | None:
        """Whether the robot may drive by this report: as `arms` says, but never True while unverified.

        None also while the arms are unverified and `arms` would say that they are
        safe, as while only a gripper goal made them unverified.
        """
        if self.arms.travel_safe and self.unverified is not None:
            return None
        return self.arms.travel_safe

    def sentence(self) -> str:
        """The arm sentence that every arm answer ends with (ADR-0024)."""
        return UNKNOWN_SENTENCE if self.travel_safe is None else self.arms.sentence()


@dataclass(frozen=True)
class ArmsMoved:
    """A pose move that the commander finished and the joint check after it confirmed."""

    # The pose's name as ctx.arms.poses() spells it, such as "hands_up".
    pose: str
    arm: Arm
    # The commander's status: "Success", or "Plan Only" when the commander runs in
    # plan-only mode and the arms were already in the pose.
    status: str
    # The largest joint error from the pose, in radians, by the read after the move.
    max_error_rad: float
    # The backend's report after the joint check.
    report: ArmsReport


@dataclass(frozen=True)
class GripperMoved:
    """One gripper goal and what the fingers did."""

    arm: Gripper
    state: GripperState
    # "moved" or "not_reached" for an open or "half". For a close: "holding" or
    # "empty" by the grasp band, "not_closed" when the fingers did not close past the
    # band, "unmeasured" when the commander did not measure the close, "superseded"
    # when a newer command took the gripper before the read, and "unread" when the
    # finger could not be read.
    outcome: GripperOutcome
    result: GripperResult
    # The finger's settled travel from the read after a close, in metres; None when
    # it was not read.
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
class GripperReading:
    """One gripper as ctx.arms.read() found it."""

    arm: Gripper
    # The travel of one finger read now, in metres; None when it could not be read.
    finger_m: float | None
    # The named position the reading counts as (arm.answers.reads_as); None when unread.
    reads_as: GripperState | None
    # When unread: why, and the state of the last command the agent sent the gripper,
    # or None when that is unknown too.
    read_error: str | None = None
    commanded: GripperState | None = None


@dataclass(frozen=True)
class ArmsReading:
    """What one ctx.arms.read() found: the arms before and after, and both grippers.

    `before` is the backend's report before the read and `after` its report after the
    last read, so a caller can tell a recovery that cleared the unverified state
    (`before.unverified` set, `after.recovery.count == before.unverified_count`) from
    arms that became unverified during the call (`after.unverified_count` changed).
    """

    before: ArmsReport
    after: ArmsReport
    # Left, then right.
    grippers: tuple[GripperReading, ...]
