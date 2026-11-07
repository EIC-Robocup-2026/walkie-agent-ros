"""The robot's capabilities on ctx: base, speech, vision, arms, facts and memory.

Each capability is a Protocol here. An atomic calls its verbs as
ctx.base.go_to("shelf") and never sees a topic or an action. A failure is a
StepError. The guards live inside the verbs, through capabilities.guards, so
whatever calls a verb gets them. There is no velocity verb.

This package imports no rclpy, rai or ROS message, so a teammate can read and
test against it without ROS. FakeRobot, in capabilities.fake, implements every
verb. The ROS-backed verbs live in walkie_agent.ros. The arm records build on
walkie_agent.arm, which imports no ROS either.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import NoReturn, Protocol

from walkie_agent.arm.poses import ArmPose
from walkie_agent.arm.state import ArmsState
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.capabilities.records import (
    Arm,
    ArmsMoved,
    ArmsReading,
    ArmsReport,
    Gripper,
    GripperMoved,
    GripperOutcome,
    GripperReading,
    GripperState,
    ObjectSeen,
    PersonSeen,
    Pose2D,
    Recognized,
    Remembered,
    Team,
)
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import Location

__all__ = [
    "NO_ARMS",
    "Arm",
    "ArmPose",
    "Arms",
    "ArmsMoved",
    "ArmsReading",
    "ArmsReport",
    "Base",
    "CancelToken",
    "Ctx",
    "Facts",
    "Gripper",
    "GripperMoved",
    "GripperOutcome",
    "GripperReading",
    "GripperState",
    "Memory",
    "ObjectSeen",
    "PersonSeen",
    "Pose2D",
    "Recognized",
    "Remembered",
    "Speech",
    "Team",
    "UnavailableArms",
    "UnavailableFacts",
    "UnavailableMemory",
    "Vision",
]


class Base(Protocol):
    """ctx.base: drives to named places and short relative moves.

    go_to and move_relative raise StepError NOT_FOUND for an unknown place;
    REFUSED for a value that is not finite, an empty move or one over the 3 m
    cap; CANCELLED on a stop; TIMEOUT past `timeout_s`, UNAVAILABLE without a
    navigation backend and SKILL_FAILED when the goal fails. That SKILL_FAILED
    is a GoalFailed when the backend reports that the goal was refused or ended
    without success. A plain StepError SKILL_FAILED is any other failure, such
    as a goal whose state is unknown or an unusable pose after a success. They
    also raise PRECONDITION_FAILED while another goal is live and, with an arm
    backend, while an arm is out of every travel-safe pose or the arms are busy
    or unverified, INTERNAL when the arm reading does not show where both arms
    are, and TIMEOUT when the arm check cannot finish in time, each a plain
    StepError, never a GoalFailed, raised before the verb sends a goal.

    pose waits for the robot's pose in the map until `timeout_s` is up, and then
    for the end of the backend read it began last. It raises
    REFUSED for a `timeout_s` that is not positive, CANCELLED on a stop, and
    TIMEOUT, UNAVAILABLE or SKILL_FAILED when no usable pose comes in time.
    """

    def places(self) -> Mapping[str, Location]: ...

    def go_to(self, place: str, *, timeout_s: float = 120.0) -> Pose2D: ...

    def move_relative(
        self,
        forward_m: float = 0.0,
        left_m: float = 0.0,
        turn_deg: float = 0.0,
        *,
        timeout_s: float = 60.0,
    ) -> Pose2D: ...

    def stop(self) -> None:
        """Cancel the active goal. Never raises."""
        ...

    def pose(self, *, timeout_s: float = 1.0) -> Pose2D: ...


class Speech(Protocol):
    """ctx.speech: says one line.

    Raises StepError REFUSED for blank text and UNAVAILABLE when nothing
    listens. It does not check the cancel token, so a bare stop can answer with
    its fixed line while motion stays latched (decision 6).
    """

    def say(self, text: str, *, timeout_s: float = 10.0) -> None: ...


class Vision(Protocol):
    """ctx.vision: what is in view now.

    A query of "" matches everything. Otherwise it matches an object's name,
    category or plural category, or a person's name, pose or gesture, ignoring
    case. Nothing matching returns []. Raises StepError UNAVAILABLE without a
    backend, TIMEOUT past `timeout_s` and CANCELLED on a stop.
    """

    def objects(self, query: str = "", *, timeout_s: float = 5.0) -> list[ObjectSeen]: ...

    def people(self, query: str = "", *, timeout_s: float = 5.0) -> list[PersonSeen]: ...


class Arms(Protocol):
    """ctx.arms: named arm poses and the grippers, through the commander (ADR-0021).

    The LLM picks poses by name and never sees joint values (ADR-0022). Every
    answer an arm atomic builds from these verbs ends with the arm sentence,
    ArmsReport.sentence(), so the LLM always knows whether the robot may drive
    (ADR-0024). A stop or a reset ends a verb's wait through the shared cancel
    token, and the backend then cancels the commander's goal.

    poses() is the poses offered here, in file order: on the robot only those
    tried on it. It raises StepError UNAVAILABLE without a commander.

    state() never raises. `read` None reads the arms only when the backend's
    memory cannot say whether the robot may drive; True always reads them;
    False never does. A read that fails leaves the memory as it was.

    read() reads both arms now, recovering first while they are unverified,
    then each gripper's finger. It raises StepError as the read of the arms
    does: PRECONDITION_FAILED while a goal runs or is outstanding or when
    recovery refuses, TIMEOUT, UNAVAILABLE, INTERNAL for a reading that lacks
    a joint, CANCELLED on a stop. A gripper that cannot be read is in its
    GripperReading, not an error, unless a stop ended the read.

    move_to() moves `arm` to the pose, waits and checks the joints against
    it. It raises StepError NOT_FOUND for a pose that is unknown or not
    offered here; REFUSED for an arm the pose does not take, or a right-arm
    pose that needs openarm_right_joint2 while walkie-ros holds it fixed;
    PRECONDITION_FAILED while a goal runs or is outstanding, or while the
    arms are unverified; SKILL_FAILED when the commander fails the goal, or
    when the joints do not end in the pose or cannot be checked after a
    reported success; TIMEOUT when the goal does not finish in time;
    UNAVAILABLE without the commander or when it cannot read the joints;
    INTERNAL for a rejected goal or joints that do not match the robot
    model; CANCELLED on a stop or a reset. Every message ends with the arm
    sentence.

    gripper() moves one gripper to a named position. It raises StepError
    PRECONDITION_FAILED, TIMEOUT, UNAVAILABLE, INTERNAL or CANCELLED as
    move_to() does. Every other end is an outcome, not an error
    (GripperMoved.outcome). Every message ends with the arm sentence.

    pick, place_on and hand_over pick, place and hand over, each returning
    the object's id. They raise StepError NOT_FOUND for an unknown object,
    place or person; PRECONDITION_FAILED when the hand or the robot's
    position does not allow the step; REFUSED for a place that cannot hold
    an object; CANCELLED on a stop; TIMEOUT past `timeout_s`; SKILL_FAILED
    when the motion fails; UNAVAILABLE where no grasp skill exists, as on
    the robot today.
    """

    def poses(self) -> Mapping[str, ArmPose]: ...

    def state(self, *, read: bool | None = None) -> ArmsReport: ...

    def read(self) -> ArmsReading: ...

    def move_to(self, pose: str, arm: Arm = "both") -> ArmsMoved: ...

    def gripper(self, arm: Gripper, state: GripperState) -> GripperMoved: ...

    def pick(self, object_id: str, *, timeout_s: float = 60.0) -> str: ...

    def place_on(self, place: str, *, timeout_s: float = 60.0) -> str: ...

    def hand_over(self, person_id: str, *, timeout_s: float = 60.0) -> str: ...


# Every verb of UnavailableArms raises UNAVAILABLE with this message.
NO_ARMS = "Walkie cannot use its arms here: no arm commander is connected."


class UnavailableArms:
    """The ctx.arms of a robot run without --arms: every verb but state() refuses.

    state() says the arms are unknown, since the Arms protocol says it never raises.
    """

    def _refuse(self) -> NoReturn:
        raise StepError(ErrorCode.UNAVAILABLE, NO_ARMS)

    def poses(self) -> Mapping[str, ArmPose]:
        self._refuse()

    def state(self, *, read: bool | None = None) -> ArmsReport:
        return ArmsReport(ArmsState())

    def read(self) -> ArmsReading:
        self._refuse()

    def move_to(self, pose: str, arm: Arm = "both") -> ArmsMoved:
        self._refuse()

    def gripper(self, arm: Gripper, state: GripperState) -> GripperMoved:
        self._refuse()

    def pick(self, object_id: str, *, timeout_s: float = 60.0) -> str:
        self._refuse()

    def place_on(self, place: str, *, timeout_s: float = 60.0) -> str:
        self._refuse()

    def hand_over(self, person_id: str, *, timeout_s: float = 60.0) -> str:
        self._refuse()


class Facts(Protocol):
    """ctx.facts: the time, the date and the team, so the robot never guesses them.

    now() returns a timezone-aware datetime in the robot's configured timezone.
    Both verbs raise StepError UNAVAILABLE without a fact source. Neither
    checks the cancel token, since neither moves anything.
    """

    def now(self) -> datetime: ...

    def team(self) -> Team: ...


class UnavailableFacts:
    """The ctx.facts of a robot with no fact source: every verb refuses."""

    def _refuse(self) -> NoReturn:
        raise StepError(
            ErrorCode.UNAVAILABLE,
            "Walkie cannot read the time or its team here: no fact source is configured.",
        )

    def now(self) -> datetime:
        self._refuse()

    def team(self) -> Team:
        self._refuse()


class Memory(Protocol):
    """ctx.memory: the people Walkie has met, by name (ADR-0070).

    `person_id` is a live id, as find_people gives it. remember_person stores that
    person under `name` with `facts`, from the face and appearance samples the backend
    collected while they were in view; recognize says who they are. No sample and no
    stored id leaves the backend. The people remembered are forgotten at the next
    task, unless the backend was told to resume the last session.

    remember_person raises StepError REFUSED for a name or facts outside
    guards.require_person_details. Both verbs raise NOT_FOUND for a person who is not
    in view or whose face and appearance were never seen; UNAVAILABLE without a
    memory, or when it failed in a way that may pass, so the same call may work later;
    REFUSED when the memory refused the call; TIMEOUT when it did not answer within
    `timeout_s`, after which remember_person may have stored the person already; and
    CANCELLED on a stop.
    """

    def remember_person(
        self,
        person_id: str,
        name: str,
        facts: Mapping[str, str] | None = None,
        *,
        timeout_s: float = 15.0,
    ) -> Remembered: ...

    def recognize(self, person_id: str, *, timeout_s: float = 15.0) -> Recognized: ...


class UnavailableMemory:
    """The ctx.memory of a robot with no people memory: every verb refuses."""

    def _refuse(self) -> NoReturn:
        raise StepError(
            ErrorCode.UNAVAILABLE,
            "Walkie cannot remember people here: no people memory is configured.",
        )

    def remember_person(
        self,
        person_id: str,
        name: str,
        facts: Mapping[str, str] | None = None,
        *,
        timeout_s: float = 15.0,
    ) -> Remembered:
        self._refuse()

    def recognize(self, person_id: str, *, timeout_s: float = 15.0) -> Recognized:
        self._refuse()


@dataclass(frozen=True)
class Ctx:
    """What an atomic runs against. Verbs hold the same `cancel` they are built with."""

    cancel: CancelToken
    base: Base
    speech: Speech
    vision: Vision
    arms: Arms = UnavailableArms()
    facts: Facts = UnavailableFacts()
    memory: Memory = UnavailableMemory()
