"""The robot's capabilities on ctx: base, speech, vision, arms, facts, memory, object memory and web.

Each capability is a Protocol here. An atomic calls its verbs as
ctx.base.go_to("shelf") and never sees a topic or an action. A failure is a
StepError. The guards live inside the verbs, through capabilities.guards, so
whatever calls a verb gets them. There is no velocity verb.

This package imports no rclpy, rai or ROS message, so a teammate can read and
test against it without ROS. FakeRobot, in capabilities.fake, implements every
verb but the object memory's and the web search's, whose ctx fields it leaves
unavailable. The ROS-backed verbs live in walkie_agent.ros, and the web search in
capabilities.web_search. The arm records build on
walkie_agent.arm, which imports no ROS either.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import NoReturn, Protocol

from walkie_agent.arm.pick import HandPose
from walkie_agent.arm.poses import ArmPose
from walkie_agent.arm.state import ArmsState
from walkie_agent.capabilities.guards import CancelToken, begin_lookup
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
    RecalledObject,
    Recognized,
    Remembered,
    RoomObjects,
    Team,
    WebAnswer,
)
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import Location

__all__ = [
    "NO_ARMS",
    "NO_OBJECT_MEMORY",
    "NO_WEB_SEARCH",
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
    "HandPose",
    "Memory",
    "ObjectMemory",
    "ObjectSeen",
    "PersonSeen",
    "Pose2D",
    "RecalledObject",
    "Recognized",
    "Remembered",
    "RoomObjects",
    "Speech",
    "Team",
    "UnavailableArms",
    "UnavailableFacts",
    "UnavailableMemory",
    "UnavailableObjectMemory",
    "UnavailableVision",
    "UnavailableWebSearch",
    "Vision",
    "WebAnswer",
    "WebSearch",
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


class UnavailableVision:
    """Checks a verb's arguments and the latch like a real backend, then raises UNAVAILABLE.

    The ctx.vision of a robot run without --vision. An empty list would say that
    nothing is in view, and a remembered database row is not a current view, so
    neither stands in for a backend.
    """

    def __init__(self, cancel: CancelToken) -> None:
        self._cancel = cancel

    def objects(self, query: str = "", *, timeout_s: float = 5.0) -> list[ObjectSeen]:
        begin_lookup(self._cancel, query, timeout_s, "looking for objects")
        raise StepError(ErrorCode.UNAVAILABLE, "Live object perception is not connected.")

    def people(self, query: str = "", *, timeout_s: float = 5.0) -> list[PersonSeen]:
        begin_lookup(self._cancel, query, timeout_s, "looking for people")
        raise StepError(ErrorCode.UNAVAILABLE, "Live people perception is not connected.")


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
    sentence. `lift_m` also moves the lift to that height, through the arm's
    *_lift group, for one arm only: pick_object brings the arm home with it
    after a pick (decision PK-5), never the LLM. `touch` is reach()'s, and a
    move without it ends the allowance.

    gripper() moves one gripper to a named position. While the allowance
    below is unknown, it first turns it off and sends no goal unless the
    commander confirms that; a known allowance is kept. It raises StepError
    PRECONDITION_FAILED, TIMEOUT, UNAVAILABLE, INTERNAL or CANCELLED as
    move_to() does, and SKILL_FAILED when the commander refuses that
    change. Every other end is an outcome, not an error
    (GripperMoved.outcome). Every message ends with the arm sentence.

    reach() moves one hand's link to a HandPose that pick_object computed
    from the object memory, never one from the LLM (decision PK-4), and
    returns the commander's status. `whole_body` plans for the base, the
    lift and the arm together; `cartesian` moves the hand along a straight
    line, for the arm and the lift. These are the only motions that move
    the lift (decision PK-5). `touch` lets both grippers' fingers touch the
    sensed octomap, which a grasp needs, and stays in effect until a reach
    or a move_to() without it, the next open or restore_touch(). `ready`, when
    given, runs right before the goal, after the allowance is set and, for a
    straight reach, after the wait for the arm to come to rest, which a backend
    may end after a bound; whatever it raises ends the reach unchanged, with no
    goal sent. The arm then
    rests in no named pose, so it is not safe for driving. It raises
    StepError REFUSED for a pose or a timeout that
    is not finite, or a whole-body straight line; SKILL_FAILED when the
    commander fails the goal or only planned it; and the rest as move_to()
    does. Every message ends with the arm sentence.

    restore_touch() turns that allowance off, the commander's default, and
    moves nothing. It goes through also while a stop is in effect, so
    pick_object ends the allowance on every path, a stop included. It raises
    StepError UNAVAILABLE, TIMEOUT or SKILL_FAILED when the commander does not
    confirm the change, or still holds the allowance on when it is read back
    after; the allowance is then unknown, and the next motion
    of an arm or a gripper sets the one it needs before its goal (off, for a
    gripper), so nothing moves until that set is confirmed. Its messages
    carry no arm sentence. Without a commander there is nothing to turn off,
    and it returns.

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

    def move_to(
        self,
        pose: str,
        arm: Arm = "both",
        *,
        lift_m: float | None = None,
        touch: bool = False,
    ) -> ArmsMoved: ...

    def gripper(self, arm: Gripper, state: GripperState) -> GripperMoved: ...

    def reach(
        self,
        arm: Gripper,
        pose: HandPose,
        *,
        whole_body: bool = False,
        cartesian: bool = False,
        touch: bool = False,
        timeout_s: float = 60.0,
        ready: Callable[[], object] | None = None,
    ) -> str: ...

    def restore_touch(self) -> None: ...

    def pick(self, object_id: str, *, timeout_s: float = 60.0) -> str: ...

    def place_on(self, place: str, *, timeout_s: float = 60.0) -> str: ...

    def hand_over(self, person_id: str, *, timeout_s: float = 60.0) -> str: ...


# Every verb of UnavailableArms raises UNAVAILABLE with this message.
NO_ARMS = "Walkie cannot use its arms here: no arm commander is connected."


class UnavailableArms:
    """The ctx.arms of a robot run without --arms: every verb but two refuses.

    state() says the arms are unknown, since the Arms protocol says it never raises,
    and restore_touch() returns, since no commander holds an allowance to turn off.
    """

    def _refuse(self) -> NoReturn:
        raise StepError(ErrorCode.UNAVAILABLE, NO_ARMS)

    def poses(self) -> Mapping[str, ArmPose]:
        self._refuse()

    def state(self, *, read: bool | None = None) -> ArmsReport:
        return ArmsReport(ArmsState())

    def read(self) -> ArmsReading:
        self._refuse()

    def move_to(
        self,
        pose: str,
        arm: Arm = "both",
        *,
        lift_m: float | None = None,
        touch: bool = False,
    ) -> ArmsMoved:
        self._refuse()

    def gripper(self, arm: Gripper, state: GripperState) -> GripperMoved:
        self._refuse()

    def reach(
        self,
        arm: Gripper,
        pose: HandPose,
        *,
        whole_body: bool = False,
        cartesian: bool = False,
        touch: bool = False,
        timeout_s: float = 60.0,
        ready: Callable[[], object] | None = None,
    ) -> str:
        self._refuse()

    def restore_touch(self) -> None:
        return None

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


class ObjectMemory(Protocol):
    """ctx.object_memory: the objects Walkie saw before, as an object memory keeps them.

    An object's id here is the memory's own, a decimal number such as "18", never a
    live id from find_objects. search gives the objects remembered as present whose
    label, another label the detector gave them or official name matches `name` as
    find_objects matches a label (perception.object_words.matches): the label, its
    category or the category's plural, ignoring case and spaces at either end, or an
    arena object's name that ends with the label as a word, as "water bottle" for
    "bottle"; an empty name matches every object. A match's own `name` may differ
    from the one asked for: search("chair") gives a stool whose official name is
    "stool".
    Objects whose own label or official name matches come first, then those that
    only another label matches, each in the memory's order, at most `limit` of
    them. room gives the objects remembered as present in the room named `room`,
    ignoring case, at most `limit`. get gives one object in any state, following a
    merge: the object it returns may have another id than `object_id`. No verb
    moves anything, so none waits for a stop to commit.

    Every verb raises StepError REFUSED for an argument outside its range, such as
    an id that is not a memory id; NOT_FOUND for an object or a room the memory does
    not know; UNAVAILABLE without a memory, or when it is not ready or failed in a
    way that may pass, so the same call may work later; TIMEOUT when it did not
    answer within `timeout_s`; and CANCELLED on a stop.
    """

    def search(
        self, name: str, *, limit: int = 10, timeout_s: float = 10.0
    ) -> tuple[RecalledObject, ...]: ...

    def room(self, room: str, *, limit: int = 50, timeout_s: float = 10.0) -> RoomObjects: ...

    def get(self, object_id: str, *, timeout_s: float = 10.0) -> RecalledObject: ...


# What UnavailableObjectMemory says, kept here so the atomics' tests can match it.
NO_OBJECT_MEMORY = "Walkie cannot recall objects here: no object memory is configured."


class UnavailableObjectMemory:
    """The ctx.object_memory of a robot with no object memory: every verb refuses."""

    def _refuse(self) -> NoReturn:
        raise StepError(ErrorCode.UNAVAILABLE, NO_OBJECT_MEMORY)

    def search(
        self, name: str, *, limit: int = 10, timeout_s: float = 10.0
    ) -> tuple[RecalledObject, ...]:
        self._refuse()

    def room(self, room: str, *, limit: int = 50, timeout_s: float = 10.0) -> RoomObjects:
        self._refuse()

    def get(self, object_id: str, *, timeout_s: float = 10.0) -> RecalledObject:
        self._refuse()


class WebSearch(Protocol):
    """ctx.web: answers a question from a web search, as a short summary and its sources.

    The answer never holds the pages' own text: a model with no tools summarizes
    the results, which it reads as untrusted data. search raises StepError REFUSED
    for a blank or overlong query; TIMEOUT when the search does not answer in time;
    NOT_FOUND when it finds nothing; and UNAVAILABLE without a search backend, or
    when the search or the summary fails. It does not check the cancel token, since
    it moves nothing.
    """

    def search(self, query: str) -> WebAnswer: ...


# What UnavailableWebSearch says, kept here so the atomic's tests can match it.
NO_WEB_SEARCH = "Walkie cannot search the web here: web search is not enabled."


class UnavailableWebSearch:
    """The ctx.web of a run without --web-search: search refuses."""

    def search(self, query: str) -> WebAnswer:
        raise StepError(ErrorCode.UNAVAILABLE, NO_WEB_SEARCH)


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
    object_memory: ObjectMemory = UnavailableObjectMemory()
    web: WebSearch = UnavailableWebSearch()
