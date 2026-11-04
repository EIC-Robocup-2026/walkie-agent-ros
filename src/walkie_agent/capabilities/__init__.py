"""The robot's capabilities on ctx: ctx.base, ctx.speech, ctx.vision and ctx.arms.

Each capability is a Protocol here. An atomic calls its verbs as
ctx.base.go_to("shelf") and never sees a topic or an action. A failure is a
StepError. The guards live inside the verbs, through capabilities.guards, so
whatever calls a verb gets them. There is no velocity verb.

This package imports no rclpy, rai or ROS message, so a teammate can read and
test against it without ROS. FakeRobot, in capabilities.fake, implements every
verb. The ROS-backed verbs live in walkie_agent.ros.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn, Protocol

from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.capabilities.records import ObjectSeen, PersonSeen, Pose2D
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import Location

__all__ = [
    "Arms",
    "Base",
    "CancelToken",
    "Ctx",
    "ObjectSeen",
    "PersonSeen",
    "Pose2D",
    "Speech",
    "UnavailableArms",
    "Vision",
]


class Base(Protocol):
    """ctx.base: drives to named places and short relative moves.

    go_to and move_relative raise StepError NOT_FOUND for an unknown place;
    REFUSED for a value that is not finite, an empty move or one over the 3 m
    cap; CANCELLED on a stop; TIMEOUT past `timeout_s`, UNAVAILABLE without a
    navigation backend and SKILL_FAILED when the goal fails.
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

    def pose(self) -> Pose2D: ...


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
    """ctx.arms: picks, places and hands over, each returning the object's id.

    Raises StepError NOT_FOUND for an unknown object, place or person;
    PRECONDITION_FAILED when the hand or the robot's position does not allow
    the step; REFUSED for a place that cannot hold an object; CANCELLED on a
    stop; TIMEOUT past `timeout_s`; SKILL_FAILED when the motion fails.
    """

    def pick(self, object_id: str, *, timeout_s: float = 60.0) -> str: ...

    def place_on(self, place: str, *, timeout_s: float = 60.0) -> str: ...

    def hand_over(self, person_id: str, *, timeout_s: float = 60.0) -> str: ...


class UnavailableArms:
    """The ctx.arms of a robot with no arm commander client yet: every verb refuses."""

    def _refuse(self) -> NoReturn:
        raise StepError(
            ErrorCode.UNAVAILABLE,
            "Walkie cannot use its arms here: the arm commander client does not exist yet.",
        )

    def pick(self, object_id: str, *, timeout_s: float = 60.0) -> str:
        self._refuse()

    def place_on(self, place: str, *, timeout_s: float = 60.0) -> str:
        self._refuse()

    def hand_over(self, person_id: str, *, timeout_s: float = 60.0) -> str:
        self._refuse()


@dataclass(frozen=True)
class Ctx:
    """What an atomic runs against. Verbs hold the same `cancel` they are built with."""

    cancel: CancelToken
    base: Base
    speech: Speech
    vision: Vision
    arms: Arms = UnavailableArms()
