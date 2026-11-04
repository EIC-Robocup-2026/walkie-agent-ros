"""The small frozen records that capability verbs return."""

from dataclasses import dataclass


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
    """One person in view. `pose` is a posture, such as "sitting person"."""

    id: str
    # None until the person has been identified.
    name: str | None
    pose: str
    gesture: str
    room: str
