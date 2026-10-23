"""What a simulated camera can see: people and objects in the map frame.

A scenario builds a Scene for each camera image, and oracle_perception draws
it as walkie-vision's detections (perception.md, "In simulation"). The types
know nothing of a task, so the HRI scenario and later the harvest fill them
in alike. They are plain data with no ROS, so a scenario's state machine can
build them in unit tests.
"""

import math
from dataclasses import dataclass

# The height of the person model's reference figure (walkie_agent.sim.projection),
# and of a person whose scenario gives no height.
STANDARD_HEIGHT_M = 1.70
# The top of the seat under a sitting person whose scenario gives none: a
# common chair seat. The cushions of chula_pat14_6's sofa are at 0.4575 m.
DEFAULT_SEAT_TOP_M = 0.45
# A person shows a face only while they face the camera within this angle.
FACE_LIMIT_DEG = 60.0

# The arm gestures the person model draws, walkie-agent-v2's gesture ids
# without the postures. raising_left_arm and raising_right_arm are the
# person's own arms; pointing_left and pointing_right are directions in the
# image, as walkie_agent.perception.gestures reads them.
ARM_GESTURES: tuple[str, ...] = (
    "waving",
    "raising_left_arm",
    "raising_right_arm",
    "pointing_left",
    "pointing_right",
)
# The postures it draws. v2's lying_down has no figure: no scenario needs one.
BODY_POSTURES: tuple[str, ...] = ("standing", "sitting")

# PersonDetection.track_id and ObjectDetection.track_id are uint32.
_MAX_TRACK_ID = 2**32 - 1


def _check_track_id(track_id: int, what: str) -> None:
    if not 0 <= track_id <= _MAX_TRACK_ID:
        raise ValueError(f"{what}.track_id must be a uint32 (0 means untracked), got {track_id}.")


@dataclass(frozen=True)
class MapBox:
    """A box in the map frame: centre and size in meters, turned by `yaw` about the vertical."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 3 or len(self.size) != 3:
            raise ValueError("MapBox.center and MapBox.size need three values each.")
        if not all(side > 0 for side in self.size):
            raise ValueError(f"MapBox.size must be positive, got {self.size}.")


@dataclass(frozen=True)
class ScenePerson:
    """One person in the map frame, and what the camera can tell about them."""

    # walkie-vision's track id, kept from image to image; 0 means untracked.
    track_id: int
    x: float
    y: float
    # The map-frame yaw that the person's front faces, in radians.
    facing: float
    height: float = STANDARD_HEIGHT_M
    # One of ARM_GESTURES, or None for hanging arms.
    gesture: str | None = None
    # One of BODY_POSTURES.
    posture: str = "standing"
    # The face vector is hash_vector("face:" + face); None shows no face at all.
    face: str | None = None
    # The appearance vector is hash_vector("reid:" + appearance); None gives none.
    appearance: str | None = None
    # The top of the seat under the person, in meters; used only when sitting.
    seat_top: float = DEFAULT_SEAT_TOP_M

    def __post_init__(self) -> None:
        _check_track_id(self.track_id, "ScenePerson")
        if not self.height > 0:
            raise ValueError(f"ScenePerson.height must be positive, got {self.height}.")
        if self.gesture is not None and self.gesture not in ARM_GESTURES:
            raise ValueError(
                f"ScenePerson.gesture must be None or one of {', '.join(ARM_GESTURES)}, "
                f"got {self.gesture!r}."
            )
        if self.posture not in BODY_POSTURES:
            raise ValueError(
                f"ScenePerson.posture must be one of {', '.join(BODY_POSTURES)}, "
                f"got {self.posture!r}."
            )
        if self.seat_top < 0:
            raise ValueError(f"ScenePerson.seat_top must not be negative, got {self.seat_top}.")

    def faces_camera(
        self, camera_xy: tuple[float, float], limit_deg: float = FACE_LIMIT_DEG
    ) -> bool:
        """True when the person's front points at `camera_xy` within `limit_deg` degrees.

        Only the horizontal direction counts: a camera above or below the
        face sees it as long as the person is turned towards it.
        """
        dx, dy = camera_xy[0] - self.x, camera_xy[1] - self.y
        if math.hypot(dx, dy) < 1e-9:
            return False
        off = math.atan2(dy, dx) - self.facing
        off = math.atan2(math.sin(off), math.cos(off))
        return abs(off) <= math.radians(limit_deg)


@dataclass(frozen=True)
class SceneObject:
    """One object, such as a seat, with the class name a detector would give it."""

    # walkie-vision's track id; 0 means untracked.
    track_id: int
    label: str
    box: MapBox

    def __post_init__(self) -> None:
        _check_track_id(self.track_id, "SceneObject")


@dataclass(frozen=True)
class Scene:
    """Everyone and everything in the world at one moment, seen or not."""

    people: tuple[ScenePerson, ...] = ()
    objects: tuple[SceneObject, ...] = ()
