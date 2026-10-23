"""Shared by the people and seat atomics: handle kinds, payloads, config and helpers.

find_people and find_seats read walkie-vision's newest arrays from the
`vision` backend, a VisionFeed, and keep what they saw as person and seat
handles. recognize_person and remember_person take a person handle back,
with the vectors VisionFeed collected for the person's track, and call the
`people_memory` backend, a PeopleMemory (hri.md, Atomics). The module's
name starts with "_", so the registry does not load it as a tool.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from tf2_ros import Buffer

from walkie_agent.perception.gestures import DEFAULT_GESTURE_CONFIG, GestureConfig
from walkie_agent.perception.geometry import box_heading
from walkie_agent.perception.identity import DEFAULT_IDENTITY_CONFIG, IdentityConfig
from walkie_agent.perception.people_memory import ModelVector
from walkie_agent.perception.seating import DEFAULT_SEATING_CONFIG, BBox, SeatingConfig
from walkie_agent.perception.vision_feed import VisionFeed
from walkie_agent.toolbox import Handle, KindPolicy, StepError, ToolContext, register_kind

# The backend names in META.requires and ctx.backend().
VISION = "vision"
PEOPLE_MEMORY = "people_memory"


@dataclass(frozen=True)
class PerceptionConfig:
    """What the people and seat atomics use.

    The thresholds are walkie-agent-v2's, tuned on the robot, and live with
    the logic they tune: the gesture rules in GestureConfig, the per-model
    face and appearance distance limits in IdentityConfig, and the seat and
    sofa classes in SeatingConfig. The timings are ADR-0050's.
    """

    # The newest array may be this old on the node clock. An image a second
    # old still shows who is where; an older one may not (ADR-0050).
    max_age_s: float = 1.0
    # How long find_people and find_seats wait for such an array.
    wait_s: float = 1.0
    # How much longer find_seats waits for the people array of the objects
    # array's image, which walkie-vision may publish after the objects.
    pair_wait_s: float = 0.5
    # How long a heading waits for the camera's transform at the image's
    # stamp. TF can lag the image a little on the robot; no longer, since
    # each person or place waits again.
    tf_timeout_s: float = 0.2
    gestures: GestureConfig = DEFAULT_GESTURE_CONFIG
    identity: IdentityConfig = DEFAULT_IDENTITY_CONFIG
    seating: SeatingConfig = DEFAULT_SEATING_CONFIG


# Read by the atomics on every call, as `_perception.CONFIG`, so a test can
# replace it.
CONFIG = PerceptionConfig()

# A person handle goes stale after 5 cm of base translation or 10 minutes.
# Rotation does not stale it: the gaze loop turns the base while it follows
# the person, and the handle names a track, which a turn does not change.
# How long a person handle may serve localize_person, which needs where the
# person stands, is still open (perception.md, question 8).
PERSON_POLICY = KindPolicy(
    prefix="per",
    max_translation_m=0.05,
    max_rotation_deg=None,
    max_age_s=600.0,
    source="find_people",
)
# A seat handle has the same limits. A turn keeps it valid as well: its
# heading is in the map frame, so look_at can turn to it from any heading,
# but a drive changes which seats the robot sees and from where.
SEAT_POLICY = KindPolicy(
    prefix="seat",
    max_translation_m=0.05,
    max_rotation_deg=None,
    max_age_s=600.0,
    source="find_seats",
)
register_kind("person", PERSON_POLICY)
register_kind("seat", SEAT_POLICY)


@dataclass(frozen=True)
class PersonPayload:
    """What find_people keeps for one person, behind a person handle.

    Boxes are x1, y1, x2, y2 in pixels of the image that `header` and the
    image size describe. The gaze loop follows `track_id`, and turns to
    `heading_map` until it sees the track. The message objects are shared
    with VisionFeed and must not be modified.
    """

    # walkie-vision's track id; 0 means the person is not tracked.
    track_id: int
    # The third of the image that holds the box's centre: "left", "centre"
    # or "right", as the robot sees it.
    position: str
    bbox: BBox
    # None when no face was found in the box.
    face_bbox: BBox | None
    # 17 COCO points as x0, y0, x1, y1, and so on, and their scores.
    keypoints: tuple[float, ...]
    keypoint_scores: tuple[float, ...]
    # The gesture and posture names of walkie_agent.perception.gestures.
    gestures: frozenset[str]
    # The face vector, only when the face passes VisionFeed's face_trusted,
    # the gate its collection uses: a doubtful face would name someone else.
    face: ModelVector | None
    appearance: ModelVector | None
    header: Header
    image_width: int
    image_height: int
    # The map-frame yaw, in radians, of the ray toward the face, else the
    # body. None when TF had no camera transform at the stamp.
    heading_map: float | None


@dataclass(frozen=True)
class SeatPayload:
    """What find_seats keeps for one place to sit, behind a seat handle.

    `heading_map` is computed when the image is seen, because a seat handle
    outlives the TF buffer's 10 s: look_at turns to this fixed heading.
    """

    # The third of the image that holds the place's centre.
    position: str
    # The place's box: the whole seat, or one part of a sofa.
    box: BBox
    # The detector's class, such as "chair" or "sofa".
    label: str
    # "left", "middle" or "right" for a place on a sofa, else None.
    part: str | None
    occupied: bool
    header: Header
    image_width: int
    image_height: int
    heading_map: float | None


def vision_feed(ctx: ToolContext) -> VisionFeed:
    return ctx.backend(VISION)


def image_position(u: float, width: int) -> str:
    """The third of the image that column `u` lies in: left, centre or right.

    Thirds, not pixels, because that is what the LLM can say to a person.
    """
    if width <= 0:
        # An empty camera_info: walkie-vision always fills it, so this
        # means a broken publisher rather than a person nowhere.
        return "unknown position"
    if u < width / 3:
        return "left"
    if u < 2 * width / 3:
        return "centre"
    return "right"


def headings(
    boxes: Sequence[BBox], camera_info: CameraInfo, header: Header, tf_buffer: Buffer
) -> list[float | None]:
    """box_heading() of every box of one image, in order.

    Once one box has no heading, the rest get None without a lookup: TF has
    no camera transform at the stamp, or camera_info has no focal length,
    and every further lookup would wait out the timeout again.
    """
    result: list[float | None] = []
    failed = False
    for box in boxes:
        heading = None
        if not failed:
            heading = box_heading(box, camera_info, header, tf_buffer, CONFIG.tf_timeout_s)
            failed = heading is None
        result.append(heading)
    return result


def is_handle_id(text: str) -> bool:
    """True for one handle's id, such as per_51c0.1; False for a set id, such as per_51c0."""
    return "." in text.strip()


def person_handles(ctx: ToolContext, person: str) -> list[Handle]:
    """The person handle `person`, or every member of the person set `person`.

    Raises the handle store's NOT_FOUND, STALE or UNAVAILABLE.
    """
    person = person.strip()
    if is_handle_id(person):
        return [ctx.handles.get(person, kind="person")]
    return ctx.handles.get_set(person, kind="person")


def person_vectors(
    feed: VisionFeed, payload: PersonPayload
) -> tuple[list[ModelVector], list[ModelVector]]:
    """The face and appearance vectors of a person handle, oldest first.

    VisionFeed collects every people array before an atomic can read it, so
    the vectors collected for the track already hold the handle's own, and
    every face the track showed while the robot talked to the person. The
    handle's own vectors stand alone only when the person is untracked, or
    VisionFeed dropped the track after the person was out of view too long.
    """
    collected = feed.collected_vectors(payload.track_id)
    faces = list(collected.faces)
    appearances = list(collected.appearances)
    if not faces and payload.face is not None:
        faces = [payload.face]
    if not appearances and payload.appearance is not None:
        appearances = [payload.appearance]
    return faces, appearances


def group_by_model(vectors: Sequence[ModelVector]) -> dict[str, list[np.ndarray]]:
    """The vectors by model name, each group in its original order."""
    groups: dict[str, list[np.ndarray]] = {}
    for model, vector in vectors:
        groups.setdefault(model, []).append(vector)
    return groups


def facts_text(facts: Mapping[str, str]) -> str:
    """Facts as "drink: milk, color: blue", in their stored order."""
    return ", ".join(f"{key}: {value}" for key, value in facts.items())


def no_vectors(handle_id: str) -> StepError:
    return StepError(
        "NOT_FOUND",
        f"No face or appearance seen for {handle_id}. Look at them and call find_people again.",
        {"handle": handle_id},
    )
