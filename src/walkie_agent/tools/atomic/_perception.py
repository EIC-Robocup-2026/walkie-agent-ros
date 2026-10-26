"""Shared by the people and seat atomics: handle kinds, payloads, config and helpers.

find_people and find_seats read walkie-vision's newest arrays from the
`vision` backend, a VisionFeed, and keep what they saw as person and seat
handles. recognize_person and remember_person take a person handle back,
with its own vectors and those VisionFeed collected for the person's track,
and call the `people_memory` backend, a PeopleMemory (hri.md, Atomics). The
module's name starts with "_", so the registry does not load it as a tool.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException
from tf_transformations import euler_from_quaternion

from walkie_agent.perception.front import DEFAULT_FRONT_CONFIG, FrontConfig
from walkie_agent.perception.gestures import DEFAULT_GESTURE_CONFIG, GestureConfig
from walkie_agent.perception.geometry import camera_transform, ray_heading
from walkie_agent.perception.identity import DEFAULT_IDENTITY_CONFIG, IdentityConfig
from walkie_agent.perception.people_memory import ModelVector
from walkie_agent.perception.seating import DEFAULT_SEATING_CONFIG, BBox, SeatingConfig
from walkie_agent.perception.vision_feed import (
    PUBLISHER_HINT,
    CollectedVectors,
    Sample,
    TrackKey,
    VisionFeed,
)
from walkie_agent.toolbox import (
    Handle,
    HandleSet,
    KindPolicy,
    Pose2D,
    StepError,
    ToolContext,
    register_kind,
)
from walkie_agent.tools.navigation import MAP_FRAME, ROBOT_FRAME, tf_publisher_hint

# The backend names in META.requires and ctx.backend().
VISION = "vision"
PEOPLE_MEMORY = "people_memory"


@dataclass(frozen=True)
class PerceptionConfig:
    """What the people and seat atomics use.

    The thresholds are walkie-agent-v2's, tuned on the robot, and live with
    the logic they tune: the gesture rules in GestureConfig, the per-model
    face and appearance distance limits in IdentityConfig, and the seat and
    sofa classes in SeatingConfig. The exception is FrontConfig, who counts
    as close in front for find_people(in_front=True), which is new and was
    tuned in simulation only. The timings are ADR-0050's. The values still
    to tune on the robot can be set in config.toml (walkie_agent.tuning).
    """

    # The newest array may be this old on the node clock. An image a second
    # old still shows who is where; an older one may not (ADR-0050).
    max_age_s: float = 1.0
    # How long find_people and find_seats wait for such an array.
    wait_s: float = 1.0
    # How much longer find_seats waits for the people array of the objects
    # array's image, which walkie-vision may publish after the objects.
    pair_wait_s: float = 0.5
    # How long a lookup at the image's stamp waits for TF: the camera's
    # transform for the headings, and the base's for the handles' capture
    # pose. TF can lag the image a little on the robot; no longer, since a
    # call makes both lookups. Chosen in simulation: how far TF lags the
    # image on the robot is not measured yet, so config.toml can set it, as
    # [walkie.perception] tf_timeout_s.
    tf_timeout_s: float = 0.2
    gestures: GestureConfig = DEFAULT_GESTURE_CONFIG
    identity: IdentityConfig = DEFAULT_IDENTITY_CONFIG
    seating: SeatingConfig = DEFAULT_SEATING_CONFIG
    front: FrontConfig = DEFAULT_FRONT_CONFIG

    def __post_init__(self) -> None:
        for name in ("max_age_s", "wait_s", "pair_wait_s", "tf_timeout_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"PerceptionConfig.{name} must be a positive number, not {value}."
                )


# Read by the atomics on every call, as `_perception.CONFIG`, so a test can
# replace it. create_walkie_agent replaces it when it is given the tuning
# that config.toml sets (walkie_agent.tuning).
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
    image size describe. The gaze loop follows the track id of `track`, and
    turns to `heading_map` until it sees the track. The message objects are
    shared with VisionFeed and must not be modified.
    """

    # The person's track collection when the image came, from its
    # PeopleFrame; None when the person is not tracked (track id 0). The
    # key, not the bare track id: VisionFeed drops a track unseen for a
    # minute, after which the id can name someone else, while the handle
    # lasts ten minutes.
    track: TrackKey | None
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
    # header.stamp in nanoseconds, the stamp VisionFeed gives the samples of
    # this image, so person_vectors() can tell the handle's own apart.
    stamp_ns: int
    image_width: int
    image_height: int
    # The map-frame yaw, in radians, of the ray toward the face, else the
    # body. None when TF had no camera transform at the stamp, or the ray
    # has no yaw.
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
    """The map-frame heading of every box of one image, in order; None for a box without one.

    The camera's transform is looked up once per image. Without it every
    box gets None, and no box waits out the timeout again. With it, each
    box's ray is turned on its own: one ray can have no yaw while the others
    have one, as the central ray of a camera that looks straight down.
    """
    transform = camera_transform(header, tf_buffer, CONFIG.tf_timeout_s)
    if transform is None:
        return [None] * len(boxes)
    return [ray_heading(box, camera_info, transform) for box in boxes]


def capture_pose(ctx: ToolContext, header: Header) -> Pose2D:
    """The base's pose at the image's stamp, from the transform ctx.robot_pose() reads.

    A handle goes stale with the base's motion since its pose, and its
    payload is the scene as the camera saw it at the stamp, so the pose is
    the one at the stamp: a drive between the image and the call counts.

    Raises StepError UNAVAILABLE when the array has no stamp or TF has no
    such transform within the timeout. The pose now would let up to a
    second of motion, CONFIG.max_age_s, pass the stale check unseen, and
    the camera's transform at the stamp, which the headings need, is
    missing then too.
    """
    stamp = Time.from_msg(header.stamp)
    # Stamp zero asks TF for its latest transform, the pose now.
    if stamp.nanoseconds == 0:
        raise StepError(
            "UNAVAILABLE",
            "The vision array has no stamp, so where the robot was when it saw "
            f"this is unknown. {PUBLISHER_HINT}",
        )
    try:
        stamped = ctx.tf_buffer.lookup_transform(
            MAP_FRAME, ROBOT_FRAME, stamp, timeout=Duration(seconds=CONFIG.tf_timeout_s)
        )
    except TransformException as e:
        raise StepError(
            "UNAVAILABLE",
            f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME}) when the camera saw "
            f"this is unknown, so nothing was stored. {e}{tf_publisher_hint(ctx.node)}",
        ) from e
    t = stamped.transform
    q = t.rotation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return Pose2D(x=t.translation.x, y=t.translation.y, yaw=yaw)


def put_handles(
    ctx: ToolContext,
    kind: str,
    payloads: Sequence[object],
    header: Header,
    pose: Pose2D | None = None,
) -> HandleSet:
    """Store the payloads of one image as a handle set, with the base's pose at its stamp.

    `pose` is that pose when the caller already looked it up with
    capture_pose, as find_people(in_front=True) does for the base's yaw,
    so TF is asked once. Without it, the pose is looked up here.

    Raises StepError CANCELLED when a stop or a reset came during the call,
    so a stopped step leaves no handles behind, and capture_pose's
    UNAVAILABLE.
    """
    if pose is None:
        try:
            pose = capture_pose(ctx, header)
        except StepError:
            # A stop that came while the lookup waited is why the step
            # ended, not the missing transform, so it is reported first.
            ctx.check_cancelled()
            raise
    # Last, after the TF lookups, which can wait. put_set is given the
    # pose, so it never calls its pose provider, and nothing that can wait
    # comes after this check.
    ctx.check_cancelled()
    return ctx.handles.put_set(kind, payloads, pose=pose)


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
    """The face and appearance vectors of a person handle, oldest stamp first.

    The handle's own vectors, from its image, always count: a track's
    collection keeps only its newest samples, so it may have dropped them.
    Added to them are the samples VisionFeed collected for the handle's
    track generation, every face the person showed while the robot talked
    to them, without the sample of the handle's own image, which would
    count twice. A generation that is no longer live has no samples: its
    track was dropped, and the track id may name someone else now. The
    handle of an untracked person has its own vectors only.
    """
    if payload.track is None:
        collected = CollectedVectors(faces=(), appearances=())
    else:
        collected = feed.collected_vectors(payload.track)
    return (
        _with_own(collected.faces, payload.face, payload.stamp_ns),
        _with_own(collected.appearances, payload.appearance, payload.stamp_ns),
    )


def _with_own(
    collected: Sequence[Sample], own: ModelVector | None, stamp_ns: int
) -> list[ModelVector]:
    samples = [sample for sample in collected if sample.stamp_ns != stamp_ns]
    if own is not None:
        samples.append(Sample(stamp_ns, own))
    samples.sort(key=lambda sample: sample.stamp_ns)
    return [sample.vector for sample in samples]


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
