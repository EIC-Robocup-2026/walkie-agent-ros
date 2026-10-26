"""Pinhole projection of map-frame points, and the person figure that oracle_perception draws.

oracle_perception publishes the detections that walkie-vision would make of
a simulated Scene (walkie_agent.sim.scene). This module is their geometry,
in numpy and without ROS, so a scenario's tests can check what the camera
sees from a pose without TF or a node.

A transform is a 4x4 matrix that takes map-frame points into the camera's
optical frame (z out of the lens, x to the image's right, y down), the
inverse of the camera's pose in the map. `k` is camera_info.k, the 3x3
intrinsic matrix row by row. Lens distortion (camera_info.d) is ignored, as
for an ideal pinhole camera such as MuJoCo's. A box is x1, y1, x2, y2 in
pixels, the layout of walkie_agent.perception.seating.BBox, which is not
imported here because that module needs ROS messages.

A person is a flat figure: the 17 COCO keypoints as (lateral, height)
offsets in the vertical plane through the person's position that is
perpendicular to the camera's horizontal ray to them. The figure always
shows its full width, and a person who turns away differs only in which
side of the image their left is on. The gesture rules of
walkie_agent.perception.gestures read no more than that: heights and
sideways offsets within one person.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from walkie_agent.perception.gestures import COCO_KEYPOINTS
from walkie_agent.sim.scene import STANDARD_HEIGHT_M, MapBox, ScenePerson

Box = tuple[float, float, float, float]

# A point nearer than this along the optical axis is behind the camera.
MIN_DEPTH_M = 0.2
# The face box is the projection of a square of this side around the face's
# centre. It does not scale with the person: faces vary less than heights.
FACE_SIDE_M = 0.18
# The person box reaches this far above the head, as a detector's box
# includes the hair.
BOX_ABOVE_HEAD_M = 0.10
# Each side of the person box is padded by this share of the keypoints'
# width, as a detector's box includes the arms and clothes around them.
BOX_SIDE_PAD = 0.10
# A sitting person's hips are this far above the top of the seat.
HIP_ABOVE_SEAT_M = 0.10

# The reference figure, a 1.70 m person standing, in meters above the floor.
# Every height scales linearly with the person's height.
_FACE = 1.60
_HEAD_TOP = 1.70
_SHOULDER = 1.42
_HIP = 0.90
_KNEE = 0.48
_ANKLE = 0.08
# A sitting person leans back, so the face is 0.59 m and the shoulders 0.47 m
# above the hips, not 0.70 and 0.52 as when standing. On a seat whose top is
# 0.46 m, the hips are at 0.56 m and the face's centre at 1.15 m. The knees are
# just below the hips: the thighs point at the camera, and their small drop is
# what walkie_agent.perception.gestures reads as sitting. The shins hang from
# the knees down to the floor, or short of it on a high seat. Their drop from
# the hips is then at most 0.47 m, under the 1.2 torso heights (0.56 m) from
# which the rules see legs extended and a person standing.
_SEATED_FACE = 0.59
_SEATED_SHOULDER = 0.47
_SEATED_KNEE_DROP = 0.05
_SHIN = 0.42

_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINTS)}
# The keypoints of the face, whose scores drop when the face is turned away.
FACE_KEYPOINTS: tuple[int, ...] = tuple(
    _INDEX[name] for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
)

# The 12 edges of a box whose 8 corners are numbered by the bits x, y, z: two
# corners share an edge when their numbers differ in one bit.
_BOX_EDGES = tuple((i, i | bit) for bit in (1, 2, 4) for i in range(8) if not i & bit)
_CORNER_SIGNS = np.array(
    [[1.0 if i & bit else -1.0 for bit in (1, 2, 4)] for i in range(8)], dtype=float
)


class Figure(NamedTuple):
    """A person's keypoints in their plane, before projection."""

    # Shape (17, 2), in COCO order: the offset towards the image's right and
    # the height above the floor, in meters.
    points: np.ndarray
    # The height of the face's centre and of the top of the head.
    face_height: float
    head_top: float


class PersonView(NamedTuple):
    """A person as the camera sees them, in pixels and not yet clipped to the image."""

    # Shape (17, 2), in COCO order; NaN for a point behind the camera.
    keypoints: np.ndarray
    # The bounds of the keypoints in front of the camera, padded by
    # BOX_SIDE_PAD of their width on each side, and from BOX_ABOVE_HEAD_M
    # above the head.
    box: Box
    # The bounds of the FACE_SIDE_M square around the face, or None when
    # part of it is behind the camera.
    face_box: Box | None


def to_camera(points_map: Sequence | np.ndarray, camera_from_map: np.ndarray) -> np.ndarray:
    """Map-frame points, shape (N, 3), in the camera's optical frame."""
    points = np.asarray(points_map, dtype=float).reshape(-1, 3)
    transform = np.asarray(camera_from_map, dtype=float)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project(
    points_map: Sequence | np.ndarray, camera_from_map: np.ndarray, k: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """The pixels (uv, shape (N, 2)) and depths (shape (N,)) of map-frame points.

    The depth is the distance along the optical axis. A point with a depth
    below MIN_DEPTH_M is behind the camera, and its pixel is NaN: the
    pinhole formula would put it on the far side of the image.
    """
    points = to_camera(points_map, camera_from_map)
    uv = _pixels(points, k)
    uv[points[:, 2] < MIN_DEPTH_M] = np.nan
    return uv, points[:, 2]


def camera_position(camera_from_map: np.ndarray) -> np.ndarray:
    """The camera's optical centre in the map frame, shape (3,)."""
    transform = np.asarray(camera_from_map, dtype=float)
    return -transform[:3, :3].T @ transform[:3, 3]


def camera_from_map_at(
    x: float, y: float, yaw: float, base_from_camera: np.ndarray
) -> np.ndarray:
    """The camera <- map transform of a camera on a base at (x, y, yaw) in the map.

    `base_from_camera` is the camera's optical frame in the base's, as TF's
    base_footprint <- camera transform.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    map_from_base = np.array(
        [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    return np.linalg.inv(map_from_base @ np.asarray(base_from_camera, dtype=float))


def in_image(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """Per pixel of `uv`, shape (N, 2): whether it is finite and inside the image."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    with np.errstate(invalid="ignore"):
        return (
            np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] <= width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] <= height)
        )


def clip_box(box: Sequence[float], width: int, height: int) -> Box | None:
    """The part of `box` inside the image, or None when no part of it is inside."""
    x1, y1 = max(float(box[0]), 0.0), max(float(box[1]), 0.0)
    x2, y2 = min(float(box[2]), float(width)), min(float(box[3]), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def box_corners(box: MapBox) -> np.ndarray:
    """The 8 corners of a map-frame box, shape (8, 3)."""
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    local = _CORNER_SIGNS * (np.asarray(box.size, dtype=float) / 2)
    return local @ rotation.T + np.asarray(box.center, dtype=float)


def project_box(box: MapBox, camera_from_map: np.ndarray, k: Sequence[float]) -> Box | None:
    """The pixel bounds of a map-frame box, not clipped to the image.

    The part of the box behind the camera is cut off first: each edge that
    crosses the plane at MIN_DEPTH_M ends there. A sofa beside the camera
    then still has a box, its visible part. None when all of it is behind.
    """
    corners = to_camera(box_corners(box), camera_from_map)
    front = corners[:, 2] >= MIN_DEPTH_M
    points = [corners[front]]
    for i, j in _BOX_EDGES:
        if front[i] != front[j]:
            a, b = corners[i], corners[j]
            t = (MIN_DEPTH_M - a[2]) / (b[2] - a[2])
            points.append((a + t * (b - a)).reshape(1, 3))
    visible = np.concatenate(points)
    if len(visible) == 0:
        return None
    return _bounds(_pixels(visible, k))


def person_figure(person: ScenePerson, toward_camera: bool = True) -> Figure:
    """The person's keypoints in their plane, for their height, posture and gesture.

    `toward_camera` says whether the person's front is turned to the camera.
    A person facing the camera has their left on the image's right, as COCO's
    left is the person's own; one turned away has it on the image's left.
    pointing_left and pointing_right point to that side of the image either
    way, with the arm on that side.
    """
    scale = person.height / STANDARD_HEIGHT_M
    if person.posture == "sitting":
        hip = person.seat_top + HIP_ABOVE_SEAT_M
        shoulder = hip + _SEATED_SHOULDER * scale
        face = hip + _SEATED_FACE * scale
        knee = hip - _SEATED_KNEE_DROP * scale
        ankle = max(knee - _SHIN * scale, _ANKLE * scale)
    else:
        hip, shoulder, face, knee = (h * scale for h in (_HIP, _SHOULDER, _FACE, _KNEE))
        ankle = _ANKLE * scale
    head_top = face + (_HEAD_TOP - _FACE) * scale

    # Laterals towards the person's own left, mirrored below for a person
    # turned away. +1 is the person's left side, -1 their right.
    points: dict[str, tuple[float, float]] = {"nose": (0.0, face - 0.02 * scale)}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        points[f"{side}_eye"] = (sign * 0.032 * scale, face + 0.02 * scale)
        points[f"{side}_ear"] = (sign * 0.075 * scale, face)
        points[f"{side}_shoulder"] = (sign * 0.19 * scale, shoulder)
        points[f"{side}_hip"] = (sign * 0.10 * scale, hip)
        points[f"{side}_knee"] = (sign * 0.10 * scale, knee)
        points[f"{side}_ankle"] = (sign * 0.10 * scale, ankle)
        if person.posture == "sitting":
            # The hands rest on the thighs.
            points[f"{side}_elbow"] = (sign * 0.22 * scale, shoulder - 0.27 * scale)
            points[f"{side}_wrist"] = (sign * 0.20 * scale, hip + 0.02 * scale)
        else:
            points[f"{side}_elbow"] = (sign * 0.22 * scale, shoulder - 0.29 * scale)
            points[f"{side}_wrist"] = (sign * 0.24 * scale, shoulder - 0.58 * scale)

    gesture = person.gesture
    if gesture in ("raising_left_arm", "raising_right_arm"):
        # Bent at the elbow, the hand 0.34 m above the shoulder: a straight
        # arm would leave the image of a camera at 1.1 m that sees the
        # person from 1.2 m.
        side = "left" if gesture == "raising_left_arm" else "right"
        sign = 1.0 if side == "left" else -1.0
        points[f"{side}_elbow"] = (sign * 0.24 * scale, shoulder + 0.16 * scale)
        points[f"{side}_wrist"] = (sign * 0.26 * scale, shoulder + 0.34 * scale)
    elif gesture == "waving":
        # The right hand held up beside the head. One image cannot show the
        # motion, so the gesture rules take any raised hand for a wave. The
        # hand is 0.30 m above the shoulder, more than the 0.26 m that the
        # pointing rule still takes for a level arm.
        points["right_elbow"] = (-0.44 * scale, shoulder - 0.05 * scale)
        points["right_wrist"] = (-0.42 * scale, shoulder + 0.30 * scale)
    elif gesture in ("pointing_left", "pointing_right"):
        # The arm held out level towards that side of the image. In the
        # person's own laterals, the image's right is their left only while
        # they face the camera.
        image_side = 1.0 if gesture == "pointing_right" else -1.0
        sign = image_side if toward_camera else -image_side
        side = "left" if sign > 0 else "right"
        points[f"{side}_elbow"] = (sign * 0.47 * scale, shoulder - 0.03 * scale)
        points[f"{side}_wrist"] = (sign * 0.75 * scale, shoulder - 0.05 * scale)

    figure = np.array([points[name] for name in COCO_KEYPOINTS], dtype=float)
    if not toward_camera:
        figure[:, 0] *= -1.0
    return Figure(points=figure, face_height=face, head_top=head_top)


def project_person(
    person: ScenePerson, camera_from_map: np.ndarray, k: Sequence[float]
) -> PersonView | None:
    """The person's keypoints and boxes in the image.

    None when every keypoint is behind the camera (see project), or when the
    person stands at the camera. The figure stands in the vertical plane
    through (person.x, person.y) that faces the camera, so its laterals run
    along the image's rows.
    """
    camera = camera_position(camera_from_map)
    ray = np.array([person.x - camera[0], person.y - camera[1]])
    distance = float(np.hypot(*ray))
    if distance < 1e-6:
        # Standing in the camera: no plane faces it.
        return None
    # The image's right for a camera that looks along the ray: the ray turned
    # by -90 degrees about the vertical.
    right = np.array([ray[1], -ray[0]]) / distance
    facing = np.array([np.cos(person.facing), np.sin(person.facing)])
    # The front is turned to the camera when it points against the ray.
    figure = person_figure(person, toward_camera=bool(facing @ ray <= 0.0))

    def in_plane(laterals: np.ndarray, heights: np.ndarray) -> np.ndarray:
        xy = np.array([person.x, person.y]) + np.outer(laterals, right)
        return np.column_stack([xy, heights])

    keypoints, _ = project(
        in_plane(figure.points[:, 0], figure.points[:, 1]), camera_from_map, k
    )
    seen = np.isfinite(keypoints).all(axis=1)
    if not seen.any():
        return None
    x1, y1 = keypoints[seen].min(axis=0)
    x2, y2 = keypoints[seen].max(axis=0)
    pad = BOX_SIDE_PAD * (x2 - x1)
    (above_head,), _ = project(
        in_plane(np.array([0.0]), np.array([figure.head_top + BOX_ABOVE_HEAD_M])),
        camera_from_map,
        k,
    )
    if np.isfinite(above_head).all():
        y1 = min(y1, above_head[1])
    box = (float(x1 - pad), float(y1), float(x2 + pad), float(y2))

    half = FACE_SIDE_M / 2
    square, _ = project(
        in_plane(
            np.array([-half, half, half, -half]),
            figure.face_height + np.array([half, half, -half, -half]),
        ),
        camera_from_map,
        k,
    )
    face_box = _bounds(square) if np.isfinite(square).all() else None
    return PersonView(keypoints=keypoints, box=box, face_box=face_box)


def _pixels(points_camera: np.ndarray, k: Sequence[float]) -> np.ndarray:
    """Optical-frame points with a positive depth in pixels; NaN for the others."""
    intrinsics = np.asarray(k, dtype=float).reshape(3, 3)
    homogeneous = points_camera @ intrinsics.T
    uv = np.full((len(points_camera), 2), np.nan)
    ahead = points_camera[:, 2] > 0
    uv[ahead] = homogeneous[ahead, :2] / homogeneous[ahead, 2:3]
    return uv


def _bounds(uv: np.ndarray) -> Box:
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    return (float(x1), float(y1), float(x2), float(y2))
