"""The map-frame yaw of a pixel's ray, how far away a person stands, and an angle wrap.

RosVision turns each person into a bearing with these: the rays through the person's
torso_pixels(), or their box's centre, in the camera's optical frame, are rotated into
the map frame with TF at the image's stamp, and their mean_yaw() is the person's
direction from the camera. With the ankles in view, floor_range() is the person's
distance from the camera, where the rays through the ankles meet the plane of an
ankle above the floor, and yaw_from() turns direction and distance into the
person's direction from the base's turning centre. This module imports no ROS, so
the fake and the tests use it without a graph.

Ported in part from the experiment/hri-agent branch at 61d30e8 (perception/geometry.py),
with three changes. That version kept only the box's column and set the ray's row to
the image's middle. The row changes the yaw as soon as the camera is pitched: with the
head camera 9 degrees down, a pixel near a corner of the 110 degree image comes out
about 4 degrees off. So the ray here goes through the pixel itself, column and row.
The box's centre moves with a raised or pointing arm, by 1.7 degrees for the
simulated guest raising their left arm, so the torso's keypoints come first. And the
camera sits 0.275 m ahead of the turning centre, so a person to the side is wider
from the camera than from the centre: the simulated guest, 1.8 m away, at 39.6
degrees against 35.6, and the first run's turn by the camera's direction ended 6.4
degrees past them. The distance from the ankles corrects that.
"""

import math
from collections.abc import Sequence

from walkie_agent.perception.gestures import COCO_KEYPOINTS

# A ray this close to vertical in the map frame has no meaningful yaw, as from a
# camera that looks straight down.
_MIN_HORIZONTAL = 1e-6
# The shoulders and the hips. A raised or pointing arm widens the box to one side,
# and these do not move with it.
_TORSO = tuple(
    COCO_KEYPOINTS.index(name)
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
)
# Fewer seen torso keypoints than this, and the box's centre stands in.
_MIN_TORSO_KEYPOINTS = 2
_ANKLES = tuple(COCO_KEYPOINTS.index(name) for name in ("left_ankle", "right_ankle"))
# The height of an ankle keypoint above the floor: the simulation's figure puts it
# there, and it is about where the ankle bone of an adult is.
ANKLE_HEIGHT_M = 0.08
# Farther than this, a ray from the head camera meets the floor so shallowly that a
# pixel of error moves the point by tens of centimetres, and the camera's own
# direction stands in.
MAX_FLOOR_RANGE_M = 8.0
# A ray that falls less steeply than this, per unit of length, never meets the floor
# within MAX_FLOOR_RANGE_M of a camera at a robot's height.
_MIN_FALL = 1e-6
# A point this close to the turning centre has no direction from it.
_MIN_DISTANCE_M = 1e-3


def torso_pixels(
    keypoints: Sequence[float], scores: Sequence[float], min_score: float
) -> list[tuple[float, float]]:
    """The pixels of the shoulders and hips scored at least `min_score`.

    `keypoints` is walkie-vision's flat x0, y0, x1, y1, ... list in COCO order, and
    `scores` one score per keypoint. Empty with fewer than two such keypoints, or
    with lists of the wrong length.
    """
    seen = _seen(keypoints, scores, min_score, _TORSO)
    return seen if len(seen) >= _MIN_TORSO_KEYPOINTS else []


def ankle_pixels(
    keypoints: Sequence[float], scores: Sequence[float], min_score: float
) -> list[tuple[float, float]]:
    """The pixels of the ankles scored at least `min_score`; empty with lists of the wrong length."""
    return _seen(keypoints, scores, min_score, _ANKLES)


def _seen(
    keypoints: Sequence[float], scores: Sequence[float], min_score: float, which: Sequence[int]
) -> list[tuple[float, float]]:
    if len(keypoints) != 2 * len(COCO_KEYPOINTS) or len(scores) != len(COCO_KEYPOINTS):
        return []
    return [
        (float(keypoints[2 * i]), float(keypoints[2 * i + 1]))
        for i in which
        if scores[i] >= min_score
        and math.isfinite(keypoints[2 * i])
        and math.isfinite(keypoints[2 * i + 1])
    ]


def mean_yaw(
    pixels: Sequence[tuple[float, float]], k: Sequence[float], rotation: Sequence[float]
) -> float | None:
    """The circular mean of pixel_yaw() over `pixels`, or None when no pixel has a yaw.

    The mean of the rays' yaws, not the yaw of the mean pixel: a column grows with
    the tangent of its angle, so the pixel midway between two shoulders lies off the
    body's middle, by 0.5 degrees for the simulated guest 40 degrees to the side.
    """
    yaws = [yaw for yaw in (pixel_yaw(u, v, k, rotation) for u, v in pixels) if yaw is not None]
    if not yaws:
        return None
    return math.atan2(sum(math.sin(yaw) for yaw in yaws), sum(math.cos(yaw) for yaw in yaws))


def floor_range(
    pixels: Sequence[tuple[float, float]],
    k: Sequence[float],
    rotation: Sequence[float],
    origin: Sequence[float],
    height_m: float = ANKLE_HEIGHT_M,
) -> float | None:
    """The mean distance along the floor from the camera to where the rays meet z = `height_m`.

    `origin` is the camera's optical centre in the map frame, x, y, z. A ray that
    does not fall, or meets the plane more than MAX_FLOOR_RANGE_M away, is left out,
    as are all of them for a camera at or below the plane. None when no ray is left.
    """
    above = origin[2] - height_m
    if not (math.isfinite(above) and above > 0):
        return None
    ranges = []
    for u, v in pixels:
        ray = pixel_ray(u, v, k, rotation)
        if ray is None:
            continue
        x, y, z = ray
        length = math.sqrt(x * x + y * y + z * z)
        if not -z > _MIN_FALL * length:
            continue
        distance = above / -z * math.hypot(x, y)
        if distance <= MAX_FLOOR_RANGE_M:
            ranges.append(distance)
    return sum(ranges) / len(ranges) if ranges else None


def yaw_from(
    centre: Sequence[float], origin: Sequence[float], yaw: float, distance: float
) -> float | None:
    """The map yaw from `centre` (x, y) of the point `distance` from `origin` along `yaw`.

    None when that point is at the centre itself.
    """
    x = origin[0] + distance * math.cos(yaw) - centre[0]
    y = origin[1] + distance * math.sin(yaw) - centre[1]
    if math.hypot(x, y) < _MIN_DISTANCE_M:
        return None
    return math.atan2(y, x)


def pixel_yaw(
    u: float,
    v: float,
    k: Sequence[float],
    rotation: Sequence[float],
) -> float | None:
    """The yaw in the map frame, in radians, of the ray through pixel (u, v).

    `k` is the image's camera_info.k, the row-major 3x3 intrinsics, and `rotation`
    the map <- optical frame rotation at the image's stamp, as a quaternion x, y, z,
    w. Returns None when `k` has no focal lengths, the quaternion has no length, the
    ray is vertical in the map frame, or a number is not finite.
    """
    ray = pixel_ray(u, v, k, rotation)
    if ray is None:
        return None
    x, y, _ = ray
    if math.hypot(x, y) < _MIN_HORIZONTAL:
        return None
    return math.atan2(y, x)


def pixel_ray(
    u: float, v: float, k: Sequence[float], rotation: Sequence[float]
) -> tuple[float, float, float] | None:
    """The direction in the map frame of the ray through pixel (u, v), not unit length.

    As pixel_yaw() takes `k` and `rotation`. None when `k` has no focal lengths, the
    quaternion has no length or a number is not finite.
    """
    if len(k) < 6:
        return None
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    # An all-zero k is a camera_info that was never filled in, and a yaw from it
    # would be a guess. `not > 0` also catches NaN.
    if not (fx > 0 and fy > 0):
        return None
    # In the optical frame z points out of the lens, x to the image's right and y down.
    rotated = rotate(rotation, ((u - cx) / fx, (v - cy) / fy, 1.0))
    if rotated is None or not all(math.isfinite(value) for value in rotated):
        return None
    return rotated


def rotate(
    q: Sequence[float], v: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    """The vector `v` turned by the quaternion `q` (x, y, z, w); None for a zero quaternion."""
    qx, qy, qz, qw = q
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not norm > 0:
        return None
    # Normalised first: the formula below holds for unit quaternions only, and a
    # published transform need not be exactly unit.
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    vx, vy, vz = v
    # v' = v + w t + q_vec x t, with t = 2 (q_vec x v).
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def wrap_deg(angle_deg: float) -> float:
    """`angle_deg` in degrees, wrapped to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0
