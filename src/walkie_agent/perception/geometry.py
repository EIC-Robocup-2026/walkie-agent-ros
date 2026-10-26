"""The map-frame heading of a box in a camera image.

find_people and find_seats keep, with each handle, the direction of the
person or seat in the map frame, so that the base can turn to it after the
image is gone; the gaze loop computes its setpoint with the same function.
Only the box's centre column matters: the ray through it is turned from the
camera's optical frame into the map frame with TF at the image's stamp, and
the ray's yaw is the heading.

The lookup and the ray are two functions. camera_transform looks up the
transform once per image, and ray_heading, which is pure, turns each box's
ray with it. A caller with several boxes of one image therefore waits for TF
at most once, and a ray with no yaw, such as the central ray of a camera
that looks straight down, leaves the other boxes' headings intact.
box_heading composes the two for a single box.
"""

import math
from collections.abc import Sequence

from geometry_msgs.msg import Quaternion, TransformStamped
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException
from vision_msgs.msg import BoundingBox2D

# The fixed frame of every heading. The same name as
# walkie_agent.tools.navigation.MAP_FRAME, repeated so that perception does
# not import the LangChain tools.
MAP_FRAME = "map"

# A ray this close to vertical in the map frame has no meaningful yaw, as
# from a camera that looks straight down.
_MIN_HORIZONTAL = 1e-6


def camera_transform(
    header: Header, tf_buffer: Buffer, timeout_s: float = 0.2
) -> TransformStamped | None:
    """The transform `map <- header.frame_id` at `header.stamp`, the camera's pose in the image.

    `header` is the image's: the camera's optical frame and the stamp.
    Returns None when TF has no such transform within `timeout_s`, for
    example when the stamp is older than the buffer or newer than its last
    transform.
    """
    try:
        return tf_buffer.lookup_transform(
            MAP_FRAME,
            header.frame_id,
            Time.from_msg(header.stamp),
            timeout=Duration(seconds=timeout_s),
        )
    except TransformException:
        return None


def ray_heading(
    bbox: BoundingBox2D | Sequence[float],
    camera_info: CameraInfo,
    transform: TransformStamped,
) -> float | None:
    """The yaw in the map frame, in radians, of the ray through the box's centre column.

    `bbox` is a message box, or an x1, y1, x2, y2 tuple such as a sofa
    place's, in pixels of the image that `camera_info` describes, and
    `transform` is that image's camera_transform(). Returns None when
    `camera_info` has no focal length, or when the ray is vertical in the
    map frame and so has no yaw. That depends on the box: with the camera
    pitched straight down, only the central column's ray is vertical.
    """
    intrinsics = _intrinsics(camera_info)
    if intrinsics is None:
        return None
    fx, cx = intrinsics
    # In the optical frame z points out of the lens and x to the image's
    # right. The row does not change the yaw, so the ray keeps y at 0.
    ray = ((_centre_column(bbox) - cx) / fx, 0.0, 1.0)
    rotated = _rotate(transform.transform.rotation, ray)
    if rotated is None:
        return None
    x, y, _ = rotated
    if math.hypot(x, y) < _MIN_HORIZONTAL:
        return None
    return math.atan2(y, x)


def box_heading(
    bbox: BoundingBox2D | Sequence[float],
    camera_info: CameraInfo,
    header: Header,
    tf_buffer: Buffer,
    timeout_s: float = 0.2,
) -> float | None:
    """ray_heading() of one box, with the camera_transform() at the image's stamp.

    `header` is the image's: the camera's optical frame and the stamp.
    Returns None when TF has no `map <- header.frame_id` transform at the
    stamp within `timeout_s`, or as ray_heading() does. For several boxes of
    one image, call camera_transform() once and ray_heading() per box.
    """
    # Checked before the lookup, so that a camera_info without a focal
    # length, which can give no heading, does not wait out the TF timeout.
    if _intrinsics(camera_info) is None:
        return None
    transform = camera_transform(header, tf_buffer, timeout_s)
    if transform is None:
        return None
    return ray_heading(bbox, camera_info, transform)


def _intrinsics(camera_info: CameraInfo) -> tuple[float, float] | None:
    """The focal length fx and the centre column cx, in pixels; None without a focal length."""
    fx, cx = camera_info.k[0], camera_info.k[2]
    # An all-zero k means a camera_info that was never filled in, and a
    # heading from it would be a guess.
    if not fx > 0:
        return None
    return float(fx), float(cx)


def _centre_column(bbox: BoundingBox2D | Sequence[float]) -> float:
    if isinstance(bbox, BoundingBox2D):
        return bbox.center.position.x
    x1, _, x2, _ = bbox
    return (x1 + x2) / 2


def _rotate(
    q: Quaternion, v: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    """The vector `v` turned by the quaternion `q`; None for a zero quaternion."""
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm == 0.0:
        return None
    # Normalised first, because the formula below holds for unit quaternions
    # only and a published transform need not be exactly unit.
    qx, qy, qz, qw = q.x / norm, q.y / norm, q.z / norm, q.w / norm
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
