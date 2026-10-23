"""The map-frame heading of a box in a camera image.

find_people and find_seats keep, with each handle, the direction of the
person or seat in the map frame, so that the base can turn to it after the
image is gone; the gaze loop computes its setpoint with the same function.
Only the box's centre column matters: the ray through it is turned from the
camera's optical frame into the map frame with TF at the image's stamp, and
the ray's yaw is the heading. The function is pure apart from its TF lookup,
so a test can give it a buffer with a hand-built transform.
"""

import math
from collections.abc import Sequence

from geometry_msgs.msg import Quaternion
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


def box_heading(
    bbox: BoundingBox2D | Sequence[float],
    camera_info: CameraInfo,
    header: Header,
    tf_buffer: Buffer,
    timeout_s: float = 0.2,
) -> float | None:
    """The yaw in the map frame, in radians, of the ray through the box's centre column.

    `bbox` is a message box, or an x1, y1, x2, y2 tuple such as a sofa
    place's, in pixels of the image that `camera_info` describes. `header`
    is that image's: the camera's optical frame and the stamp. Returns None
    when TF has no `map <- header.frame_id` transform at the stamp within
    `timeout_s`, or when `camera_info` has no focal length.
    """
    u = _centre_column(bbox)
    fx, cx = camera_info.k[0], camera_info.k[2]
    # An all-zero k means a camera_info that was never filled in, and a
    # heading from it would be a guess.
    if not fx > 0:
        return None
    # In the optical frame z points out of the lens and x to the image's
    # right. The row does not change the yaw, so the ray keeps y at 0.
    ray = ((u - cx) / fx, 0.0, 1.0)
    try:
        stamped = tf_buffer.lookup_transform(
            MAP_FRAME,
            header.frame_id,
            Time.from_msg(header.stamp),
            timeout=Duration(seconds=timeout_s),
        )
    except TransformException:
        return None
    rotated = _rotate(stamped.transform.rotation, ray)
    if rotated is None:
        return None
    x, y, _ = rotated
    if math.hypot(x, y) < _MIN_HORIZONTAL:
        return None
    return math.atan2(y, x)


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
