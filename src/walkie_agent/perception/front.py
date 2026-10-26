"""Who stands close in front of the robot: a clear, large face near the base's heading.

find_people(in_front=True) keeps only such people, so that at the door it
finds the guest and leaves out a passer-by behind them or someone beside
the robot. A face's size tells how near the person is, and the angle
between the face's heading and the base's tells whether they are in front.
Both are judged at the image's stamp, so the localization error cancels
out of the angle: the face's heading and the base's yaw come from the same
TF.

The face's size is face_trusted's measure, face_size_rad, so that the
recognition gate and this rule use one definition. Plain logic: no ROS
beyond the CameraInfo message.
"""

import math
from dataclasses import dataclass

from sensor_msgs.msg import CameraInfo

from walkie_agent.perception.seating import BBox

# Why find_people(in_front=True) left a person out, in the order of the
# checks: the first that fails names the person's reason. Each reads after
# a count ("1 too far") and after a person's description.
NO_CLEAR_FACE = "with no clear face"
TOO_FAR = "too far"
TO_THE_SIDE = "to the side"
REASONS = (NO_CLEAR_FACE, TOO_FAR, TO_THE_SIDE)


@dataclass(frozen=True)
class FrontConfig:
    """Who counts as close in front of the robot. New, not from walkie-agent-v2.

    Both values were tuned in simulation only, in chula_pat14_6 at the door
    pose, and are to tune on the robot. config.toml can set both, in
    [walkie.perception.front] (walkie_agent.tuning).
    """

    # The smallest face, as face_size_rad() measures it. Within Nav2's stop
    # tolerance of 0.1 m of the door pose, the guest's face measures at
    # least 0.129 rad, and 0.121 within AMCL's 0.2 m error. The onlooker in
    # the hallway behind them measures at most 0.067 rad, and 0.069. The
    # oracle's box noise moves the measure by about 0.009 rad, so 0.10
    # leaves the guest 2.3 noise widths above it at 0.2 m and the onlooker
    # 3.4 below. On the robot it is about 107 px at the ZED 2i's nominal fx
    # of 1066 px, and SCRFD's tighter face box would move it.
    min_face_rad: float = 0.10
    # The largest angle, in degrees, between a face's heading and the
    # base's yaw at the image's stamp. Within 0.1 m of the door pose the
    # pacing guest stays within 18.0 degrees of the base's yaw, and 22.1
    # within 0.2 m. The onlooker beside the robot by the TV desk is at least
    # 37.8 degrees off, and 29.9.
    half_angle_deg: float = 25.0

    def __post_init__(self) -> None:
        if not self.min_face_rad > 0:
            raise ValueError("FrontConfig.min_face_rad must be positive.")
        if not 0 < self.half_angle_deg < 180:
            raise ValueError("FrontConfig.half_angle_deg must lie in (0, 180).")


DEFAULT_FRONT_CONFIG = FrontConfig()


def face_size_rad(face_bbox: BBox, camera_info: CameraInfo) -> float | None:
    """The face's size as an angle, sqrt((w / fx) * (h / fy)) radians; None without a focal length.

    `face_bbox` is x1, y1, x2, y2 in pixels of the image `camera_info`
    describes. A box with no area measures 0.
    """
    # rclpy holds k as a numpy array; floats keep the result a plain float.
    fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
    # An all-zero k means a camera_info that was never filled in, and the
    # face's size is then unknown.
    if not (fx > 0 and fy > 0):
        return None
    x1, y1, x2, y2 = face_bbox
    # The width over fx and the height over fy, so an image scaled more in
    # one direction than the other measures the same face.
    area_rad2 = ((x2 - x1) / fx) * ((y2 - y1) / fy)
    return math.sqrt(max(area_rad2, 0.0))


def front_reason(
    face_det_score: float,
    face_rad: float | None,
    heading_map: float | None,
    base_yaw: float,
    min_face_det_score: float,
    config: FrontConfig = DEFAULT_FRONT_CONFIG,
) -> str | None:
    """None for a person close in front of the robot; else why not, one of REASONS.

    `face_rad` is face_size_rad() of the person's face box, None without
    one; `heading_map` is the face's heading in the map frame, and
    `base_yaw` the base's yaw at the same stamp, both in radians.
    `min_face_det_score` is the feed's, so that a face kept here can also
    be recognized. Values on a limit are kept, as face_trusted keeps them.
    """
    if face_det_score < min_face_det_score or face_rad is None:
        return NO_CLEAR_FACE
    if face_rad < config.min_face_rad:
        return TOO_FAR
    # A face whose direction is unknown is not known to be in front.
    if heading_map is None:
        return TO_THE_SIDE
    off = heading_map - base_yaw
    off_deg = math.degrees(math.atan2(math.sin(off), math.cos(off)))
    if abs(off_deg) > config.half_angle_deg:
        return TO_THE_SIDE
    return None
