"""Gestures and postures from a person's COCO keypoints, ported from walkie-agent-v2.

walkie-vision publishes 17 body keypoints per person, and find_people turns
them into the names that tasks use, such as "waving" or "sitting"
(perception.md, People). The rules are walkie-agent-v2's
tasks/GPSR/gestures.py, read from the message's keypoints and
keypoint_scores arrays instead of v2's PersonPose.

The conventions are v2's. Image coordinates: y grows downward, so a raised
hand has the smaller y. A left or right arm is the person's own side, as in
COCO's keypoint names. pointing_left and pointing_right are directions in
the image, the robot's view, because that is the direction the robot can act
on. Margins are shares of the torso height or the shoulder width, so the
same rule holds for a person near the camera and one far away.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

# The 17 COCO keypoints in the order of PersonDetection.keypoints, which
# holds x0, y0, x1, y1, and so on.
COCO_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class GestureConfig:
    """The thresholds of the gesture rules.

    The defaults are walkie-agent-v2's, tuned on the robot; the name in each
    comment is the v2 environment variable that held the value.
    """

    # A keypoint with a lower score is treated as not seen: its position is
    # a guess. GPSR_KP_MIN_CONF, tuned on the robot.
    min_keypoint_score: float = 0.3
    # How far a wrist must be above its shoulder for a raised arm, as a share
    # of the body scale. GPSR_ARM_RAISE_MARGIN, tuned on the robot.
    arm_raise_margin: float = 0.15
    # How far sideways from its shoulder a pointing wrist must be, as a share
    # of the torso height. GPSR_POINT_X_MARGIN, tuned on the robot.
    point_x_margin: float = 0.6
    # How far above or below its shoulder a pointing wrist may be, as a share
    # of the torso height; further is a raised or hanging arm.
    # GPSR_POINT_Y_TOL, tuned on the robot.
    point_y_tolerance: float = 0.5
    # A thigh whose drop from hip to knee is below this share of the torso
    # height is roughly horizontal, so the person sits. GPSR_SIT_RATIO, tuned
    # on the robot.
    sit_ratio: float = 0.6
    # Legs whose drop from hip to ankle is above this share of the torso
    # height are extended, so the person stands. GPSR_STAND_RATIO, tuned on
    # the robot.
    stand_ratio: float = 1.2
    # A body whose seen keypoints span this many times more across than down
    # lies down. GPSR_LIE_RATIO, tuned on the robot.
    lie_ratio: float = 1.3


DEFAULT_GESTURE_CONFIG = GestureConfig()

Point = tuple[float, float]


class _Body:
    """One person's keypoints, keeping only those scored at or above the threshold."""

    def __init__(
        self,
        keypoints: Sequence[float],
        keypoint_scores: Sequence[float],
        config: GestureConfig,
    ) -> None:
        # The message's arrays have fixed sizes, so a wrong length is a
        # caller's bug, not a person who is partly hidden.
        if len(keypoints) != 2 * len(COCO_KEYPOINTS) or len(keypoint_scores) != len(
            COCO_KEYPOINTS
        ):
            raise ValueError(
                f"Expected {2 * len(COCO_KEYPOINTS)} keypoint coordinates and "
                f"{len(COCO_KEYPOINTS)} scores, got {len(keypoints)} and "
                f"{len(keypoint_scores)}."
            )
        self.config = config
        self.points: dict[str, Point] = {
            name: (float(keypoints[2 * i]), float(keypoints[2 * i + 1]))
            for i, name in enumerate(COCO_KEYPOINTS)
            if keypoint_scores[i] >= config.min_keypoint_score
        }

    def point(self, name: str) -> Point | None:
        return self.points.get(name)


def _torso_height(body: _Body) -> float | None:
    """The vertical shoulder to hip distance on one side, the scale of most rules.

    A torso of zero height, as of a person lying down, counts as 1 px, so the
    rules still get a usable if small scale. None when no side has both a
    shoulder and a hip.
    """
    for side in ("left", "right"):
        shoulder, hip = body.point(f"{side}_shoulder"), body.point(f"{side}_hip")
        if shoulder is not None and hip is not None:
            return abs(hip[1] - shoulder[1]) or 1.0
    return None


def _scale(body: _Body) -> float | None:
    """The torso height, else the shoulder width.

    A waving person is often framed from the waist up, with the hips out of
    the image, but the shoulders are seen, and their width follows the body's
    scale as well.
    """
    torso = _torso_height(body)
    if torso is not None:
        return torso
    left, right = body.point("left_shoulder"), body.point("right_shoulder")
    if left is not None and right is not None:
        return abs(right[0] - left[0]) or 1.0
    return None


def _arm_raised(body: _Body, side: str) -> bool:
    """The side's wrist is clearly above its shoulder."""
    wrist, shoulder = body.point(f"{side}_wrist"), body.point(f"{side}_shoulder")
    scale = _scale(body)
    if wrist is None or shoulder is None or scale is None:
        return False
    return wrist[1] < shoulder[1] - body.config.arm_raise_margin * scale


def _arm_pointing(body: _Body, direction: str) -> bool:
    """Either arm is held out roughly level toward the image's left or right.

    The wrist must be far from its shoulder sideways and near shoulder height.
    """
    torso = _torso_height(body)
    if torso is None:
        return False
    x_margin = body.config.point_x_margin * torso
    y_tolerance = body.config.point_y_tolerance * torso
    for side in ("left", "right"):
        wrist, shoulder = body.point(f"{side}_wrist"), body.point(f"{side}_shoulder")
        if wrist is None or shoulder is None:
            continue
        if abs(wrist[1] - shoulder[1]) > y_tolerance:
            continue
        dx = wrist[0] - shoulder[0]
        if direction == "right" and dx > x_margin:
            return True
        if direction == "left" and dx < -x_margin:
            return True
    return False


def _waving(body: _Body) -> bool:
    # One image cannot show the motion, so a raised hand on either side counts.
    return _arm_raised(body, "left") or _arm_raised(body, "right")


def _sitting(body: _Body) -> bool:
    torso = _torso_height(body)
    if torso is None:
        return False
    limit = body.config.sit_ratio * torso
    for side in ("left", "right"):
        hip, knee = body.point(f"{side}_hip"), body.point(f"{side}_knee")
        if hip is not None and knee is not None and knee[1] - hip[1] < limit:
            return True
    return False


def _lying_down(body: _Body) -> bool:
    points = list(body.points.values())
    # Three points or fewer do not show the shape of a body.
    if len(points) < 4:
        return False
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x_span, y_span = max(xs) - min(xs), max(ys) - min(ys)
    return x_span > body.config.lie_ratio * max(y_span, 1.0)


def _standing(body: _Body) -> bool:
    torso = _torso_height(body)
    if torso is None:
        return False
    limit = body.config.stand_ratio * torso
    legs_extended = False
    for side in ("left", "right"):
        hip, ankle = body.point(f"{side}_hip"), body.point(f"{side}_ankle")
        if hip is not None and ankle is not None and ankle[1] - hip[1] > limit:
            legs_extended = True
    # A body lying down has almost no torso height, so nearly any drop from
    # hip to ankle passes the leg test; the body's shape rules it out.
    return legs_extended and not _lying_down(body)


_RULES: dict[str, Callable[[_Body], bool]] = {
    "waving": _waving,
    "raising_left_arm": partial(_arm_raised, side="left"),
    "raising_right_arm": partial(_arm_raised, side="right"),
    "pointing_left": partial(_arm_pointing, direction="left"),
    "pointing_right": partial(_arm_pointing, direction="right"),
    "sitting": _sitting,
    "standing": _standing,
    "lying_down": _lying_down,
}

# The names find_people accepts, walkie-agent-v2's eight gesture ids.
GESTURES: tuple[str, ...] = tuple(_RULES)
# The gestures that describe the whole body rather than an arm, so a
# description can name the posture first.
POSTURES: frozenset[str] = frozenset({"sitting", "standing", "lying_down"})


def classify_gestures(
    keypoints: Sequence[float],
    keypoint_scores: Sequence[float],
    config: GestureConfig = DEFAULT_GESTURE_CONFIG,
) -> set[str]:
    """Every gesture in GESTURES that the person shows. Several may hold at once."""
    body = _Body(keypoints, keypoint_scores, config)
    return {name for name, rule in _RULES.items() if rule(body)}


def matches_gesture(
    keypoints: Sequence[float],
    keypoint_scores: Sequence[float],
    gesture: str,
    config: GestureConfig = DEFAULT_GESTURE_CONFIG,
) -> bool:
    """True when the person shows `gesture`. A name not in GESTURES never matches."""
    rule = _RULES.get(gesture)
    return rule is not None and rule(_Body(keypoints, keypoint_scores, config))


def is_waving(
    keypoints: Sequence[float],
    keypoint_scores: Sequence[float],
    config: GestureConfig = DEFAULT_GESTURE_CONFIG,
) -> bool:
    """A hand raised above its shoulder, on either side."""
    return _waving(_Body(keypoints, keypoint_scores, config))


def is_sitting(
    keypoints: Sequence[float],
    keypoint_scores: Sequence[float],
    config: GestureConfig = DEFAULT_GESTURE_CONFIG,
) -> bool:
    """A thigh roughly horizontal: a small drop from hip to knee, against the torso."""
    return _sitting(_Body(keypoints, keypoint_scores, config))


def is_standing(
    keypoints: Sequence[float],
    keypoint_scores: Sequence[float],
    config: GestureConfig = DEFAULT_GESTURE_CONFIG,
) -> bool:
    """Legs extended from hips to ankles, and the body upright."""
    return _standing(_Body(keypoints, keypoint_scores, config))


def is_lying_down(
    keypoints: Sequence[float],
    keypoint_scores: Sequence[float],
    config: GestureConfig = DEFAULT_GESTURE_CONFIG,
) -> bool:
    """The body horizontal: its seen keypoints span more across than down."""
    return _lying_down(_Body(keypoints, keypoint_scores, config))
