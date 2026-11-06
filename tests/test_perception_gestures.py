"""The gesture rules: COCO keypoints in, gesture and posture names out.

No ROS graph. Ported from the experiment/hri-agent branch's
tests/test_perception.py at 61d30e8, whose cases come from walkie-agent-v2's
tests/test_gpsr_gestures.py. Only the message test imports a message type.
"""

import numpy as np
import pytest

from walkie_agent.perception import gestures

_INDEX = {name: i for i, name in enumerate(gestures.COCO_KEYPOINTS)}


def _keypoints(
    points: dict[str, tuple[float, float]], score: float = 0.9
) -> tuple[list[float], list[float]]:
    """The message's keypoints and keypoint_scores for the named points; the rest unseen."""
    keypoints = [0.0] * 34
    scores = [0.0] * 17
    for name, (x, y) in points.items():
        i = _INDEX[name]
        # Floats, because the message's float32 arrays refuse a list with ints.
        keypoints[2 * i], keypoints[2 * i + 1] = float(x), float(y)
        scores[i] = score
    return keypoints, scores


# Upright, arms hanging, legs extended down.
STANDING = _keypoints(
    {
        "left_shoulder": (140, 100), "right_shoulder": (160, 100),
        "left_hip": (140, 200), "right_hip": (160, 200),
        "left_knee": (140, 300), "right_knee": (160, 300),
        "left_ankle": (140, 400), "right_ankle": (160, 400),
        "left_wrist": (140, 200), "right_wrist": (160, 200),
    }
)  # fmt: skip

# Thighs roughly horizontal: knees and ankles tucked up near the hips.
SITTING = _keypoints(
    {
        "left_shoulder": (140, 100), "right_shoulder": (160, 100),
        "left_hip": (140, 200), "right_hip": (160, 200),
        "left_knee": (140, 225), "right_knee": (160, 225),
        "left_ankle": (140, 250), "right_ankle": (160, 250),
    }
)  # fmt: skip

# The left hand raised above the left shoulder.
RAISING_LEFT = _keypoints(
    {
        "left_shoulder": (140, 100), "right_shoulder": (160, 100),
        "left_hip": (140, 200), "right_hip": (160, 200),
        "left_wrist": (140, 40), "right_wrist": (160, 200),
        "left_ankle": (140, 400), "right_ankle": (160, 400),
    }
)  # fmt: skip

# The right arm held out level toward the image's right.
POINTING_RIGHT = _keypoints(
    {
        "left_shoulder": (140, 100), "right_shoulder": (160, 100),
        "left_hip": (140, 200), "right_hip": (160, 200),
        "right_wrist": (250, 108), "left_wrist": (140, 200),
    }
)  # fmt: skip

# The body horizontal: everything at about the same y, spread across x.
LYING = _keypoints(
    {
        "left_shoulder": (100, 100), "right_shoulder": (110, 102),
        "left_hip": (200, 100), "right_hip": (210, 102),
        "left_ankle": (300, 100), "right_ankle": (310, 102),
    }
)  # fmt: skip


def test_gestures_lists_v2s_eight_ids():
    assert gestures.GESTURES == (
        "waving",
        "raising_left_arm",
        "raising_right_arm",
        "pointing_left",
        "pointing_right",
        "sitting",
        "standing",
        "lying_down",
    )
    assert set(gestures.POSTURES) <= set(gestures.GESTURES)


def test_standing():
    assert gestures.is_standing(*STANDING)
    assert not gestures.is_sitting(*STANDING)
    assert not gestures.is_waving(*STANDING)
    assert not gestures.is_lying_down(*STANDING)


def test_sitting():
    assert gestures.is_sitting(*SITTING)
    assert not gestures.is_standing(*SITTING)


def test_raising_left_arm():
    assert gestures.matches_gesture(*RAISING_LEFT, "raising_left_arm")
    assert not gestures.matches_gesture(*RAISING_LEFT, "raising_right_arm")
    # One image cannot show the motion, so a raised hand reads as waving.
    assert gestures.is_waving(*RAISING_LEFT)


def test_pointing_right():
    assert gestures.matches_gesture(*POINTING_RIGHT, "pointing_right")
    assert not gestures.matches_gesture(*POINTING_RIGHT, "pointing_left")


def test_lying_down():
    assert gestures.is_lying_down(*LYING)
    assert not gestures.is_standing(*LYING)


def test_classify_and_match_dispatch():
    assert "standing" in gestures.classify_gestures(*STANDING)
    assert gestures.matches_gesture(*STANDING, "standing")
    assert not gestures.matches_gesture(*STANDING, "waving")
    assert not gestures.matches_gesture(*STANDING, "breakdancing")


def test_low_score_keypoints_are_ignored():
    keypoints, scores = _keypoints(
        {
            "left_shoulder": (140, 100), "right_shoulder": (160, 100),
            "left_hip": (140, 200), "right_hip": (160, 200),
        }
    )  # fmt: skip
    wrist = _INDEX["left_wrist"]
    keypoints[2 * wrist], keypoints[2 * wrist + 1] = 140.0, 40.0
    scores[wrist] = 0.1
    assert not gestures.matches_gesture(keypoints, scores, "raising_left_arm")
    # The same wrist counts once the threshold is below its score.
    lenient = gestures.GestureConfig(min_keypoint_score=0.05)
    assert gestures.matches_gesture(keypoints, scores, "raising_left_arm", lenient)


def test_waving_framed_from_the_waist_up_uses_the_shoulder_width():
    # No hips, so no torso height: the shoulders give the scale instead.
    upper_body = _keypoints(
        {"left_shoulder": (100, 100), "right_shoulder": (200, 100), "right_wrist": (210, 40)}
    )
    assert gestures.classify_gestures(*upper_body) == {"waving", "raising_right_arm"}


def test_gestures_read_the_message_arrays():
    from walkie_vision_msgs.msg import PersonDetection

    person = PersonDetection()
    person.keypoints, person.keypoint_scores = RAISING_LEFT
    # rosidl gives float32 numpy arrays for the fixed-size fields.
    assert isinstance(person.keypoints, np.ndarray)
    found = gestures.classify_gestures(person.keypoints, person.keypoint_scores)
    assert {"raising_left_arm", "waving", "standing"} <= found


def test_gestures_reject_arrays_of_the_wrong_size():
    with pytest.raises(ValueError, match="34 keypoint coordinates and 17 scores"):
        gestures.classify_gestures([0.0] * 17, [0.0] * 17)
