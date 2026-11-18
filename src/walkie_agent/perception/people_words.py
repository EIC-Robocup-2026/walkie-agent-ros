"""The words find_people uses for a person's posture and gesture. No ROS.

classify_gestures (gestures.py) names every rule a person meets, several at once: a
raised hand also counts as a wave, and a posture may be missing. find_people reports one
pose and one gesture per person, as FakeRobot does, so pose_phrase and gesture_phrase take
the most specific name of each kind. A query may still name any of the others, through
phrases(): someone raising their left arm is found as a "waving person" too.

The phrases are FakeRobot's POSES and GESTURES, so the model reads the same words from
the robot as from the fake. In "raising their left arm" the side is the person's own, as
COCO's keypoint names are. In "pointing to the left" it is the image's, which is Walkie's
view, as gestures.py's pointing rules are.
"""

from collections.abc import Collection

# Most specific first: the first name a person shows is the one reported.
POSE_PHRASES = {
    "lying_down": "lying person",
    "sitting": "sitting person",
    "standing": "standing person",
}
GESTURE_PHRASES = {
    "pointing_left": "person pointing to the left",
    "pointing_right": "person pointing to the right",
    "raising_left_arm": "person raising their left arm",
    "raising_right_arm": "person raising their right arm",
    # Last, because every raised hand is a wave too.
    "waving": "waving person",
}


def pose_phrase(names: Collection[str]) -> str:
    """The phrase of the most specific posture in `names`, or "" when there is none."""
    return next((phrase for name, phrase in POSE_PHRASES.items() if name in names), "")


def gesture_phrase(names: Collection[str]) -> str:
    """The phrase of the most specific gesture in `names`, or "" when there is none."""
    return next((phrase for name, phrase in GESTURE_PHRASES.items() if name in names), "")


def phrases(names: Collection[str]) -> frozenset[str]:
    """Every posture and gesture phrase of `names`, which is what a query is matched against."""
    return frozenset(
        phrase
        for table in (POSE_PHRASES, GESTURE_PHRASES)
        for name, phrase in table.items()
        if name in names
    )
