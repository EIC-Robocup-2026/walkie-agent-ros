"""The pure perception logic: gestures, seating, identity and box headings.

No ROS graph: messages are built in memory, and box_heading reads a
tf2_ros.Buffer filled by hand, which needs no node. The gesture, seating and
identity cases are ported from walkie-agent-v2's tests/test_gpsr_gestures.py,
tests/test_hri_seating.py, tests/test_hri_identity.py and
tests/test_people_store.py.
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from tf2_ros import Buffer
from tf_transformations import quaternion_from_euler, quaternion_multiply
from vision_msgs.msg import BoundingBox2D
from walkie_vision_msgs.msg import ObjectDetection, PersonDetection

from walkie_agent.perception import gestures
from walkie_agent.perception.geometry import box_heading
from walkie_agent.perception.identity import (
    IdentityConfig,
    assign_names,
    average_unit,
    cosine_distance,
    cosine_similarity,
    reject_outliers,
)
from walkie_agent.perception.seating import (
    SeatingConfig,
    bbox_xyxy,
    find_seat_places,
    hip_anchor,
    occupied_region_indices,
    seat_anchor,
    split_seat_regions,
)

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


def _box(cx: float, cy: float, w: float, h: float) -> BoundingBox2D:
    box = BoundingBox2D()
    box.center.position.x, box.center.position.y = float(cx), float(cy)
    box.size_x, box.size_y = float(w), float(h)
    return box


def _xyxy_box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox2D:
    return _box((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


# -- Gestures (v2's tests/test_gpsr_gestures.py) ------------------------------

# Upright, arms hanging, legs extended down.
STANDING = _keypoints(
    {
        "left_shoulder": (140, 100), "right_shoulder": (160, 100),
        "left_hip": (140, 200), "right_hip": (160, 200),
        "left_knee": (140, 300), "right_knee": (160, 300),
        "left_ankle": (140, 400), "right_ankle": (160, 400),
        "left_wrist": (140, 200), "right_wrist": (160, 200),
    }
)

# Thighs roughly horizontal: knees and ankles tucked up near the hips.
SITTING = _keypoints(
    {
        "left_shoulder": (140, 100), "right_shoulder": (160, 100),
        "left_hip": (140, 200), "right_hip": (160, 200),
        "left_knee": (140, 225), "right_knee": (160, 225),
        "left_ankle": (140, 250), "right_ankle": (160, 250),
    }
)

# The left hand raised above the left shoulder.
RAISING_LEFT = _keypoints(
    {
        "left_shoulder": (140, 100), "right_shoulder": (160, 100),
        "left_hip": (140, 200), "right_hip": (160, 200),
        "left_wrist": (140, 40), "right_wrist": (160, 200),
        "left_ankle": (140, 400), "right_ankle": (160, 400),
    }
)

# The right arm held out level toward the image's right.
POINTING_RIGHT = _keypoints(
    {
        "left_shoulder": (140, 100), "right_shoulder": (160, 100),
        "left_hip": (140, 200), "right_hip": (160, 200),
        "right_wrist": (250, 108), "left_wrist": (140, 200),
    }
)

# The body horizontal: everything at about the same y, spread across x.
LYING = _keypoints(
    {
        "left_shoulder": (100, 100), "right_shoulder": (110, 102),
        "left_hip": (200, 100), "right_hip": (210, 102),
        "left_ankle": (300, 100), "right_ankle": (310, 102),
    }
)


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
    assert gestures.POSTURES <= set(gestures.GESTURES)


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
    )
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
    person = PersonDetection()
    person.keypoints, person.keypoint_scores = RAISING_LEFT
    # rosidl gives float32 numpy arrays for the fixed-size fields.
    assert isinstance(person.keypoints, np.ndarray)
    found = gestures.classify_gestures(person.keypoints, person.keypoint_scores)
    assert {"raising_left_arm", "waving", "standing"} <= found


def test_gestures_reject_arrays_of_the_wrong_size():
    with pytest.raises(ValueError, match="34 keypoint coordinates and 17 scores"):
        gestures.classify_gestures([0.0] * 17, [0.0] * 17)


# -- Seating (v2's tests/test_hri_seating.py) ---------------------------------

_HIPS = ("left_hip", "right_hip")


def _person(
    cx: float, cy: float, w: float, h: float, hips: tuple[float, float] | None = None,
    hip_score: float = 0.9,
) -> PersonDetection:
    person = PersonDetection()
    person.bbox = _box(cx, cy, w, h)
    if hips is not None:
        person.keypoints, person.keypoint_scores = _keypoints(
            {name: hips for name in _HIPS}, score=hip_score
        )
    return person


def _seat(label: str, box: tuple[float, float, float, float], score: float = 0.9):
    detection = ObjectDetection()
    detection.label = label
    detection.score = score
    detection.bbox = _xyxy_box(*box)
    return detection


def _taken(places) -> dict[str | None, bool]:
    return {place.part: place.occupied for place in places}


def test_bbox_xyxy_converts_centre_and_size():
    assert bbox_xyxy(_box(100, 50, 40, 80)) == (80, 10, 120, 90)


def test_hip_anchor_is_the_midpoint():
    person = _person(0, 0, 10, 10, hips=(40, 60))
    assert hip_anchor(person.keypoints, person.keypoint_scores, 0.3) == (40, 60)


def test_hip_anchor_uses_one_hip_when_only_one_is_seen():
    keypoints, scores = _keypoints({"left_hip": (40, 60), "right_hip": (80, 60)})
    scores[_INDEX["right_hip"]] = 0.1
    assert hip_anchor(keypoints, scores, 0.3) == (40, 60)


def test_hip_anchor_none_when_low_score():
    person = _person(0, 0, 10, 10, hips=(40, 60), hip_score=0.1)
    assert hip_anchor(person.keypoints, person.keypoint_scores, 0.3) is None


def test_seat_anchor_falls_back_to_the_lower_centre_of_the_box():
    person = _person(100, 50, 40, 80)
    assert seat_anchor(person) == (100, 50 + 20)


def test_split_three_places_left_to_right():
    regions = split_seat_regions((0, 0, 300, 100), has_middle=True)
    assert [part for part, _ in regions] == ["left", "middle", "right"]
    assert [box for _, box in regions] == [
        (0, 0, 100, 100), (100, 0, 200, 100), (200, 0, 300, 100)
    ]


def test_split_two_places():
    regions = split_seat_regions((0, 0, 200, 100), has_middle=False)
    assert [part for part, _ in regions] == ["left", "right"]
    assert [box for _, box in regions] == [(0, 0, 100, 100), (100, 0, 200, 100)]


def test_one_person_on_a_sofa_leaves_the_other_places_free():
    # Seated on the left place, with a box that spreads into the middle one.
    person = _person(60, 50, 120, 100, hips=(50, 50))
    places = find_seat_places([_seat("sofa", (0, 0, 300, 100))], [person])
    assert _taken(places) == {"left": True, "middle": False, "right": False}


def test_a_neighbours_box_over_a_free_chair_does_not_take_it():
    # Seated on a chair at x 100 to 200, with a box that reaches x 220, over
    # a free chair at 200 to 300. The hips are on their own chair.
    person = _person(160, 50, 120, 100, hips=(150, 50))
    own, free = _seat("chair", (100, 0, 200, 100)), _seat("chair", (200, 0, 300, 100))
    places = find_seat_places([free, own], [person])
    assert [(place.box[0], place.occupied) for place in places] == [(100, True), (200, False)]


def test_a_person_without_hips_takes_the_place_under_their_lower_centre():
    person = _person(50, 50, 100, 100)
    places = find_seat_places([_seat("sofa", (0, 0, 300, 100))], [person])
    assert _taken(places) == {"left": True, "middle": False, "right": False}


def test_a_person_on_the_edge_between_places_takes_exactly_one():
    # The anchor lies on the line between the left and middle places.
    person = _person(100, 50, 40, 100)
    places = find_seat_places([_seat("sofa", (0, 0, 300, 100))], [person])
    assert sum(place.occupied for place in places) == 1


def test_an_anchor_on_no_place_takes_nothing():
    person = _person(400, 50, 80, 100, hips=(400, 50))
    places = find_seat_places([_seat("sofa", (0, 0, 300, 100))], [person])
    assert not any(place.occupied for place in places)


def test_occupied_region_indices_gives_each_person_one_region():
    regions = [(0, 0, 100, 100), (100, 0, 200, 100)]
    assert occupied_region_indices([(100, 50)], regions) == {0}
    assert occupied_region_indices([(50, 50), (150, 50)], regions) == {0, 1}
    assert occupied_region_indices([(250, 50)], regions) == set()


def test_seating_config_classes_ignore_case():
    config = SeatingConfig()
    assert config.is_sofa("sofa")
    assert config.is_sofa("Couch")
    assert not config.is_sofa("chair")
    assert not config.is_sofa(None)
    assert config.is_seat("Armchair")
    assert not config.is_seat("table")


def test_find_seat_places_orders_seats_left_to_right_and_keeps_a_sofa_together():
    detections = [
        _seat("chair", (500, 0, 600, 100), score=0.8),
        _seat("table", (0, 0, 50, 50)),
        _seat("sofa", (100, 0, 400, 100), score=0.7),
        _seat("stool", (20, 0, 80, 100), score=0.6),
    ]
    taken_chair = _person(550, 50, 100, 100, hips=(550, 60))
    on_the_right = _person(350, 50, 100, 100, hips=(350, 60))
    places = find_seat_places(detections, [taken_chair, on_the_right])
    assert [(p.label, p.part, p.occupied) for p in places] == [
        ("stool", None, False),
        ("sofa", "left", False),
        ("sofa", "middle", False),
        ("sofa", "right", True),
        ("chair", None, True),
    ]
    assert [p.detection_index for p in places] == [3, 2, 2, 2, 0]
    assert places[1].box == (100, 0, 200, 100)
    assert places[4].score == pytest.approx(0.8)


def test_find_seat_places_follows_the_config():
    detections = [_seat("bench", (0, 0, 200, 100)), _seat("chair", (300, 0, 400, 100))]
    config = SeatingConfig(seat_classes=("bench",), sofa_has_middle=False)
    places = find_seat_places(detections, [], config)
    assert [(p.label, p.part) for p in places] == [("bench", "left"), ("bench", "right")]


def test_find_seat_places_with_no_seat_is_empty():
    assert find_seat_places([_seat("table", (0, 0, 100, 100))], [_person(50, 50, 10, 10)]) == []


# -- Identity (v2's tests/test_hri_identity.py and tests/test_people_store.py) --


def _u(*v: float) -> list[float]:
    """A small unit vector."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def test_average_unit_of_nothing_is_none():
    assert average_unit([]) is None


def test_average_unit_of_one_vector_is_renormalized():
    assert average_unit([[2.0, 0.0, 0.0]]) == pytest.approx([1.0, 0.0, 0.0])


def test_average_unit_is_unit_and_symmetric():
    out = average_unit([_u(1.0, 0.0, 0.0), _u(0.0, 1.0, 0.0)])
    assert math.isclose(math.hypot(*out), 1.0, abs_tol=1e-9)
    assert out[0] == pytest.approx(out[1])


def test_average_unit_of_opposite_vectors_falls_back_to_the_first():
    assert average_unit([[1.0, 0.0], [-1.0, 0.0]]) == [1.0, 0.0]


def test_average_unit_reads_float32_arrays():
    out = average_unit([np.array([3.0, 4.0], dtype=np.float32)])
    assert out == pytest.approx([0.6, 0.8])
    assert all(isinstance(x, float) for x in out)


def test_reject_outliers_drops_the_odd_vector():
    a1, a2 = _u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01)
    far = _u(0.0, 1.0, 0.0)
    kept = reject_outliers([a1, a2, far])
    assert len(kept) == 2 and far not in kept


def test_reject_outliers_keeps_a_uniform_set():
    vectors = [_u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01), _u(0.99, 0.03, 0.0)]
    assert reject_outliers(vectors) == vectors


def test_reject_outliers_keeps_fewer_than_three():
    vectors = [_u(1.0, 0.0, 0.0), _u(0.0, 1.0, 0.0)]
    assert reject_outliers(vectors) == vectors


def test_reject_outliers_never_returns_nothing():
    # A spread so wide that the mean is nearly zero still keeps one vector.
    vectors = [_u(1.0, 0.0, 0.0), _u(-0.5, 0.866, 0.0), _u(-0.5, -0.866, 0.0)]
    assert len(reject_outliers(vectors)) >= 1


def test_reject_outliers_drops_one_outlier_of_four():
    cluster = [_u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01), _u(0.99, 0.03, 0.0)]
    far = _u(0.0, 1.0, 0.0)
    kept = reject_outliers(cluster + [far])
    assert far not in kept and len(kept) == 3


def test_reject_outliers_follows_the_config():
    a1, a2 = _u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01)
    far = _u(0.0, 1.0, 0.0)
    # With no floor and a gate too wide to bite, the far vector stays.
    loose = IdentityConfig(outlier_min_similarity=-1.0, outlier_mad_k=1e6)
    assert reject_outliers([a1, a2, far], loose) == [a1, a2, far]


# Three well separated people's faces, from v2's test_people_store.py.
ALICE = _u(1.0, 0.05, 0.0)
ALICE_2 = _u(0.97, 0.10, 0.05)  # A second shot of Alice.
BOB = _u(0.0, 1.0, 0.05)
CAROL = _u(0.0, 0.05, 1.0)


def test_cosine_helpers():
    assert cosine_similarity(ALICE, ALICE) == pytest.approx(1.0)
    assert cosine_distance(ALICE, ALICE) == pytest.approx(0.0, abs=1e-7)
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)
    # A zero vector is guarded rather than dividing by zero.
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


@dataclass(frozen=True)
class _Match:
    person_id: str
    name: str
    distance: float


def _nearest(query: list[float], stored: dict[str, list[float]]) -> list[_Match]:
    """What PeopleMemory returns: every stored person, nearest first."""
    matches = [
        _Match(person_id=name.lower(), name=name, distance=cosine_distance(query, vector))
        for name, vector in stored.items()
    ]
    return sorted(matches, key=lambda m: m.distance)


def test_recognize_a_second_shot_and_not_a_stranger():
    stored = {"Alice": ALICE, "Bob": BOB}
    names = assign_names([_nearest(ALICE_2, stored), _nearest(CAROL, stored)], [0.4, 0.4])
    assert names[0] is not None and names[0].name == "Alice" and names[0].distance < 0.4
    assert names[1] is None


def test_a_strict_limit_rejects_even_a_real_match():
    stored = {"Alice": ALICE}
    assert assign_names([_nearest(ALICE_2, stored)], [0.0]) == [None]
    assert assign_names([_nearest(ALICE_2, stored)], [0.4])[0].name == "Alice"


def test_assign_names_crossed_distances():
    # The guests switched places: the memory lists Sam first for both, but
    # the left person is nearer Kim, so the names follow the distances.
    candidates = [
        [_Match("kim", "Kim", 0.10), _Match("sam", "Sam", 0.30)],
        [_Match("sam", "Sam", 0.15), _Match("kim", "Kim", 0.20)],
    ]
    names = assign_names(candidates, [0.4, 0.4])
    assert [n.name for n in names] == ["Kim", "Sam"]


def test_assign_names_lets_the_closer_person_keep_a_contested_name():
    # v2's test_locate_greedy_steal_fixed: both faces are nearest to guest 2,
    # face b far more so, so face a gets guest 1, its second best.
    guest_1, guest_2 = _u(1, 0, 0), _u(0, 1, 0)
    face_a = _u(0.52, 0.55, 0.6535)
    face_b = _u(0.10, 0.90, 0.4243)
    stored = {"Guest1": guest_1, "Guest2": guest_2}
    names = assign_names([_nearest(face_a, stored), _nearest(face_b, stored)], [0.6, 0.6])
    assert [n.name for n in names] == ["Guest1", "Guest2"]


def test_assign_names_a_third_person_is_unknown():
    stored = {"Sam": _u(1, 0, 0), "Kim": _u(0, 1, 0)}
    people = [_u(1, 0.05, 0), _u(0.05, 1, 0), _u(0, 0, 1)]
    names = assign_names([_nearest(p, stored) for p in people], [0.4, 0.4, 0.4])
    assert [n.name if n else None for n in names] == ["Sam", "Kim", None]


def test_assign_names_a_limit_excludes_a_far_match():
    candidates = [[_Match("sam", "Sam", 0.6)], [_Match("kim", "Kim", 0.1)]]
    names = assign_names(candidates, [0.4, 0.4])
    assert names[0] is None and names[1].name == "Kim"
    # The second best within the limit names the person whose best is taken.
    candidates = [
        [_Match("sam", "Sam", 0.05)],
        [_Match("sam", "Sam", 0.10), _Match("kim", "Kim", 0.35), _Match("lee", "Lee", 0.45)],
    ]
    names = assign_names(candidates, [0.4, 0.4])
    assert [n.name for n in names] == ["Sam", "Kim"]


def test_assign_names_limits_are_per_person():
    candidates = [[_Match("sam", "Sam", 0.3)], [_Match("kim", "Kim", 0.3)]]
    names = assign_names(candidates, [0.2, 0.5])
    assert names[0] is None and names[1].name == "Kim"


def test_assign_names_never_matches_a_model_without_a_limit():
    candidates = [[_Match("sam", "Sam", 0.01)]]
    assert assign_names(candidates, [None]) == [None]


def test_assign_names_without_limits_takes_every_candidate():
    candidates = [[_Match("sam", "Sam", 0.9)], [_Match("sam", "Sam", 0.5)], []]
    names = assign_names(candidates)
    assert names[0] is None and names[1].name == "Sam" and names[2] is None


def test_assign_names_breaks_ties_by_order():
    candidates = [[_Match("sam", "Sam", 0.2)], [_Match("sam", "Sam", 0.2)]]
    names = assign_names(candidates, [0.4, 0.4])
    assert names[0].name == "Sam" and names[1] is None


def test_assign_names_skips_a_nan_distance():
    assert assign_names([[_Match("sam", "Sam", math.nan)]], [0.4]) == [None]


def test_assign_names_needs_one_limit_per_person():
    with pytest.raises(ValueError, match="one per person"):
        assign_names([[], []], [0.4])


def test_identity_config_limits_by_model():
    config = IdentityConfig()
    assert config.face_distance_limits["arcface_w600k_r50"] == 0.4
    assert config.face_distance_limits["fake_face"] == 0.2
    assert config.appearance_distance_limits["osnet_x1_0_msmt17"] == 0.5
    assert config.appearance_distance_limits["fake_reid"] == 0.2
    assert config.face_distance_limits.get("unknown_model") is None
    with pytest.raises(TypeError):
        config.face_distance_limits["arcface_w600k_r50"] = 1.0  # type: ignore[index]


# -- box_heading ---------------------------------------------------------------

# A camera's optical frame against a body frame: z forward, x right, y down.
_OPTICAL = quaternion_from_euler(-math.pi / 2, 0.0, -math.pi / 2)
_FX, _CX = 500.0, 320.0


def _camera_info() -> CameraInfo:
    info = CameraInfo()
    info.width, info.height = 640, 480
    info.k = [_FX, 0.0, _CX, 0.0, _FX, 240.0, 0.0, 0.0, 1.0]
    return info


def _camera_transform(yaw: float, sec: int = 0) -> TransformStamped:
    """map -> camera_optical, for a level camera that faces `yaw` in the map."""
    t = TransformStamped()
    t.header.frame_id = "map"
    t.header.stamp.sec = sec
    t.child_frame_id = "camera_optical"
    # The translation does not change a heading, so any position will do.
    position = t.transform.translation
    position.x, position.y, position.z = 1.0, 2.0, 1.2
    rotation = t.transform.rotation
    q = quaternion_multiply(quaternion_from_euler(0.0, 0.0, yaw), _OPTICAL)
    rotation.x, rotation.y, rotation.z, rotation.w = (float(v) for v in q)
    return t


def _header(sec: int = 0, frame: str = "camera_optical") -> Header:
    header = Header()
    header.frame_id = frame
    header.stamp.sec = sec
    return header


def _static_buffer(yaw: float) -> Buffer:
    buffer = Buffer()
    buffer.set_transform_static(_camera_transform(yaw), "test")
    return buffer


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -2.5])
def test_box_heading_of_the_centre_column_is_the_camera_yaw(yaw):
    heading = box_heading(_box(_CX, 100, 40, 80), _camera_info(), _header(), _static_buffer(yaw))
    assert heading == pytest.approx(yaw)


def test_box_heading_right_of_centre_turns_clockwise():
    # One focal length right of the centre is 45 degrees to the right.
    buffer = _static_buffer(0.0)
    right = box_heading(_box(_CX + _FX, 100, 40, 80), _camera_info(), _header(), buffer)
    left = box_heading(_box(_CX - _FX, 100, 40, 80), _camera_info(), _header(), buffer)
    assert right == pytest.approx(-math.pi / 4)
    assert left == pytest.approx(math.pi / 4)


def test_box_heading_ignores_the_row_and_takes_an_xyxy_box():
    buffer = _static_buffer(math.pi / 2)
    high = box_heading((_CX + _FX - 10, 0, _CX + _FX + 10, 20), _camera_info(), _header(), buffer)
    low = box_heading(_box(_CX + _FX, 470, 20, 20), _camera_info(), _header(), buffer)
    assert high == pytest.approx(math.pi / 4)
    assert low == pytest.approx(math.pi / 4)


def test_box_heading_uses_the_transform_at_the_stamp():
    buffer = Buffer()
    buffer.set_transform(_camera_transform(0.0, sec=10), "test")
    buffer.set_transform(_camera_transform(1.0, sec=12), "test")
    info, box = _camera_info(), _box(_CX, 100, 40, 80)
    assert box_heading(box, info, _header(sec=10), buffer) == pytest.approx(0.0)
    assert box_heading(box, info, _header(sec=11), buffer) == pytest.approx(0.5)
    assert box_heading(box, info, _header(sec=12), buffer) == pytest.approx(1.0)


def test_box_heading_is_none_without_the_transform():
    info, box = _camera_info(), _box(_CX, 100, 40, 80)
    buffer = _static_buffer(0.0)
    assert box_heading(box, info, _header(frame="other_camera"), buffer, timeout_s=0.05) is None
    # A stamp past the newest transform cannot be extrapolated.
    moving = Buffer()
    moving.set_transform(_camera_transform(0.0, sec=10), "test")
    assert box_heading(box, info, _header(sec=20), moving, timeout_s=0.05) is None


def test_box_heading_is_none_without_a_focal_length():
    buffer = _static_buffer(0.0)
    assert box_heading(_box(_CX, 100, 40, 80), CameraInfo(), _header(), buffer) is None


def test_box_heading_is_none_for_a_camera_that_looks_straight_down():
    t = _camera_transform(0.0)
    # The optical z axis along the map's -z: a rotation of pi about x.
    t.transform.rotation.x, t.transform.rotation.y = 1.0, 0.0
    t.transform.rotation.z, t.transform.rotation.w = 0.0, 0.0
    buffer = Buffer()
    buffer.set_transform_static(t, "test")
    assert box_heading(_box(_CX, 100, 40, 80), _camera_info(), _header(), buffer) is None
