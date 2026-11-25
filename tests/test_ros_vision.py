"""RosVision: find_people over walkie-vision's people arrays, and the words it uses.

It also collects each track's face and appearance samples for TrackMemory, and names
the people in view through a namer; tests/test_ros_people_names.py runs it with
TrackMemory itself.

Most tests drive RosVision over tests/ros_fakes.py's FakeConnector: a test calls the
subscription's callback itself and moves the node's clock, so no test needs a graph.
The last one publishes the oracle's arrays of config/scenes/chula_living_room.toml on
Fast DDS, as the simulation does, and reads them back through a real connector.
"""

import logging
import math
import random
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from rclpy.time import Time
from ros_fakes import (
    WATCHDOG_S,
    Background,
    FakeConnector,
    FakeTfBuffer,
    bounded,
    camera_transform,
    missing_tf,
)
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from test_perception_gestures import LYING, POINTING_RIGHT, RAISING_LEFT, SITTING, STANDING
from test_projection import CX, FX
from test_scene_file import CHULA_LOCATIONS, CHULA_SCENE, STREAM_ID, camera_info, living_room_camera
from tf_transformations import quaternion_from_matrix
from walkie_vision_msgs.msg import (
    ObjectDetection,
    ObjectDetectionArray,
    PersonDetection,
    PersonDetectionArray,
)

from walkie_agent.capabilities import ObjectSeen, Pose2D, UnavailableVision
from walkie_agent.capabilities.fake import GESTURES, POSES
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import Location, load_locations
from walkie_agent.people_memory.track_vectors import Collected, TrackVectorsConfig
from walkie_agent.perception import OBJECTS_TOPIC, PEOPLE_TOPIC
from walkie_agent.perception.geometry import torso_pixels, wrap_deg
from walkie_agent.perception.gestures import COCO_KEYPOINTS, DEFAULT_GESTURE_CONFIG
from walkie_agent.perception.people_words import (
    GESTURE_PHRASES,
    POSE_PHRASES,
    gesture_phrase,
    phrases,
    pose_phrase,
)
from walkie_agent.ros import vision as vision_module
from walkie_agent.ros.vision import (
    NAME_WAIT_S,
    PLACE_TOLERANCE_M,
    TF_WAIT_S,
    UNKNOWN_PLACE,
    RosVision,
)
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, FAKE_REID_MODEL, hash_vector
from walkie_agent.sim.oracle_perception import ORACLE_QOS, OracleConfig, draw_detections
from walkie_agent.sim.projection import camera_position
from walkie_agent.sim.scene_file import load_scene

CAMERA_FRAME = "camera_optical"
# The length of the test's fake vectors; walkie-vision's fakes are longer.
VECTOR_DIM = 16
# A person whose keypoints are all unseen, so only the box's centre places them.
BOX_ONLY = ([0.0] * 34, [0.0] * 17)
ANKLES = tuple(COCO_KEYPOINTS.index(name) for name in ("left_ankle", "right_ankle"))
PLACES = {
    "living room": Location("living room", 0.0, 0.0, 0.0, room="living room"),
    "sofa": Location("sofa", 1.0, 0.5, 0.0, room="living room"),
    "kitchen": Location("kitchen", 6.0, 0.0, 0.0, room="kitchen"),
}
# The wait of a test whose people() is meant to run out.
SHORT_WAIT_S = 0.05
QUIET = OracleConfig(pixel_noise_px=0.0)
# The classes of the test's object arrays: walkie-vision's memory task's, less two.
OBJECT_CLASSES = ("cup", "bottle", "apple", "chair", "table", "sofa", "coffee table")


class StubBase:
    """ctx.base's pose and places. `error` makes pose() raise it."""

    def __init__(self, x: float = 0.2, y: float = 0.0, yaw: float = 0.0) -> None:
        self.at = Pose2D(x, y, yaw)
        self.places_by_name = PLACES
        self.error: StepError | None = None
        self.poses = 0
        self.timeouts: list[float] = []

    def pose(self, *, timeout_s: float = 1.0) -> Pose2D:
        self.poses += 1
        self.timeouts.append(timeout_s)
        if self.error is not None:
            raise self.error
        return self.at

    def places(self):
        return self.places_by_name


class Rig:
    """A RosVision over a FakeConnector, and the means to feed it arrays."""

    def __init__(self, **options) -> None:
        self.connector = FakeConnector()
        self.node = self.connector.node
        self.base = StubBase()
        self.cancel = CancelToken()
        self.vision = RosVision(self.connector, self.base, self.cancel, **options)
        self.subscription, self.objects_subscription = self.node.subscriptions

    def deliver_objects(
        self,
        *detections: ObjectDetection,
        classes: tuple[str, ...] = OBJECT_CLASSES,
        age_s: float = 0.0,
    ) -> ObjectDetectionArray:
        """An object array of `detections` stamped `age_s` before the node's clock, given to the callback."""
        array = ObjectDetectionArray(
            detections=list(detections),
            camera_info=camera_info(),
            classes=list(classes),
            stream_id=STREAM_ID,
        )
        array.header.frame_id = CAMERA_FRAME
        array.header.stamp = Time(nanoseconds=self.node.clock.nanoseconds - round(age_s * 1e9)).to_msg()
        self.objects_subscription.callback(array)
        return array

    def deliver(
        self,
        *people: PersonDetection,
        age_s: float = 0.0,
        frame: str = CAMERA_FRAME,
        info: CameraInfo | None = None,
    ) -> PersonDetectionArray:
        """An array of the Chula camera's image stamped `age_s` before the node's clock, given to the callback.

        `info` replaces the Chula camera's calibration.
        """
        array = PersonDetectionArray(
            people=list(people), camera_info=camera_info() if info is None else info
        )
        array.header.frame_id = frame
        array.header.stamp = Time(nanoseconds=self.node.clock.nanoseconds - round(age_s * 1e9)).to_msg()
        self.subscription.callback(array)
        return array

    def advance(self, seconds: float) -> None:
        self.node.clock.nanoseconds += round(seconds * 1e9)

    def clock_read(self) -> threading.Event:
        """Set at the next reading of the node's clock, which a waiting people() makes."""
        read = threading.Event()
        self.node.clock.on_now = read.set
        return read


def person(x: float, body=STANDING, track_id: int = 0) -> PersonDetection:
    """A detection whose box and torso are centred at image column `x`, with `body`'s pose.

    The keypoints are `body`'s moved sideways, which no gesture rule notices.
    """
    keypoints, scores = body
    shifted = np.array(keypoints, dtype=np.float64)
    torso = torso_pixels(shifted, scores, DEFAULT_GESTURE_CONFIG.min_keypoint_score)
    if torso:
        shifted[0::2] += x - sum(u for u, _ in torso) / len(torso)
    detection = PersonDetection(score=0.9, track_id=track_id)
    detection.bbox.center.position.x = float(x)
    detection.keypoints = shifted.astype(np.float32)
    detection.keypoint_scores = np.array(scores, dtype=np.float32)
    return detection


def seen(people) -> list[tuple]:
    return [(p.id, p.name, p.pose, p.gesture, p.room) for p in people]


def thing(
    label: str, x: float, track_id: int = 0, points: PointCloud2 | None = None
) -> ObjectDetection:
    """An object detection of `label` whose box is centred at image column `x`, with `points`.

    Without `points` the detection's cloud is the message's default, which has no points.
    """
    detection = ObjectDetection(label=label, score=0.9, track_id=track_id)
    detection.bbox.center.position.x = float(x)
    if points is not None:
        detection.points = points
    return detection


def cloud(
    points,
    *,
    offsets: tuple[int, int, int] = (0, 4, 8),
    point_step: int = 12,
    big_endian: bool = False,
    datatype: int = PointField.FLOAT32,
) -> PointCloud2:
    """A one-row cloud of `points`, x, y and z in the optical frame, packed as the fields say.

    By default walkie-vision's layout: float32 x, y and z at offsets 0, 4 and 8,
    little-endian, 12 bytes a point. The bytes between and after the fields are zero.
    """
    rows = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    kind = {PointField.FLOAT32: "f4", PointField.FLOAT64: "f8"}[datatype]
    layout = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [(">" if big_endian else "<") + kind] * 3,
            "offsets": list(offsets),
            "itemsize": point_step,
        }
    )
    packed = np.zeros(len(rows), dtype=layout)
    packed["x"], packed["y"], packed["z"] = rows.T
    message = PointCloud2(
        height=1,
        width=len(rows),
        fields=[
            PointField(name=axis, offset=offset, datatype=datatype, count=1)
            for axis, offset in zip("xyz", offsets, strict=True)
        ],
        is_bigendian=big_endian,
        point_step=point_step,
        row_step=point_step * len(rows),
        is_dense=False,
    )
    message.data = packed.tobytes()
    return message


def around(x: float, y: float, z: float, spread: float = 0.05) -> list[tuple[float, float, float]]:
    """27 points in the optical frame whose per-axis median is (x, y, z), and one hole."""
    steps = (-spread, 0.0, spread)
    grid = [(x + dx, y + dy, z + dz) for dx in steps for dy in steps for dz in steps]
    return [*grid, (math.nan, math.nan, math.nan)]


def with_vectors(
    detection: PersonDetection,
    face: str = "",
    appearance: str = "",
    *,
    score: float = 0.9,
    size_px: float = 40.0,
) -> PersonDetection:
    """`detection` with the fake models' vectors of the keys `face` and `appearance`.

    The face box is `size_px` square, 0.18 rad in the Chula camera at the default,
    and the detector's confidence in it is `score`. An empty key leaves that part out.
    """
    if face:
        detection.face_bbox.size_x = detection.face_bbox.size_y = size_px
        detection.face_det_score = score
        detection.face_model = FAKE_FACE_MODEL
        detection.face_embedding = hash_vector("face:" + face, VECTOR_DIM)
    if appearance:
        detection.appearance_model = FAKE_REID_MODEL
        detection.appearance_embedding = hash_vector("reid:" + appearance, VECTOR_DIM)
    return detection


def vectors(samples) -> list[list[float]]:
    return [sample.vector.tolist() for sample in samples]


def key_vectors(kind: str, *keys: str) -> list[list[float]]:
    return [hash_vector(f"{kind}:{key}", VECTOR_DIM).tolist() for key in keys]


class StubNamer:
    """A namer that gives `names`, or raises `error`, and logs each call's ids and wait."""

    def __init__(self, names: dict[str, str] | None = None, error: StepError | None = None):
        self.names = names or {}
        self.error = error
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, person_ids, *, timeout_s: float) -> dict[str, str]:
        self.calls.append((list(person_ids), timeout_s))
        if self.error is not None:
            raise self.error
        return self.names


def transform_of(camera_from_map: np.ndarray):
    """The map <- camera TF message of a camera <- map matrix, as the oracle uses one."""
    stamped = camera_transform()
    rotation = stamped.transform.rotation
    q = quaternion_from_matrix(np.linalg.inv(camera_from_map))
    rotation.x, rotation.y, rotation.z, rotation.w = (float(v) for v in q)
    origin = stamped.transform.translation
    origin.x, origin.y, origin.z = (float(v) for v in camera_position(camera_from_map))
    return stamped


def chula_rig(rig, ankles: bool = True) -> tuple[Rig, list]:
    """A Rig on the Chula living room's place, given one oracle array of its scene.

    TF gives the oracle's own camera pose. Without `ankles`, the ankles score 0.
    """
    loaded = load_scene(CHULA_SCENE)
    place = load_locations(CHULA_LOCATIONS)["living room"]
    made = rig()
    made.base.at = Pose2D(place.x, place.y, place.yaw)
    camera_from_map = living_room_camera()
    made.connector._tf_buffer.lookup = lambda: transform_of(camera_from_map)
    info = camera_info()
    info.header.stamp = Time(nanoseconds=made.node.clock.nanoseconds).to_msg()
    array, _ = draw_detections(
        loaded.scene, info, camera_from_map, QUIET, np.random.default_rng(0), stream_id=STREAM_ID
    )
    if not ankles:
        for detection in array.people:
            scores = np.array(detection.keypoint_scores)
            scores[list(ANKLES)] = 0.0
            detection.keypoint_scores = scores.astype(np.float32)
    made.subscription.callback(array)
    return made, list(loaded.scene.people)


def degrees_from(x: float, y: float, yaw: float, who) -> float:
    """The direction of `who` from (x, y), less `yaw`, in degrees in [-180, 180)."""
    return wrap_deg(math.degrees(math.atan2(who.y - y, who.x - x) - yaw))


@pytest.fixture
def rig():
    made = []

    def make(**options) -> Rig:
        made.append(Rig(**options))
        return made[-1]

    yield make
    for each in made:
        each.vision.close()


# -- The words -------------------------------------------------------------------------


def test_the_phrases_are_fakerobots_poses_and_gestures():
    assert set(POSE_PHRASES.values()) == set(POSES)
    assert set(GESTURE_PHRASES.values()) == set(GESTURES)


def test_the_most_specific_posture_and_gesture_are_reported():
    assert pose_phrase({"standing", "sitting"}) == "sitting person"
    assert pose_phrase({"sitting", "lying_down"}) == "lying person"
    assert gesture_phrase({"waving", "raising_left_arm"}) == "person raising their left arm"
    assert gesture_phrase({"waving", "raising_right_arm", "pointing_left"}) == (
        "person pointing to the left"
    )
    assert gesture_phrase({"waving"}) == "waving person"
    assert pose_phrase(set()) == gesture_phrase({"standing"}) == ""
    assert phrases({"raising_left_arm", "waving", "standing"}) == {
        "person raising their left arm",
        "waving person",
        "standing person",
    }


# -- The subscription ------------------------------------------------------------------


def test_two_best_effort_volatile_subscriptions_of_depth_5_are_made_and_closed_once(rig):
    made = rig()
    people, objects = made.subscription, made.objects_subscription

    assert (people.topic, people.msg_type) == (PEOPLE_TOPIC, PersonDetectionArray)
    assert (objects.topic, objects.msg_type) == (OBJECTS_TOPIC, ObjectDetectionArray)
    for subscription in (people, objects):
        qos = subscription.qos
        assert (qos.history, qos.depth) == (HistoryPolicy.KEEP_LAST, 5)
        assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
        assert qos.durability == DurabilityPolicy.VOLATILE

    made.vision.close()
    made.vision.close()
    assert made.node.destroyed == [people, objects]
    assert len(made.node.subscriptions) == 2


def test_the_objects_can_come_from_another_topic(rig):
    made = rig(objects_topic="/walkie_vision/sim/objects")
    assert made.objects_subscription.topic == "/walkie_vision/sim/objects"
    assert made.subscription.topic == PEOPLE_TOPIC


@pytest.mark.parametrize("option", ["topic", "objects_topic"])
def test_a_forbidden_topic_is_refused_before_anything_is_made(option):
    connector = FakeConnector()
    with pytest.raises(StepError) as raised:
        RosVision(connector, StubBase(), CancelToken(), **{option: "/cmd_vel"})
    assert raised.value.code is ErrorCode.REFUSED
    assert connector.node.subscriptions == []


def test_a_failed_objects_subscription_destroys_the_people_one():
    connector = FakeConnector()
    made = connector.node.create_subscription

    def create(msg_type, *args, **kwargs):
        if msg_type is ObjectDetectionArray:
            raise RuntimeError("no objects")
        return made(msg_type, *args, **kwargs)

    connector.node.create_subscription = create
    with pytest.raises(RuntimeError, match="no objects"):
        RosVision(connector, StubBase(), CancelToken())
    assert len(connector.node.destroyed) == 1
    assert connector.node.destroyed == connector.node.subscriptions


def test_a_destroy_that_fails_is_logged_and_close_still_ends(rig, caplog):
    made = rig()
    made.node.destroy_error = RuntimeError("gone")
    with caplog.at_level(logging.ERROR):
        made.vision.close()
    assert "Could not destroy the people subscription." in caplog.text
    assert "Could not destroy the object subscription." in caplog.text
    for verb in (made.vision.people, made.vision.objects):
        with pytest.raises(StepError) as raised:
            verb()
        assert raised.value.code is ErrorCode.UNAVAILABLE


# -- What people() returns -------------------------------------------------------------


def test_people_come_left_to_right_with_ids_poses_gestures_and_the_room(rig):
    made = rig()
    made.deliver(
        person(500, RAISING_LEFT, track_id=1),
        person(100, SITTING, track_id=2),
        person(300),
        person(400, POINTING_RIGHT),
        person(450, LYING, track_id=7),
    )

    assert seen(made.vision.people()) == [
        ("person_2", None, "sitting person", "", "living room"),
        ("person_untracked_1", None, "standing person", "", "living room"),
        ("person_untracked_2", None, "", "person pointing to the right", "living room"),
        ("person_7", None, "lying person", "", "living room"),
        ("person_1", None, "standing person", "person raising their left arm", "living room"),
    ]
    assert made.base.poses == 1


def test_two_people_at_one_column_keep_the_arrays_order(rig):
    made = rig()
    made.deliver(person(200, track_id=4), person(200, track_id=3))
    assert [p.id for p in made.vision.people()] == ["person_4", "person_3"]


@pytest.mark.parametrize(
    ("query", "ids"),
    [
        ("", ["person_2", "person_untracked_1", "person_1"]),
        ("waving person", ["person_1"]),
        ("  Waving PERSON ", ["person_1"]),
        ("person raising their left arm", ["person_1"]),
        ("standing person", ["person_untracked_1", "person_1"]),
        ("sitting person", ["person_2"]),
        ("person pointing to the right", []),
        ("Jane", []),
        ("waving", []),
    ],
)
def test_a_query_is_one_of_a_persons_phrases_ignoring_case_and_spaces(rig, query, ids):
    made = rig()
    made.deliver(person(500, RAISING_LEFT, 1), person(100, SITTING, 2), person(300))
    assert [p.id for p in made.vision.people(query)] == ids


def test_a_query_does_not_renumber_the_untracked(rig):
    made = rig()
    made.deliver(person(100), person(200, POINTING_RIGHT))
    [found] = made.vision.people("person pointing to the right")
    assert found.id == "person_untracked_2"


def test_nobody_in_view_or_matching_is_an_empty_list_and_no_pose_read(rig):
    made = rig()
    made.deliver()
    assert made.vision.people() == []
    made.deliver(person(100, SITTING, 2))
    assert made.vision.people("waving person") == []
    assert made.base.poses == 0


def test_the_room_is_the_nearest_places(rig):
    made = rig()
    made.deliver(person(100, track_id=1))
    made.base.at = Pose2D(5.0, 0.4, 1.0)
    assert made.vision.people()[0].room == "kitchen"


@pytest.mark.parametrize("code", [ErrorCode.UNAVAILABLE, ErrorCode.TIMEOUT])
def test_a_pose_that_cannot_be_read_gives_no_room(rig, caplog, code):
    made = rig()
    made.deliver(person(100, track_id=1))
    made.base.error = StepError(code, "No map to base_footprint transform.")
    with caplog.at_level(logging.WARNING):
        [found] = made.vision.people()
    assert seen([found]) == [("person_1", None, "standing person", "", "")]
    assert (found.bearing_deg, found.distance_m) == (None, None)
    assert "find_people reports no room and no bearings or distances" in caplog.text
    # With no pose to subtract, TF is not even asked.
    assert made.connector._tf_buffer.lookups == []


def test_no_configured_place_gives_no_room(rig):
    made = rig()
    made.base.places_by_name = {}
    made.deliver(person(100, track_id=1))
    assert made.vision.people()[0].room == ""


def test_a_stop_while_the_pose_is_read_ends_the_call(rig):
    made = rig()
    made.deliver(person(100, track_id=1))
    made.base.error = StepError(ErrorCode.CANCELLED, "Cancelled reading the pose: a stop is in effect.")
    with pytest.raises(StepError) as raised:
        made.vision.people()
    assert raised.value.code is ErrorCode.CANCELLED


def classified_then(monkeypatch, then: Callable[[], None]) -> None:
    """Run `then` after each person's gestures are classified, inside people()."""
    real = vision_module.classify_gestures

    def classify(*args):
        names = real(*args)
        then()
        return names

    monkeypatch.setattr(vision_module, "classify_gestures", classify)


def test_a_stop_while_people_are_classified_ends_the_call_though_nobody_matches(rig, monkeypatch):
    made = rig()
    made.deliver(person(100, SITTING, 2))
    classified_then(monkeypatch, made.cancel.cancel)
    with pytest.raises(StepError) as raised:
        made.vision.people("waving person")
    assert raised.value.code is ErrorCode.CANCELLED
    assert made.base.poses == 0


def test_the_pose_is_waited_for_only_as_long_as_is_left_of_the_call(rig):
    made = rig()
    made.deliver(person(100, track_id=1))
    made.vision.people(timeout_s=0.5)
    [waited] = made.base.timeouts
    assert 0.0 < waited <= 0.5


def test_a_call_out_of_time_before_the_pose_reports_people_with_no_room_or_bearing(
    rig, monkeypatch, caplog
):
    made = rig()
    made.deliver(person(100, track_id=1))
    classified_then(monkeypatch, lambda: time.sleep(0.06))
    with caplog.at_level(logging.WARNING):
        [found] = made.vision.people(timeout_s=0.05)
    assert seen([found]) == [("person_1", None, "standing person", "", "")]
    assert found.bearing_deg is None
    assert "the call's time ran out before the pose was read" in caplog.text
    assert made.base.poses == 0
    assert made.connector._tf_buffer.lookups == []


# -- Bearings ----------------------------------------------------------------------------


def test_a_bearing_is_the_ray_through_the_box_centre_from_tf_at_the_arrays_stamp(rig):
    made = rig()
    array = made.deliver(
        person(CX + FX, BOX_ONLY), person(CX, BOX_ONLY), person(CX - FX, BOX_ONLY, track_id=4)
    )
    bearings = {p.id: p.bearing_deg for p in made.vision.people()}
    # The default camera is level and faces along the map's x axis, as the base does.
    assert bearings == {"person_4": 45.0, "person_untracked_1": 0.0, "person_untracked_2": -45.0}
    assert math.copysign(1.0, bearings["person_untracked_1"]) == 1.0
    stamp_ns = Time.from_msg(array.header.stamp).nanoseconds
    assert made.connector._tf_buffer.lookups == [("map", CAMERA_FRAME, stamp_ns, TF_WAIT_S)]


def test_a_turn_of_the_base_since_the_image_counts(rig):
    made = rig()
    # The image was taken with the camera turned 50 degrees left; the base is at 20 now.
    made.connector._tf_buffer.lookup = lambda: camera_transform(math.radians(50.0))
    made.base.at = Pose2D(0.2, 0.0, math.radians(20.0))
    made.deliver(person(CX, BOX_ONLY), person(CX + FX, BOX_ONLY))
    assert [p.bearing_deg for p in made.vision.people()] == [30.0, -15.0]


def test_the_torso_and_not_a_box_widened_by_an_arm_gives_the_bearing(rig):
    made = rig()
    pointing = person(CX, POINTING_RIGHT, track_id=1)
    pointing.bbox.center.position.x = CX + 150.0
    made.deliver(pointing)
    [found] = made.vision.people()
    assert found.gesture == "person pointing to the right"
    assert found.bearing_deg == pytest.approx(0.0, abs=0.5)


def test_with_the_ankles_seen_a_bearing_is_from_the_bases_centre(rig):
    """The Chula guest and host, from the oracle's own image of the living room.

    From the camera, 0.28 m ahead of the centre, the guest is 4 degrees wider; the
    first run's turn by that direction, with Nav2's overshoot, ended 6.4 degrees past them.
    """
    made, people = chula_rig(rig)
    at = made.base.at
    camera = camera_position(living_room_camera())
    by_id = {p.id: p.bearing_deg for p in made.vision.people()}
    for who in people:
        found = by_id[f"person_{who.track_id}"]
        assert found == pytest.approx(degrees_from(at.x, at.y, at.yaw, who), abs=0.2), who
    [guest] = [who for who in people if who.face == "guest1"]
    from_camera = degrees_from(camera[0], camera[1], at.yaw, guest)
    assert from_camera - by_id[f"person_{guest.track_id}"] > 3.5


def test_with_the_ankles_seen_a_distance_is_from_the_bases_centre(rig):
    made, people = chula_rig(rig)
    at = made.base.at
    camera = camera_position(living_room_camera())
    by_id = {p.id: p.distance_m for p in made.vision.people()}
    for who in people:
        found = by_id[f"person_{who.track_id}"]
        assert found == pytest.approx(math.hypot(who.x - at.x, who.y - at.y), abs=0.05), who
    # From the base's centre, not from the camera ahead of it: the guest in front is
    # farther from the centre, by about the 0.275 m the camera sits ahead.
    [guest] = [who for who in people if who.face == "guest1"]
    from_camera = math.hypot(guest.x - camera[0], guest.y - camera[1])
    assert by_id[f"person_{guest.track_id}"] - from_camera > 0.15


def test_without_an_ankle_a_bearing_is_the_cameras_direction(rig):
    made, people = chula_rig(rig, ankles=False)
    at = made.base.at
    camera = camera_position(living_room_camera())
    found = {p.id: p for p in made.vision.people()}
    for who in people:
        seen_person = found[f"person_{who.track_id}"]
        assert seen_person.bearing_deg == pytest.approx(
            degrees_from(camera[0], camera[1], at.yaw, who), abs=0.1
        )
        assert seen_person.distance_m is None


def test_a_move_and_a_turn_of_the_base_since_the_image_count_with_the_ankles_seen(rig):
    made, people = chula_rig(rig)
    image_pose = made.base.at
    # The base drove 0.3 m and turned 25 degrees left after the image.
    now = Pose2D(
        image_pose.x + 0.3 * math.cos(image_pose.yaw),
        image_pose.y + 0.3 * math.sin(image_pose.yaw),
        image_pose.yaw + math.radians(25.0),
    )
    made.base.at = now
    found = {p.id: p for p in made.vision.people()}
    for who in people:
        seen_person = found[f"person_{who.track_id}"]
        assert seen_person.bearing_deg == pytest.approx(
            degrees_from(now.x, now.y, now.yaw, who), abs=0.3
        ), who
        assert seen_person.distance_m == pytest.approx(
            math.hypot(who.x - now.x, who.y - now.y), abs=0.05
        ), who


def test_a_person_seen_only_by_the_box_has_a_bearing_but_no_distance(rig):
    made = rig()
    made.deliver(person(CX, BOX_ONLY, track_id=1))
    [found] = made.vision.people()
    assert (found.bearing_deg, found.distance_m) == (0.0, None)


def test_a_query_looks_tf_up_once_and_only_when_someone_matches(rig):
    made = rig()
    made.deliver(person(100, SITTING, track_id=1), person(300, track_id=2))
    assert made.vision.people("lying person") == []
    assert made.connector._tf_buffer.lookups == []
    made.vision.people()
    assert len(made.connector._tf_buffer.lookups) == 1


def test_without_a_transform_people_come_without_bearings_and_a_warning(rig, caplog):
    made = rig()
    made.connector._tf_buffer.lookup = missing_tf
    made.deliver(person(100, track_id=1))
    with caplog.at_level(logging.WARNING):
        [found] = made.vision.people()
    assert (found.id, found.room, found.bearing_deg, found.distance_m) == (
        "person_1",
        "living room",
        None,
        None,
    )
    assert "find_people reports no bearings or distances: map does not exist" in caplog.text


def test_an_array_that_names_no_frame_gives_no_bearings_and_asks_tf_nothing(rig, caplog):
    made = rig()
    made.deliver(person(100, track_id=1), frame="")
    with caplog.at_level(logging.WARNING):
        assert made.vision.people()[0].bearing_deg is None
    assert made.connector._tf_buffer.lookups == []
    assert "the people array names no frame" in caplog.text


def test_a_camera_info_never_filled_in_gives_no_bearings(rig):
    made = rig()
    array = PersonDetectionArray(people=[person(100, track_id=1)], camera_info=CameraInfo())
    array.header.frame_id = CAMERA_FRAME
    array.header.stamp = Time(nanoseconds=made.node.clock.nanoseconds).to_msg()
    made.subscription.callback(array)
    assert made.vision.people()[0].bearing_deg is None


def test_the_wait_for_tf_is_no_longer_than_what_is_left_of_the_call(rig):
    made = rig()
    made.deliver(person(100, track_id=1))
    made.vision.people(timeout_s=0.05)
    [(_, _, _, waited)] = made.connector._tf_buffer.lookups
    assert 0.0 <= waited <= 0.05


def test_an_unexpected_tf_error_is_internal_and_logged(rig, caplog):
    def broken():
        raise RuntimeError("the buffer is broken")

    made = rig()
    made.connector._tf_buffer.lookup = broken
    made.deliver(person(100, track_id=1))
    with caplog.at_level(logging.ERROR), pytest.raises(StepError) as raised:
        made.vision.people()
    assert raised.value.code is ErrorCode.INTERNAL
    assert "RuntimeError" in str(raised.value)
    assert "the buffer is broken" in caplog.text


def test_a_buffer_given_replaces_the_connectors(rig):
    given = FakeTfBuffer()
    made = rig(tf_buffer=given)
    made.deliver(person(100, track_id=1))
    made.vision.people()
    assert len(given.lookups) == 1
    assert made.connector._tf_buffer.lookups == []


# -- What objects() returns ------------------------------------------------------------


def test_objects_come_left_to_right_with_ids_names_categories_and_the_place(rig):
    made = rig()
    made.base.at = Pose2D(1.2, 0.4, 0.0)  # 0.22 m from the sofa place.
    made.deliver_objects(
        thing("sofa", 400, track_id=11),
        thing("cup", 90, track_id=13),
        thing("coffee table", 250, track_id=12),
    )
    assert made.vision.objects() == [
        ObjectSeen("cup_13", "cup", "dish", "sofa"),
        ObjectSeen("coffee_table_12", "coffee table", "other", "sofa"),
        ObjectSeen("sofa_11", "sofa", "furniture", "sofa"),
    ]


def test_untracked_objects_are_numbered_left_to_right_before_the_query(rig):
    made = rig()
    made.deliver_objects(thing("chair", 300), thing("cup", 100), thing("chair", 500, track_id=4))
    assert [o.id for o in made.vision.objects()] == ["cup_untracked_1", "chair_untracked_2", "chair_4"]
    assert [o.id for o in made.vision.objects("chair")] == ["chair_untracked_2", "chair_4"]


@pytest.mark.parametrize(
    ("query", "ids"),
    [
        ("", ["apple_1", "cup_2", "chair_3"]),
        ("Cup ", ["cup_2"]),
        ("dish", ["cup_2"]),
        ("FRUITS", ["apple_1"]),
        ("furniture", ["chair_3"]),
        ("bottle", []),
        ("drinks", []),
    ],
)
def test_a_query_matches_a_name_a_category_or_its_plural_ignoring_case(rig, query, ids):
    made = rig()
    made.deliver_objects(thing("apple", 10, 1), thing("cup", 20, 2), thing("chair", 30, 3))
    assert [o.id for o in made.vision.objects(query)] == ids


@pytest.mark.parametrize("query", ["banana", "snacks", "chairs"])
def test_a_query_the_detector_cannot_match_is_unavailable_and_names_its_classes(rig, query):
    made = rig()
    made.deliver_objects(thing("chair", 30, 3), classes=("chair", "sofa"))
    with pytest.raises(StepError) as raised:
        made.vision.objects(query)
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert raised.value.message == (
        f"The camera is not looking for {query!r} right now. It looks only for: chair, sofa."
    )


def test_an_array_without_classes_lets_every_query_through(rig):
    made = rig()
    made.deliver_objects(thing("chair", 30, 3), classes=())
    assert made.vision.objects("banana") == []
    assert [o.id for o in made.vision.objects("chair")] == ["chair_3"]


def test_an_empty_view_or_no_match_reads_no_pose(rig):
    made = rig()
    made.deliver_objects()
    assert made.vision.objects() == []
    made.deliver_objects(thing("cup", 20, 2))
    assert made.vision.objects("apple") == []
    assert made.base.poses == 0


def test_the_place_is_unknown_past_the_tolerance_of_every_place(rig):
    made = rig()
    made.base.at = Pose2D(1.0 + PLACE_TOLERANCE_M + 0.01, 0.5, 0.0)
    made.deliver_objects(thing("cup", 20, 2))
    assert [o.place for o in made.vision.objects()] == [UNKNOWN_PLACE]
    made.base.at = Pose2D(1.0 + PLACE_TOLERANCE_M - 0.01, 0.5, 0.0)
    assert [o.place for o in made.vision.objects()] == ["sofa"]


def test_without_places_the_place_is_unknown(rig):
    made = rig()
    made.base.places_by_name = {}
    made.deliver_objects(thing("cup", 20, 2))
    assert [o.place for o in made.vision.objects()] == [UNKNOWN_PLACE]


def test_a_pose_that_fails_leaves_the_place_unknown_with_a_warning(rig, caplog):
    made = rig()
    made.base.error = StepError(ErrorCode.UNAVAILABLE, "No pose.")
    made.deliver_objects(thing("cup", 20, 2))
    with caplog.at_level(logging.WARNING):
        assert [o.place for o in made.vision.objects()] == [UNKNOWN_PLACE]
    assert (
        "find_objects reports the place unknown and no positions: UNAVAILABLE: No pose."
        in caplog.text
    )


def test_a_stop_while_the_pose_is_read_ends_objects(rig):
    made = rig()
    made.base.error = StepError(ErrorCode.CANCELLED, "Stopped while reading the pose.")
    made.deliver_objects(thing("cup", 20, 2))
    with pytest.raises(StepError) as raised:
        made.vision.objects()
    assert raised.value.code is ErrorCode.CANCELLED


def test_objects_have_no_measures(rig):
    made = rig()
    made.deliver_objects(thing("apple", 20, 2))
    [apple] = made.vision.objects()
    assert (apple.size_cm, apple.weight_g, apple.thickness_cm) == (None, None, None)


# -- Where the objects are -----------------------------------------------------------------


def test_an_objects_position_is_its_points_middle_from_the_base_now(rig):
    made = rig()
    # The default camera is level at the map's origin and faces along x; the base is
    # at (0.2, 0). The cup's points sit 2 m ahead of the camera and 1 m to its left,
    # so 1.8 m ahead of the base: 2.06 m away, 29.1 degrees to the left.
    array = made.deliver_objects(
        thing("cup", 300, track_id=2, points=cloud(around(-1.0, 0.3, 2.0))),
        thing("bottle", 100, track_id=3),
    )
    by_id = {o.id: o for o in made.vision.objects()}
    cup, bottle = by_id["cup_2"], by_id["bottle_3"]
    assert cup.distance_m == pytest.approx(math.hypot(1.8, 1.0), abs=1e-6)
    assert cup.bearing_deg == round(math.degrees(math.atan2(1.0, 1.8)), 1) == 29.1
    # An object without points has neither, as every one of the oracle's has.
    assert (bottle.distance_m, bottle.bearing_deg) == (None, None)
    stamp_ns = Time.from_msg(array.header.stamp).nanoseconds
    assert made.connector._tf_buffer.lookups == [("map", CAMERA_FRAME, stamp_ns, TF_WAIT_S)]
    assert made.base.poses == 1


def test_a_few_points_behind_the_object_do_not_move_it(rig):
    made = rig()
    on_the_cup = around(0.0, 0.0, 1.5, spread=0.02)
    on_the_wall = [(0.0, 0.0, 4.0)] * 5
    made.deliver_objects(thing("cup", 300, track_id=2, points=cloud(on_the_cup + on_the_wall)))
    [cup] = made.vision.objects()
    assert cup.distance_m == pytest.approx(1.5 - 0.2, abs=0.03)
    assert cup.bearing_deg == 0.0
    assert math.copysign(1.0, cup.bearing_deg) == 1.0


@pytest.mark.parametrize(
    "layout",
    [
        {"offsets": (4, 8, 12), "point_step": 16},
        {"offsets": (8, 0, 4), "point_step": 32},
        {"big_endian": True},
        {"datatype": PointField.FLOAT64, "offsets": (0, 8, 16), "point_step": 24},
    ],
    ids=["padded", "shuffled", "big-endian", "float64"],
)
def test_points_are_read_by_the_layout_their_fields_give(rig, layout):
    made = rig()
    points = cloud(around(-1.0, 0.3, 2.0), **layout)
    made.deliver_objects(thing("cup", 300, track_id=2, points=points))
    [cup] = made.vision.objects()
    assert cup.distance_m == pytest.approx(math.hypot(1.8, 1.0), abs=1e-6)
    assert cup.bearing_deg == 29.1


def test_a_turn_and_a_move_of_the_base_since_the_image_count(rig):
    made = rig()
    # The image was taken with the camera turned 90 degrees left at the origin; the
    # base has since driven to (1, 1) and turned to face along x.
    made.connector._tf_buffer.lookup = lambda: camera_transform(math.radians(90.0))
    made.base.at = Pose2D(1.0, 1.0, 0.0)
    # 3 m ahead of the camera, so at (0, 3) in the map: 2.24 m from the base, 116.6 left.
    made.deliver_objects(thing("cup", 300, track_id=2, points=cloud(around(0.0, 0.0, 3.0))))
    [cup] = made.vision.objects()
    assert cup.distance_m == pytest.approx(math.hypot(1.0, 2.0), abs=1e-6)
    assert cup.bearing_deg == round(math.degrees(math.atan2(2.0, -1.0)), 1)


def test_objects_without_points_read_no_tf(rig):
    made = rig()
    made.deliver_objects(thing("cup", 300, track_id=2), thing("chair", 100, track_id=3))
    found = made.vision.objects()
    assert [(o.distance_m, o.bearing_deg) for o in found] == [(None, None), (None, None)]
    assert made.connector._tf_buffer.lookups == []


def test_tf_is_read_once_and_only_for_the_objects_that_match(rig):
    made = rig()
    made.deliver_objects(
        thing("cup", 300, track_id=2, points=cloud(around(0.0, 0.0, 2.0))),
        thing("chair", 100, track_id=3, points=cloud(around(0.5, 0.0, 3.0))),
        thing("apple", 200, track_id=4),
    )
    assert [o.distance_m for o in made.vision.objects("apple")] == [None]
    assert made.connector._tf_buffer.lookups == []
    assert all(o.distance_m is not None for o in made.vision.objects("furniture"))
    assert len(made.connector._tf_buffer.lookups) == 1
    assert len([o for o in made.vision.objects() if o.distance_m is not None]) == 2
    assert len(made.connector._tf_buffer.lookups) == 2


def test_without_a_transform_objects_come_without_positions_and_a_warning(rig, caplog):
    made = rig()
    made.connector._tf_buffer.lookup = missing_tf
    made.deliver_objects(thing("cup", 300, track_id=2, points=cloud(around(0.0, 0.0, 2.0))))
    with caplog.at_level(logging.WARNING):
        [cup] = made.vision.objects()
    assert (cup.id, cup.distance_m, cup.bearing_deg) == ("cup_2", None, None)
    assert "find_objects reports no positions: map does not exist" in caplog.text


def test_an_object_array_that_names_no_frame_gives_no_positions(rig, caplog):
    made = rig()
    array = ObjectDetectionArray(
        detections=[thing("cup", 300, track_id=2, points=cloud(around(0.0, 0.0, 2.0)))],
        camera_info=camera_info(),
        classes=list(OBJECT_CLASSES),
        stream_id=STREAM_ID,
    )
    array.header.stamp = Time(nanoseconds=made.node.clock.nanoseconds).to_msg()
    made.objects_subscription.callback(array)
    with caplog.at_level(logging.WARNING):
        [cup] = made.vision.objects()
    assert (cup.distance_m, cup.bearing_deg) == (None, None)
    assert made.connector._tf_buffer.lookups == []
    assert "find_objects reports no positions: the object array names no frame" in caplog.text


def test_a_pose_that_fails_gives_no_positions_and_asks_tf_nothing(rig):
    made = rig()
    made.base.error = StepError(ErrorCode.TIMEOUT, "No map to base_footprint transform.")
    made.deliver_objects(thing("cup", 300, track_id=2, points=cloud(around(0.0, 0.0, 2.0))))
    [cup] = made.vision.objects()
    assert (cup.place, cup.distance_m, cup.bearing_deg) == (UNKNOWN_PLACE, None, None)
    assert made.connector._tf_buffer.lookups == []


def test_the_wait_for_tf_of_objects_is_no_longer_than_what_is_left_of_the_call(rig):
    made = rig()
    made.deliver_objects(thing("cup", 300, track_id=2, points=cloud(around(0.0, 0.0, 2.0))))
    made.vision.objects(timeout_s=0.05)
    [(_, _, _, waited)] = made.connector._tf_buffer.lookups
    assert 0.0 <= waited <= 0.05


def test_an_unexpected_tf_error_ends_objects_as_internal(rig, caplog):
    def broken():
        raise RuntimeError("the buffer is broken")

    made = rig()
    made.connector._tf_buffer.lookup = broken
    made.deliver_objects(thing("cup", 300, track_id=2, points=cloud(around(0.0, 0.0, 2.0))))
    with caplog.at_level(logging.ERROR), pytest.raises(StepError) as raised:
        made.vision.objects()
    assert raised.value.code is ErrorCode.INTERNAL
    assert "RuntimeError" in str(raised.value)
    assert "the buffer is broken" in caplog.text


def test_points_that_are_all_holes_give_no_position(rig):
    made = rig()
    holes = cloud([(math.nan, math.nan, math.nan)] * 40)
    made.deliver_objects(thing("cup", 300, track_id=2, points=holes))
    [cup] = made.vision.objects()
    assert (cup.distance_m, cup.bearing_deg) == (None, None)
    assert made.connector._tf_buffer.lookups == []


def broken_clouds() -> dict[str, PointCloud2]:
    no_z = cloud(around(0.0, 0.0, 2.0))
    no_z.fields = no_z.fields[:2]
    integers = cloud(around(0.0, 0.0, 2.0))
    integers.fields[2].datatype = PointField.INT32
    short = cloud(around(0.0, 0.0, 2.0))
    short.data = short.data[:-1]
    overlapping = cloud(around(0.0, 0.0, 2.0), offsets=(0, 4, 8), point_step=12)
    overlapping.point_step = 8
    return {"no-z": no_z, "integers": integers, "short": short, "overlapping": overlapping}


@pytest.mark.parametrize("name", list(broken_clouds()))
def test_points_that_do_not_fit_their_layout_give_no_position_and_a_warning(rig, caplog, name):
    made = rig()
    made.deliver_objects(thing("cup", 300, track_id=2, points=broken_clouds()[name]))
    with caplog.at_level(logging.WARNING):
        [cup] = made.vision.objects()
    assert (cup.id, cup.distance_m, cup.bearing_deg) == ("cup_2", None, None)
    assert "find_objects reports no position for cup_2" in caplog.text


def test_objects_and_people_keep_their_own_newest_array(rig):
    made = rig(wait_s=SHORT_WAIT_S)
    made.deliver_objects(thing("cup", 20, 2))
    with pytest.raises(StepError) as raised:
        made.vision.people()
    assert raised.value.message.startswith("No message on /walkie_vision/people within")
    made.deliver(person(100, track_id=1))
    made.advance(2.0)
    made.deliver_objects(thing("chair", 20, 3))
    assert [o.id for o in made.vision.objects()] == ["chair_3"]
    with pytest.raises(StepError) as raised:
        made.vision.people()
    assert "/walkie_vision/people is 2.0 s old" in raised.value.message


def test_no_object_array_is_unavailable_and_names_the_objects_topic(rig):
    made = rig(wait_s=SHORT_WAIT_S, objects_topic="/walkie_vision/sim/objects")
    made.deliver(person(100, track_id=1))
    with pytest.raises(StepError) as raised:
        made.vision.objects()
    assert str(raised.value) == (
        "UNAVAILABLE: No message on /walkie_vision/sim/objects within 0.05 s. "
        "Is walkie-vision or oracle_perception running?"
    )


def test_an_old_object_array_is_unavailable(rig):
    made = rig(wait_s=SHORT_WAIT_S)
    made.deliver_objects(thing("cup", 20, 2), age_s=1.5)
    with pytest.raises(StepError) as raised:
        made.vision.objects()
    assert "/walkie_vision/objects is 1.5 s old" in raised.value.message


def test_a_late_older_object_array_does_not_hide_the_newer_one(rig):
    made = rig()
    made.deliver_objects(thing("chair", 20, 3))
    made.deliver_objects(thing("cup", 20, 2), age_s=0.5)
    assert [o.id for o in made.vision.objects()] == ["chair_3"]


def test_an_object_array_far_behind_the_newest_starts_over(rig):
    made = rig()
    made.advance(100.0)
    made.deliver_objects(thing("chair", 20, 3))
    made.node.clock.nanoseconds = 0
    made.deliver_objects(thing("cup", 20, 2))
    assert [o.id for o in made.vision.objects()] == ["cup_2"]


def test_a_stop_ends_a_wait_for_objects(rig):
    made = rig()
    read = made.clock_read()
    made.cancel.cancel()
    with pytest.raises(StepError) as raised:
        made.vision.objects()
    assert raised.value.code is ErrorCode.CANCELLED
    assert not read.is_set()


def test_close_ends_every_later_objects_call(rig):
    made = rig()
    made.deliver_objects(thing("cup", 20, 2))
    made.vision.close()
    made.deliver_objects(thing("cup", 20, 2))
    with pytest.raises(StepError) as raised:
        made.vision.objects()
    assert str(raised.value) == "UNAVAILABLE: Live object perception has shut down."


@pytest.mark.parametrize("vision", ["ros", "unavailable"])
@pytest.mark.parametrize(
    ("call", "words"),
    [
        (lambda v: v.people(timeout_s=0), "timeout_s must be positive"),
        (lambda v: v.people(timeout_s=math.nan), "timeout_s"),
        (lambda v: v.people(3), "query"),
        (lambda v: v.objects(timeout_s=-1), "timeout_s must be positive"),
    ],
)
def test_bad_arguments_are_refused_as_unavailablevision_refuses_them(rig, vision, call, words):
    target = rig().vision if vision == "ros" else UnavailableVision(CancelToken())
    with pytest.raises(StepError) as raised:
        call(target)
    assert raised.value.code is ErrorCode.REFUSED
    assert words in raised.value.message


# -- Freshness -------------------------------------------------------------------------


def unavailable(made: Rig, **kwargs) -> str:
    with pytest.raises(StepError) as raised:
        made.vision.people(**kwargs)
    assert raised.value.code is ErrorCode.UNAVAILABLE
    return raised.value.message


def test_no_array_is_unavailable_with_a_hint(rig):
    made = rig(wait_s=SHORT_WAIT_S)
    assert unavailable(made) == (
        "No message on /walkie_vision/people within 0.05 s. "
        "Is walkie-vision or oracle_perception running?"
    )


def test_the_wait_is_the_shorter_of_wait_s_and_timeout_s(rig):
    made = rig()
    started = time.monotonic()
    assert "within 0.01 s" in unavailable(made, timeout_s=0.01)
    assert time.monotonic() - started < 0.5


def test_an_array_older_than_max_age_s_is_unavailable(rig):
    made = rig(wait_s=SHORT_WAIT_S)
    made.deliver(person(100, track_id=1), age_s=1.5)
    assert unavailable(made) == (
        "The newest message on /walkie_vision/people is 1.5 s old, older than 1 s. "
        "Is walkie-vision or oracle_perception running?"
    )
    made.deliver(person(100, track_id=1), age_s=0.9)
    assert [p.id for p in made.vision.people()] == ["person_1"]


def test_an_array_that_grows_old_while_kept_is_unavailable(rig):
    made = rig(wait_s=SHORT_WAIT_S)
    made.deliver(person(100, track_id=1))
    assert made.vision.people()
    made.advance(2.0)
    assert "is 2.0 s old" in unavailable(made)


def test_an_array_stamped_far_ahead_names_the_clock_mismatch(rig):
    made = rig(wait_s=SHORT_WAIT_S)
    made.deliver(person(100, track_id=1), age_s=-2.0)
    assert unavailable(made) == (
        "The newest message on /walkie_vision/people is stamped 2.0 s ahead of this "
        "node's clock, so the two use different clocks. Check that both or neither use "
        "sim time."
    )
    # Ahead by less than max_age_s counts, as a small offset between two clocks.
    other = rig(wait_s=SHORT_WAIT_S)
    other.deliver(person(100, track_id=1), age_s=-0.5)
    assert other.vision.people()


def test_a_late_older_array_does_not_hide_the_newer_one(rig):
    made = rig(wait_s=SHORT_WAIT_S)
    made.deliver(person(100, track_id=1), age_s=0.1)
    made.deliver(person(100, track_id=2), age_s=3.0)
    assert [p.id for p in made.vision.people()] == ["person_1"]


def test_an_array_far_behind_the_newest_starts_over_as_after_a_simulation_restart(rig):
    made = rig(wait_s=SHORT_WAIT_S)
    made.deliver(person(100, track_id=1))
    # The simulation restarted: its clock, and the node's, went back.
    made.advance(-60.0)
    made.deliver(person(100, track_id=9))
    assert [p.id for p in made.vision.people()] == ["person_9"]


def test_an_array_that_arrives_during_the_wait_ends_it(rig):
    made = rig(wait_s=WATCHDOG_S)
    made.deliver(person(100, track_id=1), age_s=5.0)
    read = made.clock_read()
    call = Background(made.vision.people)
    assert read.wait(WATCHDOG_S)
    made.node.clock.on_now = None
    made.deliver(person(100, track_id=2))
    assert [p.id for p in call.result()] == ["person_2"]


# -- Stops and close ---------------------------------------------------------------------


def test_a_stop_ends_a_wait(rig):
    made = rig(wait_s=WATCHDOG_S)
    made.deliver(person(100, track_id=1), age_s=5.0)
    read = made.clock_read()
    call = Background(made.vision.people)
    assert read.wait(WATCHDOG_S)
    made.cancel.cancel()
    error = call.step_error()
    assert error.code is ErrorCode.CANCELLED
    assert error.message == "Cancelled looking for people: a stop is in effect."


def test_a_stop_wins_over_a_fresh_array(rig):
    made = rig()
    made.deliver(person(100, track_id=1))
    made.cancel.cancel()
    with pytest.raises(StepError) as raised:
        made.vision.people()
    assert raised.value.code is ErrorCode.CANCELLED
    assert made.base.poses == 0


def test_close_ends_a_wait_and_every_later_call(rig):
    made = rig(wait_s=WATCHDOG_S)
    made.deliver(person(100, track_id=1), age_s=5.0)
    read = made.clock_read()
    call = Background(made.vision.people)
    assert read.wait(WATCHDOG_S)
    made.node.clock.on_now = None
    bounded(made.vision.close)
    assert str(call.step_error()) == "UNAVAILABLE: Live people perception has shut down."

    # An array that comes after close is dropped.
    made.deliver(person(100, track_id=1))
    assert unavailable(made) == "Live people perception has shut down."


def test_reads_and_arrivals_on_two_threads_never_mix_arrays(rig):
    made = rig(wait_s=WATCHDOG_S)
    stop = threading.Event()

    def publish() -> None:
        track = 0
        while not stop.is_set():
            track += 1
            made.deliver(person(100, track_id=track), person(200, track_id=track))

    publisher = threading.Thread(target=publish, daemon=True)
    publisher.start()
    try:
        for _ in range(200):
            first, second = bounded(made.vision.people)
            assert first.id == second.id
    finally:
        stop.set()
        publisher.join(WATCHDOG_S)


# -- Face and appearance samples ---------------------------------------------------------


def test_a_tracked_persons_face_and_appearance_are_collected_as_read_only_copies(rig):
    made = rig()
    detection = with_vectors(person(100, track_id=3), face="sam", appearance="sam")
    made.deliver(detection)
    collected = made.vision.collected("person_3")
    [face] = collected.faces
    [appearance] = collected.appearances
    assert (face.model, appearance.model) == (FAKE_FACE_MODEL, FAKE_REID_MODEL)
    assert vectors([face]) == key_vectors("face", "sam")
    assert vectors([appearance]) == key_vectors("reid", "sam")
    for sample in (face, appearance):
        assert sample.vector.dtype == np.float32
        assert not sample.vector.flags.writeable
    # A change to the message later does not reach the sample.
    detection.face_embedding = np.zeros(VECTOR_DIM, dtype=np.float32)
    assert vectors(made.vision.collected("person_3").faces) == key_vectors("face", "sam")


@pytest.mark.parametrize(
    ("score", "size_px", "filled"),
    [(0.3, 40.0, True), (0.9, 8.0, True), (0.9, 40.0, False)],
    ids=["doubtful", "small", "no-calibration"],
)
def test_an_untrusted_face_is_not_collected_but_the_appearance_is(rig, score, size_px, filled):
    made = rig()
    detection = with_vectors(
        person(100, track_id=3), face="sam", appearance="sam", score=score, size_px=size_px
    )
    made.deliver(detection, info=None if filled else CameraInfo())
    collected = made.vision.collected("person_3")
    assert collected.faces == ()
    assert vectors(collected.appearances) == key_vectors("reid", "sam")


def test_untracked_people_and_unusable_vectors_are_never_collected(rig):
    made = rig()
    unusable = person(200, track_id=4)
    unusable.face_bbox.size_x = unusable.face_bbox.size_y = 40.0
    unusable.face_det_score = 0.9
    unusable.face_model = FAKE_FACE_MODEL
    unusable.face_embedding = np.zeros(VECTOR_DIM, dtype=np.float32)
    # An appearance vector with no model name.
    unusable.appearance_embedding = hash_vector("reid:kim", VECTOR_DIM)
    made.deliver(with_vectors(person(100), face="sam", appearance="sam"), unusable)
    assert made.vision.collected("person_untracked_1") is None
    assert made.vision.collected("person_4") == Collected()
    assert not made.vision.collected("person_4")


@pytest.mark.parametrize(
    "person_id",
    [
        "person_untracked_1",
        "person_0",
        "person_03",
        "person_",
        "person_x",
        "PERSON_3",
        " person_3",
        "person_3 ",
        "person_3.1",
        "robot_3",
        "person_٣",
        3,
    ],
)
def test_an_id_that_names_no_track_gives_no_samples(rig, person_id):
    made = rig()
    made.deliver(with_vectors(person(100, track_id=3), appearance="sam"))
    assert made.vision.collected("person_3")
    assert made.vision.collected(person_id) is None


def test_a_track_never_seen_is_tracked_but_has_no_samples(rig):
    made = rig()
    assert made.vision.collected("person_9") == Collected()


def test_late_arrays_add_samples_in_stamp_order_and_one_array_counts_once(rig):
    made = rig()
    newer = made.deliver(with_vectors(person(100, track_id=3), face="new"), age_s=0.1)
    made.deliver(with_vectors(person(100, track_id=3), face="old"), age_s=0.5)
    made.subscription.callback(newer)
    assert vectors(made.vision.collected("person_3").faces) == key_vectors("face", "old", "new")


def test_a_track_unseen_for_the_ttl_has_no_samples(rig):
    made = rig(vectors=TrackVectorsConfig(track_ttl_s=0.05))
    made.deliver(with_vectors(person(100, track_id=3), face="sam"))
    assert made.vision.collected("person_3")
    time.sleep(0.1)
    assert made.vision.collected("person_3") == Collected()


def test_a_clock_that_jumps_back_drops_the_samples(rig):
    made = rig()
    made.deliver(with_vectors(person(100, track_id=1), face="before"))
    made.deliver(with_vectors(person(300, track_id=2), face="before"))
    # The simulation restarted, and its tracker gives track 1 to someone else.
    made.advance(-60.0)
    made.deliver(with_vectors(person(100, track_id=1), face="after"))
    assert vectors(made.vision.collected("person_1").faces) == key_vectors("face", "after")
    assert made.vision.collected("person_2") == Collected()


def test_close_drops_the_samples_and_later_reads_give_none(rig):
    made = rig()
    made.deliver(with_vectors(person(100, track_id=1), face="sam"))
    made.vision.close()
    assert made.vision.collected("person_1") is None
    assert made.vision.tracked_in_view() == []
    made.deliver(with_vectors(person(100, track_id=1), face="sam"))
    # The buffer itself, which no public read can reach after close.
    assert made.vision._vectors.tracks(time.monotonic()) == []


def test_tracked_in_view_is_the_newest_arrays_tracked_people_left_to_right(rig):
    made = rig(wait_s=SHORT_WAIT_S)
    assert made.vision.tracked_in_view() == []
    made.deliver(
        person(500, track_id=1), person(100, track_id=2), person(300), person(400, track_id=2)
    )
    assert made.vision.tracked_in_view() == ["person_2", "person_1"]
    # However old the newest is, and a late array does not replace it.
    made.advance(30.0)
    made.deliver(person(100, track_id=7), age_s=31.0)
    unavailable(made)
    assert made.vision.tracked_in_view() == ["person_2", "person_1"]
    made.deliver(person(100))
    assert made.vision.tracked_in_view() == []


# -- Names -------------------------------------------------------------------------------


def test_people_shows_the_names_of_the_tracked_people_asked_before_the_query(rig):
    made = rig()
    namer = StubNamer({"person_2": "Sam"})
    made.vision.set_namer(namer)
    made.deliver(person(500, RAISING_LEFT, 1), person(100, SITTING, 2), person(300))

    assert seen(made.vision.people()) == [
        ("person_2", "Sam", "sitting person", "", "living room"),
        ("person_untracked_1", None, "standing person", "", "living room"),
        ("person_1", None, "standing person", "person raising their left arm", "living room"),
    ]
    assert seen(made.vision.people("waving person")) == [
        ("person_1", None, "standing person", "person raising their left arm", "living room"),
    ]
    assert [ids for ids, _ in namer.calls] == [["person_2", "person_1"]] * 2


@pytest.mark.parametrize(
    ("query", "ids"),
    [
        ("Sam", ["person_2"]),
        ("  sAM ", ["person_2"]),
        ("mary ann", ["person_1"]),
        ("MARY ANN ", ["person_1"]),
        ("sa", []),
        ("mary", []),
        ("sam smith", []),
        ("sitting person", ["person_2"]),
        ("waving person", ["person_1"]),
    ],
)
def test_a_query_matches_a_name_ignoring_case_and_spaces_or_a_phrase(rig, query, ids):
    made = rig()
    made.vision.set_namer(StubNamer({"person_2": "Sam", "person_1": "Mary Ann"}))
    made.deliver(person(500, RAISING_LEFT, 1), person(100, SITTING, 2), person(300))
    assert [p.id for p in made.vision.people(query)] == ids


def test_without_a_namer_or_anyone_tracked_nobody_is_named(rig):
    made = rig()
    namer = StubNamer({"person_untracked_1": "Sam", "person_1": "Sam"})
    made.vision.set_namer(namer)
    made.deliver(person(100))
    assert [p.name for p in made.vision.people()] == [None]
    made.deliver()
    assert made.vision.people() == []
    assert namer.calls == []

    made.vision.set_namer(None)
    made.deliver(person(100, track_id=1))
    assert [p.name for p in made.vision.people()] == [None]
    assert namer.calls == []


def test_a_blank_name_is_no_name(rig):
    made = rig()
    made.vision.set_namer(StubNamer({"person_1": ""}))
    made.deliver(person(100, track_id=1))
    assert [p.name for p in made.vision.people()] == [None]


def test_the_names_are_waited_for_two_seconds_at_most_and_never_past_the_call(rig):
    made = rig()
    namer = StubNamer()
    made.vision.set_namer(namer)
    made.deliver(person(100, track_id=1))
    made.vision.people()
    made.vision.people(timeout_s=0.5)
    [(_, first), (_, second)] = namer.calls
    assert NAME_WAIT_S == 2.0
    assert first == NAME_WAIT_S
    assert 0.0 < second <= 0.5


def test_a_call_out_of_time_before_the_names_shows_none(rig, caplog):
    made = rig()
    namer = StubNamer({"person_1": "Sam"})
    made.vision.set_namer(namer)
    made.deliver(person(100, track_id=1))
    # The freshness check reads the clock, and that read takes the call's whole time.
    made.node.clock.on_now = lambda: time.sleep(0.06)
    with caplog.at_level(logging.WARNING):
        [found] = made.vision.people(timeout_s=0.05)
    assert (found.id, found.name) == ("person_1", None)
    assert namer.calls == []
    assert "find_people shows no names: the call's time ran out" in caplog.text


@pytest.mark.parametrize("code", [ErrorCode.UNAVAILABLE, ErrorCode.TIMEOUT, ErrorCode.REFUSED])
def test_a_namer_that_fails_leaves_the_names_null_with_a_warning(rig, caplog, code):
    made = rig()
    made.vision.set_namer(StubNamer(error=StepError(code, "The database is down.")))
    made.deliver(person(100, track_id=1))
    with caplog.at_level(logging.WARNING):
        [found] = made.vision.people()
    assert seen([found]) == [("person_1", None, "standing person", "", "living room")]
    assert "find_people shows no names: " in caplog.text
    assert "The database is down." in caplog.text


def test_a_stop_while_people_are_named_ends_the_call(rig):
    made = rig()
    stopped = StepError(
        ErrorCode.CANCELLED, "Cancelled naming the people in view: a stop is in effect."
    )
    made.vision.set_namer(StubNamer(error=stopped))
    made.deliver(person(100, track_id=1))
    with pytest.raises(StepError) as raised:
        made.vision.people()
    assert raised.value is stopped
    assert made.base.poses == 0


# -- Over Fast DDS -----------------------------------------------------------------------


def test_the_oracles_chula_arrays_are_read_over_dds(connector):
    """The guest and the host of the Chula scene, as find_people reports them there.

    TF comes from a stand-in that gives the oracle's own camera pose, so each bearing
    can be checked against the person's true direction from the base's centre.
    """
    loaded = load_scene(CHULA_SCENE)
    places = load_locations(CHULA_LOCATIONS)
    place = places["living room"]
    base = StubBase(place.x, place.y, place.yaw)
    base.places_by_name = places
    camera_from_map = living_room_camera()
    tf_buffer = FakeTfBuffer()
    tf_buffer.lookup = lambda: transform_of(camera_from_map)
    vision = RosVision(connector, base, CancelToken(), tf_buffer=tf_buffer)
    publisher_node = rclpy.create_node(f"oracle_stand_in_{random.randint(0, 1_000_000)}")
    publisher = publisher_node.create_publisher(PersonDetectionArray, PEOPLE_TOPIC, ORACLE_QOS)
    rng = np.random.default_rng(0)
    try:
        found: list = []
        deadline = time.monotonic() + WATCHDOG_S
        while not found and time.monotonic() < deadline:
            info = camera_info()
            info.header.stamp = connector.node.get_clock().now().to_msg()
            # No pixel noise, so the gestures do not depend on how many images discovery took.
            people, _ = draw_detections(
                loaded.scene, info, living_room_camera(), QUIET, rng, stream_id=STREAM_ID
            )
            publisher.publish(people)
            try:
                found = vision.people(timeout_s=0.2)
            except StepError as error:
                assert error.code is ErrorCode.UNAVAILABLE, error
        by_id = {p.id: p for p in found}
        assert set(by_id) == {"person_1", "person_2"}
        assert seen([by_id["person_1"]]) == [
            ("person_1", None, "standing person", "person raising their left arm", "living room")
        ]
        assert seen([by_id["person_2"]]) == [
            ("person_2", None, "sitting person", "", "living room")
        ]
        assert [p.id for p in vision.people("waving person")] == ["person_1"]
        # The torso's middle, where the guest's raised arm puts the box's centre 1.7
        # degrees off, at the distance of the floor under the ankles.
        for who in loaded.scene.people:
            truth = degrees_from(place.x, place.y, place.yaw, who)
            assert by_id[f"person_{who.track_id}"].bearing_deg == pytest.approx(truth, abs=0.2)
    finally:
        vision.close()
        publisher_node.destroy_node()


def test_the_oracles_chula_objects_are_read_over_dds(connector):
    """The sofa, the coffee table and the mug of the Chula scene, as find_objects reports them.

    The oracle draws them with the scene file's classes, as `pixi run oracle-chula`
    does, and the robot stands at the living room place.
    """
    loaded = load_scene(CHULA_SCENE)
    places = load_locations(CHULA_LOCATIONS)
    place = places["living room"]
    base = StubBase(place.x, place.y, place.yaw)
    base.places_by_name = places
    config = OracleConfig(pixel_noise_px=0.0, classes=tuple(loaded.classes))
    vision = RosVision(connector, base, CancelToken(), tf_buffer=FakeTfBuffer())
    publisher_node = rclpy.create_node(f"oracle_stand_in_{random.randint(0, 1_000_000)}")
    publisher = publisher_node.create_publisher(ObjectDetectionArray, OBJECTS_TOPIC, ORACLE_QOS)
    rng = np.random.default_rng(0)
    try:
        found: list = []
        deadline = time.monotonic() + WATCHDOG_S
        while not found and time.monotonic() < deadline:
            info = camera_info()
            info.header.stamp = connector.node.get_clock().now().to_msg()
            _, objects = draw_detections(
                loaded.scene, info, living_room_camera(), config, rng, stream_id=STREAM_ID
            )
            publisher.publish(objects)
            try:
                found = vision.objects(timeout_s=0.2)
            except StepError as error:
                assert error.code is ErrorCode.UNAVAILABLE, error
        assert sorted(found, key=lambda seen: seen.id) == [
            ObjectSeen("mug_13", "mug", "dish", "living room"),
            ObjectSeen("sofa_11", "sofa", "furniture", "living room"),
            ObjectSeen("table_12", "table", "furniture", "living room"),
        ]
        assert [seen.id for seen in vision.objects("furniture")] == [
            seen.id for seen in found if seen.category == "furniture"
        ]
        with pytest.raises(StepError) as raised:
            vision.objects("apple")
        assert raised.value.message.endswith(
            "It looks only for: chair, sofa, armchair, stool, table, mug."
        )
    finally:
        vision.close()
        publisher_node.destroy_node()
