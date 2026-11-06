"""RosVision: find_people over walkie-vision's people arrays, and the words it uses.

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
from sensor_msgs.msg import CameraInfo
from test_perception_gestures import LYING, POINTING_RIGHT, RAISING_LEFT, SITTING, STANDING
from test_projection import CX, FX
from test_scene_file import CHULA_LOCATIONS, CHULA_SCENE, STREAM_ID, camera_info, living_room_camera
from tf_transformations import quaternion_from_matrix
from walkie_vision_msgs.msg import PersonDetection, PersonDetectionArray

from walkie_agent.capabilities import Pose2D
from walkie_agent.capabilities.fake import GESTURES, POSES
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import Location, load_locations
from walkie_agent.perception import PEOPLE_TOPIC
from walkie_agent.perception.geometry import torso_pixels, wrap_deg
from walkie_agent.perception.gestures import COCO_KEYPOINTS, DEFAULT_GESTURE_CONFIG
from walkie_agent.perception.people import (
    GESTURE_PHRASES,
    POSE_PHRASES,
    gesture_phrase,
    phrases,
    pose_phrase,
)
from walkie_agent.ros import RosVision, UnavailableVision
from walkie_agent.ros import vision as vision_module
from walkie_agent.ros.vision import TF_WAIT_S
from walkie_agent.sim.oracle_perception import ORACLE_QOS, OracleConfig, draw_detections
from walkie_agent.sim.projection import camera_position
from walkie_agent.sim.scene_file import load_scene

CAMERA_FRAME = "camera_optical"
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
        [self.subscription] = self.node.subscriptions

    def deliver(
        self, *people: PersonDetection, age_s: float = 0.0, frame: str = CAMERA_FRAME
    ) -> PersonDetectionArray:
        """An array of the Chula camera's image stamped `age_s` before the node's clock, given to the callback."""
        array = PersonDetectionArray(people=list(people), camera_info=camera_info())
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


def test_one_best_effort_volatile_subscription_of_depth_5_is_made_and_closed_once(rig):
    made = rig()
    subscription = made.subscription

    assert subscription.topic == PEOPLE_TOPIC
    assert subscription.msg_type is PersonDetectionArray
    qos = subscription.qos
    assert (qos.history, qos.depth) == (HistoryPolicy.KEEP_LAST, 5)
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.durability == DurabilityPolicy.VOLATILE

    made.vision.close()
    made.vision.close()
    assert made.node.destroyed == [subscription]
    assert len(made.node.subscriptions) == 1


def test_a_forbidden_topic_is_refused_before_anything_is_made():
    connector = FakeConnector()
    with pytest.raises(StepError) as raised:
        RosVision(connector, StubBase(), CancelToken(), topic="/cmd_vel")
    assert raised.value.code is ErrorCode.REFUSED
    assert connector.node.subscriptions == []


def test_a_destroy_that_fails_is_logged_and_close_still_ends(rig, caplog):
    made = rig()
    made.node.destroy_error = RuntimeError("gone")
    with caplog.at_level(logging.ERROR):
        made.vision.close()
    assert "Could not destroy the people subscription." in caplog.text
    with pytest.raises(StepError) as raised:
        made.vision.people()
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
    assert found.bearing_deg is None
    assert "find_people reports no room and no bearings" in caplog.text
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


def test_without_an_ankle_a_bearing_is_the_cameras_direction(rig):
    made, people = chula_rig(rig, ankles=False)
    at = made.base.at
    camera = camera_position(living_room_camera())
    by_id = {p.id: p.bearing_deg for p in made.vision.people()}
    for who in people:
        found = by_id[f"person_{who.track_id}"]
        assert found == pytest.approx(degrees_from(camera[0], camera[1], at.yaw, who), abs=0.1)


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
    by_id = {p.id: p.bearing_deg for p in made.vision.people()}
    for who in people:
        found = by_id[f"person_{who.track_id}"]
        assert found == pytest.approx(degrees_from(now.x, now.y, now.yaw, who), abs=0.3), who


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
    assert (found.id, found.room, found.bearing_deg) == ("person_1", "living room", None)
    assert "find_people reports no bearings: map does not exist" in caplog.text


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


def test_objects_stay_unavailable(rig):
    made = rig()
    with pytest.raises(StepError) as raised:
        made.vision.objects("cup")
    assert str(raised.value) == "UNAVAILABLE: Live object perception is not connected."


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
