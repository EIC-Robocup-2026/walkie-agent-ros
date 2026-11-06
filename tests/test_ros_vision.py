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

import numpy as np
import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from rclpy.time import Time
from ros_fakes import WATCHDOG_S, Background, FakeConnector, bounded
from test_perception_gestures import LYING, POINTING_RIGHT, RAISING_LEFT, SITTING, STANDING
from test_scene_file import CHULA_LOCATIONS, CHULA_SCENE, STREAM_ID, camera_info, living_room_camera
from walkie_vision_msgs.msg import PersonDetection, PersonDetectionArray

from walkie_agent.capabilities import Pose2D
from walkie_agent.capabilities.fake import GESTURES, POSES
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import Location, load_locations
from walkie_agent.perception import PEOPLE_TOPIC
from walkie_agent.perception.people import (
    GESTURE_PHRASES,
    POSE_PHRASES,
    gesture_phrase,
    phrases,
    pose_phrase,
)
from walkie_agent.ros import RosVision, UnavailableVision
from walkie_agent.sim.oracle_perception import ORACLE_QOS, OracleConfig, draw_detections
from walkie_agent.sim.scene_file import load_scene

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

    def __init__(self, x: float = 0.2, y: float = 0.0) -> None:
        self.at = Pose2D(x, y, 0.0)
        self.places_by_name = PLACES
        self.error: StepError | None = None
        self.poses = 0

    def pose(self) -> Pose2D:
        self.poses += 1
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

    def deliver(self, *people: PersonDetection, age_s: float = 0.0) -> PersonDetectionArray:
        """An array stamped `age_s` before the node's clock, given to the callback."""
        array = PersonDetectionArray(people=list(people))
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
    """A detection centred at image column `x`, with `body`'s keypoints."""
    keypoints, scores = body
    detection = PersonDetection(score=0.9, track_id=track_id)
    detection.bbox.center.position.x = float(x)
    detection.keypoints = np.array(keypoints, dtype=np.float32)
    detection.keypoint_scores = np.array(scores, dtype=np.float32)
    return detection


def seen(people) -> list[tuple]:
    return [(p.id, p.name, p.pose, p.gesture, p.room) for p in people]


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
        assert seen(made.vision.people()) == [("person_1", None, "standing person", "", "")]
    assert "find_people reports no room" in caplog.text


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
    """The guest and the host of the Chula scene, as find_people reports them there."""
    loaded = load_scene(CHULA_SCENE)
    places = load_locations(CHULA_LOCATIONS)
    base = StubBase(places["living room"].x, places["living room"].y)
    base.places_by_name = places
    vision = RosVision(connector, base, CancelToken())
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
    finally:
        vision.close()
        publisher_node.destroy_node()
