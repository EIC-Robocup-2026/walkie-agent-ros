"""VisionFeed over ROS, the in-memory people memory, and the fake vectors.

The VisionFeed tests publish walkie_vision_msgs arrays from a second node on
the `connector` fixture's Fast DDS domain (ADR-0019), with hash_vector
vectors under walkie-vision's fake model names, as oracle_perception will.
The tests of stamp order, clock epochs and threads hand the arrays to the
feed's callbacks instead, as its executor would, so that the order in which
they arrive is exact.
"""

import logging
import random
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from walkie_vision_msgs.msg import (
    ObjectDetection,
    ObjectDetectionArray,
    PersonDetection,
    PersonDetectionArray,
)

from walkie_agent.perception import vision_feed as vision_feed_module
from walkie_agent.perception.people_memory import InMemoryPeopleMemory, ModelVector
from walkie_agent.perception.vision_feed import (
    OBJECTS_TOPIC,
    PEOPLE_TOPIC,
    Sample,
    TrackKey,
    VisionFeed,
    VisionFeedConfig,
    face_trusted,
)
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, FAKE_REID_MODEL, hash_vector
from walkie_agent.toolbox import StepError

DIM = 16
NO_CANCEL = threading.Event()


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# -- The fake vectors ------------------------------------------------------------


def test_hash_vector_is_deterministic_unit_float32():
    vector = hash_vector("guest sam", 512)

    assert vector.dtype == np.float32
    assert vector.shape == (512,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)
    assert np.array_equal(vector, hash_vector("guest sam", 512))
    assert np.array_equal(vector, hash_vector(b"guest sam", 512))
    # Different inputs give nearly orthogonal vectors, which is what lets a
    # 0.2 distance limit tell fake people apart.
    assert abs(float(vector @ hash_vector("guest ana", 512))) < 0.2


def test_hash_vector_matches_walkie_vision():
    # Values from walkie-vision's own hash_vector in src/walkie_vision/fakes.py.
    # A change here means vectors no longer match oracle_perception's.
    assert hash_vector("guest sam", 8)[:4].tolist() == pytest.approx(
        [-0.2260013222694397, 0.31027519702911377, 0.2287740558385849, -0.33814337849617004]
    )
    assert hash_vector(b"face 1", 8)[:4].tolist() == pytest.approx(
        [-0.24494804441928864, -0.41149866580963135, 0.19372032582759857, -0.055320821702480316]
    )


def test_fake_model_names_are_walkie_visions():
    assert (FAKE_FACE_MODEL, FAKE_REID_MODEL) == ("fake_face", "fake_reid")


# -- InMemoryPeopleMemory ----------------------------------------------------------


def face(key: str, dim: int = 512) -> ModelVector:
    return ModelVector(FAKE_FACE_MODEL, hash_vector(f"face {key}", dim))


def appearance(key: str, dim: int = 512) -> ModelVector:
    return ModelVector(FAKE_REID_MODEL, hash_vector(f"appearance {key}", dim))


def near(vector: ModelVector, key: str, weight: float = 0.1) -> ModelVector:
    """A second shot of the same person: the vector nudged towards another."""
    nudged = vector.vector + weight * hash_vector(f"noise {key}", vector.vector.size)
    return ModelVector(vector.model, nudged / np.linalg.norm(nudged))


def test_memory_matches_faces_nearest_first_within_one_model():
    memory = InMemoryPeopleMemory()
    sam = memory.remember("Sam", {"drink": "milk"}, [face("sam")], [appearance("sam")])
    ana = memory.remember("Ana", {}, [face("ana")], [])

    [matches] = memory.match_faces([near(face("sam"), "1")])

    assert [m.person_id for m in matches] == [sam, ana]
    assert (matches[0].name, dict(matches[0].facts)) == ("Sam", {"drink": "milk"})
    assert matches[0].distance < 0.05
    assert matches[1].distance > 0.8
    # Another model's vectors are never compared, so nobody matches them.
    other_model = ModelVector("arcface_w600k_r50", face("sam").vector)
    assert memory.match_faces([other_model]) == [[]]


def test_memory_keeps_faces_and_appearances_apart():
    memory = InMemoryPeopleMemory()
    sam = memory.remember("Sam", {}, [], [appearance("sam")])

    assert memory.match_faces([face("sam")]) == [[]]
    [[match]] = memory.match_appearances([appearance("sam")])
    assert match.person_id == sam
    assert match.distance == pytest.approx(0.0, abs=1e-6)


def test_memory_answers_each_query_and_uses_the_nearest_sample():
    memory = InMemoryPeopleMemory()
    # Two samples, as from two angles: the query is near the second only.
    sam = memory.remember("Sam", {}, [face("sam front"), face("sam side")], [])
    ana = memory.remember("Ana", {}, [face("ana")], [])

    sam_matches, ana_matches = memory.match_faces([face("sam side"), face("ana")])

    assert sam_matches[0].person_id == sam
    assert sam_matches[0].distance == pytest.approx(0.0, abs=1e-6)
    assert ana_matches[0].person_id == ana


def test_memory_clear_forgets_everyone_and_never_reuses_an_id():
    memory = InMemoryPeopleMemory()
    first = memory.remember("Sam", {}, [face("sam")], [])

    memory.clear()

    assert len(memory) == 0
    assert memory.match_faces([face("sam")]) == [[]]
    assert memory.remember("Ana", {}, [face("ana")], []) != first


def test_memory_returns_copies_of_the_facts():
    memory = InMemoryPeopleMemory()
    facts = {"drink": "milk"}
    memory.remember("Sam", facts, [face("sam")], [])
    facts["drink"] = "tea"

    [[match]] = memory.match_faces([face("sam")])
    match.facts["drink"] = "coffee"

    [[again]] = memory.match_faces([face("sam")])
    assert again.facts == {"drink": "milk"}


def test_memory_rejects_a_person_without_vectors_and_a_wrong_length():
    memory = InMemoryPeopleMemory()
    with pytest.raises(ValueError, match="without a face or appearance"):
        memory.remember("Sam", {}, [], [])
    with pytest.raises(ValueError, match="not zero"):
        memory.remember("Sam", {}, [ModelVector(FAKE_FACE_MODEL, np.zeros(4))], [])
    memory.remember("Sam", {}, [face("sam")], [])
    with pytest.raises(ValueError, match="512"):
        memory.match_faces([face("sam", dim=8)])


# -- The face trust rule -------------------------------------------------------------


# The focal lengths VisionFeedConfig.min_face_size_rad is explained by: the
# ZED 2i's nominal one at HD1080 and HD2K, and the simulation's head camera.
ROBOT_FX = 1066.0
SIM_FX = 224.057


def camera_info(
    fx: float, width: int = 1280, height: int = 720, fy: float | None = None
) -> CameraInfo:
    info = CameraInfo(width=width, height=height)
    fy = fx if fy is None else fy
    info.k = [fx, 0.0, width / 2, 0.0, fy, height / 2, 0.0, 0.0, 1.0]
    return info


def face_detection(score: float, size_x: float, size_y: float | None = None) -> PersonDetection:
    person = PersonDetection(face_det_score=score)
    person.face_bbox.size_x = size_x
    person.face_bbox.size_y = size_x if size_y is None else size_y
    return person


@pytest.mark.parametrize(
    ("score", "side_px", "trusted"),
    [
        (0.9, 60.0, True),
        # v2's 50 x 50 px, at the robot's fx just over 0.0469 rad.
        (0.5, 50.0, True),
        (0.49, 60.0, False),
        (0.9, 49.0, False),
        # No face found: walkie-vision leaves the score and the box at 0.
        (0.0, 0.0, False),
    ],
)
def test_face_trusted_needs_score_and_size(score, side_px, trusted):
    assert face_trusted(face_detection(score, side_px), camera_info(ROBOT_FX)) is trusted


def test_face_trusted_takes_the_face_as_an_angle_on_any_camera():
    sim = camera_info(SIM_FX, 640, 360)

    # The guest at the simulation's door is 32.8 px across, far below 50 px
    # but well over the 10.5 px that 0.0469 rad is at this fx.
    assert face_trusted(face_detection(0.9, 32.8), sim)
    assert not face_trusted(face_detection(0.9, 10.0), sim)
    # The same pixels at four times the focal length span a quarter of the angle.
    assert not face_trusted(face_detection(0.9, 32.8), camera_info(4 * SIM_FX))
    # A box that is not square counts by the side of the square of its area.
    assert face_trusted(face_detection(0.9, 25.0, 100.0), camera_info(ROBOT_FX))
    assert not face_trusted(face_detection(0.9, 24.0, 100.0), camera_info(ROBOT_FX))


def test_face_trusted_measures_width_by_fx_and_height_by_fy():
    # An image stretched to twice its width: the face is twice as wide in
    # pixels and so is fx, so it spans the same 0.06 rad as before.
    assert face_trusted(face_detection(0.9, 60.0), camera_info(1000.0))
    stretched = camera_info(2000.0, fy=1000.0)
    assert face_trusted(face_detection(0.9, 120.0, 60.0), stretched)
    # A square box in the stretched image is half as wide as an angle: 0.042 rad.
    assert not face_trusted(face_detection(0.9, 60.0), stretched)


@pytest.mark.parametrize(("scale", "trusted"), [(1.0 - 1e-6, False), (1.0, True), (1.0 + 1e-6, True)])
def test_a_face_on_the_size_limit_is_trusted(scale, trusted):
    # 42.7 x 58.537... px at fx 1066 spans exactly 0.0469 rad. Its centre is
    # off the image's corner, where the box's corners, centre minus and plus
    # half a size, would give back a size a little under its own.
    person = face_detection(0.9, 42.7, 58.537237029508184 * scale)
    person.face_bbox.center.position.x = 1084.715
    person.face_bbox.center.position.y = 549.918
    assert VisionFeedConfig().min_face_size_rad == 0.0469
    assert face_trusted(person, camera_info(ROBOT_FX)) is trusted


def test_face_trusted_names_nobody_by_a_camera_info_never_filled_in():
    assert not face_trusted(face_detection(0.9, 500.0), CameraInfo(width=1280, height=720))
    # A focal length on one axis only is no better.
    only_fx = camera_info(1000.0, fy=0.0)
    assert not face_trusted(face_detection(0.9, 500.0), only_fx)


def test_vision_feed_config_defaults():
    config = VisionFeedConfig()

    assert (config.vectors_per_track, config.max_tracks, config.track_ttl_s) == (20, 32, 60.0)
    assert (config.min_face_det_score, config.min_face_size_rad) == (0.5, 0.0469)
    assert (config.history, config.epoch_jump_s) == (30, 5.0)


@pytest.mark.parametrize("field", ["track_ttl_s", "epoch_jump_s", "min_face_size_rad"])
def test_vision_feed_config_needs_positive_values(field):
    with pytest.raises(ValueError, match=field):
        VisionFeedConfig(**{field: 0.0})


# -- VisionFeed over ROS ---------------------------------------------------------------


def people_array(people, stamp: TimeMsg) -> PersonDetectionArray:
    msg = PersonDetectionArray(people=list(people))
    msg.header.stamp = stamp
    msg.header.frame_id = "zed_head_left_camera_optical_frame"
    # At the robot's fx, the tests' 50 px faces are trusted and 40 px ones are not.
    msg.camera_info = camera_info(ROBOT_FX)
    return msg


def objects_array(labels, stamp: TimeMsg) -> ObjectDetectionArray:
    msg = ObjectDetectionArray(
        classes=sorted(set(labels)), detections=[ObjectDetection(label=x) for x in labels]
    )
    msg.header.stamp = stamp
    return msg


def stamp_ns(stamp: TimeMsg) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class FakeVision:
    """A second node that publishes the two arrays as walkie-vision does."""

    def __init__(self, connector, qos) -> None:
        self.clock_node = connector.node
        self.node = rclpy.create_node(f"fake_vision_{random.randint(0, 1_000_000)}")
        self.people = self.node.create_publisher(PersonDetectionArray, PEOPLE_TOPIC, qos)
        self.objects = self.node.create_publisher(ObjectDetectionArray, OBJECTS_TOPIC, qos)
        self._count = 0

    def stamp(self, offset_s: float = 0.0) -> TimeMsg:
        """A stamp `offset_s` from now on the feed's clock, unique per call."""
        # The nanosecond counter keeps two stamps made in the same tick apart.
        self._count += 1
        now_ns = self.clock_node.get_clock().now().nanoseconds + int(offset_s * 1e9) + self._count
        return TimeMsg(sec=now_ns // 1_000_000_000, nanosec=now_ns % 1_000_000_000)

    def wait_for_subscribers(self) -> None:
        assert wait_until(
            lambda: self.people.get_subscription_count() > 0
            and self.objects.get_subscription_count() > 0
        ), "VisionFeed's subscriptions were not discovered."

    def publish_people(self, people, stamp: TimeMsg | None = None) -> PersonDetectionArray:
        msg = people_array(people, self.stamp() if stamp is None else stamp)
        self.people.publish(msg)
        return msg

    def publish_objects(self, labels, stamp: TimeMsg | None = None) -> ObjectDetectionArray:
        msg = objects_array(labels, self.stamp() if stamp is None else stamp)
        self.objects.publish(msg)
        return msg

    def close(self) -> None:
        self.node.destroy_node()


@pytest.fixture
def vision(connector):
    # Reliable, as a publisher with rclpy's default QoS is: the feed's
    # best-effort subscription must match it.
    fake = FakeVision(connector, 10)
    yield fake
    fake.close()


@pytest.fixture
def feed(connector, vision):
    feed = VisionFeed(connector)
    vision.wait_for_subscribers()
    return feed


def tracked(track_id, key=None, score=0.9, side_px=60.0, with_appearance=True):
    """A person on track `track_id` with a face and an appearance from `key`."""
    key = key or str(track_id)
    person = PersonDetection(track_id=track_id, score=0.9)
    person.bbox.center.position.x, person.bbox.center.position.y = 400.0, 360.0
    person.bbox.size_x, person.bbox.size_y = 200.0, 600.0
    person.face_bbox.size_x = person.face_bbox.size_y = side_px
    person.face_det_score = score
    person.face_model = FAKE_FACE_MODEL
    person.face_embedding = hash_vector(f"face {key}", DIM).tolist()
    if with_appearance:
        person.appearance_model = FAKE_REID_MODEL
        person.appearance_embedding = hash_vector(f"appearance {key}", DIM).tolist()
    return person


def newest_tracks(feed) -> list[int]:
    return [p.track_id for p in feed.latest_people(1.0, 0.0, NO_CANCEL).msg.people]


def keys(frame) -> dict[int, TrackKey]:
    """The TrackKey of each tracked person of a PeopleFrame, by track id."""
    return {key.track_id: key for key in frame.tracks if key is not None}


def deliver(feed, vision, people, stamp: TimeMsg | None = None):
    """Publish a people array and return its PeopleFrame once the feed has it.

    One at a time: the subscription keeps only 5, so a burst could lose some.
    """
    sent = vision.publish_people(people, stamp)
    return feed.people_at(sent.header.stamp, 3.0, NO_CANCEL)


def test_the_feed_subscribes_once_best_effort_and_volatile(connector, feed):
    for topic in (PEOPLE_TOPIC, OBJECTS_TOPIC):
        infos = [
            info
            for info in connector.node.get_subscriptions_info_by_topic(topic)
            if info.node_name == connector.node.get_name()
        ]
        assert len(infos) == 1
        assert infos[0].qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
        assert infos[0].qos_profile.durability == DurabilityPolicy.VOLATILE


def test_latest_people_returns_a_fresh_array(feed, vision):
    sent = vision.publish_people([tracked(4)])

    got = feed.latest_people(1.0, 3.0, NO_CANCEL)

    assert got.msg.header.stamp == sent.header.stamp
    assert [p.track_id for p in got.msg.people] == [4]
    assert (got.msg.camera_info.width, got.msg.camera_info.height) == (1280, 720)
    [key] = got.tracks
    assert key.track_id == 4


def test_a_best_effort_publisher_is_heard_too(connector):
    fake = FakeVision(connector, QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
    try:
        feed = VisionFeed(connector)
        fake.wait_for_subscribers()
        fake.publish_objects(["chair"])

        assert feed.latest_objects(1.0, 3.0, NO_CANCEL).classes == ["chair"]
    finally:
        fake.close()


def test_without_a_publisher_the_reads_are_unavailable_and_name_the_topic(connector):
    feed = VisionFeed(connector)

    start = time.monotonic()
    with pytest.raises(StepError) as people_error:
        feed.latest_people(1.0, 0.2, NO_CANCEL)
    waited = time.monotonic() - start
    with pytest.raises(StepError) as objects_error:
        feed.latest_objects(1.0, 0.0, NO_CANCEL)

    assert people_error.value.code == "UNAVAILABLE"
    assert people_error.value.message == (
        f"No message on {PEOPLE_TOPIC} within 0.2 s. "
        "Is walkie-vision or oracle_perception running?"
    )
    assert 0.2 <= waited < 1.5
    assert objects_error.value.code == "UNAVAILABLE"
    assert OBJECTS_TOPIC in objects_error.value.message


def test_a_stale_array_is_unavailable_with_its_age(feed, vision):
    stale = vision.publish_people([tracked(1)], stamp=vision.stamp(-5.0))
    feed.people_at(stale.header.stamp, 3.0, NO_CANCEL)

    with pytest.raises(StepError) as error:
        feed.latest_people(1.0, 0.1, NO_CANCEL)

    # The same array is fresh enough for a looser limit.
    assert feed.latest_people(10.0, 0.0, NO_CANCEL).msg.header.stamp == stale.header.stamp
    assert error.value.code == "UNAVAILABLE"
    assert f"The newest message on {PEOPLE_TOPIC} is 5." in error.value.message
    assert "older than 1 s" in error.value.message
    assert "oracle_perception" in error.value.message


def test_an_array_from_another_clock_is_unavailable(feed, vision):
    ahead = vision.publish_people([], stamp=vision.stamp(30.0))
    feed.people_at(ahead.header.stamp, 3.0, NO_CANCEL)

    with pytest.raises(StepError) as error:
        feed.latest_people(1.0, 0.0, NO_CANCEL)

    assert error.value.code == "UNAVAILABLE"
    assert "ahead of this node's clock" in error.value.message


def test_latest_people_waits_for_an_array_that_comes_later(feed, vision):
    timer = threading.Timer(0.2, lambda: vision.publish_people([tracked(9)]))
    timer.start()
    try:
        got = feed.latest_people(1.0, 3.0, NO_CANCEL)
    finally:
        timer.join()

    assert [p.track_id for p in got.msg.people] == [9]


def test_cancel_ends_a_wait(feed):
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()

    start = time.monotonic()
    with pytest.raises(StepError) as error:
        feed.latest_people(1.0, 5.0, cancel)
    timer.join()

    assert error.value.code == "CANCELLED"
    assert time.monotonic() - start < 1.0
    with pytest.raises(StepError, match="CANCELLED"):
        feed.people_at(TimeMsg(sec=1), 5.0, cancel)


def test_a_read_with_cancel_set_is_cancelled_with_a_fresh_array_at_hand(feed, vision):
    people = deliver(feed, vision, [tracked(1)])
    vision.publish_objects(["chair"])
    feed.latest_objects(1.0, 3.0, NO_CANCEL)
    cancel = threading.Event()
    cancel.set()

    # A stop set before the call: the cached arrays are fresh, and still no
    # read returns one.
    for read in (
        lambda: feed.latest_people(1.0, 0.0, cancel),
        lambda: feed.latest_objects(1.0, 0.0, cancel),
        lambda: feed.people_at(people.msg.header.stamp, 0.0, cancel),
    ):
        with pytest.raises(StepError) as error:
            read()
        assert error.value.code == "CANCELLED"


def test_a_stop_that_comes_with_an_array_cancels_the_wait(feed, vision):
    cancel = threading.Event()
    # Made beforehand, so the array follows the stop at once and the wait
    # wakes for both together.
    msg = people_array([tracked(1)], vision.stamp())

    def stop_then_arrive():
        cancel.set()
        feed._on_people(msg)

    timer = threading.Timer(0.1, stop_then_arrive)
    timer.start()
    try:
        with pytest.raises(StepError) as error:
            feed.latest_people(1.0, 3.0, cancel)
    finally:
        timer.join()

    assert error.value.code == "CANCELLED"


def test_bad_wait_arguments_are_errors(feed):
    with pytest.raises(ValueError, match="wait_s"):
        feed.latest_people(1.0, -1.0, NO_CANCEL)
    with pytest.raises(ValueError, match="max_age_s"):
        feed.latest_objects(float("nan"), 1.0, NO_CANCEL)


def test_people_at_pairs_the_objects_array_with_the_same_stamp(feed, vision):
    image = vision.stamp()
    deliver(feed, vision, [tracked(1)])
    objects = vision.publish_objects(["chair", "sofa"], stamp=image)
    # The image's people array comes after its objects array, and is followed
    # by a newer image's people array.
    later = threading.Timer(
        0.2,
        lambda: (
            vision.publish_people([tracked(2)], stamp=image),
            vision.publish_people([tracked(3)]),
        ),
    )
    later.start()
    try:
        newest_objects = feed.latest_objects(1.0, 3.0, NO_CANCEL)
        people = feed.people_at(newest_objects.header.stamp, 3.0, NO_CANCEL)
    finally:
        later.join()

    assert newest_objects.header.stamp == objects.header.stamp
    assert people.msg.header.stamp == image
    assert [p.track_id for p in people.msg.people] == [2]
    # Once the newer array is in, the pair is still the array with the stamp.
    assert wait_until(lambda: newest_tracks(feed) == [3])
    assert [p.track_id for p in feed.people_at(image, 0.0, NO_CANCEL).msg.people] == [2]


def test_people_at_without_that_stamp_is_unavailable(feed, vision):
    deliver(feed, vision, [tracked(1)])

    with pytest.raises(StepError) as error:
        feed.people_at(TimeMsg(sec=12, nanosec=5), 0.1, NO_CANCEL)

    assert error.value.code == "UNAVAILABLE"
    assert PEOPLE_TOPIC in error.value.message
    assert "12.000000005" in error.value.message


def test_collects_the_newest_20_vectors_per_track_with_their_models(feed, vision):
    for frame in range(25):
        key = keys(deliver(feed, vision, [tracked(7, key=f"7 frame {frame}")]))[7]

    collected = feed.collected_vectors(key)

    assert len(collected.faces) == len(collected.appearances) == 20
    assert {s.vector.model for s in collected.faces} == {FAKE_FACE_MODEL}
    assert {s.vector.model for s in collected.appearances} == {FAKE_REID_MODEL}
    expected = [hash_vector(f"face 7 frame {frame}", DIM) for frame in range(5, 25)]
    assert all(
        np.array_equal(s.vector.vector, e) for s, e in zip(collected.faces, expected, strict=True)
    )
    assert not collected.faces[0].vector.vector.flags.writeable
    assert feed.collected_vectors(TrackKey(8, key.generation)) == ((), ())


def test_samples_carry_the_stamps_of_their_arrays(feed, vision):
    stamps = [vision.stamp() for _ in range(3)]
    # The second image's array comes last: its samples go in stamp order.
    for stamp in (stamps[0], stamps[2], stamps[1]):
        key = keys(deliver(feed, vision, [tracked(5, key=f"5 at {stamp_ns(stamp)}")], stamp))[5]

    collected = feed.collected_vectors(key)

    expected = [stamp_ns(stamp) for stamp in stamps]
    assert [s.stamp_ns for s in collected.faces] == expected
    assert [s.stamp_ns for s in collected.appearances] == expected
    assert all(isinstance(s, Sample) for s in collected.faces + collected.appearances)
    for sample in collected.faces:
        expected_face = hash_vector(f"face 5 at {sample.stamp_ns}", DIM)
        assert np.array_equal(sample.vector.vector, expected_face)


def test_keeps_32_tracks_and_skips_track_0(feed, vision):
    # 32 tracks and an untracked person: track 0 takes no place.
    frame = deliver(feed, vision, [tracked(n) for n in range(1, 33)] + [tracked(0)])
    first = keys(frame)

    assert frame.tracks[32] is None
    assert [key.track_id for key in frame.tracks[:32]] == list(range(1, 33))
    assert all(len(feed.collected_vectors(first[n]).faces) == 1 for n in range(1, 33))

    # Track 2 is seen again, so track 1 is now the one seen longest ago.
    assert keys(deliver(feed, vision, [tracked(2, key="2 again")])) == {2: first[2]}
    key_33 = keys(deliver(feed, vision, [tracked(33)]))[33]

    assert feed.collected_vectors(first[1]) == ((), ())
    assert len(feed.collected_vectors(first[2]).faces) == 2
    assert all(len(feed.collected_vectors(first[n]).faces) == 1 for n in range(3, 33))
    assert len(feed.collected_vectors(key_33).faces) == 1
    # Track 1 seen again after it was dropped is a new collection.
    back = keys(deliver(feed, vision, [tracked(1, key="1 back")]))[1]
    assert back.generation not in {key.generation for key in (*first.values(), key_33)}
    assert len(feed.collected_vectors(back).faces) == 1


def test_skips_untrusted_faces_but_keeps_their_appearance(feed, vision):
    first = keys(deliver(feed, vision, [tracked(1, score=0.4), tracked(2, side_px=40.0)]))
    second = keys(
        deliver(feed, vision, [tracked(3, score=0.5, side_px=50.0, with_appearance=False)])
    )

    for track_id in (1, 2):
        collected = feed.collected_vectors(first[track_id])
        assert collected.faces == ()
        assert len(collected.appearances) == 1
    collected = feed.collected_vectors(second[3])
    assert len(collected.faces) == 1
    assert collected.appearances == ()


def test_skips_empty_and_bad_vectors(feed, vision):
    no_model = tracked(1)
    no_model.face_model = ""
    zero = tracked(2)
    zero.face_embedding = [0.0] * DIM
    not_finite = tracked(3)
    not_finite.appearance_embedding = [float("nan")] * DIM
    got = keys(deliver(feed, vision, [no_model, zero, not_finite]))

    assert feed.collected_vectors(got[1]).faces == ()
    assert feed.collected_vectors(got[2]).faces == ()
    assert feed.collected_vectors(got[3]).appearances == ()
    assert len(feed.collected_vectors(got[3]).faces) == 1


@pytest.fixture
def moved_clock(monkeypatch):
    """A clock that can be moved on, for the feed's track ages only: set [0]."""
    offset = [0.0]
    monkeypatch.setattr(
        vision_feed_module, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0])
    )
    return offset


def test_drops_a_track_unseen_for_60_s(feed, vision, moved_clock):
    first = keys(deliver(feed, vision, [tracked(1), tracked(2)]))
    moved_clock[0] = 30.0
    assert keys(deliver(feed, vision, [tracked(2, key="2 again")])) == {2: first[2]}

    moved_clock[0] = 59.0
    assert len(feed.collected_vectors(first[1]).faces) == 1
    assert len(feed.collected_vectors(first[2]).faces) == 2

    moved_clock[0] = 61.0
    assert feed.collected_vectors(first[1]) == ((), ())
    assert len(feed.collected_vectors(first[2]).faces) == 2

    # A new array drops the unseen tracks too, not only a read.
    moved_clock[0] = 200.0
    key_3 = keys(deliver(feed, vision, [tracked(3)]))[3]
    assert feed.collected_vectors(first[2]) == ((), ())
    assert len(feed.collected_vectors(key_3).faces) == 1


def test_a_reused_track_id_is_a_new_generation(feed, vision, moved_clock):
    sam_frame = deliver(feed, vision, [tracked(7, key="sam")])
    [sam] = sam_frame.tracks
    # Sam leaves; once the collection is dropped, walkie-vision gives Bob track 7.
    moved_clock[0] = 61.0
    [bob] = deliver(feed, vision, [tracked(7, key="bob")]).tracks

    assert bob.track_id == sam.track_id == 7
    assert bob.generation != sam.generation
    # Sam's handle collects nothing of Bob's, and Bob's has only his own.
    assert feed.collected_vectors(sam) == ((), ())
    [bob_face] = feed.collected_vectors(bob).faces
    assert np.array_equal(bob_face.vector.vector, hash_vector("face bob", DIM))
    # Sam's frame still names Sam's collection.
    assert feed.people_at(sam_frame.msg.header.stamp, 0.0, NO_CANCEL).tracks == (sam,)


# -- People listeners ---------------------------------------------------------------------


def test_a_people_listener_gets_each_frame_after_the_lock_is_released(feed, vision):
    heard = []

    def listener(frame):
        free = feed._lock.acquire(blocking=False)
        if free:
            feed._lock.release()
        # With the lock free, a listener may read the feed.
        heard.append((frame, free, feed.latest_people(60.0, 0.0, NO_CANCEL).msg))

    feed.on_people(listener)
    msg = people_array([tracked(1)], vision.stamp())
    feed._on_people(msg)
    # A late array reaches the listener too; it compares the stamps itself.
    late = people_array([tracked(2)], vision.stamp(offset_s=-0.5))
    feed._on_people(late)

    [(frame, free, newest), (late_frame, late_free, still_newest)] = heard
    assert free and late_free
    assert frame.msg is msg and newest is msg
    assert frame.tracks == feed.people_at(msg.header.stamp, 0.0, NO_CANCEL).tracks
    assert late_frame.msg is late and still_newest is msg


def test_a_failing_people_listener_is_logged_and_the_others_still_run(feed, vision, caplog):
    heard = []

    def broken(frame):
        raise RuntimeError("listener bug")

    feed.on_people(broken)
    feed.on_people(heard.append)
    msg = people_array([tracked(1)], vision.stamp())

    with caplog.at_level(logging.ERROR, logger="walkie_agent.perception.vision_feed"):
        feed._on_people(msg)

    assert [frame.msg for frame in heard] == [msg]
    assert "people listener failed" in caplog.text
    assert "listener bug" in caplog.text
    assert feed.latest_people(60.0, 0.0, NO_CANCEL).msg is msg


def test_a_removed_people_listener_is_not_called(feed, vision):
    heard, other = [], []
    feed.on_people(heard.append)
    feed.on_people(other.append)
    feed._on_people(people_array([tracked(1)], vision.stamp()))

    feed.remove_people_listener(heard.append)
    # Removing a callback that is not a listener does nothing.
    feed.remove_people_listener(heard.append)
    feed.remove_people_listener(print)
    feed._on_people(people_array([tracked(1)], vision.stamp()))

    assert (len(heard), len(other)) == (1, 2)


def test_a_published_array_reaches_the_listener_on_the_executor_thread(feed, vision):
    heard = []
    feed.on_people(lambda frame: heard.append((frame, threading.current_thread())))

    sent = vision.publish_people([tracked(1)])

    assert wait_until(lambda: heard)
    [(frame, thread)] = heard
    assert frame.msg.header.stamp == sent.header.stamp
    assert [key.track_id for key in frame.tracks] == [1]
    assert thread is not threading.current_thread()


# -- Stamp order and clock epochs, through the callbacks --------------------------------


def test_a_late_older_array_does_not_hide_a_newer_one(feed, vision):
    fresh = vision.stamp()
    feed._on_people(people_array([tracked(1)], fresh))
    feed._on_objects(objects_array(["chair"], fresh))
    # Delayed by 2 s on the way: older than the limit, and it came last.
    late = vision.stamp(-2.0)
    feed._on_people(people_array([tracked(2)], late))
    feed._on_objects(objects_array(["sofa"], late))

    assert feed.latest_people(1.0, 0.0, NO_CANCEL).msg.header.stamp == fresh
    assert feed.latest_objects(1.0, 0.0, NO_CANCEL).header.stamp == fresh
    # The late array is kept all the same, for find_seats to pair by stamp.
    assert [p.track_id for p in feed.people_at(late, 0.0, NO_CANCEL).msg.people] == [2]


def test_a_late_older_array_does_not_push_a_newer_one_out(connector, vision):
    feed = VisionFeed(connector, VisionFeedConfig(history=3, vectors_per_track=3))
    newer = [vision.stamp() for _ in range(3)]
    for stamp in newer:
        feed._on_people(people_array([tracked(1, key=f"1 at {stamp_ns(stamp)}")], stamp))
    late = vision.stamp(-1.0)
    feed._on_people(people_array([tracked(1, key="1 late")], late))

    # The late array has the oldest stamp, so it is the one that goes.
    for stamp in newer:
        assert feed.people_at(stamp, 0.0, NO_CANCEL).msg.header.stamp == stamp
    with pytest.raises(StepError, match="UNAVAILABLE"):
        feed.people_at(late, 0.0, NO_CANCEL)
    # So does its sample, from a full collection.
    [key] = feed.latest_people(1.0, 0.0, NO_CANCEL).tracks
    collected = feed.collected_vectors(key)
    assert [s.stamp_ns for s in collected.faces] == [stamp_ns(stamp) for stamp in newer]


def test_a_big_jump_back_starts_a_new_epoch(feed, vision):
    before = vision.stamp()
    feed._on_people(people_array([tracked(7, key="sam")], before))
    feed._on_objects(objects_array(["chair"], before))
    [sam] = feed.latest_people(1.0, 0.0, NO_CANCEL).tracks
    # A simulation restart: the clock goes back by more than epoch_jump_s,
    # and the restarted tracker gives track 7 to Bob.
    after = vision.stamp(-10.0)
    feed._on_people(people_array([tracked(7, key="bob")], after))
    feed._on_objects(objects_array(["sofa"], after))

    # The old run's arrays are gone, so the new run's are the newest.
    with pytest.raises(StepError, match="UNAVAILABLE"):
        feed.people_at(before, 0.0, NO_CANCEL)
    frame = feed.latest_people(20.0, 0.0, NO_CANCEL)
    assert frame.msg.header.stamp == after
    assert feed.latest_objects(20.0, 0.0, NO_CANCEL).header.stamp == after
    # The old run's collections are gone too: Bob's track 7 is a new generation.
    [bob] = frame.tracks
    assert bob.generation != sam.generation
    assert feed.collected_vectors(sam) == ((), ())
    assert [s.stamp_ns for s in feed.collected_vectors(bob).faces] == [stamp_ns(after)]


def test_reads_and_arrivals_on_two_threads_stay_consistent(feed, vision):
    count = 1500
    stamps = [vision.stamp() for _ in range(count)]
    index = {stamp_ns(stamp): i for i, stamp in enumerate(stamps)}
    expected = {
        (track_id, i): hash_vector(f"face {track_id} {i}", DIM)
        for track_id in (1, 2)
        for i in range(count)
    }
    people = [
        people_array([tracked(1, key=f"1 {i}"), tracked(2, key=f"2 {i}")], stamp)
        for i, stamp in enumerate(stamps)
    ]
    objects = [objects_array(["chair"], stamp) for stamp in stamps]
    feed._on_people(people[0])
    first = keys(feed.latest_people(60.0, 0.0, NO_CANCEL))
    errors = []

    def arrive():
        try:
            for people_msg, objects_msg in zip(people[1:], objects[1:], strict=True):
                feed._on_people(people_msg)
                feed._on_objects(objects_msg)
        except Exception as error:  # noqa: BLE001 - reported by the reading thread
            errors.append(error)

    # A short switch interval, so the two threads change places often.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    writer = threading.Thread(target=arrive)
    try:
        writer.start()
        newest = reads = 0
        while writer.is_alive() or reads == 0:
            frame = feed.latest_people(60.0, 0.0, NO_CANCEL)
            frame_ns = stamp_ns(frame.msg.header.stamp)
            assert frame_ns >= newest
            newest = frame_ns
            assert len(frame.tracks) == len(frame.msg.people) == 2
            assert keys(frame) == first
            for key in frame.tracks:
                collected = feed.collected_vectors(key)
                face_stamps = [s.stamp_ns for s in collected.faces]
                assert 0 < len(face_stamps) <= 20
                assert face_stamps == sorted(face_stamps)
                assert face_stamps == [s.stamp_ns for s in collected.appearances]
                assert face_stamps[-1] >= frame_ns
                for sample in collected.faces:
                    i = index[sample.stamp_ns]
                    assert np.array_equal(sample.vector.vector, expected[key.track_id, i])
            try:
                feed.latest_objects(60.0, 0.0, NO_CANCEL)
            except StepError as error:
                assert error.code == "UNAVAILABLE"
            reads += 1
    finally:
        writer.join()
        sys.setswitchinterval(interval)

    assert errors == []
    assert reads > 1
    assert feed.latest_people(60.0, 0.0, NO_CANCEL).msg.header.stamp == stamps[-1]
    assert feed.latest_objects(60.0, 0.0, NO_CANCEL).header.stamp == stamps[-1]
