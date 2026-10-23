"""VisionFeed over ROS, the in-memory people memory, and the fake vectors.

The VisionFeed tests publish walkie_vision_msgs arrays from a second node on
the `connector` fixture's Fast DDS domain (ADR-0019), with hash_vector
vectors under walkie-vision's fake model names, as oracle_perception will.
"""

import random
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
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


@pytest.mark.parametrize(
    ("score", "side_px", "trusted"),
    [
        (0.9, 60.0, True),
        (0.5, 50.0, True),
        (0.49, 60.0, False),
        (0.9, 49.0, False),
        # No face found: walkie-vision leaves the score and the box at 0.
        (0.0, 0.0, False),
    ],
)
def test_face_trusted_needs_score_and_area(score, side_px, trusted):
    person = PersonDetection(face_det_score=score)
    person.face_bbox.size_x = side_px
    person.face_bbox.size_y = side_px

    assert face_trusted(person) is trusted


def test_vision_feed_config_defaults():
    config = VisionFeedConfig()

    assert (config.vectors_per_track, config.max_tracks, config.track_ttl_s) == (20, 32, 60.0)
    assert (config.min_face_det_score, config.min_face_area_px2) == (0.5, 2500.0)
    assert config.history == 30


# -- VisionFeed over ROS ---------------------------------------------------------------


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
        msg = PersonDetectionArray(people=list(people))
        msg.header.stamp = self.stamp() if stamp is None else stamp
        msg.header.frame_id = "zed_head_left_camera_optical_frame"
        msg.camera_info.width, msg.camera_info.height = 1280, 720
        self.people.publish(msg)
        return msg

    def publish_objects(self, labels, stamp: TimeMsg | None = None) -> ObjectDetectionArray:
        msg = ObjectDetectionArray(
            classes=sorted(set(labels)), detections=[ObjectDetection(label=x) for x in labels]
        )
        msg.header.stamp = self.stamp() if stamp is None else stamp
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
    return [p.track_id for p in feed.latest_people(1.0, 0.0, NO_CANCEL).people]


def deliver(feed, vision, people) -> PersonDetectionArray:
    """Publish a people array and return once the feed has it.

    One at a time: the subscription keeps only 5, so a burst could lose some.
    """
    sent = vision.publish_people(people)
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

    assert got.header.stamp == sent.header.stamp
    assert [p.track_id for p in got.people] == [4]
    assert (got.camera_info.width, got.camera_info.height) == (1280, 720)


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
    assert feed.latest_people(10.0, 0.0, NO_CANCEL).header.stamp == stale.header.stamp
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

    assert [p.track_id for p in got.people] == [9]


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
    assert people.header.stamp == image
    assert [p.track_id for p in people.people] == [2]
    # Once the newer array is in, the pair is still the array with the stamp.
    assert wait_until(lambda: newest_tracks(feed) == [3])
    assert [p.track_id for p in feed.people_at(image, 0.0, NO_CANCEL).people] == [2]


def test_people_at_without_that_stamp_is_unavailable(feed, vision):
    deliver(feed, vision, [tracked(1)])

    with pytest.raises(StepError) as error:
        feed.people_at(TimeMsg(sec=12, nanosec=5), 0.1, NO_CANCEL)

    assert error.value.code == "UNAVAILABLE"
    assert PEOPLE_TOPIC in error.value.message
    assert "12.000000005" in error.value.message


def test_collects_the_newest_20_vectors_per_track_with_their_models(feed, vision):
    for frame in range(25):
        deliver(feed, vision, [tracked(7, key=f"7 frame {frame}")])

    collected = feed.collected_vectors(7)

    assert len(collected.faces) == len(collected.appearances) == 20
    assert {v.model for v in collected.faces} == {FAKE_FACE_MODEL}
    assert {v.model for v in collected.appearances} == {FAKE_REID_MODEL}
    expected = [hash_vector(f"face 7 frame {frame}", DIM) for frame in range(5, 25)]
    assert all(np.array_equal(v.vector, e) for v, e in zip(collected.faces, expected, strict=True))
    assert not collected.faces[0].vector.flags.writeable
    assert feed.collected_vectors(8) == ((), ())


def test_keeps_32_tracks_and_skips_track_0(feed, vision):
    # 32 tracks and an untracked person: track 0 takes no place.
    deliver(feed, vision, [tracked(n) for n in range(1, 33)] + [tracked(0)])

    assert all(len(feed.collected_vectors(n).faces) == 1 for n in range(1, 33))
    assert feed.collected_vectors(0) == ((), ())

    # Track 2 is seen again, so track 1 is now the one seen longest ago.
    deliver(feed, vision, [tracked(2, key="2 again")])
    deliver(feed, vision, [tracked(33)])

    assert feed.collected_vectors(1) == ((), ())
    assert len(feed.collected_vectors(2).faces) == 2
    assert all(len(feed.collected_vectors(n).faces) == 1 for n in range(3, 34))


def test_skips_untrusted_faces_but_keeps_their_appearance(feed, vision):
    deliver(feed, vision, [tracked(1, score=0.4), tracked(2, side_px=40.0)])
    deliver(feed, vision, [tracked(3, score=0.5, side_px=50.0, with_appearance=False)])

    for track_id in (1, 2):
        collected = feed.collected_vectors(track_id)
        assert collected.faces == ()
        assert len(collected.appearances) == 1
    collected = feed.collected_vectors(3)
    assert len(collected.faces) == 1
    assert collected.appearances == ()


def test_skips_empty_and_bad_vectors(feed, vision):
    no_model = tracked(1)
    no_model.face_model = ""
    zero = tracked(2)
    zero.face_embedding = [0.0] * DIM
    not_finite = tracked(3)
    not_finite.appearance_embedding = [float("nan")] * DIM
    deliver(feed, vision, [no_model, zero, not_finite])

    assert feed.collected_vectors(1).faces == ()
    assert feed.collected_vectors(2).faces == ()
    assert feed.collected_vectors(3).appearances == ()
    assert len(feed.collected_vectors(3).faces) == 1


def test_drops_a_track_unseen_for_60_s(feed, vision, monkeypatch):
    # A clock that can be moved on, for the feed's track ages only.
    offset = [0.0]
    monkeypatch.setattr(
        vision_feed_module, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0])
    )
    deliver(feed, vision, [tracked(1), tracked(2)])
    offset[0] = 30.0
    deliver(feed, vision, [tracked(2, key="2 again")])

    offset[0] = 59.0
    assert len(feed.collected_vectors(1).faces) == 1
    assert len(feed.collected_vectors(2).faces) == 2

    offset[0] = 61.0
    assert feed.collected_vectors(1) == ((), ())
    assert len(feed.collected_vectors(2).faces) == 2

    # A new array drops the unseen tracks too, not only a read.
    offset[0] = 200.0
    deliver(feed, vision, [tracked(3)])
    assert feed.collected_vectors(2) == ((), ())
    assert len(feed.collected_vectors(3).faces) == 1
