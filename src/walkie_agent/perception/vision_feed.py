"""VisionFeed: the newest walkie-vision detections, for the perception atomics.

walkie-vision publishes one PersonDetectionArray and one ObjectDetectionArray
for every image it processes, also when it found nothing (vision.md, "Topics
and services"). VisionFeed is the `vision` backend: it subscribes to both
once, keeps the newest arrays, and collects the face and appearance vectors
of every visible track, so remember_person can store a guest from many
frames rather than one (hri.md, "Faces collected while talking"). The
atomics read it and never subscribe themselves.

It only collects vectors. Outlier rejection and matching are
walkie_agent.perception.identity's, applied when a person is remembered or
recognized.

The callbacks run on the connector's executor thread and the atomics on the
agent's thread, so all state is behind one lock.
"""

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, TypeVar

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from rai.communication.ros2 import ROS2Connector
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetection, PersonDetectionArray

from walkie_agent.perception.people_memory import ModelVector
from walkie_agent.toolbox.types import StepError

PEOPLE_TOPIC = "/walkie_vision/people"
OBJECTS_TOPIC = "/walkie_vision/objects"

# BEST_EFFORT matches both a reliable and a best-effort publisher, where a
# reliable subscription would not match a best-effort one; vision.md fixes no
# QoS. VOLATILE and a short depth, because only the newest images matter.
VISION_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

# Added to every UNAVAILABLE message: one of the two publishes these topics.
PUBLISHER_HINT = "Is walkie-vision or oracle_perception running?"

# How often a wait looks at the cancel event and the clock while no array
# arrives, as ToolContext.wait() does.
_POLL_S = 0.05

_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class VisionFeedConfig:
    """What VisionFeed keeps. The face limits are walkie-agent-v2's, tuned on the robot."""

    # Arrays kept per topic, newest last: about two seconds of the camera's
    # 15 Hz, so find_seats can find the people array of an objects array's
    # stamp after both have arrived.
    history: int = 30
    # Face vectors, and separately appearance vectors, kept per track: the
    # newest ones.
    vectors_per_track: int = 20
    # Tracks kept at once. When a new track would exceed it, the track seen
    # longest ago is dropped.
    max_tracks: int = 32
    # A track unseen for this many seconds is dropped with its vectors, so a
    # guest who left is not remembered from old frames.
    track_ttl_s: float = 60.0
    # v2's HRI_RECOG_MIN_DET_SCORE: below this face_det_score the embedding
    # is too noisy to name someone by. Tuned on the robot.
    min_face_det_score: float = 0.5
    # v2's HRI_RECOG_MIN_FACE_AREA_PX: about 50 x 50 px of face_bbox, which a
    # person across a living room still clears. Tuned on the robot.
    min_face_area_px2: float = 2500.0

    def __post_init__(self) -> None:
        for name in ("history", "vectors_per_track", "max_tracks"):
            if getattr(self, name) < 1:
                raise ValueError(f"VisionFeedConfig.{name} must be at least 1.")
        if not self.track_ttl_s > 0:
            raise ValueError("VisionFeedConfig.track_ttl_s must be positive.")


DEFAULT_CONFIG = VisionFeedConfig()


def face_trusted(person: PersonDetection, config: VisionFeedConfig = DEFAULT_CONFIG) -> bool:
    """True when the person's face is good enough to name them by (v2's _face_trusted).

    The face needs both the detector's confidence and an on-screen size: a
    small or doubtful face gives a noisy embedding, and storing it would
    pull the guest's stored faces toward someone else's.
    """
    area = person.face_bbox.size_x * person.face_bbox.size_y
    return (
        person.face_det_score >= config.min_face_det_score
        and area >= config.min_face_area_px2
    )


class CollectedVectors(NamedTuple):
    """The vectors collected for one track, oldest first, each with its model name."""

    faces: tuple[ModelVector, ...]
    appearances: tuple[ModelVector, ...]


_NONE_COLLECTED = CollectedVectors(faces=(), appearances=())


@dataclass
class _Track:
    # time.monotonic() of the newest people array that showed the track.
    last_seen: float
    # Bounded at VisionFeedConfig.vectors_per_track, so the oldest goes first.
    faces: deque[ModelVector]
    appearances: deque[ModelVector]


_Array = TypeVar("_Array", PersonDetectionArray, ObjectDetectionArray)


class VisionFeed:
    """The `vision` backend: walkie-vision's newest arrays and the vectors of every track.

    Created once per agent. Without a publisher it does no harm: a read then
    fails with UNAVAILABLE. The arrays it returns are the received messages
    themselves, shared with other callers, so they must not be modified.
    """

    def __init__(self, connector: ROS2Connector, config: VisionFeedConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._node = connector.node
        self._lock = threading.Lock()
        # Notified on every array, so a wait ends as soon as one arrives.
        self._arrived = threading.Condition(self._lock)
        # In arrival order rather than sorted by stamp: a simulation restart
        # moves the stamps back, and the newest array must still be the one
        # that arrived last.
        self._people: deque[PersonDetectionArray] = deque(maxlen=config.history)
        self._objects: deque[ObjectDetectionArray] = deque(maxlen=config.history)
        # By track id, the track seen longest ago first.
        self._tracks: OrderedDict[int, _Track] = OrderedDict()
        # Created here, once: rclpy entities must not be created per call.
        self._subscriptions = (
            self._node.create_subscription(
                PersonDetectionArray, PEOPLE_TOPIC, self._on_people, VISION_QOS
            ),
            self._node.create_subscription(
                ObjectDetectionArray, OBJECTS_TOPIC, self._on_objects, VISION_QOS
            ),
        )

    # -- Reads, from the agent thread --------------------------------------------

    def latest_people(
        self, max_age_s: float, wait_s: float, cancel: threading.Event
    ) -> PersonDetectionArray:
        """The newest people array no older than `max_age_s` on the node clock.

        Waits up to `wait_s` for one. Raises StepError CANCELLED once
        `cancel` is set (an atomic passes ctx.cancel), and UNAVAILABLE,
        naming the topic, when no fresh array came in time.
        """
        return self._latest(self._people, PEOPLE_TOPIC, max_age_s, wait_s, cancel)

    def latest_objects(
        self, max_age_s: float, wait_s: float, cancel: threading.Event
    ) -> ObjectDetectionArray:
        """The newest objects array no older than `max_age_s`; see latest_people()."""
        return self._latest(self._objects, OBJECTS_TOPIC, max_age_s, wait_s, cancel)

    def people_at(
        self, stamp: TimeMsg, wait_s: float, cancel: threading.Event
    ) -> PersonDetectionArray:
        """The people array whose header.stamp equals `stamp`, such as an objects array's.

        walkie-vision gives both arrays of one image that image's header, so
        find_seats pairs them by stamp to see who sits where in one image.
        Waits up to `wait_s`, since the people array may arrive after the
        objects array. Raises StepError CANCELLED once `cancel` is set, and
        UNAVAILABLE when no such array came in time.
        """
        _check_seconds(wait_s, "wait_s")
        key = (stamp.sec, stamp.nanosec)
        deadline = time.monotonic() + wait_s
        with self._arrived:
            while True:
                for msg in reversed(self._people):
                    if (msg.header.stamp.sec, msg.header.stamp.nanosec) == key:
                        return msg
                remaining = self._remaining(deadline, cancel)
                if remaining <= 0:
                    raise StepError(
                        "UNAVAILABLE",
                        f"No message on {PEOPLE_TOPIC} within {wait_s:g} s has the stamp "
                        f"{stamp.sec}.{stamp.nanosec:09d} of the image. {PUBLISHER_HINT}",
                    )
                self._arrived.wait(min(_POLL_S, remaining))

    def collected_vectors(self, track_id: int) -> CollectedVectors:
        """The face and appearance vectors collected for a track, oldest first.

        Only trusted faces (face_trusted()) are collected; appearances need
        no such check. Track id 0 means untracked, so it has none, and so
        has a track unseen for `track_ttl_s`. The arrays are read-only.
        """
        if track_id == 0:
            return _NONE_COLLECTED
        with self._lock:
            self._drop_unseen(time.monotonic())
            track = self._tracks.get(track_id)
            if track is None:
                return _NONE_COLLECTED
            return CollectedVectors(tuple(track.faces), tuple(track.appearances))

    # -- Callbacks, on the executor thread ---------------------------------------

    def _on_people(self, msg: PersonDetectionArray) -> None:
        # The vectors are copied before the lock is taken, so a reader never
        # waits for the conversion.
        sightings = [
            (
                person.track_id,
                model_vector(person.face_model, person.face_embedding)
                if face_trusted(person, self.config)
                else None,
                model_vector(person.appearance_model, person.appearance_embedding),
            )
            for person in msg.people
            if person.track_id != 0
        ]
        now = time.monotonic()
        with self._arrived:
            self._people.append(msg)
            self._collect(sightings, now)
            self._arrived.notify_all()

    def _on_objects(self, msg: ObjectDetectionArray) -> None:
        with self._arrived:
            self._objects.append(msg)
            self._arrived.notify_all()

    # -- Helpers, called with the lock held ----------------------------------------

    def _collect(
        self,
        sightings: list[tuple[int, ModelVector | None, ModelVector | None]],
        now: float,
    ) -> None:
        self._drop_unseen(now)
        limit = self.config.vectors_per_track
        for track_id, face, appearance in sightings:
            track = self._tracks.get(track_id)
            if track is None:
                track = _Track(now, deque(maxlen=limit), deque(maxlen=limit))
                self._tracks[track_id] = track
            track.last_seen = now
            self._tracks.move_to_end(track_id)
            if face is not None:
                track.faces.append(face)
            if appearance is not None:
                track.appearances.append(appearance)
        while len(self._tracks) > self.config.max_tracks:
            self._tracks.popitem(last=False)

    def _drop_unseen(self, now: float) -> None:
        # Monotonic time rather than the node clock: dropping is about memory,
        # and a simulation restart moves the node clock back, which would keep
        # old tracks forever. The tracks are in order of last sighting, so the
        # unseen ones are at the front.
        cutoff = now - self.config.track_ttl_s
        while self._tracks and next(iter(self._tracks.values())).last_seen < cutoff:
            self._tracks.popitem(last=False)

    def _latest(
        self,
        history: deque[_Array],
        topic: str,
        max_age_s: float,
        wait_s: float,
        cancel: threading.Event,
    ) -> _Array:
        _check_seconds(max_age_s, "max_age_s")
        _check_seconds(wait_s, "wait_s")
        deadline = time.monotonic() + wait_s
        with self._arrived:
            while True:
                newest = history[-1] if history else None
                # The age is checked on every pass, since the node clock moves
                # on while the newest array stays the same.
                age_s = self._age_s(newest) if newest is not None else None
                if age_s is not None and abs(age_s) <= max_age_s:
                    return newest
                remaining = self._remaining(deadline, cancel)
                if remaining <= 0:
                    raise StepError(
                        "UNAVAILABLE", _stale_message(topic, age_s, max_age_s, wait_s)
                    )
                self._arrived.wait(min(_POLL_S, remaining))

    def _age_s(self, msg: PersonDetectionArray | ObjectDetectionArray) -> float:
        # The node clock, so the age is right under sim time, where the stamps
        # come from /clock as well.
        stamp = msg.header.stamp
        now_ns = self._node.get_clock().now().nanoseconds
        return (now_ns - (stamp.sec * _NS_PER_S + stamp.nanosec)) / _NS_PER_S

    @staticmethod
    def _remaining(deadline: float, cancel: threading.Event) -> float:
        if cancel.is_set():
            raise StepError("CANCELLED", "A stop or a reset interrupted the call.")
        return deadline - time.monotonic()


def model_vector(model: str, values: Sequence[float]) -> ModelVector | None:
    """A message's embedding as a read-only float32 array with its model, or None.

    A vector with no model name, or one that is empty, zero or not finite,
    is None: it cannot be matched, and one bad vector would spoil the mean
    that outlier rejection compares against. find_people converts a
    handle's vectors with it too, so a handle and its track's collection
    follow one rule.
    """
    if not model:
        return None
    vector = np.array(values, dtype=np.float32)
    if vector.size == 0 or not np.isfinite(vector).all() or not vector.any():
        return None
    vector.flags.writeable = False
    return ModelVector(model, vector)


def _stale_message(topic: str, age_s: float | None, max_age_s: float, wait_s: float) -> str:
    if age_s is None:
        return f"No message on {topic} within {wait_s:g} s. {PUBLISHER_HINT}"
    if age_s < 0:
        # A stamp far ahead of the node clock: the publisher and the agent
        # disagree on use_sim_time, so every array would look stale or new.
        return (
            f"The newest message on {topic} is stamped {-age_s:.1f} s ahead of this "
            "node's clock, so the two use different clocks. Check that both or "
            "neither use sim time."
        )
    return (
        f"The newest message on {topic} is {age_s:.1f} s old, older than "
        f"{max_age_s:g} s. {PUBLISHER_HINT}"
    )


def _check_seconds(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number of seconds, not {value}.")
