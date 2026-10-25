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
agent's thread, so all state is behind one lock. The gaze loop needs every
people array as it comes rather than the newest one when it asks, so the
feed also hands each one to its people listeners (on_people).
"""

import bisect
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, NamedTuple, TypeVar

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from rai.communication.ros2 import ROS2Connector
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetection, PersonDetectionArray

from walkie_agent.perception.front import face_size_rad
from walkie_agent.perception.people_memory import ModelVector
from walkie_agent.toolbox.types import StepError

logger = logging.getLogger(__name__)

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

    # Arrays kept per topic, by stamp: about two seconds of the camera's
    # 15 Hz, so find_seats can find the people array of an objects array's
    # stamp after both have arrived.
    history: int = 30
    # Face vectors, and separately appearance vectors, kept per track: the
    # newest ones by stamp.
    vectors_per_track: int = 20
    # Tracks kept at once. When a new track would exceed it, the track seen
    # longest ago is dropped.
    max_tracks: int = 32
    # A track unseen for this many seconds is dropped with its vectors, so a
    # guest who left is not remembered from old frames.
    track_ttl_s: float = 60.0
    # An array stamped more than this many seconds before the newest one of
    # its topic starts a new clock epoch, and the history is cleared first.
    # A simulation restart moves the clock back by far more, while a late
    # array is behind by well under a second at 15 Hz and a depth of 5. A
    # smaller jump back is not detected: the newest array is right again
    # once the new stamps pass the old ones, but the track collections are
    # kept, so a restart under this many seconds into a run can put a new
    # person in an old collection (hri.md, "Faces collected while talking").
    epoch_jump_s: float = 5.0
    # v2's HRI_RECOG_MIN_DET_SCORE: below this face_det_score the embedding
    # is too noisy to name someone by. Tuned on the robot.
    min_face_det_score: float = 0.5
    # The smallest face to name someone by, as the angle its face_bbox spans:
    # sqrt((size_x / fx) * (size_y / fy)) radians, with fx and fy from the
    # array's camera_info. An angle holds for any image size, where a pixel
    # area would not: the simulation's head camera is 640 x 360 with fx
    # 224 px. v2's HRI_RECOG_MIN_FACE_AREA_PX was 2500 px2, about 50 x 50
    # px, which a person across a living room still clears, tuned on the
    # robot's ZED 2i at HD1080 and later HD2K. Stereolabs gives both the
    # same nominal fx, 1066 px, as HD1080 is a crop of HD2K. The robot's
    # own calibration was not read, so check it there: 50 px over its k[0].
    # 50 / 1066 is 0.0469 rad, 2.7 degrees, and 10.5 px in the simulation.
    # config.toml can set it: [walkie.perception.vision_feed]
    # min_face_size_rad.
    min_face_size_rad: float = 0.0469

    def __post_init__(self) -> None:
        for name in ("history", "vectors_per_track", "max_tracks"):
            if getattr(self, name) < 1:
                raise ValueError(f"VisionFeedConfig.{name} must be at least 1.")
        for name in ("track_ttl_s", "epoch_jump_s", "min_face_size_rad"):
            if not getattr(self, name) > 0:
                raise ValueError(f"VisionFeedConfig.{name} must be positive.")


DEFAULT_CONFIG = VisionFeedConfig()


def face_trusted(
    person: PersonDetection, camera_info: CameraInfo, config: VisionFeedConfig = DEFAULT_CONFIG
) -> bool:
    """True when the person's face is good enough to name them by (v2's _face_trusted).

    The face needs both the detector's confidence and a size: a small or
    doubtful face gives a noisy embedding, and storing it would pull the
    guest's stored faces toward someone else's. `camera_info` is the one
    of the person's array, whose image the face box is in pixels of.
    """
    if person.face_det_score < config.min_face_det_score:
        return False
    # The box from its own size, not its corners: centre minus and plus
    # half a size is not exactly the size again in floats, and a face on
    # the limit would then fall just below it.
    box = (0.0, 0.0, person.face_bbox.size_x, person.face_bbox.size_y)
    size = face_size_rad(box, camera_info)
    # None for a camera_info that was never filled in: the face's size is
    # then unknown, and an unknown face names nobody.
    return size is not None and size >= config.min_face_size_rad


class TrackKey(NamedTuple):
    """One collection of vectors: its track id and the generation of the collection.

    A track id can name someone else later: walkie-vision resets its track
    ids when it restarts or changes its class list (vision.md), and
    VisionFeed drops a collection it has not seen for `track_ttl_s`. Each
    new collection takes the next number of one feed-wide count, so a
    handle that keeps its person's key never reads the vectors of whoever
    has the track id next.
    """

    track_id: int
    generation: int


class PeopleFrame(NamedTuple):
    """A people array, with the collection each person's vectors went to on arrival."""

    msg: PersonDetectionArray
    # One per person, in msg.people order, and None for an untracked person
    # (track id 0). Recorded when the array came, so a handle made from it
    # names the collection of the person in that image, also once the track
    # id has gone to someone else.
    tracks: tuple[TrackKey | None, ...]


PeopleListener = Callable[[PeopleFrame], None]


class Sample(NamedTuple):
    """One collected vector, with the header.stamp of the people array it came from."""

    # So a caller can leave out the sample of the image its handle came
    # from, whose vector the handle holds already.
    stamp_ns: int
    vector: ModelVector


class CollectedVectors(NamedTuple):
    """The samples collected for one track generation, oldest stamp first."""

    faces: tuple[Sample, ...]
    appearances: tuple[Sample, ...]


_NONE_COLLECTED = CollectedVectors(faces=(), appearances=())


@dataclass
class _Track:
    # The TrackKey generation of this collection.
    generation: int
    # time.monotonic() of the newest people array that showed the track.
    last_seen: float
    # By stamp, oldest first, at most VisionFeedConfig.vectors_per_track.
    faces: list[Sample]
    appearances: list[Sample]


_Item = TypeVar("_Item")


class _StampHistory(Generic[_Item]):
    """The newest arrays of one topic by header.stamp, oldest first, within one clock epoch.

    By stamp rather than in arrival order: an array that comes late must
    neither hide a newer one from the reads nor push a newer one out of the
    history. A stamp more than `epoch_jump_s` before the newest one starts a
    new epoch and clears the history: only a clock that moved back (a
    simulation restart) stamps that far behind, and the old run's arrays
    would otherwise stay the newest until the new run's stamps passed them.
    """

    def __init__(self, size: int, epoch_jump_s: float) -> None:
        self._size = size
        self._epoch_jump_ns = epoch_jump_s * _NS_PER_S
        # Two parallel lists, so that bisect searches the stamps alone.
        self._stamps: list[int] = []
        self._items: list[_Item] = []

    def starts_epoch(self, stamp_ns: int) -> bool:
        return bool(self._stamps) and self._stamps[-1] - stamp_ns > self._epoch_jump_ns

    def add(self, stamp_ns: int, item: _Item) -> None:
        if self.starts_epoch(stamp_ns):
            self._stamps.clear()
            self._items.clear()
        # After any equal stamps, so that at() finds the array that came last.
        index = bisect.bisect_right(self._stamps, stamp_ns)
        self._stamps.insert(index, stamp_ns)
        self._items.insert(index, item)
        if len(self._stamps) > self._size:
            # The oldest stamp goes, even when it is the array that just came.
            del self._stamps[0]
            del self._items[0]

    def newest(self) -> tuple[int, _Item] | None:
        if not self._stamps:
            return None
        return self._stamps[-1], self._items[-1]

    def at(self, stamp_ns: int) -> _Item | None:
        index = bisect.bisect_right(self._stamps, stamp_ns) - 1
        if index >= 0 and self._stamps[index] == stamp_ns:
            return self._items[index]
        return None


class VisionFeed:
    """The `vision` backend: walkie-vision's newest arrays and the vectors of every track.

    Created once per agent. Without a publisher it does no harm: a read then
    fails with UNAVAILABLE. The arrays it returns or hands to a listener,
    alone or in a PeopleFrame, are the received messages themselves, shared
    with other callers, so they must not be modified.
    """

    def __init__(self, connector: ROS2Connector, config: VisionFeedConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._node = connector.node
        self._lock = threading.Lock()
        # Notified on every array, so a wait ends as soon as one arrives.
        self._arrived = threading.Condition(self._lock)
        # By stamp within one clock epoch; see _StampHistory.
        self._people: _StampHistory[PeopleFrame] = _StampHistory(
            config.history, config.epoch_jump_s
        )
        self._objects: _StampHistory[ObjectDetectionArray] = _StampHistory(
            config.history, config.epoch_jump_s
        )
        # By track id, the track seen longest ago first.
        self._tracks: OrderedDict[int, _Track] = OrderedDict()
        # The newest TrackKey generation given out. Never reset, so a key
        # names one collection only.
        self._generation = 0
        # Replaced rather than changed in place, under the lock, so a
        # callback can call the listeners it read after releasing the lock
        # while another thread adds or removes one.
        self._people_listeners: tuple[PeopleListener, ...] = ()
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
    ) -> PeopleFrame:
        """The people array with the newest stamp, if no older than `max_age_s` on the node clock.

        Waits up to `wait_s` for one. Raises StepError CANCELLED when
        `cancel` is set (an atomic passes ctx.cancel), also with a fresh
        array at hand, and UNAVAILABLE, naming the topic, when no fresh
        array came in time.
        """
        return self._latest(self._people, PEOPLE_TOPIC, max_age_s, wait_s, cancel)

    def latest_objects(
        self, max_age_s: float, wait_s: float, cancel: threading.Event
    ) -> ObjectDetectionArray:
        """The objects array with the newest stamp; see latest_people()."""
        return self._latest(self._objects, OBJECTS_TOPIC, max_age_s, wait_s, cancel)

    def people_at(self, stamp: TimeMsg, wait_s: float, cancel: threading.Event) -> PeopleFrame:
        """The people array whose header.stamp equals `stamp`, such as an objects array's.

        walkie-vision gives both arrays of one image that image's header, so
        find_seats pairs them by stamp to see who sits where in one image.
        Waits up to `wait_s`, since the people array may arrive after the
        objects array. Raises StepError CANCELLED when `cancel` is set, also
        with the array at hand, and UNAVAILABLE when no such array came in
        time.
        """
        _check_seconds(wait_s, "wait_s")
        stamp_ns = _stamp_ns(stamp)
        deadline = time.monotonic() + wait_s
        with self._arrived:
            while True:
                _check_cancel(cancel)
                frame = self._people.at(stamp_ns)
                if frame is not None:
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StepError(
                        "UNAVAILABLE",
                        f"No message on {PEOPLE_TOPIC} within {wait_s:g} s has the stamp "
                        f"{stamp.sec}.{stamp.nanosec:09d} of the image. {PUBLISHER_HINT}",
                    )
                self._arrived.wait(min(_POLL_S, remaining))

    def collected_vectors(self, key: TrackKey) -> CollectedVectors:
        """The face and appearance samples of one track generation, oldest stamp first.

        `key` comes from a PeopleFrame's tracks. Only trusted faces
        (face_trusted()) are collected; appearances need no such check. A
        generation that is no longer its track id's live one has none: its
        collection was dropped (unseen for `track_ttl_s`, over `max_tracks`,
        or at a new clock epoch), and the id may name someone else now. The
        arrays are read-only.
        """
        with self._lock:
            self._drop_unseen(time.monotonic())
            track = self._tracks.get(key.track_id)
            if track is None or track.generation != key.generation:
                return _NONE_COLLECTED
            return CollectedVectors(tuple(track.faces), tuple(track.appearances))

    # -- People listeners ------------------------------------------------------------

    def on_people(self, callback: PeopleListener) -> None:
        """Call `callback(frame)` with the PeopleFrame of every people array that arrives.

        In the order they arrive, a late array included: a callback that
        wants only newer arrays compares the stamps itself. It runs on the
        connector's executor thread, after the feed's lock is released, so
        it may read the feed. It must return promptly and must not wait for
        TF or for anything else the executor delivers, since nothing else on
        the node runs while it waits. A callback that raises is logged; the
        array is kept and the other callbacks still run.
        """
        with self._lock:
            self._people_listeners = (*self._people_listeners, callback)

    def remove_people_listener(self, callback: PeopleListener) -> None:
        """Stop calling `callback`. Does nothing when it is not a listener.

        An array already being handed out may still reach it once.
        """
        with self._lock:
            listeners = list(self._people_listeners)
            if callback in listeners:
                listeners.remove(callback)
                self._people_listeners = tuple(listeners)

    # -- Callbacks, on the executor thread ---------------------------------------

    def _on_people(self, msg: PersonDetectionArray) -> None:
        # The vectors are copied before the lock is taken, so a reader never
        # waits for the conversion.
        stamp_ns = _stamp_ns(msg.header.stamp)
        sightings = [
            (
                person.track_id,
                _sample(stamp_ns, model_vector(person.face_model, person.face_embedding))
                if face_trusted(person, msg.camera_info, self.config)
                else None,
                _sample(
                    stamp_ns, model_vector(person.appearance_model, person.appearance_embedding)
                ),
            )
            for person in msg.people
            if person.track_id != 0
        ]
        track_ids = [person.track_id for person in msg.people]
        now = time.monotonic()
        with self._arrived:
            if self._people.starts_epoch(stamp_ns):
                # The clock moved back, so the simulation restarted, usually
                # with walkie-vision, whose tracker then numbers its tracks
                # anew. The collections go with the history: a track id of
                # the new run starts a new generation, and no collection
                # mixes the stamps of two runs, which stamp order would sort
                # the wrong way round.
                self._tracks.clear()
            keys = self._collect(sightings, now)
            frame = PeopleFrame(msg, tuple(keys.get(track_id) for track_id in track_ids))
            self._people.add(stamp_ns, frame)
            self._arrived.notify_all()
            listeners = self._people_listeners
        # After the lock is released, so a listener that reads the feed does
        # not deadlock and no reader waits for a listener.
        for listener in listeners:
            try:
                listener(frame)
            except Exception:
                # Never raised into the executor, where it would end the
                # executor's thread and with it every subscription of the node.
                logger.exception("A VisionFeed people listener failed.")

    def _on_objects(self, msg: ObjectDetectionArray) -> None:
        stamp_ns = _stamp_ns(msg.header.stamp)
        with self._arrived:
            self._objects.add(stamp_ns, msg)
            self._arrived.notify_all()

    # -- Helpers, called with the lock held ----------------------------------------

    def _collect(
        self,
        sightings: list[tuple[int, Sample | None, Sample | None]],
        now: float,
    ) -> dict[int, TrackKey]:
        """Add one array's samples, and return the key of each track id in it."""
        self._drop_unseen(now)
        limit = self.config.vectors_per_track
        keys: dict[int, TrackKey] = {}
        for track_id, face, appearance in sightings:
            track = self._tracks.get(track_id)
            if track is None:
                # A first sighting, or the first since the collection was
                # dropped, when the id may name someone else: a new generation.
                self._generation += 1
                track = _Track(self._generation, now, [], [])
                self._tracks[track_id] = track
            track.last_seen = now
            self._tracks.move_to_end(track_id)
            if face is not None:
                _add_sample(track.faces, face, limit)
            if appearance is not None:
                _add_sample(track.appearances, appearance, limit)
            keys[track_id] = TrackKey(track_id, track.generation)
        while len(self._tracks) > self.config.max_tracks:
            self._tracks.popitem(last=False)
        return keys

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
        history: _StampHistory[_Item],
        topic: str,
        max_age_s: float,
        wait_s: float,
        cancel: threading.Event,
    ) -> _Item:
        _check_seconds(max_age_s, "max_age_s")
        _check_seconds(wait_s, "wait_s")
        deadline = time.monotonic() + wait_s
        with self._arrived:
            while True:
                # On every pass, before anything is returned: a stop set
                # before the call, or with the array that ended a wait, must
                # end the step rather than let it make handles from a fresh
                # array.
                _check_cancel(cancel)
                newest = history.newest()
                age_s: float | None = None
                if newest is not None:
                    # The age is checked on every pass, since the node clock
                    # moves on while the newest array stays the same.
                    stamp_ns, item = newest
                    age_s = self._age_s(stamp_ns)
                    if abs(age_s) <= max_age_s:
                        return item
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StepError(
                        "UNAVAILABLE", _stale_message(topic, age_s, max_age_s, wait_s)
                    )
                self._arrived.wait(min(_POLL_S, remaining))

    def _age_s(self, stamp_ns: int) -> float:
        # The node clock, so the age is right under sim time, where the stamps
        # come from /clock as well.
        return (self._node.get_clock().now().nanoseconds - stamp_ns) / _NS_PER_S


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


def _sample(stamp_ns: int, vector: ModelVector | None) -> Sample | None:
    return None if vector is None else Sample(stamp_ns, vector)


def _add_sample(samples: list[Sample], sample: Sample, limit: int) -> None:
    # In stamp order, as the histories: a late array's sample goes in its
    # place, and the oldest stamp goes first when the collection is full.
    bisect.insort_right(samples, sample, key=_sample_stamp)
    if len(samples) > limit:
        del samples[0]


def _sample_stamp(sample: Sample) -> int:
    return sample.stamp_ns


def _stamp_ns(stamp: TimeMsg) -> int:
    return stamp.sec * _NS_PER_S + stamp.nanosec


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise StepError("CANCELLED", "A stop or a reset interrupted the call.")


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
