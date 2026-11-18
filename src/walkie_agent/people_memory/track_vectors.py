"""The face and appearance samples of each track in view, for remembering and recognizing.

RosVision (ros/vision.py) feeds a TrackVectors from its one people subscription, and
TrackMemory (track_memory.py) reads a track's samples back when the model remembers or
recognizes the person on it (ADR-0070). So a guest is stored from many images rather
than one: every face they showed while the robot talked to them. No ROS.

The rules are the experiment's VisionFeed collection (experiment/hri-agent's
perception/vision_feed.py), which walkie-agent-v2's numbers tuned on the robot:

- 20 face and 20 appearance samples per track, the newest by stamp, kept oldest
  first. A late array's sample goes in its place, and a sample older than the 20
  newest is not kept. A sample whose stamp its track holds already is skipped, so
  the same array twice counts once.
- At most 32 tracks; a new track past that drops the one seen longest ago. A track
  unseen for 60 s on the caller's monotonic clock is dropped with its samples, so a
  guest who left is not stored from old images.
- Track 0, an untracked person, is never collected: its id names nobody from one
  image to the next.
- A face counts only when face_trusted passes it, and a vector only when model_vector
  takes it: one with no model name, or an empty, zero or non-finite vector, is never
  collected.

The experiment's generation keys are left out: this tree's ids are person_<track_id>,
which carry no generation, so a track id that walkie-vision reuses within the TTL
shares one collection. RosVision clears the buffer when the clock jumps back, as at a
simulation restart, and on close().

Matching and outlier rejection are not here but in recognition.py and identity.py,
applied when a person is remembered or recognized.
"""

import bisect
import math
import threading
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from walkie_agent.people_memory.store import ModelVector


@dataclass(frozen=True)
class TrackVectorsConfig:
    """What TrackVectors keeps and which faces it trusts.

    The values are the experiment's VisionFeedConfig defaults, from walkie-agent-v2,
    tuned on the robot. Raises ValueError for a count below 1, or a time or a face
    size that is not positive.
    """

    # Face samples, and separately appearance samples, kept per track: the newest
    # ones by stamp.
    vectors_per_track: int = 20
    # Tracks kept at once. When a new track would exceed it, the track seen longest
    # ago is dropped.
    max_tracks: int = 32
    # A track unseen for this many seconds is dropped with its samples.
    track_ttl_s: float = 60.0
    # v2's HRI_RECOG_MIN_DET_SCORE: below this face_det_score the embedding is too
    # noisy to name someone by. Tuned on the robot.
    min_face_det_score: float = 0.5
    # The smallest face to name someone by, as the angle its box spans:
    # sqrt((size_x / fx) * (size_y / fy)) radians. An angle holds for any image size,
    # where a pixel area would not: the simulation's head camera is 640 x 360 with fx
    # 224 px. v2's HRI_RECOG_MIN_FACE_AREA_PX was 2500 px2, about 50 x 50 px, tuned
    # on the robot's ZED 2i at HD1080, whose nominal fx is 1066 px. 50 / 1066 is
    # 0.0469 rad, 2.7 degrees, and 10.5 px in the simulation. The robot's own
    # calibration was not read, so check it there: 50 px over its k[0].
    min_face_size_rad: float = 0.0469

    def __post_init__(self) -> None:
        for name in ("vectors_per_track", "max_tracks"):
            if getattr(self, name) < 1:
                raise ValueError(f"TrackVectorsConfig.{name} must be at least 1.")
        for name in ("track_ttl_s", "min_face_size_rad"):
            # Written so that NaN fails too.
            if not getattr(self, name) > 0:
                raise ValueError(f"TrackVectorsConfig.{name} must be positive.")


DEFAULT_TRACK_VECTORS_CONFIG = TrackVectorsConfig()


def model_vector(model: str, values: Sequence[float]) -> ModelVector | None:
    """An embedding as a read-only float32 copy with its model name, or None.

    A vector with no model name, or one that is empty, zero or not finite, is None:
    it cannot be matched, and one bad vector would spoil the mean that outlier
    rejection compares against. Values too large for float32 become infinite and
    count as not finite.
    """
    if not isinstance(model, str) or not model:
        return None
    # flatten() always copies, so nothing else holds the array's memory, and a
    # caller that changes `values` later cannot change the sample.
    with np.errstate(over="ignore"):
        vector = np.asarray(values, dtype=np.float32).flatten()
    if vector.size == 0 or not np.isfinite(vector).all() or not vector.any():
        return None
    vector.flags.writeable = False
    return ModelVector(model, vector)


def face_trusted(
    det_score: float,
    size_x: float,
    size_y: float,
    fx: float,
    fy: float,
    config: TrackVectorsConfig = DEFAULT_TRACK_VECTORS_CONFIG,
) -> bool:
    """True when a face is good enough to name someone by (v2's _face_trusted).

    The face needs the detector's confidence `det_score` of at least
    `min_face_det_score`, and an angular size of at least `min_face_size_rad`:
    sqrt((size_x / fx) * (size_y / fy)), with the face box's size in pixels and the
    focal lengths of the image's camera_info (fx = k[0], fy = k[4]). A small or
    doubtful face gives a noisy embedding, and storing it would pull the guest's
    stored faces toward someone else's. A focal length that is not positive means a
    camera_info that was never filled in: the face's size is then unknown, and an
    unknown face names nobody. A negative size, an infinite one or a NaN anywhere
    fails too.
    """
    if not det_score >= config.min_face_det_score:
        return False
    if not (fx > 0 and fy > 0 and size_x >= 0 and size_y >= 0):
        return False
    # The width over fx and the height over fy, so an image scaled more in one
    # direction than the other measures the same face.
    area_rad2 = (size_x / fx) * (size_y / fy)
    if not math.isfinite(area_rad2):
        return False
    return math.sqrt(max(area_rad2, 0.0)) >= config.min_face_size_rad


class Sighting(NamedTuple):
    """One tracked person of one people array, with the vectors to collect.

    `face` is None when the face was not trusted or not seen, and `appearance` when
    no appearance vector was computed. Build them with model_vector.
    """

    track_id: int
    face: ModelVector | None
    appearance: ModelVector | None


@dataclass(frozen=True)
class Collected:
    """The samples of one track, oldest stamp first. False when it holds none.

    The arrays are the buffer's own and read-only. Like every vector, they stay in
    the backend: no record that the model reads holds one (ADR-0070).
    """

    faces: tuple[ModelVector, ...] = ()
    appearances: tuple[ModelVector, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.faces or self.appearances)


_NOTHING = Collected()


class _Sample(NamedTuple):
    stamp_ns: int
    vector: ModelVector


@dataclass
class _Track:
    # The caller's monotonic time of the newest array that showed the track.
    last_seen: float
    # By stamp, oldest first, at most TrackVectorsConfig.vectors_per_track each.
    faces: list[_Sample] = field(default_factory=list)
    appearances: list[_Sample] = field(default_factory=list)


class TrackVectors:
    """The newest face and appearance samples of every track seen in the last TTL.

    Thread-safe: RosVision adds on the executor's thread while TrackMemory reads on
    the agent's worker. Every method takes `now`, the caller's time.monotonic(), so
    the TTL does not move with a simulated clock, and tests can set it. No method
    calls back into the caller, so RosVision may call add() while it holds its own
    lock.
    """

    def __init__(self, config: TrackVectorsConfig = DEFAULT_TRACK_VECTORS_CONFIG) -> None:
        self.config = config
        self._lock = threading.Lock()
        # By track id, the track seen longest ago first.
        self._tracks: OrderedDict[int, _Track] = OrderedDict()

    def add(self, stamp_ns: int, sightings: Iterable[Sighting], now: float) -> None:
        """Collect one people array's sightings, stamped with its header's `stamp_ns`.

        Each tracked person counts as seen at `now`, also for an array older than
        the newest. A sighting on track 0, and a vector that model_vector would
        refuse, adds nothing; a sample whose stamp its track holds already is
        skipped.
        """
        # Checked and copied before the lock is taken, so a reader never waits for it.
        kept = [
            (int(s.track_id), _collectable(s.face), _collectable(s.appearance))
            for s in sightings
            if s.track_id != 0
        ]
        limit = self.config.vectors_per_track
        with self._lock:
            self._drop_unseen(now)
            for track_id, face, appearance in kept:
                track = self._tracks.get(track_id)
                if track is None:
                    track = _Track(last_seen=now)
                    self._tracks[track_id] = track
                track.last_seen = now
                self._tracks.move_to_end(track_id)
                if face is not None:
                    _add_sample(track.faces, _Sample(stamp_ns, face), limit)
                if appearance is not None:
                    _add_sample(track.appearances, _Sample(stamp_ns, appearance), limit)
            while len(self._tracks) > self.config.max_tracks:
                self._tracks.popitem(last=False)

    def collected(self, track_id: int, now: float) -> Collected:
        """The samples of `track_id`, oldest stamp first; empty for an unknown or expired track."""
        with self._lock:
            self._drop_unseen(now)
            track = self._tracks.get(track_id)
            if track is None:
                return _NOTHING
            return Collected(
                faces=tuple(sample.vector for sample in track.faces),
                appearances=tuple(sample.vector for sample in track.appearances),
            )

    def tracks(self, now: float) -> list[int]:
        """The track ids kept at `now`, the one seen longest ago first."""
        with self._lock:
            self._drop_unseen(now)
            return list(self._tracks)

    def clear(self) -> None:
        """Forget every track and its samples."""
        with self._lock:
            self._tracks.clear()

    def _drop_unseen(self, now: float) -> None:
        # Every track is checked rather than only those at the front, so a caller's
        # clock that once went back cannot keep an expired track. There are at most
        # max_tracks of them.
        cutoff = now - self.config.track_ttl_s
        for track_id in [tid for tid, track in self._tracks.items() if track.last_seen < cutoff]:
            del self._tracks[track_id]


def _collectable(vector: ModelVector | None) -> ModelVector | None:
    """`vector` as the buffer keeps it: checked, float32 and read-only; None if refused."""
    if vector is None:
        return None
    array = vector.vector
    if (
        isinstance(array, np.ndarray)
        and array.dtype == np.float32
        and array.ndim == 1
        and array.base is None
        and not array.flags.writeable
    ):
        # A copy that model_vector made: nothing else can change it, so it is
        # kept as it is once it is checked.
        if vector.model and array.size and np.isfinite(array).all() and array.any():
            return vector
        return None
    return model_vector(vector.model, array)


def _add_sample(samples: list[_Sample], sample: _Sample, limit: int) -> None:
    # In stamp order: a late array's sample goes in its place, and the oldest stamp
    # goes first when the collection is full, even when it is the one that just came.
    index = bisect.bisect_left(samples, sample.stamp_ns, key=_stamp)
    if index < len(samples) and samples[index].stamp_ns == sample.stamp_ns:
        return
    samples.insert(index, sample)
    if len(samples) > limit:
        del samples[0]


def _stamp(sample: _Sample) -> int:
    return sample.stamp_ns
