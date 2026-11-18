"""people_memory/track_vectors.py:the per-track sample buffer, model_vector and face_trusted.

Ported from the experiment's tests/test_vision_feed.py, whose collection tests ran over
a VisionFeed node: here the buffer is fed directly, with explicit stamps and monotonic
times, as RosVision feeds it. Vectors are hash_vector vectors under walkie-vision's fake
model names, as oracle_perception publishes them, and that file's checks of
hash_vector come along.
"""

import math
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from walkie_agent.people_memory.store import ModelVector
from walkie_agent.people_memory.track_vectors import (
    DEFAULT_TRACK_VECTORS_CONFIG,
    Collected,
    Sighting,
    TrackVectors,
    TrackVectorsConfig,
    face_trusted,
    model_vector,
)
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, FAKE_REID_MODEL, hash_vector

DIM = 16
# The focal lengths min_face_size_rad is explained by: the ZED 2i's nominal one at
# HD1080 and HD2K, and the simulation's head camera.
ROBOT_FX = 1066.0
SIM_FX = 224.057


def face(key: str) -> ModelVector:
    vector = model_vector(FAKE_FACE_MODEL, hash_vector(f"face {key}", DIM))
    assert vector is not None
    return vector


def appearance(key: str) -> ModelVector:
    vector = model_vector(FAKE_REID_MODEL, hash_vector(f"appearance {key}", DIM))
    assert vector is not None
    return vector


def seen(
    track_id: int, key: str | None = None, *, with_face: bool = True, with_appearance: bool = True
) -> Sighting:
    """A sighting of track `track_id` with a face and an appearance from `key`."""
    key = str(track_id) if key is None else key
    return Sighting(
        track_id,
        face(key) if with_face else None,
        appearance(key) if with_appearance else None,
    )


def same(vectors, expected) -> bool:
    return len(vectors) == len(expected) and all(
        v.model == e.model and np.array_equal(v.vector, e.vector)
        for v, e in zip(vectors, expected, strict=True)
    )


# -- The fake vectors the buffer collects in simulation --------------------------------


def test_hash_vector_is_deterministic_unit_float32():
    vector = hash_vector("guest sam", 512)

    assert vector.dtype == np.float32
    assert vector.shape == (512,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)
    assert np.array_equal(vector, hash_vector("guest sam", 512))
    assert np.array_equal(vector, hash_vector(b"guest sam", 512))
    # Different inputs give nearly orthogonal vectors, which is what lets a 0.2
    # distance limit tell fake people apart.
    assert abs(float(vector @ hash_vector("guest ana", 512))) < 0.2


def test_hash_vector_matches_walkie_vision():
    # Values from walkie-vision's own hash_vector in src/walkie_vision/fakes.py. A
    # change here means vectors no longer match oracle_perception's.
    assert hash_vector("guest sam", 8)[:4].tolist() == pytest.approx(
        [-0.2260013222694397, 0.31027519702911377, 0.2287740558385849, -0.33814337849617004]
    )
    assert hash_vector(b"face 1", 8)[:4].tolist() == pytest.approx(
        [-0.24494804441928864, -0.41149866580963135, 0.19372032582759857, -0.055320821702480316]
    )


def test_fake_model_names_are_walkie_visions():
    assert (FAKE_FACE_MODEL, FAKE_REID_MODEL) == ("fake_face", "fake_reid")


# -- The config ----------------------------------------------------------------------


def test_config_defaults_are_the_experiments():
    config = TrackVectorsConfig()

    assert config == DEFAULT_TRACK_VECTORS_CONFIG
    assert (config.vectors_per_track, config.max_tracks, config.track_ttl_s) == (20, 32, 60.0)
    assert (config.min_face_det_score, config.min_face_size_rad) == (0.5, 0.0469)


@pytest.mark.parametrize("field", ["vectors_per_track", "max_tracks"])
def test_config_needs_counts_of_at_least_one(field):
    with pytest.raises(ValueError, match=field):
        TrackVectorsConfig(**{field: 0})


@pytest.mark.parametrize("field", ["track_ttl_s", "min_face_size_rad"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan])
def test_config_needs_positive_times_and_sizes(field, value):
    with pytest.raises(ValueError, match=field):
        TrackVectorsConfig(**{field: value})


# -- model_vector ----------------------------------------------------------------------


def test_model_vector_is_a_read_only_float32_copy():
    values = np.array([3.0, 4.0, 0.0], dtype=np.float64)

    vector = model_vector("m", values)
    values[0] = 100.0

    assert vector is not None
    assert vector.model == "m"
    assert vector.vector.dtype == np.float32
    assert vector.vector.tolist() == [3.0, 4.0, 0.0]
    assert not vector.vector.flags.writeable
    with pytest.raises(ValueError):
        vector.vector[0] = 1.0
    # A list, as a message's embedding may be, works too.
    from_list = model_vector("m", [1.0, 2.0])
    assert from_list is not None and from_list.vector.tolist() == [1.0, 2.0]


def test_model_vector_copies_a_float32_array_too():
    values = hash_vector("x", DIM)

    vector = model_vector("m", values)
    values[:] = 0.0

    assert vector is not None
    assert vector.vector is not values and vector.vector.any()


@pytest.mark.parametrize(
    ("model", "values"),
    [
        ("", [1.0, 0.0]),
        ("m", []),
        ("m", [0.0, 0.0]),
        ("m", [math.nan, 1.0]),
        ("m", [math.inf, 1.0]),
        # Too large for float32, so infinite once converted.
        ("m", [1e39, 1.0]),
    ],
)
def test_model_vector_refuses_what_cannot_be_matched(model, values):
    assert model_vector(model, values) is None


# -- face_trusted ------------------------------------------------------------------------


def trusted(score: float, size_x: float, size_y: float | None = None, fx=ROBOT_FX, fy=None):
    return face_trusted(score, size_x, size_x if size_y is None else size_y, fx, fx if fy is None else fy)


@pytest.mark.parametrize(
    ("score", "side_px", "expected"),
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
def test_face_trusted_needs_score_and_size(score, side_px, expected):
    assert trusted(score, side_px) is expected


def test_face_trusted_takes_the_face_as_an_angle_on_any_camera():
    # The guest at the simulation's door is 32.8 px across, far below 50 px but well
    # over the 10.5 px that 0.0469 rad is at this fx.
    assert trusted(0.9, 32.8, fx=SIM_FX)
    assert not trusted(0.9, 10.0, fx=SIM_FX)
    # The same pixels at four times the focal length span a quarter of the angle.
    assert not trusted(0.9, 32.8, fx=4 * SIM_FX)
    # A box that is not square counts by the side of the square of its area.
    assert trusted(0.9, 25.0, 100.0)
    assert not trusted(0.9, 24.0, 100.0)


def test_face_trusted_measures_width_by_fx_and_height_by_fy():
    # An image stretched to twice its width: the face is twice as wide in pixels and
    # so is fx, so it spans the same 0.06 rad as before.
    assert trusted(0.9, 60.0, fx=1000.0)
    assert trusted(0.9, 120.0, 60.0, fx=2000.0, fy=1000.0)
    # A square box in the stretched image is half as wide as an angle: 0.042 rad.
    assert not trusted(0.9, 60.0, fx=2000.0, fy=1000.0)


@pytest.mark.parametrize(("scale", "expected"), [(1.0 - 1e-6, False), (1.0, True), (1.0 + 1e-6, True)])
def test_a_face_on_the_size_limit_is_trusted(scale, expected):
    # 42.7 x 58.537... px at fx 1066 spans exactly 0.0469 rad.
    assert trusted(0.9, 42.7, 58.537237029508184 * scale) is expected


def test_face_trusted_names_nobody_by_a_camera_info_never_filled_in():
    assert not trusted(0.9, 500.0, fx=0.0)
    # A focal length on one axis only is no better.
    assert not trusted(0.9, 500.0, fx=1000.0, fy=0.0)
    assert not trusted(0.9, 500.0, fx=0.0, fy=1000.0)


@pytest.mark.parametrize(
    ("score", "size_x", "size_y", "fx"),
    [
        (math.nan, 60.0, 60.0, ROBOT_FX),
        (0.9, math.nan, 60.0, ROBOT_FX),
        (0.9, math.inf, 60.0, ROBOT_FX),
        (0.9, -60.0, -60.0, ROBOT_FX),
        (0.9, 60.0, 60.0, math.nan),
    ],
)
def test_face_trusted_refuses_a_nan_or_a_box_that_is_not_one(score, size_x, size_y, fx):
    assert not face_trusted(score, size_x, size_y, fx, ROBOT_FX)


def test_face_trusted_follows_the_config():
    strict = TrackVectorsConfig(min_face_det_score=0.95, min_face_size_rad=0.1)

    assert trusted(0.9, 60.0)
    assert not face_trusted(0.9, 60.0, 60.0, ROBOT_FX, ROBOT_FX, strict)
    assert face_trusted(0.96, 107.0, 107.0, ROBOT_FX, ROBOT_FX, strict)


# -- Collected -------------------------------------------------------------------------------


def test_collected_is_false_only_when_it_holds_nothing():
    assert not Collected()
    assert not Collected((), ())
    assert Collected(faces=(face("a"),))
    assert Collected(appearances=(appearance("a"),))


# -- TrackVectors --------------------------------------------------------------------------------


def test_keeps_the_newest_20_samples_per_track_oldest_first():
    buffer = TrackVectors()
    for frame in range(25):
        buffer.add(frame, [seen(7, f"7 frame {frame}")], now=0.0)

    collected = buffer.collected(7, now=0.0)

    assert same(collected.faces, [face(f"7 frame {frame}") for frame in range(5, 25)])
    assert same(collected.appearances, [appearance(f"7 frame {frame}") for frame in range(5, 25)])
    assert {v.model for v in collected.faces} == {FAKE_FACE_MODEL}
    assert {v.model for v in collected.appearances} == {FAKE_REID_MODEL}


def test_a_late_sample_goes_in_its_stamp_order():
    buffer = TrackVectors()
    for stamp in (10, 30, 20):
        buffer.add(stamp, [seen(5, f"at {stamp}")], now=0.0)

    collected = buffer.collected(5, now=0.0)

    expected = [f"at {stamp}" for stamp in (10, 20, 30)]
    assert same(collected.faces, [face(key) for key in expected])
    assert same(collected.appearances, [appearance(key) for key in expected])


def test_a_late_sample_older_than_the_newest_is_not_kept_when_the_track_is_full():
    buffer = TrackVectors(TrackVectorsConfig(vectors_per_track=3))
    for stamp in (10, 20, 30):
        buffer.add(stamp, [seen(5, f"at {stamp}")], now=0.0)

    buffer.add(5, [seen(5, "at 5")], now=0.0)
    buffer.add(25, [seen(5, "at 25")], now=0.0)

    assert same(buffer.collected(5, now=0.0).faces, [face(f"at {s}") for s in (20, 25, 30)])


def test_the_same_array_twice_counts_once():
    buffer = TrackVectors()
    array = [seen(1, "sam"), seen(2, "ana")]

    buffer.add(100, array, now=0.0)
    buffer.add(100, array, now=1.0)
    # Another sample at a stamp the track holds is skipped too, whatever it holds.
    buffer.add(100, [seen(1, "bob")], now=2.0)
    buffer.add(101, [seen(1, "sam 2")], now=3.0)

    sam = buffer.collected(1, now=3.0)
    assert same(sam.faces, [face("sam"), face("sam 2")])
    assert same(sam.appearances, [appearance("sam"), appearance("sam 2")])
    assert same(buffer.collected(2, now=3.0).faces, [face("ana")])


def test_a_stamp_held_for_faces_only_still_takes_its_appearance():
    buffer = TrackVectors()

    buffer.add(100, [seen(1, "sam", with_appearance=False)], now=0.0)
    buffer.add(100, [seen(1, "sam", with_face=False)], now=0.0)

    collected = buffer.collected(1, now=0.0)
    assert same(collected.faces, [face("sam")])
    assert same(collected.appearances, [appearance("sam")])


def test_keeps_32_tracks_and_skips_track_0():
    buffer = TrackVectors()
    # 32 tracks and an untracked person: track 0 takes no place.
    buffer.add(1, [seen(n) for n in range(1, 33)] + [seen(0)], now=0.0)

    assert buffer.tracks(now=0.0) == list(range(1, 33))
    assert not buffer.collected(0, now=0.0)

    # Track 2 is seen again, so track 1 is now the one seen longest ago.
    buffer.add(2, [seen(2, "2 again")], now=1.0)
    buffer.add(3, [seen(33)], now=2.0)

    assert not buffer.collected(1, now=2.0)
    assert len(buffer.collected(2, now=2.0).faces) == 2
    assert all(len(buffer.collected(n, now=2.0).faces) == 1 for n in range(3, 34))
    assert buffer.tracks(now=2.0) == [*range(3, 33), 2, 33]
    # Track 1 seen again after it was dropped starts a new collection.
    buffer.add(4, [seen(1, "1 back")], now=3.0)
    assert same(buffer.collected(1, now=3.0).faces, [face("1 back")])


def test_drops_a_track_unseen_for_60_s():
    buffer = TrackVectors()
    buffer.add(1, [seen(1), seen(2)], now=100.0)
    buffer.add(2, [seen(2, "2 again")], now=130.0)

    assert len(buffer.collected(1, now=159.9).faces) == 1
    assert len(buffer.collected(2, now=159.9).faces) == 2
    assert not buffer.collected(1, now=160.1)
    assert buffer.tracks(now=160.1) == [2]
    assert len(buffer.collected(2, now=160.1).faces) == 2

    # A new array drops the unseen tracks too, not only a read.
    buffer.add(3, [seen(3)], now=300.0)
    assert buffer.tracks(now=300.0) == [3]
    # A track seen again after it expired starts a new collection.
    buffer.add(4, [seen(2, "2 back")], now=301.0)
    assert same(buffer.collected(2, now=301.0).faces, [face("2 back")])


def test_the_ttl_follows_the_config():
    buffer = TrackVectors(TrackVectorsConfig(track_ttl_s=0.5))
    buffer.add(1, [seen(1)], now=10.0)

    assert buffer.collected(1, now=10.4)
    assert not buffer.collected(1, now=10.6)


def test_an_array_older_than_the_newest_still_counts_as_a_sighting():
    buffer = TrackVectors()
    buffer.add(1000, [seen(1, "new")], now=0.0)

    # A late array, stamped before the newest, arrives 50 s later.
    buffer.add(500, [seen(1, "late")], now=50.0)

    assert same(buffer.collected(1, now=100.0).faces, [face("late"), face("new")])


def test_a_track_seen_without_vectors_is_kept_but_has_no_samples():
    buffer = TrackVectors()
    buffer.add(1, [seen(1, with_face=False, with_appearance=False)], now=0.0)
    buffer.add(1, [seen(2, with_face=False)], now=0.0)

    assert buffer.tracks(now=0.0) == [1, 2]
    assert not buffer.collected(1, now=0.0)
    two = buffer.collected(2, now=0.0)
    assert two.faces == () and same(two.appearances, [appearance("2")])


def test_unknown_tracks_have_nothing():
    buffer = TrackVectors()
    buffer.add(1, [seen(1)], now=0.0)

    assert buffer.collected(2, now=0.0) == Collected()
    assert buffer.collected(0, now=0.0) == Collected()


def test_bad_vectors_are_never_collected():
    buffer = TrackVectors()
    zero = ModelVector(FAKE_FACE_MODEL, np.zeros(DIM, dtype=np.float32))
    not_finite = ModelVector(FAKE_REID_MODEL, np.full(DIM, np.nan, dtype=np.float32))
    no_model = ModelVector("", hash_vector("x", DIM))
    empty = ModelVector(FAKE_FACE_MODEL, np.zeros(0, dtype=np.float32))
    # The same faults on read-only float32 copies, as model_vector makes them.
    frozen_zero = ModelVector(FAKE_FACE_MODEL, np.zeros(DIM, dtype=np.float32))
    frozen_zero.vector.flags.writeable = False

    buffer.add(
        1,
        [
            Sighting(1, zero, not_finite),
            Sighting(2, no_model, empty),
            Sighting(3, frozen_zero, appearance("3")),
        ],
        now=0.0,
    )

    assert not buffer.collected(1, now=0.0)
    assert not buffer.collected(2, now=0.0)
    three = buffer.collected(3, now=0.0)
    assert three.faces == () and same(three.appearances, [appearance("3")])


def test_returned_arrays_are_read_only_and_never_change():
    buffer = TrackVectors()
    # A caller's own array, writable and float64: the buffer keeps a copy.
    mine = hash_vector("sam", DIM).astype(np.float64)
    buffer.add(1, [Sighting(1, ModelVector(FAKE_FACE_MODEL, mine), appearance("sam"))], now=0.0)
    mine[:] = 1.0

    before = buffer.collected(1, now=0.0)
    for stamp in range(2, 27):
        buffer.add(stamp, [seen(1, f"later {stamp}")], now=0.0)
    buffer.clear()

    [stored_face] = before.faces
    assert stored_face.vector.dtype == np.float32
    assert np.array_equal(stored_face.vector, hash_vector("sam", DIM))
    for vector in (*before.faces, *before.appearances):
        assert not vector.vector.flags.writeable
        with pytest.raises(ValueError):
            vector.vector[0] = 0.0


def test_a_read_only_model_vector_copy_is_kept_as_it_is():
    buffer = TrackVectors()
    sam = face("sam")

    buffer.add(1, [Sighting(1, sam, None)], now=0.0)

    [kept] = buffer.collected(1, now=0.0).faces
    assert kept.vector is sam.vector


def test_clear_forgets_every_track():
    buffer = TrackVectors()
    buffer.add(1, [seen(1), seen(2)], now=0.0)

    buffer.clear()

    assert buffer.tracks(now=0.0) == []
    assert not buffer.collected(1, now=0.0)
    buffer.add(2, [seen(1, "after")], now=0.0)
    assert same(buffer.collected(1, now=0.0).faces, [face("after")])


def test_adds_and_reads_on_several_threads_stay_consistent():
    buffer = TrackVectors(TrackVectorsConfig(vectors_per_track=5, max_tracks=4))
    start = threading.Barrier(3)

    def writer(offset: int) -> None:
        start.wait()
        for stamp in range(300):
            people = [seen(n, f"{n} {stamp}") for n in range(1, 7)]
            buffer.add(stamp * 2 + offset, people, now=0.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        writers = [pool.submit(writer, offset) for offset in (0, 1)]
        start.wait()
        for _ in range(300):
            assert len(buffer.tracks(now=0.0)) <= 4
            for track_id in range(1, 7):
                collected = buffer.collected(track_id, now=0.0)
                assert len(collected.faces) <= 5 and len(collected.appearances) <= 5
        # Raises what a writer raised.
        for done in writers:
            done.result()

    assert buffer.tracks(now=0.0) == [3, 4, 5, 6]
    assert all(len(buffer.collected(n, now=0.0).faces) == 5 for n in (3, 4, 5, 6))
