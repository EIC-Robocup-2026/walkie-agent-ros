"""The doorbell detector's config, audio path, rules and model (walkie_agent.doorbell).

The rules, the audio path, the config and the eval's bookkeeping are plain
logic, tested on scripted scores. The tests marked needs_model run YAMNet
itself, from models/: they check the ONNX model against the TF reference
scores in tests/data/audio, and that the clips there ring or stay quiet at
every rate and onset, except where known_failures.toml says. They skip when
models/ has no model. `pixi run fetch-doorbell-model` downloads it from the
release doorbell-model-v1, which docs/setup/07-voice.md says how to publish.
"""

import csv
import json
import math
import os
import subprocess
import sys
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pydantic
import pytest
import scipy.signal

from conftest import PROJECT_ROOT
from walkie_agent import doorbell
from walkie_agent.config_tables import TuningError
from walkie_agent.doorbell import (
    BLOCK_SAMPLES,
    CLASS_COUNT,
    CLASSES,
    EVAL_RATES,
    HOP_SAMPLES,
    OFF,
    ONSETS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    DetectorConfig,
    DoorbellEvent,
    ModelError,
    RingRules,
    Window,
)

CONFIG = PROJECT_ROOT / "config.toml"
SRC = PROJECT_ROOT / "src"
AUDIO = PROJECT_ROOT / "tests" / "data" / "audio"
REFERENCE = AUDIO / "yamnet_reference.npz"
RING_CLIPS = sorted(path for d in doorbell.RING_DIRS for path in (AUDIO / d).glob("*.wav"))
QUIET_CLIPS = sorted((AUDIO / doorbell.QUIET_DIR).glob("*.wav"))

MODEL_FILES = (PROJECT_ROOT / doorbell.MODEL_FILE, PROJECT_ROOT / doorbell.CLASS_MAP_FILE)
needs_model = pytest.mark.skipif(
    not all(path.is_file() for path in MODEL_FILES),
    reason="models/ holds no doorbell model. `pixi run fetch-doorbell-model` downloads it from "
    "the release doorbell-model-v1; if that release is not published, docs/setup/07-voice.md "
    "says how a maintainer publishes it",
)

# The seconds between two windows' starts.
HOP_S = HOP_SAMPLES / SAMPLE_RATE
START = datetime(2026, 10, 27, 16, 0, tzinfo=timezone(timedelta(hours=7)))
# A class map whose names are YAMNet's at the indices the rules read.
NAMES = tuple(
    next((name for name, index in CLASSES.items() if index == i), f"Class {i}")
    for i in range(CLASS_COUNT)
)


def scores(classes: dict[str, float]) -> np.ndarray:
    row = np.zeros(CLASS_COUNT, np.float32)
    for name, value in classes.items():
        row[CLASSES[name]] = value
    return row


def window(i: int, classes: dict[str, float] | None = None, muted: bool = False) -> Window:
    """Window i of a stream, scored with `classes`, or quiet when None."""
    return Window(
        scores=None if classes is None else scores(classes),
        level_dbfs=-30.0,
        muted=muted,
        start_s=i * HOP_S,
        start_wall=START + timedelta(seconds=i * HOP_S),
    )


def rings(
    stream: list[dict[str, float] | None],
    config: DetectorConfig | None = None,
    muted: frozenset[int] = frozenset(),
) -> list[tuple[int, DoorbellEvent]]:
    """(window index, event) for each ring that `stream`, one entry per window, makes."""
    rules = RingRules(config or DetectorConfig(), NAMES, "test_model")
    found = []
    for i, classes in enumerate(stream):
        if event := rules.step(window(i, classes, muted=i in muted), armed=True):
            found.append((i, event))
    return found


def labels(found: list[tuple[int, DoorbellEvent]]) -> list[tuple[int, str]]:
    return [(i, event.label) for i, event in found]


class ScriptedClassifier:
    """Scores every window it is given with the same classes, and counts them."""

    model = "scripted"
    class_names = NAMES

    def __init__(self, classes: dict[str, float]) -> None:
        self.row = scores(classes)
        self.classified = 0

    def classify(self, windows: np.ndarray) -> np.ndarray:
        self.classified += len(windows)
        return np.tile(self.row, (len(windows), 1))


def tone(seconds: float, dbfs: float = -20.0) -> np.ndarray:
    """A 1 kHz sine at the RMS level `dbfs`, as 16 kHz float32."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    amplitude = math.sqrt(2) * 10 ** (dbfs / 20)
    return (amplitude * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, rate: int = SAMPLE_RATE, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.asarray(samples, "<i2").tobytes())
    return path


def write_class_map(path: Path, names: tuple[str, ...] = NAMES) -> Path:
    # As YAMNet's: a header, then index, AudioSet mid and display name.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "mid", "display_name"])
        writer.writerows([i, f"/m/{i}", name] for i, name in enumerate(names))
    return path


# -- Config -------------------------------------------------------------------------------------


def test_without_the_table_the_detector_keeps_its_defaults():
    errors: list[str] = []

    assert doorbell.parse_detector_config({}, errors) == DetectorConfig()
    assert errors == []


def test_the_defaults_are_the_designs_starting_points():
    # The doorbell design's values (ADR-0054), except chime_pair: the
    # design's 0.45 misses the eval's door chime through RAI's resample.
    config = DetectorConfig()
    assert config.enabled is True
    assert (config.bell_single, config.bell_pair, config.chime_pair) == (0.25, 0.15, 0.25)
    assert (config.knock_pair, config.context_veto, config.sustain) == (0.35, 0.3, 0.1)
    assert (config.min_level_dbfs, config.refractory_s) == (-55.0, 3.0)
    assert (config.post_ring_ignore_s, config.max_ignore_s) == (2.5, 10.0)


def test_the_repository_config_loads():
    assert isinstance(doorbell.load_detector_config(CONFIG), DetectorConfig)


def test_load_detector_config_reads_only_its_own_table(tmp_path):
    # The agent checks the rest of [walkie]: a typo there is not the
    # detector's to refuse.
    path = tmp_path / "config.toml"
    path.write_text(
        "[walkie.gaze]\ntf_wait = 0.4\n\n"
        "[walkie.doorbell]\ntimeout_s = 15.0\n\n"
        "[walkie.doorbell.detector]\nenabled = false\nchime_pair = 0.4\nrefractory_s = 4\n"
    )

    assert doorbell.load_detector_config(path) == DetectorConfig(
        enabled=False, chime_pair=0.4, refractory_s=4.0
    )


def test_load_detector_config_lists_every_problem(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[walkie.doorbell.detector]\nchime_pairs = 0.4\nenabled = 1\nbell_single = 1.2\n"
    )

    with pytest.raises(TuningError) as caught:
        doorbell.load_detector_config(path)

    assert caught.value.source == str(path)
    assert caught.value.errors == (
        "[walkie.doorbell.detector]: unknown key 'chime_pairs'; it takes "
        + ", ".join(doorbell.DETECTOR_KEYS)
        + ".",
        "walkie.doorbell.detector.enabled = 1 is not true or false.",
        "[walkie.doorbell.detector]: DetectorConfig.bell_single must lie in (0, 1.1], not 1.2.",
    )


def test_a_detector_table_written_as_a_value_is_refused(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[walkie.doorbell]\ndetector = 0.4\n")

    with pytest.raises(TuningError) as caught:
        doorbell.load_detector_config(path)

    assert caught.value.errors == (
        "walkie.doorbell.detector = 0.4 is not a table; write it as [walkie.doorbell.detector].",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"chime_pair": 0.0}, "DetectorConfig.chime_pair must lie in (0, 1.1], not 0.0."),
        ({"sustain": 1.2}, "DetectorConfig.sustain must lie in (0, 1.1], not 1.2."),
        ({"min_level_dbfs": 0.0}, "DetectorConfig.min_level_dbfs must be below 0, not 0.0."),
        (
            {"refractory_s": 0.0},
            "DetectorConfig.refractory_s must be a positive number, not 0.0.",
        ),
        (
            {"post_ring_ignore_s": 12.0},
            "DetectorConfig.post_ring_ignore_s must not be above max_ignore_s.",
        ),
    ],
)
def test_the_detector_config_refuses_values_out_of_range(changes, message):
    with pytest.raises(ValueError) as caught:
        DetectorConfig(**changes)

    assert str(caught.value) == message


def test_a_threshold_of_off_is_kept():
    assert DetectorConfig(knock_pair=OFF).knock_pair == 1.1


# -- Audio --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "num"),
    [
        (3840, 1280),  # a 48 kHz block to 16 kHz
        (3528, 1280),  # a 44.1 kHz block to 16 kHz
        (3527, 1279),  # odd lengths
        (1280, 3840),  # a clip played at 48 kHz
        (40000, 110250),  # a clip played at 44.1 kHz
        (1280, 1280),
    ],
)
def test_resample_is_scipys(count, num):
    x = np.random.default_rng(count + num).normal(0, 3000, count)

    np.testing.assert_allclose(
        doorbell.resample(x, num), scipy.signal.resample(x, num), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("rate", [48000, 44100])
def test_the_eval_path_is_rais_per_block_resample(rate):
    # RAI's callback (rai_s2s sound_device/api.py) resamples each int16
    # block of the device's stream to 16 kHz with scipy, and voice-asr gets
    # float64 at int16 scale.
    clip = np.random.default_rng(rate).integers(-8000, 8000, 12000).astype(np.int16)
    played = scipy.signal.resample(clip.astype(np.float64), round(len(clip) * rate / 16000))
    pcm = np.clip(np.round(played), -32768, 32767).astype(np.int16)
    size = int(1280 * rate / 16000)
    expected = np.concatenate(
        [
            scipy.signal.resample(pcm[start : start + size], int(size / rate * 16000)) / 32768
            for start in range(0, len(pcm) - size + 1, size)
        ]
    )

    got = doorbell.through_rai(clip, rate)

    assert got.dtype == np.float32
    assert len(got) % BLOCK_SAMPLES == 0
    np.testing.assert_allclose(got, np.clip(expected, -1, 1), rtol=0, atol=1e-6)


def test_a_16_khz_device_delivers_the_clip_itself():
    clip = np.array([0, 16384, -32768, 32767], np.int16)

    np.testing.assert_array_equal(
        doorbell.through_rai(clip, 16000), np.array([0, 0.5, -1, 32767 / 32768], np.float32)
    )


def test_int16_and_float64_blocks_read_the_same_and_an_overshoot_is_clipped():
    block = np.array([-32768, -16384, 0, 16384, 32767], np.int16)

    np.testing.assert_array_equal(doorbell.to_float(block), doorbell.to_float(block / 1.0))
    np.testing.assert_array_equal(
        doorbell.to_float(np.array([40000.0, -40000.0])), np.array([1, -1], np.float32)
    )


def test_windows_start_every_hop_and_the_last_is_padded():
    waveform = np.arange(40000, dtype=np.float32)

    windows = doorbell.windows_of(waveform)

    # YAMNet's padding: 15600 + 4 * 7680 = 46320 samples, 5 windows.
    assert windows.shape == (5, WINDOW_SAMPLES)
    assert [w[0] for w in windows] == [i * HOP_SAMPLES for i in range(5)]
    assert windows[-1][-1] == 0
    # A clip shorter than one window still makes one.
    assert doorbell.windows_of(np.ones(100, np.float32)).shape == (1, WINDOW_SAMPLES)


def test_log_mel_patches_are_yamnets_shape():
    patches = doorbell.log_mel_patches(np.zeros((2, WINDOW_SAMPLES), np.float32))

    assert patches.shape == (2, 96, 64)
    assert patches.dtype == np.float32
    # Silence is log(0 + 0.001) in every band.
    np.testing.assert_allclose(patches, math.log(0.001), rtol=1e-6)


def test_the_level_of_silence_is_minus_infinity():
    assert doorbell.level_dbfs(np.zeros(100)) == -math.inf
    assert doorbell.level_dbfs(tone(1.0, dbfs=-20.0)) == pytest.approx(-20.0, abs=0.01)


# -- The muted rule -----------------------------------------------------------------------------


def test_a_window_holds_the_blocks_its_samples_fall_in():
    # Window 1 starts at block 6 and ends 15 samples into block 18.
    assert doorbell.blocks_of(HOP_SAMPLES) == range(6, 19)
    assert doorbell.blocks_of(0) == range(0, 13)


def test_a_window_whose_earliest_block_was_spoken_over_is_muted():
    speaking = [False] * 40
    speaking[6] = True

    # Window 1 starts at block 6, and window 2 at block 12.
    assert doorbell.window_muted(speaking, HOP_SAMPLES)
    assert not doorbell.window_muted(speaking, 2 * HOP_SAMPLES)


def test_a_window_whose_last_part_block_was_spoken_over_is_muted():
    speaking = [False] * 40
    speaking[18] = True

    assert doorbell.window_muted(speaking, HOP_SAMPLES)
    assert not doorbell.window_muted(speaking, 0)


def test_a_ring_with_a_muted_window_is_muted():
    found = rings([{"Chime": 0.5}, {"Chime": 0.5}], muted=frozenset({0}))

    assert labels(found) == [(1, "chime")]
    assert found[0][1].muted


def test_detect_mutes_the_windows_that_hold_a_spoken_block():
    classifier = ScriptedClassifier({"Doorbell": 0.9})
    rules = RingRules(DetectorConfig(), NAMES, classifier.model)
    speaking = [False] * 60
    speaking[0] = True

    found = doorbell.detect(classifier, rules, tone(5.0), speaking=speaking, start_wall=START)

    # A bell in every window rings at 0 s and, after the refractory time,
    # at 3.36 s, when block 0 is long gone.
    assert [(i, event.muted) for i, event in found.rings] == [(0, True), (7, False)]
    assert found.rings[1][1].onset_wall == "2026-10-27T16:00:03.360+07:00"


# -- The rules ----------------------------------------------------------------------------------


def test_a_bell_rings_on_one_window():
    found = rings([{}, {"Doorbell": 0.3}])

    assert labels(found) == [(1, "bell")]


def test_a_weaker_bell_rings_on_two_of_three_windows():
    assert labels(rings([{"Ding-dong": 0.2}, {}, {"Ding-dong": 0.2}])) == [(2, "bell")]
    assert rings([{"Ding-dong": 0.2}, {}, {}]) == []


@pytest.mark.parametrize("name", ["Ding", "Chime", "Bell", "Buzzer"])
def test_a_chime_rings_on_two_of_three_windows(name):
    assert labels(rings([{name: 0.3}, {name: 0.3}])) == [(1, "chime")]


def test_a_knock_rings_on_two_of_three_windows():
    assert labels(rings([{"Knock": 0.4}, {}, {"Knock": 0.4}])) == [(2, "knock")]


@pytest.mark.parametrize("classes", [{"Chime": 0.9}, {"Knock": 0.9}])
def test_one_window_of_a_chime_or_a_knock_does_not_ring(classes):
    assert rings([classes, {}, {}, classes]) == []


@pytest.mark.parametrize("name", doorbell.NEVER_TRIGGER)
def test_the_classes_no_rule_reads_never_ring(name):
    assert rings([{name: 1.0}] * 8) == []


def test_bell_wins_over_chime_and_chime_over_knock():
    every = {"Doorbell": 0.9, "Chime": 0.9, "Knock": 0.9}
    assert labels(rings([every, every])) == [(0, "bell")]
    both = {"Chime": 0.9, "Knock": 0.9}
    assert labels(rings([both, both])) == [(1, "chime")]


def test_a_rule_at_off_never_rings():
    config = DetectorConfig(bell_single=OFF, bell_pair=OFF, knock_pair=OFF)

    assert rings([{"Doorbell": 1.0, "Knock": 1.0}] * 8, config) == []


def test_the_refractory_time_keeps_one_event_per_ring():
    found = rings([{"Doorbell": 0.9}] * 10)

    # 7 windows are 3.36 s, the first start at least refractory_s later.
    assert labels(found) == [(0, "bell"), (7, "bell")]


def test_a_muted_ring_starts_the_refractory_time_too():
    found = rings([{"Doorbell": 0.9}, {"Doorbell": 0.9}], muted=frozenset({0}))

    assert [(i, event.muted) for i, event in found] == [(0, True)]


def test_a_quiet_window_counts_as_one_where_nothing_scored():
    assert labels(rings([{"Knock": 0.9}, None, {"Knock": 0.9}])) == [(2, "knock")]
    assert rings([{"Knock": 0.9}, None, None, {"Knock": 0.9}]) == []


def test_detect_does_not_classify_a_quiet_window():
    classifier = ScriptedClassifier({"Knock": 0.9})
    rules = RingRules(DetectorConfig(), NAMES, classifier.model)
    # 1 s of a tone at -20 dBFS, then 2 s of a tone below min_level_dbfs.
    waveform = np.concatenate([tone(1.0), tone(2.0, dbfs=-70.0)])

    found = doorbell.detect(classifier, rules, waveform, start_wall=START)

    levels = [doorbell.level_dbfs(w) for w in doorbell.windows_of(waveform)]
    loud = [level >= -55.0 for level in levels]
    assert classifier.classified == sum(loud) == 3
    assert not found.scores[~np.array(loud)].any()
    assert labels(found.rings) == [(1, "knock")]


def test_a_chime_with_music_in_it_rings():
    # A real chime lifts Music too: Chime sits under Bell, under Musical
    # instrument, in AudioSet's ontology.
    chime = {"Chime": 0.6, "Music": 0.8}

    assert labels(rings([{}] * 6 + [chime, chime])) == [(7, "chime")]


def test_a_chime_after_3_s_of_music_is_vetoed():
    music, chime = {"Music": 0.8}, {"Chime": 0.6, "Music": 0.8}

    assert rings([music] * 6 + [chime, chime]) == []
    # Television counts as Music, and 4 of the 6 windows before are enough.
    tv = {"Television": 0.5}
    assert rings([tv, {}, tv, tv, {}, tv, chime, chime]) == []
    # 3 of them are not.
    assert labels(rings([tv, {}, tv, {}, {}, tv, chime, chime])) == [(7, "chime")]


def test_the_context_veto_is_the_chimes_alone():
    music = {"Music": 0.8}

    assert labels(rings([music] * 6 + [{"Doorbell": 0.9}])) == [(6, "bell")]
    assert labels(rings([music] * 6 + [{"Knock": 0.9}] * 2)) == [(7, "knock")]


def test_the_event_describes_the_rings_best_window():
    found = rings(
        [
            {"Ding": 0.3, "Speech": 0.2},
            {"Ding": 0.7, "Chime": 0.5, "Glockenspiel": 0.6, "Speech": 0.1},
        ]
    )

    [(_, event)] = found
    assert event.label == "chime"
    assert event.score == 0.7
    assert event.top == (("Ding", 0.7), ("Glockenspiel", 0.6), ("Chime", 0.5))
    assert event.armed is True
    assert event.muted is False
    assert event.level_dbfs == -30.0
    # The ring began with its first window.
    assert event.onset_wall == "2026-10-27T16:00:00.000+07:00"
    assert event.model == "test_model"


def test_the_event_is_the_json_the_topic_carries():
    event = DoorbellEvent(
        label="bell",
        score=0.93,
        top=(("Doorbell", 0.93), ("Ding-dong", 0.61), ("Music", 0.12)),
        muted=False,
        armed=True,
        level_dbfs=-32.5,
        onset_wall="2026-10-27T16:00:00.480+07:00",
        model="yamnet_onnx_v1",
    )

    text = event.model_dump_json()

    assert json.loads(text) == {
        "v": 1,
        "label": "bell",
        "score": 0.93,
        "top": [["Doorbell", 0.93], ["Ding-dong", 0.61], ["Music", 0.12]],
        "muted": False,
        "armed": True,
        "level_dbfs": -32.5,
        "onset_wall": "2026-10-27T16:00:00.480+07:00",
        "model": "yamnet_onnx_v1",
    }
    assert DoorbellEvent.model_validate_json(text) == event
    assert doorbell.DOORBELL_TOPIC == "/walkie/doorbell"
    assert doorbell.DOORBELL_ARMED_TOPIC == "/walkie/doorbell_armed"


@pytest.mark.parametrize(
    "changes", [{"label": "buzzer"}, {"score": 1.5}, {"v": 2}, {"extra": True}]
)
def test_an_event_out_of_its_format_is_refused(changes):
    fields = {
        "label": "knock",
        "score": 0.9,
        "top": (),
        "muted": False,
        "armed": False,
        "level_dbfs": -40.0,
        "onset_wall": "2026-10-27T16:00:00.000+07:00",
        "model": "m",
    }

    with pytest.raises(pydantic.ValidationError):
        DoorbellEvent(**{**fields, **changes})


# -- The classifier -----------------------------------------------------------------------------


def test_a_class_map_with_the_rules_classes_loads(tmp_path):
    assert doorbell.read_class_map(write_class_map(tmp_path / "map.csv")) == NAMES


def test_a_wrong_name_in_the_class_map_fails_the_load(tmp_path):
    names = list(NAMES)
    names[CLASSES["Doorbell"]] = "Door"
    path = write_class_map(tmp_path / "map.csv", tuple(names))

    with pytest.raises(ModelError) as caught:
        doorbell.YamnetClassifier(tmp_path / "missing.onnx", path)

    # The names are checked before the model is looked for.
    assert str(caught.value) == (
        f"{path} does not hold the classes the rules read: 349 is 'Door', not 'Doorbell'."
    )


def test_a_class_map_of_another_size_fails_the_load(tmp_path):
    path = write_class_map(tmp_path / "map.csv", NAMES[:520])

    with pytest.raises(ModelError, match="holds 520 classes, not YAMNet's 521"):
        doorbell.read_class_map(path)


def test_a_missing_model_says_how_to_fetch_it(tmp_path):
    path = write_class_map(tmp_path / "map.csv")

    with pytest.raises(ModelError, match="run `pixi run fetch-doorbell-model`"):
        doorbell.read_class_map(tmp_path / "missing.csv")
    with pytest.raises(ModelError, match="missing; run `pixi run fetch-doorbell-model`"):
        doorbell.YamnetClassifier(tmp_path / "missing.onnx", path)


def test_a_file_that_is_not_a_model_fails_the_load(tmp_path):
    path = write_class_map(tmp_path / "map.csv")
    model = tmp_path / "yamnet.onnx"
    model.write_bytes(b"not a model")

    with pytest.raises(ModelError, match="Cannot load the doorbell model"):
        doorbell.YamnetClassifier(model, path)


# -- The eval -----------------------------------------------------------------------------------


CLIP = (tone(1.5) * 32767).astype(np.int16)


class QuietIn(ScriptedClassifier):
    """Scores as ScriptedClassifier does, except that it scores nothing in the given cases.

    A case is a (rate, onset) pair. evaluate() runs the classifier once per
    case of each clip, rate by rate and onset by onset, so the call count
    tells the case.
    """

    def __init__(self, classes: dict[str, float], cases: set[tuple[int, int]]) -> None:
        super().__init__(classes)
        self.quiet = {EVAL_RATES.index(rate) * ONSETS + onset for rate, onset in cases}
        self.calls = 0

    def classify(self, windows: np.ndarray) -> np.ndarray:
        found = super().classify(windows)
        call, self.calls = self.calls, self.calls + 1
        return found * 0 if call % (len(EVAL_RATES) * ONSETS) in self.quiet else found


def known_failures(directory: Path, *entries: str) -> None:
    (directory / doorbell.KNOWN_FAILURES_FILE).write_text("\n".join(entries))


def failure(clip: str, rates: list[int], onsets: list[int]) -> str:
    return f'[[failure]]\nclip = "{clip}"\nrates = {rates}\nonsets = {onsets}\nreason = "r"\n'


def test_the_eval_judges_each_directory_and_lists_the_failures(tmp_path):
    for name in ("knock/a.wav", "none/b.wav", "other/c.wav"):
        write_wav(tmp_path / name, CLIP)

    results = doorbell.evaluate(ScriptedClassifier({"Knock": 0.9}), DetectorConfig(), tmp_path)

    assert [(str(r.path), r.expected, r.ok) for r in results] == [
        ("knock/a.wav", "knock", True),
        ("none/b.wav", "none", False),
        ("other/c.wav", None, None),
    ]
    assert len(results[1].failures) == len(EVAL_RATES) * ONSETS
    lines = doorbell.report(results)
    assert lines[1].split() == ["knock/a.wav", "8/8", "8/8", "8/8", "ok"]
    assert lines[2].split() == ["none/b.wav", "8/8", "8/8", "8/8", "FALSE", "RING"]
    assert lines[3].split() == ["other/c.wav", "8/8", "8/8", "8/8"]
    # A line for each of none/b.wav's failing cases, and none for the
    # clip that is not judged.
    assert lines[4] == "none/b.wav at 16000 Hz, onset 0: knock 0.90, not in known_failures.toml"
    assert lines[4 + 3 * ONSETS] == "none/ at 16000 Hz: highest Knock 0.900, highest Tap 0.000"
    assert lines[-1] == (
        "FAIL: 1 of 3 clips; failing cases not in known_failures.toml: 24; "
        "listed cases that pass: 0"
    )


def test_the_eval_plays_each_clip_at_every_onset_within_a_hop():
    clip = np.array([5, 6], np.int16)

    np.testing.assert_array_equal(doorbell.delayed(clip, 0), clip)
    delayed = doorbell.delayed(clip, ONSETS - 1)
    # 7 eighths of a hop: 6720 samples, 420 ms.
    assert len(delayed) == 2 + 7 * HOP_SAMPLES // 8 == 6722
    assert delayed.dtype == np.int16
    assert not delayed[:-2].any()


def test_a_miss_at_one_onset_fails_the_eval(tmp_path):
    write_wav(tmp_path / "knock" / "a.wav", CLIP)
    classifier = QuietIn({"Knock": 0.9}, {(48000, 7)})

    [result] = doorbell.evaluate(classifier, DetectorConfig(), tmp_path)

    assert result.failures == result.unlisted == {(48000, 7)}
    assert result.ok is False
    lines = doorbell.report([result])
    assert lines[1].split() == ["knock/a.wav", "8/8", "7/8", "8/8", "MISS"]
    assert lines[2] == "knock/a.wav at 48000 Hz, onset 7: no ring, not in known_failures.toml"


def test_a_failure_that_known_failures_toml_lists_does_not_fail_the_eval(tmp_path):
    write_wav(tmp_path / "knock" / "a.wav", CLIP)
    known_failures(tmp_path, failure("knock/a.wav", [48000, 44100], [7]))
    classifier = QuietIn({"Knock": 0.9}, {(48000, 7), (44100, 7)})

    [result] = doorbell.evaluate(classifier, DetectorConfig(), tmp_path)

    assert result.failures == result.known == {(48000, 7), (44100, 7)}
    assert result.ok is True
    lines = doorbell.report([result])
    assert lines[1].split() == ["knock/a.wav", "8/8", "7/8", "7/8", "known", "failures"]
    assert lines[2] == "knock/a.wav at 48000 Hz, onset 7: no ring, as known_failures.toml lists"
    # Not a bare PASS: the eval says that cases fail.
    assert lines[-1] == (
        "KNOWN FAILURES ONLY: 2 of 24 cases fail, each listed in known_failures.toml"
    )


def test_a_listed_case_that_passes_fails_the_eval(tmp_path):
    # So that known_failures.toml follows the rules: a change that fixes a
    # case takes it out of the file.
    write_wav(tmp_path / "none" / "b.wav", CLIP)
    known_failures(tmp_path, failure("none/b.wav", [16000], [0, 1]))

    [result] = doorbell.evaluate(ScriptedClassifier({}), DetectorConfig(), tmp_path)

    assert result.failures == set()
    assert result.fixed == {(16000, 0), (16000, 1)}
    assert result.ok is False
    lines = doorbell.report([result])
    assert lines[1].split() == ["none/b.wav", "0/8", "0/8", "0/8", "FIXED"]
    assert lines[2] == (
        "none/b.wav at 16000 Hz, onset 0: no ring, though known_failures.toml lists it as a "
        "failure"
    )
    assert lines[-1].endswith("not in known_failures.toml: 0; listed cases that pass: 2")


def test_a_directory_whose_rule_is_off_is_not_judged(tmp_path):
    # As 07-voice.md says to do if speech scores as knocks on the robot.
    write_wav(tmp_path / "knock" / "a.wav", CLIP)
    write_wav(tmp_path / "none" / "b.wav", CLIP)
    known_failures(tmp_path, failure("knock/a.wav", [16000], [3]))
    config = DetectorConfig(knock_pair=OFF)

    results = doorbell.evaluate(ScriptedClassifier({"Knock": 0.9}), config, tmp_path)

    assert [(r.rule_off, r.ok, r.failures, r.fixed) for r in results] == [
        (True, None, set(), set()),
        (False, True, set(), set()),
    ]
    lines = doorbell.report(results)
    assert lines[1].split() == ["knock/a.wav", "0/8", "0/8", "0/8", "rule", "off"]
    assert "knock/ is not judged: the config turns its rule off" in lines
    assert lines[-1] == "PASS"


def test_the_bell_rule_is_off_only_with_both_its_thresholds_off():
    assert not doorbell.rule_is_off(DetectorConfig(bell_single=OFF), "bell")
    assert not doorbell.rule_is_off(DetectorConfig(bell_pair=OFF), "bell")
    assert doorbell.rule_is_off(DetectorConfig(bell_single=OFF, bell_pair=OFF), "bell")
    assert doorbell.rule_is_off(DetectorConfig(chime_pair=OFF), "chime")
    assert not doorbell.rule_is_off(DetectorConfig(chime_pair=OFF), "knock")


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (failure("other/c.wav", [16000], [0]), "other/c.wav is not a clip the eval judges in "),
        (
            failure("knock/a.wav", [22050], [0]),
            "knock/a.wav lists the rate 22050; the eval plays at 16000, 48000, 44100 Hz.",
        ),
        (
            failure("knock/a.wav", [16000], [8]),
            "knock/a.wav lists the onset 8; onsets run from 0 to 7.",
        ),
        (
            failure("knock/a.wav", [48000], [7]) + failure("knock/a.wav", [48000, 44100], [7]),
            "knock/a.wav lists 48000 Hz at onset 7 more than once.",
        ),
        (failure("knock/a.wav", [], [0]), "rates"),
        (failure("knock/a.wav", [16000], [0]) + "why = 1\n", "why"),
        (failure("knock/a.wav", [16000], [0]).replace('"r"', '""'), "reason"),
        ("[[failure]\n", "Expected ']]'"),
    ],
)
def test_known_failures_toml_out_of_its_format_is_refused(tmp_path, entry, message):
    known_failures(tmp_path, entry)

    with pytest.raises(ValueError) as caught:
        doorbell.read_known_failures(tmp_path, {"knock/a.wav"})

    assert str(caught.value).startswith(f"{tmp_path / doorbell.KNOWN_FAILURES_FILE}: ")
    assert message in str(caught.value)


def test_the_eval_refuses_a_clip_that_is_not_16_khz_mono(tmp_path):
    path = write_wav(tmp_path / "bell" / "a.wav", np.zeros(4800, np.int16), rate=48000)

    with pytest.raises(ValueError, match="is not a 16 kHz mono 16-bit WAV file"):
        doorbell.read_clip(path)


def test_the_eval_without_a_model_exits_with_the_fetch_hint(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text("")

    with pytest.raises(SystemExit) as caught:
        doorbell.main(["eval", str(AUDIO), "--config", str(config)])

    assert caught.value.code == 2
    assert "pixi run fetch-doorbell-model" in capsys.readouterr().err


# -- The model ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classifier() -> doorbell.YamnetClassifier:
    return doorbell.load_classifier(PROJECT_ROOT)


@needs_model
def test_the_model_scores_yamnets_classes(classifier):
    assert classifier.class_names[CLASSES["Doorbell"]] == "Doorbell"
    assert classifier.classify(np.zeros((3, WINDOW_SAMPLES), np.float32)).shape == (3, 521)
    assert classifier.classify(np.zeros((0, WINDOW_SAMPLES), np.float32)).shape == (0, 521)


@needs_model
@pytest.mark.parametrize("rate", EVAL_RATES)
def test_the_onnx_model_scores_as_tensorflow_does(classifier, rate):
    # The numpy features and the ONNX model, on the clip as a microphone at
    # `rate` delivers it, against TF-Hub's YAMNet end to end on the same
    # audio (tools/export_yamnet.py).
    with np.load(REFERENCE) as reference:
        keys = sorted(key for key in reference.files if key.endswith(f"@{rate}"))
        clips = sorted(path.relative_to(AUDIO).as_posix() for path in AUDIO.rglob("*.wav"))
        assert keys == [f"{clip}@{rate}" for clip in clips]
        for key in keys:
            waveform = doorbell.through_rai(doorbell.read_clip(AUDIO / key.split("@")[0]), rate)
            got = classifier.classify(doorbell.windows_of(waveform))
            expected = reference[key]
            assert got.shape == expected.shape, key
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-3, err_msg=key)


@pytest.fixture(scope="module")
def results(classifier) -> dict[str, doorbell.ClipResult]:
    """The eval of the test clips at every rate and onset, with config.toml's rules, by clip."""
    found = doorbell.evaluate(classifier, doorbell.load_detector_config(CONFIG), AUDIO)
    return {result.path.as_posix(): result for result in found}


def clip_id(path: Path) -> str:
    return path.name


@needs_model
@pytest.mark.parametrize("path", RING_CLIPS, ids=clip_id)
def test_a_ring_clip_rings_at_every_rate_and_onset_but_its_known_failures(results, path):
    result = results[path.relative_to(AUDIO).as_posix()]
    if result.rule_off:
        pytest.skip(f"config.toml turns off the rule that {path.parent.name}/ tests")

    # known_failures.toml says why each case it lists fails.
    assert result.failures == result.known
    found = [ring for at_rate in result.rings for ring in at_rate if ring is not None]
    # A door chime may ring as a bell: YAMNet hears some as a Doorbell.
    allowed = {"bell", "chime"} if path.parent.name == "chime" else {path.parent.name}
    assert {event.label for event in found} <= allowed
    assert {(event.model, event.muted) for event in found} == {("yamnet_onnx_v1", False)}


@needs_model
@pytest.mark.parametrize("path", QUIET_CLIPS, ids=clip_id)
def test_speech_music_and_room_tone_never_ring_but_in_their_known_failures(results, path):
    result = results[path.relative_to(AUDIO).as_posix()]

    assert result.failures == result.known


@needs_model
def test_the_eval_of_the_test_clips_fails_only_where_known_failures_toml_says(capsys):
    assert doorbell.main(["eval", str(AUDIO), "--config", str(CONFIG)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith(("PASS", "KNOWN FAILURES ONLY: "))


def test_the_test_clips_cover_every_judged_directory():
    assert {path.parent.name for path in RING_CLIPS} == set(doorbell.RING_DIRS)
    assert QUIET_CLIPS
    # Buzzers are scored and not judged: the rules miss most of them
    # (ADR-0054), and the eval shows by how much.
    assert "buzzer" not in doorbell.RING_DIRS
    assert list((AUDIO / "buzzer").glob("*.wav"))


def test_the_known_failures_of_the_test_clips_are_in_the_format():
    # Without the model too: each entry names a judged clip, the eval's
    # rates and onsets, and a reason.
    judged = {path.relative_to(AUDIO).as_posix() for path in (*RING_CLIPS, *QUIET_CLIPS)}

    assert doorbell.read_known_failures(AUDIO, judged)


# -- The module ---------------------------------------------------------------------------------


def test_the_detector_imports_nothing_of_the_agent():
    # In a fresh interpreter, since this one has the agent and rclpy loaded
    # already: voice-asr loads the detector and reads its table without the
    # agent, and ONNX Runtime waits for a model to load.
    code = (
        "import json, sys\n"
        "import walkie_agent.doorbell as module\n"
        "module.load_detector_config(sys.argv[1])\n"
        "roots = ('walkie_agent.tuning', 'walkie_agent.agent', 'rclpy', 'onnxruntime')\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if any(name == root or name.startswith(root + '.') for root in roots)\n"
        ")\n"
        "walkie = sorted(name for name in sys.modules if name.split('.')[0] == 'walkie_agent')\n"
        "print(json.dumps({'file': module.__file__, 'loaded': loaded, 'walkie': walkie}))\n"
    )
    path = os.pathsep.join(part for part in (str(SRC), os.environ.get("PYTHONPATH")) if part)
    result = subprocess.run(
        [sys.executable, "-c", code, str(CONFIG)],
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    found = json.loads(result.stdout.strip().splitlines()[-1])

    assert found["loaded"] == []
    assert found["walkie"] == [
        "walkie_agent",
        "walkie_agent.config_tables",
        "walkie_agent.doorbell",
    ]
    assert Path(found["file"]).resolve().is_relative_to(SRC.resolve())
