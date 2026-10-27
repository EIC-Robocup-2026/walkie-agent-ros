"""The doorbell detector: YAMNet's sound classes, and the rules that make a ring of them.

YAMNet (tensorflow/models, research/audioset/yamnet) scores the 521 AudioSet
classes on windows of 0.975 s of 16 kHz audio, one window every 0.48 s. Its
core network runs here on ONNX Runtime, exported once from the TF-Hub weights
by tools/export_yamnet.py. Its log-mel features are computed here in numpy,
ported from its features.py. `pixi run fetch-doorbell-model` downloads the
model and its class map into models/ (ADR-0054).

RingRules turns the windows into rings, with the thresholds that
[walkie.doorbell.detector] in config.toml sets (DetectorConfig):

    bell    Doorbell or Ding-dong, at bell_single in one window, or at
            bell_pair in 2 of the newest 3
    chime   Ding, Chime, Bell or Buzzer at chime_pair in 2 of the newest 3,
            unless Music or Television filled the 6 windows before it
    knock   Knock at knock_pair in 2 of the newest 3

A ring is muted when a block of its windows was captured while the robot
spoke. Each ring is a DoorbellEvent, the JSON that /walkie/doorbell carries.

    python -m walkie_agent.doorbell eval tests/data/audio

scores a directory of clips as a 16 kHz device delivers them, and as RAI's
resampled stream delivers them from a 48 kHz or a 44.1 kHz device, each at
8 onsets within a hop, and reports what rings. known_failures.toml in the
directory lists the cases known to fail, and why.

This module imports only numpy, pydantic, ONNX Runtime and
walkie_agent.config_tables, so a process without the agent and ROS can load
it. ONNX Runtime is imported only when a model is loaded: the agent imports
this module to check the detector's table, and runs no model.
"""

import argparse
import csv
import math
import sys
import tomllib
import wave
from collections import deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from walkie_agent import config_tables
from walkie_agent.config_tables import TuningError

# The detector's rings: std_msgs/String holding a DoorbellEvent's JSON,
# reliable, depth 10.
DOORBELL_TOPIC = "/walkie/doorbell"
# std_msgs/Bool: whether the agent is waiting for the doorbell.
DOORBELL_ARMED_TOPIC = "/walkie/doorbell_armed"

# The model each event names: YAMNet's TF-Hub weights, exported to ONNX.
MODEL_NAME = "yamnet_onnx_v1"
# The model and its class map, relative to the directory of the config
# file. `pixi run fetch-doorbell-model` downloads both there.
MODEL_FILE = Path("models/yamnet.onnx")
CLASS_MAP_FILE = Path("models/yamnet_class_map.csv")

SAMPLE_RATE = 16000
# voice-asr's microphone blocks, 80 ms at 16 kHz (walkie_agent.voice.BLOCK_SIZE).
BLOCK_SAMPLES = 1280
# A window is 96 frames 10 ms apart, plus the rest of the last frame's
# 25 ms, and a window starts every 0.48 s (YAMNet's params.py).
WINDOW_SAMPLES = 15600
HOP_SAMPLES = 7680
CLASS_COUNT = 521

# The AudioSet classes this module names, by their index in YAMNet's class
# map. Loading a classifier checks each name against the map, so a map with
# other indices fails the load. Names are never looked up at run time:
# "Beep, bleep" holds a comma.
CLASSES: Mapping[str, int] = MappingProxyType(
    {
        "Speech": 0,
        "Speech synthesizer": 5,
        "Music": 132,
        "Tubular bells": 173,
        "Glockenspiel": 176,
        "Bell": 195,
        "Jingle bell": 197,
        "Chime": 200,
        "Wind chime": 201,
        "Doorbell": 349,
        "Ding-dong": 350,
        "Knock": 353,
        "Tap": 354,
        "Alarm": 382,
        "Telephone bell ringing": 384,
        "Buzzer": 392,
        "Beep, bleep": 475,
        "Ding": 477,
        "Jingle, tinkle": 489,
        "Television": 518,
    }
)
# The classes of each rule. Doorbell has two parents in the AudioSet
# ontology, Door and Alarm, and Ding-dong is its child. Chime is a child of
# Bell, which sits under Musical instrument, so a real chime lifts Music too:
# the chime rule has a context veto instead of a Music veto.
BELL_CLASSES = ("Doorbell", "Ding-dong")
CHIME_CLASSES = ("Ding", "Chime", "Bell", "Buzzer")
KNOCK_CLASSES = ("Knock",)
# Music or Television in most of the windows before a chime: a chime inside
# a song, or on a TV.
CONTEXT_CLASSES = ("Music", "Television")
# Classes that no rule reads, though they sound like a ring or a knock: the
# robot's own speech, beeps, alarms, phones, musical bells and taps.
NEVER_TRIGGER = (
    "Speech",
    "Speech synthesizer",
    "Beep, bleep",
    "Alarm",
    "Telephone bell ringing",
    "Tubular bells",
    "Glockenspiel",
    "Jingle bell",
    "Wind chime",
    "Jingle, tinkle",
    "Tap",
)

# The window counts of the rules are fixed: config sets only thresholds. A
# pair rule needs a hit in the newest window and in one of the two before.
PAIR_WINDOWS = 3
PAIR_HITS = 2
# A chime is vetoed when 4 of the 6 windows before its first hit had Music
# or Television at context_veto: about 2.9 s of a song or a TV.
CONTEXT_WINDOWS = 6
CONTEXT_HITS = 4
# A threshold of this turns its rule off, since no score is above 1.
OFF = 1.1

Label = Literal["bell", "chime", "knock"]


# -- Config -------------------------------------------------------------------------------------

DETECTOR_TABLE = "walkie.doorbell.detector"
DETECTOR_KEYS = (
    "enabled",
    "bell_single",
    "bell_pair",
    "chime_pair",
    "knock_pair",
    "context_veto",
    "sustain",
    "min_level_dbfs",
    "refractory_s",
    "post_ring_ignore_s",
    "max_ignore_s",
)
# The keys that hold a class score, from 0 to 1, or OFF.
_SCORE_KEYS = ("bell_single", "bell_pair", "chime_pair", "knock_pair", "context_veto", "sustain")


@dataclass(frozen=True)
class DetectorConfig:
    """The rules' thresholds and timings, from [walkie.doorbell.detector] in config.toml.

    The values are starting points, to tune on the robot. A score threshold
    lies in (0, 1.1], and 1.1 turns its rule off. enabled, sustain and the
    post-ring timings are for voice-asr's tap, which is not built yet
    (ADR-0054); the eval reads the rest.
    """

    # false keeps the tap off: no model is loaded, and nothing is published.
    enabled: bool = True
    # max(Doorbell, Ding-dong) in one window. No value separates the eval's
    # clips at every onset: the music box in none/ reaches 0.43 in one
    # window, and bell/freesound_69386 rings only on 0.40 from a 44.1 kHz
    # device (known_failures.toml).
    bell_single: float = 0.25
    # The same, in 2 of the newest 3 windows.
    bell_pair: float = 0.15
    # max(Ding, Chime, Bell, Buzzer) in 2 of the newest 3 windows. RAI's
    # per-block resample halves a door chime's score: in 2 of 3 windows the
    # eval's chime/freesound_491404 reaches 0.69 at 16 kHz, but 0.32 and
    # 0.28 from 48 and 44.1 kHz devices, and 0.08 at some onsets
    # (known_failures.toml). The clips in none/ reach at most 0.18, at any
    # rate and onset.
    chime_pair: float = 0.25
    # Knock in 2 of the newest 3 windows. At any rate and onset, the eval's
    # speech, music and room tone score at most 0.03 as Knock and 0.05 as
    # Tap. 1.1 turns the rule off, if they score as knocks on the robot.
    knock_pair: float = 0.35
    # Music or Television at this in 4 of the 6 windows before a chime vetoes it.
    context_veto: float = 0.3
    # The post-ring window lasts while the bell and chime classes stay at this.
    sustain: float = 0.1
    # Windows quieter than this, in dB below full scale, are not classified.
    min_level_dbfs: float = -55.0
    # Seconds after a ring before the next one: one event per ring.
    refractory_s: float = 3.0
    # Seconds the microphone stays closed after a ring the agent waits for,
    # and the longest that sustain may keep it closed.
    post_ring_ignore_s: float = 2.5
    max_ignore_s: float = 10.0

    def __post_init__(self) -> None:
        for name in _SCORE_KEYS:
            value = getattr(self, name)
            if not 0 < value <= OFF:
                raise ValueError(f"DetectorConfig.{name} must lie in (0, {OFF:g}], not {value}.")
        if not (math.isfinite(self.min_level_dbfs) and self.min_level_dbfs < 0):
            raise ValueError(
                f"DetectorConfig.min_level_dbfs must be below 0, not {self.min_level_dbfs}."
            )
        for name in ("refractory_s", "post_ring_ignore_s", "max_ignore_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"DetectorConfig.{name} must be a positive number, not {value}.")
        if self.post_ring_ignore_s > self.max_ignore_s:
            raise ValueError("DetectorConfig.post_ring_ignore_s must not be above max_ignore_s.")


def parse_detector_config(raw: Mapping[str, Any], errors: list[str]) -> DetectorConfig:
    """The config that the table [walkie.doorbell.detector], `raw`, sets.

    Each problem is added to `errors`, and the default stands in for a value
    that is refused. The agent's loader (walkie_agent.tuning) checks the
    table with this too, so a typo in it stops the agent as well as the
    detector.
    """
    config_tables.refuse_unknown(raw, DETECTOR_KEYS, DETECTOR_TABLE, errors)
    changes = config_tables.read_keys(DetectorConfig, raw, DETECTOR_KEYS, DETECTOR_TABLE, errors)
    return config_tables.replace(DetectorConfig(), changes, DETECTOR_TABLE, errors)


def load_detector_config(path: str | Path) -> DetectorConfig:
    """Read [walkie.doorbell.detector] from the config file `path`.

    Raises TuningError listing every problem in the table. The file's other
    tables are left to their readers: the agent checks the rest of [walkie].
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    errors: list[str] = []
    walkie = config_tables.table(data, "walkie", errors)
    doorbell = config_tables.table(walkie, "walkie.doorbell", errors)
    raw = config_tables.table(doorbell, DETECTOR_TABLE, errors)
    config = parse_detector_config(raw, errors)
    if errors:
        raise TuningError(str(path), errors)
    return config


# -- Audio --------------------------------------------------------------------------------------

# YAMNet's features (features.py, with params.py's values): a 25 ms
# periodic Hann window every 10 ms, the magnitudes of a 512-point FFT, 64
# mel bands from 125 to 7500 Hz, and log(mel + 0.001).
_FRAME = 400
_FRAME_HOP = 160
_FFT = 512
_MEL_BANDS = 64
_MEL_MIN_HZ = 125.0
_MEL_MAX_HZ = 7500.0
_LOG_OFFSET = 0.001


def _hertz_to_mel(hertz: Any) -> Any:
    # HTK's mel scale, as tf.signal's.
    return 1127.0 * np.log(1.0 + np.asarray(hertz, np.float64) / 700.0)


def _mel_matrix() -> np.ndarray:
    """tf.signal.linear_to_mel_weight_matrix with YAMNet's values: [257, 64] float32."""
    # HTK leaves out the DC bin: its row stays zero.
    mel = _hertz_to_mel(np.linspace(0.0, SAMPLE_RATE / 2, _FFT // 2 + 1)[1:])[:, None]
    edges = np.linspace(_hertz_to_mel(_MEL_MIN_HZ), _hertz_to_mel(_MEL_MAX_HZ), _MEL_BANDS + 2)
    lower, center, upper = edges[:-2], edges[1:-1], edges[2:]
    rising = (mel - lower) / (center - lower)
    falling = (upper - mel) / (upper - center)
    weights = np.maximum(0.0, np.minimum(rising, falling))
    return np.pad(weights, ((1, 0), (0, 0))).astype(np.float32)


_MEL = _mel_matrix()
_HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_FRAME) / _FRAME)).astype(np.float32)


def log_mel_patches(windows: np.ndarray) -> np.ndarray:
    """YAMNet's input for windows of 16 kHz audio: [n, 96, 64] float32 from [n, WINDOW_SAMPLES]."""
    windows = np.asarray(windows, np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(windows, _FRAME, axis=-1)[
        ..., ::_FRAME_HOP, :
    ]
    magnitudes = np.abs(np.fft.rfft(frames * _HANN, n=_FFT)).astype(np.float32)
    return np.log(magnitudes @ _MEL + _LOG_OFFSET).astype(np.float32)


def pad_waveform(waveform: np.ndarray) -> np.ndarray:
    """`waveform` followed by zeros up to a whole number of windows, as YAMNet pads it."""
    count = len(waveform)
    hops = max(0, math.ceil((count - WINDOW_SAMPLES) / HOP_SAMPLES))
    return np.pad(waveform, (0, WINDOW_SAMPLES + hops * HOP_SAMPLES - count))


def windows_of(waveform: np.ndarray) -> np.ndarray:
    """The windows of a 16 kHz waveform, padded as YAMNet pads it: [n, WINDOW_SAMPLES].

    Window i starts at sample HOP_SAMPLES * i, so its scores are YAMNet's
    patch i.
    """
    padded = pad_waveform(np.asarray(waveform, np.float32))
    return np.lib.stride_tricks.sliding_window_view(padded, WINDOW_SAMPLES)[::HOP_SAMPLES]


def level_dbfs(window: np.ndarray) -> float:
    """The RMS level of audio in [-1, 1], in dB below full scale; -inf for silence."""
    rms = math.sqrt(float(np.mean(np.square(window, dtype=np.float64))))
    return 20 * math.log10(rms) if rms > 0 else -math.inf


def to_float(block: np.ndarray) -> np.ndarray:
    """A microphone block in [-1, 1] as float32, from int16 or from float64 at int16 scale.

    A 16 kHz stream delivers int16. Any other rate reaches voice-asr through
    RAI's resample, as float64 at int16 scale, which can overshoot it. The
    scale is Silero VAD's.
    """
    return np.clip(np.asarray(block, np.float32) * (1 / 32768), -1.0, 1.0)


def resample(x: np.ndarray, num: int) -> np.ndarray:
    """scipy.signal.resample(x, num) for a real 1-D `x`, in numpy.

    RAI resamples each block of a stream that is not at 16 kHz with it
    (rai_s2s sound_device/api.py), and the eval feeds the model the same
    audio. It treats the block as one period, so each block edge is a
    discontinuity.
    """
    count = len(x)
    kept = min(num, count)
    spectrum = np.fft.rfft(x)[: kept // 2 + 1]
    if kept % 2 == 0 and num != count:
        # The bin at kept // 2 has no pair: scipy unites the pair when
        # downsampling, and splits it when upsampling.
        spectrum[kept // 2] *= 2.0 if num < count else 0.5
    return np.fft.irfft(spectrum * (num / count), n=num)


def through_rai(clip: np.ndarray, device_rate: int) -> np.ndarray:
    """An int16 16 kHz clip as voice-asr receives it from a device at `device_rate`, in [-1, 1].

    The clip is played at the device's rate, read in 80 ms int16 blocks as
    RAI opens the stream, and each block is resampled to 16 kHz on its own,
    as RAI's callback does. A last part block is dropped: a stream delivers
    whole blocks only.
    """
    if device_rate == SAMPLE_RATE:
        return to_float(clip)
    played = resample(np.asarray(clip, np.float64), round(len(clip) * device_rate / SAMPLE_RATE))
    pcm = np.clip(np.round(played), -32768, 32767).astype(np.int16)
    # RAI's block at the device's rate: 3840 samples at 48 kHz, 3528 at 44.1 kHz.
    size = int(BLOCK_SAMPLES * device_rate / SAMPLE_RATE)
    blocks = [
        to_float(resample(pcm[start : start + size], int(size / device_rate * SAMPLE_RATE)))
        for start in range(0, len(pcm) - size + 1, size)
    ]
    return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)


def blocks_of(start: int, length: int = WINDOW_SAMPLES) -> range:
    """The 80 ms blocks, by index from sample 0, that hold a sample of the window from `start`."""
    return range(start // BLOCK_SAMPLES, (start + length - 1) // BLOCK_SAMPLES + 1)


def window_muted(speaking: Sequence[bool], start: int) -> bool:
    """Whether a block of the window from sample `start` carries the speaking flag.

    `speaking` holds each block's flag, from block 0: whether the robot was
    speaking, or had just stopped, when the block was captured. So a window
    scored after the flag dropped is still muted while it holds a block of
    the robot's own audio.
    """
    return any(speaking[i] for i in blocks_of(start) if i < len(speaking))


# -- Classifier ---------------------------------------------------------------------------------


class ModelError(RuntimeError):
    """The model or its class map is missing, unreadable, or not the classes the rules read."""


class SoundClassifier(Protocol):
    """A model that scores sound classes on windows of 16 kHz audio.

    The rules read classes by YAMNet's indices (CLASSES), so another model
    puts its scores at those indices.
    """

    # The model each event names, such as yamnet_onnx_v1.
    model: str
    # The class names, by index.
    class_names: tuple[str, ...]

    def classify(self, windows: np.ndarray) -> np.ndarray:
        """The scores [n, classes] of windows [n, WINDOW_SAMPLES] of 16 kHz audio in [-1, 1]."""
        ...


def read_class_map(path: Path) -> tuple[str, ...]:
    """The display names in YAMNet's class map (index,mid,display_name), by index.

    Raises ModelError when the file cannot be read, or a class of CLASSES has
    another name at its index.
    """
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        raise ModelError(
            f"Cannot read the class map {path} ({e.strerror}); run `pixi run fetch-doorbell-model`."
        ) from e
    names = tuple(row.get("display_name") or "" for row in rows)
    if len(names) != CLASS_COUNT:
        raise ModelError(f"{path} holds {len(names)} classes, not YAMNet's {CLASS_COUNT}.")
    wrong = [
        f"{index} is {names[index]!r}, not {name!r}"
        for name, index in CLASSES.items()
        if names[index] != name
    ]
    if wrong:
        raise ModelError(
            f"{path} does not hold the classes the rules read: " + "; ".join(wrong) + "."
        )
    return names


class YamnetClassifier:
    """YAMNet's core network on ONNX Runtime, on one CPU thread, fed the numpy log-mel."""

    model = MODEL_NAME

    def __init__(self, model_path: Path, class_map_path: Path) -> None:
        # The names first: a class map that does not match fails the load
        # before the model is read.
        self.class_names = read_class_map(class_map_path)
        if not model_path.is_file():
            raise ModelError(
                f"The doorbell model {model_path} is missing; run `pixi run fetch-doorbell-model`."
            )
        # Imported here: the agent imports this module only to check the
        # detector's table, and has no use for ONNX Runtime.
        import onnxruntime

        options = onnxruntime.SessionOptions()
        # One thread each: the detector shares the CPU with Whisper and
        # Silero VAD, and ONNX Runtime releases the GIL while it runs.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            self._session = onnxruntime.InferenceSession(
                str(model_path), options, providers=["CPUExecutionProvider"]
            )
        # ONNX Runtime's errors for a file that is not a model derive from
        # Exception only.
        except Exception as e:
            raise ModelError(f"Cannot load the doorbell model {model_path}: {e}") from e
        self._input = self._session.get_inputs()[0].name
        output = self._session.get_outputs()[0]
        if output.shape[-1] != CLASS_COUNT:
            raise ModelError(f"{model_path} does not score YAMNet's {CLASS_COUNT} classes.")
        self._output = output.name

    def classify(self, windows: np.ndarray) -> np.ndarray:
        if len(windows) == 0:
            return np.zeros((0, CLASS_COUNT), np.float32)
        patches = log_mel_patches(windows)
        (scores,) = self._session.run([self._output], {self._input: patches})
        return scores


def load_classifier(directory: str | Path) -> YamnetClassifier:
    """YAMNet from models/ in `directory`, the directory of the config file."""
    return YamnetClassifier(Path(directory) / MODEL_FILE, Path(directory) / CLASS_MAP_FILE)


# -- Rules --------------------------------------------------------------------------------------


class DoorbellEvent(BaseModel):
    """One ring, as /walkie/doorbell carries it: the JSON of a std_msgs/String."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The format's version.
    v: Literal[1] = 1
    label: Label
    # The rule's score in the ring's best window: the highest of its classes.
    score: float = Field(ge=0, le=1)
    # The three highest classes of that window, with their scores.
    top: tuple[tuple[str, float], ...]
    # A block of the ring's windows carried the speaking flag: the robot may
    # have heard itself.
    muted: bool
    # The agent was waiting for the doorbell.
    armed: bool
    # The level of the best window, in dB below full scale.
    level_dbfs: float
    # When the ring's first window began, in ISO 8601 with the UTC offset.
    onset_wall: str
    model: str


@dataclass(frozen=True, eq=False)
class Window:
    """One window, as the rules read it."""

    # Each class's score; None for a window too quiet to classify.
    scores: np.ndarray | None
    level_dbfs: float
    # A block of the window carries the speaking flag (window_muted).
    muted: bool
    # When the window's first sample was captured: on a steady clock in
    # seconds, for the refractory time, and on the wall clock, for the event.
    start_s: float
    start_wall: datetime


@dataclass(frozen=True)
class _Seen:
    """A window, with the highest score of each rule's classes."""

    window: Window
    bell: float = 0.0
    chime: float = 0.0
    knock: float = 0.0
    context: float = 0.0


def _indices(names: Sequence[str]) -> np.ndarray:
    return np.array([CLASSES[name] for name in names])


_BELL = _indices(BELL_CLASSES)
_CHIME = _indices(CHIME_CLASSES)
_KNOCK = _indices(KNOCK_CLASSES)
_CONTEXT = _indices(CONTEXT_CLASSES)


class RingRules:
    """The trigger rules over a stream of windows, oldest first: at most one ring per window.

    A quiet window counts as one where every class scored 0. When more than
    one rule holds, bell wins over chime, and chime over knock.
    """

    def __init__(self, config: DetectorConfig, class_names: Sequence[str], model: str) -> None:
        self.config = config
        self._class_names = tuple(class_names)
        self._model = model
        # The newest windows: 3 for the pair rules, and the 6 before a pair's
        # first window for the context veto.
        self._seen: deque[_Seen] = deque(maxlen=PAIR_WINDOWS + CONTEXT_WINDOWS)
        self._last_ring_s: float | None = None

    def step(self, window: Window, armed: bool = False) -> DoorbellEvent | None:
        """The ring that `window`, the newest, completes, or None. `armed` goes into the event."""
        scores = window.scores
        if scores is None:
            self._seen.append(_Seen(window))
            return None
        self._seen.append(
            _Seen(
                window,
                bell=float(scores[_BELL].max()),
                chime=float(scores[_CHIME].max()),
                knock=float(scores[_KNOCK].max()),
                context=float(scores[_CONTEXT].max()),
            )
        )
        last = self._last_ring_s
        if last is not None and window.start_s - last < self.config.refractory_s:
            return None
        ring = self._ring()
        if ring is None:
            return None
        self._last_ring_s = window.start_s
        label, hits = ring
        return self._event(label, hits, armed)

    def _ring(self) -> tuple[Label, list[_Seen]] | None:
        c = self.config
        newest = list(self._seen)[-PAIR_WINDOWS:]
        if newest[-1].bell >= c.bell_single:
            return "bell", [newest[-1]]
        if hits := self._pair(newest, "bell", c.bell_pair):
            return "bell", hits
        if (hits := self._pair(newest, "chime", c.chime_pair)) and not self._vetoed(hits[0]):
            return "chime", hits
        if hits := self._pair(newest, "knock", c.knock_pair):
            return "knock", hits
        return None

    @staticmethod
    def _pair(newest: list[_Seen], group: str, threshold: float) -> list[_Seen]:
        """The hit windows when the newest window and one of the two before it are hits."""
        hits = [seen for seen in newest if getattr(seen, group) >= threshold]
        if len(hits) >= PAIR_HITS and hits[-1] is newest[-1]:
            return hits
        return []

    def _vetoed(self, first: _Seen) -> bool:
        """Whether Music or Television filled the windows before the window `first`."""
        seen = list(self._seen)
        before = seen[: seen.index(first)][-CONTEXT_WINDOWS:]
        hits = sum(1 for other in before if other.context >= self.config.context_veto)
        return hits >= CONTEXT_HITS

    def _event(self, label: Label, hits: list[_Seen], armed: bool) -> DoorbellEvent:
        best = max(hits, key=lambda seen: getattr(seen, label))
        scores = best.window.scores
        assert scores is not None
        top = np.argsort(-scores, kind="stable")[:3]
        return DoorbellEvent(
            label=label,
            score=round(getattr(best, label), 3),
            top=tuple((self._class_names[i], round(float(scores[i]), 3)) for i in top),
            muted=any(seen.window.muted for seen in hits),
            armed=armed,
            level_dbfs=round(best.window.level_dbfs, 1),
            onset_wall=hits[0].window.start_wall.isoformat(timespec="milliseconds"),
            model=self._model,
        )


@dataclass(frozen=True, eq=False)
class Detection:
    """What detect() found in a waveform."""

    # (window index, event) for each ring, in order.
    rings: list[tuple[int, DoorbellEvent]]
    # Each window's scores, [windows, classes]; a quiet window's row is 0.
    scores: np.ndarray


def detect(
    classifier: SoundClassifier,
    rules: RingRules,
    waveform: np.ndarray,
    speaking: Sequence[bool] = (),
    start_wall: datetime | None = None,
) -> Detection:
    """The rings in a 16 kHz waveform in [-1, 1], window by window.

    `speaking` holds the speaking flag of each 80 ms block from sample 0.
    Windows quieter than the config's min_level_dbfs are not classified.
    """
    framed = windows_of(waveform)
    levels = [level_dbfs(window) for window in framed]
    loud = np.array([level >= rules.config.min_level_dbfs for level in levels], bool)
    scores = np.zeros((len(framed), len(classifier.class_names)), np.float32)
    scores[loud] = classifier.classify(framed[loud])
    wall = start_wall or datetime.now().astimezone()
    rings = []
    for i in range(len(framed)):
        start = i * HOP_SAMPLES
        window = Window(
            scores=scores[i] if loud[i] else None,
            level_dbfs=levels[i],
            muted=window_muted(speaking, start),
            start_s=start / SAMPLE_RATE,
            start_wall=wall + timedelta(seconds=start / SAMPLE_RATE),
        )
        if event := rules.step(window):
            rings.append((i, event))
    return Detection(rings=rings, scores=scores)


# -- Eval ---------------------------------------------------------------------------------------

# The device rates the eval plays each clip at: its own 16 kHz, and the
# 48 and 44.1 kHz of a PipeWire default device, through RAI's resample.
EVAL_RATES = (16000, 48000, 44100)
# The eval plays each clip after 0 to 7 eighths of a hop of silence, 60 ms
# apart: a sound starts anywhere between two windows' starts, and each
# onset shows the rules other windows of it.
ONSETS = 8
# The directories whose clips must ring, and the one whose clips must not.
# Clips in any other directory, such as buzzer/, are scored and not judged.
RING_DIRS = ("bell", "chime", "knock")
QUIET_DIR = "none"
# The thresholds of the rule that each ring directory tests. With all of
# them at OFF, the rule is off, and the directory's clips are not judged.
RULE_KEYS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"bell": ("bell_single", "bell_pair"), "chime": ("chime_pair",), "knock": ("knock_pair",)}
)
# The file in an eval directory that lists the cases known to fail.
KNOWN_FAILURES_FILE = "known_failures.toml"


def rule_is_off(config: DetectorConfig, directory: str) -> bool:
    """Whether `config` turns off the rule that the ring directory `directory` tests."""
    return all(getattr(config, key) >= OFF for key in RULE_KEYS[directory])


def delayed(clip: np.ndarray, onset: int) -> np.ndarray:
    """`clip` after `onset` eighths of a hop of silence."""
    return np.concatenate([np.zeros(onset * HOP_SAMPLES // ONSETS, clip.dtype), clip])


class KnownFailure(BaseModel):
    """An entry of known_failures.toml: cases of one clip that the eval knows to fail, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The clip, by its path under the eval directory, such as bell/a.wav.
    clip: StrictStr
    # The cases are each of the rates at each of the onsets, an onset being
    # a delay in eighths of a hop.
    rates: tuple[StrictInt, ...] = Field(min_length=1)
    onsets: tuple[StrictInt, ...] = Field(min_length=1)
    reason: StrictStr = Field(min_length=1)


class _KnownFailuresFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    failure: tuple[KnownFailure, ...] = ()


def read_known_failures(
    directory: Path, judged: Collection[str]
) -> dict[str, frozenset[tuple[int, int]]]:
    """The (rate, onset) cases known to fail, by clip, from known_failures.toml in `directory`.

    `judged` holds the paths, under `directory`, of the clips the eval
    judges. No file means no known failures. Raises ValueError for a file
    that is not TOML or not in the format, and for an entry that names a
    clip outside `judged`, a rate outside EVAL_RATES, an onset outside 0 to
    7, or a case that another entry names too.
    """
    path = directory / KNOWN_FAILURES_FILE
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as f:
            entries = _KnownFailuresFile.model_validate(tomllib.load(f)).failure
    # TOML's decode error and pydantic's ValidationError are ValueErrors.
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    known: dict[str, set[tuple[int, int]]] = {}
    for entry in entries:
        if entry.clip not in judged:
            raise ValueError(f"{path}: {entry.clip} is not a clip the eval judges in {directory}.")
        for rate in entry.rates:
            if rate not in EVAL_RATES:
                raise ValueError(
                    f"{path}: {entry.clip} lists the rate {rate}; the eval plays at "
                    + ", ".join(str(each) for each in EVAL_RATES)
                    + " Hz."
                )
        for onset in entry.onsets:
            if not 0 <= onset < ONSETS:
                raise ValueError(
                    f"{path}: {entry.clip} lists the onset {onset}; onsets run from 0 to "
                    f"{ONSETS - 1}."
                )
        cases = known.setdefault(entry.clip, set())
        for rate in entry.rates:
            for onset in entry.onsets:
                if (rate, onset) in cases:
                    raise ValueError(
                        f"{path}: {entry.clip} lists {rate} Hz at onset {onset} more than once."
                    )
                cases.add((rate, onset))
    return {clip: frozenset(cases) for clip, cases in known.items()}


def read_clip(path: Path) -> np.ndarray:
    """The samples of a 16 kHz mono 16-bit WAV file, as int16. Raises ValueError otherwise."""
    try:
        with wave.open(str(path), "rb") as f:
            shape = (f.getnchannels(), f.getsampwidth(), f.getframerate())
            frames = f.readframes(f.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError(f"{path} is not a WAV file: {e}") from e
    if shape != (1, 2, SAMPLE_RATE):
        raise ValueError(f"{path} is not a 16 kHz mono 16-bit WAV file.")
    return np.frombuffer(frames, dtype="<i2").copy()


@dataclass(frozen=True)
class ClipResult:
    """What one clip rang as, at each of EVAL_RATES after each onset."""

    path: Path
    # bell, chime or knock: it must ring; none: it must not; None: another
    # directory, whose clips are not judged.
    expected: str | None
    # rings[i][onset]: the first ring at EVAL_RATES[i] after `onset`
    # eighths of a hop, or None.
    rings: tuple[tuple[DoorbellEvent | None, ...], ...]
    # The highest Knock and Tap scores at each rate, over every onset and
    # the windows loud enough to classify.
    knock: tuple[float, ...]
    tap: tuple[float, ...]
    # The config turns off the rule that the clip's ring directory tests, so
    # the clip is not judged.
    rule_off: bool = False
    # The (rate, onset) cases that known_failures.toml lists for the clip.
    known: frozenset[tuple[int, int]] = frozenset()

    @property
    def judged(self) -> bool:
        return self.expected is not None and not self.rule_off

    @property
    def failures(self) -> frozenset[tuple[int, int]]:
        """The (rate, onset) cases where a judged clip rang, or stayed quiet, against its directory."""
        if not self.judged:
            return frozenset()
        must_ring = self.expected in RING_DIRS
        return frozenset(
            (rate, onset)
            for rate, at_rate in zip(EVAL_RATES, self.rings)
            for onset, ring in enumerate(at_rate)
            if (ring is None) == must_ring
        )

    @property
    def unlisted(self) -> frozenset[tuple[int, int]]:
        """The failures that known_failures.toml does not list."""
        return self.failures - self.known

    @property
    def fixed(self) -> frozenset[tuple[int, int]]:
        """The cases that known_failures.toml lists, and a judged clip passes."""
        return self.known - self.failures if self.judged else frozenset()

    @property
    def ok(self) -> bool | None:
        """Whether the clip fails where known_failures.toml says, and only there; None if not judged."""
        return self.failures == self.known if self.judged else None


def evaluate(
    classifier: SoundClassifier, config: DetectorConfig, directory: Path
) -> list[ClipResult]:
    """Score each WAV file under `directory` at each of EVAL_RATES and onsets, with fresh rules each time.

    A clip in a ring directory whose rule `config` turns off is not judged.
    The cases that known_failures.toml in `directory` lists go into the
    results (read_known_failures).
    """
    paths = sorted(directory.rglob("*.wav"))
    if not paths:
        raise ValueError(f"{directory} holds no WAV files.")
    names = [path.relative_to(directory) for path in paths]
    labels = [
        path.parent.name if path.parent.name in (*RING_DIRS, QUIET_DIR) else None
        for path in paths
    ]
    known = read_known_failures(
        directory, {name.as_posix() for name, label in zip(names, labels) if label}
    )
    results = []
    for path, name, label in zip(paths, names, labels):
        clip = read_clip(path)
        rings, knock, tap = [], [], []
        for rate in EVAL_RATES:
            at_rate, scores = [], []
            for onset in range(ONSETS):
                rules = RingRules(config, classifier.class_names, classifier.model)
                found = detect(classifier, rules, through_rai(delayed(clip, onset), rate))
                at_rate.append(found.rings[0][1] if found.rings else None)
                scores.append(found.scores)
            rings.append(tuple(at_rate))
            every = np.concatenate(scores)
            knock.append(float(every[:, CLASSES["Knock"]].max()))
            tap.append(float(every[:, CLASSES["Tap"]].max()))
        results.append(
            ClipResult(
                path=name,
                expected=label,
                rings=tuple(rings),
                knock=tuple(knock),
                tap=tuple(tap),
                rule_off=label in RING_DIRS and rule_is_off(config, label),
                known=known.get(name.as_posix(), frozenset()),
            )
        )
    return results


def _verdict(result: ClipResult) -> str:
    if result.expected is None:
        return ""
    if result.rule_off:
        return "rule off"
    if result.unlisted:
        return "MISS" if result.expected in RING_DIRS else "FALSE RING"
    if result.fixed:
        return "FIXED"
    return "known failures" if result.failures else "ok"


def _case_lines(result: ClipResult) -> list[str]:
    """A line for each case of `result` that fails, or that known_failures.toml lists."""
    lines = []
    for rate, at_rate in zip(EVAL_RATES, result.rings):
        for onset, ring in enumerate(at_rate):
            case = (rate, onset)
            if case in result.failures:
                note = (
                    "as known_failures.toml lists"
                    if case in result.known
                    else "not in known_failures.toml"
                )
            elif case in result.fixed:
                note = "though known_failures.toml lists it as a failure"
            else:
                continue
            rang = f"{ring.label} {ring.score:.2f}" if ring else "no ring"
            lines.append(f"{result.path} at {rate} Hz, onset {onset}: {rang}, {note}")
    return lines


def report(results: list[ClipResult]) -> list[str]:
    """The eval's lines: the table, each failing case, the Knock and Tap scores of none/, and the verdict.

    The table counts, for each clip at each rate, the onsets at which it rang.
    """
    width = max(len(str(result.path)) for result in results)
    lines = [f"{'clip':<{width}}" + "".join(f"  {f'{rate} Hz':>8}" for rate in EVAL_RATES)]
    for result in results:
        cells = "".join(
            f"  {f'{sum(ring is not None for ring in at_rate)}/{ONSETS}':>8}"
            for at_rate in result.rings
        )
        lines.append(f"{str(result.path):<{width}}{cells}  {_verdict(result)}".rstrip())
    for result in results:
        lines += _case_lines(result)
    for directory in sorted({str(result.expected) for result in results if result.rule_off}):
        lines.append(f"{directory}/ is not judged: the config turns its rule off")
    quiet = [result for result in results if result.expected == QUIET_DIR]
    for i, rate in enumerate(EVAL_RATES):
        if quiet:
            lines.append(
                f"{QUIET_DIR}/ at {rate} Hz: highest Knock "
                f"{max(result.knock[i] for result in quiet):.3f}, highest Tap "
                f"{max(result.tap[i] for result in quiet):.3f}"
            )
    unlisted = sum(len(result.unlisted) for result in results)
    fixed = sum(len(result.fixed) for result in results)
    failing = sum(len(result.failures) for result in results)
    if unlisted or fixed:
        failed = sum(1 for result in results if result.ok is False)
        lines.append(
            f"FAIL: {failed} of {len(results)} clips; failing cases not in known_failures.toml: "
            f"{unlisted}; listed cases that pass: {fixed}"
        )
    elif failing:
        cases = sum(1 for result in results if result.judged) * len(EVAL_RATES) * ONSETS
        lines.append(
            f"KNOWN FAILURES ONLY: {failing} of {cases} cases fail, each listed in "
            "known_failures.toml"
        )
    else:
        lines.append("PASS")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m walkie_agent.doorbell",
        description="The doorbell detector's offline tools.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    eval_parser = commands.add_parser(
        "eval",
        help="Score a directory of clips at 16, 48 and 44.1 kHz and 8 onsets, and report what "
        "rings.",
        description=(
            "Score each 16 kHz mono WAV file under DIRECTORY, as a 16 kHz device delivers it "
            "and as RAI's per-block resample delivers it from a 48 kHz or a 44.1 kHz device, "
            "each after 0 to 7 eighths of a hop of silence. A clip in bell/, chime/ or knock/ "
            "must ring at every rate and onset, and a clip in none/ must not, except where "
            "known_failures.toml in DIRECTORY lists the case. A directory whose rule the "
            "config turns off is not judged. Exit code 0 is a pass and 1 a fail: a failing "
            "case the file does not list, or a listed case that passes."
        ),
    )
    eval_parser.add_argument(
        "directory",
        type=Path,
        help="The clips, in bell/, chime/, knock/ and none/, and known_failures.toml.",
    )
    eval_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="The config file whose [walkie.doorbell.detector] sets the rules; the model is in "
        "models/ next to it.",
    )
    args = parser.parse_args(argv)
    try:
        config = load_detector_config(args.config)
        classifier = load_classifier(args.config.parent)
        results = evaluate(classifier, config, args.directory)
    # TuningError and TOML's decode error are ValueErrors.
    except (OSError, ValueError, ModelError) as e:
        parser.error(str(e))
    for line in report(results):
        print(line)
    return 0 if all(result.ok is not False for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
