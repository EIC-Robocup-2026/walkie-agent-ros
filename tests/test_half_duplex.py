"""Half-duplex voice: the text-to-speech agent reports when it is playing, and
the speech recognition agent ignores the microphone meanwhile.

The sound devices are replaced with fakes and the agents' samples are fed by
hand; the /walkie/speaking topic is real.
"""

import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rai_s2s.asr.agents.asr_agent as asr_module
import rai_s2s.tts.agents.tts_agent as tts_module
from pydub import AudioSegment
from rai_s2s.asr.agents.initialization import load_config as load_asr_config
from std_msgs.msg import Bool

from walkie_agent import half_duplex
from walkie_agent.half_duplex import (
    ECHO_TAIL,
    PAUSE_TIMEOUT,
    SPEAKER_STALL_TIMEOUT,
    SPEAKING_REPEAT,
    SPEAKING_TIMEOUT,
    SPEAKING_TOPIC,
    HalfDuplexSpeechRecognitionAgent,
    HalfDuplexTextToSpeechAgent,
)

CONFIG = Path(__file__).resolve().parent.parent / "config.toml"
# One microphone block, as voice.py configures it.
SAMPLE = np.zeros((1280, 1), dtype=np.int16)
# Speaker frames per callback: 20 ms at 24 kHz.
FRAMES = 480


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class FakeVAD:
    """Voice activity detector that reports whatever the test sets."""

    voice = False

    def __call__(self, audio, params):
        return self.voice, {}


class FakeSpeaker:
    def __init__(self, **kwargs):
        pass

    def get_audio_params(self, target):
        return 24000, FRAMES, 1

    def start_action(self, *args, **kwargs):
        pass


class FakeVoice:
    """Text-to-speech model that turns any text into 0.2 s of silence."""

    def get_speech(self, text):
        return AudioSegment.silent(duration=200, frame_rate=24000)


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(half_duplex, "monotonic", clock)
    return clock


@pytest.fixture
def asr(monkeypatch):
    monkeypatch.setattr(asr_module, "SoundDeviceConnector", lambda **kwargs: None)
    agent = HalfDuplexSpeechRecognitionAgent(
        None, f"test_asr_{random.randint(0, 1_000_000)}", None, FakeVAD()
    )
    # The /voice_commands messages it would send to the text-to-speech agent.
    agent.sent = []
    agent._send_ros2_message = lambda data, topic: agent.sent.append(data)
    yield agent
    agent.ros2_hri_connector.shutdown()
    agent.ros2_connector.shutdown()


@pytest.fixture
def make_tts(monkeypatch):
    """Start text-to-speech agents with a fake speaker, and stop them afterwards.

    With record=True, the agent's /walkie/speaking messages are collected in
    agent.published instead of being published.
    """
    monkeypatch.setattr(tts_module, "SoundDeviceConnector", FakeSpeaker)
    agents = []

    def make(record=False):
        agent = HalfDuplexTextToSpeechAgent(
            None, f"test_tts_{random.randint(0, 1_000_000)}", FakeVoice()
        )
        if record:
            agent.published = []
            agent.speaking_publisher = SimpleNamespace(
                publish=lambda msg: agent.published.append(msg.data)
            )
        agent.run()
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent.stop()
        agent.hri_ros2_connector.shutdown()
        agent.ros2_connector.shutdown()


@pytest.fixture
def tts(make_tts):
    return make_tts()


def start_sentence(agent, samples):
    agent.playback_data.playing = True
    agent.playback_data.data = np.zeros((samples, 1), dtype=np.int16)


def play(agent, frames=FRAMES):
    agent._speaker_callback(np.zeros((frames, 1), dtype=np.int16), frames, None, {})


def command(agent, data):
    """Deliver a /voice_commands message from speech recognition."""
    agent._on_command_message(SimpleNamespace(payload=SimpleNamespace(data=data)))


def to_human(agent, text):
    """Deliver a /to_human message, as the speak tool sends it."""
    agent._on_to_human_message(SimpleNamespace(text=text, communication_id="test"))


def test_asr_hears_the_microphone_while_walkie_is_quiet(asr):
    asr.vad.voice = True
    asr._on_new_sample(SAMPLE, {})
    assert asr.recording_started
    assert asr.sent == ["pause"]


def test_asr_ignores_the_microphone_while_walkie_speaks(asr, clock):
    asr._on_speaking(Bool(data=True))
    asr.vad.voice = True
    asr._on_new_sample(SAMPLE, {})
    assert not asr.recording_started
    assert asr.sent == []


def test_asr_listens_again_after_the_echo_tail(asr, clock):
    asr._on_speaking(Bool(data=True))
    clock.now += 0.2
    asr._on_speaking(Bool(data=False))
    asr.vad.voice = True

    clock.now += ECHO_TAIL - 0.01
    asr._on_new_sample(SAMPLE, {})
    assert asr.sent == []

    clock.now += 0.02
    asr._on_new_sample(SAMPLE, {})
    assert asr.sent == ["pause"]


def test_asr_listens_again_if_speaking_messages_stop(asr, clock):
    # A text-to-speech agent that dies mid-sentence must not leave ASR deaf.
    asr._on_speaking(Bool(data=True))
    clock.now += SPEAKING_TIMEOUT + 0.01
    asr.vad.voice = True
    asr._on_new_sample(SAMPLE, {})
    assert asr.sent == ["pause"]


def test_asr_discards_a_recording_when_walkie_starts_speaking(asr, clock):
    asr.vad.voice = True
    asr._on_new_sample(SAMPLE, {})
    assert asr.recording_started

    asr._on_speaking(Bool(data=True))
    asr._on_new_sample(SAMPLE, {})
    assert not asr.recording_started
    assert asr.transcription_threads == {}
    assert asr.sample_buffer == []  # the next recording starts without this audio
    assert asr.sent == ["pause"]  # no "stop", so nothing was transcribed

    # Once Walkie is done and the room is quiet, the paused playback resumes.
    clock.now += SPEAKING_TIMEOUT + 0.01
    asr.vad.voice = False
    asr._on_new_sample(SAMPLE, {})
    assert asr.sent == ["pause", "play"]


def test_tts_is_speaking_only_while_audio_plays(tts):
    assert not tts.speaking
    start_sentence(tts, samples=10 * FRAMES)
    play(tts)
    assert tts.speaking

    tts.playback_data.playing = False  # "pause" from ASR
    play(tts)
    assert not tts.speaking

    tts.playback_data.playing = True
    play(tts, frames=10 * FRAMES)  # plays the rest of the sentence
    assert not tts.speaking


def test_tts_holds_new_speech_while_asr_hears_a_person(tts):
    # RAI's text-to-speech agent resumes playback on every /to_human message,
    # so Walkie would talk over the person.
    command(tts, "pause")
    to_human(tts, "On my way")
    queue = tts.audio_queues[tts.current_transcription_id]
    assert wait_until(lambda: not queue.empty())
    play(tts)
    assert not tts.speaking
    assert not queue.empty()  # held, not skipped

    command(tts, "play")
    play(tts)
    assert tts.speaking


def test_tts_speaks_again_after_stop(tts):
    command(tts, "pause")
    command(tts, "stop")
    start_sentence(tts, samples=10 * FRAMES)
    play(tts)
    assert tts.speaking


def test_tts_hold_expires_if_asr_goes_quiet(tts, clock):
    # If ASR dies between "pause" and "stop", Walkie must not stay mute.
    command(tts, "pause")
    start_sentence(tts, samples=10 * FRAMES)
    play(tts)
    assert not tts.speaking

    clock.now += PAUSE_TIMEOUT + 0.01
    play(tts)
    assert tts.speaking


def test_pause_outlasts_the_asr_grace_period():
    # ASR sends "stop" up to silence_grace_period after its last "pause"; the
    # hold must not expire before that.
    cfg = load_asr_config(str(CONFIG))
    assert cfg.voice_activity_detection.silence_grace_period < PAUSE_TIMEOUT


def test_tts_reports_changes_at_once(make_tts, monkeypatch):
    # With a long repeat, only a change of state can wake the publisher.
    monkeypatch.setattr(half_duplex, "SPEAKING_REPEAT", 5.0)
    tts = make_tts(record=True)
    start_sentence(tts, samples=10 * FRAMES)
    play(tts)
    assert wait_until(lambda: tts.published == [True], timeout=1.0)
    time.sleep(0.2)
    assert tts.published == [True]  # not repeated before SPEAKING_REPEAT

    tts.playback_data.playing = False
    play(tts)
    assert wait_until(lambda: tts.published == [True, False], timeout=1.0)


def test_tts_reports_silence_if_the_speaker_stops(make_tts, clock):
    # When the audio stream dies, the callbacks stop and self.speaking stays
    # True; ASR must not stay deaf.
    tts = make_tts(record=True)
    start_sentence(tts, samples=10 * FRAMES)
    play(tts)
    assert wait_until(lambda: tts.published[-1:] == [True])

    clock.now += SPEAKER_STALL_TIMEOUT + 0.01
    assert wait_until(lambda: tts.published[-1:] == [False])
    assert tts.speaking


def test_tts_reports_silence_when_it_stops(make_tts):
    tts = make_tts(record=True)
    start_sentence(tts, samples=10 * FRAMES)
    play(tts)
    assert wait_until(lambda: tts.published[-1:] == [True])

    tts.stop()
    assert tts.published[-1] is False
    assert not tts.playback_data.playing


def test_asr_ignores_the_microphone_while_tts_plays(tts, asr, connector):
    # Over real ROS, the repeated True must keep the microphone ignored for a
    # whole sentence, not just after the first message.
    assert SPEAKING_REPEAT < SPEAKING_TIMEOUT
    heard = []
    connector.node.create_subscription(
        Bool, SPEAKING_TOPIC, lambda msg: heard.append(msg.data), 10
    )
    assert wait_until(lambda: tts.speaking_publisher.get_subscription_count() >= 2)

    start_sentence(tts, samples=200 * FRAMES)  # 4 s, longer than the loop below
    play(tts)
    assert wait_until(lambda: time.monotonic() < asr.ignore_until)
    asr.vad.voice = True
    end = time.monotonic() + SPEAKING_TIMEOUT + 2 * SPEAKING_REPEAT
    while time.monotonic() < end:
        play(tts)
        asr._on_new_sample(SAMPLE, {})
        time.sleep(0.02)
    assert asr.sent == []
    assert heard.count(True) >= 4  # repeated while playing

    play(tts, frames=200 * FRAMES)  # plays the rest of the sentence
    assert wait_until(lambda: heard[-1:] == [False])
    count = len(heard)
    time.sleep(2 * SPEAKING_REPEAT)
    assert len(heard) == count  # False is sent once, not repeated
