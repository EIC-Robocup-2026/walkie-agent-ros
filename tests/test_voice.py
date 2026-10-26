"""The voice settings in config.toml, checked with rai_s2s's own config loaders,
the command line of walkie_agent.voice, and the audio device checks that run
before the voice agents start.

The agents themselves need a microphone and a speaker, so they are tested by
hand (see docs/setup/07-voice.md). How they take turns is tested with fake
sound devices in test_half_duplex.py.
"""

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import rai_s2s.asr.models as asr_models
import sounddevice
from rai_s2s.asr.agents.initialization import load_config as load_asr_config
from rai_s2s.tts.agents.initialization import TTS_MODELS
from rai_s2s.tts.agents.initialization import load_config as load_tts_config

from walkie_agent import half_duplex, voice
from walkie_agent.voice import SAMPLE_RATE, AsrOptions, check_audio_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
CONFIG = PROJECT_ROOT / "config.toml"
ACTIVATE = PROJECT_ROOT / "scripts" / "activate.sh"

# An [asr] section that differs from config.toml's in the model, the
# microphone and the voice detection values.
OTHER_ASR_CONFIG = """\
[asr]
recording_device_name = "USB Mic"
transcription_model = "FasterWhisper"
language = "en"
vad_model = "SileroVAD"
silence_grace_period = 0.8
vad_threshold = 0.4
use_wake_word = false
wake_word_model = ""
wake_word_threshold = 0.5
wake_word_model_name = ""
transcription_model_name = "base.en"
"""

# The directories scripts/activate.sh looks for, in order.
SYSTEM_ALSA_PLUGIN_DIRS = (
    "/usr/lib/x86_64-linux-gnu/alsa-lib",
    "/usr/lib64/alsa-lib",
    "/usr/lib/alsa-lib",
)

FAKE_DEVICES = [
    {"name": "HDA NVidia: HDMI 0 (hw:0,3)", "max_input_channels": 0, "max_output_channels": 8},
    {"name": "USB Mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "default", "max_input_channels": 64, "max_output_channels": 64},
]


@pytest.fixture
def fake_devices(monkeypatch):
    monkeypatch.setattr(sounddevice, "query_devices", lambda: FAKE_DEVICES)


@pytest.fixture
def run_voice(monkeypatch):
    """Run walkie_agent.voice's main() with a command line, without ROS or .env.

    Returns the agents it would have run.
    """
    ran = []

    class FakeRunner:
        def __init__(self, agents):
            self.agents = agents

        def run_and_wait_for_shutdown(self):
            ran.extend(self.agents)

    # main() imports both when it starts, so the fakes replace them at the source.
    monkeypatch.setattr("rai.agents.AgentRunner", FakeRunner)
    monkeypatch.setattr("rai.communication.ros2.ROS2Context", contextlib.nullcontext)
    monkeypatch.setattr(voice, "load_dotenv", lambda: None)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["walkie_agent.voice", *argv])
        voice.main()
        return ran

    return run


def test_asr_config_is_valid():
    # load_config raises ValueError for an unknown transcription model.
    cfg = load_asr_config(str(CONFIG))
    assert cfg.voice_activity_detection.model_name == "SileroVAD"


def test_tts_config_is_valid():
    cfg = load_tts_config(str(CONFIG))
    assert cfg.text_to_speech.model_type in TTS_MODELS


@pytest.mark.parametrize(
    ("argv", "built"),
    [
        (["asr"], ("asr", AsrOptions(config_path=Path("config.toml")))),
        (["asr", "--config", "other.toml"], ("asr", AsrOptions(config_path=Path("other.toml")))),
        (["tts"], ("tts", Path("config.toml"))),
        (["tts", "--config", "other.toml"], ("tts", Path("other.toml"))),
    ],
)
def test_the_command_line_picks_the_builder(run_voice, monkeypatch, argv, built):
    calls = []

    def builder(name):
        def build(arg):
            calls.append((name, arg))
            return f"{name} agent"

        return build

    monkeypatch.setattr(voice, "build_asr_agent", builder("asr"))
    monkeypatch.setattr(voice, "build_tts_agent", builder("tts"))

    assert run_voice(*argv) == [f"{built[0]} agent"]
    assert calls == [built]


def test_the_config_flag_reaches_the_asr_model(run_voice, monkeypatch, tmp_path, fake_devices):
    (tmp_path / "other.toml").write_text(OTHER_ASR_CONFIG)
    monkeypatch.chdir(tmp_path)
    models = []

    def fake_model(name):
        def make(*args, **kwargs):
            models.append((name, args, kwargs))
            return name

        return make

    # Every model is faked, so a wrong match records the wrong name instead of
    # loading a real one.
    for name in ("LocalWhisper", "FasterWhisper", "OpenAIWhisper", "SileroVAD"):
        monkeypatch.setattr(asr_models, name, fake_model(name))
    # The real agent opens the microphone.
    monkeypatch.setattr(
        half_duplex,
        "HalfDuplexSpeechRecognitionAgent",
        lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )

    [agent] = run_voice("asr", "--config", "other.toml")

    assert models == [
        ("FasterWhisper", ("base.en", SAMPLE_RATE), {"language": "en"}),
        ("SileroVAD", (SAMPLE_RATE, 0.4), {}),
    ]
    microphone, _, model, vad = agent.args
    assert microphone.device_name == "USB Mic"
    assert (model, vad) == ("FasterWhisper", "SileroVAD")
    assert agent.kwargs == {
        "grace_period": 0.8,
        "options": AsrOptions(config_path=Path("other.toml")),
    }


def test_importing_voice_loads_neither_the_agent_nor_ros():
    # In a fresh interpreter, since this one has rclpy loaded already. Tests
    # read AsrOptions without ROS, and voice-asr loads none of the agent's
    # modules.
    code = (
        "import json, sys\n"
        "import walkie_agent.voice as module\n"
        "roots = ('walkie_agent.tuning', 'walkie_agent.agent', 'rclpy')\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if any(name == root or name.startswith(root + '.') for root in roots)\n"
        ")\n"
        "print(json.dumps({'file': module.__file__, 'loaded': loaded}))\n"
    )
    path = os.pathsep.join(part for part in (str(SRC), os.environ.get("PYTHONPATH")) if part)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    found = json.loads(result.stdout.strip().splitlines()[-1])

    assert found["loaded"] == []
    assert Path(found["file"]).resolve().is_relative_to(SRC.resolve())


def test_known_devices_pass(fake_devices):
    check_audio_device("USB Mic", "input")
    check_audio_device("default", "output")


def test_unknown_device_lists_the_usable_ones(fake_devices):
    with pytest.raises(SystemExit) as exc:
        check_audio_device("Missing Mic", "input")
    message = str(exc.value)
    assert "No audio input device named 'Missing Mic'" in message
    assert "  USB Mic" in message
    assert "HDMI" not in message  # output-only devices are not offered as inputs


def test_device_without_channels_in_that_direction_fails(fake_devices):
    with pytest.raises(SystemExit, match="No audio output device named 'USB Mic'"):
        check_audio_device("USB Mic", "output")


def test_missing_sound_server_device_points_to_alsa_plugins(monkeypatch, tmp_path):
    monkeypatch.setattr(sounddevice, "query_devices", lambda: FAKE_DEVICES[:2])
    (tmp_path / "pipewire-0").touch()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        check_audio_device("default", "input")
    message = str(exc.value)
    assert "ALSA_PLUGIN_DIR" in message
    assert "XDG_RUNTIME_DIR" not in message


@pytest.mark.parametrize("runtime_dir", [None, "empty"])
def test_missing_sound_server_socket_points_to_the_user_session(
    monkeypatch, tmp_path, runtime_dir
):
    # Under sudo, cron or a system service, the user's sound server is out of reach.
    monkeypatch.setattr(sounddevice, "query_devices", lambda: FAKE_DEVICES[:2])
    if runtime_dir is None:
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    else:
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="no sound server socket"):
        check_audio_device("default", "input")


def test_missing_hw_device_explains_both_causes(fake_devices):
    with pytest.raises(SystemExit) as exc:
        check_audio_device("HDA NVidia: HDMI 0 (hw:1,3)", "output")
    message = str(exc.value)
    assert "while PipeWire or another program is using the card" in message
    assert "card numbers can change" in message


def source_activation(**env_changes):
    """Return ALSA_PLUGIN_DIR after sourcing scripts/activate.sh in a clean shell."""
    env = {k: v for k, v in os.environ.items() if k != "ALSA_PLUGIN_DIR"}
    env["PIXI_PROJECT_ROOT"] = "/nonexistent"  # skip the ROS overlay
    env.update(env_changes)
    result = subprocess.run(
        ["bash", "-c", 'source "$1" && echo "${ALSA_PLUGIN_DIR:-}"', "bash", str(ACTIVATE)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.skipif(
    not any(Path(d).is_dir() for d in SYSTEM_ALSA_PLUGIN_DIRS),
    reason="no system ALSA plugin directory on this computer",
)
def test_activation_points_alsa_at_the_system_plugins():
    # Without it, the "default" and "pipewire" devices do not exist in Pixi.
    expected = next(d for d in SYSTEM_ALSA_PLUGIN_DIRS if Path(d).is_dir())
    assert source_activation() == expected


def test_activation_keeps_a_user_alsa_plugin_dir():
    assert source_activation(ALSA_PLUGIN_DIR="/opt/alsa-lib") == "/opt/alsa-lib"
