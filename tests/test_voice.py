"""The voice settings in config.toml, checked with rai_s2s's own config loaders,
and the audio device checks that run before the voice agents start.

The agents themselves need a microphone and a speaker, so they are tested by
hand (see docs/setup/07-voice.md).
"""

import os
import subprocess
from pathlib import Path

import pytest
import sounddevice
from rai_s2s.asr.agents.initialization import load_config as load_asr_config
from rai_s2s.tts.agents.initialization import TTS_MODELS
from rai_s2s.tts.agents.initialization import load_config as load_tts_config

from walkie_agent.voice import check_audio_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "config.toml"
ACTIVATE = PROJECT_ROOT / "scripts" / "activate.sh"

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


def test_asr_config_is_valid():
    # load_config raises ValueError for an unknown transcription model.
    cfg = load_asr_config(str(CONFIG))
    assert cfg.voice_activity_detection.model_name == "SileroVAD"


def test_tts_config_is_valid():
    cfg = load_tts_config(str(CONFIG))
    assert cfg.text_to_speech.model_type in TTS_MODELS


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
