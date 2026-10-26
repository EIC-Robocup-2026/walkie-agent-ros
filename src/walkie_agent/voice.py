"""Runs one of RAI's voice agents (rai_s2s) next to the Walkie agent.

    python -m walkie_agent.voice asr   # microphone -> /from_human
    python -m walkie_agent.voice tts   # /to_human -> speaker

Settings come from the [asr] and [tts] sections of config.toml in the working
directory. The two agents coordinate over two topics:

- /walkie/speaking (std_msgs/Bool): TTS reports whether it is playing, and ASR
  ignores the microphone meanwhile, so Walkie does not hear itself (see
  walkie_agent.half_duplex).
- /voice_commands (std_msgs/String): while a person is speaking, ASR pauses
  playback and TTS holds anything new, and once their speech has been
  transcribed ASR sends "stop", which drops whatever TTS had left to say.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    # Only for the annotations. rai.communication.ros2 loads rclpy, so every
    # rai import waits until an agent is built or run, and this module, with
    # AsrOptions, imports without ROS.
    from rai.agents.base import BaseAgent

# Silero VAD accepts 8 or 16 kHz, and Whisper expects 16 kHz.
SAMPLE_RATE = 16000
# Microphone samples per callback (80 ms at 16 kHz), as in RAI's examples.
BLOCK_SIZE = 1280
# Devices that exist only through the system's PipeWire or PulseAudio ALSA plugin.
SOUND_SERVER_DEVICES = ("default", "pipewire", "pulse")


@dataclass(frozen=True)
class AsrOptions:
    """The command line of `python -m walkie_agent.voice asr`, as the agent takes it.

    build_asr_agent passes it whole to the speech recognition agent. Every
    voice-side loader reads its settings from config_path, never from a
    literal path.
    """

    config_path: Path  # the TOML file with the [asr] section, from --config

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AsrOptions":
        """The options from the parsed command line."""
        return cls(config_path=args.config)


def check_audio_device(name: str, direction: str) -> None:
    """Exit with a readable message if `name` is not a usable audio device.

    `direction` is "input" or "output". rai_s2s picks the first device whose
    name matches exactly, and for an unknown name it fails with an
    AttributeError raised inside its own error handling.
    """
    import sounddevice as sd

    channels = f"max_{direction}_channels"
    devices = list(sd.query_devices())
    match = next((d for d in devices if d["name"] == name), None)
    if match is not None and match[channels] > 0:
        return

    usable = [d["name"] for d in devices if d[channels] > 0]
    lines = [
        f"No audio {direction} device named {name!r}. Available {direction} devices:",
        *(f"  {n}" for n in usable or ["(none)"]),
        "Set the device name in config.toml (see docs/setup/07-voice.md).",
    ]
    if "(hw:" in name:
        lines.append(
            "An (hw:N,M) device opens sound card N directly. It is missing, or has "
            "no channels, while PipeWire or another program is using the card, and "
            'card numbers can change between boots. "default" avoids both.'
        )
    if name in SOUND_SERVER_DEVICES:
        lines.append(
            f"{name!r} comes from a system ALSA plugin for the sound server "
            "(PipeWire or PulseAudio): check that the sound server is running and "
            "that ALSA_PLUGIN_DIR points to the system plugins (set by "
            "scripts/activate.sh)."
        )
        # Both servers listen on a socket in the user's runtime directory.
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        sockets = ("pipewire-0", "pulse/native")
        if not runtime_dir or not any(Path(runtime_dir, s).exists() for s in sockets):
            where = f"({runtime_dir}) has" if runtime_dir else "is not set, so there is"
            lines.append(
                f"XDG_RUNTIME_DIR {where} no sound server socket. Run this as the "
                "desktop user from their login session, not as root, through sudo, "
                "from cron or as a system service."
            )
    raise SystemExit("\n".join(lines))


def build_asr_agent(options: AsrOptions) -> "BaseAgent":
    """Build the speech recognition agent from the [asr] section of options.config_path.

    SpeechRecognitionAgent.from_config() is not used because in rai_s2s 1.0.x
    it expects model names such as "LocalWhisper (Free)", which its own config
    loader rejects, so it fails for every valid config.
    """
    from rai_s2s.asr.agents.initialization import load_config
    from rai_s2s.asr.models import (
        FasterWhisper,
        LocalWhisper,
        OpenAIWhisper,
        OpenWakeWord,
        SileroVAD,
    )
    from rai_s2s.sound_device import SoundDeviceConfig

    from walkie_agent.half_duplex import HalfDuplexSpeechRecognitionAgent

    cfg = load_config(str(options.config_path))
    check_audio_device(cfg.microphone.device_name, "input")
    microphone = SoundDeviceConfig(
        stream=True,
        channels=1,
        device_name=cfg.microphone.device_name,
        block_size=BLOCK_SIZE,
        consumer_sampling_rate=SAMPLE_RATE,
        dtype="int16",
        device_number=None,
        is_input=True,
        is_output=False,
    )

    transcribe = cfg.transcribe
    match transcribe.model_type:
        case "LocalWhisper":
            model = LocalWhisper(
                transcribe.model_name, SAMPLE_RATE, language=transcribe.language
            )
        case "FasterWhisper":
            model = FasterWhisper(
                transcribe.model_name, SAMPLE_RATE, language=transcribe.language
            )
        case "OpenAI":
            model = OpenAIWhisper(transcribe.model_name, SAMPLE_RATE, transcribe.language)
        case _:
            raise ValueError(f"Unknown [asr] transcription_model: {transcribe.model_type}")

    vad = SileroVAD(SAMPLE_RATE, cfg.voice_activity_detection.threshold)
    agent = HalfDuplexSpeechRecognitionAgent(
        microphone,
        "walkie_asr",
        model,
        vad,
        grace_period=cfg.voice_activity_detection.silence_grace_period,
        options=options,
    )
    if cfg.wakeword.is_used:
        agent.add_detection_model(
            OpenWakeWord(cfg.wakeword.model_name, cfg.wakeword.threshold)
        )
    return agent


def build_tts_agent(config_path: Path) -> "BaseAgent":
    """Build the text-to-speech agent from the [tts] section of config.toml."""
    from rai_s2s.tts.agents.initialization import load_config

    from walkie_agent.half_duplex import HalfDuplexTextToSpeechAgent

    check_audio_device(load_config(str(config_path)).speaker.device_name, "output")
    return HalfDuplexTextToSpeechAgent.from_config(str(config_path))


def _add_asr_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags that only the asr agent reads. AsrOptions holds their values.

    There are none yet.
    """


def main() -> None:
    from rai.agents import AgentRunner
    from rai.communication.ros2 import ROS2Context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", choices=["asr", "tts"])
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    _add_asr_arguments(parser)
    args = parser.parse_args()

    # ElevenLabs and OpenAI Whisper read their API keys from the environment.
    load_dotenv()

    with ROS2Context():
        if args.agent == "asr":
            agent = build_asr_agent(AsrOptions.from_args(args))
        else:
            agent = build_tts_agent(args.config)
        AgentRunner([agent]).run_and_wait_for_shutdown()


if __name__ == "__main__":
    main()
