"""Runs one of RAI's voice agents (rai_s2s) next to the Walkie agent.

    python -m walkie_agent.voice asr   # microphone -> /from_human
    python -m walkie_agent.voice tts   # /to_human -> speaker

Settings come from the [asr] and [tts] sections of config.toml in the working
directory. The two agents coordinate over /voice_commands (std_msgs/String):
while a person is speaking, ASR pauses playback, and once their speech has been
transcribed it sends "stop", which drops whatever TTS had left to say.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rai.agents import AgentRunner
from rai.agents.base import BaseAgent
from rai.communication.ros2 import ROS2Context

# Silero VAD accepts 8 or 16 kHz, and Whisper expects 16 kHz.
SAMPLE_RATE = 16000
# Microphone samples per callback (80 ms at 16 kHz), as in RAI's examples.
BLOCK_SIZE = 1280


def build_asr_agent(config_path: Path) -> BaseAgent:
    """Build RAI's SpeechRecognitionAgent from the [asr] section of config.toml.

    SpeechRecognitionAgent.from_config() is not used because in rai_s2s 1.0.x
    it expects model names such as "LocalWhisper (Free)", which its own config
    loader rejects, so it fails for every valid config.
    """
    from rai_s2s.asr import SpeechRecognitionAgent
    from rai_s2s.asr.agents.initialization import load_config
    from rai_s2s.asr.models import (
        FasterWhisper,
        LocalWhisper,
        OpenAIWhisper,
        OpenWakeWord,
        SileroVAD,
    )
    from rai_s2s.sound_device import SoundDeviceConfig

    cfg = load_config(str(config_path))
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
    agent = SpeechRecognitionAgent(
        microphone,
        "walkie_asr",
        model,
        vad,
        grace_period=cfg.voice_activity_detection.silence_grace_period,
    )
    if cfg.wakeword.is_used:
        agent.add_detection_model(
            OpenWakeWord(cfg.wakeword.model_name, cfg.wakeword.threshold)
        )
    return agent


def build_tts_agent(config_path: Path) -> BaseAgent:
    """Build RAI's TextToSpeechAgent from the [tts] section of config.toml."""
    from rai_s2s.tts import TextToSpeechAgent

    return TextToSpeechAgent.from_config(str(config_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", choices=["asr", "tts"])
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args()

    # ElevenLabs and OpenAI Whisper read their API keys from the environment.
    load_dotenv()

    build = build_asr_agent if args.agent == "asr" else build_tts_agent
    with ROS2Context():
        AgentRunner([build(args.config)]).run_and_wait_for_shutdown()


if __name__ == "__main__":
    main()
