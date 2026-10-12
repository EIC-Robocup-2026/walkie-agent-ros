"""The voice settings in config.toml, checked with rai_s2s's own config loaders.

The agents themselves need a microphone and a speaker, so they are tested by
hand (see docs/setup/07-voice.md).
"""

from pathlib import Path

from rai_s2s.asr.agents.initialization import load_config as load_asr_config
from rai_s2s.tts.agents.initialization import TTS_MODELS
from rai_s2s.tts.agents.initialization import load_config as load_tts_config

CONFIG = Path(__file__).resolve().parent.parent / "config.toml"


def test_asr_config_is_valid():
    # load_config raises ValueError for an unknown transcription model.
    cfg = load_asr_config(str(CONFIG))
    assert cfg.voice_activity_detection.model_name == "SileroVAD"


def test_tts_config_is_valid():
    cfg = load_tts_config(str(CONFIG))
    assert cfg.text_to_speech.model_type in TTS_MODELS
