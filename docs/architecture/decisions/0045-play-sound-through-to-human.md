# ADR-0045: Play sounds through the audios field of /to_human

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

The self-evolving agent must play short sounds, such as a real cat's meow,
as well as speak ([Competition Mode](/architecture/competition.md)). Today
the only way to the speaker is text:

- `SpeakTool` (`src/walkie_agent/tools/speech.py`) publishes
  `HRIMessage(text=..., communication_id=<speech id>, seq_end=True)` on
  `/to_human`. It keeps one speech ID and replaces it with a new
  `uuid4().hex` when it sees `"stop"` on `/voice_commands`
  ([ADR-0007](/architecture/decisions/0007-speak-tool-and-console.md)).
- `rai_interfaces/HRIMessage` already has an `audios` field: a list of
  `AudioMessage` with `int16[] audio`, `uint16 sample_rate` and
  `uint16 channels`.
- RAI's connector already decodes that field. `ROS2HRIMessage.from_ros2`
  turns each `AudioMessage` into a pydub `AudioSegment` at the clip's own
  sample rate and channel count, 16-bit, before any callback runs.
- RAI's `TextToSpeechAgent._on_to_human_message` locks onto the first
  `communication_id` it has not seen before, when `current_speech_id` is
  empty. It remembers the last 64 IDs in `remembered_speech_ids`, and
  ignores messages under any other ID. It puts `msg.text` into
  `text_queues[current_transcription_id]` and sets
  `playback_data.playing`. It ignores `msg.audios`. A message with empty
  text still queues an empty request for the TTS model.
- `_transcription_thread` takes each item, calls `model.get_speech(item)`
  and puts the resulting `AudioSegment` into `audio_queues`.
  `_speaker_callback` plays from there as mono samples, at the speaker
  stream's sample rate.
- `"stop"` clears `current_speech_id`, creates new queues under a new ID
  and drops the old ones.
- `HalfDuplexTextToSpeechAgent` (`src/walkie_agent/half_duplex.py`)
  already overrides `__init__`, `_setup_ros2_connector`,
  `_speaker_callback`, `_on_command_message`, `run` and `stop`, and
  publishes `/walkie/speaking` while audio plays
  ([ADR-0029](/architecture/decisions/0029-half-duplex-voice.md)). It does
  not override `_on_to_human_message` or `_transcription_thread`.

**Decision**

- `play_sound` and `fetch_sound` publish
  `HRIMessage(audios=[AudioMessage(audio, sample_rate, channels=1)], communication_id=<speech id>, seq_end=True)`
  on `/to_human`, with empty text.
- `speak`, `play_sound` and `fetch_sound` share one speech-ID holder. A
  `"stop"` replaces the ID once, for all three. With separate holders,
  whichever of speech or sound publishes first after a stop would lock the
  TTS onto its ID, and messages under the other ID would be dropped until
  the next stop.
- `HalfDuplexTextToSpeechAgent` overrides `_on_to_human_message` and
  `_transcription_thread`:
    - Each incoming `AudioSegment` in `msg.audios` is converted with
      `set_frame_rate(<speaker stream's rate>)`, `set_channels(1)` and
      `set_sample_width(2)`. It goes into `text_queues` as an item, after
      the message's text.
    - The transcription thread passes an `AudioSegment` item to
      `audio_queues` unchanged and in order. Only strings go to the TTS
      model.
    - Empty text is skipped, so an audio-only message queues no TTS
      request.
- Because a sound travels through the same queues as speech, it plays
  after any earlier sentence and inherits half-duplex, `"stop"` and
  `"pause"`.
- The sender enforces the caps before publishing: at most 5 s per clip at
  Competition (10 s in sim), peak at most −3 dBFS after loudness
  normalization, mono, and a sample rate of at most 48 kHz. Clips also
  count against the speech gate's audio budget.
- `play_sound` never takes a URL. It plays a name from the curated pack in
  `src/walkie_agent/assets/sounds/` (new; `SOUNDS.toml` records each
  file's source and license) or a session asset.

Rejected:

- **Putting audio straight into `audio_queues`.** The sound would play
  before an earlier sentence that is still being synthesized, so "Here is
  a cat" would come after the meow.
- **Playing out of band** with `sounddevice`, `pw-play` or `aplay`.
  `/walkie/speaking` would stay false, so ASR would hear the meow and send
  it to the agent as a new request. `"stop"` and `"pause"` would not reach
  it either.

**Consequences**

- Amends ADR-0007: `speak` is no longer the only publisher on `/to_human`,
  and its speech ID is shared with `play_sound` and `fetch_sound`. Plain
  replies still go to `/walkie/console`.
- Amends ADR-0029: the TTS subclass now also depends on
  `_on_to_human_message`, `_transcription_thread`, `text_queues`,
  `audio_queues`, `current_transcription_id`, `current_speech_id`,
  `remembered_speech_ids` and `ROS2HRIMessage.from_ros2`. `rai_s2s` stays
  at Git tag 2.12.2
  ([ADR-0012](/architecture/decisions/0012-rai-git-tag-2-12-2.md)) and
  `rai-core` below 2.13 (`pixi.toml`). An upgrade of either must recheck
  all of them.
- ASR stays deaf while a sound plays, as it does during speech.
- `sample_rate` is a `uint16`, so a rate above 65,535 Hz does not fit.
  Whether such a value is rejected or wraps around is untested. Resampling
  to at most 48 kHz before publishing avoids the question.
- A 5 s clip at 48 kHz, mono, 16-bit is about 480 KB per message over
  Zenoh.
- Tests: conversion from bytes to `AudioSegment` with no audio device; the
  caps; speak-then-play ordering with a slow fake TTS model; an audio-only
  message queues no TTS request; and a silent integration test that plays
  into a PipeWire null sink.

**See also:** [Talk to Walkie by voice](/setup/07-voice.md),
[Competition Mode](/architecture/competition.md)
<!-- --8<-- [end:body] -->
