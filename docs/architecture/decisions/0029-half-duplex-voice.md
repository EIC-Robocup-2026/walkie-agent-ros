# ADR-0029: Stop listening while Walkie speaks

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

`voice-asr` and `voice-tts` are RAI's `SpeechRecognitionAgent` and
`TextToSpeechAgent`. In RAI, their only link is `/voice_commands`:

- When the microphone hears a voice, ASR sends `"pause"`.
- When the voice ends, ASR sends `"stop"`, which drops whatever TTS still
  had to say, and sends the transcription to `/from_human`.

ASR cannot tell a person's voice from Walkie's own. When the microphone
picks up the speaker, Walkie pauses itself a moment after it starts. It
then drops the rest of the sentence, and the agent receives Walkie's own
words as a new task. On a laptop this looked like "Walkie never speaks".

We reproduced this with TTS playing into a PipeWire null sink and ASR
listening to that sink: Walkie's sentence came back on `/from_human`.

**Decision**

Walkie does not listen while it speaks (half-duplex).

- `voice-tts` publishes `std_msgs/Bool` on `/walkie/speaking`. While audio
  plays, it sends `true` every 0.25 s. When playback finishes, pauses or
  stops, it sends `false` once.
- `voice-asr` ignores the microphone while the last `true` is less than
  1 s old, and for 0.5 s after a `false`. Ignored audio skips voice
  detection, so it sends no `"pause"` or `"stop"` and starts no recording.
- If the speaker's audio stream stops mid-sentence, `voice-tts` sends
  `false` within about 0.75 s, so ASR does not stay deaf.
- After a `"pause"`, `voice-tts` holds new speech until `"play"` or
  `"stop"`, or for at most 5 s. RAI's TTS resumes playback on every
  `/to_human` message, so without this, Walkie would talk over a person
  whose request ASR is recording, and the ignored microphone would lose it.
- A recording already in progress when Walkie starts speaking is discarded,
  not transcribed. Because of the hold, this happens only when both start
  at almost the same moment, and the recording may be the start of
  Walkie's own voice.
- Both are small subclasses of RAI's agents, in
  `src/walkie_agent/half_duplex.py`.

Rejected:

- **Acoustic echo cancellation** (PipeWire's echo-cancel module). It keeps
  interruptions working, but it is audio configuration on each computer,
  outside this repository. Its quality depends on the hardware, and it
  fails silently when it pairs the wrong microphone and speaker.
- **A wake word.** RAI's ASR sends `"pause"` on any voice, wake word or
  not, so Walkie would still pause itself.
- **A headset or push-to-talk.** Neither suits a robot in a RoboCup@Home
  arena.

**Consequences**

- Works with any microphone and speaker, with no audio setup.
- A person cannot interrupt Walkie by voice mid-sentence. They can speak in
  the gaps between sentences, which still sends `"stop"` and drops anything
  Walkie had queued. `/walkie/reset` works as before.
- Speech that starts while Walkie is talking is lost, so the person has to
  repeat it.
- If the agent speaks while a person is talking, the sentence waits. When
  the person finishes, ASR sends `"stop"` and the sentence is dropped, as
  in RAI.
- If `voice-tts` dies mid-sentence, ASR stays deaf for at most 1 s. If
  `voice-asr` dies while a person is talking, Walkie stays silent for at
  most 5 s (`PAUSE_TIMEOUT`, which must stay longer than ASR's
  `silence_grace_period`).
- The 0.5 s after playback covers audio still in the output buffers and
  the room's echo. A large, echoing room may need more (`ECHO_TAIL`).
- The subclasses use internals of rai_s2s (`_speaker_callback`,
  `_on_command_message`, `playback_data`, `_on_new_sample`,
  `recording_started`, `transcription_threads`). Recheck them when
  upgrading RAI ([ADR-0012](/architecture/decisions/0012-rai-git-tag-2-12-2.md)).

**See also:** [Talk to Walkie by voice](/setup/07-voice.md)
<!-- --8<-- [end:body] -->
