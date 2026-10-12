# ADR-0007: Speak only through the speak tool; plain replies go to the console

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

RAI's ReActAgent sends its final reply to every output connector. If that
output were `/to_human`, the text-to-speech node would read out every reply,
including notes meant for the operator and any markdown the LLM writes.

RAI's `TextToSpeechAgent` also locks onto one `communication_id`. It plays
only messages with that ID, and accepts a new ID only after a `"stop"` on
`/voice_commands`, which the speech recognition agent sends when a person
has finished speaking.

**Decision**

- The agent talks to people only through the `speak` tool, which publishes
  `HRIMessage` on `/to_human`.
- The agent's plain-text replies go to `/walkie/console`, which only the
  operator reads.
- `SpeakTool` creates its publisher at startup, so discovery is done before
  the first sentence. It waits up to 2 s for a subscriber, and keeps one
  speech ID that it replaces whenever it sees `"stop"` on `/voice_commands`.

**Consequences**

- The LLM decides what is said out loud. The system prompt tells it to use
  `speak` for everything meant for a person, in short sentences without
  markdown.
- If no text-to-speech node is running, `speak` reports "Nothing was said"
  instead of pretending it spoke.
- The ID handling mirrors `TextToSpeechAgent` in rai_s2s 2.12. Recheck it
  when upgrading RAI.

**See also:** [Getting Started: topics](/getting-started.md),
[Talk to Walkie by voice](/setup/07-voice.md)
<!-- --8<-- [end:body] -->
