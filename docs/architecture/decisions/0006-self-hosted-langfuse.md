# ADR-0006: Trace agent runs with self-hosted Langfuse, off by default

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

To debug the agent we need to see each prompt, LLM answer and tool call. RAI
can send traces to Langfuse or LangSmith. The cloud versions need an account
and internet, and they send every prompt off-site.

**Decision**

We run Langfuse server v3 in Docker on the developer's computer
(`docker/langfuse/compose.yml`, `pixi run langfuse-up`). All ports listen on
`127.0.0.1` only, and `scripts/langfuse_setup.py` generates the passwords and
API keys. Each conversation is one Langfuse session, so a
[reset](/architecture/decisions/0008-reset-topic.md) starts a new session.

Tracing is opt-in: `use_langfuse = false` in Git, and the tests turn tracing
off whatever `config.toml` says.

**Consequences**

- Traces stay on the computer and work without internet.
- The server needs Docker, about 2.5 GB of RAM while running and about 6 GB
  of disk for the images.
- We pin the server to v3 because `rai-core` requires the Langfuse SDK below
  version 4, and the v4 server shows traces from older SDKs with a delay.
  Upgrading rai-core past that pin means revisiting the server version.
- A fresh clone runs without Langfuse keys. To trace, set
  `use_langfuse = true` locally, and do not commit that change.

**See also:** [Trace with local Langfuse](/setup/04-langfuse.md)
<!-- --8<-- [end:body] -->
