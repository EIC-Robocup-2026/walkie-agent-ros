# ADR-0068: Describe the robot in one static file

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-05

**Context**

[ADR-0011](/architecture/decisions/0011-whoami-identity.md) generated Walkie's identity
with `rai_whoami`, and searched its documents through a FAISS index and the
`search_robot_docs` tool. The build called an LLM, its output needed a human review, and
each search needed the network for an embedding. An LLM-written description also said
what Walkie has, not what the agent can do today.

**Decision**

`robot.md`, at the repository root, is written by hand and reviewed like code.

- `main.py` appends it once to the system prompt. `--robot-description` points at another
  file.
- It says what the agent can do now, and what it cannot. The camera and the arms are part
  of the robot but are not connected, so Walkie must not offer those things.
- There is no generated identity, no FAISS index, no embedding call and no docs tool.
  `pixi run whoami-build` and the `rai-whoami` dependency are gone.

To change what Walkie says about itself, edit `robot.md` and restart the agent. Update it
in the same change whenever a capability becomes available.

This supersedes ADR-0011. It partly supersedes
[ADR-0012](/architecture/decisions/0012-rai-git-tag-2-12-2.md), for its `rai_whoami` part
only: `rai_s2s` still installs from RAI's Git tag 2.12.2. It changes where the identity
document lives for [ADR-0020](/architecture/decisions/0020-no-urdf-in-agent.md), and the
agent still does not load the URDF.

**Consequences**

- Startup needs no LLM, no embeddings endpoint and no generated files.
- The agent cannot look up a detail that is not in `robot.md`. A longer file costs tokens
  on every model call, so it stays short.
- `robot.md` can drift from the code. Nothing checks its claims.
- Evidence that ran: unit tests that the description is appended once and that the old
  flag is gone. No real model has read it.
- `embeddings_model` stays in `config.toml` to keep the shape of RAI's own config. Nothing
  in this repository reads it.

**See also:** [The Agent](/architecture/agent.md#what-walkie-knows-about-itself),
[Talk to Walkie by voice](/setup/07-voice.md)
<!-- --8<-- [end:body] -->
