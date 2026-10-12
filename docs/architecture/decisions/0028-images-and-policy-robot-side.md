# ADR-0028: Keep camera images and policy inference out of the agent

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

Camera streams are large: a policy such as LeRobot `folding_latest` uses a
480x640 base camera and two 720x1280 wrist cameras. Policy code brings its
own dependencies, while the agent's environment is fixed to torch 2.3.1 and
`numpy<2`. Every image sent to the LLM is billed as input tokens.

**Decision**

- Camera streams and policy inference run in robot-side nodes, not in the
  agent's process.
- The agent never subscribes to raw image streams.
- If the LLM needs to see something, a robot-side service returns one
  compressed snapshot on request.

**Consequences**

- The agent's process stays small, and its environment never has to fit a
  policy's dependencies.
- Images reach the LLM only when it asks for one.
- Perception and policy code lives in walkie-ros or another robot-side
  repository
  ([ADR-0015](/architecture/decisions/0015-repository-boundary.md)).
- Whether the robot PC's GPU can run a policy next to everything else is
  still open.
<!-- --8<-- [end:body] -->
