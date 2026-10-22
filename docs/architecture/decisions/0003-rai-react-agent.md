# ADR-0003: Build the agent on RAI's ReActAgent

<!-- --8<-- [start:body] -->
**Status:** Accepted, partly superseded by
[ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md) ·
**Date:** 2026-10-12

**Context**

We need an LLM agent that receives tasks, calls tools that act on ROS 2, and
reports back. We could write our own LangGraph loop on top of `rclpy`, or use
[RAI](https://github.com/RobotecAI/rai) from Robotec.AI. RAI provides ROS 2
connectors, a message type for human-robot interaction (`HRIMessage`), a
ReAct agent, ready-made ROS 2 tools, tracing, speech agents (`rai_s2s`) and
robot self-description (`rai_whoami`).

**Decision**

We build on `rai-core` 2.12 (`>=2.12.1,<2.13`). `WalkieAgent` subclasses
`ReActAgent`. Tasks arrive as `rai_interfaces/msg/HRIMessage` through
`ROS2HRIConnector`. Tools are `BaseROS2Tool` subclasses or RAI's premade
tools. RAI's `config.toml` selects the LLM.

**Consequences**

- We write far less code: the agent loop, connectors, tracing hooks, speech
  and whoami come from RAI.
- `WalkieAgent.reset()` relies on `ReActAgent` internals (`_run_agent`, the
  message buffer and the interrupt event). They must be rechecked on every
  rai-core upgrade, which is why the version is capped below 2.13.
- RAI runs one tool at a time. All tool calls from one LLM message run inside
  one graph step, and the interrupt is checked only between steps. A reset
  therefore cannot stop a tool that is already running
  ([ADR-0023](/architecture/decisions/0023-motion-supervisor.md) addresses
  this for motion).
- RAI's tools do not all fit Walkie. `NavigateToPoseBlockingTool` waits
  forever ([ADR-0010](/architecture/decisions/0010-bounded-navigation-tool.md)),
  and its manipulation tools call a service walkie-ros does not provide
  ([ADR-0021](/architecture/decisions/0021-arm-control-through-commander.md)).
- RAI's own pins (`numpy<2`, the Langfuse SDK below 4) constrain the whole
  environment.

**See also:** [Install RAI](/setup/03-ros-and-rai.md)
<!-- --8<-- [end:body] -->
