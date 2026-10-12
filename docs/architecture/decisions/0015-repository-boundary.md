# ADR-0015: Keep only agent-side code in this repository

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

Walkie's software is split across repositories.
[walkie-ros](https://github.com/EIC-Robocup-2026/walkie-ros) runs on the
robot: bringup, drivers, ros2_control, Nav2, MoveIt and the bimanual arm
commander. This repository holds the LLM agent. If robot nodes were copied
here, the two copies would drift apart.

**Decision**

walkie-agent-ros holds only agent-side code:

- the agent, its tools and their configuration (named locations, and later
  named arm poses),
- voice, whoami and tracing,
- the documentation, and
- fakes of robot nodes, used only for tests and development
  (`src/walkie_agent/sim/`).

Navigation, manipulation, perception and drivers live in walkie-ros.

**Consequences**

- A new robot ability starts as a walkie-ros change that exposes an
  interface. Then an agent tool calls that interface.
- The agent depends on walkie-ros interfaces, which must match the robot's
  deployed version
  ([ADR-0018](/architecture/decisions/0018-pin-robot-interfaces.md)).
- The fakes must copy walkie-ros behavior closely, quirks included, or tests
  pass against behavior the robot does not have.
- Configuration the LLM chooses from, such as place names and pose names,
  stays here, even though the robot executes it.

**See also:** [The Agent: what lives where](/architecture/agent.md#what-lives-where)
<!-- --8<-- [end:body] -->
