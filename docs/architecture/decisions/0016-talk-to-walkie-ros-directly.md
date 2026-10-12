# ADR-0016: Call walkie-ros directly and retire walkie-sdk

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

The previous agent, walkie-agent-v2, ran outside ROS. It reached the robot
through [walkie-sdk](https://github.com/EIC-Robocup-2026/walkie-sdk), a Python
library that wrapped walkie-ros topics and actions. This agent is itself a
ROS 2 node, so the SDK would be an extra layer to maintain. It would also hide
the goal statuses and cancel handling the agent needs.

**Decision**

Agent tools call walkie-ros topics, services and actions directly.
walkie-sdk is retired. Its source stays a useful reference for interface
names and the values the old agent used.

**Consequences**

- One fewer library to keep in step with walkie-ros.
- Tools see real goal statuses and can cancel goals.
- Tool code uses walkie-ros message types directly, so they must be built
  into the agent's environment
  ([ADR-0018](/architecture/decisions/0018-pin-robot-interfaces.md)).
- Values copied from walkie-agent-v2 need checking before reuse. For
  example, walkie-sdk lift heights were normalized values, not metres.
<!-- --8<-- [end:body] -->
