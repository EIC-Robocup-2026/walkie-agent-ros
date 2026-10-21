# ADR-0036: Run agent-sim on the calling shell's ROS_DOMAIN_ID

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-21

**Context**

`pixi run agent-sim` set `ROS_DOMAIN_ID=42` in its task `env`
([ADR-0030](/architecture/decisions/0030-mujoco-sim-first.md)), and Getting
Started told the user to export the same value before launching the
simulation from walkie-ros. Checked against walkie-ros on 2026-10-21:

- walkie-ros never sets domain 42. Its scripts and Docker files set only 23,
  the real robot's domain. Its middleware guide says `ROS_DOMAIN_ID` can stay
  unset on one computer, and its Pixi tasks and simulation launch do not set
  it, so a simulation started the walkie-ros way runs on domain 0.
- A task's `env` replaces the value exported in the calling shell, so
  `agent-sim` ran on 42 whatever the shell said.
- Under rmw_zenoh the domain ID is part of every key expression. Nodes on
  different domains share the router but never see each other's topics or
  graph, and nothing reports an error
  ([ADR-0017](/architecture/decisions/0017-rmw-zenoh-domain-23.md)).

With the simulation on domain 0, the agent on domain 42 found no TF and no
speech node. `move_relative` answered that the robot's position was unknown,
which looked like a TF or Zenoh fault.

**Decision**

- `agent-sim` no longer sets `ROS_DOMAIN_ID`. It uses the calling shell's
  value, or domain 0 when the variable is unset, like walkie-ros's
  simulation and every other task here.
- Getting Started tells the user to use the same `ROS_DOMAIN_ID`, or none, in
  every terminal of the simulation setup. It no longer prescribes 42.
- When `move_relative` cannot find the robot's transform and no node
  publishes `/tf` or `/tf_static`, its answer names the likely cause: the
  robot is not running, or the agent uses another domain or router.

This replaces "on domain 42" in ADR-0030's decision and "Domain 42 keeps the
simulation apart from the robot's domain 23" in its consequences. The rest of
ADR-0030 stays in effect.

We rejected keeping 42 and adding a matching task or note to walkie-ros. That
leaves a convention that two repositories must keep in step, and walkie-ros's
own guide says the opposite.

**Consequences**

- The agent follows the simulation however it was launched, as long as both
  terminals export the same value or neither does.
- Nothing keeps the simulation apart from the real robot any more. A shell
  that exports `ROS_DOMAIN_ID=23` and reaches the robot's router puts
  `agent-sim` on the robot's domain, where its goals reach the robot's Nav2.
  Start the simulation and the agent from shells that do not export 23.
- A domain mismatch now shows up in the `move_relative` answer, which the
  agent's log prints. The check reads the ROS graph, so it cannot tell a
  stopped robot from one on another domain.

**See also:** `pixi.toml`, `src/walkie_agent/tools/navigation.py`,
[ADR-0030](/architecture/decisions/0030-mujoco-sim-first.md),
[Getting Started](/getting-started.md)
<!-- --8<-- [end:body] -->
