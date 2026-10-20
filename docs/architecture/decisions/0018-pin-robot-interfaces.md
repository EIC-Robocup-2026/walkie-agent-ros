# ADR-0018: Pin walkie-ros message packages to the robot's deployed commit

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

Agent tools need walkie-ros message types, for example `SetJointPosition`,
`ControlGripper` and `GetJointStates` from `my_robot_interfaces`. Under
rmw_zenoh, a client built from a different version of a message definition
does not fail: it never matches the server
([ADR-0017](/architecture/decisions/0017-rmw-zenoh-domain-23.md)).
[ADR-0035](/architecture/decisions/0035-memory-through-walkie-database.md)
has since found that a service client does find such a server, but its
requests are never answered.

walkie-ros is public, and `my_robot_interfaces` is in its main tree
(`manipulate/my_robot_interfaces`), so no submodules are needed.

**Decision**

- `ros2_ws.repos` gets a walkie-ros entry, pinned to the commit checked out
  on the robot PC.
- `pixi run build-ros2` builds only the message packages:
  `colcon build --packages-select rai_interfaces my_robot_interfaces`.
- The clone is ignored by Git.
- The agent does not add MoveIt or ros2_control message packages
  (`moveit_msgs`, `control_msgs`, `controller_manager_msgs`) to its
  environment.

**Consequences**

- When the robot's interfaces change, bump the pin and run
  `pixi run build-ros2`.
- The whole walkie-ros repository is cloned, but only its message packages
  are built.
- When an action server "did not respond", a stale pin is a likely cause,
  and tool error messages should say so.
- If cloning all of walkie-ros becomes a burden, `my_robot_interfaces` could
  move to its own small repository. Once walkie-ros is on Pixi, it could
  also publish its message packages as conda packages for the agent to
  depend on.
<!-- --8<-- [end:body] -->
