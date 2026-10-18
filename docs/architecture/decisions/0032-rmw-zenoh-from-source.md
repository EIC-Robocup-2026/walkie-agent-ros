# ADR-0032: Build rmw_zenoh_cpp 0.2.11 from source and preload it

<!-- --8<-- [start:body] -->
**Status:** Accepted, partly superseded by
[ADR-0033](/architecture/decisions/0033-zenoh-in-every-pixi-environment.md) ·
**Date:** 2026-10-16

**Context**

Under `agent-sim`, `move_relative` sometimes refused to move because the
robot's position was "out of date". The simulation was healthy: its `/clock`
ran at real time and TF was never more than 60 ms behind it. The agent's ROS
node had stopped processing callbacks. Its executor thread was blocked in
`rmw_wait` of `rmw_zenoh_cpp` 0.2.10, in a condition wait with no timeout,
while messages kept arriving. Both the sim clock and the TF buffer froze, so
no transform could ever be newer than the call.

- A plain rclpy node with sim time and a TF listener froze the same way, with
  either executor. RAI is not the cause.
- This is [ros2/rmw_zenoh#1032](https://github.com/ros2/rmw_zenoh/issues/1032):
  0.2.10 resets the wait set's `triggered` flag outside its mutex, so a
  message that arrives during `rmw_wait` can be lost and the wait never ends.
  0.2.11 fixes it ([#1040](https://github.com/ros2/rmw_zenoh/pull/1040)).
- RoboStack's 0.2.11 needs a ROS stack that requires OpenCV 5, which rai-core
  rejects ([ADR-0030](/architecture/decisions/0030-mujoco-sim-first.md)).
  0.2.11 uses the same Zenoh version as 0.2.10.
- The environment's `librmw_implementation.so` has an RPATH of its own
  directory, which the loader searches before `LD_LIBRARY_PATH`. Sourcing a
  workspace that contains another `librmw_zenoh_cpp.so` does not replace it.

Against the running simulation, a node froze in 6 of 6 one-minute runs with
0.2.10, and in 0 of 6 with 0.2.11 built from source.

**Decision**

- `pixi run build-rmw-zenoh` builds `rmw_zenoh_cpp` 0.2.11 in
  `rmw_zenoh_ws/`, against the environment's `zenoh_cpp_vendor` and
  libzenohc 1.9.0, plus the zenoh-cpp 1.9.0 headers that RoboStack does not
  ship. Sources are pinned in `rmw_zenoh_ws.repos`.
- `scripts/activate.sh` sets `LD_PRELOAD` to the built library when
  `RMW_IMPLEMENTATION` is `rmw_zenoh_cpp`. ROS then reuses the preloaded
  library instead of opening 0.2.10.
- The workspace is separate from `ros2_ws`, so CI does not build it. Tests run
  on Fast DDS ([ADR-0019](/architecture/decisions/0019-test-fakes-fast-dds.md)).
- RoboStack's 0.2.10 stays installed. It provides the router, `rmw_zenohd`.

**Consequences**

- Without `build-rmw-zenoh`, Zenoh runs fall back to 0.2.10 and can freeze.
- Every process in a Zenoh pixi session loads the library, including ones that
  do not use ROS. This costs a few milliseconds at each start.
- When RoboStack offers 0.2.11 or newer for a ROS stack that rai-core accepts,
  remove the workspace, the tasks and the preload, and bump the pin.

**See also:** `rmw_zenoh_ws.repos`, `scripts/activate.sh`,
[Getting Started](/getting-started.md)
<!-- --8<-- [end:body] -->
