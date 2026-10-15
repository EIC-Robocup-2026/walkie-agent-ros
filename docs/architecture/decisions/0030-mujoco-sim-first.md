# ADR-0030: Drive walkie-ros's MuJoCo simulation before the real robot

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-15

**Context**

The agent has only driven `FakeWalkie` so far. The first link to walkie-ros
should control navigation, with no map for now. Checked against walkie-ros
`main` (f5124c7) and `feat/Pixi` (5f29128):

- The only Nav2 launch with a blank map is the MuJoCo simulation,
  `robot_bringup nav_mujoco_omnibot.launch.py`, on `feat/Pixi` and
  `feat/mujoco-launch`. Its map is 20 x 20 m, centered on the start pose, and
  `map` equals `odom` equals the start pose. There is no AMCL, and the
  position comes from wheel odometry, which drifts. It runs on `/clock`.
- No blank-map launch exists for the real robot. On `main`, Nav2's velocity
  commands skip the collision monitor, there is no software e-stop, and the
  odometry is open-loop.
- Nav2 serves `/navigate_to_pose` (stock `nav2_msgs/action/NavigateToPose`, no
  namespace) with `base_footprint` as the robot frame. On `main`,
  `base_link` is rotated 90 degrees from `base_footprint`.
- walkie-ros `feat/Pixi` uses RoboStack's `rmw_zenoh_cpp` 0.2.11. That build
  needs a newer ROS stack that requires OpenCV 5, which rai-core rejects.
  RoboStack's 0.2.10 fits this environment and uses the same Zenoh library,
  libzenohc 1.9.0.

**Decision**

- The first target is the MuJoCo simulation. `pixi run agent-sim` runs the
  agent under `rmw_zenoh_cpp` on domain 42 with `--use-sim-time` and
  `config/locations.sim.toml`. The simulation is launched from walkie-ros on
  the same domain, and its router is the only one.
- The environment pins `ros-jazzy-rmw-zenoh-cpp` 0.2.10. Only `agent-sim`
  sets `RMW_IMPLEMENTATION`. The tests force Fast DDS
  ([ADR-0019](/architecture/decisions/0019-test-fakes-fast-dds.md)).
- The robot frame is `base_footprint`, in the tools and in `FakeWalkie`.
- `config/locations.sim.toml` holds test points within 2 m of the start
  pose. They are not rooms.

**Consequences**

- The real robot stays out of reach until walkie-ros has a launch for it and
  the safety gaps above are closed. Domain 42 keeps the simulation apart from
  the robot's domain 23.
- The agent's 0.2.10 talking to walkie-ros's 0.2.11 is not tested yet. If
  they do not connect, the agent needs RoboStack's newer ROS stack, and the
  OpenCV conflict with rai-core has to be solved first.
- The agent's `nav2_msgs` (Jazzy 1.3.12) differs from the robot's
  source-built 1.4.0 in the feedback message only. Under rmw_zenoh the agent
  receives no feedback, which no tool uses.
- Drift makes named points less accurate after each goal.
- A goal at the robot's current position fails in Nav2.

**See also:** [ADR-0017](/architecture/decisions/0017-rmw-zenoh-domain-23.md),
`config/locations.sim.toml`, [Getting Started](/getting-started.md)
<!-- --8<-- [end:body] -->
