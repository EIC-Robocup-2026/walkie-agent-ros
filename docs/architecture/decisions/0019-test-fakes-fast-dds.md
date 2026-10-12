# ADR-0019: Test against in-repo fakes on Fast DDS

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

The tests run real ROS 2 nodes with a scripted LLM and `FakeWalkie`, which
fakes the Nav2 action, TF and the speech topic. They keep traffic local with
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and a random domain ID.

Once the agent uses rmw_zenoh
([ADR-0017](/architecture/decisions/0017-rmw-zenoh-domain-23.md)), tests that
inherit that middleware need a running Zenoh router. Without one, nodes do
not discover each other, and the tests fail with timeouts that say nothing
about the cause.

**Decision**

- `tests/conftest.py` sets `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` before
  `rclpy.init`, and keeps the random domain and local discovery.
- Each walkie-ros node the agent calls gets a fake in `src/walkie_agent/sim/`
  with the same names and message types, quirks included.
- Fakes have settings to produce failures (slow motion, planning failure,
  ignored cancel), so error paths are tested.
- An optional Zenoh smoke test, skipped by default, runs one fake and one
  tool under rmw_zenoh with a local router.

**Consequences**

- `pixi run test` works anywhere, without a router or a robot, whatever the
  shell's `RMW_IMPLEMENTATION` is.
- Running the tests on the robot PC does not touch the live robot, because
  Fast DDS and Zenoh do not talk to each other.
- Zenoh-specific problems show up only in the smoke test and on the robot.
- When walkie-ros behavior changes, its fake must change too.
<!-- --8<-- [end:body] -->
