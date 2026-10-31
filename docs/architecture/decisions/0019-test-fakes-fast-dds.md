# ADR-0019: Test against in-repo fakes on Fast DDS

<!-- --8<-- [start:body] -->
**Status:** Proposed (amended 2026-10-31, see below) · **Date:** 2026-10-12

**Context**

The tests run real ROS 2 nodes with a scripted LLM and `FakeWalkie`, which
fakes the Nav2 action, TF and the speech topic. They keep traffic local with
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and a random domain ID.

Once the agent uses rmw_zenoh
([ADR-0017](/architecture/decisions/0017-rmw-zenoh-domain-23.md),
[ADR-0033](/architecture/decisions/0033-zenoh-in-every-pixi-environment.md)),
tests that inherit that middleware need a running Zenoh router. Without one,
nodes in different processes do not discover each other, and a test that
needs such a node fails with a timeout that says nothing about the cause.

**Decision**

- `tests/conftest.py` sets `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` before it
  imports rclpy, because rcl loads the middleware at import. It keeps the
  random domain and local discovery.
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

**Amendment, 2026-10-31: the test domain.** A test session no longer draws
its domain with `random.randint`, which let two sessions on one computer
collide about once in 101 pairs. When `tests/conftest.py` is imported, before
rclpy is, it takes `WALKIE_TEST_DOMAIN_ID` if that is set, and otherwise the
first free domain in a shuffled 100 to 200. Free means that the session gets
an exclusive `flock` on the domain's lock file in the temporary directory, and
it holds that lock until it ends. A requested domain that another session
holds fails the session at once. The session never reads the shell's
`ROS_DOMAIN_ID`, which may point at a running simulation or the robot. It
exports the domain it took as `ROS_DOMAIN_ID`, so a subprocess that a test
starts joins it, and prints `ros domain: <d>` in the pytest header.
<!-- --8<-- [end:body] -->
