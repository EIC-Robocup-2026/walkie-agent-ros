# ADR-0033: Use rmw_zenoh_cpp in every Pixi environment

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-19

**Context**

walkie-ros and the robot use `rmw_zenoh_cpp`. Here, only `agent-sim` set
`RMW_IMPLEMENTATION`
([ADR-0030](/architecture/decisions/0030-mujoco-sim-first.md)). Every other
command, such as `pixi run fake-walkie` or `pixi run ros2 topic pub`, used
Fast DDS unless the terminal exported the variable. A terminal that missed it
saw nothing, with no error
([ADR-0017](/architecture/decisions/0017-rmw-zenoh-domain-23.md)).

Checked on Pixi 0.80:

- Pixi runs the activation scripts first, then applies `[activation.env]`,
  then a task's `env`. `scripts/activate.sh` sees only the calling shell's
  `RMW_IMPLEMENTATION`. It preloaded 0.2.11 only when that value was
  `rmw_zenoh_cpp`
  ([ADR-0032](/architecture/decisions/0032-rmw-zenoh-from-source.md)), so
  `agent-sim` from a clean shell got RoboStack's 0.2.10, which can freeze.
- The docs environment leaves out the default feature. It gets neither the
  activation scripts nor the workspace's `[activation.env]`.
- `pixi run` and `pixi shell` do not show what an activation script prints,
  so a warning from `scripts/activate.sh` would reach no one.
- Preloading 0.2.11 into every process does not work. Its `rmw_*` functions
  then replace those of `librmw_implementation.so`, so every ROS process runs
  Zenoh. A process that asks for Fast DDS exits with code 102: "Expected RMW
  implementation identifier of 'rmw_fastrtps_cpp' but instead found
  'rmw_zenoh_cpp'".
- rcl reads `RMW_IMPLEMENTATION` and loads that middleware when rclpy is
  imported, not at `rclpy.init`. `tests/conftest.py` set Fast DDS after the
  import, so the tests ran on whatever middleware the shell chose.

**Decision**

- `[activation.env]` in `pixi.toml` sets `RMW_IMPLEMENTATION=rmw_zenoh_cpp`
  for every `pixi run` and `pixi shell`. `[feature.docs.activation.env]` sets
  it for the docs environment. The value replaces one exported in the
  calling shell.
- There is no Zenoh config. Nodes connect to a router at `localhost:7447`,
  rmw_zenoh's default. Start one with
  `pixi run ros2 run rmw_zenoh_cpp rmw_zenohd`, or use the one started from
  walkie-ros.
- `scripts/activate.sh` preloads a small loader,
  `scripts/rmw_zenoh_loader.c`, once `pixi run build-rmw-zenoh` has built it
  and 0.2.11. When a process starts, after Pixi has applied its whole
  environment, the loader opens 0.2.11 if `RMW_IMPLEMENTATION` is
  `rmw_zenoh_cpp`. It opens it with `RTLD_LOCAL`, so its functions replace
  nothing, and a process that changes the middleware before it imports
  rclpy gets the one it asked for. If the file exists but cannot be opened,
  the loader prints the reason to stderr.
- `build-rmw-zenoh` builds the loader with `cc`. It runs with `LD_PRELOAD`
  cleared, because it overwrites the installed library in place.
- To use another middleware, set `RMW_IMPLEMENTATION` in a task's `env`, or
  for one command:
  `pixi run env RMW_IMPLEMENTATION=rmw_fastrtps_cpp <command>`.
- `tests/conftest.py` sets Fast DDS before it imports rclpy
  ([ADR-0019](/architecture/decisions/0019-test-fakes-fast-dds.md)).

This replaces "Only `agent-sim` sets `RMW_IMPLEMENTATION`" in ADR-0030, and
the preload in ADR-0032. The rest of both stays in effect.

We rejected exporting the variable from `scripts/activate.sh`. It would not
reach the docs environment, a task's `env` could still change the middleware
after the script had chosen what to preload, and walkie-ros and
walkie-database set it in `[activation.env]` as well.

**Consequences**

- Every ROS process started through Pixi uses Zenoh, including the `ros2`
  command line tools, `fake-walkie` and `agent`. They need a running router.
  Without one, each prints "Unable to connect to a Zenoh router" and does not
  discover nodes in other processes.
- Only one router can listen on port 7447. Start one per computer and share
  it with walkie-ros.
- The tests stay on Fast DDS and need no router. CI does not build
  `rmw_zenoh_ws`, so it has no preload and runs as before.
- An existing checkout must run `pixi run build-rmw-zenoh` again to get the
  loader. Until then, Zenoh processes use RoboStack's 0.2.10, and nothing
  warns about it. Stop the processes started from this project first, the
  router included: they have 0.2.11 loaded, and the build overwrites it in
  place.
- Under Zenoh, every process in a Pixi session loads 0.2.11 when it starts,
  also ones that do not use ROS. This adds about 5 ms to each start, as the
  preload in ADR-0032 did. A process on another middleware loads only the
  loader, which adds about 0.1 ms. Measured with `/bin/true` and
  `python -c pass`, with the files in the page cache.
- The preload is passed on to child processes. A `pixi run` in another
  project sets its own `PIXI_PROJECT_ROOT`, so its processes do not load
  this build.
- When the workspace from ADR-0032 is removed, the loader and the preload go
  with it. `[activation.env]` stays.

**See also:** `pixi.toml`, `scripts/activate.sh`,
`scripts/rmw_zenoh_loader.c`, `tests/conftest.py`,
[Getting Started](/getting-started.md)
<!-- --8<-- [end:body] -->
