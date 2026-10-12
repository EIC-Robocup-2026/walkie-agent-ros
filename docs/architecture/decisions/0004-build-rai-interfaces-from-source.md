# ADR-0004: Build rai_interfaces from source in ros2_ws

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

RAI's message types, such as `HRIMessage`, are in the `rai_interfaces`
package, and `rai-core` imports it. RoboStack does not publish it, and it is
not on PyPI.

**Decision**

`ros2_ws.repos` lists `rai_interfaces` at a fixed commit (the main-branch
commit that RAI 2.12 tracks). `pixi run build-ros2` clones it with vcstool
and builds it with colcon in `ros2_ws/`. `scripts/activate.sh` sources the
result in every `pixi run` and `pixi shell`. CMake gets the environment's
Python explicitly, so a user's `~/.colcon/defaults.yaml` cannot point it at
the system Python.

**Consequences**

- After cloning, and whenever `ros2_ws.repos` changes, run
  `pixi run build-ros2` once.
- The pin is a commit, not a branch, so every build gets the same messages.
- The environment carries a C/C++ toolchain (`c-compiler`, `cxx-compiler`,
  `cmake`).
- The same mechanism will bring in walkie-ros's message packages
  ([ADR-0018](/architecture/decisions/0018-pin-robot-interfaces.md)).

**See also:** [Install RAI, step 5](/setup/03-ros-and-rai.md#step-5-build-rai_interfaces)
<!-- --8<-- [end:body] -->
