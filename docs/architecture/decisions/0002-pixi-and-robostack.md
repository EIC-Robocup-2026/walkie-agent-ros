# ADR-0002: Install ROS 2 and Python packages with Pixi and RoboStack

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-07

**Context**

The agent needs ROS 2 Jazzy, RAI (a PyPI package) and a C/C++ toolchain to
build message packages. Team members use different Linux setups. A system ROS
install from apt ties the project to one Ubuntu release and one system Python,
and a stray `pip install` can break it.

**Decision**

We use Pixi with two channels: RoboStack Jazzy for ROS 2 and conda-forge for
everything else. Python-only packages such as `rai-core` come from PyPI under
`[pypi-dependencies]`. Pixi solves both into one environment inside the
project and records every version in `pixi.lock`.

**Consequences**

- There is no system ROS install. `pixi install` is the whole setup, and
  everyone gets the versions in `pixi.lock`.
- Python is 3.12, because RoboStack Jazzy builds for 3.12.
- Conda packages are solved first, and PyPI packages must fit around them.
  Some conda packages need upper bounds that PyPI packages require
  (`numpy<2`, `packaging<26`, `py-opencv<5`). The reasons are commented in
  `pixi.toml`.
- Removing a package index does not re-resolve packages already in
  `pixi.lock`. Run `pixi update <package>` to move them.
- Packages that RoboStack does not publish are built from source
  ([ADR-0004](/architecture/decisions/0004-build-rai-interfaces-from-source.md)).
- walkie-ros is moving from Docker with apt packages to Pixi and RoboStack
  too, so both sides can lock the same ROS packages. Message definitions
  must match exactly either way
  ([ADR-0018](/architecture/decisions/0018-pin-robot-interfaces.md)).

**See also:** [Create the Pixi workspace](/setup/01-pixi-workspace.md),
[Install RAI](/setup/03-ros-and-rai.md)
<!-- --8<-- [end:body] -->
