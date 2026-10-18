# ADR-0017: Run the agent on the robot PC over rmw_zenoh_cpp on domain 23

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

walkie-ros runs on the robot PC with `RMW_IMPLEMENTATION=rmw_zenoh_cpp` and
`ROS_DOMAIN_ID=23` (today in Docker on the host's network,
`docker/compose.physical.yml`; walkie-ros is moving to Pixi). It starts a
Zenoh router with `ros2 run rmw_zenoh_cpp rmw_zenohd`
(`scripts/tmux_walkie.sh`), which listens on port 7447. When this was
proposed, the agent used ROS 2's default middleware, Fast DDS.
[ADR-0033](/architecture/decisions/0033-zenoh-in-every-pixi-environment.md)
has since moved every Pixi environment to Zenoh. DDS and Zenoh do not talk
to each other.

A mismatch does not raise an error. With the wrong middleware, the wrong
domain or no router, discovery simply finds nothing. Zenoh also puts the
message type's hash in every name, so a node built with a different version
of a message definition never connects either.

**Decision**

- The agent and the voice nodes run on the robot PC, next to walkie-ros,
  from this repository's Pixi environment.
- They run with `RMW_IMPLEMENTATION=rmw_zenoh_cpp`, which the Pixi
  environment sets for every command
  ([ADR-0033](/architecture/decisions/0033-zenoh-in-every-pixi-environment.md)),
  and `ROS_DOMAIN_ID=23`, set by their Pixi tasks or `.env`, never globally.
- They connect to the robot's router at `localhost:7447`, which is
  rmw_zenoh's default. The agent has no router or Zenoh config of its own.
- `ros-jazzy-rmw-zenoh-cpp` comes from RoboStack, at the version walkie-ros
  runs. Once walkie-ros is on Pixi, both `pixi.lock` files pin the same
  version.
- Tests stay on Fast DDS
  ([ADR-0019](/architecture/decisions/0019-test-fakes-fast-dds.md)).

We rejected running the agent on a separate computer over Wi-Fi. That needs
a second router, adds a wireless hop to every tool call, and makes a stop
depend on the wireless link. We also rejected running the agent on DDS
behind a Zenoh-DDS bridge. It adds a process and a second config, and
walkie-ros already uses rmw_zenoh.

**Consequences**

- Without this, the agent cannot see the robot at all.
- Commands, results and stops stay on one computer. Only the LLM calls leave
  it, so the robot PC needs internet access and holds the OpenRouter key.
- The agent shares the robot PC's CPU, memory and GPU with walkie-ros. The
  voice models' GPU memory adds to that of the camera and perception nodes.
- The robot PC holds two Pixi environments, each with its own ROS install.
- Tracing a run on the robot means running Langfuse on the robot PC as well,
  because its ports listen on `127.0.0.1` only
  ([ADR-0006](/architecture/decisions/0006-self-hosted-langfuse.md)).
- The version must match the robot's. RoboStack offers 0.2.10, which fits the
  current environment, and 0.2.11, which pulls a newer build of the whole ROS
  stack.
- Zenoh's request timeouts come from the agent's own session config, and
  waiting for an action result has no timeout. Every tool must bound its own
  waits, as `navigate_to_location` already does.
- To debug "nothing connects", run `pixi run ros2 topic list` with the same
  `ROS_DOMAIN_ID` and check that `rmw_zenohd` is running.

**See also:** [rmw_zenoh](https://github.com/ros2/rmw_zenoh)
<!-- --8<-- [end:body] -->
