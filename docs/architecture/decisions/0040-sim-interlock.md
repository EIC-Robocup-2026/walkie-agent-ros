# ADR-0040: Grant Standard and Unrestricted only behind a sim interlock

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

Generated tools at Standard and Unrestricted may move the base, and move the
arms through atomics
([ADR-0039](/architecture/decisions/0039-access-levels.md)). Those levels
must never run against the real robot.

Nothing in the setup keeps them apart today.
[ADR-0036](/architecture/decisions/0036-agent-sim-shell-domain.md) runs
`agent-sim` on the calling shell's `ROS_DOMAIN_ID`, and records that a shell
exporting 23 and reaching the robot's router puts the agent on the robot's
graph, where its goals reach the robot's Nav2. walkie-ros's
`scripts/bringup/connect_walkie_zenoh_router.sh` exports 23 and bridges to
the robot's router at `tcp/10.0.0.6:7447`. The graph has no authentication
(no SROS2, [ADR-0009](/architecture/decisions/0009-forbidden-names-guard-rail.md)).

We considered:

- **Trusting the flag alone.** One wrong shell or launch file would run
  generated motion on the robot.
- **A domain number alone.** 23 means "robot", but a sim may run on 0, 42 or
  anything else, so the domain can veto but never confirm.
- **A fixed sim domain as the only guard.** ADR-0036 removed the fixed
  domain for good reasons, and a domain alone would not catch a router
  bridged to the robot. Whether Unrestricted still gets its own domain and
  router is open (below).

**Decision**

`src/walkie_agent/toolbox/interlock.py` (new) grants Standard or Unrestricted
only when the `--access-level` flag asks for it and every check passes. It
can only lower the requested level, never raise it.

| # | Check | Passes when |
| - | ----- | ----------- |
| 1 | Sim-host marker | root-owned `/etc/walkie/sim-host` exists, read with `os.stat`; any `OSError` fails |
| 2 | Robot marker | `/etc/walkie/robot` is absent (`ENOENT`); any other `OSError` fails |
| 3 | Domain | `ROS_DOMAIN_ID` is not 23 |
| 4 | Zenoh (Unrestricted only) | the session config connects only to loopback, on a dedicated domain; otherwise at most Standard |
| 5 | Sim time | `--use-sim-time` was given |
| 6 | Hardware plugin | `/robot_description` names `mujoco_ros2_control/MujocoSystemInterface` and no real plugin |
| 7 | Sim node | `/mujoco_ros2_control_node` serves `reset_world` and publishes `/clock` |
| 8 | No real drivers | no node with the exact name of a lidar, ZED or head-touch driver is on the graph |
| 9 | Host facts | no `can0` or `can1` link and no `/dev/ttyACM0` |

- Every check value is logged at startup and on every change.
- Checks 6 to 8 re-run every 5 s. Before every N or M `ctx.tools` call,
  publish or action start at Standard or Unrestricted, checks 3 and 8 re-run
  uncached, which costs one graph query of a few ms. A failure refuses that
  call and demotes.
- A failed check demotes the session to Competition until the agent
  restarts. Running invocations above Competition are killed and cleaned
  up, bundles stop at their next step, the coder is aborted, and tools above
  Competition disappear from the next bind. The agent says "Simulation check
  failed (<check>). Running at Competition level." on `/walkie/console` and
  in the trailing reminder, and logs every check value.
- A paused sim (`/clock` stopped, same publisher) is not a demotion.
  Invocations wait for the clock or their deadline.
- Phase 2 of the [roadmap](/architecture/self-evolving-agent.md#roadmap)
  ships checks 1, 2, 3, 5 and 6 with the startup log. Checks 4 and 7 to 9,
  the periodic re-checks and the fresh checks before motion land in phase
  4a, before the harvest relies on them in phase 5.

**Amends ADR-0036.** `agent-sim` gains `--access-level standard` and keeps
the shell's domain. If that shell exports 23, the interlock grants only
Competition, so generated tools lose N and M. The task agent's own tools
(`navigate_to_location`, `move_relative`) still reach the robot's Nav2, as
ADR-0036 says, and its warning stays. Unrestricted is proposed to get its own
task, `agent-sim-unrestricted`, on a dedicated domain with a loopback-only
router. That split is still an open question on the
[Access Levels](/architecture/access-levels.md#open-questions) page.

**Consequences**

- Each dev or harvest machine needs `/etc/walkie/sim-host` installed once
  by root. The robot PC gets `/etc/walkie/robot` and never the sim marker,
  so a sim on the robot PC always runs at Competition.
- The interlock guards against mistakes: the wrong shell, the wrong launch
  file, a bridged router. It does not stop an attacker on the graph. The
  flag-plus-marker rule only keeps such an attacker from raising the level.
- A sim that disappears under the agent demotes the session until the
  agent restarts. The
  [harvest](/architecture/harvest.md) therefore stops the agent before it
  relaunches the sim, and starts a new one only after the sim serves
  `reset_world`, advances `/clock` and has published `/robot_description`.
- Watching the sim in Foxglove does not demote: `foxglove_bridge` and
  `rosbridge_websocket` are tool bridges, not hardware drivers, and check 8
  matches exact node names.
- The 5 s polling leaves a window for non-motion calls. Motion calls close
  it with their fresh checks.
- Checks 6 and 7 depend on walkie-ros's MuJoCo launch. `chula_pat14_6` uses
  the same `mujoco_omnibot.launch.py` and xacro, so both checks should see
  the same plugin and node. The world's assets and tools are on a
  walkie-ros branch that is not on main, so both checks need a re-test
  once that change is merged.

**See also:** [Access Levels: the sim interlock](/architecture/access-levels.md#the-sim-interlock),
[Sim Harvest](/architecture/harvest.md)
<!-- --8<-- [end:body] -->
