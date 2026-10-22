# ADR-0039: Give generated tools three nested access levels

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

The [self-evolving agent](/architecture/self-evolving-agent.md) runs tools it
wrote itself: bundles, which are checked data, and generated Python in the
`temporary` tier, which runs in a sandbox
([ADR-0037](/architecture/decisions/0037-tool-tiers-bundles-effects.md),
[ADR-0047](/architecture/decisions/0047-sandbox-and-bridges.md)). The same
machinery serves two very different jobs:

- In simulation, the agent should find out what is missing: new atomics,
  perception ideas, and tools that can be built from what exists. Wide
  access is cheap there and teaches the most.
- On the real robot, a stranger's spoken request can ask for a new tool.
  That request is untrusted input, and the tool runs next to people.

Today the only rule for ROS names is the exact-string `FORBIDDEN_NAMES`
list ([ADR-0009](/architecture/decisions/0009-forbidden-names-guard-rail.md)),
which ADR-0009 itself calls a guard rail, not security. RAI compares names
as given, so `cmd_vel` passes a check written for `/cmd_vel`.

We considered:

- **One policy everywhere, at the robot's level.** The simulation could
  never discover what the robot is missing.
- **Two levels, sim and robot.** The harvest could not tell "build this from
  atomics" from "a human must add an atomic".
- **Three independently written policies.** Nothing would stop a lower level
  from allowing something a higher one forbids.
- **A level key in `config.toml`.** That file reaches the robot through git.

**Decision**

- **Three levels.** Unrestricted (sim only) finds what is missing, with raw
  ROS reads, service calls and actions. Standard (sim only) finds tools that
  can be built from existing atomics, with code. Competition is the default
  and the only level that may run on the real robot.
- **Nesting by construction.** `src/walkie_agent/toolbox/levels.py` (new)
  loads `config/access_levels.toml` (new) and builds
  `STANDARD = UNRESTRICTED.narrow(standard_rules)` and
  `COMPETITION = STANDARD.narrow(competition_rules)`. Every field declares
  a polarity: `*_allow` sets intersect, `*_deny` and `*_require` sets union,
  `require_*` flags OR, switches that grant access when true (`ros.graph`,
  `ros.tf`, `tools_search`) AND, and caps take the minimum, where a missing
  key means unlimited. `narrow()` raises on a key without a declared
  polarity.
- **One never-list, applied first at every level**, for generated code and
  for the reviewed manifests of git-tier tools alike: the agent's control
  plane (writes), `/controller_manager/*`, parameter and lifecycle
  services, the simulator's own node, MoveIt and commander planning names,
  the commander's `/go_to_home` and collision services, ZED control
  services, and every arm, gripper, head and lift controller, including
  `/lift/cmd`. `/set_joint_position`, `/control_gripper` and the `RunSkill`
  action, once its name is pinned, are on it too, except in the
  human-written arm atomics' manifests.
- **Names are resolved before every check.** The bridge resolves each name,
  strips action suffixes and matches globs on the result.
- **Level stamps.** Lint stamps each generated tool with the lowest level
  that allows all of its calls. The host enforces `min(session level,
  stamp)`. A call above the stamp is refused, traced and blocks the tool
  until a human acknowledges it. Stamps never rise on their own.
- **Rule of Two at Competition.** A tool may sense private local data or
  reach the web, never both (linker check L14).
- **The level comes only from the `--access-level` flag.** A missing flag
  means Competition. The sim interlock can only lower it
  ([ADR-0040](/architecture/decisions/0040-sim-interlock.md)).

This changes five earlier decisions:

- **Supersedes ADR-0009 in part.** For generated code, the parent's policy
  on resolved names (globs plus the never-list) replaces the exact-string
  `FORBIDDEN_NAMES`. The list stays the guard rail for hand-written tools.
  Neither is access control; SROS2 still would be.
- **Amends [ADR-0010](/architecture/decisions/0010-bounded-navigation-tool.md).**
  The task LLM still drives only to configured places, and Competition tools
  have no N effect. In sim, generated code may call the internal
  `navigate_to_pose` through `ctx.tools`, as
  [ADR-0043](/architecture/decisions/0043-navigate-to-pose-provenance.md)
  proposes; that choice is still open on
  [Self-Evolving Agent](/architecture/self-evolving-agent.md#open-questions).
  Unrestricted code may start raw Nav2 actions, cancelled on exit.
- **Amends [ADR-0021](/architecture/decisions/0021-arm-control-through-commander.md).**
  Generated code moves the arms only through the arm atomics, at Standard
  and Unrestricted, and not at all at Competition
  ([ADR-0044](/architecture/decisions/0044-mocked-arm-atomics.md)). The arm
  controllers, the MoveIt and commander planning names and the commander's
  arm actions are on the never-list at every level. Hand-written arm tools
  keep ADR-0021's commander contract.
- **Amends [ADR-0031](/architecture/decisions/0031-bounded-relative-moves.md).**
  `move_relative` stays a bounded tool for the task LLM. In sim, generated
  code may call it through `ctx.tools`. Raw Nav2 goals from Unrestricted
  code are not bound by its 3 m limit. They run only in sim, after fresh
  interlock checks.
- **Amends [ADR-0028](/architecture/decisions/0028-images-and-policy-robot-side.md)**
  for image handles at Unrestricted in sim. The ROS bridge, in the agent
  process, may take one `Image`, `CompressedImage`, `PointCloud2` or
  `OccupancyGrid` message and return a handle. `ctx.handles.load` (phase 4b,
  only if harvest traces show tools blocked without it) copies at most 8 MB
  into the sandbox. Standard strips those fields, Competition has no
  `ctx.ros`, and the task agent's own tools never subscribe to raw images.

**Consequences**

- A draft that publishes `/cmd_vel` at Unrestricted is stamped Unrestricted.
  It never loads at Standard or Competition and is never promoted as it is.
- A computed ROS name or URL host stamps a tool at the session level. The
  harvest report flags such tools whose traces never left Competition.
- Policy changes are reviewed edits to `config/access_levels.toml`. The
  robot reads the same file, but its launch path pins Competition.
- Levels bound what generated code may ask for. The sandbox is what stops
  it from doing anything else.
- The `RunSkill` entry is added once the action's name is pinned
  ([ADR-0027](/architecture/decisions/0027-runskill-action.md)). Until then,
  Unrestricted code could start the skill action directly wherever it runs,
  including the harvest's `FakeSkillServer`.

**See also:** [Access Levels and Generated Tools](/architecture/access-levels.md),
[Competition Mode](/architecture/competition.md),
[Sim Harvest](/architecture/harvest.md)
<!-- --8<-- [end:body] -->
