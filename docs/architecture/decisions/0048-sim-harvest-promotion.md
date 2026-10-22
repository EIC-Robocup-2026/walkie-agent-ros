# ADR-0048: Promote generated tools only from a sim harvest, through a human-reviewed pull request

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

The self-evolving agent writes tool drafts while it works. They live under
`$WALKIE_TOOLS_HOME`, outside git, stamped with the access level their calls
need ([ADR-0037](/architecture/decisions/0037-tool-tiers-bundles-effects.md),
[ADR-0039](/architecture/decisions/0039-access-levels.md)). Some are worth
keeping, but none can be trusted on its own evidence. The agent's claim of
success can be wrong, a draft can pass once by luck, and a draft made at
Competition was shaped by a visitor's words.

Judging drafts needs many unattended episodes and a truth to score them
against. walkie-ros's MuJoCo simulation can provide both
([ADR-0030](/architecture/decisions/0030-mujoco-sim-first.md)). Checked
read-only against walkie-ros and walkie-worlds on 22 October 2026:

- chula_pat14_6 is a captured living room and kitchen, with named rooms,
  locations, a door and twelve movable objects. Its assets and world tools
  are on a walkie-ros branch, not on main. The assets briefly lived in a
  walkie-worlds repository, which has no remote and keeps only local history.
- Given an explicit model and map, `nav_mujoco_omnibot.launch.py` publishes
  `map` → `odom` as a static identity, with no AMCL. `odom` is open-loop
  odometry. So the robot must start at the map origin, and its estimate
  drifts during an episode. walkie-ros measured 0.17 to 0.43 m after one or
  two goals in simpler worlds.
- walkie-ros's MuJoCo hardware plugin publishes the true base pose on
  `/mujoco_ground_truth/odom`, on main.
- `ResetWorld` restores the whole world, robot included, and may rewind sim
  time (unverified). `SetFreeJointState` moves only the listed bodies.
- The omni-wheel controller has no odometry-reset service and no
  initial-pose parameter.

**Decision**

- **Git only through a reviewed pull request.** A generated tool reaches git
  only as a pull request that a human opens, reviews and merges. The
  evidence comes from a sim harvest report. The planned
  `pixi run toolbox promote` writes the branch, the tool file, generated
  tests, scenario fixtures and a doc stub. Nothing promotes automatically.
- **One exception: practice-session drafts.** A draft from a manual practice
  session may skip the harvest. It still needs at least two successful
  traced runs and a replay test against the planned `FakeScene`, and it
  still goes through the same command and a human-reviewed pull request.
- **Unrestricted code is evidence.** Code stamped Unrestricted goes to the
  atomic wishlist. A human writes the atomic, starting with an interface ADR
  when a new interface is needed
  ([ADR-0015](/architecture/decisions/0015-repository-boundary.md)).
- **Quarantine through the lane only.** Competition quarantine enters a
  harvest through the quarantine lane, with its card regenerated from the
  code, and joins the draft pool only after a human approves it.
- **A relaunch per episode, from the origin.** Each episode stops the agent
  and its walkie-database node, kills the sim launch's process group, and
  relaunches with an explicit world file spawned at the origin and the
  world's map. The harness waits for the sim, Nav2, the database and a
  ground-truth reading at the origin, places objects with one
  `SetFreeJointState` call, and starts a fresh agent at its level.
- **Position on ground truth.** Position checks read
  `/mujoco_ground_truth/odom`, never odometry or Nav2's estimate. Each check
  and each Nav2 abort records the localization error, and failures are split
  into agent, navigation and localization.
- **Isolated workers.** Each worker runs on `ROS_DOMAIN_ID` 77 + i, set by
  the harness so that a shell export never applies, with its own
  loopback-only Zenoh router.

We rejected `ResetWorld` plus a controller reload between episodes. It
would put the robot back at its spawn while `odom` keeps its drift, so the
identity `map` → `odom` would be wrong from the start of the next episode.
The controller has no odometry reset, and which lifecycle transition resets
its odometry is untested. A relaunch is slower but leaves nothing behind.

We rejected automatic promotion above a success threshold. v0 has no
calibrated ranking, and a strict filter almost never passes: at a 95% Wilson
lower bound of 0.8, 20 trials pass only at 20/20. Scoring position on
odometry was rejected because drift would show up as agent failures. AMCL,
through the launch's `world:=` mode, is not recommended for v0, because it
has not had a live test in this world. That choice is still open
([Sim Harvest](/architecture/harvest.md#open-questions), question 3).

This builds on ADR-0030 and amends no ADR. The harvest uses the same sim
launch with its own world file and map, and a new
`config/locations.chula_pat14_6.toml` instead of the test points in
`config/locations.sim.toml`. `agent-sim` still follows the calling shell's
domain ([ADR-0036](/architecture/decisions/0036-agent-sim-shell-domain.md)).
Harvest workers set their own domains instead, because the interlock grants
Unrestricted only on a dedicated loopback domain, and a stray shell export
of 23 must never put a harvest on the robot's domain.

**Consequences**

- Git grows only as fast as a human reviews. The reviewer and cadence are
  still open.
- Every episode pays for a relaunch: about 25 s graceful, or an estimated 12
  to 17 s by killing the process group. Start time with this world is
  unmeasured.
- Episodes are independent. Drift, object positions, the agent process and
  the walkie-database sim instance all start fresh.
- Every v0 episode starts by passing the entrance doorway, about 0.64 m at
  its narrowest gap for a 0.478 m footprint plus padding. Doorway trouble
  shows up as navigation failures, and H4 scores the tool's own step. The
  recommended fallback is a living-room start with a static `map` → `odom`.
- Ground truth covers the robot's base only. Object ground truth comes from
  the spec and the `FakeSkillServer`, so an object that an arm knocks goes
  unrecorded until walkie-ros configures an object-pose publisher.
- The recommendation, still open, is that phase 5 waits until walkie-ros's
  world change is merged to main and its commit pinned. If the assets move
  back to walkie-worlds, that repository is pushed and pinned too.
- Keepout in sim Nav2, ground-truth localization and the object-pose
  publisher are walkie-ros requests, not changes made here.

**See also:** [Sim Harvest](/architecture/harvest.md),
[Access Levels and Generated Tools](/architecture/access-levels.md),
[Self-Evolving Agent](/architecture/self-evolving-agent.md),
[ADR-0027](/architecture/decisions/0027-runskill-action.md),
[ADR-0040](/architecture/decisions/0040-sim-interlock.md),
[ADR-0044](/architecture/decisions/0044-mocked-arm-atomics.md)
<!-- --8<-- [end:body] -->
