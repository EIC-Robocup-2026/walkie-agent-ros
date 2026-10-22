# ADR-0043: Add an internal navigate_to_pose that takes only poses with handle provenance

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

[ADR-0010](/architecture/decisions/0010-bounded-navigation-tool.md) lets the
LLM send the robot only to places the team configured, because LLMs are
poor at choosing map coordinates.
[ADR-0031](/architecture/decisions/0031-bounded-relative-moves.md) adds
relative moves of up to 3 m. The module docstring of
`src/walkie_agent/tools/navigation.py` says it plainly: neither tool takes
raw map coordinates from the LLM.

The [self-evolving agent](/architecture/self-evolving-agent.md) needs to
drive to things the robot has found, such as "localize the mug, then go to
it", or an object position from memory
([ADR-0035](/architecture/decisions/0035-memory-through-walkie-database.md)).
A new atomic, `localize_object`, gives the 3D position robot-side, with TF
looked up at the image stamp. Which walkie-ros interface serves it is not
settled: walkie-ros's `cloud_to_map` node republishes the whole cloud in
`map` and offers no service, and its `/get_3d_poses` service
(`walkie_perception/srv/GetObPose`) samples its own newest depth image, not
the detection's image. Two problems come with that:

- **Where the pose comes from.** Today `find_objects` returns plain
  sentences to the LLM. In the proposed toolbox, perception and memory
  results will stay in a new `HandleStore`
  ([Self-Evolving Agent](/architecture/self-evolving-agent.md#the-bundle-runtime-and-traces)).
  The LLM sees only ids such as `pose_91bc` or `obj_3f2a.0`. Each entry
  records its producer step, stamp, frame id and the robot pose at
  capture. A target pose should come from such a handle, never from a
  number the LLM wrote.
- **Where the robot stops.** Driving to the object's own position drives
  the robot into the table the mug stands on.

Two options were compared:

| Option | What it means | Cost |
| ------ | ------------- | ---- |
| **(a) Internal `navigate_to_pose(pose: PoseRef)` with a stand-off (recommended)** | The task LLM never sees it. Reviewed presets use it, and in sim generated code may call it through `ctx.tools` | Amends ADR-0010 instead of replacing it |
| (b) An LLM-facing `move_absolute` with coordinates | The LLM picks map coordinates | Supersedes ADR-0010, and brings back the risk it removed |

The choice between (a) and (b) is open. This ADR proposes (a), the
recommendation.

**Decision**

We add `navigate_to_pose(pose: PoseRef)` as an internal atomic tool.

- **Provenance.** Linker check L08 lets handle-typed inputs, such as
  `PoseRef` and `ObjRef`, take only a `Ref` to an earlier step's output or
  a handle-typed `Arg`, never a literal. L08 checks bundles and presets, so
  in them a pose always comes from an earlier step's handle. At run time
  the tool resolves the id in the `HandleStore`, which raises `STALE` if
  the base has moved since capture. Generated code that calls the tool
  through `ctx.tools` is not linked. For that caller, the stand-off and
  range checks are the bound.
- **Stand-off and range.** The tool computes an approach pose at a
  stand-off from the object, facing it, and sends that to Nav2. It refuses
  a target beyond a set distance from the robot, or off the map. The
  stand-off and the distance are not set yet.
- **Bounded waits.** Like every atomic core, it bounds its own waits, as
  `Nav2GoalTool` already does.
- **Handle lifetime.** A detection handle is `STALE` when the base has
  moved more than 5 cm or 3° since capture. A soft age limit of 10 minutes
  covers the rest, long enough for a spoken clarification. Handles clear on
  reset and when a new task replaces the todo list
  ([ADR-0041](/architecture/decisions/0041-todo-list-and-seeding.md)).
- **One name, three things.** RAI's `NavigateToPoseTool` is also named
  `navigate_to_pose`, and it takes x, y and yaw. The agent never registers
  it. The Nav2 action that `Nav2GoalTool` sends goals to has the same name.
  Linker check L15 keeps generated tools off `navigate*` names.
- **Hidden from the task LLM.** `navigate_to_pose` and the new stamp-aware
  `get_transform` are never bound for the task LLM
  ([what the agent sees each turn](/architecture/self-evolving-agent.md#what-the-agent-sees-each-turn)).
  The system prompt and the whoami documents must be updated to match.

Who may call it, by access level
([ADR-0039](/architecture/decisions/0039-access-levels.md)):

| Caller | Unrestricted (sim) | Standard (sim) | Competition |
| ------ | ------------------ | -------------- | ----------- |
| Task LLM | Never | Never | Never |
| Reviewed presets (`bundled_preset`) | Yes | Yes | Yes |
| Generated code, through `ctx.tools` | Yes, with the stand-off and range checks | Yes, with the stand-off and range checks | No: generated tools have no N effect |

At Competition, the task LLM drives only through the bounded
`navigate_to_location` and `move_relative`. In sim, a generated tool can
compute a target as a handle, and a reviewed preset can feed it into a
motion step, where the linker sees it and the effect policy governs it.

**Amends ADR-0010.** "The LLM can only send the robot to places the team
configured" becomes: the task LLM still never chooses coordinates, but
reviewed code, and generated code in sim, can drive the robot to a pose
held as a handle, at a stand-off. At Unrestricted, generated code can also
start raw Nav2 actions
([ADR-0039](/architecture/decisions/0039-access-levels.md)), so the checks
here cover `navigate_to_pose`, presets and Standard code, not every
Unrestricted sim run. This ADR leaves ADR-0031 unchanged; ADR-0039 amends
it for generated code.

**Consequences**

- "Find the mug, then go to it" can work through a reviewed preset that
  chains `localize_object` and `navigate_to_pose` (none is planned yet),
  and no coordinate passes through the LLM.
- A wrong detection can still send the robot to the wrong place, but only
  within range, on the map and at a stand-off.
- The stand-off and the range must be chosen and tested in simulation
  before the tool is used on the robot.
- The tool depends on typed outputs from
  [ADR-0037](/architecture/decisions/0037-tool-tiers-bundles-effects.md)
  and handles from the bundle runtime, and on perception from walkie-vision
  ([ADR-0034](/architecture/decisions/0034-vision-models-in-walkie-vision.md))
  or memory from walkie-database.
- Generated code can drive the base only at Standard and Unrestricted,
  which run only on a sim host after the
  [sim interlock](/architecture/decisions/0040-sim-interlock.md) passes.
- The tool lands in phase 2. Nothing generates bundles or code then: the
  atomic agent and the L08 check arrive in phase 3, the coder in phase 4a.
  Until then, only reviewed presets call it.

**See also:** [Self-Evolving Agent](/architecture/self-evolving-agent.md),
[Access Levels](/architecture/access-levels.md),
`src/walkie_agent/tools/navigation.py`
<!-- --8<-- [end:body] -->
