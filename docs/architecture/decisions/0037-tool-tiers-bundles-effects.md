# ADR-0037: Split tools into atomic and bundled tiers, keep bundles as data, and gate them by effect

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

Today `build_walkie_tools()` in `src/walkie_agent/tools/__init__.py` builds
eight tools once, plus `search_robot_docs` after `pixi run whoami-build`.
All return plain text, and the Walkie tools return one sentence. The
[self-evolving agent](/architecture/self-evolving-agent.md) must also
combine tools into larger ones, and create new tools while it runs. Three
things stand in the way:

- A sentence cannot be wired into the next tool's input. Combining tools
  needs typed outputs.
- Text written by an LLM must not run unchecked on a robot.
- Nothing records what a tool does to the world. A tool that moves an arm
  and a tool that reads memory look the same to the agent.

Three file formats for combined tools were compared:

- **YAML.** The simplest loader, but no IDE help.
- **Plain Python functions.** The most expressive, but every generated one
  needs a sandbox, and it can be checked only in part.
- **A Python-syntax DSL, parsed with `ast` and never executed.** IDE help
  and complete checking, for a loader and renderer of about 120 lines.

The arm, skill and vision backends the new tools call do not exist yet.
[ADR-0019](/architecture/decisions/0019-test-fakes-fast-dds.md) asks for a
ROS-level fake of every walkie-ros node the agent calls, but an interface
that is not built or pinned has nothing to copy.

**Decision**

Tools come in four tiers. Two are in git and reviewed. Two are generated,
and live outside the repository under `$WALKIE_TOOLS_HOME` (default
`~/.local/share/walkie/tools`, set with `[walkie.toolbox] root`).

| Tier | Folder | Written by | Runs in |
| ---- | ------ | ---------- | ------- |
| atomic | `src/walkie_agent/tools/atomic/` | a human, or a promoted generated tool | the agent process if `META.runtime = "inprocess"`, otherwise a sandbox worker |
| bundled_preset | `src/walkie_agent/tools/bundled_preset/` | a human, often from a promoted bundle | `BundleRunner` |
| bundled | `$WALKIE_TOOLS_HOME/bundled/` | the atomic agent | `BundleRunner` |
| temporary | `$WALKIE_TOOLS_HOME/temporary/` | the tool coder | a sandbox worker |

- **Typed atomics.** An atomic tool is one module with `META`, `Args`,
  `Out` (a typed model with a `summary` line for the LLM) and
  `run(ctx, args)`. A failure raises `StepError` with a code from a fixed
  set, such as `NOT_FOUND` or `SKILL_FAILED`. Today's tools gain a typed
  `core()` in place and keep their sentences, so there is one catalogue
  without a file move.
- **Bundles are data.** A bundle file holds one `BUNDLE = Bundle(...)` with
  literal arguments. The loader accepts only the docstring, one import from
  `walkie_agent.toolbox.dsl` and that assignment, and the agent never
  imports the file. An LLM emits a JSON `BundleSpec`, and `dsl.render()`
  writes the file. A linker (checks L01 to L15) and a symbolic dry run
  check every bundle before it loads. `BundleRunner` runs and traces it.
- **Effects.** R reads, P perceives or computes, S speaks, N moves the
  base, M moves an arm or gripper. An atomic declares its effect in `META`,
  the lint computes those of generated code, and a bundle's are the union
  of its steps. Retries are allowed only on idempotent R or P steps. Check
  L09 applies the policy of the tier and the access level
  ([ADR-0039](/architecture/decisions/0039-access-levels.md)).
- **Recovery.** If any arm step started, the runtime always stows: it calls
  `get_arm_state`, then `move_arm_to_pose("home")` unless the arms are
  travel-safe. A preset may add `Recover(tool, args, if_ran=...)`, which
  runs only if the named step started, so a perception failure never moves
  an arm.
- **Promotion.** A tool reaches `atomic/` or `bundled_preset/` only through
  a human-reviewed pull request
  ([ADR-0048](/architecture/decisions/0048-sim-harvest-promotion.md)).
  Promoted generated code keeps `META.runtime = "sandbox"` until a human
  rewrites or audits it and names a reviewer.
- **Mocks behind protocols.** Unit tests run the new atomics against mocks:
  in-process ones (`FakeScene`, `MockSkillServer`), or fake ROS nodes that
  serve the real interfaces (`FakeCommander` behind the `arm` backend, a
  fake walkie-vision publisher behind `vision`). Each atomic lists its
  backends in `META.requires`. The
  registry hides mock-backed tools unless the run is a simulation, and a
  real run refuses to start if a required backend is a mock.

This changes three earlier decisions:

- **Amends [ADR-0009](/architecture/decisions/0009-forbidden-names-guard-rail.md).**
  The forbidden list stays the guard rail for hand-written ROS tools.
  Effects add a check that does not depend on names: the linker and the
  runner refuse a step whose effect the tier and level do not allow. An
  in-process atomic declares the ROS names and hosts it uses (`META.ros`,
  `META.net`), and its `ctx` enforces them, but a hand-written one may
  still use the connector directly, under review. None of this is a
  security boundary. For generated code, the sandbox is
  ([ADR-0047](/architecture/decisions/0047-sandbox-and-bridges.md)).
- **Amends [ADR-0015](/architecture/decisions/0015-repository-boundary.md).**
  Generated drafts are agent-side code. They stay under `$WALKIE_TOOLS_HOME`
  and never enter the repository or the installed package, except through a
  reviewed promotion pull request. The repository gains
  `src/walkie_toolbox_facade/`, the small package that sandboxed code
  imports. ADR-0015's order still holds: a tool that needs a new robot
  interface waits for that interface in walkie-ros.
- **Records a gap in ADR-0019.** The protocol mocks above are not ROS-level
  fakes. ROS-level fakes follow: a `FakeSkillServer` for the sim harvest,
  and a `FakeCommander` once the commander's interface is pinned. The
  arm side is in [ADR-0044](/architecture/decisions/0044-mocked-arm-atomics.md).

**Consequences**

- Bundles are fully checkable, and no LLM-written bundle text ever runs.
  They have no loops or branches, and a preset cannot call another. A loop,
  arithmetic or a new data source needs a sandboxed generated tool.
- Every perception and arm tool must return a typed `Out` before any
  bundle can use it. This refactor comes first.
- The existing sentences stay the same, so the string assertions in
  `tests/test_tools_ros.py` keep passing.
- A bundle answers with the step that failed and its code, or stops cleanly
  at a decision point, such as two matching mugs.
- A sim draft cannot reach a competition by accident: a Competition run
  loads nothing from `bundled/`, `temporary/` or `quarantine/`.
- Mocks can drift from the real interfaces. They follow the ADR-0027
  outcomes, [Arm Manipulation](/architecture/manipulation.md) and
  `walkie_vision_msgs`, and bundle tests rerun on the real clients.

**See also:** [Self-Evolving Agent: tiers and folders](/architecture/self-evolving-agent.md#tiers-and-folders),
[Access Levels](/architecture/access-levels.md)
<!-- --8<-- [end:body] -->
