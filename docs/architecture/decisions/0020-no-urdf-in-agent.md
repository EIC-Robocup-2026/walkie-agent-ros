# ADR-0020: The agent does not load the robot's URDF

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

RAI's whoami tutorial suggests adding the robot's URDF. In rai_whoami 2.12,
the build reads `whoami/urdfs/*.urdf` and `*.xacro` into memory, but nothing
uses them: neither the generated description nor the documentation index
changes. (RAI's folder initializer even creates `urdf/`, a folder the build
never reads.)

MoveIt does need the URDF, for kinematics, joint limits and collision
geometry. It already has it on the robot, together with the SRDF's groups and
named states. Walkie's URDF is also large, mostly meshes and inertias, and
would be poor input for an LLM.

**Decision**

- The agent does not load the URDF, and `whoami/urdfs/` stays unused.
- The arm facts the LLM needs go in a hand-written
  `whoami/documentation/arms.md`: two 7-DOF arms on a lift, the gripper
  opening, the named poses, and the rule that Walkie drives only with its
  arms tucked.
- Questions about geometry, such as "can I reach that?", go to robot-side
  services or skill preconditions, not to the LLM.

**Consequences**

- There is no URDF copy to keep in sync with walkie-ros.
- Named poses are checked against a snapshot of the SRDF and joint limits,
  not the URDF
  ([ADR-0022](/architecture/decisions/0022-named-arm-poses.md)).
- If a later rai_whoami version uses URDFs, revisit this decision.
<!-- --8<-- [end:body] -->
