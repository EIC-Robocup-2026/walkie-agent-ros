# ADR-0024: Verify arm motion from joint states, not goal status

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

The commander's result can be misleading:

- In plan-only mode it reports success with status `"Plan Only"` and does not
  move.
- A cancel during execution is reported as ABORTED `"Execution Failed"`.

The gripper's `grasped` flag can also be wrong. If the gripper controller is
down, the commander reports a failure with the fingers still open, and that
gap falls inside the "holding" band. LLMs tend to believe whatever a tool
returns.

**Decision**

- After every arm goal, the tool reads `/get_joint_states` for the group and
  requires every joint to be within 0.05 rad of the target. Otherwise it
  reports failure, naming the joint that is off.
- Every arm tool ends its answer with the arm state: "safe for driving",
  "not safe for driving" or "state unknown".
- `set_gripper` reports "holding" only when `grasped` is true and the finger
  gap is clearly below fully open. The detection band is calibrated on the
  competition objects.

**Consequences**

- The LLM gets what the robot actually did, not what the commander reported.
- Each motion costs one extra service call, bounded at 3 s.
- `/get_joint_states` calls MoveIt's `getCurrentState()`. A comment in the
  commander says that call is not thread-safe, and that concurrent calls
  crash the node. Until that is checked, the agent reads joint states only
  when no goal is running, and `get_arm_state` refuses during motion.
<!-- --8<-- [end:body] -->
