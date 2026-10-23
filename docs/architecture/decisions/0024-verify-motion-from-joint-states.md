# ADR-0024: Verify arm motion from joint states, not goal status

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

The commander's result can be misleading:

- In plan-only mode it reports success with status `"Plan Only"` and does not
  move.
- A cancel during execution is reported as ABORTED `"Execution Failed"`,
  and a cancel during planning as CANCELED `"Canceled"`. Either way the
  result does not say where the arms stopped.
- A close it did not measure, with grasp detection off or before its first
  finger reading, reports `grasped` false and a gap of 0, whatever the
  fingers hold. So does a close that a newer command for the gripper
  superseded, also one from another node, and the fingers then show the
  newer command.

A reading can mislead as well. A position that is not a number passes a
tolerance check, since every comparison with NaN is false, and a reading
that lacks an arm's joints says nothing about that arm. The gripper's
`grasped` flag can also be wrong. If the gripper controller is
down, the commander reports a failure with the fingers still open, and that
gap falls inside the "holding" band. LLMs tend to believe whatever a tool
returns.

`/get_joint_states` also answers for the gripper groups. Each has one
joint, the finger whose travel the commander's grasp detection reads.

**Decision**

- After every arm goal, the tool reads `/get_joint_states` for the group and
  requires every joint to be within 0.05 rad of the target. Otherwise it
  reports failure, naming the joint that is off.
- A reading is used only when every position is a finite number. Otherwise
  the read fails with `INTERNAL`, and the arms it covers become unknown. An
  arm whose joints a reading lacks becomes unknown too, and the read of
  both arms' poses that `get_arm_state` and the drive check use then fails
  with `INTERNAL`. Only a read of one arm or a gripper leaves the other
  arm's state as it was.
- Every arm tool ends its answer with the arm state: "safe for driving",
  "not safe for driving" or "state unknown".
- A move that reports `"Plan Only"` never says the arms moved. When the
  check passes, the answer says the arms were already in the pose; when it
  fails, it says the commander runs in plan-only mode.
- An open or a "half" that the commander reports failed is a failure. After
  a close that the commander reports failed with `grasped` false and a gap
  of 0, `set_gripper` does not read the finger and says that it cannot
  tell. After any other close it reads the finger, but only while the close
  is still the gripper's latest command from the agent; after a newer one,
  it says that it cannot tell. Within 4 mm of fully open, the gripper did
  not close. Otherwise it reports "holding" from `grasped` when the
  commander's gap is within 2 mm of the read, and from the commander's
  default detection band applied to the read when it is not. When the
  finger cannot be read, it says that it cannot tell. The detection band is
  calibrated on the competition objects.
- `get_arm_state` reads both fingers, and names each gripper closed, half
  open or open, with the gap. Only when a read fails does it give the last
  command instead.

**Consequences**

- The LLM gets what the robot actually did, not what the commander reported.
- Each motion costs one extra service call, and each close at most one,
  bounded at 3 s, and `get_arm_state` makes three, or four while the arms
  are unverified.
- `/get_joint_states` calls MoveIt's `getCurrentState()`. A comment in the
  commander says that call is not thread-safe, and that concurrent calls
  crash the node. Until that is checked, the agent reads joint states only
  when no goal is running, and `get_arm_state` refuses during motion. A
  goal whose call gave up, after a timeout or a cancel, still counts until
  the commander reports that it ended. The client stops waiting for that
  report once the commander's action server has stayed undiscovered for
  5 s, or 60 s after the call gave up. Neither bounds the motion, since
  MoveIt lets a goal run for twice its planned duration plus 60 s, so the
  arms are then unverified: the client refuses every goal, and
  `get_arm_state` reads both arms twice, 0.3 s apart. Two readings that
  agree within 0.01 rad clear that state.
<!-- --8<-- [end:body] -->
