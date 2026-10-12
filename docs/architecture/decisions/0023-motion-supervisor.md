# ADR-0023: One MotionSupervisor owns every motion goal and the emergency stop

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

RAI checks for an interrupt only between graph steps, and runs all tool calls
from one LLM message inside a single step. So today, `/walkie/reset` waits
for a running motion to finish, and the remaining tool calls from the same
message still run. A reset does not cancel Nav2 at all.

Cancelling a commander goal stops the arm only through MoveIt's `"stop"`
event. The commander has no lock on the arm, so two goals at once would
fight.

**Decision**

A `MotionSupervisor` in the agent owns every motion goal: arm, gripper,
Nav2, and later skills.

- It allows one motion goal at a time. It refuses arm motion while Nav2 is
  driving, and navigation while an arm, gripper or skill goal is running.
- On reset, without blocking, it:
    1. sets a reset latch;
    2. publishes `"stop"` on `/trajectory_execution_event`, and repeats it
       every 200 ms until the cancelled goals end (at most 5 s);
    3. cancels every goal, including goals that are sent but not yet
       accepted;
    4. wakes the tool that is waiting, which reports "Stopped by reset".
- While the latch is set, every motion tool refuses at once. The latch clears
  when `WalkieAgent` applies the reset between tasks. This stops the other
  tool calls from the same LLM message.
- Every wait is bounded. On timeout it cancels the goal and waits up to 5 s
  for it to end.

**Consequences**

- A reset stops the robot, not only the conversation.
- `"stop"` stops all MoveIt executions on the robot, including ones started
  by other programs.
- The stop is best effort: it needs ROS, MoveIt and the controllers to be
  working. The physical emergency stop stays the real safety boundary.
- The commander accepts a gripper cancel but keeps moving the gripper
  (walkie-ros change W3). A full stroke takes under a second, so this is
  acceptable for now.
- The lift cannot be stopped this way
  ([ADR-0026](/architecture/decisions/0026-exclude-lift.md)).
<!-- --8<-- [end:body] -->
