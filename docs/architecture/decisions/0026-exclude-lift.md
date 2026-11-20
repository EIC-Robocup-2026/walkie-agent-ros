# ADR-0026: Leave the lift out until its action supports cancel

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

The lift's `FollowJointTrajectory` server runs in the `tmotor_ros2_bridge`
node (`bringup/scripts/lift_homing_node.py` in walkie-ros):

- It has no cancel callback, so every cancel request is rejected.
- When its 60 s timeout runs out, it reports success, whether or not the
  lift reached the target.
- It moves at 3 cm/s, so a long move takes many seconds.

MoveIt's `*_lift` groups move the lift through this server. A reset could not
stop the lift, and a "success" would not mean it arrived.

**Decision**

The agent rejects `*_lift` groups and any lift motion until walkie-ros fixes
the lift action (change W1): accept cancels and stop in place, and report
failure, not success, on timeout.

**Consequences**

- In the first phase the LLM cannot change the lift height. Arm poses must
  work at whatever height the lift is at.
- When W1 lands, a new ADR enables named lift heights (in metres) and
  supersedes this one.
- Exception, decision PK-5: `pick_object` moves the lift. Its straight moves
  go to the picking arm's `*_lift` group, and its move home sends that group a
  joint goal with the lift at its SRDF home height, 0.7435 m. The LLM still
  cannot choose a lift height, and no other tool moves the lift.
<!-- --8<-- [end:body] -->
