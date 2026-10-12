# ADR-0022: Offer the LLM named arm poses from config/arm_poses.toml

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

The old agent moved the arms by pose name, and that worked well. An LLM
cannot judge whether a set of joint values is safe. The SRDF already has some
named states (`home`, `standby`, `pre-place`), but adding one there needs a
walkie-ros change and a redeploy.

`/set_joint_position` needs the positions in the group's joint order, and
rejects the goal if the count is wrong.

**Decision**

Named poses live agent-side in `config/arm_poses.toml`:

- Each pose has a group (`left_arm`, `right_arm` or `both_arms`), joint
  values keyed by joint name, a description shown to the LLM, and a
  `travel_safe` flag. It may also name the SRDF state it copies, and set its
  own timeout.
- Gripper positions for open, half and closed are in the same file.
- At startup, the loader checks every pose against
  `config/robot_model.toml`: a committed snapshot of the SRDF groups and
  states and of `joint_limits.yaml`, generated from the pinned walkie-ros
  commit. A test fails when the snapshot drifts from the pin.
- At run time, the joint order comes from `/get_joint_states`, which returns
  the group's joint names in the order the commander expects.
- Every pose is sent with `/set_joint_position`, including SRDF ones.
  `/go_to_home` returns only a bool and falls back to all zeros for an
  unknown name.

**Consequences**

- Poses can be added without a robot redeploy, and a typo or an out-of-limit
  value fails at startup, not on the robot.
- Every pose must be tried on the robot before the LLM gets it. walkie-agent-v2
  sent some poses with collision checking turned off.
- Gripper values are the travel of one finger, 0 to 0.044 m; the jaw opens
  about twice that. "Closed" must be at most 0.03 m, the commander's
  `grasp_close_threshold`, so that grasp detection runs. "Half" must be above
  0.03 m, or the commander treats it as a close and may report a grasp.
- Lift groups are rejected for now
  ([ADR-0026](/architecture/decisions/0026-exclude-lift.md)).
<!-- --8<-- [end:body] -->
