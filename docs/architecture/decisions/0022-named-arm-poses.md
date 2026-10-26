# ADR-0022: Offer the LLM named arm poses from config/arm_poses.toml

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

The old agent moved the arms by pose name, and that worked well. An LLM
cannot judge whether a set of joint values is safe. The SRDF has seven arm
presets that the team tuned on the robot and walkie-agent-v2 used in
competition: `home`, `hands_up`, `standby`, `pre-place`, `tray`, `basket`
and `basket_hold`. Most exist for the left arm, the right arm and both, and
walkie-agent-v2 often moved one arm alone. Adding a preset there needs a
walkie-ros change and a redeploy.

`/set_joint_position` needs the positions in the group's joint order, and
rejects the goal if the count is wrong.

**Decision**

Named poses live agent-side in `config/arm_poses.toml`:

- Each pose has a joint table for the left arm, the right arm or both,
  keyed by joint name, a description shown to the LLM, and a `travel_safe`
  flag. It may also name the SRDF state it copies and set its own timeout.
  Three more keys decide where it is offered:
    - `offered = false` keeps it from the LLM everywhere.
    - `tried_on_robot = false`, the default, marks a pose not yet run on
      the robot with collision checking on. The LLM gets it only on sim
      time (`--use-sim-time`), where a bad pose costs nothing.
    - `both_arms = false` marks two tables that must not move together, as
      in `carry`, where both hands would meet in front of the body.
- `move_arm_to_pose(pose, arm)` takes `arm` as left, right or both, default
  both, like `set_gripper`. Both sends the two tables as one `both_arms`
  goal, and only for a pose with both tables and `both_arms = true`. A pose
  without a table for the requested arm is refused.
- The first draft copies the SRDF presets' `left_arm` and `right_arm`
  states, without lift heights, and splits `tray`, which exists only for
  both arms, into a left and a right table. `basket` and `basket_hold` are
  not offered until named lift heights land. `pre-place` becomes
  `pre_place`, and is not offered either, since it is a step of the place
  skill.
- Gripper positions for open, half and closed are in the same file.
- At startup, the loader checks every pose against
  `config/robot_model.toml`: a committed snapshot of the SRDF groups and
  states and of `joint_limits.yaml`, generated from the pinned walkie-ros
  commit by `scripts/snapshot_robot_model.py`. A test regenerates it from
  the pinned clone and fails when it drifts, and another fails when a table
  that names an SRDF state no longer matches it. Without the clone the
  first test is skipped, except in CI, where it fails. The script stops on
  a group or a group state inside a xacro element it does not resolve.
- At run time, the joint order comes from `/get_joint_states`, which returns
  the group's joint names in the order the commander expects. If it lists
  no `openarm_right_joint2` (`right_joint2_fixed`), the tool drops that
  joint, which every preset has at 0.
- Every pose is sent with `/set_joint_position`, including SRDF ones.
  `/go_to_home` returns only a bool and falls back to all zeros for an
  unknown name.

**Consequences**

- Poses can be added without a robot redeploy, and a typo or an out-of-limit
  value fails at startup, not on the robot.
- The presets the team already trusts are the starting set, and a retune in
  walkie-ros is copied on purpose, never silently.
- Because a pose can move one arm, the drive check and `get_arm_state` match
  each arm on its own
  ([ADR-0025](/architecture/decisions/0025-travel-safe-interlock.md)).
- Both-arm `basket` and `pre-place` in the SRDF keep the right arm in
  `standby`. The agent reaches them in two moves, and moving both arms to
  `basket` is a new pairing that must be tried first.
- Every pose must be tried on the robot before the LLM gets it there.
  walkie-agent-v2 sent some poses with collision checking turned off. A
  pose recorded in the simulation, such as `receive` or `carry`, is offered
  on sim time until someone tries it on the robot and sets
  `tried_on_robot = true`.
- Gripper values are the travel of one finger, 0 to 0.044 m; the jaw opens
  about twice that. "Closed" must be at most 0.03 m, the commander's
  `grasp_close_threshold`, so that grasp detection runs. "Half" must be above
  0.03 m, or the commander treats it as a close and may report a grasp.
- Lift groups are rejected for now
  ([ADR-0026](/architecture/decisions/0026-exclude-lift.md)).
<!-- --8<-- [end:body] -->
