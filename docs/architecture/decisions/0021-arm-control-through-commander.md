# ADR-0021: Move the arms only through bimanual_commander

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

walkie-ros runs MoveIt and `bimanual_commander`
(`manipulate/openarm_bimanual_commander_cpp`). The commander offers, among
others:

- the actions `/set_joint_position`, `/go_to_home`, `/control_gripper` and
  `/go_to_pose*` (Cartesian targets),
- the service `/get_joint_states`, and
- stopping through MoveIt: `"stop"` on `/trajectory_execution_event` stops
  every running MoveIt execution.

walkie-sdk used the same names. Other ways to move the arms:

| Option | Why not |
| ------ | ------- |
| MoveIt's `MoveGroup` action directly | Duplicates the commander, needs `moveit_msgs` in the agent, and two clients would drive one `move_group`. |
| `FollowJointTrajectory` on the controllers | No planning and no collision checking: motion chosen by an LLM would go straight to the motors. |
| RAI's `MoveToPointTool` | Calls `/manipulator_move_to`, which walkie-ros does not provide. Single Cartesian target, chosen by the LLM, with no cancel. |
| The commander's `/go_to_pose*` | Cartesian targets chosen by the LLM. Reserved for robot-side skills with targets from perception. |

**Decision**

The agent's arm tools use only `/set_joint_position`, `/control_gripper` and
`/get_joint_states`, plus `"stop"` on `/trajectory_execution_event` for
emergency stops. They also read the commander's grasp detection band,
`grasp_detect_min` and `grasp_detect_max`, from
`/bimanual_commander/get_parameters`, and read nothing else there. The LLM
never chooses joint values or coordinates.

**Consequences**

- MoveIt plans every motion with collision checking. Named states and joint
  limits stay robot-side.
- The first phase needs no walkie-ros changes.
- The agent inherits the commander's behavior, which the tools must handle:
    - no feedback while a goal runs;
    - a `/set_joint_position` cancel stops both arms, not only the goal's
      group, and is then accepted: MoveIt's stop is global;
    - a cancel during planning ends the goal CANCELED `"Canceled"` once the
      plan is made, and a cancel during execution is reported as ABORTED
      `"Execution Failed"`;
    - a `/control_gripper` cancel is accepted, but the gripper finishes its
      current motion before the goal ends `"Canceled"`;
    - a `/control_gripper` goal ends ABORTED `"Failed"` whenever the
      controller reports no success, as when the fingers stall on an
      object, and it reports `grasped` and the gap only for a close it
      measured, so an unmeasured close reads as false and 0;
    - a newer `/control_gripper` goal for the same gripper, from any node,
      ends an older one that has not sent its final setpoint yet; the older
      goal then ends ABORTED `"Failed"` with `grasped` false and a gap of 0,
      or CANCELED `"Canceled"` when it was being cancelled, so a superseded
      close reads like an unmeasured one;
    - `plan_only` is read once at startup, and the launch file cannot set it;
    - velocity and acceleration scaling are hard-coded to 1.0 (full speed);
    - each goal runs in its own thread with no lock on the arm, so two goals
      at once would fight;
    - held objects are not added to the planning scene unless the
      `grasp_scene_action` parameter is changed;
    - the grasp detection band is read afresh on every close, so it can
      change while the commander runs. `set_gripper` reads it for a close
      it judges from its own read, and uses its copy of the default band
      only when that read fails
      ([ADR-0024](/architecture/decisions/0024-verify-motion-from-joint-states.md)).
- Every other commander, `move_group` and controller name goes on the
  forbidden list
  ([list](/architecture/manipulation.md#forbidden-names)).
<!-- --8<-- [end:body] -->
