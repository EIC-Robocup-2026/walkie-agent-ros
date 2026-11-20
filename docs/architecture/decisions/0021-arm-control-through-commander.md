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
emergency stops. The LLM never chooses joint values or coordinates.

The demo's client (`ros/commander.py`) also reads two of the commander's
parameters, `grasp_detect_min` and `grasp_detect_max`, from
`/bimanual_commander/get_parameters`, to judge a close by the commander's own
grasp band, and at startup `allow_gripper_vs_octomap`. It reads nothing else
there. For `pick_object` only (decision PK-4), it sets one parameter through
`/bimanual_commander/set_parameters`: `allow_gripper_vs_octomap`, which lets the
fingers touch the sensed octomap during the straight approach, the lift and the
move clear of the surface. It stays on through the move of the picking arm home,
which asks for it, and the next joint move or reach without it, or the next open
of a gripper, clears it. The pick sets it back to false, the commander's default,
however its grasp ends, a stop included, and the agent does so at startup when
an earlier run left it on. A startup read or reset that fails is logged, not
raised: the commander's value stays as it is, possibly on, and the agent holds it
as unknown, so the first motion of an arm or a gripper sets it before its goal
(false, for a gripper) and sends no goal unless the commander confirms that. The
agent sends every change of it through one client, one at a time under a lock of
its own, and the reset reads the value back after it settles: a true read then,
as from a change that reached the commander late, fails the reset and leaves the
value unknown. The read checks the parameter, not the planning scene the
commander builds from it. It sets nothing else.

**Consequences**

- MoveIt plans every motion with collision checking. Named states and joint
  limits stay robot-side.
- The first phase needs no walkie-ros changes.
- The agent inherits the commander's behavior, which the tools must handle:
    - no feedback while a goal runs;
    - a cancel during execution is reported as ABORTED `"Execution Failed"`;
    - `plan_only` is read once at startup, and the launch file cannot set it;
    - velocity and acceleration scaling are hard-coded to 1.0 (full speed);
    - each goal runs in its own thread with no lock on the arm, so two goals
      at once would fight;
    - held objects are not added to the planning scene unless the
      `grasp_scene_action` parameter is changed.
- Every other commander, `move_group` and controller name goes on the
  forbidden list
  ([list](/architecture/manipulation.md#forbidden-names)).
<!-- --8<-- [end:body] -->
