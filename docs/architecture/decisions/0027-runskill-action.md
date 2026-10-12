# ADR-0027: Run manipulation skills, including VLA policies, behind a robot-side RunSkill action

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

Tasks such as picking up an object or folding a shirt cannot be expressed as
a few named poses. Some will be scripted with MoveIt and perception. Others
will be learned policies, such as a vision-language-action (VLA) model. A
policy needs:

- control of the arm motors through ros2_control, at a steady rate;
- the base and wrist cameras;
- a GPU, and Python packages the agent's environment cannot hold
  ([ADR-0013](/architecture/decisions/0013-cuda-torch.md)).

The LLM should pick what to do, not how it is done.

**Decision**

- A robot-side skill server in walkie-ros offers a `RunSkill` action and a
  `ListSkills` service.
- The agent gets two tools: `list_skills` and
  `run_skill(skill, target, arm)`.
- The `RunSkill` result has an outcome (`SUCCEEDED`,
  `FINISHED_UNVERIFIED`, `FAILED`, `PRECONDITION_FAILED`, `CANCELED`,
  `SAFETY_STOP`, `TIMED_OUT`) and a `success_verified` flag. The flag is true
  only when a check independent of the policy confirms success.
- Whether a skill runs on MoveIt or on a policy is the skill server's
  configuration. The LLM never sees it.
- The skill server moves to the skill's start pose through the commander,
  then hands control to the policy. The first choice is streaming joint
  trajectories to the controllers MoveIt already uses. The fallback is
  switching to forward position controllers. A safety filter clamps every
  command, and a cancel stops the policy and restores the controllers within
  2 s.

**Consequences**

- A skill can move from scripted to learned without any agent change.
- It needs walkie-ros work: the interfaces (W4), the skill server (W5), the
  policy backend (W7), and forward controllers if the fallback is used (W6).
- The policy acts through ros2_control, not through LeRobot's OpenArm driver,
  which talks to the motors over CAN itself and would fight ros2_control.
- A policy adapter converts between the policy's layout and the robot's. For
  example, LeRobot `folding_latest` uses 16 values ordered right arm joints
  1-7, right gripper, left arm joints 1-7, left gripper, in degrees, with
  relative actions and a gripper range of -65 to 0.
- The wrist cameras have device names on the robot, but no ROS node
  publishes them yet.
<!-- --8<-- [end:body] -->
