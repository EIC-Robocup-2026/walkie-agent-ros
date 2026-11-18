# 5. Limit What Tools Can Touch

**Goal:** make sure no tool can use a ROS topic it should never touch, such as
`/cmd_vel`.

## Background

- The agent acts only through its atomics, and an atomic reaches the robot only
  through capability verbs on `ctx`. No tool gets a topic name, a ROS message or
  the connector.
- There is no velocity verb and no generic publish tool. RAI's generic ROS 2 tools,
  such as `PublishROS2MessageTool`, are not bound.
- Each ROS-backed verb uses one fixed endpoint. `RosBase` uses the
  `/navigate_to_pose` action and TF. `RosSpeech` uses `/to_human` and
  `/voice_commands`.
- A verb checks its own arguments, through `walkie_agent.capabilities.guards`. A
  relative move is limited to 3 m, and every wait has a bound.
- Arm verbs have no ROS endpoint yet, and `UnavailableArms` raises `UNAVAILABLE`.
  Vision verbs raise `UNAVAILABLE` through `UnavailableVision`, unless the agent runs
  with `--vision`. Then `RosVision` subscribes to `/walkie_vision/people`, a name that
  goes through the same forbidden-name check, and only reads.

!!! warning "This is a guard rail, not security"
    These are ordinary Python checks in one process. They are no sandbox, and they
    limit nothing that does not go through them. Real access control in ROS 2 is done
    with [SROS2](https://docs.ros.org/en/jazzy/Tutorials/Advanced/Security/Introducing-ros2-security.html)
    ([ADR-0009](../architecture/decisions/0009-forbidden-names-guard-rail.md)).

## Step 1: One deny list for every capability endpoint

The list is in `src/walkie_agent/ros/policy.py`. Every ROS-backed capability,
`RosBase`, `RosSpeech`, `RosVision`, the people memory and the arm commander's
client, calls `check_endpoint` for each name it uses when it is built:

```python title="src/walkie_agent/ros/policy.py"
FORBIDDEN_NAMES = (
    "/cmd_vel",
    "/cmd_vel_nav",
    "/cmd_vel_joy",
    "/cmd_vel_key",
    "/omni_wheel_drive_controller/cmd_vel",
    "/go_to_home",
    "/go_to_pose",
    "/go_to_pose_relative",
    "/go_to_pose_quat",
    "/lift_controller/follow_joint_trajectory",
    "/lift_controller/commands",
    "/lift/cmd",
    "/walkie/reset",
    "/walkie/task",
    "/walkie/todos",
)


def check_endpoint(node, name, forbidden=FORBIDDEN_NAMES):
    resolved = node.resolve_topic_name(name)
    if resolved in {node.resolve_topic_name(entry) for entry in forbidden}:
        raise StepError(ErrorCode.REFUSED, f"Refused: a capability may not use {resolved}.")
    return resolved
```

- `/cmd_vel` and the other base velocity inputs: raw velocity commands would bypass
  Nav2's obstacle avoidance. `/cmd_vel_nav` is Nav2's own input to `twist_mux`,
  `/cmd_vel_joy` and `/cmd_vel_key` the operator's, and
  `/omni_wheel_drive_controller/cmd_vel` drives the wheels directly.
- `/go_to_home` and the three `/go_to_pose*` actions: the commander's other motions.
  The arms move only through `/set_joint_position` to a named pose
  ([ADR-0021](../architecture/decisions/0021-arm-control-through-commander.md)).
- The three lift names: the agent never commands the lift, whose action rejects every
  cancel.
- `/walkie/reset`: the agent must not wipe its own memory.
- `/walkie/task`: the agent must not queue its own tasks or seed locked steps.
- `/walkie/todos`: this is the operator's view, and only `update_todos` edits the list.

`/trajectory_execution_event` is not on the list: the commander client publishes
`"stop"` there.

Both sides go through the node, so a relative name or a remap cannot hide a match.
The operator bridge, `RosBridge`, owns the three `/walkie` names on purpose. It is not a
capability, and it does not pass through this check. It has one check of its own: a
console topic that resolves to `/to_human` is refused at startup, so replies never reach
the speaker.

## Step 2: Check it

```bash
pixi run test
```

`tests/test_ros_base.py` and `tests/test_ros_speech.py` build the verbs on fake ROS
objects and try forbidden names. `tests/test_ros_integration.py` runs the assembled
agent on real rclpy against a fake robot, and checks that closing it removes its topics.

None of this has run on the robot's own graph.

## Adding a new tool

Most new tools are a new atomic over the verbs that exist:

1. Write a plain typed function in `src/walkie_agent/tools/atomic/`. It takes `ctx` first and
   raises `StepError`. Describe every parameter in its Google-style docstring.
2. Add it to the set that should have it in `src/walkie_agent/tools/sets.py`: `GPSR`, `HRI`
   or `DEFAULT`. The sets are plain
   lists, and a duplicate name is refused at build
   ([ADR-0067](../architecture/decisions/0067-static-tool-sets.md)).
3. Test it against `FakeRobot`, in `walkie_agent.capabilities.fake`.

A new way to reach the robot is a new verb:

1. Add the verb to the capability's Protocol in `walkie_agent/capabilities/`.
2. Implement it in `FakeRobot` and in `walkie_agent/ros/`, with the shared guards.
3. Call `check_endpoint` for any name it uses.
4. If you add a forbidden name, say why in the comment above the list.

An unavailable capability must keep failing with `UNAVAILABLE` until a real backend is
built. Never return an empty list or a fake result from the ROS `ctx`.
