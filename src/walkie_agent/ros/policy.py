"""What a capability endpoint may not name, and how a failed rclpy call becomes a StepError."""

import logging
from collections.abc import Iterable
from typing import Any

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy

from walkie_agent.errors import ErrorCode, StepError

logger = logging.getLogger(__name__)

# ROS names that no capability endpoint may use. This guards against a configuration
# mistake and is not a security boundary. A name matches only exactly, once the node
# has resolved it. The operator bridge owns the /walkie names on purpose, and is not
# a capability.
#   /cmd_vel and the other base velocity inputs: raw velocity commands would bypass
#                  Nav2's obstacle avoidance. /cmd_vel_nav is Nav2's own input to
#                  twist_mux, /cmd_vel_joy and /cmd_vel_key the operator's, and
#                  /omni_wheel_drive_controller/cmd_vel twist_mux's output, which
#                  drives the wheels directly
#   /go_to_home and /go_to_pose*: the commander's other motion actions. The arms
#                  move only through /set_joint_position to a named pose, whose
#                  joints the agent checks, and /go_to_pose* take Cartesian targets
#                  that ADR-0021 reserves for robot-side skills
#   /lift_controller/follow_joint_trajectory, /lift_controller/commands and
#                  /lift/cmd: the lift, which the agent never commands, since its
#                  action rejects every cancel (ADR-0026)
#   /walkie/reset  the agent must not wipe its own memory
#   /walkie/task   the agent must not queue its own tasks or seed locked steps
#   /walkie/todos  the operator's view of the list; only update_todos edits it
# /trajectory_execution_event is not listed: CommanderClient publishes "stop" there,
# as the one publisher while no MotionSupervisor exists.
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

# What an rclpy call raises once its handle or its context is gone.
ROS_GONE = (_rclpy.RCLError, _rclpy.InvalidHandle)


def check_endpoint(node: Any, name: str, forbidden: Iterable[str] = FORBIDDEN_NAMES) -> str:
    """Return `name` as the node resolves it, or raise StepError REFUSED for a forbidden one.

    Both sides go through the node, so a relative name or a remap cannot hide a match.
    """
    resolved = node.resolve_topic_name(name)
    if resolved in {node.resolve_topic_name(entry) for entry in forbidden}:
        raise StepError(ErrorCode.REFUSED, f"Refused: a capability may not use {resolved}.")
    return resolved


def ros_failure(error: Exception, what: str) -> StepError:
    """The StepError for an exception from an rclpy call, which the caller raises.

    A handle or a context that is gone means ROS is unavailable. Anything else is a
    bug, so the log gets the traceback and the model gets only the type.
    """
    if isinstance(error, ROS_GONE):
        return StepError(ErrorCode.UNAVAILABLE, f"{what} failed: ROS is not available.")
    logger.error("%s failed.", what, exc_info=error)
    return StepError(ErrorCode.INTERNAL, f"{what} failed with {type(error).__name__}.")
