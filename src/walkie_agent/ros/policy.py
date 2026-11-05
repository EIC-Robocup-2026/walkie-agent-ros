"""What a capability endpoint may not name, and how a failed rclpy call becomes a StepError."""

import logging
from collections.abc import Iterable
from typing import Any

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy

from walkie_agent.errors import ErrorCode, StepError

logger = logging.getLogger(__name__)

# ROS names that no capability endpoint may use. This guards against a configuration
# mistake and is not a security boundary. The operator bridge owns the last three
# on purpose, and is not a capability.
#   /cmd_vel       raw velocity commands would bypass Nav2's obstacle avoidance
#   /walkie/reset  the agent must not wipe its own memory
#   /walkie/task   the agent must not queue its own tasks or seed locked steps
#   /walkie/todos  the operator's view of the list; only update_todos edits it
FORBIDDEN_NAMES = ("/cmd_vel", "/walkie/reset", "/walkie/task", "/walkie/todos")

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
