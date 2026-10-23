"""Shared by the gaze atomics: the backend name, the arm-motion stop and why the loop stopped.

look_at and stop_looking reach the gaze loop, walkie_agent.gaze.GazeLoop,
only through the `gaze` backend (ADR-0051), and move_arm_to_pose stops it
through the same backend before an arm moves. The module's name starts with
"_", so the registry does not load it as a tool.
"""

from walkie_agent.gaze import (
    ARM_MOTION,
    ARM_OUT,
    ERROR,
    INTERRUPT,
    NAVIGATION,
    NAVIGATION_ACTIVE,
    NEW_TASK,
    SHUTDOWN,
    STOP_LOOKING,
    GazeLoop,
)
from walkie_agent.toolbox import ToolContext

# The backend name in META.requires and ctx.backend().
GAZE = "gaze"

# look_at's stop reason when the base does not face a seat in time.
TURN_TIMEOUT = "turn timeout"

# Why the loop stopped, as the end of "The last target, per_51c0.1, was
# stopped 2 s ago ..." and of "Stopped turning toward seat_77aa.2 ...".
# LOST is not here: stop_looking says "was lost" instead.
_CAUSES = {
    STOP_LOOKING: "with stop_looking",
    INTERRUPT: "by a stop or a reset",
    NEW_TASK: "when a new task started",
    NAVIGATION: "before a drive",
    NAVIGATION_ACTIVE: "because a navigation goal became active",
    ARM_MOTION: "before an arm moved",
    ARM_OUT: "because an arm left the poses that are safe for driving",
    SHUTDOWN: "at shutdown",
    ERROR: "after an error in the gaze loop",
    TURN_TIMEOUT: "because the turn took too long",
}


def gaze_loop(ctx: ToolContext) -> GazeLoop:
    return ctx.backend(GAZE)


def stop_before_arm_motion(ctx: ToolContext) -> None:
    """Stop the gaze loop, when the agent has one, before an arm atomic moves an arm.

    The loop turns the base only with both arms in travel-safe poses
    (ADR-0051). Stopping before the goal is sent, rather than when the
    commander's next reading shows the arm moving, keeps the base still for
    the whole motion. The grippers do not move the arm and do not stop it.
    """
    if ctx.has_backend(GAZE):
        gaze_loop(ctx).stop(ARM_MOTION)


def stop_cause(reason: str) -> str:
    """Why the loop stopped for `reason`, as the words that end a sentence about the stop."""
    return _CAUSES.get(reason, f"({reason})")
