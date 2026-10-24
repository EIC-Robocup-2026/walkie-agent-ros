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
    STOP_HEARD,
    STOP_LOOKING,
    STOP_UNCONFIRMED_NEXT,
    STOP_WAIT_S,
    GazeLoop,
)
from walkie_agent.toolbox import ToolContext

# The backend name in META.requires and ctx.backend().
GAZE = "gaze"

# look_at's stop reason when the base does not face a seat in time.
TURN_TIMEOUT = "turn timeout"

# What follows the loop's message in an arm atomic's refusal when
# stop_before_arm_motion raises.
NO_ARM_GOAL = f"No arm goal was sent. {STOP_UNCONFIRMED_NEXT}"

# Why the loop stopped, as the end of "The last target, per_51c0.1, was
# stopped 2 s ago ..." and of "Stopped turning toward seat_77aa.2 ...".
# LOST is not here: stop_looking says "was lost" instead.
_CAUSES = {
    STOP_LOOKING: "with stop_looking",
    INTERRUPT: "by a stop or a reset",
    STOP_HEARD: 'because someone said "stop" or "never mind"',
    NEW_TASK: "when a new task started",
    # Before the drive's arm check, so also for a drive that never ran.
    NAVIGATION: "when a drive was asked for",
    NAVIGATION_ACTIVE: "because a navigation goal became active",
    # Before an arm atomic's first commander call, and from the loop when an
    # arm became unknown, or a commander goal ran with an arm unknown, which
    # may be a gripper's (busy()).
    ARM_MOTION: "because an arm or a gripper was about to move, or may have been moving",
    # From the commander's last reading, which does not say when the arm went out.
    ARM_OUT: "because an arm was out of every pose that is safe for driving",
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
    the whole motion. The gaze thread publishes the stop's zeros, and this
    waits up to STOP_WAIT_S for them, so the base has been told to stop
    before the arm leaves its pose. When they are not published by then,
    it raises the loop's StepError TIMEOUT: the base may still be turning,
    so the caller refuses and sends nothing to the commander; its answer
    follows the loop's message with NO_ARM_GOAL. The loop is stopped either
    way. The grippers do not move the arm and do not stop it.
    """
    if ctx.has_backend(GAZE):
        gaze_loop(ctx).stop(ARM_MOTION, wait_s=STOP_WAIT_S)


def stop_cause(reason: str) -> str:
    """Why the loop stopped for `reason`, as the words that end a sentence about the stop."""
    return _CAUSES.get(reason, f"({reason})")
