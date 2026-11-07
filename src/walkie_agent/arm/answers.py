"""The wording and the rules that every ctx.arms backend and every arm atomic share.

RosArms (walkie_agent.ros.arms), FakeRobot's arm verbs and the arm atomics all
answer through these, so the fake refuses in the robot's words. drive_verdict is
the arm check before every base motion (ADR-0025), for RosBase and FakeRobot
alike. Plain logic: nothing here imports ROS.
"""

import logging
from collections.abc import Callable, Mapping

from walkie_agent.arm.poses import (
    ARMS,
    BOTH,
    GRASP_CLOSE_THRESHOLD,
    GRIPPER_STATES,
    ArmPose,
    GripperPositions,
    GripperState,
)
from walkie_agent.arm.results import NO_ANSWER, NO_READING, NOT_FOUND, DriveCheck
from walkie_agent.arm.state import ARMS_OUT, ArmsState
from walkie_agent.errors import ErrorCode, StepError

logger = logging.getLogger(__name__)

# The commander's status for a success in plan-only mode, where nothing moved.
PLAN_ONLY = "Plan Only"
PLAN_ONLY_TEXT = "The commander runs in plan-only mode, so it moved nothing"
# What to do about it: the commander reads plan_only once, at startup.
PLAN_ONLY_NEXT = (
    "No arm can move until the commander restarts without plan_only, so tell the person "
    "instead of trying again."
)

# The pose every drive refusal suggests: travel-safe, offered everywhere, and
# where the arms start.
SAFE_EXAMPLE = "home"

# Why a drive went ahead with the arms unread (DriveCheck.unread), for the log.
_UNREAD_WHY = {
    NOT_FOUND: "the commander's joint state service was not found",
    NO_ANSWER: "the commander did not answer the read",
    NO_READING: "the commander could not read the joints",
}


def pose_name(text: str) -> str:
    """A pose name as arm_poses.toml spells it: the LLM may write "Hands up" or "hands-up"."""
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def arms_text(pose: ArmPose) -> str:
    """Which values of move_arm_to_pose's `arm` the pose accepts, in words."""
    options = pose.arm_options
    if BOTH in options:
        return "left, right or both"
    if len(options) == len(ARMS):
        return "left or right, one arm at a time"
    return f"{options[0]} arm only"


def arm_refusal(pose: ArmPose) -> str:
    """Why the pose does not take the asked `arm`, such as "'receive' is for the left arm only."."""
    if len(pose.arms) == 1:
        return f"{pose.name!r} is for the {pose.arms[0]} arm only."
    # Both tables exist, but not as one both_arms goal (both_arms = false).
    return f"{pose.name!r} is for one arm at a time: left or right."


def offered_pose(
    offered: Mapping[str, ArmPose], pose: str, arm: str, sentence: Callable[[], str]
) -> ArmPose:
    """The offered pose that `pose` names, if it takes `arm`.

    Raises StepError NOT_FOUND for a pose not in `offered`: a hidden pose gets the
    same answer as an unknown one, so the LLM is never told about a pose it may not
    use. REFUSED for an `arm` the pose does not take. Both messages end with
    `sentence()`, which runs only then.
    """
    found = offered.get(pose_name(pose))
    if found is None:
        known = ", ".join(offered) or "none"
        raise StepError(
            ErrorCode.NOT_FOUND, f"No pose named {pose!r}. The poses are: {known}. {sentence()}"
        )
    if arm not in found.arm_options:
        raise StepError(ErrorCode.REFUSED, f"{arm_refusal(found)} {sentence()}")
    return found


def full_stop(text: str) -> str:
    """`text` ending with a full stop, unless it already ends a sentence.

    Some commander messages end with its own status text, which has none.
    """
    return text if text.endswith((".", "?", "!")) else f"{text}."


def with_sentence(error: StepError, prefix: str, sentence: str) -> StepError:
    """`error` with `prefix` before its message and `sentence` after it.

    Same code and same class, so a GoalFailed stays one.
    """
    return type(error)(error.code, f"{prefix}{full_stop(error.message)} {sentence}")


def unchecked(status: str, problem: str, sentence: str) -> StepError:
    """The SKILL_FAILED for a move that the commander reported done but the check after it failed.

    `status` is the commander's, `problem` says how the check failed, and
    `sentence` is the arm sentence. A plain StepError, not a GoalFailed: the goal
    ended, but the arms are not where it said.
    """
    if status == PLAN_ONLY:
        text = f"{PLAN_ONLY_TEXT}, and {problem} Treat this as a failure. {PLAN_ONLY_NEXT}"
    else:
        text = f"The commander reported success, but {problem} Treat this as a failure."
    return StepError(ErrorCode.SKILL_FAILED, f"{text} {sentence}")


def reads_as(positions: GripperPositions, finger_m: float) -> GripperState:
    """The named position, "open", "half" or "close", that a gripper read at `finger_m` counts as.

    "close" at or below GRASP_CLOSE_THRESHOLD: fingers stopped on an object may be
    nearer the half position than the closed one, but they closed. Above it, the
    nearer of "half" and "open", and "open" on a tie.
    """
    if finger_m <= GRASP_CLOSE_THRESHOLD:
        return "close"
    return min(("open", "half"), key=lambda state: abs(positions.position(state) - finger_m))


def commanded_state(
    positions: GripperPositions, position: float | None
) -> GripperState | None:
    """The named position whose value is exactly `position`, the last one sent, else None."""
    if position is None:
        return None
    for state in GRIPPER_STATES:
        if positions.position(state) == position:
            return state
    return None


def drive_refusal(state: ArmsState, what: str) -> StepError | None:
    """The PRECONDITION_FAILED for `what`, a base motion, while an arm is known to be out.

    None when no arm is known to be out of every travel-safe pose. The words use
    ARMS_OUT, so a check can find the refusal in a tool's answer by its text.
    """
    out = state.unsafe_arms
    if not out:
        return None
    if len(out) == len(ARMS):
        problem = f"the arms are {ARMS_OUT}. Move them"
    else:
        problem = f"the {out[0]} arm is {ARMS_OUT}. Move it"
    return StepError(
        ErrorCode.PRECONDITION_FAILED,
        f"Not starting {what}: {problem} with move_arm_to_pose first, for example "
        f"{SAFE_EXAMPLE!r}.",
    )


def drive_verdict(check: Callable[[], DriveCheck], what: str) -> None:
    """The arm check before `what`, a base motion (ADR-0025): returns to let it go, raises to refuse.

    `check` reads the arms, as CommanderClient.check_for_driving() does. It refuses
    with an arm out (drive_refusal), and with a reading that does not show both
    arms (INTERNAL). It fails open, with a WARNING in the log, when the arms were
    not read because the commander is missing or not answering, so navigation still
    works while the arm stack is down; the client itself never fails open while the
    arms are unverified, and raises PRECONDITION_FAILED then. A failure of the
    check is raised as a plain StepError, never a GoalFailed, so no caller can
    mistake it for a goal that ended: CANCELLED as it came, anything else with
    "Not starting {what}." before its message.
    """
    try:
        found = check()
    except StepError as error:
        if error.code is ErrorCode.CANCELLED:
            raise StepError(error.code, error.message) from None
        retry = " Try again in a few seconds." if error.code is ErrorCode.TIMEOUT else ""
        raise StepError(
            error.code, f"Not starting {what}. {full_stop(error.message)}{retry}"
        ) from None
    if found.state is None:
        why = _UNREAD_WHY.get(found.unread or "", "the arms were not read")
        logger.warning("Starting %s with the arm state unknown: %s.", what, why)
        return
    refusal = drive_refusal(found.state, what)
    if refusal is not None:
        raise refusal
    if found.state.travel_safe is None:
        raise StepError(
            ErrorCode.INTERNAL,
            f"Not starting {what}: the commander's reading does not show where both arms are. "
            "Tell the person that the arms cannot be checked.",
        )
