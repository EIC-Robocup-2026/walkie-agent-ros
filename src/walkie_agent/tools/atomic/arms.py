"""Atomics that move the arms to named poses, set the grippers and say where the arms are.

They reach the arms only through ctx.arms (capabilities.Arms), which holds the steps,
the waits and the joint checks; these atomics word its answers. The LLM picks poses by
name and never sees joint values (ADR-0022). Every answer, success or failure, ends
with the arm sentence, so the LLM always knows whether the robot may drive (ADR-0024).

Ported from experiment/hri-agent at 61d30e8, without the gaze loop it coordinated with
(decision S6-5). An error that ctx.arms raises reaches the LLM as it came, since its
message already ends with the arm sentence; set_gripper and get_arm_state add what
they did or found before it.

An open or a half open that the commander reports done also tells the object memory
that the hand let go of what it held (tools.atomic.held, decision PK-70): a handover
when that arm rests in the standby pose, else a place somewhere the memory does not
know. The record is best-effort and changes no answer.
"""

from walkie_agent.arm import answers
from walkie_agent.arm.poses import ARMS, BOTH, GRIPPER_GROUPS
from walkie_agent.arm.results import Recovery
from walkie_agent.arm.state import UNKNOWN_SENTENCE, ArmsState
from walkie_agent.capabilities import Arm, ArmsReport, Ctx, GripperReading, GripperState
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.tools.atomic.held import release

# The pose a guest hands the robot an object in, and how long a close there waits
# before its goal is sent. The close follows a spoken cue such as "I will close my hand
# in 3 seconds.", and say returns once the line is published, not once it has played,
# so a close sent at once would start while the cue still plays. The cue says its
# interval about 1.5 s into a line of about 3 s, and the first audio comes a fraction
# of a second after the publish; the robot check tunes it. set_gripper's docstring
# gives the delay in words, and a test keeps the two equal.
RECEIVE_POSE = "receive"
RECEIVE_CLOSE_DELAY_S = 5.0

# What an open or "half" that the commander reports failed did not do.
_NOT_REACHED = {"open": "did not open fully", "half": "did not reach half open"}

# How each named gripper position reads in get_arm_state's answer.
_STATE_WORDS = {"open": "open", "half": "half open", "close": "closed"}

# How the commander showed that a goal from an earlier call is no longer active, by
# its `how` in Recovery.goals. Both "result" and "ended" are its report that the goal
# ended: its result, or its answer to a cancel.
_RESOLVED = {
    "result": "The commander confirmed that {what} from an earlier call ended",
    "ended": "The commander confirmed that {what} from an earlier call ended",
    "rejected": "The commander rejected {what} from an earlier call",
    "unknown": "The commander does not know {what} from an earlier call, as after a restart",
}


def list_arm_poses(ctx: Ctx) -> str:
    """List the named arm poses you can use with move_arm_to_pose.

    Returns one line per pose. Each line gives the pose's name, which arms it can
    move (left, right or both), whether the robot may drive with an arm in it, and
    what it looks like. The last line says whether the arms are safe for driving now.
    """
    offered = ctx.arms.poses()
    lines = [
        f"{name}: {answers.arms_text(pose)}; "
        f"{'safe' if pose.travel_safe else 'not safe'} for driving. {pose.description}"
        for name, pose in offered.items()
    ]
    if not lines:
        lines = ["No arm poses are offered."]
    lines.append(ctx.arms.state().sentence())
    return "\n".join(lines)


def move_arm_to_pose(ctx: Ctx, pose: str, arm: Arm = "both") -> str:
    """Move one arm or both to a named pose and wait until they arrive, usually 5 to 10 seconds.

    Use a pose name from list_arm_poses. arm is left, right or both (the default);
    list_arm_poses says which arms each pose can move. Tell the people near you
    before the arms move. Returns a sentence that says where the arms moved, which
    ends with whether the arms are safe for driving.

    Args:
        pose: The pose's name, as list_arm_poses gives it, such as 'home'.
        arm: The arm to move: left, right or both.
    """
    moved = ctx.arms.move_to(pose, arm)
    name = moved.pose
    # The sentence from the one report taken after the joint check, so that a goal
    # from an earlier call that the commander accepts late cannot come between them.
    sentence = moved.report.sentence()
    if moved.status == answers.PLAN_ONLY:
        # Nothing moved, so the check passed because the arms were already there.
        where = f"Already at {name!r}." if arm == BOTH else f"The {arm} arm is already at {name!r}."
        return f"{where} {answers.PLAN_ONLY_TEXT}. {sentence}"
    done = f"Moved to {name!r}." if arm == BOTH else f"Moved the {arm} arm to {name!r}."
    return f"{done} {sentence}"


def set_gripper(ctx: Ctx, arm: Arm, state: GripperState) -> str:
    """Open, half open or close the gripper of one arm or both.

    arm is left, right or both; state is open, half or close. Returns what each
    gripper did, and after a close whether the gripper is holding something, or
    that it could not tell; the answer ends with whether the arms are safe for
    driving. Close only once the object is between the fingers. A close in the
    receive pose starts 5 s after the call, so say the close line right before it.
    Tell the person before the gripper moves.

    Args:
        arm: The gripper to move: left, right or both.
        state: open, half (half open) or close.
    """
    sides = ARMS if arm == BOTH else (arm,)
    if state == "close" and _in_receive(ctx, sides):
        _wait_before_the_close(ctx, sides)
    # One sentence per gripper that finished, in arm order.
    done: list[str] = []
    failed = False
    for side in sides:
        try:
            # Left, then right: one goal at a time, as the commander allows.
            moved = ctx.arms.gripper(side, state)
        except StepError as error:
            # Nothing more is sent, and the answer says what was already done,
            # before the message, which ends with the arm sentence.
            raise type(error)(error.code, " ".join([*done, error.message])) from None
        if moved.outcome == "moved":
            done.append(_moved(state, f"the {side} gripper"))
            if state != "close":
                # The hand let go of what the memory holds in it (decision PK-70).
                release(ctx, side)
        elif moved.outcome == "not_reached":
            # The other gripper is still moved, as asked.
            done.append(
                f"The {side} gripper {_NOT_REACHED[state]} (the commander reported "
                f"{moved.result.status!r})."
            )
            failed = True
        # In each case where whether the close holds something is unknown, the close
        # itself ended as the commander allows, so the step goes on; only the answer
        # about the object is missing.
        elif moved.outcome == "unmeasured":
            done.append(
                f"The commander reported that the {side} gripper's close failed and gave no "
                "measurement, as when another command replaced it, so whether the gripper holds "
                "something is unknown. Call get_arm_state to see how far it is open."
            )
        elif moved.outcome == "superseded":
            done.append(
                f"Another command for the {side} gripper came after the close, so whether the "
                "close held something is unknown. Call get_arm_state to see how far the "
                "gripper is open."
            )
        elif moved.outcome == "unread":
            done.append(
                f"The {side} gripper's fingers could not be read after the close, so whether it "
                f"closed on something is unknown: {answers.full_stop(moved.read_error or '')} "
                "Call get_arm_state to see how far it closed."
            )
        elif moved.outcome == "not_closed":
            # The fingers barely moved, whatever the commander's grasp detection says.
            # The other gripper is still closed, as asked.
            gap = "" if moved.finger_m is None else f" (gap {moved.finger_m:.3f} m)"
            done.append(f"The {side} gripper did not close{gap}.")
            failed = True
        else:
            done.append(
                f"Closed the {side} gripper. It is holding "
                f"{'something' if moved.holding else 'nothing'}."
            )

    if state != "close" and not failed and len(sides) == len(ARMS):
        summary = _moved(state, "both grippers")
    else:
        summary = " ".join(done)
    text = f"{summary} {ctx.arms.state().sentence()}"
    if failed:
        # A plain StepError: the gripper goal ended, but not where it was sent.
        raise StepError(ErrorCode.SKILL_FAILED, text)
    return text


def get_arm_state(ctx: Ctx) -> str:
    """Read where the arms are now, and how far each gripper is open.

    Returns one line per arm, with the named pose it is in, or 'no named pose', and
    whether it is safe for driving; then one line per gripper, open, half open or
    closed, with the gap, and for a closed gripper whether it holds something or is
    empty; and last whether the arms are safe for driving. Call it after an arm or
    gripper tool fails, before you try again.
    """
    try:
        # It recovers first while the arms are unverified.
        reading = ctx.arms.read()
    except StepError as error:
        # The cause and the sentence from one report, so that they agree.
        now = ctx.arms.state(read=False)
        raise answers.with_sentence(error, _checked_prefix(error, now), now.sentence()) from None

    # `before` is from before the first read and `after` from after the last. A
    # recovery whose count is the one the call began with cleared the state the call
    # began with, and a changed count means that the arms became unverified during
    # the call.
    start, end = reading.before, reading.after
    unsure = end.unverified_count != start.unverified_count or end.unverified is not None
    lines = []
    if unsure:
        lines.append(_unverified_again(end.unverified))
    elif (
        start.unverified is not None
        and end.recovery is not None
        and end.recovery.count == start.unverified_count
    ):
        lines.append(
            f"{_checked(start.unverified)} {_recovered(end.recovery)} "
            "Arm and gripper tools can be used again."
        )
    for side in ARMS:
        arm_state = end.arms.arm(side)
        if arm_state is None:
            # Such as the arm of a goal that the commander accepted late during the call.
            lines.append(f"{side.capitalize()} arm: state unknown.")
            continue
        where = arm_state.pose or "no named pose"
        safe = "safe" if arm_state.travel_safe else "not safe"
        lines.append(f"{side.capitalize()} arm: {where}, {safe} for driving.")
    for gripper in reading.grippers:
        which = f"{gripper.arm.capitalize()} gripper"
        if gripper.finger_m is None or gripper.reads_as is None:
            lines.append(
                f"{which}: could not be read; last command {gripper.commanded or 'unknown'}."
            )
        else:
            lines.append(
                f"{which}: {_STATE_WORDS[gripper.reads_as]}{_held(gripper)} "
                f"(gap {gripper.finger_m:.3f} m)."
            )
    lines.append(_safety(end.arms, unsure))
    return "\n".join(lines)


def _held(gripper: GripperReading) -> str:
    """", holding something" or ", empty" for a gripper read as closed, else "".

    By GripperReading.in_grasp_band. Only a closed reading gets the verdict: the half
    position lies inside the grasp band too, so a half open reading says nothing about
    a held object.
    """
    if gripper.reads_as != "close" or gripper.in_grasp_band is None:
        return ""
    return ", holding something" if gripper.in_grasp_band else ", empty"


def _in_receive(ctx: Ctx, sides: tuple[str, ...]) -> bool:
    """Whether one of `sides` is in the receive pose, by the backend's memory or a read.

    The arms are read only when the memory does not know one of `sides` and they are
    verified: a read while they are unverified would recover them, which
    get_arm_state and the drive check do, and the close must then be refused, as
    every goal is.
    """
    report = ctx.arms.state(read=False)
    if report.unverified is None and any(report.arms.arm(side) is None for side in sides):
        report = ctx.arms.state(read=True)
    return any(
        (arm_state := report.arms.arm(side)) is not None and arm_state.pose == RECEIVE_POSE
        for side in sides
    )


def _wait_before_the_close(ctx: Ctx, sides: tuple[str, ...]) -> None:
    """Wait RECEIVE_CLOSE_DELAY_S on the cancel token; a stop ends it with no goal sent.

    The wait has nothing to wait for, so its TIMEOUT is its normal end.
    """
    try:
        ctx.cancel.wait_for(lambda: False, RECEIVE_CLOSE_DELAY_S, "the wait before the close")
    except StepError as error:
        if error.code is ErrorCode.TIMEOUT:
            return
        which = "both grippers" if len(sides) == len(ARMS) else f"the {sides[0]} gripper"
        raise StepError(
            ErrorCode.CANCELLED,
            f"Stopped before {which} moved: a stop or a reset is in effect. "
            f"{ctx.arms.state(read=False).sentence()}",
        ) from None


def _moved(state: GripperState, which: str) -> str:
    """What an open or "half" of `which` did, such as "Opened both grippers."."""
    return f"Opened {which}." if state == "open" else f"Set {which} half open."


def _checked_prefix(error: StepError, now: ArmsReport) -> str:
    """What goes before the message of a read that failed: why the arms had to be checked.

    Nothing while the arms are verified, after a stop or a reset, or when the message
    already names why the arms are unverified, as when they became unverified again
    during the check.
    """
    cause = now.unverified
    if cause is None or error.code is ErrorCode.CANCELLED or cause in error.message:
        return ""
    return f"{_checked(cause)} "


def _checked(cause: str) -> str:
    """Why a recovery was made: `cause` is ArmsReport.unverified."""
    return f"The arms had to be checked, as {cause}."


def _recovered(recovery: Recovery) -> str:
    """What the recovery that cleared the unverified state checked, in sentences.

    How the commander showed each goal to be no longer active, then which groups the
    settled read checked. Nothing more: the settled read cannot see a motion slower
    than its tolerance.
    """
    checked = [side for side in ARMS if GRIPPER_GROUPS[side] in recovery.grippers]
    if len(checked) == len(ARMS):
        groups = "both arms and both grippers"
    elif checked:
        groups = f"both arms and the {checked[0]} gripper"
    else:
        groups = "both arms"
    reads = f"two reads of {groups} a moment apart agree"
    sentences = [_RESOLVED[how].format(what=what) for what, how in recovery.goals]
    if sentences:
        sentences[-1] = f"{sentences[-1]}, and {reads}"
    else:
        sentences.append(reads.capitalize())
    return " ".join(f"{sentence}." for sentence in sentences)


def _unverified_again(cause: str | None) -> str:
    """The first line when the arms became unverified during the call.

    `cause` is `unverified` in the report at the end of the call, or None when a
    recovery has cleared the state again since. The answer then still says to check
    again: the count alone does not show what that recovery checked.
    """
    happened = (
        cause
        if cause is not None
        else "a goal from an earlier call was given up on or accepted late"
    )
    return (
        f"While the arms were being read, {happened}, so the arms have to be checked again. "
        "Call get_arm_state again in a few seconds."
    )


def _safety(arms: ArmsState, unsure: bool) -> str:
    """The sentence get_arm_state ends with, by `arms`.

    With `unsure`, never that the robot may drive: the drive check then recovers
    first, and refuses when it cannot. `arms` keeps the arms known while only a
    gripper goal made them unverified, so it alone could say that they are safe.
    """
    if unsure and arms.travel_safe:
        return UNKNOWN_SENTENCE
    return arms.sentence()
