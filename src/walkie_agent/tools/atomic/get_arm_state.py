"""Read which named pose each arm is in, whether the robot may drive, and how open each gripper is.

The arms are read now, from the commander's /get_joint_states, and so is each
gripper: the one joint of its group is the finger whose travel the
commander's grasp detection reads (ADR-0024). A gripper reads as closed at or
below the commander's close threshold, and otherwise as half open or open,
whichever position is nearer. When a gripper cannot be read, its line gives
the last command the agent sent it, which the client remembers.

While the arms are unverified (CommanderClient.unverified), the read of the
arms recovers first. The commander must show every earlier goal to be no
longer active, and a settled read must agree. A goal from an earlier call
can make the arms unverified again at any time, as when the commander
accepts it late. So every sentence about recovery and safety is built after
the last read, from what the client knows then:
  - when the arms were unverified at the start, the state was cleared, and
    the arms did not become unverified again, the answer starts with why
    they had to be checked and what was checked;
  - when they became unverified during the call, it says so and to call
    get_arm_state again, and never that the tools can be used again.
While the arms are unverified, no answer and no error says that the robot
may drive. When recovery fails, so does the call, and unless a stop or a
reset ended it, its message says why the arms had to be checked.
"""

from pydantic import BaseModel

from walkie_agent.arm import commander as arm_client
from walkie_agent.arm.commander import CommanderClient, Recovery
from walkie_agent.arm.poses import (
    ARMS,
    GRASP_CLOSE_THRESHOLD,
    GRIPPER_GROUPS,
    GRIPPER_STATES,
    GripperPositions,
)
from walkie_agent.arm.state import UNKNOWN_SENTENCE, ArmsState
from walkie_agent.toolbox import Effect, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _arm as shared

META = ToolMeta(
    name="get_arm_state",
    description=(
        "Read where the arms are now: the named pose each arm is in, or 'no named pose', "
        "and whether each is safe for driving. Also how far each gripper is open: open, "
        "half open or closed, with the gap. Call it after an arm or gripper tool fails, "
        "before you try again."
    ),
    effects=frozenset({Effect.R}),
    questions=(
        "Where are your arms?",
        "Are your arms safe for driving?",
        "What pose is your right arm in?",
        "Is your left hand open?",
    ),
    examples=({},),
    requires=(shared.ARM,),
)

# How each value of set_gripper's `state` reads in the answer.
_STATE_WORDS = {"open": "open", "half": "half open", "close": "closed"}

# How the commander showed that a goal from an earlier call is no longer
# active, by its `how` in Recovery.goals. Both "result" and "ended" are its
# report that the goal ended: its result, or its answer to a cancel.
_RESOLVED = {
    "result": "The commander confirmed that {what} from an earlier call ended",
    "ended": "The commander confirmed that {what} from an earlier call ended",
    "rejected": "The commander rejected {what} from an earlier call",
    "unknown": "The commander does not know {what} from an earlier call, as after a restart",
}


class Args(BaseModel):
    pass


class Out(ToolOutput):
    # Per arm: the named pose it is in, or None when it is in none or unknown.
    poses: dict[str, str | None]
    travel_safe: bool | None
    # Per arm: the set_gripper state the gripper's reading counts as, "open",
    # "half" or "close" (_reads_as). When the gripper could not be read, its
    # last command, or None when that is unknown too.
    grippers: dict[str, str | None]
    # Per arm: the gripper's reading, the travel of one finger in metres, or
    # None when it could not be read.
    gripper_gaps_m: dict[str, float | None]


def run(ctx: ToolContext, args: Args) -> Out:
    client = shared.commander(ctx)
    # Before the first read, and the count before the cause, as
    # unverified_count says. At the end, a last_recovery with this count
    # cleared the state that the call began with, and a changed count means
    # that the arms became unverified during the call.
    count = client.unverified_count
    unverified = client.unverified
    try:
        # It recovers first while the arms are unverified.
        client.read_arms(ctx.cancel, shared.CONFIG.read_timeout_s)
    except StepError as e:
        cause = client.unverified
        # Why the arms had to be checked, unless the message says it, as
        # when they became unverified during the check, or a stop or a
        # reset ended the call.
        checked = (
            f"{_checked(cause)} "
            if cause is not None and e.code != "CANCELLED" and "unverified" not in e.data
            else ""
        )
        raise shared.with_sentence(e, checked, shared.state_sentence(client)) from None

    grippers: dict[str, str | None] = {}
    gaps: dict[str, float | None] = {}
    gripper_lines = []
    for arm in ARMS:
        try:
            gap = shared.read_finger(client, arm, ctx.cancel)
        except StepError as e:
            if e.code == "CANCELLED" and ctx.cancel.is_set():
                raise shared.with_sentence(e, "", shared.state_sentence(client)) from None
            last = _last_command(client, arm)
            grippers[arm], gaps[arm] = last, None
            gripper_lines.append(
                f"{arm.capitalize()} gripper: could not be read; last command {last or 'unknown'}."
            )
            continue
        named = _reads_as(client.poses.grippers, gap)
        grippers[arm], gaps[arm] = named, gap
        gripper_lines.append(
            f"{arm.capitalize()} gripper: {_STATE_WORDS[named]} (gap {gap:.3f} m)."
        )

    # After the last read, what the client knows at the end of the call. The
    # count before the cause, so that the arms becoming unverified between
    # the two is seen either way.
    state = client.last_state
    recovery = client.last_recovery
    changed = client.unverified_count != count
    cause = client.unverified
    lines = []
    if changed or cause is not None:
        lines.append(_unverified_again(cause))
    elif unverified is not None and recovery is not None and recovery.count == count:
        lines.append(
            f"{_checked(unverified)} {_recovered(recovery)} "
            "Arm and gripper tools can be used again."
        )
    poses: dict[str, str | None] = {}
    for arm in ARMS:
        arm_state = state.arm(arm)
        if arm_state is None:
            # Such as the arm of a goal that the commander accepted late
            # during the call.
            lines.append(f"{arm.capitalize()} arm: state unknown.")
            poses[arm] = None
            continue
        poses[arm] = arm_state.pose
        where = arm_state.pose or "no named pose"
        safe = "safe" if arm_state.travel_safe else "not safe"
        lines.append(f"{arm.capitalize()} arm: {where}, {safe} for driving.")
    lines.extend(gripper_lines)
    travel_safe, sentence = _safety(state, changed or cause is not None)
    lines.append(sentence)
    return Out(
        summary="\n".join(lines),
        poses=poses,
        travel_safe=travel_safe,
        grippers=grippers,
        gripper_gaps_m=gaps,
    )


def _checked(cause: str) -> str:
    """Why a recovery was made: `cause` is CommanderClient.unverified."""
    return f"The arms had to be checked, as {cause}."


def _recovered(recovery: Recovery) -> str:
    """What the recovery that cleared the unverified state checked, in sentences.

    How the commander showed each goal to be no longer active, then which
    groups the settled read checked. Nothing more: the settled read cannot
    see a motion slower than its tolerance (commander.SETTLE_S).
    """
    arms = [arm for arm in ARMS if GRIPPER_GROUPS[arm] in recovery.grippers]
    if len(arms) == len(ARMS):
        groups = "both arms and both grippers"
    elif arms:
        groups = f"both arms and the {arms[0]} gripper"
    else:
        groups = "both arms"
    reads = f"two reads of {groups} {arm_client.SETTLE_S:g} s apart agree"
    sentences = [_RESOLVED[how].format(what=what) for what, how in recovery.goals]
    if sentences:
        sentences[-1] = f"{sentences[-1]}, and {reads}"
    else:
        sentences.append(reads.capitalize())
    return " ".join(f"{sentence}." for sentence in sentences)


def _unverified_again(cause: str | None) -> str:
    """The first line when the arms became unverified during the call.

    `cause` is CommanderClient.unverified at the end of the call, or None
    when a recovery has cleared the state again since. The answer then still
    says to check again: the count alone does not show what that recovery
    checked.
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


def _safety(state: ArmsState, unverified: bool) -> tuple[bool | None, str]:
    """travel_safe and the sentence an answer ends with, by `state`.

    With `unverified`, never that the robot may drive: the drive check then
    recovers first, and refuses when it cannot. `state` keeps the arms
    known while only a gripper goal made them unverified, so it can say
    that they are safe for driving then.
    """
    if unverified and state.travel_safe:
        return None, UNKNOWN_SENTENCE
    return state.travel_safe, state.sentence()


def _reads_as(grippers: GripperPositions, gap: float) -> str:
    """The set_gripper state that a gripper read at `gap` counts as.

    "close" at or below GRASP_CLOSE_THRESHOLD: fingers stopped on an object
    may be nearer the half position than the closed one, but they closed.
    Above it, the nearer of "half" and "open", and "open" on a tie.
    """
    if gap <= GRASP_CLOSE_THRESHOLD:
        return "close"
    return min(("open", "half"), key=lambda state: abs(grippers.position(state) - gap))


def _last_command(client: CommanderClient, arm: str) -> str | None:
    """The `state` of the last set_gripper on `arm`, or None when the client does not know it."""
    position = client.last_gripper_position(GRIPPER_GROUPS[arm])
    if position is None:
        return None
    grippers = client.poses.grippers
    # Exact: the client remembers the very value set_gripper sent.
    for state in GRIPPER_STATES:
        if grippers.position(state) == position:
            return state
    return None
