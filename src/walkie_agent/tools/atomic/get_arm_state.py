"""Read which named pose each arm is in, whether the robot may drive, and how open each gripper is.

The arms are read now, from the commander's /get_joint_states, and so is each
gripper: the one joint of its group is the finger whose travel the
commander's grasp detection reads (ADR-0024). A gripper reads as closed at or
below the commander's close threshold, and otherwise as half open or open,
whichever position is nearer. When a gripper cannot be read, its line gives
the last command the agent sent it, which the client remembers.
"""

from pydantic import BaseModel

from walkie_agent.arm.commander import CommanderClient
from walkie_agent.arm.poses import (
    ARMS,
    GRASP_CLOSE_THRESHOLD,
    GRIPPER_GROUPS,
    GRIPPER_STATES,
    GripperPositions,
)
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


class Args(BaseModel):
    pass


class Out(ToolOutput):
    # Per arm: the named pose it is in, or None when it is in none.
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
    try:
        state = client.read_arms(ctx.cancel, shared.CONFIG.read_timeout_s)
    except StepError as e:
        raise shared.with_sentence(e, "", client.last_state.sentence()) from None

    lines = []
    poses: dict[str, str | None] = {}
    for arm in ARMS:
        arm_state = state.arm(arm)
        if arm_state is None:
            # A both_arms read always holds both arms; this guards a
            # commander that answers with one arm's joints only.
            lines.append(f"{arm.capitalize()} arm: state unknown.")
            poses[arm] = None
            continue
        poses[arm] = arm_state.pose
        where = arm_state.pose or "no named pose"
        safe = "safe" if arm_state.travel_safe else "not safe"
        lines.append(f"{arm.capitalize()} arm: {where}, {safe} for driving.")
    grippers: dict[str, str | None] = {}
    gaps: dict[str, float | None] = {}
    for arm in ARMS:
        try:
            gap = shared.read_finger(client, arm, ctx.cancel)
        except StepError as e:
            if e.code == "CANCELLED" and ctx.cancel.is_set():
                raise shared.with_sentence(e, "", client.last_state.sentence()) from None
            last = _last_command(client, arm)
            grippers[arm], gaps[arm] = last, None
            lines.append(
                f"{arm.capitalize()} gripper: could not be read; last command {last or 'unknown'}."
            )
            continue
        named = _reads_as(client.poses.grippers, gap)
        grippers[arm], gaps[arm] = named, gap
        lines.append(f"{arm.capitalize()} gripper: {_STATE_WORDS[named]} (gap {gap:.3f} m).")
    lines.append(state.sentence())
    return Out(
        summary="\n".join(lines),
        poses=poses,
        travel_safe=state.travel_safe,
        grippers=grippers,
        gripper_gaps_m=gaps,
    )


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
