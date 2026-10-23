"""Read which named pose each arm is in and whether the robot may drive (ADR-0024).

The arms are read now, from the commander's /get_joint_states. The grippers
cannot be: no commander service reports the gap outside a close, so each
gripper's line gives the last command the agent sent it, which the client
remembers.
"""

from pydantic import BaseModel

from walkie_agent.arm.commander import CommanderClient
from walkie_agent.arm.poses import ARMS, GRIPPER_GROUPS, GRIPPER_STATES
from walkie_agent.toolbox import Effect, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _arm as shared

META = ToolMeta(
    name="get_arm_state",
    description=(
        "Read where the arms are now: the named pose each arm is in, or 'no named pose', "
        "and whether each is safe for driving. For each gripper, the last command you sent "
        "it (open, half or close), since how far a gripper is open cannot be read. Call it "
        "after an arm or gripper tool fails, before you try again."
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


class Args(BaseModel):
    pass


class Out(ToolOutput):
    # Per arm: the named pose it is in, or None when it is in none.
    poses: dict[str, str | None]
    travel_safe: bool | None
    # Per arm: the gripper's last command, "open", "half" or "close", or
    # None when unknown.
    grippers: dict[str, str | None]


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
    grippers = {arm: _gripper_state(client, arm) for arm in ARMS}
    for arm, last in grippers.items():
        lines.append(
            f"{arm.capitalize()} gripper: " + (f"last command {last}." if last else "unknown.")
        )
    lines.append(state.sentence())
    return Out(
        summary="\n".join(lines), poses=poses, travel_safe=state.travel_safe, grippers=grippers
    )


def _gripper_state(client: CommanderClient, arm: str) -> str | None:
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
