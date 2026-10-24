"""List the named arm poses the LLM may use, from config/arm_poses.toml (ADR-0022).

Only the offered poses: `offered = false` hides a pose everywhere, and a pose
not yet tried on the robot is listed only in the simulation. The LLM gets
names and descriptions, never joint values.
"""

from pydantic import BaseModel

from walkie_agent.toolbox import Effect, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _arm as shared

META = ToolMeta(
    name="list_arm_poses",
    description=(
        "List the named arm poses you can use with move_arm_to_pose. Each line gives the "
        "pose's name, which arms it can move (left, right or both), whether the robot may "
        "drive with an arm in it, and what it looks like. The last line says whether the "
        "arms are safe for driving now."
    ),
    effects=frozenset({Effect.R}),
    questions=(
        "What poses can your arms make?",
        "Which arm poses are safe for driving?",
        "Can you hold your arms up?",
        "Which pose do you use to take a bag from someone?",
    ),
    examples=({},),
    requires=(shared.ARM,),
)


class Args(BaseModel):
    pass


class Out(ToolOutput):
    # The offered pose names, in file order.
    poses: list[str]


def run(ctx: ToolContext, args: Args) -> Out:
    client = shared.commander(ctx)
    offered = shared.offered_poses(ctx, client)
    lines = [
        f"{name}: {shared.arms_text(pose)}; "
        f"{'safe' if pose.travel_safe else 'not safe'} for driving. {pose.description}"
        for name, pose in offered.items()
    ]
    if not lines:
        lines = ["No arm poses are offered."]
    lines.append(shared.arm_sentence(client, ctx.cancel))
    return Out(summary="\n".join(lines), poses=list(offered))
