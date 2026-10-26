"""Move one arm or both to a named pose, wait, and check the joints (ADR-0022, ADR-0024).

The pose's joint tables go to the commander's /set_joint_position in the
order its /get_joint_states gives for the group, so a commander that lists
its joints in another order, or without openarm_right_joint2, still gets the
right values. A reported success is not trusted: the joints are read after
every move and compared with the pose. In the commander's plan-only mode it
reports success with "Plan Only" and moves nothing, so the answer never says
the arms moved: it says they were already in the pose, or fails.
"""

from typing import Literal

from pydantic import BaseModel, Field

from walkie_agent.arm.poses import BOTH
from walkie_agent.toolbox import Effect, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _arm as shared

META = ToolMeta(
    name="move_arm_to_pose",
    description=(
        "Move one arm or both to a named pose and wait until they arrive, usually 5 to 10 "
        "seconds. Use a pose name from list_arm_poses. arm is left, right or both (the "
        "default); list_arm_poses says which arms each pose can move. Tell the people near "
        "you before the arms move. The answer ends with whether the arms are safe for "
        "driving: before you drive, move both arms to a pose that is, such as 'home'."
    ),
    effects=frozenset({Effect.M}),
    questions=(
        "Put your arms down.",
        "Raise your hands.",
        "Hold out your left hand to take the bag.",
        "Tuck your arms in before you drive.",
        "Move your right arm to standby.",
    ),
    examples=({"pose": "home"}, {"pose": "receive", "arm": "left"}),
    requires=(shared.ARM,),
)


class Args(BaseModel):
    pose: str = Field(description="The pose's name, as list_arm_poses gives it, such as 'home'.")
    arm: Literal["left", "right", "both"] = Field(
        default="both", description="The arm to move: left, right or both."
    )


class Out(ToolOutput):
    pose: str
    arm: str
    # The commander's status: "Success", or "Plan Only" when the commander runs
    # with plan_only and planned without moving; the joint check then fails
    # unless the arms were already in the pose, and the answer says so.
    status: str
    # The largest joint error after the move, in radians.
    max_error_rad: float
    travel_safe: bool | None


def run(ctx: ToolContext, args: Args) -> Out:
    # ctx.arms runs the steps and the joint check, and its errors carry the
    # whole answer (_arm.RosArms.move_to).
    moved = ctx.arms.move_to(args.pose, args.arm)
    name = moved.pose
    # The sentence and travel_safe from the one report taken after the
    # check, so that a goal from an earlier call that the commander accepts
    # late cannot come between them.
    sentence = shared.state_sentence(moved.report)
    if moved.status == shared.PLAN_ONLY:
        # Nothing moved, so the check passed because the arms were already there.
        where = (
            f"Already at {name!r}."
            if args.arm == BOTH
            else f"The {args.arm} arm is already at {name!r}."
        )
        summary = f"{where} {shared.PLAN_ONLY_TEXT}. {sentence}"
    else:
        done = (
            f"Moved to {name!r}." if args.arm == BOTH else f"Moved the {args.arm} arm to {name!r}."
        )
        summary = f"{done} {sentence}"
    return Out(
        summary=summary,
        pose=name,
        arm=args.arm,
        status=moved.status,
        max_error_rad=moved.max_error_rad,
        travel_safe=shared.travel_safe(moved.report),
    )
