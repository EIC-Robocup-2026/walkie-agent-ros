"""Open, half open or close a gripper, and say after a close whether it holds something.

The commander reports `grasped` and the settled gap only on a close; on an
open or "half" it reports false and 0.0 whatever the fingers did, so those
are never read (CommanderClient returns None for them). On a close,
`grasped` and the gap are the answer, not the goal's success: a real grasp
usually ends with success false, because the fingers stall on the object.
"""

from typing import Literal

from pydantic import BaseModel, Field

from walkie_agent.arm.poses import ARMS, BOTH, GRIPPER_GROUPS, GripperPositions
from walkie_agent.toolbox import Effect, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _arm as shared

META = ToolMeta(
    name="set_gripper",
    description=(
        "Open, half open or close the gripper of one arm or both. arm is left, right or "
        "both; state is open, half or close. After a close, the answer says whether the "
        "gripper is holding something. Close only once the object is between the fingers, "
        "such as after the person says the bag is in your hand. Tell the person before the "
        "gripper moves."
    ),
    effects=frozenset({Effect.M}),
    questions=(
        "Open your left hand.",
        "Close your hand on the bag.",
        "Let go of the bag.",
        "Open both grippers.",
        "Hold the handle in your right hand.",
    ),
    examples=({"arm": "left", "state": "close"}, {"arm": "both", "state": "open"}),
    requires=(shared.ARM,),
)


class Args(BaseModel):
    arm: Literal["left", "right", "both"] = Field(
        description="The gripper to move: left, right or both."
    )
    state: Literal["open", "half", "close"] = Field(description="open, half (half open) or close.")


class Out(ToolOutput):
    arm: str
    state: str
    # Per arm moved: whether the close holds something. None after an open
    # or "half", which cannot tell.
    holding: dict[str, bool | None]


def run(ctx: ToolContext, args: Args) -> Out:
    client = shared.commander(ctx)
    grippers = client.poses.grippers
    position = grippers.position(args.state)
    arms = ARMS if args.arm == BOTH else (args.arm,)
    holding: dict[str, bool | None] = {}
    # One sentence per close, in arm order.
    closes: list[str] = []
    failed = False
    for arm in arms:
        try:
            # Left, then right: one goal at a time, as the client allows.
            result = client.control_gripper(
                GRIPPER_GROUPS[arm], position, ctx.cancel, shared.CONFIG.gripper_timeout_s
            )
        except StepError as e:
            # Nothing more is sent, and the answer says what was already done.
            done = _done(args.state, list(holding), closes)
            if e.code == "CANCELLED" and ctx.cancel.is_set():
                stopped = f"Stopped by reset before the {arm} gripper finished."
                raise StepError(
                    "CANCELLED",
                    f"{_joined(done, stopped)} {client.last_state.sentence()}",
                    e.data,
                ) from None
            raise shared.with_sentence(
                e,
                _joined(done, f"Could not move the {arm} gripper: "),
                client.last_state.sentence(),
            ) from None
        if args.state != "close":
            holding[arm] = None
            continue
        held = _holding(result.grasped, result.gripper_gap, grippers)
        holding[arm] = held
        if held is None:
            # The fingers barely moved, whatever `grasped` says. The other
            # gripper is still closed, as asked.
            gap = result.gripper_gap
            gap_text = "no gap reported" if gap is None else f"gap {gap:.3f} m"
            closes.append(f"The {arm} gripper did not close ({gap_text}).")
            failed = True
        else:
            closes.append(
                f"Closed the {arm} gripper. It is holding {'something' if held else 'nothing'}."
            )

    text = f"{_done(args.state, list(holding), closes)} {shared.arm_sentence(client, ctx.cancel)}"
    if failed:
        raise StepError(
            "SKILL_FAILED", text, {"arm": args.arm, "state": args.state, "holding": holding}
        )
    return Out(summary=text, arm=args.arm, state=args.state, holding=holding)


def _holding(grasped: bool | None, gap: float | None, grippers: GripperPositions) -> bool | None:
    """Whether a close holds something, or None when the gripper did not close.

    A settled gap near the open position means the fingers barely moved,
    which no object explains: the gripper is stuck or blocked, even when
    the gap lies inside the commander's grasp_detect range. Below that,
    `grasped` is the answer, by the commander's own detection parameters.
    """
    if gap is None or gap >= grippers.open_position - shared.CONFIG.holding_margin_m:
        return None
    return bool(grasped)


def _done(state: str, arms: list[str], closes: list[str]) -> str:
    """What the grippers that finished did, such as "Opened both grippers."."""
    if state == "close":
        return " ".join(closes)
    if not arms:
        return ""
    which = "both grippers" if len(arms) == len(ARMS) else f"the {arms[0]} gripper"
    return f"Opened {which}." if state == "open" else f"Set {which} half open."


def _joined(done: str, text: str) -> str:
    return f"{done} {text}" if done else text
