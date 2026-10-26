"""Open, half open or close a gripper, and say after a close whether it holds something.

An open or "half" must succeed: the commander reports a failure when the
gripper controller did not reach the position. A close is judged otherwise,
since a real grasp usually ends with success false, because the fingers stall
on the object. Whether a close holds something is unknown, and the answer
says so, when:
  - the commander reports it failed with `grasped` false and a gap of 0.0:
    it reports that for a close that a newer command superseded, also one
    from another node, and for a close it did not measure, as with
    grasp_detect_enable off, so the fingers may show nothing of this close
    and are not read;
  - another command for the gripper came after the close, so a read would
    not show where the close left the fingers
    (CommanderClient.joint_states_since());
  - the fingers cannot be read.
Otherwise the fingers are read from /get_joint_states:
  - fingers still near the open position mean the gripper did not close;
  - otherwise the commander's `grasped` is the answer when its gap agrees
    with the read;
  - when it does not, the read is judged by the commander's detection band,
    read from its parameters, or by its default band when they cannot be
    read. A close that reached its position without being measured, as
    before the commander's first finger reading, reports `grasped` false and
    a gap of 0.0 too, and a close that meets nothing can be measured before
    the fingers settle, a few millimetres from where they stop.
The commander reports `grasped` and the gap only on a close; on an open or
"half" it reports false and 0.0 whatever the fingers did, so those are never
read (CommanderClient returns None for them).
"""

import logging
import threading
from typing import Literal

from pydantic import BaseModel, Field

from walkie_agent.arm.commander import CommanderClient, GripperResult
from walkie_agent.arm.poses import (
    ARMS,
    BOTH,
    GRASP_DETECT_MAX,
    GRASP_DETECT_MIN,
    GRIPPER_GROUPS,
    GripperPositions,
)
from walkie_agent.toolbox import Effect, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _arm as shared

logger = logging.getLogger(__name__)

META = ToolMeta(
    name="set_gripper",
    description=(
        "Open, half open or close the gripper of one arm or both. arm is left, right or "
        "both; state is open, half or close. After a close, the answer says whether the "
        "gripper is holding something, or that it could not tell. Close only once the "
        "object is between the fingers, such as after the person says the bag is in your "
        "hand. Tell the person before the gripper moves."
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

# What an open or "half" that the commander reports failed did not do.
_NOT_REACHED = {"open": "did not open fully", "half": "did not reach half open"}


class Args(BaseModel):
    arm: Literal["left", "right", "both"] = Field(
        description="The gripper to move: left, right or both."
    )
    state: Literal["open", "half", "close"] = Field(description="open, half (half open) or close.")


class Out(ToolOutput):
    arm: str
    state: str
    # Per arm moved: whether the close holds something, judged from a read of
    # the fingers after it. None when that cannot be told: after an open or
    # "half", after a close that did not close, and when whether it holds
    # something is unknown (see the module's docstring).
    holding: dict[str, bool | None]


def run(ctx: ToolContext, args: Args) -> Out:
    client = shared.commander(ctx)
    grippers = client.poses.grippers
    position = grippers.position(args.state)
    arms = ARMS if args.arm == BOTH else (args.arm,)
    holding: dict[str, bool | None] = {}
    # One sentence per gripper that finished, in arm order.
    done: list[str] = []
    failed = False
    for arm in arms:
        try:
            # Left, then right: one goal at a time, as the client allows.
            result = client.control_gripper(
                GRIPPER_GROUPS[arm], position, ctx.cancel, shared.CONFIG.gripper_timeout_s
            )
        except StepError as e:
            # Nothing more is sent, and the answer says what was already done.
            if e.code == "CANCELLED" and ctx.cancel.is_set():
                stopped = (
                    shared.interrupted(e, gripper=True)
                    or f"Stopped by reset before the {arm} gripper moved."
                )
                raise StepError(
                    "CANCELLED",
                    f"{_joined(done, stopped)} {shared.state_sentence(client)}",
                    e.data,
                ) from None
            raise shared.with_sentence(
                e,
                _joined(done, f"Could not move the {arm} gripper: "),
                shared.state_sentence(client),
            ) from None
        holding[arm] = None
        if args.state != "close":
            if result.success:
                done.append(_moved(args.state, f"the {arm} gripper"))
            else:
                # The other gripper is still moved, as asked.
                done.append(
                    f"The {arm} gripper {_NOT_REACHED[args.state]} (the commander reported "
                    f"{result.status!r})."
                )
                failed = True
            continue
        # In each case where whether the close holds something is unknown,
        # the close itself ended as the commander allows, so the step goes
        # on; only the answer about the object is missing.
        if _unmeasured_failure(result):
            done.append(
                f"The commander reported that the {arm} gripper's close failed and gave no "
                "measurement, as when another command replaced it, so whether the gripper holds "
                "something is unknown. Call get_arm_state to see how far it is open."
            )
            continue
        try:
            finger = shared.read_finger(client, arm, ctx.cancel, result.command)
            # Raises CANCELLED only, when it reads the commander's band.
            held = _holding(result, finger, grippers, client, ctx.cancel)
        except StepError as e:
            if e.code == "CANCELLED" and ctx.cancel.is_set():
                stopped = (
                    f"Stopped by reset before checking whether the {arm} gripper holds "
                    "something."
                )
                raise StepError(
                    "CANCELLED", f"{_joined(done, stopped)} {shared.state_sentence(client)}", e.data
                ) from None
            if e.code == "PRECONDITION_FAILED" and "command" in e.data:
                # joint_states_since(): a newer command for the gripper came.
                done.append(
                    f"Another command for the {arm} gripper came after the close, so whether the "
                    "close held something is unknown. Call get_arm_state to see how far the "
                    "gripper is open."
                )
                continue
            done.append(
                f"The {arm} gripper's fingers could not be read after the close, so whether it "
                f"closed on something is unknown: {shared.full_stop(e.message)} Call "
                "get_arm_state to see how far it closed."
            )
            continue
        holding[arm] = held
        if held is None:
            # The fingers barely moved, whatever `grasped` says. The other
            # gripper is still closed, as asked.
            done.append(f"The {arm} gripper did not close (gap {finger:.3f} m).")
            failed = True
        else:
            done.append(
                f"Closed the {arm} gripper. It is holding {'something' if held else 'nothing'}."
            )

    if args.state != "close" and not failed and len(arms) == len(ARMS):
        summary = _moved(args.state, "both grippers")
    else:
        summary = " ".join(done)
    text = f"{summary} {shared.arm_sentence(client, ctx.cancel)}"
    if failed:
        raise StepError(
            "SKILL_FAILED", text, {"arm": args.arm, "state": args.state, "holding": holding}
        )
    return Out(summary=text, arm=args.arm, state=args.state, holding=holding)


def _unmeasured_failure(result: GripperResult) -> bool:
    """True for a close that the commander reports failed with a gap of 0.0 and grasped false.

    The commander reports that for a close that a newer command for the
    group superseded, also one from another node, and for a failed close
    that it did not measure, as with grasp_detect_enable off. Which of these
    happened cannot be told, and after a supersession the fingers show the
    newer command, so such a close is not judged from a read.
    """
    return not result.success and result.gripper_gap == 0.0 and result.grasped is False


def _holding(
    result: GripperResult,
    finger: float,
    grippers: GripperPositions,
    client: CommanderClient,
    cancel: threading.Event,
) -> bool | None:
    """Whether a close holds something, by `finger`, the travel read after it.

    Only for a close that _unmeasured_failure() does not exclude, read while
    it was still the gripper's latest command. None when the gripper did not
    close: fingers near the open position barely moved, which no object
    explains, so the gripper is stuck or blocked, even when the travel lies
    inside the detection band. Otherwise the commander's `grasped` is the
    answer, by its own detection parameters, when its gap agrees with the
    read. A gap that does not agree is one the commander did not measure on
    a close that reached its position, one it measured before the fingers
    settled, as on a close that meets nothing, or the fingers moved since,
    as another node can make them. The read is then judged by the same
    detection parameters, read from the commander (_band). Raises StepError
    CANCELLED only.
    """
    if finger >= grippers.open_position - shared.CONFIG.holding_margin_m:
        return None
    gap = result.gripper_gap
    if gap is not None and abs(gap - finger) <= shared.CONFIG.gap_agreement_m:
        return bool(result.grasped)
    low, high = _band(client, cancel)
    return low <= finger <= high


def _band(client: CommanderClient, cancel: threading.Event) -> tuple[float, float]:
    """The commander's grasp detection band now, or its default when that cannot be read.

    The default is the one the commander starts with, so a read that fails
    judges as the commander would unless its band was changed. Raises
    StepError CANCELLED only.
    """
    try:
        return client.grasp_band(cancel, shared.CONFIG.read_timeout_s)
    except StepError as e:
        if e.code == "CANCELLED":
            raise
        logger.warning(
            f"Judging a close by the commander's default grasp detection band: {e.message}"
        )
        return GRASP_DETECT_MIN, GRASP_DETECT_MAX


def _moved(state: str, which: str) -> str:
    """What an open or "half" of `which` did, such as "Opened both grippers."."""
    return f"Opened {which}." if state == "open" else f"Set {which} half open."


def _joined(done: list[str], text: str) -> str:
    return " ".join([*done, text])
