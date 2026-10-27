"""Hold one hand still and open in the receive pose for a set time, for a hand-over.

The robot detects the hand-over itself (ADR-0059): the guest is asked once,
and never asked whether the object is there. The wait is a fixed hold of
HOLD_S, which the LLM does not choose. It sends no goal and says nothing:
the arm must already be in the receive pose with the hand open, and the
call refuses otherwise. The arm and the hand are read again after the hold,
and a hand that closed or an arm that left the pose meanwhile fails the
call; otherwise the answer ends with the arm sentence of that read, as
every arm tool's does. The close line and set_gripper come next;
set_gripper delays a close in the receive pose until the line has played.

The hold runs on ctx.clock, which is wall time on the robot and in the
simulation alike, so at about 0.47 times real time it lasts about 2.4 sim-s.
A hand reads open when its finger travel is at least the open position of
config/arm_poses.toml less _arm.CONFIG.holding_margin_m, the margin a close
is judged by.
"""

from typing import Literal

from pydantic import BaseModel, Field

from walkie_agent.arm.poses import load_arm_poses
from walkie_agent.toolbox import Effect, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _arm as shared

# How long the hand is held still and open, in seconds. Not an argument: the
# hold is a set time (ADR-0059). The robot check tunes it for a real guest.
HOLD_S = 5.0

# The pose the hand is held out in.
RECEIVE_POSE = "receive"

META = ToolMeta(
    name="wait_for_handover",
    description=(
        "Hold the hand still and open in the receive pose for a set time, so the person "
        "can hang the object on it. It refuses unless the arm is in the receive pose and "
        "the hand is open. It sends no motion and says nothing: ask the person once with "
        "speak before it, and never ask whether the object is there. Then say the close "
        "line and close the hand with set_gripper."
    ),
    effects=frozenset({Effect.R}),
    questions=(
        "Wait for the guest to hang the bag on your hand.",
        "Hold your hand out and wait for the bag.",
        "Keep your hand still while they put the bag in it.",
        "Give the guest time to hand you the bag.",
    ),
    examples=({"arm": "left"},),
    requires=(shared.ARM,),
)


class Args(BaseModel):
    arm: Literal["left", "right"] = Field(
        default="left", description="The hand to hold out: left (the default) or right."
    )


class Out(ToolOutput):
    arm: str
    # How long the hand was held, in seconds: HOLD_S.
    waited_s: float


def run(ctx: ToolContext, args: Args) -> Out:
    arm = args.arm
    # The least finger travel that reads open, in metres.
    open_m = _open_position() - shared.CONFIG.holding_margin_m
    _check_ready(ctx, arm, open_m, after_hold=False)
    try:
        ctx.sleep(HOLD_S)
    except StepError as e:
        if e.code != "CANCELLED":
            raise
        raise StepError(
            "CANCELLED",
            f"Stopped by reset during the wait for the hand-over. The {arm} gripper did not "
            f"move. {shared.state_sentence(ctx.arms.state(read=False))}",
        ) from None
    sentence = _check_ready(ctx, arm, open_m, after_hold=True)
    return Out(
        summary=(
            f"The {arm} hand stayed open in {RECEIVE_POSE} for {HOLD_S:g} s. Say the close "
            f"line, then close the hand. {sentence}"
        ),
        arm=arm,
        waited_s=HOLD_S,
    )


def _check_ready(ctx: ToolContext, arm: str, open_m: float, *, after_hold: bool) -> str:
    """Raise StepError PRECONDITION_FAILED unless `arm` reads the receive pose with the hand open.

    The arms are read now, and the finger once: the hand is open when its
    travel is at least `open_m`. A read of the finger that fails raises its
    StepError, with the arm sentence after it. Returns the arm sentence of
    the read, which every arm tool's answer ends with.
    """
    report = ctx.arms.state(read=True)
    sentence = shared.state_sentence(report)
    state = report.arms.arm(arm)
    pose = None if state is None else state.pose
    if pose != RECEIVE_POSE:
        if state is None:
            where = "its pose is unknown"
        elif pose is None:
            where = "it is in no named pose"
        else:
            where = f"it is in {pose!r}"
        problem = (
            f"The {arm} arm left the {RECEIVE_POSE} pose during the wait: {where}."
            if after_hold
            else f"The {arm} arm is not in the {RECEIVE_POSE} pose: {where}."
        )
        raise StepError(
            "PRECONDITION_FAILED",
            f"{problem} Move the {arm} arm to {RECEIVE_POSE} first. {sentence}",
            {"arm": arm, "pose": pose},
        )
    try:
        finger = ctx.arms.finger(arm)
    except StepError as e:
        if e.code == "CANCELLED" and ctx.cancel.is_set():
            raise StepError(
                "CANCELLED",
                f"Stopped by reset before the {arm} hand was read. {sentence}",
            ) from None
        raise shared.with_sentence(e, f"Could not read the {arm} hand: ", sentence) from None
    if finger < open_m:
        problem = (
            f"The {arm} hand closed during the wait (gap {finger:.3f} m)."
            if after_hold
            else f"The {arm} hand is not open (gap {finger:.3f} m)."
        )
        raise StepError(
            "PRECONDITION_FAILED",
            f"{problem} Open the {arm} hand first. {sentence}",
            {"arm": arm, "finger_m": finger},
        )
    return sentence


def _open_position() -> float:
    """The open finger travel, in metres, from config/arm_poses.toml.

    The same file, by the same default path, that the agent's CommanderClient
    loads at startup, so the threshold matches the one set_gripper's close
    is judged by. ctx.arms gives the poses, not the gripper positions.
    """
    return load_arm_poses().grippers.open_position
