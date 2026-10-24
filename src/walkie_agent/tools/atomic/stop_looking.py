"""Stop the gaze loop, so the base stops turning toward a person or a seat (ADR-0051).

Never an error: stopping a loop that has already stopped is what the LLM
wanted anyway, and the answer then says which target stopped last, why and
when, so the LLM learns that a person it thought it was looking at was lost.
"""

from pydantic import BaseModel

from walkie_agent.gaze import LOST, STOP_LOOKING
from walkie_agent.toolbox import Effect, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _gaze as shared

META = ToolMeta(
    name="stop_looking",
    description=(
        "Stop turning toward the person or seat that look_at chose; the robot then stays "
        "still. Call it only when someone asks you to stop looking at them or to stop "
        "turning, or when you stay where you are after your last line to the person. "
        "Driving and moving an arm stop it by themselves, so you need not call it before "
        "those. Safe to call when the robot is not looking at anyone: the answer "
        "then says what it looked at last."
    ),
    effects=frozenset({Effect.N}),
    questions=(
        "Stop looking at me.",
        "Stop turning.",
        "You can stop facing the guest.",
        "Are you still looking at someone?",
    ),
    examples=({},),
    requires=(shared.GAZE,),
)


class Args(BaseModel):
    pass


class Out(ToolOutput):
    # The target this call stopped, or None when nothing was running.
    stopped: str | None
    # The last target and why it stopped, such as "lost"; None before the
    # first look_at.
    last_target: str | None
    reason: str | None


def run(ctx: ToolContext, args: Args) -> Out:
    gaze = shared.gaze_loop(ctx)
    stopped = gaze.stop(STOP_LOOKING)
    if stopped is not None:
        return Out(
            summary=f"Stopped looking at {stopped}.",
            stopped=stopped,
            last_target=stopped,
            reason=STOP_LOOKING,
        )
    status = gaze.status()
    if status.target is None or status.stop_reason is None or status.stopped_at is None:
        return Out(summary="Not looking at anyone.", stopped=None, last_target=None, reason=None)
    # max(): a clock that jumped back, as at a simulation restart, gives no
    # negative age.
    age = max(0.0, status.now - status.stopped_at)
    reason = status.stop_reason
    if reason == LOST:
        how = f"was lost {age:.0f} s ago"
    else:
        # The cause last, since some are clauses: "was stopped 2 s ago
        # because the turn took too long".
        how = f"was stopped {age:.0f} s ago {shared.stop_cause(reason)}"
    return Out(
        summary=f"Not looking at anyone. The last target, {status.target}, {how}.",
        stopped=None,
        last_target=status.target,
        reason=reason,
    )
