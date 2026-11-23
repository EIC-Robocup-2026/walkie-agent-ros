"""What a hand holds, as the object memory records it (decision PK-70).

pick_object records an object as held once the close holds it, and as lost when the
hand turns out empty after the move home. set_gripper records a release when it opens
a hand: a handover with the arm in the standby pose, else a place somewhere unknown.
While an object is held, the memory stops matching the camera's sightings of it to
the place it was picked from, where they made new tracks follow the hand (PK-67), and
once it is released the memory waits to see it again before it gives it a place.

Every record is best-effort. A memory that is missing, that does not record states
yet or that fails is logged, and the skill goes on: a pick holding an object in the
air must not stop over bookkeeping. A stop during a record is logged too, and the next
motion, which checks the same token, raises it.
"""

import logging

from walkie_agent.capabilities import Ctx, RecalledObject
from walkie_agent.errors import ErrorCode, StepError

logger = logging.getLogger(__name__)

# The pose a person takes a held object from the robot's hand in (prompts.py, PK-68).
HANDOVER_POSE = "standby"
# How long each read or record of the memory may take, in seconds. The memory answers
# in milliseconds; the cap keeps a stuck memory from holding an arm still.
RECORD_TIMEOUT_S = 3.0


def gripper_frame(arm: str) -> str:
    """The tool frame of `arm`'s hand, as walkie-ros's arm model names it."""
    return f"openarm_{arm}_hand_tcp"


def held_objects(ctx: Ctx) -> tuple[RecalledObject, ...]:
    """The objects the memory holds as in a hand; none when it cannot say."""
    try:
        return ctx.object_memory.search("", states=("held",), timeout_s=RECORD_TIMEOUT_S)
    except StepError as error:
        _skipped("read the held objects", error)
        return ()


def record(ctx: Ctx, object_id: str, operation: str, arm: str) -> bool:
    """Record `operation` for the object in `arm`'s hand; True when the memory took it."""
    try:
        ctx.object_memory.set_state(
            object_id, operation, gripper_frame=gripper_frame(arm), timeout_s=RECORD_TIMEOUT_S
        )
    except StepError as error:
        _skipped(f"record {operation} for object {object_id}", error)
        return False
    logger.info("The object memory recorded %s for object %s.", operation, object_id)
    return True


def release(ctx: Ctx, arm: str) -> None:
    """Record that `arm`'s hand let go of whatever the memory holds in it.

    The memory refuses the record for an object held in the other hand, which is
    logged and left held.
    """
    held = held_objects(ctx)
    if not held:
        return
    state = ctx.arms.state(read=False).arms.arm(arm)
    handed = state is not None and state.pose == HANDOVER_POSE
    operation = "hand_over_succeeded" if handed else "place_succeeded"
    for found in held:
        record(ctx, found.id, operation, arm)


def _skipped(what: str, error: StepError) -> None:
    quiet = error.code in (ErrorCode.UNAVAILABLE, ErrorCode.CANCELLED)
    logger.log(
        logging.INFO if quiet else logging.WARNING,
        "Could not %s in the object memory: %s",
        what,
        error.message,
    )
