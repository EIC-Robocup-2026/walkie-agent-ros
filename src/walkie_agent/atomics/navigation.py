"""Atomics that move the base and name the places it can go to."""

import logging
import math

from walkie_agent.capabilities import Ctx
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.perception.geometry import wrap_deg

logger = logging.getLogger(__name__)

# How far from straight ahead a person may be and still count as faced. A person
# within it gets no turn, and a turn that Nav2 reports as failed still faces the
# person if it ended within it: Nav2's progress checker can abort a turn on the
# spot that is still settling on its 0.6 degree yaw tolerance (decision S2-4).
FACE_TOLERANCE_DEG = 10.0


def list_places(ctx: Ctx) -> list[dict[str, str]]:
    """List the named places Walkie can go to.

    Returns a list of objects with the keys name and description.
    """
    return [
        {"name": place.name, "description": place.description}
        for place in ctx.base.places().values()
    ]


def go_to(ctx: Ctx, place: str) -> str:
    """Drive to a named place and wait until Walkie arrives.

    Returns a sentence that says where Walkie arrived.

    Args:
        place: The name of a room or a location, such as "kitchen room" or "shelf".
    """
    ctx.base.go_to(place)
    return f"Arrived at {place.strip()}."


def move_relative(
    ctx: Ctx, forward_m: float, left_m: float = 0.0, turn_deg: float = 0.0
) -> str:
    """Move a short distance relative to where Walkie stands and faces now.

    One call moves at most 3 m. Use go_to for anything with a name. Returns a
    sentence with the move and the pose Walkie ended at.

    Args:
        forward_m: Meters to move forward. Negative moves backward.
        left_m: Meters to move to the left. Negative moves right.
        turn_deg: Degrees to turn once there. Positive turns left, negative turns right.
    """
    pose = ctx.base.move_relative(forward_m, left_m, turn_deg)
    return (
        f"Moved {_describe_move(forward_m, left_m, turn_deg)}. "
        f"Now at x={_round(pose.x, 2):.2f} m, y={_round(pose.y, 2):.2f} m, "
        f"heading {_round(math.degrees(pose.yaw), 0):.0f} degrees."
    )


def face_person(ctx: Ctx, person_id: str) -> str:
    """Turn on the spot, once, to face a person in view.

    Returns a sentence that says how far Walkie turned. A person already within
    10 degrees of straight ahead gets no turn. Walkie does not keep following
    the person afterwards.

    Args:
        person_id: The id of a person, as find_people returns it.
    """
    seen = next((p for p in ctx.vision.people() if p.id == person_id), None)
    if seen is None:
        raise StepError(
            ErrorCode.NOT_FOUND, f"{person_id} is not in view here. Call find_people first."
        )
    if seen.bearing_deg is None:
        raise StepError(ErrorCode.UNAVAILABLE, f"Walkie cannot tell which way {person_id} is.")
    bearing = seen.bearing_deg
    if abs(bearing) <= FACE_TOLERANCE_DEG:
        return f"Already facing {person_id}, {_offset(bearing)}."
    start = ctx.base.pose()
    try:
        end = ctx.base.move_relative(0.0, 0.0, bearing)
    except StepError as error:
        if error.code is not ErrorCode.SKILL_FAILED:
            raise
        end = ctx.base.pose()
        missed = wrap_deg(math.degrees(start.yaw) + bearing - math.degrees(end.yaw))
        if abs(missed) > FACE_TOLERANCE_DEG:
            raise
        logger.info(
            "face_person counts the turn to %s as done, %.1f degrees off: %s",
            person_id,
            abs(missed),
            error,
        )
    turned = wrap_deg(math.degrees(end.yaw - start.yaw))
    return f"Turned {_round(abs(turned), 0):.0f} degrees {_side(turned)} to face {person_id}."


def _offset(bearing_deg: float) -> str:
    """Plain words for a small bearing, for example "3 degrees to the left"."""
    shown = _round(abs(bearing_deg), 0)
    if not shown:
        return "straight ahead"
    return f"{shown:.0f} degrees to the {_side(bearing_deg)}"


def _side(angle_deg: float) -> str:
    return "left" if angle_deg > 0 else "right"


def _describe_move(forward_m: float, left_m: float, turn_deg: float) -> str:
    """Plain words for a relative move, for example "1 m forward, 0.5 m right"."""
    parts = []
    if forward_m:
        parts.append(f"{abs(forward_m):g} m {'forward' if forward_m > 0 else 'backward'}")
    if left_m:
        parts.append(f"{abs(left_m):g} m {'left' if left_m > 0 else 'right'}")
    if turn_deg:
        parts.append(f"turning {abs(turn_deg):g} degrees {'left' if turn_deg > 0 else 'right'}")
    return ", ".join(parts)


def _round(value: float, digits: int) -> float:
    """Round for display, turning -0.0 into 0.0 so it never prints as "-0"."""
    return round(value, digits) + 0.0
