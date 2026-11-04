"""Atomics that move the base and name the places it can go to."""

import math

from walkie_agent.capabilities import Ctx


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
