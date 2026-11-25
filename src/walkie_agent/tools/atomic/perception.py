"""Atomics that report what Walkie sees where it stands.

Each object and person comes with where it is from Walkie now, in the base's own
frame (decision PK-77): distance_m and bearing_deg as ctx.vision measures them, and
forward_m and left_m, the same point as move_relative takes it. They are robot-relative,
so no map coordinate reaches the LLM (ADR-0010). An unknown one is null.
"""

import math
from dataclasses import asdict
from typing import Any

from walkie_agent.capabilities import Ctx
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.perception.geometry import wrap_deg


def find_objects(ctx: Ctx, what: str = "") -> list[dict[str, Any]]:
    """Look for objects from where Walkie stands now.

    At a location Walkie sees what is on that location. At a room it sees what
    is on the locations in that room. Returns a list of objects with the keys
    id, name, category, place, distance_m, bearing_deg, forward_m and left_m,
    which is empty if nothing matches. Pass an id to pick or object_info.

    The numbers are measured from where Walkie stands now, in its own frame, and
    are null when the camera cannot tell. forward_m and left_m are what
    move_relative takes. To walk up to an object, stop about 0.8 m short of it so
    Walkie does not bump into it.

    Args:
        what: An object name or a category, such as "apple" or "fruits". Empty means everything.
    """
    return [
        {
            "id": seen.id,
            "name": seen.name,
            "category": seen.category,
            "place": seen.place,
            **_position(seen.distance_m, seen.bearing_deg),
        }
        for seen in ctx.vision.objects(what)
    ]


def object_info(ctx: Ctx, object_id: str) -> dict[str, Any]:
    """Measure one object that is in view, to compare it with others.

    Returns an object with the keys id, name, category, place, size_cm,
    weight_g, thickness_cm, distance_m, bearing_deg, forward_m and left_m. A
    measure Walkie cannot tell is null. The last four are as find_objects gives
    them.

    Args:
        object_id: The id of an object, as find_objects returns it.
    """
    for seen in ctx.vision.objects():
        if seen.id == object_id:
            return {**asdict(seen), **_position(seen.distance_m, seen.bearing_deg)}
    raise StepError(
        ErrorCode.NOT_FOUND, f"{object_id} is not in view here. Call find_objects first."
    )


def find_people(ctx: Ctx, description: str = "") -> list[dict[str, Any]]:
    """Look for people in the room Walkie is in now.

    Returns a list of objects with the keys id, name, pose, gesture, room,
    distance_m, bearing_deg, forward_m and left_m, which is empty if nobody
    matches. The name is null until the person has given it. Pass an id to
    hand_over.

    The numbers are measured from where Walkie stands now, in its own frame, and
    are null when the camera cannot tell. forward_m and left_m are what
    move_relative takes. To walk up to a person, stop about 0.8 m short of them so
    Walkie does not bump into them.

    Args:
        description: A name, pose or gesture, such as "Jane", "sitting person" or
            "waving person". Empty means everybody.
    """
    return [
        {
            "id": seen.id,
            "name": seen.name,
            "pose": seen.pose,
            "gesture": seen.gesture,
            "room": seen.room,
            **_position(seen.distance_m, seen.bearing_deg),
        }
        for seen in ctx.vision.people(description)
    ]


def _position(distance_m: float | None, bearing_deg: float | None) -> dict[str, Any]:
    """distance_m, bearing_deg, forward_m and left_m as the LLM reads them.

    Metres to the centimetre and degrees to the whole degree, the bearing wrapped
    again after rounding so it stays in [-180, 180). forward_m and left_m come from
    the unrounded numbers, and are null whenever either of them is.
    """
    forward_m = left_m = None
    if distance_m is not None and bearing_deg is not None:
        turn = math.radians(bearing_deg)
        forward_m = _round_m(distance_m * math.cos(turn))
        left_m = _round_m(distance_m * math.sin(turn))
    return {
        "distance_m": None if distance_m is None else _round_m(distance_m),
        "bearing_deg": None if bearing_deg is None else int(wrap_deg(round(bearing_deg))),
        "forward_m": forward_m,
        "left_m": left_m,
    }


def _round_m(value: float) -> float:
    # + 0.0 turns -0.0 into 0.0, so a point straight ahead never reads as "-0.0".
    return round(value, 2) + 0.0
