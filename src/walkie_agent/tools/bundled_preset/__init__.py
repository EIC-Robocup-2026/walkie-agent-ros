"""Stand-in bundles for the Mahoraga gym's arm D, over the same atomics every other arm uses.

The plan's arm D is per-step tools plus the team's reviewed bundles. None exist
yet, so these stand in for them. Replace the list once real bundles land, because
until then arm D measures this file and not the team's curated skills.
"""

from collections.abc import Callable
from typing import Any

from walkie_agent.capabilities import Ctx
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.tools.atomic import (
    find_objects,
    find_people,
    go_to,
    hand_over,
    object_info,
    pick,
    place_on,
)

_OPERATOR = "operator"
_INSTRUCTION_POINT = "instruction point"


def fetch_object(ctx: Ctx, what: str, where: str) -> str:
    """Go to a location or room, find the first matching object and pick it up.

    Returns a sentence that names the object now held.

    Args:
        what: An object name or a category, such as "apple" or "fruit".
        where: A location or a room to look in.
    """
    go_to(ctx, where)
    found = find_objects(ctx, what)
    if not found:
        raise StepError(ErrorCode.NOT_FOUND, f"There is no {what} at {where}.")
    return pick(ctx, found[0]["id"])


def put_object_on(ctx: Ctx, place: str) -> str:
    """Go to a location and put the held object down on it.

    Returns a sentence that names the object and where it now is.

    Args:
        place: The name of a location, such as "shelf" or "trash".
    """
    go_to(ctx, place)
    return place_on(ctx, place)


def give_to_operator(ctx: Ctx) -> str:
    """Go back to the instruction point and give the held object to the operator.

    Returns a sentence that names the object and who now has it.
    """
    go_to(ctx, _INSTRUCTION_POINT)
    return hand_over(ctx, _OPERATOR)


def give_to_person(ctx: Ctx, room: str, description: str) -> str:
    """Go to a room, find a matching person and give them the held object.

    Returns a sentence that names the object and who now has it.

    Args:
        room: The room the person is in.
        description: A name, pose or gesture, such as "Jane" or "waving person".
    """
    go_to(ctx, room)
    found = [person for person in find_people(ctx, description) if person["id"] != _OPERATOR]
    if not found:
        raise StepError(ErrorCode.NOT_FOUND, f"There is no {description} in {room}.")
    return hand_over(ctx, found[0]["id"])


def count_objects(ctx: Ctx, what: str, where: str) -> int:
    """Go to a location or room and count the matching objects there.

    Returns the count.

    Args:
        what: An object name or a category. Empty means everything.
        where: A location or a room to look in.
    """
    go_to(ctx, where)
    return len(find_objects(ctx, what))


def best_object(ctx: Ctx, what: str, where: str, measure: str, most: bool) -> dict[str, Any]:
    """Go to a location or room and find the object with the most or least of a measure.

    Returns an object with the keys id, name, category, place, size_cm, weight_g
    and thickness_cm.

    Args:
        what: An object name or a category. Empty means everything.
        where: A location or a room to look in.
        measure: One of size_cm, weight_g or thickness_cm.
        most: True for the most of the measure, False for the least.
    """
    go_to(ctx, where)
    measured = [object_info(ctx, seen["id"]) for seen in find_objects(ctx, what)]
    if not measured:
        raise StepError(ErrorCode.NOT_FOUND, f"There is no {what or 'object'} at {where}.")
    return (max if most else min)(measured, key=lambda item: item[measure])


BUNDLES: list[Callable[..., Any]] = [
    fetch_object,
    put_object_on,
    give_to_operator,
    give_to_person,
    count_objects,
    best_object,
]
