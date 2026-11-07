"""Atomics that report what Walkie sees where it stands."""

from dataclasses import asdict
from typing import Any

from walkie_agent.capabilities import Ctx
from walkie_agent.errors import ErrorCode, StepError


def find_objects(ctx: Ctx, what: str = "") -> list[dict[str, Any]]:
    """Look for objects from where Walkie stands now.

    At a location Walkie sees what is on that location. At a room it sees what
    is on the locations in that room. Returns a list of objects with the keys
    id, name, category and place, which is empty if nothing matches. Pass an id
    to pick or object_info.

    Args:
        what: An object name or a category, such as "apple" or "fruits". Empty means everything.
    """
    return [
        {"id": seen.id, "name": seen.name, "category": seen.category, "place": seen.place}
        for seen in ctx.vision.objects(what)
    ]


def object_info(ctx: Ctx, object_id: str) -> dict[str, Any]:
    """Measure one object that is in view, to compare it with others.

    Returns an object with the keys id, name, category, place, size_cm,
    weight_g and thickness_cm. A measure Walkie cannot tell is null.

    Args:
        object_id: The id of an object, as find_objects returns it.
    """
    for seen in ctx.vision.objects():
        if seen.id == object_id:
            return asdict(seen)
    raise StepError(
        ErrorCode.NOT_FOUND, f"{object_id} is not in view here. Call find_objects first."
    )


def find_people(ctx: Ctx, description: str = "") -> list[dict[str, Any]]:
    """Look for people in the room Walkie is in now.

    Returns a list of objects with the keys id, name, pose, gesture and room,
    which is empty if nobody matches. The name is null until the person has
    given it. Pass an id to hand_over.

    Args:
        description: A name, pose or gesture, such as "Jane", "sitting person" or
            "waving person". Empty means everybody.
    """
    # Not asdict: the bearing stays out, so the tool text and the result the Mahoraga
    # runs measured are unchanged, and face_person reads it itself (decision S2-6).
    return [
        {
            "id": seen.id,
            "name": seen.name,
            "pose": seen.pose,
            "gesture": seen.gesture,
            "room": seen.room,
        }
        for seen in ctx.vision.people(description)
    ]
