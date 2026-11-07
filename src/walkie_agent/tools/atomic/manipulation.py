"""Atomics that use the arms: one held object at a time."""

from walkie_agent.capabilities import Ctx


def pick(ctx: Ctx, object_id: str) -> str:
    """Pick up one object that is in view. Walkie holds one object at a time.

    Returns a sentence that names the object now held.

    Args:
        object_id: The id of an object, as find_objects returns it.
    """
    return f"Holding {ctx.arms.pick(object_id)}."


def place_on(ctx: Ctx, place: str) -> str:
    """Put the held object down on the location Walkie stands at. Call go_to first.

    Returns a sentence that names the object and where it now is.

    Args:
        place: The name of a location, such as "shelf". A room cannot hold an object.
    """
    return f"Placed {ctx.arms.place_on(place)} on {place.strip()}."


def hand_over(ctx: Ctx, person_id: str) -> str:
    """Give the held object to a person in the room Walkie is in.

    Returns a sentence that names the object and who now has it.

    Args:
        person_id: The id of a person, as find_people returns it.
    """
    return f"Handed {ctx.arms.hand_over(person_id)} to {person_id}."
