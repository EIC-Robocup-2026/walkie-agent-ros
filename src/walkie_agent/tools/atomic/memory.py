"""Atomics that remember the people Walkie meets, and recall the objects it saw before.

The people atomics store and recognize people by name (ADR-0070). The object atomics
read the object memory (decisions SG-16 to SG-21): where objects were when Walkie
last saw them, never what is in view now.
"""

from typing import Any

from walkie_agent.capabilities import Ctx, RecalledObject
from walkie_agent.tools.atomic.navigation import approach_place

# How many objects one recall_object or list_room_objects answer lists at most.
RECALL_LIMIT = 10
ROOM_LIMIT = 50
# The states recall_object lists: where an object stands, or that a hand holds it
# (decision PK-70). list_room_objects lists only objects that stand somewhere.
RECALL_STATES = ("present", "held")


def remember_person(
    ctx: Ctx, person_id: str, name: str, facts: dict[str, str] | None = None
) -> dict[str, Any]:
    """Remember a person in view by their name, so Walkie knows them again later.

    Call it when the person tells you their name, while they are still in view:
    Walkie stores the face and appearance it saw of them. Returns an object with
    the keys id, name, facts, faces and appearances: the name and facts as
    stored, and how many face and appearance samples were stored. From then on
    find_people and recognize_person give their name. After a TIMEOUT the
    person may have been stored already, so remembering them again may store
    them twice.

    Args:
        person_id: The id of a person, as find_people returns it.
        name: The person's name, as they said it.
        facts: What the person told you about themselves, as a JSON object of
            text values, such as {"drink": "tea"}. At most 8 facts.
    """
    remembered = ctx.memory.remember_person(person_id, name, facts)
    return {
        "id": remembered.person_id,
        "name": remembered.name,
        "facts": dict(remembered.facts),
        "faces": remembered.faces,
        "appearances": remembered.appearances,
    }


def recognize_person(ctx: Ctx, person_id: str) -> dict[str, Any]:
    """Say who a person in view is, from the people Walkie has remembered.

    Returns an object with the keys id, name, facts and matched_by. For a
    person Walkie remembered, the name and facts are what it stored, and
    matched_by is "face" or "appearance", whichever recognized them. For
    anyone else the name and matched_by are null and the facts are empty.
    Never guess a name that is null.

    Args:
        person_id: The id of a person, as find_people returns it.
    """
    recognized = ctx.memory.recognize(person_id)
    return {
        "id": recognized.person_id,
        "name": recognized.name,
        "facts": dict(recognized.facts),
        "matched_by": recognized.matched_by,
    }


def recall_object(ctx: Ctx, name: str) -> dict[str, Any]:
    """Recall where Walkie saw objects of a kind before, from its object memory.

    It tells where objects were when Walkie last saw them, not what is in view
    now, so one may have moved since. Returns an object with the keys objects and
    complete: objects is a list of at most 10 objects with the keys id, name,
    category, room, on, near_place, last_seen_s_ago and held, and complete is
    false when Walkie remembers more such objects than the list holds. In each
    object, on names the object it stood on, such as "table 15", near_place the
    named place nearest it, last_seen_s_ago how many seconds ago Walkie last saw
    it, and held is true when Walkie holds it in a hand, where room, on and
    near_place are null. Any of room, on, near_place and last_seen_s_ago is null
    when unknown. An empty list means Walkie remembers no such object. The ids
    are memory ids for go_to_object, not ids from find_objects.

    Args:
        name: What to recall: a name such as "cup", or a category such as
            "drinks". An empty name recalls objects of any kind, at most 10;
            list_room_objects lists a whole room.
    """
    # One more than the list holds tells whether the list is all of them.
    found = ctx.object_memory.search(name, limit=RECALL_LIMIT + 1, states=RECALL_STATES)
    return {
        "objects": [_remembered(ctx, recalled) for recalled in found[:RECALL_LIMIT]],
        "complete": len(found) <= RECALL_LIMIT,
    }


def list_room_objects(ctx: Ctx, room: str) -> dict[str, Any]:
    """List the objects Walkie remembers seeing in one room, from its object memory.

    It tells what was there when Walkie last looked, not what is in view now.
    Returns an object with the keys room, objects and complete: objects is a
    list of at most 50 objects with the same keys as in recall_object's answer,
    and complete is false when the room holds more objects than the list.

    Args:
        room: The name of a room, such as "living room".
    """
    contents = ctx.object_memory.room(room, limit=ROOM_LIMIT)
    return {
        "room": contents.room,
        "objects": [_remembered(ctx, recalled) for recalled in contents.objects],
        "complete": contents.complete,
    }


def _remembered(ctx: Ctx, recalled: RecalledObject) -> dict[str, Any]:
    """One recalled object as the LLM reads it. Its position never leaves (ADR-0010).

    A held object is where the hand is, so the room, support and place the memory
    last gave it are not where it is now.
    """
    seen = recalled.seen_s_ago
    held = recalled.state == "held"
    place = None if held else approach_place(ctx, recalled)
    return {
        "id": recalled.id,
        "name": recalled.name,
        "category": recalled.category,
        "room": None if held else recalled.room,
        "on": None if held else recalled.on,
        "near_place": None if place is None else place.name,
        "last_seen_s_ago": None if seen is None else round(seen),
        "held": held,
    }
