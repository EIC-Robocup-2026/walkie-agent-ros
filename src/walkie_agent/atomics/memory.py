"""Atomics that remember the people Walkie meets, by name, and say who they are later."""

from typing import Any

from walkie_agent.capabilities import Ctx


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
