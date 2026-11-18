"""A seeded FakeRobot world in which one command can be carried out.

It starts from chula_pat14_6(seed) and adds only what the spec needs, so every
arm of the same (command, epoch) meets the same world.
"""

import hashlib
import random
from collections.abc import Mapping
from typing import Any

from walkie_agent.capabilities.fake import GESTURES, OBJECTS, POSES, SURFACES, chula_pat14_6

OPERATOR = "operator"
# Outside the generator's poses and gestures, and not one of its names, so no
# command's selector can ever match the operator.
_OPERATOR_LOOK = {"name": OPERATOR, "pose": "standing at the instruction point", "gesture": "no gesture"}


def world_seed(sample_id: int, epoch: int) -> int:
    """The seed of one (command, epoch), independent of the arm."""
    return int(hashlib.sha256(f"{sample_id}:{epoch}".encode()).hexdigest()[:16], 16)


def object_matches(item: Mapping[str, Any], want: str | None) -> bool:
    """Whether an object is the kind or the category `want`. None means any."""
    return want is None or want in (item["name"], item["category"])


def person_matches(who: Mapping[str, Any], selector: Mapping[str, str]) -> bool:
    """Whether a person is in the selector's room with its name, or its pose or gesture."""
    if who["room"] != selector["room"]:
        return False
    if "name" in selector:
        return who["name"] == selector["name"]
    return selector["description"] in (who["pose"], who["gesture"])


def at_source(places: Mapping[str, Mapping[str, Any]], at: str, source: str) -> bool:
    """Whether a holder is the place `source`, or a place in the room `source`."""
    if at == source:
        return True
    return places[source]["room"] == source and at in places and places[at]["room"] == source


def build_world(spec: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """The world description for `spec`, a JSON-able input to FakeRobot."""
    world = chula_pat14_6(seed)
    world["people"][OPERATOR].update(_OPERATOR_LOOK)
    rng = random.Random(seed)
    kind = spec["kind"]
    if kind == "move_object":
        _ensure_object(world, spec["want"], spec["source"], rng)
        if spec["dest"]["to"] == "person":
            _ensure_person(world, spec["dest"], rng)
    elif kind == "count_objects":
        _ensure_object(world, spec["category"], spec["place"], rng)
    elif kind == "best_object":
        _ensure_two_kinds(world, spec["category"], spec["place"], rng)
    elif kind == "count_people":
        _ensure_person(world, {"room": spec["room"], "description": spec["description"]}, rng)
    elif kind in ("person_info", "relay_info"):
        _leave_one_person(world, spec["room"])
    else:  # talk and escort
        _ensure_person(world, spec["person"], rng)
    return world


def _add_object(world: dict[str, Any], name: str, at: str) -> None:
    n = 1
    while (object_id := f"{name.replace(' ', '_')}_{n}") in world["objects"]:
        n += 1
    world["objects"][object_id] = {"name": name, **OBJECTS[name], "at": at}


def _kinds(want: str | None) -> list[str]:
    return [name for name, item in OBJECTS.items() if object_matches({"name": name, **item}, want)]


def _ensure_object(world: dict[str, Any], want: str, source: str, rng: random.Random) -> None:
    """At least one object matching `want` at the place, or on a surface of the room, `source`."""
    places = world["places"]
    if any(
        object_matches(item, want) and at_source(places, item["at"], source)
        for item in world["objects"].values()
    ):
        return
    in_room = [surface for surface in SURFACES if places[surface]["room"] == source]
    _add_object(world, rng.choice(_kinds(want)), rng.choice(in_room) if in_room else source)


def _ensure_two_kinds(world: dict[str, Any], category: str | None, place: str, rng: random.Random) -> None:
    """At least two kinds at `place`, so a comparison has something to decide."""
    found = {
        item["name"]
        for item in world["objects"].values()
        if item["at"] == place and object_matches(item, category)
    }
    missing = [name for name in _kinds(category) if name not in found]
    for name in rng.sample(missing, max(0, 2 - len(found))):
        _add_object(world, name, place)


def _others_in(world: Mapping[str, Any], room: str) -> list[str]:
    return sorted(
        person
        for person, who in world["people"].items()
        if person != OPERATOR and who["room"] == room
    )


def _ensure_person(world: dict[str, Any], selector: Mapping[str, str], rng: random.Random) -> None:
    """Someone other than the operator who fits the selector. Names stay unique."""
    people = world["people"]
    here = _others_in(world, selector["room"])
    if any(person_matches(people[person], selector) for person in here):
        return
    chosen = rng.choice(here)
    if "name" in selector:
        name = selector["name"]
        holder = next((person for person, who in people.items() if who["name"] == name), None)
        if holder is not None:
            people[holder]["name"] = people[chosen]["name"]
        people[chosen]["name"] = name
    elif selector["description"] in POSES:
        people[chosen]["pose"] = selector["description"]
    elif selector["description"] in GESTURES:
        people[chosen]["gesture"] = selector["description"]
    else:
        raise ValueError(f"Not a pose or a gesture: {selector['description']!r}.")


def _leave_one_person(world: dict[str, Any], room: str) -> None:
    """Only one person besides the operator in `room`, so "the person" means someone in particular."""
    people = world["people"]
    other_room = next(who["room"] for who in people.values() if who["room"] != room)
    for person in _others_in(world, room)[1:]:
        people[person]["room"] = other_room
