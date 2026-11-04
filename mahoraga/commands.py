"""GPSR commands from the official generator, with what each one asks for.

The generator returns only a sentence. RecordingGenerator overrides its hooks to
keep the template, the follow-ups and the slot values, tagged by the part that
drew them, because the main template and a follow-up both use keys like "room".
parse() turns such a record into a spec, the JSON data the world builder, the
checkers and the oracle read. The generator comes from the pinned dependency, and
its code and data stay out of this repo: it has no licence file. The data
directory here holds only the fake arena's own names.

Spec kinds, where a person selector is {"room": room, "name": name} or
{"room": room, "description": pose or gesture}:

    move_object    want (object name or category), source (place or room),
                   dest ({"to": "operator"}, {"to": "place", "place": p} or
                   {"to": "person", ...selector})
    count_objects  category, place
    best_object    comparison, category (or None), place
    count_people   description, room
    person_info    info (name, pose or gesture), room
    relay_info     info, room, to_room
    talk           person, place, topic, greet
    escort         person, place, dest (place, room or None), greet
"""

import random
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from robocupathome_generator.gpsr_commands import CommandGenerator
from robocupathome_generator.knowledge import parse_data

from walkie_agent.capabilities.fake import CHULA_PAT14_6_PLACES, PLURALS

DATA_DIR = Path(__file__).parent / "data" / "chula_pat14_6"

# FakeRobot's people have no clothing and do not move between rooms, so nothing
# could carry out or check these. The dataset skips their seeds.
UNSUPPORTED = frozenset(
    {
        "guideClothPrsFromBeacToBeac",
        "greetClothDscInRm",
        "countClothPrsInRoom",
        "meetNameAtLocThenFindInRm",
    }
)
# Disjoint seed ranges, so no command is both in training and held out.
SPLITS = {"trial": 0, "train": 100_000, "heldout": 200_000}
# The generator's own names: "" is the competition's mix of both groups.
CATEGORIES = {"mixed": "", "objects": "objects", "people": "people"}

_SINGULAR = {plural: singular for singular, plural in PLURALS.items()}
_DEFERRED = re.compile(r"\\\{(loc2|room2|plcmtLoc2)\\\}")

Slot = Callable[[str, str], str]


class RecordingGenerator(CommandGenerator):
    """The official generator, recording what it draws for one command."""

    def generate(self, seed: int, category: str = "") -> dict[str, Any]:
        """The command for `seed`, with `category` as the generator names it. Global random is restored."""
        state = random.getstate()
        random.seed(seed)
        self._record: dict[str, Any] = {"template": None, "followups": [], "slots": []}
        self._part = "main"
        try:
            text = self.generate_command_start(category)
        finally:
            random.setstate(state)
        return {"seed": seed, "text": text, **self._record}

    def _weighted_choice(self, weighted_list):
        choice = super()._weighted_choice(weighted_list)
        self._record["template"] = choice
        return choice

    def _generate_followup(self, command, cmd_category=""):
        self._record["followups"].append(command)
        outer, self._part = self._part, command
        try:
            return super()._generate_followup(command, cmd_category)
        finally:
            self._part = outer

    def insert_placeholders(self, ph):
        value = super().insert_placeholders(ph)
        key = ph.strip("{}")
        # A value in braces is a placeholder that a later pass fills, and verbs
        # and prepositions say nothing about the task.
        if not value.startswith("{") and key not in self.verb_dict and key not in self.prep_dict:
            self._record["slots"].append([self._part, key, value])
        return value

    def _resolve_duplicates(self, text):
        resolved = super()._resolve_duplicates(text)
        escaped = re.escape(text)
        if keys := _DEFERRED.findall(escaped):
            values = re.fullmatch(_DEFERRED.sub("(.+)", escaped), resolved).groups()
            self._record["slots"] += [["dup", key, value] for key, value in zip(keys, values)]
        return resolved


def _room_of(place: str) -> str:
    return CHULA_PAT14_6_PLACES[place]["room"]


def _bare_place(phrase: str) -> str:
    """The place in an {inRoom_atLoc} value, which comes with its "in the" or "at the"."""
    return re.sub(r"^(in|at) the ", "", phrase)


def _person_at(place: str, **selector: str) -> dict[str, str]:
    return {"room": _room_of(place), **selector}


def parse(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The spec of a recorded command, or None if the fake world cannot express it."""
    template, chain = record["template"], record["followups"]
    if template in UNSUPPORTED:
        return None
    slots = {(part, key): value for part, key, value in record["slots"]}

    def slot(part: str, key: str) -> str:
        return slots[(part, key)]

    for build in (_about_objects, _about_people):
        spec = build(template, chain, slot, slots)
        if spec is not None:
            return spec
    raise ValueError(f"No spec for template {template} with follow-ups {chain}.")


def _about_objects(template: str, chain: list[str], slot: Slot, slots: dict) -> dict | None:
    if template == "countObjOnPlcmt":
        return {
            "kind": "count_objects",
            "category": _SINGULAR[slot("main", "plurCat")],
            "place": slot("main", "plcmtLoc"),
        }
    if template in ("tellObjPropOnPlcmt", "tellCatPropOnPlcmt"):
        return {
            "kind": "best_object",
            "comparison": slot("main", "objComp"),
            "category": slots.get(("main", "singCat")),
            "place": slot("main", "plcmtLoc"),
        }
    if template == "bringMeObjFromPlcmt":
        want, source = slot("main", "obj"), slot("main", "plcmtLoc")
        return {"kind": "move_object", "want": want, "source": source, "dest": {"to": "operator"}}
    if template == "goToLoc" and "findObj" in chain:
        want, source = slot("findObj", "obj_singCat"), slot("main", "loc_room")
    elif template == "takeObjFromPlcmt":
        want, source = slot("main", "obj_singCat"), slot("main", "plcmtLoc")
    elif template == "findObjInRoom":
        want, source = slot("main", "obj_singCat"), slot("main", "room")
    else:
        return None
    return {"kind": "move_object", "want": want, "source": source, "dest": _object_dest(chain, slot)}


def _object_dest(chain: list[str], slot: Slot) -> dict[str, str]:
    """Where the hasObj follow-up, the last of the chain, leaves the object."""
    last = chain[-1]
    if last == "putObjInTrash":
        return {"to": "place", "place": "trash"}
    if last == "placeObjOnPlcmt":
        return {"to": "place", "place": slot("dup", "plcmtLoc2")}
    if last == "deliverObjToMe":
        return {"to": "operator"}
    if last == "deliverObjToPrsInRoom":
        return {"to": "person", "room": slot(last, "room"), "description": slot(last, "gestPers_posePers")}
    return {"to": "person", "room": slot(last, "room"), "name": slot(last, "name")}


def _about_people(template: str, chain: list[str], slot: Slot, slots: dict) -> dict | None:
    if template == "countPrsInRoom":
        plural = slot("main", "gestPersPlur_posePersPlur")
        return {
            "kind": "count_people",
            "description": plural.replace("persons", "person"),
            "room": slot("main", "room"),
        }
    if template == "tellPrsInfoInLoc":
        place = _bare_place(slot("main", "inRoom_atLoc"))
        return {"kind": "person_info", "info": slot("main", "persInfo"), "room": _room_of(place)}
    if template == "tellPrsInfoAtLocToPrsAtLoc":
        return {
            "kind": "relay_info",
            "info": slot("main", "persInfo"),
            "room": _room_of(slot("main", "loc")),
            "to_room": _room_of(slot("dup", "loc2")),
        }
    if template == "talkInfoToGestPrsInRoom":
        room = slot("main", "room")
        person = _person_at(room, description=slot("main", "gestPers"))
        return {"kind": "talk", "person": person, "place": room, "topic": slot("main", "talk"), "greet": False}
    if template in ("followNameFromBeacToRoom", "guideNameFromBeacToBeac"):
        place = slot("main", "loc")
        dest = slots.get(("main", "room")) or slot("main", "loc_room")
        person = _person_at(place, name=slot("main", "name"))
        return {"kind": "escort", "person": person, "place": place, "dest": dest, "greet": False}
    if template == "guidePrsFromBeacToBeac":
        place = slot("main", "loc")
        person = _person_at(place, description=slot("main", "gestPers_posePers"))
        return {"kind": "escort", "person": person, "place": place, "dest": slot("main", "loc_room"), "greet": False}
    if template == "followPrsAtLoc":
        place = _bare_place(slot("main", "inRoom_atLoc"))
        person = _person_at(place, description=slot("main", "gestPers_posePers"))
        return {"kind": "escort", "person": person, "place": place, "dest": None, "greet": False}
    return _found_person(template, chain, slot, slots)


def _found_person(template: str, chain: list[str], slot: Slot, slots: dict) -> dict | None:
    """A command that finds a person, then does what its foundPers follow-up says."""
    if template == "goToLoc" and "meetName" in chain:
        place, selector = slot("main", "loc_room"), {"name": slot("meetName", "name")}
    elif template == "goToLoc" and "findPrs" in chain:
        place, selector = slot("main", "loc_room"), {"description": slot("findPrs", "gestPers_posePers")}
    elif template == "findPrsInRoom":
        place, selector = slot("main", "room"), {"description": slot("main", "gestPers_posePers")}
    elif template in ("meetPrsAtBeac", "greetNameInRm"):
        place, selector = slot("main", "room"), {"name": slot("main", "name")}
    else:
        return None
    base = {"person": _person_at(place, **selector), "place": place, "greet": template == "greetNameInRm"}
    if "talkInfo" in chain:
        return {"kind": "talk", **base, "topic": slot("talkInfo", "talk")}
    return {"kind": "escort", **base, "dest": slots.get(("dup", "loc2")) or slots.get(("dup", "room2"))}


def generator() -> RecordingGenerator:
    return RecordingGenerator(parse_data(str(DATA_DIR)))


def commands(split: str, n: int, category: str = "mixed") -> list[dict[str, Any]]:
    """The first `n` usable commands of a split, each a record plus its spec.

    The seed is the sample's id. A larger `n` extends a smaller one, so a small
    slice's commands are the first ones of the full trial.
    """
    made = generator()
    found: list[dict[str, Any]] = []
    seed = SPLITS[split]
    while len(found) < n:
        record = made.generate(seed, CATEGORIES[category])
        spec = parse(record)
        if spec is not None:
            found.append({**record, "spec": spec})
        seed += 1
    return found
