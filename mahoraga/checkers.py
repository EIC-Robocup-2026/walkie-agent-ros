"""Ground truth: whether an episode did what its command asked, from FakeRobot snapshots.

Only the harness reads these. Object commands are checked on the final world
state, and answers against the world as it started. FakeRobot's people do not
move and it has no follow verb, so a people command is checked on the records of
its successful calls, which say where the robot really was: looked in the right
room, said something there, went to the destination. What a line about "the time"
says cannot be checked at all.
"""

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from mahoraga.worlds import OPERATOR, at_source, object_matches, person_matches

Snapshot = Mapping[str, Any]
Call = Mapping[str, Any]
Seen = Callable[[Call], bool]

# Each measure is unique per kind in the fake arena, so a comparison has one answer.
MEASURES = {
    "biggest": ("size_cm", max),
    "largest": ("size_cm", max),
    "smallest": ("size_cm", min),
    "heaviest": ("weight_g", max),
    "lightest": ("weight_g", min),
    "thinnest": ("thickness_cm", min),
}
_NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()
_FILLER = {"person", "their", "to", "the", "arm"}
_MOVES = {"base.go_to", "base.move_relative"}


def check(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    """Whether the episode from `start` to `end` carried out `spec`, and why or why not."""
    return _CHECKERS[spec["kind"]](spec, start, end)


def _done(end: Snapshot) -> list[Call]:
    """The calls that succeeded, each with where the robot really was as it ended."""
    return [call for call in end["calls"] if call["error"] is None]


def _after_look(calls: list[Call], verb: str, seen: Seen) -> list[Call] | None:
    """The calls after the first `verb` where `seen` holds, or None if that never happened."""
    for index, call in enumerate(calls):
        if call["verb"] == verb and seen(call):
            return calls[index + 1 :]
    return None


def _said(calls: list[Call] | None, room: str | None = None) -> list[str]:
    """The lines spoken in `calls`, in `room` if given."""
    return [
        call["args"]["text"]
        for call in calls or []
        if call["verb"] == "speech.say" and room in (None, call["room"])
    ]


def _sees_place(start: Snapshot, place: str) -> Seen:
    room = start["places"][place]["room"]
    return lambda call: call["at"] in (place, room)


def _in_room(room: str) -> Seen:
    return lambda call: call["room"] == room


def _objects_at(start: Snapshot, place: str, category: str | None) -> list[Mapping[str, Any]]:
    return [
        item
        for item in start["objects"].values()
        if item["at"] == place and object_matches(item, category)
    ]


def _numbers(line: str) -> set[int]:
    words = re.findall(r"[a-z0-9]+", line.lower())
    return {int(word) for word in words if word.isdigit()} | {
        _NUMBER_WORDS.index(word) for word in words if word in _NUMBER_WORDS
    }


def _answered_number(lines: Iterable[str], count: int) -> tuple[bool, str]:
    """The last line that holds a number decides, so an earlier remark cannot pass as the answer."""
    with_numbers = [line for line in lines if _numbers(line)]
    if not with_numbers:
        return False, f"No number was said after looking. Expected {count}."
    return count in _numbers(with_numbers[-1]), f"Expected {count}. Said: {with_numbers[-1]!r}."


def _check_moved(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    dest = spec["dest"]
    picked = {call["args"]["object_id"] for call in _done(end) if call["verb"] == "arms.pick"}
    for object_id in sorted(picked):
        item = start["objects"].get(object_id)
        if item is None or not object_matches(item, spec["want"]):
            continue
        if not at_source(start["places"], item["at"], spec["source"]):
            continue
        at = end["objects"][object_id]["at"]
        if (
            (dest["to"] == "operator" and at == OPERATOR)
            or (dest["to"] == "place" and at == dest["place"])
            or (
                dest["to"] == "person"
                and at != OPERATOR
                and at in end["people"]
                and person_matches(end["people"][at], dest)
            )
        ):
            return True, f"{object_id} reached {at}."
    return False, f"No {spec['want']} from {spec['source']} was picked and delivered to {dest}."


def _check_count_objects(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    count = len(_objects_at(start, spec["place"], spec["category"]))
    after = _after_look(_done(end), "vision.objects", _sees_place(start, spec["place"]))
    return _answered_number(_said(after), count)


def _spoken(line: str) -> str:
    return line.replace("_", " ").lower()


def _check_best(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    field, pick = MEASURES[spec["comparison"]]
    items = _objects_at(start, spec["place"], spec["category"])
    best = pick(items, key=lambda item: item[field])["name"]
    after = _after_look(_done(end), "vision.objects", _sees_place(start, spec["place"]))
    naming = [line for line in _said(after) if any(item["name"] in _spoken(line) for item in items)]
    if not naming:
        return False, f"No line named an object after looking. Expected {best}."
    return best in _spoken(naming[-1]), f"Expected {best}. Said: {naming[-1]!r}."


def _check_count_people(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    selector = {"room": spec["room"], "description": spec["description"]}
    count = sum(
        1
        for person, who in start["people"].items()
        if person != OPERATOR and person_matches(who, selector)
    )
    after = _after_look(_done(end), "vision.people", _in_room(spec["room"]))
    return _answered_number(_said(after), count)


def _info_tokens(start: Snapshot, room: str, info: str) -> set[str]:
    """The words that must be said to report `info`, of the one other person in `room`."""
    who = next(w for person, w in start["people"].items() if person != OPERATOR and w["room"] == room)
    return set(re.findall(r"[a-z]+", who[info].lower())) - _FILLER


def _reports(lines: Iterable[str], tokens: set[str]) -> bool:
    return any(tokens <= set(re.findall(r"[a-z]+", line.lower())) for line in lines)


def _check_person_info(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    tokens = _info_tokens(start, spec["room"], spec["info"])
    after = _after_look(_done(end), "vision.people", _in_room(spec["room"]))
    return _reports(_said(after), tokens), f"Expected {sorted(tokens)} in a line after looking."


def _check_relay(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    tokens = _info_tokens(start, spec["room"], spec["info"])
    room, to_room = spec["room"], spec["to_room"]
    after = _after_look(_done(end), "vision.people", _in_room(room))
    if to_room != room:
        after = _after_look(after or [], "vision.people", _in_room(to_room))
    return _reports(_said(after, to_room), tokens), f"Expected {sorted(tokens)} said in {to_room}."


def _is_at(start: Snapshot, call: Call, place: str) -> bool:
    """Whether a call ended at `place`, or anywhere in it if `place` is a room."""
    if start["places"][place]["room"] == place:
        return call["room"] == place
    return call["at"] == place


def _after_finding(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> list[Call] | None:
    """The calls after a lookup at the requested place returned the requested person, or None."""
    wanted = {
        person
        for person, who in start["people"].items()
        if person != OPERATOR and person_matches(who, spec["person"])
    }

    def found(call: Call) -> bool:
        return _is_at(start, call, spec["place"]) and bool(wanted & set(call.get("returned", ())))

    return _after_look(_done(end), "vision.people", found)


def _check_talk(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    room = spec["person"]["room"]
    after = _after_finding(spec, start, end)
    if after is None:
        return False, f"No lookup at {spec['place']} returned {spec['person']}."
    return bool(_said(after, room)), f"Expected a line in {room} after finding the person."


def _check_escort(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    room, dest = spec["person"]["room"], spec["dest"]
    after = _after_finding(spec, start, end)
    if after is None:
        return False, f"No lookup at {spec['place']} returned {spec['person']}."
    if spec["greet"]:
        lines = (i for i, call in enumerate(after) if call["verb"] == "speech.say" and call["room"] == room)
        greeted = next(lines, None)
        if greeted is None:
            return False, f"Never greeted the person in {room}."
        after = after[greeted + 1 :]
    if dest is None:
        return True, "Found the person."
    # The robot is already there when the destination is the named origin or holds it.
    if at_source(start["places"], spec["place"], dest):
        return True, f"Found the person at {spec['place']}, which is at {dest}."
    reached = any(call["verb"] in _MOVES and _is_at(start, call, dest) for call in after)
    return reached, f"Expected to reach {dest} after finding the person."


_CHECKERS = {
    "move_object": _check_moved,
    "count_objects": _check_count_objects,
    "best_object": _check_best,
    "count_people": _check_count_people,
    "person_info": _check_person_info,
    "relay_info": _check_relay,
    "talk": _check_talk,
    "escort": _check_escort,
}
