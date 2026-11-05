"""Ground truth: whether an episode did what its command asked, from FakeRobot snapshots.

Only the harness reads these. Object commands are checked on the final world
state, and answers against the world as it started. FakeRobot's people do not
move and it has no follow verb, so a people command is checked on the records of
its successful calls, which say where the robot really was and what a lookup
returned.

An answer is the first line that mentions a candidate after the last look that
showed what it is about, and it must assert one candidate, the right one. Negation
and corrections are not parsed. A talk needs the requested person returned at the
requested place, then some line in their room that says the fact its topic asks
for, from the start snapshot's facts. Any such line counts, not only the first.
"Something about yourself" takes any line in the room, since what it says is not
checked. docs/architecture/mahoraga-demo.md lists the rules and known limits.
"""

import re
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime, timedelta
from functools import partial
from typing import Any

from mahoraga.worlds import OPERATOR, at_source, object_matches, person_matches
from walkie_agent.atomics.facts import MONTHS, WEEKDAYS
from walkie_agent.capabilities.fake import NAMES

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
# Generated counts are small, so a word past twelve is a wrong answer, but still an answer.
_NUMBER_WORDS = dict(
    zip(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split(),
        range(20),
    ),
    twenty=20, thirty=30, forty=40, fifty=50, sixty=60, seventy=70, eighty=80, ninety=90, hundred=100,
    single=1, couple=2, pair=2,
)
_MOVES = {"base.go_to", "base.move_relative"}
_RAISING = {"raising", "raises", "raise", "raised", "lifting", "lifts", "lift", "lifted", "up", "aloft", "high"}
_POINTING = {"pointing", "points", "point", "pointed", "indicating", "indicates", "indicate", "gesturing", "gestures"}
_ARM = {"arm", "arms", "forearm", "forearms", "hand", "hands"}
# Words stay unattached to each other, so a line naming a leg says no raised arm, even
# with a hand in it. A true arm answer that also mentions a leg fails with it.
_NOT_ARM = {"leg", "legs", "knee", "knees", "foot", "feet"}
# How ordinary speech says each pose and gesture. A line says a value when it holds
# a word from each of the value's groups.
_INFO_WORDS = {
    "pose": {
        "sitting person": [{"sitting", "sits", "sit", "seated", "sat"}],
        "standing person": [{"standing", "stands", "stand", "upright", "feet"}],
        "lying person": [
            {
                "lying", "lies", "lie", "laying", "reclining", "reclined",
                "horizontal", "horizontally", "stretched", "prone", "supine",
            }
        ],
    },
    "gesture": {
        "waving person": [{"waving", "waves", "wave", "waved"}],
        "person raising their left arm": [_RAISING, {"left"}, _ARM],
        "person raising their right arm": [_RAISING, {"right"}, _ARM],
        "person pointing to the left": [_POINTING, {"left"}],
        "person pointing to the right": [_POINTING, {"right"}],
    },
}


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


def _after_last_look(
    calls: list[Call],
    verb: str,
    seen: Seen,
    needed: set[str],
    candidates: Callable[[str], set],
    room: str | None = None,
) -> list[Call] | None:
    """The calls after the last qualifying `verb` that an answer follows.

    A look qualifies if `seen` holds, it returned any of `needed` and all of it has been
    returned by then, so a later look at something else is not looking again. With
    nothing needed, any look there qualifies. An answer is a line in `room`, if given,
    that names one of `candidates`, so a look with none after it leaves an earlier answer
    standing. With no answer at all, these are the calls after the last qualifying look,
    or None if there was none.
    """
    shown: set[str] = set()
    tails = []
    for index, call in enumerate(calls):
        if call["verb"] == verb and seen(call):
            returned = set(call.get("returned", ()))
            shown |= returned
            if needed <= shown and (returned & needed or not needed):
                tails.append(calls[index + 1 :])
    for tail in reversed(tails):
        if any(candidates(line) for line in _said(tail, room)):
            return tail
    return tails[-1] if tails else None


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


def _objects_at(start: Snapshot, place: str, category: str | None) -> dict[str, Mapping[str, Any]]:
    return {
        object_id: item
        for object_id, item in start["objects"].items()
        if item["at"] == place and object_matches(item, category)
    }


def _people_matching(start: Snapshot, selector: Mapping[str, str]) -> set[str]:
    return {
        person
        for person, who in start["people"].items()
        if person != OPERATOR and person_matches(who, selector)
    }


def _numbers(line: str) -> set[float | str]:
    """The numbers a line says, each token read whole: "1.0" is 1, "-1" is -1 and "3rd" is none.

    A compound of numbers stays as its text, which no count equals, so "twenty-one" is
    a wrong answer, and neither a "one" nor no answer at all.
    """
    said: set[float | str] = set()
    for token in re.findall(r"-?\d+\.\d+|-?\w+(?:-\w+)*", line.lower()):
        if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
            said.add(float(token))
        elif token in _NUMBER_WORDS:
            said.add(_NUMBER_WORDS[token])
        elif all(part in _NUMBER_WORDS or part.isdigit() for part in token.split("-")):
            said.add(token)
    return said


def _named(names: Iterable[str], line: str) -> set[str]:
    """Which of `names` a line says as whole words, in any case, with or without an underscore."""
    spoken = line.replace("_", " ").lower()
    return {name for name in names if re.search(rf"\b{re.escape(name.lower())}s?\b", spoken)}


def _info_said(start: Snapshot, info: str, line: str) -> set[str]:
    """Which known values of `info` a line says. Anything else in it is not a candidate."""
    if info == "name":
        others = (who["name"] for person, who in start["people"].items() if person != OPERATOR)
        return _named({*NAMES, *others}, line)
    words = set(re.findall(r"[a-z]+", line.lower()))
    return {
        value
        for value, groups in _INFO_WORDS[info].items()
        if all(words & group for group in groups) and not (_ARM in groups and words & _NOT_ARM)
    }


def _first_answer(lines: Iterable[str], candidates: Callable[[str], set], right: Any) -> tuple[bool, str]:
    """The first line that mentions a candidate decides: it must assert `right` and no other.

    Later lines go unread, so a closing remark cannot undo the answer. Nor can a
    correction: a robot that wants to change its answer has to look again first.
    """
    for line in lines:
        said = candidates(line)
        if said:
            return said == {right}, f"Expected {right}. Said: {line!r}."
    return False, f"No answer was said after a look that showed it. Expected {right}."


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
    place = spec["place"]
    items = _objects_at(start, place, spec["category"])
    after = _after_last_look(_done(end), "vision.objects", _sees_place(start, place), set(items), _numbers)
    return _first_answer(_said(after), _numbers, len(items))


def _check_best(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    field, pick = MEASURES[spec["comparison"]]
    place = spec["place"]
    items = _objects_at(start, place, spec["category"])
    best = pick(items.values(), key=lambda item: item[field])["name"]
    known = {item["name"] for item in _objects_at(start, place, None).values()}
    names = partial(_named, known)
    after =_after_last_look(_done(end), "vision.objects", _sees_place(start, place), set(items), names)
    return _first_answer(_said(after), names, best)


def _check_count_people(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    room = spec["room"]
    counted = _people_matching(start, {"room": room, "description": spec["description"]})
    after = _after_last_look(_done(end), "vision.people", _in_room(room), counted, _numbers)
    return _first_answer(_said(after), _numbers, len(counted))


def _only_other(start: Snapshot, room: str) -> tuple[str, Mapping[str, Any]]:
    """The one person besides the operator that the world builder leaves in `room`."""
    return next(
        (person, who)
        for person, who in start["people"].items()
        if person != OPERATOR and who["room"] == room
    )


def _check_person_info(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    room, info = spec["room"], spec["info"]
    person, who = _only_other(start, room)
    values = partial(_info_said, start, info)
    after =_after_last_look(_done(end), "vision.people", _in_room(room), {person}, values)
    return _first_answer(_said(after), values, who[info])


def _check_relay(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    """Learn from the source, know the recipient and be with them, in whichever order, then say it."""
    room, to_room, info = spec["room"], spec["to_room"], spec["info"]
    calls = _done(end)
    person, who = _only_other(start, room)
    values = partial(_info_said, start, info)
    learned =_after_last_look(calls, "vision.people", _in_room(room), {person}, values, to_room)
    recipients = {
        other
        for other, them in start["people"].items()
        if other != OPERATOR and them["room"] == to_room
    }

    def knows_recipient(call: Call) -> bool:
        return call["room"] == to_room and bool(recipients & set(call.get("returned", ())))

    met = _after_look(calls, "vision.people", knows_recipient)
    if learned is None or met is None:
        return False, f"Expected a look that returned someone in each of {room} and {to_room}."
    # Both are tails of the calls, so the shorter one comes after both looks.
    after_both = min(learned, met, key=len)
    return _first_answer(_said(after_both, to_room), values, who[info])


def _is_at(start: Snapshot, call: Call, place: str) -> bool:
    """Whether a call ended at `place`, or anywhere in it if `place` is a room."""
    if start["places"][place]["room"] == place:
        return call["room"] == place
    return call["at"] == place


def _after_finding(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> list[Call] | None:
    """The calls after a lookup at the requested place returned the requested person, or None."""
    wanted = _people_matching(start, spec["person"])

    def found(call: Call) -> bool:
        return _is_at(start, call, spec["place"]) and bool(wanted & set(call.get("returned", ())))

    return _after_look(_done(end), "vision.people", found)


# A fact a talk line must say: whether a line says it, and how a reason names it.
Says = tuple[Callable[[str], bool], str]

# The talk topic whose line is not checked.
ABOUT_YOURSELF = "something about yourself"
# "a.m." or "pm" after a time, in any case, dots optional.
_HALF_DAY = r"(?:\s*([ap])\.?\s?m\b\.?)?"
_CLOCK = re.compile(rf"(?<![\d:])(\d{{1,2}}):(\d{{2}})(?![\d:]){_HALF_DAY}")
_OCLOCK = re.compile(rf"(?<![\d:])(\d{{1,2}})\s*o['’]?\s?clock\b{_HALF_DAY}")
_DAY_DIGITS = re.compile(r"(?<![\d:])(\d{1,2})(?:st|nd|rd|th)?(?![\d:])")
_ORDINAL = r"(?:st|nd|rd|th)?"
_ONES = (
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_ONES_TH = (
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
    "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth"
).split()


def _day_words() -> dict[str, int]:
    """The English cardinals and ordinals of 1 to 31, compounds with a hyphen or a space."""
    words = {}
    for n, (cardinal, ordinal) in enumerate(zip(_ONES, _ONES_TH), start=1):
        words[cardinal] = words[ordinal] = n
    for tens, cardinal, ordinal, last in ((20, "twenty", "twentieth", 9), (30, "thirty", "thirtieth", 1)):
        words[cardinal] = words[ordinal] = tens
        for n in range(1, last + 1):
            for joint in ("-", " "):
                words[f"{cardinal}{joint}{_ONES[n - 1]}"] = tens + n
                words[f"{cardinal}{joint}{_ONES_TH[n - 1]}"] = tens + n
    return words


_DAY_WORDS = _day_words()
# Longest first, so "twenty one" reads as 21 and not as a "one".
_DAY_WORD = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, _DAY_WORDS), key=len, reverse=True)) + r")\b"
)


def _says_time(now: datetime) -> Says:
    """The time as H:MM or HH:MM in 24-hour form, or as h:MM in 12-hour form.

    A 12-hour time may end in am or pm, which must then be right. On the hour, "h
    o'clock" says it too. An hour with a leading zero and no am or pm reads as 24-hour
    only, so "02:05" is not 14:05. Minutes in words are not read.
    """
    hour, minute = now.hour, now.minute
    twelve, half = hour % 12 or 12, "a" if hour < 12 else "p"

    def right_hour(digits: str, said_half: str) -> bool:
        if said_half:
            return int(digits) == twelve and said_half == half
        return int(digits) == hour or (int(digits) == twelve and not digits.startswith("0"))

    def says(line: str) -> bool:
        spoken = line.lower()
        if any(int(m) == minute and right_hour(h, s) for h, m, s in _CLOCK.findall(spoken)):
            return True
        return minute == 0 and any(
            int(h) == twelve and s in ("", half) for h, s in _OCLOCK.findall(spoken)
        )

    return says, f"{hour:02d}:{minute:02d}"


def _says_weekday(day: date) -> Says:
    name = WEEKDAYS[day.weekday()]
    pattern = re.compile(rf"\b{name}\b", re.IGNORECASE)
    return (lambda line: bool(pattern.search(line))), name


def _says_date(day: date) -> Says:
    """A month name and the day's number in either order, with an optional ordinal suffix.

    "5 November", "5th of November", "November 5th" and "November the 5th" all say
    5 November. The year is not read.
    """
    month = MONTHS[day.month - 1]
    number = rf"(?<![\d:])0?{day.day}{_ORDINAL}(?![\d:])"
    pattern = re.compile(
        rf"{number}\s+(?:of\s+)?{month}\b|\b{month}\s+(?:the\s+)?{number}", re.IGNORECASE
    )
    return (lambda line: bool(pattern.search(line))), f"{day.day} {month}"


def _says_day_number(day: date) -> Says:
    """The day of the month as digits with an optional ordinal suffix, or as an English word."""

    def says(line: str) -> bool:
        spoken = line.lower()
        said = {int(digits) for digits in _DAY_DIGITS.findall(spoken)}
        said |= {_DAY_WORDS[word] for word in _DAY_WORD.findall(spoken)}
        return day.day in said

    return says, str(day.day)


def _says_phrase(text: str) -> Says:
    """`text` as whole words, in any case and with any spacing between its words."""
    words = r"\s+".join(map(re.escape, text.split()))
    pattern = re.compile(rf"(?<!\w){words}(?!\w)", re.IGNORECASE)
    return (lambda line: bool(pattern.search(line))), text


def _either(*options: Says) -> Says:
    return (lambda line: any(says(line) for says, _ in options)), " or ".join(name for _, name in options)


def _tomorrow(now: datetime) -> date:
    return now.date() + timedelta(days=1)


# What a line must say for each checked talk topic, from the clock and the team.
_TALK_FACTS: dict[str, Callable[[datetime, Mapping[str, str]], Says]] = {
    "the time": lambda now, team: _says_time(now),
    "what day is today": lambda now, team: _either(_says_weekday(now.date()), _says_date(now.date())),
    "what day is tomorrow": lambda now, team: _either(_says_weekday(_tomorrow(now)), _says_date(_tomorrow(now))),
    "the day of the week": lambda now, team: _says_weekday(now.date()),
    "the day of the month": lambda now, team: _says_day_number(now.date()),
    "your teams name": lambda now, team: _says_phrase(team["name"]),
    "your teams country": lambda now, team: _says_phrase(team["country"]),
    "your teams affiliation": lambda now, team: _says_phrase(team["affiliation"]),
}


def _talk_fact(topic: str, start: Snapshot) -> Says | None:
    """What a talk line must say, or None for ABOUT_YOURSELF. An unknown topic raises ValueError."""
    if topic == ABOUT_YOURSELF:
        return None
    if topic not in _TALK_FACTS:
        raise ValueError(f"No check for the talk topic {topic!r}.")
    facts = start.get("facts")
    if facts is None:
        raise ValueError(f"The world has no facts to check {topic!r} against.")
    return _TALK_FACTS[topic](datetime.fromisoformat(facts["now"]), facts["team"])


def _check_talk(spec: Mapping[str, Any], start: Snapshot, end: Snapshot) -> tuple[bool, str]:
    room = spec["person"]["room"]
    fact = _talk_fact(spec["topic"], start)
    after = _after_finding(spec, start, end)
    if after is None:
        return False, f"No lookup at {spec['place']} returned {spec['person']}."
    lines = _said(after, room)
    if fact is None:
        return bool(lines), f"Expected a line in {room} after finding the person."
    says, name = fact
    right = next((line for line in lines if says(line)), None)
    if right is None:
        return False, f"Expected a line in {room} that says {name} after finding the person. Said: {lines!r}."
    return True, f"Said {name} in {room}: {right!r}."


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
