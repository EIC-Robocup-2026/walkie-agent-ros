"""The oracle: a script for each spec, written from the slots alone with no LLM.

It runs through the same driver and the same type check as an agent's script, and
reads nothing of the world beyond what the atomics return. If it fails, a checker
or the world builder is wrong, which is why it gates every LLM arm.
"""

import json
from collections.abc import Mapping
from typing import Any

from walkie_agent.mahoraga.checkers import ABOUT_YOURSELF, MEASURES
from walkie_agent.mahoraga.worlds import OPERATOR

_INSTRUCTION_POINT = "instruction point"
# Per talk topic, how the oracle's line starts and the get_facts key it ends with.
_TALK_LINES = {
    "the time": ("The time is", "time"),
    "what day is today": ("Today is", "today"),
    "what day is tomorrow": ("Tomorrow is", "tomorrow"),
    "the day of the week": ("Today is", "day_of_week"),
    "the day of the month": ("Today is day", "day_of_month"),
    "your teams name": ("My team is", "team_name"),
    "your teams country": ("My team is from", "team_country"),
    "your teams affiliation": ("My team is from", "team_affiliation"),
}


def oracle_script(spec: Mapping[str, Any]) -> str:
    """A script that carries out `spec`."""
    return _SCRIPTS[spec["kind"]](spec)


def _q(text: str) -> str:
    return json.dumps(text)


def _selector(person: Mapping[str, str]) -> str:
    return _q(person.get("name") or person["description"])


def _moved(spec: Mapping[str, Any]) -> str:
    dest = spec["dest"]
    lines = [
        f"go_to({_q(spec['source'])})",
        f"item = find_objects({_q(spec['want'])})[0]",
        'pick(item["id"])',
    ]
    if dest["to"] == "operator":
        lines += [f"go_to({_q(_INSTRUCTION_POINT)})", f"hand_over({_q(OPERATOR)})"]
    elif dest["to"] == "place":
        lines += [f"go_to({_q(dest['place'])})", f"place_on({_q(dest['place'])})"]
    else:
        lines += [
            f"go_to({_q(dest['room'])})",
            f"person = find_people({_selector(dest)})[0]",
            'hand_over(person["id"])',
        ]
    return "\n".join(lines)


def _count_objects(spec: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"go_to({_q(spec['place'])})",
            f"found = find_objects({_q(spec['category'])})",
            'say(f"There are {len(found)} here.")',
        ]
    )


def _best(spec: Mapping[str, Any]) -> str:
    field, pick = MEASURES[spec["comparison"]]
    better = ">" if pick is max else "<"
    return "\n".join(
        [
            f"go_to({_q(spec['place'])})",
            'best_name = ""',
            "best_value = 0.0",
            f"for item in find_objects({_q(spec['category'] or '')}):",
            '    info = object_info(item["id"])',
            f"    if not best_name or info[{_q(field)}] {better} best_value:",
            '        best_name = info["name"]',
            f"        best_value = info[{_q(field)}]",
            'say(f"It is the {best_name}.")',
        ]
    )


def _count_people(spec: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"go_to({_q(spec['room'])})",
            f"found = find_people({_q(spec['description'])})",
            'say(f"There are {len(found)} here.")',
        ]
    )


def _others_in(room: str) -> list[str]:
    return [
        f"go_to({_q(room)})",
        f"others = [p for p in find_people() if p[\"id\"] != {_q(OPERATOR)}]",
    ]


def _person_info(spec: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            *_others_in(spec["room"]),
            f"say(f\"The {spec['info']} is {{others[0][{_q(spec['info'])}]}}.\")",
        ]
    )


def _relay(spec: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            *_others_in(spec["room"]),
            f"known = others[0][{_q(spec['info'])}]",
            f"go_to({_q(spec['to_room'])})",
            "find_people()",
            f"say(f\"The {spec['info']} is {{known}}.\")",
        ]
    )


def _approach(spec: Mapping[str, Any]) -> list[str]:
    return [
        f"go_to({_q(spec['place'])})",
        f"found = find_people({_selector(spec['person'])})",
        *(['say("Hello.")'] if spec["greet"] else []),
    ]


def _talk(spec: Mapping[str, Any]) -> str:
    """Find the person, then say the fact the topic asks for, as get_facts returns it."""
    topic = spec["topic"]
    if topic == ABOUT_YOURSELF:
        return "\n".join([*_approach(spec), 'say("I am Walkie, a home service robot.")'])
    if topic not in _TALK_LINES:
        raise ValueError(f"No oracle line for the talk topic {topic!r}.")
    lead, key = _TALK_LINES[topic]
    return "\n".join(
        [*_approach(spec), "facts = get_facts()", f'say(f"{lead} {{facts[{_q(key)}]}}.")']
    )


def _escort(spec: Mapping[str, Any]) -> str:
    return "\n".join(
        [*_approach(spec), *([f"go_to({_q(spec['dest'])})"] if spec["dest"] else [])]
    )


_SCRIPTS = {
    "move_object": _moved,
    "count_objects": _count_objects,
    "best_object": _best,
    "count_people": _count_people,
    "person_info": _person_info,
    "relay_info": _relay,
    "talk": _talk,
    "escort": _escort,
}
