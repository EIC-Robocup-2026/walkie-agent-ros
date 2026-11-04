"""Hand-written traces that judge the checkers from outside.

The oracle and the checkers share an author, so the gate's 100% only shows that they
agree. These traces come from the review's checker probe. Each one and its expected
result was written from the command's sentence alone. Expected results stay literal
here, and nothing in this file may import or call the oracle.
"""

from functools import lru_cache

import pytest

from mahoraga.checkers import check
from mahoraga.commands import commands
from mahoraga.driver import allow_list
from mahoraga.worlds import build_world, world_seed
from walkie_agent.atomics import GPSR
from walkie_agent.capabilities.fake import FakeRobot


@lru_cache
def _records():
    return {record["seed"]: record for record in commands("trial", 300)}


def _run(seed, trace):
    """Run `trace(f, start, spec)` on command `seed`, in its epoch 1 world, and check the end."""
    spec = _records()[seed]["spec"]
    robot = FakeRobot(build_world(spec, world_seed(seed, 1)))
    start = robot.snapshot()
    trace(allow_list(GPSR, robot.ctx), start, spec)
    return check(spec, start, robot.snapshot())


# Correct solutions, written without reading the oracle. They all have to keep passing.

# The first three commands of each kind in the trial split, as the review drew them. Its
# 300 commands hold only one person_info, which makes 22.
SOLVED = {
    "best_object": (8, 25, 36),
    "count_objects": (13, 14, 21),
    "count_people": (81, 103, 163),
    "escort": (4, 6, 17),
    "move_object": (1, 3, 7),
    "person_info": (265,),
    "relay_info": (2, 23, 178),
    "talk": (10, 33, 95),
}
_MEASURE_FIELDS = {
    "biggest": "size_cm",
    "largest": "size_cm",
    "smallest": "size_cm",
    "heaviest": "weight_g",
    "lightest": "weight_g",
    "thinnest": "thickness_cm",
}


def _deliver(f, dest):
    if dest["to"] == "place":
        f["go_to"](dest["place"])
        f["place_on"](dest["place"])
    elif dest["to"] == "operator":
        f["go_to"]("instruction point")
        f["hand_over"]("operator")
    else:
        f["go_to"](dest["room"])
        taker = f["find_people"](dest.get("name", dest.get("description", "")))[0]
        f["hand_over"](taker["id"])


def _solve(f, _start, spec):
    kind = spec["kind"]
    if kind == "move_object":
        f["go_to"](spec["source"])
        f["pick"](f["find_objects"](spec["want"])[0]["id"])
        _deliver(f, spec["dest"])
    elif kind == "count_objects":
        f["go_to"](spec["place"])
        count = len(f["find_objects"](spec["category"]))
        f["go_to"]("instruction point")
        f["say"](str(count))
    elif kind == "best_object":
        f["go_to"](spec["place"])
        measured = [f["object_info"](item["id"]) for item in f["find_objects"](spec["category"] or "")]
        ordered = sorted(measured, key=lambda item: item[_MEASURE_FIELDS[spec["comparison"]]])
        chosen = ordered[-1] if spec["comparison"] in ("biggest", "largest", "heaviest") else ordered[0]
        f["go_to"]("instruction point")
        f["say"](chosen["name"])
    elif kind == "count_people":
        f["go_to"](spec["room"])
        count = len([person for person in f["find_people"](spec["description"]) if person["id"] != "operator"])
        f["go_to"]("instruction point")
        f["say"](str(count))
    elif kind in ("person_info", "relay_info"):
        f["go_to"](spec["room"])
        source = next(person for person in f["find_people"]() if person["id"] != "operator")
        if kind == "relay_info":
            f["go_to"](spec["to_room"])
            f["find_people"]()
        else:
            f["go_to"]("instruction point")
        f["say"](source[spec["info"]])
    else:
        selector = spec["person"]
        f["go_to"](spec["place"])
        found = f["find_people"](selector.get("name", selector.get("description", "")))
        if spec["greet"]:
            f["say"]("Hello " + found[0]["name"])
        if kind == "talk":
            f["say"]("Today is Thursday.")
        elif spec["dest"]:
            f["go_to"](spec["dest"])


def test_the_solutions_cover_every_kind_and_keep_their_number():
    kinds = {seed: kind for kind, seeds in SOLVED.items() for seed in seeds}

    assert len(kinds) == 22
    assert {seed: _records()[seed]["spec"]["kind"] for seed in kinds} == kinds


@pytest.mark.parametrize("seed", [seed for seeds in SOLVED.values() for seed in seeds])
def test_a_correct_solution_passes(seed):
    ok, why = _run(seed, _solve)

    assert ok, (_records()[seed]["text"], why)


def test_doing_nothing_fails_every_sampled_command():
    """Otherwise the oracle's 100% would say nothing about that command."""
    assert len(_records()) == 300
    for seed, record in _records().items():
        ok, why = _run(seed, lambda _f, _start, _spec: None)
        assert not ok, (record["text"], why)


# Delivery controls: wrong object, source, place and recipient, on real generated worlds.


def _first_move_with(wanted):
    """The first move_object command whose world holds an object `wanted(item, at_source)` accepts."""
    for seed, record in _records().items():
        spec = record["spec"]
        if spec["kind"] != "move_object":
            continue
        start = FakeRobot(build_world(spec, world_seed(seed, 1))).snapshot()
        places = start["places"]
        for object_id, item in start["objects"].items():
            source = spec["source"]
            at_source = item["at"] == source or (
                places[source]["room"] == source and places[item["at"]]["room"] == source
            )
            matches = spec["want"] in (item["name"], item["category"])
            if wanted(spec, matches, at_source):
                return seed, object_id, item["at"]
    raise AssertionError("No command fits the control.")


def _fetch(f, object_id, at):
    f["go_to"](at)
    f["find_objects"]()
    f["pick"](object_id)


def test_delivering_an_object_of_the_wrong_kind_fails():
    seed, object_id, at = _first_move_with(lambda _spec, matches, at_source: at_source and not matches)

    def wrong_object(f, _start, spec):
        _fetch(f, object_id, at)
        _deliver(f, spec["dest"])

    ok, why = _run(seed, wrong_object)

    assert not ok, why


def test_delivering_an_object_from_the_wrong_source_fails():
    seed, object_id, at = _first_move_with(lambda _spec, matches, at_source: matches and not at_source)

    def wrong_source(f, _start, spec):
        _fetch(f, object_id, at)
        _deliver(f, spec["dest"])

    ok, why = _run(seed, wrong_source)

    assert not ok, why


def test_delivering_to_the_wrong_place_fails():
    seed, object_id, at = _first_move_with(
        lambda spec, matches, at_source: matches and at_source and spec["dest"]["to"] == "place"
    )

    def wrong_place(f, _start, spec):
        _fetch(f, object_id, at)
        place = next(name for name in ("table", "shelf", "sofa") if name != spec["dest"].get("place"))
        f["go_to"](place)
        f["place_on"](place)

    ok, why = _run(seed, wrong_place)

    assert not ok, why


def test_delivering_to_the_wrong_person_fails():
    seed, object_id, at = _first_move_with(
        lambda spec, matches, at_source: matches and at_source and spec["dest"]["to"] == "person"
    )

    def wrong_person(f, _start, _spec):
        _fetch(f, object_id, at)
        f["go_to"]("instruction point")
        f["hand_over"]("operator")

    ok, why = _run(seed, wrong_person)

    assert not ok, why


# The review's checker sample: each trace as it recorded it, with what it should score
# now. The sample's other wrong verdict, the wrong final correction, is a known limit.

EVERY_NUMBER = "There are 99, not zero one two three four five six seven eight nine ten eleven twelve."
_JANE_IN_LIVING_ROOM = {
    "kind": "talk",
    "person": {"room": "living room", "name": "Jane"},
    "place": "living room",
    "greet": False,
    "topic": "the day of the week",
}
_DISHES_ON_THE_BAR = {"kind": "count_objects", "category": "dish", "place": "bar"}
_SITTING_IN_LIVING_ROOM = {"kind": "count_people", "description": "sitting person", "room": "living room"}
_POSE_AT_THE_BAR = {"kind": "person_info", "info": "pose", "room": "kitchen room"}

CHECKER_SAMPLE = [
    pytest.param(
        13,
        _DISHES_ON_THE_BAR,
        [("go_to", "bar"), ("objects", "does not exist"), ("say", EVERY_NUMBER)],
        False,
        id="wrong-number-and-empty-object-look",
    ),
    pytest.param(
        81,
        _SITTING_IN_LIVING_ROOM,
        [("go_to", "living room"), ("people", "does not exist"), ("say", EVERY_NUMBER)],
        False,
        id="wrong-number-and-empty-people-look",
    ),
    pytest.param(
        81,
        _SITTING_IN_LIVING_ROOM,
        [("go_to", "living room"), ("move_relative", 0.0, -3.0, 0.0), ("people", ""), ("say", "1")],
        False,
        id="look-in-the-wrong-room-after-a-relative-move",
    ),
    pytest.param(
        8,
        {"kind": "best_object", "comparison": "largest", "category": "fruit", "place": "bar"},
        [
            ("go_to", "bar"),
            ("objects", ""),
            ("say", "None of these is the answer: apple, banana, bowl, bowl, chips bag, red plate, red plate"),
        ],
        False,
        id="deny-every-object",
    ),
    pytest.param(
        10,
        _JANE_IN_LIVING_ROOM,
        [
            ("go_to", "living room"),
            ("people", "does not exist"),
            ("say", "Goodbye, operator; I refuse this request."),
        ],
        False,
        id="nobody-found-and-a-refusal",
    ),
    pytest.param(
        10,
        _JANE_IN_LIVING_ROOM,
        [("go_to", "living room"), ("people", "Angel"), ("say", "Angel, it is Wednesday.")],
        False,
        id="speak-only-to-another-named-person",
    ),
    pytest.param(
        6,
        {
            "kind": "escort",
            "person": {"room": "kitchen room", "name": "Simone"},
            "place": "kitchen room",
            "greet": True,
            "dest": "table",
        },
        [("go_to", "table"), ("people", "does not exist"), ("say", "Hello nobody")],
        False,
        id="destination-before-a-person-who-does-not-exist",
    ),
    pytest.param(
        27,
        {
            "kind": "escort",
            "person": {"room": "kitchen room", "name": "Charlie"},
            "place": "shelf",
            "dest": "kitchen room",
            "greet": False,
        },
        [("go_to", "kitchen room"), ("people", "does not exist")],
        False,
        id="named-origin-skipped-and-nobody-found",
    ),
    pytest.param(
        13,
        _DISHES_ON_THE_BAR,
        [
            ("go_to", "bar"),
            ("objects", "dish"),
            ("say", "There are 1 dish items."),
            ("say", "Task 99 is complete."),
        ],
        True,
        id="right-count-and-an-unrelated-number",
    ),
    pytest.param(
        265,
        _POSE_AT_THE_BAR,
        [("go_to", "kitchen room"), ("people", ""), ("say", "They are seated.")],
        True,
        id="right-pose-in-other-words",
    ),
    pytest.param(
        2,
        {"kind": "relay_info", "info": "name", "room": "kitchen room", "to_room": "living room"},
        [
            ("go_to", "living room"),
            ("people", ""),
            ("go_to", "kitchen room"),
            ("people", ""),
            ("go_to", "living room"),
            ("say", "Jane"),
        ],
        True,
        id="recipient-identified-before-the-source",
    ),
]


@pytest.mark.parametrize(("seed", "spec", "actions", "expected"), CHECKER_SAMPLE)
def test_the_review_sample_scores_as_it_should(seed, spec, actions, expected):
    robot = FakeRobot(build_world(spec, world_seed(seed, 1)))
    start = robot.snapshot()
    for verb, *args in actions:
        getattr(robot, verb)(*args)

    ok, why = check(spec, start, robot.snapshot())

    assert ok is expected, why


# Relative moves that change room (M11).

SEATED_IN_LIVING_ROOM = 81  # "tell me how many sitting persons are in the living room": one
STANDING_IN_KITCHEN = 163  # "tell me how many standing persons are in the kitchen room": one


def _look_after_moving_to_the_kitchen(f, _start, _spec):
    f["go_to"]("living room")
    f["move_relative"](0.0, -3.0)
    seen = f["find_people"]()
    assert seen and all(person["room"] == "kitchen room" for person in seen)
    f["say"]("1")


def test_a_look_from_the_wrong_room_after_a_relative_move_does_not_count():
    ok, why = _run(SEATED_IN_LIVING_ROOM, _look_after_moving_to_the_kitchen)

    assert not ok, why


def test_a_look_from_the_right_room_after_a_relative_move_counts():
    ok, why = _run(STANDING_IN_KITCHEN, _look_after_moving_to_the_kitchen)

    assert ok, why


def test_a_look_that_a_relative_move_left_the_room_before_still_counts():
    def looks_then_leaves(f, _start, _spec):
        f["go_to"]("living room")
        f["find_people"]()
        f["move_relative"](0.0, -3.0)
        f["say"]("1")

    ok, why = _run(SEATED_IN_LIVING_ROOM, looks_then_leaves)

    assert ok, why


def test_a_look_from_a_room_the_robot_walked_back_into_counts():
    def out_and_back(f, _start, _spec):
        f["go_to"]("living room")
        f["move_relative"](0.0, -3.0)
        f["move_relative"](0.0, 3.0)
        assert all(person["room"] == "living room" for person in f["find_people"]())
        f["say"]("1")

    ok, why = _run(SEATED_IN_LIVING_ROOM, out_and_back)

    assert ok, why


# The right person, at the right place, in the right order (M4).

MEET_JANE = 10  # "meet Jane in the living room and tell the day of the week"
TALK_TO_POINTING = 33  # "say the day of the week to the person pointing to the right in the kitchen room"
TAKE_CHARLIE = 27  # "take Charlie from the shelf to the kitchen room"
SALUTE_SIMONE = 6  # "salute Simone in the kitchen room and take them to the table"
MORGAN_SOFA_TO_TABLE = 4  # "navigate to the sofa then meet Morgan and escort them to the table"
GREET_ADEL = 17  # "greet Adel in the kitchen room and follow them to the living room"


def _someone_else_in_the_room(start, spec):
    wanted = spec["person"]["name"]
    return next(
        who["name"]
        for key, who in start["people"].items()
        if key != "operator" and who["room"] == spec["person"]["room"] and who["name"] != wanted
    )


def test_talk_fails_when_the_lookup_for_nobody_returned_nothing():
    def refuses(f, _start, _spec):
        f["go_to"]("living room")
        assert f["find_people"]("does not exist") == []
        f["say"]("Goodbye, operator; I refuse this request.")

    ok, why = _run(MEET_JANE, refuses)

    assert not ok, why


def test_talk_fails_when_only_another_named_person_was_found():
    def talks_to_someone_else(f, start, spec):
        other = _someone_else_in_the_room(start, spec)
        f["go_to"]("living room")
        f["find_people"](other)
        f["say"](f"{other}, it is Wednesday.")

    ok, why = _run(MEET_JANE, talks_to_someone_else)

    assert not ok, why


def test_talk_passes_when_the_named_person_was_returned_then_a_line_was_said():
    def talks_to_jane(f, _start, _spec):
        f["go_to"]("living room")
        assert [person["name"] for person in f["find_people"]("Jane")] == ["Jane"]
        f["say"]("Today is Thursday.")

    ok, why = _run(MEET_JANE, talks_to_jane)

    assert ok, why


def test_talk_accepts_a_lookup_of_everybody_that_returned_the_person():
    def looks_at_everybody(f, _start, _spec):
        f["go_to"]("living room")
        assert "Jane" in [person["name"] for person in f["find_people"]()]
        f["say"]("Today is Thursday.")

    ok, why = _run(MEET_JANE, looks_at_everybody)

    assert ok, why


def test_talk_to_a_description_needs_someone_who_fits_it_returned():
    def looks_for_the_wrong_gesture(f, start, spec):
        wrong = next(
            who["gesture"]
            for key, who in start["people"].items()
            if key != "operator" and who["room"] == "kitchen room" and who["gesture"] != spec["person"]["description"]
        )
        f["go_to"]("kitchen room")
        assert f["find_people"](wrong)
        f["say"]("Today is Thursday.")

    def looks_for_the_right_one(f, _start, spec):
        f["go_to"]("kitchen room")
        assert f["find_people"](spec["person"]["description"])
        f["say"]("Today is Thursday.")

    assert not _run(TALK_TO_POINTING, looks_for_the_wrong_gesture)[0]
    assert _run(TALK_TO_POINTING, looks_for_the_right_one)[0]


def test_talk_fails_when_the_line_comes_before_the_person_is_found():
    def says_first(f, _start, _spec):
        f["go_to"]("living room")
        f["say"]("Today is Thursday.")
        f["find_people"]("Jane")

    ok, why = _run(MEET_JANE, says_first)

    assert not ok, why


def test_talk_fails_when_the_line_is_said_after_leaving_the_room():
    def says_elsewhere(f, _start, _spec):
        f["go_to"]("living room")
        f["find_people"]("Jane")
        f["go_to"]("kitchen room")
        f["say"]("Today is Thursday.")

    ok, why = _run(MEET_JANE, says_elsewhere)

    assert not ok, why


def test_talk_keeps_its_known_limit_on_what_was_said():
    """No authoritative answer exists for a talk topic, so any line to the right person counts."""

    def finds_jane_and_refuses(f, _start, _spec):
        f["go_to"]("living room")
        f["find_people"]("Jane")
        f["say"]("Goodbye, operator; I refuse this request.")

    ok, why = _run(MEET_JANE, finds_jane_and_refuses)

    assert ok, why


def test_escort_fails_when_the_destination_came_first_and_the_lookup_found_nobody():
    def skips_the_origin(f, _start, spec):
        f["go_to"](spec["dest"])
        assert f["find_people"]("does not exist") == []
        if spec["greet"]:
            f["say"]("Hello nobody")

    salute_ok, salute_why = _run(SALUTE_SIMONE, skips_the_origin)
    take_ok, take_why = _run(TAKE_CHARLIE, skips_the_origin)

    assert not salute_ok, salute_why
    assert not take_ok, take_why


def test_escort_fails_when_the_named_origin_was_never_visited():
    def looks_from_elsewhere_in_the_room(f, _start, _spec):
        f["go_to"]("kitchen room")
        assert [person["name"] for person in f["find_people"]("Charlie")] == ["Charlie"]
        f["go_to"]("kitchen room")

    ok, why = _run(TAKE_CHARLIE, looks_from_elsewhere_in_the_room)

    assert not ok, why


def test_escort_fails_when_the_person_was_found_before_the_named_origin_was_visited():
    def origin_after_the_lookup(f, _start, _spec):
        f["go_to"]("tv")
        assert [person["name"] for person in f["find_people"]("Morgan")] == ["Morgan"]
        f["go_to"]("sofa")
        f["go_to"]("table")

    ok, why = _run(MORGAN_SOFA_TO_TABLE, origin_after_the_lookup)

    assert not ok, why


def test_escort_fails_when_the_lookup_at_the_origin_returned_someone_else():
    def finds_another_person(f, start, spec):
        other = _someone_else_in_the_room(start, spec)
        f["go_to"]("sofa")
        f["find_people"](other)
        f["go_to"]("table")

    ok, why = _run(MORGAN_SOFA_TO_TABLE, finds_another_person)

    assert not ok, why


def test_escort_fails_when_the_destination_was_reached_only_before_the_person_was_found():
    def destination_first(f, _start, _spec):
        f["go_to"]("table")
        f["go_to"]("sofa")
        f["find_people"]("Morgan")

    def stands_at_the_destination_all_along(f, _start, _spec):
        f["go_to"]("table")
        assert f["find_people"]("Simone")
        f["say"]("Hello Simone")

    assert not _run(MORGAN_SOFA_TO_TABLE, destination_first)[0]
    assert not _run(SALUTE_SIMONE, stands_at_the_destination_all_along)[0]


def test_escort_passes_from_the_origin_to_the_destination():
    def takes_morgan(f, _start, _spec):
        f["go_to"]("sofa")
        f["find_people"]("Morgan")
        f["go_to"]("table")

    ok, why = _run(MORGAN_SOFA_TO_TABLE, takes_morgan)

    assert ok, why


def test_escort_passes_when_the_robot_moves_on_after_reaching_the_destination():
    def returns_to_the_operator(f, _start, _spec):
        f["go_to"]("sofa")
        f["find_people"]("Morgan")
        f["go_to"]("table")
        f["go_to"]("instruction point")

    ok, why = _run(MORGAN_SOFA_TO_TABLE, returns_to_the_operator)

    assert ok, why


def test_escort_to_a_room_that_holds_the_named_origin_needs_no_further_move():
    def stays_in_the_room(f, _start, _spec):
        f["go_to"]("shelf")
        assert [person["name"] for person in f["find_people"]("Charlie")] == ["Charlie"]

    def looks_for_nobody(f, _start, _spec):
        f["go_to"]("shelf")
        assert f["find_people"]("does not exist") == []

    assert _run(TAKE_CHARLIE, stays_in_the_room)[0]
    assert not _run(TAKE_CHARLIE, looks_for_nobody)[0]


def test_escort_needs_the_greeting_before_it_leaves():
    def greets_after_arriving(f, _start, _spec):
        f["go_to"]("kitchen room")
        f["find_people"]("Adel")
        f["go_to"]("living room")
        f["say"]("Hello Adel")

    def greets_first(f, _start, _spec):
        f["go_to"]("kitchen room")
        f["find_people"]("Adel")
        f["say"]("Hello Adel")
        f["go_to"]("living room")

    assert not _run(GREET_ADEL, greets_after_arriving)[0]
    assert _run(GREET_ADEL, greets_first)[0]


def test_escort_fails_when_the_greeting_follows_the_only_move_inside_one_room():
    def greets_at_the_destination(f, _start, _spec):
        f["go_to"]("kitchen room")
        f["find_people"]("Simone")
        f["go_to"]("table")
        f["say"]("Hello Simone")

    ok, why = _run(SALUTE_SIMONE, greets_at_the_destination)

    assert not ok, why


def test_escort_fails_when_the_greeting_is_said_away_from_the_person():
    def greets_from_the_next_room(f, _start, _spec):
        f["go_to"]("kitchen room")
        f["find_people"]("Adel")
        f["go_to"]("instruction point")
        f["say"]("Hello Adel")
        f["go_to"]("living room")

    ok, why = _run(GREET_ADEL, greets_from_the_next_room)

    assert not ok, why


def test_a_relative_move_into_the_destination_room_reaches_it():
    def walks_there(f, _start, _spec):
        f["go_to"]("living room")
        f["move_relative"](0.0, -3.0)

    def walks_elsewhere(f, _start, _spec):
        f["go_to"]("living room")
        f["move_relative"](1.0, 0.0)

    spec = {
        "kind": "escort",
        "person": {"room": "living room", "name": "Axel"},
        "place": "living room",
        "dest": "kitchen room",
        "greet": False,
    }
    for trace, expected in ((walks_there, True), (walks_elsewhere, False)):
        robot = FakeRobot(build_world(spec, 1))
        start = robot.snapshot()
        f = allow_list(GPSR, robot.ctx)
        f["go_to"]("living room")
        assert f["find_people"]("Axel")
        trace(f, start, spec)

        assert check(spec, start, robot.snapshot())[0] is expected


# What counts as the answer (M5).

DISHES_ON_BAR = 13  # "tell me how many dishes there are on the bar": one
LARGEST_FRUIT_ON_BAR = 8  # "tell me what is the largest fruit on the bar": banana
POSE_AT_BAR = 265  # "tell me the pose of the person at the bar": sitting


def test_every_number_at_once_after_an_empty_look_fails():
    def empty_objects_look(f, _start, _spec):
        f["go_to"]("bar")
        f["find_objects"]("does not exist")
        f["say"](EVERY_NUMBER)

    def empty_people_look(f, _start, _spec):
        f["go_to"]("living room")
        f["find_people"]("does not exist")
        f["say"](EVERY_NUMBER)

    assert not _run(DISHES_ON_BAR, empty_objects_look)[0]
    assert not _run(SEATED_IN_LIVING_ROOM, empty_people_look)[0]


def test_every_number_at_once_after_a_real_look_fails():
    def real_look(f, _start, _spec):
        f["go_to"]("bar")
        assert f["find_objects"]("dish")
        f["say"](EVERY_NUMBER)

    ok, why = _run(DISHES_ON_BAR, real_look)

    assert not ok, why


def test_the_right_number_after_an_empty_look_fails():
    def guesses(f, _start, _spec):
        f["go_to"]("bar")
        assert f["find_objects"]("does not exist") == []
        f["say"]("1")

    ok, why = _run(DISHES_ON_BAR, guesses)

    assert not ok, why


def test_the_right_number_before_the_look_fails():
    def answers_first(f, _start, _spec):
        f["say"]("1")
        f["go_to"]("bar")
        f["find_objects"]()

    ok, why = _run(DISHES_ON_BAR, answers_first)

    assert not ok, why


def test_the_right_number_followed_by_an_unrelated_number_passes():
    def adds_a_remark(f, _start, _spec):
        f["go_to"]("bar")
        f["say"](f"There are {len(f['find_objects']('dish'))} dish items.")
        f["say"]("Task 99 is complete.")

    ok, why = _run(DISHES_ON_BAR, adds_a_remark)

    assert ok, why


def test_listing_every_object_while_denying_them_fails():
    def denies_all(f, _start, _spec):
        f["go_to"]("bar")
        names = [item["name"] for item in f["find_objects"]()]
        assert "banana" in names
        f["say"]("None of these is the answer: " + ", ".join(names))

    ok, why = _run(LARGEST_FRUIT_ON_BAR, denies_all)

    assert not ok, why


def test_the_largest_fruit_said_alone_passes_and_a_smaller_one_fails():
    def says(line):
        def trace(f, _start, _spec):
            f["go_to"]("bar")
            f["find_objects"]("fruit")
            f["say"](line)

        return trace

    assert _run(LARGEST_FRUIT_ON_BAR, says("The banana."))[0]
    assert not _run(LARGEST_FRUIT_ON_BAR, says("The apple."))[0]


def test_a_seated_person_is_sitting():
    def paraphrases(f, _start, _spec):
        f["go_to"]("kitchen room")
        f["find_people"]()
        f["say"]("They are seated.")

    ok, why = _run(POSE_AT_BAR, paraphrases)

    assert ok, why


def test_a_correction_to_a_value_that_is_no_candidate_is_a_known_limit():
    """The rule settles this on the first answer, which is right, so the wrong correction passes."""

    def corrects_itself(f, _start, _spec):
        f["go_to"]("kitchen room")
        f["find_people"]()
        f["say"]("The answer is sitting person")
        f["say"]("Correction: the answer is definitely ZZZ, not sitting person.")

    ok, why = _run(POSE_AT_BAR, corrects_itself)

    assert ok, why


def test_a_denial_of_the_one_right_candidate_is_a_known_limit():
    """Negation is not parsed, so a line that names only the right value counts as asserting it."""

    def denies_it(f, _start, _spec):
        f["go_to"]("bar")
        f["find_objects"]("fruit")
        f["say"]("It is not the banana.")

    ok, why = _run(LARGEST_FRUIT_ON_BAR, denies_it)

    assert ok, why


def test_a_true_line_with_a_second_number_is_a_known_limit():
    """The rule cannot tell a remark from a second answer inside one line, so this fails."""

    def adds_a_number_in_the_line(f, _start, _spec):
        f["go_to"]("bar")
        f["say"](f"I took 3 seconds and found {len(f['find_objects']('dish'))}.")

    ok, why = _run(DISHES_ON_BAR, adds_a_number_in_the_line)

    assert not ok, why


# Relaying whatever the order (M12).

NAME_TO_TV = 2  # "tell the name of the person at the table to the person at the tv": Jane


def _source_name(f, _start, _spec):
    f["go_to"]("kitchen room")
    return next(person for person in f["find_people"]() if person["id"] != "operator")["name"]


def test_a_relay_that_identifies_the_recipient_first_passes():
    def recipient_first(f, start, spec):
        f["go_to"]("living room")
        f["find_people"]()
        name = _source_name(f, start, spec)
        f["go_to"]("living room")
        f["say"](name)

    ok, why = _run(NAME_TO_TV, recipient_first)

    assert ok, why


def test_a_relay_that_looks_at_the_recipient_again_passes():
    def source_first(f, start, spec):
        name = _source_name(f, start, spec)
        f["go_to"]("living room")
        f["find_people"]()
        f["say"](name)

    ok, why = _run(NAME_TO_TV, source_first)

    assert ok, why


def test_a_relay_said_before_the_source_was_looked_at_fails():
    def says_too_soon(f, start, spec):
        f["go_to"]("living room")
        f["find_people"]()
        f["say"]("Jane")
        _source_name(f, start, spec)
        f["go_to"]("living room")

    ok, why = _run(NAME_TO_TV, says_too_soon)

    assert not ok, why


def test_a_relay_said_before_the_recipient_was_looked_at_fails():
    def looks_too_late(f, start, spec):
        name = _source_name(f, start, spec)
        f["go_to"]("living room")
        f["say"](name)
        f["find_people"]()

    ok, why = _run(NAME_TO_TV, looks_too_late)

    assert not ok, why


def test_a_relay_said_with_no_recipient_ever_identified_fails():
    def never_looks_at_them(f, start, spec):
        name = _source_name(f, start, spec)
        f["go_to"]("living room")
        f["say"](name)

    ok, why = _run(NAME_TO_TV, never_looks_at_them)

    assert not ok, why


def test_a_relay_that_walks_into_the_recipients_room_counts_as_there():
    def walks_in(f, start, spec):
        f["go_to"]("living room")
        f["find_people"]()
        name = _source_name(f, start, spec)
        f["go_to"]("living room")
        f["move_relative"](1.5, 0.0)
        f["say"](name)

    ok, why = _run(NAME_TO_TV, walks_in)

    assert ok, why


def test_a_relay_said_after_walking_out_of_the_recipients_room_fails():
    def walks_out(f, start, spec):
        f["go_to"]("living room")
        f["find_people"]()
        name = _source_name(f, start, spec)
        f["go_to"]("living room")
        f["move_relative"](0.0, -3.0)
        f["say"](name)

    ok, why = _run(NAME_TO_TV, walks_out)

    assert not ok, why
