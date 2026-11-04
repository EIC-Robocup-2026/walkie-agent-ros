"""Hand-written traces that judge the checkers from outside.

The oracle and the checkers share an author, so the gate's 100% only shows that they
agree. These traces come from the review's checker probe. Each one and its expected
result was written from the command's sentence alone. Expected results stay literal
here, and nothing in this file may import or call the oracle.
"""

from functools import lru_cache

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
