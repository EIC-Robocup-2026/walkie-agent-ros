"""Each checker passes the right episode and fails a wrong one. Doing nothing never passes."""

import pytest

from mahoraga.checkers import MEASURES, check
from mahoraga.commands import commands, generator
from mahoraga.worlds import OPERATOR, build_world, world_seed
from walkie_agent.capabilities.fake import GESTURES, OBJECTS, POSES, FakeRobot
from walkie_agent.errors import StepError


def _people(**rooms):
    defaults = {"pose": "sitting person", "gesture": "waving person"}
    people = {
        OPERATOR: {"name": OPERATOR, "room": "living room", "pose": "standing at the instruction point", "gesture": "no gesture"},
    }
    for person, (room, name, overrides) in rooms.items():
        people[person] = {"name": name, "room": room, **defaults, **overrides}
    return people


def world(objects=(), people=None):
    built = build_world({"kind": "count_objects", "category": "fruit", "place": "shelf"}, 1)
    built["objects"] = {}
    for object_id, name, at in objects:
        built["objects"][object_id] = {"name": name, **OBJECTS[name], "at": at}
    built["people"] = people or _people(
        ada=("kitchen room", "Jane", {}),
        bob=("living room", "Axel", {"pose": "lying person", "gesture": "person raising their left arm"}),
    )
    return built


def episode(spec, built, *actions):
    """Run `actions`, each (verb, *args), on a robot in `built`, and check the result."""
    robot = FakeRobot(built)
    start = robot.snapshot()
    for name, *args in actions:
        getattr(robot, name)(*args)
    return check(spec, start, robot.snapshot())


MOVE = {"kind": "move_object", "want": "fruit", "source": "shelf", "dest": {"to": "place", "place": "table"}}
APPLE_ON_SHELF = [("apple_1", "apple", "shelf")]


def test_move_passes_when_the_object_reaches_the_place():
    ok, why = episode(
        MOVE,
        world(APPLE_ON_SHELF),
        ("go_to", "shelf"),
        ("pick", "apple_1"),
        ("go_to", "table"),
        ("place_on", "table"),
    )

    assert ok, why


def test_move_fails_when_the_object_is_left_held():
    assert not episode(MOVE, world(APPLE_ON_SHELF), ("go_to", "shelf"), ("pick", "apple_1"))[0]


def test_move_fails_for_an_object_of_the_wrong_kind():
    ok, _ = episode(
        MOVE,
        world([("mug_1", "mug", "shelf")]),
        ("go_to", "shelf"),
        ("pick", "mug_1"),
        ("go_to", "table"),
        ("place_on", "table"),
    )

    assert not ok


def test_move_fails_for_an_object_from_the_wrong_place():
    ok, _ = episode(
        MOVE,
        world([("apple_1", "apple", "bar")]),
        ("go_to", "bar"),
        ("pick", "apple_1"),
        ("go_to", "table"),
        ("place_on", "table"),
    )

    assert not ok


def test_move_fails_for_an_object_already_there_and_never_picked():
    spec = {**MOVE, "source": "table"}

    assert not episode(spec, world([("apple_1", "apple", "table")]))[0]


def test_move_from_a_room_passes_for_any_surface_in_it_and_fails_outside():
    spec = {**MOVE, "source": "kitchen room"}
    steps = (("go_to", "kitchen room"), ("pick", "apple_1"), ("go_to", "table"), ("place_on", "table"))

    assert episode(spec, world([("apple_1", "apple", "bar")]), *steps)[0]
    outside = [("go_to", "living room"), ("pick", "apple_1"), ("go_to", "table"), ("place_on", "table")]
    assert not episode(spec, world([("apple_1", "apple", "sofa")]), *outside)[0]


def test_move_to_trash_needs_the_trash():
    spec = {**MOVE, "dest": {"to": "place", "place": "trash"}}
    pick = (("go_to", "shelf"), ("pick", "apple_1"))

    assert episode(spec, world(APPLE_ON_SHELF), *pick, ("go_to", "trash"), ("place_on", "trash"))[0]
    assert not episode(spec, world(APPLE_ON_SHELF), *pick, ("go_to", "bar"), ("place_on", "bar"))[0]


def test_move_to_the_operator_passes_only_for_the_operator():
    spec = {**MOVE, "dest": {"to": "operator"}}
    pick = (("go_to", "shelf"), ("pick", "apple_1"))

    assert episode(spec, world(APPLE_ON_SHELF), *pick, ("go_to", "instruction point"), ("hand_over", OPERATOR))[0]
    assert not episode(spec, world(APPLE_ON_SHELF), *pick, ("go_to", "living room"), ("hand_over", "bob"))[0]


@pytest.mark.parametrize(
    ("dest", "taker", "passes"),
    [
        ({"to": "person", "room": "living room", "description": "lying person"}, "bob", True),
        ({"to": "person", "room": "living room", "name": "Axel"}, "bob", True),
        ({"to": "person", "room": "living room", "name": "Jane"}, "bob", False),
        ({"to": "person", "room": "living room", "description": "waving person"}, "bob", False),
        ({"to": "person", "room": "living room", "name": OPERATOR}, OPERATOR, False),
    ],
)
def test_move_to_a_person_needs_the_person_the_command_named(dest, taker, passes):
    ok, _ = episode(
        {**MOVE, "dest": dest},
        world(APPLE_ON_SHELF),
        ("go_to", "shelf"),
        ("pick", "apple_1"),
        ("go_to", "living room"),
        ("hand_over", taker),
    )

    assert ok is passes


def test_a_refused_pick_does_not_count():
    robot = FakeRobot(world(APPLE_ON_SHELF))
    start = robot.snapshot()
    robot.faults["arms.pick"] = StepError("SKILL_FAILED", "Dropped.")
    with pytest.raises(StepError):
        robot.pick("apple_1")

    assert not check(MOVE, start, robot.snapshot())[0]


COUNT = {"kind": "count_objects", "category": "fruit", "place": "shelf"}
TWO_FRUITS = [("apple_1", "apple", "shelf"), ("banana_1", "banana", "shelf"), ("mug_1", "mug", "shelf"), ("orange_1", "orange", "bar")]


@pytest.mark.parametrize(
    ("line", "passes"),
    [
        ("There are 2 fruits.", True),
        ("There are two.", True),
        ("Two", True),
        ("There are 3 fruits.", False),
        ("I do not know.", False),
    ],
)
def test_count_objects_reads_the_number(line, passes):
    ok, _ = episode(COUNT, world(TWO_FRUITS), ("go_to", "shelf"), ("objects", "fruit"), ("say", line))

    assert ok is passes


def test_count_objects_needs_a_look_before_the_answer():
    assert not episode(COUNT, world(TWO_FRUITS), ("say", "There are 2."))[0]
    assert not episode(COUNT, world(TWO_FRUITS), ("go_to", "bar"), ("objects", "fruit"), ("say", "There are 2."))[0]


def test_count_objects_takes_the_first_number_said_after_the_look():
    look = (("go_to", "shelf"), ("objects", ""))

    assert not episode(COUNT, world(TWO_FRUITS), *look, ("say", "I see 5."), ("say", "No, 2."))[0]
    assert episode(COUNT, world(TWO_FRUITS), *look, ("say", "I see 2."), ("say", "Task 99 is complete."))[0]


def test_count_objects_judges_the_line_after_the_last_look_that_showed_them():
    look = (("go_to", "shelf"), ("objects", "fruit"))

    assert episode(COUNT, world(TWO_FRUITS), *look, ("say", "5"), *look, ("say", "2"))[0]
    assert not episode(COUNT, world(TWO_FRUITS), *look, ("say", "2"), *look, ("say", "5"))[0]
    assert not episode(COUNT, world(TWO_FRUITS), *look, ("say", "2"), *look)[0]


def test_a_later_look_at_something_else_needs_no_second_answer():
    look = (("go_to", "shelf"), ("objects", "fruit"))

    assert episode(COUNT, world(TWO_FRUITS), *look, ("say", "2"), ("objects", "mug"))[0]
    assert episode(COUNT, world(TWO_FRUITS), *look, ("objects", "mug"), ("say", "2"))[0]


def test_count_objects_needs_a_look_that_returned_all_it_counts():
    go = ("go_to", "shelf")

    assert not episode(COUNT, world(TWO_FRUITS), go, ("objects", "nothing"), ("say", "2"))[0]
    assert not episode(COUNT, world(TWO_FRUITS), go, ("objects", "apple"), ("say", "2"))[0]
    assert episode(COUNT, world(TWO_FRUITS), go, ("objects", "apple"), ("objects", "banana"), ("say", "2"))[0]


def test_count_objects_refuses_a_line_that_asserts_more_than_one_number():
    look = (("go_to", "shelf"), ("objects", ""))

    assert not episode(COUNT, world(TWO_FRUITS), *look, ("say", "There are 2, or maybe 3."))[0]
    assert not episode(COUNT, world(TWO_FRUITS), *look, ("say", "Either one or two."))[0]


def test_count_objects_accepts_a_look_from_the_room():
    assert episode(COUNT, world(TWO_FRUITS), ("go_to", "kitchen room"), ("objects", ""), ("say", "2"))[0]


def _best(comparison, category=None):
    return {"kind": "best_object", "comparison": comparison, "category": category, "place": "shelf"}


@pytest.mark.parametrize(
    ("comparison", "category", "line", "passes"),
    [
        ("biggest", None, "The banana.", True),
        ("largest", "fruit", "It is the banana", True),
        ("smallest", None, "The apple.", True),
        ("heaviest", None, "The mug", True),
        ("lightest", None, "The banana", True),
        ("thinnest", None, "The banana", True),
        ("biggest", None, "The apple.", False),
        ("heaviest", "fruit", "The mug", False),
        ("heaviest", "fruit", "The apple", True),
        ("biggest", None, "No idea.", False),
    ],
)
def test_best_object_names_the_right_kind(comparison, category, line, passes):
    ok, _ = episode(_best(comparison, category), world(TWO_FRUITS), ("go_to", "shelf"), ("objects", ""), ("say", line))

    assert ok is passes


def test_best_object_reads_two_word_names_with_or_without_the_underscore():
    built = world([("red_plate_1", "red plate", "shelf"), ("apple_1", "apple", "shelf")])
    steps = (("go_to", "shelf"), ("objects", ""))

    assert episode(_best("biggest"), built, *steps, ("say", "The red_plate."))[0]
    assert episode(_best("biggest"), built, *steps, ("say", "The Red Plate."))[0]


def test_best_object_refuses_a_line_that_names_more_than_one_object_there():
    look = (("go_to", "shelf"), ("objects", ""))

    assert not episode(_best("biggest"), world(TWO_FRUITS), *look, ("say", "The banana or the apple."))[0]
    assert not episode(_best("biggest", "fruit"), world(TWO_FRUITS), *look, ("say", "The banana, not the mug."))[0]


def test_best_object_ignores_a_name_that_is_not_at_that_place():
    look = (("go_to", "shelf"), ("objects", ""))

    assert episode(_best("biggest"), world(TWO_FRUITS), *look, ("say", "The orange is on the bar."), ("say", "The banana."))[0]


def test_best_object_reads_whole_words_only():
    look = (("go_to", "shelf"), ("objects", ""))

    assert not episode(_best("smallest"), world(TWO_FRUITS), *look, ("say", "A pineapple."))[0]
    assert episode(_best("smallest"), world(TWO_FRUITS), *look, ("say", "The apples."))[0]


def test_best_object_needs_a_look_that_returned_all_it_compares():
    go = ("go_to", "shelf")

    assert not episode(_best("biggest", "fruit"), world(TWO_FRUITS), go, ("objects", "banana"), ("say", "The banana."))[0]
    assert episode(_best("biggest", "fruit"), world(TWO_FRUITS), go, ("objects", "fruit"), ("say", "The banana."))[0]


def test_every_comparison_the_generator_can_draw_has_a_measure():
    assert set(generator().object_comp_list) == set(MEASURES)


COUNT_PEOPLE = {"kind": "count_people", "description": "waving person", "room": "kitchen room"}


def test_count_people_counts_only_that_room_and_that_description():
    built = world(
        people=_people(
            a=("kitchen room", "Jane", {}),
            b=("kitchen room", "Axel", {"gesture": "person pointing to the left"}),
            c=("living room", "Paris", {}),
        )
    )
    look = (("go_to", "kitchen room"), ("people", "waving person"))

    assert episode(COUNT_PEOPLE, built, *look, ("say", "There is one."))[0]
    assert not episode(COUNT_PEOPLE, built, *look, ("say", "There are 2."))[0]
    assert not episode(COUNT_PEOPLE, built, ("say", "There is one."))[0]


def test_count_people_leaves_the_operator_out():
    spec = {"kind": "count_people", "description": "no gesture", "room": "living room"}

    assert episode(spec, world(), ("people", ""), ("say", "zero"))[0]


@pytest.mark.parametrize(
    ("info", "line", "passes"),
    [
        ("name", "The name is Jane.", True),
        ("name", "The name is Axel.", False),
        ("pose", "They are sitting.", True),
        ("pose", "They are lying.", False),
        ("gesture", "They are waving.", True),
        ("gesture", "They are pointing.", False),
    ],
)
def test_person_info_needs_the_value_said_after_a_look(info, line, passes):
    spec = {"kind": "person_info", "info": info, "room": "kitchen room"}

    ok, _ = episode(spec, world(), ("go_to", "kitchen room"), ("people", ""), ("say", line))

    assert ok is passes


def test_person_info_fails_without_a_look_in_that_room():
    spec = {"kind": "person_info", "info": "name", "room": "kitchen room"}

    assert not episode(spec, world(), ("people", ""), ("say", "Jane"))[0]


def test_person_info_fails_after_a_look_that_did_not_return_the_person():
    spec = {"kind": "person_info", "info": "name", "room": "kitchen room"}

    assert not episode(spec, world(), ("go_to", "kitchen room"), ("people", "Nobody"), ("say", "Jane"))[0]


@pytest.mark.parametrize("pose", POSES)
def test_person_info_accepts_each_pose_as_the_fake_spells_it(pose):
    built = world(people=_people(a=("kitchen room", "Jane", {"pose": pose})))
    spec = {"kind": "person_info", "info": "pose", "room": "kitchen room"}

    assert episode(spec, built, ("go_to", "kitchen room"), ("people", ""), ("say", pose))[0]


@pytest.mark.parametrize("gesture", GESTURES)
def test_person_info_accepts_each_gesture_as_the_fake_spells_it(gesture):
    built = world(people=_people(a=("kitchen room", "Jane", {"gesture": gesture})))
    spec = {"kind": "person_info", "info": "gesture", "room": "kitchen room"}

    assert episode(spec, built, ("go_to", "kitchen room"), ("people", ""), ("say", gesture))[0]


@pytest.mark.parametrize(
    ("pose", "line", "passes"),
    [
        ("sitting person", "They are seated.", True),
        ("sitting person", "She sits on the sofa.", True),
        ("standing person", "They are upright on their feet.", True),
        ("standing person", "He stands by the table.", True),
        ("lying person", "They are reclined horizontally.", True),
        ("lying person", "They lie on the floor.", True),
        ("sitting person", "They are upright.", False),
        ("standing person", "They are seated.", False),
        ("lying person", "They are standing.", False),
        ("sitting person", "Sitting, standing or lying.", False),
        ("sitting person", "Either seated or upright.", False),
    ],
)
def test_person_info_accepts_an_ordinary_way_to_say_the_pose(pose, line, passes):
    built = world(people=_people(a=("kitchen room", "Jane", {"pose": pose})))
    spec = {"kind": "person_info", "info": "pose", "room": "kitchen room"}

    ok, _ = episode(spec, built, ("go_to", "kitchen room"), ("people", ""), ("say", line))

    assert ok is passes


@pytest.mark.parametrize(
    ("gesture", "line", "passes"),
    [
        ("person raising their left arm", "Their left arm is raised.", True),
        ("person raising their right arm", "She lifts her right arm.", True),
        ("person pointing to the left", "He points left.", True),
        ("person pointing to the right", "Pointing right.", True),
        ("waving person", "They wave.", True),
        ("person raising their left arm", "Their left arm points at me.", False),
        ("person pointing to the left", "Raising the left arm.", False),
        ("waving person", "Either waving or pointing left.", False),
    ],
)
def test_person_info_accepts_an_ordinary_way_to_say_the_gesture(gesture, line, passes):
    built = world(people=_people(a=("kitchen room", "Jane", {"gesture": gesture})))
    spec = {"kind": "person_info", "info": "gesture", "room": "kitchen room"}

    ok, _ = episode(spec, built, ("go_to", "kitchen room"), ("people", ""), ("say", line))

    assert ok is passes


def test_person_info_takes_the_first_line_that_names_a_known_value():
    spec = {"kind": "person_info", "info": "name", "room": "kitchen room"}
    look = (("go_to", "kitchen room"), ("people", ""))

    assert episode(spec, world(), *look, ("say", "I found someone."), ("say", "It is Jane."))[0]
    assert not episode(spec, world(), *look, ("say", "It is Axel."), ("say", "No, it is Jane."))[0]
    assert not episode(spec, world(), *look, ("say", "Jane or Robin."))[0]
    assert episode(spec, world(), *look, ("say", "Janet's friend Jane."))[0]


def test_person_info_for_a_name_reads_a_name_the_arena_data_does_not_hold():
    built = world(people=_people(a=("kitchen room", "Zed", {})))
    spec = {"kind": "person_info", "info": "name", "room": "kitchen room"}
    look = (("go_to", "kitchen room"), ("people", ""))

    assert episode(spec, built, *look, ("say", "It is Zed."))[0]
    assert not episode(spec, built, *look, ("say", "It is Jane."))[0]


def test_person_info_for_a_two_word_gesture_needs_both_words():
    built = world(people=_people(a=("kitchen room", "Jane", {"gesture": "person raising their left arm"})))
    spec = {"kind": "person_info", "info": "gesture", "room": "kitchen room"}
    look = (("go_to", "kitchen room"), ("people", ""))

    assert episode(spec, built, *look, ("say", "Raising the left arm."))[0]
    assert not episode(spec, built, *look, ("say", "Raising the right arm."))[0]


RELAY = {"kind": "relay_info", "info": "name", "room": "kitchen room", "to_room": "living room"}


def test_relay_needs_a_look_in_each_room_then_the_value_in_the_second():
    built = world()
    look_there = (("go_to", "kitchen room"), ("people", ""))

    assert episode(RELAY, built, *look_there, ("go_to", "living room"), ("people", ""), ("say", "Jane"))[0]
    assert not episode(RELAY, built, *look_there, ("say", "Jane"), ("go_to", "living room"), ("people", ""))[0]
    assert not episode(RELAY, built, *look_there, ("go_to", "living room"), ("say", "Jane"))[0]
    assert not episode(RELAY, built, *look_there, ("go_to", "living room"), ("people", ""), ("say", "Axel"))[0]


def test_relay_within_one_room_needs_only_one_look():
    spec = {**RELAY, "to_room": "kitchen room"}

    assert episode(spec, world(), ("go_to", "kitchen room"), ("people", ""), ("say", "Jane"))[0]


TALK = {"kind": "talk", "person": {"room": "kitchen room", "name": "Jane"}, "place": "kitchen room", "topic": "the time", "greet": False}


def test_talk_needs_a_look_then_a_line_in_the_room():
    look = (("go_to", "kitchen room"), ("people", ""))

    assert episode(TALK, world(), *look, ("say", "It is noon."))[0]
    assert not episode(TALK, world(), *look)[0]
    assert not episode(TALK, world(), ("go_to", "kitchen room"), ("say", "It is noon."), ("people", ""))[0]
    assert not episode(TALK, world(), *look, ("go_to", "living room"), ("say", "It is noon."))[0]


ESCORT = {"kind": "escort", "person": {"room": "kitchen room", "name": "Jane"}, "place": "bar", "dest": "sofa", "greet": False}


def test_escort_needs_the_look_then_the_destination():
    look = (("go_to", "bar"), ("people", "Jane"))

    assert episode(ESCORT, world(), *look, ("go_to", "sofa"))[0]
    assert not episode(ESCORT, world(), *look)[0]
    assert not episode(ESCORT, world(), ("go_to", "sofa"))[0]
    assert not episode(ESCORT, world(), *look, ("go_to", "tv"))[0]
    assert not episode(ESCORT, world(), ("go_to", "bar"), ("people", "Nobody"), ("go_to", "sofa"))[0]
    assert not episode(ESCORT, world(), ("go_to", "shelf"), ("people", "Jane"), ("go_to", "sofa"))[0]


def test_escort_from_the_place_it_goes_to_needs_no_second_move():
    spec = {**ESCORT, "dest": "bar"}

    assert episode(spec, world(), ("go_to", "bar"), ("people", "Jane"))[0]
    assert not episode(spec, world(), ("go_to", "bar"), ("people", "Nobody"))[0]


def test_escort_to_a_room_accepts_any_place_in_it():
    spec = {**ESCORT, "dest": "living room"}

    assert episode(spec, world(), ("go_to", "bar"), ("people", ""), ("go_to", "sofa"))[0]


def test_following_with_no_destination_needs_only_the_look():
    spec = {**ESCORT, "dest": None}

    assert episode(spec, world(), ("go_to", "bar"), ("people", ""))[0]
    assert not episode(spec, world(), ("people", ""))[0]


def test_a_greeting_needs_a_line():
    spec = {**ESCORT, "greet": True}
    look = (("go_to", "bar"), ("people", ""))

    assert episode(spec, world(), *look, ("say", "Hello."), ("go_to", "sofa"))[0]
    assert not episode(spec, world(), *look, ("go_to", "sofa"))[0]


def test_doing_nothing_passes_no_generated_command():
    """Otherwise the oracle's 100% would say nothing about that command."""
    for record in commands("trial", 300):
        robot = FakeRobot(build_world(record["spec"], world_seed(record["seed"], 1)))
        start = robot.snapshot()
        ok, _ = check(record["spec"], start, robot.snapshot())
        assert not ok, record["spec"]
