"""Each checker passes the right episode and fails a wrong one. Doing nothing never passes."""

import pytest

from walkie_agent.capabilities.fake import GESTURES, OBJECTS, POSES, TEAM, FakeRobot
from walkie_agent.errors import StepError
from walkie_agent.mahoraga.checkers import ABOUT_YOURSELF, MEASURES, check
from walkie_agent.mahoraga.commands import generator
from walkie_agent.mahoraga.worlds import OPERATOR, build_world


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


ONE_FRUIT = [("apple_1", "apple", "shelf"), ("mug_1", "mug", "shelf")]


@pytest.mark.parametrize(
    ("fruits", "line", "passes"),
    [
        (ONE_FRUIT, "There is 1 fruit.", True),
        (ONE_FRUIT, "There are 1.0 fruits.", True),
        (TWO_FRUITS, "There are 2.0 fruits.", True),
        (ONE_FRUIT, "There are 1.1 fruits.", False),
        (ONE_FRUIT, "There are -1 fruits.", False),
        (ONE_FRUIT, "There are 21 fruits.", False),
        (ONE_FRUIT, "There are twenty-one fruits.", False),
        (TWO_FRUITS, "There are thirty-two fruits.", False),
        (ONE_FRUIT, "There are thirteen fruits.", False),
        (ONE_FRUIT, "There are thirty fruits.", False),
        (ONE_FRUIT, "There are a hundred fruits.", False),
        (ONE_FRUIT, "There is a single fruit.", True),
        (TWO_FRUITS, "There are a couple of fruits.", True),
        (TWO_FRUITS, "There is a pair of fruits.", True),
        (TWO_FRUITS, "There is a single fruit.", False),
        (ONE_FRUIT, "There are a couple of fruits.", False),
        (ONE_FRUIT, "There is a pair of fruits.", False),
    ],
)
def test_count_objects_reads_each_number_as_a_whole_token(fruits, line, passes):
    ok, _ = episode(COUNT, world(fruits), ("go_to", "shelf"), ("objects", "fruit"), ("say", line))

    assert ok is passes


@pytest.mark.parametrize(
    "wrong",
    [
        "There are twenty-one fruits.",
        "There are 1-2 fruits.",
        "There are thirty fruits.",
        "There are nineteen fruits.",
        "There are a hundred fruits.",
        "There are -1 fruits.",
        "There are 1.1 fruits.",
    ],
)
def test_a_number_no_count_equals_is_still_the_answer(wrong):
    """It is a candidate that fails, so the right count on a later line cannot rescue it."""
    look = (("go_to", "shelf"), ("objects", "fruit"))

    assert not episode(COUNT, world(ONE_FRUIT), *look, ("say", wrong), ("say", "1"))[0]


@pytest.mark.parametrize(
    "remark",
    [
        "I checked the 3rd shelf.",
        "I checked the 1st shelf.",
        "I saw apple_1 here.",
        "I saw a two-handled mug.",
    ],
)
def test_a_word_that_only_holds_a_number_is_no_answer(remark):
    """"3rd" is not 3, so the count on the next line decides."""
    look = (("go_to", "shelf"), ("objects", "fruit"))

    assert episode(COUNT, world(TWO_FRUITS), *look, ("say", remark), ("say", "2"))[0]


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


def test_a_later_look_with_no_answer_after_it_leaves_the_earlier_answer_standing():
    look = (("go_to", "shelf"), ("objects", "fruit"))

    assert episode(COUNT, world(TWO_FRUITS), *look, ("say", "2"), *look)[0]
    assert not episode(COUNT, world(TWO_FRUITS), *look, ("say", "5"), *look)[0]
    assert episode(COUNT, world(TWO_FRUITS), *look, ("say", "5"), *look, ("say", "2"), *look)[0]
    assert not episode(COUNT, world(TWO_FRUITS), *look, ("say", "2"), *look, ("say", "5"), *look)[0]
    assert episode(COUNT, world(TWO_FRUITS), *look, ("say", "2"), *look, ("say", "Task done."))[0]
    assert not episode(COUNT, world(TWO_FRUITS), ("say", "2"), *look)[0]


def test_a_later_look_at_something_else_needs_no_second_answer():
    look = (("go_to", "shelf"), ("objects", "fruit"))

    assert episode(COUNT, world(TWO_FRUITS), *look, ("say", "2"), ("objects", "mug"))[0]
    assert episode(COUNT, world(TWO_FRUITS), *look, ("objects", "mug"), ("say", "2"))[0]


def test_count_objects_needs_a_look_that_returned_all_it_counts():
    go = ("go_to", "shelf")

    assert not episode(COUNT, world(TWO_FRUITS), go, ("objects", "nothing"), ("say", "2"))[0]
    assert not episode(COUNT, world(TWO_FRUITS), go, ("objects", "apple"), ("say", "2"))[0]
    assert episode(COUNT, world(TWO_FRUITS), go, ("objects", "apple"), ("objects", "banana"), ("say", "2"))[0]


def test_count_objects_of_nothing_needs_a_look_from_that_place():
    """With nothing to return, where the look happened is all that shows it was the right one."""
    orange_elsewhere = [("orange_1", "orange", "bar")]
    answer = ("say", "0")

    assert not episode(COUNT, world(orange_elsewhere), ("go_to", "bar"), ("objects", ""), answer)[0]
    assert episode(COUNT, world(orange_elsewhere), ("go_to", "shelf"), ("objects", ""), answer)[0]
    assert episode(COUNT, world(orange_elsewhere), ("go_to", "kitchen room"), ("objects", ""), answer)[0]


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


def test_count_people_needs_a_look_that_returned_all_it_counts():
    built = world(
        people=_people(
            a=("kitchen room", "Jane", {}),
            b=("kitchen room", "Axel", {}),
        )
    )
    go = ("go_to", "kitchen room")

    assert not episode(COUNT_PEOPLE, built, go, ("people", "nobody"), ("say", "2"))[0]
    assert not episode(COUNT_PEOPLE, built, go, ("people", "Jane"), ("say", "2"))[0]
    assert episode(COUNT_PEOPLE, built, go, ("people", "waving person"), ("say", "2"))[0]


def test_count_people_of_nobody_needs_a_look_from_that_room():
    """With nobody to return, where the look happened is all that shows it was the right one."""
    spec = {"kind": "count_people", "description": "person pointing to the left", "room": "living room"}
    answer = ("say", "0")

    assert not episode(spec, world(), ("go_to", "kitchen room"), ("people", ""), answer)[0]
    assert episode(spec, world(), ("go_to", "living room"), ("people", ""), answer)[0]


def test_count_people_of_nobody_judges_the_room_a_relative_move_reached():
    """No lookup result can give this away, so only the record's room can."""
    spec = {"kind": "count_people", "description": "person pointing to the left", "room": "living room"}
    walks_out = (("go_to", "living room"), ("move_relative", 0.0, -3.0), ("people", ""))

    assert not episode(spec, world(), *walks_out, ("say", "0"))[0]


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
        ("sitting person", "They are sat down.", True),
        ("sitting person", "They are sitting up.", True),
        ("standing person", "The person is on their feet.", True),
        ("lying person", "They are stretched out on the floor.", True),
        ("lying person", "The person is prone.", True),
        ("lying person", "The person is supine.", True),
        ("standing person", "They are sat down.", False),
        ("sitting person", "The person is on their feet.", False),
        ("standing person", "The person is prone.", False),
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
        ("person raising their right arm", "Their right arm is up.", True),
        ("person raising their left arm", "They have their left hand up.", True),
        ("person raising their left arm", "They are holding their left arm aloft.", True),
        ("person raising their right arm", "They hold their right hand high.", True),
        ("person pointing to the left", "They are indicating something to their left.", True),
        ("person pointing to the right", "She indicates the right.", True),
        ("person pointing to the left", "They indicate left.", True),
        ("person pointing to the right", "He is gesturing to the right.", True),
        ("person pointing to the left", "He gestures left.", True),
        ("waving person", "They are on their feet, waving.", True),
        ("person raising their right arm", "The person is raising their right leg.", False),
        ("person raising their right arm", "Their right forearm is raised.", True),
        ("person raising their right arm", "Their hand rests on their raised right leg.", False),
        ("person raising their left arm", "Their left leg is up.", False),
        ("person raising their right arm", "Raising to the right.", False),
        ("person raising their right arm", "Their left arm is up.", False),
        ("person pointing to the left", "Their left arm is up.", False),
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


def test_relay_passes_when_the_recipient_was_identified_before_the_source():
    steps = (
        ("go_to", "living room"),
        ("people", ""),
        ("go_to", "kitchen room"),
        ("people", ""),
        ("go_to", "living room"),
        ("say", "Jane"),
    )

    assert episode(RELAY, world(), *steps)[0]


def test_relay_fails_when_it_speaks_before_learning_from_the_source():
    steps = (
        ("go_to", "living room"),
        ("people", ""),
        ("say", "Jane"),
        ("go_to", "kitchen room"),
        ("people", ""),
        ("go_to", "living room"),
    )

    assert not episode(RELAY, world(), *steps)[0]


def test_relay_fails_when_it_speaks_away_from_the_recipient():
    steps = (("go_to", "living room"), ("people", ""), ("go_to", "kitchen room"), ("people", ""), ("say", "Jane"))

    assert not episode(RELAY, world(), *steps)[0]


def test_relay_fails_when_a_lookup_returned_no_recipient():
    built = world()
    look_there = (("go_to", "kitchen room"), ("people", ""))

    assert not episode(RELAY, built, *look_there, ("go_to", "living room"), ("people", "Nobody"), ("say", "Jane"))[0]
    assert not episode(RELAY, built, *look_there, ("go_to", "living room"), ("people", OPERATOR), ("say", "Jane"))[0]


def test_relay_fails_when_a_lookup_returned_no_source():
    look_there = (("go_to", "kitchen room"), ("people", "Nobody"))

    assert not episode(RELAY, world(), *look_there, ("go_to", "living room"), ("people", ""), ("say", "Jane"))[0]


def test_relay_judges_the_first_answering_line_said_with_the_recipient():
    look_there = (("go_to", "kitchen room"), ("people", ""))
    look_here = (("go_to", "living room"), ("people", ""))

    assert episode(RELAY, world(), *look_there, ("say", "Axel"), *look_here, ("say", "Jane"))[0]
    assert not episode(RELAY, world(), *look_there, *look_here, ("say", "Axel"), ("say", "Jane"))[0]
    assert not episode(RELAY, world(), *look_there, *look_here, ("say", "Jane or Axel"))[0]


def test_relay_within_one_room_needs_only_one_look():
    spec = {**RELAY, "to_room": "kitchen room"}

    assert episode(spec, world(), ("go_to", "kitchen room"), ("people", ""), ("say", "Jane"))[0]


# "Something about yourself" takes any line, so this judges only the look and the room.
TALK = {
    "kind": "talk",
    "person": {"room": "kitchen room", "name": "Jane"},
    "place": "kitchen room",
    "topic": ABOUT_YOURSELF,
    "greet": False,
}


def test_talk_needs_a_look_then_a_line_in_the_room():
    look = (("go_to", "kitchen room"), ("people", ""))

    assert episode(TALK, world(), *look, ("say", "It is noon."))[0]
    assert not episode(TALK, world(), *look)[0]
    assert not episode(TALK, world(), ("go_to", "kitchen room"), ("say", "It is noon."), ("people", ""))[0]
    assert not episode(TALK, world(), *look, ("go_to", "living room"), ("say", "It is noon."))[0]


# Thursday, 5 November 2026, at 14:05 in Bangkok, unless a row says otherwise.
THURSDAY = "2026-11-05T14:05:00+07:00"
TALK_LOOK = (("go_to", "kitchen room"), ("people", "Jane"))
CHECKED_TOPICS = [
    "the time",
    "what day is today",
    "what day is tomorrow",
    "the day of the week",
    "the day of the month",
    "your teams name",
    "your teams country",
    "your teams affiliation",
]
RIGHT_LINES = {
    "the time": "It is 14:05.",
    "what day is today": "Today is Thursday.",
    "what day is tomorrow": "Tomorrow is Friday.",
    "the day of the week": "It is Thursday.",
    "the day of the month": "It is the 5th.",
    "your teams name": "We are team EIC.",
    "your teams country": "We come from Thailand.",
    "your teams affiliation": "We are from Chulalongkorn University.",
}


def at_clock(now=THURSDAY):
    built = world()
    built["facts"] = {"now": now, "team": dict(TEAM)}
    return built


# The two facts verbs that get_facts calls, as robot actions.
READ_CLOCK, READ_TEAM = ("now",), ("team",)
READ = (READ_CLOCK, READ_TEAM)


def talk(topic, *lines, now=THURSDAY, read=READ):
    """Whether lines said in Jane's room, after a look that found her and `read`, pass a talk on `topic`."""
    spec = {**TALK, "topic": topic}
    return episode(spec, at_clock(now), *TALK_LOOK, *read, *(("say", line) for line in lines))[0]


@pytest.mark.parametrize("topic", CHECKED_TOPICS)
def test_talk_passes_with_the_right_fact_and_fails_without_it(topic):
    assert talk(topic, RIGHT_LINES[topic])
    assert not talk(topic, "What day is tomorrow?")
    assert not talk(topic, "Hello.")
    assert not talk(topic)


@pytest.mark.parametrize("topic", CHECKED_TOPICS)
def test_talk_takes_any_line_in_the_room_that_says_the_fact_not_only_the_first(topic):
    assert talk(topic, "Hello, Jane.", RIGHT_LINES[topic], "Goodbye.")


@pytest.mark.parametrize("topic", [*CHECKED_TOPICS, ABOUT_YOURSELF])
def test_talk_fails_when_the_right_line_is_said_in_another_room(topic):
    line = RIGHT_LINES.get(topic, "I am Walkie.")
    leave = ("go_to", "living room")

    assert not episode({**TALK, "topic": topic}, at_clock(), *TALK_LOOK, *READ, leave, ("say", line))[0]


@pytest.mark.parametrize(("topic", "line"), [("your teams name", "We are EIC."), ("the time", "It is 14:05.")])
def test_talk_fails_when_a_right_fact_is_said_without_reading_it(topic, line):
    assert talk(topic, line)
    assert not talk(topic, line, read=())


@pytest.mark.parametrize("topic", CHECKED_TOPICS)
def test_talk_takes_only_a_line_said_after_a_facts_read(topic):
    spec, line = {**TALK, "topic": topic}, ("say", RIGHT_LINES[topic])

    assert not episode(spec, at_clock(), *TALK_LOOK, line, *READ)[0]
    # The read may come before the lookup, and a line after both counts.
    assert episode(spec, at_clock(), *READ, *TALK_LOOK, line)[0]
    assert episode(spec, at_clock(), *TALK_LOOK, line, *READ, line)[0]


@pytest.mark.parametrize("topic", CHECKED_TOPICS)
def test_talk_needs_the_read_of_the_fact_its_topic_asks_for(topic):
    needed, other = (READ_TEAM, READ_CLOCK) if topic.startswith("your teams") else (READ_CLOCK, READ_TEAM)

    assert talk(topic, RIGHT_LINES[topic], read=[needed])
    assert not talk(topic, RIGHT_LINES[topic], read=[other])


def test_a_refused_facts_read_does_not_count():
    robot = FakeRobot(at_clock())
    start = robot.snapshot()
    robot.go_to("kitchen room")
    robot.people("Jane")
    robot.faults["facts.now"] = StepError("SKILL_FAILED", "No clock.")
    with pytest.raises(StepError):
        robot.now()
    robot.say("It is 14:05.")

    assert not check({**TALK, "topic": "the time"}, start, robot.snapshot())[0]


def test_talk_says_why_a_right_line_without_a_facts_read_fails():
    ok, why = episode({**TALK, "topic": "the time"}, at_clock(), *TALK_LOOK, ("say", "It is 14:05."))

    assert not ok
    assert "'It is 14:05.'" in why and "no successful facts.now read came before it" in why


def test_talk_about_yourself_needs_no_facts_read():
    assert episode(TALK, at_clock(), *TALK_LOOK, ("say", "I am Walkie."))[0]


@pytest.mark.parametrize(
    ("topic", "line", "now", "passes"),
    [
        # The time: 24-hour, or 12-hour with an optional and then right am or pm.
        ("the time", "The time is 14:05.", THURSDAY, True),
        ("the time", "It is 2:05 pm.", THURSDAY, True),
        ("the time", "It is 2:05 P.M.", THURSDAY, True),
        ("the time", "It is 2:05PM now.", THURSDAY, True),
        ("the time", "It is 2:05.", THURSDAY, True),
        ("the time", "It is 2:05 am.", THURSDAY, False),
        ("the time", "It is 02:05.", THURSDAY, False),
        ("the time", "It is 14:06.", THURSDAY, False),
        ("the time", "It is 4:05 pm.", THURSDAY, False),
        ("the time", "It is five past two.", THURSDAY, False),
        ("the time", "It is half past two.", "2026-11-05T14:30:00+07:00", False),
        ("the time", "It is 2:30 in the afternoon.", "2026-11-05T14:30:00+07:00", True),
        ("the time", "It is quarter past two.", "2026-11-05T14:15:00+07:00", False),
        ("the time", "It is 2 o'clock.", "2026-11-05T14:00:00+07:00", True),
        ("the time", "It is 2 o’clock in the afternoon.", "2026-11-05T14:00:00+07:00", True),
        ("the time", "It is 2 o'clock.", THURSDAY, False),
        ("the time", "It is 9:05 a.m.", "2026-11-05T09:05:00+07:00", True),
        ("the time", "It is 09:05.", "2026-11-05T09:05:00+07:00", True),
        ("the time", "It is 9:05 pm.", "2026-11-05T09:05:00+07:00", False),
        ("the time", "It is 12:30 am.", "2026-11-05T00:30:00+07:00", True),
        ("the time", "It is 00:30.", "2026-11-05T00:30:00+07:00", True),
        ("the time", "It is 12:30 pm.", "2026-11-05T00:30:00+07:00", False),
        # A part of the day after the time is a qualifier too, and must be right.
        ("the time", "It is 2:05 in the afternoon.", THURSDAY, True),
        ("the time", "It is 2:05 in the morning.", THURSDAY, False),
        ("the time", "It is 7:05 in the evening.", "2026-11-05T19:05:00+07:00", True),
        ("the time", "It is 7:05 in the evening.", "2026-11-05T07:05:00+07:00", False),
        ("the time", "It is 9:05 in the morning.", "2026-11-05T09:05:00+07:00", True),
        # Noon and midnight need 12 on the hour, at that hour.
        ("the time", "It is 12:00 noon.", "2026-11-05T12:00:00+07:00", True),
        ("the time", "It is 12:00 noon.", "2026-11-05T00:00:00+07:00", False),
        ("the time", "It is 12:00 midnight.", "2026-11-05T00:00:00+07:00", True),
        ("the time", "It is 12:00 midnight.", "2026-11-05T12:00:00+07:00", False),
        ("the time", "It is 12:30 noon.", "2026-11-05T12:30:00+07:00", False),
        ("the time", "It is 12 noon.", "2026-11-05T12:00:00+07:00", True),
        ("the time", "It is 12 midnight.", "2026-11-05T00:00:00+07:00", True),
        # A digit hour from 1 to 12 right before a qualifier is that hour on the hour.
        ("the time", "It is 12 am.", "2026-11-05T00:00:00+07:00", True),
        ("the time", "It is 12 pm.", "2026-11-05T12:00:00+07:00", True),
        ("the time", "It is 2 p.m.", "2026-11-05T14:00:00+07:00", True),
        ("the time", "It is 2PM.", "2026-11-05T14:00:00+07:00", True),
        ("the time", "It is 2 in the afternoon.", "2026-11-05T14:00:00+07:00", True),
        ("the time", "It is 12 am.", "2026-11-05T12:00:00+07:00", False),
        ("the time", "It is 12 pm.", "2026-11-05T00:00:00+07:00", False),
        ("the time", "It is 2 am.", "2026-11-05T14:00:00+07:00", False),
        ("the time", "It is 2 pm.", THURSDAY, False),
        ("the time", "It is 2 noon.", "2026-11-05T12:00:00+07:00", False),
        ("the time", "It is 2.", "2026-11-05T14:00:00+07:00", False),
        ("the time", "It is 2.05 pm.", "2026-11-05T17:00:00+07:00", False),
        # "h o'clock" reads its hour like the other forms, leading zero included.
        ("the time", "It is 2 o'clock in the afternoon.", "2026-11-05T14:00:00+07:00", True),
        ("the time", "It is 2 o'clock in the morning.", "2026-11-05T14:00:00+07:00", False),
        ("the time", "It is 2 o'clock pm.", "2026-11-05T14:00:00+07:00", True),
        ("the time", "It is 02 o'clock.", "2026-11-05T14:00:00+07:00", False),
        ("the time", "It is 02 o'clock.", "2026-11-05T02:00:00+07:00", True),
        ("the time", "It is 14 o'clock.", "2026-11-05T14:00:00+07:00", True),
        ("the time", "It is 12 o'clock midnight.", "2026-11-05T00:00:00+07:00", True),
        ("the time", "It is 12 o'clock noon.", "2026-11-05T00:00:00+07:00", False),
        # Today: the weekday, or the date in either order.
        ("what day is today", "Today is thursday.", THURSDAY, True),
        ("what day is today", "Today is 5 November.", THURSDAY, True),
        ("what day is today", "It is November 5th, 2026.", THURSDAY, True),
        ("what day is today", "It is the 5th of November.", THURSDAY, True),
        ("what day is today", "Today is Friday.", THURSDAY, False),
        ("what day is today", "Today is 15 November.", THURSDAY, False),
        ("what day is today", "Today is the 5th.", THURSDAY, False),
        # A line about another day than the one asked for does not answer it.
        ("what day is today", "Tomorrow is Thursday.", THURSDAY, False),
        ("what day is today", "TOMORROW is 5 November.", THURSDAY, False),
        ("what day is today", "Yesterday was Thursday.", THURSDAY, False),
        ("what day is today", "Yesterday was Wednesday, so today is Thursday.", THURSDAY, True),
        # Tomorrow: tomorrow's weekday or date, never today's.
        ("what day is tomorrow", "Tomorrow is November 6.", THURSDAY, True),
        ("what day is tomorrow", "Tomorrow is Thursday.", THURSDAY, False),
        ("what day is tomorrow", "Tomorrow is 5 November.", THURSDAY, False),
        ("what day is tomorrow", "Tomorrow is 1 January.", "2026-12-31T23:59:00+07:00", True),
        ("what day is tomorrow", "Tomorrow is Friday.", "2026-12-31T23:59:00+07:00", True),
        ("what day is tomorrow", "It will be Friday.", THURSDAY, True),
        ("what day is tomorrow", "Today is Friday.", THURSDAY, False),
        ("what day is tomorrow", "Today is 6 November.", THURSDAY, False),
        # The day of the week: the name as a whole word.
        ("the day of the week", "It is THURSDAY today.", THURSDAY, True),
        ("the day of the week", "It is Wednesday.", THURSDAY, False),
        ("the day of the week", "It is Friday.", THURSDAY, False),
        ("the day of the week", "Tomorrow is Thursday.", THURSDAY, False),
        # The day of the month: digits with an optional suffix, or a word from 1 to 31.
        ("the day of the month", "Tomorrow is the 5th.", THURSDAY, False),
        ("the day of the month", "Today is day 5.", THURSDAY, True),
        ("the day of the month", "It is the fifth.", THURSDAY, True),
        ("the day of the month", "It is day five.", THURSDAY, True),
        ("the day of the month", "It is the 15th.", THURSDAY, False),
        ("the day of the month", "It is 14:05.", THURSDAY, False),
        ("the day of the month", "It is the twenty-first.", "2026-11-21T10:00:00+07:00", True),
        ("the day of the month", "It is the twenty first.", "2026-11-21T10:00:00+07:00", True),
        ("the day of the month", "It is the first.", "2026-11-21T10:00:00+07:00", False),
        ("the day of the month", "It is the 31st.", "2026-12-31T10:00:00+07:00", True),
        # The team: whole-word phrases, in any case.
        ("your teams name", "We are team eic.", THURSDAY, True),
        ("your teams name", "We are the EICs.", THURSDAY, False),
        ("your teams country", "We come from THAILAND.", THURSDAY, True),
        ("your teams country", "We come from Japan.", THURSDAY, False),
        ("your teams affiliation", "We are from chulalongkorn  university.", THURSDAY, True),
        ("your teams affiliation", "We are from Chulalongkorn.", THURSDAY, False),
    ],
)
def test_talk_accepts_the_forms_of_its_fact(topic, line, now, passes):
    assert talk(topic, line, now=now) is passes


def test_talk_keeps_its_known_limits():
    # A line that names both today and tomorrow passes for either, in either order.
    for both in ("Today is Thursday and tomorrow is Friday.", "Tomorrow is Friday and today is Thursday."):
        assert talk("what day is today", both)
        assert talk("what day is tomorrow", both)
    # Times in words, seconds and "at night" are not read.
    assert not talk("the time", "It is two o'clock.", now="2026-11-05T14:00:00+07:00")
    assert not talk("the time", "It is 14:05:30.")
    assert not talk("the time", "It is 10 at night.", now="2026-11-05T22:00:00+07:00")
    # A qualifier goes with a 12-hour time, so a 24-hour time with one fails although it is right.
    assert not talk("the time", "It is 14:05 in the afternoon.")
    assert not talk("the time", "It is 00:00 midnight.", now="2026-11-05T00:00:00+07:00")
    # Any number in the line counts for the day of the month, even a pronoun.
    assert talk("the day of the month", "It is the one you wanted.", now="2026-11-01T10:00:00+07:00")
    # The year of a date is not read.
    assert talk("what day is today", "Today is 5 November 1999.")


def test_talk_judges_the_clock_of_the_start_snapshot():
    assert talk("the day of the week", "It is Saturday.", now="2026-11-07T10:00:00+07:00")
    assert not talk("the day of the week", "It is Thursday.", now="2026-11-07T10:00:00+07:00")


def test_an_unknown_talk_topic_or_a_world_without_facts_raises():
    with pytest.raises(ValueError, match="No check for the talk topic 'the weather'"):
        talk("the weather", "It is sunny.")
    no_facts = world()
    del no_facts["facts"]
    with pytest.raises(ValueError, match="no facts"):
        episode({**TALK, "topic": "the time"}, no_facts, *TALK_LOOK, ("say", "It is 14:05."))
    assert episode(TALK, no_facts, *TALK_LOOK, ("say", "I am Walkie."))[0]


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
