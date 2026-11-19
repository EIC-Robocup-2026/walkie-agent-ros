"""The words find_objects uses for a detector's labels, against FakeRobot's."""

import pytest

from walkie_agent.capabilities.fake import OBJECTS
from walkie_agent.capabilities.fake import PLURALS as FAKE_PLURALS
from walkie_agent.perception.object_words import (
    CATEGORIES,
    OTHER,
    PLURALS,
    category_of,
    looks_for,
    matches,
    words_for,
)


def test_the_arenas_objects_have_fakerobots_categories_and_plurals():
    for name, item in OBJECTS.items():
        assert category_of(name) == item["category"], name
    for category, plural in FAKE_PLURALS.items():
        assert PLURALS[category] == plural


def test_every_category_has_a_plural():
    assert set(CATEGORIES.values()) == set(PLURALS)


def test_the_table_is_in_lower_case_and_read_ignoring_case_and_spaces():
    assert all(label == label.strip().lower() for label in CATEGORIES)
    assert category_of(" Sofa ") == "furniture"


def test_a_label_the_table_lacks_is_other_and_matches_only_its_name():
    assert category_of("coffee table") == OTHER
    assert words_for("coffee table") == {"coffee table"}
    assert not matches(OTHER, "coffee table")


@pytest.mark.parametrize(
    ("wanted", "label", "expected"),
    [
        ("", "anything", True),
        ("mug", "mug", True),
        ("dish", "mug", True),
        ("dishes", "mug", True),
        ("furniture", "sofa", True),
        ("sofas", "sofa", False),
        ("cup", "mug", False),
    ],
)
def test_a_query_matches_the_name_the_category_or_its_plural(wanted, label, expected):
    assert matches(wanted, label) is expected


@pytest.mark.parametrize(
    ("wanted", "label", "expected"),
    [
        ("water bottle", "bottle", True),
        ("water bottle", " Bottle ", True),
        ("red plate", "plate", True),
        ("coffee table", "table", False),
        ("waterbottle", "bottle", False),
        ("bottle", "water bottle", False),
        ("bottle opener", "bottle", False),
        ("water bottle", "", False),
    ],
)
def test_a_query_that_names_a_kind_of_the_label_matches_it(wanted, label, expected):
    assert matches(wanted, label) is expected


def test_a_detector_that_looks_for_bottles_could_find_a_water_bottle():
    assert looks_for("water bottle", ("bottle", "cup"))


def test_looks_for_asks_whether_any_class_could_match():
    seats = ("chair", "sofa", "armchair", "stool")
    assert looks_for("furniture", seats)
    assert looks_for("stool", seats)
    assert not looks_for("apple", seats)
    assert not looks_for("fruits", seats)
    assert looks_for("", seats)
    assert looks_for("apple", ())
