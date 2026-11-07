"""The recording wrapper keeps what each command asks for, and the split seeds hold still."""

import collections
import json
import random

import pytest

from walkie_agent.capabilities.fake import (
    CHULA_PAT14_6_PLACES,
    GESTURES,
    NAMES,
    OBJECTS,
    PLURALS,
    POSES,
    ROOMS,
    SURFACES,
)
from walkie_agent.mahoraga.commands import (
    CATEGORIES,
    SPLITS,
    TRAIN_COMMANDS,
    UNSUPPORTED,
    commands,
    generator,
    parse,
)


@pytest.fixture(scope="module")
def made():
    return generator()


def test_the_data_files_name_what_the_fake_arena_has(made):
    knowledge = made.knowledge

    assert set(knowledge.names) == set(NAMES)
    assert set(knowledge.rooms) == set(ROOMS)
    assert set(knowledge.placement_locations) == set(SURFACES)
    assert set(knowledge.locations) <= set(CHULA_PAT14_6_PLACES)
    assert set(knowledge.objects) == set(OBJECTS)
    assert set(knowledge.object_categories_singular) == set(PLURALS)
    assert set(knowledge.object_categories_plural) == set(PLURALS.values())


def test_every_recorded_value_appears_in_the_sentence(made):
    for seed in range(1500):
        record = made.generate(seed, "")
        for _, key, value in record["slots"]:
            assert value in record["text"], (key, record)


def test_no_placeholder_is_left_in_a_sentence(made):
    for seed in range(1500):
        assert "{" not in made.generate(seed, "")["text"]


def test_a_followup_and_the_main_template_do_not_overwrite_each_others_slots(made):
    both = 0
    for seed in range(3000):
        record = made.generate(seed, "objects")
        if record["followups"] and record["followups"][-1] == "deliverObjToPrsInRoom":
            parts = {part for part, key, _ in record["slots"] if key == "room"}
            both += "main" in parts and "deliverObjToPrsInRoom" in parts

    assert both > 0


def test_the_same_seed_gives_the_same_command_and_leaves_global_random_alone(made):
    random.seed(99)
    expected = random.random()

    random.seed(99)
    first = made.generate(7, "")
    assert random.random() == expected

    assert first == made.generate(7, "")


@pytest.mark.parametrize("category", CATEGORIES)
def test_every_template_and_chain_parses_or_is_listed_as_unsupported(made, category):
    seen = collections.Counter()
    for seed in range(4000):
        record = made.generate(seed, CATEGORIES[category])
        spec = parse(record)
        if record["template"] in UNSUPPORTED:
            assert spec is None
        else:
            assert spec is not None, record
            json.dumps(spec)
            seen[spec["kind"]] += 1

    kinds = {
        "objects": {"move_object", "count_objects", "best_object"},
        "people": {"count_people", "person_info", "relay_info", "talk", "escort"},
    }
    expected = kinds[category] if category in kinds else kinds["objects"] | kinds["people"]
    assert set(seen) == expected


def _recorded(template, followups, *slots):
    return {"template": template, "followups": list(followups), "slots": [list(slot) for slot in slots]}


@pytest.mark.parametrize(
    ("record", "spec"),
    [
        (
            _recorded(
                "takeObjFromPlcmt",
                ["putObjInTrash"],
                ("main", "obj_singCat", "fruit"),
                ("main", "plcmtLoc", "shelf"),
            ),
            {"kind": "move_object", "want": "fruit", "source": "shelf", "dest": {"to": "place", "place": "trash"}},
        ),
        (
            _recorded("bringMeObjFromPlcmt", [], ("main", "obj", "red plate"), ("main", "plcmtLoc", "bar")),
            {"kind": "move_object", "want": "red plate", "source": "bar", "dest": {"to": "operator"}},
        ),
        (
            _recorded(
                "goToLoc",
                ["findObj", "takeObj", "deliverObjToNameAtBeac"],
                ("main", "loc_room", "kitchen room"),
                ("findObj", "obj_singCat", "mug"),
                ("deliverObjToNameAtBeac", "name", "Jane"),
                ("deliverObjToNameAtBeac", "room", "living room"),
            ),
            {
                "kind": "move_object",
                "want": "mug",
                "source": "kitchen room",
                "dest": {"to": "person", "room": "living room", "name": "Jane"},
            },
        ),
        (
            _recorded("countObjOnPlcmt", [], ("main", "plurCat", "dishes"), ("main", "plcmtLoc", "table")),
            {"kind": "count_objects", "category": "dish", "place": "table"},
        ),
        (
            _recorded(
                "tellCatPropOnPlcmt",
                [],
                ("main", "objComp", "thinnest"),
                ("main", "singCat", "snack"),
                ("main", "plcmtLoc", "shelf"),
            ),
            {"kind": "best_object", "comparison": "thinnest", "category": "snack", "place": "shelf"},
        ),
        (
            _recorded(
                "countPrsInRoom",
                [],
                ("main", "gestPersPlur_posePersPlur", "persons raising their left arm"),
                ("main", "room", "living room"),
            ),
            {"kind": "count_people", "description": "person raising their left arm", "room": "living room"},
        ),
        (
            _recorded(
                "tellPrsInfoInLoc",
                [],
                ("main", "persInfo", "pose"),
                ("main", "inRoom_atLoc", "at the sofa"),
            ),
            {"kind": "person_info", "info": "pose", "room": "living room"},
        ),
        (
            _recorded(
                "tellPrsInfoAtLocToPrsAtLoc",
                [],
                ("main", "persInfo", "name"),
                ("main", "loc", "tv"),
                ("dup", "loc2", "bar"),
            ),
            {"kind": "relay_info", "info": "name", "room": "living room", "to_room": "kitchen room"},
        ),
        (
            _recorded(
                "greetNameInRm",
                ["followPrsToRoom"],
                ("main", "name", "Paris"),
                ("main", "room", "kitchen room"),
                ("dup", "room2", "living room"),
            ),
            {
                "kind": "escort",
                "person": {"room": "kitchen room", "name": "Paris"},
                "place": "kitchen room",
                "greet": True,
                "dest": "living room",
            },
        ),
        (
            _recorded(
                "goToLoc",
                ["findPrs", "talkInfo"],
                ("main", "loc_room", "sofa"),
                ("findPrs", "gestPers_posePers", "waving person"),
                ("talkInfo", "talk", "the time"),
            ),
            {
                "kind": "talk",
                "person": {"room": "living room", "description": "waving person"},
                "place": "sofa",
                "greet": False,
                "topic": "the time",
            },
        ),
        (
            _recorded(
                "followPrsAtLoc",
                [],
                ("main", "gestPers_posePers", "lying person"),
                ("main", "inRoom_atLoc", "in the kitchen room"),
            ),
            {
                "kind": "escort",
                "person": {"room": "kitchen room", "description": "lying person"},
                "place": "kitchen room",
                "dest": None,
                "greet": False,
            },
        ),
    ],
)
def test_parse_reads_the_slots_of_a_record(record, spec):
    assert parse(record) == spec


def test_descriptions_are_ones_the_fake_people_can_have(made):
    described = {*POSES, *GESTURES}
    for seed in range(3000):
        spec = parse(made.generate(seed, "people"))
        if spec and spec["kind"] in ("talk", "escort") and "description" in spec["person"]:
            assert spec["person"]["description"] in described
        if spec and spec["kind"] == "count_people":
            assert spec["description"] in described


def test_commands_skip_what_is_unsupported_and_a_larger_n_extends_a_smaller_one():
    small = commands("trial", 40)
    large = commands("trial", 80)

    assert large[:40] == small
    assert not {record["template"] for record in large} & UNSUPPORTED
    assert len({record["seed"] for record in large}) == 80


def test_the_splits_share_no_command():
    trial, train, heldout = (commands(split, 60) for split in ("trial", "train", "heldout"))

    assert {r["seed"] for r in trial}.isdisjoint(r["seed"] for r in train)
    assert {r["seed"] for r in train}.isdisjoint(r["seed"] for r in heldout)
    assert min(r["seed"] for r in heldout) >= SPLITS["heldout"]


@pytest.mark.parametrize("category", CATEGORIES)
def test_the_held_out_split_has_no_sentence_of_the_standard_training_set(category):
    standard = {record["text"] for record in commands("train", TRAIN_COMMANDS)}
    trained = standard | {record["text"] for record in commands("train", TRAIN_COMMANDS, category)}

    assert trained.isdisjoint(record["text"] for record in commands("heldout", 160, category))


def test_the_held_out_split_repeats_and_a_larger_n_extends_a_smaller_one():
    small = commands("heldout", 40)

    assert commands("heldout", 40) == small
    assert commands("heldout", 80)[:40] == small


def test_the_mix_keeps_both_groups():
    kinds = {record["spec"]["kind"] for record in commands("trial", 150)}

    assert {"move_object", "escort", "talk"} <= kinds
