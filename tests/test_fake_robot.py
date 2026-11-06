"""FakeRobot: every verb and guard over a small world, and the seeded arena."""

import hashlib
import json
import math
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from walkie_agent.capabilities import Arms, Base, Facts, Speech, Team, Vision
from walkie_agent.capabilities.fake import (
    CHULA_PAT14_6_PLACES,
    OBJECTS,
    TEAM,
    FakeRobot,
    chula_pat14_6,
)
from walkie_agent.errors import ErrorCode, StepError

FACTS = {
    "now": "2026-11-05T14:05:00+07:00",
    "team": {"name": "EIC", "country": "Thailand", "affiliation": "Chulalongkorn University"},
}


def world() -> dict:
    return {
        "places": {
            "lobby": {"x": 0.0, "y": 0.0, "yaw": 0.0, "room": "lobby"},
            "desk": {"x": 1.0, "y": 0.0, "yaw": 0.0, "room": "lobby"},
            "den": {"x": 10.0, "y": 0.0, "yaw": 0.0, "room": "den"},
            "shelf": {"x": 11.0, "y": 0.0, "yaw": 1.5, "room": "den"},
        },
        "objects": {
            "apple_1": {
                "name": "apple",
                "category": "fruit",
                "size_cm": 8,
                "weight_g": 180,
                "thickness_cm": 8,
                "at": "shelf",
            },
            "mug_1": {"name": "mug", "category": "dish", "at": "desk"},
            "dish_1": {"name": "plate", "category": "dish", "at": "shelf"},
        },
        "people": {
            "ann": {"name": "Ann", "room": "lobby", "pose": "sitting person", "gesture": "waving person"},
            "bob": {"name": "Bob", "room": "den", "pose": "standing person", "gesture": "waving person"},
        },
        "start": "lobby",
    }


@pytest.fixture
def robot() -> FakeRobot:
    return FakeRobot(world())


def code_of(call, *args, **kwargs) -> ErrorCode:
    with pytest.raises(StepError) as caught:
        call(*args, **kwargs)
    return caught.value.code


def holding_apple() -> FakeRobot:
    robot = FakeRobot(world())
    robot.go_to("shelf")
    robot.pick("apple_1")
    return robot


def test_it_implements_every_protocol_method(robot):
    for protocol in (Base, Speech, Vision, Arms, Facts):
        for name in vars(protocol):
            if not name.startswith("_"):
                assert callable(getattr(robot, name)), name


def test_its_ctx_holds_the_robot_and_its_token(robot):
    ctx = robot.ctx
    assert ctx.cancel is robot.cancel
    assert ctx.base is ctx.speech is ctx.vision is ctx.arms is ctx.facts is robot


# ctx.base


def test_a_pose_read_takes_a_timeout_and_logs_only_what_was_given(robot):
    robot.pose()
    robot.pose(timeout_s=0.5)
    assert [call["args"] for call in robot.calls if call["verb"] == "base.pose"] == [
        {},
        {"timeout_s": 0.5},
    ]
    with pytest.raises(StepError) as raised:
        robot.pose(timeout_s=0.0)
    assert raised.value.code is ErrorCode.REFUSED


def test_go_to_moves_the_robot_to_the_place(robot):
    pose = robot.go_to("shelf")
    assert (pose.x, pose.y, pose.yaw) == (11.0, 0.0, 1.5)
    assert robot.pose() == pose
    assert robot.snapshot()["at"] == "shelf"
    assert robot.snapshot()["room"] == "den"


def test_go_to_ignores_case_and_surrounding_space(robot):
    robot.go_to("  Shelf ")
    assert robot.snapshot()["at"] == "shelf"


def test_go_to_an_unknown_place_lists_the_known_ones(robot):
    with pytest.raises(StepError) as caught:
        robot.go_to("moon")
    assert caught.value.code is ErrorCode.NOT_FOUND
    assert "shelf" in caught.value.message
    assert robot.snapshot()["at"] == "lobby"


def test_places_returns_locations_and_tells_rooms_apart(robot):
    places = robot.places()
    assert places["shelf"].x == 11.0
    assert places["shelf"].description == "In the den."
    assert places["den"].description == "A room."


def test_move_relative_moves_in_the_robot_s_frame(robot):
    pose = robot.move_relative(1.0, 0.5, 90.0)
    assert (pose.x, pose.y) == pytest.approx((1.0, 0.5))
    assert pose.yaw == pytest.approx(math.pi / 2)
    assert robot.pose() == pose


@pytest.mark.parametrize(
    "move",
    [
        {"forward_m": float("nan")},
        {"left_m": float("inf")},
        {"turn_deg": float("-inf")},
        {},
        {"forward_m": 3.01},
        {"forward_m": 2.5, "left_m": 2.5},
    ],
)
def test_move_relative_refuses_and_leaves_the_robot_where_it_was(robot, move):
    before = robot.pose()
    assert code_of(robot.move_relative, **move) is ErrorCode.REFUSED
    assert robot.pose() == before


def test_move_relative_accepts_exactly_the_cap(robot):
    assert robot.move_relative(3.0).x == pytest.approx(3.0)


def test_a_small_move_keeps_the_place_and_a_long_one_leaves_it(robot):
    robot.move_relative(0.5)
    assert robot.snapshot()["at"] == "lobby"
    robot.move_relative(2.0)
    snapshot = robot.snapshot()
    assert snapshot["at"] is None
    assert snapshot["room"] == "lobby"


def test_a_move_away_from_any_place_stays_at_no_place(robot):
    robot.move_relative(3.0)
    robot.move_relative(0.1)
    assert robot.snapshot()["at"] is None


def test_a_robot_at_no_place_sees_no_objects(robot):
    robot.move_relative(3.0)
    assert robot.objects() == []


def test_a_drive_longer_than_its_timeout_times_out_without_arriving():
    robot = FakeRobot(world(), drive_s=30.0)
    assert code_of(robot.go_to, "shelf", timeout_s=0.03) is ErrorCode.TIMEOUT
    assert robot.snapshot()["at"] == "lobby"


def test_a_cancel_mid_drive_ends_it_without_arriving():
    robot = FakeRobot(world(), drive_s=30.0)
    timer = threading.Timer(0.03, robot.cancel.cancel)
    timer.start()
    started = time.monotonic()
    assert code_of(robot.go_to, "shelf") is ErrorCode.CANCELLED
    timer.join()
    assert time.monotonic() - started < 5.0
    assert robot.snapshot()["at"] == "lobby"


def test_base_stop_mid_drive_ends_it_without_arriving():
    robot = FakeRobot(world(), drive_s=30.0)
    timer = threading.Timer(0.03, robot.stop)
    timer.start()
    started = time.monotonic()
    assert code_of(robot.move_relative, 1.0) is ErrorCode.CANCELLED
    timer.join()
    assert time.monotonic() - started < 5.0
    assert robot.pose().x == 0.0


def test_stop_never_raises_and_does_not_block_the_next_drive(robot):
    robot.stop()
    robot.go_to("desk")
    assert robot.snapshot()["at"] == "desk"


def test_a_set_token_refuses_motion_until_a_reset(robot):
    robot.cancel.cancel()
    assert code_of(robot.go_to, "desk") is ErrorCode.CANCELLED
    assert code_of(robot.move_relative, 1.0) is ErrorCode.CANCELLED
    robot.stop()
    robot.cancel.reset()
    robot.go_to("desk")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_every_wait_refuses_a_timeout_that_bounds_nothing(robot, bad):
    robot.go_to("shelf")
    robot.pick("apple_1")
    calls = [
        lambda: robot.go_to("desk", timeout_s=bad),
        lambda: robot.move_relative(1.0, timeout_s=bad),
        lambda: robot.say("hi", timeout_s=bad),
        lambda: robot.objects(timeout_s=bad),
        lambda: robot.people(timeout_s=bad),
        lambda: robot.pick("mug_1", timeout_s=bad),
        lambda: robot.place_on("shelf", timeout_s=bad),
        lambda: robot.hand_over("bob", timeout_s=bad),
    ]
    for call in calls:
        assert code_of(call) is ErrorCode.REFUSED


# ctx.speech


def test_say_records_the_line(robot):
    robot.say("hello")
    assert robot.said == ["hello"]


def test_say_refuses_blank_text(robot):
    assert code_of(robot.say, "   ") is ErrorCode.REFUSED
    assert robot.said == []


def test_say_works_while_the_token_is_set(robot):
    robot.cancel.cancel()
    robot.say("Stopping.")
    assert robot.said == ["Stopping."]


# ctx.vision


def test_standing_at_a_place_shows_what_is_there(robot):
    robot.go_to("shelf")
    assert [seen.id for seen in robot.objects()] == ["apple_1", "dish_1"]


def test_standing_at_a_room_shows_every_place_in_it(robot):
    robot.go_to("den")
    assert [seen.place for seen in robot.objects()] == ["shelf", "shelf"]
    robot.go_to("lobby")
    assert [seen.id for seen in robot.objects()] == ["mug_1"]


def test_standing_at_a_place_hides_the_rest_of_the_room(robot):
    robot.go_to("desk")
    assert [seen.id for seen in robot.objects()] == ["mug_1"]
    robot.go_to("shelf")
    assert "mug_1" not in [seen.id for seen in robot.objects()]


def test_an_object_carries_its_measures_when_the_world_has_them(robot):
    robot.go_to("shelf")
    apple, plate = robot.objects()
    assert (apple.size_cm, apple.weight_g, apple.thickness_cm) == (8, 180, 8)
    assert (plate.size_cm, plate.weight_g, plate.thickness_cm) == (None, None, None)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("apple", ["apple_1"]),
        ("APPLE", ["apple_1"]),
        (" fruit ", ["apple_1"]),
        ("fruits", ["apple_1"]),
        ("dishes", ["dish_1"]),
        ("plate", ["dish_1"]),
        ("appl", []),
        ("mug", []),
    ],
)
def test_objects_match_a_name_category_or_plural(robot, query, expected):
    robot.go_to("shelf")
    assert [seen.id for seen in robot.objects(query)] == expected


def test_people_are_those_in_the_robot_s_room(robot):
    assert [seen.id for seen in robot.people()] == ["ann"]
    robot.go_to("shelf")
    seen = robot.people()
    assert [who.id for who in seen] == ["bob"]
    assert (seen[0].name, seen[0].pose, seen[0].room) == ("Bob", "standing person", "den")


@pytest.mark.parametrize(
    ("query", "expected"),
    [("ann", ["ann"]), ("sitting person", ["ann"]), ("waving person", ["ann"]), ("Bob", [])],
)
def test_people_match_a_name_pose_or_gesture(robot, query, expected):
    assert [seen.id for seen in robot.people(query)] == expected


def test_a_bearing_is_the_persons_heading_less_the_robots_yaw_and_a_turn_by_it_faces_them():
    facing = world()
    facing["people"]["ann"]["heading_deg"] = 170.0
    facing["places"]["lobby"]["yaw"] = math.radians(-160.0)
    robot = FakeRobot(facing)
    [ann] = robot.people()
    # 170 - (-160) = 330, the same direction as 30 degrees to the left.
    assert ann.bearing_deg == pytest.approx(-30.0)
    robot.move_relative(0.0, 0.0, ann.bearing_deg)
    assert robot.people()[0].bearing_deg == 0.0
    assert math.copysign(1.0, robot.people()[0].bearing_deg) == 1.0


def test_a_person_with_no_heading_gets_one_from_their_id_that_leaves_the_world_as_it_was(robot):
    first, again = robot.people()[0].bearing_deg, FakeRobot(world()).people()[0].bearing_deg
    assert first == again
    assert -180.0 <= first < 180.0
    assert "heading_deg" not in robot.snapshot()["people"]["ann"]
    robot.go_to("shelf")
    # bob's own heading less the shelf's yaw of 1.5 rad.
    bob = robot.people()[0].bearing_deg
    assert bob != first
    assert -180.0 <= bob < 180.0


def test_vision_refuses_while_the_token_is_set(robot):
    robot.cancel.cancel()
    assert code_of(robot.objects) is ErrorCode.CANCELLED
    assert code_of(robot.people) is ErrorCode.CANCELLED


# ctx.arms


def test_pick_takes_an_object_in_view():
    robot = holding_apple()
    snapshot = robot.snapshot()
    assert snapshot["holding"] == "apple_1"
    assert snapshot["objects"]["apple_1"]["at"] == "robot"


def test_pick_returns_the_object_s_id(robot):
    robot.go_to("shelf")
    assert robot.pick("apple_1") == "apple_1"


def test_pick_refuses_an_unknown_object(robot):
    assert code_of(robot.pick, "unicorn_1") is ErrorCode.NOT_FOUND


def test_pick_refuses_with_a_full_hand():
    robot = holding_apple()
    assert code_of(robot.pick, "dish_1") is ErrorCode.PRECONDITION_FAILED


def test_pick_refuses_an_object_out_of_view_without_saying_where_it_is(robot):
    with pytest.raises(StepError) as caught:
        robot.pick("apple_1")
    assert caught.value.code is ErrorCode.PRECONDITION_FAILED
    assert "shelf" not in caught.value.message


def test_place_on_puts_the_held_object_down():
    robot = holding_apple()
    robot.go_to("desk")
    assert robot.place_on("desk") == "apple_1"
    snapshot = robot.snapshot()
    assert snapshot["holding"] is None
    assert snapshot["objects"]["apple_1"]["at"] == "desk"


def test_place_on_refuses_an_unknown_place():
    assert code_of(holding_apple().place_on, "moon") is ErrorCode.NOT_FOUND


def test_place_on_refuses_a_room():
    robot = holding_apple()
    robot.go_to("den")
    assert code_of(robot.place_on, "den") is ErrorCode.REFUSED


def test_place_on_refuses_with_an_empty_hand(robot):
    robot.go_to("desk")
    assert code_of(robot.place_on, "desk") is ErrorCode.PRECONDITION_FAILED


def test_place_on_refuses_from_elsewhere():
    robot = holding_apple()
    assert code_of(robot.place_on, "desk") is ErrorCode.PRECONDITION_FAILED
    assert robot.snapshot()["holding"] == "apple_1"


def test_hand_over_gives_the_held_object_to_a_person_in_the_room():
    robot = holding_apple()
    assert robot.hand_over("bob") == "apple_1"
    snapshot = robot.snapshot()
    assert snapshot["holding"] is None
    assert snapshot["objects"]["apple_1"]["at"] == "bob"


def test_hand_over_refuses_an_unknown_person():
    assert code_of(holding_apple().hand_over, "carol") is ErrorCode.NOT_FOUND


def test_hand_over_refuses_with_an_empty_hand(robot):
    assert code_of(robot.hand_over, "ann") is ErrorCode.PRECONDITION_FAILED


def test_hand_over_refuses_a_person_in_another_room():
    robot = holding_apple()
    assert code_of(robot.hand_over, "ann") is ErrorCode.PRECONDITION_FAILED
    assert robot.snapshot()["holding"] == "apple_1"


@pytest.mark.parametrize(
    "call",
    [
        lambda robot: robot.pick("dish_1"),
        lambda robot: robot.place_on("shelf"),
        lambda robot: robot.hand_over("bob"),
    ],
)
def test_arms_refuse_while_the_token_is_set(call):
    robot = holding_apple()
    robot.cancel.cancel()
    assert code_of(call, robot) is ErrorCode.CANCELLED
    assert robot.snapshot()["holding"] == "apple_1"


# Records and faults


def test_every_call_is_logged_with_its_arguments_and_outcome(robot):
    robot.go_to("desk")
    with pytest.raises(StepError):
        robot.go_to("moon")
    robot.move_relative(0.5, turn_deg=10.0, timeout_s=5.0)
    desk = {"at": "desk", "room": "lobby"}
    assert robot.calls == [
        {"verb": "base.go_to", "args": {"place": "desk"}, "error": None, **desk},
        {"verb": "base.go_to", "args": {"place": "moon"}, "error": "NOT_FOUND", **desk},
        {
            "verb": "base.move_relative",
            "args": {"forward_m": 0.5, "turn_deg": 10.0, "timeout_s": 5.0},
            "error": None,
            **desk,
        },
    ]


def test_a_record_says_where_the_robot_was_after_a_move_that_left_its_place(robot):
    robot.go_to("den")
    robot.move_relative(forward_m=-3.0)
    robot.say("Still in the den.")
    robot.move_relative(forward_m=-3.0)
    robot.say("Nearer the lobby.")

    assert [(call["at"], call["room"]) for call in robot.calls] == [
        ("den", "den"),
        (None, "den"),
        (None, "den"),
        (None, "lobby"),
        (None, "lobby"),
    ]


def test_a_failed_move_is_recorded_where_the_robot_stayed(robot):
    robot.faults["base.go_to"] = StepError(ErrorCode.SKILL_FAILED, "Nav2 aborted.")
    assert code_of(robot.go_to, "den") is ErrorCode.SKILL_FAILED

    assert (robot.calls[-1]["at"], robot.calls[-1]["room"]) == ("lobby", "lobby")


def test_a_lookup_records_the_ids_it_returned(robot):
    robot.people()
    robot.people("Nobody")
    robot.go_to("shelf")
    robot.objects()
    robot.objects("fruit")

    lookups = [call for call in robot.calls if call["verb"].startswith("vision.")]
    assert [call["returned"] for call in lookups] == [["ann"], [], ["apple_1", "dish_1"], ["apple_1"]]
    assert "returned" not in next(call for call in robot.calls if call["verb"] == "base.go_to")


def test_a_refused_lookup_records_nothing_returned(robot):
    assert code_of(robot.people, timeout_s=-1.0) is ErrorCode.REFUSED

    assert "returned" not in robot.calls[-1]
    assert robot.calls[-1]["error"] == "REFUSED"


def test_the_records_stay_json_able(robot):
    robot.people()
    robot.stop()

    assert json.loads(json.dumps(robot.snapshot()["calls"])) == robot.calls


def test_a_fault_makes_every_call_to_that_verb_fail(robot):
    robot.faults["base.go_to"] = StepError(ErrorCode.SKILL_FAILED, "Nav2 aborted.")
    assert code_of(robot.go_to, "desk") is ErrorCode.SKILL_FAILED
    assert robot.calls[-1]["error"] == "SKILL_FAILED"
    assert robot.snapshot()["at"] == "lobby"
    robot.say("still fine")
    del robot.faults["base.go_to"]
    robot.go_to("desk")


def test_a_fault_on_stop_neither_raises_nor_lets_the_drive_finish():
    robot = FakeRobot(world(), drive_s=30.0)
    robot.faults["base.stop"] = StepError(ErrorCode.UNAVAILABLE, "No cancel backend.")
    timer = threading.Timer(0.03, robot.stop)
    timer.start()
    assert code_of(robot.go_to, "shelf") is ErrorCode.CANCELLED
    timer.join()
    assert robot.snapshot()["at"] == "lobby"
    assert robot.calls[-1] == {
        "verb": "base.stop",
        "args": {},
        "error": None,
        "at": "lobby",
        "room": "lobby",
    }


def test_a_cancel_after_the_wait_and_before_the_commit_keeps_the_robot_in_place(
    robot, monkeypatch
):
    wait_for = robot.cancel.wait_for

    def then_cancel(*args, **kwargs) -> None:
        wait_for(*args, **kwargs)
        robot.cancel.cancel()

    monkeypatch.setattr(robot.cancel, "wait_for", then_cancel)
    before = robot.snapshot()
    assert code_of(robot.go_to, "shelf") is ErrorCode.CANCELLED
    after = robot.snapshot()
    assert (after["at"], after["pose"]) == (before["at"], before["pose"])
    assert robot.calls[-1]["error"] == "CANCELLED"


@pytest.mark.parametrize(
    ("verb", "args"),
    [
        ("go_to", (None,)),
        ("move_relative", ("far",)),
        ("move_relative", (True,)),
        ("say", (123,)),
        ("objects", (1,)),
        ("people", (None,)),
        ("pick", ([],)),
        ("hand_over", (1,)),
        ("place_on", (None,)),
    ],
)
def test_a_wrong_type_is_refused_logged_and_changes_nothing(robot, verb, args):
    before = robot.snapshot()
    assert code_of(getattr(robot, verb), *args) is ErrorCode.REFUSED
    assert robot.calls[-1]["error"] == "REFUSED"
    after = robot.snapshot()
    for key in ("at", "pose", "objects", "said"):
        assert after[key] == before[key]


def test_an_unexpected_exception_is_logged_as_internal_and_not_hidden(robot):
    robot._objects = None
    with pytest.raises(AttributeError):
        robot.pick("apple_1")
    assert robot.calls[-1]["error"] == "INTERNAL"


@pytest.mark.parametrize(
    ("break_it", "named"),
    [
        (lambda w: w.update(start="lobbby"), "start"),
        (lambda w: w["places"]["desk"].update(room="lobbby"), "places['desk'].room"),
        (lambda w: w["people"]["ann"].update(room="lobbby"), "people['ann'].room"),
        (lambda w: w["people"]["ann"].update(heading_deg=float("nan")), "people['ann'].heading_deg"),
        (lambda w: w["people"]["ann"].update(heading_deg="left"), "people['ann'].heading_deg"),
        (lambda w: w["objects"]["mug_1"].update(at="dsek"), "objects['mug_1'].at"),
        (lambda w: w["places"]["desk"].update(x=float("nan")), "places['desk'].x"),
        (lambda w: w["places"]["desk"].update(yaw=float("inf")), "places['desk'].yaw"),
        (lambda w: w["places"]["desk"].update(y=True), "places['desk'].y"),
        (lambda w: w.update(facts={**FACTS, "now": "2026-11-05T14:05:00"}), "facts.now"),
        (lambda w: w.update(facts={**FACTS, "now": "next Thursday"}), "facts.now"),
        (lambda w: w.update(facts={"team": FACTS["team"]}), "facts.now"),
        (lambda w: w.update(facts={**FACTS, "team": {**TEAM, "country": " "}}), "facts.team.country"),
        (lambda w: w.update(facts={"now": FACTS["now"]}), "facts.team.name"),
    ],
)
def test_a_broken_world_is_refused_and_the_message_names_the_field(break_it, named):
    broken = world()
    break_it(broken)
    with pytest.raises(ValueError, match="Broken world") as caught:
        FakeRobot(broken)
    assert named in str(caught.value)


def test_a_world_may_start_with_an_object_held_or_already_handed_over():
    start = world()
    start["objects"]["mug_1"]["at"] = "robot"
    start["objects"]["apple_1"]["at"] = "ann"
    assert FakeRobot(start).snapshot()["holding"] == "mug_1"


@pytest.mark.parametrize("seed", range(20))
def test_every_seeded_arena_passes_the_world_check(seed):
    FakeRobot(seed=seed)


def test_the_snapshot_is_json_able_even_after_a_refused_nan():
    robot = holding_apple()
    robot.say("hello")
    with pytest.raises(StepError):
        robot.move_relative(float("nan"))
    snapshot = robot.snapshot()
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot
    assert snapshot["calls"][-1]["args"] == {"forward_m": "nan"}
    assert snapshot["said"] == ["hello"]
    assert snapshot["pose"] == {"x": 11.0, "y": 0.0, "yaw": 1.5}


def test_the_snapshot_shares_nothing_with_the_robot():
    robot = holding_apple()
    snapshot = robot.snapshot()
    snapshot["objects"]["apple_1"]["at"] = "shelf"
    snapshot["said"].append("tampered")
    assert robot.snapshot()["objects"]["apple_1"]["at"] == "robot"
    assert robot.said == []


def test_the_robot_does_not_change_the_world_it_was_given():
    given = world()
    robot = FakeRobot(given)
    robot.go_to("shelf")
    robot.pick("apple_1")
    assert given == world()


# The seeded arena


def test_the_arena_is_deterministic_per_seed():
    assert chula_pat14_6(3) == chula_pat14_6(3)
    assert FakeRobot(seed=3).snapshot() == FakeRobot(chula_pat14_6(3)).snapshot()


def test_seeds_change_the_clutter():
    # Without the facts, which change with every seed whatever the clutter does.
    clutter = [{**chula_pat14_6(seed), "facts": None} for seed in range(5)]
    assert len({json.dumps(arena, sort_keys=True) for arena in clutter}) > 1


def test_the_arena_is_json_able():
    arena = chula_pat14_6(1)
    assert json.loads(json.dumps(arena)) == arena


def test_a_fresh_arena_does_not_share_the_module_s_places():
    chula_pat14_6()["places"]["shelf"]["x"] = 99.0
    assert CHULA_PAT14_6_PLACES["shelf"]["x"] != 99.0


def test_the_arena_has_the_real_rooms_and_locations_and_the_trash():
    places = chula_pat14_6()["places"]
    assert {"living room", "kitchen room", "shelf", "bar", "table", "tv", "sofa", "trash"} <= set(places)
    assert places["trash"]["room"] == "kitchen room"
    assert places["shelf"]["x"] == 2.4868


def test_the_arena_puts_every_object_on_a_surface_and_every_person_in_a_room():
    for seed in range(20):
        arena = chula_pat14_6(seed)
        rooms = {place["room"] for place in arena["places"].values()}
        for item in arena["objects"].values():
            assert item["at"] in arena["places"]
            assert arena["places"][item["at"]]["room"] != item["at"]
        for who in arena["people"].values():
            assert who["room"] in rooms


def test_the_arena_has_the_operator_at_the_start_and_two_people_per_room():
    arena = chula_pat14_6(2)
    people = arena["people"]
    assert people["operator"]["room"] == arena["places"][arena["start"]]["room"]
    others = [who["room"] for key, who in people.items() if key != "operator"]
    assert sorted(others) == ["kitchen room", "kitchen room", "living room", "living room"]
    assert len({who["name"] for who in people.values()}) == len(people)


@pytest.mark.parametrize("measure", ["size_cm", "weight_g", "thickness_cm"])
def test_each_measure_is_unique_so_a_superlative_has_one_answer(measure):
    values = [item[measure] for item in OBJECTS.values()]
    assert len(set(values)) == len(values)


def test_the_arena_fixes_a_clock_in_2026_and_the_configured_team_per_seed():
    facts = chula_pat14_6(5)["facts"]
    assert facts == chula_pat14_6(5)["facts"]
    now = datetime.fromisoformat(facts["now"])
    assert now.year == 2026 and now.second == 0
    assert now.utcoffset() == timedelta(hours=7)
    assert facts["team"] == TEAM == {
        "name": "EIC",
        "country": "Thailand",
        "affiliation": "Chulalongkorn University",
    }


def test_the_arena_clock_spreads_over_the_year_and_the_day():
    clocks = [datetime.fromisoformat(chula_pat14_6(seed)["facts"]["now"]) for seed in range(200)]
    assert len({clock.date() for clock in clocks}) > 150
    assert {clock.month for clock in clocks} == set(range(1, 13))
    assert {clock.hour for clock in clocks} == set(range(24))
    assert all(clock.year == 2026 for clock in clocks)


@pytest.mark.parametrize(
    ("seed", "digest"), [(0, "c97c2229378293df"), (1, "1e72e263fc77770f"), (7, "764f472db3d0e6e8")]
)
def test_the_arena_facts_draw_leaves_the_clutter_and_the_people_alone(seed, digest):
    # The digests of these arenas before they had facts, which draw from their own stream.
    # Re-pinned when decision G4-6 moved "trash" and "instruction point"; with the old
    # coordinates put back, each arena still hashes to its earlier digest.
    arena = chula_pat14_6(seed)
    del arena["facts"]
    assert hashlib.sha256(json.dumps(arena, sort_keys=True).encode()).hexdigest()[:16] == digest


# ctx.facts


def test_the_facts_verbs_read_the_world_s_facts():
    robot = FakeRobot({**world(), "facts": FACTS})
    assert robot.now() == datetime(2026, 11, 5, 14, 5, tzinfo=timezone(timedelta(hours=7)))
    assert robot.team() == Team("EIC", "Thailand", "Chulalongkorn University")
    assert [call["verb"] for call in robot.calls] == ["facts.now", "facts.team"]
    assert robot.snapshot()["facts"] == FACTS


def test_a_world_without_facts_has_no_fact_source(robot):
    assert code_of(robot.now) is ErrorCode.UNAVAILABLE
    assert code_of(robot.team) is ErrorCode.UNAVAILABLE
    assert robot.calls[-1]["error"] == "UNAVAILABLE"
    assert robot.snapshot()["facts"] is None


def test_the_facts_verbs_work_while_the_token_is_set():
    robot = FakeRobot({**world(), "facts": FACTS})
    robot.cancel.cancel()
    assert robot.now().hour == 14
    assert robot.team().name == "EIC"


def test_a_seeded_robot_starts_at_the_instruction_point_and_can_see_the_operator():
    robot = FakeRobot(seed=4)
    assert robot.snapshot()["at"] == "instruction point"
    assert "operator" in [who.id for who in robot.people()]
