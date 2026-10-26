"""The HRI pass check on synthetic logs.

passing_run() builds the event log and the tool log of a run that keeps
every rule, from small record factories. Each test changes one thing in it
and checks the one criterion that judges it, so a test names the rule it
covers. The CLI's tests write the logs to files and run main().
"""

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from walkie_agent.arm.state import ARMS_OUT
from walkie_agent.sim import hri_check
from walkie_agent.sim.hri_check import CRITERIA, Result, Run, check

Record = dict[str, Any]

GUESTS = {"guest1": {"name": "Sam", "drink": "milk"}, "guest2": {"name": "Ana", "drink": "cola"}}
# Track ids: each guest at the door and seated, the onlookers and the bystander.
DOOR_TRACK = {"guest1": 1, "guest2": 4}
SEATED_TRACK = {"guest1": 3, "guest2": 7}
ONLOOKER_TRACKS = (2, 5)
BYSTANDER_TRACK = 6
HANDLE = {"guest1": "per_aaaa.1", "guest2": "per_bbbb.1"}
ONLOOKER_HANDLE = "per_cccc.1"
# Each conversation's start; its samples come 1 s later, 0.5 s apart.
TALK_T = {"guest1": 12.0, "guest2": 40.0}
SEEDED_T = 10.0
DONE_T = 80.0


# -- Record factories ---------------------------------------------------------------------


def event(type_: str, t: float, **fields: Any) -> Record:
    return {"type": type_, "t": t, **fields}


def start(**changes: Any) -> Record:
    scenario = {"answers": {"delay": 1.0, "mumble_once": None}, "bag": {"early_answer": False}}
    return event(
        "start",
        0.0,
        scenario=scenario,
        guests=GUESTS,
        bag_owner="guest2",
        onlookers=["onlooker1"],
        **changes,
    )


def person(t: float, who: str, state: str, track_id: int | None, place: str | None = None):
    return event("person", t, person=who, state=state, track_id=track_id, place=place)


def todos(t: float, status: str) -> Record:
    return event("todos", t, items=[{"id": f"t{i}", "status": status} for i in range(6)])


def line(
    t: float,
    text: str,
    guest: str | None,
    asks: list[str] = (),
    offer: bool = False,
    bearings_deg: Record | None = None,
) -> Record:
    return event(
        "robot_line",
        t,
        text=text,
        guest=guest,
        question=text.endswith("?"),
        asks=list(asks),
        offer=offer,
        pose=pose(0.0, 0.0, 0.0),
        bearings_deg=bearings_deg,
    )


def answer(t: float, guest: str, asked_t: float, asks: list[str], mumbled: bool = False):
    return event("answer", t, guest=guest, asked_t=asked_t, asks=asks, mumbled=mumbled)


def pose(x: float, y: float, yaw: float) -> Record:
    return {"x": x, "y": y, "yaw": yaw}


def state(
    t: float,
    guest: str | None,
    robot_yaw_deg: float,
    guest_y: float,
    nav_active: bool = False,
    arms_travel_safe: bool | None = True,
) -> Record:
    """A state record with the robot at the origin and the talking guest at (2, guest_y)."""
    talking = None
    if guest is not None:
        direction = math.degrees(math.atan2(guest_y, 2.0))
        bearing = round(hri_check._wrap_deg(direction - robot_yaw_deg), 2)
        talking = {"guest": guest, "x": 2.0, "y": guest_y, "bearing_deg": bearing}
    return event(
        "state",
        t,
        pose=pose(0.0, 0.0, math.radians(robot_yaw_deg)),
        talking=talking,
        conversation=talking is not None,
        nav_active=nav_active,
        arms_travel_safe=arms_travel_safe,
    )


def conversation(guest: str, error_deg: float = 2.0, turn: bool = True) -> list[Record]:
    """Ten samples of a guest pacing ±0.5 m, with the robot `error_deg` off them or still."""
    samples = []
    for k in range(10):
        guest_y = 0.5 * math.sin(k)
        direction = math.degrees(math.atan2(guest_y, 2.0))
        yaw = direction - error_deg if turn else -error_deg
        samples.append(state(TALK_T[guest] + 1.0 + 0.5 * k, guest, yaw, guest_y))
    return samples


def cmd_vel(
    t: float,
    angular_z: float = 0.0,
    linear_x: float = 0.0,
    nav_active: bool = False,
    arms_travel_safe: bool | None = True,
) -> Record:
    return event(
        "cmd_vel",
        t,
        linear=[linear_x, 0.0, 0.0],
        angular=[0.0, 0.0, angular_z],
        nav_active=nav_active,
        arms_travel_safe=arms_travel_safe,
    )


def ref(handle: str, track_id: int | None, where: str = "args") -> Record:
    return {"handle": handle, "in": where, "kind": "person", "track_id": track_id, "generation": 1}


def call(
    t: float,
    tool: str,
    args: Record | None = None,
    outcome: str = "ok",
    result: str = "",
    out: Record | None = None,
    people: list[Record] = (),
    gaze: Record | None = None,
) -> Record:
    return {
        "type": "call",
        "t": t,
        "t_end": t + 0.2,
        "wall_s": 0.2,
        "tool": tool,
        "call_id": f"call_{t:g}",
        "args": args or {},
        "outcome": outcome,
        "code": None,
        "result": result,
        "out": out,
        "people": list(people),
        "gaze": gaze,
    }


def speak(t: float, text: str, track_id: int | None = None) -> Record:
    gaze = {"target": None, "running": track_id is not None, "track_id": track_id}
    return call(t, "speak", {"text": text}, gaze=gaze)


def gripper(t: float, state_: str, holding: bool | None = None, outcome: str = "ok", arm="left"):
    out = {"summary": "", "arm": arm, "state": state_, "holding": {}}
    if holding is not None:
        out["holding"] = {"left": holding}
    return call(t, "set_gripper", {"arm": arm, "state": state_}, outcome, out=out)


def remember(t: float, guest: str, name: str, facts: Record, track_id: int | None = None):
    track = DOOR_TRACK[guest] if track_id is None else track_id
    args = {"person": HANDLE[guest], "name": name, "facts": facts}
    return call(t, "remember_person", args, people=[ref(HANDLE[guest], track)])


# -- The passing run ----------------------------------------------------------------------


def passing_run() -> tuple[list[Record], list[Record]]:
    """The two logs of a run that keeps every rule, in the order they were written."""
    g1, g2 = TALK_T["guest1"], TALK_T["guest2"]
    events = [
        start(),
        person(0.0, "guest1", "at_door", DOOR_TRACK["guest1"]),
        event("onlooker", 0.0, onlooker="onlooker1", state="standing", track_id=2),
        todos(SEEDED_T, "pending"),
        line(g1, "Hello, what is your name?", "guest1", ["name"]),
        person(g1, "guest1", "talking", DOOR_TRACK["guest1"]),
        answer(g1 + 1, "guest1", g1, ["name"]),
        line(g1 + 2, "What is your favourite drink?", "guest1", ["drink"]),
        answer(g1 + 3, "guest1", g1 + 2, ["drink"]),
        *conversation("guest1"),
        person(17.0, "guest1", "following", None),
        event("onlooker", 17.0, onlooker="onlooker1", state="gone", track_id=2),
        cmd_vel(20.0, angular_z=0.3),
        cmd_vel(21.0, nav_active=True),
        line(30.0, "Please sit here.", "guest1", offer=True),
        event("seat_taken", 30.0, guest="guest1", place="sofa.middle"),
        person(30.0, "guest1", "seated", SEATED_TRACK["guest1"], "sofa.middle"),
        person(31.0, "guest2", "at_door", DOOR_TRACK["guest2"]),
        event("onlooker", 31.0, onlooker="onlooker1", state="standing", track_id=5),
        event("bystander", 33.0, state="walking", track_id=BYSTANDER_TRACK),
        line(g2, "Hello, what is your name?", "guest2", ["name"]),
        person(g2, "guest2", "talking", DOOR_TRACK["guest2"]),
        answer(g2 + 1, "guest2", g2, ["name"]),
        line(g2 + 2, "What would you like to drink?", "guest2", ["drink"]),
        answer(g2 + 3, "guest2", g2 + 2, ["drink"]),
        *conversation("guest2"),
        line(45.0, "Ana, I will raise my left arm for your bag.", "guest2"),
        line(47.0, "Please put your bag on my left hand.", "guest2", ["bag"]),
        event("bag_request", 47.0, guest="guest2", count=1, hang=True, early_answer=False),
        event("bag_hung", 47.5, value=-1.0, already_held=False),
        answer(48.0, "guest2", 47.0, ["bag"]),
        person(50.0, "guest2", "following", None),
        event("bystander", 52.0, state="gone", track_id=BYSTANDER_TRACK),
        line(60.0, "Please take this seat.", "guest2", offer=True),
        event("seat_taken", 60.0, guest="guest2", place="sofa.right"),
        person(60.0, "guest2", "seated", SEATED_TRACK["guest2"], "sofa.right"),
        line(70.0, "Sam, this is Ana. Ana likes cola.", None, bearings_deg=bearings(3.0, 30.0)),
        line(75.0, "Ana, this is Sam. Sam likes milk.", None, bearings_deg=bearings(30.0, -2.0)),
        todos(DONE_T, "completed"),
    ]
    tools = [
        {"type": "start", "t": 0.0, "wall_time": "2026-10-24T10:00:00+07:00"},
        {"type": "new_task", "t": SEEDED_T},
        call(11.0, "look_at", {"target": HANDLE["guest1"]}, people=[ref(HANDLE["guest1"], 1)]),
        speak(g1, "Hello, what is your name?", DOOR_TRACK["guest1"]),
        speak(g1 + 2, "What is your favourite drink?", DOOR_TRACK["guest1"]),
        remember(16.0, "guest1", "Sam", {"drink": "milk"}),
        call(20.0, "navigate_to_location", {"location": "living_room"}),
        speak(30.0, "Please sit here."),
        call(32.0, "look_at", {"target": HANDLE["guest2"]}, people=[ref(HANDLE["guest2"], 4)]),
        speak(g2, "Hello, what is your name?", DOOR_TRACK["guest2"]),
        speak(g2 + 2, "What would you like to drink?", DOOR_TRACK["guest2"]),
        # A key that holds "drink" and a value that holds the drink, in any case.
        remember(44.0, "guest2", " ana ", {"Favourite drink": "Cola, please"}),
        speak(45.0, "Ana, I will raise my left arm for your bag."),
        call(45.5, "move_arm_to_pose", {"arm": "left", "pose": "receive"}),
        speak(47.0, "Please put your bag on my left hand."),
        gripper(49.0, "close", holding=True),
        call(51.0, "navigate_to_location", {"location": "living_room"}),
        speak(60.0, "Please take this seat."),
        speak(70.0, "Sam, this is Ana. Ana likes cola."),
        speak(75.0, "Ana, this is Sam. Sam likes milk."),
    ]
    return events, tools


def bearings(guest1: float, guest2: float) -> Record:
    return {"host": 20.0, "guest1": guest1, "guest2": guest2}


def results(events: list[Record], tools: list[Record]) -> dict[str, Result]:
    return {result.name: result for result in check(Run(events, tools))}


def result_of(name: str, events: list[Record], tools: list[Record]) -> Result:
    return results(events, tools)[name]


def at(records: list[Record], type_: str, **fields: Any) -> int:
    """The index of the first record of `type_` with `fields`."""
    return next(
        i
        for i, r in enumerate(records)
        if r.get("type") == type_ and all(r.get(k) == v for k, v in fields.items())
    )


def tool_at(tools: list[Record], tool: str, t: float) -> int:
    return next(i for i, r in enumerate(tools) if r.get("tool") == tool and r["t"] == t)


def test_a_run_that_keeps_every_rule_passes_every_criterion_in_hri_md_order():
    events, tools = passing_run()

    found = check(Run(events, tools))

    assert [r.name for r in found] == [
        "memory",
        "gaze guest1",
        "gaze guest2",
        "offers",
        "introductions",
        "questions",
        "arm_motion",
        "bag",
        "drives",
        "cmd_vel",
        "valid",
        "time",
        "people",
    ]
    assert [r for r in found if not r.ok] == []
    assert len(CRITERIA) == 12


# -- memory -------------------------------------------------------------------------------


def test_memory_fails_when_a_guests_drink_is_not_remembered():
    events, tools = passing_run()
    no_drink = remember(16.0, "guest1", "Sam", {"food": "milk"})
    tools[tool_at(tools, "remember_person", 16.0)] = no_drink

    result = result_of("memory", events, tools)

    assert not result.ok
    assert "guest1 remembered as name 'Sam'" in result.reason


def test_memory_fails_when_the_name_goes_on_another_persons_track():
    events, tools = passing_run()
    tools[tool_at(tools, "remember_person", 16.0)] = remember(
        16.0, "guest1", "Sam", {"drink": "milk"}, track_id=ONLOOKER_TRACKS[0]
    )

    result = result_of("memory", events, tools)

    assert (result.ok, result.reason) == (False, "no ok remember_person on guest1's track")


def test_memory_counts_a_remember_on_the_guests_seated_track():
    events, tools = passing_run()
    tools[tool_at(tools, "remember_person", 16.0)] = remember(
        16.0, "guest1", "Sam", {"drink": "milk"}, track_id=SEATED_TRACK["guest1"]
    )

    assert result_of("memory", events, tools).ok


def test_memory_ignores_a_remember_person_that_failed():
    events, tools = passing_run()
    tools[tool_at(tools, "remember_person", 16.0)]["outcome"] = "error"

    assert not result_of("memory", events, tools).ok


def test_memory_ignores_a_remember_person_from_before_the_last_new_task():
    events, tools = passing_run()
    # A new task cleared the people memory after guest 1 was remembered.
    tools.insert(tool_at(tools, "remember_person", 16.0) + 1, {"type": "new_task", "t": 17.0})

    result = result_of("memory", events, tools)

    assert not result.ok
    assert result.reason == "no ok remember_person on guest1's track"


# -- gaze ---------------------------------------------------------------------------------


def test_gaze_reports_the_share_within_10_deg_and_the_ratio():
    events, tools = passing_run()

    result = result_of("gaze guest1", events, tools)

    assert result.ok
    assert result.reason.startswith("100% within 10 deg, ratio 0.")
    assert result.reason.endswith("(10 samples)")


def test_gaze_fails_a_robot_that_never_turns_on_the_ratio():
    events, tools = passing_run()
    first = at(events, "state")
    events[first : first + 10] = conversation("guest1", error_deg=0.0, turn=False)

    result = result_of("gaze guest1", events, tools)

    assert not result.ok
    assert "ratio 1.00" in result.reason


def test_gaze_fails_when_fewer_than_80_percent_of_samples_are_within_10_deg():
    events, tools = passing_run()
    first = at(events, "state")
    events[first : first + 10] = conversation("guest1", error_deg=12.0)

    result = result_of("gaze guest1", events, tools)

    assert not result.ok
    assert result.reason.startswith("0% within 10 deg")


@pytest.mark.parametrize(("within", "ok"), [(8, True), (7, False)])
def test_gaze_needs_80_percent_of_samples_within_10_deg(within, ok):
    events, tools = passing_run()
    first = at(events, "state")
    samples = conversation("guest1")
    # The last samples 12 deg off, so the first sample's heading, the
    # baseline's, stays the same.
    samples[within:] = conversation("guest1", error_deg=12.0)[within:]
    events[first : first + 10] = samples

    result = result_of("gaze guest1", events, tools)

    assert result.ok is ok
    assert result.reason.startswith(f"{within}0% within 10 deg, ratio 0.")


def partial_turn(guest: str, turned: float) -> list[Record]:
    """Ten samples of a guest swaying ±0.1 m, the robot turning `turned` of the way to them.

    Every sample is within 10 deg, and the first faces the guest, so the
    ratio is 1 - `turned`.
    """
    samples = []
    for k in range(10):
        guest_y = 0.1 * math.sin(k)
        direction = math.degrees(math.atan2(guest_y, 2.0))
        samples.append(state(TALK_T[guest] + 1.0 + 0.5 * k, guest, turned * direction, guest_y))
    return samples


@pytest.mark.parametrize(("turned", "ratio", "ok"), [(0.05, "0.95", False), (0.15, "0.85", True)])
def test_gaze_needs_a_mean_error_at_most_0_9_of_the_baseline(turned, ratio, ok):
    events, tools = passing_run()
    first = at(events, "state")
    events[first : first + 10] = partial_turn("guest1", turned)

    result = result_of("gaze guest1", events, tools)

    assert result.ok is ok
    assert result.reason.startswith(f"100% within 10 deg, ratio {ratio}")


def test_gaze_skips_the_ratio_when_the_baseline_is_below_1_deg():
    events, tools = passing_run()
    first = at(events, "state")
    # A guest standing still: a robot that faces them from the start has no baseline.
    still = [state(TALK_T["guest1"] + 1 + 0.5 * k, "guest1", 0.5, 0.0) for k in range(10)]
    events[first : first + 10] = still

    result = result_of("gaze guest1", events, tools)

    assert result.ok
    assert "so no ratio" in result.reason


def test_gaze_fails_a_conversation_without_samples():
    events, tools = passing_run()
    first = at(events, "state")
    for sample in events[first : first + 10]:
        sample["pose"] = None

    result = result_of("gaze guest1", events, tools)

    assert (result.ok, result.reason) == (False, "no samples")


def test_gaze_fails_a_guest_with_no_conversation():
    events, tools = passing_run()
    del events[at(events, "person", person="guest2", state="talking")]

    result = result_of("gaze guest2", events, tools)

    assert (result.ok, result.reason) == (False, "no conversation")


@pytest.mark.parametrize(
    "fields",
    [{"nav_active": True}, {"arms_travel_safe": False}, {"arms_travel_safe": None}],
    ids=["nav-active", "arm-out", "arms-unknown"],
)
def test_gaze_leaves_out_samples_while_nav2_drives_or_an_arm_is_out(fields):
    events, tools = passing_run()
    first = at(events, "state")
    # Facing away while the gaze loop is stopped by design.
    away = [state(16.0 + 0.04 * k, "guest1", 60.0, 0.0, **fields) for k in range(20)]
    events[first + 10 : first + 10] = away

    result = result_of("gaze guest1", events, tools)

    assert result.ok
    assert result.reason.endswith("(10 samples)")


# -- offers -------------------------------------------------------------------------------


def test_offers_fail_on_a_bad_offer():
    events, tools = passing_run()
    bad = event(
        "bad_offer", 29.0, guest="guest1", place="sofa.left", reason="taken", taken_by="host"
    )
    events.insert(at(events, "seat_taken"), bad)

    result = result_of("offers", events, tools)

    assert (result.ok, result.reason) == (
        False,
        "bad offer to guest1 at t=29 s (taken, sofa.left taken by host)",
    )


def test_offers_fail_when_a_guest_never_takes_a_seat():
    events, tools = passing_run()
    del events[at(events, "seat_taken", guest="guest2")]

    result = result_of("offers", events, tools)

    assert (result.ok, result.reason) == (False, "guest2 never took a seat")


# -- introductions ------------------------------------------------------------------------


def test_an_introduction_that_does_not_face_the_listener_does_not_count():
    events, tools = passing_run()
    events[at(events, "robot_line", t=70.0)]["bearings_deg"] = bearings(12.0, 30.0)

    result = result_of("introductions", events, tools)

    assert not result.ok
    assert result.reason == (
        "no introduction faced guest1: the nearest was 12.0 deg off, over 10"
    )


def test_an_introduction_before_both_guests_sit_does_not_count():
    events, tools = passing_run()
    intro = events.pop(at(events, "robot_line", t=70.0))
    events.insert(at(events, "seat_taken", guest="guest2"), intro)

    result = result_of("introductions", events, tools)

    assert (result.ok, result.reason) == (False, "no introduction to guest1")


def test_a_line_that_names_both_drinks_is_not_an_introduction():
    events, tools = passing_run()
    events[at(events, "robot_line", t=70.0)]["text"] = "Sam likes milk and Ana likes cola."

    assert not result_of("introductions", events, tools).ok


# -- questions ----------------------------------------------------------------------------


def test_questions_fail_on_an_extra_question():
    events, tools = passing_run()
    events.append(event("extra_question", 20.0, guest="guest1", text="Who are you?"))

    result = result_of("questions", events, tools)

    assert (result.ok, result.reason) == (False, "extra question to guest1: 'Who are you?'")


def test_questions_fail_when_a_guest_is_asked_the_drink_twice():
    events, tools = passing_run()
    events.insert(at(events, "person", person="guest1", state="following"), repeat("guest1", 16.0))

    result = result_of("questions", events, tools)

    assert (result.ok, result.reason) == (False, "guest1 drink asked 2 times")


def repeat(guest: str, t: float) -> Record:
    return line(t, "Sorry, what drink was that?", guest, ["drink"])


def with_mumble(events: list[Record]) -> None:
    events[0]["scenario"]["answers"]["mumble_once"] = "guest1.drink"
    events[at(events, "answer", asks=["drink"], guest="guest1")]["mumbled"] = True


def test_questions_allow_one_repeat_after_the_mumbled_answer():
    events, tools = passing_run()
    with_mumble(events)
    events.insert(at(events, "person", person="guest1", state="following"), repeat("guest1", 16.0))

    assert result_of("questions", events, tools).ok


def test_questions_allow_no_second_repeat_after_the_mumbled_answer():
    events, tools = passing_run()
    with_mumble(events)
    following = at(events, "person", person="guest1", state="following")
    events[following:following] = [repeat("guest1", 16.0), repeat("guest1", 16.5)]

    result = result_of("questions", events, tools)

    assert (result.ok, result.reason) == (False, "guest1 drink asked 3 times")


def test_questions_allow_no_repeat_when_the_answer_was_not_mumbled():
    events, tools = passing_run()
    # The variant is on, but the answer came clearly: nothing to repeat.
    events[0]["scenario"]["answers"]["mumble_once"] = "guest1.drink"
    events.insert(at(events, "person", person="guest1", state="following"), repeat("guest1", 16.0))

    result = result_of("questions", events, tools)

    assert (result.ok, result.reason) == (False, "guest1 drink asked 2 times")


def test_questions_allow_no_repeat_before_the_mumbled_answer_comes():
    events, tools = passing_run()
    with_mumble(events)
    # Asked twice before the mumble was heard: no question was repeated after it.
    events.insert(at(events, "answer", asks=["drink"], guest="guest1"), repeat("guest1", 14.5))

    result = result_of("questions", events, tools)

    assert not result.ok
    assert result.reason == "guest1 drink asked again before the mumbled answer"


def second_bag_request(events: list[Record]) -> None:
    following = at(events, "person", person="guest2", state="following")
    events[following:following] = [
        line(48.5, "Is your bag on my hand now?", "guest2", ["bag"]),
        event("bag_request", 48.5, guest="guest2", count=2, hang=True, early_answer=True),
        answer(49.5, "guest2", 48.5, ["bag"]),
    ]


def test_questions_allow_a_second_bag_request_in_the_early_answer_variant():
    events, tools = passing_run()
    events[0]["scenario"]["bag"]["early_answer"] = True
    second_bag_request(events)

    assert result_of("questions", events, tools).ok


def test_questions_fail_a_second_bag_request_without_the_early_answer():
    events, tools = passing_run()
    second_bag_request(events)

    result = result_of("questions", events, tools)

    assert (result.ok, result.reason) == (False, "guest2 bag asked 2 times")


# -- arm_motion ---------------------------------------------------------------------------


def test_arm_motion_fails_an_arm_call_with_no_line_since_the_last_one():
    events, tools = passing_run()
    del tools[tool_at(tools, "speak", 47.0)]

    result = result_of("arm_motion", events, tools)

    assert (result.ok, result.reason) == (
        False,
        "set_gripper at t=49 s with no line since the last arm call",
    )


@pytest.mark.parametrize(("outcome", "ok"), [("error", False), ("invalid", True)])
def test_arm_motion_counts_a_failed_arm_call_as_the_last_one(outcome, ok):
    events, tools = passing_run()
    # A move that failed may still have moved the arm; one that never ran did not.
    failed = call(45.2, "move_arm_to_pose", {"arm": "left", "pose": "receive"}, outcome)
    tools.insert(tool_at(tools, "move_arm_to_pose", 45.5), failed)

    result = result_of("arm_motion", events, tools)

    assert result.ok is ok
    if not ok:
        assert result.reason == "move_arm_to_pose at t=45.5 s with no line since the last arm call"


def test_arm_motion_counts_wait_for_person_as_the_line_before_an_arm_call():
    events, tools = passing_run()
    tools[tool_at(tools, "speak", 47.0)]["tool"] = "wait_for_person"

    assert result_of("arm_motion", events, tools).ok


def test_arm_motion_fails_a_close_before_the_bag_answer():
    events, tools = passing_run()
    close = tools.pop(tool_at(tools, "set_gripper", 49.0))
    close["t"] = 47.8
    tools.insert(tool_at(tools, "speak", 47.0) + 1, close)

    result = result_of("arm_motion", events, tools)

    assert (result.ok, result.reason) == (
        False,
        "left close at t=47.8 s before the answer to the bag request at t=47 s",
    )


def test_arm_motion_fails_a_close_with_no_bag_request_before_it():
    events, tools = passing_run()
    events[at(events, "bag_request")]["t"] = 49.5

    assert not result_of("arm_motion", events, tools).ok


# -- bag ----------------------------------------------------------------------------------


def test_bag_passes_when_the_hang_was_already_held():
    events, tools = passing_run()
    events[at(events, "bag_hung")]["already_held"] = True

    assert result_of("bag", events, tools).ok


def test_bag_fails_when_the_last_close_holds_nothing():
    events, tools = passing_run()
    tools[tool_at(tools, "set_gripper", 49.0)] = gripper(49.0, "close", holding=False)

    result = result_of("bag", events, tools)

    assert (result.ok, result.reason) == (
        False,
        "the last left close, at t=49 s, was ok holding False",
    )


def test_bag_fails_when_the_left_hand_opens_before_the_guest_is_guided():
    events, tools = passing_run()
    tools.insert(tool_at(tools, "set_gripper", 49.0) + 1, gripper(49.5, "open"))

    result = result_of("bag", events, tools)

    assert (result.ok, result.reason) == (False, "the left hand opened at t=49.5 s")


def test_bag_fails_without_a_hang_before_the_close():
    events, tools = passing_run()
    del events[at(events, "bag_hung")]

    assert not result_of("bag", events, tools).ok


def test_bag_judges_the_last_close_after_an_empty_one():
    events, tools = passing_run()
    tools.insert(tool_at(tools, "set_gripper", 49.0), gripper(48.2, "close", holding=False))

    assert result_of("bag", events, tools).ok


# -- drives -------------------------------------------------------------------------------


def test_drives_fail_on_a_drive_refused_for_an_arm_out():
    events, tools = passing_run()
    refusal = f"ToolException: Not driving: the left arm is {ARMS_OUT}. Move it with ..."
    tools.insert(
        tool_at(tools, "navigate_to_location", 51.0),
        call(50.5, "navigate_to_location", {"location": "living_room"}, "error", refusal),
    )

    result = result_of("drives", events, tools)

    assert (result.ok, result.reason) == (False, "1 of 3 drives refused for an arm out: t=50.5 s")


def test_drives_pass_a_drive_that_failed_for_another_reason():
    events, tools = passing_run()
    tools[tool_at(tools, "navigate_to_location", 20.0)].update(
        outcome="error", result="ToolException: Navigation failed: goal aborted"
    )

    assert result_of("drives", events, tools).ok


# -- cmd_vel ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (cmd_vel(22.0, angular_z=0.2, nav_active=True), "a turn while Nav2 drives at t=22 s"),
        (cmd_vel(22.0, angular_z=0.2, arms_travel_safe=False), "a turn with an arm out at t=22 s"),
        (
            cmd_vel(22.0, angular_z=0.2, arms_travel_safe=None),
            "a turn with the arms unknown at t=22 s",
        ),
        (cmd_vel(22.0, linear_x=0.1), "linear [0.1, 0.0, 0.0] at t=22 s"),
    ],
    ids=["turn-under-nav2", "turn-arm-out", "turn-arms-unknown", "linear"],
)
def test_cmd_vel_fails_linear_velocity_and_turns_while_nav2_drives_or_an_arm_is_out(
    message, reason
):
    events, tools = passing_run()
    events.append(message)

    result = result_of("cmd_vel", events, tools)

    assert (result.ok, result.reason) == (False, f"1 of 3 messages bad: {reason}")


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("linear", "linear [None, 0.0, 0.0] at t=22 s"),
        ("angular", "a turn while Nav2 drives at t=22 s"),
    ],
)
def test_cmd_vel_counts_a_velocity_that_was_not_a_number_as_moving(field, reason):
    events, tools = passing_run()
    # The event log writes a NaN or an infinity as null.
    message = cmd_vel(22.0, nav_active=True)
    message[field] = [None, 0.0, 0.0] if field == "linear" else [0.0, 0.0, None]
    events.append(message)

    result = result_of("cmd_vel", events, tools)

    assert (result.ok, result.reason) == (False, f"1 of 3 messages bad: {reason}")


# -- valid --------------------------------------------------------------------------------


def test_valid_fails_on_a_clock_jump():
    events, tools = passing_run()
    events.append(event("clock_jump", 5.0, **{"from": 90.0, "to": 5.0, "pending": 0}))

    result = result_of("valid", events, tools)

    assert (result.ok, result.reason) == (False, "the clock went back from 90.0 to 5.0 s")


# -- time ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("done_t", "ok"), [(SEEDED_T + 360.0, True), (SEEDED_T + 361.0, False)])
def test_time_allows_360_s_from_the_seeding_to_the_last_item_completed(done_t, ok):
    events, tools = passing_run()
    events[at(events, "todos", t=DONE_T)]["t"] = done_t

    result = result_of("time", events, tools)

    assert result.ok is ok
    assert result.reason == f"{done_t - SEEDED_T:.1f} s, {'within' if ok else 'over'} 360 s"


def test_time_fails_a_run_whose_items_are_never_all_completed():
    events, tools = passing_run()
    events[at(events, "todos", t=DONE_T)]["items"][-1]["status"] = "in_progress"

    result = result_of("time", events, tools)

    assert (result.ok, result.reason) == (False, "the todo list was never all completed")


@pytest.mark.parametrize(
    ("finished", "ok", "reason"),
    [
        (True, True, f"{DONE_T - SEEDED_T:.1f} s, within 360 s"),
        (False, False, "the todo list was never all completed"),
    ],
)
def test_time_starts_at_the_first_open_list_after_an_earlier_tasks_finished_one(
    finished, ok, reason
):
    events, tools = passing_run()
    # /walkie/todos is latched: the scenario first gets the list of a task
    # the agent finished before this run.
    events.insert(at(events, "todos", t=SEEDED_T), todos(1.0, "completed"))
    if not finished:
        events[at(events, "todos", t=DONE_T)]["items"][-1]["status"] = "in_progress"

    result = result_of("time", events, tools)

    assert (result.ok, result.reason) == (ok, reason)


# -- people -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "track_id", "who"),
    [
        ("look_at", ONLOOKER_TRACKS[0], "onlooker1"),
        ("look_at", BYSTANDER_TRACK, "the bystander"),
        ("remember_person", ONLOOKER_TRACKS[1], "onlooker1"),
        ("remember_person", BYSTANDER_TRACK, "the bystander"),
    ],
)
def test_people_fails_a_call_on_an_onlookers_or_the_bystanders_track(tool, track_id, who):
    events, tools = passing_run()
    tools.append(
        call(34.0, tool, {"person": ONLOOKER_HANDLE}, people=[ref(ONLOOKER_HANDLE, track_id)])
    )

    result = result_of("people", events, tools)

    assert (result.ok, result.reason) == (False, f"{tool} at t=34 s named {ONLOOKER_HANDLE}, {who}")


@pytest.mark.parametrize("tool", ["speak", "wait_for_person"])
@pytest.mark.parametrize(
    ("track_id", "who"), [(ONLOOKER_TRACKS[1], "onlooker1"), (BYSTANDER_TRACK, "the bystander")]
)
def test_people_fails_a_question_while_the_gaze_follows_an_onlooker_or_the_bystander(
    tool, track_id, who
):
    events, tools = passing_run()
    question = speak(TALK_T["guest2"], "Hello, what is your name?", track_id)
    question["tool"] = tool
    tools[tool_at(tools, "speak", TALK_T["guest2"])] = question

    result = result_of("people", events, tools)

    assert (result.ok, result.reason) == (False, f"{tool} at t=40 s while looking at {who}")


def test_people_passes_a_line_while_the_gaze_on_an_onlooker_is_stopped():
    events, tools = passing_run()
    question = speak(TALK_T["guest2"], "Hello, what is your name?", ONLOOKER_TRACKS[1])
    question["gaze"]["running"] = False
    tools[tool_at(tools, "speak", TALK_T["guest2"])] = question

    assert result_of("people", events, tools).ok


# -- The CLI ------------------------------------------------------------------------------


def write_logs(directory: Path, events: list[Record], tools: list[Record]) -> list[str]:
    paths = []
    for name, records in (("events.jsonl", events), ("tools.jsonl", tools)):
        path = directory / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        paths.append(str(path))
    return paths


def test_the_cli_prints_a_line_per_criterion_and_passes_with_exit_code_0(tmp_path, capsys):
    events, tools = passing_run()

    code = hri_check.main(write_logs(tmp_path, events, tools))

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert len(lines) == 14 and lines[-1] == "PASS"
    assert lines[0] == "PASS memory: guest1 Sam milk, guest2 Ana cola"
    assert all(line.startswith("PASS ") for line in lines[:-1])


def test_the_cli_fails_with_exit_code_1_when_a_criterion_fails(tmp_path, capsys):
    events, tools = passing_run()
    events.insert(at(events, "person", person="guest1", state="following"), repeat("guest1", 16.0))

    code = hri_check.main(write_logs(tmp_path, events, tools))

    lines = capsys.readouterr().out.splitlines()
    assert code == 1
    assert "FAIL questions: guest1 drink asked 2 times" in lines
    assert lines[-1] == "FAIL"


def test_the_cli_calls_a_run_with_a_clock_jump_invalid_with_exit_code_2(tmp_path, capsys):
    events, tools = passing_run()
    events.append(event("clock_jump", 5.0, **{"from": 90.0, "to": 5.0, "pending": 0}))

    code = hri_check.main(write_logs(tmp_path, events, tools))

    lines = capsys.readouterr().out.splitlines()
    assert code == 2
    assert "FAIL valid: the clock went back from 90.0 to 5.0 s" in lines
    assert lines[-1] == "INVALID"


@pytest.mark.parametrize(
    ("change", "reason"),
    [("no-start", "the event log has no start record"), ("no-tools", "the tool log is empty")],
)
def test_the_cli_calls_a_run_without_a_start_or_a_tool_log_invalid(
    tmp_path, capsys, change, reason
):
    events, tools = passing_run()
    if change == "no-start":
        del events[0]
    else:
        tools = []

    code = hri_check.main(write_logs(tmp_path, events, tools))

    assert code == 2
    assert capsys.readouterr().out.splitlines() == [f"INVALID {reason}", "INVALID"]


@pytest.mark.parametrize("log", ["events", "tools"])
def test_the_cli_calls_a_run_with_a_record_without_a_time_invalid(tmp_path, capsys, log):
    # EventLog writes a time that was not a finite number as null, which no
    # criterion can compare.
    events, tools = passing_run()
    (events if log == "events" else tools)[-1]["t"] = None

    code = hri_check.main(write_logs(tmp_path, events, tools))

    assert code == 2
    assert capsys.readouterr().out.splitlines() == ["INVALID 1 record has no time", "INVALID"]


def test_the_cli_drops_a_torn_last_line_of_a_log_still_being_written(tmp_path, capsys):
    events, tools = passing_run()
    paths = write_logs(tmp_path, events, tools)
    with open(paths[1], "a", encoding="utf-8") as f:
        f.write('{"type": "call", "t": 9')

    assert hri_check.main(paths) == 0


def test_the_check_imports_no_ros():
    # A fresh interpreter, since this suite has long imported rclpy. -I drops
    # PYTHONPATH, where the ROS workspaces' packages are, and the import must
    # not reach the ones in the environment either.
    src = Path(hri_check.__file__).parents[2]
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(src)!r}); "
        "import walkie_agent.sim.hri_check; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'rclpy', 'rai', 'nav2_msgs', "
        "'langchain_core'}))"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
