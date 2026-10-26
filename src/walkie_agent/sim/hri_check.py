"""The HRI pass check: judge a finished run from its two logs.

    python -m walkie_agent.sim.hri_check EVENTS.jsonl TOOLS.jsonl

EVENTS is the scenario's event log (hri_scenario's module docstring) and
TOOLS the agent's tool-call record (walkie_agent.tool_log). The criteria are
hri.md's, in its order ("The pass check comes with the template"), the last
for the people at the door who are not guests. Each prints one line, such as
"PASS gaze guest1: 87% within 10 deg, ratio 0.81", and a last line says
PASS, FAIL or INVALID.

Exit code 0 when every criterion passes, 1 when any fails, and 2 when the
run is not valid: its event log has a clock_jump or no start record, its
tool log is empty, or a record in either has no time. Without a start
record, a tool record or every record's time the criteria cannot be judged,
so only the reasons are printed.

The check is pure, with no ROS. It does not import hri_scenario, which
imports rclpy; the start record names the guests, the bag owner and the
onlookers instead, so the check need not repeat the scenario's rules. Both
logs stamp `t` on the simulation's clock, so their times compare directly.
"""

import argparse
import math
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from walkie_agent.arm.state import ARMS_OUT
from walkie_agent.jsonl import read_jsonl

# The gaze criterion: this share of the samples within this error, and a
# mean error at most this share of the baseline's, which a robot that never
# turns cannot reach. A baseline below 1 deg skips the ratio (hri.md).
GAZE_LIMIT_DEG = 10.0
GAZE_SHARE = 0.8
GAZE_RATIO = 0.9
MIN_BASELINE_DEG = 1.0
# An introduction faces the listener within the gaze criterion's limit.
INTRODUCTION_LIMIT_DEG = 10.0
# From the template's seeding to the last item completed, in sim seconds.
# The rulebook's scoresheet sets 7 minutes for the HRI task.
TIME_LIMIT_S = 420.0
# An answer's asked_t is its line's t, both rounded to the millisecond.
ASKED_T_TOLERANCE_S = 0.001

ARM_TOOLS = ("move_arm_to_pose", "set_gripper")
LINE_TOOLS = ("speak", "wait_for_person")
DRIVE_TOOLS = ("navigate_to_location", "move_relative")
HANDLE_TOOLS = ("look_at", "remember_person")
# The arms a set_gripper call moves the left gripper with.
LEFT_ARMS = ("left", "both")
# The topics a question asks that the questions criterion counts; "other"
# is an extra_question of its own.
COUNTED_ASKS = ("name", "drink", "bag")
# A call that ran: "invalid" and "not_run" never reached the robot.
RAN = ("ok", "error")


@dataclass(frozen=True)
class Result:
    """One criterion's verdict, printed as "PASS name: reason"."""

    name: str
    ok: bool
    reason: str

    def line(self) -> str:
        return f"{'PASS' if self.ok else 'FAIL'} {self.name}: {self.reason}"


class Run:
    """A run's event log and tool log, indexed for the criteria."""

    def __init__(self, events: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> None:
        self.events = list(events)
        self.tools = list(tools)
        self.start = next((r for r in self.events if r.get("type") == "start"), None)
        start = self.start or {}
        self.guests: dict[str, dict[str, str]] = dict(start.get("guests") or {})
        self.bag_owner: str | None = start.get("bag_owner")
        self.scenario: dict[str, Any] = start.get("scenario") or {}
        self.calls = [r for r in self.tools if r.get("type") == "call"]
        # Both logs stamp every record; EventLog writes a time that was not
        # a finite number as null.
        self.untimed = sum(
            1 for r in (*self.events, *self.tools) if not isinstance(r.get("t"), (int, float))
        )
        # A new task clears the people memory, so only the calls since the
        # last new_task can still be in it.
        last_task = max(
            (i for i, r in enumerate(self.tools) if r.get("type") == "new_task"), default=-1
        )
        self.task_calls = [r for r in self.tools[last_task + 1 :] if r.get("type") == "call"]
        # Every track each guest was seen on: at the door and seated.
        self.guest_tracks: dict[str, set[int]] = {guest: set() for guest in self.guests}
        # The tracks that are never a guest's, with whose they are.
        self.blocked_tracks: dict[int | None, str] = {}
        for record in self.events:
            type_, track = record.get("type"), record.get("track_id")
            if track is None:
                continue
            if type_ == "person" and record.get("person") in self.guest_tracks:
                self.guest_tracks[record["person"]].add(track)
            elif type_ == "onlooker":
                self.blocked_tracks[track] = record.get("onlooker") or "an onlooker"
            elif type_ == "bystander":
                self.blocked_tracks[track] = "the bystander"

    def of_type(self, type_: str) -> list[dict[str, Any]]:
        return [r for r in self.events if r.get("type") == type_]

    def person_index(self, person: str, state: str) -> int | None:
        """The index in the event log of `person`'s first record in `state`, or None."""
        return next(
            (
                i
                for i, r in enumerate(self.events)
                if r.get("type") == "person" and r.get("person") == person and r["state"] == state
            ),
            None,
        )

    def invalid_reasons(self) -> list[str]:
        reasons = []
        if self.start is None:
            reasons.append("the event log has no start record")
        if not self.tools:
            reasons.append("the tool log is empty")
        if self.untimed:
            records = "1 record has" if self.untimed == 1 else f"{self.untimed} records have"
            reasons.append(f"{records} no time")
        reasons +=[_clock_jump_reason(r) for r in self.of_type("clock_jump")]
        return reasons


# -- The criteria, in hri.md's order ------------------------------------------------------


def check_memory(run: Run) -> list[Result]:
    """Each guest's name and drink are in the people memory, on their own track."""
    problems, found = [], []
    for guest, want in run.guests.items():
        tracks = run.guest_tracks[guest]
        calls = [
            c
            for c in run.task_calls
            if c.get("tool") == "remember_person"
            and c.get("outcome") == "ok"
            and any(p.get("track_id") in tracks for p in _people(c, "args"))
        ]
        if any(_remembers(c.get("args") or {}, want) for c in calls):
            found.append(f"{guest} {want['name']} {want['drink']}")
        elif not calls:
            problems.append(f"no ok remember_person on {guest}'s track")
        else:
            args = calls[-1].get("args") or {}
            problems.append(
                f"{guest} remembered as name {args.get('name')!r}, facts {args.get('facts')!r}; "
                f"wanted {want['name']} and {want['drink']}"
            )
    if problems:
        return [Result("memory", False, "; ".join(problems))]
    return [Result("memory", True, ", ".join(found) or "no guests")]


def check_gaze(run: Run) -> list[Result]:
    """The heading follows each guest while they talk at the door."""
    return [_gaze(run, guest) for guest in run.guests]


def check_offers(run: Run) -> list[Result]:
    """Every seat offer faces a free place, and each guest gets a seat."""
    problems = []
    for r in run.of_type("bad_offer"):
        taken = f", {r['place']} taken by {r.get('taken_by')}" if r.get("place") else ""
        problems.append(
            f"bad offer to {r.get('guest')} at t={r['t']:g} s ({r.get('reason')}{taken})"
        )
    seated = {r.get("guest"): r.get("place") for r in run.of_type("seat_taken")}
    problems += [f"{guest} never took a seat" for guest in run.guests if guest not in seated]
    if problems:
        return [Result("offers", False, "; ".join(problems))]
    return [Result("offers", True, ", ".join(f"{g} at {seated[g]}" for g in run.guests))]


def check_introductions(run: Run) -> list[Result]:
    """Each guest is introduced to, with the robot facing them."""
    seated = [run.person_index(guest, "seated") for guest in run.guests]
    if not run.guests or None in seated:
        missing = [g for g, i in zip(run.guests, seated) if i is None]
        return [Result("introductions", False, f"{', '.join(missing) or 'nobody'} never seated")]
    after = max(i for i in seated if i is not None)
    drinks = {
        guest: re.compile(rf"\b{re.escape(want['drink'].strip().casefold())}\b")
        for guest, want in run.guests.items()
    }
    # The smallest bearing each guest was faced at as a listener.
    faced: dict[str, float | None] = {guest: None for guest in run.guests}
    for line in run.events[after + 1 :]:
        if line.get("type") != "robot_line" or line.get("guest") is not None:
            continue
        text = str(line.get("text", "")).casefold()
        named = [guest for guest, drink in drinks.items() if drink.search(text)]
        if len(named) != 1:
            continue
        bearings = line.get("bearings_deg") or {}
        for listener in run.guests:
            bearing = bearings.get(listener)
            if listener == named[0] or bearing is None:
                continue
            best = faced[listener]
            faced[listener] = abs(bearing) if best is None else min(best, abs(bearing))
    problems = []
    for guest, best in faced.items():
        if best is None:
            problems.append(f"no introduction to {guest}")
        elif best > INTRODUCTION_LIMIT_DEG:
            problems.append(
                f"no introduction faced {guest}: the nearest was {best:.1f} deg off, "
                f"over {INTRODUCTION_LIMIT_DEG:g}"
            )
    if problems:
        return [Result("introductions", False, "; ".join(problems))]
    return [
        Result(
            "introductions",
            True,
            ", ".join(f"{g} faced within {b:.1f} deg" for g, b in faced.items()),
        )
    ]


def check_questions(run: Run) -> list[Result]:
    """Only the essential questions, plus the repeats the scenario's variants allow."""
    problems = [
        f"extra question to {r.get('guest')}: {r.get('text')!r}"
        for r in run.of_type("extra_question")
    ]
    mumble = run.scenario.get("answers", {}).get("mumble_once")
    early_answer = bool(run.scenario.get("bag", {}).get("early_answer"))
    mumbled_t = {
        (r.get("guest"), topic): r["t"]
        for r in run.of_type("answer")
        if r.get("mumbled")
        for topic in r.get("asks") or ()
    }
    lines = run.of_type("robot_line")
    for guest in run.guests:
        for topic in COUNTED_ASKS:
            asked = [
                r["t"] for r in lines if r.get("guest") == guest and topic in (r.get("asks") or ())
            ]
            allowed = 1
            if topic == "bag":
                allowed = (1 + early_answer) if guest == run.bag_owner else 0
            if f"{guest}.{topic}" == mumble and (guest, topic) in mumbled_t:
                # The repeat counts only once the mumble was heard.
                if sum(t <= mumbled_t[guest, topic] for t in asked) > allowed:
                    problems.append(f"{guest} {topic} asked again before the mumbled answer")
                allowed += 1
            if len(asked) > allowed:
                problems.append(f"{guest} {topic} asked {len(asked)} times")
    if problems:
        return [Result("questions", False, "; ".join(problems))]
    return [Result("questions", True, f"{len(lines)} lines, no question over its allowance")]


def check_arm_motion(run: Run) -> list[Result]:
    """A line before every arm motion, and the bag hand closed only after the guest's answer."""
    problems = []
    spoke = False
    arm_calls = 0
    for call in run.calls:
        tool, outcome = call.get("tool"), call.get("outcome")
        if tool in LINE_TOOLS and outcome == "ok":
            spoke = True
        elif tool in ARM_TOOLS and outcome in RAN:
            # A call that failed may still have moved the arm, so it needs
            # its own line too, and the next call one after it.
            if outcome == "ok":
                arm_calls += 1
                if not spoke:
                    problems.append(
                        f"{tool} at t={call['t']:g} s with no line since the last arm call"
                    )
            spoke = False
    requests = run.of_type("bag_request")
    answers = [r for r in run.of_type("answer") if "bag" in (r.get("asks") or ())]
    for close in _left_grips(run, ("close",)):
        t = close["t"]
        before = [r for r in requests if r["t"] <= t]
        if not before:
            problems.append(f"left close at t={t:g} s with no bag request before it")
            continue
        request = before[-1]
        if not any(
            abs(a.get("asked_t", math.inf) - request["t"]) <= ASKED_T_TOLERANCE_S and a["t"] <= t
            for a in answers
        ):
            problems.append(
                f"left close at t={t:g} s before the answer to the bag request "
                f"at t={request['t']:g} s"
            )
    if problems:
        return [Result("arm_motion", False, "; ".join(problems))]
    return [Result("arm_motion", True, f"a line before each of {arm_calls} arm calls")]


def check_bag(run: Run) -> list[Result]:
    """The left hand holds the bag before its owner is guided."""
    owner = run.bag_owner
    if owner is None:
        return [Result("bag", True, "no guest brings a bag")]
    follows = run.person_index(owner, "following")
    if follows is None:
        return [Result("bag", False, f"{owner} was never guided")]
    guided_t = run.events[follows]["t"]
    grips = [c for c in _left_grips(run, ("open", "half", "close")) if c["t"] < guided_t]
    closes = [i for i, c in enumerate(grips) if (c.get("args") or {}).get("state") == "close"]
    if not closes:
        return [Result("bag", False, f"no left close before {owner} was guided")]
    close = grips[closes[-1]]
    holding = ((close.get("out") or {}).get("holding") or {}).get("left")
    if close.get("outcome") != "ok" or holding is not True:
        return [
            Result(
                "bag",
                False,
                f"the last left close, at t={close['t']:g} s, was {close.get('outcome')} "
                f"holding {holding}",
            )
        ]
    if closes[-1] < len(grips) - 1:
        opened = grips[closes[-1] + 1]
        return [Result("bag", False, f"the left hand opened at t={opened['t']:g} s")]
    if not any(r["t"] <= close["t"] for r in run.of_type("bag_hung")):
        reason = f"the bag was not hung before the close at t={close['t']:g} s"
        return [Result("bag", False, reason)]
    return [Result("bag", True, f"held in the left hand from t={close['t']:g} s")]


def check_drives(run: Run) -> list[Result]:
    """No drive was refused because an arm was out."""
    refused = [
        c
        for c in run.calls
        if c.get("tool") in DRIVE_TOOLS
        and c.get("outcome") == "error"
        and ARMS_OUT in str(c.get("result", ""))
    ]
    drives = sum(c.get("tool") in DRIVE_TOOLS for c in run.calls)
    if refused:
        times = ", ".join(f"t={c['t']:g} s" for c in refused)
        reason = f"{len(refused)} of {drives} drives refused for an arm out: {times}"
        return [Result("drives", False, reason)]
    return [Result("drives", True, f"none of {drives} drives refused for an arm out")]


def check_cmd_vel(run: Run) -> list[Result]:
    """No linear velocity, and no turn while Nav2 drives or an arm is out."""
    records = run.of_type("cmd_vel")
    problems = []
    for r in records:
        # The event log writes a NaN or an infinity as null, so null counts
        # as moving: `None != 0` holds.
        if any(v != 0 for v in r.get("linear") or ()):
            problems.append(f"linear {r['linear']} at t={r['t']:g} s")
        elif any(v != 0 for v in r.get("angular") or ()):
            if r.get("nav_active") is True:
                problems.append(f"a turn while Nav2 drives at t={r['t']:g} s")
            elif r.get("arms_travel_safe") is None:
                problems.append(f"a turn with the arms unknown at t={r['t']:g} s")
            elif r.get("arms_travel_safe") is not True:
                problems.append(f"a turn with an arm out at t={r['t']:g} s")
    if problems:
        shown = "; ".join(problems[:3])
        more = f" and {len(problems) - 3} more" if len(problems) > 3 else ""
        reason = f"{len(problems)} of {len(records)} messages bad: {shown}{more}"
        return [Result("cmd_vel", False, reason)]
    return [Result("cmd_vel", True, f"{len(records)} messages")]


def check_valid(run: Run) -> list[Result]:
    """The simulation never restarted under the run."""
    jumps = run.of_type("clock_jump")
    if jumps:
        return [Result("valid", False, "; ".join(_clock_jump_reason(r) for r in jumps))]
    return [Result("valid", True, "no clock_jump")]


def check_time(run: Run) -> list[Result]:
    """From the template's seeding to the last item completed, within 7 minutes of sim time."""
    lists = [r for r in run.of_type("todos") if r.get("items")]
    finished = [all(item.get("status") == "completed" for item in r["items"]) for r in lists]
    # /walkie/todos is latched, so the first list may be an earlier task's,
    # already finished. The run starts at the first list with an open item.
    opened = next((i for i, f in enumerate(finished) if not f), None)
    if opened is None:
        return [Result("time", False, "the todo list was never seeded")]
    begin = lists[opened]["t"]
    done = next((r for r, f in zip(lists[opened:], finished[opened:]) if f), None)
    if done is None:
        return [Result("time", False, "the todo list was never all completed")]
    took = done["t"] - begin
    verdict = "within" if took <= TIME_LIMIT_S else "over"
    return [Result("time", took <= TIME_LIMIT_S, f"{took:.1f} s, {verdict} {TIME_LIMIT_S:g} s")]


def check_people(run: Run) -> list[Result]:
    """The robot never treats an onlooker or the bystander as a guest."""
    problems = []
    for call in run.calls:
        tool, t = call.get("tool"), call.get("t", 0.0)
        if tool in HANDLE_TOOLS:
            for person in _people(call, "args"):
                who = run.blocked_tracks.get(person.get("track_id"))
                if who is not None:
                    problems.append(f"{tool} at t={t:g} s named {person.get('handle')}, {who}")
        elif tool in LINE_TOOLS:
            gaze = call.get("gaze") or {}
            who = run.blocked_tracks.get(gaze.get("track_id"))
            if gaze.get("running") is True and who is not None:
                problems.append(f"{tool} at t={t:g} s while looking at {who}")
    if problems:
        return [Result("people", False, "; ".join(problems))]
    return [Result("people", True, f"no call on the {len(run.blocked_tracks)} other tracks")]


CRITERIA: tuple[Callable[[Run], list[Result]], ...] = (
    check_memory,
    check_gaze,
    check_offers,
    check_introductions,
    check_questions,
    check_arm_motion,
    check_bag,
    check_drives,
    check_cmd_vel,
    check_valid,
    check_time,
    check_people,
)


def check(run: Run) -> list[Result]:
    """Every criterion's results, in CRITERIA's order."""
    return [result for criterion in CRITERIA for result in criterion(run)]


# -- Helpers --------------------------------------------------------------------------------


def _people(call: dict[str, Any], where: str) -> list[dict[str, Any]]:
    return [p for p in call.get("people") or () if p.get("in") == where]


def _norm(value: Any) -> str:
    return str(value).strip().casefold()


def _remembers(args: dict[str, Any], want: dict[str, str]) -> bool:
    facts = args.get("facts") or {}
    if not isinstance(facts, dict) or _norm(args.get("name", "")) != _norm(want["name"]):
        return False
    drink = _norm(want["drink"])
    return any("drink" in _norm(key) and drink in _norm(value) for key, value in facts.items())


def _left_grips(run: Run, states: tuple[str, ...]) -> list[dict[str, Any]]:
    """The set_gripper calls that ran on the left gripper with a state in `states`."""
    return [
        c
        for c in run.calls
        if c.get("tool") == "set_gripper"
        and c.get("outcome") in RAN
        and (c.get("args") or {}).get("arm") in LEFT_ARMS
        and (c.get("args") or {}).get("state") in states
    ]


def _wrap_deg(angle: float) -> float:
    """`angle` in degrees, wrapped into (-180, 180]."""
    wrapped = math.fmod(angle, 360.0)
    if wrapped <= -180.0:
        wrapped += 360.0
    elif wrapped > 180.0:
        wrapped -= 360.0
    return wrapped


def _gaze(run: Run, guest: str) -> Result:
    name = f"gaze {guest}"
    # A conversation runs from the guest's talking record to their following
    # record, or the end of the log.
    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for record in run.events:
        type_ = record.get("type")
        if type_ == "person" and record.get("person") == guest:
            if record.get("state") == "talking" and current is None:
                current = []
                windows.append(current)
            elif record.get("state") == "following":
                current = None
        elif type_ == "state" and current is not None:
            talking = record.get("talking") or {}
            # The gaze loop stops by design while Nav2 drives and while an arm
            # is out (ADR-0051), so those samples say nothing of it.
            if (
                record.get("pose") is not None
                and talking.get("guest") == guest
                and talking.get("bearing_deg") is not None
                and record.get("nav_active") is not True
                and record.get("arms_travel_safe") is True
            ):
                current.append(record)
    if not windows:
        return Result(name, False, "no conversation")
    errors: list[float] = []
    baseline: list[float] = []
    for samples in windows:
        if not samples:
            continue
        yaw0 = math.degrees(samples[0]["pose"]["yaw"])
        for sample in samples:
            pose, talking = sample["pose"], sample["talking"]
            errors.append(abs(talking["bearing_deg"]))
            direction = math.degrees(
                math.atan2(talking["y"] - pose["y"], talking["x"] - pose["x"])
            )
            baseline.append(abs(_wrap_deg(direction - yaw0)))
    if not errors:
        return Result(name, False, "no samples")
    share = sum(e <= GAZE_LIMIT_DEG for e in errors) / len(errors)
    mean_error = sum(errors) / len(errors)
    mean_baseline = sum(baseline) / len(baseline)
    reason = f"{share:.0%} within {GAZE_LIMIT_DEG:g} deg"
    ok = share >= GAZE_SHARE
    if mean_baseline < MIN_BASELINE_DEG:
        reason += f", baseline {mean_baseline:.2f} deg so no ratio"
    else:
        ratio = mean_error / mean_baseline
        reason += f", ratio {ratio:.2f}"
        ok = ok and ratio <= GAZE_RATIO
    return Result(name, ok, f"{reason} ({len(errors)} samples)")


def _clock_jump_reason(record: dict[str, Any]) -> str:
    return f"the clock went back from {record.get('from')} to {record.get('to')} s"


# -- Entry point ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m walkie_agent.sim.hri_check",
        description="Check an HRI run against the pass criteria of docs/architecture/hri.md.",
    )
    parser.add_argument("events", type=Path, help="The scenario's event log (JSON lines).")
    parser.add_argument("tools", type=Path, help="The agent's tool log (JSON lines).")
    args = parser.parse_args(argv)
    try:
        run = Run(read_jsonl(args.events), read_jsonl(args.tools))
    except (OSError, ValueError) as e:
        parser.error(str(e))
    reasons = run.invalid_reasons()
    results: list[Result] = []
    if run.start is not None and run.tools and not run.untimed:
        results = check(run)
        for result in results:
            print(result.line())
    else:
        for reason in reasons:
            print(f"INVALID {reason}")
    if reasons:
        print("INVALID")
        return 2
    if not all(result.ok for result in results):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
