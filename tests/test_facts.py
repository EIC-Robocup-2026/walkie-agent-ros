"""ctx.facts: SystemFacts and its [team] table, the fake world's facts, and get_facts over both."""

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from walkie_agent.adapter import as_tool
from walkie_agent.capabilities import CancelToken, Ctx, Team
from walkie_agent.capabilities.facts import SystemFacts, facts_from_config
from walkie_agent.capabilities.fake import FakeRobot, chula_pat14_6
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.runtime.model import load_config
from walkie_agent.tools.atomic import get_facts

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"
EIC = Team("EIC", "Thailand", "Chulalongkorn University")
FACTS = {
    "now": "2026-11-05T14:05:00+07:00",
    "team": {"name": "EIC", "country": "Thailand", "affiliation": "Chulalongkorn University"},
}
TEAM_TABLE = {
    "name": "EIC",
    "country": "Thailand",
    "affiliation": "Chulalongkorn University",
    "timezone": "Asia/Bangkok",
}


def world(**more) -> dict:
    return {
        "places": {"lobby": {"x": 0.0, "y": 0.0, "yaw": 0.0, "room": "lobby"}},
        "objects": {},
        "people": {},
        "start": "lobby",
        **more,
    }


def at(*utc: int) -> SystemFacts:
    """SystemFacts in Bangkok whose clock is stopped at a UTC instant."""
    instant = datetime(*utc, tzinfo=UTC).timestamp()
    return SystemFacts(EIC, "Asia/Bangkok", clock=lambda: instant)


def facts_of(facts) -> dict:
    return get_facts(Ctx(cancel=CancelToken(), base=None, speech=None, vision=None, facts=facts))


# SystemFacts


def test_system_facts_read_the_injected_clock_in_the_configured_timezone():
    facts = at(2026, 11, 5, 7, 5)
    now = facts.now()
    assert now == datetime(2026, 11, 5, 7, 5, tzinfo=UTC)
    assert (now.hour, now.minute, now.utcoffset()) == (14, 5, timedelta(hours=7))
    assert facts.timezone == "Asia/Bangkok"
    assert facts.team() == EIC


def test_system_facts_default_to_the_wall_clock():
    before = time.time()
    now = SystemFacts(EIC, "Asia/Bangkok").now().timestamp()
    assert before - 1 <= now <= time.time() + 1


@pytest.mark.parametrize("zone", ["Mars/Base", "asia/bangkok", "Asia", "", "../etc/passwd", " Asia/Bangkok"])
def test_system_facts_refuse_an_unknown_timezone_in_one_line(zone):
    with pytest.raises(ValueError, match="Unknown timezone") as caught:
        SystemFacts(EIC, zone)
    assert "\n" not in str(caught.value)


def test_the_repo_config_has_the_team_table():
    facts = facts_from_config(load_config(REPO_CONFIG))
    assert facts.team() == EIC
    assert facts.timezone == "Asia/Bangkok"


def test_a_team_table_value_is_stripped():
    facts = facts_from_config({"team": {**TEAM_TABLE, "name": " EIC ", "timezone": "Asia/Bangkok "}})
    assert facts.team().name == "EIC"
    assert facts.timezone == "Asia/Bangkok"


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({}, r"needs a \[team\] table"),
        ({"team": "EIC"}, r"needs a \[team\] table"),
        ({"team": {**TEAM_TABLE, "name": ""}}, r"for: name\.$"),
        ({"team": {**TEAM_TABLE, "country": "  "}}, r"for: country\.$"),
        ({"team": {**TEAM_TABLE, "affiliation": 1}}, r"for: affiliation\.$"),
        ({"team": {"name": "EIC"}}, r"for: country, affiliation, timezone\.$"),
        ({"team": {**TEAM_TABLE, "timezone": "Asia/Atlantis"}}, r"^\[team\] timezone: Unknown timezone 'Asia/Atlantis'"),
    ],
    ids=["no-table", "not-a-table", "blank-name", "blank-country", "number", "missing", "unknown-timezone"],
)
def test_a_bad_team_table_is_one_line(config, match):
    with pytest.raises(ValueError, match=match) as caught:
        facts_from_config(config)
    assert "\n" not in str(caught.value)


# get_facts


def test_get_facts_says_the_time_the_dates_and_the_team():
    assert facts_of(at(2026, 11, 5, 7, 5)) == {
        "time": "14:05",
        "today": "Thursday, 5 November 2026",
        "tomorrow": "Friday, 6 November 2026",
        "day_of_week": "Thursday",
        "day_of_month": 5,
        "team_name": "EIC",
        "team_country": "Thailand",
        "team_affiliation": "Chulalongkorn University",
    }


def test_get_facts_crosses_a_month_and_a_year_for_tomorrow():
    facts = facts_of(at(2026, 12, 31, 16, 59))
    assert (facts["time"], facts["day_of_month"]) == ("23:59", 31)
    assert facts["today"] == "Thursday, 31 December 2026"
    assert facts["tomorrow"] == "Friday, 1 January 2027"


def test_get_facts_crosses_february_into_march():
    facts = facts_of(at(2026, 2, 28, 3, 0))
    assert facts["today"] == "Saturday, 28 February 2026"
    assert facts["tomorrow"] == "Sunday, 1 March 2026"


def test_get_facts_uses_the_local_date_not_the_utc_one():
    # 17:30 UTC on 31 December is half past midnight on 1 January in Bangkok.
    facts = facts_of(at(2026, 12, 31, 17, 30))
    assert facts["time"] == "00:30"
    assert facts["today"] == "Friday, 1 January 2027"
    assert facts["day_of_week"] == "Friday"
    assert facts["day_of_month"] == 1


def test_get_facts_over_the_fake_robot_reads_the_world_s_facts():
    robot = FakeRobot(world(facts=FACTS))
    assert get_facts(robot.ctx) == facts_of(at(2026, 11, 5, 7, 5))
    assert [call["verb"] for call in robot.calls] == ["facts.now", "facts.team"]


@pytest.mark.parametrize("seed", [0, 11, 12345])
def test_get_facts_over_a_seeded_arena_agrees_with_system_facts_at_its_clock(seed):
    now = datetime.fromisoformat(chula_pat14_6(seed)["facts"]["now"])
    system = SystemFacts(EIC, "Asia/Bangkok", clock=now.timestamp)
    assert get_facts(FakeRobot(seed=seed).ctx) == facts_of(system)


def test_get_facts_reports_unavailable_without_a_fact_source():
    with pytest.raises(StepError) as caught:
        get_facts(FakeRobot(world()).ctx)
    assert caught.value.code is ErrorCode.UNAVAILABLE


def test_the_get_facts_tool_answers_with_json():
    tool = as_tool(get_facts, FakeRobot(world(facts=FACTS)).ctx)
    assert tool.args == {}
    assert json.loads(tool.invoke({}))["today"] == "Thursday, 5 November 2026"
