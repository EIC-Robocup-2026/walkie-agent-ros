"""The oracle's scripts carry out every command, through the driver an agent's script uses."""

import pytest

from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.evals.mahoraga.checkers import check
from walkie_agent.evals.mahoraga.commands import commands
from walkie_agent.evals.mahoraga.oracle import oracle_script
from walkie_agent.evals.mahoraga.worlds import build_world, world_seed
from walkie_agent.scripting.driver import allow_list
from walkie_agent.tools.sets import GPSR


@pytest.mark.parametrize("category", ["mixed", "objects", "people"])
def test_every_oracle_script_passes_its_checker(runner, category):
    records = commands("trial", 60, category)

    for record in records:
        robot = FakeRobot(build_world(record["spec"], world_seed(record["seed"], 1)))
        start = robot.snapshot()
        result = runner.run(oracle_script(record["spec"]), allow_list(GPSR, robot.ctx))
        ok, why = check(record["spec"], start, robot.snapshot())
        assert result.ok, (record, result.error)
        assert ok, (record, why)


TALK_TOPICS = {
    "the time",
    "what day is today",
    "what day is tomorrow",
    "the day of the week",
    "the day of the month",
    "your teams name",
    "your teams country",
    "your teams affiliation",
    "something about yourself",
}


@pytest.mark.parametrize("split", ["trial", "train", "heldout"])
def test_the_oracle_passes_every_talk_topic(runner, split):
    talks = [record for record in commands(split, 200) if record["spec"]["kind"] == "talk"]
    firsts = {}
    for record in talks:
        firsts.setdefault(record["spec"]["topic"], record)

    assert set(firsts) == TALK_TOPICS
    for record in firsts.values():
        for epoch in (1, 2):
            robot = FakeRobot(build_world(record["spec"], world_seed(record["seed"], epoch)))
            start = robot.snapshot()
            result = runner.run(oracle_script(record["spec"]), allow_list(GPSR, robot.ctx))
            ok, why = check(record["spec"], start, robot.snapshot())
            assert result.ok, (record, result.error)
            assert ok, (record, epoch, why)


def test_the_oracle_refuses_a_talk_topic_it_has_no_line_for():
    spec = {
        "kind": "talk",
        "person": {"room": "kitchen room", "name": "Jane"},
        "place": "kitchen room",
        "topic": "the weather",
        "greet": False,
    }

    with pytest.raises(ValueError, match="No oracle line for the talk topic 'the weather'"):
        oracle_script(spec)


def test_the_scripts_cover_every_kind():
    kinds = {record["spec"]["kind"] for record in commands("trial", 400)}

    assert kinds == {
        "move_object",
        "count_objects",
        "best_object",
        "count_people",
        "person_info",
        "relay_info",
        "talk",
        "escort",
    }
