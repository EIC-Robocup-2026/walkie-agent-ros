"""The oracle's scripts carry out every command, through the driver an agent's script uses."""

import pytest

from mahoraga.checkers import check
from mahoraga.commands import commands
from mahoraga.driver import allow_list
from mahoraga.oracle import oracle_script
from mahoraga.worlds import build_world, world_seed
from walkie_agent.atomics import GPSR
from walkie_agent.capabilities.fake import FakeRobot


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
