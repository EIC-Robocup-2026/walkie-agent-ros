"""Fixtures for Mahoraga's own suite, which never loads tests/conftest.py or ROS."""

import pytest

from mahoraga.driver import ScriptRunner, allow_list
from mahoraga.sandbox import open_pool
from walkie_agent.atomics import GPSR
from walkie_agent.capabilities.fake import FakeRobot


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep every shelf, gate result and log of a test out of the real home, and out of the network."""
    home = tmp_path / "mahoraga-home"
    monkeypatch.setenv("MAHORAGA_HOME", str(home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home


@pytest.fixture(scope="session")
def pool():
    with open_pool(max_processes=2) as shared:
        yield shared


@pytest.fixture
def robot():
    return FakeRobot(seed=1)


@pytest.fixture
def fns(robot):
    return allow_list(GPSR, robot.ctx)


@pytest.fixture
def runner(pool):
    return ScriptRunner(pool)
