"""Fixtures for Mahoraga's own suite, which never loads tests/conftest.py or ROS."""

import json

import pytest

from mahoraga import home
from mahoraga.driver import ScriptRunner, allow_list
from mahoraga.gate import _file_hash, code_digest, pair_digests
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
def gated():
    """Record a whole, passing gate for some data, for tests about what runs after the gate."""

    def record(records, epochs=1):
        root = home() / "gate"
        path = root / f"gate-{code_digest()}-fixture.json"
        root.mkdir(parents=True, exist_ok=True)
        # Not named .eval: a report test may read tmp_path, which holds this home, for logs.
        logs = [root / "fixture-1.log", root / "fixture-2.log"]
        for log in logs:
            log.write_text(log.name)
        path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "code_digest": code_digest(),
                    "digests": ["same", "same"],
                    "failures": [],
                    "logs": [str(log) for log in logs],
                    "log_hashes": [_file_hash(log) for log in logs],
                    "pairs": sorted(pair_digests(records, epochs)),
                }
            )
        )
        return path

    return record


@pytest.fixture
def robot():
    return FakeRobot(seed=1)


@pytest.fixture
def fns(robot):
    return allow_list(GPSR, robot.ctx)


@pytest.fixture
def runner(pool):
    return ScriptRunner(pool)
