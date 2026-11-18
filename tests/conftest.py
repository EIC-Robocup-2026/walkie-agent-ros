import importlib.util
import json
import os
import random
import threading
from pathlib import Path

import pytest

# Tests run on Fast DDS, although pixi sets rmw_zenoh_cpp for every command
# (ADR-0019, ADR-0033). Fast DDS needs no router, and it does not reach the
# Zenoh nodes of a robot or a simulation on the same computer. This must run
# before rclpy is imported: loading rcl checks RMW_IMPLEMENTATION and loads
# that middleware, so a change after the import has no effect.
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
# Keep test traffic on this machine and off any domain a real robot might use.
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
# A test must never pick up a developer's API keys, even through code that calls load_dotenv.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from walkie_agent.locations import load_locations  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The Mahoraga gym's tests need inspect-ai and the GPSR generator, which only the
# mahoraga pixi environment has, so the default environment does not collect them.
# Their fixtures are in this file because tests/mahoraga can have no conftest.py of
# its own: a second module named conftest would replace this one in sys.modules,
# and every test module that imports PROJECT_ROOT from it would fail to import.
GYM_TESTS = PROJECT_ROOT / "tests" / "mahoraga"
GYM_DEPENDENCIES = ("inspect_ai", "robocupathome_generator")
if any(importlib.util.find_spec(name) is None for name in GYM_DEPENDENCIES):
    collect_ignore_glob = ["mahoraga/test_*.py"]


@pytest.fixture(scope="session")
def ros():
    """A ROS context on a random domain, for the tests that ask for a real graph.

    Everything from ROS is imported here, so tests of the runtime, the atomics and
    the tasks collect and run on an installation that has no ROS at all.
    """
    # rai_interfaces comes from ros2_ws.repos, and scripts/activate.sh adds it to the
    # environment only once it is built. The import error alone does not say what to do.
    if importlib.util.find_spec("rai_interfaces") is None:
        raise pytest.UsageError(
            "Message package not built: rai_interfaces. "
            "Run `pixi run build-ros2` once, then run the tests again."
        )
    import rclpy

    rclpy.init(domain_id=random.randint(100, 200))
    yield
    rclpy.try_shutdown()


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    """Keep tests off LangSmith whatever the environment enables.

    A test of the tracing path turns it back on, with a fake client.
    """
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def locations():
    return load_locations(PROJECT_ROOT / "tests" / "fixtures" / "locations.toml")


# run_script's fixtures, shared by tests/scripting and tests/mahoraga. A module that
# defines its own `robot` fixture keeps it. The imports wait for a test that asks.


@pytest.fixture(scope="session")
def pool():
    """Two Monty workers inside bwrap, shared by every test of the session."""
    from walkie_agent.scripting.sandbox import open_pool

    with open_pool(max_processes=2) as shared:
        yield shared


@pytest.fixture
def robot():
    from walkie_agent.capabilities.fake import FakeRobot

    return FakeRobot(seed=1)


@pytest.fixture
def fns(robot):
    from walkie_agent.scripting.driver import allow_list
    from walkie_agent.tools.sets import GPSR

    return allow_list(GPSR, robot.ctx)


@pytest.fixture
def runner(pool):
    from walkie_agent.scripting.driver import ScriptRunner

    return ScriptRunner(pool)


@pytest.fixture(autouse=True)
def isolated_tools_home(monkeypatch, tmp_path):
    """Point every test's default shelf store at a fresh folder, so no test writes into
    src/walkie_agent/tools or reads what a real run left there.

    A test of the default store removes WALKIE_TOOLS_HOME itself.
    """
    store = tmp_path / "walkie-tools"
    monkeypatch.setenv("WALKIE_TOOLS_HOME", str(store))
    return store


@pytest.fixture(autouse=True)
def isolated_home(request):
    """Keep a gym test's gate results and logs out of the real home, and off the network.

    Every other test is left alone.
    """
    # Resolved, as GYM_TESTS is, so a test reached through a symlink is still isolated.
    if GYM_TESTS not in request.path.resolve().parents:
        return None
    monkeypatch = request.getfixturevalue("monkeypatch")
    home = request.getfixturevalue("tmp_path") / "mahoraga-home"
    monkeypatch.setenv("MAHORAGA_HOME", str(home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home


@pytest.fixture
def gated():
    """Record a whole, passing gate for some data, for gym tests about what runs after the gate."""
    from walkie_agent.evals.mahoraga import home
    from walkie_agent.evals.mahoraga.gate import _file_hash, code_digest, pair_digests

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
def fake_walkie(ros):
    from rclpy.executors import MultiThreadedExecutor

    from walkie_agent.sim.fake_walkie import FakeWalkie

    node = FakeWalkie(drive_time_sec=0.2)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    # shutdown() does not wait for spin(), and a destroy under it can raise there.
    executor.shutdown()
    thread.join(timeout=2)
    node.destroy_node()


@pytest.fixture
def connector(ros):
    from rai.communication.ros2 import ROS2Connector

    conn = ROS2Connector(node_name=f"test_connector_{random.randint(0, 1_000_000)}")
    yield conn
    conn.shutdown()
