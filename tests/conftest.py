import importlib.util
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
    executor.shutdown()
    node.destroy_node()
    thread.join(timeout=2)


@pytest.fixture
def connector(ros):
    from rai.communication.ros2 import ROS2Connector

    conn = ROS2Connector(node_name=f"test_connector_{random.randint(0, 1_000_000)}")
    yield conn
    conn.shutdown()
