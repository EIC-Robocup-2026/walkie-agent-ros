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

# The message packages from ros2_ws.repos, built by `pixi run build-ros2`.
# scripts/activate.sh adds them to the environment only once they are built.
# Without them, an import fails with an ImportError that does not say what to
# do: the FakeWalkie import below needs rai_interfaces, and the test modules
# import walkie_database_msgs. The check runs here, before those imports,
# because pytest imports this file before it calls pytest_configure.
_missing = [
    name
    for name in ("rai_interfaces", "walkie_database_msgs")
    if importlib.util.find_spec(name) is None
]
if _missing:
    raise pytest.UsageError(
        f"Message packages not built: {', '.join(_missing)}. "
        "Run `pixi run build-ros2` once, then run the tests again."
    )

import rclpy  # noqa: E402
from rai.communication.ros2 import ROS2Connector  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402

from walkie_agent.locations import load_locations  # noqa: E402
from walkie_agent.sim.fake_walkie import FakeWalkie  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def ros():
    rclpy.init(domain_id=random.randint(100, 200))
    yield
    rclpy.try_shutdown()


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    """Keep tests off Langfuse/LangSmith whatever config.toml enables.

    RAI reads the tracing settings from ./config.toml when an agent is built
    and again on every LLM call, so both lookups are patched.
    """
    monkeypatch.setattr("rai.agents.langchain.agent.get_tracing_callbacks", lambda *_: [])
    monkeypatch.setattr(
        "rai.agents.langchain.invocation_helpers.get_tracing_callbacks", lambda *_: []
    )


@pytest.fixture
def locations():
    return load_locations(PROJECT_ROOT / "config" / "locations.toml")


@pytest.fixture
def fake_walkie():
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
def connector():
    conn = ROS2Connector(node_name=f"test_connector_{random.randint(0, 1_000_000)}")
    yield conn
    conn.shutdown()
