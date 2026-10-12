import os
import random
import threading
from pathlib import Path

import pytest
import rclpy
from rai.communication.ros2 import ROS2Connector
from rclpy.executors import MultiThreadedExecutor

from walkie_agent.locations import load_locations
from walkie_agent.sim.fake_walkie import FakeWalkie

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep test traffic on this machine and off any domain a real robot might use.
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"


@pytest.fixture(scope="session", autouse=True)
def ros():
    rclpy.init(domain_id=random.randint(100, 200))
    yield
    rclpy.try_shutdown()


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
