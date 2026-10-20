"""FindObjectsTool against a fake walkie-database node over real ROS 2 (no LLM)."""

import threading
import time
from unittest.mock import Mock

import pytest
from geometry_msgs.msg import Point
from rclpy.client import Client
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from walkie_database_msgs.msg import RememberedObject
from walkie_database_msgs.srv import FindObjects

from walkie_agent.tools import FindObjectsTool


class FakeDatabase(Node):
    """Serves /walkie_database/find_objects from a fixed list of objects."""

    def __init__(self):
        super().__init__("walkie_database")
        self.objects = [
            RememberedObject(
                label="cup",
                caption="a red mug",
                frame_id="map",
                position=Point(x=3.6, y=0.4, z=0.95),
                room="kitchen",
            ),
            RememberedObject(
                label="cup", frame_id="map", position=Point(x=1.3, y=1.4, z=0.8)
            ),
        ]
        self.requests: list[tuple[str, int]] = []
        self.delay_sec = 0.0
        self.failure = ""
        self.create_service(FindObjects, "~/find_objects", self.find_objects)

    def find_objects(self, request, response):
        self.requests.append((request.label, request.max_results))
        time.sleep(self.delay_sec)
        if self.failure:
            response.success = False
            response.message = self.failure
            return response
        response.success = True
        response.objects = [o for o in self.objects if o.label == request.label]
        return response


@pytest.fixture
def fake_database():
    node = FakeDatabase()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    executor.shutdown()
    node.destroy_node()
    thread.join(timeout=2)


def test_find_objects_describes_each_remembered_object(fake_database, connector):
    result = FindObjectsTool(connector=connector).invoke({"label": " cup "})

    assert result == (
        "Remembered 'cup', most recently seen first:\n"
        "- cup (a red mug) in the kitchen, at x=3.60 m, y=0.40 m in the map frame\n"
        "- cup, at x=1.30 m, y=1.40 m in the map frame"
    )
    assert fake_database.requests == [("cup", 5)]


def test_find_objects_says_when_nothing_is_remembered(fake_database, connector):
    result = FindObjectsTool(connector=connector).invoke({"label": "unicorn"})
    assert result == (
        "No remembered object is labelled 'unicorn'. Labels must match the "
        "detector's name exactly; the object may be stored under another name."
    )


def test_find_objects_needs_a_label(fake_database, connector):
    result = FindObjectsTool(connector=connector).invoke({"label": "  "})
    assert result == "Say what kind of object to look for, for example 'cup'."
    assert fake_database.requests == []


def test_find_objects_reports_a_failed_search(fake_database, connector):
    fake_database.failure = "The database could not search its objects: disk I/O error"
    result = FindObjectsTool(connector=connector).invoke({"label": "cup"})
    assert result == (
        "Memory search for 'cup' failed. "
        "The database could not search its objects: disk I/O error"
    )


def test_find_objects_without_the_database_fails_fast(connector):
    tool = FindObjectsTool(connector=connector, server_timeout_sec=0.5)
    start = time.monotonic()
    result = tool.invoke({"label": "cup"})
    assert result.startswith("Memory is unavailable")
    assert time.monotonic() - start < 1.5


def test_find_objects_gives_up_on_a_silent_database(fake_database, connector, monkeypatch):
    # The fake is a busy node: it receives the request and answers late. A node
    # built from another walkie_database_msgs never receives the request
    # (ADR-0035). The tool cannot tell the two apart, so it handles both the
    # same way: it drops the request and logs a hint for developers.
    fake_database.delay_sec = 1.0
    tool = FindObjectsTool(connector=connector, call_timeout_sec=0.3)

    dropped = []
    remove_pending_request = Client.remove_pending_request

    def record_drop(client, future):
        dropped.append(future)
        remove_pending_request(client, future)

    monkeypatch.setattr(Client, "remove_pending_request", record_drop)
    logger = Mock(wraps=connector.node.get_logger())
    monkeypatch.setattr(connector.node, "get_logger", lambda: logger)

    result = tool.invoke({"label": "cup"})
    # rclpy also calls remove_pending_request when a future completes, so only
    # the calls made before the late answer come from the tool.
    dropped_by_tool = list(dropped)
    time.sleep(fake_database.delay_sec)  # let the late answer go out before teardown

    assert result.startswith("Memory did not answer within 0.3 s when asked for 'cup'.")
    assert fake_database.requests == [("cup", 5)]
    assert len(dropped_by_tool) == 1
    assert not dropped_by_tool[0].done()  # the late answer was not delivered to it
    logger.warning.assert_called_once()
    assert "pinned in ros2_ws.repos" in logger.warning.call_args.args[0]


def test_find_objects_respects_forbidden_service(connector):
    tool = FindObjectsTool(connector=connector, forbidden=["/walkie_database/find_objects"])
    assert "not callable" in tool.invoke({"label": "cup"})
