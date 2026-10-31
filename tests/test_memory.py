"""FindObjectsTool against a fake walkie-database node over real ROS 2 (no LLM)."""

import threading
import time
from unittest.mock import Mock

import pytest
from geometry_msgs.msg import Point
from rclpy.client import Client
from rclpy.node import Node
from walkie_database_msgs.msg import RememberedObject
from walkie_database_msgs.srv import FindObjects

from conftest import Spinner, wait_until
from walkie_agent.tools import FindObjectsTool


class FakeDatabase(Node):
    """Serves /walkie_database/find_objects from a fixed list of objects.

    The fake_database fixture spins it on one thread, so it answers one
    request at a time, in the order they came.
    """

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
        # Every answer waits on this gate. A test clears it to hold the
        # answer, as a busy node would, and sets it to let it go.
        self.answer_gate = threading.Event()
        self.answer_gate.set()
        # Set once an answer that the gate held has been sent.
        self.late_handled = threading.Event()
        # Whether the answer being made now was held. One thread serves, so
        # it is sent before the next request's callback runs.
        self._held = False
        self.failure = ""
        server = self.create_service(FindObjects, "~/find_objects", self.find_objects)
        # The executor sends what the callback returns, so only a wrapped
        # send_response knows when a held answer has gone out.
        server.send_response = self._sent(server.send_response)

    def destroy_node(self):
        # A held answer goes on at once, so its callback ends with the node.
        self.answer_gate.set()
        super().destroy_node()

    def find_objects(self, request, response):
        # Read before the request is recorded: a test that sees the request
        # may open the gate at once.
        self._held = not self.answer_gate.is_set()
        self.requests.append((request.label, request.max_results))
        self.answer_gate.wait()
        if self.failure:
            response.success = False
            response.message = self.failure
            return response
        response.success = True
        response.objects = [o for o in self.objects if o.label == request.label]
        return response

    def _sent(self, send_response):
        """`send_response`, then late_handled set if the gate held the answer."""

        def send(response, header):
            send_response(response, header)
            if self._held:
                self._held = False
                self.late_handled.set()

        return send


@pytest.fixture
def fake_database():
    node = FakeDatabase()
    # One thread, so the fake answers one request at a time, in order.
    spinner = Spinner(node, num_threads=1)
    yield node
    spinner.close()


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


def test_find_objects_marks_only_a_failure_as_an_error(fake_database, connector):
    # As a tool call, a failure comes back as an error ToolMessage, so the todo
    # list's ledger counts it as failed (ADR-0041). Finding nothing is not a
    # failure.
    tool = FindObjectsTool(connector=connector)

    def call(label: str):
        return tool.invoke(
            {"name": tool.name, "args": {"label": label}, "id": "call_1", "type": "tool_call"}
        )

    empty = call("  ")
    assert (empty.content, empty.status) == (
        "Say what kind of object to look for, for example 'cup'.",
        "error",
    )
    assert call("unicorn").status == "success"


def test_find_objects_reports_a_failed_search(fake_database, connector):
    fake_database.failure = "The database could not search its objects: disk I/O error"
    result = FindObjectsTool(connector=connector).invoke({"label": "cup"})
    assert result == (
        "Memory search for 'cup' failed. "
        "The database could not search its objects: disk I/O error"
    )


def test_find_objects_without_the_database_fails_fast(connector, monkeypatch):
    waits = []
    wait_for_service = Client.wait_for_service

    def record_wait(client, timeout_sec=None):
        waits.append((client.srv_name, timeout_sec))
        return wait_for_service(client, timeout_sec)

    monkeypatch.setattr(Client, "wait_for_service", record_wait)
    tool = FindObjectsTool(connector=connector, server_timeout_sec=0.5)
    start = time.monotonic()
    result = tool.invoke({"label": "cup"})
    elapsed = time.monotonic() - start

    assert result.startswith("Memory is unavailable")
    assert waits == [("/walkie_database/find_objects", 0.5)]
    # The spy holds the claim; the bound only shows the call timeout was not
    # waited out, with room for a stalled runner.
    assert elapsed < 5.0


def test_find_objects_gives_up_on_a_silent_database(fake_database, connector, monkeypatch):
    # The fake is a busy node: it receives the request and answers late. A node
    # built from another walkie_database_msgs never receives the request
    # (ADR-0035). The tool cannot tell the two apart, so it handles both the
    # same way: it drops the request and logs a hint for developers.
    tool = FindObjectsTool(connector=connector, call_timeout_sec=0.3)

    dropped, missed = [], []
    remove_pending_request = Client.remove_pending_request
    get_pending_request = Client.get_pending_request

    def record_drop(client, future):
        dropped.append(future)
        remove_pending_request(client, future)

    def record_miss(client, sequence_number):
        try:
            return get_pending_request(client, sequence_number)
        except KeyError:
            # The executor then throws the answer away.
            missed.append(sequence_number)
            raise

    monkeypatch.setattr(Client, "remove_pending_request", record_drop)
    monkeypatch.setattr(Client, "get_pending_request", record_miss)
    logger = Mock(wraps=connector.node.get_logger())
    monkeypatch.setattr(connector.node, "get_logger", lambda: logger)
    # The node holds its answer, so the answer cannot come in time.
    fake_database.answer_gate.clear()

    result = tool.invoke({"label": "cup"})

    assert result.startswith("Memory did not answer within 0.3 s when asked for 'cup'.")
    # rclpy also calls remove_pending_request when a future completes. The
    # held answer has not come yet, so every drop so far is the tool's.
    assert len(dropped) == 1
    logger.warning.assert_called_once()
    assert "pinned in ros2_ws.repos" in logger.warning.call_args.args[0]
    # The request may still be on its way, so wait for it before the gate
    # opens: one that came after would not be held.
    assert wait_until(lambda: fake_database.requests), "The request never reached the database."
    assert fake_database.requests == [("cup", 5)]
    # The late answer goes out, and the client has no request left for it.
    fake_database.answer_gate.set()
    assert fake_database.late_handled.wait(10), "The held answer was never sent."
    assert wait_until(lambda: missed), "No answer came for a request the tool had dropped."
    assert not dropped[0].done()  # the late answer was not delivered to it


def test_find_objects_respects_forbidden_service(connector):
    tool = FindObjectsTool(connector=connector, forbidden=["/walkie_database/find_objects"])
    assert "not callable" in tool.invoke({"label": "cup"})
