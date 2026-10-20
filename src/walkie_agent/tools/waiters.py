"""Bounded waits for the tools that call ROS 2 actions and services."""

import threading
from typing import Any

from rclpy.task import Future


def wait_for_future(future: Future, timeout_sec: float) -> Any | None:
    """Wait for a future completed by the connector's executor thread.

    Returns the result, or None on timeout.
    """
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout_sec):
        return None
    return future.result()
