"""Bounded waits for the tools that call ROS 2 actions and services."""

import threading
import time
from typing import Any

from rclpy.task import Future

# How often the wait looks at the future itself while its callback is pending.
_POLL_S = 0.05


def wait_for_future(future: Future, timeout_sec: float) -> Any | None:
    """Wait for a future completed by the connector's executor thread.

    Returns the result, or None on timeout.
    """
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    deadline = time.monotonic() + timeout_sec
    # future.done() as well as the event: a future that belongs to an
    # executor only schedules its callbacks there, so the event is set on
    # that executor's next spin, or never if it has stopped spinning.
    while not (done.is_set() or future.done()):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        done.wait(min(_POLL_S, remaining))
    return future.result()
