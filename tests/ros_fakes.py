"""Deterministic doubles of the ROS objects the wrappers touch.

Nothing here needs rclpy.init or a graph. Futures are real rclpy futures, which with no
executor run their callbacks inline on the thread that completes them, so a test picks
that thread and that moment. A hook runs a test's code at an exact point inside a
wrapper, so no test sleeps and hopes that a verb got there.
"""

import concurrent.futures
import math
import threading
from collections.abc import Callable
from typing import Any

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import TransformStamped
from nav2_msgs.action import NavigateToPose
from rclpy.task import Future
from rclpy.time import Time
from tf2_ros import LookupException

from walkie_agent.errors import StepError

# How long a test waits for a thread that should finish. Only a deadlock ever reaches it.
WATCHDOG_S = 10.0
START_NS = 1_000_000_000_000


class Background:
    """A call on a daemon thread, so that a deadlock fails its test and not the whole run."""

    def __init__(self, call: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._future: concurrent.futures.Future = concurrent.futures.Future()
        threading.Thread(target=self._run, args=(call, args, kwargs), daemon=True).start()

    def _run(self, call: Callable[..., Any], args: tuple, kwargs: dict) -> None:
        try:
            self._future.set_result(call(*args, **kwargs))
        except BaseException as error:
            self._future.set_exception(error)

    def done(self) -> bool:
        return self._future.done()

    def result(self) -> Any:
        return self._future.result(WATCHDOG_S)

    def step_error(self) -> StepError:
        """The StepError the call raised. Any other ending fails the test."""
        error = self._future.exception(WATCHDOG_S)
        assert isinstance(error, StepError), f"expected a StepError, got {error!r}"
        return error


def bounded(call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a call that must not block, on a thread the watchdog can give up on."""
    return Background(call, *args, **kwargs).result()


def missing_tf() -> TransformStamped:
    raise LookupException("map does not exist")


def transform(
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    *,
    stamp_ns: int,
    frame: str = "map",
    child: str = "base_footprint",
) -> TransformStamped:
    """A real TF message, stamped `stamp_ns`, with the robot at (x, y, yaw)."""
    stamped = TransformStamped()
    stamped.header.frame_id = frame
    stamped.child_frame_id = child
    stamped.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    stamped.transform.translation.x = float(x)
    stamped.transform.translation.y = float(y)
    stamped.transform.rotation.z = math.sin(yaw / 2.0)
    stamped.transform.rotation.w = math.cos(yaw / 2.0)
    return stamped


def nav_result(
    status: int = GoalStatus.STATUS_SUCCEEDED, error_code: int = 0, error_msg: str = ""
) -> Any:
    """A real result of the NavigateToPose action."""
    return NavigateToPose.Impl.GetResultService.Response(
        status=status, result=NavigateToPose.Result(error_code=error_code, error_msg=error_msg)
    )


class FakeClock:
    """A ROS clock that moves only when a test moves it. `on_now` runs before each reading."""

    def __init__(self, nanoseconds: int = START_NS) -> None:
        self.nanoseconds = nanoseconds
        self.on_now: Callable[[], None] | None = None

    def now(self) -> Time:
        if self.on_now is not None:
            self.on_now()
        return Time(nanoseconds=self.nanoseconds)


class FakePublisher:
    def __init__(self, msg_type: Any, topic: str, qos: Any) -> None:
        self.msg_type, self.topic, self.qos = msg_type, topic, qos
        self.subscribers = 1
        self.on_count: Callable[[], None] | None = None
        self.on_publish: Callable[[Any], None] | None = None
        self.published: list[Any] = []

    def get_subscription_count(self) -> int:
        if self.on_count is not None:
            self.on_count()
        return self.subscribers

    def publish(self, message: Any) -> None:
        if self.on_publish is not None:
            self.on_publish(message)
        self.published.append(message)


class FakeSubscription:
    def __init__(self, msg_type: Any, topic: str, callback: Callable[[Any], None], qos: Any) -> None:
        self.msg_type, self.topic, self.callback, self.qos = msg_type, topic, callback, qos


class FakeNode:
    """The parts of a node the wrappers use. It resolves names as rclpy does."""

    def __init__(self, *, namespace: str = "/", remaps: dict[str, str] | None = None) -> None:
        self.clock = FakeClock()
        self.namespace = namespace
        self.remaps = remaps or {}
        self.graph_publishers: dict[str, int] = {}
        self.publishers: list[FakePublisher] = []
        self.subscriptions: list[FakeSubscription] = []
        self.destroyed: list[Any] = []
        self.destroy_error: Exception | None = None

    def get_clock(self) -> FakeClock:
        return self.clock

    def resolve_topic_name(self, name: str, *, only_expand: bool = False) -> str:
        expanded = name if name.startswith("/") else f"{self.namespace.rstrip('/')}/{name}"
        return expanded if only_expand else self.remaps.get(expanded, expanded)

    def count_publishers(self, topic: str) -> int:
        return self.graph_publishers.get(topic, 0)

    def create_publisher(self, msg_type: Any, topic: str, qos: Any, **_: Any) -> FakePublisher:
        self.publishers.append(FakePublisher(msg_type, topic, qos))
        return self.publishers[-1]

    def create_subscription(
        self, msg_type: Any, topic: str, callback: Callable[[Any], None], qos: Any, **_: Any
    ) -> FakeSubscription:
        self.subscriptions.append(FakeSubscription(msg_type, topic, callback, qos))
        return self.subscriptions[-1]

    def _destroy(self, entity: Any) -> bool:
        self.destroyed.append(entity)
        if self.destroy_error is not None:
            raise self.destroy_error
        return True

    destroy_publisher = _destroy
    destroy_subscription = _destroy


class FakeConnector:
    """The connector's node and TF read. `tf` decides what each read returns or raises."""

    def __init__(self, node: FakeNode | None = None) -> None:
        self.node = node or FakeNode()
        self.robot = (0.0, 0.0, 0.0)
        self.tf: Callable[[], TransformStamped] = self.fresh_sample
        self.tf_reads: list[tuple[str, str, float]] = []

    def fresh_sample(self) -> TransformStamped:
        """The robot now, stamped newer than any read that has already begun."""
        return transform(*self.robot, stamp_ns=self.node.clock.nanoseconds + 1)

    def get_transform(
        self, target_frame: str, source_frame: str, timeout_sec: float = 5.0
    ) -> TransformStamped:
        self.tf_reads.append((target_frame, source_frame, timeout_sec))
        return self.tf()


class FakeGoalHandle:
    """An accepted or rejected goal. A test completes its futures on any thread it likes."""

    def __init__(
        self,
        log: list[str],
        *,
        accepted: bool = True,
        on_cancel: Callable[["FakeGoalHandle", Future], None] | None = None,
    ) -> None:
        self.accepted = accepted
        self.log = log
        self.on_cancel = on_cancel
        self.result_future = Future()
        self.result_asks = 0
        self.result_error: Exception | None = None
        self.result_asked = threading.Event()
        self.cancel_asks = 0
        self.cancel_error: Exception | None = None
        self.cancel_asked = threading.Event()
        self.cancel_futures: list[Future] = []

    def get_result_async(self) -> Future:
        self.result_asks += 1
        self.log.append("result asked")
        self.result_asked.set()
        if self.result_error is not None:
            raise self.result_error
        return self.result_future

    def cancel_goal_async(self) -> Future:
        self.cancel_asks += 1
        self.log.append("cancel")
        self.cancel_asked.set()
        if self.cancel_error is not None:
            raise self.cancel_error
        future = Future()
        self.cancel_futures.append(future)
        if self.on_cancel is not None:
            self.on_cancel(self, future)
        return future

    def acknowledge(self, return_code: int = CancelGoal.Response.ERROR_NONE) -> None:
        """Answer the newest cancel request."""
        self.cancel_futures[-1].set_result(CancelGoal.Response(return_code=return_code))

    def finish(
        self, status: int = GoalStatus.STATUS_SUCCEEDED, error_code: int = 0, error_msg: str = ""
    ) -> None:
        self.log.append("terminal")
        self.result_future.set_result(nav_result(status, error_code, error_msg))


class FakeActionClient:
    """An owned ActionClient: its ctor arguments, goals, futures and an ordered event log."""

    def __init__(
        self, node: Any, action_type: Any, action_name: str, *, callback_group: Any = None
    ) -> None:
        self.node, self.action_type, self.action_name = node, action_type, action_name
        self.callback_group = callback_group
        self.ready = True
        self.ready_polls = 0
        self.on_ready: Callable[[], None] | None = None
        self.on_send: Callable[[], None] | None = None
        self.goals: list[Any] = []
        self.send_futures: list[Future] = []
        self.handles: list[FakeGoalHandle] = []
        self.destroyed = 0
        self.log: list[str] = []
        self._sent = threading.Condition()

    def server_is_ready(self) -> bool:
        self.ready_polls += 1
        if self.on_ready is not None:
            self.on_ready()
        return self.ready

    def send_goal_async(self, goal: Any) -> Future:
        self.log.append("send")
        self.goals.append(goal)
        self.send_futures.append(Future())
        if self.on_send is not None:
            self.on_send()
        with self._sent:
            self._sent.notify_all()
        return self.send_futures[-1]

    def wait_for_goals(self, count: int = 1) -> None:
        with self._sent:
            assert self._sent.wait_for(lambda: len(self.goals) >= count, WATCHDOG_S)

    def accept(
        self,
        *,
        accepted: bool = True,
        on_cancel: Callable[[FakeGoalHandle, Future], None] | None = None,
    ) -> FakeGoalHandle:
        """Answer the newest goal on this thread, which runs its callback if one is attached."""
        handle = FakeGoalHandle(self.log, accepted=accepted, on_cancel=on_cancel)
        self.handles.append(handle)
        self.send_futures[-1].set_result(handle)
        return handle

    def destroy(self) -> None:
        self.destroyed += 1
        self.log.append("destroy")
