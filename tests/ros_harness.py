"""Probes for real ROS runs: an observed FakeWalkie and a silent operator.

tests/test_ros_integration.py and tests/ros_demo.py share these. Importing this loads
rclpy, so the importer must have chosen the middleware first: conftest does that for the
tests, and the recipe for the demo. Nothing here plays sound. /to_human is only recorded.
"""

import itertools
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rai_interfaces.msg import HRIMessage
from rclpy.action import GoalResponse
from rclpy.executors import Executor, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty, String

from walkie_agent.errors import StepError
from walkie_agent.ros.io import (
    CONSOLE_TOPIC,
    INPUT_TOPIC,
    LATCHED_QOS,
    QUEUED_QOS,
    RESET_TOPIC,
    TASK_TOPIC,
    TODOS_TOPIC,
)
from walkie_agent.ros.speech import SPEECH_TOPIC
from walkie_agent.sim.fake_walkie import FakeWalkie

Pose = tuple[float, float, float]


def wait_for(done: Callable[[], Any], what: str, timeout_s: float = 20.0) -> None:
    """Wait until done() is true, or raise AssertionError that names what never happened."""
    deadline = time.monotonic() + timeout_s
    while not done():
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed out after {timeout_s} s waiting for {what}.")
        time.sleep(0.005)


@dataclass(frozen=True)
class GoalEnd:
    """A goal as the server finished it: its GoalStatus, when, and where the robot was."""

    goal_id: str
    status: int
    at: float
    pose: Pose


class _ObservedHandle:
    """A goal handle that also reports each step of the drive."""

    def __init__(self, handle: Any, on_step: Callable[[], None]) -> None:
        self._handle = handle
        self._on_step = on_step

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def publish_feedback(self, feedback: Any) -> None:
        self._handle.publish_feedback(feedback)
        self._on_step()


class ObservedWalkie(FakeWalkie):
    """FakeWalkie, which keeps no record of its goals, with the probes tests need.

    `goals_received` counts what reached the server, so "no goal was sent" is observed
    rather than inferred from a robot that did not move. `driving` is set at the second
    step of a drive, because the first runs before the client can have heard the acceptance,
    and a test clears it before it starts the next one. `ends` holds the terminal status
    of each goal, read after the server's own code has set it.
    """

    def __init__(self, drive_time_sec: float = 0.2, **kwargs: Any) -> None:
        self._probe = threading.Lock()
        self.goals_received = 0
        self.goal_ids: list[str] = []
        self.ends: list[GoalEnd] = []
        self.samples: list[Pose] = []
        self.driving = threading.Event()
        super().__init__(drive_time_sec=drive_time_sec, **kwargs)
        self._nav_server.register_goal_callback(self._count_goal)

    def _count_goal(self, _goal: Any) -> GoalResponse:
        with self._probe:
            self.goals_received += 1
        return GoalResponse.ACCEPT

    def _execute_nav(self, goal_handle: Any) -> Any:
        goal_id = uuid.UUID(bytes=bytes(goal_handle.goal_id.uuid)).hex
        with self._probe:
            self.goal_ids.append(goal_id)
        steps = itertools.count(1)

        def on_step() -> None:
            sample = self.pose()
            with self._probe:
                self.samples.append(sample)
            if next(steps) >= 2:
                self.driving.set()

        result = super()._execute_nav(_ObservedHandle(goal_handle, on_step))
        end = GoalEnd(goal_id, goal_handle.status, time.monotonic(), self.pose())
        with self._probe:
            self.ends.append(end)
        return result


def poses_over(robot: FakeWalkie, seconds: float = 0.4) -> list[Pose]:
    """The robot's pose, sampled for a while. All equal means that it stayed still."""
    samples = [robot.pose()]
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.02)
        samples.append(robot.pose())
    return samples


class Operator(Node):
    """Stands where the operator stands: sends on the input topics, and records the outputs.

    Its /to_human subscription is the silent speech recorder. No speech node runs, so
    what the agent says is only stored here.
    """

    def __init__(self, name: str = "walkie_operator") -> None:
        super().__init__(name)
        self.console: list[str] = []
        self.todos: list[list[dict[str, Any]]] = []
        self.spoken: list[str] = []
        self._input = self.create_publisher(HRIMessage, INPUT_TOPIC, QUEUED_QOS)
        self._task = self.create_publisher(String, TASK_TOPIC, QUEUED_QOS)
        self._reset = self.create_publisher(Empty, RESET_TOPIC, QUEUED_QOS)
        self.create_subscription(
            HRIMessage, CONSOLE_TOPIC, lambda m: self.console.append(m.text), QUEUED_QOS
        )
        self.create_subscription(
            String, TODOS_TOPIC, lambda m: self.todos.append(json.loads(m.data)), LATCHED_QOS
        )
        self.create_subscription(HRIMessage, SPEECH_TOPIC, lambda m: self.spoken.append(m.text), 10)

    def say(self, text: str) -> None:
        self._input.publish(HRIMessage(text=text))

    def send_task(self, **fields: Any) -> None:
        self.send_raw_task(json.dumps(fields))

    def send_raw_task(self, data: str) -> None:
        self._task.publish(String(data=data))

    def reset(self) -> None:
        self._reset.publish(Empty())

    def statuses(self) -> list[str]:
        """The status of each step in the newest list it has heard."""
        return [item["status"] for item in self.todos[-1]] if self.todos else []

    def linked(self) -> bool:
        return all(p.get_subscription_count() for p in (self._input, self._task, self._reset))


def latched_todos(executor: Executor, timeout_s: float = 20.0) -> list[list[dict[str, Any]]]:
    """Every list that a subscriber joining now hears at once. One, if the topic is latched."""
    node = Node(f"late_listener_{uuid.uuid4().hex[:8]}")
    heard: list[list[dict[str, Any]]] = []
    node.create_subscription(
        String, TODOS_TOPIC, lambda m: heard.append(json.loads(m.data)), LATCHED_QOS
    )
    executor.add_node(node)
    try:
        wait_for(lambda: heard, "the latched todo list", timeout_s)
        return heard
    finally:
        executor.remove_node(node)
        node.destroy_node()


def wait_for_agent(app: Any, operator: Operator, timeout_s: float = 30.0) -> None:
    """Wait for TF, for Nav2 and for each link between the operator and the assembled agent.

    The agent's own bounds are short, because on a robot a missing server is a fault. A
    test's graph has only just started, so it first waits for discovery to finish.
    """

    def pose_is_fresh() -> bool:
        try:
            app.ctx.base.pose()
        except StepError:
            return False
        return True

    wait_for(pose_is_fresh, "a fresh pose from TF", timeout_s)
    wait_for(app._base._server_ready, "the Nav2 action server", timeout_s)
    wait_for(operator.linked, "the agent's input subscriptions", timeout_s)
    wait_for(app._speech._publisher.get_subscription_count, "a /to_human listener", timeout_s)
    wait_for(app.bridge._console.get_subscription_count, "a console listener", timeout_s)
    wait_for(lambda: operator.todos, "the first latched todo list", timeout_s)


@contextmanager
def spinning(*nodes: Node) -> Iterator[Executor]:
    """Spin these nodes on an executor of their own, apart from the agent's connector."""
    executor = MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield executor
    finally:
        executor.shutdown()
        for node in nodes:
            node.destroy_node()
        thread.join(timeout=2)
