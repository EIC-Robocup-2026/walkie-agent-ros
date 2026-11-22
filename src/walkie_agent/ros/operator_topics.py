"""RosBridge: the operator's topics, between ROS callbacks and the Runner.

Input side, on the connector's executor: /from_human, /walkie/task and /walkie/reset,
and with --scripts /walkie/verdict. Each callback validates, copies and queues on the
Runner, so it never runs the model, calls a blocking verb or joins anything. The
verdict's callback records the operator's judgement of the last task, a few small
file writes at most, and puts its answer on the console. A bad message is logged and
dropped, and no callback lets an exception reach the executor.

Output side: `on_todos` and `on_reply` are the listeners of build_agent and the Runner,
and `on_trace` is build_agent's ReasoningTrace listener. All are safe from any thread,
and a failed publish only logs. Replies and status lines go to the console topic, never
to the speaker: only `say`, `wait_for_person` and the stop line reach /to_human. The
trace goes to /walkie/reasoning as rcl_interfaces/Log, which Foxglove's Log panel shows.

Input is text only. A message with an image or audio payload is refused, because
acting on its text alone would silently drop part of what was asked.
"""

import json
import logging
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from rai_interfaces.msg import HRIMessage
from rcl_interfaces.msg import Log
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty, String

from walkie_agent.ros.speech import SPEECH_TOPIC
from walkie_agent.runtime.runner import is_stop_utterance
from walkie_agent.runtime.task_templates import TaskTemplate, parse_task_message
from walkie_agent.runtime.todos import console_line, public

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

    from walkie_agent.runtime import Runner

logger = logging.getLogger(__name__)

INPUT_TOPIC = "/from_human"
CONSOLE_TOPIC = "/walkie/console"
RESET_TOPIC = "/walkie/reset"
TASK_TOPIC = "/walkie/task"
TODOS_TOPIC = "/walkie/todos"
# The operator's judgement of the task that ended last, with --scripts.
VERDICT_TOPIC = "/walkie/verdict"
# The agent's steps, one Log per ReasoningTrace line, named by its kind.
REASONING_TOPIC = "/walkie/reasoning"

QUEUED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10
)
# A subscriber that joins late still gets the current list.
LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)
# A viewer that joins late still gets the last steps of the run.
REASONING_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=100,
)


class RosBridge:
    """Owns the operator's publishers and subscriptions on the connector's node.

    The publishers exist from the start, so the Runner's first list has somewhere to
    go. `open` subscribes, `close_ingress` refuses further input, and `close`, last of
    all, destroys the endpoints. Each is safe to repeat.

    With `on_verdict`, open also subscribes to `verdict_topic`. Each std_msgs Bool on
    it goes to on_verdict(data), and the line it returns goes to the console.
    """

    def __init__(
        self,
        connector: "ROS2Connector",
        templates: Mapping[str, TaskTemplate],
        *,
        input_topic: str = INPUT_TOPIC,
        output_topic: str = CONSOLE_TOPIC,
        reset_topic: str = RESET_TOPIC,
        task_topic: str = TASK_TOPIC,
        todos_topic: str = TODOS_TOPIC,
        verdict_topic: str = VERDICT_TOPIC,
        reasoning_topic: str = REASONING_TOPIC,
        on_verdict: Callable[[bool], str] | None = None,
    ) -> None:
        self._node = connector.node
        if self._node.resolve_topic_name(output_topic) == self._node.resolve_topic_name(
            SPEECH_TOPIC
        ):
            raise ValueError(f"The console topic may not be the speech topic {SPEECH_TOPIC}.")
        self._templates = dict(templates)
        self._input_topic, self._reset_topic = input_topic, reset_topic
        self._task_topic, self._output_topic = task_topic, output_topic
        self._todos_topic, self._reasoning_topic = todos_topic, reasoning_topic
        self._verdict_topic, self._on_verdict_call = verdict_topic, on_verdict
        self._runner: Runner | None = None
        # Guards the entity lists and orders a publish against close. It is never held
        # while waiting, and an ingress callback does not take it.
        self._lock = threading.Lock()
        self._ingress_closed = threading.Event()
        self._closed = False
        self._subscriptions: list[Any] = []
        self._console = self._node.create_publisher(HRIMessage, output_topic, QUEUED_QOS)
        self._todos = self._node.create_publisher(String, todos_topic, LATCHED_QOS)
        self._reasoning = self._node.create_publisher(Log, reasoning_topic, REASONING_QOS)

    # -- Lifecycle -------------------------------------------------------------

    def open(self, runner: "Runner") -> None:
        """Start taking input for `runner`, which must already be able to queue it."""
        self._runner = runner
        endpoints: list[tuple[Any, str, Callable[[Any], None]]] = [
            (HRIMessage, self._input_topic, self._on_input),
            (String, self._task_topic, self._on_task),
            (Empty, self._reset_topic, self._on_reset),
        ]
        if self._on_verdict_call is not None:
            endpoints.append((Bool, self._verdict_topic, self._on_verdict))
        for msg_type, topic, callback in endpoints:
            subscription = self._node.create_subscription(msg_type, topic, callback, QUEUED_QOS)
            with self._lock:
                self._subscriptions.append(subscription)

    def close_ingress(self) -> None:
        """Ignore all further input, also from a callback that is already on its way.

        The subscriptions stay until close(). Destroying one while the executor builds its
        wait set can raise in rclpy's spin thread and end it, and the cancel of a goal that
        is still driving needs that thread to hear Nav2.
        """
        self._ingress_closed.set()

    def close(self) -> None:
        """Close the input too, then destroy the endpoints. Later listeners publish nothing."""
        self.close_ingress()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            subscriptions, self._subscriptions = self._subscriptions, []
        self._destroy(self._node.destroy_subscription, subscriptions)
        self._destroy(
            self._node.destroy_publisher, (self._console, self._todos, self._reasoning)
        )

    @staticmethod
    def _destroy(destroy: Callable[[Any], Any], entities: Iterable[Any]) -> None:
        for entity in entities:
            try:
                destroy(entity)
            except Exception:
                logger.exception("Could not destroy a bridge endpoint.")

    # -- Input, on the executor ------------------------------------------------

    def _on_input(self, message: HRIMessage) -> None:
        if self._ingress_closed.is_set():
            return
        has_payload = len(message.images) or len(message.audios)
        if has_payload and not is_stop_utterance(message.text):  # a stop is never held back
            logger.warning(
                "Refused a message on %s with %d images and %d audios: input is text only.",
                self._input_topic,
                len(message.images),
                len(message.audios),
            )
            return
        if not message.text.strip():
            logger.warning("Ignored a message on %s with no text.", self._input_topic)
            return
        self._queue(self._input_topic, self._runner.hear, message.text)

    def _on_task(self, message: String) -> None:
        if self._ingress_closed.is_set():
            return
        self._queue(self._task_topic, self._submit, message.data)

    def _on_reset(self, _message: Empty) -> None:
        if self._ingress_closed.is_set():
            return
        self._queue(self._reset_topic, self._runner.reset)

    def _on_verdict(self, message: Bool) -> None:
        if self._ingress_closed.is_set():
            return
        self._queue(self._verdict_topic, self._judge, bool(message.data))

    def _judge(self, passed: bool) -> None:
        self.on_reply(self._on_verdict_call(passed))

    def _submit(self, data: str) -> None:
        text, todos = parse_task_message(data, self._templates)
        self._runner.submit(text, todos, source="operator")

    @staticmethod
    def _queue(topic: str, call: Callable[..., None], *args: Any) -> None:
        try:
            call(*args)
        except ValueError as error:  # a malformed task or seed, with a one-line reason
            logger.warning("Ignored a message on %s: %s", topic, error)
        except Exception:
            logger.exception("Could not take a message on %s.", topic)

    # -- Output, from any thread ----------------------------------------------

    def on_todos(self, items: list[dict[str, Any]], changed: list[dict[str, Any]]) -> None:
        """Publish the list as JSON, then one console line per item whose status changed."""
        try:
            data = json.dumps(public(items), ensure_ascii=False)
            self._publish(self._todos, String(data=data), self._todos_topic)
            for item in changed:
                if line := console_line(item):
                    self.on_reply(line)
        except Exception:
            logger.exception("Could not publish the todo list.")

    def on_reply(self, text: str) -> None:
        """Publish one console line, as a new message with its own id."""
        try:
            message = HRIMessage(text=text, communication_id=uuid.uuid4().hex, seq_end=True)
            message.header.stamp = self._node.get_clock().now().to_msg()
            self._publish(self._console, message, self._output_topic)
        except Exception:
            logger.exception("Could not publish a console line.")

    def on_trace(self, kind: str, text: str, failed: bool) -> None:
        """Publish one step of the run as a Log named `kind`, at WARN when it `failed`."""
        try:
            message = Log(level=Log.WARN if failed else Log.INFO, name=kind, msg=text)
            message.stamp = self._node.get_clock().now().to_msg()
            self._publish(self._reasoning, message, self._reasoning_topic)
        except Exception:
            logger.exception("Could not publish a step of the run.")

    def _publish(self, publisher: Any, message: Any, topic: str) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                publisher.publish(message)
            except Exception:
                logger.exception("Could not publish on %s.", topic)
