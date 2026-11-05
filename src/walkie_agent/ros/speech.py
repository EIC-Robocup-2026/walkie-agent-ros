"""RosSpeech: ctx.speech over /to_human, which RAI's TextToSpeechAgent voices.

say() returns once the line is published. That is not a claim about playback, since
nothing acknowledges it. It takes no cancel token, so the fixed stop line still speaks
while motion is latched.

Two calls can overlap: an ordinary say holds a LangChain worker while the Runner speaks
the stop line on another. Each call has its own discovery deadline, and the lock covers
only reading or rotating the speech id and publishing.
"""

import logging
import threading
import time
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING

from rai_interfaces.msg import HRIMessage
from std_msgs.msg import String

from walkie_agent.capabilities.guards import require_text, require_timeout
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.ros.policy import FORBIDDEN_NAMES, check_endpoint, ros_failure

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

logger = logging.getLogger(__name__)

SPEECH_TOPIC = "/to_human"
VOICE_COMMANDS_TOPIC = "/voice_commands"
# How long a say waits for a speech node. An implementation default, not a measured need.
DISCOVERY_S = 2.0
_POLL_S = 0.02


class RosSpeech:
    """Publishes one HRIMessage per line, under one speech id until a voice "stop".

    RAI's TextToSpeechAgent plays only messages that share one communication_id, and
    takes a new id only after a "stop" on /voice_commands, which its speech recognition
    agent sends when a person has finished speaking. So this keeps one id and rotates
    it on that word.

    The publisher exists from the start, so discovery runs while the agent starts up.
    connector.send_message would create it on first use and drop the first line.
    """

    def __init__(
        self,
        connector: "ROS2Connector",
        *,
        topic: str = SPEECH_TOPIC,
        voice_commands_topic: str = VOICE_COMMANDS_TOPIC,
        forbidden: Iterable[str] = FORBIDDEN_NAMES,
        discovery_s: float = DISCOVERY_S,
    ) -> None:
        self._node = connector.node
        self._topic = check_endpoint(self._node, topic, forbidden)
        voice_commands = check_endpoint(self._node, voice_commands_topic, forbidden)
        self._discovery_s = discovery_s
        self._lock = threading.Lock()
        self._speech_id = uuid.uuid4().hex
        self._closed = threading.Event()
        self._publisher = self._node.create_publisher(HRIMessage, self._topic, 10)
        self._subscription = self._node.create_subscription(
            String, voice_commands, self._on_voice_command, 10
        )

    def say(self, text: str, *, timeout_s: float = 10.0) -> None:
        require_timeout(timeout_s)
        require_text(text=text)
        if not text.strip():
            raise StepError(ErrorCode.REFUSED, "Nothing was said: the text is blank.")
        try:
            self._wait_for_listener(min(self._discovery_s, timeout_s))
            with self._lock:
                self._require_open()
                message = HRIMessage(text=text, communication_id=self._speech_id, seq_end=True)
                message.header.stamp = self._node.get_clock().now().to_msg()
                self._publisher.publish(message)
        except StepError:
            raise
        except Exception as error:
            raise ros_failure(error, "Speech") from error

    def close(self) -> None:
        """Destroy the publisher and the subscription. Later and waiting calls get UNAVAILABLE."""
        with self._lock:
            if self._closed.is_set():
                return
            self._closed.set()
        for destroy, entity in (
            (self._node.destroy_publisher, self._publisher),
            (self._node.destroy_subscription, self._subscription),
        ):
            try:
                destroy(entity)
            except Exception:
                logger.exception("Could not destroy a speech endpoint.")

    def _wait_for_listener(self, budget_s: float) -> None:
        deadline = time.monotonic() + budget_s
        while True:
            self._require_open()
            if self._publisher.get_subscription_count() > 0:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StepError(
                    ErrorCode.UNAVAILABLE,
                    f"Nothing was said: no speech node is listening on {self._topic}.",
                )
            self._closed.wait(min(_POLL_S, remaining))

    def _require_open(self) -> None:
        if self._closed.is_set():
            raise StepError(ErrorCode.UNAVAILABLE, "Nothing was said: speech has shut down.")

    def _on_voice_command(self, message: String) -> None:
        if message.data == "stop":
            with self._lock:
                self._speech_id = uuid.uuid4().hex
