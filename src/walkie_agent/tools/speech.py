"""Speech output tool."""

import threading
import time
import uuid
from typing import Any, Type

from pydantic import BaseModel, Field, PrivateAttr
from rai.tools.ros2.base import BaseROS2Tool
from rai_interfaces.msg import HRIMessage
from rclpy.publisher import Publisher
from std_msgs.msg import String


class SpeakToolInput(BaseModel):
    text: str = Field(description="What Walkie should say out loud, in plain words.")


class SpeakTool(BaseROS2Tool):
    """Publishes text to the speech topic; RAI's TextToSpeechAgent voices it.

    This is the only way the agent talks to people in the room. The agent's
    plain-text replies go to the operator console topic, not to the speaker.

    RAI's TextToSpeechAgent plays only messages that share one
    communication_id, and accepts a new ID only after a "stop" on
    /voice_commands (sent by the speech recognition agent when a person has
    finished speaking). The tool mirrors that: it keeps one speech ID and
    switches to a new one whenever it sees "stop".

    The tool owns its publisher instead of using connector.send_message: that
    method creates the publisher on first use and publishes immediately, so the
    first sentence is dropped before DDS discovery matches the TTS subscriber.
    """

    name: str = "speak"
    description: str = (
        "Say something out loud to the people near Walkie. "
        "Use this for every sentence meant for a person in the room."
    )
    args_schema: Type[SpeakToolInput] = SpeakToolInput
    topic: str = "/to_human"
    voice_commands_topic: str = "/voice_commands"
    subscriber_wait_sec: float = 2.0

    _publisher: Publisher = PrivateAttr()
    _speech_id: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex)
    _speech_id_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def model_post_init(self, context: Any) -> None:
        # Create the publisher now so discovery runs while the agent starts up,
        # not on the first call (cross-process discovery can take seconds).
        node = self.connector.node
        self._publisher = node.create_publisher(HRIMessage, self.topic, 10)
        node.create_subscription(
            String, self.voice_commands_topic, self._on_voice_command, 10
        )

    @property
    def speech_id(self) -> str:
        with self._speech_id_lock:
            return self._speech_id

    def _on_voice_command(self, msg: String) -> None:
        if msg.data == "stop":
            with self._speech_id_lock:
                self._speech_id = uuid.uuid4().hex

    def _run(self, text: str) -> str:
        if not self.is_writable(self.topic):
            return f"Topic {self.topic} is not writable by this agent."

        publisher = self._publisher
        deadline = time.monotonic() + self.subscriber_wait_sec
        while publisher.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                return f"Nothing was said: no speech node is listening on {self.topic}."
            time.sleep(0.05)

        msg = HRIMessage(text=text, communication_id=self.speech_id, seq_end=True)
        msg.header.stamp = self.connector.node.get_clock().now().to_msg()
        publisher.publish(msg)
        return f"Said: {text}"
