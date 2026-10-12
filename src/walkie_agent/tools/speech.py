"""Speech output tool."""

import time
from typing import Any, Type

from pydantic import BaseModel, Field, PrivateAttr
from rai.tools.ros2.base import BaseROS2Tool
from rclpy.publisher import Publisher
from std_msgs.msg import String


class SpeakToolInput(BaseModel):
    text: str = Field(description="What Walkie should say out loud, in plain words.")


class SpeakTool(BaseROS2Tool):
    """Publishes text to the speech topic; a TTS node on the robot voices it.

    This is the only way the agent talks to people in the room. Its plain-text
    replies go to /to_human (the operator console), not to the speaker.

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
    topic: str = "/walkie/speech"
    subscriber_wait_sec: float = 2.0

    _publisher: Publisher = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        # Create the publisher now so discovery runs while the agent starts up,
        # not on the first call (cross-process discovery can take seconds).
        self._publisher = self.connector.node.create_publisher(String, self.topic, 10)

    def _run(self, text: str) -> str:
        if not self.is_writable(self.topic):
            return f"Topic {self.topic} is not writable by this agent."

        publisher = self._publisher
        deadline = time.monotonic() + self.subscriber_wait_sec
        while publisher.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                return f"Nothing was said: no speech node is listening on {self.topic}."
            time.sleep(0.05)

        publisher.publish(String(data=text))
        return f"Said: {text}"
