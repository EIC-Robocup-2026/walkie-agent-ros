"""Builds the Walkie agent on top of RAI's ReActAgent."""

import secrets
import threading
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from rai.agents.langchain import ReActAgent
from rai.communication.hri_connector import HRIConnector, HRIMessage
from rai.communication.ros2 import ROS2Connector
from std_msgs.msg import Empty

from walkie_agent.locations import Location
from walkie_agent.tools import build_walkie_tools

SYSTEM_PROMPT = """\
You are Walkie, a home service robot competing in RoboCup@Home.
You have a mobile base and can drive between named places.

Rules:
- To talk to a person in the room, use the `speak` tool. Your plain-text replies
  go to the operator console only, never to the speaker.
- Use `list_locations` to see where you can go, then `navigate_to_location`.
- If a tool reports a failure, tell the person what went wrong with `speak`
  and do not pretend the action succeeded.
- Keep spoken sentences short and friendly. They are read out by a
  text-to-speech voice, so never use markdown, lists or code in them.
"""


class WalkieAgent(ReActAgent):
    """RAI's ReActAgent with a way to forget the conversation.

    RAI keeps every task, LLM reply and tool result in one message list for as
    long as the process runs, and sends all of it to the LLM on every task.
    reset() starts a new conversation without restarting the agent.

    Each conversation is also a Langfuse session (`session_id`), so traces
    from the same conversation are grouped together in the Langfuse UI.

    Relies on ReActAgent internals of rai-core 2.12 (`_run_agent`, the message
    buffer and the interrupt event); recheck them when upgrading rai-core.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._graph = self.agent
        self._reset_requested = threading.Event()
        self.session_id = ""
        self._start_session()

    def reset(self) -> None:
        """Drop queued tasks, stop the current task, and clear the conversation.

        Returns immediately; the agent thread applies the reset. A running task
        stops at its next step, so a tool call already in progress (such as a
        navigation goal) finishes first. Safe to call from any thread.
        """
        with self._buffer_lock:
            self._received_messages.clear()
        self._reset_requested.set()
        self._interrupt_event.set()

    def subscribe_reset(self, connector: ROS2Connector, topic: str) -> None:
        """Call reset() whenever a std_msgs/Empty message arrives on `topic`."""
        connector.node.create_subscription(Empty, topic, lambda _msg: self.reset(), 10)

    def _run_agent(self) -> None:
        # Runs on the agent thread between tasks, so the message list is never
        # cleared while the graph is still using it.
        if self._reset_requested.is_set():
            self._reset_requested.clear()
            self._interrupt_event.clear()
            self.state["messages"].clear()
            self._start_session()
            self.logger.info(f"Conversation reset. Langfuse session: {self.session_id}")
        super()._run_agent()

    def _start_session(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_id = f"walkie-{stamp}-{secrets.token_hex(2)}"
        # Langfuse reads the session ID from the metadata of each run.
        self.agent = self._graph.with_config(metadata={"langfuse_session_id": self.session_id})


def create_walkie_agent(
    llm: BaseChatModel,
    connector: ROS2Connector,
    locations: dict[str, Location],
    target_connectors: dict[str, HRIConnector[HRIMessage]],
    stream_response: bool = True,
) -> WalkieAgent:
    return WalkieAgent(
        target_connectors=target_connectors,
        llm=llm,
        tools=build_walkie_tools(connector, locations),
        system_prompt=SYSTEM_PROMPT,
        stream_response=stream_response,
    )
