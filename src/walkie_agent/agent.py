"""Builds the Walkie agent on top of RAI's ReActAgent."""

from langchain_core.language_models import BaseChatModel
from rai.agents.langchain import ReActAgent
from rai.communication.hri_connector import HRIConnector, HRIMessage
from rai.communication.ros2 import ROS2Connector

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
- Keep spoken sentences short and friendly.
"""


def create_walkie_agent(
    llm: BaseChatModel,
    connector: ROS2Connector,
    locations: dict[str, Location],
    target_connectors: dict[str, HRIConnector[HRIMessage]],
    stream_response: bool = True,
) -> ReActAgent:
    return ReActAgent(
        target_connectors=target_connectors,
        llm=llm,
        tools=build_walkie_tools(connector, locations),
        system_prompt=SYSTEM_PROMPT,
        stream_response=stream_response,
    )
