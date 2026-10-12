"""Tools available to the Walkie agent: custom Walkie tools plus RAI's premade ones."""

from langchain_core.tools import BaseTool
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2 import GetROS2TopicsNamesAndTypesTool
from rai.tools.ros2.navigation.nav2_blocking import GetCurrentPoseTool
from rai.tools.time import WaitForSecondsTool

from walkie_agent.locations import Location
from walkie_agent.tools.navigation import ListLocationsTool, NavigateToLocationTool
from walkie_agent.tools.speech import SpeakTool

__all__ = [
    "ListLocationsTool",
    "NavigateToLocationTool",
    "SpeakTool",
    "build_walkie_tools",
]


def build_walkie_tools(
    connector: ROS2Connector, locations: dict[str, Location]
) -> list[BaseTool]:
    return [
        # Walkie-specific tools.
        SpeakTool(connector=connector),
        ListLocationsTool(locations=locations),
        NavigateToLocationTool(connector=connector, locations=locations),
        # Premade RAI tools, used as-is.
        GetCurrentPoseTool(connector=connector, frame_id="map", robot_frame_id="base_link"),
        GetROS2TopicsNamesAndTypesTool(connector=connector),
        WaitForSecondsTool(),
    ]
