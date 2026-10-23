"""Tools available to the Walkie agent: custom Walkie tools plus RAI's premade ones."""

from pathlib import Path

from langchain_core.tools import BaseTool
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2 import GetROS2TopicsNamesAndTypesTool
from rai.tools.ros2.navigation.nav2_blocking import GetCurrentPoseTool
from rai.tools.time import WaitForSecondsTool

from walkie_agent.locations import Location
from walkie_agent.todo import TodoList
from walkie_agent.tools.memory import FindObjectsTool
from walkie_agent.tools.navigation import (
    MAP_FRAME,
    ROBOT_FRAME,
    ListLocationsTool,
    MoveRelativeTool,
    NavigateToLocationTool,
)
from walkie_agent.tools.speech import SpeakTool
from walkie_agent.tools.todos import UpdateTodosTool, WaitForPersonTool

__all__ = [
    "FORBIDDEN_NAMES",
    "FindObjectsTool",
    "ListLocationsTool",
    "MoveRelativeTool",
    "NavigateToLocationTool",
    "SpeakTool",
    "UpdateTodosTool",
    "WaitForPersonTool",
    "build_walkie_tools",
]

# ROS names that no tool may read or write. Every BaseROS2Tool checks this list
# itself, so it only covers tools that are given it; it is a guard against
# mistakes, not a security boundary.
#   /cmd_vel       raw velocity commands would bypass Nav2's obstacle avoidance
#   /walkie/reset  the agent must not wipe its own memory
FORBIDDEN_NAMES = ["/cmd_vel", "/walkie/reset"]


def build_walkie_tools(
    connector: ROS2Connector,
    locations: dict[str, Location],
    whoami_dir: Path | None = None,
    todos: TodoList | None = None,
) -> list[BaseTool]:
    """The agent's tools. With `todos`, also update_todos and wait_for_person,
    which act on that list; wait_for_person speaks through the same speak tool."""
    speak = SpeakTool(connector=connector, forbidden=FORBIDDEN_NAMES)
    tools: list[BaseTool] = [
        # Walkie-specific tools.
        speak,
        ListLocationsTool(locations=locations),
        NavigateToLocationTool(
            connector=connector, locations=locations, forbidden=FORBIDDEN_NAMES
        ),
        MoveRelativeTool(connector=connector, forbidden=FORBIDDEN_NAMES),
        FindObjectsTool(connector=connector, forbidden=FORBIDDEN_NAMES),
        # Premade RAI tools, used as-is.
        GetCurrentPoseTool(
            connector=connector,
            frame_id=MAP_FRAME,
            robot_frame_id=ROBOT_FRAME,
            forbidden=FORBIDDEN_NAMES,
        ),
        GetROS2TopicsNamesAndTypesTool(connector=connector, forbidden=FORBIDDEN_NAMES),
        WaitForSecondsTool(),
    ]
    if todos is not None:
        tools.append(UpdateTodosTool(todos=todos))
        tools.append(WaitForPersonTool(todos=todos, speak_tool=speak))
    if whoami_dir is not None:
        from walkie_agent.whoami import build_docs_tool

        docs_tool = build_docs_tool(whoami_dir)
        if docs_tool is not None:
            tools.append(docs_tool)
    return tools
