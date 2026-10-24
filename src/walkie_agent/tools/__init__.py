"""Tools available to the Walkie agent: custom Walkie tools plus RAI's premade ones."""

from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    # Only for the annotations: walkie_agent.toolbox imports FORBIDDEN_NAMES
    # from this module, and walkie_agent.arm and walkie_agent.gaze import
    # walkie_agent.toolbox.
    from walkie_agent.arm.commander import CommanderClient
    from walkie_agent.gaze import GazeLoop
    from walkie_agent.toolbox import Toolbox

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
# mistakes, not a security boundary. The atomics' ToolContext checks it for
# every entity it creates.
#   /cmd_vel and the other base velocity inputs (manipulation.md's Forbidden
#                  names): raw velocity commands would bypass Nav2's
#                  obstacle avoidance. /cmd_vel_nav is Nav2's own input to
#                  twist_mux, /cmd_vel_joy and /cmd_vel_key the operator's,
#                  and /omni_wheel_drive_controller/cmd_vel twist_mux's
#                  output, which drives the wheels directly
#   /walkie/reset  the agent must not wipe its own memory
#   /walkie/task   the agent must not queue its own tasks or seed locked steps
#   /walkie/todos  the operator's view of the list; only update_todos edits it
# The one exception is the gaze loop, walkie_agent.gaze.GazeLoop (ADR-0051):
# its RotationPublisher publishes rotation only on /cmd_vel, twist_mux's
# lowest-priority input, and names it in allow_forbidden=("/cmd_vel",) to the
# ToolContext. /cmd_vel stays listed, so every other tool and helper is still
# refused it.
FORBIDDEN_NAMES = [
    "/cmd_vel",
    "/cmd_vel_nav",
    "/cmd_vel_joy",
    "/cmd_vel_key",
    "/omni_wheel_drive_controller/cmd_vel",
    "/walkie/reset",
    "/walkie/task",
    "/walkie/todos",
]


def build_walkie_tools(
    connector: ROS2Connector,
    locations: dict[str, Location],
    whoami_dir: Path | None = None,
    todos: TodoList | None = None,
    toolbox: "Toolbox | None" = None,
    arm: "CommanderClient | None" = None,
    gaze: "GazeLoop | None" = None,
) -> list[BaseTool]:
    """The agent's tools. With `todos`, also update_todos and wait_for_person,
    which act on that list; wait_for_person speaks through the same speak tool.
    With `toolbox`, also its atomic tools. With `arm`, the agent's
    CommanderClient, both navigation tools refuse to drive with an arm out
    (ADR-0025). With `gaze`, the agent's GazeLoop, both stop it before they
    drive (ADR-0051)."""
    speak = SpeakTool(connector=connector, forbidden=FORBIDDEN_NAMES)
    tools: list[BaseTool] = [
        # Walkie-specific tools.
        speak,
        ListLocationsTool(locations=locations),
        NavigateToLocationTool(
            connector=connector,
            locations=locations,
            forbidden=FORBIDDEN_NAMES,
            arm=arm,
            gaze=gaze,
        ),
        MoveRelativeTool(connector=connector, forbidden=FORBIDDEN_NAMES, arm=arm, gaze=gaze),
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
    if toolbox is not None:
        # The agent's ToolSet keeps one tool per name, so an atomic named like
        # a tool above would replace it without a word.
        names = {tool.name for tool in tools}
        clashes = sorted(tool.name for tool in toolbox.tools() if tool.name in names)
        if clashes:
            raise ValueError(f"Atomic tools clash with existing tools: {', '.join(clashes)}.")
        tools.extend(toolbox.tools())
    return tools
