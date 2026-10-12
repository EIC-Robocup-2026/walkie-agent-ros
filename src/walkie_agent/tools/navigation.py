"""Navigation tools that work with named locations instead of raw coordinates."""

import threading
from typing import Any, Type

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from langchain_core.tools import BaseTool
from nav2_msgs.action import NavigateToPose
from pydantic import BaseModel, Field, PrivateAttr
from rai.tools.ros2.base import BaseROS2Tool
from rclpy.action import ActionClient
from rclpy.task import Future
from tf_transformations import quaternion_from_euler

from walkie_agent.locations import Location

_STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


class ListLocationsToolInput(BaseModel):
    pass


class ListLocationsTool(BaseTool):
    """Lists the named places from config/locations.toml. Does not touch ROS."""

    name: str = "list_locations"
    description: str = (
        "List the named places Walkie can navigate to, with a short description of each. "
        "Call this before navigate_to_location if you are unsure of the exact name."
    )
    args_schema: Type[ListLocationsToolInput] = ListLocationsToolInput
    locations: dict[str, Location]

    def _run(self) -> str:
        if not self.locations:
            return "No named locations are configured."
        return "\n".join(
            f"- {loc.name}: {loc.description}" for loc in self.locations.values()
        )


class NavigateToLocationToolInput(BaseModel):
    location: str = Field(
        description="Name of the place to go to, exactly as returned by list_locations."
    )


class NavigateToLocationTool(BaseROS2Tool):
    """Sends a Nav2 NavigateToPose goal for a named location and waits for the result.

    RAI ships NavigateToPoseBlockingTool, but (as of rai-core 2.12.1) it waits on
    the goal with no timeout, so a stalled or crashed Nav2 freezes the agent.
    This tool bounds every wait and cancels the goal when navigation times out.
    """

    name: str = "navigate_to_location"
    description: str = (
        "Drive Walkie to a named place and wait until it arrives. "
        "Returns whether navigation succeeded."
    )
    args_schema: Type[NavigateToLocationToolInput] = NavigateToLocationToolInput
    locations: dict[str, Location]
    frame_id: str = "map"
    action_name: str = "navigate_to_pose"
    server_timeout_sec: float = 2.0
    navigation_timeout_sec: float = 120.0

    _client: ActionClient | None = PrivateAttr(default=None)

    def _get_client(self) -> ActionClient:
        if self._client is None:
            self._client = ActionClient(
                self.connector.node, NavigateToPose, self.action_name
            )
        return self._client

    def _run(self, location: str) -> str:
        target = self.locations.get(location.strip().lower())
        if target is None:
            known = ", ".join(self.locations) or "none"
            return f"Unknown location '{location}'. Known locations: {known}."

        if not self.is_writable(self.action_name):
            return f"Action {self.action_name} is not writable by this agent."

        client = self._get_client()
        if not client.wait_for_server(timeout_sec=self.server_timeout_sec):
            return (
                f"Navigation is unavailable: action server '{self.action_name}' "
                "did not respond. Is Nav2 running?"
            )

        goal = NavigateToPose.Goal()
        goal.pose = self._make_pose(target)
        goal_handle = _wait(client.send_goal_async(goal), self.server_timeout_sec)
        if goal_handle is None:
            return f"Navigation to {target.name} failed: Nav2 did not answer the goal."
        if not goal_handle.accepted:
            return f"Navigation to {target.name} failed: Nav2 rejected the goal."

        response = _wait(goal_handle.get_result_async(), self.navigation_timeout_sec)
        if response is None:
            goal_handle.cancel_goal_async()
            return (
                f"Navigation to {target.name} timed out after "
                f"{self.navigation_timeout_sec:g} s and was cancelled."
            )
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            status = _STATUS_NAMES.get(response.status, str(response.status))
            detail = f" {response.result.error_msg}" if response.result.error_msg else ""
            return f"Navigation to {target.name} failed ({status}).{detail}"
        return f"Arrived at {target.name}."

    def _make_pose(self, target: Location) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.connector.node.get_clock().now().to_msg()
        pose.pose.position.x = target.x
        pose.pose.position.y = target.y
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, target.yaw)
        pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return pose


def _wait(future: Future, timeout_sec: float) -> Any | None:
    """Wait for a future completed by the connector's executor thread.

    Returns the result, or None on timeout.
    """
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout_sec):
        return None
    return future.result()
