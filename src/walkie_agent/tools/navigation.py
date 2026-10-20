"""Navigation tools: drive to a named place, or move a short distance from here.

Neither tool takes raw map coordinates from the LLM. Named places come from
config/locations.toml, and relative moves are bounded in length.
"""

import math
import time
from typing import Type

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion, Transform
from langchain_core.tools import BaseTool
from nav2_msgs.action import NavigateToPose
from pydantic import BaseModel, Field, PrivateAttr
from rai.tools.ros2.base import BaseROS2Tool
from rclpy.action import ActionClient
from rclpy.time import Time
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from walkie_agent.locations import Location
from walkie_agent.tools.waiters import wait_for_future

# Nav2's global frame and robot frame on walkie-ros. walkie-ros configures Nav2
# with base_footprint; on its main branch base_link is rotated 90 degrees from
# base_footprint, so headings read from base_link would be off by a quarter turn.
MAP_FRAME = "map"
ROBOT_FRAME = "base_footprint"

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


class Nav2GoalTool(BaseROS2Tool):
    """Base for tools that send a Nav2 NavigateToPose goal and wait for the result.

    RAI ships NavigateToPoseBlockingTool, but (as of rai-core 2.12.1) it waits on
    the goal with no timeout, so a stalled or crashed Nav2 freezes the agent.
    This class bounds every wait and cancels the goal when navigation times out.
    """

    frame_id: str = MAP_FRAME
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

    def _navigate(self, target: str, x: float, y: float, yaw: float) -> str | None:
        """Drive to (x, y, yaw) in `frame_id`. Returns None on arrival, or a
        message for the LLM that names `target` and says what went wrong."""
        if not self.is_writable(self.action_name):
            return f"Action {self.action_name} is not writable by this agent."

        client = self._get_client()
        if not client.wait_for_server(timeout_sec=self.server_timeout_sec):
            return (
                f"Navigation is unavailable: action server '{self.action_name}' "
                "did not respond. Is Nav2 running?"
            )

        goal = NavigateToPose.Goal()
        goal.pose = self._make_pose(x, y, yaw)
        goal_handle = wait_for_future(client.send_goal_async(goal),
                                      self.server_timeout_sec)
        if goal_handle is None:
            return f"Navigation to {target} failed: Nav2 did not answer the goal."
        if not goal_handle.accepted:
            return f"Navigation to {target} failed: Nav2 rejected the goal."

        response = wait_for_future(goal_handle.get_result_async(),
                                   self.navigation_timeout_sec)
        if response is None:
            goal_handle.cancel_goal_async()
            return (
                f"Navigation to {target} timed out after "
                f"{self.navigation_timeout_sec:g} s and was cancelled."
            )
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            status = _STATUS_NAMES.get(response.status, str(response.status))
            detail = f" {response.result.error_msg}" if response.result.error_msg else ""
            return f"Navigation to {target} failed ({status}).{detail}"
        return None

    def _make_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.connector.node.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return pose


class NavigateToLocationToolInput(BaseModel):
    location: str = Field(
        description="Name of the place to go to, exactly as returned by list_locations."
    )


class NavigateToLocationTool(Nav2GoalTool):
    """Drives to a named location from config/locations.toml."""

    name: str = "navigate_to_location"
    description: str = (
        "Drive Walkie to a named place and wait until it arrives. "
        "Returns whether navigation succeeded."
    )
    args_schema: Type[NavigateToLocationToolInput] = NavigateToLocationToolInput
    locations: dict[str, Location]

    def _run(self, location: str) -> str:
        target = self.locations.get(location.strip().lower())
        if target is None:
            known = ", ".join(self.locations) or "none"
            return f"Unknown location '{location}'. Known locations: {known}."

        error = self._navigate(target.name, target.x, target.y, target.yaw)
        return error or f"Arrived at {target.name}."


class MoveRelativeToolInput(BaseModel):
    forward_m: float = Field(
        description="Meters to move forward. Negative moves backward."
    )
    left_m: float = Field(
        default=0.0, description="Meters to move to the left. Negative moves right."
    )
    turn_deg: float = Field(
        default=0.0,
        description="Degrees to turn once there. Positive turns left, negative turns right.",
    )


class MoveRelativeTool(Nav2GoalTool):
    """Moves a short distance relative to the robot's current pose.

    Reads the current pose from TF (`frame_id` -> `robot_frame_id`), adds the
    offset in the robot's own frame, and sends the result as one Nav2 goal, so
    Nav2 still plans the path and avoids obstacles. Moves longer than
    `max_distance_m` are refused, which bounds the damage of a misread unit
    (centimeters taken as meters, for example).
    """

    name: str = "move_relative"
    description: str = (
        "Move Walkie a short distance from where it is now, relative to the way it "
        "is facing: forward or backward, left or right, then optionally turn. "
        "Waits until it arrives. Each move is limited to a few meters; "
        "the tool says so when a move is too long."
    )
    args_schema: Type[MoveRelativeToolInput] = MoveRelativeToolInput
    robot_frame_id: str = ROBOT_FRAME
    max_distance_m: float = 3.0
    tf_timeout_sec: float = 1.0

    def _run(self, forward_m: float, left_m: float = 0.0, turn_deg: float = 0.0) -> str:
        # NaN fails every comparison, so it would pass the distance limit below.
        # json.loads and pydantic both accept NaN and infinity from the LLM.
        if not all(math.isfinite(v) for v in (forward_m, left_m, turn_deg)):
            return "Refused: every value must be a finite number."
        if not (forward_m or left_m or turn_deg):
            return "Nothing to do: the move is zero."
        distance = math.hypot(forward_m, left_m)
        if distance > self.max_distance_m:
            return (
                f"Refused: a move of {distance:.2f} m is more than the "
                f"{self.max_distance_m:g} m allowed in one move."
            )

        try:
            current = self._current_transform()
        except TransformException as e:
            return (
                f"Cannot move: the robot's position ({self.frame_id} -> "
                f"{self.robot_frame_id}) is unknown. {e}"
            )
        if current is None:
            return (
                f"Cannot move: the robot's position ({self.frame_id} -> "
                f"{self.robot_frame_id}) is out of date. TF is not updating, or the "
                "agent and the robot disagree on sim time (see --use-sim-time)."
            )

        q = current.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        x = current.translation.x + forward_m * \
            math.cos(yaw) - left_m * math.sin(yaw)
        y = current.translation.y + forward_m * \
            math.sin(yaw) + left_m * math.cos(yaw)
        target_yaw = _normalize_angle(yaw + math.radians(turn_deg))

        target = f"the point {_describe_move(forward_m, left_m, turn_deg)}"
        error = self._navigate(target, x, y, target_yaw)
        return error or (
            f"Moved {_describe_move(forward_m, left_m, turn_deg)}. "
            f"Now at x={_round(x, 2):.2f} m, y={_round(y, 2):.2f} m, "
            f"heading {_round(math.degrees(target_yaw), 0):.0f} degrees "
            f"in the {self.frame_id} frame."
        )

    def _current_transform(self) -> Transform | None:
        """The robot's pose from a transform stamped no earlier than this call.

        The latest transform in the TF buffer can predate the end of the
        previous goal: the goal result and the TF update arrive on different
        topics. Moving relative to that pose would use the heading from before
        the robot's final turn. Returns None if no new enough transform
        arrives within `tf_timeout_sec`; raises TransformException if there is
        no transform at all.
        """
        clock = self.connector.node.get_clock()
        called_at = clock.now()
        deadline = time.monotonic() + self.tf_timeout_sec
        while True:
            stamped = self.connector.get_transform(
                self.frame_id, self.robot_frame_id, timeout_sec=self.tf_timeout_sec
            )
            if Time.from_msg(stamped.header.stamp) >= called_at:
                return stamped.transform
            if time.monotonic() > deadline:
                return None
            time.sleep(0.02)


def _describe_move(forward_m: float, left_m: float, turn_deg: float) -> str:
    """Plain words for a relative move, for example "1 m forward, 0.5 m right"."""
    parts = []
    if forward_m:
        parts.append(
            f"{abs(forward_m):g} m {'forward' if forward_m > 0 else 'backward'}")
    if left_m:
        parts.append(f"{abs(left_m):g} m {'left' if left_m > 0 else 'right'}")
    if turn_deg:
        parts.append(
            f"turning {abs(turn_deg):g} degrees {'left' if turn_deg > 0 else 'right'}")
    return ", ".join(parts) or "in place"


def _round(value: float, digits: int) -> float:
    """Round for display, turning -0.0 into 0.0 so it never prints as "-0"."""
    return round(value, digits) + 0.0


def _normalize_angle(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))
