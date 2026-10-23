"""Navigation tools: drive to a named place, or move a short distance from here.

Neither tool takes raw map coordinates from the LLM. Named places come from
config/locations.toml, and relative moves are bounded in length.
"""

import math
import threading
import time
from typing import Any, Type

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion, Transform
from langchain_core.tools import BaseTool, ToolException
from nav2_msgs.action import NavigateToPose
from pydantic import BaseModel, Field, PrivateAttr
from rai.tools.ros2.base import BaseROS2Tool
from rclpy.action import ActionClient
from rclpy.node import Node
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

# The arm check before a drive ends within this, as every arm read does
# (ADR-0024), the wait for another call to the commander included. It reads
# the joints once. While the arms are unverified, the whole recovery fits in
# it: the cancels of earlier goals, the commander's answers and both reads.
ARM_CHECK_TIMEOUT_S = 3.0
# Added to a drive's answer when the arm check could not read the arms: the
# commander's /get_joint_states service was not discovered, its read did not
# answer in time, or the commander answered that it could not read the joints.
ARMS_NOT_FOUND = "Arm state unknown: the commander was not found."
ARMS_UNKNOWN = "Arm state unknown: the commander did not answer."
ARMS_NOT_READ = "Arm state unknown: the commander gave no reading of the arms."

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

    A failed move raises ToolException. With handle_tool_error on, LangChain
    turns it into a ToolMessage of status "error" whose content is the
    exception's message, so the LLM reads the same sentence and the todo
    list's ledger counts the call as failed (ADR-0041). Called with a plain
    dict instead of a tool call, the tool returns that sentence as a string.

    With `arm`, the agent's CommanderClient, every drive first checks that
    both arms are in travel-safe poses (ADR-0025). Without it, as on a robot
    run without --arms, there is no check.

    With `gaze`, the agent's GazeLoop, every drive first stops the loop
    (ADR-0051): it would otherwise keep turning the base toward a person
    until Nav2's goal status reached it.
    """

    handle_tool_error: bool = True
    frame_id: str = MAP_FRAME
    action_name: str = "navigate_to_pose"
    server_timeout_sec: float = 2.0
    navigation_timeout_sec: float = 120.0
    # A walkie_agent.arm.commander.CommanderClient, or None. Typed Any: the
    # arm package imports walkie_agent.toolbox, which imports this module.
    arm: Any = None
    # A walkie_agent.gaze.GazeLoop, or None. Typed Any for the same reason:
    # walkie_agent.gaze imports walkie_agent.toolbox and this module.
    gaze: Any = None

    _client: ActionClient | None = PrivateAttr(default=None)

    def _get_client(self) -> ActionClient:
        if self._client is None:
            self._client = ActionClient(
                self.connector.node, NavigateToPose, self.action_name
            )
        return self._client

    def _navigate(self, target: str, x: float, y: float, yaw: float) -> str:
        """Drive to (x, y, yaw) in `frame_id`. Returns on arrival, or raises
        ToolException with a message for the LLM that names `target` and says
        what went wrong.

        Returns "" or, when the arms could not be checked, a note with a
        leading space for the caller to add to its answer."""
        if not self.is_writable(self.action_name):
            raise ToolException(f"Action {self.action_name} is not writable by this agent.")
        if self.gaze is not None:
            # Here, not at the top: walkie_agent.gaze imports this module.
            from walkie_agent.gaze import NAVIGATION, STOP_WAIT_S

            # Before the arm check, which may wait 3 s for the commander:
            # the base stops turning as soon as a drive is asked for, even
            # one the check then refuses. The gaze thread publishes the
            # stop's zeros; the wait, of at most STOP_WAIT_S, lets them go
            # out before the goal.
            self.gaze.stop(NAVIGATION, wait_s=STOP_WAIT_S)
        note = self._check_arms()

        client = self._get_client()
        if not client.wait_for_server(timeout_sec=self.server_timeout_sec):
            raise ToolException(
                f"Navigation is unavailable: action server '{self.action_name}' "
                "did not respond. Is Nav2 running?"
            )

        goal = NavigateToPose.Goal()
        goal.pose = self._make_pose(x, y, yaw)
        goal_handle = wait_for_future(client.send_goal_async(goal),
                                      self.server_timeout_sec)
        if goal_handle is None:
            raise ToolException(f"Navigation to {target} failed: Nav2 did not answer the goal.")
        if not goal_handle.accepted:
            raise ToolException(f"Navigation to {target} failed: Nav2 rejected the goal.")

        response = wait_for_future(goal_handle.get_result_async(),
                                   self.navigation_timeout_sec)
        if response is None:
            goal_handle.cancel_goal_async()
            raise ToolException(
                f"Navigation to {target} timed out after "
                f"{self.navigation_timeout_sec:g} s and was cancelled."
            )
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            status = _STATUS_NAMES.get(response.status, str(response.status))
            detail = f" {response.result.error_msg}" if response.result.error_msg else ""
            raise ToolException(f"Navigation to {target} failed ({status}).{detail}")
        return note

    def _check_arms(self) -> str:
        """The interlock: refuse to drive with an arm out (ADR-0025).

        check_arms() decides; this turns its refusal into the ToolException
        the LLM reads. Returns "" or the note that says why the arms were not
        checked.
        """
        # Here, not at the top: walkie_agent.toolbox imports this module.
        from walkie_agent.toolbox.types import StepError

        try:
            # The tool has no cancel event: check_arms() then uses a fresh
            # one, never set, so only the timeouts end the check.
            return check_arms(self.arm, "driving", "'home'")
        except StepError as e:
            raise ToolException(e.message) from e

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
            raise ToolException(f"Unknown location '{location}'. Known locations: {known}.")

        note = self._navigate(target.name, target.x, target.y, target.yaw)
        return f"Arrived at {target.name}.{note}"


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
            raise ToolException("Refused: every value must be a finite number.")
        # A zero move fails too: the robot did not move, so the call must not
        # count as evidence that it did.
        if not (forward_m or left_m or turn_deg):
            raise ToolException("Nothing to do: the move is zero.")
        distance = math.hypot(forward_m, left_m)
        if distance > self.max_distance_m:
            raise ToolException(
                f"Refused: a move of {distance:.2f} m is more than the "
                f"{self.max_distance_m:g} m allowed in one move."
            )

        try:
            current = self._current_transform()
        except TransformException as e:
            raise ToolException(
                f"Cannot move: the robot's position ({self.frame_id} -> "
                f"{self.robot_frame_id}) is unknown. {e}{self._tf_publisher_hint()}"
            ) from e
        if current is None:
            raise ToolException(
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
        note = self._navigate(target, x, y, target_yaw)
        return (
            f"Moved {_describe_move(forward_m, left_m, turn_deg)}. "
            f"Now at x={_round(x, 2):.2f} m, y={_round(y, 2):.2f} m, "
            f"heading {_round(math.degrees(target_yaw), 0):.0f} degrees "
            f"in the {self.frame_id} frame.{note}"
        )

    def _tf_publisher_hint(self) -> str:
        return tf_publisher_hint(self.connector.node)

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


def tf_publisher_hint(node: Node) -> str:
    """Names the likely cause when no node in reach publishes TF at all.

    Under rmw_zenoh, nodes on another ROS_DOMAIN_ID or behind another
    router are invisible, so an empty TF buffer looks the same as a robot
    that is not running. A frame missing from a TF tree that does exist
    has other causes, such as localization not having started. Returns ""
    when TF is published, or the hint with a leading space.
    """
    if node.count_publishers("/tf") or node.count_publishers("/tf_static"):
        return ""
    return (
        " No node publishes /tf: either the robot is not running, or the "
        "agent uses a different ROS_DOMAIN_ID or Zenoh router than the robot."
    )


def check_arms(
    arm: Any, action: str, example: str, cancel: threading.Event | None = None
) -> str:
    """The arm interlock before the base moves: refuse with an arm out (ADR-0025, ADR-0051).

    One decision for every base motion, so a drive and a gaze turn agree:
    the drives call it with `action` "driving", look_at with "starting a
    turn toward" and the handle, such as "starting a turn toward per_51c0.1".
    `arm` is the agent's CommanderClient, or None for no check, as on a run
    without --arms. `example` names the poses the refusal suggests for one
    arm, quoted, such as "'home'"; for both arms it is always 'home', the
    one travel-safe pose that takes both. `cancel` ends the read early;
    without it, only the timeout does.

    The decision is the arm client's check_for_driving(), made under its
    call lock, so no goal starts between its steps, all within
    ARM_CHECK_TIMEOUT_S. Raises StepError PRECONDITION_FAILED while a goal
    sent to the commander may still move an arm or a gripper; while the arms
    are unverified after the client abandoned a goal, until the commander
    shows every such goal to be no longer active and a settled read shows
    the arms still; and when an arm is known to be out of every travel-safe
    pose. TIMEOUT when another call kept the
    commander past the check's timeout; INTERNAL when the commander answers
    with a reading that cannot show the arms are safe; CANCELLED when
    `cancel` is set during the check. When the commander's
    /get_joint_states service is not found, its read does not answer or it
    gives no reading, the base moves, and the returned note says why the
    arms were not checked: failing open keeps the base working while the arm
    stack is down, at the cost that a crashed commander disables the check
    (ADR-0025). It never fails open while the arms are unverified. Returns
    "" after a check that passed, else the note with a leading space.
    """
    if arm is None:
        return ""
    # Here, not at the top: walkie_agent.toolbox imports this module, and
    # walkie_agent.arm imports walkie_agent.toolbox.
    from walkie_agent.arm.commander import NO_ANSWER, NO_READING, NOT_FOUND
    from walkie_agent.toolbox.types import StepError

    if cancel is None:
        cancel = threading.Event()
    try:
        # The module's value, read on every call, so a test can shorten it.
        check = arm.check_for_driving(cancel, timeout_s=ARM_CHECK_TIMEOUT_S)
    except StepError as e:
        if e.code == "CANCELLED" and cancel.is_set():
            # A stop or a reset, not a commander that did not answer: the
            # caller stops instead of moving with the arms unchecked.
            raise
        if e.code == "INTERNAL":
            # An answer that is malformed, lacks an arm's joints or holds a
            # position that is not a number.
            raise StepError(
                "INTERNAL",
                f"Not {action}: the commander's reading does not show that the arms are safe "
                f"for driving. {e.message} Tell the person that the arms cannot be checked.",
                e.data,
            ) from e
        # PRECONDITION_FAILED: a goal runs or is outstanding, or the arms are
        # unverified and recovery could not clear that: the commander did not
        # show an earlier goal to be no longer active, the arms could not be
        # read, or a settled read did not show them still. TIMEOUT: another
        # call kept the commander past the check's timeout.
        retry = " Try again in a few seconds." if e.code == "TIMEOUT" else ""
        raise StepError(e.code, f"Not {action}. {e.message}{retry}", e.data) from e
    if check.state is None:
        notes = {NOT_FOUND: ARMS_NOT_FOUND, NO_ANSWER: ARMS_UNKNOWN, NO_READING: ARMS_NOT_READ}
        return f" {notes[check.unread]}"
    state = check.state
    if state.unsafe_arms:
        raise StepError(
            "PRECONDITION_FAILED",
            _arms_not_safe(state.unsafe_arms, action, example),
            {"arms": list(state.unsafe_arms)},
        )
    if state.travel_safe is None:
        # Only for an arm that no pose has a table for.
        raise StepError(
            "INTERNAL",
            f"Not {action}: the commander's reading does not show where both arms are. "
            "Tell the person that the arms cannot be checked.",
        )
    return ""


def _arms_not_safe(arms: tuple[str, ...], action: str, example: str) -> str:
    """The refusal for `arms`, the arms out of every travel-safe pose, left first."""
    if len(arms) == 1:
        return (
            f"Not {action}: the {arms[0]} arm is not in a pose that is safe for driving. "
            f"Move it with move_arm_to_pose first, for example {example}."
        )
    return (
        f"Not {action}: the arms are not in a pose that is safe for driving. Move them with "
        "move_arm_to_pose first, for example 'home'."
    )


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
