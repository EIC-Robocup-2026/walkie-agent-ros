"""Walkie tools against the fake robot over real ROS 2 (no LLM)."""

import math
import time

import pytest
from geometry_msgs.msg import TransformStamped, Twist
from rai.tools.ros2.navigation.nav2_blocking import GetCurrentPoseTool
from std_msgs.msg import Empty, String
from tf2_ros import StaticTransformBroadcaster

from walkie_agent.locations import Location
from walkie_agent.tools import (
    MoveRelativeTool,
    NavigateToLocationTool,
    SpeakTool,
    build_walkie_tools,
)
from walkie_agent.tools.navigation import MAP_FRAME, ROBOT_FRAME


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_navigate_to_location_moves_robot(fake_walkie, connector, locations):
    tool = NavigateToLocationTool(connector=connector, locations=locations)

    assert tool.invoke({"location": "kitchen"}) == "Arrived at kitchen."

    x, y, yaw = fake_walkie.pose()
    assert (x, y) == pytest.approx((3.5, 1.2))
    assert yaw == pytest.approx(1.57, abs=1e-3)


def test_navigate_to_location_is_case_insensitive(fake_walkie, connector, locations):
    tool = NavigateToLocationTool(connector=connector, locations=locations)
    assert tool.invoke({"location": " Bedroom "}) == "Arrived at bedroom."


def test_navigate_to_unknown_location_lists_known_ones(connector, locations):
    result = NavigateToLocationTool(connector=connector, locations=locations).invoke(
        {"location": "garage"}
    )
    assert result.startswith("Unknown location 'garage'.")
    assert "kitchen" in result


def test_navigate_without_nav2_fails_fast(connector, locations):
    tool = NavigateToLocationTool(
        connector=connector, locations=locations, server_timeout_sec=0.5
    )
    start = time.monotonic()
    result = tool.invoke({"location": "kitchen"})
    assert "Navigation is unavailable" in result
    assert time.monotonic() - start < 1.5


def test_navigate_reports_nav2_abort(fake_walkie, connector):
    far = {"moon": Location(name="moon", x=100.0, y=0.0, yaw=0.0)}
    result = NavigateToLocationTool(connector=connector, locations=far).invoke(
        {"location": "moon"}
    )
    assert result == (
        "Navigation to moon failed (ABORTED). Goal is outside the map (10.0 m limit)."
    )


def test_navigate_timeout_cancels_goal(fake_walkie, connector, locations):
    fake_walkie.drive_time_sec = 3.0
    tool = NavigateToLocationTool(
        connector=connector, locations=locations, navigation_timeout_sec=0.5
    )

    result = tool.invoke({"location": "kitchen"})

    assert result == "Navigation to kitchen timed out after 0.5 s and was cancelled."
    time.sleep(0.6)  # let the cancel reach the fake and the drive loop stop
    x, _, _ = fake_walkie.pose()
    assert 0.0 < x < 3.5, "robot should have stopped partway to the kitchen"


def test_navigate_respects_forbidden_action(connector, locations):
    tool = NavigateToLocationTool(
        connector=connector, locations=locations, forbidden=["navigate_to_pose"]
    )
    assert "not writable" in tool.invoke({"location": "kitchen"})


def test_speak_delivers_first_message(fake_walkie, connector):
    result = SpeakTool(connector=connector).invoke({"text": "Hello, I am Walkie."})
    assert result == "Said: Hello, I am Walkie."
    assert wait_until(lambda: fake_walkie.heard == ["Hello, I am Walkie."])


def test_speak_without_listener_reports_failure(connector):
    tool = SpeakTool(connector=connector, subscriber_wait_sec=0.3)
    assert tool.invoke({"text": "anyone?"}).startswith("Nothing was said")


def as_tool_call(tool, **args) -> dict:
    return {"name": tool.name, "args": args, "id": "call_1", "type": "tool_call"}


def test_tool_failures_are_error_results_for_a_tool_call(connector, locations):
    # As a tool call, a failure comes back as an error ToolMessage with the
    # same sentence, so the todo list's ledger counts it as failed (ADR-0041).
    navigate = NavigateToLocationTool(connector=connector, locations=locations)
    unknown = navigate.invoke(as_tool_call(navigate, location="garage"))
    assert unknown.status == "error"
    assert unknown.content.startswith("Unknown location 'garage'.")

    move = MoveRelativeTool(connector=connector)
    zero = move.invoke(as_tool_call(move, forward_m=0.0))
    assert (zero.content, zero.status) == ("Nothing to do: the move is zero.", "error")

    speak = SpeakTool(connector=connector, subscriber_wait_sec=0.3)
    unheard = speak.invoke(as_tool_call(speak, text="anyone?"))
    assert unheard.status == "error"
    assert unheard.content.startswith("Nothing was said")


def test_a_successful_move_is_a_success_result(fake_walkie, connector, locations):
    tool = NavigateToLocationTool(connector=connector, locations=locations)
    result = tool.invoke(as_tool_call(tool, location="kitchen"))
    assert (result.content, result.status) == ("Arrived at kitchen.", "success")


def test_speak_changes_speech_id_only_on_stop(fake_walkie, connector):
    """RAI's TTS plays one speech ID until "stop"; the tool must follow the same rule."""
    tool = SpeakTool(connector=connector)
    commands = connector.node.create_publisher(String, "/voice_commands", 10)
    assert wait_until(lambda: commands.get_subscription_count() > 0)

    tool.invoke({"text": "One."})
    commands.publish(String(data="pause"))
    tool.invoke({"text": "Two."})
    first_id = tool.speech_id
    commands.publish(String(data="stop"))
    assert wait_until(lambda: tool.speech_id != first_id)
    tool.invoke({"text": "Three."})

    assert wait_until(lambda: len(fake_walkie.heard_speech_ids) == 3)
    one, two, three = fake_walkie.heard_speech_ids
    assert one == two == first_id
    assert three == tool.speech_id


def test_tools_hide_forbidden_topics(connector, locations):
    node = connector.node
    node.create_publisher(Twist, "/cmd_vel", 10)
    node.create_publisher(Empty, "/walkie/reset", 10)
    node.create_publisher(String, "/walkie/task", 10)
    node.create_publisher(String, "/walkie/todos", 10)
    node.create_publisher(String, "/walkie/visible_test_topic", 10)
    lister = next(
        t for t in build_walkie_tools(connector, locations)
        if t.name == "get_ros2_topics_names_and_types"
    )

    assert wait_until(lambda: "/walkie/visible_test_topic" in lister.invoke({}))
    topics = lister.invoke({})
    for forbidden in ("/cmd_vel", "/walkie/reset", "/walkie/task", "/walkie/todos"):
        assert forbidden not in topics


def test_rai_get_current_pose_reads_fake_tf(fake_walkie, connector):
    tool = GetCurrentPoseTool(
        connector=connector, frame_id=MAP_FRAME, robot_frame_id=ROBOT_FRAME
    )
    assert "frame_id='map'" in tool.invoke({})


def tf_shows(connector, x, y):
    """True once TF has caught up with a robot teleported to (x, y)."""
    try:
        t = connector.get_transform(MAP_FRAME, ROBOT_FRAME, timeout_sec=0.5).transform
    except Exception:
        return False
    return (t.translation.x, t.translation.y) == pytest.approx((x, y))


def test_move_relative_moves_in_the_robots_frame(fake_walkie, connector):
    # Facing +y (90 degrees): forward is +y and left is -x in the map frame.
    fake_walkie.teleport(1.0, 1.0, math.pi / 2)
    assert wait_until(lambda: tf_shows(connector, 1.0, 1.0))

    result = MoveRelativeTool(connector=connector).invoke(
        {"forward_m": 1.0, "left_m": 0.5, "turn_deg": -90}
    )

    assert result == (
        "Moved 1 m forward, 0.5 m left, turning 90 degrees right. "
        "Now at x=0.50 m, y=2.00 m, heading 0 degrees in the map frame."
    )
    x, y, yaw = fake_walkie.pose()
    assert (x, y) == pytest.approx((0.5, 2.0))
    assert yaw == pytest.approx(0.0, abs=1e-6)


def test_move_relative_right_after_a_goal_uses_the_new_heading(fake_walkie, connector, locations):
    # The fake turns to the goal heading at the very end, so a TF pose cached
    # from before the result arrived still faces the old way.
    NavigateToLocationTool(connector=connector, locations=locations).invoke(
        {"location": "kitchen"}  # x=3.5, y=1.2, facing +y
    )
    MoveRelativeTool(connector=connector).invoke({"forward_m": 1.0})

    assert fake_walkie.pose()[:2] == pytest.approx((3.5, 2.2), abs=1e-2)


def test_move_relative_refuses_stale_tf(connector):
    # A static transform never gets a newer stamp, like a robot whose TF stopped.
    static = StaticTransformBroadcaster(connector.node)
    t = TransformStamped()
    t.header.frame_id = MAP_FRAME
    t.child_frame_id = ROBOT_FRAME
    t.transform.rotation.w = 1.0
    static.sendTransform(t)

    tool = MoveRelativeTool(connector=connector, tf_timeout_sec=0.5)
    assert tool.invoke({"forward_m": 1.0}).startswith(
        "Cannot move: the robot's position (map -> base_footprint) is out of date."
    )


def test_move_relative_refuses_long_moves(connector):
    # No fake robot: the limit is checked before TF or Nav2 are touched.
    result = MoveRelativeTool(connector=connector).invoke({"forward_m": 300.0})
    assert result == "Refused: a move of 300.00 m is more than the 3 m allowed in one move."


@pytest.mark.parametrize(
    "args",
    [
        {"forward_m": "nan"},
        {"forward_m": 1.0, "left_m": float("nan")},
        {"forward_m": 1.0, "turn_deg": float("inf")},
    ],
)
def test_move_relative_refuses_values_that_are_not_finite(connector, args):
    # NaN passes the distance limit, because every comparison with it is False.
    result = MoveRelativeTool(connector=connector).invoke(args)
    assert result == "Refused: every value must be a finite number."


def test_move_relative_without_tf_reports_unknown_position(connector):
    tool = MoveRelativeTool(connector=connector, tf_timeout_sec=0.3)
    result = tool.invoke({"forward_m": 1.0})
    assert result.startswith(
        "Cannot move: the robot's position (map -> base_footprint) is unknown."
    )
    # No node publishes TF, as when the agent is on another ROS_DOMAIN_ID.
    assert result.endswith(
        "No node publishes /tf: either the robot is not running, or the agent "
        "uses a different ROS_DOMAIN_ID or Zenoh router than the robot."
    )


def test_move_relative_with_other_tf_skips_the_domain_hint(connector):
    # TF exists but lacks the robot frame, as before localization starts.
    static = StaticTransformBroadcaster(connector.node)
    t = TransformStamped()
    t.header.frame_id = MAP_FRAME
    t.child_frame_id = "odom"
    t.transform.rotation.w = 1.0
    static.sendTransform(t)

    tool = MoveRelativeTool(connector=connector, tf_timeout_sec=0.3)
    result = tool.invoke({"forward_m": 1.0})
    assert "is unknown." in result
    assert "ROS_DOMAIN_ID" not in result


def test_move_relative_zero_move_does_nothing(connector):
    result = MoveRelativeTool(connector=connector).invoke({"forward_m": 0.0})
    assert result == "Nothing to do: the move is zero."
