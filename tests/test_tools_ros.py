"""Walkie tools against the fake robot over real ROS 2 (no LLM)."""

import time

import pytest
from geometry_msgs.msg import Twist
from rai.tools.ros2.navigation.nav2_blocking import GetCurrentPoseTool
from std_msgs.msg import Empty, String

from walkie_agent.locations import Location
from walkie_agent.tools import NavigateToLocationTool, SpeakTool, build_walkie_tools


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


def test_tools_hide_forbidden_topics(connector, locations):
    node = connector.node
    node.create_publisher(Twist, "/cmd_vel", 10)
    node.create_publisher(Empty, "/walkie/reset", 10)
    node.create_publisher(String, "/walkie/visible_test_topic", 10)
    lister = next(
        t for t in build_walkie_tools(connector, locations)
        if t.name == "get_ros2_topics_names_and_types"
    )

    assert wait_until(lambda: "/walkie/visible_test_topic" in lister.invoke({}))
    topics = lister.invoke({})
    assert "/cmd_vel" not in topics
    assert "/walkie/reset" not in topics


def test_rai_get_current_pose_reads_fake_tf(fake_walkie, connector):
    tool = GetCurrentPoseTool(connector=connector, frame_id="map", robot_frame_id="base_link")
    assert "frame_id='map'" in tool.invoke({})
