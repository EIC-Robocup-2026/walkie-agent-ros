"""sim_check, the report of what a simulation's graph provides.

The report's text is checked without ROS. The listening is checked on a
private test graph, against a node that publishes what a simulation and the
oracle would, and FakeCommander, which serves the arm commander's names and
types.
"""

import math
import threading
import time

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo
from tf2_ros import StaticTransformBroadcaster
from tf_transformations import euler_matrix
from walkie_vision_msgs.msg import (
    ObjectDetection,
    ObjectDetectionArray,
    PersonDetection,
    PersonDetectionArray,
)

from test_oracle_perception import tf_msg
from test_projection import CAMERA_PITCH_DEG, CAMERA_XYZ, FX, HEIGHT, K, WIDTH, base_from_camera
from walkie_agent.perception import OBJECTS_TOPIC, PEOPLE_TOPIC
from walkie_agent.ros.base import MAP_FRAME, ROBOT_FRAME
from walkie_agent.sim import sim_check
from walkie_agent.sim.fake_commander import CONTROL_GRIPPER, GET_JOINT_STATES, FakeCommander
from walkie_agent.sim.oracle_perception import HEAD_CAMERA_INFO_TOPIC
from walkie_agent.sim.sim_check import (
    AGENT_NODE_NAME,
    Item,
    SimCheck,
    describe_camera,
    format_report,
    person_line,
    wait_for_items,
)

CAMERA_FRAME = "test_camera_optical"
ROBOT_X, ROBOT_Y, ROBOT_YAW_DEG = 1.0, 2.0, 90.0
COMMANDER_SERVERS = (
    "/set_joint_position (action), /control_gripper (action), /get_joint_states (service)"
)
WHOLE_BODY_SERVERS = (
    "/set_joint_position (action), /control_gripper (action), /go_to_pose (action), "
    "/get_joint_states (service)"
)


def test_the_agent_node_name_is_the_agents():
    from walkie_agent.main import NODE_NAME

    assert AGENT_NODE_NAME == NODE_NAME


def test_a_person_line_names_the_gestures_in_order():
    assert person_line(1, {"waving", "standing"}) == "track 1 (standing, waving)"
    assert person_line(3, set()) == "track 3 (no gesture)"


def test_the_camera_line_gives_the_pose_on_the_robot_and_the_pitch():
    info = CameraInfo(width=WIDTH, height=HEIGHT, k=list(K))
    info.header.frame_id = CAMERA_FRAME
    x, y, z = CAMERA_XYZ
    assert describe_camera(info, base_from_camera()) == (
        f"{WIDTH}x{HEIGHT}, fx {FX:.3f}, frame {CAMERA_FRAME}, at ({x:.3f}, {y:.3f}, {z:.3f}) m "
        f"on {ROBOT_FRAME}, {CAMERA_PITCH_DEG:.1f} deg down"
    )
    assert describe_camera(info, None).endswith(f"no transform from {ROBOT_FRAME} yet")


def test_the_report_marks_each_item_and_lines_up_the_names():
    report = format_report([Item("clock", True, "1.0 s"), Item("objects", False, "nothing")])
    assert report.splitlines() == ["ok       clock    1.0 s", "MISSING  objects  nothing"]


def test_the_defaults_wait_two_minutes_without_the_agent_or_the_arms():
    args = sim_check.parse_args([])
    assert (args.timeout, args.agent, args.arms, args.whole_body) == (
        sim_check.DEFAULT_TIMEOUT_S,
        False,
        False,
        False,
    )
    assert sim_check.parse_args(["--agent", "--timeout", "5"]).agent
    assert sim_check.parse_args(["--arms"]).arms
    assert sim_check.parse_args(["--whole-body"]).whole_body


class FakeSimulation:
    """A node that publishes what the simulation and the oracle would, ten times a second."""

    def __init__(self, rclpy_module, *, agent: bool) -> None:
        self.node = rclpy_module.create_node("fake_simulation")
        # The agent's node only has to exist in the graph.
        self.agent = rclpy_module.create_node(AGENT_NODE_NAME) if agent else None
        map_from_robot = euler_matrix(0.0, 0.0, math.radians(ROBOT_YAW_DEG))
        map_from_robot[:3, 3] = (ROBOT_X, ROBOT_Y, 0.0)
        self._tf = StaticTransformBroadcaster(self.node)
        self._tf.sendTransform([
            tf_msg(MAP_FRAME, ROBOT_FRAME, map_from_robot),
            tf_msg(ROBOT_FRAME, CAMERA_FRAME, base_from_camera()),
        ])
        self._publishers = [
            (self.node.create_publisher(Clock, "/clock", 10), self._clock),
            (self.node.create_publisher(CameraInfo, HEAD_CAMERA_INFO_TOPIC, 10), self._camera_info),
            (self.node.create_publisher(PersonDetectionArray, PEOPLE_TOPIC, 10), self._people),
            (self.node.create_publisher(ObjectDetectionArray, OBJECTS_TOPIC, 10), self._objects),
        ]
        self.node.create_timer(0.1, self._publish)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _publish(self) -> None:
        for publisher, make in self._publishers:
            publisher.publish(make())

    @staticmethod
    def _clock() -> Clock:
        msg = Clock()
        msg.clock.sec = 5
        return msg

    @staticmethod
    def _camera_info() -> CameraInfo:
        msg = CameraInfo(width=WIDTH, height=HEIGHT, k=list(K))
        msg.header.frame_id = CAMERA_FRAME
        return msg

    @staticmethod
    def _people() -> PersonDetectionArray:
        # No keypoint is seen, so the person shows no gesture.
        person = PersonDetection(track_id=3)
        person.keypoints = [0.0] * 34
        person.keypoint_scores = [0.0] * 17
        return PersonDetectionArray(people=[person])

    @staticmethod
    def _objects() -> ObjectDetectionArray:
        return ObjectDetectionArray(classes=["mug"], detections=[ObjectDetection(label="mug", track_id=5)])

    def close(self) -> None:
        self._executor.shutdown()
        self._thread.join(timeout=2)
        assert not self._thread.is_alive()
        self.node.destroy_node()
        if self.agent is not None:
            self.agent.destroy_node()


@pytest.fixture
def check_node(ros):
    node = rclpy.create_node("sim_check_under_test")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    yield node, executor
    executor.shutdown()
    node.destroy_node()


def test_every_item_arrives_and_is_described(check_node):
    node, executor = check_node
    simulation = FakeSimulation(rclpy, agent=True)
    commander = FakeCommander()
    check = SimCheck(node, expect_agent=True, expect_arms=True)
    try:
        started = time.monotonic()
        items = wait_for_items(check, executor, timeout_s=10.0)
        elapsed = time.monotonic() - started
    finally:
        check.close()
        commander.destroy_node()
        simulation.close()

    details = {item.name: item.detail for item in items}
    assert all(item.ok for item in items), format_report(items)
    assert details["clock"] == "5.0 s of simulated time"
    assert details["robot"] == f"at ({ROBOT_X:.3f}, {ROBOT_Y:.3f}) in {MAP_FRAME}, yaw {ROBOT_YAW_DEG:.1f} deg"
    assert details["camera"].endswith(f"{CAMERA_PITCH_DEG:.1f} deg down")
    assert details["people"] == "1 in view: track 3 (no gesture)"
    assert details["objects"] == "1 in view: mug (track 5); classes ['mug']"
    assert details["agent"] == f"node {AGENT_NODE_NAME} is up"
    assert details["commander"] == "serves " + COMMANDER_SERVERS
    # It returns once everything has arrived, not at the timeout.
    assert elapsed < 10.0


def test_items_that_never_arrive_are_missing_at_the_timeout(check_node):
    node, executor = check_node
    check = SimCheck(
        node,
        camera_info_topic="/sim_check_test/no_camera_info",
        people_topic="/sim_check_test/no_people",
        objects_topic="/sim_check_test/no_objects",
    )
    started = time.monotonic()
    items = wait_for_items(check, executor, timeout_s=0.5)
    assert time.monotonic() - started >= 0.5

    missing = {item.name: item.detail for item in items if not item.ok}
    assert missing["camera"] == "nothing on /sim_check_test/no_camera_info"
    assert missing["people"] == "nothing on /sim_check_test/no_people"
    assert missing["objects"] == "nothing on /sim_check_test/no_objects"
    assert {"agent", "commander"}.isdisjoint(item.name for item in items)


def test_an_empty_people_array_still_counts_as_arrived(check_node):
    node, _ = check_node
    check = SimCheck(node)
    check._on_people(PersonDetectionArray())
    people = next(item for item in check.items() if item.name == "people")
    assert people == Item("people", True, "0 people in view")


@pytest.fixture
def arms_check(check_node):
    node, _ = check_node
    check = SimCheck(node, expect_arms=True)
    yield check
    check.close()


def commander_becomes(check: SimCheck, expected: Item, timeout_s: float = 10.0) -> None:
    """Wait until the commander item is `expected`, as discovery catches up with the graph."""
    deadline = time.monotonic() + timeout_s
    while (item := next(i for i in check.items() if i.name == "commander")) != expected:
        assert time.monotonic() < deadline, f"the commander item stayed {item}"
        time.sleep(0.05)


def test_without_the_commander_the_item_names_every_server_it_misses(arms_check):
    commander_becomes(arms_check, Item("commander", False, "no server for " + COMMANDER_SERVERS))


def item_becomes(check: SimCheck, expected: Item, timeout_s: float = 10.0) -> None:
    """Wait until the item named as `expected` is `expected`, as discovery catches up."""
    deadline = time.monotonic() + timeout_s
    while (item := next(i for i in check.items() if i.name == expected.name)) != expected:
        assert time.monotonic() < deadline, f"the {expected.name} item stayed {item}"
        time.sleep(0.05)


def test_the_whole_body_check_waits_for_move_group_and_the_go_to_pose_action(check_node):
    node, _ = check_node
    check = SimCheck(node, expect_whole_body=True)
    try:
        assert [item.name for item in check.items()][-2:] == ["move_group", "commander"]
        item_becomes(check, Item("move_group", False, "no node named move_group"))
        item_becomes(check, Item("commander", False, "no server for " + WHOLE_BODY_SERVERS))
        move_group = rclpy.create_node(sim_check.MOVE_GROUP_NODE_NAME)
        commander = FakeCommander()
        try:
            item_becomes(check, Item("move_group", True, "node move_group is up"))
            item_becomes(check, Item("commander", True, "serves " + WHOLE_BODY_SERVERS))
        finally:
            commander.destroy_node()
            move_group.destroy_node()
    finally:
        check.close()


def test_the_commander_item_names_the_servers_that_go(arms_check):
    commander = FakeCommander()
    try:
        commander_becomes(arms_check, Item("commander", True, "serves " + COMMANDER_SERVERS))
        commander.vanish(CONTROL_GRIPPER, GET_JOINT_STATES)
        commander_becomes(
            arms_check,
            Item("commander", False, "no server for /control_gripper (action), /get_joint_states (service)"),
        )
    finally:
        commander.destroy_node()
