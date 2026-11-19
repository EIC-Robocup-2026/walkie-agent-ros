"""Check that a simulation's ROS graph has what the agent and the oracle need.

    scripts/chula-sim.sh check [--agent] [--arms | --whole-body]
    python -m walkie_agent.sim.sim_check [--timeout 120] [--agent] [--arms | --whole-body]

It joins the graph it is started in, and chula-sim.sh starts it in the
simulation's private graph. It listens until every item below has arrived,
or until the timeout, then prints one line per item and exits 0 only when
every item arrived:

- clock: /clock, past zero.
- robot: the transform from map to base_footprint, which AMCL provides.
- camera: the head camera's camera_info, and the camera's pose on the robot.
- people: walkie-vision's people array, with each person's gestures.
- objects: walkie-vision's objects array, with each detection's label.
- agent: the agent's node, with --agent only.
- move_group: MoveIt's node, with --whole-body only.
- commander: a server for each of the arm commander's /set_joint_position
  and /control_gripper actions and its /get_joint_states service, with
  --arms or --whole-body, and for its /go_to_pose action with --whole-body,
  which pick_object reaches through.

An array with nobody in it still counts as arrived: who is in view depends
on where the robot looks. It publishes nothing and sends no request: the
commander item looks for the servers, as CommanderClient.available() does,
through clients of the same names and types that it never calls. So, like
available(), it cannot tell a commander built from another
my_robot_interfaces, which rmw_zenoh_cpp finds but never delivers a request
to (commander.STALE_PIN_HINT).
"""

import argparse
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import rclpy
from my_robot_interfaces.action import ControlGripper, GoToPose, SetJointPosition
from my_robot_interfaces.srv import GetJointStates
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformException, TransformListener
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetectionArray

from walkie_agent.perception import OBJECTS_TOPIC, PEOPLE_TOPIC
from walkie_agent.perception.gestures import classify_gestures
from walkie_agent.ros.base import MAP_FRAME, ROBOT_FRAME
from walkie_agent.ros.commander import (
    CONTROL_GRIPPER_ACTION,
    GET_JOINT_STATES_SERVICE,
    GO_TO_POSE_ACTION,
    SET_JOINT_POSITION_ACTION,
)
from walkie_agent.sim.oracle_perception import HEAD_CAMERA_INFO_TOPIC, transform_matrix

NODE_NAME = "sim_check"

# walkie_agent.main.NODE_NAME. Importing main would load RAI and LangChain for
# one string; tests/test_sim_check.py checks that the two agree.
AGENT_NODE_NAME = "walkie_agent"
# The node that MoveIt's move_group executable starts.
MOVE_GROUP_NODE_NAME = "move_group"

CHECK_FAILED_EXIT_CODE = 1
DEFAULT_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class Item:
    """One line of the report: whether the item arrived, and what it showed."""

    name: str
    ok: bool
    detail: str


def person_line(track_id: int, gestures: set[str]) -> str:
    """One person of the people item, such as 'track 1 (standing, waving)'."""
    shown = ", ".join(sorted(gestures)) if gestures else "no gesture"
    return f"track {track_id} ({shown})"


def describe_people(msg: PersonDetectionArray) -> str:
    """The people item's detail: how many are in view, and their gestures."""
    if not msg.people:
        return "0 people in view"
    lines = [
        person_line(p.track_id, classify_gestures(p.keypoints, p.keypoint_scores))
        for p in msg.people
    ]
    return f"{len(msg.people)} in view: " + "; ".join(lines)


def describe_objects(msg: ObjectDetectionArray) -> str:
    """The objects item's detail: each detection's label and track."""
    found = ", ".join(f"{d.label} (track {d.track_id})" for d in msg.detections)
    return f"{len(msg.detections)} in view: {found or 'none'}; classes {list(msg.classes)}"


def describe_camera(info: CameraInfo, robot_from_camera: np.ndarray | None) -> str:
    """The camera item's detail: the image, fx, and the camera's pose on the robot."""
    detail = f"{info.width}x{info.height}, fx {info.k[0]:.3f}, frame {info.header.frame_id}"
    if robot_from_camera is None:
        return detail + f", no transform from {ROBOT_FRAME} yet"
    x, y, z = robot_from_camera[:3, 3]
    # An optical frame looks along its z axis; the pitch is how far it points down.
    forward = robot_from_camera[:3, 2]
    pitch_down = math.degrees(math.asin(max(-1.0, min(1.0, -forward[2]))))
    return detail + f", at ({x:.3f}, {y:.3f}, {z:.3f}) m on {ROBOT_FRAME}, {pitch_down:.1f} deg down"


class SimCheck:
    """Listens to one graph and keeps the newest of every item.

    close() destroys the action clients that expect_arms and expect_whole_body
    create, which the node's destroy_node() leaves behind; call it before
    destroying the node. expect_whole_body implies expect_arms.
    """

    def __init__(
        self,
        node: Node,
        *,
        camera_info_topic: str = HEAD_CAMERA_INFO_TOPIC,
        people_topic: str = PEOPLE_TOPIC,
        objects_topic: str = OBJECTS_TOPIC,
        expect_agent: bool = False,
        expect_arms: bool = False,
        expect_whole_body: bool = False,
    ) -> None:
        self._node = node
        self._expect_agent = expect_agent
        self._expect_arms = expect_arms or expect_whole_body
        self._expect_whole_body = expect_whole_body
        # The commander's servers, each with what tells whether it is discovered.
        self._actions: list[ActionClient] = []
        self._servers: list[tuple[str, Callable[[], bool]]] = []
        if self._expect_arms:
            actions = [
                (SetJointPosition, SET_JOINT_POSITION_ACTION),
                (ControlGripper, CONTROL_GRIPPER_ACTION),
            ]
            if expect_whole_body:
                actions.append((GoToPose, GO_TO_POSE_ACTION))
            for action_type, name in actions:
                action = ActionClient(node, action_type, name)
                self._actions.append(action)
                self._servers.append((f"{name} (action)", action.server_is_ready))
            service = node.create_client(GetJointStates, GET_JOINT_STATES_SERVICE)
            self._servers.append((f"{GET_JOINT_STATES_SERVICE} (service)", service.service_is_ready))
        self._topics = {
            "camera": camera_info_topic,
            "people": people_topic,
            "objects": objects_topic,
        }
        self._clock: Clock | None = None
        self._camera_info: CameraInfo | None = None
        self._people: PersonDetectionArray | None = None
        self._objects: ObjectDetectionArray | None = None
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, node, spin_thread=False)
        # Sensor-data QoS is BEST_EFFORT, which matches reliable publishers too.
        qos = qos_profile_sensor_data
        node.create_subscription(Clock, "/clock", self._on_clock, qos)
        node.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, qos)
        node.create_subscription(PersonDetectionArray, people_topic, self._on_people, qos)
        node.create_subscription(ObjectDetectionArray, objects_topic, self._on_objects, qos)

    def _on_clock(self, msg: Clock) -> None:
        self._clock = msg

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _on_people(self, msg: PersonDetectionArray) -> None:
        self._people = msg

    def _on_objects(self, msg: ObjectDetectionArray) -> None:
        self._objects = msg

    def _transform(self, target: str, source: str) -> np.ndarray | None:
        try:
            return transform_matrix(self._buffer.lookup_transform(target, source, Time()))
        except TransformException:
            return None

    def items(self) -> list[Item]:
        """The report as it stands now."""
        items = [self._clock_item(), self._robot_item(), self._camera_item()]
        items.append(self._array_item("people", self._people, describe_people))
        items.append(self._array_item("objects", self._objects, describe_objects))
        if self._expect_agent:
            items.append(self._agent_item())
        if self._expect_whole_body:
            items.append(self._move_group_item())
        if self._expect_arms:
            items.append(self._commander_item())
        return items

    def close(self) -> None:
        """Destroy the commander item's action clients, once the report is taken.

        items() must not be called after it. The node stays.
        """
        for action in self._actions:
            action.destroy()
        self._actions.clear()

    def _clock_item(self) -> Item:
        clock = self._clock
        if clock is None:
            return Item("clock", False, "nothing on /clock")
        seconds = clock.clock.sec + clock.clock.nanosec * 1e-9
        if seconds <= 0.0:
            return Item("clock", False, "/clock is still at zero")
        return Item("clock", True, f"{seconds:.1f} s of simulated time")

    def _robot_item(self) -> Item:
        pose = self._transform(MAP_FRAME, ROBOT_FRAME)
        if pose is None:
            return Item("robot", False, f"no transform from {MAP_FRAME} to {ROBOT_FRAME}")
        yaw = math.degrees(math.atan2(pose[1, 0], pose[0, 0]))
        return Item(
            "robot", True, f"at ({pose[0, 3]:.3f}, {pose[1, 3]:.3f}) in {MAP_FRAME}, yaw {yaw:.1f} deg"
        )

    def _camera_item(self) -> Item:
        info = self._camera_info
        if info is None:
            return Item("camera", False, f"nothing on {self._topics['camera']}")
        robot_from_camera = self._transform(ROBOT_FRAME, info.header.frame_id)
        return Item("camera", robot_from_camera is not None, describe_camera(info, robot_from_camera))

    def _array_item(self, name, msg, describe) -> Item:
        if msg is None:
            return Item(name, False, f"nothing on {self._topics[name]}")
        return Item(name, True, describe(msg))

    def _agent_item(self) -> Item:
        if AGENT_NODE_NAME in self._node.get_node_names():
            return Item("agent", True, f"node {AGENT_NODE_NAME} is up")
        return Item("agent", False, f"no node named {AGENT_NODE_NAME}")

    def _move_group_item(self) -> Item:
        if MOVE_GROUP_NODE_NAME in self._node.get_node_names():
            return Item("move_group", True, f"node {MOVE_GROUP_NODE_NAME} is up")
        return Item("move_group", False, f"no node named {MOVE_GROUP_NODE_NAME}")

    def _commander_item(self) -> Item:
        missing = [name for name, discovered in self._servers if not discovered()]
        if missing:
            return Item("commander", False, "no server for " + ", ".join(missing))
        return Item("commander", True, "serves " + ", ".join(name for name, _ in self._servers))


def wait_for_items(
    check: SimCheck, executor: SingleThreadedExecutor, timeout_s: float
) -> list[Item]:
    """Spin until every item has arrived or `timeout_s` of wall time passes."""
    deadline = time.monotonic() + timeout_s
    while True:
        executor.spin_once(timeout_sec=0.1)
        items = check.items()
        if all(item.ok for item in items) or time.monotonic() >= deadline:
            return items


def format_report(items: Sequence[Item]) -> str:
    width = max(len(item.name) for item in items)
    return "\n".join(
        f"{'ok' if item.ok else 'MISSING':7}  {item.name:{width}}  {item.detail}" for item in items
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sim_check",
        description="Wait for a simulation's clock, TF, head camera and vision arrays.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Seconds of wall time to wait for every item (default {DEFAULT_TIMEOUT_S:.0f})",
    )
    parser.add_argument(
        "--agent", action="store_true", help=f"Also wait for the agent's node, {AGENT_NODE_NAME}"
    )
    parser.add_argument(
        "--arms",
        action="store_true",
        help="Also wait for the arm commander's two actions and its joint state service",
    )
    parser.add_argument(
        "--whole-body",
        action="store_true",
        help="Also wait for move_group and the commander's servers, /go_to_pose among them",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    check: SimCheck | None = None
    try:
        check = SimCheck(
            node,
            expect_agent=args.agent,
            expect_arms=args.arms,
            expect_whole_body=args.whole_body,
        )
        items = wait_for_items(check, executor, args.timeout)
    except KeyboardInterrupt:
        return CHECK_FAILED_EXIT_CODE
    finally:
        if check is not None:
            check.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    print(format_report(items))
    return 0 if all(item.ok for item in items) else CHECK_FAILED_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
