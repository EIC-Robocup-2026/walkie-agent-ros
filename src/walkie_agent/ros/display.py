"""RosDisplay: ctx.display on the connector's node, as markers for Foxglove or RViz.

show_pick publishes one visualization_msgs/MarkerArray on PICK_MARKERS_TOPIC, latched,
so a viewer that opens later still gets the last plan. The array starts with a
DELETEALL, so each plan replaces the last one. It holds the object's box and name, the
base's stand-off pose as an arrow, each hand pose as axes (x red, y green, z blue) with
its name, and a line through the hand poses in the order the arm takes them.
"""

import logging
import math
import threading
from typing import TYPE_CHECKING

from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from walkie_agent.arm.pick import HandPose, PickPlan
from walkie_agent.capabilities.records import RecalledObject

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

logger = logging.getLogger(__name__)

PICK_MARKERS_TOPIC = "/walkie_agent/pick/markers"
# The object memory's boxes and the stand-off pose are in the map frame.
MAP_FRAME = "map"
LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)
AXIS_M = 0.08
TEXT_M = 0.05
BOX = ColorRGBA(r=1.0, g=0.8, b=0.1, a=0.35)
STANDOFF = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.9)
PATH = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)
TEXT = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
AXES = (
    ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0),
    ColorRGBA(r=0.1, g=1.0, b=0.1, a=1.0),
    ColorRGBA(r=0.2, g=0.4, b=1.0, a=1.0),
)


class RosDisplay:
    """ctx.display for a run with arms. close() destroys the publisher; later draws do nothing."""

    def __init__(self, connector: "ROS2Connector", *, topic: str = PICK_MARKERS_TOPIC) -> None:
        self._node = connector.node
        self._topic = topic
        # Orders a publish against close.
        self._lock = threading.Lock()
        self._closed = False
        self._publisher = self._node.create_publisher(MarkerArray, topic, LATCHED_QOS)

    def show_pick(self, plan: PickPlan, target: RecalledObject, stage: str) -> None:
        try:
            markers = pick_markers(plan, target, stage)
            with self._lock:
                if not self._closed:
                    self._publisher.publish(markers)
        except Exception:
            logger.exception("Could not draw the pick plan on %s.", self._topic)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._node.destroy_publisher(self._publisher)
        except Exception:
            logger.exception("Could not destroy the publisher of %s.", self._topic)


def pick_markers(plan: PickPlan, target: RecalledObject, stage: str) -> MarkerArray:
    """The drawing of `plan` for `target`, after a DELETEALL of the last one."""
    markers = [Marker(action=Marker.DELETEALL)]
    markers[0].header.frame_id = MAP_FRAME
    markers += _object(target, stage)
    standoff = _marker(Marker.ARROW, "standoff", 0, MAP_FRAME, STANDOFF)
    standoff.pose = _pose(plan.standoff_x, plan.standoff_y, 0.02, 0.0, 0.0, plan.heading)
    standoff.scale.x, standoff.scale.y, standoff.scale.z = 0.4, 0.05, 0.05
    markers.append(standoff)
    hands = plan.hand_poses()
    for index, (name, hand) in enumerate(hands):
        markers.append(_axes(index, hand))
        label = _marker(Marker.TEXT_VIEW_FACING, "hand_label", index, hand.frame_id, TEXT)
        label.pose = _pose(hand.x, hand.y, hand.z + 0.04, 0.0, 0.0, 0.0)
        label.scale.z = TEXT_M
        label.text = name
        markers.append(label)
    path = _marker(Marker.LINE_STRIP, "hand_path", 0, hands[0][1].frame_id, PATH)
    path.pose.orientation.w = 1.0
    path.scale.x = 0.008
    path.points = [Point(x=hand.x, y=hand.y, z=hand.z) for _, hand in hands]
    markers.append(path)
    return MarkerArray(markers=markers)


def _object(target: RecalledObject, stage: str) -> list[Marker]:
    """The object's box and its name; none without a centre, no box without a size."""
    if target.x is None or target.y is None or target.z is None:
        return []
    drawn = []
    top = target.z
    if target.size is not None:
        box = _marker(Marker.CUBE, "object", 0, MAP_FRAME, BOX)
        box.pose = _pose(target.x, target.y, target.z, 0.0, 0.0, target.yaw or 0.0)
        box.scale.x, box.scale.y, box.scale.z = (max(side, 0.005) for side in target.size)
        drawn.append(box)
        top += target.size[2] / 2
    label = _marker(Marker.TEXT_VIEW_FACING, "object", 1, MAP_FRAME, TEXT)
    label.pose = _pose(target.x, target.y, top + 0.08, 0.0, 0.0, 0.0)
    label.scale.z = TEXT_M
    label.text = f"{target.name} {target.id}: {stage}"
    drawn.append(label)
    return drawn


def _axes(index: int, hand: HandPose) -> Marker:
    """The hand pose's x, y and z axes as one line list, coloured red, green and blue."""
    axes = _marker(Marker.LINE_LIST, "hand", index, hand.frame_id, AXES[0])
    axes.pose.orientation.w = 1.0
    axes.scale.x = 0.008
    origin = Point(x=hand.x, y=hand.y, z=hand.z)
    for axis, color in zip(_rotation_columns(hand.roll, hand.pitch, hand.yaw), AXES, strict=True):
        tip = Point(
            x=hand.x + AXIS_M * axis[0], y=hand.y + AXIS_M * axis[1], z=hand.z + AXIS_M * axis[2]
        )
        axes.points += [origin, tip]
        axes.colors += [color, color]
    return axes


def _rotation_columns(
    roll: float, pitch: float, yaw: float
) -> tuple[tuple[float, float, float], ...]:
    """The x, y and z axes of tf2's setRPY(roll, pitch, yaw), Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, sy * cp, -sp),
        (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
        (cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr),
    )


def _marker(kind: int, namespace: str, marker_id: int, frame_id: str, color: ColorRGBA) -> Marker:
    marker = Marker(type=kind, ns=namespace, id=marker_id, action=Marker.ADD, color=color)
    marker.header.frame_id = frame_id
    return marker


def _pose(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> Pose:
    """A pose whose orientation is tf2's setRPY(roll, pitch, yaw)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    orientation = Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )
    return Pose(position=Point(x=x, y=y, z=z), orientation=orientation)
