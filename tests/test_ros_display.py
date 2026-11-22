"""RosDisplay and pick_markers: the pick plan as a latched MarkerArray, on the fake node."""

import logging
import math

import pytest
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from ros_fakes import FakeConnector
from visualization_msgs.msg import Marker, MarkerArray

from walkie_agent.arm.pick import HandPose, PickPlan
from walkie_agent.capabilities import RecalledObject
from walkie_agent.ros.display import PICK_MARKERS_TOPIC, RosDisplay, pick_markers

STAGE = "planned before the reach"
AXIS_M = 0.08


def hand(x: float, y: float, z: float, roll=0.0, pitch=0.0, yaw=0.0) -> HandPose:
    return HandPose(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, frame_id="map")


def plan(heading: float = 0.3, **hands: HandPose) -> PickPlan:
    poses = {
        "pregrasp": hand(0.80, 0.0, 0.80),
        "grasp": hand(0.92, 0.0, 0.80),
        "lift": hand(0.92, 0.0, 0.90),
        "clear": hand(0.70, -0.1, 0.90),
    }
    poses.update(hands)
    return PickPlan(standoff_x=0.35, standoff_y=0.2, heading=heading, approach=heading, **poses)


def bottle(**geometry) -> RecalledObject:
    box = {"x": 1.0, "y": 0.0, "z": 0.75, "size": (0.06, 0.08, 0.20), "yaw": 0.5}
    box.update(geometry)
    return RecalledObject(
        id="7",
        name="water bottle",
        label="bottle",
        category="drink",
        room="kitchen room",
        on=None,
        state="present",
        seen_s_ago=1.0,
        **box,
    )


def added(markers: MarkerArray, namespace: str) -> list[Marker]:
    return [m for m in markers.markers if m.action == Marker.ADD and m.ns == namespace]


def point(p) -> tuple[float, float, float]:
    return (p.x, p.y, p.z)


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def quaternion(q) -> tuple[float, float, float, float]:
    return (q.x, q.y, q.z, q.w)


def set_rpy_axes(roll: float, pitch: float, yaw: float) -> list[tuple[float, float, float]]:
    """The columns of Rz(yaw) @ Ry(pitch) @ Rx(roll), multiplied out from the three matrices."""

    def matmul(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    rz = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    ry = [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]]
    rx = [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]]
    r = matmul(matmul(rz, ry), rx)
    return [(r[0][j], r[1][j], r[2][j]) for j in range(3)]


# -- pick_markers


def test_each_drawing_starts_by_deleting_the_last_one():
    markers = pick_markers(plan(), bottle(), STAGE).markers

    assert markers[0].action == Marker.DELETEALL
    assert all(m.action == Marker.ADD for m in markers[1:])


def test_no_two_markers_share_a_namespace_and_id():
    """Foxglove replaces a marker with the same (ns, id), so a clash would hide one."""
    markers = pick_markers(plan(), bottle(), STAGE).markers[1:]

    keys = [(m.ns, m.id) for m in markers]
    assert len(keys) == len(set(keys))


def test_the_object_is_a_box_on_its_centre_with_its_size_and_yaw():
    objects = added(pick_markers(plan(), bottle(), STAGE), "object")
    (box,) = [m for m in objects if m.type == Marker.CUBE]

    assert box.header.frame_id == "map"
    assert point(box.pose.position) == (1.0, 0.0, 0.75)
    assert quaternion(box.pose.orientation) == pytest.approx(yaw_quaternion(0.5))
    assert (box.scale.x, box.scale.y, box.scale.z) == pytest.approx((0.06, 0.08, 0.20))


def test_the_object_is_named_with_its_id_and_the_stage_above_its_box():
    (label,) = [
        m
        for m in added(pick_markers(plan(), bottle(), STAGE), "object")
        if m.type == Marker.TEXT_VIEW_FACING
    ]

    assert label.text == "water bottle 7: planned before the reach"
    # 0.08 m over the box's top, which is half its height over its centre.
    assert point(label.pose.position) == pytest.approx((1.0, 0.0, 0.75 + 0.10 + 0.08))


def test_an_object_with_no_size_is_named_above_its_centre_with_no_box():
    objects = added(pick_markers(plan(), bottle(size=None, yaw=None), STAGE), "object")

    assert [m.type for m in objects] == [Marker.TEXT_VIEW_FACING]
    assert point(objects[0].pose.position) == pytest.approx((1.0, 0.0, 0.83))


def test_an_object_with_no_centre_is_not_drawn_but_its_plan_is():
    markers = pick_markers(plan(), bottle(x=None, y=None, z=None, size=None), STAGE)

    assert added(markers, "object") == []
    assert len(added(markers, "standoff")) == 1
    assert len(added(markers, "hand")) == 4


@pytest.mark.parametrize("heading", [0.0, 0.3, -2.5])
def test_the_stand_off_is_an_arrow_on_the_floor_facing_the_heading(heading):
    (arrow,) = added(pick_markers(plan(heading), bottle(), STAGE), "standoff")

    assert arrow.type == Marker.ARROW
    assert arrow.header.frame_id == "map"
    assert point(arrow.pose.position) == pytest.approx((0.35, 0.2, 0.02))
    assert quaternion(arrow.pose.orientation) == pytest.approx(yaw_quaternion(heading))


def test_each_hand_pose_is_named_axes_in_the_order_the_hand_takes_them():
    markers = pick_markers(plan(), bottle(), STAGE)

    axes = added(markers, "hand")
    labels = added(markers, "hand_label")
    assert [m.type for m in axes] == [Marker.LINE_LIST] * 4
    assert [m.text for m in labels] == ["pregrasp", "grasp", "lift", "clear"]
    for marker in axes:
        assert len(marker.points) == 6
        assert [(c.r, c.g, c.b) for c in marker.colors[::2]] == pytest.approx(
            [(1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.2, 0.4, 1.0)]
        )
    # Each name sits 0.04 m over its pose.
    assert point(labels[1].pose.position) == pytest.approx((0.92, 0.0, 0.84))


@pytest.mark.parametrize(
    "rpy",
    [
        (0.0, 0.0, 0.0),
        (0.0, math.pi / 2, 0.0),
        (0.0, 0.0, math.pi / 2),
        (math.pi / 2, 0.0, 0.0),
        (0.4, -1.1, 2.3),
        (-math.pi, 0.7, -0.2),
    ],
)
def test_the_axes_follow_tf2_set_rpy(rpy):
    """The commander builds the hand's quaternion with tf2's setRPY, Rz(yaw) Ry(pitch) Rx(roll)."""
    grasp = hand(0.92, 0.1, 0.80, *rpy)
    axes = added(pick_markers(plan(grasp=grasp), bottle(), STAGE), "hand")[1]

    origins, tips = axes.points[::2], axes.points[1::2]
    assert all(point(o) == (0.92, 0.1, 0.80) for o in origins)
    for tip, column in zip(tips, set_rpy_axes(*rpy), strict=True):
        expected = tuple(o + AXIS_M * c for o, c in zip((0.92, 0.1, 0.80), column, strict=True))
        assert point(tip) == pytest.approx(expected, abs=1e-9)


def test_a_quarter_pitch_points_the_x_axis_at_the_floor():
    """A quarter turn about y tips the hand's x axis straight down."""
    down = hand(1.0, 0.0, 0.8, pitch=math.pi / 2)
    axes = added(pick_markers(plan(grasp=down), bottle(), STAGE), "hand")

    x_tip = axes[1].points[1]
    assert point(x_tip) == pytest.approx((1.0, 0.0, 0.8 - AXIS_M), abs=1e-9)


def test_a_line_joins_the_hand_poses_in_order():
    (path,) = added(pick_markers(plan(), bottle(), STAGE), "hand_path")

    assert path.type == Marker.LINE_STRIP
    assert path.header.frame_id == "map"
    assert [point(p) for p in path.points] == [
        (0.80, 0.0, 0.80),
        (0.92, 0.0, 0.80),
        (0.92, 0.0, 0.90),
        (0.70, -0.1, 0.90),
    ]


# -- RosDisplay


def test_the_display_publishes_marker_arrays_latched_for_a_late_viewer():
    connector = FakeConnector()
    RosDisplay(connector)

    (publisher,) = connector.node.publishers
    assert publisher.topic == PICK_MARKERS_TOPIC == "/walkie_agent/pick/markers"
    assert publisher.msg_type is MarkerArray
    assert publisher.qos.reliability == ReliabilityPolicy.RELIABLE
    assert publisher.qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert publisher.qos.history == HistoryPolicy.KEEP_LAST
    assert publisher.qos.depth == 1


def test_a_pick_is_published_as_its_markers():
    connector = FakeConnector()
    display = RosDisplay(connector, topic="/picks")

    display.show_pick(plan(), bottle(), STAGE)

    (publisher,) = connector.node.publishers
    assert publisher.topic == "/picks"
    assert publisher.published == [pick_markers(plan(), bottle(), STAGE)]


def test_a_drawing_that_cannot_be_published_is_logged_and_the_pick_goes_on(caplog):
    connector = FakeConnector()
    display = RosDisplay(connector)

    def broken(_message):
        raise RuntimeError("publisher is gone")

    connector.node.publishers[0].on_publish = broken
    with caplog.at_level(logging.ERROR, logger="walkie_agent.ros.display"):
        display.show_pick(plan(), bottle(), STAGE)

    assert "Could not draw the pick plan on /walkie_agent/pick/markers" in caplog.text


def test_nothing_is_drawn_after_close_and_a_second_close_does_nothing():
    connector = FakeConnector()
    display = RosDisplay(connector)
    (publisher,) = connector.node.publishers

    display.close()
    display.close()
    display.show_pick(plan(), bottle(), STAGE)

    assert connector.node.destroyed == [publisher]
    assert publisher.published == []


def test_a_publisher_that_cannot_be_destroyed_is_logged(caplog):
    connector = FakeConnector()
    display = RosDisplay(connector)
    connector.node.destroy_error = RuntimeError("node is gone")

    with caplog.at_level(logging.ERROR, logger="walkie_agent.ros.display"):
        display.close()

    assert "Could not destroy the publisher of /walkie_agent/pick/markers" in caplog.text
