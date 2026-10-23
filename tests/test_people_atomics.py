"""The people and seat atomics end to end over ROS (docs/architecture/hri.md, Atomics).

A fake walkie-vision node publishes PersonDetectionArray and
ObjectDetectionArray as walkie-vision does, with hash_vector vectors under
its fake model names, as oracle_perception will. The same node publishes
the robot's map -> base_footprint -> camera transforms on /tf_static, and
the base's on /tf where a test needs its pose at an earlier stamp, so the
headings and the handles' motion checks read real TF. The atomics run
through a Toolbox on the `connector` fixture's node, on Fast DDS and a
random domain (ADR-0019).
"""

import dataclasses
import math
import random
import re
import threading
import time

import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped
from pydantic import ValidationError
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from tf2_ros import LookupException, TransformException
from tf_transformations import quaternion_from_euler
from walkie_vision_msgs.msg import (
    ObjectDetection,
    ObjectDetectionArray,
    PersonDetection,
    PersonDetectionArray,
)

from walkie_agent.perception.gestures import COCO_KEYPOINTS, GESTURES
from walkie_agent.perception.people_memory import InMemoryPeopleMemory, ModelVector
from walkie_agent.perception.seating import SeatingConfig
from walkie_agent.perception.vision_feed import (
    OBJECTS_TOPIC,
    PEOPLE_TOPIC,
    TrackKey,
    VisionFeed,
    VisionFeedConfig,
)
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, FAKE_REID_MODEL, hash_vector
from walkie_agent.toolbox import StepError, Toolbox
from walkie_agent.tools.atomic import _perception
from walkie_agent.tools.navigation import MAP_FRAME, ROBOT_FRAME

CAMERA_FRAME = "head_camera_optical_frame"
WIDTH, HEIGHT = 640, 480
# The focal length and the principal point's column, so the ray through
# column u is atan((CX - u) / FX) to the left of the camera's axis.
FX, CX = 320.0, 320.0
# A face side face_trusted rejects: 0.0469 rad is 15 px at FX.
SMALL_FACE = 12.0
# A face side face_trusted accepts at FX, 0.094 rad, though under v2's
# 50 x 50 px: the gate is an angle, not a pixel area.
SMALL_BUT_NEAR_FACE = 30.0
# The base's yaw in the map, not 0, so a heading is checked in the map frame.
BASE_YAW = 0.3
# The camera's optical frame in the base's, as roll, pitch and yaw: z out of
# the lens and x to the image's right, looking ahead, or looking straight
# down with the image's top ahead.
CAMERA_AHEAD = (-math.pi / 2, 0.0, -math.pi / 2)
CAMERA_DOWN = (math.pi, 0.0, -math.pi / 2)
# A face model with a distance limit other than the fake one, as walkie-vision
# would switch to.
ARCFACE_MODEL = "arcface_w600k_r50"
DIM = 512
NO_CANCEL = threading.Event()


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def heading_to(u: float, base_yaw: float = BASE_YAW) -> float:
    return base_yaw + math.atan((CX - u) / FX)


# -- The scene ---------------------------------------------------------------------


def vector(kind: str, key: str, shot: int = 0) -> np.ndarray:
    """The `kind` ("face" or "reid") vector of person `key`, as seen in image `shot`.

    Shot 0 is the person's own vector; a later shot nudges it a little, as a
    second image of one face never gives quite the same vector. Hash
    vectors of two people are nearly orthogonal.
    """
    base = hash_vector(f"{kind} {key}", DIM)
    if shot == 0:
        return base
    nudged = base + 0.1 * hash_vector(f"{kind} {key} shot {shot}", DIM)
    return (nudged / np.linalg.norm(nudged)).astype(np.float32)


def keypoints(cx: float, top: float, height: float, pose: str) -> dict[str, tuple]:
    """COCO keypoints of a person `height` px tall: standing, sitting or waving.

    The person faces the camera, so their left side is on the image's right.
    Standing: legs extended below the hips. Sitting: thighs level, shins
    short. Waving: standing, with the right wrist above the head.
    """
    h = height
    sitting = pose == "sitting"
    head = top + 0.08 * h
    shoulder = top + (0.15 if sitting else 0.2) * h
    hip = top + 0.5 * h
    knee = top + (0.55 if sitting else 0.72) * h
    ankle = top + (0.8 if sitting else 0.95) * h
    half_shoulders, half_hips = 0.08 * h, 0.06 * h
    points = {
        "nose": (cx, head),
        "left_eye": (cx + 0.02 * h, head - 0.01 * h),
        "right_eye": (cx - 0.02 * h, head - 0.01 * h),
        "left_ear": (cx + 0.04 * h, head),
        "right_ear": (cx - 0.04 * h, head),
        "left_shoulder": (cx + half_shoulders, shoulder),
        "right_shoulder": (cx - half_shoulders, shoulder),
        "left_elbow": (cx + half_shoulders, (shoulder + hip) / 2),
        "right_elbow": (cx - half_shoulders, (shoulder + hip) / 2),
        # Hanging arms: the wrists at the hips.
        "left_wrist": (cx + half_shoulders, hip),
        "right_wrist": (cx - half_shoulders, hip),
        "left_hip": (cx + half_hips, hip),
        "right_hip": (cx - half_hips, hip),
        "left_knee": (cx + half_hips, knee),
        "right_knee": (cx - half_hips, knee),
        "left_ankle": (cx + half_hips, ankle),
        "right_ankle": (cx - half_hips, ankle),
    }
    if pose == "waving":
        points["right_elbow"] = (cx - half_shoulders, shoulder - 0.1 * h)
        points["right_wrist"] = (cx - half_shoulders, shoulder - 0.25 * h)
    return points


def person(
    cx: float,
    key: str | None = None,
    *,
    track_id: int = 0,
    pose: str = "standing",
    top: float = 40.0,
    height: float = 400.0,
    shot: int = 0,
    face_side: float = 60.0,
    face_score: float = 0.9,
    face_dx: float = 0.0,
    face_key: str | None = None,
    face_model: str = FAKE_FACE_MODEL,
    appearance: bool = True,
) -> PersonDetection:
    """A person whose box is centred on column `cx`, with the vectors of `key`.

    No key means no vectors at all. `face_side` 0 means no face found in
    the box; below 15 px (0.0469 rad at FX), or below a `face_score` of
    0.5, the face is found but not trusted. `face_key` gives the face
    another person's vector, as a bystander caught in the box. Without
    `appearance`, the person has no appearance vector.
    """
    # rclpy takes an int for a float64 field without complaint but
    # serializes it as garbage, so every coordinate is made a float first.
    cx, top, height = float(cx), float(top), float(height)
    msg = PersonDetection(track_id=track_id, score=0.9)
    msg.bbox.center.position.x, msg.bbox.center.position.y = cx, top + height / 2
    msg.bbox.size_x, msg.bbox.size_y = 0.4 * height, height
    points = keypoints(cx, top, height, pose)
    msg.keypoints = [value for name in COCO_KEYPOINTS for value in points[name]]
    msg.keypoint_scores = [0.9] * len(COCO_KEYPOINTS)
    if face_side > 0:
        msg.face_bbox.center.position.x = cx + face_dx
        msg.face_bbox.center.position.y = top + 0.08 * height
        msg.face_bbox.size_x = msg.face_bbox.size_y = face_side
        msg.face_det_score = face_score
        if key is not None:
            msg.face_model = face_model
            msg.face_embedding = vector("face", face_key or key, shot).tolist()
    if key is not None and appearance:
        msg.appearance_model = FAKE_REID_MODEL
        msg.appearance_embedding = vector("reid", key, shot).tolist()
    return msg


def thing(label: str, x1: float, y1: float, x2: float, y2: float) -> ObjectDetection:
    x1, y1, x2, y2 = (float(v) for v in (x1, y1, x2, y2))
    msg = ObjectDetection(label=label, score=0.8)
    msg.bbox.center.position.x, msg.bbox.center.position.y = (x1 + x2) / 2, (y1 + y2) / 2
    msg.bbox.size_x, msg.bbox.size_y = x2 - x1, y2 - y1
    return msg


def transform(parent: str, child: str, xyz, rpy, stamp: TimeMsg) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
    q = quaternion_from_euler(*rpy)
    t.transform.rotation.x, t.transform.rotation.y = q[0], q[1]
    t.transform.rotation.z, t.transform.rotation.w = q[2], q[3]
    return t


class FakeVision:
    """A second node that publishes what walkie-vision and the robot's TF would."""

    def __init__(self, connector, feed: VisionFeed) -> None:
        self.clock_node = connector.node
        self.feed = feed
        self.node = rclpy.create_node(f"fake_vision_{random.randint(0, 1_000_000)}")
        self.people_pub = self.node.create_publisher(PersonDetectionArray, PEOPLE_TOPIC, 10)
        self.objects_pub = self.node.create_publisher(ObjectDetectionArray, OBJECTS_TOPIC, 10)
        # Published directly rather than with tf2_ros's
        # StaticTransformBroadcaster, which keeps the first transform it was
        # given for a child frame, so it could not move the base. The buffer
        # replaces a static transform with a newer one of the same frames.
        self.tf_pub = self.node.create_publisher(
            TFMessage,
            "/tf_static",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        # For a base that drives: the buffer keeps a history of /tf, so a
        # lookup at an image's stamp finds the pose at that stamp, where a
        # static transform holds only the newest pose.
        self.tf_dynamic_pub = self.node.create_publisher(
            TFMessage, "/tf", QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )
        self._count = 0

    def stamp(self) -> TimeMsg:
        """Now on the agent's clock, unique per call."""
        # The nanosecond counter keeps two stamps made in the same tick apart.
        self._count += 1
        now_ns = self.clock_node.get_clock().now().nanoseconds + self._count
        return TimeMsg(sec=now_ns // 1_000_000_000, nanosec=now_ns % 1_000_000_000)

    def wait_for_subscribers(self) -> None:
        assert wait_until(
            lambda: self.people_pub.get_subscription_count() > 0
            and self.objects_pub.get_subscription_count() > 0
        ), "VisionFeed's subscriptions were not discovered."

    def place_base(
        self, ctx, x: float = 0.0, yaw: float = BASE_YAW, camera=CAMERA_AHEAD
    ) -> None:
        """Put the base at (x, 0) with `yaw` in the map, and wait until ctx sees it.

        `camera` is the optical frame's roll, pitch and yaw in the base's.
        """
        stamp = self.stamp()
        camera_transform = transform("base_footprint", CAMERA_FRAME, (0.1, 0.0, 1.2), camera, stamp)
        self.tf_pub.publish(
            TFMessage(
                transforms=[
                    transform("map", "base_footprint", (x, 0.0, 0.0), (0.0, 0.0, yaw), stamp),
                    camera_transform,
                ]
            )
        )

        def placed() -> bool:
            try:
                pose = ctx.robot_pose(0.1)
                mounted = ctx.tf_buffer.lookup_transform("base_footprint", CAMERA_FRAME, Time())
            except (StepError, TransformException):
                return False
            q, want = mounted.transform.rotation, camera_transform.transform.rotation
            return (
                abs(pose.x - x) < 1e-6
                and abs(pose.yaw - yaw) < 1e-6
                and abs(q.x * want.x + q.y * want.y + q.z * want.z + q.w * want.w) > 1 - 1e-9
            )

        assert wait_until(placed), "The base's transform did not reach the agent's TF buffer."

    def drive_base(self, ctx, xs: list[float]) -> None:
        """Publish the base at each x of `xs` on /tf, in order and each with a newer stamp.

        Waits until ctx sees the last. The first transform on /tf turns the
        base's frame from static to dynamic in the buffer.
        """
        assert wait_until(lambda: self.tf_dynamic_pub.get_subscription_count() > 0)
        self.tf_dynamic_pub.publish(
            TFMessage(
                transforms=[
                    transform(
                        "map", "base_footprint", (x, 0.0, 0.0), (0.0, 0.0, BASE_YAW), self.stamp()
                    )
                    for x in xs
                ]
            )
        )

        def driven() -> bool:
            try:
                return abs(ctx.robot_pose(0.1).x - xs[-1]) < 1e-6
            except StepError:
                return False

        assert wait_until(driven), "The base's transform did not reach the agent's TF buffer."

    def show(self, people=None, objects=None, classes=None) -> TimeMsg:
        """Publish one image's arrays, objects first, and return once the feed has them.

        One image at a time: the feed's subscription keeps only 5 arrays, so
        a burst could lose some. None leaves that array out of the image.
        """
        stamp = self.stamp()
        if objects is not None:
            msg = ObjectDetectionArray(
                classes=sorted({o.label for o in objects}) if classes is None else classes,
                detections=list(objects),
            )
            self._fill(msg, stamp)
            self.objects_pub.publish(msg)
            assert wait_until(lambda: self._objects_arrived(stamp))
        if people is not None:
            msg = PersonDetectionArray(people=list(people))
            self._fill(msg, stamp)
            self.people_pub.publish(msg)
            self.feed.people_at(stamp, 3.0, NO_CANCEL)
        return stamp

    def close(self) -> None:
        self.node.destroy_node()

    @staticmethod
    def _fill(msg, stamp: TimeMsg) -> None:
        msg.header.stamp = stamp
        msg.header.frame_id = CAMERA_FRAME
        msg.camera_info.width, msg.camera_info.height = WIDTH, HEIGHT
        msg.camera_info.k = [FX, 0.0, CX, 0.0, FX, HEIGHT / 2, 0.0, 0.0, 1.0]

    def _objects_arrived(self, stamp: TimeMsg) -> bool:
        try:
            return self.feed.latest_objects(60.0, 0.0, NO_CANCEL).header.stamp == stamp
        except StepError:
            return False


class World:
    """The atomics, their ctx and backends, and the fake that feeds them."""

    def __init__(self, toolbox: Toolbox, vision: FakeVision, memory: InMemoryPeopleMemory):
        self.toolbox = toolbox
        self.ctx = toolbox.ctx
        self.vision = vision
        self.memory = memory
        self.tools = {tool.name: tool for tool in toolbox.tools()}

    # Positional only, since remember_person takes a `name` of its own.
    def call(self, tool: str, /, **kwargs):
        return self.tools[tool].call(**kwargs)

    def fails(self, tool: str, /, **kwargs) -> StepError:
        with pytest.raises(StepError) as error:
            self.call(tool, **kwargs)
        return error.value


@pytest.fixture
def make_world(connector):
    """A World factory: the keyword arguments go to the feed's VisionFeedConfig."""
    made: list[FakeVision] = []

    def make(**feed_config) -> World:
        feed = VisionFeed(connector, VisionFeedConfig(**feed_config))
        memory = InMemoryPeopleMemory()
        toolbox = Toolbox(connector, {"vision": feed, "people_memory": memory})
        vision = FakeVision(connector, feed)
        made.append(vision)
        vision.wait_for_subscribers()
        vision.place_base(toolbox.ctx)
        return World(toolbox, vision, memory)

    yield make
    for vision in made:
        vision.close()


@pytest.fixture
def world(make_world):
    return make_world()


def spy_on_put_set(world, monkeypatch) -> list[str]:
    """The kinds of the handle sets stored from now on, in order."""
    stored: list[str] = []
    put_set = world.ctx.handles.put_set

    def spy(kind, payloads, pose=None):
        stored.append(kind)
        return put_set(kind, payloads, pose)

    monkeypatch.setattr(world.ctx.handles, "put_set", spy)
    return stored


def lines(out) -> list[str]:
    return out.summary.splitlines()


# -- Loading -----------------------------------------------------------------------------


def test_the_people_tools_do_not_load_without_people_memory(connector):
    vision_only = Toolbox(connector, {"vision": VisionFeed(connector)})
    both = Toolbox(
        connector, {"vision": VisionFeed(connector), "people_memory": InMemoryPeopleMemory()}
    )

    assert [tool.name for tool in vision_only.tools()] == ["find_people", "find_seats"]
    assert [tool.name for tool in both.tools()] == [
        "find_people",
        "find_seats",
        "recognize_person",
        "remember_person",
    ]


# -- find_people -------------------------------------------------------------------------


def test_find_people_lists_everyone_left_to_right(world):
    world.vision.show(
        people=[
            person(530, "carol", track_id=3, pose="sitting", top=140, height=288),
            person(100, "ana", track_id=1, pose="waving"),
            person(320, "sam", track_id=2, top=200, height=200, face_side=0),
        ]
    )

    out = world.call("find_people")

    s = out.set_id
    assert re.fullmatch(r"per_[0-9a-f]{4}", s)
    assert out.handles == [f"{s}.0", f"{s}.1", f"{s}.2"]
    assert lines(out) == [
        f"{s}: 3 people in view, left to right.",
        f"{s}.0: left, height 0.83, standing, waving, raising_right_arm, face seen",
        f"{s}.1: centre, height 0.42, standing, no face seen",
        f"{s}.2: right, height 0.60, sitting, face seen",
    ]


def test_a_person_handle_keeps_the_track_and_the_heading_to_the_face(world):
    world.vision.show(
        people=[
            # The face is off the body's centre, so the heading shows which box it used.
            person(160, "ana", track_id=4, face_dx=40),
            person(480, "sam", track_id=5, face_side=0),
            person(560, "bob", track_id=6, face_side=SMALL_FACE),
        ]
    )

    out = world.call("find_people")

    ana, sam, bob = (world.ctx.handles.get(h, kind="person").payload for h in out.handles)
    assert isinstance(ana.track, TrackKey) and ana.track.track_id == 4
    assert ana.heading_map == pytest.approx(heading_to(200), abs=1e-4)
    assert ana.bbox == pytest.approx((80.0, 40.0, 240.0, 440.0))
    assert ana.face_bbox == pytest.approx((170.0, 42.0, 230.0, 102.0))
    assert ana.face.model == FAKE_FACE_MODEL
    assert np.allclose(ana.face.vector, vector("face", "ana"))
    assert ana.appearance.model == FAKE_REID_MODEL
    assert np.allclose(ana.appearance.vector, vector("reid", "ana"))
    assert ana.gestures == frozenset({"standing"})
    assert len(ana.keypoints) == 34 and len(ana.keypoint_scores) == 17
    assert (ana.header.frame_id, ana.image_width, ana.image_height) == (CAMERA_FRAME, 640, 480)
    # Without a face, the heading is to the body.
    assert sam.face is None and sam.face_bbox is None
    assert sam.heading_map == pytest.approx(heading_to(480), abs=1e-4)
    # A face too small to trust is not kept, and the line says so.
    assert bob.face is None and bob.face_bbox is not None
    assert lines(out)[3].endswith(", face too small or unclear to recognize")


def test_find_people_says_whether_a_face_was_seen_well_enough_to_recognize(world):
    world.vision.show(
        people=[
            # Under 50 x 50 px, but a face is judged by its angle.
            person(64, "ana", track_id=1, face_side=SMALL_BUT_NEAR_FACE),
            # Found, but smaller than the angle face_trusted asks for.
            person(192, "sam", track_id=2, face_side=SMALL_FACE),
            # Found, but under the detection score that face_trusted asks for.
            person(320, "kim", track_id=3, face_score=0.3),
            # Found and trusted, but with no face vector to recognize it by.
            person(448, None, track_id=5),
            # Turned away: no face in the box.
            person(576, "bob", track_id=4, face_side=0),
        ]
    )

    out = world.call("find_people")

    # The face is the last part of each person's line.
    assert [line.rsplit(", ", 1)[1] for line in lines(out)[1:]] == [
        "face seen",
        "face too small or unclear to recognize",
        "face too small or unclear to recognize",
        "face too small or unclear to recognize",
        "no face seen",
    ]
    assert (
        "face seen, face too small or unclear to recognize, or no face seen"
        in world.tools["find_people"].description
    )


def test_one_ray_without_a_heading_leaves_the_others_their_headings(world):
    # A camera that looks straight down: the ray through the image's centre
    # column is vertical and has no yaw, while a ray to either side points
    # sideways.
    world.vision.place_base(world.ctx, camera=CAMERA_DOWN)
    world.vision.show(
        people=[
            person(100, "ana", track_id=1),
            person(320, "sam", track_id=2),
            person(540, "bob", track_id=3),
        ]
    )

    out = world.call("find_people")

    left, centre, right = (
        world.ctx.handles.get(h, kind="person").payload.heading_map for h in out.handles
    )
    assert left == pytest.approx(BASE_YAW + math.pi / 2, abs=1e-4)
    assert centre is None
    assert right == pytest.approx(BASE_YAW - math.pi / 2, abs=1e-4)


def test_without_the_camera_transform_no_box_waits_for_tf_again():
    class NoTransform:
        lookups = 0

        def lookup_transform(self, *args, **kwargs):
            self.lookups += 1
            raise LookupException('"map" passed to lookupTransform does not exist.')

    buffer = NoTransform()
    info = CameraInfo(width=WIDTH, height=HEIGHT)
    info.k = [FX, 0.0, CX, 0.0, FX, HEIGHT / 2, 0.0, 0.0, 1.0]

    result = _perception.headings(
        [(0.0, 0.0, 10.0, 10.0)] * 3, info, Header(frame_id=CAMERA_FRAME), buffer
    )

    assert result == [None, None, None]
    assert buffer.lookups == 1


def test_find_people_keeps_only_those_who_show_the_gesture(world):
    world.vision.show(
        people=[
            person(100, "ana", pose="waving"),
            person(320, "sam"),
            person(530, "carol", pose="sitting", top=140, height=288),
        ]
    )

    waving = world.call("find_people", gesture="waving")
    standing = world.call("find_people", gesture="Standing")
    raising = world.call("find_people", gesture="raising right-arm")

    s = waving.set_id
    assert lines(waving) == [
        f"{s}: 1 of 3 people in view shows waving, left to right.",
        f"{s}.0: left, height 0.83, standing, waving, raising_right_arm, face seen",
    ]
    s = standing.set_id
    assert lines(standing) == [
        f"{s}: 2 of 3 people in view show standing, left to right.",
        f"{s}.0: left, height 0.83, standing, waving, raising_right_arm, face seen",
        f"{s}.1: centre, height 0.83, standing, face seen",
    ]
    assert lines(raising)[0] == (
        f"{raising.set_id}: 1 of 3 people in view shows raising_right_arm, left to right."
    )


def test_find_people_says_what_it_saw_when_nobody_matches(world):
    world.vision.show(
        people=[
            person(100, "ana", pose="waving"),
            person(320, "sam"),
            person(530, "carol", pose="sitting", top=140, height=288),
        ]
    )

    nobody = world.fails("find_people", gesture="lying_down")
    unknown = world.fails("find_people", gesture="dancing")

    assert nobody.code == "NOT_FOUND"
    assert nobody.message == (
        "Nobody in view shows lying_down. In view, left to right: "
        "left, standing, waving, raising_right_arm; centre, standing; right, sitting."
    )
    assert unknown.code == "REFUSED"
    assert unknown.message == (
        f"'dancing' is not a gesture find_people knows. Use one of {', '.join(GESTURES)}."
    )


def test_find_people_is_not_found_when_nobody_is_in_view(world):
    world.vision.show(people=[])

    error = world.fails("find_people")

    assert (error.code, error.message) == ("NOT_FOUND", "Nobody is in view.")


def test_without_a_publisher_the_vision_tools_are_unavailable(connector, monkeypatch):
    monkeypatch.setattr(
        _perception, "CONFIG", dataclasses.replace(_perception.CONFIG, wait_s=0.2)
    )
    toolbox = Toolbox(connector, {"vision": VisionFeed(connector)})
    tools = {tool.name: tool for tool in toolbox.tools()}

    start = time.monotonic()
    with pytest.raises(StepError) as people:
        tools["find_people"].call()
    with pytest.raises(StepError) as seats:
        tools["find_seats"].call()
    waited = time.monotonic() - start

    assert people.value.code == "UNAVAILABLE"
    assert people.value.message == (
        f"No message on {PEOPLE_TOPIC} within 0.2 s. "
        "Is walkie-vision or oracle_perception running?"
    )
    assert seats.value.code == "UNAVAILABLE"
    assert OBJECTS_TOPIC in seats.value.message
    assert waited < 1.5


def test_a_stop_cancels_find_people_while_it_waits(world):
    world.ctx.cancel.set()

    error = world.fails("find_people")

    assert error.code == "CANCELLED"


def show_a_seat_and_a_person(world) -> None:
    world.vision.show(
        objects=[thing("chair", 80, 200, 220, 400)],
        people=[person(320, "sam", track_id=1)],
    )


def test_a_stop_before_the_read_stores_no_handle_with_fresh_arrays_at_hand(world, monkeypatch):
    show_a_seat_and_a_person(world)
    stored = spy_on_put_set(world, monkeypatch)
    world.ctx.cancel.set()

    people = world.fails("find_people")
    seats = world.fails("find_seats")

    assert (people.code, seats.code) == ("CANCELLED", "CANCELLED")
    assert stored == []


def test_a_stop_after_the_read_stores_no_handle(world, monkeypatch):
    show_a_seat_and_a_person(world)
    stored = spy_on_put_set(world, monkeypatch)
    headings = _perception.headings

    # The stop comes while the atomic looks up the headings, after the
    # arrays were read.
    def stop_meanwhile(*args):
        result = headings(*args)
        world.ctx.cancel.set()
        return result

    monkeypatch.setattr(_perception, "headings", stop_meanwhile)

    people = world.fails("find_people")
    # As the next task would, so that find_seats reads its arrays.
    world.ctx.cancel.clear()
    seats = world.fails("find_seats")

    assert (people.code, seats.code) == ("CANCELLED", "CANCELLED")
    assert stored == []


# -- remember_person and recognize_person ----------------------------------------------------


def meet(world, guests: dict[str, dict], shots: int = 3, facts=None) -> dict[str, str]:
    """Show the guests for `shots` images, then remember each by name.

    `guests` maps a name to its person() arguments. Returns each guest's
    remember_person summary by name.
    """
    for _ in range(shots):
        world.vision.show(people=[person(**args) for args in guests.values()])
    out = world.call("find_people")
    summaries = {}
    for name, handle in zip(guests, out.handles, strict=True):
        args = {"person": handle, "name": name}
        if facts and name in facts:
            args["facts"] = facts[name]
        summaries[name] = world.call("remember_person", **args).summary
    return summaries


def test_remember_then_recognize_the_same_person(world):
    summaries = meet(
        world,
        {"Sam": {"cx": 320, "key": "sam", "track_id": 1}},
        facts={"Sam": {"drink": "milk"}},
    )
    # Later, on another track and in another image.
    world.vision.show(people=[person(500, "sam", track_id=9, shot=1)])
    found = world.call("find_people")

    out = world.call("recognize_person", person=found.handles[0])

    assert summaries["Sam"] == "Remembered Sam (drink: milk) from 3 face and 3 appearance vectors."
    assert lines(out) == [f"{found.handles[0]} (right): Sam, drink: milk"]
    [sam] = out.people
    assert (sam.name, sam.facts, sam.matched_by) == ("Sam", {"drink": "milk"}, "face")
    assert sam.person_id is not None and sam.distance < 0.2


def test_two_remembered_guests_are_named_after_they_switch_places(world):
    meet(
        world,
        {
            "Sam": {"cx": 100, "key": "sam", "track_id": 1},
            "Ana": {"cx": 530, "key": "ana", "track_id": 2},
        },
        facts={"Sam": {"drink": "milk"}},
    )
    world.vision.show(
        people=[
            person(100, "ana", track_id=3, shot=1),
            person(530, "sam", track_id=4, shot=1),
        ]
    )
    found = world.call("find_people")

    out = world.call("recognize_person", person=found.set_id)

    s = found.set_id
    assert lines(out) == [f"{s}.0 (left): Ana", f"{s}.1 (right): Sam, drink: milk"]
    assert [p.name for p in out.people] == ["Ana", "Sam"]


def test_a_third_person_never_met_is_unknown(world):
    meet(
        world,
        {
            "Sam": {"cx": 100, "key": "sam", "track_id": 1},
            "Ana": {"cx": 530, "key": "ana", "track_id": 2},
        },
    )
    world.vision.show(
        people=[
            person(100, "sam", track_id=3, shot=1),
            person(320, "dave", track_id=4, shot=1),
            person(530, "ana", track_id=5, shot=1),
        ]
    )
    found = world.call("find_people")

    out = world.call("recognize_person", person=found.set_id)

    s = found.set_id
    assert lines(out) == [f"{s}.0 (left): Sam", f"{s}.1 (centre): UNKNOWN", f"{s}.2 (right): Ana"]
    assert out.people[1].person_id is None and out.people[1].matched_by is None


def test_appearance_names_a_person_whose_face_is_not_seen(world):
    meet(world, {"Sam": {"cx": 320, "key": "sam", "track_id": 1}})
    # Turned away: no face in the box.
    world.vision.show(people=[person(320, "sam", track_id=2, shot=1, face_side=0)])
    turned = world.call("find_people")
    by_appearance = world.call("recognize_person", person=turned.handles[0])
    # Sam's face on the left, and someone in the same clothes, turned away,
    # on the right: the name goes to the face, and only once.
    world.vision.show(
        people=[
            person(100, "sam", track_id=3, shot=2),
            person(530, "sam", track_id=4, shot=3, face_side=0),
        ]
    )
    pair = world.call("find_people")
    both = world.call("recognize_person", person=pair.set_id)

    assert lines(turned)[1].endswith("no face seen")
    assert lines(by_appearance) == [f"{turned.handles[0]} (centre): Sam"]
    assert by_appearance.people[0].matched_by == "appearance"
    assert lines(both) == [f"{pair.set_id}.0 (left): Sam", f"{pair.set_id}.1 (right): UNKNOWN"]
    assert both.people[0].matched_by == "face"


def test_a_face_model_without_a_limit_is_never_matched(world):
    world.vision.show(people=[person(320, "sam", track_id=1, face_model="mystery_face")])
    found = world.call("find_people")
    world.call("remember_person", person=found.handles[0], name="Sam")

    out = world.call("recognize_person", person=found.handles[0])

    # The appearance still names Sam; the line of an unnamed person would
    # say which model had no limit.
    assert lines(out) == [f"{found.handles[0]} (centre): Sam"]
    assert out.people[0].matched_by == "appearance"
    world.vision.show(people=[person(320, "dave", track_id=2, face_model="mystery_face")])
    stranger = world.call("find_people")
    unknown = world.call("recognize_person", person=stranger.handles[0])
    # Dave's appearance was matched and missed, and the line says so: the
    # missing face limit is not the only reason he is UNKNOWN.
    assert lines(unknown) == [
        f"{stranger.handles[0]} (centre): UNKNOWN, appearance not matched, no distance "
        "limit for the face model mystery_face"
    ]


def test_a_person_without_vectors_cannot_be_recognized_or_remembered(world):
    world.vision.show(people=[person(100, "ana", track_id=1), person(530)])
    found = world.call("find_people")

    alone = world.fails("recognize_person", person=found.handles[1])
    out = world.call("recognize_person", person=found.set_id)
    remember = world.fails("remember_person", person=found.handles[1], name="Bob")

    assert alone.code == "NOT_FOUND"
    assert alone.message == (
        f"No face or appearance seen for {found.handles[1]}. "
        "Look at them and call find_people again."
    )
    assert lines(out) == [
        f"{found.set_id}.0 (left): UNKNOWN",
        f"{found.set_id}.1 (right): UNKNOWN, no face or appearance seen",
    ]
    assert remember.code == "NOT_FOUND"
    assert len(world.memory) == 0


def test_a_set_of_one_without_vectors_answers_like_a_larger_set(world):
    world.vision.show(people=[person(320)])
    found = world.call("find_people")

    out = world.call("recognize_person", person=found.set_id)

    # Only a handle asked for alone fails with NOT_FOUND; a set id gets its
    # lines, whatever the number of people in it.
    assert lines(out) == [f"{found.set_id}.0 (centre): UNKNOWN, no face or appearance seen"]
    assert out.people[0].person_id is None


def test_remember_person_drops_a_far_vector(world):
    # Five images of Sam's face, then one where a bystander's face was
    # matched to Sam's box: the bystander's face is an outlier.
    for shot in range(5):
        world.vision.show(people=[person(320, "sam", track_id=1)])
    world.vision.show(people=[person(320, "sam", track_id=1, face_key="bob")])
    found = world.call("find_people")

    out = world.call("remember_person", person=found.handles[0], name="Sam")

    assert out.summary == "Remembered Sam from 5 face and 6 appearance vectors."
    assert (out.faces, out.appearances) == (5, 6)
    assert (out.dropped_faces, out.dropped_appearances) == (1, 0)
    # Bob's face was not stored as Sam's.
    [[nearest]] = world.memory.match_faces([ModelVector(FAKE_FACE_MODEL, vector("face", "bob"))])
    assert nearest.distance > 0.5


def test_remember_person_stores_no_face_too_small_to_trust(world):
    for _ in range(3):
        world.vision.show(people=[person(320, "sam", track_id=1, face_side=SMALL_FACE)])
    found = world.call("find_people")

    out = world.call("remember_person", person=found.handles[0], name="Sam")

    assert out.summary == "Remembered Sam from 0 face and 3 appearance vectors."


def test_remember_person_checks_its_arguments(world):
    world.vision.show(people=[person(320, "sam", track_id=1)])
    found = world.call("find_people")

    one_name_for_several = world.fails("remember_person", person=found.set_id, name="Sam")

    assert one_name_for_several.code == "REFUSED"
    assert one_name_for_several.message == (
        f"{found.set_id} is not one person's handle. Give one handle, such as per_51c0.0."
    )
    for bad in (
        {"name": "   "},
        {"name": "S" * 41},
        {"name": "Sam", "facts": {f"fact{i}": "yes" for i in range(9)}},
        {"name": "Sam", "facts": {"age": 30}},
        {"name": "Sam", "facts": {"drink": "m" * 81}},
    ):
        with pytest.raises(ValidationError):
            world.call("remember_person", person=found.handles[0], **bad)
    out = world.call(
        "remember_person", person=found.handles[0], name="  Sam ", facts={" drink ": "milk"}
    )
    assert out.summary == "Remembered Sam (drink: milk) from 1 face and 1 appearance vector."
    assert len(world.memory) == 1


def test_a_stop_before_remember_person_stores_nobody(world):
    world.vision.show(people=[person(320, "sam", track_id=1)])
    found = world.call("find_people")
    world.ctx.cancel.set()

    error = world.fails("remember_person", person=found.handles[0], name="Sam")

    assert error.code == "CANCELLED"
    assert len(world.memory) == 0


def test_a_handle_whose_track_id_went_to_someone_else_keeps_only_its_own_vectors(make_world):
    world = make_world(track_ttl_s=0.5)
    world.vision.show(people=[person(320, "sam", track_id=7)])
    sam = world.call("find_people").handles[0]
    # Sam leaves for longer than the feed keeps a track unseen, and Bob
    # comes in on the same track id.
    time.sleep(0.6)
    for _ in range(3):
        world.vision.show(people=[person(320, "bob", track_id=7)])
    bob = world.call("find_people").handles[0]
    world.call("remember_person", person=bob, name="Bob")

    recognized = world.call("recognize_person", person=sam)
    remembered = world.call("remember_person", person=sam, name="Sam")

    # Sam's handle is still valid: its image showed Sam, whose own vectors
    # it holds. It reads nothing of Bob's track, so it is not Bob, and Sam
    # is stored from the image of the handle alone.
    assert lines(recognized) == [f"{sam} (centre): UNKNOWN"]
    assert remembered.summary == "Remembered Sam from 1 face and 1 appearance vector."
    [nearest, other] = world.memory.match_faces(
        [ModelVector(FAKE_FACE_MODEL, vector("face", "bob"))]
    )[0]
    assert (nearest.name, other.name) == ("Bob", "Sam")
    assert other.distance > 0.5


def test_remember_person_keeps_the_handles_own_vectors_once(make_world):
    world = make_world(vectors_per_track=3)
    # Each image is another shot, so a stored vector shows which image it came from.
    world.vision.show(people=[person(320, "sam", track_id=1, shot=1)])
    sam = world.call("find_people").handles[0]
    for shot in (2, 3):
        world.vision.show(people=[person(320, "sam", track_id=1, shot=shot)])

    # The track's collection still holds the handle's own image.
    while_collected = world.call("remember_person", person=sam, name="Sam")
    for shot in (4, 5, 6):
        world.vision.show(people=[person(320, "sam", track_id=1, shot=shot)])
    # Three newer images have pushed the handle's own out of the collection.
    world.memory.clear()
    after_rollover = world.call("remember_person", person=sam, name="Sam")

    assert while_collected.summary == "Remembered Sam from 3 face and 3 appearance vectors."
    assert after_rollover.summary == "Remembered Sam from 4 face and 4 appearance vectors."
    assert (after_rollover.dropped_faces, after_rollover.dropped_appearances) == (0, 0)

    def stored(shot: int) -> bool:
        [[match]] = world.memory.match_faces(
            [ModelVector(FAKE_FACE_MODEL, vector("face", "sam", shot))]
        )
        return match.distance < 1e-6

    # The handle's own image and the three newest, not the two pushed out.
    assert [stored(shot) for shot in range(1, 7)] == [True, False, False, True, True, True]


def unit(dim: int, *parts: tuple[int, float]) -> np.ndarray:
    """A unit vector of `dim` with the given (index, value) parts, the rest 0."""
    v = np.zeros(dim, dtype=np.float32)
    for index, value in parts:
        v[index] = value
    return v / np.linalg.norm(v)


@pytest.mark.parametrize(
    "models", [(FAKE_FACE_MODEL, ARCFACE_MODEL), (ARCFACE_MODEL, FAKE_FACE_MODEL)]
)
def test_recognize_person_scores_each_model_by_its_own_limit(world, models):
    # Ana is stored under ArcFace (limit 0.4), Bea under the fake model
    # (limit 0.2). The person matches Ana at a distance of 0.20 and Bea at
    # 0.15: Bea is nearer in raw distance, but Ana is at half her model's
    # limit and Bea at three quarters of hers, so Ana is the better match.
    world.memory.remember("Ana", {}, [ModelVector(ARCFACE_MODEL, unit(DIM, (0, 1.0)))], [])
    world.memory.remember("Bea", {}, [ModelVector(FAKE_FACE_MODEL, unit(DIM, (2, 1.0)))], [])
    seen = {
        ARCFACE_MODEL: unit(DIM, (0, 0.8), (1, 0.6)),
        FAKE_FACE_MODEL: unit(DIM, (2, 0.85), (3, math.sqrt(1 - 0.85**2))),
    }
    # The same track under each model in turn, oldest first, so a match on
    # the newest or the oldest model alone fails one of the two orders.
    for model in models:
        detection = person(320, "x", track_id=7, face_model=model, appearance=False)
        detection.face_embedding = seen[model].tolist()
        world.vision.show(people=[detection])
    found = world.call("find_people")

    out = world.call("recognize_person", person=found.handles[0])

    assert lines(out) == [f"{found.handles[0]} (centre): Ana"]
    assert out.people[0].matched_by == "face"
    # The distance reported is the raw cosine distance of the naming model.
    assert out.people[0].distance == pytest.approx(0.2, abs=1e-5)


def test_recognize_person_matches_every_face_model_with_a_limit(world):
    assert ARCFACE_MODEL in _perception.CONFIG.identity.face_distance_limits
    meet(world, {"Sam": {"cx": 320, "key": "sam", "track_id": 1}})
    # Later, on a new track, Sam is seen under the fake face model, then
    # under another, as when walkie-vision switches models. No appearance
    # vector, so only the face can name him.
    for model in (FAKE_FACE_MODEL, FAKE_FACE_MODEL, ARCFACE_MODEL):
        world.vision.show(
            people=[person(320, "sam", track_id=2, shot=1, face_model=model, appearance=False)]
        )
    found = world.call("find_people")

    out = world.call("recognize_person", person=found.handles[0])

    # The newest face is under a model Sam was never stored with, and the
    # fake_face vectors of the track still name him.
    assert world.ctx.handles.get(found.handles[0], kind="person").payload.face.model == (
        ARCFACE_MODEL
    )
    assert lines(out) == [f"{found.handles[0]} (centre): Sam"]
    assert out.people[0].matched_by == "face"
    assert out.people[0].distance < 0.2


# -- find_seats -------------------------------------------------------------------------------


def test_find_seats_marks_a_chair_taken(world):
    world.vision.show(
        objects=[
            thing("chair", 420, 200, 560, 400),
            thing("table", 250, 250, 400, 400),
            thing("chair", 80, 200, 220, 400),
        ],
        # Seated on the left chair: the hips are inside its box.
        people=[person(150, "sam", track_id=1, pose="sitting", top=60, height=320)],
    )

    out = world.call("find_seats")

    s = out.set_id
    assert re.fullmatch(r"seat_[0-9a-f]{4}", s)
    assert lines(out) == [
        f"{s}.0: chair, taken, left",
        f"{s}.1: chair, free, right",
        "Free places: 1 of 2.",
    ]
    assert out.free == [f"{s}.1"]
    taken = world.ctx.handles.get(f"{s}.0", kind="seat").payload
    assert (taken.label, taken.part, taken.occupied) == ("chair", None, True)
    assert taken.box == pytest.approx((80.0, 200.0, 220.0, 400.0))
    assert taken.heading_map == pytest.approx(heading_to(150), abs=1e-4)
    assert (taken.header.frame_id, taken.image_width, taken.image_height) == (
        CAMERA_FRAME,
        640,
        480,
    )


def test_find_seats_gives_each_sofa_place_a_handle(world):
    world.vision.show(
        objects=[thing("sofa", 140, 240, 500, 400)],
        people=[person(200, "sam", track_id=1, pose="sitting", top=100, height=320)],
    )

    out = world.call("find_seats")

    s = out.set_id
    assert lines(out) == [
        f"{s}.0: sofa, left place, taken, left",
        f"{s}.1: sofa, middle place, free, centre",
        f"{s}.2: sofa, right place, free, right",
        "Free places: 2 of 3.",
    ]
    assert out.free == [f"{s}.1", f"{s}.2"]
    middle = world.ctx.handles.get(f"{s}.1", kind="seat").payload
    assert (middle.part, middle.box) == ("middle", pytest.approx((260.0, 240.0, 380.0, 400.0)))
    assert middle.heading_map == pytest.approx(heading_to(320), abs=1e-4)


def test_find_seats_pairs_the_objects_with_the_people_of_the_same_image(world, monkeypatch):
    chair = thing("chair", 80, 200, 220, 400)
    seated = person(150, "sam", track_id=1, pose="sitting", top=60, height=320)
    world.vision.show(objects=[chair], people=[seated])
    # A newer image in which nobody sits: its people array must not be used
    # with the older objects array.
    world.vision.show(people=[])

    paired = world.call("find_seats")

    assert lines(paired)[0].endswith(": chair, taken, left")
    # An objects array whose image has no people array is unavailable, not
    # judged against another image's people.
    monkeypatch.setattr(
        _perception, "CONFIG", dataclasses.replace(_perception.CONFIG, pair_wait_s=0.2)
    )
    world.vision.show(objects=[chair])
    unpaired = world.fails("find_seats")
    assert unpaired.code == "UNAVAILABLE"
    assert "has the stamp" in unpaired.message


def test_find_seats_is_not_found_without_a_seat_and_lists_what_it_saw(world):
    world.vision.show(
        objects=[thing("table", 250, 250, 400, 400), thing("cup", 300, 220, 330, 250)],
        classes=["chair", "table", "cup"],
        people=[],
    )
    no_seat = world.fails("find_seats")
    world.vision.show(
        objects=[thing("table", 250, 250, 400, 400)], classes=["table", "cup"], people=[]
    )
    not_looking = world.fails("find_seats")

    assert no_seat.code == "NOT_FOUND"
    assert no_seat.message == (
        "No seat in view. Seats are chair, sofa, armchair, stool; in view: table, cup."
    )
    assert not_looking.message == (
        "No seat in view. Seats are chair, sofa, armchair, stool; in view: table. "
        "The detector is not looking for any seat class."
    )


def test_find_seats_names_the_seat_classes_of_the_config_it_ran_with(world, monkeypatch):
    benches = SeatingConfig(seat_classes=("bench",))
    monkeypatch.setattr(
        _perception, "CONFIG", dataclasses.replace(_perception.CONFIG, seating=benches)
    )
    world.vision.show(objects=[thing("chair", 250, 250, 400, 400)], classes=["chair"], people=[])

    error = world.fails("find_seats")

    # The chair is not a seat under this config, and the message says what
    # the call looked for rather than the classes of the default config.
    assert error.message == (
        "No seat in view. Seats are bench; in view: chair. "
        "The detector is not looking for any seat class."
    )


# -- Staleness ---------------------------------------------------------------------------------


def test_handles_go_stale_when_the_base_drives_but_not_when_it_turns(world):
    world.vision.show(
        objects=[thing("chair", 80, 200, 220, 400)],
        people=[person(320, "sam", track_id=1)],
    )
    found = world.call("find_people")
    seats = world.call("find_seats")

    # A turn on the spot: the gaze loop does this while it follows a person.
    world.vision.place_base(world.ctx, yaw=BASE_YAW + 1.0)
    turned = world.call("recognize_person", person=found.handles[0])
    # 10 cm forward, past the 5 cm limit.
    world.vision.place_base(world.ctx, x=0.1, yaw=BASE_YAW + 1.0)
    person_error = world.fails("recognize_person", person=found.handles[0])
    with pytest.raises(StepError) as seat_error:
        world.ctx.handles.get(seats.handles[0], kind="seat")

    assert lines(turned) == [f"{found.handles[0]} (centre): UNKNOWN"]
    assert person_error.code == "STALE"
    assert found.handles[0] in person_error.message
    assert "find_people" in person_error.message
    assert seat_error.value.code == "STALE"
    assert "find_seats" in seat_error.value.message


def test_without_the_base_pose_at_the_stamp_nothing_is_stored(world, monkeypatch):
    show_a_seat_and_a_person(world)
    # The base's first pose on /tf comes after the image: TF has a pose now,
    # but none at the image's stamp, and the pose now could hide a drive.
    world.vision.drive_base(world.ctx, [0.01])
    stored = spy_on_put_set(world, monkeypatch)

    for tool in ("find_people", "find_seats"):
        error = world.fails(tool)
        assert error.code == "UNAVAILABLE"
        assert "when the camera saw this is unknown, so nothing was stored" in error.message
    assert stored == []


def test_a_stop_while_the_base_pose_lookup_fails_is_a_cancel(world, monkeypatch):
    show_a_seat_and_a_person(world)
    world.vision.drive_base(world.ctx, [0.01])
    stored = spy_on_put_set(world, monkeypatch)
    lookup = world.ctx.tf_buffer.lookup_transform

    # The stop comes while capture_pose waits for the pose at the stamp,
    # which then times out.
    def stop_meanwhile(target, source, *args, **kwargs):
        if (target, source) == (MAP_FRAME, ROBOT_FRAME):
            world.ctx.cancel.set()
        return lookup(target, source, *args, **kwargs)

    monkeypatch.setattr(world.ctx.tf_buffer, "lookup_transform", stop_meanwhile)

    people = world.fails("find_people")
    world.ctx.cancel.clear()
    seats = world.fails("find_seats")

    assert (people.code, seats.code) == ("CANCELLED", "CANCELLED")
    assert stored == []


def test_an_array_without_a_stamp_has_no_capture_pose(world):
    header = Header(frame_id=CAMERA_FRAME)

    with pytest.raises(StepError) as error:
        _perception.capture_pose(world.ctx, header)

    assert error.value.code == "UNAVAILABLE"
    assert "has no stamp" in error.value.message


def test_a_handle_from_an_image_seen_before_a_drive_is_stale(world):
    world.vision.drive_base(world.ctx, [0.01])
    show_a_seat_and_a_person(world)
    # Still there just after the image, so TF has the pose at its stamp,
    # then 20 cm forward before the atomics run.
    world.vision.drive_base(world.ctx, [0.01, 0.21])

    found = world.call("find_people")
    seats = world.call("find_seats")

    for handle_id, kind in ((found.handles[0], "person"), (seats.handles[0], "seat")):
        with pytest.raises(StepError) as error:
            world.ctx.handles.get(handle_id, kind=kind)
        assert error.value.code == "STALE"
        assert "moved 0.20 m" in error.value.message
