"""OraclePerception: the arrays it draws from a scene, and the oracle over ROS.

The drawing runs without ROS through draw_detections, with test_projection's
simulated head camera on a base at the origin. The ROS tests run the oracle
on a node of its own, spun by a MultiThreadedExecutor as it needs, next to a
node that plays walkie-ros's simulation: it publishes the head camera's
camera_info and the robot's transforms at chula_pat14_6's door pose. There,
VisionFeed and find_people read the oracle's arrays on the `connector`
fixture's node. All of it runs on Fast DDS and a random domain (ADR-0019).
"""

import dataclasses
import itertools
import logging
import math
import random
import threading
from collections.abc import Callable

import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
from tf_transformations import euler_matrix, quaternion_from_matrix
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetectionArray

from test_people_atomics import NO_CANCEL, wait_until
from test_projection import CX, FX, HEIGHT, K, WIDTH, base_from_camera, facing_person, sim_camera
from walkie_agent.perception.geometry import MAP_FRAME
from walkie_agent.perception.identity import (
    DEFAULT_IDENTITY_CONFIG,
    cosine_distance,
    reject_outliers,
)
from walkie_agent.perception.people_memory import InMemoryPeopleMemory
from walkie_agent.perception.seating import DEFAULT_SEATING_CONFIG, bbox_xyxy
from walkie_agent.perception.vision_feed import (
    OBJECTS_TOPIC,
    PEOPLE_TOPIC,
    VisionFeed,
    face_trusted,
)
from walkie_agent.sim import oracle_perception
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, FAKE_REID_MODEL, hash_vector
from walkie_agent.sim.oracle_perception import (
    DEFAULT_ORACLE_CONFIG,
    DETECTION_SCORE,
    HEAD_CAMERA_INFO_TOPIC,
    KEYPOINT_SCORE,
    TURNED_FACE_KEYPOINT_SCORE,
    VECTOR_DIM,
    OracleConfig,
    OraclePerception,
    draw_detections,
    transform_matrix,
)
from walkie_agent.sim.projection import (
    FACE_KEYPOINTS,
    camera_from_map_at,
    camera_position,
    clip_box,
    in_image,
    project_box,
    project_person,
)
from walkie_agent.sim.scene import MapBox, Scene, SceneObject, ScenePerson
from walkie_agent.toolbox import StepError, Toolbox
from walkie_agent.tools.atomic.find_people import FACE_SEEN
from walkie_agent.tools.navigation import ROBOT_FRAME

# The optical frame of the simulation's head camera, which its camera_info's
# header names.
CAMERA_FRAME = "zed_head_left_camera_frame_optical"
STAMP = TimeMsg(sec=5, nanosec=7)
# No noise and no outliers, so the arrays are the projection itself.
QUIET = OracleConfig(pixel_noise_px=0.0, vector_noise=0.0, outlier_rate=0.0)
# The camera's position in the map with the base at the origin, facing +x.
CAMERA_X, CAMERA_Y = (float(v) for v in camera_position(sim_camera())[:2])
# A sofa 2.5 m ahead of the camera, wholly in the image.
SOFA = SceneObject(
    track_id=7, label="sofa", box=MapBox((CAMERA_X + 2.5, CAMERA_Y, 0.42), (0.6, 1.8, 0.84))
)

# chula_pat14_6's door pose, x, y and yaw, the base facing the entrance; and
# the guest's spot in the hallway, 1.71 m from it.
DOOR = (2.35, -0.17, 3.01)
GUEST_XY = (0.65, 0.03)
# As tf2_ros's StaticTransformBroadcaster publishes, so a listener that
# starts later still receives the transforms.
STATIC_TF_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def camera_info(stamp: TimeMsg = STAMP, frame: str = CAMERA_FRAME) -> CameraInfo:
    """The simulated head camera's camera_info for one image."""
    return CameraInfo(
        header=Header(stamp=stamp, frame_id=frame), width=WIDTH, height=HEIGHT, k=list(K)
    )


def draw(
    scene: Scene, config: OracleConfig = QUIET, seed: int = 0
) -> tuple[PersonDetectionArray, ObjectDetectionArray]:
    """The arrays of one image, from the head camera on a base at the origin."""
    return draw_detections(scene, camera_info(), sim_camera(), config, np.random.default_rng(seed))


def draw_images(
    scene: Scene, config: OracleConfig, count: int
) -> list[tuple[PersonDetectionArray, ObjectDetectionArray]]:
    """The arrays of `count` images in a row, all their noise from one generator."""
    rng = np.random.default_rng(0)
    return [draw_detections(scene, camera_info(), sim_camera(), config, rng) for _ in range(count)]


def tf_msg(parent: str, child: str, matrix: np.ndarray) -> TransformStamped:
    """`matrix`, the child frame's pose in the parent's, as a static TF transform."""
    msg = TransformStamped()
    msg.header.frame_id, msg.child_frame_id = parent, child
    t, q = msg.transform.translation, msg.transform.rotation
    t.x, t.y, t.z = (float(v) for v in matrix[:3, 3])
    q.x, q.y, q.z, q.w = (float(v) for v in quaternion_from_matrix(matrix))
    return msg


def door_guest() -> ScenePerson:
    """A 1.70 m guest on the guest spot, facing the base at the door pose."""
    facing = math.atan2(DOOR[1] - GUEST_XY[1], DOOR[0] - GUEST_XY[0])
    return ScenePerson(
        track_id=5, x=GUEST_XY[0], y=GUEST_XY[1], facing=facing, face="guest", appearance="guest"
    )


class SceneAt:
    """The oracle's scene callable: one scene, and a record of the stamps it was asked at."""

    def __init__(self, scene: Scene | None = None) -> None:
        self.scene = Scene() if scene is None else scene
        self.asked: list[Time] = []

    def __call__(self, stamp: Time) -> Scene:
        self.asked.append(stamp)
        return self.scene


def oracle_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == oracle_perception.__name__]


# -- Drawing ------------------------------------------------------------------------------


def test_without_noise_the_arrays_are_the_projection():
    guest = facing_person(2.0, track_id=3, face="ann", appearance="ann")
    camera = sim_camera()
    view = project_person(guest, camera, K)

    people, objects = draw(Scene(people=(guest,), objects=(SOFA,)))

    [seen] = people.people
    assert seen.track_id == 3
    assert seen.score == pytest.approx(DETECTION_SCORE)
    assert bbox_xyxy(seen.bbox) == pytest.approx(clip_box(view.box, WIDTH, HEIGHT))
    assert np.reshape(seen.keypoints, (-1, 2)) == pytest.approx(view.keypoints, abs=1e-3)
    assert np.asarray(seen.keypoint_scores) == pytest.approx(KEYPOINT_SCORE)
    assert bbox_xyxy(seen.face_bbox) == pytest.approx(view.face_box)
    assert seen.face_det_score == pytest.approx(DETECTION_SCORE)
    assert seen.face_model == FAKE_FACE_MODEL
    assert np.asarray(seen.face_embedding) == pytest.approx(
        hash_vector("face:ann", VECTOR_DIM), abs=1e-6
    )
    assert seen.appearance_model == FAKE_REID_MODEL
    assert np.asarray(seen.appearance_embedding) == pytest.approx(
        hash_vector("reid:ann", VECTOR_DIM), abs=1e-6
    )

    assert list(objects.classes) == list(DEFAULT_SEATING_CONFIG.seat_classes)
    [sofa] = objects.detections
    assert (sofa.label, sofa.track_id) == ("sofa", 7)
    assert sofa.score == pytest.approx(DETECTION_SCORE)
    assert bbox_xyxy(sofa.bbox) == pytest.approx(
        clip_box(project_box(SOFA.box, camera, K), WIDTH, HEIGHT)
    )
    # The oracle has no image to cut a crop from and no depth for points.
    assert (sofa.embedding_model, len(sofa.embedding)) == ("", 0)
    assert len(sofa.points.data) == 0 and len(sofa.crop.data) == 0


def test_one_seed_repeats_the_noise_and_another_seed_changes_it():
    scene = Scene(people=(facing_person(2.0, face="ann", appearance="ann"),), objects=(SOFA,))

    first = draw(scene, DEFAULT_ORACLE_CONFIG, seed=3)

    assert draw(scene, DEFAULT_ORACLE_CONFIG, seed=3) == first
    people, objects = draw(scene, DEFAULT_ORACLE_CONFIG, seed=4)
    assert people != first[0] and objects != first[1]


def test_vector_noise_keeps_a_persons_vectors_far_inside_the_fake_models_limit():
    scene = Scene(people=(facing_person(2.0, face="ann", appearance="ann"),))
    config = OracleConfig(outlier_rate=0.0)
    seen = [arrays[0].people[0] for arrays in draw_images(scene, config, 30)]
    limits = DEFAULT_IDENTITY_CONFIG
    kinds = (
        ("face_embedding", "face:ann", limits.face_distance_limits[FAKE_FACE_MODEL]),
        ("appearance_embedding", "reid:ann", limits.appearance_distance_limits[FAKE_REID_MODEL]),
    )
    for field, key, limit in kinds:
        clean = hash_vector(key, VECTOR_DIM)
        vectors = [np.asarray(getattr(person, field)) for person in seen]
        to_clean = [cosine_distance(v, clean) for v in vectors]
        between = [cosine_distance(a, b) for a, b in itertools.combinations(vectors, 2)]
        # The arithmetic of OracleConfig.vector_noise: about 0.006 from the
        # clean vector and 0.013 between two noisy ones.
        assert np.mean(to_clean) == pytest.approx(0.0063, rel=0.15)
        assert np.mean(between) == pytest.approx(0.0126, rel=0.15)
        assert max(between) < limit / 5


def test_outliers_are_random_vectors_that_reject_outliers_drops():
    config = OracleConfig(outlier_rate=0.25)
    scene = Scene(people=(facing_person(2.0, face="ann"),))
    vectors = [
        list(arrays[0].people[0].face_embedding) for arrays in draw_images(scene, config, 40)
    ]
    clean = hash_vector("face:ann", VECTOR_DIM)

    outliers = [v for v in vectors if cosine_distance(v, clean) > 0.5]
    kept = {tuple(v) for v in reject_outliers(vectors)}

    assert 3 <= len(outliers) <= 20
    # A random unit vector is nearly orthogonal to the person's: the cosine
    # of two random directions in 512 dimensions is about 1 / sqrt(512).
    assert all(abs(1.0 - cosine_distance(v, clean)) < 0.25 for v in outliers)
    assert not kept & {tuple(v) for v in outliers}
    assert len(kept) >= 0.8 * (len(vectors) - len(outliers))


@pytest.mark.parametrize(
    ("turned_deg", "limit_deg", "face_shown"),
    [(50.0, 60.0, True), (70.0, 60.0, False), (70.0, 80.0, True), (180.0, 60.0, False)],
)
def test_a_face_is_drawn_only_for_a_person_facing_the_camera(turned_deg, limit_deg, face_shown):
    person = facing_person(
        2.0, facing=math.pi + math.radians(turned_deg), face="ann", appearance="ann"
    )
    config = dataclasses.replace(QUIET, face_limit_deg=limit_deg)

    people, _ = draw(Scene(people=(person,)), config)

    [seen] = people.people
    scores = np.asarray(seen.keypoint_scores)
    face_points = list(FACE_KEYPOINTS)
    assert np.delete(scores, face_points) == pytest.approx(KEYPOINT_SCORE)
    if face_shown:
        assert seen.face_det_score == pytest.approx(DETECTION_SCORE)
        assert seen.face_model == FAKE_FACE_MODEL
        assert len(seen.face_embedding) == VECTOR_DIM
        assert scores[face_points] == pytest.approx(KEYPOINT_SCORE)
    else:
        # As PersonDetection.msg describes a person whose face was not found.
        assert seen.face_det_score == 0.0
        assert (seen.face_model, len(seen.face_embedding)) == ("", 0)
        assert (seen.face_bbox.size_x, seen.face_bbox.size_y) == (0.0, 0.0)
        assert scores[face_points] == pytest.approx(TURNED_FACE_KEYPOINT_SCORE)
    # The appearance does not need the face.
    assert seen.appearance_model == FAKE_REID_MODEL
    assert len(seen.appearance_embedding) == VECTOR_DIM


def test_a_person_without_face_or_appearance_keys_has_no_vectors():
    people, _ = draw(Scene(people=(facing_person(2.0),)))

    [seen] = people.people
    assert seen.face_det_score == 0.0
    assert (seen.face_model, len(seen.face_embedding)) == ("", 0)
    assert (seen.appearance_model, len(seen.appearance_embedding)) == ("", 0)
    # The pose estimator still finds the face's points; only the recognizer has nothing.
    assert np.asarray(seen.keypoint_scores) == pytest.approx(KEYPOINT_SCORE)


def test_dropout_leaves_each_person_out_of_an_image_on_their_own():
    two = Scene(
        people=(
            facing_person(2.0, track_id=1, y=CAMERA_Y + 0.5),
            facing_person(2.0, track_id=2, y=CAMERA_Y - 0.5),
        ),
        objects=(SOFA,),
    )

    people, objects = draw(two, dataclasses.replace(QUIET, dropout_rate=1.0))
    assert len(people.people) == 0 and len(objects.detections) == 1

    config = dataclasses.replace(QUIET, dropout_rate=0.5)
    shown = {
        tuple(person.track_id for person in arrays[0].people)
        for arrays in draw_images(two, config, 40)
    }
    assert shown == {(1, 2), (1,), (2,), ()}


def test_only_objects_of_the_configured_classes_are_detected():
    tv = SceneObject(
        track_id=8, label="tv", box=MapBox((CAMERA_X + 3.0, CAMERA_Y + 1.0, 1.0), (0.1, 1.0, 0.6))
    )
    scene = Scene(objects=(SOFA, tv))

    _, objects = draw(scene)
    assert [d.label for d in objects.detections] == ["sofa"]

    _, objects = draw(scene, dataclasses.replace(QUIET, classes=("tv",)))
    assert [d.label for d in objects.detections] == ["tv"]
    assert list(objects.classes) == ["tv"]


def test_nothing_out_of_view_is_detected_and_noise_never_brings_it_in():
    camera = sim_camera()
    # A 2 cm cube 3 m ahead whose image lies about 5 px past the image's right edge.
    centre = np.linalg.inv(camera) @ [3.0 * (WIDTH + 5 - CX) / FX, 0.0, 3.0, 1.0]
    speck = SceneObject(
        track_id=9, label="chair", box=MapBox(tuple(centre[:3]), (0.02, 0.02, 0.02))
    )
    assert WIDTH < project_box(speck.box, camera, K)[0] < WIDTH + 5
    chair_behind = MapBox((CAMERA_X - 2.0, CAMERA_Y, 0.4), (0.5, 0.5, 0.8))
    behind = Scene(
        people=(ScenePerson(track_id=1, x=CAMERA_X - 2.0, y=CAMERA_Y, facing=0.0),),
        objects=(SceneObject(track_id=9, label="chair", box=chair_behind),),
    )

    people, objects = draw(behind)
    assert len(people.people) == 0 and len(objects.detections) == 0
    # Whether a box is in view is decided before the noise: with 20 px of
    # noise, the cube's box would come into view in about 6 of 10 images.
    noisy = OracleConfig(pixel_noise_px=20.0)
    for _, objects in draw_images(Scene(objects=(speck,)), noisy, 50):
        assert len(objects.detections) == 0


def test_a_person_partly_in_view_gets_a_clipped_box_and_unseen_points_score_zero():
    # Far enough to the right that the image's edge cuts through the body.
    person = facing_person(2.0, y=CAMERA_Y - 2.75)
    view = project_person(person, sim_camera(), K)
    outside = ~in_image(view.keypoints, WIDTH, HEIGHT)
    assert outside.any() and not outside.all()

    people, _ = draw(Scene(people=(person,)))

    [seen] = people.people
    x1, _, x2, _ = bbox_xyxy(seen.bbox)
    assert (x1, x2) == pytest.approx((view.box[0], WIDTH))
    scores = np.asarray(seen.keypoint_scores)
    assert scores[outside] == pytest.approx(0.0)
    assert scores[~outside] == pytest.approx(KEYPOINT_SCORE)
    # walkie-vision's keypoints are always pixels of the image, so the
    # unseen ones are moved onto its edge.
    points = np.reshape(seen.keypoints, (-1, 2))
    assert points[outside, 0] == pytest.approx(WIDTH)
    assert points[~outside] == pytest.approx(view.keypoints[~outside], abs=1e-3)


def test_a_tf_transform_becomes_the_matrix_it_stands_for():
    matrix = sim_camera(1.0, -2.0, 0.7)

    assert transform_matrix(tf_msg(CAMERA_FRAME, MAP_FRAME, matrix)) == pytest.approx(
        matrix, abs=1e-12
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"outlier_rate": 1.5},
        {"dropout_rate": -0.1},
        {"pixel_noise_px": -1.0},
        {"vector_noise": -0.01},
        {"warning_period_s": -1.0},
        {"tf_timeout_s": 0.0},
        {"face_limit_deg": 181.0},
    ],
)
def test_an_oracle_config_out_of_range_is_refused(fields):
    with pytest.raises(ValueError, match=next(iter(fields))):
        OracleConfig(**fields)


# -- Skipped images, with the callback called directly -------------------------------------


@pytest.fixture
def idle_node():
    """A node no executor spins, for calling the oracle's callback directly."""
    node = rclpy.create_node(f"oracle_idle_{random.randint(0, 1_000_000)}")
    yield node
    node.destroy_node()


@pytest.mark.parametrize(("period_s", "warnings"), [(5.0, 1), (0.0, 3)])
def test_an_image_without_tf_is_skipped_with_a_throttled_warning(
    idle_node, caplog, period_s, warnings
):
    scene = SceneAt(Scene(people=(facing_person(2.0),)))
    config = OracleConfig(tf_timeout_s=0.01, warning_period_s=period_s)
    # An empty buffer, which no listener fills.
    oracle = OraclePerception(idle_node, scene, config, tf_buffer=Buffer())

    with caplog.at_level(logging.WARNING, logger=oracle_perception.__name__):
        for _ in range(3):
            oracle._on_camera_info(camera_info())

    assert scene.asked == []
    texts = oracle_warnings(caplog)
    assert len(texts) == warnings
    assert all(f"from {MAP_FRAME} to {CAMERA_FRAME}" in text for text in texts)


def test_an_image_without_intrinsics_is_skipped_with_a_warning(idle_node, caplog):
    scene = SceneAt(Scene(people=(facing_person(2.0),)))
    oracle = OraclePerception(idle_node, scene, tf_buffer=Buffer())
    info = camera_info()
    info.k = [0.0] * 9

    with caplog.at_level(logging.WARNING, logger=oracle_perception.__name__):
        oracle._on_camera_info(info)

    assert scene.asked == []
    [text] = oracle_warnings(caplog)
    assert "no focal length" in text and HEAD_CAMERA_INFO_TOPIC in text


def test_the_camera_pose_is_tfs_at_the_images_stamp_not_the_newest(idle_node, monkeypatch):
    # The camera turns between two images. The second transform is the
    # newest, so a lookup at the latest time would draw the first image from
    # the second pose.
    buffer = Buffer()
    first, second = sim_camera(), sim_camera(yaw=0.5)
    for stamp, camera in ((STAMP, first), (TimeMsg(sec=9), second)):
        msg = tf_msg(MAP_FRAME, CAMERA_FRAME, np.linalg.inv(camera))
        msg.header.stamp = stamp
        buffer.set_transform(msg, "test")
    drawn: list[np.ndarray] = []
    monkeypatch.setattr(
        oracle_perception,
        "draw_detections",
        lambda scene, info, camera_from_map, config, rng: (
            drawn.append(camera_from_map) or (PersonDetectionArray(), ObjectDetectionArray())
        ),
    )
    scene = SceneAt()
    oracle = OraclePerception(idle_node, scene, QUIET, tf_buffer=buffer)

    oracle._on_camera_info(camera_info(STAMP))

    [camera_from_map] = drawn
    assert camera_from_map == pytest.approx(first, abs=1e-9)
    assert scene.asked == [Time.from_msg(STAMP)]


# -- Over ROS ----------------------------------------------------------------------------


class Rig:
    """OraclePerception on a node of its own, and a node that plays the simulation.

    The simulation node publishes the head camera's camera_info and the
    robot's transforms on /tf_static, and keeps every array the oracle
    publishes, by stamp. One MultiThreadedExecutor spins both nodes.
    """

    def __init__(self, scene: Scene, **config) -> None:
        suffix = random.randint(0, 1_000_000)
        # Topics of the test's own, so no other node's messages reach it.
        topics = {
            "camera_info_topic": f"/oracle_test_{suffix}/camera_info",
            "people_topic": f"/oracle_test_{suffix}/people",
            "objects_topic": f"/oracle_test_{suffix}/objects",
        }
        self.config = OracleConfig(**(topics | config))
        self.scene = SceneAt(scene)
        self.people: dict[int, PersonDetectionArray] = {}
        self.objects: dict[int, ObjectDetectionArray] = {}
        self.oracle_node = rclpy.create_node(f"oracle_{suffix}")
        self.sim = rclpy.create_node(f"oracle_sim_{suffix}")
        self.oracle = OraclePerception(self.oracle_node, self.scene, self.config)
        self.info_pub = self.sim.create_publisher(CameraInfo, self.config.camera_info_topic, 10)
        self.tf_pub = self.sim.create_publisher(TFMessage, "/tf_static", STATIC_TF_QOS)
        # RELIABLE, which only a reliable publisher matches, so an array that
        # arrives shows the oracle's QoS too.
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.subscriptions = (
            self.sim.create_subscription(
                PersonDetectionArray, self.config.people_topic, self._on_people, reliable
            ),
            self.sim.create_subscription(
                ObjectDetectionArray, self.config.objects_topic, self._on_objects, reliable
            ),
        )
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.oracle_node)
        self.executor.add_node(self.sim)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self._count = 0

    def wait_for_discovery(self) -> None:
        assert wait_until(
            lambda: self.info_pub.get_subscription_count() > 0
            and all(s.get_publisher_count() > 0 for s in self.subscriptions)
        ), "The oracle's subscription and publishers were not discovered."

    def place_robot(self, frame: str = CAMERA_FRAME) -> None:
        """Publish the base at the door pose and the head camera's mount on it."""
        x, y, yaw = DOOR
        map_from_base = euler_matrix(0.0, 0.0, yaw)
        map_from_base[:3, 3] = (x, y, 0.0)
        self.tf_pub.publish(
            TFMessage(
                transforms=[
                    tf_msg(MAP_FRAME, ROBOT_FRAME, map_from_base),
                    tf_msg(ROBOT_FRAME, frame, base_from_camera()),
                ]
            )
        )

    def show(self, frame: str = CAMERA_FRAME) -> CameraInfo:
        """Publish the camera_info of a new image, with a stamp of its own, and return it."""
        # The nanosecond counter keeps two stamps made in the same tick apart.
        self._count += 1
        now_ns = self.sim.get_clock().now().nanoseconds + self._count
        stamp = TimeMsg(sec=now_ns // 1_000_000_000, nanosec=now_ns % 1_000_000_000)
        info = camera_info(stamp, frame)
        self.info_pub.publish(info)
        return info

    def capture(
        self, until: Callable[[CameraInfo], bool] | None = None, frame: str = CAMERA_FRAME
    ) -> CameraInfo:
        """Show images until `until(image)` holds for one, and return that image.

        By default, until the oracle's arrays of the image arrived. The first
        images can go unanswered: a publisher drops its messages until it has
        discovered the subscription, and the oracle skips an image while its
        TF listener has not received the transforms yet.
        """
        answered = self.has_arrays if until is None else until
        for _ in range(10):
            info = self.show(frame)
            if wait_until(lambda: answered(info), timeout=0.5):
                return info
        raise AssertionError("None of 10 camera_info messages was answered.")

    def has_arrays(self, info: CameraInfo) -> bool:
        key = Time.from_msg(info.header.stamp).nanoseconds
        return key in self.people and key in self.objects

    def arrays(self, info: CameraInfo) -> tuple[PersonDetectionArray, ObjectDetectionArray]:
        key = Time.from_msg(info.header.stamp).nanoseconds
        return self.people[key], self.objects[key]

    def close(self) -> None:
        self.executor.shutdown()
        self.thread.join(timeout=2)
        self.oracle.destroy()
        self.oracle_node.destroy_node()
        self.sim.destroy_node()

    def _on_people(self, msg: PersonDetectionArray) -> None:
        self.people[Time.from_msg(msg.header.stamp).nanoseconds] = msg

    def _on_objects(self, msg: ObjectDetectionArray) -> None:
        self.objects[Time.from_msg(msg.header.stamp).nanoseconds] = msg


@pytest.fixture
def make_rig():
    """A Rig factory: the keyword arguments go to the oracle's OracleConfig."""
    rigs: list[Rig] = []

    def make(scene: Scene | None = None, **config) -> Rig:
        rig = Rig(Scene() if scene is None else scene, **config)
        rigs.append(rig)
        rig.wait_for_discovery()
        return rig

    yield make
    for rig in rigs:
        rig.close()


def in_feed(feed: VisionFeed, info: CameraInfo) -> bool:
    """True once the feed holds the people array of the image `info` describes."""
    try:
        feed.people_at(info.header.stamp, 0.0, NO_CANCEL)
    except StepError:
        return False
    return True


def test_each_camera_info_gets_both_arrays_with_its_header_also_when_empty(make_rig):
    rig = make_rig()
    rig.place_robot()

    info = rig.capture()

    people, objects = rig.arrays(info)
    for array in (people, objects):
        assert array.header == info.header
        assert array.camera_info == info
    assert len(people.people) == 0 and len(objects.detections) == 0
    assert list(objects.classes) == list(rig.config.classes)
    # Asked at the image's stamp, so a person who moves is drawn where they were then.
    assert Time.from_msg(info.header.stamp) in rig.scene.asked


def test_a_guest_the_robot_faces_is_near_the_image_centre_with_the_face_in_view(make_rig):
    guest = door_guest()
    rig = make_rig(Scene(people=(guest,)), pixel_noise_px=0.0)
    rig.place_robot()

    info = rig.capture()

    people, _ = rig.arrays(info)
    assert people.header == info.header
    [seen] = people.people
    # The oracle read the camera's pose from TF. Its box is the projection
    # from the door pose only if it read camera <- map the right way round.
    expected = project_person(guest, camera_from_map_at(*DOOR, base_from_camera()), K)
    box = bbox_xyxy(seen.bbox)
    assert box == pytest.approx(clip_box(expected.box, WIDTH, HEIGHT), abs=1e-6)
    # 1.4 deg to the right of the camera's axis.
    assert abs((box[0] + box[2]) / 2 - CX) < 10.0
    x1, y1, x2, y2 = bbox_xyxy(seen.face_bbox)
    assert 0.0 < x1 < x2 < WIDTH and 0.0 < y1 < y2 < HEIGHT
    assert seen.face_det_score == pytest.approx(DETECTION_SCORE)


def test_vision_feed_and_find_people_see_the_guest_with_a_trusted_face(make_rig, connector):
    feed = VisionFeed(connector)
    toolbox = Toolbox(connector, {"vision": feed, "people_memory": InMemoryPeopleMemory()})
    rig = make_rig(
        Scene(people=(door_guest(),)), people_topic=PEOPLE_TOPIC, objects_topic=OBJECTS_TOPIC
    )
    rig.place_robot()
    # find_people looks up the base and the camera at the image's stamp.
    assert wait_until(
        lambda: toolbox.ctx.tf_buffer.can_transform(MAP_FRAME, CAMERA_FRAME, Time())
    )

    info = rig.capture(until=lambda image: in_feed(feed, image))

    frame = feed.people_at(info.header.stamp, 0.0, NO_CANCEL)
    [guest] = frame.msg.people
    # The camera_info is the simulation's, so this is the gate its faces meet.
    assert face_trusted(guest, frame.msg.camera_info, feed.config)
    collected = feed.collected_vectors(frame.tracks[0])
    assert collected.faces and {s.vector.model for s in collected.faces} == {FAKE_FACE_MODEL}
    assert collected.appearances
    assert {s.vector.model for s in collected.appearances} == {FAKE_REID_MODEL}

    out = {tool.name: tool for tool in toolbox.tools()}["find_people"].call()
    _, line = out.summary.splitlines()
    assert line.startswith(f"{out.handles[0]}: centre, ")
    assert line.split(", ")[-1] == FACE_SEEN


def test_no_arrays_while_tf_is_missing_and_they_follow_once_it_comes(make_rig, caplog):
    rig = make_rig(Scene(people=(door_guest(),)))
    # A frame no other test publishes, so no transform of another test stands in.
    frame = f"oracle_test_camera_{random.randint(0, 1_000_000)}"

    def warned(_: CameraInfo) -> bool:
        return any(f"from {MAP_FRAME} to {frame}" in text for text in oracle_warnings(caplog))

    with caplog.at_level(logging.WARNING, logger=oracle_perception.__name__):
        rig.capture(until=warned, frame=frame)
        assert not wait_until(lambda: rig.people or rig.objects, timeout=0.3)

        rig.place_robot(frame)
        info = rig.capture(frame=frame)

    people, _ = rig.arrays(info)
    assert len(people.people) == 1
