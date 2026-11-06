"""OraclePerception: the arrays it draws from a scene, and the oracle over ROS.

The drawing runs without a ROS graph through draw_detections, with
test_projection's simulated head camera on a base at the origin. The object
arrays are checked against walkie-vision 64811fe's contract
(ObjectDetectionArray.msg and walkie_vision.convert's checks). The ROS tests
run the oracle on a node of its own, spun by a MultiThreadedExecutor as it
needs, next to a node that plays walkie-ros's simulation: it publishes the
head camera's camera_info and the robot's transforms at chula_pat14_6's door
pose. All of it runs on Fast DDS and a random domain (ADR-0019).

Ported from the experiment/hri-agent branch at 61d30e8. Its check that
VisionFeed and find_people read the oracle's arrays comes back with S1, which
ports VisionFeed (see-remember-act.md).
"""

import dataclasses
import itertools
import logging
import math
import random
import threading
import time
import uuid
from collections.abc import Callable, Sequence

import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, PointField
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
from tf_transformations import euler_matrix, quaternion_from_matrix
from vision_msgs.msg import BoundingBox2D
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetectionArray

from test_projection import CX, FX, HEIGHT, K, WIDTH, base_from_camera, facing_person, sim_camera
from walkie_agent.perception import OBJECTS_TOPIC, PEOPLE_TOPIC
from walkie_agent.ros.base import MAP_FRAME, ROBOT_FRAME
from walkie_agent.sim import oracle_perception
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, FAKE_REID_MODEL, hash_vector
from walkie_agent.sim.oracle_perception import (
    DEFAULT_ORACLE_CONFIG,
    DETECTION_SCORE,
    FAKE_MODEL_DISTANCE_LIMIT,
    HEAD_CAMERA_INFO_TOPIC,
    KEYPOINT_SCORE,
    SEAT_CLASSES,
    TURNED_FACE_KEYPOINT_SCORE,
    VECTOR_DIM,
    OracleConfig,
    OraclePerception,
    draw_detections,
    is_stream_id,
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
from walkie_agent.sim.scene import COLOR_NAMES, MapBox, Scene, SceneObject, ScenePerson

# The optical frame of the simulation's head camera, which its camera_info's
# header names.
CAMERA_FRAME = "zed_head_left_camera_frame_optical"
STAMP = TimeMsg(sec=5, nanosec=7)
# The stream_id of the arrays draw_detections draws here.
STREAM_ID = "6f1c2a4e-8b3d-4f7a-9c2e-1d5b7a9e3f60"
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


def bbox_xyxy(bbox: BoundingBox2D) -> tuple[float, float, float, float]:
    """A message's box, a centre and a size, as x1, y1, x2, y2."""
    cx, cy = bbox.center.position.x, bbox.center.position.y
    half_w, half_h = bbox.size_x / 2, bbox.size_y / 2
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def wait_until(predicate: Callable[[], object], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class Spinner:
    """Spin nodes on a MultiThreadedExecutor of a test's own, on a daemon thread."""

    def __init__(self, *nodes) -> None:
        self.nodes = nodes
        self.executor = MultiThreadedExecutor()
        for node in nodes:
            self.executor.add_node(node)
        self._thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._thread.start()

    def close(self, timeout_s: float = 10.0) -> None:
        # shutdown() does not wait for spin(), and a destroy under it can raise there.
        self.executor.shutdown()
        self._thread.join(timeout_s)
        assert not self._thread.is_alive(), f"A Spinner's thread did not stop in {timeout_s} s."
        for node in self.nodes:
            node.destroy_node()


def object_array_problem(msg: ObjectDetectionArray) -> str | None:
    """What breaks walkie-vision 64811fe's rules for an ObjectDetectionArray, or None.

    The rules of walkie_vision.convert.check_object_detection_array that an
    array without embeddings, crops or points can break.
    """
    if not is_stream_id(msg.stream_id):
        return f"stream_id {msg.stream_id!r} is not a UUID"
    if msg.camera_info.header != msg.header:
        return "camera_info.header differs from the array's header"
    classes = list(msg.classes)
    if not classes or not all(classes) or len(set(classes)) != len(classes):
        return f"classes {classes} are not unique non-empty names"
    for i, det in enumerate(msg.detections):
        if det.label not in classes:
            return f"detection {i}: the label {det.label!r} is not in classes"
        if not (0.0 <= det.score <= 1.0 and 0.0 <= det.view_quality <= 1.0):
            return f"detection {i}: score or view_quality is out of [0, 1]"
        if det.bbox.center.theta != 0.0 or det.bbox.size_x < 0 or det.bbox.size_y < 0:
            return f"detection {i}: the box is turned or has a negative size"
        if bool(det.embedding_model) != (len(det.embedding) > 0):
            return f"detection {i}: embedding and embedding_model are not both set or both empty"
        cloud = det.points
        fields = [(f.name, f.offset, f.datatype, f.count) for f in cloud.fields]
        if (
            cloud.header != msg.header
            or fields != [(n, o, PointField.FLOAT32, 1) for n, o in (("x", 0), ("y", 4), ("z", 8))]
            or cloud.is_bigendian
            or not cloud.is_dense
            or cloud.height != 1
            or cloud.point_step != 12
            or cloud.row_step != 12 * cloud.width
            or len(cloud.data) != cloud.row_step
        ):
            return f"detection {i}: points does not have the cloud layout"
        if len(det.crop.data) and det.crop.header != msg.header:
            return f"detection {i}: crop.header differs from the array's header"
        shares = np.asarray(det.color_shares, dtype=np.float64)
        if shares.size and (
            shares.shape != (len(COLOR_NAMES),)
            or np.any(shares < 0)
            or abs(shares.sum() - 1.0) > 1e-4
        ):
            return f"detection {i}: color_shares is not empty or 12 shares that sum to 1"
    return None


def camera_info(stamp: TimeMsg = STAMP, frame: str = CAMERA_FRAME) -> CameraInfo:
    """The simulated head camera's camera_info for one image."""
    return CameraInfo(
        header=Header(stamp=stamp, frame_id=frame), width=WIDTH, height=HEIGHT, k=list(K)
    )


def draw(
    scene: Scene, config: OracleConfig = QUIET, seed: int = 0
) -> tuple[PersonDetectionArray, ObjectDetectionArray]:
    """The arrays of one image, from the head camera on a base at the origin."""
    return draw_detections(
        scene, camera_info(), sim_camera(), config, np.random.default_rng(seed), stream_id=STREAM_ID
    )


def draw_images(
    scene: Scene, config: OracleConfig, count: int
) -> list[tuple[PersonDetectionArray, ObjectDetectionArray]]:
    """The arrays of `count` images in a row, all their noise from one generator."""
    rng = np.random.default_rng(0)
    return [
        draw_detections(scene, camera_info(), sim_camera(), config, rng, stream_id=STREAM_ID)
        for _ in range(count)
    ]


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

    assert list(objects.classes) == list(SEAT_CLASSES)
    assert objects.stream_id == STREAM_ID
    [sofa] = objects.detections
    assert (sofa.label, sofa.track_id) == ("sofa", 7)
    assert sofa.score == pytest.approx(DETECTION_SCORE)
    assert bbox_xyxy(sofa.bbox) == pytest.approx(
        clip_box(project_box(SOFA.box, camera, K), WIDTH, HEIGHT)
    )
    # The oracle has no image to cut a crop from and no depth for points.
    assert (sofa.embedding_model, len(sofa.embedding)) == ("", 0)
    assert len(sofa.points.data) == 0 and len(sofa.crop.data) == 0
    # A scene object gives no view quality or colour unless it says so.
    assert sofa.view_quality == 0.0 and len(sofa.color_shares) == 0
    assert object_array_problem(objects) is None


def test_every_object_array_keeps_walkie_visions_rules():
    mug = SceneObject(
        track_id=8,
        label="mug",
        box=MapBox((CAMERA_X + 2.0, CAMERA_Y + 0.3, 0.5), (0.1, 0.1, 0.1)),
        view_quality=0.7,
        color="blue",
    )
    scene = Scene(people=(facing_person(2.0, face="ann"),), objects=(SOFA, mug))
    config = OracleConfig(classes=("sofa", "mug"), dropout_rate=0.3)

    drawn = draw_images(scene, config, 20) + [draw(Scene(), config)]

    assert any(len(objects.detections) == 2 for _, objects in drawn)
    for _, objects in drawn:
        assert object_array_problem(objects) is None
    assert len(drawn[-1][1].detections) == 0


def test_the_object_fields_follow_the_scene_and_share_the_arrays_header():
    mug = SceneObject(
        track_id=8,
        label="mug",
        box=MapBox((CAMERA_X + 2.0, CAMERA_Y + 0.3, 0.5), (0.1, 0.1, 0.1)),
        view_quality=0.7,
        color="blue",
    )

    _, objects = draw(Scene(objects=(SOFA, mug)), dataclasses.replace(QUIET, classes=("sofa", "mug")))

    _, seen = objects.detections
    assert seen.view_quality == pytest.approx(0.7)
    # One share per colour name, all of it on the scene's colour.
    shares = np.asarray(seen.color_shares)
    assert shares.shape == (len(COLOR_NAMES),)
    assert shares[COLOR_NAMES.index("blue")] == 1.0 and shares.sum() == 1.0
    # An empty cloud in walkie-vision's layout, and the crop stamped as the image.
    cloud = seen.points
    assert (cloud.height, cloud.width, cloud.point_step, cloud.row_step) == (1, 0, 12, 0)
    assert [f.name for f in cloud.fields] == ["x", "y", "z"]
    assert cloud.is_dense and not cloud.is_bigendian
    assert cloud.header == objects.header == seen.crop.header
    assert objects.camera_info.header == objects.header


@pytest.mark.parametrize(
    ("stream_id", "valid"),
    [
        (STREAM_ID, True),
        (STREAM_ID.upper(), False),
        (STREAM_ID.replace("-", ""), False),
        ("", False),
        ("robot-1", False),
    ],
)
def test_a_stream_id_is_a_canonical_uuid(stream_id, valid):
    # walkie-vision compares str(uuid.UUID(s)) with s, so only the lower-case
    # hyphenated form passes.
    assert is_stream_id(stream_id) is valid


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
    for field, key in (("face_embedding", "face:ann"), ("appearance_embedding", "reid:ann")):
        clean = hash_vector(key, VECTOR_DIM)
        vectors = [np.asarray(getattr(person, field)) for person in seen]
        to_clean = [cosine_distance(v, clean) for v in vectors]
        between = [cosine_distance(a, b) for a, b in itertools.combinations(vectors, 2)]
        # The arithmetic of OracleConfig.vector_noise: about 0.006 from the
        # clean vector and 0.013 between two noisy ones.
        assert np.mean(to_clean) == pytest.approx(0.0063, rel=0.15)
        assert np.mean(between) == pytest.approx(0.0126, rel=0.15)
        assert max(between) < FAKE_MODEL_DISTANCE_LIMIT / 5


def test_outliers_are_random_vectors_far_outside_the_fake_models_limit():
    config = OracleConfig(outlier_rate=0.25)
    scene = Scene(people=(facing_person(2.0, face="ann"),))
    vectors = [
        list(arrays[0].people[0].face_embedding) for arrays in draw_images(scene, config, 40)
    ]
    clean = hash_vector("face:ann", VECTOR_DIM)

    outliers = [v for v in vectors if cosine_distance(v, clean) > FAKE_MODEL_DISTANCE_LIMIT]
    inliers = [v for v in vectors if cosine_distance(v, clean) <= FAKE_MODEL_DISTANCE_LIMIT]

    assert 3 <= len(outliers) <= 20
    # A random unit vector is nearly orthogonal to the person's: the cosine
    # of two random directions in 512 dimensions is about 1 / sqrt(512).
    assert all(abs(1.0 - cosine_distance(v, clean)) < 0.25 for v in outliers)
    # Nothing lies between: the rest are the person's, with the vector noise only.
    assert all(cosine_distance(v, clean) < FAKE_MODEL_DISTANCE_LIMIT / 10 for v in inliers)


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


@pytest.mark.parametrize("classes", [(), ("sofa", "sofa"), ("sofa", "")])
def test_classes_that_walkie_vision_would_refuse_are_refused(classes):
    with pytest.raises(ValueError, match="classes"):
        OracleConfig(classes=classes)


# -- Skipped images, with the callback called directly -------------------------------------


@pytest.fixture
def idle_node(ros):
    """A node no executor spins, for calling the oracle's callback directly."""
    node = rclpy.create_node(f"oracle_idle_{random.randint(0, 1_000_000)}")
    yield node
    node.destroy_node()


def test_each_oracle_has_a_stream_id_of_its_own_unless_given_one(idle_node):
    first = OraclePerception(idle_node, SceneAt(), tf_buffer=Buffer())
    second = OraclePerception(idle_node, SceneAt(), tf_buffer=Buffer())
    given = OraclePerception(idle_node, SceneAt(), tf_buffer=Buffer(), stream_id=STREAM_ID)

    assert is_stream_id(first.stream_id) and is_stream_id(second.stream_id)
    assert uuid.UUID(first.stream_id).version == 4
    assert first.stream_id != second.stream_id
    assert given.stream_id == STREAM_ID
    with pytest.raises(ValueError, match="stream_id"):
        OraclePerception(idle_node, SceneAt(), tf_buffer=Buffer(), stream_id="robot-1")
    for oracle in (first, second, given):
        oracle.destroy()


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
        lambda scene, info, camera_from_map, config, rng, stream_id: (
            drawn.append(camera_from_map) or (PersonDetectionArray(), ObjectDetectionArray())
        ),
    )
    scene = SceneAt()
    oracle = OraclePerception(idle_node, scene, QUIET, tf_buffer=buffer)

    oracle._on_camera_info(camera_info(STAMP))

    [camera_from_map] = drawn
    assert camera_from_map == pytest.approx(first, abs=1e-9)
    assert scene.asked == [Time.from_msg(STAMP)]


def camera_buffer() -> Buffer:
    """A buffer with the head camera on a base at the origin, at any stamp."""
    buffer = Buffer()
    buffer.set_transform_static(
        tf_msg(MAP_FRAME, CAMERA_FRAME, np.linalg.inv(sim_camera())), "test"
    )
    return buffer


def published(oracle: OraclePerception, monkeypatch) -> list:
    """Every array `oracle` publishes from here on."""
    sent: list = []
    monkeypatch.setattr(oracle._people_pub, "publish", sent.append)
    monkeypatch.setattr(oracle._objects_pub, "publish", sent.append)
    return sent


def test_an_image_whose_scene_is_unknown_is_skipped_with_a_throttled_warning(
    idle_node, caplog, monkeypatch
):
    asked: list[Time] = []

    def unknown(stamp: Time) -> Scene | None:
        asked.append(stamp)
        return None

    oracle = OraclePerception(idle_node, unknown, QUIET, tf_buffer=camera_buffer())
    sent = published(oracle, monkeypatch)

    with caplog.at_level(logging.WARNING, logger=oracle_perception.__name__):
        for _ in range(3):
            oracle._on_camera_info(camera_info())

    assert len(asked) == 3 and sent == []
    [text] = oracle_warnings(caplog)
    assert "No scene is known at the image's stamp" in text


def test_an_image_during_which_the_clock_went_back_is_skipped_with_a_warning(
    idle_node, caplog, monkeypatch
):
    # The epoch is read before each image's TF lookup and after its scene:
    # the first image spans a jump, the second does not.
    epochs = iter([0, 1, 1, 1])
    scene = SceneAt(Scene(people=(facing_person(2.0),)))
    oracle = OraclePerception(
        idle_node, scene, QUIET, tf_buffer=camera_buffer(), epoch=lambda: next(epochs)
    )
    sent = published(oracle, monkeypatch)

    with caplog.at_level(logging.WARNING, logger=oracle_perception.__name__):
        oracle._on_camera_info(camera_info())
        assert sent == []
        oracle._on_camera_info(camera_info())

    assert len(scene.asked) == 2
    people, objects = sent
    assert len(people.people) == 1 and isinstance(objects, ObjectDetectionArray)
    [text] = oracle_warnings(caplog)
    assert "the clock went back" in text


def test_an_image_whose_drawing_spans_a_clock_jump_is_not_published(
    idle_node, caplog, monkeypatch
):
    # The clock goes back while the arrays are drawn, after the scene was read.
    jumps = [0]
    draw = oracle_perception.draw_detections

    def draw_across_a_jump(*args, **kwargs):
        arrays = draw(*args, **kwargs)
        jumps[0] += 1
        return arrays

    monkeypatch.setattr(oracle_perception, "draw_detections", draw_across_a_jump)
    scene = SceneAt(Scene(people=(facing_person(2.0),)))
    oracle = OraclePerception(
        idle_node, scene, QUIET, tf_buffer=camera_buffer(), epoch=lambda: jumps[0]
    )
    sent = published(oracle, monkeypatch)

    with caplog.at_level(logging.WARNING, logger=oracle_perception.__name__):
        oracle._on_camera_info(camera_info())

    assert len(scene.asked) == 1 and jumps == [1]
    assert sent == []
    [text] = oracle_warnings(caplog)
    assert "the clock went back" in text


# -- Over ROS ----------------------------------------------------------------------------


class Rig:
    """OraclePerception on a node of its own, and a node that plays the simulation.

    The simulation node publishes the head camera's camera_info and the
    robot's transforms on /tf_static, and keeps every array the oracle
    publishes, by stamp. One Spinner's MultiThreadedExecutor spins both nodes.
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
        self.spinner = Spinner(self.oracle_node, self.sim)
        self._count = 0

    def wait_for_discovery(self) -> None:
        """Wait until the two nodes' publishers and subscriptions have matched each other.

        Both ways: a publisher sends nothing to a subscription it has not
        matched, and the oracle's best-effort subscription drops a
        camera_info from a publisher it has not matched yet.
        """
        oracle = self.oracle
        assert wait_until(
            lambda: self.info_pub.get_subscription_count() > 0
            and oracle._subscription.get_publisher_count() > 0
            and oracle._people_pub.get_subscription_count() > 0
            and oracle._objects_pub.get_subscription_count() > 0
            and all(s.get_publisher_count() > 0 for s in self.subscriptions),
            timeout=10.0,
        ), "The oracle's subscription and publishers were not discovered."

    def place_robot(self, frame: str = CAMERA_FRAME) -> None:
        """Publish the base at the door pose and the head camera's mount on it.

        Returns once the oracle's TF buffer has them, so the next image is
        answered.
        """
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
        assert wait_until(
            lambda: self.oracle._tf_buffer.can_transform(frame, MAP_FRAME, Time()), timeout=10.0
        ), "The robot's transforms did not reach the oracle's TF buffer."

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
        """Show one image, and return it once `until(image)` holds.

        By default, until the oracle's arrays of the image arrived. The
        nodes have matched (wait_for_discovery) and place_robot waits for
        the oracle's TF, so the one image is answered.
        """
        answered = self.has_arrays if until is None else until
        info = self.show(frame)
        assert wait_until(lambda: answered(info), timeout=10.0), (
            "The camera_info was not answered within 10 s."
        )
        return info

    def has_arrays(self, info: CameraInfo) -> bool:
        key = Time.from_msg(info.header.stamp).nanoseconds
        return key in self.people and key in self.objects

    def arrays(self, info: CameraInfo) -> tuple[PersonDetectionArray, ObjectDetectionArray]:
        key = Time.from_msg(info.header.stamp).nanoseconds
        return self.people[key], self.objects[key]

    def close(self) -> None:
        # The oracle's subscription, publishers and TF listener go with its node.
        self.spinner.close()

    def _on_people(self, msg: PersonDetectionArray) -> None:
        self.people[Time.from_msg(msg.header.stamp).nanoseconds] = msg

    def _on_objects(self, msg: ObjectDetectionArray) -> None:
        self.objects[Time.from_msg(msg.header.stamp).nanoseconds] = msg


@pytest.fixture
def make_rig(ros):
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


def test_each_camera_info_gets_both_arrays_with_its_header_also_when_empty(make_rig):
    rig = make_rig()
    rig.place_robot()

    info = rig.capture()
    later = rig.capture()

    people, objects = rig.arrays(info)
    for array in (people, objects):
        assert array.header == info.header
        assert array.camera_info == info
    assert len(people.people) == 0 and len(objects.detections) == 0
    assert list(objects.classes) == list(rig.config.classes)
    assert object_array_problem(objects) is None
    # One stream_id for every array of the run.
    assert objects.stream_id == rig.arrays(later)[1].stream_id == rig.oracle.stream_id
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


def test_by_default_the_oracle_publishes_on_walkie_visions_topics():
    # The tests above use topics of their own; a run uses walkie-vision's.
    config = DEFAULT_ORACLE_CONFIG
    assert (config.people_topic, config.objects_topic) == (PEOPLE_TOPIC, OBJECTS_TOPIC)
    assert (PEOPLE_TOPIC, OBJECTS_TOPIC) == ("/walkie_vision/people", "/walkie_vision/objects")
    assert config.camera_info_topic == HEAD_CAMERA_INFO_TOPIC


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
