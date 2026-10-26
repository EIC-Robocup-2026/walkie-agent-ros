"""OraclePerception: walkie-vision's people and object arrays, drawn from a simulated scene.

walkie-ros's MuJoCo worlds have no person bodies, and the agent's perception
atomics read walkie-vision's arrays rather than images. In a simulation,
OraclePerception stands in for walkie-vision (perception.md, "In
simulation"): for every camera_info of the robot's camera it asks a scene
callable who and what is where in the map, projects them into that image
(walkie_agent.sim.projection), and publishes a PersonDetectionArray and an
ObjectDetectionArray as walkie-vision does, one of each per image, also when
they are empty. An image without the camera's pose, without a known scene,
or during which the clock went back gets none.

It knows nothing of a task: the HRI scenario feeds it a Scene now, and the
harvest can later. Faces and appearances are hash vectors under walkie-vision's
fake model names (walkie_agent.sim.fake_vectors), so the simulation
recognizes people by both, as the robot does.

The camera's pose comes from TF at each camera_info's stamp, not from the
simulation's ground truth: the agent turns and judges headings with the
same TF, so the boxes always agree with its frame. A localization error
then shifts the whole virtual scene against the simulated room, which
nothing here looks at. There is no occlusion model: furniture, walls and
door frames hide nobody, and people do not hide each other.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformException, TransformListener
from tf_transformations import quaternion_matrix
from vision_msgs.msg import BoundingBox2D
from walkie_vision_msgs.msg import (
    ObjectDetection,
    ObjectDetectionArray,
    PersonDetection,
    PersonDetectionArray,
)

from walkie_agent.perception.geometry import MAP_FRAME
from walkie_agent.perception.seating import DEFAULT_SEATING_CONFIG
from walkie_agent.perception.vision_feed import OBJECTS_TOPIC, PEOPLE_TOPIC
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, FAKE_REID_MODEL, hash_vector
from walkie_agent.sim.projection import (
    FACE_KEYPOINTS,
    Box,
    camera_position,
    clip_box,
    in_image,
    project_box,
    project_person,
)
from walkie_agent.sim.scene import FACE_LIMIT_DEG, Scene, ScenePerson

logger = logging.getLogger(__name__)

# The head camera of walkie-ros's MuJoCo simulation: 640 x 360 px, about
# 4.7 images per second of sim time. Its header names the optical frame,
# zed_head_left_camera_frame_optical.
HEAD_CAMERA_INFO_TOPIC = "/zed_head/zed_node/depth/camera_info"

# RELIABLE, so a reliable subscriber such as a bag recorder gets every
# array, and VisionFeed's BEST_EFFORT subscription accepts it as well.
# VOLATILE and a short depth, as walkie-vision: only the newest images matter.
ORACLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

# The length of walkie-vision's fake face and appearance vectors.
VECTOR_DIM = 512
# The detector's confidence in every person, face and object it draws.
DETECTION_SCORE = 0.9
KEYPOINT_SCORE = 0.9
# The face keypoints of a person turned away from the camera: under the
# gesture rules' 0.3, as a pose estimator guesses points it cannot see.
TURNED_FACE_KEYPOINT_SCORE = 0.2


@dataclass(frozen=True)
class OracleConfig:
    """What OraclePerception reads and publishes, and the noise it adds."""

    camera_info_topic: str = HEAD_CAMERA_INFO_TOPIC
    people_topic: str = PEOPLE_TOPIC
    objects_topic: str = OBJECTS_TOPIC
    # The detector's text prompts, ObjectDetectionArray.classes. An object
    # of another class is not detected. The default is the seat classes,
    # what find_seats looks for, the only reader of objects so far.
    classes: tuple[str, ...] = DEFAULT_SEATING_CONFIG.seat_classes
    # The seed of the one random generator behind all the noise, so a run
    # can be repeated image by image.
    seed: int = 0
    # The standard deviation of the Gaussian noise on each box edge and
    # keypoint coordinate, in pixels.
    pixel_noise_px: float = 2.0
    # The standard deviation of the Gaussian noise added to each element of
    # a vector before it is made unit length again. A per-element sigma s
    # over 512 dimensions moves a unit vector by about s * sqrt(512): 0.113
    # at the default. The cosine distance to the clean vector is then about
    # 1 - 1 / sqrt(1 + 512 s^2) = 0.006, and between two noisy vectors of
    # one person 512 s^2 / (1 + 512 s^2) = 0.013. The fake models' limit is
    # 0.2 (walkie_agent.perception.identity), which two noisy vectors reach
    # at s = 0.022 and a noisy one reaches against the clean at s = 0.033.
    vector_noise: float = 0.005
    # The share of vectors replaced by a random unit vector, as a face
    # recognizer's bad crop, so that the outlier rejection of
    # walkie_agent.perception.identity gets something to reject.
    outlier_rate: float = 0.05
    # The share of images from which a person is left out, each on its own.
    dropout_rate: float = 0.0
    # A face is drawn only for a person facing the camera within this angle.
    face_limit_deg: float = FACE_LIMIT_DEG
    # How long each image waits for TF, camera <- map at its stamp.
    tf_timeout_s: float = 0.1
    # A warning about skipped images repeats at most this often, in seconds.
    warning_period_s: float = 5.0

    def __post_init__(self) -> None:
        for name in ("outlier_rate", "dropout_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"OracleConfig.{name} must lie in [0, 1].")
        for name in ("pixel_noise_px", "vector_noise", "warning_period_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"OracleConfig.{name} must not be negative.")
        if not self.tf_timeout_s > 0:
            raise ValueError("OracleConfig.tf_timeout_s must be positive.")
        if not 0.0 <= self.face_limit_deg <= 180.0:
            raise ValueError("OracleConfig.face_limit_deg must lie in [0, 180].")


DEFAULT_ORACLE_CONFIG = OracleConfig()


class OraclePerception:
    """Publishes walkie-vision's arrays for a simulated scene, one pair per camera_info.

    `scene` gives the Scene at a camera_info's stamp, or None when it does
    not know the scene then, such as for a stamp older than the history it
    keeps. Such an image is skipped, with a warning. It is called on the
    executor thread that runs the camera_info callback, so it must be safe
    to call while the node's other callbacks run. Spin `node` with a
    MultiThreadedExecutor: each image waits up to `tf_timeout_s` for TF on
    that thread, and the TF listener's callbacks must run meanwhile.

    `tf_buffer` shares a buffer the node fills already, such as one that
    also gives the scenario the robot's pose. Without one, OraclePerception
    listens to TF on `node` itself.

    `epoch`, if given, counts the times the clock went back. It is read
    before the camera's pose is looked up and again once the scene is
    drawn. An image for which it changed is skipped, with a warning: its
    pose and its scene may come from different clocks.
    """

    def __init__(
        self,
        node: Node,
        scene: Callable[[Time], Scene | None],
        config: OracleConfig = DEFAULT_ORACLE_CONFIG,
        *,
        tf_buffer: Buffer | None = None,
        epoch: Callable[[], int] | None = None,
    ) -> None:
        self._node = node
        self._scene = scene
        self._epoch = epoch
        self._config = config
        # Used only in the camera_info callback, whose callback group runs
        # one image at a time.
        self._rng = np.random.default_rng(config.seed)
        self._warned_at: dict[str, float] = {}
        self._listener: TransformListener | None = None
        if tf_buffer is None:
            # The node's clock, so a simulation restart that moves sim time
            # back clears the buffer.
            tf_buffer = Buffer(node=node)
            self._listener = TransformListener(tf_buffer, node)
        self._tf_buffer = tf_buffer
        self._people_pub = node.create_publisher(
            PersonDetectionArray, config.people_topic, ORACLE_QOS
        )
        self._objects_pub = node.create_publisher(
            ObjectDetectionArray, config.objects_topic, ORACLE_QOS
        )
        # BEST_EFFORT, which matches the camera's publisher whatever its
        # reliability. A group of its own, so the wait for TF holds up no
        # other callback of the node.
        self._subscription = node.create_subscription(
            CameraInfo,
            config.camera_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    @property
    def config(self) -> OracleConfig:
        return self._config

    def destroy(self) -> None:
        """Stop publishing: remove the subscription, the publishers and an own TF listener."""
        self._node.destroy_subscription(self._subscription)
        self._node.destroy_publisher(self._people_pub)
        self._node.destroy_publisher(self._objects_pub)
        if self._listener is not None:
            self._listener.unregister()

    def _on_camera_info(self, info: CameraInfo) -> None:
        if not (info.k[0] > 0 and info.k[4] > 0 and info.width > 0 and info.height > 0):
            self._warn(
                "intrinsics",
                f"A camera_info on {self._config.camera_info_topic} has no focal length or "
                "no image size, so no detections were published for its image.",
            )
            return
        stamp = Time.from_msg(info.header.stamp)
        epoch = None if self._epoch is None else self._epoch()
        try:
            transform = self._tf_buffer.lookup_transform(
                info.header.frame_id,
                MAP_FRAME,
                stamp,
                timeout=Duration(seconds=self._config.tf_timeout_s),
            )
        except TransformException as e:
            # Until the simulation's localization runs, map does not exist.
            self._warn(
                "tf",
                f"TF has no transform from {MAP_FRAME} to {info.header.frame_id or '(no frame)'} "
                f"at the image's stamp, so no detections were published for it: {e}",
            )
            return
        scene = self._scene(stamp)
        if self._epoch is not None and self._epoch() != epoch:
            self._warn(
                "clock",
                "An image was drawn while the clock went back, so no detections were published "
                "for it: its camera pose and its scene may come from different clocks.",
            )
            return
        if scene is None:
            self._warn(
                "scene",
                "No scene is known at the image's stamp, as for an image older than the "
                "scene's history, so no detections were published for it.",
            )
            return
        people, objects = draw_detections(
            scene, info, transform_matrix(transform), self._config, self._rng
        )
        self._people_pub.publish(people)
        self._objects_pub.publish(objects)

    def _warn(self, key: str, text: str) -> None:
        """Log `text`, unless a warning of the same `key` was logged within the period."""
        now = time.monotonic()
        last = self._warned_at.get(key)
        if last is not None and now - last < self._config.warning_period_s:
            return
        self._warned_at[key] = now
        logger.warning(text)


def transform_matrix(transform: TransformStamped) -> np.ndarray:
    """A TF transform as a 4x4 matrix that takes points in the child frame into the parent's."""
    q = transform.transform.rotation
    t = transform.transform.translation
    matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
    matrix[:3, 3] = (t.x, t.y, t.z)
    return matrix


def draw_detections(
    scene: Scene,
    camera_info: CameraInfo,
    camera_from_map: np.ndarray,
    config: OracleConfig,
    rng: np.random.Generator,
) -> tuple[PersonDetectionArray, ObjectDetectionArray]:
    """The arrays walkie-vision would publish for the image that `camera_info` describes.

    `camera_from_map` takes map-frame points into the camera's optical
    frame at the image's stamp. Both arrays carry the camera_info's header
    and the camera_info itself. All the noise comes from `rng`.
    """
    people = PersonDetectionArray(header=camera_info.header, camera_info=camera_info)
    camera_xy = tuple(float(v) for v in camera_position(camera_from_map)[:2])
    for person in scene.people:
        if rng.random() < config.dropout_rate:
            continue
        detection = _person_detection(person, camera_info, camera_from_map, camera_xy, config, rng)
        if detection is not None:
            people.people.append(detection)

    objects = ObjectDetectionArray(
        header=camera_info.header, camera_info=camera_info, classes=list(config.classes)
    )
    for thing in scene.objects:
        if thing.label not in config.classes:
            continue
        projected = project_box(thing.box, camera_from_map, camera_info.k)
        box = None if projected is None else _noisy_box(projected, camera_info, config, rng)
        if box is None:
            continue
        objects.detections.append(
            ObjectDetection(
                label=thing.label,
                score=DETECTION_SCORE,
                track_id=thing.track_id,
                bbox=_bbox_msg(box),
            )
        )
    return people, objects


def _person_detection(
    person: ScenePerson,
    camera_info: CameraInfo,
    camera_from_map: np.ndarray,
    camera_xy: tuple[float, float],
    config: OracleConfig,
    rng: np.random.Generator,
) -> PersonDetection | None:
    """One person's detection, or None when no part of their box is in the image."""
    view = project_person(person, camera_from_map, camera_info.k)
    box = None if view is None else _noisy_box(view.box, camera_info, config, rng)
    if box is None:
        return None
    width, height = camera_info.width, camera_info.height

    # A keypoint behind the camera or outside the image scores 0, as a pose
    # estimator's guess, and is moved into the image: walkie-vision's
    # keypoints are always pixels of the image.
    scores = np.where(in_image(view.keypoints, width, height), KEYPOINT_SCORE, 0.0)
    faces = person.faces_camera(camera_xy, config.face_limit_deg)
    if not faces:
        face_scores = scores[list(FACE_KEYPOINTS)]
        scores[list(FACE_KEYPOINTS)] = np.minimum(face_scores, TURNED_FACE_KEYPOINT_SCORE)
    noisy = view.keypoints + rng.normal(0.0, config.pixel_noise_px, view.keypoints.shape)
    noisy = np.nan_to_num(noisy, nan=0.0)
    noisy[:, 0] = np.clip(noisy[:, 0], 0.0, width)
    noisy[:, 1] = np.clip(noisy[:, 1], 0.0, height)

    msg = PersonDetection(score=DETECTION_SCORE, track_id=person.track_id, bbox=_bbox_msg(box))
    msg.keypoints = noisy.reshape(-1).astype(np.float32)
    msg.keypoint_scores = scores.astype(np.float32)
    # The face as drawn, never enlarged: whether it is large enough to name
    # the person by is VisionFeed's face_trusted to decide, as on the robot.
    if faces and person.face is not None and view.face_box is not None:
        face_box = _noisy_box(view.face_box, camera_info, config, rng)
        if face_box is not None:
            msg.face_bbox = _bbox_msg(face_box)
            msg.face_det_score = DETECTION_SCORE
            msg.face_model = FAKE_FACE_MODEL
            msg.face_embedding = _vector("face:" + person.face, config, rng)
    if person.appearance is not None:
        msg.appearance_model = FAKE_REID_MODEL
        msg.appearance_embedding = _vector("reid:" + person.appearance, config, rng)
    return msg


def _noisy_box(
    box: Box, camera_info: CameraInfo, config: OracleConfig, rng: np.random.Generator
) -> Box | None:
    """The box with noise on each edge, clipped to the image; None when it is not in view.

    Whether the box is in view is decided before the noise, so the noise
    never brings a box into view that the camera cannot see.
    """
    width, height = camera_info.width, camera_info.height
    if clip_box(box, width, height) is None:
        return None
    x1, y1, x2, y2 = np.asarray(box, dtype=float) + rng.normal(0.0, config.pixel_noise_px, 4)
    return clip_box((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), width, height)


def _vector(key: str, config: OracleConfig, rng: np.random.Generator) -> list[float]:
    """The person's hash vector with noise, or at `outlier_rate` a random one; unit length."""
    if rng.random() < config.outlier_rate:
        vector = rng.standard_normal(VECTOR_DIM)
    else:
        vector = hash_vector(key, VECTOR_DIM) + rng.normal(0.0, config.vector_noise, VECTOR_DIM)
    return (vector / np.linalg.norm(vector)).astype(np.float32).tolist()


def _bbox_msg(box: Box) -> BoundingBox2D:
    x1, y1, x2, y2 = box
    msg = BoundingBox2D()
    msg.center.position.x, msg.center.position.y = (x1 + x2) / 2, (y1 + y2) / 2
    msg.size_x, msg.size_y = x2 - x1, y2 - y1
    return msg
