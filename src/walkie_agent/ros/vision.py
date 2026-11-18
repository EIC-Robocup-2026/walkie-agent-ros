"""ctx.vision on the robot with --vision: RosVision over walkie-vision's arrays.

Without --vision, main uses capabilities.UnavailableVision, which raises UNAVAILABLE
from both verbs.

RosVision reads people from the newest PersonDetectionArray on /walkie_vision/people,
and objects from the newest ObjectDetectionArray on /walkie_vision/objects, which
walkie-vision publishes on the robot and oracle_perception in the simulation. The
objects topic can be another, such as walkie-vision's /walkie_vision/sim/objects, so a
detector on the simulation's images can stand in for the oracle's objects.

- An array counts only while its header stamp is within MAX_AGE_S of the node's clock,
  so the age is right under sim time too. people() and objects() each wait up to
  WAIT_S for one, then raise UNAVAILABLE with the reason: no array, an old one, or a
  clock mismatch.
- Two subscriptions, made in the constructor and destroyed by close(). Their callbacks
  run on the connector's executor and only keep the newest array of each topic by
  stamp. The reader converts it, on the agent's worker.
- An object's name is its label, and its category comes from perception/object_words.py.
  Its id is <label>_<track_id>, with spaces in the label as underscores, as FakeRobot
  spells its ids; one the tracker has not taken up is <label>_untracked_<n>, numbered
  left to right in that one array. An id names the object only while it stays in view.
- An object's place is the configured place within PLACE_TOLERANCE_M of the robot, or
  UNKNOWN_PLACE away from every place: without depth the object's own position is not
  known, so the place is where Walkie stands. Size, weight and thickness are None.
- objects() raises UNAVAILABLE for a query that no class of the array could match,
  such as "apple" while the detector looks only for seats, since [] would claim that
  none is in view.
- A person's id is person_<track_id>, the tracker's id, which walkie-vision keeps while
  the track lasts. A person the tracker has not taken up has track_id 0 and gets
  person_untracked_<n>, numbered left to right in that one array.
- The callback also collects each tracked person's face and appearance vectors into a
  TrackVectors (people_memory/track_vectors.py), from every array, the late ones too. A
  face counts only when face_trusted passes it with the array's camera_info. The
  buffer is cleared when the clock jumps back and by close(). As a VectorSource,
  collected() and tracked_in_view() give TrackMemory (people_memory/track_memory.py) the
  samples to remember and recognize people by (ADR-0070).
- With a namer, set by set_namer(), people() names the tracked people of the array
  before it applies the query, waiting at most NAME_WAIT_S and never past the call's
  timeout_s. A query then matches a person's name too, ignoring case. A failure of
  the namer leaves the names None and is logged; a stop still ends the call.
- The room is the room of the configured place nearest the robot, read once per call,
  and only when someone matches. The pose read and the TF lookup wait only for what
  is left of the call's timeout_s; past it, the people come with no room and no
  bearing, so a call ends close to its timeout_s.
- A person's direction from the camera is the mean map-frame yaw of the rays through
  their shoulders and hips, or through their box's centre with fewer than two of
  those seen, from the array's camera_info and TF map <- the array's frame at its
  stamp. TF is looked up once per call, waiting up to TF_WAIT_S, and with no
  transform every bearing is None.
- The camera sits 0.28 m ahead of the base's turning centre, so a turn by that
  direction alone ends past a person to the side: 4 degrees past the simulated
  guest, 1.8 m away and 36 degrees to the left. With an ankle in view, the rays
  through the ankles meet the plane ANKLE_HEIGHT_M above the map's floor at the
  person's distance, and the bearing is the direction of that point from the base's
  pose now, less its yaw. So a turn or a move since the image counts. Without an
  ankle, or with a distance past MAX_FLOOR_RANGE_M, the bearing is the camera's
  direction less the base's yaw now.

Ported in part from the experiment/hri-agent branch at 61d30e8 (its perception
vision_feed module and the people half of its RosVision). The vector collection
follows that feed's, without its generation keys.
"""

import logging
import math
import re
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import TransformException
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetectionArray

from walkie_agent.capabilities import Base, ObjectSeen, PersonSeen, Pose2D
from walkie_agent.capabilities.guards import CancelToken, begin_lookup
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.people_memory.track_vectors import (
    DEFAULT_TRACK_VECTORS_CONFIG,
    Collected,
    Sighting,
    TrackVectors,
    TrackVectorsConfig,
    face_trusted,
    model_vector,
)
from walkie_agent.perception import OBJECTS_TOPIC, PEOPLE_TOPIC
from walkie_agent.perception.geometry import (
    ankle_pixels,
    floor_range,
    mean_yaw,
    torso_pixels,
    wrap_deg,
    yaw_from,
)
from walkie_agent.perception.gestures import DEFAULT_GESTURE_CONFIG, GestureConfig, classify_gestures
from walkie_agent.perception.object_words import category_of, looks_for, matches
from walkie_agent.perception.people_words import gesture_phrase, phrases, pose_phrase
from walkie_agent.ros.base import MAP_FRAME
from walkie_agent.ros.policy import FORBIDDEN_NAMES, check_endpoint, ros_failure

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

logger = logging.getLogger(__name__)

# Both subscriptions' profile. BEST_EFFORT matches a reliable and a best-effort
# publisher alike, where a reliable subscription would miss a best-effort one, and
# walkie-vision fixes no QoS. VOLATILE and a short depth, because only the newest image
# matters.
PEOPLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
# How old an array may be and still count as what is in view (ADR-0050).
MAX_AGE_S = 1.0
# How long people() and objects() wait for a fresh array before they report UNAVAILABLE.
WAIT_S = 1.0
# An array stamped this much before the newest one means the clock went back, as a
# simulation restart does. It replaces the newest, which would otherwise hide the new
# run's arrays until their stamps passed it.
EPOCH_JUMP_S = 5.0
# How long people() waits for TF to reach an array's stamp before it gives no bearings.
TF_WAIT_S = 0.2
# The longest wait of people() for the names of the people in view (decision S3-4).
NAME_WAIT_S = 2.0
# Within this distance of a configured place, the objects in view are at that place,
# as FakeRobot's NEAR_PLACE_M has it (decision S4-3).
PLACE_TOLERANCE_M = 1.0
# An object's place when Walkie stands at no configured place, or its pose is not known.
UNKNOWN_PLACE = "unknown"
PUBLISHER_HINT = "Is walkie-vision or oracle_perception running?"

_POLL_S = 0.02
_NS_PER_S = 1_000_000_000
# The id of a tracked person, as people() gives it: no leading zero, never track 0.
_TRACKED_ID = re.compile(r"person_([1-9][0-9]*)")

# The map <- camera transform at an array's stamp: the rotation as x, y, z, w, and the
# optical centre's position as x, y, z.
_Camera = tuple[tuple[float, float, float, float], tuple[float, float, float]]


class Namer(Protocol):
    """What set_namer takes: TrackMemory.names on the robot.

    Gives the names of those of `person_ids` it knows, within `timeout_s`. It raises
    StepError CANCELLED on a stop.
    """

    def __call__(self, person_ids: Sequence[str], *, timeout_s: float) -> Mapping[str, str]: ...


class RosVision:
    """people() and objects() from walkie-vision's newest people and object arrays.

    `base` gives the pose and the places that name the room and the objects' place.
    `topic` is the people topic and `objects_topic` the objects one. `tf_buffer` is a
    tf2 Buffer and defaults to the connector's own. `vectors` sets what the vector
    buffer keeps. Thread-safe: verbs run on LangChain workers, the callbacks on the
    executor and close() on any thread.
    """

    def __init__(
        self,
        connector: "ROS2Connector",
        base: Base,
        cancel: CancelToken,
        *,
        topic: str = PEOPLE_TOPIC,
        objects_topic: str = OBJECTS_TOPIC,
        forbidden: Iterable[str] = FORBIDDEN_NAMES,
        max_age_s: float = MAX_AGE_S,
        wait_s: float = WAIT_S,
        epoch_jump_s: float = EPOCH_JUMP_S,
        tf_wait_s: float = TF_WAIT_S,
        name_wait_s: float = NAME_WAIT_S,
        gestures: GestureConfig = DEFAULT_GESTURE_CONFIG,
        vectors: TrackVectorsConfig = DEFAULT_TRACK_VECTORS_CONFIG,
        tf_buffer: Any = None,
    ) -> None:
        self._cancel = cancel
        self._node = connector.node
        self._base = base
        self._topic = check_endpoint(self._node, topic, forbidden)
        self._objects_topic = check_endpoint(self._node, objects_topic, forbidden)
        # RAI's get_transform reads only the latest transform, and a bearing needs the
        # one at the image's stamp. So this reads the Buffer that the connector's own
        # listener fills, a private attribute of the pinned RAI: a RAI that renames it
        # fails here, at start, and a second listener would subscribe to TF twice.
        self._tf = connector._tf_buffer if tf_buffer is None else tf_buffer
        self._tf_wait_s = tf_wait_s
        self._max_age_s = max_age_s
        self._wait_s = wait_s
        self._epoch_jump_ns = round(epoch_jump_s * _NS_PER_S)
        self._name_wait_s = name_wait_s
        self._gestures = gestures
        self._namer: Namer | None = None
        self._lock = threading.Lock()
        # Notified on every kept array and on close(), so a wait ends at once.
        self._arrived = threading.Condition(self._lock)
        # The newest array of each topic by stamp, with that stamp in nanoseconds.
        self._newest: tuple[int, PersonDetectionArray] | None = None
        self._newest_objects: tuple[int, ObjectDetectionArray] | None = None
        # Added to and cleared under _lock, so nothing is added after close().
        self._vectors = TrackVectors(vectors)
        self._closed = False
        self._subscription = self._node.create_subscription(
            PersonDetectionArray, self._topic, self._on_people, PEOPLE_QOS
        )
        try:
            self._objects_subscription = self._node.create_subscription(
                ObjectDetectionArray, self._objects_topic, self._on_objects, PEOPLE_QOS
            )
        except BaseException:
            self._node.destroy_subscription(self._subscription)
            raise

    def set_namer(self, namer: Namer | None) -> None:
        """Let people() name the tracked people in view with `namer`; None stops it.

        main sets TrackMemory.names, which reads this vision's samples, so it cannot
        be given to the constructor.
        """
        self._namer = namer

    def objects(self, query: str = "", *, timeout_s: float = 5.0) -> list[ObjectSeen]:
        what = "looking for objects"
        begin_lookup(self._cancel, query, timeout_s, what)
        deadline = time.monotonic() + timeout_s
        message = self._fresh_array(min(self._wait_s, timeout_s), what, objects=True)
        wanted = query.strip().lower()
        if not looks_for(wanted, message.classes):
            raise StepError(ErrorCode.UNAVAILABLE, not_looked_for(query, message.classes))
        seen = [
            (label, object_id)
            for label, object_id in _objects_left_to_right(message)
            if matches(wanted, label)
        ]
        # Before the empty answer too, as in people().
        self._cancel.check(what)
        if not seen:
            return []
        place = self._place(self._pose(deadline, "find_objects reports the place unknown"))
        self._cancel.check(what)
        return [
            ObjectSeen(object_id, label, category_of(label), place) for label, object_id in seen
        ]

    def people(self, query: str = "", *, timeout_s: float = 5.0) -> list[PersonSeen]:
        what = "looking for people"
        begin_lookup(self._cancel, query, timeout_s, what)
        deadline = time.monotonic() + timeout_s
        message = self._fresh_array(min(self._wait_s, timeout_s), what)
        named = self._names(message, deadline)
        seen = self._people_in(message, query.strip().lower(), named)
        # Before the empty answer too, so a stop during the classification is not
        # reported as nobody in view.
        self._cancel.check(what)
        if not seen:
            return []
        pose = self._pose(deadline, "find_people reports no room and no bearings")
        room = self._room(pose)
        camera = None
        if pose is not None:
            wait_s = min(self._tf_wait_s, max(deadline - time.monotonic(), 0.0))
            camera = self._camera_transform(message, wait_s)
        self._cancel.check(what)
        k = list(message.camera_info.k)
        return [
            replace(
                person, room=room, bearing_deg=_bearing_deg(pixels, ankles, k, camera, pose)
            )
            for person, pixels, ankles in seen
        ]

    def close(self) -> None:
        """Destroy the subscriptions and drop the samples. Later and waiting calls get UNAVAILABLE."""
        with self._arrived:
            if self._closed:
                return
            self._closed = True
            self._newest = None
            self._newest_objects = None
            self._vectors.clear()
            self._arrived.notify_all()
        for subscription, kind in (
            (self._subscription, "people"),
            (self._objects_subscription, "object"),
        ):
            try:
                self._node.destroy_subscription(subscription)
            except Exception:
                logger.exception("Could not destroy the %s subscription.", kind)

    # -- The VectorSource of TrackMemory ------------------------------------------

    def collected(self, person_id: str) -> Collected | None:
        """The samples of the track `person_id` names, oldest first, as TrackMemory reads them.

        None for an id that names no track, such as person_untracked_1, and after
        close(). A track with no sample in the buffer's TTL gives an empty Collected.
        """
        matched = _TRACKED_ID.fullmatch(person_id) if isinstance(person_id, str) else None
        if matched is None:
            return None
        with self._lock:
            if self._closed:
                return None
            return self._vectors.collected(int(matched.group(1)), time.monotonic())

    def tracked_in_view(self) -> list[str]:
        """The ids of the tracked people of the newest array, left to right, however old it is."""
        with self._lock:
            newest = self._newest
        if newest is None:
            return []
        return _tracked_ids(newest[1])

    # -- Reads, on the agent's worker -------------------------------------------

    def _fresh_array(self, wait_s: float, what: str, *, objects: bool = False) -> Any:
        """The newest people array, or object array with `objects`, once it is fresh enough."""
        kind, topic = ("object", self._objects_topic) if objects else ("people", self._topic)
        deadline = time.monotonic() + wait_s
        with self._arrived:
            while True:
                # On every pass, so a stop ends the wait within _POLL_S.
                self._cancel.check(what)
                if self._closed:
                    raise StepError(
                        ErrorCode.UNAVAILABLE, f"Live {kind} perception has shut down."
                    )
                newest = self._newest_objects if objects else self._newest
                age_s: float | None = None
                if newest is not None:
                    stamp_ns, message = newest
                    # Again on every pass: the clock moves on while the newest stays.
                    age_s = self._age_s(stamp_ns, what)
                    if abs(age_s) <= self._max_age_s:
                        return message
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StepError(
                        ErrorCode.UNAVAILABLE,
                        stale_message(topic, age_s, self._max_age_s, wait_s),
                    )
                self._arrived.wait(min(_POLL_S, remaining))

    def _age_s(self, stamp_ns: int, what: str) -> float:
        try:
            now_ns = self._node.get_clock().now().nanoseconds
        except Exception as error:
            raise ros_failure(error, what[:1].upper() + what[1:]) from error
        return (now_ns - stamp_ns) / _NS_PER_S

    def _names(self, message: PersonDetectionArray, deadline: float) -> Mapping[str, str]:
        """The namer's names for the tracked people of `message`; {} without a namer or anyone tracked.

        The namer waits at most name_wait_s, and never past `deadline`. A failure of
        the namer gives {} and a warning; a stop raises StepError CANCELLED.
        """
        namer = self._namer
        if namer is None:
            return {}
        person_ids = _tracked_ids(message)
        if not person_ids:
            return {}
        wait_s = min(self._name_wait_s, deadline - time.monotonic())
        if wait_s <= 0:
            logger.warning(
                "find_people shows no names: the call's time ran out before the people "
                "were named."
            )
            return {}
        try:
            return namer(person_ids, timeout_s=wait_s)
        except StepError as error:
            if error.code is ErrorCode.CANCELLED:
                raise
            # The people are still in view, so they are reported without names.
            logger.warning("find_people shows no names: %s", error)
            return {}

    def _people_in(
        self, message: PersonDetectionArray, wanted: str, named: Mapping[str, str]
    ) -> list[tuple[PersonSeen, list[tuple[float, float]], list[tuple[float, float]]]]:
        """The people of `message` that match `wanted`, left to right, with two lists of pixels.

        `named` gives the names by id. A person matches by one of their phrases or by
        their name, ignoring case and spaces at either end. The first list is the
        torso's keypoints, or the box's centre with fewer than two of them seen, and
        the second the ankles seen. The records have no room and no bearing yet.
        """
        seen = []
        for person, person_id in _left_to_right(message):
            names = classify_gestures(person.keypoints, person.keypoint_scores, self._gestures)
            name = named.get(person_id) or None
            if wanted and wanted not in phrases(names) and not _is_name(wanted, name):
                continue
            record = PersonSeen(person_id, name, pose_phrase(names), gesture_phrase(names), "")
            min_score = self._gestures.min_keypoint_score
            pixels = torso_pixels(person.keypoints, person.keypoint_scores, min_score)
            if not pixels:
                box = person.bbox.center.position
                pixels = [(box.x, box.y)]
            ankles = ankle_pixels(person.keypoints, person.keypoint_scores, min_score)
            seen.append((record, pixels, ankles))
        return seen

    def _pose(self, deadline: float, without: str) -> Pose2D | None:
        """The base's pose now, or None when it cannot be read by `deadline`.

        `without` starts the warning logged then, and says what the answer lacks. A
        stop still ends the call.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("%s: the call's time ran out before the pose was read.", without)
            return None
        try:
            return self._base.pose(timeout_s=remaining)
        except StepError as error:
            if error.code is ErrorCode.CANCELLED:
                raise
            # What is in view is still reported, only without what the pose gives.
            logger.warning("%s: %s", without, error)
            return None

    def _room(self, pose: Pose2D | None) -> str:
        """The room of the place nearest `pose`, or "" without a pose or a place."""
        nearest = self._nearest_place(pose)
        return "" if nearest is None else nearest.room

    def _place(self, pose: Pose2D | None) -> str:
        """The name of the place nearest `pose` within PLACE_TOLERANCE_M, or UNKNOWN_PLACE."""
        nearest = self._nearest_place(pose)
        if nearest is None or _distance(nearest, pose) > PLACE_TOLERANCE_M:
            return UNKNOWN_PLACE
        return nearest.name

    def _nearest_place(self, pose: Pose2D | None) -> Any:
        """The configured place nearest `pose`, or None without a pose or a place."""
        if pose is None:
            return None
        places = self._base.places().values()
        if not places:
            return None
        return min(places, key=lambda place: _distance(place, pose))

    def _camera_transform(self, message: PersonDetectionArray, wait_s: float) -> _Camera | None:
        """The map <- camera transform at the array's stamp; None without one."""
        frame = message.header.frame_id
        if not frame:
            logger.warning("find_people reports no bearings: the people array names no frame.")
            return None
        try:
            transform = self._tf.lookup_transform(
                MAP_FRAME,
                frame,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=wait_s),
            )
        except TransformException as error:
            # The people are still in view, so they are reported without a bearing.
            logger.warning("find_people reports no bearings: %s", error)
            return None
        except Exception as error:
            raise ros_failure(error, "Looking for people") from error
        rotation = transform.transform.rotation
        origin = transform.transform.translation
        return (
            (rotation.x, rotation.y, rotation.z, rotation.w),
            (origin.x, origin.y, origin.z),
        )

    # -- The callbacks, on the executor -------------------------------------------

    def _on_people(self, message: PersonDetectionArray) -> None:
        stamp_ns = message.header.stamp.sec * _NS_PER_S + message.header.stamp.nanosec
        # Converted before the lock is taken, so a reader never waits for the copies.
        sightings = self._sightings(message)
        with self._arrived:
            if self._closed:
                return
            newest = self._newest
            jumped = newest is not None and newest[0] - stamp_ns > self._epoch_jump_ns
            if jumped:
                # The clock went back, so the tracks may be another run's or another
                # walkie-vision's, with the same ids.
                self._vectors.clear()
            # Also from an array older than the newest: its faces are as good.
            self._vectors.add(stamp_ns, sightings, time.monotonic())
            # By stamp, so an array that comes late does not hide a newer one.
            if newest is None or stamp_ns >= newest[0] or jumped:
                self._newest = (stamp_ns, message)
                self._arrived.notify_all()

    def _on_objects(self, message: ObjectDetectionArray) -> None:
        stamp_ns = message.header.stamp.sec * _NS_PER_S + message.header.stamp.nanosec
        with self._arrived:
            if self._closed:
                return
            newest = self._newest_objects
            # By stamp, and from the start again after the clock went back, as for people.
            if (
                newest is None
                or stamp_ns >= newest[0]
                or newest[0] - stamp_ns > self._epoch_jump_ns
            ):
                self._newest_objects = (stamp_ns, message)
                self._arrived.notify_all()

    def _sightings(self, message: PersonDetectionArray) -> list[Sighting]:
        """The tracked people of `message` with the vectors to collect of each."""
        k = message.camera_info.k
        fx, fy = float(k[0]), float(k[4])
        config = self._vectors.config
        sightings = []
        for person in message.people:
            if not person.track_id:
                continue
            face = None
            box = person.face_bbox
            if face_trusted(person.face_det_score, box.size_x, box.size_y, fx, fy, config):
                face = model_vector(person.face_model, person.face_embedding)
            appearance = model_vector(person.appearance_model, person.appearance_embedding)
            sightings.append(Sighting(person.track_id, face, appearance))
        return sightings


def _left_to_right(message: PersonDetectionArray) -> list[tuple[Any, str]]:
    """The people of `message` with their ids, by the column of their box's centre."""
    order = sorted(
        range(len(message.people)),
        key=lambda i: (message.people[i].bbox.center.position.x, i),
    )
    people, untracked = [], 0
    for i in order:
        person = message.people[i]
        if person.track_id:
            person_id = f"person_{person.track_id}"
        else:
            # Numbered before the query filters, so a query does not renumber them.
            untracked += 1
            person_id = f"person_untracked_{untracked}"
        people.append((person, person_id))
    return people


def _objects_left_to_right(message: ObjectDetectionArray) -> list[tuple[str, str]]:
    """The labels of the objects of `message` with their ids, by the column of their box's centre."""
    order = sorted(
        range(len(message.detections)),
        key=lambda i: (message.detections[i].bbox.center.position.x, i),
    )
    objects, untracked = [], 0
    for i in order:
        detection = message.detections[i]
        label = detection.label.strip()
        stem = label.replace(" ", "_")
        if detection.track_id:
            object_id = f"{stem}_{detection.track_id}"
        else:
            # Numbered before the query filters, so a query does not renumber them.
            untracked += 1
            object_id = f"{stem}_untracked_{untracked}"
        objects.append((label, object_id))
    return objects


def _distance(place: Any, pose: Pose2D) -> float:
    return math.hypot(place.x - pose.x, place.y - pose.y)


def not_looked_for(query: str, classes: Sequence[str]) -> str:
    """Why objects() cannot answer `query`: the detector looks only for `classes`."""
    return (
        f"The camera is not looking for {query.strip()!r} right now. It looks only for: "
        f"{', '.join(classes)}."
    )


def _tracked_ids(message: PersonDetectionArray) -> list[str]:
    """The ids of the tracked people of `message`, left to right, each once."""
    tracked = (person_id for person, person_id in _left_to_right(message) if person.track_id)
    return list(dict.fromkeys(tracked))


def _is_name(wanted: str, name: str | None) -> bool:
    """True when `name` is `wanted`, a query already stripped and in lower case."""
    return name is not None and name.strip().lower() == wanted


def _bearing_deg(
    pixels: list[tuple[float, float]],
    ankles: list[tuple[float, float]],
    k: list[float],
    camera: _Camera | None,
    pose: Pose2D | None,
) -> float | None:
    """The person's direction from the base's centre and heading, in degrees, or None.

    The direction of the rays through `pixels`, at the distance of the floor under
    `ankles`; the rays' own direction from the camera without that distance.
    """
    if camera is None or pose is None:
        return None
    rotation, origin = camera
    yaw = mean_yaw(pixels, k, rotation)
    if yaw is None:
        return None
    distance = floor_range(ankles, k, rotation, origin) if ankles else None
    if distance is not None:
        centred = yaw_from((pose.x, pose.y), origin, yaw, distance)
        if centred is not None:
            yaw = centred
    # + 0.0 turns -0.0 into 0.0, so a person straight ahead never reads as "-0".
    return round(wrap_deg(math.degrees(yaw - pose.yaw)), 1) + 0.0


def stale_message(topic: str, age_s: float | None, max_age_s: float, wait_s: float) -> str:
    """Why no array on `topic` was fresh enough: none within `wait_s`, or the newest's age."""
    if age_s is None:
        return f"No message on {topic} within {wait_s:g} s. {PUBLISHER_HINT}"
    if age_s < 0:
        # Far ahead of the node's clock: the publisher and the agent disagree on
        # use_sim_time, so every array would look stale or new.
        return (
            f"The newest message on {topic} is stamped {-age_s:.1f} s ahead of this "
            "node's clock, so the two use different clocks. Check that both or "
            "neither use sim time."
        )
    return (
        f"The newest message on {topic} is {age_s:.1f} s old, older than "
        f"{max_age_s:g} s. {PUBLISHER_HINT}"
    )
