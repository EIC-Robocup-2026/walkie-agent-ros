"""ctx.vision on the robot: UnavailableVision, or RosVision over walkie-vision's people.

UnavailableVision raises UNAVAILABLE from both verbs. An empty list would say that nothing
is in view, and a remembered database row is not a current view, so neither stands in
for a backend. main uses it unless the agent is started with --vision.

RosVision reads people from the newest PersonDetectionArray on /walkie_vision/people,
which walkie-vision publishes on the robot and oracle_perception in the simulation. Its
objects() stays UNAVAILABLE, since nothing reads the object arrays yet.

- An array counts only while its header stamp is within MAX_AGE_S of the node's clock,
  so the age is right under sim time too. people() waits up to WAIT_S for one, then
  raises UNAVAILABLE with the reason: no array, an old one, or a clock mismatch.
- One subscription, made in the constructor and destroyed by close(). Its callback runs
  on the connector's executor and only keeps the newest array by stamp. The reader
  converts it, on the agent's worker.
- A person's id is person_<track_id>, the tracker's id, which walkie-vision keeps while
  the track lasts. A person the tracker has not taken up has track_id 0 and gets
  person_untracked_<n>, numbered left to right in that one array.
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
vision_feed module and the people half of its RosVision). The feed there also collected
face and appearance vectors per track, which identity needs and this does not.
"""

import logging
import math
import threading
import time
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import TransformException
from walkie_vision_msgs.msg import PersonDetectionArray

from walkie_agent.capabilities import Base, ObjectSeen, PersonSeen, Pose2D
from walkie_agent.capabilities.guards import CancelToken, require_text, require_timeout
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.perception import PEOPLE_TOPIC
from walkie_agent.perception.geometry import (
    ankle_pixels,
    floor_range,
    mean_yaw,
    torso_pixels,
    wrap_deg,
    yaw_from,
)
from walkie_agent.perception.gestures import DEFAULT_GESTURE_CONFIG, GestureConfig, classify_gestures
from walkie_agent.perception.people import gesture_phrase, phrases, pose_phrase
from walkie_agent.ros.base import MAP_FRAME
from walkie_agent.ros.policy import FORBIDDEN_NAMES, check_endpoint, ros_failure

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

logger = logging.getLogger(__name__)

# BEST_EFFORT matches a reliable and a best-effort publisher alike, where a reliable
# subscription would miss a best-effort one, and walkie-vision fixes no QoS. VOLATILE
# and a short depth, because only the newest image matters.
PEOPLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
# How old an array may be and still count as what is in view (ADR-0050).
MAX_AGE_S = 1.0
# How long people() waits for a fresh array before it reports UNAVAILABLE.
WAIT_S = 1.0
# An array stamped this much before the newest one means the clock went back, as a
# simulation restart does. It replaces the newest, which would otherwise hide the new
# run's arrays until their stamps passed it.
EPOCH_JUMP_S = 5.0
# How long people() waits for TF to reach an array's stamp before it gives no bearings.
TF_WAIT_S = 0.2
PUBLISHER_HINT = "Is walkie-vision or oracle_perception running?"

_POLL_S = 0.02
_NS_PER_S = 1_000_000_000

# The map <- camera transform at an array's stamp: the rotation as x, y, z, w, and the
# optical centre's position as x, y, z.
_Camera = tuple[tuple[float, float, float, float], tuple[float, float, float]]


class UnavailableVision:
    """Checks a verb's arguments and the latch like a real backend, then raises UNAVAILABLE."""

    def __init__(self, cancel: CancelToken) -> None:
        self._cancel = cancel

    def objects(self, query: str = "", *, timeout_s: float = 5.0) -> list[ObjectSeen]:
        return _no_objects(self._cancel, query, timeout_s)

    def people(self, query: str = "", *, timeout_s: float = 5.0) -> list[PersonSeen]:
        _begin(self._cancel, query, timeout_s, "looking for people")
        raise StepError(ErrorCode.UNAVAILABLE, "Live people perception is not connected.")


class RosVision:
    """people() from walkie-vision's newest people array. objects() stays UNAVAILABLE.

    `base` gives the pose and the places that name the room. `tf_buffer` is a tf2 Buffer
    and defaults to the connector's own. Thread-safe: verbs run on LangChain workers,
    the callback on the executor and close() on any thread.
    """

    def __init__(
        self,
        connector: "ROS2Connector",
        base: Base,
        cancel: CancelToken,
        *,
        topic: str = PEOPLE_TOPIC,
        forbidden: Iterable[str] = FORBIDDEN_NAMES,
        max_age_s: float = MAX_AGE_S,
        wait_s: float = WAIT_S,
        epoch_jump_s: float = EPOCH_JUMP_S,
        tf_wait_s: float = TF_WAIT_S,
        gestures: GestureConfig = DEFAULT_GESTURE_CONFIG,
        tf_buffer: Any = None,
    ) -> None:
        self._cancel = cancel
        self._node = connector.node
        self._base = base
        self._topic = check_endpoint(self._node, topic, forbidden)
        # RAI's get_transform reads only the latest transform, and a bearing needs the
        # one at the image's stamp. So this reads the Buffer that the connector's own
        # listener fills, a private attribute of the pinned RAI: a RAI that renames it
        # fails here, at start, and a second listener would subscribe to TF twice.
        self._tf = connector._tf_buffer if tf_buffer is None else tf_buffer
        self._tf_wait_s = tf_wait_s
        self._max_age_s = max_age_s
        self._wait_s = wait_s
        self._epoch_jump_ns = round(epoch_jump_s * _NS_PER_S)
        self._gestures = gestures
        self._lock = threading.Lock()
        # Notified on every kept array and on close(), so a wait ends at once.
        self._arrived = threading.Condition(self._lock)
        self._newest: tuple[int, PersonDetectionArray] | None = None
        self._closed = False
        self._subscription = self._node.create_subscription(
            PersonDetectionArray, self._topic, self._on_people, PEOPLE_QOS
        )

    def objects(self, query: str = "", *, timeout_s: float = 5.0) -> list[ObjectSeen]:
        return _no_objects(self._cancel, query, timeout_s)

    def people(self, query: str = "", *, timeout_s: float = 5.0) -> list[PersonSeen]:
        what = "looking for people"
        _begin(self._cancel, query, timeout_s, what)
        deadline = time.monotonic() + timeout_s
        message = self._fresh_array(min(self._wait_s, timeout_s), what)
        seen = self._people_in(message, query.strip().lower())
        # Before the empty answer too, so a stop during the classification is not
        # reported as nobody in view.
        self._cancel.check(what)
        if not seen:
            return []
        pose = self._pose(deadline)
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
        """Destroy the subscription. Later and waiting calls get UNAVAILABLE."""
        with self._arrived:
            if self._closed:
                return
            self._closed = True
            self._newest = None
            self._arrived.notify_all()
        try:
            self._node.destroy_subscription(self._subscription)
        except Exception:
            logger.exception("Could not destroy the people subscription.")

    # -- Reads, on the agent's worker -------------------------------------------

    def _fresh_array(self, wait_s: float, what: str) -> PersonDetectionArray:
        deadline = time.monotonic() + wait_s
        with self._arrived:
            while True:
                # On every pass, so a stop ends the wait within _POLL_S.
                self._cancel.check(what)
                if self._closed:
                    raise StepError(
                        ErrorCode.UNAVAILABLE, "Live people perception has shut down."
                    )
                age_s: float | None = None
                if self._newest is not None:
                    stamp_ns, message = self._newest
                    # Again on every pass: the clock moves on while the newest stays.
                    age_s = self._age_s(stamp_ns)
                    if abs(age_s) <= self._max_age_s:
                        return message
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StepError(
                        ErrorCode.UNAVAILABLE,
                        stale_message(self._topic, age_s, self._max_age_s, wait_s),
                    )
                self._arrived.wait(min(_POLL_S, remaining))

    def _age_s(self, stamp_ns: int) -> float:
        try:
            now_ns = self._node.get_clock().now().nanoseconds
        except Exception as error:
            raise ros_failure(error, "Looking for people") from error
        return (now_ns - stamp_ns) / _NS_PER_S

    def _people_in(
        self, message: PersonDetectionArray, wanted: str
    ) -> list[tuple[PersonSeen, list[tuple[float, float]], list[tuple[float, float]]]]:
        """The people of `message` that match `wanted`, left to right, with two lists of pixels.

        The first is the torso's keypoints, or the box's centre with fewer than two of
        them seen, and the second the ankles seen. The records have no room and no
        bearing yet.
        """
        order = sorted(
            range(len(message.people)),
            key=lambda i: (message.people[i].bbox.center.position.x, i),
        )
        seen, untracked = [], 0
        for i in order:
            person = message.people[i]
            if person.track_id:
                person_id = f"person_{person.track_id}"
            else:
                # Numbered before the query filters, so a query does not renumber them.
                untracked += 1
                person_id = f"person_untracked_{untracked}"
            names = classify_gestures(person.keypoints, person.keypoint_scores, self._gestures)
            if wanted and wanted not in phrases(names):
                continue
            record = PersonSeen(person_id, None, pose_phrase(names), gesture_phrase(names), "")
            min_score = self._gestures.min_keypoint_score
            pixels = torso_pixels(person.keypoints, person.keypoint_scores, min_score)
            if not pixels:
                box = person.bbox.center.position
                pixels = [(box.x, box.y)]
            ankles = ankle_pixels(person.keypoints, person.keypoint_scores, min_score)
            seen.append((record, pixels, ankles))
        return seen

    def _pose(self, deadline: float) -> Pose2D | None:
        """The base's pose now, or None when it cannot be read by `deadline`.

        A stop still ends the call.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "find_people reports no room and no bearings: the call's time ran out "
                "before the pose was read."
            )
            return None
        try:
            return self._base.pose(timeout_s=remaining)
        except StepError as error:
            if error.code is ErrorCode.CANCELLED:
                raise
            # The people are still in view, so they are reported without a room or a bearing.
            logger.warning("find_people reports no room and no bearings: %s", error)
            return None

    def _room(self, pose: Pose2D | None) -> str:
        """The room of the place nearest `pose`, or "" without a pose or a place."""
        if pose is None:
            return ""
        places = self._base.places().values()
        if not places:
            return ""
        nearest = min(places, key=lambda place: math.hypot(place.x - pose.x, place.y - pose.y))
        return nearest.room

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

    # -- The callback, on the executor --------------------------------------------

    def _on_people(self, message: PersonDetectionArray) -> None:
        stamp_ns = message.header.stamp.sec * _NS_PER_S + message.header.stamp.nanosec
        with self._arrived:
            if self._closed:
                return
            newest = self._newest
            # By stamp, so an array that comes late does not hide a newer one.
            if (
                newest is None
                or stamp_ns >= newest[0]
                or newest[0] - stamp_ns > self._epoch_jump_ns
            ):
                self._newest = (stamp_ns, message)
                self._arrived.notify_all()


def _begin(cancel: CancelToken, query: str, timeout_s: float, what: str) -> None:
    require_timeout(timeout_s)
    require_text(query=query)
    cancel.check(what)


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


def _no_objects(cancel: CancelToken, query: str, timeout_s: float) -> list[ObjectSeen]:
    _begin(cancel, query, timeout_s, "looking for objects")
    raise StepError(ErrorCode.UNAVAILABLE, "Live object perception is not connected.")


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
