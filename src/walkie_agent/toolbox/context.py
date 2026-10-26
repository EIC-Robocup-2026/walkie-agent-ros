"""ToolContext: what every atomic's run(ctx, args) receives.

Atomics reach ROS, their backends and the handle store only through the ctx,
so the guard rails live in one place: the forbidden names, one rclpy entity
per name, bounded waits that also end on a stop or a reset, and backends
that tests can replace with fakes.
"""

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from rai.communication.ros2 import ROS2Connector
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription
from rclpy.task import Future
from rclpy.time import Time
from tf2_ros import Buffer, TransformException
from tf_transformations import euler_from_quaternion

from walkie_agent.toolbox.handles import HandleStore, Pose2D
from walkie_agent.toolbox.types import StepError
from walkie_agent.tools import FORBIDDEN_NAMES
from walkie_agent.tools.navigation import MAP_FRAME, ROBOT_FRAME, tf_publisher_hint

# How often wait() looks at the cancel event while the future is pending.
_CANCEL_POLL_S = 0.05


class ToolContext:
    """The connector, backends, handles and cancel token shared by the atomics.

    `backends` maps a name, such as "arm", to the client an atomic calls. Main
    passes the real clients and tests pass fakes; an atomic whose
    META.requires names a missing backend is not loaded. Building a ctx does
    not touch ROS, so a test without a node may pass a stand-in connector.

    `cancel` is set by a stop or a reset (Toolbox.interrupt) and cleared when
    the next task starts. A long wait goes through wait(), which ends with
    CANCELLED once it is set.
    """

    def __init__(
        self,
        connector: ROS2Connector,
        backends: Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.connector = connector
        self.logger = logger or logging.getLogger("walkie_agent.toolbox")
        self.cancel = threading.Event()
        self.handles = HandleStore(pose_provider=self._pose_or_none)
        self._backends = dict(backends or {})
        self._entities_lock = threading.Lock()
        # By (entity kind, name): the entity and the arguments it was made with.
        self._entities: dict[tuple[str, str], tuple[Any, tuple[Any, ...]]] = {}

    @property
    def node(self) -> Node:
        return self.connector.node

    @property
    def tf_buffer(self) -> Buffer:
        """The node's TF buffer, the one robot_pose() reads, for lookups at a stamp.

        robot_pose() only needs the latest transform, which the connector's
        get_transform() gives; perception needs the camera's transform at an
        image's stamp, which only the buffer gives.
        """
        # RAI's ROS2Connector keeps its buffer private and has no accessor.
        # It is reused rather than a second Buffer and TransformListener made
        # here: each listener subscribes to /tf and /tf_static again and keeps
        # its own copy of the tree, so one per node is enough. Recheck the
        # attribute on every rai-core upgrade.
        return self.connector._tf_buffer

    # -- Backends --------------------------------------------------------------

    def backend(self, name: str) -> Any:
        """The backend called `name`. Raises StepError UNAVAILABLE when there is none."""
        try:
            return self._backends[name]
        except KeyError:
            raise StepError("UNAVAILABLE", f"The {name} backend is not available.") from None

    def has_backend(self, name: str) -> bool:
        return name in self._backends

    # -- Cancel and time -------------------------------------------------------

    def check_cancelled(self) -> None:
        """Raises StepError CANCELLED once a stop or a reset has set `cancel`."""
        if self.cancel.is_set():
            raise StepError("CANCELLED", "A stop or a reset interrupted the call.")

    def now(self) -> Time:
        """The node clock's time, which follows /clock under sim time."""
        return self.node.get_clock().now()

    def robot_pose(self, timeout_s: float = 1.0) -> Pose2D:
        """The base's latest pose from TF, map -> base_footprint.

        Raises StepError UNAVAILABLE when TF has no such transform within
        `timeout_s`, with a hint when no node publishes TF at all.
        """
        try:
            stamped = self.connector.get_transform(MAP_FRAME, ROBOT_FRAME, timeout_sec=timeout_s)
        except TransformException as e:
            raise StepError(
                "UNAVAILABLE",
                f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME}) is unknown. "
                f"{e}{tf_publisher_hint(self.node)}",
            ) from e
        t = stamped.transform
        q = t.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return Pose2D(x=t.translation.x, y=t.translation.y, yaw=yaw)

    def wait(self, future: Future, timeout_s: float, what: str = "The call") -> Any:
        """Wait for a future completed by the connector's executor; return its result.

        Raises StepError TIMEOUT, "<what> did not finish within <timeout_s> s.",
        or CANCELLED as soon as `cancel` is set. Either way the future is left
        as it is: the caller cancels its goal or removes its pending request.
        """
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError(f"timeout_s must be a finite number of seconds, not {timeout_s}.")
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        deadline = time.monotonic() + timeout_s
        # future.done() as well as the event: a future that belongs to an
        # executor only schedules its callbacks there, so the event is set on
        # that executor's next spin, or never if it has stopped spinning.
        while not (done.is_set() or future.done()):
            self.check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StepError("TIMEOUT", f"{what} did not finish within {timeout_s:g} s.")
            done.wait(min(_CANCEL_POLL_S, remaining))
        return future.result()

    # -- ROS entities ----------------------------------------------------------
    #
    # rclpy entities must not be created per call, so each is made once per
    # name and kept. Asking again with other arguments is a programming error.

    def subscription(
        self,
        msg_type: Any,
        topic: str,
        callback: Callable[[Any], None],
        qos: QoSProfile | int,
        *,
        allow_forbidden: tuple[str, ...] = (),
    ) -> Subscription:
        """The node's subscription to `topic`, created on the first call."""
        return self._entity(
            "subscription",
            topic,
            (msg_type, callback, qos),
            allow_forbidden,
            lambda: self.node.create_subscription(msg_type, topic, callback, qos),
        )

    def publisher(
        self,
        msg_type: Any,
        topic: str,
        qos: QoSProfile | int,
        *,
        allow_forbidden: tuple[str, ...] = (),
    ) -> Publisher:
        """The node's publisher on `topic`, created on the first call."""
        return self._entity(
            "publisher",
            topic,
            (msg_type, qos),
            allow_forbidden,
            lambda: self.node.create_publisher(msg_type, topic, qos),
        )

    def service_client(
        self, srv_type: Any, name: str, *, allow_forbidden: tuple[str, ...] = ()
    ) -> Client:
        """The node's client of service `name`, created on the first call."""
        return self._entity(
            "service_client",
            name,
            (srv_type,),
            allow_forbidden,
            lambda: self.node.create_client(srv_type, name),
        )

    def action_client(
        self, action_type: Any, name: str, *, allow_forbidden: tuple[str, ...] = ()
    ) -> ActionClient:
        """The node's client of action `name`, created on the first call."""
        return self._entity(
            "action_client",
            name,
            (action_type,),
            allow_forbidden,
            lambda: ActionClient(self.node, action_type, name),
        )

    def _entity(
        self,
        kind: str,
        name: str,
        signature: tuple[Any, ...],
        allow_forbidden: tuple[str, ...],
        create: Callable[[], Any],
    ) -> Any:
        # The check and the cache use the name as the node resolves it, so a
        # relative name, "~/", or a remap onto a forbidden name is caught too.
        # Action names resolve as topic names do.
        if kind == "service_client":
            name = self.node.resolve_service_name(name)
        else:
            name = self.node.resolve_topic_name(name)
        # Raised as ValueError, not StepError: a forbidden name is a bug in
        # reviewed code, and must fail its tests rather than reach the LLM.
        if name in FORBIDDEN_NAMES and name not in allow_forbidden:
            raise ValueError(
                f"{name} is in FORBIDDEN_NAMES; pass allow_forbidden=({name!r},) "
                "only where a reviewed decision allows it."
            )
        with self._entities_lock:
            cached = self._entities.get((kind, name))
            if cached is not None:
                entity, made_with = cached
                if made_with != signature:
                    raise ValueError(
                        f"The {kind} for {name} already exists with other arguments; "
                        "create it once, for example with a module-level callback."
                    )
                return entity
            entity = create()
            self._entities[(kind, name)] = (entity, signature)
            return entity

    def _pose_or_none(self) -> Pose2D | None:
        # The handle store's pose provider: an unknown pose is an answer there,
        # not an error.
        try:
            return self.robot_pose()
        except StepError:
            return None
