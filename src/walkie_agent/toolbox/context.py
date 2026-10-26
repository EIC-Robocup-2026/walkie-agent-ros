"""ToolContext: what every atomic's run(ctx, args) receives.

Atomics reach ROS, their backends and the handle store only through the ctx,
so the guard rails live in one place: the forbidden names, one rclpy entity
per name, bounded waits that also end on a stop or a reset, and backends
that tests can replace with fakes. The typed capabilities, ctx.pose,
ctx.arms, ctx.gaze, ctx.vision, ctx.base and ctx.speech, are built on them
(walkie_agent.capabilities, ADR-0052).
"""

import logging
import math
import threading
from collections.abc import Callable, Mapping
from typing import Any

from geometry_msgs.msg import TransformStamped
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

from walkie_agent.capabilities import (
    Arms,
    Base,
    Clock,
    Gaze,
    Pose,
    Speech,
    SystemClock,
    Vision,
)
from walkie_agent.toolbox.handles import HandleStore, Pose2D
from walkie_agent.toolbox.types import StepError
from walkie_agent.tools import FORBIDDEN_NAMES
from walkie_agent.tools.navigation import MAP_FRAME, ROBOT_FRAME, tf_publisher_hint

# How often wait(), sleep() and wait_until() look at the cancel event.
_CANCEL_POLL_S = 0.05

# The capabilities a ctx has, by their attribute names.
CAPABILITIES = ("base", "speech", "pose", "arms", "gaze", "vision")


class ToolContext:
    """The connector, backends, handles and cancel token shared by the atomics.

    `backends` maps a name, such as "arm", to the client an atomic calls. Main
    passes the real clients and tests pass fakes; an atomic whose
    META.requires names a missing backend is not loaded. Building a ctx does
    not touch ROS, so a test without a node may pass a stand-in connector.

    `cancel` is set by a stop or a reset (Toolbox.interrupt) and cleared when
    the next task starts. A long wait goes through wait(), sleep() or
    wait_until(), which end with CANCELLED once it is set.

    `capabilities` maps a capability's name, one of CAPABILITIES, to a
    factory that builds it from this ctx, or returns None when it is not
    available; a name it leaves out gets the default factory. Each factory
    runs on the first use of its capability. The defaults build ctx.pose on
    the TF buffer and ctx.arms, ctx.gaze and ctx.vision on the "arm", "gaze"
    and "vision" backends when they are present; ctx.base and ctx.speech
    have no default until PR B (capabilities.md). `clock` is the clock the
    waits and the handle ages use, a fake one in tests (walkie_agent.testing).
    `in_simulation` fixes the in_simulation property, which otherwise reads
    the node's use_sim_time parameter.
    """

    def __init__(
        self,
        connector: ROS2Connector,
        backends: Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
        *,
        capabilities: Mapping[str, Callable[["ToolContext"], Any]] | None = None,
        clock: Clock | None = None,
        in_simulation: bool | None = None,
    ) -> None:
        unknown = sorted(set(capabilities or {}) - set(CAPABILITIES))
        if unknown:
            raise ValueError(
                f"Unknown capabilities {unknown}; a ctx has {', '.join(CAPABILITIES)}."
            )
        self.connector = connector
        self.logger = logger or logging.getLogger("walkie_agent.toolbox")
        self.cancel = threading.Event()
        self.clock = clock or SystemClock()
        self.handles = HandleStore(pose_provider=self._pose_or_none, clock=self.clock.monotonic)
        self._backends = dict(backends or {})
        self._entities_lock = threading.Lock()
        # By (entity kind, name): the entity and the arguments it was made with.
        self._entities: dict[tuple[str, str], tuple[Any, tuple[Any, ...]]] = {}
        self._factories: dict[str, Callable[[ToolContext], Any]] = {
            "pose": _default_pose,
            "arms": _default_arms,
            "gaze": _default_gaze,
            "vision": _default_vision,
            **(capabilities or {}),
        }
        self._in_simulation = in_simulation
        # Reentrant: a factory may read another capability, which runs that
        # capability's factory on the thread that holds the lock.
        self._capabilities_lock = threading.RLock()
        # By name: what the factory built, None included, once it ran.
        self._capabilities: dict[str, Any] = {}
        # The names whose factories are running, to catch a factory that
        # reads its own capability, directly or through another.
        self._building: set[str] = set()

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

    # -- Capabilities ----------------------------------------------------------
    #
    # Each raises StepError UNAVAILABLE, "The <name> capability is not
    # available.", when its factory returns None.

    @property
    def base(self) -> Base:
        return self._capability("base")

    @property
    def speech(self) -> Speech:
        return self._capability("speech")

    @property
    def pose(self) -> Pose:
        return self._capability("pose")

    @property
    def arms(self) -> Arms:
        return self._capability("arms")

    @property
    def gaze(self) -> Gaze:
        return self._capability("gaze")

    @property
    def vision(self) -> Vision:
        return self._capability("vision")

    def _capability(self, name: str) -> Any:
        with self._capabilities_lock:
            if name not in self._capabilities:
                if name in self._building:
                    raise RuntimeError(f"The {name} capability's factory reads ctx.{name}.")
                factory = self._factories.get(name)
                self._building.add(name)
                try:
                    self._capabilities[name] = None if factory is None else factory(self)
                finally:
                    self._building.discard(name)
            capability = self._capabilities[name]
        if capability is None:
            raise StepError("UNAVAILABLE", f"The {name} capability is not available.")
        return capability

    # -- Cancel and time -------------------------------------------------------

    def check_cancelled(self) -> None:
        """Raises StepError CANCELLED once a stop or a reset has set `cancel`."""
        if self.cancel.is_set():
            raise StepError("CANCELLED", "A stop or a reset interrupted the call.")

    def now(self) -> Time:
        """The node clock's time, which follows /clock under sim time."""
        return self.node.get_clock().now()

    def now_ns(self) -> int:
        """now() in nanoseconds."""
        return self.now().nanoseconds

    @property
    def in_simulation(self) -> bool:
        """True on sim time, where the poses not yet tried on the robot are offered too.

        main.py's --use-sim-time sets the node's use_sim_time parameter
        through the connector, so the flag that puts the agent on the
        simulation's clock also decides which poses it offers, and nothing
        else needs wiring. Read on every access, unless the ctx was built
        with in_simulation.
        """
        if self._in_simulation is not None:
            return self._in_simulation
        return bool(self.node.get_parameter("use_sim_time").value)

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
        return _pose(stamped)

    def wait(self, future: Future, timeout_s: float, what: str = "The call") -> Any:
        """Wait for a future completed by the connector's executor; return its result.

        Raises StepError TIMEOUT, "<what> did not finish within <timeout_s> s.",
        or CANCELLED as soon as `cancel` is set. Either way the future is left
        as it is: the caller cancels its goal or removes its pending request.
        """
        _check_timeout(timeout_s)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        deadline = self.clock.monotonic() + timeout_s
        # future.done() as well as the event: a future that belongs to an
        # executor only schedules its callbacks there, so the event is set on
        # that executor's next spin, or never if it has stopped spinning.
        while not (done.is_set() or future.done()):
            self.check_cancelled()
            remaining = deadline - self.clock.monotonic()
            if remaining <= 0:
                raise StepError("TIMEOUT", f"{what} did not finish within {timeout_s:g} s.")
            self.clock.wait(done, min(_CANCEL_POLL_S, remaining))
        return future.result()

    def sleep(self, seconds: float) -> None:
        """Wait `seconds` on the ctx's clock; raise StepError CANCELLED once `cancel` is set."""
        _check_timeout(seconds, "seconds")
        deadline = self.clock.monotonic() + seconds
        while True:
            self.check_cancelled()
            remaining = deadline - self.clock.monotonic()
            if remaining <= 0:
                return
            self.clock.wait(self.cancel, min(_CANCEL_POLL_S, remaining))

    def wait_until(
        self, ready: Callable[[], bool], timeout_s: float, what: str = "The call"
    ) -> None:
        """Poll `ready` until it returns True, every _CANCEL_POLL_S on the ctx's clock.

        Raises StepError TIMEOUT, "<what> did not finish within <timeout_s> s.",
        or CANCELLED as soon as `cancel` is set. `ready` is called first,
        before any wait, and again at the deadline, so a condition that holds
        at once costs no wait.
        """
        _check_timeout(timeout_s)
        deadline = self.clock.monotonic() + timeout_s
        while True:
            if ready():
                return
            self.check_cancelled()
            remaining = deadline - self.clock.monotonic()
            if remaining <= 0:
                raise StepError("TIMEOUT", f"{what} did not finish within {timeout_s:g} s.")
            self.clock.wait(self.cancel, min(_CANCEL_POLL_S, remaining))

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


def _check_timeout(value: float, name: str = "timeout_s") -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number of seconds, not {value}.")


class RosPose:
    """ctx.pose on the node's TF buffer: map -> base_footprint, by a named staleness rule.

    Every read goes through ctx.tf_buffer and ctx.now(), and every wait
    through ctx.wait_until(), so it ends on a stop or a reset.
    """

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    def latest(self, *, max_age_s: float) -> Pose2D:
        """The newest transform, without a wait.

        Raises StepError UNAVAILABLE when TF has none, and STALE when it is
        more than `max_age_s` from the node clock. Stamp 0 is a chain of
        static transforms, as in a test, which never goes stale.
        """
        stamped = self._lookup()
        stamp_ns = Time.from_msg(stamped.header.stamp).nanoseconds
        if stamp_ns:
            # abs(): a clock that moved back, as at a simulation restart,
            # makes the transform as doubtful as an old one.
            age = abs(self._ctx.now().nanoseconds - stamp_ns) / 1e9
            if age > max_age_s:
                raise StepError(
                    "STALE",
                    f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME}) is {age:.1f} s old, "
                    f"more than {max_age_s:g} s.",
                    {"age_s": age},
                )
        return _pose(stamped)

    def fresh(self, *, timeout_s: float = 1.0) -> Pose2D:
        """A transform stamped no earlier than this call, as move_relative reads it.

        The newest transform can predate the end of the last goal: the goal
        result and the TF update arrive on different topics. Raises StepError
        UNAVAILABLE when none comes within `timeout_s`, and CANCELLED on a
        stop or a reset.
        """
        ctx = self._ctx
        called_at = ctx.now()
        found: list[TransformStamped] = []
        errors: list[TransformException] = []

        def ready() -> bool:
            try:
                stamped = ctx.tf_buffer.lookup_transform(MAP_FRAME, ROBOT_FRAME, Time())
            except TransformException as e:
                errors.append(e)
                return False
            found[:] = [stamped]
            return Time.from_msg(stamped.header.stamp) >= called_at

        try:
            ctx.wait_until(ready, timeout_s)
        except StepError as e:
            if e.code != "TIMEOUT":
                raise
            hint = tf_publisher_hint(ctx.node)
            if not found:
                raise StepError(
                    "UNAVAILABLE",
                    f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME}) is unknown. "
                    f"{errors[-1]}{hint}",
                ) from None
            raise StepError(
                "UNAVAILABLE",
                f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME}) is out of date: no "
                f"transform stamped after the call came within {timeout_s:g} s. TF is not "
                "updating, or the agent and the robot disagree on sim time (see "
                f"--use-sim-time).{hint}",
            ) from None
        return _pose(found[0])

    def at(self, stamp_ns: int, *, timeout_s: float) -> Pose2D:
        """The transform at `stamp_ns`, an image's stamp, as perception reads it.

        Waits up to `timeout_s` for TF to reach the stamp. Raises StepError
        UNAVAILABLE without a stamp or such a transform, and CANCELLED on a
        stop or a reset.
        """
        ctx = self._ctx
        if stamp_ns <= 0:
            # Stamp zero would ask TF for its latest transform, the pose now.
            raise StepError(
                "UNAVAILABLE",
                f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME}) at that stamp is "
                f"unknown: {stamp_ns} is not a stamp.",
            )
        stamp = Time(nanoseconds=stamp_ns)
        buffer = ctx.tf_buffer
        try:
            ctx.wait_until(lambda: buffer.can_transform(MAP_FRAME, ROBOT_FRAME, stamp), timeout_s)
        except StepError as e:
            if e.code != "TIMEOUT":
                raise
        # Also after a timeout: the lookup's error says what TF lacks.
        try:
            stamped = buffer.lookup_transform(MAP_FRAME, ROBOT_FRAME, stamp)
        except TransformException as e:
            raise StepError(
                "UNAVAILABLE",
                f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME}) at that stamp is "
                f"unknown. {e}{tf_publisher_hint(ctx.node)}",
            ) from e
        return _pose(stamped)

    def _lookup(self) -> TransformStamped:
        ctx = self._ctx
        try:
            return ctx.tf_buffer.lookup_transform(MAP_FRAME, ROBOT_FRAME, Time())
        except TransformException as e:
            raise StepError(
                "UNAVAILABLE",
                f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME}) is unknown. "
                f"{e}{tf_publisher_hint(ctx.node)}",
            ) from e


def _pose(stamped: TransformStamped) -> Pose2D:
    t = stamped.transform
    q = t.rotation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return Pose2D(x=t.translation.x, y=t.translation.y, yaw=yaw)


# The default capability factories. The ROS implementations are imported
# here, not at the top: they import walkie_agent.toolbox, which imports
# this module.


def _default_pose(ctx: ToolContext) -> Pose:
    return RosPose(ctx)


def _default_arms(ctx: ToolContext) -> Arms | None:
    from walkie_agent.tools.atomic import _arm

    return _arm.RosArms(ctx, ctx.backend(_arm.ARM)) if ctx.has_backend(_arm.ARM) else None


def _default_gaze(ctx: ToolContext) -> Gaze | None:
    from walkie_agent.tools.atomic import _gaze

    return _gaze.RosGaze(ctx, ctx.backend(_gaze.GAZE)) if ctx.has_backend(_gaze.GAZE) else None


def _default_vision(ctx: ToolContext) -> Vision | None:
    from walkie_agent.tools.atomic import _perception

    if not ctx.has_backend(_perception.VISION):
        return None
    return _perception.RosVision(ctx, ctx.backend(_perception.VISION))
