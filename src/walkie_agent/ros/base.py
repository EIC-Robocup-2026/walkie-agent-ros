"""RosBase: ctx.base over Nav2's NavigateToPose action and the connector's TF buffer.

The robot can move while no verb is waiting for it, so one record follows each goal
from its reservation until Nav2 reports an end. The record outlives the verb that sent
the goal, and it outlives a reset of the token.

- A verb that ends in CANCELLED or TIMEOUT marks its record abandoned for good. Whatever
  the token does later, an accepted or late-accepted goal then gets cancelled.
- A cancel acknowledgement ends nothing. Only a terminal result or a rejection does.
- While a record lives, a new motion is refused, so no goal overwrites a live one.
- A goal whose state cannot be known, such as one whose acceptance never arrives, keeps
  its record. It cannot prove the robot stopped, so it never unlocks new motion.

Threads and locks: verbs run on LangChain workers, stop() and close() on any thread, and
the future callbacks on the connector's executor, or inline on whichever thread attaches
one to a future that is already done. So:

- The token's commit fence comes before self._lock when a thread needs both.
- A callback reads the token's flag and never takes the fence.
- Nothing calls rclpy or attaches a callback while holding self._lock.
- The arm check runs holding neither, so no lock of the arm backend is taken inside them.
- The fence holds only send_goal_async, or the final check of a success. No wait runs
  inside it. This orders a send and a success against cancel(), so a stop either sees a
  stored future or stops the send, and no success is committed after cancel() returned.

The pose history behind pose_at (decision PK-79) is filled by a timer on the executor,
which reads the newest map <- base transform from the connector's TF buffer without
waiting, POSE_HISTORY_HZ times a second, and keeps POSE_HISTORY_S of samples. TF keeps
only seconds of transforms, and a sighting can be minutes old. self._history_lock guards
the history alone, so the timer never waits on a drive's lock, and it is a leaf: nothing
else is taken inside it, and the clock is read inside it so that a sample and a lookup
agree on which clock reading came first. A clock that moves backwards, as a sim reset
does, clears the history, since its stamps belong to the old clock.
"""

import bisect
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time
from tf2_ros import TransformException

from walkie_agent.capabilities.guards import (
    CancelToken,
    check_relative_move,
    is_finite_number,
    relative_goal,
    require_finite,
    require_text,
    require_timeout,
)
from walkie_agent.capabilities.records import Pose2D
from walkie_agent.errors import ErrorCode, GoalFailed, StepError
from walkie_agent.locations import Location
from walkie_agent.ros.policy import FORBIDDEN_NAMES, check_endpoint, ros_failure

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

logger = logging.getLogger(__name__)

NAV_ACTION = "/navigate_to_pose"
# Nav2's global frame and robot frame on walkie-ros. On its main branch base_link is
# rotated 90 degrees from base_footprint, so a heading read from it would be a quarter
# turn off.
MAP_FRAME = "map"
ROBOT_FRAME = "base_footprint"

# Sub-budgets of a verb's deadline. These are implementation defaults, not measured
# robot requirements.
DISCOVERY_S = 2.0
ACCEPTANCE_S = 2.0
TF_S = 1.0
# RAI's get_transform never checks the buffer at 0 and sleeps 0.1 s on a miss, so a
# stop during a TF read takes about that long to land.
_TF_LOOKUP_S = 0.01
# A sample stamped further ahead of the clock than this predates a clock rollback, or
# comes from a node on another clock. A little ahead is normal between two processes.
_MAX_FUTURE_STAMP_NS = 500_000_000
# The pose history behind pose_at (decision PK-79): how often it samples the pose, and
# how far back it keeps samples, in seconds of the node's clock. Half an hour covers a
# sighting from earlier in a task at 9000 samples; an older one has no pose, and the
# atomics fall back to the named place nearest the object. Implementation defaults, not
# measured robot requirements.
POSE_HISTORY_HZ = 5.0
POSE_HISTORY_S = 1800.0
# The furthest a time may lie from the nearest kept sample for pose_at to answer. A gap
# in the samples, as while TF was down, gives no pose further than this from its ends.
POSE_MATCH_S = 1.0
_NS_PER_S = 1_000_000_000

_TERMINAL = {GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED}
_STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}
_POSITION = f"The robot's position ({MAP_FRAME} -> {ROBOT_FRAME})"
_STOP_REASON = "a stop is in effect"


@dataclass(eq=False)
class _Operation:
    """One goal. self._lock guards its fields, and the events let a verb wait without it."""

    what: str
    send_future: Any = None
    handle: Any = None
    result_future: Any = None
    response: Any = None
    # Permanent once set: a reset of the token does not clear it.
    abandoned: str | None = None
    # Why the goal's state cannot be known. It does not make the record terminal.
    error: str | None = None
    rejected: bool = False
    terminal: bool = False
    returned: bool = False
    result_requested: bool = False
    cancel_sent: bool = False
    acceptance: threading.Event = field(default_factory=threading.Event)
    answered: threading.Event = field(default_factory=threading.Event)
    settled: threading.Event = field(default_factory=threading.Event)


class RosBase:
    """ctx.base on the robot, with one owned ActionClient on the connector's node.

    It never spins, never cancels another client's goals and never publishes a velocity.
    `client_factory` exists for the test doubles.

    `arms_check`, when given, is the arm check before every drive (ADR-0025), such as
    RosArms.check_for_driving. It is called as arms_check(what, budget_s) once no other
    goal is live, before discovery, the TF read and the send, and it returns to let the
    drive go on or raises StepError to refuse it. `budget_s` is the time left to the
    verb, and the check must end within it and on a stop. Its error ends the verb
    unchanged, so a refused drive reads no TF and sends nothing. Without it, the base
    drives with no arm check.

    A drive waits for the longer of its timeout_s and `min_drive_timeout_s`, so a
    simulation that runs slower than real time can raise the bound of every drive.

    It also owns one timer on the connector's node, the pose history's (see the
    module's docstring), which close() destroys with the client.
    """

    def __init__(
        self,
        connector: "ROS2Connector",
        locations: Mapping[str, Location],
        cancel: CancelToken,
        *,
        action_name: str = NAV_ACTION,
        forbidden: Iterable[str] = FORBIDDEN_NAMES,
        discovery_s: float = DISCOVERY_S,
        acceptance_s: float = ACCEPTANCE_S,
        tf_s: float = TF_S,
        client_factory: Callable[..., Any] = ActionClient,
        arms_check: Callable[[str, float], None] | None = None,
        min_drive_timeout_s: float = 0.0,
    ) -> None:
        number = isinstance(min_drive_timeout_s, int | float) and not isinstance(
            min_drive_timeout_s, bool
        )
        if not (number and math.isfinite(min_drive_timeout_s) and min_drive_timeout_s >= 0):
            raise ValueError(
                "min_drive_timeout_s must be zero or a positive number of seconds, not "
                f"{min_drive_timeout_s!r}."
            )
        self._min_drive_timeout_s = min_drive_timeout_s
        self._connector = connector
        self._node = connector.node
        self._cancel = cancel
        self._places = MappingProxyType(dict(locations))
        self._by_name = {name.strip().casefold(): place for name, place in locations.items()}
        if len(self._by_name) != len(self._places):
            raise ValueError("Two places have the same name once case and spaces are ignored.")
        self._action_name = check_endpoint(self._node, action_name, forbidden)
        self._discovery_s = discovery_s
        self._acceptance_s = acceptance_s
        self._tf_s = tf_s
        self._arms_check = arms_check
        self._lock = threading.Lock()
        self._operation: _Operation | None = None
        self._closing = False
        # Its own group, so the node's other callbacks cannot hold up a goal's answers.
        self._client = client_factory(
            self._node,
            NavigateToPose,
            self._action_name,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        # RAI's get_transform sleeps 0.1 s on a miss, which a timer on the executor must
        # not do, so the history reads the Buffer that the connector's own listener
        # fills, as RosVision does: a private attribute of the pinned RAI.
        self._tf = connector._tf_buffer
        self._history_lock = threading.Lock()
        # (stamp in seconds of the node's clock, pose), oldest first, stamps rising.
        self._history: deque[tuple[float, Pose2D]] = deque(
            maxlen=math.ceil(POSE_HISTORY_HZ * POSE_HISTORY_S) + 1
        )
        # The latest clock reading the history saw, to notice the clock moving backwards.
        self._history_clock_ns = 0
        self._sample_failed = False
        # Its own group, so a slow callback elsewhere on the node delays no sample.
        self._pose_timer = self._node.create_timer(
            1.0 / POSE_HISTORY_HZ,
            self._sample_pose,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    # -- ctx.base --------------------------------------------------------------

    def places(self) -> Mapping[str, Location]:
        self._cancel.check("listing places")
        return self._places

    def go_to(self, place: str, *, timeout_s: float = 120.0) -> Pose2D:
        require_text(place=place)
        require_timeout(timeout_s)
        self._cancel.check(f"the drive to {place.strip()}")
        location = self._by_name.get(place.strip().casefold())
        if location is None:
            known = ", ".join(self._places) or "none"
            raise StepError(ErrorCode.NOT_FOUND, f"Unknown place '{place}'. Known places: {known}.")
        target = Pose2D(location.x, location.y, location.yaw)
        return self._drive(f"the drive to {location.name}", timeout_s, lambda _deadline: target)

    def move_relative(
        self,
        forward_m: float = 0.0,
        left_m: float = 0.0,
        turn_deg: float = 0.0,
        *,
        timeout_s: float = 60.0,
    ) -> Pose2D:
        check_relative_move(forward_m, left_m, turn_deg)
        require_timeout(timeout_s)
        what = "the move"
        self._cancel.check(what)

        def target(deadline: float) -> Pose2D:
            start = self._fresh_pose(self._budget(deadline, self._tf_s, what), what)
            return relative_goal(start, forward_m, left_m, turn_deg)

        return self._drive(what, timeout_s, target)

    def stop(self) -> None:
        """Set the latch and abandon the current goal. Never raises.

        It sends a cancel at most once per goal, and a goal that is not accepted yet
        gets it on acceptance. The acknowledgement is not proof the wheels stopped.
        """
        try:
            self._cancel.cancel()
            with self._lock:
                operation = self._operation
            if operation is not None:
                self._abandon(operation, _STOP_REASON)
        except Exception:
            logger.exception("base.stop failed.")

    def pose(self, *, timeout_s: float = TF_S) -> Pose2D:
        """The pose from a sample stamped after this call, waited for at most `timeout_s`.

        The wait is also no longer than the base's own `tf_s`. The last TF read can
        begin as the time runs out, and a read is not cut short. While TF has no
        transform between the two frames each read takes RAI's 0.1 s sleep, so the
        call can end about 0.1 s past `timeout_s`, plus scheduling delay.
        """
        require_timeout(timeout_s)
        what = "reading the pose"
        self._cancel.check(what)
        with self._lock:
            self._require_open()
        return self._fresh_pose(min(self._tf_s, timeout_s), what)

    def pose_at(self, stamp_s: float) -> Pose2D | None:
        """The pose at `stamp_s`, from the two kept samples around it, or None.

        The position is interpolated between the two samples, and the yaw turns
        the shorter way between theirs. A time past either end of the history, up
        to POSE_MATCH_S, takes that end's pose, since the newest sample can lag a
        sighting by a timer period; the pose is not extrapolated. Two samples
        further apart than POSE_MATCH_S are a gap, and each side of it is held
        the same way, never interpolated across. None when the nearest sample is
        further than POSE_MATCH_S from `stamp_s`, which covers a time outside the
        history and one inside a gap in it. A clock that moved
        backwards since the history last looked clears it first. It raises
        REFUSED for a `stamp_s` that is not finite, CANCELLED on a stop, and
        UNAVAILABLE after close().
        """
        require_finite(stamp_s=stamp_s)
        self._cancel.check("looking up where Walkie stood")
        with self._lock:
            self._require_open()
        with self._history_lock:
            self._note_clock(self._clock_ns())
            return _interpolate(self._history, float(stamp_s))

    def go_to_pose(self, pose: Pose2D, *, timeout_s: float = 120.0) -> Pose2D:
        """Drive to `pose` in the map through the same drive as go_to, with its checks and errors.

        For atomics only, such as one that drives back to where Walkie stood when
        it saw an object; no LLM tool takes map coordinates (ADR-0010), and no
        message names them. It refuses a pose that is not finite with REFUSED.
        """
        require_finite(x=pose.x, y=pose.y, yaw=pose.yaw)
        require_timeout(timeout_s)
        what = "the drive"
        self._cancel.check(what)
        target = Pose2D(float(pose.x), float(pose.y), float(pose.yaw))
        return self._drive(what, timeout_s, lambda _deadline: target)

    # -- Lifecycle -------------------------------------------------------------

    def close(self, timeout_s: float = 5.0) -> bool:
        """Stop, wait up to `timeout_s` for the goal to end, then destroy the client and the timer.

        Returns False, with the client and the pose history's timer alive, if the
        goal has not ended. The executor has to keep spinning for its answers to
        arrive, so call this before the connector's shutdown, and again to retry.
        unresolved() says what is left. pose_at refuses from the first call on.
        """
        with self._lock:
            self._closing = True
        self.stop()
        with self._lock:
            operation = self._operation
        if operation is not None:
            operation.settled.wait(timeout_s)
        with self._lock:
            if self._operation is not None:
                return False
            client, self._client = self._client, None
            timer, self._pose_timer = self._pose_timer, None
        if client is not None:
            try:
                client.destroy()
            except Exception:
                logger.exception("Could not destroy the Nav2 action client.")
        if timer is not None:
            try:
                self._node.destroy_timer(timer)
            except Exception:
                logger.exception("Could not destroy the pose history's timer.")
        return True

    def unresolved(self) -> list[str]:
        """What keeps close() from finishing, as sentences for the operator's log."""
        with self._lock:
            operation = self._operation
            if operation is None:
                return []
            if operation.handle is None:
                state = "Nav2 has not answered the goal"
            elif operation.cancel_sent:
                state = "a cancel was requested, but no terminal result has arrived"
            else:
                state = "no terminal result has arrived"
            if operation.error is not None:
                state = f"{state}, and {operation.error}"
            return [f"{operation.what}: {state}."]

    # -- One drive -------------------------------------------------------------

    def _drive(self, what: str, timeout_s: float, target: Callable[[float], Pose2D]) -> Pose2D:
        deadline = time.monotonic() + max(timeout_s, self._min_drive_timeout_s)
        with self._lock:
            self._require_free()
        if self._arms_check is not None:
            # Outside the lock and the fence, since it waits on the commander.
            self._arms_check(what, self._budget(deadline, math.inf, what))
        self._discover(deadline, what)
        goal = self._goal(target(deadline))
        operation = self._send(goal, what)
        try:
            # It may run inline, so this comes after the fence and without the lock.
            operation.send_future.add_done_callback(partial(self._on_goal_response, operation))
            self._await_answer(operation, deadline, what)
            self._check_outcome(operation, what)
            pose = self._pose_after_success(deadline, what)
            with self._cancel.unless_cancelled(what):
                with self._lock:
                    if operation.abandoned is not None:
                        raise StepError(
                            ErrorCode.CANCELLED, f"Cancelled {what}: {operation.abandoned}."
                        )
                return pose
        except BaseException as error:
            self._abandon(operation, str(error))
            raise
        finally:
            with self._lock:
                operation.returned = True
                self._settle(operation)

    def _require_free(self) -> None:
        """Call with the lock held: refuse a motion while the record of another goal lives."""
        self._require_open()
        if self._operation is not None:
            raise StepError(
                ErrorCode.PRECONDITION_FAILED,
                f"Refused: {self._operation.what} has not ended yet, so Walkie may still be moving.",
            )

    def _require_open(self) -> None:
        if self._closing:
            raise StepError(ErrorCode.UNAVAILABLE, "Navigation has shut down.")

    @staticmethod
    def _budget(deadline: float, limit_s: float, what: str) -> float:
        """A wait's budget. CancelToken.wait_for reads zero or less as REFUSED, so it never gets one."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StepError(ErrorCode.TIMEOUT, f"Timed out: {what} ran out of time.")
        return min(limit_s, remaining)

    def _discover(self, deadline: float, what: str) -> None:
        """Poll for the server, since a blocking wait would hold up a stop for its whole length."""
        budget_s = self._budget(deadline, self._discovery_s, what)
        try:
            self._cancel.wait_for(self._server_ready, budget_s, what)
        except StepError as error:
            if error.code is not ErrorCode.TIMEOUT:
                raise
            raise StepError(
                ErrorCode.UNAVAILABLE,
                f"Navigation is unavailable: action server '{self._action_name}' "
                "did not respond. Is Nav2 running?",
            ) from error

    def _server_ready(self) -> bool:
        with self._lock:
            self._require_open()
            client = self._client
        try:
            return client.server_is_ready()
        except Exception as error:
            raise ros_failure(error, "Finding Nav2") from error

    def _goal(self, target: Pose2D) -> Any:
        require_finite(x=target.x, y=target.y, yaw=target.yaw)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = MAP_FRAME
        try:
            goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        except Exception as error:
            raise ros_failure(error, "Reading the clock") from error
        goal.pose.pose.position.x = float(target.x)
        goal.pose.pose.position.y = float(target.y)
        goal.pose.pose.orientation.z = math.sin(target.yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(target.yaw / 2.0)
        return goal

    def _send(self, goal: Any, what: str) -> _Operation:
        """Reserve the record and send, inside the fence, with the future stored before it ends."""
        with self._cancel.unless_cancelled(what):
            with self._lock:
                self._require_free()
                operation = self._operation = _Operation(what)
                client = self._client
            try:
                operation.send_future = client.send_goal_async(goal)
            except Exception as error:
                # Nothing went out, so nothing can be moving. This is the one error that ends a record.
                with self._lock:
                    operation.terminal = operation.returned = True
                    self._settle(operation)
                raise ros_failure(error, "Sending the goal") from error
        return operation

    def _await_answer(self, operation: _Operation, deadline: float, what: str) -> None:
        self._wait(
            operation,
            operation.acceptance,
            self._budget(deadline, self._acceptance_s, what),
            f"Nav2's acceptance of {what}",
        )
        self._wait(operation, operation.answered, self._budget(deadline, math.inf, what), what)

    def _wait(self, operation: _Operation, event: threading.Event, budget_s: float, what: str) -> None:
        """Wait for `event`. A stop followed at once by a reset still ends the wait."""
        self._cancel.wait_for(
            lambda: event.is_set() or operation.abandoned is not None, budget_s, what
        )
        if operation.abandoned is not None:
            raise StepError(ErrorCode.CANCELLED, f"Cancelled {what}: {operation.abandoned}.")

    @staticmethod
    def _check_outcome(operation: _Operation, what: str) -> None:
        """Raise unless Nav2 reports success with error code zero.

        A goal Nav2 rejected or ended raises GoalFailed. A goal whose state is unknown
        raises a plain StepError, since the robot may still be moving under it.
        """
        if operation.error is not None:
            raise StepError(ErrorCode.SKILL_FAILED, f"Navigation failed for {what}: {operation.error}.")
        if operation.rejected:
            raise GoalFailed(ErrorCode.SKILL_FAILED, f"Nav2 rejected the goal for {what}.")
        status = operation.response.status
        result = operation.response.result
        if status == GoalStatus.STATUS_CANCELED:
            raise StepError(ErrorCode.CANCELLED, f"Nav2 cancelled {what}.")
        if status != GoalStatus.STATUS_SUCCEEDED or result.error_code != 0:
            detail = f" {result.error_msg}" if result.error_msg else ""
            raise GoalFailed(
                ErrorCode.SKILL_FAILED,
                f"Navigation failed for {what} ({_STATUS_NAMES[status]}, "
                f"error code {result.error_code}).{detail}",
            )

    def _pose_after_success(self, deadline: float, what: str) -> Pose2D:
        try:
            return self._fresh_pose(self._budget(deadline, self._tf_s, what), what)
        except StepError as error:
            if error.code is ErrorCode.CANCELLED:
                raise
            logger.warning("Nav2 finished %s, but the pose is unknown: %s", what, error)
            raise StepError(
                error.code, f"Nav2 finished {what}, so Walkie may have moved. {error.message}"
            ) from error

    # -- The record, and the callbacks that advance it -----------------------------

    def _settle(self, operation: _Operation) -> None:
        """Call with the lock held: free the gate once Nav2 is done and the verb has left."""
        if operation.terminal and operation.returned:
            if self._operation is operation:
                self._operation = None
            operation.settled.set()

    def _abandon(self, operation: _Operation, reason: str) -> None:
        with self._lock:
            if operation.abandoned is None:
                operation.abandoned = reason
        self._advance(operation)

    def _advance(self, operation: _Operation) -> None:
        """Ask for what the record still lacks: the result, and a cancel if it is abandoned.

        Each is sent once per handle. The flags are settled under the lock, and the calls
        run outside it, since a future that is already done runs its callback inline.
        """
        with self._lock:
            handle = operation.handle
            if handle is None or operation.terminal:
                return
            ask_result = not operation.result_requested
            operation.result_requested = True
        if ask_result:
            self._ask_result(operation, handle)
        with self._lock:
            ask_cancel = (
                operation.abandoned is not None
                and not operation.cancel_sent
                and not operation.terminal
            )
            operation.cancel_sent = operation.cancel_sent or ask_cancel
        if ask_cancel:
            self._ask_cancel(operation, handle)

    def _ask_result(self, operation: _Operation, handle: Any) -> None:
        try:
            operation.result_future = handle.get_result_async()
            operation.result_future.add_done_callback(partial(self._on_result, operation))
        except Exception as error:
            self._result_failed(operation, "the request for its result failed", error)

    def _ask_cancel(self, operation: _Operation, handle: Any) -> None:
        logger.info("Cancelling the goal of %s: %s", operation.what, operation.abandoned)
        try:
            handle.cancel_goal_async().add_done_callback(
                partial(self._on_cancel_response, operation)
            )
        except Exception:
            logger.exception("Could not send the cancel of %s.", operation.what)
            with self._lock:
                operation.cancel_sent = False  # A later stop or close sends it again.

    def _on_goal_response(self, operation: _Operation, future: Any) -> None:
        try:
            handle = future.result()
            accepted = handle.accepted
        except Exception as error:
            self._record_error(operation, "the goal's acceptance failed", error)
            return
        with self._lock:
            if accepted:
                # Settled against _abandon by this lock: either it sees the handle, or this
                # sees its mark. The token counts too, since the verb may not have seen it.
                operation.handle = handle
                if operation.abandoned is None and (self._closing or self._cancel.cancelled):
                    operation.abandoned = _STOP_REASON
            else:
                operation.rejected = operation.terminal = True
                self._settle(operation)
        operation.acceptance.set()
        if accepted:
            self._advance(operation)
        else:
            operation.answered.set()

    def _on_result(self, operation: _Operation, future: Any) -> None:
        try:
            response = future.result()
            status = response.status
        except Exception as error:
            self._result_failed(operation, "the request for its result failed", error)
            return
        if status not in _TERMINAL:
            # Nav2 no longer knows the goal, which does not prove that the robot stopped.
            name = _STATUS_NAMES.get(status, str(status))
            self._result_failed(operation, f"Nav2 answered with the status {name}", None)
            return
        if operation.abandoned is not None:
            logger.info("The abandoned goal of %s ended: %s.", operation.what, _STATUS_NAMES[status])
        with self._lock:
            operation.response = response
            operation.terminal = True
            self._settle(operation)
        operation.answered.set()

    def _on_cancel_response(self, operation: _Operation, future: Any) -> None:
        """Note a refused cancel. Any other answer waits for the result, which ends the record."""
        try:
            refused = future.result().return_code == CancelGoal.Response.ERROR_REJECTED
        except Exception:
            logger.exception("The cancel of %s failed.", operation.what)
            refused = True
        if refused:
            logger.warning("Nav2 did not take the cancel of %s.", operation.what)
            with self._lock:
                operation.cancel_sent = False  # A later stop or close sends it again.

    def _result_failed(self, operation: _Operation, message: str, error: Exception | None) -> None:
        """Record it, and let a later stop or close ask for the result again."""
        with self._lock:
            operation.result_requested = False
        self._record_error(operation, message, error)

    def _record_error(self, operation: _Operation, message: str, error: Exception | None) -> None:
        """Keep a callback's failure off the executor, and end the verb's wait. The record stays."""
        logger.error("The goal of %s: %s.", operation.what, message, exc_info=error)
        with self._lock:
            operation.error = message
        operation.acceptance.set()
        operation.answered.set()

    # -- The pose history -------------------------------------------------------

    def _sample_pose(self) -> None:
        """Keep the newest map <- base transform, on the executor. Never raises, never waits.

        A read with no transform keeps nothing, as does a transform that gives no
        pose, one no newer than the last sample, or one stamped well ahead of the
        clock, which predates a clock rollback.
        """
        try:
            stamped = self._tf.lookup_transform(MAP_FRAME, ROBOT_FRAME, Time())
            if _transform_problem(stamped) is not None:
                return
            stamp_ns = stamped.header.stamp.sec * _NS_PER_S + stamped.header.stamp.nanosec
            pose = _pose_of(stamped)
            with self._history_lock:
                now_ns = self._clock_ns()
                self._note_clock(now_ns)
                stamp_s = stamp_ns / _NS_PER_S
                newer = not self._history or stamp_s > self._history[-1][0]
                if newer and stamp_ns <= now_ns + _MAX_FUTURE_STAMP_NS:
                    self._history.append((stamp_s, pose))
                    while self._history[0][0] < stamp_s - POSE_HISTORY_S:
                        self._history.popleft()
        except TransformException:
            return
        except Exception:
            # Once until a sample succeeds again, so a lasting fault does not log at 5 Hz.
            if not self._sample_failed:
                logger.warning("Could not sample the robot's pose for pose_at.", exc_info=True)
            self._sample_failed = True
            return
        self._sample_failed = False

    def _clock_ns(self) -> int:
        try:
            return self._node.get_clock().now().nanoseconds
        except Exception as error:
            raise ros_failure(error, "Reading the clock") from error

    def _note_clock(self, now_ns: int) -> None:
        """Call with the history lock held: clear the history if the clock moved backwards."""
        if now_ns < self._history_clock_ns:
            logger.info("The ROS clock moved backwards, so the pose history starts again.")
            self._history.clear()
        self._history_clock_ns = now_ns

    # -- TF --------------------------------------------------------------------

    def _fresh_pose(self, budget_s: float, what: str) -> Pose2D:
        """The robot's pose from a sample stamped after this call began.

        The newest transform in the buffer can predate the end of the last goal, since
        a result and a TF update arrive separately. ROS time decides what is fresh, and
        monotonic time bounds the wait, so a paused /clock cannot hang it.
        """
        clock = self._node.get_clock()
        try:
            asked = latest = clock.now().nanoseconds
        except Exception as error:
            raise ros_failure(error, "Reading the clock") from error
        pose: Pose2D | None = None
        rewound = stale = False
        invalid: str | None = None

        def read() -> bool:
            nonlocal latest, pose, rewound, stale, invalid
            try:
                now = clock.now().nanoseconds
                stamped = self._connector.get_transform(
                    MAP_FRAME, ROBOT_FRAME, timeout_sec=_TF_LOOKUP_S
                )
            except TransformException:
                return False
            except Exception as error:
                raise ros_failure(error, "Reading TF") from error
            if now < latest:
                rewound = True
                return True
            latest = now
            invalid = _transform_problem(stamped)
            if invalid is not None:
                return False
            stamp_ns = stamped.header.stamp.sec * 1_000_000_000 + stamped.header.stamp.nanosec
            if not asked < stamp_ns <= now + _MAX_FUTURE_STAMP_NS:
                stale = True
                return False
            pose = _pose_of(stamped)
            return True

        try:
            self._cancel.wait_for(read, budget_s, what)
        except StepError as error:
            if error.code is not ErrorCode.TIMEOUT:
                raise
            if invalid is not None:
                raise StepError(
                    ErrorCode.SKILL_FAILED, f"{_POSITION} is not usable: {invalid}."
                ) from error
            if stale:
                raise StepError(
                    ErrorCode.TIMEOUT,
                    f"{_POSITION} is out of date. TF is not updating, or the agent and the "
                    "robot disagree on sim time (see --use-sim-time).",
                ) from error
            raise StepError(
                ErrorCode.UNAVAILABLE, f"{_POSITION} is unknown.{self._tf_hint()}"
            ) from error
        if rewound:
            raise StepError(
                ErrorCode.TIMEOUT, f"{_POSITION} is unknown: the ROS clock moved backwards."
            )
        return pose

    def _tf_hint(self) -> str:
        """Name the likely cause. On another domain or router, a running robot looks absent."""
        try:
            publishers = self._node.count_publishers("/tf") + self._node.count_publishers("/tf_static")
        except Exception:
            logger.debug("Could not count the TF publishers.", exc_info=True)
            return ""
        if publishers:
            return " TF is running without this frame, so check localization and the frame names."
        return (
            " No node publishes /tf: either the robot is not running, or the agent uses a "
            "different ROS_DOMAIN_ID or Zenoh router than the robot."
        )


def _transform_problem(stamped: Any) -> str | None:
    """Why a transform cannot give a pose, or None. A quaternion of zero length has no heading."""
    frames = stamped.header.frame_id, stamped.child_frame_id
    if frames != (MAP_FRAME, ROBOT_FRAME):
        return f"it came as {frames[0]} -> {frames[1]}"
    t, q = stamped.transform.translation, stamped.transform.rotation
    if not all(is_finite_number(value) for value in (t.x, t.y, q.x, q.y, q.z, q.w)):
        return "it holds a value that is not finite"
    if math.hypot(q.x, q.y, q.z, q.w) < 1e-6:
        return "its rotation has zero length"
    return None


def _pose_of(stamped: Any) -> Pose2D:
    """The transform as a pose. The yaw formula holds for a rotation of any length."""
    t, q = stamped.transform.translation, stamped.transform.rotation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), q.w**2 + q.x**2 - q.y**2 - q.z**2)
    return Pose2D(t.x, t.y, yaw)


def _interpolate(history: deque[tuple[float, Pose2D]], stamp_s: float) -> Pose2D | None:
    """The pose at `stamp_s` by the history, as RosBase.pose_at describes it."""
    if not history:
        return None
    first_s, first = history[0]
    last_s, last = history[-1]
    if stamp_s <= first_s:
        return first if first_s - stamp_s <= POSE_MATCH_S else None
    if stamp_s >= last_s:
        return last if stamp_s - last_s <= POSE_MATCH_S else None
    # The first sample stamped at or after stamp_s; the one before it is stamped earlier.
    index = bisect.bisect_left(history, stamp_s, key=lambda sample: sample[0])
    after_s, after = history[index]
    before_s, before = history[index - 1]
    if after_s - before_s > POSE_MATCH_S:
        # A gap in the history: the robot need not have driven the straight line
        # between its two sides, so each side is held as an end of the history is.
        if stamp_s - before_s <= after_s - stamp_s:
            return before if stamp_s - before_s <= POSE_MATCH_S else None
        return after if after_s - stamp_s <= POSE_MATCH_S else None
    share = (stamp_s - before_s) / (after_s - before_s)
    turn = math.remainder(after.yaw - before.yaw, math.tau)
    return Pose2D(
        before.x + share * (after.x - before.x),
        before.y + share * (after.y - before.y),
        math.remainder(before.yaw + share * turn, math.tau),
    )
