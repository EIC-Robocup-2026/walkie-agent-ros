"""CommanderClient: the agent's one client of walkie-ros's bimanual_commander (ADR-0021).

The arm atomics and the navigation interlock reach the arms only through it,
as the `arm` backend. It uses the three names ADR-0021 allows:
/set_joint_position and /control_gripper (my_robot_interfaces actions) and
/get_joint_states (a service).

It is created once per agent from the connector, like VisionFeed: its
clients are made here, once, on the connector's node, and its methods block
the calling thread, never the executor's. Each method takes the caller's
cancel event (an atomic passes ctx.cancel). A cancel or a timeout asks the
commander to cancel a running goal, because the commander keeps moving the
arms until told otherwise, and the call then returns without waiting for
the goal to end.

One call at a time. The commander runs every goal in its own thread with no
lock on the arms, so two goals would fight, and its /get_joint_states calls
MoveIt's getCurrentState(), which the commander's own comments call unsafe
to run concurrently (ADR-0021, ADR-0024). So every call holds a lock, and a
read or a second goal while a goal runs is refused at once rather than
queued behind a move that can take 30 s.

A goal outlives the call that sent it. It is outstanding from its send
until the client sees its result, sees it rejected, or abandons it, and
until then every other call is refused, also after its own call gave up on
a cancel or a timeout. Such a goal can still move the arms: the commander
may accept it late, or still be stopping it. A cancel of it would also stop
a newer goal, since the commander's cancel stops both arms.
"""

import functools
import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from my_robot_interfaces.action import ControlGripper, SetJointPosition
from my_robot_interfaces.srv import GetJointStates
from rai.communication.ros2 import ROS2Connector
from rclpy.action import ActionClient
from rclpy.task import Future

from walkie_agent.arm.poses import (
    ARM_GROUPS,
    ARMS,
    BOTH_ARMS_GROUP,
    GRASP_CLOSE_THRESHOLD,
    GRIPPER_GROUPS,
    RIGHT_JOINT2,
    ArmPoses,
    load_arm_poses,
)
from walkie_agent.arm.state import ArmsState, ArmState
from walkie_agent.toolbox.types import StepError
from walkie_agent.tools.waiters import wait_for_future

logger = logging.getLogger(__name__)

# Absolute, so the names do not depend on the agent node's namespace. The
# commander's node runs in the root namespace.
SET_JOINT_POSITION_ACTION = "/set_joint_position"
CONTROL_GRIPPER_ACTION = "/control_gripper"
GET_JOINT_STATES_SERVICE = "/get_joint_states"

# Added to every UNAVAILABLE message.
COMMANDER_HINT = "Is bimanual_commander running?"
# Logged when the commander is found but does not answer. rmw_zenoh_cpp finds
# a server by name and type name but sends requests to a key that includes the
# type hash, so a commander built from another my_robot_interfaces never
# receives them (ADR-0018, ADR-0035). A busy commander looks the same, so the
# hint goes to the log for a developer, not to the LLM, as find_objects does.
STALE_PIN_HINT = (
    "If bimanual_commander is running and not busy, it was probably built from another "
    "my_robot_interfaces than the walkie-ros commit pinned in ros2_ws.repos."
)

# ADR-0024 bounds the read after every motion at 3 s.
READ_TIMEOUT_S = 3.0
# The commander ramps a gripper over its full travel in under half a second
# at its launch default, then waits up to 5 s for the controller's result,
# which is how long a close that stalls on an object takes to end.
GRIPPER_TIMEOUT_S = 10.0
# How long a cancelled or timed-out call waits for the commander to answer
# the cancel request. The commander's cancel callback stops both arms before
# it accepts, so an accepted cancel means the arms were told to stop. The
# goal's own result can take seconds more; the call does not wait for it,
# and the goal stays outstanding until it comes.
CANCEL_ANSWER_S = 1.0
# How long after its call gave up a goal may stay outstanding before the
# client abandons it and takes calls again. The result normally comes within
# seconds: a cancelled move stops at once, and a cancelled gripper command
# finishes its ramp and the commander's 5 s wait for the controller. But a
# commander that restarted keeps its action servers discovered, and the old
# goal's result never comes, so discovery alone cannot end the wait. A goal
# whose action server is no longer discovered is abandoned at the next check.
ABANDON_AFTER_S = 60.0
# How long a call waits for the commander's server to be discovered. Not
# zero, so a call soon after the agent starts does not fail on discovery;
# short, so a missing commander costs little. available() never waits.
READY_WAIT_S = 1.0

# How often a wait looks at the cancel event and the clock, as
# ToolContext.wait() does.
_POLL_S = 0.05

# The goals the agent may send. The *_lift groups also move the lift, which
# the agent never commands (ADR-0026).
ARM_GOAL_GROUPS = frozenset((*ARM_GROUPS.values(), BOTH_ARMS_GROUP))
GRIPPER_GOAL_GROUPS = frozenset(GRIPPER_GROUPS.values())
READ_GROUPS = ARM_GOAL_GROUPS | GRIPPER_GOAL_GROUPS
# The arms each arm group moves.
_GROUP_ARMS = {ARM_GROUPS[arm]: (arm,) for arm in ARMS} | {BOTH_ARMS_GROUP: ARMS}

# The commander's status strings that mean a joint goal failed.
_PLANNING_FAILED = "Planning Failed"
_EXECUTION_FAILED = "Execution Failed"
_CANCELED = "Canceled"
_INVALID_GROUP = "Invalid Group"

_STATUS_NAMES = {
    getattr(GoalStatus, name): name.removeprefix("STATUS_")
    for name in dir(GoalStatus)
    if name.startswith("STATUS_")
}

# What a call that cancelled its goal says happened, by the commander's
# answer (_cancel_outcome): the words for an arm goal, then for a gripper
# goal. The commander's cancel callback stops both arms before it accepts,
# but a gripper finishes the motion it is in.
_CANCEL_OUTCOMES = {
    "accepted": (
        "the arms were told to stop",
        "the commander accepted the cancel, but the gripper finishes its current motion",
    ),
    "ended": ("the goal had already ended", "the command had already ended"),
    "refused": (
        "the commander refused the cancel, so an arm may still be moving",
        "the commander refused the cancel, so the gripper may still be moving",
    ),
    "unknown": (
        "the commander does not know the goal, so an arm may still be moving",
        "the commander does not know the command, so the gripper may still be moving",
    ),
    "no answer": (
        "the commander did not answer the cancel, so an arm may still be moving",
        "the commander did not answer the cancel, so the gripper may still be moving",
    ),
}


@dataclass(frozen=True)
class GripperResult:
    """The commander's answer to one gripper goal."""

    # True when the controller reached the commanded position. A real grasp
    # usually ends false, since the fingers stall on the object; on a close,
    # `grasped` and the gap are the answer.
    success: bool
    status: str
    # Only for a close, a position at or below GRASP_CLOSE_THRESHOLD. On an
    # open or "half" the commander reports false and 0.0 whatever the
    # fingers did, so they are None here and cannot be misread. A close the
    # commander did not measure, with grasp_detect_enable off, superseded,
    # or with no finger reading, reports false and 0.0 too, which looks like
    # a measured empty close.
    grasped: bool | None
    # The settled travel of one finger, in metres.
    gripper_gap: float | None


@dataclass(eq=False)
class _Goal:
    """A goal sent to the commander, kept until the client knows that it ended."""

    # Such as "the move of left_arm" or "the left_gripper command".
    what: str
    # The group it was sent for.
    group: str
    client: ActionClient
    # The send_goal_async future.
    sent: Future
    # The get_result_async future, once the goal was accepted.
    result: Future | None = None
    # time.monotonic() when the call that sent it gave up on it; None while
    # that call still waits for it.
    gave_up_at: float | None = None

    def ended(self) -> bool:
        """True once its result came, or once it was not accepted."""
        if self.result is not None:
            return self.result.done() or self.result.cancelled()
        if not self.sent.done():
            return self.sent.cancelled()
        if self.sent.exception() is not None:
            return True
        handle = self.sent.result()
        return handle is None or not handle.accepted


class CommanderClient:
    """The `arm` backend: joint goals, gripper goals and joint reads on the commander.

    Created once per agent. Without a commander it does no harm: a call
    then fails with UNAVAILABLE. `poses` are the checked named poses, used to
    say which pose each arm is in; by default config/arm_poses.toml, checked
    against config/robot_model.toml.

    Every blocking method raises StepError:
      - CANCELLED once `cancel` is set, after asking the commander to cancel
        a running goal; the message says how it answered;
      - TIMEOUT when the commander does not finish within `timeout_s`, also
        after asking it to cancel the goal;
      - UNAVAILABLE when the commander's server is not there;
      - PRECONDITION_FAILED while a goal is outstanding: one that another
        call waits for, or one that an earlier call gave up on and that has
        not ended.
    A group the agent may not use, or a value that is not a finite number,
    raises ValueError: that is a bug in reviewed code, not something for the
    LLM to retry.
    """

    def __init__(self, connector: ROS2Connector, poses: ArmPoses | None = None) -> None:
        self.poses = poses if poses is not None else load_arm_poses()
        node = connector.node
        # Created here, once: rclpy entities must not be created per call.
        self._joint_states = node.create_client(GetJointStates, GET_JOINT_STATES_SERVICE)
        self._set_joint_position = ActionClient(node, SetJointPosition, SET_JOINT_POSITION_ACTION)
        self._control_gripper = ActionClient(node, ControlGripper, CONTROL_GRIPPER_ACTION)
        self._node = node
        # The joints of each arm, as arm_state() counts them: those its
        # poses set, which are all the joints of its group.
        self._arm_joints = {
            arm: frozenset(
                joint for pose in self.poses.poses.values() for joint in pose.tables.get(arm, {})
            )
            for arm in ARMS
        }
        # Held for the whole of every call.
        self._lock = threading.Lock()
        # What the goal holding the lock does, such as "the move of
        # left_arm"; None while no goal runs. Read without the lock, to
        # refuse a second call at once.
        self._running: str | None = None
        # Guards what the methods remember, which the gaze loop reads from
        # its own thread while a call holds the call lock, and the goal
        # record, which done callbacks clear from the executor's threads.
        self._state_lock = threading.Lock()
        self._last_state = ArmsState()
        self._joint_order: dict[str, tuple[str, ...]] = {}
        self._gripper_positions: dict[str, float | None] = dict.fromkeys(GRIPPER_GOAL_GROUPS)
        # The goal whose end the client does not know yet, or None. One at
        # most: no call starts while it is set.
        self._goal: _Goal | None = None

    # -- Without waiting ----------------------------------------------------------

    def available(self) -> bool:
        """True when the commander's service and both actions are discovered. Never waits."""
        return (
            self._joint_states.service_is_ready()
            and self._set_joint_position.server_is_ready()
            and self._control_gripper.server_is_ready()
        )

    def busy(self) -> str | None:
        """The goal that keeps the commander from taking a call, or None. Never waits.

        Such as "the move of left_arm" while its call waits for it, or "the
        move of left_arm from an earlier call" once that call gave up on it
        and until it ends. Either way it may still move an arm, so the drive
        check refuses on it (ADR-0025). A running call counts even when the
        commander is no longer discovered; a goal its call gave up on is
        abandoned then, or after ABANDON_AFTER_S (_outstanding).
        """
        running = self._running
        if running is not None:
            return running
        goal = self._outstanding()
        return None if goal is None else f"{goal.what} from an earlier call"

    @property
    def last_state(self) -> ArmsState:
        """The arms as of the last read, for callers that must not call the commander.

        An arm becomes unknown when a goal for it is sent or a read of it
        is not usable, and known again with the next good read of its
        joints, so a moving arm is never reported in its old pose.
        ArmsState() until the first read.
        """
        with self._state_lock:
            return self._last_state

    def last_gripper_position(self, group: str) -> float | None:
        """The position last commanded to `group`, or None when unknown.

        get_arm_state gives it only when it cannot read the finger. It is
        the command, not where the fingers went: it is kept also when the
        commander reports the goal failed, as a close that stalls on an
        object does. None before the first command, while one runs, and
        after one was cancelled, timed out or rejected, since the fingers
        may then have stopped anywhere.
        """
        _check_group(group, GRIPPER_GOAL_GROUPS)
        with self._state_lock:
            return self._gripper_positions[group]

    # -- Reads ----------------------------------------------------------------

    def joint_states(
        self, group: str, cancel: threading.Event, timeout_s: float = READ_TIMEOUT_S
    ) -> dict[str, float]:
        """The joints of `group` by name, in the commander's order, read once.

        The order is the one /set_joint_position expects for the group, and
        it lacks openarm_right_joint2 when walkie-ros runs with
        right_joint2_fixed. Raises StepError as the class says, and INTERNAL
        when the commander does not know the group, answers malformed or
        gives a position that is not a finite number; the arms of the group
        are then unknown. An arm of the group whose joints the reading lacks
        is unknown too.
        """
        _check_group(group, READ_GROUPS)
        deadline = _deadline(timeout_s)
        with self._exclusive(f"read the joints of {group}", deadline, cancel):
            self._wait_ready(
                self._joint_states.service_is_ready,
                f"{GET_JOINT_STATES_SERVICE} service",
                deadline,
                cancel,
            )
            future = self._joint_states.call_async(GetJointStates.Request(group_name=group))
            try:
                response = _wait(future, deadline, cancel)
            except StepError as e:
                # The answer would otherwise be kept for a caller that left.
                self._joint_states.remove_pending_request(future)
                if e.code == "CANCELLED":
                    raise StepError(
                        "CANCELLED", f"A stop or a reset interrupted the read of {group}."
                    ) from None
                logger.warning(
                    f"{GET_JOINT_STATES_SERVICE} did not answer within {timeout_s:g} s. "
                    f"{STALE_PIN_HINT}"
                )
                raise StepError(
                    "TIMEOUT",
                    f"The commander's {GET_JOINT_STATES_SERVICE} did not answer within "
                    f"{timeout_s:g} s. {COMMANDER_HINT}",
                ) from None
            try:
                reading = _reading(group, response)
            except StepError as e:
                if e.code == "INTERNAL":
                    # The commander answered, but not with positions that
                    # can be trusted, so the arms may be anywhere.
                    self._forget_arms(group)
                raise
            self._remember_reading(group, reading)
            return reading

    def joint_names(
        self, group: str, cancel: threading.Event, timeout_s: float = READ_TIMEOUT_S
    ) -> tuple[str, ...]:
        """The commander's joint order for `group`: from the last read, else read now.

        The order only changes when walkie-ros restarts with other launch
        arguments; a goal it then rejects drops the cached order.
        """
        _check_group(group, READ_GROUPS)
        with self._state_lock:
            cached = self._joint_order.get(group)
        if cached is not None:
            return cached
        return tuple(self.joint_states(group, cancel, timeout_s))

    def read_arms(self, cancel: threading.Event, timeout_s: float = READ_TIMEOUT_S) -> ArmsState:
        """Both arms' named poses from one read of both_arms; see joint_states().

        Also raises StepError INTERNAL when the reading lacks a joint of
        either arm, which is then unknown.
        """
        reading = self.joint_states(BOTH_ARMS_GROUP, cancel, timeout_s)
        incomplete = [arm for arm in ARMS if self._missing(arm, reading)]
        if incomplete:
            missing = [joint for arm in incomplete for joint in self._missing(arm, reading)]
            which = "either arm" if len(incomplete) == len(ARMS) else f"the {incomplete[0]} arm"
            raise StepError(
                "INTERNAL",
                f"The commander's reading of {BOTH_ARMS_GROUP} lacks {', '.join(missing)}, so "
                f"it cannot say where {which} is.",
                {"missing": missing},
            )
        return ArmsState.from_joints(reading, self.poses.poses)

    # -- Goals ----------------------------------------------------------------

    def set_joint_position(
        self,
        group: str,
        positions: Sequence[float],
        cancel: threading.Event,
        timeout_s: float,
    ) -> str:
        """Move `group` to `positions`, in joint_names() order, and wait for the result.

        Returns the commander's status on success: "Success", or "Plan Only"
        when the commander runs in plan-only mode and nothing moved. The
        caller checks the joints afterwards either way (ADR-0024). Raises
        StepError as the class says, and:
          - INTERNAL when the commander rejects the goal, which it does for
            a position count that differs from the group's joints;
          - SKILL_FAILED for "Planning Failed" (nothing moved) and
            "Execution Failed" (the motion stopped part way);
          - CANCELLED when the commander reports the goal cancelled.
        """
        _check_group(group, ARM_GOAL_GROUPS)
        values = _finite(positions, "positions")
        deadline = _deadline(timeout_s)
        what = f"the move of {group}"
        with self._exclusive(f"move {group}", deadline, cancel, running=what):
            # Before the goal is sent: from then on the arm may move, and the
            # gaze loop must not see its old pose.
            self._forget_arms(group)
            response = self._run_goal(
                self._set_joint_position,
                SET_JOINT_POSITION_ACTION,
                SetJointPosition.Goal(group_name=group, joint_positions=values),
                group,
                what,
                deadline,
                timeout_s,
                cancel,
            )
            if response is None:
                # Under the call lock, so a newer call's read of the order
                # is not dropped.
                with self._state_lock:
                    self._joint_order.pop(group, None)
                raise StepError(
                    "INTERNAL",
                    f"The commander rejected the goal for {group} with {len(values)} positions. "
                    "It rejects a count that differs from the group's joints; the joint order "
                    "is read again on the next call.",
                    {"group": group, "count": len(values)},
                )
        result = response.result
        if response.status == GoalStatus.STATUS_SUCCEEDED and result.success:
            return result.status
        raise _move_failed(group, response.status, result.status)

    def control_gripper(
        self,
        group: str,
        position: float,
        cancel: threading.Event,
        timeout_s: float = GRIPPER_TIMEOUT_S,
    ) -> GripperResult:
        """Move `group`'s fingers to `position`, metres of one finger's travel.

        Returns the result whatever `success` is, since a real grasp usually
        ends with success false. Raises StepError as the class says, INTERNAL
        when the commander rejects the goal, and CANCELLED when it reports
        the goal cancelled.
        """
        _check_group(group, GRIPPER_GOAL_GROUPS)
        [position] = _finite([position], "position")
        deadline = _deadline(timeout_s)
        what = f"the {group} command"
        with self._exclusive(f"move {group}", deadline, cancel, running=what):
            self._remember_gripper(group, None)
            response = self._run_goal(
                self._control_gripper,
                CONTROL_GRIPPER_ACTION,
                ControlGripper.Goal(group_name=group, position=position),
                group,
                what,
                deadline,
                timeout_s,
                cancel,
            )
            if response is None:
                raise StepError(
                    "INTERNAL", f"The commander rejected the goal for {group}.", {"group": group}
                )
            result = response.result
            if response.status == GoalStatus.STATUS_CANCELED:
                raise StepError(
                    "CANCELLED",
                    f"The commander cancelled {what}.",
                    {"group": group, "status": result.status},
                )
            # Under the call lock, so an older call cannot overwrite the
            # position of a newer one.
            self._remember_gripper(group, position)
        closed = position <= GRASP_CLOSE_THRESHOLD
        return GripperResult(
            success=response.status == GoalStatus.STATUS_SUCCEEDED and result.success,
            status=result.status,
            grasped=bool(result.grasped) if closed else None,
            gripper_gap=float(result.gripper_gap) if closed else None,
        )

    def destroy(self) -> None:
        """Destroy the clients. Node.destroy_node() does not destroy action clients."""
        self._set_joint_position.destroy()
        self._control_gripper.destroy()
        self._node.destroy_client(self._joint_states)

    # -- Helpers ----------------------------------------------------------------

    @contextmanager
    def _exclusive(
        self,
        action: str,
        deadline: float,
        cancel: threading.Event,
        running: str | None = None,
    ) -> Iterator[None]:
        """Hold the call lock; `running` names the goal, if the call sends one.

        Refuses at once while a goal is outstanding (_Goal).
        """
        while True:
            _check_cancelled(cancel)
            self._check_free(action)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StepError(
                    "TIMEOUT",
                    f"Could not {action}: another call to the commander did not finish in time.",
                )
            if self._lock.acquire(timeout=min(_POLL_S, remaining)):
                break
        try:
            # The call that held the lock may have left its goal outstanding.
            self._check_free(action)
            self._running = running
            yield
        finally:
            self._running = None
            self._lock.release()

    def _check_free(self, action: str) -> None:
        """Raises StepError PRECONDITION_FAILED while a goal is outstanding."""
        running = self._running
        if running is not None:
            raise StepError(
                "PRECONDITION_FAILED",
                f"Cannot {action} while {running} is running. Wait for it to finish.",
                {"running": running},
            )
        goal = self._outstanding()
        if goal is not None:
            raise StepError(
                "PRECONDITION_FAILED",
                f"Cannot {action}: {goal.what} from an earlier call has not ended yet. "
                "Try again in a few seconds.",
                {"outstanding": goal.what},
            )

    @staticmethod
    def _wait_ready(
        ready: Callable[[], bool], name: str, deadline: float, cancel: threading.Event
    ) -> None:
        until = min(deadline, time.monotonic() + READY_WAIT_S)
        while not ready():
            _check_cancelled(cancel)
            if time.monotonic() >= until:
                raise StepError(
                    "UNAVAILABLE",
                    f"The commander's {name} is not available. {COMMANDER_HINT}",
                    {"name": name},
                )
            time.sleep(_POLL_S)

    def _run_goal(
        self,
        client: ActionClient,
        name: str,
        goal: Any,
        group: str,
        what: str,
        deadline: float,
        timeout_s: float,
        cancel: threading.Event,
    ) -> Any | None:
        """Send `goal` and return its GetResult response, or None when it was rejected.

        The goal is outstanding from its send until its end is known, also
        after this call gives up on a cancel or a timeout (_Goal).
        """
        self._wait_ready(client.server_is_ready, f"{name} action", deadline, cancel)
        record = _Goal(what, group, client, client.send_goal_async(goal))
        with self._state_lock:
            self._goal = record
        try:
            return self._wait_goal(record, name, deadline, timeout_s, cancel)
        finally:
            self._release(record)

    def _wait_goal(
        self, goal: _Goal, name: str, deadline: float, timeout_s: float, cancel: threading.Event
    ) -> Any | None:
        try:
            handle = _wait(goal.sent, deadline, cancel)
        except StepError as e:
            # The commander may still accept the goal after this call has
            # given up; it must not then move the arms with nobody waiting.
            goal.sent.add_done_callback(functools.partial(self._on_late_answer, goal))
            if e.code == "TIMEOUT":
                logger.warning(
                    f"{name} neither accepted nor rejected a goal within {timeout_s:g} s. "
                    f"{STALE_PIN_HINT}"
                )
            # A gripper finishes its motion whatever the cancel.
            outcome = (
                "if the commander accepts it later, the gripper still makes that motion"
                if goal.group in GRIPPER_GOAL_GROUPS
                else "if the commander accepts it later, it is cancelled at once"
            )
            raise _ended(e, goal.what, timeout_s, outcome) from None
        if handle is None or not handle.accepted:
            return None
        result = handle.get_result_async()
        with self._state_lock:
            goal.result = result
        # Not under the state lock: on a future that is already done, the
        # callback can run at once, in this thread.
        result.add_done_callback(functools.partial(self._end_goal, goal))
        try:
            return _wait(result, deadline, cancel)
        except StepError as e:
            raise _ended(e, goal.what, timeout_s, self._cancel(handle, goal)) from None

    def _cancel(self, handle: Any, goal: _Goal) -> str:
        """Ask the commander to cancel the goal of `handle`; what its answer means, in words."""
        # wait_for_future itself, not _wait: `cancel` may already be set.
        answer = wait_for_future(handle.cancel_goal_async(), CANCEL_ANSWER_S)
        outcome = _cancel_outcome(answer, handle.goal_id)
        words = _CANCEL_OUTCOMES[outcome][1 if goal.group in GRIPPER_GOAL_GROUPS else 0]
        if outcome not in ("accepted", "ended"):
            logger.warning(f"Cancel of {goal.what}: {words}.")
        return words

    def _on_late_answer(self, goal: _Goal, sent: Future) -> None:
        """The commander answered a goal after its call gave up waiting for that answer.

        A done callback, run by the connector's executor, so it must not
        raise. An accepted goal is cancelled at once, and stays outstanding
        until its result comes. A gripper goal still finishes its motion.
        """
        try:
            handle = None if sent.cancelled() or sent.exception() is not None else sent.result()
            if handle is None or not handle.accepted:
                self._end_goal(goal)
                return
            logger.warning(
                f"The commander accepted {goal.what} after its call gave up; cancelling it."
            )
            handle.cancel_goal_async()
            result = handle.get_result_async()
            with self._state_lock:
                goal.result = result
            result.add_done_callback(functools.partial(self._end_goal, goal))
        except Exception:
            logger.exception(f"Could not handle the late answer to {goal.what}.")

    def _end_goal(self, goal: _Goal, _future: Future | None = None) -> None:
        """Forget `goal`, which has ended. Also a done callback, so it must not raise."""
        with self._state_lock:
            if self._goal is goal:
                self._goal = None

    def _release(self, goal: _Goal) -> None:
        """As the call that sent `goal` returns: forget it if it ended, else keep it outstanding."""
        with self._state_lock:
            if self._goal is not goal:
                return
            if goal.ended():
                self._goal = None
            else:
                goal.gave_up_at = time.monotonic()

    def _outstanding(self) -> _Goal | None:
        """The goal that a call gave up on and whose end is not known yet, else None.

        Checked when asked, with no timer: forgets the goal once it ended,
        and abandons it once its action server is no longer discovered or
        ABANDON_AFTER_S has passed since its call gave up.
        """
        with self._state_lock:
            goal = self._goal
            # While gave_up_at is None, the call that sent the goal still
            # holds the call lock, and `_running` names it.
            if goal is None or goal.gave_up_at is None:
                return None
            if goal.ended():
                self._goal = None
                return None
            waited_s = time.monotonic() - goal.gave_up_at
        if not goal.client.server_is_ready():
            self._abandon(goal, "its action server is no longer discovered")
        elif waited_s >= ABANDON_AFTER_S:
            self._abandon(
                goal,
                f"no result came within {ABANDON_AFTER_S:g} s after its call gave up, as when "
                "the commander restarted",
            )
        else:
            return goal
        return None

    def _abandon(self, goal: _Goal, reason: str) -> None:
        with self._state_lock:
            if self._goal is not goal:
                return
            self._goal = None
        # What the goal moved stays unknown: its call forgot it before the
        # send, and no read ran since.
        logger.warning(
            f"Abandoned {goal.what}: {reason}. Calls to the commander are taken again; "
            "what it moved is unknown until read or commanded again."
        )

    def _forget_arms(self, group: str) -> None:
        arms = _GROUP_ARMS.get(group, ())
        with self._state_lock:
            old = self._last_state
            self._last_state = ArmsState(
                left=None if "left" in arms else old.left,
                right=None if "right" in arms else old.right,
            )

    def _missing(self, arm: str, reading: Mapping[str, float]) -> list[str]:
        """The joints of `arm` that `reading` lacks, by name.

        openarm_right_joint2 does not count: walkie-ros launched with
        right_joint2_fixed holds it at 0 and does not list it.
        """
        joints = self._arm_joints[arm] - {RIGHT_JOINT2}
        return sorted(joint for joint in joints if joint not in reading)

    def _remember_reading(self, group: str, reading: dict[str, float]) -> None:
        # An arm of the group takes its state from the reading, or becomes
        # unknown when the reading lacks one of its joints. A read of one
        # arm's group or of a gripper says nothing about the other arm, which
        # keeps its last state: only this client moves the arms (ADR-0021).
        read = ArmsState.from_joints(reading, self.poses.poses)
        incomplete = {arm for arm in _GROUP_ARMS.get(group, ()) if self._missing(arm, reading)}
        covered: dict[str, ArmState | None] = {
            arm: None if arm in incomplete else read.arm(arm) for arm in _GROUP_ARMS.get(group, ())
        }
        with self._state_lock:
            # The order of an incomplete reading is not kept, so a later move
            # reads the order again instead of being refused on it.
            if incomplete:
                self._joint_order.pop(group, None)
            else:
                self._joint_order[group] = tuple(reading)
            old = self._last_state
            self._last_state = ArmsState(
                left=covered.get("left", old.left), right=covered.get("right", old.right)
            )

    def _remember_gripper(self, group: str, position: float | None) -> None:
        with self._state_lock:
            self._gripper_positions[group] = position


def _wait(future: Future, deadline: float, cancel: threading.Event) -> Any:
    """The future's result. Raises StepError CANCELLED or TIMEOUT, with a generic message.

    wait_for_future in short slices, so a cancel ends the wait within one
    slice; each slice leaves one done callback on the future, which only
    sets an event.
    """
    while True:
        _check_cancelled(cancel)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StepError("TIMEOUT", "The commander did not answer in time.")
        result = wait_for_future(future, min(_POLL_S, remaining))
        if future.done():
            return result


def _cancel_outcome(answer: Any, goal_id: Any) -> str:
    """The key in _CANCEL_OUTCOMES for the commander's answer to a cancel of `goal_id`."""
    if answer is None:
        return "no answer"
    if answer.return_code == CancelGoal.Response.ERROR_GOAL_TERMINATED:
        return "ended"
    if answer.return_code == CancelGoal.Response.ERROR_UNKNOWN_GOAL_ID:
        return "unknown"
    if answer.return_code == CancelGoal.Response.ERROR_NONE and any(
        bytes(info.goal_id.uuid) == bytes(goal_id.uuid) for info in answer.goals_canceling
    ):
        return "accepted"
    # An rclcpp server, such as the commander's, answers a refused cancel
    # ERROR_REJECTED; an rclpy server answers ERROR_NONE without the goal.
    return "refused"


def _ended(error: StepError, what: str, timeout_s: float, outcome: str) -> StepError:
    """The StepError for a goal call that a cancel or its timeout ended."""
    if error.code == "CANCELLED":
        return StepError("CANCELLED", f"A stop or a reset interrupted {what}; {outcome}.")
    return StepError(
        "TIMEOUT", f"{what[0].upper()}{what[1:]} did not finish within {timeout_s:g} s; {outcome}."
    )


def _move_failed(group: str, goal_status: int, status: str) -> StepError:
    data = {"group": group, "status": status}
    if status == _PLANNING_FAILED:
        return StepError(
            "SKILL_FAILED", "MoveIt found no path (Planning Failed). The arms did not move.", data
        )
    if status == _EXECUTION_FAILED:
        return StepError(
            "SKILL_FAILED",
            "The motion stopped before the target (Execution Failed). The arms may have "
            "stopped part way.",
            data,
        )
    if status == _CANCELED:
        return StepError("CANCELLED", f"The commander cancelled the move of {group}.", data)
    if status == _INVALID_GROUP:
        return StepError("INTERNAL", f"The commander does not know the group {group}.", data)
    name = _STATUS_NAMES.get(goal_status, str(goal_status))
    return StepError(
        "SKILL_FAILED", f"The commander reported {status or 'no status'!r} ({name}).", data
    )


def _reading(group: str, response: Any) -> dict[str, float]:
    if response is None:
        raise StepError("UNAVAILABLE", f"The commander gave no answer for {group}.")
    if not response.success:
        status = response.status
        # Both mean the group was wrong, which is a bug here; anything else,
        # such as MoveIt having no robot state yet, is the commander's.
        if status.startswith(("Invalid group", "Joint model group not found")):
            raise StepError("INTERNAL", f"The commander does not know the group {group}: {status}")
        raise StepError(
            "UNAVAILABLE", f"The commander could not read the joints of {group}: {status}"
        )
    names, positions = list(response.joint_names), list(response.joint_positions)
    if not names or len(names) != len(positions):
        raise StepError(
            "INTERNAL",
            f"The commander answered {len(names)} joint names and {len(positions)} "
            f"positions for {group}.",
        )
    reading = {name: float(position) for name, position in zip(names, positions, strict=True)}
    # A NaN would pass the pose match's tolerance check, since every
    # comparison with it is false.
    bad = {name: value for name, value in reading.items() if not math.isfinite(value)}
    if bad:
        listed = ", ".join(f"{name} = {value}" for name, value in bad.items())
        raise StepError(
            "INTERNAL",
            f"The commander's reading of {group} is not usable: {listed}.",
            {"joints": list(bad)},
        )
    return reading


def _check_cancelled(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise StepError("CANCELLED", "A stop or a reset interrupted the call.")


def _check_group(group: str, allowed: frozenset[str]) -> None:
    if group not in allowed:
        raise ValueError(f"{group!r} is not one of the groups {sorted(allowed)}.")


def _deadline(timeout_s: float) -> float:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"timeout_s must be a positive number of seconds, not {timeout_s}.")
    return time.monotonic() + timeout_s


def _finite(values: Sequence[float], name: str) -> list[float]:
    # Every value as a float: rclpy accepts an int in a float64 field but
    # sends it wrong.
    floats = [float(value) for value in values]
    if not all(math.isfinite(value) for value in floats):
        raise ValueError(f"{name} must be finite numbers, not {list(values)}.")
    return floats
