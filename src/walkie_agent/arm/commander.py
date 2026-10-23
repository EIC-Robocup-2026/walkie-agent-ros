"""CommanderClient: the agent's one client of walkie-ros's bimanual_commander (ADR-0021).

The arm atomics and the navigation interlock reach the arms only through it,
as the `arm` backend. It uses the three names ADR-0021 allows:
/set_joint_position and /control_gripper (my_robot_interfaces actions) and
/get_joint_states (a service).

It is created once per agent from the connector, like VisionFeed: its
clients are made here, once, on the connector's node, and its methods block
the calling thread, never the executor's. Each method takes the caller's
cancel event (an atomic passes ctx.cancel). A cancel or a timeout cancels a
running goal, because the commander keeps moving the arms until told
otherwise.

One call at a time. The commander runs every goal in its own thread with no
lock on the arms, so two goals would fight, and its /get_joint_states calls
MoveIt's getCurrentState(), which the commander's own comments call unsafe
to run concurrently (ADR-0021, ADR-0024). So every call holds a lock, and a
read or a second goal while a goal runs is refused at once rather than
queued behind a move that can take 30 s.
"""

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from action_msgs.msg import GoalStatus
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
    ArmPoses,
    load_arm_poses,
)
from walkie_agent.arm.state import ArmsState
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
# the cancel request. Its cancel callback stops the arms before it answers,
# so an answer means the arms were told to stop; the goal's own result can
# take seconds more and is not waited for.
CANCEL_ANSWER_S = 1.0
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
    # fingers did, so they are None here and cannot be misread.
    grasped: bool | None
    # The settled travel of one finger, in metres.
    gripper_gap: float | None


class CommanderClient:
    """The `arm` backend: joint goals, gripper goals and joint reads on the commander.

    Created once per agent. Without a commander it does no harm: a call
    then fails with UNAVAILABLE. `poses` are the checked named poses, used to
    say which pose each arm is in; by default config/arm_poses.toml, checked
    against config/robot_model.toml.

    Every blocking method raises StepError:
      - CANCELLED once `cancel` is set, cancelling a running goal;
      - TIMEOUT when the commander does not finish within `timeout_s`, also
        cancelling the goal;
      - UNAVAILABLE when the commander's server is not there;
      - PRECONDITION_FAILED when a goal of another call is still running.
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
        # Held for the whole of every call.
        self._lock = threading.Lock()
        # What the goal holding the lock does, such as "the move of
        # left_arm"; None while no goal runs. Read without the lock, to
        # refuse a second call at once.
        self._running: str | None = None
        # Guards what the methods remember, which the gaze loop reads from
        # its own thread while a call holds the call lock.
        self._state_lock = threading.Lock()
        self._last_state = ArmsState()
        self._joint_order: dict[str, tuple[str, ...]] = {}
        self._gripper_positions: dict[str, float | None] = dict.fromkeys(GRIPPER_GOAL_GROUPS)

    # -- Without waiting ----------------------------------------------------------

    def available(self) -> bool:
        """True when the commander's service and both actions are discovered. Never waits."""
        return (
            self._joint_states.service_is_ready()
            and self._set_joint_position.server_is_ready()
            and self._control_gripper.server_is_ready()
        )

    @property
    def last_state(self) -> ArmsState:
        """The arms as of the last read, for callers that must not call the commander.

        An arm becomes unknown when a goal for it is sent, and known again
        with the next read of its joints, so a moving arm is never reported
        in its old pose. ArmsState() until the first read.
        """
        with self._state_lock:
            return self._last_state

    def last_gripper_position(self, group: str) -> float | None:
        """The position last commanded to `group`, or None when unknown.

        No service reports the gap outside a close, so this is all
        get_arm_state can say about a gripper. None before the first command,
        while one runs, and after one was cancelled or timed out, since the
        fingers may then have stopped anywhere.
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
        when the commander does not know the group or answers malformed.
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
            reading = _reading(group, response)
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
        """Both arms' named poses from one read of both_arms; see joint_states()."""
        reading = self.joint_states(BOTH_ARMS_GROUP, cancel, timeout_s)
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
                what,
                deadline,
                timeout_s,
                cancel,
            )
        if response is None:
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
        """Hold the call lock; `running` names the goal, if the call sends one."""
        while True:
            _check_cancelled(cancel)
            busy = self._running
            if busy is not None:
                raise StepError(
                    "PRECONDITION_FAILED",
                    f"Cannot {action} while {busy} is running. Wait for it to finish.",
                    {"running": busy},
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StepError(
                    "TIMEOUT",
                    f"Could not {action}: another call to the commander did not finish in time.",
                )
            if self._lock.acquire(timeout=min(_POLL_S, remaining)):
                break
        self._running = running
        try:
            yield
        finally:
            self._running = None
            self._lock.release()

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
        what: str,
        deadline: float,
        timeout_s: float,
        cancel: threading.Event,
    ) -> Any | None:
        """Send `goal` and return its GetResult response, or None when it was rejected."""
        self._wait_ready(client.server_is_ready, f"{name} action", deadline, cancel)
        sent = client.send_goal_async(goal)
        try:
            handle = _wait(sent, deadline, cancel)
        except StepError as e:
            # The commander may still accept the goal after this call has
            # given up; it must not then move the arms with nobody waiting.
            sent.add_done_callback(_cancel_if_accepted)
            if e.code == "TIMEOUT":
                logger.warning(
                    f"{name} neither accepted nor rejected a goal within {timeout_s:g} s. "
                    f"{STALE_PIN_HINT}"
                )
            outcome = "it is cancelled if the commander accepts it"
            raise _ended(e, what, timeout_s, outcome) from None
        if handle is None or not handle.accepted:
            return None
        try:
            return _wait(handle.get_result_async(), deadline, cancel)
        except StepError as e:
            # wait_for_future itself, not _wait: `cancel` may already be set.
            if wait_for_future(handle.cancel_goal_async(), CANCEL_ANSWER_S) is not None:
                outcome = "the goal was cancelled"
            else:
                logger.warning(f"The commander did not answer the cancel of {what}.")
                outcome = "the commander did not confirm the cancel"
            raise _ended(e, what, timeout_s, outcome) from None

    def _forget_arms(self, group: str) -> None:
        arms = _GROUP_ARMS[group]
        with self._state_lock:
            old = self._last_state
            self._last_state = ArmsState(
                left=None if "left" in arms else old.left,
                right=None if "right" in arms else old.right,
            )

    def _remember_reading(self, group: str, reading: dict[str, float]) -> None:
        # A read of one arm's group says nothing about the other arm, which
        # keeps its last state: only this client moves the arms (ADR-0021).
        read = ArmsState.from_joints(reading, self.poses.poses)
        with self._state_lock:
            self._joint_order[group] = tuple(reading)
            old = self._last_state
            self._last_state = ArmsState(
                left=read.left if read.left is not None else old.left,
                right=read.right if read.right is not None else old.right,
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


def _cancel_if_accepted(future: Future) -> None:
    # A done callback, run by the connector's executor.
    if future.cancelled() or future.exception() is not None:
        return
    handle = future.result()
    if handle is not None and handle.accepted:
        logger.warning("The commander accepted a goal after its call gave up; cancelling it.")
        handle.cancel_goal_async()


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
    return {name: float(position) for name, position in zip(names, positions, strict=True)}


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
