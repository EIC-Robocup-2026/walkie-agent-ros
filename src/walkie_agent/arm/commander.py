"""CommanderClient: the agent's one client of walkie-ros's bimanual_commander (ADR-0021).

The arm atomics and the navigation interlock reach the arms only through it,
as the `arm` backend. It uses the names ADR-0021 allows:
/set_joint_position and /control_gripper (my_robot_interfaces actions),
/get_joint_states (a service), and, to read the grasp detection band only,
the commander's /bimanual_commander/get_parameters.

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
to run concurrently (ADR-0021, ADR-0024). So every goal and every read of
the joints holds a lock, and one while a goal runs is refused at once
rather than queued behind a move that can take 30 s. grasp_band(), which
reads only two of the commander's parameters, takes no lock.

A goal outlives the call that sent it. It is outstanding from its send
until the client sees its result, sees it rejected, or abandons it, and
until then every other call is refused, also after its own call gave up on
a cancel or a timeout. Such a goal can still move the arms: the commander
may accept it late, or still be stopping it. A cancel of it would also stop
a newer goal, since the commander's cancel stops both arms.

Abandoning a goal only lets calls through again: its end is still unknown,
so it stays unresolved and the arms are unverified (`unverified`). Motion
goals are refused, and the drive check does not fail open, until recovery
(_recover) clears that state. Recovery needs the commander's confirmation
that every unresolved goal is no longer active: its result, its rejection,
or an answer to a cancel that the goal has ended or is unknown to it. A
settled read, two rounds of reads SETTLE_S apart that agree, then shows
the arms, and every gripper that such a goal commanded, still. Every goal
is sent with an id the client makes, so recovery can cancel it by that id
also when the commander never answered its send. A goal that the
commander accepts after it was abandoned is cancelled at once, is
unresolved again and makes the arms unverified again, and a newer arm goal
that the cancel may stop fails (_on_late_answer).
"""

import functools
import logging
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from my_robot_interfaces.action import ControlGripper, SetJointPosition
from my_robot_interfaces.srv import GetJointStates
from rai.communication.ros2 import ROS2Connector
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.task import Future
from unique_identifier_msgs.msg import UUID

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

# GripperResult and Recovery live with the capabilities, which return them
# (ADR-0052); they are re-exported here for the imports that name this
# module.
from walkie_agent.capabilities import GripperResult, Recovery
from walkie_agent.toolbox.types import StepError
from walkie_agent.tools.waiters import wait_for_future

logger = logging.getLogger(__name__)

# Absolute, so the names do not depend on the agent node's namespace. The
# commander's node runs in the root namespace.
SET_JOINT_POSITION_ACTION = "/set_joint_position"
CONTROL_GRIPPER_ACTION = "/control_gripper"
GET_JOINT_STATES_SERVICE = "/get_joint_states"
# The commander node's standard parameter service, read for its grasp
# detection band and nothing else (ADR-0021).
GET_PARAMETERS_SERVICE = "/bimanual_commander/get_parameters"
# The parameters a close's `grasped` is judged by, lowest first, in metres.
GRASP_BAND_PARAMETERS = ("grasp_detect_min", "grasp_detect_max")

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
# and the goal stays outstanding until it comes. Recovery waits as long
# after each cancel it sends, for the answer and, after an accepted cancel,
# for the result (_await_end).
CANCEL_ANSWER_S = 1.0
# How long after its call gave up a goal may stay outstanding before the
# client abandons it and takes calls again. It is not a bound on the motion:
# MoveIt lets a goal execute for twice its planned duration plus 60 s
# (move_group.yaml of walkie-ros's openarm_bimanual_moveit_config). The
# result normally comes within seconds: a cancelled move stops at once, and
# a cancelled gripper command finishes its ramp and the commander's 5 s wait
# for the controller. But a commander that restarted keeps its action
# servers discovered, and the old goal's result never comes, so discovery
# alone cannot end the wait. Abandoning only lets calls through: the arms
# are then unverified, so motion goals are refused and the robot does not
# drive until recovery clears that state (_recover).
ABANDON_AFTER_S = 60.0
# How long a goal's action server must stay undiscovered, as the checks of
# _outstanding() see it, before the goal is abandoned. Not at the first
# check: discovery can drop for a moment while the commander still runs the
# goal.
ABSENT_ABANDON_S = 5.0
# A settled read is two rounds of reads SETTLE_S apart: in each, a complete
# read of both arms, whose joints must all agree within SETTLE_TOLERANCE_RAD,
# and a read of every gripper group that made the arms unverified, whose
# finger must agree within SETTLE_TOLERANCE_M. It only supplements the
# commander's confirmation that the earlier goals ended (_confirm_ended). It
# cannot see a joint slower than about 0.03 rad/s or a finger slower than
# about 3 mm/s, their ratios, nor a goal that has not started to move. That
# matters only for motion the commander does not know about, such as a
# trajectory that move_group still executes after the commander restarted.
SETTLE_S = 0.3
SETTLE_TOLERANCE_RAD = 0.01
SETTLE_TOLERANCE_M = 0.001
# How long a call waits for the commander's server to be discovered. Not
# zero, so a call soon after the agent starts does not fail on discovery;
# short, so a missing commander costs little. available() never waits.
READY_WAIT_S = 1.0

# How often a wait looks at the cancel event and the clock, as
# ToolContext.wait() does.
_POLL_S = 0.05

# Why check_for_driving() could not read the arms (DriveCheck.unread): the
# /get_joint_states service is not discovered, its read did not answer in
# time, or the commander answered that it could not read the joints.
NOT_FOUND = "not found"
NO_ANSWER = "no answer"
NO_READING = "no reading"

# The goals the agent may send. The *_lift groups also move the lift, which
# the agent never commands (ADR-0026).
ARM_GOAL_GROUPS = frozenset((*ARM_GROUPS.values(), BOTH_ARMS_GROUP))
GRIPPER_GOAL_GROUPS = frozenset(GRIPPER_GROUPS.values())
READ_GROUPS = ARM_GOAL_GROUPS | GRIPPER_GOAL_GROUPS
# The arms each arm group moves.
_GROUP_ARMS = {ARM_GROUPS[arm]: (arm,) for arm in ARMS} | {BOTH_ARMS_GROUP: ARMS}
# Each gripper group's finger joint, as FINGER_JOINTS in tools/atomic/_arm.py
# names it: the one joint of the group, whose position is the travel of one
# finger, in metres.
_FINGER_JOINTS = {
    GRIPPER_GROUPS["left"]: "openarm_left_finger_joint1",
    GRIPPER_GROUPS["right"]: "openarm_right_finger_joint1",
}

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
# The cancel outcome in StepError.data of a goal call that gave up before the
# commander answered its send, so no cancel was sent.
_PENDING = "pending"

# How the commander showed that an unresolved goal is no longer active
# (_resolution, Recovery.goals), in words.
_RESOLUTIONS = {
    "result": "its result came",
    "rejected": "the commander rejected it",
    "ended": "the commander answered its cancel that it had already ended",
    "unknown": "the commander does not know it, as after a restart",
}


@dataclass(frozen=True)
class DriveCheck:
    """What check_for_driving() found: both arms, or why they were not read."""

    # From a complete reading of both arms; None when they were not read.
    state: ArmsState | None
    # NOT_FOUND, NO_ANSWER or NO_READING when `state` is None, else None.
    unread: str | None = None


@dataclass(frozen=True)
class ClientSnapshot:
    """What the client knows at one moment (snapshot())."""

    # last_state: the arms as of the last read.
    state: ArmsState
    # last_recovery: what the recovery that last cleared the unverified
    # state checked, else None.
    recovery: Recovery | None
    # unverified_count: how many times the arms became unverified so far.
    count: int
    # unverified: why the arms are unverified, or None.
    unverified: str | None


@dataclass(eq=False)
class _Goal:
    """A goal sent to the commander, kept until the client knows that it ended."""

    # Such as "the move of left_arm" or "the left_gripper command".
    what: str
    # The group it was sent for.
    group: str
    client: ActionClient
    # The id it was sent with, made by the client, so that it can be
    # cancelled by its id also when the commander never answered its send.
    goal_id: UUID
    # The send_goal_async future.
    sent: Future
    # The get_result_async future, once the goal was accepted.
    result: Future | None = None
    # The get_result_async futures that recovery asked for besides `result`,
    # whose answer may have been lost (_request_result).
    more_results: list[Future] = field(default_factory=list)
    # time.monotonic() when the call that sent it gave up on it; None while
    # that call still waits for it.
    gave_up_at: float | None = None
    # time.monotonic() when a check first found its action server
    # undiscovered since it was last found discovered; None while it is
    # discovered.
    absent_since: float | None = None
    # Why something other than its own call may have stopped it part way,
    # such as the cancel of an older goal, which stops both arms; else None.
    disturbed: str | None = None
    # For a gripper goal, its number among the client's commands for its
    # group (GripperResult.command); else None.
    command: int | None = None
    # The cancel_goal_async futures of every cancel sent for it, oldest
    # first, and time.monotonic() when the last one was sent.
    cancels: list[Future] = field(default_factory=list)
    cancel_sent_at: float | None = None
    # The index in `cancels` of the first cancel sent since it was abandoned
    # or last accepted late. Only the answers from there on can show that
    # it ended (_resolution).
    evidence_from: int = 0
    # How the commander showed that it is no longer active, once it was
    # resolved after it was abandoned (_resolution); else None.
    resolved: str | None = None

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
        not ended; and for a goal while the arms are unverified
        (`unverified`).
    A group the agent may not use, or a value that is not a finite number,
    raises ValueError: that is a bug in reviewed code, not something for the
    LLM to retry.
    """

    def __init__(self, connector: ROS2Connector, poses: ArmPoses | None = None) -> None:
        self.poses = poses if poses is not None else load_arm_poses()
        node = connector.node
        # Created here, once: rclpy entities must not be created per call.
        self._joint_states = node.create_client(GetJointStates, GET_JOINT_STATES_SERVICE)
        self._parameters = node.create_client(GetParameters, GET_PARAMETERS_SERVICE)
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
        # Held for short steps only, never while waiting.
        self._state_lock = threading.Lock()
        self._last_state = ArmsState()
        self._joint_order: dict[str, tuple[str, ...]] = {}
        self._gripper_positions: dict[str, float | None] = dict.fromkeys(GRIPPER_GOAL_GROUPS)
        # The goal whose end the client does not know yet, or None. One at
        # most: no call starts while it is set.
        self._goal: _Goal | None = None
        # Why the arms are unverified, or None (`unverified`). The count goes
        # up each time it is set, so a read that began before does not clear
        # it or keep a state from before it.
        self._unverified: str | None = None
        self._unverified_count = 0
        # The groups of the goals that made the arms unverified since they
        # were last verified, so a settled read checks every gripper among
        # them.
        self._unverified_groups: set[str] = set()
        # The abandoned goals that the commander has not shown to be no
        # longer active, oldest first. The arms stay unverified while any is
        # here, so no newer goal starts while one may still be active: it
        # may still move, and a cancel of it stops both arms.
        self._unresolved: list[_Goal] = []
        # The goals resolved since the arms were last verified, for
        # Recovery.goals.
        self._resolved: list[_Goal] = []
        self._last_recovery: Recovery | None = None
        # The commands sent to each gripper group so far, counting again one
        # that the commander accepts after its call gave up, so that a read
        # can tell whether a newer command came (joint_states_since).
        self._gripper_commands = dict.fromkeys(GRIPPER_GOAL_GROUPS, 0)

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
        commander is no longer discovered. A goal its call gave up on is
        abandoned once its action server stays undiscovered for
        ABSENT_ABANDON_S, or after ABANDON_AFTER_S (_outstanding); the arms
        are then unverified, which `unverified` tells, not busy().
        """
        running = self._running
        if running is not None:
            return running
        goal = self._outstanding()
        return None if goal is None else f"{goal.what} from an earlier call"

    @property
    def unverified(self) -> str | None:
        """Why the arms are unverified, or None. Never waits.

        Set when a goal is abandoned (_outstanding) and when the commander
        accepts a goal after it was abandoned, since either may still move
        an arm or a gripper; what that goal moves is then unknown in
        last_state. While it is set, every goal is refused and the drive
        check does not fail open. Recovery clears it once the commander has
        shown every such goal to be no longer active and a settled read
        agrees (_recover): read_arms() and check_for_driving() make one
        while it is set. Such as "the move of left_arm from an earlier call
        was given up on before its end was known".
        """
        with self._state_lock:
            return self._unverified

    @property
    def unverified_count(self) -> int:
        """How many times the arms became unverified so far. Never waits.

        Read before `unverified`, or with it in one snapshot(), and before a
        call that may recover: when `unverified` was set then, the call
        cleared the state if last_recovery.count equals this count
        afterwards.
        """
        with self._state_lock:
            return self._unverified_count

    @property
    def last_recovery(self) -> Recovery | None:
        """What the last recovery that cleared the unverified state checked, else None."""
        with self._state_lock:
            return self._last_recovery

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

    def snapshot(self) -> ClientSnapshot:
        """last_state, last_recovery, unverified_count and unverified, read together. Never waits.

        For an answer whose parts must agree. When the commander accepts an
        arm's goal late, the arms become unverified and that arm unknown in
        one step, and a recovery clears the state and keeps the arms it read
        in one step. Separate reads can see one half of such a step and miss
        the other, so an answer built from them could say that the arms have
        to be checked again and still give that arm's pose. A snapshot is
        one moment: a goal accepted right after it is not in it, and the
        next snapshot shows it.
        """
        with self._state_lock:
            return ClientSnapshot(
                self._last_state, self._last_recovery, self._unverified_count, self._unverified
            )

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
        is unknown too. One read does not clear the unverified state; a
        settled read (read_arms) does.
        """
        _check_group(group, READ_GROUPS)
        deadline = _deadline(timeout_s)
        with self._exclusive(f"read the joints of {group}", deadline, cancel):
            count = self._unverified_mark()
            reading = self._read(group, deadline, timeout_s, cancel)
            self._remember_reading(group, reading, count)
            return reading

    def joint_states_since(
        self,
        group: str,
        command: int,
        cancel: threading.Event,
        timeout_s: float = READ_TIMEOUT_S,
    ) -> dict[str, float]:
        """The joints of gripper `group`, read once while `command` is its latest command.

        `command` is GripperResult.command. Under the call lock, which no
        goal holds or waits on then (_exclusive refuses while one runs or is
        outstanding), it checks that no newer command for the group came,
        also none that the commander accepted late, reads, and checks again.
        Raises StepError as joint_states() does, and PRECONDITION_FAILED when
        a newer command came, since a read then would not show where
        `command` left the fingers.
        """
        _check_group(group, GRIPPER_GOAL_GROUPS)
        deadline = _deadline(timeout_s)
        with self._exclusive(f"read the joints of {group}", deadline, cancel):
            self._check_command(group, command)
            count = self._unverified_mark()
            reading = self._read(group, deadline, timeout_s, cancel)
            self._check_command(group, command)
            self._remember_reading(group, reading, count)
            return reading

    def grasp_band(
        self, cancel: threading.Event, timeout_s: float = READ_TIMEOUT_S
    ) -> tuple[float, float]:
        """The commander's grasp detection band now, (grasp_detect_min, grasp_detect_max), in m.

        The band a close's `grasped` is judged by. The commander reads it
        afresh on every close, so a band changed while it runs, as the HRI
        scenario changes it to hang a bag, is the one returned. Read once,
        from the commander's parameter service, without the call lock: it
        reads no joints and sends no goal, so it cannot fight one. Raises
        StepError CANCELLED once `cancel` is set, UNAVAILABLE when the
        service is not discovered, or the commander lacks either parameter
        or holds one that is not a finite number, and TIMEOUT when it does
        not answer within `timeout_s`.
        """
        deadline = _deadline(timeout_s)
        self._wait_ready(
            self._parameters.service_is_ready, f"{GET_PARAMETERS_SERVICE} service", deadline, cancel
        )
        future = self._parameters.call_async(
            GetParameters.Request(names=list(GRASP_BAND_PARAMETERS))
        )
        try:
            response = _wait(future, deadline, cancel)
        except StepError as e:
            # The answer would otherwise be kept for a caller that left.
            self._parameters.remove_pending_request(future)
            if e.code == "CANCELLED":
                raise StepError(
                    "CANCELLED",
                    "A stop or a reset interrupted the read of the commander's grasp detection "
                    "band.",
                ) from None
            raise StepError(
                "TIMEOUT",
                f"The commander's {GET_PARAMETERS_SERVICE} did not answer within {timeout_s:g} s.",
            ) from None
        return _band(response)

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
        """Both arms' named poses from a read of both_arms; see joint_states().

        Also raises StepError INTERNAL when the reading lacks a joint of
        either arm, which is then unknown. While the arms are unverified
        (`unverified`), it recovers instead (_recover), all within
        `timeout_s`: it clears that state, or raises as _recover() does.
        """
        deadline = _deadline(timeout_s)
        with self._exclusive(f"read the joints of {BOTH_ARMS_GROUP}", deadline, cancel):
            reading = self._read_both(deadline, timeout_s, cancel)
        return ArmsState.from_joints(reading, self.poses.poses)

    def check_for_driving(
        self, cancel: threading.Event, timeout_s: float = READ_TIMEOUT_S
    ) -> DriveCheck:
        """Read both arms for the drive check (ADR-0025), all under the call lock.

        So no goal can start between its steps. Raises StepError
        PRECONDITION_FAILED while a goal runs or is outstanding, and TIMEOUT
        when another call keeps the lock past `timeout_s`. Returns the arms'
        state, or no state and why: the /get_joint_states service is not
        discovered (NOT_FOUND; a missing action does not stop the arms being
        read), the read does not answer within `timeout_s` (NO_ANSWER), or
        the commander cannot read the joints (NO_READING). The caller may
        then drive: failing open keeps navigation working while the arm
        stack is down. Not while the arms are unverified (`unverified`): it
        then recovers as read_arms() does, within the same `timeout_s`, and
        raises PRECONDITION_FAILED instead of failing open, as well as when
        recovery refuses. INTERNAL as read_arms() raises it.
        """
        deadline = _deadline(timeout_s)
        with self._exclusive("check the arms for driving", deadline, cancel):
            if not self._joint_states.service_is_ready():
                self._refuse_unverified(
                    f"The commander's {GET_JOINT_STATES_SERVICE} service is not discovered. "
                    f"{COMMANDER_HINT}"
                )
                return DriveCheck(None, NOT_FOUND)
            try:
                reading = self._read_both(deadline, timeout_s, cancel)
            except StepError as e:
                if e.code not in ("TIMEOUT", "UNAVAILABLE"):
                    raise
                self._refuse_unverified(e.message)
                if e.code == "TIMEOUT":
                    return DriveCheck(None, NO_ANSWER)
                # _wait_ready's, when the service went away since the check.
                return DriveCheck(None, NOT_FOUND if "name" in e.data else NO_READING)
        return DriveCheck(ArmsState.from_joints(reading, self.poses.poses))

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
          - SKILL_FAILED for "Planning Failed" (nothing moved), for
            "Execution Failed" (the motion stopped part way), and when the
            cancel of an older goal that the commander accepted late may
            have stopped it (_on_late_answer);
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
            _, response = self._run_goal(
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
            record, response = self._run_goal(
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
            command=record.command,
        )

    def destroy(self) -> None:
        """Destroy the clients. Node.destroy_node() does not destroy action clients."""
        self._set_joint_position.destroy()
        self._control_gripper.destroy()
        self._node.destroy_client(self._joint_states)
        self._node.destroy_client(self._parameters)

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

        Refuses at once while a goal is outstanding (_Goal), and a call that
        sends a goal also while the arms are unverified.
        """
        sends = running is not None
        while True:
            _check_cancelled(cancel)
            self._check_free(action, sends)
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
            self._check_free(action, sends)
            self._running = running
            yield
        finally:
            self._running = None
            self._lock.release()

    def _check_free(self, action: str, sends: bool) -> None:
        """Raises StepError PRECONDITION_FAILED while a goal is outstanding.

        With `sends`, for a call that sends a goal, also while the arms are
        unverified.
        """
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
        if sends:
            cause = self.unverified
            if cause is not None:
                raise unverified_refusal(action, cause)

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

    def _read(
        self, group: str, deadline: float, timeout_s: float, cancel: threading.Event
    ) -> dict[str, float]:
        """One read of `group`, with the call lock held; see joint_states().

        Remembers nothing, except that the arms of the group are unknown
        after an INTERNAL.
        """
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
            return _reading(group, response)
        except StepError as e:
            if e.code == "INTERNAL":
                # The commander answered, but not with positions that can
                # be trusted, so the arms may be anywhere.
                self._forget_arms(group)
            raise

    def _read_both(
        self, deadline: float, timeout_s: float, cancel: threading.Event
    ) -> dict[str, float]:
        """A complete reading of both arms, with the call lock held.

        One read while the arms are verified; recovery while they are
        unverified, or once they become so during the read.
        """
        with self._state_lock:
            unverified, count = self._unverified, self._unverified_count
        if unverified is None:
            reading = self._read(BOTH_ARMS_GROUP, deadline, timeout_s, cancel)
            if self._remember_reading(BOTH_ARMS_GROUP, reading, count):
                self._check_complete(reading)
                return reading
            # A goal given up on earlier was accepted late during the read.
        return self._recover(deadline, timeout_s, cancel)

    def _recover(
        self, deadline: float, timeout_s: float, cancel: threading.Event
    ) -> dict[str, float]:
        """Clear the unverified state, with the call lock held, all before `deadline`.

        First the commander must show every unresolved goal to be no longer
        active (_confirm_ended), then a settled read must agree
        (_settled_read), and the arms must not have become unverified again
        since this began. Returns the settled read's second reading of both
        arms. Raises StepError as those two do, with `timeout_s` the time
        allowed for the whole of it, which the messages give.
        """
        count = self._unverified_mark()
        self._confirm_ended(deadline, cancel)
        return self._settled_read(count, deadline, timeout_s, cancel)

    def _confirm_ended(self, deadline: float, cancel: threading.Event) -> None:
        """Resolve every unresolved goal that the commander shows to be no longer active.

        With the call lock held. Each goal is resolved by its result or
        rejection, else sent a cancel by its id, unless one sent since it
        was abandoned waits for an answer and was sent less than
        CANCEL_ANSWER_S ago. A cancel to a commander that died is never
        answered, so an older one does not stop a new one being sent. Then
        each answer is waited for, and after an accepted cancel the result,
        up to CANCEL_ANSWER_S after that cancel was sent and before
        `deadline` (_await_end). Raises StepError CANCELLED, and
        PRECONDITION_FAILED with the goals in "unresolved" when any is left
        unresolved (_unconfirmed).
        """
        with self._state_lock:
            goals = list(self._unresolved)
        # Every cancel is sent before any wait, so the waits overlap.
        for goal in goals:
            self._resolve_or_cancel(goal)
        for goal in goals:
            self._await_end(goal, deadline, cancel)
        # Under one hold, so that the error describes the answers that left
        # each goal unresolved.
        with self._state_lock:
            resolved = [(goal, self._resolve(goal)) for goal in goals]
            left = [goal for goal in goals if goal in self._unresolved]
            error = _unconfirmed(left) if left else None
        for goal, how in resolved:
            if how is not None:
                _log_resolved(goal, how)
        if error is not None:
            raise error

    def _resolve_or_cancel(self, goal: _Goal) -> None:
        """Resolve `goal` if it can be now; else send it a cancel unless one is on its way."""
        now = time.monotonic()
        with self._state_lock:
            how = self._resolve(goal)
            latest = goal.cancels[-1] if len(goal.cancels) > goal.evidence_from else None
            waiting = (
                latest is not None
                and _cancel_state(latest, goal.goal_id) is None
                and now - goal.cancel_sent_at < CANCEL_ANSWER_S
            )
            send = how is None and goal in self._unresolved and not waiting
        if how is not None:
            _log_resolved(goal, how)
        if send:
            self._send_cancel(goal, _handle_for(goal))

    def _await_end(self, goal: _Goal, deadline: float, cancel: threading.Event) -> None:
        """Wait until `goal` is resolved, or its latest cancel shows it may still be active.

        Up to CANCEL_ANSWER_S after that cancel was sent, and before
        `deadline`. A refused or failed cancel ends the wait at once. After
        an accepted cancel, or an answer that the goal ended while an
        earlier cancel may have been accepted, the commander may still be
        stopping it, so the wait goes on for its result, which is asked for
        again by its id, once per wait (_request_result).
        """
        with self._state_lock:
            sent_at = goal.cancel_sent_at
        until = min(deadline, (time.monotonic() if sent_at is None else sent_at) + CANCEL_ANSWER_S)
        asked: Future | None = None
        while True:
            with self._state_lock:
                how = self._resolve(goal)
                unresolved = goal in self._unresolved
                latest = goal.cancels[-1] if goal.cancels else None
                answer = "no answer" if latest is None else _cancel_state(latest, goal.goal_id)
            if how is not None:
                _log_resolved(goal, how)
            # A refused or failed cancel is not waited on.
            if not unresolved or answer not in (None, "accepted", "ended"):
                return
            if answer is None:
                future = latest
            else:
                # Accepted, or ended while an earlier cancel may have been
                # accepted: only the result shows that it ended.
                if asked is None:
                    asked = self._request_result(goal)
                future = asked
            if cancel.is_set():
                raise StepError(
                    "CANCELLED", "A stop or a reset interrupted the check of the arms."
                )
            remaining = until - time.monotonic()
            if remaining <= 0:
                return
            wait_for_future(future, min(_POLL_S, remaining))

    def _request_result(self, goal: _Goal) -> Future:
        """Ask the commander for the result of `goal` by its id; its end resolves the goal.

        Also when the result was asked for before: that answer may have been
        lost, as when discovery dropped while the goal ended, and the
        commander answers again from the result it keeps, or that it does
        not know the goal.
        """
        result = _handle_for(goal).get_result_async()
        with self._state_lock:
            if goal.result is None:
                goal.result = result
            else:
                goal.more_results.append(result)
        # Not under the state lock, as in _wait_goal().
        result.add_done_callback(functools.partial(self._end_goal, goal))
        return result

    def _settled_read(
        self, count: int, deadline: float, timeout_s: float, cancel: threading.Event
    ) -> dict[str, float]:
        """Two rounds of reads SETTLE_S apart, with the call lock held, before `deadline`.

        Each round reads both arms, completely, and every gripper group that
        made the arms unverified. When every arm joint agrees within
        SETTLE_TOLERANCE_RAD, every such finger within SETTLE_TOLERANCE_M,
        no goal is unresolved, and the arms did not become unverified again
        since `count` (_unverified_mark), it keeps the second reading of
        both arms and clears the unverified state. It only supplements the
        commander's confirmation (_confirm_ended): see SETTLE_S for what it
        cannot see. Raises StepError CANCELLED, INTERNAL as _read() raises
        it and when a reading lacks a joint of either arm or a finger joint,
        UNAVAILABLE as _read() raises it, TIMEOUT when the reads do not fit
        before `deadline`, and PRECONDITION_FAILED when a joint moved
        between the rounds or the arms became unverified again; after an
        arm moved or that, both arms are unknown.
        """
        with self._state_lock:
            grippers = tuple(sorted(self._unverified_groups & GRIPPER_GOAL_GROUPS))
        first, first_fingers = self._read_round(grippers, deadline, timeout_s, cancel)
        if deadline - time.monotonic() <= SETTLE_S:
            raise _no_time_to_settle(timeout_s)
        if cancel.wait(SETTLE_S):
            raise StepError("CANCELLED", "A stop or a reset interrupted the check of the arms.")
        second, second_fingers = self._read_round(grippers, deadline, timeout_s, cancel)
        joint, moved = max(
            ((name, abs(value - first[name])) for name, value in second.items() if name in first),
            key=lambda change: change[1],
            default=("", 0.0),
        )
        if moved > SETTLE_TOLERANCE_RAD:
            self._forget_arms(BOTH_ARMS_GROUP)
            raise StepError(
                "PRECONDITION_FAILED",
                f"An arm is still moving: {joint} moved {moved:.3f} rad between two reads "
                f"{SETTLE_S:g} s apart. Wait a few seconds, then try again.",
                {"joint": joint, "moved_rad": moved},
            )
        for group in grippers:
            joint, moved = _FINGER_JOINTS[group], abs(second_fingers[group] - first_fingers[group])
            if moved > SETTLE_TOLERANCE_M:
                raise StepError(
                    "PRECONDITION_FAILED",
                    f"A gripper is still moving: {joint} moved {moved:.4f} m between two reads "
                    f"{SETTLE_S:g} s apart. Wait a few seconds, then try again.",
                    {"joint": joint, "moved_m": moved},
                )
        if not self._remember_reading(BOTH_ARMS_GROUP, second, count, verified=grippers):
            self._forget_arms(BOTH_ARMS_GROUP)
            cause = self.unverified
            raise StepError(
                "PRECONDITION_FAILED",
                f"While the arms were being checked, {cause}. Wait a few seconds, then try "
                "again.",
                {"unverified": cause},
            )
        checked = " and ".join((BOTH_ARMS_GROUP, *grippers))
        logger.info(
            f"The arms are verified: the commander showed every earlier goal to be no longer "
            f"active, and two reads of {checked} {SETTLE_S:g} s apart agree."
        )
        return second

    def _read_round(
        self, grippers: Sequence[str], deadline: float, timeout_s: float, cancel: threading.Event
    ) -> tuple[dict[str, float], dict[str, float]]:
        """One round of a settled read: both arms, then the finger of each of `grippers`.

        Returns the reading of both arms, and each finger's travel by group.
        """
        arms = self._read_complete(deadline, timeout_s, cancel)
        fingers = {
            group: self._read_finger(group, deadline, timeout_s, cancel) for group in grippers
        }
        return arms, fingers

    def _read_complete(
        self, deadline: float, timeout_s: float, cancel: threading.Event
    ) -> dict[str, float]:
        """One read of both arms for a settled read; INTERNAL when it lacks a joint."""
        reading = self._read_in_time(BOTH_ARMS_GROUP, deadline, timeout_s, cancel)
        try:
            self._check_complete(reading)
        except StepError:
            with self._state_lock:
                # As _remember_reading() does for an incomplete reading.
                self._joint_order.pop(BOTH_ARMS_GROUP, None)
                self._last_state = _without(self._last_state, BOTH_ARMS_GROUP)
            raise
        return reading

    def _read_finger(
        self, group: str, deadline: float, timeout_s: float, cancel: threading.Event
    ) -> float:
        """One read of gripper `group` for a settled read: its finger's travel, in metres.

        INTERNAL when the reading lacks the finger joint, as read_finger()
        in tools/atomic/_arm.py raises it.
        """
        reading = self._read_in_time(group, deadline, timeout_s, cancel)
        joint = _FINGER_JOINTS[group]
        if joint not in reading:
            raise StepError(
                "INTERNAL",
                f"The commander's reading of {group} lacks {joint}.",
                {"group": group, "joints": list(reading)},
            )
        return reading[joint]

    def _read_in_time(
        self, group: str, deadline: float, timeout_s: float, cancel: threading.Event
    ) -> dict[str, float]:
        """_read() for a settled read, before `deadline`.

        Its TIMEOUT says that the whole check had `timeout_s`, since the
        read itself had only what was left of it.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _no_time_to_settle(timeout_s)
        try:
            return self._read(group, deadline, round(remaining, 2), cancel)
        except StepError as e:
            if e.code != "TIMEOUT":
                raise
            raise StepError(
                "TIMEOUT",
                f"The arms could not be checked within {timeout_s:g} s: the commander's "
                f"{GET_JOINT_STATES_SERVICE} did not answer in time. {COMMANDER_HINT}",
            ) from None

    def _check_complete(self, reading: Mapping[str, float]) -> None:
        """Raises StepError INTERNAL when `reading` lacks a joint of either arm."""
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

    def _check_command(self, group: str, command: int) -> None:
        """Raises StepError PRECONDITION_FAILED once a command newer than `command` came."""
        with self._state_lock:
            latest = self._gripper_commands[group]
        if latest != command:
            raise StepError(
                "PRECONDITION_FAILED",
                f"Another {group} command came after this one, so a read now would not show "
                "where this one left the fingers.",
                {"group": group, "command": command},
            )

    def _refuse_unverified(self, why: str) -> None:
        """Raises StepError PRECONDITION_FAILED while the arms are unverified.

        For the drive check, which must not fail open then. `why` says why
        the arms cannot be read.
        """
        cause = self.unverified
        if cause is not None:
            raise StepError(
                "PRECONDITION_FAILED",
                f"An arm or a gripper may still be moving, as {cause}, and the arms cannot be "
                f"checked now. {_full_stop(why)} Try again in a few seconds; if it fails again, "
                "tell the person that the arms cannot be checked.",
                {"unverified": cause},
            )

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
    ) -> tuple[_Goal, Any | None]:
        """Send `goal`; return its record and its GetResult response, None when rejected.

        The goal is outstanding from its send until its end is known, also
        after this call gives up on a cancel or a timeout (_Goal).
        """
        self._wait_ready(client.server_is_ready, f"{name} action", deadline, cancel)
        # The check, the send and the record under one hold of the state
        # lock, which _on_late_answer() takes too: an older goal that the
        # commander accepts late either makes the arms unverified before
        # this goal is sent, which refuses it, or finds its record, which it
        # marks disturbed. send_goal_async() calls no callback of ours.
        with self._state_lock:
            cause = self._unverified
            if cause is None:
                # The id made here, so that the goal can be cancelled by it
                # whether or not the commander answers the send.
                goal_id = UUID(uuid=list(uuid.uuid4().bytes))
                record = _Goal(
                    what, group, client, goal_id, client.send_goal_async(goal, goal_uuid=goal_id)
                )
                if group in self._gripper_commands:
                    # Counted with the send, under the call lock, so no read
                    # that checks the count runs in between
                    # (joint_states_since), and a call that sends nothing
                    # counts nothing.
                    self._gripper_commands[group] += 1
                    record.command = self._gripper_commands[group]
                self._goal = record
        if cause is not None:
            raise unverified_refusal(f"move {group}", cause)
        try:
            return record, self._wait_goal(record, name, deadline, timeout_s, cancel)
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
            raise _ended(e, goal.what, timeout_s, _PENDING, outcome) from None
        if handle is None or not handle.accepted:
            return None
        result = handle.get_result_async()
        with self._state_lock:
            goal.result = result
        # Not under the state lock: on a future that is already done, the
        # callback can run at once, in this thread.
        result.add_done_callback(functools.partial(self._end_goal, goal))
        try:
            response = _wait(result, deadline, cancel)
        except StepError as e:
            raise _ended(e, goal.what, timeout_s, *self._cancel(handle, goal)) from None
        with self._state_lock:
            disturbed = goal.disturbed
        if disturbed is not None:
            self._forget_arms(goal.group)
            raise StepError(
                "SKILL_FAILED",
                f"{_capital(goal.what)} may have been stopped part way: {disturbed}. Call "
                "get_arm_state to see where the arms are.",
                {"group": goal.group, "disturbed": disturbed},
            )
        return response

    def _cancel(self, handle: Any, goal: _Goal) -> tuple[str, str]:
        """Ask the commander to cancel the goal of `handle`.

        Returns the _CANCEL_OUTCOMES key of its answer, and what that means
        in words.
        """
        # wait_for_future itself, not _wait: `cancel` may already be set.
        answer = wait_for_future(self._send_cancel(goal, handle), CANCEL_ANSWER_S)
        outcome = _cancel_outcome(answer, handle.goal_id)
        words = _CANCEL_OUTCOMES[outcome][1 if goal.group in GRIPPER_GOAL_GROUPS else 0]
        if outcome not in ("accepted", "ended"):
            logger.warning(f"Cancel of {goal.what}: {words}.")
        return outcome, words

    def _on_late_answer(self, goal: _Goal, sent: Future) -> None:
        """The commander answered a goal after its call gave up waiting for that answer.

        A done callback, run by the connector's executor, so it must not
        raise, block or take the call lock. An accepted goal is cancelled at
        once, and the answer to that cancel is applied, and logged, when it
        comes (_on_cancel_answer). A goal that is still outstanding stays so
        until its result comes. A goal that was abandoned is unresolved
        again, also when an answer to a cancel sent before the commander
        accepted it had resolved it, and makes the arms unverified again;
        a newer arm goal that runs or is outstanding is marked disturbed
        before the cancel is sent: the commander's cancel of a joint goal
        stops both arms. A joint cancel does not stop a gripper, and a
        gripper finishes its motion whatever the cancel, so a gripper goal
        is never disturbed. An answer that is not an acceptance forgets the
        goal, or resolves it once it was abandoned.
        """
        try:
            handle = None if sent.cancelled() or sent.exception() is not None else sent.result()
            if handle is None or not handle.accepted:
                with self._state_lock:
                    current = self._goal is goal
                    if current:
                        self._goal = None
                    how = self._resolve(goal)
                if how is not None:
                    _log_resolved(goal, how)
                elif not current:
                    logger.debug(f"The commander did not accept {goal.what}, which was abandoned.")
                return
            arm_goal = goal.group in ARM_GOAL_GROUPS
            disturbed = None
            with self._state_lock:
                abandoned = self._goal is not goal
                if abandoned:
                    if goal not in self._unresolved:
                        if goal in self._resolved:
                            self._resolved.remove(goal)
                        goal.resolved = None
                        self._unresolved.append(goal)
                    self._set_unverified(_accepted_late(goal), goal.group)
                    newer = self._goal
                    if (
                        arm_goal
                        and newer is not None
                        and newer.group in ARM_GOAL_GROUPS
                        and not newer.ended()
                    ):
                        newer.disturbed = (
                            f"the commander accepted {goal.what} from an earlier call late and "
                            "was asked to cancel it, and a cancel it accepts stops both arms"
                        )
                        disturbed = newer.what
                if not arm_goal:
                    # The fingers make this motion now, after any newer command.
                    self._gripper_commands[goal.group] += 1
                # The cancel and the result request in the same hold: no
                # answer from before the acceptance can resolve the goal any
                # more, and the answers to these two can.
                goal.evidence_from = len(goal.cancels)
                answer = handle.cancel_goal_async()
                goal.cancels.append(answer)
                goal.cancel_sent_at = time.monotonic()
                result = handle.get_result_async()
                goal.result = result
            logger.warning(
                f"The commander accepted {goal.what} after its call gave up; cancelling it."
            )
            if abandoned:
                logger.warning(
                    f"{_capital(goal.what)} had been abandoned, so the arms are unverified "
                    "until the commander shows that it is no longer active and a settled read "
                    "shows the arms still."
                )
            if disturbed is not None:
                logger.warning(
                    f"{_capital(disturbed)} may be stopped part way by the cancel of {goal.what}."
                )
            # Not under the state lock, as in _wait_goal().
            answer.add_done_callback(functools.partial(self._on_cancel_answer, goal, late=True))
            result.add_done_callback(functools.partial(self._end_goal, goal))
        except Exception:
            logger.exception(f"Could not handle the late answer to {goal.what}.")

    def _on_cancel_answer(self, goal: _Goal, future: Future, late: bool = False) -> None:
        """Apply the commander's answer to a cancel of `goal`, which may resolve it.

        A done callback, so it must not raise, block or take the call lock.
        With `late`, for the cancel of a goal that the commander accepted
        late, the answer is logged too.
        """
        try:
            with self._state_lock:
                how = self._resolve(goal)
            if how is not None:
                _log_resolved(goal, how)
            if not late:
                return
            outcome = _cancel_state(future, goal.goal_id) or "no answer"
            words = _CANCEL_OUTCOMES[outcome][1 if goal.group in GRIPPER_GOAL_GROUPS else 0]
            message = f"Cancel of {goal.what}, which the commander accepted late: {words}."
            if outcome in ("accepted", "ended"):
                logger.info(message)
            else:
                logger.warning(message)
        except Exception:
            logger.exception(f"Could not check the cancel of {goal.what}.")

    def _end_goal(self, goal: _Goal, _future: Future | None = None) -> None:
        """Forget `goal`, which has ended, or resolve it once it was abandoned.

        Also a done callback, so it must not raise.
        """
        try:
            with self._state_lock:
                if self._goal is goal:
                    self._goal = None
                how = self._resolve(goal)
            if how is not None:
                _log_resolved(goal, how)
        except Exception:
            logger.exception(f"Could not handle the end of {goal.what}.")

    def _send_cancel(self, goal: _Goal, handle: ClientGoalHandle) -> Future:
        """Ask the commander to cancel `goal`; its answer may resolve it (_on_cancel_answer)."""
        with self._state_lock:
            future = handle.cancel_goal_async()
            goal.cancels.append(future)
            goal.cancel_sent_at = time.monotonic()
        # Not under the state lock, as in _wait_goal().
        future.add_done_callback(functools.partial(self._on_cancel_answer, goal))
        return future

    def _resolve(self, goal: _Goal) -> str | None:
        """With the state lock held: resolve `goal` once the commander showed it no longer active.

        Only a goal that is unresolved; returns how (_resolution), else None.
        """
        if goal not in self._unresolved:
            return None
        how = _resolution(goal)
        if how is not None:
            self._unresolved.remove(goal)
            goal.resolved = how
            self._resolved.append(goal)
        return how

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
        and abandons it once its action server stayed undiscovered for
        ABSENT_ABANDON_S, from the first check that found it so to one that
        still does, or once ABANDON_AFTER_S passed since its call gave up.
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
        # Outside the state lock, so the gaze loop's reads of last_state do
        # not wait for discovery.
        present = goal.client.server_is_ready()
        now = time.monotonic()
        with self._state_lock:
            if self._goal is not goal:
                return None
            if goal.ended():
                self._goal = None
                return None
            if present:
                goal.absent_since = None
            elif goal.absent_since is None:
                goal.absent_since = now
            absent_s = 0.0 if goal.absent_since is None else now - goal.absent_since
            waited_s = now - goal.gave_up_at
        if absent_s >= ABSENT_ABANDON_S:
            self._abandon(
                goal, f"its action server has not been discovered for {ABSENT_ABANDON_S:g} s"
            )
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
        """Stop waiting for `goal`: calls are taken again, and the arms are unverified.

        The goal is unresolved until the commander shows that it is no
        longer active (_confirm_ended).
        """
        with self._state_lock:
            if self._goal is not goal:
                return
            self._goal = None
            # Only the answers to cancels sent from now on can resolve it.
            goal.evidence_from = len(goal.cancels)
            self._unresolved.append(goal)
            self._set_unverified(
                f"{goal.what} from an earlier call was given up on before its end was known",
                goal.group,
            )
        logger.warning(
            f"Abandoned {goal.what}: {reason}. Calls to the commander are taken again, but the "
            "arms are unverified: goals are refused and the robot does not drive until the "
            "commander shows that it is no longer active and a settled read shows the arms "
            "still."
        )

    def _set_unverified(self, cause: str, group: str) -> None:
        """With the state lock held: the arms are unverified, as a goal for `group` may move.

        What that goal moves is unknown from now on: the arms of its group,
        or its gripper's commanded position.
        """
        self._unverified = cause
        self._unverified_count += 1
        self._unverified_groups.add(group)
        self._last_state = _without(self._last_state, group)
        if group in self._gripper_positions:
            self._gripper_positions[group] = None

    def _unverified_mark(self) -> int:
        """How many times the arms became unverified so far, for _remember_reading()."""
        with self._state_lock:
            return self._unverified_count

    def _forget_arms(self, group: str) -> None:
        with self._state_lock:
            self._last_state = _without(self._last_state, group)

    def _missing(self, arm: str, reading: Mapping[str, float]) -> list[str]:
        """The joints of `arm` that `reading` lacks, by name.

        openarm_right_joint2 does not count: walkie-ros launched with
        right_joint2_fixed holds it at 0 and does not list it.
        """
        joints = self._arm_joints[arm] - {RIGHT_JOINT2}
        return sorted(joint for joint in joints if joint not in reading)

    def _remember_reading(
        self,
        group: str,
        reading: dict[str, float],
        count: int,
        verified: tuple[str, ...] | None = None,
    ) -> bool:
        """Keep what `reading` of `group` says about the arms, and its joint order.

        `count` is _unverified_mark() from before the read. When the arms
        became unverified since, the reading may predate a motion, so only
        its joint order is kept, and it returns False. With `verified`, the
        gripper groups that a settled read checked, it also clears the
        unverified state and keeps what the recovery checked
        (last_recovery), unless a goal is unresolved, when it keeps only
        the joint order and returns False.
        """
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
            if self._unverified_count != count or (verified is not None and self._unresolved):
                return False
            old = self._last_state
            self._last_state = ArmsState(
                left=covered.get("left", old.left), right=covered.get("right", old.right)
            )
            if verified is not None:
                self._last_recovery = Recovery(
                    count,
                    tuple((goal.what, goal.resolved) for goal in self._resolved),
                    verified,
                )
                self._unverified = None
                self._unverified_groups.clear()
                self._resolved.clear()
            return True

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


def _cancel_state(future: Future, goal_id: Any) -> str | None:
    """The _CANCEL_OUTCOMES key of the answer to a cancel of `goal_id`; None until it comes."""
    if future.cancelled():
        return "no answer"
    if not future.done():
        return None
    if future.exception() is not None:
        return "no answer"
    return _cancel_outcome(future.result(), goal_id)


def _resolution(goal: _Goal) -> str | None:
    """How the commander showed that `goal` is no longer active, or None while it may be.

    "result" once its result came, or "unknown" when that result says that
    the commander does not know it; "rejected" once it was not accepted.
    Else by the answers to the cancels sent since it was abandoned or last
    accepted late: "unknown" when one says that the commander does not
    know it, as after a restart, whose goal threads died with it, and
    "ended" when one says that it ended and no cancel of it may have been
    accepted. The commander answers a second cancel of a goal that is still
    stopping after an accepted one as if the goal had ended, so after an
    accepted cancel only the result resolves the goal. With the state lock
    held.
    """
    for result in (goal.result, *goal.more_results):
        if result is not None and result.done() and result.exception() is None:
            return "unknown" if result.result().status == GoalStatus.STATUS_UNKNOWN else "result"
    sent = goal.sent
    if sent.done() and sent.exception() is None:
        handle = sent.result()
        if handle is None or not handle.accepted:
            return "rejected"
    answers = [_cancel_state(future, goal.goal_id) for future in goal.cancels]
    recent = answers[goal.evidence_from :]
    if "unknown" in recent:
        return "unknown"
    maybe_accepted = any(answer in (None, "accepted", "no answer") for answer in answers)
    if "ended" in recent and not maybe_accepted:
        return "ended"
    return None


def _handle_for(goal: _Goal) -> ClientGoalHandle:
    """The goal handle of `goal` from the commander's acceptance, else one made from its id.

    rclpy needs only the id to cancel a goal or to ask for its result.
    """
    sent = goal.sent
    if sent.done() and sent.exception() is None:
        handle = sent.result()
        if handle is not None and handle.accepted:
            return handle
    action = SetJointPosition if goal.group in ARM_GOAL_GROUPS else ControlGripper
    response = action.Impl.SendGoalService.Response(accepted=True)
    return ClientGoalHandle(goal.client, goal.goal_id, response)


def _log_resolved(goal: _Goal, how: str) -> None:
    logger.info(
        f"{_capital(goal.what)} from an earlier call is no longer active: {_RESOLUTIONS[how]}."
    )


def _unconfirmed(goals: Sequence[_Goal]) -> StepError:
    """The StepError for the goals that recovery left unresolved, with the state lock held.

    It says, by the latest answer to a cancel of each, whether the
    commander accepted the cancel and the goal has not ended yet, or
    refused it or did not answer, when an arm may still be moving.
    """
    sentences = []
    unsure = False
    for goal in goals:
        answers = [_cancel_state(future, goal.goal_id) for future in goal.cancels]
        latest = answers[-1] if answers else None
        moving = (
            "the gripper may still be moving"
            if goal.group in GRIPPER_GOAL_GROUPS
            else "an arm may still be moving"
        )
        if latest in ("accepted", "ended") and "accepted" in answers:
            sentences.append(
                f"The commander accepted the cancel of {goal.what} from an earlier call, and "
                "its end has not come yet."
            )
            continue
        unsure = True
        if latest == "refused":
            sentences.append(
                f"The commander refused the cancel of {goal.what} from an earlier call, so "
                f"{moving}."
            )
        else:
            sentences.append(
                f"The commander did not answer a cancel of {goal.what} from an earlier call in "
                f"time, so {moving}."
            )
    then = (
        "Wait a few seconds, then try again; if it fails again, tell the person that the arms "
        "cannot be checked."
        if unsure
        else "Wait a few seconds, then try again."
    )
    return StepError(
        "PRECONDITION_FAILED",
        f"{' '.join(sentences)} {then}",
        {"unresolved": [goal.what for goal in goals]},
    )


def _no_time_to_settle(timeout_s: float) -> StepError:
    return StepError(
        "TIMEOUT",
        f"The arms could not be read twice {SETTLE_S:g} s apart within {timeout_s:g} s.",
    )


def _ended(
    error: StepError, what: str, timeout_s: float, cancel_outcome: str, outcome: str
) -> StepError:
    """The StepError for a goal call that a cancel or its timeout ended.

    `cancel_outcome` is the _CANCEL_OUTCOMES key of the commander's answer
    to the cancel, or _PENDING when no cancel was sent; it goes in the data
    as "cancel". `outcome` says in words what that means.
    """
    data = {"cancel": cancel_outcome}
    if error.code == "CANCELLED":
        return StepError("CANCELLED", f"A stop or a reset interrupted {what}; {outcome}.", data)
    return StepError(
        "TIMEOUT", f"{_capital(what)} did not finish within {timeout_s:g} s; {outcome}.", data
    )


def _accepted_late(goal: _Goal) -> str:
    """Why the arms are unverified once the commander accepts `goal` after it was abandoned."""
    if goal.group in GRIPPER_GOAL_GROUPS:
        return (
            f"the commander accepted {goal.what} from an earlier call late, and the gripper "
            "makes that motion although it was asked to cancel it"
        )
    return (
        f"the commander accepted {goal.what} from an earlier call late and was asked to "
        "cancel it"
    )


def unverified_refusal(action: str, cause: str) -> StepError:
    """The refusal of `action` while the arms are unverified by `cause`.

    Public so the fake arms in walkie_agent.testing refuse in the same words.
    """
    return StepError(
        "PRECONDITION_FAILED",
        f"Cannot {action}: an arm or a gripper may still be moving, as {cause}. Call "
        "get_arm_state to check the arms, then try again.",
        {"unverified": cause},
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


def _without(state: ArmsState, group: str) -> ArmsState:
    """`state` with the arms that `group` moves unknown."""
    arms = _GROUP_ARMS.get(group, ())
    return ArmsState(
        left=None if "left" in arms else state.left,
        right=None if "right" in arms else state.right,
    )


def _capital(text: str) -> str:
    return f"{text[0].upper()}{text[1:]}"


def _full_stop(text: str) -> str:
    """`text` ending with a full stop; the commander's status text has none."""
    return text if text.endswith((".", "?", "!")) else f"{text}."


def _check_cancelled(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise StepError("CANCELLED", "A stop or a reset interrupted the call.")


def _check_group(group: str, allowed: frozenset[str]) -> None:
    if group not in allowed:
        raise ValueError(f"{group!r} is not one of the groups {sorted(allowed)}.")


def _band(response: Any) -> tuple[float, float]:
    """The two values of GRASP_BAND_PARAMETERS in a GetParameters answer, or UNAVAILABLE.

    The commander declares both as doubles. When it lacks either, its
    parameter service answers no values at all.
    """
    values = list(response.values) if response is not None else []
    if len(values) != len(GRASP_BAND_PARAMETERS):
        raise StepError(
            "UNAVAILABLE",
            f"The commander answered {len(values)} values for {', '.join(GRASP_BAND_PARAMETERS)}.",
        )
    band = []
    for name, value in zip(GRASP_BAND_PARAMETERS, values, strict=True):
        if value.type != ParameterType.PARAMETER_DOUBLE or not math.isfinite(value.double_value):
            raise StepError(
                "UNAVAILABLE",
                f"The commander's {name} is not a finite number (parameter type {value.type}).",
            )
        band.append(value.double_value)
    return band[0], band[1]


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
