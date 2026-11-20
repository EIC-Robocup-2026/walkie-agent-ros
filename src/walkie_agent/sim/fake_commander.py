"""A stand-in for walkie-ros's bimanual_commander, for testing the arm client without MoveIt.

Provides the commander interfaces the agent uses (ADR-0021), under the same
names and types:
  - /set_joint_position  my_robot_interfaces/action/SetJointPosition
  - /control_gripper     my_robot_interfaces/action/ControlGripper
  - /go_to_pose          my_robot_interfaces/action/GoToPose, for one arm, one arm
    with the lift, or one whole-body group, which refuses a Cartesian path as
    the commander does
  - /get_joint_states    my_robot_interfaces/srv/GetJointStates
  - /bimanual_commander/get_parameters and /bimanual_commander/set_parameters,
    rclpy's parameter services of a node named bimanual_commander, with
    grasp_detect_min, grasp_detect_max and allow_gripper_vs_octomap declared
    as the commander declares them
and listens on /trajectory_execution_event (std_msgs/String) as MoveIt's
trajectory execution manager does.

The groups and their joints come from config/robot_model.toml, the snapshot
of the pinned walkie-ros commit, so the fake and the real commander agree on
names and order. It serves the one-arm *_lift groups for pose goals, joint
goals and reads, the lift's joints first, as the commander lists them; these
are the goals pick_object sends them (ADR-0026, decision PK-5). Any goal or
read for both_arms_lift is rejected as for an unknown group.

What it copies from the real commander, since the agent must handle it:
  - a joint goal whose position count differs from the group's joints is
    rejected;
  - a cancel of a joint goal stops every arm motion, including those of
    other goals, as MoveIt's stop does, and is then accepted;
  - "stop" on /trajectory_execution_event stops every arm motion in the
    same way, and leaves a gripper ramp and a goal that still plans alone;
  - a joint goal cancelled while it plans ends CANCELED "Canceled", and one
    stopped while it moves ends ABORTED "Execution Failed";
  - a cancel of a gripper goal is accepted, the fingers finish their ramp,
    and the goal then ends CANCELED "Canceled";
  - a newer command for a gripper's group, a goal or supersede(), ends the
    ramp of the older goal, which then ends ABORTED "Failed" with a 0.0 gap
    and grasped false, or CANCELED "Canceled" when it was asked to cancel;
  - a cancel of a goal that has ended is answered ERROR_GOAL_TERMINATED;
  - right_joint2_fixed removes openarm_right_joint2 from right_arm and
    both_arms;
  - a gripper reports `grasped` and the gap only on a close, by the
    commander's rule and ROS parameters (grasp_close_threshold,
    grasp_detect_min, grasp_detect_max), and a close that stalls on an
    object ends with success false.

Where it differs from the real commander, so a test does not rely on it:
  - the arms start exactly at home, the fingers fully open and the lift at 0;
  - a pose goal solves no IK: it moves the joints in pose_joints, none by default,
    and a whole-body goal never moves a base or a lift, which the fake does not
    have;
  - a move goes in a straight line, in time, and ends exactly on its
    target, and a read gives exact positions at once;
  - a failed execution stops exactly half way;
  - it has no planning scene and checks no collisions;
  - a cancel or a stop is seen within one 20 ms step;
  - planning takes no time unless plan_time_s, plan_gate or
    plan_until_cancelled says so, so a goal cancelled as soon as it is
    accepted ends "Execution Failed" by default, where the commander would
    more likely still plan and end it "Canceled";
  - a move takes move_time_s whatever its length, unless joint_speed_rad_s
    sets the speed of its largest joint change;
  - never_answer holds a goal until it is cancelled, and a held gripper
    goal then ends at once, without the ramp the commander would finish;
  - ramp_gate holds a gripper ramp part way, whatever a cancel asks, until
    a test lets it finish or a newer command supersedes it;
  - a close always reports the gap, which the commander's
    grasp_detect_enable can turn off;
  - the two grippers ramp at the same time, where the commander ramps one
    gripper at a time;
  - after vanish(), a goal that the destroyed server took goes on moving
    the joints and never sends its result, also after come_back(), where a
    commander that died stops its goals with it;
  - the joint order and right_joint2_fixed can change while it runs, and
    grasp_close_threshold is read on every gripper goal, where the
    commander reads it once at startup;
  - reads and goals are thread-safe, so it cannot show the hazard of
    calling MoveIt's getCurrentState() during a goal (ADR-0024);
  - reject_cancels refuses a cancel with ERROR_NONE and no goal, as an
    rclpy server does, where the commander's rclcpp server would answer
    ERROR_REJECTED; the commander itself never refuses a cancel.

The knobs are plain attributes, read when a goal or request arrives, so a
test can change them on a running fake. Goals run concurrently, each in its
own callback, as the commander runs each in its own thread.
"""

import functools
import math
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import rclpy
from my_robot_interfaces.action import ControlGripper, GoToPose, SetJointPosition
from my_robot_interfaces.srv import GetJointStates
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from walkie_agent.arm.poses import (
    ARM_GROUPS,
    BOTH_ARMS_GROUP,
    GRIPPER_GROUPS,
    LIFT_ARM_GROUPS,
    RIGHT_JOINT2,
    WHOLE_BODY_GROUPS,
    RobotModel,
    load_robot_model,
)

# The groups served, in robot_model.toml's order, and the arm each one-arm *_lift
# group moves with the lift.
_GROUPS = (*ARM_GROUPS.values(), BOTH_ARMS_GROUP, *GRIPPER_GROUPS.values())
_LIFT_GROUP_ARMS = {group: arm for arm, group in LIFT_ARM_GROUPS.items()}
# The groups a pose goal is accepted for, and the arm each one moves.
_POSE_GROUPS = {
    **{group: arm for arm, group in ARM_GROUPS.items()},
    **{group: arm for arm, group in WHOLE_BODY_GROUPS.items()},
    **{group: arm for arm, group in LIFT_ARM_GROUPS.items()},
}
# The interfaces, by the names vanish() takes.
SET_JOINT_POSITION = "set_joint_position"
CONTROL_GRIPPER = "control_gripper"
GO_TO_POSE = "go_to_pose"
GET_JOINT_STATES = "get_joint_states"
_INTERFACES = (SET_JOINT_POSITION, CONTROL_GRIPPER, GO_TO_POSE, GET_JOINT_STATES)
# What a whole-body group reports on success with the base standing still.
WHOLE_BODY_SUCCESS = "Success (base fixed)"
# MoveIt's trajectory execution event topic, and the event that stops every
# execution.
EXECUTION_EVENT = "trajectory_execution_event"
_STOP = "stop"
# The steps a move or a gripper command is split into, so a cancel and a
# shutdown are seen within one step.
_STEP_S = 0.02
# How long destroy_node() waits for the goals and requests it released.
_RELEASE_S = 2.0


class FakeCommander(Node):
    def __init__(
        self,
        model: RobotModel | None = None,
        move_time_s: float = 0.2,
        gripper_time_s: float = 0.05,
        right_joint2_fixed: bool = False,
    ):
        super().__init__("bimanual_commander")
        self.model = model if model is not None else load_robot_model()

        # -- Knobs --
        # How long a joint move and a gripper command take.
        self.move_time_s = move_time_s
        self.gripper_time_s = gripper_time_s
        # How long a joint goal plans before it moves. A cancel while it
        # plans ends it CANCELED "Canceled" once the plan is done.
        self.plan_time_s = 0.0
        # A joint goal that plans while this is set goes on planning, after
        # plan_time_s, until the event is set, so a test decides when the
        # plan is done. None to plan for plan_time_s only.
        self.plan_gate: threading.Event | None = None
        # A joint goal goes on planning, after plan_time_s and plan_gate,
        # until its cancel is requested, and then ends CANCELED "Canceled"
        # without moving.
        self.plan_until_cancelled = False
        # The speed, in rad/s, of a joint move's largest joint change, which
        # then sets how long the move takes instead of move_time_s. None to
        # take move_time_s.
        self.joint_speed_rad_s: float | None = None
        # A joint goal ends ABORTED "Planning Failed" without moving.
        self.plan_failure = False
        # A joint goal stops half way and ends ABORTED "Execution Failed", as
        # when a controller fails during the motion.
        self.execution_failure = False
        # A joint goal ends SUCCEEDED "Plan Only" without moving, as with the
        # commander's plan_only parameter.
        self.plan_only = False
        # Joint name -> radians added to its target, so a move ends that far off.
        self.joint_errors: dict[str, float] = {}
        # Joint name -> radians that a pose goal that succeeds moves the joint to.
        # plan_failure, execution_failure and plan_only apply to pose goals too.
        self.pose_joints: dict[str, float] = {}
        # A pose goal ends ABORTED with this status without moving, as a
        # whole-body refusal does; None to plan and move.
        self.pose_failure: str | None = None
        # Every goal is rejected.
        self.reject_goals = False
        # Goals are accepted and never finish until cancelled.
        self.never_answer = False
        # How long a goal waits to be accepted or rejected.
        self.accept_delay_s = 0.0
        # A goal that arrives while this is set waits until the event is set
        # before it is accepted or rejected, so a test decides when the
        # commander answers; reject_goals is read once it is set. None to
        # answer at once.
        self.accept_gate: threading.Event | None = None
        # How long a goal that ended waits before its result is sent, so a
        # cancel meanwhile is answered ERROR_GOAL_TERMINATED. math.inf holds
        # it until the fake closes, as when the commander restarted and the
        # result never comes.
        self.result_delay_s = 0.0
        # A joint or pose goal that ends while this is set waits, after
        # result_delay_s, until the event is set before its result is sent,
        # so a test decides when the result comes. None to send it then.
        self.result_gate: threading.Event | None = None
        # A gripper ramp that starts while this is set holds after its first
        # step until the event is set or a newer command for the group
        # supersedes it, whatever a cancel asks, so a test decides when the
        # fingers finish. None to ramp in gripper_time_s.
        self.ramp_gate: threading.Event | None = None
        # Every cancel request is refused.
        self.reject_cancels = False
        # How long a cancel request waits to be answered.
        self.cancel_delay_s = 0.0
        # A cancel request that arrives while this is set waits, after
        # cancel_delay_s, until the event is set before it is answered, so
        # a test decides when; reject_cancels is read once it is set. None
        # to answer at once. Only a cancel of a goal that has not ended
        # waits: the others are answered without one.
        self.cancel_gate: threading.Event | None = None
        # How long /get_joint_states takes to answer.
        self.joint_states_delay_s = 0.0
        # A /get_joint_states request that arrives while this is set waits,
        # after joint_states_delay_s, until the event is set before it is
        # answered, so a test decides when. None to answer at once.
        self.joint_states_gate: threading.Event | None = None
        # Called with the group and the index in joint_state_requests of
        # every /get_joint_states request, once it waited out
        # joint_states_delay_s and joint_states_gate and before its answer
        # is made, in the service's thread, so a test can move a joint
        # between two reads or move a test clock on. It must not raise.
        # None to call nothing.
        self.joint_states_hook: Callable[[str, int], None] | None = None
        # The status /get_joint_states fails with, as when MoveIt has no
        # robot state yet; None to answer.
        self.read_failure: str | None = None
        # Joints left out of every /get_joint_states answer.
        self.missing_joints: set[str] = set()
        # walkie-ros's right_joint2_fixed launch argument.
        self.right_joint2_fixed = right_joint2_fixed
        # Serve every group's joints in reverse order, so a client that
        # assumes the SRDF's order sends the wrong values.
        self.reverse_joint_order = False
        # Gripper group -> the width of the object between its fingers, as
        # the travel of one finger in metres: a close stops there, which
        # sets the settled gap. None for nothing held.
        self.object_widths: dict[str, float | None] = dict.fromkeys(GRIPPER_GROUPS.values())
        # The commander's grasp detection parameters, with its defaults. Read
        # on every gripper goal, as the commander reads them.
        self.declare_parameter("grasp_close_threshold", 0.03)
        self.declare_parameter("grasp_detect_min", 0.005)
        self.declare_parameter("grasp_detect_max", 0.044)
        # Whether the fingers may touch the octomap. The fake has no planning
        # scene, so it only keeps the value.
        self.declare_parameter("allow_gripper_vs_octomap", False)

        # -- What it received, for tests --
        # (group, positions) of every joint goal, and (group, position) of
        # every gripper goal, rejected ones included.
        self.joint_goals: list[tuple[str, tuple[float, ...]]] = []
        self.gripper_goals: list[tuple[str, float]] = []
        # Every pose goal as it came, rejected ones included.
        self.pose_goals: list[GoToPose.Goal] = []
        # (group, status) of every goal that ended, whether or not its
        # result was sent.
        self.results: list[tuple[str, str]] = []
        # Every cancel request, refused ones included. A cancel of a goal
        # that has ended is answered without one.
        self.cancel_requests = 0
        # The group of every /get_joint_states request.
        self.joint_state_requests: list[str] = []
        # How many "stop" events it received on /trajectory_execution_event.
        self.stops_heard = 0

        # Reentrant, so that a motion step can check whether it was stopped
        # while it holds the lock.
        self._lock = threading.RLock()
        # Every arm, finger and lift joint, including openarm_right_joint2,
        # which stays where it is while right_joint2_fixed hides it.
        home = self.model.states.get(BOTH_ARMS_GROUP, {}).get("home", {})
        self._joints: dict[str, float] = {}
        for group in (*ARM_GROUPS.values(), *GRIPPER_GROUPS.values()):
            for joint in self.model.groups[group]:
                # The fingers start open, at the top of their travel.
                default = self.model.limits[joint][1] if group in GRIPPER_GROUPS.values() else 0.0
                self._joints[joint] = float(home.get(joint, default))
        for joint in sorted(self.model.lift_joints):
            self._joints[joint] = 0.0
        # How many times an accepted joint cancel or a "stop" event stopped
        # the arms. A motion stops when this changes, as MoveIt's stop ends
        # every execution.
        self._arm_stops = 0
        # Gripper group -> how many commands it was given, by goals and by
        # supersede(). A ramp stops when this changes, as the commander's
        # per-group command generation ends a superseded ramp.
        self._gripper_commands = dict.fromkeys(GRIPPER_GROUPS.values(), 0)
        # Set by destroy_node(), to release every held goal and request.
        self._closing = threading.Event()
        self._idle = threading.Condition()
        self._active = 0

        # Reentrant, so a held goal or a slow request does not block the others.
        self._callbacks = ReentrantCallbackGroup()
        # Each interface by name, None once vanish() destroyed it, and how
        # many times it was created. A goal keeps the number of the server
        # that took it, by goal id, so that it sends no result once that
        # server is gone.
        self._entities: dict[str, Any] = dict.fromkeys(_INTERFACES)
        self._instances = dict.fromkeys(_INTERFACES, 0)
        self._goal_servers: dict[bytes, tuple[str, int]] = {}
        for name in _INTERFACES:
            self._create(name)
        # Not among the interfaces vanish() takes: it is move_group's, not
        # the commander's, and stays when the commander dies.
        self._execution_events = self.create_subscription(
            String, EXECUTION_EVENT, self._on_execution_event, 10, callback_group=self._callbacks
        )

    def destroy_node(self) -> None:
        # Held goals and slow requests block executor threads; they are
        # released and finish before the servers go. Node.destroy_node() does
        # not destroy action servers, which would otherwise stay on the graph.
        self._closing.set()
        with self._idle:
            self._idle.wait_for(lambda: self._active == 0, timeout=_RELEASE_S)
        for name in (SET_JOINT_POSITION, CONTROL_GRIPPER, GO_TO_POSE):
            server = self._entities[name]
            if server is not None:
                server.destroy()
        super().destroy_node()

    def vanish(self, *names: str) -> None:
        """Destroy interfaces at once, as when the commander dies mid-goal.

        `names` are among SET_JOINT_POSITION, CONTROL_GRIPPER, GO_TO_POSE and
        GET_JOINT_STATES; all four by default. A goal that a destroyed
        server runs or holds sends no result, also after come_back(): it
        ends when the fake closes.
        """
        for name in names or _INTERFACES:
            with self._lock:
                entity, self._entities[name] = self._entities[name], None
            if entity is None:
                continue
            if name == GET_JOINT_STATES:
                self.destroy_service(entity)
            else:
                entity.destroy()

    def come_back(self) -> None:
        """Create the interfaces that vanish() destroyed again, as a restarted commander.

        The new servers do not know the goals of the old ones, so a cancel
        of such a goal is answered as for an unknown goal.
        """
        for name in _INTERFACES:
            if self._entities[name] is None:
                self._create(name)

    def supersede(self, group: str, position: float) -> None:
        """Command gripper `group` to `position` from outside the agent, as another node can.

        The fingers go there at once. A goal of the group that is still in
        its ramp ends as a superseded goal: ABORTED "Failed", with a 0.0 gap
        and grasped false.
        """
        [finger] = self.model.groups[group]
        with self._lock:
            self._gripper_commands[group] += 1
            self._joints[finger] = float(position)

    def joints(self) -> dict[str, float]:
        """Every arm, finger and lift joint by name."""
        with self._lock:
            return dict(self._joints)

    def teleport(self, joints: Mapping[str, float]) -> None:
        """Set joints instantly, for tests that need a known start."""
        with self._lock:
            for joint, value in joints.items():
                if joint not in self._joints:
                    raise KeyError(f"{joint} is not an arm or finger joint.")
                self._joints[joint] = float(value)

    def group_joints(self, group: str) -> list[str] | None:
        """The joints of `group` in the order served, or None for a group it does not have."""
        if group in _LIFT_GROUP_ARMS:
            arm_group = ARM_GROUPS[_LIFT_GROUP_ARMS[group]]
            joints = [*sorted(self.model.lift_joints), *self.model.groups[arm_group]]
        elif group in _GROUPS:
            joints = list(self.model.groups[group])
        else:
            return None
        if self.right_joint2_fixed:
            joints = [joint for joint in joints if joint != RIGHT_JOINT2]
        if self.reverse_joint_order:
            joints.reverse()
        return joints

    # -- Callbacks ----------------------------------------------------------------

    def _on_joint_goal(self, goal: SetJointPosition.Goal) -> GoalResponse:
        positions = tuple(goal.joint_positions)
        self.joint_goals.append((goal.group_name, positions))
        self._delay_answer()
        # The commander accepts a gripper group here too, as one joint.
        joints = self.group_joints(goal.group_name)
        if self.reject_goals or joints is None:
            return GoalResponse.REJECT
        if len(positions) != len(joints):
            self.get_logger().error(
                f"Joint count mismatch for {goal.group_name}: expected {len(joints)}, "
                f"got {len(positions)}"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_pose_goal(self, goal: GoToPose.Goal) -> GoalResponse:
        self.pose_goals.append(goal)
        self._delay_answer()
        if self.reject_goals or goal.group_name not in _POSE_GROUPS:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_gripper_goal(self, goal: ControlGripper.Goal) -> GoalResponse:
        self.gripper_goals.append((goal.group_name, goal.position))
        self._delay_answer()
        if self.reject_goals or goal.group_name not in GRIPPER_GROUPS.values():
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        with self._serving():
            with self._lock:
                self.cancel_requests += 1
            if self.cancel_delay_s > 0:
                self._closing.wait(self.cancel_delay_s)
            self._wait_gate(self.cancel_gate)
            return CancelResponse.REJECT if self.reject_cancels else CancelResponse.ACCEPT

    def _on_joint_cancel(self, goal_handle) -> CancelResponse:
        response = self._on_cancel(goal_handle)
        if response == CancelResponse.ACCEPT:
            # The commander stops both arms before it accepts, whatever goal
            # moves them.
            with self._lock:
                self._arm_stops += 1
        return response

    def _on_execution_event(self, message: String) -> None:
        # MoveIt's trajectory execution manager stops every execution on
        # "stop", whatever goal runs it. A goal that still plans executes
        # afterwards, and a gripper ramp is not a MoveIt execution, so
        # neither stops.
        if message.data != _STOP:
            return
        with self._lock:
            self.stops_heard += 1
            self._arm_stops += 1

    def _on_joint_states(
        self, request: GetJointStates.Request, response: GetJointStates.Response
    ) -> GetJointStates.Response:
        with self._serving():
            group = request.group_name
            with self._lock:
                index = len(self.joint_state_requests)
                self.joint_state_requests.append(group)
            if self.joint_states_delay_s > 0:
                self._closing.wait(self.joint_states_delay_s)
            self._wait_gate(self.joint_states_gate)
            hook = self.joint_states_hook
            if hook is not None:
                hook(group, index)
            joints = self.group_joints(group)
            if joints is None:
                response.success = False
                response.status = f"Invalid group: {group}"
                return response
            if self.read_failure is not None:
                response.success = False
                response.status = self.read_failure
                return response
            joints = [joint for joint in joints if joint not in self.missing_joints]
            values = self.joints()
            response.joint_names = joints
            response.joint_positions = [values[joint] for joint in joints]
            response.success = True
            response.status = "OK"
            return response

    def _execute_move(self, server: tuple[str, int], goal_handle) -> SetJointPosition.Result:
        with self._serving():
            self._took(server, goal_handle)
            group = goal_handle.request.group_name
            result = SetJointPosition.Result()
            if self.never_answer:
                self._hold(self._arm_stopped(goal_handle))
                return self._finish(goal_handle, result, group, False, "Execution Failed")
            # The commander plans first, and looks at a cancel only once the
            # plan is done.
            if self.plan_time_s > 0:
                self._closing.wait(self.plan_time_s)
            self._wait_gate(self.plan_gate)
            if self.plan_until_cancelled:
                self._hold(lambda: goal_handle.is_cancel_requested)
            if goal_handle.is_cancel_requested:
                return self._finish(goal_handle, result, group, False, "Canceled")
            if self.plan_failure:
                return self._finish(goal_handle, result, group, False, "Planning Failed")
            if self.plan_only:
                return self._finish(goal_handle, result, group, True, "Plan Only")
            joints = self.group_joints(group)
            targets = {
                joint: position + self.joint_errors.get(joint, 0.0)
                for joint, position in zip(joints, goal_handle.request.joint_positions, strict=True)
            }
            stopped = self._arm_stopped(goal_handle)
            if self.execution_failure:
                with self._lock:
                    targets = {
                        joint: (self._joints[joint] + target) / 2
                        for joint, target in targets.items()
                    }
                self._move(targets, self._move_time(targets, self.move_time_s / 2), stopped)
                return self._finish(goal_handle, result, group, False, "Execution Failed")
            # The commander stops the arms on a cancel, and MoveIt then ends
            # the execution as failed rather than cancelled.
            if not self._move(targets, self._move_time(targets, self.move_time_s), stopped):
                return self._finish(goal_handle, result, group, False, "Execution Failed")
            return self._finish(goal_handle, result, group, True, "Success")

    def _execute_pose(self, server: tuple[str, int], goal_handle) -> GoToPose.Result:
        with self._serving():
            self._took(server, goal_handle)
            request = goal_handle.request
            group = request.group_name
            whole_body = group in WHOLE_BODY_GROUPS.values()
            result = GoToPose.Result()
            if self.never_answer:
                self._hold(self._arm_stopped(goal_handle))
                return self._finish(goal_handle, result, group, False, "Execution Failed")
            if whole_body and request.cartesian_path:
                arm = ARM_GROUPS[_POSE_GROUPS[group]]
                status = f"Cartesian paths are not supported on {group}: use {arm}_lift"
                return self._finish(goal_handle, result, group, False, status)
            if self.plan_time_s > 0:
                self._closing.wait(self.plan_time_s)
            self._wait_gate(self.plan_gate)
            if self.plan_until_cancelled:
                self._hold(lambda: goal_handle.is_cancel_requested)
            if goal_handle.is_cancel_requested:
                return self._finish(goal_handle, result, group, False, "Canceled")
            if self.pose_failure is not None:
                return self._finish(goal_handle, result, group, False, self.pose_failure)
            if self.plan_failure:
                return self._finish(goal_handle, result, group, False, "Planning Failed")
            if self.plan_only:
                status = "Plan Only (base fixed)" if whole_body else "Plan Only"
                return self._finish(goal_handle, result, group, True, status)
            targets = dict(self.pose_joints)
            stopped = self._arm_stopped(goal_handle)
            if self.execution_failure:
                with self._lock:
                    targets = {
                        joint: (self._joints[joint] + target) / 2
                        for joint, target in targets.items()
                    }
                self._move(targets, self.move_time_s / 2, stopped)
                return self._finish(goal_handle, result, group, False, "Execution Failed")
            if not self._move(targets, self.move_time_s, stopped):
                return self._finish(goal_handle, result, group, False, "Execution Failed")
            status = WHOLE_BODY_SUCCESS if whole_body else "Success"
            return self._finish(goal_handle, result, group, True, status)

    def _execute_gripper(self, server: tuple[str, int], goal_handle) -> ControlGripper.Result:
        with self._serving():
            self._took(server, goal_handle)
            group = goal_handle.request.group_name
            position = goal_handle.request.position
            result = ControlGripper.Result()
            superseded = self._claim_gripper(group)
            if self.never_answer:
                self._hold(lambda: goal_handle.is_cancel_requested)
                status = "Canceled" if goal_handle.is_cancel_requested else "Failed"
                return self._finish(goal_handle, result, group, False, status)
            [finger] = self.model.groups[group]
            width = self.object_widths.get(group)
            settled = position if width is None else max(position, float(width))
            # The commander finishes the ramp before it looks at a cancel,
            # and a joint cancel's stop does not reach the gripper. A newer
            # command for the group ends the ramp, and the goal then reports
            # the result's defaults: a 0.0 gap and grasped false.
            moved = self._move(
                {finger: settled}, self.gripper_time_s, superseded, hold=self.ramp_gate
            )
            if not moved and superseded():
                status = "Canceled" if goal_handle.is_cancel_requested else "Failed"
                return self._finish(goal_handle, result, group, False, status)
            if goal_handle.is_cancel_requested:
                return self._finish(goal_handle, result, group, False, "Canceled")
            if position <= self.get_parameter("grasp_close_threshold").value:
                low = self.get_parameter("grasp_detect_min").value
                high = self.get_parameter("grasp_detect_max").value
                result.gripper_gap = float(settled)
                result.grasped = low <= settled <= high
            # The controller ends STALLED when the fingers stop on an object
            # before the commanded position, and the commander reports that
            # as a failure.
            reached = settled == position
            return self._finish(
                goal_handle, result, group, reached, "Success" if reached else "Failed"
            )

    # -- Helpers ----------------------------------------------------------------

    def _create(self, name: str) -> None:
        """Create interface `name`, as a new instance of it."""
        with self._lock:
            self._instances[name] += 1
            server = (name, self._instances[name])
        if name == GET_JOINT_STATES:
            entity = self.create_service(
                GetJointStates, name, self._on_joint_states, callback_group=self._callbacks
            )
        elif name == SET_JOINT_POSITION:
            entity = ActionServer(
                self,
                SetJointPosition,
                name,
                execute_callback=functools.partial(self._execute_move, server),
                goal_callback=self._on_joint_goal,
                cancel_callback=self._on_joint_cancel,
                callback_group=self._callbacks,
            )
        elif name == GO_TO_POSE:
            # The commander's pose cancel stops both arms, as a joint cancel does.
            entity = ActionServer(
                self,
                GoToPose,
                name,
                execute_callback=functools.partial(self._execute_pose, server),
                goal_callback=self._on_pose_goal,
                cancel_callback=self._on_joint_cancel,
                callback_group=self._callbacks,
            )
        else:
            entity = ActionServer(
                self,
                ControlGripper,
                name,
                execute_callback=functools.partial(self._execute_gripper, server),
                goal_callback=self._on_gripper_goal,
                cancel_callback=self._on_cancel,
                callback_group=self._callbacks,
            )
        with self._lock:
            self._entities[name] = entity

    def _took(self, server: tuple[str, int], goal_handle) -> None:
        """Note that `server`, an (interface, instance) pair, took the goal of `goal_handle`."""
        with self._lock:
            self._goal_servers[bytes(goal_handle.goal_id.uuid)] = server

    def _server_gone(self, goal_handle) -> bool:
        """True when the server that took the goal of `goal_handle` was destroyed."""
        with self._lock:
            server = self._goal_servers.get(bytes(goal_handle.goal_id.uuid))
            if server is None:
                return False
            name, instance = server
            return self._entities[name] is None or self._instances[name] != instance

    @contextmanager
    def _serving(self) -> Iterator[None]:
        with self._idle:
            self._active += 1
        try:
            yield
        finally:
            with self._idle:
                self._active -= 1
                self._idle.notify_all()

    def _delay_answer(self) -> None:
        gate = self.accept_gate
        if self.accept_delay_s <= 0 and gate is None:
            return
        with self._serving():
            if self.accept_delay_s > 0:
                self._closing.wait(self.accept_delay_s)
            self._wait_gate(gate)

    def _wait_gate(self, gate: threading.Event | None) -> None:
        """Wait until `gate` is set or the fake closes; at once when `gate` is None."""
        while gate is not None and not (gate.is_set() or self._closing.is_set()):
            gate.wait(_STEP_S)

    def _move_time(self, targets: Mapping[str, float], default_s: float) -> float:
        """How long a joint move to `targets` takes: by joint_speed_rad_s, else `default_s`."""
        speed = self.joint_speed_rad_s
        if speed is None:
            return default_s
        with self._lock:
            change = max(
                (abs(target - self._joints[joint]) for joint, target in targets.items()),
                default=0.0,
            )
        return change / speed

    def _arm_stopped(self, goal_handle) -> Callable[[], bool]:
        """True once the goal is cancelled, or once a joint cancel or a stop event stops the arms."""
        with self._lock:
            stops = self._arm_stops

        def stopped() -> bool:
            with self._lock:
                stopped_since = self._arm_stops != stops
            return goal_handle.is_cancel_requested or stopped_since

        return stopped

    def _claim_gripper(self, group: str) -> Callable[[], bool]:
        """Count a command for `group`; the check returned is true once a newer one came."""
        with self._lock:
            self._gripper_commands[group] += 1
            command = self._gripper_commands[group]

        def superseded() -> bool:
            with self._lock:
                return self._gripper_commands[group] != command

        return superseded

    def _hold(self, stopped: Callable[[], bool]) -> None:
        while not (stopped() or self._closing.is_set()):
            self._closing.wait(_STEP_S)

    def _move(
        self,
        targets: dict[str, float],
        duration_s: float,
        stopped: Callable[[], bool],
        hold: threading.Event | None = None,
    ) -> bool:
        """Move `targets` in steps over `duration_s`. False when `stopped` ended it part way.

        With `hold`, it takes two steps at least and waits after the first
        until `hold` is set or `stopped` is true.
        """
        with self._lock:
            start = {joint: self._joints[joint] for joint in targets}
        steps = max(1 if hold is None else 2, round(duration_s / _STEP_S))
        for i in range(1, steps + 1):
            if i == 2 and hold is not None:
                self._hold(lambda: hold.is_set() or stopped())
            if stopped():
                return False
            if self._closing.wait(duration_s / steps):
                return False
            fraction = i / steps
            with self._lock:
                # Checked again with the lock held, so no step lands after a
                # stop or a newer command.
                if stopped():
                    return False
                for joint, target in targets.items():
                    # The last step sets the target itself, which the
                    # interpolation can miss by a rounding error.
                    self._joints[joint] = (
                        target if i == steps else start[joint] + (target - start[joint]) * fraction
                    )
        return True

    def _finish(self, goal_handle, result, group: str, success: bool, status: str):
        result.success = success
        result.status = status
        self.results.append((group, status))
        if success:
            goal_handle.succeed()
        elif status == "Canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()
        # The goal has ended; its result goes when this returns. A result sent
        # through a server that vanish() destroyed would fail, so it waits
        # for the fake to close.
        delay = math.inf if self._server_gone(goal_handle) else self.result_delay_s
        if delay > 0:
            self._closing.wait(None if math.isinf(delay) else delay)
        if isinstance(result, SetJointPosition.Result | GoToPose.Result):
            self._wait_gate(self.result_gate)
        return result


def main() -> None:
    rclpy.init()
    node = FakeCommander()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    node.get_logger().info(
        "Fake commander is up (set_joint_position, control_gripper, go_to_pose, "
        "get_joint_states, trajectory_execution_event)."
    )
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
