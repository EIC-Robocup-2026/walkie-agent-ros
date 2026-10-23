"""The arm atomics and the drive check against FakeCommander over ROS.

The fake serves the commander's names and types on the test domain (Fast
DDS, ADR-0019), with the groups of config/robot_model.toml, and the atomics
reach it through a real CommanderClient in a real Toolbox. So these tests
cover the answers the LLM reads for every outcome the commander can give.
"""

import math
import random
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest
from action_msgs.msg import GoalStatus
from langchain_core.messages import AIMessage
from rai.communication.hri_connector import HRIMessage
from rai.communication.ros2 import ROS2Connector
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from conftest import PROJECT_ROOT
from scripted_llm import ScriptedChatModel, tool_call
from walkie_agent.agent import WalkieAgent
from walkie_agent.arm import commander as commander_module
from walkie_agent.arm.commander import CommanderClient
from walkie_agent.arm.poses import (
    GRIPPER_GROUPS,
    RIGHT_JOINT2,
    ArmPoses,
    load_arm_poses,
    load_robot_model,
)
from walkie_agent.arm.state import (
    NOT_SAFE_SENTENCE,
    SAFE_SENTENCE,
    UNKNOWN_SENTENCE,
    ArmsState,
    ArmState,
)
from walkie_agent.sim.fake_commander import FakeCommander
from walkie_agent.toolbox import StepError, Toolbox, ToolOutput
from walkie_agent.tools import FORBIDDEN_NAMES
from walkie_agent.tools import navigation as navigation_module
from walkie_agent.tools.atomic import _arm as arm_shared
from walkie_agent.tools.atomic._arm import FINGER_JOINTS
from walkie_agent.tools.navigation import MoveRelativeTool, NavigateToLocationTool

MODEL_PATH = PROJECT_ROOT / "config" / "robot_model.toml"
POSES_PATH = PROJECT_ROOT / "config" / "arm_poses.toml"

# The answer when a stop or a reset interrupts a move of both arms, and the
# commander accepts the cancel.
MOVE_STOPPED = (
    "A stop or a reset interrupted the move of both_arms; the arms were told to stop. They "
    "may take a moment to stop, so call get_arm_state before relying on where they are. "
    f"{UNKNOWN_SENTENCE}"
)
# What follows the client's message on a stop or a reset when the commander
# may still be moving an arm or the gripper.
WAIT_THEN_CHECK = "Wait a few seconds, then call get_arm_state."
ARMS_UNKNOWN = "Arm state unknown: the commander did not answer."
LEFT_OUT = (
    "Not driving: the left arm is not in a pose that is safe for driving. Move it with "
    "move_arm_to_pose first, for example 'home'."
)


def wait_until(predicate, timeout=3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture(scope="module")
def model():
    return load_robot_model(MODEL_PATH)


@pytest.fixture(scope="module")
def poses():
    return load_arm_poses(POSES_PATH, MODEL_PATH)


@pytest.fixture
def fake(model):
    node = FakeCommander(model, move_time_s=0.2, gripper_time_s=0.05)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    executor.shutdown()
    node.destroy_node()
    thread.join(timeout=2)


@pytest.fixture
def sim_connector():
    """A connector on sim time, as main.py's --use-sim-time makes it."""
    conn = ROS2Connector(
        node_name=f"test_sim_connector_{random.randint(0, 1_000_000)}", use_sim_time=True
    )
    yield conn
    conn.shutdown()


@contextmanager
def arm_toolbox(connector, poses: ArmPoses, wait: bool = True) -> Iterator[Toolbox]:
    """A Toolbox whose only backend is a CommanderClient on `connector`."""
    client = CommanderClient(connector, poses)
    try:
        if wait:
            assert wait_until(client.available, 5.0), "The client never discovered the fake."
        yield Toolbox(connector, {"arm": client})
    finally:
        client.destroy()


@pytest.fixture
def toolbox(fake, connector, poses):
    with arm_toolbox(connector, poses) as box:
        yield box


@pytest.fixture
def sim_toolbox(fake, sim_connector, poses):
    with arm_toolbox(sim_connector, poses) as box:
        yield box


def call(toolbox: Toolbox, name: str, **kwargs) -> ToolOutput:
    [tool] = [t for t in toolbox.tools() if t.name == name]
    return tool.call(**kwargs)


def error_of(toolbox: Toolbox, name: str, **kwargs) -> StepError:
    with pytest.raises(StepError) as info:
        call(toolbox, name, **kwargs)
    return info.value


def client_of(toolbox: Toolbox) -> CommanderClient:
    return toolbox.ctx.backend("arm")


def targets(poses: ArmPoses, pose: str, *arms: str) -> dict[str, float]:
    table = poses.poses[pose].tables
    return {joint: value for arm in arms for joint, value in table[arm].items()}


def in_pose(fake: FakeCommander, wanted: dict[str, float], tolerance=1e-6) -> bool:
    joints = fake.joints()
    return all(abs(joints[joint] - value) <= tolerance for joint, value in wanted.items())


def pose_line(poses: ArmPoses, name: str, arms: str, safe: bool) -> str:
    driving = "safe" if safe else "not safe"
    return f"{name}: {arms}; {driving} for driving. {poses.poses[name].description}"


# -- list_arm_poses -----------------------------------------------------------------


def test_list_arm_poses_offers_only_the_poses_tried_on_the_robot(toolbox, poses):
    out = call(toolbox, "list_arm_poses")

    assert out.poses == ["home", "hands_up", "standby"]
    assert out.summary.splitlines() == [
        pose_line(poses, "home", "left, right or both", True),
        pose_line(poses, "hands_up", "left, right or both", False),
        pose_line(poses, "standby", "left, right or both", False),
        # Nothing had read the arms, so the tool read them: the fake starts in home.
        SAFE_SENTENCE,
    ]


def test_list_arm_poses_in_the_simulation_adds_the_untried_poses(sim_toolbox, poses):
    out = call(sim_toolbox, "list_arm_poses")

    # pre_place, basket and basket_hold have offered = false everywhere.
    assert out.poses == ["home", "hands_up", "standby", "tray", "receive", "carry"]
    lines = out.summary.splitlines()
    assert lines[4] == pose_line(poses, "receive", "left arm only", False)
    assert lines[5] == pose_line(poses, "carry", "left or right, one arm at a time", True)
    assert lines[-1] == SAFE_SENTENCE


def test_without_a_commander_list_arm_poses_still_answers_at_once(connector, poses):
    with arm_toolbox(connector, poses, wait=False) as box:
        start = time.monotonic()
        out = call(box, "list_arm_poses")

    assert time.monotonic() - start < 0.5
    assert out.summary.splitlines()[-1] == UNKNOWN_SENTENCE


# -- move_arm_to_pose: success --------------------------------------------------------


def test_a_move_of_both_arms_sends_the_pose_in_the_commanders_order(toolbox, fake, poses):
    # The LLM may write the name as it would say it.
    out = call(toolbox, "move_arm_to_pose", pose="Hands up")

    assert out.summary == f"Moved to 'hands_up'. {NOT_SAFE_SENTENCE}"
    assert (out.pose, out.arm, out.status, out.travel_safe) == (
        "hands_up", "both", "Success", False
    )
    assert out.max_error_rad < 1e-6
    wanted = targets(poses, "hands_up", "left", "right")
    order = fake.group_joints("both_arms")
    assert fake.joint_goals == [("both_arms", tuple(wanted[joint] for joint in order))]
    assert in_pose(fake, wanted)

    back = call(toolbox, "move_arm_to_pose", pose="home")
    assert back.summary == f"Moved to 'home'. {SAFE_SENTENCE}"
    assert back.travel_safe is True


def test_a_move_follows_a_joint_order_unlike_the_srdfs(toolbox, fake, poses):
    fake.reverse_joint_order = True

    out = call(toolbox, "move_arm_to_pose", pose="standby", arm="left")

    assert out.summary == f"Moved the left arm to 'standby'. {NOT_SAFE_SENTENCE}"
    assert in_pose(fake, targets(poses, "standby", "left"))


def test_a_one_arm_pose_moves_only_that_arm(sim_toolbox, fake, poses):
    out = call(sim_toolbox, "move_arm_to_pose", pose="receive", arm="left")

    assert out.summary == f"Moved the left arm to 'receive'. {NOT_SAFE_SENTENCE}"
    [(group, positions)] = fake.joint_goals
    assert (group, len(positions)) == ("left_arm", 7)
    assert in_pose(fake, targets(poses, "receive", "left"))
    assert in_pose(fake, targets(poses, "home", "right"))

    carry = call(sim_toolbox, "move_arm_to_pose", pose="carry", arm="left")
    # carry is travel-safe, and the right arm is still in home.
    assert carry.summary == f"Moved the left arm to 'carry'. {SAFE_SENTENCE}"


# -- move_arm_to_pose: refused before anything moves ------------------------------------


@pytest.mark.parametrize(
    ("pose", "use_sim"),
    [("wave", False), ("receive", False), ("carry", False), ("pre_place", True)],
)
def test_an_unknown_or_hidden_pose_is_not_found(pose, use_sim, toolbox, sim_toolbox, fake):
    box = sim_toolbox if use_sim else toolbox
    offered = (
        "home, hands_up, standby, tray, receive, carry" if use_sim else "home, hands_up, standby"
    )

    error = error_of(box, "move_arm_to_pose", pose=pose)

    # A hidden pose gets the same answer as a pose that does not exist.
    assert error.code == "NOT_FOUND"
    assert error.message == f"No pose named {pose!r}. The poses are: {offered}. {SAFE_SENTENCE}"
    assert fake.joint_goals == []


@pytest.mark.parametrize(
    ("pose", "arm", "message"),
    [
        ("receive", "right", "'receive' is for the left arm only."),
        ("receive", "both", "'receive' is for the left arm only."),
        ("carry", "both", "'carry' is for one arm at a time: left or right."),
    ],
)
def test_an_arm_the_pose_does_not_take_is_refused(sim_toolbox, fake, pose, arm, message):
    error = error_of(sim_toolbox, "move_arm_to_pose", pose=pose, arm=arm)

    assert (error.code, error.message) == ("REFUSED", f"{message} {SAFE_SENTENCE}")
    assert fake.joint_goals == []


def test_right_joint2_fixed_drops_the_joint_at_zero_and_refuses_it_elsewhere(
    sim_toolbox, fake, poses
):
    fake.right_joint2_fixed = True

    out = call(sim_toolbox, "move_arm_to_pose", pose="hands_up")

    assert out.summary == f"Moved to 'hands_up'. {NOT_SAFE_SENTENCE}"
    [(group, positions)] = fake.joint_goals
    assert (group, len(positions)) == ("both_arms", 13)
    wanted = targets(poses, "hands_up", "left", "right")
    del wanted[RIGHT_JOINT2]
    assert in_pose(fake, wanted)

    # carry's right table needs the joint at 0.5 rad, which the fixed joint cannot reach.
    error = error_of(sim_toolbox, "move_arm_to_pose", pose="carry", arm="right")
    assert error.code == "REFUSED"
    assert error.message == (
        "'carry' cannot move the right arm while walkie-ros runs with right_joint2_fixed: "
        f"the pose needs {RIGHT_JOINT2} at 0.50 rad, and that joint is held at 0. Use the "
        f"left arm or another pose. {NOT_SAFE_SENTENCE}"
    )
    assert len(fake.joint_goals) == 1
    # The left table has no such joint.
    assert call(sim_toolbox, "move_arm_to_pose", pose="carry", arm="left").pose == "carry"


# -- move_arm_to_pose: the commander's failures ---------------------------------------


def test_a_planning_failure_says_the_arms_did_not_move(toolbox, fake):
    fake.plan_failure = True

    error = error_of(toolbox, "move_arm_to_pose", pose="standby")

    assert error.code == "SKILL_FAILED"
    # The arms were read again after the failure, so the state is known.
    assert error.message == (
        "Could not move to 'standby': MoveIt found no path (Planning Failed). The arms did "
        f"not move. {SAFE_SENTENCE}"
    )
    assert error.data == {"group": "both_arms", "status": "Planning Failed"}


def test_an_execution_failure_reads_where_the_arms_stopped(toolbox, fake):
    fake.execution_failure = True

    error = error_of(toolbox, "move_arm_to_pose", pose="hands_up")

    assert error.code == "SKILL_FAILED"
    # Half way to hands_up is in no named pose.
    assert error.message == (
        "Could not move to 'hands_up': The motion stopped before the target (Execution "
        f"Failed). The arms may have stopped part way. {NOT_SAFE_SENTENCE}"
    )


def test_a_rejected_goal_is_internal_with_the_group_and_count(toolbox, fake):
    fake.reject_goals = True

    error = error_of(toolbox, "move_arm_to_pose", pose="hands_up", arm="right")

    assert error.code == "INTERNAL"
    assert error.message.startswith(
        "Could not move to 'hands_up': The commander rejected the goal for right_arm with "
        "7 positions."
    )
    assert error.message.endswith(SAFE_SENTENCE)
    assert error.data == {"group": "right_arm", "count": 7}


def test_a_joint_left_off_its_target_is_a_postcondition_failure(toolbox, fake):
    fake.joint_errors = {"openarm_left_joint4": 0.31}

    error = error_of(toolbox, "move_arm_to_pose", pose="standby")

    assert error.code == "POSTCONDITION"
    assert error.message == (
        "The commander reported success, but openarm_left_joint4 is 0.31 rad from "
        f"'standby'. Treat this as a failure. {NOT_SAFE_SENTENCE}"
    )
    assert error.data["joint"] == "openarm_left_joint4"
    assert error.data["error_rad"] == pytest.approx(0.31)


def test_a_move_whose_reading_afterwards_fails_is_a_postcondition_failure(toolbox, fake, poses):
    # The joint order is known from an earlier read, so the move is sent.
    client_of(toolbox).joint_names("both_arms", threading.Event())
    fake.read_failure = "Failed to get current robot state"

    error = error_of(toolbox, "move_arm_to_pose", pose="standby")

    assert error.code == "POSTCONDITION"
    assert error.message == (
        "The commander reported success, but the arms could not be checked against 'standby': "
        "The commander could not read the joints of both_arms: Failed to get current robot "
        f"state. Treat this as a failure. {UNKNOWN_SENTENCE}"
    )
    assert error.data == {"pose": "standby", "status": "Success", "read_error": "UNAVAILABLE"}
    assert in_pose(fake, targets(poses, "standby", "left", "right"))


def test_a_move_whose_reading_afterwards_lacks_a_joint_is_a_postcondition_failure(toolbox, fake):
    client_of(toolbox).joint_names("both_arms", threading.Event())
    fake.missing_joints = {"openarm_left_joint4"}

    error = error_of(toolbox, "move_arm_to_pose", pose="standby")

    assert error.code == "POSTCONDITION"
    # The left arm is unknown, and the right one is read in standby.
    assert error.message == (
        "The commander reported success, but its reading lacks a usable position for "
        "openarm_left_joint4, so the arms cannot be checked against 'standby'. Treat this as a "
        f"failure. {NOT_SAFE_SENTENCE}"
    )
    assert error.data == {
        "pose": "standby",
        "status": "Success",
        "missing": ["openarm_left_joint4"],
    }


def test_a_joint_read_as_not_a_number_after_a_move_fails_the_check(toolbox, fake):
    # A NaN passes every tolerance check, since every comparison with it is
    # false, so the move must not count as arrived.
    fake.joint_errors = {"openarm_left_joint1": math.nan}

    error = error_of(toolbox, "move_arm_to_pose", pose="standby")

    assert error.code == "POSTCONDITION"
    assert error.message == (
        "The commander reported success, but the arms could not be checked against 'standby': "
        "The commander's reading of both_arms is not usable: openarm_left_joint1 = nan. Treat "
        f"this as a failure. {UNKNOWN_SENTENCE}"
    )
    assert error.data == {"pose": "standby", "status": "Success", "read_error": "INTERNAL"}


def test_plan_only_never_says_the_arms_moved(toolbox, fake):
    fake.plan_only = True

    error = error_of(toolbox, "move_arm_to_pose", pose="standby")

    # The joint check catches it, and the answer says why nothing moved.
    assert error.code == "POSTCONDITION"
    assert error.message == (
        "The commander runs in plan-only mode, so it moved nothing, and openarm_left_joint4 is "
        "1.82 rad from 'standby'. Treat this as a failure. No arm can move until the commander "
        f"restarts without plan_only, so tell the person instead of trying again. {SAFE_SENTENCE}"
    )
    assert error.data["status"] == "Plan Only"

    # Already in the pose, the check passes, and the answer still says nothing moved.
    out = call(toolbox, "move_arm_to_pose", pose="home")
    assert (out.summary, out.status) == (
        "Already at 'home'. The commander runs in plan-only mode, so it moved nothing. "
        f"{SAFE_SENTENCE}",
        "Plan Only",
    )
    left = call(toolbox, "move_arm_to_pose", pose="home", arm="left")
    assert left.summary == (
        "The left arm is already at 'home'. The commander runs in plan-only mode, so it moved "
        f"nothing. {SAFE_SENTENCE}"
    )
    assert fake.results == [
        ("both_arms", "Plan Only"),
        ("both_arms", "Plan Only"),
        ("left_arm", "Plan Only"),
    ]


def test_a_move_that_runs_past_its_pose_timeout_times_out(fake, connector, poses):
    fake.never_answer = True
    quick = ArmPoses(
        poses={**poses.poses, "hands_up": replace(poses.poses["hands_up"], timeout_s=0.5)},
        grippers=poses.grippers,
    )

    with arm_toolbox(connector, quick) as box:
        start = time.monotonic()
        error = error_of(box, "move_arm_to_pose", pose="hands_up")
        elapsed = time.monotonic() - start

    assert error.code == "TIMEOUT"
    assert error.message == (
        "Could not move to 'hands_up': The move of both_arms did not finish within 0.5 s; "
        f"the arms were told to stop. {UNKNOWN_SENTENCE}"
    )
    assert elapsed < 2.0
    assert fake.cancel_requests == 1


def accepted_handle(client: CommanderClient):
    """The goal handle of the client's running goal once the commander accepted it, else None."""
    goal = client._goal
    if goal is None or goal.result is None:
        return None
    return goal.sent.result()


def interrupt_for(
    outcome: str, toolbox: Toolbox, fake: FakeCommander, monkeypatch, gripper: bool
) -> threading.Thread:
    """A started thread that interrupts the next goal so that its cancel gets `outcome`.

    `outcome` is a value of StepError.data["cancel"] other than "accepted",
    which the fake gives by default. The goal is the next arm goal, or the
    next gripper goal with `gripper`.
    """
    client = client_of(toolbox)
    name = "control_gripper" if gripper else "set_joint_position"
    action = client._control_gripper if gripper else client._set_joint_position
    # Long enough to be interrupted while it runs.
    fake.move_time_s = fake.gripper_time_s = 2.0
    if outcome == "refused":
        fake.reject_cancels = True
    elif outcome == "no answer":
        monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.2)
        fake.cancel_delay_s = 0.5
    elif outcome == "pending":
        # The send stays unanswered until the fake closes.
        fake.accept_gate = threading.Event()
    elif outcome == "ended":
        # The goal ends at once, and its result is held.
        fake.move_time_s = fake.gripper_time_s = 0.05
        fake.result_delay_s = math.inf

    def ready() -> bool:
        if outcome == "pending":
            return bool(fake.gripper_goals if gripper else fake.joint_goals)
        handle = accepted_handle(client)
        if outcome == "ended":
            # The client saw the goal end on the status topic, and has no result.
            return handle is not None and handle.status == GoalStatus.STATUS_SUCCEEDED
        return handle is not None

    def gave_up() -> bool:
        goal = client._goal
        return goal is None or goal.gave_up_at is not None

    def interrupt() -> None:
        if wait_until(ready, 5.0) and outcome == "unknown":
            # The commander restarts, and its new server does not know the goal.
            fake.vanish(name)
            wait_until(lambda: not action.server_is_ready(), 5.0)
            fake.come_back()
            wait_until(action.server_is_ready, 5.0)
        toolbox.interrupt()
        if outcome == "pending" and wait_until(gave_up, 5.0):
            # The commander answers at last, with a rejection, so the goal never runs.
            fake.reject_goals = True
            fake.accept_gate.set()

    thread = threading.Thread(target=interrupt)
    thread.start()
    return thread


def test_an_interrupt_mid_move_cancels_the_goal(toolbox, fake, poses):
    fake.move_time_s = 5.0
    interrupted_at: list[float] = []

    def interrupt_once_moving() -> None:
        assert wait_until(lambda: fake.joint_goals, 5.0)
        time.sleep(0.3)
        interrupted_at.append(time.monotonic())
        toolbox.interrupt()

    thread = threading.Thread(target=interrupt_once_moving)
    thread.start()
    error = error_of(toolbox, "move_arm_to_pose", pose="hands_up")
    returned_at = time.monotonic()
    thread.join()

    # The client's message, and what to do about the cancel it accepted.
    assert (error.code, error.message, error.data) == (
        "CANCELLED", MOVE_STOPPED, {"cancel": "accepted"}
    )
    assert returned_at - interrupted_at[0] < 1.0
    assert fake.cancel_requests == 1
    # The commander stops the arms and ends the goal as failed.
    assert wait_until(lambda: fake.results == [("both_arms", "Execution Failed")], 1.0)
    assert not in_pose(fake, targets(poses, "hands_up", "left", "right"), tolerance=0.05)
    assert client_of(toolbox).last_state.sentence() == UNKNOWN_SENTENCE


@pytest.mark.parametrize(
    ("outcome", "answer"),
    [
        (
            "refused",
            f"the commander refused the cancel, so an arm may still be moving. {WAIT_THEN_CHECK}",
        ),
        (
            "unknown",
            "the commander does not know the goal, so an arm may still be moving. "
            f"{WAIT_THEN_CHECK}",
        ),
        (
            "no answer",
            "the commander did not answer the cancel, so an arm may still be moving. "
            f"{WAIT_THEN_CHECK}",
        ),
        (
            "pending",
            f"if the commander accepts it later, it is cancelled at once. {WAIT_THEN_CHECK}",
        ),
        # Nothing to wait for.
        ("ended", "the goal had already ended."),
    ],
    ids=["refused", "unknown", "no answer", "pending", "ended"],
)
def test_an_interrupted_move_says_how_the_commander_answered_the_cancel(
    toolbox, fake, monkeypatch, outcome, answer
):
    # Read first, so the answer shows that the move made the arms unknown.
    call(toolbox, "get_arm_state")
    thread = interrupt_for(outcome, toolbox, fake, monkeypatch, gripper=False)

    error = error_of(toolbox, "move_arm_to_pose", pose="hands_up")
    thread.join(timeout=10)

    assert (error.code, error.message, error.data) == (
        "CANCELLED",
        f"A stop or a reset interrupted the move of both_arms; {answer} {UNKNOWN_SENTENCE}",
        {"cancel": outcome},
    )


def test_an_interrupt_before_the_goal_is_sent_says_the_arms_did_not_move(toolbox, fake):
    # The read leaves the joint order known, so the move gets as far as its send.
    call(toolbox, "get_arm_state")
    toolbox.interrupt()

    error = error_of(toolbox, "move_arm_to_pose", pose="hands_up")

    assert (error.code, error.message, error.data) == (
        "CANCELLED", f"Stopped by reset before the arms moved. {SAFE_SENTENCE}", {}
    )
    assert fake.joint_goals == []


def test_a_move_names_the_action_when_its_server_is_not_ready(toolbox, fake, monkeypatch):
    monkeypatch.setattr(commander_module, "READY_WAIT_S", 0.2)
    # The service answers, so only the action is missing, as when the
    # commander's node is still starting its servers.
    action = client_of(toolbox)._set_joint_position
    monkeypatch.setattr(action, "server_is_ready", lambda: False)

    error = error_of(toolbox, "move_arm_to_pose", pose="hands_up")

    assert error.code == "UNAVAILABLE"
    assert error.message == (
        "Could not move to 'hands_up': The commander's /set_joint_position action is not "
        f"available. Is bimanual_commander running? {UNKNOWN_SENTENCE}"
    )
    assert error.data == {"name": "/set_joint_position action"}
    assert fake.joint_goals == []


def test_without_a_commander_every_call_is_unavailable(connector, poses, monkeypatch):
    monkeypatch.setattr(commander_module, "READY_WAIT_S", 0.2)
    hint = "Is bimanual_commander running?"

    with arm_toolbox(connector, poses, wait=False) as box:
        start = time.monotonic()
        errors = [
            error_of(box, "move_arm_to_pose", pose="home"),
            error_of(box, "set_gripper", arm="left", state="open"),
            error_of(box, "get_arm_state"),
        ]

    assert time.monotonic() - start < 2.0
    assert [(e.code, e.message) for e in errors] == [
        (
            "UNAVAILABLE",
            "Could not move to 'home': The commander's /get_joint_states service is not "
            f"available. {hint} {UNKNOWN_SENTENCE}",
        ),
        (
            "UNAVAILABLE",
            "Could not move the left gripper: The commander's /control_gripper action is not "
            f"available. {hint} {UNKNOWN_SENTENCE}",
        ),
        (
            "UNAVAILABLE",
            f"The commander's /get_joint_states service is not available. {hint} "
            f"{UNKNOWN_SENTENCE}",
        ),
    ]


# -- set_gripper --------------------------------------------------------------------


def fail_gripper(monkeypatch, fake: FakeCommander, group: str) -> None:
    """End every goal of `group` that reaches its position ABORTED "Failed".

    As when the gripper controller fails: the commander then reports success
    false, whatever the fingers did.
    """
    finish = fake._finish

    def failed(goal_handle, result, goal_group, success, status):
        if goal_group == group and status == "Success":
            success, status = False, "Failed"
        return finish(goal_handle, result, goal_group, success, status)

    monkeypatch.setattr(fake, "_finish", failed)


def test_open_reports_only_that_it_moved(toolbox, fake):
    # An object between the fingers: on an open the commander still reports
    # grasped false and a 0.0 gap, which must not read as "holding nothing".
    fake.object_widths["left_gripper"] = 0.02

    out = call(toolbox, "set_gripper", arm="left", state="open")

    assert out.summary == f"Opened the left gripper. {SAFE_SENTENCE}"
    assert out.holding == {"left": None}
    assert fake.gripper_goals == [("left_gripper", 0.044)]


def test_half_on_both_sends_left_then_right(toolbox, fake):
    out = call(toolbox, "set_gripper", arm="both", state="half")

    assert out.summary == f"Set both grippers half open. {SAFE_SENTENCE}"
    assert out.holding == {"left": None, "right": None}
    assert fake.gripper_goals == [("left_gripper", 0.036), ("right_gripper", 0.036)]


def test_an_open_the_commander_reports_failed_is_a_failure(toolbox, fake, monkeypatch):
    fail_gripper(monkeypatch, fake, "left_gripper")

    error = error_of(toolbox, "set_gripper", arm="left", state="open")

    assert error.code == "SKILL_FAILED"
    assert error.message == (
        f"The left gripper did not open fully (the commander reported 'Failed'). {SAFE_SENTENCE}"
    )
    assert error.data == {"arm": "left", "state": "open", "holding": {"left": None}}


def test_when_the_second_gripper_fails_the_answer_says_what_each_did(toolbox, fake):
    # Something between the right fingers stops them short of half open.
    fake.object_widths["right_gripper"] = 0.04

    error = error_of(toolbox, "set_gripper", arm="both", state="half")

    assert error.code == "SKILL_FAILED"
    assert error.message == (
        "Set the left gripper half open. The right gripper did not reach half open (the "
        f"commander reported 'Failed'). {SAFE_SENTENCE}"
    )
    assert error.data["holding"] == {"left": None, "right": None}
    assert fake.gripper_goals == [("left_gripper", 0.036), ("right_gripper", 0.036)]


def test_a_second_gripper_goal_the_commander_rejects_keeps_what_the_first_did(
    toolbox, fake, monkeypatch
):
    finish = fake._finish

    def reject_from_now_on(goal_handle, result, group, success, status):
        # Every goal after the left one is rejected.
        fake.reject_goals = True
        return finish(goal_handle, result, group, success, status)

    monkeypatch.setattr(fake, "_finish", reject_from_now_on)

    error = error_of(toolbox, "set_gripper", arm="both", state="close")

    # Nothing read the arms, and a failed call does not read them.
    assert (error.code, error.message) == (
        "INTERNAL",
        "Closed the left gripper. It is holding nothing. Could not move the right gripper: The "
        f"commander rejected the goal for right_gripper. {UNKNOWN_SENTENCE}",
    )
    assert fake.gripper_goals == [("left_gripper", 0.0), ("right_gripper", 0.0)]


def test_a_close_on_an_object_holds_it(toolbox, fake):
    fake.object_widths["left_gripper"] = 0.02

    out = call(toolbox, "set_gripper", arm="left", state="close")

    # The fingers stall on the object, so the goal fails; grasped is the answer.
    assert out.summary == f"Closed the left gripper. It is holding something. {SAFE_SENTENCE}"
    assert out.holding == {"left": True}
    assert fake.gripper_goals == [("left_gripper", 0.0)]


def test_a_close_on_nothing_holds_nothing(toolbox):
    out = call(toolbox, "set_gripper", arm="right", state="close")

    assert out.summary == f"Closed the right gripper. It is holding nothing. {SAFE_SENTENCE}"
    assert out.holding == {"right": False}


def test_a_stuck_gripper_fails_and_the_other_still_closes(toolbox, fake):
    # The fingers stop near the open position, inside the commander's grasp
    # range: grasped is true, but nothing that thin is held that wide.
    fake.object_widths["left_gripper"] = 0.043

    error = error_of(toolbox, "set_gripper", arm="left", state="close")

    assert (error.code, error.message) == (
        "SKILL_FAILED",
        f"The left gripper did not close (gap 0.043 m). {SAFE_SENTENCE}",
    )

    both = error_of(toolbox, "set_gripper", arm="both", state="close")
    assert both.message == (
        "The left gripper did not close (gap 0.043 m). Closed the right gripper. It is "
        f"holding nothing. {SAFE_SENTENCE}"
    )
    assert both.data["holding"] == {"left": None, "right": False}
    assert fake.gripper_goals[-2:] == [("left_gripper", 0.0), ("right_gripper", 0.0)]


def test_grasp_detect_min_below_zero_reads_an_empty_close_as_held(toolbox, fake):
    # The atomic trusts the commander's detection parameters, so a band
    # that includes a closed gripper reads every close as a grasp.
    fake.set_parameters([Parameter("grasp_detect_min", value=-1.0)])

    out = call(toolbox, "set_gripper", arm="left", state="close")

    assert out.summary == f"Closed the left gripper. It is holding something. {SAFE_SENTENCE}"
    assert out.holding == {"left": True}


@pytest.mark.parametrize(
    ("width", "held"), [(0.02, "something"), (None, "nothing")], ids=["object", "empty"]
)
def test_a_close_reported_reached_without_a_measurement_is_judged_from_a_read(
    toolbox, fake, monkeypatch, width, held
):
    # A threshold below every goal: the commander runs no grasp detection and
    # reports grasped false with a gap of 0.0, as with grasp_detect_enable off.
    fake.set_parameters([Parameter("grasp_close_threshold", value=-1.0)])
    fake.object_widths["left_gripper"] = width
    finish = fake._finish

    def reached(goal_handle, result, group, success, status):
        # A controller that allows stalling reports fingers stopped on an
        # object as having reached the position.
        if group == "left_gripper":
            success, status = True, "Success"
        return finish(goal_handle, result, group, success, status)

    monkeypatch.setattr(fake, "_finish", reached)

    out = call(toolbox, "set_gripper", arm="left", state="close")

    # With an object, the gap of 0.0 does not agree with the read, so the band decides.
    assert out.summary == f"Closed the left gripper. It is holding {held}. {SAFE_SENTENCE}"
    assert out.holding == {"left": held == "something"}


# set_gripper's answer after a close of the left gripper that the commander
# reports failed, with grasped false and a gap of 0.0.
LEFT_CLOSE_UNMEASURED = (
    "The commander reported that the left gripper's close failed and gave no measurement, as "
    "when another command replaced it, so whether the gripper holds something is unknown. "
    "Call get_arm_state to see how far it is open."
)


@pytest.mark.parametrize("width", [0.02, 0.043], ids=["object", "barely moved"])
def test_a_close_reported_failed_without_a_measurement_cannot_tell(toolbox, fake, width):
    # No grasp detection, and the fingers stall on the object, so the
    # commander reports the close failed with grasped false and a gap of
    # 0.0, as it does for a close that a newer command superseded.
    fake.set_parameters([Parameter("grasp_close_threshold", value=-1.0)])
    fake.object_widths["left_gripper"] = width

    out = call(toolbox, "set_gripper", arm="left", state="close")

    # Not a failure of the step: the close ended as the commander allows.
    assert out.summary == f"{LEFT_CLOSE_UNMEASURED} {SAFE_SENTENCE}"
    assert out.holding == {"left": None}
    # The fingers may show a newer command, so they are not read.
    assert "left_gripper" not in fake.joint_state_requests


def test_a_close_that_another_node_supersedes_cannot_tell(toolbox, fake):
    fake.gripper_time_s = 1.0
    finger = FINGER_JOINTS["left"]
    opened = fake.joints()[finger]

    def supersede_mid_ramp() -> None:
        if wait_until(lambda: fake.joints()[finger] < opened, 5.0):
            # Half open: a read would find the fingers inside the band.
            fake.supersede("left_gripper", 0.036)

    thread = threading.Thread(target=supersede_mid_ramp)
    thread.start()
    out = call(toolbox, "set_gripper", arm="left", state="close")
    thread.join(timeout=10)

    assert fake.results == [("left_gripper", "Failed")]
    assert out.summary == f"{LEFT_CLOSE_UNMEASURED} {SAFE_SENTENCE}"
    assert out.holding == {"left": None}
    assert "left_gripper" not in fake.joint_state_requests


def test_a_close_followed_by_another_command_before_its_read_cannot_tell(
    toolbox, fake, monkeypatch
):
    # The commander measures the close on the object. Another call then
    # sets the gripper half open before the close's read, which would find
    # the fingers inside the band.
    fake.object_widths["left_gripper"] = 0.02
    client = client_of(toolbox)
    read_finger = arm_shared.read_finger

    def another_command_first(*args, **kwargs):
        client.control_gripper("left_gripper", 0.036, threading.Event())
        return read_finger(*args, **kwargs)

    monkeypatch.setattr(arm_shared, "read_finger", another_command_first)

    out = call(toolbox, "set_gripper", arm="left", state="close")

    assert out.summary == (
        "Another command for the left gripper came after the close, so whether the close held "
        "something is unknown. Call get_arm_state to see how far the gripper is open. "
        f"{SAFE_SENTENCE}"
    )
    assert out.holding == {"left": None}
    assert fake.gripper_goals == [("left_gripper", 0.0), ("left_gripper", 0.036)]
    # The read is refused before it is made.
    assert "left_gripper" not in fake.joint_state_requests


def test_a_close_whose_fingers_cannot_be_read_says_it_cannot_tell(toolbox, fake):
    fake.object_widths["right_gripper"] = 0.02
    fake.missing_joints = {"openarm_right_finger_joint1"}

    out = call(toolbox, "set_gripper", arm="both", state="close")

    # Both closes ended as the commander allows, so the call does not fail.
    assert out.summary == (
        "Closed the left gripper. It is holding nothing. The right gripper's fingers could not "
        "be read after the close, so whether it closed on something is unknown: The commander "
        "answered 0 joint names and 0 positions for right_gripper. Call get_arm_state to see "
        f"how far it closed. {SAFE_SENTENCE}"
    )
    assert out.holding == {"left": False, "right": None}


def test_an_interrupt_during_a_gripper_command_stops_it(toolbox, fake):
    fake.gripper_time_s = 1.0
    timer = threading.Timer(0.3, toolbox.interrupt)
    timer.start()

    error = error_of(toolbox, "set_gripper", arm="both", state="close")
    timer.join()

    assert error.code == "CANCELLED"
    assert error.message == (
        "A stop or a reset interrupted the left_gripper command; the commander accepted the "
        "cancel, but the gripper finishes its current motion. Wait a few seconds for it to "
        f"finish. {UNKNOWN_SENTENCE}"
    )
    assert error.data == {"cancel": "accepted"}
    # The right gripper was never sent.
    assert fake.gripper_goals == [("left_gripper", 0.0)]
    # The cancel reached the commander, which accepts it but finishes the
    # ramp; until then the command is outstanding and every call is refused.
    assert fake.cancel_requests == 1
    client = client_of(toolbox)
    assert client.busy() == "the left_gripper command from an earlier call"
    assert fake.results == []
    assert wait_until(lambda: client.busy() is None)
    assert fake.results == [("left_gripper", "Canceled")]


@pytest.mark.parametrize(
    ("outcome", "answer"),
    [
        (
            "refused",
            "the commander refused the cancel, so the gripper may still be moving. "
            f"{WAIT_THEN_CHECK}",
        ),
        (
            "unknown",
            "the commander does not know the command, so the gripper may still be moving. "
            f"{WAIT_THEN_CHECK}",
        ),
        (
            "no answer",
            "the commander did not answer the cancel, so the gripper may still be moving. "
            f"{WAIT_THEN_CHECK}",
        ),
        (
            "pending",
            "if the commander accepts it later, the gripper still makes that motion. "
            f"{WAIT_THEN_CHECK}",
        ),
        # Nothing to wait for.
        ("ended", "the command had already ended."),
    ],
    ids=["refused", "unknown", "no answer", "pending", "ended"],
)
def test_an_interrupted_gripper_command_says_how_the_commander_answered_the_cancel(
    toolbox, fake, monkeypatch, outcome, answer
):
    thread = interrupt_for(outcome, toolbox, fake, monkeypatch, gripper=True)

    error = error_of(toolbox, "set_gripper", arm="both", state="close")
    thread.join(timeout=10)

    assert (error.code, error.message, error.data) == (
        "CANCELLED",
        f"A stop or a reset interrupted the left_gripper command; {answer} {UNKNOWN_SENTENCE}",
        {"cancel": outcome},
    )
    # The right gripper was never sent.
    assert fake.gripper_goals == [("left_gripper", 0.0)]


def test_an_interrupt_before_a_gripper_command_is_sent_says_it_did_not_move(toolbox, fake):
    toolbox.interrupt()

    error = error_of(toolbox, "set_gripper", arm="left", state="close")

    assert (error.code, error.message, error.data) == (
        "CANCELLED", f"Stopped by reset before the left gripper moved. {UNKNOWN_SENTENCE}", {}
    )
    assert fake.gripper_goals == []


def test_an_interrupt_while_the_fingers_are_read_stops_the_call(toolbox, fake):
    fake.joint_states_delay_s = 2.0
    timer = threading.Timer(0.5, toolbox.interrupt)
    timer.start()

    start = time.monotonic()
    error = error_of(toolbox, "set_gripper", arm="both", state="close")
    elapsed = time.monotonic() - start
    timer.join()

    assert (error.code, error.message) == (
        "CANCELLED",
        "Stopped by reset before checking whether the left gripper holds something. "
        f"{UNKNOWN_SENTENCE}",
    )
    assert elapsed < 1.5
    # The right gripper was never sent.
    assert fake.gripper_goals == [("left_gripper", 0.0)]


# -- get_arm_state ------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["left", "right"])
def test_each_gripper_is_read_by_the_one_joint_of_its_group(model, arm):
    assert model.groups[GRIPPER_GROUPS[arm]] == (FINGER_JOINTS[arm],)


def test_get_arm_state_reads_the_arms_and_the_grippers(toolbox, fake):
    call(toolbox, "set_gripper", arm="left", state="close")

    out = call(toolbox, "get_arm_state")

    assert out.summary.splitlines() == [
        "Left arm: home, safe for driving.",
        "Right arm: home, safe for driving.",
        "Left gripper: closed (gap 0.000 m).",
        # Never commanded, but read: the fingers start open.
        "Right gripper: open (gap 0.044 m).",
        SAFE_SENTENCE,
    ]
    assert (out.poses, out.travel_safe) == ({"left": "home", "right": "home"}, True)
    assert out.grippers == {"left": "close", "right": "open"}
    assert out.gripper_gaps_m == pytest.approx({"left": 0.0, "right": 0.044})
    assert fake.joint_state_requests[-2:] == ["left_gripper", "right_gripper"]


def test_get_arm_state_names_a_pose_out_of_the_offered_ones_and_no_pose(toolbox, fake, poses):
    # receive is not offered outside the simulation, but an arm in it is
    # still reported as in it.
    fake.teleport(targets(poses, "receive", "left"))
    fake.teleport({"openarm_right_joint1": 0.7})
    # Stopped on an object, the left fingers are nearer the half position
    # than the closed one, and still read as closed.
    fake.object_widths["left_gripper"] = 0.02
    call(toolbox, "set_gripper", arm="left", state="close")
    call(toolbox, "set_gripper", arm="right", state="half")

    out = call(toolbox, "get_arm_state")

    assert out.summary.splitlines() == [
        "Left arm: receive, not safe for driving.",
        "Right arm: no named pose, not safe for driving.",
        "Left gripper: closed (gap 0.020 m).",
        "Right gripper: half open (gap 0.036 m).",
        NOT_SAFE_SENTENCE,
    ]
    assert out.poses == {"left": "receive", "right": None}
    assert out.grippers == {"left": "close", "right": "half"}


def test_a_gripper_that_cannot_be_read_falls_back_to_its_last_command(toolbox, fake):
    call(toolbox, "set_gripper", arm="left", state="close")
    # A finger read as not a number, and one the reading lacks.
    fake.teleport({"openarm_left_finger_joint1": math.nan})
    fake.missing_joints = {"openarm_right_finger_joint1"}

    out = call(toolbox, "get_arm_state")

    assert out.summary.splitlines() == [
        "Left arm: home, safe for driving.",
        "Right arm: home, safe for driving.",
        "Left gripper: could not be read; last command close.",
        "Right gripper: could not be read; last command unknown.",
        SAFE_SENTENCE,
    ]
    assert out.grippers == {"left": "close", "right": None}
    assert out.gripper_gaps_m == {"left": None, "right": None}


def test_get_arm_state_does_not_name_a_pose_from_a_joint_that_is_not_a_number(toolbox, fake):
    # Every other joint is at home, so a NaN that passed the tolerance check
    # would read as home.
    fake.teleport({"openarm_left_joint1": math.nan})

    error = error_of(toolbox, "get_arm_state")

    assert (error.code, error.message) == (
        "INTERNAL",
        "The commander's reading of both_arms is not usable: openarm_left_joint1 = nan. "
        f"{UNKNOWN_SENTENCE}",
    )


# -- The drive check (ADR-0025) ---------------------------------------------------------


def navigate(connector, locations, arm) -> NavigateToLocationTool:
    return NavigateToLocationTool(
        connector=connector, locations=locations, forbidden=FORBIDDEN_NAMES, arm=arm
    )


def tf_ready(connector) -> bool:
    try:
        connector.get_transform("map", "base_footprint", timeout_sec=0.1)
    except Exception:
        return False
    return True


def test_the_robot_does_not_drive_with_the_left_arm_in_receive(
    fake_walkie, toolbox, fake, connector, locations, poses
):
    fake.teleport(targets(poses, "receive", "left"))
    client = client_of(toolbox)

    assert navigate(connector, locations, client).invoke({"location": "kitchen"}) == LEFT_OUT
    assert wait_until(lambda: tf_ready(connector))
    relative = MoveRelativeTool(connector=connector, forbidden=FORBIDDEN_NAMES, arm=client)
    assert relative.invoke({"forward_m": 1.0}) == LEFT_OUT
    assert fake_walkie.pose() == (0.0, 0.0, 0.0)

    fake.teleport(targets(poses, "hands_up", "right"))
    assert navigate(connector, locations, client).invoke({"location": "kitchen"}) == (
        "Not driving: the arms are not in a pose that is safe for driving. Move them with "
        "move_arm_to_pose first, for example 'home'."
    )


def test_the_robot_drives_with_both_arms_travel_safe(
    fake_walkie, toolbox, fake, connector, locations, poses
):
    client = client_of(toolbox)
    assert navigate(connector, locations, client).invoke({"location": "kitchen"}) == (
        "Arrived at kitchen."
    )
    assert fake_walkie.pose()[:2] == pytest.approx((3.5, 1.2))

    # carry is travel-safe too: the robot guides the guest holding the bag.
    fake.teleport(targets(poses, "carry", "left"))
    assert navigate(connector, locations, client).invoke({"location": "bedroom"}) == (
        "Arrived at bedroom."
    )
    assert fake.joint_state_requests == ["both_arms", "both_arms"]


def test_without_a_commander_the_robot_drives_at_once_and_says_so(
    fake_walkie, connector, locations, poses
):
    not_found = "Arm state unknown: the commander was not found."
    with arm_toolbox(connector, poses, wait=False) as box:
        tool = navigate(connector, locations, client_of(box))
        start = time.monotonic()
        note = tool._check_arms()
        # No 3 s read: the client knows at once that the commander is missing.
        assert time.monotonic() - start < 0.5
        assert note == f" {not_found}"

        assert tool.invoke({"location": "kitchen"}) == f"Arrived at kitchen. {not_found}"
    assert fake_walkie.pose()[:2] == pytest.approx((3.5, 1.2))


def test_a_commander_that_does_not_answer_lets_the_robot_drive(
    fake_walkie, toolbox, fake, connector, locations, monkeypatch
):
    monkeypatch.setattr(navigation_module, "ARM_CHECK_TIMEOUT_S", 0.3)
    fake.joint_states_delay_s = 2.0

    result = navigate(connector, locations, client_of(toolbox)).invoke({"location": "kitchen"})

    assert result == f"Arrived at kitchen. {ARMS_UNKNOWN}"


def test_a_commander_that_cannot_read_the_joints_lets_the_robot_drive(
    fake_walkie, toolbox, fake, connector, locations
):
    fake.read_failure = "Failed to get current robot state"

    result = navigate(connector, locations, client_of(toolbox)).invoke({"location": "kitchen"})

    assert result == (
        "Arrived at kitchen. Arm state unknown: the commander gave no reading of the arms."
    )


@pytest.mark.parametrize(
    ("joints", "missing", "detail"),
    [
        (
            {"openarm_left_joint1": float("nan")},
            set(),
            "The commander's reading of both_arms is not usable: openarm_left_joint1 = nan.",
        ),
        (
            {"openarm_right_joint4": float("inf")},
            set(),
            "The commander's reading of both_arms is not usable: openarm_right_joint4 = inf.",
        ),
        (
            {},
            {"openarm_right_joint3"},
            "The commander's reading of both_arms lacks openarm_right_joint3, so it cannot say "
            "where the right arm is.",
        ),
    ],
    ids=["nan", "inf", "missing joint"],
)
def test_the_robot_does_not_drive_on_a_reading_that_cannot_show_the_arms(
    fake_walkie, toolbox, fake, connector, locations, joints, missing, detail
):
    # Every other joint is at home, which is travel-safe.
    fake.teleport(joints)
    fake.missing_joints = missing

    result = navigate(connector, locations, client_of(toolbox)).invoke({"location": "kitchen"})

    assert result == (
        "Not driving: the commander's reading does not show that the arms are safe for "
        f"driving. {detail} Tell the person that the arms cannot be checked."
    )
    assert fake_walkie.pose() == (0.0, 0.0, 0.0)


def test_the_robot_does_not_drive_while_an_arm_moves(
    fake_walkie, toolbox, fake, connector, locations
):
    fake.move_time_s = 1.0
    client = client_of(toolbox)
    moving = threading.Thread(target=lambda: call(toolbox, "move_arm_to_pose", pose="hands_up"))
    moving.start()
    assert wait_until(lambda: fake.joint_goals, 5.0)

    result = navigate(connector, locations, client).invoke({"location": "kitchen"})
    moving.join()

    assert result == (
        "Not driving. Cannot check the arms for driving while the move of both_arms is "
        "running. Wait for it to finish."
    )
    assert fake_walkie.pose() == (0.0, 0.0, 0.0)


def test_a_move_that_starts_during_the_drive_check_waits_for_it(
    fake_walkie, toolbox, fake, connector, locations, poses, monkeypatch
):
    # Only the gripper action is missing, which does not stop the arms being
    # read.
    client = client_of(toolbox)
    fake.vanish("control_gripper")
    assert wait_until(lambda: not client.available(), 5.0)
    standby = targets(poses, "standby", "left")
    order = client.joint_names("left_arm", threading.Event())
    service_is_ready = client._joint_states.service_is_ready
    outcome: list[str] = []
    in_flight: list[bool] = []
    moving = threading.Thread(
        target=lambda: outcome.append(
            client.set_joint_position(
                "left_arm", [standby[j] for j in order], threading.Event(), 5.0
            )
        )
    )

    def ready_while_a_move_starts() -> bool:
        # The check's first step, with the call lock held: a move that
        # starts now must not send its goal before the check has decided.
        if not in_flight:
            moving.start()
            in_flight.append(wait_until(lambda: fake.joint_goals, 0.5))
        return service_is_ready()

    monkeypatch.setattr(client._joint_states, "service_is_ready", ready_while_a_move_starts)

    result = navigate(connector, locations, client).invoke({"location": "kitchen"})
    moving.join(timeout=5)

    assert in_flight == [False]
    assert result == "Arrived at kitchen."
    assert fake.joint_state_requests == ["left_arm", "both_arms"]
    # The move went once the check was done.
    assert outcome == ["Success"]


@pytest.mark.parametrize("missing", ["set_joint_position", "control_gripper"])
def test_a_missing_goal_action_does_not_stop_the_check_reading_the_arms(
    fake_walkie, toolbox, fake, connector, locations, poses, missing
):
    client = client_of(toolbox)
    fake.vanish(missing)
    assert wait_until(lambda: not client.available(), 5.0)
    fake.teleport(targets(poses, "receive", "left"))

    assert navigate(connector, locations, client).invoke({"location": "kitchen"}) == LEFT_OUT
    assert fake.joint_state_requests == ["both_arms"]
    assert fake_walkie.pose() == (0.0, 0.0, 0.0)


def test_the_robot_does_not_drive_on_a_reading_of_one_arm(
    fake_walkie, toolbox, fake, connector, locations, monkeypatch
):
    # The check refuses a reading that lacks an arm's joints, so only an arm
    # that no pose has a table for is left unknown. Its state is faked here.
    client = client_of(toolbox)
    monkeypatch.setattr(
        client,
        "check_for_driving",
        lambda *_, **__: commander_module.DriveCheck(ArmsState(left=ArmState("home", True))),
    )

    result = navigate(connector, locations, client).invoke({"location": "kitchen"})

    assert result == (
        "Not driving: the commander's reading does not show where both arms are. Tell the "
        "person that the arms cannot be checked."
    )
    assert fake_walkie.pose() == (0.0, 0.0, 0.0)


ABANDONED = "the move of left_arm from an earlier call was given up on before its end was known"
UNVERIFIED = (
    f"Not driving. An arm or a gripper may still be moving, as {ABANDONED}, and the arms "
    "cannot be checked now. {why} Try again in a few seconds; if it fails again, tell the "
    "person that the arms cannot be checked."
)


def abandon_a_move(client: CommanderClient, fake: FakeCommander, poses, monkeypatch) -> None:
    """Leave the arms unverified: a move stopped part way whose result never comes."""
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.2)
    fake.move_time_s = 5.0
    fake.result_delay_s = math.inf
    standby = targets(poses, "standby", "left")
    order = client.joint_names("left_arm", threading.Event())
    with pytest.raises(StepError):
        client.set_joint_position("left_arm", [standby[j] for j in order], threading.Event(), 0.3)
    assert wait_until(lambda: client.busy() is None)
    assert client.unverified == ABANDONED


def test_the_robot_does_not_drive_while_a_move_outlives_the_commander(
    fake_walkie, toolbox, fake, connector, locations, poses, monkeypatch
):
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.2)
    monkeypatch.setattr(commander_module, "ABSENT_ABANDON_S", 0.3)
    fake.never_answer = True
    client = client_of(toolbox)
    standby = targets(poses, "standby", "left")
    order = client.joint_names("left_arm", threading.Event())
    cancel = threading.Event()
    errors: list[StepError] = []

    def move() -> None:
        try:
            client.set_joint_position("left_arm", [standby[j] for j in order], cancel, 30.0)
        except StepError as e:
            errors.append(e)

    moving = threading.Thread(target=move)
    moving.start()
    assert wait_until(lambda: fake.joint_goals, 5.0)
    # The commander dies mid-move. Not found any more, but the arm it was
    # moving may still be moving.
    fake.vanish()
    assert wait_until(lambda: not client.available(), 5.0)
    tool = navigate(connector, locations, client)

    assert tool.invoke({"location": "kitchen"}) == (
        "Not driving. Cannot check the arms for driving while the move of left_arm is "
        "running. Wait for it to finish."
    )
    assert fake_walkie.pose() == (0.0, 0.0, 0.0)

    cancel.set()
    moving.join(timeout=5)
    assert [(e.code, e.message) for e in errors] == [
        (
            "CANCELLED",
            "A stop or a reset interrupted the move of left_arm; the commander did not "
            "answer the cancel, so an arm may still be moving.",
        )
    ]
    # Its call gave up on it, but it may still move the arm.
    assert tool.invoke({"location": "kitchen"}) == (
        "Not driving. Cannot check the arms for driving: the move of left_arm from an earlier "
        "call has not ended yet. Try again in a few seconds."
    )
    # Abandoned once its action server stays missing. The arms are then
    # unverified and cannot be read, so the robot still does not drive.
    assert wait_until(lambda: client.busy() is None)
    assert tool.invoke({"location": "kitchen"}) == UNVERIFIED.format(
        why="The commander's /get_joint_states service is not discovered. Is "
        "bimanual_commander running?"
    )
    assert fake_walkie.pose() == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("knob", "value", "why"),
    [
        (
            "joint_states_delay_s",
            2.0,
            "The commander's /get_joint_states did not answer within 0.3 s. Is "
            "bimanual_commander running?",
        ),
        (
            "read_failure",
            "Failed to get current robot state",
            "The commander could not read the joints of both_arms: Failed to get current "
            "robot state.",
        ),
    ],
    ids=["no answer", "no reading"],
)
def test_after_an_abandoned_goal_the_robot_does_not_drive_without_a_reading(
    fake_walkie, toolbox, fake, connector, locations, poses, monkeypatch, knob, value, why
):
    client = client_of(toolbox)
    abandon_a_move(client, fake, poses, monkeypatch)
    monkeypatch.setattr(navigation_module, "ARM_CHECK_TIMEOUT_S", 0.3)
    setattr(fake, knob, value)

    result = navigate(connector, locations, client).invoke({"location": "kitchen"})

    assert result == UNVERIFIED.format(why=why)
    assert fake_walkie.pose() == (0.0, 0.0, 0.0)
    assert client.unverified == ABANDONED


def test_after_an_abandoned_goal_the_robot_drives_once_a_settled_read_shows_the_arms_still(
    fake_walkie, toolbox, fake, connector, locations, poses, monkeypatch
):
    # The commander refuses the cancel, so the arm goes on moving after the
    # goal was abandoned.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    fake.teleport(targets(poses, "standby", "left"))
    fake.move_time_s = 1.5
    fake.reject_cancels = True
    client = client_of(toolbox)
    home = targets(poses, "home", "left")
    order = client.joint_names("left_arm", threading.Event())
    with pytest.raises(StepError):
        client.set_joint_position("left_arm", [home[j] for j in order], threading.Event(), 0.3)
    assert wait_until(lambda: client.busy() is None)
    tool = navigate(connector, locations, client)

    result = tool.invoke({"location": "kitchen"})

    assert result.startswith("Not driving. An arm is still moving: openarm_left_joint")
    assert result.endswith(
        " rad between two reads 0.3 s apart. Wait a few seconds, then try again."
    )
    assert fake_walkie.pose() == (0.0, 0.0, 0.0)
    assert client.unverified == ABANDONED
    # The move ends at home, and the next check's settled read shows it.
    assert wait_until(lambda: fake.results == [("left_arm", "Success")], 5.0)
    requests = len(fake.joint_state_requests)
    assert tool.invoke({"location": "kitchen"}) == "Arrived at kitchen."
    assert fake.joint_state_requests[requests:] == ["both_arms", "both_arms"]
    assert client.unverified is None


def test_without_the_arm_backend_there_is_no_check(
    fake_walkie, fake, connector, locations, poses
):
    fake.teleport(targets(poses, "receive", "left"))

    assert navigate(connector, locations, None).invoke({"location": "kitchen"}) == (
        "Arrived at kitchen."
    )
    assert fake.joint_state_requests == []


# -- get_arm_state while the arms are unverified ---------------------------------------------

# The first line of get_arm_state's answer once its settled read verified the arms.
VERIFIED = (
    f"The arms had to be checked, as {ABANDONED}. Two reads of both arms 0.3 s apart agree, "
    "so they have stopped. Arm and gripper tools can be used again."
)


def test_get_arm_state_says_when_its_settled_read_verified_the_arms(
    toolbox, fake, poses, monkeypatch
):
    client = client_of(toolbox)
    abandon_a_move(client, fake, poses, monkeypatch)
    requests = len(fake.joint_state_requests)

    out = call(toolbox, "get_arm_state")

    assert out.summary.splitlines()[0] == VERIFIED
    # Two reads of both arms, then one of each gripper.
    assert fake.joint_state_requests[requests:] == [
        "both_arms", "both_arms", "left_gripper", "right_gripper"
    ]
    assert client.unverified is None
    # Verified, so the next answer starts with the arms.
    assert call(toolbox, "get_arm_state").summary.splitlines()[0].startswith("Left arm: ")


def test_get_arm_state_says_why_it_checked_when_an_arm_still_moves(
    toolbox, fake, poses, monkeypatch
):
    # The commander refuses the cancel, so the arm goes on moving after the
    # goal was abandoned.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    fake.move_time_s = 1.5
    fake.reject_cancels = True
    client = client_of(toolbox)
    standby = targets(poses, "standby", "left")
    order = client.joint_names("left_arm", threading.Event())
    with pytest.raises(StepError):
        client.set_joint_position("left_arm", [standby[j] for j in order], threading.Event(), 0.3)
    assert wait_until(lambda: client.busy() is None)

    error = error_of(toolbox, "get_arm_state")

    assert error.code == "PRECONDITION_FAILED"
    assert error.message.startswith(
        f"The arms had to be checked, as {ABANDONED}. An arm is still moving: openarm_left_joint"
    )
    assert error.message.endswith(
        " rad between two reads 0.3 s apart. Wait a few seconds, then try again. "
        f"{UNKNOWN_SENTENCE}"
    )
    assert client.unverified == ABANDONED
    # The move ends in standby, and the next call's settled read shows it.
    assert wait_until(lambda: fake.results == [("left_arm", "Success")], 5.0)
    out = call(toolbox, "get_arm_state")
    assert out.summary.splitlines()[:3] == [
        VERIFIED,
        "Left arm: standby, not safe for driving.",
        "Right arm: home, safe for driving.",
    ]
    assert client.unverified is None


# -- Stop and reset through the agent ------------------------------------------------------


@pytest.fixture
def start():
    """Starts an agent's thread, and stops it at the end of the test."""
    started: list[WalkieAgent] = []

    def _start(agent: WalkieAgent) -> WalkieAgent:
        agent.run()
        started.append(agent)
        return agent

    yield _start
    for agent in started:
        agent.stop()


@pytest.mark.parametrize("how", ["reset", "stop utterance", "stop"])
def test_reset_stop_and_a_stop_utterance_cancel_a_running_arm_goal(how, start, toolbox, fake):
    """WalkieAgent interrupts the toolbox, and the atomic waiting on the move cancels it."""
    fake.move_time_s = 5.0
    llm = ScriptedChatModel(
        responses=[
            tool_call("move_arm_to_pose", {"pose": "hands_up"}, "call_arm"),
            AIMessage(content="Stopped."),
        ]
    )
    agent = WalkieAgent(
        target_connectors={},
        llm=llm,
        tools=toolbox.tools(),
        stream_response=False,
        toolbox=toolbox,
    )
    start(agent)

    agent(HRIMessage(text="Raise your hands.", message_author="human"))
    assert wait_until(lambda: fake.joint_goals, 5.0)
    time.sleep(0.3)
    interrupted_at = time.monotonic()
    if how == "reset":
        agent.reset()
    elif how == "stop utterance":
        agent(HRIMessage(text="Stop.", message_author="human"))
    else:
        # Returns once the running task has ended.
        agent.stop()

    assert wait_until(lambda: fake.results == [("both_arms", "Execution Failed")], 2.0)
    assert time.monotonic() - interrupted_at < 2.0
    assert fake.cancel_requests == 1
    if how == "stop utterance":
        assert wait_until(lambda: len(llm.received) == 2 and agent.ready(), 5.0)
        tool_messages = [m.content for m in llm.received[1] if m.type == "tool"]
        assert tool_messages == [f"CANCELLED: {MOVE_STOPPED}"]
