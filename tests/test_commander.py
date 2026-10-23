"""CommanderClient against FakeCommander over ROS.

The fake serves the commander's names and types on the `connector`
fixture's Fast DDS domain (ADR-0019), with the groups of
config/robot_model.toml, so these tests cover the client's message
handling, its waits and its cancels the way the real commander would see
them.
"""

import logging
import threading
import time
from pathlib import Path

import pytest
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from walkie_agent.arm import commander as commander_module
from walkie_agent.arm.commander import CommanderClient
from walkie_agent.arm.poses import RIGHT_JOINT2, load_arm_poses, load_robot_model
from walkie_agent.arm.state import (
    NOT_SAFE_SENTENCE,
    SAFE_SENTENCE,
    UNKNOWN_SENTENCE,
    ArmsState,
    ArmState,
    worst_joint,
)
from walkie_agent.sim.fake_commander import FakeCommander
from walkie_agent.toolbox import StepError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "config" / "robot_model.toml"
POSES_PATH = PROJECT_ROOT / "config" / "arm_poses.toml"

NO_CANCEL = threading.Event()
GROUPS = ("left_arm", "right_arm", "both_arms", "left_gripper", "right_gripper")


def wait_until(predicate, timeout=3.0):
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
def fake_commander(model):
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
def commander(connector, poses):
    client = CommanderClient(connector, poses)
    yield client
    client.destroy()


@pytest.fixture
def client(fake_commander, commander):
    """A CommanderClient that has discovered the fake commander."""
    assert wait_until(commander.available, 5.0), "The client never discovered the fake commander."
    return commander


def table(poses, pose: str, arm: str) -> dict[str, float]:
    return dict(poses.poses[pose].tables[arm])


def move(client, group: str, targets: dict[str, float], cancel=NO_CANCEL, timeout_s=5.0):
    """Send `targets` in the commander's joint order, as move_arm_to_pose does."""
    order = client.joint_names(group, NO_CANCEL)
    return client.set_joint_position(group, [targets[j] for j in order], cancel, timeout_s)


# -- Reads ------------------------------------------------------------------------


def test_joint_states_lists_every_group_in_the_commanders_order(client, model, poses):
    for group in GROUPS:
        assert list(client.joint_states(group, NO_CANCEL)) == list(model.groups[group])

    assert client.joint_states("left_arm", NO_CANCEL) == pytest.approx(table(poses, "home", "left"))
    # The fingers start open.
    assert client.joint_states("right_gripper", NO_CANCEL) == {
        "openarm_right_finger_joint1": pytest.approx(0.044)
    }


def test_joint_names_are_read_once_per_group(client, fake_commander):
    names = client.joint_names("right_arm", NO_CANCEL)

    assert client.joint_names("right_arm", NO_CANCEL) == names
    assert fake_commander.joint_state_requests == ["right_arm"]


def test_read_arms_and_last_state(client, fake_commander, poses):
    assert client.last_state == ArmsState()
    assert client.last_state.sentence() == UNKNOWN_SENTENCE

    arms = client.read_arms(NO_CANCEL)

    assert arms == ArmsState(left=ArmState("home", True), right=ArmState("home", True))
    assert arms.sentence() == SAFE_SENTENCE
    assert client.last_state == arms
    assert fake_commander.joint_state_requests == ["both_arms"]

    # A read of one arm updates that arm and keeps the other's last state.
    fake_commander.teleport(table(poses, "standby", "left"))
    client.joint_states("left_arm", NO_CANCEL)
    assert client.last_state == ArmsState(left=ArmState("standby", False), right=arms.right)
    assert client.last_state.sentence() == NOT_SAFE_SENTENCE
    # A gripper read says nothing about the arms.
    client.joint_states("left_gripper", NO_CANCEL)
    assert client.last_state.left == ArmState("standby", False)


def test_a_slow_read_times_out(client, fake_commander, caplog):
    fake_commander.joint_states_delay_s = 1.0
    start = time.monotonic()

    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        with pytest.raises(StepError) as error:
            client.joint_states("both_arms", NO_CANCEL, timeout_s=0.2)

    assert error.value.code == "TIMEOUT"
    assert "/get_joint_states did not answer within 0.2 s" in error.value.message
    assert time.monotonic() - start < 0.6
    # An unanswered call is what a stale my_robot_interfaces pin looks like
    # under rmw_zenoh (ADR-0035); the hint is for the log, not the LLM.
    assert "my_robot_interfaces" not in error.value.message
    assert commander_module.STALE_PIN_HINT in caplog.text
    fake_commander.joint_states_delay_s = 0.0
    assert client.read_arms(NO_CANCEL).travel_safe is True


# -- Joint goals ------------------------------------------------------------------


def test_a_move_reaches_the_pose_and_the_read_matches_it(client, fake_commander, poses):
    standby = table(poses, "standby", "left")

    assert move(client, "left_arm", standby) == "Success"

    group, sent = fake_commander.joint_goals[-1]
    assert (group, sent) == ("left_arm", tuple(standby.values()))
    reading = client.joint_states("left_arm", NO_CANCEL)
    assert worst_joint(standby, reading)[1] < 1e-9
    arms = client.read_arms(NO_CANCEL)
    assert arms == ArmsState(left=ArmState("standby", False), right=ArmState("home", True))
    assert client.last_state == arms


def test_a_move_follows_the_order_the_commander_reports(client, fake_commander, poses):
    fake_commander.reverse_joint_order = True
    hands_up = table(poses, "hands_up", "right")

    assert client.joint_names("right_arm", NO_CANCEL)[0] == "openarm_right_joint7"
    assert move(client, "right_arm", hands_up) == "Success"

    assert client.read_arms(NO_CANCEL).right == ArmState("hands_up", False)


def test_a_planning_failure_moves_nothing(client, fake_commander, poses):
    fake_commander.plan_failure = True
    before = fake_commander.joints()

    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"))

    assert error.value.code == "SKILL_FAILED"
    assert error.value.message == "MoveIt found no path (Planning Failed). The arms did not move."
    assert error.value.data == {"group": "left_arm", "status": "Planning Failed"}
    assert fake_commander.joints() == before


def test_an_execution_failure_stops_part_way(client, fake_commander, poses):
    fake_commander.execution_failure = True
    standby = table(poses, "standby", "left")

    with pytest.raises(StepError) as error:
        move(client, "left_arm", standby)

    assert error.value.code == "SKILL_FAILED"
    assert "(Execution Failed)" in error.value.message
    assert error.value.data == {"group": "left_arm", "status": "Execution Failed"}
    assert client.read_arms(NO_CANCEL).left == ArmState(None, False)


def test_plan_only_reports_success_without_moving(client, fake_commander, poses):
    fake_commander.plan_only = True

    assert move(client, "left_arm", table(poses, "standby", "left")) == "Plan Only"

    # The read after the goal shows that nothing moved (ADR-0024).
    assert client.read_arms(NO_CANCEL).left == ArmState("home", True)


def test_a_joint_left_off_shows_in_the_read(client, fake_commander, poses):
    fake_commander.joint_errors = {"openarm_left_joint4": 0.31}
    standby = table(poses, "standby", "left")

    assert move(client, "left_arm", standby) == "Success"

    joint, error = worst_joint(standby, client.joint_states("left_arm", NO_CANCEL))
    assert (joint, error) == ("openarm_left_joint4", pytest.approx(0.31))
    assert client.read_arms(NO_CANCEL).left == ArmState(None, False)


def test_a_rejected_goal_is_internal_and_drops_the_joint_order(client, fake_commander, poses):
    fake_commander.reject_goals = True

    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "home", "left"))

    assert error.value.code == "INTERNAL"
    assert error.value.data == {"group": "left_arm", "count": 7}
    assert "rejected the goal for left_arm with 7 positions" in error.value.message
    # The order is read again, in case walkie-ros restarted with other groups.
    requests = len(fake_commander.joint_state_requests)
    client.joint_names("left_arm", NO_CANCEL)
    assert len(fake_commander.joint_state_requests) == requests + 1

    # The commander's own rule: a count that differs from the group's joints.
    fake_commander.reject_goals = False
    with pytest.raises(StepError) as error:
        client.set_joint_position("left_arm", [0.0] * 6, NO_CANCEL, 5.0)
    assert error.value.data == {"group": "left_arm", "count": 6}


def test_a_goal_never_answered_times_out_and_is_cancelled(client, fake_commander, poses):
    fake_commander.never_answer = True
    client.joint_names("left_arm", NO_CANCEL)
    start = time.monotonic()

    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.5)

    assert error.value.code == "TIMEOUT"
    assert error.value.message == (
        "The move of left_arm did not finish within 0.5 s; the goal was cancelled."
    )
    assert time.monotonic() - start < 1.0
    assert fake_commander.cancel_requests == 1
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])


def test_a_goal_accepted_after_its_timeout_is_cancelled(client, fake_commander, poses, caplog):
    client.joint_names("left_arm", NO_CANCEL)
    fake_commander.accept_delay_s = 0.5

    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        with pytest.raises(StepError) as error:
            move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.2)

    assert error.value.code == "TIMEOUT"
    assert error.value.message.endswith("; it is cancelled if the commander accepts it.")
    assert commander_module.STALE_PIN_HINT in caplog.text
    # Nobody waits for it any more, so it must not move the arm to the end.
    assert wait_until(lambda: fake_commander.cancel_requests == 1)
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])


def test_a_cancel_mid_move_ends_the_goal_promptly(client, fake_commander, poses):
    fake_commander.move_time_s = 5.0
    home = client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    standby = table(poses, "standby", "left")
    cancel = threading.Event()
    cancelled_at = []
    timer = threading.Timer(0.4, lambda: (cancelled_at.append(time.monotonic()), cancel.set()))
    timer.start()

    with pytest.raises(StepError) as error:
        move(client, "left_arm", standby, cancel=cancel)
    returned_at = time.monotonic()

    assert error.value.code == "CANCELLED"
    assert error.value.message == (
        "A stop or a reset interrupted the move of left_arm; the goal was cancelled."
    )
    assert returned_at - cancelled_at[0] < 0.5
    assert fake_commander.cancel_requests == 1
    # The commander stops the arms and ends the goal as failed, not cancelled.
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")], 1.0)
    # The arm stopped part way, and stays unknown until it is read again.
    joint1 = fake_commander.joints()["openarm_left_joint1"]
    assert table(poses, "home", "left")["openarm_left_joint1"] < joint1
    assert joint1 < standby["openarm_left_joint1"]
    assert client.last_state == ArmsState(left=None, right=home.right)
    assert client.last_state.sentence() == UNKNOWN_SENTENCE


def test_a_set_cancel_sends_nothing(client, fake_commander, poses):
    cancel = threading.Event()
    cancel.set()

    for call in (
        lambda: client.read_arms(cancel),
        lambda: client.set_joint_position("left_arm", [0.0] * 7, cancel, 5.0),
        lambda: client.control_gripper("left_gripper", 0.0, cancel),
    ):
        with pytest.raises(StepError) as error:
            call()
        assert error.value.code == "CANCELLED"

    assert fake_commander.joint_state_requests == []
    assert fake_commander.joint_goals == []
    assert fake_commander.gripper_goals == []


def test_a_read_or_a_second_goal_during_a_move_is_refused(client, fake_commander, poses):
    fake_commander.move_time_s = 1.0
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    outcome = []
    thread = threading.Thread(
        target=lambda: outcome.append(move(client, "left_arm", table(poses, "standby", "left")))
    )
    thread.start()
    assert wait_until(lambda: fake_commander.joint_goals)
    start = time.monotonic()

    for call in (
        lambda: client.read_arms(NO_CANCEL),
        lambda: client.control_gripper("left_gripper", 0.0, NO_CANCEL),
    ):
        with pytest.raises(StepError) as error:
            call()
        assert error.value.code == "PRECONDITION_FAILED"
        assert "while the move of left_arm is running" in error.value.message

    assert time.monotonic() - start < 0.3
    assert client.last_state.left is None
    thread.join(timeout=5)
    assert outcome == ["Success"]
    assert fake_commander.gripper_goals == []


def test_right_joint2_fixed(client, fake_commander, poses):
    fake_commander.right_joint2_fixed = True

    right = client.joint_names("right_arm", NO_CANCEL)
    assert RIGHT_JOINT2 not in right
    assert len(right) == 6
    assert len(client.joint_names("both_arms", NO_CANCEL)) == 13

    # Every preset has joint2 at 0, which the fixed joint holds.
    assert move(client, "right_arm", table(poses, "standby", "right")) == "Success"
    assert client.read_arms(NO_CANCEL).right == ArmState("standby", False)
    with pytest.raises(StepError) as error:
        client.set_joint_position("right_arm", [0.0] * 7, NO_CANCEL, 5.0)
    assert error.value.data == {"group": "right_arm", "count": 7}


# -- Gripper goals ----------------------------------------------------------------


def test_a_close_on_an_object_holds_it(client, fake_commander):
    fake_commander.object_widths["left_gripper"] = 0.02

    result = client.control_gripper("left_gripper", 0.0, NO_CANCEL)

    # The fingers stall on the object, so the commander reports a failure
    # and the grasp at once.
    assert (result.success, result.status) == (False, "Failed")
    assert result.grasped is True
    assert result.gripper_gap == pytest.approx(0.02)
    assert client.last_gripper_position("left_gripper") == 0.0
    assert client.last_gripper_position("right_gripper") is None


def test_a_close_on_nothing_holds_nothing(client):
    result = client.control_gripper("right_gripper", 0.0, NO_CANCEL)

    assert (result.success, result.status) == (True, "Success")
    assert result.grasped is False
    assert result.gripper_gap == 0.0


def test_grasp_detect_min_below_zero_reads_an_empty_close_as_held(client, fake_commander):
    fake_commander.set_parameters([Parameter("grasp_detect_min", value=-1.0)])

    result = client.control_gripper("left_gripper", 0.0, NO_CANCEL)

    assert result.grasped is True
    assert result.gripper_gap == 0.0


def test_open_and_half_ignore_the_commanders_gap(client, fake_commander):
    fake_commander.object_widths["left_gripper"] = 0.02
    client.control_gripper("left_gripper", 0.0, NO_CANCEL)

    for position in (0.044, 0.036):
        result = client.control_gripper("left_gripper", position, NO_CANCEL)
        # The commander answers grasped false and a 0.0 gap, which says
        # nothing about the fingers.
        assert (result.success, result.grasped, result.gripper_gap) == (True, None, None)
        assert fake_commander.joints()["openarm_left_finger_joint1"] == pytest.approx(position)
        assert client.last_gripper_position("left_gripper") == position


def test_a_gripper_goal_never_answered_times_out(client, fake_commander):
    client.control_gripper("left_gripper", 0.044, NO_CANCEL)
    fake_commander.never_answer = True

    with pytest.raises(StepError) as error:
        client.control_gripper("left_gripper", 0.0, NO_CANCEL, timeout_s=0.3)

    assert error.value.code == "TIMEOUT"
    assert error.value.message == (
        "The left_gripper command did not finish within 0.3 s; the goal was cancelled."
    )
    # Stopped somewhere unknown.
    assert client.last_gripper_position("left_gripper") is None
    assert wait_until(lambda: fake_commander.results[-1:] == [("left_gripper", "Canceled")])


# -- Availability and arguments -----------------------------------------------------


def test_available_answers_at_once(client):
    start = time.monotonic()

    assert client.available()
    assert time.monotonic() - start < 0.1


def test_without_a_commander_available_is_false_at_once_and_calls_fail(commander, monkeypatch):
    monkeypatch.setattr(commander_module, "READY_WAIT_S", 0.2)
    start = time.monotonic()

    assert not commander.available()
    assert time.monotonic() - start < 0.1

    with pytest.raises(StepError) as error:
        commander.read_arms(NO_CANCEL)
    assert error.value.code == "UNAVAILABLE"
    assert error.value.message == (
        "The commander's /get_joint_states service is not available. "
        "Is bimanual_commander running?"
    )
    for call, name in (
        (lambda: commander.set_joint_position("left_arm", [0.0] * 7, NO_CANCEL, 5.0),
         "/set_joint_position action"),
        (lambda: commander.control_gripper("left_gripper", 0.0, NO_CANCEL),
         "/control_gripper action"),
    ):
        with pytest.raises(StepError) as error:
            call()
        assert (error.value.code, error.value.data) == ("UNAVAILABLE", {"name": name})
    assert time.monotonic() - start < 2.0


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.set_joint_position("left_arm_lift", [0.0] * 8, NO_CANCEL, 5.0),
        lambda c: c.set_joint_position("left_gripper", [0.0], NO_CANCEL, 5.0),
        lambda c: c.set_joint_position("left_arm", [float("nan")] * 7, NO_CANCEL, 5.0),
        lambda c: c.control_gripper("left_arm", 0.0, NO_CANCEL),
        lambda c: c.joint_states("both_arms_lift", NO_CANCEL),
        lambda c: c.joint_states("left_arm", NO_CANCEL, timeout_s=0.0),
    ],
)
def test_a_group_or_value_the_agent_may_not_send_is_a_bug(commander, call):
    with pytest.raises(ValueError):
        call(commander)


def test_integer_positions_arrive_as_their_values(client, fake_commander):
    # rclpy accepts an int in a float64 field but sends it wrong, so the
    # client converts every value.
    client.set_joint_position("left_arm", [0, 0, 0, 1, 0, 0, 0], NO_CANCEL, 5.0)

    assert fake_commander.joint_goals[-1] == ("left_arm", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    assert fake_commander.joints()["openarm_left_joint4"] == 1.0
