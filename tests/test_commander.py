"""CommanderClient against FakeCommander over ROS.

The fake serves the commander's names and types on the `connector`
fixture's Fast DDS domain (ADR-0019), with the groups of
config/robot_model.toml, so these tests cover the client's message
handling, its waits and its cancels the way the real commander would see
them.
"""

import logging
import math
import threading
import time
from pathlib import Path

import pytest
from action_msgs.msg import GoalInfo
from action_msgs.srv import CancelGoal
from rclpy.action.client import ClientGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.task import Future
from unique_identifier_msgs.msg import UUID

from walkie_agent.arm import commander as commander_module
from walkie_agent.arm.commander import CommanderClient, Recovery
from walkie_agent.arm.poses import RIGHT_JOINT2, load_arm_poses, load_robot_model
from walkie_agent.arm.state import (
    NOT_SAFE_SENTENCE,
    SAFE_SENTENCE,
    UNKNOWN_SENTENCE,
    ArmsState,
    ArmState,
    worst_joint,
)
from walkie_agent.sim.fake_commander import CONTROL_GRIPPER, SET_JOINT_POSITION, FakeCommander
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


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_a_reading_that_is_not_a_number_is_internal_and_forgets_the_arms(
    client, fake_commander, value
):
    client.read_arms(NO_CANCEL)
    # Every other joint at home but joint4 far off: a NaN passes every
    # tolerance check, so a match that let it through would call the arm home.
    fake_commander.teleport({"openarm_left_joint1": value, "openarm_left_joint4": 2.0})

    with pytest.raises(StepError) as error:
        client.read_arms(NO_CANCEL)

    assert error.value.code == "INTERNAL"
    assert error.value.message == (
        f"The commander's reading of both_arms is not usable: openarm_left_joint1 = {value}."
    )
    assert error.value.data == {"joints": ["openarm_left_joint1"]}
    assert client.last_state == ArmsState()
    # A read of the other arm alone says nothing about this one.
    client.joint_states("right_arm", NO_CANCEL)
    assert client.last_state == ArmsState(right=ArmState("home", True))


def test_a_reading_that_lacks_an_arms_joints_is_internal_and_forgets_that_arm(
    client, fake_commander, model
):
    home = client.read_arms(NO_CANCEL)
    # Only the left arm's joints, as in a malformed answer.
    fake_commander.missing_joints = set(model.groups["right_arm"])
    # openarm_right_joint2 is not missed: right_joint2_fixed leaves it out.
    missing = sorted(set(model.groups["right_arm"]) - {RIGHT_JOINT2})

    with pytest.raises(StepError) as error:
        client.read_arms(NO_CANCEL)

    assert error.value.code == "INTERNAL"
    assert error.value.message == (
        f"The commander's reading of both_arms lacks {', '.join(missing)}, so it cannot say "
        "where the right arm is."
    )
    assert error.value.data == {"missing": missing}
    assert client.last_state == ArmsState(left=home.left, right=None)

    # A read of one arm that lacks a joint forgets that arm, and keeps the other.
    fake_commander.missing_joints = {"openarm_left_joint3"}
    assert "openarm_left_joint3" not in client.joint_states("left_arm", NO_CANCEL)
    assert client.last_state == ArmsState()
    client.joint_states("right_arm", NO_CANCEL)
    assert client.last_state == ArmsState(right=home.right)

    fake_commander.missing_joints = {"openarm_left_joint3", "openarm_right_joint5"}
    with pytest.raises(StepError) as error:
        client.read_arms(NO_CANCEL)
    assert error.value.message == (
        "The commander's reading of both_arms lacks openarm_left_joint3, openarm_right_joint5, "
        "so it cannot say where either arm is."
    )
    assert client.last_state == ArmsState()


def test_the_joint_order_of_a_reading_that_lacks_a_joint_is_not_kept(
    client, fake_commander, model
):
    fake_commander.missing_joints = {"openarm_left_joint3"}
    client.joint_states("left_arm", NO_CANCEL)
    fake_commander.missing_joints = set()
    requests = len(fake_commander.joint_state_requests)

    # Read again, not taken from the incomplete reading.
    assert set(client.joint_names("left_arm", NO_CANCEL)) == set(model.groups["left_arm"])
    assert len(fake_commander.joint_state_requests) == requests + 1
    # A complete reading is kept.
    client.joint_names("left_arm", NO_CANCEL)
    assert len(fake_commander.joint_state_requests) == requests + 1


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
        "The move of left_arm did not finish within 0.5 s; the arms were told to stop."
    )
    assert error.value.data == {"cancel": "accepted"}
    assert time.monotonic() - start < 1.0
    assert fake_commander.cancel_requests == 1
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])
    assert wait_until(lambda: client.busy() is None)


def refused(call) -> StepError:
    with pytest.raises(StepError) as error:
        call()
    assert error.value.code == "PRECONDITION_FAILED"
    return error.value


def test_a_goal_accepted_after_its_call_gave_up_holds_off_every_call_until_it_ends(
    client, fake_commander, poses, caplog
):
    client.joint_names("left_arm", NO_CANCEL)
    before = fake_commander.joints()
    fake_commander.accept_delay_s = 0.5
    fake_commander.plan_time_s = 0.5
    calls = (
        lambda: client.read_arms(NO_CANCEL),
        lambda: client.control_gripper("left_gripper", 0.0, NO_CANCEL),
    )

    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        with pytest.raises(StepError) as error:
            move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.2)

        assert error.value.code == "TIMEOUT"
        assert error.value.message == (
            "The move of left_arm did not finish within 0.2 s; if the commander accepts it "
            "later, it is cancelled at once."
        )
        # No cancel was sent: the commander had not answered the send.
        assert error.value.data == {"cancel": "pending"}
        assert commander_module.STALE_PIN_HINT in caplog.text
        # Not answered yet: the commander may still accept it and move the arm.
        assert client.busy() == "the move of left_arm from an earlier call"
        for call in calls:
            assert refused(call).message.endswith(
                ": the move of left_arm from an earlier call has not ended yet. "
                "Try again in a few seconds."
            )
        # Accepted, and cancelled at once; still refused until its result.
        assert wait_until(lambda: fake_commander.cancel_requests == 1)
        for call in calls:
            refused(call)
        assert wait_until(lambda: client.busy() is None)

    assert "accepted the move of left_arm after its call gave up" in caplog.text
    # Cancelled while it planned, so the arm never moved.
    assert fake_commander.results == [("left_arm", "Canceled")]
    assert fake_commander.joints() == before
    assert fake_commander.gripper_goals == []
    # It was outstanding, not abandoned, when the commander accepted it, so
    # the arms stay verified.
    assert client.unverified is None
    assert client.read_arms(NO_CANCEL).left == ArmState("home", True)


def test_a_gripper_goal_accepted_after_its_call_gave_up_still_moves_the_fingers(
    client, fake_commander
):
    fake_commander.accept_delay_s = 0.5

    with pytest.raises(StepError) as error:
        client.control_gripper("left_gripper", 0.0, NO_CANCEL, timeout_s=0.2)

    assert error.value.code == "TIMEOUT"
    assert error.value.message == (
        "The left_gripper command did not finish within 0.2 s; if the commander accepts it "
        "later, the gripper still makes that motion."
    )
    assert client.busy() == "the left_gripper command from an earlier call"
    # Accepted and cancelled, but the fingers finish their ramp first.
    assert wait_until(lambda: client.busy() is None)
    assert fake_commander.cancel_requests == 1
    assert fake_commander.results == [("left_gripper", "Canceled")]
    assert fake_commander.joints()["openarm_left_finger_joint1"] == 0.0


ABANDONED = "the move of left_arm from an earlier call was given up on before its end was known"
ACCEPTED_LATE = (
    "the commander accepted the move of left_arm from an earlier call late and was asked to "
    "cancel it"
)


def unverified_refusal(action: str, cause: str) -> str:
    return (
        f"Cannot {action}: an arm or a gripper may still be moving, as {cause}. Call "
        "get_arm_state to check the arms, then try again."
    )


def arm_moved(fake_commander, poses, pose: str, arm: str):
    """A check that is true once `arm` left its current pose for `pose`, by its largest change."""
    start = fake_commander.joints()
    target = table(poses, pose, arm)
    joint = max(target, key=lambda name: abs(target[name] - start[name]))
    return lambda: abs(fake_commander.joints()[joint] - start[joint]) > 1e-3


def restart(client, fake_commander, name: str = SET_JOINT_POSITION) -> None:
    """Restart the fake's action server `name`, as when the commander restarts.

    The new server does not know the goals of the old one, which go on as
    the fake's docstring says.
    """
    fake_commander.vanish(name)
    assert wait_until(lambda: not client.available(), 5.0)
    fake_commander.come_back()
    assert wait_until(client.available, 5.0)


def test_a_goal_whose_result_never_comes_is_abandoned_after_the_limit(
    client, fake_commander, poses, monkeypatch, caplog
):
    # As when the commander restarted: its servers are still discovered,
    # but the old goal's result never comes.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.5)
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.3)
    fake_commander.move_time_s = 5.0
    fake_commander.result_delay_s = math.inf
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    client.joint_names("right_arm", NO_CANCEL)

    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    gave_up_at = time.monotonic()

    assert error.value.message == (
        "The move of left_arm did not finish within 0.3 s; the arms were told to stop."
    )
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])
    refused(lambda: client.read_arms(NO_CANCEL))
    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        assert wait_until(lambda: client.busy() is None)
    assert time.monotonic() - gave_up_at >= 0.5
    assert (
        "Abandoned the move of left_arm: no result came within 0.5 s after its call gave up"
        in caplog.text
    )
    # Calls are taken again, but the goal's end is unknown: the arms are
    # unverified, and every goal is refused until a settled read.
    assert client.unverified == ABANDONED
    assert client.last_state.left is None
    for call, action in (
        (lambda: move(client, "right_arm", table(poses, "standby", "right")), "move right_arm"),
        (lambda: client.control_gripper("left_gripper", 0.0, NO_CANCEL), "move left_gripper"),
    ):
        error = refused(call)
        assert error.message == unverified_refusal(action, ABANDONED)
        assert error.data == {"unverified": ABANDONED}
    assert len(fake_commander.joint_goals) == 1
    assert fake_commander.gripper_goals == []
    # One read does not verify the arms.
    client.joint_states("left_arm", NO_CANCEL)
    assert client.unverified == ABANDONED
    # read_arms recovers. The commander answers its cancel that the goal
    # ended, but it accepted the call's cancel before, and then only the
    # result shows that the goal ended.
    error = refused(lambda: client.read_arms(NO_CANCEL))
    assert error.message == (
        "The commander accepted the cancel of the move of left_arm from an earlier call, and "
        "its end has not come yet. Wait a few seconds, then try again."
    )
    assert error.data == {"unresolved": ["the move of left_arm"]}
    assert client.unverified == ABANDONED
    # A restarted commander does not know the goal, which resolves it, and
    # the settled read shows the arm stopped part way.
    restart(client, fake_commander)
    count = client.unverified_count
    requests = len(fake_commander.joint_state_requests)
    assert client.read_arms(NO_CANCEL).left == ArmState(None, False)
    assert fake_commander.joint_state_requests[requests:] == ["both_arms", "both_arms"]
    assert client.unverified is None
    assert client.last_recovery == Recovery(count, (("the move of left_arm", "unknown"),), ())
    fake_commander.move_time_s = 0.2
    fake_commander.result_delay_s = 0.0
    assert move(client, "left_arm", table(poses, "home", "left")) == "Success"


def test_a_goal_whose_server_stays_away_is_abandoned(
    client, fake_commander, poses, monkeypatch, caplog
):
    monkeypatch.setattr(commander_module, "ABSENT_ABANDON_S", 0.5)
    fake_commander.never_answer = True
    fake_commander.reject_cancels = True
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)

    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)

    assert error.value.code == "TIMEOUT"
    assert error.value.data == {"cancel": "refused"}
    refused(lambda: client.read_arms(NO_CANCEL))
    assert client.busy() == "the move of left_arm from an earlier call"

    fake_commander.vanish()
    assert wait_until(lambda: not client._set_joint_position.server_is_ready())
    first_absent = time.monotonic()
    # One check that finds the server missing does not abandon the goal:
    # discovery can drop for a moment while the commander still runs it.
    assert client.busy() == "the move of left_arm from an earlier call"
    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        assert wait_until(lambda: client.busy() is None)
    assert time.monotonic() - first_absent >= 0.5
    assert (
        "Abandoned the move of left_arm: its action server has not been discovered for 0.5 s"
        in caplog.text
    )
    assert client.unverified == ABANDONED
    assert client.last_state.left is None
    error = refused(lambda: move(client, "left_arm", table(poses, "home", "left")))
    assert error.message == unverified_refusal("move left_arm", ABANDONED)
    # Reads are taken again and fail as without a commander.
    monkeypatch.setattr(commander_module, "READY_WAIT_S", 0.2)
    with pytest.raises(StepError) as error:
        client.joint_states("left_arm", NO_CANCEL)
    assert error.value.code == "UNAVAILABLE"
    # Nothing answers recovery's cancel, so the arms stay unverified.
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.2)
    error = refused(lambda: client.read_arms(NO_CANCEL))
    assert error.message == (
        "The commander did not answer a cancel of the move of left_arm from an earlier call in "
        "time, so an arm may still be moving. Wait a few seconds, then try again; if it fails "
        "again, tell the person that the arms cannot be checked."
    )
    assert error.data == {"unresolved": ["the move of left_arm"]}
    assert client.unverified == ABANDONED


def test_a_goal_whose_server_comes_back_is_kept_until_the_time_limit(
    client, fake_commander, poses, monkeypatch, caplog
):
    # As when the commander restarts: its servers go for a moment, and the
    # new ones do not know the old goal, whose result never comes.
    monkeypatch.setattr(commander_module, "ABSENT_ABANDON_S", 1.0)
    fake_commander.never_answer = True
    fake_commander.reject_cancels = True
    client.joint_names("left_arm", NO_CANCEL)
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": "refused"}

    fake_commander.vanish()
    assert wait_until(lambda: not client._set_joint_position.server_is_ready())
    fake_commander.come_back()
    # Checked while the servers are away and once they are back: an absence
    # shorter than ABSENT_ABANDON_S abandons nothing.
    assert wait_until(lambda: client.busy() is not None and client.available())
    # Discovered again, so the absence no longer counts, however short the
    # limit is now.
    monkeypatch.setattr(commander_module, "ABSENT_ABANDON_S", 0.01)
    time.sleep(0.05)
    assert client.busy() == "the move of left_arm from an earlier call"
    assert client.unverified is None

    # The new servers never send the old goal's result, so only the time
    # limit ends the wait.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.01)
    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        assert client.busy() is None
    assert (
        "Abandoned the move of left_arm: no result came within 0.01 s after its call gave up"
        in caplog.text
    )
    assert client.unverified == ABANDONED
    # The old server's goal never ended.
    assert fake_commander.results == []


def test_a_settled_read_refuses_while_an_arm_still_moves(
    client, fake_commander, poses, monkeypatch
):
    # The commander refuses the call's cancel and restarts while the goal
    # plans. The old goal then moves the arm, which the new commander does
    # not know, as a trajectory that move_group still executes would.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    plan = threading.Event()
    fake_commander.plan_gate = plan
    fake_commander.move_time_s = 1.0
    fake_commander.reject_cancels = True
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": "refused"}
    restart(client, fake_commander)
    assert wait_until(lambda: client.busy() is None)
    assert client.unverified == ABANDONED
    moved = arm_moved(fake_commander, poses, "standby", "left")
    plan.set()
    assert wait_until(moved)

    error = refused(lambda: client.read_arms(NO_CANCEL))

    assert error.message.startswith("An arm is still moving: openarm_left_joint")
    assert error.message.endswith(
        " rad between two reads 0.3 s apart. Wait a few seconds, then try again."
    )
    assert error.data["joint"].startswith("openarm_left_joint")
    assert error.data["moved_rad"] > commander_module.SETTLE_TOLERANCE_RAD
    assert client.unverified == ABANDONED
    assert client.last_state == ArmsState()
    # Once the move ends, the next settled read shows the arms still.
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Success")])
    assert client.read_arms(NO_CANCEL).left == ArmState("standby", False)
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "unknown"),)


@pytest.mark.parametrize("call_cancel", ["accepted", "refused"])
def test_recovery_waits_for_the_result_of_a_goal_that_still_plans(
    client, fake_commander, poses, monkeypatch, call_cancel
):
    # The goal plans until the test lets it finish, so the arms are still
    # meanwhile. The commander accepts the call's cancel, or refuses it and
    # accepts recovery's.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.3)
    plan = threading.Event()
    fake_commander.plan_gate = plan
    fake_commander.reject_cancels = call_cancel == "refused"
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    before = fake_commander.joints()
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": call_cancel}
    assert wait_until(lambda: client.busy() is None)
    assert client.unverified == ABANDONED
    fake_commander.reject_cancels = False

    error = refused(lambda: client.read_arms(NO_CANCEL))

    # After an accepted cancel, the commander answers another one as if the
    # goal had ended, without asking its cancel callback. Only the result
    # shows that the goal ended.
    assert error.message == (
        "The commander accepted the cancel of the move of left_arm from an earlier call, and "
        "its end has not come yet. Wait a few seconds, then try again."
    )
    assert error.data == {"unresolved": ["the move of left_arm"]}
    assert client.unverified == ABANDONED
    assert fake_commander.cancel_requests == (1 if call_cancel == "accepted" else 2)
    # Once the plan is done, the goal sees the cancel and ends without moving.
    plan.set()
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Canceled")])
    count = client.unverified_count
    assert client.read_arms(NO_CANCEL).left == ArmState("home", True)
    assert client.unverified is None
    assert client.last_recovery == Recovery(count, (("the move of left_arm", "result"),), ())
    assert fake_commander.joints() == before


@pytest.mark.parametrize("answer", ["refused", "no answer"])
def test_recovery_refuses_while_the_cancel_of_a_goal_is_refused_or_unanswered(
    client, fake_commander, poses, monkeypatch, caplog, answer
):
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.2)
    # Held until a cancel of it is accepted.
    fake_commander.never_answer = True
    held = threading.Event()
    if answer == "refused":
        fake_commander.reject_cancels = True
    else:
        fake_commander.cancel_gate = held
    client.joint_names("left_arm", NO_CANCEL)
    client.joint_names("right_arm", NO_CANCEL)
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": answer}
    assert wait_until(lambda: client.busy() is None)

    error = refused(lambda: client.read_arms(NO_CANCEL))

    reply = (
        "refused the cancel of the move of left_arm from an earlier call"
        if answer == "refused"
        else "did not answer a cancel of the move of left_arm from an earlier call in time"
    )
    assert error.message == (
        f"The commander {reply}, so an arm may still be moving. Wait a few seconds, then try "
        "again; if it fails again, tell the person that the arms cannot be checked."
    )
    assert error.data == {"unresolved": ["the move of left_arm"]}
    assert client.unverified == ABANDONED
    # No newer goal starts while the old one may still be active.
    for call, action in (
        (lambda: move(client, "right_arm", table(poses, "standby", "right")), "move right_arm"),
        (lambda: client.control_gripper("left_gripper", 0.0, NO_CANCEL), "move left_gripper"),
    ):
        assert refused(call).message == unverified_refusal(action, ABANDONED)
    assert len(fake_commander.joint_goals) == 1
    assert fake_commander.gripper_goals == []
    # Once the commander accepts a cancel, the goal ends, and its result
    # resolves it.
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 1.0)
    fake_commander.reject_cancels = False
    with caplog.at_level(logging.INFO, logger=commander_module.__name__):
        held.set()
        if answer == "no answer":
            # The held cancels are answered now.
            assert wait_until(lambda: "is no longer active: its result came" in caplog.text)
        count = client.unverified_count
        client.read_arms(NO_CANCEL)
    assert (
        "The move of left_arm from an earlier call is no longer active: its result came."
        in caplog.text
    )
    assert fake_commander.results == [("left_arm", "Execution Failed")]
    assert client.unverified is None
    assert client.last_recovery == Recovery(count, (("the move of left_arm", "result"),), ())


def test_a_newer_goal_waits_until_the_commander_confirms_that_an_older_one_ended(
    client, fake_commander, poses, monkeypatch, caplog
):
    # The commander accepts an abandoned goal late and holds the cancel of
    # it unanswered while recovery runs. A cancel that it accepts stops both
    # arms, so no newer goal may start until it was answered and the goal
    # ended.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    # Longer than the read's timeout, so recovery waits for the cancel that
    # is on its way instead of sending another.
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 5.0)
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    client.joint_names("right_arm", NO_CANCEL)
    before = fake_commander.joints()
    accept = threading.Event()
    fake_commander.accept_gate = accept
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": "pending"}
    assert wait_until(lambda: client.busy() is None)
    assert client.unverified == ABANDONED
    # Accepted late, the goal plans until the test lets it finish, and the
    # cancel that follows the acceptance is held.
    plan = threading.Event()
    answer = threading.Event()
    fake_commander.plan_gate = plan
    fake_commander.cancel_gate = answer
    accept.set()
    assert wait_until(lambda: fake_commander.cancel_requests == 1)
    assert client.unverified == ACCEPTED_LATE

    error = refused(lambda: client.read_arms(NO_CANCEL, timeout_s=0.5))

    assert error.message == (
        "The commander did not answer a cancel of the move of left_arm from an earlier call in "
        "time, so an arm may still be moving. Wait a few seconds, then try again; if it fails "
        "again, tell the person that the arms cannot be checked."
    )
    assert error.data == {"unresolved": ["the move of left_arm"]}
    assert fake_commander.cancel_requests == 1
    error = refused(lambda: move(client, "right_arm", table(poses, "standby", "right")))
    assert error.message == unverified_refusal("move right_arm", ACCEPTED_LATE)
    assert "right_arm" not in [goal[0] for goal in fake_commander.joint_goals]
    # The commander accepts the cancel, and the goal ends without moving once
    # its plan is done.
    with caplog.at_level(logging.INFO, logger=commander_module.__name__):
        answer.set()
        assert wait_until(
            lambda: "Cancel of the move of left_arm, which the commander accepted late: the "
            "arms were told to stop." in caplog.text
        )
        plan.set()
        assert wait_until(
            lambda: "The move of left_arm from an earlier call is no longer active: its result "
            "came." in caplog.text
        )
    assert fake_commander.results == [("left_arm", "Canceled")]
    assert fake_commander.joints() == before
    assert client.read_arms(NO_CANCEL).left == ArmState("home", True)
    assert client.last_recovery.goals == (("the move of left_arm", "result"),)
    # Nothing is left that could stop the newer goal.
    fake_commander.move_time_s = 0.5
    assert move(client, "right_arm", table(poses, "standby", "right")) == "Success"
    assert fake_commander.results[-1] == ("right_arm", "Success")


def test_recovery_waits_for_the_answer_to_a_cancel_already_on_its_way(
    client, fake_commander, poses, monkeypatch
):
    # As above, but the commander answers the held cancel while recovery
    # waits, so recovery uses that answer instead of sending another cancel.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 5.0)
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    before = fake_commander.joints()
    accept = threading.Event()
    fake_commander.accept_gate = accept
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": "pending"}
    assert wait_until(lambda: client.busy() is None)
    plan = threading.Event()
    answer = threading.Event()
    fake_commander.plan_gate = plan
    fake_commander.cancel_gate = answer
    accept.set()
    assert wait_until(lambda: fake_commander.cancel_requests == 1)
    assert client.unverified == ACCEPTED_LATE
    # The cancel is answered 0.3 s into recovery, and the plan is done 0.3 s
    # after that, when the goal sees the cancel and ends without moving.
    timers = [threading.Timer(0.3, answer.set), threading.Timer(0.6, plan.set)]
    for timer in timers:
        timer.start()

    try:
        state = client.read_arms(NO_CANCEL)
    finally:
        for timer in timers:
            timer.join()

    assert state.left == ArmState("home", True)
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "result"),)
    assert fake_commander.cancel_requests == 1
    assert fake_commander.results == [("left_arm", "Canceled")]
    assert fake_commander.joints() == before


def test_a_goal_accepted_late_after_recovery_resolved_it_is_unresolved_again(
    client, fake_commander, poses, monkeypatch
):
    # The commander answers the send only after the call gave up, so
    # recovery's cancel finds no goal and resolves it as unknown.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.3)
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    client.joint_names("right_arm", NO_CANCEL)
    before = fake_commander.joints()
    accept = threading.Event()
    fake_commander.accept_gate = accept
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": "pending"}
    assert wait_until(lambda: client.busy() is None)
    client.read_arms(NO_CANCEL)
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "unknown"),)
    # Accepted after all, the goal plans until the test lets it finish. The
    # commander accepts the cancel that follows the acceptance.
    plan = threading.Event()
    fake_commander.plan_gate = plan
    accept.set()
    assert wait_until(lambda: client.unverified == ACCEPTED_LATE)

    error = refused(lambda: client.read_arms(NO_CANCEL))

    # The arms are still, but the goal is not over until its result comes.
    assert error.message == (
        "The commander accepted the cancel of the move of left_arm from an earlier call, and "
        "its end has not come yet. Wait a few seconds, then try again."
    )
    assert error.data == {"unresolved": ["the move of left_arm"]}
    error = refused(lambda: move(client, "right_arm", table(poses, "standby", "right")))
    assert error.message == unverified_refusal("move right_arm", ACCEPTED_LATE)
    assert "right_arm" not in [goal[0] for goal in fake_commander.joint_goals]
    # Once the plan is done, the goal sees the cancel and ends without moving.
    plan.set()
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Canceled")])
    assert client.read_arms(NO_CANCEL).left == ArmState("home", True)
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "result"),)
    assert fake_commander.joints() == before


def test_recovery_cancels_a_goal_whose_send_was_never_answered_by_its_id(
    client, fake_commander, poses, monkeypatch
):
    # The commander accepts the goal, which then plans, but its answer to
    # the send never reaches the client, which so has no goal handle.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.3)
    plan = threading.Event()
    fake_commander.plan_gate = plan
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    before = fake_commander.joints()
    send_goal_async = client._set_joint_position.send_goal_async

    def send_and_lose_the_answer(goal, **kwargs):
        send_goal_async(goal, **kwargs)
        return Future()

    monkeypatch.setattr(client._set_joint_position, "send_goal_async", send_and_lose_the_answer)
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": "pending"}
    assert wait_until(lambda: client.busy() is None)
    assert client.unverified == ABANDONED

    error = refused(lambda: client.read_arms(NO_CANCEL))

    # Cancelled by the id the client made, which the commander accepted.
    assert fake_commander.cancel_requests == 1
    assert error.message == (
        "The commander accepted the cancel of the move of left_arm from an earlier call, and "
        "its end has not come yet. Wait a few seconds, then try again."
    )
    # Its result, asked for by the same id, comes once the plan is done.
    plan.set()
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Canceled")])
    assert client.read_arms(NO_CANCEL).left == ArmState("home", True)
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "result"),)
    assert fake_commander.joints() == before


def test_recovery_asks_again_for_a_result_whose_answer_was_lost(
    client, fake_commander, poses, monkeypatch
):
    # The commander accepts the call's cancel and ends the goal, but its
    # answer to the call's request for the result never reaches the client,
    # as when discovery dropped meanwhile. The commander then answers
    # recovery's cancel that the goal ended, which after an accepted cancel
    # only the result can confirm.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.3)
    fake_commander.move_time_s = 2.0
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    get_result_async = ClientGoalHandle.get_result_async
    requests = []

    def lose_the_first_answer(handle):
        future = get_result_async(handle)
        requests.append(future)
        return Future() if len(requests) == 1 else future

    monkeypatch.setattr(ClientGoalHandle, "get_result_async", lose_the_first_answer)
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": "accepted"}
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Execution Failed")])
    assert wait_until(lambda: client.busy() is None)
    assert client.unverified == ABANDONED
    count = client.unverified_count

    client.read_arms(NO_CANCEL)

    assert len(requests) == 2
    assert client.unverified is None
    assert client.last_recovery == Recovery(count, (("the move of left_arm", "result"),), ())


GRIPPER_ABANDONED = (
    "the left_gripper command from an earlier call was given up on before its end was known"
)


def test_recovery_waits_for_an_abandoned_gripper_command_and_reads_its_finger(
    client, fake_commander, monkeypatch
):
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.3)
    # The commander accepts the cancel, but the fingers finish their 2 s ramp.
    fake_commander.gripper_time_s = 2.0
    client.read_arms(NO_CANCEL)
    with pytest.raises(StepError) as error:
        client.control_gripper("left_gripper", 0.0, NO_CANCEL, timeout_s=0.3)
    assert error.value.data == {"cancel": "accepted"}
    assert wait_until(lambda: client.busy() is None)
    assert client.unverified == GRIPPER_ABANDONED

    error = refused(lambda: client.read_arms(NO_CANCEL))

    assert error.message == (
        "The commander accepted the cancel of the left_gripper command from an earlier call, "
        "and its end has not come yet. Wait a few seconds, then try again."
    )
    assert error.data == {"unresolved": ["the left_gripper command"]}
    # The ramp ends, and its result comes. The settled read then reads the
    # finger too, so a reading of the gripper that lacks it refuses.
    assert wait_until(lambda: fake_commander.results == [("left_gripper", "Canceled")])
    fake_commander.missing_joints = {"openarm_left_finger_joint1"}
    with pytest.raises(StepError) as error:
        client.read_arms(NO_CANCEL)
    assert (error.value.code, error.value.message) == (
        "INTERNAL",
        "The commander answered 0 joint names and 0 positions for left_gripper.",
    )
    assert client.unverified == GRIPPER_ABANDONED
    fake_commander.missing_joints = set()
    count = client.unverified_count
    requests = len(fake_commander.joint_state_requests)
    client.read_arms(NO_CANCEL)
    assert fake_commander.joint_state_requests[requests:] == [
        "both_arms", "left_gripper", "both_arms", "left_gripper"
    ]
    assert client.unverified is None
    assert client.last_recovery == Recovery(
        count, (("the left_gripper command", "result"),), ("left_gripper",)
    )


def test_a_settled_read_refuses_while_a_gripper_still_moves(client, fake_commander, monkeypatch):
    # The commander restarts during a close. The new one does not know the
    # command, whose ramp goes on.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    fake_commander.gripper_time_s = 2.5
    client.read_arms(NO_CANCEL)
    with pytest.raises(StepError) as error:
        client.control_gripper("left_gripper", 0.0, NO_CANCEL, timeout_s=0.3)
    assert error.value.data == {"cancel": "accepted"}
    restart(client, fake_commander, CONTROL_GRIPPER)
    assert wait_until(lambda: client.busy() is None)

    error = refused(lambda: client.read_arms(NO_CANCEL))

    assert error.message.startswith(
        "A gripper is still moving: openarm_left_finger_joint1 moved 0.0"
    )
    assert error.message.endswith(
        " m between two reads 0.3 s apart. Wait a few seconds, then try again."
    )
    assert error.data["joint"] == "openarm_left_finger_joint1"
    assert error.data["moved_m"] > commander_module.SETTLE_TOLERANCE_M
    assert client.unverified == GRIPPER_ABANDONED
    # Once the ramp ends, the next settled read shows the finger still.
    assert wait_until(lambda: fake_commander.results, 4.0)
    count = client.unverified_count
    client.read_arms(NO_CANCEL)
    assert client.unverified is None
    assert client.last_recovery == Recovery(
        count, (("the left_gripper command", "unknown"),), ("left_gripper",)
    )


@pytest.mark.parametrize("during", ["first read", "settle"])
def test_a_goal_accepted_late_during_a_settled_read_keeps_the_arms_unverified(
    client, fake_commander, poses, monkeypatch, caplog, during
):
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    monkeypatch.setattr(commander_module, "SETTLE_S", 0.5)
    client.read_arms(NO_CANCEL)
    client.joint_names("left_arm", NO_CANCEL)
    before = fake_commander.joints()
    gate = threading.Event()
    fake_commander.accept_gate = gate
    # Cancelled while it plans, so it never moves.
    fake_commander.plan_time_s = 0.5
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": "pending"}
    assert wait_until(lambda: client.busy() is None)
    assert client.unverified == ABANDONED

    if during == "first read":
        # The commander accepts the goal while it holds the settled read's
        # first answer back, so the mark must be taken before that read.
        fake_commander.joint_states_delay_s = 0.5

    def accept_during_the_settle():
        # The settled read's first request: the commander accepts the goal
        # before it answers that request, or before the second request.
        assert wait_until(lambda: fake_commander.joint_state_requests.count("both_arms") == 2)
        gate.set()

    releaser = threading.Thread(target=accept_during_the_settle)
    releaser.start()
    with caplog.at_level(logging.INFO, logger=commander_module.__name__):
        error = refused(lambda: client.read_arms(NO_CANCEL))
        releaser.join(timeout=5)
        fake_commander.joint_states_delay_s = 0.0

        assert error.message == (
            f"While the arms were being checked, {ACCEPTED_LATE}. Wait a few seconds, then try "
            "again."
        )
        assert error.data == {"unverified": ACCEPTED_LATE}
        assert client.unverified == ACCEPTED_LATE
        assert client.last_state == ArmsState()
        assert wait_until(
            lambda: "Cancel of the move of left_arm, which the commander accepted late: the "
            "arms were told to stop." in caplog.text
        )
    assert "The move of left_arm had been abandoned, so the arms are unverified" in caplog.text
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Canceled")])
    assert fake_commander.joints() == before
    assert client.read_arms(NO_CANCEL).left == ArmState("home", True)
    assert client.unverified is None


@pytest.mark.parametrize("reject_cancels", [False, True], ids=["accepted", "refused"])
def test_a_goal_accepted_late_fails_a_newer_move_that_its_cancel_may_stop(
    client, fake_commander, poses, monkeypatch, caplog, reject_cancels
):
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    client.joint_names("left_arm", NO_CANCEL)
    client.joint_names("right_arm", NO_CANCEL)
    gate = threading.Event()
    fake_commander.accept_gate = gate
    # Given up on before the commander answered it, then abandoned.
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": "pending"}
    assert wait_until(lambda: client.busy() is None)
    # A settled read verifies the arms, so goals are taken again.
    home = ArmState("home", True)
    assert client.read_arms(NO_CANCEL) == ArmsState(left=home, right=home)

    fake_commander.accept_gate = None
    fake_commander.move_time_s = 1.0
    fake_commander.reject_cancels = reject_cancels
    moved = arm_moved(fake_commander, poses, "standby", "right")
    errors = []

    def move_right_arm():
        try:
            move(client, "right_arm", table(poses, "standby", "right"))
        except StepError as e:
            errors.append(e)

    moving = threading.Thread(target=move_right_arm)
    with caplog.at_level(logging.INFO, logger=commander_module.__name__):
        moving.start()
        assert wait_until(moved)
        # The commander accepts the old goal while the newer one moves.
        gate.set()
        moving.join(timeout=5)
        answer = (
            "the commander refused the cancel, so an arm may still be moving"
            if reject_cancels
            else "the arms were told to stop"
        )
        assert wait_until(
            lambda: f"Cancel of the move of left_arm, which the commander accepted late: {answer}."
            in caplog.text
        )
        # A refused cancel is a warning; an accepted one only information.
        [logged] = [
            record
            for record in caplog.records
            if record.getMessage().startswith("Cancel of the move of left_arm, which")
        ]
        assert logged.levelno == (logging.WARNING if reject_cancels else logging.INFO)

    disturbed = f"{ACCEPTED_LATE}, and a cancel it accepts stops both arms"
    assert [(e.code, e.message, e.data) for e in errors] == [
        (
            "SKILL_FAILED",
            f"The move of right_arm may have been stopped part way: {disturbed}. Call "
            "get_arm_state to see where the arms are.",
            {"group": "right_arm", "disturbed": disturbed},
        )
    ]
    assert (
        "The move of right_arm may be stopped part way by the cancel of the move of left_arm."
        in caplog.text
    )
    assert client.unverified == ACCEPTED_LATE
    assert client.last_state == ArmsState()
    # An accepted cancel of the old goal stopped the newer one too.
    status = "Success" if reject_cancels else "Execution Failed"
    assert ("right_arm", status) in fake_commander.results


def test_a_goal_accepted_late_just_before_a_newer_send_refuses_that_send(
    client, fake_commander, poses, monkeypatch
):
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    client.joint_names("left_arm", NO_CANCEL)
    client.joint_names("right_arm", NO_CANCEL)
    gate = threading.Event()
    fake_commander.accept_gate = gate
    with pytest.raises(StepError):
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: client.busy() is None)
    client.read_arms(NO_CANCEL)
    assert client.unverified is None
    fake_commander.accept_gate = None
    server_is_ready = client._set_joint_position.server_is_ready

    def accepted_after_the_check() -> bool:
        # The newer call has passed the checks of _exclusive(), and the
        # commander accepts the old goal before the newer one is sent.
        gate.set()
        assert wait_until(lambda: client.unverified == ACCEPTED_LATE)
        return server_is_ready()

    monkeypatch.setattr(client._set_joint_position, "server_is_ready", accepted_after_the_check)

    error = refused(lambda: move(client, "right_arm", table(poses, "standby", "right")))

    assert error.message == unverified_refusal("move right_arm", ACCEPTED_LATE)
    assert error.data == {"unverified": ACCEPTED_LATE}
    # The newer goal was never sent, so the old goal's cancel cannot stop it.
    assert "right_arm" not in [goal[0] for goal in fake_commander.joint_goals]


def test_a_late_rejection_of_an_abandoned_goal_changes_nothing(
    client, fake_commander, poses, monkeypatch, caplog
):
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    client.joint_names("left_arm", NO_CANCEL)
    client.joint_names("right_arm", NO_CANCEL)
    gate = threading.Event()
    fake_commander.accept_gate = gate
    with pytest.raises(StepError):
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert wait_until(lambda: client.busy() is None)
    client.read_arms(NO_CANCEL)
    fake_commander.accept_gate = None
    fake_commander.move_time_s = 1.0
    moved = arm_moved(fake_commander, poses, "standby", "right")
    outcome = []
    moving = threading.Thread(
        target=lambda: outcome.append(move(client, "right_arm", table(poses, "standby", "right")))
    )

    with caplog.at_level(logging.DEBUG, logger=commander_module.__name__):
        moving.start()
        assert wait_until(moved)
        fake_commander.reject_goals = True
        gate.set()
        assert wait_until(
            lambda: "The commander did not accept the move of left_arm, which was abandoned."
            in caplog.text
        )
        # The newer goal's record is untouched.
        record = client._goal
        assert record is not None
        assert (record.what, record.disturbed) == ("the move of right_arm", None)
        assert client.unverified is None
        moving.join(timeout=5)

    assert outcome == ["Success"]
    assert fake_commander.cancel_requests == 0
    assert fake_commander.results == [("right_arm", "Success")]


def test_a_late_result_of_an_abandoned_goal_does_not_end_a_newer_one(
    client, fake_commander, poses, monkeypatch
):
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    # The commander refuses the call's cancel, and the move's result comes
    # 2 s after it ends.
    fake_commander.reject_cancels = True
    fake_commander.move_time_s = 0.5
    fake_commander.result_delay_s = 2.0
    client.joint_names("left_arm", NO_CANCEL)
    client.joint_names("right_arm", NO_CANCEL)
    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)
    assert error.value.data == {"cancel": "refused"}
    older = client._goal
    assert wait_until(lambda: client.busy() is None)
    # The move has ended, so the commander answers recovery's cancel that it
    # ended, which resolves it before its result comes.
    assert wait_until(lambda: fake_commander.results == [("left_arm", "Success")])
    client.read_arms(NO_CANCEL)
    assert client.unverified is None
    assert client.last_recovery.goals == (("the move of left_arm", "ended"),)
    fake_commander.reject_cancels = False
    # Held until its call cancels it.
    fake_commander.never_answer = True
    cancel = threading.Event()
    errors = []

    def move_right_arm():
        try:
            move(client, "right_arm", table(poses, "standby", "right"), cancel=cancel)
        except StepError as e:
            errors.append(e)

    moving = threading.Thread(target=move_right_arm)
    moving.start()
    assert wait_until(lambda: fake_commander.joint_goals[-1][0] == "right_arm")
    newer = client._goal
    assert newer is not None and newer is not older
    assert not older.result.done()

    # The old goal's result comes while the newer one runs.
    assert wait_until(older.result.done)
    fake_commander.result_delay_s = 0.0
    assert client._goal is newer
    cancel.set()
    moving.join(timeout=5)
    assert [e.code for e in errors] == ["CANCELLED"]


class _LockThatAnotherCallTakesFirst:
    """The call lock, taken once by another call while the first acquire() waits.

    That call leaves its goal outstanding, so a call that passed the check
    before it waited for the lock must check again once it holds it.
    """

    def __init__(self, client, other_call):
        self._client = client
        self._lock = client._lock
        self._other_call = other_call

    def acquire(self, timeout=-1):
        self._client._lock = self._lock
        with pytest.raises(StepError) as error:
            self._other_call()
        assert error.value.code == "TIMEOUT"
        return self._lock.acquire(timeout=timeout)


def test_a_call_that_waited_for_the_lock_is_refused_once_it_holds_it(
    client, fake_commander, poses, monkeypatch
):
    fake_commander.never_answer = True
    fake_commander.reject_cancels = True
    client.joint_names("left_arm", NO_CANCEL)

    def other_move():
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)

    monkeypatch.setattr(client, "_lock", _LockThatAnotherCallTakesFirst(client, other_move))

    error = refused(lambda: client.read_arms(NO_CANCEL))

    assert error.message == (
        "Cannot read the joints of both_arms: the move of left_arm from an earlier call has not "
        "ended yet. Try again in a few seconds."
    )
    assert fake_commander.joint_state_requests == ["left_arm"]


def test_a_refused_cancel_says_the_arm_may_still_move(client, fake_commander, poses, caplog):
    fake_commander.move_time_s = 1.0
    fake_commander.reject_cancels = True
    client.joint_names("left_arm", NO_CANCEL)

    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        with pytest.raises(StepError) as error:
            move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)

    assert error.value.message == (
        "The move of left_arm did not finish within 0.3 s; the commander refused the cancel, "
        "so an arm may still be moving."
    )
    assert error.value.data == {"cancel": "refused"}
    assert "Cancel of the move of left_arm: the commander refused the cancel" in caplog.text
    assert fake_commander.cancel_requests == 1
    assert client.busy() == "the move of left_arm from an earlier call"
    # The move went on to its end.
    assert wait_until(lambda: client.busy() is None)
    assert fake_commander.results == [("left_arm", "Success")]
    assert client.read_arms(NO_CANCEL).left == ArmState("standby", False)


def test_a_cancel_after_the_goal_ended_says_so(client, fake_commander, poses):
    # The move ends at 0.2 s, but its result is only sent at 1.2 s.
    fake_commander.result_delay_s = 1.0
    client.joint_names("left_arm", NO_CANCEL)

    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.5)

    assert error.value.message == (
        "The move of left_arm did not finish within 0.5 s; the goal had already ended."
    )
    assert error.value.data == {"cancel": "ended"}
    # The commander answers ERROR_GOAL_TERMINATED without calling its
    # cancel callback.
    assert fake_commander.cancel_requests == 0
    assert client.busy() == "the move of left_arm from an earlier call"
    assert wait_until(lambda: client.busy() is None)
    assert fake_commander.results == [("left_arm", "Success")]


def test_an_unanswered_cancel_says_the_arm_may_still_move(
    client, fake_commander, poses, monkeypatch
):
    monkeypatch.setattr(commander_module, "CANCEL_ANSWER_S", 0.2)
    fake_commander.move_time_s = 2.0
    fake_commander.cancel_delay_s = 0.5
    client.joint_names("left_arm", NO_CANCEL)
    start = time.monotonic()

    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)

    assert error.value.message == (
        "The move of left_arm did not finish within 0.3 s; the commander did not answer the "
        "cancel, so an arm may still be moving."
    )
    assert error.value.data == {"cancel": "no answer"}
    assert time.monotonic() - start < 0.8
    assert client.busy() == "the move of left_arm from an earlier call"
    # The late answer stops the arm.
    assert wait_until(lambda: client.busy() is None)
    assert fake_commander.results == [("left_arm", "Execution Failed")]


GOAL_ID = UUID(uuid=list(range(16)))
OTHER_ID = UUID(uuid=list(range(1, 17)))


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        (CancelGoal.Response(goals_canceling=[GoalInfo(goal_id=GOAL_ID)]), "accepted"),
        # An rclpy server's refusal, and an rclcpp server's.
        (CancelGoal.Response(), "refused"),
        (CancelGoal.Response(goals_canceling=[GoalInfo(goal_id=OTHER_ID)]), "refused"),
        (CancelGoal.Response(return_code=CancelGoal.Response.ERROR_REJECTED), "refused"),
        (CancelGoal.Response(return_code=CancelGoal.Response.ERROR_UNKNOWN_GOAL_ID), "unknown"),
        (CancelGoal.Response(return_code=CancelGoal.Response.ERROR_GOAL_TERMINATED), "ended"),
        (None, "no answer"),
    ],
)
def test_the_cancel_outcome_follows_the_commanders_answer(answer, outcome):
    assert commander_module._cancel_outcome(answer, GOAL_ID) == outcome


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
        "A stop or a reset interrupted the move of left_arm; the arms were told to stop."
    )
    assert error.value.data == {"cancel": "accepted"}
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
        "The left_gripper command did not finish within 0.3 s; the commander accepted the "
        "cancel, but the gripper finishes its current motion."
    )
    # Stopped somewhere unknown.
    assert client.last_gripper_position("left_gripper") is None
    assert wait_until(lambda: fake_commander.results[-1:] == [("left_gripper", "Canceled")])


def test_an_older_gripper_call_cannot_overwrite_the_position_of_a_newer_one(client, monkeypatch):
    remember = client._remember_gripper
    paused = threading.Event()
    resume = threading.Event()

    def remember_slowly(group, position):
        # The close's result, kept back until the main thread has tried
        # its own command.
        if position == 0.0:
            paused.set()
            assert resume.wait(5.0)
        remember(group, position)

    monkeypatch.setattr(client, "_remember_gripper", remember_slowly)
    older = threading.Thread(
        target=lambda: client.control_gripper("left_gripper", 0.0, NO_CANCEL)
    )
    older.start()
    assert paused.wait(5.0)

    # The older call still holds the call lock while it records the close.
    try:
        with pytest.raises(StepError) as error:
            client.control_gripper("left_gripper", 0.044, NO_CANCEL)
    finally:
        resume.set()
        older.join(timeout=5)

    assert error.value.code == "PRECONDITION_FAILED"
    assert client.last_gripper_position("left_gripper") == 0.0
    client.control_gripper("left_gripper", 0.044, NO_CANCEL)
    assert client.last_gripper_position("left_gripper") == 0.044


def test_a_gripper_command_superseded_from_outside_ends_failed_and_unmeasured(
    client, fake_commander
):
    fake_commander.gripper_time_s = 1.0
    results = []
    closing = threading.Thread(
        target=lambda: results.append(client.control_gripper("left_gripper", 0.0, NO_CANCEL))
    )
    closing.start()
    assert wait_until(lambda: fake_commander.joints()["openarm_left_finger_joint1"] < 0.04)

    # Another node commands the gripper while the close ramps.
    fake_commander.supersede("left_gripper", 0.044)
    closing.join(timeout=5)

    # The commander's answer looks like a measured empty close.
    [result] = results
    assert (result.success, result.status, result.grasped, result.gripper_gap) == (
        False,
        "Failed",
        False,
        0.0,
    )
    assert fake_commander.joints()["openarm_left_finger_joint1"] == 0.044
    assert fake_commander.results == [("left_gripper", "Failed")]


def test_a_read_since_a_gripper_command_is_refused_once_a_newer_one_came(client, fake_commander):
    close = client.control_gripper("left_gripper", 0.0, NO_CANCEL)
    assert client.joint_states_since("left_gripper", close.command, NO_CANCEL) == pytest.approx(
        {"openarm_left_finger_joint1": 0.0}
    )
    opened = client.control_gripper("left_gripper", 0.044, NO_CANCEL)
    assert opened.command == close.command + 1
    # The other gripper counts its own commands.
    client.control_gripper("right_gripper", 0.0, NO_CANCEL)
    requests = len(fake_commander.joint_state_requests)

    error = refused(lambda: client.joint_states_since("left_gripper", close.command, NO_CANCEL))

    assert error.message == (
        "Another left_gripper command came after this one, so a read now would not show "
        "where this one left the fingers."
    )
    assert error.data == {"group": "left_gripper", "command": close.command}
    assert len(fake_commander.joint_state_requests) == requests
    assert client.joint_states_since("left_gripper", opened.command, NO_CANCEL) == pytest.approx(
        {"openarm_left_finger_joint1": 0.044}
    )


def test_a_gripper_call_that_sends_nothing_does_not_count_as_a_command(
    client, fake_commander, monkeypatch
):
    monkeypatch.setattr(commander_module, "READY_WAIT_S", 0.2)
    close = client.control_gripper("left_gripper", 0.0, NO_CANCEL)
    fake_commander.vanish("control_gripper")
    assert wait_until(lambda: not client._control_gripper.server_is_ready())

    with pytest.raises(StepError) as error:
        client.control_gripper("left_gripper", 0.044, NO_CANCEL)

    assert error.value.code == "UNAVAILABLE"
    # Nothing was sent, so the close is still the gripper's latest command.
    assert client.joint_states_since("left_gripper", close.command, NO_CANCEL) == pytest.approx(
        {"openarm_left_finger_joint1": 0.0}
    )


def test_a_gripper_goal_accepted_late_refuses_the_read_after_a_newer_close(
    client, fake_commander, monkeypatch, caplog
):
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.3)
    gate = threading.Event()
    fake_commander.accept_gate = gate
    with pytest.raises(StepError) as error:
        client.control_gripper("left_gripper", 0.044, NO_CANCEL, timeout_s=0.3)
    assert error.value.data == {"cancel": "pending"}
    assert wait_until(lambda: client.busy() is None)
    assert client.unverified == (
        "the left_gripper command from an earlier call was given up on before its end was known"
    )
    client.read_arms(NO_CANCEL)
    fake_commander.accept_gate = None
    close = client.control_gripper("left_gripper", 0.0, NO_CANCEL)
    assert client.last_gripper_position("left_gripper") == 0.0
    late = (
        "the commander accepted the left_gripper command from an earlier call late, and the "
        "gripper makes that motion although it was asked to cancel it"
    )

    with caplog.at_level(logging.INFO, logger=commander_module.__name__):
        gate.set()
        assert wait_until(lambda: client.unverified == late)
        error = refused(
            lambda: client.joint_states_since("left_gripper", close.command, NO_CANCEL)
        )
        assert error.data == {"group": "left_gripper", "command": close.command}
        assert wait_until(
            lambda: "Cancel of the left_gripper command, which the commander accepted late: the "
            "commander accepted the cancel, but the gripper finishes its current motion."
            in caplog.text
        )

    # The fingers make the old command's motion after the close.
    assert wait_until(lambda: fake_commander.results[-1:] == [("left_gripper", "Canceled")])
    assert fake_commander.joints()["openarm_left_finger_joint1"] == pytest.approx(0.044)
    assert client.last_gripper_position("left_gripper") is None
    error = refused(lambda: client.control_gripper("left_gripper", 0.0, NO_CANCEL))
    assert error.data == {"unverified": late}


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
