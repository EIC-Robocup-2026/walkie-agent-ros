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
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from unique_identifier_msgs.msg import UUID

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


def test_a_goal_whose_result_never_comes_is_abandoned_after_the_limit(
    client, fake_commander, poses, monkeypatch, caplog
):
    # As when the commander restarted: its servers are still discovered,
    # but the old goal's result never comes.
    monkeypatch.setattr(commander_module, "ABANDON_AFTER_S", 0.5)
    fake_commander.move_time_s = 5.0
    fake_commander.result_delay_s = math.inf
    client.joint_names("left_arm", NO_CANCEL)

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
    # Stopped part way, which the next read shows.
    assert client.read_arms(NO_CANCEL).left == ArmState(None, False)


def test_a_goal_whose_server_vanishes_is_abandoned(
    client, fake_commander, poses, monkeypatch, caplog
):
    fake_commander.never_answer = True
    fake_commander.reject_cancels = True
    client.joint_names("left_arm", NO_CANCEL)

    with pytest.raises(StepError) as error:
        move(client, "left_arm", table(poses, "standby", "left"), timeout_s=0.3)

    assert error.value.code == "TIMEOUT"
    refused(lambda: client.read_arms(NO_CANCEL))
    assert client.busy() == "the move of left_arm from an earlier call"

    with caplog.at_level(logging.WARNING, logger=commander_module.__name__):
        fake_commander.vanish()
        assert wait_until(lambda: client.busy() is None)
    assert (
        "Abandoned the move of left_arm: its action server is no longer discovered"
        in caplog.text
    )
    # Taken again, and failing as without a commander.
    monkeypatch.setattr(commander_module, "READY_WAIT_S", 0.2)
    with pytest.raises(StepError) as error:
        client.read_arms(NO_CANCEL)
    assert error.value.code == "UNAVAILABLE"


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
