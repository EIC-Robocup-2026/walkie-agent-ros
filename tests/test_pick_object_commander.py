"""pick_object through RosArms and a real CommanderClient against FakeCommander over ROS.

tests/test_pick_object.py covers the pick's steps over FakeRobot. These tests cover
what the commander is left with: its allow_gripper_vs_octomap after every way a pick
can end, read back from the fake commander's own parameter, and the arm goals it got
when the pick refused before reaching. The base and the object memory are the same
stand-ins as there: FakeRobot without arm poses, so its drives check no arms, and
Remembers.
"""

import threading
from dataclasses import replace

import pytest
from rclpy.parameter import Parameter

from conftest import PROJECT_ROOT
from test_object_memory_atomics import WORLD, Remembers
from test_pick_object import read_again, remembered
from test_ros_arms import Spinner, wait_until
from walkie_agent.arm.poses import GRIPPER_GROUPS, load_arm_poses, load_robot_model
from walkie_agent.capabilities import Ctx
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.errors import ErrorCode, GoalFailed, StepError
from walkie_agent.ros import arms as arms_module
from walkie_agent.ros import commander as commander_module
from walkie_agent.ros.arms import RosArms
from walkie_agent.ros.commander import CommanderClient
from walkie_agent.sim.fake_commander import FakeCommander
from walkie_agent.tools.atomic import pick_object

TOUCH = "allow_gripper_vs_octomap"
# The travel of one finger on the water bottle: inside the commander's grasp band.
BOTTLE_WIDTH_M = 0.02


@pytest.fixture(autouse=True)
def project_dir(monkeypatch):
    # pick_object reads config/pick.toml from the working directory, as main.py does.
    monkeypatch.chdir(PROJECT_ROOT)


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    """No settle after a change of the allowance, and quick reads before a straight move."""
    monkeypatch.setattr(commander_module, "TOUCH_SETTLE_S", 0.0)
    monkeypatch.setattr(arms_module, "STILL_POLL_S", 0.01)


@pytest.fixture(scope="module")
def model():
    return load_robot_model(PROJECT_ROOT / "config" / "robot_model.toml")


@pytest.fixture(scope="module")
def poses():
    return load_arm_poses(
        PROJECT_ROOT / "config" / "arm_poses.toml", PROJECT_ROOT / "config" / "robot_model.toml"
    )


@pytest.fixture
def fake_commander(ros, model):
    node = FakeCommander(model, move_time_s=0.05, gripper_time_s=0.02)
    node.object_widths[GRIPPER_GROUPS["right"]] = BOTTLE_WIDTH_M
    spinner = Spinner(node)
    yield node
    spinner.close()


@pytest.fixture
def robot():
    return FakeRobot(WORLD)


@pytest.fixture
def client(connector, robot, poses, fake_commander):
    made = CommanderClient(connector, robot.cancel, poses)
    assert wait_until(made.available), "The client never discovered the fake commander."
    yield made
    made.destroy()


@pytest.fixture
def ctx(robot, client):
    arms = RosArms(client, robot.cancel, sim=True)
    # As main does at startup.
    arms.reset_touch_at_start()
    return Ctx(
        cancel=robot.cancel,
        base=robot,
        speech=robot,
        vision=robot,
        arms=arms,
        object_memory=Remembers(remembered()),
    )


def allowance(fake: FakeCommander) -> bool:
    return fake.get_parameter(TOUCH).value


def before_the_reset(monkeypatch, client: CommanderClient, fake: FakeCommander) -> list[bool]:
    """The commander's allowance as each reset of it began, so a test sees it was on."""
    seen: list[bool] = []
    real = client.restore_gripper_touch

    def recording(timeout_s=None):
        seen.append(allowance(fake))
        real(timeout_s)

    monkeypatch.setattr(client, "restore_gripper_touch", recording)
    return seen


def after_the_close(monkeypatch, ctx: Ctx, then) -> None:
    """Run then() right after the right gripper's close, inside the pick."""
    real = ctx.arms.gripper

    def gripper(arm, state):
        moved = real(arm, state)
        if state == "close":
            then()
        return moved

    monkeypatch.setattr(ctx.arms, "gripper", gripper)


def fails(ctx: Ctx) -> StepError:
    with pytest.raises(StepError) as raised:
        pick_object(ctx, "water bottle")
    return raised.value


# Every way out of the grasp


def test_a_pick_that_holds_the_object_ends_with_the_allowance_off(
    ctx, client, fake_commander, monkeypatch
):
    seen = before_the_reset(monkeypatch, client, fake_commander)

    answer = pick_object(ctx, "water bottle")

    assert answer.startswith("Picked up the water bottle with the right hand")
    assert allowance(fake_commander) is False
    # The four reaches: whole body, then the approach, lift and move clear.
    assert len(fake_commander.pose_goals) == 4
    # The last move home had already turned it off; the reset writes it off anyway.
    assert seen == [False]


def test_a_failed_lift_with_the_allowance_on_ends_with_it_off(
    ctx, client, fake_commander, monkeypatch
):
    seen = before_the_reset(monkeypatch, client, fake_commander)

    def fail_the_next_pose_goal():
        fake_commander.pose_failure = "Planning Failed"

    after_the_close(monkeypatch, ctx, fail_the_next_pose_goal)

    error = fails(ctx)

    assert isinstance(error, GoalFailed)
    assert error.message.startswith("The right hand holds the water bottle, but the lift failed")
    assert seen == [True]
    assert allowance(fake_commander) is False


def test_an_unexpected_error_with_the_allowance_on_ends_with_it_off(
    ctx, client, fake_commander, monkeypatch
):
    seen = before_the_reset(monkeypatch, client, fake_commander)

    def broken():
        raise RuntimeError("a bug after the close")

    after_the_close(monkeypatch, ctx, broken)

    with pytest.raises(RuntimeError, match="a bug after the close"):
        pick_object(ctx, "water bottle")

    assert seen == [True]
    assert allowance(fake_commander) is False


def test_a_stop_during_the_approach_turns_the_allowance_off_and_sends_no_motion_after(
    ctx, client, fake_commander, robot, monkeypatch
):
    seen = before_the_reset(monkeypatch, client, fake_commander)
    real = client.go_to_pose
    stops: list[threading.Timer] = []

    def stopped_while_moving(group, pose, timeout_s, *, cartesian=False):
        if cartesian and not stops:
            # The approach, with the allowance on: a stop comes while the arm moves.
            fake_commander.move_time_s = 2.0
            stops.append(threading.Timer(0.2, robot.cancel.cancel))
            stops[0].start()
        return real(group, pose, timeout_s, cartesian=cartesian)

    monkeypatch.setattr(client, "go_to_pose", stopped_while_moving)

    error = fails(ctx)
    stops[0].join()

    assert error.code is ErrorCode.CANCELLED
    assert seen == [True]
    assert allowance(fake_commander) is False
    # The pre-grasp reach and the approach; no move home or any other motion after.
    assert len(fake_commander.pose_goals) == 2
    assert [group for group, _ in fake_commander.joint_goals] == ["left_arm"]
    assert [group for group, _ in fake_commander.gripper_goals] == ["right_gripper"]


def test_a_reset_that_fails_fails_the_pick_and_no_arm_moves_until_one_works(
    ctx, client, fake_commander, monkeypatch
):
    def unanswered(timeout_s=None):
        raise StepError(ErrorCode.TIMEOUT, "The commander did not answer.")

    def fail_the_next_pose_goal():
        fake_commander.pose_failure = "Planning Failed"

    monkeypatch.setattr(client, "restore_gripper_touch", unanswered)
    after_the_close(monkeypatch, ctx, fail_the_next_pose_goal)

    error = fails(ctx)

    assert error.message.startswith(
        "Walkie could not turn off its fingers' allowance to touch what its cameras sense: "
        "The commander did not answer."
    )
    assert allowance(fake_commander) is True
    # The commander's parameter service is gone too: the next motion, of an arm or a
    # gripper, sends no goal.
    goals = (
        len(fake_commander.pose_goals),
        len(fake_commander.joint_goals),
        len(fake_commander.gripper_goals),
    )
    monkeypatch.setattr(client._set_parameters, "service_is_ready", lambda: False)
    monkeypatch.setattr(client, "_ready_wait_s", 0.1)
    with pytest.raises(StepError):
        ctx.arms.move_to("home", "right")
    with pytest.raises(StepError):
        ctx.arms.gripper("right", "open")
    assert (
        len(fake_commander.pose_goals),
        len(fake_commander.joint_goals),
        len(fake_commander.gripper_goals),
    ) == goals
    assert allowance(fake_commander) is True


@pytest.mark.parametrize("state", ["open", "half", "close"])
def test_after_a_failed_startup_reset_no_gripper_moves_until_the_allowance_is_set_off(
    robot, client, fake_commander, monkeypatch, state
):
    # An earlier run left the allowance on, and the commander's parameter service is
    # gone when the agent starts: the reset at startup fails and is only logged.
    fake_commander.set_parameters([Parameter(TOUCH, value=True)])
    ready = client._set_parameters.service_is_ready
    service = {"up": False}
    monkeypatch.setattr(
        client._set_parameters, "service_is_ready", lambda: service["up"] and ready()
    )
    monkeypatch.setattr(client, "_ready_wait_s", 0.1)
    arms = RosArms(client, robot.cancel, sim=True)
    arms.reset_touch_at_start()
    assert allowance(fake_commander) is True

    with pytest.raises(StepError) as raised:
        arms.gripper("right", state)

    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert raised.value.message.startswith(
        "Did not move the right gripper: its fingers' allowance to touch what the cameras "
        "sense is unknown, and turning it off first failed: "
    )
    assert fake_commander.gripper_goals == []
    assert (fake_commander.joint_goals, fake_commander.pose_goals) == ([], [])
    assert allowance(fake_commander) is True

    # Once the service answers, the allowance goes off before the gripper's goal.
    service["up"] = True
    arms.gripper("right", state)
    assert allowance(fake_commander) is False
    assert [group for group, _ in fake_commander.gripper_goals] == ["right_gripper"]


# Refusals before the first arm goal


def test_a_restart_with_the_allowance_left_on_turns_it_off_before_the_pick(
    robot, client, fake_commander
):
    # As a run killed mid-grasp leaves the commander, which keeps it until it restarts.
    fake_commander.set_parameters([Parameter(TOUCH, value=True)])
    arms = RosArms(client, robot.cancel, sim=True)

    arms.reset_touch_at_start()

    assert allowance(fake_commander) is False
    assert (fake_commander.pose_goals, fake_commander.joint_goals) == ([], [])


@pytest.mark.parametrize(
    "answer",
    [
        StepError(ErrorCode.NOT_FOUND, "Walkie remembers no object 7."),
        replace(remembered(), state="held"),
        replace(remembered(), state="missing"),
        StepError(ErrorCode.UNAVAILABLE, "The object memory is not connected."),
        # Lowered out of the reach envelope by the sighting after the drive.
        replace(remembered(), z=0.48),
        # Seen after the drive further from where the drive aimed than allowed.
        replace(remembered(), y=-0.2),
    ],
    ids=["vanished", "held", "missing", "unreadable", "too-low", "moved"],
)
def test_a_refused_pick_sends_no_arm_goal_and_leaves_the_allowance_off(
    ctx, fake_commander, monkeypatch, answer
):
    read_again(monkeypatch, ctx, answer)

    error = fails(ctx)

    assert error.code is not ErrorCode.CANCELLED
    assert fake_commander.pose_goals == []
    assert fake_commander.joint_goals == []
    assert fake_commander.gripper_goals == []
    assert allowance(fake_commander) is False


# Refusals between the arm goals


@pytest.mark.parametrize(
    ("during", "pose_goals"), [("standby", 0), ("reach", 1)], ids=["standby", "reach"]
)
def test_an_object_taken_during_an_arm_motion_gets_no_further_goal(
    ctx, client, fake_commander, monkeypatch, during, pose_goals
):
    def taken():
        memory = ctx.object_memory
        memory.objects["7"] = replace(memory.objects["7"], state="held")

    if during == "standby":
        real_joint = client.set_joint_position

        def standby(group, positions, timeout_s):
            status = real_joint(group, positions, timeout_s)
            taken()
            return status

        monkeypatch.setattr(client, "set_joint_position", standby)
    else:
        real_pose = client.go_to_pose

        def reach(group, pose, timeout_s, *, cartesian=False):
            status = real_pose(group, pose, timeout_s, cartesian=cartesian)
            taken()
            return status

        monkeypatch.setattr(client, "go_to_pose", reach)

    error = fails(ctx)

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert "as held" in error.message
    # The other arm's standby and the open, then nothing after the motion it was taken in.
    assert [group for group, _ in fake_commander.joint_goals] == ["left_arm"]
    assert [group for group, _ in fake_commander.gripper_goals] == ["right_gripper"]
    assert len(fake_commander.pose_goals) == pose_goals
    assert allowance(fake_commander) is False
