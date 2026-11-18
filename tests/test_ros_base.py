"""RosBase and its operation record, on a fake connector, a fake clock and real rclpy futures.

Each race is forced at one exact point: a hook inside a fake or a thread held on an event.
The waits that do expire are the wrapper's own budgets, and nothing sleeps to let a verb
get somewhere. A call that could deadlock runs under a watchdog, so a deadlock fails a test.
"""

import dataclasses
import inspect
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import astuple

import pytest
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from nav2_msgs.action import NavigateToPose
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.task import Future
from ros_fakes import (
    WATCHDOG_S,
    Background,
    FakeActionClient,
    FakeConnector,
    FakeGoalHandle,
    FakeNode,
    bounded,
    missing_tf,
    transform,
)

from walkie_agent.arm.answers import drive_verdict
from walkie_agent.arm.results import NOT_FOUND, DriveCheck
from walkie_agent.arm.state import ARMS_OUT, ArmsState, ArmState
from walkie_agent.capabilities import Ctx, PersonSeen
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import ErrorCode, GoalFailed, StepError
from walkie_agent.locations import Location
from walkie_agent.ros.base import RosBase
from walkie_agent.ros.commander import STOP_TOPIC
from walkie_agent.ros.policy import FORBIDDEN_NAMES
from walkie_agent.ros.speech import RosSpeech
from walkie_agent.ros.vision import UnavailableVision
from walkie_agent.tools.atomic import face_person, move_relative

PLACES = {
    "kitchen": Location("kitchen", 1.0, 2.0, math.pi / 2),
    "sofa": Location("sofa", -1.0, 0.5, 0.0),
}
# Long enough that only a test that passes a shorter one ever sees a budget expire.
LONG_S = 10 * WATCHDOG_S


class Rig:
    """A RosBase with its fakes. The budgets are long unless a test passes a shorter one."""

    def __init__(self, *, places=PLACES, token: CancelToken | None = None, **budgets) -> None:
        self.connector = FakeConnector()
        self.clock = self.connector.node.clock
        self.token = token or CancelToken()
        self.clients: list[FakeActionClient] = []
        self.base = RosBase(
            self.connector,
            places,
            self.token,
            client_factory=self._make_client,
            **{"discovery_s": LONG_S, "acceptance_s": LONG_S, "tf_s": LONG_S, **budgets},
        )
        self.client = self.clients[0]

    def _make_client(self, *args, **kwargs) -> FakeActionClient:
        self.clients.append(FakeActionClient(*args, **kwargs))
        return self.clients[-1]

    def accept(self, **kwargs):
        return bounded(self.client.accept, **kwargs)

    def stop(self) -> None:
        bounded(self.base.stop)

    def assert_gate_held(self, what: str = "") -> None:
        """A new motion is refused before it touches ROS, and it replaces nothing."""
        polls, goals = self.client.ready_polls, len(self.client.goals)
        self.token.reset()
        with pytest.raises(StepError) as raised:
            self.base.go_to("sofa")
        assert raised.value.code is ErrorCode.PRECONDITION_FAILED
        assert what in raised.value.message
        assert (self.client.ready_polls, len(self.client.goals)) == (polls, goals)

    def assert_gate_free(self) -> None:
        """The next motion sends a goal, which Nav2 answers before any budget can expire."""
        self.token.reset()
        goals = len(self.client.goals)
        answered_at_once(self)
        assert bounded(self.base.go_to, "sofa") is not None
        assert len(self.client.goals) == goals + 1
        self.client.on_send = None


class SpyToken(CancelToken):
    """A real token with hooks around the Nth entry of its commit fence.

    A drive enters it twice: the send is entry 1, and the commit of a success is entry 2.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entries = 0
        self.before: dict[int, Callable[[], None]] = {}
        self.inside: dict[int, Callable[[], None]] = {}
        self.after: dict[int, Callable[[], None]] = {}
        self.on_wait: Callable[[], None] | None = None
        self.cancel_called = threading.Event()

    def cancel(self) -> None:
        self.cancel_called.set()
        super().cancel()

    @contextmanager
    def unless_cancelled(self, what: str) -> Iterator[None]:
        self.entries += 1
        entry = self.entries
        if entry in self.before:
            self.before[entry]()
        with super().unless_cancelled(what):
            if entry in self.inside:
                self.inside[entry]()
            yield
        if entry in self.after:
            self.after[entry]()

    def wait_for(self, *args, **kwargs) -> None:
        if self.on_wait is not None:
            self.on_wait()
        super().wait_for(*args, **kwargs)


def send_and_drive(rig: Rig):
    """Run go_to up to a live, accepted goal whose result request has gone out."""
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    handle = rig.accept()
    assert handle.result_asked.wait(WATCHDOG_S)
    return verb, handle


def answered_at_once(rig: Rig) -> None:
    """Have Nav2 accept and finish the next goal, while the send is still on the stack."""

    def answer():
        rig.client.accept().finish()

    rig.client.on_send = answer


# -- Places -----------------------------------------------------------------------


def test_places_are_local_and_immutable():
    source = dict(PLACES)
    rig = Rig(places=source)
    places = rig.base.places()
    assert dict(places) == PLACES

    source["garage"] = Location("garage", 9.0, 9.0, 0.0)
    assert "garage" not in rig.base.places()
    with pytest.raises(TypeError):
        places["garage"] = Location("garage", 9.0, 9.0, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        places["kitchen"].x = 9.0
    assert (rig.client.ready_polls, rig.client.goals, rig.connector.tf_reads) == (0, [], [])

    assert dict(Rig(places={}).base.places()) == {}

    rig.token.cancel()
    with pytest.raises(StepError) as raised:
        rig.base.places()
    assert raised.value.code is ErrorCode.CANCELLED


def test_named_goal_uses_map_clock_and_quaternion():
    rig = Rig()
    assert rig.client.node is rig.connector.node
    assert rig.client.action_type is NavigateToPose
    assert rig.client.action_name == "/navigate_to_pose"
    assert isinstance(rig.client.callback_group, MutuallyExclusiveCallbackGroup)

    rig.connector.robot = (1.1, 1.9, 0.5)
    rig.clock.nanoseconds = 5_000_000_123
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    goal = rig.client.goals[0].pose

    assert goal.header.frame_id == "map"
    assert (goal.header.stamp.sec, goal.header.stamp.nanosec) == (5, 123)
    assert (goal.pose.position.x, goal.pose.position.y) == (1.0, 2.0)
    assert goal.pose.orientation.z == pytest.approx(math.sin(math.pi / 4))
    assert goal.pose.orientation.w == pytest.approx(math.cos(math.pi / 4))

    handle = rig.accept()
    bounded(handle.finish)
    # The pose comes from TF, not from the place that was asked for.
    assert astuple(verb.result()) == pytest.approx((1.1, 1.9, 0.5))
    assert (handle.result_asks, handle.cancel_asks) == (1, 0)
    rig.assert_gate_free()


def test_place_lookup_is_case_insensitive_and_unknown_lists_names():
    rig = Rig()
    verb = Background(rig.base.go_to, "  KitCHen ")
    rig.client.wait_for_goals()
    assert rig.client.goals[0].pose.pose.position.x == 1.0
    bounded(rig.accept().finish)
    verb.result()

    polls, goals = rig.client.ready_polls, len(rig.client.goals)
    with pytest.raises(StepError) as raised:
        rig.base.go_to("garage")
    assert raised.value.code is ErrorCode.NOT_FOUND
    assert "kitchen" in raised.value.message and "sofa" in raised.value.message
    assert (rig.client.ready_polls, len(rig.client.goals)) == (polls, goals)

    with pytest.raises(StepError) as raised:
        Rig(places={}).base.go_to("kitchen")
    assert raised.value.code is ErrorCode.NOT_FOUND
    assert "none" in raised.value.message

    with pytest.raises(ValueError, match="same name"):
        Rig(places={"Sofa": PLACES["sofa"], " sofa": PLACES["sofa"]})


def test_missing_server_is_unavailable_and_bounded():
    rig = Rig(discovery_s=0.05)
    rig.client.ready = False
    with pytest.raises(StepError) as raised:
        rig.base.go_to("kitchen")
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert "Nav2" in raised.value.message
    assert rig.client.goals == []
    assert rig.client.ready_polls > 0

    # Discovery yields to a shorter overall timeout. Its own budget would outlast the watchdog.
    rig = Rig()
    rig.client.ready = False
    error = Background(rig.base.go_to, "kitchen", timeout_s=0.5).step_error()
    assert error.code is ErrorCode.UNAVAILABLE

    # A failure of ROS itself during discovery is the same answer, at once.
    def gone():
        raise _rclpy.InvalidHandle("destroyed")

    rig = Rig()
    rig.client.on_ready = gone
    assert Background(rig.base.go_to, "kitchen").step_error().code is ErrorCode.UNAVAILABLE


def test_spent_budget_is_timeout_and_not_refused():
    rig = Rig()
    rig.client.ready = False
    with pytest.raises(StepError) as raised:
        rig.base.go_to("kitchen", timeout_s=1e-9)
    assert raised.value.code is ErrorCode.TIMEOUT
    assert rig.client.ready_polls == 0
    with pytest.raises(StepError) as raised:
        RosBase._budget(time.monotonic() - 1.0, 2.0, "the move")
    assert raised.value.code is ErrorCode.TIMEOUT


@pytest.mark.parametrize(
    ("answer", "code", "text"),
    [
        (dict(status=GoalStatus.STATUS_ABORTED, error_code=103), ErrorCode.SKILL_FAILED, "ABORTED"),
        (
            dict(status=GoalStatus.STATUS_SUCCEEDED, error_code=7, error_msg="no path"),
            ErrorCode.SKILL_FAILED,
            "no path",
        ),
        (dict(status=GoalStatus.STATUS_CANCELED), ErrorCode.CANCELLED, "cancelled"),
    ],
)
def test_rejected_aborted_and_error_results_fail(answer, code, text):
    rig = Rig()
    verb, handle = send_and_drive(rig)
    bounded(handle.finish, **answer)
    error = verb.step_error()
    assert error.code is code
    assert text in error.message
    # Nav2 ended the goal, so the failure says so.
    assert isinstance(error, GoalFailed) is (code is ErrorCode.SKILL_FAILED)
    assert handle.cancel_asks == 0
    rig.assert_gate_free()

    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    handle = rig.accept(accepted=False)
    error = verb.step_error()
    assert isinstance(error, GoalFailed)
    assert error.code is ErrorCode.SKILL_FAILED
    assert "rejected" in error.message
    assert (handle.result_asks, handle.cancel_asks) == (0, 0)
    rig.assert_gate_free()


# -- TF ---------------------------------------------------------------------------


def test_relative_goal_uses_fresh_robot_frame_pose():
    rig = Rig()
    reads = iter(
        [
            # Stamped when the read began, which is not newer, and with an old heading.
            lambda: transform(5.0, 5.0, 0.0, stamp_ns=rig.clock.nanoseconds),
            lambda: transform(1.0, 2.0, math.pi / 2, stamp_ns=rig.clock.nanoseconds + 1),
        ]
    )
    rig.connector.tf = lambda: next(reads)()
    verb = Background(rig.base.move_relative, forward_m=1.0, left_m=0.5, turn_deg=90.0)
    rig.client.wait_for_goals()
    goal = rig.client.goals[0].pose
    # Forward is +y at this heading, and left is -x.
    assert goal.header.frame_id == "map"
    assert goal.pose.position.x == pytest.approx(0.5)
    assert goal.pose.position.y == pytest.approx(3.0)
    assert goal.pose.orientation.z == pytest.approx(1.0)
    assert goal.pose.orientation.w == pytest.approx(0.0, abs=1e-9)

    rig.connector.tf = rig.connector.fresh_sample
    rig.connector.robot = (0.5, 3.0, math.pi)
    bounded(rig.accept().finish)
    assert astuple(verb.result()) == pytest.approx((0.5, 3.0, math.pi))

    # Right after an arrival the buffer still holds the old heading, stamped before this
    # read began. The next move has to use the sample that follows it.
    rig.clock.nanoseconds += 1_000_000_000
    last = rig.clock.nanoseconds
    reads = iter(
        [
            lambda: transform(0.5, 3.0, math.pi, stamp_ns=last - 500_000_000),
            lambda: transform(4.0, 0.0, 0.0, stamp_ns=rig.clock.nanoseconds + 1),
        ]
    )
    rig.connector.tf = lambda: next(reads)()
    verb = Background(rig.base.move_relative, forward_m=1.0)
    rig.client.wait_for_goals(2)
    goal = rig.client.goals[1].pose
    assert (goal.pose.position.x, goal.pose.position.y) == pytest.approx((5.0, 0.0))
    rig.connector.tf = rig.connector.fresh_sample
    bounded(rig.accept().finish)
    verb.result()


def test_pose_is_fresh_and_finite():
    rig = Rig()
    rig.connector.robot = (1.0, -2.0, -0.75)
    assert astuple(rig.base.pose()) == pytest.approx((1.0, -2.0, -0.75))
    assert rig.connector.tf_reads[0] == ("map", "base_footprint", 0.01)

    def bad_sample(field: str, value: float, part: str = "translation"):
        stamped = transform(stamp_ns=rig.clock.nanoseconds + 1)
        setattr(getattr(stamped.transform, part), field, value)
        return stamped

    rig = Rig(tf_s=0.05)
    for field, value, part in (
        ("x", math.nan, "translation"),
        ("y", math.inf, "translation"),
        ("z", math.nan, "rotation"),
        ("w", 0.0, "rotation"),
    ):
        rig.connector.tf = lambda f=field, v=value, p=part: bad_sample(f, v, p)
        with pytest.raises(StepError) as raised:
            rig.base.pose()
        assert raised.value.code is ErrorCode.SKILL_FAILED

    rig.token.cancel()
    reads = len(rig.connector.tf_reads)
    with pytest.raises(StepError) as raised:
        rig.base.pose()
    assert raised.value.code is ErrorCode.CANCELLED
    assert len(rig.connector.tf_reads) == reads


def test_a_pose_read_waits_no_longer_than_its_timeout():
    rig = Rig()
    rig.connector.tf = missing_tf
    started = time.monotonic()
    with pytest.raises(StepError) as raised:
        rig.base.pose(timeout_s=0.05)
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert time.monotonic() - started < 1.0

    reads = len(rig.connector.tf_reads)
    for timeout_s in (0.0, -1.0, math.nan):
        with pytest.raises(StepError) as raised:
            rig.base.pose(timeout_s=timeout_s)
        assert raised.value.code is ErrorCode.REFUSED
    assert len(rig.connector.tf_reads) == reads


def test_with_tf_missing_a_pose_read_overruns_its_timeout_by_the_read_it_began():
    # RAI's get_transform sleeps 0.1 s before it gives up on a missing transform, and
    # a read that has begun runs to its end. The read count is the bound this checks;
    # the wall-clock limit only catches a hang.
    rai_miss_s = 0.1

    def slow_miss():
        time.sleep(rai_miss_s)
        return missing_tf()

    rig = Rig()
    rig.connector.tf = slow_miss
    started = time.monotonic()
    with pytest.raises(StepError) as raised:
        rig.base.pose(timeout_s=0.01)
    elapsed = time.monotonic() - started
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert len(rig.connector.tf_reads) == 1
    assert rai_miss_s <= elapsed < 1.0


def test_stale_missing_and_wrong_frame_tf_errors():
    rig = Rig(tf_s=0.05)

    # Only an old sample: the stream exists, but nothing new has arrived.
    rig.connector.tf = lambda: transform(stamp_ns=rig.clock.nanoseconds - 1)
    with pytest.raises(StepError) as raised:
        rig.base.pose()
    assert raised.value.code is ErrorCode.TIMEOUT
    assert "--use-sim-time" in raised.value.message

    # No transform and no publisher: the robot is off, or on another domain or router.
    rig.connector.tf = missing_tf
    with pytest.raises(StepError) as raised:
        rig.base.pose()
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert "ROS_DOMAIN_ID" in raised.value.message
    assert "Zenoh" in raised.value.message

    # No transform, but TF is running: the domain is fine, and the frames are the problem.
    rig.connector.node.graph_publishers["/tf"] = 1
    with pytest.raises(StepError) as raised:
        rig.base.pose()
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert "ROS_DOMAIN_ID" not in raised.value.message
    assert "localization" in raised.value.message

    # A sample of the wrong frames is not a pose of this robot.
    rig.connector.tf = lambda: transform(stamp_ns=rig.clock.nanoseconds + 1, child="base_link")
    with pytest.raises(StepError) as raised:
        rig.base.pose()
    assert raised.value.code is ErrorCode.SKILL_FAILED
    assert "base_link" in raised.value.message

    # A move with no usable start sends nothing.
    rig.connector.tf = lambda: transform(stamp_ns=rig.clock.nanoseconds - 1)
    with pytest.raises(StepError) as raised:
        rig.base.move_relative(forward_m=1.0)
    assert raised.value.code is ErrorCode.TIMEOUT
    assert rig.client.goals == []

    # A failure of ROS itself comes out as that, without waiting out the budget.
    def gone():
        raise _rclpy.RCLError("context is invalid")

    rig = Rig()
    rig.connector.tf = gone
    assert Background(rig.base.pose).step_error().code is ErrorCode.UNAVAILABLE


@pytest.mark.parametrize("error_ending", [ErrorCode.UNAVAILABLE, ErrorCode.TIMEOUT])
def test_success_without_fresh_tf_is_an_error(error_ending, caplog):
    rig = Rig(tf_s=0.05)
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    rig.connector.tf = (
        missing_tf
        if error_ending is ErrorCode.UNAVAILABLE
        else lambda: transform(stamp_ns=rig.clock.nanoseconds)
    )
    handle = rig.accept()
    with caplog.at_level("WARNING"):
        bounded(handle.finish)
        error = verb.step_error()
    assert error.code is error_ending
    assert "may have moved" in error.message
    assert "pose is unknown" in caplog.text
    rig.connector.tf = rig.connector.fresh_sample
    rig.assert_gate_free()


def test_paused_or_rewound_sim_clock_does_not_hang():
    # Paused, with a sample stamped exactly at the paused time: that sample is not newer.
    rig = Rig(tf_s=0.05)
    rig.connector.tf = lambda: transform(stamp_ns=rig.clock.nanoseconds)
    assert Background(rig.base.pose).step_error().code is ErrorCode.TIMEOUT

    # A clock that has not started: every real sample is far ahead of it.
    rig = Rig(tf_s=0.05)
    rig.clock.nanoseconds = 0
    rig.connector.tf = lambda: transform(stamp_ns=1_000_000_000_000)
    assert Background(rig.base.pose).step_error().code is ErrorCode.TIMEOUT

    # Rewound before this read began: the sample from before the rollback is not accepted.
    rig = Rig(tf_s=0.05)
    old_stamp = rig.clock.nanoseconds + 1
    rig.connector.tf = lambda: transform(stamp_ns=old_stamp)
    rig.clock.nanoseconds = 1_000_000_000
    assert Background(rig.base.pose).step_error().code is ErrorCode.TIMEOUT

    # Rewound during this read: an explicit failure, with no wait for the budget.
    rig = Rig()
    rig.connector.tf = lambda: transform(stamp_ns=rig.clock.nanoseconds + 1)
    readings = []

    def rewind_after_the_first_reading():
        readings.append(1)
        if len(readings) == 2:
            rig.clock.nanoseconds -= 1_000_000_000

    rig.clock.on_now = rewind_after_the_first_reading
    error = Background(rig.base.pose).step_error()
    assert error.code is ErrorCode.TIMEOUT
    assert "backwards" in error.message


def test_stop_during_tf_read_is_cancelled():
    rig = Rig()
    reading = threading.Event()

    def miss_like_rai():
        # RAI's helper sleeps 0.1 s on a miss, which is what a stop has to outlast here.
        reading.set()
        time.sleep(0.1)
        return missing_tf()

    rig.connector.tf = miss_like_rai
    verb = Background(rig.base.move_relative, forward_m=1.0)
    assert reading.wait(WATCHDOG_S)
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert rig.client.goals == []


# -- Guards and policy -------------------------------------------------------------


@pytest.mark.parametrize(
    "move",
    [
        dict(forward_m=math.nan),
        dict(left_m=math.inf),
        dict(turn_deg=-math.inf),
        dict(forward_m=True),
        dict(left_m="1"),
        dict(turn_deg=None),
        dict(forward_m=10**400),
        dict(forward_m=1e308, left_m=1e308),
        dict(),
        dict(forward_m=0.0, left_m=0.0, turn_deg=0.0),
        dict(forward_m=3.5),
        dict(forward_m=2.5, left_m=2.5),
        dict(forward_m=1.0, timeout_s=0),
        dict(forward_m=1.0, timeout_s=math.nan),
        dict(forward_m=1.0, timeout_s=True),
    ],
)
def test_guards_refuse_without_sending(move):
    rig = Rig()
    with pytest.raises(StepError) as raised:
        rig.base.move_relative(**move)
    assert raised.value.code is ErrorCode.REFUSED
    assert (rig.client.ready_polls, rig.connector.tf_reads, rig.client.goals) == (0, [], [])

    for call in (
        lambda: rig.base.go_to(5),
        lambda: rig.base.go_to(None),
        lambda: rig.base.go_to("kitchen", timeout_s=-1.0),
    ):
        with pytest.raises(StepError) as raised:
            call()
        assert raised.value.code is ErrorCode.REFUSED
    assert (rig.client.ready_polls, rig.connector.tf_reads, rig.client.goals) == (0, [], [])

    # A place with bad coordinates never reaches Nav2, whatever its file said.
    rig = Rig(places={"nowhere": Location("nowhere", math.nan, 0.0, 0.0)})
    with pytest.raises(StepError) as raised:
        rig.base.go_to("nowhere")
    assert raised.value.code is ErrorCode.REFUSED
    assert rig.client.goals == []


@pytest.mark.parametrize(
    ("name", "node"),
    [
        ("/cmd_vel", FakeNode()),
        ("cmd_vel", FakeNode()),
        ("/walkie/reset", FakeNode()),
        ("walkie/task", FakeNode()),
        ("/walkie/todos", FakeNode()),
        ("walkie/verdict", FakeNode()),
        ("/nav_alias", FakeNode(remaps={"/nav_alias": "/cmd_vel"})),
        ("cmd_vel_nav", FakeNode()),
        ("/go_to_pose", FakeNode()),
        ("lift/cmd", FakeNode()),
        (
            "/arm_alias",
            FakeNode(remaps={"/arm_alias": "/lift_controller/follow_joint_trajectory"}),
        ),
    ],
)
def test_forbidden_action_is_refused(name, node):
    clients = []
    with pytest.raises(StepError) as raised:
        RosBase(
            FakeConnector(node),
            PLACES,
            CancelToken(),
            action_name=name,
            client_factory=lambda *a, **k: clients.append(a),
        )
    assert raised.value.code is ErrorCode.REFUSED
    assert clients == []


def test_forbidden_names_cover_velocity_pose_goal_and_lift_names():
    assert set(FORBIDDEN_NAMES) == {
        "/cmd_vel",
        "/cmd_vel_nav",
        "/cmd_vel_joy",
        "/cmd_vel_key",
        "/omni_wheel_drive_controller/cmd_vel",
        "/go_to_home",
        "/go_to_pose",
        "/go_to_pose_relative",
        "/go_to_pose_quat",
        "/lift_controller/follow_joint_trajectory",
        "/lift_controller/commands",
        "/lift/cmd",
        "/walkie/reset",
        "/walkie/task",
        "/walkie/todos",
        "/walkie/verdict",
    }
    assert len(FORBIDDEN_NAMES) == len(set(FORBIDDEN_NAMES))
    # CommanderClient publishes its stop there, as the one publisher while no
    # MotionSupervisor exists.
    assert STOP_TOPIC not in FORBIDDEN_NAMES


@pytest.mark.parametrize(
    "node",
    [FakeNode(), FakeNode(remaps={"/to_operator": "/walkie/verdict"})],
)
def test_no_capability_may_publish_the_operators_verdict(node):
    # Speech is the one capability that publishes. A verdict verifies the scripts'
    # functions, so the agent must never send one itself.
    topic = "/to_operator" if node.remaps else "/walkie/verdict"
    with pytest.raises(StepError) as raised:
        RosSpeech(FakeConnector(node), topic=topic)
    assert raised.value.code is ErrorCode.REFUSED
    assert node.publishers == []


# -- Arm check ----------------------------------------------------------------------


class ArmsCheck:
    """Stands in for RosArms.check_for_driving: notes each call, runs `during`, then raises `refusal`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []
        self.during: Callable[[str, float], None] | None = None
        self.refusal: StepError | None = None

    def __call__(self, what: str, budget_s: float) -> None:
        self.calls.append((what, budget_s))
        if self.during is not None:
            self.during(what, budget_s)
        if self.refusal is not None:
            raise self.refusal


DRIVES = {
    "go_to": (lambda base, **kw: base.go_to("kitchen", **kw), "the drive to kitchen"),
    "move_relative": (lambda base, **kw: base.move_relative(forward_m=1.0, **kw), "the move"),
}


@pytest.mark.parametrize("verb", DRIVES)
def test_every_drive_checks_the_arms_before_discovery_tf_and_the_send(verb):
    drive, what = DRIVES[verb]
    check = ArmsCheck()
    rig = Rig(arms_check=check)
    seen = []

    def note_what_the_base_had_done(_what, _budget_s):
        # unresolved() takes the base's lock, so a check run under it would deadlock here.
        seen.append(
            (
                rig.client.ready_polls,
                len(rig.connector.tf_reads),
                len(rig.client.goals),
                rig.base.unresolved(),
            )
        )

    check.during = note_what_the_base_had_done
    answered_at_once(rig)
    assert bounded(drive, rig.base, timeout_s=5.0) is not None
    assert seen == [(0, 0, 0, [])]
    [(checked, budget_s)] = check.calls
    assert checked == what
    # The time left to the verb, and not a sub-budget of the base's own.
    assert 4.0 < budget_s <= 5.0
    assert len(rig.client.goals) == 1


@pytest.mark.parametrize(
    "code", [ErrorCode.PRECONDITION_FAILED, ErrorCode.INTERNAL, ErrorCode.TIMEOUT]
)
def test_a_refused_drive_reads_no_tf_and_sends_nothing(code):
    check = ArmsCheck()
    check.refusal = StepError(code, "Not starting the drive: the left arm is out.")
    rig = Rig(arms_check=check)
    for drive, _what in DRIVES.values():
        with pytest.raises(StepError) as raised:
            drive(rig.base)
        assert raised.value is check.refusal
    assert len(check.calls) == len(DRIVES)
    assert (rig.client.ready_polls, rig.connector.tf_reads, rig.client.goals) == (0, [], [])

    # A refusal leaves no record, so the next drive goes once the arms allow it.
    check.refusal = None
    rig.assert_gate_free()


HOME = ArmState(pose="home", travel_safe=True)
HANDS_UP = ArmState(pose="hands_up", travel_safe=False)


@pytest.mark.parametrize(
    ("found", "code", "text"),
    [
        (
            DriveCheck(ArmsState(left=HANDS_UP, right=HOME)),
            ErrorCode.PRECONDITION_FAILED,
            f"the left arm is {ARMS_OUT}",
        ),
        (DriveCheck(ArmsState(left=HOME)), ErrorCode.INTERNAL, "cannot be checked"),
        (DriveCheck(ArmsState(left=HOME, right=HOME)), None, ""),
        (DriveCheck(None, NOT_FOUND), None, ""),
    ],
    ids=["arm_out", "arm_unknown", "both_home", "commander_not_found"],
)
def test_the_drive_verdict_judges_a_turn(found, code, text):
    rig = Rig(arms_check=lambda what, _budget_s: drive_verdict(lambda: found, what))
    if code is None:
        answered_at_once(rig)
        assert bounded(rig.base.move_relative, turn_deg=90.0) is not None
        assert len(rig.client.goals) == 1
        return
    with pytest.raises(StepError) as raised:
        rig.base.move_relative(turn_deg=90.0)
    assert raised.value.code is code
    assert raised.value.message.startswith("Not starting the move")
    assert text in raised.value.message
    # face_person's turn takes a GoalFailed as a turn that ended, so a refusal is never one.
    assert not isinstance(raised.value, GoalFailed)
    assert (rig.client.ready_polls, rig.connector.tf_reads, rig.client.goals) == (0, [], [])


def test_the_turn_atomics_answer_the_arm_refusal_while_tf_is_missing(caplog):
    # Both atomics read the start pose before they turn. With TF missing that read fails,
    # and the turn must still reach the arm check: its refusal says to move the arm home,
    # where the failed read would only say that the position is unknown.
    reads_at_the_check = []

    def check_like_the_commander(what, _budget_s):
        reads_at_the_check.append(len(rig.connector.tf_reads))
        drive_verdict(lambda: DriveCheck(ArmsState(left=HANDS_UP, right=HOME)), what)

    class OnePerson:
        def people(self, query="", *, timeout_s=5.0):
            return [PersonSeen("person_1", None, "standing person", "", "living room", 90.0)]

    rig = Rig(arms_check=check_like_the_commander, tf_s=0.05)
    rig.connector.tf = missing_tf
    ctx = Ctx(cancel=rig.token, base=rig.base, speech=None, vision=OnePerson())
    turns = {
        "the turn of 180 degrees": lambda: move_relative(ctx, 0.0, 0.0, 180.0),
        "the turn to person_1": lambda: face_person(ctx, "person_1"),
    }
    for what, turn in turns.items():
        reads = len(rig.connector.tf_reads)
        with (
            caplog.at_level("INFO", logger="walkie_agent.tools.atomic.navigation"),
            pytest.raises(StepError) as raised,
        ):
            bounded(turn)
        assert raised.value.code is ErrorCode.PRECONDITION_FAILED
        assert raised.value.message.startswith("Not starting the move")
        assert f"the left arm is {ARMS_OUT}" in raised.value.message
        assert f"Starting {what} without its start pose: UNAVAILABLE" in caplog.text
        # The atomic's own start read made every TF read, all before the check, and the
        # refused turn read none of its own.
        assert reads_at_the_check[-1] > reads
        assert len(rig.connector.tf_reads) == reads_at_the_check[-1]
    assert len(reads_at_the_check) == len(turns)
    assert (rig.client.ready_polls, rig.client.goals) == (0, [])
    assert rig.base.unresolved() == []


def test_no_arm_check_while_another_goal_is_live():
    check = ArmsCheck()
    rig = Rig(arms_check=check)
    verb, handle = send_and_drive(rig)
    assert len(check.calls) == 1
    for drive, _what in DRIVES.values():
        with pytest.raises(StepError) as raised:
            drive(rig.base)
        assert raised.value.code is ErrorCode.PRECONDITION_FAILED
    assert len(check.calls) == 1
    assert len(rig.client.goals) == 1
    bounded(handle.finish)
    verb.result()


def test_stop_during_the_arm_check_is_cancelled_and_sends_nothing():
    check = ArmsCheck()
    rig = Rig(arms_check=check)
    checking = threading.Event()

    def wait_like_the_commander(what, budget_s):
        checking.set()
        rig.token.wait_for(lambda: False, budget_s, f"checking the arms for {what}")

    check.during = wait_like_the_commander
    verb = Background(rig.base.move_relative, forward_m=1.0)
    assert checking.wait(WATCHDOG_S)
    # A check run inside the fence would hold this stop up until the watchdog.
    rig.stop()
    error = verb.step_error()
    assert error.code is ErrorCode.CANCELLED
    assert "checking the arms" in error.message
    assert (rig.client.ready_polls, rig.connector.tf_reads, rig.client.goals) == (0, [], [])
    assert rig.base.unresolved() == []

    check.during = None
    rig.assert_gate_free()


def test_a_spent_budget_times_out_before_the_arm_check():
    check = ArmsCheck()
    rig = Rig(arms_check=check)
    with pytest.raises(StepError) as raised:
        rig.base.go_to("kitchen", timeout_s=1e-9)
    assert raised.value.code is ErrorCode.TIMEOUT
    assert check.calls == []


def test_without_an_arms_check_the_base_drives_unchecked():
    # main builds the base with no check unless it runs with arms.
    assert inspect.signature(RosBase).parameters["arms_check"].default is None
    rig = Rig()
    answered_at_once(rig)
    assert bounded(rig.base.move_relative, turn_deg=90.0) is not None
    assert len(rig.client.goals) == 1


# -- Stop --------------------------------------------------------------------------


def test_stop_never_raises():
    class BrokenToken(CancelToken):
        def cancel(self) -> None:
            raise RuntimeError("broken")

    # With no goal, and twice: a no-op apart from the latch.
    rig = Rig()
    rig.stop()
    rig.stop()
    assert rig.token.cancelled
    assert rig.client.goals == []
    Rig(token=BrokenToken()).stop()

    # A cancel that cannot be sent is not a failure of stop. The next stop sends it.
    rig = Rig()
    verb, handle = send_and_drive(rig)
    handle.cancel_error = _rclpy.RCLError("context is invalid")
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED
    tries = handle.cancel_asks
    assert tries >= 1
    handle.cancel_error = None
    rig.stop()
    assert handle.cancel_asks == tries + 1
    rig.stop()
    assert handle.cancel_asks == tries + 1


def test_stop_before_send_sends_nothing():
    rig = Rig()
    rig.stop()
    for place in ("kitchen", "garage"):
        with pytest.raises(StepError) as raised:
            rig.base.go_to(place)
        assert raised.value.code is ErrorCode.CANCELLED
    with pytest.raises(StepError) as raised:
        rig.base.move_relative(forward_m=1.0)
    assert raised.value.code is ErrorCode.CANCELLED
    assert (rig.client.ready_polls, rig.connector.tf_reads, rig.client.goals) == (0, [], [])


def test_stop_during_server_wait_sends_nothing():
    rig = Rig()
    rig.client.ready = False
    waiting = threading.Event()
    rig.client.on_ready = waiting.set
    verb = Background(rig.base.go_to, "kitchen")
    assert waiting.wait(WATCHDOG_S)
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert rig.client.goals == []
    rig.client.ready = True
    rig.assert_gate_free()


def test_stop_between_ready_and_send_sends_nothing():
    rig = Rig()
    # Discovery has finished, and the goal's stamp is the one thing read before the send.
    rig.clock.on_now = rig.base.stop
    verb = Background(rig.base.go_to, "kitchen")
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert rig.client.ready_polls == 1
    assert rig.client.goals == []
    rig.clock.on_now = None
    rig.assert_gate_free()


def test_stop_during_send_waits_for_the_stored_future():
    rig = Rig()
    stop_landed = []

    def stop_while_sending():
        thread = threading.Thread(target=rig.base.stop, daemon=True)
        thread.start()
        # A stop cannot finish while the fence holds the send, so it cannot miss the future.
        thread.join(0.05)
        stop_landed.append(thread.is_alive())

    rig.client.on_send = stop_while_sending
    verb = Background(rig.base.go_to, "kitchen")
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert stop_landed == [True]
    rig.client.on_send = None

    handle = rig.accept()
    assert handle.cancel_asks == 1
    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    rig.assert_gate_free()


def test_stop_during_acceptance_cancels_late_handle():
    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED

    # The verb has returned, and the goal it may still have started is not forgotten.
    rig.assert_gate_held("the drive to kitchen")
    assert not rig.client.send_futures[0].cancelled()

    handle = rig.accept()
    assert rig.client.handles == [handle]
    assert (handle.result_asks, handle.cancel_asks) == (1, 1)
    rig.assert_gate_held()

    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    assert handle.cancel_asks == 1
    rig.assert_gate_free()


def test_timeout_during_acceptance_cancels_late_handle():
    rig = Rig(acceptance_s=0.05)
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    assert verb.step_error().code is ErrorCode.TIMEOUT
    assert not rig.token.cancelled
    rig.assert_gate_held()

    handle = rig.accept()
    assert (handle.result_asks, handle.cancel_asks) == (1, 1)
    rig.assert_gate_held()
    # A late success changes nothing about the call that already timed out.
    bounded(handle.finish)
    assert verb.step_error().code is ErrorCode.TIMEOUT
    rig.assert_gate_free()


def test_late_acceptance_after_token_reset_still_cancels():
    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED

    rig.token.reset()
    assert not rig.token.cancelled
    handle = rig.accept()
    assert (handle.result_asks, handle.cancel_asks) == (1, 1)
    rig.assert_gate_held()
    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    rig.assert_gate_free()

    # A late rejection also ends the record, with nothing to cancel.
    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED
    rig.token.reset()
    handle = rig.accept(accepted=False)
    assert (handle.result_asks, handle.cancel_asks) == (0, 0)
    rig.assert_gate_free()


def test_stop_during_drive_cancels_owned_goal():
    rig = Rig()
    verb, handle = send_and_drive(rig)
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert rig.client.handles == [handle]
    assert handle.cancel_asks == 1

    # Stopping again asks nothing more of the same goal.
    rig.stop()
    assert handle.cancel_asks == 1
    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    rig.assert_gate_free()


def test_token_alone_cancels_the_goal_through_the_verb():
    # The token alone, with no call to base.stop, still cancels the goal through the verb.
    rig = Rig()
    verb, handle = send_and_drive(rig)
    bounded(rig.token.cancel)
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert handle.cancel_asks == 1
    rig.assert_gate_held()
    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    rig.assert_gate_free()

    # A goal accepted after the token is set is cancelled even before the verb reacts.
    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    bounded(rig.token.cancel)
    handle = rig.accept()
    assert handle.cancel_asked.wait(WATCHDOG_S)
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert handle.cancel_asks == 1


def test_acceptance_with_the_token_set_cancels_at_once():
    token = SpyToken()
    rig = Rig(token=token)
    rig.client.on_send = rig.client.accept
    # The send has left the fence and its callback is not attached, so the token alone says stop.
    token.after[1] = token.cancel
    asks_when_the_verb_first_waits = []

    def note_the_cancels():
        if rig.client.handles and not asks_when_the_verb_first_waits:
            asks_when_the_verb_first_waits.append(rig.client.handles[0].cancel_asks)

    token.on_wait = note_the_cancels
    assert Background(rig.base.go_to, "kitchen").step_error().code is ErrorCode.CANCELLED
    assert asks_when_the_verb_first_waits == [1]
    assert rig.client.handles[0].cancel_asks == 1


@pytest.mark.parametrize(
    ("the_wait", "accepted_at_send"),
    [(2, False), (3, True)],
    ids=["during_acceptance", "during_the_drive"],
)
def test_stop_then_reset_still_ends_the_waiting_verb(the_wait, accepted_at_send):
    # A reset can come before the verb's next poll, so the verb may never see the token set.
    # This token never shows as set, which leaves only the mark on the record to end the wait.
    token = SpyToken()
    token.cancel = lambda: None
    rig = Rig(token=token)
    if accepted_at_send:
        rig.client.on_send = rig.client.accept

    # The waits of a drive come in order: discovery, acceptance, and then the result.
    waits = []
    reached = threading.Event()

    def count_the_waits():
        waits.append(1)
        if len(waits) == the_wait:
            reached.set()

    token.on_wait = count_the_waits
    verb = Background(rig.base.go_to, "kitchen")
    assert reached.wait(WATCHDOG_S)
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED

    handle = rig.client.handles[0] if accepted_at_send else rig.accept()
    assert handle.cancel_asks == 1
    rig.assert_gate_held()
    bounded(handle.finish)
    assert verb.step_error().code is ErrorCode.CANCELLED
    rig.assert_gate_free()


def test_result_timeout_cancels_owned_goal():
    rig = Rig()
    rig.client.on_send = rig.client.accept
    verb = Background(rig.base.go_to, "kitchen", timeout_s=0.1)
    error = verb.step_error()
    assert error.code is ErrorCode.TIMEOUT
    handle = rig.client.handles[0]
    assert (handle.result_asks, handle.cancel_asks) == (1, 1)
    assert not rig.token.cancelled

    # The record lives on until the end comes, and a late success changes nothing.
    rig.client.on_send = None
    rig.assert_gate_held()
    bounded(handle.finish)
    assert handle.cancel_asks == 1
    rig.assert_gate_free()


@pytest.mark.parametrize(
    "scenario",
    [
        "cancel_before_the_fence",
        "stop_and_reset_before_the_fence",
        "cancel_during_the_final_read",
        "cancel_while_the_fence_is_held",
    ],
)
def test_success_cancel_race_obeys_commit_order(scenario):
    token = SpyToken()
    rig = Rig(token=token)
    rig.connector.robot = (1.0, 2.0, 0.0)
    answered_at_once(rig)
    late_cancels = []
    cancel_waited = []

    if scenario == "cancel_before_the_fence":
        token.before[2] = token.cancel
    elif scenario == "stop_and_reset_before_the_fence":

        def stop_and_reset():
            rig.base.stop()
            token.reset()

        token.before[2] = stop_and_reset
    elif scenario == "cancel_during_the_final_read":

        def cancel_then_read():
            token.cancel()
            return rig.connector.fresh_sample()

        rig.connector.tf = cancel_then_read
    else:

        def cancel_from_another_thread():
            thread = threading.Thread(target=token.cancel, daemon=True)
            thread.start()
            # Only a commit that already holds the fence may win against this cancel.
            thread.join(0.05)
            cancel_waited.append(thread.is_alive())
            late_cancels.append(thread)

        token.inside[2] = cancel_from_another_thread

    verb = Background(rig.base.go_to, "kitchen")
    if scenario == "cancel_while_the_fence_is_held":
        assert astuple(verb.result()) == pytest.approx((1.0, 2.0, 0.0))
        assert cancel_waited == [True]
        late_cancels[0].join(WATCHDOG_S)
        assert token.cancelled
    else:
        assert verb.step_error().code is ErrorCode.CANCELLED
    # Nav2 had already finished, so there was nothing left to cancel.
    assert rig.client.handles[0].cancel_asks == 0
    rig.connector.tf = rig.connector.fresh_sample
    rig.assert_gate_free()


def test_success_after_a_cancel_with_no_goals_left_ends_as_cancelled():
    rig = Rig()
    verb, handle = send_and_drive(rig)
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED

    # Nav2 finished first, so the cancel finds no goal. The result still ends the record.
    bounded(handle.acknowledge, CancelGoal.Response.ERROR_GOAL_TERMINATED)
    rig.assert_gate_held()
    bounded(handle.finish)
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert handle.cancel_asks == 1
    rig.assert_gate_free()


def test_cancel_ack_is_not_terminal():
    rig = Rig()
    verb, handle = send_and_drive(rig)
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED

    bounded(handle.acknowledge)
    rig.assert_gate_held()
    assert rig.base.close(timeout_s=1e-9) is False
    assert rig.client.destroyed == 0
    assert "cancel was requested" in rig.base.unresolved()[0]

    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    assert rig.base.unresolved() == []
    assert bounded(rig.base.close) is True
    assert rig.client.destroyed == 1


def test_refused_cancel_is_sent_again_by_the_next_stop():
    rig = Rig()
    verb, handle = send_and_drive(rig)
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED
    bounded(handle.acknowledge, CancelGoal.Response.ERROR_REJECTED)
    rig.stop()
    assert handle.cancel_asks == 2
    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    rig.assert_gate_free()


def test_second_motion_during_a_live_goal_is_refused():
    rig = Rig()
    verb, handle = send_and_drive(rig)
    with pytest.raises(StepError) as raised:
        rig.base.go_to("sofa")
    assert raised.value.code is ErrorCode.PRECONDITION_FAILED
    with pytest.raises(StepError) as raised:
        rig.base.move_relative(forward_m=1.0)
    assert raised.value.code is ErrorCode.PRECONDITION_FAILED
    assert len(rig.client.goals) == 1
    assert handle.cancel_asks == 0

    # A result does not free the gate until the verb has its pose.
    release = threading.Event()
    reading = threading.Event()

    def hold_the_final_read():
        reading.set()
        assert release.wait(WATCHDOG_S)
        return rig.connector.fresh_sample()

    rig.connector.tf = hold_the_final_read
    bounded(handle.finish)
    assert reading.wait(WATCHDOG_S)
    rig.assert_gate_held()
    release.set()
    rig.connector.tf = rig.connector.fresh_sample
    verb.result()
    rig.assert_gate_free()


@pytest.mark.parametrize(
    "ending",
    [
        "acceptance_never_arrives",
        "acceptance_future_fails",
        "cancel_never_ends",
        "result_status_unknown",
        "result_future_fails",
    ],
)
def test_unsettled_old_goal_blocks_new_motion(ending):
    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()

    if ending == "acceptance_never_arrives":
        rig.stop()
        assert verb.step_error().code is ErrorCode.CANCELLED
    elif ending == "acceptance_future_fails":
        bounded(rig.client.send_futures[0].set_exception, RuntimeError("lost"))
        error = verb.step_error()
        assert error.code is ErrorCode.SKILL_FAILED
        assert not isinstance(error, GoalFailed)
    else:
        handle = rig.accept()
        assert handle.result_asked.wait(WATCHDOG_S)
        if ending == "cancel_never_ends":
            rig.stop()
            assert verb.step_error().code is ErrorCode.CANCELLED
            bounded(handle.acknowledge)
        else:
            if ending == "result_status_unknown":
                bounded(handle.finish, status=GoalStatus.STATUS_UNKNOWN)
            else:
                bounded(handle.result_future.set_exception, RuntimeError("lost"))
            # The goal's state is unknown, so the verb fails and cancels it for safety.
            error = verb.step_error()
            assert error.code is ErrorCode.SKILL_FAILED
            assert not isinstance(error, GoalFailed)
            assert handle.cancel_asks == 1

    rig.assert_gate_held("the drive to kitchen")
    assert rig.base.close(timeout_s=1e-9) is False
    assert rig.client.destroyed == 0
    assert rig.base.unresolved()[0].startswith("the drive to kitchen")


def test_failed_result_request_is_asked_again():
    rig = Rig()
    verb, handle = send_and_drive(rig)
    bounded(handle.finish, status=GoalStatus.STATUS_UNKNOWN)
    assert verb.step_error().code is ErrorCode.SKILL_FAILED
    rig.assert_gate_held()

    handle.result_future = Future()
    rig.stop()
    assert handle.result_asks == 3  # The verb's own abandon asked once, and this stop once more.
    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    rig.assert_gate_free()

    # A request that raises is the same case, and it does not reach the executor.
    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED

    handle = FakeGoalHandle(rig.client.log)
    handle.result_error = _rclpy.RCLError("context is invalid")
    bounded(rig.client.send_futures[0].set_result, handle)
    assert (handle.result_asks, handle.cancel_asks) == (1, 1)
    rig.assert_gate_held()
    handle.result_error = None
    rig.stop()
    assert handle.result_asks == 2
    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    rig.assert_gate_free()


# -- Threads ----------------------------------------------------------------------


def test_callbacks_can_complete_inline_without_deadlock():
    # Everything finishes while the send is on the stack, so the verb's own attach runs it all.
    rig = Rig()
    rig.connector.robot = (1.0, 2.0, 0.0)
    answered_at_once(rig)
    assert astuple(bounded(rig.base.go_to, "kitchen")) == pytest.approx((1.0, 2.0, 0.0))
    assert len(rig.client.handles) == 1
    rig.assert_gate_free()

    def answer_inline(handle, cancel_future):
        # Runs inside cancel_goal_async. A lock held across the dispatch deadlocks here.
        assert rig.base.unresolved()
        cancel_future.set_result(CancelGoal.Response(return_code=0))
        handle.finish(status=GoalStatus.STATUS_CANCELED)

    # A late acceptance: its callback dispatches a cancel, which ends the goal inline.
    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED
    handle = rig.accept(on_cancel=answer_inline)
    assert handle.cancel_asks == 1
    rig.assert_gate_free()

    # A stop on a live goal: the same, on the stopping thread.
    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    handle = rig.accept(on_cancel=answer_inline)
    assert handle.result_asked.wait(WATCHDOG_S)
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert handle.cancel_asks == 1
    rig.assert_gate_free()

    # The send holds the fence but not the record's lock, so a callback can still run.
    rig = Rig()
    seen = []

    def read_the_record():
        seen.append(rig.base.unresolved())
        rig.client.accept().finish()

    rig.client.on_send = read_the_record
    assert bounded(rig.base.go_to, "kitchen") is not None
    assert seen and seen[0]


def test_callbacks_do_not_need_the_commit_fence():
    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()

    entered = threading.Event()
    release = threading.Event()

    def hold_the_fence():
        with rig.token.unless_cancelled("the test"):
            entered.set()
            assert release.wait(WATCHDOG_S)

    holder = Background(hold_the_fence)
    assert entered.wait(WATCHDOG_S)

    # With a stop or a commit holding the fence, Nav2's answers still land.
    handle = rig.accept()
    assert handle.result_asked.wait(WATCHDOG_S)
    bounded(handle.finish)
    assert not verb.done()

    release.set()
    holder.result()
    assert astuple(verb.result()) == pytest.approx((0.0, 0.0, 0.0))


# -- Close --------------------------------------------------------------------------


def test_close_drains_before_destroy():
    # With nothing running, close is immediate and repeatable.
    rig = Rig()
    assert bounded(rig.base.close) is True
    assert bounded(rig.base.close) is True
    assert rig.client.destroyed == 1

    # A goal that is still awaiting acceptance, from a verb that has already returned.
    rig = Rig()
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()
    rig.stop()
    assert verb.step_error().code is ErrorCode.CANCELLED
    rig.token.reset()

    assert bounded(rig.base.close, timeout_s=1e-9) is False
    assert rig.client.destroyed == 0
    assert "has not answered" in rig.base.unresolved()[0]

    # Late acceptance during the close: cancelled, though the token was reset.
    rig.token.reset()
    handle = rig.accept()
    assert handle.cancel_asks == 1
    assert bounded(rig.base.close, timeout_s=1e-9) is False
    assert rig.client.destroyed == 0
    assert handle.cancel_asks == 1

    # A close that is already waiting completes when the result comes, and only then destroys.
    closing = Background(rig.base.close, timeout_s=WATCHDOG_S)
    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    assert closing.result() is True
    assert rig.client.log == ["send", "result asked", "cancel", "terminal", "destroy"]
    assert rig.base.unresolved() == []

    # After a close, motion and the pose are unavailable, and places stay local.
    assert bounded(rig.base.close) is True
    assert rig.client.destroyed == 1
    rig.token.reset()
    for call in (
        lambda: rig.base.go_to("kitchen"),
        lambda: rig.base.move_relative(forward_m=1.0),
        rig.base.pose,
    ):
        with pytest.raises(StepError) as raised:
            call()
        assert raised.value.code is ErrorCode.UNAVAILABLE
    assert "kitchen" in rig.base.places()
    assert len(rig.client.goals) == 1


def test_close_stops_a_running_verb_and_waits_for_it():
    rig = Rig()
    verb, handle = send_and_drive(rig)
    closing = Background(rig.base.close, timeout_s=WATCHDOG_S)
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert handle.cancel_asked.wait(WATCHDOG_S)
    assert not closing.done()
    assert rig.client.destroyed == 0
    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    assert closing.result() is True
    assert rig.client.destroyed == 1


def test_acceptance_during_a_close_cancels_at_once():
    token = SpyToken()
    rig = Rig(token=token)
    verb = Background(rig.base.go_to, "kitchen")
    rig.client.wait_for_goals()

    entered = threading.Event()
    release = threading.Event()

    def hold_the_fence():
        with super(SpyToken, token).unless_cancelled("the test"):
            entered.set()
            assert release.wait(WATCHDOG_S)

    holder = Background(hold_the_fence)
    assert entered.wait(WATCHDOG_S)

    # The close has begun and is stuck on the fence, so neither the token nor the record
    # says stop yet. The goal accepted now gets its cancel from the closing flag alone.
    closing = Background(rig.base.close, timeout_s=WATCHDOG_S)
    assert token.cancel_called.wait(WATCHDOG_S)
    assert not token.cancelled
    handle = rig.accept()
    assert handle.cancel_asked.wait(WATCHDOG_S)

    release.set()
    holder.result()
    assert verb.step_error().code is ErrorCode.CANCELLED
    assert handle.cancel_asks == 1
    bounded(handle.finish, status=GoalStatus.STATUS_CANCELED)
    assert closing.result() is True


def test_failed_send_ends_the_record():
    def gone():
        raise _rclpy.InvalidHandle("destroyed")

    rig = Rig()
    rig.client.on_send = gone
    assert Background(rig.base.go_to, "kitchen").step_error().code is ErrorCode.UNAVAILABLE
    assert rig.base.unresolved() == []
    rig.client.on_send = None
    rig.assert_gate_free()

    def bug():
        raise ValueError("a bug")

    rig = Rig()
    rig.client.on_send = bug
    assert Background(rig.base.go_to, "kitchen").step_error().code is ErrorCode.INTERNAL
    assert rig.base.unresolved() == []


# -- Vision -------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["objects", "people"])
def test_unavailable_vision_validates_then_raises(verb):
    token = CancelToken()
    call = getattr(UnavailableVision(token), verb)
    with pytest.raises(StepError) as raised:
        call()
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert "perception is not connected" in raised.value.message

    for args, kwargs in (((5,), {}), (("cup",), {"timeout_s": 0})):
        with pytest.raises(StepError) as raised:
            call(*args, **kwargs)
        assert raised.value.code is ErrorCode.REFUSED

    token.cancel()
    with pytest.raises(StepError) as raised:
        call()
    assert raised.value.code is ErrorCode.CANCELLED
