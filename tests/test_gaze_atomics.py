"""The gaze atomics, look_at and stop_looking, end to end over ROS (ADR-0051).

The atomics run through a real Toolbox, with a real VisionFeed and GazeLoop
on the `connector` fixture's node, on Fast DDS and a random domain
(ADR-0019), so the /cmd_vel the loop publishes reaches no robot. The fake
walkie-vision node of test_people_atomics publishes the people and objects
arrays. The base is either fixed, as static transforms, or test_gaze's fake
base, which turns at the last /cmd_vel command the recorder hears; the
recorder also stands in for Nav2's goal status. The arm tests run against
FakeCommander through a real CommanderClient, and the drive tests against
FakeWalkie's navigate_to_pose.
"""

import math
import re
import threading
import time
from collections.abc import Callable

import pytest
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import TwistStamped
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Empty, Header

from test_arm_atomics import MODEL_PATH, POSES_PATH, targets
from test_gaze import (
    DEADBAND,
    FakeBase,
    Recorder,
    assert_rotation_only,
    assert_silent_after_stop,
    mount_camera,
    place_base,
    wait_until,
)
from test_people_atomics import CAMERA_FRAME, CX, FX, HEIGHT, WIDTH, FakeVision, person, thing
from walkie_agent.arm.commander import CommanderClient
from walkie_agent.arm.poses import load_arm_poses, load_robot_model
from walkie_agent.gaze import (
    ARM_MOTION,
    HEADING,
    INTERRUPT,
    LOST,
    NAVIGATION,
    NAVIGATION_ACTIVE,
    NEW_TASK,
    PERSON,
    SHUTDOWN,
    STOP_LOOKING,
    GazeLoop,
    wrap_angle,
)
from walkie_agent.perception.geometry import camera_transform
from walkie_agent.perception.vision_feed import PEOPLE_TOPIC, VisionFeed
from walkie_agent.sim.fake_commander import FakeCommander
from walkie_agent.toolbox import StepError, Toolbox, ToolOutput
from walkie_agent.tools import FORBIDDEN_NAMES, build_walkie_tools
from walkie_agent.tools.atomic import look_at as look_at_module
from walkie_agent.tools.atomic._gaze import TURN_TIMEOUT
from walkie_agent.tools.atomic._perception import SeatPayload
from walkie_agent.tools.navigation import Nav2GoalTool

TRACK_ID = 7
# How far past the deadband a base that has just stopped may be: it turns at
# the old command until the loop's zero reaches it.
SETTLE = 0.02
LOOKING = "The robot turns to keep them in view, and stops if it loses them for 1 s."
ARMS_NOT_FOUND = "Arm state unknown: the commander was not found."


def column(heading: float, base_yaw: float) -> float:
    """The image column of something at map `heading`, seen from a base at `base_yaw`."""
    return CX - FX * math.tan(wrap_angle(heading - base_yaw))


def faces(base: FakeBase, heading: float, tolerance: float = DEADBAND + SETTLE) -> bool:
    return abs(wrap_angle(base.yaw - heading)) <= tolerance


def pose_known(ctx) -> bool:
    try:
        ctx.robot_pose(0.1)
    except StepError:
        return False
    return True


class Camera:
    """Publishes one image's people at about 15 Hz, as walkie-vision would.

    The person on track TRACK_ID stands at map heading `heading`, seen from
    the base's yaw now; `heading` None shows nobody, an empty array.
    """

    def __init__(
        self, vision: FakeVision, base_yaw: Callable[[], float], heading: float | None
    ) -> None:
        self.heading = heading
        self._vision, self._base_yaw = vision, base_yaw
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._closed.wait(1 / 15):
            heading = self.heading
            people = (
                []
                if heading is None
                else [person(column(heading, self._base_yaw()), track_id=TRACK_ID)]
            )
            try:
                self._vision.show(people=people)
            except (AssertionError, StepError):
                # One array that did not reach the feed in time; the next may.
                continue

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)


class World:
    """A Toolbox with the vision and gaze backends, the fakes that feed it, and its tools."""

    def __init__(
        self, connector, toolbox: Toolbox, gaze: GazeLoop, vision: FakeVision, recorder: Recorder
    ) -> None:
        self.connector = connector
        self.toolbox = toolbox
        self.ctx = toolbox.ctx
        self.gaze = gaze
        self.vision = vision
        self.recorder = recorder
        self.tools = {tool.name: tool for tool in toolbox.tools()}
        self._bases: list[FakeBase] = []
        self._cameras: list[Camera] = []
        self._closed = False

    def call(self, tool: str, /, **kwargs) -> ToolOutput:
        return self.tools[tool].call(**kwargs)

    def fails(self, tool: str, /, **kwargs) -> StepError:
        with pytest.raises(StepError) as error:
            self.call(tool, **kwargs)
        return error.value

    def base(self, yaw: float = 0.0) -> FakeBase:
        """A base at `yaw` that turns at the loop's commands, once ctx sees it."""
        base = FakeBase(self.connector, self.recorder, yaw)
        self._bases.append(base)
        assert wait_until(lambda: pose_known(self.ctx))
        return base

    def camera(self, base_yaw: Callable[[], float], heading: float | None) -> Camera:
        camera = Camera(self.vision, base_yaw, heading)
        self._cameras.append(camera)
        return camera

    def find_person(self) -> str:
        out = self.call("find_people")
        [handle] = out.handles
        return handle

    def show_person(self, heading: float, base_yaw: float = 0.0, track_id: int = TRACK_ID) -> str:
        """One image with one person at map `heading`, and their handle."""
        self.vision.show(people=[person(column(heading, base_yaw), track_id=track_id)])
        return self.find_person()

    def find_seat(self, objects=None) -> str:
        """The first place of one image with `objects` and nobody in it; a chair by default.

        The default chair's centre is at column 150: atan(170 / 320), 0.49 rad,
        left of the camera's axis.
        """
        self.vision.show(objects=objects or [thing("chair", 80, 200, 220, 400)], people=[])
        return self.call("find_seats").handles[0]

    def payload(self, handle: str):
        return self.ctx.handles.get(handle).payload

    def turning(self) -> bool:
        """True once the loop has sent a command that turns the base."""
        return any(z != 0.0 for z in self.recorder.commands())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for camera in self._cameras:
            camera.close()
        self.gaze.shutdown()
        for base in self._bases:
            base.close()
        self.vision.close()


@pytest.fixture
def recorder():
    node = Recorder()
    yield node
    try:
        # Whatever a test did, the loop never sent anything but a turn in place.
        assert_rotation_only(node.messages())
    finally:
        node.close()


@pytest.fixture
def make_world(connector, recorder):
    """A World factory; `arm` is the CommanderClient for the arm backend, if any."""
    made: list[World] = []

    def make(arm: CommanderClient | None = None) -> World:
        feed = VisionFeed(connector)
        gaze = GazeLoop(connector, vision=feed, arm=arm)
        backends = {"vision": feed, "gaze": gaze}
        if arm is not None:
            backends["arm"] = arm
        toolbox = Toolbox(connector, backends)
        gaze.attach(toolbox)
        world = World(connector, toolbox, gaze, FakeVision(connector, feed), recorder)
        made.append(world)
        world.vision.wait_for_subscribers()
        assert wait_until(
            lambda: gaze._publisher._publisher.get_subscription_count() > 0
            and recorder.nav_status.get_subscription_count() > 0
        ), "The recorder and the loop did not discover each other."
        return world

    yield make
    for world in made:
        world.close()


@pytest.fixture
def world(make_world):
    return make_world()


@pytest.fixture(scope="module")
def poses():
    return load_arm_poses(POSES_PATH, MODEL_PATH)


@pytest.fixture
def fake():
    node = FakeCommander(load_robot_model(MODEL_PATH), move_time_s=0.2, gripper_time_s=0.05)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    executor.shutdown()
    node.destroy_node()
    thread.join(timeout=2)


@pytest.fixture
def arm_world(fake, connector, poses, make_world):
    """A World whose arm backend is a CommanderClient that has found FakeCommander."""
    client = CommanderClient(connector, poses)
    try:
        assert wait_until(client.available, 5.0), "The client never discovered the fake."
        world = make_world(arm=client)
        yield world
        # The loop reads the client's last state, so it ends first.
        world.close()
    finally:
        client.destroy()


# -- look_at on a person ---------------------------------------------------------------


def test_look_at_a_person_turns_toward_them_and_follows_their_box(world):
    base = world.base(0.0)
    camera = world.camera(lambda: base.yaw, heading=0.5)
    target = world.find_person()

    out = world.call("look_at", target=target)

    assert out.summary == f"Looking at {target}. {LOOKING}"
    assert (out.target, out.kind, out.arms_unknown) == (target, "person", False)
    assert out.heading == pytest.approx(0.5, abs=0.02)
    assert wait_until(lambda: faces(base, 0.5), 5.0)

    # The person walks to the robot's right: the loop follows the new boxes.
    camera.heading = -0.3
    assert wait_until(lambda: faces(base, -0.3), 6.0)
    status = world.gaze.status()
    assert (status.running, status.target, status.kind) == (True, target, PERSON)
    assert status.setpoint == pytest.approx(-0.3, abs=0.05)

    stopped = world.call("stop_looking")

    assert stopped.summary == f"Stopped looking at {target}."
    assert (stopped.stopped, stopped.last_target, stopped.reason) == (target, target, STOP_LOOKING)
    assert_silent_after_stop(world.recorder)


def test_look_at_a_person_stops_once_they_are_out_of_view_for_one_second(world):
    base = world.base(0.0)
    camera = world.camera(lambda: base.yaw, heading=0.4)
    target = world.find_person()
    world.call("look_at", target=target)
    assert wait_until(world.turning)

    camera.heading = None
    hidden = time.monotonic()
    assert wait_until(lambda: not world.gaze.status().running, 3.0)
    lost_after = time.monotonic() - hidden

    # 1 s after the last sighting, which came up to one image before they left.
    assert 0.8 <= lost_after <= 1.8
    assert world.gaze.status().stop_reason == LOST
    assert_silent_after_stop(world.recorder)
    out = world.call("stop_looking")
    assert re.fullmatch(
        rf"Not looking at anyone\. The last target, {re.escape(target)}, was lost \d+ s ago\.",
        out.summary,
    )
    assert (out.stopped, out.last_target, out.reason) == (None, target, LOST)


def test_look_at_refuses_an_untracked_person(world, connector):
    place_base(connector, 0.0)
    target = world.show_person(0.3, track_id=0)

    error = world.fails("look_at", target=target)

    assert (error.code, error.message) == (
        "REFUSED",
        f"{target} is not tracked, cannot follow. Call find_people again for a handle the "
        "robot can follow.",
    )
    assert error.data == {"handle": target}
    time.sleep(0.2)
    assert world.recorder.count() == 0


def test_look_at_names_the_handles_it_takes_for_an_unknown_one(world):
    error = world.fails("look_at", target="per_0000.0")

    assert (error.code, error.message) == (
        "NOT_FOUND",
        "per_0000.0 is not a known handle. look_at takes one person handle from find_people, "
        "such as per_51c0.1, or one seat handle from find_seats.",
    )


# -- look_at on a seat -----------------------------------------------------------------


def test_look_at_a_seat_answers_once_the_base_faces_it_and_holds_the_heading(world):
    base = world.base(0.0)
    seat = world.find_seat()
    heading = world.payload(seat).heading_map
    assert heading == pytest.approx(math.atan(170 / 320), abs=1e-3)

    out = world.call("look_at", target=seat)

    assert out.summary == f"Turned toward {seat} (chair)."
    assert (out.target, out.kind, out.arms_unknown) == (seat, "seat", False)
    assert out.heading == pytest.approx(heading)
    assert faces(base, heading)
    # A heading is never lost: the loop holds it past lost_after_s.
    time.sleep(1.2)
    status = world.gaze.status()
    assert (status.running, status.target, status.kind) == (True, seat, HEADING)
    assert faces(base, heading)

    world.call("stop_looking")
    assert_silent_after_stop(world.recorder)


def test_look_at_a_seat_turns_to_its_stored_heading_after_tf_has_dropped_its_stamp(
    world, connector
):
    base = world.base(0.0)
    seat = world.find_seat(objects=[thing("sofa", 140, 240, 500, 400)])
    payload = world.payload(seat)
    assert (payload.label, payload.part) == ("sofa", "left")
    # As 10 s later: the buffer forgets the base's history, and keeps only
    # the static camera mount. The base is now elsewhere in its turn.
    base.close()
    connector._tf_buffer.clear()
    turned = world.base(-0.4)
    assert camera_transform(payload.header, connector._tf_buffer, 0.0) is None

    out = world.call("look_at", target=seat)

    assert out.summary == f"Turned toward {seat} (sofa, left)."
    assert out.heading == pytest.approx(payload.heading_map)
    assert faces(turned, payload.heading_map)
    world.call("stop_looking")


def test_look_at_a_seat_without_a_heading_is_unavailable(world, connector):
    place_base(connector, 0.0)
    header = Header(frame_id=CAMERA_FRAME, stamp=world.vision.stamp())
    seat = SeatPayload(
        position="left",
        box=(80.0, 200.0, 220.0, 400.0),
        label="chair",
        part=None,
        occupied=False,
        header=header,
        image_width=WIDTH,
        image_height=HEIGHT,
        heading_map=None,
    )
    [handle] = world.ctx.handles.put_set("seat", [seat]).member_ids

    error = world.fails("look_at", target=handle)

    assert (error.code, error.message) == (
        "UNAVAILABLE",
        f"No heading toward {handle}: TF had no camera position when it was seen. "
        "Call find_seats again.",
    )
    assert world.recorder.count() == 0


def test_a_turn_to_a_seat_that_takes_too_long_stops_the_loop(world, connector, monkeypatch):
    # A base that never turns, as with a stalled drive.
    place_base(connector, 0.0)
    monkeypatch.setattr(look_at_module, "TURN_TIMEOUT_S", 0.3)
    seat = world.find_seat()

    error = world.fails("look_at", target=seat)

    assert (error.code, error.message) == (
        "TIMEOUT",
        f"The robot did not face {seat} within 0.3 s, so it stopped turning.",
    )
    status = world.gaze.status()
    assert (status.running, status.stop_reason) == (False, TURN_TIMEOUT)
    assert_silent_after_stop(world.recorder)
    out = world.call("stop_looking")
    assert re.fullmatch(
        rf"Not looking at anyone\. The last target, {re.escape(seat)}, was stopped \d+ s ago "
        r"because the turn took too long\.",
        out.summary,
    )


# -- Stops ---------------------------------------------------------------------------------


def test_an_interrupt_stops_a_person_look_at(world):
    base = world.base(0.0)
    world.camera(lambda: base.yaw, heading=0.5)
    world.call("look_at", target=world.find_person())
    assert wait_until(world.turning)

    world.toolbox.interrupt()

    status = world.gaze.status()
    assert (status.running, status.stop_reason) == (False, INTERRUPT)
    assert_silent_after_stop(world.recorder)


def test_an_interrupt_during_a_turn_to_a_seat_cancels_look_at(world):
    world.base(0.0)
    seat = world.find_seat()
    errors: list[StepError] = []

    def look() -> None:
        try:
            world.call("look_at", target=seat)
        except StepError as e:
            errors.append(e)

    looking = threading.Thread(target=look)
    looking.start()
    assert wait_until(world.turning)
    world.toolbox.interrupt()
    looking.join(timeout=2)

    assert not looking.is_alive()
    assert [e.code for e in errors] == ["CANCELLED"]
    status = world.gaze.status()
    assert (status.running, status.stop_reason) == (False, INTERRUPT)
    assert_silent_after_stop(world.recorder)


def test_stop_looking_says_what_it_stopped_and_what_was_looked_at_last(world, connector):
    nothing = world.call("stop_looking")
    assert nothing.summary == "Not looking at anyone."
    assert (nothing.stopped, nothing.last_target, nothing.reason) == (None, None, None)

    place_base(connector, 0.0)
    target = world.show_person(0.3)
    world.call("look_at", target=target)

    assert world.call("stop_looking").summary == f"Stopped looking at {target}."
    assert_silent_after_stop(world.recorder)
    again = world.call("stop_looking")
    assert re.fullmatch(
        rf"Not looking at anyone\. The last target, {re.escape(target)}, was stopped \d+ s ago "
        r"with stop_looking\.",
        again.summary,
    )
    assert (again.stopped, again.last_target, again.reason) == (None, target, STOP_LOOKING)


def test_a_new_task_stops_the_loop_and_its_handles_are_gone(world):
    base = world.base(0.0)
    world.camera(lambda: base.yaw, heading=0.5)
    target = world.find_person()
    world.call("look_at", target=target)

    world.toolbox.new_task()

    status = world.gaze.status()
    assert (status.running, status.stop_reason) == (False, NEW_TASK)
    assert_silent_after_stop(world.recorder)
    assert world.fails("look_at", target=target).code == "NOT_FOUND"
    assert world.call("stop_looking").summary.endswith("when a new task started.")


def test_after_shutdown_look_at_is_unavailable(world, connector):
    place_base(connector, 0.0)
    target = world.show_person(0.3)
    world.call("look_at", target=target)

    world.gaze.shutdown()

    assert world.call("stop_looking").summary.endswith("at shutdown.")
    assert world.gaze.status().stop_reason == SHUTDOWN
    error = world.fails("look_at", target=target)
    assert (error.code, error.message) == ("UNAVAILABLE", "The gaze loop has shut down.")


@pytest.mark.parametrize(
    ("drive", "args", "answer"),
    [
        ("navigate_to_location", {"location": "kitchen"}, "Arrived at kitchen."),
        ("move_relative", {"forward_m": 1.0}, "Moved 1 m forward."),
    ],
)
def test_a_drive_stops_the_loop_before_it_checks_the_arms_and_the_handle_goes_stale(
    drive, args, answer, fake_walkie, world, connector, locations, monkeypatch
):
    mount_camera(connector)
    assert wait_until(lambda: pose_known(world.ctx))
    tools = {tool.name: tool for tool in build_walkie_tools(connector, locations, gaze=world.gaze)}
    assert tools[drive].gaze is world.gaze
    # FakeWalkie does not turn, so the loop keeps commanding a turn.
    world.camera(lambda: fake_walkie.pose()[2], heading=0.5)
    target = world.find_person()
    world.call("look_at", target=target)
    assert wait_until(world.turning)

    at_arm_check = []
    check_arms = Nav2GoalTool._check_arms

    def spy(self) -> str:
        at_arm_check.append(world.gaze.status())
        return check_arms(self)

    monkeypatch.setattr(Nav2GoalTool, "_check_arms", spy)

    result = tools[drive].invoke(args)

    assert result.startswith(answer)
    [status] = at_arm_check
    assert (status.running, status.stop_reason) == (False, NAVIGATION)
    # Stopped by the drive, not later by Nav2's goal status.
    assert world.gaze.status().stop_reason == NAVIGATION
    assert_silent_after_stop(world.recorder)
    stale = world.fails("look_at", target=target)
    assert stale.code == "STALE"
    assert stale.message.startswith(f"{target} is stale: the robot has moved")
    assert stale.message.endswith("Person handles come from find_people.")


def test_the_loop_stays_silent_while_a_navigation_goal_is_active(world, connector):
    place_base(connector, 0.0)
    target = world.show_person(0.4)
    world.call("look_at", target=target)
    assert wait_until(world.turning)

    world.recorder.publish_goal_status(GoalStatus.STATUS_EXECUTING)

    assert wait_until(lambda: world.gaze.status().navigation_active)
    status = world.gaze.status()
    assert (status.running, status.stop_reason) == (False, NAVIGATION_ACTIVE)
    assert_silent_after_stop(world.recorder)
    refused = world.fails("look_at", target=target)
    assert (refused.code, refused.message) == (
        "REFUSED",
        f"Not turning toward {target}: a navigation goal is active.",
    )
    count = world.recorder.count()
    time.sleep(0.3)
    assert world.recorder.count() == count

    world.recorder.publish_goal_status(GoalStatus.STATUS_SUCCEEDED)
    assert wait_until(lambda: not world.gaze.status().navigation_active)
    world.call("look_at", target=target)
    assert world.gaze.status().running
    world.call("stop_looking")


# -- The arms ------------------------------------------------------------------------------


def test_look_at_refuses_to_turn_with_the_left_arm_in_receive(arm_world, fake, poses, connector):
    place_base(connector, 0.0)
    target = arm_world.show_person(0.4)
    fake.teleport(targets(poses, "receive", "left"))

    error = arm_world.fails("look_at", target=target)

    assert (error.code, error.message) == (
        "PRECONDITION_FAILED",
        "Not turning: the left arm is not in a pose that is safe for driving. Move it with "
        "move_arm_to_pose first, for example 'carry' or 'home'.",
    )
    assert error.data == {"arms": ["left"]}
    assert arm_world.gaze.status().target is None
    time.sleep(0.2)
    assert arm_world.recorder.count() == 0


def test_look_at_turns_with_both_arms_home(arm_world, fake):
    base = arm_world.base(0.0)
    arm_world.camera(lambda: base.yaw, heading=0.4)
    target = arm_world.find_person()

    out = arm_world.call("look_at", target=target)

    assert out.summary == f"Looking at {target}. {LOOKING}"
    assert out.arms_unknown is False
    assert fake.joint_state_requests == ["both_arms"]
    assert wait_until(lambda: faces(base, 0.4), 5.0)
    arm_world.call("stop_looking")


def test_move_arm_to_pose_stops_the_loop_before_the_commander_and_set_gripper_does_not(
    arm_world, monkeypatch
):
    base = arm_world.base(0.0)
    arm_world.camera(lambda: base.yaw, heading=0.5)
    target = arm_world.find_person()
    arm_world.call("look_at", target=target)

    arm_world.call("set_gripper", arm="left", state="close")
    # A few ticks: the gripper leaves the arms known, so the loop goes on.
    time.sleep(0.3)
    status = arm_world.gaze.status()
    assert (status.running, status.target) == (True, target)

    client = arm_world.ctx.backend("arm")
    joint_names = client.joint_names
    at_first_call = []

    def spy(*args, **kwargs):
        at_first_call.append(arm_world.gaze.status())
        return joint_names(*args, **kwargs)

    monkeypatch.setattr(client, "joint_names", spy)

    arm_world.call("move_arm_to_pose", pose="home")

    [status] = at_first_call
    assert (status.running, status.stop_reason) == (False, ARM_MOTION)
    assert_silent_after_stop(arm_world.recorder)
    assert arm_world.call("stop_looking").summary.endswith("before an arm moved.")


def test_look_at_turns_and_says_so_when_no_commander_is_found(make_world, connector, poses):
    client = CommanderClient(connector, poses)
    try:
        # A commander of an earlier test may linger on the graph for a moment.
        assert wait_until(lambda: not client.available())
        world = make_world(arm=client)
        base = world.base(0.0)
        world.camera(lambda: base.yaw, heading=0.4)
        target = world.find_person()

        start = time.monotonic()
        out = world.call("look_at", target=target)

        # No read and no timeout: the client knows at once that it is alone.
        assert time.monotonic() - start < 0.5
        assert out.summary == f"Looking at {target}. {LOOKING} {ARMS_NOT_FOUND}"
        assert out.arms_unknown is True
        assert wait_until(lambda: faces(base, 0.4), 5.0)
        world.close()
    finally:
        client.destroy()


# -- /cmd_vel stays forbidden --------------------------------------------------------------


def test_cmd_vel_stays_forbidden_to_everything_but_the_gaze_loop(world, connector, locations):
    assert "/cmd_vel" in FORBIDDEN_NAMES
    assert world.gaze._publisher.topic_name == "/cmd_vel"

    with pytest.raises(ValueError, match="/cmd_vel is in FORBIDDEN_NAMES"):
        world.ctx.publisher(TwistStamped, "/cmd_vel", 10)
    with pytest.raises(ValueError, match="/cmd_vel is in FORBIDDEN_NAMES"):
        world.ctx.subscription(TwistStamped, "/cmd_vel", lambda _msg: None, 10)
    with pytest.raises(ValueError, match="/walkie/reset is in FORBIDDEN_NAMES"):
        world.ctx.publisher(Empty, "/walkie/reset", 10, allow_forbidden=("/cmd_vel",))

    # The agent's own node now publishes /cmd_vel, and the LLM still does not see it.
    lister = next(
        tool
        for tool in build_walkie_tools(connector, locations, toolbox=world.toolbox, gaze=world.gaze)
        if tool.name == "get_ros2_topics_names_and_types"
    )
    assert wait_until(lambda: PEOPLE_TOPIC in lister.invoke({}))
    # The name alone, not a prefix of a longer one such as /cmd_vel_nav.
    assert re.search(r"/cmd_vel(?![\w/])", lister.invoke({})) is None
