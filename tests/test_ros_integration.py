"""The assembled agent on real rclpy: a real RAI connector, topics and a Nav2 action.

The server is an observed FakeWalkie, and the model is scripted. So this proves the
wiring and the goal's life on a controlled server, not TF on hardware or physical braking.
The acceptance races need a precise order, so tests/test_ros_base.py plays them with
deterministic doubles. Stops wait for a drive-start event, never for a guessed time.
"""

import asyncio
import math
import threading
import time
from dataclasses import dataclass

import pytest
from action_msgs.msg import GoalStatus
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from ros_harness import (
    ObservedWalkie,
    Operator,
    latched_todos,
    poses_over,
    spinning,
    wait_for,
    wait_for_agent,
)
from runtime_harness import ScriptedModel, ai, call, drive, todo_call, update

from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.main import Application, assemble
from walkie_agent.ros.operator_topics import (
    CONSOLE_TOPIC,
    INPUT_TOPIC,
    RESET_TOPIC,
    TASK_TOPIC,
    TODOS_TOPIC,
)
from walkie_agent.ros.speech import SPEECH_TOPIC
from walkie_agent.runtime import STOP_LINE
from walkie_agent.runtime.middleware import AFTER_STOP
from walkie_agent.runtime.task_templates import TaskTemplate

DONE = AIMessage("Done.")
# A drive this long has ten steps of 0.2 s, so a cancel lands well before the arrival.
SLOW_DRIVE_S = 2.0
# What a controlled fake must do. One step of its loop, plus the way through ROS.
TERMINAL_CANCEL_S = 0.5
DEMO_MOVE = TaskTemplate(
    name="demo_move",
    task="Move, then report.",
    todos=[
        {"content": "Drive there", "active_form": "Driving there", "requires_tool": "go_to"},
        {"content": "Report", "requires_tool": "say"},
    ],
)
TEMPLATES = {DEMO_MOVE.name: DEMO_MOVE}
MOVE_AND_REPORT = [
    ai(todo_call(update("t1", status="in_progress"), call_id="start")),
    ai(drive("drive", "kitchen")),
    ai(
        todo_call(
            update("t1", status="completed"), update("t2", status="in_progress"), call_id="next"
        )
    ),
    ai(call("say", {"text": "I am in the kitchen."}, "report")),
    ai(todo_call(update("t2", status="completed"), call_id="end")),
    AIMessage("The check is complete."),
]


@dataclass
class Rig:
    app: Application
    model: ScriptedModel
    robot: ObservedWalkie
    operator: Operator

    def history(self, thread_id: str | None = None) -> list:
        thread_id = thread_id or self.app.runner.session.thread_id
        state = self.app.runner.agent.get_state({"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])

    def answer(self, call_id: str) -> ToolMessage:
        answers = [m for m in self.history() if isinstance(m, ToolMessage)]
        found = [m for m in answers if m.tool_call_id == call_id]
        assert len(found) == 1, f"{call_id} has {len(found)} answers"
        return found[0]

    def wait_settled(self) -> None:
        """Wait for the Runner to go idle and for the client to hear its goal's end."""
        assert self.app.runner.wait_idle(20.0), "The Runner did not go idle."
        wait_for(lambda: not self.app.unresolved(), "the goal's end to reach the client")

    def start_slow_drive(self, send) -> None:
        """Make drives slow, run `send`, and return once the server is at its first step."""
        self.robot.drive_time_sec = SLOW_DRIVE_S
        self.robot.driving.clear()
        send()
        assert self.robot.driving.wait(20.0), "The drive never started."

    def assert_cancelled_within_bound(self, since: float, goal: int = -1) -> None:
        wait_for(lambda: len(self.robot.ends) >= self.robot.goals_received, "the goal's end")
        end = self.robot.ends[goal]
        assert end.status == GoalStatus.STATUS_CANCELED
        assert end.at - since <= TERMINAL_CANCEL_S

    def assert_still_after_end(self) -> None:
        samples = poses_over(self.robot)
        assert set(samples) == {self.robot.ends[-1].pose}


@pytest.fixture
def rig(ros, locations, connector):
    robot = ObservedWalkie(drive_time_sec=0.2)
    operator = Operator()
    apps: list[Application] = []
    with spinning(robot, operator):

        def start(*script, then=None, templates=None, tool_set="hri"):
            model = ScriptedModel(script=list(script), then=then)
            app = assemble(
                connector=connector,
                model=model,
                config={},
                locations=locations,
                templates=templates or {},
                robot_description="",
                tool_set=tool_set,
            )
            apps.append(app)
            app.start()
            wait_for_agent(app, operator)
            return Rig(app, model, robot, operator)

        yield start
        for app in apps:
            assert app.close(10.0), app.unresolved()


def test_ros_hri_task_reply_and_status_flow(rig):
    # test_agent.py:77, :96, :131, :269, :378. First a task by exact template name, whose
    # status lines reach the console. Then an utterance, whose reply is a real HRIMessage.
    r = rig(
        *MOVE_AND_REPORT,
        # One message, run in order: the line is heard before the goal reaches the server.
        ai(call("say", {"text": "On my way."}, "first"), drive("second", "living_room")),
        # Failed calls go back to the model, and what they planned does not run.
        ai(drive("lost", "attic"), call("say", {"text": "Arrived in the attic."}, "skipped")),
        ai(call("find_people", {}, "people")),
        AIMessage("I could not do all of that."),
        templates=TEMPLATES,
    )
    assert r.operator.todos == [[]]

    r.operator.send_task(task="Run the check.", template="demo_move")
    wait_for(lambda: "The check is complete." in r.operator.console, "the task's reply")

    assert r.model.received[0][1].content == "Run the check.\n\nMove, then report."
    assert r.operator.console == [
        "t1: Driving there",
        "t1 completed: Drive there",
        "t2: Report",
        "t2 completed: Report",
        "The check is complete.",
    ]
    wait_for(lambda: "I am in the kitchen." in r.operator.spoken, "the report on /to_human")
    wait_for(lambda: r.operator.statuses() == ["completed", "completed"], "the last list")
    assert all(i["source"] == "operator" and i["locked"] for i in r.operator.todos[-1])
    assert not {"added", "since", "used"} & set(r.operator.todos[-1][0])
    assert r.robot.goals_received == 1
    assert [end.status for end in r.robot.ends] == [GoalStatus.STATUS_SUCCEEDED]
    assert r.answer("drive").content == "Arrived at kitchen."

    r.operator.say("Go to the living room, then the attic.")
    wait_for(lambda: "I could not do all of that." in r.operator.console, "the second reply")
    wait_for(lambda: "On my way." in r.operator.spoken, "the second line on /to_human")
    assert r.answer("second").status == "success"
    assert r.answer("lost").status == "error"
    assert r.answer("lost").content.startswith(f"{ErrorCode.NOT_FOUND}: Unknown place 'attic'")
    assert r.answer("skipped").content.startswith("Not run: go_to failed earlier")
    assert r.answer("people").status == "error"
    assert r.answer("people").content.startswith(f"{ErrorCode.UNAVAILABLE}:")
    assert "Arrived in the attic." not in r.operator.spoken
    # Neither the refused name nor the unavailable capability sent a goal.
    assert r.robot.goals_received == 2
    # The reply went to the console, and only `say` reached the speaker.
    assert "I could not do all of that." not in r.operator.spoken
    assert r.operator.spoken == ["I am in the kitchen.", "On my way."]


def test_named_and_relative_moves_read_tf(rig, locations):
    # test_tools_ros.py:31, :128, :172, :188
    r = rig(
        ai(drive("named", "KITCHEN")),
        ai(call("move_relative", {"forward_m": 1.0, "turn_deg": -90.0}, "relative")),
        ai(call("move_relative", {"forward_m": 1.0, "left_m": 0.5}, "after_arrival")),
        DONE,
    )
    r.operator.say("Go to the kitchen, then move.")
    wait_for(lambda: "Done." in r.operator.console, "the reply")

    assert r.answer("named").content == "Arrived at KITCHEN."
    # The kitchen faces along y, so forward is +y. Each move starts from a pose read after
    # the last one arrived, so the second uses the heading the first ended on.
    kitchen = locations["kitchen"]
    x, y = kitchen.x + math.cos(kitchen.yaw), kitchen.y + math.sin(kitchen.yaw)
    yaw = kitchen.yaw - math.pi / 2
    assert r.answer("relative").content.startswith("Moved 1 m forward, turning 90 degrees right.")
    assert f"x={x:.2f} m, y={y:.2f} m" in r.answer("relative").content
    x, y = x + math.cos(yaw) - 0.5 * math.sin(yaw), y + math.sin(yaw) + 0.5 * math.cos(yaw)
    assert r.answer("after_arrival").content.startswith("Moved 1 m forward, 0.5 m left.")
    assert f"x={x:.2f} m, y={y:.2f} m" in r.answer("after_arrival").content

    assert r.robot.pose() == pytest.approx((x, y, yaw), abs=0.01)
    wait_for(
        lambda: r.app.ctx.base.pose().x == pytest.approx(x, abs=0.01),
        "TF to reach the final position",
    )
    pose = r.app.ctx.base.pose()
    assert (pose.x, pose.y, pose.yaw) == pytest.approx((x, y, yaw), abs=0.01)
    assert [end.status for end in r.robot.ends] == [GoalStatus.STATUS_SUCCEEDED] * 3


def test_timeout_reaches_terminal_cancel_and_pose_stays_still(rig, locations):
    # test_tools_ros.py:74, stronger: it proves the end of the goal, not only the cancel.
    r = rig()
    r.robot.drive_time_sec = SLOW_DRIVE_S
    with pytest.raises(StepError) as raised:
        r.app.ctx.base.go_to("kitchen", timeout_s=1.0)
    timed_out = time.monotonic()
    assert raised.value.code is ErrorCode.TIMEOUT
    r.assert_cancelled_within_bound(timed_out)
    r.assert_still_after_end()
    kitchen = (locations["kitchen"].x, locations["kitchen"].y)
    assert r.robot.ends[-1].pose[:2] != pytest.approx(kitchen, abs=0.1)
    assert r.robot.goals_received == 1

    # A timeout is not a stop: no latch remains, and motion works again once it has settled.
    wait_for(lambda: not r.app.unresolved(), "the goal's end to reach the client")
    r.robot.drive_time_sec = 0.2
    assert r.app.ctx.base.go_to("kitchen").x == pytest.approx(locations["kitchen"].x, abs=0.01)
    assert r.robot.ends[-1].status == GoalStatus.STATUS_SUCCEEDED


def test_stop_reaches_terminal_cancel_and_skips_later_tools(rig, locations):
    # test_graph.py:835, on a real server
    r = rig(ai(drive("drive", "kitchen"), call("say", {"text": "I arrived."}, "later")), then=DONE)
    r.start_slow_drive(lambda: r.operator.say("Go to the kitchen."))

    stopped = time.monotonic()
    r.operator.say("stop")
    r.assert_cancelled_within_bound(stopped)
    r.assert_still_after_end()
    r.wait_settled()
    wait_for(lambda: STOP_LINE in r.operator.spoken, "the stop line")

    kitchen = (locations["kitchen"].x, locations["kitchen"].y)
    assert r.robot.ends[-1].pose[:2] != pytest.approx(kitchen, abs=0.1)
    assert r.answer("drive").status == "error"
    assert r.answer("drive").content.startswith(f"{ErrorCode.CANCELLED}:")
    assert r.answer("later").content == AFTER_STOP
    assert "I arrived." not in r.operator.spoken
    # No model call or goal came after the stop. The server's count shows no later send.
    assert len(r.model.received) == 1
    assert r.robot.goals_received == 1


def test_stop_before_goal_has_zero_goal_count(rig):
    entered, release = threading.Event(), threading.Event()

    async def held_reply(_messages):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return ai(drive("drive", "kitchen"), call("say", {"text": "I arrived."}, "later"))

    r = rig(held_reply, then=DONE)
    r.operator.say("Go to the kitchen.")
    assert entered.wait(20.0), "The model call never started."

    r.operator.say("stop")
    wait_for(lambda: STOP_LINE in r.operator.spoken, "the stop line")
    release.set()
    r.wait_settled()

    assert r.answer("drive").content == AFTER_STOP
    assert r.answer("later").content == AFTER_STOP
    assert r.robot.goals_received == 0
    assert r.robot.ends == []
    assert r.robot.pose() == (0.0, 0.0, 0.0)


def test_reset_cancels_motion_and_clears_checkpoint_and_latched_todos(rig):
    # test_agent.py:169, :221, :341 and test_graph.py:979. The old host let the current tool
    # finish. A reset now cancels the drive, and the server's terminal status proves it.
    r = rig(
        ai(todo_call(update("t1", status="in_progress"), call_id="start")),
        ai(drive("drive", "kitchen"), call("say", {"text": "I arrived."}, "later")),
        AIMessage("A new start."),
        templates=TEMPLATES,
    )
    r.start_slow_drive(lambda: r.operator.send_task(task="Run the check.", template="demo_move"))
    wait_for(lambda: r.operator.statuses() == ["in_progress", "pending"], "the list in progress")
    old_session = r.app.runner.session.thread_id

    reset = time.monotonic()
    r.operator.reset()
    r.assert_cancelled_within_bound(reset)
    r.assert_still_after_end()
    wait_for(lambda: r.app.runner.session.thread_id != old_session, "a new session")
    r.wait_settled()

    wait_for(lambda: r.operator.todos[-1] == [], "the empty list")
    assert latched_todos() == [[]]
    old_config = {"configurable": {"thread_id": old_session}}
    assert r.app.runner.agent.checkpointer.get_tuple(old_config) is None
    assert r.history() == []
    assert "I arrived." not in r.operator.spoken
    # A reset forgets without a word.
    assert STOP_LINE not in r.operator.spoken

    r.operator.say("Hello again.")
    wait_for(lambda: "A new start." in r.operator.console, "the next reply")
    sent = [m for m in r.model.received[-1] if not isinstance(m, SystemMessage)]
    assert [m.content for m in sent] == ["Hello again."]
    assert r.robot.goals_received == 1


def test_late_subscriber_gets_latest_todos(rig):
    # test_agent.py:341
    r = rig(*MOVE_AND_REPORT, templates=TEMPLATES)
    assert latched_todos() == [[]]

    r.operator.send_task(task="Run the check.", template="demo_move")
    wait_for(lambda: "The check is complete." in r.operator.console, "the task's reply")
    wait_for(lambda: r.operator.statuses() == ["completed", "completed"], "the last list")

    late = latched_todos()
    assert late == [r.operator.todos[-1]]
    assert [i["status"] for i in late[0]] == ["completed", "completed"]
    assert late[0][0]["evidence"] == ["drive"]
    assert late[0][1]["evidence"] == ["report"]


def test_next_task_runs_after_stop(rig, locations):
    # test_graph.py:835 and the Runner's latch. The stop holds only until the next task.
    r = rig(
        ai(drive("first", "kitchen"), call("say", {"text": "I arrived."}, "later")),
        ai(drive("second", "bedroom")),
        AIMessage("I am in the bedroom."),
    )
    r.start_slow_drive(lambda: r.operator.say("Go to the kitchen."))
    stopped = time.monotonic()
    r.operator.say("stop")
    r.assert_cancelled_within_bound(stopped)
    r.wait_settled()
    session = r.app.runner.session.thread_id

    r.robot.drive_time_sec = 0.2
    r.operator.say("Go to the bedroom.")
    wait_for(lambda: "I am in the bedroom." in r.operator.console, "the next task's reply")

    assert r.answer("second").status == "success"
    assert r.robot.goals_received == 2
    assert [end.status for end in r.robot.ends] == [
        GoalStatus.STATUS_CANCELED,
        GoalStatus.STATUS_SUCCEEDED,
    ]
    assert r.robot.ends[-1].pose == pytest.approx(
        (locations["bedroom"].x, locations["bedroom"].y, locations["bedroom"].yaw), abs=0.01
    )
    # The conversation goes on. The stop and its answer are in the history.
    assert r.app.runner.session.thread_id == session
    contents = [m.content for m in r.model.received[-1]]
    assert "stop" in contents
    assert STOP_LINE in contents


def test_shutdown_has_no_live_workers_or_goals(rig, connector):
    # test_graph.py:933. The latch used to end a task after a step, and now it ends the drive.
    r = rig(ai(drive("drive", "kitchen")), then=DONE)
    r.start_slow_drive(lambda: r.operator.say("Go to the kitchen."))

    closing = time.monotonic()
    assert r.app.close(10.0), r.app.unresolved()
    assert r.app.unresolved() == []
    r.assert_cancelled_within_bound(closing)
    r.assert_still_after_end()

    names = [t.name for t in threading.enumerate()]
    assert [name for name in names if name.startswith(("walkie-runtime", "asyncio_"))] == []
    assert r.app._base._client is None
    node = connector.node
    published = {p.topic_name for p in node.publishers}
    subscribed = {s.topic_name for s in node.subscriptions}
    assert not {CONSOLE_TOPIC, TODOS_TOPIC, SPEECH_TOPIC} & published
    assert not {INPUT_TOPIC, TASK_TOPIC, RESET_TOPIC} & subscribed

    with pytest.raises(RuntimeError, match="shut down"):
        r.app.runner.submit("Too late.")
    assert r.robot.goals_received == 1
    assert len(r.model.received) == 1
    # Closing again is safe, and the fixture does.
    assert r.app.close(1.0)
