"""A silent demo of the assembled agent, in a throwaway network with a private router.

Run it inside `unshare -Urn`, with an rmw_zenoh router on 127.0.0.1:7499. It refuses any
other network, so it cannot reach a live ROS graph. The model is scripted, the callbacks
are empty, and nothing plays sound: /to_human is only recorded. A run writes a JSON report
and exits 0 only if every assertion in that report passed.

--robot fake starts an observed FakeWalkie in this process. --robot external finds the
robot on the graph: no fake server and no fake TF, so a sim has to provide /clock, TF and
Nav2. Then the demo proves only what it could observe, and marks the cancellation of a
goal it could not catch active as "not exercised". The robot is another process, so the
router and every session must leave Zenoh's gossip on. With scouting/gossip/enabled=false
on either side, two processes on this router do not discover each other.
"""

import argparse
import json
import logging
import math
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rai.communication.ros2 import ROS2Connector
from rclpy.action import get_action_server_names_and_types_by_node
from rclpy.node import Node
from rclpy.qos import qos_profile_action_status_default, qos_profile_sensor_data
from ros_harness import (
    ObservedWalkie,
    Operator,
    latched_todos,
    poses_over,
    spinning,
    wait_for,
    wait_for_agent,
)
from rosgraph_msgs.msg import Clock
from rclpy.time import Time
from runtime_harness import ScriptedModel, ai, call, drive, todo_call, update
from tf2_ros import Buffer, TransformListener

from walkie_agent.errors import ErrorCode
from walkie_agent.locations import load_locations
from walkie_agent.main import NODE_NAME, Application, assemble
from walkie_agent.ros.io import CONSOLE_TOPIC, TODOS_TOPIC
from walkie_agent.ros.speech import SPEECH_TOPIC
from walkie_agent.runtime import STOP_LINE, ToolLog
from walkie_agent.runtime.middleware import AFTER_STOP
from walkie_agent.tasks import TaskTemplate

ROUTER_HOST, ROUTER_PORT = "127.0.0.1", 7499
FAST_DRIVE_S = 0.2
SLOW_DRIVE_S = 2.0
TERMINAL_CANCEL_S = 0.5
WAIT_S = 30.0
TURN_DEG = 30.0
TURN_TOLERANCE_DEG = 15.0
STOP_TURN_DEG = 120.0
# A simulated robot coasts, so an external robot only has to settle within these.
STILL_M, STILL_RAD = 0.03, 0.05
AUDIO_MODULES = ("sounddevice", "pyaudio", "rai_s2s", "walkie_agent.voice")
AGENT_PUBLISHERS = {CONSOLE_TOPIC, TODOS_TOPIC, SPEECH_TOPIC, "/rosout", "/parameter_events"}
MOTION_TOPICS = ("/tf", "/tf_static", "/cmd_vel", "/goal_pose")
NAV2_STATUS_TOPIC = "/navigate_to_pose/_action/status"
STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}
TERMINAL = {
    GoalStatus.STATUS_SUCCEEDED,
    GoalStatus.STATUS_CANCELED,
    GoalStatus.STATUS_ABORTED,
}
DEMO_MOVE = TaskTemplate(
    name="demo_move",
    task="Move, then report.",
    todos=[
        {"content": "Drive there", "active_form": "Driving there", "requires_tool": "go_to"},
        {"content": "Report", "requires_tool": "say"},
    ],
)
DEMO_TURN = TaskTemplate(
    name="demo_turn",
    task="Turn, then report.",
    todos=[
        {"content": "Turn", "active_form": "Turning", "requires_tool": "move_relative"},
        {"content": "Report", "requires_tool": "say"},
    ],
)
MALFORMED_TASKS = (
    "{ not json",
    json.dumps({"task": "Run it.", "template": "no_such_template"}),
    json.dumps({"task": "Run it.", "template": "demo_move", "todos": ["A step"]}),
)


class Report:
    """What the demo observed. An assertion that fails is kept, and fails the whole run."""

    def __init__(self, path: Path, args: argparse.Namespace) -> None:
        self.path = path
        self.tool_log = path.with_suffix(".tools.jsonl")
        self.data: dict[str, Any] = {
            "robot": args.robot,
            "tool_set": args.tool_set,
            "passed": False,
            "error": None,
            "assertions": [],
            "audio_backend_started": None,
            "transport": {},
            "poses": {},
            "goals": [],
            "model_calls": 0,
            "tools": [],
            "speech": [],
            "console": [],
            "seed_source": None,
            "todo_snapshots": [],
            "session_ids": [],
            "cleanup": {},
        }

    def check(self, name: str, passed: Any, detail: Any = None) -> bool:
        self.data["assertions"].append({"name": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    def answer(self, call_id: str) -> dict[str, Any]:
        """The tool log's one record of this call."""
        found = [r for r in self.tools() if r.get("call_id") == call_id]
        return found[0] if len(found) == 1 else {"status": "absent", "result": f"{len(found)} found"}

    def tools(self) -> list[dict[str, Any]]:
        if not self.tool_log.exists():
            return []
        lines = self.tool_log.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def check_cancelled_answer(self, name: str, call_id: str) -> None:
        answer = self.answer(call_id)
        cancelled = str(answer["result"]).startswith(f"{ErrorCode.CANCELLED}:")
        self.check(name, answer["status"] == "error" and cancelled, answer)

    def check_not_run(self, name: str, call_id: str) -> None:
        answer = self.answer(call_id)
        self.check(name, answer["status"] == "error" and answer["result"] == AFTER_STOP, answer)

    def record_run(self, model: ScriptedModel, operator: Operator) -> None:
        self.data["model_calls"] = len(model.received)
        self.data["speech"] = list(operator.spoken)
        self.data["console"] = list(operator.console)
        self.data["todo_snapshots"] = list(operator.todos)
        self.data["tools"] = self.tools()

    def close(self) -> bool:
        self.data["passed"] = (
            self.data["error"] is None
            and bool(self.data["assertions"])
            and all(item["passed"] for item in self.data["assertions"])
        )
        self.path.write_text(json.dumps(self.data, indent=2, default=str) + "\n", encoding="utf-8")
        return self.data["passed"]


class _Warnings(logging.Handler):
    """Keeps the bridge's warnings, so the demo can wait for a refusal rather than sleep."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def capture_output(log_path: Path) -> int:
    """Send everything printed, by Python or by C, to the log. Returns the real stdout."""
    real_stdout = os.dup(1)
    log = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return real_stdout


def interfaces() -> list[str]:
    # /proc/self/net follows this process's network namespace, whereas /sys/class/net
    # follows the namespace that sysfs was mounted in.
    lines = Path("/proc/self/net/dev").read_text(encoding="utf-8").splitlines()[2:]
    return sorted(line.partition(":")[0].strip() for line in lines)


def isolation_problems() -> list[str]:
    """Why this process could reach a graph other than its private router. Empty is good."""
    problems = []
    if os.environ.get("RMW_IMPLEMENTATION") != "rmw_zenoh_cpp":
        problems.append("RMW_IMPLEMENTATION is not rmw_zenoh_cpp.")
    if f"tcp/{ROUTER_HOST}:{ROUTER_PORT}" not in os.environ.get("ZENOH_CONFIG_OVERRIDE", ""):
        problems.append(f"ZENOH_CONFIG_OVERRIDE does not name the router on port {ROUTER_PORT}.")
    if interfaces() != ["lo"]:
        problems.append(f"The network namespace has more than loopback: {interfaces()}.")
    if not os.environ.get("ROS_DOMAIN_ID"):
        problems.append("ROS_DOMAIN_ID is not set.")
    return problems


def wait_for_router(timeout_s: float) -> None:
    def reachable() -> bool:
        try:
            with socket.create_connection((ROUTER_HOST, ROUTER_PORT), timeout=1.0):
                return True
        except OSError:
            return False

    wait_for(reachable, f"the router on port {ROUTER_PORT}", timeout_s)


def transport_settings() -> dict[str, Any]:
    return {
        "rmw_requested": os.environ.get("RMW_IMPLEMENTATION"),
        "rmw_active": rclpy.get_rmw_implementation_identifier(),
        "domain": os.environ.get("ROS_DOMAIN_ID"),
        "discovery_range": os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE"),
        "zenoh_override": os.environ.get("ZENOH_CONFIG_OVERRIDE"),
        "interfaces": interfaces(),
    }


def audio_in_use() -> list[str]:
    in_use = [name for name in AUDIO_MODULES if name in sys.modules]
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("/dev/snd"):
            in_use.append(target)
    return in_use


def check_agent_graph(report: Report, observer: Node) -> None:
    """The agent only reaches Nav2 through its client. It never publishes motion or TF."""
    topics = observer.get_publisher_names_and_types_by_node(NODE_NAME, "/")
    published = sorted(name for name, _ in topics)
    report.check("agent_publishes_only_its_own_topics", set(published) <= AGENT_PUBLISHERS, published)
    motion = {
        topic: sorted(info.node_name for info in observer.get_publishers_info_by_topic(topic))
        for topic in MOTION_TOPICS
    }
    report.check(
        "agent_publishes_no_motion_or_tf",
        not any(NODE_NAME in nodes for nodes in motion.values()),
        motion,
    )
    hosted = get_action_server_names_and_types_by_node(observer, NODE_NAME, "/")
    report.check("agent_hosts_no_action_server", not hosted, hosted)


def human_texts(messages: list[BaseMessage]) -> list[str]:
    return [m.content for m in messages if isinstance(m, HumanMessage)]


def build_app(
    args: argparse.Namespace,
    connector: ROS2Connector,
    model: ScriptedModel,
    template: TaskTemplate,
    report: Report,
) -> Application:
    return assemble(
        connector=connector,
        model=model,
        config={},
        locations=load_locations(args.locations),
        templates={template.name: template},
        robot_description="",
        tool_set=args.tool_set,
        callbacks=(),
        tool_log=ToolLog(report.tool_log),
        use_sim_time=args.use_sim_time,
    )


def workers_alive() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("walkie-runtime")]


def wait_idle(app: Application) -> None:
    assert app.runner.wait_idle(WAIT_S), "The Runner did not go idle."


def wait_settled(app: Application) -> None:
    wait_idle(app)
    wait_for(lambda: not app.unresolved(), "the goal's end to reach the client", WAIT_S)


def check_reset_cleared_everything(
    report: Report, app: Application, operator: Operator, executor: Any, session: str
) -> None:
    wait_for(lambda: app.runner.session.thread_id != session, "a new session", WAIT_S)
    wait_settled(app)
    wait_for(lambda: operator.todos[-1] == [], "the empty list", WAIT_S)
    report.data["session_ids"].append(app.runner.session.thread_id)
    late = latched_todos(executor)
    report.check("late_subscriber_gets_the_empty_list", late == [[]], late)
    config = {"configurable": {"thread_id": session}}
    checkpoint = app.runner.agent.checkpointer.get_tuple(config)
    report.check("reset_deleted_the_old_checkpoint", checkpoint is None)


def finish(
    report: Report,
    app: Application | None,
    connector: ROS2Connector | None,
    args: argparse.Namespace,
) -> None:
    """Close what a failed run left open. Both calls are safe a second time."""
    cleanup = report.data["cleanup"]
    if app is not None:
        cleanup["final_close"] = app.close(args.close_timeout)
        cleanup["unresolved"] = app.unresolved()
    if connector is not None:
        connector.shutdown()
        cleanup["connector_shut_down"] = True


def fake_script() -> list[Any]:
    """Every model reply of the fake demo, in the order its scenarios ask for them."""
    return [
        # A: a seeded task that drives, then reports.
        ai(todo_call(update("t1", status="in_progress"), call_id="a_start")),
        ai(drive("a_drive", "kitchen")),
        ai(
            todo_call(
                update("t1", status="completed"), update("t2", status="in_progress"), call_id="a_next"
            )
        ),
        ai(call("say", {"text": "I am in the kitchen."}, "a_report")),
        ai(todo_call(update("t2", status="completed"), call_id="a_end")),
        AIMessage("The isolated navigation check is complete."),
        # C: an ordinary utterance.
        AIMessage("That is only a word to me."),
        # D: a stop in the middle of a drive.
        ai(drive("d_drive", "living_room"), call("say", {"text": "I arrived."}, "d_later")),
        # E: a reset in the middle of a seeded task's drive.
        ai(todo_call(update("t1", status="in_progress"), call_id="e_start")),
        ai(drive("e_drive", "bedroom"), call("say", {"text": "I arrived."}, "e_later")),
        # F: the next task, on a fresh history.
        ai(drive("f_drive", "kitchen")),
        AIMessage("I am back in the kitchen."),
        # G: a shutdown in the middle of a drive.
        ai(drive("g_drive", "living_room")),
    ]


def run_fake(args: argparse.Namespace, report: Report) -> None:
    places = load_locations(args.locations)
    kitchen = (places["kitchen"].x, places["kitchen"].y)
    living_room = (places["living_room"].x, places["living_room"].y)
    robot = ObservedWalkie(drive_time_sec=FAST_DRIVE_S)
    operator = Operator()
    warnings = _Warnings()
    logging.getLogger("walkie_agent.ros.io").addHandler(warnings)
    model = ScriptedModel(script=fake_script())
    app: Application | None = None
    connector: ROS2Connector | None = None
    poses, check = report.data["poses"], report.check

    def start_slow_drive(send: Callable[[], None]) -> None:
        robot.drive_time_sec = SLOW_DRIVE_S
        robot.driving.clear()
        send()
        assert robot.driving.wait(WAIT_S), "The drive never started."

    def check_cancelled(name: str, since: float) -> None:
        wait_for(lambda: len(robot.ends) == robot.goals_received, "the goal's end", WAIT_S)
        end = robot.ends[-1]
        status = STATUS_NAMES.get(end.status)
        check(f"{name}_terminal_canceled", end.status == GoalStatus.STATUS_CANCELED, status)
        waited = round(end.at - since, 3)
        check(f"{name}_canceled_within_bound", end.at - since <= TERMINAL_CANCEL_S, waited)
        samples = poses_over(robot)
        poses[name] = {"at_end": end.pose, "stable": samples[:: max(1, len(samples) // 3)]}
        check(f"{name}_pose_stable", set(samples) == {end.pose}, len(samples))

    with spinning(robot, operator) as executor:
        try:
            connector = ROS2Connector(node_name=NODE_NAME, use_sim_time=args.use_sim_time)
            app = build_app(args, connector, model, DEMO_MOVE, report)
            app.start()
            wait_for_agent(app, operator, args.ready_timeout)
            report.data["session_ids"].append(app.runner.session.thread_id)
            check("first_list_is_latched_and_empty", operator.todos[0] == [], operator.todos[0])
            poses["start"] = robot.pose()

            # A: an exact template name on the task topic, with normal success.
            reply = "The isolated navigation check is complete."
            operator.send_task(task="Run the isolated navigation check.", template="demo_move")
            wait_for(lambda: reply in operator.console, "the task's reply", WAIT_S)
            wait_for(lambda: operator.statuses() == ["completed"] * 2, "the last list", WAIT_S)
            wait_for(lambda: "I am in the kitchen." in operator.spoken, "the report", WAIT_S)
            wait_settled(app)
            seeded = next(items for items in operator.todos if items)
            sources = sorted({item["source"] for item in seeded})
            report.data["seed_source"] = sources
            check("seed_source_is_operator", sources == ["operator"], sources)
            task, *reminder = human_texts(model.received[0])
            check(
                "seed_uses_the_exact_template",
                task == "Run the isolated navigation check.\n\nMove, then report."
                and "t1 Drive there" in str(reminder)
                and "t2 Report" in str(reminder),
                [task, *reminder],
            )
            lines = [
                "t1: Driving there",
                "t1 completed: Drive there",
                "t2: Report",
                "t2 completed: Report",
                reply,
            ]
            check("status_goes_to_the_console", operator.console == lines, list(operator.console))
            spoken = list(operator.spoken)
            check("only_say_reaches_the_speaker", spoken == ["I am in the kitchen."], spoken)
            statuses = [STATUS_NAMES.get(end.status) for end in robot.ends]
            check("first_goal_succeeded", statuses == ["SUCCEEDED"], statuses)
            samples = poses_over(robot)
            poses["after_success"] = samples[-1]
            check("arrived_at_the_place", math.dist(samples[-1][:2], kitchen) < 0.01, samples[-1])
            check("still_after_success", len(set(samples)) == 1, len(samples))
            check_agent_graph(report, operator)

            # B: malformed tasks are logged and ignored, and start nothing.
            calls, goals = len(model.received), robot.goals_received
            lines, refused = len(operator.console), len(warnings.messages)
            for data in MALFORMED_TASKS:
                operator.send_raw_task(data)
            wanted = refused + len(MALFORMED_TASKS)
            wait_for(lambda: len(warnings.messages) >= wanted, "three refusals", WAIT_S)
            wait_idle(app)
            check("malformed_tasks_are_logged", True, warnings.messages[refused:])
            check(
                "malformed_tasks_start_nothing",
                (len(model.received), len(operator.console), robot.goals_received)
                == (calls, lines, goals),
            )

            # C: a template's name, said aloud, is only a word.
            reply = "That is only a word to me."
            operator.say("demo_move")
            wait_for(lambda: reply in operator.console, "the utterance's reply", WAIT_S)
            wait_idle(app)
            sent = human_texts(model.received[-1])
            check("utterance_is_not_a_template", sent[-1] == "demo_move", sent)
            # The reminder is added to each model call, and the list here is empty.
            check("utterance_has_no_seed_or_reminder", "<todos>" not in str(sent), sent)
            wait_for(lambda: operator.todos[-1] == [], "the empty list", WAIT_S)
            check("utterance_starts_with_no_list", operator.todos[-1] == [])
            check("utterance_sends_no_goal", robot.goals_received == goals)

            # D: a stop in the middle of a drive.
            poses["before_stop"] = robot.pose()
            calls = len(model.received) + 1
            start_slow_drive(lambda: operator.say("Go to the living room."))
            stopped = time.monotonic()
            operator.say("stop")
            check_cancelled("stop", stopped)
            wait_settled(app)
            wait_for(lambda: STOP_LINE in operator.spoken, "the stop line", WAIT_S)
            short = math.dist(robot.pose()[:2], living_room) > 0.1
            check("stop_left_the_robot_short", short, robot.pose())
            report.check_cancelled_answer("stopped_drive_is_cancelled", "d_drive")
            report.check_not_run("later_tool_did_not_run", "d_later")
            spoken = list(operator.spoken)
            check("later_line_was_not_spoken", "I arrived." not in spoken, spoken)
            check("no_model_call_after_stop", len(model.received) == calls, len(model.received))
            check("no_goal_after_stop", robot.goals_received == goals + 1, robot.goals_received)

            # E: a reset in the middle of a seeded task's drive, with a late subscriber each side.
            session, goals = app.runner.session.thread_id, robot.goals_received
            start_slow_drive(
                lambda: operator.send_task(task="Run the check again.", template="demo_move")
            )
            in_progress = ["in_progress", "pending"]
            wait_for(lambda: operator.statuses() == in_progress, "the list in progress", WAIT_S)
            late = latched_todos(executor)
            check("late_subscriber_gets_the_latest_list", late == [operator.todos[-1]], late)
            reset = time.monotonic()
            operator.reset()
            check_cancelled("reset", reset)
            check_reset_cleared_everything(report, app, operator, executor, session)
            check("no_goal_after_reset", robot.goals_received == goals + 1, robot.goals_received)
            report.check_not_run("reset_skipped_the_later_tool", "e_later")
            spoken = list(operator.spoken)
            check("reset_spoke_nothing", spoken == ["I am in the kitchen.", STOP_LINE], spoken)

            # F: the next task has a fresh history, and its drive succeeds.
            reply, goals = "I am back in the kitchen.", robot.goals_received
            robot.drive_time_sec = FAST_DRIVE_S
            operator.say("Go back to the kitchen.")
            wait_for(lambda: reply in operator.console, "the next task's reply", WAIT_S)
            wait_settled(app)
            sent = human_texts(model.received[-1])
            check("next_task_has_a_fresh_history", sent == ["Go back to the kitchen."], sent)
            last = STATUS_NAMES.get(robot.ends[-1].status)
            check("next_goal_succeeded", last == "SUCCEEDED" and robot.goals_received == goals + 1)
            poses["after_next_task"] = robot.pose()
            arrived = math.dist(robot.pose()[:2], kitchen) < 0.01
            check("next_task_arrived", arrived, robot.pose())

            # G: a shutdown in the middle of a drive.
            start_slow_drive(lambda: operator.say("Go to the living room once more."))
            goals = robot.goals_received
            closing = time.monotonic()
            closed = app.close(args.close_timeout)
            report.data["cleanup"]["close_returned"] = closed
            check("shutdown_settled", closed, app.unresolved())
            check_cancelled("shutdown", closing)
            check("shutdown_left_nothing_unresolved", app.unresolved() == [], app.unresolved())
            check("shutdown_left_no_workers", workers_alive() == [], workers_alive())
            check("shutdown_destroyed_the_client", app._base._client is None)
            check("shutdown_sent_no_more_goals", robot.goals_received == goals, robot.goals_received)
        finally:
            report.data["goals"] = [
                {"id": end.goal_id, "status": STATUS_NAMES.get(end.status), "pose": end.pose}
                for end in robot.ends
            ]
            report.data["goals_received"] = robot.goals_received
            report.record_run(model, operator)
            finish(report, app, connector, args)


class GoalWatch(Node):
    """Watches Nav2's own status topic and TF. On a real server, that is the proof of a goal.

    It reads TF itself, because the agent's pose verb refuses once a stop is in effect.
    """

    def __init__(self) -> None:
        super().__init__("walkie_goal_watch")
        self.latest: dict[str, int] = {}
        self.clock: list[float] = []
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self.create_subscription(
            GoalStatusArray, NAV2_STATUS_TOPIC, self._on_status, qos_profile_action_status_default
        )
        self.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data)

    def _on_status(self, message: GoalStatusArray) -> None:
        for goal in message.status_list:
            self.latest[bytes(goal.goal_info.goal_id.uuid).hex()] = goal.status

    def _on_clock(self, message: Clock) -> None:
        stamp = message.clock.sec + message.clock.nanosec / 1e9
        self.clock = [*self.clock[-1:], stamp]

    def clock_advances(self) -> bool:
        return len(self.clock) == 2 and self.clock[1] > self.clock[0]

    def pose(self) -> tuple[float, float, float]:
        moved = self._tf.lookup_transform("map", "base_footprint", Time()).transform
        q = moved.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (moved.translation.x, moved.translation.y, yaw)

    def executing(self, known: set[str]) -> str | None:
        for goal, status in list(self.latest.items()):
            if goal not in known and status == GoalStatus.STATUS_EXECUTING:
                return goal
        return None

    def ended(self, goal: str) -> bool:
        return self.latest.get(goal) in TERMINAL


def external_script() -> list[Any]:
    return [
        # The camera is not there, and the model recovers from that.
        ai(call("find_people", {}, "people")),
        ai(todo_call(update("t1", status="in_progress"), call_id="start")),
        ai(call("move_relative", {"forward_m": 0.0, "turn_deg": TURN_DEG}, "turn")),
        ai(
            todo_call(
                update("t1", status="completed"), update("t2", status="in_progress"), call_id="next"
            )
        ),
        ai(call("say", {"text": "I turned."}, "report")),
        ai(todo_call(update("t2", status="completed"), call_id="end")),
        AIMessage("The turn is complete."),
        # A goal of its own to stop, if it stays active long enough to catch.
        ai(
            call("move_relative", {"forward_m": 0.0, "turn_deg": STOP_TURN_DEG}, "stop_turn"),
            call("say", {"text": "I turned again."}, "stop_later"),
        ),
    ]


def run_external(args: argparse.Namespace, report: Report) -> None:
    operator = Operator()
    watch = GoalWatch()
    model = ScriptedModel(script=external_script())
    app: Application | None = None
    connector: ROS2Connector | None = None
    poses, check = report.data["poses"], report.check
    report.data["cancellation"] = "not exercised"

    with spinning(operator, watch) as executor:
        try:
            wait_for(watch.clock_advances, "an advancing /clock", args.ready_timeout)
            check("clock_advances", True, list(watch.clock))
            connector = ROS2Connector(node_name=NODE_NAME, use_sim_time=args.use_sim_time)
            app = build_app(args, connector, model, DEMO_TURN, report)
            app.start()
            wait_for_agent(app, operator, args.ready_timeout)
            check("fresh_tf_and_nav2_ready", True)
            report.data["session_ids"].append(app.runner.session.thread_id)
            check_agent_graph(report, operator)

            # A measured turn in place, chosen by the exact template name.
            before = watch.pose()
            poses["before_turn"] = before
            operator.send_task(task="Run the isolated turn check.", template="demo_turn")
            wait_for(lambda: "The turn is complete." in operator.console, "the reply", WAIT_S * 4)
            wait_settled(app)
            after = watch.pose()
            poses["after_turn"] = after
            delta = after[2] - before[2]
            turned = math.degrees(math.atan2(math.sin(delta), math.cos(delta)))
            measured = abs(turned - TURN_DEG) <= TURN_TOLERANCE_DEG
            check("turn_is_measured_on_tf", measured, round(turned, 2))
            turn = report.answer("turn")
            check("turn_succeeded", turn["status"] == "success", turn)
            statuses = [STATUS_NAMES.get(status) for status in watch.latest.values()]
            check("nav2_reports_success", "SUCCEEDED" in statuses, statuses)
            people = report.answer("people")
            unavailable = str(people["result"]).startswith(f"{ErrorCode.UNAVAILABLE}:")
            check("people_are_unavailable", people["status"] == "error" and unavailable, people)
            wait_for(lambda: "I turned." in operator.spoken, "the report", WAIT_S)
            check("report_was_spoken", operator.spoken == ["I turned."], list(operator.spoken))
            wait_for(lambda: operator.statuses() == ["completed"] * 2, "the last list", WAIT_S)
            sources = sorted({item["source"] for item in operator.todos[-1]})
            report.data["seed_source"] = sources
            check("seed_source_is_operator", sources == ["operator"], sources)

            # The latched list, and a reset that clears it.
            late = latched_todos(executor)
            check("late_subscriber_gets_the_latest_list", late == [operator.todos[-1]], late)
            session = app.runner.session.thread_id
            operator.reset()
            check_reset_cleared_everything(report, app, operator, executor, session)

            # Stop a separate goal, once the server says it is running.
            known = set(watch.latest)
            calls = len(model.received) + 1
            operator.say("Turn around.")

            def goal_seen() -> bool:
                return bool(watch.executing(known)) or report.answer("stop_turn")["status"] != "absent"

            wait_for(goal_seen, "the second goal", WAIT_S * 2)
            goal = watch.executing(known)
            if goal is None:
                detail: Any = "The goal ended before the demo could see it running."
            else:
                operator.say("stop")
                wait_for(lambda: watch.ended(goal), "the second goal's end", WAIT_S * 2)
                wait_settled(app)
                status = watch.latest[goal]
                time.sleep(1.0)  # a simulated robot needs to coast to rest
                first = watch.pose()
                time.sleep(1.0)
                second = watch.pose()
                poses["after_stop"] = [first, second]
                moved = math.dist(first[:2], second[:2])
                still = moved <= STILL_M and abs(second[2] - first[2]) <= STILL_RAD
                detail = {"status": STATUS_NAMES.get(status), "still": still}
                # Any other end means that the goal beat the stop, which proves nothing.
                if status == GoalStatus.STATUS_CANCELED:
                    check("cancelled_goal_came_to_rest", still, poses["after_stop"])
                    wait_for(lambda: STOP_LINE in operator.spoken, "the stop line", WAIT_S)
                    report.check_not_run("later_tool_did_not_run", "stop_later")
                    check("no_call_after_stop", len(model.received) == calls, len(model.received))
                    if still:
                        report.data["cancellation"] = "passed"
            report.data["cancellation_detail"] = detail
            wait_idle(app)
            closed = app.close(args.close_timeout)
            report.data["cleanup"]["close_returned"] = closed
            check("shutdown_settled", closed, app.unresolved())
            check("shutdown_left_no_workers", workers_alive() == [], workers_alive())
        finally:
            report.data["goals"] = [
                {"id": goal, "status": STATUS_NAMES.get(status)}
                for goal, status in watch.latest.items()
            ]
            report.record_run(model, operator)
            finish(report, app, connector, args)


def run(args: argparse.Namespace, report: Report) -> None:
    problems = isolation_problems()
    report.check("network_is_isolated", not problems, problems)
    if problems:
        return
    wait_for_router(args.ready_timeout)
    rclpy.init()
    try:
        report.data["transport"] = transport_settings()
        report.tool_log.unlink(missing_ok=True)
        (run_fake if args.robot == "fake" else run_external)(args, report)
    finally:
        rclpy.try_shutdown()
        report.data["cleanup"]["context_shut_down"] = not rclpy.ok()
        in_use = audio_in_use()
        report.data["audio_backend_started"] = bool(in_use)
        report.check("no_audio_module_or_device", not in_use, in_use)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--robot", choices=("fake", "external"), required=True)
    parser.add_argument("--tool-set", choices=("hri",), default="hri")
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument("--use-sim-time", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    parser.add_argument("--close-timeout", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    real_stdout = capture_output(args.report.with_suffix(".log"))
    report = Report(args.report, args)
    try:
        run(args, report)
    except BaseException as error:  # the report must say why the run ended
        report.data["error"] = "".join(traceback.format_exception(error))
        logging.exception("The demo failed.")
    passed = report.close()
    failed = [item["name"] for item in report.data["assertions"] if not item["passed"]]
    summary = f"{'PASS' if passed else 'FAIL'} {args.report}"
    summary += f" failed={failed}" if failed else ""
    summary += " error" if report.data["error"] else ""
    os.write(real_stdout, (summary + "\n").encode())
    return 0 if passed else 1


if __name__ == "__main__":
    # A daemon thread left by a failed run must not hold the process open.
    exit_code = main()
    sys.stdout.flush()
    os._exit(exit_code)
