"""RosBridge with the fake connector: real Runner, real stack, and captured ROS callbacks.

Each callback is invoked as the executor would, from the test's own thread. Nothing here
needs a graph, an LLM or a speaker.
"""

import asyncio
import json
import logging
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rai_interfaces.msg import AudioMessage, HRIMessage
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from ros_fakes import WATCHDOG_S, Background, FakeConnector, FakeNode, bounded
from runtime_harness import (
    ScriptedModel,
    add,
    ai,
    call,
    make_stack,
    todo_call,
    update,
    wait_until,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, String

from walkie_agent.adapter import as_tools
from walkie_agent.capabilities import Ctx, UnavailableVision
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.ros.io import RosBridge
from walkie_agent.runtime import STOP_LINE, Runner, build_agent
from walkie_agent.tasks import TaskTemplate
from walkie_agent.tools.atomic import DEFAULT, GPSR, HRI, find_objects, find_people

DONE = AIMessage("Done.")
TOUR = TaskTemplate(
    name="tour",
    task="Show the person the house.",
    todos=["Meet the guest", {"content": "Greet", "requires_tool": "say"}],
)
PRIVATE = {"added", "since", "used"}


def bridge_only(templates=None, **topics):
    connector = FakeConnector()
    return RosBridge(connector, templates or {}, **topics), connector


def publisher(connector, topic):
    return next(p for p in connector.node.publishers if p.topic == topic)


def subscription(connector, topic):
    return next(s for s in connector.node.subscriptions if s.topic == topic)


def assert_queued_qos(qos):
    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.history == HistoryPolicy.KEEP_LAST
    assert qos.depth == 10
    assert qos.durability == DurabilityPolicy.VOLATILE


def texts(connector, topic="/walkie/console"):
    return [message.text for message in publisher(connector, topic).published]


def lists(connector):
    return [json.loads(m.data) for m in publisher(connector, "/walkie/todos").published]


@pytest.fixture
def serve():
    """A real Runner over the real stack, on a bridge over the fake connector."""
    apps = []

    def start(*script, then=None, templates=None, robot=None, delay_s=0.0):
        connector = FakeConnector()
        bridge = RosBridge(connector, templates or {})
        stack = make_stack(
            *script, then=then, robot=robot, delay_s=delay_s, on_todos=bridge.on_todos
        )
        stack.runner = Runner(
            stack.agent, stack.robot.ctx, on_todos=bridge.on_todos, on_reply=bridge.on_reply
        )
        stack.runner.start()
        bridge.open(stack.runner)
        app = SimpleNamespace(stack=stack, bridge=bridge, connector=connector)
        apps.append(app)
        return app

    yield start
    for app in apps:
        assert app.stack.runner.shutdown(WATCHDOG_S), "the Runner did not shut down"


def idle(app):
    assert app.stack.runner.wait_idle(WATCHDOG_S), "the Runner did not go idle"


def humans(stack):
    return [m.content for m in stack.model.received[-1] if isinstance(m, HumanMessage)]


# -- Input ---------------------------------------------------------------------------


def test_hri_input_queues_text(serve):
    app = serve(then=DONE)
    subscribed = subscription(app.connector, "/from_human")
    assert subscribed.msg_type is HRIMessage
    assert_queued_qos(subscribed.qos)

    assert bounded(subscribed.callback, HRIMessage(text="Bring me the mug.")) is None
    idle(app)

    assert humans(app.stack) == ["Bring me the mug."]
    assert texts(app.connector) == ["Done."]


def test_a_bare_stop_on_the_input_topic_stops_at_once(serve):
    app = serve(then=DONE)
    subscription(app.connector, "/from_human").callback(HRIMessage(text="stop"))
    assert app.stack.robot.cancel.cancelled
    idle(app)
    assert app.stack.model.received == []
    assert app.stack.robot.said == [STOP_LINE]


@pytest.mark.parametrize(
    "payload",
    [
        {"images": [Image()]},
        {"audios": [AudioMessage()]},
        {"images": [Image()], "audios": [AudioMessage()]},
    ],
    ids=["image", "audio", "both"],
)
def test_input_with_an_image_or_audio_is_refused_and_logged(serve, caplog, payload):
    app = serve(then=DONE)
    callback = subscription(app.connector, "/from_human").callback
    with caplog.at_level(logging.WARNING, logger="walkie_agent.ros.io"):
        callback(HRIMessage(text="What is this?", **payload))
    idle(app)

    assert app.stack.model.received == []
    assert texts(app.connector) == []
    assert "text only" in caplog.text

    callback(HRIMessage(text="Go on."))
    idle(app)
    assert humans(app.stack) == ["Go on."]


def test_a_bare_stop_with_a_payload_still_stops(serve):
    app = serve(then=DONE)
    subscription(app.connector, "/from_human").callback(
        HRIMessage(text="stop", images=[Image()], audios=[AudioMessage()])
    )
    assert app.stack.robot.cancel.cancelled
    idle(app)
    assert app.stack.model.received == []
    assert app.stack.robot.said == [STOP_LINE]


@pytest.mark.parametrize("text", ["", "  \n"])
def test_input_with_no_text_is_ignored(serve, caplog, text):
    app = serve(then=DONE)
    with caplog.at_level(logging.WARNING, logger="walkie_agent.ros.io"):
        subscription(app.connector, "/from_human").callback(HRIMessage(text=text))
    idle(app)
    assert app.stack.model.received == []
    assert "no text" in caplog.text


def test_task_topic_uses_exact_template_and_operator_source(serve):
    app = serve(then=DONE, templates={"tour": TOUR})
    subscribed = subscription(app.connector, "/walkie/task")
    assert subscribed.msg_type is String
    assert_queued_qos(subscribed.qos)

    data = json.dumps({"task": "Show me around.", "template": "tour"})
    assert bounded(subscribed.callback, String(data=data)) is None
    idle(app)

    assert humans(app.stack)[0] == "Show me around.\n\nShow the person the house."
    seeded = app.stack.items()
    assert [item["content"] for item in seeded] == ["Meet the guest", "Greet"]
    assert {item["source"] for item in seeded} == {"operator"}
    assert seeded[1]["requires_tool"] == "say"

    # A template name must match exactly, spaces and case included.
    received = len(app.stack.model.received)
    for name in (" tour ", "Tour"):
        subscribed.callback(String(data=json.dumps({"task": "Again.", "template": name})))
    idle(app)
    assert len(app.stack.model.received) == received


def test_task_topic_takes_an_inline_seed_as_the_operator(serve):
    app = serve(then=DONE)
    data = json.dumps({"task": "Go.", "todos": ["Go there"]})
    subscription(app.connector, "/walkie/task").callback(String(data=data))
    idle(app)
    assert [(i["content"], i["source"]) for i in app.stack.items()] == [("Go there", "operator")]


def test_plain_utterance_never_selects_template(serve):
    app = serve(then=DONE, templates={"tour": TOUR})
    subscription(app.connector, "/from_human").callback(HRIMessage(text="tour"))
    idle(app)
    assert humans(app.stack) == ["tour"]
    assert app.stack.items() == []


@pytest.mark.parametrize(
    "data",
    [
        "Go to the kitchen",
        '["Go"]',
        '{"task": "Go."}',
        '{"task": "Go.", "template": "garage"}',
        '{"task": "Go.", "todos": [{"content": " "}]}',
        '{"task": "Go.", "todos": [{"content": "a", "require_tool": "say"}]}',
    ],
    ids=["not-json", "not-object", "no-seed", "unknown-template", "blank-item", "unknown-key"],
)
def test_invalid_task_is_logged_and_ignored(serve, caplog, data):
    app = serve(then=DONE, templates={"tour": TOUR})
    callback = subscription(app.connector, "/walkie/task").callback
    with caplog.at_level(logging.WARNING, logger="walkie_agent.ros.io"):
        assert bounded(callback, String(data=data)) is None
    idle(app)

    assert app.stack.model.received == []
    assert app.stack.robot.said == []
    assert texts(app.connector) == []
    assert "Ignored a message on /walkie/task" in caplog.text

    # The executor's callback survived, so the next good task runs.
    callback(String(data='{"task": "Go.", "todos": []}'))
    idle(app)
    assert humans(app.stack) == ["Go."]


def test_reset_queues_without_blocking_callback(serve):
    release = threading.Event()
    entered = threading.Event()

    async def hung(_messages):
        entered.set()
        await asyncio.to_thread(release.wait, WATCHDOG_S)
        return DONE

    app = serve(hung, then=DONE)
    subscribed = subscription(app.connector, "/walkie/reset")
    assert subscribed.msg_type is Empty
    assert_queued_qos(subscribed.qos)
    session = app.stack.runner.session.thread_id
    subscription(app.connector, "/from_human").callback(HRIMessage(text="Go."))
    assert entered.wait(WATCHDOG_S)

    # It returns while the run is still hung: the session changes only after that run.
    assert bounded(subscribed.callback, Empty()) is None
    assert app.stack.robot.cancel.cancelled
    assert app.stack.runner.session.thread_id == session

    release.set()
    idle(app)
    assert app.stack.runner.session.thread_id != session
    assert lists(app.connector)[-1] == []


def test_a_verdict_goes_to_its_callback_and_its_answer_to_the_console():
    verdicts = []

    def judge(passed):
        verdicts.append(passed)
        return f"Judged {passed}."

    bridge, connector = bridge_only(on_verdict=judge)
    bridge.open(SimpleNamespace())
    subscribed = subscription(connector, "/walkie/verdict")
    assert subscribed.msg_type is Bool
    assert_queued_qos(subscribed.qos)

    assert bounded(subscribed.callback, Bool(data=True)) is None
    subscribed.callback(Bool(data=False))
    assert verdicts == [True, False]
    assert texts(connector) == ["Judged True.", "Judged False."]

    # A closed ingress takes no verdict either.
    bridge.close_ingress()
    subscribed.callback(Bool(data=True))
    assert verdicts == [True, False]


def test_without_a_verdict_callback_there_is_no_verdict_topic():
    bridge, connector = bridge_only()
    bridge.open(SimpleNamespace())
    assert "/walkie/verdict" not in {s.topic for s in connector.node.subscriptions}


def test_a_verdict_callback_that_raises_is_logged_and_publishes_nothing(caplog):
    def broken(_passed):
        raise RuntimeError("shelf gone")

    bridge, connector = bridge_only(on_verdict=broken)
    bridge.open(SimpleNamespace())
    with caplog.at_level(logging.ERROR, logger="walkie_agent.ros.io"):
        assert bounded(subscription(connector, "/walkie/verdict").callback, Bool(data=True)) is None
    assert "Could not take a message on /walkie/verdict" in caplog.text
    assert texts(connector) == []


# -- Output --------------------------------------------------------------------------


def test_todos_qos_initial_empty_and_private_fields(serve):
    script = [
        ai(todo_call(update("t1", status="in_progress"), call_id="begin")),
        ai(call("say", {"text": "Hello"}, "spoken")),
        ai(todo_call(update("t1", status="completed"), call_id="end")),
    ]
    app = serve(*script, then=DONE)
    published = publisher(app.connector, "/walkie/todos")
    assert published.msg_type is String
    assert published.qos.reliability == ReliabilityPolicy.RELIABLE
    assert published.qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert published.qos.history == HistoryPolicy.KEEP_LAST
    assert published.qos.depth == 1
    # The first list comes from the loop thread, with no task for wait_idle to wait for.
    wait_until(lambda: lists(app.connector))
    assert lists(app.connector) == [[]]

    data = json.dumps(
        {"task": "Greet them.", "todos": [{"content": "Say hello", "requires_tool": "say"}]}
    )
    subscription(app.connector, "/walkie/task").callback(String(data=data))
    idle(app)

    assert app.stack.statuses() == ["completed"]
    last = lists(app.connector)[-1]
    assert last[0]["status"] == "completed"
    assert last[0]["evidence"] == ["spoken"]
    for snapshot in lists(app.connector):
        for item in snapshot:
            assert not PRIVATE & set(item)

    subscription(app.connector, "/walkie/reset").callback(Empty())
    idle(app)
    assert lists(app.connector)[-1] == []


def test_changed_status_goes_to_console_not_speaker(serve):
    script = [
        ai(todo_call(add("Plan the route", active_form="Planning the route"), call_id="plan")),
        ai(todo_call(update("t1", status="in_progress"), call_id="begin")),
        ai(todo_call(update("t1", status="completed"), call_id="end")),
    ]
    app = serve(*script, then=DONE)
    subscription(app.connector, "/from_human").callback(HRIMessage(text="Plan it."))
    idle(app)

    # Status lines and the reply reach the console, and nothing reaches the speaker.
    assert texts(app.connector) == ["t1: Planning the route", "t1 completed: Plan the route", "Done."]
    assert app.stack.robot.said == []
    assert all(p.topic != "/to_human" for p in app.connector.node.publishers)


@pytest.mark.parametrize(
    ("output_topic", "remaps", "namespace"),
    [
        ("/to_human", {}, "/"),
        ("to_human", {}, "/"),
        ("/console", {"/console": "/to_human"}, "/"),
    ],
    ids=["same-name", "relative-name", "remapped"],
)
def test_the_console_topic_may_not_be_the_speech_topic(output_topic, remaps, namespace):
    connector = FakeConnector(FakeNode(namespace=namespace, remaps=remaps))
    with pytest.raises(ValueError, match="/to_human"):
        RosBridge(connector, {}, output_topic=output_topic)
    assert connector.node.publishers == []


def test_console_reply_is_hri_message():
    bridge, connector = bridge_only()
    console = publisher(connector, "/walkie/console")
    assert console.msg_type is HRIMessage
    assert_queued_qos(console.qos)

    bridge.on_reply("First.")
    bridge.on_reply("Second.")

    first, second = console.published
    assert isinstance(first, HRIMessage)
    assert (first.text, second.text) == ("First.", "Second.")
    assert first is not second
    assert first.communication_id and first.communication_id != second.communication_id
    assert first.seq_end is True and second.seq_end is True
    assert first.header.stamp == connector.node.clock.now().to_msg()


def test_custom_topics_are_used(serve):
    bridge, connector = bridge_only(
        output_topic="/c", todos_topic="/t", input_topic="/i", task_topic="/k", reset_topic="/r"
    )
    bridge.open(SimpleNamespace())
    assert {p.topic for p in connector.node.publishers} == {"/c", "/t"}
    assert {s.topic for s in connector.node.subscriptions} == {"/i", "/k", "/r"}


def test_publish_error_does_not_kill_runner(serve, caplog):
    script = [
        ai(todo_call(add("Plan the route"), call_id="plan")),
        ai(todo_call(update("t1", status="in_progress"), call_id="begin")),
        ai(todo_call(update("t1", status="completed"), call_id="end")),
    ]
    app = serve(*script, then=DONE)
    idle(app)
    console = publisher(app.connector, "/walkie/console")
    todos = publisher(app.connector, "/walkie/todos")

    def broken(_message):
        raise RuntimeError("publisher is gone")

    console.on_publish = todos.on_publish = broken
    callback = subscription(app.connector, "/from_human").callback
    with caplog.at_level(logging.ERROR, logger="walkie_agent.ros.io"):
        callback(HRIMessage(text="Plan it."))
        idle(app)

    assert app.stack.statuses() == ["completed"], "the run finished despite the failed publishes"
    assert "Could not publish on /walkie/console" in caplog.text
    assert "Could not publish on /walkie/todos" in caplog.text
    assert console.published == []

    console.on_publish = todos.on_publish = None
    callback(HRIMessage(text="Again."))
    idle(app)
    assert texts(app.connector) == ["Done."]


def test_a_broken_clock_loses_the_line_and_nothing_else(caplog):
    bridge, connector = bridge_only()

    def broken():
        raise RuntimeError("the context is gone")

    connector.node.clock.on_now = broken
    with caplog.at_level(logging.ERROR, logger="walkie_agent.ros.io"):
        bridge.on_reply("Lost.")
    assert publisher(connector, "/walkie/console").published == []
    assert "Could not publish a console line" in caplog.text


def test_closed_ingress_ignores_late_callbacks(serve):
    app = serve(then=DONE)
    node = app.connector.node
    callbacks = {s.topic: s.callback for s in node.subscriptions}
    subscribed = list(node.subscriptions)
    idle(app)

    app.bridge.close_ingress()
    assert node.destroyed == [], "a destroy can end the spin thread that a goal still needs"

    callbacks["/from_human"](HRIMessage(text="Late."))
    callbacks["/from_human"](HRIMessage(text="stop"))
    callbacks["/walkie/task"](String(data='{"task": "Late.", "todos": []}'))
    callbacks["/walkie/reset"](Empty())
    idle(app)
    assert app.stack.model.received == []
    assert not app.stack.robot.cancel.cancelled

    # The Runner can still report a run that is ending, so the publishers stay up.
    app.bridge.on_reply("Still here.")
    assert texts(app.connector) == ["Still here."]
    assert not set(node.publishers) & set(node.destroyed)

    app.bridge.close()
    assert set(subscribed) | set(node.publishers) <= set(node.destroyed)
    count = len(node.destroyed)
    published = len(lists(app.connector))
    app.bridge.on_reply("Too late.")
    app.bridge.on_todos([], [])
    assert texts(app.connector) == ["Still here."]
    assert len(lists(app.connector)) == published

    # Repeating any of it destroys nothing twice.
    app.bridge.close_ingress()
    app.bridge.close()
    assert len(node.destroyed) == count


def test_a_callback_after_the_runner_closed_logs_and_returns(serve, caplog):
    app = serve(then=DONE)
    assert app.stack.runner.shutdown(WATCHDOG_S)
    callback = subscription(app.connector, "/walkie/reset").callback
    with caplog.at_level(logging.ERROR, logger="walkie_agent.ros.io"):
        assert bounded(callback, Empty()) is None
    assert "Could not take a message on /walkie/reset" in caplog.text


def test_a_destroy_failure_does_not_stop_the_rest(caplog):
    bridge, connector = bridge_only()
    bridge.open(SimpleNamespace())
    connector.node.destroy_error = RuntimeError("already gone")
    with caplog.at_level(logging.ERROR, logger="walkie_agent.ros.io"):
        bridge.close()
    assert len(connector.node.destroyed) == 5
    assert "Could not destroy a bridge endpoint" in caplog.text


def test_progress_callbacks_accept_loop_and_worker_threads():
    """Both listeners work from the Runner's loop thread and from a worker thread."""
    bridge, connector = bridge_only()
    item = {
        "id": "t1",
        "content": "Go",
        "active_form": "",
        "status": "in_progress",
        "added": 0,
        "since": 0,
        "used": [],
    }
    threads = []

    def publish():
        threads.append(threading.get_ident())
        bridge.on_todos([item], [item])
        bridge.on_reply("Reply.")

    async def from_both() -> None:
        publish()  # on the loop's thread, as a graph hook runs
        await asyncio.to_thread(publish)  # on a worker, as a sync tool or listener would

    bounded(asyncio.run, from_both())

    assert len(set(threads)) == 2
    assert len(lists(connector)) == 2
    assert lists(connector)[0] == [{k: v for k, v in item.items() if k not in PRIVATE}]
    assert texts(connector) == ["t1: Go", "Reply.", "t1: Go", "Reply."]


def test_close_waits_for_a_publish_in_flight_and_nothing_follows_it():
    bridge, connector = bridge_only()
    console = publisher(connector, "/walkie/console")
    publishing, release = threading.Event(), threading.Event()

    def slow(_message):
        publishing.set()
        assert release.wait(WATCHDOG_S)

    console.on_publish = slow
    worker = Background(bridge.on_reply, "In flight.")
    assert publishing.wait(WATCHDOG_S)

    closing = Background(bridge.close)
    # The close has to pass the lock this publish holds, so it cannot have finished.
    assert not closing.done()
    assert not set(connector.node.publishers) & set(connector.node.destroyed)

    release.set()
    worker.result()
    closing.result()
    console.on_publish = None
    bridge.on_reply("After.")
    assert texts(connector) == ["In flight."]
    assert set(connector.node.publishers) <= set(connector.node.destroyed)


# -- What the tools may reach ------------------------------------------------------


def test_unavailable_vision_is_not_an_empty_view():
    robot = FakeRobot()
    ctx = Ctx(robot.cancel, robot, robot, UnavailableVision(robot.cancel))
    for atomic in (find_objects, find_people):
        with pytest.raises(StepError) as raised:
            atomic(ctx)
        assert raised.value.code is ErrorCode.UNAVAILABLE

    script = [ai(call("find_objects", {}, "objects")), ai(call("find_people", {}, "people"))]
    stack = make_stack(*script, then=DONE, robot=robot)
    stack.agent = build_agent(stack.model, ctx, tools=DEFAULT)
    stack.run("What do you see?")
    answers = {m.tool_call_id: m for m in stack.messages if isinstance(m, ToolMessage)}
    assert answers["objects"].status == "error"
    assert answers["objects"].content == "UNAVAILABLE: Live object perception is not connected."
    assert answers["people"].status == "error"
    assert answers["people"].content == "UNAVAILABLE: Live people perception is not connected."


@pytest.mark.parametrize("name", ["update_todos", "wait_for_person"])
def test_no_generic_topics_or_reserved_tool_collision(monkeypatch, name):
    robot = FakeRobot()
    for tool_set in (DEFAULT, GPSR, HRI):
        for tool in as_tools(tool_set, robot.ctx):
            described = f"{tool.name} {' '.join(tool.args)}".lower()
            assert "topic" not in described and "cmd_vel" not in described
            assert "velocity" not in described

    def colliding(ctx, text: str) -> str:
        """A tool that takes the name of a middleware tool.

        Args:
            text: Anything.
        """
        return text

    colliding.__name__ = name

    def never(*_args, **_kwargs):
        raise AssertionError("The graph must not be built.")

    monkeypatch.setattr("walkie_agent.runtime.agent.create_agent", never)
    with pytest.raises(ValueError, match=f"reserved by the middleware: {name}"):
        build_agent(ScriptedModel(then=DONE), robot.ctx, tools=[*DEFAULT, colliding])
