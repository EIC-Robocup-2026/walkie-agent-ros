"""The full Walkie agent (Walkie's graph on RAI's LangChainAgent + tools) with a scripted
LLM and the fake robot."""

import json
import logging
import random
import re
import threading
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from rai.communication.hri_connector import HRIMessage
from rai.communication.ros2 import ROS2Connector, ROS2HRIConnector, ROS2HRIMessage
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

from conftest import PROJECT_ROOT
from scripted_llm import ScriptedChatModel, tool_call
from walkie_agent.agent import (
    ARM_RULES,
    SYSTEM_PROMPT,
    WalkieAgent,
    _Entry,
    _Seed,
    create_walkie_agent,
)
from walkie_agent.arm.commander import CommanderClient
from walkie_agent.gaze import STOP_HEARD, GazeLoop
from walkie_agent.perception.people_memory import InMemoryPeopleMemory, ModelVector
from walkie_agent.perception.ros_people_memory import RosPeopleMemory
from walkie_agent.perception.vision_feed import VisionFeed
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, hash_vector
from walkie_agent.tasks import TaskTemplate, load_task_templates
from walkie_agent.todo import TodoRejected
from walkie_agent.toolbox import AtomicTool, StepError, Toolbox

# The agent's plain-text replies; speech goes to /to_human instead.
CONSOLE = "/walkie/console"
TASK = "/walkie/task"
TODOS = "/walkie/todos"
# A late subscriber needs transient-local durability to get the latched list.
# Its depth is above 1, so it would get every older message the publisher kept.
LATCHED = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
TOUR = TaskTemplate(
    name="tour",
    keywords=("house tour",),
    task="Show the person the house.",
    todos=["Go to the kitchen", {"content": "Describe the kitchen", "requires_tool": "speak"}],
)
TOUR_REMINDER = (
    "<todos>\n[ ] t1 Go to the kitchen\n[ ] t2 Describe the kitchen  needs: speak\n</todos>"
)


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def operator():
    """The operator's own node, apart from the agent's, for the control topics."""
    conn = ROS2Connector(node_name=f"test_operator_{random.randint(0, 1_000_000)}")
    yield conn
    conn.shutdown()


def listen_to_todos(connector: ROS2Connector) -> list[str]:
    """Subscribes to the todo topic as a late viewer; returns the messages as they arrive."""
    received: list[str] = []
    connector.node.create_subscription(
        String, TODOS, lambda msg: received.append(msg.data), LATCHED
    )
    return received


def fetch_drink_script() -> list[AIMessage]:
    return [
        tool_call("list_locations"),
        tool_call("navigate_to_location", {"location": "kitchen"}),
        tool_call("speak", {"text": "I am at the kitchen."}),
        AIMessage(content="Reached the kitchen and told the guest."),
    ]


def test_agent_runs_tools_in_order(fake_walkie, connector, locations):
    llm = ScriptedChatModel(responses=fetch_drink_script())
    agent = create_walkie_agent(llm, connector, locations, target_connectors={})

    state = {"messages": [HumanMessage(content="Go to the kitchen and say where you are.")]}
    agent.agent.invoke(state)

    assert fake_walkie.pose()[:2] == pytest.approx((3.5, 1.2))
    assert wait_until(lambda: fake_walkie.heard == ["I am at the kitchen."])

    # The LLM saw the system prompt first and each tool's output before its next step.
    final_input = llm.received[-1]
    assert isinstance(final_input[0], SystemMessage)
    assert final_input[0].content == SYSTEM_PROMPT
    tool_outputs = [m.content for m in final_input if isinstance(m, ToolMessage)]
    assert tool_outputs[1:] == ["Arrived at kitchen.", "Said: I am at the kitchen."]
    assert "- kitchen:" in tool_outputs[0]


def test_agent_sees_tool_failure(connector, locations):
    # No fake robot: navigation and speech both fail, and the LLM must be told so.
    llm = ScriptedChatModel(
        responses=[
            tool_call("navigate_to_location", {"location": "kitchen"}),
            AIMessage(content="Navigation is down."),
        ]
    )
    agent = create_walkie_agent(llm, connector, locations, target_connectors={})

    agent.agent.invoke({"messages": [HumanMessage(content="Go to the kitchen.")]})

    last_tool = [m for m in llm.received[-1] if isinstance(m, ToolMessage)][-1]
    assert "Navigation is unavailable" in last_tool.content


def test_the_agent_gets_a_toolbox_with_only_the_vision_and_gaze_atomics(connector, locations):
    """create_walkie_agent builds one Toolbox with a VisionFeed, a GazeLoop and no people memory."""
    agent = create_walkie_agent(
        ScriptedChatModel(responses=[]), connector, locations, target_connectors={}
    )

    assert isinstance(agent.toolbox, Toolbox)
    assert agent.toolbox.ctx.connector is connector
    assert isinstance(agent.toolbox.ctx.backend("vision"), VisionFeed)
    # recognize_person and remember_person need people_memory, which only a
    # people_memory argument adds (main.py always gives one).
    assert not agent.toolbox.ctx.has_backend("people_memory")
    # The arm atomics need the arm backend, which only --arms adds, so a robot
    # run without it never offers those tools, nor checks the arms before a drive.
    assert not agent.toolbox.ctx.has_backend("arm")
    # The gaze loop loads in every run, sim or robot, with or without --arms.
    gaze = agent.toolbox.ctx.backend("gaze")
    assert isinstance(gaze, GazeLoop)
    assert agent.gaze is gaze
    assert [t.name for t in agent.toolbox.tools()] == [
        "find_people",
        "find_seats",
        "look_at",
        "stop_looking",
    ]
    assert [t.name for t in agent.tools.visible() if isinstance(t, AtomicTool)] == [
        "find_people",
        "find_seats",
        "look_at",
        "stop_looking",
    ]
    assert agent.tools.get("navigate_to_location").arm is None
    assert agent.tools.get("move_relative").arm is None
    # Both drives stop the loop before they send a goal.
    assert agent.tools.get("navigate_to_location").gaze is gaze
    assert agent.tools.get("move_relative").gaze is gaze
    assert "`look_at`" in SYSTEM_PROMPT and "`stop_looking`" in SYSTEM_PROMPT
    # The arm clause is only for runs with arms.
    assert "with an arm out" not in SYSTEM_PROMPT
    assert "Never call `look_at` with an arm out" in ARM_RULES


def test_stopping_the_agent_stops_the_gaze_loop_and_shuts_it_down(connector, locations):
    agent = create_walkie_agent(
        ScriptedChatModel(responses=[]), connector, locations, target_connectors={}
    )
    gaze = agent.gaze
    gaze.hold_heading("seat_77aa.2", 0.5, cancel=threading.Event())
    agent.run()

    agent.stop()

    status = gaze.status()
    # The interrupt listener stopped the target; shutdown() then ended the thread.
    assert (status.running, status.stop_reason) == (False, "interrupt")
    assert gaze._thread is not None and not gaze._thread.is_alive()
    with pytest.raises(StepError) as error:
        gaze.hold_heading("seat_77aa.2", 0.5, cancel=threading.Event())
    assert error.value.code == "UNAVAILABLE"


def test_a_stop_while_the_agent_is_idle_stops_the_gaze_loop(connector, locations):
    llm = ScriptedChatModel(responses=[AIMessage("There is nothing to stop.")])
    agent = create_walkie_agent(llm, connector, locations, target_connectors={})
    gaze = agent.gaze
    agent.run()
    try:
        assert wait_until(agent.ready)
        # As after a look_at at a seat that ended its task.
        gaze.hold_heading("seat_77aa.2", 0.5, cancel=threading.Event())

        agent(HRIMessage(text="stop", message_author="human"))

        # Stopped before agent() returned, though there was no run to interrupt.
        status = gaze.status()
        assert (status.running, status.stop_reason) == (False, STOP_HEARD)
        stop_looking = next(t for t in agent.toolbox.tools() if t.name == "stop_looking")
        assert re.fullmatch(
            r"Not looking at anyone\. The last target, seat_77aa\.2, was stopped \d+ s ago "
            r'because someone said "stop" or "never mind"\.',
            stop_looking.call().summary,
        )
        # The stop still reaches the LLM as an ordinary message.
        assert wait_until(lambda: llm.received and agent.ready())
        assert "stop\n" in [m.content for m in llm.received[0]]
    finally:
        agent.stop()


def test_the_mock_people_memory_loads_the_people_atomics_and_clears_on_a_new_task(
    connector, locations
):
    agent = create_walkie_agent(
        ScriptedChatModel(responses=[]),
        connector,
        locations,
        target_connectors={},
        people_memory="mock",
    )
    memory = agent.toolbox.ctx.backend("people_memory")
    memory.remember("Sam", {}, [ModelVector(FAKE_FACE_MODEL, hash_vector("sam", 512))], [])

    assert isinstance(memory, InMemoryPeopleMemory)
    # No arm atomic: the mock does not imply --arms.
    assert [t.name for t in agent.toolbox.tools()] == [
        "find_people",
        "find_seats",
        "look_at",
        "recognize_person",
        "remember_person",
        "stop_looking",
    ]
    assert len(memory) == 1
    agent.toolbox.new_task()
    assert len(memory) == 0


def test_the_database_people_memory_loads_the_people_atomics_and_clears_on_a_new_task(
    connector, locations, monkeypatch
):
    # No database node runs here; test_ros_people_memory.py covers the calls.
    cleared = []
    monkeypatch.setattr(RosPeopleMemory, "clear", lambda memory: cleared.append(memory))
    agent = create_walkie_agent(
        ScriptedChatModel(responses=[]),
        connector,
        locations,
        target_connectors={},
        people_memory="database",
    )
    memory = agent.toolbox.ctx.backend("people_memory")

    assert isinstance(memory, RosPeopleMemory)
    assert [t.name for t in agent.toolbox.tools()] == [
        "find_people",
        "find_seats",
        "look_at",
        "recognize_person",
        "remember_person",
        "stop_looking",
    ]
    agent.toolbox.new_task()
    assert cleared == [memory]
    # By default the first start forgets the open session.
    assert not (memory._resume_pending or memory._adopt_next)


def test_resume_people_session_reaches_the_database_people_memory(connector, locations):
    agent = create_walkie_agent(
        ScriptedChatModel(responses=[]),
        connector,
        locations,
        target_connectors={},
        people_memory="database",
        resume_people_session=True,
    )
    memory = agent.toolbox.ctx.backend("people_memory")

    # The first task after the restart keeps the open session.
    assert memory._resume_pending and memory._adopt_next


def test_arms_loads_the_arm_atomics_the_arm_rules_and_the_drive_check(
    fake_walkie, connector, locations, monkeypatch
):
    """--arms adds a CommanderClient as the arm backend, shared by the atomics and the drives.

    No commander runs here, so the drive goes ahead and says the arms were
    not checked (ADR-0025).
    """
    # CommanderClient reads config/arm_poses.toml from the working
    # directory, as main.py's other config files are read.
    monkeypatch.chdir(PROJECT_ROOT)
    llm = ScriptedChatModel(
        responses=[
            tool_call("navigate_to_location", {"location": "kitchen"}),
            AIMessage(content="At the kitchen."),
        ]
    )
    agent = create_walkie_agent(llm, connector, locations, target_connectors={}, arms=True)
    commander = agent.toolbox.ctx.backend("arm")
    try:
        assert isinstance(commander, CommanderClient)
        assert [t.name for t in agent.toolbox.tools()] == [
            "find_people",
            "find_seats",
            "get_arm_state",
            "list_arm_poses",
            "look_at",
            "move_arm_to_pose",
            "set_gripper",
            "stop_looking",
            "wait_for_handover",
        ]
        assert agent.tools.get("navigate_to_location").arm is commander
        assert agent.tools.get("move_relative").arm is commander
        # The gaze loop reads the same client's last state, and look_at
        # checks the arms through the arm backend.
        assert agent.gaze._arm is commander

        agent.agent.invoke({"messages": [HumanMessage(content="Go to the kitchen.")]})

        final_input = llm.received[-1]
        assert final_input[0].content == SYSTEM_PROMPT + ARM_RULES
        tool_outputs = [m.content for m in final_input if isinstance(m, ToolMessage)]
        assert tool_outputs == [
            "Arrived at kitchen. Arm state unknown: the commander was not found."
        ]
    finally:
        commander.destroy()


def test_todo_tools_and_the_graph_share_one_list(connector, locations):
    """create_walkie_agent gives update_todos, wait_for_person and the graph the same list."""
    add_step = {"op": "add", "content": "Say hello", "status": "completed"}
    llm = ScriptedChatModel(
        responses=[tool_call("update_todos", {"ops": [add_step]}), AIMessage(content="Hello.")]
    )
    agent = create_walkie_agent(llm, connector, locations, target_connectors={})

    assert {"update_todos", "wait_for_person"} <= set(agent.tools.names())
    assert agent.tools.get("wait_for_person").speak_tool is agent.tools.get("speak")

    agent.agent.invoke({"messages": [HumanMessage(content="Say hello.")]})

    items = agent.todos.snapshot()
    assert [(i.content, i.status) for i in items] == [("Say hello", "completed")]
    # The graph reads the list that the tool wrote: the next call carries it.
    assert llm.reminders == ["", "<todos>\n[x] t1 Say hello\n</todos>"]


def test_agent_over_ros_hri_topics(fake_walkie, connector, locations):
    """A human sends a task on /from_human; the agent acts and replies on the console topic."""
    agent_hri = ROS2HRIConnector(node_name="walkie_agent_hri_test")
    human_hri = ROS2HRIConnector(node_name="human_console_test")
    replies: list[str] = []
    human_hri.register_callback(CONSOLE, lambda msg: replies.append(msg.text))

    llm = ScriptedChatModel(responses=fetch_drink_script())
    agent = create_walkie_agent(
        llm,
        connector,
        locations,
        target_connectors={CONSOLE: agent_hri},
        stream_response=False,
    )
    agent.subscribe_source("/from_human", agent_hri)
    agent.run()
    try:
        # Send exactly once: messages queue, so a repeated task would wait for
        # the first run to end and then run a second time.
        assert wait_until(lambda: human_hri.node.count_subscribers("/from_human") > 0)
        task = ROS2HRIMessage(text="Go to the kitchen.", message_author="human")
        human_hri.send_message(task, "/from_human")
        assert wait_until(lambda: "Reached the kitchen and told the guest." in replies)
    finally:
        agent.stop()
        agent_hri.shutdown()
        human_hri.shutdown()

    assert fake_walkie.heard == ["I am at the kitchen."]
    task_message = llm.received[0][1]
    assert isinstance(task_message, HumanMessage)
    # The agent takes one queued message per task and ends its text with a
    # newline, as RAI's own message handling does.
    assert task_message.content == "Go to the kitchen.\n"
    assert len(llm.received) == len(fetch_drink_script())


def test_reset_topic_clears_memory(connector, locations):
    """The agent remembers earlier tasks until a message arrives on the reset topic."""
    agent_hri = ROS2HRIConnector(node_name="walkie_agent_hri_reset_test")
    human_hri = ROS2HRIConnector(node_name="human_console_reset_test")
    replies: list[str] = []
    human_hri.register_callback(CONSOLE, lambda msg: replies.append(msg.text))
    reset_pub = human_hri.node.create_publisher(Empty, "/walkie/reset", 10)

    llm = ScriptedChatModel(
        responses=[AIMessage(content="first"), AIMessage(content="second"), AIMessage(content="third")]
    )
    agent = create_walkie_agent(
        llm, connector, locations, target_connectors={CONSOLE: agent_hri}, stream_response=False
    )
    agent.subscribe_source("/from_human", agent_hri)
    agent.subscribe_reset(connector, "/walkie/reset")
    first_session = agent.session_id

    def ask(text: str, reply: str) -> None:
        human_hri.send_message(ROS2HRIMessage(text=text, message_author="human"), "/from_human")
        assert wait_until(lambda: reply in replies)

    agent.run()
    try:
        assert wait_until(lambda: human_hri.node.count_subscribers("/from_human") > 0)
        assert wait_until(lambda: reset_pub.get_subscription_count() > 0)

        ask("Task one.", "first")
        ask("Task two.", "second")
        # Without a reset, the second task is sent along with the whole first one.
        assert [m.content for m in llm.received[1][1:]] == [
            "Task one.\n",
            "first",
            "Task two.\n",
        ]

        reset_pub.publish(Empty())
        assert wait_until(lambda: agent.session_id != first_session)
        assert agent.state["messages"] == []

        ask("Task three.", "third")
    finally:
        agent.stop()
        agent_hri.shutdown()
        human_hri.shutdown()

    third_input = llm.received[2]
    assert [type(m) for m in third_input] == [SystemMessage, HumanMessage]
    assert third_input[1].content == "Task three.\n"
    assert agent.agent.config["metadata"]["langfuse_session_id"] == agent.session_id


def test_reset_stops_task_after_current_tool(connector):
    """A reset during a tool call lets the tool finish, then ends the task."""
    tool_started = threading.Event()
    release_tool = threading.Event()

    @tool
    def slow_action() -> str:
        """Takes a while, like driving somewhere."""
        tool_started.set()
        release_tool.wait(timeout=5)
        return "done"

    llm = ScriptedChatModel(
        responses=[tool_call("slow_action"), AIMessage(content="Should never be requested.")]
    )
    agent = WalkieAgent(target_connectors={}, llm=llm, tools=[slow_action], stream_response=False)
    agent.run()
    try:
        agent(ROS2HRIMessage(text="Do the slow thing.", message_author="human"))
        assert tool_started.wait(timeout=5)
        agent.reset()
        release_tool.set()
        assert wait_until(lambda: agent.state["messages"] == [] and agent.ready())
        # Give a wrongly continued run time to reach the LLM again.
        time.sleep(0.3)
    finally:
        agent.stop()

    assert len(llm.received) == 1


# -- One entry per queued message ------------------------------------------------


def human(text: str) -> HRIMessage:
    return HRIMessage(text=text, message_author="human")


def queued(agent: WalkieAgent) -> list[tuple[str, list | None]]:
    """The queue as (text, seed items) pairs, once its entries are checked against it."""
    with agent._buffer_lock:
        messages = list(agent._received_messages)
        keys = list(agent._entries)
        entries = list(agent._entries.values())
    # One entry per queued message, in queue order, each holding its own message.
    assert keys == [id(msg) for msg in messages]
    assert all(entry.msg is msg for entry, msg in zip(entries, messages, strict=True))
    return [(entry.msg.text, None if entry.seed is None else entry.seed.items) for entry in entries]


def slow_agent(
    responses: list[AIMessage],
) -> tuple[WalkieAgent, ScriptedChatModel, threading.Event, threading.Event]:
    """An agent whose slow_action tool runs until the test sets the release event."""
    started, release = threading.Event(), threading.Event()

    @tool
    def slow_action() -> str:
        """Takes a while, like driving somewhere."""
        started.set()
        release.wait(timeout=5)
        return "done"

    llm = ScriptedChatModel(responses=responses)
    agent = WalkieAgent(target_connectors={}, llm=llm, tools=[slow_action], stream_response=False)
    return agent, llm, started, release


def test_each_queued_message_has_one_entry_in_queue_order():
    agent = WalkieAgent(target_connectors={}, llm=ScriptedChatModel(responses=[]))

    agent(human("Go to the kitchen."))
    agent.submit("Then the bedroom.", todos=["Go to the bedroom"], source="operator")
    # A line that is not speech is never queued, so it has no entry.
    agent(human("[BLANK_AUDIO]"))
    agent.submit("Start over.", todos=[])

    assert queued(agent) == [
        ("Go to the kitchen.", None),
        ("Then the bedroom.", ["Go to the bedroom"]),
        ("Start over.", []),
    ]
    entries = list(agent._entries.values())
    assert [entry.seed.source for entry in entries[1:]] == ["operator", "seed"]
    assert [(entry.status, entry.annotations) for entry in entries] == [(None, [])] * 3


def test_the_same_message_queued_twice_gets_two_entries():
    agent = WalkieAgent(target_connectors={}, llm=ScriptedChatModel(responses=[]))
    msg = human("Say hi.")

    agent(msg)
    agent(msg)

    # The second is queued as a copy, so that the ids stay unique.
    assert queued(agent) == [("Say hi.", None), ("Say hi.", None)]
    assert agent._received_messages[0] is msg


def test_a_full_queue_drops_the_oldest_message_with_its_entry():
    agent = WalkieAgent(target_connectors={}, llm=ScriptedChatModel(responses=[]))
    agent.max_size = 2

    agent.submit("Go to the kitchen.", todos=["Go to the kitchen"])
    agent(human("Then the bedroom."))
    agent(human("Then the garage."))

    assert queued(agent) == [("Then the bedroom.", None), ("Then the garage.", None)]


def test_reset_drops_every_entry():
    agent = WalkieAgent(target_connectors={}, llm=ScriptedChatModel(responses=[]))
    agent(human("Go to the kitchen."))
    agent.submit("Then the bedroom.", todos=["Go to the bedroom"])

    agent.reset()

    assert queued(agent) == []
    assert agent._entries == {}


def test_a_stop_during_a_task_keeps_only_its_own_entry():
    agent, llm, started, release = slow_agent(
        [tool_call("slow_action"), AIMessage(content="Stopped.")]
    )
    agent.run()
    try:
        agent(human("Drive to the kitchen."))
        assert started.wait(timeout=5)
        agent.submit("Then the bedroom.", todos=["Go to the bedroom"])
        agent(human("Say hi."))
        assert queued(agent) == [("Then the bedroom.", ["Go to the bedroom"]), ("Say hi.", None)]

        agent(human("Stop."))

        assert queued(agent) == [("Stop.", None)]
        release.set()
        # The stop runs next, and its entry leaves with it.
        assert wait_until(lambda: len(llm.received) == 2 and agent.ready())
    finally:
        release.set()
        agent.stop()

    assert queued(agent) == []
    assert llm.received[1][-1].content == "Stop.\n"


def test_a_task_takes_its_message_with_its_entry():
    agent, llm, started, release = slow_agent(
        [tool_call("slow_action"), AIMessage(content="At the kitchen.")]
        # The seeded step stays open, so the gate sends the LLM back twice.
        + [AIMessage(content=f"On it {n}.") for n in range(3)]
    )
    agent.run()
    try:
        agent(human("Drive to the kitchen."))
        assert started.wait(timeout=5)
        # The running task took its message, and the entry went with it.
        assert queued(agent) == []
        agent.submit("Say hi.", todos=["Say hi"])
        assert queued(agent) == [("Say hi.", ["Say hi"])]

        release.set()
        assert wait_until(lambda: len(llm.received) == 5 and agent.ready())
    finally:
        release.set()
        agent.stop()

    assert queued(agent) == []
    # The seed came off the entry when its own task started.
    assert llm.reminders[2] == "<todos>\n[ ] t1 Say hi\n</todos>"


@pytest.mark.parametrize(
    ("text", "seed", "status", "kept"),
    [
        ("Say hi.", None, None, False),
        ("Walkie, stop.", None, None, False),
        # An empty seed is still a seed: it empties the list when its task starts.
        ("Say hi.", _Seed(items=[], source="seed"), None, True),
        ("Say hi.", _Seed(items=["Say hi"], source="operator"), None, True),
        ("Say hi.", None, object(), True),
        ("Stop.", None, None, True),
        ("never mind", None, None, True),
    ],
    ids=["utterance", "not a stop", "empty seed", "seed", "status", "stop", "never mind"],
)
def test_an_overheard_drop_keeps_seeds_statuses_and_stops(text, seed, status, kept):
    entry = _Entry(msg=human(text), seed=seed, received_mono=0.0, status=status)

    assert entry.kept_on_overheard_drop() is kept


def test_a_message_is_stamped_on_the_monotonic_clock_when_it_arrives():
    agent = WalkieAgent(target_connectors={}, llm=ScriptedChatModel(responses=[]))

    before = time.monotonic()
    agent(human("Go to the kitchen."))
    agent.submit("Then the bedroom.", todos=["Go to the bedroom"])
    after = time.monotonic()

    stamps = [entry.received_mono for entry in agent._entries.values()]
    assert before <= stamps[0] <= stamps[1] <= after


def test_a_caller_can_pass_the_receive_stamp_in():
    agent = WalkieAgent(target_connectors={}, llm=ScriptedChatModel(responses=[]))
    arrived = time.monotonic() - 2.0

    agent._enqueue(human("Go to the kitchen."), None, received_mono=arrived)

    assert [entry.received_mono for entry in agent._entries.values()] == [arrived]


# -- The operator's task topic and the todo topic (ADR-0041) ---------------------


@pytest.mark.parametrize(
    ("message", "text"),
    [
        (
            {"task": "Show me around.", "todos": TOUR.todos},
            "Show me around.\n",
        ),
        (
            {"task": "Show me around.", "template": "tour"},
            "Show me around.\n\nShow the person the house.\n",
        ),
    ],
    ids=["todos", "template"],
)
def test_task_topic_starts_a_seeded_task(connector, operator, locations, message, text):
    # The steps stay open, so the gate sends the LLM back twice.
    llm = ScriptedChatModel(responses=[AIMessage(content=f"On it {n}.") for n in range(3)])
    agent = create_walkie_agent(
        llm, connector, locations, target_connectors={}, templates={"tour": TOUR}
    )
    agent.subscribe_task(connector, TASK)
    publisher = operator.node.create_publisher(String, TASK, 10)

    agent.run()
    try:
        assert wait_until(lambda: publisher.get_subscription_count() > 0)
        publisher.publish(String(data=json.dumps(message)))
        assert wait_until(lambda: len(llm.received) == 3 and agent.ready())
    finally:
        agent.stop()

    assert llm.received[0][1].content == text
    assert llm.reminders[0] == TOUR_REMINDER
    assert [(i.source, i.locked) for i in agent.todos.snapshot()] == [("operator", True)] * 2


def test_malformed_task_messages_are_ignored(connector, operator, locations, caplog):
    llm = ScriptedChatModel(responses=[AIMessage(content="Hello.")])
    agent = create_walkie_agent(
        llm, connector, locations, target_connectors={}, templates={"tour": TOUR}
    )
    agent.subscribe_task(connector, TASK)
    publisher = operator.node.create_publisher(String, TASK, 10)
    malformed = [
        "Go to the kitchen.",  # not JSON
        json.dumps({"todos": ["Say hello"]}),  # no task
        json.dumps({"task": "Show me around.", "template": "garage"}),  # unknown template
        json.dumps({"task": "Say hello.", "todos": [{"content": ""}]}),  # a seed the list rejects
        # A step whose tool is not loaded: set_gripper needs --arms.
        json.dumps(
            {"task": "Take it.", "todos": [{"content": "Take", "requires_tool": "set_gripper"}]}
        ),
    ]

    agent.run()
    try:
        assert wait_until(lambda: publisher.get_subscription_count() > 0)
        with caplog.at_level(logging.WARNING):
            for data in malformed:
                publisher.publish(String(data=data))
            # The subscriber outlives them all and still takes a good message.
            publisher.publish(String(data=json.dumps({"task": "Say hello.", "todos": []})))
            assert wait_until(lambda: len(llm.received) == 1 and agent.ready())
    finally:
        agent.stop()

    assert llm.received[0][1].content == "Say hello.\n"
    warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and TASK in r.getMessage()
    ]
    assert len(warnings) == len(malformed)


@pytest.mark.parametrize("arms", [False, True], ids=["without arms", "with arms"])
def test_the_hri_template_starts_only_with_the_arms(
    connector, locations, caplog, monkeypatch, arms
):
    """The shipped HRI template's bag step needs set_gripper, which loads only with --arms."""
    # CommanderClient reads config/arm_poses.toml from the working directory.
    monkeypatch.chdir(PROJECT_ROOT)
    templates = load_task_templates(PROJECT_ROOT / "config" / "task_templates.toml")
    llm = ScriptedChatModel(responses=[])
    with caplog.at_level(logging.WARNING):
        agent = create_walkie_agent(
            llm,
            connector,
            locations,
            target_connectors={},
            templates=templates,
            people_memory="mock",
            arms=arms,
        )
    try:
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        reason = "t4 (Take the second guest's bag) needs set_gripper, which loads only with --arms."
        seed = templates["hri"].todos
        if arms:
            assert not [w for w in warnings if "cannot start" in w]
            agent.submit("Receive the guests.", seed, source="operator")
        else:
            assert f"Task template 'hri' cannot start in this run: {reason}" in warnings
            with pytest.raises(TodoRejected) as e:
                agent.submit("Receive the guests.", seed, source="operator")
            assert str(e.value) == f"The task cannot start in this run: {reason}"
    finally:
        if arms:
            agent.toolbox.ctx.backend("arm").destroy()


def test_todos_topic_is_latched_and_cleared_on_reset(connector, operator, locations):
    complete = {"op": "update", "id": "t1", "status": "completed"}
    llm = ScriptedChatModel(
        responses=[tool_call("update_todos", {"ops": [complete]}), AIMessage(content="Hello.")]
    )
    agent = create_walkie_agent(llm, connector, locations, target_connectors={})
    agent.publish_todos(connector, TODOS)
    agent.subscribe_reset(connector, "/walkie/reset")
    reset_pub = operator.node.create_publisher(Empty, "/walkie/reset", 10)

    # A viewer that starts before any task gets the empty list.
    early = listen_to_todos(operator)
    assert wait_until(lambda: early == ["[]"])

    agent.run()
    try:
        agent.submit("Say hello.", todos=["Say hello"])
        assert wait_until(lambda: len(llm.received) == 2 and agent.ready())
        current = agent.todos.to_json()
        assert [i["status"] for i in json.loads(current)] == ["completed"]
        assert wait_until(lambda: early[-1] == current)

        # A viewer that starts late gets only the current list.
        late = listen_to_todos(operator)
        assert wait_until(lambda: late)
        time.sleep(0.3)
        assert late == [current]

        assert wait_until(lambda: reset_pub.get_subscription_count() > 0)
        reset_pub.publish(Empty())
        assert wait_until(lambda: early[-1] == "[]" and late[-1] == "[]")
    finally:
        agent.stop()

    assert agent.todos.is_empty


def test_todo_status_changes_go_to_the_console(fake_walkie, connector, locations):
    agent_hri = ROS2HRIConnector(node_name="walkie_agent_hri_status_test")
    human_hri = ROS2HRIConnector(node_name="human_console_status_test")
    lines: list[str] = []
    # RAI's own QoS matching would give the subscription a depth of 1, and one
    # change can send several lines at once. ros2 topic echo keeps 5.
    human_hri.register_callback(
        CONSOLE,
        lambda msg: lines.append(msg.text),
        qos_profile=QoSProfile(depth=50),
        auto_qos_matching=False,
    )

    def update(item_id: str, status: str, note: str = "") -> dict:
        return {"op": "update", "id": item_id, "status": status, "note": note}

    llm = ScriptedChatModel(
        responses=[
            tool_call("update_todos", {"ops": [update("t1", "in_progress")]}, "c1"),
            tool_call(
                "update_todos", {"ops": [update("t1", "completed"), update("t2", "in_progress")]}, "c2"
            ),
            tool_call("wait_for_person", {"question": "Which mug?"}, "c3"),
            # The answer, "The red one.", puts t2 back in progress.
            tool_call(
                "update_todos",
                {
                    "ops": [
                        update("t2", "failed", "There is no red mug."),
                        update("t3", "cancelled", "There is nothing to bring."),
                    ]
                },
                "c4",
            ),
            AIMessage(content="I could not find a red mug."),
        ]
    )
    agent = create_walkie_agent(
        llm, connector, locations, target_connectors={CONSOLE: agent_hri}, stream_response=False
    )

    # RAI creates the console publisher on the first send. Until discovery has
    # matched it to the test's subscriber, a message can be lost, so the test
    # waits until one gets through before the task starts.
    def console_works() -> bool:
        agent_hri.send_message(ROS2HRIMessage(text="ready", message_author="ai"), CONSOLE)
        return wait_until(lambda: "ready" in lines, timeout=0.2)

    agent.run()
    try:
        assert wait_until(console_works)
        agent.submit(
            "Bring me the red mug.",
            todos=[
                {"content": "Go to the kitchen", "active_form": "Going to the kitchen"},
                "Pick up the mug",
                "Bring it back",
            ],
        )
        assert wait_until(lambda: len(llm.received) == 3 and agent.ready())
        agent.submit("The red one.")
        assert wait_until(lambda: "I could not find a red mug." in lines)
    finally:
        agent.stop()
        agent_hri.shutdown()
        human_hri.shutdown()

    # The seed adds no line: pending steps are not news. Each tool-call message
    # also sends its empty text, as RAI does for every LLM answer.
    assert [line for line in lines if line not in ("", "ready")] == [
        "t1: Going to the kitchen",
        "t1 completed: Go to the kitchen",
        "t2: Pick up the mug",
        "t2 waiting: Pick up the mug",
        "t2: Pick up the mug",
        "t2 failed: Pick up the mug",
        "t3 cancelled: Bring it back",
        "I could not find a red mug.",
    ]
    # The lines went to the console only, never to the speaker.
    assert fake_walkie.heard == ["Which mug?"]
