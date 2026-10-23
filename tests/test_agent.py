"""The full Walkie agent (Walkie's graph on RAI's LangChainAgent + tools) with a scripted
LLM and the fake robot."""

import json
import logging
import random
import threading
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from rai.communication.ros2 import ROS2Connector, ROS2HRIConnector, ROS2HRIMessage
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

from conftest import PROJECT_ROOT
from scripted_llm import ScriptedChatModel, tool_call
from walkie_agent.agent import ARM_RULES, SYSTEM_PROMPT, WalkieAgent, create_walkie_agent
from walkie_agent.arm.commander import CommanderClient
from walkie_agent.perception.people_memory import InMemoryPeopleMemory, ModelVector
from walkie_agent.perception.vision_feed import VisionFeed
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, hash_vector
from walkie_agent.tasks import TaskTemplate
from walkie_agent.toolbox import AtomicTool, Toolbox

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


def test_the_agent_gets_a_toolbox_with_only_the_vision_atomics(connector, locations):
    """create_walkie_agent builds one Toolbox with a VisionFeed and no people memory."""
    agent = create_walkie_agent(
        ScriptedChatModel(responses=[]), connector, locations, target_connectors={}
    )

    assert isinstance(agent.toolbox, Toolbox)
    assert agent.toolbox.ctx.connector is connector
    assert isinstance(agent.toolbox.ctx.backend("vision"), VisionFeed)
    assert not agent.toolbox.ctx.has_backend("people_memory")
    # recognize_person and remember_person need people_memory, and the arm
    # atomics need the arm backend, which only --arms adds, so a robot run
    # without them never offers those tools, nor checks the arms before a drive.
    assert not agent.toolbox.ctx.has_backend("arm")
    assert [t.name for t in agent.toolbox.tools()] == ["find_people", "find_seats"]
    assert [t.name for t in agent.tools.visible() if isinstance(t, AtomicTool)] == [
        "find_people",
        "find_seats",
    ]
    assert agent.tools.get("navigate_to_location").arm is None
    assert agent.tools.get("move_relative").arm is None


def test_the_mock_people_memory_loads_the_people_atomics_and_clears_on_a_new_task(
    connector, locations
):
    agent = create_walkie_agent(
        ScriptedChatModel(responses=[]),
        connector,
        locations,
        target_connectors={},
        mock_people_memory=True,
    )
    memory = agent.toolbox.ctx.backend("people_memory")
    memory.remember("Sam", {}, [ModelVector(FAKE_FACE_MODEL, hash_vector("sam", 512))], [])

    assert isinstance(memory, InMemoryPeopleMemory)
    # No arm atomic: the mock does not imply --arms.
    assert [t.name for t in agent.toolbox.tools()] == [
        "find_people",
        "find_seats",
        "recognize_person",
        "remember_person",
    ]
    assert len(memory) == 1
    agent.toolbox.new_task()
    assert len(memory) == 0


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
            "move_arm_to_pose",
            "set_gripper",
        ]
        assert agent.tools.get("navigate_to_location").arm is commander
        assert agent.tools.get("move_relative").arm is commander

        agent.agent.invoke({"messages": [HumanMessage(content="Go to the kitchen.")]})

        final_input = llm.received[-1]
        assert final_input[0].content == SYSTEM_PROMPT + ARM_RULES
        tool_outputs = [m.content for m in final_input if isinstance(m, ToolMessage)]
        assert tool_outputs == [
            "Arrived at kitchen. Arm state unknown: the commander did not answer."
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
