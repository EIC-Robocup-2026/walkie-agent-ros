"""The full Walkie agent (RAI ReActAgent + tools) with a scripted LLM and the fake robot."""

import threading
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from rai.communication.ros2 import ROS2HRIConnector, ROS2HRIMessage
from std_msgs.msg import Empty

from scripted_llm import ScriptedChatModel, tool_call
from walkie_agent.agent import SYSTEM_PROMPT, WalkieAgent, create_walkie_agent


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


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


def test_agent_over_ros_hri_topics(fake_walkie, connector, locations):
    """A human sends a task on /from_human; the agent acts and replies on /to_human."""
    agent_hri = ROS2HRIConnector(node_name="walkie_agent_hri_test")
    human_hri = ROS2HRIConnector(node_name="human_console_test")
    replies: list[str] = []
    human_hri.register_callback("/to_human", lambda msg: replies.append(msg.text))

    llm = ScriptedChatModel(responses=fetch_drink_script())
    agent = create_walkie_agent(
        llm,
        connector,
        locations,
        target_connectors={"/to_human": agent_hri},
        stream_response=False,
    )
    agent.subscribe_source("/from_human", agent_hri)
    agent.run()
    try:
        # Send exactly once: with RAI's default "interrupt_keep_last" behavior a
        # repeated task would interrupt the run already in progress.
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
    # RAI joins queued human messages with newlines before handing them to the LLM.
    assert task_message.content == "Go to the kitchen.\n"
    assert len(llm.received) == len(fetch_drink_script())


def test_reset_topic_clears_memory(connector, locations):
    """The agent remembers earlier tasks until a message arrives on the reset topic."""
    agent_hri = ROS2HRIConnector(node_name="walkie_agent_hri_reset_test")
    human_hri = ROS2HRIConnector(node_name="human_console_reset_test")
    replies: list[str] = []
    human_hri.register_callback("/to_human", lambda msg: replies.append(msg.text))
    reset_pub = human_hri.node.create_publisher(Empty, "/walkie/reset", 10)

    llm = ScriptedChatModel(
        responses=[AIMessage(content="first"), AIMessage(content="second"), AIMessage(content="third")]
    )
    agent = create_walkie_agent(
        llm, connector, locations, target_connectors={"/to_human": agent_hri}, stream_response=False
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
