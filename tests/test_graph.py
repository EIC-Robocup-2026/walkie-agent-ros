"""Walkie's own graph (walkie_agent.graph) and the agent that runs it (ADR-0038, ADR-0041).

None of these tests need ROS: the agent gets plain tools, no target
connectors and a scripted LLM. Tests that run the agent thread wait on events
and conditions with timeouts, never on long sleeps.
"""

import logging
import threading
import time
from collections.abc import Iterable

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException, tool
from rai.communication.hri_connector import HRIMessage
from rai.messages import HumanMultimodalMessage

from scripted_llm import ScriptedChatModel, tool_call
from walkie_agent.agent import RECURSION_LIMIT, WalkieAgent, is_stop_utterance
from walkie_agent.graph import (
    AFTER_QUESTION,
    INTERRUPTED,
    MAX_BOUND_TOOLS,
    UNREADABLE_ARGUMENTS,
    ToolSet,
    repair_history,
)
from walkie_agent.tasks import TaskTemplate
from walkie_agent.todo import TodoList, TodoRejected
from walkie_agent.tools.todos import UpdateTodosTool, WaitForPersonTool

GATE_T1_T2 = "[todo] Open items: t1, t2. Finish them, or mark them failed or cancelled with a note."
TOUR = TaskTemplate(
    name="tour",
    keywords=("house tour",),
    task="Show the person the house.",
    todos=["Go to the kitchen", "Come back"],
)


def wait_until(predicate, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def settle(agent: WalkieAgent, llm: ScriptedChatModel, calls: int) -> bool:
    """Waits until the LLM has had `calls` calls and the agent is idle again."""
    return wait_until(lambda: len(llm.received) >= calls and agent.ready())


def make_agent(
    responses: list[AIMessage],
    tools: Iterable[BaseTool] = (),
    todos: TodoList | None = None,
    templates: dict[str, TaskTemplate] | None = None,
) -> tuple[WalkieAgent, ScriptedChatModel]:
    llm = ScriptedChatModel(responses=list(responses))
    agent = WalkieAgent(
        target_connectors={},
        llm=llm,
        tools=list(tools),
        stream_response=False,
        todos=todos,
        templates=templates,
    )
    return agent, llm


@pytest.fixture
def start():
    """Starts an agent's thread, and stops it at the end of the test."""
    started: list[WalkieAgent] = []

    def _start(agent: WalkieAgent) -> WalkieAgent:
        agent.run()
        started.append(agent)
        return agent

    yield _start
    for agent in started:
        agent.stop()


def human(text: str) -> HRIMessage:
    return HRIMessage(text=text, message_author="human")


def calls(*name_and_ids: tuple[str, str]) -> AIMessage:
    """An AI message with several tool calls, each without arguments."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {}, "id": call_id} for name, call_id in name_and_ids],
    )


def unreadable(name: str, call_id: str) -> dict:
    """A tool call whose arguments are not valid JSON, as LangChain keeps it in
    AIMessage.invalid_tool_calls."""
    return {"name": name, "args": '{"to": kitchen', "id": call_id, "error": None}


def update(item_id: str, **fields: str) -> dict:
    return {"op": "update", "id": item_id, **fields}


def add(content: str, **fields: str) -> dict:
    return {"op": "add", "content": content, **fields}


def todo_call(ops: list[dict], call_id: str) -> AIMessage:
    return tool_call("update_todos", {"ops": ops}, call_id)


def run_graph(agent: WalkieAgent, text: str = "Do it.") -> list[BaseMessage]:
    """Runs one task through the graph on this thread and returns the history."""
    state = {"messages": [HumanMessage(content=text)]}
    agent.agent.invoke(state)
    return state["messages"]


def result_of(messages: list[BaseMessage], call_id: str) -> ToolMessage:
    return next(m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == call_id)


def contents(messages: list[BaseMessage]) -> list:
    return [m.content for m in messages]


def make_tool(name: str, result: str = "ok") -> BaseTool:
    return StructuredTool.from_function(
        func=lambda: result, name=name, description=f"The {name} tool."
    )


class Blocker:
    """A tool that runs until the test releases it, like a long drive."""

    def __init__(self, name: str = "slow_action"):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

        def run() -> str:
            self.calls += 1
            self.started.set()
            self.release.wait(timeout=5)
            return "done"

        self.tool = StructuredTool.from_function(
            func=run, name=name, description="Takes a while, like driving somewhere."
        )


class ContendedLock:
    """A lock that records when a thread had to wait for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self) -> "ContendedLock":
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()
        return self

    def __exit__(self, *exc_info) -> None:
        self._lock.release()


class HookOnFirstClear(threading.Event):
    """An event that runs `hook` the first time it is cleared, just before the clear."""

    def __init__(self, hook) -> None:
        super().__init__()
        self._hook = hook

    def clear(self) -> None:
        hook, self._hook = self._hook, None
        if hook is not None:
            hook()
        super().clear()


class PauseOnce(threading.Event):
    """An event whose is_set() holds the calling thread once, after pause_next()."""

    def __init__(self) -> None:
        super().__init__()
        self._armed = False
        self.paused = threading.Event()
        self.resume = threading.Event()

    def pause_next(self) -> None:
        self._armed = True

    def is_set(self) -> bool:
        if self._armed:
            self._armed = False
            self.paused.set()
            self.resume.wait(timeout=5)
        return super().is_set()


def fake_speak(reply: str = ""):
    """A speak tool that records what it was asked to say."""
    said: list[str] = []

    @tool
    def speak(text: str) -> str:
        """Say something out loud."""
        said.append(text)
        return reply or f"Said: {text}"

    return speak, said


def todo_tools(todos: TodoList, speak: BaseTool) -> list[BaseTool]:
    return [
        speak,
        UpdateTodosTool(todos=todos),
        WaitForPersonTool(todos=todos, speak_tool=speak),
    ]


def seeded(items: list, in_progress: str = "") -> TodoList:
    todos = TodoList()
    todos.seed(items)
    if in_progress:
        todos.apply([update(in_progress, status="in_progress")])
    return todos


# -- ToolSet ---------------------------------------------------------------------


def test_toolset_keeps_order_and_replaces_by_name():
    a, b = make_tool("a"), make_tool("b")
    toolset = ToolSet([a, b])
    new_a = make_tool("a", "new")

    toolset.add(new_a)

    assert toolset.names() == ["a", "b"]
    assert toolset.visible() == [new_a, b]
    assert toolset.get("a") is new_a
    assert toolset.remove("a") is True
    assert toolset.remove("a") is False
    assert toolset.get("a") is None
    assert toolset.names() == ["b"]


# -- History repair --------------------------------------------------------------


def test_repair_answers_every_unanswered_call():
    messages = [
        HumanMessage("Go."),
        calls(("drive", "c1"), ("speak", "c2")),
        HumanMessage("stop\n"),
    ]

    assert repair_history(messages) == 2

    assert [type(m) for m in messages] == [
        HumanMessage, AIMessage, ToolMessage, ToolMessage, HumanMessage
    ]
    first, second = messages[2], messages[3]
    assert (first.tool_call_id, first.name, first.content, first.status) == (
        "c1", "drive", INTERRUPTED, "error"
    )
    assert (second.tool_call_id, second.name, second.status) == ("c2", "speak", "error")


def test_repair_adds_missing_answers_after_the_existing_ones():
    messages = [
        calls(("look", "c1"), ("drive", "c2"), ("speak", "c3")),
        ToolMessage(content="Arrived.", name="drive", tool_call_id="c2"),
        HumanMessage("Next task.\n"),
    ]

    assert repair_history(messages) == 2

    assert [m.tool_call_id for m in messages[1:4]] == ["c2", "c1", "c3"]
    assert messages[1].content == "Arrived."
    assert isinstance(messages[4], HumanMessage)


def test_repair_answers_calls_at_the_end_of_the_history():
    messages = [HumanMessage("Go."), calls(("drive", "c1"))]

    assert repair_history(messages) == 1
    assert messages[-1].tool_call_id == "c1"


def test_repair_puts_answers_before_a_tool_image_message():
    """RAI's ToolRunner adds a tool's images as a human message after the tool results."""
    image = HumanMultimodalMessage(
        content="Image returned by a tool call c1", images=[], tool_call_id="c1"
    )
    messages = [
        calls(("look", "c1"), ("drive", "c2")),
        ToolMessage(content="Saw a mug.", name="look", tool_call_id="c1"),
        image,
    ]

    assert repair_history(messages) == 1

    assert [type(m) for m in messages] == [
        AIMessage, ToolMessage, ToolMessage, HumanMultimodalMessage
    ]
    assert messages[2].tool_call_id == "c2"


def test_repair_is_idempotent():
    messages = [
        HumanMessage("Go."),
        calls(("drive", "c1"), ("speak", "c2")),
        ToolMessage(content="Arrived.", name="drive", tool_call_id="c1"),
        AIMessage("Done."),
        calls(("look", "c3")),
    ]
    assert repair_history(messages) == 2
    repaired = list(messages)

    assert repair_history(messages) == 0
    assert messages == repaired


def test_repair_answers_calls_with_unreadable_arguments():
    """langchain_openai sends invalid_tool_calls back to the endpoint, so they need answers too."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "look", "args": {}, "id": "c1"}],
            invalid_tool_calls=[unreadable("drive", "c2")],
        ),
        ToolMessage(content="Saw a mug.", name="look", tool_call_id="c1"),
        HumanMessage("Next task.\n"),
    ]

    assert repair_history(messages) == 1

    assert [type(m) for m in messages] == [AIMessage, ToolMessage, ToolMessage, HumanMessage]
    answer = messages[2]
    assert (answer.tool_call_id, answer.name, answer.content, answer.status) == (
        "c2", "drive", UNREADABLE_ARGUMENTS, "error"
    )
    assert repair_history(messages) == 0


def test_a_call_with_unreadable_arguments_is_answered_and_the_llm_tries_again():
    driven: list[str] = []

    @tool
    def drive(to: str) -> str:
        """Drive to a place."""
        driven.append(to)
        return f"Arrived at {to}."

    # An open item, so ending the run here would send the gate message right
    # after the unanswered call.
    todos = seeded(["Go to the kitchen"], in_progress="t1")
    agent, llm = make_agent(
        [
            AIMessage(content="", invalid_tool_calls=[unreadable("drive", "call_bad")]),
            tool_call("drive", {"to": "kitchen"}, "call_ok"),
            todo_call([update("t1", status="completed")], "call_done"),
            AIMessage("At the kitchen."),
        ],
        tools=[drive, UpdateTodosTool(todos=todos)],
        todos=todos,
    )

    messages = run_graph(agent, "Drive to the kitchen.")

    # The retry's input answers the call right after its AI message.
    retry = llm.received[1]
    assert [type(m) for m in retry] == [HumanMessage, AIMessage, ToolMessage, HumanMessage]
    answer = retry[2]
    assert (answer.tool_call_id, answer.content, answer.status) == (
        "call_bad", UNREADABLE_ARGUMENTS, "error"
    )
    assert driven == ["kitchen"]
    assert len(llm.received) == 4
    assert not any(str(m.content).startswith("[todo]") for m in messages)


# -- Unknown tools and the ledger ------------------------------------------------


def test_unknown_tool_gets_an_error_and_the_run_goes_on():
    todos = seeded(["Fly to the kitchen"], in_progress="t1")
    agent, llm = make_agent(
        [
            tool_call("fly", call_id="call_fly"),
            todo_call([update("t1", status="completed")], "call_done"),
            AIMessage("I cannot fly, so I walked."),
        ],
        tools=[UpdateTodosTool(todos=todos)],
        todos=todos,
    )

    messages = run_graph(agent, "Fly to the kitchen.")

    unknown = result_of(messages, "call_fly")
    assert (unknown.content, unknown.name, unknown.status) == ("Unknown tool fly.", "fly", "error")
    # An unknown name is not a failed tool, so the ledger does not block "completed".
    assert result_of(messages, "call_done").status == "success"
    assert len(llm.received) == 3


def test_completion_waits_until_the_failed_tool_succeeds():
    attempts: list[int] = []

    @tool
    def hand_over() -> str:
        """Hand the object to the person."""
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("The gripper did not open.")
        return "Handed over."

    todos = seeded(["Hand over the mug"])
    agent, llm = make_agent(
        [
            todo_call([update("t1", status="in_progress")], "call_start"),
            tool_call("hand_over", call_id="call_h1"),
            todo_call([update("t1", status="completed")], "call_done1"),
            tool_call("hand_over", call_id="call_h2"),
            todo_call([update("t1", status="completed")], "call_done2"),
            AIMessage("Here is your mug."),
        ],
        tools=[hand_over, UpdateTodosTool(todos=todos)],
        todos=todos,
    )

    messages = run_graph(agent, "Hand me the mug.")

    failed = result_of(messages, "call_h1")
    assert failed.status == "error"
    assert failed.content == "Failed to run tool. Error: The gripper did not open."
    refused = result_of(messages, "call_done1")
    assert refused.status == "error"
    assert "t1 cannot be completed because hand_over failed" in refused.content
    assert result_of(messages, "call_done2").status == "success"
    item = todos.snapshot()[0]
    assert item.status == "completed"
    assert item.evidence == ["call_h2"]
    assert len(llm.received) == 6


def test_a_handled_tool_exception_counts_as_failed():
    # The ROS tools report a failure this way (handle_tool_error on): the LLM
    # reads the plain sentence, and the ledger still sees a failed call.
    def drive() -> str:
        """Drive to the kitchen."""
        raise ToolException("Navigation to kitchen failed: Nav2 rejected the goal.")

    todos = seeded([{"content": "Go to the kitchen", "requires_tool": "drive"}], in_progress="t1")
    agent, llm = make_agent(
        [
            tool_call("drive", call_id="call_drive"),
            todo_call([update("t1", status="completed")], "call_done"),
            todo_call([update("t1", status="failed", note="Nav2 rejected the goal.")], "call_fail"),
            AIMessage("I could not get to the kitchen."),
        ],
        tools=[
            StructuredTool.from_function(drive, handle_tool_error=True),
            UpdateTodosTool(todos=todos),
        ],
        todos=todos,
    )

    messages = run_graph(agent)

    failed = result_of(messages, "call_drive")
    assert failed.content == "Navigation to kitchen failed: Nav2 rejected the goal."
    assert failed.status == "error"
    assert result_of(messages, "call_done").status == "error"
    item = todos.snapshot()[0]
    assert (item.status, item.evidence) == ("failed", [])
    assert len(llm.received) == 4


def test_a_call_with_bad_arguments_counts_as_failed():
    @tool
    def count(n: int) -> str:
        """Count to n."""
        return str(n)

    todos = seeded(["Count the cups"], in_progress="t1")
    agent, _ = make_agent(
        [
            tool_call("count", {"n": "many"}, "call_bad"),
            todo_call([update("t1", status="completed")], "call_done1"),
            todo_call([update("t1", status="completed", note="Counted by eye.")], "call_done2"),
            AIMessage("There are three cups."),
        ],
        tools=[count, UpdateTodosTool(todos=todos)],
        todos=todos,
    )

    messages = run_graph(agent, "Count the cups.")

    # ToolRunner's own message for a pydantic ValidationError is kept.
    assert "Validation error in tool count" in result_of(messages, "call_bad").content
    assert result_of(messages, "call_done1").status == "error"
    assert result_of(messages, "call_done2").status == "success"


def test_only_the_first_update_todos_call_in_a_message_is_applied():
    todos = TodoList()
    first = {
        "name": "update_todos",
        "args": {"ops": [add("Say hello", status="completed")]},
        "id": "u1",
    }
    second = {"name": "update_todos", "args": {"ops": [add("Say goodbye")]}, "id": "u2"}
    agent, llm = make_agent(
        [AIMessage(content="", tool_calls=[first, second]), AIMessage("Hello.")],
        tools=[UpdateTodosTool(todos=todos)],
        todos=todos,
    )

    messages = run_graph(agent)

    assert [i.content for i in todos.snapshot()] == ["Say hello"]
    assert result_of(messages, "u1").status == "success"
    refused = result_of(messages, "u2")
    assert refused.status == "error"
    assert refused.content == (
        "Only one update_todos call per message is accepted, so this one was not applied. "
        "Put all ops in one call."
    )
    assert len(llm.received) == 2


# -- Per-turn binding ------------------------------------------------------------


def test_tools_are_bound_on_every_call():
    agent, llm = make_agent(
        [
            tool_call("grow"),
            tool_call("extra", call_id="call_extra"),
            AIMessage("Done."),
        ],
        tools=[make_tool("drop")],
    )

    def grow() -> str:
        agent.tools.add(make_tool("extra", "Extra ran."))
        agent.tools.remove("drop")
        return "Grown."

    agent.tools.add(StructuredTool.from_function(func=grow, name="grow", description="Grow."))

    messages = run_graph(agent)

    assert llm.bound == [["drop", "grow"], ["grow", "extra"], ["grow", "extra"]]
    # A tool added during the task is callable on the next turn.
    assert result_of(messages, "call_extra").content == "Extra ran."


def test_more_than_20_tools_warns_once_per_count(caplog):
    tools = [make_tool(f"noop_{i}") for i in range(MAX_BOUND_TOOLS + 1)]
    agent, llm = make_agent(
        [tool_call("noop_0"), AIMessage("Done."), AIMessage("Done again.")], tools=tools
    )

    def warnings() -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.name == "walkie_agent.graph" and r.levelno == logging.WARNING
        ]

    with caplog.at_level(logging.WARNING, logger="walkie_agent.graph"):
        run_graph(agent)
        assert len(warnings()) == 1
        assert warnings()[0].startswith("21 tools are bound, more than 20")
        # Every tool is still bound.
        assert [len(names) for names in llm.bound] == [21, 21]

        agent.tools.add(make_tool("noop_21"))
        run_graph(agent)
        assert len(warnings()) == 2
        assert warnings()[1].startswith("22 tools are bound")


def test_a_long_task_is_not_cut_short():
    """LangGraph's default limit of 25 steps would stop this task and raise."""
    rounds = 15
    agent, llm = make_agent(
        [tool_call("noop", call_id=f"call_{n}") for n in range(rounds)] + [AIMessage("Done.")],
        tools=[make_tool("noop")],
    )

    run_graph(agent)

    assert len(llm.received) == rounds + 1
    assert agent.agent.config["recursion_limit"] == RECURSION_LIMIT


# -- The completion gate ---------------------------------------------------------


def test_gate_sends_the_llm_back_at_most_twice_per_task():
    todos = seeded(["Go to the kitchen", "Come back"])
    agent, llm = make_agent(
        [AIMessage(f"Done {n}.") for n in range(6)],
        todos=todos,
    )
    state = {"messages": [HumanMessage("Go and come back.")]}

    agent.agent.invoke(state)

    messages = state["messages"]
    # The gate messages are saved in the history.
    assert contents(messages) == [
        "Go and come back.", "Done 0.", GATE_T1_T2, "Done 1.", GATE_T1_T2, "Done 2."
    ]
    assert len(llm.received) == 3
    # The reminder goes out with every call, after the history, and is never saved.
    reminder = "<todos>\n[ ] t1 Go to the kitchen\n[ ] t2 Come back\n</todos>"
    assert llm.reminders == [reminder] * 3
    assert contents(llm.received[1][-2:]) == [GATE_T1_T2, reminder]
    assert not any("<todos>" in str(m.content) for m in messages)

    # A new message from the person starts a new task, with two gates again.
    messages.append(HumanMessage("Please finish."))
    agent.agent.invoke(state)
    assert len(llm.received) == 6
    assert contents(messages[6:]) == [
        "Please finish.", "Done 3.", GATE_T1_T2, "Done 4.", GATE_T1_T2, "Done 5."
    ]


def test_a_tool_image_message_does_not_start_a_new_task():
    """The human message that RAI's ToolRunner adds for a tool's images is not the person."""
    todos = seeded(["Find the mug"])
    agent, llm = make_agent([AIMessage("Found it."), AIMessage("Really found it.")], todos=todos)
    gate = "[todo] Open items: t1. Finish them, or mark them failed or cancelled with a note."
    state = {
        "messages": [
            HumanMessage("Find the mug."),
            AIMessage("Looking."),
            HumanMessage(gate),
            calls(("look", "c1")),
            ToolMessage(content="A mug on the table.", name="look", tool_call_id="c1"),
            HumanMultimodalMessage(
                content="Image returned by a tool call c1", images=[], tool_call_id="c1"
            ),
        ]
    }

    agent.agent.invoke(state)

    # One gate was already used, so only one more fires.
    assert len(llm.received) == 2
    assert contents(state["messages"][6:]) == ["Found it.", gate, "Really found it."]


@pytest.mark.parametrize("finished", [False, True], ids=["empty list", "nothing open"])
def test_no_gate_without_open_items(finished):
    todos = TodoList()
    if finished:
        todos.seed(["Say hello", "Wave"])
        todos.apply([
            update("t1", status="completed"),
            update("t2", status="cancelled", note="No arm."),
        ])
    agent, llm = make_agent([AIMessage("Hello.")], todos=todos)

    messages = run_graph(agent, "Say hello.")

    assert len(llm.received) == 1
    assert contents(messages) == ["Say hello.", "Hello."]


def test_no_gate_while_an_item_is_waiting():
    todos = seeded(["Pick up the mug", "Bring it back"], in_progress="t1")
    todos.set_waiting("Which mug?")
    agent, llm = make_agent([AIMessage("I will wait for the answer.")], todos=todos)

    run_graph(agent)

    assert len(llm.received) == 1


# -- wait_for_person -------------------------------------------------------------


def test_wait_for_person_ends_the_run_and_the_answer_resumes_the_item(start):
    todos = TodoList()
    speak, said = fake_speak()
    agent, llm = make_agent(
        [
            todo_call([update("t1", status="in_progress")], "call_start"),
            tool_call("wait_for_person", {"question": "Which mug?"}, "call_ask"),
            todo_call(
                [
                    update("t1", status="completed"),
                    update("t2", status="cancelled", note="The person came to fetch it."),
                ],
                "call_done",
            ),
            AIMessage("Done."),
        ],
        tools=todo_tools(todos, speak),
        todos=todos,
    )
    start(agent)

    agent.submit("Bring me a mug.", todos=["Pick up the mug", "Bring it back"])
    assert settle(agent, llm, 2)

    # No gate although items are open: the question ended the run.
    assert len(llm.received) == 2
    assert said == ["Which mug?"]
    assert [(i.id, i.status) for i in todos.snapshot()] == [("t1", "waiting"), ("t2", "pending")]
    assert agent.state["messages"][-1].content == "Asked: Which mug?"

    agent(human("The red one."))
    assert settle(agent, llm, 4)

    assert len(llm.received) == 4
    assert llm.reminders[2] == "<todos>\n[>] t1 Pick up the mug\n[ ] t2 Bring it back\n</todos>"
    assert llm.received[2][-2].content == "The red one.\n"


def test_calls_after_a_question_in_the_same_message_do_not_run():
    driven: list[str] = []

    @tool
    def drive(place: str) -> str:
        """Drive to a place."""
        driven.append(place)
        return f"Arrived at {place}."

    todos = seeded(["Bring the mug"], in_progress="t1")
    speak, said = fake_speak()
    question = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "wait_for_person",
                "args": {"question": "Kitchen or bedroom?"},
                "id": "call_ask",
                "type": "tool_call",
            },
            {"name": "drive", "args": {"place": "kitchen"}, "id": "call_drive", "type": "tool_call"},
        ],
    )
    agent, llm = make_agent([question], tools=[*todo_tools(todos, speak), drive], todos=todos)

    messages = run_graph(agent)

    assert said == ["Kitchen or bedroom?"]
    assert driven == []
    skipped = result_of(messages, "call_drive")
    assert (skipped.content, skipped.status) == (AFTER_QUESTION, "error")
    item = todos.snapshot()[0]
    assert item.status == "waiting"
    assert "call_drive" not in item.evidence
    assert len(llm.received) == 1


def test_failed_wait_for_person_does_not_end_the_run():
    todos = TodoList()
    speak, _ = fake_speak("Nothing was said: no speech node is listening on /to_human.")
    agent, llm = make_agent(
        [
            tool_call("wait_for_person", {"question": "Which mug?"}, "call_ask"),
            AIMessage("I could not ask which mug you mean."),
        ],
        tools=todo_tools(todos, speak),
        todos=todos,
    )

    messages = run_graph(agent)

    assert result_of(messages, "call_ask").status == "error"
    assert len(llm.received) == 2


# -- Queue, stop and reset -------------------------------------------------------


def test_utterances_during_a_task_wait_and_run_one_by_one_in_order(start):
    blocker = Blocker()
    agent, llm = make_agent(
        [
            tool_call("slow_action"),
            AIMessage("At the kitchen."),
            AIMessage("A."),
            AIMessage("B."),
        ],
        tools=[blocker.tool],
    )
    start(agent)

    agent(human("Drive to the kitchen."))
    assert blocker.started.wait(timeout=5)
    agent(human("Say A."))
    agent(human("Say B."))
    blocker.release.set()
    assert settle(agent, llm, 4)

    assert blocker.calls == 1
    # The first task ran to its end undisturbed.
    assert contents(llm.received[1]) == ["Drive to the kitchen.\n", "", "done"]
    # Each queued utterance is its own task, oldest first, with the whole
    # history before it in its input.
    assert contents(llm.received[2]) == [
        "Drive to the kitchen.\n", "", "done", "At the kitchen.", "Say A.\n"
    ]
    assert contents(llm.received[3])[-3:] == ["Say A.\n", "A.", "Say B.\n"]


@pytest.mark.parametrize("stop", ["stop", "Stop.", "Never mind!", "nevermind"])
def test_stop_interrupts_the_task_and_drops_older_messages(start, stop):
    blocker = Blocker()
    speak, said = fake_speak()
    agent, llm = make_agent(
        [
            tool_call("slow_action"),
            tool_call("speak", {"text": "Okay, I stopped."}),
            AIMessage("Stopped."),
        ],
        tools=[blocker.tool, speak],
    )
    start(agent)

    agent(human("Drive to the kitchen."))
    assert blocker.started.wait(timeout=5)
    agent.submit("Then the bedroom.", todos=["Go to the bedroom"])
    agent(human(stop))
    blocker.release.set()
    assert settle(agent, llm, 3)

    # The task ended after the running tool, and the stop came next; the
    # queued task and its seed were dropped.
    assert len(llm.received) == 3
    assert contents(llm.received[1]) == ["Drive to the kitchen.\n", "", "done", f"{stop}\n"]
    # The interrupt that the stop set does not cut the stop's own run short.
    assert said == ["Okay, I stopped."]
    assert llm.reminders == ["", "", ""]
    assert agent.todos.is_empty


def test_stop_during_the_llm_call_cancels_its_tool_calls(start):
    drives: list[int] = []

    @tool
    def drive() -> str:
        """Drive somewhere."""
        drives.append(1)
        return "Arrived."

    llm_started, release_llm = threading.Event(), threading.Event()
    agent, llm = make_agent(
        [tool_call("drive", call_id="call_drive"), AIMessage("Okay, I stopped.")], tools=[drive]
    )
    llm.hooks[0] = lambda: (llm_started.set(), release_llm.wait(timeout=5))
    start(agent)

    agent(human("Drive to the kitchen."))
    assert llm_started.wait(timeout=5)
    agent(human("Stop!"))
    release_llm.set()
    assert settle(agent, llm, 2)

    assert drives == []
    second = llm.received[1]
    assert [type(m) for m in second] == [HumanMessage, AIMessage, ToolMessage, HumanMessage]
    repaired = second[2]
    assert (repaired.tool_call_id, repaired.content, repaired.status) == (
        "call_drive", INTERRUPTED, "error"
    )
    assert second[3].content == "Stop!\n"


def test_stop_while_idle_is_an_ordinary_message(start):
    agent, llm = make_agent([AIMessage("There is nothing to stop.")])
    start(agent)

    agent(human("stop"))
    assert settle(agent, llm, 1)

    assert contents(llm.received[0]) == ["stop\n"]
    assert contents(agent.state["messages"]) == ["stop\n", "There is nothing to stop."]


@pytest.mark.parametrize("when", ["before the agent starts", "between tasks"])
def test_stop_while_idle_keeps_the_messages_queued_before_it(start, when):
    agent, llm = make_agent([AIMessage("Hello."), AIMessage("There is nothing to stop.")])
    poll = PauseOnce()
    if when == "between tasks":
        agent._reset_requested = poll
        start(agent)
        assert wait_until(agent.ready)
        # Holds the idle agent thread at the start of its next poll, so both
        # messages are queued before it takes one.
        poll.pause_next()
        assert poll.paused.wait(timeout=5)

    agent(human("Say hello."))
    agent(human("stop"))
    if when == "between tasks":
        poll.resume.set()
    else:
        start(agent)
    assert settle(agent, llm, 2)

    assert contents(llm.received[0]) == ["Say hello.\n"]
    assert contents(llm.received[1]) == ["Say hello.\n", "Hello.", "stop\n"]


def test_agent_stop_while_a_task_starts_ends_it_after_one_step(start):
    """stop() that lands while a run starts must not have its interrupt cleared by that run."""
    agent, llm = make_agent(
        [tool_call("noop", call_id=f"call_{n}") for n in range(3)] + [AIMessage("Done.")],
        tools=[make_tool("noop")],
    )
    lock = ContendedLock()
    agent._buffer_lock = lock
    stopper = threading.Thread(target=agent.stop)
    stop_called = threading.Event()

    def stop_now() -> None:
        # Runs on the agent thread while it starts the run, holding the
        # buffer lock, right before the run clears the interrupt event. Waits
        # until stop() has set the interrupt or is waiting for the lock.
        lock.contended.clear()
        stopper.start()
        stop_called.set()
        wait_until(lambda: agent._interrupt_event.is_set() or lock.contended.is_set())

    agent._interrupt_event = HookOnFirstClear(stop_now)
    start(agent)

    agent(human("Do it."))
    assert stop_called.wait(timeout=5)
    stopper.join(timeout=5)

    assert not stopper.is_alive()
    # The run ended after its first step instead of running every tool round.
    assert len(llm.received) == 1


@pytest.mark.parametrize(
    "text", ["stop", "Stop.", "  STOP!  ", "stop?", "Never mind!", "never   mind", "Nevermind."]
)
def test_stop_utterances(text):
    assert is_stop_utterance(text)


@pytest.mark.parametrize(
    "text", ["stop the music", "Walkie, stop", "don't stop", "never mind the cup", "stopping", ""]
)
def test_other_utterances_are_not_stops(text):
    assert not is_stop_utterance(text)


def test_reset_during_a_run_clears_history_todos_and_seeds(start):
    blocker = Blocker()
    agent, llm = make_agent(
        [tool_call("slow_action"), AIMessage("Hello.")], tools=[blocker.tool]
    )
    start(agent)
    first_session = agent.session_id

    agent.submit("Drive to the kitchen.", todos=["Go to the kitchen"])
    assert blocker.started.wait(timeout=5)
    assert not agent.todos.is_empty
    agent.submit("Then the bedroom.", todos=["Go to the bedroom"])
    agent.reset()
    blocker.release.set()

    assert wait_until(lambda: agent.session_id != first_session and agent.ready())
    # The task stopped after the running tool.
    assert len(llm.received) == 1
    assert agent.state["messages"] == []
    assert agent.todos.is_empty

    agent(human("Say hello."))
    assert settle(agent, llm, 2)
    # Neither the queued task nor its seed survived the reset.
    assert contents(llm.received[1]) == ["Say hello.\n"]
    assert llm.reminders[1] == ""


# -- Seeds and the list between tasks --------------------------------------------


@pytest.mark.parametrize("source", ["seed", "operator"])
def test_submit_seeds_the_list_before_the_first_call(start, source):
    todos = TodoList()
    todos.apply([add("An old step")])
    # The steps stay open, so the gate sends the LLM back twice.
    agent, llm = make_agent([AIMessage(f"On it {n}.") for n in range(3)], todos=todos)
    start(agent)

    agent.submit(
        "Bring me the mug.",
        todos=["Go to the kitchen", {"content": "Hand over the mug", "requires_tool": "hand_over"}],
        source=source,
    )
    assert settle(agent, llm, 3)

    assert llm.reminders[0] == (
        "<todos>\n[ ] t1 Go to the kitchen\n[ ] t2 Hand over the mug  needs: hand_over\n</todos>"
    )
    assert [(i.content, i.source, i.locked) for i in todos.snapshot()] == [
        ("Go to the kitchen", source, True),
        ("Hand over the mug", source, True),
    ]


def test_submit_with_an_empty_seed_empties_the_list(start):
    todos = TodoList()
    todos.apply([add("An old step")])
    agent, llm = make_agent([AIMessage("Hello.")], todos=todos)
    start(agent)

    agent.submit("Say hello.", todos=[])
    assert settle(agent, llm, 1)

    assert todos.is_empty
    assert llm.reminders == [""]


def test_submit_rejects_a_bad_seed(start):
    agent, llm = make_agent([AIMessage("Hello.")])
    start(agent)

    with pytest.raises(TodoRejected):
        agent.submit("Do it.", todos=[{"content": ""}])
    with pytest.raises(TodoRejected):
        agent.submit("Do it.", todos="Go to the kitchen")  # type: ignore[arg-type]
    agent.submit("Say hello.")
    assert settle(agent, llm, 1)

    # The rejected tasks were never queued.
    assert len(llm.received) == 1
    assert contents(llm.received[0]) == ["Say hello.\n"]


def test_a_finished_list_is_cleared_when_the_next_task_starts(start):
    todos = seeded(["Say hello"])
    todos.apply([update("t1", status="completed")])
    agent, llm = make_agent([AIMessage("Hello again.")], todos=todos)
    start(agent)

    agent(human("Say hello again."))
    assert settle(agent, llm, 1)

    assert todos.is_empty
    assert llm.reminders == [""]


def test_a_list_with_open_items_is_kept_for_the_next_message(start):
    todos = seeded(["Go to the kitchen", "Come back"])
    todos.apply([update("t1", status="completed")])
    # t2 stays open, so the gate sends the LLM back twice.
    agent, llm = make_agent([AIMessage(f"Coming back {n}.") for n in range(3)], todos=todos)
    start(agent)

    agent(human("Carry on."))
    assert settle(agent, llm, 3)

    assert llm.reminders[0] == "<todos>\n[x] t1 Go to the kitchen\n[ ] t2 Come back\n</todos>"
    assert [i.status for i in todos.snapshot()] == ["completed", "pending"]


# -- Task templates --------------------------------------------------------------


@pytest.mark.parametrize("before", ["empty list", "finished list"])
def test_an_utterance_with_a_keyword_starts_the_template(start, before):
    todos = TodoList()
    if before == "finished list":
        todos = seeded(["Say hello"])
        todos.apply([update("t1", status="completed")])
    # The template's steps stay open, so the gate sends the LLM back twice.
    agent, llm = make_agent(
        [AIMessage(f"On it {n}.") for n in range(3)], todos=todos, templates={"tour": TOUR}
    )
    start(agent)

    agent(human("Walkie, give me a House Tour."))
    assert settle(agent, llm, 3)

    task = llm.received[0][0]
    assert task.content == "Walkie, give me a House Tour.\n\nShow the person the house.\n"
    assert llm.reminders[0] == "<todos>\n[ ] t1 Go to the kitchen\n[ ] t2 Come back\n</todos>"
    assert [(i.source, i.locked) for i in todos.snapshot()] == [("seed", True), ("seed", True)]


@pytest.mark.parametrize("open_item", ["pending", "waiting"])
def test_no_template_while_items_are_open(start, open_item):
    todos = seeded(["Go to the bedroom"], in_progress="t1" if open_item == "waiting" else "")
    if open_item == "waiting":
        todos.set_waiting("Which bedroom?")
    agent, llm = make_agent(
        [AIMessage(f"Going {n}.") for n in range(3)], todos=todos, templates={"tour": TOUR}
    )
    start(agent)

    agent(human("The one after the house tour."))
    assert settle(agent, llm, 3)

    assert llm.received[0][0].content == "The one after the house tour.\n"
    assert [i.content for i in todos.snapshot()] == ["Go to the bedroom"]


def test_a_seeded_task_does_not_use_a_template(start):
    agent, llm = make_agent(
        [AIMessage(f"Hello {n}.") for n in range(3)], templates={"tour": TOUR}
    )
    start(agent)

    agent.submit("Say hello before the house tour.", todos=["Say hello"])
    assert settle(agent, llm, 3)

    assert llm.received[0][0].content == "Say hello before the house tour.\n"
    assert llm.reminders[0] == "<todos>\n[ ] t1 Say hello\n</todos>"


# -- Errors ----------------------------------------------------------------------


@pytest.mark.parametrize("failure", ["model raises", "out of responses"])
def test_an_llm_error_is_logged_and_the_agent_keeps_serving(start, caplog, failure):
    agent, llm = make_agent([])
    if failure == "model raises":

        def fail() -> None:
            raise RuntimeError("The LLM endpoint is down.")

        llm.hooks[0] = fail
    error = RuntimeError if failure == "model raises" else AssertionError
    start(agent)

    with caplog.at_level(logging.ERROR):
        agent(human("First task."))
        assert wait_until(
            lambda: any(
                r.exc_info is not None and isinstance(r.exc_info[1], error)
                for r in caplog.records
            )
        )
    assert wait_until(agent.ready)

    llm.responses.append(AIMessage("Back again."))
    agent(human("Second task."))
    assert settle(agent, llm, 2)

    assert contents(llm.received[1]) == ["First task.\n", "Second task.\n"]
    assert contents(agent.state["messages"])[-1] == "Back again."
