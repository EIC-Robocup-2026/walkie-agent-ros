"""The agent's own LangGraph graph (ADR-0038).

RAI's create_react_runnable binds the tools once, when the graph is compiled.
Walkie needs more on every LLM call: the tools bound at that moment, the todo
list sent as a reminder that is not saved, a history without unanswered tool
calls, and routes for the completion gate and wait_for_person. This module
builds that graph on RAI's state type, its ToolRunner and its LLM tracing
helper.

The graph keeps RAI's in-place state contract. Every node edits
state["messages"], the same list object that the agent holds, and returns
None, except the tools node, which returns its input as ToolRunner does.
LangChainAgent and the tests read that list after a run, so a node must never
replace it.
"""

import logging
import threading
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from rai.agents.langchain.core import ReActAgentState, ToolRunner
from rai.agents.langchain.invocation_helpers import invoke_llm_with_tracing
from rai.messages import SystemMultimodalMessage

from walkie_agent.todo import TodoList

logger = logging.getLogger(__name__)

# The small local fallback model picks tools poorly beyond this many. Phase 1
# still binds every tool and only warns; later phases keep the bound set under
# it.
MAX_BOUND_TOOLS = 20
# How many times the completion gate may send the LLM back within one task.
MAX_GATES = 2
GATE_PREFIX = "[todo] Open items"
# The answer that repair_history gives a tool call that never ran.
INTERRUPTED = "Interrupted before it ran."
# Its answer to a call whose arguments LangChain could not parse.
UNREADABLE_ARGUMENTS = "The arguments were not a valid JSON object, so the call did not run."
ONE_UPDATE_TODOS = (
    "Only one update_todos call per message is accepted, so this one was not applied. "
    "Put all ops in one call."
)
# The answer to a call that follows a successful wait_for_person in the same
# AI message.
AFTER_QUESTION = (
    "Not run: you asked the person a question, so the task stopped to wait for "
    "the answer. Call it again after the answer if it is still needed."
)


class ToolSet:
    """The agent's tools by name, in the order they were added.

    Thread-safe: the agent thread reads it before every LLM call, while a tool
    or another thread may add or remove tools during a task. A change takes
    effect on the next LLM call, because the llm node binds the visible tools
    every time.
    """

    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        self._lock = threading.Lock()
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.add(tool)

    def add(self, tool: BaseTool) -> None:
        """Adds a tool. A tool with the same name is replaced in its place."""
        with self._lock:
            self._tools[tool.name] = tool

    def remove(self, name: str) -> bool:
        """Removes the tool called `name`. Returns False if there was none."""
        with self._lock:
            return self._tools.pop(name, None) is not None

    def get(self, name: str) -> BaseTool | None:
        with self._lock:
            return self._tools.get(name)

    def visible(self) -> list[BaseTool]:
        """The tools the LLM may call now.

        Phase 1 shows every tool. Later phases filter by access level and
        turn taint here (ADR-0038).
        """
        with self._lock:
            return list(self._tools.values())

    def names(self) -> list[str]:
        with self._lock:
            return list(self._tools)


def repair_history(messages: list[BaseMessage]) -> int:
    """Answers every tool call that has no result, in place.

    A run that stops between the llm node and the tools node, after a stop or
    a reset, leaves an AI message whose tool calls have no results. So does a
    call whose arguments are not a valid JSON object: LangChain keeps it in
    invalid_tool_calls, which ToolRunner never runs, but langchain_openai
    still sends it back to the endpoint. OpenAI-compatible endpoints reject
    such a history. They also need each result in the block of tool messages
    right after its AI message, so a missing answer goes at the end of that
    block, before anything that follows it, such as the human message that
    RAI's ToolRunner adds for a tool's images. Returns how many answers were
    added; a second call adds none.
    """
    added = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        if not isinstance(message, AIMessage) or not (
            message.tool_calls or message.invalid_tool_calls
        ):
            continue
        answered: set[str] = set()
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            answered.add(messages[index].tool_call_id)
            index += 1
        calls_and_answers = [
            *((call, INTERRUPTED) for call in message.tool_calls),
            *((call, UNREADABLE_ARGUMENTS) for call in message.invalid_tool_calls),
        ]
        for call, content in calls_and_answers:
            call_id = call.get("id") or ""
            if call_id in answered:
                continue
            answer = ToolMessage(
                content=content, name=call.get("name"), tool_call_id=call_id, status="error"
            )
            messages.insert(index, answer)
            answered.add(call_id)
            index += 1
            added += 1
    return added


class _ToolHandle:
    """What WalkieToolRunner.tools_by_name returns for a name.

    ToolRunner.run_one only calls invoke(call, config) on it.
    """

    def __init__(self, runner: "WalkieToolRunner", name: str) -> None:
        self._runner = runner
        self._name = name

    def invoke(self, call: ToolCall, config: RunnableConfig | None = None) -> Any:
        return self._runner._run_call(self._name, call, config)


class _LiveTools(Mapping[str, _ToolHandle]):
    """A live view of the ToolSet, used as ToolRunner's tools_by_name.

    Every name gives a handle, so ToolRunner never sees a KeyError for an
    unknown name: the handle answers it instead.
    """

    def __init__(self, runner: "WalkieToolRunner") -> None:
        self._runner = runner

    def __getitem__(self, name: str) -> _ToolHandle:
        return _ToolHandle(self._runner, name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._runner.toolset.get(name) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self._runner.toolset.names())

    def __len__(self) -> int:
        return len(self._runner.toolset.names())


class WalkieToolRunner(ToolRunner):
    """RAI's ToolRunner over a live ToolSet, reporting each result to the todo list.

    ToolRunner runs the calls one at a time and turns a tool's exception into
    an error ToolMessage; both are inherited. Only the lookup changes:
    run_one calls tools_by_name[name].invoke(call, config), and here
    tools_by_name is a live view of the ToolSet. So a tool added during a task
    is found, and an unknown name gets a plain answer instead of a KeyError.
    The result of every known tool goes to the todo list's error ledger. The
    calls that follow a successful wait_for_person in the same AI message do
    not run.

    Relies on ToolRunner internals of rai-core 2.12 (tools_by_name, _func and
    the try block in run_one); recheck them on every upgrade (ADR-0038).
    """

    def __init__(self, toolset: ToolSet, todos: TodoList) -> None:
        super().__init__([], name="tools")
        self.toolset = toolset
        self.todos = todos
        self.tools_by_name = _LiveTools(self)  # type: ignore[assignment]
        # The update_todos calls of the current AI message that are answered
        # without running. The graph runs one task at a time, so one list is
        # enough.
        self._refused: list[ToolCall] = []
        # Set once wait_for_person succeeds; the later calls of the same AI
        # message are answered without running.
        self._asked = False

    def _func(self, input: dict[str, Any], config: RunnableConfig) -> Any:
        # One AI message may hold several update_todos calls. Each applies on
        # its own, so a model that splits its ops would get a half-applied
        # plan; only the first call runs.
        messages = self.get_messages(input)
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage):
            todo_calls = [call for call in last.tool_calls if call["name"] == "update_todos"]
            self._refused = todo_calls[1:]
        self._asked = False
        try:
            return super()._func(input, config)
        finally:
            self._refused = []
            self._asked = False

    def _run_call(self, name: str, call: ToolCall, config: RunnableConfig | None) -> Any:
        call_id = call.get("id") or ""
        if self._asked:
            # ToolRunner runs the calls in order, one at a time (max_concurrency
            # 1), so every call after the question lands here. Running one,
            # such as a drive, before the answer arrives would act on a guess
            # and give the waiting item evidence it has not earned. Not a tool
            # failure, so the ledger does not hear of it.
            return ToolMessage(
                content=AFTER_QUESTION, name=name, tool_call_id=call_id, status="error"
            )
        tool = self.toolset.get(name)
        if tool is None:
            # Not a tool failure, so the ledger does not hear of it.
            return ToolMessage(
                content=f"Unknown tool {name}.", name=name, tool_call_id=call_id, status="error"
            )
        # run_one passes the call objects of the AI message itself, so
        # identity tells a refused call apart even when call ids repeat.
        if any(call is refused for refused in self._refused):
            return ToolMessage(
                content=ONE_UPDATE_TODOS, name=name, tool_call_id=call_id, status="error"
            )
        try:
            output = tool.invoke(call, config)
        except Exception:
            # Re-raised so that ToolRunner writes its usual error message,
            # including its format for a pydantic ValidationError.
            self.todos.record_tool_result(name, call_id, ok=False)
            raise
        if isinstance(output, ToolMessage):
            ok = output.status != "error"
            self.todos.record_tool_result(name, call_id, ok=ok)
            if ok and name == "wait_for_person":
                self._asked = True
        return output


def build_walkie_graph(
    llm: BaseChatModel,
    toolset: ToolSet,
    todos: TodoList,
    system_prompt: str | SystemMultimodalMessage | None = None,
) -> Runnable[ReActAgentState, ReActAgentState]:
    """Builds the agent's graph: llm, tools and the completion gate.

    llm -> tools when the LLM calls a tool; the tools go back to llm, or end
    the run once wait_for_person has asked its question. llm -> llm when every
    call it made has arguments that are not valid JSON; the history repair
    answers them, so the LLM can try again. llm -> gate when the LLM calls no
    tool while todo items are open and none is waiting, at most MAX_GATES
    times per task; the gate goes back to llm. Otherwise the run ends.
    """
    # Tool counts already warned about, so a long task does not log the same
    # warning on every turn.
    warned_counts: set[int] = set()

    def llm_node(state: ReActAgentState, config: RunnableConfig) -> None:
        messages = state["messages"]
        _insert_system_prompt(messages, system_prompt)
        repaired = repair_history(messages)
        if repaired:
            logger.info(f"Answered {repaired} tool calls that had no result before the LLM call.")

        tools = toolset.visible()
        if len(tools) > MAX_BOUND_TOOLS and len(tools) not in warned_counts:
            warned_counts.add(len(tools))
            logger.warning(
                f"{len(tools)} tools are bound, more than {MAX_BOUND_TOOLS}; "
                "a small model may pick the wrong one."
            )
        # Bound on every call, so a tool added during a task is callable on
        # the next turn.
        bound = llm.bind_tools(tools) if tools else llm

        # The reminder is sent but not saved. It is a human message because
        # some providers reject a system message that is not the first one.
        reminder = todos.render()
        outgoing = [*messages, HumanMessage(content=reminder)] if reminder else messages
        ai_message = invoke_llm_with_tracing(bound, outgoing, config)  # type: ignore[arg-type]
        messages.append(ai_message)

    def gate_node(state: ReActAgentState) -> None:
        open_ids = ", ".join(todos.open_ids())
        state["messages"].append(
            HumanMessage(
                content=(
                    f"{GATE_PREFIX}: {open_ids}. "
                    "Finish them, or mark them failed or cancelled with a note."
                )
            )
        )

    def route_after_llm(state: ReActAgentState) -> str:
        messages = state["messages"]
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        # Calls with unreadable arguments never reach ToolRunner. Going back
        # to llm lets its history repair answer them, so the LLM sees the
        # error and can try again; ending here or going to the gate would
        # leave them unanswered. Invalid calls next to valid ones go through
        # the tools node and are answered on the next LLM call.
        if isinstance(last, AIMessage) and last.invalid_tool_calls:
            return "llm"
        # While an item waits for an answer, the gate would push the LLM to
        # guess instead of waiting.
        if (
            todos.open_ids()
            and not todos.has_waiting()
            and _gates_since_task_start(messages) < MAX_GATES
        ):
            return "gate"
        return END

    def route_after_tools(state: ReActAgentState) -> str:
        # The question was spoken, so the answer arrives as the next message.
        return END if _person_was_asked(state["messages"]) else "llm"

    graph = StateGraph(ReActAgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", WalkieToolRunner(toolset, todos))
    graph.add_node("gate", gate_node)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", route_after_llm, ["tools", "llm", "gate", END])
    graph.add_conditional_edges("tools", route_after_tools, ["llm", END])
    graph.add_edge("gate", "llm")
    return graph.compile()


def _insert_system_prompt(
    messages: list[BaseMessage], system_prompt: str | SystemMultimodalMessage | None
) -> None:
    """Puts the system prompt first once, as RAI's llm_node does."""
    if isinstance(system_prompt, SystemMultimodalMessage):
        prompt: SystemMessage = system_prompt
    elif system_prompt:
        prompt = SystemMessage(content=system_prompt)
    else:
        return
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, prompt)


def _is_gate_message(message: BaseMessage) -> bool:
    return isinstance(message.content, str) and message.content.startswith(GATE_PREFIX)


def _gates_since_task_start(messages: list[BaseMessage]) -> int:
    """How many gate messages follow the human message that started the task.

    Counted from the history rather than kept in the graph, so the count also
    holds when the graph is invoked directly. The task starts at the last
    human message that is neither a gate message nor the message with a
    tool's images that RAI's ToolRunner adds after the tool results (that one
    carries the tool's call id).
    """
    count = 0
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        if _is_gate_message(message):
            count += 1
        elif not getattr(message, "tool_call_id", None):
            break
    return count


def _person_was_asked(messages: list[BaseMessage]) -> bool:
    """True when the last AI message called wait_for_person and the call succeeded.

    A failed call, such as when speech is down, means the person never heard
    the question, so the run goes on.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, AIMessage):
            break
    else:
        return False
    asked = {call.get("id") for call in message.tool_calls if call["name"] == "wait_for_person"}
    return any(
        isinstance(result, ToolMessage)
        and result.tool_call_id in asked
        and result.status == "success"
        for result in messages[index + 1 :]
    )
