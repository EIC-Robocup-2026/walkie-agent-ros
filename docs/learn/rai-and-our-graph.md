# RAI Runs It, We Own the Graph

RAI's `LangChainAgent` runs the Walkie agent, but the graph it runs is
Walkie's own. RAI owns the thread, the message buffer and the connectors.
Walkie owns the graph, from `build_walkie_graph` in `src/walkie_agent/graph.py`,
and the body of the run loop.

## The two layers

The runner is `LangChainAgent` in `rai/agents/langchain/agent.py`. It owns a
thread, a message deque guarded by `_buffer_lock`, and three events:
`_interrupt_event`, `_agent_ready_event` and `_stop_event`. RAI calls
`__call__(msg)` for every utterance, which appends it to the deque.

The thread's `_run_loop` calls `_run_agent` over and over. Each call takes
messages from the deque, appends one `HumanMessage` to `state["messages"]`
and streams the graph. Between graph steps it checks `_interrupt_event` and
breaks out if it is set.

```mermaid
flowchart LR
    hri["HRI connector<br/>/from_human"] --> call
    subgraph runner["RAI: LangChainAgent"]
        call["__call__(msg)"] --> buffer["message deque"]
        buffer --> loop["_run_loop, _run_agent"]
    end
    loop -- "stream(state)" --> nodes["Walkie's graph:<br/>llm, tools, gate"]
    nodes -- "LLM replies" --> console["/walkie/console"]
```

## RAI's graph

Before phase 1, the graph came from RAI's `ReActAgent`. Its `__init__` only
calls `create_react_runnable(llm, tools, system_prompt)` and hands the result
to `LangChainAgent`.

That graph has two nodes, `llm` and `tools`. LangGraph's stock
`tools_condition` goes to `tools` when the reply has tool calls and ends the
run otherwise; `tools` always goes back to `llm`. The tools are bound once,
while `create_react_runnable` builds the graph.

`ReActAgent` never passes `new_message_behavior`, so the old agent ran with
RAI's default:

```pycon
>>> import inspect
>>> from rai.agents.langchain import LangChainAgent, ReActAgent
>>> inspect.signature(LangChainAgent.__init__).parameters["new_message_behavior"].default
'interrupt_keep_last'
>>> "new_message_behavior" in inspect.signature(ReActAgent.__init__).parameters
False

```

## Why Walkie owns the graph

Walkie needs more on every LLM call than RAI's graph gives: the tools bound
at that moment, the todo list as a reminder, a history with no unanswered
tool calls, and routes for the `gate` and `wait_for_person`. None of that
fits through `ReActAgent`'s constructor
([ADR-0038](../architecture/decisions/0038-own-graph-per-turn-binding.md)).

So `build_walkie_graph` builds its own `StateGraph` on RAI's parts: the state
type `ReActAgentState`, `ToolRunner` and the tracing helper
`invoke_llm_with_tracing`. The biggest visible difference is when tools are
bound. A fake chat model that records each `bind_tools` call shows it:

```pycon
>>> from langchain_core.language_models.fake_chat_models import FakeListChatModel
>>> from langchain_core.messages import HumanMessage
>>> from langchain_core.tools import tool
>>> from rai.agents.langchain.core import create_react_runnable
>>> from walkie_agent.graph import ToolSet, build_walkie_graph
>>> from walkie_agent.todo import TodoList
>>> class CountingLLM(FakeListChatModel):
...     binds: list[list[str]] = []
...     def bind_tools(self, tools, **kwargs):
...         self.binds.append([t.name for t in tools])
...         return self
...
>>> @tool
... def speak(text: str) -> str:
...     """Say something out loud."""
...     return "Said."
...
>>> @tool
... def wave() -> str:
...     """Wave at the person."""
...     return "Waved."
...
>>> rai_llm = CountingLLM(responses=["Hello."])
>>> rai_graph = create_react_runnable(rai_llm, [speak])
>>> rai_llm.binds
[['speak']]
>>> walkie_llm = CountingLLM(responses=["Hello."])
>>> toolset = ToolSet([speak])
>>> walkie_graph = build_walkie_graph(walkie_llm, toolset, TodoList())
>>> walkie_llm.binds
[]
>>> state = {"messages": [HumanMessage("Hi")]}
>>> result = walkie_graph.invoke(state)
>>> toolset.add(wave)
>>> state["messages"].append(HumanMessage("Wave at me"))
>>> result = walkie_graph.invoke(state)
>>> walkie_llm.binds
[['speak'], ['speak', 'wave']]

```

RAI's graph bound `speak` once, before any message arrived. Walkie's `llm`
node binds `toolset.visible()` on every call, so `wave` is bound at once.

## What the agent class owns

`WalkieAgent` in `src/walkie_agent/agent.py` subclasses `LangChainAgent` and
passes it the graph. Besides the graph, it owns the parts below;
[The Agent](../architecture/agent.md) lists every component.

- **One shared `TodoList`.** `create_walkie_agent` passes the same object to
  the todo tools and to the graph.
- **A live `ToolSet`.** A thread-safe map of tools by name, read on every
  LLM call.
- **A queue instead of interrupts.** With `new_message_behavior="queue"`,
  utterances wait their turn, oldest first. Only `reset()`, or an utterance
  that is just "stop" or "never mind", interrupts a task.
- **Its own `__call__` and `_run_agent`.** Both replace RAI's, so a seed
  travels with its message and an error does not end the thread. `stop()`
  sets its events under the buffer lock, then calls RAI's `stop()`.
- **`RECURSION_LIMIT = 100`.** LangGraph ends a run with an error after
  100 graph steps, about 50 tool rounds.

Everything else, such as `run()` and the thread's `_run_loop`, is inherited:

```pycon
>>> from walkie_agent.agent import WalkieAgent
>>> [cls.__name__ for cls in WalkieAgent.__mro__]
['WalkieAgent', 'LangChainAgent', 'BaseAgent', 'ABC', 'object']
>>> walkie_attrs, rai_attrs = vars(WalkieAgent), vars(LangChainAgent)
>>> [name for name in walkie_attrs if name in rai_attrs and callable(walkie_attrs[name])]
['__init__', '__call__', 'stop', '_run_agent']
>>> agent = WalkieAgent(target_connectors={}, llm=FakeListChatModel(responses=["Done."]))
>>> agent.new_message_behavior
'queue'
>>> agent.agent.config["recursion_limit"]
100

```

!!! example "Try it"
    Read RAI's `_run_agent` in
    `.pixi/envs/default/lib/python3.12/site-packages/rai/agents/langchain/agent.py`
    next to `WalkieAgent._run_agent` in `src/walkie_agent/agent.py`. Find
    what Walkie's version adds: the pending reset at the top, the seed taken
    with the message, and the `except Exception` around the stream.

Previous: [Learn the Agent](index.md) · Next: [What Changed from the Old Agent](old-vs-new.md)
