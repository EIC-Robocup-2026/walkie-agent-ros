# ADR-0038: Run our own graph on LangChainAgent and bind the visible tools every turn

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-22

**Context**

`WalkieAgent` subclasses RAI's `ReActAgent`
([ADR-0003](/architecture/decisions/0003-rai-react-agent.md)).
`ReActAgent` builds its graph with `create_react_runnable`, which calls
`llm.bind_tools(tools)` once, when the graph is compiled
(`rai/agents/langchain/core/react_agent.py` in rai-core 2.12.1). The
[self-evolving agent](/architecture/self-evolving-agent.md) needs more than
that:

- A tool found or created in the middle of a task must be callable on the
  next turn.
- The set of tools the LLM sees must change during a run: after a level
  demotion, while a turn is tainted by web text, and as session slots fill.
- At most 20 tools may be bound at once, so that the small local fallback
  model still works. The catalogue will hold many more.
- A todo reminder must reach the LLM on every turn without being saved in
  the history.

What RAI does with a message that arrives mid-task was checked in
`rai/agents/langchain/agent.py`. The default, `interrupt_keep_last`, sets
`_interrupt_event`. The stream loop checks it only between graph steps, so
a running tool call, a whole bundle included, finishes first. If the break
lands right after the LLM step, an AI message with tool calls stays in the
history with no tool results, and OpenAI-compatible endpoints reject that
on the next call.

Two hosts were compared:

- **Our own small LangGraph graph on RAI's `LangChainAgent`.** About 150
  lines. It keeps RAI's in-place state contract, so `reset()` and the
  existing tests keep working. It depends on private RAI members, under the
  existing `<2.13` pin.
- **LangChain's `create_agent` with middleware.** A larger ecosystem, with
  model fallback, tool retry and human-in-the-loop. It would need a
  checkpointer, an error middleware, serial tool calls, a history repair
  and a port of RAI's multimodal messages, and it risks losing history
  silently.

**Decision**

- `WalkieAgent` subclasses `LangChainAgent` instead of `ReActAgent`. It keeps
  today's constructor and builds its graph, `build_walkie_graph()`, inside
  it.
- The **llm node** inserts the system prompt once, computes the visible
  tools for this turn (filtered by the access level and the turn's taint)
  and binds them, repairs the history, and calls the model through RAI's
  `invoke_llm_with_tracing`. It sends the todo list as one trailing message
  that is not saved. After a demotion, that message also carries
  `[level] Running at Competition level: <reason>`. It appends the AI
  message to `state["messages"]` in place.
- The **tools node** is `WalkieToolRunner`, a subclass of RAI's
  `ToolRunner`. Its `tools_by_name` is a live map that asks the registry
  for a name it does not hold, and otherwise answers "Unknown tool X. Use
  find_tools." It reports every tool result's status to the todo list's
  error ledger. Serial execution and error handling are inherited.
- **Routes.** Tool calls go to the tools node. When the LLM calls no tool
  while todo items are open and none is waiting, a completion gate sends it
  back, at most twice per task. After `wait_for_person`, the run ends.
- **History repair.** Before every LLM call, each tool call without a
  result gets `ToolMessage(status="error", content="Interrupted before it
  ran.")`.
- Session state (the todo list, active tools, handles, the cancel token,
  the error ledger, the level and the taint flag) lives in plain Python
  objects owned by the agent, not in graph channels.
- `reset()` keeps what it does today, and also trips the cancel token,
  clears the session state, stops a running sandbox worker and a running
  tool coder, and flushes queued audio. Draft files survive it. At
  Competition, session tools survive it too, and clear only when the agent
  stops or on the new `/walkie/reset_session` topic.
- **New utterances mid-task queue.** `WalkieAgent` passes
  `new_message_behavior="queue"` to RAI, so an utterance that arrives during
  a task waits until the task ends. Only a reset, or an utterance that is
  just "stop" or "never mind", interrupts. `WalkieAgent` checks each
  utterance as it arrives; an interrupt sets the interrupt event and the
  cancel token, so it also aborts a running tool creation. The history
  repair still ships, because an interrupt can land right after the llm
  node.

Keeping RAI's default, `interrupt_keep_last`, was rejected. Corrections
would land at once, but at a noisy booth any overheard speech would cancel
the current step, so a 20 to 35 s tool creation would rarely finish.

**Amends ADR-0003.** The agent is still built on rai-core 2.12 under the
`<2.13` pin, with RAI's connectors, `HRIMessage`, `ToolRunner`, tracing and
speech. Only the graph changes: our own replaces `create_react_runnable`,
and the base class becomes `LangChainAgent`.

**Consequences**

- A tool activated by `find_tools` or `request_tool` is bound on the next
  turn, without rebuilding the agent.
- The bound set stays at 20 or fewer. Everything else is reached through
  `find_tools`.
- The static system prompt stays first, so prompt caching keeps working.
  The reminder is added only when the list is not empty, so the tests in
  `tests/test_agent.py` that check the exact messages sent to the LLM still
  pass.
- `WalkieAgent` overrides `_run_agent` in full, about 20 lines copied from
  RAI, to apply a seeded todo list
  ([ADR-0041](/architecture/decisions/0041-todo-list-and-seeding.md)). This,
  `_interrupt_event` and the `ToolRunner` internals must be rechecked on
  every rai-core upgrade.
- A correction such as "no, the red one" waits until the current task
  ends. Overheard speech can no longer abort a pick or a tool creation.
- RAI still runs one tool at a time, and a reset is still noticed only
  between steps. A bundle checks the cancel token between its own steps.
  [ADR-0023](/architecture/decisions/0023-motion-supervisor.md) remains the
  real fix for stopping motion.
- The tool coder is built with `create_agent` internally. That does not
  change the host.
- `create_agent` is worth revisiting when the project leaves rai-core 2.12.
- `ScriptedChatModel` in `tests/scripted_llm.py` must record the bound tool
  names and the trailing reminder on each call.

**See also:** [Self-Evolving Agent: the host graph](/architecture/self-evolving-agent.md#the-host-graph-and-interrupts),
`src/walkie_agent/agent.py`
<!-- --8<-- [end:body] -->
