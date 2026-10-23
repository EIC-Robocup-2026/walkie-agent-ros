# What Changed from the Old Agent

Before phase 1, `WalkieAgent` was RAI's `ReActAgent` with a reset added, so
nearly all of its behaviour came from rai-core 2.12. Phase 1 keeps RAI's
runner and changes what happens inside a task and between tasks. Most changes
trace back to three causes: the graph is Walkie's, messages queue, and tool
results carry a real status.

## What Walkie added to ReActAgent

The old agent is the code at commit `1c694ca`, the last one on `main` before
phase 1. Read it with `git show 1c694ca:src/walkie_agent/agent.py`.

- `reset()` clears the message deque, then sets `_reset_requested` and
  `_interrupt_event`. `subscribe_reset` calls it for every `std_msgs/Empty`
  on `/walkie/reset`.
- An `_run_agent` override clears `state["messages"]` when a reset is
  pending, then calls RAI's `_run_agent`.
- `_start_session()` makes a Langfuse session id such as
  `walkie-20261023-141502-a3f9`, so one conversation is one Langfuse session.

## What the old agent could not do

- **New speech interrupted the task.** With `"interrupt_keep_last"`, any
  utterance stopped the task at the next graph step, and only the newest
  queued message ran. Overheard speech at a noisy venue could cancel a drive.
- **No plan.** The only record of a task was the chat history. Nothing
  stopped the LLM from dropping a step or saying "done" early.
- **Failures looked like successes.** A tool returned failure text, such as
  "Navigation to kitchen failed: Nav2 rejected the goal.", as a normal
  result. Its `ToolMessage` had status `success`.
- **An error killed the agent.** RAI's `_run_agent` has only `try`/`finally`.
  An LLM or graph exception ended the agent thread, so the robot stopped
  listening.
- **Interrupts could poison the history.** A stop between `llm` and `tools`
  left an AI message whose tool calls had no results. OpenAI-compatible
  endpoints reject that history on the next call.

## Old vs new, side by side

|  | Old agent (before phase 1) | New agent (phase 1) |
| --- | --- | --- |
| Base class | `ReActAgent` | `LangChainAgent` running Walkie's graph |
| Graph | RAI's `llm` and `tools` | `llm`, `tools` and `gate`; the run also ends after `wait_for_person` |
| Tool binding | Once, when the graph is built | On every LLM call, from the live `ToolSet` |
| Speech during a task | Interrupts it; only the newest queued message runs | Waits in the queue; only "stop", "never mind" or a reset interrupts |
| Plan | None | `TodoList`, edited with `update_todos`, rules enforced in code |
| Plan seen by the LLM | Only through the chat history | A `<todos>` reminder on every call while the list has items, never saved |
| Tool failure | Failure text in a `success` result | `ToolException`, so an `error` result that the ledger records |
| LLM or graph exception | Ends the agent thread | Logged; the agent serves the next message |
| Tool calls cut off by a stop | Left unanswered in the history | Answered by `repair_history` before the next LLM call |
| Asking a question | `speak`, and the run goes back to the LLM | `wait_for_person` ends the run; the answer resumes the step |
| Seeding a task | Not possible | `submit(text, todos)`, `/walkie/task` or a template |
| Progress for the operator | LLM replies on the console | Also `/walkie/todos` and one console line per status change |
| Reset | Clears the queue and the history | Also clears the todo list, which publishes `[]` |
| Step limit per run | LangGraph's default, 10,007 in the locked 1.2.14 | 100, from `RECURSION_LIMIT` |

!!! note "Phase 1 lowered the step limit"
    Older LangGraph releases stopped a run after 25 steps. The LangGraph in
    `pixi.lock`, 1.2.14, allows 10,007, or `LANGGRAPH_DEFAULT_RECURSION_LIMIT`
    when it is set, so the old agent had no practical limit. Now a looping
    run ends after 100 steps, with an error that `_run_agent` logs.

## Replay three rows

RAI's `_apply_reduction_behavior` shows what each setting takes from the
buffer. Here three utterances wait while the robot is busy.
`WalkieAgent._run_agent` takes messages the way `"queue"` does.

```pycon
>>> from collections import deque
>>> from rai.agents.langchain import LangChainAgent
>>> from rai.communication.hri_connector import HRIMessage
>>> heard = ["go to the kitchen", "nice weather today", "where is the bathroom"]
>>> def take(behavior):
...     buffer = deque(HRIMessage(text=t, message_author="human") for t in heard)
...     taken = LangChainAgent._apply_reduction_behavior(behavior, buffer)
...     return [m.text for m in taken], [m.text for m in buffer]
...
>>> take("interrupt_keep_last")
(['where is the bathroom'], [])
>>> take("queue")
(['go to the kitchen'], ['nice weather today', 'where is the bathroom'])

```

A failed drive now raises `ToolException`. Walkie's tools set
`handle_tool_error = True`, so LangChain turns the exception into an `error`
result with the same sentence:

```pycon
>>> from langchain_core.tools import ToolException, tool
>>> @tool
... def old_drive(location: str) -> str:
...     """Drive to a named place, the way the old tool did."""
...     return f"Navigation to {location} failed: Nav2 rejected the goal."
...
>>> @tool
... def new_drive(location: str) -> str:
...     """Drive to a named place, the way phase 1's tool does."""
...     raise ToolException(f"Navigation to {location} failed: Nav2 rejected the goal.")
...
>>> new_drive.handle_tool_error = True
>>> for drive in (old_drive, new_drive):
...     call = {"name": drive.name, "args": {"location": "kitchen"}, "id": "c1", "type": "tool_call"}
...     result = drive.invoke(call)
...     print(result.status, "|", result.content)
...
success | Navigation to kitchen failed: Nav2 rejected the goal.
error | Navigation to kitchen failed: Nav2 rejected the goal.

```

A call cut off by a stop gets an `error` answer before the next LLM call.
Running the repair again adds nothing:

```pycon
>>> from langchain_core.messages import AIMessage, HumanMessage
>>> from walkie_agent.graph import repair_history
>>> drive_call = {"name": "navigate_to_location", "args": {"location": "kitchen"}, "id": "c2"}
>>> history = [HumanMessage("Go to the kitchen"), AIMessage("", tool_calls=[drive_call])]
>>> repair_history(history)
1
>>> print(history[-1].status, "|", history[-1].content)
error | Interrupted before it ran.
>>> repair_history(history)
0

```

## What did not change

The HRI connectors and topics stay: `/from_human` in, `/walkie/console` out,
and speech to `/to_human` through `speak`. So do the speak and navigation
rules in `SYSTEM_PROMPT`, `/walkie/reset` and the Langfuse session ids. The
`FORBIDDEN_NAMES` guard rail stays too, and gains `/walkie/task` and
`/walkie/todos` so the agent's own tools cannot publish to its control topics.

!!! example "Try it"
    Run `git diff 1c694ca -- src/walkie_agent/tools/navigation.py` and find
    every `return` of failure text that became `raise ToolException`. Then
    pick one row of the table above and find the line in
    `src/walkie_agent/agent.py` or `src/walkie_agent/graph.py` that makes it
    true.

Previous: [RAI Runs It, We Own the Graph](rai-and-our-graph.md) · Next: [The Graph: llm, tools and gate](the-graph.md)
