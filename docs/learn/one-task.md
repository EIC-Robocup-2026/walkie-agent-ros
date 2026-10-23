# One Task, Start to Finish

This page follows one unseeded task through the graph and the todo list, from
the first message to the cleared list. The tool results are made up; the
agent's messages and routes are what the code does. The person says: *"Go to
the kitchen, find my cup, then come back and tell me."*

Turns 1 and 2 are one graph run that ends with a question, and turn 3 is a
second run started by the answer. The code blocks replay the list's side with
the calls the agent makes: `apply` for `update_todos`, `record_tool_result`
for the tools node, and `set_waiting` and `resume_waiting` for the question.
A listener prints the console lines, as `_send_status_lines` does.

## Turn 1: the plan and the first step

1. **`_run_agent` takes the message.** It has no seed, so `_prepare_todos`
   finds nothing waiting and nothing open, and tries `match_template`. No
   template matches, so the text goes to the graph unchanged.
2. **`llm` runs with no reminder,** because the list is empty. The model
   replies with two calls in one message: `update_todos` adding t1 as
   `in_progress` plus t2 and t3, then `navigate_to_location("kitchen")`.
3. **`tools` runs them in order.** The drive succeeds, so its call id becomes
   t1's evidence. The ledger ignores the `update_todos` result.

```pycon
>>> from walkie_agent.todo import TodoList, TodoRejected, console_line
>>> todos = TodoList()
>>> def console(items, changed):
...     for item in changed:
...         if line := console_line(item):
...             print(line)
...
>>> todos.add_listener(console)
>>> print(todos.render() or "(no reminder)")
(no reminder)
>>> print(todos.apply([
...     {"op": "add", "content": "Go to the kitchen",
...      "active_form": "Going to the kitchen", "status": "in_progress"},
...     {"op": "add", "content": "Find the cup", "active_form": "Finding the cup"},
...     {"op": "add", "content": "Come back and tell the person", "active_form": "Coming back"},
... ]))
t1: Going to the kitchen
Added: t1, t2, t3. Open: t1, t2, t3.
>>> todos.record_tool_result("update_todos", "call_1", ok=True)
>>> todos.record_tool_result("navigate_to_location", "call_2", ok=True)
>>> todos.snapshot()[0].evidence
['call_2']

```

## Turn 2: a question

1. **`route_after_tools` goes back to `llm`, which now sends the reminder.**
   The model marks t1 `completed` and t2 `in_progress`, and calls
   `find_objects("cup")`. Completing t1 passes, because its window holds one
   success and no failure.
2. **Two cups are remembered,** so the model calls
   `wait_for_person("Is yours the red cup or the blue one?")`. The question is
   spoken, t2 becomes `waiting`, and `route_after_tools` ends the run.

```pycon
>>> print(todos.render())
<todos>
[>] t1 Go to the kitchen
[ ] t2 Find the cup
[ ] t3 Come back and tell the person
</todos>
>>> print(todos.apply([
...     {"op": "update", "id": "t1", "status": "completed"},
...     {"op": "update", "id": "t2", "status": "in_progress"},
... ]))
t1 completed: Go to the kitchen
t2: Finding the cup
Updated: t1 → completed, t2 → in_progress. Open: t2, t3.
>>> todos.record_tool_result("find_objects", "call_4", ok=True)
>>> todos.set_waiting("Is yours the red cup or the blue one?")
t2 waiting: Find the cup
't2'
>>> todos.record_tool_result("wait_for_person", "call_5", ok=True)

```

## Turn 3: the answer, a failure and a refusal

1. **"The red one" arrives while the agent is idle.** `_prepare_todos` calls
   `resume_waiting`, so t2 is back `in_progress` with the same window. Items
   are open, so the message belongs to this task and no template is tried.
2. **The model completes t2, starts t3 and drives to `start`.** Nav2 rejects
   the goal. The tool's `ToolException` becomes a result with status `error`,
   and the ledger records a failure.
3. **`apply` refuses to mark t3 `completed`.** The model retries the drive,
   and that later success resolves the failure. It calls `speak`, then
   completes t3.

```pycon
>>> todos.resume_waiting()
t2: Finding the cup
>>> print(todos.apply([
...     {"op": "update", "id": "t2", "status": "completed"},
...     {"op": "update", "id": "t3", "status": "in_progress"},
... ]))
t2 completed: Find the cup
t3: Coming back
Updated: t2 → completed, t3 → in_progress. Open: t3.
>>> todos.record_tool_result("navigate_to_location", "call_7", ok=False)
>>> try:
...     todos.apply([{"op": "update", "id": "t3", "status": "completed"}])
... except TodoRejected as e:
...     print(e)
t3 cannot be completed because navigate_to_location failed and has not succeeded since; retry it, or add a note that explains why the step is done anyway.
>>> todos.record_tool_result("navigate_to_location", "call_9", ok=True)
>>> todos.record_tool_result("speak", "call_10", ok=True)
>>> print(todos.apply([{"op": "update", "id": "t3", "status": "completed"}]))
t3 completed: Come back and tell the person
Updated: t3 → completed. Open: none.

```

The model sees more than `print(e)`: `UpdateTodosTool` adds "None of the
changes were applied.", and RAI's `ToolRunner` prefixes "Failed to run tool.
Error: " in a result with status `error`.

## The end

1. **The model replies with plain text and no tool calls.** Nothing is open,
   so `route_after_llm` ends the run.
2. **The next message clears the finished list.** `_prepare_todos` sees a
   list with no open items and calls `clear()` before the LLM sees it.

Had the model stopped with t3 still open, the gate would have sent it back
with "[todo] Open items: t3. ...". After two gates the run ends with t3 open,
which shows on `/walkie/todos`.

!!! example "Try it"
    Replay the blocks up to the failed drive. Instead of retrying, run
    `todos.apply([{"op": "update", "id": "t3", "status": "completed", "note": "The person came to me."}])`.
    A note may stand in for a retry; `print(todos.render())` shows it. The
    rules are on [The Rules: Ledger and Window](todo-rules.md).

Previous: [The Graph: llm, tools and gate](the-graph.md) · Next: [Todo Items and Ops](todo-items.md)
