# Todo Items and Ops

The todo list is a list of `TodoItem` steps that the LLM changes with
`TodoOp` edits and reads back as a `<todos>` reminder on every LLM call. All
of it lives in `src/walkie_agent/todo.py`, which imports no ROS, so you can
try it in a plain Python shell. The rules that refuse a bad edit come on the
next page.

## TodoItem

Each step of the current task is one `TodoItem`, a pydantic model.

| Field | Meaning |
| --- | --- |
| `id` | `t1`, `t2` and so on, assigned by the list and stable for the whole task |
| `content` | The step, such as "Go to the kitchen" |
| `active_form` | Its "-ing" line for the console, such as "Going to the kitchen"; defaults to `content` |
| `status` | `pending`, `in_progress`, `waiting`, `completed`, `failed` or `cancelled` |
| `note` | Why the step failed, was cancelled, or is done despite a tool failure |
| `source` | `agent` when the LLM added it; `seed` or `operator` when it was seeded |
| `locked` | `True` for seeded items: status and note may change, wording may not |
| `requires_tool` | A tool that must succeed before the item can complete |
| `evidence` | Call ids of the successful tool calls made while the item was active |
| `question` | The question asked while the item is `waiting` |

`TodoList.apply` takes a list of ops. Each op is a `TodoOp` or a plain dict,
the shape the LLM's JSON arguments arrive in. `to_json()` gives the JSON
array that `/walkie/todos` publishes.

```pycon
>>> from walkie_agent.todo import TodoList, TodoOp, TodoRejected
>>> todos = TodoList()
>>> todos.apply([
...     TodoOp(op="add", content="Go to the kitchen",
...            active_form="Going to the kitchen", status="in_progress"),
...     {"op": "add", "content": "Find the cup"},
...     {"op": "add", "content": "Come back and tell the person"},
... ])
'Added: t1, t2, t3. Open: t1, t2, t3.'
>>> import json
>>> t2 = json.loads(todos.to_json())[1]
>>> t2["active_form"], t2["source"], t2["locked"]
('Find the cup', 'agent', False)

```

## Statuses and markers

`render()` shows each status as a short marker, so a small model can scan the
list at a glance. Pending, in-progress and waiting items are open.

| Marker | Status | Open | Set by |
| --- | --- | --- | --- |
| `[ ]` | `pending` | Yes | A new item, or an op |
| `[>]` | `in_progress` | Yes | An op, or `resume_waiting` when an answer arrives |
| `[?]` | `waiting` | Yes | Only `wait_for_person`, through `set_waiting` |
| `[x]` | `completed` | No | An op, after the ledger checks pass |
| `[!]` | `failed` | No | An op with a note |
| `[-]` | `cancelled` | No | An op with a note |

One call can finish t1 and start t2. Here `set_waiting` stands in for
`wait_for_person`, which [Asking a Question](asking-questions.md) covers.

```pycon
>>> todos.apply([
...     {"op": "update", "id": "t1", "status": "completed"},
...     {"op": "update", "id": "t2", "status": "in_progress"},
... ])
'Updated: t1 → completed, t2 → in_progress. Open: t2, t3.'
>>> todos.set_waiting("Is yours the red cup or the blue one?")
't2'
>>> print(todos.render())
<todos>
[x] t1 Go to the kitchen
[?] t2 Find the cup  waiting: "Is yours the red cup or the blue one?"
[ ] t3 Come back and tell the person
</todos>

```

## TodoOp: one flat shape for every op

Every op has the same fields: `op` (`add`, `update` or `remove`), `id`,
`content`, `active_form`, `status`, `note` and `after`. An empty string means
"not given". A union of three op types, or `Optional` fields, would become
`anyOf` in the tool schema, which Gemini and small Ollama models handle
poorly ([ADR-0041](../architecture/decisions/0041-todo-list-and-seeding.md)).

The list also cleans each op before checking it. A validator on `TodoOp`
reads `null` as `""`, and `_parse_op` squashes every text field to one trimmed
line and lowercases the status. A newline in a note cannot break the list.

```pycon
>>> todos.resume_waiting()
>>> todos.apply([
...     {"op": "update", "id": "t2", "status": "Failed", "content": None,
...      "note": "No red cup\n   in the kitchen"},
...     {"op": "update", "id": "t3", "status": "cancelled",
...      "note": "The person came to the kitchen"},
... ])
'Updated: t2 → failed, t3 → cancelled. Open: none.'
>>> print(todos.render())
<todos>
[x] t1 Go to the kitchen
[!] t2 Find the cup  note: "No red cup in the kitchen"
[-] t3 Come back and tell the person  note: "The person came to the kitchen"
</todos>
>>> try:
...     todos.apply([{"op": "finish", "id": "t1"}])
... except TodoRejected as e:
...     print(e)
An op is malformed at 'op': Input should be 'add', 'update' or 'remove'.

```

## What the LLM gets back

A good call returns one sentence: what was added, updated and removed, then
the open ids. An update that keeps the status shows the id without an arrow.
A bad call raises `TodoRejected`, a `ValueError` whose message tells the LLM
what to do instead.

`UpdateTodosTool` in `src/walkie_agent/tools/todos.py` wraps `apply` for the
LLM. On a refusal it raises `ToolException` and adds "None of the changes were
applied.", because `apply` is all or nothing. RAI's `ToolRunner` turns the
exception into a `ToolMessage` with status `error`.

```pycon
>>> from langchain_core.tools import ToolException
>>> from walkie_agent.tools.todos import UpdateTodosTool
>>> tool = UpdateTodosTool(todos=todos)
>>> tool.invoke({"ops": [{"op": "add", "content": "Say goodbye", "after": "t1"}]})
'Added: t4. Open: t4.'
>>> tool.invoke({"ops": [{"op": "update", "id": "t4", "note": "Wave as well"}]})
'Updated: t4. Open: t4.'
>>> try:
...     tool.invoke({"ops": [{"op": "update", "id": "t4", "status": "failed"}]})
... except ToolException as e:
...     print(e)
Marking t4 failed needs a note that says why. None of the changes were applied.

```

## The reminder

Before each LLM call, the `llm` node in `src/walkie_agent/graph.py` appends
`render()` as one `HumanMessage` after the history. It is a human message
because some providers reject a system message that is not the first one. It
goes on the outgoing list only, so it is never saved in `state["messages"]`.
An empty list renders as `''`, and then no reminder is sent.

!!! example "Try it"
    Continue the session above. Reword t1 and t4 with update ops that set only
    `content`, then print `active_form` for each item in `todos.snapshot()`.
    Only t4's `active_form` follows the new wording. Find the line in
    `_Draft._update` in `src/walkie_agent/todo.py` that decides this.

Previous: [One Task, Start to Finish](one-task.md) · Next: [The Rules: Ledger and Window](todo-rules.md)
