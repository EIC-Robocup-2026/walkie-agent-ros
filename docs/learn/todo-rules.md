# The Rules: Ledger and Window

`TodoList.apply` enforces the todo rules in code, so the LLM cannot talk its
way past them. It edits a `_Draft` copy and keeps it only if every op and the
final check pass, so one call is all or nothing. An error ledger of tool
results then decides when an item may become `completed`.

## Rules on the list

| Rule | Checked in |
| --- | --- |
| At most one item `in_progress` or `waiting` | `_Draft.check_one_active`, once after all ops |
| `failed` and `cancelled` need a note | `_check_status` |
| Seeded items keep their wording | `_Draft._update` |
| Seeded items cannot be removed | `_Draft._remove` |
| Only `wait_for_person` sets `waiting` | `_check_status` |
| Ids are assigned by the list, never chosen | `_Draft._add` |
| An update must change something | `_Draft._update` |

The reasons come from the robot: physical work is serial, the published list
must say why a step was dropped, and the LLM must not drop a seeded step
([ADR-0041](../architecture/decisions/0041-todo-list-and-seeding.md)).

The examples seed three steps, which locks them
([Seeding a Task](seeding.md) covers seeding). The helper `attempt` prints the
summary of a good call, or the refusal the LLM would read.

```pycon
>>> from walkie_agent.todo import TodoList, TodoRejected
>>> todos = TodoList()
>>> todos.seed([
...     "Go to the kitchen",
...     "Find the cup",
...     {"content": "Hand over the cup", "requires_tool": "hand_over"},
... ])
>>> def attempt(ops):
...     try:
...         print(todos.apply(ops))
...     except TodoRejected as e:
...         print(e)
...
>>> attempt([{"op": "update", "id": "t1", "status": "in_progress"},
...          {"op": "update", "id": "t2", "status": "in_progress"}])
Only one item can be in progress or waiting at a time, but t1 and t2 would be; finish the others or set them back to pending.
>>> attempt([{"op": "update", "id": "t1", "status": "failed"}])
Marking t1 failed needs a note that says why.
>>> attempt([{"op": "update", "id": "t1", "content": "Go to the living room"}])
t1 is a seeded step, so its wording cannot change.
>>> attempt([{"op": "remove", "id": "t1"}])
t1 is a seeded step, so it cannot be removed; mark it cancelled with a note instead.
>>> attempt([{"op": "update", "id": "t1", "status": "waiting"}])
Only wait_for_person can set waiting; call it to ask the person a question.
>>> attempt([{"op": "update", "id": "t1", "status": "pending"}])
The update to t1 changes nothing; leave it out or change a field.

```

A refused call changes nothing, even its valid ops. Below, the add is fine,
but the second op names an item that does not exist, so t4 is never created.

```pycon
>>> attempt([{"op": "add", "content": "Say goodbye"},
...          {"op": "update", "id": "t9", "status": "completed"}])
There is no item t9; use an id from the todo list.
>>> todos.open_ids()
['t1', 't2', 't3']
>>> attempt([{"op": "update", "id": "t1", "status": "in_progress"}])
Updated: t1 → in_progress. Open: t1, t2, t3.

```

## The error ledger

`WalkieToolRunner` in `src/walkie_agent/graph.py` calls
`record_tool_result(tool, call_id, ok)` for each result of a known tool. A
raised exception or a `ToolMessage` with status `error` counts as `ok=False`.
A success while an item is `in_progress` or `waiting` also becomes that item's
evidence.

Some results never reach the ledger. `update_todos` is ignored, because a
refused edit is a mistake in the list, not a failed step. Calls the runner
refuses itself are skipped too: an unknown tool, a second `update_todos` in
one message, and calls after a question.

Here you play the tools node. A failed drive blocks t1, and a later success
of the same tool resolves the failure.

```pycon
>>> todos.record_tool_result("navigate_to_location", "call_1", ok=False)
>>> attempt([{"op": "update", "id": "t1", "status": "completed"}])
t1 cannot be completed because navigate_to_location failed and has not succeeded since; retry it, or add a note that explains why the step is done anyway.
>>> todos.record_tool_result("navigate_to_location", "call_2", ok=True)
>>> attempt([{"op": "update", "id": "t1", "status": "completed"},
...          {"op": "update", "id": "t2", "status": "in_progress"}])
Updated: t1 → completed, t2 → in_progress. Open: t2, t3.
>>> todos.snapshot()[0].evidence
['call_2']

```

Only the same tool resolves a failure, and only if it succeeds after the
failure. The other way out is a note that explains why the step is done
anyway. The note is visible on `/walkie/todos`, so the claim is on record.

```pycon
>>> todos.record_tool_result("find_objects", "call_3", ok=False)
>>> attempt([{"op": "update", "id": "t2", "status": "completed",
...           "note": "The person pointed at the cup"}])
Updated: t2 → completed. Open: t3.

```

## The window and requires_tool

Each item's window is the part of the ledger that counts for it. It starts
when the item is created or seeded, and again whenever an op moves the item
into `in_progress` from another status. `resume_waiting` keeps the window,
because an answer continues the same attempt.

An item with `requires_tool` needs a success of that tool inside its window.
A note cannot stand in for it, so "Hand over the cup" needs a real
`hand_over`, not the LLM saying "here you go".

```pycon
>>> attempt([{"op": "update", "id": "t3", "status": "in_progress"}])
Updated: t3 → in_progress. Open: t3.
>>> todos.record_tool_result("speak", "call_4", ok=True)
>>> attempt([{"op": "update", "id": "t3", "status": "completed",
...           "note": "I said here you go"}])
t3 needs a successful hand_over before it can be completed.
>>> todos.record_tool_result("hand_over", "call_5", ok=True)
>>> attempt([{"op": "update", "id": "t3", "status": "completed"}])
Updated: t3 → completed. Open: none.

```

The last op needs no note, although `find_objects` failed and never
succeeded. That failure lies outside t3's window:

| Seq | Tool | Result | Active item | In t3's window |
| --- | --- | --- | --- | --- |
| 1 | `navigate_to_location` | failed | t1 | No |
| 2 | `navigate_to_location` | ok | t1 | No |
| 3 | `find_objects` | failed | t2 | No |
| 4 | `speak` | ok | t3 | Yes |
| 5 | `hand_over` | ok | t3 | Yes |

Seeding and `clear()` empty the ledger, so a new task starts clean. A retry
from `failed` back to `in_progress` also gets a new window, so an old failure
does not block the new attempt.

!!! example "Try it"
    Seed a fresh list with one step and start it. Record a failed
    `navigate_to_location`, mark the step `failed` with a note, then set it
    back to `in_progress` and complete it without a note. Find the line in
    `_Draft._update` in `src/walkie_agent/todo.py` that let the retry pass.

Previous: [Todo Items and Ops](todo-items.md) · Next: [Asking a Question](asking-questions.md)
