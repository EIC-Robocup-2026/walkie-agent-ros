# Todo Items and Ops

The todo list is plain data in the graph state, so a checkpoint holds it. The LLM changes it
through one tool, `update_todos`, by sending a list of flat ops. Everything here is in
`src/walkie_agent/runtime/todos.py`, and it needs no ROS and no LLM.

## Items

An item is a plain dict, not a class, because the checkpoint stores it. `parse_seed` makes
one, and it shows every key:

```pycon
>>> from pprint import pprint
>>> from walkie_agent.runtime import console_line, parse_seed, public, render
>>> item = parse_seed([{"content": "Go to the shelf", "active_form": "Going to the shelf", "requires_tool": "go_to"}])[0]
>>> pprint(item, sort_dicts=False)
{'id': 't1',
 'content': 'Go to the shelf',
 'active_form': 'Going to the shelf',
 'status': 'pending',
 'note': '',
 'source': 'seed',
 'locked': True,
 'requires_tool': 'go_to',
 'evidence': [],
 'question': '',
 'used': []}

```

| Key | What it means |
| --- | -------------- |
| `id` | `t1`, `t2`, and so on. The list assigns it, and doesn't reuse it within a task |
| `content`, `active_form` | The step, and its "-ing" line for the console |
| `status`, `note` | Where it stands, and why it failed or was cancelled |
| `source`, `locked` | Who wrote it. A seeded step is locked, so its wording stays |
| `requires_tool`, `evidence` | The tool that must succeed, and the call that did |
| `question` | What the step is waiting on |

Three more keys are private: `added`, `since` and `used`. They hold positions in the error
ledger, as the [next page](todo-rules.md) explains. A seed gets the first two when it's
applied. `public` removes them, so they never reach the LLM or `/walkie/todos`:

```pycon
>>> sorted(public([item])[0])
['active_form', 'content', 'evidence', 'id', 'locked', 'note', 'question', 'requires_tool', 'source', 'status']

```

## Statuses and markers

A step moves between six statuses, and each has a marker in the reminder:

```pycon
>>> steps = [
...     {**item, "id": f"t{n}", "status": status, "requires_tool": ""}
...     for n, status in enumerate(["pending", "in_progress", "waiting", "completed", "failed", "cancelled"], 1)
... ]
>>> for line in render(steps).splitlines()[1:-1]:
...     print(line)
...
[ ] t1 Go to the shelf
[>] t2 Go to the shelf
[?] t3 Go to the shelf  waiting: ""
[x] t4 Go to the shelf
[!] t5 Go to the shelf
[-] t6 Go to the shelf

```

The LLM can set five of them. Only a successful `wait_for_person` sets `waiting`.

## TodoOp: one flat shape

`TodoOp` becomes the schema of `update_todos`. Every op has the same flat fields:

```pycon
>>> from walkie_agent.runtime.todos import TodoOp
>>> sorted(TodoOp.model_json_schema()["properties"])
['active_form', 'after', 'content', 'id', 'note', 'op', 'status']
>>> TodoOp.model_json_schema()["properties"]["status"]["type"]
'string'

```

The obvious design is a union of three op classes, with optional fields. That becomes
`anyOf` in the schema, and Gemini function declarations and small Ollama models handle
`anyOf` poorly. So each field is a plain string where empty means "not given", and the code
checks the rest. Some models fill unused fields with null, and text can hold line breaks:

```pycon
>>> op = TodoOp(op="add", content="  Go to\nthe   shelf ", status=None)
>>> repr(op.content), repr(op.status)
("'Go to the shelf'", "''")

```

## What the LLM gets back

`_Draft` is the working copy that one `update_todos` call edits. It's private, but it's the
simplest way to see the rules. `attempt` here prints its answer:

```pycon
>>> from walkie_agent.runtime import TodoRejected
>>> from walkie_agent.runtime.todos import _Draft
>>> def attempt(items, *ops, next_id=1, ledger=()):
...     draft = _Draft(items, next_id, list(ledger))
...     try:
...         print(draft.apply([TodoOp(**op) for op in ops]))
...     except TodoRejected as error:
...         print("Refused:", error)
...     return draft
...
>>> draft = attempt([], {"op": "add", "content": "Go to the shelf", "status": "in_progress"}, {"op": "add", "content": "Say hello"})
Added: t1, t2. Open: t1, t2.
>>> items, next_id = draft.items, draft.next_id
>>> draft = attempt(items, {"op": "update", "id": "t1", "status": "completed"}, {"op": "add", "content": "Come back"}, next_id=next_id)
Added: t3. Updated: t1 → completed. Open: t2, t3.

```

The sentence is short on purpose. The LLM needs the ids and what's still open.

A call is all or nothing. Here the first op is fine and the second breaks a rule, so
neither applies. `_Draft` works on a deep copy, so the old list is untouched:

```pycon
>>> draft = attempt(items, {"op": "update", "id": "t1", "status": "completed"}, {"op": "update", "id": "t2", "status": "failed"}, next_id=next_id)
Refused: Marking t2 failed needs a note that says why.
>>> [item["status"] for item in items]
['in_progress', 'pending']

```

The tool turns that refusal into `REFUSED: ... None of the changes were applied.`
The new list goes into the graph state only through the `Command` that a successful
call returns, because a checkpoint must never see half an edit.

## The reminder

Before each LLM call, `render` writes the list as a message after the history. It's a
`HumanMessage`, because some providers reject a system message that is not the first. It's
sent and never saved, so old copies can't pile up in the history or go stale. An empty list
sends nothing:

```pycon
>>> print(render(items))
<todos>
[>] t1 Go to the shelf
[ ] t2 Say hello
</todos>
>>> render([])
''

```

`test_every_call_gets_the_current_list_and_it_is_never_saved` proves both halves on the
real stack.

The console gets a different line, written by `console_line`. It's blank for a pending step,
and uses the "-ing" line for a step in progress:

```pycon
>>> for one in items:
...     print(repr(console_line(one)))
...
't1: Go to the shelf'
''

```

!!! example "Try it"
    Pass `{"op": "update", "id": "t2", "status": "waiting"}` to `attempt`, with `items`.
    Then pass `{"op": "update", "id": "t2"}`. Each refusal says what to do instead.

Previous: [One Task, Start to Finish](one-task.md) · Next: [The Rules: Ledger and Window](todo-rules.md)
