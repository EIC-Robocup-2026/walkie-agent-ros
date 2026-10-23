# Watching Progress

Every change to the todo list is pushed to its listeners, and the agent
registers two. One publishes the whole list as JSON on `/walkie/todos`; the
other sends one line per status change to the operator console,
`/walkie/console`. Neither reaches the speaker.

## Listeners

`TodoList.add_listener(callback)` registers `callback(items, changed)`.
`items` is a snapshot of the new list. `changed` holds the items whose
status changed: every item for a seed, none for a clear or new evidence.

```pycon
>>> from walkie_agent.todo import TodoList, console_line
>>> todos = TodoList()
>>> def show(items, changed):
...     print(len(items), "items, changed:", [item.id for item in changed])
>>> todos.add_listener(show)
>>> todos.seed([{"content": "Go to the kitchen", "active_form": "Going to the kitchen"}, "Pick up the mug"])
2 items, changed: ['t1', 't2']
>>> todos.apply([{"op": "update", "id": "t1", "status": "in_progress"}])
2 items, changed: ['t1']
'Updated: t1 → in_progress. Open: t1, t2.'
>>> todos.record_tool_result("navigate_to_location", "call_1", ok=True)
2 items, changed: []

```

`TodoList` calls listeners outside its lock, so a listener may change the
list itself. Changes are queued, and one thread at a time drains the queue.
So an edit on the agent thread and a reset from a ROS callback reach the
listeners in the order they were made. A listener that raises is logged,
and the list is unaffected.

## Console lines

`console_line(item)` turns one changed item into one line. The agent's
listener, `WalkieAgent._send_status_lines`, sends each non-empty line to the
target connectors. This listener does the same with `print`:

```pycon
>>> def console(items, changed):
...     for item in changed:
...         if console_line(item):
...             print(console_line(item))
>>> todos.add_listener(console)
>>> _ = todos.apply([{"op": "update", "id": "t1", "status": "completed"},
...                  {"op": "update", "id": "t2", "status": "in_progress"}])
2 items, changed: ['t1', 't2']
t1 completed: Go to the kitchen
t2: Pick up the mug
>>> _ = todos.set_waiting("Which mug?")
2 items, changed: ['t2']
t2 waiting: Pick up the mug

```

| Status change | Console line |
| --- | --- |
| To `in_progress` | The active form: `t1: Going to the kitchen` |
| To `pending` | None, because a step that has not started is not news |
| To anything else | The content: `t1 completed: Go to the kitchen`, `t2 waiting: Pick up the mug` |

The lines go out as RAI's `HRICallbackHandler` sends the LLM's replies, so
they reach `/walkie/console` and never `/to_human`. One known limit: RAI
creates the console publisher on the first send, with depth 1. The first
line can be lost, and a burst of lines can drop some.

## The list on /walkie/todos

`WalkieAgent.publish_todos` publishes `TodoList.to_json()`, a JSON array of
every item with all its fields, as a `std_msgs/msg/String`. Here is the item
that is waiting:

```pycon
>>> import json
>>> print(json.dumps(json.loads(todos.to_json())[1], indent=2))
{
  "id": "t2",
  "content": "Pick up the mug",
  "active_form": "Pick up the mug",
  "status": "waiting",
  "note": "",
  "source": "seed",
  "locked": true,
  "requires_tool": "",
  "evidence": [],
  "question": "Which mug?"
}
>>> todos.clear()
0 items, changed: []
>>> todos.to_json()
'[]'

```

The publisher's QoS makes the topic latched:

| QoS setting | Value | What it gives a viewer |
| --- | --- | --- |
| Reliability | `RELIABLE` | A lost message is resent, so the last list arrives |
| Durability | `TRANSIENT_LOCAL` | A viewer that starts late still gets the last list |
| Depth | 1 | Only the current list is kept, never older ones |

`publish_todos` adds its listener first, then publishes once, so the topic
holds `[]` from startup. The listener ignores its snapshot and publishes the
current `to_json()`, so the last message is always the current list. A
reset runs `todos.clear()` on the agent thread, which publishes `[]`.

```bash
pixi run ros2 topic echo /walkie/todos std_msgs/msg/String --field data
```

`ros2 topic echo` asks for transient-local durability when every publisher
offers it, so it prints the current list even if started late. A viewer of
your own must ask for `TRANSIENT_LOCAL` itself.

## Wiring in main.py

`main.py` loads the templates from `--task-templates` before ROS starts,
then calls `subscribe_task` on `--task-topic` (default `/walkie/task`) and
`publish_todos` on `--todos-topic` (default `/walkie/todos`). Console lines
go to `--output-topic`, `/walkie/console`, the agent's target connector.

`FORBIDDEN_NAMES` in `src/walkie_agent/tools/__init__.py` lists
`/walkie/task` and `/walkie/todos`, so the agent's ROS tools cannot read or
write its own control plane. The list holds those default names, so a topic
renamed with a flag is not covered.

!!! example "Try it"
    Make a fresh `TodoList`, add the `console` listener, seed two steps and
    set `t1` to `in_progress`. Then call `todos.seed(...)` again with new
    steps. What does the console print, and why? Then read `publish_todos`
    in `src/walkie_agent/agent.py` and say why `publish` ignores its
    arguments.

Previous: [Seeding a Task](seeding.md) · Next: [Read the Code Next](read-the-code.md)
