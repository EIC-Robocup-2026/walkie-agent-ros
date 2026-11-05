# Watching Progress

The runtime tells its listeners about each change to the list. In `main.py`, `RosBridge`
is that listener. It publishes the list on `/walkie/todos`, and a line for each change on
the console topic. The listener itself is plain Python, so this page needs no ROS.

## Listeners

`on_todos(items, changed)` gets the whole list, and the items whose status just changed.
Both hold public copies, so a listener can keep them or change them safely.

```pycon
>>> import asyncio, json
>>> from langchain_core.language_models import BaseChatModel
>>> from langchain_core.messages import AIMessage, HumanMessage
>>> from langchain_core.outputs import ChatGeneration, ChatResult
>>> from walkie_agent.capabilities.fake import FakeRobot
>>> from walkie_agent.runtime import Session, build_agent, console_line, parse_seed
>>> class Scripted(BaseChatModel):
...     script: list
...     seen: list = []
...     @property
...     def _llm_type(self):
...         return "scripted"
...     def bind_tools(self, tools, **kwargs):
...         return self
...     def _generate(self, *args, **kwargs):
...         raise NotImplementedError("The runtime only uses the async path.")
...     async def _agenerate(self, messages, *args, **kwargs):
...         self.seen.append(list(messages))
...         reply = self.script.pop(0).model_copy(update={"id": None})
...         return ChatResult(generations=[ChatGeneration(message=reply)])
...
>>> def todo(call_id, *ops):
...     return AIMessage("", tool_calls=[{"name": "update_todos", "args": {"ops": list(ops)}, "id": call_id, "type": "tool_call"}])
...
>>> def status(item, value, **fields):
...     return {"op": "update", "id": item, "status": value, **fields}
...
>>> script = [
...     todo("u1", status("t1", "in_progress")),
...     todo("u2", status("t1", "completed"), status("t2", "failed", note="Nobody was there.")),
...     AIMessage("One step done, one failed."),
... ]
>>> heard = []
>>> def on_todos(items, changed):
...     heard.append((items, changed))
...
>>> agent = build_agent(Scripted(script=script), FakeRobot().ctx, on_todos=on_todos)
>>> seed = parse_seed([{"content": "Go to the table", "active_form": "Going to the table"}, "Greet the person"])
>>> _ = asyncio.run(agent.ainvoke({"messages": [HumanMessage("Do it.")], "todo_seed": seed}, Session().config()))
>>> len(heard)
3

```

Three changes: the seed, the first update and the second. A change that moves no status
still publishes, with `changed` empty, and a refused call publishes nothing.

Listeners run on the Runner's thread, so they must not block. If one raises, the failure is
logged and the list is unaffected. A slow listener would delay the whole agent, so the
bridge only publishes and returns.

## Console lines

`console_line` writes one line for a step whose status just changed. A pending step has no
line. A step in progress uses its "-ing" form:

```pycon
>>> for items, changed in heard:
...     print([console_line(item) for item in changed])
...
['', '']
['t1: Going to the table']
['t1 completed: Go to the table', 't2 failed: Greet the person']

```

## The list on /walkie/todos

`RosBridge.on_todos` publishes `json.dumps(public(items))` as a `std_msgs/String`:

```pycon
>>> items, _ = heard[-1]
>>> print(json.dumps(items[1], indent=1))
{
 "id": "t2",
 "content": "Greet the person",
 "active_form": "Greet the person",
 "status": "failed",
 "note": "Nobody was there.",
 "source": "seed",
 "locked": true,
 "requires_tool": "",
 "evidence": [],
 "question": ""
}

```

The two output topics differ on purpose:

| Topic | What it carries | QoS | Why |
| ----- | ---------------- | --- | --- |
| `/walkie/todos` | The whole list, as JSON | Reliable, transient local, depth 1 | Latched, so a monitor that starts late gets the current list |
| `/walkie/console` | One line for each change, and the task's final text | Reliable, depth 10 | A log, read in order. Never spoken |
| `/to_human` | What `ctx.speech.say` says | Reliable, depth 10 | The speaker, through the separate `voice-tts` process |

The console is not the speaker. A plain-text reply goes to the console only. To reach a
person, the agent must call `say` or `wait_for_person`. The system prompt says so, and
`robot.said` here stays empty, whatever the model wrote:

```pycon
>>> robot = FakeRobot()
>>> agent = build_agent(Scripted(script=[AIMessage("I am telling the console, not the room.")]), robot.ctx)
>>> _ = asyncio.run(agent.ainvoke({"messages": [HumanMessage("Hello.")]}, Session().config()))
>>> robot.said
[]

```

A reset and a Runner start publish `[]`, so a monitor doesn't keep showing an old task.

## Wiring in main.py

`assemble` in `src/walkie_agent/main.py` connects these, in about this shape:

```python
bridge = RosBridge(connector, templates, ...)
agent = build_agent(model, ctx, on_todos=bridge.on_todos, ...)
runner = Runner(agent, ctx, on_todos=bridge.on_todos, on_reply=bridge.on_reply, ...)
```

Both `build_agent` and `Runner` take the same listener. The middleware reports changes
during a run, and the Runner reports the empty list at a start or a reset.

`tests/test_ros_bridge.py` checks that bridge with a fake connector.
`tests/test_ros_integration.py` runs it on real rclpy, against a fake robot. It has not run
on the robot's own graph.

!!! example "Try it"
    Make `on_todos` raise an exception, and rerun the first block. Does the run finish?
    Then read `test_a_listener_that_raises_does_not_affect_the_list`.

Previous: [Seeding a Task](seeding.md) · Next: [Read the Code Next](read-the-code.md)
