# One Task, Start to Finish

This page traces one seeded task through the real stack: the commands the model sends, a
question and its answer, a failed call, and a refusal. The model is scripted and the robot
is `FakeRobot`. So the point is what the code does with each reply, not how a real model
would choose them.

## The setup

The same scripted model as before, and small helpers. `show` prints each tool's answer.

```pycon
>>> import asyncio
>>> from langchain_core.language_models import BaseChatModel
>>> from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
>>> from langchain_core.outputs import ChatGeneration, ChatResult
>>> from walkie_agent.capabilities.fake import FakeRobot
>>> from walkie_agent.runtime import Session, build_agent, parse_seed, render
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
>>> def call(name, call_id, **args):
...     return {"name": name, "args": args, "id": call_id, "type": "tool_call"}
...
>>> def ai(*calls, text=""):
...     return AIMessage(text, tool_calls=list(calls))
...
>>> def update(call_id, *changes):
...     ops = [{"op": "update", "id": item, "status": status} for item, status in changes]
...     return call("update_todos", call_id, ops=ops)
...
>>> def show(result, start):
...     for message in result["messages"][start:]:
...         if isinstance(message, ToolMessage):
...             print(f"{message.name} ({message.status}): {message.content}")
...

```

The task has two seeded steps. The first requires a successful `go_to`.

```pycon
>>> seed = parse_seed([{"content": "Go to the table", "requires_tool": "go_to"}, "Greet the person"])
>>> print(render(seed))
<todos>
[ ] t1 Go to the table  needs: go_to
[ ] t2 Greet the person
</todos>

```

## Run 1: the commands

The model starts the first step, drives and completes the step. Then it starts the second,
and asks a question.

```pycon
>>> script = [
...     ai(update("u1", ("t1", "in_progress"))),
...     ai(call("go_to", "c1", place="table")),
...     ai(update("u2", ("t1", "completed"), ("t2", "in_progress"))),
...     ai(call("wait_for_person", "q1", question="What should I say?")),
... ]
>>> model, robot = Scripted(script=script), FakeRobot()
>>> agent = build_agent(model, robot.ctx)
>>> config = Session().config()
>>> payload = {"messages": [HumanMessage("Go to the table and greet the person.")], "todo_seed": seed}
>>> result = asyncio.run(agent.ainvoke(payload, config))
>>> show(result, 0)
update_todos (success): Updated: t1 → in_progress. Open: t1, t2.
go_to (success): Arrived at table.
update_todos (success): Updated: t1 → completed, t2 → in_progress. Open: t2.
wait_for_person (success): Asked: What should I say?

```

The run has ended, because a successful question ends it. The step shows as waiting, and
t1 kept the call that proved it:

```pycon
>>> print(render(result["todos"]))
<todos>
[x] t1 Go to the table
[?] t2 Greet the person  waiting: "What should I say?"
</todos>
>>> result["todos"][0]["evidence"]
['c1']
>>> robot.said
['What should I say?']

```

The evidence names the one `go_to` call that completed t1. The list took it from the
ledger, not from the model's word.

## Run 2: the answer, a failure and a refusal

The answer arrives as a new message. A question was pending, so it resumes t2 and keeps the
list. In this run the model first drives to a place that doesn't exist, then tries to
say hello in the same message. Next it tries to call the step done anyway. Then it
recovers.

```pycon
>>> model.script += [
...     ai(call("go_to", "c2", place="garage"), call("say", "c3", text="Hello.")),
...     ai(update("u3", ("t2", "completed"))),
...     ai(call("go_to", "c4", place="table"), call("say", "c5", text="Hello.")),
...     ai(update("u4", ("t2", "completed"))),
...     ai(text="I said hello."),
... ]
>>> start = len(result["messages"])
>>> result = asyncio.run(agent.ainvoke({"messages": [HumanMessage("Say hello.")]}, config))
>>> show(result, start)
go_to (error): NOT_FOUND: ...
say (error): Not run: go_to failed earlier in this message. Decide again.
update_todos (error): REFUSED: t2 cannot be completed because go_to failed and has not succeeded since; retry it, or add a note that explains why the step is done anyway. None of the changes were applied.
go_to (success): Arrived at table.
say (success): Said: Hello.
update_todos (success): Updated: t2 → completed. Open: none.

```

Read those answers in order:

1. `go_to` failed, and the error went back to the model as text.
2. `say` never ran. It was planned on the drive working, so `SerialGuard` skipped it.
3. The list refused to complete t2, because the failed `go_to` was in t2's window.
4. A later success of the same tool resolved the failure, so the next try went through.

The first model call of this run shows that the answer resumed t2. `[>]` is back, and the
reminder is the last thing the model got:

```pycon
>>> print(model.seen[4][-1].content)
<todos>
[x] t1 Go to the table
[>] t2 Greet the person
</todos>

```

## The end

The final text ends the run, since no step is open. It goes to the operator console, not to
the speaker. Only `say` and `wait_for_person` reach the person:

```pycon
>>> print(render(result["todos"]))
<todos>
[x] t1 Go to the table
[x] t2 Greet the person
</todos>
>>> result["messages"][-1].content
'I said hello.'
>>> robot.said
['What should I say?', 'Hello.']
>>> [entry["verb"] for entry in robot.calls if entry["verb"] == "base.go_to"]
['base.go_to', 'base.go_to', 'base.go_to']
>>> [entry["error"] for entry in robot.calls if entry["verb"] == "base.go_to"]
[None, 'NOT_FOUND', None]

```

The model's words never completed a step. On a real robot, `Arrived at table.` would rest
on a Nav2 result that has not been measured for this runtime yet. See
[What has been tested](../architecture/agent.md#what-has-been-tested).

!!! example "Try it"
    Change the last `update` in the script to complete t2 with the failed drive still
    unresolved, but with a note. `update` here has no note field, so write the op by hand.
    Then look at what `t2`'s `note` holds in `result["todos"]`.

Previous: [The Graph: create_agent and Middleware](the-graph.md) · Next: [Todo Items and Ops](todo-items.md)
