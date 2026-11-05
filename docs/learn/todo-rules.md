# The Rules: Ledger and Window

In a task with several steps, an LLM tends to drop a step, to call the task done once it has
spoken, or to report success after a tool failed. So the rules are in code, and the model
can't talk its way past them. A refused call changes nothing and tells the model what to do
instead.

## Setup

`attempt` is the helper from the last page. It runs one `update_todos` worth of ops on a
draft of the list, and prints the answer.

```pycon
>>> from walkie_agent.runtime import TodoRejected, parse_seed
>>> from walkie_agent.runtime.todos import TodoOp, _Draft
>>> def attempt(items, *ops, next_id=1, ledger=()):
...     draft = _Draft(items, next_id, list(ledger))
...     try:
...         print(draft.apply([TodoOp(**op) for op in ops]))
...     except TodoRejected as error:
...         print("Refused:", error)
...     return draft
...

```

## Rules on the list

One step at a time, because physical work is serial:

```pycon
>>> items = attempt([], {"op": "add", "content": "Drive", "status": "in_progress"}, {"op": "add", "content": "Speak"}).items
Added: t1, t2. Open: t1, t2.
>>> _ = attempt(items, {"op": "update", "id": "t2", "status": "in_progress"}, next_id=3)
Refused: Only one item can be in progress or waiting at a time, but t1 and t2 would be; finish the others or set them back to pending.

```

A failed or cancelled step says why:

```pycon
>>> _ = attempt(items, {"op": "update", "id": "t1", "status": "cancelled"}, next_id=3)
Refused: Marking t1 cancelled needs a note that says why.
>>> _ = attempt(items, {"op": "update", "id": "t1", "status": "cancelled", "note": "The door is shut."}, next_id=3)
Updated: t1 → cancelled. Open: t2.

```

Seeded steps are the operator's plan, so the model can't reword them, remove them or move
one past another. It can add steps of its own. And `before_agent` is what gives a seed its
`added` and `since`, so this block does the same:

```pycon
>>> seed = [{**item, "added": 0, "since": 0} for item in parse_seed(["Fetch the mug", "Bring it back"])]
>>> _ = attempt(seed, {"op": "update", "id": "t1", "content": "Fetch a cup"}, next_id=3)
Refused: t1 is a seeded step, so its wording cannot change.
>>> _ = attempt(seed, {"op": "remove", "id": "t2"}, next_id=3)
Refused: t2 is a seeded step, so it cannot be removed; mark it cancelled with a note instead.
>>> _ = attempt(seed, {"op": "update", "id": "t1", "after": "t2"}, next_id=3)
Refused: Seeded steps keep their order, so one cannot move past another.
>>> _ = attempt(seed, {"op": "add", "content": "Open the door", "after": "t1"}, next_id=3)
Added: t3. Open: t1, t3, t2.

```

## The error ledger

Each call a tool actually runs adds one entry to `ledger`, an additive part of the state.
This is what the list checks when a step completes:

```pycon
>>> failed = {"tool": "go_to", "id": "c1", "ok": False}
>>> spoke = {"tool": "say", "id": "c2", "ok": True}
>>> drove = {"tool": "go_to", "id": "c3", "ok": True}
>>> done = {"op": "update", "id": "t1", "status": "completed"}
>>> step = attempt([], {"op": "add", "content": "Drive", "status": "in_progress"}).items
Added: t1. Open: t1.
>>> _ = attempt(step, done, next_id=2, ledger=[failed])
Refused: t1 cannot be completed because go_to failed and has not succeeded since; retry it, or add a note that explains why the step is done anyway.

```

Speaking doesn't fix a failed drive. A later success of the same tool does:

```pycon
>>> _ = attempt(step, done, next_id=2, ledger=[failed, spoke])
Refused: t1 cannot be completed because go_to failed and has not succeeded since; retry it, or add a note that explains why the step is done anyway.
>>> _ = attempt(step, done, next_id=2, ledger=[failed, spoke, drove])
Updated: t1 → completed. Open: none.

```

A note can excuse a failure, so the model can say the step is done another way:

```pycon
>>> _ = attempt(step, {**done, "note": "The person came to me instead."}, next_id=2, ledger=[failed])
Updated: t1 → completed. Open: none.

```

## The window and requires_tool

The ledger is the whole task's history, and two positions on each step say which part
counts for it. `since` opens the failure window, and setting `in_progress` again moves it.
So a new attempt starts with a clean slate:

```pycon
>>> restart = attempt(step, {"op": "update", "id": "t1", "status": "pending"}, next_id=2, ledger=[failed]).items
Updated: t1 → pending. Open: t1.
>>> again = attempt(restart, {"op": "update", "id": "t1", "status": "in_progress"}, next_id=2, ledger=[failed]).items
Updated: t1 → in_progress. Open: t1.
>>> _ = attempt(again, done, next_id=2, ledger=[failed])
Updated: t1 → completed. Open: none.

```

`added` opens the other window. A step with `requires_tool` needs a successful call of that
tool since the step was added. A note can't excuse a missing requirement, because a note
excuses failures, and this is a thing that never happened:

```pycon
>>> two = [
...     {**item, "added": 0, "since": 0}
...     for item in parse_seed([
...         {"content": "Go to the table", "requires_tool": "go_to"},
...         {"content": "Go to the shelf", "requires_tool": "go_to"},
...     ])
... ]
>>> _ = attempt(two, {"op": "update", "id": "t1", "status": "completed", "note": "I am sure."}, next_id=3)
Refused: t1 needs a successful go_to before it can be completed.

```

Each result completes one step only. One drive can't finish two steps, whatever the model
claims:

```pycon
>>> first = attempt(two, {"op": "update", "id": "t1", "status": "completed"}, next_id=3, ledger=[drove])
Updated: t1 → completed. Open: t2.
>>> first.items[0]["evidence"]
['c3']
>>> _ = attempt(first.items, {"op": "update", "id": "t2", "status": "completed"}, next_id=3, ledger=[drove])
Refused: t2 needs a successful go_to before it can be completed.

```

## Failed and never run

A call that was refused before it ran is not a failure of the robot, so it must not block a
step. `StopLatch` and `SerialGuard` sit outside `WalkieTodos`, so their answers never reach
the ledger. An unknown tool never ran either. The real stack shows it:

```pycon
>>> import asyncio
>>> from langchain_core.language_models import BaseChatModel
>>> from langchain_core.messages import AIMessage, HumanMessage
>>> from langchain_core.outputs import ChatGeneration, ChatResult
>>> from walkie_agent.capabilities.fake import FakeRobot
>>> from walkie_agent.runtime import Session, build_agent
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
>>> script = [
...     AIMessage("", tool_calls=[call("go_to", "c1", place="garage"), call("say", "c2", text="Hello.")]),
...     AIMessage("", tool_calls=[call("fly", "c3")]),
...     AIMessage("I could not do it."),
... ]
>>> agent = build_agent(Scripted(script=script), FakeRobot().ctx)
>>> result = asyncio.run(agent.ainvoke({"messages": [HumanMessage("Go.")]}, Session().config()))
>>> [message.content.split(":")[0] for message in result["messages"] if message.type == "tool"]
['NOT_FOUND', 'Not run', 'Error']
>>> result["ledger"]
[{'tool': 'go_to', 'id': 'c1', 'ok': False}]

```

`go_to` ran and failed, so it's the one entry. `say` and `fly` left no trace in the ledger.

!!! example "Try it"
    Read `test_a_call_that_never_ran_does_not_poison_completion` in
    `tests/test_runtime_todos.py`. What would go wrong if `SerialGuard` sat inside
    `WalkieTodos`?

Previous: [Todo Items and Ops](todo-items.md) · Next: [Asking a Question](asking-questions.md)
