# Queue, Stop, Reset and Repair

A message that arrives during a task waits its turn. Only a bare "stop", a bare "never
mind" or a reset interrupts the task. Around that queue, the agent survives its own errors
and repairs the history. Most of these rules exist because the robot is physical: a drive
can't be undone, and a venue is noisy.

Everything here runs on a scripted model and `FakeRobot`. Stop latency on the robot has not
been measured.

## The queue and "stop"

`Runner.hear` takes an utterance from any thread. `is_stop_utterance` decides what counts
as a stop. It lowercases the text, trims spaces and `. , ! ?` from both ends, and collapses
inner spaces. The result must be exactly `stop`, `never mind` or `nevermind`:

```pycon
>>> from walkie_agent.runtime import is_stop_utterance
>>> for text in ["Stop.", "  NEVER   mind! ", "nevermind?", "Walkie, stop", "stop the music", "never-mind"]:
...     print(repr(text), is_stop_utterance(text))
...
'Stop.' True
'  NEVER   mind! ' True
'nevermind?' True
'Walkie, stop' False
'stop the music' False
'never-mind' False

```

Anything else is a task, and it waits in the queue. Overheard booth speech can't cancel a
drive that way. A correction like "stop and come here" goes to the LLM as the next task.

The setup is the usual scripted model, plus a `Held` version that waits on an event, so the
page can do something while a model call is in flight:

```pycon
>>> import asyncio, threading
>>> from typing import Any
>>> from langchain_core.language_models import BaseChatModel
>>> from langchain_core.messages import AIMessage, ToolMessage
>>> from langchain_core.outputs import ChatGeneration, ChatResult
>>> from walkie_agent.capabilities.fake import FakeRobot
>>> from walkie_agent.runtime import Runner, build_agent
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
...         reply = self.script.pop(0)
...         if isinstance(reply, Exception):
...             raise reply
...         return ChatResult(generations=[ChatGeneration(message=reply.model_copy(update={"id": None}))])
...
>>> class Held(Scripted):
...     entered: Any = None
...     release: Any = None
...     async def _agenerate(self, messages, *args, **kwargs):
...         self.entered.set()
...         await asyncio.to_thread(self.release.wait, 5)
...         return await super()._agenerate(messages, *args, **kwargs)
...
>>> def call(name, call_id, **args):
...     return {"name": name, "args": args, "id": call_id, "type": "tool_call"}
...
>>> def setup(model):
...     robot, replies = FakeRobot(), []
...     runner = Runner(build_agent(model, robot.ctx), robot.ctx, on_reply=replies.append)
...     runner.start()
...     return robot, runner, replies
...
>>> robot, runner, replies = setup(Scripted(script=[AIMessage("First."), AIMessage("Second.")]))
>>> runner.hear("Do one thing.")
>>> runner.hear("Do another.")
>>> runner.wait_idle(5)
True
>>> replies
['First.', 'Second.']
>>> runner.shutdown(5)
True

```

Now a stop that lands while the model is thinking, with another task queued. A stop does
four things:

1. It sets `ctx.cancel` at once, on the caller's thread. A drive that a verb is waiting on
   ends through this token.
2. It drops the queued tasks. A queued reset stays.
3. It speaks `Okay, stopping.` right away, on a worker thread. The line doesn't wait for a
   hung LLM call.
4. It queues a `stopping` graph pass, which cancels the open steps and makes no model call.

```pycon
>>> entered, release = threading.Event(), threading.Event()
>>> go = AIMessage("", tool_calls=[call("go_to", "c1", place="sofa")])
>>> model = Held(script=[go], entered=entered, release=release)
>>> robot, runner, replies = setup(model)
>>> runner.submit("Go to the sofa.", todos=["Go to the sofa"])
>>> entered.wait(5)
True
>>> runner.hear("Then say hi.")
>>> runner.hear("Stop.")
>>> robot.ctx.cancel.cancelled
True
>>> release.set()
>>> runner.wait_idle(5)
True
>>> robot.said
['Okay, stopping.']
>>> [entry["verb"] for entry in robot.calls if entry["verb"] == "base.go_to"]
[]
>>> len(model.seen)
1

```

The model's reply was on its way when the stop came, and its drive never ran. "Then say
hi." never reached the model. The list and the history show what happened:

```pycon
>>> state = runner.agent.get_state(runner.session.config()).values
>>> [(item["status"], item["note"]) for item in state["todos"]]
[('cancelled', 'Stopped by the person.')]
>>> [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
['Not run: the person said stop, so the task ended.']
>>> state["messages"][-1].content
'Okay, stopping.'

```

The latch holds until the next task is taken. Then the token resets, and that task runs as
usual:

```pycon
>>> model.script.append(AIMessage("Back at it."))
>>> runner.hear("Hello again.")
>>> runner.wait_idle(5)
True
>>> robot.ctx.cancel.cancelled, replies
(False, ['Back at it.'])

```

A bare stop works when nothing is running too. It speaks and makes no model call:

```pycon
>>> calls = len(model.seen)
>>> runner.hear("never mind")
>>> runner.wait_idle(5)
True
>>> robot.said, len(model.seen) == calls
(['Okay, stopping.', 'Okay, stopping.'], True)

```

## Reset, stop() and errors

`reset` forgets everything. It cancels, drops the queue, deletes the checkpoint thread after
the active run ends, starts a new session and publishes an empty list. The active run ends
first, because deleting a thread under a running checkpoint would race. A hung LLM call can
delay it by up to the deadline, and a stop line in flight by up to the say timeout.

```pycon
>>> before = runner.session.thread_id
>>> runner.reset()
>>> runner.wait_idle(5)
True
>>> runner.session.thread_id != before
True
>>> runner.agent.get_state(runner.session.config()).values
{}
>>> runner.shutdown(5)
True

```

A failed task doesn't end the worker. Python logs the traceback, so you'll see it in your
terminal, and the operator's console gets one fixed line. The next task runs on the same
checkpoint:

```pycon
>>> robot, runner, replies = setup(Scripted(script=[RuntimeError("The provider broke."), AIMessage("Fine now.")]))
>>> runner.hear("Try this.")
>>> runner.hear("Try again.")
>>> runner.wait_idle(5)
True
>>> replies
['The task failed with an error, so I stopped working on it.', 'Fine now.']

```

After `shutdown`, the Runner refuses more work, so a late caller gets a loud error:

```pycon
>>> runner.shutdown(5)
True
>>> runner.hear("One more.")
Traceback (most recent call last):
    ...
RuntimeError: The Runner has shut down.

```

## History repair

An OpenAI-compatible endpoint rejects a history where a tool call has no answer. A run that
fails or is cancelled can leave one, so `RepairHistory.before_agent` answers each with
`Interrupted before it ran.` It doesn't run the call. That drive might be stale by now.

A call whose arguments didn't parse never reaches the tools node, since LangChain files it
under `invalid_tool_calls`. `RepairHistory.after_model` answers those, and the model can
try again:

```pycon
>>> from langchain_core.messages import HumanMessage
>>> from walkie_agent.runtime import Session
>>> bad = {"name": "go_to", "args": '{"place": ', "id": "c1", "error": "Bad JSON", "type": "invalid_tool_call"}
>>> model = Scripted(script=[AIMessage("", invalid_tool_calls=[bad]), AIMessage("Sorry.")])
>>> robot = FakeRobot()
>>> result = asyncio.run(build_agent(model, robot.ctx).ainvoke({"messages": [HumanMessage("Go.")]}, Session().config()))
>>> [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
['The arguments were not a valid JSON object, so the call did not run.']
>>> robot.calls
[]

```

## One update_todos per message

A message can hold several calls, and they run in order. Only the first `update_todos` of a
message runs. A second one is a failed call like any other, so it skips what comes after it:

```pycon
>>> first = call("update_todos", "u1", ops=[{"op": "add", "content": "One"}])
>>> second = call("update_todos", "u2", ops=[{"op": "add", "content": "Two"}])
>>> model = Scripted(script=[AIMessage("", tool_calls=[first, second, call("say", "c1", text="Hi.")]), AIMessage("", tool_calls=[call("update_todos", "u3", ops=[{"op": "update", "id": "t1", "status": "cancelled", "note": "Done with it."}])]), AIMessage("Done.")])
>>> robot = FakeRobot()
>>> result = asyncio.run(build_agent(model, robot.ctx).ainvoke({"messages": [HumanMessage("Go.")]}, Session().config()))
>>> for message in result["messages"]:
...     if isinstance(message, ToolMessage) and message.tool_call_id in ("u1", "u2", "c1"):
...         print(message.tool_call_id, message.content)
...
u1 Added: t1. Open: t1.
u2 Only one update_todos call per message is accepted, so this one was not applied. Put all ops in one call.
c1 Not run: update_todos failed earlier in this message. Decide again.
>>> robot.said
[]

```

!!! example "Try it"
    Read `test_a_broken_run_is_repaired_and_its_stale_drive_stays_unrun` in
    `tests/test_runtime.py`. Why does repair answer the call and not run it?

Previous: [Asking a Question](asking-questions.md) · Next: [Seeding a Task](seeding.md)
