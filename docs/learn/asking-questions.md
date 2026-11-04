# Asking a Question

When the agent can't go on without an answer, it calls `wait_for_person(question)` instead
of `say`. A successful question ends the run, and the answer arrives as the next task. It
then resumes the waiting step, exactly once.

## What the tool does

`WalkieTodos` supplies the tool, next to `update_todos`. It does four things, in order:

1. It checks the cancel token. A stop refuses the question.
2. It makes the question one trimmed line, so it can't break the rendered list.
3. It calls `ctx.speech.say`, which returns `None` once the line is published, or raises
   `StepError`. So `Asked:` means the line was sent, and not that anyone heard it. There is
   no playback acknowledgement.
4. It answers `Asked: ...`, and sets `question` in the state.

```pycon
>>> import asyncio
>>> from langchain_core.language_models import BaseChatModel
>>> from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
>>> from langchain_core.outputs import ChatGeneration, ChatResult
>>> from walkie_agent.capabilities.fake import FakeRobot
>>> from walkie_agent.errors import ErrorCode, StepError
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
>>> def answers(result):
...     return [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
...
>>> start = call("update_todos", "u1", ops=[{"op": "update", "id": "t1", "status": "in_progress"}])
>>> ask = call("wait_for_person", "q1", question="Which   mug\ndo you mean?")
>>> model, robot = Scripted(script=[ai(start), ai(ask)]), FakeRobot()
>>> agent = build_agent(model, robot.ctx)
>>> config = Session().config()
>>> payload = {"messages": [HumanMessage("Fetch the mug.")], "todo_seed": parse_seed(["Fetch the mug"])}
>>> result = asyncio.run(agent.ainvoke(payload, config))
>>> answers(result)[-1]
'Asked: Which mug do you mean?'
>>> robot.said
['Which mug do you mean?']
>>> len(model.seen)
2

```

The model was not called again after the question. That's the whole point of the tool.

## The graph ends the run

`EndAfterQuestion.before_model` sees `question` in the state and jumps to the end. Just
before that, `WalkieTodos.before_model` folds the step in progress into `waiting`. So the
list changes at one boundary, not in the tool:

```pycon
>>> print(render(result["todos"]))
<todos>
[?] t1 Fetch the mug  waiting: "Which mug do you mean?"
</todos>

```

The other calls of that message don't run, because they must wait for the answer:

```pycon
>>> robot = FakeRobot()
>>> script = [ai(call("wait_for_person", "q1", question="Which one?"), call("go_to", "c1", place="sofa"))]
>>> agent = build_agent(Scripted(script=script), robot.ctx)
>>> done = asyncio.run(agent.ainvoke({"messages": [HumanMessage("Fetch it.")]}, Session().config()))
>>> answers(done)
['Asked: Which one?', 'Not run: you asked the person a question, so the task stopped to wait for the answer. Call it again after the answer if it is still needed.']
>>> [entry["verb"] for entry in robot.calls]
['speech.say']

```

Only a successful question ends the run. If the line can't be sent, the model gets the error
and decides again, and the run goes on. That's why the tool has no `return_direct`, which
would end the run whatever the status:

```pycon
>>> robot = FakeRobot()
>>> robot.faults["speech.say"] = StepError(ErrorCode.UNAVAILABLE, "Nothing listens.")
>>> script = [ai(call("wait_for_person", "q1", question="Which one?")), ai(text="I cannot ask.")]
>>> model = Scripted(script=script)
>>> done = asyncio.run(build_agent(model, robot.ctx).ainvoke({"messages": [HumanMessage("Fetch it.")]}, Session().config()))
>>> answers(done)
['UNAVAILABLE: Nothing listens.']
>>> len(model.seen), done.get("question")
(2, None)

```

## The answer resumes the step

The next task's text is the answer. `before_agent` sees the pending `question`, and sets the
waiting step back to `in_progress`. It keeps the step's window and the task's gate budget,
because this is the same attempt. Then it clears `question`:

```pycon
>>> finish = call("update_todos", "u2", ops=[{"op": "update", "id": "t1", "status": "completed"}])
>>> first = Scripted(script=[ai(start), ai(ask), ai(finish), ai(text="Got it.")])
>>> agent = build_agent(first, FakeRobot().ctx)
>>> config = Session().config()
>>> _ = asyncio.run(agent.ainvoke(payload, config))
>>> resumed = asyncio.run(agent.ainvoke({"messages": [HumanMessage("The red one.")]}, config))
>>> print(first.seen[2][-1].content)
<todos>
[>] t1 Fetch the mug
</todos>
>>> first.seen[2][-2].content
'The red one.'

```

It resumes exactly once. The question is gone now, so the next message has nothing to answer.
It's a new task, and it starts with an empty list:

```pycon
>>> resumed.get("question") is None
True
>>> first.script.append(ai(text="Hello."))
>>> after = asyncio.run(agent.ainvoke({"messages": [HumanMessage("Something else.")]}, config))
>>> after["todos"]
[]

```

A seed wins over a pending question, since the operator started a new task.
`test_a_seed_wins_over_a_pending_question` shows that.

!!! example "Try it"
    Read `test_a_waiting_question_does_not_make_the_next_task_look_like_its_answer` in
    `tests/test_runtime_runner.py`. A reset or a stop between question and answer changes
    what the next message means. How?

Previous: [The Rules: Ledger and Window](todo-rules.md) · Next: [Queue, Stop, Reset and Repair](queue-stop-reset.md)
