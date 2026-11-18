# Seeding a Task

A fixed task can start with its steps already on the list. That's a seed. Seeded steps keep
their wording and order, so the LLM follows the plan and can't rewrite it. An
unseeded task starts with an empty list, and the LLM writes its own.

## Three ways to seed

| From | How | `source` |
| ---- | --- | -------- |
| Python | `runner.submit("Do it.", todos=[...])` | `seed` |
| ROS, with the steps | `{"task": "...", "todos": [...]}` on `/walkie/task` | `operator` |
| ROS, with a template's name | `{"task": "...", "template": "name"}` on `/walkie/task` | `operator` |

All three end at `Runner.submit`, and `parse_seed` checks the steps there. So a bad seed
raises in the caller, before anything is queued. Each entry is a string, or a dict with
`content` and optionally `active_form` and `requires_tool`:

```pycon
>>> from walkie_agent.runtime import TodoRejected, parse_seed, render
>>> seed = parse_seed(["Go to the table", {"content": "Say hello", "requires_tool": "say"}], "operator")
>>> print(render(seed))
<todos>
[ ] t1 Go to the table
[ ] t2 Say hello  needs: say
</todos>
>>> [(item["source"], item["locked"]) for item in seed]
[('operator', True), ('operator', True)]

```

The refusals say what's wrong. An unknown key is most likely a typo, and it would silently
drop a check if the code let it through:

```pycon
>>> for bad in [[{"content": "Go", "require_tool": "go_to"}], [""], "Go to the table", [3]]:
...     try:
...         parse_seed(bad)
...     except TodoRejected as error:
...         print(error)
...
Seed item t1 has unknown keys: require_tool.
Seed item t1 has no content.
A seed is a list of items, not a single string.
Seed item t1 must be a string or a dict, not int.

```

`parse_seed` returns new items, so the queued seed is a copy. The caller may change its own
list afterwards. `test_a_queued_seed_is_a_copy_and_keeps_its_source` holds this.

## When a seed applies

`submit` puts the seed in the task's input as `todo_seed`. `WalkieTodos.before_agent` reads
it when the task's run starts. So it's the task's first step, and a queued seed waits its
turn with the task.

| Input | What happens to the list |
| ----- | ------------------------- |
| A seed | Replaces the list. Ids start again from `t1`, and the windows and the gate budget are reset |
| `todos=[]` | That's a seed too, an empty one. The list is emptied |
| `todos=None`, with a question pending | Resumes the waiting step |
| `todos=None`, with nothing pending | A new task. The list is cleared |

Here's the difference between a seed and no seed, on the real stack:

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
>>> # The model stops talking each time, and the gate sends it back at most twice.
>>> model = Scripted(script=[AIMessage(f"Reply {n}.") for n in range(9)])
>>> agent = build_agent(model, FakeRobot().ctx)
>>> config = Session().config()
>>> def run(text, seed=None):
...     payload = {"messages": [HumanMessage(text)], "todo_seed": seed}
...     return asyncio.run(agent.ainvoke(payload, config))
...
>>> [item["id"] for item in run("Do it.", parse_seed(["One", "Two"]))["todos"]]
['t1', 't2']
>>> print(model.seen[0][-1].content)
<todos>
[ ] t1 One
[ ] t2 Two
</todos>
>>> run("Something else.")["todos"]
[]
>>> [item["id"] for item in run("Again.", parse_seed(["Three"]))["todos"]]
['t1']
>>> run("Nothing to do.", parse_seed([]))["todos"]
[]

```

## Operator tasks on /walkie/task

`/walkie/task` is a `std_msgs/String` holding JSON. `parse_task_message` reads it, with no
ROS. `RosBridge` hands the result to `Runner.submit` with `source="operator"`. A bad message
is logged and refused. It never queues half a task:

```pycon
>>> from walkie_agent.runtime.task_templates import parse_task_message
>>> parse_task_message('{"task": "Fetch the mug.", "todos": ["Go to the table", "Pick it up"]}', {})
('Fetch the mug.', ['Go to the table', 'Pick it up'])
>>> parse_task_message('{"task": "Just do it.", "todos": []}', {})
('Just do it.', [])
>>> parse_task_message('{"task": "Both.", "todos": [], "template": "tour"}', {})
Traceback (most recent call last):
    ...
ValueError: The task message is malformed: Value error, needs exactly one of todos and template.
>>> parse_task_message('{"task": "Neither."}', {})
Traceback (most recent call last):
    ...
ValueError: The task message is malformed: Value error, needs exactly one of todos and template.
>>> parse_task_message('{"task": "Typo.", "items": []}', {})
Traceback (most recent call last):
    ...
ValueError: The task message is malformed: ...

```

## Templates

A template is a fixed task in `config/task_templates.toml`. The file ships with none. An
operator picks one by its exact name, and nothing matches keywords, so a stray word in
speech can't start the wrong plan ([ADR-0069](../architecture/decisions/0069-task-templates-by-name.md)).

```pycon
>>> import pathlib, tempfile
>>> from walkie_agent.runtime.task_templates import check_required_tools, load_task_templates
>>> folder = tempfile.TemporaryDirectory()
>>> path = pathlib.Path(folder.name) / "templates.toml"
>>> _ = path.write_text('''
... [templates.tour]
... task = "Show the person around, and say one sentence at each place."
... todos = ["Go to the sofa", {content = "Say something", requires_tool = "say"}]
... ''')
>>> templates = load_task_templates(path)
>>> list(templates)
['tour']
>>> text, todos = parse_task_message('{"task": "Please give a tour.", "template": "tour"}', templates)
>>> print(text)
Please give a tour.
<BLANKLINE>
Show the person around, and say one sentence at each place.
>>> [item["content"] for item in parse_seed(todos)]
['Go to the sofa', 'Say something']
>>> parse_task_message('{"task": "Tour?", "template": "Tour"}', templates)
Traceback (most recent call last):
    ...
ValueError: There is no task template 'Tour'. Known templates: tour.

```

A malformed template stops the agent at startup. So does a step that requires a tool outside
the running tool set, because such a step could never complete:

```pycon
>>> check_required_tools(templates, {"say", "go_to"})
>>> check_required_tools(templates, {"go_to"})
Traceback (most recent call last):
    ...
ValueError: Task templates require tools this tool set lacks: tour.t2 requires say. Available: go_to.
>>> folder.cleanup()

```

The failure is loud on purpose. A template that can never finish should stop the run at
startup, not in front of a referee.

!!! example "Try it"
    Open `config/task_templates.toml`. Read its commented example, then load the file with
    `load_task_templates`. What does it return, and why?

Previous: [Queue, Stop, Reset and Repair](queue-stop-reset.md) · Next: [Watching Progress](watching-progress.md)
