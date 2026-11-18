# Learn the Agent

This section explains how the Walkie agent works, one idea per page. Each
page takes five to ten minutes and has an example you can run. Read the pages
in order the first time; later, use them as a reference.

The pages cover the ROS-free runtime: LangChain's `create_agent`, Walkie's
middlewares, the todo list and the Runner. For the full reference, see
[The Agent](../architecture/agent.md) and its ADRs.

!!! note "What these examples prove"
    Every example runs on a scripted model and `FakeRobot`, with no ROS. They show how the
    code behaves, and they fail when the code changes. They say nothing about real Nav2, the
    simulator, the robot or a real model. [What has been tested](../architecture/agent.md#what-has-been-tested)
    lists the evidence, and its limits.

## The short version

The agent is `create_agent` with a fixed stack of middlewares, and a Runner that
owns its thread. RAI remains underneath it as the ROS connector and the voice stack.

- **RAI no longer hosts the agent.** `main.py` uses RAI's connector to reach ROS. Its
  agent host is gone from there, and only the voice entry points still use `AgentRunner`.
- **The graph comes from `create_agent`.** Its loop is a model node and a tools node.
  Each Walkie rule is one small middleware around that loop, and their order is pinned.
- **The todo list is plain data in the graph state.** The LLM edits it with
  `update_todos` and sees it on every LLM call, as a reminder that is sent but never saved.
- **The rules live in code, not in the prompt.** One step is in progress at a time.
  Failed and cancelled steps need a note. Seeded steps cannot be reworded. A step that
  requires a tool cannot be completed without a successful call to that tool.
- **Tasks can come with their steps.** An operator seeds them on `/walkie/task`, by
  giving the steps or naming a template exactly. Progress shows on `/walkie/todos` and
  the console.
- **Only the Runner's thread touches the agent.** ROS callbacks queue work and return.
  A stop sets the cancel token at once, from any thread.

## The pages

| # | Page | What you learn |
| - | ---- | -------------- |
| 1 | [RAI's Connector, Our Runtime](rai-and-our-graph.md) | Which parts come from RAI, which are Walkie's, and who owns which thread |
| 2 | [The Graph: create_agent and Middleware](the-graph.md) | The loop, the middleware order and the limits on a run |
| 3 | [One Task, Start to Finish](one-task.md) | A task traced through the graph and the list |
| 4 | [Todo Items and Ops](todo-items.md) | Items, statuses, `TodoOp` and the reminder |
| 5 | [The Rules: Ledger and Window](todo-rules.md) | The rules enforced in code, and how tool failures block "done" |
| 6 | [Asking a Question](asking-questions.md) | `wait_for_person` and how the answer resumes the step |
| 7 | [Queue, Stop, Reset and Repair](queue-stop-reset.md) | What happens to speech mid-task, stops, resets and errors |
| 8 | [Seeding a Task](seeding.md) | Giving the agent its steps from Python, ROS or a template |
| 9 | [Watching Progress](watching-progress.md) | Listeners, console lines and `/walkie/todos` |
| 10 | [Read the Code Next](read-the-code.md) | The package map, where to change what, which files to read, in what order, and exercises |

## Run the examples

Finish [Getting Started](../getting-started.md) first, so `pixi install` has
run. Then start Python from the repository root:

```bash
pixi run python
```

Type the lines after `>>>` and `...`; the lines without a prompt are what
Python prints. The examples need no robot, no simulator, no ROS graph and no LLM key. Each
page's examples continue from the previous block on the same page.

The examples are tested. `tests/test_learn_docs.py` runs every page as a
doctest, so `pixi run test` fails when the code changes and an example no
longer matches it. It also fails if a page loses all its examples. If you change the code,
update the page in the same pull request.

Next: [RAI's Connector, Our Runtime](rai-and-our-graph.md)
