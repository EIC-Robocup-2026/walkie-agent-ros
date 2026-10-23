# Learn the Agent

This section explains how the Walkie agent works, one idea per page. Each
page takes five to ten minutes and has an example you can run. Read the pages
in order the first time; later, use them as a reference.

The pages cover phase 1 of the redesign: the agent's own graph and its todo
list. For the full reference, see [The Agent](../architecture/agent.md) and
its ADRs.

## The short version

Phase 1 replaced RAI's `ReActAgent` with a small graph that Walkie owns, and
gave the agent a todo list for each task.

- **RAI still runs the agent.** RAI's `LangChainAgent` owns the agent
  thread, the message buffer and the connectors. Only the graph inside it,
  and how messages are taken from the buffer, are Walkie's.
- **The graph has three nodes:** `llm`, `tools` and a completion `gate`. The
  gate sends the LLM back when it stops with steps still open, at most twice
  per task.
- **The todo list is plain Python** in `src/walkie_agent/todo.py`, with no
  ROS. The LLM edits it with `update_todos` and sees it on every LLM call, as
  a reminder that is sent but never saved.
- **The rules live in code, not in the prompt.** One step is in progress at
  a time. Failed and cancelled steps need a note. Seeded steps cannot be
  reworded. A step that requires a tool cannot be completed without a
  successful call to that tool.
- **Tasks can come with their steps.** An operator seeds them on
  `/walkie/task`, or a template seeds them from a keyword. Progress shows on
  `/walkie/todos` and the console.

## The pages

| # | Page | What you learn |
| - | ---- | -------------- |
| 1 | [RAI Runs It, We Own the Graph](rai-and-our-graph.md) | Which parts come from RAI and which are Walkie's |
| 2 | [What Changed from the Old Agent](old-vs-new.md) | The old `ReActAgent` and what phase 1 changed |
| 3 | [The Graph: llm, tools and gate](the-graph.md) | The three nodes and every route between them |
| 4 | [One Task, Start to Finish](one-task.md) | A three-turn task traced through the graph and the list |
| 5 | [Todo Items and Ops](todo-items.md) | `TodoItem`, statuses, `TodoOp` and the reminder |
| 6 | [The Rules: Ledger and Window](todo-rules.md) | The rules enforced in code, and how tool failures block "done" |
| 7 | [Asking a Question](asking-questions.md) | `wait_for_person` and how the answer resumes the step |
| 8 | [Queue, Stop, Reset and Repair](queue-stop-reset.md) | What happens to speech mid-task, stops, resets and errors |
| 9 | [Seeding a Task](seeding.md) | Giving the agent its steps from Python, ROS or a template |
| 10 | [Watching Progress](watching-progress.md) | `/walkie/todos`, console lines and listeners |
| 11 | [Read the Code Next](read-the-code.md) | Which files to read, in what order, and exercises |

## Run the examples

Finish [Getting Started](../getting-started.md) first, so `pixi install` has
run. Then start Python from the repository root:

```bash
pixi run python
```

Type the lines after `>>>` and `...`; the lines without a prompt are what
Python prints. The examples need no robot, no simulator and no LLM key. Each
page's examples continue from the previous block on the same page.

The examples are tested. `tests/test_learn_docs.py` runs every page as a
doctest, so `pixi run test` fails when the code changes and an example no
longer matches it. If you change the code, update the page in the same pull
request.

Next: [RAI Runs It, We Own the Graph](rai-and-our-graph.md)
