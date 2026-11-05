# ADR-0069: Pick a task template only by its exact name on /walkie/task

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-05

**Context**

[ADR-0041](/architecture/decisions/0041-todo-list-and-seeding.md) proposed templates
"chosen by keyword or by name". A stray word in speech could then start the wrong plan,
and at a noisy booth that is likely. A typo in a message could also silently drop a seed.

**Decision**

- The operator picks a template by exact name, on `/walkie/task`:
  `{"task": "...", "template": "tour"}`. Case and spaces count.
- Speech never starts a template. `/from_human` only ever carries a task or a bare stop.
- `/walkie/task` is a `std_msgs/String` that holds strict JSON. An unknown key, both or
  neither of `todos` and `template`, an unknown template or a bad step is logged and
  dropped, and the topic keeps working.
- `config/task_templates.toml` is strict as well. A bad table, or a `requires_tool` that
  the running tool set does not have, stops the agent at startup.
- The shipped file holds no templates.

This partly supersedes ADR-0041, for its selection part only. Seeded steps, evidence and
public progress stay as ADR-0041 decided.

**Consequences**

- Starting a template takes an operator or a script that knows the name.
- A person cannot start a fixed task by voice alone.
- Only templates have their `requires_tool` checked against the tool set. An inline seed
  on `/walkie/task` is not, so a step can name a tool that the running set lacks, and
  that step can then never complete.
- Evidence that ran: unit tests of both formats, and integration tests that publish on
  the topic with real rclpy and a fake robot.

**See also:** [The Agent: seeding](/architecture/agent.md#seeding-a-task-on-walkietask),
[Seeding](/learn/seeding.md)
<!-- --8<-- [end:body] -->
