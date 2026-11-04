# ADR-0067: Bind one fixed tool set when the agent is built

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-05

**Context**

[ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md) bound the tools
on every LLM call, so that a later phase could add tools during a task. That phase does
not exist. The live discovery behind it added state, tests and a way for one task's tools
to change under it. No current test needs it.

**Decision**

The agent binds one complete set at build time.

- `main.py` picks `DEFAULT`, `GPSR` or `HRI` with `--tool-set`. These are plain lists in
  `walkie_agent.atomics`.
- `ToolSet`, the live discovery and the dynamic slots are gone. A running agent cannot
  gain or lose a tool.
- `update_todos` and `wait_for_person` come from the `WalkieTodos` middleware and are
  bound along with the set.
- A duplicate name is refused at build, including a name that collides with a middleware
  tool.
- More than 20 bound tools logs a warning. The count includes the middleware tools.

This records the dynamic-binding part of ADR-0038 as superseded. It does not decide the
catalogue in proposed
[ADR-0046](/architecture/decisions/0046-hybrid-tool-index.md), which stays unbuilt.

**Consequences**

- The prompt, the schema and the tools are fixed for a process, so a run is easier to
  reproduce.
- Choosing another set means a restart.
- A generated or discovered tool would need a new decision, and work in the runtime.
- Evidence that ran: unit tests of the set contents and the refused duplicates. No real
  model has run with any set.
- A set is not a promise that its verbs work on the robot. `find_objects` is in `DEFAULT`
  and answers `UNAVAILABLE` there.

**See also:** [The Agent: tools](/architecture/agent.md#tools),
[ADR-0060](/architecture/decisions/0060-atomic-tools-and-adapter.md)
<!-- --8<-- [end:body] -->
