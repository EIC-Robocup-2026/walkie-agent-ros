# ADR-0060: Write tools as plain typed atomics behind one adapter

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-05

**Context**

The old tools were LangChain and RAI tool classes, each with its own error text and
connector handling. Mahoraga needs to call the same behaviour from a script, with no
LangChain tool around it. Proposed
[ADR-0037](/architecture/decisions/0037-tool-tiers-bundles-effects.md) planned a `META`,
`Args`, `Out` and `run(ctx, args)` shape for this. It was never built.

**Decision**

An atomic is a plain typed function in `walkie_agent.atomics`. It takes `ctx` first,
returns a JSON-able value and raises `StepError`. Its Google-style docstring is what the
LLM reads.

`walkie_agent.adapter` is the one bridge to the callers.

- `as_tool(fn, ctx)` builds a LangChain `StructuredTool`. `ctx` is hidden from the
  schema, a parameter with no description raises `ValueError`, and a non-string result
  goes back as JSON text.
- `bind(fn, ctx)` builds a plain callable for Mahoraga's scripts, with the same checks.
- Both check `ctx.cancel` before an atomic runs.
- The schema validates strictly. A JSON `true` or a numeric string is refused for a
  number, and an integer still counts as a number.
- `tool_error_middleware()` is the one error path. A `StepError` becomes
  `"CODE: message"`. Anything else logs its traceback and becomes
  `"INTERNAL: <tool> failed with <ExceptionType>."`, so one bad tool does not end a run.

The sets are fixed lists in code: `GPSR`, `HRI` and `DEFAULT`
([ADR-0067](/architecture/decisions/0067-static-tool-sets.md)). There is no registry and
no tool search, and nothing here promises either.

This replaces the tool-shape part of proposed ADR-0037, and the matching implementation
choices in [ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md) and
[ADR-0041](/architecture/decisions/0041-todo-list-and-seeding.md). ADR-0037's tiers,
bundles and effect gating stay an unbuilt proposal.

**Consequences**

- The agent, the unit tests and Mahoraga run the same function bodies.
- Evidence that ran: unit tests and Mahoraga's fake-world suite. No real model has read
  these schemas or docstrings yet, so whether they suit a real model is unknown.
- `find_objects`, `object_info`, `find_people` and the arm atomics exist and pass against
  `FakeRobot`. On the robot today they answer `UNAVAILABLE`
  ([ADR-0052](/architecture/decisions/0052-capability-verbs-and-backends.md)).
- LangChain drops a `Returns:` section from a tool's description, so an atomic has to
  describe its return value in the summary.
- Adding an atomic means editing code and a list, on purpose. A running agent cannot
  gain a tool.
- **As built in the demo** (decision T2 of the
  [demo decisions](/architecture/mahoraga-demo-decisions.md)): the atomics and their
  sets moved to `walkie_agent.tools.atomic`, the atomic tier's folder in ADR-0037's
  layout, and nothing is left at `walkie_agent.atomics`.

**See also:** [Mahoraga demo: atomics](/architecture/mahoraga-demo.md#atomics),
[Tool access](/setup/05-tool-access.md)
<!-- --8<-- [end:body] -->
