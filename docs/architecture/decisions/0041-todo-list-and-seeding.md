# ADR-0041: Keep a seedable todo list for every task

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

The agent keeps no plan today. In a task with several steps, such as "go to
the kitchen, pick up the red mug, bring it back", an LLM tends to drop a
step, to call the task done once it has spoken, or to report success after
a tool failed. For fixed tests such as Restaurant or Receptionist, the
steps are known before the first turn, and the operator should set them.

Two existing designs were compared with what a robot needs:

| | A coding agent's TodoWrite tool | LangChain's `TodoListMiddleware` |
| - | ----------------------- | -------------------------------- |
| Update | Rewrites the whole list | Rewrites the whole list through a LangGraph `Command` |
| One item in progress | A prompt rule | Not enforced |
| Seed before the first turn | No | No: the `todos` key is marked `OmitFromInput` |
| Shown again each turn | A harness reminder | No, only the old tool message |
| Runs on RAI's `ToolRunner` | Not applicable | No: `ToolRunner` reads `content` from each result and fails on a `Command` |

**Decision**

We keep TodoWrite's active form, its single item in progress and its
per-turn reminder. We change three things: the LLM edits items by id
instead of rewriting the list, the rules are enforced in code, and a list
can be seeded before the first turn. Newer task tools from the same
product also track dependencies and owners between items. Those are not
copied, because one robot runs one serial plan.

| `TodoItem` field | Meaning |
| ---------------- | ------- |
| `id` | Short id such as `t1`, stable for the whole task |
| `content`, `active_form` | The step, and its "-ing" line for `/walkie/console` |
| `status` | `pending`, `in_progress`, `waiting`, `completed`, `failed` or `cancelled` |
| `note` | Required for `failed` and `cancelled` |
| `source`, `locked` | `seed`, `agent` or `operator`; seeded items are locked |
| `requires_tool` | A tool that must succeed before the item can complete |
| `evidence` | Tool call ids and bundle run ids |

- **`update_todos(ops)`** takes a list of `TodoOp`: one flat shape with
  `op` (`add`, `update` or `remove`), `id`, `content`, `active_form`,
  `status`, `note` and `after`. A union of three op shapes would become
  `anyOf` in the tool schema, which Gemini function declarations and small
  Ollama models handle poorly, so per-op requirements are checked in code.
  An op cannot set `waiting`; only `wait_for_person` does. The tool returns
  one sentence, such as "Updated: t2 → completed. Open: t3, t4.", and
  raises `ToolException` on a rejected op. It never returns a `Command`.
  The graph accepts one `update_todos` call per AI message.
- **`wait_for_person(question)`** speaks the question, sets the item in
  progress to `waiting` and ends the turn. The answer arrives as a new
  message while the agent is idle, so it never interrupts a run. This is
  how a `STOPPED` bundle result or a Restaurant order gets clarified.

| Rule enforced in code | Why |
| --------------------- | --- |
| At most one item `in_progress` or `waiting` | Physical work is serial |
| `failed` and `cancelled` need a `note` | The published list shows why a step was dropped |
| Locked items can change status, but not wording, and cannot be removed | The LLM cannot drop a seeded step |
| An error ledger records every `ToolMessage` status that the tools node sees. `completed` is refused if a tool or bundle failed since the item went in progress and no later call of the same tool succeeded, unless a `note` explains it | Stops "done" after a failure |
| `requires_tool` must have succeeded | "Hand over" needs a successful `hand_over`, not "here you go" |
| In a turn tainted by an untrusted web result, only `status` can change | Web text cannot plant a task that outlives the turn |

- **Reminder.** Each LLM call gets one trailing `HumanMessage` that is not
  saved in the history. It is a human message because some providers
  reject a system message that is not first. It is added only when the
  list is not empty.

  ```text
  <todos>
  [x] t1 Go to the kitchen
  [?] t2 Pick up the red mug  waiting: "The left one or the right one?"
  [ ] t3 Come back to the operator
  [ ] t4 Hand over the mug  needs: hand_over
  </todos>
  ```

- **Completion gate.** If the LLM stops calling tools while items are
  open, the graph sends it back with "[todo] Open items: t3, t4. Finish
  them, or mark them failed or cancelled with a note." It fires at most
  twice per task, and never while an item is `waiting`. Otherwise it would
  push the agent to guess every time it asked a question.
- **Progress lines.** The fixed lines that `request_tool` and `fetch_sound`
  speak during tool creation do not change the list.
- **Topics.** `/walkie/todos` carries a JSON snapshot with transient-local
  QoS and depth 1, so a viewer that starts late still gets the current
  list. `/walkie/console` gets one `active_form` line per status change.
  `/walkie/reset` clears the list and publishes `[]`. `/walkie/task` and
  `/walkie/todos` join `FORBIDDEN_NAMES`, for the same reason as
  `/walkie/reset`: the agent must not drive its own control plane.
- **Seeding.** `WalkieAgent.submit(text, todos=None)` stores the seed in a
  pending slot with the message. The agent overrides `_run_agent` in full,
  about 20 lines copied from RAI, and applies the seed before streaming, so
  no graph channel is involved. Seeds come from the Python API, from
  templates in a new `config/task_templates.toml` chosen by keyword or by
  the operator, or from a new `/walkie/task` topic, a `std_msgs/String`
  holding JSON `{task, todos}`. `HRIMessage` has no metadata field, so it
  cannot carry a seed. Free-form commands (GPSR) have no seed: the first
  turn writes one.

How to seed over ROS is an open choice. The recommendation is the topic
now and a typed `RunTask` action later, if it is needed: cheap now, typed
later. The alternative is `RunTask` from the start, which is typed but
needs a new message package to pass the build allow-list and the
`ros2_ws.repos` pins first.

**Consequences**

- A finished task shows evidence for each step. A skipped or failed step
  is visible in the list, on the console and in the trace.
- The reminder adds tokens to every turn, in proportion to the list. The
  static system prompt stays first, so prompt caching keeps working. The
  reset test in `tests/test_agent.py`, `test_reset_topic_clears_memory`,
  expects exactly `[SystemMessage, HumanMessage]` and still passes,
  because an empty list adds nothing. The gate can cost two extra LLM
  calls per task.
- The copied `_run_agent` must be rechecked on every rai-core upgrade
  ([ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md)).
  That ADR also covers a new utterance that arrives mid-task.
- This is phase 1 work, tested in new `tests/test_todo.py` and
  `tests/test_graph.py`, including no gate after `wait_for_person`.

**See also:** [Self-Evolving Agent](/architecture/self-evolving-agent.md#the-todo-list),
[ADR-0008](/architecture/decisions/0008-reset-topic.md), [ADR-0009](/architecture/decisions/0009-forbidden-names-guard-rail.md)
<!-- --8<-- [end:body] -->
