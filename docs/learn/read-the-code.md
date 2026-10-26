# Read the Code Next

Phase 1 lives mostly in six source files and four test files, and the tests
are the fastest way into them. This page gives a reading order, the checks to run on
a rai-core upgrade, what phase 1 left for later, and five exercises.

## The files

| File | What is in it | Lines |
| ---- | ------------- | ----- |
| [`src/walkie_agent/todo.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/src/walkie_agent/todo.py) | `TodoItem`, `TodoOp`, `TodoList` with its rules, ledger and listeners, and `console_line`. No ROS | ~680 |
| [`src/walkie_agent/tools/todos.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/src/walkie_agent/tools/todos.py) | `UpdateTodosTool` and `WaitForPersonTool` | ~100 |
| [`src/walkie_agent/graph.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/src/walkie_agent/graph.py) | `ToolSet`, `repair_history`, `WalkieToolRunner` and `build_walkie_graph` | ~420 |
| [`src/walkie_agent/agent.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/src/walkie_agent/agent.py) | `WalkieAgent`: queue, stop, reset, seeds, `/walkie/task`, `/walkie/todos` and console lines. Also `create_walkie_agent` | ~570 |
| [`src/walkie_agent/tasks.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/src/walkie_agent/tasks.py) | `TaskTemplate`, `load_task_templates`, `match_template` and `parse_task_message` | ~150 |
| [`src/walkie_agent/main.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/src/walkie_agent/main.py) | The ROS entry point and its flags, such as `--task-templates` and `--todos-topic` | ~100 |
| [`config/task_templates.toml`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/config/task_templates.toml) | The task templates. It ships with one commented example | ~30 |
| [`src/walkie_agent/tools/`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/tree/main/src/walkie_agent/tools) | The ROS tools in `navigation.py`, `speech.py` and `memory.py`. A failure raises `ToolException`, so its result has status `error` | ~660 |
| [`tests/test_todo.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/tests/test_todo.py) | One test per list rule, on a plain `TodoList` | ~880 |
| [`tests/test_graph.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/tests/test_graph.py) | Routes, gate, repair, queue, stop, reset and seeding, with no ROS | ~1650 |
| [`tests/test_tasks.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/tests/test_tasks.py) | Template loading, keyword matching and `/walkie/task` parsing | ~170 |
| [`tests/test_agent.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/tests/test_agent.py) | The agent over ROS topics, against the fake robot | ~620 |
| [`tests/scripted_llm.py`](https://github.com/EIC-Robocup-2026/walkie-agent-ros/blob/main/tests/scripted_llm.py) | `ScriptedChatModel`, the fake LLM that replays scripted replies and records what it was sent | ~70 |

## A reading order

1. **The `todo.py` module docstring,** then `TodoList.apply` and
   `_Draft._check_completion`. With the rest of `_Draft`, that is the whole rulebook.
2. **`tests/test_todo.py`.** Each test is one rule on a concrete list.
3. **`build_walkie_graph` in `graph.py`.** It is about 90 lines and maps onto
   [The Graph](the-graph.md). Then read `WalkieToolRunner._run_call`.
4. **`WalkieAgent._run_agent` and `_prepare_todos` in `agent.py`,** next to
   RAI's `LangChainAgent._run_agent`, as the next section shows.
5. **The ADRs, for the why.** [ADR-0038](../architecture/decisions/0038-own-graph-per-turn-binding.md)
   covers the graph and the queue; [ADR-0041](../architecture/decisions/0041-todo-list-and-seeding.md)
   covers the list, its rules and seeding.

## Checking a rai-core upgrade

`WalkieAgent` copies RAI methods and reads RAI internals, so every rai-core
upgrade needs a check by hand. The pin is `rai-core = ">=2.12.1, <2.13"` in
`pixi.toml`. First ask Python which version is installed, which RAI methods
Walkie defines again, and where RAI's source is:

```pycon
>>> import importlib.metadata, inspect, textwrap
>>> importlib.metadata.version("rai-core")
'2.12.1'
>>> from rai.agents.langchain import LangChainAgent
>>> from rai.agents.langchain.core import ToolRunner
>>> from walkie_agent.agent import WalkieAgent
>>> from walkie_agent.graph import WalkieToolRunner
>>> def replaced(ours, theirs):
...     return [name for name in vars(ours)
...             if name != "__init__" and callable(getattr(theirs, name, None))]
>>> replaced(WalkieAgent, LangChainAgent), replaced(WalkieToolRunner, ToolRunner)
(['__call__', 'stop', '_run_agent'], ['_func'])
>>> print(inspect.getsourcefile(LangChainAgent).split("site-packages/")[1])
rai/agents/langchain/agent.py
>>> source = textwrap.dedent(inspect.getsource(LangChainAgent._run_agent))
>>> print("\n".join(source.splitlines()[:7]))
def _run_agent(self):
    if len(self._received_messages) == 0:
        self._agent_ready_event.set()
        self.logger.debug("Waiting for messages...")
        time.sleep(0.1)
        return
    self._agent_ready_event.clear()

```

`__call__` hands each message to `_enqueue`, which holds the code copied from
RAI. `WalkieToolRunner._func` wraps RAI's `_func`; its real hook is the
`tools_by_name` attribute.

The block is also a tripwire. `tests/test_learn_docs.py` runs this page, so
an upgrade fails the version line, and a changed `_run_agent` fails the
source lines. To see every change, diff RAI's module from before and after:

```bash
pixi run python -c "import inspect, rai.agents.langchain.agent as m; print(inspect.getsource(m))" > /tmp/rai-agent-old.py
# change the pin in pixi.toml, run pixi install, then:
pixi run python -c "import inspect, rai.agents.langchain.agent as m; print(inspect.getsource(m))" > /tmp/rai-agent-new.py
diff /tmp/rai-agent-old.py /tmp/rai-agent-new.py
```

Then walk this table, run `pixi run test`, and only then update the block above.

| Walkie code | The RAI 2.12 code it copies or uses | What to check |
| ----------- | ----------------------------------- | ------------- |
| `WalkieAgent._run_agent` | `LangChainAgent._run_agent` and `_reduce_messages`, in `rai/agents/langchain/agent.py` | The `stream` call and its callbacks, the interrupt check between steps, the `"\n"` after the text |
| `WalkieAgent._enqueue`, behind `__call__` | `LangChainAgent.__call__` | The `max_size` overflow rule. RAI's interrupt branch is left out, because the behaviour is `"queue"` |
| `WalkieAgent.stop` | `LangChainAgent.stop` | It still sets `_stop_event` and `_interrupt_event`, waits for `_agent_ready_event` and joins the thread |
| `reset`, `_enqueue` and `_run_agent` | `_received_messages`, `_buffer_lock`, the stop, interrupt and ready events, and `_run_loop` | The same names and types. `_run_loop` still calls `_run_agent` whenever the ready event is set |
| `WalkieToolRunner` | `ToolRunner._func` and its `run_one`, in `rai/agents/langchain/core/tool_runner.py` | `run_one` still calls `tools_by_name[name].invoke(call, config)`, its `try` block still turns an exception into an `error` result, and `max_concurrency` is still 1 |
| `llm_node` and `_insert_system_prompt`, in `graph.py` | `llm_node` in `rai/agents/langchain/core/react_agent.py`, and `invoke_llm_with_tracing` | The system prompt goes first, once. The tracing helper still takes `(llm, messages, config)` |
| `_gates_since_task_start` | `ToolMultimodalMessage` in `rai/messages/multimodal.py` | The image message after a tool result still carries `tool_call_id` |
| `_send_to_targets` | `HRICallbackHandler._send_all_targets` in `rai/agents/langchain/callback.py` | Console lines still go out through `build_message` and `send_message` |

!!! example "Try it"
    In `pixi run python`, print `inspect.getsource(ToolRunner._func)` and
    find the `try` block inside `run_one`. Which two exceptions does it
    catch? Then read `WalkieToolRunner._run_call` in `graph.py` and say why
    it records `ok=False` and re-raises, instead of returning its own error.

## Not in phase 1 yet

These parts are proposed but not built yet.

- **Tool filtering.** `ToolSet.visible()` returns every tool. Filtering by
  access level and turn taint comes later ([Self-Evolving Agent](../architecture/self-evolving-agent.md#the-host-graph-and-interrupts)).
- **The taint rule on the list.** ADR-0041 lets a tainted turn change only a
  status. `todo.py` has no such check yet.
- **The tool cap.** `MAX_BOUND_TOOLS = 20` only logs a warning.
- **A typed task action.** `/walkie/task` takes JSON in a string. A typed
  `RunTask` action waits until something needs it.
- **The rest of the [roadmap](../architecture/self-evolving-agent.md#roadmap)'s
  phase 1 row:** `models.py` and per-role keys, the `--access-level` flag,
  and Ollama with an eval of the local model. None is in the code yet.
- **The HRI template.** It is not in `config/task_templates.toml` yet. Its
  atomics are built, and it comes with the sim run, which needs
  `oracle_perception` ([HRI Task](../architecture/hri.md#the-template)).

## Exercises

Run each test with `pixi run pytest -k <name>` from the repository root.

1. **The ledger:** `-k later_success_of_the_same_tool_resolves_a_failure`.
   The `speak` success does not resolve the failed drive; the later drive does.
   Predict what `apply` says without that drive, then check
   `-k success_of_another_tool_does_not_resolve_a_failure`.
2. **The gate:** `-k gate_sends_the_llm_back_at_most_twice_per_task`. Set
   `MAX_GATES = 1` in `graph.py` and rerun it: the first failing assert shows
   one gate message instead of two. Undo with `git restore src/walkie_agent/graph.py`.
3. **A question:** `-k wait_for_person_ends_the_run_and_the_answer_resumes_the_item`.
   Predict t1's marker in the third reminder, `[?]` or `[>]`, before you read
   the last asserts. Which line of `_prepare_todos` decides it?
4. **A stop:** `-k stop_interrupts_the_task_and_drops_older_messages` runs once
   per stop phrase. Why does "Walkie, stop" not stop a task? Find the answer
   in `is_stop_utterance`, then run `-k other_utterances_are_not_stops`.
5. **A keyword:** `-k a_keyword_matches_as_whole_words_in_any_case`. Read
   `_keyword_pattern` in `tasks.py` and predict whether "a lighthouse tour"
   and "house tours" match. Check with `-k no_match_without_the_whole_keyword`.

Previous: [Watching Progress](watching-progress.md)
