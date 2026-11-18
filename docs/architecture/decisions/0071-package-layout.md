# ADR-0071: Lay out the agent package by concept, with its import rules checked

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-18

**Context**

The package had grown flat, and its folder names did not say where things were.

- `main.py`, 1,049 lines, was the composition root, the system prompt and the tool
  policy at once: it held the prompt rules, `build_system_prompt` and `TOOL_SETS`.
- Switching a model touched several modules. The agent's LLM settings and builders sat
  in `runtime/model.py`, beside the runtime. The gym built its own chat model in
  `gym.py`, `voice.py` read the speech models, and the face and re-identification
  models' match limits sat inside `perception/identity.py`.
- `perception/` was mostly the people memory: the track memory, the people store,
  identity matching, recognition and the per-track vectors, beside a few word tables.
- `ros/__init__.py` imported every backend eagerly. Importing one backend loaded them
  all, and with `ros/io.py` the runtime, so `sim/` loaded LangChain.
- `mahoraga` was a codename for an offline evaluation gym, at the package's top level.
- The import rules were checked by hand-written tests that import modules in a fresh
  interpreter, such as `tests/test_mahoraga_absent.py` and `tests/test_ros_free.py`.

**Decision**

The package is laid out by concept, as decision
[RL-1](/architecture/mahoraga-demo-decisions.md#repository-layout) chose. Each step was
a pure move with its references. Both test suites passed after each step, and the system
prompts and every tool's name, description and JSON schema stayed byte-identical.

| Path in `src/walkie_agent/` | What it holds |
| --- | --- |
| `main.py` | The agent process: flags, `assemble`, signals and exit codes |
| `config.py` | Reads `config.toml`, and the agent's run limit and deadline from it |
| `chat_models.py` | The LLM vendors: `build_model`, `build_fallback`, `build_callbacks` |
| `prompts.py` | The system prompts: a rule per bound tool, the places and `robot.md` |
| `errors.py` | `ErrorCode` and `StepError` |
| `locations.py` | The places file and its name rules |
| `voice.py` | The voice entry points, on RAI |
| `half_duplex.py` | The listening gate that `voice.py` uses |
| `capabilities/` | The verbs on `ctx`, their guards, records and stand-ins, and `FakeRobot` |
| `tools/` | The LLM tools: `atomic/` and its catalogue, `sets.py`, `adapter.py`, and the shelf tiers |
| `runtime/` | Our layer on `create_agent`: middlewares, todo list, Runner and `task_templates.py` |
| `scripting/` | `run_script`'s sandbox, driver, evidence and shelf |
| `perception/` | Rules that read walkie-vision's detections, its topic names and the word tables |
| `people_memory/` | Who is who: the track memory, the store, identity, recognition and `model_limits.py` |
| `arm/` | Arm poses, arm states and their answers. No ROS. |
| `ros/` | One ROS 2 backend per capability, and `operator_topics.py` for the operator's topics |
| `sim/` | Stand-in ROS nodes and the oracle scene |
| `evals/` | Offline evaluations; `evals/mahoraga/` is the gym, with its prompt in `experiment_arms.py` |

import-linter checks four contracts in `pyproject.toml`, from
`tests/test_import_contracts.py` inside pytest (decision
[RL-9](/architecture/mahoraga-demo-decisions.md#repository-layout)). A forbidden import
fails directly or through other modules, unless the contract says otherwise.

- **C1, `gym-stays-out`.** No top-level module or package of `walkie_agent` imports
  `walkie_agent.evals`.
- **C2, `core-is-ros-free`.** `capabilities`, `tools`, `runtime`, `scripting`,
  `perception`, `people_memory`, `arm`, `errors`, `locations`, `config`, `chat_models`
  and `prompts` never import `main`, `ros`, `voice`, `sim` or `evals`.
- **C3, `package-inits-stay-empty`.** The inits of `walkie_agent`, `ros`, `tools`,
  `people_memory` and `evals`, each on its own, import nothing from `walkie_agent`,
  `langchain`, `langchain_core`, `langchain_ollama`, `langchain_openai`, `langgraph` or
  `langfuse`.
- **C4, `backends-dont-see-the-llm`.** Every module of `ros` but `operator_topics`, and
  `sim`, never import `runtime`, `tools`, `prompts`, `chat_models`, `langchain`,
  `langchain_core`, `langchain_ollama`, `langchain_openai`, `langgraph` or `langfuse`. A
  test fails when a module of `ros` is missing from the contract's list.

import-linter treats an external package as a leaf, so it never sees that
`langchain_openai` imports `langchain_core`. C3 and C4 therefore name each LangChain
package, and a test fails when `walkie_agent` imports a LangChain, LangGraph or Langfuse
package that either contract does not forbid.

The old paths stay gone. `REMOVED` in `tests/test_ros_free.py` fails when code, a build
or config file, or a live docs page names `walkie_agent.adapter`, `walkie_agent.tasks`,
`walkie_agent.runtime.model`, `walkie_agent.ros.io`, `walkie_agent.mahoraga`, the gym's
`arms` module, `UnavailableVision` or the lookup guard in `ros.vision`, the prompt
rules or `build_system_prompt` in `main`, `SYSTEM_PROMPT` in `runtime`, or
`perception`'s `memory`, `people_memory`, `identity`, `recognition`, `track_vectors`,
`people` and `objects`. Other guards there fail when a moved name is bound again at its
old home, as a re-export would. No old path has a shim.

Package inits hold only a docstring, except `capabilities/` and `runtime/`, which are
the facades their callers import from (decision
[RL-8](/architecture/mahoraga-demo-decisions.md#repository-layout)). For the five inits
C3 names, C3 keeps out imports, and a test in `tests/test_import_contracts.py` fails when
one holds anything but its docstring.

**Consequences**

- A newcomer can tell from the tree where the LLM is chosen, where the prompts, tools,
  tool sets and ROS adapters live, where the people memory lives, and that the gym is an
  offline evaluation. [Read the Code Next](/learn/read-the-code.md) answers these
  questions one by one.
- An import that breaks a rule fails `pixi run test` with the chain of imports.
- The paused HRI line, `experiment/hri-agent`, is harder to merge. Merge it first into
  85a705d, the commit before the move, with `git merge -X find-renames=20%`, and resolve
  its conflicts there. From the HRI line's merge base 91ffe25 to 85a705d, `todo.py` became
  `runtime/todos.py` (27% similar) and `whoami/documentation/walkie.md` became `robot.md`
  (28%), both under git's default of 50%, and the HRI line edits both. The lower limit
  makes git follow them as renames. Then merge the result across the move with a plain
  `git merge`: every rename in the move is at least 84% similar. Do not merge the HRI line
  across the move directly: from 91ffe25, `tasks.py` to `runtime/task_templates.py` is
  only 24% similar, because `tasks.py` was rewritten before the move.
- The Mahoraga gate digests each file's repository path, so the move changed the gym's
  code digest, from 4b50cc1cf6164aec to 29464480cfce22a5, and voided the earlier gate
  results. In each Mahoraga home, rerun the gate once before any LLM arm. The rerun
  after the move passed every split, trial, train and held-out (300, 300 and 480
  episodes), and each split's two passes gave the same canonical log. Those digests
  differ from the run before the move, because a canonical log keeps the git commit it
  ran at.
- The moved agent ran in the Chula sim on 2026-11-19, at 6abe298.
  `scripts/chula-sim.sh check --agent` passed. A `/walkie/task` message,
  `{"task": "Go to the kitchen room, then say that you arrived.", "todos": []}`, made
  the agent drive to the kitchen room in 38 s and say it had arrived, with no warnings in
  its log.
- An environment built before import-linter was added fails to collect
  `tests/test_import_contracts.py`. Run `pixi install --frozen` in that checkout, the
  `/var/tmp/wa-demo` worktree included.
- Known weaknesses are kept. `tools/` is both the tool catalogue and the shelf's store
  (`bundled/`, `temporary/` and the ledger); moving the store would change behaviour.
  `run_script`, `update_todos` and `wait_for_person` live outside `tools/`, in
  `scripting/agent_tool.py` and `runtime/todos.py`. Names are reused: `arm/`,
  `ros/arms.py` and `tools/atomic/arms.py`, and `people_memory/` and
  `ros/people_memory.py`. `main.py` is still about 835 lines. `evals/` has one child
  until a second benchmark lands.
- The inits of `perception/`, `tools/atomic/`, `tools/bundled_preset/` and
  `evals/mahoraga/` still hold code (topic names, the catalogue, the gym's stand-in
  bundles and the gym's `home`), outside C3. Moving it out would change import paths
  that this layout did not plan.

**Alternatives**

- **The smallest change**, about 49 files: empty `ros/__init__.py` and split the model
  module, and leave the top level as it was. It keeps the HRI line easy to merge, but a
  reader still cannot tell where the prompt or the tool sets live.
- **Core and adapters**, about 175 files: split the package into a transport-free core
  and its adapters. It pays off only with a second transport or robot backend, and it
  changes every entry point.

**See also:** [Mahoraga Demo Decisions: repository layout](/architecture/mahoraga-demo-decisions.md#repository-layout)
(RL-1, RL-8, RL-9), [Read the Code Next](/learn/read-the-code.md),
[ADR-0052](/architecture/decisions/0052-capability-verbs-and-backends.md),
[ADR-0060](/architecture/decisions/0060-atomic-tools-and-adapter.md)
<!-- --8<-- [end:body] -->
