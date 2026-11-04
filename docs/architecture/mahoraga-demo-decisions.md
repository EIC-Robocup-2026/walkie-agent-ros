# Mahoraga Demo Decisions

The running record of decisions for the local prototype on `feat/mahoraga-demo`
and for the preparation of walkie-vision and walkie-database.

On 2026-11-05 the owner delegated demo decisions to the implementing agent and
asked for them to be recorded. These decisions are provisional and do not bind
main. Each entry states what reversing it costs, where that is not obvious.

Status values:

- **Owner**: the owner decided.
- **Default**: the owner accepted the plan's recommended option for the prototype.
- **Delegated**: the agent decided, under the delegation above or as a reported
  choice during a review fix.

## Scope and process

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| S1 | Build on the local branch `feat/mahoraga-demo`. Nothing goes to main. Ask before pushing. | The prototype tests the pipeline first. | Committing to main would bind the team to unproven code. | Owner |
| S2 | The demo covers the Phase 2 runtime, the Phase 3 verbs and atomics, and Mahoraga R0 to R2 in the fake world. | It exercises both the robot path and the research add-on. | The CI and ruff jobs, the voice environment split and leaving RAI stay undone. | Owner |
| S3 | Real-model runs are small slices on OpenRouter with `deepseek-v4.1-flash`. They stop before the 600-episode trial. Tests stay on fake models. | They prove that the pipeline runs at low cost. | The demo cannot say whether scripts beat per-step calls. | Owner |
| S4 | Claude implements and fixes. Codex `gpt-6-astra` reviews, verifies, researches and writes briefs. | Reviews stay independent of the builder. | None. | Owner |
| S5 | The agent decides demo questions alone and records them here. | This is a demo, and the choices can be reversed. | Asking first would stall the build. | Owner |
| S6 | `run_script` stays off the robot agent until a small real-model run shows that scripts work. | Monty is not in the environment the robot agent runs from, so joining them is more than a flag. | Joining them now shows a script driving the simulated robot, but adds environment work and delays the demo. | Delegated |

## Plan decisions

Numbers refer to "Decisions owed" in the Mahoraga Agent Plan.

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| 1 | Atomics are plain typed functions over `ctx`. | The schema comes from the signature and docstring. | A class or effect-field design adds own code. | Default |
| 2 | One adapter on `parse_docstring`, with `ToolErrorMiddleware` as the single error path. | Tools need no error settings of their own. | Validation errors still bypass this middleware inside ToolNode. | Default |
| 3 | Per-test tool sets are plain lists. | There are few sets, and they are easy to read. | Tool search stays out. Two held edits to the memory plan depend on this. | Default |
| 4 | Memory uses plain database ids. | No handle layer to build. | The `obj_` handle from the experiment branch is not ported. | Default |
| 5 | Own `WalkieTodos` instead of LangChain's todo middleware. | The library cannot seed a list ahead of time. | About 455 own lines, which is most of the runtime's overrun. | Owner |
| 6 | A bare "stop" gives a fixed spoken line and a latch, with no LLM. A compound "stop and come here" goes to the LLM. | A stop must not wait on a model. | The fixed line can overlap a `say` already speaking. See R5. | Owner |
| 7 | The three runtime fixes land inside the port. | There is no reason to fix code that is being replaced. | Main keeps those bugs until the port lands. | Owner |
| 8 | Keep `task_templates.toml` and its loader. Pick a template by name on `/walkie/task`, with no keyword matching. | Selection stays explicit. | Callers must know template names. | Owner |
| 22 | The trial model is `deepseek-v4.1-flash`. | It is cheap enough for small slices. | Results may not carry over to other models. | Default |
| 23 | Mahoraga is a top-level `mahoraga/` directory with its own pixi environment. | It stays removable, and nothing in `src/` imports it. | Running it from a worktree needs PYTHONPATH set inside the pixi run. | Default |
| 26 | The arena data adds a trash location and more objects. | Generated commands need them. | The data is a stand-in until real places exist. | Default |
| - | The test trim stays local. | It is not part of this work's history on any shared branch. | None. | Owner |

## Runtime

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| R1 | A call to `submit`, `stop` or `reset` after shutdown is refused. A shutdown clears the queue and ends the thread. | A late call must not hang or start work. | None. | Delegated |
| R2 | A stop line is logged under the session that heard it. | A reset can replace the session before the line finishes. | None. | Delegated |
| R3 | A failed call skips the later calls of the same AI message, which are answered "Not run". A duplicate `update_todos` counts as a failed call. | The robot must not announce something false. | If `deepseek` sends duplicates often, each costs an extra model call. | Delegated |
| R4 | Every tool checks the cancel token first. The fixed stop line and a direct call to the `say` atomic are exempt. | Nothing new may start once a stop is set, but the stop line must still speak. | None. | Delegated |
| R5 | A `say` already speaking is not cut off, so the stop line can overlap it. The real speech verb must allow this. | Interrupting playback belongs to the speech layer. | Two voices can overlap briefly on the robot. | Delegated |
| R6 | The runtime is async only. | The runner owns one event loop. | Concurrent Runners sharing one compiled graph, and a sync route, are not certified. | Delegated |

## Mahoraga measurement and checkers

These settle the 2026-11-04 review findings of the same number.

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| M1 | The code digest covers `mahoraga/*.py` except `report.py` and `__main__.py`, plus data files, `errors.py`, `adapter.py`, locations if used, `capabilities/*.py`, `atomics/*.py` and `pixi.lock`, with repo-relative paths. | A gate result must go stale when anything it depends on changes. | Any such change forces a gate rerun, which takes about a minute. | Delegated |
| M2 | A gate result counts only with passed true, no failures, two equal digests, and both logs present with matching recorded hashes. | It guards against accidents. | It is not signed, so it does not stop deliberate tampering. | Delegated |
| M3 | Score 1 needs a checker pass, no limit event, and a completed last script. Earlier recovered failures are allowed. The gate needs every oracle script to succeed. | The gate could pass with crashing oracle scripts. | None. | Delegated |
| M4 | The requested person must be returned by a lookup at the requested place. Escort needs origin, person found, then destination. Talk content stays loosely checked. | A checker must not pass a command that was not carried out. | Talk answers can still pass with weak content. | Delegated |
| M5 | Answers are judged by text rules on the first answering utterance after the last qualifying look, with exactly one distinct candidate that must be right. Equivalents count. Undecidable traces become listed limits. | The tool set and agent stay unchanged for scoring. | An `answer` tool would be stricter but changes what the agent does. | Delegated |
| M6, M13, M14 | Used means entered at runtime. Raised means an exception crossed the boundary, even if caught. Verified means ran and returned in a successful episode, keyed to exact source. | Each shelf metric needs one meaning. | None. | Delegated |
| M7 | Induction runs only on split `train` and never mixes arms. | Held-out results must stay unseen. | None. | Delegated |
| M8 | Errored episodes stay in the denominator as failures, their usage counts, and the report says incomplete. | Dropping them would flatter an arm. | A provider outage lowers scores. | Delegated |
| M9 | Held-out commands exclude the standard training command text. | The set must really be unseen. | The alternative was to keep the overlap and describe the split as unseen worlds only. | Delegated |
| M10 | `retry_on_error=0`, keeping the model-level `max_retries`. | Per-episode retries bias results. | A transient failure now costs that episode. | Delegated |
| M11 | The fake's call records hold the actual room and lookup results. | Checkers need the facts, not the request. | None. | Delegated |
| M12 | A relay tracks what the recipient knows, regardless of order. | Commands can be solved in more than one order. | None. | Delegated |
| M15 | The deadline is checked after each host call and before completion. A running call still finishes first. | A host call cannot be interrupted safely. | An episode can overrun by one call. | Delegated |

## Not built on purpose

| Item | Reason | What it would take |
| --- | --- | --- |
| R3 and R4 | R3 needs walkie-ros prerequisites and R4 needs verified shelf entries. | The walkie-ros work, then the R1 and R2 results. |
| Arm E, the secondary template-chain set, frozen paraphrases | Optional for the first results. | Small additions to the gym. |
| The pre-registration file and numeric pass marks | These belong to NineAI and NineCP. | Recording them before R1. |
| Real bundles for arm D | Stand-ins are enough to run the pipeline. | Written and reviewed bundles. |
| The 600-episode trial and the R2 trial | Outside S3. | Budget, and the pre-registration first. |

## walkie-vision and walkie-database preparation

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| P1 | "Ready" means both repositories can take the walkie-ros plan's work and the scene-graph core with no further interface, environment or contract work. The core itself (steps 4.7 to 4.9, 5.1 to 5.3, 5.8) is later work. | The request asks for readiness ahead of that work. The readiness brief can correct this reading. | Building the core now would be larger and would fix a design before it runs on real data. | Delegated |
| P2 | Work on local branches named `feat/scene-graph-prep`, in worktrees at `/var/tmp/wv-prep` and `/var/tmp/wd-prep`. Do not push until the owner agrees. | Direct commits to main are allowed, but other people read and pin those public branches, and these interfaces are unproven. Git also cannot check out main in a second worktree. | GitHub pins cannot reach the commits, so the demo builds against local checkouts until a push. | Delegated |
| P3 | Codex writes the brief, Claude builds from it, and Codex reviews the result. | This follows S4. | None. | Delegated |

## Still with the team

These stay open, and the prototype takes no position on them beyond working
around them.

- Decision 9: leave RAI. The prototype keeps RAI's connector for now.
- Decision 18: where real places live.
- Decision 19: the local fallback model.
- The walkie-database commit the robot runs.
