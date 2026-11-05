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
| S6 | `run_script` stays off the robot agent for this demo. The small real-model runs below show that scripts work in the fake world, which was the condition to reconsider, so this is now a choice and not a wait. | Scripts cost fewer rounds and fewer tokens but gained no success, the fake world's moves take no time, and the script budget of 30 s of wall clock would not survive real moves. Monty is also not in the environment the robot agent runs from. | Joining them now shows a script driving the simulated robot, but needs a budget for real motion and environment work. The robot agent keeps per-step tools, at about 7 model calls a command against 4 for scripts, while arm B spent 30% to 39% fewer tokens per success than arm A in the fake world. Revisit once the robot path runs. | Delegated |

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

## ROS wiring

These settle the forks in the ROS wiring brief.

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| W1 | `vision.objects` and `vision.people` raise `UNAVAILABLE` in the first ROS slice, and the old remembered-object tool is retired. A real vision backend follows once the walkie-vision contract lands. | A remembered row is not what the camera sees now, and an empty list would claim a view that never happened. | Keeping remembered search as its own capability adds code now. Until the backend lands, the HRI demo cannot find people. | Delegated |
| W2 | HRI input is text only. A message carrying images or audio is logged and refused, except a bare stop, which still stops (W18). | `Runner.hear` takes text, and dropping a payload silently would hide the loss. | Extending the runner's input contract keeps multimodal input, at the cost of new tests before the cutover. | Delegated |
| W3 | If shutdown cannot settle, keep the executor alive and ingress closed, retry within bounds and report what is unresolved. A second signal forces the exit. | Destroying ROS under a live worker removes the only route to cancel a goal. | A timed forced exit is simpler but can leave the robot driving with no cancel route. This is not an emergency stop. | Delegated |
| W4 | Mark obsolete code assumptions in the larger proposals, and record exact partial supersessions. Do not withdraw whole proposals. | Whole withdrawal is wider than this slice. | The docs stay longer than the plan's target for now. | Delegated |
| W5 | Place names are stored as trimmed lowercase keys. Other forms and duplicates are refused at load, and lookup ignores case. Production places stay empty, and the invented places move to test fixtures. | Places must come from a reviewed map, which is decision 18. | The agent has nowhere to go by name on a real map until someone fills that file. | Delegated |
| W6 | `say` returns once the message is published. | RAI's TTS gives no playback acknowledgement. | A successful `say` does not mean the words were heard. | Delegated |
| W7 | Keep RAI's connector, `rai-core`, `rai-s2s` and both voice entry points. Remove RAI's agent host from `main` only. | `voice.py` still uses `AgentRunner`. | RAI's OpenCV pin stays, so Zenoh is still built from source. | Delegated |
| W8 | A static `robot.md` replaces the generated whoami tree. | The agent needs a description, not a vector index. | There is no document search tool. | Delegated |
| W9 | New ADRs take 0052, 0060, 0061, 0062, 0067, 0068 and 0069. Eleven learn pages stay, and `old-vs-new` goes. | The plan reserves 0053 to 0059, and shrinking the learn pages further is a later editorial cut. | The learn section stays longer than the plan's six pages. | Delegated |
| W10 | Build this slice on `feat/mahoraga-demo-ros` from `3e345f2`, in parallel with the Mahoraga fixes. | The two touch almost no shared files. | Both need merging, and the gate reruns after, because `pixi.lock` is in its digest. | Delegated |
| W11 | A forbidden endpoint (`/cmd_vel` and the other three reserved names) is refused when `RosBase` is built, not per call. | A name is fixed for the object's life, so the failure belongs at startup. | An invalid ROS name raises rclpy's own exception there rather than `REFUSED`. | Delegated, built |
| W12 | A pose sample stamped over 0.5 s ahead of the node's clock counts as stale. | A sample from before a sim rollback or an unstarted clock must not pass as fresh. | Clock skew over 0.5 s between machines reads as a stale `TIMEOUT`, never as a wrong pose. Check this on the robot. `_MAX_FUTURE_STAMP_NS` in `ros/base.py` is the place to change it. | Delegated, built, unmeasured on the robot |
| W13 | A goal whose end is unknown (a failed acceptance or result future, `STATUS_UNKNOWN`) keeps the motion gate closed. | A goal that may still be driving must not unlock a second one. | If Nav2 restarts and forgets a goal, motion stays refused until the agent restarts. | Delegated, built |
| W14 | `close_ingress()` only sets a flag, and the subscriptions are destroyed after the goal drain. | Destroying an endpoint under a spinning executor can end the spin thread (`rclpy/executors.py`, `event_handler.py:124`). That thread is the only way to hear Nav2 answer a cancel. | A late message reaches a callback that returns at once. The spin thread can still die in the final destroys or inside RAI's `connector.shutdown()`, once every goal has settled. That prints an occasional traceback on stderr and does not change the exit code. Fixing it needs RAI's private attributes, so it stays under W7. | Delegated, built |
| W15 | A second signal exits through `os._exit(3)` after logging what is unresolved. A failure during startup or running counts as the first signal. Signal handlers raise a counter, not an event. | `os._exit` is the only way past the connector's non-daemon spin thread, and `Event.set` takes a lock a handler could deadlock on. | Forced exit skips all cleanup, so it can leave a goal running. Checked on a real process with a patched close, and by fakes with a goal. The combination, a real process and a real unanswered Nav2 goal, has not run. | Delegated, built |
| W16 | ADR-0035 stays Accepted, amended by ADR-0052: its `find_objects` tool is retired, and its rule stands. | The memory capability still has to read through the database node's services, and the pin and build steps stay for it. | ADR-0035's consequences still describe the old tool, so read them with the note under its status line. ADR-0009, 0010 and 0031 were left alone in the same pass and have not been checked against the new code. | Delegated, documented |
| W17 | The Chula silent recipe drops `scouting/gossip/enabled=false` from both `router_cfg` and `node_cfg`. | Measured in the namespace: with gossip off on the router, on the nodes or on both, two processes on the router do not discover each other. Multicast and shared memory off with gossip on delivered 235 clock messages. | Isolation now rests on the namespace, which has only `lo` and a private router. | Delegated, recipe changed, Chula run still to come |
| W18 | A bare stop carrying an image or audio still stops. This narrows W2. | The build took W2 literally, and a refusal there would withhold a stop. A stop has no use for its payload, so nothing is lost by ignoring it. | The payload is dropped with no log line on that one path. | Delegated, built |
| W19 | A spoken stop and a reset also call `ctx.base.stop()`, before the Runner takes its lock. | Merge review R1. Once Nav2 refused the first cancel, the verb had returned, so nothing read the token again, and a second stop never asked Nav2 again: the probe left the goal live after two stops. Called after the lock, base.stop() could set the token of a task that had just started. | The fake robot's call log now shows a `base.stop` for each stop and reset. A stop on a Runner that has shut down still reaches the base before it raises. | Delegated, built |

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

These came up while building those fixes.

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| M16 | The shelf loader shadows `eval`, `exec` and `locals`. | A probe showed that a script could read the counter token from a caught error and forge evidence through them. | They raise `PermissionError` only in a script with something to watch, so their behaviour depends on the script. This is no security boundary. | Delegated |
| M17 | Evidence counters run inside the Monty worker. After the wall budget is spent, a script gets up to one more feed period to unwind, with every call refused. | A host call per function would spend the script's call budget and could not run inside a builtin's key function. Without the unwind the counters could not be read. | A wrapper adds a frame, so recursion through a watched function hits the depth limit in half the calls. Evidence is lost, failing closed, on a worker crash. | Delegated |
| M18 | A verified function that names a saved function which never ran stays off the frozen shelf. | Stricter verification would otherwise leave it unable to stand alone. | A useful function can miss the shelf until its helper has run. | Delegated |
| M19 | Scheduled epochs missing from the log count as failures. A log whose status is not success is incomplete, and so is a comparison of arms given different commands. | This goes past M8, because each case would flatter an arm. | A run that lost one episode reads as incomplete. | Delegated |
| M20 | `run --arms induct` needs an explicit `--split train`. | The default split is trial, and M7 refuses induction there. | The plan's flow is unaffected, but a bare `--arms induct` now exits with code 2. | Delegated |
| M21 | Keep both retry layers: the OpenAI client's 3 requests per attempt, inside Inspect's 4 attempts. | Nothing asked to change the client layer. | A dead provider sees up to 12 requests per model call, a computed figure. Passing `max_retries=0` as a model argument would remove the client layer. | Delegated |
| M22 | The first line that mentions a candidate decides, and later lines go unread. | Reading corrections and negation needs the language parsing M5 ruled out. | Known wrong passes: "I see 2." then "No, 5.", and "It is not the banana." Known wrong fails: "There are two, one apple and one banana", and "No one is waving". Read failed answers from real-model runs before trusting a score. | Delegated |
| M23 | Reaching an escort's destination is judged from the records after the find, and a destination that is or holds the named origin needs no further move. | The generator can draw "from the shelf to the shelf", and a robot that moves on afterwards still got there. | The fake's people do not move, so an escort whose person never came along still passes. | Delegated |
| M24 | `induct`'s prompt says the model is building a library for later commands, to write each command's steps as a function with the names as parameters, call it and name it in `save`. | Offered as an option, 88 scripts held no function and nothing was saved, so C and C2 were B under other names. The plan left this prompt open. | 22 functions saved from 30 commands, all general. The shelf is uneven: it holds near duplicates, functions that do nothing when a search comes back empty, and functions whose only verification is an episode the model later rescued by hand. Verified means the function returned in a passing episode, not that it did the right thing. | Delegated |
| M25 | `Shelf.save` refuses a function with a parameter default that is not a literal, judged by `ast.literal_eval` with any call also refused, and a function named `locals`, `eval` or `exec`. Those three names are never watched. | Recheck findings R1, R2 and R4. The shelf feeds before the loader, so a default kept the real `eval`, which forged and erased counters, or an unwrapped helper, whose runs and raises went uncounted. `ast.literal_eval` alone accepts `set()`, which ran a saved `set` in that feed and reopened the first hole. The loader defines the three names itself, so a wrapper gave its replacement the source's sha. | A function-valued default can no longer be saved, so a model hits a refusal and must rewrite around it, naming the helper in the body. Feeding the shadows first would keep `eval` out but not a helper, whose wrapper needs the shelf fed first. A shelf written before this rule is not rechecked. | Delegated |
| M26 | For each function name, an episode verifies the last version that returned and that nothing raised through, walking the scripts in order. | Recheck finding R3. Collapsing versions by sorted sha let hash order decide, which lost the current version in about half of two-version cases. | An earlier version that also ran cleanly earns no `verified` event, so saving that same text again later starts with nothing. | Delegated |
| M27 | `_numbers` reads whole tokens with a regex and a word table. "1.0" is 1, "1.1" and "-1" are their own values, a hyphenated compound such as "twenty-one" is a candidate no count equals, number words run to nineteen, the tens and "hundred", and "3rd" is no number. "single" means 1, and "couple" and "pair" mean 2. Pose and gesture gain ordinary words such as "sat", "feet", "prone", "up", "aloft" and "indicating", and a raised arm now needs an arm or a hand in the line. | Recheck findings R5, R6 and R7. The tokenizer read a part of a number as the answer, so "twenty-one" and "-1" passed for 1 while "1.0" failed. Common correct phrasings scored 0, and "raising their right leg" passed as the arm gesture. | The limit that remains: "There is a dish" still fails, because reading "a" as one would make "on a bar" a second number. This is no number parser, so a word past the table goes unread: "two thousand" reads as 2 and "a dozen" as no answer. "couple" is loose for some speakers, and "feet" makes "sitting with their feet up" name two poses and fail. | Delegated |
| M28 | A comparison's `complete` also needs equal scheduled (command, epoch) sets and, for each id logged in both arms, the same `sample.input`. When `sample_ids` is missing, a log schedules `dataset.samples` times its epochs. | Recheck findings R8 and R9. Comparing command ids alone called one epoch against three complete, and accepted changed text under one id. A log without ids saw only the episodes it kept, so a lost one left the denominator and the arm read as complete. | A partial rerun with a different `--epochs` in one log directory now reads incomplete, and `compare` still averages by command. A command a log without ids never logged gets a stand-in id unique to that log, so it counts as a failure in M1 and pairs with nothing. Text is compared only over logged samples, so an id one arm never logged goes unchecked. `warning` stays null when only a comparison is incomplete. | Delegated |
| M29 | Each model attempt in the gym has 30 s, then it is abandoned and retried. `run_limit` stays 25 and `deadline_s` stays 60. | 5 of 1750 model calls ran away. One wrote 131072 tokens in 275 s, most of the first shelf build's output, and the other 4 had not finished when their episode reached its 300 s limit, which fails an episode through no fault of its answer. Inspect's `timeout` only stops further retries. The longest normal call took 13 s and the longest episode 15 calls. | Abandoned attempts may still be billed by the provider, and Inspect's usage does not count them. Both limits are measured only on the fake world, on a fast model, so a slower model needs a higher attempt limit. | Delegated |
| M30 | A `talk` command that asks for a fact gets a source and a check: a clock and team facts on `ctx`, the same in the fake world and on the robot, and a checker that compares the line for those topics. "Something about yourself" stays unchecked. To build after the ROS merge, which is changing the same capability files. | Third run: 4 of 30 held-out and 8 of 30 training commands ask for a fact that nothing supplies. Every arm invented one ("It is 3 o'clock"), and in A, B and D the line "What day is tomorrow?", which reads the command back as a question, passed. The robot has no clock either, so it would state a made-up time. The hunt for a clock also caused the longest failure of the run. | Keeping the documented limit leaves about 13% of held-out commands free in every arm and the robot inventing answers. The cost is one more capability, a new function in every arm's prompt, and a gate rerun. Runs before this are not comparable on these commands. | Delegated, not built |
| M31 | A later look with no answer after it no longer discards an earlier answer. The checker judges the last look that a line naming a candidate follows. | Third run, training command 100030: the model said the right answer, "chips bag", then looked again without speaking and scored 0. The limit was documented, and a real model hit it. | A right answer, a look and a wrong answer still fails, and a wrong answer, a look and a right answer still passes. Only a silent look after an answer changes, and a wrong answer followed by one still fails. | Delegated |
| M32 | M3's input counts everything the model was sent, with the cached part also given alone. Its cost is what the provider recorded in each raw response, with calls that have no record counted beside it, and `run` sets `log_model_api=True`. The plan's prices are gone from the report. | Third review, N1 and N2. Inspect's `input_tokens` leaves out cache reads, about 90% of input here, so a figure of 2177 tokens per success was really 17343, and the order of the arms changed. The plan's prices made the second training run read $0.023 when the provider recorded at least $0.074. Inspect keeps only the first few raw responses unless told otherwise. | Logs from before this have a lower bound for cost, and larger logs follow from now on: 31 KB for 3 commands. Abandoned attempts have no record, so they remain a lower bound. A provider that returns no cost reports null, with no estimate. | Delegated |
| M33 | A raise counts against the version that ran, whether or not the script saved it. | Third review, N3. A failed script that defined the same text as a saved version lost its raise, because only saved or shelf versions were kept, and a later recovery then verified that version. | None seen. A version that never reaches the shelf only adds an entry nothing reads. | Delegated |
| M34 | "Forearm" counts as an arm, and a line naming a leg, knee or foot says no raised arm. | Third review, N4 and N5. The arm requirement of M27 rejected "Their right forearm is raised" and still passed "Their hand rests on their raised right leg", since words are not attached to each other. | A true arm answer that also mentions a leg now fails. Limits left as they are, all low in likelihood and short of parsing language: "I can see the person's feet" reads as standing, "waist-high" holds "high", "rightward" and "palm" are not read, and "TV stand" reads as standing. | Delegated |

### Small real-model runs

All three ran `deepseek/deepseek-v4.1-flash:nitro` on 30 held-out commands, one epoch
each, in the fake world, on 2026-11-05. These are checks that the pipeline works, not
the planned comparison. No difference in success between arms is significant (McNemar
p is 0.5 or higher in each pair), and success is near the ceiling, so it cannot separate
arms at this size. Tokens per success include cached input (M32). Costs are what the
provider recorded and are lower bounds, because these runs kept only some raw responses.

Third run, after M25 to M29. Shelf: 23 functions from 30 training commands, with 27 of
30 training episodes passing, which cost at least $0.123. The held-out arms cost at
least $0.162.

| Arm | Success | Mean rounds | Longest episode | Tokens per success | Shelf used |
|---|---|---|---|---|---|
| A | 30/30 | 7.0 | 13 | 16911 | n/a |
| B | 29/30 | 4.2 | 7 | 10277 | n/a |
| C | 28/30 | 5.2 | 22 | 20885 | 30% of episodes |
| C2 | 30/30 | 3.4 | 6 | 12554 | as examples |
| D | 30/30 | 6.2 | 13 | 19908 | n/a |

What the third run showed:

- `200001`, in C: "tell the time", with no clock in the world. B and C2 made up a time
  and passed. C guessed function names such as `tell_time` and `clock` through `eval`,
  and the runner refused 60 of those calls as unknown names. They are 60 of the 61
  refused calls in M9, and the other is the `import_stubs` call that `import inspect`
  makes. Then 7 attempts were abandoned at 30 s, about 210 s of the 300 s limit, and the
  episode ended with nothing said. The log cannot show whether those attempts were
  runaway completions or provider stalls. Retries of the same requests finished in 3 to
  6 s. M30 addresses it.
- `200020`, in B and C: the same fair failure as the second run.
- M8 is 4 of 9 shelf episodes. A shelf function raised in `200001`, `200003` and
  `200017`, because `tell_day_of_month(room, description, day)` was passed a number and
  `say` takes text, and in `200016`, because `throw_object_in_trash` picks up the object
  and the model already held it. The other three recovered. `Shelf.describe` gives C
  each definition line and docstring, and none of the 23 functions has a docstring or a
  type annotation, so C cannot see either problem from its prompt. C2 reads whole
  scripts.
- M29 held. `200017` in D, lost to an unfinished call in the second run, passed here
  after 2 abandoned attempts. Across the run 11 attempts were abandoned, against 899
  completed calls.
- Training `100021`: `throw_object_in_trash` searched the room, so the apple did not
  come from the shelf the command named. This is a fair failure.
- Training `100030`: a false failure, which M31 fixes. The scores above predate M31.
- Training `100031`: "follow them" names no destination. The model followed the person
  to three places and kept checking, until 205K tokens passed the 200K limit. 143K of
  those were cached input, which counts toward the limit.

In both runs with a real shelf, the script arms B and C2 spent the fewest tokens per
success, 30% to 39% fewer for B than for A. C spent the most, with more rounds than C2,
which holds the same scripts as examples. Two small runs are a direction to test, not a
result.

A very small check after M32 ran arm B on 3 commands, on the gate digest
`09fd4c35c5483df3`: 3/3, with every call's cost recorded, for a total of $0.0028.

Second run, after M24. Shelf: 22 functions from 30 training commands, which cost at
least $0.074. The held-out arms cost at least $0.164.

| Arm | Success | Mean rounds | Longest episode | Tokens per success | Shelf used |
|---|---|---|---|---|---|
| A | 29/30 | 6.9 | 13 | 17343 | n/a |
| B | 29/30 | 4.4 | 8 | 12113 | n/a |
| C | 28/30 | 4.9 | 15 | 19160 | 37% of episodes |
| C2 | 29/30 | 3.4 | 8 | 13709 | as examples |
| D | 29/30 | 5.8 | 12 | 18035 | n/a |

What the failures were:

- `200020`, in A, B and C: the instruction point is in the living room, so the model
  took its empty view from there for the whole room and fetched a bowl from another
  room. This is a fair failure. The first run failed C and C2 on this command the same
  way.
- `200031`, in C: the model passed "salute" to a shelf function as a person
  description, the function did nothing because its search came back empty, and the
  model spent 15 rounds trying other shelf functions. This is the one failure caused
  by the shelf.
- `200001`, in C2: the last script said its line only `if people`, the search found
  nobody, and the model reported success anyway. The shelf's examples use the same
  guard, though one trace cannot show that they caused it.
- `200017`, in D: a model call never finished and the episode reached its time limit,
  which M29 addresses.

First run, before M24, with an empty shelf, so C and C2 were B under other names: A
30/30, B 30/30, C 29/30, C2 29/30, D 30/30. Mean rounds were 6.7, 4.0, 4.0, 4.2 and 6.1.

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
| P1 | "Ready" means both repositories offer the selected M0 to M5 seam and working independent services, so the walkie-ros plan's work and the scene-graph core can start. The core itself (steps 4.7 to 4.9, 5.1 to 5.3, 5.8) is later work. Amended after the readiness brief: this does not promise that no interface work is ever needed again. | M7's bake-off can need a new vector migration, a second fingerprint needs a versioned interface, GraspNet needs its own dependencies, captions need a later model decision, and baseline formats belong to walkie-ros. | Building the core now would be larger and would fix a design before it runs on real data. | Delegated |
| P2 | Work on local branches named `feat/scene-graph-prep`, in worktrees at `/var/tmp/wv-prep` and `/var/tmp/wd-prep`. Do not push until the owner agrees. | Direct commits to main are allowed, but other people read and pin those public branches, and these interfaces are unproven. Git also cannot check out main in a second worktree. Named branches also keep commits reachable, which the brief's detached worktrees do not. | GitHub pins cannot reach the commits, so the demo builds against local checkouts until a push. | Delegated |
| P3 | Codex writes the brief, Claude builds from it, and Codex reviews the result. | This follows S4. | None. | Delegated |
| P4 | Adopt the readiness brief's 36 defaults, listed below as B1 to B36. | They were checked against both repositories, and none blocks the build. | Each can be reversed at the cost stated in its row. | Delegated |
| P5 | Build all eight vision slices and all nine database slices, not a smaller demo set. | The request is readiness ahead of later work. A smaller set would leave the seed, map gate or services unusable by their callers. | The brief sizes this at about 34 engineer-days and about 16 GB of disk. A people-only slice would reach the HRI demo sooner, but would not meet the request. | Delegated |
| P6 | The agent's real people backend is its own slice, built after the ROS wiring merges. `UnavailableVision` stays until then. | `/walkie_vision/people` already exists and its fields do not change, so this slice does not wait on the object contract. | The HRI demo cannot see people until that slice lands. | Delegated |
| P7 | Keep a copy of the brief in git at `plans/scene-graph-prep-brief.md`. | `/var/tmp` has been cleaned by hand before, and the brief is expensive to rebuild. It sits outside `docs/`, so the strict docs build does not check it, and outside the Mahoraga gate digest. | Its line references go stale as the repositories change. | Delegated |
| P8 | The build runs vision and database stages side by side, with database stage D7 starting after vision stage V4. Each slice is one commit. No review runs inside the build. | The brief lists V4 as a dependency of D7, and S4 keeps reviews with Codex. | Bugs found in review come back as later commits instead of being caught slice by slice. | Delegated |

### Readiness brief defaults

Numbers match section 6 of the brief. All are status Delegated. Hardware performance
is not settled by any of them: on-demand detection latency, CLIP latency and memory,
pairing, colour accuracy and localizer error stay open until measured.

| ID | Decision | Why | Alternative and consequence |
| --- | --- | --- | --- |
| B1 | "Ready" means the selected M0 to M5 seam and working independent services, not a finished scene graph. | The core is excluded, and types alone would leave depth, embedding and seed callers unusable. | M0's full consumer exit and the M1, M2, M4 and M5 behaviour stay separate gates. |
| B2 | Persistent object IDs are positive int64 database IDs, shown to the agent as decimal strings. Live `obj_` references stay separate. | Plan decision 4 rejected the `mem_` registry. | Fake-world IDs such as `apple_1` are not a naming rule, and no string may be read as a track ID. |
| B3 | IDs are never reused within a map and store lineage. An offline reset carries the ID high-water mark, sets a fresh `store_id` and relaunches the agent. | Resetting counters from the seed would reuse IDs, and UUID keys would break the int64 API. | Restoring a seed restores logical contents, not an identical file. |
| B4 | Keep the old database messages and add `ObjectSummary` and new services. | Expanding `RememberedObject` would break the working `FindObjects`. | The old `find_objects` stays limited, so the agent needs a new memory capability. |
| B5 | Edit `ObjectDetection` once for quality and colour, and `ObjectDetectionArray` once for `stream_id`. | A second envelope type would double the streaming paths, and restarts are hard to detect without the ID. | Every object consumer must rebuild together. People fields stay stable. |
| B6 | Add colour shares now and keep captions disabled. | Masks are discarded, so adding shares later would need another type change. | The field stays empty in HRI mode, and the palette order becomes part of the contract. |
| B7 | Use CLIP ViT-B/16 OpenAI with QuickGELU, 512 dimensions and a fixed template and crop margin. | SigLIP has not won a bake-off, and DINO is not the default. | SigLIP thresholds do not transfer. A new margin changes what stored vectors mean and needs re-embedding. |
| B8 | Recreate `object_clip_vit_b16` with map, room, state and layer columns. | Filtering after a global top-k loses valid answers. | Every insert site and fixture must supply the metadata. A vec0 table cannot be renamed. |
| B9 | On-demand YOLOE is a separate instance in the same process, and only the caller gets the result. | Re-prompting the stream corrupts tracking, a second topic conflicts with ADR-0009, and a second process conflicts with ADR-0010. | A new class list stalls the shared worker, reported at about 1.5 s. Second-model memory is unmeasured. |
| B10 | The agent forwards `DetectObjects` results through `SubmitDetections`, and `ReportSighting` carries target-specific evidence. | Vision publishing on its own would mix acquisition intents, and a bare boolean cannot prove a miss. | The same array must be deduplicated across both calls. Memory does not change before M1. |
| B11 | `LocalizeObjects` lives in the database as a stateless centroid service and arrives ahead of the core. | TF in vision breaks ownership, geometry in the agent duplicates policy, and waiting for M1 blocks localization. | It returns per-detection status and no persistent ID, so it does not replace `GetObPose` directly. |
| B12 | The fake observation publisher lives in vision, outputs points and has no embeddings by default. | The agent's oracle has no geometry, and fake vectors under real names would contaminate memory. | It validates plumbing, not detector quality or occlusion. |
| B13 | Fake embedding model names start with `fake_`, and the real store refuses them. | Vectors posing as CLIP would make semantic tests meaningless. | Artificial vectors are allowed only in isolated test databases. |
| B14 | The map hash helper is a standard-library module owned by walkie-database, with YAML parsing outside it. | A new shared repository adds release work, and separate copies drift. | Consumers need a source pin even when they use no database runtime. |
| B15 | `map_id` hashes normalized decoded pixels plus semantic metadata, and the gate checks the full digest. | Hashing file bytes changes on a harmless re-save, and an 8-character suffix alone is too weak. | Pixel parsing and float formatting become versioned public behaviour. A change means new map IDs and a reseed. |
| B16 | Rooms and anchors persist across tests. Events and movable objects do not. Online `ArenaReset` only lowers active movable belief. | A file reset over ROS is not needed. | Events last one test. An online reset keeps age and held-skill state. |
| B17 | Seeds come from reviewed room and anchor files with stable keys and optional real fingerprints. | Navigation waypoints and degenerate polygons do not give boxes, and seeding from a live database leaks movable objects. | A rooms-only seed is valid. An anchor with no geometry cannot support on, inside or grasp. |
| B18 | Crop embedding needs the current image within 5 ms. There is no historical image service. | Keeping history needs new memory rules, and an old box may embed the wrong object. | Proposal embeddings may often refuse under latency, which leaves the candidate honestly unknown. |
| B19 | Structured `ObjectQuery` replaces a parser, and superlatives are a finite enum. | Free-text filters weaken validation, and a second LLM parser adds latency. | Unknown attributes and weights stay explicit. Room contents are not the current view. |
| B20 | Use an exact filtered scan at arena size. | A global top-20 before filtering can lose valid answers. | It may cost more CPU than KNN, and is measured at 500 objects. |
| B21 | `DescribeCrop` and `DescribeObject` are reserved and disabled. Florence-2 with deterministic colour is the future default. | A VLM adds unknown memory use and quality, and the plan allows cutting captions. | There is no caption table or queue. Empty attributes mean unknown. |
| B22 | One database connection owner now, behind a repository API that a live cache can replace later. | A writer thread now would pull in core scope. | M1 must supersede the executor ADR and prove read-your-writes. |
| B23 | The map gate lives only in the database, and preflight reads `GetStoreStatus`. | Launcher copies of the gate would drift. | A node that fails the gate does not start, so preflight sees it as unavailable and reads the startup log. |
| B24 | All development tests use domain 77 and up, Fast DDS and localhost, with no router. | Production settings would join the live system. | A Fast DDS pass is not a Zenoh or robot pass. |
| B25 | Each prep worktree has its own `.pixi` and overlays. | The main manifest and editable paths silently target old code. | It costs about 12 GB plus about 4 GB of builds. Installation stops below 8 GB free. |
| B26 | Consumers move to the final tested commits. The database baseline is 64403e9. | The agent's 7f657b0 lacks newer services. | Old object calls can keep working. New imports need pin and build updates. |
| B27 | The M0 contract gate does not need the world repository. | Only the M1 exit needs real free-joint truth. | M0 can run now. "Every body Present after patrol" stays unmeasured. |
| B28 | Budget 24 GB on the robot and 6 GB in development. Optional models stay off outside the memory profile. | Assuming everything fits has no evidence behind it. | GPU startup, memory and latency block real-model use, but not the fake build. |
| B29 | No photo tool, no D415 consumer, no pets, no extra detector and no second embedder. | None has a demonstrated need. | `DescribeCrop` stays internal and gives the LLM no image access. |
| B30 | Accept small baseline leftovers, with no automatic speck filter. | Baseline rules belong to later evidence logic and ROS capture. | Fake tests cannot show missing-object timing on small or transparent objects. |
| B31 | Defer GraspNet and keep the old perception services. | Moving it brings untested CUDA, TF cache and MoveIt work. | The new type compiling does not mean grasp or localize replacement is complete. |
| B32 | Scene input stamps use the deployment ROS clock, service deadlines are monotonic, and people timestamps move to the node clock in sim. | `time.time` in the people code contradicts the one-clock plan. | It needs a small injected `now()`. Old wall-time people data cannot be compared with sim times. |
| B33 | Hash, camera and baseline file contracts are versioned. The raw grid and proposal format is a walkie-ros prerequisite. | Standard messages do not specify grouping or encoding. | M3c cannot finish before that contract exists. |
| B34 | No library upgrades. Add only the declared preparation and M1 CPU needs. | Borrowing the agent's environment or moving ROS builds causes ABI work. | The database adds vision-msgs, sensor-msgs-py, tf2-ros, PyYAML, SciPy and scikit-learn. |
| B35 | Reported NAIS figures and transferred thresholds count as unverified. CLIP prototype thresholds apply to fixtures only. | Nothing here verified them. | The 50-phrase and robot-camera benchmarks stay open. |
| B36 | Seed tools need explicit paths and `--replace`, and a fresh-seed marker refreshes timestamps once. | Refreshing on every boot falsely raises stale belief, and implicit defaults risk live data. | In sim, a clock sample must arrive before fresh anchors are stamped. |

## Still with the team

These stay open, and the prototype takes no position on them beyond working
around them.

- Decision 9: leave RAI. The prototype keeps RAI's connector for now.
- Decision 18: where real places live.
- Decision 19: the local fallback model.
- The walkie-database commit the robot runs.
