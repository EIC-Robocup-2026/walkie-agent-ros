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
| M30 | A `talk` command that asks for a fact gets a source and a check: a clock and team facts on `ctx`, the same in the fake world and on the robot, and a checker that compares the line for those topics. "Something about yourself" stays unchecked. To build after the ROS merge, which is changing the same capability files. | Third run: 4 of 30 held-out and 8 of 30 training commands ask for a fact that nothing supplies. Every arm invented one ("It is 3 o'clock"), and in A, B and D the line "What day is tomorrow?", which reads the command back as a question, passed. The robot has no clock either, so it would state a made-up time. The hunt for a clock also caused the longest failure of the run. | Keeping the documented limit leaves about 13% of held-out commands free in every arm and the robot inventing answers. The cost is one more capability, a new function in every arm's prompt, and a gate rerun. Runs before this are not comparable on these commands. | Delegated, built (M35 to M41) |
| M31 | A later look with no answer after it no longer discards an earlier answer. The checker judges the last look that a line naming a candidate follows. | Third run, training command 100030: the model said the right answer, "chips bag", then looked again without speaking and scored 0. The limit was documented, and a real model hit it. | A right answer, a look and a wrong answer still fails, and a wrong answer, a look and a right answer still passes. Only a silent look after an answer changes, and a wrong answer followed by one still fails. | Delegated |
| M32 | M3's input counts everything the model was sent, with the cached part also given alone. Its cost is what the provider recorded in each raw response, with calls that have no record counted beside it, and `run` sets `log_model_api=True`. The plan's prices are gone from the report. | Third review, N1 and N2. Inspect's `input_tokens` leaves out cache reads, about 90% of input here, so a figure of 2177 tokens per success was really 17343, and the order of the arms changed. The plan's prices made the second training run read $0.023 when the provider recorded at least $0.074. Inspect keeps only the first few raw responses unless told otherwise. | Logs from before this have a lower bound for cost, and larger logs follow from now on: 31 KB for 3 commands. Abandoned attempts have no record, so they remain a lower bound. A provider that returns no cost reports null, with no estimate. | Delegated |
| M33 | A raise counts against the version that ran, whether or not the script saved it. | Third review, N3. A failed script that defined the same text as a saved version lost its raise, because only saved or shelf versions were kept, and a later recovery then verified that version. | None seen. A version that never reaches the shelf only adds an entry nothing reads. | Delegated |
| M34 | "Forearm" counts as an arm, and a line naming a leg, knee or foot says no raised arm. | Third review, N4 and N5. The arm requirement of M27 rejected "Their right forearm is raised" and still passed "Their hand rests on their raised right leg", since words are not attached to each other. | A true arm answer that also mentions a leg now fails. Limits left as they are, all low in likelihood and short of parsing language: "I can see the person's feet" reads as standing, "waist-high" holds "high", "rightward" and "palm" are not read, and "TV stand" reads as standing. | Delegated |

These build M30.

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| M35 | A capability `ctx.facts`: a `Facts` Protocol with `now()`, a timezone-aware datetime, and `team()`, a frozen `Team` of name, country and affiliation. `Ctx.facts` defaults to `UnavailableFacts`, which raises `UNAVAILABLE`. The atomic `get_facts` returns the time as 24-hour `HH:MM`, today and tomorrow such as "Thursday, 5 November 2026", the weekday, the day of the month as a number, and the three team values. It joins GPSR and the default set, not HRI. | A talk command that asks for a fact needs a source the robot and the fake share. | One more verb set on `Ctx` and one more tool in two sets. GPSR is no longer the exact set the spike used. | Delegated, built |
| M36 | `SystemFacts` reads the wall clock, not ROS time, in a configured timezone, and imports no ROS. | The time of day is the computer's, and simulated time would give a wrong answer in the simulator. | A robot with a wrong system clock states a wrong time. Nothing checks the clock against a time server. | Delegated, built |
| M37 | `config.toml` gains `[team]`: `name = "EIC"`, `country = "Thailand"`, `affiliation = "Chulalongkorn University"`, `timezone = "Asia/Bangkok"`. `main` builds `SystemFacts` from it. A missing table, a blank value or an unknown timezone stops the agent with a one-line error before `rclpy.init`. `robot.md` lists telling the time, the date and the team. | The team values belong in settings, not code, and a fact source that fails late would answer `UNAVAILABLE` during a test. | Every `config.toml`, including a copy made with `rai-config-init`, now needs the table, or the agent does not start. | Delegated, built, exit code set by M56 |
| M38 | In the fake world the clock is fixed for an episode and drawn from the seed: any date in 2026 and any minute of the day. The team is M37's. Both are in the world and in the snapshots. | The checker needs the value the agent could read, and one seed must give one answer. | The checker reads the start snapshot. A fake clock that moved during an episode would need the time of each line. | Delegated, built |
| M39 | The talk checker judges the fact a line says, from the start snapshot's facts. Any line in the person's room after the lookup that found them counts, so M22's first-line rule does not apply. A time is `H:MM` or `HH:MM` in 24-hour form, or `h:MM` in 12-hour form with an optional am or pm, and "h o'clock" counts only on the hour. A weekday is a whole word. A date is a month's name and the day's number in either order, with an optional ordinal suffix. A day of the month is digits with an optional suffix, or a word from one to thirty-one. A team value is a whole-word phrase in any case. "what day is today" takes today's weekday or date, "the day of the week" today's weekday, "what day is tomorrow" tomorrow's weekday or date, and "the day of the month" today's number. This narrows M4 for talk. | Third run: every arm made up a fact, and in A, B and D "What day is tomorrow?" passed. | Known limits: "quarter past two", "half past" and other times in words fail, and a line that names both today and tomorrow passes for either. | Delegated, built, narrowed by M53 and M54 |
| M40 | One sentence in the shared prompt of `mahoraga/arms.py` tells every arm to read the time, the date and the team with `get_facts` and never to guess a fact. The oracle's `talk` scripts call `get_facts`. The bundles are unchanged. | A tool the model is never told about is easy to miss, and the third run's longest failure was a hunt for a clock. | Runs before this are not comparable on talk commands. | Delegated, built |
| M41 | "Something about yourself" stays unchecked beyond a line in the person's room after the lookup that found them. | Nothing defines a right answer about the robot. | A refusal to the right person still passes that topic. | Delegated, built |

These came up while building M35 to M41.

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| M42 | `assemble` takes `facts` with `UnavailableFacts` as the default. `main` reads `[team]` after the model, the fallback and the tracing callbacks are built, and strips each value. Extra keys in `[team]` are ignored. | Existing callers of `assemble` keep working, and the existing config error tests still reach the error they test. | A config with both a model error and a `[team]` error reports the model error first. A misspelled extra key such as `timezon` reads as a missing `timezone`, not as an unknown key. | Delegated, built |
| M43 | The robot agent's system prompt gains a rule for `get_facts` in the sets that bind it: call it for the time, the date or the team, and never guess a fact. | The robot agent would otherwise guess, as every arm did in the third run. Like the other tool rules, it appears only in a set that binds the tool. | Two more lines in the default and GPSR prompts. | Delegated, built |
| M44 | The fake's clock comes from a random stream of its own, seeded with `facts:{seed}`, at a fixed UTC+07:00. The world seed is per (command, epoch), so each epoch gets its own clock. | The rest of every world stays as it was at `a8ef63a`, which a test pins for three seeds, and the offset matches Bangkok. | No daylight saving or other offsets are exercised in the fake world. The fake does not read `config.toml`, so a changed team there leaves the fake's team as it was. | Delegated, built |
| M45 | A world description may leave out `facts`. Both facts verbs then raise `UNAVAILABLE`, the snapshot's `facts` is `None`, and a checked talk topic raises `ValueError` in the checker. The facts verbs are recorded in `calls` and do not check the cancel token. | Hand-written test worlds stay valid, and reading a clock takes no time. | A hand-written world must add `facts` before it can test a checked talk. | Delegated, built |
| M46 | Time forms: an am or pm must be right when given. An hour with a leading zero and no am or pm is 24-hour only, so "02:05" is not 14:05. "o'clock" needs the hour in digits. A time with seconds is not read. | Each choice keeps a wrong time from passing on a loose reading. | "two o'clock" and "14:05:30" fail although they are right. | Delegated, built, extended by M55 |
| M47 | Date forms: "of" and "the" may sit between the number and the month, and a leading zero is allowed. The ordinal suffix is not matched to the number, the year is not read, and a month's short name such as "Nov" is not read. | The common spoken and written forms pass, and nothing in a command asks for the year. | "5rd November" passes, a wrong year passes, and "5 Nov" fails. | Delegated, built |
| M48 | For the day of the month, any number in the line counts, as digits or as a word from one to thirty-one. Digits beside a colon are a time and do not count. | Telling a pronoun from a number needs language parsing, as in M22. | "It is the one you wanted" passes on the 1st, and "wait a second" on the 2nd. | Delegated, built |
| M49 | A talk topic with no check raises `ValueError` in the checker, and one with no line raises it in the oracle. | A new topic from the generator must not pass or fail unnoticed. | A generator change stops the gate until the topic has a check and a line. | Delegated, built |
| M50 | The arm tests that script a generic line now pick a "something about yourself" talk command, and the talk trace tests say the fact through `get_facts`. | Their lines say no fact, so they would now fail for a reason they do not test. | The arm tests run more mock episodes to reach that command: 33 on the trial split, 6 on the training split and 68 on the held-out split, against 9, 3 and 2 before. | Delegated, built |

This settles finding R2 of the merge review and its re-check, R2-a.

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| M51 | While the shelf can still verify, a script's own top-level definitions are watched whether or not it saves them. Every definition no wrapper covers, at any depth, is kept as a (name, sha) pair, one per version, and counts as raised for the episode. If the counters cannot be read, every version the run watched counts as raised. Frozen and shelf-less runs watch no own definitions. | R2: a raise through an unsaved copy of a shelf version went unseen, so a later success verified that version. R2-a: the first fix kept one entry per name, so an earlier or conditional definition of the same name was lost. | Measured on writable shelves: an unsaved function now runs behind a wrapper, so `locals` and `eval` are shadowed for it, recursion through it reaches the depth limit in half the calls, its stub allows a forward reference, and the run's token can appear in the printed text of an error the script catches. An unwatched copy blocks verification of its text in that episode even when it ran cleanly. Frozen and shelf-less runs behave as before. | Delegated, built, sha fixed by M52 |

These settle findings F1 to F6 of the 2026-11-05 review of M30 and the merge.

| ID | Decision | Why | Alternative and consequence | Status |
| --- | --- | --- | --- | --- |
| M52 | A definition's sha is taken from its text without the indentation of the scope it sits in. Each line after the first loses the definition's column offset of leading characters when they are all whitespace, and a line with fewer, such as a multi-line string line at a shallower indent, stays as it is. A top-level definition keeps its text, so every sha on a shelf stays valid. | F1. `ast.get_source_segment` drops the scope's indentation on the first line only. A two-line copy of a saved function under an `if` or inside another function therefore had another sha, a raise through it missed the saved version, and a later success froze that version. | The match is textual. A copy whose multi-line string lines were indented along with the rest shares a sha with a top-level version whose string was not, which errs toward blocking. A top-level definition after a form feed on its own line has a column offset of 1, so a new save of it writes one space less on each later line. None was found among 45,178 top-level definitions and 5,454 shelf files on this machine. | Delegated, built |
| M53 | A checked talk topic passes only on a line said after a successful read of its fact: `facts.now` for the time and the four day topics, and `facts.team` for the three team topics. The read may come before or after the lookup. "Something about yourself" needs none. A right line with no read before it fails, and the reason says so. This narrows M39. | F2. "We are EIC." passed with no read, and since the team is the same in every world, a remembered answer always passed and measured nothing. | The checker sees order, not data flow, so a robot that reads and then says a value it did not take from the read passes. A robot that knew the answer some other way fails. Talk results from earlier runs are not comparable. | Delegated, built |
| M54 | The day topics ask about today, except "what day is tomorrow". A line that names another of today, tomorrow and yesterday, as a whole word in any case, and not the day asked about, does not count for it. A line that names both today and tomorrow still passes for either, as in M39. | F3. "Tomorrow is Thursday." answered "what day is today" on a Thursday, and "Today is Friday." answered "what day is tomorrow". | A right line that also names another day fails, such as "It is Thursday, see you tomorrow", or "Tomorrow's date is 5 November" said for today. | Delegated, built |
| M55 | One reading of the hour serves `H:MM`, "h o'clock", and a digit hour right before a qualifier, such as "12 am", "2 p.m.", "2 in the afternoon" or "12 noon", which is that hour on the hour. The qualifiers are M46's am and pm, "in the morning" for am, "in the afternoon" and "in the evening" for pm, and "noon" and "midnight", which count only after 12 on the hour, at 12:00 and 00:00. A qualifier must be right when given. M46's leading-zero rule covers every form, so "02 o'clock" is not 14:00. Digits right after a digit and a dot are not an hour, so "2.05 pm" is not 5 pm. This extends M46. | F4 and F5. Only am and pm were read, so "12:00 noon" passed at midnight and "2:05 in the morning" at 14:05, and the o'clock form skipped the leading-zero rule. "12 am", "12 pm" and "2 p.m." failed although they are right. | A qualifier goes with the 12-hour value, so a right 24-hour time with one now fails, such as "14:05 in the afternoon" or "00:00 midnight", as "14:05 pm" already did. "at night" is not read, so "10 at night" fails at 22:00 and "2:05 at night" passes at 14:05. "November 5 in the morning" reads as 05:00 for the time topic, "12 in the evening" passes at 12:00, and "14 o'clock" now passes at 14:00. | Delegated, built |
| M56 | Before `rclpy.init`, `main` catches OSError, ValueError and KeyError from reading its inputs. It prints one line on stderr, `walkie-agent: <message>`, with the lines of a longer message joined, and returns 2. A KeyError reads `<config path>: missing key '<key>'.`. `build_model` turns the OpenAI client's missing-key error into a ValueError, and `load_config` and `load_task_templates` put the file's path before a TOML syntax error. A failure once ROS has started keeps its traceback and code 1. | F6. M37 promised a one-line error, but the ValueError escaped with a traceback. Code 2 is what argparse already returns for a bad command line. Without a traceback, the line itself has to name the broken file. | A single value where a table belongs, such as `vendor = "openai"`, still ends with a TypeError or AttributeError traceback and code 1, because catching those would also hide real bugs. A file that is not UTF-8 gives a one-line error that does not name it. A startup error now exits 2, not 1, and `main()` returns 2 instead of raising. | Delegated, built |

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
  after 2 abandoned attempts. Across the run 11 attempts were abandoned at the 30 s
  limit, and 1 more was cancelled when its episode ended, against 900 completed calls:
  A 210, B 127, C 147, C2 101, D 185 and the shelf build 130.
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
| P9 | Keep the nine stage reports, the handoff and the pair gate log in git at `plans/scene-graph-prep-build/`. This log lifts only the build decisions and deviations a consumer or the demo can see, as P10 to P31. The other build decisions stay in the reports, each with its cost. | The reports hold 167 build decisions and 63 deviations, and `/var/tmp` has been cleaned by hand before. Copying every one here would bury the ones that change behaviour. | The reports' line and commit references go stale as the repositories change, and the strict docs build does not check them. | Delegated |
| P10 | Vision publishes observation arrays as RELIABLE, VOLATILE, KEEP_LAST 5 from both the real node and the fake publisher. Brief line 145, which says BEST_EFFORT and KEEP_LAST 1, is corrected to match. | A RELIABLE publisher matches both RELIABLE and BEST_EFFORT subscribers, and the database subscribes BEST_EFFORT, KEEP_LAST 1. Moving the real node to BEST_EFFORT would leave any RELIABLE subscriber with no data and no error. The fake must behave like the real node for the pair gate to mean anything. | A slow RELIABLE subscriber can make the publisher hold up to 5 large arrays. This is unmeasured on the robot and on Zenoh. | Delegated, not built: the fake publisher and brief change wait for the vision review |
| P11 | The agent's pins and build changes from section 8 of the brief wait until the vision and database review fixes land, and then go in with P6, the people backend. | The fixes may move both final commits, and the pins would then move twice. P6 is the first agent code that needs the new build. | Until then the agent builds against walkie-database 7f657b0 and no walkie-vision messages. A plain `pixi install` in an agent worktree would also change `pixi.lock`, so the pin change must use the locked recipe. | Delegated, waiting for the review fixes |
| P12 | Under `pixi run --clean-env`, the commands call `/usr/bin/env` and `/usr/bin/bash`, and colcon runs with `/usr/bin:/bin` appended to `PATH`. | `--clean-env` leaves only the environment's own `bin` on `PATH`, which has no `env`, `bash`, `git` or `timeout`, so the brief's commands fail as written. | Commands copied from the brief fail with `env: command not found`. walkie-database's `tools/import_interfaces.py` finds `git` through the same fallback, and `git` is not a conda dependency. | Delegated, built |
| P13 | An array's `stream_id` is a canonical lower-case UUID made once by the vision pipeline. The database's memory-mode ingest refuses an array whose `stream_id` is not one. | Without the ID a consumer cannot tell a vision restart from a gap. | Any other publisher of `ObjectDetectionArray`, such as a future agent or simulator oracle, must set a UUID or memory mode refuses all of its arrays. | Delegated, built |
| P14 | The real vision node drops an object whose score is outside 0 to 1 and counts it, instead of failing the frame. | One bad score should not lose the frame's other objects. | Before this, a score above 1 was published. A detector that reports scores on another scale loses every object, seen only in the counter. | Delegated, built |
| P15 | The `fake-detections` task fixes Fast DDS, localhost and domain 77, and publishes on `/walkie_vision/objects` by default. | It gives one known isolated form for tests and the pair gate. | A shell `ROS_DOMAIN_ID` does not move it, so another domain needs the explicit `env` form from the guide. It must never run in a domain with a real vision node, because both use the same topic. | Delegated, built |
| P16 | With geometry on, depth pairs with an image only if the camera reports zero distortion, the rays agree within 1 px, and the depth stamp is within 20 ms. Otherwise the image publishes no points, or no array at all under `require_depth`, with a throttled warning. | Points from a distorted or misregistered depth would put objects in the wrong place. | Whether the ZED reports zero distortion for rectified images, and its real stamp gap, are unmeasured. If either fails on the robot, objects arrive with no points and memory cannot place them. | Delegated, built, unverified on the robot |
| P17 | `pixi run fetch-models` always fetches CLIP, and CLIP is not warmed up at start. | One fetch command for every task keeps setup simple. | An HRI-only machine downloads about 335 to 571 MB it does not use. The first embedding after start can be slow. | Delegated, built |
| P18 | The embedding and caption services always exist. They answer `DISABLED` when their flag is off, and `NO_IMAGE` at once when no image has arrived. A start with `captions_enabled` true is refused. `EmbedText` failures put their code at the front of the message, because the type has no code field. | A caller can find the service at the same name in every task, and an answer at once is easier to handle than a timeout. | HRI has three more endpoints with no model behind them. Callers must retry `NO_IMAGE` themselves and parse the `EmbedText` prefix. A caption slice must lift the refusal. | Delegated, built |
| P19 | The vision watchdog covers service jobs, and an embedder exception fails the image like any other provider error. | A hung job would otherwise block the worker with no restart. | A hung job restarts the node, and a broken CLIP counts toward the 30 failures that restart it. | Delegated, built |
| P20 | `DetectObjects` builds and warms its own YOLOE at start, with the task's classes. A new class list answers `BUSY` for 2 s after the last preparation, and a reordered list counts as new. | B9 keeps the on-demand model apart from the stream, and warming at start keeps the first call fast. | Start is slower and the second model's GPU memory stays in use while the flag is on. Its size is unmeasured. Callers must retry `BUSY` and keep their class order stable. | Delegated, built |
| P21 | Each sim worker has one store, restored to the seed before each episode, instead of ADR-0016's new file per episode. | A new path restarts the ID counters, which B3 forbids within a lineage. | IDs differ between episodes, so a scenario or checker must never name a literal object ID. | Delegated, built |
| P22 | Memory mode never creates a store. A store must be restored from a seed first. A legacy people-only store cannot enter memory mode at its own path. | A wrong `db_path` should fail loudly, not start an empty memory. | The run recipe must call `db-restore` before the first start. The robot's existing people file needs a restore to a new path before memory mode can use it. | Delegated, built |
| P23 | The sim seed reads each room's stand pose from a locations file, under exactly the room's display name, such as `[locations."living room"]`. | The room-names file maps keys to display names, and one spelling per room keeps the two files from drifting. | The agent's `config/locations.sim.toml` holds blank-map test points, so the seed builder refuses it. `chula_pat14_6` needs its own locations file, which does not exist yet. The agent's loader accepts names with spaces, so one file can serve both. | Delegated, built |
| P24 | `FindObjects` leaves out discarded and merged objects and anchors with no geometry, also when memory mode is off. | Without the filter a geometryless anchor comes back at a zero box, which is a false place. | This goes past the brief, which kept the old behaviour with memory off. Only rows that D6 introduces are affected, because legacy rows default to valid geometry. | Delegated, built, in review |
| P25 | With `use_sim_time`, `start_session` and `remember_person` fail with the retryable prefix until the first `/clock` sample, also when memory mode is off. A runtime change of `use_sim_time` is refused. | A time of 0 before the clock arrives would stamp people at the epoch. | In sim the agent's first session can be refused for a moment after start and must retry. This goes past the brief for memory-off mode. | Delegated, built, in review |
| P26 | The database takes a lock on its store file, also when memory mode is off. | Two writers on one SQLite file would break the ID and ledger rules. | A second `db-node` on the same file now exits with code 1, where two used to run. The lock does not hold over NFS. | Delegated, built, in review |
| P27 | Memory-mode ingest refuses a whole array when any cloud has 1 to 29 points. | The vision contract says such a detection carries an empty cloud, so a short cloud means the array is malformed. | One malformed detection loses the whole array. A publisher other than walkie-vision must follow the empty-cloud rule. | Delegated, built, in review |
| P28 | Room and event queries answer `NOT_FOUND` for an unknown room. An event's pose comes from a TF lookup at the node's time with a 50 ms wait and no fallback. `find_events` treats `since` as inclusive. | A typo should be a refusal, not an empty list, and a stale pose is worse than none. | A line written while TF lags has no pose, and the wait blocks the database's executor. A poller sees its last line again and must skip it. | Delegated, built |
| P29 | Forgetting a session deletes every event that names one of its guests, including a guest's event filed under another session or none. A forgotten line's ledger hash stays, and `event.id` can be reused after a delete. | The brief's narrower rule could leave a guest's line with no person through raw SQL. The other two follow the brief's schema as written, and ADR-0022 records them. | The stored hash could confirm a guessed sentence, though it cannot be read back. A recorded ledger outcome can name an ID that later means another line. | Delegated, built, in review |
| P30 | walkie-database's harvest page puts `remember_event`, `find_events` and the five object writes on the agent's never-list, with the people services. | They write or return lines that can carry a `person_id`. | The demo has no never-list. It holds the rule by construction: generated scripts reach ROS only through the shelf's atomics, and no atomic calls a database service. The never-list must name these services before any Unrestricted harvest run. | Delegated, built, agent side open |
| P31 | Both repositories' CI is left without the pair gate, with a comment in walkie-database's workflow. Links to the agent's pages point at commit 91ffe25. New ADRs 0021 to 0024 stay Proposed, and ADR-0019 is not edited. | CI cannot fetch an unpublished pin, and no replacement agent pages exist yet. Accepted ADRs are not rewritten. | The gate runs only on this machine until the pins are pushed. Remote CI, including the 9.6 GB vision environment's cache, has never run. ADRs describe a schema already in use while Proposed, and someone must accept them. | Delegated, built |

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
