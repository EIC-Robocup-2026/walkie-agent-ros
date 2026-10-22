# Competition Mode (Proposed)

!!! note "Status: Proposed"
    Nothing on this page is built yet. It is the design for team review. The
    [open questions](#open-questions) must be answered before the
    Competition work is finished.

Competition is the access level the robot runs at in RoboCup@Home tests and
at the booth. People talk to it, nobody from the team may type or approve
anything, and the internet may disappear at any moment. The agent starts at
Competition unless told otherwise, and the sim checks can only lower the
level, never raise it ([Access Levels](access-levels.md)).

This page describes what the self-evolving agent may do at that level, how
long each path takes, and what keeps its speech, its network use and its new
tools safe. How tools are found, composed and created is in
[Self-Evolving Agent](self-evolving-agent.md). How sim runs grow the tool
library is in [Sim Harvest](harvest.md).

## Summary

- **A profile per test** decides whether the agent may reuse, compose,
  create or search the web.
- **Cheapest rung first.** Act directly, reuse, compose, and create code
  only as a last resort. Pre-baked tools answer common requests in 3 to 5 s.
- **Sounds as well as speech.** `play_sound` and `fetch_sound` send clips in
  the `audios` field of `/to_human`, so a sound waits for earlier speech and
  obeys half-duplex, stop and pause
  ([ADR-0045](decisions/0045-play-sound-through-to-human.md)).
- **A speech gate on everything said.** Local checks always run, plus a
  cloud check on risky text at Competition. If that check fails, creation
  and search are off.
- **Search without a text argument.** `look_up_online` queries only with the
  person's own words, returns at most three short snippets and taints the
  turn.
- **Named sources only.** Generated tools reach a short, human-edited list
  of APIs through pinned request templates
  ([ADR-0049](decisions/0049-competition-internet-egress.md)).
- **Offline means reuse only.** Anything that needs the network fails fast.

## Run profiles

`[walkie.mode] profile` in `config.toml` is set before a test. Creating code
is settled per profile. The web-search column is a recommendation.

| Profile                            | Reuse    | Compose     | Create                            | Web search                        |
| ---------------------------------- | -------- | ----------- | --------------------------------- | --------------------------------- |
| `booth`, `open_challenge`          | yes      | yes, online | yes, online                       | yes, online                       |
| `practice`                         | as booth | as booth    | as booth                          | as booth                          |
| `finals`                           | yes      | yes, online | yes, online, person requests only | yes, online, person requests only |
| `gpsr`                             | yes      | yes, online | no                                | yes, online (recommended)         |
| `hri`, `pick_and_place`, `laundry` | yes      | no          | no                                | no                                |
| `restaurant`                       | yes      | no          | no                                | no                                |

- **`practice`** is for rehearsals on the real robot. **`harvest`** is never
  used at Competition. In sim, the access level decides instead.
- **Fixed-task tests** gain nothing from writing code mid-test, and 30 s
  without visible progress removes the robot. Restaurant always behaves as
  offline, since it assumes no wireless.
- **GPSR** composes but does not create. A command that would have needed a
  new tool is logged as a harvest miss, so the sim learns it before the next
  event. Dropping search from GPSR too would remove any chance of a wrong
  spoken fact in a scored test, at the cost of unanswered questions.

## The decision ladder

The agent takes the cheapest rung that works. Times run from the end of the
person's speech and are estimates until measured on the robot PC.

| Rung              | When                                                                                                    | Time         |
| ----------------- | ------------------------------------------------------------------------------------------------------- | ------------ |
| 0. Act directly   | 3 or fewer calls of tools already bound, with literal arguments                                         | 1.5 to 3 s   |
| 1. Retrieve       | Hybrid search (under 0.1 s) finds a tool above the threshold, pre-baked tools included; one or two turns | 3 to 5 s; `look_up_online` about 7 to 8 s, filler first |
| 1b. Fetch a sound | `play_sound` returns `NOT_FOUND` and `fetch_sound` finds a clip on Wikimedia Commons; no LLM call        | 6 to 9 s     |
| 2. Compose        | A bundle of exposed atomic tools passes the linker, with effects within R, P and S                       | 6 to 10 s    |
| 3. Create code    | Rungs 1 and 2 failed, and the request needs live data, exact computation, aggregation over perception or a bounded timing loop; online, with budget left | 20 to 35 s |
| 4. Refuse         | None of the above; a polite refusal with an alternative                                                 | about 2 s    |

Effects are R (reads), P (perceives or computes), S (speaks), N (moves the
base) and M (moves an arm or gripper), so a bundle composed here never moves
the robot. Creating code does not pay off for saying a text, translating,
static facts, facts `look_up_online` can answer, or one call of an existing
tool.

**Pre-baking is the main answer to latency.** The likeliest booth requests
are written and reviewed before the event, so they sit on rung 1:
`tell_time`, `tell_date`, `count_people_by_clothing`,
`count_objects_on_placement`, `compare_object_property`, `countdown`,
`weather_forecast`, `exchange_rate`, `wikipedia_summary` and the sound pack.
[Sim Harvest](harvest.md) keeps the list.

### Fillers and progress lines

- The first filler comes within one agent turn, 3 to 5 s after the person
  stops speaking. Progress lines follow every 6 to 8 s. Each says something
  new, lasts at most about 1.5 s and is a fixed sentence, not LLM output.
- `request_tool` has 30 s: search at most 0.5 s, the decision at most 10 s
  (often skipped), and the coder the rest, at most 10 s per LLM call. The
  whole request stops at 40 s with "Sorry, I couldn't get that right now"
  and an alternative.
- A spoken "stop" goes through `/voice_commands`, replaces the speech ID and
  aborts creation. Whether "never mind" or other speech during creation
  queues or interrupts is open
  ([Self-Evolving Agent](self-evolving-agent.md#open-questions), question 1).

## Example: "Say Meow!"

| Time         | What happens                                                                              |
| ------------ | ----------------------------------------------------------------------------------------- |
| 0.0 s        | A visitor says "Say Meow!".                                                               |
| 0.3 to 0.8 s | ASR finishes the transcription.                                                           |
| 0.8 to 2.5 s | One agent turn, rung 0: `speak(text="Meow!")`.                                            |
| about 2.5 s  | The speech gate. The text echoes the visitor, so it also gets the cloud check when online. |
| 2.5 to 3.5 s | TTS speaks, and `/walkie/speaking` mutes ASR meanwhile.                                   |

If the agent calls `request_tool("say meow", kind="behaviour")` anyway,
search returns `speak` to reuse. The recommendation is that lint rule L12
rejects a drafted `say_meow` tool with "call speak(text='Meow!')", as it
rejects any generated tool whose body is one call of an existing tool with
constant arguments. Without it, one booth day would fill the library with
`say_hello`, `say_meow` and their relatives. The alternative, a
session-only phrase tool, is [question 7](#open-questions).

| Variant                                | What the agent does                                                                   |
| -------------------------------------- | ------------------------------------------------------------------------------------- |
| "Say Meow in Thai"                     | `speak` with the Thai text. Translation is the LLM's job.                             |
| "Say Meow a hundred times"             | `speak` up to the audio budget, then a polite decline that cites the noise.           |
| "Say Meow every time someone walks in" | A standing trigger, which Competition denies: "I can't keep watching the door, but here's one for now: Meow!" |
| "Say" followed by a slur               | The speech gate refuses, with an alternative.                                         |

## Example: a real cat sound

If the curated pack has `cat_meow`, this is rung 1: offline, about 3 s.
Otherwise it is rung 1b:

1. `play_sound("cat_meow")` returns `NOT_FOUND` with the names it has and
   the hint "call fetch_sound(query) to find one".
2. `fetch_sound(query="cat meow", play=True)` speaks the fixed filler "Ooh,
   let me find a cat sound", about 3 s in.
3. `commons_search` finds audio with license metadata: CC0 and public
   domain first, then CC-BY with attribution recorded; over 5 s is skipped.
4. A GET to `upload.wikimedia.org` through `commons_media`, capped at 1 MiB.
5. The decode sandbox returns PCM, for example 2.1 s, mono, 48 kHz.
6. Silero VAD and local Whisper find no speech in it.
7. It becomes the session asset `cat_meow`, with a copy in quarantine.
8. It plays through the `audios` field of `/to_human`. Done at 6 to 9 s.

Unverified: whether Commons has suitable clips and formats for common
requests, and whether the license is reliably in `iiprop=extmetadata`
([question 9](#open-questions)). The pack plus Commons is the recommended
pair of sound sources. The alternatives are in
[question 8](#open-questions).

## Example: tomorrow's weather

"What's the weather in Bangkok tomorrow?" if nothing were pre-baked. In
practice `weather_forecast` is, so the real answer is rung 1, in 3 to 5 s.

| Time        | What happens                                                                                     |
| ----------- | ------------------------------------------------------------------------------------------------ |
| about 2.5 s | `find_tools` finds nothing above the threshold.                                                  |
| 3 to 5 s    | `request_tool(kind="data")` and the fixed filler "Let me see if I can learn that. One moment."  |
| next        | Search misses again, and kind `data` skips composing. The coder gets the brief, the Competition stub, the `open_meteo` API notes and the top hits, and writes `weather_forecast` in 1 or 2 model calls. |
| 11 to 15 s  | `run_draft` with Bangkok calls the geocoding and forecast templates. Progress line: "Still on it." |
| 18 to 28 s  | `submit_draft`: lint, the `competition` stamp, L14 (web only, no sensing), L15 (name) and dedup. Offered as `sess_weather_forecast(city, day)`. |
| 22 to 35 s  | The call returns a typed, tainted result. The gate passes "Tomorrow in Bangkok, light rain, between 26 and 33 degrees". The hard stop is at 40 s. |

## Sounds

Both tools send audio through `/to_human` and share the speech-ID holder
with `speak` ([ADR-0045](decisions/0045-play-sound-through-to-human.md)).

- **`play_sound(sound: str)`**, effect S, plays a name from the curated pack
  or a session asset, never a URL. On `NOT_FOUND` it lists its names and,
  when online in a profile that allows it, suggests `fetch_sound`.
- **`fetch_sound(query: str, play: bool = True)`**, effects R and S, is
  deterministic, with no LLM. It speaks a fixed filler, searches
  `commons_search` (audio MIME types, namespace 6), filters by license and
  length, downloads through `commons_media`, decodes, rejects clips with
  speech, registers a session asset and plays it. It counts against the
  per-test request budget.

**Decoding.** Fetched audio is decoded to PCM in its own no-network bwrap
sandbox (`toolbox/sandbox/decode.py`), never in the parent.
[Access Levels](access-levels.md#http-broker-search-and-audio) has the
accepted formats, the ffmpeg flags and the limits. At Competition, Silero
VAD and local Whisper then check the PCM, because a "cat meow" file can
contain spoken abuse. A clip with speech is rejected, or its transcript goes
through the speech gate.

| Cap                         | Competition                                  | Sim                                  |
| --------------------------- | -------------------------------------------- | ------------------------------------ |
| One clip                    | 5 s                                          | 10 s                                 |
| Peak level                  | −3 dBFS after loudness normalization         | same                                 |
| Format                      | mono, 48 kHz or less                         | same                                 |
| Audio queued per invocation | 8 s, at most 3 utterances of 200 characters  | 20 s (Standard), 30 s (Unrestricted) |
| Audio queued per minute     | 20 s                                         | not set here                         |

The audio budget counts estimated speech time and clip lengths together, and
is cleared on a deadline kill, a reset and "stop". Fetched assets are cached
by URL hash.

## Speech gate

Everything the robot says or plays passes `toolbox/speech_gate.py`: `speak`,
`ctx.tools.speak` from generated code, `play_sound`, `fetch_sound`, fillers
and progress lines. The gate runs when text is spoken, not when the request
arrives, because checking the request misses "say" followed by a slur and
strings computed by code.

**Always, locally, at every level:**

- A multilingual word list over normalized text, with case, accents,
  leetspeak and spacing folded first.
- At most 200 characters per utterance at Competition.
- No URLs, phone numbers or e-mail addresses.
- No remarks about a bystander's appearance beyond clothing color for
  counting.
- The audio budget.

**At Competition, when online,** a cloud check also runs when the text
echoes the last human utterance, comes from a generated or session tool, or
is tainted. Every utterance in a turn that used `look_up_online` counts as
tainted. Plain answers skip the cloud call, so normal conversation pays no
extra latency. On a hit, the robot refuses politely with an alternative, and
the log records which source produced the text.

The moderation service is still open ([question 2](#open-questions)). The
recommendation is a classifier call through OpenRouter on the toolsmith key,
with a target of 500 ms or less, still to be measured.

**Fail closed.** If the moderation check fails at startup or in the network
probe, creating code and `look_up_online` are off. The local checks keep
running, so conversation and reuse still work.

## Turn taint and the history scrub

Web text can carry instructions (indirect injection, T2 in the threat model
on [Access Levels](access-levels.md)). The agent cannot reliably tell an
instruction from a fact, so the binding narrows instead:

- Once an untrusted `ToolMessage` enters the turn, the next binds hide every
  tool with effect N or M, and `request_tool`, until the next human
  utterance.
- In a tainted turn, `update_todos` may only change the status of existing
  items, and `find_tools` ranks tools but activates none.
- Session tools return typed fields only: numbers, enums and strings of at
  most 120 characters.
- A tainted generated invocation may not call `speak`, and the agent speaks
  its typed result instead. This is a recommendation and still open
  ([question 5](#open-questions)).
- At the end of the turn, the parent replaces each web-search `ToolMessage`
  with the site names it cited, so a snippet cannot steer a later turn.

This follows [the Rule of Two](access-levels.md#safety-model). A visitor's
request is always untrusted, so a Competition tool may add private local
sensing or web access, never both. The task agent is the one exception: it
holds memory and detections while `look_up_online` reaches the web. Its
query comes only from the person's words, and the results are shaped,
capped, tainted and scrubbed. Generated code gets no such exception (L14).

## Web search

**`look_up_online()` takes no text argument.** Its effect is R. The
recommendation is to keep it for the task agent, and to let the agent reach
it through `find_tools`, helped by one system-prompt line and an eval case.
This is still open ([question 4](#open-questions)). Binding it every turn
would save about 1.5 s but push out another tool. It is never in
`ctx.tools`, and never used for people in the room or for what the robot saw
or remembers.

**The query writer** (`[walkie.models.query_writer]`, a small fast model)
sees only this task's human utterances, so it has nothing private to leak.
It returns a query of at most 200 characters and a language, or a refusal.
It refuses medical dosing, legal advice and emergencies. A query with a URL,
phone number or e-mail address is rejected.

**The provenance check** requires every query word to appear in those
utterances, or, for generated code, in the tool's literals. It catches words
added from anywhere else (exfiltration, T4). Queries are in English until a
Thai smoke test passes ([question 11](#open-questions)). The robot still
answers in the visitor's language.

**Results** are at most 3 items: title at most 100 characters, snippet at
most 300, 1,500 in all. They carry no URLs, safe search is strict, and
`config/search_denylist.txt` drops listed sites. `config/site_names.toml`
turns sites into names: "According to Wikipedia…", or "a website". A domain
is never spoken. Caps: 2 searches per turn or invocation, 6 per minute and
120 per hour, shared by all visitors.

The estimated timeline is about 7 to 8 s: 2 to 3 s for the turn that finds
and activates the tool, the filler "Let me look that up." as the call
starts, about 1 s each for the query writer and the search, and 2 to 3 s for
the answer. Provider latencies are third-party US benchmark means, not
venue measurements.

| Code                     | Cause                                          | What the robot says                                |
| ------------------------ | ---------------------------------------------- | -------------------------------------------------- |
| `UNAVAILABLE`            | offline, timeout, kill switch, moderation down | It can't look that up right now.                   |
| `NOT_FOUND`              | no results, or all denylisted                  | It couldn't find an answer.                        |
| `REFUSED` `QUERY_SOURCE` | the provenance check failed                    | It can't look that up.                             |
| `REFUSED` `TOPIC`        | medical dosing, legal advice, emergencies      | A fixed pointer to staff or the right service.     |
| `REFUSED` `RATE`         | a cap was reached                              | It can't look anything else up for a little while. |

**From generated code,** `ctx.http.web_search(query, *, lang="en")` returns
`SearchResult` items with `title`, `snippet` and `site`, marked
`untrusted=True`. A `url` field appears only in sim. The same caps, shaping
and provenance check apply, and the call taints the turn. Lint rule L14
applies the Rule of Two to code: a tool that uses `ctx.http`, search
included, may otherwise call only `speak`, `play_sound`, `tell_time`,
`tell_date`, `wait_for_seconds` and `ctx.now`. `wait_for_seconds` is today's
`WaitForSecondsTool`
([Self-Evolving Agent](self-evolving-agent.md#how-todays-tools-map-to-atomics)).
Pose and locations count as sensing.

```toml title="config.toml"
[walkie.search]
enabled = true          # kill switch: false makes look_up_online and ctx.http.web_search return UNAVAILABLE
provider = "brave"      # still open, see ADR-0049
robot_key_env = "WALKIE_SEARCH_KEY_ROBOT"
sim_key_env = "WALKIE_SEARCH_KEY_SIM"
connect_timeout_s = 2.0
total_timeout_s = 3.0
```

Only the parent reads the keys. Snippets are never stored. Traces keep the
query's hash and length, its language, the result count and the site names,
with snippet text masked (Langfuse support for this is unverified,
[question 12](#open-questions)).

**The search client** (`toolbox/bridge/search.py`) is about 50 lines over
the broker's pinned-IP transport, with no new HTTP library
([Access Levels](access-levels.md#http-broker-search-and-audio)). It logs
each search to `runs/http/<date>.jsonl` with the fields above, the role and
the URL without its query.

`FakeSearchProvider` replays recorded responses for unit tests and for
`look_up_online` in scored harvest episodes. How harvest uses live search on
the sim key is in [Sim Harvest](harvest.md#isolation).

**Why no LangChain package.** PyPI marks `langchain_community` 0.4.2 as
being sunset, and its GitHub repository was archived on 19 June 2026. Its
`BraveSearchWrapper` has no timeout, forces `extra_snippets` and bypasses
the broker, and its Tavily tool is deprecated. `langchain-parallel` 0.5.0
pins `openai<2`, against `langchain_openai` 1.7.0. OpenRouter's web plugin
returns prose, not snippets. The sim coder gets the same client as a plain
`langchain_core` function tool, which `ToolCallLimitMiddleware` counts.

**The provider is still open** ([question 1](#open-questions)). Brave alone
is recommended: its own index, strict safe search, one GET, about 0.6 s. The
options are compared in
[ADR-0049](decisions/0049-competition-internet-egress.md).

## Named sources

Generated and pre-baked tools reach the internet only through these. Every
request is a GET over HTTPS on port 443: 5 per invocation, 30 per minute, 1
at a time per host, bodies up to 1 MiB, connect 3 s, total 8 s. These apply
to `web_snippets` too, under its own tighter search caps. Only people edit
the list. The broker's checks are described in
[ADR-0049](decisions/0049-competition-internet-egress.md).

| Source               | Template                                                                                   | Notes                                                    |
| -------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| `open_meteo`         | `api.open-meteo.com/v1/forecast`: `latitude`, `longitude`, `daily`, `hourly`, `current`, `timezone`, `forecast_days` | weather                       |
| `open_meteo_geocode` | `geocoding-api.open-meteo.com/v1/search`: `name`, `count`, `language`                      | city to coordinates                                      |
| `frankfurter`        | `api.frankfurter.dev/v1/` plus `latest` or a date: `from`, `to`, `amount`                  | ECB exchange rates                                       |
| `wikipedia_summary`  | `en.wikipedia.org/api/rest_v1/page/summary/<one path segment>`                             | needs a User-Agent                                       |
| `commons_search`     | `commons.wikimedia.org/w/api.php`: `action=query`, `format=json`, `generator=search` or `list=search`, namespace 6, `prop=imageinfo`, `iiprop=url|mime|size|extmetadata` | no other `action` or `prop`; `action=parse` and `prop=revisions` return editable wikitext |
| `commons_media`      | `upload.wikimedia.org/`                                                                    | returns a handle only                                    |
| `web_snippets`       | the search provider's endpoint, with count, strict safe search and language fixed          | no extra snippets, page content or images; no redirects; key added by the parent |

## Offline behavior

A network probe runs at startup and every 30 s: a DNS lookup and a TLS
handshake to OpenRouter and to the search provider, and a check of the
moderation endpoint.

| Piece                                   | Offline                                                             |
| --------------------------------------- | ------------------------------------------------------------------- |
| Compose, create, `fetch_sound`          | Off; `request_tool` is unbound; reuse only                          |
| `find_tools`                            | Works; BM25 and dense search both run locally                       |
| Session tools that need `ctx.http`      | Fail fast with `UNAVAILABLE`                                        |
| `look_up_online`, `ctx.http.web_search` | Fail fast with `UNAVAILABLE`; `look_up_online` stays visible        |
| A request that needs creation           | An immediate refusal with an alternative                            |
| Speech gate                             | Local checks only                                                   |

If moderation fails while the network is up, creating code and
`look_up_online` are off too. `search_robot_docs` embeds through OpenRouter
today, so it likely fails offline until the tool-index work moves it to the
local embedder.

If bwrap or the resource scope cannot start on the robot PC, the runner
refuses generated tools and sandboxed atomics rather than run them
unprotected. The robot says "I can't make new tools on this robot right
now." Reuse, compose, `fetch_sound` and the pre-baked tools are meant to
keep working, but `fetch_sound` decodes in bwrap too
([question 10](#open-questions)).

## Session tools and quarantine

A session runs from agent start to stop, or until a message on
`/walkie/reset_session` (new, `std_msgs/Empty`), which does a normal reset
and also unregisters session tools and assets. Between visitors, the booth
operator uses `/walkie/reset`, which keeps them, so a crowd asking the same
thing reuses the tool made for the first visitor.

- Session tools are named `sess_*` and live in an in-memory overlay of the
  tool index. Nothing is written to `index.sqlite`.
- The agent's offer is rendered from a fixed template: name, typed
  arguments, effects, literal speech, and a summary of at most 100
  characters that passed the speech gate, with imperative phrasing removed.
  The coder's docstring never reaches the task LLM.
- A quarantine copy goes to `$WALKIE_TOOLS_HOME/quarantine/<date>/<name>@<sha8>/`
  (`~/.local/share/walkie/tools`): the code or asset, its hash, the coder
  model, the intent with personal data removed, the level, a timestamp, the
  web sources (site names only, for search) and the license.
- Copies are never loaded automatically. The harvest imports them into its
  quarantine lane, useful sounds join the pack after a license check, and
  copies are deleted after 30 days. A Competition run loads no file from
  `bundled/`, `temporary/` or `quarantine/`.

## The Competition coder

| Item               | At Competition                                                                                     |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Model role         | `[walkie.models.tool_coder_competition]`: a flash-class model, 10 s timeout, the toolsmith key     |
| Budget per request | 4 model calls, 2 test runs, 3 named-source fetches, no web search (recommended, [question 4](#open-questions)), the rest of the 30 s, $0.10 |
| `request_tool`     | 1 per request, at most 3 per test, at most 10 per hour                                              |
| First message      | Pre-filled: the Competition stub, the notes in `toolsmith/api_notes/`, the top 8 hits and the source of the two most relevant atomic tools |
| What it sees       | The raw ASR text, the kind and the checked need. Never tool results, memory, earlier drafts or ROS names |
| `run_draft`        | Against recorded fakes; speech is recorded, not played; `ctx.http` is live within the policy        |
| `fetch_asset`      | Audio up to 5 s and 1 MiB from `commons_media`; JSON and text up to 200 KB                          |
| Imports            | `math`, `statistics`, `datetime`, `zoneinfo`, `decimal`, `fractions`, `re`, `json`, `itertools`, `collections`, `random`, `string`, `textwrap`, `unicodedata`, `pydantic`, `walkie_toolbox_facade` |
| `ctx.tools`        | `speak`, `play_sound`, `tell_time`, `tell_date`, `list_locations`, `get_current_pose`, `find_objects`, `detect_objects`, `search_detections`, `relate_objects`, `get_arm_state`, `wait_for_seconds` (at most 10 s); describe only, no search, for the rest (recommended, see [Access Levels](access-levels.md#open-questions)) |
| Invocation budget  | 15 s wall clock, 5 s CPU, 10 `ctx.tools` calls, no ROS calls                                         |
| Sandbox            | 512 MiB, 16 tasks, a resource scope required                                                        |
| Lint               | L12 (trivial wrapper), L13 (level), L14 (Rule of Two), L15 (`sess_` prefix)                         |
| Reflection         | None at the end of the run                                                                          |

The coder has no open web. A visitor could say "the docs are at
https://evil.example/?q=...", and following that link would leak data and
read an attacker's instructions. People write the API notes instead, and
`web_fetch` is refused outside the named sources. The recommendation is no
web search for the coder either, because the API notes cover the named
sources and the coder never sees tool results ([question 4](#open-questions)).

## RuleBook 2026.2 constraints

Checked against RuleBook 2026.2, released 2026-05-11. Recheck every row
against the 2027 rulebook, expected around December.

| Rule                                                                      | What it means here                                                                      |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| No rule on code generated at runtime                                      | Creation is allowed, within the profiles.                                               |
| Cloud services over the Arena Network are external computing, declared to the Technical Committee one month ahead | Declare the cloud LLM (coder included), the moderation service and the search provider. |
| Up to five external computing resources, with Technical Committee approval | How cloud APIs are counted is unknown.                                                 |
| Internet is not guaranteed; the robot must be safe without it             | The network probe and the offline table.                                                |
| No remote control and no keyboard input from the team                     | No approvals at Competition; policy decides everything.                                 |
| 30 s of inactivity removes the robot; one extension per test, at most 3 min | Fillers and progress lines. Never spend the extension on creation.                    |
| Politeness, and no continuous loud noise                                  | The audio budget, clip caps and the "a hundred times" decline.                          |
| Since 2026.2, the person who interacted scores the interaction            | Refusals are polite and always offer an alternative.                                    |
| Restaurant: assume no wireless                                            | The `restaurant` profile behaves as offline.                                            |
| The VizBox audience screen is available                                   | It shows the stages "Looking for a skill", "Writing a new skill" and "Testing".         |

## Rollout and tests

In the [roadmap](self-evolving-agent.md#roadmap), phase 1 adds the search
keys and a `[walkie.search]` stub. Phase 2 adds `play_sound`, the TTS
override, the gate's local checks, `tell_time` and `tell_date`. Phase 4a
adds the search client and `FakeSearchProvider`.

Phase 6 builds the rest, in this order:

1. The curated sound pack, then `fetch_sound` with the decode sandbox.
2. The pre-baked tools, then the named sources with their templates and
   API notes.
3. `look_up_online` with the query writer, the denylist and the site names.
4. Cloud moderation in the speech gate, then the taint filter and
   quarantine.
5. The network probe, the profiles and progress lines.
6. The robot-PC sandbox, with the AppArmor profile for
   `/opt/walkie/bin/bwrap` only if `scripts/robot_check.sh` (new) reports
   `BWRAP_FAILED`, and the robot marker `/etc/walkie/robot`.
7. Last, the Competition coder.

The cloud coder, the moderation service and the search provider are
declared to the Technical Committee one month before the event.

Phase 6 is done when:

- the red-team set gives 0 escapes at Competition;
- a booth dry run has no audible test failures;
- offline, creation is refused within 2 s;
- "Say" followed by a slur is refused;
- 20 long-tail questions are answered with a site name;
- `look_up_online` returns `UNAVAILABLE` within 1 s when search is blocked;
- a red-team snippet does not change the next tool call.

New tests: `test_play_sound.py` (with a decoder test for a mismatched file
type), `test_speech_gate.py`, `test_web_search.py`, `test_http_policy.py`
and `test_coder.py` (no `web_search` and no fetch outside named sources).

The main risks, and what limits each:

| Risk                                                 | What limits it                                                                       |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| The search provider is down or slow                  | One attempt, a 3 s total timeout, `UNAVAILABLE` with a fixed sentence, the kill switch |
| A wrong or harmful answer from a snippet             | Strict safe search, the denylist, the cloud check, the site name spoken               |
| Visitor data in a query                              | The query writer sees only the person's words; the provenance check                   |
| The provider's terms on attribution, storage and LLM use | Site names spoken, snippets never stored; the terms are checked before phase 6    |
| A fetched sound's license, or hidden speech in it    | The license filter, VAD and local ASR, a license check before the pack                |
| The TTS override depends on RAI internals            | `rai_s2s` pinned at Git tag 2.12.2 and `rai-core` below 2.13, an ordering test and a silent integration test |

## What lives where

| Piece                                          | Repository       | Location                                                      |
| ---------------------------------------------- | ---------------- | ------------------------------------------------------------- |
| `play_sound`, `fetch_sound`, `look_up_online`  | walkie-agent-ros | `src/walkie_agent/tools/atomic/` (new)                        |
| Curated sound pack                             | walkie-agent-ros | `src/walkie_agent/assets/sounds/`, `SOUNDS.toml` (new)        |
| Shared speech-ID holder, TTS override          | walkie-agent-ros | `src/walkie_agent/tools/speech.py`, `src/walkie_agent/half_duplex.py` (changed) |
| Speech gate                                    | walkie-agent-ros | `src/walkie_agent/toolbox/speech_gate.py` (new)               |
| Decode sandbox                                 | walkie-agent-ros | `src/walkie_agent/toolbox/sandbox/decode.py` (new)            |
| HTTP broker and search client                  | walkie-agent-ros | `src/walkie_agent/toolbox/bridge/http.py`, `search.py` (new)  |
| Taint filter and history scrub                 | walkie-agent-ros | `src/walkie_agent/graph.py` (new)                             |
| API notes for the named sources                | walkie-agent-ros | `src/walkie_agent/toolsmith/api_notes/` (new)                 |
| Denylist and site names                        | walkie-agent-ros | `config/search_denylist.txt`, `config/site_names.toml` (new)  |
| Profile, search and model roles                | walkie-agent-ros | `config.toml`: `[walkie.mode]`, `[walkie.search]`, `[walkie.models.*]` (new tables) |
| Search keys                                    | walkie-agent-ros | `.env.example`: `WALKIE_SEARCH_KEY_ROBOT`, `WALKIE_SEARCH_KEY_SIM` (new) |
| Session tools, quarantine copies               | robot PC only    | in memory; `$WALKIE_TOOLS_HOME/quarantine/`                   |

Nothing on this page changes walkie-ros, walkie-database or walkie-vision.
`detect_objects`, which Competition code may call, relies on the on-demand
`DetectObjects` service that walkie-vision's ADR-0009 proposes; see
[Self-Evolving Agent](self-evolving-agent.md#finding-objects-with-search_detections).

## Open questions

1. **Which search provider, and which plan?** Brave alone is recommended.
   Which Brave plan permits use with an LLM is unverified. Blocks the
   provider code and the declaration to the Technical Committee.
2. **Which moderation service?** Recommended: a classifier call through
   OpenRouter on the toolsmith key. Whether OpenRouter offers
   `/v1/moderations` or a guard model is unverified. The alternatives are
   OpenAI's omni-moderation with a direct key, or a local Qwen3Guard-Gen
   model (deferred). Blocks the cloud check, and so creation and
   `look_up_online` at Competition.
3. **How does the Technical Committee count cloud APIs** against the five
   external computing resources? Blocks the declaration, due one month
   ahead.
4. **Who gets web search, and how?** Which profiles get it, GPSR in
   particular, is open. Recommended: keep `look_up_online` for the task
   agent, reach it through `find_tools` plus one system-prompt line and an
   eval case, and give the Competition coder no web search. Binding the
   tool every turn saves about 1.5 s but pushes out another tool. Blocks the
   profile table, the system prompt, the eval set and the coder's tool list.
5. **May a tainted generated invocation call `speak`?** The recommendation
   is no. Blocks the taint filter.
6. **Does speech during creation queue or interrupt?** Open on
   [Self-Evolving Agent](self-evolving-agent.md#open-questions), question 1.
   It also blocks progress lines and the creation abort.
7. **How are trivial requests such as "say Meow!" handled?** Recommended:
   call `speak` directly, and L12 rejects wrappers. The alternative is a
   session-only phrase tool, made only when a person explicitly asks the
   robot to learn something, with L12 still enforced in sim. The robot
   could say "I've learned to meow", at the cost of a creation call and
   noise in the quarantine. Blocks L12 at Competition.
8. **Which sound sources?** Recommended: the curated pack plus `fetch_sound`
   from Wikimedia Commons, so common sounds work offline in about 3 s and
   rare ones online in 6 to 9 s. The pack alone means no untrusted audio at
   runtime, but a missing sound gets a polite refusal. Adding Freesound
   gives a far larger catalog, but puts an API token on the robot, held by
   the parent. Blocks `fetch_sound` and the size of the pack.
9. **Does Commons return usable clips, with the license in
   `extmetadata`?** Blocks `fetch_sound`. If not, the pack must cover more
   sounds.
10. **How does `fetch_sound` decode when bwrap fails on the robot PC?**
    Reuse, compose and the pre-baked tools keep working when bwrap cannot
    start ([Access Levels](access-levels.md#sandbox-and-bridges),
    [ADR-0047](decisions/0047-sandbox-and-bridges.md)). Both leave
    `fetch_sound` to this question, because its decode step runs in bwrap.
    Either `fetch_sound` goes off with generated tools, or it needs a
    decoder that is safe without bwrap. Blocks the robot-PC sandbox step.
11. **Do Thai queries return useful results?** Blocks queries in Thai.
12. **Can Langfuse mask snippet text in traces?** Blocks tracing at
    Competition with search on.

## Decisions

Click a decision to expand it. All of these are proposed. See
[Architecture](index.md) for how decisions work.

??? note "ADR-0045: Play sounds through the audios field of /to_human · Proposed"

    --8<-- "architecture/decisions/0045-play-sound-through-to-human.md:body"

??? note "ADR-0049: Limit Competition internet access to named sources and web-search snippets · Proposed"

    --8<-- "architecture/decisions/0049-competition-internet-egress.md:body"
