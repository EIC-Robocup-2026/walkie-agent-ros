# ADR-0042: Pick a model and an API key per role

<!-- --8<-- [start:body] -->
**Status:** Proposed, with the agent role's call bounds built (amended
2026-10-27, see below) · **Date:** 2026-10-22

**Context**

Today the agent builds one model, `get_llm_model("complex_model",
streaming=True)` in `src/walkie_agent/main.py`, through RAI's `[vendor]` and
`[openai]` tables in `config.toml` and one OpenRouter key in
`OPENAI_API_KEY` ([ADR-0005](/architecture/decisions/0005-openrouter-via-openai-vendor.md)).
The [self-evolving agent](/architecture/self-evolving-agent.md) adds more
LLM roles: an atomic agent that composes bundles, a tool coder, a judge for
the harvest, a query writer for web lookups and an embedder. One model and one key do not fit them:

- **Timeouts.** `langchain_openai` 1.7 passes `timeout=None` to the OpenAI
  SDK. That overrides the SDK's 600 s default, so a call has no timeout at
  all. `langchain_openai` leaves `max_retries` unset, so the SDK's 2
  retries apply, each after the server's `Retry-After` of up to 120 s. A
  hung cloud call would freeze the robot until the connection drops.
  (Corrected on 2026-10-27: this bullet said the 600 s default applied.)
- **Budgets.** With one key, a runaway tool coder or harvest can use up the
  spend limit or the rate limit and cut off the task agent mid-task.
- **Config.** RAI's `load_config` ignores unknown top-level tables, but
  `VendorConfig` and `OpenAIConfig` are dataclasses, so an unknown key
  inside `[vendor]` or `[openai]` raises. New settings cannot go there.
- **Offline.** The `[ollama]` complex model, `llama3.1:70b`, needs about
  40 GB at q4. It does not fit on the robot's 32 GB RTX 5090 (expected,
  [ADR-0034](/architecture/decisions/0034-vision-models-in-walkie-vision.md))
  next to walkie-vision.

**Decision**

Each role gets a top-level `[walkie.models.<role>]` table in `config.toml`
with a primary model, ordered fallbacks, `api_key_env` and `timeout_s`.
Every model is built with RAI's `get_llm_model_direct()` and the role's own
key. That function takes `base_url` from the vendor's table and passes the
rest to the chat model. OpenAI-compatible models get `timeout` and
`max_retries=0`. Ollama models get the timeout as
`client_kwargs={"timeout": ...}`, because `ChatOllama` has neither field
and silently drops both (checked with langchain-ollama 1.1.0).

| Role | Primary model | Fallback | Key | Timeout | Notes |
| ---- | ------------- | -------- | --- | ------- | ----- |
| `agent` | `google/gemini-3-flash-preview` | a local 7 to 14B tool-calling model on Ollama | `OPENAI_API_KEY` | 20 s | Critical path, streaming |
| `atomic_agent` | `google/gemini-3-flash-preview` | none: creation is optional offline | `WALKIE_TOOLSMITH_API_KEY` | 10 s | Structured output |
| `tool_coder` | strongest coding model on OpenRouter (not chosen) | never a local model | `WALKIE_TOOLSMITH_API_KEY` | 60 s | Unrestricted and Standard, in sim |
| `tool_coder_competition` | a flash-class coding model (not chosen) | none | `WALKIE_TOOLSMITH_API_KEY` | 10 s | The whole job must fit in about 25 s |
| `judge` | a different model family from the agent (not chosen) | none | `WALKIE_HARVEST_API_KEY` | 60 s | Harvest v1 only; v0 has no judge |
| `query_writer` | a small, fast model (not chosen) | not set | not set | not set | Writes the query for `look_up_online`, only if that tool is kept (open, recommended; see [Competition Mode](/architecture/competition.md#web-search)) |

- **Keys.** `WALKIE_TOOLSMITH_API_KEY` and `WALKIE_HARVEST_API_KEY` join
  `.env.example`. Each is an OpenRouter key with its own spend limit, so a
  tool creation or a harvest that hits its limit never cuts off the task
  agent. `OPENAI_API_KEY` stays as the task agent's key. RAI's own roles,
  such as the whoami build, still read it. The web search keys,
  `WALKIE_BRAVE_KEY_ROBOT` and `WALKIE_TAVILY_KEY_ROBOT` with their `_SIM`
  twins, follow the same idea and are read only by the parent process, so
  a sim run cannot spend the robot's search quota
  ([ADR-0049](/architecture/decisions/0049-competition-internet-egress.md)).
- **Fallbacks.** Each turn, the agent role binds the visible tools on every
  model and then wraps them with `.with_fallbacks()`. Structured roles call
  `.with_structured_output()` on each model: `function_calling` for
  OpenRouter, which also works with the `ScriptedChatModel` in the tests,
  and `json_schema` for Ollama, because `ChatOllama.bind_tools` ignores
  `tool_choice`.
- **Callbacks.** Nested LLM calls, such as the atomic agent and the coder,
  get tracing callbacks only. Otherwise RAI's `HRICallbackHandler`,
  inherited through the runnable config, streams their output to
  `/walkie/console`.
- **Coder models** are pinned with `use_responses_api=False`. Binding a
  provider-side tool such as `openrouter:web_search` would switch
  `langchain_openai` to the Responses API and escape the coder's
  `ToolCallLimitMiddleware`. The coder's `web_search` is an ordinary
  function tool that the parent runs on the sim search keys.
- **Embedder.** It has its own table, `[walkie.toolbox.embedder]`, because
  RAI's `[ollama] embeddings_model` is a chat model and RAI's embeddings
  role has no timeout or provider pin. It calls `qwen/qwen3-embedding-8b`
  through OpenRouter on `OPENAI_API_KEY`, with a 1 s timeout and no retry,
  everywhere, Competition included
  ([ADR-0046](/architecture/decisions/0046-hybrid-tool-index.md)).
- **No moderation role.** The speech gate has no cloud check
  ([ADR-0049](/architecture/decisions/0049-competition-internet-egress.md)).
  If one is added later, a guard model on OpenRouter would run on
  `WALKIE_TOOLSMITH_API_KEY`; OpenRouter has no `/v1/moderations`
  endpoint.
- **Level.** Config has no access level key. The level comes only from the
  `--access-level` flag ([ADR-0039](/architecture/decisions/0039-access-levels.md)).
  `[walkie.mode] profile` selects the run profile.
- **Offline model.** A 7 to 14B tool-calling model replaces `llama3.1:70b`
  (a 14B model at q4 is about 9 GB). A phase 1 eval with the exact model
  picks it. The cap of 20 bound tools in
  [ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md)
  was set with small models in mind. Installing Ollama moves to phase 1,
  with a check for an sm_120 (RTX 5090) capable build.
- A circuit breaker (start on the local model for 120 s after 2 cloud
  failures) waits until cloud failures actually show up.

**Amends [ADR-0005](/architecture/decisions/0005-openrouter-via-openai-vendor.md).**
The agent no longer reads `complex_model`, and two more OpenRouter keys
join `OPENAI_API_KEY`. The `[vendor]` and `[openai]` tables stay, with the
OpenRouter `base_url`, for RAI's own roles: the whoami build's LLM call,
and the embeddings behind `search_robot_docs`.

**Amendment, 2026-10-27: the agent role's call bounds are built.** This is
the first slice of this ADR. `config.toml` gains `[walkie.models.agent]`
with two keys, read by `src/walkie_agent/tuning.py` before `main.py`
builds the LLM. The agent still uses RAI's `complex_model`. The primary
model, the fallbacks and `api_key_env` come with the rest of this ADR, and
until then the loader refuses them as unknown keys. The rest of this ADR
stays Proposed.

- **`timeout_s` is the whole-call deadline**, 20 s. httpx applies a
  timeout to each connect, read and write, not to the whole call, and
  OpenRouter sends SSE comments (`: OPENROUTER PROCESSING`) while a model
  has not answered, so a per-read timeout alone lets a slow call run on.
  `DeadlineTransport` in `src/walkie_agent/llm_deadline.py`, a subclass of
  `httpx2.HTTPTransport`, stamps the start of each request and checks the
  deadline on every chunk of the reply, comments included. Once it has
  passed, the chunk raises `httpx2.ReadTimeout("call deadline")`, which the
  OpenAI SDK raises as `APITimeoutError`. The check runs in the thread that
  reads the stream. A watchdog thread that closes the client was rejected:
  on Linux, `close()` on a socket does not wake a `recv()` blocked in
  another thread, and the SSL socket is not safe to tear down from another
  thread.
- **`read_timeout_s` is new**, 10 s: the longest silence, as the httpx read
  timeout, for a server that stops sending anything. It may not be above
  `timeout_s`. Connecting has 5 s, the SDK's own default, fixed in the
  code.
- **The measured gap.** Langfuse tracing was off in the sim runs, so no
  trace kept chunk times. A scratch script outside the repository streamed
  8 calls with the agent's settings: `deepseek/deepseek-v4.1-flash:nitro`
  through OpenRouter, streaming, with the messages and the 20 bound tools
  of the HRI task's first 8 turns, requests of 19 to 21 KB. The longest
  silence was 1.52 s, before the reply's headers. The longest gap between
  two chunks of a reply was 0.29 s. No SSE comment was sent, and the
  longest call took 3.41 s. That is well under 10 s, so `read_timeout_s`
  ships at 10 s. Measure again before lowering it: late-task histories and
  other models were not measured.
- **`max_retries=0`.** A 429 or a 5xx raises at once, whatever its
  `Retry-After`.
- **The openai vendor only.** `build_llm` in `main.py` passes the bounds,
  with `http_client=openai.DefaultHttpxClient(transport=DeadlineTransport(timeout_s))`,
  only when RAI's `complex_model` vendor is `openai`. An Ollama model's
  `client_kwargs` timeout comes with the rest of this ADR.

**Consequences**

- A cloud call no longer runs without a bound. A call that keeps sending
  bytes, SSE comments included, raises at the first chunk after 20 s. A
  silent one raises 10 s after its last byte. The worst case, bytes until
  just under 20 s and then silence, is about 30 s. Until the fallbacks are
  built, the error ends the task: `_run_agent` logs it and waits for the
  next message. With the fallbacks, the call goes to the next model
  instead. With `max_retries=0`, a brief server error or a 429 also raises
  at once instead of a retry.
- The agent's HTTP client has its own transport. httpx then reads no proxy
  settings from the environment for it, and `langchain_openai`'s default
  TCP keepalive options do not apply; the read timeout bounds a silent
  connection instead.
- A runaway tool coder or harvest cannot starve the task agent of budget.
- The team manages three OpenRouter keys and two search keys instead of one
  key.
- Offline, the agent plans with a smaller model, and tool creation is not
  available.
- Model ids marked "not chosen" are placeholders until an eval or a team
  decision sets them.
- Ollama is not installed on the development machine, and an sm_120 build
  on the robot is not confirmed.
- This is phase 1 work.

**See also:** [Self-Evolving Agent](/architecture/self-evolving-agent.md#roles-and-models),
`config.toml`, `.env.example`
<!-- --8<-- [end:body] -->
