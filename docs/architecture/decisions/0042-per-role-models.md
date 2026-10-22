# ADR-0042: Pick a model and an API key per role

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

Today the agent builds one model, `get_llm_model("complex_model",
streaming=True)` in `src/walkie_agent/main.py`, through RAI's `[vendor]` and
`[openai]` tables in `config.toml` and one OpenRouter key in
`OPENAI_API_KEY` ([ADR-0005](/architecture/decisions/0005-openrouter-via-openai-vendor.md)).
The [self-evolving agent](/architecture/self-evolving-agent.md) adds more
LLM roles: an atomic agent that composes bundles, a tool coder, a judge for
the harvest, a query writer for web lookups and an embedder. One model and one key do not fit them:

- **Timeouts.** The OpenAI SDK defaults to a 600 s timeout and 2 retries,
  and `langchain_openai` 1.7 leaves `max_retries` unset, so those defaults
  apply. A hung cloud call would freeze the robot for minutes.
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

**Consequences**

- A cloud call that stops answering hits the 20 s timeout and goes to the
  fallback, instead of waiting out the SDK's 600 s default. httpx applies
  the timeout to each connect, read and write, not to the whole call, so a
  slow streamed reply can run longer. With `max_retries=0`, a brief server
  error also goes straight to the fallback instead of a retry.
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
