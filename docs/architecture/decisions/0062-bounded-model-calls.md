# ADR-0062: Bound every model call, with provisional numbers

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-11-05

This stays Proposed although the code is in place and its tests pass. Those tests use a
scripted model and a local fake provider. No real provider has been called, so neither
the numbers nor the SDK settings have met a real endpoint.

**Context**

A hung provider call or a looping model must not hold the robot's only task for ever.
RAI's `get_llm_model` reads `config.toml`, but provider options differ. `ChatOllama`
silently ignores `timeout` and `max_retries`, and honours only `client_kwargs`.

**Decision**

`walkie_agent.runtime.model` reads the same `config.toml` as RAI, without
`get_llm_model`. Provider packages load only when used, and keys stay in the environment.

- **Model.** `build_model` picks `complex_model` from the vendor that `[vendor]` names.
  The vendors are `openai` and `ollama`.
- **Run limit.** `ModelCallLimitMiddleware(run_limit, exit_behavior="end")` counts every
  model call in a run. The default is 25.
- **Deadline.** `Deadline` puts `asyncio.timeout` around each model attempt. The default
  is 60 s.
- **SDK settings.** `ChatOpenAI` gets `timeout` set to the deadline and `max_retries=0`,
  since retries would only spend the deadline. `ChatOllama` gets the same timeout through
  `client_kwargs`.
- **Fallback.** `ModelFallbackMiddleware` is added only when an `[agent]` table sets both
  `fallback_vendor` and `fallback_model`. One without the other raises `ValueError`. No
  fallback is selected, and that choice is the team's.
- **Tracing.** Langfuse stays off by default. `Session.config` sends a
  `langfuse_session_id`, so a reset starts a new session.

This replaces the implemented agent's model selection and bounds in proposed
[ADR-0042](/architecture/decisions/0042-per-role-models.md). It does not build ADR-0042's
roles or per-role keys. It amends the loading path of
[ADR-0005](/architecture/decisions/0005-openrouter-via-openai-vendor.md): OpenRouter
through the `openai` vendor stays the default.

**Consequences**

- A scripted model and a local fake provider have hit the limit and the deadline in
  tests. A real provider has not.
- The numbers 25 and 60 are provisional until real runs measure them.
- `deadline_s` sets the SDK timeout too, so one setting moves both.
- This is one model for one agent. It is not a multi-role model system.
- With no fallback, a provider outage ends each task with the failure line.

**See also:** [Install RAI, step 4](/setup/03-ros-and-rai.md#step-4-create-configtoml),
[Mahoraga demo: runtime](/architecture/mahoraga-demo.md#runtime)
<!-- --8<-- [end:body] -->
