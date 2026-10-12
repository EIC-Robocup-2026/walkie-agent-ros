# ADR-0005: Reach LLMs through OpenRouter with RAI's openai vendor

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

RAI supports several LLM vendors (OpenAI, Ollama, AWS, Google). The previous
agent, walkie-agent-v2, used OpenRouter, which offers models from many
providers behind one OpenAI-compatible API and one key.

**Decision**

All three RAI model roles (simple, complex and embeddings) use RAI's `openai`
vendor with `base_url = "https://openrouter.ai/api/v1"`. The model IDs are
set in `config.toml`. The OpenRouter key goes in `.env` as `OPENAI_API_KEY`,
because that is the variable the `openai` vendor reads. The `[ollama]`
section stays, for local models.

**Consequences**

- Changing the model is a one-line change in `config.toml`.
- The key's name is misleading. The docs explain it where the key is set up.
- RAI code that calls OpenAI directly ignores `base_url`. For example, the
  `OpenAI` speech recognition model always calls `api.openai.com` and needs a
  real OpenAI key.
- whoami's documentation index needs an embeddings endpoint. Whether
  OpenRouter serves `openai/text-embedding-3-small` for it has not been
  tested yet.
- The agent needs internet access. Without it, point `base_url` at a local
  server (vLLM, llama.cpp) or switch a role to `ollama`.

**See also:** [Install RAI, step 4](/setup/03-ros-and-rai.md#step-4-create-configtoml)
<!-- --8<-- [end:body] -->
