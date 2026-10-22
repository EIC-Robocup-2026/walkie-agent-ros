# ADR-0046: Find tools with a hybrid keyword and vector index

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

The [self-evolving agent](/architecture/self-evolving-agent.md) binds at
most 20 tools per turn and reaches the rest through `find_tools`
([ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md)).
Sim harvesting will grow the catalogue to hundreds or a few thousand drafts
([ADR-0048](/architecture/decisions/0048-sim-harvest-promotion.md)), where
ranking quality and duplicate detection both matter.

Keyword search alone (SQLite FTS5 with BM25) does as well as vectors below
about 60 tools, but spoken requests rarely share words with tool names.
"Can you make a cat noise" shares no word with `play_sound`, and a local
FTS5 check returned nothing for "meow" against a four-tool catalogue. On a
public tool-retrieval benchmark, a 0.6B embedding model beat BM25 by about
8 points of nDCG@10 on natural-language queries. BM25 still finds exact
identifiers, such as `navigate_to_location` or handle type names, that
embeddings blur. It costs nothing and works offline.

**Decision**

One index fuses BM25 and dense vectors by rank, behind `ToolIndex.search()`.

| Caller | Interface | Results |
| ------ | --------- | ------- |
| Task agent | `find_tools(query, name_guess="")` | 5 |
| Atomic agent | step 1 of `request_tool` | 8 |
| Tool coder | `search_tools(query, scope="tools" or "code", k=8)` | 8 |
| Generated code, Unrestricted and Standard only (recommended) | read-only `ctx.tools.search(query, k=5)` | 5 cards |

Whether running generated code may search is open. The recommendation is
read-only `ctx.tools.search` at Unrestricted and Standard, where a tool
that searches is stamped at the session level, and `describe` only at
Competition. The alternatives are `describe` only everywhere, or search
everywhere ([Access Levels](/architecture/access-levels.md#open-questions)).

Each tool has several views, each embedded on its own. A tool's dense score
is the best cosine over its views.

| View | Content | Searched by |
| ---- | ------- | ----------- |
| card | Name, namespace, description, signature, effects, level stamp, handle types and examples, about 100 to 250 tokens from one template for every author | everyone |
| questions | 3 to 5 spoken-style requests from `META.questions` | task agent, atomic agent, `ctx.tools.search` |
| code chunks | Function-level chunks cut with `ast` | tool coder only |

The card uses one template because human-written atomics and LLM-written
drafts are two writing styles, and retrievers degrade across styles. Whole
files, run traces and utterances from real runs are not embedded.

**Query path.**

1. An exact match of `name_guess` or the query on a name or alias wins.
2. Pre-filter by level stamp, by the tiers and statuses loaded at this
   level, by allowed effects, and by handle type if the caller names one.
3. BM25: FTS5 top 30, with column weights name 3, description 2,
   questions 1.5, params 1 and types 0.5. Every query token is quoted, so
   user text cannot inject FTS5 syntax. Stopwords are dropped.
4. Dense: the query is embedded with a fixed instruction, as
   `Instruct: Given a spoken request to a service robot, retrieve the tool
   that fulfils it\nQuery:<query>`, top 30. Tool views are embedded
   without an instruction.
5. Reciprocal rank fusion, score = Σ 1/(60 + rank). Ties break by tier,
   then by success count. At most 2 results per namespace in the top 5.

There is no rewrite or rerank call. The task LLM already writes a
docstring-style query, and the coder reads the top 8 cards and chooses. A
reranker is revisited only if the eval shows correct tools at ranks 6 to
15. A query costs one embedding call to OpenRouter, under 15 ms in numpy
and about 1 ms in FTS5. The call's latency from the robot PC is not
measured yet. From the development machine, OpenRouter's edge answered a
plain request in 60 to 110 ms, before any provider was involved. BM25 runs
while the embedding is in flight, and answers alone when:

- the network probe finds no internet, so no call is made;
- the embedding fails, or takes longer than 1 s;
- the startup check below fails.

The trace records whether the dense branch answered.

**Embedder.** `qwen/qwen3-embedding-8b` through OpenRouter, in sim and on
the robot, Competition included. It is configured in
`[walkie.toolbox.embedder]`
([ADR-0042](/architecture/decisions/0042-per-role-models.md)) and called
with `OPENAI_API_KEY`, a 1 s timeout and no retry.

- **One provider, pinned.** OpenRouter serves this model from Nebius,
  DeepInfra and SiliconFlow, the last in fp8. Vectors from two providers
  differ slightly, enough to move the duplicate cut-offs below. The index
  pins one provider that is not fp8, with `allow_fallbacks = false` and
  `data_collection = "deny"`. Phase 3 picks it by latency measured from
  the robot PC.
- **1024 dimensions.** The model returns 4096. It is trained for
  Matryoshka truncation, so the client keeps the first 1024 and
  re-normalizes, rather than rely on each provider honouring the
  `dimensions` parameter. The store stays at the size measured below.
- **A startup check.** 20 fixed strings are embedded at startup and
  compared with their stored vectors. Below a cosine of 0.99, the model
  behind the name has changed, and the dense branch stays off until a
  re-embed.
- `model_id` is the model, the provider and the dimensions.

Internet is available this season, and OpenRouter is already the host the
LLM calls reach
([ADR-0049](/architecture/decisions/0049-competition-internet-egress.md)).
With a cloud embedder, no embedding model is installed, warmed or kept
loaded on any machine, and the robot's CPU and GPU stay free for vision
and the local LLM. The same model runs in sim and on the robot, so cached
vectors copy across. At $0.01 per million tokens (OpenRouter's listed
price on 2026-10-22), embedding a catalogue of a few thousand drafts costs
about a cent.

Rejected:

| Option | Why not |
| ------ | ------- |
| `qwen3-embedding:0.6b` on Ollama, CPU only | The earlier recommendation, and the only small model measured directly on tool retrieval, where it beat BM25 on every slice. It works offline. It needs the model installed, warmed and kept loaded (`keep_alive` -1) on every machine, and CPU time on the robot. OpenRouter does not serve it |
| The same model through ONNX Runtime, in process | No daemon, but it needs an export, its own `model_id`, and a reference check on every machine |
| `openai/text-embedding-3-small`, RAI's embeddings model today | Twice the price, and it takes no query instruction |

RAI's `get_embeddings_model()` is not used: `[ollama] embeddings_model` is
`llama3.2`, a chat model; the RAI role has no timeout, provider pin or
truncation; and its OpenRouter path may need
`check_embedding_ctx_length=False` (not verified). walkie-vision's
`EmbedText` is not used either: it is an image-text model, and it would tie
tool search to the vision service.

**Store.** `$WALKIE_TOOLS_HOME` is the folder for generated tools outside
the repository, by default `~/.local/share/walkie/tools`. `embed_cache` is
a table of vectors keyed by text hash and `model_id`.

| | In sim | At Competition |
| - | ------ | -------------- |
| Index | `$WALKIE_TOOLS_HOME/index.sqlite`, WAL mode, never in git, always rebuildable from the tool files | Built in memory at startup from the git tiers |
| Writer | The interactive agent's registry only. Harvest workers keep their own index under `$WALKIE_TOOLS_HOME/harvest/<run-id>/w<i>/`, and the merge step alone writes a run's combined index | Nothing is written. Session tools live in an in-memory overlay |
| Vectors | Only views whose text hash is new are embedded, from `embed_cache` when possible | From `embed_cache` by text hash only, with no other column read from disk, or embedded fresh. The cache is copied from the sim machine, so only new or changed tools are embedded |
| Paths | From the tool files | The loader checks that each path's `realpath` is under the git tier folders |

If the configured `model_id` differs from the index, the dense branch is
disabled and a background re-embed is swapped in atomically. Search is
numpy brute force. On the development machine, 15,000 × 1024 vectors took
10.7 ms per query, and all pairs over 2,000 drafts took 111 ms. FAISS or
sqlite-vec can replace it above about 200k vectors.

**Duplicates.** Linker check L10 blocks a tool with the same atomic
dependencies and parameter names as an existing one. Semantic duplicates
use card cosine, gated at mean + 1σ (clipped to 0.35 to 0.75) until about
100 labelled pairs set two cut-offs: τ_block, above which a submit is
rejected and pointed to the existing tool, and τ_review, above which a
draft is kept with a `similar_to` edge. First guesses are 0.92 and 0.80.

**Amends [ADR-0005](/architecture/decisions/0005-openrouter-via-openai-vendor.md).**
The tool index embeds through OpenRouter with its own client and model,
not through RAI's embeddings role. That role keeps
`openai/text-embedding-3-small` for `search_robot_docs`, whose index is
separate, so the two models never share a vector space.

**Consequences**

- Spoken requests can find tools whose names share no words with them.
  Exact names still win through BM25 and the exact-match step.
- Offline, tool search keeps working on BM25 alone, which misses requests
  that share no word with a tool, such as "can you make a cat noise".
  `META.questions` puts spoken phrasings into BM25 to narrow that gap.
  `search_robot_docs` needs the network.
- Every `find_tools` call can wait up to 1 s on the network. One of five
  plain requests to OpenRouter's edge took 2.9 s from the development
  machine, so the cap will trip at times. The eval reports how often BM25
  answered alone.
- An outage at the pinned provider turns the dense branch off. A second
  provider can be allowed once its vectors pass the startup check against
  the first's.
- Query text leaves the robot: the task LLM's tool description, or a
  `request_tool` need. The same text already reaches OpenRouter in the LLM
  calls, but the embedding provider is one more party, declared with the
  others ([ADR-0049](/architecture/decisions/0049-competition-internet-egress.md)).
- A new `pixi run eval-toolbox` task reports Recall@5 and MRR for BM25,
  dense and hybrid, per level, on about 100 spoken requests, sim logs and
  held-out questions. Hybrid must beat BM25 on Recall@5 before it is used
  at Competition, or the result goes to the team as a decision.
- This is phase 3 work, tested in a new `tests/test_index.py`.

**See also:** [Self-Evolving Agent](/architecture/self-evolving-agent.md#tool-search)
<!-- --8<-- [end:body] -->
