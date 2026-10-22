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
4. Dense: the query is embedded with a fixed instruction ("Given a spoken
   request to a service robot, retrieve the tool that fulfils it"), top 30.
5. Reciprocal rank fusion, score = Σ 1/(60 + rank). Ties break by tier,
   then by success count. At most 2 results per namespace in the top 5.

There is no rewrite or rerank call. The task LLM already writes a
docstring-style query, and the coder reads the top 8 cards and chooses. A
reranker is revisited only if the eval shows correct tools at ranks 6 to
15. A query costs an estimated 20 to 100 ms to embed on CPU, under 15 ms in
numpy and about 1 ms in FTS5. BM25 alone is used when the embedder is
unreachable, when the model digest does not match the index, or when an
embedding takes longer than 300 ms.

**Embedder.** This is an open choice, configured in `[walkie.toolbox.embedder]`
([ADR-0042](/architecture/decisions/0042-per-role-models.md)). It runs on
the CPU, to keep the robot's GPU (an RTX 5090, expected) for vision and the
local LLM.

| Option | Cost |
| ------ | ---- |
| **`qwen3-embedding:0.6b` on Ollama, CPU only, 1024 dimensions, `num_ctx` 2048, `keep_alive` -1, warmed at startup (recommended)** | The only small model measured directly on tool retrieval, where it beat BM25 on every slice. The same model runs in sim and on the robot, so vectors can be cached by content hash and copied. Needs Ollama on the robot, which the local LLM needs anyway, in a build that supports the RTX 5090. Apache-2.0, about 640 MB. Without `keep_alive` -1, Ollama unloads after 5 minutes and the next query falls back to BM25 silently |
| The same model through ONNX Runtime, in process | No daemon, and `onnxruntime` 1.24 and `tokenizers` 0.23 are already in the lock. Needs an export, its own `model_id`, and a cosine of at least 0.99 against reference vectors on 20 fixed strings |
| Cloud `openai/text-embedding-3-small` | Already configured, but the dense branch is lost offline. Allowed only as the interim in sim, under its own `model_id`, never mixed with another model |

RAI's `get_embeddings_model()` is not used: `[ollama] embeddings_model` is
`llama3.2`, a chat model, and its OpenRouter path may need
`check_embedding_ctx_length=False` (not verified). walkie-vision's
`EmbedText` is not used either: it is an image-text model, and it would tie
tool search to the vision service.

**Store.** `$WALKIE_TOOLS_HOME` is the folder for generated tools outside
the repository, by default `~/.local/share/walkie/tools`. `embed_cache` is
a table of vectors keyed by text hash and `model_id`, which is runtime, tag
and weight digest.

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
On the robot, run-time embeddings leave RAI's embeddings role and
OpenRouter. Only the interim sim index may still use OpenRouter.
**Amends [ADR-0011](/architecture/decisions/0011-whoami-identity.md).**
`search_robot_docs` and its index move to the local embedder together,
because vectors from two models live in different spaces. rai_whoami's
`QueryDatabaseTool` and `FAISSBuilder` both accept an embeddings object.
The `build-whoami` command does not, so `whoami-build` needs a small new
script around rai_whoami's pipeline.

**Consequences**

- Spoken requests can find tools whose names share no words with them.
  Exact names still win through BM25 and the exact-match step.
- Offline, tool search and the robot docs search keep working, with the
  dense branch if the embedder is local.
- A new `pixi run eval-toolbox` task reports Recall@5 and MRR for BM25,
  dense and hybrid, per level, on about 100 spoken requests, sim logs and
  held-out questions. Hybrid must beat BM25 on Recall@5 before it is used
  at Competition, or the result goes to the team as a decision.
- This is phase 3 work, tested in a new `tests/test_index.py`.

**See also:** [Self-Evolving Agent](/architecture/self-evolving-agent.md#tool-search)
<!-- --8<-- [end:body] -->
