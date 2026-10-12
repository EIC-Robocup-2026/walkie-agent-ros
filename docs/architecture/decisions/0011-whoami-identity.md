# ADR-0011: Give the agent Walkie's identity with rai_whoami

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

The LLM must know what Walkie is (its base, arms, sensors and limits) to
answer questions about itself and to plan sensibly. RAI's `rai_whoami` turns
a folder of documents and photos into a text description and a searchable
index of the documents.

**Decision**

- The sources live in `whoami/documentation/` and `whoami/images/`.
  `pixi run whoami-build` writes `whoami/generated/`.
- The text of the generated description is appended to the system prompt.
  The photos are not attached: the build has already described them in the
  text, and attached images would be sent and billed with every LLM call.
- The documentation index becomes the `search_robot_docs` tool, which returns
  the 4 closest passages.
- Both are optional. Without `whoami/generated/`, the agent runs on its base
  prompt and has no search tool.

**Consequences**

- Each build calls the LLM and the embeddings model, so it costs money.
  Rebuild after changing the sources.
- The build sends all documentation to the LLM in one message, and the index
  has no text splitter: one passage per PDF page, and one per Markdown or text
  file. A large document, such as a 200-page rulebook, must be split before it
  goes in, or indexed with our own splitter.
- Each search embeds the query through the embeddings API at run time, so the
  tool needs network access.
- `rai_whoami` ignores URDF files
  ([ADR-0020](/architecture/decisions/0020-no-urdf-in-agent.md)).

**See also:** [Give Walkie an identity](/setup/06-whoami.md)
<!-- --8<-- [end:body] -->
