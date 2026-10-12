# ADR-0001: Record architecture decisions

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

The Setup Log explains how each part was installed. It does not say which
alternatives we rejected, or what a choice will cost later. That knowledge
lives in chat logs and in people's heads, and the team changes every season.

**Decision**

We record each significant decision as an Architecture Decision Record (ADR)
in `docs/architecture/decisions/`, one numbered file per decision. Each ADR
has a status, a date, the context, the decision and its consequences. The
architecture pages embed the ADRs as collapsed blocks, so a reader gets the
overview first and the reasons on demand.

An accepted ADR is not rewritten. A changed decision gets a new ADR, and the
old one becomes "Superseded by ADR-NNNN".

**Consequences**

- A pull request that makes a significant decision adds an ADR in the same
  pull request.
- A design is reviewed as a set of Proposed ADRs before it is built.
- Each ADR file has its own Git history and its own page, so it can be linked
  from issues and pull requests.

**See also:** [How to add a decision](/architecture/index.md#add-a-decision)
<!-- --8<-- [end:body] -->
