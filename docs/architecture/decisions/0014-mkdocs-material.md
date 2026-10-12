# ADR-0014: Document the project with MkDocs Material

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-07

**Context**

The team changes every season, and new members need to learn both ROS 2 and
this project. A single README does not scale to setup guides, explanations
and design records.

**Decision**

The documentation is a MkDocs Material site built from `docs/`. MkDocs runs
in its own Pixi environment (`docs`), which has no ROS packages. GitHub
Actions builds it with `--strict` and publishes it to GitHub Pages. The site
has three parts: Getting Started, the Setup Log (every setup step, what and
why) and Architecture (overviews and ADRs).

**Consequences**

- Docs live next to the code and change in the same pull request.
- `--strict` turns warnings into errors, so a broken link, a missing
  embedded ADR or a mistyped ADR section fails the build.
- MkDocs is capped below 2.0, because 2.0 does not work with the Material
  theme.

**See also:** [Add the documentation site](/setup/02-documentation.md)
<!-- --8<-- [end:body] -->
