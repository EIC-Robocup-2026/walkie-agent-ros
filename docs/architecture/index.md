# Architecture

This section explains how the agent is put together and **why**. The
[Setup Log](../setup/index.md) shows how each part was installed. These pages
show how the parts fit, which alternatives we rejected, and what each choice
costs later.

| Page                                             | What it covers                                                                  |
| ------------------------------------------------ | ------------------------------------------------------------------------------- |
| [The Agent](agent.md)                            | The agent node, its topics and tools, how it reaches the robot, and what lives where |
| [Robot Capabilities (Proposed)](capabilities.md) | The typed `ctx.base`, `ctx.speech` and other capabilities atomics use instead of ROS, and the MotionSupervisor beneath them |
| [Arm Manipulation (Proposed)](manipulation.md)   | How the agent will move the arms and grippers, and run skills such as a VLA policy |
| [Perception (Proposed)](perception.md)           | How the agent will use walkie-vision's models and walkie-database's memory to find objects and people |
| [HRI Task (Proposed)](hri.md)                    | The first task the new agent runs end to end: greeting, seating and introducing guests, and the gaze loop |
| [Self-Evolving Agent (Proposed)](self-evolving-agent.md) | The redesigned agent: the todo list, atomic and bundled tools, tool search, and the tool creator |
| [Access Levels and Generated Tools (Proposed)](access-levels.md) | The Unrestricted, Standard and Competition levels, the sim interlock, and the sandbox for generated code |
| [Competition Mode (Proposed)](competition.md)    | What the agent may create at a competition, run profiles, sounds, web search and the speech gate |
| [Sim Harvest (Proposed)](harvest.md)             | Running the agent in MuJoCo on many tasks, checking what it built, and promoting tools by pull request |

## Architecture Decision Records

Each significant decision is written down as an **Architecture Decision
Record (ADR)**: a short file with the context, the decision and its
consequences. The idea comes from
[Michael Nygard's article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

- Every ADR is one file in `docs/architecture/decisions/`, numbered in order,
  for example `0007-speak-tool-and-console.md`.
- The architecture pages embed the ADRs as collapsed blocks at the bottom.
  Click one to expand it.
- A decision is significant if it is hard to reverse, if it changes what
  walkie-ros must provide, or if a new member would ask "why is it like this?"

Each ADR has a status, shown by the color of its block:

| Status                                  | Block              | Meaning                                                          |
| --------------------------------------- | ------------------ | ---------------------------------------------------------------- |
| Proposed                                | `note` (blue)      | Written for review. Nothing depends on it yet.                   |
| Accepted                                | `success` (green)  | Agreed and in effect.                                            |
| Accepted, partly superseded by ADR-NNNN | `success` (green)  | In effect, except for the points the newer ADR replaces.         |
| Superseded by ADR-NNNN                  | `quote` (gray)     | Replaced by a newer decision. Kept so the reasons stay readable. |
| Rejected                                | `failure` (red)    | Considered and turned down.                                      |

After an ADR is accepted, do not change its decision. If the decision
changes, write a new ADR and set the old one to "Superseded by ADR-NNNN"
([ADR-0001](decisions/0001-record-architecture-decisions.md)).
ADR-0001 covers only a full replacement. If the new ADR replaces only part
of the old one, it names that part, and the old one becomes "Accepted,
partly superseded by ADR-NNNN". Change only the old ADR's status line, and
the title of its block on the architecture page. Fixing typos and links is
fine.

## Add a decision

1. Copy the template below into
   `docs/architecture/decisions/NNNN-short-title.md`, with the next free
   number.
2. Fill it in. Keep everything between the two marker lines: that is the
   part shown on the architecture page.
3. Add a collapsed block for it at the bottom of the architecture page it
   belongs to, in number order.
4. Run `pixi run docs-build`. It fails if a link or the ADR file name is
   wrong.

```markdown title="docs/architecture/decisions/NNNN-short-title.md"
# ADR-NNNN: Short title in plain words

<!-- ;--8<-- [start:body] -->
**Status:** Proposed · **Date:** YYYY-MM-DD

**Context**

What problem or situation forces a decision. Facts only, with file paths
where they help.

**Decision**

What we will do, stated plainly.

**Consequences**

- What gets easier, what gets harder, and what we must now keep in mind.

**See also:** [a related page](/setup/index.md)
<!-- ;--8<-- [end:body] -->
```

The block on the architecture page:

```markdown title="docs/architecture/agent.md"
??? note "ADR-NNNN: Short title in plain words · Proposed"

    ;--8<-- "architecture/decisions/NNNN-short-title.md:body"
```

!!! tip "Links inside an ADR start with `/`"
    Write links from the `docs/` folder, for example
    `[Voice](/setup/07-voice.md)`. The same text appears on two pages in
    different folders, and a relative link would break on one of them.
