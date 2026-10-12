# 6. Give Walkie an Identity

**Goal:** tell the agent what robot it is (its name, team, hardware and what
it may and may not do) and let it look up details in the robot's documents.
This follows RAI's
[whoami tutorial](https://robotecai.github.io/rai/tutorials/create_robots_whoami/).

## Background

- RAI's `rai_whoami` package reads a folder of documents, photos and URDF
  files about a robot and asks an LLM to write a short description of it.
- The agent adds that description to its system prompt at startup.
- The documents are also split into chunks and stored in a small vector
  database (FAISS). A `search_robot_docs` tool lets the agent search them
  when it needs a detail that is not in the description.
- If `whoami/generated/` does not exist, the agent starts with its normal
  prompt and without the search tool. Nothing breaks.

## Step 1: Install rai_whoami

`rai_whoami` is not on PyPI for our RAI version, so it comes from the RAI
repository:

```toml title="pixi.toml"
[pypi-dependencies]
rai-whoami = { git = "https://github.com/RobotecAI/rai.git", tag = "2.12.2", subdirectory = "src/rai_whoami" }
```

??? info "Why tag 2.12.2 and not 2.12.1?"
    Our `rai-core` is 2.12.1, but installing from the 2.12.1 tag fails with
    `No url found for submodule path 'src/src/rai_interfaces'`. That tag has
    a broken submodule entry, and the installer always checks submodules.
    Tag 2.12.2 removes the broken entry. Comparing the two tags on GitHub
    shows no changes inside `src/rai_whoami` or `src/rai_s2s`, so the code
    is the same.

## Step 2: Write the source files

Everything lives in the `whoami/` folder at the project root:

| Folder           | Put here                          | Formats                        |
| ---------------- | --------------------------------- | ------------------------------ |
| `documentation/` | Specs, manuals, rules of behavior | `.md`, `.txt`, `.pdf`, `.docx` |
| `images/`        | Photos of the robot               | `.png`, `.jpg`                 |
| `urdfs/`         | The robot's URDF or xacro         | `.urdf`, `.xacro`              |

`whoami/documentation/walkie.md` describes Walkie: the team, the hardware,
**what the agent can do today**, and how it should behave. Keep the "can do
today" list honest. If it mentions the arm, the agent will offer to pick
things up even though it has no tool for that.

!!! warning "The folder is `urdfs/`, not `urdf/`"
    RAI's tutorial says `urdf/`, but the build only reads `urdfs/`. Files in
    a folder named `urdf/` are silently ignored.

## Step 3: Build

```bash
pixi run whoami-build
```

This runs RAI's `build-whoami whoami --build-vector-db`. It:

1. asks the `complex_model` from `config.toml` to summarize the documents,
   photos and URDF, and writes `whoami/generated/info.json`;
2. embeds the documents with the `embeddings_model` and writes
   `whoami/generated/index.faiss`.

Each run sends every file to the LLM, so it costs a little money. Run it
again only after changing a file in `whoami/`.

## Step 4: Review the result

Open `whoami/generated/info.json` and read it before you commit it. An LLM
wrote it, and the agent will believe everything in it. If something is
wrong, fix the source file and build again rather than editing the JSON.

Then start the agent as usual. It loads `whoami/` by default. To use a
different folder:

```bash
pixi run agent --whoami-dir path/to/folder
```

## How it fits in the code

- `src/walkie_agent/whoami.py` builds the system prompt and the
  `search_robot_docs` tool.
- The description goes inside `<embodiment>` tags at the end of the system
  prompt.
- Photos are **not** sent with the prompt. The build already describes them
  in text, and attaching them would add image tokens to every LLM call.
