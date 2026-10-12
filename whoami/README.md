# whoami: Walkie's identity for the agent

Source files for RAI's `rai_whoami`. `pixi run whoami-build` turns them into
`generated/`, which the agent loads at startup:

- `generated/info.json` is added to the system prompt.
- `generated/index.faiss` backs the `search_robot_docs` tool.

| Folder           | Put here                                         | Formats                                   |
| ---------------- | ------------------------------------------------ | ----------------------------------------- |
| `documentation/` | Specs, manuals, rules of behavior                | `.md`, `.txt`, `.pdf`, `.docx`            |
| `images/`        | Photos of the robot                              | `.png`, `.jpg`                            |

There is no `urdfs/` folder: rai_whoami 2.12 loads URDF files but never uses
them. Describe the arms in plain words in `documentation/` instead (see
docs/architecture/decisions/0020-no-urdf-in-agent.md).

Rebuild after changing anything here, and review `generated/info.json` before
committing it: an LLM wrote it, and the agent will believe it.

This README is not read by the build; only the two folders above are.
See docs/setup/06-whoami.md.
