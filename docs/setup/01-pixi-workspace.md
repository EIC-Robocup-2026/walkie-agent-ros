# 1. Create the Pixi Workspace

**Goal:** create an empty project that can install ROS 2 Jazzy without
installing ROS on the whole computer.

## Background

- **ROS 2 Jazzy** is the version of ROS 2 we use.
- **Pixi** is a package manager. It installs tools and libraries into the
  project folder instead of into your system. Everyone on the team gets the
  exact same versions.
- **RoboStack** is a project that packages ROS so Pixi can install it.
  Pixi downloads packages from **channels**. We use two:

| Channel           | What it gives us                                      |
| ----------------- | ----------------------------------------------------- |
| `robostack-jazzy` | ROS 2 Jazzy packages                                  |
| `conda-forge`     | Everything else (Python, compilers, common libraries) |

## The command

```bash
pixi init ~/walkie-agent-ros \
  --channel https://prefix.dev/robostack-jazzy \
  --channel https://prefix.dev/conda-forge
cd ~/walkie-agent-ros
```

- `pixi init <folder>` creates a new project in that folder.
- `--channel` tells Pixi where to look for packages. Order matters: Pixi
  checks `robostack-jazzy` first, then `conda-forge`.

## What it created

| File                           | What it is                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------- |
| `pixi.toml`                    | The project's settings: channels, packages, and tasks. You edit this file.                  |
| `pixi.lock`                    | The exact version of every package installed. Pixi writes this file, never edit it by hand. |
| `.pixi/`                       | The installed packages. Ignored by Git, because it is large and can be recreated.           |
| `.gitignore`, `.gitattributes` | Git settings so `.pixi/` is not committed and `pixi.lock` is not merged by hand.            |

## Useful Pixi commands

| Command              | What it does                                          |
| -------------------- | ----------------------------------------------------- |
| `pixi install`       | Install everything from `pixi.lock`                   |
| `pixi add <package>` | Add a package to `pixi.toml` and install it           |
| `pixi run <task>`    | Run a task defined in `pixi.toml`                     |
| `pixi shell`         | Open a terminal with the project's packages available |
