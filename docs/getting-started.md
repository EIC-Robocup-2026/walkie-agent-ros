# Getting Started

This page gets the project running on your machine. If you want to know _how_
the project was built step by step, read the [Setup Log](setup/index.md) after
this.

## 1. Install Pixi

Pixi is the tool that installs ROS 2 and everything else for this project.
Install it once per computer:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Close and reopen your terminal, then check it works:

```bash
pixi --version
```

!!! note "Requirements"
You need Linux on an x86_64 (normal Intel/AMD) computer. You do **not**
need to install ROS 2 yourself. Pixi does that for you.

## 2. Get the code

```bash
git clone https://github.com/EIC-Robocup-2026/walkie-agent-ros.git
cd walkie-agent-ros
pixi install
```

`pixi install` downloads everything listed in `pixi.toml` into a hidden
`.pixi/` folder inside the project. Nothing is installed system-wide.

## 3. Open the docs locally

```bash
pixi run docs
```

Open <http://127.0.0.1:8000> in your browser. The page reloads by itself when
you edit a file in `docs/`. Press ++ctrl+c++ to stop.

## Next

- [Setup Log](setup/index.md): how this repository was built, one step at a
  time.
- [Writing Docs](setup/02-documentation.md#writing-a-new-page): how to add
  your own page.
