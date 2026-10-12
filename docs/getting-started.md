# Getting Started

This page gets the project running on your machine. Every command is meant to
be copied and pasted. If you want to know _how_ the project was built step by
step, read the [Setup Log](setup/index.md) after this.

![mambo](assets/mambo.gif)

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

## 3. Build the ROS message package

RAI's message types (`rai_interfaces`) are not available as a package, so we
build them from source once:

```bash
pixi run build-ros2
```

This clones `rai_interfaces` into `ros2_ws/src/` and builds it with colcon.
Pixi loads the result automatically every time you use `pixi run` or
`pixi shell`. Run it again only if `ros2_ws.repos` changes.

## 4. Add your API keys

The agent reads secret keys from a file called `.env` in the project root.
Git ignores this file, so your keys are never committed.

```bash
cp .env.example .env
```

Open `.env` and fill in your [OpenRouter key](https://openrouter.ai/keys):

```bash title=".env"
OPENAI_API_KEY=sk-or-v1-...
```

!!! info "Why is an OpenRouter key called `OPENAI_API_KEY`?"
RAI talks to OpenRouter through its `openai` vendor, because OpenRouter
uses the same API as OpenAI. The model and URL are set in `config.toml`.

The Langfuse keys in `.env.example` are optional. `pixi run langfuse-setup`
fills them in for you, see [Local Langfuse](setup/04-langfuse.md).

## 5. Run the tests

```bash
pixi run test
```

The tests use the real ROS 2 middleware with a fake robot and a scripted LLM,
so they need no API key and no hardware. All tests should pass.

## 6. Run the agent with the fake robot

Open three terminals in the project folder.

```bash title="Terminal 1: fake robot (navigation, TF, speech topic)"
pixi run fake-walkie
```

```bash title="Terminal 2: the agent"
pixi run agent
```

```bash title="Terminal 3: listen to what the robot says"
pixi run ros2 topic echo /walkie/speech
```

Then send the agent a task, from any terminal:

```bash
pixi run ros2 topic pub --once /from_human rai_interfaces/msg/HRIMessage \
  "{text: 'Please go to the kitchen and say hi when you get there.'}"
```

The agent's log in terminal 2 shows each tool call. Terminal 3 prints what
the robot says, for example `data: Hi everyone! I have arrived in the kitchen.`

| Topic            | Direction  | What it carries                                       |
| ---------------- | ---------- | ----------------------------------------------------- |
| `/from_human`    | to agent   | Tasks for the agent (`rai_interfaces/msg/HRIMessage`) |
| `/to_human`      | from agent | The agent's text replies, for the operator            |
| `/walkie/speech` | from agent | What the robot says out loud (`std_msgs/msg/String`)  |
| `/walkie/reset`  | to agent   | Makes the agent forget the conversation (`std_msgs/msg/Empty`) |

The places the robot can go are listed in `config/locations.toml`.

### Start a fresh conversation

The agent remembers every task, reply and tool result since it started, and
sends all of it to the LLM with each new task. To make it forget, without
restarting it:

```bash
pixi run ros2 topic pub --once /walkie/reset std_msgs/msg/Empty
```

This also drops any tasks still waiting in line, and stops the current task.
A tool that is already running (for example, driving to the kitchen) finishes
first. In Langfuse, each conversation shows up as its own **session**.

## 7. Open the docs locally

```bash
pixi run docs
```

Open <http://127.0.0.1:8000> in your browser. The page reloads by itself when
you edit a file in `docs/`. Press ++ctrl+c++ to stop.

## Next

- [Local Langfuse](setup/04-langfuse.md): see every step the agent takes in
  a web UI.
- [Setup Log](setup/index.md): how this repository was built, one step at a
  time.
- [Writing Docs](setup/02-documentation.md#writing-a-new-page): how to add
  your own page.
