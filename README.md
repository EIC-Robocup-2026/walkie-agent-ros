# Walkie Agent++ (ROS Port)

The LLM agent for Walkie, the EIC RoboCup 2026 team's robot. It takes tasks
by text or speech and carries them out with ROS 2 tools, such as driving to a
named place and speaking. It is built on [RAI](https://github.com/RobotecAI/rai)
and ROS 2 Jazzy, and [Pixi](https://pixi.sh) installs all of it, ROS
included.

**Documentation:** <https://eic-robocup-2026.github.io/walkie-agent-ros/>

- [Getting Started](https://eic-robocup-2026.github.io/walkie-agent-ros/getting-started/):
  install and run the agent, step by step.
- [Setup Log](https://eic-robocup-2026.github.io/walkie-agent-ros/setup/):
  how this repository was built, with explanations.
- [Architecture](https://eic-robocup-2026.github.io/walkie-agent-ros/architecture/):
  how the parts fit together, and the decisions behind them.

## Quick start

```bash
curl -fsSL https://pixi.sh/install.sh | sh  # once per computer, then reopen the terminal
git clone https://github.com/EIC-Robocup-2026/walkie-agent-ros.git
cd walkie-agent-ros
pixi install
pixi run build-ros2  # builds the message packages from ros2_ws.repos once
pixi run build-rmw-zenoh  # builds the Zenoh middleware once
cp .env.example .env # then set OPENAI_API_KEY to your OpenRouter key
pixi run test
```

Every `pixi run` uses the Zenoh middleware, as walkie-ros does, so nodes need
a Zenoh router. Run the agent against a fake robot, in three terminals:

```bash
pixi run ros2 run rmw_zenoh_cpp rmw_zenohd  # skip if a router already runs
```

```bash
pixi run fake-walkie
```

```bash
pixi run agent
```

Then send it a task:

```bash
pixi run ros2 topic pub --once /from_human rai_interfaces/msg/HRIMessage \
  "{text: 'Please go to the kitchen and say hi when you get there.'}"
```

The fake robot's terminal logs what Walkie says.
[Getting Started](https://eic-robocup-2026.github.io/walkie-agent-ros/getting-started/)
explains each step and lists the topics.

## Tasks

| Command                          | What it does                                           |
| -------------------------------- | ------------------------------------------------------ |
| `pixi run agent`                 | Run the agent. It reads tasks from `/from_human`.      |
| `pixi run fake-walkie`           | Run a fake robot for testing without hardware.         |
| `pixi run agent-sim`             | Run the agent against walkie-ros's MuJoCo simulation.  |
| `pixi run test`                  | Run the tests. They need no API key and no hardware.   |
| `pixi run voice-asr`             | Listen on the microphone and send speech to the agent. |
| `pixi run voice-tts`             | Speak the agent's replies through the speaker.         |
| `pixi run list-audio-devices`    | List microphone and speaker names for `config.toml`.   |
| `pixi run whoami-build`          | Build Walkie's identity files. Calls the LLM.          |
| `pixi run langfuse-up` / `-down` | Start or stop local Langfuse tracing. Needs Docker.    |
| `pixi run docs`                  | Serve the documentation at <http://127.0.0.1:8000>.    |

`pixi task list` shows every task.

## Repository layout

| Path                    | What it holds                                                     |
| ----------------------- | ----------------------------------------------------------------- |
| `src/walkie_agent/`     | The agent, its tools, the voice nodes and the fake robot          |
| `tests/`                | Tests that run on real ROS 2 with a fake robot and a scripted LLM |
| `config.toml`           | RAI settings: LLM vendor and models, voice, tracing               |
| `config/locations.toml` | The places Walkie can drive to                                    |
| `whoami/`               | Walkie's identity sources for RAI's whoami                        |
| `ros2_ws/`              | Message packages built from source (`rai_interfaces`, `walkie_database_msgs`) |
| `docker/langfuse/`      | The local Langfuse server                                         |
| `docs/`                 | The documentation site (MkDocs)                                   |
