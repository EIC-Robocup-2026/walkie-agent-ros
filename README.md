# Walkie Agent++ (ROS Port)

The LLM agent for Walkie, the EIC RoboCup 2026 team's robot. It takes tasks
by text or speech and carries them out through a small set of tools, such as
driving to a named place and speaking. It runs on LangChain's `create_agent`,
reaches ROS 2 Jazzy through [RAI](https://github.com/RobotecAI/rai)'s connector,
and [Pixi](https://pixi.sh) installs all of it, ROS included.

**Status: a provisional cutover.** The new runtime has run on unit tests, on real
rclpy against a fake robot and in one silent demo, all with a scripted model. It has not
run on real Nav2, in the simulator, on the robot or with a real model. Live vision and
arms are not connected, so those verbs answer `UNAVAILABLE`.

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

The production place list ships empty, so this uses the invented test places. This step
calls your real model, and costs tokens:

```bash
pixi run agent --locations tests/fixtures/locations.toml
```

Then send it a task:

```bash
pixi run ros2 topic pub --once /from_human rai_interfaces/msg/HRIMessage \
  "{text: 'Please go to the kitchen and say hi when you get there.'}"
```

The fake robot's terminal logs what Walkie says.
[Getting Started](https://eic-robocup-2026.github.io/walkie-agent-ros/getting-started/)
explains each step, lists the topics and shows a silent recipe with no real model.

## Tasks

| Command                          | What it does                                           |
| -------------------------------- | ------------------------------------------------------ |
| `pixi run agent`                 | Run the agent. It reads tasks from `/from_human` and `/walkie/task`. |
| `pixi run fake-walkie`           | Run a fake robot for testing without hardware.         |
| `pixi run agent-sim`             | Run the agent against walkie-ros's MuJoCo simulation. Not yet run with this runtime. |
| `pixi run agent-sim-chula`       | The same, on the places of the `chula_pat14_6` world. Not yet run. |
| `pixi run test`                  | Run the tests. They need no API key and no hardware.   |
| `pixi run voice-asr`             | Listen on the microphone and send speech to the agent. |
| `pixi run voice-tts`             | Speak what the agent says through the speaker.         |
| `pixi run list-audio-devices`    | List microphone and speaker names for `config.toml`.   |
| `pixi run langfuse-up` / `-down` | Start or stop local Langfuse tracing. Needs Docker.    |
| `pixi run docs`                  | Serve the documentation at <http://127.0.0.1:8000>.    |

`pixi task list` shows every task.

## Repository layout

| Path                         | What it holds                                                     |
| ---------------------------- | ----------------------------------------------------------------- |
| `src/walkie_agent/main.py`   | The entry point, which assembles the agent                        |
| `src/walkie_agent/capabilities/` | The verbs on `ctx`, their guards and `FakeRobot`. No ROS.     |
| `src/walkie_agent/atomics/`  | The tools, as plain functions, and the fixed tool sets. No ROS.   |
| `src/walkie_agent/runtime/`  | `create_agent`, the middlewares, the todo list and the Runner. No ROS. |
| `src/walkie_agent/ros/`      | The ROS-backed verbs and the operator's topics                    |
| `src/walkie_agent/voice.py`  | The voice entry points, on RAI's `AgentRunner`                    |
| `src/walkie_agent/sim/`      | The fake robot node                                               |
| `tests/`                     | Tests with a scripted LLM. Some use real rclpy with a fake robot. |
| `mahoraga/`                  | The research gym, in its own Pixi environment. Fake world only.   |
| `config.toml`                | LLM vendor and models, the team and its timezone, voice, tracing  |
| `config/locations.toml`      | The robot's real places. Ships empty until they are measured.     |
| `config/locations.sim.toml`  | Test points for the blank simulation map                          |
| `config/locations.chula_pat14_6.toml` | Places and rooms of the `chula_pat14_6` simulation world |
| `tests/fixtures/locations.toml` | Invented places for tests and the fake robot                   |
| `config/task_templates.toml` | Fixed tasks by name. Ships with none.                             |
| `robot.md`                   | What Walkie is and can do today, for the system prompt            |
| `ros2_ws/`                   | Message packages built from source (`rai_interfaces`, `walkie_database_msgs`, `walkie_vision_msgs`) |
| `docker/langfuse/`           | The local Langfuse server                                         |
| `docs/`                      | The documentation site (MkDocs)                                   |
