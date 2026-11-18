# Walkie Agent++ (ROS Port)

The LLM agent for Walkie, the EIC RoboCup 2026 team's robot. It takes tasks
by text or speech and carries them out through a small set of tools, such as
driving to a named place and speaking. It runs on LangChain's `create_agent`,
reaches ROS 2 Jazzy through [RAI](https://github.com/RobotecAI/rai)'s connector,
and [Pixi](https://pixi.sh) installs all of it, ROS included.

**Status: a provisional cutover.** The new runtime has run on unit tests, on real
rclpy against a fake robot and in one silent demo, all with a scripted model. With a real
model it has run two one-step tasks on real Nav2 in the MuJoCo simulation of the Chula
room. It has not run on the robot. Live vision and arms are not connected, so those verbs
answer `UNAVAILABLE`.

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
| `pixi run agent-sim-chula`       | The same, on the places of the `chula_pat14_6` world. |
| `pixi run oracle-chula`          | Publish simulated people and objects for the `chula_pat14_6` living room, in walkie-vision's messages. Simulation only. |
| `scripts/chula-sim.sh up`        | Run the `chula_pat14_6` simulation, the oracle and the agent on a private ROS graph. See [Run the Chula simulation](https://eic-robocup-2026.github.io/walkie-agent-ros/setup/08-chula-simulation/). |
| `pixi run test`                  | Run the tests. They need no API key and no hardware.   |
| `pixi run voice-asr`             | Listen on the microphone and send speech to the agent. |
| `pixi run voice-tts`             | Speak what the agent says through the speaker.         |
| `pixi run list-audio-devices`    | List microphone and speaker names for `config.toml`.   |
| `pixi run langfuse-up` / `-down` | Start or stop local Langfuse tracing. Needs Docker.    |
| `pixi run docs`                  | Serve the documentation at <http://127.0.0.1:8000>.    |

`pixi task list` shows every task.

## Repository layout

The package is laid out by concept, so the folder names say where things are
([ADR-0071](docs/architecture/decisions/0071-package-layout.md)). "No ROS" means the
module never imports ROS, which `tests/test_ros_free.py` checks.
`tests/test_import_contracts.py` checks the package's import rules in `pyproject.toml`.

| Path | What it holds |
| --- | --- |
| `src/walkie_agent/main.py` | The agent process: flags, assembly, signals and exit codes |
| `src/walkie_agent/config.py` | Reads `config.toml`, and the agent's run limit and deadline from it. No ROS. |
| `src/walkie_agent/chat_models.py` | The LLM vendors: builds the model, the optional fallback and the tracing callbacks. No ROS. |
| `src/walkie_agent/prompts.py` | The system prompts: a rule for each bound tool, the places and `robot.md`. No ROS. |
| `src/walkie_agent/errors.py` | `ErrorCode` and `StepError`, the failures every verb and tool reports. No ROS. |
| `src/walkie_agent/locations.py` | The places file and its name rules. No ROS. |
| `src/walkie_agent/voice.py` | The voice entry points, `python -m walkie_agent.voice asr` and `tts`, on RAI |
| `src/walkie_agent/half_duplex.py` | Stops listening while Walkie speaks, for `voice.py` |
| `src/walkie_agent/capabilities/` | What the robot can do: the verbs on `ctx`, their guards and records, the `Unavailable*` stand-ins and `FakeRobot`. No ROS. |
| `src/walkie_agent/tools/` | The LLM tools: the atomics in `atomic/`, whose init is the catalogue, the fixed sets in `sets.py`, the LangChain adapter in `adapter.py`, the gym's stand-in bundles in `bundled_preset/`, and the ignored `bundled/` and `temporary/` folders where the shelf keeps saved functions. No ROS. |
| `src/walkie_agent/runtime/` | Our layer on LangChain's `create_agent`: the middlewares, the todo list, the Runner and the task templates. No ROS. |
| `src/walkie_agent/scripting/` | `run_script`'s runtime: the Monty sandbox, the driver, the shelf of saved functions and its evidence. No ROS. |
| `src/walkie_agent/perception/` | Rules that read walkie-vision's detections: geometry, gestures, topic names and the words for people and objects. No ROS. |
| `src/walkie_agent/people_memory/` | Who is who: the track memory behind `ctx.memory`, the people store, identity matching, recognition and each vector model's match limit. No ROS. |
| `src/walkie_agent/arm/` | The named arm poses, which pose each arm is in, and the answers every arm backend shares. No ROS. |
| `src/walkie_agent/ros/` | One ROS 2 backend per capability, and the operator's topics in `operator_topics.py` |
| `src/walkie_agent/sim/` | The fake robot node, the oracle that stands in for walkie-vision in a simulation, and the check of a simulation's graph |
| `src/walkie_agent/evals/` | Offline evaluations, which the agent never imports. `evals/mahoraga/` is the research gym, in its own Pixi environment and in the fake world only. Its prompt and arms are in `experiment_arms.py`. |
| `tests/` | Tests with a scripted LLM. Some use real rclpy with a fake robot. The gym's tests are in `tests/mahoraga/`, and invented places in `tests/fixtures/`. |
| `docs/` | The documentation site (MkDocs). The decisions are in `docs/architecture/decisions/`. |
| `scripts/` | Sets things up or starts them, such as `chula-sim.sh`, which runs the `chula_pat14_6` simulation, the oracle and the agent on a private ROS graph |
| `config/` | The places (`locations.toml` ships empty until they are measured), the arm poses and robot model, the oracle's scenes, the task templates and a Foxglove layout |
| `config.toml` | LLM vendor and models, the team and its timezone, voice, tracing |
| `pyproject.toml` | The package, and the import rules that import-linter checks |
| `robot.md` | What Walkie is and can do today, for the system prompt |
| `ros2_ws/` | Message packages built from source (`rai_interfaces`, `walkie_database_msgs`, `walkie_vision_msgs`) |
| `docker/langfuse/` | The local Langfuse server |

`scripts/` sets things up or starts them. A scored run, an evaluation, lives in
`src/walkie_agent/evals/`, beside the code it measures, and never in `scripts/`.
