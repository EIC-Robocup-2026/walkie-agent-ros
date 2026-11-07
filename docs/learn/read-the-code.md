# Read the Code Next

You've seen how each part behaves. This page says which files to read, in what order, and
gives you exercises that make you change something and watch it break.

## The files

The first group imports no ROS, and most of the tests for it need no ROS either.

| File | What's in it | Tests |
| ---- | ------------- | ----- |
| `src/walkie_agent/errors.py` | `StepError` and its codes | `tests/test_errors.py` |
| `src/walkie_agent/capabilities/` | The verb protocols on `ctx`, the guards, the cancel token, `SystemFacts`, `FakeRobot` | `tests/test_capabilities.py`, `tests/test_facts.py`, `tests/test_fake_robot.py` |
| `src/walkie_agent/atomics/` | The tools, as plain functions, and the four tool sets | `tests/test_atomics.py`, `tests/test_arm_atomics.py`, `tests/test_facts.py`, `tests/test_tool_sets.py` |
| `src/walkie_agent/arm/` | The named arm poses and the robot model they are checked against, what the arms are known to be in, and the wording and the drive check every arm backend shares | `tests/test_arm_poses.py`, `tests/test_arm_answers.py`, `tests/test_fake_arms.py` |
| `src/walkie_agent/adapter.py` | Turns an atomic into a LangChain tool or a plain callable | `tests/test_adapter.py` |
| `src/walkie_agent/runtime/todos.py` | The list, the rules, the ledger and `WalkieTodos` | `tests/test_runtime_todos.py` |
| `src/walkie_agent/runtime/middleware.py` | The smaller middlewares | `tests/test_runtime.py` |
| `src/walkie_agent/runtime/agent.py` | `build_agent`, and the middleware order | `tests/test_runtime.py` |
| `src/walkie_agent/runtime/runner.py` | The queue, stop, reset and shutdown | `tests/test_runtime_runner.py` |
| `src/walkie_agent/runtime/model.py` | Loads the model and the optional fallback from `config.toml` | `tests/test_runtime_model.py` |
| `src/walkie_agent/tasks.py` | Templates and `/walkie/task` messages | `tests/test_tasks.py` |
| `src/walkie_agent/locations.py` | The places file, with its name rules | `tests/test_locations.py` |
| `src/walkie_agent/perception/gestures.py` | Gesture and posture names from COCO keypoints | `tests/test_perception_gestures.py` |
| `src/walkie_agent/sim/scene.py`, `scene_file.py`, `projection.py` | The simulation's people and objects, their TOML file, and how the head camera sees them | `tests/test_projection.py`, `tests/test_scene_file.py` |

The second group is where ROS comes in. Most of these tests use fake ROS objects.
`tests/test_ros_integration.py` runs the assembled agent on real rclpy and RAI's real
connector, against a fake robot on a private domain. None of them is the robot's own graph.

| File | What's in it | Tests |
| ---- | ------------- | ----- |
| `src/walkie_agent/ros/base.py` | `RosBase`: goals, TF, cancels and their limits | `tests/test_ros_base.py` |
| `src/walkie_agent/ros/commander.py`, `arms.py` | `CommanderClient`, for `bimanual_commander`'s actions and services, and `RosArms`, the `ctx.arms` built on it | `tests/test_commander.py`, `tests/test_ros_arms.py` |
| `src/walkie_agent/ros/speech.py` | `RosSpeech`, which publishes on `/to_human` | `tests/test_ros_speech.py` |
| `src/walkie_agent/ros/vision.py` | `UnavailableVision`, which refuses, and `RosVision`, which reads `/walkie_vision/people` with `--vision` | `tests/test_ros_vision.py`, `tests/test_ros_base.py` |
| `src/walkie_agent/perception/people.py` | The pose and gesture phrases `find_people` gives, from the gesture rules' names | `tests/test_ros_vision.py` |
| `src/walkie_agent/perception/geometry.py` | The map-frame yaw of a pixel's ray, the torso and ankle pixels a person's bearing comes from, and the person's distance where the ankles' rays meet the floor | `tests/test_perception_geometry.py`, `tests/test_ros_vision.py` |
| `src/walkie_agent/ros/io.py` | `RosBridge`, for the operator's topics | `tests/test_ros_bridge.py` |
| `src/walkie_agent/ros/policy.py` | `FORBIDDEN_NAMES`, and the check on every endpoint | `tests/test_ros_base.py`, `tests/test_commander.py` |
| `src/walkie_agent/main.py` | Assembly, flags, signals, exit codes | `tests/test_main.py` |
| `src/walkie_agent/voice.py`, `half_duplex.py` | The voice processes, on RAI | `tests/test_voice.py`, `tests/test_half_duplex.py` |
| `src/walkie_agent/sim/fake_walkie.py` | A fake robot as a ROS node | `tests/test_ros_integration.py` |
| `src/walkie_agent/sim/oracle_perception.py`, `oracle_scene.py` | The oracle that publishes walkie-vision's arrays from a scene, and its runner | `tests/test_oracle_perception.py`, `tests/test_scene_file.py` |
| `src/walkie_agent/sim/sim_check.py`, `scripts/chula-sim.sh` | The Chula simulation on a private graph, and the check of what its graph provides | `tests/test_sim_check.py`, `tests/test_chula_sim_script.py` |

## A reading order

1. **`capabilities/__init__.py`.** The verbs are the contract, and it's short. Note what's
   missing: there's no velocity verb.
2. **`atomics/navigation.py`.** See how thin a tool is. The docstring is what the LLM reads.
3. **`runtime/agent.py`.** Read the docstring of `build_middleware`. It explains the order.
4. **`runtime/middleware.py`.** Each class guards one failure, and the docstring names it.
5. **`runtime/todos.py`.** Start at `_Draft`, then `WalkieTodos`. Keep
   [The Rules](todo-rules.md) open.
6. **`runtime/runner.py`.** Read `stop` and `_next`, then the comment on the lock.
7. **`ros/base.py`.** Read it last. The complexity is in the waiting and the cancel paths.
   Its limits are in [The Agent](../architecture/agent.md#limits-of-the-navigation-verbs).

## Checking a rai-core upgrade

Walkie no longer copies RAI's private agent loop, so an upgrade has less to break. It still
touches RAI in a few places. The pin is exact, and this block fails when the installed
version moves:

```pycon
>>> from importlib.metadata import version
>>> version("rai-core")
'2.12.1'

```

This reads the installed package's metadata, so it doesn't import RAI. `rai_s2s` comes from
RAI's Git tag `2.12.2`, because a git install of 2.12.1 fails. `pixi.toml` explains why.

To upgrade, change the pin, run `pixi install`, then walk this table and run `pixi run test`.

| Walkie code | What it uses from RAI | What to check |
| ----------- | ---------------------- | -------------- |
| `main.py` | `ROS2Connector(node_name, use_sim_time)`, `.node` and `.shutdown()` | The connector still owns and spins the node |
| `ros/base.py` | `connector.get_transform` | The signature, and what it raises on a stale or missing transform |
| `ros/io.py`, `ros/speech.py` | `rai_interfaces/msg/HRIMessage` | The fields `text`, `images`, `audios`, `communication_id` and `seq_end` |
| `voice.py` | `AgentRunner`, `BaseAgent`, `ROS2Context` | These are the only uses of RAI's agent host |
| `half_duplex.py` | Private methods of `TextToSpeechAgent` and `SpeechRecognitionAgent` | `_setup_ros2_connector`, `_on_command_message`, `_speaker_callback`, `_on_new_sample` |

`half_duplex.py` is the riskiest row, since those are private. `tests/test_half_duplex.py`
tests it without audio hardware, so a pass doesn't prove the speaker behaves.

## Not built yet

These parts are proposed but not built.

- **Live object vision.** `find_objects` answers `UNAVAILABLE`, and so does
  `find_people` unless the agent runs with `--vision`, when `RosVision` reads
  walkie-vision's people. The old tool that read remembered rows is retired. Reading
  objects live is a later slice.
- **Arms.** `UnavailableArms` refuses every arm verb. Only `FakeRobot` can pick.
- **Real places.** `config/locations.toml` ships empty until the team measures them.
- **A fallback model.** The code can load one. Nobody has selected one, and `run_limit` and
  `deadline_s` are provisional ([ADR-0062](../architecture/decisions/0062-bounded-model-calls.md)).
- **Tool filtering, levels and taint.** The agent binds one fixed set at startup
  ([ADR-0067](../architecture/decisions/0067-static-tool-sets.md)). `MAX_BOUND_TOOLS = 20`
  only logs a warning. See [Self-Evolving Agent](../architecture/self-evolving-agent.md#the-host-graph-and-interrupts).
- **A typed task action.** `/walkie/task` takes JSON in a string. A typed `RunTask` action
  waits until something needs it.
- **The HRI template.** It isn't in `config/task_templates.toml`, because it needs proposed
  atomics such as `find_seats` ([HRI Task](../architecture/hri.md#the-template)).

Nor has the new runtime run on the robot, or with a real model beyond two one-step tasks in
the Chula simulation. [What has been tested](../architecture/agent.md#what-has-been-tested)
has the evidence.

## Exercises

Run each test with `pixi run pytest -k <name>` from the repository root. Undo each change
with `git restore`.

1. **The ledger:** `-k completion_waits_until_the_failed_tool_succeeds`. Predict what the
   list answers if the retry succeeds with a different tool. Check your answer with `attempt`
   from [The Rules](todo-rules.md).
2. **The gate:** `-k gate_sends_the_llm_back_at_most_twice_per_task`. Set `MAX_GATES = 1` in
   `runtime/todos.py` and rerun it. The first failing assert shows one gate message instead
   of two.
3. **The order:** `-k the_whole_middleware_list_is_pinned`. Then read
   `test_control_with_the_limit_first_gate_calls_overrun_it`, which runs the wrong order on
   purpose. What goes wrong, and why does the right order prevent it?
4. **A question:** `-k wait_for_person_ends_the_run_and_the_answer_resumes_the_item`. Predict
   t1's marker in the next reminder, `[?]` or `[>]`, before you read the asserts. Which line
   of `WalkieTodos` decides it?
5. **A stop:** `-k stop_during_an_llm_call_runs_none_of_the_tools_it_returns`. Why does the
   model's reply still enter the history? Then run `-k other_utterances_are_not_stops`, and
   find out why "Walkie, stop" doesn't stop a task.
6. **A template:** `-k a_task_message_with_a_template`. Read `parse_task_message` and predict
   what a message with both `todos` and `template` does. Check with
   `-k a_malformed_task_message_raises`.
7. **The serial order:** `-k calls_of_one_message_run_one_at_a_time_in_order`. Then read the
   control that follows it, `test_control_without_max_concurrency_calls_overlap`. It's why
   you rerun this file on any dependency update.

Previous: [Watching Progress](watching-progress.md)
