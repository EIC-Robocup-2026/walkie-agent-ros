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

## 3. Build the ROS packages

RAI's message types (`rai_interfaces`), the messages of Walkie's memory
(`walkie_database_msgs`, from
[walkie-database](https://github.com/EIC-Robocup-2026/walkie-database)) and
the messages of its vision node (`walkie_vision_msgs`, from
[walkie-vision](https://github.com/EIC-Robocup-2026/walkie-vision)) are not
published as packages, so we build them from source once:

```bash
pixi run build-ros2
```

This clones `rai_interfaces`, walkie-database and walkie-vision into
`ros2_ws/src/`, at the commits pinned in `ros2_ws.repos`, and builds the
three message packages with colcon. Pixi loads the result automatically
every time you use `pixi run` or `pixi shell`. Run it again only if
`ros2_ws.repos` changes. No agent tool reads walkie-database or walkie-vision
yet. When one does, the walkie-database pin must match the database node you
run, or the agent's memory requests are never answered
([ADR-0035](architecture/decisions/0035-memory-through-walkie-database.md)).

On `feat/mahoraga-demo` the two pins name commits on the local
`feat/scene-graph-prep` branches of walkie-vision and walkie-database, which
are not on GitHub. `pixi run build-ros2` cannot fetch them, so import them
from the local clones first; see
[Local message pins](setup/03-ros-and-rai.md#local-message-pins).

Every `pixi run` and `pixi shell` here uses the Zenoh middleware
(`rmw_zenoh_cpp`), as walkie-ros and the robot do
([ADR-0033](architecture/decisions/0033-zenoh-in-every-pixi-environment.md)).
RoboStack's version of it can stop receiving messages after a few seconds,
so build version 0.2.11 from source once as well
([ADR-0032](architecture/decisions/0032-rmw-zenoh-from-source.md)):

```bash
pixi run build-rmw-zenoh
```

Pixi uses it in place of RoboStack's version from then on. Until you build
it, nothing warns you: Zenoh nodes quietly use RoboStack's version. If the
build exists but cannot be loaded, every command prints a line that starts
with `rmw_zenoh_loader: cannot load rmw_zenoh_cpp 0.2.11`, with the reason.

Run it again when `rmw_zenoh_ws.repos` or `scripts/rmw_zenoh_loader.c`
changes. First stop everything you started with `pixi run` here, the Zenoh
router included, because the build replaces a library those processes have
loaded.

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
    The agent talks to OpenRouter through the `openai` vendor, because OpenRouter
    uses the same API as OpenAI. The model and URL are set in `config.toml`.

The Langfuse keys in `.env.example` are optional. `pixi run langfuse-setup`
fills them in for you, see [Local Langfuse](setup/04-langfuse.md).

## 5. Run the tests

```bash
pixi run --frozen test
```

`--frozen` uses `pixi.lock` as it is, so the command cannot change your environment.

The tests use a scripted LLM, so they need no API key and no hardware. Most import no
ROS at all. Some run real rclpy and RAI's real connector against a fake robot, on a
random domain. They switch to the Fast DDS middleware themselves, so they need no Zenoh
router either ([ADR-0019](architecture/decisions/0019-test-fakes-fast-dds.md)). All tests
should pass. The runnable examples on the [Learn the Agent](learn/index.md) pages run
here too.

A pass says the code matches its tests. It does not say that the robot works, because
nothing in the suite touches real Nav2, a simulator or hardware.

!!! warning "In a second Git worktree"
    A worktree that borrows another checkout's environment imports that checkout's code.
    Set `PYTHONPATH=$PWD/src` there, or the tests silently run the other code.

## 6. Run the agent with the fake robot

!!! warning "This calls your real model"
    The next steps need the API key from step 4, and they spend tokens. For a run with no
    real model and no sound, use the [silent demo](#run-a-silent-demo).

Open four terminals in the project folder. Zenoh nodes find each other
through a router, so start that first.

```bash title="Terminal 1: Zenoh router"
pixi run ros2 run rmw_zenoh_cpp rmw_zenohd
```

Only one router can listen on `localhost:7447`, where every node looks for
it. If walkie-ros already runs one on this computer, skip terminal 1 and use
that router. A node that finds no router logs
`Unable to connect to a Zenoh router` and sees no node in another terminal.

```bash title="Terminal 2: fake robot (navigation, TF, speech topic)"
pixi run fake-walkie
```

`config/locations.toml` ships empty, because a place written there is where the robot
really drives. So use the invented places in the test fixture: start, kitchen,
living_room and bedroom.

```bash title="Terminal 3: the agent"
pixi run agent --locations tests/fixtures/locations.toml
```

```bash title="Terminal 4: read the agent's text replies"
pixi run ros2 topic echo /walkie/console rai_interfaces/msg/HRIMessage --field text
```

Then send the agent a task, from any terminal:

```bash
pixi run ros2 topic pub --once /from_human rai_interfaces/msg/HRIMessage \
  "{text: 'Please go to the kitchen and say hi when you get there.'}"
```

To log each tool call as one JSON line, add `--tool-log logs/tools.jsonl` in terminal 3.
Terminal 2 logs what the robot says, for example `[Walkie says] Hi everyone! I have
arrived in the kitchen.`, and terminal 4 shows the agent's final text reply.

The `default` tool set has `list_places`, `go_to`, `move_relative`, `say`,
`find_objects`, `object_info`, `find_people`, `face_person` and `get_facts`, which reads the time, the
date and the team from `config.toml`'s `[team]`. The agent also has `update_todos` and
`wait_for_person`. `--tool-set gpsr` and `--tool-set hri` pick the other fixed sets.
`find_objects`, `find_people` and `face_person` answer `UNAVAILABLE` unless you add
`--vision`: they then read the people on `/walkie_vision/people` and the objects on
`/walkie_vision/objects`, from walkie-vision or the simulation's oracle.
`--objects-topic` reads the objects from another topic. `pixi run agent-sim-chula` adds
`--vision`. `object_info` answers `UNAVAILABLE` without it and gives null measures with
it, since nothing measures a live object. In the `gpsr` set, `pick`, `place_on` and `hand_over` answer `UNAVAILABLE` as well.

| Topic             | Direction  | What it carries                                                          |
| ----------------- | ---------- | ------------------------------------------------------------------------ |
| `/from_human`     | to agent   | Tasks or a bare stop, typed or from speech recognition, text only (`rai_interfaces/msg/HRIMessage`) |
| `/to_human`       | from agent | What the robot says out loud, for text-to-speech (`HRIMessage`)          |
| `/walkie/console` | from agent | The agent's text replies and a line per todo status change, for the operator (`HRIMessage`) |
| `/walkie/task`    | to agent   | A task with its steps, or with a template name, as JSON (`std_msgs/msg/String`) |
| `/walkie/todos`   | from agent | The todo list as JSON, kept for late readers (`std_msgs/msg/String`)     |
| `/walkie/reset`   | to agent   | Cancels the task and forgets the conversation (`std_msgs/msg/Empty`)     |
| `/voice_commands` | voice      | Pause/stop speech playback (`std_msgs/msg/String`), see [Voice](setup/07-voice.md) |

To stop, publish a bare `stop` on `/from_human`. Walkie cancels the task and its goal at
once and says "Okay, stopping.", even when idle.

The places come from the file that `--locations` names. A place name must be written
trimmed and in lowercase. Any other form stops the agent at startup, and lookups ignore
case.

### Run a silent demo

`tests/ros_demo.py` runs the assembled agent with a scripted model and a fake robot. It
needs no API key and plays no sound. `unshare -Urn` starts a shell in a new network with
only a loopback interface, so nothing in it can reach a live ROS graph. The demo refuses
to run on any other network. Its Zenoh router listens on its own port, 7499.

```bash title="One terminal, from the project folder"
mkdir -p /tmp/walkie-demo
unshare -Urn bash --noprofile --norc <<'SH'
set -euo pipefail
ip link set lo up
export ROS_DOMAIN_ID=77 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export PYTHON_DOTENV_DISABLED=1 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false
unset ZENOH_SESSION_CONFIG_URI ZENOH_ROUTER_CONFIG_URI
common='scouting/multicast/enabled=false;transport/shared_memory/enabled=false'
router_cfg="listen/endpoints=[\"tcp/127.0.0.1:7499\"];connect/endpoints=[];$common"
node_cfg="connect/endpoints=[\"tcp/127.0.0.1:7499\"];$common"
setsid pixi run --frozen env RMW_IMPLEMENTATION=rmw_zenoh_cpp \
  ZENOH_CONFIG_OVERRIDE="$router_cfg" ros2 run rmw_zenoh_cpp rmw_zenohd \
  > /tmp/walkie-demo/router.log 2>&1 &
router=$!
trap 'kill -INT -- -"$router" 2>/dev/null || true; sleep 2; kill -KILL -- -"$router" 2>/dev/null || true' EXIT
pixi run --frozen env RMW_IMPLEMENTATION=rmw_zenoh_cpp \
  ZENOH_CONFIG_OVERRIDE="$node_cfg" python tests/ros_demo.py --robot fake \
  --tool-set hri --locations tests/fixtures/locations.toml \
  --report /tmp/walkie-demo/report.json
SH
```

It prints `PASS` and the report's path, and exits 0 only if every assertion in the report
passed. Leave Zenoh's gossip scouting on: with `scouting/gossip/enabled=false`, two
processes on this router do not discover each other.

This proves that the parts are wired together on a real rmw_zenoh transport. The robot
in it is still a fake, so it says nothing about Nav2, TF on hardware or stop latency.

### Give the agent its steps

A task on `/walkie/task` arrives with its steps already on the todo list,
so the LLM cannot drop one. Each change of a step's status shows up in
terminal 4, such as `t1: Going to the kitchen`:

```bash
pixi run ros2 topic pub --once /walkie/task std_msgs/msg/String \
  "{data: '{\"task\": \"Go to the kitchen and say hi.\", \"todos\": [\"Go to the kitchen\", \"Say hi\"]}'}"
```

To watch the whole list:

```bash
pixi run ros2 topic echo /walkie/todos std_msgs/msg/String --field data
```

The message format, and the templates in `config/task_templates.toml`, are
on [The Agent](architecture/agent.md#seeding-a-task-on-walkietask).

### Start a fresh conversation

The agent remembers every task, reply and tool result since it started, and
sends all of it to the LLM with each new task. To make it forget, without
restarting it:

```bash
pixi run ros2 topic pub --once /walkie/reset std_msgs/msg/Empty
```

This drops any tasks still waiting in line, and cancels the current task and its goal.
The list and the history are cleared once the active run has ended, so a hung LLM call
can delay that up to the deadline. In Langfuse, each conversation shows up as its own
**session**.

### Walkie's memory is not connected

The old tool that read remembered objects from walkie-database is retired. `find_objects`
now means a live view: with `--vision` it reads walkie-vision's objects, and without it
it answers `UNAVAILABLE`, because a remembered row is not a current view. Object memory
is a later slice
([ADR-0052](architecture/decisions/0052-capability-verbs-and-backends.md)).

### Drive walkie-ros's simulation instead

!!! warning "The blank map is not yet run with this runtime"
    Nobody has run the new runtime against the blank map. The Chula world has run, through
    `scripts/chula-sim.sh` ([Run the Chula Simulation](setup/08-chula-simulation.md)).
    Follow these steps as a recipe to try, and expect to find problems. It also calls your
    real model.

walkie-ros has a MuJoCo simulation of Walkie with Nav2 on a blank map
([ADR-0030](architecture/decisions/0030-mujoco-sim-first.md)). Its Pixi
environment uses Zenoh for every command too, so both projects share one
router. Every terminal must also use the same `ROS_DOMAIN_ID`, or the nodes
silently do not see each other. `agent-sim` uses the terminal's value, or
domain 0 when it is not set
([ADR-0036](architecture/decisions/0036-agent-sim-shell-domain.md)). The
commands below leave it unset everywhere. Do not run them in a terminal that
exports `ROS_DOMAIN_ID=23`: that is the real robot's domain.

In a walkie-ros checkout, start the Zenoh router and the simulation. If the
router from step 6 still runs, keep it and skip terminal 1.

```bash title="walkie-ros, terminal 1: Zenoh router (the only one)"
pixi run ros2 run rmw_zenoh_cpp rmw_zenohd
```

```bash title="walkie-ros, terminal 2: simulation and Nav2"
pixi run ros2 launch robot_bringup nav_mujoco_omnibot.launch.py headless:=true use_rviz:=false
```

Then, in this project, start the agent.

```bash title="Terminal 3: the agent"
pixi run agent-sim
```

If a move answers that no node publishes `/tf`, the agent and the simulation are on
different domains or routers. Check `echo $ROS_DOMAIN_ID` in each terminal.

```bash title="Terminal 4: send a task"
pixi run ros2 action list   # should show /navigate_to_pose
pixi run ros2 topic pub --once /from_human rai_interfaces/msg/HRIMessage \
  "{text: 'Move forward one meter, then go to the corner.'}"
```

`agent-sim` uses the simulation's clock and the test points in
`config/locations.sim.toml` (start, ahead, left, corner). The map frame is
the robot's start pose, so all points are relative to where it started.

Those points belong to the **blank map** only. walkie-ros also has a world of the real
lab room, `chula_pat14_6`, launched with `world:=chula_pat14_6` from its
`feat/chula-pat14-6-simulation` branch. Its places are in
`config/locations.chula_pat14_6.toml`: the two rooms, the shelf, bar, table, TV, sofa
and entrance, and two invented places that generated commands assume. Each place names
its room. They are the nav approach goals of the world's authoring file and the same
places `FakeRobot` uses. Nav2 reached each of them in the simulation
([What has been run](setup/08-chula-simulation.md#what-has-been-run)). Run the agent on
them with:

```bash
pixi run agent-sim-chula
```

The simulation has no people in it, so walkie-vision would see no one. For that world,
an oracle can stand in for walkie-vision: it publishes the people and objects of
`config/scenes/chula_living_room.toml`, a guest raising a hand and the host on the sofa,
for every image of the head camera
([In simulation](architecture/perception.md#in-simulation)). Start it after the
simulation, in the same terminal setup as the agent:

```bash
pixi run oracle-chula
```

`find_people` reads its people and `find_objects` its sofa, table and mug when the
agent runs with `--vision`, which `agent-sim-chula` passes.

`scripts/chula-sim.sh` starts that world, the oracle and the agent together, on a
Zenoh router and domain of their own, and its `check` shows what the oracle draws
([Run the Chula simulation](setup/08-chula-simulation.md)).

`tests/ros_demo.py --robot external` could report on a run against that world
with no real model, but that run has not happened. Even when it does, it can prove
navigation and the runtime only: that launch has no text-to-speech, and vision
and arms are unavailable in the agent. So it cannot prove the full HRI task. The
[see, remember and act roadmap](architecture/see-remember-act.md) lists the steps that
add them.

## 7. Open the docs locally

```bash
pixi run docs
```

Open <http://127.0.0.1:8000> in your browser. The page reloads by itself when
you edit a file in `docs/`. Press ++ctrl+c++ to stop.

## Next

- [Local Langfuse](setup/04-langfuse.md): see every step the agent takes in
  a web UI.
- [Voice](setup/07-voice.md): talk to the agent with a microphone and hear
  it through a speaker.
- [Learn the Agent](learn/index.md): how the runtime works, with examples you
  can run.
- [The Agent](architecture/agent.md): the layers, the exact topics, and what
  has and has not been proven.
- [Setup Log](setup/index.md): how this repository was built, one step at a
  time.
- [Writing Docs](setup/02-documentation.md#writing-a-new-page): how to add
  your own page.
