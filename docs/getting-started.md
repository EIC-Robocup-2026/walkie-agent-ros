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
[walkie-database](https://github.com/EIC-Robocup-2026/walkie-database)), the
messages of its vision nodes (`walkie_vision_msgs`, from
[walkie-vision](https://github.com/EIC-Robocup-2026/walkie-vision)) and the
interfaces of its arm commander (`my_robot_interfaces`, from
[walkie-ros](https://github.com/EIC-Robocup-2026/walkie-ros)) are not
published as packages, so we build them from source once:

```bash
pixi run build-ros2
```

This clones `rai_interfaces`, walkie-database, walkie-vision and walkie-ros
into `ros2_ws/src/`, at the commits pinned in `ros2_ws.repos`, and builds the
four message packages with colcon. The walkie-ros clone is the largest: it
downloads about 115 MB and takes about 250 MB on disk. Pixi loads the result
automatically every time you use `pixi run` or `pixi shell`. Run it again
only if `ros2_ws.repos` changes. The
walkie-database pin must match the database node you run, or the agent's
memory requests are never answered
([ADR-0035](architecture/decisions/0035-memory-through-walkie-database.md)).

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
RAI talks to OpenRouter through its `openai` vendor, because OpenRouter
uses the same API as OpenAI. The model and URL are set in `config.toml`.

The Langfuse keys in `.env.example` are optional. `pixi run langfuse-setup`
fills them in for you, see [Local Langfuse](setup/04-langfuse.md).

## 5. Run the tests

```bash
pixi run test
```

The tests run real ROS 2 nodes with a fake robot and a scripted LLM, so they
need no API key and no hardware. They switch to the Fast DDS middleware
themselves, so they need no Zenoh router either
([ADR-0019](architecture/decisions/0019-test-fakes-fast-dds.md)). All tests
should pass.

## 6. Run the agent with the fake robot

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

```bash title="Terminal 3: the agent"
pixi run agent
```

```bash title="Terminal 4: read the agent's text replies"
pixi run ros2 topic echo /walkie/console rai_interfaces/msg/HRIMessage --field text
```

Then send the agent a task, from any terminal:

```bash
pixi run ros2 topic pub --once /from_human rai_interfaces/msg/HRIMessage \
  "{text: 'Please go to the kitchen and say hi when you get there.'}"
```

The agent's log in terminal 3 shows each tool call. Terminal 2 logs what the
robot says, for example `[Walkie says] Hi everyone! I have arrived in the
kitchen.`, and terminal 4 shows the agent's final text reply.

| Topic             | Direction  | What it carries                                                          |
| ----------------- | ---------- | ------------------------------------------------------------------------ |
| `/from_human`     | to agent   | Tasks, typed or from speech recognition (`rai_interfaces/msg/HRIMessage`) |
| `/to_human`       | from agent | What the robot says out loud, for text-to-speech (`HRIMessage`)          |
| `/walkie/console` | from agent | The agent's text replies and a line per todo status change, for the operator (`HRIMessage`) |
| `/walkie/task`    | to agent   | A task with its steps, or with a template, as JSON (`std_msgs/msg/String`) |
| `/walkie/todos`   | from agent | The todo list as JSON, kept for late readers (`std_msgs/msg/String`)     |
| `/walkie/reset`   | to agent   | Makes the agent forget the conversation (`std_msgs/msg/Empty`)           |
| `/voice_commands` | voice      | Pause/stop speech playback (`std_msgs/msg/String`), see [Voice](setup/07-voice.md) |

The places the robot can go are listed in `config/locations.toml`.

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

This also drops any tasks still waiting in line, and stops the current task.
A tool that is already running (for example, driving to the kitchen) finishes
first. In Langfuse, each conversation shows up as its own **session**.

### Ask Walkie's memory

The fake robot has no memory. The `find_objects` tool asks the
walkie-database node where objects were last seen, through the
`/walkie_database/find_objects` service. Without that node, the tool tells
the agent that memory is unavailable, and the other tools still work.

To try it, set up walkie-database with its
[Getting Started](https://eic-robocup-2026.github.io/walkie-database/getting-started/),
and check out the commit pinned in this project's `ros2_ws.repos`. A new
`data/walkie.db` has no objects yet, so write the sample house to
`data/demo.db` and run the node on it:

```bash title="walkie-database, terminal 5: the database node"
pixi run db-seed
pixi run db-node --ros-args -p db_path:=data/demo.db -p map_id:=demo-house
```

The node searches only the map named by `map_id`. The sample house's map is
`demo-house`. On the robot, `map_id` is the id of the map the robot uses. If
`map_id` is empty, every search fails and the agent is told why. The node
must use the same router and the same `ROS_DOMAIN_ID` as the agent.

Then ask the agent:

```bash
pixi run ros2 topic pub --once /from_human rai_interfaces/msg/HRIMessage \
  "{text: 'Where is my cup?'}"
```

The agent's log shows a `find_objects` call for `cup`, which finds the two
cups in the sample house's kitchen.

### Drive walkie-ros's simulation instead

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

If `move_relative` answers that no node publishes `/tf`, the agent and the
simulation are on different domains or routers. Check `echo $ROS_DOMAIN_ID`
in each terminal.

```bash title="Terminal 4: send a task"
pixi run ros2 action list   # should show /navigate_to_pose
pixi run ros2 topic pub --once /from_human rai_interfaces/msg/HRIMessage \
  "{text: 'Move forward one meter, then go to the corner.'}"
```

`agent-sim` uses the simulation's clock and the test points in
`config/locations.sim.toml` (start, ahead, left, corner). The map frame is
the robot's start pose, so all points are relative to where it started.

`agent-sim` also passes `--mock-people-memory`, so `recognize_person` and
`remember_person` load. They keep people in memory, and forget them on a
reset or when a new task replaces the todo list. `pixi run agent` leaves
the flag out, so those two tools do not load there. walkie-ros's
simulation itself publishes no `/walkie_vision/people` or
`/walkie_vision/objects`, so on the blank map `find_people` and
`find_seats` answer that vision is unavailable
([HRI Task](architecture/hri.md#atomics)). `pixi run hri-scenario`
publishes both, in the `chula_pat14_6` room
([Run the HRI scenario](#run-the-hri-scenario)).

### Arms in the simulation

`agent-sim` also passes `--arms`, so the four arm tools load and the robot
checks its arms before it drives
([Arm Manipulation](architecture/manipulation.md#tools)). They need MoveIt
and walkie-ros's arm commander, `bimanual_commander`. Start both in two more
walkie-ros terminals, on the simulation's domain: leave `ROS_DOMAIN_ID`
unset, as above.

```bash title="walkie-ros, terminal 5: MoveIt"
pixi run ros2 launch openarm_bimanual_moveit_config moveit_only.launch.py hardware_type:=mujoco use_rviz:=false
```

```bash title="walkie-ros, terminal 6: the commander"
pixi run ros2 launch /path/to/walkie-agent-ros/launch/commander_sim.launch.py
```

Replace `/path/to/walkie-agent-ros` with the path of this project. The
launch file runs walkie-ros's `commander.launch.py` on the simulation's
clock, which that file does not set. It runs in walkie-ros's environment,
where the commander is built.

- **Without the commander,** the arm tools answer `UNAVAILABLE`, and every
  drive ends with "Arm state unknown: the commander was not found."
- **Untried poses.** On sim time the agent also offers the poses not yet
  tried on the robot: `tray`, `receive` and `carry`
  ([Pose configuration](architecture/manipulation.md#pose-configuration)).
- **Lift height.** The simulation's lift starts at 0.507 m, and the agent
  never moves it, so the hands sit lower than on the robot. The poses are
  joint poses, so they work at any lift height.

### Run the HRI scenario in the chula_pat14_6 room { #run-the-hri-scenario }

The HRI task runs in walkie-ros's MuJoCo world `chula_pat14_6`, a
reconstruction of a real living room and kitchen with a captured map, AMCL
and Nav2. A scenario plays the guests, answers the robot and publishes the
vision arrays for them ([HRI Task](architecture/hri.md#in-simulation)). The
world is on walkie-ros's branch `feat/chula-pat14-6-simulation` until its
PR #58 merges. Check it out in your walkie-ros checkout, and build it:

```bash title="walkie-ros: the world's branch"
git fetch origin
git checkout feat/chula-pat14-6-simulation
pixi run build
```

Use one `ROS_DOMAIN_ID` in every terminal below, and never 23. Leaving it
unset everywhere, as above, works. Start the router, then the room.

```bash title="walkie-ros, terminal 1: Zenoh router (the only one)"
pixi run ros2 run rmw_zenoh_cpp rmw_zenohd
```

```bash title="walkie-ros, terminal 2: the room, AMCL and Nav2"
pixi run ros2 launch robot_bringup nav_mujoco_omnibot.launch.py world:=chula_pat14_6 headless:=true use_rviz:=false
```

Wait until its log shows `Managed nodes are active`, about 10 s after the
start. Until then AMCL publishes no `map` frame, so the robot's pose is
unknown. Then start MoveIt and the commander, as in terminals 5 and 6 of
[Arms in the simulation](#arms-in-the-simulation). The scenario hangs the
guest's bag through the commander. In this project, start the scenario and
the agent:

```bash title="Terminal 3: the scenario"
pixi run hri-scenario
```

```bash title="Terminal 4: the agent"
pixi run agent-sim-hri
```

- **The scenario** publishes `/walkie_vision/people` and
  `/walkie_vision/objects` for each image of the head camera, and answers
  the robot's questions on `/from_human`. Until AMCL runs, it logs once
  that it waits for AMCL, and publishes no arrays. It writes its event log
  to `logs/hri_scenario_<time>.jsonl`, or to the file given with `--log`.
  A simulation restart sends the clock back, and the log then has
  `clock_jump`: that run is not valid, so start the scenario again too.
  `pixi run hri-scenario-early-bag` plays the variant in which guest 2
  answers the first hand-over request before the bag hangs.
- **Onlookers stand at the door.** With each guest, two other people stand
  in view of `door`: one far behind the guest in the hallway, and one
  beside the robot by the TV desk. They never answer or follow, and they
  leave when the guest follows the robot. `find_people(in_front=True)`
  leaves both out, so the agent finds only the guest.
- **Faces count.** The faces the scenario draws clear the agent's angular
  face gate on the simulation's camera, so recognition in the simulation
  uses faces and appearance, as on the robot.
- **The agent** is `agent-sim` on `config/locations.chula_pat14_6.toml`,
  with `door`, `living_room` and `kitchen`. The map frame there is the
  captured map, not the start pose.
- **The task.** Send it on `/from_human` as in the blank map's terminal 4.
  The HRI template is not in `config/task_templates.toml` yet, so use the
  task text from [The template](architecture/hri.md#the-template).

!!! warning "One simulation per machine, unless you separate them"
    Every launch rewrites the generated world files under
    `$ROS_HOME/mujoco_worlds/chula_pat14_6/<variant>/`
    (`~/.ros/mujoco_worlds/` by default). A second simulation on the same
    machine needs its own `world_output_dir:=` on the launch, and its own
    `ROS_DOMAIN_ID` in all of its terminals.

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
- [Setup Log](setup/index.md): how this repository was built, one step at a
  time.
- [Writing Docs](setup/02-documentation.md#writing-a-new-page): how to add
  your own page.
