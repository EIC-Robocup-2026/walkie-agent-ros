# Run the Chula Simulation on a Private Graph

`scripts/chula-sim.sh` runs walkie-ros's MuJoCo world of the Chula PAT14-6 room
with Nav2, the oracle that stands in for walkie-vision, a speech log and the
agent, and, when asked, MoveIt and the arm commander. Everything it starts joins
a ROS graph of its own.

**Why a private graph.** Every `pixi run` in this repository and in walkie-ros
connects to a Zenoh router on `localhost:7447`. A simulation started there shares
`/tf`, `/clock` and `/navigate_to_pose` with whatever else uses that router: a
robot on the network, a teammate's simulation, a second agent. The script's
processes connect only to a router of their own on port 7499 and use
`ROS_DOMAIN_ID` 77, so the two graphs never meet.

## What you need

**walkie-ros, with the Chula world.** The world is on walkie-ros's
`feat/chula-pat14-6-simulation` branch (adb9725), not on main. Its pixi
environment must be installed and its `install/` built. Either check the branch
out and build it, or keep your checkout and build the branch's two changed
packages in a worktree over it:

```bash
git -C ~/walkie-ros worktree add --detach /var/tmp/wr-chula feat/chula-pat14-6-simulation
cd /var/tmp/wr-chula
pixi run --as-is --manifest-path ~/walkie-ros/pixi.toml bash -c \
  'colcon build --symlink-install --packages-select robot_bringup walkie_mujoco_worlds \
     --cmake-args -DCMAKE_PREFIX_PATH=$CONDA_PREFIX'
```

- `--detach` checks out the branch's commit, not the branch, so you can still
  check the branch out in `~/walkie-ros`.
- `--as-is` runs walkie-ros's environment as it is installed. Pixi neither
  installs nor updates it, whatever the worktree's `pixi.lock` says.
- The build takes seconds. The two packages in the worktree's `install/` take
  precedence over the checkout's own.

The overlay relies on the rest of the checkout's `install/`. Apart from the two
packages, the branch differs from main 6ad4e42 only in the `navigation` submodule
(main b6bb293, the branch 359719f) and in one dependency,
`ros-jazzy-trac-ik-kinematics-plugin`. Nav2 therefore runs whatever navigation
commit the checkout last built, and only `up --arms` uses MoveIt.

**For `--arms`, walkie-ros's arm packages.** `up --arms` runs MoveIt from
`openarm_bimanual_moveit_config` and the arm commander from
`openarm_bimanual_commander_cpp`. The overlay builds neither, so both come from
the checkout's `install/`, and `up --arms` refuses to start if either is not
built there. MoveIt's `kinematics.yaml` names TRAC-IK for every arm group, and
the plugin is the one dependency only the branch adds. If the checkout's
environment lacks it, move_group starts with no IK solver for the arms
(walkie-ros f695b3a). The agent's arm tools send joint goals and gripper
commands only, and neither needs one.

**This repository, built.** You need `pixi run build-ros2`, for
`walkie_vision_msgs` and `rai_interfaces`. You also need `pixi run build-rmw-zenoh`
for `rmw_zenoh_cpp` 0.2.11, whose `rmw_wait` does not block forever
([ADR-0032](../architecture/decisions/0032-rmw-zenoh-from-source.md)). The script
refuses to start without them.

**A model key, for `--agent`.** The agent reads `OPENAI_API_KEY` from the
environment or from `.env` at the root of this repository. A Git worktree has no
`.env` of its own. Without a key, the agent exits at startup, and `up` fails with
`the agent did not start`, leaving the other parts running.

**`python3` 3.9 or newer on `PATH`, on Linux 5.3 or newer, x86-64 or arm64.**
`scripts/chula_sim_owned.py` finds and signals each part's processes for the
script (see [Stop it](#stop-it)). Ubuntu 24.04's own `python3` will do.

## Start it

```bash
export WALKIE_ROS=~/walkie-ros WALKIE_ROS_OVERLAY=/var/tmp/wr-chula
scripts/chula-sim.sh up --agent
scripts/chula-sim.sh check --agent
```

`up` starts its parts in order and waits for each one that others need.

| Part | What it runs | `up` waits until |
| --- | --- | --- |
| `router` | `rmw_zenohd`, listening on `localhost:7499` | the port accepts connections |
| `bringup` | `nav_mujoco_omnibot.launch.py world:=chula_pat14_6`, headless | Nav2 logs `Managed nodes are active` (300 s at most) |
| `moveit` | `moveit_only.launch.py hardware_type:=mujoco use_rviz:=false`, with `--arms` only | move_group logs `MoveGroup context initialization complete` (120 s at most) |
| `commander` | `launch/commander_sim.launch.py`, the arm commander on sim time, with `--arms` only | the commander logs `Bimanual Commander Node Initialized` (120 s at most) |
| `oracle` | `pixi run oracle-chula` | |
| `speech` | `ros2 topic echo /to_human`, into `speech.log` | |
| `agent` | `pixi run agent-sim-chula --tool-log ...`, with `--agent` only | |

- Running `up` again starts only the parts that are not running. A part that
  runs keeps the arguments it started with, so an agent started without
  `--arms` stays without it until `down`.
- `--gui` opens MuJoCo's viewer and RViz.
- Arguments after `--` go to the agent, and need `--agent`:
  `scripts/chula-sim.sh up --agent -- --tool-set hri` runs the HRI tool set.
  `agent-sim-chula` already passes `--vision`, so `find_people` reads the
  oracle's people.
- `--arms` starts MoveIt once Nav2 is up, and the arm commander once MoveIt
  is ready, since the commander connects to move_group as it starts. Both run
  in walkie-ros's environment with the overlay sourced, as `bringup` does, so
  they load the packages that `bringup` loads. With `--agent`, the agent also
  gets `--arms`, after the arguments given after `--`, so those must pick a set
  with the arm tools: `-- --tool-set hri` or `-- --tool-set arms`. With the
  default set the agent exits with code 2 at startup. Without `--arms`, `up`
  starts what it always did.
- walkie-ros's own `commander.launch.py` declares no `use_sim_time`, so
  passing `use_sim_time:=true` to it does nothing. `launch/commander_sim.launch.py`
  includes that file and sets the parameter on its node.
- `up --arms` was first run on 2026-11-07 (decision S6-17 in
  [Mahoraga Demo Decisions](../architecture/mahoraga-demo-decisions.md)). The
  real-time factor was 0.91 over 10 s of wall-clock time. A move of both arms
  to `hands_up` or `home` took 8.4 s to 8.9 s, of which MoveIt's planning
  took up to 5 s, a read 0.1 s and a gripper command 0.2 s to 0.5 s, well
  inside `SIM_ARM_BOUNDS`. With the arms in `hands_up`, a drive was refused
  before it started, and after `home` it drove. MuJoCo starts the right
  gripper closed, so the first read reports it closed.
- The commander takes a cancel during MoveIt's planning but answers it only
  once the plan is done, a few seconds later. A `down` during a move then
  waits for that answer: the agent logs that the move "has not ended: a
  cancel was sent, but no result has come", publishes "stop" again on each
  retry, and exits once the result comes. walkie-ros's commander then
  panics in Zenoh's thread-local storage as it exits, with code -6, after
  the goal has ended.
- `up` refuses to start beside another `nav_mujoco_omnibot.launch.py`, on any
  graph, because two simulations compete for the CPU.
- `up` returns only when the oracle and the agent have logged that they are up,
  and every part is still running a few seconds later. If a part exits first,
  `up` fails, names it, and lists the parts it leaves running.
- `up` and `down` hold a lock on `SIM_DIR`, so a second `up` or `down` at the
  same time refuses to run. Each part's starter holds the lock too, until it has
  written the part's record. If `up` gives up waiting for a record (5 s), a `down`
  run at once still refuses, instead of finding no record and missing the part.

**The robot starts facing the people.** The world's own spawn faces south, away
from both people of the oracle's scene. The script spawns the robot at the
`living room` place of `config/locations.chula_pat14_6.toml`, facing east, so the
head camera sees the guest and the host from the first image. AMCL's initial pose
follows the spawn. `SIM_SPAWN_POSE` takes another `"X Y YAW_DEG"`.

**The speech log.** The agent's `say` needs a listener on `/to_human`. The
`speech` part writes each line to `speech.log` instead of voicing it, so the
simulation makes no sound.

`check` waits until each item has arrived (120 s at most), prints one line for
each, and exits 0 only when all arrived. It publishes nothing. This was the
output on 2026-11-06:

```text
ok       clock    26.4 s of simulated time
ok       robot    at (1.763, -1.483) in map, yaw 0.0 deg
ok       camera   640x360, fx 224.057, frame zed_head_left_camera_frame_optical, at (0.275, 0.060, 1.145) m on base_footprint, 9.0 deg down
ok       people   2 in view: track 1 (raising_left_arm, standing, waving); track 2 (sitting)
ok       objects  3 in view: sofa (track 11), table (track 12), mug (track 13); classes ['chair', 'sofa', 'armchair', 'stool', 'table', 'mug']
ok       agent    node walkie_agent is up
```

The camera line matches the constants that the oracle's tests assume, in
`tests/test_projection.py`. The gestures are those that `perception/gestures.py`
reads from the oracle's keypoints, as walkie-vision's would be read.

`check --arms` adds a `commander` line. It passes once it finds a server for
each of the three names the agent's arm client moves and reads the arms
through: the `/set_joint_position` and `/control_gripper` actions and the
`/get_joint_states` service. It looks for them through clients of the same
names and types, and sends no request. So it cannot tell a commander built
from another `my_robot_interfaces`: `rmw_zenoh_cpp` finds such a server by its
type's name, but delivers requests by the type's hash, so none ever arrives.

## Talk to the agent

`env` prints three exports. Run any `pixi run` command after them, and it joins
the private graph.

```bash
eval "$(scripts/chula-sim.sh env)"
pixi run ros2 topic pub -w 1 --once /from_human rai_interfaces/msg/HRIMessage \
  "{text: 'Please go to the kitchen room.'}"
tail -f /var/tmp/walkie-chula-sim/logs/speech.log
```

`-w 1` waits for one subscriber before it publishes, so the task is not lost
while discovery is still running. Each tool call is a line of
`logs/agent-tools.jsonl`.

With `up --agent -- --tool-set hri`, the agent also has the HRI tools, and
`find_people` reads the oracle's two guests. Ask "Is anyone waving at you?" and
the tool log shows `find_people` returning `person_1`.

### Scripts and the verdict

`--scripts` gives the agent `run_script`
([Scripts on the agent](../architecture/agent.md#scripts-on-the-agent)). It needs
`bwrap` on the computer. Without it the agent exits with code 2 at startup, and
`up` fails with `the agent did not start; see .../logs/agent.log`. The log's last
lines, which `up` prints, include the agent's own `walkie-agent: --scripts runs
scripts only inside bwrap` line.

```bash
scripts/chula-sim.sh up --agent --arms -- --tool-set hri --people-memory mock --scripts
```

Give it a task with several steps, such as "Go to the sofa, then the table, then
come back to the living room." The tool log shows each `run_script` call with its
script and a `calls` list, one `tool: status` line for each call the script made.

Once the task has ended and you have checked what the robot did, judge it on
`/walkie/verdict`, `true` if it did the task and `false` if not. The console
keeps no messages for a late listener, so start the first command in another
terminal before you publish:

```bash
eval "$(scripts/chula-sim.sh env)"
pixi run ros2 topic echo /walkie/console rai_interfaces/msg/HRIMessage --field text
```

```bash
eval "$(scripts/chula-sim.sh env)"
pixi run ros2 topic pub -w 1 --once /walkie/verdict std_msgs/msg/Bool "{data: true}"
```

The agent answers on `/walkie/console` with one line: which functions it
verified, which a `false` verdict judged, that the task proved none, or that it
ignored the verdict. A `false` verdict verifies nothing, and the functions it
judged get a `rejected` event in the store's `ledger.jsonl`. It ignores
a verdict while a task runs and a second verdict on the same task. A question
the agent asks ends a task too, so judge after its last reply. Each verdict is
also a line of `logs/agent-tools.jsonl`, with the tool `verdict`.

## Stop it

```bash
scripts/chula-sim.sh down
```

`down` stops the parts in reverse order: the agent first, and the commander and
MoveIt before `bringup`. It sends each process of a part SIGINT,
as Ctrl+C would. If the part still runs 10 s later, it sends SIGTERM, and 5 s
after that SIGKILL, then SIGKILL again 2 s later. Each check starts the helper,
so these waits run a little longer.

A part keeps its record if it is still running after all that, or if its
processes could not be looked up or signalled. `down` still goes on to the other
parts, then exits 1. `status` shows which parts are running, and says `unknown`
for a part whose processes it could not look up.

Each signal goes through a pidfd, a descriptor that refers to one process and
never to a later one with the same ID. `chula_sim_owned.py` opens it, checks the
process's group and token, and only then signals it. Without it, a part that
exits just after the check could pass its group's number to another program
before the signal arrives.

A listing of `/proc` is not a snapshot. A child that a part's process starts
just as it exits could appear in no listing, and `down` would then remove the
record of a part still running. So the helper reads the last process ID the
kernel gave out, from `/proc/sys/kernel/ns_last_pid`, before it lists `/proc`
and again after it has read every listed process. Linux gives out IDs in turn,
so every process started in between has an ID in that range. The helper reads
those IDs one by one, and reads the last ID again, until it has stopped moving.
Processes that other programs start and end, as a parallel build does, cost a
read of their IDs, not another listing.

That finds most children, not every one. Linux gives a new process its ID before
`/proc` shows it, so a child read in that moment looks absent. `stop` therefore
reads a part that looks stopped a second time, 0.2 s later (`CONFIRM_S`), before
it removes the record. That is a margin, not a proof. A strict answer would need
the kernel to hold the part's processes together, as a cgroup does.

The helper tells a process from a later one with the same ID by its start time.
So a child also counts when it takes the ID of a process already read. If the
IDs pass `pid_max` and start again from the bottom, the helper lists `/proc`
again. A wrap that brings the last ID back past the earlier reading goes unseen.
The kernel skips IDs in use, so that takes one new process for each free ID:
about four million on this laptop, but few when nearly every ID is taken. If the
last ID still moves after 1000 rounds of reading, the helper reports an error
rather than a list that may be stale.

## Where things go

Everything goes under `SIM_DIR`, `/var/tmp/walkie-chula-sim` by default.

- `logs/`: one log per part, and the agent's tool log.
- `pids/`: one record for each part: its process group, the boot it started in,
  and a token that every process of the part carries in `CHULA_SIM_TOKEN`. A
  part's processes are the members of its group that carry its token. `status`
  counts only those, and `down` signals only those. So after a reboot, or once
  the system gives the group's number to another program, `down` never signals
  that program. A process that a part started in a session of its own, such as
  the `ros2` daemon, is not in the group and is left running.
- `ros/`: `ROS_HOME` and the ROS logs.
- `worlds/`: the world the launch file generates.

The functions an agent with `--scripts` saves are not under `SIM_DIR`. They go to
the function store, `WALKIE_TOOLS_HOME` if the shell that runs `up` sets it, and
otherwise `src/walkie_agent/tools/` of the checkout that runs the agent. A saved
function is in `temporary/<name>.py`, a version a passed verdict verified is also
in `bundled/<name>.py`, and `ledger.jsonl` records each save, each script that
used a saved function, and each version a verdict verified or rejected. Git
ignores all three there. To start from an empty store, set `WALKIE_TOOLS_HOME`
to a new folder outside any git work tree before `up`.

`SIM_ZENOH_PORT`, `SIM_DOMAIN_ID` and `SIM_CPUS` (`taskset -c`) are the other
settings. The script's header lists them.

## What has been run

On 2026-11-06, with the overlay above and every part on 8 of the laptop's 12 CPUs:

- `check --agent` passed, with the output above.
- One task with the real model, "Please go to the kitchen room.". `go_to` took
  32 s and stopped 0.10 m from the goal. The `say` after it failed, because the
  recipe had no `speech` part yet.
- Each Chula place, as a direct Nav2 goal with no model, in the order below.
  The script waited 180 s at most for each result.

| Place | Result | Time | Stopped off the goal by |
| --- | --- | --- | --- |
| shelf | SUCCEEDED | 42.4 s | 0.091 m, +0.5° |
| bar | SUCCEEDED | 30.0 s | 0.096 m, -0.4° |
| table | SUCCEEDED | 54.2 s | 0.063 m, +0.6° |
| tv | SUCCEEDED | 67.6 s | 0.096 m, -0.3° |
| sofa | SUCCEEDED | 23.7 s | 0.099 m, +0.1° |
| entrance | SUCCEEDED | 25.4 s | 0.042 m, -0.5° |
| trash, at (3.0, -5.5) | ABORTED | 115.4 s | 0.155 m, +2.8° |
| instruction point, at (2.8, -2.5) | no result | 180.1 s | 0.451 m, +0.3° |
| kitchen room | SUCCEEDED | 34.8 s | 0.052 m, -0.4° |
| living room | SUCCEEDED | 36.9 s | 0.025 m, +0.3° |

The two failures were the invented places. Nav2's goal checker accepts 0.1 m.
On both legs the controller logged `Failed to make progress` again and again,
and the collision monitor kept stopping the robot. The first `trash` sat 0.41 m
from the east wall, and the first `instruction point` 0.23 m from the map's
keepout zone by the sofa.

Both moved to more open floor (decision G4-6 in
[Mahoraga Demo Decisions](../architecture/mahoraga-demo-decisions.md)), and
were driven again:

| Leg | Result | Time | Stopped off the goal by |
| --- | --- | --- | --- |
| entrance | SUCCEEDED | 15.4 s | 0.023 m, +0.2° |
| trash, at (2.7, -5.45) | ABORTED | 140.3 s | 0.114 m, +2.7° |
| instruction point, at (2.2, -2.45) | SUCCEEDED | 69.6 s | 0.099 m, +0.0° |
| living room | SUCCEEDED | 21.3 s | 0.041 m, +0.6° |
| trash, at (2.75, -4.2) | SUCCEEDED | 81.2 s | 0.019 m, -0.4° |
| shelf | SUCCEEDED | 15.9 s | 0.091 m, -0.2° |
| trash, at (2.75, -4.2) | SUCCEEDED | 37.3 s | 0.003 m, -0.5° |
| living room | SUCCEEDED | 76.6 s | 0.044 m, +0.4° |

The second `trash`, between the kitchen table and the east wall, failed the
same way, 0.53 m from the table. The `sofa` goal is 0.31 m from the coffee
table and succeeds, so the gap to furniture alone does not predict a failure.
The cause was not found. The third `trash`, in the open corner north of the
kitchen table, succeeded from the living room and from the shelf.

- After the review fixes (G4-7), `up --agent` in a worktree without `.env`
  stopped at `the agent did not start` and listed the four parts still running.
  With the key exported, a second `up --agent` started only the agent, `check
  --agent` passed again, and all 32 processes of the five parts carried their
  part's token. `down` took 12 s and left none of them.
- After the pidfd signalling (G4-8), on 2026-11-07, `up --agent` and `check
  --agent` passed again, and the five parts held 32 processes with their tokens.
  `down` took 11.8 s, 10.3 s of them for `bringup`. Nav2's `planner_server` and
  `controller_server` outlived SIGINT and SIGTERM until `ros2 launch` sent them
  SIGKILL, inside `down`'s 10 s SIGINT wait. `down` sent nothing after SIGINT.
  With the repeated listings and the unknown state (G4-9), a second run gave
  the same result. `status` showed all five parts running, and `down` exited 0
  in 12.5 s with no record left.
- The helper of G4-10 failed while another program started about 200 short
  processes a second. 20 calls out of 20 on an empty group reported an error
  after about 1 s each. With the last-ID reading (G4-11), all 20 calls exited 0
  in 37 ms on average. Under a tighter loop of `/bin/true` on two CPUs, the old
  helper failed 2 calls of 20 and took 721 ms on average; the new one passed all
  20 at 59 ms. With nothing else running, a call takes 35 ms. On the real
  stack, `up` and `check` passed, the helper found 22 processes for `bringup`,
  3 for `router`, 3 for `speech` and 2 for `oracle`, and `down` exited 0 in
  11.1 s with no record left.
- One more task with the real model, "Please say hello.". The agent called
  `say("Hello!")`, and `speech.log` showed the line. The log also begins with
  two of Zenoh's configuration warnings, which `ros2 topic echo` prints at
  startup.
- With people perception (S1), on 2026-11-07: `up --agent -- --tool-set hri`
  and `check --agent` passed. Two tasks with the real model:
    - "Is anyone waving at you?" The agent called
      `find_people("waving person")`, which returned in 0.05 s with `person_1`,
      a standing person raising their left arm, in the living room. It said
      "Yes, I can see one person standing in front of me with a hand raised."
    - "Who can you see right now?" `find_people()` returned `person_1` and
      `person_2`, a sitting person with no gesture, both in the living room.
      The agent said both.

    `down` left no record, and only the user's own router kept running.
- Turns on the spot (S2), on 2026-11-07, as direct `RosBase.move_relative(0,
  0, angle)` calls with no model: 17 turns from 4° to 90°, left and right. 15
  succeeded, each within 0.72° of the goal, in 1.6 s to 4.9 s, with the base
  drifting 0.002 m to 0.067 m. Two aborted with Nav2 error code 105, a 40° turn
  after 9.6 s and a 4° turn after 10.1 s, 2.90° and 2.97° past the goal, with
  0.080 m and 0.093 m of drift.
- "Look at me, I am the one waving." with the real model and the `hri` tool
  set, twice. The guest stands 1.83 m from the base's centre, 35.6° to the left.
    - Before the floor distance (S2-3), the agent called
      `find_people("waving person")`, then `face_person("person_1")`, which
      turned the base 41° in 11.1 s, by the camera's direction of 39.6°, and
      said "Here I am, looking right at you. Nice wave!". Nav2 aborted 1.3°
      off the goal, which `face_person` counted as done, and the base drifted
      0.08 m. The guest ended 6.4° to the right of the base's heading.
    - With the floor distance, the same calls. `face_person` turned 39.3° in
      9.1 s; Nav2 aborted with code 105 3.0° past the goal, which `face_person`
      counted as done. The agent said "All right, I am looking at you now.".
      The base drifted 0.08 m again, and the guest ended 4.0° to the right of
      its heading. Five readings of `RosVision.people()` after the turn, with
      no model, gave the guest's bearing as -3.4° to -4.2° against a true
      -3.99°, so what is left is Nav2's overshoot.

    `down` left no record.
- With the people memory (S3), on 2026-11-07: `up --agent -- --tool-set hri
  --vision --people-memory mock`, then four tasks with the real model:
    - "Hi, I am the one waving at you. My name is Sam, please remember me."
      `find_people("waving person")` returned `person_1`, and
      `remember_person("person_1", "Sam")` stored 18 faces and 20 appearance
      vectors. The agent said "Nice to meet you, Sam. I have remembered you."
    - "Please turn around to look behind you, and then turn back to face us."
      The first `move_relative(0, 0, 180)` failed after 10.4 s with Nav2 error
      code 105. By the headings that followed, the base had turned about 184°.
      The model retried the turn, which ended at a heading of 4°, and the turn
      back left the base at -177°, facing away from the guests. The agent said
      it had turned back to face them. Decision S2-11 now counts such a turn as
      done.
    - "You are facing away from us now. Please turn around once more, and then
      tell me who you can see." One more 180° turn, to 3°, then `find_people()`
      returned `person_1` named Sam and `person_2` with no name. The agent said
      Sam was standing and raising a hand, beside someone it did not know yet.
    - After `/walkie/reset`, "Who can you see right now?" `find_people()`
      named nobody, and the agent said it did not know either name.

    `down` left no record.

## Not in the recipe

- **MoveIt's RViz.** `--gui` opens `bringup`'s RViz only. MoveIt runs with
  `use_rviz:=false`.
- **The people database.** `--people-memory mock` keeps the people memory in
  the agent's process. walkie-database's node, for `--people-memory ros`, and
  its sim seed are not started by the script.
- **Voice.** Speech is logged, not voiced. RAI's voice nodes on a PipeWire null
  sink are not part of it ([Talk to Walkie by voice](07-voice.md)).
