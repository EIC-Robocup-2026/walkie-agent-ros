# Run the Chula Simulation on a Private Graph

`scripts/chula-sim.sh` runs walkie-ros's MuJoCo world of the Chula PAT14-6 room
with Nav2, the oracle that stands in for walkie-vision, a speech log and the
agent. Everything it starts joins a ROS graph of its own.

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
commit the checkout last built, and nothing here uses MoveIt.

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

`up` starts five parts in order and waits for each one that others need.

| Part | What it runs | `up` waits until |
| --- | --- | --- |
| `router` | `rmw_zenohd`, listening on `localhost:7499` | the port accepts connections |
| `bringup` | `nav_mujoco_omnibot.launch.py world:=chula_pat14_6`, headless | Nav2 logs `Managed nodes are active` (300 s at most) |
| `oracle` | `pixi run oracle-chula` | |
| `speech` | `ros2 topic echo /to_human`, into `speech.log` | |
| `agent` | `pixi run agent-sim-chula --tool-log ...`, with `--agent` only | |

- Running `up` again starts only the parts that are not running.
- `--gui` opens MuJoCo's viewer and RViz.
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

## Stop it

```bash
scripts/chula-sim.sh down
```

`down` stops the parts in reverse order. It sends each process of a part SIGINT,
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

The helper lists `/proc` again until a listing shows no new process that might
be in the part's group. Otherwise a child that a part's process starts just as
it exits would appear in no listing, and `down` would remove the record of a
part still running. The helper tells a process from a later one with the same ID
by its start time. So a child also counts when its parent's ID passes to another
program before the helper reads it, or when the child takes the ID of a process
already read. If the group keeps gaining processes for 100 listings, the helper
reports an error rather than a list that may be stale.

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
- One more task with the real model, "Please say hello.". The agent called
  `say("Hello!")`, and `speech.log` showed the line. The log also begins with
  two of Zenoh's configuration warnings, which `ros2 topic echo` prints at
  startup.

## Not in the recipe

- **Arms.** MoveIt and the arm commander come with S6 of the
  [roadmap](../architecture/see-remember-act.md). The commander is not in this
  repository yet.
- **Memory.** walkie-database's sim seed and memory node come with S3.
- **Voice.** Speech is logged, not voiced. RAI's voice nodes on a PipeWire null
  sink are not part of it ([Talk to Walkie by voice](07-voice.md)).
