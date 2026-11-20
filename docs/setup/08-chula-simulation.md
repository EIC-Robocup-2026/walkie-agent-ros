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

**For `--whole-body`, whole-body MoveIt and a world-aware bringup.**
`up --whole-body` runs walkie-ros's `whole_body_demo.launch.py` as `bringup`:
Nav2, move_group with `base_mode:=whole_body`, the commander and the base
follower in one launch, with map to odom from MuJoCo's ground truth and no
AMCL (decision PK-2). `pick_object` needs it. Two workspaces provide it:

- `WALKIE_ROS`, a walkie-ros workspace on `feat/moveit-full-body` with
  `openarm_bimanual_moveit_config` and `openarm_bimanual_commander_cpp` built,
  such as `~/worktrees/walkie-ros-fullbody`.
- `WALKIE_ROS_OVERLAY`, an overlay whose `robot_bringup` runs
  `whole_body_demo.launch.py` in a named world (`world:=`), with
  `walkie_mujoco_worlds` built beside it, such as a worktree of
  `feat/sim-bottle-pick` built like the overlay above, in `/var/tmp/wr-stack`.

`up --whole-body` refuses to start if either package is missing from
`WALKIE_ROS`, or if the `whole_body_demo.launch.py` it would load takes no
`world:=`; an older one would start its own obstacle scene.

**This repository, built.** You need `pixi run build-ros2`, for
`walkie_vision_msgs` and `rai_interfaces`. You also need `pixi run build-rmw-zenoh`
for `rmw_zenoh_cpp` 0.2.11, whose `rmw_wait` does not block forever
([ADR-0032](../architecture/decisions/0032-rmw-zenoh-from-source.md)). The script
refuses to start without them.

**For `--vision`, walkie-vision.** Set `WALKIE_VISION` to a walkie-vision
checkout of `feat/scene-graph-prep` or later, the first with the debug
visualizer. Its pixi environment must be installed, and `pixi run build-msgs`
and `pixi run fetch-models` run there. `up --vision` refuses to start without
the environment, the visualizer, the task's file or the built messages. The
vision launch checks the models itself, and names the command that fetches them.

**For `--memory`, walkie-database and a seeded store.** Set `WALKIE_DATABASE`
to a walkie-database checkout of `feat/scene-graph-prep` at 45228e6, the
commit `ros2_ws.repos` pins, or later, with its pixi environment installed and
`pixi run build-msgs` run there. Three of its commits matter here. From 583bcd8,
the memory merges a furniture track that lies mostly inside a seeded piece of
furniture's box into that piece, whatever label the detector gave it; cups and
other movable objects are never merged, and stand `on` the furniture instead
(walkie-database's ADR-0028). From 8546d28, `db-seed-sim` measures the
furniture from the world's MuJoCo scene. From ca52814, the furniture file
gives each piece an official name, such as `coffee table`, which the agent
uses as the piece's name (decision SG-27). The node at 45228e6 already serves
official names, so only the seed needs ca52814 or later.
The memory keeps what it merges in a store seeded for this map,
`SIM_MEMORY_DB`. `scripts/chula-sim.sh reset-memory` makes that store, or
empties it again, from the seed `SIM_MEMORY_SEED` (decision SG-26):

```bash
scripts/chula-sim.sh reset-memory
```

It runs walkie-database's `db-restore` in `WALKIE_DATABASE`, and refuses while
the memory runs, so run `down` first. `db-restore` keeps the store's ID
lineage. It refuses, and changes nothing, when the seed would give an object
ID that the store has already given to something else, and its message says
how to rebuild the seed. A new `SIM_MEMORY_DB` path starts a lineage of its
own. `up --memory` refuses to start without the environment, the object
memory, the built messages, the store or the map folder. The node itself checks
that the store was seeded for this map, map ID and clock, and stops with
`Not starting:` when it was not. These defaults are current and will change as
the scene graph work goes on:

| Setting | Default |
| --- | --- |
| `SIM_MEMORY_DB` | `/var/tmp/scene-graph-demo/sim/walkie.db` |
| `SIM_MEMORY_SEED` | `/var/tmp/scene-graph-demo/furniture/seed.db` |
| `SIM_MEMORY_MAP_ID` | `chula-2026-11-12-714b087d` |
| `SIM_MEMORY_MAP_FOLDER` | the `chula_pat14_6` map that `bringup` loads, from the overlay's `install/` |

The seed holds the two rooms and the world's nine pieces of furniture,
measured from its MuJoCo scene, with object IDs 1 to 9 and their official
names. It was first built as walkie-database's
`docs/setup/15-scene-graph-demo.md` describes, with this command in
walkie-database's checkout, and then rebuilt from ca52814 with
`--previous-seed` to add the names, keeping the IDs. Build it from ca52814 or
later: the furniture file before it has no names, and the coffee table then
reads `table 2`. A rebuild cannot write over the seed it
starts from: write it to a new `--output`, add `--previous-seed` with the old
seed so that each piece keeps its ID, and point `SIM_MEMORY_SEED` at the new
file. If `reset-memory` then refuses it, `db-restore`'s message names the
`--id-floor` to rebuild with.

```bash
W=/var/tmp/wr-chula/simulation/mujoco/worlds/chula_pat14_6
A=/var/tmp/wa-demo
pixi run --as-is db-seed-sim --map-folder $W/map --arena chula --date 2026-11-12 \
  --clock-kind ros_sim \
  --world-source $W/authoring/source/30_16102026_ChulaPat14_6_world.toml \
  --locations $A/config/locations.chula_pat14_6.toml \
  --room-names /var/tmp/scene-graph-demo/room-names.yaml \
  --scene $W/scene.xml --furniture tools/furniture/chula_pat14_6.yaml \
  --previous-seed /var/tmp/scene-graph-demo/furniture/seed.db \
  --output /var/tmp/scene-graph-demo/furniture/seed.new.db
```

The seed already exists, so run this only to change the furniture file or the
rooms. walkie-database refuses to write a seed over a seed of the same map, even
with `--replace`, and refuses an `--output` that is the previous seed itself.
To run the command again to the same `seed.new.db`, add `--replace`; with
`--previous-seed` given, it is allowed. Then set `SIM_MEMORY_SEED=/var/tmp/scene-graph-demo/furniture/seed.new.db` for
`reset-memory`, and keep the old seed for a rebuild with `--id-floor`. On
2026-11-12, at walkie-database 41dae92, this command wrote a seed with 2 rooms
and 9 anchors.

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
| `bringup` | `nav_mujoco_omnibot.launch.py world:=chula_pat14_6`, headless; with `--whole-body`, `whole_body_demo.launch.py world:=chula_pat14_6 start_router:=false` instead | Nav2 logs `Managed nodes are active` (300 s at most); with `--whole-body`, then also `MoveGroup context initialization complete` and `Bimanual Commander Node Initialized` (120 s each at most) |
| `moveit` | `moveit_only.launch.py hardware_type:=mujoco use_rviz:=false`, with `--arms` only | move_group logs `MoveGroup context initialization complete` (120 s at most) |
| `commander` | `launch/commander_sim.launch.py`, the arm commander on sim time, with `--arms` only | the commander logs `Bimanual Commander Node Initialized` (120 s at most) |
| `vision` | walkie-vision's `pixi run vision task:=memory sim:=true viz:=true`, with `--vision` only | the node logs `loaded the models in` (300 s at most), after the oracle and the agent |
| `memory` | walkie-database's `pixi run db-memory` on `SIM_MEMORY_DB`, reading `/walkie_vision/sim/objects` on sim time, with `--memory` only, which needs `--vision` | the node logs `The object memory starts with` (120 s at most), after `vision` |
| `foxglove` | `foxglove_bridge_launch.xml` on `127.0.0.1:8799`, from walkie-ros's environment, with `--foxglove` only | the bridge logs `Server listening on port` (120 s at most), after `memory` |
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
- `--whole-body` takes the place of `--arms`, and `up` refuses the two
  together. Its launch starts move_group and the commander itself, so there is
  no `moveit` or `commander` part, and `down` stops them with `bringup`. With
  `--agent`, the agent gets `--arms` as with `--arms`. `up --whole-body`
  refuses to start while the `moveit` or `commander` part of an earlier
  `up --arms` runs, since two commanders would serve the same actions.
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
- `up` refuses to start beside another `nav_mujoco_omnibot.launch.py` or
  `whole_body_demo.launch.py`, on any graph, because two simulations compete
  for the CPU.
- `up` returns only when the oracle, the agent, `vision` and `foxglove`, each
  when asked for, have logged that they are up, and every part is still
  running a few seconds later. If a part exits first,
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

`check --whole-body` adds a `move_group` line, which passes once a node named
`move_group` is on the graph, and a `commander` line that also needs the
`/go_to_pose` action, through which `pick_object` reaches.

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

### Ask about remembered objects

`--object-tools` starts the agent with `--object-memory`, which adds three
tools that read walkie-database's object memory: `recall_object`,
`list_room_objects` and `go_to_object` (decisions SG-16 to SG-21 in
[Mahoraga Demo Decisions](../architecture/mahoraga-demo-decisions.md)). It
needs `--memory` and `--agent`, and the `default` or `hri` tool set; `gpsr`
and `arms` have no object memory tools, and the agent refuses to start with
them. It also points the agent's `find_objects` at walkie-vision's
`/walkie_vision/sim/objects`, the detections that fill the memory, since the
oracle's scene has no cup to check (SG-23). An `--objects-topic` after `--`
is kept instead.

The agent's `walkie_database_msgs` must define the services as
`WALKIE_DATABASE`'s build does: build both from the commit `ros2_ws.repos`
pins, or from commits with the same messages. Under `rmw_zenoh_cpp` a client built from another
version of a service finds it but is never answered
([ADR-0035](../architecture/decisions/0035-memory-through-walkie-database.md)):
each object tool call would time out after 7 s as if the memory were busy, and
only a warning in the agent's log would hint at the stale build. So
`up --object-tools` compares the type hashes of the four services the tools
call and refuses to start when they differ. For a memory node that `up` started
and that still runs, it compares against the copy of the services `up` recorded
when it started that node, and refuses one that an `up` from before this check
started; stop that one with `down` first. `up` and `down` see only the parts
`up` started, so stop a memory node you started yourself before you run
`up --memory`.
`ros2_ws.repos` pins walkie-database 45228e6, on walkie-database's local
branch `feat/scene-graph-prep` in `/var/tmp/wd-prep`, which is not on GitHub
yet (SG-25, SG-26). Until it is, fetch it into the clone before building, as
[Local message pins](03-ros-and-rai.md#local-message-pins) describes:

```bash
git -C ros2_ws/src/walkie-database fetch /var/tmp/wd-prep \
  feat/scene-graph-prep:feat/scene-graph-prep
pixi run --frozen build-ros2
```

Then start it:

```bash
scripts/chula-sim.sh up --agent --vision --memory --object-tools --foxglove
```

Run `scripts/chula-sim.sh reset-memory` before each run, with the memory
stopped. A store kept from an earlier run holds stamps from that
run's clock, so the seconds since an object was seen read as unknown or wrong
(SG-20). The memory only knows what vision has shown it, so drive the robot
round the rooms first. Then ask:

- "Where did you see a cup?" The tool log shows `recall_object` returning up
  to 10 remembered cups, each with its memory id, its room, the anchor it
  stands on (`on`, such as `coffee table 2`), the named place nearest it
  (`near_place`) and the seconds since vision last saw it. `complete` is false
  when there are more. An object the detector only once called a cup, such as
  a bottle, comes after the cups, under its own name. An object's name, and
  the anchor in `on`, are its official name when the memory has one, else its
  label (SG-27). A seed built before walkie-database ca52814 has none, and the
  coffee table reads `table 2`.
- "What is in the living room?" `list_room_objects` lists what the memory
  holds there. A room name in any case works.
- "Take me to the cup on the table." `go_to_object` drives to the named place
  nearest the cup, within 2 m, and turns to face it. The agent then calls
  `find_objects` to check that the cup is still there. When an object's name
  is not a word the detector uses, such as a seeded `stool`, the reply names
  the label to look for instead, `chair`.

The ids these tools give are the memory's, numbers such as `18`; they are not
the ids `find_objects` gives for objects in view. An object more than 2 m from
every place in `config/locations.chula_pat14_6.toml` cannot be driven to, and
the agent is told to go to its room and look instead. Without `--object-tools`
the agent has none of these tools, and its prompt is the one it had before
them.

### Pick up a remembered object

`pick_object` picks up an object the memory holds, by its name, with the right
hand (decision PK-4). The agent has it with the `hri` tool set when it gets
both `--arms` and `--object-memory`, so start the whole-body run with the
object tools:

```bash
export WALKIE_ROS=~/worktrees/walkie-ros-fullbody WALKIE_ROS_OVERLAY=/var/tmp/wr-stack
scripts/chula-sim.sh reset-memory
scripts/chula-sim.sh up --agent --whole-body --vision --memory --object-tools -- --tool-set hri
scripts/chula-sim.sh check --agent --whole-body
```

Drive the robot past the object first, so that vision has shown it to the
memory, then ask "Bring me the water bottle." One pick runs these steps, and
the tool log shows each as a call:

1. **Find it.** The first object of that name the memory holds with a full
   position, centre height included.
   A name the arena uses, such as "water bottle", also finds what the detector
   labelled with its last word, "bottle".
2. **Drive to a stand-off pose.** 0.65 m back from the object's centre along
   the approach, and 0.20 m to the left of it, so the object is ahead of the
   right hand, through `move_relative`. The first approach is the one whose
   line the robot already stands on. When that pose is more than 3 m away, a
   `go_to` to the named place within 2 m of the object comes first. If Nav2
   cannot reach it, the headings 30° and 60° to either side are tried in turn,
   then 120° to either side and the far side, 180°.
   A drive that Nav2 aborts within 0.10 m and 10° of its goal counts as
   arrived, and so does a `go_to` that ends close enough to the stand-off pose.
   The pick then reads the object's position from the memory again.
3. **Reach.** The left arm goes to `standby`, out of the way, and the right
   gripper opens. The hand goes to a pre-grasp pose 0.10 m back from the grasp
   pose through `whole_body_right`, which may move the base, the lift and the
   arm. Once the arm is at rest, the hand moves in a straight line through
   `right_arm_lift` until its tool centre point is 0.02 m past the object's
   centre. The fingers point along the approach from where the base stopped,
   with the wrist camera bracket up. For the straight moves the agent sets the
   commander's `allow_gripper_vs_octomap`, so the fingers may touch the octomap
   voxels of the object. Without it MoveIt finds no path into the bottle.
4. **Close and lift.** The gripper closes. Only when the commander's grasp
   band says that it holds something does the hand rise 0.10 m in a straight
   line. A close on nothing brings both arms home and says so.
5. **Clear and home.** The hand moves in a straight line to 0.30 m ahead of
   the base and 0.35 m to its right, at the lifted height. Then the right arm
   goes home through `right_arm_lift`, with the lift at 0.7435 m, its SRDF home,
   and the left arm goes home, which ends the octomap allowance. Without the
   move clear, the swing home dragged the bottle across the table's corner and
   out of the fingers.

After a pick both arms are home, a travel-safe pose, so the robot may drive
with the object. All the numbers are in `config/pick.toml`, which the agent
checks at startup and reads again at each pick, so a change applies to the
next pick without a restart. On 2026-11-20 the pick of the kitchen table's
water bottle from the spawn point held it through the drive back to the living
room. The memory must have a fresh position for the object: after earlier runs
the memory had merged the bottle's carried and reset positions into one 0.6 m
object, so run `reset-memory` between runs that move objects.

## Watch it in Foxglove

```bash
export WALKIE_VISION=/var/tmp/wv-prep     # a checkout of feat/scene-graph-prep
export WALKIE_DATABASE=/var/tmp/wd-prep   # a checkout of feat/scene-graph-prep
scripts/chula-sim.sh up --vision --memory --foxglove
```

**`--vision`** runs walkie-vision's node on the head camera, with the `memory`
task, on the sim names. `SIM_VISION_TASK` picks another file of its
`config/tasks/`. The node publishes `/walkie_vision/sim/objects`. The oracle
still owns `/walkie_vision/objects`, so the agent does not read the node's
arrays. Its debug visualizer publishes two topics of standard types:

| Topic | Type | What it shows |
| --- | --- | --- |
| `/walkie_vision/sim/debug/objects_image/compressed` | `sensor_msgs/CompressedImage` | each array drawn on its own colour image: boxes, classes, scores and track ids |
| `/walkie_vision/sim/debug/objects_markers` | `visualization_msgs/MarkerArray` | each detection's points and label, in the camera's optical frame |

On MuJoCo's rendered frames, YOLOE finds small textured objects, such as the
cup, the bottle and the apple, and misses or mislabels most furniture
(decision S4-8 in
[Mahoraga Demo Decisions](../architecture/mahoraga-demo-decisions.md)).

**`--memory`** runs walkie-database's object memory on those arrays. Every 2 s
of sim time it merges the detections it has placed in the map, keeps an object
once three arrays have shown it (`confirm_views`, which the script leaves at
its default), saves the result in `SIM_MEMORY_DB`, and publishes:

| Topic | Type | What it shows |
| --- | --- | --- |
| `/walkie_database/objects/markers` | `visualization_msgs/MarkerArray` | each confirmed object's box and label in `map`, and its `on` relations |
| `/walkie_database/objects/cloud` | `sensor_msgs/PointCloud2` | the objects' points in `map` |

Both are latched, so a client that connects later still gets the last one. The
memory also serves `/walkie_database/search_objects`, `get_object`,
`find_objects`, `get_room_contents` and `list_rooms`. It saves once more as it
stops, and `down` stops it before `vision`.

**`--foxglove`** runs `foxglove_bridge` on sim time, from walkie-ros's
environment, on `ws://localhost:8799`. `SIM_FOXGLOVE_PORT` picks another port.
The script leaves Foxglove's default, 8765, free for a bridge of your own.

To watch it, use the Foxglove desktop app (on this computer, `foxglove-studio`)
or Lichtblick, which reads the same layout file:

1. Open connection, then Foxglove WebSocket, and enter `ws://localhost:8799`.
2. Import the layout `config/foxglove/chula-memory-layout.json` (Layouts,
   then Import from file).

The layout has three panels:

| Panel | What it shows |
| --- | --- |
| Map, robot and objects (3D, display frame `map`) | `/map`, the robot model from `/robot_description`, Nav2's `/plan`, the visualizer's markers, and walkie-database's `/walkie_database/objects/markers` and `/walkie_database/objects/cloud` once its memory node runs |
| Detections (annotated) | `/walkie_vision/sim/debug/objects_image/compressed` |
| Logs | `/rosout`, at Info and above |

The layout does not subscribe to the raw camera image, which is about 0.7 MB
a frame in the simulation. The 3D panel needs the whole TF chain from `map` to
`zed_head_left_camera_frame_optical` to place the visualizer's markers, which
are in the camera's optical frame.

- The bridge listens on 127.0.0.1 only. To watch from another computer,
  forward the port: `ssh -L 8799:localhost:8799 <this computer>`.
- walkie-ros's environment has no `walkie_vision_msgs` and no
  `rai_interfaces`. The bridge cannot decode `/walkie_vision/objects`,
  `/walkie_vision/sim/objects`, `/walkie_vision/people` or `/to_human`, so it
  leaves them out and logs `Failed to load schemaDefinition` for each, again
  whenever the graph changes. That is expected, and it is why the visualizer
  publishes standard types.
- foxglove_bridge 3 speaks the `foxglove.sdk.v1` WebSocket subprotocol. A
  client that offers only the older `foxglove.websocket.v1` gets HTTP 400 and
  the bridge logs `handshake failed`.

## Stop it

```bash
scripts/chula-sim.sh down
```

`down` stops the parts in reverse order: the agent first, the object memory
before `vision`, and the commander and MoveIt before `bringup`. It sends each process of a part SIGINT,
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

The object memory's store, `SIM_MEMORY_DB`, is not under `SIM_DIR` either, so
`down` and a new `up` keep what it has merged. `scripts/chula-sim.sh
reset-memory` restores it from `SIM_MEMORY_SEED`, which drops what the last
run merged; copy the store first to keep it.

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
- With scripts (T18 to T35), on 2026-11-08 at commit 4643811: `up --agent --arms --
  --tool-set hri --people-memory mock --scripts`, with the function store at its
  default, `src/walkie_agent/tools/` of the checkout, which held no function. Four
  tasks with the real model, and two verdicts on `/walkie/verdict`:
    - "Go to the sofa, then the table, then come back to the living room." The
      model did not use `run_script`. It called `go_to` three times, with todo
      updates between them, in 10.3 s, 27.9 s and 32.0 s. A `true` verdict got
      "It proved no saved function, so nothing was verified."
    - "Use one script for this. Visit the tv and then the sofa, and at each one
      say which place you reached. Save the step that goes to a place and
      announces it as a function, so you can reuse it later." One `run_script`
      call of 37.4 s defined `visit_and_announce(place)`, called it twice (four
      calls, each `ok`), and saved it to `temporary/`. A `true` verdict verified
      it, and `bundled/visit_and_announce.py` was byte for byte the file in
      `temporary/`. The ledger held a `saved` and a `verified` event for the
      same sha, and `git status` showed nothing.
    - "Please visit the bar and then the entrance, announcing each place as you
      reach it." The model called the saved `visit_and_announce` without
      defining it again. The bar leg succeeded, and `go_to("entrance")` timed
      out at 120 s, so `run_script` raised `SKILL_FAILED` after 185.2 s, with
      the traceback and `calls: go_to: ok, say: ok, go_to: TIMEOUT`. The ledger
      gained a `fail` event. The model's next call then failed with `[Errno -3]
      Temporary failure in name resolution`, and the model client makes no
      retry (`max_retries=0` in `chat_models.py`), so the task ended there.
    - "Use one script: go to the kitchen room, then the shelf, then the tv,
      announcing each place with your saved function." The model's first
      `update_todos` tried to close task 3's steps, which belong to that task
      only, so it failed and the `say` beside it was not run; the next message
      made a fresh list. A bare "stop" 12.6 s into the script cancelled the
      Nav2 goal and sent the arm commander's stop, and the agent said "Okay,
      stopping." in the same second. `run_script` raised `CANCELLED: A stop
      ended the script`, with `calls: go_to: CANCELLED`, so the shelf and tv
      legs never started. The ledger counted the stop as a second `fail` of
      `visit_and_announce`, which T36 then fixed.

    `down` left no record. The saved function stayed in the store, ignored by git.
- With object perception (S4), on 2026-11-09 at commit 9e8a9b4: `up --agent --
  --tool-set hri`, then two tasks with the real model against the oracle:
    - "What do you see in front of you?" `find_people()` returned both guests,
      and `find_objects()` returned `sofa_11` and `table_12` (furniture) and
      `mug_13` (dish), all in the living room, in 0.07 s. The agent said "I see
      a standing person raising their left arm and a sitting person. In front
      of me there is a sofa, a table and a mug."
    - "Is there an apple here?" `find_objects("apple")` answered `UNAVAILABLE:
      The camera is not looking for 'apple' right now. It looks only for: chair,
      sofa, armchair, stool, table, mug.` The agent said it could not look for
      an apple, not that none was there.

    Then walkie-vision 64811fe's YOLOE-26s stood in for the oracle's objects:
    `pixi run vision task:=memory sim:=true` in its own worktree, on the same
    graph, publishing on `/walkie_vision/sim/objects` at 3.8 Hz on the laptop's
    RTX 2060, and the agent restarted with `--objects-topic
    /walkie_vision/sim/objects`. From the living room place it published only a
    `cup` at score 0.50, the mug. After `/walkie/reset`, "What do you see in
    front of you? Is there a sofa?" got `find_objects()` with
    `cup_3158345780`, and the answer "I see one cup, in the living room. There's
    no sofa in what my camera picks up right now." YOLOE on the saved frame at
    a 0.02 threshold never gave "sofa" or "couch"; it called the sofa a "coffee
    table" at 0.26, or a "chair" at 0.43 at 1280 pixels. At five more places,
    as direct Nav2 goals:

    | Place | Nav2 | YOLOE published | Seen but missed or mislabelled |
    | --- | --- | --- | --- |
    | table | no result after 180 s, 0.92 m off | cup 0.80, bottle 0.64, apple 0.52, bottle 0.49, cup 0.34 | the table itself |
    | bar | SUCCEEDED, 106.9 s | bottle 0.85, table 0.66 (the bar) | the red plate (no such class) |
    | shelf | SUCCEEDED, 25.2 s | chair 0.37 (the shelf), apple 0.51 | the chips and the Pringles (no such class) |
    | tv | SUCCEEDED, 74.4 s | table 0.36 (the TV stand) | none |
    | sofa | SUCCEEDED, 21.6 s | table 0.33 (the sofa, close up) | none |

    So in MuJoCo's flat-shaded renders YOLOE finds the small textured objects
    and mislabels or misses the furniture (decision S4-8). The `table` goal had
    reached its place in 54.2 s on 2026-11-06; this time the vision node shared
    the CPUs. `down` left no record.
- With vision and Foxglove, on 2026-11-12, with walkie-vision at ad7c833 in
  `/var/tmp/wv-prep`: `up --vision --foxglove`. The models loaded in about
  18 s. A client on the bridge, offering the `foxglove.sdk.v1` subprotocol
  as the desktop app does, saw 141 channels and received over 15 s:

    | Topic | Messages | Rate | Mean size |
    | --- | --- | --- | --- |
    | `/walkie_vision/sim/debug/objects_markers` | 51 | 3.4 Hz | 0.3 KiB |
    | `/walkie_vision/sim/debug/objects_image/compressed` | 51 | 3.4 Hz | 11.3 KiB |
    | `/tf` | 1600 | 107 Hz | 0.4 KiB |
    | `/rosout` | 61 | 4.1 Hz | 0.3 KiB |
    | `/map` | 1 | latched | 79.7 KiB |
    | `/robot_description` | 1 | latched | 60.0 KiB |

    The debug visualizer drew 796 arrays with one missing its image.
    `down` stopped every part in about 23 s. Both vision processes finished
    cleanly, within about 4 s of `down` starting (decision SG-11).
- With the object memory, later on 2026-11-12, with walkie-database's
  uncommitted `feat/scene-graph-prep` work in `/var/tmp/wd-prep` and the store
  freshly seeded, holding no objects: `up --vision --memory --foxglove`. The
  memory logged `The object memory starts with no objects.` within a second.
  The bridge advertised `/walkie_database/objects/markers` and
  `/walkie_database/objects/cloud`, about one message per merge period
  (0.45 to 0.5 Hz of wall time). Nav2 goals to the kitchen room, the shelf and the bar succeeded; the goal to
  the TV did not finish within 200 s. By then the store held 20 objects (4
  cups, 4 chairs, 4 bottles, 2 tables, 2 sofas, 2 apples, a shelf and a
  cabinet), 1839 observations and 3 `on` relations, and a marker array was
  about 10.5 KiB. Over the run, the memory could not place 12 arrays in `map`
  for want of a transform, and dropped them. `down` stopped every part in
  about 17 s. The memory stopped 0.8 s after its SIGINT, with no traceback,
  and released its store lock. The store then passed `PRAGMA integrity_check`
  and held the same 20 objects and 3 relations, and 1916 observations.
- With the object tools, late on 2026-11-12, with walkie-vision at ad7c833
  and walkie-database at 92c4b19, the store restored from
  `/var/tmp/scene-graph-demo/seed.db` first: `up --vision --memory
  --foxglove`, a tour of Nav2 goals (table, shelf, bar and living room
  succeeded; sofa had no result after 180 s; entrance aborted), then
  `up --agent --vision --memory --object-tools --foxglove` beside it, on the
  real model with the `default` tool set. Each request went on `/from_human`:
    - "Where did you see a cup?" `recall_object("cup")` answered with
      `complete: true` and 7 objects, ids, rooms, nearest places and ages
      included, but 3 of them were bottles: `ObjectSummary.labels` holds every
      label the detector gave, and YOLOE had once called each bottle a cup.
      The model answered in plain text, which goes to `/walkie/console`, not
      through `say`.
    - "What is in the living room?" `list_room_objects("living room")`
      answered with 7 objects (3 chairs, a sofa, a cabinet and 2 cups),
      `complete: true`.
    - "Take me to a cup in the kitchen." The model added two todos, then
      `go_to_object("26")` drove to `bar` in 98 s and turned 124 degrees left
      to face the cup. `find_objects("cup")`, on the detector's topic, found a
      cup at `bar`, and the agent said "Here we are at the bar. The cup is
      right in front of me, still on the bar."
    - The agent restarted with cups ranked before alias-only matches
      (SG-17). "Where did you see a cup? Please tell me out loud." got 5 cups
      first, then the 3 bottles, with `on` read as `table 51` and `chair 45`
      from the memory's perceived anchors. The model said "I last saw a cup in
      the kitchen room, on table 51 near the bar, just a moment ago."

    The memory could not place 27 arrays in `map` for want of a transform.
    `down` stopped every part. Neither the memory nor the agent logged a
    traceback. The store passed `PRAGMA integrity_check` and held 28 present
    objects (10 chairs, 5 cups, 4 bottles, 2 tables, 2 apples, a sofa, a
    shelf, a cabinet, a bowl and a banana), 7 merged ones, 9 relations and
    7225 observations.
- With the furniture seed (SG-26), on 2026-11-12 at commit 961d537, with
  walkie-vision at ad7c833 and walkie-database at 45228e6: `reset-memory`,
  then `up --agent --vision --memory --object-tools --foxglove`, on the real
  model with the `default` tool set. The store started with the seed's two
  rooms and nine pieces of furniture, and no other object. A tour of direct
  Nav2 goals came first, 180 s at most each:

    | Place | Result | Time | Stopped off the goal by |
    | --- | --- | --- | --- |
    | table | SUCCEEDED | 32.4 s | 0.098 m, +0.5° |
    | sofa | SUCCEEDED | 84.8 s | 0.100 m, -0.1° |
    | tv | SUCCEEDED | 8.3 s | 0.080 m, +0.0° |
    | living room | SUCCEEDED | 7.3 s | 0.010 m, +0.4° |
    | kitchen room | SUCCEEDED | 30.7 s | 0.061 m, -0.3° |
    | bar | SUCCEEDED | 36.8 s | 0.095 m, -0.2° |
    | shelf | SUCCEEDED | 25.4 s | 0.053 m, +0.5° |

    Then three requests on `/from_human`:
    - "Where is the mug?" `recall_object("mug")` answered no objects, with
      `complete: true`. The model asked again for "cup", and got cups 16 and
      14 on `table 3`, the kitchen table, near the bar and the shelf, last
      seen 55 s before; cup 10 on `table 2`, the coffee table, near the sofa,
      138 s before; and bottle 12, which YOLOE had called a cup in 60 of its
      207 detections. It said
      "I have not seen a mug, sorry. I did see cups: two on table 3 in the
      kitchen room, about a minute ago, and one on table 2 in the living
      room."
    - "What's in the living room?" `list_room_objects("living room")`
      answered the sofa (5), table 9 near the tv, table 2 near the sofa,
      cabinet 20 near the entrance and cup 10 on table 2, with
      `complete: true`. Table 2's age, 226 s, was the seed's own stamp: the
      detector never saw the coffee table (SG-24). The model then ended its
      turn with an empty message, no text and no call, so the person got no
      answer, since the Runner publishes nothing for an empty reply. Decision
      R7 in [Mahoraga Demo Decisions](../architecture/mahoraga-demo-decisions.md)
      now asks the model for an answer once in that case.
    - "Where did you see a cup? Please tell me out loud." The model called no
      tool. It answered from the conversation with two `say` calls: the cups,
      and then the living room list it had not given before.

    `down` stopped every part in about 18 s. No part logged a traceback, and
    the memory could not place 3 arrays in `map` for want of a transform. The
    store passed `PRAGMA integrity_check` and held 21 present objects (5
    chairs, 4 tables, 4 bottles, 3 cups, 2 apples, a shelf, a sofa and a
    cabinet), 3 bottles merged into others, 12 relations and 2223
    observations. The nine seeded anchors kept their IDs. The detector added
    three anchors of its own: chair 15, a box of 0.78 m by 0.36 m whose
    centre is 0.27 m from the seeded `stool_3` (chair 8), which that anchor
    did not absorb; chair 19, a box of 0.89 m by 0.04 m in the kitchen; and cabinet
    20, a box of 0.64 m by 0.03 m by the entrance, where SG-14 found a wall or
    a door. 9 of the 12 relations belong to present objects: 5 on the kitchen
    table, 2 on the shelf, and one each on the coffee table and the bar
    (`table 1`).
- With official names (SG-27), on 2026-11-12, with walkie-vision at ad7c833
  and walkie-database at 6229e70, whose memory node is the same as at the
  pinned 45228e6. Two runs, each after `reset-memory` from the seed rebuilt
  from ca52814 (sha256 `8ed5b209…`), then
  `up --agent --vision --memory --object-tools`, on the real model with the
  `default` tool set. The first ran the agent's name change only (`name` and
  `on`); the second ran the whole SG-27 change that records these runs, with
  `label` and the camera's word in `go_to_object`.
    - The first toured the same seven places; Nav2 reached each in 7.7 to
      136.0 s, within 0.10 m and 0.5° of the goal. "Where is the cup?"
      `recall_object("cup")` gave cups 29 and 26 `on` `kitchen table 3`, cup
      25 `on` `coffee table 2`, and bottle 28. The model said "Two of them
      were on the kitchen table 3 in the kitchen room ... One was on the
      coffee table 2 in the living room". "What's in the living room?"
      `list_room_objects` gave the sofa, the tv desk, the coffee table,
      cabinets 33 and 34 near the entrance, and cup 25 on `coffee table 2`.
      The model listed them all, the two cabinets included. Cabinet 33, 0.63 m
      by 0.03 m, matches the entrance door's leaf in size and place, as the
      walkie-ros session found cabinet 20 of the run before does. Cabinet 34,
      0.14 m by 0.03 m from 4 detections, lies on the living-room face of the
      north-east return wall (`chula_north_east_return`), along its line.
      "Where is the coffee table?" was answered from that list, without a
      call. The detector also made chair 32 in the kitchen room, 1.79 m by
      0.16 m by 0.79 m at 50°, whose centre is 0.05 m from that of the
      diagonal wall (`chula_west_diagonal`, 1.84 m by 0.08 m by 0.80 m at
      48.5°): the wall, at twice its thickness.
    - The second started a new agent, with no tour. "Where is the coffee
      table?" `recall_object("coffee table")` gave the coffee table 2 near
      the sofa. "Take me to a stool, and check that it is still there."
      `recall_object("stool")` gave stools 6, 7 and 8 in the kitchen room.
      `go_to_object("6")` drove to `table` in 46.3 s, turned 50° right to
      face the stool, and ended its reply with `The camera calls it "chair",
      so call find_objects("chair").` The model called
      `find_objects("chair")`, which saw two chairs, then `find_objects()`,
      and said "I still see the stool, though my camera calls it a chair, so
      it is still there." The detector's one furniture anchor of its own was
      chair 46, 0.83 m by 0.31 m at 52°, over the diagonal wall's south-west
      half.

    After the second run, `down` stopped every part in about 17 s. As they
    stopped, the oracle logged a traceback, an `RCLError` from
    `rclpy.try_shutdown()` because the context was already shut down, likely
    the race that SG-11 fixed in walkie-vision's nodes; speech logged the same
    error without a traceback, and the agent an unretrieved `cannot use
    Destroyable` exception. The first run's logs were not kept.
- With walkie-database's thin-slab gate, on 2026-11-12, with walkie-vision at
  ad7c833 and the memory node at walkie-database 6d44ec9, whose messages are
  those of the pinned 45228e6. 6d44ec9 stores an anchor the memory makes only
  once at least three of its detections had 50 points or more after
  downsampling to 0.025 m, and not while more than half of all such
  detections span less than 0.05 m across (walkie-database's ADR-0029). One
  run after `reset-memory` from the same seed, then
  `up --agent --vision --memory --object-tools`, on the real model with the
  `default` tool set. The store saved at `down` is kept as
  `/var/tmp/scene-graph-demo/sim/walkie.run-2026-11-12-1020.db`.
    - The same seven-place tour. Nav2 reached five places; `table` timed out
      after 180 s, 1.05 m from its goal, and `sofa` aborted after 58 s, 0.13 m
      from it, the controller logging "Failed to make progress". "What is in
      the living room?" `list_room_objects` gave the tv desk, the sofa, the
      coffee table and cup 48 `on` `coffee table 2`, and no cabinet. "What is
      in the kitchen room?" gave the shelf, the bar, the kitchen table, stools
      6, 7 and 8, and nine objects on the bar, the shelf and the kitchen
      table, and no chair of the detector's own.
    - The tour again, this time with every place reached, recorded as a bag
      of what the memory reads: the detector's arrays, `/tf`, `/tf_static` and
      `/clock`, 766 arrays in 302 s, in
      `/var/tmp/scene-graph-demo/bags/live-slab-2026-11-12`. The saved store
      holds one anchor of the detector's own: chair 58, 0.61 m by 0.39 m by
      0.77 m at 60°, from 9 detections. Its centre is 0.15 m from the
      diagonal wall's footprint and 0.34 m from the kitchen's west wall, near
      the corner where the two meet; the next nearest things, the kitchen
      table's top and stool 6's seat, are 0.84 m from it. Its detections came
      in 5 arrays, from a camera near (1.6, -4.15). Six are centred in the
      corner. The other three, each in the same array as one of those, are
      centred on stool 6's seat, which the chair's box does not reach.
    - walkie-database's `memory-replay` of that bag, from the same seed and
      with its truth file, confirms the same chair, at the same centre and
      from 9 detections in 5 arrays. With the gate off
      (`--set slab_thickness=0`) it confirms instead a chair over the
      diagonal wall, from 20 detections, and a cabinet on the entrance door's
      leaf, from 11. The other 17 confirmed tracks are the same in both
      replays. Both hold the 9 seeded pieces of furniture, the coffee table
      without a detection, find 8 of the 12 objects, and state 8 `on`
      relations: 7 of the 9 the replay's score expects, all right (neither
      red plate is found), and an apple on a shelf board below the shelf's
      top, which the score does not count, since it expects `on` only on
      furniture tops. So the gate held back the door's 11 detections and 11
      of the wall's 20, but not the 9 that made the chair. The walkie-database
      session traced those 9 detections. Three were thin faces of the
      diagonal wall. Three were detections of the wall's south-west end that
      also took in stool 6's seat, and two of those also the floor in front
      of the wall. These three spanned 0.27 to 0.45 m across, so none was
      thin. The other three, of the seat alone, matched stool 6 and the
      chair's track equally well, and the memory gave each tie to the track
      made later, the chair's. Only 3 of the 9 were thin, so the memory
      stored the chair.

    `down` took 23 s. The oracle logged a traceback again, this time a
    `KeyboardInterrupt` raised inside its cleanup, in `oracle.destroy()`: a
    second interrupt arrived while it stopped, the case walkie-vision's
    ad7c833 handles in its own nodes. The memory node logged 12 Zenoh
    "Received Data for unknown expr_id" errors as the parts stopped. Over the
    run, the memory could not place at least 52 arrays in `map` for want of a
    transform at their stamps, and dropped them.
- With walkie-database's fix for chair 58, on 2026-11-12, with walkie-vision
  at ad7c833 and the memory node at walkie-database 41dae92, whose messages
  are those of the pinned 45228e6. Two walkie-database commits since 6d44ec9
  matter here. From 4d62bf3, the gate counts and measures each detection of
  an anchor the memory makes only on the detection's points at least 0.05 m
  above the floor. When one cluster of those points, a group lying close
  together, holds at least 80% of them, the gate keeps only that cluster and
  the clusters within 0.5 m of it. From 41dae92, a detection that matches two
  tracks equally well goes to the seeded one if either is seeded, and
  otherwise to the stored one if either is stored (walkie-database's
  ADR-0029). One run after `reset-memory` from the same seed, then
  `up --agent --vision --memory --object-tools`, on the real model with the
  `default` tool set. The store saved at `down` is kept as
  `/var/tmp/scene-graph-demo/sim/walkie.run-2026-11-12-1120.db`.
    - The same seven-place tour, twice, recorded as a bag of what the memory
      reads, from 38 s after the memory node started: 2495 arrays in 922 s
      of wall-clock time, stamped from 37.5 to 556.5 s of simulation time,
      in `/var/tmp/scene-graph-demo/bags/live-fix-2026-11-12`. Nav2 reached
      every place except `sofa` on the second tour, where it timed out after
      180 s, 0.12 m from the goal.
    - "What is in the kitchen room?" `list_room_objects` gave the seeded
      shelf, bar, kitchen table and stools 6, 7 and 8, nine objects on the
      shelf, the bar and the kitchen table, and two anchors of the
      detector's own, cabinet 68 and shelf 70. The spoken answer named a
      cabinet. Both are second copies of seeded pieces: cabinet 68 lies over
      the shelf, and shelf 70 over the west part of the bar's back panel. A shelf and a chair the memory also made there merged into
      cabinet 68. "What is in the living room?" then gave the sofa, the tv
      desk, the coffee table and cup 59 `on` `coffee table 2`, and nothing
      else.
    - The memory merges an anchor it made into a seeded piece when at least
      80% of its points lie inside that piece's box grown by 0.10 m
      (walkie-database's ADR-0028), or when the two share their main label,
      at least half of its points lie inside that grown box and, where both
      carry image vectors, the two look alike. Cabinet 68 and shelf 70 do
      neither. Cabinet 68's main label, `cabinet`, is not the seeded shelf's
      `shelf`, and shelf 70's, `shelf`, is not the bar's `table`. Only 74% of
      cabinet 68's points lie inside the shelf's grown box, and 52% of shelf
      70's inside the bar's. Their other points lie on no surface the camera
      could see. Shelf 70's lie past the bar's west end, all but 14 of them
      behind the kitchen's west wall, outside the room. Cabinet 68's lie
      inside the south window wall below its sill or up to 0.06 m behind its
      glass, 0.08 to 0.11 m behind the south-east wall's room face, and 0.10
      to 0.25 m west of the shelf's west end, in open air.
    - Shelf 70's other points, and cabinet 68's west of the shelf, are most
      likely points of the bar and the shelf themselves, placed too far west
      by an error in the robot's estimate of its own pose in the map. The
      bag holds no true pose, so the error is inferred from how the detected
      points fit the scene's geometry. From about 186 s of simulation time,
      points landed west of the surfaces they belong to. By 190 s they lay
      about 0.1 m west, and the pose error was still mostly one of position
      rather than heading. Between 197 and 199 s, partway through a
      clockwise turn from 194.5 to 202.6 s, the shift at the shelf's range
      of about 3 m grew to about 0.2 m. From 199 to 217 s,
      points lay west by 0.10 to 0.14 m at 0.7 m range and by about 0.2 m
      at 3 m. Fits to the detections from 203.6 to 214.3 s put the error at
      roughly 1 to 3 degrees of heading and up to about 0.15 m of position;
      the fits cannot separate the two well. All 56 of shelf 70's
      detections and 44 of cabinet 68's 71 were made from 199 to 215 s.
      Moved back by any of the fitted corrections, every point of shelf
      70's detections falls inside the bar's grown box, and 99% of the
      points of cabinet 68's detections from 199 to 215 s fall inside the
      shelf's. In walkie-database's `memory-replay` of the bag at 41dae92
      (below), the detections of cabinet 68's counterpart from 199 to 215 s
      have a median of 28% of their points inside the shelf's grown box,
      against 82% for those from 215 to 300 s and 96% for later ones.
    - Two known weaknesses of this simulation's localization meet here.
      MuJoCo's wheel odometry over-reports turns by about 2 to 3% against
      the true pose (walkie-ros's `docs/guides/simulation-mujoco.md`). In
      this bag AMCL's correction mostly turns back against the odometry's
      turns, by about 2 to 6 degrees for every 100 degrees of heading change
      the odometry reports, depending on the stretch of the run. AMCL
      itself is 0.1 to 0.2 m off in this room, because its map has no
      furniture and lidar hits on the furniture pull its estimate
      (walkie-ros's `docs/guides/simulation-chula-pat14.md`, on its branch
      `feat/chula-pat14-6-simulation`, not on main). AMCL dates each
      transform about 1 s after the scan it comes from. Its 57 updates
      stamped from 185.6 to 204.6 s, made from scans taken while the robot
      set off, turned and slowed, did not remove the error. The run's
      generated `nav2.yaml` lets AMCL update only after the robot has moved
      0.05 m or turned 0.05 rad. The robot moved less than that from about
      203.6 s and stood still from 207.0 to 214.4 s. So AMCL's correction
      stayed the same from its update stamped 204.6 s to the next, stamped
      215.8 s. Those two come from the scans at about 203.6 s and 214.8 s,
      the second just after the robot drove on at 214.4 s. The error then
      faded and was gone by about 233 s. Points also lay 0.1 to 0.2 m west
      earlier in the run, at about 44 to 66 s and 97 to 100 s, and about
      0.2 m to the north-west at 110 to 111 s.
    - A time offset between the camera images and the robot's pose does not
      explain the shift. From 207 to 214 s nothing moved: the base was
      still, the camera fixed relative to it, and from 208 s successive
      images gave nearly identical points, yet the points lay 0.1 to 0.2 m
      west of where they belong. The images also stop and start changing
      within one image (0.2 s) of when the robot's transforms do, and
      points from successive images line up best with no time offset.
    - Cabinet 68's other outside points are not explained. Those 0.08 to
      0.11 m behind the south-east wall's room face come from three
      detections at 248.1 to 248.5 s, none from 199 to 215 s. Those in the
      south window wall come mostly from the chair that merged into cabinet
      68, seen at 225.9 to 227.5 s. Why these points lie there is not
      established.
    - The bag cannot tell how much of the error came from the odometry and
      how much from AMCL, since it holds no true pose and no laser scan. No
      run is planned for this, but any bag recorded later should also record
      `/mujoco_ground_truth/odom`, the true pose, which the simulation
      already publishes. The demo leaves both copies as a known limitation
      and changes neither the memory nor the simulation's localization for
      them (decision SG-28).
    - walkie-database's `memory-replay` of that bag, from the same seed and
      with its truth file, at 6d44ec9, at 4d62bf3 and at 41dae92. At 6d44ec9
      the memory confirms two anchors that stand for no furniture: a chair in
      the corner where the diagonal wall meets the kitchen's west wall, and a
      `table` lying on the floor, each from 3 detections. All the floor
      table's points lie below 0.02 m, so from 4d62bf3 none of its
      detections is measured, and the memory never confirms it. At 4d62bf3
      and 41dae92 the memory confirms neither, and its other 21 confirmed
      tracks are the same at all three commits, the replay's counterparts of
      cabinet 68 and shelf 70 among them. All three
      replays find 10 of the 12 objects and the 9 seeded pieces, and state 10
      `on` relations: 8 of the 9 the replay's score expects, all right, and a
      can of Pringles and an apple on shelf boards below the shelf's top,
      which the score does not count.
    - The replay of the earlier bag, `live-slab-2026-11-12`, at 41dae92
      confirms no chair at the corner. The three detections of stool 6's seat
      join stool 6.

    `down` took 18 s. The oracle logged the `RCLError` traceback of the second
    official-names run again, from `rclpy.try_shutdown()`, and speech the
    same error without a traceback. Over the run, the
    memory could not place at least 30 arrays in `map` for want of a
    transform at their stamps, and dropped them.

## Not in the recipe

- **MoveIt's RViz.** `--gui` opens `bringup`'s RViz only. MoveIt runs with
  `use_rviz:=false`.
- **The people database.** `--people-memory mock` keeps the people memory in
  the agent's process. walkie-database's node, for `--people-memory ros`, and
  its sim seed are not started by the script.
- **Voice.** Speech is logged, not voiced. RAI's voice nodes on a PipeWire null
  sink are not part of it ([Talk to Walkie by voice](07-voice.md)).
