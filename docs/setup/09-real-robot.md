# Run the Agent on the Real Robot

This page is a runbook for one engineer at the robot. It goes from a map of the
room to talking to Walkie. Follow it in order.

It was written on 2026-11-19 for these checkouts on the robot's PC
(`eic@100.68.160.38`, KDE neon, RTX 3080 Ti Laptop):

| Checkout | Commit | Used for |
| --- | --- | --- |
| `~/Project/robocup_ros_ws/walkie-ros` | `fcbac5c` plus uncommitted edits | The team's bringup, lift, SLAM, Nav2, MoveIt, commander |
| `~/Project/robocup_experiments/walkie-agent-ros` | `f9b7a3f` (`feat/mahoraga-demo`) | The agent, voice and every check on this page |
| `~/Project/robocup_experiments/walkie-vision` | `ad7c833` (`feat/scene-graph-prep`) | People and objects from the head ZED |
| `~/Project/robocup_experiments/walkie-database` | `41dae92` (`feat/scene-graph-prep`) | People memory, and the optional object memory |
| `~/Project/robocup_experiments/walkie-ros-fullbody` | `a475236` (`feat/sim-bottle-pick`) | Whole-body MoveIt ([Optional C](#optional-c-the-whole-body-moveit-test) and [E](#optional-e-pick-up-a-remembered-object)), and `$MAP` |

The robot runs walkie-ros's **working tree**, not `fcbac5c`. Its uncommitted
edits change the launch files this page uses. Every walkie-ros command here was
read from that working tree. If someone changes it, check the command again.

!!! danger "There is no software emergency stop"
    walkie-ros documents none, and the agent has none. Joystick button 0
    cancels Nav2's goals, the stick steers over Nav2 only while you push it,
    and a bare `stop` cancels the agent's task, but none of them proves that
    the wheels stopped. Know where the robot's motor power switch is before you
    start, and keep it in reach. A second person on it is better.

The first run is the **basic run**: map, Nav2, vision, the people database, the
agent and voice. The arms and the object memory are optional steps after it.
Do the basic run first. The whole-body MoveIt experiment is a separate test
([Optional C](#optional-c-the-whole-body-moveit-test)).

## 1. What runs where

Every process runs on the robot's PC. Each has its own terminal.

| Terminal | Process | Folder | Basic run |
| --- | --- | --- | --- |
| — | Zenoh router (`rmw_zenohd`) | `~/Project/robocup_ros_ws/walkie-ros` | Already running |
| T1 | Lift homing (`lift_homing.launch.py`) | `~/Project/robocup_ros_ws/walkie-ros` | Yes |
| T2 | Bringup (`real_omnibot.launch.py`) | `~/Project/robocup_ros_ws/walkie-ros` | Yes |
| T3 | Joystick (`joy_control_smooth.launch.py`) | `~/Project/robocup_ros_ws/walkie-ros` | Yes |
| T4 | SLAM (`mapping_Nav2_real.launch.py`), later Nav2 (`localization_Nav2_real.launch.py`) | `~/Project/robocup_ros_ws/walkie-ros` | Yes |
| T5 | MoveIt (`moveit_only.launch.py`) | `~/Project/robocup_ros_ws/walkie-ros` | Optional (arms) |
| T6 | Arm commander (`commander.launch.py`) | `~/Project/robocup_ros_ws/walkie-ros` | Optional (arms) |
| T7 | Vision (`pixi run vision`) | `~/Project/robocup_experiments/walkie-vision` | Yes |
| T8 | Database (`db-node`, or `db-memory` for the object memory) | `~/Project/robocup_experiments/walkie-database` | Yes |
| T9 | Speech out (`voice-tts`) | `~/Project/robocup_experiments/walkie-agent-ros` | Yes |
| T10 | Speech in (`voice-asr`) | `~/Project/robocup_experiments/walkie-agent-ros` | Yes |
| T11 | Agent (`pixi run agent`) | `~/Project/robocup_experiments/walkie-agent-ros` | Yes |
| T12 | Watch the console, send text, run checks | `~/Project/robocup_experiments/walkie-agent-ros` | Yes |

Use a terminal on the robot's desktop, logged in as `eic`. Voice needs that
user's PipeWire session, and RViz needs the display. SSH as `eic` also works
once `eic` is logged in at the desktop. Over SSH, start the bringup with
`use_rviz:=false`.

### Or use our commander

Our copy of the team's walkie-commander, in
`~/Project/robocup_experiments/walkie-commander`, has a button for each
terminal above, plus Save map, the Foxglove bridge and a test task. Open
`http://100.68.160.38:8083`. The team's commander keeps port 8082.

- Every button starts disabled, so **Run Enabled** starts nothing. Click the
  buttons in group order, and wait for each step's ready line in its log.
- Its buttons run the arms, as in [Optional A](#optional-a-the-arms).
  **T2 bringup** is the team's bringup with the arms. It prints a warning and
  waits 5 s, so you can still press Stop before the arms move. Group
  **3 arms** starts MoveIt and then the arm commander, and **T11 agent** runs
  with `--arms`. For the basic run, use **T2 bringup, no arms** and
  **T11 agent, no arms** instead, and skip group 3.
- Each button exports the same variables as the terminals below. `$MAP` names
  `31_19112026_ChulaPat14_7`.
- **Save map** refuses to overwrite a map. For a new map, write the buttons
  again with a new name (below), then restart the commander.
- **T1 lift homing** has no terminal for `sudo`, so the lift node's own
  `ip link` calls fail and `can2` stays as it is, at start and at exit. Bring
  `can2` up first ([step 2](#2-before-you-start)).
- The agent buttons read `config/locations.lab.toml`, which is not in git. Until
  you write it in [step 4](#4-mark-the-places), they stop at once with exit
  code 2.
- The text-task buttons wait at most 10 s for the agent. If no agent listens,
  they send nothing and show as failed. Start the agent first, then click
  again.
- Groups **5** to **10** are the pick test
  ([Optional E](#optional-e-pick-up-a-remembered-object)). Group **0** holds
  **Stop the task**, the typed stop of [step 8](#8-stop-and-reset).
- **Foxglove bridge** only lets clients watch. It leaves out publishing,
  parameters and service calls, so a laptop on the LAN cannot drive the robot
  through it.

!!! danger "Run group, Stop All and profile load act on every button at once"
    **Run group**, in each group's header, starts every button of the group at
    the same moment, enabled or not, with no order and no ready line in
    between. Never use it on a group that starts hardware: it would start the
    lift homing and the bringup together. Click the buttons one by one. Each
    button that moves the robot by itself, such as a lift command or a task
    that drives, sits alone in its group, so its Run group starts only that
    button.

    **Stop All Running**, loading a profile under **Profiles** (it asks first)
    and stopping or restarting the commander service stop every running
    command at once, in no safe order. To shut down, press **Stop** on one card
    at a time, in the order of [step 8](#8-stop-and-reset), and wait for each
    card to end. **Stop** sends Ctrl+C to the command, then waits up to 60 s
    before it sends SIGTERM. For the agent that SIGTERM counts as a second
    Ctrl+C, which forces exit code 3 with the Nav2 goal's state unknown.

It runs as a user service, which ends at a reboot or a logout. To start it
again, or to write the buttons again:

```bash
systemctl --user stop walkie-commander-experiments    # only to write the buttons again
cd ~/Project/robocup_experiments/walkie-agent-ros
python3 scripts/robot_commander.py ../walkie-commander/walkie-commander-data --replace --map-name 31_19112026_ChulaPat14_7
R=~/Project/robocup_experiments/walkie-commander
systemd-run --user --unit=walkie-commander-experiments --working-directory=$R \
    --setenv=WALKIE_COMMANDER_PORT=8083 "--setenv=WALKIE_COMMANDER_TITLE=Walkie Commander (experiments)" \
    --property=KillMode=mixed --property=TimeoutStopSec=20 $R/.venv/bin/python main.py
```

`--day` (default: today) names the `~/walkie-data/<day>` folder of the T8
buttons. For the object memory, pass the day you built the seed, such as
`--day 2026-11-19`. The ids come from the button names, so writing the buttons
again keeps each card's output history.

### Start every terminal the same way

Choose the map's folder name once. walkie-ros names maps
`<NN>_<DDMMYYYY>_<Place>`, and the last one on the robot is `30_16102026_ChulaPat14_6`.
This page uses `31_19112026_lab`. Replace `lab` with the room's name, the same
in every terminal.

In **T1 to T6** (the team's walkie-ros):

```bash
export PATH=$HOME/.pixi/bin:$PATH
export ROS_DOMAIN_ID=23
unset ZENOH_CONFIG_OVERRIDE ZENOH_SESSION_CONFIG_URI ZENOH_ROUTER_CONFIG_URI
export MAP_NAME=31_19112026_lab
export MAP=$HOME/Project/robocup_experiments/walkie-ros-fullbody/navigation/walkie-navigation/map/$MAP_NAME
cd ~/Project/robocup_ros_ws/walkie-ros
```

In **T7 to T12**, the same lines, and then the checkout of that terminal:

```bash
export PATH=$HOME/.pixi/bin:$PATH
export ROS_DOMAIN_ID=23
unset ZENOH_CONFIG_OVERRIDE ZENOH_SESSION_CONFIG_URI ZENOH_ROUTER_CONFIG_URI
export MAP_NAME=31_19112026_lab
export MAP=$HOME/Project/robocup_experiments/walkie-ros-fullbody/navigation/walkie-navigation/map/$MAP_NAME
cd ~/Project/robocup_experiments/walkie-agent-ros    # T7: walkie-vision, T8: walkie-database
```

- **`$MAP` is in our walkie-ros checkout** (`robocup_experiments/walkie-ros-fullbody`),
  not in the team's `robocup_ros_ws`. Our maps stay out of the team's
  workspace. Nav2 and the database take the full path, so the map can live in
  any checkout.

- The `PATH` line matters only in an SSH shell. The desktop's `~/.bashrc` already
  adds pixi.
- **Every terminal needs `ROS_DOMAIN_ID=23`.** It is the team's domain
  (walkie-ros `docs/guides/middleware.md`). No pixi environment and no
  `~/.bashrc` on the robot sets it. Under Zenoh the domain is part of every
  topic's key, so a terminal without it sees nothing, and nothing reports an
  error.
- **The `unset` line matters after the whole-body test** ([Optional C](#optional-c-the-whole-body-moveit-test)).
  That test's terminals export `ZENOH_CONFIG_OVERRIDE`, which sends every
  process started there to its private router on port 7499. A process there
  is cut off from this page's graph, and `ROS_DOMAIN_ID` still reads `23`.
  New terminals are safer than reused ones.
- In the team's walkie-ros, always use **`pixi run --as-is`**. Its `pixi.toml`
  and `pixi.lock` have uncommitted edits, and a plain `pixi run` could re-solve
  or reinstall the team's environment. `--as-is` runs the environment as it is
  installed.
- In the three `robocup_experiments` checkouts, use `pixi run --frozen`. Their
  environments were installed and checked on 2026-11-19.

## 2. Before you start

Work through this list before the test window.

**Nothing else is driving the robot.** No bringup, lift, vision or database
node may run twice. This should print nothing:

```bash
ps -eo pid,args | grep -E '[r]eal_omnibot|[l]ift_homing|[m]apping_Nav2|[l]ocalization_Nav2|[w]alkie_vision|[w]alkie_database'
```

The team's walkie-commander web UI (`walkie-commander/.venv/bin/python main.py`)
runs on the robot. It is a launcher for saved commands. Make sure no teammate
starts the bringup through it while you test. The whole-body MoveIt test
([Optional C](#optional-c-the-whole-body-moveit-test)) drives the same CAN
buses from `~/Project/robocup_experiments/walkie-ros-fullbody`. Check that none
of its nodes run:

```bash
ps -eo pid,args | grep -E '[w]alkie-ros-fullbody'
```

It should print nothing, or only that test's own `rmw_zenohd`.

**The Zenoh router runs.** One router on port 7447 serves every process on the
robot. It has run from the team's walkie-ros since early November. Do not start
a second one.

```bash
ps -eo pid,args | grep [r]mw_zenohd
```

It should show `.../walkie-ros/.pixi/envs/default/lib/rmw_zenoh_cpp/rmw_zenohd`.
A router under `walkie-ros-fullbody` belongs to the whole-body test, not to
the team's graph. Leave it, and wait for that test to finish as above. If you
ran the whole-body test, close all of its terminals first. Each one exports
`ZENOH_CONFIG_OVERRIDE`.
Only if nothing is printed (for example after a reboot), start it in a terminal
of its own, set up like T1:

```bash
pixi run --as-is ros2 run rmw_zenoh_cpp rmw_zenohd
```

**The CAN adapters have the right names.** The base and the lift use `can2`,
which must be the candleLight adapter (driver `gs_usb`) at classic 1 Mbit/s.
The arms use `can0` (right) and `can1` (left), the two channels of the PCAN-USB
Pro FD (driver `peak_usb`) with CAN FD. Check the driver, the channel
(`dev_id`) and the state behind each name:

```bash
for n in can0 can1 can2; do echo "$n $(basename $(readlink -f /sys/class/net/$n/device/driver)) $(cat /sys/class/net/$n/dev_id) $(cat /sys/class/net/$n/operstate)"; done
```

It must print exactly:

```text
can0 peak_usb 0x0 up
can1 peak_usb 0x1 up
can2 gs_usb 0x0 up
```

The `dev_id` column tells the two PCAN channels, and so the two arms, apart.
The driver alone cannot. The whole-body guide runs the same check
([Optional C](#optional-c-the-whole-body-moveit-test)).

!!! danger "On 2026-11-19 the names were swapped"
    By 16:06 the same day the check printed the expected three lines again,
    with `can0` on `3-2.4:1.0`. Keep running it before every start.

    At 06:04 the robot printed `can0 gs_usb 0x0 down`, `can1 peak_usb 0x0
    down` and `can2 peak_usb 0x1 up`. With
    that, the base and the lift would talk to an arm bus, and the arms' FD bus
    would land on the candleLight. **Do not start the lift or the bringup while
    the drivers do not match.** The names come from
    `/etc/systemd/network/70-can0.link`, `70-can1.link` and `70-can2.link`,
    which are applied when an adapter is plugged in.

    The names date from the boot of 2026-10-31. The likely cause: the
    candleLight came up first as `can0`. Both PCAN channels then asked for
    `can0` (`70-can0.link` matches the PCAN's USB port, so it matches both
    channels), could not have it, and kept `can1` and `can2`. The order in
    which the adapters appear decides the names. Replugging only the PCAN
    gives the same names, since the candleLight keeps `can0`.

    The swap also explains the bus states. `can0-up.service` asks `can0` for
    CAN FD, which the candleLight refuses, so the service fails and `can0` and
    `can1` stay down. `can2-up.service` brings up a PCAN channel as `can2`. A
    `can2` that is `up` therefore proves nothing; only the driver check does.

    **Unverified fix,** with nothing running on the buses:

    1. Make sure the arms rest or hang where they cannot hit anything. Switch
       off the motor power for the arms, the wheels and the lift. Keep the
       computer on.
    2. Unplug both CAN adapters, the PCAN-USB Pro FD and the candleLight.
    3. Plug the PCAN back into **the same USB port it came out of**:
       `70-can0.link` matches that port's path, so another port gives other
       names. Wait until `ls /sys/class/net` shows `can0` and `can1`.
    4. Plug in the candleLight.
    5. Run the check above again, and check that the PCAN sits on the port the
       link file expects. This path must end in `3-2.4:1.0`:

        ```bash
        readlink -f /sys/class/net/can0/device
        ```

    6. Switch the motor power back on.

    Do not reboot to fix this: the boot of 2026-10-31 produced exactly this
    swap. If the names are still wrong, ask the team before going on.

If the buses are not up after a replug (the last column shows `down`), the systemd
units `can0-up.service` and `can2-up.service` normally bring them up. By hand:

```bash
sudo ip link set can2 up type can bitrate 1000000
sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 up type can bitrate 1000000 dbitrate 5000000 fd on
```

The first two lines are walkie-ros's `docs/guides/real-robot.md`. The `can1`
line follows `can0-up.service`, which sets both PCAN channels this way. The lift
node also takes `can2` down and up itself (see [step 3](#3-capture-the-map)).

**The camera and the gamepad are attached.** The ZED 2i is on USB. On
2026-11-19 **no gamepad was attached.** The only joystick device,
`/dev/input/js0`, was `mouce-library-fake-mouse`: RustDesk's virtual mouse.
Plug in or pair the team's gamepad first, then find its device by name:

```bash
for j in /sys/class/input/js*; do echo "/dev/input/${j##*/} $(cat $j/device/name)"; done
```

Use the gamepad's line, and note its device (such as `/dev/input/js1`) for
T3. **Never use `mouce-library-fake-mouse`:** moving the remote RustDesk
pointer would drive the robot. Without a gamepad you cannot drive the robot to
map the room, and you lose button 0, which cancels Nav2's goals.

**The GPU is free.** Before the bringup, the GPU held only the desktop (45 MiB
used of 16 GiB). Look again after the bringup, since the ZED's depth runs there
too:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

**The agent has its model key.** Since 2026-11-19 the robot's agent checkout
has a `.env` (mode 600) that sets only `OPENAI_API_KEY`, the team's OpenRouter
key. OpenRouter accepted it on that day. Do not run `cp .env.example .env`: it
would replace the key with an empty one. Check that the key is set without
printing it:

```bash
cd ~/Project/robocup_experiments/walkie-agent-ros
grep -c '^OPENAI_API_KEY=.' .env
```

It prints `1`. If it prints `0` or `.env` is missing, write the file with
`nano .env` and never paste the key into a chat. The other names in
`.env.example` (`GOOGLE_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`)
stay unset. `TAVILY_API_KEY` is needed only for `--web-search`
([Optional D](#optional-d-web-search)). The agent reads `.env` at startup. Without a key it stops at once
with one `walkie-agent:` line and exit code 2.

**The robot has internet.** The agent's model is
`deepseek/deepseek-v4.1-flash:nitro` on OpenRouter, with no local fallback. The
first `voice-asr` start also downloads Whisper `tiny` and Silero VAD, which are
not cached on the robot.

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://openrouter.ai/api/v1/models
```

`200` means the robot reaches OpenRouter. On 2026-11-19 it printed `200` on
the robot, and so did `https://huggingface.co`, where the Whisper download
comes from. The first `Please say hello to everyone.` in
[step 7](#7-talk-to-the-robot) proves the whole path.

**The microphone and speaker.** As `eic` on the desktop:

```bash
wpctl status
```

Under **Sources**, the default (`*`) should be the fifine USB microphone. On
2026-11-19 WirePlumber's saved default source was
`alsa_input.usb-3142_fifine_Microphone-00.analog-stereo`, and the default sink
the PC's own analog output.

!!! warning "Unverified"
    Whether a speaker is plugged into that analog output, and whether the
    microphone is muted, was not checked. `wpctl get-volume @DEFAULT_AUDIO_SOURCE@`
    shows `[MUTED]` when it is, and `wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0`
    unmutes it. See [Talk to Walkie by Voice](07-voice.md#step-2-configure).

**Battery.** No battery topic or check was found in the code read for this
page. Check the charge the way the team usually does.

## 3. Capture the map

You drive the robot by joystick while slam_toolbox builds the map.

### Lift homing (T1)

```bash
sudo -v
pixi run --as-is ros2 launch robot_bringup lift_homing.launch.py
```

- Wait for `Homing Successful. Origin set.`
- **The lift moves as soon as this starts,** and its homing search has no
  timeout. Keep a hand near the motor power.
- `sudo -v` caches your password. The lift node runs
  `sudo /sbin/ip link set can2 down` and `up` itself, so without it the node
  may stop at a password prompt.
- **Keep it running for the whole test.** When it exits, it takes `can2` down
  (unless its `sudo` failed, as under our commander), and the base on the same
  bus stops working. It also publishes `lift_joint`:
  the bringup relays `lift/joint_states` to `/joint_states`, and without it
  there is no TF from the base to the camera.

### Bringup (T2)

The basic run leaves the arms out. Its agent runs without `--arms`, so nothing
in it uses them, and with them the bringup swings every arm joint to 0 rad when
it starts and drops the arms when it stops. Without them the arm motors stay
off and the arms hang where they are. The team's commander button and our
commander's **T2 bringup** start the bringup with the arms, as
[Optional A](#optional-a-the-arms) does.

```bash
pixi run --as-is ros2 launch robot_bringup real_omnibot.launch.py use_arm:=false use_fake_arm_hardware:=true
```

- This starts ros2_control (the omni base on `can2`), twist_mux, the lidars
  (`/scan` and `/unilidar/cloud/filtered`), the ZED (`zed_head`) under its
  watchdog, robot_state_publisher, rosbridge and RViz.
- `use_arm:=false` needs `use_fake_arm_hardware:=true`. Alone, it still loads
  the real arm hardware, which fails without the arms on CAN.
- Over SSH, add `use_rviz:=false`. RViz needs a display.
- With the arms (see [Optional A](#optional-a-the-arms)), drop both arm
  arguments. **The arms then move at start:** to 0 rad when the hardware
  activates, then to a home pose over 5 s. Clear the space around them first.
- The head servo is broken. Uncommitted edits in the team's checkout work
  around it: the head hardware is `mock_components/GenericSystem`, `joint_broad`
  no longer publishes `head_servo_joint`, and `head_imu_corrector` publishes it
  from the ZED's IMU instead. Head commands move nothing. Our
  `walkie-ros-fullbody` checkout does not have these edits and still loads the
  Dynamixel head driver, so run the bringup from the team's checkout.

The bringup is up after about 6 s, when these lines have appeared:

- `[zed_watchdog]: ZED streaming healthy after (re)launch`
- `[head_imu_corrector]: [IMU Pitch: ...] Motor: N/A -> Corrected Joint: ...`,
  once a second. `Motor: N/A` is expected with the mock head.

These lines are normal and do not mean a failure:

- `No CAN message received from CAN ID: <n>` once or twice at start.
- `[cloud_to_map]: TF lookup failed: "map" ... does not exist` every 2 s, until
  SLAM or Nav2 (T4) runs.
- `[ZED][WARNING] Gravity alignment issues detected. Recomputing alignment...`
  once.
- After Ctrl+C: `process has died` for `tf_server`, `head_imu_corrector`,
  `current_pose_publisher.py`, `lakibeam1_scan_node`, `zed_watchdog.py` and
  `rviz2`, and a `zed_watchdog.py` traceback ending in `RuntimeError: Unable to
  convert call argument '0'`. These nodes do not exit cleanly on Ctrl+C. The
  team's own bringup also reports `tf_server`, `head_imu_corrector`,
  `current_pose_publisher.py`, `lakibeam1_scan_node` and `zed_watchdog.py` as
  died when it stops.

Then check from T12 that the sensors publish
([step 6](#6-first-checks-before-talking)).

### Joystick (T3)

Use the gamepad's device from the name check in
[step 2](#2-before-you-start), never RustDesk's `mouce-library-fake-mouse`:

```bash
export JOY=/dev/input/js1    # the gamepad's device from step 2
pixi run --as-is ros2 launch joy_interface joy_control_smooth.launch.py dev:=$JOY
```

- The launch's default device is `/dev/input/js2`. Always pass `dev:=`.
- The stick has priority 100 in twist_mux, over Nav2's 10, but **only while
  you push it.** At rest the node sends one zero command and then nothing, and
  0.5 s later twist_mux hands the base back to Nav2. A goal that is still
  active then drives on.
- **Button 0 cancels all Nav2 goals** (`/navigate_to_pose` and
  `/navigate_through_poses`). Press it before you let go of the stick.
- **Joystick commands bypass the collision monitor.** The maximum linear speed
  is 1.0 m/s, and the turbo button (5) multiplies it by 5. Do not press turbo
  indoors.
- **Do not drive yet.** The map's origin is where the robot stands now (see
  SLAM below). Try the stick, at low speed, only after SLAM runs.

### SLAM (T4)

```bash
pixi run --as-is ros2 launch robot_navigation mapping_Nav2_real.launch.py
```

- This runs slam_toolbox (online sync, 0.05 m cells) with Nav2. It takes no map
  argument.
- slam_toolbox's `map` frame starts where the bringup (T2) started odometry.
  If the robot has not moved since T2 started, that is where it stands now.
  Note that spot and heading on the floor (tape works). Placing the robot there later starts AMCL in the
  right pose, and the optional object memory's `stand_pose` can use it.
- This launch also starts `head_tilt_near_goal`. With the mock head it moves
  nothing.

Drive slowly through the whole room with the joystick. Watch the map grow in
RViz. Close the loop by returning to where you started.

### Save the map (T12 or a new terminal set up like T1)

Save while SLAM still runs:

```bash
cd ~/Project/robocup_ros_ws/walkie-ros
mkdir -p $MAP
pixi run --as-is ros2 run nav2_map_server map_saver_cli -t /map -f $MAP/map
ls $MAP
```

`ls` should show `map.pgm` and `map.yaml`.

- Do not use the team's `pixi run save-map` task. It writes into the team's
  `robocup_ros_ws/walkie-ros/navigation/walkie-navigation/map/`, not into `$MAP`.
- Clean up `map.pgm` by hand now, if you want to. Do it before seeding the
  optional object memory: any later change to the image, its resolution or its
  origin makes the memory refuse to start.
- No `install/share` has the map, because it is outside the team's workspace.
  Always pass the full path, `$MAP`.
- A new map has no `keepout.yaml`. Nav2 therefore runs with keepout zones off
  ([step 5](#5-start-the-stack)).

Then stop SLAM with Ctrl+C in T4.

## 4. Mark the places

The agent drives only to places named in a locations file. The file in the
checkout, `config/locations.toml`, is empty on purpose. Do not use
`config/locations.sim.toml` or `config/locations.chula_pat14_6.toml`: they are
simulation places.

Start Nav2 on the new map first ([step 5](#5-start-the-stack), T4), and check
that the robot is localized. Then, for each place:

1. Drive the robot there with the joystick, facing the way it should face on
   arrival.
2. Read its pose in T12:

    ```bash
    pixi run --frozen ros2 run tf2_ros tf2_echo map base_footprint
    ```

    Copy `x` and `y` from `Translation`, and the third value of
    `Rotation: in RPY (radian)` as `yaw`. Stop it with Ctrl+C.

Use `base_footprint`, not `base_link`. It is the frame Nav2 and AMCL use
(`base_frame_id: base_footprint`), and on the robot it sits on the floor,
0.0312 m below `base_link`.

Put places on open floor. In the simulation, goals 0.41 m from a wall and
0.23 m from a keepout zone never finished. The agent's `go_to` gives up after
120 s and cancels the goal.

Write the file in the agent checkout, for example with
`nano ~/Project/robocup_experiments/walkie-agent-ros/config/locations.lab.toml`:

```toml title="config/locations.lab.toml"
# Places on map 31_19112026_lab, measured with tf2_echo map base_footprint.

[locations.lab]
x = 0.0
y = 0.0
yaw = 0.0
description = "The middle of the lab, where mapping started."
room = "lab"

[locations.door]
x = 1.23
y = -0.45
yaw = 3.14
description = "Just inside the lab door, facing into the room."
room = "lab"

[locations."coffee table"]
x = 2.0
y = -1.5
yaw = 1.57
description = "In front of the low coffee table."
room = "lab"
```

- The numbers above are examples. Replace them with what you measured.
- The key is the place's name. Write it in lowercase with no spaces at either
  end. A name with spaces needs quotes.
- `x` and `y` are required, in meters in the `map` frame. `yaw` is in radians,
  0 faces +x, and defaults to 0.
- `description` is what the LLM reads about the place. Write it for someone who
  cannot see the map.
- `room` is optional. A room is a place whose `room` is its own name, like `lab`
  above. Any other key stops the agent at startup.

Check the file without starting anything (in T12):

```bash
pixi run --frozen python -c "from walkie_agent.locations import load_locations; [print(p) for p in load_locations('config/locations.lab.toml').values()]"
```

It prints one `Location(...)` line per place, or one line naming what is wrong.

## 5. Start the stack

Start each part in order, in its own terminal, and wait for its ready line
before the next. T1 (lift), T2 (bringup) and T3 (joystick) from
[step 3](#3-capture-the-map), and Nav2 in T4 if you started it in
[step 4](#4-mark-the-places), keep running.

### Nav2 on the saved map (T4)

Skip this if Nav2 already runs and is localized from step 4.

```bash
pixi run --as-is ros2 launch robot_navigation localization_Nav2_real.launch.py map:=$MAP use_keepout_zones:=false
```

- Ready: the lifecycle manager logs `Managed nodes are active`.
- `map:=` is the **folder**. The launch appends `map.yaml` itself. The
  `map:=.../map.yaml` form in walkie-ros's `docs/guides/real-robot.md` is stale
  and makes the map server fail.
- Without `map:=`, it loads the default map `30_16102026_ChulaPat14_6` without
  an error. Make sure the log names your map.
- `use_keepout_zones:=false` is needed because a new map has no `keepout.yaml`.
- nav2_bringup's `use_composition` already defaults to `False` here, so it
  needs no argument.
- `head_tilt_near_goal` is commented out of this launch in the robot's working
  tree. It does not run.

**Set the initial pose.** AMCL starts at (0, 0, 0), the spot where mapping
started. If the robot stands there, facing the same way, it is localized.
Otherwise, in RViz:

1. Under **Global Options**, set **Fixed Frame** to `map`. The robot's RViz
   config starts with `base_link`, and AMCL ignores a pose estimate in any frame
   but `map`.
2. Click **2D Pose Estimate**, then click and drag on the map where the robot
   stands and the way it faces.
3. Check that the laser scan lines up with the map's walls.

Without RViz, place the robot back on the mapping start spot before starting
Nav2.

### Vision (T7)

```bash
cd ~/Project/robocup_experiments/walkie-vision
pixi run --frozen vision respawn_max_retries:=3
```

- This runs the `hri` task on the real ZED (`task:=hri sim:=false` are the
  defaults): people with pose, tracking, faces and appearance, and the seat
  classes chair, sofa, armchair and stool.
- It publishes `/walkie_vision/people` and `/walkie_vision/objects`.
- Ready: `loaded the models in N s`, then a line that names the image topic
  `/zed_head/zed_node/rgb/color/rect/image`.
- Every 10 s it logs `processed N in 10.0 s (X Hz)` with image ages. Aim for at
  least 5 Hz and an age p95 under 300 ms. The agent drops arrays older than 1 s.
- `cannot start: ...` is a startup failure. The launch restarts the node every
  3 s, and `respawn_max_retries:=3` stops that after three tries. Without it,
  the launch retries forever.
- Run `nvidia-smi` again once the models are loaded. The GPU memory of vision
  beside the ZED's depth has never been measured on this robot.
- Only one vision node may run on the graph. The team's own walkie-vision
  publishes the same topics. The node warns `another node also publishes on`
  when it finds one.

### Database, people memory (T8)

```bash
cd ~/Project/robocup_experiments/walkie-database
mkdir -p ~/walkie-data/2026-11-19
pixi run --frozen db-node --ros-args -p db_path:=$HOME/walkie-data/2026-11-19/people.db
```

- Ready: `Opened /home/eic/walkie-data/2026-11-19/people.db at schema version N`.
- It serves `/walkie_database/start_session`, `remember_person`, `match_faces`
  and `match_appearances`, which the agent's `--people-memory ros` uses.
- The explicit `db_path` keeps the test's people out of the checkout. Without
  it, the node uses `data/walkie.db` in the checkout.

### Speech out (T9)

```bash
pixi run --frozen voice-tts
```

- It speaks every line on `/to_human` with Kokoro, which runs locally. Its model
  is already cached on the robot (`~/.cache/rai/kokoro`).
- Start it before the agent. The agent's `say` fails with
  `Nothing was said: no speech node is listening on /to_human` when nothing
  listens.

For a silent test, run this in T9 instead. It prints each line:

```bash
pixi run --frozen ros2 topic echo /to_human rai_interfaces/msg/HRIMessage --field text
```

### Speech in (T10)

Start it after the agent works by text ([step 7](#7-talk-to-the-robot)):

```bash
pixi run --frozen voice-asr
```

### Agent (T11)

```bash
pixi run --frozen agent --locations config/locations.lab.toml --tool-set hri --vision --people-memory ros --tool-log logs/agent-tools.jsonl
```

- Ready: `Walkie's agent is running, with the hri tool set.`
- `agent` is the robot task. Never use `agent-sim` or `agent-sim-chula` here:
  they add `--use-sim-time` and simulation places.
- Do not pass `--use-sim-time` or `--objects-topic`. The defaults are the
  robot's topics.
- `--tool-set hri` gives `list_places`, `go_to`, `move_relative`, `say`,
  `find_people`, `find_objects`, `face_person`, `remember_person` and
  `recognize_person`, plus `update_todos` and `wait_for_person`, which every set
  has.
- A bad flag combination prints the usage and one `error:` line. A broken
  locations file or a missing key prints one `walkie-agent:` line. Each one
  stops the agent at once with exit code 2.
- `--tool-log` makes `logs/` in the agent checkout on the first tool call.
  `logs/` is not git-ignored. Do not commit it.
- The agent checks nothing on the graph at startup. A missing part shows up only
  when a tool needs it ([step 9](#9-when-something-goes-wrong)).

If vision or the database will not start, run the agent without them. This is
the smallest useful agent: it drives and talks only.

```bash
pixi run --frozen agent --locations config/locations.lab.toml --tool-log logs/agent-tools.jsonl
```

## 6. First checks before talking

Run these in T12 (the agent checkout). They only read the graph.

```bash
echo $ROS_DOMAIN_ID
```

It must print `23`. Check it in every terminal you opened.

```bash
grep -lz '^ZENOH_CONFIG_OVERRIDE=' /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3 | xargs -r ps -o pid,args -p
```

This lists every process that started with the whole-body test's router
setting. It must print nothing, apart from that test's own `rmw_zenohd` while
its router is up.

- A `bash` line is a terminal opened for the whole-body test. `/proc` keeps
  the environment a process started with, so the shell still shows after the
  `unset` line. Processes started after that line are fine.
- Any other line is a process connected to the wrong router. Stop it, and
  start it again from a new terminal set up as in
  [step 1](#start-every-terminal-the-same-way).

Then stop the `ros2` command-line daemon. It caches the graph, and one started
from a whole-body terminal keeps showing that test's router:

```bash
pixi run --frozen ros2 daemon stop
```

The next `ros2` command starts a fresh one from T12.

```bash
pixi run --frozen ros2 action list
```

It must list `/navigate_to_pose`. Without it, `go_to` answers
`Navigation is unavailable`.

```bash
pixi run --frozen ros2 run tf2_ros tf2_echo map base_footprint
```

It must print a transform within a few seconds, at the robot's real pose. Stop
it with Ctrl+C.

```bash
pixi run --frozen ros2 topic list | grep -E '^/scan$|^/unilidar/cloud/filtered$|^/zed_head/zed_node/rgb/color/rect/image$|^/zed_head/zed_node/depth/depth_registered$'
```

It must print all four. Nav2's collision monitor stops the robot when `/scan`
or `/unilidar/cloud/filtered` goes stale.

```bash
pixi run --frozen ros2 topic echo --once /walkie_vision/people walkie_vision_msgs/msg/PersonDetectionArray --field header
pixi run --frozen ros2 topic echo --once /walkie_vision/objects walkie_vision_msgs/msg/ObjectDetectionArray --field header
```

Each must print one header, with a stamp of the current wall-clock time.

```bash
pixi run --frozen ros2 service list | grep /walkie_database/
```

It must list `start_session`, `remember_person`, `match_faces` and
`match_appearances` under `/walkie_database/`.

Start this watcher in T12 before you send anything, and keep it open. The
console keeps no messages for a late listener.

```bash
pixi run --frozen ros2 topic echo /walkie/console rai_interfaces/msg/HRIMessage --field text
```

It shows the agent's replies, and one line per change of a todo step. Each tool
call is also one JSON line in `logs/agent-tools.jsonl`
(`tail -F logs/agent-tools.jsonl` in another terminal; `-F` waits for the
file, which the agent creates at its first tool call).

??? info "Message type hashes (checked on 2026-11-19)"
    rmw_zenoh never answers a client whose message was built from another
    definition. On 2026-11-19 the installed hashes of `walkie_vision_msgs` and
    `walkie_database_msgs` were the same in the agent, walkie-database and
    walkie-vision on the robot. Check again only after rebuilding one of them:

    ```bash
    cd ~/Project/robocup_experiments
    for r in walkie-agent-ros walkie-database walkie-vision; do echo "== $r"; for f in $r/ros2_ws/install/walkie_*_msgs/share/walkie_*_msgs/{msg,srv}/*.json; do echo "${f##*/share/} $(grep -o -m1 'RIHS01_[0-9a-f]*' $f)"; done | sort; done
    ```

    The same file must show the same hash in every checkout that has it.

## 7. Talk to the robot

Talk by text first. Voice has no wake word, so everything said in the room
becomes a task.

### By text (T12, or a new terminal set up like T12)

A first task with no motion:

```bash
pixi run --frozen ros2 topic pub -w 1 --once /from_human rai_interfaces/msg/HRIMessage "{text: 'Please say hello to everyone.'}"
```

`-w 1` waits for the agent's subscriber before it publishes. Expect a spoken
(or printed) hello in T9, and a reply on the console. This also proves the model
key and the internet.

Then one short drive, to a place 1 m to 2 m away on open floor. Keep your hand
on the joystick:

```bash
pixi run --frozen ros2 topic pub -w 1 --once /from_human rai_interfaces/msg/HRIMessage "{text: 'Go to the door.'}"
```

A task with its steps seeded, on `/walkie/task`:

```bash
pixi run --frozen ros2 topic pub -w 1 --once /walkie/task std_msgs/msg/String "{data: '{\"task\": \"Go to the door and say hello.\", \"todos\": [\"Go to the door\", \"Say hello\"]}'}"
```

- The JSON has `task` and exactly one of `todos` or `template`. No templates ship,
  so use `todos`.
- A malformed message is dropped, and the agent logs
  `Ignored a message on /walkie/task: ...`.
- A message sent during a task waits in the queue. Only a bare stop or a reset
  interrupts ([step 8](#8-stop-and-reset)).

To watch the whole todo list:

```bash
pixi run --frozen ros2 topic echo /walkie/todos std_msgs/msg/String --field data
```

### By voice (T10)

Start `voice-asr` once the text tasks work:

```bash
pixi run --frozen voice-asr
```

- It listens on the default microphone, the fifine USB microphone
  ([step 2](#2-before-you-start)), and sends what it hears to `/from_human`.
- The first start downloads Whisper `tiny` and Silero VAD, which needs internet.
  Start it well before the window. PyTorch warns that it is about to run code
  from an untrusted repository. That is Silero VAD from GitHub.
- It prints `starting recording...` when it hears you, then what it understood.
- Speak, then stop for about a second. One second of silence ends a sentence.
- Walkie does not listen while it speaks. Wait until it finishes.
- Say for example _"Please say hello to everyone."_, then _"Go to the door."_

??? warning "Unverified: Whisper small from the cache"
    The robot's Hugging Face cache already holds `Systran/faster-whisper-small`.
    In `config.toml`, `[asr]` with `transcription_model = "FasterWhisper"` and
    `transcription_model_name = "small"` should use it instead of downloading
    Whisper `tiny`. This was not tried. Silero VAD still downloads on the first
    start, and the edit changes a tracked file.

## Optional A: the arms

Do this only after the basic run works. Arm poses go from the agent's
`config/arm_poses.toml` to walkie-ros's `bimanual_commander`. Receive and carry
were never tried on the robot.

The arms come up with the bringup, so adding them restarts the bringup and
Nav2. The lift (T1) keeps running.

1. Stop the agent (Ctrl+C in T11), then Nav2 (T4), then the bringup (T2).
2. Start the bringup in T2 without the arm arguments. **The arms move at
   start:** the arm driver commands every joint to 0 rad at once. Clear the
   space around them, and keep hands away from the grippers.

    ```bash
    pixi run --as-is ros2 launch robot_bringup real_omnibot.launch.py
    ```

3. Start Nav2 in T4 again and set the initial pose, as in
   [step 5](#nav2-on-the-saved-map-t4).
4. MoveIt in T5. Ready: `MoveGroup context initialization complete`.

    ```bash
    pixi run --as-is ros2 launch openarm_bimanual_moveit_config moveit_only.launch.py hardware_type:=real_robot use_rviz:=false
    ```

5. The arm commander in T6, after MoveIt is ready. Ready:
   `Bimanual Commander Node Initialized`.

    ```bash
    pixi run --as-is ros2 launch openarm_bimanual_commander_cpp commander.launch.py
    ```

6. The agent in T11 with `--arms`:

    ```bash
    pixi run --frozen agent --locations config/locations.lab.toml --tool-set hri --vision --people-memory ros --arms --tool-log logs/agent-tools.jsonl
    ```

Without `--use-sim-time`, the agent offers no pose marked
`tried_on_robot = false` in `config/arm_poses.toml`. With an arm out, `go_to` and
`move_relative` refuse to drive.

**When the bringup stops, the arms fall.** The arm driver turns the arm motors
off, and nothing holds the arms up. How far they drop has not been measured.
The agent does not lower the arms when it exits, and `hands_up` or `standby`
leave the elbows raised. So before stopping T11, ask the agent to put both
arms at `home`, and wait until they are still. Before stopping T2, clear the
space below the arms.

## Optional B: the object memory

Do this only after the basic run works. It adds `recall_object`,
`list_room_objects` and `go_to_object`. The seed and restore steps below were
run on 2026-11-19 against walkie-database 41dae92 with a map from map_saver.
Real camera data has never reached the memory, so everything after the node
starts is untested on the robot.

The memory needs:

- the map folder `$MAP` from [step 3](#3-capture-the-map), final, with Nav2
  localized on it;
- the lift node running, since `lift_joint` is part of the camera's TF;
- vision's `memory` task instead of `hri`;
- the database node in memory mode instead of `db-node`.

### a. Get the map ID and the map's bounds (a new terminal set up like T8)

Leave `db-node` running in T8 until step e.

```bash
cd ~/Project/robocup_experiments/walkie-database
pixi run --frozen python -c "from walkie_database.map_config import digest_map_files; from walkie_map_id import make_map_id; print(make_map_id('lab', '2026-11-19', digest_map_files('$MAP/map.yaml')))"
```

It prints the map ID, such as `lab-2026-11-19-714b087d`. The first part,
`lab`, is the arena name, and must match `--arena` below.

The map's corners in the `map` frame, for a room covering the whole map:

```bash
pixi run --frozen python - "$MAP" <<'EOF'
import math, pathlib, sys, yaml
folder = pathlib.Path(sys.argv[1])
meta = yaml.safe_load((folder / "map.yaml").read_text())
data = (folder / meta["image"]).read_bytes()
tokens, i = [], 0
while len(tokens) < 3:
    while data[i:i + 1].isspace():
        i += 1
    if data[i:i + 1] == b"#":
        i = data.index(b"\n", i)
        continue
    j = i
    while not data[j:j + 1].isspace():
        j += 1
    tokens.append(data[i:j].decode())
    i = j
w, h = int(tokens[1]) * meta["resolution"], int(tokens[2]) * meta["resolution"]
ox, oy, yaw = meta["origin"]
c, s = math.cos(yaw), math.sin(yaw)
print([[round(ox + c * x - s * y, 3), round(oy + s * x + c * y, 3)] for x, y in ((0, 0), (w, 0), (w, h), (0, h))])
EOF
```

It prints four corners, such as
`[[-4.698, -7.82], [3.702, -7.82], [3.702, 16.43], [-4.698, 16.43]]`.

### b. Write the seed's three files

In `~/walkie-data/2026-11-19/`. Put the map ID and the corners in:

```yaml title="~/walkie-data/2026-11-19/rooms.yaml"
schema_version: 1
map_id: lab-2026-11-19-XXXXXXXX
rooms:
  - key: lab
    name: lab
    description: The whole mapped area.
    floor: 0
    polygon: [[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]]
    stand_pose: [0.0, 0.0, 0.0]
```

```yaml title="~/walkie-data/2026-11-19/anchors.yaml"
schema_version: 1
map_id: lab-2026-11-19-XXXXXXXX
baseline_rev: ""
anchors: []
```

```yaml title="~/walkie-data/2026-11-19/catalogue.yaml"
schema_version: 1
objects: []
```

- `stand_pose` `[x, y, yaw]` must lie inside the polygon, on free floor.
  `[0.0, 0.0, 0.0]` is the mapping start.
- The room's `name` is what `list_room_objects` matches. Make it the name you
  will say, and the same as the room in the locations file.
- Keys match `[a-z][a-z0-9_]*`. Numbers are plain decimals (`1e3` is refused).

### c. Build the seed and the store (the same terminal as step a)

```bash
cd ~/Project/robocup_experiments/walkie-database
pixi run --frozen db-seed-store --map-folder $MAP --arena lab --date 2026-11-19 \
    --rooms ~/walkie-data/2026-11-19/rooms.yaml --anchors ~/walkie-data/2026-11-19/anchors.yaml \
    --catalogue ~/walkie-data/2026-11-19/catalogue.yaml --clock-kind wall \
    --output ~/walkie-data/2026-11-19/seed.db
pixi run --frozen db-restore --seed ~/walkie-data/2026-11-19/seed.db --db ~/walkie-data/2026-11-19/memory.db \
    --map-folder $MAP --clock-kind wall --replace
```

- The seed step prints `Wrote ...: map <id>, 1 rooms, 0 anchors, ...`. A bad
  input prints `Refused, and wrote nothing:` with the reason. A wrong map ID in
  the files is refused with a message that names the right one.
- The restore step prints `Restored ...`. Run it only with no node on that
  store.
- **To seed again after the map changed,** write new files, such as
  `--output ~/walkie-data/2026-11-19/seed-2.db` for the seed and
  `--seed .../seed-2.db --db ~/walkie-data/2026-11-19/memory-2.db` for the
  restore, and pass `memory-2.db` to `db-memory`. The seed step refuses an
  existing output (`... exists. Pass --replace to replace it.`), and the
  restore step refuses to replace a store that holds another map.
- Keep the store on the robot's local disk. Never delete a `.lock` file next to
  a store.

### d. Switch vision to the memory task (T7)

Stop the `hri` node with Ctrl+C, then:

```bash
pixi run --frozen vision task:=memory respawn_max_retries:=3
```

- It adds depth points, object crops and CLIP vectors, and a second YOLOE.
- Its classes are cup, bottle, bowl, apple, banana, backpack, chair, table,
  shelf, cabinet and sofa. Armchair and stool are gone, so `find_objects` sees
  different things than with `hri`.
- Its GPU memory and rate beside the ZED's depth have never been measured. If it
  runs out of memory, go back to `task:=hri`. The memory then stores nothing.

### e. Database in memory mode (T8)

Stop `db-node` with Ctrl+C first. Both are the node `/walkie_database`, so never
run both.

```bash
pixi run --frozen db-memory -p db_path:=$HOME/walkie-data/2026-11-19/memory.db \
    -p map_folder:=$MAP -p map_id:=lab-2026-11-19-XXXXXXXX
```

- Ready: `The object memory starts with no objects.`
- Refused: `Not starting: <reason>`, exit code 1. A map that changed after
  seeding is refused this way.
- It also serves the people services, so `--people-memory ros` keeps working.
- It drops arrays it cannot place on the map: TF from the camera to `map` must
  exist at each array's stamp. Set the initial pose before starting it.
  Objects stored under a wrong pose stay wrong. To clear them, stop the node and
  run the restore step again.

### f. Agent with the memory (T11)

Stop the agent with Ctrl+C, then:

```bash
pixi run --frozen agent --locations config/locations.lab.toml --tool-set hri --vision --people-memory ros --object-memory --tool-log logs/agent-tools.jsonl
```

`memory.db` is a new store, so the people remembered in the basic run
(`people.db`) are not in it. Remember them again after this restart.

### g. Ready check (T12)

```bash
pixi run --frozen ros2 service call /walkie_database/get_store_status walkie_database_msgs/srv/GetStoreStatus
```

Call it twice, a few seconds apart. `arrays_received` should climb, and
`arrays_rejected` should not climb with it. Then let the robot look at a chair
or a cup for a few seconds, and ask _"Where did you see a chair?"_.

??? info "Record the run for tuning"
    walkie-database's owner suggests recording the memory's input, since every
    value was tuned on simulated depth. In walkie-database's checkout:

    ```bash
    pixi run --frozen ros2 bag record -s mcap -o ~/walkie-data/2026-11-19/real-drive-01 /walkie_vision/objects /tf /tf_static
    ```

### Fallback

If `db-memory` refuses or misbehaves:

1. Stop it with Ctrl+C in T8, and start `db-node` again as in
   [step 5](#database-people-memory-t8).
2. Stop vision in T7, and start it again with the `hri` task.
3. Restart the agent without `--object-memory`.

The agent then loses only `recall_object`, `list_room_objects` and
`go_to_object`. The people memory and the live `find_objects` keep working.

## Optional C: the whole-body MoveIt test

The whole-body MoveIt experiment is a separate test with its own guide. It uses
no agent, vision or database. Its checkout is
`~/Project/robocup_experiments/walkie-ros-fullbody`, now on
`feat/sim-bottle-pick` (`a475236`), and its guide is
`docs/guides/whole-body-real-robot.md` there. That branch sits on
`feat/moveit-full-body` (`1da4535`). It changes simulation launches, worlds and
the scripted sim pick, and one thing the robot runs: the commander now makes its
collision-matrix writes one at a time, and a collision toggle turned back on
restores the matrix (PK-52). The real-robot launches and the MoveIt config are
the same as at `1da4535`.

- **Never run it at the same time as this page's stack.** Stop every terminal
  from this page first ([step 8](#8-stop-and-reset)). Both drive the same CAN
  buses, and a separate router or ROS domain does not separate CAN.
- It runs on its own Zenoh router, port 7499, with `ROS_DOMAIN_ID=77`. Open new
  terminals and set them up as that guide says, not as T1.
- Its stage 1 is a MuJoCo rehearsal with RViz and needs no hardware. Before its
  stage 2, which powers the arms, do the CAN driver check in
  [step 2](#2-before-you-start).
- Its stage 3, whole-body goals that move the lift on the real robot, is held
  back there, because the lift node does not follow a plan's timing. The pick
  runs it anyway, on this page's graph
  ([Optional E](#optional-e-pick-up-a-remembered-object)), as the owner
  decided (PK-5).
- `feat/sim-bottle-pick` installs `sim_bottle_pick.py` on the robot too. It
  refuses to run outside the simulation; never start it on the robot's graph.

## Optional D: web search

`--web-search` adds `search_web`, which looks up current facts such as news, the
weather or a recipe with Tavily. It needs a Tavily key in the agent's `.env`.

1. Sign up at [tavily.com](https://tavily.com) and copy the API key. The free
   plan gives 1,000 credits a month, and one search uses one credit.
2. In the agent checkout, open `.env` with `nano .env` and add one line,
   `TAVILY_API_KEY=` followed by the key. Never paste the key into a chat.
3. Check that it is set without printing it:

    ```bash
    grep -c '^TAVILY_API_KEY=.' .env
    ```

    It prints `1`.

4. Add `--web-search` to the agent's command in T11, for example:

    ```bash
    pixi run --frozen agent --locations config/locations.lab.toml --tool-set hri --vision --people-memory ros --web-search --tool-log logs/agent-tools.jsonl
    ```

Without the key, the agent stops at once with one `walkie-agent:` line that
names `TAVILY_API_KEY`. The web search has only been tested on recorded
answers, not against Tavily or on the robot (decision PK-7).

In our commander, **T11 agent, pick and web search** passes `--web-search`, and
**Ask the web (test task)** asks it a question that moves nothing
([Optional E](#optional-e-pick-up-a-remembered-object)).

## Optional E: pick up a remembered object

The agent fetches an object it remembers, such as _"Bring me the water
bottle."_ Its `pick_object` tool finds the object in the object memory, drives
to a spot facing it, reaches with the right arm through whole-body MoveIt,
closes the gripper and lifts the object 10 cm. It then moves the hand beside
the base and brings both arms home, a pose that is safe for driving, and the
agent brings the object back.

Once the robot stands by the object, the pick reaches only for a sighting of it
at most 5 s old, and waits up to 6 s for one. It checks again before the
whole-body reach, with the left arm in standby, and before the straight
approach, with the hand at the pre-grasp pose. It sends no further goal for an
object the memory no longer holds as present, one that moved more than 15 cm
since the last check, or, after the drive and before the reach, one that it or
any hand pose puts outside the reach envelope of `config/pick.toml`. A refusal
leaves the arms where they are, and its answer asks the agent to bring them
home. However the grasp ends, stopped included, the pick sets the commander's
`allow_gripper_vs_octomap` back to false without moving anything, and the agent
does the same at startup if an earlier run left it on. If that startup read or
reset fails, the agent only logs it: the allowance stays as it is until the
first motion of an arm or a gripper sets it, and that motion sends no goal
unless the commander confirms the change. So the bottle must be in view of the
camera that feeds the object memory from the spot facing it, during the whole
pick.

This is stage 3 of the whole-body guide (`docs/guides/whole-body-real-robot.md`
in `walkie-ros-fullbody`), which that guide holds back. The owner accepted
running it (decision PK-5).

!!! danger "Whole-body goals move the base, the lift and the right arm together"
    - **The lift ignores the plan's timing and rejects a cancel.** It drives
      straight to the plan's last height at up to 3 cm/s. A stop halts the arm
      and the base, and the lift finishes its move.
    - **There is no e-stop.** Keep a hand on the robot's power switch for the
      whole test. A second person on it is better.
    - **Only the pre-grasp reach** goes through `whole_body_right` and may move
      the base. The approach, the 10 cm lift and the move beside the base go
      through `right_arm_lift` in straight lines, and may move the lift. The
      move home then raises the lift to 74 cm, its SRDF home.
    - **The pick has never run on the real robot.** It was written for the
      Chula MuJoCo simulation, and its numbers (`config/pick.toml`) are
      provisional.

### What runs

Do the basic run, [Optional A](#optional-a-the-arms) and
[Optional B](#optional-b-the-object-memory) first, on the same map. The pick
uses all of them, with whole-body MoveIt in place of T5 and T6.

| Card in our commander | Group | Checkout |
| --- | --- | --- |
| T1 lift homing, T2 bringup (with the arms), T3 joystick | 1 base | team's walkie-ros |
| T4 Nav2 on $MAP | 2 map | team's walkie-ros |
| T9 speech out | 4 agent | walkie-agent-ros |
| P1 whole-body base parts, P2 MoveIt, whole body, P3 commander, whole body | 5 pick | walkie-ros-fullbody |
| T7 vision, memory task, T8 database, object memory, T11 agent, pick and web search | 6 pick | each its own |

- **P1** is `whole_body_real.launch.py static_map_odom:=false
  collision_monitor:=false`: the base follower and the hand target. AMCL gives
  `map` to `odom`, and Nav2's own collision monitor guards the base, as the
  guide's "What this stage relies on" says for a run with a map.
- **P2** and **P3** are MoveIt and the arm commander with
  `base_mode:=whole_body`. They refuse to start while another MoveIt or arm
  commander runs, such as T5 or T6, since both would drive the same
  controllers.
- **Every part joins this page's graph** (domain 23, the router on 7447), not
  the guide's private router on 7499. The pick needs Nav2, vision, the
  database and the agent on the same graph. So no teammate's joystick or
  teleop may run on domain 23 during the test.
- **T3, the joystick, keeps running** for button 0, which cancels Nav2's
  goals. Push the stick only to stop the robot: it bypasses the collision
  monitor and overrides the base follower.
- **T11 agent, pick and web search** needs `TAVILY_API_KEY` in `.env`
  ([Optional D](#optional-d-web-search)). Without it, the button stops at once
  with one `walkie-agent:` line. To run without web search, type its command
  without `--web-search` in the Shell tab.

### Check first

1. Every check of [step 2](#2-before-you-start), the CAN check above all. It
   must print exactly the three expected lines.
2. The whole-body guide's "Before you power the arms" list: the space around
   the arms and the lift is clear, the grippers are empty, and no hand is near
   them. If the right arm's joint 2 is locked (`right_joint2_fixed`), do not
   run the pick: it uses the right arm, and `whole_body_right` with a locked
   joint 2 is untested.
3. At least 1.5 m of free floor around the robot's path and around the table.
   Only the bottle, upright, near the table's edge on the robot's side, and
   nothing else within the arm's reach.
4. The object memory store for today exists: Optional B steps a to c ran, so
   `~/walkie-data/<day>/memory.db` is there. **T8 database, object memory**
   refuses to start without it, and makes the map ID from `$MAP` itself.
5. Someone holds the robot's power switch.

### Click in this order

Wait for each card's ready line before the next. Never use **Run group** here.

1. **T1 lift homing**. Ready: `Homing Successful. Origin set.`
2. **T2 bringup** (with the arms). It warns, then waits 5 s. Ready as in
   [step 3](#bringup-t2).
3. **T3 joystick**.
4. **T4 Nav2 on $MAP**. Ready: `Managed nodes are active`. Set the initial
   pose ([step 5](#nav2-on-the-saved-map-t4)), and check it with
   **Robot pose on the map**.
5. **T9 speech out**.
6. **T7 vision, memory task**, then **T8 database, object memory**. Ready as
   in [Optional B](#optional-b-the-object-memory), steps d and e.
7. **P1 whole-body base parts**. It warns, then waits 5 s. **P2 MoveIt, whole
   body**. Ready: `MoveGroup context initialization complete`. **P3
   commander, whole body**, which also warns first. Ready:
   `Bimanual Commander Node Initialized`.
8. **P check: MoveIt, commander, collision monitor**. It must print
   `OK: MoveIt runs`, `OK: commander runs` and `active`.
9. **P slow whole-body plans**. It sets the commander's whole-body velocity
   and acceleration scaling to 0.3, as the guide asks before the first goal.
10. **P lift to 74 cm**, alone in group 8. Look along the whole travel first.
    Wait until the lift stops at the top.
11. **T11 agent, pick and web search**. Ready:
    `Walkie's agent is running, with the hri tool set.`
12. **Ask the web (test task)**, if you set the Tavily key. It moves nothing.
13. Let the robot see the bottle for a few seconds, from where it stands or
    driven with the joystick. **P check: object memory status** twice:
    `arrays_received` climbs. Then **Where did you see a bottle? (test task)**.
    The reply must place the bottle. Vision's memory task labels it `bottle`.
14. **P bring me the water bottle**, alone in group 9. It warns, waits 5 s,
    then sends the task. Watch the robot, the console and
    `logs/agent-tools.jsonl`.

**To stop at any moment,** click **Stop the task** in group 0, or press
joystick button 0, and cut the power if anything still moves. The lift
finishes its move either way.

### Afterwards

1. **Stop the task**, if the agent still works.
2. Hold the bottle, and type in the Shell tab:
   `pixi run --frozen ros2 topic pub -w 1 --once --max-wait-time-secs 10 /from_human rai_interfaces/msg/HRIMessage "{text: 'Open your right gripper.'}"`
3. **P lift to 74 cm**, and wait until the lift stops. `home` needs the lift
   at 0.50 m or higher, or the fingers hit the base.
4. **P both arms home**, alone in group 10. Wait until the arms are still.
5. Stop the cards one at a time, in this order: T11, T9, T8, T7, P3, P2, P1,
   T4, T3, T2, and T1 last ([step 8](#8-stop-and-reset)).

### Not verified

- No part of this has run on the robot: not the pick, not whole-body MoveIt
  from `walkie-ros-fullbody` beside the team's bringup and Nav2, and not the
  base follower feeding Nav2's collision monitor.
- MoveIt plans with `walkie-ros-fullbody`'s robot model, while the team's
  bringup publishes its own. They are expected to match, but this was not
  compared.
- The hand's orientation for the side grasp, the stand-off distance and the
  grasp band were tuned in simulation at most.
- The sighting age, the 15 cm move limit and the reach envelope were derived
  from one simulated pick and the arm's reach. End-to-end picks in the Chula
  simulation passed all three checks, after the drive, before the reach and
  right before the approach, also after PK-51 moved the last one after the
  wait for the arm to rest; none has run on the robot.
  A pick that refuses a bottle in plain view points at one of them first.
- Whether the object memory finds a "water bottle" when vision stores a
  `bottle`.

## 8. Stop and reset

**Stop the task.** A bare `stop`, `never mind` or `nevermind`, typed or spoken,
cancels the task and its Nav2 goal, drops queued tasks, and says
`Okay, stopping.`:

```bash
pixi run --frozen ros2 topic pub -w 1 --once /from_human rai_interfaces/msg/HRIMessage "{text: 'stop'}"
```

"Walkie, stop" or "stop the music" are not stops. They go to the LLM as a new
task. Keep this command typed in a terminal, ready to send.

**Reset.** This also forgets the conversation and the people remembered in the
task:

```bash
pixi run --frozen ros2 topic pub -w 1 --once /walkie/reset std_msgs/msg/Empty
```

**Stop the robot.** In this order:

1. Press joystick button 0. It cancels all Nav2 goals.
2. Send the typed `stop` you keep ready. It ends the agent's task, so the LLM
   sends no new goal.
3. Cut the motor power if the robot still moves.

The stick only steers. It holds the base while you push it, but Nav2 takes
over again 0.5 s after you let go, unless its goal was cancelled. A cancelled
goal does not prove that the wheels stopped. The agent's stop
latency on the real robot was never measured.

**Shut down.** With the arms running ([Optional A](#optional-a-the-arms)), first
ask the agent to put both arms at `home`, and wait until they are still:

```bash
pixi run --frozen ros2 topic pub -w 1 --once /from_human rai_interfaces/msg/HRIMessage "{text: 'put both arms at home'}"
```

Then press Ctrl+C once in each terminal, in this order, and wait for each to
exit:

1. T11 agent. It cancels a running Nav2 goal. A second Ctrl+C forces an exit
   with code 3, and the goal's state is then unknown.
2. T10 voice-asr, T9 voice-tts.
3. T8 database, T7 vision.
4. T6 commander, T5 MoveIt (if running).
5. T4 Nav2 or SLAM.
6. T3 joystick.
7. T2 bringup. With the arms, they fall when it stops
   ([Optional A](#optional-a-the-arms)).
8. T1 lift homing, **last**. Started from a terminal where `sudo -v` ran, it
   takes `can2` down when it exits. Started from our commander, its `sudo`
   fails without a terminal, and `can2` stays up.

Leave the Zenoh router running.

In our commander, press **Stop** on each card in this order, and wait until
the card shows that it ended before the next. Never use **Stop All Running**
(see [Or use our commander](#or-use-our-commander)).

## 9. When something goes wrong

| Symptom | Likely cause | Check |
| --- | --- | --- |
| The lift or the base does not move, or CAN errors | CAN names swapped | The driver check in [step 2](#2-before-you-start) |
| The base stops working mid-test | The lift node exited and took `can2` down (only when it started from a terminal with `sudo`) | T1 is still running, and the CAN check in [step 2](#2-before-you-start). Restart the bringup after the lift, never the lift alone |
| The joystick does nothing | No gamepad, or the wrong device | The name check in [step 2](#2-before-you-start), then `dev:=` the gamepad's device in T3 |
| The robot moves when you move the remote mouse | T3 runs on RustDesk's `mouce-library-fake-mouse` | Stop T3 at once, and start it again with the gamepad's device |
| `tf2_echo` says `frame map does not exist`, or T4 never logs `Managed nodes are active` | Nav2 or AMCL is not up, or has no initial pose yet | T4's log for a failed lifecycle node, then set the initial pose ([step 5](#nav2-on-the-saved-map-t4)) |
| `voice-asr` hears nothing, or cannot open a device | The default source is not the fifine, it is muted, or `voice-asr` runs outside the desktop user's session | `wpctl status` as `eic` on the desktop ([step 2](#2-before-you-start)), and `wpctl get-volume @DEFAULT_AUDIO_SOURCE@` |
| Map server fails, `.../map.yaml/map.yaml` in the log | `map:=` given a file | Pass the folder: `map:=$MAP` |
| Keepout servers fail on a new map | No `keepout.yaml` | `use_keepout_zones:=false` |
| Nav2 loads the wrong map | No `map:=` given | The default is `30_16102026_ChulaPat14_6` |
| The robot stays at (0, 0) after 2D Pose Estimate | RViz's Fixed Frame is `base_link` | Set Fixed Frame to `map`, then estimate again |
| Nav2 goals stop at once | The collision monitor lost `/scan` or `/unilidar/cloud/filtered` | The topic check in [step 6](#6-first-checks-before-talking) and T2's lidar logs |
| `go_to` answers `Navigation is unavailable ... Is Nav2 running?` | Nav2 is down, or the terminal is on another domain | `echo $ROS_DOMAIN_ID` and `ros2 action list` in T11's setup |
| `go_to` times out after 120 s | The place is too close to a wall or obstacle | Move the place to open floor |
| Every drive is refused after Nav2 restarted | The agent keeps the old goal's record | Restart the agent after any Nav2 restart |
| `say` answers `no speech node is listening on /to_human` | Neither voice-tts nor the echo runs | Start T9 |
| The agent exits at once, `walkie-agent: ...` | Missing key, bad locations file or bad flags | The line names the reason. `ls .env` in the agent checkout |
| Every task fails after about 60 s | No internet or an OpenRouter outage | The `curl` check in [step 2](#2-before-you-start) |
| `find_people` answers that it cannot see | Vision not running, or its arrays are older than 1 s | Vision's 10 s rate line, and the `echo --once` checks |
| Vision restarts every 3 s | A startup failure, such as GPU out of memory or a missing model | The `cannot start:` line. A missing model names `pixi run fetch-models` |
| Vision publishes nothing for 30 s or more | The ZED watchdog restarted the camera | T2's `zed_watchdog` lines |
| Person IDs change mid-task | The vision node restarted | Vision's log |
| People calls answer UNAVAILABLE | The database node is not up | `Opened ...` in T8, and `ros2 service list` |
| Walkie hears itself or random talk | No wake word | Use text, and keep the stop command ready |
| `voice-asr` fails at its first start | No internet for Whisper and Silero VAD | Internet, then start it again |
| `db-memory` stops with `Not starting: The store holds the map '<A>', not '<B>'.` | Wrong `map_id:=` | Pass `map_id:=<A>`, the ID step a printed |
| `db-memory` stops with another `Not starting:` reason | The map changed after seeding | Run steps a to c of [Optional B](#optional-b-the-object-memory) again, writing new files (step c) |
| `arrays_rejected` climbs | No TF to `map` at the array's stamp, or bad arrays | The robot is localized, T1 runs, and `Refused an array:` in T8 |

## 10. Known limits of this first run

- **The agent has never run on the real robot.** Before this, it ran only on a
  fake robot and in the Chula MuJoCo simulation. Stop latency, TF freshness,
  discovery times and real Nav2 behaviour are unmeasured on hardware.
- **There is no software emergency stop.** See the warning at the top.
- **The CAN names were wrong on 2026-11-19,** and the fix is unverified.
- **No gamepad was attached on 2026-11-19.** The only joystick device was
  RustDesk's virtual mouse. Mapping and the button-0 stop need the team's
  gamepad.
- **The head is a mock** in the robot's working tree. Nothing tilts the ZED,
  so near tables it sees what it sees from its fixed height.
- **Vision was never run on this robot.** Its rate and GPU memory beside the
  ZED's depth are unknown. Only the laptop's numbers exist (RTX 2060: models
  loaded in 4.9 s, 86.6 ms per image at p50).
- **`--people-memory ros` has never run against a live database node.** The
  simulation used the in-process mock.
- **The object memory is untested on real data.** Its thresholds were tuned on
  noiseless simulated depth. Walls may be stored as furniture, and furniture can
  split. Stored objects never expire.
- **The arms:** receive and carry were never tried on the robot. Poses marked
  `tried_on_robot = false` are not offered.
- **Voice:** no wake word, half duplex, and the first `voice-asr` start needs
  internet.
- **HRI end to end** cannot run yet, in simulation or on the robot
  ([HRI](../architecture/hri.md)).
- **Not run before writing this page:** `map_saver_cli` into `$MAP`, the CAN
  fix, the internet check, the `--show-args` of the walkie-ros launch files
  (their arguments were read from the robot's files), and every command that
  joins the robot's ROS graph.
