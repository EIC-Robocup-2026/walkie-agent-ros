# Run the HRI Task on the Robot

This page is the procedure for running the HRI task on the real robot,
from the repositories to the logs. Follow it in order. Why each part works
the way it does is on [HRI Task](architecture/hri.md); this page links
there instead of repeating it.

!!! warning "The full run waits for the MotionSupervisor"
    The `hri` template's bag step needs `set_gripper`, which loads only
    with `--arms`. `--arms` waits for the MotionSupervisor, in the
    capability layer's PR C
    ([Robot Capabilities](architecture/capabilities.md#pr-c-the-motionsupervisor)),
    and for the arm steps in [step 14](#14-before-any-run-with-the-arms).
    Until then, `pixi run agent-robot-hri` refuses the template at start,
    and steps 12 and 13 test the gaze and the seats without the arms.

Every terminal on the robot PC uses `ROS_DOMAIN_ID=23` and the robot's
Zenoh router on `localhost:7447`
([ADR-0017](architecture/decisions/0017-rmw-zenoh-domain-23.md)).
`pixi run agent-robot-hri` sets the domain itself. In every other terminal,
run `export ROS_DOMAIN_ID=23` first. A node on another domain sees nothing,
and nothing reports an error.

## 1. Put each repository at its commit

On a robot PC that does not have this repository yet, first do steps 2 to
4 of [Getting Started](getting-started.md#2-get-the-code) there: get the
code, build the ROS packages with `pixi run build-ros2` and
`pixi run build-rmw-zenoh`, and put the API keys in `.env`. Without
`build-rmw-zenoh`, nothing warns you. The agent then runs on RoboStack's
rmw_zenoh, which can stop receiving messages in the middle of a run
([ADR-0032](architecture/decisions/0032-rmw-zenoh-from-source.md)).

| Repository | Commit | What the run uses from it |
| --- | --- | --- |
| walkie-agent-ros (this one) | The commit under test | The agent, and the pins in `ros2_ws.repos` |
| [walkie-vision](https://github.com/EIC-Robocup-2026/walkie-vision) | main, `1576d6f` | The vision node, one process for every model |
| [walkie-database](https://github.com/EIC-Robocup-2026/walkie-database) | main, `64403e9` | The database node with the people services |
| [walkie-ros](https://github.com/EIC-Robocup-2026/walkie-ros) | main, and two changes once they are pushed | The robot's bringup, Nav2, MoveIt and the arm commander |

walkie-ros needs two changes that are not yet pushed, so no commit with
them can be checked out on the robot yet:

- **W2:** the commander's `max_velocity_scaling_factor` and
  `max_acceleration_scaling_factor` become live parameters, 0.5 and 0.3 by
  default. Without it the arms move at full speed, so until it is pushed,
  only a team member may play the guest in a run with the arms
  ([step 14](#14-before-any-run-with-the-arms)).
- **The head-tilt switch:** a `head_tilt_near_goal` launch argument in the
  `navigation` submodule's real Nav2 launches. Until it is pushed, step 5
  turns the tilt off through the tilt node's service
  ([step 5](#5-start-nav2-on-the-arena-map)).

After every checkout or pull in the robot's walkie-ros checkout, update its
submodules and build it again. The commander is C++, so a new commit takes
effect only after the build:

```bash
git submodule update --init --recursive
pixi run build
```

The walkie-vision and walkie-database checkouts must be at the commits that
`ros2_ws.repos` pins. A client built from another version of a message
finds the server, but its requests are never answered
([ADR-0035](architecture/decisions/0035-memory-through-walkie-database.md)).
After every pull of this repository, build the message packages again:

```bash
pixi run build-ros2
```

Run `pixi run build-rmw-zenoh` again when the pull changes
`rmw_zenoh_ws.repos` or `scripts/rmw_zenoh_loader.c`.

`ros2_ws.repos` also pins walkie-ros at `6b26daa`, for the arm commander's
interfaces
([ADR-0018](architecture/decisions/0018-pin-robot-interfaces.md)). In the
robot's walkie-ros checkout, check that they have not changed since:

```bash
git diff --stat 6b26daa HEAD -- manipulate/my_robot_interfaces
```

No output means the pin still matches. Otherwise, move the pin and run
`pixi run build-ros2` again.

## 2. Check walkie-vision on the robot's GPU

walkie-vision has not yet been measured on the robot's GPU. In its
checkout, do its
[steps before the first run](https://eic-robocup-2026.github.io/walkie-vision/guides/vision-node/#before-the-first-run)
once, then run its checks:

```bash
pixi run check-env
```

All 18 checks must pass
([Check this computer](https://eic-robocup-2026.github.io/walkie-vision/getting-started/#5-check-this-computer)).
Run it again after every driver update. Keep `checks/output/summary.txt`,
which records the GPU and the driver.

## 3. Confirm the emergency stop

Before anything on the robot moves, confirm two things with the team:

- **The physical emergency stop works.** Test it today, before the
  bringup.
- **Who holds it.** One named person stays within reach of it for the
  whole session, the tests included.

Nothing in software replaces it. walkie-ros has no software emergency
stop. Parts of the robot move as soon as their launches start
([walkie-ros's safety notes](https://eic-robocup-2026.github.io/walkie-ros/guides/real-robot/#safety-notes)):

- the lift homes as soon as its launch starts;
- `real_omnibot.launch.py` moves both arms to their home pose over 5 s
  once their controllers start, and the head to 0.25 rad. Keep everyone
  clear of the arms.

The agent's "stop" and `/walkie/reset` need ROS, MoveIt and the
controllers to work
([Arm Manipulation](architecture/manipulation.md#safety)). Until the
capability layer's PR B, neither ends a drive that is already under way:
the robot drives on to its goal, for up to 120 s
([Stop, reset and the cancel](architecture/agent.md#stop-reset-and-the-cancel)).
To stop a drive, press the joystick's button 0, which cancels Nav2 goals,
or the emergency stop. Keep the joystick at hand for that: it also
overrides Nav2 and the gaze loop. It does not stop the arms.

Do not go on until both points are confirmed.

## 4. Start the robot

In walkie-ros, read the
[safety notes](https://eic-robocup-2026.github.io/walkie-ros/guides/real-robot/#safety-notes),
then follow
[Start the robot](https://eic-robocup-2026.github.io/walkie-ros/guides/real-robot/#bringup-order):
the router, the lift homing, `real_omnibot.launch.py` and the joystick.
Leave out its Nav2 step, which [step 5](#5-start-nav2-on-the-arena-map)
replaces. MoveIt and the commander are needed only for
[step 14](#14-before-any-run-with-the-arms) and the run with the arms.

The router from that guide is the only one on the robot PC. Do not start
another.

## 5. Start Nav2 on the arena map

Use the map of the arena the run is in. If it has none yet, build and save
one, as in walkie-ros's
[Localization and mapping on the real robot](https://eic-robocup-2026.github.io/walkie-ros/guides/navigation/#localization-and-mapping-on-the-real-robot).
Then, in walkie-ros's checkout, start localization and Nav2 on it, with
the head tilt off:

```bash
export ROS_DOMAIN_ID=23
MAP_DIR=<the arena map's folder>
pixi run ros2 launch robot_navigation localization_Nav2_real.launch.py \
  map:=$MAP_DIR head_tilt_near_goal:=false
```

`map:=` takes the map's folder, not its `map.yaml`. The launch reads
`map.yaml` from the folder, and by default the keepout mask from
`keepout.yaml` in the same folder. If the folder's mask has another name,
also pass `keepout_mask:=$MAP_DIR/<the mask's file>`. If the folder has no
mask, as a map just saved by the map saver has none, pass
`use_keepout_zones:=false`. walkie-ros's
[Choose a map](https://eic-robocup-2026.github.io/walkie-ros/guides/navigation/#choosing-a-map-on-the-real-robot)
still shows the older form, `map:=$MAP_DIR/map.yaml`, which the launch now
reads as `$MAP_DIR/map.yaml/map.yaml`. Without `map:=`, the launch uses
walkie-ros's `27_01072026_Arena` map. Nav2 is ready when its log shows
`Managed nodes are active`.

AMCL then starts at the map's origin, with x, y and yaw at 0, and does not
find the robot by itself. Set the robot's real pose with RViz's
**2D Pose Estimate** tool (`real_omnibot.launch.py` starts RViz), or
publish it on `/initialpose`. Check in RViz that the laser scan lines up
with the map's walls before you measure a place or send a task.

With the tilt on, the ZED points 45° down as the robot nears the door or
the sofa, just when `find_people` and `find_seats` look
([HRI Task](architecture/hri.md#delivery-order)). `ros2 launch` ignores an
argument the launch does not declare, so check, in another terminal, that
the tilt node is not running:

```bash
export ROS_DOMAIN_ID=23
pixi run ros2 node list | grep -E '^/(amcl|head_tilt_near_goal)$'
```

The output must show `/amcl` and must not show `/head_tilt_near_goal`. No
output at all means the terminal is not on the robot's graph: check its
domain and the router. If it also prints `/head_tilt_near_goal`, the robot's
walkie-ros does not have the switch yet. Turn the node off through its
service instead, after every Nav2 start:

```bash
export ROS_DOMAIN_ID=23
pixi run ros2 service call /head_tilt_near_goal/enable std_srvs/srv/SetBool "{data: false}"
```

## 6. Measure the places

`config/locations.robot.toml` lists the places the HRI task drives to,
`door` and `living_room`, without coordinates. Until both have them,
`pixi run agent-robot-hri` stops at start with
`door, living_room have no measured x and y`.

Once AMCL has localized the robot, with the scan on the map's walls as in
[step 5](#5-start-nav2-on-the-arena-map), measure each place on the map:

1. Drive the robot with the joystick to the place, facing the way the
   file's comment says.
2. Read its pose in the map, in this repository:

    ```bash
    export ROS_DOMAIN_ID=23
    pixi run ros2 run tf2_ros tf2_echo map base_footprint
    ```

3. Copy `Translation`'s x and y, and the last value of `Rotation: in RPY
   (radian)`, into the place's `x`, `y` and `yaw`, and remove the `#` in
   front of each. Rewrite the place's `description` for this arena.

Then commit the measured file, so the commit under test carries the poses
it runs with. The tests accept the file both before and after it is
measured.

The places:

- **`door`:** inside the arena, near the entrance door, facing out through
  the doorway. The robot stands far enough in that the left arm's
  `receive` pose clears the open door leaf.
- **`living_room`:** one pose that sees every seat, since the robot looks
  for seats from one view
  ([Seats](architecture/hri.md#seats)).

Measure them again whenever the map changes.

## 7. Start the database node

In walkie-database's checkout, in its own terminal:

```bash
export ROS_DOMAIN_ID=23
pixi run db-node
```

Its migration 0003 applies when it starts. Without the node,
`recognize_person` and `remember_person` answer `UNAVAILABLE`
([Getting Started](https://eic-robocup-2026.github.io/walkie-database/getting-started/#4-run-the-database-node)).

## 8. Start the vision node

In walkie-vision's checkout, in its own terminal:

```bash
export ROS_DOMAIN_ID=23
pixi run vision
```

It publishes `/walkie_vision/people` and `/walkie_vision/objects`
([Run the vision node](https://eic-robocup-2026.github.io/walkie-vision/guides/vision-node/)).

- **The first start on a new GPU** compiles kernels and can take minutes.
  Keep `~/.nv/ComputeCache` between runs.
- **Its log line** should show at least 5 Hz and an age under 300 ms at
  p95
  ([Read the log](https://eic-robocup-2026.github.io/walkie-vision/guides/vision-node/#read-the-log)).
- **One vision node per ROS graph.** Never start a second one, or this
  repository's `hri-scenario` or `oracle_perception`, on domain 23.

## 9. Set the camera's values in config.toml

The agent judges face sizes as angles, from the camera's focal length.
Read the robot's own fx, the first value of `k`, from the vision node's
output, in this repository, whose `pixi run build-ros2` built the vision
node's messages:

```bash
export ROS_DOMAIN_ID=23
pixi run ros2 topic echo --once /walkie_vision/people --field camera_info.k
```

Then set, in this repository's `config.toml`:

- **`[walkie.perception.vision_feed]`:** `min_face_size_rad` to 50 / fx.
  For the ZED 2i's nominal fx of 1066 px that is the default, 0.0469.
- **`[walkie.perception.front]`:** keep `min_face_rad = 0.10` and
  `half_angle_deg = 25.0` to start. `min_face_rad` may not go below
  `min_face_size_rad`. Tune both at the door in
  [step 12](#12-test-the-gaze-alone).

The other robot values, the face and appearance distance limits and the TF
waits, are in the table under
[Delivery order](architecture/hri.md#delivery-order). The agent reads
`config.toml` at start, so restart it after each change.

## 10. Start the voice nodes

Run them as the user logged in to the robot PC's desktop
([Running on the robot PC](setup/07-voice.md#running-on-the-robot-pc)),
each in its own terminal:

```bash
export ROS_DOMAIN_ID=23
pixi run voice-tts
```

```bash
export ROS_DOMAIN_ID=23
pixi run voice-asr
```

## 11. Start the agent

In this repository:

```bash
mkdir -p logs
pixi run agent-robot-hri 2>&1 | tee logs/agent_$(date +%Y%m%d-%H%M%S).log
```

The task sets domain 23, uses `config/locations.robot.toml` and the
people services of walkie-database, and writes every tool call to
`logs/agent_tools_<time>.jsonl`. Without `--arms`, its log warns at start
that the `hri` template cannot start in this run, because its bag step
needs `set_gripper`.

Read the agent's text replies, and a line per todo status change, in
another terminal in this repository, which has `rai_interfaces`:

```bash
export ROS_DOMAIN_ID=23
pixi run ros2 topic echo /walkie/console rai_interfaces/msg/HRIMessage --field text
```

## 12. Test the gaze alone

With the robot at `door`, a team member pacing in front of it and the
joystick at hand, say: "Please look at the person in front of you." The
agent calls `find_people` and `look_at`, and its log shows
`Gaze: turning toward ...`. The base should turn to follow the person and
never drive. "Stop looking at me." ends it. "Stop." ends the gaze loop and
the task, but not a drive already under way: stop that with the joystick
([step 3](#3-confirm-the-emergency-stop)).

Tune `[walkie.perception.front]` here, with team members walking past
behind the guest and standing beside the robot
([One run](architecture/hri.md#one-run)). Tune the face distance limit
with team members too
([Perception question 2](architecture/perception.md#open-questions)).

## 13. Test the seats

With team members on some of the living room's seats, say: "Please go to
the living room and tell me which seat is free." The agent drives to
`living_room`, calls `find_seats`, faces a free place and names it.
Check that it never offers a taken one.

## 14. Before any run with the arms

Do all of these before any run with `--arms`. That run also waits for the
MotionSupervisor: until PR C has landed, never start the agent with
`--arms` on the robot.

1. **W2 on the robot** ([step 1](#1-put-each-repository-at-its-commit)).
   Without it the commander moves the arms at full speed, toward a
   person. Then only a team member may play the guest
   ([The bag hand-over](architecture/hri.md#the-bag-hand-over)). With the
   commander running, check that W2 is live:

    ```bash
    export ROS_DOMAIN_ID=23
    pixi run ros2 param get /bimanual_commander max_velocity_scaling_factor
    pixi run ros2 param get /bimanual_commander max_acceleration_scaling_factor
    ```

    They must print 0.5 and 0.3. `Parameter not set` means the commander
    was built without W2.
2. **Manipulation's robot steps R0 to R4** pass, on top of PR C
   ([Testing](architecture/manipulation.md#testing)).
3. **Record `receive` and `carry` on the robot.** They were recorded in
   the simulation, so only the simulation offers them. Record them the
   same way on the robot, try them with collision checking on, and set
   `tried_on_robot = true` in `config/arm_poses.toml`
   ([The bag hand-over](architecture/hri.md#the-bag-hand-over)).
4. **Check the grasp band.** `set_gripper` judges a close by the
   commander's `grasp_detect_min` and `grasp_detect_max`, which it reads
   back from the commander. Close the left gripper on nothing and on the
   bag's handle, as in step R5. A close on the handle must land inside
   the band and a close on nothing below it. Set the band in walkie-ros's
   commander parameters, not here.

Then start the agent with the arms:

```bash
pixi run agent-robot-hri --arms 2>&1 | tee logs/agent_$(date +%Y%m%d-%H%M%S).log
```

Its log no longer warns about the `hri` template. Try the bag hand-over
with a team member as the guest before anyone else plays one. Once this
run works, add `--arms` to the task's `cmd` in `pixi.toml`.

## 15. Run the task

Put the robot at its start pose in the living room, with the guests
outside the door. If you moved it by hand, set its pose again with
**2D Pose Estimate**, and check that the scan lines up with the map's
walls ([step 5](#5-start-nav2-on-the-arena-map)).

Rulebook 2026.2 gives the HRI task 7 minutes. Time the practice run
yourself. Start it with the `hri` template
([The template](architecture/hri.md#the-template)):

```bash
export ROS_DOMAIN_ID=23
pixi run ros2 topic pub --once /walkie/task std_msgs/msg/String \
  "{data: '{\"task\": \"Please receive the guests.\", \"template\": \"hri\"}'}"
```

Saying a sentence with one of the template's keywords, such as "Please
receive the guests.", starts it too. Without `--arms` the agent refuses
the template, and the reason is in its log.

At the competition the robot starts inside the arena, so the rulebook's
Start Signal rule applies. The referee gives a manual start signal that
was approved at Robot Inspection. The robot must tell the referee that it
is ready and what to do to start it. A spoken instruction is one of the
rulebook's examples. A start signal that needs the referee to use a mouse
or a keyboard is forbidden. The team registers a spoken keyword, said
after a ready line from the robot
([HRI question 12](architecture/hri.md#open-questions)). The ready line
is not built yet.

## 16. Collect the logs

After the run, keep these from `logs/`:

- `agent_tools_<time>.jsonl`, every tool call with its arguments and
  result;
- `agent_<time>.log`, the agent's own log from step 11.

`pixi run hri-check` judges a simulation run. It needs the event log that
`hri-scenario` writes, which a robot run does not have. So before the
robot, run the simulation of the same commit on another computer, never
on domain 23, and check it
([Getting Started](getting-started.md#run-the-hri-scenario)):

```bash
pixi run hri-check logs/hri_scenario_<time>.jsonl logs/agent_tools_<time>.jsonl
```

Check the robot run by hand, against the same criteria
([In simulation](architecture/hri.md#in-simulation)). Each line of the
tool log is one call. For example, the names and drinks the robot stored:

```bash
grep '"tool": "remember_person"' logs/agent_tools_<time>.jsonl
```
