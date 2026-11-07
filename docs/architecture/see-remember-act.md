# See, Remember and Act (Roadmap)

The order in which the `feat/mahoraga-demo` prototype gains real perception,
memory and arm actions, and how each step is tested without the robot.
Groundwork items are G1 to G6, and the steps are S1 to S13. Decisions taken
while building them go in the
[decision log](mahoraga-demo-decisions.md#see-remember-and-act), under the
ID of the step they belong to.

!!! warning "Prototype plan"
    Every step is tested in the fake world, in unit tests on Fast DDS, and in
    the MuJoCo simulation of the `chula_pat14_6` room. None is tested on the
    robot. Real-robot runs need W2 speed scaling, an e-stop operator, the
    robot's GPU measured and the robot's walkie-ros and walkie-database
    commits settled, and this plan covers none of these.

## Starting point

On the robot path, `main.py` assembles a `Ctx` whose base drives through
Nav2 (`RosBase`) and whose speech publishes on `/to_human` (`RosSpeech`).
Vision is `UnavailableVision`, arms are `UnavailableArms`, and there is no
memory capability. `find_people`, `find_objects` and the arm atomics work
only on `FakeRobot`. The gaps this plan closes:

- **See.** No backend reads `/walkie_vision/people` or
  `/walkie_vision/objects`, and the simulation has no source for either.
  MuJoCo has no person bodies.
- **Remember.** walkie-database serves people (`StartSession`,
  `RememberPerson`, `MatchPeople`) and seeded rooms and anchors. Every object
  write answers `NOT_READY` until its milestone M1, so "where did you see the
  mug" cannot come from the database yet.
- **Act.** The arms are not wired. Decision 12 (the arm path) and decision 14
  (where gaze runs) are open with the team and the HRI owner.

## Constraints

- **No `/cmd_vel` from the agent.** `policy.py` forbids it, and `RosBase`
  refuses a forbidden endpoint when it is built (W11). Turning toward a
  person uses Nav2 through
  `ctx.base.move_relative(0, 0, turn_deg)` until decision 14 is taken.
- **Mahoraga stays fake-world only.** New people, memory and arm atomics stay
  out of the GPSR tool set, so the gym and P30's "no atomic calls a database
  service" are unaffected. The Mahoraga gate's digest covers
  `capabilities/`, `atomics/`, `locations.py` and `pixi.lock`, so a change
  there reruns the gate.
- **Named arm presets only.** The first arm steps use `set_joint_position`
  and `control_gripper` on `left_arm` and `right_arm`, which
  [ADR-0021](decisions/0021-arm-control-through-commander.md) already allows.
  Whole-body groups stay refused, and
  [ADR-0026](decisions/0026-exclude-lift.md) bars lift motion.
- **Port from the experiment line.** Code comes from `experiment/hri-agent`
  at 61d30e8, not from 6426f47, which is not one of its ancestors.
- **No vectors in sim memory.** walkie-database refuses vectors from models
  that have no table, and the simulation's oracle sends `fake_` vectors. Sim
  runs therefore use an in-memory people store, and the database path is
  tested on a laptop with a real camera and real vision models.
- **Memory ADR number.** The memory capability's ADR takes 0070. 0060 to
  0069 are used, and 0053 to 0059 are reserved for the experiment.

## Groundwork

| ID | Item | Size | Needed by | Status |
| --- | --- | --- | --- | --- |
| G1 | Pin walkie-vision 64811fe and walkie-database 522ab92, and build `walkie_vision_msgs` ([Local message pins](../setup/03-ros-and-rai.md#local-message-pins)) | S | S1, S3, S8 | Built |
| G2 | A Chula place list with rooms, an optional `room` on `Location`, and an `agent-sim-chula` task | S | S1, S8 | Built |
| G3 | A sim perception source: port `oracle_perception` with a small scene driver | M | S1, S4 | Built |
| G4 | A scripted, isolated Chula simulation recipe | S | Every sim test | Built |
| G5 | Pin walkie-ros `my_robot_interfaces` at 6b26daa and build it | S | S6 | Built |
| G6 | Record the prototype's position on decisions 12 and 14 | S | S6, S11 | Recorded |

**G2.** `config/locations.chula_pat14_6.toml`, keyed by display names
(`living room`, `kitchen room`, `table`, `shelf`, `bar`, `tv`, `sofa`,
`entrance`), with the coordinates `FakeRobot` already copies from the
walkie-ros world. The same file serves walkie-database's sim seed, which
reads stand poses under the rooms' display names (P23). An optional `room`
field on `Location` lets the vision backend fill `PersonSeen.room`.
`agent-sim-chula` runs beside `agent-sim`, which keeps the blank-map points.
Built as G2-1 and G2-2. G4 drove Nav2 to every place in MuJoCo; the two
invented places moved to more open floor before they were reached (G4-6).

**G3.** Port `sim/oracle_perception.py`, `scene.py`, `projection.py` and
`fake_vectors.py` and rebuild them against 64811fe. Person messages are
unchanged since 1576d6f. The seat and object arrays need a UUID
`stream_id`, `view_quality` and `color_shares` (P13). A small scene
driver places static people, with gesture and posture, and seats and table
objects in the Chula room, instead of the experiment's full HRI scenario
driver. The oracle publishes on the plain `/walkie_vision/people` and
`/walkie_vision/objects` topics, as the vision contract requires in sim.
Built as G3-1 to G3-6: the scene is TOML, `sim/scene_file.py` loads it, and
`pixi run oracle-chula` (`sim/oracle_scene.py`) draws
`config/scenes/chula_living_room.toml`, the living room only, with a guest
raising the left arm and the host on the sofa. `perception/gestures.py`
came with it. G4 ran it against MuJoCo: from the living room place it draws
both people with the scene's gestures, and the sofa, the table and the mug
([In simulation](perception.md#in-simulation)).

**G4.** One recipe that never joins another running ROS graph: a private
Zenoh router on port 7499 and `ROS_DOMAIN_ID` 77. It launches
`nav_mujoco_omnibot.launch.py world:=chula_pat14_6` from a local walkie-ros
worktree of `feat/chula-pat14-6-simulation`, the G3 scene driver, and the
agent with `--use-sim-time` and the Chula places. For the arms it adds
`moveit_only.launch.py hardware_type:=mujoco` and the commander on sim time.
The commander's own `commander.launch.py` declares no `use_sim_time`, so
passing `use_sim_time:=true` to it does nothing; this tree's
`launch/commander_sim.launch.py` sets the parameter around it. For memory it runs `db-seed-sim`, `db-restore` (P22)
and `db-memory`, retrying until `/clock` arrives (P25). Voice goes to a
null sink, so nothing is audible.
Built as G4-1 to G4-12: `scripts/chula-sim.sh up` starts the router, the
world from a worktree that builds only the branch's two changed packages, the
oracle, a text log of `/to_human` and, with `--agent`, the agent. `check`
reports what the graph provides. Arms join with S6 and memory with S3, and
speech is logged rather than voiced
([Run the Chula simulation](../setup/08-chula-simulation.md)). Nav2 reached
every Chula place once `trash` and `instruction point` moved.

**G5.** 6b26daa is the commit the experiment's `robot_model.toml` snapshot and
arm tests were built on. `my_robot_interfaces` is unchanged from there to
walkie-ros `origin/main` 813be41, so the pin is fetchable from GitHub. The
full-body branch's 8074d22 changed the SRDF and is not pinned.
Built as G5-1: `build-ros2` clones walkie-ros at the pin and builds
`my_robot_interfaces`. The package and the MoveIt config that the snapshot
reads are the same on 6b26daa, 813be41 and the simulation's adb9725.

**G6.** Decision 12 (the arm path) and decision 14 (where gaze runs) stay
with the team and the HRI owner. The log records that the prototype uses only
named presets, grippers and discrete Nav2 turns, which pre-empt neither.
Recorded as G6-1 and G6-2. S7's and S10's agent-side bundles are the one
place the prototype leans toward decision 12's recommended option; under the
other option they become `run_skill` calls.

## Steps

| Step | What the user can ask | Depends on | Size |
| --- | --- | --- | --- |
| S1 | "Is anyone here?" "Is someone waving?" "Who is sitting down?" | G1 to G4 | M |
| S2 | "Look at me." "Turn to the person who is waving." | S1 | S |
| S3 | "I'm Anna, please remember me." Later: "Do you know who I am?" | S1, G1 | M |
| S4 | "What's on the table?" "Is there a mug here?" | S1, G3 | M |
| S5 | "Can you find me a seat?" | S2, S4 | M |
| S6 | "Raise your hands." "Open your left hand." (MuJoCo) | G4, G5, G6 | L |
| S7 | "Can you hold my bag?" "Give me my bag back." | S6, S2 | M |
| S8 | "What rooms do you know?" "Remember that Anna likes tea." | S3, G1, G2, G4 | M |
| S9 | "Where did you see the mug?" | S4 | S |
| S10 | "Bring the apple from the kitchen table to the bar." (MuJoCo) | S4, S6 | L |
| S11 | Keeps facing the guest while talking | S1, S2, decision 14 | L |
| S12 | HRI slice 1 end to end, scored by `hri_check` | S3, S5, S7, S11 | L |
| S13 | "What is the guest wearing?" | S1 | M |

S1, S3 and S6 go first. S1 unblocks the most: it gives the existing
`find_people` a real backend with no new tool, and it forces the groundwork
everything else needs. S2 follows S1 at once, since it is small and visible.
S3 is the one kind of remembering walkie-database fully supports today. S6 is
the only acting step that crosses no ADR and pre-empts neither team
decision, and it can run beside the vision chain.

### S1: See people in view

- **Build.** A `RosVision` in `ros/vision.py` whose `people()` reads a ported
  `VisionFeed`: the people subscription, BEST_EFFORT with depth 5, arrays
  older than 1 s dropped
  ([ADR-0050](decisions/0050-perception-atomics-over-walkie-vision.md)).
  Gesture and posture come from the ported `gestures.py`, phrased as
  `FakeRobot` phrases them. `objects()` stays unavailable. `main.py` gets an
  opt-in `--vision` flag, and `UnavailableVision` stays the default.
- **Ported from `experiment/hri-agent`.** `perception/vision_feed.py` (the
  people path), the `RosVision` shape in `tools/atomic/_perception.py` and
  `FakeVision` in `testing/__init__.py`. `perception/gestures.py` came with
  G3.
- **Test.** The `FakeRobot` and `find_people` tests stay green, and the
  Mahoraga gate reruns. Unit tests feed recorded `PersonDetectionArray`
  messages on Fast DDS: gestures, the age cutoff, query filters, room. In
  sim, G3's oracle scene of two people in the living room, one waving and one
  sitting; the agent runs with `--tool-set hri --vision`, and "is anyone
  waving?" must show `find_people` returning the waving guest in the tool log.
- **Defaults (delegated).** A person ID is an opaque string per track,
  `person_<track_id>`, valid while the track lives, and the tool docstring
  says IDs can expire. `PersonSeen.room` is the room of the named place
  nearest the robot, through G2's `Location.room`. Real vision is opt-in
  until a sim run passes.
- **Team call.** Which walkie-vision task the robot launches per test, and who
  starts the node. Not needed in sim.
- **Risks.** Tracks drop after 30 frames, so an ID from an earlier call can go
  unknown (Perception Q8). The keypoint gesture rules are unvalidated on real
  walkie-vision output. On the robot, `head_tilt_near_goal` points the camera
  down near goals. The oracle has no occlusion, so sim is optimistic.
- **Built** as S1-1 to S1-9. `RosVision` keeps only the newest array, and the
  phrases live in `perception/people.py`, which imports no ROS. The plan
  changed in one place: the robot-only notes, including that an ID can
  expire, are in the `--vision` prompt rule, not in the docstring. So the
  docstring and the gate's digest are unchanged, and the gate did not rerun
  (S1-8). In sim, with `--tool-set hri`, "Is anyone waving at you?" got
  `find_people("waving person")` returning `person_1` in the living room, and
  "Who can you see right now?" got both guests
  ([What has been run](../setup/08-chula-simulation.md#what-has-been-run)).

### S2: Turn to face a person

- **Build.** `PersonSeen` gains an optional `bearing_deg`, relative to the
  base heading, computed from the box's centre column, `camera_info` and TF
  at the image stamp, through a port of `perception/geometry.py`. A new
  `face_person(ctx, person_id)` atomic finds the person again and turns with
  `ctx.base.move_relative(0, 0, bearing)` through Nav2. It goes in the HRI
  and default sets, not GPSR.
- **Test.** `FakeRobot` people with bearings, and a gate rerun. In sim, a
  guest about 60 degrees to the left; after "look at me, I'm the one
  waving", the heading in `RosBase.pose()` must be within about 10 degrees.
- **Default (delegated).** Discrete turns ship before decision 14. This is not
  [ADR-0051](decisions/0051-agent-side-gaze-loop.md)'s loop and does not
  touch `/cmd_vel`.
- **Risks.** Whether Nav2's controller turns in place for a goal with no
  translation, and its yaw tolerance, are untested. One turn is not
  continuous gaze, so it does not meet HRI's gaze criterion (S11).
- **Built** as S2-1 to S2-10, with S2-9 and S2-10 from the review. The plan
  changed in four places. The ray goes
  through the pixel's row as well as its column, and through the shoulders and
  hips before the box's centre, since the camera's pitch and a raised arm each
  move the box-centre bearing by up to a few degrees (S2-1). The bearing is
  from the base's turning centre, not the camera 0.275 m ahead of it: the
  ankles' rays meet the floor at the person's distance, and the oracle's Chula
  guest reads 35.6° against a true 35.56°, where the camera saw 39.6° (S2-3).
  A person within 10° gets no turn, and a turn Nav2 aborts within 10° of its
  goal counts as done, but not one whose end is unknown (S2-4, S2-9). Nav2 does turn in place: in 17 turns from 4° to
  90°, 15 succeeded within 0.72° and two aborted with code 105 about 3° past
  the goal. In sim, "Look at me, I am the one waving." with the real model
  found the guest and turned to them. With the floor distance, the guest ended
  4.0° off the base's heading, within the test's 10°; Nav2 had aborted 3.0°
  past the goal, and the live bearing after the turn matched the truth within
  0.6° ([What has been run](../setup/08-chula-simulation.md#what-has-been-run)).

### S3: Remember people by name

- **Build.** A `Memory` protocol on `Ctx` (`ctx.memory`, default
  `UnavailableMemory`), starting with `remember_person(person_id, name,
  facts)` and `recognize(person_id)`. The robot backend ports
  `RosPeopleMemory` over `StartSession`, `RememberPerson` and `MatchPeople`;
  it takes vectors from the feed's per-track buffer and never returns them.
  Sim uses the ported `InMemoryPeopleMemory`, chosen with a
  `--people-memory mock` flag. The `remember_person` and `recognize_person`
  atomics go in the HRI set only. Retries follow P25's prefix. ADR-0070
  records the capability.
- **Ported from `experiment/hri-agent`.** `perception/people_memory.py`,
  `perception/ros_people_memory.py`, `perception/identity.py`, the two
  atomics' descriptions and `tests/test_ros_people_memory`.
- **Test.** A remember and recognize round trip, and an unknown person, in
  the fake world. In sim, oracle guests with the in-memory store: the guest
  says their name, the robot turns away, and `find_people` then returns the
  name. The database path runs on a laptop: `db-node` and walkie-vision with
  real models on a laptop camera, remember someone, restart the agent with a
  resumed session, and recognize them.
- **Defaults (delegated).** One `ctx.memory` protocol: people verbs now,
  object, room and event reads in S8. People memory is per task by default,
  and a flag resumes the previous session. The effect letter of
  `remember_person` is deferred, since the atomic stays out of GPSR and
  Mahoraga; Perception Q3 stays open.
- **Risks.** Sim never exercises the database path. The face distance limit
  is untuned (Perception Q2). Stored faces raise a privacy and retention
  question on the robot.
- **Built** as S3-1 to S3-10. The plan changed in three places. The model
  sees only live ids, and the store's `stored_<n>` ids stay in the backend
  (S3-2). find_people names the people in view, so `find_people("Sam")`
  works, and `recognize` names everyone in view at once (S3-4). People are
  forgotten at each reset on `/walkie/reset`, which this runtime treats as
  the end of a task (S3-6). The buffer is `perception/track_vectors.py`, the
  matching `perception/recognition.py` and `perception/memory.py`, and the
  database client `ros/people_memory.py`. A Fast DDS test sends oracle-style
  people arrays through `RosVision` into the in-memory store and finds the
  name again. The laptop run of the database path has not been done. The
  Codex review added S3-11: a stop fences every store write, so a stop that
  lands first stores nobody. In sim, with `--people-memory mock` and the real
  model, the agent remembered the waving guest as Sam, named Sam again after
  turning away and back, and named nobody after a reset
  ([What has been run](../setup/08-chula-simulation.md#what-has-been-run)).
  The same run found a 180° `move_relative` that Nav2 aborted 4° past its
  goal; the model retried it and ended facing away. A turn on the spot now
  counts as done within 10° of its goal, as face_person's does (S2-11).

### S4: See objects in view

- **Build.** `RosVision.objects()` over the feed's objects path, rebuilt for
  64811fe's `ObjectDetectionArray`. Category comes from a name table.
  `place` is the named place Walkie stands at, or `unknown`. Size, weight and
  thickness stay `None` for live views. IDs are transient live IDs, never
  database IDs (plan decision 4). In sim the oracle publishes the Chula table
  objects from a static scene.
- **Test.** In sim, `go_to('table')` and "what's on the table?" must list
  `apple_1`, `mug_1` and `pringles_can_1` in the tool log.
- **Defaults (delegated).** Place is the named place within tolerance, moving
  to `LocalizeObjects` centroids once `db-memory` runs. A static oracle scene
  first; MuJoCo's `FreeJointStatePublisherPlugin` comes with S10, when
  objects move.
- **Team call.** The vision task per test on the robot: `hri` for HRI,
  `memory` for GPSR, `DetectObjects` only on demand.
- **Risks.** One vision node runs one task: `hri` gives only seats, and
  `memory` lacks armchair and stool. The robot's place mislabels objects seen
  across the room. Whether the MuJoCo world frame matches the map frame is
  unverified.

### S5: Find a free seat and offer it

- **Build.** A `find_seats` atomic, ported from `find_seats.py` and
  `seating.py`, returning each seat's ID, kind, bearing and whether it is
  free. The offer turns as in S2 and names the sofa or chair. `FakeRobot`
  gets seat objects with occupancy. HRI set only. Pointing waits for S6.
- **Test.** Two seats, one taken, in the fake world. In sim, a living-room
  scene with a sofa, a chair and one seated guest.
- **Default (delegated).** A dedicated atomic rather than
  `Vision.objects('seat')`, since occupancy needs people and seats together.
- **Risks.** One view may not show every seat (HRI Q2). Occupancy from box
  overlap is crude.

### S6: Arm poses and grippers in MuJoCo

- **Build.** Port `CommanderClient`, `arm/poses.py`, `arm/state.py`,
  `config/arm_poses.toml` and `config/robot_model.toml`. A `RosArms` backend
  runs behind `--arms`. The `Arms` protocol gains pose-level verbs (`poses`,
  `move_to(pose, arm)`, `gripper(arm, state)`, `state`). The atomics
  `list_arm_poses`, `move_arm_to_pose`, `set_gripper` and `get_arm_state` go
  in HRI and a new arms set, not GPSR. `RosBase.go_to` gets the travel-safe
  check of [ADR-0025](decisions/0025-travel-safe-interlock.md). The commander's
  error codes map onto `ErrorCode`, and `CancelToken` is bridged.
  `FORBIDDEN_NAMES` gains the pose-goal, whole-body and lift names and the
  four other velocity names. `pick`, `place_on` and `hand_over` stay
  unavailable on the robot path.
- **Test.** `FakeRobot` pose verbs and a gate rerun; the ported fake-commander
  integration tests; in sim, "raise your hands" and then "drive to the
  kitchen", which must be refused until the arms reach home or carry.
- **Defaults (delegated).** Pose verbs are exposed in the HRI and arms sets
  only. The walkie-ros pin is 6b26daa until the robot's commit is known
  (Manipulation Q2).
- **Team call.** Decision 12. The prototype builds only presets, which work
  under either answer.
- **Risks.** A commander restart mid-goal leaves a stale status, and its only
  fix (`wip/p14`) has never run. About a third of the ported tests are flaky
  in CI. Sim slower than real time strains the timeouts.
- **Core built** as S6-1 to S6-5: the poses, the arm state, the two TOMLs,
  `ros/commander.py`'s `CommanderClient` and `sim/fake_commander.py`. The
  104 commander tests passed in every run against the fake.
- **Second part built** as S6-6 to S6-13: the `Arms` verbs and records,
  `RosArms`, the four atomics in HRI and the new arms set, `--arms`, the
  drive check before every base motion, the `FORBIDDEN_NAMES` additions, the
  fake arm model, the commander's stop on every runtime stop, and the sim's
  `moveit` and `commander` parts. The gate rerun passed with episode logs
  identical to the previous code's but for the commit. The sim check below
  has not run yet, and nothing has run against the real commander.
- **Review fixes** as S6-14 and S6-15, from the Codex review of the second
  part: a failed start-pose read no longer hides the base's refusal of a
  turn, and shutdown closes the commander client only once no arm goal is
  outstanding, so a goal accepted after a stop is still cancelled. The gate
  rerun on the new digest passed with identical episode logs.

### S7: Take a bag, and hand an object over

- **Build.** A `receive_object` bundle over the S6 verbs: the receive pose,
  a wait for a still, open hand, close, a grip check (a gap near 0.044 m
  counts as not held), then carry. A `release_object` bundle, and
  `Arms.hand_over(person_id)` from `face_person`, the receive pose, open and
  home. HRI set.
- **Default (delegated).** A bundle rather than letting the model sequence the
  pose tools, so the safety order cannot be skipped.
- **Risks.** MuJoCo has no bag or person, so the grip check is not really
  tested in sim. `grasped` may miss bag handles (HRI Q5).

### S8: Places, furniture and told facts through walkie-database

- **Build.** Object, room and event verbs on `ctx.memory`: `query_objects`,
  `get_object`, `find_near`, `list_rooms`, `get_room_contents`,
  `remember_event` and `find_events`, over `db-memory` mode. Vectors, clouds
  and poses are stripped, and IDs are decimal strings (B2). `NOT_READY`
  becomes "not in memory yet", never "remembered".
- **Test.** In sim: `db-seed-sim` against the Chula map, `db-restore`,
  `db-memory` on sim time, then "what rooms do you know?" and "remember that
  the guest likes tea".
- **Default (delegated).** Default and memory tool sets only; GPSR waits for an
  agent never-list (P30).
- **Team call.** The walkie-database commit the robot runs.
- **Risks.** Object rows are only seeded anchors until M1. Events carry
  `person_id`, which puts them on the never-list.

### S9: Recall objects seen earlier in the task

- **Build.** A session sighting log: every `vision.objects` result with its
  place and time, cleared on `/walkie/reset` and at task end. A
  `where_seen(ctx, what)` atomic answers from it, labelled "seen this
  session", never as stored memory.
- **Default (delegated).** Allowed until walkie-database M1, labelled, and
  removed when M1 lands. M1 is unbuilt and undated.
- **Risks.** It bends [ADR-0035](decisions/0035-memory-through-walkie-database.md),
  which routes all memory through walkie-database, and overlaps M1's design.

### S10: Pick and place a known object in MuJoCo

- **Option A, first.** Sim-only pick and place bundles in `RosArms` behind
  `--sim-grasp`: check reach, move through presets, close the gripper, and
  move the object with MuJoCo's `set_free_joint_state`, the route
  [ADR-0044](decisions/0044-mocked-arm-atomics.md) planned. Results say the
  grasp is simulated.
- **Option B, a spike.** A real grasp with `go_to_pose` on `left_arm` in
  `base_footprint`, after one manual `go_to_pose` and a pose readback match.
  It needs an ADR-0021 and ADR-0044 exception, recorded only then.
- **Default (delegated).** The labelled teleport pick is acceptable for the sim
  demo, behind its flag, so GPSR runs end to end while real grasping is a
  separate spike.
- **Risks.** Option A overstates the robot unless labelled. No grasp has ever
  run in MuJoCo, and the hand-frame convention is unverified.

### S11: Keep looking at the person (continuous gaze)

Decision 14 comes first. The plan's default is a robot-side gaze node next to
`twist_mux`, with the agent sending only a target. If the agent-side loop is
chosen instead, the prototype accepts a named `/cmd_vel` exception in sim
only, with the arms in, per ADR-0051. Scored by `hri_check`'s gaze
criterion: within 10 degrees for 80% of the time.

### S12: HRI slice 1 end to end

Port the scenario driver and `hri_check`, reconcile the tool log format, and
rewrite the HRI task template with the demo's tool names. Repeated Chula runs
judged by `hri_check`. Until S11 lands, every criterion except gaze runs, and
gaze is reported separately. Whether to include the doorbell is the owner's
pick.

### S13: Describe a guest

A `describe_person` atomic: `GetPersonCrop` for colour shares and a crop, a
cloud VLM for enum answers, and the ported `attributes.py` rules, which say
nothing when the two disagree. The local fallback is colour shares only. The
cloud VLM counts as External Computing and must be declared to the Technical
Committee a month ahead; that call is the team's.

## Not planned

- **Following the host.** It needs person range from lidar leg clustering,
  synthetic legs in the sim scans and an unpushed navigation commit, and
  whether it is an agent loop or a robot-side skill is undecided.
- **Whole-body MoveIt.** Whole-body groups stay refused, and the team owns a
  competing planar-base design.
- **GraspNet.** Deferred (B31): its own CUDA environment and weights, and no
  launch file starts it.
- **walkie-database M1 and the scene graph.** A separate milestone. The agent
  switches to it when it lands, and S9 goes.
- **Open-vocabulary `DetectObjects` and `EmbedText` search in sim.** The sim
  stand-in is vision open question 7.
- **The experiment's stall nudge.** It hangs after two nudges.
- **Mahoraga on ROS or the robot.**
- **Pushing anything, or touching `main` in any repository.**
