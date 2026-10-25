# ADR-0053: Follow the host through Nav2's FollowPoint tree, ranged by the 2D lidar

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-25

**Context**

The HRI task scores following the host to the bag drop area (200) and
dropping the bag in the correct area (50)
([HRI Task](/architecture/hri.md#the-task)). The rulebook sets the order:

1. The robot tells the seated host that it has a bag and asks for
   guidance.
2. When the robot asks, the host stands in front of it. The task's
   overview adds that the host tells the robot to follow them.
3. The robot says when it is ready to follow.
4. The host walks naturally to the drop location and tells the robot to
   place the bag on the floor.

The score sheet takes 50 points for dropping the bag while following, 50
for rediscovering the host by natural interaction, 50 for asking the host
to wait, and 150 for guiding the robot by physical contact.

Following is a closed loop at sensor rate, which an LLM turn cannot run
([Perception question 5](/architecture/perception.md#open-questions)). The
robot-side home for a closed loop is a skill behind `run_skill`
([ADR-0027](/architecture/decisions/0027-runskill-action.md)), and the
skill server is not built. Gaze met the same problem with a reviewed loop
in the agent
([ADR-0051](/architecture/decisions/0051-agent-side-gaze-loop.md)).

Nav2 ships a tree for following, `follow_point.xml`:

- Once a second it takes the newest goal from `/goal_update`, plans to it,
  and cuts the last 1.0 m off the path, turning the end pose toward the
  goal. `FollowPath` runs until it fails, and there is no recovery branch.
- The goal updater ignores a `PoseStamped` with a zero stamp, or one
  stamped before the `navigate_to_pose` goal. It uses the last one
  received, and keeps it into the next goal on the same tree.
- `bt_navigator` registers every tree in its default directory. walkie-ros
  sets no `bt_search_directories`, so a goal can name the tree by its ID.
  A goal can also name a tree file by its path.
- The tree's `FollowPath` names no goal checker. walkie-ros loads two,
  `general_goal_checker` and `precise_goal_checker`, in sim and on the
  robot. The controller server accepts an unnamed checker only when one is
  loaded, so it rejects every `FollowPath` from the stock tree, and the
  goal fails at once. walkie-ros's own trees name their checker.
- Nothing has run the tree on walkie-ros's stack yet.

Facts that shape the design:

- **Where the host is.** `PersonDetection` has a box and keypoints but no
  depth, and walkie-vision has not written its depth cut
  ([Perception question 1](/architecture/perception.md#open-questions)).
  The Unitree L2 is mounted upside down and tilted to the rear. Straight
  ahead it sees only what lies more than about 28° below the horizontal.
  The front and back 2D lidars sit about 0.3 m above the floor, at leg
  height. In sim each covers ±120° at 1°. walkie-ros merges them into
  `/scan`, the scan Nav2 uses.
- **Who the host is.** Face and appearance vectors come through
  `VisionFeed`, which keeps the newest 20 vectors of a track: about 4 s at
  the sim camera's 5 Hz, and 1.3 s at the robot's 15 Hz. walkie-vision is
  stateless apart from tracker ids (walkie-vision ADR-0002), so a lock on
  one person cannot live there. Slice 1 never enrolls the host, so
  `recognize_person` reports them as unknown.
- **The base.** Nav2 caps the base at 0.3 m/s in x and y, in sim and on the
  robot. Its progress checker fails a goal that moves the base less than
  0.5 m in 8 s. A goal that names a different tree cannot preempt a
  running one. Nav2 accepts it, then aborts it at once.
- **Waiting.** `wait_for_person` always says its question out loud, marks
  the step waiting and ends the task. The answer arrives as the next
  message. Without a waiting step, the completion gate pushes the LLM to
  act on every open item
  ([The Agent](/architecture/agent.md#the-agents-graph)).
- **The simulation.** The scenario's people exist only in
  `oracle_perception`. They have no MuJoCo bodies, so neither lidar, the
  depth image nor Nav2's costmaps see them, and the oracle hides no one
  behind anything. The oracle places people from the camera's TF pose,
  which AMCL puts 0.1 to 0.2 m off the ground truth.
- **The capability layer.**
  [ADR-0052](/architecture/decisions/0052-typed-robot-capabilities-on-ctx.md),
  Proposed, puts a typed `ctx.base` in front of Nav2. Its first change adds
  the types and fakes. Its second moves the navigation atomics' Nav2 goals
  behind `ctx.base` and makes a reset cancel a drive. A third adds the
  MotionSupervisor.

**Decision**

- **One loop, in the agent.** A reviewed `FollowLoop` in walkie-agent-ros
  follows one person. It is `ctx.base.follow`, built after the capability
  layer's second change, so the stops in the navigation atomics are
  written once. The LLM starts it with `follow_person(person, line)` and
  ends it with `stop_following()`.
    - `follow_person` has effects N and S. It says its line through the
      speak tool, as `wait_for_person` does.
    - `stop_following` has effect N.
    - No level puts either in `ctx.tools`, so generated tools cannot start
      the loop.
- **Nav2 drives.**
    - The loop sends one `navigate_to_pose` goal that names a copy of the
      FollowPoint tree. The copy's `FollowPath` sets
      `goal_checker_id="general_goal_checker"`. It lives in walkie-ros's
      `robot_navigation`, beside walkie's other trees, and the goal names
      it by its ID
      ([HRI question 9](/architecture/hri.md#open-questions)).
    - It publishes the host's map position on `/goal_update` at 2 Hz,
      twice the tree's rate.
    - The goal and every update are stamped with the node clock, which is
      sim time in sim. An update left from an earlier follow is then older
      than the goal and is never used.
    - The costmap may mark that position occupied. On the robot it will,
      once the host's legs are in `/scan`. Then the goal moves to the
      nearest free cell toward the robot.
    - The robot stops about 1.0 m short of the goal, the tree's own cut.
    - The loop never publishes a velocity. `/goal_update` and
      `/goals_update` join `FORBIDDEN_NAMES`, and the loop is the only
      code allowed to publish on `/goal_update`.
- **The lidar says where, the camera says who.**
    - Each beam of the scan is compared with the range cast through `/map`
      from the lidar's pose at the scan's stamp. A return shorter than the
      map's by more than a margin that grows with range is foreground. The
      margin allows for AMCL's error in that pose, in sim and on the
      robot.
    - Points near walls are not deleted, because the host stops by a wall
      or a counter to have the bag put down.
    - Foreground points are clustered into legs, and leg pairs into
      people. The clustering's thresholds are in metres, not point counts,
      so they hold for both the sim's and the real lidars' steps.
    - The loop tracks one target, in the map frame, within a gate around
      its predicted position.
    - A track starts only where a camera detection's bearing meets a
      cluster, at the detection's stamp. Furniture missing from the map,
      such as the sofa and the table legs, also shows as foreground.
    - In sim the oracle places people from the TF pose and the leg node
      from the ground truth, so a detection's bearing and its cluster
      differ by AMCL's error, up to about 0.2 m. On the robot they agree.
      Either the bearing gate allows for it, or the leg node casts from
      the TF pose, as the oracle does.
    - The camera keeps checking the lock. A face or appearance that does
      not match the host's ends it.
    - At the lock, the loop copies the host's vectors from the people
      store and from the live track, so its gallery does not age out with
      `VisionFeed`.
    - The L2 is not used.
- **The host is enrolled, then refreshed.**
    - At the first seat offer, the template has the LLM call `find_people`
      and `remember_person` on the one seated person, under the name
      `host`, as walkie-agent-v2 does
      ([HRI question 8](/architecture/hri.md#open-questions)).
    - When the host says to follow, the agent finds them with
      `find_people(in_front=True)` and confirms them with
      `recognize_person`.
    - `follow_person` adds their standing appearance to the gallery.
- **A lost host is searched for in silence.**
    - When the track is lost, the loop cancels the goal and the base holds.
    - The tracker keeps watching the whole scan. A track that reappears
      near the prediction within 20 s resumes the goal. If the camera sees
      it, its face or appearance must match.
    - The robot says nothing meanwhile.
    - Only when the 20 s run out does the loop end and the LLM hear of it
      ([HRI question 7](/architecture/hri.md#open-questions)).
- **Talk during the follow.**
    1. With `wait_for_person`, the robot tells the host that it has their
       bag, asks where it goes, and asks them to stand in front of it and
       say when they are ready.
    2. The host's reply, such as "I am here, follow me", resumes the step.
       The agent finds and confirms them.
    3. `follow_person(person, line)` locks onto the host, says `line`, the
       robot's ready line, and marks the current step waiting. The graph
       ends the task after it, as after `wait_for_person`. The loop
       outlives the task, as the gaze loop does.
    4. A line heard while following arrives as the reply. If it neither
       says where to put the bag nor says to stop, the LLM calls
       `follow_person` again on the same person. That call says nothing,
       keeps the loop running and marks the step waiting again.
    5. On "put it here", the LLM calls `stop_following()`. It cancels the
       goal and returns once the base is still. The LLM says it is putting
       the bag down, then calls `set_gripper("left", "open")`. The bag
       falls from the `carry` pose to the floor.
- **Stops.**
    - The loop stops on `stop_following`, a reset, a new task, "stop" or
      "never mind"
      ([ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md)),
      and shutdown, through the toolbox's listeners, as the gaze loop does.
    - Every navigation and arm atomic stops it first and waits for the base
      to be still. `look_at` refuses while it follows.
    - A watchdog cancels the goal when the scan or TF is more than 0.5 s
      old, or the people feed is older than a limit set from
      walkie-vision's measured latency on the robot. The loop then holds
      as for a lost host, resumes once every feed is fresh, and counts the
      time toward the same 20 s.
    - When Nav2 aborts the goal, the loop sends it again. Repeated aborts
      end the loop as a timeout does, after a count the smoke test sets.
    - It starts only with both arms in a travel-safe pose, the check the
      navigation atomics make
      ([ADR-0025](/architecture/decisions/0025-travel-safe-interlock.md)).
      `carry` is one.
- **Closing in.** Nav2 aborts a goal that switches trees. A drive closer
  than the tree's 1.0 m is therefore a separate goal, sent after the
  cancel.
- **The simulation, in this repository.**
    - The scenario scripts the host. Asked for guidance, the host stands up
      and walks to a free spot in front of the robot. There they say a line
      such as "I am here, follow me." After the ready line they lead
      through the gap in the divider into the kitchen, stop, and say to
      put the bag down.
    - The host takes a step only while the robot is within the set gap and
      the step keeps their legs in the line of sight of a robot lidar.
      Otherwise they wait where they are, still visible.
    - After the introductions, the scenario routes the robot's questions
      to the host's script, not to a guest. The guidance request is
      answered only by the host's line once they stand in front.
    - **Legs in the scan.** A node reads `/front_lidar` and `/back_lidar`
      at their native 1°. In each scan it shortens every beam, cast from
      the lidar's ground-truth pose, that meets one of a scenario person's
      two leg circles nearer than the beam's own return. Walls, furniture
      and the divider stay in, and they hide the legs behind them. The node
      merges the two scans into `/sim/scan_people`, the way walkie-ros's
      `dual_laser_merger` builds `/scan`.
    - The legs come out as sparse as the sim's lidars would see a body, at
      1°. Cast into the merged `/scan`, at 0.001 rad, they would be about
      17 times denser. The real lidars' step is not recorded. The robot
      bags show how dense real legs are.
    - The loop reads `/sim/scan_people` in sim and `/scan` on the robot.
      Nav2 keeps reading `/scan`.
    - The event log gains the host's walking state and map pose in the
      10 Hz state record, and a `host_line` record for each line the host
      says unprompted.
    - `hri_check` gains these criteria, measured from the ground truth,
      not the TF pose:
        - the host's route is long enough that a robot that never moves
          fails;
        - while the host walks, the robot moves and stays within a set
          gap of them;
        - the base never comes within 0.5 m of the host;
        - the gripper opens only after the host's line to put the bag
          down, with the base still and within 1.5 m of where the host
          stopped;
        - no line during the follow asks the host anything.
    - The question criterion also allows the one request for guidance to
      the host, and `follow_person` joins the tools whose lines
      `hri_check` counts.
- **Order of work.**
    - First, a smoke test of the tree on the sim stack, before any loop
      code. It names a scratch copy of the tree by its path, so it needs no
      walkie-ros change. It sends stamped updates, stops the target for
      10 s, stops the updates, then cancels and sends an ordinary goal.
    - The test settles:
        - whether the copied tree loads and its `FollowPath` is accepted;
        - how the 1 Hz update and the 1.0 m cut behave;
        - whether the progress checker fails a goal behind a standing host,
          with `general_goal_checker`'s 0.01 rad yaw tolerance;
        - whether a goal whose updates stop stays active;
        - how many aborts in a row end the loop;
        - whether the base faces where it drives.
    - In parallel, the team records robot bags of a person walking ahead of
      the robot, beside a wall, and across its path. The leg clustering is
      tuned on those real returns.

**Consequences**

- Nav2's planner, `velocity_smoother` and `collision_monitor` stay in the
  path. Nothing new publishes a velocity, so
  [ADR-0009](/architecture/decisions/0009-forbidden-names-guard-rail.md)
  and `check_cmd_vel` stand.
- The follow loop becomes the second exception to `FORBIDDEN_NAMES`, for
  `/goal_update` alone. When it is built, the comment on
  `FORBIDDEN_NAMES`, ADR-0051's "only exception" and the "one exception"
  lines in HRI Task and The Agent name both loops.
- The tree replans once a second and stands off 1.0 m. Both can be tuned in
  the copy.
- At 0.3 m/s the robot is slower than a host who walks naturally. In sim
  the host waits for it. On the robot the host may pull ahead and be lost
  behind the divider. A faster follow profile is a walkie-ros question for
  after the first pass.
- A host who stops just beyond the standoff can make the progress checker
  fail the goal, and the tree has no recovery. The loop then sends the goal
  again. The smoke test decides whether that is enough.
- If the agent dies, nothing cancels the goal. The base stops about 1.0 m
  short of the last update. A `FollowPath` that reaches its end starts
  again instead of ending the tree, so the goal most likely stays active
  until it is cancelled or the planner or controller fails. While it is
  active, Nav2 aborts every goal on another tree. `ctx.base` therefore
  cancels any active `navigate_to_pose` goal at start-up and before each
  drive of its own.
- In sim Nav2 does not see the people, so only the loop's standoff keeps
  the robot off the host, and the 0.5 m criterion judges the loop alone.
  On the robot the host's legs are in `/scan`, so the costmaps and
  `collision_monitor` see them too.
- One target is tracked. While the host is out of the camera's view, a
  person crossing close to the prediction can take the lock. These wait
  until a plain run passes:
    - tracking everyone;
    - a clutter layer learned from the scans;
    - occlusion in the oracle;
    - a look-alike and view-dependent appearance in the scenario;
    - person bodies in walkie-ros's MuJoCo world.
- Slice 1's template keeps the bag in `carry` until the end and ends the
  task with the introductions. Slice 2 changes both.
- `follow_person` doubles as the follow's silent wait, and the graph ends
  the task after it as after `wait_for_person`. Outside a follow, a step
  still cannot wait without the robot speaking.
- The atomic shares its name with the gaze loop's method,
  `GazeLoop.follow_person()`. The docs name the method in full.
- Following works before any skill server exists. If it moves behind
  `run_skill` later, `follow_person` keeps its interface, as `look_at`
  does.
<!-- --8<-- [end:body] -->
