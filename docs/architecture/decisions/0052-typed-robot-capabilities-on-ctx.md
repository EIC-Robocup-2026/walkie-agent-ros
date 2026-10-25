# ADR-0052: Reach the robot through typed capabilities on ctx, with no velocity verb

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-26

**Context**

Teammates write atomics, and most of them do not know ROS 2. Today an
atomic that drives or speaks has to rebuild what the LangChain tools do, or
call those tools:

- Driving and speaking have no backend. They live only in `Nav2GoalTool`,
  `MoveRelativeTool` and `SpeakTool` (`src/walkie_agent/tools/navigation.py`,
  `src/walkie_agent/tools/speech.py`).
- Three backends have call surfaces free of ROS types: `people_memory`,
  `arm` and `gaze`. `vision` hands `walkie_vision_msgs` arrays and ROS stamps
  to the atomics. `ctx.backend()` returns `Any`, and `ctx.tf_buffer` is
  RAI's private `_tf_buffer`.
- The robot's yaw is read from TF in five places, with four staleness rules.
- Before a drive, the two drive tools share these steps in
  `Nav2GoalTool._navigate`: stop the gaze loop and wait for its zero
  commands ([ADR-0051](/architecture/decisions/0051-agent-side-gaze-loop.md)),
  check the arms
  ([ADR-0025](/architecture/decisions/0025-travel-safe-interlock.md)), then
  send the goal. Only `move_relative` reads a pose stamped after the call
  ([ADR-0031](/architecture/decisions/0031-bounded-relative-moves.md)), and
  it reads it first, before `_navigate` runs.
- A reset does not cancel a drive. The navigation tools do not see
  `ctx.cancel`. On their own timeout they send a cancel and do not wait for
  its answer
  ([ADR-0010](/architecture/decisions/0010-bounded-navigation-tool.md)).

Earlier decisions bound the answer:

- [ADR-0016](/architecture/decisions/0016-talk-to-walkie-ros-directly.md)
  retired walkie-sdk, partly because a wrapper "would also hide the goal
  statuses and cancel handling the agent needs".
- [ADR-0007](/architecture/decisions/0007-speak-tool-and-console.md) has the
  agent speak only through the `speak` tool, which keeps the one speech ID
  that RAI's text-to-speech accepts.
- [ADR-0023](/architecture/decisions/0023-motion-supervisor.md), Proposed,
  plans a `MotionSupervisor` that owns every motion goal and stops MoveIt on
  a reset. It is not built: `src/walkie_agent/motion.py` does not exist.
  Manipulation's real-backends track lists it as the gate for `--arms` on
  the robot.
- [Access Levels](/architecture/access-levels.md) plans, for phase 4b and
  only on harvest evidence, raw base-command publishing at 20 Hz for
  generated Unrestricted code in sim.

A raw velocity verb, such as `ctx.nav.send_vel`, does not fit walkie-ros's
base:

- `twist_mux` drops an input 0.5 s after its last message. One call cannot
  hold a velocity. Holding one takes a control loop, as the gaze loop runs
  at 15 Hz.
- Only Nav2's own output passes `velocity_smoother` and `collision_monitor`
  on its way to `/cmd_vel_nav`. `/cmd_vel` and the joystick and keyboard
  inputs bypass both.
- `/cmd_vel` already belongs to the gaze loop. All five base-velocity names
  are in `FORBIDDEN_NAMES`.

An audit of the robot found four more facts:

- **Head tilt.** walkie-ros's two real Nav2 launches start
  `head_tilt_near_goal_node.py`. The sim launch does not.
    - The node reads `navigate_to_pose` feedback. On each new goal it sets
      the ZED head to 0.25 rad.
    - Once a goal's remaining distance has been above 1.5 m, it tilts the
      head down to 0.785 rad (45°) when the distance drops to 1.5 m.
    - It raises the head again once the distance is back at 1.7 m, or 2 s
      after the feedback stops. It ramps at 0.6 rad/s. It ignores distances
      of 0.3 m or less, so turns in place never tilt the head.
    - The tilt was made for STVL, which marks tabletops from the head
      camera. STVL is disabled on the robot, because the head camera joint
      is broken. Today the tilt only changes what the ZED sees.
    - Commit `22dc4ef` in walkie-ros's `navigation` submodule adds a
      `head_tilt_near_goal` launch switch, default `true`. With `false` the
      node does not start, and nothing commands the head. The commit is on
      the navigation repository's branch `feat/hri-follow-and-head-tilt`,
      not yet pushed. walkie-ros picks it up with a submodule bump.
- **Orphaned goals.** A follow goal on
  [ADR-0053](/architecture/decisions/0053-follow-through-nav2-followpoint.md)'s
  tree stays active after its loop dies. While it is active, Nav2 accepts a
  goal on any other tree and then aborts it at once, and the gaze loop
  stays silent.
- **Exceptions.** The gaze loop's `/cmd_vel` publisher is the one exception
  to `FORBIDDEN_NAMES` so far. The follow loop's `/goal_update` will be the
  second (ADR-0053).
- **The commander.** The arm atomics call `bimanual_commander` through
  `CommanderClient`
  ([ADR-0021](/architecture/decisions/0021-arm-control-through-commander.md)).
  In sim this is the real commander in walkie-ros's MuJoCo simulation.
  `FakeCommander` serves only the tests.

A turn in place is a `navigate_to_pose` goal at the current position, and
no one has tested it in Nav2 (ADR-0031). The sim and the robot load the
same default tree, `nav_to_pose_real_recovery_bt.xml`.
`general_goal_checker` accepts 0.1 m and 0.01 rad.

**Decision**

- **Six typed capabilities on ctx.** `ToolContext` gains these members. Each
  is bound to `ctx.cancel`, so no atomic passes a cancel event. A missing one
  raises `UNAVAILABLE`, as `ctx.backend()` does.
    - `ctx.base`, effect N: `locations()`, `go_to(location)`,
      `move_relative(forward_m, left_m, turn_deg)`, `turn_by(degrees)`,
      `turn_to(heading_rad)`, `stop()` and `busy()`. The name `follow` is
      reserved for ADR-0053's loop, which sets its arguments.
    - `ctx.speech`, effect S: `say(text)`.
    - `ctx.pose`: `latest(max_age_s)`, `fresh(timeout_s)` and
      `at(stamp_ns, timeout_s)`. Each method names its staleness rule.
    - `ctx.arms`, effect M: `poses()`, `state()`, `move_to(pose, arm)` and
      `gripper(arm, state)`, with `open`, `half` or `close`.
    - `ctx.gaze`, effect N: `look_at(handle)`, `stop()` and `status()`.
    - `ctx.vision`, effect P: `people(max_age_s)` and
      `objects(labels, max_age_s)`. They return typed records with handles,
      never `walkie_vision_msgs`.
    - Plain members: `now_ns()`, `in_simulation`, `sleep(seconds)` and
      `wait_until(ready, timeout_s, what)`.
- **Where they live.** The Protocols and records go in
  `src/walkie_agent/capabilities/`, which imports no rclpy. The records it
  reuses today live in modules that load rclpy, so `Pose2D`, `GazeStatus`,
  `GripperResult` and the commander's `Recovery` move into it, and their
  old modules re-export them. Each ROS implementation sits beside the code
  it comes from. They exist only in the agent process, on the in-process
  `ToolContext` of human-written atomics.
- **Generated code gets none of them.** It reaches M only through atomics
  in `ctx.tools`, as
  [ADR-0039](/architecture/decisions/0039-access-levels.md) sets. It
  reaches N through atomics, and at Unrestricted also through the ROS
  bridge's raw Nav2 actions, which ask the supervisor once it exists.
- **What they hide:** message types, goal handles and futures, quaternions,
  frame names, the locations file, the TF and discovery waits, the speech ID,
  MoveIt groups, joint values and track keys.
- **What they do not hide:**
    - **Goal statuses.** Every failure is a `StepError` with a fixed code. Its
      `data` carries Nav2's status name, `error_code` and `error_msg`, or the
      commander's codes as `CommanderClient` reports them. A drive's result
      keeps the arm check's notes, such as an arm state that could not be
      read.
    - **Cancel.** Every wait polls `ctx.cancel`, and every cancel's answer
      is reported in `data`. On a cancel or a timeout, `ctx.base` cancels
      its goal, waits up to 1 s for the answer and up to 5 s for the goal to
      end. `ctx.arms` keeps `CommanderClient`'s rule: it waits up to 1 s for
      the cancel's answer and returns. The goal stays outstanding until it
      ends, so later arm calls are refused, and the arms may become
      unverified. `ctx.base.stop()` stops the gaze and follow loops and
      cancels the base's goal. It never raises.
    - **Unverified arms,** and walkie-ros faults that need an owner, such as
      a sim-time mismatch. The implementations log them.
- **The base's safety order, written once.** Every base verb:
    1. validates first: a known location, finite values, no empty move, at
       most 3 m (ADR-0010, ADR-0031);
    2. stops the agent's own loops: the gaze loop, waiting up to 0.5 s for
       its zero commands (ADR-0051), and later the follow loop, waiting for
       its goal to end and the base to be still (ADR-0053);
    3. allows one base goal at a time. The follow loop's goal does not
       count here, since step 2 stopped it;
    4. cancels every other active `navigate_to_pose` goal, from any
       client, and waits up to 5 s for them to end (ADR-0053);
    5. checks the arms with `ctx.cancel`, keeping the check's code
       (ADR-0025);
    6. for `move_relative`, `turn_by` and `turn_to`, takes the start pose
       from TF stamped after the call (ADR-0031). `go_to` needs none;
    7. sends one goal and waits for it, bounded.
- **At start-up,** `ctx.base` cancels every active `navigate_to_pose` goal
  (ADR-0053). An arm or gripper goal left by an agent that died is not
  cancelled. The supervisor sees it on the commander's status topics and
  refuses drives until it ends.
- **Turns are Nav2 goals.** `turn_by` and `turn_to` send one goal at the
  current position, as `move_relative`'s turn does. `turn_to` takes a map
  heading. An atomic may pass one from the LLM. That is a new kind of input,
  though it reaches no heading that `turn_deg` cannot. The guide does not
  teach turns until the sim check of turns in place passes.
- **No velocity verb.** No capability and no reviewed atomic publishes a
  velocity. A closed loop that needs one is a reviewed component with its
  own exception to `FORBIDDEN_NAMES`, as the gaze loop is. A velocity verb
  waits for a collision-checked input in walkie-ros. It would then be a
  bounded session with a deadman, speed caps and a duration, not a raw
  publish.
- **The head stays the robot's.** The agent never commands the head.
  Robot runs launch walkie-ros's Nav2 with `head_tilt_near_goal:=false`.
  `ctx.base` does not wait for the head. It logs a warning at start-up when
  the node is on the graph.
- **Fakes.** `walkie_agent.testing` provides `fake_ctx()`: in-memory fakes of
  the six Protocols, a log of the capability calls, scripted faults and a
  fake clock, passed to `ToolContext` as its clock. Contract tests run the
  same cases on the fakes and on the ROS implementations over `FakeWalkie`
  and `FakeCommander`
  ([ADR-0019](/architecture/decisions/0019-test-fakes-fast-dds.md)). They
  check the safety order on both sides. Atomic tests check only the calls
  the atomic makes. The fakes are test code, never registered as backends.
- **The MotionSupervisor sits beneath `ctx.base` and `ctx.arms`.** It is a
  separate change, after `ctx.base` drives. It owns every base, arm, gripper
  and, later, skill goal.
    - **The agent's loops stop first.** An arm or gripper goal stops the
      follow loop and waits for the base to be still (ADR-0053). An arm
      goal also stops the gaze loop, as `move_arm_to_pose` does today. A
      gripper goal does not, as `set_gripper` does not (ADR-0051). The
      follow change adds the follow loop's stop to `ctx.base` and
      `ctx.arms`, and the supervisor takes that hook over.
    - **One goal at a time.** It then refuses an arm or gripper goal while
      any `navigate_to_pose` goal is active, whoever sent it. It refuses a
      base goal while an arm, gripper or skill goal runs, whoever sent it,
      or while the arms are unverified. It sees other clients' commander
      goals on the `_action/status` topics of `/set_joint_position` and
      `/control_gripper`, as the gaze loop watches Nav2's.
    - **The latch is `ctx.cancel`.** A reset, or a "stop" or "never mind"
      during a task, sets it
      ([ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md)).
      `begin_task()` clears it when the next task starts. While it is set,
      every capability refuses motion at once, so the other tool calls of
      the same LLM answer do not move the robot.
    - **On every interrupt, and every "stop" or "never mind" between
      tasks,** it:
        1. stops the gaze and follow loops, so neither sends again;
        2. cancels every `navigate_to_pose` goal with one request for all
           goals, a zero goal ID and a zero stamp;
        3. cancels the agent's arm, gripper and skill goals, including goals
           sent but not yet accepted, through a public cancel that
           `CommanderClient` gains;
        4. publishes `"stop"` on `/trajectory_execution_event`, and repeats
           it every 200 ms until the cancelled goals end, at most 5 s.
    - **Listeners return at once.** A thread repeats the stop. The waiting
      tool wakes through `ctx.cancel` and reports "Stopped by reset".
    - **Every wait is bounded.** A base goal's timeout cancels it and waits
      up to 5 s for it to end. An arm goal keeps `CommanderClient`'s rule,
      above. The 5 s bound of the interrupt is on the repeated stop, not on
      the arm call.
    - `CommanderClient` keeps its call lock, its unverified state and its
      recovery. The supervisor adds the exclusion between base and arms
      above it.
    - `/trajectory_execution_event` joins `FORBIDDEN_NAMES`. The
      supervisor's publisher is a reviewed exception, as the gaze loop's is.

This changes five earlier decisions:

- **Amends ADR-0016.** New atomics reach walkie-ros through the
  capabilities, not through their own entities. Existing atomics move over
  as they are ported. The capabilities are in-process code in this
  repository and call walkie-ros directly. They pass goal statuses and
  cancels through, as above. walkie-sdk stays retired.
- **Amends ADR-0007.** The agent speaks only through one `SpeechClient`,
  made from `SpeakTool`'s publisher, discovery wait and speech ID. `speak`,
  `wait_for_person`, `follow_person` and `ctx.speech` share it. It holds
  the one speech ID that
  [ADR-0045](/architecture/decisions/0045-play-sound-through-to-human.md)'s
  `play_sound` and `fetch_sound` also share.
- **Amends ADR-0039 and ADR-0047.** Phase 4b no longer adds raw
  base-command publishing. As with the raw arm controllers
  ([ADR-0044](/architecture/decisions/0044-mocked-arm-atomics.md)), it is
  reconsidered only if the harvest's atomic wishlist shows tools blocked
  without it. The answer would then be a reviewed `ctx.base` verb on a
  collision-checked input, not a raw publish. No level publishes on a base
  topic, so ADR-0039's draft that publishes `/cmd_vel` at Unrestricted is
  refused rather than stamped. With no publishes, ADR-0047's bridge cleanup
  sends no stop messages. ADR-0044's refusal of raw arm controllers stands.
  The bridge it cites, which knows stop messages only for `Twist` and
  `TwistStamped`, now sends no stop messages at all.
- **Supersedes ADR-0023.** The MotionSupervisor above replaces it. It keeps
  ADR-0023's one-goal rule, the repeated MoveIt stop and the bounded waits
  of base goals. Arm goals keep `CommanderClient`'s cancel rule. It moves
  beneath the capabilities. Its latch becomes `ctx.cancel`, set on every
  interrupt, not on a reset alone. It cancels every Nav2 goal, from any
  client. It refuses arm motion during anyone's drive, and drives during
  anyone's arm goal. ADR-0023 keeps its Proposed status until this ADR is
  accepted.

**Consequences**

- A teammate writes an atomic that drives, speaks, reads the pose, moves the
  arms, turns toward a person and reads detections, with no rclpy and no
  message types. Capability owners still need ROS.
- A reset cancels a drive. Nav2's recovery behaviours may race the cancel.
  The sim check of a reset mid-drive covers it.
- Every drive first cancels whatever else drives the base, an operator's
  goal included. A "stop" heard between tasks does the same. A person who
  says "stop" means the robot.
- A new verb needs a reviewed method first. Methods are added rather than
  changed, because a signature change touches every atomic and fake.
- `move_relative` reads its start pose after the gaze stop and the arm
  check, not before them. The pose is fresher, and a refused drive reads
  no TF.
- While the agent is down, a Nav2 drive runs on to its goal, and a
  commander goal runs to its end. At start-up `ctx.base` cancels the drive.
  The supervisor refuses drives until the arm goal ends.
- The fakes can drift. The contract tests are not optional.
- Tests on the fakes need no DDS traffic. They still run under the suite's
  `rclpy.init`, because `ToolContext` imports rclpy.
- The import cycle around `walkie_agent.tools` stays. Breaking it needs
  `FORBIDDEN_NAMES` moved, and
  [ADR-0009](/architecture/decisions/0009-forbidden-names-guard-rail.md)
  places it there.
- A turn in place may fail in Nav2. If the sim check fails, the fallback is
  Nav2's `spin` behaviour. Its output also passes `velocity_smoother` and
  `collision_monitor`, but it is a second action, which the supervisor and
  the gaze loop would have to watch.
- With the head tilt off, the ZED stays level near goals, so `find_people`
  and `find_seats` see people on arrival. If walkie-ros turns STVL back on,
  the choice reopens: the costmap then wants the tilt on approach.
- The MoveIt stop ends every MoveIt execution on the robot, including other
  programs'. It is best effort. The physical emergency stop stays the
  safety boundary. A gripper still finishes its stroke (walkie-ros change
  W3), and the lift cannot be stopped this way
  ([ADR-0026](/architecture/decisions/0026-exclude-lift.md)).
- Three reviewed components will publish on forbidden names: the gaze loop
  on `/cmd_vel`, the follow loop on `/goal_update`, and the supervisor on
  `/trajectory_execution_event`. The comment on `FORBIDDEN_NAMES` names each
  one when it lands, as do ADR-0051's "only exception" and the "one
  exception" lines in HRI Task and The Agent.
- The ROS bridge's raw Nav2 actions at Unrestricted bypass `ctx.base`. Once
  both exist, the bridge asks the supervisor before it starts one.
- A gripper goal during anyone's drive is refused once the supervisor
  lands. Today `set_gripper` does not look at Nav2.
- After the supervisor and Manipulation's robot steps R0 to R4, the agent
  can run with `--arms` on the robot.

**See also:** [Robot Capabilities](/architecture/capabilities.md)
<!-- --8<-- [end:body] -->
