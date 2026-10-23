# ADR-0051: Turn the base toward a person from an agent-side gaze loop

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

The HRI task scores gaze twice
([HRI Task](/architecture/hri.md#the-task)). While a guest gives their
name, the robot must keep looking at them as they pace left and right,
and points are given only for continuous tracking (2×50). While
introducing, it must look at the guest it is talking to (2×50). The head
servo only tilts, so looking at someone means turning the base.

Three rules stand in the way:

- New robot abilities start as walkie-ros changes
  ([ADR-0015](/architecture/decisions/0015-repository-boundary.md)). The
  robot-side home for a closed loop is a skill behind `run_skill`
  ([ADR-0027](/architecture/decisions/0027-runskill-action.md)), and the
  skill server is not built.
- No tool may touch `/cmd_vel`, because raw velocity commands bypass
  Nav2's obstacle avoidance
  ([ADR-0009](/architecture/decisions/0009-forbidden-names-guard-rail.md)).
- A turn through `move_relative` sends one Nav2 goal per correction and is
  untested in sim
  ([ADR-0031](/architecture/decisions/0031-bounded-relative-moves.md)).
  It cannot follow someone who keeps moving.

walkie-agent-v2 solved this with `FaceTracker` in `tasks/skills/people.py`.
One thread turned the biggest face into a map-frame heading, using the
robot's heading at the moment the image was taken. A second thread ran a
proportional loop at 15 Hz on the live heading: gain 1.2, speed between
0.15 and 0.6 rad/s, no command inside 6°, and a stop after 1 s without a
face.

walkie-ros runs the same `twist_mux` in sim and on the robot. Its inputs
are `/cmd_vel` at priority 5, `/cmd_vel_nav` at 10, and `/cmd_vel_joy` and
`/cmd_vel_key` at 100. All take `TwistStamped`, and each input is dropped
after 0.5 s without a message. Nav2's commands pass `velocity_smoother`
and `collision_monitor` before they reach `/cmd_vel_nav`. The other
inputs pass neither.

**Decision**

- A reviewed `GazeLoop` in walkie-agent-ros turns the base toward a
  target. The LLM starts it with `look_at(target)` and ends it with
  `stop_looking()`. Both have effect N.
- The target is a person handle, followed by its track id, or a seat or
  object handle, held as a fixed heading.
- The setpoint is a map-frame heading. It comes from the column of the
  face box, or of the person box when there is no face, through the
  message's `camera_info`, plus the camera's heading from TF at the
  message's stamp. This replaces v2's linear field-of-view estimate.
- The control loop is v2's, on the live heading from TF, with v2's values
  in config. It is capped at 0.6 rad/s, below the 1.0 rad/s that
  `velocity_smoother` allows Nav2. Its acceleration is capped at
  3.2 rad/s², the smoother's own limit, because nothing smooths this
  input.
- It publishes `geometry_msgs/TwistStamped` on `/cmd_vel`, the
  lowest-priority input of `twist_mux`. This is the only exception to
  `FORBIDDEN_NAMES`. It is allowed because the loop can only do what was
  judged safe:
    - **Rotation only.** The publisher sets every linear field to zero
      itself. Callers cannot pass a linear velocity.
    - **Nav2 and the operator win.** `twist_mux` prefers `/cmd_vel_nav`,
      the joystick and the keyboard. The loop also stays silent while a
      `navigate_to_pose` goal is active, and every navigation atomic
      stops it before sending a goal.
    - **Arms in.** It turns only while both arms are in a travel-safe
      pose, the check the navigation atomics make
      ([ADR-0025](/architecture/decisions/0025-travel-safe-interlock.md)).
      Every arm atomic stops it before moving an arm, and `look_at`
      refuses while an arm is out.
    - **It stops by itself.** It sends a zero command and goes silent
      when the target is unseen for 1 s, on `stop_looking`, on a reset,
      "stop" or "never mind" interrupt
      ([ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md)),
      and on shutdown. If the agent dies, `twist_mux` drops the input
      after 0.5 s.
    - **Reviewed code only.** `/cmd_vel` stays in `FORBIDDEN_NAMES` for
      every tool. No level puts `look_at` or `stop_looking` in
      `ctx.tools`, so generated tools cannot start it.
- The loop outlives the LLM turn. It keeps tracking the guest while
  `wait_for_person` waits for their answer.

**Consequences**

- Gaze works before any skill server exists. If it moves behind
  `run_skill` later, `look_at` keeps its interface.
- `/cmd_vel` bypasses `collision_monitor`, including its rotation polygon.
  A turn in place sweeps only the circle around the footprint, 0.30 m at
  its widest in walkie-ros's parameters, and guests stand at talking
  distance. Something inside that circle is not seen, though. Whether the
  loop should go through `collision_monitor` instead is open
  ([HRI Task](/architecture/hri.md#open-questions)).
- That circle holds only with the arms in. A held object can still stick
  out of it: in the HRI task, the bag hangs from the left hand in `carry`
  while the robot looks at seats and guests.
- During the HRI bag hand-over the robot stays still, since the arm is out
  ([HRI Task](/architecture/hri.md#the-bag-hand-over)).
- A wrong track turns the robot toward the wrong person, at no more than
  the cap, until the track is lost.
- The robot cannot drive and look at someone at once. While guiding a
  guest it faces where it drives, as Nav2 already does.
- When this is built, the `FORBIDDEN_NAMES` comment in
  `src/walkie_agent/tools/__init__.py` names the exception.
- As built, `look_at` takes person and seat handles, not object handles,
  and its arm check fails open as the drives' does: when the commander is
  not found, does not answer or gives no reading, the base turns. It
  refuses while a commander goal may still move an arm or a gripper, and
  on a reading that cannot show the arms are safe
  ([HRI Task](/architecture/hri.md#gaze)).
- As built, 0.6 rad/s and 3.2 rad/s² are hard ceilings in `gaze.py`,
  apart from the adjustable config, which may only go lower. The
  publisher clamps every command to 0.6 rad/s whatever its config says,
  and the acceleration cap counts the time since the last command went
  out, so a new target cannot jump the command.
- As built, one thread is the only writer of `/cmd_vel`, and it publishes
  outside the loop's lock. A stop queues its zeros and returns at once, so
  a publish that blocks in the transport holds up no stop and no
  callback. If the publish stays blocked, `twist_mux` drops the input
  0.5 s after its last message, and the drive controller stops the base
  0.5 s after its last command.
- As built, a person is lost 1 s after their last sighting, with no
  allowance for the `look_at` call. `look_at` refuses with `STALE` a
  handle whose sighting is already older than that, unless the newest
  people array shows the person. It also stops the loop and refuses with
  `STALE` when the sighting the loop starts from has passed that age by
  the time the loop takes it, as after a slow wait for TF. So it never
  answers "Looking at" for a person the loop has already lost.
- As built, every "stop" or "never mind" stops the loop, during a task
  or between tasks. Between tasks it interrupts nothing and waits as an
  ordinary message
  ([ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md)),
  but the loop outlives the turn. So the agent hands every such stop to
  the Toolbox's stop-heard listeners before it queues it, and the loop
  stops for `stop heard`.
<!-- --8<-- [end:body] -->
