# ADR-0059: Wait with a still, open hand before closing on a hand-over

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-27

**Context**

The HRI task takes the second guest's bag at the door
([The bag hand-over](/architecture/hri.md#the-bag-hand-over)). The earlier
flow asked "Please hang your bag on my hand, and tell me when it is
there." with `wait_for_person`, closed the gripper after the answer, and
asked once more after an empty close.

Rulebook 2026.2 sets the conditions for the full 50 points of the
hand-over (T:101-107) and a reduction of 25 points for requesting
hand-over assistance (S:19):

- The robot should ask for the bag to be handed over (T:47).
- Asking the guest for help during the hand-over reduces the score
  (T:101). A guest who says when the bag is there times the close, which
  may read as that help.
- The robot shows that it is ready (T:103), with the gripper within 10 cm
  of the guest's hand (T:104) and no dangerous movement toward the guest
  (T:105).
- T:106's penalized example is "Place the item in my gripper, I'll close
  it in 3…2…1". T:107 allows a safety cue that gives an interval, "I will
  close my gripper in 3 seconds".
- From the second request for help on, the referee may stop the task
  (G/ContinueRules:72).

The robot cannot see the bag arrive with what the agent reads today. The
commander measures a grasp only on a close, so grasp detection cannot
start one. walkie-agent-v2 waited for joint 4's effort to change by 0.4,
after asking the guest to press down on the hand. That needs efforts from
`/joint_states`, which `/get_joint_states` and the MuJoCo simulation do
not give. `ctx.arms` gave no way to read a finger without sending a goal:
only `get_arm_state` reads one, through `CommanderClient` directly.

`speak` returns once the line is published, not once it has played
([Against rulebook 2026.2](/architecture/hri.md#against-rulebook-20262)).
A close sent right after the cue would start while the cue still plays.

**Decision**

The robot detects the hand-over itself: it asks once, holds the open hand
still for a set time, announces the close with an interval, closes, and
judges the close from its own read. It never asks whether the bag is
there.

1. **A read-only atomic, `wait_for_handover(arm="left")`** (effect R):
    - It reads the arms with `ctx.arms.state(read=True)` and the finger
      with `ctx.arms.finger(arm)`, a method added to the `Arms` Protocol
      ([ADR-0052](/architecture/decisions/0052-typed-robot-capabilities-on-ctx.md)).
    - It refuses with PRECONDITION_FAILED unless the arm is in `receive`
      and the finger travel is at least the open position less
      `holding_margin_m`, 0.040 m in the simulation. The message names
      the fix: move the arm to `receive` first, or open the hand first.
    - It sends no goal and says nothing. It holds for `HOLD_S` = 5 s, a
      module constant rather than an argument, so the LLM never chooses
      the hold. The hold runs on `ctx.sleep`, which ends at once on a stop
      or a reset with CANCELLED. It is wall time, so in the simulation at
      about 0.47 times real time it lasts about 2.4 sim-s.
    - It reads the arm and the finger again after the hold, and fails
      with PRECONDITION_FAILED when the arm left `receive` or the hand
      closed meanwhile.
2. **The request only asks for the bag.** "Please hang your bag on my
   hand." is its own `speak`, and `wait_for_handover` follows. It no
   longer says "tell me when it is there", so the guest does not time the
   close (T:101, S:19), and nothing waits for an answer.
3. **The cue gives an interval, apart from the request.** After the hold,
   `speak("I will close my hand in 3 seconds. Please be careful.")` takes
   T:107's form. Spoken with the request, the two would make T:106's
   penalized example.
4. **A close in `receive` waits.** `set_gripper` waits
   `RECEIVE_CLOSE_DELAY_S` = 5 s through `ctx.sleep` before it sends a
   close for an arm whose pose reads `receive` by `ctx.arms.state()`. A
   stop or a reset during the wait sends no goal and returns CANCELLED.
   Opens, "half", and closes in any other pose are sent at once. The cue
   says its interval about 1.5 s into a line of about 3 s, and the first
   audio comes a fraction of a second after the publish, so the close
   starts after the cue has played, about 3 s after "in 3 seconds" is
   heard.
5. **One silent retry.** If `set_gripper` reports the gripper holds
   nothing, the robot says "I will open my hand again.", opens the hand,
   calls `wait_for_handover` again with no second request, says the same
   cue and closes. After a second miss it goes on without the bag. The
   template says so, and says never to ask whether the bag is in the
   hand.
6. **The simulation and its check follow.** The scenario's guest hangs
   the bag silently on the request, with no answer. In
   `hri_missed_bag.yaml` the guest misses the first request, and hangs the
   bag once the hand reads open in `receive` after the empty close, so the
   silent retry finds it. `hri-check` allows one bag request in both
   variants, and its `arm_motion` criterion wants an ok left
   `wait_for_handover` before each left close, since the later of the
   run's start and the last left open.
7. **The wrist-effort early end is deferred.** Ending the hold on a change
   of the wrist's effort would make the close one on detection, as T:107's
   cue has it. It needs efforts from `/joint_states`, which the agent
   does not read and the simulation does not give
   ([question 10](/architecture/hri.md#open-questions)).

**Consequences**

- The guest is asked once and never asked for help, so the hand-over
  gives the referee no request for assistance (S:19) and no second
  request to stop the task on (G/ContinueRules:72).
- The guest's answer no longer keeps the close off a hand still placing
  the bag. The hold, the cue and the close delay are meant to leave time,
  and a close after a fixed hold may still read as a dangerous movement
  toward the guest (T:105). A robot check with a teammate's hand near the
  gripper tries them before anyone else plays the guest.
- Both times are constants, tuned by the robot check: whether a 5 s hold
  suits a real guest, and whether the close starts after the cue has
  played, timed on the robot's TTS from publish to end. A fixed delay
  suits one fixed sentence. A cue of another length needs the constant
  changed.
- Each attempt adds the 5 s hold and the 5 s close delay, in wall time,
  and the retry adds them again.
- A close reads the arms first when the arm client's memory does not know
  where its arm is, so its answer's last sentence comes from that read.
  It reads nothing while the arms are unverified, since that read would
  recover them: the close is then refused, as every goal is, until
  `get_arm_state` or the drive check recovers them.
- `wait_for_handover` takes the open position from
  `config/arm_poses.toml`, the file the commander client loads, because
  `ctx.arms` gives the poses and not the gripper positions.
- A guest who never hangs the bag costs two empty closes, and the robot
  goes on without it.
- Whether the robot's own request still reads as T:106's guidance is
  [question 13](/architecture/hri.md#open-questions). The 10 cm of T:104
  is [question 11](/architecture/hri.md#open-questions).

**See also:** [HRI Task](/architecture/hri.md#the-bag-hand-over),
[Robot Capabilities](/architecture/capabilities.md#pose-arms-gaze-and-vision)
<!-- --8<-- [end:body] -->
