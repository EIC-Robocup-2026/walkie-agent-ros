# ADR-0025: Refuse to drive unless the arms are in a travel-safe pose

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

Nothing on the robot stops Nav2 from driving while an arm is stretched out.
An LLM can forget to tuck the arms before a `navigate_to_location` call, and
an outstretched arm can hit people, furniture or door frames.

**Decision**

Before sending a navigation goal, `navigate_to_location` and
`move_relative` read both arms' joint states (3 s limit). The arm client
makes the whole decision while it holds its call lock, so no arm goal can
start between the steps of the check:

- If each arm matches a pose with `travel_safe = true`, it drives. Arms are
  matched one at a time, since a pose can move one arm
  ([ADR-0022](/architecture/decisions/0022-named-arm-poses.md)).
- If either arm matches no such pose, it refuses, and tells the LLM to move
  the arms to one first, for example `home`.
- If a commander call is running, or a goal whose call gave up has not
  ended, it refuses at once without reading, and says why. This check comes
  first, so a running call holds off a drive until it ends, even when the
  commander is no longer discovered. A goal whose call gave up holds off a
  drive until its result comes, until its action server has stayed
  undiscovered for 5 s, or for at most 60 s after the call gave up.
- When such a goal stops holding off a drive before its result came, the
  client abandons it. Its end is unknown, and it may still move an arm,
  since MoveIt lets a goal run for twice its planned duration plus 60 s,
  so the goal is then unresolved and the arms are unverified. A goal that
  the commander accepts after it was abandoned is cancelled at once, is
  unresolved again and makes the arms unverified again. While the arms are
  unverified, the check recovers, all within its 3 s limit, as
  `get_arm_state` does. It refuses when recovery does not clear the state,
  and when it cannot read the arms for any reason: it never fails open
  while the arms are unverified. Recovery has two steps, and clears the
  state only when both pass:
    1. The commander shows that every unresolved goal is no longer active.
       A goal's result or rejection resolves it. Otherwise the check sends
       it a cancel by its id, which the client made when it sent the goal,
       so even a goal whose send the commander never answered can be
       cancelled. The answer `unknown` resolves the goal: the commander
       does not know it, as after a restart. So does `ended`, unless a
       cancel of the goal was accepted or is unanswered, since the
       commander also answers `ended` to a second cancel while it still
       stops the goal. After `accepted`, the commander has stopped both
       arms, or the gripper finishes its current motion, and the goal's
       result resolves it. After `refused` or no answer, the goal may
       still move an arm or a gripper, and the check refuses.
    2. A settled read: two reads of both arms, 0.3 s apart, agree within
       0.01 rad, and two reads of the finger of every gripper whose
       command was abandoned agree within 1 mm. Otherwise an arm or a
       gripper still moves.
- No arm or gripper goal starts until recovery clears the state. The
  commander stops both arms for a joint cancel only while the cancelled
  goal is active, so a cancel of an older goal that reaches the commander
  after that goal ended cannot stop a newer one.
- If the commander answers with a reading that cannot show the arms are
  safe, it refuses, and tells the LLM to say that the arms cannot be
  checked. Such a reading is malformed, lacks a joint of either arm, or
  holds a position that is not a finite number.
- If the commander's `/get_joint_states` service is not discovered, does
  not answer, or answers without a reading, it drives anyway, and adds a
  note that says which, such as "Arm state unknown: the commander was not
  found." When the service is not discovered, it drives at once, without
  the 3 s wait. A missing action does not stop the arms being read, so the
  check still reads them then.

The check runs only when the agent has an arm backend (`--arms`).

**Consequences**

- The LLM gets a clear reason and a fix instead of a collision.
- Failing open keeps navigation working when the arm stack is down, and on
  setups with no arms running. The cost is that a crashed commander disables
  the check. It fails open only when there is no reading at all and no goal
  may still move an arm: a goal that has not ended, a goal abandoned before
  its end was known, or a reading that cannot be trusted refuses the drive.
- A commander that dies while a goal is outstanding keeps the robot from
  driving until the commander is back, answers the goal's cancel, and two
  reads agree. So does a commander that refuses the cancel of an abandoned
  goal or does not answer it; walkie-ros's commander never refuses one.
  The refusal tells the LLM to try again, then to tell the person that the
  arms cannot be checked.
- The client's state lives in the agent process. An agent restart forgets
  it, abandoned goals included, so the new agent drives after one read. A
  restart is not evidence that a motion ended.
- The settled read only supplements the commander's answers. It cannot
  see a motion slower than 0.01 rad in 0.3 s. That matters only for a
  trajectory that move_group still executes after the commander
  restarted, since the new commander does not know its goal and answers
  `unknown`. The full answer is the MotionSupervisor's stop
  ([ADR-0023](/architecture/decisions/0023-motion-supervisor.md)), which
  ends every MoveIt execution.
- While the arms are unverified, a drive costs the cancels of abandoned
  goals and their answers, a second round of reads and 0.3 s, all within
  the 3 s limit.
- It is advisory. Teleoperation or another program can still drive with the
  arms out. A robot-side interlock in walkie-ros would be stronger.
- The team must decide which poses are safe for driving.
<!-- --8<-- [end:body] -->
