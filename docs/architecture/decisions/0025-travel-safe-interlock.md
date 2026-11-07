# ADR-0025: Refuse to drive unless the arms are in a travel-safe pose

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

Nothing on the robot stops Nav2 from driving while an arm is stretched out.
An LLM can forget to tuck the arms before a `navigate_to_location` call, and
an outstretched arm can hit people, furniture or door frames.

**Decision**

Before sending a navigation goal, `navigate_to_location` reads both arms'
joint states (3 s limit):

- If each arm matches a pose with `travel_safe = true`, it drives. Arms are
  matched one at a time, since a pose can move one arm
  ([ADR-0022](/architecture/decisions/0022-named-arm-poses.md)).
- If either arm matches no such pose, it refuses, and tells the LLM to move
  the arms to one first, for example `home`.
- If the commander does not answer, it drives anyway, and says that the arm
  state is unknown.

**As built in the demo** (decision S6-8 of the
[demo decisions](/architecture/mahoraga-demo-decisions.md)):

- The check runs before every base motion, not only a drive to a place:
  `RosBase` runs it for `go_to` and `move_relative`, so `face_person` and a
  turn are checked too, since an arm out sweeps a circle in a turn. It runs
  before the goal's target is computed, so a refused motion reads no TF and
  sends nothing. Its limit is the commander client's read bound, or what the
  motion has left, whichever is shorter.
- `FakeRobot` runs the same rule (`arm.answers.drive_verdict`) when it is
  given arm poses, so a fake run refuses in the robot's words.
- It fails open only when the commander's joint state service is not found,
  does not answer, or cannot read the joints. It never fails open while the
  arms are unverified, that is while a goal whose end is unknown may still
  move them; it then refuses. A busy commander refuses too. A reading that
  does not show both arms refuses.
- When it fails open, the warning goes to the agent's log. The LLM is not
  told, since `Base` verbs return only a pose.

**Consequences**

- The LLM gets a clear reason and a fix instead of a collision.
- Failing open keeps navigation working when the arm stack is down, and on
  setups with no arms running. The cost is that a crashed commander disables
  the check.
- It is advisory. Teleoperation or another program can still drive with the
  arms out. A robot-side interlock in walkie-ros would be stronger.
- The check and the drive are not one step: an arm goal from another program
  could start after the check passes. Within the agent, tools run one at a
  time, so its own arm goals cannot.
- The team must decide which poses are safe for driving.
<!-- --8<-- [end:body] -->
