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

- If the arms match a pose with `travel_safe = true`, it drives.
- If they match no such pose, it refuses, and tells the LLM to move them to
  one first, for example `home`.
- If the commander does not answer, it drives anyway, and says that the arm
  state is unknown.

**Consequences**

- The LLM gets a clear reason and a fix instead of a collision.
- Failing open keeps navigation working when the arm stack is down, and on
  setups with no arms running. The cost is that a crashed commander disables
  the check.
- It is advisory. Teleoperation or another program can still drive with the
  arms out. A robot-side interlock in walkie-ros would be stronger.
- The team must decide which poses are safe for driving.
<!-- --8<-- [end:body] -->
