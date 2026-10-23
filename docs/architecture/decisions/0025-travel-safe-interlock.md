# ADR-0025: Refuse to drive unless the arms are in a travel-safe pose

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-12

**Context**

Nothing on the robot stops Nav2 from driving while an arm is stretched out.
An LLM can forget to tuck the arms before a `navigate_to_location` call, and
an outstretched arm can hit people, furniture or door frames.

**Decision**

Before sending a navigation goal, `navigate_to_location` and
`move_relative` read both arms' joint states (3 s limit):

- If each arm matches a pose with `travel_safe = true`, it drives. Arms are
  matched one at a time, since a pose can move one arm
  ([ADR-0022](/architecture/decisions/0022-named-arm-poses.md)).
- If either arm matches no such pose, it refuses, and tells the LLM to move
  the arms to one first, for example `home`.
- If an arm goal is still running, it refuses, and says an arm is still
  moving.
- If the commander does not answer, it drives anyway, and says that the arm
  state is unknown. When the commander's service or actions are not
  discovered, it drives at once, without the 3 s wait.

The check runs only when the agent has an arm backend (`--arms`).

**Consequences**

- The LLM gets a clear reason and a fix instead of a collision.
- Failing open keeps navigation working when the arm stack is down, and on
  setups with no arms running. The cost is that a crashed commander disables
  the check.
- It is advisory. Teleoperation or another program can still drive with the
  arms out. A robot-side interlock in walkie-ros would be stronger.
- The team must decide which poses are safe for driving.
<!-- --8<-- [end:body] -->
