# ADR-0031: Move short distances relative to the robot, bounded in length

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-15

**Context**

On a blank map there are no rooms to name, and people give directions such
as "move forward one meter" or "turn around".
[ADR-0010](/architecture/decisions/0010-bounded-navigation-tool.md) keeps raw
map coordinates away from the LLM, because LLMs are poor at choosing them.

**Decision**

We added `MoveRelativeTool` (`move_relative`):

- It takes meters forward, meters left and degrees to turn, in the robot's
  own frame.
- It reads the current pose from TF (`map` -> `base_footprint`), adds the
  offset, and sends one Nav2 `NavigateToPose` goal, so Nav2 still plans the
  path and avoids obstacles.
- It uses only a transform stamped at or after the call. The goal result and
  the TF update arrive on different topics, so the latest cached transform
  can still hold the heading from before the last goal's final turn.
- Moves longer than 3 m are refused.
- It shares the bounded waits and goal cancel of `navigate_to_location`
  through a common base class, `Nav2GoalTool`.

**Consequences**

- The LLM can drive anywhere in steps of up to 3 m. The limit catches a
  misread unit, such as centimeters taken as meters, but does not keep the
  robot inside an area.
- If TF stops updating, or the agent and the robot disagree on sim time, the
  tool refuses to move and says why.
- A turn in place sends a goal at the current position, which may fail in
  Nav2. This is not tested against the simulation yet.

**See also:** `src/walkie_agent/tools/navigation.py`
<!-- --8<-- [end:body] -->
