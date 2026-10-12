# ADR-0010: Navigate to named places with every wait bounded

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

RAI ships `NavigateToPoseBlockingTool`, but in rai-core 2.12.1 it waits for
the goal with no timeout. A stalled or crashed Nav2 would freeze the agent
for good. LLMs are also poor at choosing map coordinates.

**Decision**

We wrote `NavigateToLocationTool` (`navigate_to_location`):

- It takes the name of a place from `config/locations.toml` (x, y and yaw in
  the `map` frame). `list_locations` lists the names.
- It waits at most 2 s for the Nav2 server and for the goal to be accepted,
  and at most 120 s for the result. On timeout it cancels the goal and says
  so.

**Consequences**

- The LLM can only send the robot to places the team configured.
- A dead Nav2 costs the agent 2 s, not the rest of the run.
- The locations file must be updated for each arena map.
- On timeout the cancel is sent but its result is not awaited, and a reset
  does not cancel navigation.
  [ADR-0023](/architecture/decisions/0023-motion-supervisor.md) proposes
  fixing both.

**See also:** `src/walkie_agent/tools/navigation.py`
<!-- --8<-- [end:body] -->
