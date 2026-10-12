# ADR-0008: Reset the conversation with a message on /walkie/reset

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

RAI keeps every task, LLM reply and tool result for as long as the process
runs, and sends all of it to the LLM with every new task. Over a competition
run the history grows: each call costs more, and old tasks confuse new ones.
Restarting the agent takes time and drops ROS discovery.

**Decision**

Any `std_msgs/msg/Empty` message on `/walkie/reset` makes `WalkieAgent`:

1. drop the tasks still waiting in its queue,
2. interrupt the current task at its next step, and
3. clear the conversation on the agent thread, then start a new Langfuse
   session.

`/walkie/reset` is on the forbidden list, so the agent cannot reset itself.

**Consequences**

- The operator, or a task-level state machine, decides where conversations
  start and end.
- A tool call that is already running finishes first. Today a navigation goal
  still drives to the end after a reset.
  [ADR-0023](/architecture/decisions/0023-motion-supervisor.md) proposes
  stopping motion on reset.
- It relies on `ReActAgent` internals
  ([ADR-0003](/architecture/decisions/0003-rai-react-agent.md)).

**See also:** `src/walkie_agent/agent.py` (`WalkieAgent`)
<!-- --8<-- [end:body] -->
