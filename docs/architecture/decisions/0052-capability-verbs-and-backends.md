# ADR-0052: Reach the robot through capability verbs with separate backends

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-05

Accepted for the contract and the backends in this cut, as far as the evidence below
goes. Nothing here is proven on real Nav2, in the simulator or on the robot.

**Context**

The old tools each called rclpy themselves, so a test needed a ROS graph, and each
tool repeated its own checks. The atomics, the tests and the Mahoraga gym need one
interface to the robot that imports no ROS. A Nav2 goal also outlives the call that
sent it: acceptance can arrive after the tool gave up, and the robot keeps moving
until Nav2 reports an end.

**Decision**

`ctx` holds four capabilities, as `typing.Protocol` classes in
`walkie_agent.capabilities`: `ctx.base`, `ctx.speech`, `ctx.vision` and `ctx.arms`.

- **Records.** Verbs return frozen records: `Pose2D`, `ObjectSeen` and `PersonSeen`.
- **Errors.** A verb fails only with `StepError`, which carries one `ErrorCode`.
- **Guards.** The checks live in the verbs, through `capabilities.guards`, so every
  caller gets them. There is no velocity verb. Moves are named places, or relative moves
  of at most 3 m.
- **Waits.** Every wait is bounded and goes through `CancelToken`.

| Capability | In tests and in the gym | In `main.py` |
| --- | --- | --- |
| `ctx.base` | `FakeRobot` | `RosBase`, on `/navigate_to_pose` and TF `map` to `base_footprint` |
| `ctx.speech` | `FakeRobot` | `RosSpeech`, on `/to_human`, with `/voice_commands` rotating the speech id |
| `ctx.vision` | `FakeRobot` | `UnavailableVision`: every verb raises `UNAVAILABLE` |
| `ctx.arms` | `FakeRobot` | `UnavailableArms`: every verb raises `UNAVAILABLE` |

An unavailable verb never returns an empty list, because that would say nothing is in
view. A remembered database row is not a current view, so the old remembered-object
tool is retired and nothing stands in for live vision. A real backend for vision or arms
is a later slice with its own decision.

RAI's `ROS2Connector` stays. It owns the node, its executor and the TF buffer, and the
verbs attach to it.

`RosBase` owns the goals it sends, and only those. One record follows each goal from its
reservation until Nav2 reports an end.

- A verb that ends in `CANCELLED` or `TIMEOUT` marks its goal abandoned for good. A goal
  accepted after that gets cancelled, even if the token was reset in between.
- While a record lives, a new motion is refused, so no goal overwrites a live one.
- A cancel acknowledgement ends nothing. Only a terminal result or a rejection does.
- A goal whose state cannot be known keeps its record, and motion stays refused. That
  covers a failed acceptance future, a failed result future and `STATUS_UNKNOWN`.

`say` returns once the line is published. Nothing acknowledges playback, so nothing
claims it. `say` does not check the stop token, so the fixed stop line still speaks.

`ros/policy.py` refuses four names for a capability endpoint: `/cmd_vel`,
`/walkie/reset`, `/walkie/task` and `/walkie/todos`. Both sides are resolved through the
node. This guards against a configuration mistake, and is not a security boundary
([ADR-0009](/architecture/decisions/0009-forbidden-names-guard-rail.md)).

This amends two accepted decisions and replaces neither.

- [ADR-0007](/architecture/decisions/0007-speak-tool-and-console.md): the `speak` tool
  is now the `say` atomic over `ctx.speech`, and the fixed stop line also reaches
  `/to_human`. Plain replies still go to the console only.
- [ADR-0016](/architecture/decisions/0016-talk-to-walkie-ros-directly.md): the agent
  still calls walkie-ros directly, but only inside `walkie_agent/ros/`.

Proposed [ADR-0023](/architecture/decisions/0023-motion-supervisor.md) and
[ADR-0043](/architecture/decisions/0043-navigate-to-pose-provenance.md) stay proposed.
This decision builds neither of them.

**Consequences**

- A teammate can read and test against the capabilities without ROS.
- Evidence that ran: unit tests on fake ROS objects, integration tests with real rclpy
  and a real RAI connector against a fake Nav2 server, and one silent demo on rmw_zenoh
  against the same fake.
- Evidence that has not run: real Nav2 and its races, the walkie-ros simulator and the
  robot. TF freshness on hardware, stop latency and physical braking are unmeasured.
- The 2 s discovery, 2 s acceptance and 1 s TF budgets are implementation defaults, not
  measured requirements. The 0.5 s terminal cancel is a test criterion against the fake.
- RAI's transform helper sleeps 0.1 s on a miss, so a stop during a TF read takes about
  that long, plus scheduling.
- A sample stamped more than 0.5 s ahead of the clock reads as stale. Clock skew between
  machines would show up as `TIMEOUT`. This needs checking on the real robot.
- If Nav2 restarts and forgets a goal, motion stays refused until the agent restarts.
- A cancel request cannot prove that the wheels stopped. The physical emergency stop
  stays the real safety boundary.
- Vision and arms verbs fail, so tasks that need them fail honestly. This is not a full
  HRI or GPSR robot.

**See also:** [Mahoraga demo: capability verbs](/architecture/mahoraga-demo.md#capability-verbs),
[The Agent](/architecture/agent.md),
[Tool access](/setup/05-tool-access.md)
<!-- --8<-- [end:body] -->
