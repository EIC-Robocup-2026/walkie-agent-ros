# ADR-0061: Run the agent on create_agent, with one Runner and one cancel token

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-05

Accepted for the mechanism in this cut, as far as the evidence below goes. No real
model, real Nav2, simulator or robot has run it.

**Context**

[ADR-0038](/architecture/decisions/0038-own-graph-per-turn-binding.md) kept a
handwritten graph on RAI's `LangChainAgent`, which changed RAI's message list in place.
A stop or a reset acted only between graph steps, so a drive ran to its end. ADR-0038
named `create_agent` as worth revisiting. The Mahoraga prototype needed the runtime to
be free of ROS anyway.

**Decision**

`walkie_agent.runtime` is LangChain's `create_agent`, our middlewares, a checkpointer and
a `Runner`. It imports no ROS.

**Middleware order.** `build_middleware` has one order, and a test pins it: `ToolLog`
(optional), `StopLatch`, `SerialGuard`, `WalkieTodos`, `EndAfterQuestion`,
`RepairHistory`, `ModelCallLimitMiddleware`, `ModelFallbackMiddleware` (optional),
`Deadline`, `tool_error_middleware()`.

**Serial calls.** `Session.config` sets `max_concurrency=1`, so the calls of one message
run in order. This is observed behaviour of the installed versions and is not documented
by LangGraph, so any dependency update has to rerun the tests.

**The Runner.** One daemon thread, named `walkie-runtime`, runs one asyncio loop and one
task at a time. Its methods are safe from any thread, and only queue work.

- **Stop.** A bare stop sets the token at once, even when idle. It drops queued tasks
  and keeps a queued reset. The Runner speaks `STOP_LINE` on a worker thread, and a
  graph pass with no model call cancels the open steps. The token resets when the next
  task is taken.
- **Reset.** It cancels and clears the queue. After the active run has ended, it deletes
  the thread, starts a new session and publishes an empty list.
- **Failure.** An exception or a timeout ends that task only. `FAILURE_LINE` goes to the
  console, and the next task runs on the same checkpoint.
- **Failed calls.** A failed call skips the later calls of the same AI message, which are
  answered `Not run: ...`. A duplicate `update_todos` counts as a failed call.
- **History.** `RepairHistory` answers calls left open by an earlier run, and calls with
  unreadable arguments.
- **Evidence.** Tools add to an append-only ledger. One success completes one step.
  Refused and skipped calls never reach the ledger.
- **Questions.** A successful `wait_for_person` ends the run, and the next task resumes
  the waiting step once.

**Input.** `/from_human` is text only. A message with an image or audio payload is logged
and refused, not silently stripped. That includes a bare "stop" that carries a payload.

**Shutdown.** Ingress closes first. If the run or a goal cannot settle, the executor
stays alive, the joins and the drain retry within bounds, and `main.py` reports what is
unresolved. A second SIGINT or SIGTERM is the explicit forced exit: it logs the
unresolved items and exits with code 3.

**Scope.** This supersedes ADR-0038. It partly supersedes
[ADR-0003](/architecture/decisions/0003-rai-react-agent.md): RAI's agent host leaves
`main.py`, while its connector, rai-core and rai-s2s stay. It amends
[ADR-0008](/architecture/decisions/0008-reset-topic.md), because a reset now also stops
the motion, and [ADR-0041](/architecture/decisions/0041-todo-list-and-seeding.md),
because the list is now plain dictionaries in checkpoint state. The voice entry points
in `walkie_agent.voice` keep RAI's `AgentRunner`.

**Consequences**

- A stop no longer waits for a graph step. A running drive ends through the verb.
- A late Nav2 acceptance after a stop is handled by the goal's record
  ([ADR-0052](/architecture/decisions/0052-capability-verbs-and-backends.md)), not by the
  runtime.
- Evidence that ran: unit tests with a scripted model and `FakeRobot`, integration tests
  with real rclpy and a fake robot, and one silent demo. The demo saw the stop line and
  a terminal cancel in about 0.2 s against the fake. That number says nothing about
  hardware.
- Evidence that has not run: a real model, real Nav2, the simulator and the robot.
- `say` that is already speaking is not cut off.
- A reset can wait for a hung model call, up to the deadline, and for a stop line, up to
  the say timeout.
- The `Deadline` middleware is cooperative. It ends the client's wait and does not prove
  that the provider stopped computing.
- The hold-then-force shutdown ran only against fakes. It has not met a real goal that
  Nav2 never answers.
- The connector's spin thread can still die during its final teardown inside RAI's
  `connector.shutdown()`. That can print a traceback at shutdown. The exit code does not
  change. It stays as long as we keep RAI's connector.

**See also:** [Mahoraga demo: runtime](/architecture/mahoraga-demo.md#runtime),
[The Agent](/architecture/agent.md#how-a-task-runs),
[Queue, stop and reset](/learn/queue-stop-reset.md)
<!-- --8<-- [end:body] -->
