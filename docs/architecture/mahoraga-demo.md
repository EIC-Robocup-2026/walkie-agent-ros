# Mahoraga Demo Prototype

A prototype, on branch `feat/mahoraga-demo`, of the layered agent that the
Mahoraga agent plan proposes. The competition agent and Mahoraga share its
ROS-free foundation, so the same atomics run on the robot, in unit tests and
in the fake-world gym. It sits beside main's agent, whose `agent.py`,
`graph.py`, `todo.py` and `tools/` stay unchanged until a later slice replaces
them. `tasks.py` has already moved to templates by name only, for both agents.
This page is the contract those slices follow.

!!! warning "Prototype"
    Nothing here has run against a real model, in MuJoCo or on the robot.
    Every choice below is a provisional pick of the plan's recommended option.

## Layer map

Each layer calls only the layer below it. Mahoraga reaches layer 3 only.

| Layer | Module | Slice |
| --- | --- | --- |
| 2. Capability verbs, guards, `Ctx` | `walkie_agent/capabilities/` | This one |
| 2. FakeRobot, Mahoraga's FakeCtx | `walkie_agent/capabilities/fake.py` | This one |
| 3. Error type | `walkie_agent/errors.py` | This one |
| 3. Atomics and tool sets | `walkie_agent/atomics/` | This one |
| 4. Adapter to LangChain tools | `walkie_agent/adapter.py` | This one |
| 5. `create_agent`, middlewares, Runner | `walkie_agent/runtime/` | [Built, unused](#runtime) |
| 1 to 2. rclpy glue, ROS-backed verbs | `walkie_agent/ros/` | Later |
| 6. `run_script`, shelf, gym | `mahoraga/`, with its own pixi environment | [Built](#mahoraga), fake world only |

`errors`, `capabilities`, `atomics`, `adapter` and `runtime` never import
rclpy, rai, rai_interfaces or any ROS message. `tests/test_ros_free.py` checks
this in a fresh interpreter.

## Errors

```python
class ErrorCode(StrEnum):  # walkie_agent.errors
    UNAVAILABLE, TIMEOUT, NOT_FOUND, REFUSED, PRECONDITION_FAILED, SKILL_FAILED, CANCELLED, INTERNAL

class StepError(Exception):
    def __init__(self, code: ErrorCode | str, message: str) -> None: ...  # ValueError for an unknown code
    code: ErrorCode
    message: str  # one plain sentence for the LLM; str(error) is "CODE: message"
```

Verbs and atomics raise only `StepError`. Anything else that escapes a tool is
a bug.

## Capability verbs

```python
# walkie_agent.capabilities: frozen records and Protocols
Pose2D(x: float, y: float, yaw: float)  # map frame, yaw in radians
ObjectSeen(id: str, name: str, category: str, place: str, size_cm: float | None = None,
           weight_g: float | None = None, thickness_cm: float | None = None)
PersonSeen(id: str, name: str | None, pose: str, gesture: str, room: str)

class Base(Protocol):
    def places(self) -> Mapping[str, Location]: ...  # walkie_agent.locations.Location
    def go_to(self, place: str, *, timeout_s: float = 120.0) -> Pose2D: ...
    def move_relative(self, forward_m: float = 0.0, left_m: float = 0.0, turn_deg: float = 0.0,
                      *, timeout_s: float = 60.0) -> Pose2D: ...
    def stop(self) -> None: ...  # never raises
    def pose(self) -> Pose2D: ...
class Speech(Protocol):
    def say(self, text: str, *, timeout_s: float = 10.0) -> None: ...
class Vision(Protocol):
    def objects(self, query: str = "", *, timeout_s: float = 5.0) -> list[ObjectSeen]: ...
    def people(self, query: str = "", *, timeout_s: float = 5.0) -> list[PersonSeen]: ...
class Arms(Protocol):  # each returns the id of the object picked, placed or handed over
    def pick(self, object_id: str, *, timeout_s: float = 60.0) -> str: ...
    def place_on(self, place: str, *, timeout_s: float = 60.0) -> str: ...
    def hand_over(self, person_id: str, *, timeout_s: float = 60.0) -> str: ...

Ctx(cancel: CancelToken, base: Base, speech: Speech, vision: Vision,  # frozen dataclass
    arms: Arms = UnavailableArms())  # UnavailableArms raises UNAVAILABLE on every verb
```

A query of `""` matches everything. Otherwise it matches an object's name,
category or plural category, or a person's name, pose or gesture, ignoring
case. The arm verbs are fake-only until the arm commander client exists.

### Guards live in the verbs

`walkie_agent.capabilities.guards` holds the shared checks. Every
implementation of a verb, fake or ROS-backed, calls them.

```python
MAX_MOVE_M = 3.0  # main's navigation.py cap
def is_finite_number(value: object) -> bool    # a bool is not a number here
def require_finite(**values: float) -> None    # REFUSED for NaN, infinity, a bool or a non-number
def require_text(**values: str) -> None        # REFUSED for anything but a string
def require_timeout(timeout_s: float) -> None  # REFUSED unless finite and positive
def check_relative_move(forward_m: float, left_m: float, turn_deg: float) -> None  # REFUSED
def relative_goal(start: Pose2D, forward_m: float, left_m: float, turn_deg: float) -> Pose2D

class CancelToken:
    cancelled: bool  # property
    def cancel(self) -> None: ...
    def reset(self) -> None: ...
    def check(self, what: str) -> None: ...  # CANCELLED once cancelled
    def wait_for(self, done: Callable[[], bool], timeout_s: float, what: str,
                 *, poll_s: float = 0.02) -> None: ...  # CANCELLED or TIMEOUT
    def unless_cancelled(self, what: str) -> ContextManager[None]: ...  # CANCELLED, or runs the block
```

- There is no velocity verb, only named places and bounded relative moves.
- Every wait is bounded and goes through `CancelToken.wait_for`, so a stop
  ends it within `poll_s`. A cancel that lands while `done()` runs wins over
  the completion.
- A verb that changes state when a wait ends does so inside
  `with cancel.unless_cancelled(what):`. A cancel either comes first and the
  block never runs, or waits for the block, so nothing changes after `cancel()`
  has returned.
- Base, vision and arm verbs check the token before they start, so a set token
  latches them until the runtime resets it. `Speech.say` does not, so a bare
  "stop" can answer with its fixed line (decision 6).
- A ROS-backed verb cancels its own goal when a wait ends in `CANCELLED` or
  `TIMEOUT`, then raises.

## FakeRobot

```python
# walkie_agent.capabilities.fake
class FakeRobot:  # implements Base, Speech, Vision and Arms
    def __init__(self, world: Mapping[str, Any] | None = None, *, seed: int = 0,
                 drive_s: float = 0.0) -> None: ...
    ctx: Ctx
    cancel: CancelToken
    calls: list[dict[str, Any]]   # {"verb": "base.go_to", "args": {...}, "error": None or a code}
    said: list[str]
    faults: dict[str, StepError]  # verb name -> the error every call to it raises; not base.stop
    def snapshot(self) -> dict[str, Any]: ...  # JSON-able copy for checkers: at, room, pose,
                                               # holding, places, objects, people, said, calls

def chula_pat14_6(seed: int = 0) -> dict[str, Any]  # a seeded world description
```

A world description is JSON-able, so the gym can build, store and compare it:

```python
{"places": {"shelf": {"x": 2.487, "y": -6.409, "yaw": -1.57, "room": "kitchen room"}},
 "objects": {"apple_1": {"name": "apple", "category": "fruit", "size_cm": 8, "weight_g": 180,
                         "thickness_cm": 8, "at": "shelf"}},
 "people": {"operator": {"name": "Jane", "room": "living room", "pose": "standing person",
                         "gesture": "waving person"}},
 "start": "instruction point"}
```

A place whose `room` is its own name is a room. An object's `at` is a place,
`"robot"` while held, or the id of the person who received it. The robot sees
what is at its place, or at every place in its room while it stands at the room
itself, and the people in its room. `world=None` builds `chula_pat14_6(seed)`,
and `drive_s` makes each drive take real seconds.

The constructor raises `ValueError` for a broken world, naming the record and
field: an unknown start, a place or person in an unknown room, an object at
something that does not exist, or a coordinate that is not a finite number.
Without this, a misspelled name hides the object or person, and the episode
reads as the agent's failure. Whether a world suits a given command is the gym's
check, not this one.

Places and rooms come from walkie-ros `origin/feat/chula-pat14-6-simulation`
(`simulation/mujoco/worlds/chula_pat14_6/authoring/source/30_16102026_ChulaPat14_6_world.toml`).
The `trash` and `instruction point` places, six objects, all measures and the
people are invented (decision 26). A builder that seeds a feasible world for
one generated command belongs to the gym.

## Atomics

Plain typed functions over `ctx` in `walkie_agent.atomics`. They return
JSON-able values and raise `StepError`. The LLM reads their Google-style
docstrings. Describe the return value in the summary paragraphs, because
LangChain drops a `Returns:` section from the tool description.

```python
def list_places(ctx: Ctx) -> list[dict[str, str]]
def go_to(ctx: Ctx, place: str) -> str
def move_relative(ctx: Ctx, forward_m: float, left_m: float = 0.0, turn_deg: float = 0.0) -> str
def say(ctx: Ctx, text: str) -> str
def find_objects(ctx: Ctx, what: str = "") -> list[dict[str, Any]]
def object_info(ctx: Ctx, object_id: str) -> dict[str, Any]
def find_people(ctx: Ctx, description: str = "") -> list[dict[str, Any]]
def pick(ctx: Ctx, object_id: str) -> str
def place_on(ctx: Ctx, place: str) -> str
def hand_over(ctx: Ctx, person_id: str) -> str

GPSR: list[Callable[..., Any]]     # the nine the spike used: every atomic but list_places
HRI: list[Callable[..., Any]]      # list_places, go_to, move_relative, say, find_people
DEFAULT: list[Callable[..., Any]]  # every atomic that needs no arms
```

## Adapter

```python
# walkie_agent.adapter
def bind(fn: Callable[Concatenate[Ctx, P], R], ctx: Ctx) -> Callable[P, R]
def as_tool(fn: Callable[..., Any], ctx: Ctx) -> StructuredTool
def as_tools(fns: Iterable[Callable[..., Any]], ctx: Ctx) -> list[StructuredTool]
def format_tool_error(exc: Exception, request: ToolCallRequest) -> str
def tool_error_middleware() -> ToolErrorMiddleware
```

`bind` returns a plain callable for the Monty driver, with the atomic's name,
docstring, signature without `ctx` and raw return value. `as_tool` builds
`StructuredTool.from_function(parse_docstring=True,
error_on_invalid_docstring=True)` over the same wrapper, raises `ValueError` for
a parameter with no description, and sends a non-string result as JSON text, so
an empty list reads `[]` rather than an empty message. Its schema validates
strictly, because pydantic would otherwise turn a JSON `true` into `1.0` before
any guard saw it. A model that sends a bool or a numeric string for a number
gets a validation error back and can retry, and an int still counts as a number.

`ToolErrorMiddleware(on_error=None, *, aon_error=None, tools=None)` ships in
langchain 1.4.4 under `langchain.agents.middleware` and serves the sync and
async paths. `ToolNode` answers argument validation errors before it.
`format_tool_error` gives `"CODE: message"` for a `StepError`. Anything else
logs its traceback and gives `"INTERNAL: <tool> failed with <ExceptionType>."`.

## Mahoraga

The research gym, in the `mahoraga` pixi environment and solve group. Its
dependencies stay out of `default`, which `tests/test_mahoraga_absent.py` checks,
and it never imports ROS, which `mahoraga/tests/test_ros_free.py` checks. It reaches
the foundation through the atomics and `adapter`, and it covers rungs R0 to R2 in
the fake world. MuJoCo validation and promotion are not built.

| Module | What it holds |
| --- | --- |
| `driver` | `ScriptRunner`, `Budgets`, `ScriptResult`, `allow_list`, `stubs_for`, `describe_functions`, `script_summary` |
| `shelf` | `Shelf`: saved functions and `ledger.jsonl`, outside git |
| `sandbox` | `open_pool`, the Monty pool with each worker inside bwrap. It fails closed. |
| `commands` | `RecordingGenerator`, `parse`, `commands(split, n, category)`, and the arena's names in `data/` |
| `worlds` | `build_world(spec, seed)`, `world_seed(sample_id, epoch)` |
| `checkers` | `check(spec, start, end)` over `FakeRobot.snapshot()` |
| `oracle` | `oracle_script(spec)`, written from the slots with no LLM |
| `bundles` | `BUNDLES`, stand-ins for arm D until the team's bundles exist |
| `arms` | `ARMS`, `INDUCT`, `ArmSetup(arm, pool=, shelf_root=).episode(ctx)` |
| `tasks` | `dataset`, `episode_robot`, the `ground_truth` scorer |
| `gate` | `run_gate`, `run_standard_gate`, `require_gate`, `GateError` |
| `gym` | `gpsr_task`, `run_episodes`, `resolve_model` |
| `report` | `report(log_dir)`, `arm_metrics`, `compare`, `mcnemar_p` |

### `run_script`

```python
ScriptRunner(pool, *, shelf=None, budgets=Budgets()).run(code, fns, *, save=()) -> ScriptResult
```

A script calls only the functions in `fns`, which `allow_list(atomics, ctx)` builds
with ctx bound. Every other call or name lookup is refused, including any OS call,
so a script reaches no ctx, file, network or unknown function. Before a script
runs, it gets a type check against stubs made from the same signatures.

| What fails | What the script sees |
| --- | --- |
| A `StepError` | `RuntimeError("CODE: message")`, which it may catch |
| Bad arguments | `TypeError("tool: argument: reason")` |
| A call off the list | `PermissionError` |
| A bug in the host function | `RuntimeError("INTERNAL: tool failed with Type.")` |
| A budget, by default 40 calls, 2 s a feed, 30 s of wall clock, 64 MiB or 4000 bytes printed | An error it cannot catch, which ends the run |

`ScriptResult.calls` is the ledger: one entry per call, with its status. The model
sees only `script_summary(result)`.

### Shelf

`Shelf(root=None, *, frozen=False)` keeps `<name>.py` files and `ledger.jsonl` in
`root`, by default `<home>/shelf`. The home is the `--home` argument, then
`MAHORAGA_HOME`, then `~/.local/share/mahoraga`, and it may not be inside a git work
tree. A run with `save=[names]` keeps top-level functions only if it succeeds. A
frozen shelf fixes its callable set when it is built, to the functions verified by
then. Only the scorer calls `mark_verified`, after a success.

### Arms

| Arm | The model gets |
| --- | --- |
| `A` | The per-step GPSR tools |
| `B` | `run_script`, with no shelf |
| `C` | `run_script`, with the frozen, verified shelf callable |
| `C2` | `run_script`, with the same scripts as prompt examples, not callable |
| `D` | The per-step tools plus `BUNDLES` |
| `induct` | `run_script` with `save`, on a writable shelf, for training only |

All arms share the prompt, the atomics and the eval's model, temperature and seed.
One (command, epoch) is one seeded world, the same for every arm.

### The oracle gate

An LLM arm refuses to start until the oracle has scored 100% twice with identical
logs, for the same code and the same data.

`run_gate` plays the oracle's scripts through the same driver, world builder and
scorer an arm uses, twice, into `.eval` logs. It reads both back and compares them
as JSON, with samples in a fixed order and without the fields in `gate.VOLATILE`:
clocks, random ids and the log's location. The result lands in `<home>/gate/`.

`require_gate(records, epochs)` raises `GateError` unless passing results for the
current `code_digest()` cover a digest of every (command, epoch, world) requested.
A result counts only if `passed` is true, `failures` is empty, its two pass digests
are equal, and both logs exist and still hash as the gate recorded. This catches a
stale, partial or hand-edited result made by mistake. It is no security boundary:
nothing is signed. `gpsr_task` and `run_episodes` both call it.

The code digest covers each file's repository path and content, for the files
`gate._digested_files()` lists: every `mahoraga/*.py` except `report.py` and
`__main__.py`, the data files, `errors.py`, `adapter.py`, `locations.py`, the
capabilities and atomics packages, and `pixi.lock`. So an edit to any of them,
related to the oracle or not, means running the gate again, which takes about a minute.

### Running it

Run these with `pixi run --frozen -e mahoraga`, from the repository root. The gate
is offline and costs nothing. `run` calls a real model, so `--commands` has no default.

```bash
# The tests
python -m pytest mahoraga/tests

# The gate on the plan's trial, training and held-out data: 1080 episodes, twice
python -m mahoraga gate
# Or on other data
python -m mahoraga gate --split trial --commands 20 --epochs 1 --category people

# A small real slice, with the key in the environment
OPENAI_API_KEY=... python -m mahoraga run --arms A,B --commands 5

# Metrics M1 to M10 and the paired comparison, as JSON
python -m mahoraga report --log-dir <the run's directory>
```

The defaults are `deepseek/deepseek-v4.1-flash:nitro`, through OpenRouter at
`https://openrouter.ai/api/v1`, with the key in `OPENAI_API_KEY`. Other flags are
`--epochs`, `--split`, `--category`, `--model`, `--base-url`, `--temperature`,
`--seed`, `--max-samples`, `--log-dir`, `--no-bwrap` and `--home`.

The plan's R2 is then `run --arms induct --split train --commands 300`, followed by
`run --arms A,B,C,C2,D --split heldout --commands 160 --epochs 3`.

## Runtime

`walkie_agent.runtime` is `create_agent` plus a small stack of middlewares, the
todo list and a Runner. Tools come from `as_tools` over a `Ctx`, errors go through
`tool_error_middleware`, and `ctx.cancel` is the only cancel signal. Nothing starts
it yet: the cutover slice adds `walkie_agent/ros/` and a new `main.py`.

```python
# walkie_agent.runtime
def load_config(path="config.toml") -> dict: ...
def agent_settings(config) -> dict: ...  # run_limit and deadline_s, from the optional [agent]
def build_model(config, role="complex_model") -> BaseChatModel: ...
def build_fallback(config) -> BaseChatModel | None: ...  # needs fallback_vendor and fallback_model
def build_callbacks(config) -> list[BaseCallbackHandler]: ...  # Langfuse, when enabled

def build_agent(model, ctx, *, tools=DEFAULT, run_limit=25, deadline_s=60.0, on_todos=None,
                fallback=None, tool_log=None, checkpointer=None,
                system_prompt=SYSTEM_PROMPT) -> CompiledStateGraph: ...
def build_middleware(ctx, *, run_limit, deadline_s, on_todos=None, fallback=None,
                     tool_log=None) -> list[AgentMiddleware]: ...

class ToolLog(AgentMiddleware):  # one JSON line per tool call
    def __init__(self, path): ...
    def write(self, **record) -> None: ...

class Session:  # one checkpointer thread and one Langfuse session
    thread_id: str
    def config(self, callbacks=()) -> dict: ...  # ainvoke's config, with max_concurrency=1

class Runner:  # every method is safe from any thread, and only queues work
    session: Session
    def __init__(self, agent, ctx, *, on_todos=None, on_reply=None, callbacks=(),
                 tool_log=None): ...
    def start(self) -> None: ...
    def hear(self, text) -> None: ...  # a bare stop stops, and anything else is a task
    def submit(self, text, todos=None, source="seed") -> None: ...  # raises on a bad seed
    def stop(self, text="stop") -> None: ...
    def reset(self) -> None: ...
    def wait_idle(self, timeout=None) -> bool: ...
    def shutdown(self, timeout=None) -> bool: ...  # later use raises RuntimeError

def is_stop_utterance(text) -> bool: ...
def parse_seed(items, source="seed") -> list[dict]: ...  # raises TodoRejected
def render(items) -> str: ...  # the <todos> reminder
def public(items) -> list[dict]: ...  # items without their private ledger keys
def console_line(item) -> str: ...
```

Wiring, as the cutover slice will do it:

```python
config = load_config()
log = ToolLog("logs/tools.jsonl")
agent = build_agent(build_model(config), ctx, on_todos=publish, tool_log=log,
                    fallback=build_fallback(config), **agent_settings(config))
runner = Runner(agent, ctx, on_todos=publish, on_reply=print, tool_log=log,
                callbacks=build_callbacks(config))
runner.start()
runner.hear("Go to the sofa.")
```

`on_todos(items, changed)` gets public copies, and `changed` holds the items whose
status changed. Both listeners run on the Runner's thread and must not block. The
model gets two tools beyond `tools`: `update_todos(ops)` and
`wait_for_person(question)`. The warning at more than 20 bound tools counts them.

### The middleware order

`build_middleware` has the one order that works, and `tests/test_runtime.py` pins it.
`after_model` hooks run in reverse list order, and a jump skips the ones still to run.

| Middleware | Guards |
| --- | --- |
| `ToolLog`, when configured | Logs every answer, refusals included |
| `StopLatch` | Once the token is set: no model call, and every tool call refused |
| `SerialGuard` | A failed call or a question skips the later calls of its message; only the first `update_todos` of a message runs, and a refused copy counts as a failed call |
| `WalkieTodos` | The list, the reminder, the completion gate and the two tools |
| `EndAfterQuestion` | Ends the run after a successful question |
| `RepairHistory` | Answers calls left open by an earlier run and calls with unreadable arguments |
| `ModelCallLimitMiddleware` | Counts every model call, since nothing later in the list can jump past it |
| `ModelFallbackMiddleware`, when configured | No fallback is configured by default (decision 19) |
| `Deadline` | `asyncio.timeout` around each model attempt |
| `tool_error_middleware()` | Innermost, so the ledger sees errors as results |

### Behaviour the runtime promises

- **Tool calls.** The calls of one message run one at a time, in order. That rests on
  `max_concurrency=1`, which is observed behaviour of the installed versions and not
  documented, so rerun `tests/test_runtime.py` on any dependency update.
- **Stop.** A bare stop sets the token at once, which ends a drive through the verb and
  refuses every remaining call of the run. `StopLatch` refuses a call before it is
  dispatched. `as_tool`, `bind` and the two todo tools check the token again when the
  body starts, for a stop that lands in between. A `say` already speaking is not cut
  off. It drops queued tasks but not a queued reset. The Runner speaks `STOP_LINE` once
  through `ctx.speech.say` and logs it under the session it was heard in. A later graph
  pass with no model call cancels the open steps and clears a waiting question. The
  token stays set until the next task starts. A compound such as "stop and come here"
  is an ordinary task, so it queues behind the active run.
- **Reset.** It cancels and drops the queue. After the active run has finished, it
  deletes the thread, starts a new session and publishes an empty list. A hung LLM call
  can delay this by up to the deadline.
- **Shutdown.** It cancels, drops the queue, and waits for the active run and any stop
  line. It returns False if the thread outlives `timeout`, and calling it again waits
  again. From the moment it starts, `submit`, `stop` and `reset` raise.
- **Failure.** An exception or a timeout ends that task only. The Runner logs it and
  sends `FAILURE_LINE` to `on_reply`, and the next task runs on the same checkpoint.
- **Deadline.** It is cooperative: `asyncio.timeout` around the awaited call, which
  blocking code on the loop delays. It ends the client's wait, and does not prove the
  provider stopped computing.
- **Todo evidence.** Tools add entries to an append-only ledger and never write the
  list. One successful result completes one step. Refused and skipped calls are not in
  the ledger.

`config.toml` may add an `[agent]` table with `run_limit`, `deadline_s`,
`fallback_vendor` and `fallback_model`. The defaults of 25 calls and 60 s are
provisional until real runs measure them.

## Provisional decisions

| # | Decision | Pick |
| --- | --- | --- |
| 1 | Atomic tool shape | Plain typed functions over ctx verbs, guards inside the verbs |
| 2 | The as_tool adapter | `StructuredTool.from_function(parse_docstring=True)`, ToolErrorMiddleware as the one error path |
| 3 | How tools are listed | Per-test tool sets as plain lists in code |
| 4 | Memory id kind | Plain database ids |
| 5 to 8 | Decided earlier | Own WalkieTodos; bare stop is a fixed line with no LLM; runtime fixes in the `create_agent` port; task templates by name only |
| 22 | Models | deepseek-v4.1-flash for runs |
| 23 | Where Mahoraga lives | Top-level `mahoraga/` with its own pixi environment |
| 26 | Arena data | A trash location and more objects in the fake arena |

Out of scope: CI and ruff jobs, the voice environment split, leaving RAI
(decision 9), MuJoCo validation (Phase R3) and promotion (Phase R4).

## Where this differs from the plan

- `ctx.arms` has the GPSR verbs, not the plan's `move_to` and `set_gripper`,
  which come with the commander client. Pick and place then become bundles.
- `ctx.base.pose()` replaces `ctx.pose`, and `ctx.base.places()` lets
  `list_places` replace main's `list_locations`.
- `FakeRobot` is one class, so its `ctx` holds the same object four times.
- `format_tool_error` answers a bug with `INTERNAL` rather than re-raising it,
  so one bad tool does not end a run.
- `BUNDLES` are stand-ins, because the team has no reviewed bundles yet. Until it
  does, arm D says nothing about curated skills.
- Only callability separates C2 from C. Both use `run_script`, on the same scripts.
- `induct` is one more arm, because induction has to run through something.
- Shelf functions run as a separate first feed in the script's session, not as text
  before it. They pass the same driver, and the script keeps its own line numbers.
  A save also type-checks each definition standing alone.
- Four people templates are out: `guideClothPrsFromBeacToBeac`,
  `greetClothDscInRm`, `countClothPrsInRoom` and `meetNameAtLocThenFindInRm`.
- `FakeRobot` has no follow verb and its people do not move, so follow and guide
  are checked from the robot's trail. What a line says about "the time" is not
  checked, and a relative move does not change the trail's place.
- The world builder renames the operator and takes them out of every pose and
  gesture, so no command's selector can match them.
- M9 counts refused calls, since an escape would not show in a log. The bwrap
  tests are what stand behind it.
- There is no secondary template-chain set, no frozen paraphrases, no
  pre-registration file and no flag registering `run_script` in the runtime.
- The plan's target for what replaces `agent.py`, `graph.py`, `todo.py`, `tasks.py`
  and `tools/todos.py` is about 575 own lines, against 1,111 today. It measures 1,074:
  995 in `runtime/` and 79 in `tasks.py`. The decided fixes and the op-based todo list,
  455 lines against the plan's 175, cost more than the estimate.
- The stop's list update is a graph pass with a `stopping` input key rather than
  `aupdate_state`, so it keeps the same order as a task in the Runner's queue.
- The stop line is spoken at once rather than after the active run, which may be a
  hung LLM call. Each `stop()` speaks once, so the speech verb must tolerate it
  overlapping another `say`.
