# Mahoraga Demo Prototype

A prototype, on branch `feat/mahoraga-demo`, of the layered agent that the
Mahoraga agent plan proposes. The competition agent and Mahoraga share its
ROS-free foundation, so the same atomics run through the ROS wrappers, in unit
tests and in the fake-world gym. The main entry point now uses this stack. The old
`agent.py`, `graph.py`, `todo.py` and `tools/` are deleted, along with RAI's agent host.
RAI's connector stays, and the two voice entry points keep RAI's `AgentRunner`.
This page is the contract each layer follows.
[The Agent](agent.md) describes the assembled node.

!!! warning "Prototype: what the evidence covers"
    Every choice below is a provisional pick of the plan's recommended option. The
    evidence comes in six kinds, and they do not stand in for each other:

    | Kind | What ran | Status |
    | --- | --- | --- |
    | Fake and unit | `FakeRobot`, fake ROS objects, a scripted model | Run |
    | ROS graph with a fake robot | Real rclpy and RAI's connector against a `FakeWalkie` node, on Fast DDS | Run |
    | Silent demo | Separate processes on rmw_zenoh and a private router, a scripted model, no audio | Run, on fake robots only |
    | Gym with a real model | Small slices in the fake world, listed in the [decision log](mahoraga-demo-decisions.md#small-real-model-runs) | Run, no ROS involved |
    | MuJoCo simulation with a real model | Real Nav2, MoveIt and the arm commander in the `chula_pat14_6` world, oracle perception, the in-process people memory, small tasks from S1 to S3 and S6 of [See, Remember and Act](see-remember-act.md), and four tasks with `run_script` and the operator's verdict | Run, listed in [What has been run](../setup/08-chula-simulation.md#what-has-been-run) |
    | Hardware | The robot's own graph, walkie-vision's node and walkie-database's people services | **Not run** |

    The last row is the one that would show the robot works. Nothing here claims it.

## Layer map

Each layer calls only the layer below it. Mahoraga reaches layer 3 only. The
ROS-backed verbs implement layer 2's Protocols, and only `main.py` connects them.

| Layer | Module | State |
| --- | --- | --- |
| 2. Capability verbs, guards, `Ctx` | `walkie_agent/capabilities/` | Built |
| 2. FakeRobot, Mahoraga's FakeCtx | `walkie_agent/capabilities/fake.py` | Built, fake only |
| 3. Error type | `walkie_agent/errors.py` | Built |
| 3. Atomics | `walkie_agent/tools/atomic/` | Built |
| 3. Tool sets | `walkie_agent/tools/sets.py` | Built |
| 4. Adapter to LangChain tools | `walkie_agent/tools/adapter.py` | Built |
| 5. `create_agent`, middlewares, Runner | `walkie_agent/runtime/` | [Built, and used by `main.py`](#runtime) |
| 5. System prompts | `walkie_agent/prompts.py` | [Built](#runtime) |
| 1 to 2. ROS-backed verbs, operator topics | `walkie_agent/ros/` | [Built, run on fake robots and in the MuJoCo simulation](#ros-wrappers) |
| Entry point: flags, signals, assembly | `walkie_agent/main.py` | [Built, run on fake robots and in the MuJoCo simulation](#wiring-in-mainpy) |
| 6. `run_script` and its shelf | `walkie_agent/scripting/` | [Built](#run_script), used by the gym and, behind `--scripts`, by the agent; run on fakes only |
| 6. Arm D's stand-in bundles | `walkie_agent/tools/bundled_preset/` | Built, stand-ins only |
| 6. Gym | `walkie_agent/mahoraga/`, with its own pixi environment | [Built](#mahoraga), fake world only |

`errors`, `capabilities`, `tools` (with its adapter), `runtime` and `scripting` never
import rclpy, rai, rai_interfaces or any ROS message. `tests/test_ros_free.py`
checks this in a fresh interpreter.

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
Team(name: str, country: str, affiliation: str)

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
class Facts(Protocol):
    def now(self) -> datetime: ...  # timezone-aware
    def team(self) -> Team: ...

Ctx(cancel: CancelToken, base: Base, speech: Speech, vision: Vision,  # frozen dataclass
    arms: Arms = UnavailableArms(),  # UnavailableArms raises UNAVAILABLE on every verb
    facts: Facts = UnavailableFacts())  # so does UnavailableFacts
```

A query of `""` matches everything. Otherwise it matches an object's name,
category or plural category, or a person's name, pose or gesture, ignoring
case. The vision verbs and the arm verbs work only on `FakeRobot`. On the robot's
`ctx`, `UnavailableVision` and `UnavailableArms` raise `UNAVAILABLE`, until a live
perception backend and an arm commander client exist. S1 later added `RosVision`,
whose `people` reads walkie-vision's people when the agent runs with `--vision`
([Perception](perception.md#what-is-built)).

`ctx.facts` answers the talk commands that ask for the time, the date or the team
(decision M35). Its verbs never wait, so they do not check the cancel token.
`walkie_agent.capabilities.facts.SystemFacts` is the robot's source, and imports no
ROS (M36). It reads the computer's wall clock, not ROS time, in the timezone of
`config.toml`'s `[team]` table, which also holds the team's name, country and
affiliation (M37). `main` builds it from that table and passes it to `assemble`. A
missing table, a blank value or an unknown timezone stops the agent before
`rclpy.init`, with one line on stderr and exit code 2 (M56). `assemble` without `facts` keeps `UnavailableFacts`.

### Guards live in the verbs

`walkie_agent.capabilities.guards` holds the shared checks. Every
implementation of a verb, fake or ROS-backed, calls them.

```python
MAX_MOVE_M = 3.0  # the cap the old navigation tool had (ADR-0031)
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
  "stop" can answer with its fixed line (decision 6). `as_tool` and `bind` also check
  the token before an atomic runs, so the `say` tool refuses once a stop is set. Only
  the stop line calls the speech verb directly.
- `RosBase` keeps one record per goal, from its reservation until Nav2 reports an
  end. A verb that ends in `CANCELLED` or `TIMEOUT` marks the record abandoned for
  good, and sends the cancel when the goal is accepted, however late. A cancel
  acknowledgement ends nothing. While a record lives, a new motion gets
  `PRECONDITION_FAILED`. See [ROS wrappers](#ros-wrappers) for the limits.

## FakeRobot

```python
# walkie_agent.capabilities.fake
class FakeRobot:  # implements Base, Speech, Vision, Arms and Facts
    def __init__(self, world: Mapping[str, Any] | None = None, *, seed: int = 0,
                 drive_s: float = 0.0) -> None: ...
    ctx: Ctx
    cancel: CancelToken
    calls: list[dict[str, Any]]   # {"verb": "base.go_to", "args": {...}, "error": None or a code,
                                  #  "at": place or None, "room": room}, and for a lookup "returned": [ids]
    said: list[str]
    faults: dict[str, StepError]  # verb name -> the error every call to it raises; not base.stop
    def snapshot(self) -> dict[str, Any]: ...  # JSON-able copy for checkers: at, room, pose,
                                               # holding, places, objects, people, facts, said, calls

def chula_pat14_6(seed: int = 0) -> dict[str, Any]  # a seeded world description
def fake_facts(seed: int) -> dict[str, Any]         # the clock and the team of that world
```

A world description is JSON-able, so the gym can build, store and compare it:

```python
{"places": {"shelf": {"x": 2.487, "y": -6.409, "yaw": -1.57, "room": "kitchen room"}},
 "objects": {"apple_1": {"name": "apple", "category": "fruit", "size_cm": 8, "weight_g": 180,
                         "thickness_cm": 8, "at": "shelf"}},
 "people": {"operator": {"name": "Jane", "room": "living room", "pose": "standing person",
                         "gesture": "waving person"}},
 "facts": {"now": "2026-11-05T14:05:00+07:00",
           "team": {"name": "EIC", "country": "Thailand", "affiliation": "Chulalongkorn University"}},
 "start": "instruction point"}
```

The clock is fixed for an episode (M38). `fake_facts` draws it from the seed, on its
own random stream so the rest of the world is unchanged: any date in 2026 and any
minute of the day, at UTC+07:00. The same seed gives the same clock. The team is
the one in `config.toml`. `facts` is optional. Without it both facts verbs raise
`UNAVAILABLE`, and the snapshot's `facts` is `None`.

A place whose `room` is its own name is a room. An object's `at` is a place,
`"robot"` while held, or the id of the person who received it. The robot sees
what is at its place, or at every place in its room while it stands at the room
itself, and the people in its room. `world=None` builds `chula_pat14_6(seed)`,
and `drive_s` makes each drive take real seconds.

A call record's `at` and `room` are where the robot was as that call ended. `at` is
`None` once a relative move has left its place by more than 1 m, and `room` is then
the room of the nearest place. A successful `vision.objects` or `vision.people`
record also holds the ids it returned in `returned`.

The constructor raises `ValueError` for a broken world, naming the record and
field: an unknown start, a place or person in an unknown room, an object at
something that does not exist, a coordinate that is not a finite number, a clock
that is no ISO 8601 date and time with a UTC offset, or a blank team value.
Without this, a misspelled name hides the object or person, and the episode
reads as the agent's failure. Whether a world suits a given command is the gym's
check, not this one.

Places and rooms come from walkie-ros `origin/feat/chula-pat14-6-simulation`
(`simulation/mujoco/worlds/chula_pat14_6/authoring/source/30_16102026_ChulaPat14_6_world.toml`).
The `trash` and `instruction point` places, six objects, all measures and the
people are invented (decision 26). A builder that seeds a feasible world for
one generated command belongs to the gym.

## Atomics

Plain typed functions over `ctx` in `walkie_agent.tools.atomic`. They return
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
def face_person(ctx: Ctx, person_id: str) -> str
def pick(ctx: Ctx, object_id: str) -> str
def place_on(ctx: Ctx, place: str) -> str
def hand_over(ctx: Ctx, person_id: str) -> str
def get_facts(ctx: Ctx) -> dict[str, Any]

# walkie_agent.tools.sets
GPSR: list[Callable[..., Any]]     # the nine the spike used, every atomic but list_places and face_person, then get_facts
HRI: list[Callable[..., Any]]      # list_places, go_to, move_relative, say, find_people, face_person
DEFAULT: list[Callable[..., Any]]  # every atomic that needs no arms, get_facts included
```

`get_facts` returns the time as 24-hour `HH:MM`, `today` and `tomorrow` such as
"Thursday, 5 November 2026", `day_of_week`, `day_of_month` as a number, and
`team_name`, `team_country` and `team_affiliation`. Tomorrow is the next calendar
day in the clock's timezone, so it crosses a month or a year. HRI does not take it.

## Adapter

```python
# walkie_agent.tools.adapter
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

The research gym, the package `walkie_agent.mahoraga`, in the `mahoraga` pixi
environment and solve group. inspect-ai and the GPSR generator stay out of
`default`, which `tests/test_mahoraga_absent.py` checks, and so do the gym's tests
in `tests/mahoraga/`, which the default environment does not collect. The gym never
imports ROS, and the agent never imports the gym. `tests/test_ros_free.py` and
`tests/test_mahoraga_absent.py` check both. It reaches the foundation through
`walkie_agent.tools`, with its adapter, and `walkie_agent.scripting`, and it covers rungs R0
to R2 in the fake world. MuJoCo validation and promotion are not built.

`run_script`'s runtime is the package `walkie_agent.scripting`. It imports no ROS
and nothing from the gym, and its only dependency beyond the foundation,
pydantic-monty, is in `default` too.

| Module | What it holds |
| --- | --- |
| `scripting.driver` | `ScriptRunner`, `Budgets`, `ScriptResult`, `allow_list`, `stubs_for`, `describe_functions`, `script_summary`, and the evidence rule shared by the scorer and the verdict: `script_record`, `proven` |
| `scripting.agent_tool` | `AgentScripts`: the agent's `run_script` behind `--scripts`, its task records and the operator's verdict |
| `scripting.shelf` | `Shelf`: saved functions in the git-ignored tier folders `temporary/` and `bundled/`, and `ledger.jsonl`; `store_root`, `require_ignored` |
| `scripting.evidence` | `Watch`: which functions a run entered, returned from or raised through, from counters in the worker |
| `scripting.sandbox` | `open_pool`, the Monty pool with each worker inside bwrap. It fails closed. |
| `tools.bundled_preset` | `BUNDLES`, stand-ins for arm D until the team's bundles exist |
| `mahoraga` | `home()`, the gym's home for gate results and run logs |
| `mahoraga.commands` | `RecordingGenerator`, `parse`, `commands(split, n, category)`, and the arena's names in `data/` |
| `mahoraga.worlds` | `build_world(spec, seed)`, `world_seed(sample_id, epoch)` |
| `mahoraga.checkers` | `check(spec, start, end)` over `FakeRobot.snapshot()` |
| `mahoraga.oracle` | `oracle_script(spec)`, written from the slots with no LLM |
| `mahoraga.arms` | `ARMS`, `INDUCT`, `ArmSetup(arm, pool=, shelf_root=).episode(ctx)` |
| `mahoraga.tasks` | `dataset`, `episode_robot`, the `ground_truth` scorer |
| `mahoraga.gate` | `run_gate`, `run_standard_gate`, `require_gate`, `GateError` |
| `mahoraga.gym` | `gpsr_task`, `run_episodes`, `resolve_model` |
| `mahoraga.report` | `report(log_dir, shelf_root=)`, `arm_metrics`, `compare`, `mcnemar_p`, `recorded_shelf_root` |

### `run_script`

```python
ScriptRunner(pool, *, shelf=None, budgets=Budgets()).run(code, fns, *, save=(), cancel=None) -> ScriptResult
```

The gym and, behind `--scripts`, the competition agent run scripts through it. The
agent's tool, its budgets and the operator's verdict are in
[Scripts on the agent](agent.md#scripts-on-the-agent).

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

The wall clock is checked after every host call and before a run reports success,
so a last slow call that overruns fails the budget, and no host call starts after
the budget is spent. A call already running still finishes first: nothing interrupts
a host function, so a run can outlast its budget by one call. The script then gets
up to one more feed's time to finish, with every further call refused, so that the
evidence below stays readable.

`cancel` is a `CancelToken`. Once it is set, no further host call runs and the
script ends the same way, with phase `cancelled` and an error it cannot catch. The
agent passes the Runner's token, so a stop or a reset ends a running script.

`ScriptResult.calls` is the ledger: one entry per call, with its status. In the gym
the model sees only `script_summary(result)`. The agent's tool adds the ledger as
`tool: status` lines, so that the model and the tool log see each call.

### Shelf

`Shelf(root=None, *, frozen=False)` keeps its functions in two tier folders of a
store root, with `ledger.jsonl` beside them:

| Path | Holds | Written by |
| --- | --- | --- |
| `temporary/<name>.py` | The latest saved version of each function | A run with `save=[names]` that succeeds |
| `bundled/<name>.py` | The last verified version of each function, with exactly the text that ran, so its sha is the one in a verified event | `mark_verified`, which the gym's scorer calls after a success and the agent after the operator's passed verdict, and a save of a version verified before |
| `ledger.jsonl` | The saved, run, fail and verified events, each keyed to a version's sha, and on the agent the rejected events of an operator's failed verdict, which `counts()` does not count | Every save, run and verification, and the agent's failed verdicts |

Which version `bundled/` holds follows one rule: the version in `temporary/` if
the ledger verifies it, and otherwise the version verified last. `mark_verified`
writes that after its events, and a save writes a version to `bundled/` when the
ledger already verifies it. So an episode that ran an older version and is scored
after the latest one was verified, as can happen while `induct` runs eight episodes
at once, does not replace the latest, and a script that saves an earlier verified
version again puts it back.

The ledger stays the record. The store root is `root`, then `WALKIE_TOOLS_HOME`,
then the tools package's own folder, `src/walkie_agent/tools`. The tier folders'
tracked `.gitignore` files and a scoped rule in the repository's `.gitignore` keep
everything the shelf writes there out of git. `require_ignored` checks that before
anything is created, and it fails closed: inside a git work tree, `git check-ignore`
must report a saved file and an atomic write's scratch file in each tier folder,
and the ledger, as ignored. A tracked file at one of those paths, or git failing,
refuses the store too. Without git, only the package's own folder passes inside a
work tree. A store outside every work tree always passes. The error names what git
does not ignore, and says to ignore it or to set `WALKIE_TOOLS_HOME` to a folder
outside any work tree.

A writable shelf, as arm `induct` has, offers everything in `temporary/`. A frozen
shelf, as arms C and C2 have, fixes its callable set when it is built, from
`bundled/`: a function counts only if the ledger holds a verified event for the
sha of its file there and `temporary/` still holds that same version. So a rewrite
that has not been verified keeps its function off, and a file copied into
`bundled/` by hand, with no verified event, is ignored. With the rule above, the
callable set is the latest version of each function that the ledger verifies, as
`counts()` reports it and as the shelf chose before the tier folders, whatever order
episodes were scored in. A verified function that names a saved function left off
stays off as well (M18). `mark_verified` refuses a version whose text the shelf
never read or wrote, with nothing logged.

The gym's home keeps only gate results and run logs. It is the `--home` argument,
then `MAHORAGA_HOME`, then `~/.local/share/mahoraga`, and it may not be inside a git
work tree. The shelf is not in it: `--shelf` names the store root for `run` and
`report`, and each run's log records the store root and the callable set's digest
in its metadata, as `shelf_root` and `shelf_digest`. A report without `--shelf`
reads the shelf its logs record, and refuses logs that ran on different shelves.
It only reads that store: it creates no folder there and runs no git (T34).
Logs that record none, of arms A and B alone or from before the tier folders, read
the default store: `WALKIE_TOOLS_HOME`, then the package folder. The report's
`shelf.root` names the store its M10 read. The `report` command prints `Refused:`
and exits 2 only for a store the shelf refuses (`StoreRefused`), and any other error
keeps its traceback.

What a function earns comes from what it did at run time, never from its name in a
script. `evidence.Watch` wraps each shelf function a script loads and each definition
it may save, and counts entries and normal returns inside the worker:

| Evidence | Means | Feeds |
| --- | --- | --- |
| Used | Entered at least once | `shelf_used`, run and fail events, M7 |
| Raised | Entered more often than it returned: an exception crossed its boundary, even if a caller caught it, or the run ended inside it | `shelf_raised`, M8 |
| Verified | Returned, never raised in that episode, and the episode scored 1, or, on the agent, the operator passed the task | The `verified` event |

Each event carries the sha of the source that ran, so a rewrite starts with nothing.
If an episode ran several versions of one name, only the last that returned and never
raised is verified. A function saved but never run stays unverified, so it stays off
a frozen shelf. So does a verified function that names a saved function which never
ran, because it could not stand alone without it. A script's own definition earns a
return only if the script saves it, and it is no use of the shelf. While the shelf can
still verify, a definition the script bound more than once, a decorated one, or one
under an `if` or inside another function goes unwatched and counts as raised for its
version (M51).

The runner gives Monty each script with its line endings as `\n`, as Python reads a
source file, so a string across lines holds the same value when the script runs and
once its definition is saved (M62). It runs only a script that Python's parser
accepts, because the evidence reads definitions with that parser: Monty would run
some source it refuses, such as one with a null byte, with no definition watched
(M63). The host parses the script before Monty's budgets apply, so a parser or an
evidence step that runs out of stack or memory ends the script as a refusal, like
any other failure, with nothing run or saved (M67). A version's sha is of its text as
written, from its `def`, which is what the shelf file and the ledger hold. Each
script's record also maps the sha of every version it named to a shape: a loaded
shelf function, and, while the shelf can still verify, each of its own definitions
at any depth. A shape is the sha of the definition's syntax tree without positions,
its own decorators or `u` string prefixes (M64). A returned version is not verified
when a version with the same name and shape, or the same sha, raised or went
unwatched anywhere in the episode. A syntax tree keeps no layout, comments or line
endings, only the code's structure, names and values, so a copy under an `if`, a
copy with a comment, and the same function at top level are one version for this
veto, while a copy whose code differs, or whose string holds other spaces, is not
(M60). Equal trees do not mean equal behaviour: a copy that reads other names from
its scope, or that another decorator wraps, can compute something else under the
same shape, which errs toward blocking. A copy is matched only as a definition under
the saved name. One bound to that name another way, such as a renamed definition assigned to
it, a lambda or a comprehension variable, is not seen, and neither is a copy with a
docstring or an annotation added (M65). A record without shapes, from before M60,
compares shas.

The counters' names carry a random token per run, and a script that loads or saves
a function gets `locals`, `eval` and `exec` shadowed, so it cannot name or reach
them.

The shelf's source feeds before the loader, because a wrapper needs the function it
stands in for. A parameter default on a saved function is evaluated in that feed,
before the shadows and wrappers exist. So `evaluate=eval` would keep the real `eval`,
which can forge or erase counters, and `fn=helper` would keep a helper whose runs and
raises nobody counts. `Shelf.save` therefore refuses a function with a default that is
not a literal, such as a name, call, lambda or attribute, and tells the model why. A
function that needs another one names it in its body instead, where the lookup happens
at run time and finds the wrapper. `Shelf.save` also refuses the names `locals`, `eval`
and `exec`, and they are never watched, because the loader defines them itself and a
wrapper would give its replacement the source's sha. A script may still define such a
name for its own use. Decorators are refused too, and Monty does not evaluate
annotations, so no other part of a saved definition runs in that feed.

With those rules the counters hold against a careless or clever model, and they are
no security boundary. The rules apply when a function is saved, so a shelf written
before they existed is not rechecked.

Two limits come with it. A wrapper adds a frame, so recursion through a watched
function reaches the depth limit in half the calls. If the counters cannot be read,
after a worker crash or a script that keeps running well past its wall budget, the
run has no evidence, so it earns nothing.

### Arms

| Arm | The model gets |
| --- | --- |
| `A` | The per-step GPSR tools |
| `B` | `run_script`, with no shelf |
| `C` | `run_script`, with the frozen, verified shelf callable |
| `C2` | `run_script`, with the same scripts as prompt examples, not callable |
| `D` | The per-step tools plus `BUNDLES` |
| `induct` | `run_script` with `save`, on a writable shelf, for training only |

`induct` runs alone and only on the train split. `run_episodes` and the CLI refuse
anything else with a `ValueError`, and `gpsr_task` refuses it for commands outside
that split, because a later arm's frozen shelf would then hold answers to the
commands it is measured on.

`induct`'s prompt tells the model it is building a library for later commands, to
write a command's steps as a function with the names as parameters, and to name that
function in `save`. Offered only as an option, saving did not happen at all.

All arms share the prompt, the atomics and the eval's model, temperature and seed.
One (command, epoch) is one seeded world, the same for every arm. The prompt tells
each arm to read the time, the date and the team with `get_facts` and never to guess
a fact (M40). The oracle's `talk` scripts read `get_facts` too.

An episode scores 1 only if its checker passes, no limit ended it (rounds, tokens or
time), and on a script arm its last script ran to completion. So work done before a
script is cut off, or before a limit, does not count. An earlier failed script the
agent recovered from stays allowed.

### Checkers

`check(spec, start, end)` judges an episode from two snapshots and the records of
its successful calls. Each record says where the robot was and what a lookup
returned, so a check never rebuilds a trail from `go_to` calls, and a relative move
counts for what it really did.

| Kind | Passes when |
| --- | --- |
| `move_object` | An object of the wanted kind picked from the source ends at the destination |
| `count_objects`, `best_object`, `count_people` | The answer is right after a look from there that returned what it is about |
| `person_info` | The answer is right after a look in that room that returned the person |
| `relay_info` | A look returned the source and a look in the recipient's room returned someone there, in either order, then the right answer is said in the recipient's room. A second look at the recipient is not needed |
| `talk` | A lookup at the requested place returned the requested person and a read of the topic's fact succeeded, in either order, then some line in their room says that fact. For "something about yourself", any line in their room, with no read |
| `escort` | That same lookup, then the greeting if asked for, then a move that reaches the destination. The robot may move on afterwards. A destination that is the named origin or holds it needs no further move |

An answer is the first line that mentions a candidate after the last look that
showed what it is about. That line must assert one distinct candidate, the right
one. Later lines go unread, so "Task 99 is complete." after a correct count does not
undo it. Looks that each show part of what is asked add up, and a later look at
something else is not looking again.

A candidate is a number for a count, the name of any object at that place for
`best_object`, and a known value of the attribute for `person_info` and
`relay_info`.

A number is read as a whole token. "1.0" is 1, while "1.1", "-1" and "twenty-one"
are numbers of their own that no small count equals, and "3rd" is no number. Number
words run to nineteen, the tens and "hundred", so "thirty dishes" is a wrong answer
and not a line to skip. "single" means 1, and "couple" and "pair" mean 2.

Ordinary equivalents count for pose and gesture, so "seated" says sitting. A raised
arm also needs an arm or a hand in the line, so "raising their right leg" does not
say it. Pointing and waving need no body part.

A `talk` fact is judged against the start snapshot's `facts` (M39). The first-line
rule does not apply: any line said in the person's room after the lookup that found
them counts. The line must also come after a successful read of the fact: `facts.now`
for the time and the four day topics, and `facts.team` for the three team topics.
`get_facts` makes both reads, before or after the lookup. A right line with no read
before it fails, and the reason says so (M53).

| Topic | A line passes when it says |
| --- | --- |
| "the time" | The time |
| "what day is today" | Today's weekday or today's date |
| "what day is tomorrow" | Tomorrow's weekday or tomorrow's date |
| "the day of the week" | Today's weekday |
| "the day of the month" | Today's day number |
| "your teams name", "your teams country", "your teams affiliation" | That team value |
| "something about yourself" | Anything (M41) |

- A time is `H:MM` or `HH:MM` in 24-hour form, `h:MM` in 12-hour form, "h o'clock"
  on the hour with the hour in digits, or a digit hour right before a qualifier, such
  as "12 am", "2 p.m." or "12 noon", which is that hour on the hour. Every form reads
  its hour the same way (M55). A qualifier is optional and must be right when given:
  am or pm in any case and with or without dots, "in the morning" for am, "in the
  afternoon" or "in the evening" for pm, and "noon" or "midnight" only after 12 on
  the hour, at 12:00 or 00:00. An hour with a leading zero and no qualifier reads as
  24-hour, so "02:05" is not 14:05 and "02 o'clock" is not 14:00.
- A day topic asks about today, except "what day is tomorrow". A line that names
  another of today, tomorrow and yesterday, and not the day asked about, does not
  count for it, so "Tomorrow is Thursday." does not say that today is Thursday (M54).
- A weekday is its English name as a whole word, in any case.
- A date is the month's full name and the day's number in either order, with an
  optional ordinal suffix, "of" or "the": "5 November", "5th of November",
  "November 5th". The year is not read.
- A day of the month is digits with an optional ordinal suffix, or an English word
  from one to thirty-one, such as "fifth", "twenty-one" or "twenty first". Digits
  beside a colon are a time and do not count.
- A team value is its words as a whole phrase, in any case.
- A topic with no check, or a checked topic in a world without facts, raises
  `ValueError`.

Known limits, each pinned by a test or stated here because reading more would mean
parsing language:

- A correct answer followed by a correction still passes, whether or not the
  correction names a candidate. A robot that changes its mind has to look again.
- Negation is not read. "It is not the banana." counts as saying banana.
- A true line with a second number or candidate fails, such as "There are two, one
  apple and one banana" or "The banana is bigger than the apple". "One" counts as a
  number even as a pronoun.
- A bare article is not a count, so "There is a dish" fails. Reading "a" as one would
  make "on a bar" a second number.
- A look again at the same things with no answer after it leaves the earlier answer
  standing, right or wrong. An answer after it takes over from the earlier one.
- A line that also names a leg or a foot says no raised arm, so "Their left arm is
  raised and their legs are crossed" fails. Words are not attached to each other.
- A shortened name such as "the plate" for "red plate" is not a candidate.
- What a "something about yourself" line says is not checked, so a refusal to the
  right person, found at the right place, passes.
- A time in words fails, such as "quarter past two", "half past two", "five past
  two" or "noon" alone. So do "two o'clock", a time with seconds such as "14:05:30",
  and a month's short name such as "5 Nov".
- A qualifier goes with a 12-hour time, so a right 24-hour time with one fails, such
  as "14:05 pm", "14:05 in the afternoon" or "00:00 midnight".
- "at night" is not read, so "10 at night" fails at 22:00 and "2:05 at night" passes
  at 14:05.
- The checker sees that a read came before the line, not that the line used it.
- A line that names both today and tomorrow passes for either, such as "Today is
  Thursday and tomorrow is Friday."
- Any number in a line counts for the day of the month, so "It is the one you
  wanted" passes on the 1st, and "wait a second" on the 2nd.
- Ordinal suffixes are not matched to the number, so "5rd November" says 5 November.
- An escort cannot show that the person came along, since the fake's people do not
  move. A `go_to` that leaves the robot where it stands counts as the move.

### The oracle gate

An LLM arm refuses to start until the oracle has scored 100% twice with identical
logs, for the same code and the same data.

`run_gate` plays the oracle's scripts through the same driver, world builder and
scorer an arm uses, twice, into `.eval` logs. It reads both back and compares them
as JSON, with samples in a fixed order and without the fields in `gate.VOLATILE`:
clocks, random ids and the log's location. A pass also needs every oracle script to
have run to completion and no limit event. The result lands in `<home>/gate/`.

`require_gate(records, epochs)` raises `GateError` unless passing results for the
current `code_digest()` cover a digest of every (command, epoch, world) requested.
A result counts only if `passed` is true, `failures` is empty, its two pass digests
are equal, and both logs exist and still hash as the gate recorded. This catches a
stale, partial or hand-edited result made by mistake. It is no security boundary:
nothing is signed. `gpsr_task` and `run_episodes` both call it.

The code digest covers each file's repository path and content, for the files
`gate._digested_files()` lists: every `walkie_agent/mahoraga/*.py` except
`report.py` and `__main__.py`, the gym's data files, `walkie_agent/__init__.py`,
`errors.py`, `locations.py`, the `scripting`, `capabilities` and `arm` packages,
`tools/__init__.py`, `tools/adapter.py` and `tools/sets.py` with the `tools/atomic`
and `tools/bundled_preset` packages, the perception modules the fake world uses, and `pixi.lock`. So an edit to
any of them, related to the oracle or not, means running the gate again, which takes
about a minute.

### Running it

Run these with `pixi run --frozen -e mahoraga`, from the repository root. The gate
is offline and costs nothing. `run` calls a real model, so `--commands` has no default.

```bash
# The tests: the gym's, run_script's, and the gym's ROS-free check
python -m pytest tests/mahoraga tests/scripting tests/test_ros_free.py

# The gate on the plan's trial, training and held-out data: 1080 episodes, twice
python -m walkie_agent.mahoraga gate
# Or on other data
python -m walkie_agent.mahoraga gate --split trial --commands 20 --epochs 1 --category people

# A small real slice, with the key in the environment
OPENAI_API_KEY=... python -m walkie_agent.mahoraga run --arms A,B --commands 5

# Metrics M1 to M10 and the paired comparison, as JSON
python -m walkie_agent.mahoraga report --log-dir <the run's directory>
```

The defaults are `deepseek/deepseek-v4.1-flash:nitro`, through OpenRouter at
`https://openrouter.ai/api/v1`, with the key in `OPENAI_API_KEY`. Other flags are
`--epochs`, `--split`, `--category`, `--model`, `--base-url`, `--temperature`,
`--seed`, `--max-samples`, `--log-dir`, `--shelf`, `--no-bwrap` and `--home`.

The shelf's store is `--shelf`, then `WALKIE_TOOLS_HOME`, then
`src/walkie_agent/tools`, whose `temporary/` and `bundled/` git ignores. To start a
run from an empty shelf, point `--shelf` at a new folder outside the repository, and
give `report` the same `--shelf`, or let it read the store root the logs record:

```bash
OPENAI_API_KEY=... python -m walkie_agent.mahoraga run --arms induct --split train \
    --commands 5 --shelf /var/tmp/shelf-trial-1
python -m walkie_agent.mahoraga report --log-dir <the run's directory>
```

Stores from before the tier folders, such as `~/.local/share/mahoraga/shelf` or a
`shelf/` folder in another home, are not migrated, read or touched. They hold
`<name>.py` files beside their ledger with no tier folders, so passing one to
`--shelf` gives an empty shelf, and its first save would mix the two layouts. Leave
them where they are, and start a new store instead.

The plan's R2 is then `run --arms induct --split train --commands 300`, followed by
`run --arms A,B,C,C2,D --split heldout --commands 160 --epochs 3`, both on the same
store. These are two runs on purpose, and each arm's shelf freezes as it starts. Generation keeps the splits
apart: the seed ranges differ, and `commands("heldout", ...)` skips any
sentence in the standard 300-command training set. A training run longer than that
gets no such promise.

An episode that errors, such as on a provider failure, gets no second attempt: a
retry would replay the shelf's ledger and lose the failed attempt's usage. A transient
provider error is retried inside the model call instead, so it costs no attempt. Two
layers do that: the OpenAI client makes up to three requests, and Inspect repeats that
up to three more times with backoff, so a provider that stays down sees up to twelve
requests per model call. An attempt that takes over 30 s is abandoned and retried the
same way, because a completion that runs away would otherwise spend the episode's
300 s limit and fail it. An episode that still errors, or never ran, stays in
M1 as a failure, and the usage it logged counts towards tokens and cost. Rounds, wall
time and the script metrics cover every episode logged. Any such episode sets the
arm's `complete` to false, and that of each comparison with the arm. `report` then
gives a plain `warning`, which is null only if every arm is complete.

M3's input tokens are everything the model was sent, with the cached part also given
alone. Its cost is what the provider recorded in each raw response, which needs
`log_model_api=True`. A call with no record, such as an abandoned attempt, counts in
`M3_cost_unrecorded_calls`, and any such call makes the cost a lower bound. A provider
that returns no cost at all makes it null. The report holds no price list, because the
plan's prices were wrong for this route.

A comparison's own `complete` is also false when its arms were scheduled different
(command, epoch) pairs, or when one sample id holds different command text in the two
logs. A world's seed follows from (command, epoch), so the same pairs mean the same
worlds. A log with no `sample_ids` still schedules `dataset.samples` times its epochs,
so an episode it lost counts as never run.

## Runtime

`walkie_agent.runtime` is `create_agent` plus a small stack of middlewares, the
todo list and a Runner. Tools come from `as_tools` over a `Ctx`, errors go through
`tool_error_middleware`, and `ctx.cancel` is the only cancel signal. `main.py` starts
it through `assemble`, and the [wiring](#wiring-in-mainpy) is below.

```python
# walkie_agent.config
def load_config(path="config.toml") -> dict: ...
def agent_settings(config) -> dict: ...  # run_limit and deadline_s, from the optional [agent]

# walkie_agent.chat_models
def build_model(config, role="complex_model") -> BaseChatModel: ...
def build_fallback(config) -> BaseChatModel | None: ...  # needs fallback_vendor and fallback_model
def build_callbacks(config) -> list[BaseCallbackHandler]: ...  # Langfuse, when enabled

# walkie_agent.prompts
def build_system_prompt(tool_set, locations, robot_description, *, vision=False, memory=False,
                        arms=False, object_memory=False, scripts=False) -> str: ...
SYSTEM_PROMPT: str  # build_agent's default

# walkie_agent.runtime
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

`on_todos(items, changed)` gets public copies, and `changed` holds the items whose
status changed. Both listeners run on the Runner's thread and must not block. The
model gets two tools beyond `tools`: `update_todos(ops)` and
`wait_for_person(question)`. The warning at more than 20 bound tools counts them. A
tool whose name collides with these two is refused at build.

### The middleware order

`build_middleware` has the one order that works, and `tests/test_runtime.py` pins it.
`after_model` hooks run in reverse list order, and a jump skips the ones still to run.

| Middleware | Guards |
| --- | --- |
| `ToolLog`, when configured | Logs every answer, refusals included |
| `StopLatch` | Once the token is set: no model call, and every tool call refused |
| `SerialGuard` | A failed call or a question skips the later calls of its message; only the first `update_todos` of a message runs, and a refused copy counts as a failed call |
| `AnswerEmptyReply` | A final message with no text goes back to the model once per run, unless a `say` in the run succeeded |
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
  refuses every remaining call of the run, even when the agent is idle. It also calls
  `ctx.base.stop()`, as a reset does, so a goal whose first cancel Nav2 refused gets
  another one after its verb has returned (W19 in the
  [decision log](mahoraga-demo-decisions.md#ros-wiring)). `StopLatch`
  refuses a call before it is dispatched. `as_tool`, `bind` and the two todo tools check
  the token again when the body starts, for a stop that lands in between. A `say` already
  speaking is not cut off. It drops queued tasks but not a queued reset. The Runner speaks
  `STOP_LINE` once through `ctx.speech.say` and logs it under the session it was heard in.
  A later graph pass with no model call cancels the open steps and clears a waiting
  question. The token stays set until the next task starts. A compound such as "stop and
  come here" is an ordinary task, so it queues behind the active run.
- **Reset.** It cancels and drops the queue. After the active run has finished, it
  deletes the thread, starts a new session and publishes an empty list. A hung LLM call
  can delay this by up to the deadline. A reset during a stop line waits for that line,
  bounded by the say timeout.
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
provisional until real runs measure them. Its `[team]` table is required, and is
read by `walkie_agent.capabilities.facts`, not by the runtime.

All of these promises are tested with a scripted model. `tests/test_chat_models.py`
checks the provider settings against a local fake provider. No real provider has been
called.

## ROS wrappers

`walkie_agent.ros` holds the capability backends and the operator's topics, one module
each. The modules load rclpy, so the foundation never imports them. The package init
imports nothing, so each class is imported from its own module.

```python
# walkie_agent.ros.base
class RosBase:  # Base over Nav2's /navigate_to_pose and TF map <- base_footprint
    def __init__(self, connector, locations, cancel, *, ...): ...
    def close(self, timeout_s=5.0) -> bool: ...  # False while a goal lives; the client stays
    def unresolved(self) -> list[str]: ...       # what kept the last close from finishing

# walkie_agent.ros.speech
class RosSpeech:  # Speech over /to_human, with the speech id rotated on /voice_commands
    def __init__(self, connector, *, ...): ...
    def close(self) -> None: ...

# walkie_agent.ros.vision
class RosVision:  # Vision with --vision: people from /walkie_vision/people (added in S1)
    def __init__(self, connector, base, cancel, *, topic=..., max_age_s=1.0, wait_s=1.0, ...): ...
    def close(self) -> None: ...

# walkie_agent.ros.operator_topics
class RosBridge:  # the operator's topics; see the topic table on The Agent
    def __init__(self, connector, templates, *, input_topic=..., output_topic=..., ...): ...
    def open(self, runner) -> None: ...
    def on_todos(self, items, changed) -> None: ...  # safe from any thread
    def on_reply(self, text) -> None: ...            # safe from any thread
    def close_ingress(self) -> None: ...             # ignores input, keeps subscriptions
    def close(self) -> None: ...

# walkie_agent.ros.policy
FORBIDDEN_NAMES = ("/cmd_vel", ..., "/walkie/reset", "/walkie/task", "/walkie/todos", "/walkie/verdict")
def check_endpoint(node, name, forbidden=FORBIDDEN_NAMES) -> str: ...  # StepError REFUSED
```

- **`say` returns once published.** There is no playback acknowledgement, so it means
  "sent".
- **Input is text only.** A `/from_human` message with an image or audio payload is logged
  and refused, and that includes a bare "stop" with a payload.
- **Forbidden names are checked at construction.** `RosBase` and `RosSpeech` call
  `check_endpoint`. It is a guard against a configuration mistake and not security.

Limits that the fakes cannot settle:

- **TF poll.** RAI's transform helper never checks the buffer at a zero timeout, and
  sleeps 0.1 s on a miss. So `_TF_LOOKUP_S` is 0.01 s, and a stop during a TF read can take
  about 0.1 s plus scheduling to land.
- **TF freshness.** A sample stamped more than 0.5 s ahead of the clock reads as stale, so
  clock skew reads as `TIMEOUT`. Stale TF is `TIMEOUT`. Missing TF with no publishers is
  `UNAVAILABLE`, and the message names the domain and the router. Freshness on the real
  robot is unmeasured.
- **Pending-cancel cleanup.** A cancel acknowledgement is not terminal and does not prove
  the wheels stopped. `RosBase.close` returns False, and `unresolved()` says so, while a
  goal has no terminal result: "Nav2 has not answered the goal", "a cancel was requested,
  but no terminal result has arrived" or "no terminal result has arrived". The hold, retry
  and forced exit have run only against fakes, never against a real unanswered Nav2 goal.
- **Unknown state holds the gate.** A failed acceptance future, a failed result future or
  `STATUS_UNKNOWN` keeps the record. If Nav2 restarts and forgets a goal, motion stays
  refused until the agent restarts.
- **Real Nav2 races.** The race matrix runs against fakes, and the acceptance tests run
  against a `FakeWalkie` action server. Real Nav2's timing is unknown.
- **Stop latency.** The 0.5 s terminal cancel is a fake-only test criterion. It is not a
  braking figure. The 2 s discovery, 2 s acceptance and 1 s TF budgets are implementation
  defaults, not measured needs.
- **Shutdown noise.** The connector's spin thread can still die during the final destroys or
  RAI's `connector.shutdown()`. That can print a traceback at shutdown, and the exit code
  is unaffected.

## Wiring in main.py

```python
# walkie_agent.main
def assemble(*, connector, model, config, locations, templates, robot_description,
             tool_set="default", callbacks=(), tool_log=None, input_topic=..., output_topic=...,
             reset_topic=..., task_topic=..., todos_topic=..., use_sim_time=False,
             fallback=None, facts=UnavailableFacts()) -> Application: ...

class Application:
    ctx: Ctx
    runner: Runner
    bridge: RosBridge
    def start(self) -> None: ...  # the Runner, then the subscriptions
    def close(self, timeout_s=5.0) -> bool: ...  # False leaves endpoints alive; call it again
    def unresolved(self) -> list[str]: ...
```

Without `--vision`, `ctx.vision` is `walkie_agent.capabilities.UnavailableVision`,
which imports no ROS: it checks its arguments and the latch, then raises `UNAVAILABLE`.

`assemble` builds `RosBase`, `RosSpeech`, `UnavailableVision`, or `RosVision` when asked
with `vision=True`, and `RosBridge`, then
`build_agent` and `Runner`. `Ctx.arms` stays `UnavailableArms`. `Ctx.facts` is the `facts`
it is given, and `main` gives it the `SystemFacts` of `[team]`. It builds no model or tracer,
and it does not start or destroy ROS. A failure part of the way through closes what was built
before it raises.

`Application.close` runs in one order: close the input, cancel and stop the base, shut down
the Runner, drain the base, close speech, close a `RosVision` if there is one, close the
bridge, and with `--scripts` close the script sandbox's workers. Only then does `main` shut down
the connector and the context, on a thread of its own.

With `--scripts`, `main` opens `AgentScripts` from its main thread before ROS starts,
because each bwrap worker dies with the thread that started it, and passes it to
`assemble`. `assemble` binds its `run_script` after the set's atomics through
`build_agent(extra_tools=...)`, gives the bridge its `verdict`, and gives the Runner
its `begin` and `end` as `on_task_start` and `on_task_end`. `main` closes it again
in a `finally`, for a run that failed before `assemble` finished.

Signals use `SignalHandlerOptions.NO`, because rclpy's own handler would shut the context
down before the cancel of a running goal could be sent. The first SIGINT or SIGTERM starts
the shutdown. A second one logs what is unresolved and calls `os._exit(3)`. A failure counts
as the first signal, and it exits with code 1.

The real `main()`, signalled by PID, exits 0 in about 0.8 s on one signal, and exits 3 when
`close` is patched to never finish. Both ran against fakes.

## Provisional decisions

| # | Decision | Pick |
| --- | --- | --- |
| 1 | Atomic tool shape | Plain typed functions over ctx verbs, guards inside the verbs |
| 2 | The as_tool adapter | `StructuredTool.from_function(parse_docstring=True)`, ToolErrorMiddleware as the one error path |
| 3 | How tools are listed | Per-test tool sets as plain lists in code |
| 4 | Memory id kind | Plain database ids |
| 5 to 8 | Decided earlier | Own WalkieTodos; bare stop is a fixed line with no LLM; runtime fixes in the `create_agent` port; task templates by name only |
| 22 | Models | deepseek-v4.1-flash for runs |
| 23 | Where Mahoraga lives | The package `walkie_agent.mahoraga`, with its own pixi environment. It was a top-level `mahoraga/` until [decision T1](mahoraga-demo-decisions.md#tools-and-scripts). |
| 26 | Arena data | A trash location and more objects in the fake arena |

The cutover settled a few forks of its own, also provisionally:

| Fork | Pick |
| --- | --- |
| Vision | `objects` and `people` raise `UNAVAILABLE`. The remembered-object tool is retired, and a remembered row is never presented as a current view. S1 later let `people` read the live view with `--vision`. |
| HRI input | Text only. A message with an image or audio payload is logged and refused. |
| Shutdown that cannot settle | Keep the executor alive and input closed, retry within bounds, and report what is unresolved. A second signal forces the exit. |
| Larger proposals | Mark what is obsolete and record exact partial supersessions. Withdraw none of them. |
| Place names | Trimmed lowercase keys. Other forms and duplicates are refused at load, and lookup ignores case. `config/locations.toml` stays empty. |
| `say` | Returns once published. No playback acknowledgement is claimed. |
| RAI | The connector, `rai-core`, `rai-s2s` and both voice entry points stay. The agent host leaves `main` only. |
| Identity | Static `robot.md` replaces the whoami tree. |

Still the team's: the fallback model (none is selected), the walkie-database pin in
`ros2_ws.repos` (unchanged) and the real places (none invented).

Out of scope: CI and ruff jobs, the voice environment split, leaving RAI
(decision 9), MuJoCo validation (Phase R3) and promotion (Phase R4).

## Where this differs from the plan

- `ctx.arms` has the GPSR verbs, not the plan's `move_to` and `set_gripper`,
  which come with the commander client. Pick and place then become bundles.
- `ctx.base.pose()` replaces `ctx.pose`, and `ctx.base.places()` lets
  `list_places` replace the old `list_locations`.
- `FakeRobot` is one class, so its `ctx` holds the same object five times.
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
  are checked from the robot's call records. A `talk` line is checked for the fact
  its topic asks for, after a read of that fact, except "something about yourself". [Checkers](#checkers)
  lists the rules and the known limits.
- The world builder renames the operator and takes them out of every pose and
  gesture, so no command's selector can match them.
- M9 counts refused calls, since an escape would not show in a log. The bwrap
  tests are what stand behind it.
- There is no secondary template-chain set, no frozen paraphrases and no
  pre-registration file. `run_script` is in the runtime behind `--scripts`
  ([decision T20](mahoraga-demo-decisions.md#tools-and-scripts)), where the operator's
  verdict stands in for the scorer.
- The plan's target for what replaces `agent.py`, `graph.py`, `todo.py`, `tasks.py`
  and `tools/todos.py` is about 575 own lines, against 1,111 before. It measures 1,609:
  1,447 in `runtime/` and 162 in `tasks.py`, since moved to `runtime/task_templates.py`.
  The decided fixes, the Runner's lifecycle and the op-based todo list, 625 lines against the plan's 175, cost more than the estimate.
  The ROS wrappers add 1,063 lines, 638 of them in `ros/base.py` because a goal can outlive
  its verb. `main.py` adds 499.
- The stop's list update is a graph pass with a `stopping` input key rather than
  `aupdate_state`, so it keeps the same order as a task in the Runner's queue.
- The stop line is spoken at once rather than after the active run, which may be a
  hung LLM call. Each `stop()` speaks once, so the speech verb must tolerate it
  overlapping another `say`.
