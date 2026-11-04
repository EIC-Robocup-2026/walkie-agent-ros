# Mahoraga Demo Prototype

A prototype, on branch `feat/mahoraga-demo`, of the layered agent that the
Mahoraga agent plan proposes. Its ROS-free foundation is shared by the
competition agent and Mahoraga, so the same atomics run on the robot, in unit
tests and in the fake-world gym. It is built beside main's agent, whose
`agent.py`, `graph.py`, `todo.py`, `tasks.py` and `tools/` stay unchanged
until a later slice replaces them. This page is the interface contract that
those later slices follow.

!!! warning "Prototype"
    Nothing here has run against a real model, in MuJoCo or on the robot,
    and nothing on the robot uses it yet. Every choice below is a
    provisional pick of the plan's recommended option.

## Layer map

Each layer calls only the layer below it. Mahoraga reaches layer 3 only.

| Layer | Module | Slice |
| --- | --- | --- |
| 2. Capability verbs, guards, `Ctx` | `walkie_agent/capabilities/` | This one |
| 2. FakeRobot, Mahoraga's FakeCtx | `walkie_agent/capabilities/fake.py` | This one |
| 3. Error type | `walkie_agent/errors.py` | This one |
| 3. Atomics and tool sets | `walkie_agent/atomics/` | This one |
| 4. Adapter to LangChain tools | `walkie_agent/adapter.py` | This one |
| 5. `create_agent`, middlewares, Runner | `walkie_agent/runtime/` | Later |
| 1 to 2. rclpy glue, ROS-backed verbs | `walkie_agent/ros/` | Later |
| 6. `run_script`, shelf, gym | `mahoraga/`, with its own pixi environment | Later |

`errors`, `capabilities`, `atomics` and `adapter` never import rclpy, rai,
rai_interfaces or any ROS message. `tests/test_ros_free.py` checks this in a
fresh interpreter.

## Errors

```python
class ErrorCode(StrEnum):  # walkie_agent.errors
    UNAVAILABLE, TIMEOUT, NOT_FOUND, REFUSED, PRECONDITION_FAILED, SKILL_FAILED, CANCELLED, INTERNAL

class StepError(Exception):
    def __init__(self, code: ErrorCode | str, message: str) -> None: ...  # ValueError for an unknown code
    code: ErrorCode
    message: str  # one plain sentence for the LLM; str(error) is "CODE: message"
```

Verbs and atomics raise only `StepError`. Anything else that escapes a tool
is a bug, and reaches the model as `INTERNAL`.

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

@dataclass(frozen=True)
class Ctx:
    cancel: CancelToken
    base: Base
    speech: Speech
    vision: Vision
    arms: Arms = UnavailableArms()  # raises UNAVAILABLE until the arm commander client exists
```

A query of `""` matches everything. Otherwise it matches an object's name,
category or plural category, or a person's name, pose or gesture, ignoring
case. The arm verbs are fake-only in this prototype.

### Guards live in the verbs

`walkie_agent.capabilities.guards` holds the shared checks. Every
implementation of a verb, fake or ROS-backed, calls them.

```python
MAX_MOVE_M = 3.0  # main's navigation.py cap
def require_finite(**values: float) -> None    # REFUSED for NaN or infinity
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
```

- There is no velocity verb. The base moves only to a named place or by a
  bounded relative move.
- Every wait is bounded and goes through `CancelToken.wait_for`, so a stop
  ends it within `poll_s`.
- Base, vision and arm verbs call `ctx.cancel.check` before they start, so a
  set token latches motion until the runtime resets it.
- `Speech.say` does not check the token, so a bare "stop" can answer with
  its fixed line while motion stays latched (decision 6).
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
    faults: dict[str, StepError]  # verb name -> the error every call to it raises
    def snapshot(self) -> dict[str, Any]: ...  # JSON-able copy for checkers: at, room, pose,
                                               # holding, places, objects, people, said, calls

def chula_pat14_6(seed: int = 0) -> dict[str, Any]  # a seeded world description
```

A world description is JSON-able, so the gym can build, store and compare
it:

```python
{"places": {"shelf": {"x": 2.487, "y": -6.409, "yaw": -1.57, "room": "kitchen room"}},
 "objects": {"apple_1": {"name": "apple", "category": "fruit", "size_cm": 8, "weight_g": 180,
                         "thickness_cm": 8, "at": "shelf"}},
 "people": {"operator": {"name": "Jane", "room": "living room", "pose": "standing person",
                         "gesture": "waving person"}},
 "start": "instruction point"}
```

A place whose `room` is its own name is a room. An object's `at` is a place,
`"robot"` while held, or the id of the person who received it. The robot
sees what is at its place, or at every place in its room while it stands at
the room itself, and the people in its room. `world=None` builds
`chula_pat14_6(seed)`, and `drive_s` makes each drive take real seconds.
Places and rooms come from walkie-ros `origin/feat/chula-pat14-6-simulation`
(`simulation/mujoco/worlds/chula_pat14_6/authoring/source/30_16102026_ChulaPat14_6_world.toml`).
The `trash` and `instruction point` places, six extra objects, all sizes and
weights, and the people are invented (decision 26). A builder that seeds a
feasible world for one generated command belongs to the gym.

## Atomics

Plain typed functions over `ctx` in `walkie_agent.atomics`. The LLM reads
their Google-style docstrings. They return JSON-able values and raise
`StepError`. Describe the return value in the summary paragraphs: LangChain
drops a `Returns:` section from the tool description.

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

`bind` returns a plain callable with `ctx` bound, for the Monty driver. It
keeps the name, the docstring and the signature without `ctx`. `as_tool`
passes it to `StructuredTool.from_function(parse_docstring=True,
error_on_invalid_docstring=True)`, refuses a parameter with no description,
and sends a non-string result as JSON text, so an empty list reads `[]`
rather than an empty message. `as_tools` refuses duplicate names.

`ToolErrorMiddleware(on_error=None, *, aon_error=None, tools=None)` ships in
langchain 1.4.4 under `langchain.agents.middleware`. `on_error(exc,
request)` returns an error `ToolMessage`'s content, or `None` to re-raise.
`ToolNode` answers argument validation errors before it. `format_tool_error`
gives `"CODE: message"` for a `StepError` and `"INTERNAL: <tool> failed with
<ExceptionType>."` for anything else, which keeps internal detail out.

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
