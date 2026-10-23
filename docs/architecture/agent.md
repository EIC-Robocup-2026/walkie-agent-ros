# The Agent

Walkie's agent is one ROS 2 node, `walkie_agent`. It receives tasks from
people, asks an LLM what to do, and acts through a small set of tools. The
robot's own software, including navigation and manipulation, runs in
[walkie-ros](https://github.com/EIC-Robocup-2026/walkie-ros). The agent calls
it over ROS 2.

!!! info "Redesign in progress"
    The agent is being redesigned into a self-evolving agent: see
    [Self-Evolving Agent (Proposed)](self-evolving-agent.md). Phase 1's host
    graph and todo list are built. The rest of the redesign is proposed.
    This page describes the agent as it is today. For a guided tour with
    runnable examples, start at [Learn the Agent](../learn/index.md).

## Summary

- **Built on RAI.** `WalkieAgent` runs its own small graph on RAI's
  `LangChainAgent`. It keeps RAI's connectors, `HRIMessage`, `ToolRunner`
  and tracing, binds the tools on every LLM call, and adds a reset
  ([ADR-0003](decisions/0003-rai-react-agent.md),
  [ADR-0038](decisions/0038-own-graph-per-turn-binding.md),
  [ADR-0008](decisions/0008-reset-topic.md)).
- **A todo list for every task.** For a task with several steps, the LLM
  writes the steps with `update_todos` and marks them as it goes. The rules
  are enforced in code: for example, a step whose tool failed cannot be
  marked completed until that tool succeeds, unless a note explains why.
  A task can also arrive with its steps already seeded, from the operator
  on `/walkie/task` or from a template that an utterance names
  ([ADR-0041](decisions/0041-todo-list-and-seeding.md)).
- **Tasks in, speech and console out.** Tasks arrive on `/from_human`, typed
  or from speech recognition, or on `/walkie/task` with their steps. The
  agent speaks only through the `speak` tool on `/to_human`. Its plain-text
  replies, and one line per change of a step's status, go to the operator
  on `/walkie/console` ([ADR-0007](decisions/0007-speak-tool-and-console.md)).
  The whole todo list is published on `/walkie/todos`.
- **Narrow tools.** Each tool does one named thing, such as "drive to the
  kitchen". There is no raw ROS access, and a forbidden list keeps generic
  tools away from motion topics
  ([ADR-0009](decisions/0009-forbidden-names-guard-rail.md),
  [ADR-0010](decisions/0010-bounded-navigation-tool.md)).
- **Knows what it is.** rai_whoami adds a description of Walkie to the system
  prompt and a search tool over its documents
  ([ADR-0011](decisions/0011-whoami-identity.md)).
- **Agent-side only.** This repository holds the agent, its configuration and
  fakes for testing. Robot nodes live in walkie-ros
  ([ADR-0015](decisions/0015-repository-boundary.md),
  [ADR-0016](decisions/0016-talk-to-walkie-ros-directly.md)). The vision
  models will run in a new robot-side repository, walkie-vision
  ([ADR-0034](decisions/0034-vision-models-in-walkie-vision.md)).
- **Reads Walkie's memory.** `find_objects` asks the walkie-database node
  where objects of one kind were last seen. The agent reads the memory only
  through the node's ROS 2 services
  ([ADR-0035](decisions/0035-memory-through-walkie-database.md)).
- **On the robot PC.** The agent will run next to walkie-ros and reach it
  over rmw_zenoh on the same computer
  ([ADR-0017](decisions/0017-rmw-zenoh-domain-23.md)). Every Pixi command
  already uses rmw_zenoh
  ([ADR-0033](decisions/0033-zenoh-in-every-pixi-environment.md)).

## Components

```mermaid
flowchart TB
  person([Person])
  operator([Operator])

  subgraph agent_repo["walkie-agent-ros"]
    voice["voice-asr and voice-tts<br/>speech in and out"]
    agent["walkie_agent<br/>WalkieAgent and its tools"]
    whoami[("whoami/generated<br/>description + index")]
  end

  llm["LLM<br/>via OpenRouter"]
  langfuse["Langfuse<br/>local Docker, optional"]

  subgraph robot["walkie-ros"]
    nav2["Nav2"]
    tf["TF"]
  end

  subgraph memory["walkie-database"]
    db[("walkie_database<br/>Walkie's memory")]
  end

  person -- speech --> voice
  voice -- speaker --> person
  voice <-- "in: /from_human<br/>out: /to_human (speak tool)" --> agent
  operator <-- "in: /from_human, /walkie/task, /walkie/reset<br/>out: /walkie/console, /walkie/todos" --> agent
  whoami --> agent
  agent <--> llm
  agent -.-> langfuse
  agent -- "navigate_to_pose" --> nav2
  agent -- "get_current_pose" --> tf
  agent -- "find_objects" --> db
```

The voice box is two nodes, speech recognition and text-to-speech. Speech
recognition ignores the microphone while Walkie speaks, so Walkie does not
hear itself ([ADR-0029](decisions/0029-half-duplex-voice.md)). See
[Voice](../setup/07-voice.md) for how they connect.

`pixi run fake-walkie` stands in for the robot during development: it
serves the Nav2 action and TF, and listens on the speech topic. It has no
memory. `find_objects` needs the walkie-database node, see
[Getting Started](../getting-started.md#ask-walkies-memory).

## How a task runs

```mermaid
sequenceDiagram
  participant P as Person
  participant A as walkie_agent
  participant L as LLM
  participant T as voice-tts
  participant N as Nav2

  P->>A: "Go to the kitchen" (/from_human)
  A->>L: system prompt + conversation + task,<br/>plus the todo list if it has items
  L-->>A: call speak("On my way") and navigate_to_location("kitchen")
  A->>T: "On my way" (/to_human)
  A->>N: NavigateToPose goal (every wait bounded)
  N-->>A: SUCCEEDED
  A->>L: tool results
  L-->>A: final reply
  A-->>P: reply on /walkie/console (operator only)
```

Messages wait in a queue. When the agent is idle, it takes the oldest one
and runs it as a task until the LLM stops calling tools. An utterance that
arrives during a task waits until the task ends. RAI's default would
interrupt the task instead, and at a noisy booth any overheard speech would
then cancel the current step
([ADR-0038](decisions/0038-own-graph-per-turn-binding.md)). Every task
joins the same conversation, so the LLM also sees the earlier tasks until a
reset.

### The agent's graph

`build_walkie_graph()` in `src/walkie_agent/graph.py` builds the graph.
Its nodes change RAI's message list in place, so a reset and the tests
work on the same history as the graph.

```mermaid
flowchart TD
  IN["Next message from the queue: an utterance,<br/>/walkie/task or submit(text, todos)"] --> LLM["llm node<br/>repairs the history, binds the tools,<br/>adds the todo reminder"]
  LLM -->|"tool calls"| TR["tools node: WalkieToolRunner<br/>runs the calls one at a time"]
  TR -->|"results"| LLM
  TR -->|"wait_for_person asked its question"| DONE(["Task ends"])
  LLM -->|"only calls with unreadable arguments"| LLM
  LLM -->|"no tool call"| GATE{"Todo items open,<br/>none waiting?"}
  GATE -->|"yes: completion gate,<br/>at most twice per task"| LLM
  GATE -->|"no, or already twice"| DONE
```

- **llm node.** It puts the system prompt first and repairs the history.
  Then it binds the tools in `agent.tools` and calls the LLM. The tools are
  bound on every call, so a tool added during a task is callable on the
  next turn. Phase 1 binds every tool. Above 20 tools it logs a warning,
  once per count, because a small local model copes poorly with that many.
- **History repair.** OpenAI-compatible endpoints reject an AI message
  whose tool calls have no results. That happens when a stop lands after
  the LLM asked for tools but before they ran, or when an error ends a
  turn halfway. Before every LLM call, each such call gets an
  error result, "Interrupted before it ran.", right after its AI message.
  A call whose arguments are not valid JSON never runs either. LangChain
  keeps it apart from the other calls, but still sends it back to the
  endpoint, so it gets the error "The arguments were not a valid JSON
  object, so the call did not run." When the LLM made no other call, the
  graph goes straight back to the llm node, so the LLM can try again.
- **tools node.** `WalkieToolRunner` is a subclass of RAI's `ToolRunner`,
  so the calls still run one at a time with RAI's error handling. It
  reports whether each call succeeded to the todo list's error ledger. A
  call to a tool that does not exist, such as `fly`, gets the error
  "Unknown tool fly." and the task goes on. Only the first `update_todos`
  call in one LLM answer is applied. Any later one gets an error that asks
  for all the changes in one call.
- **Completion gate.** The LLM may answer without a tool call while todo
  items are still open and none is waiting. The graph then saves this
  message in the history and calls the LLM again: "[todo] Open items: t3,
  t4. Finish them, or mark them failed or cancelled with a note." It does
  this at most twice per task.
- **`wait_for_person` ends the task.** It speaks the question and marks the
  step in progress as waiting. The task then ends, with no completion gate.
  The person's answer arrives as the next message and puts the step back in
  progress. Any tool calls after the question in the same LLM reply do not
  run; each is answered "Not run" so the LLM can repeat it after the answer.
  If the question could not be spoken, the call fails and the task goes on,
  so the LLM can try something else.
- **At most 100 graph steps per task,** about 50 rounds of tool calls.
  LangGraph's default of 25 is too few for a seeded task with several
  steps. Past the limit, the task stops with a logged error.

### The todo list in each turn

When the list has items, each LLM call gets it as one extra human message
at the end. That message is not saved in the history, so the history holds
only what happened. It is a human message, not a system message, because
some providers reject a system message that is not first.

```text title="Todo reminder"
<todos>
[x] t1 Go to the kitchen
[>] t2 Pick up the red mug
[ ] t3 Come back to the person
</todos>
```

`[ ]` is pending, `[>]` in progress, `[?]` waiting for an answer, `[x]`
completed, `[!]` failed and `[-]` cancelled.

When the next task starts, a list with no open items is cleared, so a
finished list does not carry over. A list with open items is kept, for
example while the agent waits for an answer.

`WalkieAgent.submit(text, todos)` queues a task with its steps already
written. The seed replaces the list when that task starts, before the first
LLM call. Seeded steps cannot be reworded or removed. A bad seed raises
`TodoRejected` from `submit()` at once, not later on the agent thread.
The operator can seed a task over ROS on `/walkie/task`, and an utterance
can start a template
([ADR-0041](decisions/0041-todo-list-and-seeding.md)).

### Seeding a task on /walkie/task

`/walkie/task` takes a `std_msgs/msg/String` that holds a JSON object in
one of two forms:

```json title="With its steps"
{"task": "Bring me a cup from the kitchen.",
 "todos": ["Go to the kitchen",
           {"content": "Pick up a cup", "active_form": "Picking up a cup"},
           "Come back"]}
```

```json title="With a template"
{"task": "Two guests are on their way.", "template": "hri"}
```

- **`todos`** is the same list that `submit()` takes. A step is a string,
  or an object with `content` and, optionally, `active_form` and
  `requires_tool`.
- **`template`** names a template in `config/task_templates.toml`. The LLM
  gets the message's task, a blank line and the template's task text. The
  template's steps are seeded.
- **Seeded steps from this topic have the source `operator`.** The task
  joins the same queue as an utterance.
- **A malformed message is logged as a warning and ignored.** Examples are
  text that is not JSON, an unknown key, both or neither of `todos` and
  `template`, an unknown template or a bad step. The topic keeps working.

### Task templates

`config/task_templates.toml` holds named templates, one
`[templates.<name>]` table each. The shipped file has only a commented
example. `--task-templates` points the agent at another file.

```toml title="config/task_templates.toml"
[templates.tour]
keywords = ["house tour", "show me around"]
task = """
Show the person the main rooms. In each room, say what it is for.
"""
todos = [
  "Go to the kitchen",
  { content = "Say what the kitchen is for", requires_tool = "speak" },
  "Come back to the person",
]
```

- **`task` and `todos` are required.** `todos` uses the seed format above.
- **`keywords` is optional.** A template without keywords can only be
  started on `/walkie/task`.
- **Matching happens when a task starts.** A task without a seed is matched
  only when no step is open. Its text is checked against each template's
  keywords, in file order, and the first match wins. A keyword matches as
  whole words in any case, so "house tour" matches "Give me a House Tour!"
  but not "a lighthouse tour".
- **A match seeds the template.** The LLM gets the utterance, a blank line
  and the template's task text. The steps get the source `seed`.
- **Why at task start.** A task that is still queued must not start a
  template while an earlier task may still open or finish steps. While a
  step is open, for example while the agent waits for an answer, an answer
  that contains a keyword is just an answer.
- **The file is checked at startup.** A missing file means no templates. A
  bad template, such as an unknown key, no task text or a bad step, stops
  the agent with an error that names the file and the template.

### Following the list

- **`/walkie/todos`** carries the whole list as a JSON array of steps, in a
  `std_msgs/msg/String`, after every change. Each step has all its fields,
  such as `id`, `content`, `status`, `note` and `source`. It is reliable,
  transient-local and keeps one message, so a viewer that starts late
  still gets the current list. The agent publishes `[]` at startup and
  after a reset.
- **`/walkie/console`** gets one line for each change of a step's status.
  A step that goes in progress shows its active form, or its content when
  it has none: `t1: Going to the kitchen`. A step that is completed,
  failed, cancelled or waiting shows its content:
  `t1 completed: Go to the kitchen`. A new pending step shows no line.
  These lines are never spoken.

### Stop, reset and errors

- **Stop.** An utterance that is just "stop", "never mind" or "nevermind"
  interrupts a running task. Case, extra spaces and `.`, `,`, `!` or `?`
  around it are ignored, so "Stop." counts, but "Walkie, stop" and "stop
  the music" do not. Messages queued before it are dropped, with their
  seeds. The stop message itself runs next, so the LLM can cancel the open
  steps and say so. While the agent is idle, "stop" is an ordinary message.
- **Reset.** A message on `/walkie/reset` drops the queued messages and
  their seeds, and stops the running task. Before the next task, the agent
  clears the history and the todo list, publishes `[]` on `/walkie/todos`
  and starts a new Langfuse session
  ([ADR-0008](decisions/0008-reset-topic.md)).
- **Both act between steps.** All tool calls from one LLM answer run inside
  one step of the graph, and a stop or a reset is noticed only between
  steps. A tool that is already running finishes first. If the stop lands
  after the LLM asked for tools, those tools never run, and the history
  repair answers them.
- **Errors.** An error from the LLM or the graph is logged, and the agent
  goes on to the next message. In RAI's own loop, such an error ends the
  agent thread. If the error left tool calls without results, the history
  repair answers them on the next call.

## Tools

| Tool                              | From       | What it does                                                    |
| --------------------------------- | ---------- | --------------------------------------------------------------- |
| `speak`                           | this repo  | Says a sentence out loud through `/to_human`                    |
| `list_locations`                  | this repo  | Lists the named places in `config/locations.toml`               |
| `navigate_to_location`            | this repo  | Drives to a named place with Nav2, with every wait bounded      |
| `move_relative`                   | this repo  | Moves up to 3 m forward, back or sideways from here, then turns |
| `find_objects`                    | this repo  | Asks walkie-database where objects of one kind were last seen   |
| `update_todos`                    | this repo  | Adds, updates and removes steps on the task's todo list, by id  |
| `wait_for_person`                 | this repo  | Asks the person a question out loud and ends the task until they answer |
| `get_current_pose`                | RAI        | Reads the robot's position in the `map` frame from TF           |
| `get_ros2_topics_names_and_types` | RAI        | Lists ROS topics, minus the forbidden ones                      |
| `WaitForSecondsTool`              | RAI        | Waits a number of seconds                                       |
| `search_robot_docs`               | rai_whoami | Searches Walkie's documentation (only after `pixi run whoami-build`) |

`find_objects` calls `/walkie_database/find_objects` with a label and asks
for at most 5 objects. It returns each object's room and position on the
map, most recently seen first. The database node's `map_id` parameter
decides which map is searched. The tool waits up to 2 s for the node to
appear and reports memory as unavailable after that. It then waits up to
5 s for the answer. A node that does not answer in time is reported to the
LLM as busy, and the agent's log says that the node may have been built
from a different `walkie_database_msgs` than the pinned commit
([ADR-0035](decisions/0035-memory-through-walkie-database.md)).

When `navigate_to_location`, `move_relative`, `speak` or `find_objects`
fails, the LLM reads a sentence saying why, and the result is marked as an
error. The todo list counts only successful calls as evidence that a step
was done ([ADR-0041](decisions/0041-todo-list-and-seeding.md)). Finding no
remembered object is a successful search.

`update_todos` applies all its changes or none. A refused change, such as
completing a step whose tool failed, comes back to the LLM as an error that
says why. `wait_for_person` speaks through the same `speak` tool. When no
speech node is listening, it fails and the task goes on.

The [arm manipulation design](manipulation.md) proposes four arm tools and,
later, two skill tools.

For the full list of topics, see
[Getting Started](../getting-started.md).

## Running on the robot

!!! note "Proposed"
    This part is not built yet. Today the agent drives the fake robot, or
    walkie-ros's MuJoCo simulation with `pixi run agent-sim`
    ([ADR-0030](decisions/0030-mujoco-sim-first.md)). Both run over
    rmw_zenoh
    ([ADR-0033](decisions/0033-zenoh-in-every-pixi-environment.md)).

The agent runs on the robot PC, next to walkie-ros. walkie-ros uses Zenoh as
its ROS 2 middleware (`rmw_zenoh_cpp`) on domain 23. The agent must use the
same middleware, the same domain and the same message definitions, or it
silently sees nothing. A service client built from different definitions
still finds the server, but its requests are never answered
([ADR-0017](decisions/0017-rmw-zenoh-domain-23.md),
[ADR-0018](decisions/0018-pin-robot-interfaces.md),
[ADR-0035](decisions/0035-memory-through-walkie-database.md)).

```mermaid
flowchart LR
  subgraph pc["Robot PC"]
    a["walkie_agent<br/>+ voice nodes<br/>(Pixi)"]
    r["Zenoh router<br/>rmw_zenohd, port 7447"]
    w["walkie-ros nodes"]
  end
  llm["LLM<br/>via OpenRouter"]
  a -- localhost --- r
  r --- w
  a -- internet --> llm
```

Only the LLM calls leave the robot PC. The tests stay on Fast DDS, with
fakes, so they need neither a router nor a robot, and running them on the
robot PC does not touch the live robot
([ADR-0019](decisions/0019-test-fakes-fast-dds.md)).

## What lives where

| Piece                                                        | Repository                 |
| ------------------------------------------------------------ | -------------------------- |
| Agent, tools, system prompt, forbidden names                 | walkie-agent-ros           |
| Named locations, task templates, named arm poses             | walkie-agent-ros           |
| Voice nodes, whoami sources, Langfuse setup                  | walkie-agent-ros           |
| Fake robot nodes for tests                                   | walkie-agent-ros           |
| Drivers, ros2_control, Nav2, MoveIt, `bimanual_commander`    | walkie-ros                 |
| URDF, SRDF, joint limits                                     | walkie-ros                 |
| `my_robot_interfaces` and future skill interfaces            | walkie-ros, pinned here    |
| `walkie_perception`: grasp pipeline, classic 3D positions    | walkie-ros                 |
| Vision: detection, embeddings, pose, faces, re-ID, captions  | walkie-vision              |
| `walkie_vision_msgs`                                         | walkie-vision, to be pinned here |
| Long-term memory: maps, rooms, objects, people, conversations | walkie-database           |
| `walkie_database_msgs`                                       | walkie-database, pinned here |
| Skill server, policy inference                               | walkie-ros (or another robot-side repository) |

The agent does not load the URDF. MoveIt uses it on the robot, and rai_whoami
ignores it ([ADR-0020](decisions/0020-no-urdf-in-agent.md)).

## Decisions

Click a decision to expand it. Green blocks are accepted, blue blocks are
proposed. See [Architecture](index.md) for how to add one.

??? success "ADR-0001: Record architecture decisions · Accepted"

    --8<-- "architecture/decisions/0001-record-architecture-decisions.md:body"

??? success "ADR-0002: Install ROS 2 and Python packages with Pixi and RoboStack · Accepted"

    --8<-- "architecture/decisions/0002-pixi-and-robostack.md:body"

??? success "ADR-0003: Build the agent on RAI's ReActAgent · Accepted, partly superseded by ADR-0038"

    --8<-- "architecture/decisions/0003-rai-react-agent.md:body"

??? success "ADR-0004: Build rai_interfaces from source in ros2_ws · Accepted"

    --8<-- "architecture/decisions/0004-build-rai-interfaces-from-source.md:body"

??? success "ADR-0005: Reach LLMs through OpenRouter with RAI's openai vendor · Accepted"

    --8<-- "architecture/decisions/0005-openrouter-via-openai-vendor.md:body"

??? success "ADR-0006: Trace agent runs with self-hosted Langfuse, off by default · Accepted"

    --8<-- "architecture/decisions/0006-self-hosted-langfuse.md:body"

??? success "ADR-0007: Speak only through the speak tool; plain replies go to the console · Accepted"

    --8<-- "architecture/decisions/0007-speak-tool-and-console.md:body"

??? success "ADR-0008: Reset the conversation with a message on /walkie/reset · Accepted"

    --8<-- "architecture/decisions/0008-reset-topic.md:body"

??? success "ADR-0009: Forbidden ROS names are a guard rail, not a security boundary · Accepted"

    --8<-- "architecture/decisions/0009-forbidden-names-guard-rail.md:body"

??? success "ADR-0010: Navigate to named places with every wait bounded · Accepted"

    --8<-- "architecture/decisions/0010-bounded-navigation-tool.md:body"

??? success "ADR-0011: Give the agent Walkie's identity with rai_whoami · Accepted"

    --8<-- "architecture/decisions/0011-whoami-identity.md:body"

??? success "ADR-0012: Install rai_s2s and rai_whoami from RAI's Git tag 2.12.2 · Accepted"

    --8<-- "architecture/decisions/0012-rai-git-tag-2-12-2.md:body"

??? success "ADR-0013: Use the CUDA 12.1 PyPI builds of torch · Accepted"

    --8<-- "architecture/decisions/0013-cuda-torch.md:body"

??? success "ADR-0014: Document the project with MkDocs Material · Accepted"

    --8<-- "architecture/decisions/0014-mkdocs-material.md:body"

??? success "ADR-0015: Keep only agent-side code in this repository · Accepted, partly superseded by ADR-0034"

    --8<-- "architecture/decisions/0015-repository-boundary.md:body"

??? success "ADR-0016: Call walkie-ros directly and retire walkie-sdk · Accepted"

    --8<-- "architecture/decisions/0016-talk-to-walkie-ros-directly.md:body"

??? note "ADR-0017: Run the agent on the robot PC over rmw_zenoh_cpp on domain 23 · Proposed"

    --8<-- "architecture/decisions/0017-rmw-zenoh-domain-23.md:body"

??? note "ADR-0018: Pin walkie-ros message packages to the robot's deployed commit · Proposed"

    --8<-- "architecture/decisions/0018-pin-robot-interfaces.md:body"

??? note "ADR-0019: Test against in-repo fakes on Fast DDS · Proposed"

    --8<-- "architecture/decisions/0019-test-fakes-fast-dds.md:body"

??? note "ADR-0020: The agent does not load the robot's URDF · Proposed"

    --8<-- "architecture/decisions/0020-no-urdf-in-agent.md:body"

The arm manipulation decisions, ADR-0021 to ADR-0028, are on the
[Arm Manipulation](manipulation.md#decisions) page.

??? success "ADR-0029: Stop listening while Walkie speaks · Accepted"

    --8<-- "architecture/decisions/0029-half-duplex-voice.md:body"

??? success "ADR-0030: Drive walkie-ros's MuJoCo simulation before the real robot · Accepted, partly superseded by ADR-0033 and ADR-0036"

    --8<-- "architecture/decisions/0030-mujoco-sim-first.md:body"

??? success "ADR-0031: Move short distances relative to the robot, bounded in length · Accepted"

    --8<-- "architecture/decisions/0031-bounded-relative-moves.md:body"

??? success "ADR-0032: Build rmw_zenoh_cpp 0.2.11 from source and preload it · Accepted, partly superseded by ADR-0033"

    --8<-- "architecture/decisions/0032-rmw-zenoh-from-source.md:body"

??? success "ADR-0033: Use rmw_zenoh_cpp in every Pixi environment · Accepted"

    --8<-- "architecture/decisions/0033-zenoh-in-every-pixi-environment.md:body"

??? success "ADR-0034: Run the vision models in a new repository, walkie-vision · Accepted"

    --8<-- "architecture/decisions/0034-vision-models-in-walkie-vision.md:body"

??? success "ADR-0035: Read Walkie's memory through walkie-database's ROS 2 services · Accepted"

    --8<-- "architecture/decisions/0035-memory-through-walkie-database.md:body"

??? success "ADR-0036: Run agent-sim on the calling shell's ROS_DOMAIN_ID · Accepted"

    --8<-- "architecture/decisions/0036-agent-sim-shell-domain.md:body"
