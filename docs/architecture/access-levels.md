# Access Levels and Generated Tools (Proposed)

!!! note "Status: Proposed"
    Nothing on this page is built yet. It is the design for team review. Each
    of the [open questions](#open-questions) says which phase it blocks.

The [self-evolving agent](self-evolving-agent.md) runs tools it wrote
itself: bundles (checked data) and generated Python in the `temporary` tier. This
page says what that code may do at each level, how the agent proves it is in
a simulation, and how the code is kept from the robot, the network and the
agent's secrets. Effects follow
[ADR-0037](decisions/0037-tool-tiers-bundles-effects.md): **R** reads, **P**
perceives or computes, **S** speaks, **N** moves the base, **M** moves an
arm or gripper. The head and lift controllers are on the never-list with the
arm's. Phases refer to the [roadmap](self-evolving-agent.md#roadmap).

## Summary

- Three nested levels: Unrestricted ⊃ Standard ⊃ Competition, each built by
  narrowing the one above. Only Competition, the default, may run on the
  real robot ([ADR-0039](decisions/0039-access-levels.md)).
- The level comes only from the `--access-level` flag. A sim interlock grants
  Standard or Unrestricted only on a marked sim host with a live MuJoCo
  graph. A failed check demotes the session until the agent restarts
  ([ADR-0040](decisions/0040-sim-interlock.md)).
- Generated code runs in bubblewrap with no network, no secrets and no ROS.
  Bridges in the parent check, trace and clean up every call
  ([ADR-0047](decisions/0047-sandbox-and-bridges.md)).
- Each tool is stamped with the lowest level its calls need, and runs at
  `min(session level, stamp)`. Arms move only through atomics
  ([ADR-0044](decisions/0044-mocked-arm-atomics.md)).
- At Competition a tool may sense private local data or reach the web, never
  both: the Rule of Two, linker check L14.

## The three levels

| Level | Where it may run | What it is for |
| ----- | ---------------- | -------------- |
| Unrestricted | Simulation only, on a marked sim host | Find what is missing, including new atomics and perception ideas. Raw ROS reads, service calls and actions. Raw base-command publish may come in phase 4b, only if harvest traces show tools blocked without it. |
| Standard | Simulation only, on a marked sim host | Find tools that can be built from existing atomics, with code. |
| Competition | Anywhere, including the real robot. The default. | Answer a live stranger's request safely. |

Two sim levels exist because they answer different questions. A Standard tool
shows what code can do with today's atomics. An Unrestricted tool that needed
raw ROS shows which atomic a human should add.

"Unrestricted" describes the ROS and web policy, not the sandbox. It still
forbids secrets, its own sockets, never-list names, MoveIt 2, the raw arm,
gripper, head and lift controllers, files outside its tmpfs, package installs
and running outside the sandbox.

### Capabilities by tier and level

"Gated" means every utterance and sound passes the
[speech gate](competition.md#speech-gate).

| Tier or level | R, P | S | N | M | Raw ROS writes | Internet | On the robot | Gate before first use |
| ------------- | ---- | - | - | - | -------------- | -------- | ------------ | --------------------- |
| atomic, human-written | yes | gated | yes | yes | as declared | as declared | yes, in-process | review, fake-backed tests |
| atomic, promoted generated code | per manifest | gated | per manifest | per manifest | per manifest | per manifest | yes, in the sandbox | manifest review, fake-backed tests, CI provenance check |
| `bundled_preset` | yes | gated | yes | yes, mocked until real arms | none | none | yes | review, replay tests, 2 traced runs or harvest evidence |
| generated, Unrestricted (sim) | yes | gated | raw actions; base-command publish in phase 4b, if it ships | through atomics only, never MoveIt | base commands in phase 4b, if it ships, minus the never-list | broad, vetted; web search | no | lint, stamp, sandbox test run |
| generated, Standard (sim) | yes | gated | through atomics | through atomics | none | public hosts, vetted; web search | no | lint, stamp, sandbox test run |
| generated, Competition | R; P through atomics only | gated, capped | none | none | none | named sources; search snippets under L14 | yes, session only | lint, stamp, L14, L15, literal checks, test run against fakes, resource scope |

A promoted tool runs at every level, but only within its manifest. The level
caps apply to generated code.

## How nesting is enforced

`src/walkie_agent/toolbox/levels.py` (new) loads
`config/access_levels.toml` (new), builds Unrestricted from it, then derives
`STANDARD = UNRESTRICTED.narrow(standard_rules)` and
`COMPETITION = STANDARD.narrow(competition_rules)`. Every field declares how
it narrows:

| Field kind | Examples | When narrowing |
| ---------- | -------- | -------------- |
| `*_allow` sets | `effects_allow`, `tools_allow`, `ros.call_allow`, `http.sources_allow` | intersect |
| `*_deny` and `*_require` sets | `write_deny`, `handle_types_require`, `read_strip_require` | union |
| `require_*` flags | `require_rule_of_two`, `require_literal_hosts`, `require_scope` | OR |
| caps | `invoke_s`, `http_per_minute`, `search_k` | minimum over the union of keys; missing means unlimited |
| boolean switches | `ros.graph`, `ros.tf`, `tools_search` | AND (proposed; not yet in the outline below) |

- `LevelPolicy.narrow(rules)` applies only these operations, so it can only
  return a subset. `tool_allowed`, `ros_allowed` and `http_allowed` check the
  never-list first. `tests/test_levels.py` checks the subset relation with
  random inputs.
- `narrow()` raises on a key without a declared polarity, and a rule file
  cannot add an allow-list its parent lacks.
- Switches that grant access when true have no bucket in the outline yet.
  They must narrow with AND, or a lower level could switch one back on.

??? example "Policy file and `narrow()` outline (abbreviated)"

    Values are those in the tables on this page. Lists marked "and the rest"
    and most caps are cut short.

    ```toml
    # config/access_levels.toml (new)
    [never]
    any_deny = ["/controller_manager/*", "/go_to_pose*", "/*arm_controller/*", "/lift/cmd"]  # and the rest
    write_deny = ["/from_human", "/walkie/console", "/clock"]                                 # and the rest

    [unrestricted]
    effects_allow = ["R", "P", "S", "N", "M"]
    tools_allow = ["*"]
    tools_search = true
    ros = { read_allow = ["*"], call_allow = ["*"], publish_allow = ["*"], action_allow = ["*"], graph = true, tf = true }
    http = { methods_allow = ["GET", "HEAD", "POST"], schemes_allow = ["http", "https"], sources_allow = ["*"] }
    caps = { invoke_s = 300, coder_s = 900, search_k = 10, search_per_invocation = 10 }

    [standard]  # narrows unrestricted
    ros = { call_allow = ["*/get_parameters", "/walkie_database/find_objects"], publish_allow = [] }  # and the rest
    http = { schemes_allow = ["https"] }
    caps = { invoke_s = 60, coder_s = 600, search_k = 5, search_per_invocation = 5 }

    [competition]  # narrows standard
    effects_allow = ["R", "P", "S"]
    tools_allow = ["speak", "play_sound", "tell_time", "tell_date", "list_locations"]  # and the rest
    tools_search = false
    require_rule_of_two = true
    require_literal_hosts = true
    ros = { read_allow = [], call_allow = [], publish_allow = [], action_allow = [], graph = false, tf = false }
    http = { methods_allow = ["GET"], sources_allow = ["open_meteo", "wikipedia_summary", "web_snippets"] }  # and the rest
    caps = { invoke_s = 15, coder_s = 25, search_k = 3, search_per_turn = 2, search_per_invocation = 2 }
    ```

    ```python
    # src/walkie_agent/toolbox/levels.py (outline)
    def narrow(self, rules: Rules) -> "LevelPolicy":
        """Return a policy that allows a subset of this one. Never widens."""
        rules.check_known_keys()  # raises on any key without a declared polarity
        keys = self.caps.keys() | rules.caps.keys()
        return LevelPolicy(
            level=Level(self.level - 1),
            allow={k: self.allow[k] & rules.allow.get(k, self.allow[k]) for k in self.allow},
            require={k: self.require.get(k, frozenset()) | rules.require.get(k, frozenset())
                     for k in self.require.keys() | rules.require.keys()},
            flags={k: self.flags.get(k, False) or rules.flags.get(k, False)
                   for k in self.flags.keys() | rules.flags.keys()},
            caps={k: min(self.caps.get(k, INF), rules.caps.get(k, INF)) for k in keys},
        )
    ```

    The outline intersects literal sets. With a glob such as `"*"` in the
    parent, the intersection has to match patterns rather than strings,
    which the outline does not show yet.

## Policy per level

| Dimension | Unrestricted (sim) | Standard (sim) | Competition |
| --------- | ------------------ | -------------- | ----------- |
| Creation paths | reuse, compose, write code | same | reuse; compose and write code only when online and the run profile allows |
| Effects of generated tools | R P S N M, M only through atomics | R P S N M, N and M only through atomics | R P S |
| `ctx.tools` exposes | every atomic, including internal ones (`navigate_to_pose`, `run_skill`, `set_gripper`, `localize_object`, `get_transform`), plus read-only `ctx.tools.search` | same | `speak`, `play_sound`, `tell_time`, `tell_date`, `list_locations`, `get_current_pose`, `find_objects`, `detect_objects`, `search_detections`, `relate_objects`, `get_arm_state`, `wait_for_seconds` (≤ 10 s); `describe` only; perception returns labels, counts and enums |
| ROS read | any name off the never-list; images, point clouds, occupancy grids and messages over 256 KB come back as handles | any name off the never-list; nested images, point clouds and `float32[]` fields named `*embedding` stripped | none |
| ROS call | any service off the never-list | `*/get_parameters`, `*/list_parameters`, `*/describe_parameters`, `*/get_type_description`, `/walkie_database/find_objects` | none |
| ROS publish | none; phase 4b, only on harvest evidence: base-command topics with a known stop message (`Twist`, `TwistStamped`), 20 Hz | none | none |
| ROS action | any action off the never-list, cancelled on exit | `/compute_path_to_pose`, `/compute_path_through_poses` | none |
| ROS graph and TF | yes | yes | none; pose comes from `get_current_pose` |
| Web text | marked untrusted | same | marked untrusted; taints the invocation and the turn; returns as typed fields (numbers, enums, strings ≤ 120 characters) |
| Filesystem | read-only pixi env and facade, 64 MB tmpfs `/tmp`, tmpfs `$HOME`, assets as handles | same | same |
| Tiers that load | atomic, bundled_preset, drafts stamped ≤ Unrestricted | atomic, bundled_preset, drafts stamped ≤ Standard | atomic, bundled_preset, this session's creations; nothing from `bundled/`, `temporary/`, `quarantine/` or `quarantine_lane/` |
| Persisted | drafts with provenance and stamp, until the harvest trims them | same | session registration, and a copy in `quarantine/<date>/<name>@<sha8>/` kept 30 days, never auto-loaded |

`navigate_to_pose` is the internal atomic that
[ADR-0043](decisions/0043-navigate-to-pose-provenance.md) proposes. The
choice is still open on
[Self-Evolving Agent](self-evolving-agent.md#open-questions).

No level puts `update_todos`, `wait_for_person`, `find_tools`, `request_tool`
or `look_up_online` in `ctx.tools`: no tool may steer the agent or create
tools. Generated code searches through `ctx.http.web_search` instead. A
Competition creation reaches git only through the quarantine lane, sim
re-verification and a human pull request.

### The never-list

It applies at every level, to generated code and to the reviewed manifests of
git-tier tools.

| Verbs | Names | Why |
| ----- | ----- | --- |
| writes | `/from_human`, `/walkie/console`, `/walkie/reset`, `/walkie/speaking`, `/voice_commands`, `/clock`; the new `/walkie/reset_session`, `/walkie/task` and `/walkie/todos` ([ADR-0038](decisions/0038-own-graph-per-turn-binding.md), [ADR-0041](decisions/0041-todo-list-and-seeding.md)); `/walkie/approve`, reserved | the agent's control plane, and sim time |
| any | `/controller_manager/*`, `*/set_parameters`, `*/set_parameters_atomically`, `*/change_state`, `/lifecycle_manager*/*` | reconfiguring or stopping nodes |
| any | `/mujoco_ros2_control_node/*` | the simulator is also the harvest's evaluator |
| any | `/move_action`, `/go_to_pose*`, `/execute_stored_plan`, `/apply_planning_scene` | MoveIt 2 and commander planning |
| any | `/go_to_home`, `/toggle_gripper_collision`, `/clear_collision_objects`, `/toggle_octomap_collision`, `/toggle_all_collision_checking` | commander motion and collision settings that no atomic uses |
| any, except in the human-written arm atomics' manifests | `/set_joint_position`, `/control_gripper`, and the `RunSkill` action once its name is pinned | the arm moves only through the arm atomics ([ADR-0021](decisions/0021-arm-control-through-commander.md), [ADR-0027](decisions/0027-runskill-action.md)) |
| any | ZED control services (names still to be taken from walkie-ros) | camera configuration |
| any | `/*arm_controller/*`, `/*gripper_controller/*`, `/*_forward_position_controller/*`, `/*_forward_velocity_controller/*`, `/*_joint_trajectory_controller/*`, `/lift_controller/*`, `/head_servo_controller/*`, `/lift/cmd` | arms only through atomics, head and lift with them; names from walkie-ros's `mujoco_controllers.yaml` and `real_controllers.yaml`, and the lift nodes' input |

- walkie-ros's MuJoCo launch starts neither the commander nor `move_group`
  today. The commander and skill entries matter once it does, and already in
  the harvest, where a `FakeSkillServer` serves the skill action. They land
  in phase 4a with raw actions at Unrestricted.
- Names drift as walkie-ros changes. The Competition allow-lists are
  generated from the robot launch graph, a property test checks the list
  against a captured set of names, and interface ADRs are reviewed for new
  names.

### Internet and web search

Every request goes through the parent's broker. Address vetting is the same
at every level.

| | Unrestricted | Standard | Competition |
| - | ------------ | -------- | ----------- |
| Methods | GET, HEAD, POST (JSON or form, ≤ 256 KB) | GET, HEAD, POST JSON ≤ 64 KB | GET; the parent's Tavily fallback search is the only POST |
| Schemes and ports | http, https on 80, 443, 8080, 8443 | https on 443 | https on 443 |
| Hosts | any public host by name | any public host by name | named sources with pinned templates |
| Requests | 50 per invocation, 300 per minute | 20, 120 | 5, 30, 1 concurrent per host |
| Body, timeout | 10 MiB, 30 s | 2 MiB, 15 s | 1 MiB, connect 3 s, total 8 s |
| Redirects | ≤ 5 | ≤ 3 | ≤ 3, same source only; none for `web_snippets` |
| Search results (`search_k`) | 10 | 5 | 3 |
| Search text | snippets ≤ 500, 6,000 in all | ≤ 300, 2,500 | titles ≤ 100, snippets ≤ 300, 1,500 in all, no URLs |
| Search rate | 10 per invocation, 60 per minute | 5, 30 | 2 per turn or invocation, 6 per minute, 120 per hour for all visitors |

- Text, JSON, XML, RSS and Atom return as content. Audio returns as a
  handle, as do images at Standard and Unrestricted and
  `application/octet-stream` at Unrestricted. No IP-literal hosts, unless a
  harvest config names one.
- The Competition named sources are `open_meteo`, `open_meteo_geocode`,
  `frankfurter`, `wikipedia_summary`, `commons_search`, `commons_media` and
  `web_snippets`. Only people edit the list.
  [ADR-0049](decisions/0049-competition-internet-egress.md) has the templates.
- Web search adds no new kind of policy field: `web_snippets` is a source,
  its limits are `search_*` caps, and Competition adds
  `require_search_no_urls` and `require_query_provenance`.
- `[competition]` sets both `search_per_turn = 2` and
  `search_per_invocation = 2`. Without the second, narrowing would keep
  Standard's 5.
- Search keys are split by provider, machine and caller. The robot uses
  `WALKIE_BRAVE_KEY_ROBOT` and `WALKIE_TAVILY_KEY_ROBOT`. The sim coder, sim
  generated tools and the harvest use `WALKIE_BRAVE_KEY_SIM` and
  `WALKIE_TAVILY_KEY_SIM`. The search client picks the keys for the
  caller's role. The keys and the kill switch, `[walkie.search] enabled` in
  `config.toml`, are new and live outside the policy file.

### Budgets

| Budget | Unrestricted | Standard | Competition |
| ------ | ------------ | -------- | ----------- |
| `request_tool` | 3 per task | 2 per task | 1 per request, ≤ 3 per test, ≤ 10 per hour at a booth |
| Coder | 30 model calls, 6 test runs, 20 web calls, 15 min, $2 | 20, 4, 10, 10 min, $1 | 4 model calls, 2 test runs, 3 named-source fetches, no search, $0.10, 25 s, within `request_tool`'s 30 s |
| Tool invocation | 300 s wall, 60 s CPU, 200 `ctx.tools` calls, 1,000 ROS calls | 60 s, 20 s, 50, 200 | 15 s, 5 s, 10, 0 |
| Audio per invocation | 30 s queued | 20 s | 8 s, 3 utterances of ≤ 200 characters, 20 s per minute |
| Sandbox | 1 GiB, 48 tasks | 1 GiB, 48 tasks | 512 MiB, 16 tasks; the resource scope is required |

Every level caps sandbox files at 16 MiB and disables core dumps. At
Competition the whole spoken request is capped at 40 s.

## Choosing the level

Only the `--access-level` flag requests a level, and a missing flag means
Competition. There is no `config.toml` key, because that file reaches the
robot through git.

| Launch | Flag | Domain and router |
| ------ | ---- | ----------------- |
| `pixi run agent-sim` (changed; today it passes no level) | `--access-level standard` | the shell's `ROS_DOMAIN_ID` ([ADR-0036](decisions/0036-agent-sim-shell-domain.md)) |
| `pixi run agent-sim-unrestricted` (new) | `--access-level unrestricted` | a dedicated domain with a loopback-only router (proposed) |
| Harvest tasks | a level per episode | the harvest's own sim |
| `pixi run agent`, the robot launch (changed; today it passes no level) | `--access-level competition`, hard-pinned | 23 |

A request for Standard or Unrestricted needs checks 1 to 3 and 5 to 9 below
to pass, or the agent runs at Competition. Unrestricted also needs check 4,
or it becomes Standard.

## The sim interlock

`src/walkie_agent/toolbox/interlock.py` (new) can only lower the requested
level ([ADR-0040](decisions/0040-sim-interlock.md)).

| # | Check | Sim value | Real robot | Notes |
| - | ----- | --------- | ---------- | ----- |
| 1 | Sim-host marker | root-owned `/etc/walkie/sim-host`, read with `os.stat` | absent | `ENOENT` or any other `OSError`, such as `EACCES`, fails. `os.path.exists` would hide `EACCES`. |
| 2 | No robot marker | `/etc/walkie/robot` absent (`ENOENT`) | present | A veto. Any other `OSError` counts as present. |
| 3 | Domain | `ROS_DOMAIN_ID` is not 23 | 23 | walkie-ros's `connect_walkie_zenoh_router.sh` exports 23 and bridges to `tcp/10.0.0.6:7447`. A negative signal only. |
| 4 | Zenoh (Unrestricted only) | `ZENOH_SESSION_CONFIG_URI` and `ZENOH_CONFIG_OVERRIDE` connect only to loopback, on a dedicated domain | — | Otherwise at most Standard. The agent cannot see its router's endpoints; checks 3 and 8 cover a bridged router. |
| 5 | Sim time | `--use-sim-time` given | not given | |
| 6 | Hardware plugin | `/robot_description` contains `mujoco_ros2_control/MujocoSystemInterface` | `CubeMarsSystemHardware`, `DynamixelHardware` or `OpenArm_v10HW` | One latched read. Missing, mock or gz fails. |
| 7 | Sim node | `/mujoco_ros2_control_node` serves `.../reset_world` (`mujoco_ros2_control_msgs/srv/ResetWorld`) and publishes `/clock` | absent | |
| 8 | No real drivers | none of the nodes named exactly `unitree_lidar_ros2_node`, `urg_node2`, `lakibeam_lidar_node`, `zed_node`, `zed_watchdog`, `head_touch_beep` | present | Exact node names, not topic prefixes. `foxglove_bridge` and `rosbridge_websocket` are tool bridges, not hardware drivers. |
| 9 | Host facts | no `can0`, `can1` or `/dev/ttyACM0` | present | Cheap, though it adds little after checks 1 and 2. |

Every value is logged at startup and on change. A sim on the robot PC always
runs at Competition (check 2). The signals are not authenticated (no SROS2,
[ADR-0009](decisions/0009-forbidden-names-guard-rail.md)), so the interlock
guards against mistakes, not against an attacker on the graph.

- **Rollout.** Phase 2 ships checks 1, 2, 3, 5 and 6 and the startup log.
  Checks 4 and 7 to 9, the periodic re-checks and the fresh checks before
  motion land in phase 4a, before the harvest relies on them in phase 5.
- **Re-checks.** Checks 6 to 8 re-run every 5 s. Before every N or M
  `ctx.tools` call, publish or action start at Standard or Unrestricted,
  checks 3 and 8 re-run uncached: one graph query of a few ms, which closes
  the 5 s window for motion. A failed fresh check refuses that call and
  demotes the session.
- **Demotion.** A failed check drops the session to Competition until the
  agent restarts. It never rises again on its own. Invocations above
  Competition are killed and cleaned up, bundles get `REFUSED` at their next
  step (with a stow if an arm step started), and the coder is aborted. Tools
  above Competition disappear from the next bind. The agent says
  "Simulation check failed (<check>). Running at Competition level." on
  `/walkie/console` and in the trailing reminder. The event is logged with
  every check value.
- **Paused sim.** If `/clock` stops but `mujoco_ros2_control_node` still
  publishes it, invocations wait for the clock or their deadline. This is
  not a demotion.
- **Harvest relaunch.** A vanished sim demotes the agent until it restarts,
  so the harvest stops the agent before each relaunch of `chula_pat14_6`. It
  starts a new one only after the sim serves `reset_world`, advances `/clock`
  and has published `/robot_description`. `chula_pat14_6` uses the same
  `mujoco_omnibot.launch.py` and xacro as the default world, so checks 6 and
  7 should see the same plugin and node. The world's assets and tools are on
  a walkie-ros branch that is not on main
  ([Sim Harvest](harvest.md#the-world-chula_pat14_6)). Re-test both checks
  once that change is merged; the walkie-ros paths on this page may change
  with it.

## Level stamps

Every generated tool is stamped with the lowest level that allows all its
calls, and loads only at that level or above. Git-tier tools carry a reviewed
manifest (`META.ros`, `META.net`) instead, under the same never-list and
MoveIt ban.

- **From lint.** For each call, lint finds the lowest level that exposes
  that tool, allows that verb, literal name and type, or allows that literal
  source, method and scheme. A computed ROS name or host, or a call to
  `ctx.tools.search`, stamps the tool at the session level. At Competition,
  `ctx.http` plus anything outside the L14 list stamps it Standard. A bundle
  takes the maximum over its steps.
- **At runtime.** The host enforces `min(session level, stamp)`. A call above
  the stamp gets `POLICY_REFUSED`, is traced, and blocks the tool until it is
  re-linted and a human acknowledges it. A stamp is never raised
  automatically, so a tool that tried more than it declared gains nothing.
- **From traces.** Runs record the highest level their calls needed. The
  harvest report flags tools stamped Standard only for a computed name whose
  traces never left Competition.

## Lint

Bundles are data, so the linker checks them completely. Diagnostics have the
shape `{code, step, port, message, hint}` and go back to the atomic agent.

| Code | Check | Phase |
| ---- | ----- | ----- |
| L01 | Shape: a `BundleSpec` with no extra fields, ≤ 12 steps and ≤ 6 params | 2 |
| L02 | Unknown, deprecated or tier-disallowed tool | 2 |
| L03 | Duplicate step id, or a `Ref` to an unknown or later step; proves no cycles | 2 |
| L04 | Missing required input, or an input the tool does not have | 2 |
| L05 | Type mismatch: assignable only for equal types, a `Literal` subset, element-wise lists, `T` into an optional `T`, or the same handle type | 2 |
| L06 | A literal fails its validator, e.g. `pose="both_arms_lift"` | 2 |
| L07 | `on` names a code the tool cannot raise, or a retry on a non-idempotent, N or M step | 3 |
| L08 | Provenance: handle inputs take a `Ref` or a handle-typed `Arg`, never a literal | 3 |
| L09 | Effect policy for the tier and level | 3 |
| L10 | Duplicate tool: same atomic steps and parameter names | 3 |
| L11 | Templates use only existing `{param}` or `{step.field}` placeholders, filled by a plain substituter, never `str.format` | 3 |
| L12 | Trivial wrapper; the hint names the call to make, e.g. `speak(text='Meow!')` | 3 |
| L13 | Level: a step, `ctx` call or name needs more than the session allows; also computes the stamp | 3 |
| L14 | Rule of Two at Competition: a `ctx.http` user may otherwise call only `speak`, `play_sound`, `tell_time`, `tell_date`, `wait_for_seconds` and `ctx.now`; `get_current_pose` and `list_locations` count as sensing | 6 |
| L15 | Name matches `^[a-z][a-z0-9_]{2,40}$`, is unique and not reserved; `tmp_` for drafts, `sess_` for Competition session tools | 3 |

L14 lists `speak`, but a web-using invocation is tainted. Whether a tainted
generated invocation may call `speak` is open on
[Competition Mode](competition.md#open-questions).

A symbolic dry run follows linking. **Generated code** gets an AST lint per
level (`lint_vibed.py`) for fast feedback and stamps. It bans `eval`, `exec`,
`compile`, `open`, `__import__`, `getattr`, `setattr`, `globals`, `locals`,
`vars`, attributes starting with `_` or named `sys`, `os`, `modules` or
`builtins`, attributes off a per-module allow-list, and a bare `except` that
returns success. Competition imports only `math`, `statistics`, `datetime`,
`zoneinfo`, `decimal`, `fractions`, `re`, `json`, `itertools`,
`collections`, `random`, `string`, `textwrap`, `unicodedata`, `pydantic` and
`walkie_toolbox_facade`, and needs literal hosts. Standard adds `numpy`.
Unrestricted adds `scipy`, and `cv2` if phase 4b ships.

The lint is not a boundary. `statistics.sys.modules["os"]` and `random._os`
reach `os` without a dunder, and Python has more such paths than a lint can
close. The sandbox is the boundary.

## The shared ctx API

`src/walkie_toolbox_facade/ctx.py` defines one `Ctx` protocol with two
backings. Human-written atomics with `META.runtime = "inprocess"` get
`ctx_inproc.Ctx`, which calls the connector, registry and broker directly and
enforces the tool's manifest. Sandboxed code gets `walkie_toolbox_facade.Ctx`,
which sends each call as JSON-RPC on fd 3.

| Member | What it does | Levels |
| ------ | ------------ | ------ |
| `level`, `deadline`, `now()` | the session level, a monotonic deadline, aware venue time | all |
| `tools.<name>(...)`, `tools.describe(name)` | call or describe an exposed atomic | all, per the `ctx.tools` row |
| `tools.search(query, k=5)` | read-only tool search | Unrestricted, Standard (open question 2) |
| `ros.graph()`, `ros.read(topic)`, `ros.tf(target, source)` | the graph, one message (stripped, or a handle), a transform | Unrestricted, Standard |
| `ros.call(service, request)` | a service call | Unrestricted; read-only list at Standard |
| `ros.action.start`, `poll`, `wait`, `cancel` | actions, cancelled on exit | Unrestricted; planning only at Standard |
| `ros.publish(topic, msg)` | base commands | Unrestricted, only if phase 4b ships |
| `http.get(url, params=...)` | a GET; `HttpResult` is always `untrusted=True` | all |
| `http.post_json(url, body)` | a JSON POST | Unrestricted, Standard |
| `http.web_search(query, lang="en")` | `SearchResult` items `{title, snippet, site, url?}`; `url` only in sim | all; under L14 at Competition |
| `assets.get(name)` | an asset handle, never bytes at Competition | all |
| `handles.load(handle)` | copies ≤ 8 MB into the sandbox as a numpy array | Unrestricted, only if phase 4b ships |

Web search sits under `ctx.http`, so stamps, caps, vetting and L14 apply.
Promoted generated code gets a reviewed manifest and keeps
`runtime = "sandbox"` until a human rewrites it.

## Sandbox and bridges

Bubblewrap runs the same way at every level, and only the parent holds the
OpenRouter, Langfuse and search keys
([ADR-0047](decisions/0047-sandbox-and-bridges.md)). Direct `rclpy` in the
sandbox would need `--share-net`, which exposes the Zenoh router on `*:7447`,
loopback services, the LAN and D-Bus, and orphans goals when the sandbox
dies. So the bridges are host-mediated: an RPC costs p50 26–48 µs plus
conversion (744 µs for a 720-ray `LaserScan`), against about 0.73 s per
`rclpy` spawn, measured without a Zenoh router. `sandbox/launch.py` builds
roughly this command:

```bash
systemd-run --user --scope --quiet -p TasksMax=$TASKS -p MemoryMax=${MEM}M -p MemorySwapMax=0 -- \
/usr/bin/prlimit --as=2147483648 --cpu=$CPU_S --fsize=16777216 --core=0 -- \
$BWRAP --unshare-all --unshare-user --disable-userns --die-with-parent --new-session --as-pid-1 --clearenv \
  --ro-bind /usr /usr --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
  --symlink usr/bin /bin --symlink usr/sbin /sbin \
  --proc /proc --dev /dev --size 67108864 --tmpfs /tmp --tmpfs "$HOME" \
  --ro-bind "$PIXI_ENV" "$PIXI_ENV" \
  --ro-bind "$REPO/src/walkie_toolbox_facade" /worker/walkie_toolbox_facade --chdir /worker \
  --setenv HOME "$HOME" --setenv PATH /usr/bin --setenv MALLOC_ARENA_MAX 2 \
  --setenv OPENBLAS_NUM_THREADS 1 --setenv OMP_NUM_THREADS 1 \
  --setenv PYDANTIC_DISABLE_PLUGINS __all__ \
  -- "$PIXI_ENV/bin/python" -I /worker/walkie_toolbox_facade/worker.py
```

- `Popen` passes only `XDG_RUNTIME_DIR` and the RPC fd; without the variable,
  `systemd-run --user` fails with "Failed to connect to bus" (verified).
  `--clearenv` with `--as-pid-1` leaves `/proc/1/environ` clean (verified).
- Limits come from `prlimit`, because `preexec_fn` is unsafe with threads.
  `RLIMIT_NPROC` breaks bwrap, so `TasksMax` and `MemoryMax` stop fork and
  memory bombs. Deadlines use SIGKILL, because PID 1 ignores SIGTERM.
- `--ro-bind / /` is not used: it made `.env` and `~/.ssh` readable
  (verified).
- If bwrap or the scope cannot start, generated tools and sandboxed atomics
  are refused, loudly. At Competition the robot says it cannot make new tools
  right now, and reuse and compose of tools that need no sandbox keep
  working. Whether `fetch_sound` survives is open on
  [Competition Mode](competition.md#open-questions), since its decode step
  needs bwrap. In sim the run stops.
- **RPC.** JSON-RPC on fd 3 only; fd 1 and 2 go to a capped log pipe. The
  parent takes tool, `sha8`, stamp and invocation id from its own record of
  the worker, rejects frames naming anything else, and caps frames at 1 MiB
  and their count at `tools_calls + ros_calls + http_per_invocation`.
- **Warm workers.** One per level imports pydantic, runs one request, exits
  and is replaced. Landlock and seccomp inside the worker are deferred;
  `bwrap --seccomp` could add them later.

Medians on a team dev machine: bwrap alone 6.2 ms, with trivial Python
16.5 ms, with pydantic 140.7 ms; warm spawn 120 ms, off the critical path; a
call on a warm worker 0.8 ms; RPC p50 26–48 µs, p99 69–288 µs; the
`systemd-run` prefix 15–30 ms. A 640×480 image as JSON took 654 ms and
4.2 MB, which is why images are handles.

### ROS bridge

`bridge/ros.py` runs on its own node in the agent process. Per call it
resolves the name (`resolve_topic_name`, `resolve_service_name`), collapses
`//`, strips `/_action/send_goal`, `cancel_goal`, `feedback` and `status`,
takes the type from the graph (a caller's type must match), checks the
never-list and the globs, and re-runs the fresh interlock checks before
motion.

- A read creates a subscription, takes one message and destroys it. RAI's
  `receive_message` keeps subscribers forever.
- A publish waits up to 1 s for a subscriber, else returns `NO_SUBSCRIBER`,
  the same workaround as the `speak` tool.
- Calls and actions use plain `rclpy` clients, because RAI 2.12's action
  `done_callback` is a TODO. Feedback is polled. Each deadline is the smaller
  of the call timeout and the invocation's time left.
- Messages convert with `set_message_fields` and `message_to_ordereddict`.
  Nested images, point clouds and `*embedding` fields become handles at
  Unrestricted and are dropped at Standard. Anything over 256 KB is a handle.
- Each call adds a line to `runs/ros/<date>.jsonl` with its verb, name,
  decision and duration.
- On end, worker death, reset or demotion, the bridge cancels the
  invocation's goals, publishes one zero `Twist` or `TwistStamped` on each
  topic it published, and drops its publishers and clients. A type without a
  known stop message cannot be published to a command topic.

### HTTP broker, search and audio

`bridge/http.py` runs in the parent, on a worker thread.

- **Normalize.** Reject percent-encoded `.`, `/` and `\`, `;`, empty segments
  and userinfo; apply `remove_dot_segments`; match paths on segment
  boundaries, so `/v1/forecastX` misses `/v1/forecast`.
- **Vet and pin.** Resolve once. Reject non-global, local-interface, on-link
  and docker-subnet addresses, IPv6 forms that embed a private IPv4
  (v4-mapped, 6to4, Teredo, NAT64), and numeric spellings such as `127.1` or
  `0x7f.0.0.1`. Connect to the vetted IP with urllib3, with
  `server_hostname` and `assert_hostname` set to the name and proxies
  ignored. Never re-resolving defeats DNS rebinding.
- **Cap and log.** Caps count decoded bytes, so a gzip bomb stops at the cap.
  Every request carries a fixed User-Agent with the team contact, which
  Wikimedia requires. Generated code cannot set `Authorization`, `Cookie` or
  `Proxy-*`; only the search client adds a key, for the provider's pinned
  host. Requests go to `runs/http/<date>.jsonl` and Langfuse. Offline, calls
  fail with `UNAVAILABLE`.
- **Status.** The prototype passed network-free tests for addresses,
  spellings, redirects, gzip bombs and rebinding. Path normalization,
  templates, the local and on-link checks, live TLS and IPv6 fallback are
  untested.

`bridge/search.py` (about 50 lines) tries Brave, then Tavily only after a
fast Brave failure, with one attempt per provider (connect 2 s, total 3 s)
and 4 s for the whole search
([ADR-0049](decisions/0049-competition-internet-egress.md)). It strips
HTML, entities, control and bidi characters, applies
`config/search_denylist.txt` and the caps, drops URLs at Competition, and
names sites from `config/site_names.toml`. A failing provider is skipped
for 60 s, or for the session on 401 or 403. Logs keep a query hash, never the
query. `FakeSearchProvider` serves tests.

`sandbox/decode.py` accepts only Ogg, WAV, FLAC and MP3 by magic bytes, then
runs `ffmpeg -nostdin -protocol_whitelist pipe -f <fmt> -i pipe:0 -t 10 -f
s16le -ac 1 -ar 48000 pipe:1` in its own bwrap invocation, with the same
profile and a 5 s CPU cap. `ffmpeg` is already in the pixi env. The parent
only handles PCM; `pydub` would run `ffmpeg` next to the keys. At
Competition, Silero VAD and local Whisper check the clip. A clip with speech
is rejected, or its transcript goes through the speech gate.

### Robot PC

`scripts/robot_check.sh` (new) runs as the agent's user, read-only, with no
root. It reports the kernel, the userns sysctls, AppArmor, a bwrap smoke test
with the exact flags (`BWRAP_OK` or `BWRAP_FAILED`), whether `systemd-run
--user --scope` works with only `XDG_RUNTIME_DIR`, and whether the agent runs
inside Docker. On `BWRAP_FAILED` from Ubuntu 24.04's userns restriction, an
AppArmor profile grants `userns` to a root-owned copy at
`/opt/walkie/bin/bwrap`: root once, about 10 minutes. The copy misses apt
updates, so `robot_check.sh` fails if its version differs from
`/usr/bin/bwrap`. It also fails if `/etc/walkie/sim-host` exists. The robot
gets `/etc/walkie/robot`, never `/etc/walkie/sim-host`, and no runtime
installs.

??? example "Illustrative AppArmor profile (untested)"

    Modeled on Ubuntu's own `/etc/apparmor.d/chrome`. The profile name is
    not chosen, and nothing here has run on the robot PC.

    ```text
    abi <abi/4.0>,
    include <tunables/global>
    profile <profile-name> /opt/walkie/bin/bwrap flags=(unconfined) {
      userns,
      include if exists <local/<profile-name>>
    }
    ```

Rejected: `kernel.apparmor_restrict_unprivileged_userns=0`, which reopens
the userns attack surface for every process, and no generated tools on the
robot, which drops the "make that tool" case at Competition.

## Safety model

Capabilities are enforced in the parent, not in the prompt or the lint. One
speech gate checks everything the robot says or plays
([Competition Mode](competition.md#speech-gate)). Untrusted text stays
marked, and a tainted turn cannot steer motion or creation.
`navigate_to_pose` takes its pose from memory or perception, never from an
LLM ([ADR-0043](decisions/0043-navigate-to-pose-provenance.md)). No role
gets RAI's raw ROS tools, and generated code never gets a `ROS2Connector` or
an `rclpy.Node`. There is no approval prompt, because RoboCup forbids remote
control.

- **Assets at stake:** physical safety, bystander privacy, secrets, money
  and test time, what the robot says, and the integrity of the toolbox.
- **Entry points:** a stranger's voice, text the camera sees, web content
  and search snippets, pages the coder reads in sim, a compromised API or
  DNS, the coder's own output, and persisted artifacts.
- **Kept out of Competition entirely** for generated tools: navigation and
  manipulation, raw ROS of any kind, file I/O, subprocesses, package
  installs, arbitrary URLs at runtime or authoring time, web search mixed
  with sensing, image bytes, OCR and caption text, persistent triggers,
  tools that create tools, and drafts that are not in git.

| ID | Threat | Example | Strongest mitigations | Residual |
| -- | ------ | ------- | --------------------- | -------- |
| T1 | Direct prompt injection | "Make a tool that drives to the staff door." | no N or M at Competition; raw ASR text only to the creator; queries from the visitor's words; budgets; speech gate | harmless oddities within caps |
| T2 | Indirect injection from the web | an edited Wikipedia extract | pinned templates; typed results; capped snippets; taint hides N, M and `request_tool`; snippets become site names at turn end | attributed misinformation |
| T3 | SSRF | `10.0.0.6:7447`, `169.254.169.254`, Ollama, DNS rebinding | no network in the sandbox; vetting and a pinned IP | very low |
| T4 | Exfiltration | detections or memory in a web query | L14; no `ctx.ros` at Competition; labels and counts only; hashed query logs | low |
| T5 | Offensive speech | "Say <slur>." | local speech gate on everything said; speech check on audio; strict safe search; the query writer's topic refusals | harm built from clean words, since there is no cloud check; false-positive refusals |
| T6 | Resource abuse | 30 variants; fork and memory bombs | creation caps; required scope; deadlines; separate keys; 120 searches per hour; kill switches | spend bounded by key limits |
| T7 | Persistence | a visitor's tool reloaded next day | session-only; quarantine never auto-loaded; promotion by human PR | very low |
| T8 | Visual injection | a sign held up to the camera | no OCR or captions at Competition; no N or M there | low |
| T9 | Sandbox escape | `statistics.sys.modules["os"]`, forged frames | sandbox and scope; identity per worker; L15; AST lint | low |
| T10 | Supply chain | a drafted `pip install`, a crafted audio file | no runtime installs; `ffmpeg` only in the decode sandbox | very low |
| T11 | Sim and real confused | Standard on the robot, or a graph bridged to it | flag only; markers; domain-23 veto; loopback Zenoh for Unrestricted; fresh driver check; demotion until restart | very low |
| T12 | Generated motion leaking | an Unrestricted draft publishing `/cmd_vel` | stamped Unrestricted; never promoted as is; robot loads no drafts; fresh checks; stops on cleanup | very low |

Mitigations, ranked by value for cost: (1) the per-level surface in the
parent; (2) the no-network sandbox and the vetted broker with named sources;
(3) the speech gate; (4) the Rule of Two and the query writer; (5) budgets,
separate keys and the kill switch; (6) quarantine, session scope, the
quarantine lane and git-only loading; (7) untrusted wrapping, typed results,
capped snippets and turn taint; (8) the AST lint and import allow-list; (9)
no OCR or captions at Competition; (10) prompt rules.

**The Rule of Two and its one exception.** A Competition tool may add private
local sensing or web access, never both. The coder follows the same rule: at
Competition it has no open web and no session data. The task agent is the
exception. It holds memory and detections, and `look_up_online` reaches the
web in the same turn. The guard is that `look_up_online` takes no text
argument: a separate query writer sees only this task's human utterances.

## What lives where

| Piece | Repository | Location |
| ----- | ---------- | -------- |
| Level policy | walkie-agent-ros | `config/access_levels.toml` (new) |
| Levels, interlock | walkie-agent-ros | `src/walkie_agent/toolbox/levels.py`, `interlock.py`; `tests/test_levels.py`, `test_interlock.py` (new) |
| `--access-level` flag | walkie-agent-ros | `src/walkie_agent/main.py` (changed) |
| Launch tasks | walkie-agent-ros | `pixi.toml`: `agent` and `agent-sim` (changed), `agent-sim-unrestricted` (new) |
| Linker, AST lint | walkie-agent-ros | `src/walkie_agent/toolbox/linker.py`, `lint_vibed.py` (new) |
| `ctx` API | walkie-agent-ros | `src/walkie_toolbox_facade/` and `src/walkie_agent/toolbox/ctx_inproc.py` (new) |
| Sandbox, bridges | walkie-agent-ros | `src/walkie_agent/toolbox/sandbox/` and `bridge/` (new) |
| Search keys, kill switch | walkie-agent-ros | `.env.example`: `WALKIE_BRAVE_KEY_ROBOT`, `WALKIE_TAVILY_KEY_ROBOT`, `WALKIE_BRAVE_KEY_SIM`, `WALKIE_TAVILY_KEY_SIM`; `config.toml`: `[walkie.search]` (new) |
| Search shaping | walkie-agent-ros | `config/search_denylist.txt`, `config/site_names.toml` (new) |
| Robot-PC check | walkie-agent-ros | `scripts/robot_check.sh` (new) |
| Traces | each sim or robot machine | `$WALKIE_TOOLS_HOME/runs/` (new, never in git) |
| Markers | dev and harvest machines; robot PC | `/etc/walkie/sim-host`; `/etc/walkie/robot` (new, root-owned) |
| bwrap copy, AppArmor profile | robot PC, only on `BWRAP_FAILED` | `/opt/walkie/bin/bwrap`, a profile in `/etc/apparmor.d/` (new) |
| Controller names | walkie-ros (existing) | `description/config/ros2_controller/mujoco_controllers.yaml` and `real_controllers.yaml` (robot-description submodule), `bringup/config/openarm/openarm_v10_bimanual_controllers.yaml` |
| Router script, sim launch | walkie-ros (existing) | `scripts/bringup/connect_walkie_zenoh_router.sh`, `bringup/launch/mujoco_omnibot.launch.py` |

## Open questions

1. **Domain and router per level.** Recommended: Standard on the shell's
   domain, as ADR-0036 intends; Unrestricted only on a dedicated domain with
   a loopback-only router. The alternatives are both on the shell's domain
   (only the domain-23 veto and driver check guard it) or both on a dedicated
   domain (any sim level needs a relaunch). The domain number is not chosen.
   Blocks `agent-sim-unrestricted` and check 4, in phase 4a.
2. **Runtime tool search for generated tools.** Recommended: read-only
   `ctx.tools.search` at Unrestricted and Standard, `describe` only at
   Competition, where search would make stamps computed. Blocks
   `bridge/tools.py`, in phase 4a.
3. **May generated code read `/mujoco_ground_truth/odom`?** It is the sim's
   true base pose, and the harvest scores position checks on it. A tool that
   reads it works in sim and finds nothing on the robot. The options are to
   put it on the never-list for reads at every level, or to leave it
   readable at Standard and Unrestricted and rely on harvest review to
   reject tools whose traces read it. No recommendation yet. Blocks the sim
   entries of the never-list, in phase 4a.
4. **Unverified: does bwrap run on the robot PC?** The approach is decided:
   run `robot_check.sh`, and install the AppArmor profile only on
   `BWRAP_FAILED`. The script does not exist yet; it is written and run in
   phase 0. Blocks Competition creation on the robot, in phase 6.

Related questions live on [Competition Mode](competition.md#open-questions)
(whether a tainted generated invocation may call `speak`, how `fetch_sound`
decodes without bwrap, trivial requests),
[Self-Evolving Agent](self-evolving-agent.md#open-questions)
(`navigate_to_pose`) and
[Sim Harvest](harvest.md) (the share of episodes at Unrestricted).

## Decisions

Click a decision to expand it. All of these are proposed. See
[Architecture](index.md) for how decisions work.

??? note "ADR-0039: Give generated tools three nested access levels · Proposed"

    --8<-- "architecture/decisions/0039-access-levels.md:body"

??? note "ADR-0040: Grant Standard and Unrestricted only behind a sim interlock · Proposed"

    --8<-- "architecture/decisions/0040-sim-interlock.md:body"

??? note "ADR-0047: Run generated code in a bubblewrap sandbox behind host-mediated bridges · Proposed"

    --8<-- "architecture/decisions/0047-sandbox-and-bridges.md:body"
