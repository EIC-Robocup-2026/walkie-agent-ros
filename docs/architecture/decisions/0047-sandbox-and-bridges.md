# ADR-0047: Run generated code in a bubblewrap sandbox behind host-mediated bridges

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

Generated tools in the `temporary` tier, Competition session tools and
promoted generated atomics are Python written by an LLM, often at a
stranger's request
([ADR-0037](/architecture/decisions/0037-tool-tiers-bundles-effects.md)).
They run in sim and, at Competition, on the robot PC. The access levels
([ADR-0039](/architecture/decisions/0039-access-levels.md)) say what such
code may ask for. Something has to stop it from doing anything else.

The AST lint cannot. `statistics.sys.modules["os"]` and `random._os` reach
`os` without a dunder. The planned AST lint bans those attribute names, but
Python has too many such paths for a lint to close them all.

The agent process holds the OpenRouter, Langfuse and search-provider keys,
and sits on the ROS graph and the LAN. Generated code must reach none of
them directly.

**Decision**

- **One sandbox at every level.** Each worker runs under
  `systemd-run --user --scope` (`TasksMax`, `MemoryMax`, no swap), then
  `prlimit` (address space, CPU seconds, 16 MiB file size, no core dumps),
  then bubblewrap with `--unshare-all`, `--disable-userns`, `--clearenv`,
  `--as-pid-1`, read-only binds of `/usr`, the pixi env and
  `src/walkie_toolbox_facade/` only, a 64 MB tmpfs `/tmp` and a tmpfs
  `$HOME`. No network, no secrets, no ROS. "Unrestricted" describes the ROS
  and web policy, never the sandbox.
- **One channel.** The worker speaks JSON-RPC on fd 3 only. The parent takes
  the tool, hash, stamp and invocation id from its own record of that
  worker, never from the frame, and caps frame size and count.
- **Warm workers.** One per level, spawned ahead of time, used once, then
  replaced.
- **Host-mediated bridges** in the parent serve `ctx.ros`, `ctx.http`,
  `ctx.tools` and web search. They check each call against
  `min(session level, stamp)` or the reviewed manifest, trace it, and clean
  up when the invocation ends: cancel goals, send stop messages, drop
  publishers.
- **The HTTP broker** normalizes URLs, vets every resolved address, pins the
  IP for the connection and caps decoded bytes. Only the search client adds
  a key, and only for the provider's pinned host.
- **Fetched audio** is decoded by `ffmpeg` (already in the pixi env) in its
  own bwrap invocation with the same profile (`sandbox/decode.py`), after a
  magic-byte check. The parent only handles PCM.
- **On the robot PC**, `scripts/robot_check.sh` (new, read-only, no root)
  runs a bwrap smoke test with the exact flags. If it reports
  `BWRAP_FAILED` because Ubuntu 24.04 restricts unprivileged user
  namespaces, an AppArmor profile grants `userns` to a root-owned copy at
  `/opt/walkie/bin/bwrap`. That takes root once, about 10 minutes, and no
  code change. The runner then uses that path.
- **No silent fallback.** If bwrap or the scope cannot start, generated
  tools and sandboxed atomics are refused. At Competition the robot says it
  cannot make new tools right now; reuse and compose of tools that need no
  sandbox keep working. Whether `fetch_sound` survives is open on
  [Competition Mode](/architecture/competition.md#open-questions), since its
  decode step needs bwrap. In sim the run stops.

We rejected:

| Option | Why not |
| ------ | ------- |
| Direct `rclpy` in the sandbox | Needs `--share-net`, which exposes the Zenoh router on `*:7447` with no access control, every loopback service (Ollama, Langfuse, MinIO), the LAN, `docker0` and 8 D-Bus abstract sockets. Goals are orphaned when the sandbox dies, because Nav2 does not track client liveness. About 0.73 s per spawn, measured without a Zenoh router. |
| `--ro-bind / /` | Makes `.env` and `~/.ssh` readable (verified). |
| Limits through `preexec_fn` | Unsafe when the parent has threads, and the agent always has rclpy and LangGraph threads. |
| `RLIMIT_NPROC` | Counts every task under the same UID and breaks bwrap. `TasksMax` stops fork bombs instead. |
| Decoding with `pydub` | Runs `ffmpeg` in the parent, with the parent's environment and format probing, next to the keys. |
| `sysctl kernel.apparmor_restrict_unprivileged_userns=0` on the robot | Reopens the user-namespace attack surface for every process on the robot PC. |
| No generated tools on the robot | Drops the "make that tool" case at Competition. |
| Landlock and seccomp from inside the worker | No bindings in the pixi env, so hand-written syscalls for a layer that is redundant inside bwrap. Deferred; `bwrap --seccomp` can load a filter later. |

**Consequences**

- A small tool costs about 1 ms plus its RPCs. A warm spawn takes about
  120 ms off the critical path. Creation time is almost all LLM time.
- Anything the bridges do not offer, generated code cannot do. A missing
  capability becomes a new bridge verb, reviewed like any other code. Direct
  `rclpy` is revisited only if Unrestricted traces show tools repeatedly
  blocked by bridge gaps.
- Images and point clouds travel as handles, never as JSON: a 640×480 image
  as JSON took 654 ms and 4.2 MB.
- Promoted generated code keeps running in the sandbox until a human
  rewrites or audits it.
- The bwrap copy misses apt updates, so `robot_check.sh` fails when its
  version differs from `/usr/bin/bwrap`. It also fails if
  `/etc/walkie/sim-host` exists on the robot. No package is installed at
  runtime, at any level.
- Not yet tested: the broker's path normalization and templates, its local
  and on-link address checks, live TLS over a pinned IP, IPv6 fallback, and
  whether bwrap runs on the robot PC at all.

**See also:** [Access Levels: sandbox and bridges](/architecture/access-levels.md#sandbox-and-bridges),
[ADR-0049](/architecture/decisions/0049-competition-internet-egress.md),
[Competition Mode](/architecture/competition.md)
<!-- --8<-- [end:body] -->
