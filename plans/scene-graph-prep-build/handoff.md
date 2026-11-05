# Handoff: walkie-vision and walkie-database preparation pair

Date: 2026-11-05. Both repositories are on local branch `feat/scene-graph-prep`.
Nothing is pushed. Both worktrees are clean.

## Final commits

| Repository | Worktree | Final commit | Subject |
|---|---|---|---|
| walkie-vision | `/var/tmp/wv-prep` | `00a4fed98e850c44de60bbe63256239b565cd6be` | feat: log the added stages, check memory mode and add the test workflow |
| walkie-database | `/var/tmp/wd-prep` | `704eb247958b60b534641a10b0dde71ccdae06bc` | feat: add the local pair gate for vision and database |

## The vision pin

`/var/tmp/wd-prep/ros2_ws.repos` at the database commit:

```yaml
repositories:
  ros2_ws/src/walkie-vision:
    type: git
    url: https://github.com/EIC-Robocup-2026/walkie-vision.git
    version: 00a4fed98e850c44de60bbe63256239b565cd6be
```

The imported clone `/var/tmp/wd-prep/ros2_ws/src/walkie-vision` is at the same commit,
with no changes. The pin is the vision commit the pair gate tested.

## Lock hashes (SHA-256)

| File | SHA-256 |
|---|---|
| `/var/tmp/wv-prep/pixi.lock` | `2e164b7a2215f9bb14ecc101371cd15d23bf6ea401853930ee8fe134c9a47380` |
| `/var/tmp/wd-prep/pixi.lock` | `84f7af2adca533b5b25e3a5db8ab07e8b2f487cd07470ed9e441da4c3d238846` |

## Unit tests and strict docs, on the final commits

All commands ran under `taskset -c 0,1,6-11` with `/home/tawan/.pixi/bin/pixi`. The
brief's `env` is `/usr/bin/env` here, because `--clean-env` leaves only the
environment's `bin` on `PATH`.

| Run | Command | Result | Log |
|---|---|---|---|
| walkie-database tests | `pixi run --locked --clean-env --manifest-path /var/tmp/wd-prep/pixi.toml /usr/bin/env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 ROS_PYTHON_CHECK_FIELDS=1 CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -rs --no-header -p no:cacheprovider` | 1581 passed, 0 failed, 0 skipped, 2 warnings (lark deprecation) | `build-reports/d9-final-tests.log` |
| walkie-database docs | `pixi run --locked --clean-env -e docs --manifest-path /var/tmp/wd-prep/pixi.toml docs-build` (`mkdocs build --strict`) | Built, no warning, exit 0 | `build-reports/d9-final-docs.log` |
| walkie-vision tests | Same form with `/var/tmp/wv-prep/pixi.toml` and `YOLO_AUTOINSTALL=false` | 1157 passed, 0 failed, 1 skipped, 2 warnings | `build-reports/d9-v-final-pytest.log` |
| walkie-vision docs | `pixi run --locked --clean-env -e docs --manifest-path /var/tmp/wv-prep/pixi.toml docs-build` | Built, no warning, exit 0 | `build-reports/d9-v-final-docs.log` |

The one vision skip is `tests/test_face.py:77`: `buffalo_l` is missing. It needs
`pixi run fetch-models`, which this work must not run.

## The local pair report

- Main run: `/var/tmp/wa-impl-prep/db-vision/build-reports/pair.json`, exit 0,
  `PASSED`. Console output: `build-reports/pair-gate.log`. Run files:
  `build-reports/pair-run/`. Client overlay: `build-reports/client-overlay/`.
- It ran by the brief's recipe, with `--build`, on domain 78, Fast DDS, localhost only,
  no GPU and offline models. All 20 planned checks passed. The overlay build took
  94 s. The run took 2 min 12 s.
- Negative controls, without `--build`, reusing the overlay:
  - `build-reports/pair-negative-hash.json`: exit 1. `hashes` fails on the altered
    `walkie_vision_msgs/msg/ObjectDetectionArray` in `srv/LocalizeObjects.json`.
    Every wire check is a skip.
  - `build-reports/pair-negative-import.json`: exit 1. The vision probe that imports
    `/var/tmp/wd-prep/ros2_ws/install/walkie_database_msgs` fails. Every wire check
    is a skip.
- No process group from any of the three runs is alive afterwards.

Neither repository's remote CI was run. Once both are on GitHub, walkie-database's CI
should run `tools/pair_check.py` against its declared pin.

## Outstanding checks

The pair gate covers these two repositories with fake data only. These remain open.

### Hardware and recordings

- With model files and a GPU: combined VRAM, first-use and steady CLIP latency,
  `DetectObjects` re-prompt gaps, the real 5 Hz and 0.3 s age behaviour, and the
  30-minute OOM target.
- A registered RGB-D recording: mask and depth alignment, point error, colour
  accuracy and crop usefulness.
- MuJoCo: rendered-depth integration, `/clock` resets, and M1's later patrol and
  Present behaviour.
- The robot: TF, calibration and motion effects, localizer accuracy, grasp
  compatibility and query latency under load.
- Zenoh: no peer of the gate ran on Zenoh or through a router.

### walkie-agent-ros (owner's follow-up, not done)

- Pin walkie-vision at `00a4fed98e850c44de60bbe63256239b565cd6be`, and move
  walkie-database from `7f657b0` to `704eb247958b60b534641a10b0dde71ccdae06bc`. Full
  SHAs only. Record this pair report.
- Add the vision clone to `clone-ros2-ws`'s outputs and `walkie_vision_msgs` to
  `build-ros2`'s package list. Declare `ros-jazzy-vision-msgs`. Rebuild both packages.
- Keep `UnavailableVision` until a real adapter exists. Never turn
  `SearchObjects`/`FindObjects` into "currently visible".
- Add separate memory atomics for the object reads and the log, with `ObjectQuery`'s
  filters as arguments. Strip `vectors`, clouds and poses from LLM-visible output.
- Database ids are decimal strings of int64 ids. A live `obj_` reference is never
  cast to one. Sizes and weight only when measured.
- Writes: `RememberEvent` from the event atomic. Never claim a detection was
  remembered while the writes answer `NOT_READY`. Put `remember_event`,
  `find_events` and the five object writes on the never-list.
- Update the agent's docs where the database's harvest page links to its headings
  `#relaunch-per-episode`, `#isolation` and `#localization`: keep those three
  headings. The database pages link to agent commit `91ffe25` for
  `self-evolving-agent.md`, `access-levels.md`, `competition.md#web-search` and
  `manipulation.md` until replacement pages exist.

### walkie-ros (owner's follow-up, not done)

- Pin and build `walkie_vision_msgs` at the tested commit wherever the arrays are
  used, with `vision_msgs` declared. Rebuild every subscriber and client of the edited
  `ObjectDetection` and `ObjectDetectionArray`.
- Use `walkie_map_id.py` and its golden fixtures in map and baseline preparation. Keep
  its file hash in the deployment manifest. Use `GetStoreStatus` as the single map
  check.
- Supply registered RGB-D with capture stamps, exact-stamp TF, a consistent `/clock`
  in the sim, and the chosen camera topic and frame profile.
- Define the raw baseline grid and proposal encoding before M3c, and emit the frozen
  seed and anchor file schema.
- Move localize callers to `LocalizeObjects`, keeping each entry's status. Keep
  `/get_3d_poses` until every old caller and GraspNet have moved and a recorded or
  robot comparison passes.
- For later memory reads, events or a scene bridge, build `walkie_database_msgs` at
  the tested database commit. Keep `walkie_database/<id>` collision names.

### Plan and documentation

- The Robot Memory plan may mark M0 "contract-ready" now that the pair report exists.
  M1, M2, M4 and M5 stay open.
- walkie-vision's links into withdrawn agent pages were handled in V8.

## After the review fixes, 2026-11-05

The sections above describe the pair as built. The review of both branches found
D-F1 to D-F4 and V-F1 to V-F5, and the fixes moved both final commits. The
agent's decision log records each fix and its cost as P32 to P41.

| Repository | Worktree | Commit | Subject |
|---|---|---|---|
| walkie-vision | `/var/tmp/wv-prep` | `60721b63733991e0f8a78b56c6fdf5987979ea02` | fix: take a DetectObjects pair only while it arrived within the request's age |
| walkie-database | `/var/tmp/wd-prep` | `9ad2d36bb299aa8246aa8c176dffd32086efe4d5` | build: pin walkie-vision at its review fixes |

- `ros2_ws.repos` now pins walkie-vision at `60721b6`, and the imported clone is
  at that commit. `walkie_vision_msgs` did not change between `00a4fed` and
  `60721b6`.
- Neither `pixi.lock` changed, so the lock hashes above still hold. walkie-database's
  `pixi.toml` gained a two-line comment on `--id-floor`.
- On the final commits, walkie-vision's suite gave 1208 passed and 1 skipped (the
  missing `buffalo_l` model), and walkie-database's gave 1636 passed. Both strict
  docs builds passed.
- The pair gate ran once at these commits with `--build` on domain 89 and passed.
  Its report is `/var/tmp/wa-impl-prep/pair-gate-60721b6/report.json`, and its
  summary is kept beside this file as `pair-gate-60721b6.log`.
- Observation QoS is settled. Both observation publishers, the real node and the
  fake, use RELIABLE, VOLATILE, KEEP_LAST 5, and a vision test holds the fake's
  profile equal to the node's `OUTPUT_QOS`. The database's observation
  subscription stays BEST_EFFORT, VOLATILE, KEEP_LAST 1. A RELIABLE publisher
  matches both kinds of subscriber, and the gate's observation checks passed over
  this pairing. Brief line 145 says the same since agent commit `6418405`.
- For walkie-agent-ros, the follow-up above now means walkie-vision `60721b6` and
  walkie-database `9ad2d36`, not `00a4fed` and `704eb24`.
