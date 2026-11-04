# Vision and database preparation brief

Evidence date: 2026-11-05. This is a build specification, not an implementation report. No source repository was changed, built, installed, launched, or pushed during this research. Scratch clones are under `scratch/`. No model, GPU workload, router, robot node, simulator, or audio was started.

## 1. Scope and definition of ready

**Prepare the seams and the useful services; do not implement the scene graph.** The deliverable is eight vision slices and nine database slices below. They provide frozen M0–M5 interfaces, camera-frame observations, local geometry conversion, persistent schema, safe seed/restore tools, schema-backed reads, and a GPU-free cross-environment contract test. The later builder supplies fusion, belief, relations derived from observations, and skill-state behavior behind those interfaces.

I disagree with a literal promise of “no further interface, environment or contract work” for *all later milestones*. M7's embedder bake-off can require a new vector migration; a second image fingerprint needs a versioned interface; GraspNet needs separate dependencies and interfaces; optional captions still need a later runtime/model decision and dependency check despite the reserved type; baseline formats and localization validity belong to walkie-ros and have no implemented contract yet. Preparation can freeze the selected single-embedder M0–M5 seam. It cannot truthfully complete M1, migration step 1's persistent ingestion, or the M2/M4/M5 exit tests without the excluded core and consumers. Contract availability is different from feature availability.

The one early exception to the excluded geometry work is a **stateless database LocalizeObjects service**. It validates supplied camera-frame clouds, transforms at the observation stamp, and returns their centroids. It neither fits persistent boxes nor associates, fuses, or writes objects. This replaces `/get_3d_poses` without adding TF to vision or making the agent implement geometry. M1 reuses the same transform helper.

The “arena reset service” lowers movable-object belief between GPSR commands. It does **not** restore a database file. A new test restores `seed.db` offline with the node stopped. These are different requirements (memory plan :1210, :1538–1555).

| Repository is ready when | Plan requirement | Later work unblocked | Proof in this preparation |
|---|---|---|---|
| Vision publishes valid stamped, bounded camera-frame clouds, sampled crops and CLIP vectors; no map knowledge | :939–944, :1010–1025, :1057–1087, :1110–1123 | M1 synthetic input, M2 real input, migration 1 | Analytic RGB-D fixtures, fake providers, serialization and wire exchange |
| Vision serves EmbedText, DetectObjects and EmbedCrop with bounded failure behavior | :960–965, :1181, :1579–1582 | M3c, M4, M5 | Fake models exercise actual node/service paths |
| DescribeCrop has its final type and an explicit disabled answer | :1234–1241, :1383–1384, :1581; cut 7 at :179 | M4 can add captions without a message bump | Type/response tests; no caption-quality claim |
| Vision has a standalone synthetic observation publisher and distinct sim endpoints | :2075–2077, :1123 | M0 and M1 without World/GPU; World adapter later | Fixed scene, fake camera/TF and restart tests |
| Database builds the pinned vision interfaces, validates observations, and serves stateless localization | :1110, :1576, :2496–2497 | M1 and migration 2; agent localize adapter | Separate environments exchange a point-bearing array and LocalizeObjects call |
| Database migrations cover identities, lifecycle fields, filtered vectors, rooms/surfaces/catalogue, relations and events | :1447–1474, :1494–1519, :1641–1642 | M1–M5 storage | Migration from v3, rollback, index/cascade and filter tests |
| Map hash, gate, seeds and offline restore are complete | :1521–1555, :2075–2076, :2155 | M1; M3a seed-file producer; later preflight | Golden hash fixtures, mismatch refusal, closed-file restore and lock tests |
| Schema-backed reads and event writes work before fusion | :1591, :1634–1660 | M4 agent work and fixture demos | Seeded database service tests; no fabricated live sightings |
| Mutation types needed by later core exist and refuse explicitly until implemented | :1203–1210, :1232, :1267, :1335–1343, :1583–1592 | M3/M5 can implement handlers without rebuilding consumers | NOT_READY contract tests, never “accepted” while discarding input |
| Independent builds agree on types and pass real wire traffic on Fast DDS | :2075, :1592, :1121 | All consumers | Local pair-build report with source SHAs, hash inventory, import paths and messages |

**Evidence notation.** `V:path:line` means walkie-vision **0a49428d4b92e2f7c3ba75e621d31d8d87679774** (main and origin/main). `D:path:line` means walkie-database **64403e982e253db0ca00c372249eeb798d89d1f6** (main and origin/main). `A:path:line` means the agent prototype **15bbb6a9a6cae887296fd1e567b9bae415a577eb**, the requested snapshot of feat/mahoraga-demo. `R:path:line` means walkie-ros origin/main **fcbac5cc5bdc8e4d5ddcb86db926a8a9359fc299**, read only with Git. `W:path:line` means its origin/feat/chula-pat14-6-simulation **adb9725e2a97ce24d4f17266d0b314940b7439df**, also read only with Git. Paths are relative to their repositories. Bare Python filenames after V/D abbreviate `src/walkie_vision/` or `src/walkie_database/`; bare A `capabilities/`, `atomics/` and `tools/` abbreviate `src/walkie_agent/`. Bare V/D `msg/` and `srv/` mean their packages under `ros2_ws/src/`; bare numbered SQL filenames mean database `src/walkie_database/migrations/`. Bare ADR numbers mean the uniquely numbered file under `docs/architecture/decisions/`. `P:n` means `/var/tmp/wa-impl-prep/db-vision/memory_plan.txt:n`; `I:n` means `moveit_plan.txt:n` in that directory; `M:n` means `/var/tmp/wa-plans-out/plan.md:n`. These plan copies are not Git commits. Proposed paths and behavior below are requirements, not claims about existing code. Negative implementation claims were checked against the tracked-file inventory and the node's complete endpoint setup.

## 2. Verified state, including stale claims

### Owned or shared plan items through M5

“Deferred” is a scope decision, not evidence that an item exists. M3a and M3b have no implementation owned here; their shared inputs are listed explicitly.

| Milestone / step | Plan claim and requirement | Code at the named commit | Preparation disposition |
|---|---|---|---|
| M0 message freeze | P:2075; add view_quality and agree fields once (P:1111–1121) | V:ros2_ws/src/walkie_vision_msgs/msg/ObjectDetection.msg:15–28 has optional vector/cloud/crop, no quality. Array has header/info/classes/detections (:7–14). D:ros2_ws/src/walkie_database_msgs/CMakeLists.txt:9–21 lists only legacy objects and people services. | V1 + D1 add final types; old database types unchanged |
| M0 fake publisher | P:2075–2076, world-known boxes | V:src/walkie_vision/fakes.py supplies provider fakes, not a standalone ROS scene publisher; V:tests/test_node.py:279 runs injected providers. A:capabilities/fake.py:95 and :289 is a ROS-free world, not an observation publisher. | V2 owns publisher and scene fixture format |
| M0 map_id helper | P:2075, :1542–1544 | No helper in either tracked inventory. D:src/walkie_database/node.py:84–100 accepts a string and only warns if the map is absent. | D2 shared stdlib module, D6 startup gate |
| M0 migrations | P:1447, :1642 | D:migrations/0001_initial.sql:63–128 has no state/belief/layer; vec metadata only map_id/room_id. D:migrations/0003_one_open_session_and_unique_ids.sql:41–56 covers session/person only. Here and below `D:migrations/` abbreviates `D:src/walkie_database/migrations/`. | D3–D5 |
| M0 pair CI | P:2075, :1592 | V:.github/workflows/docs.yml:24–37 builds docs only; no test workflow. D:.github/workflows/test.yml:34–38 builds only its own messages and runs own tests. Neither checks a pinned pair. | D9 harness plus V8/D9 CI documentation; local run is required for unpushed SHAs |
| M1 4.1 pose / guards | P:941, :945–946, :2076 | D:src/walkie_database/node.py:81–116 creates no subscriptions or TF buffer. | D7 supplies validated ingress and exact-stamp transform helper only. Motion/localization/jump guards belong to M1/M7, with frozen input assumptions below |
| 4.1 pairing | P:940 says no depth yet | Correct: V:src/walkie_vision/node.py:625–626 subscribes only Image and CameraInfo. V:src/walkie_vision/pipeline.py:214 accepts no depth argument. | V3 |
| 4.2 detector / 4.2a tracks | P:953, :970–971 | Detector/tracker exist: V:src/walkie_vision/yolo.py:175–267. **Stale:** published IDs are startup-time offset IDs, not raw IDs restarted at one: V:src/walkie_vision/pipeline.py:43–67, :277. Wrap/clock rollback still prevents treating them as persistent IDs. | Preserve offset, add explicit producer stream id; no identity inference |
| 4.3 first floor | P:978 | Correct: V:src/walkie_vision/pipeline.py:227–232 applies configured floor; V:config/vision.yaml:39 is 0.25. | Preserve; fake boundary tests |
| 4.3 second floor / 4.4 routing | P:979–987 | Not built in D node endpoint/setup (:81–116). | M1, not prep; freeze config vocabulary and class-table input |
| 4.4a people / 4.4b self-removal | P:988–1009 | Person boxes/keypoints exist, no mask/depth: V:ros2_ws/src/walkie_vision_msgs/msg/PersonDetection.msg:5–39. D has neither subscription nor robot geometry loader. | Consumer dependency now; occlusion/self-filter algorithms later M3/M7 |
| 4.5–4.6 mask/depth/lift | P:1010–1025 | V:src/walkie_vision/yolo.py:52–85 discards masks; detect (:260–267) never reads r.masks. Pipeline (:276–279) supplies no points. | V3/V4; masks must be retained *inside* provider output first |
| M1 4.7–4.9 map transform, stripping, DBSCAN, boxes | P:1026–1055 | D:docs/architecture/object-memory.md:3–6 explicitly proposed. No implementing module. Its point-overlap matcher (:84–96) differs from P:1178–1180. | Core deferred. Stateless centroid localization is the sole early subset |
| 4.10 quality / best views | P:1057–1059 | V ObjectDetection lacks quality; pipeline has no object-crop path. D:object_observation in migration 0001:116–128 has crop_path only. | V4 scores; D3 stores view metadata; best-view retention algorithm deferred to M1 |
| M2 4.11 crops | P:1061–1070 | Optional crop exists; converter accepts it (V:src/walkie_vision/convert.py:97–115); pipeline never supplies it (:276–279). Person crops already exist and are a separate service (V:node.py:633). | V5 object sampler; keep GetPersonCrop contract unchanged |
| M2 4.13 image fingerprint | P:1080–1087, :2077 | Embedder protocol and fakes exist. Runtime Providers has objects/pose/faces/reid only (V:pipeline.py:70–77). CLIP bake-off code exists (V:tools/bakeoff/embedders.py:109–144), not runtime wiring. | V5, CLIP 512, no SigLIP migration |
| 4.12 second fingerprint | P:1073–1078 | Not in runtime; bake-off candidates do not constitute deployed models. | Deferred until measured win; new version if needed |
| M1 5.1–5.3 / 5.8 | P:1174–1210, :1257–1269 | No ingest or lifecycle implementation. D:docs/architecture/object-memory.md:98–142 sketches different fusion/cleanup, no belief or durable tombstones. | Schema and contracts now; algorithms deferred |
| M1 seed + restore + map gate | P:2076 | Existing `db-seed` is a demo-house generator with random vectors and people (D:src/walkie_database/seed.py:6, :26, :191–225, :282). ADR-0016:57–83 proposes seed_sim but no such module exists. | D6 real seed tooling; do not adapt demo.db into a production seed |
| M1 SQLite flush / ownership | P:1273–1280 | D:src/walkie_database/db.py:15–46 loads sqlite-vec, FK/WAL/NORMAL. Node owns a same-thread connection (:89–92); ADR-0005:30–32 mandates single executor. No writer thread/live cache. | Keep single DB owner in prep; add repository boundary, defer threading/live-state swap to M1 and a new ADR |
| M2 sim:=true | P:1123, :2077 | V:config/sim.yaml:8–11 sets sim time, sim object/people topics and GetPersonCrop only. It adds no depth subscriber. Launch (:36–51) applies overrides. **Qualification:** oracle_perception is experiment code, not in A's tracked src tree. | V3/V7 use real render path with depth, V2 for synthetic M0/M1 |
| M3 5.4 and p_now | P:1211–1233, :1370, :2078 | No belief fields, grid input or existence checker in D. FindObjects has no state filter (D:node.py:67–78). | Read-time age/p_now and schema now; evidence/belief evolution later |
| M3a baseline capture products | P:2079 | Entirely walkie-ros. World main lacks this scene; R:bringup/launch/nav_mujoco_omnibot.launch.py:118–127 is static map→odom. | Freeze seed schema and shared hash only; no baseline builder here |
| M3b diff / overlay clear | P:2081 | No producer to integrate. Its reset is std_srvs/Trigger, owned by walkie-ros. | Name/gate requirement recorded; no implementation here |
| M3c baseline files / proposals | P:2082, :1181 | D has no reader; V has no crop embedding request. | EmbedCrop now; seed importer accepts baseline_rev now; baseline runtime deferred |
| M4 EmbedText | P:1087, :1632 | Type and conversion helper exist (V:srv/EmbedText.srv:6–18; V:convert.py:226–254, paths expanded in §3). Node serves only GetPersonCrop (:633). | V5/V6 serve it |
| M4 captions | P:1234–1241 | No DescribeCrop type/server, queue or caption vector table. Caption scalar exists in object schema (D:0001:68). | Type + disabled handler now; select cut 7, defer model/queue/caption vectors |
| M4 5.6 relations | P:1396–1404 | No surfaces/relation table or derivation. | Tables and queries over seeded relations now; derivation later |
| M4 7.3–7.7 search | P:1634–1660 | Only FindObjects exact ASCII-case-insensitive label and most-recent sort (D:node.py:67–78). No semantic/attribute/relation/state search service. | D8 actual seeded-schema query services and strict D4 filter schema |
| M4 rooms | P:1521–1527, :1676 | Rooms/geopoly exist (D:0001:36–51), no read services. | D5 schema additions, D6 seed, D8 reads |
| M5 write states / Held / merge resolution | P:1335–1343, :2084 | No object-state services; RememberedObject has no state (D:msg/RememberedObject.msg:1–28). | New types; get_object follows stored links; mutation handlers NOT_READY until core/skill integration |
| M5 DetectObjects | P:964, :1125, :2084 | ADR-0009:53–85 is a proposal, with separate model and caller-only array; no srv file. | V6/V7 implement separate model; forwarding contract D1/D7 |
| M5 report_sighting and arena reset | P:1232, :1210, :1583 | Neither type nor service exists. | Define now; honest NOT_READY runtime until M3/M5 |
| M5 remember_event / find_events | P:1513–1520, :1690 | Conversations/messages are session data (D:0001:185–214), not events. | D5 event schema and D8 working event service; task/cause fields and privacy deletion |
| M5 updates/snapshot | P:1267–1269, :1574–1588 | No object publishers (D:node.py:81–116). ADR-0013:15–21 proposes standard marker/cloud/MoveIt outputs. | ObjectUpdates and snapshot schema now; D8 seeded snapshot/read support; dynamic publications later |
| M1–M5 6.2 durability / backups | P:1485–1493 | D:db.py:32–43 uses WAL/NORMAL, with no object live cache to flush; existing migrate.py:56–85 is transactional. | Preserve settings for ordinary writes; D3 durably reserves object IDs, D6 checkpoints/backups/restores; 5 s core flush and crash-loss measurement deferred |
| M1–M5 6.3 IDs | P:1494–1502 | D:0001:63–64 uses ordinary INTEGER PRIMARY KEY; D:migrations/0003_one_open_session_and_unique_ids.sql:41–56 restricts last_id to session/person. | D3 rebuilds counter CHECK, adds durable object reservations and tombstone fields; get_object follows stored links now, merging later |
| M2/M4 6.4 vectors | P:1503–1512 | D:0001:107–112 holds CLIP 512 with map/room metadata; V runtime does not supply object vectors. ADR-0008 is proposed, not a selected SigLIP deployment. | V5 CLIP; D4 state/layer metadata; reject unsupported model/fake production vectors; caption/second-model tables deferred |
| M1–M5 6.7 clocks | P:1531–1537 | D:node.py:165–166,:225 uses time.time even for people; migration 0001:3–5 declares Unix timestamps. No store clock_kind. | D3/D6 store clock_kind and injected node time; sim seed per episode; never mix clocks |
| M1–M5 6.8 consistency / reset | P:1538–1555 | D:node.py:84–100 accepts a map_id and warns, but computes no hash or baseline/clock gate. No restore utility in tracked tree. | D2/D6 enforce full hash and restore; online belief reset remains M5; baseline_rev must match supplied manifest when present |
| M1 debug markers/cloud | P:1586–1587 | D:docs/architecture/decisions/0013-publish-boxes-to-rviz-and-moveit.md:15–31 proposes them; D:node.py:81–116 has no publishers. | Standard types need no new custom contract. Seeded JSON snapshot now; marker/cloud publication waits for core-owned geometry |
| Parked 5.7 | P:1245–1254 | No RTAB-Map graph consumer in D tracked source; D object-memory proposal predates fixed-AMCL choice. | Nullable anchor fields only; no pose-graph dependency or implementation |
| Migration 1 | P:2496: points filled and consumed | Both missing as above. | Vision completion + DB ingress validator now; persistent consumption only when M1 lands. Do not declare migration 1 fully done |
| Migration 2 | P:2497: replace /get_3d_poses | R:perception/srv/GetObPose.srv:2–7 takes Detection2DArray and returns PoseArray + bool. R:perception/src/ob_pose_service.cpp:77–124 samples *latest* depth, stamps output now and omits failed entries, losing positional correspondence. | D7 replacement returns one indexed result per detection from its supplied points and original stamp |

Path abbreviations in this table: bare `V:node.py`, `V:pipeline.py`, `V:convert.py` refer to `src/walkie_vision/`; `D:node.py` refers to `src/walkie_database/node.py`; `V:srv/`, `D:msg/` refer to their respective `ros2_ws/src/*_msgs/` directories. `D:0001` means the full migration-0001 path cited above.

### Pins and consumers

These are Git source pins, not a claim about what the owner's live robot runs. No live node was probed.

| Consumer snapshot / branch | Vision pin | Database pin | Evidence |
|---|---|---|---|
| Agent main/origin/main 91ffe25cc411700e2fac33adbbd815998a3296c9 | None | 7f657b0960944c4d4399ef19f5032b3c5d726f9d | `ros2_ws.repos:4–19`, `pixi.toml:112–116` at that commit |
| Agent feat/mahoraga-demo at requested A=15bbb6a | None | 7f657b0960944c4d4399ef19f5032b3c5d726f9d | A:ros2_ws.repos:4–19; A:pixi.toml:112–116 |
| Agent experiment/hri-agent 61d30e8db701ccb080c3fb57df4abb1f5c5ac601 | 1576d6f979e9513c9d2e9b752232341f01a5bc0a | 64403e982e253db0ca00c372249eeb798d89d1f6 | `ros2_ws.repos:20–33` at that commit; also pins walkie-ros 6b26daa at :47–50 |
| Agent port/02-vision-msgs-pin 38eeeb989a47305a6f088e797bae0edeb75d4edc | 1576d6f979e9513c9d2e9b752232341f01a5bc0a | 7f657b0960944c4d4399ef19f5032b3c5d726f9d | `ros2_ws.repos:16–28`, `pixi.toml:114–119` at that commit; not main |
| Database D main | None | Own D commit | D:pixi.toml:73–76; no ros2_ws.repos or vision dependency in tracked tree/CMake |
| Vision V main | Own V commit | None; intentionally no runtime DB import | V:pixi.toml:136–139 and V:ros2_ws/src/walkie_vision_msgs/CMakeLists.txt:6–17 |
| walkie-ros R main, W Chula branch | Neither repository pinned | Neither repository pinned | Git inventories and no dependency matches; R:walkie_real.repos:1–17 imports ZED only. Neither has walkie_vision_msgs in its tracked build inputs |

The branch name `feat/mahoraga-demo` moved during this research. All A evidence remains anchored to **15bbb6a**, not the moving branch head. No conclusion uses `/var/tmp/wa-demo-fix` uncommitted files.

Other stale or conflicting claims that matter:

- P:882 says only detector/tracker/pose run today. V already constructs OSNet re-identification and SCRFD/ArcFace head-face providers too (V:src/walkie_vision/node.py:520–542; V:pipeline.py:70–77). The object fingerprint is the missing runtime model, not all non-detector vision.
- P:1797 and :1824 require `mem_` handles. M:787–791 and the user's provisional decision 4 select plain database IDs. A:capabilities/records.py:16–27 already exposes string IDs. Use decimal strings for persistent integer IDs; keep live `obj_` references separate.
- P:1628–1629 proposes a separate parser. M:710 and :757 (D4) remove it. The query capability's arguments must express attributes, relation and superlatives directly.
- V:docs/architecture/vision.md:1020–1023 claims the agent pins vision b050ea1. That is not A/main; the experiment now pins 1576d6f.
- D:docs/architecture/harvest.md:320–327 calls the vision node/topic unbuilt and unsettled. They are built. D:docs/architecture/object-memory.md:151 still records `/walkie_database/detections`; replace with the selected input topic.
- ADR-0016's “no objects in seed” (:63–68) conflicts with P:1550's keyed L2 anchors. Its proposed world path is also stale: W uses `simulation/mujoco/worlds/chula_pat14_6/authoring/source/30_16102026_ChulaPat14_6_world.toml`. The adjacent `world.toml:3–16` is a launch manifest, **not room polygons**.
- P:2084/2087's RunSkill/FakeSkillServer is not the provisional lean arm choice (M:758). No such interface is added here. A:capabilities/__init__.py:90–122 has pick/place_on/hand_over and UnavailableArms.
- Existing fetch-models downloads DOWNLOADS only (V:src/walkie_vision/models.py:183–195). Pinning CLIP in HUB_MODELS (:110) does not make `fetch-models` fetch its safetensors. Add an explicit object-embedder fetch/offline check.
- The plan's timing numbers are measurements reported by documents, not measurements repeated in this research. V:docs/architecture/vision.md:312 agrees with 86.6/108.2 ms. No added-stage latency or combined GPU total is verified.

### Full Body MoveIt dependencies

I:124, :134 and :248–250 need a map-fixed planning frame and a depth_image_proc cloud in walkie-ros. They do not need object fusion to start. R:manipulate/openarm_bimanual_moveit_config/config/openarm_bimanual.srdf:29 still has a fixed `world`→`base_footprint` virtual joint; sensors_3d.yaml:15–22 consumes the filtered head cloud. Do not send per-object clouds as the whole-scene octomap input.

Database's later M7a duty is confirmed-object PlanningScene diffs with IDs `walkie_database/<integer>`, suppressing grasping/held targets and removing merged losers (D:docs/architecture/decisions/0013-publish-boxes-to-rviz-and-moveit.md:15–31; P:1269, :1335–1338). Prep preserves map-frame box/state/ID contracts but does not add moveit_msgs or publish collision objects. GraspNet remains in walkie-ros, with its existing live-cloud request semantics (R:perception/srv/GraspPos.srv:2–15). Its move is migration 3, not preparation.

The parallel `/var/tmp/wa-impl-prep/ros/brief.md` was read. Its §3/§8 deliberately selects `UnavailableVision` and retires the old remembered-object tool for that cut. Keep that honest behavior until a separate live adapter is built. Its open DB-pin decision is resolved **for this preparation** by D-commit selection below; it does not authorize changing that parallel agent worktree.

## 3. The contract, once

### Rules shared by all new interfaces

All text below is the proposed file content. Paths in headings are relative to `ros2_ws/src/<owning_package>/`. Add every new type to CMakeLists.txt and package.xml dependencies in the first contract slice. Add field/default pins and serialization tests at the same time. A comment-only edit does not change the structural type; **field edits and nested field edits do**. Rebuilding a package is required for newly added types even when its old types keep their hashes.

| Type / endpoint | Define now? Runtime in prep? | Existing type impact | Consumers |
|---|---|---|---|
| ObjectDetection, ObjectDetectionArray; `/walkie_vision/objects` | Yes / yes | Edit ObjectDetection, add stream_id to array; both hashes change; all enclosing service hashes follow | Database, agent experiment/oracle, later ROS baseline_diff |
| EmbedText; `/walkie_vision/embed_text` | Yes, unchanged / yes | Preserve existing type exactly | Agent semantic queries, database future caption vectors |
| DetectObjects; `/walkie_vision/detect_objects` | Yes / yes | New, nests edited array | Agent live check/verify |
| EmbedCrop; `/walkie_vision/embed_crop` | Yes / yes | New | Database baseline proposals |
| DescribeCrop; `/walkie_vision/describe_crop` | Yes / disabled response | New; no model loaded | Database future caption queue |
| LocalizeObjects; `/walkie_database/localize_objects` | Yes / yes | New; replaces GetObPose usage, not its type | Agent localize/navigation/grasp backend |
| SubmitDetections; `/walkie_database/submit_detections` | Yes / NOT_READY until M1 | New | Agent forwards on-demand arrays; not vision |
| ObjectUpdates; `/walkie_database/objects/updates` | Yes / serialization fixture only | New | Agent invalidation, future MoveIt/debug consumers |
| ObjectSummary, query/room/event/result support messages | Yes / yes | New; **do not edit RememberedObject or ModelVector** | New database services |
| SearchObjects, GetObject, FindNear, ListRooms, GetRoomContents, FindEvents | Yes / schema-backed | All new | Agent memory capability, preflight read probes |
| RememberEvent | Yes / yes | New | Reviewed agent event atomic |
| SetObjectState, ReportSighting, ArenaReset | Yes / NOT_READY until M1/M3/M5 | All new | Reviewed skill/agent capabilities only |
| DescribeObject | Yes / NOT_READY while captions disabled | New | Agent async description request |
| GetStoreStatus | Yes / yes | New, read only | Preflight and local contract harness |
| Offline seed/reset and map hash | Yes / yes | Python/CLI/file contract, **not ROS reset** | Database, later ROS launcher/baseline builder, agent episode harness |

New database endpoint names are exactly `/walkie_database/` plus `search_objects`, `get_object`, `find_near`, `list_rooms`, `get_room_contents`, `set_object_state`, `report_sighting`, `remember_event`, `find_events`, `describe_object`, `arena_reset`, `submit_detections`, `localize_objects` and `get_store_status`. Use ordinary ROS service QoS. Observation publishers and database subscribers use BEST_EFFORT/VOLATILE/KEEP_LAST 1. ObjectUpdates uses RELIABLE/VOLATILE/KEEP_LAST 100; clients detect a sequence gap or changed store_id and refresh with reads. Snapshot uses RELIABLE/TRANSIENT_LOCAL/KEEP_LAST 1 so a late baseline overlay gets seeded boxes. These QoS choices apply to both fake and real paths.

Keep people messages, GetPersonCrop and the five people/session service endpoints unchanged; the latter use four existing service types (MatchPeople serves two endpoints). In sim, all four **new vision service names** use `/walkie_vision/sim/…`; GetPersonCrop already does. Database service names stay `/walkie_database/…`; a separate domain/node instance selects its input and map. No automatic fallback between sim and live endpoints.

All optional ROS fields use a documented empty value or an explicit `has_*` boolean. No NaN sentinel. Validate finite scalars, bounds, image sizes and vector lengths at each server boundary. Fail the whole request on malformed data. Empty successful search/detection is distinct from an unavailable backend. New write requests require a nonempty `request_id`, `task_id` and `cause`. Repeating a request ID with the identical payload returns the recorded outcome; a different payload is CONFLICT. Persist that ledger in the same transaction as a successful write. No retained rows for rejected NOT_READY requests.

#### Vision: `msg/ObjectDetection.msg` (edit)

```text
# One observation. Pixel coordinates use the containing array's camera_info.
# No map pose, persistent object ID, segmentation mask or person record here.
string label                       # One member of ObjectDetectionArray.classes.
float32 score                      # Finite detector confidence in [0,1].
uint32 track_id                    # Hint within stream_id; 0 means untracked.
vision_msgs/BoundingBox2D bbox      # Original box in full-image pixels; theta=0.
# Image and text vectors share the exact weights/preprocessing named here.
# Both empty when not sampled; otherwise finite unit vector (CLIP: 512).
# Crop uses 15% padding PER SIDE, clipped to image, before model preprocessing.
string embedding_model
float32[] embedding
# XYZ float32 little-endian metres, x right/y down/z forward, height=1.
# Header equals array.header. Cap 2000 points. Empty data/width=0 if <30
# valid points or depth unavailable. Never re-stamp or use a map frame.
sensor_msgs/PointCloud2 points
# JPEG quality 90; same header. The padded crop, not just bbox pixels.
# Empty data when not sampled. Longest side <=512; never upscale.
sensor_msgs/CompressedImage crop
# Finite [0,1]; zero when disabled or unusable for best-view selection.
float32 view_quality
# Optional mask-only colour evidence, computed BEFORE JPEG/padding/resizing.
# Empty when not sampled or <50 valid mask pixels; otherwise 12 shares
# summing to 1: black,white,grey,red,orange,yellow,green,blue,purple,pink,brown,beige.
# This freezes the colour seam while masks still exist; no mask is transmitted.
float32[] color_shares
```

#### Vision: `msg/ObjectDetectionArray.msg` (edit)

```text
# One complete successful detector result, including a successful empty frame.
# No publication for an inference failure. classes is the searched vocabulary.
std_msgs/Header header             # Colour image capture time and optical frame.
sensor_msgs/CameraInfo camera_info # Rectified calibration for bbox/points pixels.
string[] classes                   # Ordered unique nonempty class names.
ObjectDetection[] detections
# UUID generated once at publisher-process startup, including fake publisher.
# Identity key for a track is (source topic, stream_id, kind, track_id).
# Not a frame handle and cannot retrieve an old image. Empty is invalid.
string stream_id
```

Points use fields x/y/z at offsets 0/4/8, FLOAT32 count 1, point_step 12, row_step `12*width`, is_bigendian false, is_dense true. Empty clouds still carry these fields and the header. Array, cloud and crop stamps must agree exactly. `camera_info.header` is copied to that capture stamp only after calibration/frame/size validation; this does not authorize pairing unrelated frames. On-demand arrays share the node's stream_id and set every track_id to zero. Negative evidence for them is restricted to ReportSighting's named object; forwarding an array never means “all remembered objects were absent.”

Adding stream_id is deliberate. Startup offset IDs normally survive a quick restart, but are not a durable run identifier and wrap in about 49.7 days (V:pipeline.py:43–67). A UUID costs one small field now and avoids another later message change. `PersonDetectionArray` remains unchanged; pair people by exact frame/stamp, and clear person occluder caches on stream change or backwards ROS time.

#### Vision: `srv/EmbedText.srv` (keep exact fields)

```text
# At least one text; runtime caps batch at 16 and each text at 512 characters.
string[] texts
---
bool success                       # False leaves model/dim/vectors empty or zero.
string message                     # Empty on success; bounded reason on failure.
string embedding_model             # clip_vit_b16 initially.
uint32 dim                         # 512 initially.
# Row-major [len(texts),dim], finite unit rows, input order preserved.
float32[] vectors
```

Do not rename `vectors`. Existing hashes stay stable. EmbedText has a fixed 2 s monotonic server deadline because its existing request has no timeout field; the caller allows 5 s. Refuse blank texts and enforce the batch/length caps before admission. A future agent adapter must explicitly strip this response field from LLM-visible output. V:docs/architecture/vision.md:1041 and :1111–1115 describes a suffix-based filter that would miss it, but no such filter is implemented in A's tracked `src/` tree. Do not treat that cross-repository doc claim as verification of A's runtime.

#### Vision: `srv/DetectObjects.srv` (new)

```text
string[] classes                   # 1..32 unique nouns, each 1..80 characters.
# Reject images older than this in ROS time; finite (0,1], default 0.3 s.
float32 max_image_age_s 0.3
# Wall-clock deadline including admission, re-prompting and inference; (0,5].
float32 timeout_s 4.0
---
bool success                       # True includes a successful zero-detection array.
# Empty on success. INVALID, NO_IMAGE, STALE_IMAGE, BUSY, TIMEOUT,
# DISABLED, MODEL_UNAVAILABLE or INFERENCE_FAILED otherwise.
string error_code
string message
ObjectDetectionArray objects       # Empty default on failure; image header on success.
```

The default 4 s server deadline leaves margin inside the agent's 5 s result wait. New-class work is admitted at most once every 2 s, one pending on-demand request, no unbounded queue. Same-list requests skip set_classes. Never mutate streaming classes or trackers. DetectObjects embeds every returned usable crop, rather than applying the streaming 1-in-10 sampler. Limit result processing to the highest 20 scores; declare that cap in node docs. Capture a fresh RGB-D bundle **after** expensive class preparation, so a 1.5 s re-prompt does not make the chosen image stale before inference begins. Check age again before success; do not label stale data current. A timeout is not preemption of a CUDA kernel. Discard a late answer, record the stall, and resume streaming when the worker returns.

#### Vision: `srv/EmbedCrop.srv` (new; unnamed request resolved)

```text
# Pixel box projected by database from a proposal into this specific image.
std_msgs/Header header
uint32 image_width                 # Must equal the current RGB width.
uint32 image_height                # Must equal the current RGB height.
vision_msgs/BoundingBox2D bbox      # Unpadded, theta=0, positive area.
float32 timeout_s 1.0              # Wall deadline, finite (0,5].
---
bool success
# INVALID, NO_IMAGE, STALE_IMAGE, BUSY, TIMEOUT, DISABLED, MODEL_UNAVAILABLE,
# INFERENCE_FAILED; empty on success.
string error_code
string message
std_msgs/Header header             # Actual source image, never request time copied blindly.
sensor_msgs/CompressedImage crop   # Same padding/JPEG convention as ObjectDetection.
string embedding_model
float32[] embedding                # One finite unit vector.
float32 view_quality               # Same scoring; depth term zero when unavailable.
```

For EmbedCrop quality, use the requested rectangle as an approximate mask; there is no segmentation mask for a proposal. Use the same four-term formula, with zero depth coverage if unpaired. The caller knows this was proposal-derived and must not treat the score as segmentation confidence.

The input buffer has only the newest colour/depth pair, with no historical image cache. Require exact optical frame and image dimensions, stamp difference <=5 ms, current image age <=0.3 s, and no future request >5 ms. Otherwise STALE_IMAGE; never crop a later view using an old box. A refused proposal remains unembedded and retries only on a new proposal. This strict default can reject most proposals under load; measure that later, then change the sampling architecture rather than silently widening spatial misregistration.

#### Vision: `srv/DescribeCrop.srv` (new; inference deferred)

```text
sensor_msgs/CompressedImage crop   # JPEG, <=512 on longest side, <=512 KiB compressed.
# Optional source mask colour evidence in the fixed 12-colour order above.
float32[] color_shares
float32 timeout_s 4.0              # Wall deadline, finite (0,5].
---
bool success
string error_code                  # DISABLED in prep; later INVALID/BUSY/TIMEOUT/INFERENCE_FAILED.
string message
string caption                     # <=512 characters; untrusted camera-derived text.
# JSON object with allowed keys colour,material,shape,size,fragile,container.
# Values: validated enums, boolean for last two; absent means unknown.
# No extra keys, free-text values, NaN, lists or nested objects.
string attributes_json
string printed_text                # <=256 characters; untrusted, kept separate.
string caption_model               # Exact pinned weights ID; empty when disabled.
string prompt_version              # Fixed implementation version, not caller-supplied prompt.
```

Colour enum is the fixed colour list above. Material: metal, plastic, glass, ceramic, wood, paper, cardboard, fabric, rubber. Shape: round, rectangular, cylindrical, irregular. Size: small, medium, large. Unknown is omission, never guessed. The server receives a crop supplied by the database, so it needs no past image cache. Defer Florence-2, attributes beyond colour, OCR, caption vectors and queue scheduling. Keep the endpoint as a bounded DISABLED response; GetStoreStatus advertises captions=false. Future prompts are implementation constants, not agent text.

### Database support messages (all new)

#### `msg/OperationStatus.msg`

```text
uint8 OK=0
uint8 INVALID=1
uint8 NOT_FOUND=2
uint8 STALE=3
uint8 TF_UNAVAILABLE=4
uint8 WRONG_MAP=5
uint8 UNSUPPORTED=6
uint8 BUSY=7
uint8 NOT_READY=8
uint8 INTERNAL=9
uint8 CONFLICT=10
uint8 code                         # Always assigned explicitly by the server.
string message                     # Human-readable reason, <=512 characters.
```

#### `msg/ObjectSummary.msg`

```text
int64 id                           # Positive object.id; never a track_id or obj_ handle.
string map_id
uint64 revision                    # Increment on any committed semantic/state/geometry write.
string key                         # Stable per-map anchor key, else empty.
string label
string[] labels                    # All observed/seeded aliases, sorted unique.
string official_name               # Empty when not on setup-day catalogue.
string category
string layer                       # anchor or movable.
string state                       # candidate,present,grasping,held,missing,handed_over,discarded,merged.
string source                      # perception,map,manual.
int64 room_id                      # 0 means unknown.
string room
string place                       # Anchor/surface key from stored on/inside relation, else empty.
Fact[] attributes                  # Validated fixed vocabulary; missing key means unknown.
string caption                     # <=512 characters; always untrusted to an LLM.
string printed_text                # <=256 characters; always untrusted to an LLM.
string caption_model
string prompt_version
float32 p_exist                    # Stored [0.05,0.95].
float32 p_now                      # Read-time decay; [0,0.95].
bool has_last_verified
builtin_interfaces/Time last_verified
float64 last_verified_age_s        # 0 if unknown; use has_last_verified, never infer freshness.
builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen
uint32 n_detections
bool geometry_valid                # False means box dimensions/pose cannot guide motion.
geometry_msgs/PoseStamped pose     # Map frame, last geometry stamp, internal client use only.
geometry_msgs/Vector3 size          # Full upright box dimensions in metres.
bool has_weight
float32 weight_g                   # Measured catalogue value only, not image inference.
bool in_baseline
bool depth_seen
int64 merged_into                  # 0 unless tombstone points to winner.
int64 split_from                   # 0 unless created by a split.
```

`geometry_valid` is required because the current SQL geometry columns are NOT NULL, while no-depth candidates must exist later. Store zeros only with geometry_valid=false; exclude those rows from object_box. For the agent record, size_cm is `100*max(size)`, thickness_cm `100*min(size)` only when geometry_valid; weight is null when has_weight=false. These are explicitly estimated box measures, unlike FakeRobot's invented ground truth (A:capabilities/fake.py:60–75). ROS clients need pose and cloud for navigation/manipulation; atomic outputs must strip them. P:1674 describes the tool output, not a prohibition on internal ROS geometry.

#### `msg/ObjectQuery.msg`

```text
string map_id                      # Empty selects node map; a different map is WRONG_MAP.
string room                        # Exact canonical room name, empty means any.
string[] states                    # Empty defaults to present,held.
string layer                       # Empty, anchor or movable (maps movable? in query_objects).
string thing                       # Label/alias/official name; empty means any.
string category                    # Canonical category, empty means any.
Fact[] attributes                  # AND of requested fixed-vocabulary values; unknown is distinct.
string relation                    # Empty,on,inside,near,in_room.
int64 reference_id                 # Related anchor/object; 0 if reference_key/room is used.
string reference_key               # Anchor or surface key; mutually exclusive with reference_id.
float32 near_distance_m 0.6         # Finite >0; used for near only.
# Empty,largest,smallest,heaviest,lightest,thickest,thinnest,nearest,farthest,leftmost,rightmost.
string superlative
# For directional/distance ranking: server looks up this allow-listed frame
# in map at its query time. Default base_footprint. No caller-supplied pose.
string reference_frame "base_footprint"
```

This is D4's schema. The agent exposes structured query arguments directly; it does not add a separate LLM parser. `in_room` uses room and forbids an object reference. on/inside require exactly one reference. near uses one reference or, if neither, the server's reference_frame. `reference_frame` is validated against deployment frames, not arbitrary text. Superlatives rank after the hard filters; unknown size/weight is reported separately, never guessed or treated as zero. Leftmost/rightmost use y in the reference frame (positive y is left). No raw position argument is exposed to the LLM.

#### `msg/ObjectMatch.msg`

```text
ObjectSummary object
float32 score                      # Bounded [0,1]; meaningful only within this response.
bool has_similarity
float32 similarity                 # Cosine [-1,1]; 0 if absent.
string[] matched_attributes
string[] unknown_attributes
string[] mismatched_attributes
bool measure_unknown               # Requested superlative cannot measure this object.
```

#### `msg/SearchPrior.msg`

```text
string official_name
string category
string place                       # Catalogue default location; NOT an observation or object ID.
string room
```

#### `msg/Surface.msg`

```text
string key                         # Unique per map, e.g. shelf_2.
int64 anchor_id
string map_id
float64 z                          # Horizontal support plane height in map, metres.
geometry_msgs/Polygon outline      # Map XY, all point.z equals z; no repeated final vertex on wire.
bool geometry_valid
```

#### `msg/RoomSummary.msg`

```text
int64 id
string map_id
string name                        # Canonical display name shared with agent places.
string description
int32 floor
geometry_msgs/Polygon outline      # Map XY, z=0.
bool has_stand_pose
geometry_msgs/PoseStamped stand_pose # Map frame; internal navigation use, not free LLM coordinates.
```

#### `msg/ObjectIdChange.msg`

```text
int64 from_id                      # Loser for merge; retained parent for split.
int64 to_id                        # Winner for merge; newly allocated child for split.
```

#### `msg/ObjectUpdates.msg`

```text
std_msgs/Header header             # Commit time in deployment clock, frame_id=map frame.
string map_id
string store_id                    # UUID of this run/store incarnation; changes on seed restore.
uint64 sequence                    # Monotonic within store_id; a gap requires re-reading snapshot.
int64[] added
int64[] updated
int64[] removed                    # Removed from active scene; tombstone still resolves.
ObjectIdChange[] merged
ObjectIdChange[] split
```

Only publish after commit. A split publishes both the retained parent's update and the new child's add. A merge removes the loser and updates the winner. Ordinary last_seen/belief-only changes do not emit updates (P:1278). Restart changes publication stream store_id and resets sequence; IDs still persist. Snapshot is reliable transient-local depth 1; updates reliable volatile depth 100. A consumer must recover a sequence gap with GetObject/search or the snapshot, not assume reliable transport never drops.

#### `msg/LocalizedDetection.msg`

```text
uint32 detection_index             # Original request array index, including failed entries.
OperationStatus status
geometry_msgs/PointStamped centroid # Exact observation stamp, node's map frame.
uint32 points_used                 # Valid transformed input points; no fused/history points.
```

#### `msg/Event.msg`

```text
int64 id
builtin_interfaces/Time stamp      # Node clock at commit; not caller clock.
string kind                        # action,note,error,handover initially.
string text                        # <=512 characters, no personal details.
string map_id
bool has_robot_pose
geometry_msgs/PoseStamped robot_pose # TF at stamp; zero/empty if unavailable, never latest fallback.
int64 room_id                      # 0 unknown.
int64 object_id                    # 0 none.
int64 person_id                    # 0 none; tools never expose person records.
int64 session_id                   # 0 none.
string task_id
string cause
```

### Database services (all new)

#### `srv/LocalizeObjects.srv`

```text
string map_id
walkie_vision_msgs/ObjectDetectionArray objects
---
OperationStatus status             # Request validity; individual geometry failures are in results.
LocalizedDetection[] results       # Exactly len(objects.detections), in original order.
```

Require a valid map gate; <=1 s array age and <=5 ms future tolerance in ROS time. One TF lookup from optical frame into configured map frame at header.stamp, timeout 50 ms. Check every cloud's frame/stamp/layout; reject malformed input rather than reinterpret it. Each valid entry needs >=30 finite positive-camera-z points. Return arithmetic centroid of those already filtered points; no orientation/box confidence is fabricated. Missing geometry is NOT_FOUND for that entry. If TF is missing, every entry is TF_UNAVAILABLE. Successful empty input yields an empty results array. No fallback to latest TF or latest depth. No persistence or ID allocation. This read/compute operation can be called before M1.

#### `srv/SubmitDetections.srv`

```text
string request_id
string task_id
string cause
string map_id
walkie_vision_msgs/ObjectDetectionArray objects
---
OperationStatus status
# True only when the later M1 ingest worker has accepted this observation.
# NOT_READY and false in prep. Never pretend this has been durably fused.
bool queued
```

This is the sole forwarding route for DetectObjects. Later ingestion keys duplicates by map, stream_id, stamp and requested vocabulary, and processes each observation at most once. Requests must have untracked detections. Their absence does not trigger streaming negative evidence. ReportSighting can associate one named target from the same array without double-counting it. Forwarding can fail while the agent still uses a valid live detection; the agent must say memory was not updated.

#### `srv/SearchObjects.srv`

```text
ObjectQuery query
bool use_embedding                 # False is structured/lexical-only search.
ModelVector vector                 # Required nonempty known model only when use_embedding.
string text                        # <=512 characters; lexical terms only, not parsed into filters.
uint32 max_results 20               # Range 1..100; reject values outside it.
---
OperationStatus status
ObjectMatch[] matches
SearchPrior[] priors                # Unseen catalogue entries, separate from actual objects.
bool lexical_only
bool truncated
```

Map/room/state/layer filters apply before vector ranking. Thing/attribute/relation constraints also apply before the final result limit. Do not query global top-20 then discard other rooms. Start with exact scoring of all SQL-filtered candidates using vec_distance_cosine (small arena), with vec0's map/room/state/layer metadata kept correct. Using vec0 KNN is allowed only if a test proves top_k completeness for the remaining predicates. No unfiltered fixed shortlist. No image/caption rank fusion while captions are disabled. Unknown model is UNSUPPORTED, invalid dimension/NaN/zero vector INVALID. Never silently switch vector models.

#### `srv/GetObject.srv`

```text
string map_id
int64 object_id                    # Positive database ID, not a live obj_ handle.
bool include_cloud false           # Internal motion client only; LLM atomic always false.
---
OperationStatus status
int64 requested_id
bool merged                        # True when following merged_into changed the ID.
ObjectSummary object               # Current row; discarded is OK with state=discarded.
Surface[] surfaces                 # Anchor's surfaces, empty for ordinary movable objects.
sensor_msgs/PointCloud2 cloud       # Optional map-frame merged cloud; empty when unavailable.
```

Follow merged_into with a visited set and finite hop cap (32). Broken links/cycles are INTERNAL, not an infinite loop or NOT_FOUND. NOT_FOUND applies only to an unknown ID and may be accompanied by a caller-side search for valid candidates. Read-time resolution is implemented now using fixture links, even though maintenance creates no merges yet.

#### `srv/FindNear.srv`

```text
ObjectQuery query                  # relation required; text references resolved by SearchObjects first.
uint32 max_results 20
---
OperationStatus status
ObjectMatch[] matches
bool truncated
```

No overloaded raw natural-language reference. Agent resolves `id or text` into reference_id/reference_key first. on/inside read stored relations; near uses a bounding-box broad phase then exact centre distance, excluding the reference itself. Missing geometry is not a match. No derived relation is invented by prep.

#### `srv/ListRooms.srv`

```text
string map_id
---
OperationStatus status
RoomSummary[] rooms               # Sorted canonical names; empty valid map succeeds.
```

#### `srv/GetRoomContents.srv`

```text
ObjectQuery query                  # Nonempty room required; same state/layer/filter semantics.
uint32 max_results 100
---
OperationStatus status
RoomSummary room
ObjectMatch[] objects
bool truncated
```

#### `srv/SetObjectState.srv`

```text
string request_id
string task_id
string cause
string map_id
int64 object_id
uint64 expected_revision           # Must equal resolved current row revision.
# begin_grasp,grasp_succeeded,grasp_failed,prepare_place,place_succeeded,
# hand_over_succeeded,grip_lost. Not an arbitrary state setter.
string operation
string gripper_frame               # Required for grasp_succeeded, prepare_place and grip_lost.
bool has_pose
geometry_msgs/PoseStamped pose     # Trusted skill observation/release target; not LLM numbers.
string surface_key                 # Intended/verified placement surface, else empty.
---
OperationStatus status
ObjectSummary object              # Resolved current row, when available.
```

Define the operation list now; implement no state transitions in prep. Later `prepare_place` reserves expected release pose without leaving held. Other transitions follow P:1335–1340. The server validates operation/state pairing, gripper identity, frame and stamp, and commits live-state/SQL atomically from the caller's perspective. It never accepts a generic “set missing” from a failed verify. Gripper loss is the explicit held→missing exception. Movement of held geometry, MoveIt attachments and target reservations are later implementation, not promised by this stub.

#### `srv/ReportSighting.srv`

```text
string request_id
string task_id
string cause
string map_id
int64 object_id
bool seen
walkie_vision_msgs/ObjectDetectionArray objects # The actual on-demand observation, even if empty.
uint32 detection_index             # Used only when seen=true; index into objects.detections.
---
OperationStatus status
bool applied                       # True only if later evidence gates accepted a new independent look.
ObjectSummary object
```

A bare `(id,false)` is insufficient evidence. The array proves the stamp/classes and lets the later server consult TF, people and depth grids before applying a miss. Invalid index/label, duplicate observation, stale input, occlusion or failed gate never decrements belief. The agent must not convert a timeout into seen=false. Prep returns NOT_READY/applied=false.

#### `srv/ArenaReset.srv`

```text
string request_id
string task_id
string cause
string map_id
---
OperationStatus status
uint32 objects_changed
```

Later M5: set p_exist to 0.5 for active movable candidate/present/missing rows; preserve last_verified so old positions do not become fresh. Skip grasping/held/handed_over/discarded/merged. This narrows P:1210's “every movable” wording to protect explicit skill state. Transactional, idempotent by request_id. No deleting rows, reloading files, changing anchors, or clearing people. Agent's reviewed preset separately clears baseline overlay when that component is enabled. Prep returns NOT_READY.

#### `srv/RememberEvent.srv`

```text
string request_id
string task_id
string cause
string map_id
string kind                       # action,note,error,handover.
string text                       # 1..512 characters; no personal details.
int64 object_id                    # 0 none; otherwise resolve current object.
int64 person_id                    # 0 none; must belong to named/open session if provided.
int64 session_id                   # 0 selects open session only when person_id is nonzero.
---
OperationStatus status
Event event
```

Server supplies clock, TF robot pose and room. Missing TF still records the event with has_robot_pose=false and room_id=0; never claims a position. The agent supplies reviewed non-personal text; a server can enforce length/structure, not prove free text contains no personal information. Include that limitation in docs. Event write functionality is independent of fusion and is implemented in D8.

#### `srv/FindEvents.srv`

```text
string map_id
string text                       # FTS terms, not raw SQL or caller-authored MATCH syntax.
bool has_since
builtin_interfaces/Time since     # Same clock as this store.
string room                       # Empty means any.
uint32 max_results 20
---
OperationStatus status
Event[] events                    # Newest first, stable ID tie-break.
bool truncated
```

#### `srv/DescribeObject.srv`

```text
string request_id
string task_id
string cause
string map_id
int64 object_id
---
OperationStatus status
int64 resolved_id
bool queued                       # Async acceptance only; read eventual caption with GetObject.
```

Prep returns NOT_READY/queued=false. No blocking model call inside database. Later returns OK only for a present object with a valid retained crop and queue admission.

#### `srv/GetStoreStatus.srv`

```text
# Read-only probe; no request fields.
---
OperationStatus status
string map_id
string map_hash                   # Full canonical SHA-256, not just the 8-character suffix.
string baseline_rev               # Empty is valid no-baseline mode.
string store_id
string clock_kind                 # wall or ros_sim.
uint32 schema_version
bool map_valid
bool ingest_enabled               # False in preparation.
bool captions_enabled             # False in preparation.
string[] object_embedding_models  # clip_vit_b16; no fake model on a real deployment.
uint32 arrays_received
uint32 arrays_rejected
```

### Snapshot and baseline-file seams

Define snapshot schema version 1 now, even though live object maintenance is later. `/walkie_database/objects/snapshot` remains std_msgs/String, JSON keys: `schema_version:1`, `map_id`, `store_id`, `sequence`, `stamp:{sec,nanosec}`, `objects:[…]`. Every object includes `id`, `key`, `label`, `layer`, `state`, `geometry_valid`, `frame_id`, `position:[x,y,z]`, `orientation:[x,y,z,w]`, `size:[x,y,z]`, `p_exist`, `last_verified` (null or sec/nanosec). Boxes are mandatory whenever geometry_valid. Publish seeded snapshot in D8; no raw cloud/vector/crop/people in JSON. baseline_diff's overlay is a real consumer, so P:1585's “debugging only” is incomplete (P:21 and :156 describe that consumer). Do not use snapshot to replace transactional services.

Freeze input schema version 1 for D6 (JSON is accepted as YAML-compatible data; YAML uses safe_load). `rooms.yaml`: `schema_version`, `map_id`, `rooms:[{key,name,description,floor,polygon:[[x,y],…],stand_pose:[x,y,yaw]}]`. Key is a machine identifier; name is display text. `anchors.yaml`/`baseline/anchor_seeds.yaml`: `schema_version`, `map_id`, `baseline_rev` (empty permitted), `anchors:[{key,label,room_key,center:[x,y,z],size:[x,y,z],yaw,geometry_valid,enclosed,surfaces:[{key,z,outline:[[x,y],…]}]}]`. Optional `fingerprint:{model,embedding}` is accepted only with verified real-model provenance; omit it in manual seeds. No fabricated vectors. `catalogue.yaml`: `schema_version`, `objects:[{official_name,category,aliases,default_place,weight_g,attributes}]`; weight_g omitted when unmeasured. Labels never imply weight.

No need to wait for a baseline builder: manual anchor inputs have this same schema. A future baseline producer must emit it, or a versioned adapter in that producer must translate it. The raw baseline surface/grid/proposal formats remain walkie-ros-owned and cannot be declared implemented by these two repos. Record the required minimum now: optical capture header and CameraInfo for each depth/change grid; registered pixels; depth metres, invalid zero/NaN; proposal PointCloud2 preserves its original stamp/frame; manifest carries map_id, baseline_rev, full map hash, grid origin/resolution/dimensions and file hashes. Before M3c, walkie-ros must settle the exact grid encoding and proposal grouping; that is an explicit limit to the broad “ready forever” reading.

### Shared map identity helper (new, not a ROS type)

Own `src/walkie_map_id.py` in database. Package it as a top-level py_module beside walkie_database. It imports **stdlib only**. Exact API:

```python
def decode_pgm(data: bytes) -> tuple[int, int, int, tuple[int, ...]]:
    """Return width,height,maxval,row-major samples; accept P2/P5, reject malformed data."""

def map_digest(pgm: bytes, metadata: dict[str, object]) -> str:
    """Return lowercase 64-character canonical SHA-256; never read a path or import YAML."""

def make_map_id(arena: str, date: str, digest: str) -> str:
    """Return <arena>-<YYYY-MM-DD>-<digest[:8]> after validation."""
```

Algorithm version `walkie-map-v1`: SHA-256 over `b"walkie-map-v1\0"`, width and height as two unsigned big-endian 32-bit integers, each decoded pixel normalized to a uint16 sample using `(sample*65535 + maxval//2)//maxval` and encoded big-endian, then UTF-8 compact JSON with sorted keys and no trailing newline. JSON keys are `resolution`, `origin` (3 entries), `negate`, `occupied_thresh`, `free_thresh`, `mode`. Numeric real values become Python binary64 `float.hex()` strings, with negative zero normalized to zero; negate is integer 0 or 1. Require resolution>0, finite origin, 0<=free_thresh<occupied_thresh<=1, mode in trinary/scale/raw, width/height>0 and valid samples. Missing mode defaults to trinary; all other fields required. Reject boolean real values. Ignore YAML image path, whitespace, key order, comments, description and unrelated metadata. Reject malformed/duplicate required YAML keys in the **caller** before it passes the dict. P5 parser consumes only the header terminator, never strips whitespace-valued raster bytes. Support maxval 1..65535 and two-byte big-endian P5 samples when maxval>255.

make_map_id accepts arena matching `[a-z][a-z0-9_-]*`, a real ISO calendar date, and a lowercase 64-hex digest; reject all other inputs.

The dependency-free module deliberately does not parse YAML. Database's wrapper uses an explicitly declared PyYAML dependency with safe loading and duplicate-key rejection. walkie-ros uses its own safe parser and this exact module. Every consumer pins the database SHA and obtains this file verbatim, with a recorded file SHA-256, through its dependency setup; no hand-copied reimplementation. Golden P2/P5, comments, renamed image, key-order, negative-zero and changed-pixel fixtures ship with it. The full digest goes into the map row/manifest; an 8-character directory suffix is not a collision-proof gate.

## 4. Ordered build slices

Estimates are focused engineer time, not measured duration. S=0.5–1 day, M=1–2 days, L=2–3 days. Each row is one reviewable commit; do not mix model selection or the core into it. A row may add several small test fixtures. Both repositories require an ADR for a significant contract/design decision and an embedded entry on the architecture page. Accepted ADR decisions are superseded, not silently rewritten (V:CLAUDE.md:56–68; D:docs/architecture/index.md:21–52).

### Vision: eight slices

| # / title | Files | Behavior | GPU-free acceptance | Docs / ADR | Size / dependencies |
|---|---|---|---|---|---|
| V1 Freeze observation and service types | `ros2_ws/src/walkie_vision_msgs/{msg,srv,CMakeLists.txt,package.xml}`, `src/walkie_vision/convert.py`, `tests/test_convert.py` | §3 types; quality/colour/stream_id validation; old people services untouched | Exact field/default pins; malformed vectors/quality/cloud headers refused; nonempty and empty round trips; all new types build | vision.md message tables; new ADR-0013 `memory-observation-contract`; add architecture snippet | M; none |
| V2 Publish deterministic fake observations | New `src/walkie_vision/fake_publisher.py`, `config/fakes/boxes.json`, `tests/test_fake_publisher.py`, `pixi.toml` fake-detections task | Analytic box-surface clouds projected into a synthetic camera; configurable topic/classes/rate; empty frames and fresh stream_id on restart; no real model imports | Seeded two-box fixture, known stamps/points/bboxes, points<=2000, no person/object cross-talk, finite dropout/move sequence; actual ROS subscriber sees empty and nonempty arrays | vision-node guide and fake profile; ADR-0013 section distinguishes synthetic geometry from perception | M; V1 |
| V3 Add depth subscription and bounded pairing | New `src/walkie_vision/depth.py`, `node.py`, `image.py`, config files, `tests/test_depth.py`, `tests/test_node.py` | Newest RGB/depth slots, exact stamp preferred; gap<=5 ms; 20 ms max wall wait then profile policy; registered CameraInfo validation; 32FC1 metres/16UC1 mm with endian/step support; no TF | Test reversed arrival, 0/4/6 ms gaps, paused/reset sim clock, queue replacement, padded rows, big-endian values, wrong encoding/frame/calibration, depth missing/invalid; no stale pair reused | vision.md threads/QoS/config; new ADR-0014 `bounded-rgbd-and-service-work` supersedes only input-buffer/stateless wording of accepted ADR-0002 | L; V1 |
| V4 Keep masks and fill clouds/quality | `yolo.py`, `providers.py`, `depth.py`, `pipeline.py`, `convert.py`, `fakes.py`, tests of yolo/depth/pipeline | Preserve per-box mask alignment after tracker filtering; cleanup/lift/cap; exact quality formula below; no geometry fit/map lookup | Letterboxed mask tests and filtered-row association; analytic plane/cube, invalid/MAD depth, erosion fallback, 29/30-point boundary, 2000 cap, finite score bounds, invalid calibration; points in optical coordinates | vision.md geometry contract; ADR-0013 implemented parts marked; c12 extension planned in V8 | L; V3 |
| V5 Sample crops and add interim CLIP provider | New `src/walkie_vision/embeddings.py`, pipeline/providers/convert/models, `config/tasks/memory.yaml`, config, bakeoff adapter if sharing provider, tests | Padding/JPEG/sampling below; CLIP pinned weights, QuickGELU, runtime offline-only; explicit fetch of required HF file; no loader when disabled | Inject fake Embedder; N/K/counter boundary, clipping, tiny crop, JPEG header, normalize vectors, zero/NaN rejection, embedding/crop same source; stub loader asserts revision/local_files_only, no model download in tests | bakeoff guide fixes 15% margin/template; ADR-0014 permits one image counter; ADR-0008 remains Proposed pending measurements | L; V4 |
| V6 Serve bounded embedding work | node plus new `src/walkie_vision/requests.py`, embeddings/convert, tests of services/concurrency | EmbedText, EmbedCrop; DescribeCrop DISABLED; bounded admission and cancellation; one model worker, input callbacks continue; service work can run without a camera for EmbedText | Text service before first image; busy/timeout/shutdown, stale crop stamp/frame/size, late result discarded, no historical frame retention; no queue growth; streamed arrays preserved on a service failure | Service guide and ADR-0014; update ADR-0010 consequence through new ADR, not by rewriting decision | L; V5 |
| V7 Serve DetectObjects and finish sim profile | New no-tracker detector adapter in yolo.py; requests/node/config/sim.yaml/launch; tests | Separate YOLOE instance, last-list cache, newest pair chosen after re-prompt; caller-only result; sim names; enable flags and no lazy surprise load when off | Stub two model objects prove streaming classes/track IDs unchanged; 0 track_id; own array header/classes; points/embeddings filled; no topic publication from service; errors/timeout/cap; launch config test remaps every service | Update proposed ADR-0009 with final type and tests, keep robot-measurement acceptance condition; fix old agent-doc links | L; V6 |
| V8 Close tests, diagnostics and build recipe | `checks/c12_vision_node.py`, `tests/`, new `.github/workflows/test.yml`, `docs/`, `pixi.toml` as needed | Per-stage depth/mask/lift/crop/embed/service logs and skips; disabled path asserts zero added work; add own fake-test CI and document D9 pair command | Full tests, strict docs, clean message build; validate disabled-stage spies; D9 wire test once both SHAs exist. Do not run check-env in prep | Update README/getting-started, guides, architecture/ADR snippets; record unmeasured budgets explicitly | M; V1–V7, final pair waits D9 |

New ADR numbers above are the next free numbers at V's named commit. Recheck numbering if another change lands; update every snippet/link together. Leave ADR-0010's one-process choice in force. The new service scheduling ADR explains the bounded exceptions to the worker loop, not a second GPU process.

#### Precise vision behavior

- Default `config/vision.yaml` preserves the existing HRI workload: `geometry_enabled:false`, `object_embeddings_enabled:false`, `object_crops_enabled:false`, `on_demand_enabled:false`, `captions_enabled:false`. With all object additions disabled, do not subscribe to depth, copy masks to CPU, calculate Laplacian/colour shares, construct CLIP/second YOLOE, or encode JPEGs. New fields are valid empty/zero values; stream_id still exists. This is how the added work stays off the path.
- New `config/tasks/memory.yaml` supplies classes `[cup,bottle,bowl,apple,banana,backpack,chair,table,shelf,cabinet,sofa]`, without person and enables geometry, crops, embeddings and on-demand work. Make capability flags orthogonal. `object_crops_enabled:true` may be used without embeddings; the same sampler selects both. Do not select extra models merely because a service type exists. `sim:=true` chooses time/topics/services; it does not imply fakes or oracle data. Run `task:=memory sim:=true` for M2.
- Add `depth_topic:/zed_head/zed_node/depth/depth_registered`, `depth_camera_info_topic` defaulting to the existing camera-info topic, `max_pair_gap_s:0.005`, `pair_wait_s:0.020`, `require_depth:false` in HRI and true in memory. Reuse one CameraInfo subscription when topics coincide. Sensor input QoS is BEST_EFFORT/KEEP_LAST 1. Retain newest pending colour, depth and completed input bundle only; no backlog. When memory requires depth, an unpaired colour is skipped and counted, not published as empty. A correctly paired frame with invalid depth still publishes 2D detections with empty points.
- Input is rectified registered depth in the colour optical frame. Unregistered depth is refused; extrinsic registration remains a camera/prep responsibility in walkie-ros. For a registered lower-resolution depth raster, scale mask with nearest-neighbour and use its own valid CameraInfo; verify it describes the same optical rays/FOV. If synthesizing scaled intrinsics from colour, pixel-centre rule is `fx'=sx*fx`, `cx'=sx*(cx+0.5)-0.5`, and likewise y. Refuse arbitrary calibration/frame mismatch. This intentionally strengthens today's frame warning at V:node.py:807–812 into a geometry gate.
- Retain the YOLO mask for the exact selected result row before dropping untracked/low-score boxes. Remove letterbox padding and resize to native pixels once. Do not use a neighbouring detection's mask after row filtering. Pure fixture tests cover asymmetric shapes and two differently filtered rows.
- Erosion amount is `clamp(round(0.05*min(mask_extent_width,mask_extent_height)),1,5)` on the depth raster. Use a square kernel radius equal to that amount (diameter `2*r+1`), zero border. Mask depth must be finite and inside profile range; start ZED/sim 0.2–3.0 m and document as tunable. Retain z within `max(3*1.4826*MAD,0.01 m)` of median. If fewer than 30 survive the eroded path, retry the uneroded mask with the same depth rules. If still <30, emit no points. The 1 cm MAD floor prevents zero-MAD fixtures from rejecting small valid noise.
- Back-project `x=(u-cx)*z/fx`, `y=(v-cy)*z/fy`, `z=z`. Select at most 2000 row-major points at evenly spaced deterministic indices, **after** validation and before serialization. No random reshuffle, map transform, DBSCAN, plane removal, or box fit in vision.
- Quality is `0.25*(area+centred+depth_cover+sharpness)`, clamped [0,1]. `area=min(1,sqrt(mask_pixels/(0.10*W*H)))`. `centred=max(0,1-hypot((cx_box-W/2)/(W/2),(cy_box-H/2)/(H/2))/sqrt(2))`. `depth_cover` is valid range-filtered depth pixels / mask pixels before erosion, zero without depth. `sharpness=min(1,var(Laplacian(gray)[mask])/1000)`, zero for fewer than 50 mask pixels. Use native RGB mask for area/sharpness and nearest-neighbour depth mask for coverage. Use OpenCV BGR-to-gray uint8, Laplacian ddepth=CV_32F, ksize=3 and default reflect-101 border. Compute the grey image/Laplacian once per frame, not per detection. These are initial formulas, not a calibrated probability. Test relative blur/edge/coverage behavior; tune only on recordings with a versioned quality recipe.
- One processed-image counter starts at zero and increments once per successful processed image. Sampling eligibility uses the published track_id and this counter: `(track_id+counter)%10==0`, original box short side>=32 pixels. Choose up to K=3 eligible objects by descending score, then track_id/index as stable tie-break. All other detections leave crop/vector/colour shares empty. Reject per-object fake vectors from real-model paths.
- Crop box grows by 15% **on each side**, clipped to image. Use BGR pixels converted to RGB before CLIP preprocessing. Compute colour from the original mask; use the deterministic palette recipe below. Encode JPEG after size reduction to max_side=512, quality=90. Compute embedding from the padded source crop, then CLIP's pinned 224 preprocessing. Keep text template `a photo of {}.` and QuickGELU from V:tools/bakeoff/embedders.py:115–141; do not mix raw-token and templated-text vectors under an unchanged model name. Pin `timm/vit_base_patch16_clip_224.openai` revision `977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d` (V:models.py:110). Model identifier stays `clip_vit_b16`.
- Colour recipe v1 uses OpenCV float32 sRGB-to-CIELAB conversion and nearest Euclidean Lab palette centre for each original masked pixel; ties choose the earlier centre. RGB centres, in field order, are `(0,0,0),(255,255,255),(128,128,128),(220,35,35),(245,130,25),(245,220,35),(45,155,65),(40,90,205),(135,65,175),(235,145,180),(115,75,45),(210,190,145)`. Normalize counts by mask pixel count. This is deterministic plumbing, not a claim of colour constancy; recording-based accuracy remains a caption re-entry gate. No computation when unsampled/disabled.
- The worker runs every model call, including services. Replace its frame-only wait loop with a wakeable frame/request scheduler. One pending service job total; per-service BUSY response rather than executor threads waiting behind each other. Use a reentrant service group or sufficient nonblocking admission so a second request can actually observe BUSY; callbacks cannot all occupy the only input-callback threads. After each service job, process the newest frame before admitting another. EmbedText must wake the worker with no image. Use monotonic time for deadlines, ROS time for image age; /clock pause must not make requests hang.
- Do not claim concurrent service latency and 5 Hz are guaranteed. Serial `set_classes` can stall streaming for approximately 1.5 s. Default memory behavior accepts this measured-design tradeoff for rare on-demand checks, records the gap, and does not produce fake empty frames. If continuous 5 Hz during re-prompt is required, change the scheduling/process ADR after measuring a supported alternative.

### Database: nine slices

| # / title | Files | Behavior | GPU-free acceptance | Docs / ADR | Size / dependencies |
|---|---|---|---|---|---|
| D1 Pin vision and freeze database interfaces | New `ros2_ws.repos`, `tools/import_interfaces.py`, pixi/lock, message package/CMake/package.xml, field tests | Pin V1 or later full SHA; build vision before DB; all §3 types; no edit to legacy messages | Independent clean build/import, dependency-order test, old FindObjects/RememberedObject fields and hashes unchanged, new type serialization | database.md, object-memory input correction; new ADR-0019 `object-service-contracts`; update proposed ADR-0012 | M; V1 |
| D2 Shared canonical map hash | `src/walkie_map_id.py`, pyproject py-modules, tests/golden fixtures, docs | Exact §3 algorithm; dependency-free public module | P2/P5 equal, metadata reorder/name change equal, pixel/origin/resolution change unequal; P5 whitespace bytes and 16-bit edge cases; reject malformed/NaN/duplicate-key wrapper inputs | new ADR-0020 `map-identity-and-seed-lineage`; share helper consumption instructions | M; none |
| D3 Lifecycle-ready object schema and IDs | New migration `0004_object_lifecycle.sql`, new `src/walkie_database/objects.py`, schema/migration tests | Exact additions below; durable allocator API and tombstones; no association | v3 populated upgrade, twice idempotent, rollback failure, nonreuse after delete/restart/reservation, FK self-links/cycle validation, geometryless candidate not spatially indexed | database.md fields; object-memory target design; ADR-0019 plus proposed ADR-0011 update only for storage contract | M; D1 |
| D4 Rebuild interim CLIP metadata safely | `0005_object_vector_filters.sql`, objects/vector helpers, tests/seed.py column inserts | Preserve real CLIP vectors, map/room/state/layer metadata, delete/update triggers | Exact bytes preserved from v3; map/room/state/layer filtering before rank; high global similarity in wrong room cannot hide requested object; update cascades and full rollback | database.md queries; ADR-0007 implementation notes; setup migration log | M; D3 |
| D5 Rooms, surfaces, catalogue, relations and events schema | `0006_object_context.sql`, pure record/query helpers, tests | Schema below; FTS and privacy triggers; no relation inference | Polygons, FK/cascade/set-null, surface-target links, catalogue aliases; event FTS insert/update/delete; forgetting session removes person-linked events *before* set-null | new ADR-0021 `rooms-and-object-queries`; new ADR-0022 `task-event-log`; embed both | L; D3 |
| D6 Seed, restore and enforce map identity | New `seed_store.py`, `seed_sim.py`, `restore.py`, `map_config.py`, `store_lock.py`, node startup, tasks/config, tests | Build empty seed with rooms/L2 anchors, no L3/events/people; stable keys; safe closed-file restore; startup map gate and clock_kind; inject node-clock now() into people callbacks; old demo tool explicitly separate | Invalid/missing inputs write nothing; same inputs same rows; reseed preserves anchor IDs; no fake vectors; live file lock refusal; WAL checkpoint copy test; restore resets run UUID but carries ID highwater; wrong map/full digest/clock refused; people timestamps follow fake sim clock | Rewrite harvest.md stale seed contract; revise proposed ADR-0016; ADR-0020 implements stopped-node exception to ADR-0005 via new ADR | L; D2–D5 |
| D7 Validate observations and replace GetObPose | New `ingress.py`, `localize.py`, config, node subscriptions/services, tests | KEEP_LAST 1 latest validated array, counters only in prep; no accumulation; stateless LocalizeObjects; mutation stubs; TF and request validation shared for later M1 | Exact-stamp moving-camera TF test, no-latest fallback, index preservation, empty/no-depth, stale/future/reset time, wrong map/frame/layout, strict stream/model validation; receiving array creates **zero object rows** | object-memory.md records validator vs core state; ADR-0019; ADR-0012 no longer assumes undefined input type | L; D1,D6,V4 |
| D8 Serve schema-backed queries/events and seeded snapshot | New `queries.py`, `events.py`, result conversion, node handlers/tests | All read types + RememberEvent, current-ID resolution, read-time p_now, catalogue priors, GetStoreStatus, snapshot; no fabricated fusion | Filter/attribute/relation/superlative suite, missing dimensions/weight, expired ages, aliases/merged/discarded, event retry/privacy, lexical fallback, TF absent event, 500-row fixture benchmark; NOT_READY writes | ADR-0021/0022 and all query docs; document new client atomics and internal geometry separation | L; D4–D7 |
| D9 Local pair-build gate and handoff | New `tools/pair_check.py`, `tools/contract_peer.py`, tests, test workflow, integration docs, final V pin | Two separate envs/type builds; local candidate sources/SHAs; source/import/hash checks; exchange messages/services; reports; no production writes | Deliberately mismatched type/import negative control; valid pair exchange; no GPU/import/model/router; strict docs/full DB suite; final pin uses tested V commit | docs/getting-started + setup logs; M0 evidence and outstanding consumers table | M; V8,D1–D8 |

D's next ADR numbers start at 0019. Proposed ADRs can be brought into agreement with this chosen prototype scope. Accepted ADR-0005's “single executor owns connection” remains true during prep; D6's offline utilities are a documented stopped-node exception. M1's eventual writer-thread/live-cache design requires a separate superseding ADR and must preserve the repository API introduced here. Do not implement that thread merely to make prep look complete.

#### Required schema details

Append migrations; do not edit 0001–0003. `migrate()` already wraps each file in BEGIN IMMEDIATE/ROLLBACK and refuses newer schemas (D:src/walkie_database/migrate.py:56–85). Preserve that behavior. sqlite-vec virtual tables must be rebuilt by copy/drop/create, not renamed (D:migrations/0002_name_vector_tables_by_weights.sql:18–23).

**0004** extends `map` with `map_hash TEXT NOT NULL DEFAULT ''`, `baseline_rev TEXT NOT NULL DEFAULT ''`, `clock_kind TEXT NOT NULL DEFAULT 'wall' CHECK(clock_kind IN ('wall','ros_sim'))`. Add single-row `store_meta(id INTEGER PRIMARY KEY CHECK(id=1), store_id TEXT NOT NULL, clock_kind TEXT NOT NULL, map_id TEXT NOT NULL REFERENCES map(id), created_at REAL NOT NULL, refresh_anchors_pending INTEGER NOT NULL DEFAULT 0 CHECK(refresh_anchors_pending IN (0,1)))` for active node/run identity. On every node start generate and commit a new store_id before accepting subscriptions/services, and reset publication sequence to zero; do not refresh last_verified. Legacy data starts with blank hash and cannot start in map-gated memory mode until registered; existing legacy people-only mode remains usable without a selected map.

Add `object` columns:

| Column | SQL semantics |
|---|---|
| key | TEXT nullable, partial UNIQUE(map_id,key) WHERE key IS NOT NULL |
| layer | TEXT NOT NULL DEFAULT 'movable', CHECK in anchor/movable; backfill source=map to anchor |
| state | TEXT NOT NULL DEFAULT 'candidate', CHECK eight lifecycle values; source=map backfill present; other legacy rows stay candidate until reviewed/reseeded |
| p_exist | REAL NOT NULL DEFAULT 0.5 CHECK 0.05..0.95; map anchors backfill 0.95 |
| last_verified | REAL nullable; unknown legacy sightings do not become verified |
| geometry_valid | INTEGER NOT NULL DEFAULT 1 CHECK 0/1 for legacy measured rows; new no-depth candidates explicitly 0 |
| geometry_stamp | REAL nullable; last geometry observation stamp, separate from last_seen |
| in_baseline, depth_seen | INTEGER NOT NULL DEFAULT 0 CHECK 0/1; in_baseline set by confirmed baseline seed, not every historical map row |
| revision | INTEGER NOT NULL DEFAULT 0 CHECK >=0 |
| official_name, category | TEXT nullable |
| attributes | TEXT NOT NULL DEFAULT '{}' CHECK json_valid and json_type='object'; application validates enum keys/values |
| printed_text, caption_model, prompt_version | TEXT nullable |
| caption_view_id | INTEGER nullable; validated observation ownership by application (avoid a circular-delete FK) |
| best_view_quality, fingerprint_weight | REAL NOT NULL DEFAULT 0, quality CHECK 0..1, weight CHECK >=0 |
| viewpoints | INTEGER NOT NULL DEFAULT 0 CHECK >=0; detailed view samples go in object_observation |
| anchor_node_id, anchor_offset | INTEGER nullable and TEXT nullable JSON pose; unused in prototype, no RTAB-Map dependency |
| merged_into, split_from | INTEGER nullable REFERENCES object(id) ON DELETE SET NULL; application rejects self/cycles/cross-map links |
| gripper_frame | TEXT nullable; later held-state owner |
| expected_pose, expected_surface_key | TEXT nullable JSON stamped pose, TEXT nullable; later prepare_place reservation |

Rebuild `last_id` CHECK to allow object alongside session/person, preserving both counters and initializing object to max(existing object.id). Add object insert highwater trigger. All new object allocation goes through an allocator that reserves positive IDs in a **committed transaction before publishing/returning an ID**. Reserve outside any caller-owned SQL transaction; reject nested reservation calls. Set SQLite synchronous=FULL outside that reservation transaction, commit it, then restore NORMAL; otherwise a power loss could discard the highwater and reuse an issued ID. Test the reservation/commit ordering and PRAGMA changes; actual storage power-loss durability remains a hardware limit. M1 can reserve blocks and leave gaps after a crash; it must not allocate solely in RAM then crash before persisting highwater, contrary to a loose reading of P:1495. Delete/reinsert must never reclaim an issued ID. Guard insertions below highwater in application except IDs owned by the current committed reservation and retained seed anchor identities; fixtures may set explicit IDs only in a fresh store. Same-map offline restore carries highwater from the outgoing closed store; separate sim episodes are separate store_id scopes and agents are relaunched.

Add `object_label(object_id FK CASCADE,label TEXT,count INTEGER>=0,PRIMARY KEY(object_id,label))`. Extend `object_observation` with view_score REAL nullable CHECK 0..1 and camera_x/y/z plus camera_qx/qy/qz/qw REAL nullable. Also add `object_observation.color_shares TEXT` nullable, valid JSON array of 12 finite [0,1] values summing to 1 when present; future DescribeCrop must receive the evidence for its retained crop, not colours from a newer frame. No raw masks or full uncompressed image columns. Retention at 100 rows/object and one observation/second, best crop selection and file garbage collection are M1 behavior, but these columns exist now.

**0005** recreates `object_clip_vit_b16` with `object_id INTEGER PRIMARY KEY`, `embedding FLOAT[512] distance_metric=cosine`, `map_id TEXT`, `room_id INTEGER`, `state TEXT`, `layer TEXT`. All metadata non-null in every insert; room_id=0 for none. Stage old vectors in a temporary ordinary table inside the migration, drop old triggers/vec table, recreate the same name, repopulate by joining object rows for authoritative metadata, then recreate delete and metadata-update triggers. Do not rename a vec0 table or manipulate its shadow tables. Keep triggers for room deletion's SET NULL, state/layer changes and map reassignment; better reject map reassignment outside a full rebuild. Preserve room_clip_vit_b16 unchanged. Use named insert columns throughout seed/tests; existing positional inserts such as D:tests/test_schema.py:49–51 otherwise break. No SigLIP or caption vec table now.

**0006** adds:

- `room.key TEXT` with unique `(map_id,key)`; canonical names retain existing `(map_id,name)` uniqueness. Room polygons stay in `room_shape` through geopoly_ccw. Use a separate mapping from input key to display name; do not globally replace underscores with spaces.
- `surface(id INTEGER PRIMARY KEY, map_id FK map CASCADE, anchor_id FK object CASCADE, key TEXT, z REAL, outline TEXT CHECK valid JSON array, geometry_valid INTEGER CHECK 0/1, UNIQUE(map_id,key))`. Add `object.enclosed INTEGER NOT NULL DEFAULT 0 CHECK 0/1`. Invalid/unmeasured anchors have no usable on/inside surfaces.
- `object_catalogue(id INTEGER PRIMARY KEY, map_id FK map CASCADE, official_name TEXT, category TEXT, default_place TEXT, weight_g REAL nullable CHECK >0, attributes TEXT valid object, UNIQUE(map_id,official_name))`; `catalogue_alias(catalogue_id FK CASCADE, alias TEXT, PRIMARY KEY(catalogue_id,alias))`; `object.catalogue_id INTEGER nullable FK catalogue ON DELETE SET NULL`. ObjectSummary weight comes from this relation.
- `object_relation(src_id FK object CASCADE, dst_id FK object CASCADE, predicate TEXT CHECK on/inside, surface_id INTEGER nullable FK surface CASCADE, weight REAL CHECK 0..1, updated_at REAL)`. Unique indexes distinguish anchor-level inside and surface-level on rows. Require a surface for on; require surface NULL for inside. Validate same map and anchor ownership. near and in_room are computed, not persisted edges.
- `object_event(id INTEGER PRIMARY KEY,object_id FK object CASCADE,ts REAL,old_state TEXT,new_state TEXT,cause TEXT,task_id TEXT)` for later lifecycle audit, distinct from general events.
- `event(id INTEGER PRIMARY KEY,ts REAL,kind TEXT,text TEXT,map_id FK map CASCADE,x REAL nullable,y REAL nullable,yaw REAL nullable,room_id FK room SET NULL,object_id FK object SET NULL,person_id FK person SET NULL,session_id FK session SET NULL,task_id TEXT,cause TEXT)` plus external-content FTS5 index on text with insert/delete/update triggers. Add BEFORE DELETE ON session deleting events WHERE session_id=old.id AND person_id IS NOT NULL, before cascading people deletes set person_id NULL. Other event references SET NULL. Preserve all existing people/session semantics.
- `write_request(request_id TEXT PRIMARY KEY,task_id TEXT,operation TEXT,payload_hash TEXT,outcome_json TEXT,committed_at REAL)`. Lifetime is one test/store lineage, including node restarts; offline seed restore clears it. A new publication store_id does not clear retry history. Use bounded canonical JSON for hash/outcome; reject reused ID with different operation/payload. Do not store raw image/crop/embedding payloads in this ledger.

No cloud accumulator, lifecycle transitions, relation inference, embedding average or dirty-flush scheduler is introduced in these migrations.

#### Seed, map gate and read behavior

A fresh production seed contains one map, rooms, reviewed keyed L2 anchors and their surfaces/catalogue entries, plus counters/metadata. It contains no L3 objects, sessions, people, conversations, events, observation histories, synthetic embeddings or active request ledger. An anchor with no measured size sets geometry_valid=false; never turn a navigation stand pose or degenerate furniture polygon into a physical bounding box. W's authoring file :11–15 has degenerate furniture polygons; only :7–8 supplies real room outlines. FakeRobot's stand poses/weights are not surveyed seed geometry.

`seed_store` loads all inputs and validates names, polygons, finite coordinates, frame/map/hash and unique keys **before** creating its output. Stand pose must lie inside the corresponding room polygon. A repeated named anchor updates the same ID when reseeding from a closed existing seed; new keys allocate above highwater. Removed anchors become tombstones in a re-seed lineage, not ID reuse. Building the final seed uses a fresh logical store reconstructed from validated rooms/anchors and the old key→ID map; never copy live L3 rows. Optional real crop fingerprints can be supplied with model/dimension/provenance; database does not load a model or synthesize vectors.

`seed_sim` is an adapter from the W authoring TOML and an explicitly supplied locations TOML into that same seed tool. It requires an explicit room key→display-name map (`living_room`→`living room`, `kitchen_room`→`kitchen room` for the prototype) and exact corresponding location names. The current A:config/locations.sim.toml describes a blank-map demo; it is not an approved Chula seed. No invented fallback coordinates. A rooms-only seed is valid when anchor measurements are unavailable.

The node and offline seed/restore utilities acquire an exclusive advisory lock at `<db_path>.lock` before opening for write. Lock held for node lifetime; no PID-file-only check. The restore utility refuses a held lock. It validates seed schema/map/hash/clock and the outgoing store, then writes a temporary copy in the destination directory, sets a fresh store_id and carries the maximum outgoing/seed counters for object, person and session, checkpoints/closes it, fsyncs it, removes old target WAL/SHM only while locked, and atomically replaces the destination. A crash before replacement leaves the old database usable; a crash afterward leaves a complete restored one. Never copy an open SQLite file; use SQLite backup/VACUUM INTO or a closed seed. If an outgoing WAL exists, open/checkpoint the closed outgoing store before reading highwater. Do not erase WAL before recovering its committed counter.

Preserve existing `db-node` compatibility with `memory_enabled:false` by default. Add a documented `db-memory` task invoking `python -m walkie_database.node --ros-args -p memory_enabled:=true`; callers append explicit db_path, map_folder and map_id parameters. The pair harness always enables memory mode explicitly.

The map-gated memory launch requires map_folder, map_id and a matching store map row, full digest and clock_kind. Make these read-only startup parameters. A mismatch fails startup before accepting requests. Legacy people-only operation can explicitly set `memory_enabled:false`; it starts without a map, keeps old people services and old FindObjects behavior, and returns NOT_READY for new map-dependent services. It cannot ingest detections. Memory tests must not use this escape hatch to bypass the gate. A no-baseline folder is valid: baseline_rev empty; missing baseline is not missing map. Require manifest baseline_rev to equal the store when a baseline is selected; a nonempty store baseline_rev with a missing manifest is an error. Empty on both sides is valid no-baseline mode. The seed and restore tool set store_meta.refresh_anchors_pending=1. On the first initialized deployment-clock sample, one startup transaction stamps seeded anchors first_seen/last_seen/last_verified and clears that flag. In sim, wait for a /clock sample rather than treating clock zero as uninitialized forever. On later crash restart leave those times unchanged. Otherwise a reboot would falsely refresh stale anchors.

D6 CLI contract: `python -m walkie_database.seed_store --map-folder PATH --arena NAME --date YYYY-MM-DD --rooms PATH --anchors PATH --catalogue PATH --clock-kind wall|ros_sim --output PATH [--previous-seed PATH] [--replace]`; map_id is computed with the helper, never guessed. `seed_sim` takes the same map/identity/output arguments plus `--world-source PATH --locations PATH --room-names PATH` and optional anchors/catalogue; it emits the normalized room inputs before calling seed_store. `python -m walkie_database.restore --seed PATH --db PATH --map-folder PATH --clock-kind wall|ros_sim --replace` performs only the locked offline restore. All paths are explicit, no default writes to the owner's data directory, and output replacement requires an explicit `--replace` flag. A rooms-only seed uses empty schema-versioned anchors/catalogue lists.

Query handlers call a small `ObjectRepository` API; prep implementation reads SQL. M1 replaces its live state/pose source behind the API and joins SQLite vectors/history, as P:1274 requires. Do not expose conn to agent/ingest. Keep all SQL on its owning executor thread in prep. WriteEvent and reads are bounded. The later live cache must see every skill/caption mutation first and commit immediately for state writes; tests must cover read-your-writes when that arrives.

Read-time age is max(0,now-last_verified) on the configured clock. Unknown verification time yields has_last_verified=false and p_now=0 for ranking; no freshness claim. Decay half-life is 1800 s movable and 86400 s anchor. State never changes from age alone. Filter attributes have match/unknown/mismatch results. Return mismatches only if no non-mismatch candidate passes the hard filters. Initial score within each group is `clamp(0.6*((cosine+1)/2 if vector else lexical_match)+0.25*p_now+0.15*attribute_fraction,0,1)`; unknown attributes contribute zero, not false matches. Explicit superlatives order by the requested known measure first, then score and ID; return unknown-measure entries flagged after measured entries. Largest/smallest use box volume; thickest/thinnest use its minimum side; heaviest/lightest require measured catalogue weight. Lexical v1 uses Unicode casefolded whole-word/token matching over label set, official name, aliases, category, caption and printed_text, with every query token required for the fallback. No extra object FTS table is required at arena size; the plan calls it a suggestion. If use_embedding and the thing has no lexical/alias match, require cosine>=0.25 for CLIP as a labelled prototype threshold, configurable by model and pending the M4 phrase benchmark. `lexical_match` is 1 for a matching nonempty query and 0 for an unconstrained/absent lexical query. These ranking constants are prototype defaults, not an accuracy claim. Test on seeded facts; the 50-phrase real benchmark stays M4.

### Cost, latency and memory

P:1122's 86.6 ms p50 leaves **13.4 ms** against a 100 ms/10 Hz budget; p95 108.2 ms already exceeds it by 8.2 ms. The documents report 4–6 Hz under realistic load, not a spare half-frame. Do not claim that adding features preserves 10 Hz. The 5 Hz target offers 200 ms service time per processed image, but publish age also includes arrival/queue/conversion and can still exceed 0.3 s.

| Addition | Known cost | Preparation budget / disabled behavior |
|---|---|---|
| Pairing/depth decode/masks/lifting | Unmeasured here; CPU work and mask transfer are new | Initial engineering target <=8 ms p50 for all geometry at 5 detections/2000 points; measure, never assert as result. Zero work/subscriber when geometry off |
| Crop/quality/JPEG | Unmeasured; full-image Laplacian and JPEG can matter | Cache one grayscale/Laplacian per frame; K=3; log each stage. Quality/crop work absent when geometry and object crops/embeddings off |
| CLIP ViT-B/16 | Runtime batch cost and combined VRAM unmeasured. SigLIP's 4 ms/crop is **not** a CLIP measurement | N=10/K=3 bounds work, at 10 eligible tracks average about one crop/frame before cap. With embeddings off no CLIP construction or HF access |
| DetectObjects | Plan/ADR reports ~1.5 s for a new 5-class list; second-model VRAM unknown | Off by default; single bounded request; rare call may stall stream; measure first/same/new-list cases. No streaming-class changes |
| Captioner | Plan reports 0.18 s/caption and 786 MiB alone | Deferred and disabled; zero model VRAM and no database caption queue |
| Transport | 2000*12 = 24 KB XYZ/object; 5 objects*10 Hz = 1.2 MB/s raw XYZ before headers/crops | Cap point clouds and JPEGs; do not send full RGB/depth across repository service calls |
| Database reads/localization | New; sqlite-vec laptop timing is not loaded-node latency | Test 500-row query timing diagnostically, no universal wall-time CI gate. Full p95<=200 ms with ingest waits for M4 |

V8 logs stage p50/p95, array age, skipped/unpaired frames, queue refusals, sampled crop count, class-prepare latency and model load failures. c12 should exercise memory mode as an explicitly invoked GPU check, with a recorded fixture and model files, later. If geometry makes <5 Hz, reduce the cap/number of lifted objects by score as P:1025 permits before changing process architecture. If model additions still miss the target, disable optional embeddings/on-demand/captions in the task profile; do not report an optimistic estimate as a pass. Budget the robot as 24 GB until measured; the development RTX 2060's 6 GB is not enough evidence for combined models, MuJoCo, ZED and GraspNet.

## 5. Out of scope and re-entry conditions

| Work | Milestone | Why out now | What brings it in |
|---|---|---|---|
| Persistent transforms, baseline stripping, DBSCAN/plane cleanup, upright persistent boxes | M1 4.7–4.9 / M3c | Scene-graph core expressly excluded; only stateless localization subset is early | A separate core implementation using this contract, fixtures and repository API |
| Assignment, fusion, candidates/present, maintenance merge/split, dirty flushing and live cache | M1 5.1–5.3, 5.8 | These are the later algorithms, not “preparation” | M1 builder; preserve ID allocation, schema and services; add writer-thread ADR |
| Motion/head/lift gates, person/robot occluders and full existence evidence | M3/M7 4.1, 4.4a/b, 5.4 | Need robot state, depth grids and recordings; no useful fake proof of accuracy | Core builder plus walkie-ros validity/grid contract; recordings and robot test |
| Baseline capture/diff/nav layer/MoveIt furniture exporter | M3a/M3b | Owned by walkie-ros; no code changes there allowed | walkie-ros builder emits agreed seed/hash inputs and defines grid/proposal encoding |
| Relation inference and robot query-accuracy benchmark | M4 5.6/7 | Stored-row reads can be built now; inference needs fused geometry and labelled data | M1/M2 geometry and 50-phrase dataset; semantic scoring tuned there |
| Florence-2/other captioner, caption queue, caption vec table, OCR | M4 5.5, cut 7 | Optional and costs latency/VRAM; schema/type reserved now | A task demonstrably needs it and recorded crop accuracy/combined GPU budget pass; implement behind frozen types |
| Real skill-state transitions, evidence application, arena belief reset | M5 | Depends on core state, live cache and reviewed skills | Core + skill clients; replace NOT_READY only after durable/read-your-writes tests |
| RunSkill/FakeSkillServer | M5/M7a old plan | Conflicts with provisional commander-bundle direction | Explicit architecture reversal; it is not silently introduced here |
| GraspNet move and walkie_perception service migration | Migration 3 / M7a | Torch/CUDA extensions, TF/live cloud cache and service ownership are separate work | Dedicated move, build against V torch 2.11/CUDA12.8, GPU/in-distribution cloud tests, new ADR superseding GraspNet part of ADR-0002 |
| Remove perception/ or stop old pose/grasp nodes | Migration 4 | Would remove current 3D/grasp path before replacement core and GraspNet are ready | Migration steps 1–3 proven end to end; robot launch edits in walkie-ros |
| RViz markers and merged scene cloud | M1 diagnostics | No fused geometry exists; standard types do not need a new custom interface | M1 core owns live boxes/clouds; add ADR-0013 publishers then |
| MoveIt PlanningScene publication/attachment lifecycle | M7a | No confirmed dynamic objects or real skill transitions yet | Map-fixed planning frame + core + real arm verification; use frozen IDs/geometry |
| SigLIP/EmbeddingGemma/second fingerprint | M7 bake-off | No robot-camera dataset or winning model yet | Measured bake-off win, new table/re-embedding, possible ObjectDetectionV2 if second vector required |
| RTAB-Map pose graph / 5.7 | Parked | Fixed AMCL map has no task-time graph | Explicit localizer change, new map/baseline and interface/hash validation |
| Robot preflight/launcher, router recovery, OOM supervision, hardware/noise tuning | M7 | Owned across deployed stack and cannot be proved by fake unit tests | Integration plan and measurements on actual deployment |
| ToF/D415/pet support, camera-model selection | M8 / conditional | No selected ToF driver/profile; no reader needs D415; pets omitted | Hardware/task requirement and real profile recording |
| Broad agent/ROS implementation | Later consumers | User excludes changes here | Separate consumer work, requirements only in §8 |

## 6. Decisions made for this prototype

All defaults are chosen. None asks the owner for a decision. “No” in the final column means no external decision blocks beginning the build; listed code dependencies still apply.

| # | Chosen default | Alternatives considered / reason | Non-obvious consequence | Reversal cost | Blocks build? |
|---|---|---|---|---|---|
| 1 | Ready means selected M0–M5 seam + working independent services, not finished scene graph or all future releases | Whole pipeline now violates core exclusion; types-only prep leaves depth/embedding/seed callers unusable | M0 full consumer exit and M1/M2/M4/M5 behavior remain distinct gates | Scope/acceptance rewrite | No |
| 2 | Plain positive DB int64 IDs, decimal strings in agent records; live obj_ references separately | mem_ registry rejected by provisional decision 4 | FakeRobot's apple_1-style IDs are fake identity, not a database naming rule; no string-to-track guess | Adapter and traces; no SQL ID migration needed | No |
| 3 | Preserve IDs within map/store lineage; carry highwater on offline reset, use fresh store_id and relaunch agent across episodes | Resetting counters from seed contradicts “never reused”; UUID primary keys would break existing int64 API | “Restore seed” is logical contents plus highwater/run metadata, not a bit-identical copy | Restore/ID policy migration, no reuse of earlier IDs | No |
| 4 | Preserve old DB types; new ObjectSummary/services | Expanding RememberedObject would break currently working main FindObjects | Old find_objects remains semantically limited; later agent must select a new memory capability | Cheap to remove legacy endpoint once no clients remain | No |
| 5 | Edit ObjectDetection once for quality/colour; array once for stream_id | Optional new envelope type would force duplicate streaming paths; implicit restart detection is weak | All existing object consumers must rebuild together; people fields remain stable | Coordinated type bump/versioned new type | No |
| 6 | Add colour shares now; captions disabled | Deferred shares would need another ObjectDetection hash change because masks are discarded | Extra field stays empty in disabled/HRI mode; palette order is a contract | New quality/colour recipe and re-computation, possibly new version | No |
| 7 | CLIP ViT-B/16 OpenAI/QuickGELU, 512, fixed template/margin | SigLIP proposal lacks bake-off; DINO not default | Similarity thresholds cannot be copied from SigLIP examples; new margins change stored image meaning | Re-embed retained crops and migrate vector provenance | No |
| 8 | Recreate existing object_clip_vit_b16 with map/room/state/layer | New SigLIP table now unnecessary; post-filtered global top-k incorrect | All explicit insert sites/fixtures must add metadata; vec0 rename is unsafe | Another numbered migration/backup | No |
| 9 | Separate on-demand YOLOE, same process, only caller receives result | Re-prompt stream corrupts tracking/vocabulary; second topic violates ADR-0009; second process breaks ADR-0010 | New list stalls the shared worker, and second-model VRAM remains unmeasured | New scheduling/process ADR and benchmark | No; hardware acceptance later |
| 10 | Agent forwards DetectObjects via SubmitDetections; ReportSighting supplies target-specific evidence | Vision publishing autonomously would mix acquisition intents; bare boolean cannot prove a miss | Same array must be deduplicated across submission/report; no memory update before M1 | Agent adapter and DB ingest behavior, no new message needed | No |
| 11 | LocalizeObjects lives in database, stateless centroid subset now | Vision TF violates ownership; agent geometry duplicates policy; wait for M1 blocks localize preparation | Returns per-detection status and no persistent ID, not a drop-in GetObPose type | Change only consumer adapter if a future richer localization type is added | No |
| 12 | Fake publisher lives in vision, outputs points; default no embeddings | Agent oracle lacks geometry and isn't in A; fake vectors under real names contaminate memory | Synthetic observations validate plumbing, not rendered occlusion/detector quality | New fixture adapter only | No |
| 13 | Fake embedding service names start fake_; real store rejects them | Pretend fake vectors are CLIP makes semantic tests meaningless | Unit query fixtures may explicitly create artificial vectors in isolated test DBs; production seed/publisher never does | Dedicated test model table only if M1 later needs fake semantic matching | No |
| 14 | Map helper stdlib module owned/pinned in database; YAML outside | New shared repo adds release coordination; independent implementations drift | Consumer needs a source pin even if importing no database runtime | Move module with golden compatibility tests | No |
| 15 | map_id hashes normalized decoded pixels + semantic metadata; full digest gate | Hash YAML/PGM bytes changes on harmless re-save; 8-char suffix alone too weak | Tight pixel parsing and float canonicalization become versioned public behavior | New hash version and new map IDs/reseed; never silently reinterpret old IDs | No |
| 16 | Rooms/anchors only across tests; events/L3 gone; online ArenaReset only reduces active movable belief | Keep L3/events only if task requires across-test history; a ROS file-reset is unnecessary | Events last one test. Online reset preserves age and held skill state | Policy migration and replay tests; no wire change | No |
| 17 | Reviewed room/anchor input files, with stable keys and optional real fingerprint | Navigation waypoints/degenerate polygons do not give boxes; seed from live DB leaks L3 | Rooms-only manual seed is valid; geometryless anchor cannot support on/inside/grasp | Reseed same keys once surveyed/baseline seeds exist | No |
| 18 | Exact current-image crop stamp within 5 ms; no historical image service | Retaining history needs new memory rules; relaxed old box crop may embed wrong object | Proposal embeddings may often refuse under latency; candidate stays honest unknown | Bounded-history ADR or proposal batching measured later | No |
| 19 | Structured ObjectQuery covers D4; no separate parser; finite enum superlatives | Free text filter JSON weakens validation; a second LLM parser adds latency | Unknown attributes/weight are explicit, room contents is not current view | Add new enum/typed field only for a demonstrated task; rebuild if fields change | No |
| 20 | Exact filtered scan at arena size; don't truncate before all hard filters | Global top-20 vec0 then filtering can lose valid answers | May cost more CPU than KNN; correctness first, measured at 500 objects | Optimize query implementation under same API | No |
| 21 | Captions use reserved DescribeCrop/DescribeObject interfaces, disabled; future Florence-2 + deterministic colour is default | VLM JSON adds unknown VRAM/quality; caption can be cut per P:179 | No stored caption vec table and no live caption queue yet; empty attrs are unknown | Add implementation/table after measured need | No |
| 22 | Single DB connection owner now; repository API ready for live-cache source later | Premature writer/ingest-thread implementation pulls in core scope | M1 must explicitly supersede executor ADR and prove read-your-writes; SQL reads alone are not later live truth | Internal refactor, no message change | No |
| 23 | Map gate in database only; preflight reads GetStoreStatus | Duplicated launcher hash gates drift | Invalid node does not start, so preflight may report unavailable and use startup log; no “gate failed” live service is promised | Launcher policy changes only | No |
| 24 | Domain 77+ / Fast DDS / localhost for all development tests; no router | Production domain23/Zenoh would join live system | Fast DDS wire pass is not a Zenoh network/robot release pass | Separate approved deployment test, no code rewrite | No |
| 25 | Independent .pixi/install overlays per prep worktree | Main-manifest activation and editable .pth paths silently target old code | Install cost is real, but no retargeting/sharing mutable environments | Remove disposable worktrees/environments after handoff | No; disk monitored |
| 26 | D's pin baseline is 64403e9; move every new consumer to the final tested D SHA, V to final tested V SHA | Agent's 7f657b0 lacks newer services; adopting unrelated experiment branch is unnecessary | Existing legacy object calls can keep working, but new imports require pin/build list update | Revert pins only together with consumer feature usage and schema backup | No |
| 27 | Do not require World for the fake M0 contract gate | Original M1 exit does need real free-joint truth | M0 can run immediately; “every body Present after patrol” remains unmeasured | Add world-driven fixture adapter later | No |
| 28 | Budget 24 GB robot / 6 GB development; service/model flags off by default outside memory profile | Assume RTX5090 and all models fit would be unsupported | GPU startup/memory/latency are release blockers for real-model use, not fake prep | Re-profile on actual hardware | No for fake build |
| 29 | No photo tool, no D415 consumer, no pets, no extra detector/second embedder | All optional and lack a demonstrated prototype need | DescribeCrop is internal and does not grant the LLM image access | Scoped capability/ADR/model change | No |
| 30 | Accept small baseline leftovers; clear arena/review occupants later; do not add automatic speck filter here | Changing baseline rules now belongs to later evidence logic/ROS capture | Fake tests cannot establish missing-object timing on transparent or small objects | Recordings, threshold revision and baseline review | No |
| 31 | Defer GraspNet with old services retained; no migration-4 removal claim | Move now entails untested CUDA, TF/cache exception and MoveIt work | Full grasp/localize replacement cannot be declared complete just because the new type compiles | Dedicated migration, caller pin updates and hardware checks | No |
| 32 | Scene input stamps use deployment ROS clock; service deadlines monotonic; people timestamps also move to node clock when sim-enabled | Existing people code calls time.time (D:node.py:165–166, :225), contradicting one-clock plan | Add a narrow injected now() provider and tests; do not compare old wall-time people DB against sim epochs | Seed/restart on clock-kind change | No |
| 33 | Hash/camera/baseline file contracts versioned; future raw grid/proposal format is an explicit ROS-side prerequisite | Pretend standard PointCloud2/Image alone specifies grouping/encoding would hide a seam | No prep promise that M3c can finish before that ROS contract exists | Document and test ROS format before M3c; may need an adapter | No for M0/M1 |
| 34 | No phase-wide library upgrades; add only declared preparation/M1 CPU needs | Borrowing agent environment or moving all ROS builds causes unrelated ABI work | D adds vision-msgs, sensor-msgs-py, tf2-ros, PyYAML, SciPy and scikit-learn; V reuses existing CV/CLIP deps. Review lock for RoboStack drift | Re-solve/rebuild only if required, then re-run pair gate | No; solve is a builder check |
| 35 | Treat uncited NAIS figures and model-transfer thresholds as unverified; use explicit CLIP prototype thresholds only for fixture behavior | Importing reported accuracy/thresholds as acceptance proof would be unsupported | The 50-phrase and robot-camera benchmarks remain open; no external paper was independently verified here | Tune config from a recorded benchmark, preserve model provenance | No |
| 36 | Seed tools require explicit paths and replace flag; fresh-seed marker controls timestamp refresh once | Refresh on every boot falsely raises stale belief; overwrite by implicit defaults risks live data | A ROS sim clock sample is required before fresh seeded anchors are stamped | Small tool/schema migration; no ROS interface change | No |

“Default chosen” does not mean the hardware performance questions are solved. DetectObjects new-list latency, batch CLIP latency/VRAM, pairing reality, colour accuracy and localizer error remain measured acceptance work. No owner input is needed to begin the fake-capable build.

## 7. Environment and test recipe

### Verified activation and space

These are **commands for the builder to run later**. None of these worktrees, installations or builds was made during research. Start with the two named commits. Keep subsequent edits and commits in the detached preparation worktrees. The main working trees must remain untouched.

| Finding | Evidence | Required behavior |
|---|---|---|
| Both activation scripts source `${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh` | V and D `scripts/activate.sh:7–9` | Passing the main checkout manifest activates the main overlay even if cwd is a worktree. Always pass the preparation manifest |
| Both project packages are editable relative to the manifest | V:pixi.toml:44; D:pixi.toml:49–50 | The worktree needs its own installation, not the main interpreter with a new cwd |
| Main environments already have absolute editable source paths | Observed `.pixi/envs/default/lib/python3.12/site-packages/__editable__.walkie_vision-0.1.0.pth:1` and the database equivalent point to their `/home/tawan/…/src` directories | Do not reuse, copy, symlink or retarget those environments. Verify imports after installing |
| Activation forces Zenoh even over an exported Fast DDS value | V:pixi.toml:110–129; D:pixi.toml:52–66 | Put `env RMW_IMPLEMENTATION=rmw_fastrtps_cpp` **after** `pixi run` activation |
| The build shell does not acquire the newly built overlay | V:CLAUDE.md:20–21; D:pixi.toml:68–80 | Build and test in separate Pixi invocations |
| Existing tests use fakes and Fast DDS; domains are not uniformly supplied by the shell | V:tests/conftest.py:12–14; V:tests/test_node.py:137,:602; D:tests/conftest.py:13–20,:50 | Set domain 77 externally; retain D fixtures choosing 100–200. Pair test uses 78. Never use the default graph |
| Main default environments are large | Read-only `du`: vision 9.9 GB; database `.pixi` 2.3 GB with symlinks followed. Plain D `du` undercounts shared/symlinked contents | Budget a full additional **12.2 GB** plus docs/build/cache overhead. Hardlink/cache sharing may save space; do not depend on it |

The task began with a stated 36 GB free. A later read-only `df` during research showed **33 GB free**, while other work was running. At 33 GB, a conservative 12.2 GB environment budget plus 4 GB for builds/docs/cache leaves about 16.8 GB. This is sufficient on the observed sizes, not a reservation. Check free space before installing, after each install, and before the pair build. If free space would fall below 8 GB, stop installation and report the actual growth; do not delete another session’s caches/environments. Do not fetch model weights or copy recordings for preparation. A separate vision environment still installs its locked CUDA/PyTorch packages even though tests do not use a GPU. The docs environments are smaller and do not include the default feature.

Pixi **0.80.0** was observed at `/home/tawan/.pixi/bin/pixi`. Its local `run --help` confirms `--manifest-path`, `--locked`, `--clean-env` and `--skip-deps`. Plain `pixi run` can resolve/install; use `--locked` for reproducible checks. No main-manifest `--as-is` shortcut is acceptable.

### Create isolated worktrees and install

Every command below is CPU-affined. Commands inside a `taskset … bash` inherit that affinity. `--clean-env` avoids inherited main overlays. Each runtime command supplies Fast DDS/domain/GPU and offline flags after activation. The install/lock commands do not launch ROS or models.

```bash
taskset -c 0,1,6-11 git -C /home/tawan/walkie-vision worktree add --detach /var/tmp/wv-prep 0a49428d4b92e2f7c3ba75e621d31d8d87679774
taskset -c 0,1,6-11 git -C /home/tawan/walkie-database worktree add --detach /var/tmp/wd-prep 64403e982e253db0ca00c372249eeb798d89d1f6
taskset -c 0,1,6-11 df -h /var/tmp
taskset -c 0,1,6-11 /home/tawan/.pixi/bin/pixi install --locked --manifest-path /var/tmp/wv-prep/pixi.toml
taskset -c 0,1,6-11 /home/tawan/.pixi/bin/pixi install --locked --manifest-path /var/tmp/wd-prep/pixi.toml
taskset -c 0,1,6-11 /home/tawan/.pixi/bin/pixi install --locked -e docs --manifest-path /var/tmp/wv-prep/pixi.toml
taskset -c 0,1,6-11 /home/tawan/.pixi/bin/pixi install --locked -e docs --manifest-path /var/tmp/wd-prep/pixi.toml
taskset -c 0,1,6-11 df -h /var/tmp
```

If a target path already exists, inspect and reuse only the intended preparation worktree; do not remove it automatically. The commands above establish the baseline environments. D1 adds explicit Conda dependencies `ros-jazzy-vision-msgs`, `ros-jazzy-sensor-msgs-py`, `ros-jazzy-tf2-ros`, `pyyaml`, `scipy`, `scikit-learn`; ROS package declarations also name builtin_interfaces, std_msgs, geometry_msgs, sensor_msgs, vision_msgs and walkie_vision_msgs where used. Pin compatible versions in the generated lock; keep existing Python 3.12/RoboStack bounds. SciPy and scikit-learn are declared now for M1’s expected geometry/matching work, but no DBSCAN/fusion executes in prep. V uses its existing OpenCV, NumPy and open-clip dependencies. Do not upgrade Torch, torchvision, ONNX Runtime or the whole ROS stack as incidental cleanup.

After an intentional manifest dependency edit, update **that worktree’s** lock, review the diff, then install it. This is the only unlocked solve in the recipe:

```bash
taskset -c 0,1,6-11 /home/tawan/.pixi/bin/pixi lock --manifest-path /var/tmp/wd-prep/pixi.toml
taskset -c 0,1,6-11 /home/tawan/.pixi/bin/pixi install --locked --manifest-path /var/tmp/wd-prep/pixi.toml
```

If V’s dependencies genuinely change, perform the same two commands against `/var/tmp/wv-prep/pixi.toml`. A failed solve is unverified environmental work, not permission to borrow the main checkout environment. Record any required stack changes and rerun the pair gate.

### Import the local pinned vision dependency into database

D1 must provide `tools/import_interfaces.py` with exactly these arguments: `--vision-source PATH` and `--vision-ref FULL_SHA`. Without them it reads the full SHA and URL from `ros2_ws.repos`. With them it clones/fetches from the local Git repository and verifies the SHA equals the manifest pin. It checks out that commit detached at `ros2_ws/src/walkie-vision`, an ignored dependency path. It refuses a dirty existing dependency clone. It never checks out or edits the source worktree. Clone the repository; do not symlink V’s `install/` or mutable source package. No remote fetch is needed when the commit exists locally.

After committing the relevant V slices and setting D’s `ros2_ws.repos` to the same full SHA:

```bash
taskset -c 0,1,6-11 bash <<'SH'
set -euo pipefail
wv_commit=$(git -C /var/tmp/wv-prep rev-parse HEAD)
cd /var/tmp/wd-prep
/home/tawan/.pixi/bin/pixi run --locked --clean-env --manifest-path /var/tmp/wd-prep/pixi.toml env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/import_interfaces.py --vision-source /var/tmp/wv-prep --vision-ref "$wv_commit"
test "$(git -C ros2_ws/src/walkie-vision rev-parse HEAD)" = "$wv_commit"
SH
```

D’s build includes only walkie_vision_msgs and walkie_database_msgs in dependency order. It does not install the vision Python project, GPU models or its environment. The final D commit pins the final V commit, not merely the preliminary V1 type commit. Because V has no runtime dependency on D, this pin order is acyclic.

### Build messages, then test in fresh activations

Use an explicit empty colcon defaults file to avoid the owner’s `~/.colcon/defaults.yaml`; the current tasks protect the Python executable but not every other possible user default. This recipe also limits parallel compilation to two workers. It spells out the equivalent of build-msgs so the package allow-list and environment are visible.

```bash
taskset -c 0,1,6-11 bash <<'SH'
set -euo pipefail
mkdir -p /var/tmp/wa-impl-prep/db-vision/build-reports
printf '{}\n' > /var/tmp/wa-impl-prep/db-vision/build-reports/colcon-defaults.yaml
cd /var/tmp/wv-prep/ros2_ws
/home/tawan/.pixi/bin/pixi run --locked --clean-env --manifest-path /var/tmp/wv-prep/pixi.toml env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 CUDA_VISIBLE_DEVICES= COLCON_DEFAULTS_FILE=/var/tmp/wa-impl-prep/db-vision/build-reports/colcon-defaults.yaml CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 bash -c 'cd /var/tmp/wv-prep/ros2_ws && exec colcon build --symlink-install --parallel-workers 2 --packages-select walkie_vision_msgs --cmake-args -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"'
cd /var/tmp/wd-prep/ros2_ws
/home/tawan/.pixi/bin/pixi run --locked --clean-env --manifest-path /var/tmp/wd-prep/pixi.toml env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 CUDA_VISIBLE_DEVICES= COLCON_DEFAULTS_FILE=/var/tmp/wa-impl-prep/db-vision/build-reports/colcon-defaults.yaml CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 bash -c 'cd /var/tmp/wd-prep/ros2_ws && exec colcon build --symlink-install --parallel-workers 2 --packages-select walkie_vision_msgs walkie_database_msgs --cmake-args -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"'
SH
```

D1 updates its build-msgs description and dependency import instructions to match this two-package build. Package selection does not make missing dependencies appear: import first. If the prefix or message sources were previously built from another worktree, clean only this worktree’s colcon build/install/log directories before rebuilding. Never source the main overlay to make an import pass.

Run these four checks separately after the build:

```bash
taskset -c 0,1,6-11 bash <<'SH'
set -euo pipefail
cd /var/tmp/wv-prep
/home/tawan/.pixi/bin/pixi run --locked --clean-env --manifest-path /var/tmp/wv-prep/pixi.toml env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 ROS_PYTHON_CHECK_FIELDS=1 CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_AUTOINSTALL=false python -m pytest
/home/tawan/.pixi/bin/pixi run --locked --clean-env -e docs --manifest-path /var/tmp/wv-prep/pixi.toml docs-build
cd /var/tmp/wd-prep
/home/tawan/.pixi/bin/pixi run --locked --clean-env --manifest-path /var/tmp/wd-prep/pixi.toml env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 ROS_PYTHON_CHECK_FIELDS=1 CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest
/home/tawan/.pixi/bin/pixi run --locked --clean-env -e docs --manifest-path /var/tmp/wd-prep/pixi.toml docs-build
SH
```

No test may call fetch-models, check-env, a real provider constructor, voice playback or `ros2` CLI discovery. Use injected clocks, fake model results, analytic depth/masks, temporary SQLite files and local rclpy peers. A hidden GPU through another process is not exercised by these tests. CI uses the same dependency/model restrictions and a clean workspace; V adds a test workflow, D imports the pinned interfaces before its test build.

### M0 local pair gate, including unpushed commits

D9 implements `tools/pair_check.py` as a standard-library orchestrator callable with host Python 3. Its exact CLI is below. `--build` rebuilds the two independent message installations with the commands above and builds a small **client overlay** containing D’s message package using V’s interpreter and V’s already-built vision interfaces. Put this client overlay in the report directory. Do not source D’s compiled environment inside V’s environment: rebuilding the extra client types avoids that ABI shortcut. The tool runs short-lived peers with each worktree’s own Pixi manifest and injects fake providers into V. Every child inherits affinity, Fast DDS, localhost, domain 78, no GPU and offline-model flags. It supplies its own colcon defaults and uses PID-owned subprocess cleanup with bounded terminate/wait/kill; no name-based killing and no router.

The harness must:

1. Require clean tracked sources at both supplied commits. Verify D’s pin and imported dependency commit equal the supplied V commit. Record full SHAs and `pixi.lock` hashes. Fail rather than testing dirty uncommitted code under a clean SHA label.
2. Record `sys.executable`, `PIXI_PROJECT_ROOT`, source `__file__` and generated message module paths. Require V source under `/var/tmp/wv-prep`, D source under `/var/tmp/wd-prep`, and each peer’s message extensions under its own build prefix/client overlay. Reject `/home/tawan/walkie-vision`, `/home/tawan/walkie-database` and the other runtime’s site-packages. The shared package-cache path of an ordinary dependency is not an error.
3. Compare every generated type hash for both packages and nested dependencies. Jazzy’s generated installed `share/<package>/{msg,srv}/*.json` files contain `type_hashes:[{type_name,hash_string}]`; this layout was checked in both current overlays. Compare those inventories, not file-comment hashes. Include existing people/legacy object types and new services. A deliberately altered expected hash or old import path must make the harness fail before a wire test; never introduce a mismatched test node into any live graph.
4. Build a map/rooms/anchor seed from tiny tracked synthetic fixtures in D’s tests. Start a temporary database node in map-gated mode and a V fake publisher in domain 78. Send a successful nonempty point/crop observation with empty embedding, then a successful empty observation. Separately exchange a vector-bearing fake-model array, assert the peer decodes its vector exactly, and assert database rejects that unsupported model; never label fake vectors CLIP. Assert reception/validation counters and zero created object rows, since fusion is absent. Send a separately malformed array and assert rejection, not an empty successful view.
5. Exercise an actual LocalizeObjects request across the independently built packages with known camera→map TF at the exact stamp. Verify index-preserving successes/failures and numeric centroids. Test no-depth, stale and unavailable-transform cases. Inject a moving-camera TF fixture to catch use of latest instead of stamped TF.
6. Exchange DetectObjects, EmbedText and EmbedCrop requests with V’s fake providers; test successful empty detection, service errors and stable streaming vocabulary. Exercise D’s seeded SearchObjects/GetObject/ListRooms/GetRoomContents/FindEvents and RememberEvent paths. The synthetic CLIP ranking unit fixture is a separate temporary test database, not the production seed. Check unsupported core writes return NOT_READY and do not claim queue admission.
7. Exit zero only after bounded teardown and all assertions. Write JSON with per-check status, type-hash inventories, source/import evidence and captured stdout/stderr. Keep failures as failures; absence of a subscriber message is not a pass. No pytest/GPU/model logs should be omitted from the handoff report merely because a hash comparison passed.

After V8, D1–D9, final pins and clean local commits:

```bash
taskset -c 0,1,6-11 bash <<'SH'
set -euo pipefail
wv_commit=$(git -C /var/tmp/wv-prep rev-parse HEAD)
wd_commit=$(git -C /var/tmp/wd-prep rev-parse HEAD)
cd /var/tmp/wd-prep
env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=78 ROS_LOCALHOST_ONLY=1 CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 tools/pair_check.py --vision-root /var/tmp/wv-prep --database-root /var/tmp/wd-prep --vision-commit "$wv_commit" --database-commit "$wd_commit" --domain 78 --build --report /var/tmp/wa-impl-prep/db-vision/build-reports/pair.json
SH
```

This is the local substitute for a CI checkout of the unpushed pair. V’s ordinary CI cannot fetch an unpublished D candidate, nor can D’s ordinary remote CI fetch an unpublished V pin. Do not claim either remote CI was run. Once published later, D CI should invoke the same harness against its declared V pin. V CI runs its own suite; a coordinated candidate pair gate is the release condition for future type edits. The M0 two-repository exit is the real wire pass plus matching hashes. Whole-stack compatibility additionally requires the agent and walkie-ros consumers to rebuild and pass their own probes; the two-repository test does not impersonate those consumers.

### What remains unverified

Research parsed **31** proposed `.msg`/`.srv` specifications with the installed Jazzy `rosidl_adapter` parser using the read-only vision interpreter, `PYTHONDONTWRITEBYTECODE=1`, no rclpy initialization, no GPU and a scratch-clone cwd. All parsed. The script is `scratch/validate_contracts.py`. This checks IDL syntax only. It does not prove CMake dependencies, generated type hashes, DDS exchange, migration correctness or implementation behavior. No repository test suite, new build, new lock solve or strict docs build was run during research because the implementation does not yet exist and the source checkouts were read-only.

The builder must supply those results. A model-file/GPU run must additionally measure combined VRAM, first-use and steady CLIP latency, DetectObjects re-prompt gaps, actual 5 Hz/0.3 s age behavior and the 30-minute OOM target. A registered RGB-D recording must establish mask/depth alignment, point error, colour accuracy and crop usefulness. MuJoCo must establish rendered-depth integration, /clock resets and M1’s eventual patrol/Present behavior. The robot must establish TF/calibration/motion effects, localizer accuracy, grasp compatibility and loaded query latency. None is inferred from passing fake tests.

## 8. Effects on consumers

These are required changes for their owners, **listed only**. This preparation changes no agent or walkie-ros source and designs neither repository’s internal ROS implementation.

### Agent prototype

| Area | Required follow-up |
|---|---|
| Source pins | Add walkie-vision at final tested V SHA to `ros2_ws.repos`; move walkie-database from 7f657b0 to final tested D SHA. Do not use a branch name or retain 1576d6f from the experiment. Record the pair report |
| Dependency import/build | Add the V package path to clone-ros2-ws outputs. Add `walkie_vision_msgs` to build-ros2’s explicit package list, alongside `rai_interfaces walkie_database_msgs` and any other message packages selected by the parallel ROS brief. Declare `ros-jazzy-vision-msgs`; rebuild both custom packages. A:pixi.toml:97–116 currently imports/builds only RAI and DB |
| Live perception | Keep UnavailableVision until a genuine adapter exists. Later implement the existing `ctx.vision.objects(query,timeout_s)` and `people` contract from fresh perception; never turn SearchObjects/FindObjects into “currently visible” results. Empty success, unavailable and timeout remain distinct |
| Memory reads | Introduce a separately named memory capability/atomics for query_objects/get_object/find_near/list_rooms/get_room_contents/find_events. A currently has none. Expose ObjectQuery’s room/state/layer/category/attributes/relation/reference/superlative arguments directly; D4 rules out a second parser |
| Embedding/query composition | Use EmbedText vectors with SearchObjects model identity/dimension checks. Treat failure as explicit lexical-only search. Explicitly strip `vectors` from LLM-visible output; a suffix-only embedding filter would miss it, and an active such filter was not found in A. The LLM sees names/IDs/places/freshness and unknown flags, not clouds/vectors/poses |
| IDs and records | Persistent object IDs are decimal strings backed by int64 DB IDs. Live `obj_` references stay transient and cannot be cast to DB IDs or track IDs. Respect merged/split/discarded invalidation, stream/store identity and the plan’s motion/age invalidation rules. `ObjectSeen.size_cm`, weight_g and thickness_cm are unknown unless measured; do not carry FakeRobot’s invented dimensions/weights into real adapters |
| Existing perception atomics | A:atomics/perception.py:10–25 find_objects and :27–41 object_info call the current-view capability. Preserve that meaning; add separate remembered queries. Map categories through the reviewed catalogue; display `place` from known room/surface evidence or an honest unknown |
| On-demand results | DetectObjects returns only to the caller. Forward via SubmitDetections once M1 is enabled; call ReportSighting only with the named target and evidence required by its contract. Do not claim a detection entered memory when NOT_READY. Use LocalizeObjects for camera-cloud→map point conversion; never infer PoseArray index correspondence from the old service |
| Writes | Reviewed skill outcomes call SetObjectState/ReportSighting with request/task/cause and revision. Event atomic calls RememberEvent. ArenaReset is invoked at the task boundary only after its core handler exists. DescribeObject is asynchronous and unavailable while captions are disabled. A does not yet have these capability methods; reserved types do not enable them |
| Arms and ROS wiring | Keep `ctx.arms.pick(object_id)`, `place_on(place)` and `hand_over(person_id)` names aligned with the parallel brief; no RunSkill action is introduced here. Leave its UnavailableArms/UnavailableVision choices until backend-specific work passes. Rebuilding messages does not itself make manipulation or perception available |

For an unpushed demo, the agent builder uses **its own authorized worktree and environment**. Its dependency clones can fetch the local preparation commits; the public manifest still records the final full SHAs. It must not overwrite the shared `/var/tmp/wa-demo` or `/var/tmp/wa-demo-fix` worktree from this task. The exact dependency commands inside that agent worktree are:

```bash
# Run from the agent builder's own worktree. Each target clone must be clean.
taskset -c 0,1,6-11 git -C ros2_ws/src/walkie-database fetch /var/tmp/wd-prep HEAD
taskset -c 0,1,6-11 git -C ros2_ws/src/walkie-database checkout --detach FETCH_HEAD
# If vision has not been imported there yet:
taskset -c 0,1,6-11 git clone --no-hardlinks --no-checkout /var/tmp/wv-prep ros2_ws/src/walkie-vision
taskset -c 0,1,6-11 git -C ros2_ws/src/walkie-vision fetch /var/tmp/wv-prep HEAD
taskset -c 0,1,6-11 git -C ros2_ws/src/walkie-vision checkout --detach FETCH_HEAD
```

If the vision clone already exists, omit only the clone command. Compare each dependency HEAD with the recorded tested SHA before building. With the updated package allow-list, `taskset -c 0,1,6-11 pixi run --locked --clean-env --skip-deps build-ros2` avoids re-running a remote clone task for unpublished commits. Run it from that agent worktree using its manifest, then start a fresh activation for its tests. It launches no ROS node. Any agent runtime probe must explicitly set Fast DDS/domain>=77 after activation, use silent fakes and keep CUDA disabled. A private temporary `.repos` file with local repository URLs and the same full SHAs is an equivalent import route; never change a committed remote URL into a machine-local path.

### walkie-ros

- Pin/fetch the tested walkie_vision_msgs source and declare/build it wherever new arrays are consumed. Add vision_msgs and other transitive ROS package dependencies; a Python import alone is not a CMake/package dependency. Neither inspected ROS branch currently pins these repositories.
- Rebuild every typed subscriber/service client against the edited ObjectDetection/Array fields. Baseline_diff, the future memory input path and any agent bridge must match the complete type hash. People consumers still need the package even though their fields are unchanged.
- Pin the database helper source/commit and use its canonical hash and golden fixtures in map/baseline preparation. Keep its file hash in the deployment manifest. Use GetStoreStatus for the single database map gate.
- Supply registered RGB-D/calibration with capture stamps, exact-stamp TF, consistent /clock in sim, and the selected camera topic/frame profile. `sim:=true` in vision does not create a camera or register depth.
- Define the remaining raw baseline grid/proposal encoding before M3c and emit the frozen seed/anchor file schema. Consume snapshot geometry with its declared QoS/version; no inference from a label-only JSON object.
- Move localize callers to LocalizeObjects and preserve per-detection status. Do not stop `/get_3d_poses` or remove perception while old callers and GraspNet still need it. Migration step 2 is complete only when callers switch and a real recorded/robot comparison passes.
- For later memory reads/events/updates or PlanningScene bridging, add/rebuild walkie_database_msgs at the same tested D pin. Keep `walkie_database/<id>` collision names and filter states as ADR-0013 requires. MoveIt can begin its own map-frame/depth-cloud work independently of scene-graph core; no moveit_msgs dependency is added to these preparation repositories.

### Documentation links and handoff

M:703–704 calls out removed agent pages. V:docs/architecture/vision.md:808,:816,:837,:976,:987 and ADR-0009:9 link into the old agent tools/graph/agent design. D:docs/architecture/harvest.md:10,:259,:431 and object-memory.md:185 also point to old agent documentation. Update these affected references during V8/D9 to the agent owner’s real replacement pages, or to a stable repository/commit reference with a brief explanation until a replacement exists. Do not invent a live `capabilities.md` URL. Strict MkDocs checks do not prove an external GitHub URL resolves.

The agent owner must keep the agent harvest headings `#relaunch-per-episode`, `#isolation` and `#localization`, which D:docs/architecture/harvest.md:177,:240,:349 links to. The database owner must keep its own harvest heading “How the self-evolving agent reaches memory”, linked by D:docs/architecture/decisions/0017-one-open-session.md:76, or supply an explicit compatible anchor while updating its stale design text. Reconcile proposed object-memory/ADR-0012 input topics, proposed ADR-0016 rooms-and-anchors policy, and V’s stale agent pin statement. Accepted ADR history remains; new decisions explain changed policy. Agent docs must distinguish live vision from memory, plain DB IDs from live references, and implemented services from NOT_READY contracts. The Robot Memory plan should mark M0 “contract-ready” only after the pair report, keep M1/M2/M4/M5 exits open, replace mem_ and parser/RunSkill claims with the provisional choices, and record all §6 defaults. No plan publication or repository change was performed in this research.

The handoff consists of the two final clean local commit SHAs, D’s exact V pin, both lock hashes, unit/strict-doc results, the local pair JSON and the outstanding hardware/consumer checks above. **No owner decision blocks the fake-capable implementation.** Availability of packages and later GPU/recording acceptance remain explicit verification work, not decisions to silently assume complete.
