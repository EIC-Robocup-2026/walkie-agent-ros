# ADR-0034: Run the vision models in a new repository, walkie-vision

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-19

**Context**

The old HTTP server,
[walkie-ai-server](https://github.com/EIC-Robocup-2026/walkie-ai-server),
runs the robot's vision models: object detection, image embeddings, body
pose, image captions, face recognition and appearance re-identification.
Checked against its `master` (d9596ff), one Flask process serves them one
request at a time, together with GraspNet grasping, speech-to-text and
text-to-speech.
[ADR-0015](/architecture/decisions/0015-repository-boundary.md) puts
perception in walkie-ros.

walkie-ros keeps its perception code in `perception/`, the
`walkie_perception` package. Checked against walkie-ros `main` (4e0b20b):

- The grasp pipeline's Python nodes, `yolo_node`, `obb3d` and
  `grasp_server`, run in a venv of their own, `~/perception-venv`, locked by
  `perception/uv.lock`: torch 2.11.0 from the CUDA 12.8 index, numpy 1.26.4
  and OpenCV 4.11.
  GraspNet's two CUDA extensions are compiled outside the lock, against
  that exact torch build: on each machine, or in the physical Docker image.
  After a torch change in the lock, they must be compiled again, or
  `grasp_server` cannot load them.
- CI builds the package's C++ nodes and service types. It never creates the
  venv, and the package registers no tests (walkie-ros
  `docs/guides/testing.md`). Nothing in CI imports a Python node.
- Under `pixi run`, `PYTHONPATH` comes before the venv's own packages. The
  venv then imports Pixi's numpy 2.5.3 and OpenCV 5.0 instead of its pinned
  versions (walkie-ros `docs/guides/perception.md`, Known issues).
- `obb3d` is the grasp side's 3D tracker. It keys each object by its
  ByteTrack ID from `yolo_node`, and drops an object after 5 frames without
  it (`decay_miss_frames`). `/grasp/from_cloud` and `/grasp/pos` take a
  segmented object cloud, such as the one `obb3d` serves on
  `/apex/get_object_cloud`.
- The package also has the classic pipeline, `ob_pose`,
  `ob_pose_service_cpp` and `cloud_to_map`, which turns 2D detections into
  3D positions in the `map` frame.

This repository's environment is fixed by RAI (`pixi.toml`). The voice
package `rai_s2s` pins torchaudio 2.3.1, which pulls torch 2.3.1
([ADR-0013](/architecture/decisions/0013-cuda-torch.md)), and rai-core
requires `numpy<2` and OpenCV below 5. torch 2.3.1's CUDA 12.1 build has GPU
code for `sm_50` to `sm_90` only (`torch._C._cuda_getArchFlags()`). The
robot is expected to have an RTX 5090, an `sm_120` GPU (walkie-vision
ADR-0005).

walkie-vision now has an environment for the models: ROS 2 Jazzy on
`rmw_zenoh_cpp` 0.2.11, torch 2.11.0 and torchvision 0.26.0 from the CUDA
12.8 index, and `onnxruntime-gpu` 1.26 for the face models. All 16 checks of
its `pixi run check-env` passed on an RTX 2060, among them rclpy on Zenoh
next to every model library (walkie-vision
`docs/setup/01-vision-environment.md`).

**Decision**

- The vision models run in walkie-vision, a new robot-side repository:
  object detection and segmentation with tracking, object and room
  embeddings with a text-embedding service, body pose, face detection and
  recognition, appearance re-identification, and captions. walkie-vision is
  not on GitHub yet.
- walkie-vision talks to the rest of the robot over ROS 2 only, and keeps no
  state beyond the tracker's ids within a stream. walkie-database stays the
  only memory (walkie-vision ADR-0002).
- It has its own Pixi environment. Its messages and its service are in
  `walkie_vision_msgs`, a package in that repository that holds only
  interface files (walkie-vision ADR-0003).
- Speech-to-text and text-to-speech stay in this repository's voice nodes.
- GraspNet, `yolo_node`, `obb3d` and the rest of `walkie_perception` stay
  in walkie-ros. `obb3d` forgets an object 5 frames after it last saw it,
  and serves grasping only. It is not the robot's memory.
- walkie-ai-server's GraspNet endpoint does not move to walkie-vision.
  `grasp_server` in walkie-ros already serves grasps.

This replaces two parts of ADR-0015, for the vision models only. The first
is the perception part of "Navigation, manipulation, perception and drivers
live in walkie-ros". The second is its first consequence: a new vision
ability starts as a walkie-vision change, not a walkie-ros change. The rest
of ADR-0015 stays in effect: this repository
holds only agent-side code, and navigation, manipulation, `walkie_perception`
and the drivers stay in walkie-ros.

We rejected walkie-ros's `perception/`. New models there would share one
lock with GraspNet's compiled extensions, nothing in its CI would catch a
break, and its venv mixes in Pixi's numpy and OpenCV under `pixi run`. We
also rejected this repository. RAI fixes its torch, numpy and OpenCV, its
torch has no code for the robot's expected GPU, and the proposed
[ADR-0028](/architecture/decisions/0028-images-and-policy-robot-side.md)
would keep camera images out of the agent's process.

**Consequences**

- The agent will use `walkie_vision_msgs`, for example the proposed
  `/walkie_vision/embed_text` service for text search. It must build the
  package in `ros2_ws` from a pinned commit, as it builds `rai_interfaces`
  ([ADR-0004](/architecture/decisions/0004-build-rai-interfaces-from-source.md),
  [ADR-0018](/architecture/decisions/0018-pin-robot-interfaces.md)). That
  waits until walkie-vision is on GitHub, because `ros2_ws.repos` needs its
  URL. The package needs only `std_msgs`, `sensor_msgs` and `vision_msgs`,
  which this environment already has.
- After every message change in walkie-vision, this repository must build
  the package again from the new commit. Under rmw_zenoh, a node built from
  the old definition receives nothing and prints no error.
- Agent tools will call walkie-vision directly, as they call walkie-ros
  ([ADR-0016](/architecture/decisions/0016-talk-to-walkie-ros-directly.md)).
  No tool does yet.
- This repository's environment does not change. torch 2.3.1 and `numpy<2`
  stay fixed for voice only.
- The same torch has no `sm_120` code, and no PTX to compile it from. On an
  RTX 5090, `torch.cuda.is_available()` is still true, so `LocalWhisper`
  loads on the GPU, and its kernels would fail with "no kernel image is
  available". The fallback to the CPU in
  [ADR-0013](/architecture/decisions/0013-cuda-torch.md) would not happen.
  Speech recognition on the robot needs a follow-up decision.
- One more repository to clone, install and run on the robot PC. Its
  environment is about 9.6 GB, and its nodes use the same Zenoh router on
  port 7447.
- While the grasp pipeline runs, `yolo_node` and walkie-vision both run a
  detector on the head camera, on the same GPU.
- The robot's GPU, its memory and its driver are not measured. Whether
  walkie-vision, GraspNet and the voice nodes fit on it together is open.

**See also:** [The Agent: what lives where](/architecture/agent.md#what-lives-where),
walkie-vision `docs/architecture/vision.md`, walkie-ros
`docs/guides/perception.md`
<!-- --8<-- [end:body] -->
