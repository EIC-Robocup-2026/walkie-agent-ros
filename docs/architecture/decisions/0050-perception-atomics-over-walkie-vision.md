# ADR-0050: Perceive objects and people through atomics over walkie-vision

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

walkie-vision runs the robot's vision models
([ADR-0034](/architecture/decisions/0034-vision-models-in-walkie-vision.md)).
Its model choices are proposed, pending an embedding bake-off and a
measurement on the robot (walkie-vision ADR-0008): YOLOE-26 with ByteTrack
for objects, an object embedder (SigLIP 2 proposed, CLIP ViT-B/16 the
baseline), YOLO26 pose, SCRFD and ArcFace for faces, OSNet for appearance,
and Florence-2 for captions. It publishes `/walkie_vision/objects` and
`/walkie_vision/people`, serves `EmbedText`, and proposes an on-demand
`DetectObjects` (walkie-vision ADR-0009).

The self-evolving agent plans atomics for objects only. A
`PersonDetection` already carries 17 keypoints, a face embedding and an
appearance embedding, and walkie-agent-v2 recognized eight gestures and
postures from keypoints for GPSR: waving, raising the left or right arm,
pointing left or right, sitting, standing and lying down.

Vectors compare only within one model. The object embedder is expected to
change after the bake-off, and walkie-database keeps one table per model
(walkie-database ADR-0007).

**Decision**

- The agent reaches walkie-vision only through perception atomics. The LLM
  sees labels, enums, scores and handles, never vectors or images
  ([ADR-0028](/architecture/decisions/0028-images-and-policy-robot-side.md)).
- `find_people` reads the newest `/walkie_vision/people` message, waiting
  at most 1 s. It needs no new walkie-vision service.
- Gestures and postures are classified in the agent from the keypoints,
  by porting walkie-agent-v2's predicates, their eight ids and their tests.
- Faces and appearance are matched by walkie-database, which owns the
  vector tables. The agent sends the vector from its handle store and
  stores no vector itself.
- Every vector comparison checks the model name that comes with each
  vector: `embedding_model`, `face_model` or `appearance_model`. The
  agent's thresholds live in config, keyed by model name. A model without
  tuned values fails the call instead of borrowing another model's values.

**Consequences**

- Gestures need no message change in walkie-vision, and stay testable
  without a GPU. A new gesture is an agent change.
- Switching the object embedder needs three changes together:
  walkie-database's migration, the new model name from the torch process,
  and the agent's tuned values. Until all three land, the old model keeps
  working.
- `recognize_person` and `remember_person` need walkie-database services
  that do not exist yet. A person's position needs depth, which
  `PersonDetection` does not carry
  ([Perception](/architecture/perception.md#open-questions)).
- Generated tools reach people only through these atomics. Which of them
  Competition code may call is open.
<!-- --8<-- [end:body] -->
