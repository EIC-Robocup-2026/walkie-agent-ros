# ADR-0013: Use the CUDA 12.1 PyPI builds of torch

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

`rai_s2s` pins torchaudio 2.3.1, which pulls torch 2.3.1. Whisper speech
recognition runs on torch. We first installed the CPU-only wheels to keep the
environment small, but the robot and the team's development computers have
NVIDIA GPUs.

**Decision**

We use the default PyPI wheels of torch 2.3.1, which are CUDA 12.1 builds.
`pixi.toml` does not pin torch itself: it comes from `rai_s2s`, and
`pixi.lock` fixes the version.

**Consequences**

- Whisper runs on the GPU when there is one (NVIDIA driver 525 or newer) and
  falls back to the CPU otherwise.
- The environment is large, about 11 GB, mostly CUDA libraries.
- KokoroTTS runs on `onnxruntime`, which stays the CPU build.
- To change torch, edit `pixi.toml` and run `pixi update torch torchaudio`.
  Removing a package index alone does not replace locked wheels.
- torch 2.3.1 and `numpy<2` are fixed for the whole environment. A policy
  model that needs newer versions cannot run in the agent's environment
  ([ADR-0028](/architecture/decisions/0028-images-and-policy-robot-side.md)).

**See also:** [Talk to Walkie by voice](/setup/07-voice.md)
<!-- --8<-- [end:body] -->
