# ADR-0012: Install rai_s2s and rai_whoami from RAI's Git tag 2.12.2

<!-- --8<-- [start:body] -->
**Status:** Accepted, partly superseded by
[ADR-0068](/architecture/decisions/0068-static-robot-description.md) ·
**Date:** 2026-10-12

ADR-0068 removes `rai_whoami`. `rai_s2s` still installs from tag 2.12.2.

**Context**

`rai_s2s` (speech) and `rai_whoami` (identity) are not published on PyPI for
the RAI release we use. They are subfolders of the RAI repository. Installing
from Git at tag 2.12.1, which matches our `rai-core`, fails on a broken
submodule entry.

**Decision**

Both packages are installed from `https://github.com/RobotecAI/rai.git` at
tag 2.12.2, from `src/rai_s2s` and `src/rai_whoami`. Tag 2.12.2 has the same
sources for these two packages as 2.12.1 and drops the broken entry.
`rai_s2s` gets every extra except the GPU one, because it imports every
speech backend when it loads.

**Consequences**

- Pixi locks the exact commit in `pixi.lock`.
- When `rai-core` is upgraded, this tag must move with it.
- The speech extras pull in torch and add about 4 GB to the environment
  ([ADR-0013](/architecture/decisions/0013-cuda-torch.md)).

**See also:** [Talk to Walkie by voice](/setup/07-voice.md)
<!-- --8<-- [end:body] -->
