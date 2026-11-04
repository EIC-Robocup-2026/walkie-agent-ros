"""Mahoraga: a script tool, a script shelf and a fake-world gym for GPSR commands.

Research code in its own pixi environment. It reaches the robot's code only
through walkie_agent's atomics, the adapter and FakeRobot, and it imports no ROS.
docs/architecture/mahoraga-demo.md has the public API and the run commands.
"""

import os
from pathlib import Path


def home(path: str | Path | None = None) -> Path:
    """Where the shelf, gate results and logs live: `path`, MAHORAGA_HOME or ~/.local/share/mahoraga.

    Raises ValueError for a directory inside a git work tree, because agent-written
    scripts and logs must never end up in a commit.
    """
    chosen = path or os.environ.get("MAHORAGA_HOME") or "~/.local/share/mahoraga"
    root = Path(chosen).expanduser().resolve()
    for parent in (root, *root.parents):
        if (parent / ".git").exists():
            raise ValueError(f"{root} is inside the git work tree at {parent}.")
    root.mkdir(parents=True, exist_ok=True)
    return root
