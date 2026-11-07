"""Mahoraga: a fake-world gym for GPSR commands, over run_script and its shelf.

Research code with its own pixi environment, because inspect-ai and the GPSR
generator stay out of the robot's. It reaches the robot's code only through
walkie_agent.tools, walkie_agent.scripting, the adapter and FakeRobot, and it
imports no ROS. The agent never imports it.
docs/architecture/mahoraga-demo.md has the public API and the run commands.
"""

import os
from pathlib import Path


def home(path: str | Path | None = None) -> Path:
    """Where gate results and run logs live: `path`, MAHORAGA_HOME or ~/.local/share/mahoraga.

    The shelf is not here: its store is scripting.shelf.store_root's, which the gym's
    --shelf names. Raises ValueError for a directory inside a git work tree, because
    agent-written scripts in run logs must never end up in a commit.
    """
    root = Path(path or os.environ.get("MAHORAGA_HOME") or "~/.local/share/mahoraga").expanduser().resolve()
    for parent in (root, *root.parents):
        if (parent / ".git").exists():
            raise ValueError(f"{root} is inside the git work tree at {parent}.")
    root.mkdir(parents=True, exist_ok=True)
    return root
