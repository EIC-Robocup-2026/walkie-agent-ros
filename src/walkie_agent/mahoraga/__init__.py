"""Mahoraga: a fake-world gym for GPSR commands, over run_script and its shelf.

Research code with its own pixi environment, because inspect-ai and the GPSR
generator stay out of the robot's. It reaches the robot's code only through
walkie_agent.tools, walkie_agent.scripting, the adapter and FakeRobot, and it
imports no ROS. The agent never imports it.
docs/architecture/mahoraga-demo.md has the public API and the run commands.
"""

import os
from pathlib import Path

from walkie_agent.scripting.shelf import private_dir


def home(path: str | Path | None = None) -> Path:
    """Where the shelf, gate results and logs live: `path`, MAHORAGA_HOME or ~/.local/share/mahoraga.

    Raises ValueError for a directory inside a git work tree, because agent-written
    scripts and logs must never end up in a commit.
    """
    return private_dir(path or os.environ.get("MAHORAGA_HOME") or "~/.local/share/mahoraga")


def shelf_dir(path: str | Path | None = None) -> Path:
    """The gym's shelf directory: `path`, or the shelf/ directory of home()."""
    return Path(path) if path else home() / "shelf"
