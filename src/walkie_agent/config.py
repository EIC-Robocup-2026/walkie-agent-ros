"""Reads config.toml, and the agent's run limit and deadline from its optional `[agent]` table."""

import tomllib
from pathlib import Path
from typing import Any

# Provisional until real runs measure them; `[agent]` in config.toml overrides them.
DEFAULT_RUN_LIMIT = 25
DEFAULT_DEADLINE_S = 60.0


def load_config(path: str | Path = "config.toml") -> dict[str, Any]:
    """The parsed file. A TOML syntax error is a ValueError that names the file."""
    with open(path, "rb") as f:
        try:
            return tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path}: {e}.") from e


def agent_settings(config: dict[str, Any]) -> dict[str, Any]:
    """The `run_limit` and `deadline_s` keyword arguments of build_agent."""
    agent = config.get("agent", {})
    return {
        "run_limit": agent.get("run_limit", DEFAULT_RUN_LIMIT),
        "deadline_s": agent.get("deadline_s", DEFAULT_DEADLINE_S),
    }
