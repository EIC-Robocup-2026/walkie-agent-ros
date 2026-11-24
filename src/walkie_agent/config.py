"""Reads config.toml, and the agent's run settings from its optional `[agent]` table."""

import tomllib
from pathlib import Path
from typing import Any

# Provisional until real runs measure them; `[agent]` in config.toml overrides them.
DEFAULT_RUN_LIMIT = 25
DEFAULT_DEADLINE_S = 60.0
# A history of this many approximate tokens or more is summarized between tasks, and
# its newest turns, at least COMPACT_KEEP_TOKENS of them, stay (PK-74). Also provisional.
DEFAULT_COMPACT_AFTER_TOKENS = 12_000
DEFAULT_COMPACT_KEEP_TOKENS = 3_000


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


def compaction_settings(config: dict[str, Any]) -> dict[str, int]:
    """The `after_tokens` and `keep_tokens` of a Compactor. after_tokens 0 turns it off.

    A threshold that is not a whole number, an after_tokens below 0 or a keep_tokens
    below 1 is a ValueError that names the key.
    """
    agent = config.get("agent", {})
    settings = {}
    for name, key, default, least in (
        ("after_tokens", "compact_after_tokens", DEFAULT_COMPACT_AFTER_TOKENS, 0),
        ("keep_tokens", "compact_keep_tokens", DEFAULT_COMPACT_KEEP_TOKENS, 1),
    ):
        value = agent.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            raise ValueError(f"[agent] {key} must be a whole number of at least {least}.")
        settings[name] = value
    return settings
