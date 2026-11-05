"""SystemFacts: ctx.facts from the computer's wall clock and the [team] table of config.toml.

It reads the wall clock rather than ROS time, so a simulator's /clock never
changes the time or the date Walkie says. It imports no ROS, so the fake-world
tests and the robot share it.
"""

import time
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from walkie_agent.capabilities.records import Team

# The keys of the [team] table, each a non-blank string.
TEAM_KEYS = ("name", "country", "affiliation", "timezone")


class SystemFacts:
    """Implements Facts: the wall clock in one IANA timezone, and a fixed team.

    `clock` returns seconds since the epoch, as time.time does. Tests inject
    one to fix the instant. An unknown timezone raises ValueError.
    """

    def __init__(
        self, team: Team, timezone: str, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._team = team
        self._zone = _zone(timezone)
        self._clock = clock

    @property
    def timezone(self) -> str:
        return self._zone.key

    def now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), self._zone)

    def team(self) -> Team:
        return self._team


def facts_from_config(config: Mapping[str, Any]) -> SystemFacts:
    """The SystemFacts that the [team] table of a loaded config.toml describes.

    Raises ValueError with a one-line message for a missing table, a missing or
    blank value, or an unknown timezone.
    """
    table = config.get("team")
    if not isinstance(table, Mapping):
        raise ValueError("The config needs a [team] table with name, country, affiliation and timezone.")
    blank = [key for key in TEAM_KEYS if not isinstance(table.get(key), str) or not table[key].strip()]
    if blank:
        raise ValueError(f"[team] needs a non-blank text value for: {', '.join(blank)}.")
    team = Team(table["name"].strip(), table["country"].strip(), table["affiliation"].strip())
    try:
        return SystemFacts(team, table["timezone"].strip())
    except ValueError as exc:
        raise ValueError(f"[team] timezone: {exc}") from None


def _zone(timezone: str) -> ZoneInfo:
    """The ZoneInfo of an IANA key such as "Asia/Bangkok", or ValueError in one line."""
    try:
        return ZoneInfo(timezone)
    except (KeyError, ValueError, OSError):
        raise ValueError(
            f'Unknown timezone {timezone!r}. Use an IANA name such as "Asia/Bangkok".'
        ) from None
