"""Named map locations loaded from a TOML file."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Location:
    name: str
    x: float
    y: float
    yaw: float
    description: str = ""


def load_locations(path: str | Path) -> dict[str, Location]:
    """Load `[locations.<name>]` tables from a TOML file, keyed by name."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {
        name: Location(
            name=name,
            x=float(entry["x"]),
            y=float(entry["y"]),
            yaw=float(entry.get("yaw", 0.0)),
            description=entry.get("description", ""),
        )
        for name, entry in data.get("locations", {}).items()
    }
