"""Named map locations loaded from a TOML file."""

import math
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
    """Load `[locations.<name>]` tables from a TOML file, keyed by name.

    A place without a finite number for x or y is refused, and so is a yaw
    that is given but is not a finite number. Every such place is named in one
    ValueError. A file can then list the places a task needs before they are
    measured, and the agent will not start on it until they are.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    entries = data.get("locations", {})
    unmeasured = [
        name
        for name, entry in entries.items()
        if not all(_is_number(entry.get(key)) for key in ("x", "y"))
    ]
    bad_yaw = [
        name for name, entry in entries.items() if "yaw" in entry and not _is_number(entry["yaw"])
    ]
    problems = []
    if unmeasured:
        problems.append(
            f"{', '.join(unmeasured)} {'has' if len(unmeasured) == 1 else 'have'} "
            "no measured x and y."
        )
    if bad_yaw:
        problems.append(
            f"{', '.join(bad_yaw)} {'has' if len(bad_yaw) == 1 else 'have'} "
            "a yaw that is not a finite number."
        )
    if problems:
        raise ValueError(
            f"{path}: {' '.join(problems)} Measure each place on the map the robot "
            "localizes in, and fill in its x, y and yaw."
        )
    return {
        name: Location(
            name=name,
            x=float(entry["x"]),
            y=float(entry["y"]),
            yaw=float(entry.get("yaw", 0.0)),
            description=entry.get("description", ""),
        )
        for name, entry in entries.items()
    }


def _is_number(value: object) -> bool:
    # bool is a subclass of int, but `x = true` is not a coordinate. TOML also
    # has inf and nan, which no pose can be.
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
