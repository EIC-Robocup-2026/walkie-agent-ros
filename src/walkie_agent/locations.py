"""Named map locations loaded from a TOML file.

A place name is stored as a trimmed lowercase key, so a lookup that ignores case can
never be ambiguous. A file with any other name, an unknown key or a coordinate that is
not a finite number raises ValueError, which stops the agent at startup.
"""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)


@dataclass(frozen=True)
class Location:
    name: str
    x: float
    y: float
    yaw: float
    description: str = ""


def _finite(value: int | float) -> float:
    try:
        number = float(value)
    except OverflowError as e:  # A TOML integer has no size limit.
        raise ValueError("is too large to be a coordinate") from e
    if not math.isfinite(number):  # A TOML float can be nan or inf.
        raise ValueError("is not a finite number")
    return number


# Strict, so a boolean or a quoted number is refused instead of being converted.
_Coordinate = Annotated[StrictInt | StrictFloat, AfterValidator(_finite)]


class _Entry(BaseModel):
    """One `[locations.<name>]` table. The name is the table's key, not a field."""

    model_config = ConfigDict(extra="forbid")

    x: _Coordinate
    y: _Coordinate
    yaw: _Coordinate = 0.0
    description: StrictStr = ""


class _File(BaseModel):
    model_config = ConfigDict(extra="forbid")

    locations: dict[str, _Entry] = Field(default_factory=dict)


def load_locations(path: str | Path) -> dict[str, Location]:
    """Load `[locations.<name>]` tables from a TOML file, keyed by name, in file order.

    Raises ValueError with a one-line reason for a file that breaks the rules above.
    """
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path}: {e}.") from e
    try:
        entries = _File.model_validate(data).locations
    except ValidationError as e:
        reason = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in e.errors()
        )
        raise ValueError(f"{path}: {reason}.") from e
    for name in entries:
        _check_name(path, name)
    return {
        name: Location(name, entry.x, entry.y, entry.yaw, entry.description)
        for name, entry in entries.items()
    }


def _check_name(path: str | Path, name: str) -> None:
    # casefold, like RosBase's lookup, so two accepted names can never collide there.
    canonical = name.strip().casefold()
    if not canonical:
        raise ValueError(f"{path}: a place name is blank.")
    if name != canonical:
        raise ValueError(f"{path}: the place name {name!r} must be written {canonical!r}.")
