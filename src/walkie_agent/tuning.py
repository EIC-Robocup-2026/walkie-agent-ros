"""The perception and gaze values to tune on the robot, read from config.toml.

Most of perception's and the gaze loop's values are walkie-agent-v2's,
tuned on the robot, and stay in their dataclasses. The gaze loop's gain,
speeds and deadband are among them. The few that are still to tune there,
depend on the robot's camera or TF, or were tuned in simulation only
(hri.md, Delivery order) can be set in config.toml, so that tuning them at
the arena is a config edit:

    [walkie.perception]                    PerceptionConfig: tf_timeout_s
    [walkie.perception.front]              FrontConfig: min_face_rad, half_angle_deg
    [walkie.perception.vision_feed]        VisionFeedConfig: min_face_size_rad
    [walkie.perception.identity.face_distance_limits]        a limit per face_model
    [walkie.perception.identity.appearance_distance_limits]  a limit per appearance_model
    [walkie.gaze]                          GazeConfig: tf_wait_s

A missing table or key keeps the dataclass's default, so a config.toml
without these tables changes nothing. A distance limit table changes the
limits of the models it names; the other models keep theirs.

load_tuning() raises one TuningError that lists every problem: a table or
key that nothing reads, a value that is not a number, one its dataclass
refuses, or a pair of values that do not fit together. So a typo stops the
agent at start-up instead of leaving a default in place unnoticed. RAI's own
tables, such as [vendor] and [asr], are not checked here. A top-level table
that looks meant for [walkie], such as [walky.gaze] or [gaze], is refused.
"""

import dataclasses
import difflib
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

from walkie_agent.gaze import DEFAULT_CONFIG as DEFAULT_GAZE_CONFIG
from walkie_agent.gaze import GazeConfig
from walkie_agent.perception.front import FrontConfig
from walkie_agent.perception.identity import IdentityConfig
from walkie_agent.perception.vision_feed import DEFAULT_CONFIG as DEFAULT_VISION_FEED_CONFIG
from walkie_agent.perception.vision_feed import VisionFeedConfig
from walkie_agent.tools.atomic._perception import PerceptionConfig

# Relative to the working directory, where RAI reads config.toml too.
CONFIG_PATH = Path("config.toml")

# The tables under [walkie] that the agent reads. A new [walkie.<name>]
# table adds its name here, or this loader refuses it as a typo.
WALKIE_TABLES = ("perception", "gaze")

# The keys of each table, named as the dataclass fields they set.
PERCEPTION_KEYS = ("tf_timeout_s",)
FRONT_KEYS = ("min_face_rad", "half_angle_deg")
VISION_FEED_KEYS = ("min_face_size_rad",)
IDENTITY_TABLES = ("face_distance_limits", "appearance_distance_limits")
GAZE_KEYS = ("tf_wait_s",)

# The tables inside [walkie.perception].
_PERCEPTION_TABLES = ("front", "vision_feed", "identity")
# Every table name the loader reads. One of them at the top level of the
# file is a table whose [walkie.] prefix was left out.
_TABLE_NAMES = (*WALKIE_TABLES, *_PERCEPTION_TABLES, *IDENTITY_TABLES)

_Config = TypeVar("_Config")


class TuningError(ValueError):
    """The tuning tables of config.toml are wrong. `errors` holds every problem found."""

    def __init__(self, source: str, errors: list[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        count = "1 problem" if len(errors) == 1 else f"{len(errors)} problems"
        super().__init__(f"{source}: {count}:\n" + "\n".join(f"- {e}" for e in errors))


@dataclass(frozen=True)
class Tuning:
    """The configs the agent is built with: the dataclass defaults, changed by config.toml."""

    # The perception atomics' config, which they read as _perception.CONFIG.
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    vision_feed: VisionFeedConfig = DEFAULT_VISION_FEED_CONFIG
    gaze: GazeConfig = DEFAULT_GAZE_CONFIG


def load_tuning(path: str | Path = CONFIG_PATH) -> Tuning:
    """Read the tuning tables of config.toml. Raises TuningError listing every problem."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return parse_tuning(data, source=str(path))


def parse_tuning(data: Mapping[str, Any], source: str = "config.toml") -> Tuning:
    """The configs that a parsed config.toml sets. Raises TuningError listing every problem."""
    errors: list[str] = []
    _refuse_misplaced(data, errors)
    walkie = _table(data, "walkie", errors)
    _refuse_unknown(walkie, WALKIE_TABLES, "walkie", errors)

    where = "walkie.perception"
    raw = _table(walkie, where, errors)
    _refuse_unknown(raw, (*PERCEPTION_KEYS, *_PERCEPTION_TABLES), where, errors)
    defaults = PerceptionConfig()
    front = _section(defaults.front, raw, f"{where}.front", FRONT_KEYS, errors)
    vision_feed = _section(
        DEFAULT_VISION_FEED_CONFIG, raw, f"{where}.vision_feed", VISION_FEED_KEYS, errors
    )
    identity = _identity(defaults.identity, raw, errors)
    changes = _read_numbers(raw, PERCEPTION_KEYS, where, errors)
    perception = _replace(
        defaults, {**changes, "front": front, "identity": identity}, where, errors
    )
    gaze = _section(DEFAULT_GAZE_CONFIG, walkie, "walkie.gaze", GAZE_KEYS, errors)
    _check_front_faces(perception.front, vision_feed, errors)
    if errors:
        raise TuningError(source, errors)
    return Tuning(perception=perception, vision_feed=vision_feed, gaze=gaze)


def _refuse_misplaced(data: Mapping[str, Any], errors: list[str]) -> None:
    """Refuse a top-level table meant for [walkie] whose name misses or misspells it.

    Such a table would otherwise be read by nothing, and the defaults would
    stay in place. The other top-level tables are RAI's, such as [vendor]
    and [asr], and are not checked.
    """
    for key in data:
        if key == "walkie":
            continue
        name = key.lower()
        if (
            key in _TABLE_NAMES
            or name.startswith("walkie")
            or difflib.get_close_matches(name, ["walkie"])
        ):
            errors.append(
                f"Unknown table [{key}]; the agent's tables go under [walkie], which takes "
                + ", ".join(WALKIE_TABLES)
                + "."
            )


def _check_front_faces(
    front: FrontConfig, vision_feed: VisionFeedConfig, errors: list[str]
) -> None:
    """Refuse a front rule that keeps faces too small to be recognized.

    find_people(in_front=True) keeps a face only when it is at least
    min_face_rad, and face_trusted names a person by a face only when it is
    at least min_face_size_rad, on the same measure (perception/front.py).
    With the defaults, 0.10 rad and 0.0469 rad, every face kept in front can
    be recognized.
    """
    if front.min_face_rad < vision_feed.min_face_size_rad:
        errors.append(
            f"[walkie.perception.front]: min_face_rad {front.min_face_rad:g} is below "
            f"[walkie.perception.vision_feed] min_face_size_rad "
            f"{vision_feed.min_face_size_rad:g}; a person kept in front must have a face "
            "that can be recognized."
        )


def _identity(
    default: IdentityConfig, perception: Mapping[str, Any], errors: list[str]
) -> IdentityConfig:
    """`default` with the distance limits that [walkie.perception.identity] sets."""
    where = "walkie.perception.identity"
    raw = _table(perception, where, errors)
    _refuse_unknown(raw, IDENTITY_TABLES, where, errors)
    changes: dict[str, Mapping[str, float]] = {}
    for name in IDENTITY_TABLES:
        known: Mapping[str, float] = getattr(default, name)
        table = _table(raw, f"{where}.{name}", errors)
        limits = dict(known)
        for model, value in table.items():
            if model not in known:
                # A misspelt model would add a limit that nothing reads,
                # while the model meant kept its old one.
                errors.append(
                    f"[{where}.{name}]: unknown model {model!r}; the models with a limit "
                    "are " + ", ".join(known) + "."
                )
                continue
            number = _number(value, f"{where}.{name}.{model}", errors)
            if number is not None:
                limits[model] = number
        if table:
            changes[name] = MappingProxyType(limits)
    return _replace(default, changes, where, errors)


def _section(
    default: _Config,
    parent: Mapping[str, Any],
    name: str,
    keys: tuple[str, ...],
    errors: list[str],
) -> _Config:
    """`default` with the numbers that the table `name` sets for `keys`."""
    raw = _table(parent, name, errors)
    _refuse_unknown(raw, keys, name, errors)
    return _replace(default, _read_numbers(raw, keys, name, errors), name, errors)


def _table(parent: Mapping[str, Any], name: str, errors: list[str]) -> Mapping[str, Any]:
    """The table `name`, whose last dotted part is its key in `parent`; empty when missing."""
    value = parent.get(name.rpartition(".")[2])
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"{name} = {value!r} is not a table; write it as [{name}].")
        return {}
    return value


def _refuse_unknown(
    raw: Mapping[str, Any], known: tuple[str, ...], where: str, errors: list[str]
) -> None:
    errors.extend(
        f"[{where}]: unknown key {key!r}; it takes " + ", ".join(known) + "."
        for key in raw
        if key not in known
    )


def _read_numbers(
    raw: Mapping[str, Any], keys: tuple[str, ...], where: str, errors: list[str]
) -> dict[str, float]:
    numbers: dict[str, float] = {}
    for key in keys:
        if key in raw:
            number = _number(raw[key], f"{where}.{key}", errors)
            if number is not None:
                numbers[key] = number
    return numbers


def _number(value: Any, name: str, errors: list[str]) -> float | None:
    # TOML booleans are Python bools, which are ints: `true` is a mistake,
    # not 1. TOML also has inf and nan, which none of these values can be.
    if isinstance(value, bool) or not isinstance(value, int | float):
        errors.append(f"{name} = {value!r} is not a number.")
        return None
    if not math.isfinite(value):
        errors.append(f"{name} = {value!r} is not a finite number.")
        return None
    return float(value)


def _replace(
    default: _Config, changes: Mapping[str, Any], where: str, errors: list[str]
) -> _Config:
    # Each dataclass checks its own values, so a value it refuses is
    # reported with its reason, and the default stands in for the rest.
    if not changes:
        return default
    try:
        return dataclasses.replace(default, **changes)  # type: ignore[type-var]
    except ValueError as e:
        errors.append(f"[{where}]: {e}")
        return default
