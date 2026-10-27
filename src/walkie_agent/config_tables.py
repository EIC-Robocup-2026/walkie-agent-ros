"""The readers of config.toml's [walkie] tables, which type each key by its dataclass field.

A table sets fields of a frozen dataclass, and read_keys() reads each key as
its field is annotated:

    float              a number, as a float; true, inf and nan are refused
    int                an integer; true, 1.5 and 2.0 are refused
    bool               true or false; 1 is refused
    str                a string
    tuple[str, ...]    a list of strings, as a tuple
    float | None       a number, as for float; a missing key keeps None

A missing key keeps its field's default. Each dataclass checks its own
values in __post_init__, and replace() reports a value the dataclass
refuses, with its reason. The readers add each problem to a list instead of
raising, so that one TuningError can list every problem in the file.

Only the standard library is imported, so importing this module loads
nothing of the agent. walkie_agent.tuning, which reads the agent's tables
with it, imports the gaze loop and perception, and through them ROS.
"""

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, get_type_hints

_Config = TypeVar("_Config")
# A key's reader: its value, its dotted name for the messages, and the
# list of problems it adds to. It returns None for a value it refuses.
_Reader = Callable[[Any, str, list[str]], Any]


class TuningError(ValueError):
    """The walkie tables of config.toml are wrong. `errors` holds every problem found."""

    def __init__(self, source: str, errors: list[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        count = "1 problem" if len(errors) == 1 else f"{len(errors)} problems"
        super().__init__(f"{source}: {count}:\n" + "\n".join(f"- {e}" for e in errors))


def table(parent: Mapping[str, Any], name: str, errors: list[str]) -> Mapping[str, Any]:
    """The table `name`, whose last dotted part is its key in `parent`; empty when missing."""
    value = parent.get(name.rpartition(".")[2])
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"{name} = {value!r} is not a table; write it as [{name}].")
        return {}
    return value


def refuse_unknown(
    raw: Mapping[str, Any], known: tuple[str, ...], where: str, errors: list[str]
) -> None:
    """Refuse each key of the table `where` that is not one of `known`."""
    errors.extend(
        f"[{where}]: unknown key {key!r}; it takes " + ", ".join(known) + "."
        for key in raw
        if key not in known
    )


def read_keys(
    config: Any, raw: Mapping[str, Any], keys: tuple[str, ...], where: str, errors: list[str]
) -> dict[str, Any]:
    """The values that the table `where` sets for `keys`, each read as its field is typed.

    `config` is the dataclass, or an instance of it, whose fields the keys
    name. A key the table leaves out is not in the result, so its field
    keeps its default. A key that is not a field, or a field of a type
    without a reader, is a mistake in the code rather than in config.toml,
    and raises TypeError whether or not the table sets the key.
    """
    readers = _readers(config, keys)
    values: dict[str, Any] = {}
    for key in keys:
        if key in raw:
            value = readers[key](raw[key], f"{where}.{key}", errors)
            if value is not None:
                values[key] = value
    return values


def number(value: Any, name: str, errors: list[str]) -> float | None:
    """`value` as a float, or None when it is not a finite number."""
    # TOML booleans are Python bools, which are ints: `true` is a mistake,
    # not 1. TOML also has inf and nan, which no key's value can be.
    if isinstance(value, bool) or not isinstance(value, int | float):
        errors.append(f"{name} = {value!r} is not a number.")
        return None
    if not math.isfinite(value):
        errors.append(f"{name} = {value!r} is not a finite number.")
        return None
    return float(value)


def replace(
    default: _Config, changes: Mapping[str, Any], where: str, errors: list[str]
) -> _Config:
    """`default` with `changes`, or `default` itself when its dataclass refuses them."""
    # Each dataclass checks its own values, so a value it refuses is
    # reported with its reason, and the default stands in for the rest.
    if not changes:
        return default
    try:
        return dataclasses.replace(default, **changes)  # type: ignore[type-var]
    except ValueError as e:
        errors.append(f"[{where}]: {e}")
        return default


def _integer(value: Any, name: str, errors: list[str]) -> int | None:
    # `true` is a bool, which Python counts as an int. TOML writes an
    # integer without a point, so 1.5 and 2.0 are both refused.
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{name} = {value!r} is not an integer.")
        return None
    return value


def _boolean(value: Any, name: str, errors: list[str]) -> bool | None:
    # Only TOML's true and false: 1 or "yes" may not mean what was meant.
    if not isinstance(value, bool):
        errors.append(f"{name} = {value!r} is not true or false.")
        return None
    return value


def _text(value: Any, name: str, errors: list[str]) -> str | None:
    if not isinstance(value, str):
        errors.append(f"{name} = {value!r} is not a string.")
        return None
    return value


def _strings(value: Any, name: str, errors: list[str]) -> tuple[str, ...] | None:
    # A tuple, as the frozen dataclass holds it, not TOML's list.
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{name} = {value!r} is not a list of strings.")
        return None
    return tuple(value)


# The reader of each field annotation. A refused value reads as None, which
# no key can set: TOML has no null, so a `float | None` field is None only
# while its key is left out.
_READERS: dict[Any, _Reader] = {
    float: number,
    int: _integer,
    bool: _boolean,
    str: _text,
    tuple[str, ...]: _strings,
    float | None: number,
}


def _readers(config: Any, keys: tuple[str, ...]) -> dict[str, _Reader]:
    """The reader of each of `keys`, by the annotation of its field of `config`."""
    cls = config if isinstance(config, type) else type(config)
    fields = {f.name for f in dataclasses.fields(cls)}
    hints = get_type_hints(cls)
    readers: dict[str, _Reader] = {}
    for key in keys:
        if key not in fields:
            raise TypeError(f"{cls.__name__} has no field {key!r} for a key to set.")
        reader = _READERS.get(hints[key])
        if reader is None:
            raise TypeError(
                f"{cls.__name__}.{key} is a {hints[key]!r}, which no key can set; the keys "
                "set float, int, bool, str, tuple[str, ...] and float | None fields."
            )
        readers[key] = reader
    return readers
