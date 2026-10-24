"""JSON Lines logs: the HRI scenario's event log and the agent's tool-call record.

One JSON object per line, flushed as written, so a run that is killed keeps
every complete line. No ROS imports: the pass check reads these logs without
a ROS install.
"""

import json
import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class EventLog:
    """A JSON Lines log, one object per line, flushed as written.

    Thread-safe. `t` is rounded to the millisecond, and a `t` that is not a
    finite number is written as null. The record of type "end" is the last
    one: anything written after it is dropped, as is anything written after
    close().
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self._ended = False

    def write(self, record: Mapping[str, Any]) -> None:
        record = dict(record)
        t = record.get("t")
        record["t"] = round(t, 3) if isinstance(t, (int, float)) and math.isfinite(t) else None
        line = json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False)
        with self._lock:
            if self._ended or self._file.closed:
                return
            self._file.write(line + "\n")
            self._file.flush()
            self._ended = record.get("type") == "end"

    def close(self) -> None:
        with self._lock:
            self._file.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """The complete lines of a JSON Lines log, read while it may still be written.

    A last line without its newline is dropped: it was cut off, by a killed
    process or by a writer that has not finished it yet.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n")]


def _json_safe(value: Any) -> Any:
    """`value` with tuples as lists, paths as text and every non-finite number as None."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None
