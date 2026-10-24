"""The agent's tool-call record: one JSON line per tool call the LLM makes.

The HRI pass check reads it beside the scenario's event log (hri.md, "The
pass check comes with the template"). Both stamp `t` on the node clock, which
follows /clock under sim time, so the two logs line up.

Records, each with `type` and `t`, the node clock in seconds:

- `start`: written when the log opens; `wall_time` (ISO 8601).
- `new_task`: a new task started, so the handles and the people memory were
  cleared.
- `call`: one tool call, written when it ends. `t` and `t_end` bound it on
  the node clock, `wall_s` is its duration on the monotonic clock, since the
  sim may run slower than real time. `tool`, `call_id` and `args` are the
  LLM's call as it made it. `outcome` is `ok`, `error`, `invalid` (the
  arguments did not validate) or `not_run` (answered without running).
  `code` is the error code of an atomic's "CODE: message", else null.
  `result` is what the LLM read, or for an exception its type and text.
  `out` is an atomic's full Out on success, else null. `people` lists every
  person or seat handle named in `args` or in `result`, set ids expanded to
  their members, with the track behind each. `gaze` is what the gaze loop
  was doing when a `speak` or `wait_for_person` call started, else null.

WalkieToolRunner writes the records; every call the LLM makes passes it. The
log never raises into the agent: a failure is logged once and the call goes
on.
"""

import logging
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from walkie_agent.jsonl import EventLog
from walkie_agent.toolbox.handles import HandleStore
from walkie_agent.toolbox.types import ERROR_CODES, ToolOutput

if TYPE_CHECKING:
    from walkie_agent.gaze import GazeLoop

logger = logging.getLogger(__name__)

HANDLE_PATTERN = re.compile(r"\b(?:per|seat)_[0-9a-f]{4}(?:\.\d{1,3})?\b")
_CODE = re.compile(r"^([A-Z_]+): ")
# The tools that speak to a person, whose record says where the robot looked.
SPEAKING_TOOLS = frozenset({"speak", "wait_for_person"})


@dataclass
class PendingCall:
    """One call while it runs: its start, and what the tool hands back on the side."""

    tool: str
    call_id: str
    args: Any
    t: float | None
    started: float
    gaze: dict[str, Any] | None
    # Set once the runner hands the call to its tool. A call answered
    # without running, such as one after a question, keeps False.
    ran: bool = False
    out: ToolOutput | None = None


# The recorded call that runs in this context; WalkieToolRunner sets it
# around each call. LangChain runs a tool's _run in a copy of the context,
# where a set() would not reach the runner, so the var holds a PendingCall
# that the tool fills in place.
CURRENT_CALL: ContextVar[PendingCall | None] = ContextVar("walkie_tool_call", default=None)


def hand_out(out: ToolOutput) -> None:
    """Give the running call's Out to the record. Does nothing outside a recorded call.

    AtomicTool.call calls it, since the LLM only reads the summary. An
    atomic that runs another atomic through call() hands both; the outer one
    ends last, so its Out is kept.
    """
    pending = CURRENT_CALL.get()
    if pending is not None:
        pending.out = out


def mark_run() -> None:
    """Note that the running call reached its tool. Does nothing outside a recorded call."""
    pending = CURRENT_CALL.get()
    if pending is not None:
        pending.ran = True


class ToolLog:
    """Writes the tool-call record to `path` (module docstring).

    `clock` gives the node clock in seconds, `handles` resolves the handles
    in a call, and `gaze`, when given, says what the gaze loop was doing when
    the robot spoke. Thread-safe.
    """

    def __init__(
        self,
        path: str | Path,
        clock: Callable[[], float],
        handles: HandleStore,
        gaze: "GazeLoop | None" = None,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self._handles = handles
        self._gaze = gaze
        self._log = EventLog(self.path)
        self._lock = threading.Lock()
        self._failed = False
        self._safely(
            lambda: self._log.write(
                {
                    "type": "start",
                    "t": self._clock(),
                    "wall_time": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            )
        )

    def new_task(self) -> None:
        """Note a new task. Registered with Toolbox.on_new_task."""
        self._safely(lambda: self._log.write({"type": "new_task", "t": self._clock()}))

    def begin(self, tool: str, call: Mapping[str, Any]) -> PendingCall:
        """Start the record of `call`. Never raises: what cannot be read is left null."""
        pending = PendingCall(
            tool=tool,
            call_id=str(call.get("id") or ""),
            args=call.get("args"),
            t=self._safely(self._clock),
            started=time.monotonic(),
            gaze=None,
        )
        if tool in SPEAKING_TOOLS and self._gaze is not None:
            pending.gaze = self._safely(self._gaze_now)
        return pending

    def end(self, pending: PendingCall, output: Any, error: BaseException | None) -> None:
        """Write the record of a call that returned `output` or raised `error`."""
        self._safely(lambda: self._log.write(self._record(pending, output, error)))

    def close(self) -> None:
        self._safely(self._log.close)

    def _record(
        self, pending: PendingCall, output: Any, error: BaseException | None
    ) -> dict[str, Any]:
        wall_s = time.monotonic() - pending.started
        if error is not None:
            outcome = "invalid" if isinstance(error, ValidationError) else "error"
            result = f"{type(error).__name__}: {error}"
        elif isinstance(output, ToolMessage):
            if not pending.ran:
                outcome = "not_run"
            else:
                outcome = "error" if output.status == "error" else "ok"
            result = _text(output.content)
        else:
            outcome = "ok" if pending.ran else "not_run"
            result = _text(output)
        code = None
        if outcome == "error":
            match = _CODE.match(result)
            if match is not None and match.group(1) in ERROR_CODES:
                code = match.group(1)
        out = None
        if outcome == "ok" and pending.out is not None:
            out = pending.out.model_dump(mode="json")
        t_end = self._safely(self._clock)
        return {
            "type": "call",
            "t": pending.t,
            "t_end": _round(t_end),
            "wall_s": _round(wall_s),
            "tool": pending.tool,
            "call_id": pending.call_id,
            "args": pending.args,
            "outcome": outcome,
            "code": code,
            "result": result,
            "out": out,
            "people": [
                *self._people(_strings(pending.args), "args"),
                *self._people([result], "result"),
            ],
            "gaze": pending.gaze,
        }

    def _people(self, texts: list[str], where: str) -> list[dict[str, Any]]:
        """Every handle named in `texts`, set ids as their members, each once."""
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for text in texts:
            for handle_id in HANDLE_PATTERN.findall(text):
                handles = self._handles.peek(handle_id)
                if not handles:
                    # An id the store does not know, such as one the LLM
                    # made up, is still named.
                    entries = [_entry(handle_id, where, None, None)]
                else:
                    entries = [
                        _entry(handle.id, where, handle.kind, _track(handle.payload))
                        for handle in handles
                    ]
                for entry in entries:
                    if entry["handle"] not in seen:
                        seen.add(entry["handle"])
                        found.append(entry)
        return found

    def _gaze_now(self) -> dict[str, Any]:
        status = self._gaze.status()
        track_id = None
        if status.target is not None:
            handles = self._handles.peek(status.target)
            track = _track(handles[0].payload) if handles else None
            track_id = None if track is None else track.track_id
        return {"target": status.target, "running": status.running, "track_id": track_id}

    def _safely(self, fn: Callable[[], Any]) -> Any:
        """fn(), or None when it raises. The first failure is logged; later ones are not."""
        try:
            return fn()
        except Exception:
            with self._lock:
                first = not self._failed
                self._failed = True
            if first:
                logger.exception(
                    f"The tool-call record {self.path} failed; later failures are not logged."
                )
            return None


def _entry(handle_id: str, where: str, kind: str | None, track: Any) -> dict[str, Any]:
    return {
        "handle": handle_id,
        "in": where,
        "kind": kind,
        "track_id": None if track is None else track.track_id,
        "generation": None if track is None else track.generation,
    }


def _track(payload: Any) -> Any:
    """A person payload's TrackKey; None for a seat or an untracked person."""
    return getattr(payload, "track", None)


def _strings(value: Any) -> list[str]:
    """Every string in `value`, through nested dicts and lists."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _strings(item)]
    return []


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _round(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 3)
