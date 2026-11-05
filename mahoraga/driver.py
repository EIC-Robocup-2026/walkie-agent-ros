"""run_script's driver: a deny-by-default host for scripts on a Monty worker.

A script runs in a Monty worker and suspends at every call it cannot serve
itself. The driver answers a suspension only for the functions on the allow-list,
each validated strictly and with ctx already bound. Everything else, including
any OS call, gets PermissionError or NameError. So a script reaches no ctx, no
file, no network and no unknown function, whatever it type-checks to.

How a failed call reaches the script:

- StepError: RuntimeError("CODE: message"), which the script may catch.
- Bad arguments: TypeError("tool: argument: reason").
- A call off the allow-list: PermissionError.
- A bug in the host function: RuntimeError("INTERNAL: tool failed with Type."),
  and the traceback goes to the host's log, as in adapter.format_tool_error.

Spending a budget ends the script with an error no script can catch.
"""

import contextlib
import inspect
import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ConfigDict, ValidationError, validate_call
from pydantic_monty import (
    CollectString,
    FunctionSnapshot,
    Monty,
    MontyComplete,
    MontyCrashedError,
    MontyError,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
    NameLookupSnapshot,
)

from mahoraga.evidence import Watch
from mahoraga.shelf import top_level_defs
from walkie_agent.adapter import bind
from walkie_agent.capabilities import Ctx
from walkie_agent.errors import ErrorCode, StepError

_logger = logging.getLogger(__name__)

# Strict, as in adapter.as_tool: otherwise pydantic turns true into 1.0 before a
# guard sees it. An int still counts as a float.
_STRICT = ConfigDict(strict=True)
_ERROR_CHARS = 2000
_OUTPUT_CHARS = 4000
_ARGUMENT_CHARS = 500


@dataclass(frozen=True)
class Budgets:
    """What one script may spend, per the plan's runner probes."""

    max_calls: int = 40
    feed_s: float = 2.0
    wall_s: float = 30.0
    memory_bytes: int = 64 * 1024 * 1024
    print_bytes: int = 4000


@dataclass
class ScriptResult:
    """What one run did. `phase` is ok, type_check, syntax, runtime, budget or crash.

    `shelf_used` and `shelf_raised` name the shelf functions the run entered and
    those an exception crossed, as evidence.Watch saw at run time. `returned` and
    `raised` hold the same evidence by name and source sha, which is what the scorer
    verifies from. `returned` covers a shelf function or a definition this run saved.
    `raised` also covers a definition the run did not save, and one no wrapper could
    watch, since the same text may be on the shelf.
    """

    ok: bool = False
    phase: str = "crash"
    output: Any = None
    printed: str = ""
    error: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    shelf_used: list[str] = field(default_factory=list)
    shelf_raised: list[str] = field(default_factory=list)
    returned: dict[str, str] = field(default_factory=dict)
    raised: dict[str, str] = field(default_factory=dict)
    saved: list[str] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)


class _WallClockSpent(Exception):
    """The script outlived its wall-clock budget between two host calls."""


class _Unservable(Exception):
    """The worker suspended in a way the driver does not serve, such as on a future."""


def allow_list(atomics: Iterable[Callable[..., Any]], ctx: Ctx) -> dict[str, Callable[..., Any]]:
    """The functions a script may call, by name, with ctx bound. Raises ValueError for a name used twice."""
    fns: dict[str, Callable[..., Any]] = {}
    for fn in atomics:
        if fn.__name__ in fns:
            raise ValueError(f"Function name used twice: {fn.__name__}.")
        fns[fn.__name__] = bind(fn, ctx)
    return fns


def stubs_for(fns: Mapping[str, Callable[..., Any]]) -> str:
    """Type-check stubs for the allow-list, in the same text the prompt shows."""
    lines = ["from typing import Any"]
    lines += [f"def {_signature(name, fn)}: ..." for name, fn in fns.items()]
    return "\n".join(lines) + "\n"


def describe_functions(fns: Mapping[str, Callable[..., Any]]) -> str:
    """Each function's signature and docstring, for the tool description."""
    return "\n\n".join(f"{_signature(name, fn)}\n{inspect.getdoc(fn)}" for name, fn in fns.items())


def _signature(name: str, fn: Callable[..., Any]) -> str:
    return f"{name}{inspect.signature(fn)}".replace("typing.", "")


class ScriptRunner:
    """Runs scripts on one Monty pool, with the shelf's functions fed in first.

    Safe to share between threads. Each run takes a fresh worker session, so no
    state crosses from one script to the next.
    """

    def __init__(self, pool: Monty, *, shelf: Any = None, budgets: Budgets = Budgets()) -> None:
        self.pool = pool
        self.shelf = shelf
        self.budgets = budgets

    def run(
        self,
        code: str,
        fns: Mapping[str, Callable[..., Any]],
        *,
        save: Sequence[str] = (),
    ) -> ScriptResult:
        """Run `code` with `fns` callable. After a success, keep the definitions named in `save`."""
        result = ScriptResult()
        prefix, shelf_versions = self.shelf.source_for(code) if self.shelf is not None else ("", {})
        may_save = self.shelf is not None and not self.shelf.frozen
        # Every own definition is watched while the shelf can still verify, saved or not.
        watch = Watch(code, shelf_versions, top_level_defs(code) if may_save else (), set(fns))

        def fail(phase: str, error: str) -> None:
            result.ok, result.phase, result.error = False, phase, _clip(watch.clean(error))

        try:
            self._drive(prefix, fns, result, watch)
        except MontyTypingError as exc:
            fail("type_check", exc.display())
        except MontySyntaxError as exc:
            fail("syntax", exc.display("type-msg"))
        except MontyRuntimeError as exc:
            fail("budget" if _spent_a_budget(exc) else "runtime", exc.display())
        except MontyCrashedError as exc:
            fail("budget" if exc.timed_out else "crash", str(exc))
        except _WallClockSpent:
            fail("budget", f"TimeoutError: over {self.budgets.wall_s} s of wall clock.")
        except _Unservable as exc:
            fail("runtime", str(exc))
        except (MontyError, TimeoutError) as exc:
            fail("crash", f"{type(exc).__name__}: {exc}")
        for call in result.calls:
            call["tool"] = watch.clean(call["tool"])
        used = {name: sha for name, sha in shelf_versions.items() if name in watch.entered}
        result.shelf_used = sorted(used)
        result.shelf_raised = sorted(watch.raised & set(used))
        if self.shelf is not None:
            self.shelf.record_run(used, result.ok)
            if save and result.ok:
                result.saved, result.refused = self.shelf.save(
                    code,
                    save,
                    reserved=set(fns),
                    check=lambda source: self._type_error_of(source, fns),
                )
            elif save:
                result.refused = {name: "The script did not succeed." for name in save}
        elif save:
            result.refused = {name: "This run has no shelf." for name in save}
        kept = {name: sha for name, sha in watch.own_versions.items() if name in result.saved}
        kept.update(shelf_versions)
        result.returned = {name: sha for name, sha in kept.items() if name in watch.returned}
        # A raise counts whether or not this script saved the version: the same text may
        # already be on the shelf, and a return in another script must not verify it.
        ran = {**watch.own_versions, **shelf_versions}
        result.raised = {name: sha for name, sha in ran.items() if name in watch.raised}
        result.raised.update(watch.unseen_versions)
        return result

    def _session(self, fns: Mapping[str, Callable[..., Any]], extra_stubs: str = "") -> Any:
        return self.pool.checkout(
            limits={
                "max_feed_duration_secs": self.budgets.feed_s,
                "max_memory": self.budgets.memory_bytes,
                # Monty counts each host call and name lookup as one suspension,
                # and ends the feed at the limit in a way no script can catch.
                "max_suspensions": self.budgets.max_calls,
            },
            type_check=True,
            type_check_stubs=stubs_for(fns) + extra_stubs,
            type_check_format="concise",
            # A sleep returns at once, so a script cannot stall its episode.
            os_policy={"sleep": "zero"},
        )

    def _drive(
        self,
        prefix: str,
        fns: Mapping[str, Callable[..., Any]],
        result: ScriptResult,
        watch: Watch,
    ) -> None:
        deadline = time.monotonic() + self.budgets.wall_s
        checked = {name: validate_call(config=_STRICT)(fn) for name, fn in fns.items()}
        printed = CollectString(max_bytes=self.budgets.print_bytes)
        try:
            with self._session(fns, watch.stubs) as session:
                # A separate feed keeps the script's own line numbers in its errors.
                # Monty type-checks each feed against the earlier ones, and the loader
                # stays out of that, so a script's calls still check against each
                # function's real signature.
                if prefix:
                    session.feed_run(prefix, print_callback=printed)
                if watch.loader:
                    session.feed_run(watch.loader, print_callback=printed, skip_type_check=True)
                try:
                    snapshot = session.feed_start(watch.code, print_callback=printed)
                    while not isinstance(snapshot, MontyComplete):
                        if time.monotonic() > deadline:
                            self._unwind(snapshot)
                            raise _WallClockSpent
                        snapshot = self._answer(snapshot, checked, result.calls)
                    # A last slow call overruns with no suspension left to catch it.
                    if time.monotonic() > deadline:
                        raise _WallClockSpent
                    result.ok, result.phase = True, "ok"
                    result.output = _jsonable(snapshot.output, _OUTPUT_CHARS)
                finally:
                    watch.read(session)
        finally:
            result.printed = watch.clean(printed.output)

    def _unwind(self, snapshot: Any) -> None:
        """Let a script out of wall clock finish, with every further call refused and no host call run.

        The worker takes no other feed while a suspension waits, so this is what keeps
        the run's counters readable. A script that dawdles past another feed's worth of
        time, catching the refusals, is left suspended and its evidence is lost.
        """
        stop = time.monotonic() + self.budgets.feed_s
        with contextlib.suppress(MontyError):
            while time.monotonic() < stop:
                if isinstance(snapshot, NameLookupSnapshot):
                    snapshot = snapshot.resume()
                elif isinstance(snapshot, FunctionSnapshot):
                    snapshot = snapshot.resume({"exception": TimeoutError("Out of wall clock.")})
                else:
                    return

    def _answer(
        self,
        snapshot: Any,
        checked: Mapping[str, Callable[..., Any]],
        calls: list[dict[str, Any]],
    ) -> Any:
        """Answer one suspension and return the next one. Anything off the allow-list is refused."""
        if isinstance(snapshot, NameLookupSnapshot):
            # Resuming with no value raises NameError in the script, so a script
            # can call an allowed function but never hold it as a value.
            calls.append({"tool": snapshot.variable_name, "status": "unknown_name"})
            return snapshot.resume()
        if not isinstance(snapshot, FunctionSnapshot):
            raise _Unservable(f"{type(snapshot).__name__}: only plain synchronous calls are served.")
        name = str(snapshot.function_name)
        if snapshot.is_os_function or name not in checked:
            calls.append({"tool": name, "status": "denied"})
            return snapshot.resume({"exception": PermissionError(f"{name} is not allowed.")})
        entry: dict[str, Any] = {
            "tool": name,
            "args": _jsonable(list(snapshot.args), _ARGUMENT_CHARS),
            "kwargs": _jsonable(dict(snapshot.kwargs), _ARGUMENT_CHARS),
        }
        calls.append(entry)
        try:
            value = checked[name](*snapshot.args, **snapshot.kwargs)
        except StepError as exc:
            entry["status"] = str(exc.code)
            return snapshot.resume({"exception": RuntimeError(str(exc))})
        except ValidationError as exc:
            entry["status"] = "invalid"
            return snapshot.resume({"exception": TypeError(_invalid(name, checked[name], exc))})
        except Exception as exc:
            _logger.error("Script call %s failed.", name, exc_info=exc)
            entry["status"] = str(ErrorCode.INTERNAL)
            text = f"{ErrorCode.INTERNAL}: {name} failed with {type(exc).__name__}."
            return snapshot.resume({"exception": RuntimeError(text)})
        entry["status"] = "ok"
        return snapshot.resume({"return_value": value})

    def _type_error_of(self, source: str, fns: Mapping[str, Callable[..., Any]]) -> str | None:
        """Why definitions cannot stand alone, for example a call to an unsaved helper."""
        prefix = self.shelf.source_for(source)[0] if self.shelf is not None else ""
        try:
            with self._session(fns) as session:
                session.feed_run(f"{prefix}\n{source}", print_callback=CollectString(max_bytes=0))
        except MontyTypingError as exc:
            return _clip(exc.display())
        except MontyError as exc:
            return _clip(str(exc))
        return None


def _spent_a_budget(exc: MontyRuntimeError) -> bool:
    inner = exc.exception()
    return isinstance(inner, TimeoutError | MemoryError | RecursionError) or "suspension limit" in str(inner)


def script_summary(result: ScriptResult) -> dict[str, Any]:
    """What the model sees of a run. The ledger stays with the harness."""
    summary: dict[str, Any] = {
        "ok": result.ok,
        "output": result.output,
        "printed": result.printed,
        "calls_made": len(result.calls),
    }
    if result.error is not None:
        summary["error"] = result.error
    if result.saved:
        summary["saved"] = result.saved
    if result.refused:
        summary["not_saved"] = result.refused
    return summary


def _invalid(name: str, fn: Callable[..., Any], exc: ValidationError) -> str:
    """One line per bad argument. Pydantic gives a positional argument's index, so name it."""
    parameters = list(inspect.signature(fn).parameters)

    def label(part: str | int) -> str:
        return parameters[part] if isinstance(part, int) and part < len(parameters) else str(part)

    reasons = [
        f"{'.'.join(label(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors(include_url=False, include_input=False)
    ]
    return f"{name}: {'; '.join(reasons)}"


def _jsonable(value: Any, limit: int) -> Any:
    """`value` as JSON data, or as clipped text if it is not JSON or too long to pass on."""
    try:
        text = json.dumps(value, default=repr, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return _clip(repr(value), limit)
    return _clip(text, limit) if len(text) > limit else json.loads(text)


def _clip(text: str, limit: int = _ERROR_CHARS) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... [cut]"
