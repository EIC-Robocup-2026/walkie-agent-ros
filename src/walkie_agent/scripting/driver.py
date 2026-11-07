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

Spending a budget ends the script with an error no script can catch, and so does a
stop, through the cancel token a run may be given. A run a stop ended has phase
cancelled, whether or not the script caught the error of the call the stop ended,
and it writes no run or fail event to the shelf's ledger.
"""

import ast
import contextlib
import inspect
import json
import logging
import time
import traceback
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

from walkie_agent.adapter import bind
from walkie_agent.capabilities import CancelToken, Ctx
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.scripting.evidence import Watch

_logger = logging.getLogger(__name__)

# Strict, as in adapter.as_tool: otherwise pydantic turns true into 1.0 before a
# guard sees it. An int still counts as a float.
_STRICT = ConfigDict(strict=True)
_ERROR_CHARS = 2000
_OUTPUT_CHARS = 4000
_ARGUMENT_CHARS = 500
_STOPPED = f"{ErrorCode.CANCELLED}: a stop ended the script."
# The phases a script that ran can end in. Once a stop is in effect, each of them
# counts as the stop's. A type or syntax error comes before the script runs at all.
_ENDED_BY_A_STOP = frozenset({"runtime", "budget", "crash"})


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
    """What one run did. `phase` is ok, type_check, syntax, runtime, budget, cancelled or crash.

    `shelf_used` and `shelf_raised` name the shelf functions the run entered and
    those an exception crossed, as evidence.Watch saw at run time. `returned` and
    `raised` hold the same evidence by name and source sha, which is what the scorer
    verifies from. `returned` covers a shelf function or a definition this run saved.
    `raised` also covers a definition the run did not save, since the same text may be
    on the shelf. `unwatched` lists, as (name, sha) pairs, every definition no wrapper
    could watch, which the scorer counts as raised. A name may appear in it more than
    once, one pair per version. A sha is of the version's text, as the ledger keeps it.
    `shapes` maps the sha of each version the run named, a loaded shelf function or,
    while the shelf can still verify, one of the script's own definitions, to its shape
    (shelf.shape), which the scorer's veto compares, so a copy at another depth or laid
    out another way counts as the same version. A frozen or shelf-less run watches no own
    definitions (M51), so it maps none of them.
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
    unwatched: list[tuple[str, str]] = field(default_factory=list)
    shapes: dict[str, str] = field(default_factory=dict)
    saved: list[str] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)


class _WallClockSpent(Exception):
    """The script outlived its wall-clock budget between two host calls."""


class _Cancelled(Exception):
    """A stop or a reset set the run's cancel token."""


class _Unservable(Exception):
    """The worker suspended in a way the driver does not serve, such as on a future."""


def _with_newlines(code: str) -> str:
    """`code` with each line ending as `\\n`, the way Python reads a source file.

    Monty keeps a `\\r` inside a triple-quoted string, and numbers a lone-`\\r` script as
    one line. The shelf saves a definition with `\\n` endings, so without this a saved
    version could hold another string than the one that ran under its sha (M62).
    """
    return code.replace("\r\n", "\n").replace("\r", "\n")


def _parse_error(code: str) -> str | None:
    """Why Python's parser refuses `code`, or None.

    The evidence reads the script's definitions with Python's parser. Monty runs some
    source that parser refuses, such as a null byte, and such a script would run with
    none of its definitions watched, so it does not run at all (M63). The message is
    Python's own SyntaxError, which names the file and line when Python gives them.
    A string that cannot be encoded as UTF-8, such as one holding a lone surrogate, is
    refused the same way (M67).
    """
    try:
        ast.parse(code, filename="main.py")
    except SyntaxError as exc:
        return "".join(traceback.format_exception_only(exc)).rstrip()
    except UnicodeEncodeError as exc:
        return f"SyntaxError: the script is not valid UTF-8 ({exc.reason})."
    return None


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
        cancel: CancelToken | None = None,
    ) -> ScriptResult:
        """Run `code` with `fns` callable. After a success, keep the definitions named in `save`.

        The script runs, is read and is saved with `\\n` line endings, and only if
        Python's parser accepts it and the evidence can read it. A refusal is a result
        like any failure, with no host call made and nothing saved.

        Once `cancel` is set, the driver runs no further host call and ends the script
        as it ends one out of wall clock. A host call already running is not
        interrupted: the stop that set the token ends it, as it ends the same call made
        as a tool, and its CANCELLED reaches the script. The run's phase is then
        "cancelled" whatever the script does with that error: lets it end the script,
        catches it and calls again, or catches it and loops without a call until the
        feed budget ends it. Only a type or syntax error, found before the script runs,
        keeps its own phase.
        """
        code = _with_newlines(code)
        result = ScriptResult()
        may_save = self.shelf is not None and not self.shelf.frozen
        # The host parses the script before Monty's budgets apply, so a parser or the
        # evidence running out of stack or memory is a refusal, not an exception (M67).
        refusal: tuple[str, str] | None = None
        try:
            if (unparsable := _parse_error(code)) is not None:
                refusal = ("syntax", unparsable)
            else:
                prefix, shelf_versions = self.shelf.source_for(code) if self.shelf is not None else ("", {})
                # Own definitions are watched while the shelf can still verify, saved or not.
                watch = Watch(code, shelf_versions, shelf_source=prefix, watch_own=may_save, reserved=set(fns))
        except RecursionError:
            refusal = ("syntax", "RecursionError: the script nests too deeply for the host to read it.")
        except MemoryError:
            refusal = ("budget", "MemoryError: the host ran out of memory while it read the script.")
        if refusal is not None:
            result.ok, result.phase = False, refusal[0]
            result.error = _clip(refusal[1])
            if save:
                reason = "The script did not succeed." if self.shelf is not None else "This run has no shelf."
                result.refused = dict.fromkeys(save, reason)
            return result

        def fail(phase: str, error: str) -> None:
            result.ok, result.phase, result.error = False, phase, _clip(watch.clean(error))

        try:
            self._drive(prefix, fns, result, watch, cancel)
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
        except _Cancelled:
            fail("cancelled", _STOPPED)
        except _Unservable as exc:
            fail("runtime", str(exc))
        except (MontyError, TimeoutError) as exc:
            fail("crash", f"{type(exc).__name__}: {exc}")
        # The stop's CANCELLED reaches the script through the call it ended. A script that
        # does not catch it dies of it, and one that catches it and then loops without a
        # call runs out of feed time. Either way the stop ended the script.
        if cancel is not None and cancel.cancelled and result.phase in _ENDED_BY_A_STOP:
            fail("cancelled", _STOPPED)
        for call in result.calls:
            call["tool"] = watch.clean(call["tool"])
        used = {name: sha for name, sha in shelf_versions.items() if name in watch.entered}
        result.shelf_used = sorted(used)
        result.shelf_raised = sorted(watch.raised & set(used))
        if self.shelf is not None:
            # A stop ended the run, not the functions in it, and they did not finish
            # either, so the ledger counts the run neither for nor against them (T36).
            if result.phase != "cancelled":
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
        result.unwatched = watch.unwatched
        result.shapes = watch.shapes
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
        cancel: CancelToken | None,
    ) -> None:
        def cancelled() -> bool:
            return cancel is not None and cancel.cancelled

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
                        if cancelled():
                            self._unwind(snapshot, RuntimeError(f"{ErrorCode.CANCELLED}: a stop is in effect."))
                            raise _Cancelled
                        if time.monotonic() > deadline:
                            self._unwind(snapshot, TimeoutError("Out of wall clock."))
                            raise _WallClockSpent
                        snapshot = self._answer(snapshot, checked, result.calls)
                    # A stop or a last slow call can land with no suspension left to catch it.
                    if cancelled():
                        raise _Cancelled
                    if time.monotonic() > deadline:
                        raise _WallClockSpent
                    result.ok, result.phase = True, "ok"
                    result.output = _jsonable(snapshot.output, _OUTPUT_CHARS)
                finally:
                    watch.read(session)
        finally:
            result.printed = watch.clean(printed.output)

    def _unwind(self, snapshot: Any, refusal: Exception) -> None:
        """Let a script that has to end finish, with every further call refused and no host call run.

        A script ends this way out of wall clock or after a stop, and each call it still
        makes raises `refusal`. The worker takes no other feed while a suspension waits,
        so this is what keeps the run's counters readable. A script that dawdles past
        another feed's worth of time, catching the refusals, is left suspended and its
        evidence is lost.
        """
        stop = time.monotonic() + self.budgets.feed_s
        with contextlib.suppress(MontyError):
            while time.monotonic() < stop:
                if isinstance(snapshot, NameLookupSnapshot):
                    snapshot = snapshot.resume()
                elif isinstance(snapshot, FunctionSnapshot):
                    snapshot = snapshot.resume({"exception": refusal})
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


def script_record(code: str, result: ScriptResult) -> dict[str, Any]:
    """One script's record for whoever verifies after the task, which the model never sees."""
    return {
        "code": code,
        "ok": result.ok,
        "phase": result.phase,
        "error": result.error,
        "calls": result.calls,
        "shelf_used": result.shelf_used,
        "shelf_raised": result.shelf_raised,
        "returned": result.returned,
        "raised": result.raised,
        "unwatched": [list(version) for version in result.unwatched],
        "shapes": result.shapes,
        "saved": result.saved,
        "refused": result.refused,
    }


def proven(scripts: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Per name, the last version that returned in a task and that nothing raised through in it.

    `scripts` holds the script_record of each script of one task. A version that ran
    where no wrapper watched it counts as raised. Versions match by name and shape,
    from each script's own `shapes`, so a raise through a copy at another depth or laid
    out another way counts against the version it copies. They also match by name and
    sha, which is all a record from before `shapes` existed, or a sha missing from its
    record's shapes, has to go on. A record from before `unwatched` existed has none.
    What is proven is the text sha, as the ledger keeps it.
    """

    def keys(script: Mapping[str, Any], name: str, sha: str) -> set[tuple[str, str, str]]:
        shape = script.get("shapes", {}).get(sha)
        return {("sha", name, sha)} | ({("shape", name, shape)} if shape else set())

    raised: set[tuple[str, str, str]] = set()
    for script in scripts:
        for name, sha in [*script["raised"].items(), *script.get("unwatched", ())]:
            raised |= keys(script, name, sha)
    versions: dict[str, str] = {}
    for script in scripts:
        for name, sha in script["returned"].items():
            if not keys(script, name, sha) & raised:
                versions[name] = sha
    return versions


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
