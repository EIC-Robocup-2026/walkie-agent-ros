"""run_script on the competition agent, behind --scripts: one tool and the operator's verdict.

The tool runs a script over the run's own atomics, bound to the robot's ctx, and the
functions on the shelf. The shelf is writable, so a script may save functions into
temporary/. No scorer runs on the robot path. Only the operator's verdict, about the
task that ended last, verifies what that task proved, by the gym's evidence rule
(driver.proven). Without a verdict nothing is ever verified here. A failed verdict
records the same versions as rejected in the shelf's ledger, and verifies nothing.

A task is one run of the agent, from a message to its reply, so a question and its
answer are two tasks. The Runner calls begin() and end() around each run.
"""

import contextlib
import json
import logging
import threading
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from langchain_core.tools import StructuredTool

from walkie_agent.capabilities import Ctx
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.scripting.driver import (
    Budgets,
    ScriptResult,
    ScriptRunner,
    allow_list,
    describe_functions,
    proven,
    script_record,
    script_summary,
)
from walkie_agent.scripting.sandbox import SandboxUnavailable, open_pool
from walkie_agent.scripting.shelf import Shelf
from walkie_agent.tools.adapter import as_tool

logger = logging.getLogger(__name__)

TOOL_NAME = "run_script"
# Never on a script's allow-list: a script may not start another script, edit the
# todo list, or end the run with a question.
NOT_CALLABLE = frozenset({TOOL_NAME, "update_todos", "wait_for_person"})
# Sized for real motion (decision T19). Nothing interrupts a host call, so the wall
# clock is checked between calls, and each atomic keeps its own bounds. A drive takes
# 15 to 70 s and an arm move about 9 s, so 300 s fits several of them, and a script
# that hangs between calls still ends within feed_s.
AGENT_BUDGETS = Budgets(max_calls=40, feed_s=2.0, wall_s=300.0)
# Scripts run one at a time. The second worker stands in while Monty replaces one
# that a spent budget killed.
POOL_SIZE = 2
UNVERIFIED = "# Unverified: not yet confirmed by the operator, so it may be wrong."
REJECTED = "# Unverified: the operator failed a task that ran this version, so it is likely wrong."

_NOTES = """\
A script is Python, and the functions below are its only way to act. Nothing
carries over from one script to the next. Call the functions in plain loops, not
inside a builtin key= lambda. They return plain lists and dicts, with the keys
their docstrings give.

A failed call raises RuntimeError("CODE: message"), or TypeError for a bad
argument, and your script may catch either. A script may make at most
{budgets.max_calls} calls, run for {budgets.feed_s:g} seconds between calls and
{budgets.wall_s:g} seconds in all, and print {budgets.print_bytes} bytes. A stop
ends it at once. The script's last expression, what it printed and each call it
made come back to you. A script that does not run to its end fails the call.

Functions:"""
_SHELF = """\
Functions that earlier scripts saved. A script may call them directly:"""


class ToolLogLike(Protocol):
    """The part of runtime.middleware.ToolLog that the verdict writes to.

    scripting imports nothing from the runtime: the oracle gate digests every module
    that a scripting module imports, even for type checking only.
    """

    def write(self, **record: Any) -> None: ...


class AgentScripts:
    """The agent's run_script: a runner over the pool, the shelf, and the verdict's record.

    begin(), end(), verdict() and the tool are safe from any thread. close() ends the
    pool through `closer`, as open() sets it; a pool the caller owns has none.
    """

    def __init__(
        self,
        pool: Any,
        shelf: Shelf,
        *,
        budgets: Budgets = AGENT_BUDGETS,
        tool_log: ToolLogLike | None = None,
        closer: Callable[[], None] | None = None,
    ) -> None:
        self.shelf = shelf
        self.budgets = budgets
        self._runner = ScriptRunner(pool, shelf=shelf, budgets=budgets)
        self._tool_log = tool_log
        self._closer = closer
        self._closing: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._current: list[dict[str, Any]] = []
        # The records of the task that ended last, until a verdict takes them.
        self._ended: list[dict[str, Any]] | None = None
        self._tool: StructuredTool | None = None
        self._functions = ""

    @classmethod
    def open(
        cls,
        shelf: Shelf,
        *,
        tool_log: ToolLogLike | None = None,
        budgets: Budgets = AGENT_BUDGETS,
        max_processes: int = POOL_SIZE,
    ) -> "AgentScripts":
        """Start the pool's workers inside bwrap, from the calling thread.

        Call it from a thread that lives as long as the agent, the main thread in
        main(): each worker dies with the thread that started it. Raises
        SandboxUnavailable if bwrap is missing or cannot start a worker, with a message
        for the agent's operator: the sandbox's own one names the gym's --no-bwrap.
        """
        stack = contextlib.ExitStack()
        try:
            pool = stack.enter_context(open_pool(max_processes=max_processes))
        except SandboxUnavailable as exc:
            # open_pool chains the worker's error, and raises without one when bwrap is missing.
            reason = f"bwrap could not start a worker: {exc.__cause__}" if exc.__cause__ else "bwrap is not installed"
            raise SandboxUnavailable(
                f"--scripts runs scripts only inside bwrap, and {reason}. Install bwrap, or start without --scripts."
            ) from exc
        return cls(pool, shelf, budgets=budgets, tool_log=tool_log, closer=stack.close)

    # -- The tool --------------------------------------------------------------

    def tool(self, atomics: Iterable[Callable[..., Any]], ctx: Ctx) -> StructuredTool:
        """The run_script tool over `atomics`, without those in NOT_CALLABLE, bound to `ctx`.

        Its description lists each callable function with its signature and docstring,
        the shelf's last, each unverified one marked. A save or a verdict refreshes it,
        and the next model call sends the new text.
        """
        fns = allow_list([fn for fn in atomics if fn.__name__ not in NOT_CALLABLE], ctx)
        runner, budgets = self._runner, self.budgets

        def run_script(ctx: Ctx, code: str, save: list[str] | None = None) -> dict[str, Any]:
            """Run one Python script that calls the robot's functions, and keep the functions it names in save.

            Returns whether it worked, its output, what it printed, each call it made
            with its status, and which functions it saved.

            Args:
                code: The script's source.
                save: Names of top-level functions in the script to keep for later tasks.
            """
            result = runner.run(code, fns, save=save or (), cancel=ctx.cancel)
            self._record(code, result)
            if result.saved:
                self._refresh()
            summary = {**script_summary(result), "calls": _ledger(result)}
            if result.ok:
                return summary
            text = json.dumps(summary, ensure_ascii=False)
            if result.phase == "cancelled":
                raise StepError(ErrorCode.CANCELLED, f"A stop ended the script: {text}")
            raise StepError(ErrorCode.SKILL_FAILED, f"The script did not run to its end: {text}")

        self._tool = as_tool(run_script, ctx)
        self._functions = f"{self._tool.description}\n\n{_NOTES.format(budgets=budgets)}\n\n{describe_functions(fns)}"
        self._refresh()
        return self._tool

    def _refresh(self) -> None:
        if self._tool is None:
            return
        shelf = self.shelf.describe_marked(UNVERIFIED, REJECTED)
        self._tool.description = f"{self._functions}\n\n{_SHELF}\n\n{shelf}" if shelf else self._functions

    def _record(self, code: str, result: ScriptResult) -> None:
        with self._lock:
            self._current.append(script_record(code, result))

    # -- Tasks and the verdict -------------------------------------------------

    def begin(self) -> None:
        """A task starts: its scripts are recorded from here."""
        with self._lock:
            self._running, self._current = True, []

    def end(self) -> None:
        """The task ended, however it ended. Its record waits for a verdict."""
        with self._lock:
            self._running, self._ended, self._current = False, self._current, []

    def verdict(self, passed: bool) -> str:
        """Judge the task that ended last, once, and return a line for the operator's console.

        The verdict judges each function version that the task proved: it returned in
        one of the task's scripts, and nothing raised through it in any of them. True
        verifies those versions, and the shelf writes them into bundled/. False
        verifies nothing and gives each of them a rejected event in the shelf's ledger,
        so the description marks it apart from one never judged. Either way the verdict
        goes to the tool log. A verdict while a task runs, or with no task ended since
        the last verdict, changes nothing and is logged.
        """
        with self._lock:
            if self._running:
                return self._ignored(passed, "Ignored the verdict: a task is running.")
            if self._ended is None:
                return self._ignored(passed, "Ignored the verdict: no task has ended since the last one.")
            scripts, self._ended = self._ended, None
        versions = proven(scripts)
        if versions:
            try:
                if passed:
                    self.shelf.mark_verified(versions)
                else:
                    self.shelf.mark_rejected(versions)
            except (OSError, ValueError):
                logger.exception("Could not record the verdict on %s.", ", ".join(sorted(versions)))
                return self._judged(passed, versions, "error", "The verdict could not be recorded on the shelf.")
            self._refresh()
        if not passed and versions:
            line = f"Recorded a failed verdict on the last task, against {', '.join(sorted(versions))}. "
            line += "Nothing was verified."
        elif not passed:
            line = "Recorded a failed verdict on the last task. Nothing was verified."
        elif versions:
            line = f"Verified {', '.join(sorted(versions))} from the last task."
        else:
            line = "Recorded a passed verdict on the last task. It proved no saved function, so nothing was verified."
        return self._judged(passed, versions, "success", line)

    def _ignored(self, passed: bool, line: str) -> str:
        logger.warning(line)
        self._log(passed, {}, "ignored", line)
        return line

    def _judged(self, passed: bool, versions: dict[str, str], status: str, line: str) -> str:
        logger.info(line)
        self._log(passed, versions, status, line)
        return line

    def _log(self, passed: bool, versions: dict[str, str], status: str, line: str) -> None:
        """One tool-log line. `versions` are those the verdict judged, verified or rejected as it passed."""
        if self._tool_log is not None:
            self._tool_log.write(
                tool="verdict",
                args={"passed": passed},
                status=status,
                result=line,
                verified=versions if passed else {},
                rejected={} if passed else versions,
            )

    # -- Shutdown --------------------------------------------------------------

    def close(self, timeout_s: float) -> bool:
        """Close the pool, waiting up to `timeout_s`. True once it has closed. Safe to repeat.

        The workers close on a thread of their own, so a stuck worker cannot hold up
        the caller: a later call waits for the same thread again.
        """
        with self._lock:
            if self._closer is not None and self._closing is None:
                self._closing = threading.Thread(target=self._close_pool, name="walkie-scripts-close", daemon=True)
                self._closing.start()
            closing = self._closing
        if closing is None:
            return True
        closing.join(timeout_s)
        return not closing.is_alive()

    def unresolved(self) -> list[str]:
        """What kept the last close() from finishing, as sentences for the operator's log."""
        if self._closing is not None and self._closing.is_alive():
            return ["The script sandbox's workers have not closed."]
        return []

    def _close_pool(self) -> None:
        try:
            self._closer()
        except Exception:
            logger.exception("Could not close the script sandbox.")


def _ledger(result: ScriptResult) -> list[str]:
    """Each call the script made, as "tool: status", for the model and the tool log."""
    return [f"{call['tool']}: {call.get('status', 'unknown')}" for call in result.calls]
