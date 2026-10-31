"""Turn a hung test into an error with every thread's stack, and end a session that cannot exit.

tests/conftest.py imports these hooks by name, so they run in every
session. `-p hang_watchdog`, with tests/ on PYTHONPATH, loads the same
module without the conftest.

- A test's setup, call and teardown together get `walkie_hang_timeout`
  seconds (an ini option, 150 by default; 0 turns the limit off). Past
  that, and every `walkie_hang_timeout` seconds after it, the stack of
  every thread goes to stderr and TestHangError is raised in the main
  thread: the test errors, and the session goes on. A join that never
  returns then says where each thread was, instead of stalling the run
  until the CI job is cancelled.
- Opening the debugger, through breakpoint(), --trace or the post-mortem
  of --pdb, turns the limit off for the rest of that test, so a debugging
  session is not cut short. pytest does the same for faulthandler_timeout.
- It owns SIGALRM, through an ITIMER_REAL timer. Nothing else in src/ or
  tests/ may use either.
- The exit guard: a process still alive EXIT_GRACE_S after
  pytest_unconfigure has a thread that blocks the interpreter's exit, such
  as a non-daemon spin thread or a pool worker held in a callback. The guard
  prints every thread's stack and ends the process with pytest's own exit
  status, or 3 (an internal error) when the session passed: a session that
  cannot exit has a fault even when every test passed. It does not cover a
  thread whose join a TestHangError interrupted: CPython 3.12 then marks
  that thread stopped (bpo-45274) while it still runs, so its is_alive() is
  False and the interpreter's exit no longer waits for it. The hang's own
  dump already holds its stack.
"""

import faulthandler
import os
import signal
import sys
import threading

import pytest

# How long the process may stay alive after pytest_unconfigure before the
# exit guard ends it. Its normal exit takes well under a second.
EXIT_GRACE_S = 30.0

_state: dict = {
    # A copy of the real stderr, taken before any test runs: pytest captures
    # fd 2 during a test, and the stacks must reach the terminal or CI log.
    "stderr": None,
    "exitstatus": None,
    # Guards for a session that registers these hooks twice, through the
    # conftest and through -p hang_watchdog.
    "armed": False,
    "guarded": False,
}


class TestHangError(pytest.fail.Exception):
    """A test's setup, call and teardown together ran past walkie_hang_timeout.

    It is one of pytest's own outcomes, the one pytest.fail raises, and
    pytest reports it as the test's failure or error.
    - Like every outcome, it is a BaseException: the hung code may sit under
      an `except Exception`, such as rai's ToolRunner's, which would otherwise
      turn the hang into an ordinary result.
    - pytest handles an outcome where any other BaseException would escape.
      A teardown that hangs still lets the test's other teardowns run,
      monkeypatch's undo among them, and a module fixture that hangs at setup
      gives its error to each later test that uses it. A plain BaseException
      skips those teardowns and leaves the later tests to fail inside pytest.
    """

    # The name starts with Test, so a test module that imports it would
    # otherwise have it collected as a test class.
    __test__ = False


def pytest_addoption(parser) -> None:
    parser.addini(
        "walkie_hang_timeout",
        "Seconds a test's setup, call and teardown may take together before it errors "
        "with the stack of every thread (0: no limit)",
        type="float",
        default=150.0,
    )


def pytest_configure(config) -> None:
    if _state["stderr"] is None:
        _state["stderr"] = os.dup(_stderr_fileno())


def _stderr_fileno() -> int:
    try:
        return sys.stderr.fileno()
    except (AttributeError, ValueError, OSError):
        return sys.__stderr__.fileno()


def _dump(header: str) -> None:
    """Write `header`, then the stack of every thread, to stderr.

    It goes to the real stderr, and also to fd 2 while pytest captures that:
    the captured copy stays with the test, in the "Captured stderr" section of
    its failure report.
    """
    real = _state["stderr"] if _state["stderr"] is not None else _stderr_fileno()
    for fd in (real,) if os.path.sameopenfile(real, 2) else (real, 2):
        os.write(fd, f"\n{header}\n".encode())
        faulthandler.dump_traceback(file=fd, all_threads=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    timeout_s = float(item.config.getini("walkie_hang_timeout"))
    if timeout_s <= 0 or _state["armed"]:
        yield
        return

    def on_alarm(signum, frame) -> None:
        message = (
            f"{item.nodeid} ran past walkie_hang_timeout ({timeout_s:g} s) "
            "in its setup, call and teardown together"
        )
        _dump(f"{message}. The stack of every thread follows.")
        raise TestHangError(f"{message}. The stack of every thread is on stderr.")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    _state["armed"] = True
    # The interval fires it again: a call that hung often leaves a teardown
    # that hangs too, and that teardown needs its own error.
    signal.setitimer(signal.ITIMER_REAL, timeout_s, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        _state["armed"] = False


@pytest.hookimpl(tryfirst=True)
def pytest_enter_pdb() -> None:
    """Turn the limit off for the rest of the test once the debugger opens.

    pytest calls this for breakpoint(), --trace and the post-mortem of --pdb
    alike. pytest_exception_interact, which pytest's faulthandler plugin also
    uses, would be wrong here: pytest calls it after every failed phase, with
    or without --pdb, and the teardown after a hung call still needs the limit.
    """
    if _state["armed"]:
        signal.setitimer(signal.ITIMER_REAL, 0)


def pytest_sessionfinish(session, exitstatus) -> None:
    _state["exitstatus"] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config) -> None:
    if _state["guarded"]:
        return
    _state["guarded"] = True
    guard = threading.Timer(EXIT_GRACE_S, _exit_guard)
    guard.name = "hang_watchdog exit guard"
    guard.daemon = True
    guard.start()


def _exit_guard() -> None:
    _dump(
        f"The process is still alive {EXIT_GRACE_S:g} s after the test session ended: a "
        "non-daemon thread or a thread pool worker, which the interpreter also joins at exit, "
        "blocks its exit. The stack of every thread follows."
    )
    os._exit(_state["exitstatus"] or 3)
