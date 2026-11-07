"""The Monty worker pool, each worker wrapped in bubblewrap.

The deny driver already keeps scripts from the host. bwrap is the second wall,
for a bug in Monty itself: the worker runs with an empty environment, no network
and only /usr plus its own binary in view. Dev machines hold API keys, `.env` and
`~/.ssh`, so a missing or broken bwrap stops the run rather than dropping the wall.
"""

import shlex
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic_monty import Monty
from pydantic_monty._binary import find_monty_binary

_ISOLATION = (
    "--clearenv",
    "--unshare-all",
    "--die-with-parent",
    "--new-session",
    "--cap-drop",
    "ALL",
)
_WORKER = "/monty"


class SandboxUnavailable(RuntimeError):
    """bwrap is missing or cannot start a worker. Nothing runs without it."""


def bwrap_command(binary: str, *, bwrap: str = "bwrap", worker: str = _WORKER) -> list[str]:
    """The bwrap argv up to the worker's own arguments, with `binary` mounted at `worker`."""
    return [
        bwrap,
        *_ISOLATION,
        "--ro-bind", "/usr", "/usr",
        "--ro-bind-try", "/lib", "/lib",
        "--ro-bind-try", "/lib64", "/lib64",
        "--ro-bind", binary, worker,
        worker,
    ]  # fmt: skip


def _write_wrapper(directory: Path, bwrap: str) -> Path:
    """A script to pass as Monty's binary_path, since the pool takes a path and not an argv."""
    resolved = shutil.which(bwrap)
    if resolved is None:
        raise SandboxUnavailable(f"{bwrap} is not installed. Install it, or pass --no-bwrap on purpose.")
    command = bwrap_command(find_monty_binary(), bwrap=resolved)
    wrapper = directory / "monty-bwrap"
    wrapper.write_text(f'#!/bin/sh\nexec {shlex.join(command)} "$@"\n')
    wrapper.chmod(stat.S_IRWXU)
    return wrapper


@contextmanager
def open_pool(
    *,
    sandbox: bool = True,
    max_processes: int = 8,
    checkout_timeout: float = 120.0,
    bwrap: str = "bwrap",
) -> Iterator[Monty]:
    """A pool of `max_processes` workers, closed on exit. Raises SandboxUnavailable if bwrap fails.

    All workers start here, from the calling thread. `--die-with-parent` ties a
    worker to the thread that spawned it, so a worker spawned lazily from a
    short-lived tool thread would die with that thread.
    """
    with tempfile.TemporaryDirectory(prefix="mahoraga-bwrap-") as private:
        binary = _write_wrapper(Path(private), bwrap) if sandbox else None
        try:
            pool = Monty(
                binary_path=binary,
                min_processes=max_processes,
                max_processes=max_processes,
                checkout_timeout=checkout_timeout,
            )
            pool.__enter__()
            with pool.checkout() as session:
                session.feed_run("1 + 1")
        except Exception as exc:
            if sandbox:
                raise SandboxUnavailable(f"bwrap could not start a Monty worker: {exc}") from exc
            raise
        try:
            yield pool
        finally:
            pool.__exit__(None, None, None)
