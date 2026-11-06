#!/usr/bin/env python3
"""Find and signal the processes of one part of scripts/chula-sim.sh, and no others.

A part's processes are the members of its process group whose environment holds its
token, CHULA_SIM_TOKEN=<token>. The system can give a process ID, and so a group ID, to
another program as soon as the last process holding it exits. A check of the group
followed by kill(-PGID) therefore leaves a moment in which the signal could reach that
other program.

Each process is signalled through a pidfd instead: a descriptor that refers to one
process and never to a later one with the same ID. The descriptor is opened first, then
the group and the token are read from /proc, then a null signal through the descriptor
confirms that the process read is still the one it refers to, and only then is the
signal sent through it. A process that exits at any of these steps is skipped.

A listing of /proc is no snapshot. A process of the part can start a child and exit
while /proc is listed and read, and the child would be in no listing. The helper
therefore reads the ID that the kernel gave last to a new process or thread, from
/proc/sys/kernel/ns_last_pid, before it lists /proc and again once it has read every
listed process. Linux gives out IDs in turn, so each process started in between has an
ID above the first reading and up to the second. Those IDs are read directly, and the
last ID read again, until it has not moved. A process that exits before it is read needs
no second look: any child it started is among those IDs. Processes of other programs
that start and exit cost a read of their IDs, not another listing.

This finds most such children, not all. Linux gives a new process its ID, and moves the
last ID, before /proc shows the process. A child read in that moment looks absent, and
if its parent then exits and the last ID stays put, the helper misses it.
scripts/chula-sim.sh therefore reads a part that looks stopped a second time, a moment
later, before it removes the part's record. A strict answer needs the kernel to keep the
part's processes together, as a cgroup does, not a reading of IDs.

A process is told from a later one with the same ID by its start time, in 10 ms ticks.
When the last ID reads below the one before, the IDs went past pid_max and began again
from the bottom, and /proc is listed again, as it is when more than MAX_RANGE IDs were
given out between two readings. A wrap that brings the last ID back to, or past, the
earlier reading goes unseen. Any wrap takes one new process for each free ID, since the
kernel skips IDs in use: about four million on a desktop with the default pid_max of
4194304, but only a few when nearly every ID is taken. A process that gets the ID of one
already read, in the same tick, is taken for it and skipped.

Two things defeat the reading of IDs, and both need CAP_SYS_ADMIN or
CAP_CHECKPOINT_RESTORE: a process started with an ID of its choice, through clone3()'s
set_tid, and a write to ns_last_pid, which can start a process below the earlier reading
and put the last ID back.

A thread group whose first thread has exited while others run is missed too. The first
thread stays a zombie, its environment reads empty, so the token is not found, and the
other threads' IDs are not listed. None of the simulation's programs exit that way.

    chula_sim_owned.py pids PGID TOKEN         print the part's process IDs, one a line
    chula_sim_owned.py signal PGID TOKEN SIG   send SIG (INT, TERM or KILL) to each

Both exit 0 when no process matches, and 1 on an error, such as a failed system call or
a last ID that still moved after MAX_READINGS rounds. Needs Linux 5.3 or newer on
x86-64 or arm64, built with CONFIG_CHECKPOINT_RESTORE for ns_last_pid, as Ubuntu's
kernels are, and Python 3.9 or newer. The two pidfd calls go through libc's syscall(): a
Python built against older kernel headers, such as Pixi's, has no os.pidfd_open. The
script runs this with -I.
"""

import ctypes
import os
import platform
import signal
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

TOKEN_VARIABLE = b"CHULA_SIM_TOKEN"
SIGNALS = {"INT": signal.SIGINT, "TERM": signal.SIGTERM, "KILL": signal.SIGKILL}
# The same numbers on x86-64 and arm64, from Linux's common system call table.
SYSCALLS = {"pidfd_send_signal": 424, "pidfd_open": 434}
MACHINES = ("x86_64", "aarch64")
LAST_PID = Path("/proc/sys/kernel/ns_last_pid")
# The last ID usually stops moving within a few rounds, as each reads only the IDs given
# out since the round before. A round reads the last ID once, or twice when it lists
# /proc. This many rounds means processes kept starting throughout, and the caller is
# told so rather than given an answer that may be stale.
MAX_READINGS = 1000
# More IDs given out between two readings than this are cheaper to find by a listing.
MAX_RANGE = 4096

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def _syscall(name: str, *args) -> int:
    """Call system call `name`, raising OSError (ProcessLookupError for ESRCH) on failure."""
    result = _libc.syscall(ctypes.c_long(SYSCALLS[name]), *args)
    if result == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def pidfd_open(pid: int) -> int:
    return _syscall("pidfd_open", ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_send_signal(pidfd: int, signum: int) -> None:
    _syscall(
        "pidfd_send_signal",
        ctypes.c_int(pidfd),
        ctypes.c_int(signum),
        ctypes.c_void_p(None),
        ctypes.c_uint(0),
    )


def _process_ids() -> list[int]:
    return [int(entry) for entry in os.listdir("/proc") if entry.isdigit()]


def _last_pid() -> int:
    """The ID the kernel gave last to a new process or thread in this PID namespace."""
    return int(LAST_PID.read_bytes())


def _is_thread(pid: int) -> bool:
    """`pid` is a thread of another process. /proc reads such an ID but does not list it."""
    for line in Path(f"/proc/{pid}/status").read_bytes().splitlines():
        if line.startswith(b"Tgid:"):
            return int(line.split()[1]) != pid
    raise OSError(f"/proc/{pid}/status has no Tgid line")


def _stat_of(pid: int) -> tuple[int, int]:
    """The process group of `pid`, and its start time in clock ticks since boot.

    Both come from /proc/PID/stat. The command name, the second field, is in parentheses
    and may itself hold spaces or parentheses, so the fields are counted from the last
    closing parenthesis: the group ID is the third, the start time the twentieth.
    """
    stat = Path(f"/proc/{pid}/stat").read_bytes()
    fields = stat[stat.rindex(b")") + 2 :].split()
    return int(fields[2]), int(fields[19])


def _group_of(pid: int) -> int:
    return _stat_of(pid)[0]


def _is_member(pid: int, pgid: int, assignment: bytes) -> bool:
    """`pid` is in group `pgid`, and its environment holds `assignment` exactly."""
    try:
        if _group_of(pid) != pgid:
            return False
        return assignment in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        # Exited, or another user's process, whose environment cannot be read.
        return False


def _in_group(pgid: int) -> Iterator[int]:
    """Yield the ID of each process of group `pgid` once, until the last ID given out stays put.

    A process is its ID and its start time, as an ID can pass to a later process. A
    process is yielded as soon as it is read, before the last ID is read again, so a
    child that a caller's signal makes it start is read in turn. A child that /proc
    does not show yet when its ID is read is missed (see the module docstring).
    """
    seen: set[tuple[int, int]] = set()
    ids: Iterable[int] | None = None
    listed = True
    for _ in range(MAX_READINGS):
        if ids is None:
            last = _last_pid()
            ids, listed = _process_ids(), True
        for pid in ids:
            try:
                if not listed and _is_thread(pid):
                    continue
                group, started = _stat_of(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue  # exited, or the ID was free; a child it started is read in turn
            if (pid, started) in seen:
                continue
            seen.add((pid, started))
            if group == pgid:
                yield pid
        newest = _last_pid()
        if newest == last:
            return
        if last < newest <= last + MAX_RANGE:
            ids, listed, last = range(last + 1, newest + 1), False, newest
        else:
            ids = None  # the IDs went past pid_max and began again, or very many started
    raise RuntimeError(
        f"processes kept starting through {MAX_READINGS} rounds of reading {LAST_PID}, "
        f"so the processes of group {pgid} could not all be found"
    )


def members(pgid: int, token: str) -> list[int]:
    """The IDs of the part's processes now. For a report only: nothing is signalled."""
    assignment = TOKEN_VARIABLE + b"=" + token.encode()
    return [pid for pid in _in_group(pgid) if _is_member(pid, pgid, assignment)]


def send(pgid: int, token: str, signum: int) -> int:
    """Send `signum` to each of the part's processes through a pidfd; return how many."""
    assignment = TOKEN_VARIABLE + b"=" + token.encode()
    sent = 0
    for pid in _in_group(pgid):
        try:
            pidfd = pidfd_open(pid)
        except (ProcessLookupError, PermissionError):
            continue
        try:
            if not _is_member(pid, pgid, assignment):
                continue
            # The process the descriptor refers to is still alive, so the ID was not
            # reused while its group and environment were read.
            pidfd_send_signal(pidfd, 0)
            pidfd_send_signal(pidfd, signum)
            sent += 1
        except (ProcessLookupError, PermissionError):
            continue
        finally:
            os.close(pidfd)
    return sent


def main(argv: list[str]) -> int:
    usage = "usage: chula_sim_owned.py pids PGID TOKEN | signal PGID TOKEN INT|TERM|KILL"
    if len(argv) == 3 and argv[0] == "pids":
        command, group, token = argv
    elif len(argv) == 4 and argv[0] == "signal" and argv[3] in SIGNALS:
        command, group, token, name = argv
    else:
        print(usage, file=sys.stderr)
        return 2
    if not group.isdigit() or int(group) < 1 or not token:
        print(usage, file=sys.stderr)
        return 2
    try:
        if command == "pids":
            for pid in members(int(group), token):
                print(pid)
            return 0
        if platform.machine() not in MACHINES:
            print(f"chula_sim_owned.py: the pidfd system calls are set up for {MACHINES} only", file=sys.stderr)
            return 2
        print(send(int(group), token, SIGNALS[name]))
        return 0
    except (OSError, RuntimeError) as error:
        print(f"chula_sim_owned.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
