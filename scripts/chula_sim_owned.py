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

A listing of /proc is a snapshot. A process of the part can start a child after the
listing and exit before it is read, and the child, in no listing yet, would be missed.
/proc is therefore listed again until a listing adds no process that may be in the
group: none in it, and none that exited before its group could be read, since that one
may have started a child first.

    chula_sim_owned.py pids PGID TOKEN         print the part's process IDs, one a line
    chula_sim_owned.py signal PGID TOKEN SIG   send SIG (INT, TERM or KILL) to each

Both exit 0 when no process matches, and 1 on an error, such as a failed system call or
a group that still gains processes after MAX_LISTINGS listings. Needs Linux 5.3 or newer
on x86-64 or arm64, and Python 3.9 or newer. The two pidfd calls go through libc's
syscall(): a Python built against older kernel headers, such as Pixi's, has no
os.pidfd_open. The script runs this with -I.
"""

import ctypes
import os
import platform
import signal
import sys
from collections.abc import Iterator
from pathlib import Path

TOKEN_VARIABLE = b"CHULA_SIM_TOKEN"
SIGNALS = {"INT": signal.SIGINT, "TERM": signal.SIGTERM, "KILL": signal.SIGKILL}
# The same numbers on x86-64 and arm64, from Linux's common system call table.
SYSCALLS = {"pidfd_send_signal": 424, "pidfd_open": 434}
MACHINES = ("x86_64", "aarch64")
# A part has usually settled by the second listing; this many means it keeps starting
# processes, and the caller is told so rather than given an answer that may be stale.
MAX_LISTINGS = 100

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


def _group_of(pid: int) -> int:
    """The process group of `pid`, from /proc/PID/stat.

    The command name, the second field, is in parentheses and may itself hold spaces or
    parentheses, so the fields are counted from the last closing parenthesis: state,
    parent ID, then the group ID.
    """
    stat = Path(f"/proc/{pid}/stat").read_bytes()
    return int(stat[stat.rindex(b")") + 2 :].split()[2])


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
    """Yield each process ID of group `pgid` once, listing /proc until no new one may be.

    Each listing's new IDs are yielded before /proc is listed again, so a child that a
    caller's signal makes a process start is in a later listing.
    """
    seen: set[int] = set()
    for _ in range(MAX_LISTINGS):
        new = [pid for pid in _process_ids() if pid not in seen]
        seen.update(new)
        again = False
        for pid in new:
            try:
                in_group = _group_of(pid) == pgid
            except (FileNotFoundError, ProcessLookupError):
                again = True  # exited before it was read, perhaps after starting a child
                continue
            if in_group:
                again = True
                yield pid
        if not again:
            return
    raise RuntimeError(f"group {pgid} still gained processes after {MAX_LISTINGS} listings of /proc")


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
