"""scripts/chula-sim.sh, in the cases that start nothing but stand-ins.

Each run gets a SIM_DIR of its own and no WALKIE_ROS, so none of them can
reach a simulation or a router. The ownership tests stand a `sleep` in for a
part, in a session of its own. Tests that call the script's functions, or
shorten its timeouts, source it in a bash of their own.
"""

import fcntl
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "chula-sim.sh"
OWNED = SCRIPT.parent / "chula_sim_owned.py"
BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def environment(sim_dir: Path, **env: str) -> dict[str, str]:
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("SIM_", "WALKIE_ROS"))}
    return clean | {"SIM_DIR": str(sim_dir)} | env


def run(*args: str, sim_dir: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=environment(sim_dir, **env),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def sourced(commands: str, sim_dir: Path) -> subprocess.CompletedProcess:
    """Run `commands` in a bash that has sourced the script, so it can call its functions."""
    return subprocess.run(
        ["bash", "-c", f'source "$1"\n{commands}', "bash", str(SCRIPT)],
        env=environment(sim_dir),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def wait_for(condition: Callable[[], object], seconds: float = 10) -> object:
    deadline = time.monotonic() + seconds
    while not (value := condition()):
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)
    return value


def runs_sleep(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().startswith(b"sleep\0")
    except OSError:
        return False


def down_once_no_start_is_pending(sim_dir: Path) -> None:
    """Clean up after a failed test: down, retried while a pending start holds the lock."""
    for _ in range(100):
        if "another scripts/chula-sim.sh" not in run("down", sim_dir=sim_dir).stderr:
            return
        time.sleep(0.1)


def test_the_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], timeout=30, check=True)


@pytest.mark.parametrize("args", [(), ("restart",)])
def test_without_a_known_command_it_prints_the_usage(tmp_path, args):
    result = run(*args, sim_dir=tmp_path)
    assert result.returncode == 2
    assert "up [--agent] [--gui]" in result.stdout
    assert "down" in result.stdout


def test_env_prints_the_private_graph(tmp_path):
    result = run("env", sim_dir=tmp_path, SIM_ZENOH_PORT="7600", SIM_DOMAIN_ID="88")
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "export ROS_DOMAIN_ID=88",
        "export RMW_IMPLEMENTATION=rmw_zenoh_cpp",
        """export ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/localhost:7600"]'""",
    ]


def test_up_needs_a_walkie_ros_workspace(tmp_path):
    result = run("up", sim_dir=tmp_path)
    assert result.returncode == 1
    assert "set WALKIE_ROS" in result.stderr
    assert not (tmp_path / "pids").exists()


def test_up_refuses_a_workspace_that_is_not_built(tmp_path):
    workspace = tmp_path / "walkie-ros"
    workspace.mkdir()
    (workspace / "pixi.toml").touch()
    result = run("up", sim_dir=tmp_path / "sim", WALKIE_ROS=str(workspace))
    assert result.returncode == 1
    assert "no built install/" in result.stderr


def test_up_refuses_an_unknown_option(tmp_path):
    result = run("up", "--arms", sim_dir=tmp_path)
    assert result.returncode == 1
    assert "unknown option for up: --arms" in result.stderr


def test_with_nothing_started_every_part_is_stopped(tmp_path):
    result = run("status", sim_dir=tmp_path)
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        f"{part}: stopped" for part in ("router", "bringup", "oracle", "speech", "agent")
    ]
    assert run("down", sim_dir=tmp_path).returncode == 0


def sleep_with_token(token: str) -> subprocess.Popen:
    """A sleep in a session and process group of its own, carrying `token`."""
    return subprocess.Popen(
        ["sleep", "60"], start_new_session=True, env={**os.environ, "CHULA_SIM_TOKEN": token}
    )


@pytest.fixture
def stand_in() -> Iterator[subprocess.Popen]:
    """A process group of its own, carrying the token "agent-1"."""
    process = sleep_with_token("agent-1")
    yield process
    process.kill()
    process.wait(timeout=10)


def write_record(sim_dir: Path, record: str, name: str = "agent") -> Path:
    path = sim_dir / "pids" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record + "\n")
    return path


def helper_stand_in(tmp_path: Path, body: str) -> Path:
    """A bash script to put in OWNED's place; it gets the helper's arguments."""
    path = tmp_path / "owned.sh"
    path.write_text("#!/bin/bash\n" + body)
    return path


@pytest.mark.parametrize(
    "record",
    [
        "{pid} {boot} agent-2",  # another part's token: the group ID was reused
        "{pid} 00000000-0000-0000-0000-000000000000 agent-1",  # from another boot
        "{pid}",  # no boot and no token
        "not-a-pid {boot} agent-1",
    ],
)
def test_a_group_that_is_not_the_part_s_own_is_never_signalled(tmp_path, stand_in, record):
    path = write_record(tmp_path, record.format(pid=stand_in.pid, boot=BOOT_ID))
    assert "agent: stopped" in run("status", sim_dir=tmp_path).stdout
    result = run("down", sim_dir=tmp_path)
    assert result.returncode == 0
    assert "stopped agent" not in result.stdout
    assert stand_in.poll() is None
    assert not path.exists()


def test_the_part_s_own_group_is_running_and_down_stops_it(tmp_path, stand_in):
    write_record(tmp_path, f"{stand_in.pid} {BOOT_ID} agent-1")
    status = run("status", sim_dir=tmp_path).stdout
    assert f"agent: running, process group {stand_in.pid}," in status
    result = run("down", sim_dir=tmp_path)
    assert result.returncode == 0
    assert "stopped agent" in result.stdout
    assert stand_in.wait(timeout=10) == -2  # SIGINT, the first signal down sends


def test_down_refuses_while_another_up_or_down_holds_the_lock(tmp_path):
    with (tmp_path / "lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run("down", sim_dir=tmp_path)
    assert result.returncode == 1
    assert "another scripts/chula-sim.sh up or down is running" in result.stderr
    assert run("down", sim_dir=tmp_path).returncode == 0


# The leader starts a sleep without the token in its own group, prints its ID and
# becomes a sleep itself. A shell's background job would not do: it ignores SIGINT,
# so it would outlive a signal to the whole group as well.
LEADER_WITH_A_MEMBER_WITHOUT_THE_TOKEN = """
import os, subprocess
env = {k: v for k, v in os.environ.items() if k != "CHULA_SIM_TOKEN"}
print(subprocess.Popen(["sleep", "60"], env=env).pid, flush=True)
os.execvp("sleep", ["sleep", "60"])
"""


def ignores(pid: int, signum: int) -> bool:
    """`pid` ignores `signum`, from the SigIgn mask in /proc/PID/status."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("SigIgn:"):
            return bool(int(line.split()[1], 16) >> (signum - 1) & 1)
    raise AssertionError(f"no SigIgn line for {pid}")


def test_a_process_of_the_group_without_the_token_is_never_signalled(tmp_path):
    leader = subprocess.Popen(
        [sys.executable, "-I", "-c", LEADER_WITH_A_MEMBER_WITHOUT_THE_TOKEN],
        start_new_session=True,
        env={**os.environ, "CHULA_SIM_TOKEN": "agent-1"},
        stdout=subprocess.PIPE,
        text=True,
    )
    other = int(leader.stdout.readline())
    try:
        wait_for(lambda: runs_sleep(leader.pid))
        assert os.getpgid(other) == leader.pid
        assert not ignores(other, signal.SIGINT)
        write_record(tmp_path, f"{leader.pid} {BOOT_ID} agent-1")
        result = run("down", sim_dir=tmp_path)
        assert result.returncode == 0
        assert "stopped agent" in result.stdout
        assert leader.wait(timeout=10) == -signal.SIGINT
        assert runs_sleep(other)
    finally:
        leader.kill()
        leader.wait(timeout=10)
        leader.stdout.close()
        try:
            os.kill(other, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_a_process_with_the_token_in_another_group_is_left_alone(tmp_path, stand_in):
    """Such as a ros2 daemon that a part's ros2 command started in a session of its own."""
    outsider = subprocess.Popen(
        ["sleep", "60"], start_new_session=True, env={**os.environ, "CHULA_SIM_TOKEN": "agent-1"}
    )
    try:
        write_record(tmp_path, f"{stand_in.pid} {BOOT_ID} agent-1")
        result = run("down", sim_dir=tmp_path)
        assert result.returncode == 0
        assert stand_in.wait(timeout=10) == -signal.SIGINT
        assert outsider.poll() is None
    finally:
        outsider.kill()
        outsider.wait(timeout=10)


def test_down_sends_sigkill_to_a_part_that_ignores_sigint_and_sigterm(tmp_path):
    stubborn = subprocess.Popen(
        ["sh", "-c", "trap '' INT TERM; exec sleep 60"],
        start_new_session=True,
        env={**os.environ, "CHULA_SIM_TOKEN": "agent-1"},
    )
    try:
        wait_for(lambda: runs_sleep(stubborn.pid))
        write_record(tmp_path, f"{stubborn.pid} {BOOT_ID} agent-1")
        result = sourced("STOP_INT_S=1 STOP_TERM_S=1; down", tmp_path)
        assert result.returncode == 0, result.stderr
        assert "stopped agent" in result.stdout
        assert stubborn.wait(timeout=10) == -signal.SIGKILL
    finally:
        stubborn.kill()
        stubborn.wait(timeout=10)


def test_a_start_up_gave_up_on_keeps_down_out_until_its_part_has_a_record(tmp_path):
    """The starter holds up's lock until it has written the part's record.

    A stand-in for mv stalls the starter before its record exists. up gives up,
    and a down at that moment would otherwise find no record and leave the part
    to start unwatched.
    """
    shims = tmp_path / "bin"
    shims.mkdir()
    (shims / "mv").write_text(f'#!/bin/sh\nsleep 4\nexec {shutil.which("mv")} "$@"\n')
    (shims / "mv").chmod(0o755)
    sim_dir = tmp_path / "sim"
    record = sim_dir / "pids" / "agent"
    try:
        gave_up = sourced(
            'START_TIMEOUT_S=1; lock; mkdir -p "$SIM_DIR/logs" "$SIM_DIR/pids"\n'
            f'PATH={shims}:$PATH; start agent "$SIM_DIR" sleep 60',
            sim_dir,
        )
        assert gave_up.returncode == 1
        assert "agent wrote no record within 1 s" in gave_up.stderr
        assert not record.exists()

        refused = run("down", sim_dir=sim_dir)
        assert refused.returncode == 1
        assert "another scripts/chula-sim.sh up or down is running" in refused.stderr

        wait_for(record.exists)
        leader = int(record.read_text().split()[0])
        wait_for(lambda: runs_sleep(leader))
        # Started in the background, but not with SIGINT ignored: down's first signal works.
        assert not ignores(leader, signal.SIGINT)
        assert "agent: running" in run("status", sim_dir=sim_dir).stdout
        result = run("down", sim_dir=sim_dir)
        assert result.returncode == 0, result.stderr
        assert "stopped agent" in result.stdout
        wait_for(lambda: not runs_sleep(leader))
    finally:
        down_once_no_start_is_pending(sim_dir)


@pytest.mark.parametrize("failing", ["pids", "signal"])
def test_down_goes_on_to_the_other_parts_when_the_helper_fails_for_one(tmp_path, failing):
    helper = helper_stand_in(
        tmp_path,
        f'if [[ $1 == {failing} && $3 == agent-1 ]]; then echo "simulated failure" >&2; exit 1; fi\n'
        f'exec python3 -I {OWNED} "$@"\n',
    )
    agent, speech = sleep_with_token("agent-1"), sleep_with_token("speech-1")
    try:
        agent_record = write_record(tmp_path, f"{agent.pid} {BOOT_ID} agent-1")
        speech_record = write_record(tmp_path, f"{speech.pid} {BOOT_ID} speech-1", "speech")
        result = sourced(f"OWNED=(bash {helper}); down", tmp_path)
        assert result.returncode == 1
        assert "could not look up or signal the processes of agent" in result.stderr
        assert "stopped speech" in result.stdout
        assert speech.wait(timeout=10) == -signal.SIGINT
        assert agent.poll() is None
        assert agent_record.exists()
        assert not speech_record.exists()
    finally:
        for process in (agent, speech):
            process.kill()
            process.wait(timeout=10)


def test_status_shows_a_part_whose_processes_cannot_be_looked_up(tmp_path, stand_in):
    helper = helper_stand_in(tmp_path, 'echo "simulated failure" >&2\nexit 1\n')
    write_record(tmp_path, f"{stand_in.pid} {BOOT_ID} agent-1")
    result = sourced(f"OWNED=(bash {helper}); status", tmp_path)
    assert result.returncode == 1
    assert "agent: unknown, its processes could not be looked up" in result.stdout
    assert "speech: stopped" in result.stdout


def test_a_part_that_outlives_every_signal_keeps_its_record(tmp_path):
    # The stand-in helper reports one process however often it is signalled.
    helper = helper_stand_in(
        tmp_path,
        'case $1 in\n  pids) echo 4242 ;;\n  signal) echo "$4" >> "$SIM_DIR/sent"; echo 1 ;;\nesac\n',
    )
    record = write_record(tmp_path, f"4242 {BOOT_ID} agent-1")
    result = sourced(f"STOP_INT_S=0 STOP_TERM_S=0 STOP_KILL_S=0; OWNED=(bash {helper}); down", tmp_path)
    assert result.returncode == 1
    assert "agent did not stop; its record stays" in result.stderr
    assert (tmp_path / "sent").read_text().split() == ["INT", "TERM", "KILL", "KILL"]
    assert record.exists()


@pytest.fixture
def owned() -> ModuleType:
    """scripts/chula_sim_owned.py, loaded as a module, for tests that replace its /proc."""
    spec = importlib.util.spec_from_file_location("chula_sim_owned", OWNED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProc:
    """Processes as the helper reads them from /proc, with an action on a first read.

    `processes` maps each ID to its group and whether it carries the token. An action in
    `on_read` runs the first time a process is read, before the read, as when a process
    starts a child and exits just after /proc was listed.
    """

    def __init__(self, owned: ModuleType, monkeypatch, processes: dict[int, tuple[int, bool]]):
        self.processes = dict(processes)
        self.on_read: dict[int, Callable[[], None]] = {}
        self.events: list[str] = []
        self.descriptors: dict[int, int] = {}
        monkeypatch.setattr(owned, "_process_ids", lambda: sorted(self.processes))
        monkeypatch.setattr(owned, "_group_of", lambda pid: self._read(pid)[0])
        monkeypatch.setattr(owned, "_is_member", self.is_member)
        monkeypatch.setattr(owned, "pidfd_open", self.pidfd_open)
        monkeypatch.setattr(owned, "pidfd_send_signal", self.pidfd_send_signal)

    def _read(self, pid: int) -> tuple[int, bool]:
        if action := self.on_read.pop(pid, None):
            action()
        if pid not in self.processes:
            raise FileNotFoundError(pid)
        return self.processes[pid]

    def is_member(self, pid: int, pgid: int, assignment: bytes) -> bool:
        self.events.append(f"check {pid}")
        try:
            group, carries_token = self._read(pid)
        except FileNotFoundError:
            return False
        return group == pgid and carries_token

    def pidfd_open(self, pid: int) -> int:
        if pid not in self.processes:
            raise ProcessLookupError(pid)
        self.events.append(f"open {pid}")
        descriptor = os.open(os.devnull, os.O_RDONLY)  # a real descriptor, for send to close
        self.descriptors[descriptor] = pid
        return descriptor

    def pidfd_send_signal(self, descriptor: int, signum: int) -> None:
        pid = self.descriptors[descriptor]
        if pid not in self.processes:
            raise ProcessLookupError(pid)
        self.events.append(f"signal {pid} {signum}")


def started_a_child_and_exited(fake: FakeProc, parent: int, child: int) -> Callable[[], None]:
    def action() -> None:
        fake.processes[child] = fake.processes.pop(parent)

    return action


def test_the_helper_finds_a_child_its_parent_started_as_it_exited(owned, monkeypatch):
    fake = FakeProc(owned, monkeypatch, {1: (1, False), 10: (10, True)})
    fake.on_read[10] = started_a_child_and_exited(fake, 10, 11)
    assert owned.members(10, "agent-1") == [11]


def test_the_helper_signals_a_child_its_parent_started_as_it_exited(owned, monkeypatch):
    fake = FakeProc(owned, monkeypatch, {1: (1, False), 10: (10, True)})
    fake.on_read[10] = started_a_child_and_exited(fake, 10, 11)
    assert owned.send(10, "agent-1", signal.SIGINT) == 1
    assert f"signal 11 {signal.SIGINT}" in fake.events


def test_the_helper_signals_through_a_pidfd_opened_before_the_check(owned, monkeypatch):
    def no_kill(*args):
        raise AssertionError(f"kill{args}: a signal by ID can reach a later holder of the ID")

    monkeypatch.setattr(owned.os, "kill", no_kill)
    monkeypatch.setattr(owned.os, "killpg", no_kill)
    # 12 is in the group but does not carry the token.
    fake = FakeProc(owned, monkeypatch, {10: (10, True), 12: (10, False)})
    assert owned.send(10, "agent-1", signal.SIGINT) == 1
    assert fake.events == [
        "open 10",
        "check 10",
        "signal 10 0",
        f"signal 10 {signal.SIGINT}",
        "open 12",
        "check 12",
    ]


def test_the_helper_does_not_signal_a_process_that_exits_after_its_check(owned, monkeypatch):
    fake = FakeProc(owned, monkeypatch, {10: (10, True)})
    check = fake.is_member

    def checked_then_exited(pid: int, pgid: int, assignment: bytes) -> bool:
        member = check(pid, pgid, assignment)
        del fake.processes[pid]
        return member

    monkeypatch.setattr(owned, "_is_member", checked_then_exited)
    assert owned.send(10, "agent-1", signal.SIGINT) == 0
    assert fake.events == ["open 10", "check 10"]


def test_the_helper_signals_a_child_started_while_its_parent_was_signalled(owned, monkeypatch):
    fake = FakeProc(owned, monkeypatch, {10: (10, True)})
    signal_one = fake.pidfd_send_signal

    def starts_a_child_on_sigint(descriptor: int, signum: int) -> None:
        signal_one(descriptor, signum)
        if signum == signal.SIGINT:
            fake.processes.setdefault(11, (10, True))

    monkeypatch.setattr(owned, "pidfd_send_signal", starts_a_child_on_sigint)
    assert owned.send(10, "agent-1", signal.SIGINT) == 2
    assert f"signal 11 {signal.SIGINT}" in fake.events


def test_the_helper_gives_up_on_a_group_that_keeps_starting_processes(owned, monkeypatch, capsys):
    fake = FakeProc(owned, monkeypatch, {10: (10, True)})
    new_ids = iter(range(11, 1000))

    def a_new_member_each_listing() -> list[int]:
        fake.processes[next(new_ids)] = (10, True)
        return sorted(fake.processes)

    monkeypatch.setattr(owned, "_process_ids", a_new_member_each_listing)
    assert owned.main(["pids", "10", "agent-1"]) == 1
    assert "still gained processes" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [("pids", "0", "agent-1"), ("pids", "12", ""), ("signal", "12", "agent-1", "HUP"), ("kill",)],
)
def test_the_ownership_helper_refuses_bad_arguments(args):
    result = subprocess.run(
        ["python3", "-I", str(OWNED), *args], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr
