"""scripts/chula-sim.sh, in the cases that start nothing.

Each run gets a SIM_DIR of its own and no WALKIE_ROS, so none of them can
reach a simulation or a router. The ownership tests stand a `sleep` in for a
part, in a session of its own.
"""

import fcntl
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "chula-sim.sh"
BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def run(*args: str, sim_dir: Path, **env: str) -> subprocess.CompletedProcess:
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("SIM_", "WALKIE_ROS"))}
    environment |= {"SIM_DIR": str(sim_dir)} | env
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


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


@pytest.fixture
def stand_in() -> Iterator[subprocess.Popen]:
    """A process group of its own, carrying the token "agent-1"."""
    process = subprocess.Popen(
        ["sleep", "60"], start_new_session=True, env={**os.environ, "CHULA_SIM_TOKEN": "agent-1"}
    )
    yield process
    process.kill()
    process.wait(timeout=10)


def write_record(sim_dir: Path, record: str) -> Path:
    path = sim_dir / "pids" / "agent"
    path.parent.mkdir(parents=True)
    path.write_text(record + "\n")
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
