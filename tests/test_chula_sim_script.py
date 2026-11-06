"""scripts/chula-sim.sh, in the cases that start nothing.

Each run gets a SIM_DIR of its own and no WALKIE_ROS, so none of them can
reach a simulation or a router.
"""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "chula-sim.sh"


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
