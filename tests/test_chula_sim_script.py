"""scripts/chula-sim.sh, in the cases that start nothing but stand-ins.

Each run gets a SIM_DIR of its own and no WALKIE_ROS, so none of them can
reach a simulation or a router. The ownership tests stand a `sleep` in for a
part, in a session of its own. Tests that call the script's functions, or
shorten its timeouts, source it in a bash of their own. The tests of up's
steps replace the functions that start, find and wait for parts with
stand-ins that record each call, so up runs through and starts nothing.
"""

import fcntl
import importlib.util
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "chula-sim.sh"
OWNED = SCRIPT.parent / "chula_sim_owned.py"
PARTS = (
    "router", "bringup", "moveit", "commander", "vision", "memory", "foxglove", "oracle", "speech",
    "agent",
)  # fmt: skip
MOVEIT_LAUNCH = Path(
    "openarm_bimanual_moveit_config/share/openarm_bimanual_moveit_config/launch/moveit_only.launch.py"
)
COMMANDER_EXECUTABLE = Path(
    "openarm_bimanual_commander_cpp/lib/openarm_bimanual_commander_cpp/commander"
)
BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
# The lowest ID Linux gives out once the IDs have gone past pid_max.
RESERVED_PIDS = 300


def environment(sim_dir: Path, **env: str) -> dict[str, str]:
    prefixes = ("SIM_", "WALKIE_ROS", "WALKIE_VISION", "WALKIE_DATABASE")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
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


def sourced(commands: str, sim_dir: Path, **env: str) -> subprocess.CompletedProcess:
    """Run `commands` in a bash that has sourced the script, so it can call its functions."""
    return subprocess.run(
        ["bash", "-c", f'source "$1"\n{commands}', "bash", str(SCRIPT)],
        env=environment(sim_dir, **env),
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
    assert (
        "up [--agent] [--arms | --whole-body] [--vision] [--memory] [--object-tools] [--foxglove]"
        " [--gui] [-- ARGS]" in result.stdout
    )
    assert "check [--agent] [--arms | --whole-body]" in result.stdout
    assert "  scripts/chula-sim.sh reset-memory " in result.stdout
    # From the header's first line to its last command, and no further.
    lines = result.stdout.splitlines()
    assert lines[0].startswith("Run walkie-ros's Chula PAT14-6 simulation")
    assert lines[-1].startswith("  scripts/chula-sim.sh down ")


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
    result = run("up", "--robot", sim_dir=tmp_path)
    assert result.returncode == 1
    assert "unknown option for up: --robot" in result.stderr


def walkie_ros_workspace(path: Path, *, moveit: bool, commander: bool) -> Path:
    """A built-looking walkie-ros workspace at `path`, with or without the arm packages."""
    (path / "install").mkdir(parents=True)
    (path / "pixi.toml").touch()
    (path / "install" / "setup.bash").touch()
    if moveit:
        (path / "install" / MOVEIT_LAUNCH).parent.mkdir(parents=True)
        (path / "install" / MOVEIT_LAUNCH).touch()
    if commander:
        (path / "install" / COMMANDER_EXECUTABLE).parent.mkdir(parents=True)
        (path / "install" / COMMANDER_EXECUTABLE).touch(mode=0o755)
    return path


@pytest.mark.parametrize(
    ("moveit", "commander", "missing"),
    [
        (False, True, "openarm_bimanual_moveit_config"),
        (True, False, "openarm_bimanual_commander_cpp"),
        (False, False, "openarm_bimanual_moveit_config"),
    ],
)
def test_up_with_the_arms_needs_moveit_and_the_commander_built_in_walkie_ros(
    tmp_path, moveit, commander, missing
):
    workspace = walkie_ros_workspace(tmp_path / "walkie-ros", moveit=moveit, commander=commander)
    result = run("up", "--arms", sim_dir=tmp_path / "sim", WALKIE_ROS=str(workspace))
    assert result.returncode == 1
    assert f"--arms needs {missing}, and WALKIE_ROS={workspace} has not built it" in result.stderr
    assert not (tmp_path / "sim").exists()


def test_up_with_the_arms_refuses_a_commander_that_is_not_executable(tmp_path):
    workspace = walkie_ros_workspace(tmp_path / "walkie-ros", moveit=True, commander=True)
    (workspace / "install" / COMMANDER_EXECUTABLE).chmod(0o644)
    result = run("up", "--arms", sim_dir=tmp_path / "sim", WALKIE_ROS=str(workspace))
    assert result.returncode == 1
    assert "--arms needs openarm_bimanual_commander_cpp" in result.stderr


def test_the_arm_packages_are_looked_for_only_with_the_arms(tmp_path):
    repo = tmp_path / "repo"
    (repo / "ros2_ws" / "install" / "walkie_vision_msgs").mkdir(parents=True)
    zenoh = repo / "rmw_zenoh_ws" / "install" / "rmw_zenoh_cpp" / "lib" / "librmw_zenoh_cpp.so"
    zenoh.parent.mkdir(parents=True)
    zenoh.touch()
    bare = walkie_ros_workspace(tmp_path / "bare", moveit=False, commander=False)
    armed = walkie_ros_workspace(tmp_path / "armed", moveit=True, commander=True)
    for workspace, with_arms in ((bare, "false"), (armed, "true")):
        result = sourced(
            f"WALKIE_ROS={shlex.quote(str(workspace))} REPO={shlex.quote(str(repo))}\n"
            f"check_prerequisites {with_arms} && echo passed",
            tmp_path,
        )
        assert (result.returncode, result.stdout, result.stderr) == (0, "passed\n", "")


def walkie_vision_checkout(path: Path, *, task: str = "memory") -> Path:
    """A walkie-vision checkout at `path` that looks installed, built and new enough for --vision."""
    (path / ".pixi" / "envs" / "default").mkdir(parents=True)
    (path / "pixi.toml").touch()
    (path / "src" / "walkie_vision").mkdir(parents=True)
    (path / "src" / "walkie_vision" / "debug_viz.py").touch()
    (path / "config" / "tasks").mkdir(parents=True)
    (path / "config" / "tasks" / f"{task}.yaml").touch()
    (path / "ros2_ws" / "install" / "walkie_vision_msgs").mkdir(parents=True)
    return path


def test_up_with_vision_needs_walkie_vision(tmp_path):
    result = sourced("check_vision_prerequisites", tmp_path)
    assert result.returncode == 1
    assert "--vision needs WALKIE_VISION" in result.stderr


@pytest.mark.parametrize(
    ("remove", "message"),
    [
        (".pixi/envs/default", "no installed environment"),
        ("src/walkie_vision/debug_viz.py", "needs feat/scene-graph-prep or later"),
        ("config/tasks/memory.yaml", "has no task memory in config/tasks/"),
        ("ros2_ws/install/walkie_vision_msgs", "walkie_vision_msgs is not built"),
    ],
)
def test_up_with_vision_refuses_a_checkout_that_lacks_a_part(tmp_path, remove, message):
    checkout = walkie_vision_checkout(tmp_path / "walkie-vision")
    target = checkout / remove
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    result = sourced(f"WALKIE_VISION={shlex.quote(str(checkout))}\ncheck_vision_prerequisites", tmp_path)
    assert result.returncode == 1
    assert message in result.stderr


def test_up_with_vision_takes_the_task_from_sim_vision_task(tmp_path):
    checkout = walkie_vision_checkout(tmp_path / "walkie-vision", task="hri")
    quoted = shlex.quote(str(checkout))
    result = sourced(f"WALKIE_VISION={quoted}\ncheck_vision_prerequisites && echo passed", tmp_path)
    assert "has no task memory" in result.stderr
    result = sourced(
        f"WALKIE_VISION={quoted} SIM_VISION_TASK=hri\ncheck_vision_prerequisites && echo passed", tmp_path
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, "passed\n", "")


def test_up_with_vision_makes_a_relative_walkie_vision_absolute(tmp_path):
    # The vision part starts in WALKIE_VISION, where a relative path would no longer resolve.
    checkout = walkie_vision_checkout(tmp_path / "walkie-vision")
    result = sourced(
        f"cd {shlex.quote(str(tmp_path))}\nWALKIE_VISION=walkie-vision\n"
        'check_vision_prerequisites && echo "$WALKIE_VISION"',
        tmp_path,
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, f"{checkout}\n", "")


def walkie_database_checkout(path: Path) -> Path:
    """A walkie-database checkout at `path` that looks installed, built and new enough for --memory."""
    (path / ".pixi" / "envs" / "default").mkdir(parents=True)
    (path / "pixi.toml").touch()
    (path / "src" / "walkie_database").mkdir(parents=True)
    (path / "src" / "walkie_database" / "node.py").write_text('log("The object memory starts with 0 objects.")\n')
    (path / "ros2_ws" / "install" / "walkie_database_msgs").mkdir(parents=True)
    return path


def memory_inputs(root: Path) -> str:
    """A walkie-database checkout, a store and a map folder under `root`, as shell assignments."""
    checkout = walkie_database_checkout(root / "walkie-database")
    (root / "store").mkdir()
    (root / "store" / "walkie.db").touch()
    (root / "map").mkdir()
    return (
        f"WALKIE_DATABASE={shlex.quote(str(checkout))} "
        f"SIM_MEMORY_DB={shlex.quote(str(root / 'store' / 'walkie.db'))} "
        f"SIM_MEMORY_MAP_FOLDER={shlex.quote(str(root / 'map'))}"
    )


def test_up_with_memory_needs_walkie_database(tmp_path):
    result = sourced("check_memory_prerequisites", tmp_path)
    assert result.returncode == 1
    assert "--memory needs WALKIE_DATABASE" in result.stderr


@pytest.mark.parametrize(
    ("remove", "message"),
    [
        ("walkie-database/.pixi/envs/default", "no installed environment"),
        ("walkie-database/src/walkie_database/node.py", "needs feat/scene-graph-prep or later"),
        ("walkie-database/ros2_ws/install/walkie_database_msgs", "walkie_database_msgs is not built"),
        ("store/walkie.db", "/store/walkie.db does not exist; scripts/chula-sim.sh reset-memory restores it"),
        ("map", "/map is not a folder"),
    ],
)
def test_up_with_memory_refuses_a_missing_checkout_part_store_or_map(tmp_path, remove, message):
    inputs = memory_inputs(tmp_path)
    target = tmp_path / remove
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    result = sourced(f"{inputs}\ncheck_memory_prerequisites", tmp_path)
    assert result.returncode == 1
    assert message in result.stderr


def test_up_with_memory_finds_the_object_memory_wherever_its_module_lives(tmp_path):
    inputs = memory_inputs(tmp_path)
    package = tmp_path / "walkie-database" / "src" / "walkie_database"
    (package / "node.py").unlink()
    (package / "ros").mkdir()
    (package / "ros" / "memory_node.py").write_text('log("The object memory starts with 3 objects.")\n')
    result = sourced(f"{inputs}\ncheck_memory_prerequisites", tmp_path)
    assert (result.returncode, result.stderr) == (0, "")


def test_up_with_memory_makes_its_relative_paths_absolute(tmp_path):
    # The memory part starts in WALKIE_DATABASE, where a relative path would no longer resolve.
    memory_inputs(tmp_path)
    result = sourced(
        f"cd {shlex.quote(str(tmp_path))}\n"
        "WALKIE_DATABASE=walkie-database SIM_MEMORY_DB=store/walkie.db SIM_MEMORY_MAP_FOLDER=map\n"
        'check_memory_prerequisites && printf "%s\\n" "$WALKIE_DATABASE" "$SIM_MEMORY_DB" "$SIM_MEMORY_MAP_FOLDER"',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(tmp_path / "walkie-database"),
        str(tmp_path / "store" / "walkie.db"),
        str(tmp_path / "map"),
    ]


@pytest.mark.parametrize(("overlay", "workspace"), [("/overlay", "/overlay"), ("", "/walkie-ros")])
def test_the_memory_map_is_the_one_bringup_loads(tmp_path, overlay, workspace):
    result = sourced(
        'echo "$SIM_MEMORY_MAP_FOLDER"', tmp_path, WALKIE_ROS="/walkie-ros", WALKIE_ROS_OVERLAY=overlay
    )
    assert result.stdout == (
        f"{workspace}/install/walkie_mujoco_worlds/share/walkie_mujoco_worlds/worlds/chula_pat14_6/map\n"
    )


def stand_in_pixi(path: Path, status: int = 0) -> Path:
    """An executable at `path` that writes its arguments to `path`.args, one a line, and exits `status`."""
    path.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$0.args"\nexit {status}\n')
    path.chmod(0o755)
    return path


def reset_memory_inputs(root: Path, status: int = 0) -> tuple[dict[str, str], Path]:
    """A walkie-database checkout, a seed, a map folder and a stand-in pixi under `root`.

    Returns the environment that names them and the store, a path not yet made, and the pixi.
    """
    checkout = walkie_database_checkout(root / "walkie-database")
    (root / "seed.db").touch()
    (root / "map").mkdir()
    pixi = stand_in_pixi(root / "pixi", status)
    env = {
        "PIXI": str(pixi),
        "WALKIE_DATABASE": str(checkout),
        "SIM_MEMORY_SEED": str(root / "seed.db"),
        "SIM_MEMORY_DB": str(root / "new" / "walkie.db"),
        "SIM_MEMORY_MAP_FOLDER": str(root / "map"),
    }
    return env, pixi


def test_reset_memory_restores_the_store_from_the_seed_with_db_restore(tmp_path):
    env, pixi = reset_memory_inputs(tmp_path)
    result = run("reset-memory", sim_dir=tmp_path / "sim", **env)
    assert result.returncode == 0, result.stderr
    assert Path(f"{pixi}.args").read_text().splitlines() == [
        "run", "--as-is", "--manifest-path", f"{tmp_path}/walkie-database/pixi.toml", "db-restore",
        "--seed", f"{tmp_path}/seed.db", "--db", f"{tmp_path}/new/walkie.db",
        "--map-folder", f"{tmp_path}/map", "--clock-kind", "ros_sim", "--replace",
    ]  # fmt: skip
    assert (tmp_path / "new").is_dir()


def test_reset_memory_makes_its_relative_paths_absolute(tmp_path):
    # db-restore runs in WALKIE_DATABASE, where a relative path would no longer resolve.
    _, pixi = reset_memory_inputs(tmp_path)
    result = sourced(
        f"cd {shlex.quote(str(tmp_path))}\n"
        "WALKIE_DATABASE=walkie-database SIM_MEMORY_SEED=seed.db SIM_MEMORY_DB=new/walkie.db\n"
        "SIM_MEMORY_MAP_FOLDER=map\nreset_memory",
        tmp_path / "sim",
        PIXI=str(pixi),
    )
    assert result.returncode == 0, result.stderr
    args = Path(f"{pixi}.args").read_text().splitlines()
    assert args[3] == f"{tmp_path}/walkie-database/pixi.toml"
    assert args[5:11] == [
        "--seed", f"{tmp_path}/seed.db", "--db", f"{tmp_path}/new/walkie.db",
        "--map-folder", f"{tmp_path}/map",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("remove", "message"),
    [
        ("WALKIE_DATABASE", "reset-memory needs WALKIE_DATABASE"),
        ("walkie-database/src/walkie_database/node.py", "reset-memory needs feat/scene-graph-prep"),
        ("map", "/map is not a folder"),
        ("seed.db", "/seed.db does not exist; docs/setup/08-chula-simulation.md says how to build it"),
    ],
)
def test_reset_memory_refuses_a_missing_checkout_map_or_seed(tmp_path, remove, message):
    env, pixi = reset_memory_inputs(tmp_path)
    if remove in env:
        del env[remove]
    elif (tmp_path / remove).is_dir():
        shutil.rmtree(tmp_path / remove)
    else:
        (tmp_path / remove).unlink()
    result = run("reset-memory", sim_dir=tmp_path / "sim", **env)
    assert result.returncode == 1
    assert message in result.stderr
    assert not Path(f"{pixi}.args").exists()


def test_reset_memory_refuses_while_the_memory_runs(tmp_path):
    env, pixi = reset_memory_inputs(tmp_path)
    result = sourced("running() { [[ $1 == memory ]]; }\nreset_memory", tmp_path / "sim", **env)
    assert result.returncode == 1
    assert "the object memory is running" in result.stderr
    assert not Path(f"{pixi}.args").exists()


def test_reset_memory_refuses_while_another_up_or_down_holds_the_lock(tmp_path):
    env, pixi = reset_memory_inputs(tmp_path)
    (tmp_path / "sim").mkdir()
    with (tmp_path / "sim" / "lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run("reset-memory", sim_dir=tmp_path / "sim", **env)
    assert result.returncode == 1
    assert "another scripts/chula-sim.sh up, down or reset-memory is running" in result.stderr
    assert not Path(f"{pixi}.args").exists()


def test_reset_memory_holds_the_lock_from_its_memory_check_through_db_restore(tmp_path):
    # Each stand-in notes its call, and whether another up could have taken the lock then.
    env, pixi = reset_memory_inputs(tmp_path)
    sim_dir = tmp_path / "sim"
    pixi.write_text(
        '#!/usr/bin/env bash\necho db-restore >> "$SIM_DIR/calls"\n'
        'flock -n "$SIM_DIR/lock" true && echo db-restore >> "$SIM_DIR/unlocked"\nexit 0\n'
    )
    result = sourced(
        'running() { echo "running $1" >> "$SIM_DIR/calls"\n'
        '  flock -n "$SIM_DIR/lock" true && echo "running $1" >> "$SIM_DIR/unlocked"; return 1; }\n'
        "reset_memory",
        sim_dir,
        **env,
    )
    assert result.returncode == 0, result.stderr
    assert (sim_dir / "calls").read_text().splitlines() == ["running memory", "db-restore"]
    assert not (sim_dir / "unlocked").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root enters any folder")
def test_reset_memory_stops_at_a_store_folder_it_cannot_enter(tmp_path):
    env, pixi = reset_memory_inputs(tmp_path)
    shut = tmp_path / "shut"
    shut.mkdir(mode=0o600)
    env["SIM_MEMORY_DB"] = str(shut / "walkie.db")
    try:
        result = run("reset-memory", sim_dir=tmp_path / "sim", **env)
    finally:
        shut.chmod(0o700)
    assert result.returncode == 1
    assert "is in a folder that cannot be made or entered" in result.stderr
    assert not Path(f"{pixi}.args").exists()


def test_reset_memory_fails_when_db_restore_refuses(tmp_path):
    env, _ = reset_memory_inputs(tmp_path, status=1)
    assert run("reset-memory", sim_dir=tmp_path / "sim", **env).returncode == 1


# The services the object tools call, by name: their type hashes.
OBJECT_SERVICES = {
    name: f"RIHS01_{name.lower()}"
    for name in ("SearchObjects", "GetRoomContents", "ListRooms", "GetObject", "SetObjectState")
}


def without(name: str) -> dict[str, str]:
    return {other: digest for other, digest in OBJECT_SERVICES.items() if other != name}


DATABASE_SRV = Path("ros2_ws/install/walkie_database_msgs/share/walkie_database_msgs/srv")


def built_services(root: Path, hashes: dict[str, str]) -> Path:
    """A checkout at `root` whose walkie_database_msgs give each service named the hash given.

    Each file also lists the hash of a type the service uses, as the real ones do.
    """
    srv = root / DATABASE_SRV
    srv.mkdir(parents=True)
    for name, digest in hashes.items():
        hashed = {
            "type_description_msg": {},
            "type_hashes": [
                {"type_name": f"walkie_database_msgs/srv/{name}", "hash_string": digest},
                {"type_name": "builtin_interfaces/msg/Time", "hash_string": f"RIHS01_time_{root.name}"},
            ],
        }
        (srv / f"{name}.json").write_text(json.dumps(hashed))
    return root


def check_object_tools(tmp_path: Path, agent: dict[str, str], memory: dict[str, str]):
    repo = built_services(tmp_path / "agent", agent)
    memory_srv = built_services(tmp_path / "memory", memory) / DATABASE_SRV
    return sourced(
        f"REPO={shlex.quote(str(repo))}\ncheck_object_tools_prerequisites {shlex.quote(str(memory_srv))}",
        tmp_path,
    )


def test_up_with_the_object_tools_compares_each_services_own_hash(tmp_path):
    # The hashes of the types the services use differ, and are not compared.
    result = check_object_tools(tmp_path, OBJECT_SERVICES, OBJECT_SERVICES)
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


@pytest.mark.parametrize(
    ("agent", "memory", "named"),
    [
        ({**OBJECT_SERVICES, "GetObject": "RIHS01_old"}, OBJECT_SERVICES, "GetObject differ"),
        (
            {**OBJECT_SERVICES, "SearchObjects": "RIHS01_old", "GetRoomContents": "RIHS01_old"},
            OBJECT_SERVICES,
            "SearchObjects, GetRoomContents differ",
        ),
        (without("ListRooms"), OBJECT_SERVICES, "ListRooms differ"),
        (OBJECT_SERVICES, without("GetObject"), "GetObject differ"),
    ],
    ids=["one-differs", "two-differ", "agent-lacks-one", "memory-lacks-one"],
)
def test_up_with_the_object_tools_refuses_services_the_memory_node_would_not_answer(
    tmp_path, agent, memory, named
):
    # Under rmw_zenoh_cpp the node would never answer these services' calls (ADR-0035).
    result = check_object_tools(tmp_path, agent, memory)
    assert result.returncode == 1
    assert f"never reach the memory node: {named} between" in result.stderr
    assert f"memory node's in {tmp_path}/memory/{DATABASE_SRV}," in result.stderr
    assert "pixi run build-msgs in WALKIE_DATABASE and pixi run build-ros2 here" in result.stderr


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({"type_hashes": [{"type_name": "builtin_interfaces/msg/Time", "hash_string": "x"}]}),
        "{",
        json.dumps({"type_description_msg": {}}),
        json.dumps(["RIHS01_getobject"]),
        json.dumps({"type_hashes": [{"type_name": "walkie_database_msgs/srv/GetObject"}]}),
    ],
    ids=["no-entry-for-it", "not-json", "no-hashes", "not-an-object", "entry-without-a-hash"],
)
def test_up_with_the_object_tools_counts_a_file_it_cannot_read_as_differing(tmp_path, content):
    repo = built_services(tmp_path / "agent", OBJECT_SERVICES)
    memory_srv = built_services(tmp_path / "memory", OBJECT_SERVICES) / DATABASE_SRV
    (memory_srv / "GetObject.json").write_text(content)
    result = sourced(
        f"REPO={shlex.quote(str(repo))}\ncheck_object_tools_prerequisites {shlex.quote(str(memory_srv))}",
        tmp_path,
    )
    assert result.returncode == 1
    assert "never reach the memory node: GetObject differ between" in result.stderr


def test_record_memory_services_replaces_the_last_record_with_the_build_the_node_loads(tmp_path):
    database = built_services(tmp_path / "database", OBJECT_SERVICES)
    stale = tmp_path / "memory-srv"
    stale.mkdir()
    (stale / "Old.json").write_text("{}")
    result = sourced(
        f"WALKIE_DATABASE={shlex.quote(str(database))}\nrecord_memory_services", tmp_path
    )
    assert (result.returncode, result.stderr) == (0, "")
    assert sorted(path.name for path in stale.iterdir()) == sorted(
        f"{name}.json" for name in OBJECT_SERVICES
    )
    assert (stale / "GetObject.json").read_text() == (database / DATABASE_SRV / "GetObject.json").read_text()


def test_record_memory_services_copies_what_a_symlink_install_links_to(tmp_path):
    # colcon build --symlink-install installs each file as a link into build/, which the
    # next build rewrites; the record must keep what the node loaded.
    database = built_services(tmp_path / "database", OBJECT_SERVICES)
    srv = database / DATABASE_SRV
    built = tmp_path / "database" / "build"
    built.mkdir()
    for name in OBJECT_SERVICES:
        (srv / f"{name}.json").rename(built / f"{name}.json")
        (srv / f"{name}.json").symlink_to(built / f"{name}.json")
    loaded = (built / "GetObject.json").read_text()
    result = sourced(f"WALKIE_DATABASE={shlex.quote(str(database))}\nrecord_memory_services", tmp_path)
    assert (result.returncode, result.stderr) == (0, "")
    (built / "GetObject.json").write_text(loaded.replace("RIHS01_getobject", "RIHS01_rebuilt"))
    recorded = tmp_path / "memory-srv" / "GetObject.json"
    assert not recorded.is_symlink()
    assert recorded.read_text() == loaded


def test_wait_for_log_stops_at_the_fail_text_of_a_part_that_keeps_running(tmp_path):
    # walkie-vision's launch respawns a node that cannot start, so the part never stops.
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "vision.log").write_text("[walkie_vision-1] [FATAL] cannot start: no models\n")
    started = time.monotonic()
    result = sourced(
        'running() { true; }\nwait_for_log vision "loaded the models in" 30 "cannot start:" || echo "returned $?"',
        tmp_path,
    )
    assert time.monotonic() - started < 10
    assert result.stdout == "returned 1\n"
    assert "cannot start: no models" in result.stderr


# Stand-ins for what up does to the system, for a bash that has sourced the script.
# Each records its call as a line of SIM_DIR/calls, the arguments separated by \x1f.
# A part runs once started, or when SIM_DIR/started-NAME exists before up, and the
# router listens while it runs. Another port is in use when SIM_DIR/listening-PORT
# exists. wait_for_log fails for the part named in NOT_READY.
UP_STAND_INS = r"""
record() { local IFS=$'\x1f'; echo "$*" >> "$SIM_DIR/calls"; }
check_prerequisites() { record check "$@"; }
check_whole_body_prerequisites() { record check-whole-body "$@"; }
check_vision_prerequisites() { record check-vision "$@"; }
check_memory_prerequisites() { record check-memory "$@"; }
check_object_tools_prerequisites() { record check-object-tools "$@"; }
record_memory_services() { record record-memory-services "$@"; }
pgrep() { :; }
running() { [[ -e $SIM_DIR/started-$1 ]]; }
port_open() { running router; }
tcp_listening() { [[ -e $SIM_DIR/listening-$1 ]]; }
start() { record start "$@"; touch "$SIM_DIR/started-$1"; }
wait_for_log() { record wait "$@"; [[ $1 != "$NOT_READY" ]]; }
SETTLE_S=0 PIXI=pixi WALKIE_ROS=/walkie-ros WALKIE_ROS_OVERLAY=/overlay WALKIE_VISION=/walkie-vision
WALKIE_DATABASE=/walkie-database SIM_MEMORY_DB=/store/walkie.db SIM_MEMORY_MAP_FOLDER=/map
"""
WALKIE_ROS_RUN = (
    "pixi",
    "run",
    "--as-is",
    "--manifest-path",
    "/walkie-ros/pixi.toml",
    "bash",
    "-c",
    '[[ -z $0 ]] || source "$0"; exec "$@"',
    "/overlay/install/local_setup.bash",
)


def up_with_stand_ins(
    sim_dir: Path, *args: str, running: Sequence[str] = (), not_ready: str = "", setup: str = ""
) -> tuple[subprocess.CompletedProcess, list[tuple[str, ...]]]:
    """Run up with UP_STAND_INS, after the shell lines in `setup`, and return its result and calls."""
    sim_dir.mkdir(parents=True, exist_ok=True)
    for name in running:
        (sim_dir / f"started-{name}").touch()
    result = sourced(
        UP_STAND_INS + f"{setup}\nNOT_READY={shlex.quote(not_ready)}\nup {shlex.join(args)}", sim_dir
    )
    calls = sim_dir / "calls"
    lines = calls.read_text().splitlines() if calls.exists() else []
    return result, [tuple(line.split("\x1f")) for line in lines]


def router_and_bringup(sim_dir: Path) -> list[tuple[str, ...]]:
    """The calls up makes to start the router and bringup and wait for Nav2."""
    return [
        (
            "start", "router", str(sim_dir), "env",
            'ZENOH_CONFIG_OVERRIDE=listen/endpoints=["tcp/localhost:7499"]',
            "pixi", "run", "--as-is", "--manifest-path", "/walkie-ros/pixi.toml",
            "ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd",
        ),
        (
            "start", "bringup", str(sim_dir), *WALKIE_ROS_RUN,
            "ros2", "launch", "robot_bringup", "nav_mujoco_omnibot.launch.py",
            "world:=chula_pat14_6", "spawn_pose:=1.7629 -1.483 0", "headless:=true",
            "use_rviz:=false", f"world_output_dir:={sim_dir}/worlds",
        ),
        ("wait", "bringup", "Managed nodes are active", "300"),
    ]  # fmt: skip


def the_arms(sim_dir: Path) -> list[tuple[str, ...]]:
    """The calls up --arms makes to start MoveIt and the commander, each after the last."""
    return [
        (
            "start", "moveit", str(sim_dir), *WALKIE_ROS_RUN,
            "ros2", "launch", "openarm_bimanual_moveit_config", "moveit_only.launch.py",
            "hardware_type:=mujoco", "use_rviz:=false",
        ),
        ("wait", "moveit", "MoveGroup context initialization complete", "120"),
        (
            "start", "commander", str(sim_dir), *WALKIE_ROS_RUN,
            "ros2", "launch", f"{REPO}/launch/commander_sim.launch.py",
        ),
        ("wait", "commander", "Bimanual Commander Node Initialized", "120"),
    ]  # fmt: skip


def oracle_speech_and_agent(sim_dir: Path, *agent_args: str) -> list[tuple[str, ...]]:
    """The calls up --agent makes after bringup, or after the arms, with the agent's arguments."""
    pixi_run = ("pixi", "run", "--frozen", "--manifest-path", f"{REPO}/pixi.toml")
    return [
        ("start", "oracle", str(REPO), *pixi_run, "oracle-chula"),
        (
            "start", "speech", str(REPO), "env", "PYTHONUNBUFFERED=1", *pixi_run,
            "ros2", "topic", "echo", "/to_human", "rai_interfaces/msg/HRIMessage", "--field", "text",
        ),
        (
            "start", "agent", str(REPO), *pixi_run, "agent-sim-chula",
            "--tool-log", f"{sim_dir}/logs/agent-tools.jsonl", *agent_args,
        ),
        ("wait", "oracle", "Drawing ", "120"),
        ("wait", "agent", "Walkie's agent is running", "120"),
    ]  # fmt: skip


def vision_and_foxglove(sim_dir: Path) -> list[tuple[str, ...]]:
    """The calls up --vision --foxglove makes to start the two parts, after bringup and the arms."""
    return [
        (
            "start", "vision", "/walkie-vision",
            "pixi", "run", "--as-is", "--manifest-path", "/walkie-vision/pixi.toml",
            "vision", "task:=memory", "sim:=true", "viz:=true",
        ),
        (
            "start", "foxglove", str(sim_dir), *WALKIE_ROS_RUN,
            "ros2", "launch", "foxglove_bridge", "foxglove_bridge_launch.xml",
            "port:=8799", "address:=127.0.0.1", "use_sim_time:=true",
        ),
    ]  # fmt: skip


VISION_AND_FOXGLOVE_WAITS = [
    ("wait", "vision", "loaded the models in", "300", "cannot start:"),
    ("wait", "foxglove", "Server listening on port", "120"),
]


def test_up_with_vision_and_foxglove_starts_them_before_the_oracle_and_waits_after_the_agent(
    tmp_path,
):
    result, calls = up_with_stand_ins(tmp_path, "--agent", "--vision", "--foxglove")
    assert result.returncode == 0, result.stderr
    after = oracle_speech_and_agent(tmp_path)
    assert calls == [
        ("check", "false"),
        ("check-vision",),
        *router_and_bringup(tmp_path),
        *vision_and_foxglove(tmp_path),
        *after,
        *VISION_AND_FOXGLOVE_WAITS,
    ]
    lines = result.stdout.splitlines()
    assert lines[-2:] == [
        "Foxglove: open a connection to ws://localhost:8799",
        "up. Next: scripts/chula-sim.sh check --agent",
    ]


def test_up_with_the_arms_starts_vision_after_the_commander(tmp_path):
    result, calls = up_with_stand_ins(tmp_path, "--arms", "--vision")
    assert result.returncode == 0, result.stderr
    assert calls == [
        ("check", "true"),
        ("check-vision",),
        *router_and_bringup(tmp_path),
        *the_arms(tmp_path),
        vision_and_foxglove(tmp_path)[0],
        *oracle_speech_and_agent(tmp_path)[:2],
        ("wait", "oracle", "Drawing ", "120"),
        VISION_AND_FOXGLOVE_WAITS[0],
    ]


def test_up_with_vision_and_foxglove_already_running_only_waits_for_them(tmp_path):
    result, calls = up_with_stand_ins(
        tmp_path,
        "--vision",
        "--foxglove",
        running=("router", "bringup", "vision", "foxglove", "oracle", "speech"),
    )
    assert result.returncode == 0, result.stderr
    assert calls == [
        ("check", "false"),
        ("check-vision",),
        router_and_bringup(tmp_path)[-1],
        ("wait", "oracle", "Drawing ", "120"),
        *VISION_AND_FOXGLOVE_WAITS,
    ]


MEMORY_START = (
    "start", "memory", "/walkie-database",
    "pixi", "run", "--as-is", "--manifest-path", "/walkie-database/pixi.toml",
    "db-memory", "-p", "db_path:=/store/walkie.db", "-p", "map_folder:=/map",
    "-p", "map_id:=chula-2026-11-12-714b087d", "-p", "use_sim_time:=true",
    "-p", "objects_topic:=/walkie_vision/sim/objects",
)  # fmt: skip
MEMORY_WAIT = ("wait", "memory", "The object memory starts with", "120", "Not starting:")


def test_up_with_memory_starts_it_after_vision_and_waits_for_it_after_vision(tmp_path):
    result, calls = up_with_stand_ins(tmp_path, "--agent", "--vision", "--memory", "--foxglove")
    assert result.returncode == 0, result.stderr
    vision, foxglove = vision_and_foxglove(tmp_path)
    vision_wait, foxglove_wait = VISION_AND_FOXGLOVE_WAITS
    assert calls == [
        ("check", "false"),
        ("check-vision",),
        ("check-memory",),
        *router_and_bringup(tmp_path),
        vision,
        ("record-memory-services",),
        MEMORY_START,
        foxglove,
        *oracle_speech_and_agent(tmp_path),
        vision_wait,
        MEMORY_WAIT,
        foxglove_wait,
    ]


def test_up_with_memory_already_running_only_waits_for_it(tmp_path):
    result, calls = up_with_stand_ins(
        tmp_path, "--vision", "--memory", running=("router", "bringup", "vision", "memory", "oracle", "speech")
    )
    assert result.returncode == 0, result.stderr
    assert calls == [
        ("check", "false"),
        ("check-vision",),
        ("check-memory",),
        router_and_bringup(tmp_path)[-1],
        ("wait", "oracle", "Drawing ", "120"),
        VISION_AND_FOXGLOVE_WAITS[0],
        MEMORY_WAIT,
    ]


def test_up_with_memory_and_no_vision_stops_before_it_starts_anything(tmp_path):
    result = run("up", "--memory", sim_dir=tmp_path)
    assert result.returncode == 1
    assert "so --memory needs --vision" in result.stderr
    assert not (tmp_path / "pids").exists()


@pytest.mark.parametrize(
    ("args", "added"),
    [
        (
            ("--tool-set", "hri"),
            ("--object-memory", "--objects-topic", "/walkie_vision/sim/objects"),
        ),
        (("--objects-topic", "/elsewhere"), ("--object-memory",)),
        (("--objects-topic=/elsewhere",), ("--object-memory",)),
    ],
    ids=["detector", "own-topic", "own-topic-with-equals"],
)
def test_up_with_the_object_tools_gives_the_agent_object_memory_and_the_detector(
    tmp_path, args, added
):
    result, calls = up_with_stand_ins(
        tmp_path, "--agent", "--vision", "--memory", "--object-tools", "--", *args
    )
    assert result.returncode == 0, result.stderr
    vision, _ = vision_and_foxglove(tmp_path)
    assert calls == [
        ("check", "false"),
        ("check-vision",),
        ("check-memory",),
        ("check-object-tools", f"/walkie-database/{DATABASE_SRV}"),
        *router_and_bringup(tmp_path),
        vision,
        ("record-memory-services",),
        MEMORY_START,
        *oracle_speech_and_agent(tmp_path, *args, *added),
        VISION_AND_FOXGLOVE_WAITS[0],
        MEMORY_WAIT,
    ]


RUNNING_BUT_THE_AGENT = ("router", "bringup", "vision", "memory", "oracle", "speech")


def test_up_with_the_object_tools_checks_a_running_memory_against_the_build_it_started_with(
    tmp_path,
):
    (tmp_path / "memory-srv").mkdir(parents=True)
    result, calls = up_with_stand_ins(
        tmp_path, "--agent", "--vision", "--memory", "--object-tools", running=RUNNING_BUT_THE_AGENT
    )
    assert result.returncode == 0, result.stderr
    assert calls[:4] == [
        ("check", "false"),
        ("check-vision",),
        ("check-memory",),
        ("check-object-tools", f"{tmp_path}/memory-srv"),
    ]
    assert ("record-memory-services",) not in calls
    assert MEMORY_START not in calls


def test_up_with_the_object_tools_refuses_a_running_memory_whose_build_it_did_not_record(tmp_path):
    result, calls = up_with_stand_ins(
        tmp_path, "--agent", "--vision", "--memory", "--object-tools", running=RUNNING_BUT_THE_AGENT
    )
    assert result.returncode == 1
    assert "the object memory runs from a build that up did not record" in result.stderr
    assert calls == [("check", "false"), ("check-vision",), ("check-memory",)]


@pytest.mark.parametrize(
    "flags",
    [
        ("--object-tools",),
        ("--agent", "--object-tools"),
        ("--vision", "--memory", "--object-tools"),
    ],
)
def test_up_with_the_object_tools_and_no_memory_or_agent_stops_before_it_starts_anything(
    tmp_path, flags
):
    result = run("up", *flags, sim_dir=tmp_path)
    assert result.returncode == 1
    assert "so --object-tools needs --memory and --agent" in result.stderr
    assert not (tmp_path / "pids").exists()


def test_up_stops_at_an_object_memory_that_never_logs_it_is_ready(tmp_path):
    result, calls = up_with_stand_ins(tmp_path, "--vision", "--memory", "--foxglove", not_ready="memory")
    assert result.returncode == 1
    assert f"chula-sim: the object memory did not start; see {tmp_path}/logs/memory.log" in result.stderr
    assert calls[-1] == MEMORY_WAIT


def test_up_with_foxglove_refuses_a_port_in_use_and_starts_nothing_after(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "listening-8799").touch()
    result, calls = up_with_stand_ins(tmp_path, "--foxglove")
    assert result.returncode == 1
    assert "port 8799 is in use; set SIM_FOXGLOVE_PORT to a free port" in result.stderr
    assert calls[-1] == router_and_bringup(tmp_path)[-1]


def test_up_with_foxglove_takes_its_port_from_sim_foxglove_port(tmp_path):
    result, calls = up_with_stand_ins(tmp_path, "--foxglove", setup="SIM_FOXGLOVE_PORT=8811")
    assert result.returncode == 0, result.stderr
    (foxglove,) = [call for call in calls if call[:2] == ("start", "foxglove")]
    assert "port:=8811" in foxglove
    assert "Foxglove: open a connection to ws://localhost:8811" in result.stdout


@pytest.mark.parametrize(
    ("part", "message"),
    [("vision", "walkie-vision did not load its models"), ("foxglove", "foxglove_bridge did not start")],
)
def test_up_stops_at_a_vision_or_foxglove_part_that_never_logs_it_is_ready(tmp_path, part, message):
    result, calls = up_with_stand_ins(tmp_path, "--vision", "--foxglove", not_ready=part)
    assert result.returncode == 1
    assert f"chula-sim: {message}; see {tmp_path}/logs/{part}.log" in result.stderr
    waits = dict(zip(("vision", "foxglove"), VISION_AND_FOXGLOVE_WAITS))
    assert calls[-1] == waits[part]


def test_up_without_the_arms_starts_what_it_always_did(tmp_path):
    result, calls = up_with_stand_ins(tmp_path, "--agent", "--", "--tool-set", "hri")
    assert result.returncode == 0, result.stderr
    assert calls == [
        ("check", "false"),
        *router_and_bringup(tmp_path),
        *oracle_speech_and_agent(tmp_path, "--tool-set", "hri"),
    ]
    assert result.stdout.splitlines()[-1] == "up. Next: scripts/chula-sim.sh check --agent"


def test_up_loads_the_world_from_sim_worlds_dir_when_it_is_set(tmp_path):
    result, calls = up_with_stand_ins(
        tmp_path, "--agent", "--", "--tool-set", "hri", setup="SIM_WORLDS_DIR=/checkout/worlds"
    )
    assert result.returncode == 0, result.stderr
    bringup = router_and_bringup(tmp_path)[1]
    assert calls[2] == (*bringup, "worlds_dir:=/checkout/worlds")


def test_up_with_the_arms_starts_moveit_then_the_commander_once_nav2_is_up(tmp_path):
    result, calls = up_with_stand_ins(tmp_path, "--agent", "--arms", "--", "--tool-set", "hri")
    assert result.returncode == 0, result.stderr
    assert calls == [
        ("check", "true"),
        *router_and_bringup(tmp_path),
        *the_arms(tmp_path),
        # The agent gets --arms after the arguments given to up.
        *oracle_speech_and_agent(tmp_path, "--tool-set", "hri", "--arms"),
    ]
    assert result.stdout.splitlines()[-1] == "up. Next: scripts/chula-sim.sh check --agent --arms"


def test_up_with_the_arms_and_no_agent_starts_no_agent(tmp_path):
    result, calls = up_with_stand_ins(tmp_path, "--arms")
    assert result.returncode == 0, result.stderr
    assert calls == [
        ("check", "true"),
        *router_and_bringup(tmp_path),
        *the_arms(tmp_path),
        *oracle_speech_and_agent(tmp_path)[:2],
        ("wait", "oracle", "Drawing ", "120"),
    ]
    assert result.stdout.splitlines()[-1] == "up. Next: scripts/chula-sim.sh check --arms"


def test_up_with_the_arms_starts_only_what_is_not_running_and_waits_for_each(tmp_path):
    result, calls = up_with_stand_ins(
        tmp_path, "--agent", "--arms", running=("router", "bringup", "moveit", "oracle", "speech")
    )
    assert result.returncode == 0, result.stderr
    arms = the_arms(tmp_path)
    assert calls == [
        ("check", "true"),
        router_and_bringup(tmp_path)[-1],
        *arms[1:],
        *oracle_speech_and_agent(tmp_path, "--arms")[2:],
    ]


@pytest.mark.parametrize(
    ("part", "message"),
    [("moveit", "MoveIt did not start"), ("commander", "the arm commander did not start")],
)
def test_up_stops_at_an_arm_part_that_never_logs_it_is_ready(tmp_path, part, message):
    result, calls = up_with_stand_ins(tmp_path, "--agent", "--arms", not_ready=part)
    assert result.returncode == 1
    assert f"chula-sim: {message}; see {tmp_path}/logs/{part}.log" in result.stderr
    # The wait for that part's line is the last call: up starts nothing after it.
    arms = the_arms(tmp_path)
    waits = {"moveit": arms[1], "commander": arms[3]}
    assert calls[-1] == waits[part]


def the_whole_body_bringup(sim_dir: Path) -> list[tuple[str, ...]]:
    """The calls up --whole-body makes to start bringup and wait for Nav2, MoveIt and the commander."""
    return [
        (
            "start", "bringup", str(sim_dir), *WALKIE_ROS_RUN,
            "ros2", "launch", "robot_bringup", "whole_body_demo.launch.py",
            "world:=chula_pat14_6", "spawn_pose:=1.7629 -1.483 0", "headless:=true",
            "use_rviz:=false", "start_router:=false", f"world_output_dir:={sim_dir}/worlds",
        ),
        ("wait", "bringup", "Managed nodes are active", "300"),
        ("wait", "bringup", "MoveGroup context initialization complete", "120"),
        ("wait", "bringup", "Bimanual Commander Node Initialized", "120"),
    ]  # fmt: skip


def test_up_with_the_whole_body_runs_the_whole_body_launch_as_bringup(tmp_path):
    result, calls = up_with_stand_ins(
        tmp_path, "--agent", "--whole-body", "--", "--tool-set", "hri"
    )
    assert result.returncode == 0, result.stderr
    assert calls == [
        # MoveIt and the commander must be built in WALKIE_ROS, as for --arms.
        ("check", "true"),
        ("check-whole-body",),
        router_and_bringup(tmp_path)[0],
        *the_whole_body_bringup(tmp_path),
        # No separate MoveIt or commander part, and the agent still gets --arms.
        *oracle_speech_and_agent(tmp_path, "--tool-set", "hri", "--arms"),
    ]
    assert result.stdout.splitlines()[-1] == (
        "up. Next: scripts/chula-sim.sh check --agent --whole-body"
    )


def test_up_with_the_whole_body_and_its_bringup_running_only_waits_for_it(tmp_path):
    result, calls = up_with_stand_ins(
        tmp_path, "--whole-body", running=("router", "bringup", "oracle", "speech")
    )
    assert result.returncode == 0, result.stderr
    assert calls == [
        ("check", "true"),
        ("check-whole-body",),
        *the_whole_body_bringup(tmp_path)[1:],
        ("wait", "oracle", "Drawing ", "120"),
    ]


@pytest.mark.parametrize(
    ("ready", "message"),
    [
        ("MoveGroup context initialization complete", "whole-body MoveIt did not start"),
        ("Bimanual Commander Node Initialized", "the arm commander did not start"),
    ],
)
def test_up_with_the_whole_body_stops_at_a_part_that_never_logs_it_is_ready(
    tmp_path, ready, message
):
    never = f'wait_for_log() {{ record wait "$@"; [[ $2 != {shlex.quote(ready)} ]]; }}'
    result, calls = up_with_stand_ins(tmp_path, "--agent", "--whole-body", setup=never)
    assert result.returncode == 1
    assert f"chula-sim: {message}; see {tmp_path}/logs/bringup.log" in result.stderr
    assert calls[-1] == ("wait", "bringup", ready, "120")


@pytest.mark.parametrize("part", ["moveit", "commander"])
def test_up_with_the_whole_body_refuses_while_an_arm_part_of_up_arms_runs(tmp_path, part):
    result, calls = up_with_stand_ins(tmp_path, "--whole-body", running=(part,))
    assert result.returncode == 1
    assert "MoveIt or the arm commander of an up --arms is running" in result.stderr
    assert calls == [("check", "true"), ("check-whole-body",)]


def test_up_refuses_the_arms_with_the_whole_body(tmp_path):
    result = run("up", "--arms", "--whole-body", sim_dir=tmp_path / "sim")
    assert result.returncode == 1
    assert "--whole-body starts MoveIt and the arm commander in bringup, so it takes no --arms" in (
        result.stderr
    )
    assert not (tmp_path / "sim").exists()


WHOLE_BODY_LAUNCH = Path("robot_bringup/share/robot_bringup/launch/whole_body_demo.launch.py")


def whole_body_launch(workspace: Path, text: str) -> Path:
    launch = workspace / "install" / WHOLE_BODY_LAUNCH
    launch.parent.mkdir(parents=True)
    launch.write_text(text)
    return launch


@pytest.mark.parametrize(
    ("in_overlay", "in_walkie_ros", "passes"),
    [
        # The overlay's robot_bringup wins over WALKIE_ROS's.
        ('LaunchConfiguration("world")', "no named worlds", True),
        ("no named worlds", 'LaunchConfiguration("world")', False),
        # Without one in the overlay, WALKIE_ROS's is the one bringup loads.
        (None, 'LaunchConfiguration("world")', True),
        (None, "no named worlds", False),
    ],
)
def test_the_whole_body_launch_must_take_a_world(tmp_path, in_overlay, in_walkie_ros, passes):
    walkie_ros = walkie_ros_workspace(tmp_path / "walkie-ros", moveit=True, commander=True)
    whole_body_launch(walkie_ros, in_walkie_ros)
    overlay = tmp_path / "overlay"
    (overlay / "install").mkdir(parents=True)
    if in_overlay is not None:
        whole_body_launch(overlay, in_overlay)
    result = sourced(
        f"WALKIE_ROS={shlex.quote(str(walkie_ros))} WALKIE_ROS_OVERLAY={shlex.quote(str(overlay))}\n"
        "check_whole_body_prerequisites && echo passed",
        tmp_path,
    )
    if passes:
        assert (result.returncode, result.stdout, result.stderr) == (0, "passed\n", "")
    else:
        assert result.returncode == 1
        assert "--whole-body needs a whole_body_demo.launch.py that takes world:=" in result.stderr


def test_the_whole_body_launch_must_be_built(tmp_path):
    walkie_ros = walkie_ros_workspace(tmp_path / "walkie-ros", moveit=True, commander=True)
    result = sourced(
        f"WALKIE_ROS={shlex.quote(str(walkie_ros))}\ncheck_whole_body_prerequisites", tmp_path
    )
    assert result.returncode == 1
    assert f"--whole-body needs {walkie_ros}/install/{WHOLE_BODY_LAUNCH}, which is not built" in (
        result.stderr
    )


def test_up_takes_the_agents_arguments_after_two_dashes_only_with_the_agent(tmp_path):
    result = run("up", "--", "--tool-set", "hri", sim_dir=tmp_path)
    assert result.returncode == 1
    assert "so up needs --agent with them" in result.stderr
    # With --agent the options parse, and up goes on to its prerequisites.
    result = run("up", "--agent", "--", "--tool-set", "hri", "--arms", sim_dir=tmp_path)
    assert result.returncode == 1
    assert "set WALKIE_ROS" in result.stderr
    assert not (tmp_path / "pids").exists()


def test_with_nothing_started_every_part_is_stopped(tmp_path):
    result = run("status", sim_dir=tmp_path)
    assert result.returncode == 0
    assert result.stdout.splitlines() == [f"{part}: stopped" for part in PARTS]
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


def test_status_lists_the_arm_parts_and_down_stops_them_before_bringup(tmp_path):
    parts = {name: sleep_with_token(f"{name}-1") for name in ("bringup", "moveit", "commander")}
    try:
        for name, process in parts.items():
            write_record(tmp_path, f"{process.pid} {BOOT_ID} {name}-1", name)
        status = run("status", sim_dir=tmp_path).stdout
        for name, process in parts.items():
            assert f"{name}: running, process group {process.pid}," in status
        result = run("down", sim_dir=tmp_path)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ["stopped commander", "stopped moveit", "stopped bringup"]
        for process in parts.values():
            assert process.wait(timeout=10) == -signal.SIGINT
    finally:
        for process in parts.values():
            process.kill()
            process.wait(timeout=10)


def test_down_refuses_while_another_up_or_down_holds_the_lock(tmp_path):
    with (tmp_path / "lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run("down", sim_dir=tmp_path)
    assert result.returncode == 1
    assert "another scripts/chula-sim.sh up, down or reset-memory is running" in result.stderr
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
        assert "another scripts/chula-sim.sh up, down or reset-memory is running" in refused.stderr

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


def test_a_part_that_looks_stopped_once_is_read_again_before_its_record_goes(tmp_path):
    # The stand-in helper finds no process on its second look only, as when a child's ID
    # was given out but /proc did not show it yet, and the part never stops.
    helper = helper_stand_in(
        tmp_path,
        "case $1 in\n"
        '  pids) n=$(( $(cat "$SIM_DIR/looks" 2>/dev/null || echo 0) + 1 ))\n'
        '        echo $n > "$SIM_DIR/looks"\n'
        "        (( n == 2 )) || echo 4242 ;;\n"
        '  signal) echo "$4" >> "$SIM_DIR/sent"; echo 1 ;;\n'
        "esac\n",
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

    `processes` maps each ID to its group and whether it carries the token. Like Linux,
    the fake gives a new process or thread the next free ID after the last one it gave,
    `last_pid`, below `pid_max`, and goes back to RESERVED_PIDS after the highest. A
    process's start time is the count of processes started before it; those given at
    first all started at 0. An action in `on_read` runs the first time an ID is read,
    before the read, as when a process starts a child and exits just after /proc was
    listed. `reserve` gives out an ID that /proc does not show until `publish`, as Linux
    gives a child its ID before it shows the child.
    """

    def __init__(
        self,
        owned: ModuleType,
        monkeypatch,
        processes: dict[int, tuple[int, bool]],
        *,
        last_pid: int | None = None,
        pid_max: int = 1000,
    ):
        self.processes = {pid: (group, token, 0) for pid, (group, token) in processes.items()}
        self.threads: dict[int, int] = {}  # each thread's ID, and its process's
        self.reserved: dict[int, tuple[int, bool]] = {}  # given out, not shown yet
        self.last_pid = max(processes) if last_pid is None else last_pid
        self.pid_max = pid_max
        self.started = 0
        self.listings = 0
        self.on_read: dict[int, Callable[[], None]] = {}
        self.events: list[str] = []
        self.descriptors: dict[int, int] = {}
        monkeypatch.setattr(owned, "_last_pid", lambda: self.last_pid)
        monkeypatch.setattr(owned, "_process_ids", self.list_ids)
        monkeypatch.setattr(owned, "_is_thread", self.is_thread)
        monkeypatch.setattr(owned, "_stat_of", self.stat_of)
        monkeypatch.setattr(owned, "_is_member", self.is_member)
        monkeypatch.setattr(owned, "pidfd_open", self.pidfd_open)
        monkeypatch.setattr(owned, "pidfd_send_signal", self.pidfd_send_signal)

    def list_ids(self) -> list[int]:
        self.listings += 1
        return sorted(self.processes)

    def _next_id(self) -> int:
        pid = self.last_pid
        while True:
            pid = pid + 1 if pid + 1 < self.pid_max else RESERVED_PIDS
            if pid not in self.processes and pid not in self.threads and pid not in self.reserved:
                self.last_pid = pid
                return pid

    def start(self, group: int, carries_token: bool) -> int:
        """Start a process in `group`, and return its ID."""
        return self.publish(self.reserve(group, carries_token))

    def reserve(self, group: int, carries_token: bool) -> int:
        """Give a process in `group` an ID, and return it; /proc does not show it yet."""
        pid = self._next_id()
        self.reserved[pid] = (group, carries_token)
        return pid

    def publish(self, pid: int) -> int:
        """Show the process given `pid` by `reserve` in /proc, and return its ID."""
        group, carries_token = self.reserved.pop(pid)
        self.started += 1
        self.processes[pid] = (group, carries_token, self.started)
        return pid

    def start_thread(self, process: int) -> int:
        """Start a thread in `process`, and return its ID."""
        thread = self._next_id()
        self.threads[thread] = process
        return thread

    def _read(self, pid: int) -> tuple[int, bool, int]:
        if action := self.on_read.pop(pid, None):
            action()
        if pid in self.threads:
            return self.processes[self.threads[pid]]
        if pid not in self.processes:
            raise FileNotFoundError(pid)
        return self.processes[pid]

    def is_thread(self, pid: int) -> bool:
        self._read(pid)
        return pid in self.threads

    def stat_of(self, pid: int) -> tuple[int, int]:
        group, _, started = self._read(pid)
        return group, started

    def is_member(self, pid: int, pgid: int, assignment: bytes) -> bool:
        self.events.append(f"check {pid}")
        try:
            group, carries_token, _ = self._read(pid)
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


def started_a_child_and_exited(fake: FakeProc, parent: int) -> Callable[[], None]:
    def action() -> None:
        group, carries_token, _ = fake.processes.pop(parent)
        fake.start(group, carries_token)

    return action


def test_the_helper_finds_a_child_its_parent_started_as_it_exited(owned, monkeypatch):
    fake = FakeProc(owned, monkeypatch, {1: (1, False), 10: (10, True)})
    fake.on_read[10] = started_a_child_and_exited(fake, 10)
    assert owned.members(10, "agent-1") == [11]
    assert fake.listings == 1  # the child is read by its ID, not found by a listing


def test_the_helper_finds_a_child_that_got_the_id_of_a_process_already_read(owned, monkeypatch):
    # The IDs have gone round once, so the next free one, 305, is below running
    # processes. 305, of another group, is read first. Then 305 exits, and 310 starts a
    # child that gets ID 305, and exits before the helper reads it.
    fake = FakeProc(owned, monkeypatch, {305: (99, False), 310: (310, True)}, last_pid=304)

    def a_child_got_305_and_its_parent_exited() -> None:
        del fake.processes[305]
        started_a_child_and_exited(fake, 310)()

    fake.on_read[310] = a_child_got_305_and_its_parent_exited
    assert owned.members(310, "agent-1") == [305]


def test_the_helper_lists_proc_again_when_the_ids_go_past_pid_max(owned, monkeypatch):
    # IDs 300 to 311 are taken, 314 was the last given, and 315 is the highest below a
    # pid_max of 316. 312 is a process of group 10, whose leader has exited. Once listed,
    # it starts a child, which gets 315, and exits. The IDs go back to 300, and a process
    # of another group gets 312.
    processes = {pid: (99, False) for pid in range(300, 312)} | {312: (10, True)}
    fake = FakeProc(owned, monkeypatch, processes, last_pid=314, pid_max=316)

    def started_a_child_exited_and_its_id_passed_on() -> None:
        started_a_child_and_exited(fake, 312)()
        assert fake.start(99, False) == 312

    fake.on_read[312] = started_a_child_exited_and_its_id_passed_on
    assert owned.members(10, "agent-1") == [315]
    assert fake.listings == 2


def test_the_helper_misses_a_child_proc_does_not_show_yet_and_finds_it_once_shown(owned, monkeypatch):
    # Once listed, 10 starts a child and exits. The child has ID 11, but /proc does not
    # show it when the helper reads 11. This is why chula-sim.sh reads a part that looks
    # stopped a second time before it removes its record.
    fake = FakeProc(owned, monkeypatch, {1: (1, False), 10: (10, True)})
    child: list[int] = []

    def gave_a_child_its_id_and_exited() -> None:
        del fake.processes[10]
        child.append(fake.reserve(10, True))

    fake.on_read[10] = gave_a_child_its_id_and_exited
    assert owned.members(10, "agent-1") == []
    fake.publish(child[0])
    assert owned.members(10, "agent-1") == [11]


def test_the_helper_does_not_take_a_thread_for_a_process(owned, monkeypatch):
    # /proc reads a thread's ID like a process's, with its process's group and token.
    fake = FakeProc(owned, monkeypatch, {10: (10, True)})
    fake.on_read[10] = lambda: fake.start_thread(10)
    assert owned.members(10, "agent-1") == [10]


def test_the_helper_reads_the_ids_of_processes_other_programs_start_and_lists_once(owned, monkeypatch):
    fake = FakeProc(owned, monkeypatch, {1: (1, False)})

    def another_program_started_and_ended_50() -> None:
        for _ in range(50):
            del fake.processes[fake.start(99, False)]

    fake.on_read[1] = another_program_started_and_ended_50
    assert owned.members(10, "agent-1") == []
    assert fake.listings == 1


# Starts a `sleep 0.02` every 5 ms, as a parallel build might start processes.
CHURN = """
import subprocess, time
running = []
while True:
    running = [p for p in running if p.poll() is None]
    running.append(subprocess.Popen(["sleep", "0.02"]))
    time.sleep(0.005)
"""


def test_the_helper_finds_an_empty_group_empty_while_other_programs_start_processes():
    # A process ID that no process has, nor any group: a stand-in that was in this
    # test's group, and has exited.
    gone = subprocess.Popen(["true"])
    gone.wait()
    churn = subprocess.Popen([sys.executable, "-I", "-c", CHURN])
    try:
        time.sleep(0.3)
        for _ in range(5):
            result = subprocess.run(
                ["python3", "-I", str(OWNED), "pids", str(gone.pid), "agent-1"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    finally:
        churn.kill()
        churn.wait()


def test_the_helper_signals_a_child_its_parent_started_as_it_exited(owned, monkeypatch):
    fake = FakeProc(owned, monkeypatch, {1: (1, False), 10: (10, True)})
    fake.on_read[10] = started_a_child_and_exited(fake, 10)
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
        if signum == signal.SIGINT and fake.descriptors[descriptor] == 10:
            fake.start(10, True)

    monkeypatch.setattr(owned, "pidfd_send_signal", starts_a_child_on_sigint)
    assert owned.send(10, "agent-1", signal.SIGINT) == 2
    assert f"signal 11 {signal.SIGINT}" in fake.events


def test_the_helper_gives_up_when_processes_keep_starting(owned, monkeypatch, capsys):
    # Linux's default pid_max, so that the IDs do not go round during the test.
    fake = FakeProc(owned, monkeypatch, {10: (10, True)}, pid_max=4194304)
    readings = []

    def a_process_started_and_ended_before_each_reading() -> int:
        del fake.processes[fake.start(99, False)]
        readings.append(fake.last_pid)
        return fake.last_pid

    monkeypatch.setattr(owned, "_last_pid", a_process_started_and_ended_before_each_reading)
    assert owned.main(["pids", "10", "agent-1"]) == 1
    assert "processes kept starting through 1000 rounds" in capsys.readouterr().err
    # The first round lists /proc and reads the last ID twice; each later round reads
    # only the IDs given out since, and the last ID once.
    assert fake.listings == 1
    assert len(readings) == owned.MAX_READINGS + 1


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
