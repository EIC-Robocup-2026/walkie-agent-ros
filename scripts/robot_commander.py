"""Write walkie-commander's saved commands for the real-robot runbook.

The commands follow docs/setup/09-real-robot.md, one per terminal or check.
Our copy of the commander lives in ~/Project/robocup_experiments/walkie-commander
on the robot. It reads its data folder once, at start, so stop it first, run
this with the system Python, and start it again:

    python3 scripts/robot_commander.py \\
        ~/Project/robocup_experiments/walkie-commander/walkie-commander-data

It refuses to replace existing files unless --replace is given, and then keeps
each old file as <name>.bak. The ids come from the names, so a command keeps its
id, and the commander its output history, across runs. Every command starts
disabled, so "Run Enabled" starts nothing by accident.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

ID_NAMESPACE = uuid.UUID("5f0c8a52-2d55-4a5e-9b8e-3c1f6f1d7a40")
PROFILE_NAME = "Real-robot runbook"
DEFAULT_MAP_NAME = "31_19112026_ChulaPat14_7"

TEAM_WALKIE_ROS = "~/Project/robocup_ros_ws/walkie-ros"
EXPERIMENTS = "~/Project/robocup_experiments"


@dataclass(frozen=True)
class Preset:
    name: str
    directory: str
    extra_setup: str = ""


@dataclass(frozen=True)
class Command:
    name: str
    group: str
    preset: str
    command: str
    use_sudo: bool = False
    web_port: int | None = None


def common_setup(map_name: str) -> str:
    """The runbook's "Start every terminal the same way" block, for bash -lc.

    The commander runs as a systemd user service, so PATH has no pixi. Output
    goes through pipes, so Python must not buffer it.
    """
    return "\n".join(
        [
            'export PATH="$HOME/.pixi/bin:$PATH"',
            "export ROS_DOMAIN_ID=23",
            "unset ZENOH_CONFIG_OVERRIDE ZENOH_SESSION_CONFIG_URI ZENOH_ROUTER_CONFIG_URI",
            f"export MAP_NAME={map_name}",
            'export MAP="$HOME/Project/robocup_experiments/walkie-ros-fullbody'
            '/navigation/walkie-navigation/map/$MAP_NAME"',
            "export DISPLAY=:0",
            "export PYTHONUNBUFFERED=1",
        ]
    )


PRESETS = [
    Preset("team walkie-ros", TEAM_WALKIE_ROS),
    Preset("walkie-agent-ros", f"{EXPERIMENTS}/walkie-agent-ros"),
    Preset("walkie-vision", f"{EXPERIMENTS}/walkie-vision"),
    Preset("walkie-database", f"{EXPERIMENTS}/walkie-database"),
]

BASE, MAP, ARMS, AGENT = "1 base", "2 map", "3 arms", "4 agent"
GROUPS = [BASE, MAP, ARMS, AGENT]

# The team's bringup with the arms: the arm driver moves every joint to 0 rad and
# then to a home pose when it starts, and turns the motors off when it stops.
BRINGUP_WITH_ARMS = (
    'echo "The arms swing to 0 rad when the bringup starts, and drop when it stops."\n'
    'echo "Clear the space around them. Starting in 5 s. Press Stop to cancel."\n'
    "sleep 5\n"
    "pixi run --as-is ros2 launch robot_bringup real_omnibot.launch.py"
)

# The gamepad's js number changes between connections, and the DualSense also
# shows up as a second "Motion Sensors" joystick, so find it by its name.
FIND_GAMEPAD = (
    "JOY=; for j in /sys/class/input/js*; do "
    '[ "$(cat "$j/device/name")" = "DualSense Wireless Controller" ] && JOY=/dev/input/${j##*/}; '
    "done\n"
    '[ -n "$JOY" ] || { echo "No DualSense gamepad found. Pair it, then run this again."; exit 1; }\n'
    'echo "Gamepad: $JOY"\n'
)

SAVE_MAP = (
    'if [ -e "$MAP/map.yaml" ]; then\n'
    '  echo "Refused: $MAP already has a map. Write the commands again with a new --map-name."\n'
    "  exit 1\n"
    "fi\n"
    'mkdir -p "$MAP"\n'
    'pixi run --as-is ros2 run nav2_map_server map_saver_cli -t /map -f "$MAP/map"\n'
    'ls -l "$MAP"'
)

AGENT_HRI = (
    "pixi run --frozen agent --locations config/locations.lab.toml --tool-set hri --vision"
    " --people-memory ros{arms} --tool-log logs/agent-tools.jsonl"
)


def runbook_commands(day: str) -> list[Command]:
    """The runbook's commands, in click order. day names the people database's folder."""
    data = f"$HOME/walkie-data/{day}"
    team, agent = "team walkie-ros", "walkie-agent-ros"
    return [
        Command("Zenoh router", BASE, team, "pixi run --as-is ros2 run rmw_zenoh_cpp rmw_zenohd"),
        Command("T1 lift homing", BASE, team, "pixi run --as-is ros2 launch robot_bringup lift_homing.launch.py"),
        Command("T2 bringup", BASE, team, BRINGUP_WITH_ARMS),
        Command(
            "T2 bringup, no arms",
            BASE,
            team,
            "pixi run --as-is ros2 launch robot_bringup real_omnibot.launch.py"
            " use_arm:=false use_fake_arm_hardware:=true",
        ),
        Command(
            "T3 joystick",
            BASE,
            team,
            FIND_GAMEPAD + 'pixi run --as-is ros2 launch joy_interface joy_control_smooth.launch.py dev:="$JOY"',
        ),
        Command(
            "Foxglove bridge",
            BASE,
            team,
            "pixi run --as-is ros2 launch foxglove_bridge foxglove_bridge_launch.xml send_buffer_limit:=50000000",
        ),
        Command("T4 SLAM", MAP, team, "pixi run --as-is ros2 launch robot_navigation mapping_Nav2_real.launch.py"),
        Command("Save map", MAP, team, SAVE_MAP),
        Command(
            "T4 Nav2 on $MAP",
            MAP,
            team,
            'pixi run --as-is ros2 launch robot_navigation localization_Nav2_real.launch.py map:="$MAP"'
            " use_keepout_zones:=false",
        ),
        Command("Robot pose on the map", MAP, agent, "pixi run --frozen ros2 run tf2_ros tf2_echo map base_footprint"),
        Command(
            "T5 MoveIt",
            ARMS,
            team,
            "pixi run --as-is ros2 launch openarm_bimanual_moveit_config moveit_only.launch.py"
            " hardware_type:=real_robot use_rviz:=false",
        ),
        Command(
            "T6 arm commander",
            ARMS,
            team,
            "pixi run --as-is ros2 launch openarm_bimanual_commander_cpp commander.launch.py",
        ),
        Command("T7 vision", AGENT, "walkie-vision", "pixi run --frozen vision respawn_max_retries:=3"),
        Command(
            "T8 database",
            AGENT,
            "walkie-database",
            f'mkdir -p "{data}"\npixi run --frozen db-node --ros-args -p db_path:="{data}/people.db"',
        ),
        Command("T9 speech out", AGENT, agent, "pixi run --frozen voice-tts"),
        Command("T10 speech in", AGENT, agent, "pixi run --frozen voice-asr"),
        Command("T11 agent", AGENT, agent, AGENT_HRI.format(arms=" --arms")),
        Command("T11 agent, no arms", AGENT, agent, AGENT_HRI.format(arms="")),
        Command(
            "Say hello (test task)",
            AGENT,
            agent,
            "pixi run --frozen ros2 topic pub -w 1 --once /from_human rai_interfaces/msg/HRIMessage"
            " \"{text: 'Please say hello to everyone.'}\"",
        ),
    ]


def _id(kind: str, name: str) -> str:
    return uuid.uuid5(ID_NAMESPACE, f"{kind}:{name}").hex


def build(map_name: str, day: str, now: str) -> dict[str, list[dict]]:
    """The four data files, shaped like walkie_commander.models."""
    presets_by_name = {preset.name: preset for preset in PRESETS}
    saved = runbook_commands(day)
    for command in saved:
        if command.preset not in presets_by_name:
            raise ValueError(f"{command.name}: no preset {command.preset!r}")
        if command.group not in GROUPS:
            raise ValueError(f"{command.name}: no group {command.group!r}")
    names = [command.name for command in saved]
    if len(set(names)) != len(names):
        raise ValueError("two commands share a name")

    base = common_setup(map_name)
    presets = [
        {
            "id": _id("preset", preset.name),
            "name": preset.name,
            "setup": f"{base}\n{preset.extra_setup}".rstrip(),
            "directory": preset.directory,
            "docker_container": "",
            "created_at": now,
        }
        for preset in PRESETS
    ]
    groups = [{"id": _id("group", name), "name": name, "created_at": now} for name in GROUPS]
    commands = [
        {
            "id": _id("command", command.name),
            "name": command.name,
            "command": command.command,
            "directory": presets_by_name[command.preset].directory,
            "setup": "",
            "preset_id": _id("preset", command.preset),
            "group_id": _id("group", command.group),
            "enabled": False,
            "use_sudo": command.use_sudo,
            "web_scheme": "http",
            "web_port": command.web_port,
            "web_path": "",
            "created_at": now,
        }
        for command in saved
    ]
    profile = {
        "id": _id("profile", PROFILE_NAME),
        "name": PROFILE_NAME,
        "commands": commands,
        "groups": groups,
        "presets": presets,
        "created_at": now,
    }
    return {
        "presets.json": presets,
        "groups.json": groups,
        "commands.json": commands,
        "profiles.json": [profile],
    }


def _write_private(path: Path, payload: object) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".json")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="the commander's walkie-commander-data folder")
    parser.add_argument("--map-name", default=DEFAULT_MAP_NAME, help="the map folder name for $MAP")
    parser.add_argument(
        "--day",
        default=date.today().isoformat(),
        help="the ~/walkie-data/<day> folder for the people database (default: today)",
    )
    parser.add_argument("--replace", action="store_true", help="replace existing files, keeping .bak copies")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    files = build(args.map_name, args.day, now)
    data_dir = args.data_dir.expanduser()
    existing = [name for name in files if (data_dir / name).exists()]
    if existing and not args.replace:
        raise SystemExit(f"Refused: {', '.join(existing)} already in {data_dir}. Pass --replace.")

    data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name in existing:
        shutil.copy2(data_dir / name, data_dir / f"{name}.bak")
    for name, payload in files.items():
        _write_private(data_dir / name, payload)
    print(
        f"Wrote {len(files['commands.json'])} commands in {len(GROUPS)} groups to {data_dir}"
        f" for map {args.map_name}."
    )


if __name__ == "__main__":
    main()
