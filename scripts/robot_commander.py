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

A group's "Run group" button starts every member at once, enabled or not. So a
button that moves the robot by itself, such as a lift command or a task that
drives, sits alone in its group, and no group mixes such a button with checks.
"Stop All Running", loading a profile and stopping the commander service stop
every command at once; the runbook's shutdown stops them card by card instead.
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
    goes through pipes, so Python must not buffer it, and ROS logs must not
    carry colour codes.
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
            "export RCUTILS_COLORIZED_OUTPUT=0",
        ]
    )


PRESETS = [
    Preset("team walkie-ros", TEAM_WALKIE_ROS),
    Preset("walkie-agent-ros", f"{EXPERIMENTS}/walkie-agent-ros"),
    Preset("walkie-vision", f"{EXPERIMENTS}/walkie-vision"),
    Preset("walkie-database", f"{EXPERIMENTS}/walkie-database"),
    # Our whole-body MoveIt checkout (feat/sim-bottle-pick on feat/moveit-full-body).
    # Its pixi.toml and pixi.lock are committed, but --as-is keeps a button from
    # ever installing into its environment, as for the team's checkout.
    Preset("walkie-ros-fullbody", f"{EXPERIMENTS}/walkie-ros-fullbody"),
]

BASE, MAP, ARMS, AGENT = "1 base", "2 map", "3 arms", "4 agent"
STOP = "0 stop the task"
WHOLE_BODY = "5 pick: whole body, one at a time"
PICK_AGENT = "6 pick: memory and agent, one at a time"
PICK_CHECKS = "7 pick: checks and settings"
# One button each: they move the robot by themselves.
LIFT_TO_TOP = "8 pick: lift to 74 cm (moves the lift)"
FETCH = "9 pick: bring the water bottle (moves)"
ARMS_HOME = "10 pick: both arms home (moves the arms)"
GROUPS = [STOP, BASE, MAP, ARMS, AGENT, WHOLE_BODY, PICK_AGENT, PICK_CHECKS, LIFT_TO_TOP, FETCH, ARMS_HOME]

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


def agent_hri(flags: str = "") -> str:
    """The runbook's T11 agent with the hri set, plus flags such as " --arms".

    It refuses to start beside another agent: two agents on /from_human would
    both act on every task. config/locations.lab.toml is not in git: the
    engineer writes it on the robot in the runbook's step 4, and until then
    every agent button stops with exit code 2.
    """
    return (
        "pgrep -af '[w]alkie_agent.main' && { echo 'An agent already runs (listed above). Stop it first.'; exit 1; }\n"
        "pixi run --frozen agent --locations config/locations.lab.toml --tool-set hri --vision"
        f" --people-memory ros{flags} --tool-log logs/agent-tools.jsonl"
    )


# What docs/guides/whole-body-real-robot.md in walkie-ros-fullbody warns about for
# its stage 3, said before a whole-body part starts or a task that uses it.
WHOLE_BODY_WARNING = (
    'echo "Whole-body goals move the base, the lift and the right arm together."\n'
    'echo "The lift ignores the plan\'s timing and rejects a cancel: a stop halts the arm and the base,'
    ' and the lift finishes its move."\n'
    'echo "There is no e-stop. Keep a hand on the robot\'s power switch."\n'
    'echo "Starting in 5 s. Press Stop to cancel."\n'
    "sleep 5\n"
)

# Stage 3 of the whole-body guide, beside the team's Nav2 instead of an identity
# map->odom: AMCL gives map->odom, and Nav2's own collision monitor runs.
WHOLE_BODY_BASE = (
    "pgrep -af '[w]hole_body_base_follower' && { echo 'The whole-body base parts already run. Stop them first.'; exit 1; }\n"
    + WHOLE_BODY_WARNING
    + "pixi run --as-is ros2 launch robot_bringup whole_body_real.launch.py"
    " static_map_odom:=false collision_monitor:=false"
)

# A second MoveIt or commander would drive the same controllers. The patterns
# match the running executables. Nothing else in a button's text may match them,
# since pgrep also sees the button's own bash process.
ONE_MOVE_GROUP = (
    "pgrep -af '[m]ove_group' && { echo 'A MoveIt already runs (listed above). Stop it first.'; exit 1; }\n"
)
ONE_COMMANDER = (
    "pgrep -af 'openarm_bimanual_commander_cpp/[c]ommander' && "
    "{ echo 'An arm commander already runs (listed above). Stop it first.'; exit 1; }\n"
)

WHOLE_BODY_MOVEIT = (
    ONE_MOVE_GROUP
    + "pixi run --as-is ros2 launch openarm_bimanual_moveit_config moveit_only.launch.py"
    " hardware_type:=real_robot base_mode:=whole_body use_rviz:=false"
)

WHOLE_BODY_COMMANDER = (
    ONE_COMMANDER
    + WHOLE_BODY_WARNING
    + "pixi run --as-is ros2 launch openarm_bimanual_commander_cpp commander.launch.py base_mode:=whole_body"
)

# The guide's "plans that move the base run slowly", before the first goal.
SLOW_WHOLE_BODY = (
    "pixi run --as-is ros2 param set /bimanual_commander whole_body_velocity_scaling 0.3 &&\n"
    "pixi run --as-is ros2 param set /bimanual_commander whole_body_acceleration_scaling 0.3"
)

# It never spells the MoveIt node's name out, so ONE_MOVE_GROUP's pgrep cannot
# mistake this button's bash process for a MoveIt.
PICK_PARTS_CHECK = (
    "NODES=$(timeout 30 pixi run --frozen ros2 node list --no-daemon)\n"
    "echo \"$NODES\" | grep -qx '/move_gr[o]up' && echo 'OK: MoveIt runs' || echo 'MISSING: MoveIt'\n"
    "echo \"$NODES\" | grep -qx '/bimanual_commander' && echo 'OK: commander runs' || echo 'MISSING: commander'\n"
    "timeout 20 pixi run --frozen ros2 lifecycle get /collision_monitor || echo 'MISSING: /collision_monitor'"
)

# The guide's "the lift is near the top" before the first whole-body goal, and
# its stop steps. 74.0 cm sits 3.5 mm below where homing stopped.
LIFT_TO_74 = (
    'echo "The lift drives to 74 cm at up to 3 cm/s, and the arms ride along in their posture."\n'
    'echo "Look along the whole travel. Nothing else may move. Starting in 5 s. Press Stop to cancel."\n'
    "sleep 5\n"
    "pixi run --frozen ros2 topic pub -w 1 --once --max-wait-time-secs 10 /lift/cmd"
    ' std_msgs/msg/Float64MultiArray "{data: [74.0]}"'
)


def db_memory(day: str) -> str:
    """The object memory node of Optional B step e, with the map ID of step a.

    The ID is made again from the map, the arena "lab" and the seed's date, so a
    map that changed after seeding is refused by the node, as in the runbook.
    """
    data = f"$HOME/walkie-data/{day}"
    return (
        f'D="{data}"\n'
        '[ -f "$D/memory.db" ] || { echo "No $D/memory.db. Build the seed and the store first'
        ' (Optional B, steps a to c)."; exit 1; }\n'
        'MAP_ID=$(pixi run --frozen python -c "from walkie_database.map_config import digest_map_files;'
        " from walkie_map_id import make_map_id;"
        f" print(make_map_id('lab', '{day}', digest_map_files('$MAP/map.yaml')))\" | tail -n 1)\n"
        '[ -n "$MAP_ID" ] || { echo "Could not make the map ID from $MAP."; exit 1; }\n'
        'echo "Map ID: $MAP_ID"\n'
        'pixi run --frozen db-memory -p db_path:="$D/memory.db" -p map_folder:="$MAP" -p map_id:="$MAP_ID"'
    )


def say_to_agent(text: str) -> str:
    """A text task on /from_human, as the runbook's step 7 sends it.

    -w 1 waits for the agent's subscriber. --max-wait-time-secs ends the wait
    after 10 s with nothing sent, so a click while no agent runs cannot fire
    minutes later, when one starts.
    """
    if "'" in text or '"' in text:
        raise ValueError(f"{text!r}: no quotes in a text task")
    return (
        "pixi run --frozen ros2 topic pub -w 1 --once --max-wait-time-secs 10 /from_human"
        f" rai_interfaces/msg/HRIMessage \"{{text: '{text}'}}\""
    )


def runbook_commands(day: str) -> list[Command]:
    """The runbook's commands, in click order. day names the people database's folder."""
    data = f"$HOME/walkie-data/{day}"
    team, agent, fullbody = "team walkie-ros", "walkie-agent-ros", "walkie-ros-fullbody"
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
            # Without clientPublish and services, a client on the LAN can only watch.
            "pixi run --as-is ros2 launch foxglove_bridge foxglove_bridge_launch.xml send_buffer_limit:=50000000"
            " 'capabilities:=[connectionGraph,assets]'",
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
            ONE_MOVE_GROUP + "pixi run --as-is ros2 launch openarm_bimanual_moveit_config moveit_only.launch.py"
            " hardware_type:=real_robot use_rviz:=false",
        ),
        Command(
            "T6 arm commander",
            ARMS,
            team,
            ONE_COMMANDER + "pixi run --as-is ros2 launch openarm_bimanual_commander_cpp commander.launch.py",
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
        Command("T11 agent", AGENT, agent, agent_hri(" --arms")),
        Command("T11 agent, no arms", AGENT, agent, agent_hri()),
        Command("Say hello (test task)", AGENT, agent, say_to_agent("Please say hello to everyone.")),
        # Optional E, the pick: whole-body MoveIt from walkie-ros-fullbody on the
        # team's graph, beside T1, T2 (with the arms), T3, T4 Nav2 and T9.
        Command("Stop the task", STOP, agent, say_to_agent("stop")),
        Command("P1 whole-body base parts", WHOLE_BODY, fullbody, WHOLE_BODY_BASE),
        Command("P2 MoveIt, whole body", WHOLE_BODY, fullbody, WHOLE_BODY_MOVEIT),
        Command("P3 commander, whole body", WHOLE_BODY, fullbody, WHOLE_BODY_COMMANDER),
        Command(
            "T7 vision, memory task",
            PICK_AGENT,
            "walkie-vision",
            "pixi run --frozen vision task:=memory respawn_max_retries:=3",
        ),
        Command("T8 database, object memory", PICK_AGENT, "walkie-database", db_memory(day)),
        Command(
            "T11 agent, pick and web search",
            PICK_AGENT,
            agent,
            agent_hri(" --arms --object-memory --web-search"),
        ),
        Command("Where did you see a bottle? (test task)", PICK_AGENT, agent, say_to_agent("Where did you see a bottle?")),
        Command(
            "Ask the web (test task)",
            PICK_AGENT,
            agent,
            say_to_agent("Look up the weather in Bangkok today and tell me."),
        ),
        Command("P check: MoveIt, commander, collision monitor", PICK_CHECKS, agent, PICK_PARTS_CHECK),
        Command(
            "P check: object memory status",
            PICK_CHECKS,
            agent,
            "timeout 20 pixi run --frozen ros2 service call /walkie_database/get_store_status"
            " walkie_database_msgs/srv/GetStoreStatus",
        ),
        Command("P slow whole-body plans", PICK_CHECKS, fullbody, SLOW_WHOLE_BODY),
        Command("P lift to 74 cm", LIFT_TO_TOP, agent, LIFT_TO_74),
        Command(
            "P bring me the water bottle",
            FETCH,
            agent,
            WHOLE_BODY_WARNING + say_to_agent("Bring me the water bottle."),
        ),
        Command("P both arms home", ARMS_HOME, agent, say_to_agent("Put both arms at home.")),
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
