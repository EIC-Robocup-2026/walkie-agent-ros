#!/usr/bin/env bash
# Run walkie-ros's Chula PAT14-6 simulation with the oracle perception, a speech log, and
# the agent when asked, on a private ROS graph: a Zenoh router of its own on SIM_ZENOH_PORT
# and ROS_DOMAIN_ID SIM_DOMAIN_ID. Nothing started here connects to the router on 7447, so
# it neither sees nor disturbs a robot or another simulation on this computer.
#
#   scripts/chula-sim.sh up [--agent] [--arms | --whole-body] [--vision] [--memory] [--object-tools] [--foxglove] [--gui] [-- ARGS]
#                                               router, MuJoCo with Nav2, oracle, speech log;
#                                               --agent adds the agent, given ARGS such as
#                                               --tool-set hri; --arms adds MoveIt and the arm
#                                               commander, and gives the agent --arms;
#                                               --whole-body runs walkie-ros's whole-body
#                                               demo as bringup instead, with whole-body
#                                               MoveIt and its commander, and gives the
#                                               agent --arms;
#                                               --vision adds walkie-vision's node and its
#                                               debug visualizer; --memory, with --vision,
#                                               adds walkie-database's object memory;
#                                               --object-tools, with --memory and --agent,
#                                               gives the agent --object-memory, and the
#                                               detector's objects unless ARGS name a topic;
#                                               --foxglove adds foxglove_bridge on
#                                               SIM_FOXGLOVE_PORT
#   scripts/chula-sim.sh check [--agent] [--arms | --whole-body]
#                                               wait for the clock, TF, head camera and arrays;
#                                               --whole-body also for move_group and /go_to_pose
#   scripts/chula-sim.sh reset-memory           restore the object memory's store from its
#                                               seed, while the memory is stopped
#   scripts/chula-sim.sh status                 which parts are running, and their logs
#   scripts/chula-sim.sh env                    the exports that put a shell on the private graph
#   scripts/chula-sim.sh down                   stop everything this script started
#
# docs/setup/08-chula-simulation.md explains each step. Environment:
#   WALKIE_ROS          a walkie-ros workspace, with its pixi environment installed and its
#                       install/ built (required); for --arms and --whole-body, install/
#                       must hold openarm_bimanual_moveit_config and
#                       openarm_bimanual_commander_cpp, as feat/moveit-full-body builds them
#   WALKIE_ROS_OVERLAY  a colcon workspace sourced over WALKIE_ROS's install, such as a
#                       worktree of feat/chula-pat14-6-simulation with only robot_bringup and
#                       walkie_mujoco_worlds built (optional); for --whole-body, the
#                       robot_bringup it loads must run whole_body_demo.launch.py in a
#                       named world (world:=)
#   WALKIE_VISION       for --vision, a walkie-vision checkout with its pixi environment
#                       installed, its messages built and its models fetched
#   SIM_VISION_TASK     the vision task, a file in its config/tasks/ (default memory)
#   WALKIE_DATABASE     for --memory and reset-memory, a walkie-database checkout on
#                       feat/scene-graph-prep or later, with its pixi environment installed
#                       and its messages built
#   SIM_MEMORY_DB       the object memory's store, restored from SIM_MEMORY_SEED by
#                       reset-memory (default /var/tmp/scene-graph-demo/sim/walkie.db)
#   SIM_MEMORY_SEED     the seed reset-memory restores the store from, built for this map
#                       by walkie-database's db-seed-sim
#                       (default /var/tmp/scene-graph-demo/furniture/seed.db)
#   SIM_MEMORY_MAP_ID   the map ID the store was seeded with (default chula-2026-11-12-714b087d)
#   SIM_MEMORY_MAP_FOLDER
#                       the map folder the store was seeded with (default the chula_pat14_6
#                       map that bringup loads, from WALKIE_ROS_OVERLAY's install/ when set,
#                       else WALKIE_ROS's)
#   SIM_FOXGLOVE_PORT   foxglove_bridge's port on 127.0.0.1 (default 8799)
#   SIM_DIR             logs, process groups, ROS logs and the world cache
#                       (default /var/tmp/walkie-chula-sim)
#   SIM_ZENOH_PORT      the private router's port (default 7499)
#   SIM_DOMAIN_ID       ROS_DOMAIN_ID (default 77)
#   SIM_SPAWN_POSE      "X Y YAW_DEG" in the map frame (default the living room place,
#                       facing the oracle scene's two people)
#   SIM_CPUS            when set, every process runs on these CPUs (taskset -c)
#   SIM_WORLDS_DIR      when set, bringup loads chula_pat14_6 from this worlds directory,
#                       such as a walkie-ros checkout's simulation/mujoco/worlds, instead
#                       of the installed walkie_mujoco_worlds (worlds_dir:=)
#
# The agent reads its model keys as it always does: from the environment, or from the
# .env at the root of this repository. The script itself needs python3, 3.9 or newer, on
# PATH, for scripts/chula_sim_owned.py.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PIXI=${PIXI:-$(command -v pixi || echo "$HOME/.pixi/bin/pixi")}
WALKIE_ROS=${WALKIE_ROS:-}
WALKIE_ROS_OVERLAY=${WALKIE_ROS_OVERLAY:-}
WALKIE_VISION=${WALKIE_VISION:-}
SIM_VISION_TASK=${SIM_VISION_TASK:-memory}
WALKIE_DATABASE=${WALKIE_DATABASE:-}
SIM_MEMORY_DB=${SIM_MEMORY_DB:-/var/tmp/scene-graph-demo/sim/walkie.db}
SIM_MEMORY_SEED=${SIM_MEMORY_SEED:-/var/tmp/scene-graph-demo/furniture/seed.db}
SIM_MEMORY_MAP_ID=${SIM_MEMORY_MAP_ID:-chula-2026-11-12-714b087d}
SIM_MEMORY_MAP_FOLDER=${SIM_MEMORY_MAP_FOLDER:-${WALKIE_ROS_OVERLAY:-$WALKIE_ROS}/install/walkie_mujoco_worlds/share/walkie_mujoco_worlds/worlds/chula_pat14_6/map}
SIM_FOXGLOVE_PORT=${SIM_FOXGLOVE_PORT:-8799}
SIM_DIR=${SIM_DIR:-/var/tmp/walkie-chula-sim}
SIM_ZENOH_PORT=${SIM_ZENOH_PORT:-7499}
SIM_DOMAIN_ID=${SIM_DOMAIN_ID:-77}
# config/locations.chula_pat14_6.toml's "living room" (yaw 0 rad). The world's own
# spawn faces south, away from both people of config/scenes/chula_living_room.toml.
SIM_SPAWN_POSE=${SIM_SPAWN_POSE:-1.7629 -1.483 0}
SIM_CPUS=${SIM_CPUS:-}
SIM_WORLDS_DIR=${SIM_WORLDS_DIR:-}

# Where a walkie-database message build keeps the services' type descriptions.
DATABASE_SRV=ros2_ws/install/walkie_database_msgs/share/walkie_database_msgs/srv
WORLD=chula_pat14_6
# down stops them in reverse order: the commander and MoveIt before bringup, and the
# object memory, which saves what it merged as it stops, before vision.
PARTS=(router bringup moveit commander vision memory foxglove oracle speech agent)
# Nav2's lifecycle manager logs this once every navigation server is active.
BRINGUP_READY="Managed nodes are active"
BRINGUP_TIMEOUT_S=300
ROUTER_TIMEOUT_S=30
# What move_group logs once it is ready, and what walkie-ros's commander logs once its
# node is up (commander_template.cpp). The commander needs move_group at its start.
MOVEIT_READY="MoveGroup context initialization complete"
COMMANDER_READY="Bimanual Commander Node Initialized"
# What up --arms needs in WALKIE_ROS's install/. The overlay that
# docs/setup/08-chula-simulation.md builds holds neither package.
MOVEIT_LAUNCH=openarm_bimanual_moveit_config/share/openarm_bimanual_moveit_config/launch/moveit_only.launch.py
COMMANDER_EXECUTABLE=openarm_bimanual_commander_cpp/lib/openarm_bimanual_commander_cpp/commander
# What up --whole-body runs as bringup: Nav2, move_group with base_mode:=whole_body, the
# commander and the base follower, in one launch, from the overlay's robot_bringup when
# it has one. Its MoveIt and commander come from WALKIE_ROS's install/, as for --arms.
WHOLE_BODY_LAUNCH=robot_bringup/share/robot_bringup/launch/whole_body_demo.launch.py
# What walkie-vision's node logs once its models are loaded, how long that may take, and
# what it logs when it cannot start (its launch then respawns it, so it never stops).
# The models load in about 18 s here. On a GPU that ONNX Runtime has no machine code
# for (an RTX 50), the first start also compiles the face models' kernels, which can
# take minutes; the driver caches them in ~/.nv/ComputeCache.
VISION_READY="loaded the models in"
VISION_TIMEOUT_S=300
VISION_FAILED="cannot start:"
# What walkie-database's node logs once it has loaded the store's objects, and what it
# logs, before it exits, when the store does not match the map, map ID or clock.
MEMORY_READY="The object memory starts with"
MEMORY_FAILED="Not starting:"
# walkie-vision's sim detections: the object memory's input, and with --object-tools the
# agent's find_objects too.
DETECTOR_OBJECTS=/walkie_vision/sim/objects
# What foxglove_bridge logs once its server listens.
FOXGLOVE_READY="Server listening on port"
# What the oracle and the agent log once their nodes exist (oracle_scene.py, main.py).
ORACLE_READY="Drawing "
AGENT_READY="Walkie's agent is running"
PART_TIMEOUT_S=120
# The speech listener logs nothing until a line arrives; a bad start exits within this.
SETTLE_S=3
# How long start waits for a part's record, and stop for a part to exit after each signal
# before it sends the next. Each of stop's ten checks a second also starts the helper, so
# its waits run a little longer.
START_TIMEOUT_S=5
STOP_INT_S=10
STOP_TERM_S=5
STOP_KILL_S=2
# How long stop waits before it reads a part that looked stopped a second time (settled).
CONFIRM_S=0.2
# Every process a part starts carries CHULA_SIM_TOKEN, and the part's record keeps it.
TOKEN_VARIABLE=CHULA_SIM_TOKEN
# Lists and signals a part's processes through pidfds (scripts/chula_sim_owned.py).
OWNED=(python3 -I "$REPO/scripts/chula_sim_owned.py")

# Every process joins the private graph. Pixi sets RMW_IMPLEMENTATION=rmw_zenoh_cpp in
# both repositories; the override replaces rmw_zenoh's default router, localhost:7447.
export ROS_DOMAIN_ID=$SIM_DOMAIN_ID
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"tcp/localhost:$SIM_ZENOH_PORT\"]"
export ROS_HOME=$SIM_DIR/ros
export ROS_LOG_DIR=$SIM_DIR/ros/log

PIN=()
if [[ -n $SIM_CPUS ]]; then
    PIN=(taskset -c "$SIM_CPUS")
fi

die() {
    echo "chula-sim: $*" >&2
    exit 1
}

boot_id() {
    cat /proc/sys/kernel/random/boot_id
}

# A part's record, pids/NAME, is one line: its process group ID, the boot it started in,
# and the token that every process of the part carries in CHULA_SIM_TOKEN.
pgid_of() {
    local pgid= _
    if [[ -f $SIM_DIR/pids/$1 ]]; then
        read -r pgid _ < "$SIM_DIR/pids/$1" || true
    fi
    echo "$pgid"
}

# owner NAME: print NAME's process group ID and token, if its record is from this boot.
owner() {
    local pgid boot token
    [[ -f $SIM_DIR/pids/$1 ]] || return 1
    read -r pgid boot token < "$SIM_DIR/pids/$1" || return 1
    [[ $pgid =~ ^[1-9][0-9]*$ && $boot == "$(boot_id)" && $token =~ ^[a-z]+-[0-9a-f-]+$ ]] || return 1
    echo "$pgid $token"
}

# state NAME: print "running" while NAME's process group holds a process that carries
# NAME's token, "stopped" when it holds none or NAME has no record from this boot, and
# "unknown" when the helper fails. A group ID that the system has since given to another
# group, after a reboot or after the part exited, does not count.
state() {
    local who pids
    if ! who=$(owner "$1"); then
        echo stopped
    # shellcheck disable=SC2086 # owner prints two words, a group ID and a token
    elif ! pids=$("${OWNED[@]}" pids $who); then
        echo unknown
    elif [[ -n $pids ]]; then
        echo running
    else
        echo stopped
    fi
}

# running NAME: NAME's state is running. For up, which stops when the state is unknown;
# stop and status read the state themselves, so that they can go on to the other parts.
running() {
    case $(state "$1") in
        running) return 0 ;;
        stopped) return 1 ;;
        *) die "could not look up the processes of $1" ;;
    esac
}

# signal_part NAME SIGNAL: send SIGNAL to each of NAME's processes, each through a pidfd,
# so that a process that exits meanwhile never passes the signal to a later holder of
# its ID. Fails when the helper does.
signal_part() {
    local who
    who=$(owner "$1") || return 0
    # shellcheck disable=SC2086 # owner prints two words, a group ID and a token
    "${OWNED[@]}" signal $who "$2" > /dev/null
}

# start NAME DIR COMMAND...: run COMMAND in DIR as the leader of a new process group,
# so that stop can signal it with every child it starts. The leader writes the part's
# record, through a temporary file so that a reader never sees half of it. Its PID is
# the group's ID. The leader keeps descriptor 9, the lock, until it has written the
# record, and closes it as it executes COMMAND. A down therefore cannot run between a
# start that up gave up on and the part that start then runs, and no part holds the lock.
start() {
    local name=$1 dir=$2 token
    shift 2
    local pidfile=$SIM_DIR/pids/$name
    token=$name-$(cat /proc/sys/kernel/random/uuid)
    rm -f "$pidfile"
    (cd "$dir" && exec env "$TOKEN_VARIABLE=$token" setsid bash -c \
        'echo "$$ $1 $2" > "$0.new" && mv "$0.new" "$0" && shift 2 && exec "$@" 9>&-' \
        "$pidfile" "$(boot_id)" "$token" ${PIN[@]+"${PIN[@]}"} "$@" > "$SIM_DIR/logs/$name.log" 2>&1 < /dev/null &)
    for _ in $(seq $((START_TIMEOUT_S * 10))); do
        [[ -s $pidfile ]] && break
        sleep 0.1
    done
    [[ -s $pidfile ]] || fail "$name wrote no record within $START_TIMEOUT_S s; see $SIM_DIR/logs/$name.log. If it starts later, status shows it and down stops it"
    echo "started $name, process group $(pgid_of "$name"), log $SIM_DIR/logs/$name.log"
}

# settled NAME: print NAME's state, but read a part with a record that looks stopped
# again after CONFIRM_S. Linux gives a new process its ID before /proc shows it, so a
# child that the part's last process started just before it exited can look absent to
# the helper for a moment. Its parent is gone, so no later listing would find it, and
# stop would remove the record of a part still running. It shows within microseconds
# as a rule, so the second reading finds it. A child held back longer than CONFIRM_S
# could still be missed: this is a margin, not a proof.
settled() {
    local now
    now=$(state "$1")
    if [[ $now == stopped ]] && owner "$1" > /dev/null; then
        sleep "$CONFIRM_S"
        now=$(state "$1")
    fi
    echo "$now"
}

# settle NAME SECONDS: print NAME's settled state once it is no longer running, or after
# SECONDS, whichever comes first.
settle() {
    local name=$1 seconds=$2 now
    for _ in $(seq $((seconds * 10))); do
        now=$(settled "$name")
        if [[ $now != running ]]; then
            echo "$now"
            return 0
        fi
        sleep 0.1
    done
    settled "$name"
}

# stop NAME: SIGINT to every process of NAME, as Ctrl+C sends it to every process of a
# terminal's foreground group, then SIGTERM, then SIGKILL. Each signal goes only if NAME
# still runs after the wait for the one before. SIGKILL goes twice, for a child started
# during the first. A part that outlives them all, or whose processes the helper cannot
# look up or signal, keeps its record, and stop fails. Every state it acts on is
# settled, so a record is removed only after two readings found no process.
stop() {
    local name=$1 now step signalled=false
    now=$(settled "$name")
    for step in "INT $STOP_INT_S" "TERM $STOP_TERM_S" "KILL $STOP_KILL_S" "KILL $STOP_KILL_S"; do
        [[ $now == running ]] || break
        if ! signal_part "$name" "${step% *}"; then
            now=unknown
            break
        fi
        signalled=true
        now=$(settle "$name" "${step#* }")
    done
    case $now in
        stopped)
            if $signalled; then
                echo "stopped $name"
            fi
            rm -f "$SIM_DIR/pids/$name"
            ;;
        running)
            echo "chula-sim: $name did not stop; its record stays in $SIM_DIR/pids" >&2
            return 1
            ;;
        *)
            echo "chula-sim: could not look up or signal the processes of $name; its record stays in $SIM_DIR/pids" >&2
            return 1
            ;;
    esac
}

# fail MESSAGE: stop up with MESSAGE, and say which parts it leaves running, counting a
# part whose state is unknown as running.
fail() {
    local name left=()
    for name in "${PARTS[@]}"; do
        if [[ $(state "$name") != stopped ]]; then
            left+=("$name")
        fi
    done
    if ((${#left[@]})); then
        die "$*. Still running: ${left[*]}. scripts/chula-sim.sh down stops them."
    fi
    die "$*"
}

# lock: one up, down or reset-memory at a time, on this SIM_DIR. The lock is released
# when the script exits.
lock() {
    mkdir -p "$SIM_DIR"
    exec 9> "$SIM_DIR/lock"
    flock -n 9 || die "another scripts/chula-sim.sh up, down or reset-memory is running on $SIM_DIR"
}

# tcp_listening PORT: something accepts connections on 127.0.0.1:PORT.
tcp_listening() {
    (exec 3<> "/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

port_open() {
    tcp_listening "$SIM_ZENOH_PORT"
}

# wait_for_log NAME TEXT SECONDS [FAIL_TEXT]: until TEXT is in NAME's log; fails when
# NAME stops or FAIL_TEXT appears. A part whose launch respawns it never stops, so
# its startup failure is only in the log.
wait_for_log() {
    local name=$1 text=$2 seconds=$3 fail_text=${4:-}
    for _ in $(seq "$seconds"); do
        grep -q -F "$text" "$SIM_DIR/logs/$name.log" 2>/dev/null && return 0
        [[ -n $fail_text ]] && grep -q -F "$fail_text" "$SIM_DIR/logs/$name.log" 2>/dev/null && break
        running "$name" || break
        sleep 1
    done
    tail -n 20 "$SIM_DIR/logs/$name.log" >&2 || true
    return 1
}

# check_prerequisites WITH_ARMS: what up needs, and with WITH_ARMS true also MoveIt's
# config and the commander, both from WALKIE_ROS's install/.
check_prerequisites() {
    local with_arms=$1
    [[ -n $WALKIE_ROS ]] || die "set WALKIE_ROS to a walkie-ros workspace"
    [[ -f $WALKIE_ROS/pixi.toml && -f $WALKIE_ROS/install/setup.bash ]] ||
        die "WALKIE_ROS=$WALKIE_ROS has no pixi.toml or no built install/"
    if $with_arms; then
        [[ -f $WALKIE_ROS/install/$MOVEIT_LAUNCH ]] ||
            die "--arms needs openarm_bimanual_moveit_config, and WALKIE_ROS=$WALKIE_ROS has not built it in install/"
        [[ -x $WALKIE_ROS/install/$COMMANDER_EXECUTABLE ]] ||
            die "--arms needs openarm_bimanual_commander_cpp, and WALKIE_ROS=$WALKIE_ROS has not built it in install/"
    fi
    if [[ -n $WALKIE_ROS_OVERLAY && ! -f $WALKIE_ROS_OVERLAY/install/local_setup.bash ]]; then
        die "WALKIE_ROS_OVERLAY=$WALKIE_ROS_OVERLAY has no built install/"
    fi
    [[ -d $REPO/ros2_ws/install/walkie_vision_msgs ]] ||
        die "walkie_vision_msgs is not built here: run pixi run build-ros2"
    # Without it the oracle and the agent load RoboStack's rmw_zenoh_cpp 0.2.10, whose
    # rmw_wait can block forever (ADR-0032).
    [[ -f $REPO/rmw_zenoh_ws/install/rmw_zenoh_cpp/lib/librmw_zenoh_cpp.so ]] ||
        die "rmw_zenoh_cpp 0.2.11 is not built here: run pixi run build-rmw-zenoh"
}

# check_whole_body_prerequisites: the whole-body launch that up --whole-body runs, from
# the overlay's robot_bringup when it has one, else WALKIE_ROS's, must take world:=. An
# older one, without named worlds, would start its own obstacle scene instead.
check_whole_body_prerequisites() {
    local launch=$WALKIE_ROS/install/$WHOLE_BODY_LAUNCH
    if [[ -n $WALKIE_ROS_OVERLAY && -d $WALKIE_ROS_OVERLAY/install/robot_bringup ]]; then
        launch=$WALKIE_ROS_OVERLAY/install/$WHOLE_BODY_LAUNCH
    fi
    [[ -f $launch ]] || die "--whole-body needs $launch, which is not built"
    grep -q -w world "$launch" ||
        die "--whole-body needs a whole_body_demo.launch.py that takes world:=, and $launch does not"
}

# check_vision_prerequisites: what up --vision needs from WALKIE_VISION. The models are
# left to the vision launch, which names the command that fetches them.
check_vision_prerequisites() {
    [[ -n $WALKIE_VISION ]] || die "--vision needs WALKIE_VISION, a walkie-vision checkout"
    [[ -f $WALKIE_VISION/pixi.toml && -d $WALKIE_VISION/.pixi/envs/default ]] ||
        die "WALKIE_VISION=$WALKIE_VISION has no pixi.toml or no installed environment: run pixi install there"
    [[ -f $WALKIE_VISION/src/walkie_vision/debug_viz.py ]] ||
        die "WALKIE_VISION=$WALKIE_VISION has no debug visualizer; --vision needs feat/scene-graph-prep or later"
    [[ -f $WALKIE_VISION/config/tasks/$SIM_VISION_TASK.yaml ]] ||
        die "WALKIE_VISION=$WALKIE_VISION has no task $SIM_VISION_TASK in config/tasks/"
    [[ -d $WALKIE_VISION/ros2_ws/install/walkie_vision_msgs ]] ||
        die "walkie_vision_msgs is not built in WALKIE_VISION=$WALKIE_VISION: run pixi run build-msgs there"
    # The vision part starts in WALKIE_VISION, so a relative path would resolve twice.
    WALKIE_VISION=$(cd "$WALKIE_VISION" && pwd)
}

# check_database_prerequisites COMMAND: what COMMAND, --memory or reset-memory, needs
# from WALKIE_DATABASE, and that the map folder exists.
check_database_prerequisites() {
    [[ -n $WALKIE_DATABASE ]] || die "$1 needs WALKIE_DATABASE, a walkie-database checkout"
    [[ -f $WALKIE_DATABASE/pixi.toml && -d $WALKIE_DATABASE/.pixi/envs/default ]] ||
        die "WALKIE_DATABASE=$WALKIE_DATABASE has no pixi.toml or no installed environment: run pixi install there"
    # A checkout with the object memory logs MEMORY_READY; looking for that line, not for
    # a module, keeps the check true when walkie-database moves its files.
    grep -rqF --include='*.py' -- "$MEMORY_READY" "$WALKIE_DATABASE/src/walkie_database" 2>/dev/null ||
        die "WALKIE_DATABASE=$WALKIE_DATABASE has no object memory; $1 needs feat/scene-graph-prep or later"
    [[ -d $WALKIE_DATABASE/ros2_ws/install/walkie_database_msgs ]] ||
        die "walkie_database_msgs is not built in WALKIE_DATABASE=$WALKIE_DATABASE: run pixi run build-msgs there"
    [[ -d $SIM_MEMORY_MAP_FOLDER ]] || die "SIM_MEMORY_MAP_FOLDER=$SIM_MEMORY_MAP_FOLDER is not a folder"
    # walkie-database's tasks run in WALKIE_DATABASE, so a relative path would resolve twice.
    WALKIE_DATABASE=$(cd "$WALKIE_DATABASE" && pwd)
    SIM_MEMORY_MAP_FOLDER=$(cd "$SIM_MEMORY_MAP_FOLDER" && pwd)
}

# check_memory_prerequisites: what up --memory needs from WALKIE_DATABASE, and that the
# store and the map folder exist. The node itself checks that the store was seeded for
# this map, map ID and clock, and logs what does not match.
check_memory_prerequisites() {
    check_database_prerequisites --memory
    [[ -f $SIM_MEMORY_DB ]] ||
        die "SIM_MEMORY_DB=$SIM_MEMORY_DB does not exist; scripts/chula-sim.sh reset-memory restores it from SIM_MEMORY_SEED"
    SIM_MEMORY_DB=$(cd "$(dirname "$SIM_MEMORY_DB")" && pwd)/$(basename "$SIM_MEMORY_DB")
}

# check_object_tools_prerequisites MEMORY_SRV: that the services the object tools call
# have the same type hashes in the agent's walkie_database_msgs as in MEMORY_SRV, the
# memory node's build of them. Under rmw_zenoh_cpp a client built from another version
# of a service finds it but is never answered: each call times out after 7 s as if the
# memory were busy, and only a warning in the agent's log hints at a stale build
# (ADR-0035, decision SG-25). A service's RIHS01 hash covers every type it uses, so only
# each service's own hash is compared. Run under the lock, after check_memory_prerequisites.
check_object_tools_prerequisites() {
    local differ
    differ=$(python3 -I -c '
import json
import sys

def hash_of(folder, name):
    try:
        with open(f"{folder}/{name}.json") as file:
            hashes = json.load(file)["type_hashes"]
        return next(
            entry["hash_string"] for entry in hashes
            if entry["type_name"] == f"walkie_database_msgs/srv/{name}"
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None

agent, memory, names = sys.argv[1], sys.argv[2], sys.argv[3:]
print(", ".join(
    name for name in names
    if (found := hash_of(agent, name)) is None or found != hash_of(memory, name)
))
' "$REPO/$DATABASE_SRV" "$1" SearchObjects GetRoomContents ListRooms GetObject SetObjectState) ||
        fail "could not compare the object services' type hashes"
    [[ -z $differ ]] ||
        fail "the object tools would never reach the memory node: $differ differ between the agent's walkie_database_msgs and the memory node's in $1, or are missing. Build both from the walkie-database commit ros2_ws.repos pins: pixi run build-msgs in WALKIE_DATABASE and pixi run build-ros2 here, and stop a memory node that runs from another build"
}

# record_memory_services: keep a copy of the services the memory node is about to load,
# so that a later up --object-tools checks the agent against the build that runs. A
# --symlink-install build installs each file as a link into its build folder, which the
# next build rewrites, so the copy follows the links (-L).
record_memory_services() {
    rm -rf "$SIM_DIR/memory-srv"
    cp -RL "$WALKIE_DATABASE/$DATABASE_SRV" "$SIM_DIR/memory-srv" ||
        fail "could not record the memory node's services in $SIM_DIR/memory-srv"
}

up() {
    local with_agent=false with_arms=false with_vision=false with_memory=false with_foxglove=false
    local with_object_tools=false with_whole_body=false headless=true rviz=false agent_args=()
    while (($#)); do
        case $1 in
            --agent) with_agent=true ;;
            --arms) with_arms=true ;;
            --whole-body) with_whole_body=true ;;
            --vision) with_vision=true ;;
            --memory) with_memory=true ;;
            --object-tools) with_object_tools=true ;;
            --foxglove) with_foxglove=true ;;
            --gui) headless=false rviz=true ;;
            --) shift; agent_args=("$@"); break ;;
            *) die "unknown option for up: $1" ;;
        esac
        shift
    done
    ((${#agent_args[@]} == 0)) || $with_agent ||
        die "the arguments after -- are the agent's, so up needs --agent with them"
    if $with_memory && ! $with_vision; then
        die "the object memory reads vision's detections, so --memory needs --vision"
    fi
    if $with_object_tools && ! { $with_memory && $with_agent; }; then
        die "the object tools are the agent's and read the object memory, so --object-tools needs --memory and --agent"
    fi
    if $with_arms && $with_whole_body; then
        die "--whole-body starts MoveIt and the arm commander in bringup, so it takes no --arms"
    fi
    if $with_whole_body; then
        check_prerequisites true
        check_whole_body_prerequisites
    else
        check_prerequisites "$with_arms"
    fi
    if $with_vision; then
        check_vision_prerequisites
    fi
    if $with_memory; then
        check_memory_prerequisites
    fi
    lock
    mkdir -p "$SIM_DIR"/{logs,pids,worlds,ros/log}
    # A memory node that is already running keeps the build it started with.
    if $with_object_tools; then
        if running memory; then
            [[ -d $SIM_DIR/memory-srv ]] ||
                fail "the object memory runs from a build that up did not record, so the object tools cannot be checked against it; stop it first"
            check_object_tools_prerequisites "$SIM_DIR/memory-srv"
        else
            check_object_tools_prerequisites "$WALKIE_DATABASE/$DATABASE_SRV"
        fi
    fi

    # The whole-body launch runs a commander of its own, and a second one would serve the
    # same actions on the same graph.
    if $with_whole_body && { running moveit || running commander; }; then
        fail "MoveIt or the arm commander of an up --arms is running; scripts/chula-sim.sh down stops them"
    fi
    if ! running bringup; then
        # A second MuJoCo simulation, on any graph, competes for the CPU with the first.
        local others
        others=$(pgrep -f "[n]av_mujoco_omnibot.launch.py|[w]hole_body_demo.launch.py" || true)
        [[ -z $others ]] || fail "another simulation is running (PIDs $(echo $others)); stop it first"
    fi

    if ! running router; then
        port_open && fail "port $SIM_ZENOH_PORT is in use; set SIM_ZENOH_PORT to a free port"
        start router "$SIM_DIR" env ZENOH_CONFIG_OVERRIDE="listen/endpoints=[\"tcp/localhost:$SIM_ZENOH_PORT\"]" \
            "$PIXI" run --as-is --manifest-path "$WALKIE_ROS/pixi.toml" ros2 run rmw_zenoh_cpp rmw_zenohd
        for _ in $(seq "$ROUTER_TIMEOUT_S"); do
            port_open && break
            sleep 1
        done
        port_open || fail "the router is not listening on $SIM_ZENOH_PORT; see $SIM_DIR/logs/router.log"
    fi

    # walkie-ros's parts run in its environment, with the overlay, when set, sourced over
    # its install/, so that MoveIt and the commander load the packages bringup loads.
    # --as-is: run walkie-ros's environment as installed; never install or update it.
    local walkie_ros=("$PIXI" run --as-is --manifest-path "$WALKIE_ROS/pixi.toml"
        bash -c '[[ -z $0 ]] || source "$0"; exec "$@"'
        "${WALKIE_ROS_OVERLAY:+$WALKIE_ROS_OVERLAY/install/local_setup.bash}")
    if ! running bringup; then
        local launch=nav_mujoco_omnibot.launch.py launch_args=()
        if $with_whole_body; then
            # This script's router serves the graph, so the launch starts none.
            launch=whole_body_demo.launch.py launch_args=(start_router:=false)
        fi
        start bringup "$SIM_DIR" "${walkie_ros[@]}" \
            ros2 launch robot_bringup "$launch" "world:=$WORLD" \
            "spawn_pose:=$SIM_SPAWN_POSE" "headless:=$headless" "use_rviz:=$rviz" \
            "${launch_args[@]}" "world_output_dir:=$SIM_DIR/worlds" \
            ${SIM_WORLDS_DIR:+"worlds_dir:=$SIM_WORLDS_DIR"}
    fi
    echo "waiting up to ${BRINGUP_TIMEOUT_S} s for Nav2 ..."
    wait_for_log bringup "$BRINGUP_READY" "$BRINGUP_TIMEOUT_S" ||
        fail "the simulation did not become ready; see $SIM_DIR/logs/bringup.log"
    if $with_whole_body; then
        # The launch starts the commander once move_group serves its actions; both log
        # to bringup's log.
        echo "waiting up to ${PART_TIMEOUT_S} s each for whole-body MoveIt and the arm commander ..."
        wait_for_log bringup "$MOVEIT_READY" "$PART_TIMEOUT_S" ||
            fail "whole-body MoveIt did not start; see $SIM_DIR/logs/bringup.log"
        wait_for_log bringup "$COMMANDER_READY" "$PART_TIMEOUT_S" ||
            fail "the arm commander did not start; see $SIM_DIR/logs/bringup.log"
        # With --agent, the agent then drives the arms through the commander.
        agent_args+=(--arms)
    fi

    if $with_arms; then
        # MoveIt drives the arm controllers that bringup spawns, and the commander
        # connects to MoveIt as it starts, so each waits for the part before it.
        echo "waiting up to ${PART_TIMEOUT_S} s each for MoveIt and the arm commander ..."
        running moveit || start moveit "$SIM_DIR" "${walkie_ros[@]}" \
            ros2 launch openarm_bimanual_moveit_config moveit_only.launch.py \
            hardware_type:=mujoco use_rviz:=false
        wait_for_log moveit "$MOVEIT_READY" "$PART_TIMEOUT_S" ||
            fail "MoveIt did not start; see $SIM_DIR/logs/moveit.log"
        # commander.launch.py sets no use_sim_time; this launch file sets it around it.
        running commander || start commander "$SIM_DIR" "${walkie_ros[@]}" \
            ros2 launch "$REPO/launch/commander_sim.launch.py"
        wait_for_log commander "$COMMANDER_READY" "$PART_TIMEOUT_S" ||
            fail "the arm commander did not start; see $SIM_DIR/logs/commander.log"
        # With --agent, the agent then drives the arms through the commander.
        agent_args+=(--arms)
    fi

    # Vision loads its models while the oracle and the agent start; up waits for it
    # after them. It runs in walkie-vision's own environment, whose activation sources
    # its message overlay, on the sim topics (sim:=true) with the debug visualizer.
    if $with_vision && ! running vision; then
        start vision "$WALKIE_VISION" "$PIXI" run --as-is --manifest-path "$WALKIE_VISION/pixi.toml" \
            vision "task:=$SIM_VISION_TASK" sim:=true viz:=true
    fi
    # The object memory merges vision's sim detections into SIM_MEMORY_DB on the sim
    # clock, and saves them once more as it stops. It runs in walkie-database's own
    # environment, whose activation sources its message overlay.
    if $with_memory && ! running memory; then
        record_memory_services
        start memory "$WALKIE_DATABASE" "$PIXI" run --as-is --manifest-path "$WALKIE_DATABASE/pixi.toml" \
            db-memory -p "db_path:=$SIM_MEMORY_DB" -p "map_folder:=$SIM_MEMORY_MAP_FOLDER" \
            -p "map_id:=$SIM_MEMORY_MAP_ID" -p use_sim_time:=true -p "objects_topic:=$DETECTOR_OBJECTS"
    fi
    # foxglove_bridge comes from walkie-ros's environment, which has no walkie_vision_msgs,
    # so it serves standard message types only; the visualizer's outputs are all standard.
    # It listens on the loopback interface only.
    if $with_foxglove && ! running foxglove; then
        tcp_listening "$SIM_FOXGLOVE_PORT" &&
            fail "port $SIM_FOXGLOVE_PORT is in use; set SIM_FOXGLOVE_PORT to a free port"
        start foxglove "$SIM_DIR" "${walkie_ros[@]}" \
            ros2 launch foxglove_bridge foxglove_bridge_launch.xml "port:=$SIM_FOXGLOVE_PORT" \
            address:=127.0.0.1 use_sim_time:=true
    fi

    running oracle || start oracle "$REPO" "$PIXI" run --frozen --manifest-path "$REPO/pixi.toml" oracle-chula
    # The agent's say needs a listener on /to_human. This one writes each line to
    # speech.log instead of voicing it, so the simulation is silent.
    running speech || start speech "$REPO" env PYTHONUNBUFFERED=1 "$PIXI" run --frozen \
        --manifest-path "$REPO/pixi.toml" ros2 topic echo /to_human rai_interfaces/msg/HRIMessage --field text
    # The agent's recall_object, list_room_objects and go_to_object ask the object memory.
    # go_to_object leaves find_objects to check that the object is still there, so
    # find_objects reads the detector that fills the memory, not the oracle, whose scene
    # has no cup (SG-23). An --objects-topic in ARGS is kept instead.
    if $with_object_tools; then
        agent_args+=(--object-memory)
        [[ " ${agent_args[*]} " == *" --objects-topic"* ]] ||
            agent_args+=(--objects-topic "$DETECTOR_OBJECTS")
    fi
    if $with_agent && ! running agent; then
        start agent "$REPO" "$PIXI" run --frozen --manifest-path "$REPO/pixi.toml" agent-sim-chula \
            --tool-log "$SIM_DIR/logs/agent-tools.jsonl" "${agent_args[@]}"
    fi

    wait_for_log oracle "$ORACLE_READY" "$PART_TIMEOUT_S" ||
        fail "the oracle did not start; see $SIM_DIR/logs/oracle.log"
    if $with_agent; then
        wait_for_log agent "$AGENT_READY" "$PART_TIMEOUT_S" ||
            fail "the agent did not start; see $SIM_DIR/logs/agent.log"
    fi
    if $with_vision; then
        echo "waiting up to ${VISION_TIMEOUT_S} s for walkie-vision to load its models ..."
        wait_for_log vision "$VISION_READY" "$VISION_TIMEOUT_S" "$VISION_FAILED" ||
            fail "walkie-vision did not load its models; see $SIM_DIR/logs/vision.log"
    fi
    if $with_memory; then
        wait_for_log memory "$MEMORY_READY" "$PART_TIMEOUT_S" "$MEMORY_FAILED" ||
            fail "the object memory did not start; see $SIM_DIR/logs/memory.log"
    fi
    if $with_foxglove; then
        wait_for_log foxglove "$FOXGLOVE_READY" "$PART_TIMEOUT_S" ||
            fail "foxglove_bridge did not start; see $SIM_DIR/logs/foxglove.log"
    fi
    sleep "$SETTLE_S"
    local name wanted=(router bringup oracle speech)
    if $with_arms; then
        wanted+=(moveit commander)
    fi
    if $with_vision; then
        wanted+=(vision)
    fi
    if $with_memory; then
        wanted+=(memory)
    fi
    if $with_foxglove; then
        wanted+=(foxglove)
    fi
    if $with_agent; then
        wanted+=(agent)
    fi
    for name in "${wanted[@]}"; do
        running "$name" || fail "$name exited; see $SIM_DIR/logs/$name.log"
    done
    if $with_foxglove; then
        echo "Foxglove: open a connection to ws://localhost:$SIM_FOXGLOVE_PORT"
    fi
    echo "up. Next: scripts/chula-sim.sh check$($with_agent && echo " --agent" || true)$($with_arms && echo " --arms" || true)$($with_whole_body && echo " --whole-body" || true)"
}

check() {
    mkdir -p "$ROS_LOG_DIR"
    "$PIXI" run --frozen --manifest-path "$REPO/pixi.toml" python -m walkie_agent.sim.sim_check "$@"
}

# status: one line for each part. Exits 1 if the state of any is unknown.
status() {
    local name result=0
    for name in "${PARTS[@]}"; do
        case $(state "$name") in
            running) echo "$name: running, process group $(pgid_of "$name"), log $SIM_DIR/logs/$name.log" ;;
            stopped) echo "$name: stopped" ;;
            *)
                echo "$name: unknown, its processes could not be looked up"
                result=1
                ;;
        esac
    done
    return "$result"
}

print_env() {
    echo "export ROS_DOMAIN_ID=$SIM_DOMAIN_ID"
    echo "export RMW_IMPLEMENTATION=rmw_zenoh_cpp"
    echo "export ZENOH_CONFIG_OVERRIDE='$ZENOH_CONFIG_OVERRIDE'"
}

# reset-memory: restore SIM_MEMORY_DB from SIM_MEMORY_SEED with walkie-database's
# db-restore. db-restore keeps the store's ID lineage: it refuses, and changes nothing,
# when the seed would give an object ID that the store has already given to something
# else. A store that does not exist yet starts a lineage of its own. db-restore runs
# only while the memory is stopped, so this refuses while up's memory runs, and
# db-restore refuses a store whose lock another process holds, as a memory node
# started by hand does for its whole life. A reader such as db-shell takes no lock.
reset_memory() {
    local store_dir
    check_database_prerequisites reset-memory
    [[ -f $SIM_MEMORY_SEED ]] ||
        die "SIM_MEMORY_SEED=$SIM_MEMORY_SEED does not exist; docs/setup/08-chula-simulation.md says how to build it"
    SIM_MEMORY_SEED=$(cd "$(dirname "$SIM_MEMORY_SEED")" && pwd)/$(basename "$SIM_MEMORY_SEED")
    # An assignment takes the status of its last substitution, so the folder is found
    # on its own: a folder that cannot be entered must stop here, not become /.
    store_dir=$(mkdir -p "$(dirname "$SIM_MEMORY_DB")" && cd "$(dirname "$SIM_MEMORY_DB")" && pwd) ||
        die "SIM_MEMORY_DB=$SIM_MEMORY_DB is in a folder that cannot be made or entered"
    SIM_MEMORY_DB=$store_dir/$(basename "$SIM_MEMORY_DB")
    lock
    if running memory; then
        die "the object memory is running, and its store can be restored only while it is stopped; scripts/chula-sim.sh down stops it"
    fi
    "$PIXI" run --as-is --manifest-path "$WALKIE_DATABASE/pixi.toml" db-restore \
        --seed "$SIM_MEMORY_SEED" --db "$SIM_MEMORY_DB" --map-folder "$SIM_MEMORY_MAP_FOLDER" \
        --clock-kind ros_sim --replace
}

# down: stop every part, in reverse order. A part that does not stop, or whose processes
# cannot be looked up or signalled, does not keep down from stopping the others; down
# then exits 1.
down() {
    local i result=0
    lock
    for ((i = ${#PARTS[@]} - 1; i >= 0; i--)); do
        stop "${PARTS[i]}" || result=1
    done
    return "$result"
}

# tests/test_chula_sim_script.py sources this file to call its functions.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    case ${1:-} in
        up) shift; up "$@" ;;
        check) shift; check "$@" ;;
        status) status ;;
        env) print_env ;;
        reset-memory) reset_memory ;;
        down) down ;;
        *) sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
    esac
fi
