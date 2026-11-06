#!/usr/bin/env bash
# Run walkie-ros's Chula PAT14-6 simulation with the oracle perception, a speech log, and
# the agent when asked, on a private ROS graph: a Zenoh router of its own on SIM_ZENOH_PORT
# and ROS_DOMAIN_ID SIM_DOMAIN_ID. Nothing started here connects to the router on 7447, so
# it neither sees nor disturbs a robot or another simulation on this computer.
#
#   scripts/chula-sim.sh up [--agent] [--gui]   router, MuJoCo with Nav2, oracle, speech log, agent
#   scripts/chula-sim.sh check [--agent]        wait for the clock, TF, head camera and arrays
#   scripts/chula-sim.sh status                 which parts are running, and their logs
#   scripts/chula-sim.sh env                    the exports that put a shell on the private graph
#   scripts/chula-sim.sh down                   stop everything this script started
#
# docs/setup/08-chula-simulation.md explains each step. Environment:
#   WALKIE_ROS          a walkie-ros workspace, with its pixi environment installed and its
#                       install/ built (required)
#   WALKIE_ROS_OVERLAY  a colcon workspace sourced over WALKIE_ROS's install, such as a
#                       worktree of feat/chula-pat14-6-simulation with only robot_bringup and
#                       walkie_mujoco_worlds built (optional)
#   SIM_DIR             logs, process groups, ROS logs and the world cache
#                       (default /var/tmp/walkie-chula-sim)
#   SIM_ZENOH_PORT      the private router's port (default 7499)
#   SIM_DOMAIN_ID       ROS_DOMAIN_ID (default 77)
#   SIM_SPAWN_POSE      "X Y YAW_DEG" in the map frame (default the living room place,
#                       facing the oracle scene's two people)
#   SIM_CPUS            when set, every process runs on these CPUs (taskset -c)
#
# The agent reads its model keys as it always does: from the environment, or from the
# .env at the root of this repository.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PIXI=${PIXI:-$(command -v pixi || echo "$HOME/.pixi/bin/pixi")}
WALKIE_ROS=${WALKIE_ROS:-}
WALKIE_ROS_OVERLAY=${WALKIE_ROS_OVERLAY:-}
SIM_DIR=${SIM_DIR:-/var/tmp/walkie-chula-sim}
SIM_ZENOH_PORT=${SIM_ZENOH_PORT:-7499}
SIM_DOMAIN_ID=${SIM_DOMAIN_ID:-77}
# config/locations.chula_pat14_6.toml's "living room" (yaw 0 rad). The world's own
# spawn faces south, away from both people of config/scenes/chula_living_room.toml.
SIM_SPAWN_POSE=${SIM_SPAWN_POSE:-1.7629 -1.483 0}
SIM_CPUS=${SIM_CPUS:-}

WORLD=chula_pat14_6
PARTS=(router bringup oracle speech agent)
# Nav2's lifecycle manager logs this once every navigation server is active.
BRINGUP_READY="Managed nodes are active"
BRINGUP_TIMEOUT_S=300
ROUTER_TIMEOUT_S=30

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

pgid_of() {
    cat "$SIM_DIR/pids/$1" 2>/dev/null || true
}

running() {
    local pgid
    pgid=$(pgid_of "$1")
    [[ -n $pgid ]] && kill -0 -- "-$pgid" 2>/dev/null
}

# start NAME DIR COMMAND...: run COMMAND in DIR as the leader of a new process group,
# so that stop can signal it with every child it starts. The leader writes its own PID,
# which is the group's ID.
start() {
    local name=$1 dir=$2
    shift 2
    local pidfile=$SIM_DIR/pids/$name
    rm -f "$pidfile"
    (cd "$dir" && exec setsid bash -c 'echo $$ > "$0"; exec "$@"' "$pidfile" ${PIN[@]+"${PIN[@]}"} "$@" \
        > "$SIM_DIR/logs/$name.log" 2>&1 < /dev/null &)
    for _ in $(seq 50); do
        [[ -s $pidfile ]] && break
        sleep 0.1
    done
    [[ -s $pidfile ]] || die "$name did not start; see $SIM_DIR/logs/$name.log"
    echo "started $name, process group $(pgid_of "$name"), log $SIM_DIR/logs/$name.log"
}

# stop NAME: SIGINT to the whole group, as Ctrl+C would send, then SIGTERM, then SIGKILL.
stop() {
    local name=$1 pgid
    pgid=$(pgid_of "$name")
    [[ -n $pgid ]] || return 0
    if kill -0 -- "-$pgid" 2>/dev/null; then
        kill -INT -- "-$pgid" 2>/dev/null || true
        for _ in $(seq 100); do
            kill -0 -- "-$pgid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 -- "-$pgid" 2>/dev/null; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
            sleep 5
        fi
        if kill -0 -- "-$pgid" 2>/dev/null; then
            kill -KILL -- "-$pgid" 2>/dev/null || true
        fi
        echo "stopped $name"
    fi
    rm -f "$SIM_DIR/pids/$name"
}

port_open() {
    (exec 3<> "/dev/tcp/127.0.0.1/$SIM_ZENOH_PORT") 2>/dev/null
}

# wait_for_log NAME TEXT SECONDS: until TEXT is in NAME's log; fails when NAME stops.
wait_for_log() {
    local name=$1 text=$2 seconds=$3
    for _ in $(seq "$seconds"); do
        grep -q -F "$text" "$SIM_DIR/logs/$name.log" 2>/dev/null && return 0
        running "$name" || break
        sleep 1
    done
    tail -n 20 "$SIM_DIR/logs/$name.log" >&2 || true
    return 1
}

check_prerequisites() {
    [[ -n $WALKIE_ROS ]] || die "set WALKIE_ROS to a walkie-ros workspace"
    [[ -f $WALKIE_ROS/pixi.toml && -f $WALKIE_ROS/install/setup.bash ]] ||
        die "WALKIE_ROS=$WALKIE_ROS has no pixi.toml or no built install/"
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

up() {
    local with_agent=false headless=true rviz=false
    while (($#)); do
        case $1 in
            --agent) with_agent=true ;;
            --gui) headless=false rviz=true ;;
            *) die "unknown option for up: $1" ;;
        esac
        shift
    done
    check_prerequisites
    mkdir -p "$SIM_DIR"/{logs,pids,worlds,ros/log}

    if ! running bringup; then
        # A second MuJoCo simulation, on any graph, competes for the CPU with the first.
        local others
        others=$(pgrep -f "[n]av_mujoco_omnibot.launch.py" || true)
        [[ -z $others ]] || die "another simulation is running (PIDs $(echo $others)); stop it first"
    fi

    if ! running router; then
        port_open && die "port $SIM_ZENOH_PORT is in use; set SIM_ZENOH_PORT to a free port"
        start router "$SIM_DIR" env ZENOH_CONFIG_OVERRIDE="listen/endpoints=[\"tcp/localhost:$SIM_ZENOH_PORT\"]" \
            "$PIXI" run --as-is --manifest-path "$WALKIE_ROS/pixi.toml" ros2 run rmw_zenoh_cpp rmw_zenohd
        for _ in $(seq "$ROUTER_TIMEOUT_S"); do
            port_open && break
            sleep 1
        done
        port_open || die "the router is not listening on $SIM_ZENOH_PORT; see $SIM_DIR/logs/router.log"
    fi

    if ! running bringup; then
        # --as-is: run walkie-ros's environment as installed; never install or update it.
        start bringup "$SIM_DIR" "$PIXI" run --as-is --manifest-path "$WALKIE_ROS/pixi.toml" \
            bash -c '[[ -z $0 ]] || source "$0"; exec "$@"' "${WALKIE_ROS_OVERLAY:+$WALKIE_ROS_OVERLAY/install/local_setup.bash}" \
            ros2 launch robot_bringup nav_mujoco_omnibot.launch.py "world:=$WORLD" \
            "spawn_pose:=$SIM_SPAWN_POSE" "headless:=$headless" "use_rviz:=$rviz" \
            "world_output_dir:=$SIM_DIR/worlds"
    fi
    echo "waiting up to ${BRINGUP_TIMEOUT_S} s for Nav2 ..."
    wait_for_log bringup "$BRINGUP_READY" "$BRINGUP_TIMEOUT_S" ||
        die "the simulation did not become ready; see $SIM_DIR/logs/bringup.log"

    running oracle || start oracle "$REPO" "$PIXI" run --frozen --manifest-path "$REPO/pixi.toml" oracle-chula
    # The agent's say needs a listener on /to_human. This one writes each line to
    # speech.log instead of voicing it, so the simulation is silent.
    running speech || start speech "$REPO" env PYTHONUNBUFFERED=1 "$PIXI" run --frozen \
        --manifest-path "$REPO/pixi.toml" ros2 topic echo /to_human rai_interfaces/msg/HRIMessage --field text
    if $with_agent && ! running agent; then
        start agent "$REPO" "$PIXI" run --frozen --manifest-path "$REPO/pixi.toml" agent-sim-chula \
            --tool-log "$SIM_DIR/logs/agent-tools.jsonl"
    fi
    echo "up. Next: scripts/chula-sim.sh check$($with_agent && echo " --agent" || true)"
}

check() {
    mkdir -p "$ROS_LOG_DIR"
    "$PIXI" run --frozen --manifest-path "$REPO/pixi.toml" python -m walkie_agent.sim.sim_check "$@"
}

status() {
    local name
    for name in "${PARTS[@]}"; do
        if running "$name"; then
            echo "$name: running, process group $(pgid_of "$name"), log $SIM_DIR/logs/$name.log"
        else
            echo "$name: stopped"
        fi
    done
}

print_env() {
    echo "export ROS_DOMAIN_ID=$SIM_DOMAIN_ID"
    echo "export RMW_IMPLEMENTATION=rmw_zenoh_cpp"
    echo "export ZENOH_CONFIG_OVERRIDE='$ZENOH_CONFIG_OVERRIDE'"
}

down() {
    local i
    for ((i = ${#PARTS[@]} - 1; i >= 0; i--)); do
        stop "${PARTS[i]}"
    done
}

case ${1:-} in
    up) shift; up "$@" ;;
    check) shift; check "$@" ;;
    status) status ;;
    env) print_env ;;
    down) down ;;
    *) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
