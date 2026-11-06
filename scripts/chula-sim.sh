#!/usr/bin/env bash
# Run walkie-ros's Chula PAT14-6 simulation with the oracle perception, a speech log, and
# the agent when asked, on a private ROS graph: a Zenoh router of its own on SIM_ZENOH_PORT
# and ROS_DOMAIN_ID SIM_DOMAIN_ID. Nothing started here connects to the router on 7447, so
# it neither sees nor disturbs a robot or another simulation on this computer.
#
#   scripts/chula-sim.sh up [--agent] [--gui]   router, MuJoCo with Nav2, oracle, speech log, agent
#       [-- ARGS]                               ARGS go to the agent, such as --tool-set hri
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
# .env at the root of this repository. The script itself needs python3, 3.9 or newer, on
# PATH, for scripts/chula_sim_owned.py.

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

# lock: one up or down at a time, on this SIM_DIR. The lock is released when the
# script exits.
lock() {
    mkdir -p "$SIM_DIR"
    exec 9> "$SIM_DIR/lock"
    flock -n 9 || die "another scripts/chula-sim.sh up or down is running on $SIM_DIR"
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
    local with_agent=false headless=true rviz=false agent_args=()
    while (($#)); do
        case $1 in
            --agent) with_agent=true ;;
            --gui) headless=false rviz=true ;;
            --) shift; agent_args=("$@"); break ;;
            *) die "unknown option for up: $1" ;;
        esac
        shift
    done
    ((${#agent_args[@]} == 0)) || $with_agent ||
        die "the arguments after -- are the agent's, so up needs --agent with them"
    check_prerequisites
    lock
    mkdir -p "$SIM_DIR"/{logs,pids,worlds,ros/log}

    if ! running bringup; then
        # A second MuJoCo simulation, on any graph, competes for the CPU with the first.
        local others
        others=$(pgrep -f "[n]av_mujoco_omnibot.launch.py" || true)
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
        fail "the simulation did not become ready; see $SIM_DIR/logs/bringup.log"

    running oracle || start oracle "$REPO" "$PIXI" run --frozen --manifest-path "$REPO/pixi.toml" oracle-chula
    # The agent's say needs a listener on /to_human. This one writes each line to
    # speech.log instead of voicing it, so the simulation is silent.
    running speech || start speech "$REPO" env PYTHONUNBUFFERED=1 "$PIXI" run --frozen \
        --manifest-path "$REPO/pixi.toml" ros2 topic echo /to_human rai_interfaces/msg/HRIMessage --field text
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
    sleep "$SETTLE_S"
    local name wanted=(router bringup oracle speech)
    if $with_agent; then
        wanted+=(agent)
    fi
    for name in "${wanted[@]}"; do
        running "$name" || fail "$name exited; see $SIM_DIR/logs/$name.log"
    done
    echo "up. Next: scripts/chula-sim.sh check$($with_agent && echo " --agent" || true)"
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
        down) down ;;
        *) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
    esac
fi
