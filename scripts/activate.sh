#!/usr/bin/env bash
# Sourced by pixi on environment activation.

# The environment's own alsa-lib looks for ALSA plugins only inside the
# environment, so the system's PipeWire plugin, which provides the "default"
# and "pipewire" sound devices, is missing and PortAudio lists only raw hw:
# devices. Point alsa-lib at the system plugins unless the user already has.
# ALSA_PLUGIN_DIR replaces the plugin directory rather than adding one; the
# environment has no ALSA plugins of its own, so nothing is lost.
if [[ -z "${ALSA_PLUGIN_DIR:-}" ]]; then
    for dir in /usr/lib/x86_64-linux-gnu/alsa-lib /usr/lib64/alsa-lib /usr/lib/alsa-lib; do
        if [[ -d "${dir}" ]]; then
            export ALSA_PLUGIN_DIR="${dir}"
            break
        fi
    done
fi

# Under Zenoh, load rmw_zenoh_cpp 0.2.11 (`pixi run build-rmw-zenoh`) instead
# of RoboStack's 0.2.10, whose rmw_wait can block forever (ADR-0032). ROS
# opens the middleware by file name, and the environment's
# librmw_implementation.so has an RPATH of its own directory, which the loader
# searches before LD_LIBRARY_PATH, so it always finds 0.2.10 there.
# A preloaded library with the same name is reused instead. pixi applies a
# task's env before this script, so agent-sim's RMW_IMPLEMENTATION counts.
_rmw_zenoh_lib="${PIXI_PROJECT_ROOT}/rmw_zenoh_ws/install/rmw_zenoh_cpp/lib/librmw_zenoh_cpp.so"
if [[ "${RMW_IMPLEMENTATION:-}" == "rmw_zenoh_cpp" && -f "${_rmw_zenoh_lib}" ]]; then
    export LD_PRELOAD="${_rmw_zenoh_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
unset _rmw_zenoh_lib

# Project ROS 2 workspace overlay (available after `pixi run build-ros2`).
if [[ -f "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh"
fi
