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

# Under Zenoh, use rmw_zenoh_cpp 0.2.11 (`pixi run build-rmw-zenoh`) instead
# of RoboStack's 0.2.10, whose rmw_wait can block forever (ADR-0032). ROS
# opens the middleware by file name, and the environment's
# librmw_implementation.so has an RPATH of its own directory, which the
# dynamic linker searches before LD_LIBRARY_PATH, so it always finds 0.2.10
# there. A library that is already loaded under the same name is reused
# instead.
#
# This script cannot tell which middleware a process will use: pixi runs it
# before it applies [activation.env] and a task's env, so it sees only the
# calling shell's RMW_IMPLEMENTATION. It therefore preloads a small loader
# (scripts/rmw_zenoh_loader.c) into every process, and the loader opens 0.2.11
# when the process starts, only if RMW_IMPLEMENTATION is rmw_zenoh_cpp then.
# Preloading 0.2.11 itself would put Zenoh under every process, also one that
# asks for Fast DDS, and rcl would exit with "Expected RMW implementation
# identifier of 'rmw_fastrtps_cpp'" (ADR-0033).
#
# This script prints no warning when the build is missing, because pixi run
# and pixi shell do not show its output (Pixi 0.80). The loader prints an
# error itself when it cannot open 0.2.11.
_rmw_zenoh_install="${PIXI_PROJECT_ROOT}/rmw_zenoh_ws/install"
_rmw_zenoh_loader="${_rmw_zenoh_install}/rmw_zenoh_loader/lib/librmw_zenoh_loader.so"
if [[ -f "${_rmw_zenoh_loader}" && -f "${_rmw_zenoh_install}/rmw_zenoh_cpp/lib/librmw_zenoh_cpp.so" &&
      ":${LD_PRELOAD:-}:" != *":${_rmw_zenoh_loader}:"* ]]; then
    export LD_PRELOAD="${_rmw_zenoh_loader}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
unset _rmw_zenoh_install _rmw_zenoh_loader

# Project ROS 2 workspace overlay (available after `pixi run build-ros2`).
if [[ -f "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh"
fi
