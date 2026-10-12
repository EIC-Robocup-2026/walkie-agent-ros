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

# Project ROS 2 workspace overlay (available after `pixi run build-ros2`).
if [[ -f "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh"
fi
