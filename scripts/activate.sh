#!/usr/bin/env bash
# Sourced by pixi on environment activation.

# Project ROS 2 workspace overlay (available after `pixi run build-ros2`).
if [[ -f "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh"
fi
