"""Run walkie-ros's arm commander on the simulation's clock.

openarm_bimanual_commander_cpp's commander.launch.py declares no use_sim_time, so
passing use_sim_time:=true to it does nothing. The commander then keeps wall time
while move_group and the controllers keep the simulation's, and walkie-ros's
manipulation guide notes that it can then fail to get the robot's current state.
This file sets use_sim_time on every node of that launch file instead.

Run it in walkie-ros's environment, where the commander is built:

    ros2 launch /path/to/walkie-agent-ros/launch/commander_sim.launch.py

scripts/chula-sim.sh up --arms starts it that way (docs/setup/08-chula-simulation.md).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import SetParameter


def generate_launch_description() -> LaunchDescription:
    path = os.path.join(
        get_package_share_directory("openarm_bimanual_commander_cpp"), "launch", "commander.launch.py"
    )
    return LaunchDescription([
        GroupAction([
            SetParameter(name="use_sim_time", value=True),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(path)),
        ])
    ])
