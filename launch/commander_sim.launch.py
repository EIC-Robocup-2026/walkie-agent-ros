# Runs walkie-ros's commander.launch.py on sim time (it sets no use_sim_time).
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import SetParameter

def generate_launch_description():
    path = os.path.join(get_package_share_directory("openarm_bimanual_commander_cpp"),
                        "launch", "commander.launch.py")
    return LaunchDescription([GroupAction([
        SetParameter(name="use_sim_time", value=True),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(path)),
    ])])
