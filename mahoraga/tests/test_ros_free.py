"""Importing Mahoraga leaves ROS out, checked in a fresh interpreter."""

import subprocess
import sys


def test_importing_mahoraga_loads_no_ros_module():
    code = (
        "import importlib, pkgutil, sys\n"
        "import mahoraga\n"
        "for module in pkgutil.iter_modules(mahoraga.__path__):\n"
        "    if module.name not in ('tests', '__main__'):\n"
        "        importlib.import_module(f'mahoraga.{module.name}')\n"
        "ros = {'rclpy', 'rai', 'rai_interfaces'} & {name.split('.')[0] for name in sys.modules}\n"
        "print(sorted(ros))\n"
    )

    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
