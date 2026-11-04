"""The foundation imports no ROS.

This runs in a fresh interpreter, since conftest imports rclpy into this one.
PYTHONPATH puts this tree's src first, because the environment's editable
install of walkie_agent can point at another checkout.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

IMPORT_EVERY_NEW_MODULE = """
import importlib, json, pkgutil, sys
import walkie_agent.atomics, walkie_agent.capabilities, walkie_agent.runtime
names = ["walkie_agent.errors", "walkie_agent.adapter"]
for package in (walkie_agent.capabilities, walkie_agent.atomics, walkie_agent.runtime):
    names.append(package.__name__)
    names += [found.name for found in pkgutil.iter_modules(package.__path__, package.__name__ + ".")]
files = {name: importlib.import_module(name).__file__ for name in names}
loaded = sorted({name.partition(".")[0] for name in sys.modules})
print(json.dumps({"files": files, "loaded": loaded}))
"""

ROS = re.compile(r"rclpy|rai|rai_.*|rosidl_.*|.*_msgs|.*_srvs|.*_interfaces")


def ros_modules_loaded_by(code: str) -> tuple[dict[str, str], list[str]]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    report = json.loads(done.stdout.splitlines()[-1])
    return report["files"], [name for name in report["loaded"] if ROS.fullmatch(name)]


def test_importing_every_new_module_loads_no_ros():
    files, ros = ros_modules_loaded_by(IMPORT_EVERY_NEW_MODULE)
    assert ros == []
    assert {
        "walkie_agent.errors",
        "walkie_agent.adapter",
        "walkie_agent.capabilities",
        "walkie_agent.capabilities.fake",
        "walkie_agent.capabilities.guards",
        "walkie_agent.capabilities.records",
        "walkie_agent.atomics",
        "walkie_agent.atomics.navigation",
        "walkie_agent.atomics.speech",
        "walkie_agent.atomics.perception",
        "walkie_agent.atomics.manipulation",
        "walkie_agent.runtime",
        "walkie_agent.runtime.agent",
        "walkie_agent.runtime.middleware",
        "walkie_agent.runtime.todos",
    } <= set(files)
    assert all(Path(file).is_relative_to(SRC) for file in files.values())


def test_the_check_would_catch_a_ros_import():
    _, ros = ros_modules_loaded_by("import rclpy\n" + IMPORT_EVERY_NEW_MODULE)
    assert "rclpy" in ros
