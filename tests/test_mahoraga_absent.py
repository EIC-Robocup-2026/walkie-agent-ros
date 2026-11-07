"""The gym's research dependencies stay in Mahoraga's own pixi environment, and the agent
never imports the gym.

pydantic-monty is in the default environment too, because the agent runs scripts
through walkie_agent.scripting.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# A fresh interpreter, since other test modules may import the gym into this one.
IMPORT_MAIN = """
import json, sys
import walkie_agent.main
print(json.dumps(sorted(name for name in sys.modules if name.startswith("walkie_agent"))))
"""


@pytest.mark.parametrize("module", ["inspect_ai", "robocupathome_generator"])
def test_a_mahoraga_dependency_is_not_in_the_default_environment(module):
    assert importlib.util.find_spec(module) is None


def test_importing_the_agent_never_imports_the_gym():
    # PYTHONPATH puts this tree's src first, because the editable install can point at another checkout.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run(
        [sys.executable, "-c", IMPORT_MAIN], capture_output=True, text=True, env=env, check=True
    )
    loaded = json.loads(done.stdout.splitlines()[-1])

    assert "walkie_agent.main" in loaded
    assert [name for name in loaded if name.startswith("walkie_agent.mahoraga")] == []


def test_the_script_runtime_and_the_tools_never_import_the_gym():
    import walkie_agent.scripting

    scripting = sorted(
        f"walkie_agent.scripting.{path.stem}"
        for path in Path(walkie_agent.scripting.__file__).parent.glob("*.py")
        if path.stem != "__init__"
    )
    code = "\n".join(
        [
            "import importlib, json, sys",
            f"for name in {[*scripting, 'walkie_agent.tools.atomic', 'walkie_agent.tools.bundled_preset']!r}:",
            "    importlib.import_module(name)",
            'print(json.dumps(sorted(name for name in sys.modules if name.startswith("walkie_agent"))))',
        ]
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    loaded = json.loads(done.stdout.splitlines()[-1])

    assert "walkie_agent.scripting.agent_tool" in loaded
    assert [name for name in loaded if name.startswith("walkie_agent.mahoraga")] == []
