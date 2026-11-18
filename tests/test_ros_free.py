"""The foundation, run_script's runtime and the Mahoraga gym import no ROS, and nothing
still names a removed module.

The import checks run in a fresh interpreter, since other test modules import rclpy
into this one. PYTHONPATH puts this tree's src first, because the environment's
editable install of walkie_agent can point at another checkout.
"""

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

IMPORT_EVERY_NEW_MODULE = """
import importlib, json, pkgutil, sys
import walkie_agent.tools.atomic, walkie_agent.capabilities, walkie_agent.perception, walkie_agent.runtime
import walkie_agent.arm, walkie_agent.scripting
names = ["walkie_agent.errors", "walkie_agent.adapter", "walkie_agent.tasks", "walkie_agent.locations"]
names += ["walkie_agent.tools", "walkie_agent.tools.bundled_preset"]
# The simulation's scene, its file and its projection; the sim package's ROS nodes are left out.
names += ["walkie_agent.sim." + name for name in ("scene", "scene_file", "projection", "fake_vectors")]
for package in (
    walkie_agent.capabilities,
    walkie_agent.tools.atomic,
    walkie_agent.perception,
    walkie_agent.runtime,
    walkie_agent.arm,
    walkie_agent.scripting,
):
    names.append(package.__name__)
    names += [found.name for found in pkgutil.iter_modules(package.__path__, package.__name__ + ".")]
files = {name: importlib.import_module(name).__file__ for name in names}
loaded = sorted({name.partition(".")[0] for name in sys.modules})
print(json.dumps({"files": files, "loaded": loaded}))
"""

# Every module of the gym but its command line, which only `python -m` runs.
IMPORT_THE_GYM = """
import importlib, json, pkgutil, sys
import walkie_agent.mahoraga as gym
names = [gym.__name__]
names += [
    found.name
    for found in pkgutil.iter_modules(gym.__path__, gym.__name__ + ".")
    if found.name != gym.__name__ + ".__main__"
]
files = {name: importlib.import_module(name).__file__ for name in names}
loaded = sorted({name.partition(".")[0] for name in sys.modules})
print(json.dumps({"files": files, "loaded": loaded}))
"""

# Only the mahoraga pixi environment has them (tests/test_mahoraga_absent.py).
GYM_DEPENDENCIES = ("inspect_ai", "robocupathome_generator")

# Stands in for a real import, so the negative control needs no ROS installation.
FAKE_ROS_IMPORT = 'import sys, types\nsys.modules["rclpy"] = types.ModuleType("rclpy")\n'

ROS = re.compile(r"rclpy|rai|rai_.*|rosidl_.*|.*_msgs|.*_srvs|.*_interfaces")

# The modules of the RAI-hosted agent, in any spelling of an import or a path. The
# walkie_agent.tools package that holds the atomics now is not one of them. Also the
# names that left walkie_agent.ros.vision, on one line; the syntax-tree check below
# catches them in every other shape.
REMOVED = re.compile(
    r"walkie_agent[./](agent|graph|todo|whoami)\b"
    r"|walkie_agent[./]tools[./](memory|navigation|speech|todos|waiters)\b"
    r"|from walkie_agent import [^\n]*\b(agent|graph|todo|whoami)\b"
    r"|from walkie_agent\.tools import [^\n]*\b(memory|navigation|speech|todos|waiters)\b"
    r"|walkie_agent[./]ros[./]vision[.:](UnavailableVision|_begin|_no_objects)\b"
    r"|from walkie_agent\.ros\.vision import [^\n]*\b(UnavailableVision|_begin|_no_objects)\b"
)
# UnavailableVision moved to walkie_agent.capabilities, and the lookup guard _begin,
# which _no_objects wrapped, to walkie_agent.capabilities.guards as begin_lookup.
ROS_VISION = "walkie_agent.ros.vision"
LEFT_ROS_VISION = frozenset({"UnavailableVision", "_begin", "_no_objects"})
REMOVED_NAMES = re.compile(r"\b(TodoList|WalkieToolRunner|WalkieAgent|build_walkie_graph)\b")
CODE_AND_BUILD = [
    "src",
    "tests",
    "scripts",
    "docker",
    ".github",
    "pixi.toml",
    "pyproject.toml",
]
TEXT_SUFFIXES = {".py", ".toml", ".sh", ".yml", ".yaml", ".md", ".json", ".txt", ".cfg", ".launch"}
# The shelf's default store writes model-written functions here, which git ignores.
GENERATED = [SRC / "walkie_agent" / "tools" / "temporary", SRC / "walkie_agent" / "tools" / "bundled"]


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
        "walkie_agent.tasks",
        "walkie_agent.locations",
        "walkie_agent.capabilities",
        "walkie_agent.capabilities.facts",
        "walkie_agent.capabilities.fake",
        "walkie_agent.capabilities.guards",
        "walkie_agent.capabilities.records",
        "walkie_agent.tools",
        "walkie_agent.tools.atomic",
        "walkie_agent.tools.atomic.facts",
        "walkie_agent.tools.atomic.navigation",
        "walkie_agent.tools.atomic.speech",
        "walkie_agent.tools.atomic.perception",
        "walkie_agent.tools.atomic.manipulation",
        "walkie_agent.tools.bundled_preset",
        "walkie_agent.scripting",
        "walkie_agent.scripting.driver",
        "walkie_agent.scripting.evidence",
        "walkie_agent.scripting.sandbox",
        "walkie_agent.scripting.shelf",
        "walkie_agent.runtime",
        "walkie_agent.runtime.agent",
        "walkie_agent.runtime.middleware",
        "walkie_agent.runtime.model",
        "walkie_agent.runtime.runner",
        "walkie_agent.runtime.todos",
        "walkie_agent.perception",
        "walkie_agent.perception.gestures",
        "walkie_agent.sim.scene",
        "walkie_agent.sim.scene_file",
        "walkie_agent.sim.projection",
        "walkie_agent.sim.fake_vectors",
        "walkie_agent.arm",
        "walkie_agent.arm.poses",
        "walkie_agent.arm.state",
    } <= set(files)
    assert all(Path(file).is_relative_to(SRC) for file in files.values())


def test_the_check_would_catch_a_ros_import():
    _, ros = ros_modules_loaded_by(FAKE_ROS_IMPORT + IMPORT_EVERY_NEW_MODULE)
    assert "rclpy" in ros


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in GYM_DEPENDENCIES),
    reason="The gym's dependencies are only in the mahoraga pixi environment.",
)
def test_importing_the_gym_loads_no_ros():
    files, ros = ros_modules_loaded_by(IMPORT_THE_GYM)
    assert ros == []
    assert {
        "walkie_agent.mahoraga." + name
        for name in ("arms", "checkers", "commands", "gate", "gym", "oracle", "report", "tasks", "worlds")
    } <= set(files)
    assert all(Path(file).is_relative_to(SRC) for file in files.values())


def _files_under(*relative: str) -> list[Path]:
    files: list[Path] = []
    for name in relative:
        path = ROOT / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files += [f for f in path.rglob("*") if f.is_file() and f.suffix in TEXT_SUFFIXES]
    # An editable install writes ignored packaging metadata that can list deleted files.
    return [
        f
        for f in files
        if f != Path(__file__).resolve()
        and not any(p.endswith(".egg-info") for p in f.parts)
        and not (f.suffix == ".py" and any(f.is_relative_to(folder) for folder in GENERATED))
    ]


def _mentions(files: list[Path], pattern: re.Pattern[str]) -> list[str]:
    return [
        f"{file.name}:{number}: {line.strip()}"
        for file in files
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line)
    ]


def test_no_code_or_build_file_names_a_removed_module():
    assert _mentions(_files_under(*CODE_AND_BUILD), REMOVED) == []


def test_no_code_uses_a_removed_class():
    assert _mentions(_files_under("src", "scripts"), REMOVED_NAMES) == []


def _imported_module(node: ast.ImportFrom, file: Path) -> str | None:
    """The absolute module an import names, resolving a relative one by the file's package."""
    if node.level == 0:
        return node.module
    if not file.is_relative_to(SRC):
        return None
    package = file.relative_to(SRC).with_suffix("").parts[: -node.level]
    return ".".join([*package, node.module] if node.module else package)


def _reads_a_name_that_left_ros_vision(file: Path) -> list[str]:
    """Where `file` imports a name in LEFT_ROS_VISION from walkie_agent.ros.vision or reads it
    as an attribute of that module, under any name the file gives the module.

    It reads the syntax tree rather than lines, so an import in parentheses over several
    lines counts, and a docstring that names the old home does not.
    """
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    module_names = {ROS_VISION}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names |= {a.asname for a in node.names if a.name == ROS_VISION and a.asname}
        elif isinstance(node, ast.ImportFrom) and _imported_module(node, file) == "walkie_agent.ros":
            module_names |= {a.asname or a.name for a in node.names if a.name == "vision"}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _imported_module(node, file) == ROS_VISION:
            found += [
                f"{file.name}:{node.lineno}: imports {a.name}"
                for a in node.names
                if a.name in LEFT_ROS_VISION
            ]
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in LEFT_ROS_VISION
            and ast.unparse(node.value) in module_names
        ):
            found.append(f"{file.name}:{node.lineno}: reads {ast.unparse(node)}")
    return sorted(found)


def _module_level_names(tree: ast.Module) -> Iterator[tuple[str, int]]:
    """Each name a module binds outside its functions and classes, with its line."""
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name, node.lineno
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                yield alias.asname or alias.name.partition(".")[0], node.lineno
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id, node.lineno
        stack.extend(ast.iter_child_nodes(node))


def _binds_a_name_that_left_ros_vision(file: Path) -> list[str]:
    """Where `file` binds a name in LEFT_ROS_VISION at module level, which in
    walkie_agent/ros/vision.py would put it back at its old home: a re-export or a copy.
    """
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    return sorted(
        f"{file.name}:{line}: binds {name}"
        for name, line in _module_level_names(tree)
        if name in LEFT_ROS_VISION
    )


def test_no_code_takes_a_name_from_its_old_home_in_ros_vision():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert [found for file in code for found in _reads_a_name_that_left_ros_vision(file)] == []
    assert _binds_a_name_that_left_ros_vision(SRC / "walkie_agent" / "ros" / "vision.py") == []


def test_the_tree_check_would_catch_a_name_back_in_ros_vision(tmp_path):
    caller = tmp_path / "caller.py"
    caller.write_text(
        "import walkie_agent.ros.vision\n"
        "import walkie_agent.ros.vision as live\n"
        "from walkie_agent.ros import vision as vision_module\n"
        "from walkie_agent.ros.vision import (\n"
        "    RosVision,\n"
        "    UnavailableVision,\n"
        ")\n"
        "a = vision_module.UnavailableVision\n"
        "b = live._begin\n"
        "c = walkie_agent.ros.vision._no_objects\n"
    )
    assert len(_reads_a_name_that_left_ros_vision(caller)) == 4
    shim = tmp_path / "vision.py"
    shim.write_text(
        "from walkie_agent.capabilities import (\n"
        "    ObjectSeen,\n"
        "    UnavailableVision,\n"
        ")\n"
        "def _begin(cancel, query, timeout_s, what): ...\n"
        "_no_objects = None\n"
    )
    assert len(_binds_a_name_that_left_ros_vision(shim)) == 3
    # RosVision stays in walkie_agent.ros.vision, a fake's own _begin is not the guard,
    # and an import inside a function is not a re-export.
    good = tmp_path / "good.py"
    good.write_text(
        '"""Without --vision, main uses capabilities.UnavailableVision."""\n'
        "from walkie_agent.ros import vision as vision_module\n"
        "from walkie_agent.ros.vision import RosVision\n"
        "from walkie_agent.capabilities import UnavailableVision as Stand_in\n"
        "from walkie_agent.capabilities.guards import begin_lookup\n"
        "a = vision_module.RosVision\n"
        "class Fake:\n"
        "    def _begin(self): ...\n"
        "    def objects(self):\n"
        "        self._begin()\n"
        "def stand_in(cancel):\n"
        "    from walkie_agent.capabilities import UnavailableVision\n"
        "    return UnavailableVision(cancel)\n"
    )
    assert _reads_a_name_that_left_ros_vision(good) == []
    assert _binds_a_name_that_left_ros_vision(good) == []


def test_tests_cannot_read_a_dotenv_file(tmp_path, monkeypatch):
    dotenv = pytest.importorskip("dotenv")
    secrets = tmp_path / ".env"
    secrets.write_text("WALKIE_TEST_SECRET=leaked\n")
    monkeypatch.delenv("WALKIE_TEST_SECRET", raising=False)
    assert dotenv.load_dotenv(secrets) is False
    assert "WALKIE_TEST_SECRET" not in os.environ


def test_the_sweep_would_catch_a_removed_module(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from walkie_agent.graph import x\n"
        "from walkie_agent.tools.todos import y\n"
        "from walkie_agent.tools import waiters, memory\n"
        "from walkie_agent.ros.vision import RosVision, UnavailableVision\n"
        'monkeypatch.setattr("walkie_agent.ros.vision._begin", fake)\n'
    )
    assert len(_mentions([bad], REMOVED)) == 5
    # The tools package that holds the atomics is not the removed one, and
    # walkie_agent.ros.vision still holds RosVision.
    good = tmp_path / "good.py"
    good.write_text(
        "from walkie_agent import tools, errors\n"
        "from walkie_agent.tools import atomic\n"
        "import walkie_agent.tools.atomic.memory\n"
        "from walkie_agent.ros.vision import RosVision\n"
        "from walkie_agent.capabilities import UnavailableVision\n"
    )
    assert _mentions([good], REMOVED) == []


def test_the_sweep_skips_the_functions_a_shelf_saved_in_the_package(tmp_path, monkeypatch):
    saved = tmp_path / "src" / "walkie_agent" / "tools" / "temporary"
    saved.mkdir(parents=True)
    (saved / "plan.py").write_text("def plan() -> None:\n    TodoList()\n")
    (saved / "README.md").write_text("Saved functions.\n")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "GENERATED", [saved])

    assert _files_under("src") == [saved / "README.md"]


def test_no_docs_page_names_a_removed_module():
    # Decision records keep their history, so their bodies may name old modules.
    pages = [
        page
        for page in [*_files_under("docs"), ROOT / "README.md"]
        if not page.is_relative_to(ROOT / "docs" / "architecture" / "decisions")
    ]
    assert _mentions(pages, REMOVED) == []
