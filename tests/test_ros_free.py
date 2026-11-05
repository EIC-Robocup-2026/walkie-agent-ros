"""The foundation imports no ROS, and nothing still names a removed module.

The import checks run in a fresh interpreter, since other test modules import rclpy
into this one. PYTHONPATH puts this tree's src first, because the environment's
editable install of walkie_agent can point at another checkout.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

IMPORT_EVERY_NEW_MODULE = """
import importlib, json, pkgutil, sys
import walkie_agent.atomics, walkie_agent.capabilities, walkie_agent.runtime
names = ["walkie_agent.errors", "walkie_agent.adapter", "walkie_agent.tasks", "walkie_agent.locations"]
for package in (walkie_agent.capabilities, walkie_agent.atomics, walkie_agent.runtime):
    names.append(package.__name__)
    names += [found.name for found in pkgutil.iter_modules(package.__path__, package.__name__ + ".")]
files = {name: importlib.import_module(name).__file__ for name in names}
loaded = sorted({name.partition(".")[0] for name in sys.modules})
print(json.dumps({"files": files, "loaded": loaded}))
"""

# Stands in for a real import, so the negative control needs no ROS installation.
FAKE_ROS_IMPORT = 'import sys, types\nsys.modules["rclpy"] = types.ModuleType("rclpy")\n'

ROS = re.compile(r"rclpy|rai|rai_.*|rosidl_.*|.*_msgs|.*_srvs|.*_interfaces")

# The modules of the RAI-hosted agent, in any spelling of an import or a path.
REMOVED = re.compile(
    r"walkie_agent[./](agent|graph|todo|whoami|tools)\b"
    r"|from walkie_agent import [^\n]*\b(agent|graph|todo|whoami|tools)\b"
)
REMOVED_NAMES = re.compile(r"\b(TodoList|WalkieToolRunner|WalkieAgent|build_walkie_graph)\b")
CODE_AND_BUILD = [
    "src",
    "tests",
    "mahoraga",
    "scripts",
    "docker",
    ".github",
    "pixi.toml",
    "pyproject.toml",
]
TEXT_SUFFIXES = {".py", ".toml", ".sh", ".yml", ".yaml", ".md", ".json", ".txt", ".cfg", ".launch"}


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
        "walkie_agent.atomics",
        "walkie_agent.atomics.facts",
        "walkie_agent.atomics.navigation",
        "walkie_agent.atomics.speech",
        "walkie_agent.atomics.perception",
        "walkie_agent.atomics.manipulation",
        "walkie_agent.runtime",
        "walkie_agent.runtime.agent",
        "walkie_agent.runtime.middleware",
        "walkie_agent.runtime.model",
        "walkie_agent.runtime.runner",
        "walkie_agent.runtime.todos",
    } <= set(files)
    assert all(Path(file).is_relative_to(SRC) for file in files.values())


def test_the_check_would_catch_a_ros_import():
    _, ros = ros_modules_loaded_by(FAKE_ROS_IMPORT + IMPORT_EVERY_NEW_MODULE)
    assert "rclpy" in ros


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
        if f != Path(__file__).resolve() and not any(p.endswith(".egg-info") for p in f.parts)
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
    assert _mentions(_files_under("src", "mahoraga", "scripts"), REMOVED_NAMES) == []


def test_tests_cannot_read_a_dotenv_file(tmp_path, monkeypatch):
    dotenv = pytest.importorskip("dotenv")
    secrets = tmp_path / ".env"
    secrets.write_text("WALKIE_TEST_SECRET=leaked\n")
    monkeypatch.delenv("WALKIE_TEST_SECRET", raising=False)
    assert dotenv.load_dotenv(secrets) is False
    assert "WALKIE_TEST_SECRET" not in os.environ


def test_the_sweep_would_catch_a_removed_module(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from walkie_agent.graph import x\nfrom walkie_agent import tools, errors\n")
    assert len(_mentions([bad], REMOVED)) == 2


def test_no_docs_page_names_a_removed_module():
    # Decision records keep their history, so their bodies may name old modules.
    pages = [
        page
        for page in [*_files_under("docs"), ROOT / "README.md"]
        if not page.is_relative_to(ROOT / "docs" / "architecture" / "decisions")
    ]
    assert _mentions(pages, REMOVED) == []
