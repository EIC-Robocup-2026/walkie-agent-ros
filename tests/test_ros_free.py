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
names = ["walkie_agent.errors", "walkie_agent.tasks", "walkie_agent.locations"]
names += ["walkie_agent.config", "walkie_agent.chat_models", "walkie_agent.prompts"]
names += ["walkie_agent.tools", "walkie_agent.tools.adapter", "walkie_agent.tools.sets"]
names += ["walkie_agent.tools.bundled_preset"]
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
# names that left walkie_agent.ros.vision and the system prompts that left main.py and
# the runtime, on one line; the syntax-tree check below catches them in every other shape.
REMOVED = re.compile(
    r"walkie_agent[./](agent|graph|todo|whoami)\b"
    r"|walkie_agent[./]tools[./](memory|navigation|speech|todos|waiters)\b"
    r"|from walkie_agent import [^\n]*\b(agent|graph|todo|whoami)\b"
    r"|from walkie_agent\.tools import [^\n]*\b(memory|navigation|speech|todos|waiters)\b"
    r"|walkie_agent[./]ros[./]vision[.:](UnavailableVision|_begin|_no_objects)\b"
    r"|from walkie_agent\.ros\.vision import [^\n]*\b(UnavailableVision|_begin|_no_objects)\b"
    r"|walkie_agent[./]runtime[./]model\b"
    r"|from walkie_agent\.runtime import [^\n]*\bmodel\b"
    r"|walkie_agent[./]adapter\b"
    r"|from walkie_agent import [^\n]*\badapter\b"
    r"|\bmain[.:](build_system_prompt|_places_section|_RULES_START|_TOOL_RULES|_LIVE_PEOPLE_RULE"
    r"|_FIND_OBJECTS_ONLY_RULE|_LIVE_OBJECTS_RULE|_LIVE_OBJECT_INFO_LINE|_LIVE_MEMORY_RULE"
    r"|_SCRIPTS_RULE|_RULES_END)\b"
    r"|from walkie_agent\.main import [^\n]*\b(build_system_prompt|_places_section|_RULES_START"
    r"|_TOOL_RULES|_LIVE_PEOPLE_RULE|_FIND_OBJECTS_ONLY_RULE|_LIVE_OBJECTS_RULE"
    r"|_LIVE_OBJECT_INFO_LINE|_LIVE_MEMORY_RULE|_SCRIPTS_RULE|_RULES_END)\b"
    r"|\bruntime([./]agent)?[.:]SYSTEM_PROMPT\b"
    r"|from walkie_agent\.runtime(\.agent)? import [^\n]*\bSYSTEM_PROMPT\b"
)
# UnavailableVision moved to walkie_agent.capabilities, and the lookup guard _begin,
# which _no_objects wrapped, to walkie_agent.capabilities.guards as begin_lookup.
# The runtime's model module split into walkie_agent.config and walkie_agent.chat_models.
# The adapter moved to walkie_agent.tools.adapter. The system prompts, their rules and
# build_system_prompt left main.py, and SYSTEM_PROMPT walkie_agent.runtime, for
# walkie_agent.prompts; `main.` also catches a test's `import walkie_agent.main as main`.
ROS_VISION = "walkie_agent.ros.vision"
LEFT_ROS_VISION = frozenset({"UnavailableVision", "_begin", "_no_objects"})
RUNTIME = SRC / "walkie_agent" / "runtime"
LEFT_RUNTIME = frozenset(
    {
        "agent_settings",
        "build_callbacks",
        "build_fallback",
        "build_model",
        "load_config",
        "_chat_model",
        "_check_base_url",
        "DEFAULT_RUN_LIMIT",
        "DEFAULT_DEADLINE_S",
    }
)
# The tool sets left the atomics' catalogue and main.py for walkie_agent.tools.sets,
# and main.py's _atomic_names became bound_tool_names there.
ATOMIC_INIT = SRC / "walkie_agent" / "tools" / "atomic" / "__init__.py"
LEFT_ATOMIC_INIT = frozenset(
    {
        "DEFAULT",
        "GPSR",
        "HRI",
        "ARMS",
        "ARM_TOOLS",
        "OBJECT_MEMORY_TOOLS",
        "_ARM_ATOMICS",
        "_OBJECT_MEMORY_ATOMICS",
    }
)
MAIN = SRC / "walkie_agent" / "main.py"
LEFT_MAIN = frozenset({"TOOL_SETS", "bound_tools", "bound_tool_names", "_atomic_names"})
# The system prompts left main.py and runtime/agent.py for walkie_agent.prompts.
LEFT_FOR_PROMPTS = frozenset(
    {
        "SYSTEM_PROMPT",
        "build_system_prompt",
        "_places_section",
        "_RULES_START",
        "_TOOL_RULES",
        "_LIVE_PEOPLE_RULE",
        "_FIND_OBJECTS_ONLY_RULE",
        "_LIVE_OBJECTS_RULE",
        "_LIVE_OBJECT_INFO_LINE",
        "_LIVE_MEMORY_RULE",
        "_SCRIPTS_RULE",
        "_RULES_END",
    }
)
# main.py still imports build_system_prompt and runtime/agent.py SYSTEM_PROMPT, so an
# import from these modules would still work; the syntax-tree check rejects it.
PROMPTS_OLD_HOMES = ("walkie_agent.main", "walkie_agent.runtime", "walkie_agent.runtime.agent")
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
        "walkie_agent.tasks",
        "walkie_agent.locations",
        "walkie_agent.config",
        "walkie_agent.chat_models",
        "walkie_agent.prompts",
        "walkie_agent.capabilities",
        "walkie_agent.capabilities.facts",
        "walkie_agent.capabilities.fake",
        "walkie_agent.capabilities.guards",
        "walkie_agent.capabilities.records",
        "walkie_agent.tools",
        "walkie_agent.tools.adapter",
        "walkie_agent.tools.sets",
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


def _reads_a_name_that_left(file: Path, module: str, names: frozenset[str]) -> list[str]:
    """Where `file` imports one of `names` from `module`, reads it as an attribute of that
    module, or names it as a string next to the module in a call such as
    `monkeypatch.setattr(module, "name", ...)`, under any name the file gives the module.

    It reads the syntax tree rather than lines, so an import in parentheses over several
    lines counts, and a docstring that names the old home does not.
    """
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    parent, _, child = module.rpartition(".")
    module_names = {module}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names |= {a.asname for a in node.names if a.name == module and a.asname}
        elif isinstance(node, ast.ImportFrom) and _imported_module(node, file) == parent:
            module_names |= {a.asname or a.name for a in node.names if a.name == child}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _imported_module(node, file) == module:
            found += [
                f"{file.name}:{node.lineno}: imports {a.name}" for a in node.names if a.name in names
            ]
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in names
            and ast.unparse(node.value) in module_names
        ):
            found.append(f"{file.name}:{node.lineno}: reads {ast.unparse(node)}")
        elif (
            isinstance(node, ast.Call)
            and len(node.args) >= 2
            and ast.unparse(node.args[0]) in module_names
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in names
        ):
            found.append(f"{file.name}:{node.lineno}: names {node.args[1].value}")
    return sorted(found)


def _module_level_names(tree: ast.Module, *, imports: bool = True) -> Iterator[tuple[str, int]]:
    """Each name a module binds outside its functions and classes, with its line.

    Without `imports`, only the names it defines: an imported name is left out, unless
    the import renames it (`import x as y`, `from m import x as y`), which binds the new name.
    """
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
                # Without `imports`, an import under a new name still binds that name.
                if imports or (alias.asname and alias.asname != alias.name):
                    yield alias.asname or alias.name.partition(".")[0], node.lineno
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id, node.lineno
        stack.extend(ast.iter_child_nodes(node))


def _binds_one_of(file: Path, names: frozenset[str], *, imports: bool = True) -> list[str]:
    """Where `file` binds one of `names` at module level, which in a module the names
    left would put them back at their old home: a re-export or a copy.

    Without `imports`, an import of one of them under its own name does not count, for a
    module that may use the names but must not define them again; an import that renames
    something into one of them still counts.
    """
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    return sorted(
        f"{file.name}:{line}: binds {name}"
        for name, line in _module_level_names(tree, imports=imports)
        if name in names
    )


def test_no_code_takes_a_name_from_its_old_home_in_ros_vision():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    found = [hit for file in code for hit in _reads_a_name_that_left(file, ROS_VISION, LEFT_ROS_VISION)]
    assert found == []
    assert _binds_one_of(SRC / "walkie_agent" / "ros" / "vision.py", LEFT_ROS_VISION) == []


def test_the_model_builders_did_not_come_back_to_the_runtime():
    # They live in walkie_agent.config and walkie_agent.chat_models now.
    assert not (RUNTIME / "model.py").exists()
    assert _binds_one_of(RUNTIME / "__init__.py", LEFT_RUNTIME) == []


def test_the_tree_check_would_catch_a_model_builder_back_in_the_runtime(tmp_path):
    shim = tmp_path / "__init__.py"
    shim.write_text(
        "from walkie_agent.chat_models import (\n"
        "    build_callbacks,\n"
        "    build_fallback,\n"
        "    build_model,\n"
        ")\n"
        "from walkie_agent.config import agent_settings as agent_settings\n"
        "def load_config(path): ...\n"
    )
    assert len(_binds_one_of(shim, LEFT_RUNTIME)) == 5
    good = tmp_path / "good.py"
    good.write_text(
        '"""main.py builds the model with walkie_agent.chat_models.build_model."""\n'
        "from walkie_agent.runtime.runner import Runner\n"
        "def build(config):\n"
        "    from walkie_agent.chat_models import build_model\n"
        "    return build_model(config)\n"
    )
    assert _binds_one_of(good, LEFT_RUNTIME) == []


def test_the_tool_sets_did_not_come_back_to_the_atomics_or_main():
    # They live in walkie_agent.tools.sets now, beside the adapter in walkie_agent.tools.adapter.
    assert not (SRC / "walkie_agent" / "adapter.py").exists()
    assert _binds_one_of(ATOMIC_INIT, LEFT_ATOMIC_INIT) == []
    # main.py imports TOOL_SETS, bound_tools and bound_tool_names to use them, but defines none.
    assert _binds_one_of(MAIN, LEFT_MAIN, imports=False) == []


def test_the_tree_check_would_catch_a_tool_set_back_in_the_atomics_or_main(tmp_path):
    shim = tmp_path / "__init__.py"
    shim.write_text(
        "from walkie_agent.tools.atomic.speech import say\n"
        "from walkie_agent.tools.sets import (\n"
        "    DEFAULT,\n"
        "    GPSR,\n"
        ")\n"
        "from walkie_agent.tools.sets import ARMS as ARMS\n"
        "HRI = [say]\n"
        "ARM_TOOLS: frozenset[str] = frozenset()\n"
        "OBJECT_MEMORY_TOOLS = frozenset()\n"
    )
    assert len(_binds_one_of(shim, LEFT_ATOMIC_INIT)) == 6
    copy = tmp_path / "main.py"
    copy.write_text(
        "from walkie_agent.tools.sets import TOOL_SETS, bound_tool_names, bound_tools\n"
        "from walkie_agent.tools.sets import bound_tool_names as _atomic_names\n"
        'TOOL_SETS = {"gpsr": []}\n'
        "def bound_tools(tool_set): ...\n"
        "def _atomic_names(tool_set): ...\n"
    )
    assert len(_binds_one_of(copy, LEFT_MAIN, imports=False)) == 4
    # The catalogue keeps every atomic and its docstring may name the sets; main.py may
    # import the sets and bind its own names from them inside a function.
    good_init = tmp_path / "atomic.py"
    good_init.write_text(
        '"""The atomics; DEFAULT, GPSR, HRI and ARMS are in walkie_agent.tools.sets."""\n'
        "from walkie_agent.tools.atomic.speech import say\n"
        '__all__ = ["say"]\n'
    )
    assert _binds_one_of(good_init, LEFT_ATOMIC_INIT) == []
    good_main = tmp_path / "good_main.py"
    good_main.write_text(
        "from walkie_agent.tools.sets import TOOL_SETS, bound_tool_names, bound_tools\n"
        "from walkie_agent.tools.sets import bound_tools as bound_tools\n"
        "ARMS_TOOL_SET = 'arms'\n"
        "def check(tool_set):\n"
        "    TOOL_SETS = sorted(bound_tool_names(tool_set))\n"
        "    return TOOL_SETS, bound_tools(tool_set)\n"
    )
    assert _binds_one_of(good_main, LEFT_MAIN, imports=False) == []


def test_the_system_prompts_did_not_come_back_to_main_or_the_runtime():
    # They live in walkie_agent.prompts now. main.py imports build_system_prompt and
    # runtime/agent.py imports SYSTEM_PROMPT to use them, but neither defines one, and
    # the runtime package no longer re-exports SYSTEM_PROMPT.
    assert _binds_one_of(MAIN, LEFT_FOR_PROMPTS, imports=False) == []
    assert _binds_one_of(RUNTIME / "agent.py", LEFT_FOR_PROMPTS, imports=False) == []
    assert _binds_one_of(RUNTIME / "__init__.py", LEFT_FOR_PROMPTS) == []
    # No caller takes one from its old home, where main.py's and runtime/agent.py's own
    # imports would still hand it over.
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    found = [
        hit
        for file in code
        for module in PROMPTS_OLD_HOMES
        for hit in _reads_a_name_that_left(file, module, LEFT_FOR_PROMPTS)
    ]
    assert found == []


def test_the_tree_check_would_catch_a_system_prompt_back_in_main_or_the_runtime(tmp_path):
    copy = tmp_path / "main.py"
    copy.write_text(
        "from walkie_agent.prompts import build_system_prompt\n"
        "from walkie_agent.prompts import _TOOL_RULES as _SCRIPTS_RULE\n"
        "_RULES_START = 'You are Walkie.'\n"
        "def build_system_prompt(tool_set, locations, robot_description): ...\n"
        "def _places_section(locations): ...\n"
    )
    assert len(_binds_one_of(copy, LEFT_FOR_PROMPTS, imports=False)) == 4
    agent = tmp_path / "agent.py"
    agent.write_text("SYSTEM_PROMPT = 'You are Walkie, a home service robot.'\n")
    assert len(_binds_one_of(agent, LEFT_FOR_PROMPTS, imports=False)) == 1
    shim = tmp_path / "__init__.py"
    shim.write_text(
        "from walkie_agent.prompts import SYSTEM_PROMPT\n"
        "from walkie_agent.runtime.agent import build_agent, build_middleware\n"
    )
    assert len(_binds_one_of(shim, LEFT_FOR_PROMPTS)) == 1
    # build_agent may take its default from walkie_agent.prompts, and main may import
    # build_system_prompt and keep the prompt it builds in a local name.
    good = tmp_path / "good.py"
    good.write_text(
        '"""build_agent; its default prompt, SYSTEM_PROMPT, is in walkie_agent.prompts."""\n'
        "from walkie_agent.prompts import SYSTEM_PROMPT, build_system_prompt\n"
        "def build_agent(model, *, system_prompt=SYSTEM_PROMPT): ...\n"
        "def assemble(tool_set, locations, robot_description):\n"
        "    _RULES_START = build_system_prompt(tool_set, locations, robot_description)\n"
        "    return build_agent(None, system_prompt=_RULES_START)\n"
    )
    assert _binds_one_of(good, LEFT_FOR_PROMPTS, imports=False) == []

    def reads(file: Path) -> list[str]:
        return [
            hit
            for module in PROMPTS_OLD_HOMES
            for hit in _reads_a_name_that_left(file, module, LEFT_FOR_PROMPTS)
        ]

    caller = tmp_path / "caller.py"
    caller.write_text(
        "import walkie_agent.main as main\n"
        "import walkie_agent.runtime.agent as agent_module\n"
        "from walkie_agent import runtime\n"
        "from walkie_agent.main import (\n"
        "    assemble,\n"
        "    build_system_prompt,\n"
        ")\n"
        "from walkie_agent.runtime.agent import (\n"
        "    SYSTEM_PROMPT,\n"
        "    build_agent,\n"
        ")\n"
        "a = agent_module.SYSTEM_PROMPT\n"
        "b = runtime.SYSTEM_PROMPT\n"
        'monkeypatch.setattr(main, "_SCRIPTS_RULE", rule)\n'
        'mock.patch.object(agent_module, "SYSTEM_PROMPT", "x")\n'
    )
    assert len(reads(caller)) == 6
    # The new home, main's own names, and a string beside another module are fine.
    good_caller = tmp_path / "good_caller.py"
    good_caller.write_text(
        '"""main.build_system_prompt moved to walkie_agent.prompts."""\n'
        "import walkie_agent.main as main\n"
        "import walkie_agent.prompts as prompts\n"
        "from walkie_agent.main import (\n"
        "    assemble,\n"
        "    check_templates,\n"
        ")\n"
        "from walkie_agent.prompts import (\n"
        "    SYSTEM_PROMPT,\n"
        "    build_system_prompt,\n"
        ")\n"
        "from walkie_agent.runtime.agent import build_agent\n"
        "a = prompts.SYSTEM_PROMPT\n"
        "b = main.assemble\n"
        'monkeypatch.setattr(prompts, "_SCRIPTS_RULE", rule)\n'
    )
    assert reads(good_caller) == []


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
    assert len(_reads_a_name_that_left(caller, ROS_VISION, LEFT_ROS_VISION)) == 4
    shim = tmp_path / "vision.py"
    shim.write_text(
        "from walkie_agent.capabilities import (\n"
        "    ObjectSeen,\n"
        "    UnavailableVision,\n"
        ")\n"
        "def _begin(cancel, query, timeout_s, what): ...\n"
        "_no_objects = None\n"
    )
    assert len(_binds_one_of(shim, LEFT_ROS_VISION)) == 3
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
    assert _reads_a_name_that_left(good, ROS_VISION, LEFT_ROS_VISION) == []
    assert _binds_one_of(good, LEFT_ROS_VISION) == []


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
        "from walkie_agent.runtime.model import build_model\n"
        "from walkie_agent.runtime import model\n"
        'monkeypatch.setattr("walkie_agent.runtime.model._chat_model", fake)\n'
        "# src/walkie_agent/runtime/model.py reads config.toml\n"
        "from walkie_agent.adapter import as_tool\n"
        "# src/walkie_agent/adapter.py hides ctx\n"
        "from walkie_agent import errors, adapter\n"
        "from walkie_agent.main import assemble, build_system_prompt\n"
        "from walkie_agent.main import (_TOOL_RULES, _RULES_END)\n"
        "assert system == main.build_system_prompt(tool_set, PLACES, ROBOT)\n"
        "assert with_scripts.count(main._SCRIPTS_RULE) == 1\n"
        'monkeypatch.setattr("walkie_agent.main._LIVE_PEOPLE_RULE", rule)\n'
        "# walkie_agent/main:_places_section lists the places\n"
        "from walkie_agent.runtime import SYSTEM_PROMPT, build_agent\n"
        "from walkie_agent.runtime.agent import SYSTEM_PROMPT\n"
        "assert received == SystemMessage(runtime.SYSTEM_PROMPT)\n"
        'monkeypatch.setattr("walkie_agent.runtime.agent.SYSTEM_PROMPT", prompt)\n'
    )
    assert len(_mentions([bad], REMOVED)) == 22
    # The tools package that holds the atomics is not the removed one,
    # walkie_agent.ros.vision still holds RosVision, the runtime package stays,
    # and the adapter's new home in the tools package is not the removed module.
    good = tmp_path / "good.py"
    good.write_text(
        "from walkie_agent import tools, errors\n"
        "from walkie_agent.tools import atomic\n"
        "import walkie_agent.tools.atomic.memory\n"
        "from walkie_agent.ros.vision import RosVision\n"
        "from walkie_agent.capabilities import UnavailableVision\n"
        "from walkie_agent.runtime import Runner, build_agent\n"
        "from walkie_agent.chat_models import build_model\n"
        "from walkie_agent.tools.adapter import as_tool\n"
        "from walkie_agent.tools import adapter, sets\n"
        "# src/walkie_agent/tools/adapter.py hides ctx\n"
        'caplog.at_level(logging.ERROR, logger="walkie_agent.tools.adapter")\n'
        "from walkie_agent.prompts import SYSTEM_PROMPT, _SCRIPTS_RULE, build_system_prompt\n"
        "assert system == prompts.build_system_prompt(tool_set, PLACES, ROBOT)\n"
        'monkeypatch.setattr("walkie_agent.prompts._LIVE_PEOPLE_RULE", rule)\n'
        "from walkie_agent.main import assemble, check_templates\n"
        "assert main.ARM_TOOL in names and main.assemble is not None\n"
        "# src/walkie_agent/main.py builds the prompt with build_system_prompt\n"
        "assert domain.build_system_prompt is not None\n"
        "def build_agent(model, ctx, *, system_prompt: str = SYSTEM_PROMPT): ...\n"
        'caplog.at_level(logging.WARNING, logger="walkie_agent.runtime.agent")\n'
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
