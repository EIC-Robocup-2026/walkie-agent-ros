"""The package keeps the import rules in pyproject.toml's [tool.importlinter] section.

import-linter runs in this process through its Python API, so the check needs no
lint-imports executable and works in any environment that has the package. It builds
its graph from the walkie_agent that the import system finds, which PYTHONPATH can
point at another checkout, so the first test pins it to this tree.
"""

import importlib.util
import pkgutil
import tomllib
from pathlib import Path

from importlinter import api  # noqa: F401  Importing it configures import-linter's adapters.
from importlinter.application import output, use_cases

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYPROJECT = ROOT / "pyproject.toml"


def lint(config: Path) -> tuple[bool, str]:
    """Check every contract in config; return whether all were kept, and the report."""
    with output.console.capture() as captured:
        kept = use_cases.lint_imports(str(config), cache_dir=None, no_logo=True)
    return kept, captured.get()


def test_the_contracts_check_this_tree():
    spec = importlib.util.find_spec("walkie_agent")
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).is_relative_to(SRC)


def test_the_import_contracts_hold():
    kept, report = lint(PYPROJECT)
    assert kept, report


def test_the_backend_contract_names_every_ros_module_but_io():
    contracts = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["importlinter"]["contracts"]
    (backends,) = [contract for contract in contracts if contract["id"] == "backends-dont-see-the-llm"]
    # Direct children, subpackages included; a named subpackage covers its descendants.
    ros = {
        f"walkie_agent.ros.{child.name}"
        for child in pkgutil.iter_modules([str(SRC / "walkie_agent" / "ros")])
        if child.name != "io"
    }
    named = {name for name in backends["source_modules"] if name.startswith("walkie_agent.ros.")}
    assert named == ros


def test_a_broken_contract_fails_with_the_chain(tmp_path, monkeypatch):
    package = tmp_path / "contract_probe"
    package.mkdir()
    (package / "__init__.py").write_text("from contract_probe import inner\n")
    (package / "inner.py").write_text("")
    config = tmp_path / "pyproject.toml"
    config.write_text(
        "[tool.importlinter]\n"
        'root_package = "contract_probe"\n'
        "[[tool.importlinter.contracts]]\n"
        'name = "The probe init stays empty"\n'
        'type = "forbidden"\n'
        "as_packages = false\n"
        "allow_indirect_imports = true\n"
        'source_modules = ["contract_probe"]\n'
        'forbidden_modules = ["contract_probe.**"]\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    kept, report = lint(config)

    assert not kept
    assert "contract_probe -> contract_probe.inner" in report
