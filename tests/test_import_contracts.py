"""The package keeps the import rules in pyproject.toml's [tool.importlinter] section.

import-linter runs in this process through its Python API, so the check needs no
lint-imports executable and works in any environment that has the package. It builds
its graph from the walkie_agent that the import system finds, which PYTHONPATH can
point at another checkout, so the first test pins it to this tree.
"""

import ast
import importlib.util
import pkgutil
import re
import tomllib
from pathlib import Path

import grimp
from importlinter import api  # noqa: F401  Importing it configures import-linter's adapters.
from importlinter.application import output, use_cases

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYPROJECT = ROOT / "pyproject.toml"
# The LangChain, LangGraph and Langfuse packages, integrations such as langchain_openai included.
LLM_PACKAGE = re.compile(r"(langchain|langgraph|langfuse)(_\w+)?")
LLM_CONTRACTS = ("package-inits-stay-empty", "backends-dont-see-the-llm")


def lint(config: Path) -> tuple[bool, str]:
    """Check every contract in config; return whether all were kept, and the report."""
    with output.console.capture() as captured:
        kept = use_cases.lint_imports(str(config), cache_dir=None, no_logo=True)
    return kept, captured.get()


def contract(contract_id: str) -> dict:
    """The contract with this id in pyproject.toml."""
    contracts = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["importlinter"]["contracts"]
    (found,) = [entry for entry in contracts if entry["id"] == contract_id]
    return found


def holds_at_most_a_docstring(source: str) -> bool:
    """Whether a module's source is empty or holds nothing but its docstring."""
    body = ast.parse(source).body
    return not body or (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )


def test_the_contracts_check_this_tree():
    spec = importlib.util.find_spec("walkie_agent")
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).is_relative_to(SRC)


def test_the_import_contracts_hold():
    kept, report = lint(PYPROJECT)
    assert kept, report


def test_the_backend_contract_names_every_ros_module_but_operator_topics():
    contracts = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["importlinter"]["contracts"]
    (backends,) = [contract for contract in contracts if contract["id"] == "backends-dont-see-the-llm"]
    # Direct children, subpackages included; a named subpackage covers its descendants.
    ros = {
        f"walkie_agent.ros.{child.name}"
        for child in pkgutil.iter_modules([str(SRC / "walkie_agent" / "ros")])
        if child.name != "operator_topics"
    }
    named = {name for name in backends["source_modules"] if name.startswith("walkie_agent.ros.")}
    assert named == ros


def test_the_llm_contracts_forbid_every_langchain_package_the_agent_imports():
    # import-linter treats an external package as a leaf, so it never sees that
    # langchain_openai imports langchain_core: a backend that imports langchain_openai
    # keeps both contracts unless langchain_openai is itself forbidden.
    graph = grimp.build_graph("walkie_agent", include_external_packages=True, cache_dir=None)
    imported = {module for module in graph.modules if LLM_PACKAGE.fullmatch(module)}
    assert {"langchain_core", "langchain_openai", "langchain_ollama"} <= imported
    for contract_id in LLM_CONTRACTS:
        assert imported <= set(contract(contract_id)["forbidden_modules"]), contract_id


def test_the_inits_that_must_stay_empty_hold_at_most_a_docstring():
    # package-inits-stay-empty forbids imports only; an init could still define code.
    for package in contract("package-inits-stay-empty")["source_modules"]:
        init = SRC.joinpath(*package.split("."), "__init__.py")
        assert holds_at_most_a_docstring(init.read_text(encoding="utf-8")), init


def test_the_init_check_would_catch_code_beside_the_docstring():
    assert holds_at_most_a_docstring("")
    assert holds_at_most_a_docstring('"""The ROS 2 backends."""\n')
    assert not holds_at_most_a_docstring('"""The ROS 2 backends."""\nCOUNT = 1\n')
    assert not holds_at_most_a_docstring('"""The ROS 2 backends."""\n\n\ndef helper():\n    return 1\n')
    assert not holds_at_most_a_docstring("COUNT = 1\n")


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
