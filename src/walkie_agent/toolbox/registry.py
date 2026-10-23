"""Finds the atomics in a package and builds their tools.

Atomics are reviewed code, so a broken one stops the agent at startup with
every problem listed, not at its first call in the middle of a task.
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Any

from pydantic import BaseModel, ValidationError

from walkie_agent.toolbox.atomic import Atomic, AtomicTool
from walkie_agent.toolbox.types import ToolMeta, ToolOutput

logger = logging.getLogger(__name__)

DEFAULT_PACKAGE = "walkie_agent.tools.atomic"


class AtomicLoadError(Exception):
    """One or more atomics are broken. `problems` lists every one found."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        lines = "\n".join(f"- {problem}" for problem in self.problems)
        super().__init__(f"{len(self.problems)} problems with the atomic tools:\n{lines}")


def discover(package: str = DEFAULT_PACKAGE) -> list[Atomic]:
    """Import every module of `package` and return the atomics, sorted by name.

    Modules whose names start with "_" are skipped, so helpers can live next
    to the atomics. Raises AtomicLoadError listing every problem found.
    """
    root = importlib.import_module(package)
    problems: list[str] = []
    atomics: list[Any] = []
    owners: dict[str, list[str]] = {}
    for info in sorted(pkgutil.iter_modules(root.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module_name = f"{package}.{info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name}: import failed: {type(e).__name__}: {e}")
            continue
        found = _check(module, info.name)
        # A module whose META.name is not its own name may still clash with
        # another tool, so every valid META counts towards the name check.
        if isinstance(getattr(module, "META", None), ToolMeta):
            owners.setdefault(module.META.name, []).append(module_name)
        if found:
            problems.extend(f"{module_name}: {problem}" for problem in found)
            continue
        atomics.append(module)
    for name, modules in owners.items():
        if len(modules) > 1:
            problems.append(f"The tool name {name!r} is used by {', '.join(modules)}.")
    if problems:
        raise AtomicLoadError(problems)
    return atomics


def build_atomic_tools(ctx: Any, atomics: list[Atomic]) -> list[AtomicTool]:
    """One AtomicTool per atomic whose required backends `ctx` has."""
    tools = []
    for atomic in atomics:
        missing = [name for name in atomic.META.requires if not ctx.has_backend(name)]
        if missing:
            logger.info(
                f"Skipped the atomic tool {atomic.META.name}: "
                f"missing backends {', '.join(missing)}."
            )
            continue
        tools.append(AtomicTool(atomic, ctx))
    return tools


def _check(module: Any, module_name: str) -> list[str]:
    """The problems with one module, or [] for a sound atomic."""
    problems = []
    meta = getattr(module, "META", None)
    args = getattr(module, "Args", None)
    out = getattr(module, "Out", None)
    run = getattr(module, "run", None)

    if not isinstance(meta, ToolMeta):
        problems.append("META is missing or not a ToolMeta.")
        meta = None
    elif meta.name != module_name:
        problems.append(f"META.name is {meta.name!r}, but the module is named {module_name!r}.")

    if not (isinstance(args, type) and issubclass(args, BaseModel)):
        problems.append("Args is missing or not a pydantic BaseModel subclass.")
        args = None
    if not (isinstance(out, type) and issubclass(out, ToolOutput)):
        problems.append("Out is missing or not a ToolOutput subclass.")

    if not callable(run):
        problems.append("run is missing or not callable.")
    elif inspect.iscoroutinefunction(run) or inspect.isasyncgenfunction(run):
        # AtomicTool calls run() and never awaits it, so an async run would
        # only fail on its first call.
        problems.append("run is async; AtomicTool calls it synchronously.")
    else:
        try:
            inspect.signature(run).bind(None, None)
        except (TypeError, ValueError):
            problems.append("run cannot be called as run(ctx, args).")

    if meta is not None and args is not None:
        for number, example in enumerate(meta.examples, start=1):
            try:
                args.model_validate(example)
            except ValidationError as e:
                problems.append(f"example {number} does not fit Args: {_first_line(e)}")
            except Exception as e:
                # A validator may raise something else; it is still this
                # module's problem, reported with the others.
                problems.append(
                    f"example {number} does not fit Args: {type(e).__name__}: {e}"
                )
    return problems


def _first_line(error: ValidationError) -> str:
    details = error.errors()
    first = details[0] if details else {}
    location = ".".join(str(part) for part in first.get("loc", ())) or "(root)"
    extra = f" and {len(details) - 1} more" if len(details) > 1 else ""
    return f"{location}: {first.get('msg', str(error))}{extra}"
