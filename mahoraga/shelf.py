"""The script shelf: saved functions outside git, with a ledger of how they fared.

One `<name>.py` per function, plus `ledger.jsonl` with saved, run, fail and
verified events, each keyed to the exact source it is about. A script's run saves
into a writable shelf. Only the harness calls mark_verified, after the ground-truth
scorer, with what the driver saw run. A frozen shelf fixes its callable set at
construction, to the functions verified by then, and takes no more definitions,
while it keeps counting runs for the report.
"""

import ast
import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from mahoraga import home

LEDGER = "ledger.jsonl"
# The loader defines these itself, so a saved function under one could not run as saved.
SHADOWED = frozenset({"locals", "eval", "exec"})


def source_sha(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _names_in(code: str) -> set[str]:
    """Every bare name in `code`. Broken code has none, so Monty reports the syntax error."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def top_level_defs(code: str) -> dict[str, ast.FunctionDef]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def definition_source(code: str, node: ast.FunctionDef) -> str:
    """A definition's text, as its file on the shelf holds it."""
    return ast.get_source_segment(code, node) + "\n"


def _defaults_are_literal(node: ast.FunctionDef) -> bool:
    """Whether every parameter default is plain data, as evidence's docstring explains.

    A call is refused by hand, because literal_eval accepts `set()`, and in the shelf's
    feed that would run a saved function named `set`.
    """
    # kw_defaults holds None for a keyword-only parameter that has no default.
    for default in [*node.args.defaults, *filter(None, node.args.kw_defaults)]:
        if any(isinstance(part, ast.Call) for part in ast.walk(default)):
            return False
        try:
            ast.literal_eval(default)
        except (ValueError, TypeError):
            return False
    return True


class Shelf:
    """Saved functions under `root`, by default the shelf/ directory of mahoraga.home()."""

    def __init__(self, root: str | Path | None = None, *, frozen: bool = False) -> None:
        self.root = Path(root).expanduser().resolve() if root else home() / "shelf"
        home(self.root)
        self.frozen = frozen
        self._ledger = self.root / LEDGER
        self._lock = threading.Lock()
        self._frozen_sources = self._verified_sources() if frozen else None

    def names(self) -> list[str]:
        """The functions a script may call, sorted."""
        return sorted(self._sources())

    def digest(self) -> str:
        """Identifies the callable set, for the log of a run on a frozen shelf."""
        return source_sha(json.dumps(self._sources(), sort_keys=True))

    def versions(self) -> dict[str, str]:
        """The sha of each callable function's current source, by name."""
        return {name: source_sha(source) for name, source in self._sources().items()}

    def source_for(self, code: str) -> tuple[str, dict[str, str]]:
        """The shelf functions `code` names, transitively, as source to run first, and each one's sha.

        Naming a function only loads it: Watch tells whether it ran. A name the
        script defines itself is the script's own. A shelf function is plain source,
        so its atomic calls pass the same driver as the script's.
        """
        sources = self._sources()
        own = set(top_level_defs(code))
        used: set[str] = set()
        todo = [code]
        while todo:
            for name in _names_in(todo.pop()):
                if name in sources and name not in used and name not in own:
                    used.add(name)
                    todo.append(sources[name])
        names = sorted(used)
        return "\n".join(sources[name] for name in names), {name: source_sha(sources[name]) for name in names}

    def save(
        self,
        code: str,
        names: Sequence[str],
        *,
        reserved: Collection[str] = (),
        check: Callable[[str], str | None] = lambda source: None,
    ) -> tuple[list[str], dict[str, str]]:
        """Keep the named top-level definitions of a script that succeeded.

        Returns the names saved and, by name, why others were refused. `reserved`
        holds the allow-list's names, which a definition must not shadow. `check`
        gets the source together and returns why it cannot stand alone, if so.
        """
        if self.frozen:
            return [], {name: "The shelf is frozen." for name in names}
        defs = top_level_defs(code)
        chosen: dict[str, str] = {}
        refused: dict[str, str] = {}
        for name in names:
            node = defs.get(name)
            if node is None:
                refused[name] = "No top-level function has this name."
            elif name in reserved or name in SHADOWED:
                refused[name] = "This name belongs to a built-in function."
            elif node.decorator_list:
                refused[name] = "A decorated function cannot be saved."
            elif not _defaults_are_literal(node):
                refused[name] = "A saved function's parameter defaults must be literal values."
            else:
                chosen[name] = definition_source(code, node)
        problem = check("\n".join(chosen.values())) if chosen else None
        if problem:
            return [], {**refused, **{name: problem for name in chosen}}
        self.root.mkdir(parents=True, exist_ok=True)
        for name, source in chosen.items():
            _write_atomically(self.root / f"{name}.py", source)
            self._log("saved", name, source_sha(source))
        return sorted(chosen), refused

    def record_run(self, versions: Mapping[str, str], ok: bool) -> None:
        """One run or fail event per version of a shelf function a script entered."""
        for name, sha in versions.items():
            self._log("run" if ok else "fail", name, sha)

    def mark_verified(self, versions: Mapping[str, str]) -> None:
        """For the harness only, with the versions that ran in an episode the scorer passed.

        A version is the sha of the source that ran, so a rewrite since then stays unverified.
        """
        for name, sha in versions.items():
            self._log("verified", name, sha)

    def counts(self) -> dict[str, dict[str, int]]:
        """Saved, run, fail and verified counts per function.

        `saved` counts every version, which is the churn. The rest count the
        version on the shelf now, because a rewrite earned none of the old runs.
        """
        current = {name: source_sha(source) for name, source in self._all_sources().items()}
        counts = {
            name: {"saved": 0, "run": 0, "fail": 0, "verified": 0} for name in current
        }
        for event in self._events():
            name = event["name"]
            if name in counts and (event["event"] == "saved" or event["sha"] == current[name]):
                counts[name][event["event"]] += 1
        return counts

    def describe(self) -> str:
        """Each callable function's definition line and docstring, for a prompt."""
        return "\n\n".join(header(source) for source in self._sources().values())

    def examples(self) -> str:
        """The full source of each function, for a prompt that shows them as examples."""
        return "\n".join(self._sources().values())

    def _sources(self) -> dict[str, str]:
        return self._all_sources() if self._frozen_sources is None else self._frozen_sources

    def _all_sources(self) -> dict[str, str]:
        if not self.root.is_dir():
            return {}
        return {path.stem: path.read_text() for path in sorted(self.root.glob("*.py"))}

    def _verified_sources(self) -> dict[str, str]:
        """The current versions verified so far, without any that names a saved function left out.

        A verified function may name another that never ran, in a branch it did not
        take. On a frozen shelf it could not stand alone, so it stays off as well.
        """
        counts = self.counts()
        sources = self._all_sources()
        kept = {name: source for name, source in sources.items() if counts[name]["verified"] > 0}
        while short := [name for name, source in kept.items() if (_names_in(source) & set(sources)) - set(kept)]:
            for name in short:
                del kept[name]
        return kept

    def _events(self) -> list[dict[str, Any]]:
        if not self._ledger.exists():
            return []
        return [json.loads(line) for line in self._ledger.read_text().splitlines()]

    def _log(self, event: str, name: str, sha: str) -> None:
        line = json.dumps({"ts": time.time(), "event": event, "name": name, "sha": sha})
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self._ledger.open("a") as ledger:
                ledger.write(line + "\n")


def header(source: str) -> str:
    """A definition with its body cut down to the docstring."""
    node = ast.parse(source).body[0]
    doc = ast.get_docstring(node)
    node.body = [ast.Expr(ast.Constant(doc if doc else ...))]
    return ast.unparse(node)


def _write_atomically(path: Path, text: str) -> None:
    """Replace `path` in one step, so a concurrent reader sees the old file or the new one."""
    handle, scratch = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(handle, "w") as out:
        out.write(text)
    os.replace(scratch, path)
