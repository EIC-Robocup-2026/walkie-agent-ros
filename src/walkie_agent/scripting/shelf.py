"""The script shelf: saved functions in two tier folders, with a ledger of how they fared.

A store root holds `temporary/<name>.py`, the latest saved version of each
function, `bundled/<name>.py`, a version of each that a passing episode verified,
the latest saved one whenever that one is verified, and `ledger.jsonl` with saved,
run, fail, verified and rejected events, each keyed to the exact source it is
about. Only an operator's failed verdict on the agent writes a rejected event. The
ledger is the record, and the two folders hold the text. A script's run saves into
a writable shelf, whose callable set is temporary/. Two callers verify, each with
what the driver saw run and by the same evidence rule (driver.proven): the gym's
harness, after its ground-truth scorer passes an episode, and the agent's
run_script (scripting.agent_tool), after its operator sends a passing verdict on
/walkie/verdict. Nothing else calls mark_verified. A frozen shelf fixes its
callable set at construction, to the functions verified by then, read from
bundled/, and takes no more definitions, while it keeps counting runs for the
report.

The store root is the tools package's own folder, src/walkie_agent/tools, unless
WALKIE_TOOLS_HOME names another. Git must ignore every file the shelf writes there
(require_ignored).
"""

import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import walkie_agent.tools

TOOLS_HOME = "WALKIE_TOOLS_HOME"
TEMPORARY = "temporary"
BUNDLED = "bundled"
LEDGER = "ledger.jsonl"
# The default store root. The tier folders' tracked .gitignore files and the
# repository's .gitignore ignore every file the shelf writes there.
PACKAGE_STORE = Path(walkie_agent.tools.__file__).resolve().parent
# What require_ignored asks git about: a saved file and the scratch file of an atomic
# write in each tier folder, and the ledger.
_PROBES = (
    f"{TEMPORARY}/ignore_probe.py",
    f"{TEMPORARY}/ignore_probe.tmp",
    f"{BUNDLED}/ignore_probe.py",
    f"{BUNDLED}/ignore_probe.tmp",
    LEDGER,
)
# The loader defines these itself, so a saved function under one could not run as saved.
SHADOWED = frozenset({"locals", "eval", "exec"})


class StoreRefused(ValueError):
    """A store root the shelf will not use. The message says what to do instead."""


def store_root(path: str | Path | None = None) -> Path:
    """A shelf's store root, resolved: `path`, then WALKIE_TOOLS_HOME, then the tools package's own folder."""
    chosen = path or os.environ.get(TOOLS_HOME)
    return Path(chosen).expanduser().resolve() if chosen else PACKAGE_STORE


def require_ignored(root: Path) -> None:
    """Raise StoreRefused unless git ignores every file a shelf writes under the resolved `root`.

    A root with no `.git` at or above it is outside every git work tree, so it passes,
    with or without git. Inside a work tree, `git check-ignore` must report a saved
    file and an atomic write's scratch file in each tier folder, and the ledger, as
    ignored. Anything else fails closed: a path git does not ignore, a tracked file at
    a probed path, or git failing. Without git nothing can be checked there, so only
    the package's own folder passes, whose tracked .gitignore files ignore what the
    shelf writes.
    """
    top = next((parent for parent in (root, *root.parents) if (parent / ".git").exists()), None)
    if top is None:
        return
    elsewhere = f"set {TOOLS_HOME} to a folder outside any git work tree"
    git = shutil.which("git")
    if git is None:
        if root == PACKAGE_STORE:
            return
        raise StoreRefused(
            f"{root} is inside the git work tree at {top}, and git is not installed to check that it "
            f"ignores what the shelf writes there. Install git, {elsewhere}, or unset {TOOLS_HOME} "
            f"to use {PACKAGE_STORE}."
        )
    probes = [(root / probe).relative_to(top).as_posix() for probe in _PROBES]
    # A GIT_DIR or GIT_WORK_TREE from the caller would point git at another repository.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        # check-ignore reads the index, which runs the program core.fsmonitor names in the
        # repository's own config. Turning it off keeps the check from running anything.
        done = subprocess.run(
            [git, "-c", "core.fsmonitor=false", "-C", str(top), "check-ignore", "-z", "--stdin"],
            input="\0".join(probes) + "\0",
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StoreRefused(f"git check-ignore could not run in {top}: {exc}. Fix git there, or {elsewhere}.") from exc
    # 0 means some path is ignored and 1 means none is. Anything else is an error.
    if done.returncode not in (0, 1):
        raise StoreRefused(
            f"git check-ignore failed in {top}: {done.stderr.strip()} Fix git there, or {elsewhere}."
        )
    missing = sorted(set(probes) - set(done.stdout.split("\0")))
    if missing:
        raise StoreRefused(
            f"{root} is inside the git work tree at {top}, and git does not ignore {', '.join(missing)}, "
            f"so saved functions could be committed. Ignore them in a .gitignore there, or {elsewhere}."
        )


def source_sha(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def shape(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The sha of a definition's syntax tree, which an episode's veto compares versions by.

    The tree keeps no positions, comments, line endings or layout, only the values the
    code holds, string values included, so a copy under an `if`, inside another
    function or laid out another way has the shape of the same definition at top
    level, while a copy whose string holds other spaces does not. A string's `u`
    prefix is left out too, as it changes no value (M64). Decorators are left out, as
    a definition's text starts at its `def`, so a decorated copy has its undecorated
    version's shape. A copy matches only as a definition under the same name: one
    bound to the name another way, as a lambda or a renamed definition, is not seen
    (M65). `ast.dump` differs between Python versions, so a shape is only compared
    within the episode that recorded it.
    """
    # A shallow copy, so the caller's node keeps its decorators. The prefixes are cleared
    # in place and put back, because a deep copy recurses several frames per level of
    # nesting and fails on code that ast.dump, the parser and Monty all accept (M66).
    # Every kind that is not None is cleared, as M64 cleared them, so a built tree with
    # an empty kind has the shape of the same tree with none.
    bare = copy.copy(node)
    bare.decorator_list = []
    prefixed = [
        (part, part.kind) for part in ast.walk(bare) if isinstance(part, ast.Constant) and part.kind is not None
    ]
    try:
        for part, _ in prefixed:
            part.kind = None
        return source_sha(ast.dump(bare, include_attributes=False))
    finally:
        for part, kind in prefixed:
            part.kind = kind


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


def definition_source(code: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """A definition's text as written, from its `def` to its last line, as a file on the shelf holds it.

    Its sha names the version in the ledger. Each line ends in `\\n`, because the shelf
    reads its files with universal newlines, so the sha of a version saved from a
    script with `\\r` or `\\r\\n` endings must be the sha the shelf reports for it later.
    The runner gives Monty the script with `\\n` endings already (M62), so this text is
    the text that ran. Monty would keep a `\\r` inside a triple-quoted string, so a
    caller that passes other endings gets the text Python reads, which may hold
    another string than Monty ran. A top-level definition with `\\n` endings keeps the
    text it had before, so every sha on a shelf stays valid. A copy at another depth
    keeps the scope's indentation on its later lines, so its sha differs from the same
    definition's at top level. An episode's veto compares shapes, not shas, for that.
    """
    text = ast.get_source_segment(code, node)
    return text.replace("\r\n", "\n").replace("\r", "\n") + "\n"


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


def _count(sources: Mapping[str, str], events: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Shelf.counts() from the latest saved version of each function and the ledger's events."""
    current = {name: source_sha(source) for name, source in sources.items()}
    counts = {name: {"saved": 0, "run": 0, "fail": 0, "verified": 0} for name in current}
    for event in events:
        name = event["name"]
        if name not in counts or event["event"] not in counts[name]:
            continue
        if event["event"] == "saved" or event["sha"] == current[name]:
            counts[name][event["event"]] += 1
    return counts


def read_counts(root: str | Path | None = None) -> dict[str, dict[str, int]]:
    """Shelf.counts() of the store at store_root(root), read without opening a Shelf.

    It creates no folder and runs no git, so it is safe on a root read out of a log,
    which may name any folder. A folder that holds no store reads as an empty one.
    """
    resolved = store_root(root)
    folder = resolved / TEMPORARY
    sources = {path.stem: path.read_text() for path in sorted(folder.glob("*.py"))} if folder.is_dir() else {}
    ledger = resolved / LEDGER
    events = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.is_file() else []
    return _count(sources, events)


class Shelf:
    """Saved functions in the tier folders of the store that store_root(root) names.

    The root must pass require_ignored before anything is created under it.
    """

    def __init__(self, root: str | Path | None = None, *, frozen: bool = False) -> None:
        self.root = store_root(root)
        require_ignored(self.root)
        self.temporary = self.root / TEMPORARY
        self.bundled = self.root / BUNDLED
        self.temporary.mkdir(parents=True, exist_ok=True)
        self.bundled.mkdir(exist_ok=True)
        self.frozen = frozen
        self._ledger = self.root / LEDGER
        self._lock = threading.Lock()
        # The text of every version this shelf has read or written, by name and sha, so that
        # mark_verified can copy a version to bundled/ after a rewrite replaced it in temporary/.
        self._texts: dict[tuple[str, str], str] = {}
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

    def unverified(self) -> list[str]:
        """The callable functions whose current version has no verified event, sorted.

        A frozen shelf calls only verified versions, so it has none.
        """
        verified = self._verified_versions()
        return sorted(name for name, sha in self.versions().items() if (name, sha) not in verified)

    def rejected(self) -> list[str]:
        """The unverified callable functions whose current version has a rejected event, sorted."""
        unverified = set(self.unverified())
        rejected = self._versions_with("rejected")
        return sorted(name for name, sha in self.versions().items() if name in unverified and (name, sha) in rejected)

    def describe_marked(self, mark: str, rejected_mark: str | None = None) -> str:
        """As describe(), with `mark` on a line of its own above each unverified function.

        With `rejected_mark`, a function in rejected() gets that line instead.
        """
        unverified = set(self.unverified())
        rejected = set(self.rejected()) if rejected_mark is not None else set()

        def marked(name: str, source: str) -> str:
            if name in rejected:
                return f"{rejected_mark}\n{header(source)}"
            return f"{mark}\n{header(source)}" if name in unverified else header(source)

        return "\n\n".join(marked(name, source) for name, source in self._sources().items())

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
            elif name.startswith("__") and name.endswith("__"):
                # Python gives such names their meaning. Saved as __init__.py, one would
                # make a tier folder in the source tree a package of model-written code.
                refused[name] = "A name that starts and ends with two underscores cannot be saved."
            elif node.decorator_list:
                refused[name] = "A decorated function cannot be saved."
            elif not _defaults_are_literal(node):
                refused[name] = "A saved function's parameter defaults must be literal values."
            else:
                chosen[name] = definition_source(code, node)
        problem = check("\n".join(chosen.values())) if chosen else None
        if problem:
            return [], {**refused, **{name: problem for name in chosen}}
        with self._lock:
            self.temporary.mkdir(parents=True, exist_ok=True)
            verified = self._verified_versions() if chosen else set()
            for name, source in chosen.items():
                sha = source_sha(source)
                self._texts[name, sha] = source
                _write_atomically(self.temporary / f"{name}.py", source)
                self._append("saved", name, sha)
                # A version verified before is the latest saved one again, so bundled/ holds it.
                if (name, sha) in verified:
                    self.bundled.mkdir(parents=True, exist_ok=True)
                    _write_atomically(self.bundled / f"{name}.py", source)
        return sorted(chosen), refused

    def record_run(self, versions: Mapping[str, str], ok: bool) -> None:
        """One run or fail event per version of a shelf function a script entered."""
        for name, sha in versions.items():
            self._log("run" if ok else "fail", name, sha)

    def mark_verified(self, versions: Mapping[str, str]) -> None:
        """Verify `versions`, the versions that a passing judgement proved.

        There are two callers. The gym's harness passes what an episode proved once its
        scorer passed the episode. The agent's run_script passes what a task proved once
        the operator's verdict on the task passed. A version is the sha of the source
        that ran, so a rewrite since then stays unverified. Each version gets its verified event. Then bundled/<name>.py gets
        the text, as written, of the version temporary/ holds if that one is verified,
        and otherwise of the version just verified. So bundled/ holds a verified
        latest version whatever order episodes are scored in: an episode that ran an
        older version and is scored after the latest version was verified does not
        replace it. Raises ValueError, with nothing logged or written, for a version
        this shelf never read or wrote, because its text is not known.
        """
        texts = {name: self._text_of(name, sha) for name, sha in versions.items()}
        unknown = [f"{name} {versions[name]}" for name, text in texts.items() if text is None]
        if unknown:
            raise ValueError(f"This shelf never held these versions, so none was verified: {', '.join(unknown)}.")
        with self._lock:
            for name, sha in versions.items():
                self._append("verified", name, sha)
            verified = self._verified_versions()
            self.bundled.mkdir(parents=True, exist_ok=True)
            for name in versions:
                latest = self.temporary / f"{name}.py"
                text = latest.read_text() if latest.is_file() else None
                if text is None or (name, source_sha(text)) not in verified:
                    text = texts[name]
                _write_atomically(self.bundled / f"{name}.py", text)

    def mark_rejected(self, versions: Mapping[str, str]) -> None:
        """One rejected event per version, which an operator's failed verdict judged.

        Only the agent's run_script calls it, with the versions a passing verdict on
        the same task would have verified. It verifies nothing and leaves the files as
        they are, so the function stays callable, and rejected() tells it from one no
        verdict has judged. A later verified event still verifies the version.
        """
        with self._lock:
            for name, sha in versions.items():
                self._append("rejected", name, sha)

    def counts(self) -> dict[str, dict[str, int]]:
        """Saved, run, fail and verified counts per function.

        `saved` counts every version, which is the churn. The rest count the
        version on the shelf now, because a rewrite earned none of the old runs.
        Rejected events are not counted: no gym run writes them.
        """
        return _count(self._all_sources(), self._events())

    def describe(self) -> str:
        """Each callable function's definition line and docstring, for a prompt."""
        return "\n\n".join(header(source) for source in self._sources().values())

    def examples(self) -> str:
        """The full source of each function, for a prompt that shows them as examples."""
        return "\n".join(self._sources().values())

    def _sources(self) -> dict[str, str]:
        return self._all_sources() if self._frozen_sources is None else self._frozen_sources

    def _all_sources(self) -> dict[str, str]:
        """The latest saved version of each function, from temporary/."""
        return self._read(self.temporary)

    def _read(self, folder: Path) -> dict[str, str]:
        """Each `<name>.py` in `folder` by name, with its text remembered by name and sha."""
        if not folder.is_dir():
            return {}
        sources = {path.stem: path.read_text() for path in sorted(folder.glob("*.py"))}
        for name, source in sources.items():
            self._texts[name, source_sha(source)] = source
        return sources

    def _text_of(self, name: str, sha: str) -> str | None:
        """The text of one version, from what this shelf has read or written, or its files now."""
        if (name, sha) not in self._texts:
            for folder in (self.temporary, self.bundled):
                path = folder / f"{name}.py"
                if path.is_file():
                    text = path.read_text()
                    self._texts[name, source_sha(text)] = text
        return self._texts.get((name, sha))

    def _verified_sources(self) -> dict[str, str]:
        """The functions whose verified version in bundled/ is still their latest saved one.

        A file in bundled/ counts only if the ledger holds a verified event for its sha,
        and only while temporary/ holds the same version, so a rewrite that has not been
        verified keeps its function off, as a rewrite starts with nothing. Whenever the
        version in temporary/ is verified, save and mark_verified keep that version in
        bundled/, so this is the set of latest versions with a verified event, as counts()
        reports them. A verified function may name another that never ran, in a branch
        it did not take. On a frozen shelf it could not stand alone, so it stays off as
        well (M18).
        """
        saved = {name: source_sha(source) for name, source in self._all_sources().items()}
        verified = self._verified_versions()
        kept = {
            name: source
            for name, source in self._read(self.bundled).items()
            if saved.get(name) == source_sha(source) and (name, saved[name]) in verified
        }
        while short := [name for name, source in kept.items() if (_names_in(source) & set(saved)) - set(kept)]:
            for name in short:
                del kept[name]
        return kept

    def _events(self) -> list[dict[str, Any]]:
        if not self._ledger.exists():
            return []
        return [json.loads(line) for line in self._ledger.read_text().splitlines()]

    def _verified_versions(self) -> set[tuple[str, str]]:
        """Each (name, sha) the ledger holds a verified event for."""
        return self._versions_with("verified")

    def _versions_with(self, kind: str) -> set[tuple[str, str]]:
        """Each (name, sha) the ledger holds an event of `kind` for."""
        return {(event["name"], event["sha"]) for event in self._events() if event["event"] == kind}

    def _log(self, event: str, name: str, sha: str) -> None:
        with self._lock:
            self._append(event, name, sha)

    def _append(self, event: str, name: str, sha: str) -> None:
        """One line of the ledger. The caller holds the lock."""
        line = json.dumps({"ts": time.time(), "event": event, "name": name, "sha": sha})
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
