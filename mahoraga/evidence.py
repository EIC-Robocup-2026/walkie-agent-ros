"""Which of a script's functions ran, from counters inside the worker.

A name in a script's source proves nothing: it may sit in dead code, and a caller
may catch what the function raised, so no traceback ever names it. So a watched
function runs behind a wrapper that counts its entries and its normal returns:

    used      entered at least once
    returned  returned at least once
    raised    entered more often than it returned: an exception crossed its
              boundary, or the run ended inside it

The counting stays in the worker. A host call per function would spend the script's
call budget, and could not happen inside a builtin's key function. No script can
reach the counters: their names carry a random token per run, so no source written
beforehand can name them. A script may read the token from a caught error, but
`locals`, `eval` and `exec` are shadowed here, Monty has no `del` to unshadow them,
and it refuses `globals`, `vars` and the like, so text cannot become a name.

The shelf's source feeds before the loader, because a wrapper needs the function it
stands in for. So a parameter default on a saved function is evaluated before the
shadows and wrappers exist: `evaluate=eval` would keep the real eval, and `fn=helper`
a helper no wrapper counts. Shelf.save therefore takes only literal defaults, and only
a function whose name the loader does not define. Nothing else in a saved definition
runs in that feed: decorators are refused too, and Monty does not evaluate annotations.
This holds against a careless or clever model. It is not a security boundary.

If the counters cannot be read, as after a worker crash, the run has no evidence, so
nothing counts as used and nothing gets verified. A wrapper adds a frame per call,
so recursion through a watched function reaches the depth limit in half the calls.
"""

import ast
import logging
import re
import secrets
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from pydantic_monty import CollectString, MontyError

from mahoraga.shelf import SHADOWED, definition_source, header, source_sha, top_level_defs

_logger = logging.getLogger(__name__)


class Watch:
    """The counters of one run, over the shelf functions it loads and the definitions it may save.

    A definition the script itself will save is renamed on its own line, so its line
    numbers hold, and the wrapper takes its name. A stub keeps its signature in the
    type check. With nothing to watch, the run has no loader and no token.
    """

    def __init__(
        self,
        code: str,
        shelf_versions: Mapping[str, str],
        save: Collection[str] = (),
        reserved: Collection[str] = (),
    ) -> None:
        own = _watchable(code, save, reserved)
        self._tag = f"_h{secrets.token_hex(8)}"
        self.own_versions = {name: source_sha(definition_source(code, node)) for name, node in own.items()}
        self.code = _renamed(code, own, self._tag)
        self.stubs = "".join(f"{header(definition_source(code, node))}\n" for node in own.values())
        self.loader = _loader(self._tag, shelf_versions, own) if shelf_versions or own else ""
        self.entered: dict[str, int] = {}
        self.returned: dict[str, int] = {}

    @property
    def raised(self) -> set[str]:
        return {name for name, count in self.entered.items() if count > self.returned.get(name, 0)}

    def read(self, session: Any) -> None:
        """Take the counters from the worker. This must be the session's last feed."""
        if not self.loader:
            return
        try:
            self.entered, self.returned = session.feed_run(
                f"{self._tag}_in, {self._tag}_out",
                print_callback=CollectString(max_bytes=0),
                skip_type_check=True,
            )
        except (MontyError, RuntimeError) as exc:
            _logger.warning("A script's function counters could not be read: %s", exc)

    def clean(self, text: str) -> str:
        """`text` with the wrappers' frames and the token gone, so it shows only the script's own code."""
        if not self.loader:
            return text
        frame = rf'  File "[^"\n]*", line \d+, in [^\n]*\n    value = {self._tag}_[^\n]*\n(?: +[~^]+\n)?'
        return re.sub(frame, "", text).replace(f"{self._tag}_", "")


def _watchable(code: str, save: Collection[str], reserved: Collection[str]) -> dict[str, ast.FunctionDef]:
    """The definitions in `save` a wrapper can stand in for: bound once, on a line that reads `def name`.

    Anything else could rebind the name past the wrapper. It simply goes unwatched,
    so its run earns no evidence. A name the loader defines itself goes unwatched as
    well, because a wrapper under it would give the loader's replacement the source's sha.
    """
    defs = {
        name: node
        for name, node in top_level_defs(code).items()
        if name in save and name not in reserved and name not in SHADOWED
    }
    if not defs:
        return {}
    bound: Counter[str] = Counter()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound[node.name] += 1
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound[node.id] += 1
    lines = code.split("\n")
    return {
        name: node
        for name, node in defs.items()
        if not node.decorator_list and bound[name] == 1 and re.match(rf"def\s+{name}\b", lines[node.lineno - 1])
    }


def _renamed(code: str, own: Mapping[str, ast.FunctionDef], tag: str) -> str:
    lines = code.split("\n")
    for name, node in own.items():
        lines[node.lineno - 1] = re.sub(rf"^def(\s+){name}\b", rf"def\g<1>{tag}_{name}", lines[node.lineno - 1])
    return "\n".join(lines)


def _loader(tag: str, shelf: Iterable[str], own: Iterable[str]) -> str:
    """Source for a feed after the shelf's: the counters, and a wrapper under each watched name."""
    shelf = list(shelf)
    lines = [f"{tag}_in = {{}}", f"{tag}_out = {{}}", "def locals():", "    return {}"]
    for refused in ("eval", "exec"):
        lines += [f"def {refused}(*args, **kwargs):", f'    raise PermissionError("{refused} is not allowed.")']
    lines += [f"{tag}_{name} = {name}" for name in shelf]
    for name in (*shelf, *own):
        lines += [
            f"def {name}(*args, **kwargs):",
            f'    {tag}_in["{name}"] = {tag}_in.get("{name}", 0) + 1',
            f"    value = {tag}_{name}(*args, **kwargs)",
            f'    {tag}_out["{name}"] = {tag}_out.get("{name}", 0) + 1',
            "    return value",
        ]
    return "\n".join(lines) + "\n"
