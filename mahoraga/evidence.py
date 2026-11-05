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

If the counters cannot be read, as after a worker crash, the run has no evidence for
anything: nothing counts as used or returned. Every version it watched counts as
raised, since a raise through it may be what was lost, so no other script of the
episode verifies that version. A wrapper adds a frame per call,
so recursion through a watched function reaches the depth limit in half the calls.

A version is named by the sha of its text, which the ledger keeps, and has a shape,
the sha of its syntax tree (shelf.shape). The episode's veto compares shapes, so a
raise through a copy at another depth, or laid out another way, counts against the
version it copies.
"""

import ast
import logging
import re
import secrets
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from pydantic_monty import CollectString, MontyError

from mahoraga.shelf import SHADOWED, definition_source, header, shape, source_sha, top_level_defs

_logger = logging.getLogger(__name__)
# Where the parser ends a line: after \n, and after a \r that no \n follows.
_LINE_ENDS = re.compile(r"(?<=\n)|(?<=\r)(?!\n)")


class Watch:
    """The counters of one run, over the shelf functions it loads and the script's own definitions.

    With `watch_own`, the script's definitions are watched too, saved or not, since the
    same function may already be on the shelf. A watched definition is renamed on its
    own line, so its line numbers hold, and the wrapper takes its name. A stub keeps its
    signature in the type check. Every other definition in the script, at any depth, is
    in `unwatched` as a (name, sha) pair: a raise through it would go unseen, so it
    counts as raised. `shapes` maps the sha of each version the run names to its shape:
    each shelf version, read from `shelf_source`, the source the run feeds first, and
    each of the script's watched and unwatched definitions. With nothing to watch, the
    run has no loader and no token.
    """

    def __init__(
        self,
        code: str,
        shelf_versions: Mapping[str, str],
        *,
        shelf_source: str = "",
        watch_own: bool = False,
        reserved: Collection[str] = (),
    ) -> None:
        watched = _watchable(code, reserved) if watch_own else {}
        others = _unwatched(code, watched) if watch_own else []
        self._tag = f"_h{secrets.token_hex(8)}"
        self.own_versions = {name: _sha(code, node) for name, node in watched.items()}
        self.unwatched = sorted({(node.name, _sha(code, node)) for node in others})
        loaded = top_level_defs(shelf_source)
        self.shapes = {sha: shape(loaded[name]) for name, sha in shelf_versions.items() if name in loaded}
        self.shapes.update((_sha(code, node), shape(node)) for node in [*watched.values(), *others])
        self.code = _renamed(code, watched, self._tag)
        self.stubs = "".join(f"{header(definition_source(code, node))}\n" for node in watched.values())
        self.loader = _loader(self._tag, shelf_versions, watched) if shelf_versions or watched else ""
        self.entered: dict[str, int] = {}
        self.returned: dict[str, int] = {}
        self.unreadable = False
        self._names = set(shelf_versions) | set(watched)

    @property
    def raised(self) -> set[str]:
        """The names entered more often than they returned, or every name if the counters were lost."""
        if self.unreadable:
            return set(self._names)
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
            self.unreadable = True
            _logger.warning("A script's function counters could not be read: %s", exc)

    def clean(self, text: str) -> str:
        """`text` with the wrappers' frames and the token gone, so it shows only the script's own code."""
        if not self.loader:
            return text
        frame = rf'  File "[^"\n]*", line \d+, in [^\n]*\n    value = {self._tag}_[^\n]*\n(?: +[~^]+\n)?'
        return re.sub(frame, "", text).replace(f"{self._tag}_", "")


def _sha(code: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return source_sha(definition_source(code, node))


def _watchable(code: str, reserved: Collection[str]) -> dict[str, ast.FunctionDef]:
    """The top-level definitions a wrapper can stand in for: bound once, on a line that reads `def name`.

    Anything else could rebind the name past the wrapper, so it goes unwatched. A name
    the loader defines itself goes unwatched as well, because a wrapper under it would
    give the loader's replacement the source's sha, and so does a reserved name.
    """
    defs = {
        name: node
        for name, node in top_level_defs(code).items()
        if name not in reserved and name not in SHADOWED
    }
    if not defs:
        return {}
    bound: Counter[str] = Counter()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound[node.name] += 1
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound[node.id] += 1
    lines = _lines(code)
    return {
        name: node
        for name, node in defs.items()
        if not node.decorator_list and bound[name] == 1 and re.match(rf"def\s+{name}\b", lines[node.lineno - 1])
    }


def _unwatched(
    code: str, watched: Mapping[str, ast.FunctionDef]
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every other definition in the script, at any depth.

    That covers a second definition of a name, one under an `if` or inside another
    function, and a decorated one. Each may run and raise where no wrapper sees it,
    and its shape may match a shelf version's, so the caller counts it as raised.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    kept = {(node.lineno, node.col_offset) for node in watched.values()}
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and (node.lineno, node.col_offset) not in kept
    ]


def _lines(code: str) -> list[str]:
    """The script's lines as the parser numbers them, each with its line ending as written.

    The parser ends a line at CR LF, a lone CR or LF, and nowhere else, so
    `lines[node.lineno - 1]` is a node's line, and joining the lines gives the script back.
    """
    return _LINE_ENDS.split(code)


def _renamed(code: str, own: Mapping[str, ast.FunctionDef], tag: str) -> str:
    lines = _lines(code)
    for name, node in own.items():
        lines[node.lineno - 1] = re.sub(rf"^def(\s+){name}\b", rf"def\g<1>{tag}_{name}", lines[node.lineno - 1])
    return "".join(lines)


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
