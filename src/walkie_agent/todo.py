"""The todo list that the agent keeps for each task (ADR-0041).

In a task with several steps, an LLM tends to drop a step, to call the task
done once it has spoken, or to report success after a tool failed. The todo
list makes the plan explicit, and its rules are enforced here in code rather
than in the prompt, so the model cannot talk its way past them:

- at most one item is in progress or waiting, because physical work is serial;
- a failed or cancelled item carries a note that says why;
- seeded items keep their wording and cannot be removed;
- an error ledger refuses "completed" while a tool failure in the item's window
  is unresolved, or while the tool the item requires has not succeeded.

The module does not import ROS, so the rules can be tested on their own. The
agent graph renders the list into a reminder before each LLM call and reports
every tool result to the ledger. Listeners receive a snapshot after every
change, which the ROS side publishes.
"""

import json
import logging
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

TodoStatus = Literal["pending", "in_progress", "waiting", "completed", "failed", "cancelled"]

# Items that still need work.
OPEN_STATUSES: tuple[str, ...] = ("pending", "in_progress", "waiting")
# The statuses an update_todos op may set. Only wait_for_person sets "waiting".
OP_STATUSES: tuple[str, ...] = ("pending", "in_progress", "completed", "failed", "cancelled")
# At most one item may have one of these statuses at a time.
_ACTIVE_STATUSES = ("in_progress", "waiting")
_NOTE_REQUIRED = ("failed", "cancelled")

_MARKERS = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "waiting": "[?]",
    "completed": "[x]",
    "failed": "[!]",
    "cancelled": "[-]",
}

# A rejected update_todos call is a mistake in the list, not a failed step, so
# the ledger ignores the todo tool's own results.
_LEDGER_IGNORES = frozenset({"update_todos"})

_SEED_KEYS = frozenset({"content", "active_form", "requires_tool"})


class TodoItem(BaseModel):
    """One step of the current task, as rendered for the LLM and published."""

    id: str  # "t1", "t2", ...; stable for the whole task
    content: str  # the step, such as "Go to the kitchen"
    active_form: str  # its "-ing" line for the console, such as "Going to the kitchen"
    status: TodoStatus = "pending"
    note: str = ""  # why the step failed, was cancelled, or is done despite a failure
    source: Literal["seed", "agent", "operator"] = "agent"
    locked: bool = False  # seeded items: status and note may change, wording may not
    requires_tool: str = ""  # a tool that must succeed before the item can complete
    evidence: list[str] = Field(default_factory=list)  # ids of successful tool calls
    question: str = ""  # the question asked while the item is waiting


# TodoOp becomes the update_todos tool schema, and its docstring becomes the
# description the LLM reads, so the reasoning for its shape lives here.
# Every op has the same flat shape. A union of separate add, update and remove
# shapes would become anyOf in the schema, and so would Optional fields (anyOf
# with null). Gemini function declarations and small Ollama models handle anyOf
# poorly, so every field is a plain string where empty means "not given".
# TodoList.apply checks which fields each op needs and which statuses are
# allowed.
class TodoOp(BaseModel):
    """One change to the todo list: add, update or remove an item."""

    op: Literal["add", "update", "remove"] = Field(
        description="add a new item, update an existing item, or remove an item."
    )
    id: str = Field(
        default="",
        description=(
            "The id of the item to update or remove, such as t2. "
            "Leave empty for add: ids are assigned."
        ),
    )
    content: str = Field(
        default="",
        description="The step in a few words, such as 'Go to the kitchen'. Required for add.",
    )
    active_form: str = Field(
        default="",
        description=(
            "The step as an -ing phrase, such as 'Going to the kitchen'. "
            "Optional; defaults to content."
        ),
    )
    status: str = Field(
        default="",
        description=(
            "One of pending, in_progress, completed, failed, cancelled. "
            "Optional; a new item starts as pending."
        ),
    )
    note: str = Field(
        default="",
        description=(
            "Why the step failed or was cancelled, or why it is complete although "
            "a tool failed. Required with failed and cancelled."
        ),
    )
    after: str = Field(
        default="",
        description=(
            "Place the item right after the item with this id. "
            "Optional; a new item goes at the end."
        ),
    )

    # Some models fill the fields they do not use with null. Reading null as
    # "not given" keeps the schema free of anyOf and spares the model a
    # validation error it may not recover from.
    @field_validator("id", "content", "active_form", "status", "note", "after", mode="before")
    @classmethod
    def _null_is_empty(cls, value: Any) -> Any:
        return "" if value is None else value


class TodoRejected(ValueError):
    """Raised when a change to the todo list breaks a rule.

    The message is one sentence that tells the LLM what to do instead.
    """


TodoListener = Callable[[list[TodoItem], list[TodoItem]], None]


@dataclass(frozen=True)
class _LedgerEntry:
    seq: int
    tool: str
    call_id: str
    ok: bool


class TodoList:
    """The todo list of the current task, with its rules and error ledger.

    Thread-safe: one lock guards all state, because the agent thread edits the
    list while a ROS callback may reset or seed it. Listeners are called
    outside that lock.

    Each item has a window in the error ledger. It starts when the item is
    created and starts again whenever an op moves the item into in_progress
    from another status, waiting included, so a retry begins with a clean
    slate. resume_waiting keeps the window, because the answer to a question
    continues the same attempt. Failures inside the window gate "completed".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[TodoItem] = []
        # The ledger length when each item's window started, by item id. Kept
        # out of TodoItem so it never reaches the LLM or the published JSON.
        self._window_start: dict[str, int] = {}
        self._next_id = 1
        self._ledger: list[_LedgerEntry] = []
        self._listeners: list[TodoListener] = []
        # Listener calls not yet delivered, in the order the changes were made.
        self._pending: deque[tuple[list[TodoItem], list[TodoItem]]] = deque()
        # True while a thread is calling the listeners. Guarded by _lock.
        self._delivering = False

    # -- Changes ---------------------------------------------------------------

    def seed(
        self,
        items: Iterable[str | dict[str, Any]],
        source: Literal["seed", "operator"] = "seed",
    ) -> None:
        """Replace the whole list with locked items, before the task starts.

        Each item is its content as a string, or a dict with content and
        optionally active_form and requires_tool. Seeding also resets the id
        counter and the error ledger.
        """
        if source not in ("seed", "operator"):
            raise ValueError(f"A seed's source must be 'seed' or 'operator', not {source!r}.")
        # A string is iterable too, and would seed one item per character.
        if isinstance(items, str):
            raise TodoRejected("A seed is a list of items, not a single string.")
        seeded = [_seed_item(f"t{n}", entry, source) for n, entry in enumerate(items, start=1)]
        with self._lock:
            had_items = bool(self._items)
            self._items = seeded
            self._window_start = {item.id: 0 for item in seeded}
            self._next_id = len(seeded) + 1
            self._ledger = []
            if not (seeded or had_items):
                return
            self._queue([item.id for item in seeded])
        self._deliver()

    def apply(self, ops: list[TodoOp | dict[str, Any]]) -> str:
        """Apply the LLM's ops in order, all or nothing.

        Returns one sentence that sums up the change. Raises TodoRejected,
        leaving the list unchanged, if any op is invalid or the final list
        breaks a rule.
        """
        parsed = [_parse_op(op) for op in ops]
        if not parsed:
            raise TodoRejected("The call has no ops; send at least one add, update or remove.")
        with self._lock:
            draft = _Draft(self._items, self._window_start, self._next_id, self._ledger)
            for op in parsed:
                draft.apply(op)
            draft.check_one_active()

            old_status = {item.id: item.status for item in self._items}
            self._items = draft.items
            self._window_start = draft.window_start
            self._next_id = draft.next_id
            # Added items count as a status change, as seeded items do.
            self._queue([i.id for i in self._items if old_status.get(i.id) != i.status])
            summary = draft.summary()
        self._deliver()
        return summary

    def record_tool_result(self, tool_name: str, call_id: str, ok: bool) -> None:
        """Add a tool result to the error ledger.

        A successful call also becomes evidence for the item in progress or
        waiting, if there is one. New evidence is a change to the list, so
        the listeners hear of it, with no item in `changed`.
        """
        if tool_name in _LEDGER_IGNORES:
            return
        with self._lock:
            seq = len(self._ledger) + 1
            self._ledger.append(_LedgerEntry(seq=seq, tool=tool_name, call_id=call_id, ok=ok))
            active = self._active_item()
            if not (ok and call_id and active is not None):
                return
            active.evidence.append(call_id)
            self._queue([])
        self._deliver()

    def set_waiting(self, question: str) -> str | None:
        """Mark the item in progress as waiting for an answer to `question`.

        Returns its id, or None when no item is in progress (nothing changes).
        """
        with self._lock:
            item = next((i for i in self._items if i.status == "in_progress"), None)
            if item is None:
                return None
            item.status = "waiting"
            item.question = _one_line(question)
            self._queue([item.id])
        self._deliver()
        return item.id

    def resume_waiting(self) -> None:
        """Put the waiting item back in progress, once the answer has arrived.

        The item's ledger window is kept: asking a question does not start a
        new attempt, so a failure from before the question still counts.
        """
        with self._lock:
            item = next((i for i in self._items if i.status == "waiting"), None)
            if item is None:
                return
            item.status = "in_progress"
            item.question = ""
            self._queue([item.id])
        self._deliver()

    def clear(self) -> None:
        """Empty the list and reset the id counter and the ledger."""
        with self._lock:
            had_items = bool(self._items)
            self._items = []
            self._window_start = {}
            self._next_id = 1
            self._ledger = []
            if not had_items:
                return
            self._queue([])
        self._deliver()

    def add_listener(self, callback: TodoListener) -> None:
        """Call `callback(items, changed)` after every change to the list.

        `items` is a snapshot of the new list and `changed` holds the items
        whose status changed (every item for a seed, none for a clear or new
        evidence). A
        listener that raises is logged and does not affect the list. While
        one thread is calling the listeners, a change made on another thread
        is delivered by that first thread, so the call may come after the
        method that made the change has returned.
        """
        with self._lock:
            self._listeners.append(callback)

    # -- Reading ---------------------------------------------------------------

    def snapshot(self) -> list[TodoItem]:
        """Deep copies of the items, in list order."""
        with self._lock:
            return self._copy_items()

    def open_ids(self) -> list[str]:
        """Ids of the pending, in-progress and waiting items, in list order."""
        with self._lock:
            return [item.id for item in self._items if item.status in OPEN_STATUSES]

    def has_waiting(self) -> bool:
        with self._lock:
            return any(item.status == "waiting" for item in self._items)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return not self._items

    def render(self) -> str:
        """The reminder sent to the LLM after the history; "" for an empty list."""
        with self._lock:
            if not self._items:
                return ""
            lines = [_render_line(item) for item in self._items]
        return "\n".join(["<todos>", *lines, "</todos>"])

    def to_json(self) -> str:
        """The items as a JSON array, in list order."""
        with self._lock:
            return json.dumps([item.model_dump() for item in self._items], ensure_ascii=False)

    # -- Internals -------------------------------------------------------------

    def _copy_items(self) -> list[TodoItem]:
        return [item.model_copy(deep=True) for item in self._items]

    def _active_item(self) -> TodoItem | None:
        return next((i for i in self._items if i.status in _ACTIVE_STATUSES), None)

    def _queue(self, changed_ids: list[str]) -> None:
        """Queue a listener call for the current list. Call with the lock held."""
        items = self._copy_items()
        changed = [item for item in items if item.id in changed_ids]
        self._pending.append((items, changed))

    def _deliver(self) -> None:
        """Call the listeners for every queued change, outside the lock.

        One thread at a time drains the queue, so changes made on different
        threads, such as an edit on the agent thread and a reset from a ROS
        callback, reach the listeners in the order they were made, and the
        last snapshot each listener sees is the current list. A change made
        while the queue is being drained, by a listener or by another thread,
        is only queued here: the draining thread delivers it after the change
        it is delivering now. So a listener may change the list, and a change
        on another thread never waits for the listeners to finish.
        """
        with self._lock:
            if self._delivering:
                return
            self._delivering = True
        try:
            while True:
                with self._lock:
                    if not self._pending:
                        # Cleared together with the empty check, so a change
                        # queued after it is delivered by the thread that
                        # made it.
                        self._delivering = False
                        return
                    items, changed = self._pending.popleft()
                    listeners = list(self._listeners)
                for listener in listeners:
                    try:
                        listener(items, changed)
                    except Exception:
                        logger.exception("A todo list listener failed; the list is unaffected.")
        except BaseException:
            # Only an interrupt such as KeyboardInterrupt gets here. The next
            # change delivers whatever is still queued.
            with self._lock:
                self._delivering = False
            raise


class _Draft:
    """A working copy of the list that one apply() call edits.

    apply() keeps the draft only if every op and the final check pass, which
    makes a call all or nothing. It also collects the summary sentence.
    """

    def __init__(
        self,
        items: list[TodoItem],
        window_start: dict[str, int],
        next_id: int,
        ledger: list[_LedgerEntry],
    ):
        self.items = [item.model_copy(deep=True) for item in items]
        self.window_start = dict(window_start)
        self.next_id = next_id
        self.ledger = ledger  # read only
        self.added: list[str] = []
        # Updated ids in first-update order, each with its new status, or ""
        # when its status did not change.
        self.updated: dict[str, str] = {}
        self.removed: list[str] = []

    def apply(self, op: TodoOp) -> None:
        if op.op == "add":
            self._add(op)
        elif op.op == "update":
            self._update(op)
        else:
            self._remove(op)

    def check_one_active(self) -> None:
        active = [item.id for item in self.items if item.status in _ACTIVE_STATUSES]
        if len(active) > 1:
            raise TodoRejected(
                f"Only one item can be in progress or waiting at a time, but {_join(active)} "
                "would be; finish the others or set them back to pending."
            )

    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f"Added: {', '.join(self.added)}.")
        if self.updated:
            updates = [f"{i} → {s}" if s else i for i, s in self.updated.items()]
            parts.append(f"Updated: {', '.join(updates)}.")
        if self.removed:
            parts.append(f"Removed: {', '.join(self.removed)}.")
        open_ids = [item.id for item in self.items if item.status in OPEN_STATUSES]
        parts.append(f"Open: {', '.join(open_ids) or 'none'}.")
        return " ".join(parts)

    def _add(self, op: TodoOp) -> None:
        if op.id:
            raise TodoRejected(
                "Ids are assigned by the list, so leave id empty when you add an item."
            )
        if not op.content:
            raise TodoRejected("An add op needs content that says what the step is.")
        item_id = f"t{self.next_id}"
        status = op.status or "pending"
        _check_status(item_id, status, op.note)
        item = TodoItem(
            id=item_id,
            content=op.content,
            active_form=op.active_form or op.content,
            status=status,
            note=op.note,
        )
        index = len(self.items)
        if op.after:
            missing = f"There is no item {op.after} to add the new item after."
            index = self._index(op.after, missing) + 1
        self.window_start[item_id] = len(self.ledger)
        if status == "completed":
            self._check_completion(item, op.note)
        self.items.insert(index, item)
        self.next_id += 1
        self.added.append(item_id)

    def _update(self, op: TodoOp) -> None:
        if not op.id:
            raise TodoRejected("An update op needs the id of the item to change.")
        index = self._index(op.id)
        item = self.items[index]
        before = item.model_copy(deep=True)

        rewords = (op.content and op.content != item.content) or (
            op.active_form and op.active_form != item.active_form
        )
        if item.locked and rewords:
            raise TodoRejected(f"{item.id} is a seeded step, so its wording cannot change.")
        if op.content:
            # An active form that was only defaulted from the content follows it.
            if not op.active_form and item.active_form == item.content:
                item.active_form = op.content
            item.content = op.content
        if op.active_form:
            item.active_form = op.active_form

        if op.status:
            _check_status(item.id, op.status, op.note)
            if op.status == "completed" and item.status != "completed":
                self._check_completion(item, op.note)
            # An op that sets in_progress starts a new attempt, even from
            # waiting, so it starts a new window. Only resume_waiting keeps
            # the window of the attempt that asked the question.
            if op.status == "in_progress" and item.status != "in_progress":
                self.window_start[item.id] = len(self.ledger)
            # An op never sets waiting, so the item is no longer asking.
            item.question = ""
            item.status = op.status
        if op.note:
            item.note = op.note

        moved = False
        if op.after:
            if op.after == item.id:
                raise TodoRejected(f"{item.id} cannot be moved after itself.")
            self._index(op.after, f"There is no item {op.after} to move {item.id} after.")
            self.items.pop(index)
            target = self._index(op.after) + 1
            self.items.insert(target, item)
            moved = target != index

        if item == before and not moved:
            raise TodoRejected(
                f"The update to {item.id} changes nothing; leave it out or change a field."
            )
        if item.status != before.status:
            self.updated[item.id] = item.status
        else:
            self.updated.setdefault(item.id, "")

    def _remove(self, op: TodoOp) -> None:
        if not op.id:
            raise TodoRejected("A remove op needs the id of the item to remove.")
        index = self._index(op.id)
        item = self.items[index]
        if item.locked:
            raise TodoRejected(
                f"{item.id} is a seeded step, so it cannot be removed; "
                "mark it cancelled with a note instead."
            )
        del self.items[index]
        self.window_start.pop(item.id, None)
        self.removed.append(item.id)

    def _index(self, item_id: str, missing: str = "") -> int:
        for index, item in enumerate(self.items):
            if item.id == item_id:
                return index
        raise TodoRejected(
            missing or f"There is no item {item_id}; use an id from the todo list."
        )

    def _check_completion(self, item: TodoItem, note: str) -> None:
        """The two ledger checks that run before an item becomes completed."""
        start = self.window_start.get(item.id, 0)
        window = [entry for entry in self.ledger if entry.seq > start]

        # A note cannot stand in for the required tool.
        if item.requires_tool and not any(
            entry.ok and entry.tool == item.requires_tool for entry in window
        ):
            raise TodoRejected(
                f"{item.id} needs a successful {item.requires_tool} before it can be completed."
            )

        # A failure is resolved by any later success of the same tool. Walking
        # the window backwards collects the tools that succeeded later on.
        succeeded_later: set[str] = set()
        unresolved: list[str] = []
        for entry in reversed(window):
            if entry.ok:
                succeeded_later.add(entry.tool)
            elif entry.tool not in succeeded_later and entry.tool not in unresolved:
                unresolved.append(entry.tool)
        if unresolved and not note:
            unresolved.reverse()
            verb = "has" if len(unresolved) == 1 else "have"
            raise TodoRejected(
                f"{item.id} cannot be completed because {_join(unresolved)} failed and "
                f"{verb} not succeeded since; retry it, or add a note that explains "
                "why the step is done anyway."
            )


def _check_status(item_id: str, status: str, note: str) -> None:
    """Checks a status set by an op, and the note it needs."""
    if status == "waiting":
        raise TodoRejected(
            "Only wait_for_person can set waiting; call it to ask the person a question."
        )
    if status not in OP_STATUSES:
        raise TodoRejected(
            f"Status '{status}' is not allowed; use one of {', '.join(OP_STATUSES)}."
        )
    if status in _NOTE_REQUIRED and not note:
        raise TodoRejected(f"Marking {item_id} {status} needs a note that says why.")


def _parse_op(op: TodoOp | dict[str, Any]) -> TodoOp:
    """Validates an op and normalizes its text fields to single trimmed lines."""
    if not isinstance(op, TodoOp):
        try:
            op = TodoOp.model_validate(op)
        except ValidationError as e:
            error = e.errors()[0]
            where = ".".join(str(part) for part in error["loc"]) or "op"
            raise TodoRejected(f"An op is malformed at '{where}': {error['msg']}.") from e
    fields = {
        name: _one_line(getattr(op, name))
        for name in ("id", "content", "active_form", "status", "note", "after")
    }
    fields["status"] = fields["status"].lower()
    return op.model_copy(update=fields)


def _seed_item(item_id: str, entry: str | dict[str, Any], source: str) -> TodoItem:
    if isinstance(entry, str):
        entry = {"content": entry}
    if not isinstance(entry, dict):
        raise TodoRejected(
            f"Seed item {item_id} must be a string or a dict, not {type(entry).__name__}."
        )
    # An unknown key is most likely a typo, such as "require_tool", that would
    # silently drop a check.
    unknown = sorted(set(entry) - _SEED_KEYS)
    if unknown:
        raise TodoRejected(f"Seed item {item_id} has unknown keys: {', '.join(unknown)}.")
    content = _one_line(entry.get("content"))
    if not content:
        raise TodoRejected(f"Seed item {item_id} has no content.")
    return TodoItem(
        id=item_id,
        content=content,
        active_form=_one_line(entry.get("active_form")) or content,
        source=source,
        locked=True,
        requires_tool=_one_line(entry.get("requires_tool")),
    )


def console_line(item: TodoItem) -> str:
    """The operator console's line for an item whose status just changed.

    "t1: Going to the kitchen" when the item goes in progress, and "t1
    completed: Go to the kitchen" when it is completed, failed, cancelled or
    waiting. A pending item gets "", because a step that has not started yet
    is not news on the console.
    """
    if item.status == "in_progress":
        return f"{item.id}: {item.active_form or item.content}"
    if item.status == "pending":
        return ""
    return f"{item.id} {item.status}: {item.content}"


def _render_line(item: TodoItem) -> str:
    line = f"{_MARKERS[item.status]} {item.id} {item.content}"
    if item.status == "waiting":
        line += f'  waiting: "{item.question}"'
    if item.requires_tool and item.status in OPEN_STATUSES:
        line += f"  needs: {item.requires_tool}"
    if item.note:
        line += f'  note: "{item.note}"'
    return line


def _one_line(value: Any) -> str:
    """Text as one trimmed line, so it cannot break the rendered list."""
    return "" if value is None else " ".join(str(value).split())


def _join(words: list[str]) -> str:
    """Joins words as "a", "a and b" or "a, b and c"."""
    if len(words) < 2:
        return "".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"
