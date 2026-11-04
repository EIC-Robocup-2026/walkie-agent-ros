"""The todo list as create_agent middleware (decision 5; the rules are ADR-0041's).

In a task with several steps, an LLM tends to drop a step, to call the task done
once it has spoken, or to report success after a tool failed. The rules are
enforced here rather than in the prompt, so the model cannot talk its way past them:

- at most one step is in progress or waiting, because physical work is serial;
- a failed or cancelled step carries a note that says why;
- seeded steps keep their wording and their order, and cannot be removed;
- a step cannot complete while a failure in its window is unresolved;
- a step with requires_tool needs a successful result of that tool, and each
  result completes one step only, so one drive cannot finish two steps.

All of it lives in the graph state, so a checkpoint holds it. `todos` is a plain
channel that only update_todos and the run boundaries write, and langgraph refuses
two writes to one in a superstep. So tools never write it: a question sets
`question`, and each executed call adds one entry to the additive `ledger`.
"""

import copy
import logging
import operator
from collections.abc import Callable, Iterable
from typing import Annotated, Any, Literal, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from walkie_agent.capabilities import CancelToken
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.runtime.middleware import (
    ToolHandler,
    ToolResult,
    owning_message,
    tool_result,
)

logger = logging.getLogger(__name__)

MAX_GATES = 2  # how often one task may be sent back for stopping with open steps
OPEN_STATUSES = ("pending", "in_progress", "waiting")
OP_STATUSES = ("pending", "in_progress", "completed", "failed", "cancelled")
TODO_TOOL_COUNT = 2  # update_todos and wait_for_person, which this middleware supplies
STOPPED_NOTE = "Stopped by the person."

_MARKERS = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "waiting": "[?]",
    "completed": "[x]",
    "failed": "[!]",
    "cancelled": "[-]",
}
_SEED_KEYS = frozenset({"content", "active_form", "requires_tool"})
# Ledger positions, which never reach the LLM or the published list. `added` and
# `since` open a step's two windows, and `used` holds the results it consumed.
_PRIVATE_KEYS = ("added", "since", "used")


class TodoRejected(ValueError):
    """A change to the list broke a rule. The message tells the LLM what to do instead."""


# TodoOp becomes the update_todos schema. Every op has the same flat shape, because
# a union would become anyOf, and so would Optional fields (anyOf with null), which
# Gemini function declarations and small Ollama models handle poorly. So every field
# is a plain string where empty means "not given", and the code checks the rest.
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

    # Some models fill the fields they do not use with null. Reading null as "not
    # given" spares them a validation error they may not recover from.
    @field_validator("id", "content", "active_form", "status", "note", "after", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        return _one_line(value) if value is None or isinstance(value, str) else value


class TodoState(AgentState):
    """What WalkieTodos adds to the agent state. Items are plain dicts, for the checkpoint."""

    todos: NotRequired[list[dict[str, Any]]]
    todo_seed: NotRequired[list[dict[str, Any]] | None]  # input only; [] is a seed, None is not
    stopping: NotRequired[bool]  # input only; the Runner's bare-stop pass
    next_id: NotRequired[int]
    gates: NotRequired[int]
    ledger: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    question: NotRequired[str | None]  # asked in this run, or waiting for the next one


def parse_seed(
    items: Iterable[str | dict[str, Any]], source: Literal["seed", "operator"] = "seed"
) -> list[dict[str, Any]]:
    """Check a seed and return new locked items, with ids from t1. Raises at submit time.

    Each entry is its content as a string, or a dict with content and optionally
    active_form and requires_tool. Duplicate wording is fine, since ids keep the
    steps apart.
    """
    if source not in ("seed", "operator"):
        raise ValueError(f"A seed's source must be 'seed' or 'operator', not {source!r}.")
    # A string is iterable too, and would seed one item per character.
    if isinstance(items, str):
        raise TodoRejected("A seed is a list of items, not a single string.")
    return [_seed_item(f"t{n}", entry, source) for n, entry in enumerate(items, start=1)]


def render(items: list[dict[str, Any]]) -> str:
    """The reminder sent to the LLM after the history; "" for an empty list."""
    if not items:
        return ""
    return "\n".join(["<todos>", *map(_render_line, items), "</todos>"])


def console_line(item: dict[str, Any]) -> str:
    """The operator console's line for a step whose status just changed; "" for pending."""
    if item["status"] == "in_progress":
        return f"{item['id']}: {item['active_form'] or item['content']}"
    if item["status"] == "pending":
        return ""
    return f"{item['id']} {item['status']}: {item['content']}"


def public(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of the items without the private ledger positions, ready for JSON."""
    return [
        {key: copy.copy(value) for key, value in item.items() if key not in _PRIVATE_KEYS}
        for item in items
    ]


class _Draft:
    """A working copy of the list that one update_todos call edits, all or nothing."""

    def __init__(
        self, items: list[dict[str, Any]], next_id: int, ledger: list[dict[str, Any]]
    ) -> None:
        self.items = copy.deepcopy(items)
        self.next_id = next_id
        self.ledger = ledger  # read only
        self.added: list[str] = []
        self.updated: dict[str, str] = {}  # first-update order; "" when the status is unchanged
        self.removed: list[str] = []

    def apply(self, ops: list[TodoOp]) -> str:
        """Apply `ops` and return one sentence that sums up the change."""
        if not ops:
            raise TodoRejected("The call has no ops; send at least one add, update or remove.")
        locked_order = _locked_ids(self.items)
        for op in ops:
            {"add": self._add, "update": self._update, "remove": self._remove}[op.op](op)
        if _locked_ids(self.items) != locked_order:
            raise TodoRejected("Seeded steps keep their order, so one cannot move past another.")
        active = [i["id"] for i in self.items if i["status"] in ("in_progress", "waiting")]
        if len(active) > 1:
            raise TodoRejected(
                f"Only one item can be in progress or waiting at a time, but {_join(active)} "
                "would be; finish the others or set them back to pending."
            )
        return self._summary()

    def _summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f"Added: {', '.join(self.added)}.")
        if self.updated:
            updates = [f"{i} → {s}" if s else i for i, s in self.updated.items()]
            parts.append(f"Updated: {', '.join(updates)}.")
        if self.removed:
            parts.append(f"Removed: {', '.join(self.removed)}.")
        open_ids = [i["id"] for i in self.items if i["status"] in OPEN_STATUSES]
        parts.append(f"Open: {', '.join(open_ids) or 'none'}.")
        return " ".join(parts)

    def _add(self, op: TodoOp) -> None:
        if op.id:
            raise TodoRejected(
                "Ids are assigned by the list, so leave id empty when you add an item."
            )
        if not op.content:
            raise TodoRejected("An add op needs content that says what the step is.")
        item_id, status = f"t{self.next_id}", op.status.lower() or "pending"
        _check_status(item_id, status, op.note)
        item = _new_item(item_id, op.content, op.active_form, status=status, note=op.note)
        item["added"] = item["since"] = len(self.ledger)
        index = len(self.items)
        if op.after:
            missing = f"There is no item {op.after} to add the new item after."
            index = self._index(op.after, missing) + 1
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
        before = copy.deepcopy(item)

        rewords = (op.content and op.content != item["content"]) or (
            op.active_form and op.active_form != item["active_form"]
        )
        if item["locked"] and rewords:
            raise TodoRejected(f"{item['id']} is a seeded step, so its wording cannot change.")
        if op.content:
            # An active form that was only defaulted from the content follows it.
            if not op.active_form and item["active_form"] == item["content"]:
                item["active_form"] = op.content
            item["content"] = op.content
        item["active_form"] = op.active_form or item["active_form"]

        status = op.status.lower()
        if status:
            _check_status(item["id"], status, op.note)
            if status == "completed" and item["status"] != "completed":
                self._check_completion(item, op.note)
            # An op that sets in_progress starts a new attempt, with a clean slate.
            if status == "in_progress" and item["status"] != "in_progress":
                item["since"] = len(self.ledger)
            item["question"], item["status"] = "", status
        item["note"] = op.note or item["note"]

        moved = False
        if op.after:
            if op.after == item["id"]:
                raise TodoRejected(f"{item['id']} cannot be moved after itself.")
            self._index(op.after, f"There is no item {op.after} to move {item['id']} after.")
            self.items.pop(index)
            target = self._index(op.after) + 1
            self.items.insert(target, item)
            moved = target != index

        if item == before and not moved:
            raise TodoRejected(
                f"The update to {item['id']} changes nothing; leave it out or change a field."
            )
        if item["status"] != before["status"]:
            self.updated[item["id"]] = item["status"]
        else:
            self.updated.setdefault(item["id"], "")

    def _remove(self, op: TodoOp) -> None:
        if not op.id:
            raise TodoRejected("A remove op needs the id of the item to remove.")
        index = self._index(op.id)
        if self.items[index]["locked"]:
            raise TodoRejected(
                f"{op.id} is a seeded step, so it cannot be removed; "
                "mark it cancelled with a note instead."
            )
        del self.items[index]
        self.removed.append(op.id)

    def _index(self, item_id: str, missing: str = "") -> int:
        for index, item in enumerate(self.items):
            if item["id"] == item_id:
                return index
        raise TodoRejected(missing or f"There is no item {item_id}; use an id from the todo list.")

    def _check_completion(self, item: dict[str, Any], note: str) -> None:
        """The two ledger checks before a step completes. A note can only excuse a failure."""
        need = item["requires_tool"]
        if need:
            used = {position for other in self.items for position in other["used"]}
            for position in range(item["added"], len(self.ledger)):
                entry = self.ledger[position]
                if entry["ok"] and entry["tool"] == need and position not in used:
                    item["used"].append(position)
                    item["evidence"].append(entry["id"])
                    break
            else:
                raise TodoRejected(
                    f"{item['id']} needs a successful {need} before it can be completed."
                )

        # A later success of the same tool resolves a failure. Walking the window
        # backwards collects the tools that succeeded later on.
        succeeded_later: set[str] = set()
        unresolved: list[str] = []
        for entry in reversed(self.ledger[item["since"] :]):
            if entry["ok"]:
                succeeded_later.add(entry["tool"])
            elif entry["tool"] not in succeeded_later and entry["tool"] not in unresolved:
                unresolved.append(entry["tool"])
        if unresolved and not note:
            unresolved.reverse()
            verb = "has" if len(unresolved) == 1 else "have"
            raise TodoRejected(
                f"{item['id']} cannot be completed because {_join(unresolved)} failed and "
                f"{verb} not succeeded since; retry it, or add a note that explains "
                "why the step is done anyway."
            )


class WalkieTodos(AgentMiddleware):
    """Keeps the list in state, reminds the LLM of it and supplies its two tools.

    `cancel` is the stop token, which the two tools check before they do anything.
    `say` speaks a question and raises StepError on failure. `on_todos(items, changed)`
    hears every change, as public copies; `changed` holds the items whose status
    changed. It runs on the runtime's thread, so it must not block, and if it raises
    the failure is logged and the list is unaffected.
    """

    state_schema = TodoState

    def __init__(
        self,
        cancel: CancelToken,
        say: Callable[[str], None],
        on_todos: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_todos = on_todos
        # Results of the running AI message, which reach state only when its
        # superstep ends. [go_to, update_todos] in one message must still see the
        # drive. Calls run one at a time (max_concurrency=1), so no lock is needed.
        self._batch_owner: str | None = None
        self._batch: list[dict[str, Any]] = []

        # Async so that it runs on the loop, the thread on_todos is promised. A sync
        # tool would run in an executor thread under ainvoke.
        async def update_todos(
            ops: Annotated[
                list[TodoOp],
                Field(
                    description="The changes to make, applied in order. "
                    "Either all of them apply, or none."
                ),
            ],
            runtime: ToolRuntime,
        ) -> Command:
            """Edit your todo list for the current task, referring to items by id.

            Ops: add (content, optional status and after), update (id plus any of
            content, active_form, status, note, after), remove (id). Statuses: pending,
            in_progress, completed, failed, cancelled. failed and cancelled need a note.
            Only one item can be in progress at a time. Seeded steps cannot be reworded
            or removed. Put all changes in one call.
            """
            cancel.check("update_todos")
            old = runtime.state.get("todos") or []
            draft = _Draft(old, runtime.state.get("next_id", 1), self._ledger(runtime.state))
            try:
                summary = draft.apply(ops)
            except TodoRejected as e:
                raise StepError(
                    ErrorCode.REFUSED, f"{e} None of the changes were applied."
                ) from e
            self._publish(old, draft.items)
            answer = ToolMessage(summary, name="update_todos", tool_call_id=runtime.tool_call_id)
            return Command(
                update={"todos": draft.items, "next_id": draft.next_id, "messages": [answer]}
            )

        def wait_for_person(
            question: Annotated[
                str,
                Field(
                    description="The question to ask out loud, in plain words, "
                    "for example 'Do you mean the red mug or the blue one?'"
                ),
            ],
            runtime: ToolRuntime,
        ) -> Command:
            """Ask the person a question out loud and wait for the answer.

            Use this instead of say when you cannot go on without an answer, for
            example to learn which object they mean. Your turn ends, and the answer
            arrives as the next message.
            """
            cancel.check("wait_for_person")
            question = _one_line(question)
            say(question)
            answer = ToolMessage(
                f"Asked: {question}", name="wait_for_person", tool_call_id=runtime.tool_call_id
            )
            return Command(update={"question": question, "messages": [answer]})

        self.tools = [
            StructuredTool.from_function(coroutine=update_todos),
            StructuredTool.from_function(wait_for_person),
        ]

    # -- Run boundary ----------------------------------------------------------

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any]:
        todos, seed = state.get("todos") or [], state.get("todo_seed")
        update: dict[str, Any] = {"todo_seed": None, "stopping": False, "question": None}
        if state.get("stopping"):
            stopped = [_cancelled(i) if i["status"] in OPEN_STATUSES else i for i in todos]
            self._publish(todos, stopped)
            return {**update, "todos": stopped, "jump_to": "end"}
        if seed is not None:
            start = len(state.get("ledger") or [])
            seeded = [{**item, "added": start, "since": start} for item in seed]
            self._publish([] if seeded else todos, seeded)
            return {**update, "todos": seeded, "next_id": len(seeded) + 1, "gates": 0}
        if state.get("question"):
            # The answer continues the same attempt: same window, same gate budget.
            resumed = [_resumed(item) if item["status"] == "waiting" else item for item in todos]
            self._publish(todos, resumed)
            return {**update, "todos": resumed}
        self._publish(todos, [])
        return {**update, "todos": [], "next_id": 1, "gates": 0}

    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Fold a question from the last tools step into the list, at one boundary."""
        question, todos = state.get("question"), state.get("todos") or []
        if not question or not any(item["status"] == "in_progress" for item in todos):
            return None
        marked = [
            {**item, "status": "waiting", "question": question}
            if item["status"] == "in_progress"
            else item
            for item in todos
        ]
        self._publish(todos, marked)
        return {"todos": marked}

    # -- Around the model ------------------------------------------------------

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Send the list after the history on every call, without saving it there."""
        reminder = render(request.state.get("todos") or [])
        if reminder:
            request = request.override(messages=[*request.messages, HumanMessage(reminder)])
        return await handler(request)

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """The gate: send back a model that stops talking with steps still open.

        A question never gets here, since it ends the run before the next model call.
        """
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        open_ids = [i["id"] for i in state.get("todos") or [] if i["status"] in OPEN_STATUSES]
        gates = state.get("gates", 0)
        if not open_ids or gates >= MAX_GATES:
            return None
        text = (
            f"[todo] Open items: {', '.join(open_ids)}. "
            "Finish them, or mark them failed or cancelled with a note."
        )
        return {"jump_to": "model", "gates": gates + 1, "messages": [HumanMessage(text)]}

    # -- Around the tools ------------------------------------------------------

    async def awrap_tool_call(self, request: Any, handler: ToolHandler) -> ToolResult:
        """Add each executed call to the ledger. SerialGuard's refusals never get here.

        An unknown tool never ran, and update_todos is a mistake in the list rather
        than a failed step. A call with bad arguments does count as a failure.
        """
        result = await handler(request)
        call = request.tool_call
        message = tool_result(result, call["id"])
        if request.tool is None or call["name"] == "update_todos" or message is None:
            return result
        entry = {"tool": call["name"], "id": call["id"], "ok": message.status != "error"}
        self._ledger(request.state)  # moves the batch on to this AI message
        self._batch.append(entry)
        if isinstance(result, Command):
            return Command(update={**result.update, "ledger": [entry]})
        return Command(update={"messages": [result], "ledger": [entry]})

    def _ledger(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """The ledger as of now: the state's, plus this AI message's results so far."""
        owner = owning_message(state).id
        if owner != self._batch_owner:
            self._batch_owner, self._batch = owner, []
        return [*(state.get("ledger") or []), *self._batch]

    def _publish(self, old: list[dict[str, Any]], new: list[dict[str, Any]]) -> None:
        if self._on_todos is None or old == new:
            return
        before = {item["id"]: item["status"] for item in old}
        changed = [item for item in new if before.get(item["id"]) != item["status"]]
        try:
            self._on_todos(public(new), public(changed))
        except Exception:
            logger.exception("A todo listener failed; the list is unaffected.")


def _new_item(item_id: str, content: str, active_form: str = "", **fields: Any) -> dict[str, Any]:
    return {
        "id": item_id,
        "content": content,
        "active_form": active_form or content,
        "status": "pending",
        "note": "",
        "source": "agent",
        "locked": False,
        "requires_tool": "",
        "evidence": [],
        "question": "",
        "used": [],
        **fields,
    }


def _seed_item(item_id: str, entry: str | dict[str, Any], source: str) -> dict[str, Any]:
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
    return _new_item(
        item_id,
        content,
        _one_line(entry.get("active_form")),
        source=source,
        locked=True,
        requires_tool=_one_line(entry.get("requires_tool")),
    )


def _check_status(item_id: str, status: str, note: str) -> None:
    if status == "waiting":
        raise TodoRejected(
            "Only wait_for_person can set waiting; call it to ask the person a question."
        )
    if status not in OP_STATUSES:
        raise TodoRejected(
            f"Status '{status}' is not allowed; use one of {', '.join(OP_STATUSES)}."
        )
    if status in ("failed", "cancelled") and not note:
        raise TodoRejected(f"Marking {item_id} {status} needs a note that says why.")


def _cancelled(item: dict[str, Any]) -> dict[str, Any]:
    return {**item, "status": "cancelled", "note": STOPPED_NOTE, "question": ""}


def _resumed(item: dict[str, Any]) -> dict[str, Any]:
    return {**item, "status": "in_progress", "question": ""}


def _locked_ids(items: list[dict[str, Any]]) -> list[str]:
    return [item["id"] for item in items if item["locked"]]


def _render_line(item: dict[str, Any]) -> str:
    line = f"{_MARKERS[item['status']]} {item['id']} {item['content']}"
    if item["status"] == "waiting":
        line += f'  waiting: "{item["question"]}"'
    if item["requires_tool"] and item["status"] in OPEN_STATUSES:
        line += f"  needs: {item['requires_tool']}"
    if item["note"]:
        line += f'  note: "{item["note"]}"'
    return line


def _one_line(value: Any) -> str:
    """Text as one trimmed line, so it cannot break the rendered list."""
    return "" if value is None else " ".join(str(value).split())


def _join(words: list[str]) -> str:
    """Joins words as "a", "a and b" or "a, b and c"."""
    if len(words) < 2:
        return "".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"
