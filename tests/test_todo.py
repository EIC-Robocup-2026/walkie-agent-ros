"""The todo list (walkie_agent.todo) and its tools, update_todos and wait_for_person.

None of these tests need ROS: the list is plain Python, and wait_for_person is
given a fake speak tool.
"""

import json
import threading

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from rai.agents.langchain.core.tool_runner import ToolRunner

from walkie_agent.todo import TodoItem, TodoList, TodoOp, TodoRejected, console_line
from walkie_agent.tools.todos import UpdateTodosTool, WaitForPersonTool

# The task from ADR-0041.
MUG_TASK = [
    "Go to the kitchen",
    "Pick up the red mug",
    "Come back to the operator",
    {"content": "Hand over the mug", "requires_tool": "hand_over"},
]


def add(content: str, **fields: str) -> dict:
    return {"op": "add", "content": content, **fields}


def update(item_id: str, **fields: str) -> dict:
    return {"op": "update", "id": item_id, **fields}


def remove(item_id: str) -> dict:
    return {"op": "remove", "id": item_id}


def seeded(items=MUG_TASK) -> TodoList:
    todos = TodoList()
    todos.seed(items)
    return todos


def item(todos: TodoList, item_id: str) -> TodoItem:
    return next(i for i in todos.snapshot() if i.id == item_id)


def statuses(todos: TodoList) -> dict[str, str]:
    return {i.id: i.status for i in todos.snapshot()}


def rejected(todos: TodoList, ops: list) -> str:
    """Applies ops that must be rejected, and returns the reason."""
    with pytest.raises(TodoRejected) as info:
        todos.apply(ops)
    return str(info.value)


def fake_speak(reply: str = ""):
    """A speak tool that records what it was asked to say."""
    said: list[str] = []

    @tool
    def speak(text: str) -> str:
        """Say something out loud."""
        said.append(text)
        return reply or f"Said: {text}"

    return speak, said


# -- Seeding --------------------------------------------------------------------


def test_seed_numbers_and_locks_items():
    todos = seeded(MUG_TASK + [{"content": "Say goodbye", "active_form": "Saying goodbye"}])

    items = todos.snapshot()
    assert [i.id for i in items] == ["t1", "t2", "t3", "t4", "t5"]
    assert all(i.locked and i.source == "seed" and i.status == "pending" for i in items)
    assert items[0].active_form == "Go to the kitchen"
    assert items[3].requires_tool == "hand_over"
    assert items[4].active_form == "Saying goodbye"


def test_seed_from_the_operator():
    todos = TodoList()
    todos.seed(["Take the order"], source="operator")
    assert item(todos, "t1").source == "operator"
    assert item(todos, "t1").locked


def test_seed_replaces_the_list_the_ids_and_the_ledger():
    todos = seeded()
    todos.apply([add("Wipe the table")])
    todos.record_tool_result("navigate_to_location", "call_1", ok=False)

    todos.seed(["Go to the door", "Open the door"])

    assert [i.id for i in todos.snapshot()] == ["t1", "t2"]
    assert todos.apply([add("Greet the guest")]) == "Added: t3. Open: t1, t2, t3."
    # The failure belonged to the old task, so it no longer blocks completion.
    todos.apply([update("t1", status="completed")])


def test_seed_rejects_an_empty_item_and_keeps_the_old_list():
    todos = seeded()
    with pytest.raises(TodoRejected, match="t2 has no content"):
        todos.seed(["Go to the door", "   "])
    with pytest.raises(TodoRejected, match="unknown keys: require_tool"):
        todos.seed([{"content": "Hand over", "require_tool": "hand_over"}])
    with pytest.raises(TodoRejected, match="not a single string"):
        todos.seed("Wave")
    assert len(todos.snapshot()) == 4


def test_clear_resets_ids_and_ledger():
    todos = TodoList()
    todos.apply([add("Go to the kitchen"), add("Pick up the mug")])
    todos.apply([update("t1", status="in_progress")])
    todos.record_tool_result("navigate_to_location", "call_1", ok=False)

    todos.clear()

    assert todos.is_empty
    assert todos.apply([add("Go to the door")]) == "Added: t1. Open: t1."
    todos.apply([update("t1", status="completed")])
    # A new window starts at the current ledger length, so an old entry would
    # not show through the public API; check the reset itself.
    assert todos._ledger == []


# -- Adding, updating and removing ------------------------------------------------


def test_add_assigns_ids_and_appends():
    todos = TodoList()
    todos.apply([add("Go to the kitchen"), add("Look for the mug", status="in_progress")])

    first, second = todos.snapshot()
    assert (first.id, second.id) == ("t1", "t2")
    assert (first.source, first.locked) == ("agent", False)
    assert first.active_form == "Go to the kitchen"
    assert second.status == "in_progress"


def test_add_after_an_item():
    todos = seeded()
    todos.apply([add("Open the cupboard", after="t1")])
    assert [i.id for i in todos.snapshot()] == ["t1", "t5", "t2", "t3", "t4"]


def test_ids_are_not_reused_after_a_remove():
    todos = TodoList()
    todos.apply([add("One"), add("Two")])
    todos.apply([remove("t2")])
    assert todos.apply([add("Three")]) == "Added: t3. Open: t1, t3."


def test_add_rejects_a_given_id_and_a_missing_content():
    todos = TodoList()
    assert "assigned" in rejected(todos, [{"op": "add", "id": "t7", "content": "Go"}])
    assert "needs content" in rejected(todos, [{"op": "add", "content": "  "}])


@pytest.mark.parametrize("op", [update("t9", status="completed"), remove("t9")])
def test_unknown_id_is_rejected(op):
    assert rejected(seeded(), [op]) == "There is no item t9; use an id from the todo list."


def test_an_op_can_refer_to_an_item_added_earlier_in_the_call():
    todos = TodoList()
    todos.apply([add("Go to the kitchen"), update("t1", status="in_progress")])
    assert statuses(todos) == {"t1": "in_progress"}


def test_update_moves_an_item_after_another():
    todos = seeded()
    todos.apply([add("Wipe the table"), update("t5", after="t2")])
    assert [i.id for i in todos.snapshot()] == ["t1", "t2", "t5", "t3", "t4"]
    assert "after itself" in rejected(todos, [update("t5", after="t5")])


def test_after_an_unknown_id_is_rejected():
    todos = seeded()
    assert rejected(todos, [add("Open the cupboard", after="t9")]) == (
        "There is no item t9 to add the new item after."
    )
    assert rejected(todos, [update("t1", after="t9")]) == "There is no item t9 to move t1 after."
    assert [i.id for i in todos.snapshot()] == ["t1", "t2", "t3", "t4"]


def test_update_that_changes_nothing_is_rejected():
    todos = seeded()
    todos.apply([update("t1", status="in_progress")])
    assert "changes nothing" in rejected(todos, [update("t1", status="in_progress")])
    assert "changes nothing" in rejected(todos, [update("t2", after="t1")])


def test_rewording_an_agent_item_moves_its_default_active_form():
    todos = TodoList()
    todos.apply([add("Go to the kitchen")])
    todos.apply([update("t1", content="Go to the living room")])
    assert item(todos, "t1").active_form == "Go to the living room"


# -- Locked items -------------------------------------------------------------------


def test_seeded_wording_cannot_change():
    todos = seeded()
    message = "t1 is a seeded step, so its wording cannot change."
    assert rejected(todos, [update("t1", content="Go to the bedroom")]) == message
    assert rejected(todos, [update("t1", active_form="Heading off")]) == message


def test_seeded_item_cannot_be_removed():
    assert "mark it cancelled with a note" in rejected(seeded(), [remove("t2")])


def test_seeded_item_can_change_status_and_note():
    todos = seeded()
    # Repeating the same wording is not a change, so it is allowed.
    todos.apply([update("t2", content="Pick up the red mug", status="cancelled",
                        note="There is no red mug.")])
    todos.apply([update("t2", note="The guest took it.")])
    assert item(todos, "t2").status == "cancelled"
    assert item(todos, "t2").note == "The guest took it."


# -- Statuses -----------------------------------------------------------------------


# The status rules hold for both ops that can set a status. On the mug task an
# added item becomes t5.
STATUS_OPS = pytest.mark.parametrize(
    ("make_op", "item_id"),
    [
        (lambda **fields: add("Wave", **fields), "t5"),
        (lambda **fields: update("t1", **fields), "t1"),
    ],
    ids=["add", "update"],
)


@STATUS_OPS
def test_waiting_cannot_be_set_by_an_op(make_op, item_id):
    todos = seeded()
    message = rejected(todos, [make_op(status="waiting")])
    assert "Only wait_for_person can set waiting" in message
    assert not todos.has_waiting()


@STATUS_OPS
def test_unknown_status_is_rejected(make_op, item_id):
    message = rejected(seeded(), [make_op(status="done")])
    assert message == (
        "Status 'done' is not allowed; use one of pending, in_progress, completed, failed, "
        "cancelled."
    )


@STATUS_OPS
@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_and_cancelled_need_a_note(make_op, item_id, status):
    assert rejected(seeded(), [make_op(status=status)]) == (
        f"Marking {item_id} {status} needs a note that says why."
    )
    assert rejected(seeded(), [make_op(status=status, note="  ")])


def test_only_one_item_is_active():
    todos = seeded()
    todos.apply([update("t1", status="in_progress")])
    message = rejected(todos, [update("t2", status="in_progress")])
    assert "Only one item can be in progress or waiting" in message
    assert "t1 and t2" in message

    # A waiting item counts as active too.
    todos.set_waiting("Which kitchen?")
    assert "t1 and t2" in rejected(todos, [update("t2", status="in_progress")])


def test_one_active_item_is_checked_after_all_ops():
    todos = seeded()
    todos.apply([update("t1", status="in_progress")])
    # Finishing t1 and starting t2 in one call is fine.
    todos.apply([update("t1", status="completed"), update("t2", status="in_progress")])
    assert statuses(todos)["t2"] == "in_progress"
    # Starting two items in one call is not.
    message = rejected(todos, [update("t2", status="pending"), update("t3", status="in_progress"),
                               update("t4", status="in_progress")])
    assert "t3 and t4" in message


def test_rejected_call_changes_nothing():
    todos = seeded()
    calls = []
    todos.add_listener(lambda items, changed: calls.append(items))
    before = todos.snapshot()

    rejected(todos, [update("t1", status="in_progress"), add("Wipe the table"), remove("t9")])

    assert todos.snapshot() == before
    assert calls == []
    # The id counter did not move either.
    assert todos.apply([add("Wipe the table")]).startswith("Added: t5.")


def test_rejected_call_keeps_the_ledger_window():
    todos = seeded()
    todos.apply([update("t1", status="in_progress")])
    todos.record_tool_result("navigate_to_location", "call_1", ok=False)

    # Going back to pending and in progress again would start a new window,
    # but the call is rejected, so the window must stay where it was.
    rejected(todos, [update("t1", status="pending"), update("t1", status="in_progress"),
                     remove("t9")])

    message = rejected(todos, [update("t1", status="completed")])
    assert "navigate_to_location failed" in message


def test_empty_op_list_is_rejected():
    assert "no ops" in rejected(seeded(), [])


def test_summary_sentence():
    todos = seeded()
    assert todos.apply([update("t1", status="in_progress")]) == (
        "Updated: t1 → in_progress. Open: t1, t2, t3, t4."
    )
    assert todos.apply([add("Wipe the table"), add("Close the door")]) == (
        "Added: t5, t6. Open: t1, t2, t3, t4, t5, t6."
    )
    # The parts come in a fixed order, whatever the order of the ops.
    summary = todos.apply([
        update("t1", status="completed"),
        update("t2", status="in_progress"),
        update("t3", note="Use the side door."),
        remove("t6"),
        add("Close the window"),
    ])
    assert summary == (
        "Added: t7. Updated: t1 → completed, t2 → in_progress, t3. Removed: t6. "
        "Open: t2, t3, t4, t5, t7."
    )


def test_summary_says_none_when_nothing_is_open():
    todos = TodoList()
    todos.apply([add("Wave")])
    assert todos.apply([update("t1", status="completed")]) == (
        "Updated: t1 → completed. Open: none."
    )


# -- The error ledger --------------------------------------------------------------


def start_hand_over() -> TodoList:
    """The mug task with t1 to t3 done and t4, which needs hand_over, in progress."""
    todos = seeded()
    todos.apply([update(i, status="completed") for i in ("t1", "t2", "t3")])
    todos.apply([update("t4", status="in_progress")])
    return todos


def test_requires_tool_without_a_success_is_refused():
    todos = start_hand_over()
    message = "t4 needs a successful hand_over before it can be completed."
    assert rejected(todos, [update("t4", status="completed")]) == message

    todos.record_tool_result("hand_over", "call_1", ok=False)
    assert rejected(todos, [update("t4", status="completed")]) == message
    # A note explains a failure, but cannot replace the required tool.
    note = "The guest took the mug from the tray."
    assert rejected(todos, [update("t4", status="completed", note=note)]) == message


def test_requires_tool_success_completes_the_item():
    todos = start_hand_over()
    todos.record_tool_result("hand_over", "call_1", ok=True)
    todos.apply([update("t4", status="completed")])
    assert item(todos, "t4").status == "completed"


def test_requires_tool_success_before_the_window_does_not_count():
    todos = seeded()
    todos.record_tool_result("hand_over", "call_1", ok=True)
    # Going in progress starts t4's window after that success.
    todos.apply([update("t4", status="in_progress")])
    assert "needs a successful hand_over" in rejected(todos, [update("t4", status="completed")])


def test_unresolved_failure_needs_a_retry_or_a_note():
    todos = seeded()
    todos.apply([update("t1", status="in_progress")])
    todos.record_tool_result("navigate_to_location", "call_1", ok=False)

    message = rejected(todos, [update("t1", status="completed")])
    assert message == (
        "t1 cannot be completed because navigate_to_location failed and has not succeeded "
        "since; retry it, or add a note that explains why the step is done anyway."
    )
    todos.apply([update("t1", status="completed", note="The person brought me there.")])
    assert item(todos, "t1").status == "completed"


def test_later_success_of_the_same_tool_resolves_a_failure():
    todos = seeded()
    todos.apply([update("t1", status="in_progress")])
    todos.record_tool_result("navigate_to_location", "call_1", ok=False)
    todos.record_tool_result("speak", "call_2", ok=True)
    todos.record_tool_result("navigate_to_location", "call_3", ok=True)
    todos.apply([update("t1", status="completed")])


def test_success_of_another_tool_does_not_resolve_a_failure():
    todos = seeded()
    todos.apply([update("t1", status="in_progress")])
    todos.record_tool_result("navigate_to_location", "call_1", ok=False)
    todos.record_tool_result("speak", "call_2", ok=True)
    message = rejected(todos, [update("t1", status="completed")])
    assert "navigate_to_location failed" in message


def test_success_before_a_failure_does_not_resolve_it():
    todos = seeded()
    todos.apply([update("t1", status="in_progress")])
    todos.record_tool_result("navigate_to_location", "call_1", ok=True)
    todos.record_tool_result("navigate_to_location", "call_2", ok=False)
    message = rejected(todos, [update("t1", status="completed")])
    assert "navigate_to_location failed" in message


def test_failure_before_a_new_attempt_does_not_count():
    todos = seeded()
    todos.record_tool_result("navigate_to_location", "call_1", ok=False)
    todos.apply([update("t1", status="in_progress")])
    todos.apply([update("t1", status="completed")])


def test_failure_before_an_item_is_added_does_not_count():
    todos = TodoList()
    todos.record_tool_result("navigate_to_location", "call_1", ok=False)
    todos.apply([add("Wave")])
    # Completed straight from pending, so only the window from creation applies.
    todos.apply([update("t1", status="completed")])
    assert item(todos, "t1").status == "completed"


def test_waiting_keeps_the_window():
    todos = seeded()
    todos.apply([update("t2", status="in_progress")])
    todos.record_tool_result("pick", "call_1", ok=False)
    todos.set_waiting("The left one or the right one?")
    todos.resume_waiting()
    assert "pick failed" in rejected(todos, [update("t2", status="completed")])


def test_an_op_from_waiting_to_in_progress_starts_a_new_window():
    todos = seeded()
    todos.apply([update("t2", status="in_progress")])
    todos.record_tool_result("pick", "call_1", ok=False)
    todos.set_waiting("The left one or the right one?")

    # Unlike resume_waiting, an op that sets in_progress starts a new attempt.
    todos.apply([update("t2", status="in_progress")])
    assert item(todos, "t2").question == ""
    todos.apply([update("t2", status="completed")])
    assert item(todos, "t2").status == "completed"


def test_ledger_ignores_update_todos():
    todos = seeded()
    todos.apply([update("t1", status="in_progress")])
    todos.record_tool_result("update_todos", "call_1", ok=False)
    todos.record_tool_result("update_todos", "call_2", ok=True)
    todos.apply([update("t1", status="completed")])
    assert item(todos, "t1").evidence == []


def test_evidence_collects_successful_calls_of_the_active_item():
    todos = seeded()
    todos.record_tool_result("list_locations", "call_0", ok=True)  # nothing active yet
    todos.apply([update("t1", status="in_progress")])
    todos.record_tool_result("navigate_to_location", "call_1", ok=True)
    todos.record_tool_result("speak", "call_2", ok=False)
    todos.set_waiting("Is this the kitchen?")
    todos.record_tool_result("speak", "call_3", ok=True)

    assert item(todos, "t1").evidence == ["call_1", "call_3"]
    assert item(todos, "t2").evidence == []


# -- Waiting ------------------------------------------------------------------------


def test_set_waiting_and_resume():
    todos = seeded()
    todos.apply([update("t2", status="in_progress")])

    assert todos.set_waiting("The left one or the right one?") == "t2"
    assert item(todos, "t2").status == "waiting"
    assert item(todos, "t2").question == "The left one or the right one?"
    assert todos.has_waiting()
    assert todos.open_ids() == ["t1", "t2", "t3", "t4"]

    todos.resume_waiting()
    assert item(todos, "t2").status == "in_progress"
    assert item(todos, "t2").question == ""
    assert not todos.has_waiting()


def test_set_waiting_without_an_item_in_progress():
    todos = seeded()
    assert todos.set_waiting("Anything else?") is None
    assert not todos.has_waiting()


def test_open_ids_leave_out_finished_items():
    todos = seeded()
    todos.apply([
        update("t1", status="completed"),
        update("t2", status="cancelled", note="There is no red mug."),
        update("t3", status="in_progress"),
    ])
    todos.set_waiting("Where are you?")
    assert todos.open_ids() == ["t3", "t4"]

    todos.resume_waiting()
    todos.apply([update("t3", status="failed", note="The operator left.")])
    assert todos.open_ids() == ["t4"]


# -- Rendering and JSON -----------------------------------------------------------


def test_render_matches_the_adr_example():
    todos = seeded()
    todos.apply([update("t1", status="completed")])
    todos.apply([update("t2", status="in_progress")])
    todos.set_waiting("The left one or the right one?")

    assert todos.render() == (
        "<todos>\n"
        "[x] t1 Go to the kitchen\n"
        '[?] t2 Pick up the red mug  waiting: "The left one or the right one?"\n'
        "[ ] t3 Come back to the operator\n"
        "[ ] t4 Hand over the mug  needs: hand_over\n"
        "</todos>"
    )


def test_render_other_markers_and_notes():
    todos = seeded()
    todos.apply([
        update("t1", status="failed", note="The kitchen door is closed."),
        update("t2", status="in_progress"),
        update("t4", status="cancelled", note="The operator left."),
    ])

    assert todos.render() == (
        "<todos>\n"
        '[!] t1 Go to the kitchen  note: "The kitchen door is closed."\n'
        "[>] t2 Pick up the red mug\n"
        "[ ] t3 Come back to the operator\n"
        # needs: is only shown while the item is open.
        '[-] t4 Hand over the mug  note: "The operator left."\n'
        "</todos>"
    )


def test_render_puts_suffixes_in_order_on_one_line():
    todos = TodoList()
    todos.seed([{"content": "Hand over the mug", "requires_tool": "hand_over"}])
    todos.apply([update("t1", status="in_progress", note="Second try,\nleft hand.")])
    todos.set_waiting("Ready?")
    assert todos.render().splitlines()[1] == (
        '[?] t1 Hand over the mug  waiting: "Ready?"  needs: hand_over  '
        'note: "Second try, left hand."'
    )


def test_render_and_json_of_an_empty_list():
    todos = TodoList()
    assert todos.render() == ""
    assert todos.to_json() == "[]"
    assert todos.is_empty


def test_to_json():
    todos = seeded()
    todos.apply([update("t1", status="in_progress")])

    data = json.loads(todos.to_json())
    assert [entry["id"] for entry in data] == ["t1", "t2", "t3", "t4"]
    assert data == [i.model_dump() for i in todos.snapshot()]
    assert data[0]["status"] == "in_progress"
    assert data[3]["requires_tool"] == "hand_over"


def test_snapshot_is_a_copy():
    todos = seeded()
    todos.snapshot()[0].evidence.append("call_x")
    assert item(todos, "t1").evidence == []


@pytest.mark.parametrize(
    ("status", "active_form", "line"),
    [
        ("in_progress", "Going to the kitchen", "t1: Going to the kitchen"),
        ("in_progress", "", "t1: Go to the kitchen"),
        ("completed", "Going to the kitchen", "t1 completed: Go to the kitchen"),
        ("failed", "Going to the kitchen", "t1 failed: Go to the kitchen"),
        ("cancelled", "Going to the kitchen", "t1 cancelled: Go to the kitchen"),
        ("waiting", "Going to the kitchen", "t1 waiting: Go to the kitchen"),
        ("pending", "Going to the kitchen", ""),
    ],
)
def test_console_line(status, active_form, line):
    entry = TodoItem(id="t1", content="Go to the kitchen", active_form=active_form, status=status)
    assert console_line(entry) == line


# -- Listeners ----------------------------------------------------------------------


def test_listener_gets_the_list_and_the_changed_items():
    todos = TodoList()
    calls: list[tuple[list[str], list[str]]] = []
    todos.add_listener(
        lambda items, changed: calls.append(([i.id for i in items], [i.id for i in changed]))
    )

    todos.seed(["Go to the kitchen", "Pick up the mug"])
    todos.apply([update("t1", status="in_progress"), update("t2", note="The red one.")])
    todos.apply([add("Wipe the table")])
    todos.set_waiting("Which kitchen?")
    todos.resume_waiting()
    todos.clear()

    assert calls == [
        (["t1", "t2"], ["t1", "t2"]),
        (["t1", "t2"], ["t1"]),
        (["t1", "t2", "t3"], ["t3"]),
        (["t1", "t2", "t3"], ["t1"]),
        (["t1", "t2", "t3"], ["t1"]),
        ([], []),
    ]


def test_listener_hears_of_new_evidence_only():
    # /walkie/todos publishes from a listener, so new evidence must reach it,
    # or the latched list would lack it while an item waits.
    todos = TodoList()
    todos.seed(["Go to the kitchen"])
    calls: list[tuple[list[list[str]], list[str]]] = []
    todos.add_listener(
        lambda items, changed: calls.append(([i.evidence for i in items], [i.id for i in changed]))
    )

    todos.record_tool_result("navigate_to_location", "call_0", ok=True)  # nothing in progress
    todos.apply([update("t1", status="in_progress")])
    todos.record_tool_result("navigate_to_location", "call_1", ok=False)
    todos.record_tool_result("navigate_to_location", "call_2", ok=True)

    assert calls == [([[]], ["t1"]), ([["call_2"]], [])]


def test_raising_listener_does_not_break_the_list():
    todos = seeded()
    seen = []

    def broken(items, changed):
        raise RuntimeError("publisher is gone")

    todos.add_listener(broken)
    todos.add_listener(lambda items, changed: seen.append([i.status for i in changed]))

    assert todos.apply([update("t1", status="in_progress")]).startswith("Updated: t1")
    assert item(todos, "t1").status == "in_progress"
    assert seen == [["in_progress"]]


def test_no_listener_call_without_a_change():
    todos = TodoList()
    calls = []
    todos.add_listener(lambda items, changed: calls.append(items))

    todos.clear()
    todos.seed([])
    todos.resume_waiting()
    assert todos.set_waiting("Anything else?") is None
    assert calls == []


def test_listener_can_read_the_list_back():
    todos = seeded()
    seen: list[tuple[str, str]] = []
    todos.add_listener(lambda items, changed: seen.append((todos.to_json(), todos.render())))

    # Listeners run outside the lock. If one ran inside it, reading the list
    # back would block forever, so the change runs on a thread with a timeout.
    worker = threading.Thread(
        target=todos.apply, args=([update("t1", status="in_progress")],), daemon=True
    )
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert seen == [(todos.to_json(), todos.render())]


def test_listener_may_change_the_list():
    todos = seeded(["Go to the kitchen", "Pick up the mug"])
    first_seen: list[list[str]] = []
    second_seen: list[list[str]] = []

    def starts_t2(items, changed):
        first_seen.append([i.status for i in items])
        if items[1].status == "pending":
            todos.apply([update("t2", status="in_progress")])

    todos.add_listener(starts_t2)
    todos.add_listener(lambda items, changed: second_seen.append([i.status for i in items]))

    todos.apply([update("t1", status="cancelled", note="The kitchen is closed.")])

    # Every listener sees both changes in order, so the last snapshot each one
    # sees is the current list.
    expected = [["cancelled", "pending"], ["cancelled", "in_progress"]]
    assert first_seen == expected
    assert second_seen == expected
    assert [i.status for i in todos.snapshot()] == expected[-1]


def test_change_on_another_thread_does_not_wait_for_the_listeners():
    todos = seeded(["Go to the kitchen", "Pick up the mug"])
    seen: list[list[str]] = []
    worker_finished: list[bool] = []

    def listener(items, changed):
        seen.append([i.note for i in items])
        if len(seen) == 1:
            # Like a ROS callback that edits the list while the agent thread
            # is still calling the listeners, and a listener that waits for it.
            worker = threading.Thread(
                target=todos.apply, args=([update("t2", note="The red one.")],), daemon=True
            )
            worker.start()
            worker.join(timeout=5)
            worker_finished.append(not worker.is_alive())

    todos.add_listener(listener)
    todos.apply([update("t1", note="Use the side door.")])

    assert worker_finished == [True]
    assert seen == [["Use the side door.", ""], ["Use the side door.", "The red one."]]


def test_concurrent_adds_get_unique_ids():
    todos = TodoList()

    def worker():
        for _ in range(25):
            todos.apply([add("Step")])
            todos.record_tool_result("speak", "call", ok=True)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(int(i.id[1:]) for i in todos.snapshot()) == list(range(1, 201))


# -- The update_todos tool -------------------------------------------------------


def test_update_todos_returns_the_summary():
    todos = seeded()
    update_todos = UpdateTodosTool(todos=todos)
    result = update_todos.invoke({"ops": [update("t1", status="in_progress")]})
    assert result == "Updated: t1 → in_progress. Open: t1, t2, t3, t4."


def test_update_todos_accepts_ops_and_dicts():
    todos = TodoList()
    update_todos = UpdateTodosTool(todos=todos)
    result = update_todos._run([TodoOp(op="add", content="Go"), add("Come back")])
    assert result == "Added: t1, t2. Open: t1, t2."


def test_update_todos_raises_tool_exception_on_rejection():
    todos = seeded()
    update_todos = UpdateTodosTool(todos=todos)
    with pytest.raises(ToolException) as info:
        update_todos.invoke({"ops": [update("t1", status="in_progress"), remove("t2")]})
    assert str(info.value).startswith("t2 is a seeded step, so it cannot be removed")
    assert str(info.value).endswith("None of the changes were applied.")
    assert statuses(todos)["t1"] == "pending"


def test_update_todos_rejection_reaches_the_llm_as_an_error():
    """RAI's ToolRunner turns the ToolException into an error ToolMessage."""
    update_todos = UpdateTodosTool(todos=seeded())
    call = {"name": "update_todos", "args": {"ops": [remove("t1")]}, "id": "call_1"}
    state = {"messages": [AIMessage(content="", tool_calls=[call])]}

    ToolRunner([update_todos]).invoke(state)

    result = state["messages"][-1]
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.content.startswith("Failed to run tool. Error: t1 is a seeded step")


def test_update_todos_reads_null_fields_as_not_given():
    todos = seeded()
    update_todos = UpdateTodosTool(todos=todos)
    op = {"op": "update", "id": "t1", "status": "in_progress", "note": None, "after": None}
    assert update_todos.invoke({"ops": [op]}) == "Updated: t1 → in_progress. Open: t1, t2, t3, t4."


def test_update_todos_schema_has_no_any_of():
    schema = convert_to_openai_tool(UpdateTodosTool(todos=TodoList()))
    assert "anyOf" not in json.dumps(schema)
    op_schema = schema["function"]["parameters"]["properties"]["ops"]["items"]
    assert set(op_schema["properties"]) == {
        "op", "id", "content", "active_form", "status", "note", "after"
    }
    assert op_schema["required"] == ["op"]


# -- The wait_for_person tool ----------------------------------------------------


def test_wait_for_person_asks_and_waits():
    todos = seeded()
    todos.apply([update("t2", status="in_progress")])
    speak, said = fake_speak()
    wait_for_person = WaitForPersonTool(todos=todos, speak_tool=speak)

    result = wait_for_person.invoke({"question": "The left one or the right one?"})

    assert result == "Asked: The left one or the right one?"
    assert said == ["The left one or the right one?"]
    assert item(todos, "t2").status == "waiting"
    assert item(todos, "t2").question == "The left one or the right one?"


def test_wait_for_person_without_an_item_in_progress_still_asks():
    todos = seeded()
    speak, said = fake_speak()
    wait_for_person = WaitForPersonTool(todos=todos, speak_tool=speak)

    assert wait_for_person.invoke({"question": "Anything else?"}) == "Asked: Anything else?"
    assert said == ["Anything else?"]
    assert not todos.has_waiting()


def test_wait_for_person_fails_when_speech_fails():
    todos = seeded()
    todos.apply([update("t2", status="in_progress")])
    calls = []
    todos.add_listener(lambda items, changed: calls.append(items))
    speak, _ = fake_speak("Nothing was said: no speech node is listening on /to_human.")
    wait_for_person = WaitForPersonTool(todos=todos, speak_tool=speak)

    with pytest.raises(ToolException) as info:
        wait_for_person.invoke({"question": "Which mug?"})

    assert str(info.value) == (
        "Could not ask the question: Nothing was said: no speech node is listening on /to_human."
    )
    assert item(todos, "t2").status == "in_progress"
    assert calls == []
