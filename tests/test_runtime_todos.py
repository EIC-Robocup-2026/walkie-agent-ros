"""WalkieTodos through the real stack on ainvoke, with FakeRobot and a scripted model.

The R-numbers are the spike's 39 checks (/var/tmp/wa-todo-seed/spike-own/spike_own.py).
tests/test_runtime.py holds the map of what happened to each of them.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from runtime_harness import add, ai, call, drive, make_stack, remove, todo_call, update

from walkie_agent.capabilities import CancelToken
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.runtime.todos import (
    TodoOp,
    TodoRejected,
    WalkieTodos,
    _Draft,
    console_line,
    parse_seed,
    public,
    render,
)
from walkie_agent.tasks import parse_task_message

NEEDS_DRIVE = {"content": "Go to the sofa", "requires_tool": "go_to"}
DONE = AIMessage("Done.")
UNKNOWN_STATUS = (
    "Status 'done' is not allowed; use one of pending, in_progress, completed, failed, cancelled."
)
# The task from ADR-0041.
MUG_TASK = [
    "Go to the kitchen",
    "Pick up the red mug",
    "Come back to the operator",
    {"content": "Hand over the mug", "requires_tool": "hand_over"},
]


def refused_with(stack, call_id: str, text: str) -> None:
    answer = stack.answer(call_id)
    assert answer.status == "error"
    assert answer.content.startswith("REFUSED: ")
    assert text in answer.content
    assert answer.content.endswith("None of the changes were applied.")


# -- Seeds -----------------------------------------------------------------------


def test_the_first_model_call_sees_the_seed_as_a_trailing_reminder():
    # R1a, R1b, R1c, R2a, T3a
    stack = make_stack(DONE, then=DONE)
    stack.run(todos=[NEEDS_DRIVE, "Say hello"])

    sent = stack.model.received[0]
    assert [type(m) for m in sent] == [SystemMessage, HumanMessage, HumanMessage]
    assert stack.reminder(0) == (
        "<todos>\n[ ] t1 Go to the sofa  needs: go_to\n[ ] t2 Say hello\n</todos>"
    )
    assert {"update_todos", "wait_for_person"} <= set(stack.model.bound[0])
    assert stack.state.get("todo_seed") is None
    items = stack.items()
    assert [i["id"] for i in items] == ["t1", "t2"]
    assert all(i["locked"] and i["source"] == "seed" for i in items)


def test_an_operator_seed_from_a_task_message_keeps_its_source():
    # R2b; test_graph.py's test_submit_seeds_the_list_before_the_first_call, both sources
    text, seed = parse_task_message('{"task": "Greet.", "todos": ["Say hello"]}', {})
    stack = make_stack(DONE, then=DONE)
    stack.run(text, todos=seed, source="operator")

    assert stack.items()[0]["source"] == "operator"
    assert stack.items()[0]["locked"] is True
    assert stack.model.received[0][1].content == "Greet."


def test_a_new_seed_replaces_the_old_list_and_restarts_ids():
    # R3b
    stack = make_stack(DONE, DONE, DONE, DONE, DONE, DONE)
    stack.run(todos=["Old one", "Old two"])
    stack.run(todos=["New one"])

    assert [(i["id"], i["content"]) for i in stack.items()] == [("t1", "New one")]
    assert stack.reminder() == "<todos>\n[ ] t1 New one\n</todos>"


def test_an_empty_seed_empties_the_list_and_none_is_no_seed():
    # [] differs from None: test_graph.py's test_submit_with_an_empty_seed_empties_the_list
    stack = make_stack(DONE, DONE, DONE, DONE)
    stack.run(todos=["Step"])
    assert stack.items() != []
    stack.run(todos=[])
    assert stack.items() == []
    assert stack.reminder() == ""
    assert stack.published[-1] == ([], [])


def test_an_unseeded_task_does_not_inherit_the_last_tasks_open_steps():
    # R3c: an intentional change from main, decided in the plan
    stack = make_stack(DONE, DONE, DONE, DONE)
    stack.run(todos=["Open step"])
    assert stack.statuses() == ["pending"]
    stack.run("Something else.")
    assert stack.items() == []
    assert stack.reminder() == ""


def test_two_seeded_steps_with_the_same_wording_stay_apart():
    # Duplicate wording: the spike refused it, ids make it fine
    stack = make_stack(
        ai(todo_call(update("t2", status="in_progress"))),
        ai(todo_call(update("t2", status="completed"), call_id="u2")),
        ai(todo_call(update("t1", status="completed"), call_id="u3")),
        DONE,
    )
    stack.run(todos=["Greet a guest", "Greet a guest"])
    assert stack.statuses() == ["completed", "completed"]
    assert stack.model.received[1][-1].content == (
        "<todos>\n[ ] t1 Greet a guest\n[>] t2 Greet a guest\n</todos>"
    )


@pytest.mark.parametrize(
    ("seed", "error"),
    [
        ("Say hi", "A seed is a list of items, not a single string."),
        ([{"content": "Hi", "require_tool": "say"}], "Seed item t1 has unknown keys: require_tool."),
        ([{"content": " "}], "Seed item t1 has no content."),
        ([3], "Seed item t1 must be a string or a dict, not int."),
    ],
)
def test_a_bad_seed_is_refused_as_a_whole_before_the_run(seed, error):
    with pytest.raises(TodoRejected) as caught:
        parse_seed(seed)
    assert str(caught.value) == error


def test_a_seeds_source_must_be_seed_or_operator():
    with pytest.raises(ValueError, match="source must be"):
        parse_seed(["Hi"], source="agent")


def test_parse_seed_shares_nothing_with_its_input():
    entry = {"content": "Hi"}
    items = parse_seed([entry])
    entry["content"] = "Changed"
    assert items[0]["content"] == "Hi"
    assert items[0]["evidence"] == []


# -- Seeded steps are locked -----------------------------------------------------


def test_the_rules_for_seeded_steps():
    # R4 drop, reorder, reword, extra keys, early completed, insert; R5a; T3b, T3c, T3d, T3e
    stack = make_stack(
        ai(todo_call(remove("t1"), call_id="drop")),
        ai(todo_call(update("t1", after="t2"), call_id="reorder")),
        ai(todo_call(update("t1", content="Go somewhere"), call_id="reword")),
        ai(todo_call(update("t1", status="waiting"), call_id="waiting")),
        # Keys the schema does not know are ignored, so they cannot unlock a step.
        ai(todo_call(update("t1", status="in_progress", locked=False, requires_tool=""), call_id="keys")),
        ai(todo_call(update("t1", status="completed"), call_id="early")),
        ai(todo_call(update("t2", status="failed"), call_id="no_note")),
        ai(drive("drive")),
        ai(
            todo_call(
                update("t1", status="completed"),
                update("t2", status="in_progress"),
                add("Check the door", after="t2"),
                call_id="insert",
            )
        ),
        ai(todo_call(update("t3", status="cancelled", note="Not needed."), call_id="own")),
        ai(todo_call(update("t2", status="completed"), call_id="done")),
        DONE,
    )
    stack.run(todos=[NEEDS_DRIVE, "Say hello"])

    refused_with(stack, "drop", "t1 is a seeded step, so it cannot be removed")
    refused_with(stack, "reorder", "Seeded steps keep their order")
    refused_with(stack, "reword", "t1 is a seeded step, so its wording cannot change.")
    refused_with(stack, "waiting", "Only wait_for_person can set waiting")
    assert stack.answer("keys").status == "success"
    refused_with(stack, "early", "t1 needs a successful go_to before it can be completed.")
    refused_with(stack, "no_note", "Marking t2 failed needs a note that says why.")
    assert stack.answer("insert").status == "success"
    assert stack.answer("done").content == "Updated: t2 → completed. Open: none."
    items = stack.items()
    assert [i["id"] for i in items] == ["t1", "t2", "t3"]
    assert items[0]["locked"] and items[0]["requires_tool"] == "go_to"
    assert not items[2]["locked"] and items[2]["source"] == "agent"
    assert stack.statuses() == ["completed", "completed", "cancelled"]


def test_a_refused_call_changes_nothing_at_all():
    # All or nothing: the good op before the bad one is not applied either
    stack = make_stack(
        ai(todo_call(update("t2", status="in_progress"), remove("t1"), call_id="mixed")),
        DONE,
        DONE,
        DONE,
    )
    stack.run(todos=["One", "Two"])
    refused_with(stack, "mixed", "t1 is a seeded step")
    assert stack.statuses() == ["pending", "pending"]


def test_a_seeded_step_can_move_past_an_own_step_but_not_past_another_seeded_one():
    stack = make_stack(
        ai(todo_call(add("Own", after="t1"), update("t1", after="t3"), call_id="own")),
        ai(todo_call(update("t1", after="t2"), call_id="seeded")),
        DONE,
        DONE,
        DONE,
    )
    stack.run(todos=["One", "Two"])
    assert stack.answer("own").status == "success"
    assert [i["id"] for i in stack.items()] == ["t3", "t1", "t2"]
    refused_with(stack, "seeded", "Seeded steps keep their order")


def test_the_agents_own_steps_are_not_locked():
    # R4 own steps
    stack = make_stack(
        ai(todo_call(add("Find the bottle"), add("Bring it back"))),
        ai(todo_call(update("t1", content="Find the cup"), remove("t2"), call_id="u2")),
        DONE,
        DONE,
        DONE,
    )
    stack.run()
    assert [(i["id"], i["content"]) for i in stack.items()] == [("t1", "Find the cup")]
    assert stack.items()[0]["active_form"] == "Find the cup"


def test_ids_are_never_reused_within_a_task():
    stack = make_stack(
        ai(todo_call(add("One"), add("Two"))),
        ai(todo_call(remove("t2"), call_id="u2")),
        ai(todo_call(add("Three"), call_id="u3")),
        ai(todo_call(update("t1", status="cancelled", note="n"), update("t3", status="cancelled", note="n"), call_id="u4")),
        DONE,
    )
    stack.run()
    assert [i["id"] for i in stack.items()] == ["t1", "t3"]


@pytest.mark.parametrize(
    ("op", "error"),
    [
        (add("x", id="t9"), "Ids are assigned by the list"),
        ({"op": "add"}, "An add op needs content"),
        ({"op": "update"}, "An update op needs the id"),
        ({"op": "remove"}, "A remove op needs the id"),
        (update("t9", status="pending"), "There is no item t9; use an id from the todo list."),
        (remove("t9"), "There is no item t9"),
        (add("x", after="t9"), "There is no item t9 to add the new item after."),
        (update("t1", after="t9"), "There is no item t9 to move t1 after."),
        (update("t1", after="t1"), "t1 cannot be moved after itself."),
        (update("t1", status="pending"), "The update to t1 changes nothing"),
        (update("t1", status="paused"), "Status 'paused' is not allowed"),
        (add("x", status="failed"), "Marking t2 failed needs a note"),
        # The complete matrices of test_todo.py:162, :169, :249, :257 and :267.
        (add("  "), "An add op needs content"),
        (update("t9", status="completed"), "There is no item t9; use an id from the todo list."),
        (remove("t9"), "There is no item t9; use an id from the todo list."),
        (add("x", status="failed"), "Marking t2 failed needs a note that says why."),
        (add("x", status="cancelled"), "Marking t2 cancelled needs a note that says why."),
        (update("t1", status="failed"), "Marking t1 failed needs a note that says why."),
        (update("t1", status="cancelled"), "Marking t1 cancelled needs a note that says why."),
        (add("x", status="failed", note="  "), "Marking t2 failed needs a note that says why."),
        (add("x", status="cancelled", note="  "), "Marking t2 cancelled needs a note that says why."),
        (update("t1", status="failed", note="  "), "Marking t1 failed needs a note that says why."),
        (update("t1", status="cancelled", note="  "), "Marking t1 cancelled needs a note that says why."),
        (add("x", status="waiting"), "Only wait_for_person can set waiting"),
        (update("t1", status="waiting"), "Only wait_for_person can set waiting"),
        (add("x", status="done"), UNKNOWN_STATUS),
        (update("t1", status="done"), UNKNOWN_STATUS),
    ],
)
def test_each_refusal_says_what_to_do(op, error):
    stack = make_stack(ai(todo_call(op, call_id="bad")), DONE, DONE, DONE)
    stack.run(todos=["One"])
    refused_with(stack, "bad", error)


def test_an_empty_update_is_refused():
    stack = make_stack(ai(todo_call(call_id="bad")), DONE, DONE, DONE)
    stack.run(todos=["One"])
    refused_with(stack, "bad", "The call has no ops")


def test_only_one_step_can_be_in_progress():
    # R5f
    stack = make_stack(
        ai(todo_call(add("One", status="in_progress"), add("Two", status="in_progress"), call_id="two")),
        DONE,
    )
    stack.run()
    refused_with(stack, "two", "Only one item can be in progress or waiting at a time, but t1 and t2")


def test_null_fields_read_as_not_given_and_text_becomes_one_line():
    stack = make_stack(
        ai(
            todo_call(
                {
                    "op": "add",
                    "id": None,
                    "content": " Two\nlines ",
                    "status": "IN_PROGRESS",
                    "note": None,
                    "after": None,
                    "active_form": None,
                }
            )
        ),
        ai(todo_call(update("t1", status="cancelled", note="Not needed."), call_id="u2")),
        DONE,
    )
    stack.run()
    item = stack.items()[0]
    assert stack.answer("u1").status == "success"
    assert item["content"] == "Two lines"
    assert stack.model.received[1][-1].content == "<todos>\n[>] t1 Two lines\n</todos>"


# -- The ledger: step evidence and failures --------------------------------------


def test_one_drive_cannot_complete_two_steps():
    # Verified bug: main's shared ledger window completed both from one drive.
    stack = make_stack(
        ai(drive("drive")),
        ai(todo_call(update("t1", status="completed"), update("t2", status="completed"), call_id="both")),
        ai(todo_call(update("t1", status="completed"), call_id="one")),
        ai(todo_call(update("t2", status="completed"), call_id="other")),
        ai(drive("drive2")),
        ai(todo_call(update("t2", status="completed"), call_id="second")),
        DONE,
    )
    stack.run(todos=[NEEDS_DRIVE, {"content": "Go back", "requires_tool": "go_to"}])

    refused_with(stack, "both", "t2 needs a successful go_to before it can be completed.")
    assert stack.answer("one").status == "success"
    refused_with(stack, "other", "t2 needs a successful go_to")
    assert stack.answer("second").status == "success"
    assert [i["evidence"] for i in stack.items()] == [["drive"], ["drive2"]]


def test_a_repeated_call_id_still_counts_as_a_new_result():
    # Some providers reuse ids across messages, so consumption goes by ledger position.
    stack = make_stack(
        ai(drive("same")),
        ai(todo_call(update("t1", status="completed"), call_id="one")),
        ai(drive("same")),
        ai(todo_call(update("t2", status="completed"), call_id="two")),
        DONE,
    )
    stack.run(todos=[NEEDS_DRIVE, {"content": "Go back", "requires_tool": "go_to"}])
    assert stack.statuses() == ["completed", "completed"]


def test_two_successful_calls_in_one_message_do_not_collide():
    # R8e; verified bug: a whole-list write per tool raises InvalidUpdateError
    stack = make_stack(
        ai(drive("d1"), drive("d2", "table")),
        ai(todo_call(update("t1", status="completed"), update("t2", status="completed"))),
        DONE,
    )
    stack.run(todos=[NEEDS_DRIVE, {"content": "Go back", "requires_tool": "go_to"}])
    assert stack.statuses() == ["completed", "completed"]
    assert [i["evidence"] for i in stack.items()] == [["d1"], ["d2"]]
    assert stack.state["ledger"] == [
        {"tool": "go_to", "id": "d1", "ok": True},
        {"tool": "go_to", "id": "d2", "ok": True},
    ]


def test_a_mixed_batch_of_action_then_update_sees_the_action():
    stack = make_stack(
        ai(drive("d1"), todo_call(update("t1", status="completed"))),
        DONE,
    )
    stack.run(todos=[NEEDS_DRIVE])
    assert stack.answer("u1").status == "success"
    assert stack.statuses() == ["completed"]
    assert stack.items()[0]["evidence"] == ["d1"]


def test_a_mixed_batch_of_update_then_action_writes_both():
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress")), drive("d1")),
        ai(todo_call(update("t1", status="completed"), call_id="u2")),
        DONE,
    )
    stack.run(todos=[NEEDS_DRIVE])
    assert stack.statuses() == ["completed"]
    assert stack.state["ledger"] == [{"tool": "go_to", "id": "d1", "ok": True}]


def test_completion_waits_until_the_failed_tool_succeeds():
    # test_graph.py:419; R5d, R5g
    stack = make_stack(
        ai(todo_call(add("Go", status="in_progress"))),
        ai(drive("bad", "attic")),
        ai(todo_call(update("t1", status="completed"), call_id="early")),
        ai(drive("good")),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        DONE,
    )
    stack.run()
    refused_with(stack, "early", "t1 cannot be completed because go_to failed and has not succeeded since")
    assert stack.answer("done").status == "success"


def test_a_note_excuses_a_failure_but_not_a_missing_required_tool():
    # R5g
    stack = make_stack(
        ai(drive("bad", "attic")),
        ai(todo_call(update("t1", status="completed", note="Close enough."), call_id="excuse")),
        ai(todo_call(update("t2", status="completed", note="Close enough."), call_id="no_tool")),
        ai(todo_call(update("t2", status="cancelled", note="No need."), call_id="cancel")),
        DONE,
    )
    stack.run(todos=["Go", NEEDS_DRIVE])
    assert stack.answer("excuse").status == "success"
    refused_with(stack, "no_tool", "t2 needs a successful go_to")
    assert stack.items()[0]["note"] == "Close enough."


def test_a_handled_tool_exception_counts_as_failed():
    # test_graph.py:459: a verb's StepError goes through ToolErrorMiddleware
    stack = make_stack(
        ai(todo_call(add("Say it", status="in_progress"))),
        ai(call("say", {"text": "Hello"}, "spoke")),
        ai(todo_call(update("t1", status="completed"), call_id="early")),
        DONE,
        DONE,
        DONE,
    )
    stack.robot.faults["speech.say"] = StepError(ErrorCode.TIMEOUT, "The speaker hung.")
    stack.run()
    assert stack.answer("spoke").status == "error"
    assert stack.answer("spoke").content == "TIMEOUT: The speaker hung."
    assert stack.state["ledger"] == [{"tool": "say", "id": "spoke", "ok": False}]
    refused_with(stack, "early", "because say failed")


def test_an_unexpected_exception_counts_as_failed_and_stays_out_of_the_conversation():
    stack = make_stack(
        ai(call("say", {"text": "Hello"}, "spoke")),
        ai(todo_call(update("t1", status="completed"), call_id="early")),
        DONE,
        DONE,
        DONE,
    )
    stack.robot.faults["speech.say"] = RuntimeError("secret detail")
    stack.run(todos=["Say it"])
    assert stack.answer("spoke").content == "INTERNAL: say failed with RuntimeError."
    assert stack.state["ledger"][0]["ok"] is False
    refused_with(stack, "early", "because say failed")


def test_a_call_with_bad_arguments_counts_as_failed():
    # test_graph.py:492, with ToolNode's wording; validation errors skip ToolErrorMiddleware
    stack = make_stack(
        ai(call("go_to", {"place": 3}, "bad")),
        ai(todo_call(update("t1", status="completed"), call_id="early")),
        ai(drive("good")),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        DONE,
    )
    stack.run(todos=["Go"])
    answer = stack.answer("bad")
    assert answer.status == "error"
    assert answer.content.startswith("Error invoking tool 'go_to' with kwargs")
    assert "Please fix the error and try again." in answer.content
    assert stack.state["ledger"][0] == {"tool": "go_to", "id": "bad", "ok": False}
    refused_with(stack, "early", "because go_to failed")
    assert stack.answer("done").status == "success"


def test_a_call_that_never_ran_does_not_poison_completion():
    # Refused or skipped calls are not executed failures.
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress")), drive("bad", "attic"), drive("skipped")),
        ai(drive("good")),
        ai(call("no_such_tool", {}, "unknown")),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        DONE,
    )
    stack.run(todos=["Go"])
    assert stack.answer("skipped").content == (
        "Not run: go_to failed earlier in this message. Decide again."
    )
    assert stack.answer("unknown").status == "error"
    assert [(e["id"], e["ok"]) for e in stack.state["ledger"]] == [("bad", False), ("good", True)]
    assert stack.answer("done").status == "success"


def test_going_in_progress_again_starts_a_clean_window():
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"))),
        ai(drive("bad", "attic")),
        ai(todo_call(update("t1", status="pending"), call_id="back")),
        ai(todo_call(update("t1", status="in_progress"), call_id="again")),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        DONE,
    )
    stack.run(todos=["Go"])
    assert stack.answer("done").status == "success"


def test_a_requirement_counts_from_when_the_step_was_added():
    # The drive came before the step went in progress, and still counts for it.
    stack = make_stack(
        ai(drive("d1")),
        ai(todo_call(update("t1", status="in_progress"))),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        DONE,
    )
    stack.run(todos=[NEEDS_DRIVE])
    assert stack.answer("done").status == "success"


def test_a_new_seed_resets_the_windows():
    # The old task's drive does not count for the new task's step.
    stack = make_stack(
        ai(drive("old")),
        DONE,
        DONE,
        DONE,
        ai(todo_call(update("t1", status="completed"), call_id="early")),
        DONE,
        DONE,
        DONE,
    )
    stack.run(todos=["First"])
    stack.run(todos=[NEEDS_DRIVE])
    refused_with(stack, "early", "t1 needs a successful go_to")


def test_only_the_first_update_todos_call_in_a_message_is_applied():
    # test_graph.py:518; R8g changed: the spike refused both, now only the duplicate is
    # refused. It counts as a failed call, so the drive after it does not run.
    stack = make_stack(
        ai(todo_call(add("Say hello"), call_id="u1"), todo_call(add("Say goodbye"), call_id="u2"), drive("d1")),
        DONE,
        DONE,
        DONE,
    )
    stack.run()
    assert [i["content"] for i in stack.items()] == ["Say hello"]
    assert stack.answer("u1").status == "success"
    assert stack.answer("u2").status == "error"
    assert stack.answer("u2").content == (
        "Only one update_todos call per message is accepted, so this one was not applied. "
        "Put all ops in one call."
    )
    assert stack.answer("d1").content == (
        "Not run: update_todos failed earlier in this message. Decide again."
    )
    assert stack.verbs() == []


# -- The reminder and the gate ---------------------------------------------------


def test_every_call_gets_the_current_list_and_it_is_never_saved():
    # R7a, T6
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"))),
        ai(todo_call(update("t1", status="completed"), call_id="u2")),
        DONE,
    )
    stack.run(todos=["One"])
    assert stack.model.received[0][-1].content == "<todos>\n[ ] t1 One\n</todos>"
    assert stack.model.received[1][-1].content == "<todos>\n[>] t1 One\n</todos>"
    assert stack.model.received[2][-1].content == "<todos>\n[x] t1 One\n</todos>"
    assert not any("<todos>" in str(m.content) for m in stack.messages)


def test_the_reminder_lines():
    items = parse_seed([NEEDS_DRIVE])
    assert render([]) == ""
    assert render(items) == "<todos>\n[ ] t1 Go to the sofa  needs: go_to\n</todos>"
    waiting = {**items[0], "status": "waiting", "question": "Which?"}
    assert 'waiting: "Which?"' in render([waiting])
    done = {**items[0], "status": "failed", "note": "Blocked."}
    assert render([done]) == '<todos>\n[!] t1 Go to the sofa  note: "Blocked."\n</todos>'


def test_gate_sends_the_llm_back_at_most_twice_per_task():
    # test_graph.py:630; R8a, T4, T4b
    stack = make_stack(then=DONE)
    stack.run("Go and come back.", todos=["Go", "Come back"])
    gate = "[todo] Open items: t1, t2. Finish them, or mark them failed or cancelled with a note."
    history = [m.content for m in stack.messages]
    assert history == ["Go and come back.", "Done.", gate, "Done.", gate, "Done."]
    assert len(stack.model.received) == 3
    reminder = "<todos>\n[ ] t1 Go\n[ ] t2 Come back\n</todos>"
    assert [sent[-1].content for sent in stack.model.received] == [reminder] * 3
    assert stack.model.received[1][-2].content == gate

    # A new task gets two gates again.
    stack.run("And again.", todos=["Go"])
    assert len(stack.model.received) == 6


def test_no_gate_without_open_items():
    # test_graph.py:687
    stack = make_stack(
        ai(todo_call(add("Go", status="cancelled", note="Not needed."))),
        DONE,
    )
    stack.run()
    assert len(stack.model.received) == 2
    assert stack.messages[-1].content == "Done."


def test_a_question_ends_the_run_with_no_gate_and_the_answer_keeps_the_gate_budget():
    # test_graph.py:703 (no gate while waiting) and R5h changed: a question now ends the
    # run at once, so a model call can never see a waiting step. R8b: one gate spent
    # before the question leaves one after the answer.
    stack = make_stack(
        DONE,  # gate 1
        ai(todo_call(update("t1", status="in_progress"))),
        ai(call("wait_for_person", {"question": "Which one?"}, "ask")),
        then=DONE,
    )
    stack.run(todos=["Ask", "Serve"])
    assert len(stack.model.received) == 3
    assert stack.statuses() == ["waiting", "pending"]
    gates = [m for m in stack.messages if str(m.content).startswith("[todo]")]
    assert len(gates) == 1

    stack.run("The red one.")
    assert len(stack.model.received) == 5  # the answer, then one more gate
    gates = [m for m in stack.messages if str(m.content).startswith("[todo]")]
    assert len(gates) == 2
    assert stack.statuses() == ["in_progress", "pending"]


# -- Questions -------------------------------------------------------------------


def test_wait_for_person_ends_the_run_and_the_answer_resumes_the_item():
    # test_graph.py:716; R3a, R5b; verified bug: the spike never ran the model again
    stack = make_stack(
        ai(todo_call(add("Find the cup", status="in_progress"))),
        ai(call("wait_for_person", {"question": "Red  or\nblue?"}, "ask")),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        DONE,
    )
    stack.run("Find my cup.")

    assert len(stack.model.received) == 2
    assert stack.robot.said == ["Red or blue?"]
    assert isinstance(stack.messages[-1], ToolMessage)
    assert stack.answer("ask").content == "Asked: Red or blue?"
    item = stack.items()[0]
    assert (item["status"], item["question"]) == ("waiting", "Red or blue?")
    assert stack.state["question"] == "Red or blue?"

    stack.run("The red one.")
    assert len(stack.model.received) == 4
    assert stack.model.received[2][-1].content == "<todos>\n[>] t1 Find the cup\n</todos>"
    assert stack.model.received[2][-2].content == "The red one."
    assert stack.statuses() == ["completed"]
    assert not stack.state["question"]


def test_an_answer_keeps_the_window_of_the_attempt_that_asked():
    # A failure before the question still counts after the answer.
    stack = make_stack(
        ai(todo_call(add("Go", status="in_progress"))),
        ai(drive("bad", "attic")),
        ai(call("wait_for_person", {"question": "Where?"}, "ask")),
        ai(todo_call(update("t1", status="completed"), call_id="early")),
        DONE,
        DONE,
        DONE,
    )
    stack.run()
    stack.run("The sofa.")
    refused_with(stack, "early", "because go_to failed")


def test_a_question_with_no_active_step_still_ends_the_run_and_keeps_the_list():
    # Question state is defined with no active step: nothing is marked waiting.
    stack = make_stack(
        ai(call("wait_for_person", {"question": "Which?"}, "ask")),
        DONE,
        DONE,
        DONE,
    )
    stack.run(todos=["One"])
    assert len(stack.model.received) == 1
    assert stack.statuses() == ["pending"]

    stack.run("That one.")
    assert stack.model.received[1][-2].content == "That one."
    assert len(stack.model.received) == 4  # the answer, then the task's two gates
    assert stack.statuses() == ["pending"]  # the answer did not clear the list


def test_a_question_with_no_list_at_all():
    stack = make_stack(ai(call("wait_for_person", {"question": "Which?"}, "ask")), DONE)
    stack.run()
    assert len(stack.model.received) == 1
    stack.run("That one.")
    assert len(stack.model.received) == 2
    assert stack.messages[-1].content == "Done."


def test_calls_after_a_question_in_the_same_message_do_not_run():
    # test_graph.py:754; 3b, 5
    stack = make_stack(
        ai(
            call("wait_for_person", {"question": "Which?"}, "ask"),
            drive("d1"),
            todo_call(add("Later")),
            call("say", {"text": "Hi"}, "s1"),
        ),
    )
    stack.run()
    for call_id in ("d1", "u1", "s1"):
        answer = stack.answer(call_id)
        assert answer.status == "error"
        assert answer.content.startswith("Not run: you asked the person a question")
    assert stack.verbs() == ["speech.say"]
    assert stack.robot.said == ["Which?"]
    assert stack.state["ledger"] == [{"tool": "wait_for_person", "id": "ask", "ok": True}]
    assert len(stack.model.received) == 1


def test_a_step_going_in_progress_in_the_same_message_as_the_question_waits():
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress")), call("wait_for_person", {"question": "Which?"}, "ask")),
    )
    stack.run(todos=["One"])
    assert stack.statuses() == ["waiting"]


def test_failed_wait_for_person_does_not_end_the_run():
    # test_graph.py:791; verified bug: return_direct ended the run whatever the status
    stack = make_stack(
        ai(todo_call(add("Ask", status="in_progress"))),
        ai(call("wait_for_person", {"question": "Which?"}, "ask")),
        ai(todo_call(update("t1", status="failed", note="No voice."), call_id="u2")),
        DONE,
    )
    stack.robot.faults["speech.say"] = StepError(ErrorCode.TIMEOUT, "The speaker hung.")
    stack.run()
    assert stack.answer("ask").status == "error"
    assert stack.answer("ask").content == "TIMEOUT: The speaker hung."
    assert not stack.state["question"]
    assert stack.statuses() == ["failed"]
    assert len(stack.model.received) == 4


def test_a_seed_wins_over_a_pending_question():
    stack = make_stack(
        ai(todo_call(add("Old", status="in_progress"))),
        ai(call("wait_for_person", {"question": "Which?"}, "ask")),
        DONE,
        DONE,
        DONE,
    )
    stack.run()
    stack.run("New task.", todos=["New"])
    assert [i["content"] for i in stack.items()] == ["New"]
    assert not stack.state["question"]


# -- Publishing ------------------------------------------------------------------


def test_publishing_follows_every_accepted_change_and_no_refused_one():
    # R6a, R6b changed (a run start publishes only if the list changed), R6c
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"), call_id="start")),
        ai(todo_call(remove("t1"), call_id="refused")),
        ai(call("wait_for_person", {"question": "Which?"}, "ask")),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        DONE,
    )
    stack.run(todos=["One"])
    stack.run("That one.")

    statuses = [[i["status"] for i in items] for items, _ in stack.published]
    assert statuses == [
        ["pending"],  # the seed, first
        ["in_progress"],
        ["waiting"],
        ["in_progress"],
        ["completed"],
    ]
    assert [[i["id"] for i in changed] for _, changed in stack.published] == [["t1"]] * 5
    assert stack.published[-1][0] == public(stack.items())
    for items, _ in stack.published:
        json.dumps(items)
        assert not {"added", "since", "used"} & set(items[0])


def test_a_listener_that_raises_does_not_affect_the_list():
    def broken(items, changed):
        raise RuntimeError("no subscribers")

    stack = make_stack(DONE, DONE, DONE, on_todos=broken)
    stack.run(todos=["One"])
    assert stack.statuses() == ["pending"]


@pytest.mark.parametrize(
    ("status", "active_form", "line"),
    [
        ("pending", "Going", ""),
        ("in_progress", "Going", "t1: Going"),
        ("in_progress", "", "t1: Go"),
        ("completed", "Going", "t1 completed: Go"),
        ("failed", "Going", "t1 failed: Go"),
        ("cancelled", "Going", "t1 cancelled: Go"),
        ("waiting", "Going", "t1 waiting: Go"),
    ],
)
def test_console_lines(status, active_form, line):
    # test_todo.py:624, all its parameters
    item = {**parse_seed(["Go"])[0], "status": status, "active_form": active_form}
    assert console_line(item) == line


# -- The tools' schemas ----------------------------------------------------------


def test_the_tool_schemas_are_flat_and_hide_the_injected_runtime():
    # Schema: on the final tools, including the injected argument
    update_todos, wait_for_person = WalkieTodos(CancelToken(), lambda text: None).tools
    openai = [convert_to_openai_tool(tool) for tool in (update_todos, wait_for_person)]
    blob = json.dumps(openai)
    assert not any(word in blob for word in ('"anyOf"', '"oneOf"', '"allOf"', '"$ref"', "runtime"))
    ops = openai[0]["function"]["parameters"]["properties"]["ops"]
    assert ops["description"].startswith("The changes to make, applied in order.")
    assert set(ops["items"]["properties"]) == set(TodoOp.model_fields)
    assert all(p["type"] == "string" for k, p in ops["items"]["properties"].items() if k != "op")
    question = openai[1]["function"]["parameters"]["properties"]["question"]
    assert question["type"] == "string"
    assert question["description"].startswith("The question to ask out loud")
    assert openai[0]["function"]["description"].startswith("Edit your todo list")
    assert wait_for_person.return_direct is False


# -- Moved from test_todo.py, which tested TodoList ---------------------------------


def waiting_draft(ledger: list[dict]) -> _Draft:
    """A list with t1 waiting and t2 pending. Only a draft can show this to an op.

    A question ends the run, and its answer resumes the step before any model call, so
    no real run lets an op see a waiting step. The rules for one still hold if a model
    or a checkpoint ever does.
    """
    first, second = parse_seed(["Pick up the mug", "Come back"])
    first = {**first, "status": "waiting", "question": "Which one?", "added": 0, "since": 0}
    return _Draft([first, {**second, "added": 0, "since": 0}], 3, ledger)


def test_legacy_op_transactions():
    # test_todo.py:138: add assigns ids, appends, and fills in the defaults.
    stack = make_stack(
        ai(todo_call(add("Go to the kitchen"), add("Look for the mug", status="in_progress"))),
        then=DONE,
    )
    stack.run()
    first, second = stack.items()
    assert (first["id"], second["id"]) == ("t1", "t2")
    assert (first["source"], first["locked"]) == ("agent", False)
    assert first["active_form"] == "Go to the kitchen"
    assert first["status"] == "pending"
    assert second["status"] == "in_progress"

    # test_todo.py:149: insert after an item.
    stack = make_stack(ai(todo_call(add("Open the cupboard", after="t1"))), then=DONE)
    stack.run(todos=MUG_TASK)
    assert [i["id"] for i in stack.items()] == ["t1", "t5", "t2", "t3", "t4"]

    # test_todo.py:173: an op can name an item added earlier in the same call.
    stack = make_stack(
        ai(todo_call(add("Go to the kitchen"), update("t1", status="in_progress"))), then=DONE
    )
    stack.run()
    assert stack.statuses() == ["in_progress"]

    # test_todo.py:179: an own step can move, and not after itself.
    stack = make_stack(
        ai(todo_call(add("Wipe the table"), update("t5", after="t2"))),
        ai(todo_call(update("t5", after="t5"), call_id="itself")),
        then=DONE,
    )
    stack.run(todos=MUG_TASK)
    assert [i["id"] for i in stack.items()] == ["t1", "t2", "t5", "t3", "t4"]
    refused_with(stack, "itself", "t5 cannot be moved after itself.")

    # test_todo.py:195: a move to where the step already is changes nothing either.
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"))),
        ai(todo_call(update("t1", status="in_progress"), call_id="same")),
        ai(todo_call(update("t2", after="t1"), call_id="same_place")),
        then=DONE,
    )
    stack.run(todos=MUG_TASK)
    refused_with(stack, "same", "The update to t1 changes nothing")
    refused_with(stack, "same_place", "The update to t2 changes nothing")

    # test_todo.py:286: the one-active rule looks at the end of the whole call.
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"))),
        ai(todo_call(update("t1", status="completed"), update("t2", status="in_progress"), call_id="swap")),
        ai(
            todo_call(
                update("t2", status="pending"),
                update("t3", status="in_progress"),
                update("t4", status="in_progress"),
                call_id="two",
            )
        ),
        then=DONE,
    )
    stack.run(todos=MUG_TASK)
    assert stack.answer("swap").status == "success"
    refused_with(stack, "two", "t3 and t4")
    assert stack.statuses() == ["completed", "in_progress", "pending", "pending"]

    # test_todo.py:298: a refused call does not move the id counter.
    stack = make_stack(
        ai(todo_call(add("Wipe the table"), remove("t9"), call_id="bad")),
        ai(todo_call(add("Wipe the table"), call_id="good")),
        then=DONE,
    )
    stack.run(todos=MUG_TASK)
    refused_with(stack, "bad", "There is no item t9")
    assert stack.answer("good").content.startswith("Added: t5.")

    # test_todo.py:274: a waiting step counts as active too.
    with pytest.raises(TodoRejected, match="but t1 and t2 would be"):
        waiting_draft([]).apply([TodoOp(op="update", id="t2", status="in_progress")])


def test_a_bad_seed_keeps_the_old_list_in_the_checkpoint():
    # test_todo.py:108. A seed is checked in the caller, so nothing reaches the agent.
    stack = make_stack(ai(todo_call(update("t1", status="in_progress"))), then=DONE)
    stack.run(todos=["Keep me"])
    before, calls = stack.items(), len(stack.model.received)
    with pytest.raises(TodoRejected, match="Seed item t2 has no content"):
        stack.run(todos=["Replace it", {"content": " "}])
    assert stack.items() == before
    assert len(stack.model.received) == calls


def test_refused_transaction_keeps_evidence_window():
    # test_todo.py:312. Going back to pending and in progress again would start a new
    # window, but the call is refused, so the window stays where it was.
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"))),
        ai(drive("bad", "attic")),
        ai(
            todo_call(
                update("t1", status="pending"),
                update("t1", status="in_progress"),
                remove("t9"),
                call_id="refused",
            )
        ),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        then=DONE,
    )
    stack.run(todos=["Go"])
    refused_with(stack, "refused", "There is no item t9")
    refused_with(stack, "done", "because go_to failed")


def test_legacy_summary_and_render_cases():
    # test_todo.py:330 and :785: the sentence update_todos answers with, in a fixed order.
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"), call_id="a")),
        ai(todo_call(add("Wipe the table"), add("Close the door"), call_id="b")),
        ai(
            todo_call(
                update("t1", status="completed"),
                update("t2", status="in_progress"),
                update("t3", note="Use the side door."),
                remove("t6"),
                add("Close the window"),
                call_id="c",
            )
        ),
        then=DONE,
    )
    stack.run(todos=MUG_TASK)
    assert stack.answer("a").status == "success"
    assert stack.answer("a").content == "Updated: t1 → in_progress. Open: t1, t2, t3, t4."
    assert stack.answer("b").content == "Added: t5, t6. Open: t1, t2, t3, t4, t5, t6."
    assert stack.answer("c").content == (
        "Added: t7. Updated: t1 → completed, t2 → in_progress, t3. Removed: t6. "
        "Open: t2, t3, t4, t5, t7."
    )

    # test_todo.py:352
    stack = make_stack(
        ai(todo_call(add("Wave"))), ai(todo_call(update("t1", status="completed"), call_id="d")), DONE
    )
    stack.run()
    assert stack.answer("d").content == "Updated: t1 → completed. Open: none."

    # test_todo.py:524: finished steps leave the open ids, and a waiting one stays.
    stack = make_stack(
        ai(
            todo_call(
                update("t1", status="completed"),
                update("t2", status="cancelled", note="There is no red mug."),
                update("t3", status="in_progress"),
                call_id="e",
            )
        ),
        ai(call("wait_for_person", {"question": "Where are you?"}, "ask")),
        ai(todo_call(update("t3", status="failed", note="The operator left."), call_id="f")),
        then=DONE,
    )
    stack.run(todos=MUG_TASK)
    assert stack.answer("e").content.endswith("Open: t3, t4.")
    assert stack.statuses() == ["completed", "cancelled", "waiting", "pending"]
    stack.run("Over here.")
    assert stack.answer("f").content.endswith("Open: t4.")

    # test_todo.py:542: the ADR's example.
    stack = make_stack(
        ai(todo_call(update("t1", status="completed"), update("t2", status="in_progress"))),
        ai(call("wait_for_person", {"question": "The left one or the right one?"}, "ask")),
    )
    stack.run(todos=MUG_TASK)
    assert render(stack.items()) == (
        "<todos>\n"
        "[x] t1 Go to the kitchen\n"
        '[?] t2 Pick up the red mug  waiting: "The left one or the right one?"\n'
        "[ ] t3 Come back to the operator\n"
        "[ ] t4 Hand over the mug  needs: hand_over\n"
        "</todos>"
    )

    # test_todo.py:558: the other markers and notes. needs: shows only on an open step.
    stack = make_stack(
        ai(
            todo_call(
                update("t1", status="failed", note="The kitchen door is closed."),
                update("t2", status="in_progress"),
                update("t4", status="cancelled", note="The operator left."),
            )
        ),
        then=DONE,
    )
    stack.run(todos=MUG_TASK)
    assert render(stack.items()) == (
        "<todos>\n"
        '[!] t1 Go to the kitchen  note: "The kitchen door is closed."\n'
        "[>] t2 Pick up the red mug\n"
        "[ ] t3 Come back to the operator\n"
        '[-] t4 Hand over the mug  note: "The operator left."\n'
        "</todos>"
    )

    # test_todo.py:577: the suffixes stay on one line, in order.
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress", note="Second try,\nleft hand."))),
        ai(call("wait_for_person", {"question": "Ready?"}, "ask")),
    )
    stack.run(todos=[MUG_TASK[3]])
    assert render(stack.items()).splitlines()[1] == (
        '[?] t1 Hand over the mug  waiting: "Ready?"  needs: hand_over  '
        'note: "Second try, left hand."'
    )

    # test_todo.py:588: an empty list.
    assert render([]) == ""
    assert public([]) == []
    assert json.dumps(public([])) == "[]"


def test_legacy_evidence_windows():
    # test_todo.py:390, changed on purpose: a requirement counts from when the step was
    # added, not from when it went in progress. So what must not count is a success from
    # before the step existed, which here means the previous task's.
    stack = make_stack(
        ai(drive("old")),
        DONE,
        DONE,
        DONE,
        ai(todo_call(update("t1", status="in_progress"))),
        ai(todo_call(update("t1", status="completed"), call_id="early")),
        then=DONE,
    )
    stack.run(todos=["First"])
    stack.run(todos=[NEEDS_DRIVE])
    refused_with(stack, "early", "t1 needs a successful go_to")

    # test_todo.py:421: another tool's success does not resolve a failure.
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"))),
        ai(drive("bad", "attic")),
        ai(call("say", {"text": "Trying."}, "spoke")),
        ai(todo_call(update("t1", status="completed"), call_id="early")),
        then=DONE,
    )
    stack.run(todos=["Go"])
    refused_with(stack, "early", "because go_to failed")

    # test_todo.py:430: a success before the failure does not resolve it.
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"))),
        ai(drive("good")),
        ai(drive("bad", "attic")),
        ai(todo_call(update("t1", status="completed"), call_id="early")),
        then=DONE,
    )
    stack.run(todos=["Go"])
    refused_with(stack, "early", "because go_to failed")

    # test_todo.py:446: a failure before the step was added does not count.
    stack = make_stack(
        ai(drive("bad", "attic")),
        ai(todo_call(add("Wave"))),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        then=DONE,
    )
    stack.run()
    assert stack.answer("done").status == "success"
    assert stack.statuses() == ["completed"]

    # test_todo.py:464: an op from waiting to in_progress starts a new attempt, unlike an
    # answer, which resumes the same one. See waiting_draft for why this uses a draft.
    failed = [{"tool": "pick", "id": "c1", "ok": False}]
    draft = waiting_draft(failed)
    draft.apply([TodoOp(op="update", id="t1", status="in_progress")])
    assert draft.items[0]["question"] == ""
    assert draft.items[0]["since"] == 1
    draft.apply([TodoOp(op="update", id="t1", status="completed")])
    assert draft.items[0]["status"] == "completed"
    # The control: without the new attempt, the same failure does block it.
    with pytest.raises(TodoRejected, match="because pick failed"):
        waiting_draft(failed).apply([TodoOp(op="update", id="t1", status="completed")])


def test_update_todos_never_reaches_the_ledger():
    # test_todo.py:477. Whether it worked or was refused, it is neither failure nor evidence.
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"), call_id="good")),
        ai(todo_call(update("t9", status="completed"), call_id="refused")),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        then=DONE,
    )
    stack.run(todos=["Go"])
    assert stack.answer("refused").status == "error"
    assert stack.answer("done").status == "success"
    assert not stack.state.get("ledger")
    assert stack.items()[0]["evidence"] == []


def test_public_snapshot_is_a_copy():
    # test_todo.py:606. A listener cannot reach the list through what it was given.
    items = parse_seed(["Go"])
    public(items)[0]["evidence"].append("call_x")
    assert items[0]["evidence"] == []

    stack = make_stack(ai(todo_call(update("t1", status="in_progress"))), then=DONE)
    stack.run(todos=["Go"])
    listened, changed = stack.published[-1]
    listened[0]["evidence"].append("call_x")
    changed[0]["evidence"].append("call_x")
    assert stack.items()[0]["evidence"] == []


def test_listener_can_read_supplied_snapshot():
    # test_todo.py:701. There is no list to read back any more, so what a listener needs
    # for JSON and for the reminder is in what it is given.
    seen = []
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"))),
        then=DONE,
        on_todos=lambda items, changed: seen.append((json.dumps(items), render(items))),
    )
    stack.run(todos=["Go"])
    assert seen[-1] == (json.dumps(public(stack.items())), render(stack.items()))
    assert seen[-1][1] == "<todos>\n[>] t1 Go\n</todos>"


def test_the_listener_gets_the_list_and_only_the_steps_whose_status_changed():
    # test_todo.py:632
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"), update("t2", note="The red one."))),
        ai(todo_call(add("Wipe the table"), call_id="u2")),
        ai(call("wait_for_person", {"question": "Which kitchen?"}, "ask")),
        then=DONE,
    )
    stack.run(todos=["Go to the kitchen", "Pick up the mug"])
    stack.run("The big one.")
    stack.run("Something new.")

    assert [([i["id"] for i in items], [i["id"] for i in changed]) for items, changed in stack.published] == [
        (["t1", "t2"], ["t1", "t2"]),
        (["t1", "t2"], ["t1"]),
        (["t1", "t2", "t3"], ["t3"]),
        (["t1", "t2", "t3"], ["t1"]),
        (["t1", "t2", "t3"], ["t1"]),
        ([], []),
    ]


def test_a_change_with_no_status_change_publishes_with_nothing_changed():
    # test_todo.py:656, changed on purpose: evidence used to arrive on its own, and now it
    # is recorded when the step completes. A note is what changes with no status change.
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"), call_id="start")),
        ai(drive("d1")),
        ai(todo_call(update("t1", note="The long way round."), call_id="noted")),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        then=DONE,
    )
    stack.run(todos=[NEEDS_DRIVE])

    noted, nothing = stack.published[2]
    assert noted[0]["note"] == "The long way round."
    assert nothing == []
    finished, changed = stack.published[3]
    assert finished[0]["evidence"] == ["d1"]
    assert [i["id"] for i in changed] == ["t1"]


def test_no_change_publishes_nothing():
    # test_todo.py:689: a run on an empty list, an empty seed, and a question with no list.
    stack = make_stack(DONE, ai(call("wait_for_person", {"question": "Anything else?"}, "ask")), then=DONE)
    stack.run()
    stack.run(todos=[])
    stack.run("No.")
    assert len(stack.model.received) == 3
    assert stack.published == []
