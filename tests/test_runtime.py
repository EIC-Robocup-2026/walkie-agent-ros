"""The whole create_agent stack on ainvoke, with FakeRobot, as_tools and a scripted model.

Claim map. It accounts for the 35 runtime spike claims, the 3 spike_00 regressions and
the 39 todo checks. Each is ported, changed on purpose, or obsolete with the reason.
File keys: R = this file, N = test_runtime_runner.py, M = test_runtime_model.py,
T = test_runtime_todos.py. Spikes: /var/tmp/wa-plan-scratch/runtime/ and
/var/tmp/wa-todo-seed/spike-own/spike_own.py.

spike_00, the three bugs (behaviour inverted, so each reproduction is now a regression)
  B1  stop leaves the drive running   N test_stop_ends_a_drive_through_the_token...
  B2  the stop's own run drives       N test_stop_ends_a_drive_through_the_token...
  B3  later calls run after a failure R test_a_failed_call_skips_the_later_calls_of_its_message

spike_01, tool calls
  1   default overlap                 R test_control_without_max_concurrency_calls_overlap
  2   serial with max_concurrency=1   R test_calls_of_one_message_run_one_at_a_time_in_order
  3a  failure skips later calls       R test_a_failed_call_skips_the_later_calls_of_its_message
  3b  nothing runs after a question   T test_calls_after_a_question_in_the_same_message_do_not_run
  5   a question ends the run         T test_wait_for_person_ends_the_run_and_the_answer_resumes...
  4a  a StepError is an error answer  T test_a_handled_tool_exception_counts_as_failed
  4b  an unexpected error ends a run  R test_control_without_the_error_middleware_a_tool_bug_ends...
  4c  the middleware answers instead  T test_an_unexpected_exception_counts_as_failed_and_stays...

spike_02, stop, latch, reset, history
  S1  stop ends the drive quickly     N test_stop_ends_a_drive_through_the_token... (bound 0.5 s)
  S2  no model call afterwards        N test_stop_ends_a_drive_through_the_token...
  S3  CHANGED, decision 6: the stop has no LLM run, so nothing can ask to drive. Same test.
  S4  the next task clears the latch  N test_the_latch_holds_until_the_next_task...
  S5a the checkpoint keeps every run  N test_every_run_stays_on_the_checkpoint...
  S5b CHANGED: reset deletes the old thread rather than leaving it. N test_reset_during_a_run...
  S6a a broken run leaves a call open R test_a_broken_run_is_repaired_and_its_stale_drive_stays...
  S6b the stale drive does not run    R test_a_broken_run_is_repaired_and_its_stale_drive_stays...
  S7  repair satisfies a strict model R test_a_broken_run_is_repaired_and_its_stale_drive_stays...

spike_03, todos, gate, binding
  T1  OBSOLETE: pinned why the library's TodoListMiddleware was rejected; decision 5 closed it.
  T2  OBSOLETE: the same, for its completion check.
  T3a the seed arrives by state       T test_the_first_model_call_sees_the_seed_as_a_trailing...
  T3b dropping a seeded step refused  T test_the_rules_for_seeded_steps
  T3c early completion refused        T test_the_rules_for_seeded_steps
  T3d failed without a note refused   T test_each_refusal_says_what_to_do
  T3e completed after the drive       T test_one_drive_cannot_complete_two_steps
  T4  two gates, then the run ends    T test_gate_sends_the_llm_back_at_most_twice_per_task
  T4b the next task has its own gates T test_gate_sends_the_llm_back_at_most_twice_per_task
  T5  OBSOLETE: no mid-task tool binding in this design; tools are fixed at build_agent.
  T6  reminder sent and never saved   T test_every_call_gets_the_current_list_and_it_is_never_saved
  T7  CHANGED: the spike's hardcoded True is a real assertion, counting the todo tools.
                                      R test_the_tool_warning_counts_the_two_todo_tools

spike_04, deadline
  D1  SDK timeout ends a silent call  M test_d1_...
  D2  it does not end a dripping one  M test_d2_... (short negative pin)
  D3  Deadline then fallback          M test_d3_d4_... and R test_a_fallback_answers_after_the...
  D4  the client lets go              M test_d3_d4_...
  D5  OBSOLETE: pinned the sync path, and the runtime is async only. ScriptedModel raises on it.
  D6  async and max_concurrency=1     R test_calls_of_one_message_run_one_at_a_time_in_order

spike_own, the 39 todo checks, all in T unless marked. CHANGED entries say how.
  R1a seed as trailing reminder       test_the_first_model_call_sees_the_seed_as_a_trailing...
  R1b source survives                 test_an_operator_seed_from_a_task_message_keeps_its_source
  R1c a new seed replaces the list    test_a_new_seed_replaces_the_old_list_and_restarts_ids
  R2a seed by state                   test_the_first_model_call_sees_the_seed_as_a_trailing...
  R4  drop                            test_the_rules_for_seeded_steps
  R4  reorder                         test_the_rules_for_seeded_steps
  R4  reword                          test_the_rules_for_seeded_steps
  R5a CHANGED: an invalid status is refused by the status check, not a schema literal.
                                      test_the_rules_for_seeded_steps
  R4  extra keys                      test_the_rules_for_seeded_steps
  R4  early completed                 test_the_rules_for_seeded_steps
  R4  insert accepted                 test_the_rules_for_seeded_steps
  R5b question sets waiting and ends  test_wait_for_person_ends_the_run_and_the_answer_resumes...
  R7a every call gets the reminder    test_every_call_gets_the_current_list_and_it_is_never_saved
  R3a an answer keeps the list        test_wait_for_person_ends_the_run_and_the_answer_resumes...
  R5c cancelled needs a note          R test_a_whole_seeded_task_ends_with_the_expected_statuses
  R5d completed after a failed tool   test_completion_waits_until_the_failed_tool_succeeds
  R5e final statuses                  R test_a_whole_seeded_task_ends_with_the_expected_statuses
  R6a publish after each change       test_publishing_follows_every_accepted_change_and_no_refused...
  R6b CHANGED: only accepted changes publish, and the listener gets (items, changed).
                                      test_publishing_follows_every_accepted_change_and_no_refused...
  R6c listener failure                test_a_listener_that_raises_does_not_affect_the_list
  R2b none is no seed, [] is empty    test_an_empty_seed_empties_the_list_and_none_is_no_seed
  R3b a new seed wins over a question test_a_seed_wins_over_a_pending_question
  R8a at most two gates               test_gate_sends_the_llm_back_at_most_twice_per_task
  R3c an unseeded task starts empty   test_an_unseeded_task_does_not_inherit_the_last_tasks_open...
  R5f one in progress                 test_only_one_step_can_be_in_progress
  R5g a note excuses a failure        test_a_note_excuses_a_failure_but_not_a_missing_required_tool
  R4  own steps are not locked        test_the_agents_own_steps_are_not_locked
  R3d reset                           N test_reset_during_a_run_deletes_the_thread...
  R5h CHANGED: a failed bad-arguments call counts, and refused calls do not.
                                      test_a_call_with_bad_arguments_counts_as_failed and
                                      test_a_call_that_never_ran_does_not_poison_completion
  R8b CHANGED: no gate while waiting, because a question ends the run first.
                                      test_a_question_ends_the_run_with_no_gate_and_the_answer...
  R8c the limit stops a runaway       R test_a_runaway_tool_loop_stops_at_the_limit...
  R8d todos_first counts gate calls   R test_the_limit_counts_gate_calls
  R8d limit_first overruns (control)  R test_control_with_the_limit_first_gate_calls_overrun_it
  R8e two calls, no channel conflict  test_two_successful_calls_in_one_message_do_not_collide
  R8f control: they overlap otherwise R test_control_without_max_concurrency_calls_overlap
  R8g CHANGED: only the duplicate update is refused, and it skips the later calls; the
      first is accepted.              test_only_the_first_update_todos_call_in_a_message_is_applied
  R8h OBSOLETE as a crash reproduction of the old spike; replaced by R8e's no-collision test.
  R8i OBSOLETE as a crash reproduction of the old spike; replaced by R8g's test.
  Schema flat tool schemas            test_the_tool_schemas_are_flat_and_hide_the_injected_runtime

tests/test_graph.py integration cases, all in T, where each test carries its line number
  419 test_completion_waits_until_the_failed_tool_succeeds
  459 test_a_handled_tool_exception_counts_as_failed
  492 test_a_call_with_bad_arguments_counts_as_failed
  518 test_only_the_first_update_todos_call_in_a_message_is_applied
  630 test_gate_sends_the_llm_back_at_most_twice_per_task
  687 test_no_gate_without_open_items
  703 test_a_question_ends_the_run_with_no_gate_and_the_answer_keeps_the_gate_budget
  716 test_wait_for_person_ends_the_run_and_the_answer_resumes_the_item
  754 test_calls_after_a_question_in_the_same_message_do_not_run
  791 test_failed_wait_for_person_does_not_end_the_run
"""

import asyncio
import json
import logging
import re
import time

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from runtime_harness import (
    ScriptedModel,
    add,
    ai,
    assert_every_call_answered,
    bad_call,
    call,
    drive,
    make_stack,
    todo_call,
    update,
)

from walkie_agent.adapter import as_tools
from walkie_agent.atomics import DEFAULT
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.runtime import SYSTEM_PROMPT, ToolLog, build_middleware
from walkie_agent.runtime.middleware import INTERRUPTED, UNREADABLE_ARGUMENTS, RepairHistory
from walkie_agent.runtime.todos import parse_seed

DONE = AIMessage("Done.")
SERIAL_RUNS = 20


async def hang(_messages):
    await asyncio.sleep(30)


# -- The order of the stack -------------------------------------------------------


def test_the_whole_middleware_list_is_pinned(tmp_path):
    ctx = FakeRobot().ctx

    def names(**build):
        return [
            type(m).__name__ for m in build_middleware(ctx, run_limit=3, deadline_s=1.0, **build)
        ]

    assert names() == [
        "StopLatch",
        "SerialGuard",
        "WalkieTodos",
        "EndAfterQuestion",
        "RepairHistory",
        "ModelCallLimitMiddleware",
        "Deadline",
        "ToolErrorMiddleware",
    ]
    assert names(fallback=ScriptedModel(), tool_log=ToolLog(tmp_path / "log.jsonl")) == [
        "ToolLog",
        "StopLatch",
        "SerialGuard",
        "WalkieTodos",
        "EndAfterQuestion",
        "RepairHistory",
        "ModelCallLimitMiddleware",
        "ModelFallbackMiddleware",
        "Deadline",
        "ToolErrorMiddleware",
    ]


def limit_first_agent(model: ScriptedModel, robot: FakeRobot, run_limit: int):
    """The real stack with the limit moved to the front, to show why it does not go there."""
    stack = build_middleware(robot.ctx, run_limit=run_limit, deadline_s=5.0)
    limit = next(m for m in stack if isinstance(m, ModelCallLimitMiddleware))
    stack.remove(limit)
    return create_agent(
        model, as_tools(DEFAULT, robot.ctx), middleware=[limit, *stack], checkpointer=InMemorySaver()
    )


def invoke(agent, **inputs) -> dict:
    config = {"configurable": {"thread_id": "control"}, "max_concurrency": 1}
    payload = {"messages": [HumanMessage("Do it.")], "todo_seed": None, **inputs}
    return asyncio.run(agent.ainvoke(payload, config))


# -- The limit --------------------------------------------------------------------


def test_a_model_that_repeats_invalid_calls_stops_at_the_run_limit():
    stack = make_stack(then=ai(invalid=[bad_call("bad")]), run_limit=3)
    result = stack.run()
    assert len(stack.model.received) == 3
    assert "run limit (3/3)" in result["messages"][-1].content
    assert_every_call_answered(result["messages"])


def test_a_model_that_repeats_invalid_calls_with_an_open_list_stops_at_the_run_limit():
    stack = make_stack(then=ai(invalid=[bad_call("bad")]), run_limit=3)
    stack.run(todos=["Open step"])
    assert len(stack.model.received) == 3


def test_control_with_the_limit_first_invalid_calls_overrun_it():
    # The numbers verify.json measured: 7 calls under run_limit=3.
    robot = FakeRobot()
    model = ScriptedModel(script=[ai(invalid=[bad_call(f"bad{n}")]) for n in range(6)], then=DONE)
    invoke(limit_first_agent(model, robot, run_limit=3))
    assert len(model.received) == 7


def test_a_runaway_tool_loop_stops_at_the_limit_and_the_next_task_has_a_new_budget():
    # R8c at a limit of 5. test_graph.py:603 stopped this with a recursion limit.
    stack = make_stack(then=ai(drive("d")), run_limit=5)
    first = stack.run()
    assert len(stack.model.received) == 5
    assert first["messages"][-1].content.startswith("Model call limits exceeded")
    second = stack.run("Again.")
    assert len(stack.model.received) == 10
    assert "run limit (5/5)" in second["messages"][-1].content


def patrol() -> list[AIMessage]:
    return [ai(drive(f"d{n}")) for n in range(3)]


def test_the_limit_counts_gate_calls():
    # R8d todos_first: 3 tool calls, then text and its gates, at a limit of 5.
    stack = make_stack(*patrol(), then=DONE, run_limit=5)
    stack.run(todos=["Patrol the hall"])
    assert len(stack.model.received) == 5


def test_control_with_the_limit_first_gate_calls_overrun_it():
    # R8d limit_first: a gate's jump skips the limit's after_model, so it never counts.
    robot = FakeRobot()
    model = ScriptedModel(script=patrol(), then=DONE)
    agent = limit_first_agent(model, robot, run_limit=5)
    invoke(agent, todo_seed=parse_seed(["Patrol the hall"]))
    assert len(model.received) == 6


# -- Invalid calls and unknown tools ----------------------------------------------


def test_a_call_with_unreadable_arguments_is_answered_and_the_model_tries_again():
    # test_graph.py:349. An open item, so that ending the run here would send the gate
    # message right after an unanswered call.
    stack = make_stack(
        ai(invalid=[bad_call("bad")]),
        ai(drive("ok")),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        DONE,
    )
    stack.run(todos=["Go to the sofa"])

    retry = stack.model.received[1]
    assert [type(m) for m in retry] == [
        SystemMessage, HumanMessage, AIMessage, ToolMessage, HumanMessage
    ]
    answer = stack.answer("bad")
    assert (answer.content, answer.status) == (UNREADABLE_ARGUMENTS, "error")
    assert stack.verbs() == ["base.go_to"]
    assert len(stack.model.received) == 4
    assert stack.state["ledger"] == [{"tool": "go_to", "id": "ok", "ok": True}]


def test_valid_calls_of_a_mixed_message_run_first_and_every_call_is_answered():
    stack = make_stack(
        ai(drive("ok"), call("say", {"text": "Going."}, "spoke"), invalid=[bad_call("bad")]),
        DONE,
    )
    stack.run()
    assert stack.verbs() == ["base.go_to", "speech.say"]
    assert len(stack.model.received) == 2  # one model call for the batch, no immediate retry
    assert stack.answer("ok").status == "success"
    assert stack.answer("spoke").status == "success"
    assert stack.answer("bad").content == UNREADABLE_ARGUMENTS


def test_two_invalid_calls_are_both_answered():
    stack = make_stack(ai(invalid=[bad_call("one"), bad_call("two")]), DONE)
    stack.run()
    assert stack.answer("one").status == stack.answer("two").status == "error"


def test_an_unknown_tool_gets_an_error_and_the_run_goes_on():
    # test_graph.py:389. An unknown name is not a failed tool, so the ledger stays clean.
    stack = make_stack(
        ai(call("fly", {}, "fly")),
        ai(todo_call(update("t1", status="completed"), call_id="done")),
        DONE,
    )
    stack.run(todos=["Fly to the kitchen"])
    assert stack.answer("fly").status == "error"
    assert "fly is not a valid tool" in stack.answer("fly").content
    assert stack.answer("done").status == "success"
    assert stack.state.get("ledger", []) == []
    assert len(stack.model.received) == 3


def test_bad_arguments_get_the_nodes_answer_and_the_run_goes_on():
    stack = make_stack(ai(call("go_to", {"place": 7}, "bad")), DONE)
    stack.run()
    assert stack.answer("bad").status == "error"
    assert stack.verbs() == []
    assert len(stack.model.received) == 2


# -- One message's calls ----------------------------------------------------------


def test_a_failed_call_skips_the_later_calls_of_its_message():
    # B3, 3a, decision 7. plan.md:21, :487 and :507 all name this as the fix.
    stack = make_stack(
        ai(drive("bad", "attic"), call("say", {"text": "Arrived."}, "later"), drive("last")),
        DONE,
    )
    stack.run()
    assert stack.answer("bad").status == "error"
    for call_id in ("later", "last"):
        assert stack.answer(call_id).status == "error"
        assert stack.answer(call_id).content.startswith("Not run: go_to failed earlier")
    assert stack.verbs() == ["base.go_to"]
    assert stack.robot.said == []
    assert len(stack.model.received) == 2


def test_a_failed_say_goes_back_to_the_model_and_the_next_message_runs():
    robot = FakeRobot()
    robot.faults["speech.say"] = StepError(ErrorCode.UNAVAILABLE, "The speaker is off.")
    stack = make_stack(
        ai(call("say", {"text": "Hello."}, "hi"), drive("skipped")),
        ai(drive("next")),
        DONE,
        robot=robot,
    )
    stack.run()
    assert stack.answer("hi").status == "error"
    assert stack.answer("skipped").content.startswith("Not run: say failed earlier")
    assert stack.answer("next").status == "success"
    assert stack.verbs() == ["speech.say", "base.go_to"]
    assert len(stack.model.received) == 3


def test_a_refused_or_duplicate_update_todos_skips_later_calls():
    refused = make_stack(ai(todo_call(update("t9", status="pending")), drive("later")), DONE)
    refused.run()
    assert refused.answer("later").content.startswith("Not run: update_todos failed earlier")
    assert refused.verbs() == []

    duplicate = make_stack(
        ai(
            todo_call(add("One")),
            todo_call(add("Two"), call_id="u2"),
            call("say", {"text": "Both steps were added."}, "later"),
        ),
        DONE,
        DONE,
        DONE,
    )
    duplicate.run()
    assert duplicate.answer("u2").status == "error"
    assert duplicate.answer("later").content.startswith("Not run: update_todos failed earlier")
    assert duplicate.robot.said == []
    assert [i["content"] for i in duplicate.items()] == ["One"]


def slow_tools(spans: list[tuple[str, float, float]]) -> list:
    """Three sync tools that take a moment and record when they ran."""

    def make(name: str):
        def slow(ctx) -> str:
            """Take a moment."""
            started = time.monotonic()
            time.sleep(0.02)
            spans.append((name, started, time.monotonic()))
            return "ok"

        slow.__name__ = name
        return slow

    return [make(name) for name in ("a", "b", "c")]


def three_calls() -> AIMessage:
    return ai(call("a", {}, "1"), call("b", {}, "2"), call("c", {}, "3"))


def test_calls_of_one_message_run_one_at_a_time_in_order():
    # 2, D6. Observed behaviour of the installed versions, not documented, and SerialGuard,
    # WalkieTodos' batch and the ledger's order all rest on it. Rerun on a dependency update.
    for _ in range(SERIAL_RUNS):
        spans: list[tuple[str, float, float]] = []
        stack = make_stack(three_calls(), DONE, tools=slow_tools(spans))
        stack.run()
        assert [name for name, _, _ in spans] == ["a", "b", "c"]
        assert all(later[1] >= earlier[2] for earlier, later in zip(spans, spans[1:]))


def test_control_without_max_concurrency_calls_overlap():
    # 1, R8f: Session.config's max_concurrency=1 is what makes the test above pass.
    spans: list[tuple[str, float, float]] = []
    stack = make_stack(three_calls(), DONE, tools=slow_tools(spans))
    payload = {"messages": [HumanMessage("Go.")], "todo_seed": None}
    asyncio.run(stack.agent.ainvoke(payload, {"configurable": {"thread_id": "overlap"}}))
    ordered = sorted(spans, key=lambda span: span[1])
    assert any(later[1] < earlier[2] for earlier, later in zip(ordered, ordered[1:]))


def test_control_without_the_error_middleware_a_tool_bug_ends_the_run():
    # 4b: why tool_error_middleware stays in the stack.
    def broken(ctx) -> str:
        """Break."""
        raise RuntimeError("A bug.")

    robot = FakeRobot()
    agent = create_agent(
        ScriptedModel(script=[ai(call("broken", {}, "b"))]), as_tools([broken], robot.ctx)
    )
    with pytest.raises(RuntimeError, match="A bug"):
        asyncio.run(agent.ainvoke({"messages": [HumanMessage("Go.")]}))


# -- A whole list -----------------------------------------------------------------


def test_a_whole_seeded_task_ends_with_the_expected_statuses():
    # R5c, R5e
    no_note = (update("t1", status="completed"), update("t2", status="cancelled"))
    with_note = (update("t1", status="completed"), update("t2", status="cancelled", note="None."))
    stack = make_stack(
        ai(todo_call(update("t1", status="in_progress"), call_id="start")),
        ai(drive("d")),
        ai(todo_call(*no_note, call_id="no_note")),
        ai(todo_call(*with_note, call_id="with_note")),
        ai(todo_call(update("t3", status="completed"), call_id="last")),
        DONE,
    )
    stack.run(
        todos=[{"content": "Go there", "requires_tool": "go_to"}, "Ask about allergies", "Serve"]
    )
    assert stack.answer("no_note").status == "error"
    assert "needs a note" in stack.answer("no_note").content
    assert stack.answer("with_note").status == "success"
    assert stack.statuses() == ["completed", "cancelled", "completed"]
    assert not [m for m in stack.messages if str(m.content).startswith("[todo]")]
    assert len(stack.model.received) == 6


# -- Repair -----------------------------------------------------------------------


def test_a_broken_run_is_repaired_and_its_stale_drive_stays_unrun():
    # S6a, S6b, S7: what is left when a run dies between the model and the tools.
    stack = make_stack(ai(drive("stale")), DONE)

    async def break_after_the_model() -> None:
        payload = {"messages": [HumanMessage("Go.")], "todo_seed": None}
        stream = stack.agent.astream(payload, stack.config, stream_mode="updates")
        async for chunk in stream:
            if "WalkieTodos.after_model" in chunk:
                break
        await stream.aclose()

    asyncio.run(break_after_the_model())
    snapshot = stack.agent.get_state(stack.config)
    assert "tools" in snapshot.next
    assert stack.verbs() == []

    stack.run("Next.")
    assert stack.verbs() == []
    assert stack.answer("stale").content == INTERRUPTED
    assert len(stack.model.received) == 2  # the strict model accepted this history


def fixed_by_repair(messages: list) -> list:
    result = RepairHistory().before_agent({"messages": messages}, None)
    return messages if result is None else result["messages"][1:]


def answer(call_id: str, text: str = "ok") -> ToolMessage:
    return ToolMessage(text, tool_call_id=call_id)


def test_repair_answers_every_unanswered_call():
    # test_graph.py:253
    fixed = fixed_by_repair([ai(drive("a"), drive("b")), HumanMessage("Next.")])
    assert [getattr(m, "tool_call_id", None) for m in fixed] == [None, "a", "b", None]
    assert fixed[1].content == INTERRUPTED
    assert fixed[1].status == "error"


def test_repair_adds_missing_answers_after_the_existing_ones():
    # test_graph.py:272
    fixed = fixed_by_repair([ai(drive("a"), drive("b")), answer("b"), HumanMessage("Next.")])
    assert [getattr(m, "tool_call_id", None) for m in fixed] == [None, "b", "a", None]
    assert_every_call_answered(fixed)


def test_repair_answers_calls_at_the_end_of_the_history():
    # test_graph.py:286
    fixed = fixed_by_repair([HumanMessage("Go."), ai(drive("a"))])
    assert fixed[-1].tool_call_id == "a"


def test_repair_answers_calls_with_unreadable_arguments():
    # test_graph.py:327
    fixed = fixed_by_repair([ai(invalid=[bad_call("bad")])])
    assert fixed[-1].tool_call_id == "bad"


def test_repair_is_idempotent():
    # test_graph.py:312
    fixed = fixed_by_repair([ai(drive("a")), HumanMessage("Next.")])
    assert RepairHistory().before_agent({"messages": fixed}, None) is None


# -- The deadline -----------------------------------------------------------------


def test_a_hung_model_times_out_with_no_fallback_configured():
    stack = make_stack(hang, deadline_s=0.1)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        stack.run()
    assert time.monotonic() - started < 1


def test_a_fallback_answers_after_the_deadline():
    # D3 with a fake primary.
    fallback = ScriptedModel(then=AIMessage("From the fallback."))
    stack = make_stack(hang, deadline_s=0.1, fallback=fallback)
    started = time.monotonic()
    result = stack.run()
    assert time.monotonic() - started < 1
    assert result["messages"][-1].content == "From the fallback."


def test_each_attempt_of_a_fallback_chain_has_its_own_deadline():
    fallback = ScriptedModel(then=hang)
    stack = make_stack(hang, deadline_s=0.2, fallback=fallback)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        stack.run()
    assert len(fallback.received) == 1
    assert 0.35 < time.monotonic() - started < 1.5


# -- The tool log -----------------------------------------------------------------


def log_lines(log: ToolLog) -> list[dict]:
    return [json.loads(line) for line in log.path.read_text().splitlines()]


def test_the_tool_log_sees_commands_messages_and_refusals_and_changes_nothing(tmp_path):
    log = ToolLog(tmp_path / "not" / "there" / "yet" / "tools.jsonl")
    stack = make_stack(
        ai(todo_call(add("Go there"))),
        ai(drive("ok")),
        ai(drive("bad", "attic"), call("say", {"text": "Hi"}, "skipped"), call("fly", {}, "fly")),
        ai(todo_call(update("t1", status="completed", note="Close enough."), call_id="u2")),
        DONE,
        tool_log=log,
    )
    stack.run()

    lines = log_lines(log)
    assert [(r["tool"], r["call_id"], r["status"]) for r in lines] == [
        ("update_todos", "u1", "success"),
        ("go_to", "ok", "success"),
        ("go_to", "bad", "error"),
        ("say", "skipped", "error"),
        ("fly", "fly", "error"),
        ("update_todos", "u2", "success"),
    ]
    assert {r["session"] for r in lines} == {stack.thread_id}
    assert lines[0]["result"] == stack.answer("u1").content
    assert lines[1]["args"] == {"place": "sofa"}
    assert lines[1]["seconds"] >= 0
    assert lines[3]["result"].startswith("Not run: go_to failed earlier")
    # The Command outputs reached the state untouched.
    assert stack.statuses() == ["completed"]
    assert [entry["id"] for entry in stack.state["ledger"]] == ["ok", "bad"]


def test_a_log_failure_does_not_lose_the_tool_result(tmp_path, caplog):
    blocker = tmp_path / "a_file"
    blocker.write_text("")
    stack = make_stack(ai(drive("ok")), DONE, tool_log=ToolLog(blocker / "tools.jsonl"))
    with caplog.at_level(logging.ERROR, logger="walkie_agent.runtime.middleware"):
        stack.run()
    assert stack.answer("ok").status == "success"
    assert stack.state["ledger"][0]["ok"] is True
    assert any("Could not write to the tool log" in message for message in caplog.messages)


# -- The system prompt and the tool count -----------------------------------------


def test_the_system_prompt_names_only_tools_the_default_set_binds():
    stack = make_stack(DONE)
    stack.run()
    assert stack.model.received[0][0] == SystemMessage(SYSTEM_PROMPT)
    named = set(re.findall(r"`(\w+)`", SYSTEM_PROMPT))
    assert {"say", "list_places", "go_to", "update_todos", "wait_for_person"} <= named
    assert named <= set(stack.model.bound[0])
    assert not {"speak", "list_locations", "navigate_to_location", "get_todos"} & named


def noop_tools(count: int) -> list:
    def make(n: int):
        def noop(ctx) -> str:
            """Do nothing."""
            return "ok"

        noop.__name__ = f"noop_{n}"
        return noop

    return [make(n) for n in range(count)]


def warnings_of(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "walkie_agent.runtime.agent"]


def test_the_tool_warning_counts_the_two_todo_tools(caplog):
    # T7, and test_graph.py:563. The warning is at build time, as the tools are fixed there.
    with caplog.at_level(logging.WARNING, logger="walkie_agent.runtime.agent"):
        make_stack(DONE, tools=noop_tools(18))
        assert warnings_of(caplog) == []

        stack = make_stack(DONE, tools=noop_tools(19))
        assert len(warnings_of(caplog)) == 1
        assert warnings_of(caplog)[0].startswith("21 tools are bound, more than 20")
    stack.run()
    assert len(stack.model.bound[0]) == 21
    assert {"update_todos", "wait_for_person"} <= set(stack.model.bound[0])
