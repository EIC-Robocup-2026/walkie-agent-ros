"""Compaction (PK-74): where a history splits, what the summarizer reads, and the Runner
moving a long history to a new thread while its queue is empty.
"""

import asyncio
import json
import logging
import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from runtime_harness import ScriptedModel, ai, call, make_stack

from walkie_agent.runtime import STOP_LINE, Runner, ToolLog
from walkie_agent.runtime.compaction import (
    ANSWER_CHARS,
    SUMMARY_PREFIX,
    SUMMARY_PROMPT,
    Compactor,
    transcript,
)

DONE = AIMessage("Done.")
# About 100 approximate tokens each, so two tasks pass AFTER_TOKENS and one does not.
MUG = "Please remember that the blue mug on the kitchen counter is mine. " * 6
PLATE = "Please remember that the white plate in the dish rack is Ben's. " * 6
AFTER_TOKENS = 200
SUMMARY = AIMessage("Ann said the blue mug on the counter is hers.")


def tokens(*messages):
    return count_tokens_approximately(messages)


def humans(messages):
    return [m.text for m in messages if isinstance(m, HumanMessage)]


@pytest.fixture
def serve():
    """Starts a Runner with a Compactor over a new stack, and checks that it shuts down."""
    runners = []

    def start(*script, summarizer, after_tokens=AFTER_TOKENS, tool_log=None):
        stack = make_stack(*script, then=DONE, tool_log=tool_log)
        compactor = Compactor(summarizer, after_tokens=after_tokens, keep_tokens=1)
        stack.runner = Runner(
            stack.agent, stack.robot.ctx, on_reply=stack.replies.append, compactor=compactor
        )
        runners.append(stack.runner)
        stack.runner.start()
        return stack

    yield start
    for runner in runners:
        assert runner.shutdown(10)


def run(stack, *texts):
    for text in texts:
        stack.runner.submit(text)
        assert stack.runner.wait_idle(10), "The Runner did not finish in time."


# -- Where a history splits ---------------------------------------------------


def test_a_history_below_the_threshold_or_of_one_turn_is_not_split():
    history = [
        HumanMessage(MUG),
        ai(call("go_to", {"place": "kitchen"}, "g1")),
        ToolMessage("Arrived.", tool_call_id="g1", name="go_to"),
        AIMessage("I am in the kitchen."),
    ]
    model = ScriptedModel()
    assert Compactor(model, after_tokens=tokens(*history) + 1, keep_tokens=1).split(history) == 0
    # One turn has no person's message to start the kept part at, after its first.
    assert Compactor(model, after_tokens=1, keep_tokens=1).split(history) == 0


def test_a_split_starts_at_a_persons_message_so_calls_keep_their_answers():
    history = [
        HumanMessage(MUG),
        ai(call("go_to", {"place": "kitchen"}, "g1")),
        ToolMessage("Arrived.", tool_call_id="g1", name="go_to"),
        AIMessage("I am in the kitchen."),
        HumanMessage("Bring the mug."),
        ai(call("pick_object", {"name": "mug"}, "p1")),
        ToolMessage("Picked.", tool_call_id="p1", name="pick_object"),
        HumanMessage("[todo] Open items: t1."),
        AIMessage("Here it is."),
    ]
    # Only the last message is needed, but the runtime's note does not start a turn,
    # so the kept part goes back to the person's request and the call it led to.
    compactor = Compactor(ScriptedModel(), after_tokens=1, keep_tokens=tokens(history[-1]))
    assert compactor.split(history) == 4


def test_a_split_that_would_summarize_only_an_earlier_summary_is_no_split():
    history = [
        HumanMessage(f"{SUMMARY_PREFIX} Ann owns the blue mug."),
        HumanMessage(MUG),
        AIMessage("Noted."),
    ]
    model = ScriptedModel()
    assert Compactor(model, after_tokens=1, keep_tokens=1).split(history) == 0
    # A newer turn lets the summary and the turn after it be summarized together.
    longer = [*history, HumanMessage("And the plate?"), AIMessage("That one is Ben's.")]
    assert Compactor(model, after_tokens=1, keep_tokens=1).split(longer) == 3


# -- What the summarizer reads and answers -------------------------------------


def test_the_transcript_says_who_said_what_and_leaves_out_the_runtimes_notes():
    history = [
        HumanMessage(f"{SUMMARY_PREFIX} Ann owns the blue mug."),
        HumanMessage("Bring my mug."),
        ai(
            call("say", {"text": "Just a moment."}, "s1"),
            call("go_to", {"place": "kitchen"}, "g1"),
        ),
        ToolMessage("Said: Just a moment.", tool_call_id="s1", name="say"),
        ToolMessage("x" * 400, tool_call_id="g1", name="go_to", status="error"),
        HumanMessage("[todo] Open items: t1."),
        ai(call("say", {"text": "I am in the kitchen."}, "s2")),
        ToolMessage("The speaker hung.", tool_call_id="s2", name="say", status="error"),
        ai(call("wait_for_person", {"question": "Which mug?"}, "q1")),
        ToolMessage("Asked: Which mug?", tool_call_id="q1", name="wait_for_person"),
        HumanMessage("The blue one."),
        AIMessage("Here it is."),
    ]
    assert transcript(history).splitlines() == [
        "Earlier summary: Ann owns the blue mug.",
        "Person: Bring my mug.",
        "Walkie said: Just a moment.",
        'Walkie called go_to with {"place": "kitchen"}',
        f"  go_to failed: {'x' * ANSWER_CHARS} [cut]",
        # A line that never played is not claimed as said.
        "Walkie tried to say: I am in the kitchen.",
        "  say failed: The speaker hung.",
        "Walkie asked: Which mug?",
        "Person: The blue one.",
        "Walkie said: Here it is.",
    ]


def test_a_summary_is_a_human_message_with_the_prefix():
    summarizer = ScriptedModel(script=[AIMessage("Ann owns\n the blue mug.")])
    older = [HumanMessage("The blue mug is mine."), AIMessage("Noted.")]
    summary = asyncio.run(Compactor(summarizer, 1, 1).summarize(older))

    assert summary == HumanMessage(f"{SUMMARY_PREFIX} Ann owns the blue mug.")
    assert summarizer.received == [
        [SystemMessage(SUMMARY_PROMPT), HumanMessage(transcript(older))]
    ]
    assert summarizer.bound == [[]]  # the summarizer never gets tools


def test_an_empty_or_late_summary_raises_so_it_never_replaces_a_history():
    older = [HumanMessage("The blue mug is mine."), AIMessage("Noted.")]
    with pytest.raises(ValueError, match="no text"):
        asyncio.run(Compactor(ScriptedModel(script=[AIMessage(" ")]), 1, 1).summarize(older))
    late = ScriptedModel(script=[SUMMARY], delay_s=5)
    with pytest.raises(TimeoutError):
        asyncio.run(Compactor(late, 1, 1, timeout_s=0.05).summarize(older))


# -- The Runner ------------------------------------------------------------------


def test_a_long_history_moves_to_a_new_thread_as_a_summary_and_its_newest_turn(
    serve, tmp_path
):
    log = ToolLog(tmp_path / "tools.jsonl")
    summarizer = ScriptedModel(script=[SUMMARY])
    stack = serve(
        DONE, DONE, ai(call("say", {"text": "The blue mug."}, "s1")), DONE,
        summarizer=summarizer, tool_log=log,
    )
    first = stack.runner.session.thread_id

    run(stack, MUG)
    assert stack.runner.session.thread_id == first
    run(stack, PLATE)
    # The old thread is gone, and the conversation goes on under its first id.
    assert stack.runner.session.thread_id != first
    assert stack.runner.session.id == first
    assert stack.agent.get_state({"configurable": {"thread_id": first}}).values == {}
    assert stack.runner.session.config()["metadata"]["langfuse_session_id"] == first
    read = summarizer.received[0][1].text
    assert f"Person: {MUG.strip()}" in read and "Walkie said: Done." in read
    assert PLATE.strip() not in read

    run(stack, "Which one is mine?")
    sent = humans(stack.model.received[2])
    assert sent[:3] == [f"{SUMMARY_PREFIX} {SUMMARY.text}", PLATE, "Which one is mine?"]
    assert MUG not in sent
    records = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [(r["call_id"], r["session"]) for r in records] == [("s1", first)]


def test_a_pending_question_keeps_the_thread_until_it_is_answered(serve):
    summarizer = ScriptedModel(script=[SUMMARY])
    stack = serve(
        ai(call("wait_for_person", {"question": "Which mug?"}, "q1")),
        DONE,
        summarizer=summarizer,
        after_tokens=100,
    )
    first = stack.runner.session.thread_id

    run(stack, MUG)
    assert stack.state["question"] == "Which mug?"
    assert (stack.runner.session.thread_id, summarizer.received) == (first, [])
    run(stack, "The blue one.")
    assert stack.runner.session.thread_id != first
    assert len(summarizer.received) == 1


def test_new_input_cancels_a_summary_and_the_history_stays(serve, caplog):
    started = threading.Event()

    async def slow(messages):
        started.set()
        await asyncio.sleep(30)

    summarizer = ScriptedModel(script=[slow, SUMMARY])
    stack = serve(summarizer=summarizer)
    first = stack.runner.session.thread_id
    caplog.set_level(logging.INFO, logger="walkie_agent.runtime.runner")

    stack.runner.submit(MUG)
    stack.runner.submit(PLATE)
    assert started.wait(10)
    run(stack, "Which one is mine?")

    # The third task ran on the whole history, and the next quiet moment compacted it.
    assert humans(stack.model.received[2])[:3] == [MUG, PLATE, "Which one is mine?"]
    assert any("New input came in" in m for m in caplog.messages)
    assert len(summarizer.received) == 2
    assert stack.runner.session.thread_id != first


def test_a_shutdown_during_a_summary_waits_for_it_to_end(serve):
    started, ended = threading.Event(), threading.Event()

    async def slow(messages):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            await asyncio.sleep(0.2)  # cleanup that outlasts one loop step, like a connection
            ended.set()

    stack = serve(summarizer=ScriptedModel(script=[slow]))
    stack.runner.submit(MUG)
    stack.runner.submit(PLATE)
    assert started.wait(10)
    assert stack.runner.shutdown(10)
    assert ended.is_set()


def test_the_todo_list_moves_with_the_history_so_a_stop_and_a_task_still_update_it(serve):
    stack = serve(summarizer=ScriptedModel(then=SUMMARY))
    run(stack, MUG)
    first = stack.runner.session.thread_id
    # The model never works the step, so the gate gives up and leaves it open.
    stack.runner.submit(PLATE, todos=["Fetch the plate"])
    assert stack.runner.wait_idle(10)
    assert stack.runner.session.thread_id != first
    assert [item["status"] for item in stack.published[-1][0]] == ["pending"]

    stack.runner.stop("stop")
    assert stack.runner.wait_idle(10)
    assert [item["status"] for item in stack.published[-1][0]] == ["cancelled"]
    run(stack, "Which one is mine?")
    assert stack.published[-1] == ([], [])


def test_a_failed_summary_keeps_the_history(serve, caplog):
    summarizer = ScriptedModel(then=RuntimeError("The summarizer is down."))
    stack = serve(summarizer=summarizer)
    first = stack.runner.session.thread_id

    run(stack, MUG, PLATE, "Which one is mine?")
    assert stack.runner.session.thread_id == first
    assert humans(stack.model.received[2])[:3] == [MUG, PLATE, "Which one is mine?"]
    assert any("The compaction failed" in m for m in caplog.messages)


def test_a_stop_carries_the_summary_and_a_reset_drops_it(serve):
    stack = serve(summarizer=ScriptedModel(then=SUMMARY))
    run(stack, MUG, PLATE)
    compacted = stack.runner.session

    stack.runner.stop("stop")
    assert stack.runner.wait_idle(10)
    assert humans(stack.messages)[:3] == [f"{SUMMARY_PREFIX} {SUMMARY.text}", PLATE, "stop"]
    assert stack.messages[-1].content == STOP_LINE

    run(stack, MUG, PLATE)  # compacted again, so a summary waits in the carry
    assert stack.runner.session.id == compacted.id
    stack.runner.reset()
    run(stack, "Which one is mine?")
    assert stack.runner.session.id != compacted.id
    assert humans(stack.model.received[-1]) == ["Which one is mine?"]
