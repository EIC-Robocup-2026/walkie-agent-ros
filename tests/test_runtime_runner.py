"""The Runner over the real stack: its queue, stop, reset, shutdown and failures.

Each test starts the real loop thread and talks to it as a ROS executor thread
would. The claim tags are explained in tests/test_runtime.py.
"""

import asyncio
import json
import logging
import threading
import time

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from runtime_harness import (
    add,
    ai,
    assert_every_call_answered,
    call,
    drive,
    make_stack,
    todo_call,
    update,
    wait_until,
)

from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.runtime import STOP_LINE, Runner, Session, ToolLog, is_stop_utterance
from walkie_agent.runtime.middleware import AFTER_STOP
from walkie_agent.runtime.runner import FAILURE_LINE
from walkie_agent.runtime.todos import STOPPED_NOTE, TodoRejected

DONE = AIMessage("Done.")


@pytest.fixture
def serve():
    """Starts a Runner over a new stack, and checks at the end that it shuts down."""
    runners = []

    def start(
        *script, then=None, delay_s=0.0, robot=None, tool_log=None, callbacks=(), on_reply=None,
        **build,
    ):
        stack = make_stack(
            *script, then=then, delay_s=delay_s, robot=robot, tool_log=tool_log, **build
        )
        stack.runner = Runner(
            stack.agent,
            stack.robot.ctx,
            on_todos=lambda items, changed: stack.published.append((items, changed)),
            on_reply=on_reply or stack.replies.append,
            callbacks=callbacks,
            tool_log=tool_log,
        )
        runners.append(stack.runner)
        stack.runner.start()
        return stack

    yield start
    for runner in runners:
        assert runner.shutdown(10)


def idle(stack) -> None:
    assert stack.runner.wait_idle(10), "The Runner did not finish in time."


def drive_started(stack) -> None:
    wait_until(lambda: "base.go_to" in stack.verbs())


def humans(messages) -> list[str]:
    return [m.content for m in messages if isinstance(m, HumanMessage)]


def log_records(log: ToolLog) -> list[dict]:
    return [json.loads(line) for line in log.path.read_text().splitlines()]


async def _hang(_messages):
    await asyncio.sleep(30)


# -- What counts as a stop --------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["stop", "Stop.", "  STOP!  ", "stop?", "Never mind!", "never   mind", "Nevermind."]
)
def test_stop_utterances(text):
    assert is_stop_utterance(text)


@pytest.mark.parametrize(
    "text",
    [
        "stop the music",
        "Walkie, stop",
        "don't stop",
        "never mind the cup",
        "stopping",
        "",
        "stop and come here",
    ],
)
def test_other_utterances_are_not_stops(text):
    assert not is_stop_utterance(text)


# -- The queue --------------------------------------------------------------------


def test_start_publishes_the_empty_list(serve):
    stack = serve()
    wait_until(lambda: stack.published == [([], [])])


def test_utterances_during_a_task_wait_and_run_one_by_one_in_order(serve):
    # test_graph.py:803
    stack = serve(then=DONE, delay_s=0.02)
    for text in ("First.", "Second.", "Third."):
        stack.runner.hear(text)
    idle(stack)
    assert len(stack.model.received) == 3
    assert humans(stack.model.received[-1]) == ["First.", "Second.", "Third."]
    assert stack.replies == ["Done."] * 3


def test_each_session_has_its_own_thread_and_tracing_session():
    first, second = Session(), Session()
    assert first.thread_id != second.thread_id
    handler = BaseCallbackHandler()
    assert first.config([handler]) == {
        "configurable": {"thread_id": first.thread_id},
        "max_concurrency": 1,
        "metadata": {"langfuse_session_id": first.thread_id},
        "callbacks": [handler],
    }


def test_a_queued_seed_is_a_copy_and_keeps_its_source(serve):
    stack = serve(then=DONE, delay_s=0.05)
    seed = [{"content": "Go there", "requires_tool": "go_to"}]
    stack.runner.submit("First.")
    stack.runner.submit("Second.", todos=seed, source="operator")
    seed[0]["content"] = "Changed"
    seed.append({"content": "Added later"})
    idle(stack)
    assert [(i["content"], i["source"]) for i in stack.items()] == [("Go there", "operator")]


def test_a_bad_seed_is_refused_in_the_caller_and_queues_nothing(serve):
    stack = serve(then=DONE)
    with pytest.raises(TodoRejected, match="has no content"):
        stack.runner.submit("Go.", todos=[{"content": " "}])
    assert stack.runner.wait_idle(0.2)
    assert stack.model.received == []


def test_an_empty_seed_differs_from_no_seed(serve):
    ask = ai(todo_call(update("t1", status="in_progress")))
    question = ai(call("wait_for_person", {"question": "Which?"}, "ask"))
    stack = serve(ask, question, DONE, DONE, DONE, DONE)
    stack.runner.submit("Go.", todos=["Step"])
    idle(stack)
    assert stack.statuses() == ["waiting"]

    # No seed: this is the answer, so the list stays.
    stack.runner.submit("The red one.")
    idle(stack)
    assert stack.statuses() == ["in_progress"]

    # An empty seed: the list is emptied.
    stack.runner.submit("Forget that.", todos=[])
    idle(stack)
    assert stack.items() == []


# -- Stop -------------------------------------------------------------------------


def test_stop_ends_a_drive_through_the_token_and_makes_no_more_model_calls(serve):
    # B1, S1, S2, B2, S3 (changed: the stop's pass has no LLM call at all)
    stack = serve(ai(drive("d")), robot=FakeRobot(drive_s=30))
    stack.runner.submit("Go to the sofa.")
    drive_started(stack)

    started = time.monotonic()
    stack.runner.hear("Stop!")
    idle(stack)

    assert time.monotonic() - started < 0.5
    assert stack.answer("d").status == "error"
    assert ErrorCode.CANCELLED in stack.answer("d").content
    assert len(stack.model.received) == 1
    assert stack.verbs().count("base.go_to") == 1
    assert stack.robot.said == [STOP_LINE]
    assert stack.robot.cancel.cancelled


def test_stop_during_an_llm_call_runs_none_of_the_tools_it_returns(serve):
    # test_graph.py:865. Not only motion: speech, todo updates and reads are refused too.
    calls = {
        "said": call("say", {"text": "On my way."}, "said"),
        "listed": todo_call(add("New step"), call_id="listed"),
        "read": call("list_places", {}, "read"),
        "found": call("find_objects", {"what": "cup"}, "found"),
        "drove": drive("drove"),
        "asked": call("wait_for_person", {"question": "Which?"}, "asked"),
    }
    stack = serve(ai(*calls.values()), DONE, delay_s=0.3)
    stack.runner.submit("Go.")
    wait_until(lambda: stack.model.received)
    stack.runner.stop()
    idle(stack)

    for call_id in calls:
        assert stack.answer(call_id).content == AFTER_STOP
        assert stack.answer(call_id).status == "error"
    assert stack.verbs() == ["speech.say"]
    assert stack.robot.said == [STOP_LINE]
    assert len(stack.model.received) == 1
    assert stack.items() == []
    assert not stack.state.get("question")

    # The next task's history is one a provider accepts: the model checks it.
    stack.runner.submit("Next.")
    idle(stack)
    assert len(stack.model.received) == 2
    assert stack.replies[-1] == "Done."


def test_a_stop_during_an_llm_call_that_answers_in_text_makes_no_gate_call(serve):
    stack = serve(AIMessage("I will."), delay_s=0.3)
    stack.runner.submit("Go.", todos=["Open step"])
    wait_until(lambda: stack.model.received)
    stack.runner.stop()
    idle(stack)
    assert len(stack.model.received) == 1
    assert stack.statuses() == ["cancelled"]


def test_a_refused_call_of_a_stopped_run_does_not_reach_the_ledger(serve):
    stack = serve(ai(drive("d")), delay_s=0.3)
    stack.runner.submit("Go.")
    wait_until(lambda: stack.model.received)
    stack.runner.stop()
    idle(stack)
    assert stack.state.get("ledger", []) == []


def test_a_stop_during_a_hung_llm_call_speaks_at_once_and_updates_after_the_deadline(serve):
    stack = serve(_hang, DONE, deadline_s=0.5)
    stack.runner.submit("Go.", todos=["Open step"])
    wait_until(lambda: stack.model.received)
    stack.runner.stop()

    wait_until(lambda: stack.robot.said == [STOP_LINE], timeout_s=0.3)
    assert stack.statuses() == ["pending"]  # the active run still holds the checkpoint
    idle(stack)
    assert stack.statuses() == ["cancelled"]

    stack.runner.submit("Next.")
    idle(stack)
    assert stack.replies[-1] == "Done."


def test_the_stop_line_is_spoken_once_and_logged_by_the_runner(serve, tmp_path, caplog):
    log = ToolLog(tmp_path / "tools.jsonl")
    stack = serve(ai(drive("d")), robot=FakeRobot(drive_s=30), tool_log=log)
    stack.runner.submit("Go.")
    drive_started(stack)
    with caplog.at_level(logging.INFO, logger="walkie_agent.runtime.runner"):
        stack.runner.hear("Never mind.")
        idle(stack)

    assert stack.robot.said == [STOP_LINE]
    lines = [r for r in log_records(log) if r["tool"] == "stop_line"]
    assert len(lines) == 1
    assert lines[0]["args"] == {"heard": "Never mind."}
    assert lines[0]["status"] == "success"
    assert lines[0]["result"] == STOP_LINE
    assert lines[0]["session"] == stack.runner.session.thread_id
    assert [r for r in caplog.messages if "Never mind." in r and STOP_LINE in r]
    assert stack.messages[-1].content == STOP_LINE


def test_a_stop_line_that_cannot_be_spoken_is_logged_and_the_stop_still_holds(serve, tmp_path):
    log = ToolLog(tmp_path / "tools.jsonl")
    robot = FakeRobot()
    robot.faults["speech.say"] = StepError(ErrorCode.TIMEOUT, "The speaker is stuck.")
    stack = serve(robot=robot, tool_log=log)
    stack.runner.stop()
    idle(stack)
    assert [r["status"] for r in log_records(log)] == ["error"]
    assert robot.cancel.cancelled


def test_a_stop_line_is_logged_under_the_session_it_was_heard_in(serve, tmp_path):
    class SlowStopLine(FakeRobot):
        speaking, release = threading.Event(), threading.Event()

        def say(self, text, **kwargs):
            self.speaking.set()
            assert self.release.wait(5)
            super().say(text, **kwargs)

    log = ToolLog(tmp_path / "tools.jsonl")
    stack = serve(robot=SlowStopLine(), tool_log=log)
    heard_in = stack.runner.session.thread_id
    stack.runner.stop()
    assert stack.robot.speaking.wait(5)
    stack.runner.reset()
    wait_until(lambda: stack.runner.session.thread_id != heard_in)
    stack.robot.release.set()
    idle(stack)
    assert [r["session"] for r in log_records(log)] == [heard_in]


class PauseBeforeTool(BaseCallbackHandler):
    """Holds a tool after StopLatch has looked at the token and before the tool's body runs."""

    def __init__(self, name):
        self.name = name
        self.entered, self.release = threading.Event(), threading.Event()

    def on_tool_start(self, serialized, input_str, **kwargs):
        if serialized.get("name") == self.name:
            self.entered.set()
            assert self.release.wait(5)


@pytest.mark.parametrize(
    "late",
    [
        call("say", {"text": "The task succeeded."}, "late"),
        call("list_places", {}, "late"),
        call("wait_for_person", {"question": "Which?"}, "late"),
        todo_call(update("t1", status="completed"), call_id="late"),
    ],
    ids=["say", "read", "question", "update_todos"],
)
def test_a_stop_that_lands_just_before_a_tool_body_still_stops_the_tool(serve, late):
    pause = PauseBeforeTool(late["name"])
    stack = serve(ai(late), DONE, callbacks=[pause])
    stack.runner.submit("Go.", todos=["Open step"])
    assert pause.entered.wait(5)
    stack.runner.stop()
    wait_until(lambda: stack.robot.said == [STOP_LINE])
    pause.release.set()
    idle(stack)

    assert stack.robot.said == [STOP_LINE]
    assert stack.answer("late").status == "error"
    assert stack.answer("late").content.startswith("CANCELLED:")
    assert stack.statuses() == ["cancelled"]
    assert not stack.state.get("question")
    assert len(stack.model.received) == 1


def test_both_todo_listeners_run_on_the_runner_s_thread():
    heard = []

    def listen(items, changed):
        heard.append((threading.current_thread().name, [i["status"] for i in items]))

    stack = make_stack(ai(todo_call(update("t1", status="completed"))), DONE, on_todos=listen)
    runner = Runner(stack.agent, stack.robot.ctx, on_todos=listen)
    runner.start()
    runner.submit("Go.", todos=["Step"])
    assert runner.wait_idle(10)
    assert runner.shutdown(10)
    assert (runner._thread.name, ["completed"]) in heard
    assert {thread for thread, _ in heard} == {runner._thread.name}


def test_a_bare_stop_cancels_the_open_steps_with_no_model_call(serve):
    # test_graph.py:897 changed: a stop while idle is no longer an ordinary message (decision 6).
    first = ai(todo_call(update("t1", status="completed")), call("say", {"text": "Hi"}, "s"))
    stack = serve(first, ai(todo_call(update("t2", status="in_progress"))), DONE, DONE, DONE)
    stack.runner.submit("Go.", todos=["One", "Two", "Three"])
    idle(stack)
    calls = len(stack.model.received)

    stack.runner.hear("stop")
    idle(stack)
    assert len(stack.model.received) == calls
    assert stack.statuses() == ["completed", "cancelled", "cancelled"]
    assert [i["note"] for i in stack.items()][1:] == [STOPPED_NOTE, STOPPED_NOTE]
    latest, changed = stack.published[-1]
    assert [i["status"] for i in latest] == ["completed", "cancelled", "cancelled"]
    assert [i["id"] for i in changed] == ["t2", "t3"]
    assert stack.messages[-2].content == "stop"
    assert isinstance(stack.messages[-1], AIMessage)
    assert stack.messages[-1].content == STOP_LINE


def test_a_bare_stop_while_idle_with_no_list_only_speaks(serve):
    stack = serve()
    stack.runner.hear("stop")
    idle(stack)
    assert stack.model.received == []
    assert stack.robot.said == [STOP_LINE]
    assert stack.published == [([], [])]


def test_a_stop_drops_the_queued_tasks_but_not_later_ones(serve):
    # test_graph.py:835
    stack = serve(ai(drive("d")), DONE, robot=FakeRobot(drive_s=30))
    stack.runner.submit("Go.")
    drive_started(stack)
    stack.runner.submit("Queued one.")
    stack.runner.submit("Queued two.", todos=["Never seeded"])
    stack.runner.hear("stop")
    stack.runner.submit("After.")
    idle(stack)
    assert humans(stack.messages) == ["Go.", "stop", "After."]
    assert stack.items() == []


def test_the_latch_holds_until_the_next_task_and_a_compound_stop_goes_to_the_llm(serve):
    # S4, decisions 6 and 7
    stack = serve(ai(drive("d")), AIMessage("Here."))
    stack.runner.hear("stop")
    idle(stack)
    assert stack.robot.cancel.cancelled
    assert stack.model.received == []

    stack.runner.hear("Stop and come here.")
    idle(stack)
    assert not stack.robot.cancel.cancelled
    assert humans(stack.model.received[0]) == ["stop", "Stop and come here."]
    assert stack.verbs().count("base.go_to") == 1
    assert stack.answer("d").status == "success"
    assert stack.replies == ["Here."]
    assert stack.robot.said == [STOP_LINE]


def test_a_compound_stop_during_a_task_does_not_cancel_it(serve):
    stack = serve(ai(drive("d")), DONE, DONE, robot=FakeRobot(drive_s=0.3))
    stack.runner.submit("Go.")
    drive_started(stack)
    stack.runner.hear("Stop and come here.")
    assert not stack.robot.cancel.cancelled
    idle(stack)
    assert stack.answer("d").status == "success"
    assert humans(stack.model.received[-1]) == ["Go.", "Stop and come here."]


def test_every_run_stays_on_the_checkpoint_with_every_call_answered(serve):
    # S5a
    stack = serve(ai(drive("d")), ai(drive("e", "bar")), DONE, robot=FakeRobot(drive_s=30))
    stack.runner.submit("Go to the sofa.")
    drive_started(stack)
    stack.runner.hear("stop")
    idle(stack)
    stack.robot.drive_s = 0.0
    stack.runner.submit("Go to the bar.")
    idle(stack)
    assert humans(stack.messages) == ["Go to the sofa.", "stop", "Go to the bar."]
    assert_every_call_answered(stack.messages)


# -- Questions --------------------------------------------------------------------


def test_question_then_answer_makes_a_second_model_call(serve):
    stack = serve(ai(call("wait_for_person", {"question": "Which?"}, "ask")), AIMessage("Okay."))
    stack.runner.submit("Bring it.")
    idle(stack)
    assert stack.robot.said == ["Which?"]
    assert len(stack.model.received) == 1
    assert stack.replies == []  # the person heard the question, so nothing goes to the console

    stack.runner.submit("The red one.")
    idle(stack)
    assert len(stack.model.received) == 2
    assert stack.replies == ["Okay."]


def test_a_waiting_question_does_not_make_the_next_task_look_like_its_answer(serve):
    stack = serve(
        ai(todo_call(update("t1", status="in_progress"))),
        ai(call("wait_for_person", {"question": "Which?"}, "ask")),
        DONE,
    )
    stack.runner.submit("Bring it.", todos=["Bring the cup"])
    idle(stack)
    assert stack.statuses() == ["waiting"]

    stack.runner.hear("stop")
    idle(stack)
    assert stack.statuses() == ["cancelled"]
    assert not stack.state.get("question")

    stack.runner.submit("Go to the sofa.")
    idle(stack)
    assert stack.items() == []
    assert stack.model.received[-1][-1].content == "Go to the sofa."  # no reminder after it


def test_a_question_asked_after_a_bare_stop_works(serve):
    stack = serve(ai(call("wait_for_person", {"question": "Which?"}, "ask")), AIMessage("Okay."))
    stack.runner.hear("stop")
    idle(stack)
    stack.runner.submit("Bring it.")
    idle(stack)
    assert stack.robot.said == [STOP_LINE, "Which?"]
    stack.runner.submit("The red one.")
    idle(stack)
    assert len(stack.model.received) == 2
    assert stack.replies == ["Okay."]


def test_a_stop_cancels_a_question_with_no_active_step(serve):
    stack = serve(ai(call("wait_for_person", {"question": "Which?"}, "ask")), DONE)
    stack.runner.submit("Bring it.")
    idle(stack)
    stack.runner.hear("stop")
    idle(stack)
    assert not stack.state.get("question")
    stack.runner.submit("Go on.")
    idle(stack)
    assert stack.replies == ["Done."]


# -- Reset ------------------------------------------------------------------------


def test_reset_during_a_run_deletes_the_thread_and_starts_a_new_session(serve):
    # R3d, S5b (changed: the old thread is deleted, not kept). test_graph.py:979
    stack = serve(
        ai(todo_call(update("t1", status="in_progress")), drive("d")),
        AIMessage("Hi."),
        robot=FakeRobot(drive_s=30),
    )
    stack.runner.submit("Go.", todos=["Go there"])
    drive_started(stack)
    old = stack.runner.session
    stack.runner.submit("Queued.", todos=["Never seeded"])
    stack.runner.reset()
    idle(stack)

    assert stack.runner.session != old
    assert stack.agent.get_state(old.config()).values == {}
    assert list(stack.agent.checkpointer.list(old.config())) == []
    assert stack.published[-1] == ([], [])
    assert stack.state == {} or not stack.state.get("messages")
    assert len(stack.model.received) == 1

    stack.runner.submit("Hello.")
    idle(stack)
    assert [type(m) for m in stack.model.received[-1]] == [SystemMessage, HumanMessage]
    assert stack.replies == ["Hi."]
    assert not stack.robot.cancel.cancelled


def test_reset_waits_for_the_active_llm_call(serve):
    stack = serve(ai(drive("d")), delay_s=0.3)
    stack.runner.submit("Go.")
    wait_until(lambda: stack.model.received)
    old = stack.runner.session
    stack.runner.reset()
    assert stack.runner.session == old
    idle(stack)
    assert stack.runner.session != old
    assert list(stack.agent.checkpointer.list(old.config())) == []
    assert stack.verbs() == []  # the reply's drive was refused, since reset also cancels


def test_a_task_after_a_queued_reset_runs_in_the_new_session(serve):
    stack = serve(ai(drive("d")), then=DONE, robot=FakeRobot(drive_s=30))
    stack.runner.submit("Go.")
    drive_started(stack)
    old = stack.runner.session
    stack.runner.reset()
    stack.runner.submit("After.", todos=["New step"])
    idle(stack)
    assert stack.runner.session != old
    assert humans(stack.messages)[0] == "After."
    assert "Go." not in humans(stack.messages)
    assert [i["content"] for i in stack.items()] == ["New step"]


def test_a_stop_keeps_a_queued_reset_and_a_reset_drops_a_queued_stop(serve):
    stack = serve(ai(drive("d")), robot=FakeRobot(drive_s=30))
    stack.runner.submit("Go.")
    drive_started(stack)
    old = stack.runner.session
    stack.runner.reset()
    stack.runner.stop()
    idle(stack)
    assert stack.runner.session != old
    assert humans(stack.messages) == ["stop"]
    assert stack.robot.said == [STOP_LINE]

    second = stack.runner.session
    stack.runner.stop()
    stack.runner.reset()
    idle(stack)
    assert stack.runner.session != second
    assert stack.state == {} or not stack.state.get("messages")
    assert stack.robot.said == [STOP_LINE, STOP_LINE]  # each stop speaks once


def test_consecutive_sessions_log_distinct_ids(serve, tmp_path):
    log = ToolLog(tmp_path / "tools.jsonl")
    stack = serve(ai(drive("a")), DONE, ai(drive("b")), DONE, tool_log=log)
    stack.runner.submit("Go.")
    idle(stack)
    first = stack.runner.session.thread_id
    stack.runner.reset()
    idle(stack)
    stack.runner.submit("Go again.")
    idle(stack)
    second = stack.runner.session.thread_id
    assert first != second
    assert [(r["call_id"], r["session"]) for r in log_records(log)] == [("a", first), ("b", second)]


# -- Shutdown ---------------------------------------------------------------------


def test_shutdown_cancels_the_active_run_and_ends_the_thread(serve):
    stack = serve(ai(drive("d")), robot=FakeRobot(drive_s=30))
    stack.runner.submit("Go.")
    drive_started(stack)
    started = time.monotonic()
    assert stack.runner.shutdown(5)
    assert time.monotonic() - started < 1
    assert stack.answer("d").status == "error"
    assert stack.runner.wait_idle(0)


def test_shutdown_drops_the_queue_and_later_use_raises(serve):
    stack = serve(DONE, delay_s=0.1)
    stack.runner.submit("First.")
    wait_until(lambda: stack.model.received)
    stack.runner.submit("Never runs.")
    assert stack.runner.shutdown(5)
    assert len(stack.model.received) == 1

    for use in (
        lambda: stack.runner.submit("Late."),
        lambda: stack.runner.hear("Late."),
        lambda: stack.runner.hear("stop"),
        stack.runner.stop,
        stack.runner.reset,
    ):
        with pytest.raises(RuntimeError, match="has shut down"):
            use()
    assert stack.runner.shutdown(1)  # twice is fine


def test_shutdown_lets_a_stop_line_finish(serve):
    stack = serve()
    stack.runner.stop()
    assert stack.runner.shutdown(5)
    assert stack.robot.said == [STOP_LINE]


@pytest.mark.parametrize("late", ["stop", "reset", "submit"])
def test_a_call_that_lands_during_a_shutdown_is_refused_and_the_thread_still_ends(serve, late):
    release = threading.Event()

    async def hold(_messages):
        while not release.is_set():
            await asyncio.sleep(0.005)
        return DONE

    stack = serve(hold)
    stack.runner.submit("First.")
    wait_until(lambda: stack.model.received)
    assert not stack.runner.shutdown(0)
    with pytest.raises(RuntimeError, match="has shut down"):
        getattr(stack.runner, late)(*(["Late."] if late == "submit" else []))
    release.set()
    assert stack.runner.shutdown(5)


def test_shutdown_before_start():
    stack = make_stack()
    runner = Runner(stack.agent, stack.robot.ctx)
    assert runner.shutdown(1)
    with pytest.raises(RuntimeError, match="has shut down"):
        runner.submit("Late.")


# -- Failures ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "error"),
    [(_hang, TimeoutError), (RuntimeError("The LLM endpoint is down."), RuntimeError)],
    ids=["timeout", "error"],
)
def test_a_failed_task_is_logged_and_the_next_runs_on_the_same_checkpoint(
    serve, caplog, first, error
):
    # test_graph.py:1094, and the deadline with no fallback
    stack = serve(first, AIMessage("Back again."), deadline_s=0.1)
    with caplog.at_level(logging.ERROR, logger="walkie_agent.runtime.runner"):
        stack.runner.submit("First task.")
        idle(stack)
    assert stack.replies == [FAILURE_LINE]
    assert any(r.exc_info and isinstance(r.exc_info[1], error) for r in caplog.records)

    stack.runner.submit("Second task.")
    idle(stack)
    assert humans(stack.model.received[1]) == ["First task.", "Second task."]
    assert stack.replies == [FAILURE_LINE, "Back again."]


def test_a_timeout_after_a_tool_step_leaves_a_history_the_next_task_can_use(serve):
    stack = serve(ai(drive("d")), _hang, DONE, deadline_s=0.1)
    stack.runner.submit("Go.")
    idle(stack)
    stack.runner.submit("Again.")
    idle(stack)
    assert_every_call_answered(stack.messages)
    assert stack.answer("d").status == "success"
    assert stack.replies == [FAILURE_LINE, "Done."]


def test_listeners_that_raise_do_not_stop_the_runner(serve, caplog):
    def broken(_text):
        raise RuntimeError("The console is gone.")

    stack = serve(DONE, DONE, on_reply=broken)
    with caplog.at_level(logging.ERROR, logger="walkie_agent.runtime.runner"):
        stack.runner.submit("First.")
        idle(stack)
    stack.runner.submit("Second.")
    idle(stack)
    assert len(stack.model.received) == 2
    assert any("A Runner listener failed" in message for message in caplog.messages)


def test_a_listener_is_optional():
    stack = make_stack(DONE)
    runner = Runner(stack.agent, stack.robot.ctx)
    runner.start()
    runner.submit("Go.")
    assert runner.wait_idle(10)
    assert runner.shutdown(5)


# -- Tracing ----------------------------------------------------------------------


class _Recorder(BaseCallbackHandler):
    """Stands in for Langfuse's CallbackHandler, with no network."""

    def __init__(self) -> None:
        self.sessions: list[str] = []

    def on_chat_model_start(self, serialized, messages, *, metadata=None, **kwargs):
        self.sessions.append(metadata["langfuse_session_id"])


def test_callbacks_see_each_model_call_with_its_session(serve):
    recorder = _Recorder()
    stack = serve(then=DONE, callbacks=[recorder])
    stack.runner.submit("First.")
    idle(stack)
    first = stack.runner.session.thread_id
    stack.runner.reset()
    stack.runner.submit("Second.")
    idle(stack)
    assert recorder.sessions == [first, stack.runner.session.thread_id]
    assert recorder.sessions[0] != recorder.sessions[1]
