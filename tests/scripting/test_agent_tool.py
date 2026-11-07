"""run_script on the competition agent: the tool, the shelf it saves to, the verdict and shutdown.

Every test runs over FakeRobot and a shelf under the test's own WALKIE_TOOLS_HOME, so
nothing reaches the repository's tool folders.
"""

import json
import threading
import time

import pytest

from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.runtime.middleware import ToolLog
from walkie_agent.scripting import agent_tool
from walkie_agent.scripting.agent_tool import AGENT_BUDGETS, NOT_CALLABLE, UNVERIFIED, AgentScripts
from walkie_agent.scripting.sandbox import SandboxUnavailable
from walkie_agent.scripting.shelf import Shelf
from walkie_agent.tools.atomic import DEFAULT, GPSR

VISIT = (
    "def visit(place: str) -> str:\n"
    '    """Drive to a place and say so."""\n'
    "    go_to(place)\n"
    '    say("Here is the " + place + ".")\n'
    "    return place\n"
    "\n"
    'visit("shelf")\n'
)
BROKEN_VISIT = (
    "def visit(place: str) -> str:\n"
    '    """Drive to a place, but never arrive."""\n'
    '    go_to("nowhere")\n'
    "    return place\n"
    "\n"
    'visit("shelf")\n'
)


@pytest.fixture
def log(tmp_path):
    return ToolLog(tmp_path / "tools.jsonl")


@pytest.fixture
def scripts(pool, log):
    return AgentScripts(pool, Shelf(), tool_log=log)


@pytest.fixture
def run(scripts, robot):
    """The run_script tool over GPSR on the fake robot, called as the agent calls it."""
    tool = scripts.tool(GPSR, robot.ctx)

    def call(code, save=None):
        args = {"code": code} if save is None else {"code": code, "save": save}
        return json.loads(tool.invoke(args))

    call.tool = tool
    return call


def records(log):
    return [json.loads(line) for line in log.path.read_text().splitlines()]


def task(scripts, *calls):
    """One task: begin, the calls, end, as the Runner marks a run."""
    scripts.begin()
    try:
        for each in calls:
            each()
    finally:
        scripts.end()


# -- The tool ------------------------------------------------------------------------


def test_a_script_runs_the_atomics_on_the_robot_and_returns_each_call(run, robot):
    result = run('go_to("shelf")\nsay("hello")\n7')

    assert (result["ok"], result["output"]) == (True, 7)
    assert result["calls"] == ["go_to: ok", "say: ok"]
    assert robot.said == ["hello"]
    assert robot.snapshot()["at"] == "shelf"


def test_the_description_lists_the_bound_atomics_and_the_budgets(run):
    description = run.tool.description

    assert run.tool.name == "run_script"
    for fn in GPSR:
        assert f"\n{fn.__name__}(" in description, fn.__name__
    assert f"at most\n{AGENT_BUDGETS.max_calls} calls" in description
    assert f"{AGENT_BUDGETS.wall_s:g} seconds in all" in description
    assert "A stop\nends it at once." in description
    assert set(run.tool.args) == {"code", "save"}


def test_a_script_cannot_call_run_script_update_todos_or_wait_for_person(scripts, robot):
    def wait_for_person(ctx, question: str) -> str:
        """Ask a question and end the turn.

        Args:
            question: What to ask.
        """
        return question

    tool = scripts.tool([*DEFAULT, wait_for_person], robot.ctx)

    assert NOT_CALLABLE == {"run_script", "update_todos", "wait_for_person"}
    assert "\nsay(" in tool.description
    assert "\nwait_for_person(" not in tool.description
    with pytest.raises(StepError) as raised:
        tool.invoke({"code": 'wait_for_person("Which cup?")'})
    assert raised.value.code is ErrorCode.SKILL_FAILED
    assert robot.calls == []


def test_the_gpsr_preset_stays_off_the_robots_allow_list(scripts, robot):
    tool = scripts.tool(DEFAULT, robot.ctx)

    assert "\ngo_to(" in tool.description
    for name in ("fetch_object", "put_object_on", "give_to_operator", "count_objects", "pick"):
        assert f"{name}(" not in tool.description, name
    with pytest.raises(StepError) as raised:
        tool.invoke({"code": 'fetch_object("cup", "shelf")'})
    assert raised.value.code is ErrorCode.SKILL_FAILED


def test_a_script_that_fails_raises_a_step_error_with_its_calls(run, robot):
    with pytest.raises(StepError) as raised:
        run('go_to("nowhere")')

    assert raised.value.code is ErrorCode.SKILL_FAILED
    assert "The script did not run to its end" in str(raised.value)
    assert "go_to: NOT_FOUND" in str(raised.value)
    assert robot.snapshot()["at"] != "nowhere"


# -- The shelf -----------------------------------------------------------------------


def test_save_keeps_a_function_in_temporary_and_marks_it_unverified(run, isolated_tools_home):
    result = run(VISIT, save=["visit"])

    assert result["saved"] == ["visit"]
    assert (isolated_tools_home / "temporary" / "visit.py").is_file()
    assert list((isolated_tools_home / "bundled").glob("*.py")) == []
    description = run.tool.description
    assert f"{UNVERIFIED}\ndef visit(place: str) -> str:" in description
    assert "Drive to a place and say so." in description


def test_a_saved_function_is_callable_in_a_later_script(run, robot):
    run(VISIT, save=["visit"])

    result = run('visit("bar")')

    assert (result["ok"], result["output"]) == (True, "bar")
    assert robot.snapshot()["at"] == "bar"


def test_a_script_on_a_failed_run_saves_nothing(run, isolated_tools_home):
    with pytest.raises(StepError):
        run(BROKEN_VISIT, save=["visit"])

    assert list((isolated_tools_home / "temporary").glob("*.py")) == []
    assert UNVERIFIED not in run.tool.description


# -- The verdict ---------------------------------------------------------------------


def test_a_passed_verdict_verifies_what_the_task_proved_into_bundled(scripts, run, log, isolated_tools_home):
    task(scripts, lambda: run(VISIT, save=["visit"]))

    line = scripts.verdict(True)

    assert line == "Verified visit from the last task."
    assert (isolated_tools_home / "bundled" / "visit.py").is_file()
    assert scripts.shelf.unverified() == []
    assert UNVERIFIED not in run.tool.description
    assert "def visit(place: str) -> str:" in run.tool.description
    verdict = records(log)[-1]
    assert (verdict["tool"], verdict["args"], verdict["status"]) == ("verdict", {"passed": True}, "success")
    assert verdict["verified"] == {"visit": scripts.shelf.versions()["visit"]}
    assert verdict["result"] == line


def test_a_verified_function_stays_callable(scripts, run, robot):
    task(scripts, lambda: run(VISIT, save=["visit"]))
    scripts.verdict(True)

    assert run('visit("bar")')["output"] == "bar"


def test_a_failed_verdict_verifies_nothing(scripts, run, log, isolated_tools_home):
    task(scripts, lambda: run(VISIT, save=["visit"]))

    line = scripts.verdict(False)

    assert line == "Recorded a failed verdict on the last task. Nothing was verified."
    assert list((isolated_tools_home / "bundled").glob("*.py")) == []
    assert scripts.shelf.unverified() == ["visit"]
    assert UNVERIFIED in run.tool.description
    verdict = records(log)[-1]
    assert (verdict["args"], verdict["status"], verdict["verified"]) == ({"passed": False}, "success", {})


def test_a_passed_verdict_on_a_task_without_a_saved_return_verifies_nothing(scripts, run, isolated_tools_home):
    task(scripts, lambda: run('say("hello")'))

    line = scripts.verdict(True)

    assert line.startswith("Recorded a passed verdict on the last task.")
    assert "nothing was verified" in line
    assert list((isolated_tools_home / "bundled").glob("*.py")) == []


def test_a_function_that_raised_in_the_task_is_not_verified(scripts, run, isolated_tools_home):
    run(VISIT, save=["visit"])

    def raises():
        with pytest.raises(StepError):
            run('visit("nowhere")')

    task(scripts, lambda: run('visit("bar")'), raises)

    assert scripts.verdict(True).startswith("Recorded a passed verdict")
    assert list((isolated_tools_home / "bundled").glob("*.py")) == []


@pytest.mark.parametrize("passed", [True, False])
def test_a_stray_verdict_is_logged_and_ignored(scripts, run, log, isolated_tools_home, passed):
    # Before any task has ended.
    assert scripts.verdict(passed) == "Ignored the verdict: no task has ended since the last one."

    # While a task runs, which leaves the task to judge once it ends.
    scripts.begin()
    run(VISIT, save=["visit"])
    assert scripts.verdict(passed) == "Ignored the verdict: a task is running."
    scripts.end()
    scripts.verdict(passed)

    # A second verdict on the same task.
    assert scripts.verdict(True) == "Ignored the verdict: no task has ended since the last one."

    statuses = [record["status"] for record in records(log) if record["tool"] == "verdict"]
    assert statuses == ["ignored", "ignored", "success", "ignored"]
    assert (isolated_tools_home / "bundled" / "visit.py").is_file() == passed


def test_a_verdict_judges_only_the_task_that_ended_last(scripts, run, isolated_tools_home):
    task(scripts, lambda: run(VISIT, save=["visit"]))
    task(scripts, lambda: run('say("hello")'))

    scripts.verdict(True)

    assert list((isolated_tools_home / "bundled").glob("*.py")) == []
    assert scripts.shelf.unverified() == ["visit"]


# -- A stop --------------------------------------------------------------------------


def test_a_stop_ends_a_running_script_as_cancelled(scripts):
    robot = FakeRobot(seed=1, drive_s=60.0)
    tool = scripts.tool(GPSR, robot.ctx)
    code = (
        'for place in ["shelf", "bar", "table"]:\n'
        "    try:\n"
        "        go_to(place)\n"
        "    except BaseException:\n"
        "        pass\n"
        'say("done")\n'
    )
    outcome = {}

    def call():
        try:
            tool.invoke({"code": code})
        except StepError as error:
            outcome["error"] = error

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    deadline = time.monotonic() + 10.0
    while not any(entry["verb"] == "base.go_to" for entry in list(robot.calls)):
        assert time.monotonic() < deadline, "the drive never started"
        time.sleep(0.005)
    stopped = time.monotonic()
    robot.stop()
    robot.cancel.cancel()
    worker.join(10.0)

    assert not worker.is_alive(), "the stop did not end the script"
    assert time.monotonic() - stopped < 5.0
    assert outcome["error"].code is ErrorCode.CANCELLED
    assert "A stop ended the script" in str(outcome["error"])
    assert "go_to: CANCELLED" in str(outcome["error"])
    assert robot.said == []
    assert [entry["verb"] for entry in robot.calls].count("base.go_to") == 1


# -- Startup and shutdown ------------------------------------------------------------


def test_close_waits_for_the_pool_and_says_what_is_left():
    release = threading.Event()
    scripts = AgentScripts(object(), Shelf(), closer=lambda: release.wait(10.0))

    assert scripts.close(0.05) is False
    assert scripts.unresolved() == ["The script sandbox's workers have not closed."]
    release.set()
    assert scripts.close(5.0) is True
    assert scripts.unresolved() == []
    assert scripts.close(0.0) is True


def test_a_closer_that_raises_still_counts_as_closed(caplog):
    def fails():
        raise RuntimeError("worker gone")

    scripts = AgentScripts(object(), Shelf(), closer=fails)

    assert scripts.close(5.0) is True
    assert "Could not close the script sandbox" in caplog.text


def test_without_a_closer_close_has_nothing_to_wait_for(scripts):
    assert scripts.close(0.0) is True
    assert scripts.unresolved() == []


def test_open_starts_a_pool_that_close_ends(robot):
    scripts = AgentScripts.open(Shelf(), max_processes=1)
    tool = scripts.tool(GPSR, robot.ctx)
    assert json.loads(tool.invoke({"code": "1 + 1"}))["output"] == 2

    assert scripts.close(10.0) is True
    with pytest.raises(RuntimeError, match="not active"), scripts._runner.pool.checkout():
        pass


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (SandboxUnavailable("bwrap is not installed. Install it, or pass --no-bwrap on purpose."), "bwrap is not installed"),
        (None, "bwrap could not start a worker: Permission denied"),
    ],
)
def test_open_refuses_without_the_sandbox_in_the_agents_words(monkeypatch, failure, reason):
    def open_pool(**_kwargs):
        if failure is not None:
            raise failure
        try:
            raise OSError("Permission denied")
        except OSError as exc:
            raise SandboxUnavailable(f"bwrap could not start a Monty worker: {exc}") from exc

    monkeypatch.setattr(agent_tool, "open_pool", open_pool)

    with pytest.raises(SandboxUnavailable) as raised:
        AgentScripts.open(Shelf())

    message = str(raised.value)
    assert message == f"--scripts runs scripts only inside bwrap, and {reason}. Install bwrap, or start without --scripts."
    assert "--no-bwrap" not in message
