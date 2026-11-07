"""The driver refuses what is off the allow-list and enforces each budget."""

import functools
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

import walkie_agent.scripting.driver as driver
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.scripting.driver import (
    Budgets,
    ScriptRunner,
    allow_list,
    script_summary,
    stubs_for,
)
from walkie_agent.tools.atomic import GPSR, go_to


def test_a_script_runs_atomics_on_the_robot_and_keeps_a_ledger(runner, fns, robot):
    result = runner.run('go_to("shelf")\nprint("there")\nsay("hello")\n7', fns)

    assert (result.ok, result.phase, result.output) == (True, "ok", 7)
    assert result.printed == "there\n"
    assert [(call["tool"], call["status"]) for call in result.calls] == [("go_to", "ok"), ("say", "ok")]
    assert result.calls[0]["args"] == ["shelf"]
    assert robot.said == ["hello"]
    assert robot.snapshot()["at"] == "shelf"


def test_stubs_keep_defaults_and_hide_ctx(fns):
    stubs = stubs_for(fns)

    assert "def find_objects(what: str = '') -> list[dict[str, Any]]: ..." in stubs
    assert "def move_relative(forward_m: float, left_m: float = 0.0, turn_deg: float = 0.0)" in stubs
    assert "ctx" not in stubs


def test_a_name_used_twice_in_the_allow_list_is_refused(robot):
    with pytest.raises(ValueError, match="go_to"):
        allow_list([go_to, go_to], robot.ctx)


@pytest.mark.parametrize(
    "code",
    [
        "ctx",
        "ctx.base.stop()",
        "send_vel(1.0)",
        'open("/etc/passwd")',
    ],
)
def test_the_type_check_stops_a_script_that_names_what_does_not_exist(runner, fns, robot, code):
    result = runner.run(code, fns)

    assert result.phase == "type_check"
    assert not result.calls
    assert not robot.snapshot()["calls"]


@pytest.mark.parametrize(
    ("code", "tool", "status", "exception"),
    [
        ("ctx.base.stop()  # type: ignore", "ctx", "unknown_name", "NameError"),
        ("x = go_to", "go_to", "unknown_name", "NameError"),
        ("send_vel(1.0)  # type: ignore", "send_vel", "denied", "PermissionError"),
        ('open("/etc/passwd")  # type: ignore', "open", "denied", "PermissionError"),
        ('import os\nos.getenv("HOME")', "os.getenv", "denied", "PermissionError"),
        ('import os\nos.environ["PATH"]', "os.environ", "denied", "PermissionError"),
        ('import pathlib\npathlib.Path("/etc/passwd").read_text()', "Path.read_text", "denied", "PermissionError"),
    ],
)
def test_the_driver_denies_at_run_time_what_passes_the_type_check(
    runner, fns, robot, code, tool, status, exception
):
    result = runner.run(code, fns)

    assert (result.ok, result.phase) == (False, "runtime")
    assert {"tool": tool, "status": status} in result.calls
    assert exception in result.error
    assert not robot.snapshot()["calls"]


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        ("move_relative(True)", "forward_m"),
        ('go_to(1)  # ty: ignore', "place"),
        ('move_relative("1")  # ty: ignore', "forward_m"),
        ("go_to()  # ty: ignore", "place"),
        ('go_to("shelf", extra=1)  # ty: ignore', "extra"),
    ],
)
def test_wrong_arguments_never_reach_the_atomic(runner, fns, robot, code, fragment):
    result = runner.run(code, fns)

    assert not result.ok
    assert fragment in result.error
    assert not robot.snapshot()["calls"]


def test_an_int_still_counts_as_a_number(runner, fns):
    result = runner.run("move_relative(1)", fns)

    assert result.ok, result.error


def test_a_step_error_reaches_the_script_as_a_catchable_runtime_error(runner, fns):
    code = 'try:\n    go_to("garage")\nexcept RuntimeError as error:\n    print(error)\n"after"'

    result = runner.run(code, fns)

    assert (result.ok, result.output) == (True, "after")
    assert result.printed.startswith("NOT_FOUND: Unknown place 'garage'.")
    assert result.calls[0]["status"] == "NOT_FOUND"


def test_an_uncaught_step_error_ends_the_script_with_its_code(runner, fns):
    result = runner.run('go_to("garage")', fns)

    assert (result.ok, result.phase) == (False, "runtime")
    assert "RuntimeError: NOT_FOUND: Unknown place 'garage'." in result.error


def test_a_bug_in_a_host_function_reaches_the_script_without_detail(runner, caplog):
    def broken() -> str:
        raise KeyError("secret detail")

    result = runner.run("broken()", {"broken": broken})

    assert "RuntimeError: INTERNAL: broken failed with KeyError." in result.error
    assert "secret detail" not in result.error
    assert result.calls[0]["status"] == "INTERNAL"
    assert "broken" in caplog.text


def test_the_call_budget_ends_a_script_and_no_script_can_catch_it(runner, fns, robot):
    code = (
        "while True:\n"
        "    try:\n"
        '        say("again")\n'
        "    except BaseException:\n"
        "        pass\n"
    )

    result = runner.run(code, fns)

    assert (result.ok, result.phase) == (False, "budget")
    assert len(robot.said) == 40
    assert "suspension limit 40" in result.error


def test_the_feed_budget_stops_an_endless_loop(pool, fns):
    runner = ScriptRunner(pool, budgets=Budgets(feed_s=0.2))

    result = runner.run("while True:\n    pass", fns)

    assert (result.ok, result.phase) == (False, "budget")
    assert "TimeoutError" in result.error


def test_the_memory_budget_stops_a_big_allocation(runner, fns):
    result = runner.run('x = "a" * (200 * 1024 * 1024)', fns)

    assert (result.ok, result.phase) == (False, "budget")
    assert "MemoryError" in result.error


def test_the_print_budget_stops_a_flood(runner, fns):
    result = runner.run('while True:\n    print("x" * 500)', fns)

    assert (result.ok, result.phase) == (False, "budget")


def test_the_wall_budget_stops_a_slow_script_and_the_pool_stays_usable(pool):
    robot = FakeRobot(seed=1, drive_s=0.1)
    fns = allow_list(GPSR, robot.ctx)
    runner = ScriptRunner(pool, budgets=replace(Budgets(), wall_s=0.05))

    result = runner.run('go_to("shelf")\ngo_to("bar")\ngo_to("table")', fns)

    assert (result.ok, result.phase) == (False, "budget")
    assert "wall clock" in result.error
    assert ScriptRunner(pool).run("1 + 1", fns).output == 2


@pytest.fixture
def slow_go_to(monkeypatch, robot, fns):
    """The driver's clock moves only when go_to runs, so a loaded machine cannot change a result."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: clock.now))
    go_to_on_robot = fns["go_to"]

    @functools.wraps(go_to_on_robot)
    def slow(*args, **kwargs):
        clock.now += 31.0
        return go_to_on_robot(*args, **kwargs)

    return {**fns, "go_to": slow}


def test_a_last_call_that_overruns_the_wall_budget_fails_it(runner, slow_go_to, robot):
    result = runner.run('go_to("shelf")', slow_go_to)

    assert (result.ok, result.phase) == (False, "budget")
    assert "wall clock" in result.error
    # A call already running finishes first: the budget does not interrupt it.
    assert [(call["tool"], call["status"]) for call in result.calls] == [("go_to", "ok")]
    assert robot.snapshot()["at"] == "shelf"


def test_no_host_call_runs_after_the_wall_budget_is_spent(runner, slow_go_to, robot):
    result = runner.run('go_to("shelf")\nsay("too late")', slow_go_to)

    assert (result.ok, result.phase) == (False, "budget")
    assert robot.said == []


def test_a_run_inside_the_wall_budget_is_not_failed_by_the_check(runner, slow_go_to, robot):
    result = runner.run('say("in time")', slow_go_to)

    assert (result.ok, result.phase) == (True, "ok")
    assert robot.said == ["in time"]


# What a script may do with the CANCELLED of the drive a stop ends.
STOPPED_SCRIPTS = {
    "uncaught": 'for place in ["shelf", "bar", "table"]:\n    go_to(place)\nsay("done")\n',
    "caught_then_called": (
        'for place in ["shelf", "bar", "table"]:\n'
        "    try:\n"
        "        go_to(place)\n"
        "    except BaseException:\n"
        "        pass\n"
        'say("done")\n'
    ),
    "caught_then_looped": (
        'try:\n    go_to("shelf")\nexcept BaseException:\n    pass\nwhile True:\n    pass\n'
    ),
}


@pytest.mark.parametrize("code", STOPPED_SCRIPTS.values(), ids=STOPPED_SCRIPTS)
def test_a_stop_ends_a_running_script_and_no_script_can_catch_it(runner, code):
    robot = FakeRobot(seed=1, drive_s=60.0)
    fns = allow_list(GPSR, robot.ctx)

    def stop_once_driving():
        deadline = time.monotonic() + 10.0
        while not any(call["verb"] == "base.go_to" for call in list(robot.calls)):
            assert time.monotonic() < deadline, "the drive never started"
            time.sleep(0.005)
        robot.stop()
        robot.cancel.cancel()

    stopper = threading.Thread(target=stop_once_driving, daemon=True)
    stopper.start()
    started = time.monotonic()
    result = runner.run(code, fns, cancel=robot.cancel)
    stopper.join(1.0)

    assert time.monotonic() - started < 10.0, "the stop did not end the 60 s drive"
    assert (result.ok, result.phase) == (False, "cancelled")
    assert result.error == "CANCELLED: a stop ended the script."
    # The drive the stop ended is the only call: the refusals after it run nothing.
    assert [(call["tool"], call["status"]) for call in result.calls] == [("go_to", "CANCELLED")]
    assert robot.said == []
    assert robot.snapshot()["at"] != "table"


def test_a_stop_before_the_script_starts_runs_no_call(runner, fns, robot):
    robot.cancel.cancel()

    result = runner.run('say("hello")', fns, cancel=robot.cancel)

    assert (result.ok, result.phase) == (False, "cancelled")
    assert result.calls == []
    assert robot.said == []


def test_a_type_error_keeps_its_phase_after_a_stop(runner, fns, robot):
    # The script never ran, so the stop ended nothing and the model learns of the error.
    robot.cancel.cancel()

    result = runner.run("go_to(place=1)", fns, cancel=robot.cancel)

    assert (result.ok, result.phase) == (False, "type_check")
    assert result.calls == []


def test_a_recursion_bomb_is_a_spent_budget(runner, fns):
    result = runner.run("def f(n: int) -> int:\n    return f(n + 1)\nf(0)", fns)

    assert (result.ok, result.phase) == (False, "budget")


def test_a_host_call_inside_a_builtin_callback_fails_cleanly(runner, fns):
    result = runner.run('sorted(["shelf"], key=lambda place: go_to(place))', fns)

    assert (result.ok, result.phase) == (False, "runtime")
    assert "cannot suspend" in result.error


def test_a_type_error_is_reported_with_the_scripts_own_line(runner, fns):
    result = runner.run('\n\ngo_to(place=1)', fns)

    assert result.phase == "type_check"
    assert "main.py:3:" in result.error
    assert not result.calls


def test_broken_syntax_is_reported(runner, fns):
    result = runner.run("def (", fns)

    assert (result.ok, result.phase) == (False, "syntax")
    assert 'File "main.py", line 1' in result.error and "SyntaxError: invalid syntax" in result.error


def test_no_state_crosses_from_one_script_to_the_next(pool, fns):
    runner = ScriptRunner(pool)

    assert runner.run("kept = 41", fns).ok
    result = runner.run("kept + 1  # type: ignore", fns)

    assert not result.ok
    assert "kept" in result.error


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('{"a": [1, 2.5, None]}', {"a": [1, 2.5, None]}),
        ("(1, 2)", [1, 2]),
        ("{1}", "{1}"),
        ('float("nan")', "nan"),
        ("{(1, 2): 3}", "{(1, 2): 3}"),
    ],
)
def test_any_output_comes_back_as_json_data(runner, fns, code, expected):
    assert runner.run(code, fns).output == expected


def test_a_huge_output_is_cut(runner, fns):
    result = runner.run('"a" * 100000', fns)

    assert result.ok
    assert len(result.output) < 5000
    assert result.output.endswith("[cut]")


def test_the_summary_for_the_model_leaves_the_ledger_out(runner, fns):
    summary = script_summary(runner.run('go_to("shelf")', fns))

    assert summary == {"ok": True, "output": "Arrived at shelf.", "printed": "", "calls_made": 1}
    assert "error" in script_summary(runner.run('go_to("garage")', fns))
