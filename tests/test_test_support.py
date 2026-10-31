"""The test infrastructure in conftest.py and hang_watchdog.py, which every other test relies on."""

import fcntl
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import rclpy
from action_msgs.msg import GoalStatusArray
from nav2_msgs.action import NavigateToPose
from rai.communication.ros2 import ROS2Connector
from rai.communication.ros2.connectors import base as rai_connector_base
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_action_status_default

from conftest import (
    PROJECT_ROOT,
    TEST_DOMAIN_ID,
    Call,
    ModuleClock,
    Spinner,
    StoppableExecutor,
    _choose_test_domain,
    domain_lock_path,
    wait_until,
)
from walkie_agent.sim.fake_walkie import FakeWalkie

# The rounds of the race test. A plain rai shutdown hung in 9 of 20 rounds under
# this same pause, so 10 rounds miss a return of the race in under 1% of runs.
RACE_ROUNDS = 10
# How long a spin thread pauses in the race window. A plain shutdown reaches
# Executor.shutdown in about 10 ms.
RACE_PAUSE_S = 0.2


class _PausingExecutor(StoppableExecutor):
    """An executor whose spin thread pauses in the shutdown race window, once.

    _wait_for_ready_callbacks reads _guard and then _sigint_gc into its guard
    list, and enters them a few lines later. A pause after both reads stands in
    for a descheduled spin thread, while Executor.shutdown destroys both guard
    conditions.
    """

    def __init__(self) -> None:
        self.pause_armed = threading.Event()
        self.pausing = threading.Event()
        super().__init__()

    @property
    def _sigint_gc(self):
        value = self.__dict__["_sigint_gc"]
        if (
            sys._getframe(1).f_code.co_name == "_wait_for_ready_callbacks"
            and self.pause_armed.is_set()
            and not self.pausing.is_set()
        ):
            self.pausing.set()
            time.sleep(RACE_PAUSE_S)
        return value

    @_sigint_gc.setter
    def _sigint_gc(self, value) -> None:
        self.__dict__["_sigint_gc"] = value


def test_a_connector_shutdown_returns_while_its_spin_thread_is_paused_in_the_race_window(
    monkeypatch,
):
    monkeypatch.setattr(rai_connector_base, "MultiThreadedExecutor", _PausingExecutor)
    waker_node = rclpy.create_node("test_support_race_waker")
    hung = []
    try:
        for round_number in range(RACE_ROUNDS):
            conn = ROS2Connector(node_name=f"test_support_race_{round_number}")
            # An action of its own each round: its status topic is transient
            # local, so a status published to wake one round would reach the
            # clients of the later rounds too, and wake them out of the race.
            action = f"test_support_race_{round_number}"
            ActionClient(conn.node, NavigateToPose, action)
            conn._executor.pause_armed.set()
            conn._executor.wake()
            assert conn._executor.pausing.wait(10), "The spin thread never reached the pause."
            call = Call(conn.shutdown)
            try:
                call.result(timeout_s=10)
                assert not conn._thread.is_alive()
            except AssertionError as error:
                hung.append(f"round {round_number}: {error}")
                # A stuck spin thread is in wait_set.wait with only the
                # ActionClient left to wake it. Without this it would block the
                # session's exit, because rai's spin thread is not a daemon.
                waker = waker_node.create_publisher(
                    GoalStatusArray, f"/{action}/_action/status", qos_profile_action_status_default
                )
                wait_until(
                    lambda: waker.publish(GoalStatusArray()) or not conn._thread.is_alive(),
                    poll_s=0.1,
                )
                call.thread.join(10)
    finally:
        waker_node.destroy_node()
    assert not hung, f"{len(hung)} of {RACE_ROUNDS} rounds hung.\n" + "\n".join(hung)


def test_a_spinner_whose_thread_cannot_stop_fails_its_close(monkeypatch):
    node = rclpy.create_node("test_support_stuck_spinner")
    spinner = Spinner(node)
    # From here on the spin thread is inside spin_until_future_complete, so its
    # stack names it whenever the close looks.
    assert wait_until(lambda: spinner.executor.is_spinning), "The spin never started."
    with monkeypatch.context() as stuck:
        stuck.setattr(spinner.executor, "stop", lambda: None)
        with pytest.raises(AssertionError, match="did not stop(.|\n)*spin_until_future_complete"):
            spinner.close(timeout_s=0.5)
    # The real stop still ends it, so that this test leaves nothing behind.
    spinner.close()
    assert not spinner._thread.is_alive()


def test_the_session_domain_is_exported_and_locked():
    assert os.environ["ROS_DOMAIN_ID"] == str(TEST_DOMAIN_ID)
    assert rclpy.get_default_context().get_domain_id() == TEST_DOMAIN_ID
    fd = os.open(domain_lock_path(TEST_DOMAIN_ID), os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def _unopenable(path: Path) -> None:
    """Make `path` a lock file that no account can open, root included.

    A symlink to itself fails with ELOOP, as another account's lock file
    without read access fails with EACCES.
    """
    path.symlink_to(path)


def test_a_session_skips_a_domain_whose_lock_file_it_cannot_open(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("WALKIE_TEST_DOMAIN_ID", raising=False)
    for domain in range(100, 201):
        if domain != 150:
            _unopenable(domain_lock_path(domain))
    # Read-only, like a lock file that another account left: flock needs no
    # write access, so the session can still take it.
    domain_lock_path(150).touch()
    domain_lock_path(150).chmod(0o444)
    domain, fd = _choose_test_domain()
    os.close(fd)
    assert domain == 150


def test_a_requested_domain_whose_lock_file_cannot_be_opened_fails_the_session(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("WALKIE_TEST_DOMAIN_ID", "150")
    _unopenable(domain_lock_path(150))
    path = domain_lock_path(150)
    message = f"WALKIE_TEST_DOMAIN_ID=150: cannot open the domain's lock file {path}"
    with pytest.raises(pytest.UsageError, match=re.escape(message)):
        _choose_test_domain()


def test_a_session_refuses_a_requested_domain_that_another_session_holds():
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            "tests/test_test_support.py",
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "WALKIE_TEST_DOMAIN_ID": str(TEST_DOMAIN_ID)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode != 0
    assert f"WALKIE_TEST_DOMAIN_ID={TEST_DOMAIN_ID}: another test session" in (
        done.stdout + done.stderr
    )


def _run_with_watchdog(
    directory: Path, *options: str, limit_s: float = 2, stdin: str | None = None
) -> subprocess.CompletedProcess:
    """Run pytest on test_hangs.py in `directory`, with a `limit_s` limit and no conftest."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "hang_watchdog",
            "-o",
            f"walkie_hang_timeout={limit_s:g}",
            *options,
            "test_hangs.py",
        ],
        cwd=directory,
        # The ROS 2 pytest plugins are not needed for a test that only sleeps.
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT / "tests"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_a_hung_test_becomes_an_error_with_stacks(tmp_path):
    (tmp_path / "test_hangs.py").write_text(
        "import time\n\n\ndef test_hangs():\n    time.sleep(3600)\n"
    )
    start = time.monotonic()
    done = _run_with_watchdog(tmp_path)
    elapsed = time.monotonic() - start
    assert done.returncode == 1, done.stdout + done.stderr
    assert "1 failed" in done.stdout or "1 error" in done.stdout, done.stdout
    assert "TestHangError" in done.stdout
    assert "most recent call first" in done.stderr
    assert "test_hangs.py" in done.stderr
    assert elapsed < 30


def test_a_teardown_that_hangs_after_a_hung_call_gets_its_own_error(tmp_path):
    (tmp_path / "test_hangs.py").write_text(
        "import time\n\nimport pytest\n\n\n"
        "@pytest.fixture\ndef hangs_at_teardown():\n    yield\n    time.sleep(3600)\n\n\n"
        "def test_hangs(hangs_at_teardown):\n    time.sleep(3600)\n"
    )
    done = _run_with_watchdog(tmp_path)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "1 failed, 1 error" in done.stdout, done.stdout


def test_a_teardown_that_hangs_still_lets_the_other_teardowns_of_its_test_run(tmp_path):
    # Without them, monkeypatch's undo, which every test here has through the
    # autouse no_tracing fixture, would leave its patches for the whole session.
    (tmp_path / "test_hangs.py").write_text(
        "import time\n\nimport pytest\n\n"
        'STATE = {"patched": False, "torn_down": False}\n\n\n'
        "@pytest.fixture\ndef patched(monkeypatch):\n"
        '    monkeypatch.setitem(STATE, "patched", True)\n\n\n'
        '@pytest.fixture\ndef torn_down(patched):\n    yield\n    STATE["torn_down"] = True\n\n\n'
        "@pytest.fixture\ndef hangs_at_teardown(torn_down):\n    yield\n    time.sleep(3600)\n\n\n"
        "def test_hangs(hangs_at_teardown):\n    pass\n\n\n"
        "def test_the_other_teardowns_ran():\n"
        '    assert STATE == {"patched": False, "torn_down": True}\n'
    )
    # The setup takes milliseconds, so with 4 s even a 2 s freeze during it
    # cannot make the limit fire before the teardown.
    done = _run_with_watchdog(tmp_path, limit_s=4)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "2 passed, 1 error" in done.stdout, done.stdout
    assert "ERROR at teardown of test_hangs" in done.stdout, done.stdout
    assert "TestHangError" in done.stdout


def test_a_module_fixture_that_hangs_at_setup_gives_its_error_to_each_test_that_uses_it(
    tmp_path,
):
    (tmp_path / "test_hangs.py").write_text(
        "import time\n\nimport pytest\n\n\n"
        '@pytest.fixture(scope="module")\ndef hangs_at_setup():\n    time.sleep(3600)\n\n\n'
        "def test_first(hangs_at_setup):\n    pass\n\n\n"
        "def test_second(hangs_at_setup):\n    pass\n"
    )
    done = _run_with_watchdog(tmp_path)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "2 errors" in done.stdout, done.stdout
    for name in ("test_first", "test_second"):
        assert f"ERROR test_hangs.py::{name} - hang_watchdog.TestHangError" in done.stdout, (
            done.stdout
        )


def test_a_hang_under_a_broad_except_still_becomes_an_error(tmp_path):
    # The code a test calls may catch every Exception, as rai's ToolRunner does.
    (tmp_path / "test_hangs.py").write_text(
        "import time\n\n\ndef test_hangs():\n"
        "    try:\n        time.sleep(3600)\n    except Exception:\n        pass\n"
    )
    done = _run_with_watchdog(tmp_path)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "1 failed" in done.stdout, done.stdout
    assert "TestHangError" in done.stdout


@pytest.mark.parametrize(
    ("body", "options", "outcome"),
    [("breakpoint()", (), "1 passed"), ("assert False", ("--pdb",), "1 failed")],
    ids=["breakpoint", "post_mortem"],
)
def test_opening_the_debugger_turns_the_hang_limit_off(tmp_path, body, options, outcome):
    (tmp_path / "test_hangs.py").write_text(f"def test_debugged():\n    {body}\n")
    # The prompt reports the watchdog's timer, then lets the test go on. The
    # limit is far above the time the test takes to reach the prompt, so the
    # check does not depend on how fast that is.
    done = _run_with_watchdog(
        tmp_path,
        *options,
        limit_s=60,
        stdin="!import signal; print('itimer', signal.getitimer(signal.ITIMER_REAL))\ncontinue\n",
    )
    assert "itimer (0.0, 0.0)" in done.stdout, done.stdout + done.stderr
    assert outcome in done.stdout, done.stdout + done.stderr


def test_closing_a_spinner_during_a_drive_aborts_the_drive(connector, monkeypatch):
    fake = FakeWalkie(drive_time_sec=600.0)
    results = []
    drive = fake._drive
    monkeypatch.setattr(fake, "_drive", lambda handle: results.append(drive(handle)) or results[-1])
    spinner = Spinner(fake)
    try:
        client = ActionClient(connector.node, NavigateToPose, "navigate_to_pose")
        assert client.wait_for_server(timeout_sec=10)
        goal = NavigateToPose.Goal()
        goal.pose.pose.position.x = 1.0
        client.send_goal_async(goal)
        assert wait_until(lambda: fake.pose()[0] > 0.0), "The drive never started."
    finally:
        # Without the release, the drive holds its handler past the close's bound.
        spinner.close()
    assert results[0].error_msg == "The fake robot closed during the drive."
    assert fake.pose()[0] < 1.0


def test_a_module_clock_moves_only_when_advanced_and_forwards_the_rest():
    clock = ModuleClock()
    assert clock.monotonic() == 1000.0
    real_start = time.monotonic()
    assert wait_until(lambda: time.monotonic() > real_start)
    assert clock.monotonic() == 1000.0
    clock.advance(2.5)
    assert clock.monotonic() == 1002.5
    assert clock.sleep is time.sleep
    assert clock.perf_counter is time.perf_counter


def test_a_call_gives_the_result_or_the_error_of_its_function():
    assert Call(lambda a, b=0: a + b, 1, b=2).result() == 3

    def fails():
        raise ValueError("from the call")

    with pytest.raises(ValueError, match="from the call"):
        Call(fails).result()


def test_a_call_that_runs_on_fails_with_its_stack():
    entered = threading.Event()
    release = threading.Event()

    def blocks() -> bool:
        entered.set()
        return release.wait()

    call = Call(blocks)
    try:
        # Once entered is set, the call's stack holds blocks until the release.
        assert entered.wait(10), "The call never started."
        with pytest.raises(AssertionError, match="did not return(.|\n)*in blocks"):
            call.result(timeout_s=0.1)
    finally:
        release.set()
    assert call.result() is True


@pytest.fixture
def clocks_moved_on():
    """Clocks to check after module_clock's teardown, as this fixture is set up before it."""
    clocks = []
    yield clocks
    for clock in clocks:
        assert clock.monotonic() >= 1e6


def test_module_clock_patches_the_module_and_moves_on_at_teardown(clocks_moved_on, module_clock):
    class Module:
        time = time

    clock = module_clock(Module)
    assert Module.time is clock
    clocks_moved_on.append(clock)
