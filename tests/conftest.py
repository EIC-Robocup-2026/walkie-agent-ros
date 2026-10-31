import fcntl
import importlib.util
import os
import random
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# Conftest imports these by name, so every session has the hang watchdog.
from hang_watchdog import (  # noqa: F401
    pytest_addoption,
    pytest_configure,
    pytest_enter_pdb,
    pytest_runtest_protocol,
    pytest_sessionfinish,
    pytest_unconfigure,
)

# Tests run on Fast DDS, although pixi sets rmw_zenoh_cpp for every command
# (ADR-0019, ADR-0033). Fast DDS needs no router, and it does not reach the
# Zenoh nodes of a robot or a simulation on the same computer. This must run
# before rclpy is imported: loading rcl checks RMW_IMPLEMENTATION and loads
# that middleware, so a change after the import has no effect.
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
# Keep test traffic on this machine and off any domain a real robot might use.
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"


def domain_lock_path(domain: int) -> Path:
    """The file whose flock marks `domain` as taken by a test session on this machine."""
    return Path(tempfile.gettempdir()) / f"walkie-agent-ros-test-domain-{domain}.lock"


def _lock_domain(domain: int) -> int | None:
    """Lock `domain` for this session: the lock's file descriptor, or None if it is taken.

    flock needs no write access, so the file is opened read-only, and with
    O_CREAT only when it does not exist yet. Lock files stay behind, and one
    that another account left need not be writable by this one; in a sticky
    directory such as /tmp, fs.protected_regular also refuses an O_CREAT open
    of another user's file whatever its mode. A file that still cannot be
    opened raises OSError.
    """
    path = domain_lock_path(domain)
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _choose_test_domain() -> tuple[int, int]:
    """The session's ROS domain and the file descriptor that holds its lock.

    WALKIE_TEST_DOMAIN_ID picks the domain; otherwise it is the first free one
    in a shuffled 100..200, where a domain whose lock file this account cannot
    open counts as taken. The lock ends with the process, so a killed
    session frees its domain. The shell's ROS_DOMAIN_ID is never read: it may
    point at a running simulation or the robot, and FakeWalkie would then
    serve navigate_to_pose and publish /tf there.
    """
    requested = os.environ.get("WALKIE_TEST_DOMAIN_ID")
    if requested is not None:
        try:
            domain = int(requested)
        except ValueError:
            domain = -1
        if not 0 <= domain <= 232:
            raise pytest.UsageError(
                f"WALKIE_TEST_DOMAIN_ID must be an integer from 0 to 232, not {requested!r}."
            )
        try:
            fd = _lock_domain(domain)
        except OSError as error:
            raise pytest.UsageError(
                f"WALKIE_TEST_DOMAIN_ID={domain}: cannot open the domain's lock file "
                f"{domain_lock_path(domain)} ({error.strerror}). Unset the variable, or pick "
                "another domain."
            ) from error
        if fd is None:
            raise pytest.UsageError(
                f"WALKIE_TEST_DOMAIN_ID={domain}: another test session holds this domain "
                f"({domain_lock_path(domain)}). Unset the variable, or pick another domain."
            )
        return domain, fd
    domains = list(range(100, 201))
    random.shuffle(domains)
    for domain in domains:
        try:
            fd = _lock_domain(domain)
        except OSError:
            # Such as a lock file that another account left without read access.
            continue
        if fd is not None:
            return domain, fd
    raise pytest.UsageError(
        "Every test domain from 100 to 200 is held by another test session, or has a lock file "
        "that this account cannot open."
    )


# No two sessions on this machine share a domain, which random.randint alone
# could not promise. The descriptor stays open for the whole session, because
# closing it would release the lock. The export makes a subprocess that a test
# starts join the same domain.
TEST_DOMAIN_ID, _TEST_DOMAIN_LOCK_FD = _choose_test_domain()
os.environ["ROS_DOMAIN_ID"] = str(TEST_DOMAIN_ID)

# The message packages from ros2_ws.repos, built by `pixi run build-ros2`.
# scripts/activate.sh adds them to the environment only once they are built.
# Without them, an import fails with an ImportError that does not say what to
# do: the FakeWalkie import below needs rai_interfaces, and the test modules
# import walkie_database_msgs. The check runs here, before those imports,
# because pytest imports this file before it calls pytest_configure.
_missing = [
    name
    for name in ("rai_interfaces", "walkie_database_msgs")
    if importlib.util.find_spec(name) is None
]
if _missing:
    raise pytest.UsageError(
        f"Message packages not built: {', '.join(_missing)}. "
        "Run `pixi run build-ros2` once, then run the tests again."
    )

import rclpy  # noqa: E402
import rclpy.task  # noqa: E402
from nav2_msgs.action import NavigateToPose  # noqa: E402
from rai.communication.ros2 import ROS2Connector  # noqa: E402
from rai.communication.ros2.connectors import base as rai_connector_base  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402

from walkie_agent.locations import load_locations  # noqa: E402
from walkie_agent.sim.fake_walkie import FakeWalkie  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The executor threads of a fake's executor. A running drive and a held answer
# each keep one until they end, so a test can need several at once. Without
# num_threads, MultiThreadedExecutor takes one per CPU the process may use, and
# on a runner with one or two CPUs the fake would then not answer.
FAKE_THREADS = 8


def _stack_of(thread: threading.Thread) -> str:
    """The stack of `thread` right now, for a failure message."""
    frame = sys._current_frames().get(thread.ident)
    if frame is None:
        return f"{thread.name}: no longer running\n"
    return f"{thread.name}:\n{''.join(traceback.format_stack(frame))}"


def wait_until(
    predicate: Callable[[], Any], timeout_s: float = 10.0, poll_s: float = 0.01
) -> bool:
    """Whether `predicate` came true within `timeout_s`. For positive waits only.

    It returns as soon as the predicate holds, so the bound can stay far above
    a stalled runner's delays. Do not use it to show that something stays
    false: that would wait out the whole bound.
    """
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)
    return True


def wait_for_nav_server(node, timeout_s: float = 10.0) -> None:
    """Wait until `node` sees a navigate_to_pose server, such as the fake robot's.

    The navigation tools wait server_timeout_sec (2 s) for the server before
    they say Nav2 is unavailable; after this wait, that budget only meets a
    server already on the graph. The client is a throwaway, destroyed here.
    """
    client = ActionClient(node, NavigateToPose, "navigate_to_pose")
    try:
        assert client.wait_for_server(timeout_sec=timeout_s), (
            f"No navigate_to_pose server within {timeout_s} s."
        )
    finally:
        client.destroy()


def wait_for_subscriber(node, topic: str, timeout_s: float = 10.0) -> None:
    """Wait until `node` sees a subscriber on `topic`, such as the fake robot's /to_human.

    SpeakTool waits subscriber_wait_sec (2 s) for one before it says nothing
    was said; after this wait, that budget only meets a subscriber already on
    the graph.
    """
    assert wait_until(lambda: node.count_subscribers(topic) > 0, timeout_s), (
        f"No subscriber on {topic} within {timeout_s} s."
    )


class StoppableExecutor(MultiThreadedExecutor):
    """An executor whose spin ends before anything it waits on is destroyed.

    rclpy's Executor.shutdown destroys _guard and _sigint_gc without waiting
    for the spin thread: its "if not self._is_shutdown" check is always false
    once it has set that flag. A thread descheduled between reading the guards
    and entering them then waits in wait_set.wait with nothing to wake it, and
    the join after the shutdown never returns. So stop() first, join the
    thread, and only then shutdown().

    stop_future has no executor, so its set_result wakes the spin inline, while
    _guard is still alive.
    """

    def __init__(self, num_threads: int | None = FAKE_THREADS, **kwargs: Any) -> None:
        super().__init__(num_threads, **kwargs)
        self.stop_future = rclpy.task.Future()

    def spin(self) -> None:
        self.spin_until_future_complete(self.stop_future)

    def stop(self) -> None:
        """End the spin. A call after the first does nothing."""
        if not self.stop_future.done():
            self.stop_future.set_result(True)

    def shutdown(self, timeout_sec: float | None = None) -> bool:
        finished = super().shutdown(timeout_sec)
        # Otherwise the idle pool workers of every executor stay until exit.
        self._executor.shutdown(wait=False)
        return finished


# Every rai connector takes its executor from this module global when it is
# built: ROS2Connector, ROS2HRIConnector and those inside the HalfDuplex
# agents. A rai upgrade that renames it fails here, at collection, instead of
# bringing the shutdown race back unnoticed.
assert hasattr(rai_connector_base, "MultiThreadedExecutor"), (
    "rai.communication.ros2.connectors.base no longer has MultiThreadedExecutor, "
    "so tests/conftest.py cannot give its connectors a StoppableExecutor."
)
rai_connector_base.MultiThreadedExecutor = StoppableExecutor

_rai_connector_shutdown = rai_connector_base.ROS2BaseConnector.shutdown


def _connector_shutdown(self: rai_connector_base.ROS2BaseConnector) -> None:
    """rai's shutdown, after the spin thread has ended (see StoppableExecutor)."""
    self._executor.stop()
    self._thread.join(10)
    if self._thread.is_alive():
        stack = _stack_of(self._thread)
        self._tf_listener.unregister()
        self._node.destroy_node()
        # rai's shutdown would reach an untimed join on this thread.
        raise AssertionError(f"A connector's spin thread did not stop within 10 s.\n{stack}")
    # rai never destroys the ActionClients of its send_goal.
    for waitable in list(self._node.waitables):
        waitable.destroy()
    _rai_connector_shutdown(self)


rai_connector_base.ROS2BaseConnector.shutdown = _connector_shutdown


class Spinner:
    """Spin nodes on a StoppableExecutor of a test's own, on a daemon thread.

    FAKE_THREADS is the default, so one held callback cannot starve a fake's
    timers. A fake that is single-threaded on purpose passes num_threads=1.

    A close that fails its assert has found a real leak: fix the release order,
    never raise the bound.
    """

    def __init__(self, *nodes: Any, num_threads: int = FAKE_THREADS) -> None:
        self.nodes = nodes
        self.executor = StoppableExecutor(num_threads=num_threads)
        for node in nodes:
            self.executor.add_node(node)
        self._thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._thread.start()

    def close(self, timeout_s: float = 10) -> None:
        """Stop the spin, destroy the nodes, and end the handlers still running.

        A fake's destroy_node releases the work it holds, as FakeWalkie's ends
        its running drives, so a handler left past `timeout_s` is a leak and
        fails the close.
        """
        self.executor.stop()
        self._thread.join(timeout_s)
        assert not self._thread.is_alive(), (
            f"A Spinner's thread did not stop within {timeout_s} s.\n{_stack_of(self._thread)}"
        )
        for node in self.nodes:
            node.destroy_node()
        handlers = self.executor._futures
        try:
            assert wait_until(
                lambda: all(handler.done() or handler.cancelled() for handler in handlers),
                timeout_s,
            ), (
                f"A Spinner's callback did not end within {timeout_s} s of its nodes closing.\n"
                + "".join(_stack_of(thread) for thread in self.executor._executor._threads)
            )
        finally:
            self.executor.shutdown()


class ModuleClock:
    """A manual monotonic() for a module's `time` global, which a test moves on.

    Every other attribute (sleep, time, perf_counter and so on) is the real
    time module's. Use the module_clock fixture to install one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._now = 1000.0

    def monotonic(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


@pytest.fixture
def module_clock(monkeypatch):
    """A function that sets `module.time` to a new ModuleClock and returns it.

    At teardown every clock moves on by 1e6 s, so no call stays parked on one.
    """
    clocks = []

    def patch(module) -> ModuleClock:
        clock = ModuleClock()
        monkeypatch.setattr(module, "time", clock)
        clocks.append(clock)
        return clock

    yield patch
    for clock in clocks:
        clock.advance(1e6)


class Call:
    """Run `fn(*args, **kwargs)` on a daemon thread, for a call a test has to watch."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._value: Any = None
        self._error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, args=(fn, args, kwargs), daemon=True)
        self.thread.start()

    def _run(self, fn: Callable[..., Any], args: tuple, kwargs: dict) -> None:
        try:
            self._value = fn(*args, **kwargs)
        except BaseException as error:
            self._error = error

    def result(self, timeout_s: float = 10) -> Any:
        """What the call returned, or its exception raised here; fails if it runs on."""
        self.thread.join(timeout_s)
        assert not self.thread.is_alive(), (
            f"The call did not return within {timeout_s} s.\n{_stack_of(self.thread)}"
        )
        if self._error is not None:
            raise self._error
        return self._value


def pytest_report_header(config) -> str:
    return f"ros domain: {TEST_DOMAIN_ID}"


@pytest.fixture(scope="session", autouse=True)
def ros():
    rclpy.init(domain_id=TEST_DOMAIN_ID)
    yield
    rclpy.try_shutdown()


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    """Keep tests off Langfuse/LangSmith whatever config.toml enables.

    RAI reads the tracing settings from ./config.toml when an agent is built
    and again on every LLM call, so both lookups are patched.
    """
    monkeypatch.setattr("rai.agents.langchain.agent.get_tracing_callbacks", lambda *_: [])
    monkeypatch.setattr(
        "rai.agents.langchain.invocation_helpers.get_tracing_callbacks", lambda *_: []
    )


@pytest.fixture
def locations():
    return load_locations(PROJECT_ROOT / "config" / "locations.toml")


@pytest.fixture
def fake_walkie():
    node = FakeWalkie(drive_time_sec=0.2)
    spinner = Spinner(node)
    yield node
    spinner.close()


@pytest.fixture
def connector():
    conn = ROS2Connector(node_name=f"test_connector_{random.randint(0, 1_000_000)}")
    yield conn
    conn.shutdown()
