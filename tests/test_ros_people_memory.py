"""RosPeopleMemory against a fake walkie-database node over Fast DDS (no LLM).

Each test's fake serves the four people services under a node name of its own, so a
fake torn down in one test never still looks discovered in the next. The fake spins on
one thread of its own, so it answers one request at a time, in the order they came,
and the connector's executor completes the client's futures, as on the robot.
"""

import itertools
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pytest
from rclpy.client import Client
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from ros_fakes import WATCHDOG_S, Background, bounded
from walkie_database_msgs.msg import Fact, ModelInfo, PersonMatch, PersonMatches
from walkie_database_msgs.srv import MatchPeople, RememberPerson, StartSession

from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.perception.people_memory import Match, ModelVector, Stored
from walkie_agent.ros import people_memory as people_memory_module
from walkie_agent.ros.people_memory import RosPeopleMemory

LOGGER = "walkie_agent.ros.people_memory"
FACE = "arcface_w600k_r50"
APPEARANCE = "osnet_x1_0_msmt17"
SERVICES = {
    "start_session": StartSession,
    "remember_person": RememberPerson,
    "match_faces": MatchPeople,
    "match_appearances": MatchPeople,
}
DATABASE_FAILURE = "The database could not start a session: database is locked"
MATCH_FAILURE = "The database could not match faces: disk I/O error"
MAY_HAVE_STORED = (
    " The person may have been stored already, so remembering them again may store "
    "them twice."
)
# Short timeouts are flaky under Fast DDS discovery, so the tests wait 2 s unless they
# test a timeout, and a timeout test lowers it only for the call under test.
TIMEOUTS = {"server_timeout_s": 2.0, "call_timeout_s": 2.0, "clear_timeout_s": 2.0}

# Names unique within this run and among runs on this computer.
_NAMES = itertools.count(1)


def unique_name(prefix: str) -> str:
    return f"{prefix}_{os.getpid()}_{next(_NAMES)}"


def vector(i: int = 0, model: str = FACE) -> ModelVector:
    """A vector whose first value is `i`, so the fake can tell the queries apart."""
    return ModelVector(model, np.array([float(i), 1.0, 0.0, 0.0], dtype=np.float32))


def by_index(queries) -> list[list[PersonMatch]]:
    """One match per query, whose person id is the query's first value."""
    return [
        [PersonMatch(person_id=int(q.embedding[0]), session_id=1, name="p", distance=0.1)]
        for q in queries
    ]


class FakeDatabase(Node):
    """Serves ~/start_session, ~/remember_person, ~/match_faces and ~/match_appearances.

    serving() spins it on one thread, so it answers one request at a time, in the
    order they came.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.requests: dict[str, list] = {service: [] for service in SERVICES}
        # The monotonic time at which each request arrived.
        self.arrivals: dict[str, list[float]] = {service: [] for service in SERVICES}
        # Every answer waits on this gate. A test clears it to hold the answers, as a
        # busy node would, and sets it to let them go.
        self.answer_gate = threading.Event()
        self.answer_gate.set()
        # Set once an answer that the gate held has been sent.
        self.late_handled = threading.Event()
        # Whether the answer being made now was held. One thread serves, so it is sent
        # before the next request's callback runs.
        self._held = False
        # By service: the message of every answer, which then has success=False.
        self.failure: dict[str, str] = {}
        # By service: the messages of the next answers that fail, in order.
        self.fail_next: dict[str, list[str]] = {}
        self.face_models = {FACE: 4}
        self.appearance_models = {APPEARANCE: 4}
        # The match lists of both match services, one per query.
        self.results = by_index
        self.session_id = 0
        self.person_id = 40
        answers = {
            "start_session": self._start,
            "remember_person": self._remember,
            "match_faces": self._match,
            "match_appearances": self._match,
        }
        for service, srv_type in SERVICES.items():
            server = self.create_service(
                srv_type, f"~/{service}", self._serve(service, answers[service])
            )
            # The executor sends what the callback returns, so only a wrapped
            # send_response knows when a held answer has gone out.
            server.send_response = self._sent(server.send_response)

    def _serve(self, service, answer):
        def callback(request, response):
            # Read before the request is recorded: a test that sees the request may
            # open the gate at once.
            self._held = not self.answer_gate.is_set()
            self.arrivals[service].append(time.monotonic())
            self.requests[service].append(request)
            self.answer_gate.wait()
            queued = self.fail_next.get(service)
            failure = queued.pop(0) if queued else self.failure.get(service)
            if failure is not None:
                response.success = False
                response.message = failure
                return response
            response.success = True
            answer(request, response)
            return response

        return callback

    def _sent(self, send_response):
        """`send_response`, then late_handled set if the gate held the answer."""

        def send(response, header):
            send_response(response, header)
            if self._held:
                self._held = False
                self.late_handled.set()

        return send

    def _start(self, request, response):
        self.session_id += 1
        response.session_id = self.session_id
        response.resumed = request.resume
        response.message = f"Started session {self.session_id}."
        response.face_models = [ModelInfo(model=m, length=n) for m, n in self.face_models.items()]
        response.appearance_models = [
            ModelInfo(model=m, length=n) for m, n in self.appearance_models.items()
        ]

    def _remember(self, request, response):
        self.person_id += 1
        response.person_id = self.person_id
        response.session_id = self.session_id
        response.message = f"Remembered {request.name} as person {self.person_id}."

    def _match(self, request, response):
        response.results = [
            PersonMatches(people=people) for people in self.results(request.queries)
        ]


@contextmanager
def serving(name: str) -> Iterator[FakeDatabase]:
    """A FakeDatabase named `name`, spinning on one thread of its own."""
    node = FakeDatabase(name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield node
    finally:
        # A held answer goes out at once, so its callback ends before the executor.
        node.answer_gate.set()
        # shutdown() does not wait for spin(), and a destroy under it can raise there.
        executor.shutdown()
        thread.join(timeout=2)
        node.destroy_node()


@pytest.fixture
def database(ros):
    with serving(unique_name("fake_walkie_database")) as node:
        yield node


@pytest.fixture
def cancel():
    return CancelToken()


@pytest.fixture
def make_memory(connector, database, cancel):
    """A RosPeopleMemory factory on the fake's services; keyword arguments go to it."""

    def make(**kwargs) -> RosPeopleMemory:
        options = {**TIMEOUTS, **kwargs}
        return RosPeopleMemory(connector, cancel, namespace=f"/{database.get_name()}", **options)

    return make


@pytest.fixture
def memory(make_memory):
    return make_memory()


@pytest.fixture
def no_database(connector, cancel):
    """A RosPeopleMemory whose namespace no node serves."""
    namespace = f"/{unique_name('no_walkie_database')}"
    return RosPeopleMemory(connector, cancel, namespace=namespace, server_timeout_s=0.5)


@pytest.fixture
def no_retries(monkeypatch):
    """A database failure is UNAVAILABLE at once, for the tests of the session rules."""
    monkeypatch.setattr(people_memory_module, "RETRY_BUDGET_S", 0.0)


def wait_until(done: Callable[[], object], what: str, timeout_s: float = WATCHDOG_S) -> None:
    """Wait until done() is true, or raise AssertionError that names what never happened."""
    deadline = time.monotonic() + timeout_s
    while not done():
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed out after {timeout_s} s waiting for {what}.")
        time.sleep(0.005)


def discovered(memory: RosPeopleMemory) -> None:
    """Wait until the fake's start_session is discovered, as clear() does not wait."""
    assert memory._start_client.wait_for_service(timeout_sec=5.0)


def resumes(database: FakeDatabase) -> list[bool]:
    return [request.resume for request in database.requests["start_session"]]


def fails(call, *args, **kwargs) -> StepError:
    with pytest.raises(StepError) as error:
        call(*args, **kwargs)
    return error.value


def warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING
    ]


def spy_on_requests(monkeypatch) -> tuple[list, list]:
    """The futures the clients drop, and the sequence numbers of answers that find none.

    rclpy also calls remove_pending_request when a future completes, so only the drops
    recorded before a late answer comes are RosPeopleMemory's.
    """
    dropped, missed = [], []
    remove_pending_request = Client.remove_pending_request
    get_pending_request = Client.get_pending_request

    def record_drop(client, future):
        dropped.append(future)
        remove_pending_request(client, future)

    def record_miss(client, sequence_number):
        try:
            return get_pending_request(client, sequence_number)
        except KeyError:
            # The executor then throws the answer away.
            missed.append(sequence_number)
            raise

    monkeypatch.setattr(Client, "remove_pending_request", record_drop)
    monkeypatch.setattr(Client, "get_pending_request", record_miss)
    return dropped, missed


def send_held_answer(database: FakeDatabase, service: str, count: int) -> None:
    """Open the gate once `service` has had `count` requests, and wait until the held one is sent.

    The wait for the request comes first: one that arrived after the gate opened would
    not be held, and late_handled would never be set.
    """
    wait_until(lambda: len(database.requests[service]) == count, f"request {count} of {service}")
    database.answer_gate.set()
    assert database.late_handled.wait(WATCHDOG_S), f"The held answer of {service} was never sent."


# -- Sessions ---------------------------------------------------------------------------


def test_default_mode_forgets_at_the_first_call_and_at_each_new_task(database, memory):
    memory.match_faces([vector()])
    memory.match_faces([vector()])
    memory.clear()
    memory.clear()

    assert resumes(database) == [False, False, False]
    assert {request.label for request in database.requests["start_session"]} == {"walkie-agent"}
    assert len(database.requests["match_faces"]) == 2


def test_a_reset_before_any_call_forgets_in_resume_mode(database, make_memory):
    # The first task begins without a reset, so a reset that comes before any people
    # call ends it, and the next task must not see the guests from before the restart.
    memory = make_memory(resume_session=True)
    discovered(memory)

    memory.clear()
    assert resumes(database) == [False]
    memory.match_faces([vector()])

    assert resumes(database) == [False]
    assert len(database.requests["match_faces"]) == 1


def test_a_first_task_without_a_reset_uses_up_the_resume(database, make_memory):
    # The first task after start-up needs no reset, so the first call comes before
    # any clear().
    memory = make_memory(resume_session=True)

    memory.match_faces([vector()])
    memory.clear()

    assert resumes(database) == [True, False]


def test_a_failed_forget_stays_pending_until_the_next_call(database, memory, no_retries):
    memory.match_faces([vector()])
    database.failure["start_session"] = DATABASE_FAILURE

    memory.clear()
    # The forget fails again, so the call fails rather than match the last task's
    # guests.
    error = fails(memory.match_faces, [vector()])
    del database.failure["start_session"]
    memory.match_faces([vector()])

    assert (error.code, error.message) == (ErrorCode.UNAVAILABLE, DATABASE_FAILURE)
    assert resumes(database) == [False, False, False, False]
    assert len(database.requests["match_faces"]) == 2


def test_a_failed_resume_then_a_failed_forget_starts_fresh(database, make_memory, no_retries):
    memory = make_memory(resume_session=True)
    discovered(memory)
    database.failure["start_session"] = DATABASE_FAILURE

    error = fails(memory.match_faces, [vector()])  # the first task's resume fails
    memory.clear()  # the reset's forget fails
    del database.failure["start_session"]
    memory.match_faces([vector()])

    # The second task must not resume the session from before the restart.
    assert error.code is ErrorCode.UNAVAILABLE
    assert resumes(database) == [True, False, False]
    assert len(database.requests["match_faces"]) == 1


def test_clear_without_a_database_returns_at_once_and_warns_once(no_database, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        start = time.monotonic()
        no_database.clear()
        no_database.clear()
        elapsed = time.monotonic() - start

    assert elapsed < 0.5
    [warning] = warnings(caplog)
    assert "forget stays pending" in warning


def test_a_reset_before_the_node_is_found_forgets_once_it_is(connector, cancel, caplog):
    # The agent and the node restart together, and the operator resets before
    # start_session is discovered. The reset must not wait, and its forget, not the
    # resume, waits for the next call.
    name = unique_name("fake_walkie_database")
    options = {**TIMEOUTS, "server_timeout_s": 5.0}
    memory = RosPeopleMemory(
        connector, cancel, namespace=f"/{name}", resume_session=True, **options
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        start = time.monotonic()
        memory.clear()
        elapsed = time.monotonic() - start
    with serving(name) as database:
        memory.match_faces([vector()])

    assert elapsed < 0.5
    [warning] = warnings(caplog)
    assert "forget stays pending" in warning
    assert resumes(database) == [False]


def test_the_first_call_waits_for_the_node_and_resumes(connector, cancel):
    # The agent and the node restart together, and the first task's first call
    # comes before start_session is discovered: it waits for it and resumes.
    name = unique_name("fake_walkie_database")
    options = {**TIMEOUTS, "server_timeout_s": 5.0}
    memory = RosPeopleMemory(
        connector, cancel, namespace=f"/{name}", resume_session=True, **options
    )

    with serving(name) as database:
        memory.match_faces([vector()])
        memory.clear()

    assert resumes(database) == [True, False]


@pytest.mark.parametrize(
    ("clear_timeout_s", "timeout_s"), [(0.3, None), (5.0, 0.3)], ids=["clear_timeout_s", "timeout_s"]
)
def test_clear_waits_at_most_its_timeout_on_a_silent_node(
    database, make_memory, monkeypatch, caplog, clear_timeout_s, timeout_s
):
    # A node that is found but does not answer, such as one built from another
    # walkie_database_msgs under Zenoh (ADR-0035), costs a new task the clear
    # timeout, or the shorter timeout_s, not the call timeout.
    memory = make_memory(call_timeout_s=5.0, clear_timeout_s=clear_timeout_s)
    discovered(memory)
    waits = []
    wait = memory._wait

    def record_wait(done, wait_s, what, cancellable):
        waits.append(wait_s)
        return wait(done, wait_s, what, cancellable)

    monkeypatch.setattr(memory, "_wait", record_wait)
    # The node holds its answer for the whole call.
    database.answer_gate.clear()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        start = time.monotonic()
        memory.clear(timeout_s=timeout_s)
        elapsed = time.monotonic() - start
    clear_waits = list(waits)
    database.answer_gate.set()
    # The forget stays pending, so the next call starts a session first.
    memory.match_faces([vector()])

    assert clear_waits and max(clear_waits) <= 0.3
    # The spy holds the claim; the bound only shows the call timeout was not waited
    # out, with room for a stalled runner.
    assert elapsed < 5.0
    assert any("could not forget" in warning for warning in warnings(caplog))
    assert resumes(database) == [False, False]
    assert len(database.requests["match_faces"]) == 1


def test_a_failed_forget_warns_once_until_a_start_succeeds(
    database, make_memory, caplog, no_retries
):
    memory = make_memory()
    discovered(memory)
    database.failure["start_session"] = DATABASE_FAILURE

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        memory.clear()  # the first reset's forget fails: a warning
        memory.clear()  # the second reset's forget fails: no second warning
        del database.failure["start_session"]
        memory.match_faces([vector()])  # a start succeeds
        database.failure["start_session"] = DATABASE_FAILURE
        memory.clear()  # a failed forget warns again

    assert resumes(database) == [False, False, False, False]
    assert [warning.split(" (")[0] for warning in warnings(caplog)] == [
        "People memory could not forget the last task's guests",
        "People memory could not forget the last task's guests",
    ]


def test_clear_forgets_while_the_token_is_set(database, memory, cancel, caplog):
    # The Runner resets the conversation, and so calls clear(), before the next task
    # releases the token.
    memory.match_faces([vector()])
    cancel.cancel()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        memory.clear()
    cancel.reset()
    memory.match_faces([vector()])

    assert warnings(caplog) == []
    assert resumes(database) == [False, False]
    assert len(database.requests["match_faces"]) == 2


def test_a_clear_that_cannot_take_the_lock_leaves_the_forget_pending(
    database, make_memory, caplog
):
    memory = make_memory(call_timeout_s=WATCHDOG_S, clear_timeout_s=0.3)
    memory.match_faces([vector()])
    database.answer_gate.clear()
    held = Background(memory.match_faces, [vector()])
    wait_until(lambda: len(database.requests["match_faces"]) == 2, "the held match")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        # The held match keeps the lock until the gate opens, which comes only after
        # clear() returns; a clear() that waited for the lock would never return.
        bounded(memory.clear)
    database.answer_gate.set()
    held.result()
    memory.match_faces([vector()])

    [warning] = warnings(caplog)
    assert "busy with another call" in warning
    assert resumes(database) == [False, False]
    assert len(database.requests["match_faces"]) == 3


# -- Errors -----------------------------------------------------------------------------


def test_no_server_is_unavailable(no_database):
    error = fails(no_database.match_faces, [vector()])

    service = no_database._start_client.srv_name
    assert service.endswith("/start_session")
    assert error.code is ErrorCode.UNAVAILABLE
    assert error.message == (
        f"People memory is unavailable: no node offers {service}. The walkie-database node "
        "is not running, or not on this ROS domain."
    )


def test_a_slow_server_times_out_and_drops_the_pending_request(
    database, memory, monkeypatch, caplog
):
    # A busy node and one built from another walkie_database_msgs look the same, so
    # both are dropped with a hint.
    memory.match_faces([vector()])
    # Only the call under test has the short timeout, and the node holds its answer,
    # so the answer cannot come in time.
    memory._call_timeout_s = 0.3
    dropped, missed = spy_on_requests(monkeypatch)
    database.answer_gate.clear()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        error = fails(memory.match_faces, [vector()])

    assert error.code is ErrorCode.TIMEOUT
    assert error.message == (
        "People memory did not answer within 0.3 s. It may be busy; try again once."
    )
    assert len(dropped) == 1
    [warning] = warnings(caplog)
    assert "pinned in ros2_ws.repos" in warning
    # The late answer goes out, and the client has no request left for it.
    send_held_answer(database, "match_faces", 2)
    wait_until(lambda: missed, "an answer for a request the client had dropped")
    assert not dropped[0].done()


def test_a_slow_remember_says_it_may_have_stored_the_person(
    database, memory, monkeypatch, caplog
):
    memory.match_faces([vector()])
    # As above: the short timeout and the held answer are the remember's only.
    memory._call_timeout_s = 0.3
    dropped, missed = spy_on_requests(monkeypatch)
    database.answer_gate.clear()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        error = fails(memory.remember, "Sam", {}, [vector()], [])

    assert error.code is ErrorCode.TIMEOUT
    assert error.message == (
        f"People memory did not answer within 0.3 s. It may be busy; try again once.{MAY_HAVE_STORED}"
    )
    # A remember that got no answer is never sent again.
    assert len(database.requests["remember_person"]) == 1
    assert len(dropped) == 1
    [warning] = warnings(caplog)
    assert "pinned in ros2_ws.repos" in warning
    send_held_answer(database, "remember_person", 1)
    wait_until(lambda: missed, "an answer for a request the client had dropped")
    assert not dropped[0].done()
    assert len(database.requests["remember_person"]) == 1


def test_any_other_refusal_is_refused_at_once(database, memory):
    database.failure["remember_person"] = "At most 32 facts, got 33."

    error = fails(memory.remember, "Sam", {"drink": "milk"}, [vector()], [])

    assert (error.code, error.message) == (ErrorCode.REFUSED, "At most 32 facts, got 33.")
    assert len(database.requests["remember_person"]) == 1


def test_a_results_length_mismatch_is_a_runtime_error(database, memory):
    database.results = lambda queries: by_index(queries)[1:]

    with pytest.raises(RuntimeError, match="answered 1 of 2 queries"):
        memory.match_faces([vector(0), vector(1)])


# -- Database failures, asked again (P25) -------------------------------------------------


def test_a_database_failure_is_asked_again_until_it_passes(database, memory):
    # Under sim time, walkie-database fails every start until the first /clock.
    database.fail_next["start_session"] = [DATABASE_FAILURE, DATABASE_FAILURE]

    [matches] = memory.match_faces([vector(3)])

    assert [m.person_id for m in matches] == ["stored_3"]
    assert resumes(database) == [False, False, False]
    arrivals = database.arrivals["start_session"]
    assert all(
        later - earlier >= people_memory_module.RETRY_STEP_S
        for earlier, later in itertools.pairwise(arrivals)
    )


def test_a_remember_that_met_a_database_failure_is_sent_again(database, memory):
    # A database failure stored nothing, so sending the request again stores no one twice.
    database.fail_next["remember_person"] = ["The database could not remember Sam: locked"]

    stored = memory.remember("Sam", {}, [vector()], [])

    assert stored == Stored("stored_41", 1, 0)
    assert [request.name for request in database.requests["remember_person"]] == ["Sam", "Sam"]


def test_a_database_failure_that_lasts_is_unavailable_after_the_retries(database, memory):
    memory.match_faces([vector()])
    database.failure["match_faces"] = MATCH_FAILURE

    start = time.monotonic()
    error = fails(memory.match_faces, [vector()])
    elapsed = time.monotonic() - start

    assert (error.code, error.message) == (ErrorCode.UNAVAILABLE, MATCH_FAILURE)
    budget_s = people_memory_module.RETRY_BUDGET_S
    step_s = people_memory_module.RETRY_STEP_S
    assert budget_s - step_s <= elapsed < budget_s + 2.0
    # The first match, the failed one, and its retries.
    assert 3 <= len(database.requests["match_faces"]) <= 1 + 1 + round(budget_s / step_s)


def test_the_retries_stop_at_the_calls_deadline(database, memory):
    memory.match_faces([vector()])
    database.failure["match_faces"] = MATCH_FAILURE

    start = time.monotonic()
    error = fails(memory.match_faces, [vector()], timeout_s=0.6)
    elapsed = time.monotonic() - start

    assert (error.code, error.message) == (ErrorCode.UNAVAILABLE, MATCH_FAILURE)
    assert elapsed < people_memory_module.RETRY_BUDGET_S
    # One retry at most every RETRY_STEP_S within the 0.6 s.
    assert 2 <= len(database.requests["match_faces"]) <= 1 + 3


def test_a_stop_during_the_pause_before_a_retry_ends_the_call(
    database, memory, cancel, monkeypatch
):
    # A pause far past the watchdog, so only a pause that ends at the stop lets the
    # call end in time.
    monkeypatch.setattr(people_memory_module, "RETRY_STEP_S", 30.0)
    monkeypatch.setattr(people_memory_module, "RETRY_BUDGET_S", 60.0)
    memory.match_faces([vector()])
    database.failure["match_faces"] = MATCH_FAILURE
    pausing = threading.Event()
    pause = memory._pause

    def record_pause(seconds, what, cancellable):
        pausing.set()
        return pause(seconds, what, cancellable)

    monkeypatch.setattr(memory, "_pause", record_pause)

    call = Background(memory.match_faces, [vector()])
    assert pausing.wait(WATCHDOG_S), "The call never paused before a retry."
    cancel.cancel()
    error = call.step_error()

    assert error.code is ErrorCode.CANCELLED
    assert len(database.requests["match_faces"]) == 2


# -- Stops and deadlines ------------------------------------------------------------------


def test_a_stop_during_the_wait_for_an_answer_ends_it_and_drops_the_request(
    database, make_memory, cancel, monkeypatch
):
    # A call timeout far past the watchdog, so only a wait that ends at the stop lets
    # the call end in time.
    memory = make_memory(call_timeout_s=30.0)
    memory.match_faces([vector()])
    dropped, _ = spy_on_requests(monkeypatch)
    database.answer_gate.clear()

    call = Background(memory.match_faces, [vector()])
    wait_until(lambda: len(database.requests["match_faces"]) == 2, "the held match")
    cancel.cancel()
    error = call.step_error()

    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        "Cancelled matching faces: a stop is in effect.",
    )
    assert len(dropped) == 1
    assert not dropped[0].done()


def test_a_stop_during_a_remember_says_it_may_have_stored_the_person(
    database, make_memory, cancel, monkeypatch
):
    memory = make_memory(call_timeout_s=30.0)
    memory.match_faces([vector()])
    dropped, _ = spy_on_requests(monkeypatch)
    database.answer_gate.clear()

    call = Background(memory.remember, "Sam", {}, [vector()], [])
    wait_until(lambda: database.requests["remember_person"], "the held remember")
    cancel.cancel()
    error = call.step_error()

    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        f"Cancelled remembering a person: a stop is in effect.{MAY_HAVE_STORED}",
    )
    assert len(dropped) == 1


def test_a_call_during_a_stop_sends_nothing(database, memory, cancel):
    discovered(memory)
    cancel.cancel()

    remember = fails(memory.remember, "Sam", {}, [vector()], [])
    match = fails(memory.match_appearances, [vector(model=APPEARANCE)])

    assert (remember.code, remember.message) == (
        ErrorCode.CANCELLED,
        "Cancelled remembering a person: a stop is in effect.",
    )
    assert match.code is ErrorCode.CANCELLED
    assert all(not requests for requests in database.requests.values())


def test_timeout_s_cuts_the_wait_for_an_answer(database, memory):
    memory.match_faces([vector()])
    database.answer_gate.clear()

    start = time.monotonic()
    error = fails(memory.match_faces, [vector()], timeout_s=0.3)
    elapsed = time.monotonic() - start

    assert (error.code, error.message) == (
        ErrorCode.TIMEOUT,
        "People memory did not answer within 0.3 s. It may be busy; try again once.",
    )
    assert elapsed < TIMEOUTS["call_timeout_s"]


def test_a_remember_whose_time_ran_out_before_it_was_sent_says_nothing_was_stored(
    database, memory
):
    # The session's start is held, so the remember itself never goes out, and its
    # TIMEOUT does not say that the person may have been stored.
    discovered(memory)
    database.answer_gate.clear()

    error = fails(memory.remember, "Sam", {}, [vector()], [], timeout_s=0.3)

    assert (error.code, error.message) == (
        ErrorCode.TIMEOUT,
        "People memory did not answer within 0.3 s. It may be busy; try again once.",
    )
    assert len(database.requests["start_session"]) == 1
    assert database.requests["remember_person"] == []


# -- Models, chunks and order -------------------------------------------------------------


def test_unknown_models_get_no_matches_and_are_not_sent(database, memory, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        results = memory.match_faces([vector(0), vector(1, "fake_face"), vector(2)])

    [request] = database.requests["match_faces"]
    assert [(q.model, int(q.embedding[0])) for q in request.queries] == [(FACE, 0), (FACE, 2)]
    assert [[m.person_id for m in matches] for matches in results] == [
        ["stored_0"],
        [],
        ["stored_2"],
    ]
    [warning] = warnings(caplog)
    assert "fake_face" in warning


def test_remember_drops_unknown_models_and_refuses_when_none_are_left(database, memory, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        stored = memory.remember(
            "Sam", {}, [vector(1), vector(2, "fake_face")], [vector(3, "fake_reid")]
        )
    error = fails(memory.remember, "Bob", {}, [vector(4, "fake_face")], [vector(5, "fake_reid")])

    # TrackMemory tells the model what was left out.
    assert stored == Stored("stored_41", 1, 0, ("fake_face", "fake_reid"))
    [warning] = warnings(caplog)
    assert "fake_face, fake_reid" in warning
    [request] = database.requests["remember_person"]
    assert [(v.model, int(v.embedding[0])) for v in request.faces] == [(FACE, 1)]
    assert list(request.appearances) == []
    assert error.code is ErrorCode.REFUSED
    assert error.message == (
        "The database takes none of these vector models: fake_face, fake_reid. It takes "
        f"faces of {FACE} and appearances of {APPEARANCE}."
    )


def test_remember_with_no_vectors_raises_value_error(database, memory):
    with pytest.raises(ValueError, match="without a face or appearance vector"):
        memory.remember("Sam", {}, [], [])

    assert database.requests["remember_person"] == []


def test_seventy_queries_go_in_chunks_of_32_32_6_in_order(database, memory):
    results = memory.match_faces([vector(i) for i in range(70)])

    requests = database.requests["match_faces"]
    assert [len(request.queries) for request in requests] == [32, 32, 6]
    assert {request.max_results for request in requests} == {100}
    assert [matches[0].person_id for matches in results] == [f"stored_{i}" for i in range(70)]


def test_no_queries_make_no_call_and_no_start(database, memory):
    assert memory.match_faces([]) == []
    assert memory.match_appearances([]) == []

    assert all(not requests for requests in database.requests.values())


# -- What goes over the wire, and what the log says -------------------------------------


def test_the_request_carries_float32_embeddings_facts_and_max_results(database, memory):
    face = np.array([0.1, 0.2, 0.3, 0.4])  # float64, as a caller may pass
    appearance = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)

    memory.remember(
        "Sam",
        {"drink": "milk", "seat": "sofa"},
        [ModelVector(FACE, face)],
        [ModelVector(APPEARANCE, appearance)],
    )
    memory.match_appearances([ModelVector(APPEARANCE, appearance)])

    [start] = database.requests["start_session"]
    assert (start.label, start.resume) == ("walkie-agent", False)
    [request] = database.requests["remember_person"]
    assert request.name == "Sam"
    assert [(f.key, f.value) for f in request.facts] == [("drink", "milk"), ("seat", "sofa")]
    assert [v.model for v in request.faces] == [FACE]
    assert np.array_equal(np.asarray(request.faces[0].embedding), face.astype(np.float32))
    assert np.array_equal(np.asarray(request.appearances[0].embedding), appearance.ravel())
    [match] = database.requests["match_appearances"]
    assert match.max_results == 100
    assert np.array_equal(np.asarray(match.queries[0].embedding), appearance.ravel())


def test_matches_map_ids_names_facts_and_distances(database, memory):
    # A person kept across sessions (session 0) is matched, and an empty name is
    # passed on for TrackMemory to read as unknown.
    database.results = lambda queries: [
        [
            PersonMatch(
                person_id=7,
                session_id=0,
                name="",
                facts=[Fact(key="drink", value="tea")],
                distance=0.25,
            ),
            PersonMatch(person_id=9, session_id=3, name="Ana", distance=0.5),
        ]
    ]

    [matches] = memory.match_appearances([vector(model=APPEARANCE)])

    assert matches == [
        Match("stored_7", "", {"drink": "tea"}, 0.25),
        Match("stored_9", "Ana", {}, 0.5),
    ]
    assert all(type(m.distance) is float for m in matches)


def test_remember_returns_the_stored_id_and_what_it_stored(database, memory):
    assert memory.remember("Sam", {}, [vector(), vector(1)], []) == Stored("stored_41", 2, 0)
    assert memory.remember("Ana", {}, [], [vector(model=APPEARANCE)]) == Stored(
        "stored_42", 0, 1
    )


def test_no_stored_id_reaches_a_log_at_info_or_above(database, memory, caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        memory.remember("Sam", {}, [vector(1), vector(2, "fake_face")], [])
        memory.match_faces([vector(41), vector(3, "fake_face")])

    loud = [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno >= logging.INFO]
    # The start and both dropped models are logged.
    assert len(loud) == 3
    assert not any("stored_" in message or "person 41" in message for message in loud)


def test_a_forbidden_service_name_is_refused(connector, cancel):
    error = fails(
        RosPeopleMemory,
        connector,
        cancel,
        namespace="/walkie",
        forbidden=("/walkie/remember_person",),
    )

    assert (error.code, error.message) == (
        ErrorCode.REFUSED,
        "Refused: a capability may not use /walkie/remember_person.",
    )
