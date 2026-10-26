"""RosPeopleMemory against a fake walkie-database node over real ROS 2 (no LLM).

Each test's fake serves the four people services under a node name of its
own, so a fake torn down in one test never still looks discovered in the
next.
"""

import contextlib
import logging
import random
import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest
from rclpy.client import Client
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from walkie_database_msgs.msg import Fact, ModelInfo, PersonMatch, PersonMatches
from walkie_database_msgs.msg import ModelVector as ModelVectorMsg
from walkie_database_msgs.srv import MatchPeople, RememberPerson, StartSession

from walkie_agent.perception.people_memory import Match, ModelVector, Stored
from walkie_agent.perception.ros_people_memory import RosPeopleMemory
from walkie_agent.toolbox import StepError

LOGGER = "walkie_agent.perception.ros_people_memory"
FACE = "arcface_w600k_r50"
APPEARANCE = "osnet_x1_0_msmt17"
SERVICES = {
    "start_session": StartSession,
    "remember_person": RememberPerson,
    "match_faces": MatchPeople,
    "match_appearances": MatchPeople,
}
DATABASE_FAILURE = "The database could not start a session: database is locked"


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
    """Serves ~/start_session, ~/remember_person, ~/match_faces and ~/match_appearances."""

    def __init__(self, name: str):
        super().__init__(name)
        self.requests: dict[str, list] = {service: [] for service in SERVICES}
        # By service: how long to wait before answering, and the message of
        # a success=False answer.
        self.delay_sec: dict[str, float] = {}
        self.failure: dict[str, str] = {}
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
            self.create_service(srv_type, f"~/{service}", self._serve(service, answers[service]))

    def _serve(self, service, answer):
        def callback(request, response):
            self.requests[service].append(request)
            time.sleep(self.delay_sec.get(service, 0.0))
            if service in self.failure:
                response.success = False
                response.message = self.failure[service]
                return response
            response.success = True
            answer(request, response)
            return response

        return callback

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

    def _match(self, request, response):
        response.results = [
            PersonMatches(people=people) for people in self.results(request.queries)
        ]


@contextlib.contextmanager
def serving(name: str) -> Iterator[FakeDatabase]:
    """A FakeDatabase named `name`, spinning on its own thread."""
    node = FakeDatabase(name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield node
    finally:
        executor.shutdown()
        node.destroy_node()
        thread.join(timeout=2)


def fake_name() -> str:
    return f"fake_walkie_database_{random.randint(0, 1_000_000)}"


# Short timeouts are flaky under Fast DDS discovery, so the tests wait 2 s
# unless they test a timeout.
TIMEOUTS = {"server_timeout_s": 2.0, "call_timeout_s": 2.0, "clear_timeout_s": 2.0}


@pytest.fixture
def database():
    with serving(fake_name()) as node:
        yield node


@pytest.fixture
def make_memory(connector, database):
    """A RosPeopleMemory factory on the fake's services; keyword arguments go to it."""

    def make(**kwargs) -> RosPeopleMemory:
        options = {**TIMEOUTS, **kwargs}
        return RosPeopleMemory(connector, namespace=f"/{database.get_name()}", **options)

    return make


@pytest.fixture
def memory(make_memory):
    return make_memory()


@pytest.fixture
def no_database(connector):
    """A RosPeopleMemory whose namespace no node serves."""
    namespace = f"/no_walkie_database_{random.randint(0, 1_000_000)}"
    return RosPeopleMemory(connector, namespace=namespace, server_timeout_s=0.5)


def discovered(memory: RosPeopleMemory) -> None:
    """Wait until the fake's start_session is discovered, as clear() does not wait."""
    assert memory._start_client.wait_for_service(timeout_sec=5.0)


def resumes(database: FakeDatabase) -> list[bool]:
    return [request.resume for request in database.requests["start_session"]]


def fails(call, *args) -> StepError:
    with pytest.raises(StepError) as error:
        call(*args)
    return error.value


def warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING
    ]


# -- Sessions ---------------------------------------------------------------------------


def test_default_mode_forgets_at_the_first_call_and_at_each_new_task(database, memory):
    memory.match_faces([vector()])
    memory.match_faces([vector()])
    memory.clear()
    memory.clear()

    assert resumes(database) == [False, False, False]
    assert {request.label for request in database.requests["start_session"]} == {"walkie-agent"}
    assert len(database.requests["match_faces"]) == 2


def test_resume_mode_resumes_once_and_the_next_new_task_forgets(database, make_memory):
    memory = make_memory(resume_session=True)
    discovered(memory)

    memory.clear()
    assert resumes(database) == [True]
    memory.match_faces([vector()])
    assert resumes(database) == [True]
    memory.clear()
    assert resumes(database) == [True, False]


def test_a_first_task_without_a_new_task_uses_up_the_resume(database, make_memory):
    # A first message with an empty todo list starts no new task, so the
    # first call comes before any clear().
    memory = make_memory(resume_session=True)

    memory.match_faces([vector()])
    memory.clear()

    assert resumes(database) == [True, False]


def test_a_failed_forget_stays_pending_until_the_next_call(database, memory):
    memory.match_faces([vector()])
    database.failure["start_session"] = DATABASE_FAILURE

    memory.clear()
    # The forget fails again, so the call fails rather than match the last
    # task's guests.
    error = fails(memory.match_faces, [vector()])
    del database.failure["start_session"]
    memory.match_faces([vector()])

    assert error.code == "UNAVAILABLE"
    assert resumes(database) == [False, False, False, False]
    assert len(database.requests["match_faces"]) == 2


def test_a_failed_resume_then_a_failed_forget_starts_fresh(database, make_memory):
    memory = make_memory(resume_session=True)
    discovered(memory)
    database.failure["start_session"] = DATABASE_FAILURE

    memory.clear()  # the first task's resume fails
    memory.clear()  # the second task's forget fails
    del database.failure["start_session"]
    memory.match_faces([vector()])

    # The second task must not resume the session from before the restart.
    assert resumes(database) == [True, False, False]


def test_clear_without_a_database_returns_at_once_and_warns_once(no_database, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        start = time.monotonic()
        no_database.clear()
        no_database.clear()
        elapsed = time.monotonic() - start

    assert elapsed < 0.5
    [warning] = warnings(caplog)
    assert "forget stays pending" in warning


def test_a_resume_before_the_node_is_found_stays_pending(connector, caplog):
    # The agent and the node restart together, so the first new task comes
    # before start_session is discovered. It must not wait, warn, or lose
    # the resume.
    name = fake_name()
    options = {**TIMEOUTS, "server_timeout_s": 5.0}
    memory = RosPeopleMemory(connector, namespace=f"/{name}", resume_session=True, **options)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        start = time.monotonic()
        memory.clear()
        elapsed = time.monotonic() - start
    with serving(name) as database:
        memory.match_faces([vector()])
        memory.clear()

    assert elapsed < 0.5
    assert warnings(caplog) == []
    assert resumes(database) == [True, False]


def test_clear_waits_at_most_the_clear_timeout_on_a_silent_node(database, make_memory, caplog):
    # A node that is found but does not answer, such as one built from
    # another walkie_database_msgs under Zenoh (ADR-0035), costs a new task
    # the clear timeout, not the call timeout.
    memory = make_memory(call_timeout_s=5.0, clear_timeout_s=0.3)
    discovered(memory)
    database.delay_sec["start_session"] = 1.0

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        start = time.monotonic()
        memory.clear()
        elapsed = time.monotonic() - start
    del database.delay_sec["start_session"]
    # The forget stays pending, so the next call starts a session first.
    memory.match_faces([vector()])

    assert elapsed < 0.9
    assert any("could not forget" in warning for warning in warnings(caplog))
    assert resumes(database) == [False, False]
    assert len(database.requests["match_faces"]) == 1


def test_each_kind_of_failure_warns_once_until_a_start_succeeds(database, make_memory, caplog):
    memory = make_memory(resume_session=True)
    discovered(memory)
    database.failure["start_session"] = DATABASE_FAILURE

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        memory.clear()  # the first task's resume fails
        memory.clear()  # the second task's forget fails: another kind, so a warning
        memory.clear()  # the third task's forget fails: no second warning
        del database.failure["start_session"]
        memory.match_faces([vector()])  # a start succeeds
        database.failure["start_session"] = DATABASE_FAILURE
        memory.clear()  # a failed forget warns again

    assert resumes(database) == [True, False, False, False, False]
    assert [warning.split(" (")[0] for warning in warnings(caplog)] == [
        "People memory could not resume the open session",
        "People memory could not forget the last task's guests",
        "People memory could not forget the last task's guests",
    ]


# -- Errors -----------------------------------------------------------------------------


def test_no_server_is_unavailable(no_database):
    error = fails(no_database.match_faces, [vector()])

    service = error.data["service"]
    assert service.endswith("/start_session")
    assert error.code == "UNAVAILABLE"
    assert error.message == (
        f"People memory is unavailable: no node offers {service}. The walkie-database node "
        "is not running, or not on this ROS domain."
    )


def test_a_slow_server_times_out_and_drops_the_pending_request(
    database, make_memory, monkeypatch, caplog
):
    # As in test_memory.py: a busy node and one built from another
    # walkie_database_msgs look the same, so both are dropped with a hint.
    memory = make_memory(call_timeout_s=0.3)
    memory.match_faces([vector()])
    database.delay_sec["match_faces"] = 1.0
    dropped = []
    remove_pending_request = Client.remove_pending_request

    def record_drop(client, future):
        dropped.append(future)
        remove_pending_request(client, future)

    monkeypatch.setattr(Client, "remove_pending_request", record_drop)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        error = fails(memory.match_faces, [vector()])
    # rclpy also calls remove_pending_request when a future completes, so
    # only the calls made before the late answer come from the client.
    dropped_by_client = list(dropped)
    time.sleep(database.delay_sec["match_faces"])  # let the late answer go out before teardown

    assert error.code == "TIMEOUT"
    assert error.message == (
        "People memory did not answer within 0.3 s. It may be busy; try again once."
    )
    assert len(dropped_by_client) == 1
    assert not dropped_by_client[0].done()
    [warning] = warnings(caplog)
    assert "pinned in ros2_ws.repos" in warning


def test_a_slow_remember_says_it_may_have_stored_the_person(database, make_memory):
    memory = make_memory(call_timeout_s=0.3)
    memory.match_faces([vector()])
    database.delay_sec["remember_person"] = 1.0

    error = fails(memory.remember, "Sam", {}, [vector()], [])
    time.sleep(database.delay_sec["remember_person"])

    assert error.code == "TIMEOUT"
    assert error.message == (
        "People memory did not answer within 0.3 s. It may be busy; try again once. The "
        "person may have been stored already, so remembering them again may store them twice."
    )


def test_a_database_failure_is_unavailable(database, memory):
    failure = "The database could not match appearances: disk I/O error"
    database.failure["match_appearances"] = failure

    error = fails(memory.match_appearances, [vector(model=APPEARANCE)])

    assert (error.code, error.message) == ("UNAVAILABLE", failure)


def test_any_other_refusal_is_refused(database, memory):
    database.failure["remember_person"] = "At most 32 facts, got 33."

    error = fails(memory.remember, "Sam", {"drink": "milk"}, [vector()], [])

    assert (error.code, error.message) == ("REFUSED", "At most 32 facts, got 33.")
    assert error.data == {"service": f"/{database.get_name()}/remember_person"}


def test_a_results_length_mismatch_is_a_runtime_error(database, memory):
    database.results = lambda queries: by_index(queries)[1:]

    with pytest.raises(RuntimeError, match="answered 1 of 2 queries"):
        memory.match_faces([vector(0), vector(1)])


# -- Models, chunks and order -------------------------------------------------------------


def test_unknown_models_get_no_matches_and_are_not_sent(database, memory, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        results = memory.match_faces([vector(0), vector(1, "fake_face"), vector(2)])

    [request] = database.requests["match_faces"]
    assert [(q.model, int(q.embedding[0])) for q in request.queries] == [(FACE, 0), (FACE, 2)]
    assert [[m.person_id for m in matches] for matches in results] == [
        ["person_0"],
        [],
        ["person_2"],
    ]
    [warning] = warnings(caplog)
    assert "fake_face" in warning


def test_remember_drops_unknown_models_and_refuses_when_none_are_left(database, memory, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        stored = memory.remember(
            "Sam", {}, [vector(1), vector(2, "fake_face")], [vector(3, "fake_reid")]
        )
    error = fails(memory.remember, "Bob", {}, [vector(4, "fake_face")], [vector(5, "fake_reid")])

    # remember_person tells the LLM what was left out.
    assert stored == Stored("person_41", 1, 0, ("fake_face", "fake_reid"))
    [warning] = warnings(caplog)
    assert "fake_face, fake_reid" in warning
    [request] = database.requests["remember_person"]
    assert [(v.model, int(v.embedding[0])) for v in request.faces] == [(FACE, 1)]
    assert list(request.appearances) == []
    assert error.code == "REFUSED"
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
    assert [matches[0].person_id for matches in results] == [f"person_{i}" for i in range(70)]


def test_no_queries_make_no_call_and_no_start(database, memory):
    assert memory.match_faces([]) == []
    assert memory.match_appearances([]) == []

    assert all(not requests for requests in database.requests.values())


# -- What goes over the wire ------------------------------------------------------------


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
    # A person kept across sessions (session 0) is matched, and an empty
    # name is passed on for recognize_person to read as UNKNOWN.
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
        Match("person_7", "", {"drink": "tea"}, 0.25),
        Match("person_9", "Ana", {}, 0.5),
    ]
    assert all(type(m.distance) is float for m in matches)


def test_remember_returns_a_person_handle_id_and_what_it_stored(database, memory):
    assert memory.remember("Sam", {}, [vector(), vector(1)], []) == Stored("person_41", 2, 0)
    assert memory.remember("Ana", {}, [], [vector(model=APPEARANCE)]) == Stored(
        "person_42", 0, 1
    )


def test_check_fields_is_on():
    # conftest.py sets ROS_PYTHON_CHECK_FIELDS. Without it, this builds a
    # vector with an empty embedding, and the node refuses it only for its
    # length.
    with pytest.raises(AssertionError):
        ModelVectorMsg(model="x", values=[1.0])
