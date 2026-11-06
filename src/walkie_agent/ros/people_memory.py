"""RosPeopleMemory: the people store on walkie-database's people services.

It is the store of TrackMemory (perception/memory.py) on the robot, chosen with
`--people-memory ros`: a PeopleMemory over the node's /walkie_database/start_session,
remember_person, match_faces and match_appearances (walkie-database database.md, "The
node's interface"). main creates it once from the connector. Its clients are made
here, once, on the connector's node, and its methods block the calling thread, never
the executor's. One call at a time: every public method holds one lock.

A stored person's id is `stored_<id>`, from the database's person.id. It stays in the
store and its logs; the model reads only the live ids that find_people gives.

Sessions (walkie-database ADR-0010, ADR-0017). The database keeps one open session,
and the guests remembered in it. start_session with resume false forgets the open
session and opens the next; with resume true it keeps the open one. The client starts
a session before its first call, and again at each new task, through clear(). main
passes TrackMemory's clear(), which calls this one, to the Runner as on_reset, so a
reset on /walkie/reset ends a task:

- By default it forgets: the first call or clear() after start-up sends resume false,
  and so does every later clear().
- With `resume_session`, the first task after a restart keeps the open session: the
  first clear() or the first call, whichever comes first, sends resume true. The first
  task needs no reset, so a call can come before any clear(). Every later clear()
  forgets.
- A forget that fails stays pending: the next call sends resume false before it
  matches or stores, and fails if that start fails, so the guests of an earlier task
  are never matched in a later one. A failed forget also uses up the resume, so a
  failed resume followed by a failed forget still starts fresh.
- clear() never waits for discovery, waits at most `clear_timeout_s` in all, and never
  raises StepError. Without a discovered start_session it leaves the forget pending at
  once. A start that fails or times out leaves it pending too, and so does another
  call that holds the lock for the whole wait. Each kind of failure is logged as a
  warning once until a start succeeds. A new task therefore costs no wait without a
  database, and at most `clear_timeout_s` with a node that is found but does not
  answer.
- clear() ignores the cancel token, because the Runner calls it during a reset, while
  the token is still set. Every other wait ends at a stop.

No end_session at shutdown: a restart with `resume_session` would find the guests
gone, and by default the next start forgets them anyway.

Each wait for discovery lasts at most `server_timeout_s`, and each wait for an answer
at most `call_timeout_s`. A call's `timeout_s` bounds the whole call: every wait is
cut to what is left of it.

Errors of every call, as StepError codes:

- UNAVAILABLE: no node offers the service within `server_timeout_s`, or the node
  answered that the database failed (its message starts with
  DATABASE_FAILURE_PREFIX), so the same call may work later. Such an answer means
  that nothing was stored, so the request is sent again every RETRY_STEP_S, for at
  most RETRY_BUDGET_S after the first such answer and never past the call's
  `timeout_s`. Only then is it UNAVAILABLE, with the database's message. Under sim
  time, walkie-database answers so until the first /clock (P25).
- TIMEOUT: no answer in time. The request is dropped, never sent again, and
  STALE_PIN_HINT is logged. A remember may have stored the person already, and its
  message says so. The call timeout is above the node's SQLite busy_timeout, so a
  write that waits for a locked database is UNAVAILABLE, not TIMEOUT.
- REFUSED: any other refusal of the node, with its message, such as a vector of the
  wrong length; and a remember none of whose vectors is of a model the database takes.
- CANCELLED: a stop during a wait. A remember whose request went out may have stored
  the person already, and its message says so.

The models the database takes come from each start's answer. A vector of any other
model is not sent: a query gets no matches, and a remember drops it, since one unknown
model would make the node refuse the whole call. The Stored that remember returns
names the dropped models. A match sends at most MAX_QUERIES queries at a time and asks
for MAX_RESULTS people per query. Every vector goes out as a float32 array.array.
People kept across sessions are matched as the node returns them, and an empty name is
passed on as it is; TrackMemory reads it as unknown.

Ported from the experiment/hri-agent branch at 61d30e8 (its perception
ros_people_memory module), with this tree's errors, waits and endpoint check, and the
retry of a database failure.
"""

import array
import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
from rclpy.client import Client
from walkie_database_msgs.msg import Fact, PersonMatch
from walkie_database_msgs.msg import ModelVector as ModelVectorMsg
from walkie_database_msgs.srv import MatchPeople, RememberPerson, StartSession

from walkie_agent.capabilities.guards import CancelToken, require_finite
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.perception.memory import MAY_HAVE_STORED
from walkie_agent.perception.people_memory import Match, ModelVector, Stored
from walkie_agent.ros.policy import FORBIDDEN_NAMES, check_endpoint, ros_failure

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

logger = logging.getLogger(__name__)

# The walkie-database node's namespace, absolute so the names do not depend on the
# agent node's namespace.
NAMESPACE = "/walkie_database"
# Stored with every session the client starts, so the database shows which sessions
# the agent opened. Ignored on a resume.
SESSION_LABEL = "walkie-agent"
# walkie-database's MAX_QUERIES (people.py): the node refuses a match of more.
MAX_QUERIES = 32
# The node's cap on max_results; 0 would mean its default, 10.
MAX_RESULTS = 100
SERVER_TIMEOUT_S = 2.0
# Above walkie-database's SQLite busy_timeout ("PRAGMA busy_timeout = 5000" in its
# db.py), so a write that waits for a locked database is answered as a database
# failure before the client gives up. Keep the two apart.
CALL_TIMEOUT_S = 7.0
# clear() runs on the Runner's thread at each reset, so it waits less. A forget that
# times out stays pending, and the next call waits CALL_TIMEOUT_S.
CLEAR_TIMEOUT_S = 1.0
# The node's _failed() starts every database failure's message with this; a refusal
# (_refused()) has no prefix.
DATABASE_FAILURE_PREFIX = "The database could not"
# The pause before a request that met a database failure is sent again, and how long
# after the first such answer it may still be sent.
RETRY_STEP_S = 0.25
RETRY_BUDGET_S = 2.0
# Logged when the node is found but does not answer. rmw_zenoh_cpp finds a server by
# name and type name but sends requests to a key that includes the type hash, so a node
# built from another walkie_database_msgs never receives them (ADR-0035). A busy node
# looks the same, so the hint goes to the log for a developer, not to the LLM.
STALE_PIN_HINT = (
    "If the database node is not busy, it was probably built from another "
    "walkie_database_msgs than the walkie-database commit pinned in ros2_ws.repos."
)
# Added to a remember's TIMEOUT and CANCELLED once its request went out. The node
# stores a new person on every call (walkie-database ADR-0017). TrackMemory finds the
# sentence and does not add it a second time.
_CAVEAT = f" {MAY_HAVE_STORED}"

_POLL_S = 0.02
_SERVICES = ("start_session", "remember_person", "match_faces", "match_appearances")


class RosPeopleMemory:
    """A PeopleMemory on walkie-database's people services; see the module docstring.

    `cancel` is the agent's token: a stop ends every wait but clear()'s. `forbidden`
    lists the names no endpoint may use, as in the other ROS capabilities. A service
    name among them raises StepError REFUSED here.
    """

    def __init__(
        self,
        connector: "ROS2Connector",
        cancel: CancelToken,
        *,
        resume_session: bool = False,
        namespace: str = NAMESPACE,
        server_timeout_s: float = SERVER_TIMEOUT_S,
        call_timeout_s: float = CALL_TIMEOUT_S,
        clear_timeout_s: float = CLEAR_TIMEOUT_S,
        forbidden: Iterable[str] = FORBIDDEN_NAMES,
    ) -> None:
        node = connector.node
        self._cancel = cancel
        self._server_timeout_s = server_timeout_s
        self._call_timeout_s = call_timeout_s
        self._clear_timeout_s = clear_timeout_s
        forbidden = tuple(forbidden)
        names = {
            service: check_endpoint(node, f"{namespace}/{service}", forbidden)
            for service in _SERVICES
        }
        self._start_client = node.create_client(StartSession, names["start_session"])
        self._remember_client = node.create_client(RememberPerson, names["remember_person"])
        self._faces_client = node.create_client(MatchPeople, names["match_faces"])
        self._appearances_client = node.create_client(MatchPeople, names["match_appearances"])
        self._lock = threading.Lock()
        # Guarded by _lock. A start is needed before the next call: at start-up, and
        # after a new task whose forget has not happened yet.
        self._need_start = True
        # The next start resumes the open session instead of forgetting it.
        self._resume_pending = resume_session
        # The next clear() is the first task's, which keeps the open session.
        self._adopt_next = resume_session
        # Set by a clear() that could not take the lock, and taken under it by the next
        # call or clear(), which then forgets first.
        self._forget_asked = threading.Event()
        # The models the database takes, with their vector lengths, from the last
        # start's answer.
        self._face_models: dict[str, int] = {}
        self._appearance_models: dict[str, int] = {}
        # The kinds of failure clear() has warned of since the last start that
        # succeeded: once per kind, not at every task. Guarded by _warn_lock, since a
        # clear() that cannot take _lock warns too.
        self._warn_lock = threading.Lock()
        self._warned: set[str] = set()

    def remember(
        self,
        name: str,
        facts: Mapping[str, str],
        faces: Sequence[ModelVector],
        appearances: Sequence[ModelVector],
        *,
        timeout_s: float | None = None,
    ) -> Stored:
        if not faces and not appearances:
            # As InMemoryPeopleMemory: TrackMemory checks for vectors before it calls,
            # so this is a bug in the caller.
            raise ValueError(f"Cannot remember {name!r} without a face or appearance vector.")
        what = "remembering a person"
        deadline = _deadline(timeout_s)
        with self._held(deadline, what):
            self._ready(deadline, what)
            kept_faces = _known(faces, self._face_models)
            kept_appearances = _known(appearances, self._appearance_models)
            dropped = sorted(
                {model for model, _ in faces if model not in self._face_models}
                | {model for model, _ in appearances if model not in self._appearance_models}
            )
            if not kept_faces and not kept_appearances:
                raise StepError(
                    ErrorCode.REFUSED,
                    f"The database takes none of these vector models: {', '.join(dropped)}. "
                    f"It takes faces of {_models(self._face_models)} and appearances of "
                    f"{_models(self._appearance_models)}.",
                )
            if dropped:
                logger.warning(
                    f"Remembering {name!r} without the vectors of {', '.join(dropped)}: "
                    "walkie-database has no vector table for them."
                )
            request = RememberPerson.Request(
                name=name,
                facts=[Fact(key=k, value=v, check_fields=True) for k, v in facts.items()],
                faces=[_message(v) for v in kept_faces],
                appearances=[_message(v) for v in kept_appearances],
                check_fields=True,
            )
            response = self._call(self._remember_client, request, deadline, what, remember=True)
            return Stored(
                f"stored_{response.person_id}",
                len(kept_faces),
                len(kept_appearances),
                tuple(dropped),
            )

    def match_faces(
        self, queries: Sequence[ModelVector], *, timeout_s: float | None = None
    ) -> list[list[Match]]:
        return self._matches(queries, timeout_s, faces=True)

    def match_appearances(
        self, queries: Sequence[ModelVector], *, timeout_s: float | None = None
    ) -> list[list[Match]]:
        return self._matches(queries, timeout_s, faces=False)

    def clear(self, *, timeout_s: float | None = None) -> None:
        """Forget the guests of the last task: the Runner's on_reset.

        Waits at most `clear_timeout_s`, or `timeout_s` when that is shorter, and
        returns at once without a discovered database. It ignores the cancel token and
        raises no StepError: a failed forget stays pending for the next call.
        """
        wait_s = self._clear_timeout_s
        if timeout_s is not None:
            wait_s = min(wait_s, timeout_s)
        deadline = time.monotonic() + wait_s
        if not self._lock.acquire(timeout=max(wait_s, 0.0)):
            self._forget_asked.set()
            self._warn_once(
                "busy",
                "People memory is busy with another call, so the new task's forget stays "
                "pending until the next people call.",
            )
            return
        try:
            self._clear(deadline)
        finally:
            self._lock.release()

    def _matches(
        self, queries: Sequence[ModelVector], timeout_s: float | None, *, faces: bool
    ) -> list[list[Match]]:
        deadline = _deadline(timeout_s)
        if not queries:
            return []
        kind = "face" if faces else "appearance"
        what = f"matching {kind}s"
        with self._held(deadline, what):
            self._ready(deadline, what)
            if faces:
                client, known = self._faces_client, self._face_models
            else:
                client, known = self._appearances_client, self._appearance_models
            return self._match(client, kind, known, queries, deadline, what)

    # -- With the lock held ---------------------------------------------------------

    def _clear(self, deadline: float) -> None:
        """Resume at the first task's clear() in resume mode, and forget at any other."""
        self._take_asked_forget()
        adopt = self._adopt_next
        self._adopt_next = False
        if not adopt:
            # Set first, so a failure leaves the forget pending, and a pending resume
            # can never come back in a later task.
            self._resume_pending = False
            self._need_start = True
        try:
            if not self._is_ready(self._start_client):
                if not adopt:
                    self._warn_once(
                        "no node",
                        f"People memory is unavailable: no node offers "
                        f"{self._start_client.srv_name}. The new task's forget stays "
                        "pending until the next people call.",
                    )
                # A resume stays pending for the first call, which waits for discovery.
                return
            self._start(
                self._resume_pending, deadline, "starting a people session", cancellable=False
            )
        except StepError as error:
            if adopt:
                self._warn_once(
                    "resume",
                    f"People memory could not resume the open session ({error}). "
                    "The next people call tries again.",
                )
            else:
                self._warn_once(
                    "forget",
                    f"People memory could not forget the last task's guests ({error}). "
                    "The forget stays pending until the next people call.",
                )

    def _ready(self, deadline: float | None, what: str) -> None:
        """Start the session the next call needs."""
        self._take_asked_forget()
        # This call belongs to the first task, so a later clear() forgets.
        self._adopt_next = False
        if self._need_start:
            self._start(self._resume_pending, deadline, what)

    def _take_asked_forget(self) -> None:
        """Turn a forget that clear() could not make into a pending one."""
        if self._forget_asked.is_set():
            self._forget_asked.clear()
            self._adopt_next = False
            self._resume_pending = False
            self._need_start = True

    def _start(
        self, resume: bool, deadline: float | None, what: str, *, cancellable: bool = True
    ) -> None:
        """Start a session, or resume the open one; read the models it takes."""
        request = StartSession.Request(label=SESSION_LABEL, resume=resume, check_fields=True)
        response = self._call(self._start_client, request, deadline, what, cancellable=cancellable)
        self._need_start = False
        self._resume_pending = False
        self._face_models = {m.model: m.length for m in response.face_models}
        self._appearance_models = {m.model: m.length for m in response.appearance_models}
        with self._warn_lock:
            self._warned.clear()
        # The node's message names the session, and its people on a resume.
        logger.info(
            f"People memory: {response.message} It takes faces of "
            f"{_models(self._face_models)} and appearances of "
            f"{_models(self._appearance_models)}."
        )
        for kind, models in (("face", self._face_models), ("appearance", self._appearance_models)):
            if not models:
                logger.warning(
                    f"walkie-database has no {kind} vector table, so people memory "
                    f"neither stores nor matches {kind}s."
                )

    def _match(
        self,
        client: Client,
        kind: str,
        known: Mapping[str, int],
        queries: Sequence[ModelVector],
        deadline: float | None,
        what: str,
    ) -> list[list[Match]]:
        """Match the queries of known models in chunks; the others get no matches."""
        results: list[list[Match]] = [[] for _ in queries]
        sent = [i for i, (model, _) in enumerate(queries) if model in known]
        unknown = sorted({model for model, _ in queries if model not in known})
        if unknown:
            logger.warning(
                f"No {kind} matches for the queries of {', '.join(unknown)}: walkie-database "
                "has no vector table for them."
            )
        for start in range(0, len(sent), MAX_QUERIES):
            chunk = sent[start : start + MAX_QUERIES]
            request = MatchPeople.Request(
                queries=[_message(queries[i]) for i in chunk],
                max_results=MAX_RESULTS,
                check_fields=True,
            )
            response = self._call(client, request, deadline, what)
            if len(response.results) != len(chunk):
                raise RuntimeError(
                    f"{client.srv_name} answered {len(response.results)} of "
                    f"{len(chunk)} queries."
                )
            for i, matches in zip(chunk, response.results, strict=True):
                results[i] = [_match_of(person) for person in matches.people]
        return results

    def _call(
        self,
        client: Client,
        request: Any,
        deadline: float | None,
        what: str,
        *,
        cancellable: bool = True,
        remember: bool = False,
    ) -> Any:
        """Call one service and return its successful response; see the module docstring.

        A database failure is asked again; any other refusal is REFUSED at once.
        """
        retry_until: float | None = None
        while True:
            response = self._answer(
                client, request, deadline, what, cancellable=cancellable, remember=remember
            )
            if response.success:
                return response
            message = response.message
            if not message.startswith(DATABASE_FAILURE_PREFIX):
                raise StepError(ErrorCode.REFUSED, message)
            now = time.monotonic()
            if retry_until is None:
                retry_until = now + RETRY_BUDGET_S
                if deadline is not None:
                    retry_until = min(retry_until, deadline)
            if now + RETRY_STEP_S >= retry_until:
                raise StepError(ErrorCode.UNAVAILABLE, message)
            logger.debug(f"{client.srv_name} answered {message!r}; asking again.")
            self._pause(RETRY_STEP_S, what, cancellable)

    def _answer(
        self,
        client: Client,
        request: Any,
        deadline: float | None,
        what: str,
        *,
        cancellable: bool,
        remember: bool,
    ) -> Any:
        """Send one request and return the node's answer, whether it succeeded or not."""
        name = client.srv_name
        discovery_s = _budget(deadline, self._server_timeout_s, what)
        try:
            self._wait(lambda: self._is_ready(client), discovery_s, what, cancellable)
        except StepError as error:
            if error.code is not ErrorCode.TIMEOUT:
                raise
            raise StepError(
                ErrorCode.UNAVAILABLE,
                f"People memory is unavailable: no node offers {name}. The walkie-database "
                "node is not running, or not on this ROS domain.",
            ) from error
        wait_s = _budget(deadline, self._call_timeout_s, what)
        if cancellable:
            # In the fence, so a stop either comes first and nothing goes out, or waits
            # until the request has gone.
            with self._cancel.unless_cancelled(what):
                future = _send(client, request, what)
        else:
            future = _send(client, request, what)
        try:
            self._wait(future.done, wait_s, what, cancellable)
        except StepError as error:
            # The answer would otherwise be kept for a caller that left.
            client.remove_pending_request(future)
            caveat = _CAVEAT if remember else ""
            if error.code is ErrorCode.TIMEOUT:
                logger.warning(f"{name} did not answer within {wait_s:.3g} s. {STALE_PIN_HINT}")
                raise StepError(
                    ErrorCode.TIMEOUT,
                    f"People memory did not answer within {wait_s:.3g} s. It may be busy; "
                    f"try again once.{caveat}",
                ) from error
            if caveat:
                raise StepError(error.code, f"{error.message}{caveat}") from error
            raise
        return future.result()

    # -- Waits ------------------------------------------------------------------------

    @contextmanager
    def _held(self, deadline: float | None, what: str) -> Iterator[None]:
        """Hold the lock for one call. A stop or the call's deadline ends the wait for it."""
        while True:
            self._cancel.check(what)
            wait_s = _POLL_S
            if deadline is not None:
                wait_s = min(wait_s, deadline - time.monotonic())
                if wait_s <= 0:
                    raise StepError(ErrorCode.TIMEOUT, f"Timed out: {what} ran out of time.")
            if self._lock.acquire(timeout=wait_s):
                break
        try:
            yield
        finally:
            self._lock.release()

    def _wait(self, done: Callable[[], bool], wait_s: float, what: str, cancellable: bool) -> None:
        """Wait until done() is true, for at most `wait_s`; StepError TIMEOUT after it.

        A cancellable wait also ends at a stop, with StepError CANCELLED.
        """
        if cancellable:
            self._cancel.wait_for(done, wait_s, what, poll_s=_POLL_S)
            return
        end = time.monotonic() + wait_s
        while not done():
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise StepError(
                    ErrorCode.TIMEOUT, f"Timed out: {what} did not finish within {wait_s:.3g} s."
                )
            time.sleep(min(_POLL_S, remaining))

    def _pause(self, seconds: float, what: str, cancellable: bool) -> None:
        """Sleep before a request is sent again. A cancellable pause ends at a stop."""
        end = time.monotonic() + seconds
        while (remaining := end - time.monotonic()) > 0:
            if cancellable:
                self._cancel.check(what)
            time.sleep(min(_POLL_S, remaining))

    def _is_ready(self, client: Client) -> bool:
        try:
            return client.service_is_ready()
        except Exception as error:
            raise ros_failure(error, "Finding the people memory") from error

    def _warn_once(self, kind: str, message: str) -> None:
        """Log a warning of clear(), once per kind of failure until a start succeeds."""
        with self._warn_lock:
            if kind in self._warned:
                return
            self._warned.add(kind)
        logger.warning(message)


def _deadline(timeout_s: float | None) -> float | None:
    if timeout_s is None:
        return None
    require_finite(timeout_s=timeout_s)
    return time.monotonic() + timeout_s


def _budget(deadline: float | None, limit_s: float, what: str) -> float:
    """A wait's budget. CancelToken.wait_for reads zero or less as REFUSED, so it never gets one."""
    if deadline is None:
        return limit_s
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise StepError(ErrorCode.TIMEOUT, f"Timed out: {what} ran out of time.")
    return min(limit_s, remaining)


def _send(client: Client, request: Any, what: str) -> Any:
    try:
        return client.call_async(request)
    except Exception as error:
        raise ros_failure(error, f"Sending a request for {what}") from error


def _known(vectors: Sequence[ModelVector], models: Mapping[str, int]) -> list[ModelVector]:
    return [vector for vector in vectors if vector.model in models]


def _models(models: Mapping[str, int]) -> str:
    return ", ".join(sorted(models)) or "no model"


def _f32(vector: Any) -> array.array:
    """The vector as a float32 array.array, the fastest value for a float32[] field."""
    return array.array("f", np.asarray(vector, dtype=np.float32).ravel().tobytes())


def _message(vector: ModelVector) -> ModelVectorMsg:
    model, values = vector
    return ModelVectorMsg(model=model, embedding=_f32(values), check_fields=True)


def _match_of(person: PersonMatch) -> Match:
    return Match(
        f"stored_{person.person_id}",
        person.name,
        {fact.key: fact.value for fact in person.facts},
        float(person.distance),
    )
