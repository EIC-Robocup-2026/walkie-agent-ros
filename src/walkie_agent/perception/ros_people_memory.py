"""RosPeopleMemory: the people memory on walkie-database's people services.

It is the `people_memory` backend of a robot run, a PeopleMemory over the
node's /walkie_database/start_session, remember_person, match_faces and
match_appearances (walkie-database database.md, "The node's interface").
It is created once per agent from the connector, like VisionFeed: its
clients are made here, once, on the connector's node, and its methods
block the calling thread, never the executor's. One call at a time: every
public method holds one lock.

Sessions (walkie-database ADR-0010, ADR-0017). The database keeps one open
session, and the guests remembered in it. start_session with resume false
forgets the open session and opens the next; with resume true it keeps the
open one. The client starts a session before its first call, and again at
each new task, through clear(), which the agent registers with
Toolbox.on_new_task:

- By default it forgets: the first call or clear() after start-up sends
  resume false, and so does every later clear().
- With `resume_session`, the first task after a restart keeps the open
  session: the first clear() or the first call, whichever comes first,
  sends resume true. A first message with an empty todo list starts no new
  task, so a call can come before any clear(). Every later clear() forgets.
- A forget that fails stays pending: the next call sends resume false
  before it matches or stores, and fails if that start fails, so the
  guests of an earlier task are never matched in a later one. A failed
  forget also uses up the resume, so a failed resume followed by a failed
  forget still starts fresh.
- clear() never waits for discovery, waits at most `clear_timeout_s` for
  an answer, and never raises StepError. Without a discovered
  start_session it leaves the forget pending at once. A start that fails
  or times out leaves it pending too. Each kind of failure is logged as a
  warning once until a start succeeds. A new task therefore costs no wait
  without a database, and at most `clear_timeout_s` with a node that is
  found but does not answer.

No end_session at shutdown: a restart with `resume_session` would find the
guests gone, and by default the next start forgets them anyway.

Errors of every call, as StepError codes:

- UNAVAILABLE: no node offers the service within `server_timeout_s`, or the
  node answered that the database failed (its message starts with
  DATABASE_FAILURE_PREFIX), so the same call may work later.
- TIMEOUT: no answer within `call_timeout_s`. The request is dropped and
  STALE_PIN_HINT logged. A remember may have stored the person already.
  The timeout is above the node's SQLite busy_timeout, so a write that
  waits for a locked database is UNAVAILABLE, not TIMEOUT.
- REFUSED: any other refusal of the node, with its message, such as a
  vector of the wrong length; and a remember none of whose vectors is of a
  model the database takes.

The models the database takes come from each start's answer. A vector of
any other model is not sent: a query gets no matches, and a remember drops
it, since one unknown model would make the node refuse the whole call. The
Stored that remember returns names the dropped models.
People kept across sessions are matched as the node returns them, and an
empty name is passed on as it is; recognize_person reads it as UNKNOWN.
"""

import array
import logging
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from rai.communication.ros2 import ROS2Connector
from rclpy.client import Client
from walkie_database_msgs.msg import Fact, PersonMatch
from walkie_database_msgs.msg import ModelVector as ModelVectorMsg
from walkie_database_msgs.srv import MatchPeople, RememberPerson, StartSession

from walkie_agent.perception.people_memory import Match, ModelVector, Stored
from walkie_agent.toolbox.types import StepError
from walkie_agent.tools.waiters import wait_for_future

logger = logging.getLogger(__name__)

# The walkie-database node's namespace, absolute so the names do not depend
# on the agent node's namespace.
NAMESPACE = "/walkie_database"
# Stored with every session the client starts, so the database shows which
# sessions the agent opened. Ignored on a resume.
SESSION_LABEL = "walkie-agent"
# walkie-database's MAX_QUERIES (people.py): the node refuses a match of more.
MAX_QUERIES = 32
# The node's cap on max_results; 0 would mean its default, 10.
MAX_RESULTS = 100
SERVER_TIMEOUT_S = 2.0
# Above walkie-database's SQLite busy_timeout ("PRAGMA busy_timeout = 5000"
# in its db.py), so a write that waits for a locked database is answered as a
# database failure before the client gives up. Keep the two apart.
CALL_TIMEOUT_S = 7.0
# clear() runs on the agent's thread before each task, so it waits less. A
# forget that times out stays pending, and the next call waits CALL_TIMEOUT_S.
CLEAR_TIMEOUT_S = 1.0
# The node's _failed() starts every database failure's message with this; a
# refusal (_refused()) has no prefix.
DATABASE_FAILURE_PREFIX = "The database could not"
# Logged when the node is found but does not answer. rmw_zenoh_cpp finds a
# server by name and type name but sends requests to a key that includes the
# type hash, so a node built from another walkie_database_msgs never
# receives them (ADR-0035). A busy node looks the same, so the hint goes to
# the log for a developer, not to the LLM, as find_objects does.
STALE_PIN_HINT = (
    "If the database node is not busy, it was probably built from another "
    "walkie_database_msgs than the walkie-database commit pinned in ros2_ws.repos."
)


class RosPeopleMemory:
    """A PeopleMemory on walkie-database's people services; see the module docstring."""

    def __init__(
        self,
        connector: ROS2Connector,
        *,
        resume_session: bool = False,
        namespace: str = NAMESPACE,
        server_timeout_s: float = SERVER_TIMEOUT_S,
        call_timeout_s: float = CALL_TIMEOUT_S,
        clear_timeout_s: float = CLEAR_TIMEOUT_S,
    ) -> None:
        node = connector.node
        self._server_timeout_s = server_timeout_s
        self._call_timeout_s = call_timeout_s
        self._clear_timeout_s = clear_timeout_s
        self._start_client = node.create_client(StartSession, f"{namespace}/start_session")
        self._remember_client = node.create_client(
            RememberPerson, f"{namespace}/remember_person"
        )
        self._faces_client = node.create_client(MatchPeople, f"{namespace}/match_faces")
        self._appearances_client = node.create_client(
            MatchPeople, f"{namespace}/match_appearances"
        )
        self._lock = threading.Lock()
        # Guarded by _lock. A start is needed before the next call: at
        # start-up, and after a new task whose forget has not happened yet.
        self._need_start = True
        # The next start resumes the open session instead of forgetting it.
        self._resume_pending = resume_session
        # The next clear() is the first task's, which keeps the open session.
        self._adopt_next = resume_session
        # The models the database takes, with their vector lengths, from the
        # last start's answer.
        self._face_models: dict[str, int] = {}
        self._appearance_models: dict[str, int] = {}
        # The kinds of failure clear() has warned of since the last start
        # that succeeded: once per kind, not at every task.
        self._warned: set[str] = set()

    def remember(
        self,
        name: str,
        facts: Mapping[str, str],
        faces: Sequence[ModelVector],
        appearances: Sequence[ModelVector],
    ) -> Stored:
        if not faces and not appearances:
            # As InMemoryPeopleMemory: remember_person checks for vectors
            # before it calls, so this is a bug in the caller.
            raise ValueError(f"Cannot remember {name!r} without a face or appearance vector.")
        with self._lock:
            self._ready()
            kept_faces = _known(faces, self._face_models)
            kept_appearances = _known(appearances, self._appearance_models)
            dropped = sorted(
                {model for model, _ in faces if model not in self._face_models}
                | {model for model, _ in appearances if model not in self._appearance_models}
            )
            if not kept_faces and not kept_appearances:
                raise StepError(
                    "REFUSED",
                    f"The database takes none of these vector models: {', '.join(dropped)}. "
                    f"It takes faces of {_models(self._face_models)} and appearances of "
                    f"{_models(self._appearance_models)}.",
                    {"models": dropped},
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
            response = self._call(self._remember_client, request, remember=True)
            return Stored(
                f"person_{response.person_id}",
                len(kept_faces),
                len(kept_appearances),
                tuple(dropped),
            )

    def match_faces(self, queries: Sequence[ModelVector]) -> list[list[Match]]:
        if not queries:
            return []
        with self._lock:
            self._ready()
            return self._match(self._faces_client, "face", self._face_models, queries)

    def match_appearances(self, queries: Sequence[ModelVector]) -> list[list[Match]]:
        if not queries:
            return []
        with self._lock:
            self._ready()
            return self._match(
                self._appearances_client, "appearance", self._appearance_models, queries
            )

    def clear(self) -> None:
        """Forget the guests of the last task: the Toolbox.on_new_task listener.

        Returns at once without a discovered database, waits at most
        `clear_timeout_s` for an answer, and raises no StepError: a failed
        forget stays pending for the next call.
        """
        with self._lock:
            if self._adopt_next:
                # The first task after a restart in resume mode keeps the
                # open session.
                self._adopt_next = False
                if self._need_start and self._start_client.service_is_ready():
                    try:
                        self._start(True, self._clear_timeout_s)
                    except StepError as e:
                        self._warn_once(
                            "resume",
                            f"People memory could not resume the open session ({e}). "
                            "The next people call tries again.",
                        )
                return
            # Set first, so a failure leaves the forget pending, and a
            # pending resume can never come back in a later task.
            self._resume_pending = False
            self._need_start = True
            if not self._start_client.service_is_ready():
                self._warn_once(
                    "no node",
                    f"People memory is unavailable: no node offers "
                    f"{self._start_client.srv_name}. The new task's forget stays pending "
                    "until the next people call.",
                )
                return
            try:
                self._start(False, self._clear_timeout_s)
            except StepError as e:
                self._warn_once(
                    "forget",
                    f"People memory could not forget the last task's guests ({e}). "
                    "The forget stays pending until the next people call.",
                )

    def _ready(self) -> None:
        """Start the session the next call needs. Called with the lock held."""
        # This call belongs to the first task, so a later clear() forgets.
        self._adopt_next = False
        if self._need_start:
            self._start(self._resume_pending)

    def _start(self, resume: bool, timeout_s: float | None = None) -> None:
        """Start a session, or resume the open one; read the models it takes."""
        response = self._call(
            self._start_client,
            StartSession.Request(label=SESSION_LABEL, resume=resume, check_fields=True),
            timeout_s=timeout_s,
        )
        self._need_start = False
        self._resume_pending = False
        self._face_models = {m.model: m.length for m in response.face_models}
        self._appearance_models = {m.model: m.length for m in response.appearance_models}
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

    def _call(
        self,
        client: Client,
        request: Any,
        *,
        timeout_s: float | None = None,
        remember: bool = False,
    ) -> Any:
        """Call one service and return its successful response; see the module docstring.

        Waits `timeout_s` for the answer, by default `call_timeout_s`.
        """
        name = client.srv_name
        timeout_s = self._call_timeout_s if timeout_s is None else timeout_s
        if not client.wait_for_service(timeout_sec=self._server_timeout_s):
            raise StepError(
                "UNAVAILABLE",
                f"People memory is unavailable: no node offers {name}. The walkie-database "
                "node is not running, or not on this ROS domain.",
                {"service": name},
            )
        future = client.call_async(request)
        response = wait_for_future(future, timeout_s)
        if response is None:
            # The answer would otherwise be kept for a caller that left.
            client.remove_pending_request(future)
            logger.warning(f"{name} did not answer within {timeout_s:g} s. {STALE_PIN_HINT}")
            message = (
                f"People memory did not answer within {timeout_s:g} s. "
                "It may be busy; try again once."
            )
            if remember:
                # The node stores a new person on every call (walkie-database
                # ADR-0017).
                message += (
                    " The person may have been stored already, so remembering them "
                    "again may store them twice."
                )
            raise StepError("TIMEOUT", message, {"service": name})
        if not response.success:
            failed = response.message.startswith(DATABASE_FAILURE_PREFIX)
            raise StepError(
                "UNAVAILABLE" if failed else "REFUSED", response.message, {"service": name}
            )
        return response

    def _match(
        self,
        client: Client,
        kind: str,
        known: Mapping[str, int],
        queries: Sequence[ModelVector],
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
            response = self._call(client, request)
            if len(response.results) != len(chunk):
                raise RuntimeError(
                    f"{client.srv_name} answered {len(response.results)} of "
                    f"{len(chunk)} queries."
                )
            for i, matches in zip(chunk, response.results, strict=True):
                results[i] = [_match_of(person) for person in matches.people]
        return results

    def _warn_once(self, kind: str, message: str) -> None:
        """Log a warning of clear(), once per kind of failure until a start succeeds."""
        if kind not in self._warned:
            self._warned.add(kind)
            logger.warning(message)


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
        f"person_{person.person_id}",
        person.name,
        {fact.key: fact.value for fact in person.facts},
        float(person.distance),
    )
