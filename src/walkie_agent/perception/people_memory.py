"""The people store: who the robot has met, with their facts, faces and appearances.

PeopleMemory is the store that TrackMemory (memory.py) calls for ctx.memory
(ADR-0070). It follows walkie-database's people services (its database.md, "The
node's interface"). On the robot, RosPeopleMemory (ros/people_memory.py) calls them.
InMemoryPeopleMemory stands in for them in simulation and in the tests, chosen with
main.py's `--people-memory mock`: the simulation's fake vision models have no vector
tables in walkie-database. No ROS.

A stored person's id, such as `stored_3`, stays in the store and its logs. The model
only ever reads the live ids that find_people gives.
"""

import itertools
import threading
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import numpy as np

from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import StepError
from walkie_agent.perception.identity import cosine_distance

# Ends a remember's TIMEOUT or CANCELLED that came before anything could be stored.
NOTHING_STORED = "Nothing was stored."


class ModelVector(NamedTuple):
    """One embedding and the name of the model that made it.

    A face or appearance sample to store, or a query to match. `model` is
    the <model> part of walkie-database's vector table name, such as
    arcface_w600k_r50: vectors of different models are never compared.
    """

    model: str
    vector: np.ndarray


class Match(NamedTuple):
    """A stored person near a query, as the people services return it.

    `distance` is the cosine distance from the query to the person's nearest
    sample of the same model: 0 is the same direction, 1 is orthogonal.
    `name` may be blank for a person stored without one.
    """

    person_id: str
    name: str
    facts: Mapping[str, str]
    distance: float


class Stored(NamedTuple):
    """What remember stored: the new person's id and the vectors it kept.

    A backend stores only the vectors of models it has a table for.
    `faces` and `appearances` count the vectors stored, and
    `unstored_models` names the models whose vectors were left out, so
    those models can never match the person.
    """

    person_id: str
    faces: int
    appearances: int
    unstored_models: tuple[str, ...] = ()


class PeopleMemory(Protocol):
    """walkie-database's people services, as TrackMemory calls them.

    A match always returns the nearest stored people, nearest first, and
    never decides whether a query is a stranger: the caller applies a
    distance limit per model, since the limit depends on the model and the
    cameras (database.md, "Recognize a face"). The services return at most
    100 people per query. Person ids are never reused, so an id from an
    earlier task never names a new person.

    `timeout_s` is the longest wait for one answer of the store; None leaves
    the backend's own limit. A backend that cannot wait ignores it.

    remember and the matches raise StepError: UNAVAILABLE when the store is
    missing or failed in a way that may pass, so the same call may work later;
    REFUSED when the store refused the call, with its message; TIMEOUT when it
    did not answer in time, after which a remember may have stored the person
    already; and CANCELLED on a stop. A remember's TIMEOUT or CANCELLED whose
    message ends with NOTHING_STORED stored nobody. A ValueError is a bug in the caller, such
    as a remember without any vector, or an empty, zero or non-finite vector.
    """

    def remember(
        self,
        name: str,
        facts: Mapping[str, str],
        faces: Sequence[ModelVector],
        appearances: Sequence[ModelVector],
        *,
        timeout_s: float | None = None,
    ) -> Stored:
        """Store a person with their facts and samples; return their id and what was kept."""
        ...

    def match_faces(
        self, queries: Sequence[ModelVector], *, timeout_s: float | None = None
    ) -> list[list[Match]]:
        """For each query, the stored people with faces of its model, nearest first."""
        ...

    def match_appearances(
        self, queries: Sequence[ModelVector], *, timeout_s: float | None = None
    ) -> list[list[Match]]:
        """For each query, the stored people with appearances of its model, nearest first."""
        ...

    def clear(self, *, timeout_s: float | None = None) -> None:
        """Forget the people of this session, as at a new task.

        Never raises StepError. A backend that cannot forget now forgets
        before its next call, so the people of an earlier task are never
        matched in a later one.
        """
        ...


@dataclass
class _Person:
    name: str
    facts: dict[str, str]
    # Samples by model name, each a float32 row of that model's length.
    faces: dict[str, list[np.ndarray]] = field(default_factory=dict)
    appearances: dict[str, list[np.ndarray]] = field(default_factory=dict)


class InMemoryPeopleMemory:
    """A PeopleMemory kept in a dict, chosen with `--people-memory mock`.

    It serves simulation and the tests. Like walkie-database's match services,
    a person's distance to a query is the distance of their nearest sample, so
    several samples from different angles help. Unlike them, every person with
    a sample of the query's model is returned, with no cap of 100, and nobody
    is kept across tasks; the caller reads the list nearest first either way.

    Ids are `stored_<n>`, from 1, and are never reused, as in the database.
    clear() forgets everyone, as walkie-database forgets a session's guests
    (its ADR-0010). Every method answers at once and ignores `timeout_s`.

    Given the runtime's `cancel` token, remember writes through its
    unless_cancelled() fence, as FakeRobot's verbs commit: a stop that comes
    first raises StepError CANCELLED ending with NOTHING_STORED, and one that
    comes later waits for the write. That is its only StepError; a ValueError
    is a bug in the caller, as in PeopleMemory, and so is a query whose length
    differs from the stored vectors of its model.
    """

    def __init__(self, cancel: CancelToken | None = None) -> None:
        self._cancel = cancel
        self._lock = threading.Lock()
        self._people: dict[str, _Person] = {}
        self._ids = itertools.count(1)

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
            # Such a person could never be matched; remember_person checks
            # for vectors before it calls, so this is a bug in the caller.
            raise ValueError(f"Cannot remember {name!r} without a face or appearance vector.")
        person = _Person(name=name, facts=dict(facts))
        _add_samples(person.faces, faces)
        _add_samples(person.appearances, appearances)
        commit = (
            nullcontext()
            if self._cancel is None
            else self._cancel.unless_cancelled("remembering a person")
        )
        try:
            with commit, self._lock:
                person_id = f"stored_{next(self._ids)}"
                self._people[person_id] = person
        except StepError as error:
            # Only the fence raises it, before the write.
            raise StepError(error.code, f"{error.message} {NOTHING_STORED}") from error
        # The mock takes every model.
        return Stored(person_id, len(faces), len(appearances))

    def match_faces(
        self, queries: Sequence[ModelVector], *, timeout_s: float | None = None
    ) -> list[list[Match]]:
        return self._match(queries, faces=True)

    def match_appearances(
        self, queries: Sequence[ModelVector], *, timeout_s: float | None = None
    ) -> list[list[Match]]:
        return self._match(queries, faces=False)

    def clear(self, *, timeout_s: float | None = None) -> None:
        """Forget every person. Their ids are not given again."""
        with self._lock:
            self._people.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._people)

    def _match(self, queries: Sequence[ModelVector], *, faces: bool) -> list[list[Match]]:
        with self._lock:
            people = list(self._people.items())
        results = []
        for model, vector in queries:
            query = _checked(vector, model)
            matches = []
            for person_id, person in people:
                samples = (person.faces if faces else person.appearances).get(model)
                if not samples:
                    continue
                for sample in samples:
                    if sample.shape != query.shape:
                        # One model makes vectors of one length, so this is
                        # a bug, as a query on the wrong vector table would be.
                        raise ValueError(
                            f"A {model} query has {query.size} values, but a stored "
                            f"{model} vector has {sample.size}."
                        )
                distance = min(cosine_distance(sample, query) for sample in samples)
                matches.append(Match(person_id, person.name, dict(person.facts), distance))
            matches.sort(key=lambda match: match.distance)
            results.append(matches)
        return results


def _add_samples(store: dict[str, list[np.ndarray]], samples: Sequence[ModelVector]) -> None:
    for model, vector in samples:
        store.setdefault(model, []).append(_checked(vector, model))


def _checked(vector: np.ndarray, model: str) -> np.ndarray:
    # A copy, so a caller that reuses its array cannot change a stored
    # sample. A vector with no direction is refused rather than matched:
    # cosine_distance would put it at distance 1 from everyone, which hides
    # the bug that made it.
    row = np.array(vector, dtype=np.float32).ravel()
    if row.size == 0 or not np.isfinite(row).all() or not row.any():
        raise ValueError(f"A {model} vector must be non-empty, finite and not zero.")
    return row
