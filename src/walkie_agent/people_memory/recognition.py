"""Naming the people in view from their samples, and cleaning the samples to store.

TrackMemory (track_memory.py) calls these for ctx.memory (ADR-0070). They are the rules of
the experiment's remember_person and recognize_person atomics (experiment/hri-agent's
tools/atomic), taken out of its toolbox. No ROS.

- without_outliers drops each model's outliers from the samples to store, model by
  model, since vectors of two models are never compared.
- name_people names a batch of people together, face first, then appearance:
  - A person's query of one kind is, per model with a distance limit, the mean of
    that model's samples without outliers. A model with no limit never matches, so
    an unknown model cannot give a confident wrong name.
  - Each match is scored by its cosine distance over its model's limit, so 1 is the
    limit of every model and the matches of two models compare without their
    vectors ever being compared. A stored person found by several of a person's
    models keeps the best score.
  - identity.assign_names then names over the scores with the limit 1.0, so one
    stored person goes to one person in view at most.
  - Appearance is matched only for the people the face left unnamed, and only to the
    stored people the face did not name: v2's fallback for a guest whose face is
    turned away or too far to match.
  - A stored person with a blank name is matched like any other, so they still block
    a weaker named match and the appearance fallback, but they read as unknown: the
    model must not guess a name. remember_person always gives a name, so such a
    person comes only from walkie-database's seed or another writer.
"""

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, NamedTuple

import numpy as np

from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.people_memory.identity import (
    DEFAULT_IDENTITY_CONFIG,
    IdentityConfig,
    assign_names,
    average_unit,
    reject_outliers,
)
from walkie_agent.people_memory.store import Match, ModelVector, PeopleMemory
from walkie_agent.people_memory.track_vectors import Collected

MatchedBy = Literal["face", "appearance"]


class Naming(NamedTuple):
    """Who one person of a batch is: the stored person that names them, and how.

    `match` is None, and `matched_by` with it, for someone not recognized: nobody
    stored within the limits, or a stored person with a blank name.
    """

    match: Match | None
    matched_by: MatchedBy | None


_UNNAMED = Naming(None, None)

# One query of a person, the mean of one model's samples, with that model's limit.
_Query = tuple[ModelVector, float]


class _Scored(NamedTuple):
    """A stored person near one of a person's queries, scored for assign_names.

    `distance` is the score, the cosine distance over the limit of the query's model.
    """

    person_id: str
    distance: float
    match: Match


def without_outliers(
    vectors: Sequence[ModelVector], config: IdentityConfig = DEFAULT_IDENTITY_CONFIG
) -> list[ModelVector]:
    """The vectors without each model's outliers, model by model in order of first sight.

    Outliers are judged within one model, by identity.reject_outliers: a bystander's
    face caught in one image, or a blurred body. Each model keeps at least one vector,
    so a non-empty input never gives an empty output. The arrays are the input's own.
    """
    kept: list[ModelVector] = []
    for model, group in _by_model(vectors).items():
        kept.extend(ModelVector(model, vector) for vector in reject_outliers(group, config))
    return kept


def name_people(
    store: PeopleMemory,
    people: Sequence[Collected],
    config: IdentityConfig = DEFAULT_IDENTITY_CONFIG,
    *,
    timeout_s: float | None = None,
) -> list[Naming]:
    """Name each person of `people` from their samples, one stored person per person.

    Returns one Naming per person, in order. Makes at most one match_faces call and
    one match_appearances call for the whole batch, and none of a kind without a
    query. `timeout_s` bounds the whole naming: each call gets what is left of it as
    its own `timeout_s`, and None leaves the store its own limits.

    Raises the store's StepError, and TIMEOUT when no time is left for a call. A
    ValueError is a bug, such as a store that answers a different number of queries
    than it was asked, or samples of one model with different lengths.
    """
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    face_queries = [_queries(person.faces, config.face_distance_limits, config) for person in people]
    by_face = _assign(store.match_faces, face_queries, set(), deadline, timeout_s)
    given = {match.person_id for match in by_face if match is not None}
    appearance_queries = [
        []
        if by_face[i] is not None
        else _queries(person.appearances, config.appearance_distance_limits, config)
        for i, person in enumerate(people)
    ]
    by_appearance = _assign(
        store.match_appearances, appearance_queries, given, deadline, timeout_s
    )
    namings: list[Naming] = []
    for face, appearance in zip(by_face, by_appearance, strict=True):
        naming = _UNNAMED
        if face is not None:
            naming = Naming(face, "face")
        elif appearance is not None:
            naming = Naming(appearance, "appearance")
        if naming.match is not None and not naming.match.name.strip():
            naming = _UNNAMED
        namings.append(naming)
    return namings


def _by_model(vectors: Sequence[ModelVector]) -> dict[str, list[np.ndarray]]:
    """The vectors by model name, in order of first sight, each group in its own order."""
    groups: dict[str, list[np.ndarray]] = {}
    for model, vector in vectors:
        groups.setdefault(model, []).append(vector)
    return groups


def _queries(
    vectors: Sequence[ModelVector], limits: Mapping[str, float], config: IdentityConfig
) -> list[_Query]:
    """A person's queries of one kind: per model with a limit, the mean without outliers.

    A track shows two models after walkie-vision switches models, and the stored
    person may have been met under either, so every model with a limit is queried.
    """
    queries: list[_Query] = []
    for model, group in _by_model(vectors).items():
        limit = limits.get(model)
        if limit is None:
            continue
        mean = average_unit(reject_outliers(group, config))
        assert mean is not None  # A group holds at least one vector.
        queries.append((ModelVector(model, np.asarray(mean, dtype=np.float32)), limit))
    return queries


def _assign(
    match: Callable[..., list[list[Match]]],
    queries: Sequence[Sequence[_Query]],
    given: set[str],
    deadline: float | None,
    timeout_s: float | None,
) -> list[Match | None]:
    """The stored person each person's queries name, under the limits, or None.

    One call for every query of the batch, as walkie-database's match services take
    a set of queries at once. The stored people in `given` are named already.
    """
    asked = [(i, query, limit) for i, person in enumerate(queries) for query, limit in person]
    if not asked:
        return [None] * len(queries)
    results = match([query for _, query, _ in asked], timeout_s=_left(deadline, timeout_s))
    if len(results) != len(asked):
        raise ValueError(
            f"The people memory answered {len(results)} queries, not the {len(asked)} asked."
        )
    best: list[dict[str, _Scored]] = [{} for _ in queries]
    for (i, _, limit), matches in zip(asked, results, strict=True):
        for m in matches:
            score = m.distance / limit
            # A NaN distance is never within a limit, and would stop a finite score of
            # the same stored person from replacing it.
            if m.person_id in given or math.isnan(score):
                continue
            kept = best[i].get(m.person_id)
            if kept is None or score < kept.distance:
                best[i][m.person_id] = _Scored(m.person_id, score, m)
    candidates = [sorted(scored.values(), key=lambda s: s.distance) for scored in best]
    named = assign_names(candidates, [1.0] * len(queries))
    return [None if scored is None else scored.match for scored in named]


def _left(deadline: float | None, timeout_s: float | None) -> float | None:
    """What is left of `timeout_s` at `deadline` for the next call; TIMEOUT when nothing is."""
    if deadline is None:
        return None
    left = deadline - time.monotonic()
    # Written so that a NaN timeout fails too.
    if not left > 0:
        # Rounded, since a caller passes what is left of its own timeout, such as
        # 14.9993 s of 15 s.
        raise StepError(
            ErrorCode.TIMEOUT,
            "Timed out: matching the people in view did not finish within "
            f"{round(timeout_s, 1):g} s.",
        )
    return left
