"""Recognize people met before: by face, else by appearance.

The vectors of each person handle, its own and those VisionFeed collected
for its track, are matched by the `people_memory` backend, which always
returns the nearest stored people. Each model is matched on its own, with
its own distance limit (perception.md, People), and each stored person goes
to one person in view at most (identity.assign_names).

A stored person without a name is matched like any other, so they still
block a weaker named match and the appearance fallback, but the line says
UNKNOWN: the LLM must not guess a name. remember_person always gives a
name, so such a person comes only from walkie-database's seed or another
writer. If the LLM then learns the name, remember_person stores a second
person rather than naming the first.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, Field

from walkie_agent.perception.identity import (
    IdentityConfig,
    assign_names,
    average_unit,
    reject_outliers,
)
from walkie_agent.perception.people_memory import Match, ModelVector
from walkie_agent.toolbox import Effect, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _perception as shared

META = ToolMeta(
    name="recognize_person",
    description=(
        "Recognize people you met before and stored with remember_person, by their face, "
        "else by their appearance. Give one person handle from find_people, such as "
        "per_51c0.1, or the set id from its first line, such as per_51c0, to recognize "
        "everyone it found at once; a name then goes to one person at most. One line per "
        "person: the handle, where they are in the image, and their name and stored "
        "facts, or UNKNOWN for someone you have not met or cannot recognize. Never guess "
        "the name of an UNKNOWN person."
    ),
    effects=frozenset({Effect.R}),
    questions=(
        "Who is this?",
        "Do you know these people?",
        "Which of the guests is Sam?",
        "Have we met before?",
    ),
    examples=({"person": "per_51c0.1"}, {"person": "per_51c0"}),
    requires=(shared.VISION, shared.PEOPLE_MEMORY),
)


class Args(BaseModel):
    person: str = Field(
        description=(
            "A person handle from find_people, such as per_51c0.1, or the set id from its "
            "first line, such as per_51c0, for everyone it found."
        )
    )


class Recognition(BaseModel):
    handle: str
    # None for UNKNOWN.
    person_id: str | None
    name: str | None
    facts: dict[str, str]
    matched_by: Literal["face", "appearance"] | None
    # The cosine distance of the match, under its model's limit: the match
    # of the model that named the person.
    distance: float | None


class Out(ToolOutput):
    people: list[Recognition]


# One query of a person, the mean of one model's vectors, and that model's
# distance limit. A person has one per model with a limit among their
# vectors of one kind.
_Query = tuple[ModelVector, float]


class _Scored(NamedTuple):
    """A stored person near one of a person's queries, scored for assign_names.

    `distance` is the score: the cosine distance over the limit of the
    query's model, so 1 is the limit of every model, and the matches of two
    models compare without their vectors ever being compared.
    """

    person_id: str
    distance: float
    match: Match


def run(ctx: ToolContext, args: Args) -> Out:
    identity = shared.CONFIG.identity
    handles = shared.person_handles(ctx, args.person)
    feed = shared.vision_feed(ctx)
    memory = ctx.backend(shared.PEOPLE_MEMORY)
    vectors = [shared.person_vectors(feed, handle.payload) for handle in handles]
    # Only a handle asked for alone fails: over a set, the others can still
    # be named, and a set of one answers as a larger set does.
    if shared.is_handle_id(args.person) and not any(vectors[0]):
        raise shared.no_vectors(handles[0].id)

    # Models without a limit, per person, to say why they stay UNKNOWN.
    unlimited: list[list[str]] = [[] for _ in handles]
    face_queries = [
        _queries(faces, identity.face_distance_limits, "face", unlimited[i], identity)
        for i, (faces, _) in enumerate(vectors)
    ]
    by_face = _assign(memory.match_faces, face_queries, given=set())
    # Appearance only for the people the face left unnamed, and only to the
    # stored people not named already: v2's fallback for a guest whose face
    # is turned away or too far to match.
    given = {match.person_id for match in by_face if match is not None}
    appearance_queries = [
        []
        if by_face[i] is not None
        else _queries(
            appearances, identity.appearance_distance_limits, "appearance",
            unlimited[i], identity,
        )
        for i, (_, appearances) in enumerate(vectors)
    ]
    by_appearance = _assign(memory.match_appearances, appearance_queries, given)

    people: list[Recognition] = []
    lines: list[str] = []
    for i, handle in enumerate(handles):
        prefix = f"{handle.id} ({handle.payload.position})"
        match = by_face[i] or by_appearance[i]
        if match is not None and not match.name.strip():
            lines.append(f"{prefix}: UNKNOWN, stored without a name")
            people.append(
                Recognition(
                    handle=handle.id,
                    person_id=None,
                    name=None,
                    facts={},
                    matched_by=None,
                    distance=None,
                )
            )
            continue
        if match is not None:
            how = "face" if by_face[i] is not None else "appearance"
            facts = dict(match.facts)
            lines.append(
                f"{prefix}: {match.name}" + (f", {shared.facts_text(facts)}" if facts else "")
            )
            people.append(
                Recognition(
                    handle=handle.id,
                    person_id=match.person_id,
                    name=match.name,
                    facts=facts,
                    matched_by=how,
                    distance=match.distance,
                )
            )
            continue
        if not any(vectors[i]):
            lines.append(f"{prefix}: UNKNOWN, no face or appearance seen")
        elif unlimited[i]:
            # A model without a tuned limit is never matched, so an unknown
            # model cannot give a confident wrong name (ADR-0050). The kind
            # that was matched and missed is named too, so the line does not
            # read as if the limit were the only reason.
            reasons = [
                f"{kind} not matched"
                for kind, queries in (("face", face_queries), ("appearance", appearance_queries))
                if queries[i]
            ]
            reasons.append(f"no distance limit for {' or '.join(unlimited[i])}")
            lines.append(f"{prefix}: UNKNOWN, {', '.join(reasons)}")
        else:
            lines.append(f"{prefix}: UNKNOWN")
        people.append(
            Recognition(
                handle=handle.id,
                person_id=None,
                name=None,
                facts={},
                matched_by=None,
                distance=None,
            )
        )
    return Out(summary="\n".join(lines), people=people)


def _queries(
    vectors: Sequence[ModelVector],
    limits: Mapping[str, float],
    kind: str,
    unlimited: list[str],
    identity: IdentityConfig,
) -> list[_Query]:
    """A person's queries of one kind: per model with a limit, the mean without outliers.

    Each model is queried on its own, since vectors of two models live in
    different spaces. A track shows two models after a model switch, and
    the stored person may have been met under either, so none is left out.
    A model with no limit adds "the <kind> model <name>" to `unlimited`.
    """
    queries: list[_Query] = []
    for model, group in shared.group_by_model(vectors).items():
        limit = limits.get(model)
        if limit is None:
            unlimited.append(f"the {kind} model {model}")
            continue
        mean = average_unit(reject_outliers(group, identity))
        assert mean is not None  # a group holds at least one vector
        queries.append((ModelVector(model, np.asarray(mean, dtype=np.float32)), limit))
    return queries


def _assign(
    match: Callable[[Sequence[ModelVector]], list[list[Match]]],
    queries: Sequence[Sequence[_Query]],
    given: set[str],
) -> list[Match | None]:
    """The stored person each person's queries name, under the limits, or None.

    One call for the whole set, as walkie-database's match services take a
    set of queries at once; RosPeopleMemory sends more than 32 in chunks.
    Each match is scored by its distance over its
    model's limit, and a stored person found by several of a person's
    models keeps the best score. assign_names then names over the scores,
    with the limit 1 for everyone. The people in `given` are named already.
    """
    asked = [(i, query, limit) for i, person in enumerate(queries) for query, limit in person]
    if not asked:
        return [None] * len(queries)
    results = match([query for _, query, _ in asked])
    best: list[dict[str, _Scored]] = [{} for _ in queries]
    for (i, _, limit), matches in zip(asked, results, strict=True):
        for m in matches:
            score = m.distance / limit
            # A NaN distance is never within a limit, and would stop a
            # finite score of the same stored person from replacing it.
            if m.person_id in given or math.isnan(score):
                continue
            kept = best[i].get(m.person_id)
            if kept is None or score < kept.distance:
                best[i][m.person_id] = _Scored(m.person_id, score, m)
    candidates = [sorted(scored.values(), key=lambda s: s.distance) for scored in best]
    named = assign_names(candidates, [1.0] * len(queries))
    return [None if scored is None else scored.match for scored in named]
