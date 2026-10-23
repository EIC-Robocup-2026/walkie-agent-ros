"""Recognize people met before: by face, else by appearance.

The vectors of each person handle, those VisionFeed collected for its
track, are matched by the `people_memory` backend, which always returns the
nearest stored people. The agent applies a distance limit per model
(perception.md, People), and gives each stored person to one person in
view at most (identity.assign_names).
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

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
    # The cosine distance of the match, under its model's limit.
    distance: float | None


class Out(ToolOutput):
    people: list[Recognition]


# A person's query and the distance limit of its model, or None when the
# person has no vector of that kind or its model has no limit.
_Query = tuple[ModelVector, float] | None


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
        _query(faces, identity.face_distance_limits, "face", unlimited[i], identity)
        for i, (faces, _) in enumerate(vectors)
    ]
    by_face = _assign(memory.match_faces, face_queries, given=set())
    # Appearance only for the people the face left unnamed, and only to the
    # stored people not named already: v2's fallback for a guest whose face
    # is turned away or too far to match.
    given = {match.person_id for match in by_face if match is not None}
    appearance_queries: list[_Query] = []
    for i, (_, appearances) in enumerate(vectors):
        if by_face[i] is not None:
            appearance_queries.append(None)
            continue
        appearance_queries.append(
            _query(
                appearances, identity.appearance_distance_limits, "appearance",
                unlimited[i], identity,
            )
        )
    by_appearance = _assign(memory.match_appearances, appearance_queries, given)

    people: list[Recognition] = []
    lines: list[str] = []
    for i, handle in enumerate(handles):
        prefix = f"{handle.id} ({handle.payload.position})"
        match = by_face[i] or by_appearance[i]
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
                if queries[i] is not None
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


def _query(
    vectors: Sequence[ModelVector],
    limits: Mapping[str, float],
    kind: str,
    unlimited: list[str],
    identity: IdentityConfig,
) -> _Query:
    """One query from a person's vectors: their mean, without outliers.

    Only the newest model's vectors count, since vectors of two models live
    in different spaces; a track shows two only across a model switch. A
    model with no limit adds "the <kind> model <name>" to `unlimited`.
    """
    if not vectors:
        return None
    model = vectors[-1].model
    limit = limits.get(model)
    if limit is None:
        unlimited.append(f"the {kind} model {model}")
        return None
    group = [vector for name, vector in vectors if name == model]
    mean = average_unit(reject_outliers(group, identity))
    assert mean is not None  # group holds at least the newest vector
    return ModelVector(model, np.asarray(mean, dtype=np.float32)), limit


def _assign(
    match: Callable[[Sequence[ModelVector]], list[list[Match]]],
    queries: Sequence[_Query],
    given: set[str],
) -> list[Match | None]:
    """The stored person each query names, under its limit, or None.

    One call for the whole set, as walkie-database's people services take a
    set of faces at once. The people in `given` are named already.
    """
    asked = [i for i, query in enumerate(queries) if query is not None]
    if not asked:
        return [None] * len(queries)
    results = match([queries[i][0] for i in asked])
    candidates: list[list[Match]] = [[] for _ in queries]
    limits: list[float | None] = [None] * len(queries)
    for i, matches in zip(asked, results, strict=True):
        candidates[i] = [m for m in matches if m.person_id not in given]
        limits[i] = queries[i][1]
    return assign_names(candidates, limits)
