"""Face and appearance vectors: comparing, cleaning and naming, ported from walkie-agent-v2.

recognize_person matches a person's vectors against the people that
remember_person stored, and remember_person sends every vector collected
for the person's track (hri.md, Faces collected while talking). The cosine
helpers and the outlier rejection are walkie-agent-v2's tasks/HRI/identity.py
and walkie_world/people/store.py. v2's face quality gate is VisionFeed's
face_trusted, since VisionFeed applies it when it collects a face.
assign_names is new: it names the people of one find_people set together,
so no name is given twice.

Vectors are unit length, as walkie-vision's message comments state, and
are compared only within one model name: vectors of different weights live
in different spaces.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypeVar

import numpy as np

Vector = Sequence[float]


def _face_limits() -> Mapping[str, float]:
    return MappingProxyType(
        {
            # walkie-agent-v2's FACE_MATCH_THRESHOLD for the same ArcFace
            # weights, tuned on the robot.
            "arcface_w600k_r50": 0.4,
            # walkie-vision's fake face model (src/walkie_vision/fakes.py),
            # which oracle_perception and the tests publish. Hash vectors of
            # different inputs are nearly orthogonal, a distance near 1, so
            # 0.2 separates them.
            "fake_face": 0.2,
        }
    )


def _appearance_limits() -> Mapping[str, float]:
    return MappingProxyType(
        {
            # From walkie-agent-v2's APPEARANCE_MATCH_THRESHOLD, which gated a
            # score fused from face and appearance rather than appearance
            # alone, so this value is still to be tuned on the robot.
            "osnet_x1_0_msmt17": 0.5,
            # walkie-vision's fake re-identification model, as for fake_face.
            "fake_reid": 0.2,
        }
    )


@dataclass(frozen=True)
class IdentityConfig:
    """The thresholds for cleaning and matching vectors.

    The defaults are walkie-agent-v2's, tuned on the robot; the name in each
    comment is the v2 environment variable or constant that held the value.
    The face quality gate is not here but in VisionFeedConfig: VisionFeed
    applies it when it collects a face, and find_people applies the same
    function with the feed's config, so one copy decides both.
    """

    # A vector whose cosine similarity to the mean of its set is below this
    # is an outlier. HRI_BURST_OUTLIER_MIN_SIM, tuned on the robot.
    outlier_min_similarity: float = 0.5
    # A vector more than this many median absolute deviations below the
    # median similarity to the mean is an outlier. The median deviation, not
    # the standard one, so the outlier cannot widen the spread that should
    # catch it. HRI_BURST_OUTLIER_MAD_K, tuned on the robot.
    outlier_mad_k: float = 3.0
    # The largest cosine distance at which a stored face names a person, by
    # face_model. A model without a limit is never matched, so an unknown
    # model cannot produce a confident wrong name. Tuned again on the robot
    # with team members (perception.md, question 2), so config.toml can set
    # the limit of a model listed here, in
    # [walkie.perception.identity.face_distance_limits].
    face_distance_limits: Mapping[str, float] = field(default_factory=_face_limits)
    # The same for appearance vectors, by appearance_model, set in
    # [walkie.perception.identity.appearance_distance_limits].
    appearance_distance_limits: Mapping[str, float] = field(default_factory=_appearance_limits)

    def __post_init__(self) -> None:
        # A cosine distance lies from 0 to 2, and a limit of 2 would name
        # every person by the nearest stored one.
        for name in ("face_distance_limits", "appearance_distance_limits"):
            for model, limit in getattr(self, name).items():
                if not 0 < limit < 2:
                    raise ValueError(
                        f"IdentityConfig.{name}[{model!r}] must lie in (0, 2), not {limit}."
                    )


DEFAULT_IDENTITY_CONFIG = IdentityConfig()

# Keeps the division finite for a zero vector, as in walkie-agent-v2.
_NORM_EPSILON = 1e-8
# A mean shorter than this has no direction: its vectors cancel out.
_DEGENERATE_NORM = 1e-12


def cosine_similarity(a: Vector, b: Vector) -> float:
    """The cosine of the angle between two vectors of the same length."""
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + _NORM_EPSILON))


def cosine_distance(a: Vector, b: Vector) -> float:
    """1 minus the cosine similarity: 0 for the same direction, 1 for orthogonal."""
    return 1.0 - cosine_similarity(a, b)


def average_unit(vectors: Sequence[Vector]) -> list[float] | None:
    """The mean of the vectors, scaled to unit length; None when there are none.

    Averaging several shots of a person is steadier than any single one.
    When the vectors cancel out, the first vector is returned instead.
    """
    if not vectors:
        return None
    total = np.sum(np.asarray(vectors, dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(total))
    if norm <= _DEGENERATE_NORM:
        return [float(x) for x in vectors[0]]
    return [float(x) for x in total / norm]


def reject_outliers(
    vectors: Sequence[Vector], config: IdentityConfig = DEFAULT_IDENTITY_CONFIG
) -> list[Vector]:
    """The vectors without those far from the others' mean, in their order.

    A collection can catch a bystander, a half-turned head or a blurred body
    in one image, and averaging that in spoils the stored person. A vector
    goes when its cosine similarity to the mean is below
    `outlier_min_similarity`, or below the median similarity minus
    `outlier_mad_k` median absolute deviations. Fewer than 3 vectors are
    kept as they are, too few to tell a person from an outlier, and the
    vector closest to the mean is always kept, so a model is never left
    with nothing.
    """
    if len(vectors) < 3:
        return list(vectors)
    centroid = average_unit(vectors)
    assert centroid is not None
    similarities = [cosine_similarity(v, centroid) for v in vectors]
    # The upper median, as in walkie-agent-v2.
    median = sorted(similarities)[len(similarities) // 2]
    deviations = sorted(abs(s - median) for s in similarities)
    mad = deviations[len(deviations) // 2]
    floor = max(config.outlier_min_similarity, median - config.outlier_mad_k * mad)
    kept = [v for v, s in zip(vectors, similarities, strict=True) if s >= floor]
    if not kept:
        best = max(range(len(vectors)), key=lambda i: similarities[i])
        return [vectors[best]]
    return kept


class Match(Protocol):
    """A stored person near a query, such as PeopleMemory's matches."""

    @property
    def person_id(self) -> str: ...

    @property
    def distance(self) -> float: ...


M = TypeVar("M", bound=Match)


def assign_names(
    candidates: Sequence[Sequence[M]],
    limits: Sequence[float | None] | None = None,
) -> list[M | None]:
    """Give each person at most one stored person, closest match first.

    `candidates[i]` holds the stored people nearest to person i, for example
    the matches of its face. `limits[i]` is the largest distance that names
    person i, the limit of the model its matches came from; None there means
    the model has no limit, so person i is never named. Without `limits`
    every candidate counts, for a caller that applied the limits already.

    All (person, stored person) pairs within the limit are taken in order of
    ascending distance, and a pair is skipped when its person is named or
    its stored person is given already. So no name is given twice, no person
    gets two names, and a close match is never lost to a weaker one. Returns
    the match that names each person, or None for UNKNOWN.
    """
    if limits is not None and len(limits) != len(candidates):
        raise ValueError(
            f"Got {len(limits)} limits for {len(candidates)} people; give one per person."
        )
    pairs: list[tuple[float, int, int, M]] = []
    for person, matches in enumerate(candidates):
        limit = limits[person] if limits is not None else float("inf")
        if limit is None:
            continue
        for rank, match in enumerate(matches):
            # A NaN distance compares false with everything, so it is left out.
            if match.distance <= limit:
                pairs.append((match.distance, person, rank, match))
    # Ties go to the earlier person and then the earlier match, so the
    # result does not depend on the sort's stability.
    pairs.sort(key=lambda pair: pair[:3])
    names: list[M | None] = [None] * len(candidates)
    given: set[str] = set()
    for _, person, _, match in pairs:
        if names[person] is None and match.person_id not in given:
            names[person] = match
            given.add(match.person_id)
    return names
