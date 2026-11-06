"""perception/identity.py: the cosine helpers, outlier rejection, the limits and naming.

Ported from the experiment's tests/test_perception.py, whose identity cases came from
walkie-agent-v2's tests/test_hri_identity.py and tests/test_people_store.py.
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from walkie_agent.perception.identity import (
    DEFAULT_IDENTITY_CONFIG,
    IdentityConfig,
    assign_names,
    average_unit,
    cosine_distance,
    cosine_similarity,
    reject_outliers,
)


def _u(*v: float) -> list[float]:
    """A small unit vector."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


# -- average_unit and the cosine helpers ---------------------------------------------


def test_average_unit_of_nothing_is_none():
    assert average_unit([]) is None


def test_average_unit_of_one_vector_is_renormalized():
    assert average_unit([[2.0, 0.0, 0.0]]) == pytest.approx([1.0, 0.0, 0.0])


def test_average_unit_is_unit_and_symmetric():
    out = average_unit([_u(1.0, 0.0, 0.0), _u(0.0, 1.0, 0.0)])
    assert math.isclose(math.hypot(*out), 1.0, abs_tol=1e-9)
    assert out[0] == pytest.approx(out[1])


def test_average_unit_of_opposite_vectors_falls_back_to_the_first():
    assert average_unit([[1.0, 0.0], [-1.0, 0.0]]) == [1.0, 0.0]


def test_average_unit_reads_float32_arrays():
    out = average_unit([np.array([3.0, 4.0], dtype=np.float32)])
    assert out == pytest.approx([0.6, 0.8])
    assert all(isinstance(x, float) for x in out)


# Three well separated people's faces, from v2's test_people_store.py.
ALICE = _u(1.0, 0.05, 0.0)
ALICE_2 = _u(0.97, 0.10, 0.05)  # A second shot of Alice.
BOB = _u(0.0, 1.0, 0.05)
CAROL = _u(0.0, 0.05, 1.0)


def test_cosine_helpers():
    assert cosine_similarity(ALICE, ALICE) == pytest.approx(1.0)
    assert cosine_distance(ALICE, ALICE) == pytest.approx(0.0, abs=1e-7)
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)
    # A zero vector is guarded rather than dividing by zero.
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_helpers_ignore_the_length_and_read_float32_arrays():
    a = np.array([3.0, 4.0], dtype=np.float32)
    assert cosine_similarity(a, [0.6, 0.8]) == pytest.approx(1.0)
    assert isinstance(cosine_distance(a, a), float)


# -- reject_outliers -----------------------------------------------------------------


def test_reject_outliers_drops_the_odd_vector():
    a1, a2 = _u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01)
    far = _u(0.0, 1.0, 0.0)
    kept = reject_outliers([a1, a2, far])
    assert len(kept) == 2 and far not in kept


def test_reject_outliers_keeps_a_uniform_set():
    vectors = [_u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01), _u(0.99, 0.03, 0.0)]
    assert reject_outliers(vectors) == vectors


def test_reject_outliers_keeps_fewer_than_three():
    vectors = [_u(1.0, 0.0, 0.0), _u(0.0, 1.0, 0.0)]
    assert reject_outliers(vectors) == vectors
    assert reject_outliers([]) == []


def test_reject_outliers_never_returns_nothing():
    # A spread so wide that the mean is nearly zero still keeps one vector.
    vectors = [_u(1.0, 0.0, 0.0), _u(-0.5, 0.866, 0.0), _u(-0.5, -0.866, 0.0)]
    assert len(reject_outliers(vectors)) >= 1


def test_reject_outliers_keeps_the_vector_closest_to_the_mean_when_all_fail_the_floor():
    # Every vector is below a floor of 0.99, so only the closest one stays.
    vectors = [_u(1.0, 0.3, 0.0), _u(1.0, 0.0, 0.3), _u(1.0, 0.05, 0.05)]
    strict = IdentityConfig(outlier_min_similarity=0.999)
    assert reject_outliers(vectors, strict) == [vectors[2]]


def test_reject_outliers_drops_one_outlier_of_four():
    cluster = [_u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01), _u(0.99, 0.03, 0.0)]
    far = _u(0.0, 1.0, 0.0)
    kept = reject_outliers(cluster + [far])
    assert far not in kept and len(kept) == 3


def test_reject_outliers_keeps_the_order_of_the_vectors():
    cluster = [_u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01), _u(0.99, 0.03, 0.0)]
    far = _u(0.0, 1.0, 0.0)
    assert reject_outliers([cluster[0], far, cluster[1], cluster[2]]) == cluster


def test_reject_outliers_follows_the_config():
    a1, a2 = _u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01)
    far = _u(0.0, 1.0, 0.0)
    # With no floor and a gate too wide to bite, the far vector stays.
    loose = IdentityConfig(outlier_min_similarity=-1.0, outlier_mad_k=1e6)
    assert reject_outliers([a1, a2, far], loose) == [a1, a2, far]


# -- assign_names --------------------------------------------------------------------


@dataclass(frozen=True)
class _Match:
    person_id: str
    name: str
    distance: float


def _nearest(query: list[float], stored: dict[str, list[float]]) -> list[_Match]:
    """What PeopleMemory returns: every stored person, nearest first."""
    matches = [
        _Match(person_id=name.lower(), name=name, distance=cosine_distance(query, vector))
        for name, vector in stored.items()
    ]
    return sorted(matches, key=lambda m: m.distance)


def test_recognize_a_second_shot_and_not_a_stranger():
    stored = {"Alice": ALICE, "Bob": BOB}
    names = assign_names([_nearest(ALICE_2, stored), _nearest(CAROL, stored)], [0.4, 0.4])
    assert names[0] is not None and names[0].name == "Alice" and names[0].distance < 0.4
    assert names[1] is None


def test_a_strict_limit_rejects_even_a_real_match():
    stored = {"Alice": ALICE}
    assert assign_names([_nearest(ALICE_2, stored)], [0.0]) == [None]
    assert assign_names([_nearest(ALICE_2, stored)], [0.4])[0].name == "Alice"


def test_assign_names_crossed_distances():
    # The guests switched places since they were remembered: the left person
    # is nearer Kim and the right one nearer Sam, so the names follow the
    # distances, not the order the guests were stored in.
    candidates = [
        [_Match("kim", "Kim", 0.10), _Match("sam", "Sam", 0.30)],
        [_Match("sam", "Sam", 0.15), _Match("kim", "Kim", 0.20)],
    ]
    names = assign_names(candidates, [0.4, 0.4])
    assert [n.name for n in names] == ["Kim", "Sam"]


def test_assign_names_two_people_nearest_the_same_guest():
    # Both people are nearest Sam, so naming each person by their own
    # nearest match would call both Sam. The right person is the closer
    # match and keeps Sam; the left person gets Kim, their second best,
    # although the left person comes first.
    candidates = [
        [_Match("sam", "Sam", 0.10), _Match("kim", "Kim", 0.20)],
        [_Match("sam", "Sam", 0.05), _Match("kim", "Kim", 0.30)],
    ]
    names = assign_names(candidates, [0.4, 0.4])
    assert [n.name for n in names] == ["Kim", "Sam"]


def test_assign_names_lets_the_closer_person_keep_a_contested_name():
    # v2's test_locate_greedy_steal_fixed: both faces are nearest to guest 2,
    # face b far more so, so face a gets guest 1, its second best.
    guest_1, guest_2 = _u(1, 0, 0), _u(0, 1, 0)
    face_a = _u(0.52, 0.55, 0.6535)
    face_b = _u(0.10, 0.90, 0.4243)
    stored = {"Guest1": guest_1, "Guest2": guest_2}
    names = assign_names([_nearest(face_a, stored), _nearest(face_b, stored)], [0.6, 0.6])
    assert [n.name for n in names] == ["Guest1", "Guest2"]


def test_assign_names_a_third_person_is_unknown():
    stored = {"Sam": _u(1, 0, 0), "Kim": _u(0, 1, 0)}
    people = [_u(1, 0.05, 0), _u(0.05, 1, 0), _u(0, 0, 1)]
    names = assign_names([_nearest(p, stored) for p in people], [0.4, 0.4, 0.4])
    assert [n.name if n else None for n in names] == ["Sam", "Kim", None]


def test_assign_names_gives_one_stored_person_to_one_person_only():
    # Three people all near one stored guest: only the closest is named.
    candidates = [
        [_Match("sam", "Sam", 0.20)],
        [_Match("sam", "Sam", 0.05)],
        [_Match("sam", "Sam", 0.10)],
    ]
    names = assign_names(candidates, [0.4, 0.4, 0.4])
    assert [n.name if n else None for n in names] == [None, "Sam", None]


def test_assign_names_a_limit_excludes_a_far_match():
    candidates = [[_Match("sam", "Sam", 0.6)], [_Match("kim", "Kim", 0.1)]]
    names = assign_names(candidates, [0.4, 0.4])
    assert names[0] is None and names[1].name == "Kim"
    # The second best within the limit names the person whose best is taken.
    candidates = [
        [_Match("sam", "Sam", 0.05)],
        [_Match("sam", "Sam", 0.10), _Match("kim", "Kim", 0.35), _Match("lee", "Lee", 0.45)],
    ]
    names = assign_names(candidates, [0.4, 0.4])
    assert [n.name for n in names] == ["Sam", "Kim"]


def test_assign_names_a_match_on_the_limit_names_the_person():
    assert assign_names([[_Match("sam", "Sam", 0.4)]], [0.4])[0].name == "Sam"


def test_assign_names_limits_are_per_person():
    candidates = [[_Match("sam", "Sam", 0.3)], [_Match("kim", "Kim", 0.3)]]
    names = assign_names(candidates, [0.2, 0.5])
    assert names[0] is None and names[1].name == "Kim"


def test_assign_names_never_matches_a_model_without_a_limit():
    candidates = [[_Match("sam", "Sam", 0.01)]]
    assert assign_names(candidates, [None]) == [None]


def test_assign_names_a_person_without_a_limit_leaves_the_name_to_the_others():
    # The first person's model has no limit, so its near match does not take Sam.
    candidates = [[_Match("sam", "Sam", 0.01)], [_Match("sam", "Sam", 0.3)]]
    names = assign_names(candidates, [None, 0.4])
    assert names[0] is None and names[1].name == "Sam"


def test_assign_names_without_limits_takes_every_candidate():
    candidates = [[_Match("sam", "Sam", 0.9)], [_Match("sam", "Sam", 0.5)], []]
    names = assign_names(candidates)
    assert names[0] is None and names[1].name == "Sam" and names[2] is None


def test_assign_names_of_nobody_is_empty():
    assert assign_names([], []) == []


def test_assign_names_breaks_ties_by_order():
    candidates = [[_Match("sam", "Sam", 0.2)], [_Match("sam", "Sam", 0.2)]]
    names = assign_names(candidates, [0.4, 0.4])
    assert names[0].name == "Sam" and names[1] is None


def test_assign_names_skips_a_nan_distance():
    assert assign_names([[_Match("sam", "Sam", math.nan)]], [0.4]) == [None]


def test_assign_names_needs_one_limit_per_person():
    with pytest.raises(ValueError, match="one per person"):
        assign_names([[], []], [0.4])


def test_assign_names_returns_the_callers_match_objects():
    sam = _Match("sam", "Sam", 0.1)
    assert assign_names([[sam]], [0.4])[0] is sam


# -- IdentityConfig ------------------------------------------------------------------


def test_identity_config_limits_by_model():
    config = IdentityConfig()
    assert config.face_distance_limits["arcface_w600k_r50"] == 0.4
    assert config.face_distance_limits["fake_face"] == 0.2
    assert config.appearance_distance_limits["osnet_x1_0_msmt17"] == 0.5
    assert config.appearance_distance_limits["fake_reid"] == 0.2
    assert config.face_distance_limits.get("unknown_model") is None
    assert config.appearance_distance_limits.get("unknown_model") is None
    with pytest.raises(TypeError):
        config.face_distance_limits["arcface_w600k_r50"] = 1.0  # type: ignore[index]


def test_identity_config_defaults_are_the_experiments():
    assert DEFAULT_IDENTITY_CONFIG == IdentityConfig()
    assert DEFAULT_IDENTITY_CONFIG.outlier_min_similarity == 0.5
    assert DEFAULT_IDENTITY_CONFIG.outlier_mad_k == 3.0
    assert dict(DEFAULT_IDENTITY_CONFIG.face_distance_limits) == {
        "arcface_w600k_r50": 0.4,
        "fake_face": 0.2,
    }
    assert dict(DEFAULT_IDENTITY_CONFIG.appearance_distance_limits) == {
        "osnet_x1_0_msmt17": 0.5,
        "fake_reid": 0.2,
    }


def test_each_identity_config_has_its_own_limits():
    assert IdentityConfig().face_distance_limits is not IdentityConfig().face_distance_limits


def test_identity_config_is_frozen():
    with pytest.raises(AttributeError):
        DEFAULT_IDENTITY_CONFIG.outlier_mad_k = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("limit", [0.0, -0.1, 2.0, 2.5])
@pytest.mark.parametrize("field", ["face_distance_limits", "appearance_distance_limits"])
def test_identity_config_refuses_a_limit_outside_0_to_2(field, limit):
    with pytest.raises(ValueError, match=rf"{field}\['m'\] must lie in \(0, 2\)"):
        IdentityConfig(**{field: {"m": limit}})


def test_identity_config_takes_a_limit_inside_0_to_2():
    config = IdentityConfig(face_distance_limits={"m": 1.99}, appearance_distance_limits={})
    assert config.face_distance_limits["m"] == 1.99
    assert config.appearance_distance_limits == {}
