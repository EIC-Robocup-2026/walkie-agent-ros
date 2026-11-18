"""people_memory/recognition.py:without_outliers and name_people.

The naming cases are ported from the experiment's tests/test_people_atomics.py, where
recognize_person ran them through its toolbox; here they run on the pure functions
against InMemoryPeopleMemory, with hash_vector vectors under walkie-vision's fake model
names, as oracle_perception publishes them.
"""

import math

import numpy as np
import pytest

from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.people_memory import recognition
from walkie_agent.people_memory.identity import DEFAULT_IDENTITY_CONFIG, average_unit
from walkie_agent.people_memory.recognition import Naming, name_people, without_outliers
from walkie_agent.people_memory.store import InMemoryPeopleMemory, Match, ModelVector
from walkie_agent.people_memory.track_vectors import Collected
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, FAKE_REID_MODEL, hash_vector

DIM = 64
ARCFACE_MODEL = "arcface_w600k_r50"


def vector(kind: str, key: str, shot: int = 0, weight: float = 0.1) -> np.ndarray:
    """A unit vector of one person; another shot is the same one nudged a little."""
    base = hash_vector(f"{kind} {key}", DIM)
    if shot == 0:
        return base
    nudged = base + weight * hash_vector(f"{kind} {key} shot {shot}", DIM)
    return (nudged / np.linalg.norm(nudged)).astype(np.float32)


def face(key: str, shot: int = 0, model: str = FAKE_FACE_MODEL, **kw) -> ModelVector:
    return ModelVector(model, vector("face", key, shot, **kw))


def appearance(key: str, shot: int = 0, **kw) -> ModelVector:
    return ModelVector(FAKE_REID_MODEL, vector("reid", key, shot, **kw))


def person(key: str | None, shots=(1, 2, 3), *, faces: bool = True, appearances: bool = True):
    """The samples of `key` over a few shots; None for nobody."""
    if key is None:
        return Collected()
    return Collected(
        faces=tuple(face(key, s) for s in shots) if faces else (),
        appearances=tuple(appearance(key, s) for s in shots) if appearances else (),
    )


class Spy:
    """InMemoryPeopleMemory with a record of every match call."""

    def __init__(self, store: InMemoryPeopleMemory | None = None) -> None:
        self.store = store or InMemoryPeopleMemory()
        self.calls: list[tuple[str, list[ModelVector], float | None]] = []

    def remember(self, *args, **kwargs):
        return self.store.remember(*args, **kwargs)

    def match_faces(self, queries, *, timeout_s=None):
        self.calls.append(("faces", list(queries), timeout_s))
        return self.store.match_faces(queries, timeout_s=timeout_s)

    def match_appearances(self, queries, *, timeout_s=None):
        self.calls.append(("appearances", list(queries), timeout_s))
        return self.store.match_appearances(queries, timeout_s=timeout_s)

    def clear(self, *, timeout_s=None):
        self.store.clear(timeout_s=timeout_s)

    def kinds(self) -> list[str]:
        return [kind for kind, _, _ in self.calls]


def meet(store, name: str, key: str, facts=None, *, faces=True, appearances=True) -> str:
    """Store `key` under `name` from its first shot, as remember_person would."""
    return store.remember(
        name,
        facts or {},
        [face(key)] if faces else [],
        [appearance(key)] if appearances else [],
    ).person_id


def names(namings: list[Naming]) -> list[str | None]:
    return [None if n.match is None else n.match.name for n in namings]


def ways(namings: list[Naming]) -> list[str | None]:
    return [n.matched_by for n in namings]


# -- without_outliers -------------------------------------------------------------------


def test_without_outliers_drops_a_far_vector_within_its_model():
    sam = [face("sam", shot) for shot in range(1, 6)]
    bob = face("bob")

    kept = without_outliers([*sam[:3], bob, *sam[3:]])

    assert [v.vector is s.vector for v, s in zip(kept, sam, strict=True)] == [True] * 5


def test_without_outliers_goes_model_by_model_in_order_of_first_sight():
    reid = [appearance("sam", shot) for shot in (1, 2)]
    faces = [face("sam", shot) for shot in (1, 2)]

    kept = without_outliers([reid[0], faces[0], reid[1], faces[1]])

    assert [v.vector is e.vector for v, e in zip(kept, [*reid, *faces], strict=True)] == [True] * 4
    assert [v.model for v in kept] == [FAKE_REID_MODEL] * 2 + [FAKE_FACE_MODEL] * 2


def test_without_outliers_judges_each_model_on_its_own():
    # Two samples of a model are too few to tell an outlier, whatever the other
    # model's samples are.
    odd_pair = [face("sam"), face("bob")]
    many = [face("ana", shot, model=ARCFACE_MODEL) for shot in range(1, 5)]

    kept = without_outliers([*odd_pair, *many])

    assert len(kept) == 6


def test_without_outliers_of_nothing_is_nothing():
    assert without_outliers([]) == []


# -- name_people: face first, then appearance ------------------------------------------


def test_a_remembered_face_names_the_person():
    spy = Spy()
    meet(spy.store, "Sam", "sam", {"drink": "milk"})

    [naming] = name_people(spy, [person("sam")])

    assert naming.matched_by == "face"
    assert (naming.match.name, dict(naming.match.facts)) == ("Sam", {"drink": "milk"})
    assert naming.match.distance < 0.05
    # The face named him, so his appearance was never matched.
    assert spy.kinds() == ["faces"]


def test_a_stranger_is_not_named():
    spy = Spy()
    meet(spy.store, "Sam", "sam")

    assert name_people(spy, [person("dave")]) == [Naming(None, None)]


def test_nobody_stored_names_nobody():
    assert name_people(InMemoryPeopleMemory(), [person("sam"), person("ana")]) == [
        Naming(None, None)
    ] * 2


def test_appearance_names_a_person_whose_face_is_not_seen():
    spy = Spy()
    meet(spy.store, "Sam", "sam")

    [naming] = name_people(spy, [person("sam", faces=False)])

    assert (naming.match.name, naming.matched_by) == ("Sam", "appearance")
    # Nobody had a face query, so no face call was made.
    assert spy.kinds() == ["appearances"]


def test_appearance_names_only_the_unnamed_and_only_to_the_stored_people_left():
    spy = Spy()
    meet(spy.store, "Sam", "sam")
    # Sam's face on the left, and someone in the same clothes, turned away, on the
    # right: the name goes to the face, and only once.
    pair = [person("sam"), person("sam", shots=(4, 5), faces=False)]

    namings = name_people(spy, pair)

    assert names(namings) == ["Sam", None]
    assert ways(namings) == ["face", None]
    # Only the unnamed person's appearance was asked for.
    [_, (kind, queries, _)] = spy.calls
    assert kind == "appearances" and len(queries) == 1


def test_a_face_that_misses_falls_back_to_the_appearance():
    spy = Spy()
    meet(spy.store, "Sam", "sam")
    # Sam's clothes on someone whose face is Dave's, as when Sam's face is too far
    # or turned to match: the appearance names him.
    seen = Collected(
        faces=tuple(face("dave", s) for s in (1, 2)),
        appearances=tuple(appearance("sam", s) for s in (1, 2)),
    )

    [naming] = name_people(spy, [seen])

    assert (naming.match.name, naming.matched_by) == ("Sam", "appearance")
    assert spy.kinds() == ["faces", "appearances"]


def test_two_remembered_guests_are_named_after_they_switch_places():
    store = InMemoryPeopleMemory()
    meet(store, "Sam", "sam", {"drink": "milk"})
    meet(store, "Ana", "ana")

    namings = name_people(store, [person("ana"), person("sam")])

    assert names(namings) == ["Ana", "Sam"]
    assert dict(namings[1].match.facts) == {"drink": "milk"}


def test_a_third_person_never_met_is_unknown():
    store = InMemoryPeopleMemory()
    meet(store, "Sam", "sam")
    meet(store, "Ana", "ana")

    namings = name_people(store, [person("sam"), person("dave"), person("ana")])

    assert names(namings) == ["Sam", None, "Ana"]
    assert ways(namings) == ["face", None, "face"]


def test_one_stored_person_names_one_person_in_view_only():
    spy = Spy()
    meet(spy.store, "Sam", "sam", appearances=False)
    # Two people near Sam's face; the nearer is Sam, and the other stays unknown,
    # though alone they would be named Sam too.
    near = person("sam", shots=(1, 2), appearances=False)
    farther = Collected(faces=(face("sam", 7, weight=0.5), face("sam", 8, weight=0.5)))

    assert names(name_people(spy, [farther])) == ["Sam"]
    assert names(name_people(spy, [farther, near])) == [None, "Sam"]
    assert names(name_people(spy, [near, farther])) == ["Sam", None]


def test_the_whole_batch_takes_one_call_per_kind():
    spy = Spy()
    meet(spy.store, "Sam", "sam")
    people = [person("sam"), person("dave"), person("ana", faces=False), person(None)]

    namings = name_people(spy, people)

    assert names(namings) == ["Sam", None, None, None]
    assert spy.kinds() == ["faces", "appearances"]
    face_queries, appearance_queries = (queries for _, queries, _ in spy.calls)
    # Sam's and Dave's faces; then the appearances of the unnamed Dave and Ana.
    assert len(face_queries) == 2 and len(appearance_queries) == 2


def test_people_without_samples_make_no_call():
    spy = Spy()
    meet(spy.store, "Sam", "sam")

    assert name_people(spy, [person(None), person(None)]) == [Naming(None, None)] * 2
    assert name_people(spy, []) == []
    assert spy.calls == []


def test_the_query_is_the_mean_of_a_models_samples_without_outliers():
    spy = Spy()
    meet(spy.store, "Sam", "sam")
    sam = [face("sam", shot) for shot in range(1, 6)]
    # One image where a bystander's face was matched to Sam's box.
    seen = Collected(faces=(*sam, face("bob")))

    [naming] = name_people(spy, [seen])

    [(_, [query], _)] = spy.calls
    assert query.model == FAKE_FACE_MODEL and query.vector.dtype == np.float32
    expected = average_unit([v.vector for v in sam])
    assert np.allclose(query.vector, expected, atol=1e-6)
    assert naming.match.name == "Sam"


# -- name_people: blank names and models ------------------------------------------------


def test_a_stored_person_without_a_name_is_unknown_and_blocks_a_weaker_name():
    spy = Spy()
    # Only walkie-database's seed or another writer stores someone without a name.
    # The nameless person is the nearer match, so neither Sam's face nor his
    # appearance names them.
    spy.store.remember("", {"drink": "tea"}, [face("sam")], [appearance("sam")])
    spy.store.remember("Sam", {"drink": "milk"}, [face("sam", 9)], [appearance("sam", 9)])

    assert name_people(spy, [person("sam")]) == [Naming(None, None)]
    # The blank-named match counts as a face match, so no appearance call is made.
    assert spy.kinds() == ["faces"]


def test_a_blank_named_match_by_appearance_is_unknown_too():
    store = InMemoryPeopleMemory()
    store.remember("   ", {}, [], [appearance("sam")])

    assert name_people(store, [person("sam", faces=False)]) == [Naming(None, None)]


def test_a_face_model_without_a_limit_is_never_matched():
    spy = Spy()
    spy.store.remember("Sam", {}, [face("sam", model="mystery_face")], [])
    seen = Collected(faces=tuple(face("sam", s, model="mystery_face") for s in (1, 2)))

    assert name_people(spy, [seen]) == [Naming(None, None)]
    # Not even asked: the model has no limit.
    assert spy.calls == []


def test_an_unknown_face_model_still_leaves_the_appearance():
    store = InMemoryPeopleMemory()
    store.remember("Sam", {}, [face("sam", model="mystery_face")], [appearance("sam")])
    seen = Collected(
        faces=(face("sam", 1, model="mystery_face"),), appearances=(appearance("sam", 1),)
    )

    [naming] = name_people(store, [seen])

    assert (naming.match.name, naming.matched_by) == ("Sam", "appearance")


def unit(*parts: tuple[int, float]) -> np.ndarray:
    """A unit vector with the given (index, value) parts, the rest 0."""
    v = np.zeros(DIM, dtype=np.float32)
    for index, value in parts:
        v[index] = value
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("models", [(FAKE_FACE_MODEL, ARCFACE_MODEL), (ARCFACE_MODEL, FAKE_FACE_MODEL)])
def test_each_model_is_scored_by_its_own_limit(models):
    store = InMemoryPeopleMemory()
    # Ana is stored under ArcFace (limit 0.4), Bea under the fake model (limit 0.2).
    # The person matches Ana at a distance of 0.20 and Bea at 0.15: Bea is nearer in
    # raw distance, but Ana is at half her model's limit and Bea at three quarters of
    # hers, so Ana is the better match.
    store.remember("Ana", {}, [ModelVector(ARCFACE_MODEL, unit((0, 1.0)))], [])
    store.remember("Bea", {}, [ModelVector(FAKE_FACE_MODEL, unit((2, 1.0)))], [])
    shown = {
        ARCFACE_MODEL: unit((0, 0.8), (1, 0.6)),
        FAKE_FACE_MODEL: unit((2, 0.85), (3, math.sqrt(1 - 0.85**2))),
    }
    # The same track under each model in turn, so a match on the first or the last
    # model alone fails one of the two orders.
    seen = Collected(faces=tuple(ModelVector(model, shown[model]) for model in models))

    [naming] = name_people(store, [seen])

    assert (naming.match.name, naming.matched_by) == ("Ana", "face")
    # The match's distance is the raw cosine distance of the model that named her.
    assert naming.match.distance == pytest.approx(0.2, abs=1e-5)


def test_every_face_model_with_a_limit_is_matched():
    assert ARCFACE_MODEL in DEFAULT_IDENTITY_CONFIG.face_distance_limits
    spy = Spy()
    meet(spy.store, "Sam", "sam", appearances=False)
    # Sam is seen under the fake face model, then under ArcFace, as when walkie-vision
    # switches models: the fake_face samples still name him.
    seen = Collected(
        faces=(face("sam", 1), face("sam", 2), face("sam", 3, model=ARCFACE_MODEL))
    )

    [naming] = name_people(spy, [seen])

    assert (naming.match.name, naming.matched_by) == ("Sam", "face")
    [(_, queries, _)] = spy.calls
    assert [q.model for q in queries] == [FAKE_FACE_MODEL, ARCFACE_MODEL]


class Answers:
    """A store that answers every face query with the same matches."""

    def __init__(self, matches: list[Match], answers: int | None = None) -> None:
        self.matches = matches
        self.answers = answers

    def match_faces(self, queries, *, timeout_s=None):
        count = len(queries) if self.answers is None else self.answers
        return [list(self.matches) for _ in range(count)]

    def match_appearances(self, queries, *, timeout_s=None):
        return [[] for _ in queries]


def test_a_nan_distance_never_names_and_never_hides_a_real_match():
    nan = Match("stored_1", "Sam", {}, math.nan)
    real = Match("stored_1", "Sam", {}, 0.05)

    assert name_people(Answers([nan]), [person("sam")]) == [Naming(None, None)]
    [naming] = name_people(Answers([nan, real]), [person("sam")])
    assert naming.match == real


def test_a_store_that_answers_the_wrong_number_of_queries_is_a_bug():
    with pytest.raises(ValueError, match="answered 2 queries, not the 1 asked"):
        name_people(Answers([], answers=2), [person("sam")])


# -- name_people: the time ---------------------------------------------------------------


def test_each_call_gets_what_is_left_of_the_timeout():
    spy = Spy()
    meet(spy.store, "Sam", "sam")

    name_people(spy, [person("dave")], timeout_s=3.0)
    name_people(spy, [person("dave")])

    first, second, *untimed = [timeout for _, _, timeout in spy.calls]
    assert 0 < second <= first <= 3.0
    assert untimed == [None, None]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


def test_no_time_left_for_the_appearance_call_is_a_timeout(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(recognition, "time", clock)
    spy = Spy()
    meet(spy.store, "Sam", "sam")
    match_faces = spy.match_faces

    def slow(queries, *, timeout_s=None):
        clock.now += timeout_s
        return match_faces(queries, timeout_s=timeout_s)

    spy.match_faces = slow

    with pytest.raises(StepError) as error:
        name_people(spy, [person("dave")], timeout_s=2.0)

    assert error.value.code is ErrorCode.TIMEOUT
    assert error.value.message == (
        "Timed out: matching the people in view did not finish within 2 s."
    )
    assert spy.kinds() == ["faces"]


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, math.nan])
def test_no_time_at_all_is_a_timeout_only_when_a_call_is_needed(timeout_s):
    spy = Spy()

    with pytest.raises(StepError) as error:
        name_people(spy, [person("sam")], timeout_s=timeout_s)

    assert error.value.code is ErrorCode.TIMEOUT
    assert spy.calls == []
    assert name_people(spy, [person(None)], timeout_s=timeout_s) == [Naming(None, None)]
