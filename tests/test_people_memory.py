"""perception/people_memory.py: InMemoryPeopleMemory, the store of `--people-memory mock`.

Ported from the experiment's tests/test_vision_feed.py, with hash_vector vectors under
walkie-vision's fake model names, as oracle_perception publishes them.
"""

import threading

import numpy as np
import pytest

from walkie_agent.perception.people_memory import (
    InMemoryPeopleMemory,
    Match,
    ModelVector,
    PeopleMemory,
    Stored,
)
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, FAKE_REID_MODEL, hash_vector


def face(key: str, dim: int = 512) -> ModelVector:
    return ModelVector(FAKE_FACE_MODEL, hash_vector(f"face {key}", dim))


def appearance(key: str, dim: int = 512) -> ModelVector:
    return ModelVector(FAKE_REID_MODEL, hash_vector(f"appearance {key}", dim))


def near(vector: ModelVector, key: str, weight: float = 0.1) -> ModelVector:
    """A second shot of the same person: the vector nudged towards another."""
    nudged = vector.vector + weight * hash_vector(f"noise {key}", vector.vector.size)
    return ModelVector(vector.model, nudged / np.linalg.norm(nudged))


def test_the_mock_is_a_people_memory():
    memory: PeopleMemory = InMemoryPeopleMemory()
    assert memory.match_faces([]) == []


def test_memory_matches_faces_nearest_first_within_one_model():
    memory = InMemoryPeopleMemory()
    stored = memory.remember("Sam", {"drink": "milk"}, [face("sam")], [appearance("sam")])
    sam = stored.person_id
    ana = memory.remember("Ana", {}, [face("ana")], []).person_id

    # The mock takes every model, so it stores every vector.
    assert stored == Stored(sam, 1, 1, ())
    [matches] = memory.match_faces([near(face("sam"), "1")])

    assert [m.person_id for m in matches] == [sam, ana]
    assert (matches[0].name, dict(matches[0].facts)) == ("Sam", {"drink": "milk"})
    assert matches[0].distance < 0.05
    assert matches[1].distance > 0.8
    # Another model's vectors are never compared, so nobody matches them.
    other_model = ModelVector("arcface_w600k_r50", face("sam").vector)
    assert memory.match_faces([other_model]) == [[]]


def test_ids_are_stored_n_from_1():
    memory = InMemoryPeopleMemory()
    first = memory.remember("Sam", {}, [face("sam")], [])
    second = memory.remember("Ana", {}, [], [appearance("ana")])

    assert (first.person_id, second.person_id) == ("stored_1", "stored_2")
    [[match]] = memory.match_faces([face("sam")])
    assert match == Match("stored_1", "Sam", {}, pytest.approx(0.0, abs=1e-6))


def test_remember_counts_every_vector_it_stored():
    memory = InMemoryPeopleMemory()
    faces = [face("sam 1"), face("sam 2")]
    appearances = [appearance("sam 1"), appearance("sam 2"), appearance("sam 3")]
    stored = memory.remember("Sam", {}, faces, appearances)
    assert (stored.faces, stored.appearances, stored.unstored_models) == (2, 3, ())


def test_memory_keeps_faces_and_appearances_apart():
    memory = InMemoryPeopleMemory()
    sam = memory.remember("Sam", {}, [], [appearance("sam")]).person_id

    assert memory.match_faces([face("sam")]) == [[]]
    [[match]] = memory.match_appearances([appearance("sam")])
    assert match.person_id == sam
    assert match.distance == pytest.approx(0.0, abs=1e-6)


def test_memory_answers_each_query_and_uses_the_nearest_sample():
    memory = InMemoryPeopleMemory()
    # Two samples, as from two angles: the query is near the second only.
    sam = memory.remember("Sam", {}, [face("sam front"), face("sam side")], []).person_id
    ana = memory.remember("Ana", {}, [face("ana")], []).person_id

    sam_matches, ana_matches = memory.match_faces([face("sam side"), face("ana")])

    assert sam_matches[0].person_id == sam
    assert sam_matches[0].distance == pytest.approx(0.0, abs=1e-6)
    assert ana_matches[0].person_id == ana


def test_a_person_with_samples_of_two_models_matches_a_query_of_either():
    memory = InMemoryPeopleMemory()
    other = ModelVector("arcface_w600k_r50", hash_vector("arcface sam", 8))
    sam = memory.remember("Sam", {}, [face("sam"), other], []).person_id

    fake_matches, other_matches = memory.match_faces([face("sam"), other])

    assert [m.person_id for m in fake_matches] == [sam]
    assert [m.person_id for m in other_matches] == [sam]


def test_an_empty_query_list_gets_no_answers():
    memory = InMemoryPeopleMemory()
    memory.remember("Sam", {}, [face("sam")], [appearance("sam")])
    assert memory.match_faces([]) == []
    assert memory.match_appearances([]) == []


def test_nobody_stored_gives_an_empty_list_per_query():
    memory = InMemoryPeopleMemory()
    assert memory.match_faces([face("sam"), face("ana")]) == [[], []]


def test_a_blank_name_is_returned_as_it_was_stored():
    # The caller reads a blank name as unknown; the store passes it on.
    memory = InMemoryPeopleMemory()
    memory.remember("", {}, [face("someone")], [])
    [[match]] = memory.match_faces([face("someone")])
    assert match.name == ""


def test_every_method_takes_a_timeout_and_the_mock_ignores_it():
    memory = InMemoryPeopleMemory()
    stored = memory.remember("Sam", {}, [face("sam")], [appearance("sam")], timeout_s=0.0)

    [[by_face]] = memory.match_faces([face("sam")], timeout_s=0.0)
    [[by_appearance]] = memory.match_appearances([appearance("sam")], timeout_s=None)
    assert by_face.person_id == by_appearance.person_id == stored.person_id

    memory.clear(timeout_s=0.0)
    assert len(memory) == 0


def test_memory_clear_forgets_everyone_and_never_reuses_an_id():
    memory = InMemoryPeopleMemory()
    first = memory.remember("Sam", {}, [face("sam")], []).person_id

    memory.clear()

    assert len(memory) == 0
    assert memory.match_faces([face("sam")]) == [[]]
    assert memory.remember("Ana", {}, [face("ana")], []).person_id == "stored_2" != first


def test_memory_returns_copies_of_the_facts():
    memory = InMemoryPeopleMemory()
    facts = {"drink": "milk"}
    memory.remember("Sam", facts, [face("sam")], [])
    facts["drink"] = "tea"

    [[match]] = memory.match_faces([face("sam")])
    match.facts["drink"] = "coffee"

    [[again]] = memory.match_faces([face("sam")])
    assert again.facts == {"drink": "milk"}


def test_memory_keeps_a_copy_of_each_vector():
    # A caller that reuses its array cannot change a stored sample.
    memory = InMemoryPeopleMemory()
    sample = face("sam")
    memory.remember("Sam", {}, [sample], [])
    original = sample.vector.copy()
    sample.vector[:] = hash_vector("face ana", 512)

    [[match]] = memory.match_faces([ModelVector(FAKE_FACE_MODEL, original)])
    assert match.distance == pytest.approx(0.0, abs=1e-6)


def test_memory_reads_a_vector_of_any_float_type_or_shape():
    memory = InMemoryPeopleMemory()
    vector = hash_vector("face sam", 8).astype(np.float64)
    memory.remember("Sam", {}, [ModelVector(FAKE_FACE_MODEL, vector.reshape(2, 4))], [])

    [[match]] = memory.match_faces([ModelVector(FAKE_FACE_MODEL, list(vector))])
    assert match.distance == pytest.approx(0.0, abs=1e-6)


def test_memory_rejects_a_person_without_vectors_and_a_wrong_length():
    memory = InMemoryPeopleMemory()
    with pytest.raises(ValueError, match="without a face or appearance"):
        memory.remember("Sam", {}, [], [])
    with pytest.raises(ValueError, match="not zero"):
        memory.remember("Sam", {}, [ModelVector(FAKE_FACE_MODEL, np.zeros(4))], [])
    memory.remember("Sam", {}, [face("sam")], [])
    with pytest.raises(ValueError, match="512"):
        memory.match_faces([face("sam", dim=8)])


@pytest.mark.parametrize(
    "values", [np.array([], dtype=np.float32), np.array([1.0, np.nan]), np.array([np.inf, 0.0])]
)
def test_memory_rejects_an_empty_or_non_finite_vector(values):
    memory = InMemoryPeopleMemory()
    with pytest.raises(ValueError, match="non-empty, finite and not zero"):
        memory.remember("Sam", {}, [], [ModelVector(FAKE_REID_MODEL, values)])
    with pytest.raises(ValueError, match="non-empty, finite and not zero"):
        memory.match_appearances([ModelVector(FAKE_REID_MODEL, values)])


def test_a_refused_remember_stores_nobody_and_uses_no_id():
    memory = InMemoryPeopleMemory()
    with pytest.raises(ValueError):
        memory.remember("Sam", {}, [face("sam")], [ModelVector(FAKE_REID_MODEL, np.zeros(4))])

    assert len(memory) == 0
    assert memory.remember("Ana", {}, [face("ana")], []).person_id == "stored_1"


def test_remembers_from_many_threads_get_distinct_ids():
    memory = InMemoryPeopleMemory()
    ids: list[str] = []
    lock = threading.Lock()

    def remember(n: int) -> None:
        stored = memory.remember(f"Guest {n}", {}, [face(f"guest {n}")], [])
        with lock:
            ids.append(stored.person_id)

    threads = [threading.Thread(target=remember, args=(n,)) for n in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(ids) == sorted(f"stored_{n}" for n in range(1, 17))
    assert len(memory) == 16
