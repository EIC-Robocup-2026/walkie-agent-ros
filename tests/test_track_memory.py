"""people_memory/track_memory.py: TrackMemory, ctx.memory over a track buffer and a people store.

The source is a small stand-in for RosVision: a TrackVectors fed with hash_vector
vectors under walkie-vision's fake model names, where person_<n> is track n, as on the
robot. The store is InMemoryPeopleMemory, the store of `--people-memory mock`. The
cases follow the experiment's remember_person and recognize_person tests in
tests/test_people_atomics.py, over live ids instead of handles.
"""

import inspect
import logging
import math
import re
from dataclasses import fields

import numpy as np
import pytest

from walkie_agent.capabilities import Memory, Recognized, Remembered
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.people_memory import store as store_module
from walkie_agent.people_memory import track_memory as memory_module
from walkie_agent.people_memory.store import (
    NOTHING_STORED,
    InMemoryPeopleMemory,
    ModelVector,
    Stored,
)
from walkie_agent.people_memory.track_memory import MAY_HAVE_STORED, TrackMemory, VectorSource
from walkie_agent.people_memory.track_vectors import Collected, Sighting, TrackVectors
from walkie_agent.sim.fake_vectors import FAKE_FACE_MODEL, FAKE_REID_MODEL, hash_vector

DIM = 64
NOT_IN_VIEW = "{} is not in view now. Call find_people again while they face you."


def vector(kind: str, key: str, shot: int = 0, weight: float = 0.1) -> np.ndarray:
    """A unit vector of one person; another shot is the same one nudged a little."""
    base = hash_vector(f"{kind} {key}", DIM)
    if shot == 0:
        return base
    nudged = base + weight * hash_vector(f"{kind} {key} shot {shot}", DIM)
    return (nudged / np.linalg.norm(nudged)).astype(np.float32)


def face(key: str, shot: int = 0, **kw) -> ModelVector:
    return ModelVector(FAKE_FACE_MODEL, vector("face", key, shot, **kw))


def appearance(key: str, shot: int = 0, **kw) -> ModelVector:
    return ModelVector(FAKE_REID_MODEL, vector("reid", key, shot, **kw))


class Vision:
    """A VectorSource over a TrackVectors, as RosVision is one: person_<n> is track n."""

    def __init__(self) -> None:
        self.buffer = TrackVectors()
        self.now = 0.0
        self.newest: list[int] = []
        self._stamp = 0
        self.reads: list[str] = []

    def show(self, *people: Sighting) -> None:
        """One people array, left to right, a tenth of a second after the last."""
        self._stamp += 1
        self.now += 0.1
        self.buffer.add(self._stamp, people, self.now)
        self.newest = [p.track_id for p in people if p.track_id]

    def collected(self, person_id: str) -> Collected | None:
        self.reads.append(person_id)
        found = re.fullmatch(r"person_([1-9][0-9]*)", person_id)
        if found is None or int(found[1]) not in self.buffer.tracks(self.now):
            return None
        return self.buffer.collected(int(found[1]), self.now)

    def tracked_in_view(self) -> list[str]:
        return [f"person_{track_id}" for track_id in self.newest]


def seen(track_id: int, key: str | None, shot: int = 1, *, faces=True, appearances=True, **kw):
    """`key` on track `track_id` in one image; None for a sighting without vectors."""
    return Sighting(
        track_id,
        face(key, shot, **kw) if key is not None and faces else None,
        appearance(key, shot, **kw) if key is not None and appearances else None,
    )


class Spy:
    """InMemoryPeopleMemory with a record of every call and a fault to raise."""

    def __init__(self) -> None:
        self.store = InMemoryPeopleMemory()
        self.calls: list[tuple[str, float | None]] = []
        self.fault: BaseException | None = None
        self.stored: list[tuple] = []

    def _call(self, name: str, timeout_s: float | None) -> None:
        self.calls.append((name, timeout_s))
        if self.fault is not None:
            raise self.fault

    def remember(self, name, facts, faces, appearances, *, timeout_s=None):
        self._call("remember", timeout_s)
        self.stored.append((name, dict(facts), list(faces), list(appearances)))
        return self.store.remember(name, facts, faces, appearances, timeout_s=timeout_s)

    def match_faces(self, queries, *, timeout_s=None):
        self._call("match_faces", timeout_s)
        return self.store.match_faces(queries, timeout_s=timeout_s)

    def match_appearances(self, queries, *, timeout_s=None):
        self._call("match_appearances", timeout_s)
        return self.store.match_appearances(queries, timeout_s=timeout_s)

    def clear(self, *, timeout_s=None):
        self.calls.append(("clear", timeout_s))
        self.store.clear(timeout_s=timeout_s)

    def __len__(self) -> int:
        return len(self.store)


@pytest.fixture
def vision():
    return Vision()


@pytest.fixture
def store():
    return Spy()


@pytest.fixture
def cancel():
    return CancelToken()


@pytest.fixture
def memory(vision, store, cancel):
    return TrackMemory(vision, store, cancel)


def meet(vision: Vision, memory: TrackMemory, track_id: int, key: str, name: str, facts=None):
    """Show `key` on `track_id` in three images, and remember them as `name`."""
    for shot in (1, 2, 3):
        vision.show(seen(track_id, key, shot))
    return memory.remember_person(f"person_{track_id}", name, facts)


def fails(call, *args, **kwargs) -> StepError:
    with pytest.raises(StepError) as error:
        call(*args, **kwargs)
    return error.value


# -- The protocol --------------------------------------------------------------------------


def test_track_memory_has_the_memory_verbs():
    for verb in ("remember_person", "recognize"):
        mine, protocol = (inspect.signature(getattr(cls, verb)) for cls in (TrackMemory, Memory))
        assert mine.parameters == protocol.parameters
        assert mine.return_annotation is protocol.return_annotation


def test_the_test_vision_is_a_vector_source():
    for method in ("collected", "tracked_in_view"):
        mine, protocol = (inspect.signature(getattr(cls, method)) for cls in (Vision, VectorSource))
        assert list(mine.parameters) == list(protocol.parameters)


# -- remember_person and recognize ------------------------------------------------------------


def test_remember_then_recognize_the_same_person_on_another_track(vision, memory, store):
    remembered = meet(vision, memory, 1, "sam", "  Sam ", {" drink ": "tea"})
    # Later, on another track and in other images.
    vision.show(seen(9, "sam", 4), seen(2, "ana", 1))
    vision.show(seen(9, "sam", 5))

    recognized = memory.recognize("person_9")

    assert remembered == Remembered("person_1", "Sam", {"drink": "tea"}, 3, 3, ())
    assert recognized == Recognized("person_9", "Sam", {"drink": "tea"}, "face")
    [(name, facts, faces, appearances)] = store.stored
    assert (name, facts, len(faces), len(appearances)) == ("Sam", {"drink": "tea"}, 3, 3)


def test_a_stranger_is_nobody(vision, memory):
    meet(vision, memory, 1, "sam", "Sam")
    vision.show(seen(2, "dave"))

    assert memory.recognize("person_2") == Recognized("person_2", None, {}, None)


def test_appearance_names_a_guest_whose_face_is_turned_away(vision, memory):
    meet(vision, memory, 1, "sam", "Sam")
    vision.show(seen(4, "sam", 7, faces=False))

    assert memory.recognize("person_4") == Recognized("person_4", "Sam", {}, "appearance")


def test_a_person_not_in_view_now_is_not_found(vision, memory, store):
    vision.show(seen(1, None), seen(0, "bob"))
    vision.show(seen(3, "ana"))

    for person_id in ("person_untracked_1", "person_77", "person_1", "person_0", "sam"):
        for error in (
            fails(memory.remember_person, person_id, "Bob"),
            fails(memory.recognize, person_id),
        ):
            assert error.code is ErrorCode.NOT_FOUND
            assert error.message == NOT_IN_VIEW.format(person_id)
    assert len(store) == 0 and store.calls == []


def test_a_track_unseen_for_a_minute_is_not_found(vision, memory, store):
    meet(vision, memory, 1, "sam", "Sam")
    vision.now += 61.0

    assert fails(memory.recognize, "person_1").code is ErrorCode.NOT_FOUND
    assert fails(memory.remember_person, "person_1", "Sam").code is ErrorCode.NOT_FOUND
    assert len(store) == 1


def test_a_tracked_person_out_of_the_newest_array_is_still_recognized(vision, memory):
    meet(vision, memory, 1, "sam", "Sam")
    vision.show(seen(2, "ana"))

    assert memory.recognize("person_1").name == "Sam"


def test_remember_stores_the_samples_without_outliers(vision, memory, store):
    # Five images of Sam's face, then one where a bystander's face was matched to
    # Sam's box: the bystander's face is an outlier.
    for shot in range(1, 6):
        vision.show(seen(1, "sam", shot))
    vision.show(Sighting(1, face("bob"), appearance("sam", 6)))

    remembered = memory.remember_person("person_1", "Sam")

    assert (remembered.faces, remembered.appearances) == (5, 6)
    [[nearest]] = store.store.match_faces([face("bob")])
    assert nearest.distance > 0.5


def test_remember_says_which_models_the_store_left_out(vision, memory, store, monkeypatch):
    remember = store.remember

    def without_appearances(name, facts, faces, appearances, *, timeout_s=None):
        stored = remember(name, facts, faces, [], timeout_s=timeout_s)
        return stored._replace(unstored_models=(FAKE_REID_MODEL,))

    monkeypatch.setattr(store, "remember", without_appearances)

    remembered = meet(vision, memory, 1, "sam", "Sam")

    assert remembered == Remembered("person_1", "Sam", {}, 3, 0, (FAKE_REID_MODEL,))


def test_recognize_names_the_person_together_with_the_others_in_view(vision, memory):
    meet(vision, memory, 1, "sam", "Sam")
    # Track 4 looks like Sam from afar; alone it would be named Sam.
    vision.show(seen(4, "sam", 7, weight=0.5))
    vision.show(seen(4, "sam", 8, weight=0.5))
    alone = memory.recognize("person_4")
    # Sam himself comes into view on track 3: the name goes to him only.
    vision.show(seen(3, "sam", 9), seen(4, "sam", 10, weight=0.5))

    assert alone.name == "Sam"
    assert memory.recognize("person_4") == Recognized("person_4", None, {}, None)
    assert memory.recognize("person_3").name == "Sam"
    # find_people's names agree with recognize.
    assert memory.names(["person_3", "person_4"], timeout_s=2.0) == {"person_3": "Sam"}


def test_two_guests_are_named_after_they_switch_places(vision, memory):
    meet(vision, memory, 1, "sam", "Sam", {"drink": "milk"})
    meet(vision, memory, 2, "ana", "Ana")
    vision.show(seen(5, "ana", 4), seen(6, "sam", 4))

    assert memory.recognize("person_5") == Recognized("person_5", "Ana", {}, "face")
    assert memory.recognize("person_6") == Recognized("person_6", "Sam", {"drink": "milk"}, "face")


def test_people_in_view_without_samples_are_left_out_of_the_batch(vision, memory, store):
    meet(vision, memory, 1, "sam", "Sam")
    vision.show(seen(1, "sam", 4), seen(2, None), seen(0, "bob"))
    store.calls.clear()

    assert memory.recognize("person_1").name == "Sam"
    # One face call for Sam alone; nothing for person_2, who has no samples.
    assert [name for name, _ in store.calls] == ["match_faces"]


def test_a_stored_person_with_a_blank_name_reads_as_nobody(vision, memory, store):
    store.store.remember("", {"drink": "tea"}, [face("sam")], [appearance("sam")])
    vision.show(seen(1, "sam"))

    assert memory.recognize("person_1") == Recognized("person_1", None, {}, None)
    assert memory.names(["person_1"], timeout_s=2.0) == {}


# -- Refusals and stops --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "facts"),
    [
        ("   ", None),
        ("S" * 41, None),
        ("Sam", {f"fact{i}": "yes" for i in range(9)}),
        ("Sam", {"age": 30}),
        ("Sam", {"drink": "m" * 81}),
        ("Sam", {" ": "tea"}),
    ],
)
def test_a_bad_name_or_fact_is_refused_and_stores_nobody(vision, memory, store, name, facts):
    vision.show(seen(1, "sam"))

    error = fails(memory.remember_person, "person_1", name, facts)

    assert error.code is ErrorCode.REFUSED
    assert len(store) == 0 and store.calls == []


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, math.nan, math.inf])
def test_a_bad_timeout_or_id_is_refused(vision, memory, timeout_s):
    vision.show(seen(1, "sam"))

    assert fails(memory.remember_person, "person_1", "Sam", timeout_s=timeout_s).code is (
        ErrorCode.REFUSED
    )
    assert fails(memory.recognize, "person_1", timeout_s=timeout_s).code is ErrorCode.REFUSED
    assert fails(memory.remember_person, 1, "Sam").code is ErrorCode.REFUSED
    assert fails(memory.recognize, None).code is ErrorCode.REFUSED


def test_a_stop_before_the_call_stores_nobody(vision, memory, store, cancel):
    vision.show(seen(1, "sam"))
    cancel.cancel()

    assert fails(memory.remember_person, "person_1", "Sam").code is ErrorCode.CANCELLED
    assert fails(memory.recognize, "person_1").code is ErrorCode.CANCELLED
    assert len(store) == 0 and store.calls == []


class Hooked:
    """A VectorSource that runs `hook` before each read of `vision`."""

    def __init__(self, vision: Vision, hook) -> None:
        self.vision = vision
        self.hook = hook

    def collected(self, person_id: str) -> Collected | None:
        self.hook()
        return self.vision.collected(person_id)

    def tracked_in_view(self) -> list[str]:
        return self.vision.tracked_in_view()


def test_a_stop_while_the_samples_are_read_stores_nobody(vision, store, cancel):
    vision.show(seen(1, "sam"))
    memory = TrackMemory(Hooked(vision, cancel.cancel), store, cancel)

    error = fails(memory.remember_person, "person_1", "Sam")

    assert error.code is ErrorCode.CANCELLED
    assert error.message == "Cancelled remembering person_1: a stop is in effect."
    assert len(store) == 0 and store.calls == []


def test_a_stop_while_the_mock_store_writes_stores_nobody(vision, cancel, monkeypatch):
    # The stop lands after TrackMemory's last check, while the mock copies the samples.
    # The mock's fence refuses the write, and the error keeps its "nothing stored".
    store = InMemoryPeopleMemory(cancel)
    memory = TrackMemory(vision, store, cancel)
    vision.show(seen(1, "sam"))
    add_samples = store_module._add_samples

    def stop_then_add(*args):
        cancel.cancel()
        add_samples(*args)

    monkeypatch.setattr(store_module, "_add_samples", stop_then_add)

    error = fails(memory.remember_person, "person_1", "Sam")

    assert error.code is ErrorCode.CANCELLED
    assert error.message == (
        f"Cancelled remembering a person: a stop is in effect. {NOTHING_STORED}"
    )
    assert len(store) == 0


def test_a_stop_during_the_matching_ends_recognize(vision, memory, store, cancel):
    meet(vision, memory, 1, "sam", "Sam")
    match_faces = store.match_faces

    def stopped(queries, *, timeout_s=None):
        cancel.cancel()
        return match_faces(queries, timeout_s=timeout_s)

    store.match_faces = stopped

    assert fails(memory.recognize, "person_1").code is ErrorCode.CANCELLED


# -- The store's failures and the time -----------------------------------------------------------


@pytest.mark.parametrize("code", [ErrorCode.TIMEOUT, ErrorCode.CANCELLED])
def test_a_remember_without_an_answer_may_have_stored_the_person(vision, memory, store, code):
    vision.show(seen(1, "sam"))
    store.fault = StepError(code, "The people memory did not answer within 7 s.")

    error = fails(memory.remember_person, "person_1", "Sam")

    assert error.code is code
    assert error.message == f"The people memory did not answer within 7 s. {MAY_HAVE_STORED}"
    # Said once, also when the store said it already.
    store.fault = StepError(code, f"No answer. {MAY_HAVE_STORED}")
    assert fails(memory.remember_person, "person_1", "Sam").message == f"No answer. {MAY_HAVE_STORED}"


@pytest.mark.parametrize("code", [ErrorCode.UNAVAILABLE, ErrorCode.REFUSED])
def test_other_store_failures_reach_the_caller_as_they_are(vision, memory, store, code):
    vision.show(seen(1, "sam"))
    store.fault = StepError(code, "The database could not store the person.")

    for call, args in ((memory.remember_person, ("person_1", "Sam")), (memory.recognize, ("person_1",))):
        error = fails(call, *args)
        assert (error.code, error.message) == (code, "The database could not store the person.")


def test_a_recognize_timeout_says_nothing_about_storing(vision, memory, store):
    vision.show(seen(1, "sam"))
    store.fault = StepError(ErrorCode.TIMEOUT, "No answer within 7 s.")

    assert fails(memory.recognize, "person_1").message == "No answer within 7 s."


def test_a_value_error_from_the_store_is_a_bug_and_propagates(vision, memory, store):
    vision.show(seen(1, "sam"))
    store.fault = ValueError("A fake_face vector must be non-empty, finite and not zero.")

    with pytest.raises(ValueError, match="non-empty"):
        memory.remember_person("person_1", "Sam")


def test_each_store_call_gets_what_is_left_of_the_verbs_timeout(vision, memory, store):
    meet(vision, memory, 1, "sam", "Sam")
    vision.show(seen(2, "dave"))
    store.calls.clear()

    memory.remember_person("person_2", "Dave", timeout_s=4.0)
    memory.recognize("person_1", timeout_s=3.0)
    memory.names(["person_2"], timeout_s=1.5)

    [remember, recognize, *names] = store.calls
    assert remember[0] == "remember" and 0 < remember[1] <= 4.0
    assert recognize[0] == "match_faces" and 0 < recognize[1] <= 3.0
    assert [name for name, _ in names] == ["match_faces"]
    assert all(0 < timeout <= 1.5 for _, timeout in names)


class Clock:
    def __init__(self) -> None:
        self.now = 500.0

    def monotonic(self) -> float:
        return self.now


def test_no_time_left_before_the_write_is_a_timeout_that_stored_nothing(
    vision, store, cancel, monkeypatch
):
    clock = Clock()
    monkeypatch.setattr(memory_module, "time", clock)
    vision.show(seen(1, "sam"))

    def slow() -> None:
        clock.now += 15.0

    memory = TrackMemory(Hooked(vision, slow), store, cancel)

    error = fails(memory.remember_person, "person_1", "Sam")

    assert error.code is ErrorCode.TIMEOUT
    assert error.message == (
        "Timed out: remembering person_1 did not finish within 15 s. Nothing was stored."
    )
    assert store.calls == []


# -- names ------------------------------------------------------------------------------------------


def test_names_gives_the_names_of_the_remembered_people_in_view_only(vision, memory, store):
    meet(vision, memory, 1, "sam", "Sam")
    meet(vision, memory, 2, "ana", "Ana")
    vision.show(seen(3, "dave"), seen(1, "sam", 4), seen(2, None), seen(0, "bob"))
    store.calls.clear()
    vision.reads.clear()

    named = memory.names(
        ["person_3", "person_1", "person_1", "person_untracked_1", "person_9"], timeout_s=2.0
    )

    assert named == {"person_1": "Sam"}
    # One call per kind for everyone: Sam's face named him, Dave's matched nobody.
    assert [name for name, _ in store.calls] == ["match_faces", "match_appearances"]
    # An id given twice is read and named once.
    assert vision.reads == ["person_3", "person_1", "person_untracked_1", "person_9"]


def test_names_makes_no_call_when_nobody_has_samples(vision, memory, store):
    vision.show(seen(1, None))

    assert memory.names(["person_1", "person_untracked_1"], timeout_s=2.0) == {}
    assert memory.names([], timeout_s=2.0) == {}
    assert store.calls == []


@pytest.mark.parametrize(
    "fault",
    [
        StepError(ErrorCode.UNAVAILABLE, "The database could not answer."),
        StepError(ErrorCode.REFUSED, "Unknown model."),
        StepError(ErrorCode.TIMEOUT, "No answer within 2 s."),
    ],
)
def test_a_memory_failure_leaves_the_names_out_with_one_warning(
    vision, memory, store, fault, caplog
):
    meet(vision, memory, 1, "sam", "Sam")
    vision.show(seen(1, "sam", 4), seen(2, "dave"))
    store.fault = fault

    with caplog.at_level(logging.WARNING, logger=memory_module.__name__):
        named = memory.names(["person_1", "person_2"], timeout_s=2.0)

    assert named == {}
    [warning] = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning.getMessage() == f"find_people shows no names: {fault}"


def test_names_without_time_left_gives_no_names(vision, memory, store, caplog):
    meet(vision, memory, 1, "sam", "Sam")
    store.calls.clear()

    with caplog.at_level(logging.WARNING, logger=memory_module.__name__):
        assert memory.names(["person_1"], timeout_s=0.0) == {}

    assert store.calls == []
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


def test_a_stop_propagates_out_of_names(vision, memory, store, cancel):
    meet(vision, memory, 1, "sam", "Sam")
    store.fault = StepError(ErrorCode.CANCELLED, "Cancelled matching: a stop is in effect.")

    assert fails(memory.names, ["person_1"], timeout_s=2.0).code is ErrorCode.CANCELLED
    store.fault = None
    cancel.cancel()
    assert fails(memory.names, ["person_1"], timeout_s=2.0).code is ErrorCode.CANCELLED


# -- clear, records and logs -----------------------------------------------------------------------


def test_clear_forgets_the_people_remembered(vision, memory, store):
    meet(vision, memory, 1, "sam", "Sam")

    memory.clear()

    assert store.calls[-1] == ("clear", None)
    assert len(store) == 0
    assert memory.recognize("person_1") == Recognized("person_1", None, {}, None)
    assert memory.names(["person_1"], timeout_s=2.0) == {}


def test_no_vector_or_stored_id_leaves_the_backend(vision, memory, caplog):
    with caplog.at_level(logging.DEBUG, logger=memory_module.__name__):
        remembered = meet(vision, memory, 1, "sam", "Sam", {"drink": "tea"})
        vision.show(seen(2, "sam", 4))
        recognized = memory.recognize("person_2")
        named = memory.names(["person_2"], timeout_s=2.0)

    stored_id = re.compile(r"\bstored_\d")
    for record in (remembered, recognized):
        for field in fields(record):
            value = getattr(record, field.name)
            assert not isinstance(value, np.ndarray | ModelVector | Stored)
        assert not stored_id.search(repr(record))
    assert not stored_id.search(repr(named))
    mine = [r for r in caplog.records if r.name == memory_module.__name__]
    loud = [r.getMessage() for r in mine if r.levelno >= logging.INFO]
    quiet = [r.getMessage() for r in mine if r.levelno < logging.INFO]
    assert loud == [
        "Remembered person_1 as Sam from 3 face and 3 appearance samples.",
        "Recognized person_2 as Sam by face.",
    ]
    # The store's id is for debugging only.
    assert any(stored_id.search(line) for line in quiet)
