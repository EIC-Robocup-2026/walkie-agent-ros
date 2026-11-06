"""TrackMemory: ctx.memory over the samples of each track in view and a people store.

ctx.memory names people by the live ids find_people gives (ADR-0070). TrackMemory reads
a person's face and appearance samples from a VectorSource, which RosVision is on the
robot and in simulation, and stores or matches them in a PeopleMemory: RosPeopleMemory
over walkie-database, or InMemoryPeopleMemory with `--people-memory mock`. No ROS.

- Only live ids reach the model. The store's own ids, such as `stored_3`, appear in
  DEBUG lines only, and no vector leaves this module in a record or a log line.
- A person counts as in view while their track has samples: an untracked person, or a
  track with no sample in the last 60 s, is NOT_FOUND.
- remember_person stores the track's samples without each model's outliers, under the
  name and facts that guards.require_person_details cleaned.
- recognize names the person together with the other tracked people of the newest
  people array, with the rules of recognition.name_people, so recognize and
  find_people agree and one stored person names one person in view at most. names()
  is the same naming for find_people, which shows the names.
- Each store call gets what is left of the verb's `timeout_s` as its own.
"""

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Protocol

from walkie_agent.capabilities.guards import (
    CancelToken,
    require_person_details,
    require_text,
    require_timeout,
)
from walkie_agent.capabilities.records import Recognized, Remembered
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.perception.identity import DEFAULT_IDENTITY_CONFIG, IdentityConfig
from walkie_agent.perception.people_memory import NOTHING_STORED, PeopleMemory
from walkie_agent.perception.recognition import name_people, without_outliers
from walkie_agent.perception.track_vectors import Collected

logger = logging.getLogger(__name__)

# Added to remember_person's TIMEOUT and CANCELLED from the store, whose request may
# have reached it: every remember stores a new person, so a second one may store two.
MAY_HAVE_STORED = (
    "The person may have been stored already, so remembering them again may store "
    "them twice."
)


class VectorSource(Protocol):
    """Where TrackMemory reads the samples of the people in view: RosVision on the robot."""

    def collected(self, person_id: str) -> Collected | None:
        """The samples of the live id `person_id`; None when it is not a tracked person now.

        An empty Collected is a tracked person with no usable sample.
        """
        ...

    def tracked_in_view(self) -> list[str]:
        """The live ids of the tracked people in the newest people array, left to right."""
        ...


class TrackMemory:
    """ctx.memory over `source`'s samples and `store`, a PeopleMemory.

    The verbs hold `cancel`, the CancelToken of the Ctx they serve. `identity` holds
    the outlier rule and the distance limits of each model. Thread-safe as long as
    `source` and `store` are: it keeps no state of its own.
    """

    def __init__(
        self,
        source: VectorSource,
        store: PeopleMemory,
        cancel: CancelToken,
        *,
        identity: IdentityConfig = DEFAULT_IDENTITY_CONFIG,
    ) -> None:
        self._source = source
        self._store = store
        self._cancel = cancel
        self._identity = identity

    def remember_person(
        self,
        person_id: str,
        name: str,
        facts: Mapping[str, str] | None = None,
        *,
        timeout_s: float = 15.0,
    ) -> Remembered:
        """Store the person on `person_id`'s track under `name`, with `facts`.

        Raises StepError REFUSED for a bad argument; NOT_FOUND for a person not in
        view now; CANCELLED on a stop, also one that came while the samples were read,
        so a stop never stores anyone; and the store's errors. A TIMEOUT or CANCELLED
        from the store says that the person may have been stored already, unless the
        store says that nothing was. A
        ValueError from the store is a bug and propagates.
        """
        require_timeout(timeout_s)
        deadline = time.monotonic() + timeout_s
        require_text(person_id=person_id)
        name, kept_facts = require_person_details(name, facts)
        what = f"remembering {person_id}"
        self._cancel.check(what)
        collected = self._samples(person_id)
        faces = without_outliers(collected.faces, self._identity)
        appearances = without_outliers(collected.appearances, self._identity)
        # Again just before the write: the guest's answer may come long after the
        # question, and a stop since then must not store them.
        self._cancel.check(what)
        left = _left(deadline, timeout_s, what, f" {NOTHING_STORED}")
        try:
            stored = self._store.remember(
                name, kept_facts, faces, appearances, timeout_s=left
            )
        except StepError as error:
            # A store that already says whether anything was stored is left as it is.
            if error.code in (ErrorCode.TIMEOUT, ErrorCode.CANCELLED) and not (
                "may have been stored" in error.message or error.message.endswith(NOTHING_STORED)
            ):
                raise StepError(error.code, f"{error.message} {MAY_HAVE_STORED}") from error
            raise
        logger.info(
            "Remembered %s as %s from %d face and %d appearance samples.",
            person_id,
            name,
            stored.faces,
            stored.appearances,
        )
        logger.debug("%s is stored as %s.", person_id, stored.person_id)
        return Remembered(
            person_id,
            name,
            dict(kept_facts),
            stored.faces,
            stored.appearances,
            tuple(stored.unstored_models),
        )

    def recognize(self, person_id: str, *, timeout_s: float = 15.0) -> Recognized:
        """Who the person on `person_id`'s track is, among the people remembered.

        The person is named together with the other tracked people in view that have
        samples, and the answer is theirs alone. Someone not recognized gets
        Recognized(person_id, None, {}, None). Raises StepError REFUSED for a bad
        argument, NOT_FOUND for a person not in view now, CANCELLED on a stop,
        TIMEOUT when no time is left for a store call, and the store's errors.
        """
        require_timeout(timeout_s)
        deadline = time.monotonic() + timeout_s
        require_text(person_id=person_id)
        what = f"recognizing {person_id}"
        self._cancel.check(what)
        people = [self._samples(person_id)]
        for other in dict.fromkeys(self._source.tracked_in_view()):
            if other == person_id:
                continue
            collected = self._source.collected(other)
            if collected:
                people.append(collected)
        namings = name_people(
            self._store,
            people,
            self._identity,
            timeout_s=_left(deadline, timeout_s, what),
        )
        # A stop during the matching ends the step rather than answering it.
        self._cancel.check(what)
        match, matched_by = namings[0]
        if match is None:
            logger.info("%s is nobody remembered.", person_id)
            return Recognized(person_id, None, {}, None)
        logger.info("Recognized %s as %s by %s.", person_id, match.name, matched_by)
        logger.debug(
            "%s is %s at a distance of %.3f.", person_id, match.person_id, match.distance
        )
        return Recognized(person_id, match.name, dict(match.facts), matched_by)

    def names(self, person_ids: Sequence[str], *, timeout_s: float) -> dict[str, str]:
        """The names that find_people shows for `person_ids`, live ids in view.

        Only the ids with samples and a stored person with a name are keys, and the
        ids are named together, as recognize names them. A memory failure gives {}
        and one WARNING line, so find_people still answers without names. Raises
        StepError CANCELLED on a stop, also one from the store.
        """
        what = "naming the people in view"
        self._cancel.check(what)
        ids: list[str] = []
        people: list[Collected] = []
        for person_id in dict.fromkeys(person_ids):
            collected = self._source.collected(person_id)
            if collected:
                ids.append(person_id)
                people.append(collected)
        if not people:
            return {}
        try:
            require_timeout(timeout_s)
            namings = name_people(self._store, people, self._identity, timeout_s=timeout_s)
        except StepError as error:
            if error.code is ErrorCode.CANCELLED:
                raise
            logger.warning("find_people shows no names: %s", error)
            return {}
        self._cancel.check(what)
        named = {
            person_id: naming.match.name
            for person_id, naming in zip(ids, namings, strict=True)
            if naming.match is not None
        }
        logger.debug("Named %s.", named)
        return named

    def clear(self) -> None:
        """Forget the people remembered, as at a new task: the store's clear().

        Never raises StepError, and waits no longer than the store's clear() does.
        """
        self._store.clear()

    def _samples(self, person_id: str) -> Collected:
        """The samples of `person_id`, or StepError NOT_FOUND when it has none now."""
        collected = self._source.collected(person_id)
        if not collected:
            raise StepError(
                ErrorCode.NOT_FOUND,
                f"{person_id} is not in view now. Call find_people again while they face you.",
            )
        return collected


def _left(deadline: float, timeout_s: float, what: str, outcome: str = "") -> float:
    """What is left of the verb's `timeout_s` at `deadline`; StepError TIMEOUT when nothing is."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise StepError(
            ErrorCode.TIMEOUT,
            f"Timed out: {what} did not finish within {timeout_s:g} s.{outcome}",
        )
    return left
