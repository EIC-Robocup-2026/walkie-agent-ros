"""Handles: short ids that stand for things a tool saw, such as people or seats.

A perception atomic stores what it found as a set and gives the LLM ids
like `per_51c0.1`. A later call, in the same turn or after a spoken
question, passes the id back, and the store returns the payload if it is
still valid. A handle is valid while it is young enough and the base has
not moved too far since it was captured, because the payload describes
the scene as the robot saw it from there (self-evolving-agent.md,
Handles).
"""

import math
import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from walkie_agent.toolbox.types import StepError


@dataclass(frozen=True)
class Pose2D:
    """The base's pose in the map frame: metres, and yaw in radians."""

    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class KindPolicy:
    """How handles of one kind are named and when they go stale.

    A limit of None is not checked. `source` names where such handles come
    from, such as "find_people", and is added to NOT_FOUND and STALE
    messages so the LLM knows how to get a new one.
    """

    prefix: str
    max_translation_m: float | None
    max_rotation_deg: float | None
    max_age_s: float | None
    source: str = ""

    @property
    def checks_motion(self) -> bool:
        return self.max_translation_m is not None or self.max_rotation_deg is not None


# The default for objects: 5 cm or 3 degrees of base motion, or 10 minutes,
# which leaves time for a spoken clarification. No object atomic ships yet,
# so no source is named.
OBJECT_POLICY = KindPolicy(
    prefix="obj", max_translation_m=0.05, max_rotation_deg=3.0, max_age_s=600.0
)

_KIND = re.compile(r"^[a-z]+( [a-z]+)*$")
_PREFIX = re.compile(r"^[a-z]{2,8}$")
# At most this many members per set, far more than a camera image holds.
# It bounds the index in an id, so int() never sees a number too long to
# convert and a malformed id is NOT_FOUND, never a ValueError.
MAX_SET_MEMBERS = 1000
_HANDLE_ID = re.compile(r"^(?P<set>[a-z]{2,8}_[0-9a-f]{4})\.(?P<index>\d{1,3})$")
_SET_ID = re.compile(r"^[a-z]{2,8}_[0-9a-f]{4}$")
# How many random ids put_set tries before it gives up. With at most a few
# hundred live sets among 65536 ids, a second try is already rare.
_ID_ATTEMPTS = 100

_kinds_lock = threading.Lock()
_kinds: dict[str, KindPolicy] = {"object": OBJECT_POLICY}


def register_kind(kind: str, policy: KindPolicy) -> None:
    """Register a handle kind, such as "person", with its policy.

    Normally called when an atomic's module is imported. Registering the same
    kind again with the same policy does nothing; a different policy, or a
    prefix that another kind uses, raises ValueError.
    """
    # The kind goes into sentences for the LLM, such as "is a seat, not a
    # person", so it is a plain lowercase noun.
    if not _KIND.match(kind):
        raise ValueError(f"A handle kind is a lowercase noun, such as 'person', not {kind!r}.")
    if not _PREFIX.match(policy.prefix):
        raise ValueError(f"A handle prefix is 2 to 8 lowercase letters, not {policy.prefix!r}.")
    with _kinds_lock:
        current = _kinds.get(kind)
        if current is not None:
            if current != policy:
                raise ValueError(f"Handle kind {kind!r} is already registered with {current}.")
            return
        for other, other_policy in _kinds.items():
            if other_policy.prefix == policy.prefix:
                raise ValueError(f"Handle prefix {policy.prefix!r} is already used by {other!r}.")
        _kinds[kind] = policy


def kind_policy(kind: str) -> KindPolicy:
    """The policy of a registered kind. Raises ValueError for an unknown kind."""
    with _kinds_lock:
        policy = _kinds.get(kind)
    if policy is None:
        raise ValueError(f"Unknown handle kind {kind!r}; register it with register_kind().")
    return policy


@dataclass(frozen=True)
class Handle:
    """One stored member of a set."""

    id: str
    kind: str
    payload: Any
    pose: Pose2D | None
    captured_at: float


@dataclass(frozen=True)
class HandleSet:
    """What put_set returns: the set's id and its member ids, in payload order."""

    id: str
    kind: str
    member_ids: tuple[str, ...]


@dataclass(frozen=True)
class _StoredSet:
    kind: str
    payloads: tuple[Any, ...]
    pose: Pose2D | None
    captured_at: float


class HandleStore:
    """Keeps handle sets with the robot pose at capture, and checks them on lookup.

    Thread-safe: one lock guards the sets. The pose provider is never called
    while the lock is held, because it may wait for TF.

    `pose_provider` returns the robot's pose now, or None when it is unknown.
    `id_source` returns a number below 0x10000 for the hex part of a new set
    id; tests pass a fixed sequence. At most `max_sets` sets are kept, and the
    oldest is dropped first.
    """

    def __init__(
        self,
        pose_provider: Callable[[], Pose2D | None],
        clock: Callable[[], float] = time.monotonic,
        max_sets: int = 64,
        id_source: Callable[[], int] | None = None,
    ) -> None:
        if max_sets < 1:
            raise ValueError("max_sets must be at least 1.")
        self._pose_provider = pose_provider
        self._clock = clock
        self._max_sets = max_sets
        self._id_source = id_source or (lambda: secrets.randbelow(0x10000))
        self._lock = threading.Lock()
        # By set id, oldest first.
        self._sets: OrderedDict[str, _StoredSet] = OrderedDict()

    def put_set(
        self, kind: str, payloads: Sequence[Any], pose: Pose2D | None = None
    ) -> HandleSet:
        """Store `payloads` as one set captured now, and return its ids.

        With no `pose`, the store asks the pose provider; None is stored when
        the pose is unknown, and a kind with motion limits then reports
        UNAVAILABLE on lookup. More than MAX_SET_MEMBERS payloads raise
        ValueError, because their ids could not be looked up.
        """
        policy = kind_policy(kind)
        # Copied first and the copy checked, so a caller's list that changes
        # while the pose provider waits cannot grow past the limit.
        members = tuple(payloads)
        if len(members) > MAX_SET_MEMBERS:
            raise ValueError(
                f"A handle set holds at most {MAX_SET_MEMBERS} members, not {len(members)}."
            )
        if pose is None:
            pose = self._pose_provider()
        stored = _StoredSet(kind=kind, payloads=members, pose=pose, captured_at=self._clock())
        with self._lock:
            set_id = self._new_set_id(policy.prefix)
            self._sets[set_id] = stored
            while len(self._sets) > self._max_sets:
                self._sets.popitem(last=False)
        member_ids = tuple(f"{set_id}.{index}" for index in range(len(stored.payloads)))
        return HandleSet(id=set_id, kind=kind, member_ids=member_ids)

    def get(self, handle_id: str, kind: str | None = None) -> Handle:
        """The member `handle_id`, checked for kind, age and base motion.

        Raises StepError: NOT_FOUND for an unknown or malformed id or one of
        another kind than `kind`, STALE when the handle is too old or the
        base has moved past a limit, and UNAVAILABLE when a motion limit
        applies but the robot's pose is unknown.
        """
        handle_id = handle_id.strip()
        match = _HANDLE_ID.match(handle_id)
        stored = self._lookup(match.group("set") if match else None)
        if match is None or stored is None:
            raise _unknown(handle_id, kind)
        index = int(match.group("index"))
        if index >= len(stored.payloads):
            raise _unknown(handle_id, kind)
        self._check(handle_id, stored, kind)
        return _member(match.group("set"), index, stored)

    def get_set(self, set_id: str, kind: str | None = None) -> list[Handle]:
        """Every member of the set `set_id`, in order, with the checks of get()."""
        set_id = set_id.strip()
        stored = self._lookup(set_id if _SET_ID.match(set_id) else None)
        if stored is None:
            raise _unknown(set_id, kind)
        self._check(set_id, stored, kind)
        return [_member(set_id, index, stored) for index in range(len(stored.payloads))]

    def peek(self, handle_or_set_id: str) -> list[Handle]:
        """The handle, or every member of the set, with no check at all.

        For records, not for tools: a stale handle is still returned, and the
        pose provider is never asked, so a caller never waits on TF. An
        unknown or malformed id gives [].
        """
        handle_or_set_id = handle_or_set_id.strip()
        match = _HANDLE_ID.match(handle_or_set_id)
        if match is not None:
            stored = self._lookup(match.group("set"))
            index = int(match.group("index"))
            if stored is None or index >= len(stored.payloads):
                return []
            return [_member(match.group("set"), index, stored)]
        if not _SET_ID.match(handle_or_set_id):
            return []
        stored = self._lookup(handle_or_set_id)
        if stored is None:
            return []
        return [_member(handle_or_set_id, index, stored) for index in range(len(stored.payloads))]

    def clear(self) -> None:
        """Forget every set."""
        with self._lock:
            self._sets.clear()

    def _lookup(self, set_id: str | None) -> _StoredSet | None:
        if set_id is None:
            return None
        with self._lock:
            return self._sets.get(set_id)

    def _new_set_id(self, prefix: str) -> str:
        # Called with the lock held. The hex part is unique among all live
        # sets, whatever their kind, so no two live ids differ only in prefix.
        taken = {set_id.rpartition("_")[2] for set_id in self._sets}
        for _ in range(_ID_ATTEMPTS):
            digits = f"{self._id_source() % 0x10000:04x}"
            if digits not in taken:
                return f"{prefix}_{digits}"
        raise RuntimeError(f"No free handle id after {_ID_ATTEMPTS} attempts.")

    def _check(self, handle_id: str, stored: _StoredSet, kind: str | None) -> None:
        if kind is not None and stored.kind != kind:
            message = f"{handle_id} is {_a(stored.kind)}, not {_a(kind)}."
            raise StepError("NOT_FOUND", message + _source_hint(kind), {"handle": handle_id})
        policy = kind_policy(stored.kind)
        age = self._clock() - stored.captured_at
        if policy.max_age_s is not None and age > policy.max_age_s:
            raise StepError(
                "STALE",
                f"{handle_id} is stale: it was captured {age:.0f} s ago, and "
                f"{_a(stored.kind)} handle lasts {policy.max_age_s:g} s."
                + _source_hint(stored.kind),
                {"handle": handle_id, "age_s": age},
            )
        if not policy.checks_motion:
            return
        # Outside the lock: the provider may wait for TF, so it is not asked
        # when the capture pose is unknown and the answer is UNAVAILABLE anyway.
        now = self._pose_provider() if stored.pose is not None else None
        if now is None or stored.pose is None:
            raise StepError(
                "UNAVAILABLE",
                "The robot's position is unknown, so the handle cannot be checked.",
                {"handle": handle_id},
            )
        moved_m = math.hypot(now.x - stored.pose.x, now.y - stored.pose.y)
        if policy.max_translation_m is not None and moved_m > policy.max_translation_m:
            raise StepError(
                "STALE",
                f"{handle_id} is stale: the robot has moved {moved_m:.2f} m since it was "
                f"captured, more than the {policy.max_translation_m:g} m allowed."
                + _source_hint(stored.kind),
                {"handle": handle_id, "moved_m": moved_m},
            )
        # Compared with wrap-around, so 179 and -179 degrees are 2 degrees apart.
        turned = now.yaw - stored.pose.yaw
        turned_deg = abs(math.degrees(math.atan2(math.sin(turned), math.cos(turned))))
        if policy.max_rotation_deg is not None and turned_deg > policy.max_rotation_deg:
            raise StepError(
                "STALE",
                f"{handle_id} is stale: the robot has turned {turned_deg:.0f} degrees since "
                f"it was captured, more than the {policy.max_rotation_deg:g} degrees allowed."
                + _source_hint(stored.kind),
                {"handle": handle_id, "turned_deg": turned_deg},
            )


def _member(set_id: str, index: int, stored: _StoredSet) -> Handle:
    return Handle(
        id=f"{set_id}.{index}",
        kind=stored.kind,
        payload=stored.payloads[index],
        pose=stored.pose,
        captured_at=stored.captured_at,
    )


def _unknown(handle_id: str, kind: str | None) -> StepError:
    if kind is None:
        return StepError("NOT_FOUND", f"{handle_id} is not a known handle.", {"handle": handle_id})
    kind_policy(kind)  # an unknown kind is a programming error
    return StepError(
        "NOT_FOUND",
        f"{handle_id} is not a known {kind} handle." + _source_hint(kind),
        {"handle": handle_id},
    )


def _source_hint(kind: str) -> str:
    source = kind_policy(kind).source
    return f" {kind.capitalize()} handles come from {source}." if source else ""


def _a(kind: str) -> str:
    """The kind with its article, such as "a seat" or "an object"."""
    return f"{'an' if kind[:1] in 'aeiou' else 'a'} {kind}"
