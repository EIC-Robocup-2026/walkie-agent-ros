"""RosObjectMemory: ctx.object_memory on walkie-database's object services.

It answers recall_object, list_room_objects and go_to_object (decisions SG-16 to
SG-21) from the node's /walkie_database/search_objects, get_room_contents,
list_rooms and get_object (walkie-database feat/scene-graph-prep). main creates it
once from the connector, with --object-memory. Its clients are made here, once, on
the connector's node, and its methods block the calling thread, never the
executor's. The Runner runs one tool call at a time, so it holds no lock.

What each verb asks:

- search: SearchObjects for every object in the state "present", with no thing, no
  text and no vector, up to MAX_RESULTS, then keeps those whose label, an alias or the
  official name matches `name` as find_objects matches a label. The node's own match
  only folds case, so "cups" or "dishes" would find nothing there (decision SG-17).
  An answer that the node truncated is logged, since objects past it are never
  matched.
- room: GetRoomContents for the room's present objects. A room that the node does not
  know by that exact name is looked up in ListRooms ignoring case and asked for again
  by its stored name; NOT_FOUND then lists the rooms the memory knows.
- get: GetObject without the cloud. The node follows a merge, so the object may come
  back with another id.

A record's `name` is the object's official name, such as "coffee table", when the
memory has one, else its label, such as "table" (decision SG-27). Its `label` is the
stored label either way, the word the detector and find_objects use, and its category
still comes from the label when the memory has none.

An object's `on` is "<name> <id>" of the anchor that ObjectSummary.place_anchor_id
names, when the same answer holds that anchor, else None (decisions SG-19 and SG-27),
with the anchor's name chosen as a record's is. The id is 0
when the memory holds no place for the object, as when no surface holds it.
ObjectSummary.place, the surface or anchor key, is never read:
the node sets the id whenever it sets the key. get()'s answer holds one object, so
its `on` is always None.

`seen_s_ago` is the node clock's now, sim time under --use-sim-time, less
ObjectSummary.last_seen. It is None when last_seen is zero, the clock has not
started, or last_seen is later than now. A store kept from an earlier simulation run
holds stamps from that run's clock, which restarted at zero. Such a stamp reads as
None until the new clock passes it and as a wrong age after, and the node keeps the
later of the old and new stamps, so an object seen again early in the new run also
reads as None. ObjectSummary tells no run from another, so a sim run restores the
store from its seed first, with walkie-database's `db-restore` (decision SG-20).

`x` and `y` are the pose's position when geometry_valid and the pose is in the map
frame, else None.

Each wait for discovery lasts at most `server_timeout_s`, and each wait for an answer
at most `call_timeout_s`. A call's `timeout_s` bounds the whole call.

Errors, as StepError codes:

- UNAVAILABLE: no node offers the service within `server_timeout_s`, or the node
  answered NOT_READY (it runs without memory mode, or its clock has not started) or a
  database failure (INTERNAL with DATABASE_FAILURE_PREFIX). Those two answers are sent
  again every RETRY_STEP_S for at most RETRY_BUDGET_S after the first, never past the
  call's `timeout_s`, and only then are UNAVAILABLE with the node's message.
- TIMEOUT: no answer in time. The request is dropped, never sent again, and
  STALE_PIN_HINT is logged.
- NOT_FOUND: get() of an id the memory does not know on its map (NOT_FOUND or
  WRONG_MAP), or room() of a room it does not know.
- REFUSED: an argument outside its range, before anything is sent.
- INTERNAL: any other answer, as "Object memory answered <CODE>: <message>". It
  means a bug on one side or the other, such as a node on another map.
- CANCELLED: a stop during a wait. Nothing is written, so a stop never leaves
  anything half done.
"""

import logging
import math
import re
import time
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from rclpy.client import Client
from walkie_database_msgs.msg import ObjectQuery, ObjectSummary, OperationStatus
from walkie_database_msgs.srv import GetObject, GetRoomContents, ListRooms, SearchObjects

from walkie_agent.capabilities import RecalledObject, RoomObjects
from walkie_agent.capabilities.guards import CancelToken, require_text, require_timeout
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.perception.object_words import category_of, matches
from walkie_agent.ros.base import MAP_FRAME
from walkie_agent.ros.people_memory import (
    CALL_TIMEOUT_S,
    DATABASE_FAILURE_PREFIX,
    NAMESPACE,
    SERVER_TIMEOUT_S,
    STALE_PIN_HINT,
)
from walkie_agent.ros.policy import FORBIDDEN_NAMES, check_endpoint, ros_failure

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

logger = logging.getLogger(__name__)

# The node's cap on max_results (walkie-database queries.py MAX_RESULTS).
MAX_RESULTS = 100
# The states search and room ask for. "held" is left out: an object in a hand is
# not where Walkie last saw it.
STATES = ("present",)
# The pause before a NOT_READY or a database failure is asked again, and how long
# after the first such answer it may still be asked.
RETRY_STEP_S = 0.25
RETRY_BUDGET_S = 2.0
# A memory id: ObjectSummary.id is a positive int64.
_ID = re.compile(r"[1-9][0-9]{0,18}")
_MAX_ID = 2**63 - 1
_POLL_S = 0.02
_SERVICES = ("search_objects", "get_room_contents", "list_rooms", "get_object")
_CODE_NAMES = {
    getattr(OperationStatus, name): name
    for name in (
        "OK",
        "INVALID",
        "NOT_FOUND",
        "STALE",
        "TF_UNAVAILABLE",
        "WRONG_MAP",
        "UNSUPPORTED",
        "BUSY",
        "NOT_READY",
        "INTERNAL",
        "CONFLICT",
    )
}
# The answers that _call returns for the verb to read. Every other one but a retried
# one is INTERNAL.
_ANSWERED = frozenset({OperationStatus.OK, OperationStatus.NOT_FOUND, OperationStatus.WRONG_MAP})


class RosObjectMemory:
    """An ObjectMemory on walkie-database's object services; see the module docstring.

    `cancel` is the agent's token: a stop ends every wait. `forbidden` lists the names
    no endpoint may use, as in the other ROS capabilities. A service name among them
    raises StepError REFUSED here.
    """

    def __init__(
        self,
        connector: "ROS2Connector",
        cancel: CancelToken,
        *,
        namespace: str = NAMESPACE,
        server_timeout_s: float = SERVER_TIMEOUT_S,
        call_timeout_s: float = CALL_TIMEOUT_S,
        forbidden: Iterable[str] = FORBIDDEN_NAMES,
    ) -> None:
        node = connector.node
        self._node = node
        self._cancel = cancel
        self._server_timeout_s = server_timeout_s
        self._call_timeout_s = call_timeout_s
        forbidden = tuple(forbidden)
        names = {
            service: check_endpoint(node, f"{namespace}/{service}", forbidden)
            for service in _SERVICES
        }
        self._search_client = node.create_client(SearchObjects, names["search_objects"])
        self._room_client = node.create_client(GetRoomContents, names["get_room_contents"])
        self._rooms_client = node.create_client(ListRooms, names["list_rooms"])
        self._get_client = node.create_client(GetObject, names["get_object"])

    def search(
        self, name: str, *, limit: int = 10, timeout_s: float = 10.0
    ) -> tuple[RecalledObject, ...]:
        require_text(name=name)
        _require_limit(limit)
        wanted = name.strip().lower()
        what = f"recalling {wanted}" if wanted else "recalling objects"
        deadline = _deadline(timeout_s)
        request = SearchObjects.Request(
            query=ObjectQuery(states=list(STATES)),
            use_embedding=False,
            max_results=MAX_RESULTS,
        )
        response = self._call(self._search_client, request, deadline, what)
        _expect_ok(response.status)
        if response.truncated:
            logger.warning(
                "The object memory holds more than %d present objects; %s looked at the "
                "first %d only.",
                MAX_RESULTS,
                what,
                MAX_RESULTS,
            )
        batch = [match.object for match in response.matches]
        named = [summary for summary in batch if _is_named(wanted, summary)]
        # An object called `wanted` comes before one that only an alias matches, such
        # as a bottle the detector once called a cup. The sort keeps the node's order.
        named.sort(key=lambda summary: not _is_called(wanted, summary))
        return self._recalled(named[:limit], batch)

    def room(self, room: str, *, limit: int = 50, timeout_s: float = 10.0) -> RoomObjects:
        require_text(room=room)
        _require_limit(limit)
        asked = room.strip()
        if not asked:
            raise StepError(ErrorCode.REFUSED, "Refused: room must name a room.")
        what = f"recalling the objects in {asked}"
        deadline = _deadline(timeout_s)
        response = self._call(self._room_client, _room_request(asked, limit), deadline, what)
        if response.status.code == OperationStatus.NOT_FOUND:
            rooms = self._rooms(deadline, what)
            stored = next((r for r in rooms if r.casefold() == asked.casefold()), None)
            if stored is None or stored == asked:
                known = ", ".join(rooms) or "none"
                raise StepError(
                    ErrorCode.NOT_FOUND,
                    f"Walkie remembers no room {asked!r}. The rooms it remembers: {known}.",
                )
            response = self._call(self._room_client, _room_request(stored, limit), deadline, what)
        _expect_ok(response.status)
        batch = [match.object for match in response.objects]
        return RoomObjects(
            room=response.room.name,
            objects=self._recalled(batch, batch),
            complete=not response.truncated,
        )

    def get(self, object_id: str, *, timeout_s: float = 10.0) -> RecalledObject:
        require_text(object_id=object_id)
        number = memory_id(object_id)
        what = f"recalling object {number}"
        deadline = _deadline(timeout_s)
        request = GetObject.Request(object_id=number, include_cloud=False)
        response = self._call(self._get_client, request, deadline, what)
        if response.status.code in (OperationStatus.NOT_FOUND, OperationStatus.WRONG_MAP):
            raise StepError(ErrorCode.NOT_FOUND, f"Walkie remembers no object {number}.")
        _expect_ok(response.status)
        return self._recalled([response.object], [response.object])[0]

    # -- Answers ----------------------------------------------------------------------

    def _rooms(self, deadline: float, what: str) -> list[str]:
        """The names of the rooms the memory knows, as it spells them."""
        response = self._call(self._rooms_client, ListRooms.Request(), deadline, what)
        _expect_ok(response.status)
        return [room.name for room in response.rooms]

    def _recalled(
        self, chosen: Sequence[ObjectSummary], batch: Sequence[ObjectSummary]
    ) -> tuple[RecalledObject, ...]:
        """The records of `chosen`, whose `on` names an anchor of `batch`."""
        now = self._now_s()
        by_id = {summary.id: summary for summary in batch}
        return tuple(_record(summary, _on(summary, by_id), now) for summary in chosen)

    def _now_s(self) -> float | None:
        """The node clock's now in seconds, or None before it has started."""
        try:
            nanoseconds = self._node.get_clock().now().nanoseconds
        except Exception as error:
            raise ros_failure(error, "Reading the clock") from error
        return nanoseconds * 1e-9 if nanoseconds > 0 else None

    # -- Calls ------------------------------------------------------------------------

    def _call(self, client: Client, request: Any, deadline: float, what: str) -> Any:
        """The node's answer with status OK, NOT_FOUND or WRONG_MAP; see the module docstring.

        NOT_READY and a database failure are asked again; any other status is INTERNAL.
        """
        retry_until: float | None = None
        while True:
            response = self._answer(client, request, deadline, what)
            status = response.status
            if status.code in _ANSWERED:
                return response
            if not _may_pass(status):
                _expect_ok(status)
            now = time.monotonic()
            if retry_until is None:
                retry_until = min(now + RETRY_BUDGET_S, deadline)
            if now + RETRY_STEP_S >= retry_until:
                raise StepError(
                    ErrorCode.UNAVAILABLE, f"Object memory is unavailable: {status.message}"
                )
            logger.debug(f"{client.srv_name} answered {status.message!r}; asking again.")
            self._pause(RETRY_STEP_S, what)

    def _answer(self, client: Client, request: Any, deadline: float, what: str) -> Any:
        """Send one request and return the node's answer, whatever its status."""
        name = client.srv_name
        discovery_s = _budget(deadline, self._server_timeout_s, what)
        try:
            self._wait(lambda: self._is_ready(client), discovery_s, what)
        except StepError as error:
            if error.code is not ErrorCode.TIMEOUT:
                raise
            raise StepError(
                ErrorCode.UNAVAILABLE,
                f"Object memory is unavailable: no node offers {name}. The walkie-database "
                "node is not running, or not on this ROS domain.",
            ) from error
        wait_s = _budget(deadline, self._call_timeout_s, what)
        # In the fence, so a stop either comes first and nothing goes out, or waits
        # until the request has gone.
        with self._cancel.unless_cancelled(what):
            future = _send(client, request, what)
        try:
            self._wait(future.done, wait_s, what)
        except StepError as error:
            # The answer would otherwise be kept for a caller that left.
            client.remove_pending_request(future)
            if error.code is ErrorCode.TIMEOUT:
                logger.warning(f"{name} did not answer within {wait_s:.3g} s. {STALE_PIN_HINT}")
                raise StepError(
                    ErrorCode.TIMEOUT,
                    f"Object memory did not answer within {wait_s:.3g} s. It may be busy; "
                    "try again once.",
                ) from error
            raise
        return future.result()

    def _wait(self, done: Callable[[], bool], wait_s: float, what: str) -> None:
        """Wait until done() is true, for at most `wait_s`. A stop ends the wait."""
        self._cancel.wait_for(done, wait_s, what, poll_s=_POLL_S)

    def _pause(self, seconds: float, what: str) -> None:
        """Sleep before a request is sent again. A stop ends the pause."""
        end = time.monotonic() + seconds
        while (remaining := end - time.monotonic()) > 0:
            self._cancel.check(what)
            time.sleep(min(_POLL_S, remaining))

    def _is_ready(self, client: Client) -> bool:
        try:
            return client.service_is_ready()
        except Exception as error:
            raise ros_failure(error, "Finding the object memory") from error


def memory_id(object_id: str) -> int:
    """The object memory id that `object_id` spells, or StepError REFUSED.

    An id is a whole number from 1 to 2**63 - 1, as ObjectSummary.id is, written in
    decimal with no sign and no leading zero. Spaces at either end are ignored.
    """
    text = object_id.strip()
    if _ID.fullmatch(text) and int(text) <= _MAX_ID:
        return int(text)
    raise StepError(
        ErrorCode.REFUSED,
        f"Refused: {object_id!r} is not a remembered object id. Remembered ids are "
        'numbers, such as "18", from recall_object or list_room_objects; the ids '
        "find_objects gives are for objects in view.",
    )


def _require_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_RESULTS:
        raise StepError(
            ErrorCode.REFUSED, f"Refused: limit must be a whole number from 1 to {MAX_RESULTS}."
        )


def _room_request(room: str, limit: int) -> GetRoomContents.Request:
    return GetRoomContents.Request(
        query=ObjectQuery(room=room, states=list(STATES)), max_results=limit
    )


def _is_named(wanted: str, summary: ObjectSummary) -> bool:
    """True when `wanted`, stripped and in lower case, matches one of the object's names."""
    names = {summary.label, *summary.labels, summary.official_name}
    return not wanted or any(matches(wanted, name) for name in names if name.strip())


def _is_called(wanted: str, summary: ObjectSummary) -> bool:
    """True when `wanted` matches the object's own label or its official name."""
    names = {summary.label, summary.official_name}
    return not wanted or any(matches(wanted, name) for name in names if name.strip())


def _name(summary: ObjectSummary) -> str:
    """The object's official name when the memory has one, else its label."""
    return summary.official_name.strip() or summary.label.strip()


def _on(summary: ObjectSummary, by_id: dict[int, ObjectSummary]) -> str | None:
    """'<name> <id>' of the anchor place_anchor_id names, if it is in the answer."""
    anchor = by_id.get(summary.place_anchor_id) if summary.place_anchor_id else None
    return None if anchor is None else f"{_name(anchor)} {anchor.id}"


def _record(summary: ObjectSummary, on: str | None, now: float | None) -> RecalledObject:
    label = summary.label.strip()
    seen = summary.last_seen.sec + summary.last_seen.nanosec * 1e-9
    seen_s_ago = None if not seen or now is None or now < seen else now - seen
    x = y = None
    if summary.geometry_valid:
        frame = summary.pose.header.frame_id.lstrip("/")
        position = summary.pose.pose.position
        if frame != MAP_FRAME:
            logger.warning(
                "Object %d's pose is in the frame %r, not %r; it has no position here.",
                summary.id,
                frame,
                MAP_FRAME,
            )
        elif math.isfinite(position.x) and math.isfinite(position.y):
            x, y = float(position.x), float(position.y)
    return RecalledObject(
        id=str(summary.id),
        name=_name(summary),
        label=label,
        category=summary.category.strip() or category_of(label),
        room=summary.room.strip() or None,
        on=on,
        state=summary.state,
        seen_s_ago=seen_s_ago,
        x=x,
        y=y,
    )


def _may_pass(status: OperationStatus) -> bool:
    """True for an answer that may pass if asked again: NOT_READY, or a database failure."""
    return status.code == OperationStatus.NOT_READY or (
        status.code == OperationStatus.INTERNAL
        and status.message.startswith(DATABASE_FAILURE_PREFIX)
    )


def _expect_ok(status: OperationStatus) -> None:
    if status.code != OperationStatus.OK:
        name = _CODE_NAMES.get(status.code, str(status.code))
        raise StepError(ErrorCode.INTERNAL, f"Object memory answered {name}: {status.message}")


def _deadline(timeout_s: float) -> float:
    require_timeout(timeout_s)
    return time.monotonic() + timeout_s


def _budget(deadline: float, limit_s: float, what: str) -> float:
    """A wait's budget. CancelToken.wait_for reads zero or less as REFUSED, so it never gets one."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise StepError(ErrorCode.TIMEOUT, f"Timed out: {what} ran out of time.")
    return min(limit_s, remaining)


def _send(client: Client, request: Any, what: str) -> Any:
    try:
        return client.call_async(request)
    except Exception as error:
        raise ros_failure(error, f"Sending a request for {what}") from error
