"""RosObjectMemory against a fake walkie-database node over Fast DDS (no LLM).

Each test's fake serves the five object services under a node name of its own, so a
fake torn down in one test never still looks discovered in the next. The fake spins on
one thread of its own, so it answers one request at a time, in the order they came.
"""

import itertools
import logging
import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from ros_fakes import WATCHDOG_S, Background
from test_ros_people_memory import fails, spy_on_requests, unique_name, wait_until
from walkie_database_msgs.msg import ObjectMatch, ObjectSummary, OperationStatus, RoomSummary
from walkie_database_msgs.srv import (
    GetObject,
    GetRoomContents,
    ListRooms,
    SearchObjects,
    SetObjectState,
)

from walkie_agent.capabilities import RecalledObject, RoomObjects
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import ErrorCode
from walkie_agent.ros import object_memory as object_memory_module
from walkie_agent.ros.object_memory import RosObjectMemory, memory_id

LOGGER = "walkie_agent.ros.object_memory"
SERVICES = {
    "search_objects": SearchObjects,
    "get_room_contents": GetRoomContents,
    "list_rooms": ListRooms,
    "get_object": GetObject,
    "set_object_state": SetObjectState,
}
# The state each operation the agent sends leaves an object in, as walkie-database's
# SetObjectState moves it.
STATE_AFTER = {
    "grasp_succeeded": "held",
    "grip_lost": "missing",
    "place_succeeded": "missing",
    "hand_over_succeeded": "handed_over",
}
RIGHT_HAND = "openarm_right_hand_tcp"
DATABASE_FAILURE = "The database could not search objects: database is locked"
NOT_READY = "The object memory is not ready: no clock yet."
# A cup's box as the memory stores it: along its yaw, across it, and its height, in metres.
BOX = (0.07, 0.08, 0.12)
# Short timeouts are flaky under Fast DDS discovery, so the tests wait 2 s unless they
# test a timeout, and a timeout test lowers it only for the call under test.
TIMEOUTS = {"server_timeout_s": 2.0, "call_timeout_s": 2.0}


def summary(
    object_id: int,
    label: str,
    *,
    room: str = "kitchen room",
    place: str = "",
    place_anchor_id: int = 0,
    key: str = "",
    state: str = "present",
    labels: tuple[str, ...] = (),
    official_name: str = "",
    category: str = "",
    x: float = 1.0,
    y: float = 2.0,
    z: float = 0.75,
    frame: str = "map",
    geometry_valid: bool = True,
    seen_s: float = 0.0,
    size: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> ObjectSummary:
    """An ObjectSummary with the fields RosObjectMemory reads; seen_s is last_seen.

    `size` is the box's sides, as ObjectSummary.size holds them, and `orientation` the
    pose's quaternion, (x, y, z, w).
    """
    stored = ObjectSummary(
        id=object_id,
        label=label,
        labels=list(labels),
        official_name=official_name,
        category=category,
        state=state,
        room=room,
        place=place,
        place_anchor_id=place_anchor_id,
        key=key,
        geometry_valid=geometry_valid,
    )
    stored.pose.header.frame_id = frame
    stored.pose.pose.position.x = x
    stored.pose.pose.position.y = y
    stored.pose.pose.position.z = z
    stored.size.x, stored.size.y, stored.size.z = size
    quaternion = stored.pose.pose.orientation
    quaternion.x, quaternion.y, quaternion.z, quaternion.w = orientation
    whole = math.floor(seen_s)
    stored.last_seen = TimeMsg(sec=whole, nanosec=round((seen_s - whole) * 1e9))
    return stored


def store() -> list[ObjectSummary]:
    """A store shaped like the Chula simulation's: anchors with no key, surfaces keyed by id.

    Each object with a place also names its anchor's id, as the node sets both together.
    """
    return [
        summary(15, "table", x=3.0, y=1.0),
        summary(18, "cup", place="object-15-top", place_anchor_id=15, x=3.1, y=1.1),
        summary(4, "apple", x=0.5, y=0.5),
        summary(14, "chair", room="living room", x=-2.0, y=4.0),
        summary(
            13, "bottle", room="living room", place="object-14-top", place_anchor_id=14,
            x=-2.0, y=4.1,
        ),
        summary(16, "sofa", room="living room", key="sofa_a", x=-4.0, y=5.0),
        summary(
            6, "cup", room="living room", place="sofa_a", place_anchor_id=16, x=-4.1, y=5.2
        ),
        summary(9, "cup", state="held", x=0.0, y=0.0),
    ]


class FakeObjectDatabase(Node):
    """Serves ~/search_objects, ~/get_room_contents, ~/list_rooms, ~/get_object and
    ~/set_object_state.

    serving() spins it on one thread, so it answers one request at a time, in the
    order they came. set_object_state answers CONFLICT unless the request carries the
    object's revision, and each record bumps it; `saves_before_set` bumps it that many
    more times, once per set_object_state request, before the check, as the node's
    save of a fresh sighting does between a read and a record.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.objects = store()
        self.rooms = ["kitchen room", "living room"]
        self.truncated = False
        # By the asked id: the id of the object a merge kept.
        self.merged: dict[int, int] = {}
        self.requests: dict[str, list] = {service: [] for service in SERVICES}
        self.arrivals: dict[str, list[float]] = {service: [] for service in SERVICES}
        # Every answer waits on this gate. A test clears it to hold the answers.
        self.answer_gate = threading.Event()
        self.answer_gate.set()
        # By service: the (code, message) of every answer.
        self.status: dict[str, tuple[int, str]] = {}
        # By service: the (code, message) of the next answers, in order.
        self.status_next: dict[str, list[tuple[int, str]]] = {}
        self.saves_before_set = 0
        answers = {
            "search_objects": self._search,
            "get_room_contents": self._room,
            "list_rooms": self._rooms,
            "get_object": self._get,
            "set_object_state": self._set_state,
        }
        for service, srv_type in SERVICES.items():
            self.create_service(srv_type, f"~/{service}", self._serve(service, answers[service]))

    def _serve(self, service, answer):
        def callback(request, response):
            self.arrivals[service].append(time.monotonic())
            self.requests[service].append(request)
            self.answer_gate.wait()
            queued = self.status_next.get(service)
            status = queued.pop(0) if queued else self.status.get(service)
            if status is not None:
                response.status = OperationStatus(code=status[0], message=status[1])
                return response
            response.status = OperationStatus(code=OperationStatus.OK, message="")
            answer(request, response)
            return response

        return callback

    def _present(self, request) -> list[ObjectSummary]:
        return [o for o in self.objects if o.state in request.query.states]

    def _search(self, request, response):
        found = self._present(request)[: request.max_results]
        response.matches = [ObjectMatch(object=o, score=1.0) for o in found]
        response.truncated = self.truncated

    def _room(self, request, response):
        name = request.query.room
        if name not in self.rooms:
            response.status = OperationStatus(
                code=OperationStatus.NOT_FOUND, message=f"No room {name!r}."
            )
            return
        inside = [o for o in self._present(request) if o.room == name]
        response.room = RoomSummary(name=name)
        response.objects = [ObjectMatch(object=o) for o in inside[: request.max_results]]
        response.truncated = len(inside) > request.max_results

    def _rooms(self, request, response):
        response.rooms = [RoomSummary(name=name) for name in self.rooms]

    def _get(self, request, response):
        kept = self.merged.get(request.object_id, request.object_id)
        found = next((o for o in self.objects if o.id == kept), None)
        if found is None:
            response.status = OperationStatus(
                code=OperationStatus.NOT_FOUND, message=f"No object {request.object_id}."
            )
            return
        response.requested_id = request.object_id
        response.merged = kept != request.object_id
        response.object = found

    def _set_state(self, request, response):
        found = next((o for o in self.objects if o.id == request.object_id), None)
        if found is None:
            response.status = OperationStatus(
                code=OperationStatus.NOT_FOUND, message=f"No object {request.object_id}."
            )
            return
        if self.saves_before_set:
            self.saves_before_set -= 1
            found.revision += 1
        if request.expected_revision != found.revision:
            response.status = OperationStatus(
                code=OperationStatus.CONFLICT,
                message=f"Object {found.id} is at revision {found.revision}.",
            )
            response.object = found
            return
        found.state = STATE_AFTER[request.operation]
        found.revision += 1
        response.object = found


@contextmanager
def serving(name: str) -> Iterator[FakeObjectDatabase]:
    """A FakeObjectDatabase named `name`, spinning on one thread of its own."""
    node = FakeObjectDatabase(name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield node
    finally:
        # A held answer goes out at once, so its callback ends before the executor.
        node.answer_gate.set()
        # shutdown() does not wait for spin(), and a destroy under it can raise there.
        executor.shutdown()
        thread.join(timeout=2)
        node.destroy_node()


@pytest.fixture
def database(ros):
    with serving(unique_name("fake_object_database")) as node:
        yield node


@pytest.fixture
def cancel():
    return CancelToken()


@pytest.fixture
def make_memory(connector, database, cancel):
    """A RosObjectMemory factory on the fake's services; keyword arguments go to it."""

    def make(**kwargs) -> RosObjectMemory:
        options = {**TIMEOUTS, **kwargs}
        return RosObjectMemory(connector, cancel, namespace=f"/{database.get_name()}", **options)

    return make


@pytest.fixture
def memory(make_memory):
    return make_memory()


@pytest.fixture
def no_database(connector, cancel):
    """A RosObjectMemory whose namespace no node serves."""
    namespace = f"/{unique_name('no_object_database')}"
    return RosObjectMemory(connector, cancel, namespace=namespace, server_timeout_s=0.5)


def ids(found) -> list[str]:
    return [recalled.id for recalled in found]


def warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING
    ]


def sent(database: FakeObjectDatabase) -> int:
    return sum(len(requests) for requests in database.requests.values())


# -- search -----------------------------------------------------------------------------


def test_search_asks_for_every_present_object_and_keeps_those_named(database, memory):
    found = memory.search("cup")

    assert ids(found) == ["18", "6"]
    [request] = database.requests["search_objects"]
    assert list(request.query.states) == ["present"]
    assert (request.query.thing, request.text) == ("", "")
    assert request.use_embedding is False
    assert request.max_results == object_memory_module.MAX_RESULTS


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (" Cup ", ["18", "6"]),
        ("dishes", ["18", "6"]),
        ("drink", ["13"]),
        ("furniture", ["15", "14", "16"]),
        ("spoon", []),
    ],
)
def test_search_matches_a_name_or_a_category_as_find_objects_does(memory, name, expected):
    assert ids(memory.search(name)) == expected


def test_search_matches_an_alias_or_the_official_name(database, memory):
    database.objects = [
        summary(21, "tumbler", labels=("cup",)),
        summary(22, "can", official_name="Cola"),
        summary(23, "glass"),
    ]

    assert ids(memory.search("cup")) == ["21"]
    assert ids(memory.search("cola")) == ["22"]


def test_objects_called_the_name_come_before_those_only_an_alias_matches(database, memory):
    # The node's order: a bottle the detector once called a cup, then two cups.
    database.objects = [
        summary(34, "bottle", labels=("bottle", "cup")),
        summary(21, "cup", labels=("cup",)),
        summary(26, "Mug", official_name="Cup"),
    ]

    assert ids(memory.search("cup")) == ["21", "26", "34"]
    assert ids(memory.search("cup", limit=2)) == ["21", "26"]
    assert ids(memory.search("")) == ["34", "21", "26"]


def test_an_object_the_detector_also_called_something_else_comes_last(database, memory):
    # The node's order: a Pringles can the detector called a bottle and a cup, then a
    # bottle it only ever called a bottle, then a catalogued bottle with an alias.
    database.objects = [
        summary(176, "bottle", labels=("bottle", "cup")),
        summary(178, "bottle", labels=("bottle",)),
        summary(40, "bottle", labels=("bottle", "flask"), official_name="Aqua"),
    ]

    assert ids(memory.search("bottle")) == ["178", "40", "176"]
    assert ids(memory.search("")) == ["176", "178", "40"]


# Mixed objects. The ranking reads only the labels and the official names, so it is a
# heuristic, not proof of what an object is. pick_object reaches for the first match
# that has a position, so these pin which object that is.


def test_two_bottles_with_the_same_labels_keep_the_node_s_order(database, memory):
    # Either one is a water bottle, and pick_object takes the first.
    database.objects = [
        summary(181, "bottle", labels=("bottle",)),
        summary(4, "cup"),
        summary(178, "bottle", labels=("bottle",)),
    ]

    assert ids(memory.search("water bottle")) == ["181", "178"]
    assert ids(memory.search("bottle")) == ["181", "178"]


def test_two_bottles_with_the_same_labels_come_nearest_the_point_first(database, memory):
    # pick_object asks with where the robot stands, so it takes the bottle in front of it.
    # A bottle with no position in the map comes last.
    database.objects = [
        summary(181, "bottle", labels=("bottle",), x=5.0, y=2.0),
        summary(179, "bottle", labels=("bottle",), geometry_valid=False),
        summary(178, "bottle", labels=("bottle",), x=1.0, y=0.5),
    ]

    assert ids(memory.search("water bottle", near=(0.0, 0.0))) == ["178", "181", "179"]
    assert ids(memory.search("water bottle", limit=1, near=(0.0, 0.0))) == ["178"]
    assert ids(memory.search("water bottle", near=(6.0, 2.0))) == ["181", "178", "179"]


def test_a_nearer_bottle_the_detector_once_disputed_still_comes_after_a_steady_one(
    database, memory
):
    database.objects = [
        summary(176, "bottle", labels=("bottle", "cup"), x=0.5, y=0.0),
        summary(178, "bottle", labels=("bottle",), x=4.0, y=0.0),
    ]

    assert ids(memory.search("water bottle", near=(0.0, 0.0))) == ["178", "176"]


def test_a_steady_wrong_label_comes_before_a_right_label_the_detector_once_disputed(
    database, memory
):
    # The node's order: a water bottle the detector once called a cup, then a can it
    # called a bottle every time. The labels alone cannot tell the can is wrong, so it
    # comes first.
    database.objects = [
        summary(178, "bottle", labels=("bottle", "cup")),
        summary(176, "bottle", labels=("bottle",)),
    ]

    assert ids(memory.search("water bottle")) == ["176", "178"]


def test_an_official_name_is_the_object_s_own_name_and_is_never_disputed(database, memory):
    # The node's order: a bottle the detector once called a cup; a cup the catalogue
    # names "Water Bottle"; a catalogued bottle the detector once called a cup; a bottle
    # it only ever called a bottle.
    database.objects = [
        summary(50, "bottle", labels=("bottle", "cup")),
        summary(51, "cup", labels=("cup",), official_name="Water Bottle"),
        summary(52, "bottle", labels=("bottle", "cup"), official_name="Aqua"),
        summary(53, "bottle", labels=("bottle",)),
    ]

    assert ids(memory.search("water bottle")) == ["51", "52", "53", "50"]
    assert ids(memory.search("aqua")) == ["52"]
    # The detector's label is the object's own name too, so a cup comes first for "cup"
    # even when the catalogue names it a water bottle.
    assert ids(memory.search("cup")) == ["51", "52", "50"]


def test_an_empty_name_recalls_every_present_object_up_to_the_limit(memory):
    assert ids(memory.search("")) == ["15", "18", "4", "14", "13", "16", "6"]
    assert ids(memory.search("", limit=2)) == ["15", "18"]


def test_a_truncated_answer_is_logged(database, memory, caplog):
    database.truncated = True

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        memory.search("cup")

    [warning] = warnings(caplog)
    assert "more than 100 present objects" in warning
    assert "recalling cup" in warning


@pytest.mark.parametrize("limit", [0, 101, True, 2.0])
def test_a_limit_out_of_range_is_refused_before_anything_is_sent(database, memory, limit):
    error = fails(memory.search, "cup", limit=limit)

    assert (error.code, error.message) == (
        ErrorCode.REFUSED,
        "Refused: limit must be a whole number from 1 to 100.",
    )
    assert sent(database) == 0


@pytest.mark.parametrize(
    ("near", "message"),
    [
        ((math.nan, 0.0), "Refused: near_x must be a finite number."),
        ((0.0, math.inf), "Refused: near_y must be a finite number."),
        ((True, 0.0), "Refused: near_x must be a finite number."),
        ((1.0,), "Refused: near must be a point (x, y) in the map."),
        ([1.0, 2.0], "Refused: near must be a point (x, y) in the map."),
    ],
)
def test_a_near_that_is_not_a_point_is_refused_before_anything_is_sent(
    database, memory, near, message
):
    error = fails(memory.search, "cup", near=near)

    assert (error.code, error.message) == (ErrorCode.REFUSED, message)
    assert sent(database) == 0


def test_search_asks_for_the_states_it_is_given(database, memory):
    assert ids(memory.search("cup", states=("present", "held"))) == ["18", "6", "9"]
    assert ids(memory.search("", states=("held",))) == ["9"]

    asked = [list(r.query.states) for r in database.requests["search_objects"]]
    assert asked == [["present", "held"], ["held"]]


@pytest.mark.parametrize("states", [(), ("flying",), ("present", "lost"), ["present"], "held"])
def test_states_the_memory_does_not_know_are_refused_before_anything_is_sent(
    database, memory, states
):
    error = fails(memory.search, "cup", states=states)

    assert error.code is ErrorCode.REFUSED
    assert error.message.startswith("Refused: states must name some of candidate, present,")
    assert sent(database) == 0


# -- What a record says -------------------------------------------------------------------


def test_on_names_the_anchor_the_anchor_id_names(memory):
    on = {recalled.id: recalled.on for recalled in memory.search("")}

    assert on["18"] == "table 15"
    assert on["13"] == "chair 14"
    assert on["6"] == "sofa 16"
    assert on["4"] is None


def test_without_an_anchor_id_on_is_none_whatever_the_key_names(database, memory):
    # The node never sends a place without its anchor's id; the key alone is not read
    # (decision SG-19).
    database.objects = [
        summary(15, "table"),
        summary(16, "sofa", key="sofa_a"),
        summary(18, "cup", place="object-15-top"),
        summary(6, "cup", place="sofa_a"),
    ]

    assert [recalled.on for recalled in memory.search("cup")] == [None, None]


def test_on_reads_the_whole_answer_and_is_none_without_the_anchor_in_it(database, memory):
    # Only the cups are kept, but `on` reads every object the node answered with.
    assert [recalled.on for recalled in memory.search("cup")] == ["table 15", "sofa 16"]
    database.objects = [summary(18, "cup", place="object-15-top", place_anchor_id=15)]

    [cup] = memory.search("cup")

    assert cup.on is None


def test_the_anchor_id_names_the_anchor_whatever_the_key_says(database, memory):
    database.objects = [
        summary(30, "shelf"),
        summary(31, "table", key="table_a"),
        # A seeded surface key.
        summary(32, "cup", place="shelf_2", place_anchor_id=30),
        # A key naming another anchor of the answer.
        summary(33, "cup", place="table_a", place_anchor_id=30),
        # No key at all.
        summary(34, "cup", place_anchor_id=31),
    ]

    on = {recalled.id: recalled.on for recalled in memory.search("cup")}

    assert on == {"32": "shelf 30", "33": "shelf 30", "34": "table 31"}


def test_on_and_the_name_prefer_the_official_name_to_the_label(database, memory):
    # The seeded furniture: four anchors share the label "table" (decision SG-27).
    database.objects = [
        summary(2, "table", official_name="coffee table"),
        summary(3, "table", official_name="  "),
        summary(5, "sofa"),
        summary(10, "cup", place_anchor_id=2),
        summary(14, "cup", place_anchor_id=3),
        summary(12, "bottle", labels=("cup",), official_name=" Aqua ", place_anchor_id=5),
    ]

    recalled = {r.id: (r.name, r.label, r.category, r.on) for r in memory.search("")}

    assert recalled == {
        "2": ("coffee table", "table", "furniture", None),
        "3": ("table", "table", "furniture", None),
        "5": ("sofa", "sofa", "furniture", None),
        "10": ("cup", "cup", "dish", "coffee table 2"),
        "14": ("cup", "cup", "dish", "table 3"),
        # The label and the category stay the detector's, not the official name's.
        "12": ("Aqua", "bottle", "drink", "sofa 5"),
    }
    assert ids(memory.search("coffee table")) == ["2"]


def test_an_arena_name_finds_what_the_detector_labelled_with_its_last_word(database, memory):
    # The detector's prompt is "bottle"; the arena and the people say "water bottle".
    database.objects = [summary(4, "cup"), summary(7, "bottle"), summary(9, "water bottle")]

    assert ids(memory.search("water bottle")) == ["7", "9"]


def test_the_arena_name_of_a_brand_finds_what_the_detector_labelled_with_the_brand(
    database, memory
):
    # The memory task's prompt is "pringles"; the arena says "pringles can". A can the
    # detector mostly called a bottle answers through its second label, after the can
    # it called only a Pringles.
    database.objects = [
        summary(176, "bottle", labels=("bottle", "pringles")),
        summary(178, "bottle", labels=("bottle",)),
        summary(180, "pringles", labels=("pringles",)),
    ]

    assert ids(memory.search("pringles can")) == ["180", "176"]


def test_an_anchor_id_outside_the_answer_gives_none_whatever_the_key_names(database, memory):
    database.objects = [
        summary(15, "table"),
        summary(18, "cup", place="object-15-top", place_anchor_id=40),
    ]

    [cup] = memory.search("cup")

    assert cup.on is None


def test_a_record_carries_the_memory_fields_and_the_map_position(database, memory):
    database.objects = [
        summary(18, "cup", room="kitchen room", x=3.1, y=1.1, z=0.8, size=BOX),
        summary(30, "thing", room="", category="  gadget "),
    ]

    cup, thing = memory.search("")

    assert cup == RecalledObject(
        id="18",
        name="cup",
        label="cup",
        category="dish",
        room="kitchen room",
        on=None,
        state="present",
        seen_s_ago=None,
        x=3.1,
        y=1.1,
        z=0.8,
        size=BOX,
        yaw=0.0,
    )
    assert (thing.category, thing.room) == ("gadget", None)


def turned(yaw: float, scale: float = 1.0) -> tuple[float, float, float, float]:
    """The quaternion (x, y, z, w) of a turn by `yaw` about +z, times `scale`."""
    return (0.0, 0.0, scale * math.sin(yaw / 2.0), scale * math.cos(yaw / 2.0))


@pytest.mark.parametrize(
    ("options", "size", "yaw"),
    [
        ({"orientation": turned(0.5)}, BOX, 0.5),
        ({"orientation": turned(-2.0)}, BOX, -2.0),
        # A quaternion that is not of unit length still gives its yaw.
        ({"orientation": turned(1.0, scale=2.0)}, BOX, 1.0),
        # A box the memory has not measured holds zeros.
        ({"size": (0.0, 0.0, 0.0)}, None, 0.0),
        ({"size": (0.07, 0.0, 0.12)}, None, 0.0),
        ({"size": (0.07, -0.08, 0.12)}, None, 0.0),
        ({"size": (0.07, math.nan, 0.12)}, None, 0.0),
        ({"size": (0.07, 0.08, math.inf)}, None, 0.0),
        ({"orientation": (0.0, 0.0, 0.0, 0.0)}, BOX, None),
        ({"orientation": (0.0, 0.0, math.nan, 1.0)}, BOX, None),
        ({"geometry_valid": False}, None, None),
        ({"frame": "odom"}, None, None),
    ],
)
def test_only_a_valid_map_box_gives_a_size_and_a_yaw(database, memory, options, size, yaw):
    database.objects = [summary(18, "cup", **{"size": BOX, **options})]

    [cup] = memory.search("cup")

    assert cup.size == size
    assert cup.yaw == (None if yaw is None else pytest.approx(yaw))


@pytest.mark.parametrize(
    ("options", "position", "warned"),
    [
        ({"frame": "/map"}, (1.0, 2.0, 0.75), False),
        ({"frame": "odom"}, (None, None, None), True),
        ({"geometry_valid": False}, (None, None, None), False),
        ({"x": math.nan}, (None, None, None), False),
        ({"y": math.inf}, (None, None, None), False),
        # A broken height leaves the floor position, which go_to_object needs.
        ({"z": math.nan}, (1.0, 2.0, None), False),
    ],
)
def test_only_a_valid_finite_map_pose_gives_a_position(
    database, memory, caplog, options, position, warned
):
    database.objects = [summary(18, "cup", **options)]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        [cup] = memory.search("cup")

    assert (cup.x, cup.y, cup.z) == position
    assert bool(warnings(caplog)) is warned


def test_seen_s_ago_is_the_node_clock_less_last_seen(database, memory):
    now = time.time()
    database.objects = [
        summary(18, "cup", seen_s=now - 30.0),
        summary(6, "cup", seen_s=now + 3600.0),
        summary(9, "cup"),
    ]

    seen, future, never = memory.search("cup")

    assert 30.0 <= seen.seen_s_ago < 30.0 + WATCHDOG_S
    # A last_seen after now, as from a store an earlier simulation run wrote.
    assert future.seen_s_ago is None
    assert never.seen_s_ago is None


def test_seen_s_ago_is_none_before_the_clock_has_started(database, memory, monkeypatch):
    database.objects = [summary(18, "cup", seen_s=5.0)]
    stopped = SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))
    monkeypatch.setattr(memory, "_node", SimpleNamespace(get_clock=lambda: stopped))

    [cup] = memory.search("cup")

    assert cup.seen_s_ago is None


# -- room ---------------------------------------------------------------------------------


def test_room_lists_the_present_objects_there(database, memory):
    contents = memory.room("living room")

    assert isinstance(contents, RoomObjects)
    assert (contents.room, contents.complete) == ("living room", True)
    assert ids(contents.objects) == ["14", "13", "16", "6"]
    assert [recalled.on for recalled in contents.objects] == [None, "chair 14", None, "sofa 16"]
    request = database.requests["get_room_contents"][0]
    assert (request.query.room, list(request.query.states)) == ("living room", ["present"])
    assert request.max_results == 50


def test_room_says_when_the_list_is_not_complete(memory):
    contents = memory.room("living room", limit=2)

    assert ids(contents.objects) == ["14", "13"]
    assert contents.complete is False


def test_a_room_named_in_another_case_is_asked_for_again_by_its_stored_name(database, memory):
    contents = memory.room("  Living Room ")

    assert contents.room == "living room"
    assert [r.query.room for r in database.requests["get_room_contents"]] == [
        "Living Room",
        "living room",
    ]
    assert len(database.requests["list_rooms"]) == 1


def test_an_unknown_room_is_not_found_and_lists_the_rooms_it_knows(database, memory):
    error = fails(memory.room, "garage")

    assert (error.code, error.message) == (
        ErrorCode.NOT_FOUND,
        "Walkie remembers no room 'garage'. The rooms it remembers: kitchen room, living room.",
    )
    assert len(database.requests["get_room_contents"]) == 1


def test_a_known_room_the_node_cannot_find_is_not_asked_for_again(database, memory):
    database.status_next["get_room_contents"] = [(OperationStatus.NOT_FOUND, "No room.")]

    error = fails(memory.room, "kitchen room")

    assert error.code is ErrorCode.NOT_FOUND
    assert len(database.requests["get_room_contents"]) == 1


def test_no_rooms_at_all_says_none(database, memory):
    database.rooms = []

    error = fails(memory.room, "kitchen room")

    assert error.message.endswith("The rooms it remembers: none.")


def test_an_empty_room_is_refused_before_anything_is_sent(database, memory):
    error = fails(memory.room, "  ")

    assert (error.code, error.message) == (ErrorCode.REFUSED, "Refused: room must name a room.")
    assert sent(database) == 0


# -- get ----------------------------------------------------------------------------------


def test_get_recalls_one_object_without_its_cloud(database, memory):
    cup = memory.get(" 18 ")

    assert (cup.id, cup.name, cup.on, cup.x, cup.y) == ("18", "cup", None, 3.1, 1.1)
    [request] = database.requests["get_object"]
    assert (request.object_id, request.include_cloud) == (18, False)


def test_get_follows_a_merge_to_the_object_that_kept_it(database, memory):
    database.merged[5] = 18

    assert memory.get("5").id == "18"


@pytest.mark.parametrize("code", [OperationStatus.NOT_FOUND, OperationStatus.WRONG_MAP])
def test_get_of_an_object_the_map_does_not_hold_is_not_found(database, memory, code):
    database.status_next["get_object"] = [(code, "No such object here.")]

    error = fails(memory.get, "99")

    assert (error.code, error.message) == (ErrorCode.NOT_FOUND, "Walkie remembers no object 99.")


@pytest.mark.parametrize(
    "object_id", ["cup_1", "0", "-3", "+3", "018", "1.5", "", "1 8", str(2**63)]
)
def test_an_id_that_is_not_a_memory_id_is_refused_before_anything_is_sent(
    database, memory, object_id
):
    error = fails(memory.get, object_id)

    assert error.code is ErrorCode.REFUSED
    assert error.message.startswith(f"Refused: {object_id!r} is not a remembered object id.")
    assert "the ids find_objects gives are for objects in view" in error.message
    assert sent(database) == 0


def test_memory_id_takes_the_largest_int64():
    assert memory_id(str(2**63 - 1)) == 2**63 - 1


# -- set_state ----------------------------------------------------------------------------


def stored(database: FakeObjectDatabase, object_id: int) -> ObjectSummary:
    return next(o for o in database.objects if o.id == object_id)


def test_set_state_records_the_operation_at_the_objects_current_revision(database, memory):
    cup = stored(database, 18)
    cup.map_id, cup.revision = "chula_pat14_6", 3

    held = memory.set_state("18", "grasp_succeeded", gripper_frame=RIGHT_HAND)

    assert (held.id, held.state) == ("18", "held")
    [read] = database.requests["get_object"]
    assert (read.object_id, read.include_cloud) == (18, False)
    [record] = database.requests["set_object_state"]
    assert (record.map_id, record.object_id, record.expected_revision) == ("chula_pat14_6", 18, 3)
    assert (record.operation, record.gripper_frame) == ("grasp_succeeded", RIGHT_HAND)
    assert record.has_pose is False
    assert record.cause == "walkie-agent"
    assert len(record.request_id) == 32
    assert cup.revision == 4


def test_a_revision_that_changed_before_the_record_is_read_again_once(database, memory):
    database.saves_before_set = 1

    assert memory.set_state("18", "grip_lost", gripper_frame=RIGHT_HAND).state == "missing"

    records = database.requests["set_object_state"]
    assert [r.expected_revision for r in records] == [0, 1]
    assert len(database.requests["get_object"]) == 2
    # Each try is a request of its own.
    assert records[0].request_id != records[1].request_id


def test_a_revision_that_changes_again_is_internal_after_the_second_try(database, memory):
    database.saves_before_set = 2

    error = fails(memory.set_state, "18", "place_succeeded", gripper_frame=RIGHT_HAND)

    assert (error.code, error.message) == (
        ErrorCode.INTERNAL,
        "Object memory answered CONFLICT: Object 18 is at revision 2.",
    )
    assert len(database.requests["set_object_state"]) == 2
    assert stored(database, 18).state == "present"


def test_a_memory_that_does_not_record_states_is_unavailable_at_once(database, memory):
    database.status["set_object_state"] = (OperationStatus.NOT_READY, "Not implemented yet.")

    error = fails(memory.set_state, "18", "grasp_succeeded", gripper_frame=RIGHT_HAND)

    assert (error.code, error.message) == (
        ErrorCode.UNAVAILABLE,
        "Object memory is unavailable: Not implemented yet.",
    )
    # Asked once: a pick holding an object in the air does not wait for it.
    assert len(database.requests["set_object_state"]) == 1


@pytest.mark.parametrize("service", ["get_object", "set_object_state"])
def test_a_record_for_an_object_the_map_does_not_hold_is_not_found(database, memory, service):
    database.status_next[service] = [(OperationStatus.WRONG_MAP, "Another map.")]

    error = fails(memory.set_state, "18", "grip_lost", gripper_frame=RIGHT_HAND)

    assert (error.code, error.message) == (ErrorCode.NOT_FOUND, "Walkie remembers no object 18.")


def test_a_record_the_memory_refuses_is_internal(database, memory):
    refused = "Object 9 is held by openarm_left_hand_tcp."
    database.status["set_object_state"] = (OperationStatus.INVALID, refused)

    error = fails(memory.set_state, "9", "hand_over_succeeded", gripper_frame=RIGHT_HAND)

    assert (error.code, error.message) == (
        ErrorCode.INTERNAL,
        f"Object memory answered INVALID: {refused}",
    )
    assert len(database.requests["set_object_state"]) == 1


@pytest.mark.parametrize("operation", ["begin_grasp", "grasp_failed", "prepare_place", "held"])
def test_an_operation_the_agent_does_not_send_is_refused_before_anything_is_sent(
    database, memory, operation
):
    error = fails(memory.set_state, "18", operation, gripper_frame=RIGHT_HAND)

    assert error.code is ErrorCode.REFUSED
    assert error.message == (
        "Refused: operation must be one of grasp_succeeded, grip_lost, place_succeeded, "
        "hand_over_succeeded."
    )
    assert sent(database) == 0


# -- Answers asked again ------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [(OperationStatus.NOT_READY, NOT_READY), (OperationStatus.INTERNAL, DATABASE_FAILURE)],
)
def test_not_ready_or_a_database_failure_is_asked_again_until_it_passes(
    database, memory, status
):
    database.status_next["search_objects"] = [status, status]

    assert ids(memory.search("cup")) == ["18", "6"]

    arrivals = database.arrivals["search_objects"]
    assert len(arrivals) == 3
    assert all(
        later - earlier >= object_memory_module.RETRY_STEP_S
        for earlier, later in itertools.pairwise(arrivals)
    )


def test_not_ready_that_lasts_is_unavailable_after_the_retries(database, memory):
    database.status["search_objects"] = (OperationStatus.NOT_READY, NOT_READY)

    start = time.monotonic()
    error = fails(memory.search, "cup")
    elapsed = time.monotonic() - start

    assert (error.code, error.message) == (
        ErrorCode.UNAVAILABLE,
        f"Object memory is unavailable: {NOT_READY}",
    )
    budget_s = object_memory_module.RETRY_BUDGET_S
    step_s = object_memory_module.RETRY_STEP_S
    assert budget_s - step_s <= elapsed < budget_s + 2.0
    assert 2 <= len(database.requests["search_objects"]) <= 1 + round(budget_s / step_s)


def test_the_retries_stop_at_the_calls_deadline(database, memory):
    database.status["get_object"] = (OperationStatus.INTERNAL, DATABASE_FAILURE)

    start = time.monotonic()
    error = fails(memory.get, "18", timeout_s=0.6)
    elapsed = time.monotonic() - start

    assert (error.code, error.message) == (
        ErrorCode.UNAVAILABLE,
        f"Object memory is unavailable: {DATABASE_FAILURE}",
    )
    assert elapsed < object_memory_module.RETRY_BUDGET_S


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ((OperationStatus.INTERNAL, "boom"), "Object memory answered INTERNAL: boom"),
        ((OperationStatus.INVALID, "bad query"), "Object memory answered INVALID: bad query"),
        ((OperationStatus.BUSY, "later"), "Object memory answered BUSY: later"),
        ((42, "what"), "Object memory answered 42: what"),
    ],
)
def test_any_other_answer_is_internal_at_once(database, memory, status, message):
    database.status_next["search_objects"] = [status]

    error = fails(memory.search, "cup")

    assert (error.code, error.message) == (ErrorCode.INTERNAL, message)
    assert len(database.requests["search_objects"]) == 1


def test_a_failed_list_of_rooms_is_internal(database, memory):
    database.status_next["list_rooms"] = [(OperationStatus.INVALID, "no map")]

    error = fails(memory.room, "garage")

    assert (error.code, error.message) == (
        ErrorCode.INTERNAL,
        "Object memory answered INVALID: no map",
    )


def test_a_stop_during_the_pause_before_a_retry_ends_the_call(
    database, memory, cancel, monkeypatch
):
    # A pause far past the watchdog, so only a pause that ends at the stop lets the
    # call end in time.
    monkeypatch.setattr(object_memory_module, "RETRY_STEP_S", 30.0)
    monkeypatch.setattr(object_memory_module, "RETRY_BUDGET_S", 60.0)
    database.status["search_objects"] = (OperationStatus.NOT_READY, NOT_READY)
    pausing = threading.Event()
    pause = memory._pause

    def record_pause(seconds, what):
        pausing.set()
        return pause(seconds, what)

    monkeypatch.setattr(memory, "_pause", record_pause)

    call = Background(memory.search, "cup", timeout_s=120.0)
    assert pausing.wait(WATCHDOG_S), "The call never paused before a retry."
    cancel.cancel()
    error = call.step_error()

    assert error.code is ErrorCode.CANCELLED
    assert len(database.requests["search_objects"]) == 1


# -- Errors, stops and deadlines --------------------------------------------------------


def test_no_server_is_unavailable(no_database):
    error = fails(no_database.search, "cup")

    service = no_database._search_client.srv_name
    assert service.endswith("/search_objects")
    assert (error.code, error.message) == (
        ErrorCode.UNAVAILABLE,
        f"Object memory is unavailable: no node offers {service}. The walkie-database node "
        "is not running, or not on this ROS domain.",
    )


def test_a_slow_server_times_out_and_drops_the_pending_request(
    database, memory, monkeypatch, caplog
):
    memory.search("cup")
    memory._call_timeout_s = 0.3
    dropped, _ = spy_on_requests(monkeypatch)
    database.answer_gate.clear()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        error = fails(memory.search, "cup")

    assert (error.code, error.message) == (
        ErrorCode.TIMEOUT,
        "Object memory did not answer within 0.3 s. It may be busy; try again once.",
    )
    assert len(dropped) == 1
    [warning] = warnings(caplog)
    assert "pinned in ros2_ws.repos" in warning


def test_timeout_s_cuts_the_wait_for_an_answer(database, memory):
    memory.search("cup")
    database.answer_gate.clear()

    start = time.monotonic()
    error = fails(memory.get, "18", timeout_s=0.3)
    elapsed = time.monotonic() - start

    assert (error.code, error.message) == (
        ErrorCode.TIMEOUT,
        "Object memory did not answer within 0.3 s. It may be busy; try again once.",
    )
    assert elapsed < TIMEOUTS["call_timeout_s"]


def test_a_stop_during_the_wait_for_an_answer_ends_it_and_drops_the_request(
    database, make_memory, cancel, monkeypatch
):
    memory = make_memory(call_timeout_s=30.0)
    memory.search("cup")
    dropped, _ = spy_on_requests(monkeypatch)
    database.answer_gate.clear()

    call = Background(memory.room, "living room", timeout_s=60.0)
    wait_until(lambda: database.requests["get_room_contents"], "the held room request")
    cancel.cancel()
    error = call.step_error()

    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        "Cancelled recalling the objects in living room: a stop is in effect.",
    )
    assert len(dropped) == 1
    assert not dropped[0].done()


def test_a_call_during_a_stop_sends_nothing(database, memory, cancel):
    assert memory._search_client.wait_for_service(timeout_sec=5.0)
    cancel.cancel()

    error = fails(memory.search, "cup")

    assert (error.code, error.message) == (
        ErrorCode.CANCELLED,
        "Cancelled recalling cup: a stop is in effect.",
    )
    assert sent(database) == 0


def test_a_forbidden_service_name_is_refused(connector, cancel):
    error = fails(
        RosObjectMemory,
        connector,
        cancel,
        namespace="/walkie",
        forbidden=("/walkie/get_object",),
    )

    assert (error.code, error.message) == (
        ErrorCode.REFUSED,
        "Refused: a capability may not use /walkie/get_object.",
    )
