"""recall_object, list_room_objects and go_to_object over FakeRobot and a stand-in memory."""

import json
import math

import pytest

from walkie_agent.capabilities import (
    NO_OBJECT_MEMORY,
    Ctx,
    RecalledObject,
    RoomObjects,
    UnavailableObjectMemory,
)
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.tools.atomic import go_to_object, list_room_objects, recall_object
from walkie_agent.tools.atomic.memory import RECALL_LIMIT, ROOM_LIMIT
from walkie_agent.tools.atomic.navigation import APPROACH_PLACE_MAX_M

WORLD = {
    "places": {
        "kitchen room": {"x": 0.0, "y": 0.0, "yaw": 0.0, "room": "kitchen room"},
        "kitchen table": {"x": 3.0, "y": 0.0, "yaw": 0.0, "room": "kitchen room"},
        "shelf": {"x": 3.0, "y": 3.0, "yaw": 0.0, "room": "kitchen room"},
        "kitchen door": {"x": 7.0, "y": 1.0, "yaw": 0.0, "room": "kitchen room"},
        "living room": {"x": 10.0, "y": 0.0, "yaw": 0.0, "room": "living room"},
        "sofa": {"x": 10.0, "y": 2.0, "yaw": 0.0, "room": "living room"},
    },
    "objects": {},
    "people": {},
    "start": "kitchen room",
}


def recalled(
    object_id: str = "18",
    name: str = "cup",
    *,
    label: str | None = None,
    room: str | None = "kitchen room",
    on: str | None = "table 15",
    state: str = "present",
    seen_s_ago: float | None = 42.4,
    x: float | None = 4.0,
    y: float | None = 0.0,
) -> RecalledObject:
    return RecalledObject(
        id=object_id,
        name=name,
        label=name if label is None else label,
        category="dish",
        room=room,
        on=on,
        state=state,
        seen_s_ago=seen_s_ago,
        x=x,
        y=y,
    )


class Remembers:
    """An ObjectMemory over fixed records. `merged` maps an asked id to the id that kept it."""

    def __init__(self, *objects: RecalledObject, merged: dict[str, str] | None = None) -> None:
        self.objects = {found.id: found for found in objects}
        self.merged = merged or {}
        self.calls: list[tuple] = []

    def search(self, name, *, limit=10, near=None, timeout_s=10.0):
        self.calls.append(("search", name, limit, near))
        found = [o for o in self.objects.values() if not name or o.name == name]
        if near is not None:
            found.sort(key=lambda o: _distance(o, near))
        return tuple(found[:limit])

    def room(self, room, *, limit=50, timeout_s=10.0):
        self.calls.append(("room", room, limit))
        inside = [o for o in self.objects.values() if o.room == room]
        return RoomObjects(room, tuple(inside[:limit]), len(inside) <= limit)

    def get(self, object_id, *, timeout_s=10.0):
        self.calls.append(("get", object_id))
        kept = self.merged.get(object_id.strip(), object_id.strip())
        if kept not in self.objects:
            raise StepError(ErrorCode.NOT_FOUND, f"Walkie remembers no object {kept}.")
        return self.objects[kept]


def _distance(found: RecalledObject, near: tuple[float, float]) -> float:
    if found.x is None or found.y is None:
        return math.inf
    return math.hypot(found.x - near[0], found.y - near[1])


def remembering(*objects: RecalledObject, **options) -> tuple[FakeRobot, Ctx]:
    robot = FakeRobot(WORLD)
    memory = Remembers(*objects, **options)
    ctx = Ctx(
        cancel=robot.cancel, base=robot, speech=robot, vision=robot, object_memory=memory
    )
    return robot, ctx


def base_verbs(robot: FakeRobot) -> list[str]:
    return [call["verb"] for call in robot.calls if call["verb"].startswith("base.")]


def turns(robot: FakeRobot) -> list[float]:
    return [
        call["args"]["turn_deg"] for call in robot.calls if call["verb"] == "base.move_relative"
    ]


def fails(call, *args) -> StepError:
    with pytest.raises(StepError) as raised:
        call(*args)
    return raised.value


# -- recall_object and list_room_objects ------------------------------------------------


def test_recall_object_gives_where_each_was_seen_and_never_its_position():
    robot, ctx = remembering(
        recalled("18"),
        recalled("6", room=None, on=None, seen_s_ago=None, x=None, y=None),
    )

    assert recall_object(ctx, "cup") == {
        "objects": [
            {
                "id": "18",
                "name": "cup",
                "category": "dish",
                "room": "kitchen room",
                "on": "table 15",
                "near_place": "kitchen table",
                "last_seen_s_ago": 42,
            },
            {
                "id": "6",
                "name": "cup",
                "category": "dish",
                "room": None,
                "on": None,
                "near_place": None,
                "last_seen_s_ago": None,
            },
        ],
        "complete": True,
    }
    # One more than the list holds, to tell whether the list is all of them.
    assert ctx.object_memory.calls == [("search", "cup", RECALL_LIMIT + 1, None)]
    # Only the object with a position reads the places, and nothing moves.
    assert base_verbs(robot) == ["base.places"]


@pytest.mark.parametrize(("count", "complete"), [(RECALL_LIMIT, True), (RECALL_LIMIT + 1, False)])
def test_recall_object_says_whether_its_list_is_every_match(count, complete):
    _, ctx = remembering(*(recalled(str(n), x=None, y=None) for n in range(1, count + 1)))

    answer = recall_object(ctx, "")

    assert [found["id"] for found in answer["objects"]] == [
        str(n) for n in range(1, RECALL_LIMIT + 1)
    ]
    assert answer["complete"] is complete


def test_list_room_objects_gives_the_room_its_objects_and_whether_that_is_all():
    _, ctx = remembering(
        recalled("18"), recalled("16", "sofa", room="living room", on=None, x=10.0, y=2.5)
    )

    assert list_room_objects(ctx, "living room") == {
        "room": "living room",
        "objects": [
            {
                "id": "16",
                "name": "sofa",
                "category": "dish",
                "room": "living room",
                "on": None,
                "near_place": "sofa",
                "last_seen_s_ago": 42,
            }
        ],
        "complete": True,
    }
    assert ctx.object_memory.calls == [("room", "living room", ROOM_LIMIT)]


def test_the_object_memory_replies_are_json_able():
    _, ctx = remembering(recalled("18"), recalled("6", x=None, y=None, seen_s_ago=None))
    replies = [
        recall_object(ctx, ""),
        list_room_objects(ctx, "kitchen room"),
        go_to_object(ctx, "18"),
    ]

    assert json.loads(json.dumps(replies, allow_nan=False)) == replies


# -- go_to_object -------------------------------------------------------------------------


def test_go_to_object_drives_to_the_nearest_place_and_says_it_is_ahead():
    robot, ctx = remembering(recalled("18", x=4.0, y=0.0))

    assert go_to_object(ctx, "18") == (
        "Arrived at kitchen table, the named place nearest the cup 18, and it is straight "
        "ahead. Walkie has not looked for it yet: call find_objects to see whether it is "
        "still there."
    )
    assert robot.snapshot()["at"] == "kitchen table"
    assert turns(robot) == []


@pytest.mark.parametrize(
    ("y", "words", "turn"),
    [(1.0, "turned 90 degrees left", 90.0), (-1.0, "turned 90 degrees right", -90.0)],
)
def test_go_to_object_turns_to_face_the_object_once_there(y, words, turn):
    robot, ctx = remembering(recalled("18", x=3.0, y=y))

    reply = go_to_object(ctx, "18")

    assert f"the named place nearest the cup 18, and {words} to face it." in reply
    assert turns(robot) == [pytest.approx(turn)]
    assert math.degrees(robot.pose().yaw) == pytest.approx(turn)


@pytest.mark.parametrize(
    ("name", "label", "word"),
    [
        # A seeded stool's official name is not a word the detector uses (SG-27).
        ("stool", "chair", ' The camera calls it "chair", so call find_objects("chair").'),
        ("Cup", "cup", ""),
        ("cup", "", ""),
    ],
)
def test_go_to_object_names_the_camera_s_word_when_the_name_is_not_one(name, label, word):
    _, ctx = remembering(recalled("18", name, label=label, x=4.0, y=0.0))

    assert go_to_object(ctx, "18").endswith(f"to see whether it is still there.{word}")


def test_an_object_with_no_position_names_the_camera_s_word_with_its_room():
    _, ctx = remembering(recalled("18", "stool", label="chair", x=None, y=None))

    assert fails(go_to_object, ctx, "18").message == (
        "Walkie does not know where the stool 18 is. Go to the kitchen room and call "
        'find_objects there. The camera calls it "chair", so call find_objects("chair").'
    )


def test_go_to_object_says_when_the_memory_followed_a_merge():
    _, ctx = remembering(recalled("18"), merged={"5": "18"})

    reply = go_to_object(ctx, " 5 ")

    assert reply.startswith("Object 5 is now remembered as 18. Arrived at kitchen table,")


def test_go_to_object_says_a_state_other_than_present():
    _, ctx = remembering(recalled("18", state="missing"))

    assert " Walkie remembers it as missing. Walkie has not looked" in go_to_object(ctx, "18")


@pytest.mark.parametrize("state", ["grasping", "held", "handed_over", "discarded"])
def test_an_object_that_was_moved_is_not_driven_to(state):
    robot, ctx = remembering(recalled("18", state=state))

    error = fails(go_to_object, ctx, "18")

    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.message == (
        f"Walkie remembers the cup 18 as {state.replace('_', ' ')}, so it is not where "
        "Walkie saw it."
    )
    assert base_verbs(robot) == []


@pytest.mark.parametrize(
    ("room", "hint"),
    [("kitchen room", " Go to the kitchen room and call find_objects there."), (None, "")],
)
def test_an_object_with_no_position_is_not_driven_to(room, hint):
    robot, ctx = remembering(recalled("18", room=room, x=None, y=None))

    error = fails(go_to_object, ctx, "18")

    assert (error.code, error.message) == (
        ErrorCode.PRECONDITION_FAILED,
        f"Walkie does not know where the cup 18 is.{hint}",
    )
    assert base_verbs(robot) == []


def test_an_object_with_no_named_place_near_it_is_not_driven_to():
    # The room's own place is 0.5 m away, but a room is never a place to approach.
    robot, ctx = remembering(recalled("18", x=0.5, y=0.0))

    error = fails(go_to_object, ctx, "18")

    assert (error.code, error.message) == (
        ErrorCode.PRECONDITION_FAILED,
        f"Walkie knows no named place within {APPROACH_PLACE_MAX_M:g} m of the cup 18. Go to "
        "the kitchen room and call find_objects there.",
    )
    assert base_verbs(robot) == ["base.places"]


def test_a_place_in_the_object_s_room_comes_before_a_nearer_one():
    # The kitchen door is 1.5 m from the object and the sofa 1.7 m, but the object was
    # seen in the living room.
    _, ctx = remembering(recalled("6", room="living room", x=8.4, y=1.4))

    assert go_to_object(ctx, "6").startswith("Arrived at sofa,")
    assert recall_object(ctx, "cup")["objects"][0]["near_place"] == "sofa"


@pytest.mark.parametrize("verb", ["base.move_relative", "base.pose"])
def test_a_turn_that_fails_after_arriving_is_said_and_the_step_succeeds(verb):
    robot, ctx = remembering(recalled("18", x=3.0, y=1.0))
    robot.faults[verb] = StepError(ErrorCode.UNAVAILABLE, "The base is not answering.")

    reply = go_to_object(ctx, "18")

    assert (
        "the named place nearest the cup 18, but could not turn to face it: The base is not "
        "answering. Walkie has not looked"
    ) in reply
    assert robot.snapshot()["at"] == "kitchen table"


@pytest.mark.parametrize("verb", ["base.move_relative", "base.pose"])
def test_a_stop_during_the_turn_is_raised(verb):
    robot, ctx = remembering(recalled("18", x=3.0, y=1.0))
    stop = StepError(ErrorCode.CANCELLED, "Cancelled the turn: a stop is in effect.")
    robot.faults[verb] = stop

    assert fails(go_to_object, ctx, "18") is stop


def test_a_failed_drive_is_raised_and_nothing_turns():
    robot, ctx = remembering(recalled("18", x=3.0, y=1.0))
    failure = StepError(ErrorCode.SKILL_FAILED, "Navigation failed for the drive.")
    robot.faults["base.go_to"] = failure

    assert fails(go_to_object, ctx, "18") is failure
    assert turns(robot) == []


def test_an_id_the_memory_does_not_know_is_not_found():
    robot, ctx = remembering(recalled("18"))

    error = fails(go_to_object, ctx, "99")

    assert (error.code, error.message) == (ErrorCode.NOT_FOUND, "Walkie remembers no object 99.")
    assert base_verbs(robot) == []


# -- Without an object memory -------------------------------------------------------------


def test_the_object_memory_atomics_are_unavailable_without_an_object_memory():
    robot = FakeRobot(WORLD)
    assert isinstance(robot.ctx.object_memory, UnavailableObjectMemory)
    calls = [
        lambda: recall_object(robot.ctx, "cup"),
        lambda: list_room_objects(robot.ctx, "kitchen room"),
        lambda: go_to_object(robot.ctx, "18"),
    ]
    for call in calls:
        with pytest.raises(StepError) as raised:
            call()
        assert (raised.value.code, raised.value.message) == (
            ErrorCode.UNAVAILABLE,
            NO_OBJECT_MEMORY,
        )
    assert base_verbs(robot) == []
