"""FakeRobot: every capability verb over a fake world, with no ROS.

The same class is Mahoraga's FakeCtx, so main's real atomics run in unit tests
and in the fake-world gym. It takes a world description, which is plain
JSON-able data, and chula_pat14_6() builds one for the real arena's names. A
builder that seeds a feasible world for one generated command belongs to the
gym. Verbs call the same guards as the ROS-backed ones, so the fake refuses
what the robot refuses.
"""

import copy
import functools
import inspect
import math
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any, Concatenate, ParamSpec, TypeVar

from walkie_agent.capabilities import Ctx, ObjectSeen, PersonSeen, Pose2D
from walkie_agent.capabilities.guards import (
    CancelToken,
    check_relative_move,
    relative_goal,
    require_timeout,
)
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import Location

P = ParamSpec("P")
R = TypeVar("R")

# Rooms and locations of chula_pat14_6, from walkie-ros
# origin/feat/chula-pat14-6-simulation, simulation/mujoco/worlds/chula_pat14_6/
# authoring/source/30_16102026_ChulaPat14_6_world.toml. The world names its
# rooms living_room and kitchen_room; the GPSR command generator's data spells
# them with a space. "entrance" is its [doors] entrance_door. "trash" and
# "instruction point" are invented: the arena data has neither, and generated
# commands assume both (decision 26).
CHULA_PAT14_6_PLACES: dict[str, dict[str, Any]] = {
    "living room": {"x": 1.7629, "y": -1.483, "yaw": 0.0, "room": "living room"},
    "kitchen room": {"x": 1.3176, "y": -4.0123, "yaw": -1.5605, "room": "kitchen room"},
    "shelf": {"x": 2.4868, "y": -6.4086, "yaw": -1.5697, "room": "kitchen room"},
    "bar": {"x": 0.2148, "y": -6.3009, "yaw": -1.5855, "room": "kitchen room"},
    "table": {"x": 1.7023, "y": -3.957, "yaw": -1.556, "room": "kitchen room"},
    "tv": {"x": 1.6071, "y": -1.6133, "yaw": 3.1328, "room": "living room"},
    "sofa": {"x": 2.252, "y": -1.4753, "yaw": 0.0, "room": "living room"},
    "entrance": {"x": 1.9935, "y": -0.0069, "yaw": -3.1227, "room": "living room"},
    "trash": {"x": 3.0, "y": -5.5, "yaw": 0.0, "room": "kitchen room"},
    "instruction point": {"x": 2.8, "y": -2.5, "yaw": 3.14, "room": "living room"},
}
# The placement locations of the arena's GPSR data, where objects start.
SURFACES = ("shelf", "bar", "table", "sofa")
ROOMS = ("living room", "kitchen room")

# The arena's six objects and six invented ones. All sizes and weights are
# invented. Each measure is unique, so a "biggest" or "thinnest" command has
# one right answer on any surface.
OBJECTS: dict[str, dict[str, Any]] = {
    "apple": {"category": "fruit", "size_cm": 8, "weight_g": 180, "thickness_cm": 8},
    "banana": {"category": "fruit", "size_cm": 19, "weight_g": 120, "thickness_cm": 4},
    "orange": {"category": "fruit", "size_cm": 7, "weight_g": 150, "thickness_cm": 7.5},
    "mug": {"category": "dish", "size_cm": 10, "weight_g": 300, "thickness_cm": 9},
    "red plate": {"category": "dish", "size_cm": 25, "weight_g": 400, "thickness_cm": 2},
    "bowl": {"category": "dish", "size_cm": 15, "weight_g": 250, "thickness_cm": 5.5},
    "chips bag": {"category": "snack", "size_cm": 30, "weight_g": 60, "thickness_cm": 6},
    "pringles can": {"category": "snack", "size_cm": 23, "weight_g": 190, "thickness_cm": 7},
    "cookies box": {"category": "snack", "size_cm": 20, "weight_g": 230, "thickness_cm": 4.5},
    "water bottle": {"category": "drink", "size_cm": 24, "weight_g": 520, "thickness_cm": 6.5},
    "cola can": {"category": "drink", "size_cm": 12, "weight_g": 350, "thickness_cm": 6.6},
    "milk carton": {"category": "drink", "size_cm": 21, "weight_g": 1050, "thickness_cm": 7.2},
}
PLURALS = {"fruit": "fruits", "dish": "dishes", "snack": "snacks", "drink": "drinks"}
NAMES = ("Adel", "Angel", "Axel", "Charlie", "Jane", "Jules", "Morgan", "Paris", "Robin", "Simone")
POSES = ("sitting person", "standing person", "lying person")
GESTURES = (
    "waving person",
    "person raising their left arm",
    "person raising their right arm",
    "person pointing to the left",
    "person pointing to the right",
)

# After a relative move the robot still counts as at its place within this
# distance, which is enough for a small approach or a turn on the spot.
NEAR_PLACE_M = 1.0
# An object's "at" while the robot holds it.
HELD = "robot"


def chula_pat14_6(seed: int = 0) -> dict[str, Any]:
    """A world description of the arena: seeded clutter, two people per room and the operator."""
    rng = random.Random(seed)
    objects = {
        f"{name.replace(' ', '_')}_{n + 1}": {"name": name, **item, "at": rng.choice(SURFACES)}
        for name, item in OBJECTS.items()
        for n in range(rng.randint(0, 2))
    }
    rooms = [CHULA_PAT14_6_PLACES["instruction point"]["room"], *ROOMS, *ROOMS]
    ids = ["operator", *(f"person_{n + 1}" for n in range(len(rooms) - 1))]
    people = {
        person: {
            "name": name,
            "room": room,
            "pose": rng.choice(POSES),
            "gesture": rng.choice(GESTURES),
        }
        for person, room, name in zip(ids, rooms, rng.sample(NAMES, len(rooms)))
    }
    return {
        "places": copy.deepcopy(CHULA_PAT14_6_PLACES),
        "objects": objects,
        "people": people,
        "start": "instruction point",
    }


def _verb(
    name: str,
) -> Callable[
    [Callable[Concatenate["FakeRobot", P], R]], Callable[Concatenate["FakeRobot", P], R]
]:
    """Record every call of a verb, with its outcome, and raise its fault if one is set."""

    def decorate(method: Callable[Concatenate["FakeRobot", P], R]):
        signature = inspect.signature(method)

        @functools.wraps(method)
        def run(self: "FakeRobot", *args: P.args, **kwargs: P.kwargs) -> R:
            given = signature.bind(self, *args, **kwargs).arguments
            logged = {key: _json_safe(value) for key, value in given.items() if key != "self"}
            entry: dict[str, Any] = {"verb": name, "args": logged, "error": None}
            self.calls.append(entry)
            try:
                if name in self.faults:
                    raise self.faults[name]
                return method(self, *args, **kwargs)
            except StepError as error:
                entry["error"] = str(error.code)
                raise

        return run

    return decorate


class FakeRobot:
    """Implements Base, Speech, Vision and Arms over one world description.

    One object holds all four verb sets, since their names do not clash; `ctx`
    is the Ctx that atomics take. `world=None` builds chula_pat14_6(seed).
    `drive_s` makes a drive take real seconds, for stop tests. `calls` logs
    every verb call, `said` every line, and `faults` maps a verb name such as
    "base.go_to" to the StepError that each call to it raises.
    """

    def __init__(
        self,
        world: Mapping[str, Any] | None = None,
        *,
        seed: int = 0,
        drive_s: float = 0.0,
    ) -> None:
        described = chula_pat14_6(seed) if world is None else copy.deepcopy(dict(world))
        self._places: dict[str, dict[str, Any]] = described["places"]
        self._objects: dict[str, dict[str, Any]] = described["objects"]
        self._people: dict[str, dict[str, Any]] = described["people"]
        self._at: str | None = described["start"]
        self._pose = self._pose_of(described["start"])
        self.drive_s = drive_s
        self.cancel = CancelToken()
        self.calls: list[dict[str, Any]] = []
        self.said: list[str] = []
        self.faults: dict[str, StepError] = {}
        self._stopped = threading.Event()
        self.ctx = Ctx(cancel=self.cancel, base=self, speech=self, vision=self, arms=self)

    def snapshot(self) -> dict[str, Any]:
        """The state a checker reads, as JSON-able data that shares nothing with the robot."""
        return copy.deepcopy(
            {
                "at": self._at,
                "room": self._room(),
                "pose": asdict(self._pose),
                "holding": self._holding(),
                "places": self._places,
                "objects": self._objects,
                "people": self._people,
                "said": self.said,
                "calls": self.calls,
            }
        )

    # ctx.base

    @_verb("base.places")
    def places(self) -> Mapping[str, Location]:
        return {
            name: Location(
                name,
                place["x"],
                place["y"],
                place["yaw"],
                "A room." if place["room"] == name else f"In the {place['room']}.",
            )
            for name, place in self._places.items()
        }

    @_verb("base.go_to")
    def go_to(self, place: str, *, timeout_s: float = 120.0) -> Pose2D:
        name = self._known_place(place)
        return self._drive(f"the drive to {name}", self._pose_of(name), name, timeout_s)

    @_verb("base.move_relative")
    def move_relative(
        self,
        forward_m: float = 0.0,
        left_m: float = 0.0,
        turn_deg: float = 0.0,
        *,
        timeout_s: float = 60.0,
    ) -> Pose2D:
        check_relative_move(forward_m, left_m, turn_deg)
        goal = relative_goal(self._pose, forward_m, left_m, turn_deg)
        return self._drive("the move", goal, self._place_after(goal), timeout_s)

    @_verb("base.stop")
    def stop(self) -> None:
        self._stopped.set()

    @_verb("base.pose")
    def pose(self) -> Pose2D:
        return self._pose

    # ctx.speech

    @_verb("speech.say")
    def say(self, text: str, *, timeout_s: float = 10.0) -> None:
        require_timeout(timeout_s)
        if not text.strip():
            raise StepError(ErrorCode.REFUSED, "Nothing was said: the text is blank.")
        self.said.append(text)

    # ctx.vision

    @_verb("vision.objects")
    def objects(self, query: str = "", *, timeout_s: float = 5.0) -> list[ObjectSeen]:
        self._begin("looking for objects", timeout_s)
        return [
            ObjectSeen(
                id=object_id,
                name=item["name"],
                category=item["category"],
                place=item["at"],
                size_cm=item.get("size_cm"),
                weight_g=item.get("weight_g"),
                thickness_cm=item.get("thickness_cm"),
            )
            for object_id, item in sorted(self._objects.items())
            if self._in_view(item["at"])
            and _matches(
                query,
                item["name"],
                item["category"],
                PLURALS.get(item["category"], f"{item['category']}s"),
            )
        ]

    @_verb("vision.people")
    def people(self, query: str = "", *, timeout_s: float = 5.0) -> list[PersonSeen]:
        self._begin("looking for people", timeout_s)
        room = self._room()
        return [
            PersonSeen(person_id, who["name"], who["pose"], who["gesture"], who["room"])
            for person_id, who in sorted(self._people.items())
            if who["room"] == room and _matches(query, who["name"], who["pose"], who["gesture"])
        ]

    # ctx.arms

    @_verb("arms.pick")
    def pick(self, object_id: str, *, timeout_s: float = 60.0) -> str:
        self._begin(f"the pick of {object_id}", timeout_s)
        item = self._objects.get(object_id)
        if item is None:
            raise StepError(
                ErrorCode.NOT_FOUND, f"No object '{object_id}'. Call find_objects first."
            )
        held = self._holding()
        if held is not None:
            raise StepError(
                ErrorCode.PRECONDITION_FAILED,
                f"The hand already holds {held}. Place it or hand it over first.",
            )
        if not self._in_view(item["at"]):
            raise StepError(
                ErrorCode.PRECONDITION_FAILED,
                f"{object_id} is not in reach here. Go to where it was seen first.",
            )
        item["at"] = HELD
        return object_id

    @_verb("arms.place_on")
    def place_on(self, place: str, *, timeout_s: float = 60.0) -> str:
        self._begin(f"the placing on {place}", timeout_s)
        name = self._known_place(place)
        if self._places[name]["room"] == name:
            raise StepError(
                ErrorCode.REFUSED, f"{name} is a room. Place objects on a location in it."
            )
        held = self._require_held()
        if self._at != name:
            raise StepError(
                ErrorCode.PRECONDITION_FAILED,
                f"Walkie is not at {name}. Call go_to('{name}') first.",
            )
        self._objects[held]["at"] = name
        return held

    @_verb("arms.hand_over")
    def hand_over(self, person_id: str, *, timeout_s: float = 60.0) -> str:
        self._begin(f"the hand-over to {person_id}", timeout_s)
        who = self._people.get(person_id)
        if who is None:
            raise StepError(
                ErrorCode.NOT_FOUND, f"No person '{person_id}'. Call find_people first."
            )
        held = self._require_held()
        if who["room"] != self._room():
            raise StepError(
                ErrorCode.PRECONDITION_FAILED,
                f"{person_id} is not in this room. Go to them and call find_people first.",
            )
        self._objects[held]["at"] = person_id
        return held

    # Shared by the verbs.

    def _begin(self, what: str, timeout_s: float) -> None:
        """The guards of a verb that does not wait in the fake: a bound, and the latch."""
        require_timeout(timeout_s)
        self.cancel.check(what)

    def _drive(self, what: str, goal: Pose2D, at: str | None, timeout_s: float) -> Pose2D:
        """Wait out one drive of `drive_s`, which a stop or the token ends. Nothing moves until then."""
        self._stopped.clear()
        started = time.monotonic()
        self.cancel.wait_for(
            lambda: self._stopped.is_set() or time.monotonic() - started >= self.drive_s,
            timeout_s,
            what,
        )
        if self._stopped.is_set():
            raise StepError(ErrorCode.CANCELLED, f"The base stopped {what}.")
        self._pose, self._at = goal, at
        return goal

    def _known_place(self, place: str) -> str:
        name = place.strip().lower()
        if name not in self._places:
            known = ", ".join(self._places)
            raise StepError(
                ErrorCode.NOT_FOUND, f"Unknown place '{place}'. Known places: {known}."
            )
        return name

    def _pose_of(self, name: str) -> Pose2D:
        place = self._places[name]
        return Pose2D(place["x"], place["y"], place["yaw"])

    def _place_after(self, goal: Pose2D) -> str | None:
        """The place the robot still counts as at after a relative move, if any."""
        if self._at is None:
            return None
        place = self._places[self._at]
        return self._at if self._distance(place, goal) <= NEAR_PLACE_M else None

    def _room(self) -> str:
        """The room of the robot's place, or of the nearest place once it left its place."""
        if self._at is not None:
            return self._places[self._at]["room"]
        return min(self._places.values(), key=lambda place: self._distance(place, self._pose))[
            "room"
        ]

    def _in_view(self, at: str) -> bool:
        """Whether something at `at` is in view: at the robot's place, or in the room it stands at."""
        if self._at is None or at not in self._places:
            return False
        standing_at_room = self._places[self._at]["room"] == self._at
        return at == self._at or (standing_at_room and self._places[at]["room"] == self._at)

    def _holding(self) -> str | None:
        return next((key for key, item in self._objects.items() if item["at"] == HELD), None)

    def _require_held(self) -> str:
        held = self._holding()
        if held is None:
            raise StepError(ErrorCode.PRECONDITION_FAILED, "The hand is empty. Pick something first.")
        return held

    @staticmethod
    def _distance(place: Mapping[str, Any], pose: Pose2D) -> float:
        return math.hypot(place["x"] - pose.x, place["y"] - pose.y)


def _matches(query: str, *fields: str | None) -> bool:
    wanted = query.strip().lower()
    return not wanted or wanted in (str(field).lower() for field in fields)


def _json_safe(value: Any) -> Any:
    """A refused NaN or infinity stays in the call log as text, since JSON has no such number."""
    return repr(value) if isinstance(value, float) and not math.isfinite(value) else value
