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
import hashlib
import inspect
import math
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Concatenate, ParamSpec, TypeVar

from walkie_agent.arm import answers
from walkie_agent.arm.poses import ARMS, BOTH, ArmPoses
from walkie_agent.arm.results import DriveCheck, GripperResult
from walkie_agent.arm.state import ArmsState, ArmState
from walkie_agent.capabilities import (
    NO_ARMS,
    Arm,
    ArmPose,
    ArmsMoved,
    ArmsReading,
    ArmsReport,
    Ctx,
    Gripper,
    GripperMoved,
    GripperReading,
    GripperState,
    HandPose,
    ObjectSeen,
    PersonSeen,
    Pose2D,
    Recognized,
    Remembered,
    Team,
)
from walkie_agent.capabilities.guards import (
    CancelToken,
    check_relative_move,
    is_finite_number,
    relative_goal,
    require_person_details,
    require_text,
    require_timeout,
)
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import Location
from walkie_agent.perception.geometry import wrap_deg

P = ParamSpec("P")
R = TypeVar("R")

# Rooms and locations of chula_pat14_6, from walkie-ros
# origin/feat/chula-pat14-6-simulation, simulation/mujoco/worlds/chula_pat14_6/
# authoring/source/30_16102026_ChulaPat14_6_world.toml. The world names its
# rooms living_room and kitchen_room; the GPSR command generator's data spells
# them with a space. "entrance" is its [doors] entrance_door. "trash" and
# "instruction point" are invented: the arena data has neither, and generated
# commands assume both (decision 26). Their poses are where Nav2 finished a goal
# in the MuJoCo world (decision G4-6).
CHULA_PAT14_6_PLACES: dict[str, dict[str, Any]] = {
    "living room": {"x": 1.7629, "y": -1.483, "yaw": 0.0, "room": "living room"},
    "kitchen room": {"x": 1.3176, "y": -4.0123, "yaw": -1.5605, "room": "kitchen room"},
    "shelf": {"x": 2.4868, "y": -6.4086, "yaw": -1.5697, "room": "kitchen room"},
    "bar": {"x": 0.2148, "y": -6.3009, "yaw": -1.5855, "room": "kitchen room"},
    "table": {"x": 1.7023, "y": -3.957, "yaw": -1.556, "room": "kitchen room"},
    "tv": {"x": 1.6071, "y": -1.6133, "yaw": 3.1328, "room": "living room"},
    "sofa": {"x": 2.252, "y": -1.4753, "yaw": 0.0, "room": "living room"},
    "entrance": {"x": 1.9935, "y": -0.0069, "yaw": -3.1227, "room": "living room"},
    "trash": {"x": 2.75, "y": -4.2, "yaw": 0.0, "room": "kitchen room"},
    "instruction point": {"x": 2.2, "y": -2.45, "yaw": 3.14, "room": "living room"},
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

# The team of config.toml's [team] table. The fake clock keeps Asia/Bangkok's
# offset, which has no daylight saving time, and draws a minute of FACTS_YEAR.
TEAM = {"name": "EIC", "country": "Thailand", "affiliation": "Chulalongkorn University"}
FACTS_YEAR = 2026
FACTS_OFFSET = timezone(timedelta(hours=7))

# After a relative move the robot still counts as at its place within this
# distance, which is enough for a small approach or a turn on the spot.
NEAR_PLACE_M = 1.0
# An object's "at" while the robot holds it.
HELD = "robot"
# The verbs whose records keep the ids they returned.
LOOKUPS = frozenset({"vision.objects", "vision.people"})
# The pose both arms start in, which every drive refusal names as its example.
ARMS_START_POSE = answers.SAFE_EXAMPLE
# The commander's status for every arm and gripper goal, all of which succeed in the fake.
ARM_GOAL_STATUS = "Success"
# Its status for a whole-body reach, which holds the base still.
WHOLE_BODY_STATUS = "Success (base fixed)"
# The finger travel, in metres, where a close of a gripper in FakeRobot.grasping stops:
# on an object, inside the commander's default grasp band.
HELD_FINGER_M = 0.02


def fake_facts(seed: int = 0) -> dict[str, Any]:
    """The facts of one fake world: a clock fixed at a seeded minute of FACTS_YEAR, and TEAM.

    It draws from its own stream, so the clutter and the people of a seed do
    not change when this does. `now` is ISO 8601 text with its UTC offset.
    """
    rng = random.Random(f"facts:{seed}")
    first = datetime(FACTS_YEAR, 1, 1, tzinfo=FACTS_OFFSET)
    days = (datetime(FACTS_YEAR + 1, 1, 1, tzinfo=FACTS_OFFSET) - first).days
    now = first + timedelta(days=rng.randrange(days), minutes=rng.randrange(24 * 60))
    return {"now": now.isoformat(), "team": dict(TEAM)}


def person_heading_deg(person_id: str) -> float:
    """The map-frame direction of a person with no "heading_deg", in degrees in [-180, 180).

    From a hash of the id rather than the world's rng, so the people and the clutter
    of every seed stay as they were before people had a direction.
    """
    digest = hashlib.sha256(f"heading:{person_id}".encode()).digest()
    return round(int.from_bytes(digest[:4], "big") / 2**32 * 360.0 - 180.0, 1)


def chula_pat14_6(seed: int = 0) -> dict[str, Any]:
    """A world description of the arena: seeded clutter, two people per room, the operator and the facts."""
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
        "facts": fake_facts(seed),
    }


def _verb(
    name: str,
) -> Callable[
    [Callable[Concatenate["FakeRobot", P], R]], Callable[Concatenate["FakeRobot", P], R]
]:
    """Record every call of a verb, with its outcome, and raise its fault if one is set.

    A record also holds where the robot was when the call ended, and the ids a
    lookup returned, so a checker reads what happened and rebuilds nothing.
    """

    def decorate(method: Callable[Concatenate["FakeRobot", P], R]):
        signature = inspect.signature(method)

        @functools.wraps(method)
        def run(self: "FakeRobot", *args: P.args, **kwargs: P.kwargs) -> R:
            given = signature.bind(self, *args, **kwargs).arguments
            logged = {key: _json_safe(value) for key, value in given.items() if key != "self"}
            entry: dict[str, Any] = {"verb": name, "args": logged, "error": None, **self._where()}
            self.calls.append(entry)
            try:
                if name in self.faults:
                    raise self.faults[name]
                result = method(self, *args, **kwargs)
                if name in LOOKUPS:
                    entry["returned"] = [seen.id for seen in result]
                return result
            except StepError as error:
                entry["error"] = str(error.code)
                raise
            except Exception:
                entry["error"] = str(ErrorCode.INTERNAL)
                raise
            finally:
                entry.update(self._where())

        return run

    return decorate


def _check_world(world: Mapping[str, Any]) -> None:
    """Raise ValueError for a world description with a broken name or coordinate.

    A misspelled name does not fail by itself: the object or person just never
    comes into view, and the episode reads as the agent's failure.
    """
    places, people = world["places"], world["people"]
    rooms = {name for name, place in places.items() if place["room"] == name}
    holders = {*places, *people, HELD}
    problems = [f"start: unknown place {world['start']!r}"] if world["start"] not in places else []
    for name, place in places.items():
        if place["room"] not in rooms:
            problems.append(f"places[{name!r}].room: unknown room {place['room']!r}")
        problems += [
            f"places[{name!r}].{field}: {place[field]!r} is not a finite number"
            for field in ("x", "y", "yaw")
            if not is_finite_number(place[field])
        ]
    problems += [
        f"people[{person!r}].room: unknown room {who['room']!r}"
        for person, who in people.items()
        if who["room"] not in rooms
    ]
    problems += [
        f"people[{person!r}].heading_deg: {who['heading_deg']!r} is not a finite number"
        for person, who in people.items()
        if "heading_deg" in who and not is_finite_number(who["heading_deg"])
    ]
    problems += [
        f"objects[{object_id!r}].at: nothing called {item['at']!r}"
        for object_id, item in world["objects"].items()
        if item["at"] not in holders
    ]
    if world.get("facts") is not None:
        problems += _facts_problems(world["facts"])
    if problems:
        raise ValueError(f"Broken world: {'; '.join(problems)}.")


def _facts_problems(facts: Mapping[str, Any]) -> list[str]:
    """What is wrong with a world's facts: `now` must carry a UTC offset, and the team three texts."""
    problems = []
    try:
        if datetime.fromisoformat(facts["now"]).utcoffset() is None:
            problems.append(f"facts.now: {facts['now']!r} has no UTC offset")
    except (KeyError, TypeError, ValueError):
        problems.append(f"facts.now: {facts.get('now')!r} is not an ISO 8601 date and time")
    team = facts.get("team")
    problems += [
        f"facts.team.{key}: not a non-blank text"
        for key in TEAM
        if not isinstance(team, Mapping)
        or not isinstance(team.get(key), str)
        or not team[key].strip()
    ]
    return problems


def _check_arm_poses(arm_poses: ArmPoses) -> None:
    """Raise ValueError unless both arms can start in ARMS_START_POSE: a table for each arm."""
    start = arm_poses.poses.get(ARMS_START_POSE)
    if start is None or start.arms != ARMS:
        raise ValueError(
            f"Broken arm poses: the arms start in {ARMS_START_POSE!r}, which needs a table "
            "for each arm."
        )


class FakeRobot:
    """Implements Base, Speech, Vision, Arms, Facts and Memory over one world description.

    One object holds all six verb sets, since their names do not clash; `ctx`
    is the Ctx that atomics take. `world=None` builds chula_pat14_6(seed), and a
    world with a broken name, coordinate or fact raises ValueError. A world with
    no "facts" has no fact source, so the facts verbs raise UNAVAILABLE.
    A person's bearing is their "heading_deg", a fixed direction in the map, or one
    from a hash of their id, less the robot's yaw, so a turn by it faces them.
    The memory keeps each person remembered by their world id, apart from the
    world's names that the people verb shows. recognize gives back the name and
    facts remembered, matched by face, and no name for anyone else; remembering a
    person again replaces what was kept, and forget() clears it all. The memory
    stays out of snapshot(), so a run that remembers nobody reads as before.

    The arms take named poses only with `arm_poses`, as load_arm_poses() gives
    them; without, as in Mahoraga, poses, read, move_to and gripper raise
    UNAVAILABLE as UnavailableArms does, state() says the arms are unknown, and
    nothing checks the arms before a drive. With them, the poses offered are the
    simulation's, both arms start in ARMS_START_POSE and both grippers open, a
    move or a gripper command succeeds at once and lands exactly, and a close
    closes empty, since the fake's grippers meet no object, unless the gripper's
    arm is in `grasping`: its close then holds something. A reach succeeds at once
    and leaves its arm in no named pose, so not travel-safe. go_to
    and move_relative then refuse to start, as the robot's drive check does,
    while an arm is in a pose that is not travel-safe; arm_poses lacking a table
    of ARMS_START_POSE for each arm raises ValueError. pick, place_on and
    hand_over move the world's objects either way, and never the arms or the
    grippers. The arm state stays out of snapshot().

    `drive_s` makes a drive take real seconds, for stop tests. `calls` logs
    every verb call with the robot's place and room as it ended and, for a
    lookup, the ids returned. `said` logs every line, and `faults` maps a verb
    name such as "base.go_to" to the StepError that each call to it raises.
    base.stop takes no fault, since it never raises, and neither does
    arms.state, which `calls` does not log either (state()).
    """

    def __init__(
        self,
        world: Mapping[str, Any] | None = None,
        *,
        seed: int = 0,
        drive_s: float = 0.0,
        arm_poses: ArmPoses | None = None,
    ) -> None:
        described = chula_pat14_6(seed) if world is None else copy.deepcopy(dict(world))
        _check_world(described)
        self._places: dict[str, dict[str, Any]] = described["places"]
        self._objects: dict[str, dict[str, Any]] = described["objects"]
        self._people: dict[str, dict[str, Any]] = described["people"]
        self._facts: dict[str, Any] | None = described.get("facts")
        self._at: str | None = described["start"]
        self._pose = self._pose_of(described["start"])
        self.drive_s = drive_s
        self.cancel = CancelToken()
        self.calls: list[dict[str, Any]] = []
        self.said: list[str] = []
        self.faults: dict[str, StepError] = {}
        self._stopped = threading.Event()
        # The name and facts of each person remembered, by world person id.
        self._remembered: dict[str, tuple[str, dict[str, str]]] = {}
        if arm_poses is not None:
            _check_arm_poses(arm_poses)
        self._arm_poses = arm_poses
        # The pose each arm is in, each gripper's finger travel in metres, and how
        # many commands each gripper took, as GripperResult.command counts them.
        self._arm_at: dict[str, str | None] = dict.fromkeys(ARMS, ARMS_START_POSE)
        # The grippers whose closes meet an object and hold it (FakeRobot's docstring).
        self.grasping: set[str] = set()
        self._finger_m: dict[str, float] = (
            {} if arm_poses is None else dict.fromkeys(ARMS, arm_poses.grippers.open_position)
        )
        self._gripper_commands = dict.fromkeys(ARMS, 0)
        self.ctx = Ctx(
            cancel=self.cancel,
            base=self,
            speech=self,
            vision=self,
            arms=self,
            facts=self,
            memory=self,
        )

    def snapshot(self) -> dict[str, Any]:
        """The state a checker reads, as JSON-able data that shares nothing with the robot.

        "facts" is the world's facts, or None for a world without them.
        """
        return copy.deepcopy(
            {
                "at": self._at,
                "room": self._room(),
                "pose": asdict(self._pose),
                "holding": self._holding(),
                "places": self._places,
                "objects": self._objects,
                "people": self._people,
                "facts": self._facts,
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
                place["room"],
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

    def stop(self) -> None:
        self._stopped.set()
        self.calls.append({"verb": "base.stop", "args": {}, "error": None, **self._where()})

    @_verb("base.pose")
    def pose(self, *, timeout_s: float = 1.0) -> Pose2D:
        require_timeout(timeout_s)
        return self._pose

    # ctx.speech

    @_verb("speech.say")
    def say(self, text: str, *, timeout_s: float = 10.0) -> None:
        require_timeout(timeout_s)
        require_text(text=text)
        if not text.strip():
            raise StepError(ErrorCode.REFUSED, "Nothing was said: the text is blank.")
        self.said.append(text)

    # ctx.vision

    @_verb("vision.objects")
    def objects(self, query: str = "", *, timeout_s: float = 5.0) -> list[ObjectSeen]:
        self._begin("looking for objects", timeout_s)
        require_text(query=query)
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
        require_text(query=query)
        room = self._room()
        return [
            PersonSeen(
                person_id,
                who["name"],
                who["pose"],
                who["gesture"],
                who["room"],
                self._bearing_deg(person_id, who),
            )
            for person_id, who in sorted(self._people.items())
            if who["room"] == room and _matches(query, who["name"], who["pose"], who["gesture"])
        ]

    # ctx.arms. The pose and gripper verbs answer in the robot's words
    # (walkie_agent.arm.answers), and move nothing until the token's commit.

    @_verb("arms.poses")
    def poses(self) -> Mapping[str, ArmPose]:
        # No token check, as on the robot: listing the poses moves nothing.
        return self._require_arms().offered_poses(sim=True)

    def state(self, *, read: bool | None = None) -> ArmsReport:
        """The arms as the fake knows them. Never raises; `read` changes nothing.

        The fake always knows where its arms are, so there is nothing to read and
        the arms are never unverified. Not logged in `calls`: it moves nothing and
        cannot fail, and the arm atomics call it to end nearly every answer, so a
        record would only bury the calls the LLM made.
        """
        return ArmsReport(self._arms_state())

    @_verb("arms.read")
    def read(self) -> ArmsReading:
        positions = self._require_arms().grippers
        self.cancel.check("the read of the arms")
        report = self.state()
        fingers = dict(self._finger_m)
        return ArmsReading(
            before=report,
            after=report,
            grippers=tuple(
                GripperReading(side, fingers[side], answers.reads_as(positions, fingers[side]))
                for side in ARMS
            ),
        )

    @_verb("arms.move_to")
    def move_to(self, pose: str, arm: Arm = "both") -> ArmsMoved:
        offered = self._require_arms().offered_poses(sim=True)
        found = answers.offered_pose(offered, pose, arm, self._arms_sentence)
        with self._arm_goal(f"the move to {found.name!r}"):
            for side in ARMS if arm == BOTH else (arm,):
                self._arm_at[side] = found.name
        return ArmsMoved(
            pose=found.name,
            arm=arm,
            status=ARM_GOAL_STATUS,
            max_error_rad=0.0,
            report=self.state(),
        )

    @_verb("arms.gripper")
    def gripper(self, arm: Gripper, state: GripperState) -> GripperMoved:
        position = self._require_arms().grippers.position(state)
        holds = state == "close" and arm in self.grasping
        if holds:
            # The fingers stop on the object, inside the commander's grasp band.
            position = HELD_FINGER_M
        with self._arm_goal(f"the {arm} gripper command"):
            # The count first, so an arm with no gripper changes nothing.
            self._gripper_commands[arm] += 1
            self._finger_m[arm] = position
            command = self._gripper_commands[arm]
        if state != "close":
            result = GripperResult(True, ARM_GOAL_STATUS, None, None, command)
            return GripperMoved(arm=arm, state=state, outcome="moved", result=result)
        if holds:
            # As on the robot, a close that stalls on an object reports success false.
            result = GripperResult(False, ARM_GOAL_STATUS, True, position, command)
            return GripperMoved(
                arm=arm, state=state, outcome="holding", result=result, finger_m=position
            )
        result = GripperResult(True, ARM_GOAL_STATUS, False, position, command)
        return GripperMoved(arm=arm, state=state, outcome="empty", result=result, finger_m=position)

    @_verb("arms.reach")
    def reach(
        self,
        arm: Gripper,
        pose: HandPose,
        *,
        whole_body: bool = False,
        cartesian: bool = False,
        touch: bool = False,
        timeout_s: float = 60.0,
    ) -> str:
        self._require_arms()
        answers.check_reach(pose, whole_body=whole_body, cartesian=cartesian, timeout_s=timeout_s)
        with self._arm_goal(f"the reach of the {arm} hand"):
            self._arm_at[arm] = None
        return WHOLE_BODY_STATUS if whole_body else ARM_GOAL_STATUS

    @_verb("arms.pick")
    def pick(self, object_id: str, *, timeout_s: float = 60.0) -> str:
        self._begin(f"the pick of {object_id}", timeout_s)
        require_text(object_id=object_id)
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
        require_text(person_id=person_id)
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

    # ctx.facts. Neither verb checks the token, as SystemFacts does not.

    @_verb("facts.now")
    def now(self) -> datetime:
        return datetime.fromisoformat(self._require_facts()["now"])

    @_verb("facts.team")
    def team(self) -> Team:
        team = self._require_facts()["team"]
        return Team(team["name"], team["country"], team["affiliation"])

    # ctx.memory

    @_verb("memory.remember_person")
    def remember_person(
        self,
        person_id: str,
        name: str,
        facts: Mapping[str, str] | None = None,
        *,
        timeout_s: float = 15.0,
    ) -> Remembered:
        what = f"remembering {person_id}"
        self._begin(what, timeout_s)
        require_text(person_id=person_id)
        name, kept = require_person_details(name, facts)
        self._require_person_here(person_id)
        with self.cancel.unless_cancelled(what):
            self._remembered[person_id] = (name, kept)
        return Remembered(person_id, name, dict(kept), faces=1, appearances=1)

    @_verb("memory.recognize")
    def recognize(self, person_id: str, *, timeout_s: float = 15.0) -> Recognized:
        self._begin(f"recognizing {person_id}", timeout_s)
        require_text(person_id=person_id)
        self._require_person_here(person_id)
        if person_id not in self._remembered:
            return Recognized(person_id, None, {}, None)
        name, facts = self._remembered[person_id]
        return Recognized(person_id, name, dict(facts), "face")

    def forget(self) -> None:
        """Forget every person remembered, as the robot does at the next task. Never raises."""
        self._remembered.clear()

    # Shared by the verbs.

    def _require_facts(self) -> dict[str, Any]:
        if self._facts is None:
            raise StepError(
                ErrorCode.UNAVAILABLE, "Walkie cannot read the time or its team: this world has no facts."
            )
        return self._facts

    def _begin(self, what: str, timeout_s: float) -> None:
        """The guards of a verb that does not wait in the fake: a bound, and the latch."""
        require_timeout(timeout_s)
        self.cancel.check(what)

    def _drive(self, what: str, goal: Pose2D, at: str | None, timeout_s: float) -> Pose2D:
        """Wait out one drive of `drive_s`, which a stop or the token ends. Nothing moves until then.

        With arm poses, the arm check (ADR-0025) comes first, after the bound and
        the latch as on the robot: answers.drive_verdict on the arms as the fake
        knows them, which raises PRECONDITION_FAILED while an arm is out. It logs
        no call of its own; the base verb's record gets its error.
        """
        require_timeout(timeout_s)
        self.cancel.check(what)
        if self._arm_poses is not None:
            answers.drive_verdict(lambda: DriveCheck(self._arms_state()), what)
        self._stopped.clear()
        started = time.monotonic()
        self.cancel.wait_for(
            lambda: self._stopped.is_set() or time.monotonic() - started >= self.drive_s,
            timeout_s,
            what,
        )
        if self._stopped.is_set():
            raise StepError(ErrorCode.CANCELLED, f"The base stopped {what}.")
        with self.cancel.unless_cancelled(what):
            self._pose, self._at = goal, at
        return goal

    def _require_arms(self) -> ArmPoses:
        """The arm poses, or StepError UNAVAILABLE without them, in UnavailableArms' words."""
        if self._arm_poses is None:
            raise StepError(ErrorCode.UNAVAILABLE, NO_ARMS)
        return self._arm_poses

    def _arms_state(self) -> ArmsState:
        """Both arms' poses, each travel-safe as its pose says; ArmsState() without arm poses."""
        if self._arm_poses is None:
            return ArmsState()
        poses = self._arm_poses.poses

        def arm(side: str) -> ArmState:
            # None after a reach: in no named pose, so not safe for driving.
            name = self._arm_at[side]
            return ArmState(pose=name, travel_safe=name is not None and poses[name].travel_safe)

        return ArmsState(left=arm("left"), right=arm("right"))

    def _arms_sentence(self) -> str:
        return self.state().sentence()

    @contextmanager
    def _arm_goal(self, what: str) -> Iterator[None]:
        """Commit the block, the change of one arm or gripper goal, unless a stop is in effect.

        As CancelToken.unless_cancelled, but its CANCELLED ends with the arm
        sentence, as every message of move_to and gripper does. An error the
        block raises passes through as it is.
        """
        with ExitStack() as stack:
            try:
                stack.enter_context(self.cancel.unless_cancelled(what))
            except StepError as error:
                raise answers.with_sentence(error, "", self._arms_sentence()) from None
            yield

    def _known_place(self, place: str) -> str:
        require_text(place=place)
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

    def _where(self) -> dict[str, str | None]:
        """Where a call record says the robot was. `at` is None once it has left its place."""
        return {"at": self._at, "room": self._room()}

    def _in_view(self, at: str) -> bool:
        """Whether something at `at` is in view: at the robot's place, or in the room it stands at."""
        if self._at is None or at not in self._places:
            return False
        standing_at_room = self._places[self._at]["room"] == self._at
        return at == self._at or (standing_at_room and self._places[at]["room"] == self._at)

    def _bearing_deg(self, person_id: str, who: Mapping[str, Any]) -> float:
        """The person's direction from the base's heading, so a turn by it faces them."""
        heading = who["heading_deg"] if "heading_deg" in who else person_heading_deg(person_id)
        # + 0.0 turns -0.0 into 0.0, so a person straight ahead never reads as "-0".
        return round(wrap_deg(heading - math.degrees(self._pose.yaw)), 1) + 0.0

    def _holding(self) -> str | None:
        return next((key for key, item in self._objects.items() if item["at"] == HELD), None)

    def _require_person_here(self, person_id: str) -> None:
        """Raise NOT_FOUND unless `person_id` is a person the people verb shows here."""
        who = self._people.get(person_id)
        if who is None:
            raise StepError(
                ErrorCode.NOT_FOUND, f"No person '{person_id}'. Call find_people first."
            )
        if who["room"] != self._room():
            raise StepError(
                ErrorCode.NOT_FOUND, f"{person_id} is not in view here. Call find_people first."
            )

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
    """An argument as the call log keeps it: a copy that JSON can hold.

    A refused NaN or infinity stays as text, since JSON has no such number, and so
    does a value of a type JSON lacks. A mapping, such as the facts of a person to
    remember, is copied with text keys, so the record shares nothing with the caller.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if value is None or isinstance(value, str | int):
        return value
    if isinstance(value, Mapping):
        return {
            key if isinstance(key, str) else repr(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return repr(value)
