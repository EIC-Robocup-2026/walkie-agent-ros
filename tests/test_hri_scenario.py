"""The HRI scenario: its rules on a fake clock, the default files, and the node over ROS.

HriScenario runs without ROS on a FakeClock, from the default scenario file
with the changes a test needs. The tests of the default files put
test_projection's simulated head camera at the poses of
config/locations.chula_pat14_6.toml and draw the scenario's scene with the
oracle's draw_detections, without noise, so they check what walkie-ros's
world chula_pat14_6 shows the agent. GraspParameter's own tests use fake
service clients whose requests the test answers, and a fake clock. The ROS
tests run HriScenarioNode on topics of their own, next to a node that plays
the agent and the simulation and a fake bimanual_commander that has only its
parameters. All of it runs on Fast DDS and a random domain (ADR-0019).
"""

import copy
import dataclasses
import json
import logging
import math
import random
import re
import threading
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rclpy
import yaml
from action_msgs.msg import GoalStatus, GoalStatusArray
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rai_interfaces.msg import HRIMessage
from rcl_interfaces.msg import ParameterType, ParameterValue, SetParametersResult
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_action_status_default
from rclpy.task import Future
from sensor_msgs.msg import CameraInfo, JointState
from std_msgs.msg import Empty, String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
from tf_transformations import euler_matrix, quaternion_from_euler
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetection, PersonDetectionArray

from conftest import PROJECT_ROOT
from test_oracle_perception import CAMERA_FRAME, QUIET, STAMP, STATIC_TF_QOS, camera_info, tf_msg
from test_people_atomics import wait_until
from test_projection import CX, HEIGHT, WIDTH, base_from_camera, sim_camera
from walkie_agent.arm.poses import (
    ARMS,
    GRASP_DETECT_MIN,
    GRIPPER_GROUPS,
    ArmPoseError,
    load_arm_poses,
    load_robot_model,
)
from walkie_agent.locations import load_locations
from walkie_agent.perception.geometry import MAP_FRAME, box_heading
from walkie_agent.perception.gestures import COCO_KEYPOINTS
from walkie_agent.perception.seating import (
    DEFAULT_SEATING_CONFIG,
    bbox_xyxy,
    hip_anchor,
    split_seat_regions,
)
from walkie_agent.perception.vision_feed import face_trusted
from walkie_agent.sim import hri_scenario
from walkie_agent.sim.hri_scenario import (
    ABSENT,
    AT_DOOR,
    BAG,
    BYSTANDER_ID,
    DRINK,
    FOLLOWING,
    FROM_HUMAN_QOS,
    GRASP_DETECT_MIN_PARAM,
    HANG,
    HUNG_VALUE,
    ITEM_FINISHED,
    ITEM_GONE,
    NAME,
    NO_PLACE,
    OTHER,
    RESTORE,
    ROBOT_FRAME,
    SEATED,
    TAKEN,
    TALKING,
    TO_HUMAN_QOS,
    TODOS_QOS,
    Answer,
    EventLog,
    GraspParameter,
    HriScenario,
    HriScenarioNode,
    Line,
    LoadedScenario,
    PersonState,
    Pose,
    ScenarioError,
    ScenarioSpec,
    ScenarioTopics,
    Step,
    bearing_deg,
    classify_line,
    load_scenario,
)
from walkie_agent.sim.oracle_perception import DETECTION_SCORE, KEYPOINT_SCORE, draw_detections
from walkie_agent.sim.projection import camera_position
from walkie_agent.sim.scene import DEFAULT_SEAT_TOP_M, FACE_LIMIT_DEG, MapBox, Scene
from walkie_agent.todo import TodoItem
from walkie_agent.tools.navigation import ROBOT_FRAME as NAVIGATION_ROBOT_FRAME

SCENARIOS = PROJECT_ROOT / "config" / "scenarios"
DEFAULT_FILE = SCENARIOS / "hri.yaml"
EARLY_BAG_FILE = SCENARIOS / "hri_early_bag.yaml"
LOCATIONS_FILE = PROJECT_ROOT / "config" / "locations.chula_pat14_6.toml"

# The poses of config/locations.chula_pat14_6.toml that the geometry below
# was computed at. A test keeps the file and these equal.
DOOR = Pose(2.35, -0.17, 3.01)
LIVING_ROOM = Pose(1.85, -1.71, 0.0)

# The hallway and the entrance of walkie-ros's world chula_pat14_6, in its
# map frame (simulation/mujoco/worlds/chula_pat14_6/ on
# feat/chula-pat14-6-simulation, adb9725). map/map.pgm is free at x 0.65 for
# y -0.32 to 0.32. In scene.xml the body chula_entrance_frame spans x 0.765 to
# 0.875 around x 0.82, and its posts leave the opening y -0.35 to 0.40.
HALLWAY_X = 0.65
HALLWAY_FREE_Y = (-0.32, 0.32)
FRAME_X = (0.765, 0.875)
FRAME_CENTRE_X = 0.82
FRAME_OPENING_Y = (-0.35, 0.40)

# A seat the world does not have, for the seat switch's move: an armchair in
# the living room's north-east corner, as a scenario file may add one.
ARMCHAIR = {
    "id": "armchair",
    "label": "armchair",
    "box": {"center": [3.05, -0.25, 0.425], "size": [0.70, 0.70, 0.85]},
    "facing": -1.5708,
}
# The todo item whose start fires the seat switch.
INTRODUCTION = "Introduce the guests to each other"
# The fake commander declares no grasp_detect_min.
UNDECLARED = "undeclared"

LEFT_HIP = COCO_KEYPOINTS.index("left_hip")
RIGHT_HIP = COCO_KEYPOINTS.index("right_hip")


@pytest.fixture(autouse=True)
def project_root(monkeypatch):
    """Run in the project, since a scenario file names its locations file relative to it."""
    monkeypatch.chdir(PROJECT_ROOT)


class FakeClock:
    """The scenario's clock: seconds that only the test moves."""

    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def deep_update(data: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """`data` with `changes` merged in: mappings merge key by key, anything else replaces."""
    merged = copy.deepcopy(data)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def make_spec(path: Path = DEFAULT_FILE, **changes: Any) -> ScenarioSpec:
    return ScenarioSpec.model_validate(deep_update(read_yaml(path), changes))


def make_scenario(path: Path = DEFAULT_FILE, **changes: Any) -> tuple[HriScenario, FakeClock]:
    """The scenario of `path` with `changes`, on a fake clock, with the robot at the door.

    The first step, which holds the events of the start, is dropped.
    """
    clock = FakeClock()
    spec = make_spec(path, **changes)
    scenario = HriScenario(spec, load_locations(spec.locations), clock)
    scenario.robot_pose(DOOR)
    return scenario, clock


def with_armchair(**changes: Any) -> list[dict[str, Any]]:
    """The default file's seats and the armchair with `changes`."""
    return [*read_yaml(DEFAULT_FILE)["seats"], {**ARMCHAIR, **changes}]


def facing(scenario: HriScenario, place: str, at: Pose = LIVING_ROOM) -> Pose:
    """The robot at `at`, turned straight to `place`."""
    target = scenario.places[place]
    return Pose(at.x, at.y, math.atan2(target.y - at.y, target.x - at.x))


def guide(scenario: HriScenario) -> Step:
    """Talk to the guest at the door and drive to the living room, so that they follow."""
    scenario.robot_said("What is your name?")
    return scenario.robot_pose(LIVING_ROOM)


def seat(scenario: HriScenario, place: str) -> Step:
    """Guide the guest at the door, offer them `place`, and go back to the door.

    Returns the offer's step.
    """
    guide(scenario)
    scenario.robot_pose(facing(scenario, place))
    step = scenario.robot_said("Please sit here.")
    scenario.robot_pose(DOOR)
    return step


def of_type(events, type_: str) -> list[dict[str, Any]]:
    return [event for event in events if event["type"] == type_]


def types(step: Step) -> list[str]:
    return [event["type"] for event in step.events]


def offer_of(step: Step) -> dict[str, Any]:
    """The step's seat_taken or bad_offer record."""
    [offer] = of_type(step.events, "seat_taken") + of_type(step.events, "bad_offer")
    return offer


def todos(*items: tuple[str, str, str]) -> list[dict[str, Any]]:
    """A todo list as the agent publishes it on /walkie/todos, from (id, content, status)."""
    return [
        TodoItem(id=item_id, content=content, active_form=content, status=status).model_dump()
        for item_id, content, status in items
    ]


def write_scenario(directory: Path, data: Any) -> Path:
    path = directory / "scenario.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# -- The scenario's rules, on a fake clock --------------------------------------------------


def test_the_host_sits_and_guest1_waits_at_the_guest_spot_from_the_start():
    clock = FakeClock()
    scenario = HriScenario(make_spec(), load_locations(LOCATIONS_FILE), clock)

    host, guest1 = of_type(scenario.tick().events, "person")

    assert host == {
        "type": "person",
        "t": clock.t,
        "person": "host",
        "name": None,
        "state": SEATED,
        "track_id": scenario.person("host").track_id,
        "place": "sofa.left",
        "x": 3.115,
        "y": -1.10,
    }
    assert guest1 == {
        "type": "person",
        "t": clock.t,
        "person": "guest1",
        "name": "Sam",
        "state": AT_DOOR,
        "track_id": scenario.person("guest1").track_id,
        "place": None,
        "x": 0.65,
        "y": 0.03,
    }
    assert scenario.person("guest2") == PersonState("guest2", "Ana", ABSENT, 0, None)

    scene = scenario.scene()
    [sofa] = scene.objects
    assert sofa.label == "sofa"
    assert sofa.box == MapBox((3.1675, -1.65, 0.4175), (0.575, 1.88, 0.835), 0.0)
    people = scenario.people_at()
    assert set(people) == {"host", "guest1"}
    assert len(scene.people) == 2
    # The seat, the host and guest 1 each have a track of their own; 0 is no track.
    tracks = {sofa.track_id, people["host"].track_id, people["guest1"].track_id}
    assert len(tracks) == 3 and 0 not in tracks
    host_seen, guest_seen = people["host"], people["guest1"]
    # The file gives the sofa its cushions' top; a seat without one gets the default.
    assert (host_seen.posture, host_seen.seat_top) == ("sitting", 0.4575)
    assert make_spec(seats=with_armchair()).places()["armchair"].seat_top == DEFAULT_SEAT_TOP_M
    assert host_seen.facing == pytest.approx(3.1416)
    assert (guest_seen.face, guest_seen.appearance, guest_seen.height) == ("guest1", "guest1", 1.70)
    # Before the robot's pose comes, guest 1 faces the door pose, where it will stand.
    assert guest_seen.facing == pytest.approx(math.atan2(DOOR.y - 0.03, DOOR.x - 0.65))


def test_the_scenario_needs_door_and_living_room():
    locations = load_locations(LOCATIONS_FILE)

    with pytest.raises(ScenarioError, match="no living_room"):
        HriScenario(make_spec(), {"door": locations["door"]}, FakeClock())


def test_name_and_drink_questions_get_the_guests_answers_after_the_delay():
    scenario, clock = make_scenario(answers={"mumble_once": None})
    t0 = clock.t
    text = "Hello, I am Walkie. What is your name?"

    step = scenario.robot_said(text)

    [line] = of_type(step.events, "robot_line")
    assert line == {
        "type": "robot_line",
        "t": t0,
        "text": text,
        "guest": "guest1",
        "question": True,
        "asks": [NAME],
        "offer": False,
        "pose": {"x": 2.35, "y": -0.17, "yaw": 3.01},
        "bearings_deg": {"host": line["bearings_deg"]["host"], "guest1": 0.83},
    }
    assert of_type(step.events, "person")[0]["state"] == TALKING
    assert scenario.tick(t0 + 0.99).answers == ()

    step = scenario.tick(t0 + 1.0)

    assert step.answers == (Answer("My name is Sam.", "guest1", text, t0, (NAME,), False),)
    assert of_type(step.events, "answer") == [
        {
            "type": "answer",
            "t": t0 + 1.0,
            "guest": "guest1",
            "text": "My name is Sam.",
            "to": text,
            "asked_t": t0,
            "asks": [NAME],
            "mumbled": False,
        }
    ]

    t1 = clock.advance(5.0)
    scenario.robot_said("What would you like to drink?")
    scenario.robot_said("Sorry, could you tell me your name and your favourite drink again?")
    answers = scenario.tick(t1 + 1.0).answers
    assert [(a.text, a.asks) for a in answers] == [
        ("I would like milk.", (DRINK,)),
        ("My name is Sam. I would like milk.", (NAME, DRINK)),
    ]


def test_the_mumble_answers_only_the_first_ask_of_its_question():
    scenario, clock = make_scenario()

    scenario.robot_said("What would you like to drink?")
    scenario.robot_said("What is your name?")
    scenario.robot_said("Sorry, what would you like to drink?")

    answers = scenario.tick(clock.t + 1.0).answers
    assert [(a.text, a.mumbled) for a in answers] == [
        ("Hmm.", True),
        ("My name is Sam.", False),
        ("I would like milk.", False),
    ]


@pytest.mark.parametrize(
    "text",
    ["Welcome! Please come in.", "I will hold out my hand for your bag.", "Follow me, please."],
)
def test_a_line_that_asks_nothing_gets_no_answer(text):
    scenario, clock = make_scenario()

    step = scenario.robot_said(text)

    assert types(step) == ["robot_line"]
    assert scenario.tick(clock.t + 5.0).answers == ()
    # Only a question starts the conversation.
    assert scenario.person("guest1").state == AT_DOOR


def test_the_bag_owners_request_hangs_the_bag_and_gets_its_answer():
    scenario, clock = make_scenario()
    seat(scenario, "sofa.middle")
    clock.advance(5.0)
    text = "Could you hand me your bag, please?"

    step = scenario.robot_said(text)

    assert step.bag == HANG
    assert of_type(step.events, "bag_request") == [
        {
            "type": "bag_request",
            "t": clock.t,
            "guest": "guest2",
            "count": 1,
            "hang": True,
            "early_answer": False,
        }
    ]
    [answer] = [a for a in scenario.tick(clock.t + 1.0).answers if a.to == text]
    assert (answer.text, answer.guest, answer.asks) == ("It is on your hand.", "guest2", (BAG,))
    # Every request of the owner asks for the hang; the node skips it while the bag hangs.
    step = scenario.robot_said("Please give me your bag.")
    assert step.bag == HANG
    assert of_type(step.events, "bag_request")[0]["count"] == 2


def test_a_bag_request_to_a_guest_without_the_bag_is_an_extra_question():
    scenario, clock = make_scenario()

    step = scenario.robot_said("Could you hand me your bag, please?")

    assert step.bag is None
    assert types(step) == ["robot_line", "person", "extra_question"]
    [answer] = scenario.tick(clock.t + 1.0).answers
    assert (answer.text, answer.asks) == ("Okay.", (OTHER,))


def test_other_questions_get_okay_and_are_logged_as_extra_questions():
    scenario, clock = make_scenario()

    step = scenario.robot_said("How was your trip?")

    assert of_type(step.events, "extra_question") == [
        {"type": "extra_question", "t": clock.t, "guest": "guest1", "text": "How was your trip?"}
    ]
    step = scenario.robot_said("What is your name? How was your trip?")
    assert len(of_type(step.events, "extra_question")) == 1
    answers = scenario.tick(clock.t + 1.0).answers
    assert [(a.text, a.asks) for a in answers] == [
        ("Okay.", (OTHER,)),
        ("My name is Sam.", (NAME, OTHER)),
    ]


def test_guest2_arrives_at_the_guest_spot_once_guest1_is_seated():
    scenario, clock = make_scenario()
    guide(scenario)
    assert scenario.person("guest2").state == ABSENT
    scenario.robot_pose(facing(scenario, "sofa.middle"))

    step = scenario.robot_said("Please have a seat.")

    seated, arrived = of_type(step.events, "person")
    assert (seated["person"], seated["state"], seated["place"]) == ("guest1", SEATED, "sofa.middle")
    assert arrived == {
        "type": "person",
        "t": clock.t,
        "person": "guest2",
        "name": "Ana",
        "state": AT_DOOR,
        "track_id": scenario.person("guest2").track_id,
        "place": None,
        "x": 0.65,
        "y": 0.03,
    }
    assert arrived["track_id"] not in {
        scenario.person(other).track_id for other in ("host", "guest1")
    }
    assert (scenario.people_at()["guest2"].x, scenario.people_at()["guest2"].y) == (0.65, 0.03)

    # Guest 2 answers now, and the mumble was guest 1's drink.
    scenario.robot_pose(DOOR)
    text = "What is your name, and what would you like to drink?"
    scenario.robot_said(text)
    [answer] = [a for a in scenario.tick(clock.t + 1.0).answers if a.to == text]
    assert (answer.guest, answer.text) == ("guest2", "My name is Ana. I would like cola.")


def test_the_guest_who_talked_follows_once_the_robot_is_away_from_the_door():
    scenario, clock = make_scenario()
    # Driving off before guest 1 talked leaves them at the door.
    scenario.robot_pose(Pose(DOOR.x + 1.0, DOOR.y, DOOR.yaw))
    assert scenario.person("guest1").state == AT_DOOR
    scenario.robot_pose(DOOR)
    scenario.robot_said("What is your name?")

    assert scenario.robot_pose(Pose(DOOR.x + 0.4, DOOR.y, DOOR.yaw)).events == ()
    assert scenario.person("guest1").state == TALKING
    step = scenario.robot_pose(Pose(DOOR.x + 0.6, DOOR.y, DOOR.yaw))

    assert step.events == (
        {
            "type": "person",
            "t": clock.t,
            "person": "guest1",
            "name": "Sam",
            "state": FOLLOWING,
            "track_id": None,
            "place": None,
            "x": None,
            "y": None,
        },
    )
    assert "guest1" not in scenario.people_at()
    assert scenario.talking() is None
    # A following guest still answers.
    text = "What would you like to drink?"
    scenario.robot_said(text)
    [answer] = [a for a in scenario.tick(clock.t + 1.0).answers if a.to == text]
    assert answer.guest == "guest1"


def test_a_seat_line_while_nobody_follows_is_no_offer():
    scenario, _ = make_scenario()

    step = scenario.robot_said("Please sit here.")

    [line] = of_type(step.events, "robot_line")
    assert line["offer"] is False
    assert types(step) == ["robot_line"]


def test_an_offer_facing_a_free_place_seats_the_following_guest():
    scenario, _ = make_scenario()
    guide(scenario)
    pose = facing(scenario, "sofa.middle")
    scenario.robot_pose(pose)

    step = scenario.robot_said("Please take a seat here.")

    [line] = of_type(step.events, "robot_line")
    assert (line["offer"], line["question"], line["asks"]) == (True, False, [])
    taken = offer_of(step)
    assert taken["type"] == "seat_taken"
    assert (taken["guest"], taken["place"], taken["reason"], taken["taken_by"]) == (
        "guest1",
        "sofa.middle",
        None,
        None,
    )
    assert taken["pose"] == {"x": 1.85, "y": -1.71, "yaw": round(pose.yaw, 4)}
    assert taken["heading_deg"] == pytest.approx(math.degrees(pose.yaw), abs=0.01)
    assert taken["bearings_deg"] == {
        pid: pytest.approx(bearing_deg(pose, place.x, place.y), abs=0.006)
        for pid, place in scenario.places.items()
    }
    assert taken["bearings_deg"]["sofa.middle"] == pytest.approx(0.0, abs=0.006)
    assert taken["window_deg"] == 25
    assert scenario.person("guest1").state == SEATED
    assert scenario.person("guest1").place == "sofa.middle"
    sitting = scenario.people_at()["guest1"]
    assert (sitting.x, sitting.y, sitting.posture) == (3.115, -1.65, "sitting")
    assert sitting.facing == pytest.approx(3.1416)


def test_an_offer_facing_a_taken_place_is_a_bad_offer_and_the_guest_keeps_following():
    scenario, _ = make_scenario()
    guide(scenario)
    scenario.robot_pose(facing(scenario, "sofa.left"))

    step = scenario.robot_said("Please sit here.")

    bad = offer_of(step)
    assert (bad["type"], bad["guest"], bad["place"], bad["reason"], bad["taken_by"]) == (
        "bad_offer",
        "guest1",
        "sofa.left",
        TAKEN,
        "host",
    )
    assert not of_type(step.events, "person")
    assert scenario.person("guest1").state == FOLLOWING
    # A second offer, facing a free place, still seats them.
    scenario.robot_pose(facing(scenario, "sofa.right"))
    assert offer_of(scenario.robot_said("Please sit here instead."))["place"] == "sofa.right"


def test_an_offer_with_no_place_within_the_window_is_a_bad_offer():
    scenario, _ = make_scenario()
    guide(scenario)
    # Turned away from the sofa.
    scenario.robot_pose(Pose(LIVING_ROOM.x, LIVING_ROOM.y, math.pi))

    bad = offer_of(scenario.robot_said("Please sit down."))

    assert (bad["type"], bad["place"], bad["reason"], bad["taken_by"]) == (
        "bad_offer",
        None,
        NO_PLACE,
        None,
    )
    assert min(abs(b) for b in bad["bearings_deg"].values()) > 25
    assert scenario.person("guest1").state == FOLLOWING


@pytest.mark.parametrize(
    ("off_deg", "type_", "place"), [(24.0, "seat_taken", "sofa.right"), (26.0, "bad_offer", None)]
)
def test_the_offer_window_is_25_deg_from_the_heading(off_deg, type_, place):
    scenario, _ = make_scenario()
    guide(scenario)
    # Turned right of sofa.right, which then lies off_deg to the left. The
    # middle place is about 24 deg further left still.
    straight = facing(scenario, "sofa.right")
    scenario.robot_pose(straight._replace(yaw=straight.yaw - math.radians(off_deg)))

    offer = offer_of(scenario.robot_said("Please sit down."))

    assert offer["bearings_deg"]["sofa.right"] == pytest.approx(off_deg, abs=0.01)
    assert (offer["type"], offer["place"]) == (type_, place)


@pytest.mark.parametrize(
    ("heading_deg", "type_", "place"),
    [
        (12.0, "seat_taken", "sofa.middle"),
        (16.0, "bad_offer", "sofa.left"),
        (-12.0, "seat_taken", "sofa.right"),
    ],
)
def test_the_place_nearest_the_heading_wins_among_places_in_the_window(heading_deg, type_, place):
    scenario, _ = make_scenario(offer_window_deg=60)
    guide(scenario)
    scenario.robot_pose(Pose(LIVING_ROOM.x, LIVING_ROOM.y, math.radians(heading_deg)))

    offer = offer_of(scenario.robot_said("Please sit down."))

    # Every place lies within the window, so only the bearing decides; a
    # taken nearest place is a bad offer, not a reason to take the next one.
    assert all(abs(b) <= 60 for b in offer["bearings_deg"].values())
    assert (offer["type"], offer["place"]) == (type_, place)


def test_the_swap_fires_once_both_guests_sit_and_the_introduction_is_in_progress():
    scenario, clock = make_scenario()
    seat(scenario, "sofa.middle")
    seat(scenario, "sofa.right")
    tracks = {pid: scenario.person(pid).track_id for pid in ("guest1", "guest2")}
    items = todos(("t1", "Greet the guests", "completed"), ("t2", INTRODUCTION, "in_progress"))

    step = scenario.todos_changed(items)

    listed, switch, *moved = step.events
    assert listed == {"type": "todos", "t": clock.t, "items": items}
    assert switch == {
        "type": "seat_switch",
        "t": clock.t,
        "mode": "swap",
        "item": "t2",
        "old": {"guest1": "sofa.middle", "guest2": "sofa.right"},
        "new": {"guest1": "sofa.right", "guest2": "sofa.middle"},
    }
    assert [(p["person"], p["state"], p["place"]) for p in moved] == [
        ("guest1", SEATED, "sofa.right"),
        ("guest2", SEATED, "sofa.middle"),
    ]
    # New tracks: each of them changed places at once.
    assert all(scenario.person(pid).track_id != track for pid, track in tracks.items())
    assert scenario.people_at()["guest1"].y == -2.20
    assert scenario.people_at()["guest2"].y == -1.65
    # It fires once.
    assert types(scenario.todos_changed(todos(("t2", INTRODUCTION, "waiting")))) == ["todos"]


def test_the_swap_armed_before_guest2_sits_fires_when_guest2_sits():
    scenario, _ = make_scenario()
    seat(scenario, "sofa.middle")

    step = scenario.todos_changed(todos(("t2", INTRODUCTION, "in_progress")))

    assert types(step) == ["todos"]
    step = seat(scenario, "sofa.right")
    assert types(step) == ["robot_line", "seat_taken", "person", "seat_switch", "person", "person"]
    assert of_type(step.events, "seat_switch")[0]["new"] == {
        "guest1": "sofa.right",
        "guest2": "sofa.middle",
    }


def test_move_takes_the_person_to_the_first_free_place_in_file_order():
    scenario, _ = make_scenario(
        seats=with_armchair(), seat_switch={"mode": "move", "people": ["guest1"]}
    )
    seat(scenario, "sofa.middle")
    seat(scenario, "sofa.right")

    step = scenario.todos_changed(todos(("t2", INTRODUCTION, "in_progress")))

    [switch] = of_type(step.events, "seat_switch")
    assert (switch["mode"], switch["old"], switch["new"]) == (
        "move",
        {"guest1": "sofa.middle"},
        {"guest1": "armchair"},
    )
    moved = scenario.people_at()["guest1"]
    # A seat other than a sofa is one place, at its box's centre.
    assert (moved.x, moved.y, moved.facing) == (3.05, -0.25, -1.5708)


def test_move_to_a_named_place_frees_the_old_one():
    scenario, _ = make_scenario(
        seats=with_armchair(),
        seat_switch={"mode": "move", "people": ["host"], "to": "armchair"},
    )

    # The host sits from the start, so the switch fires at once.
    step = scenario.todos_changed(todos(("t2", INTRODUCTION, "in_progress")))

    [switch] = of_type(step.events, "seat_switch")
    assert (switch["old"], switch["new"]) == ({"host": "sofa.left"}, {"host": "armchair"})
    assert offer_of(seat(scenario, "sofa.left"))["type"] == "seat_taken"


def test_move_to_a_taken_place_is_skipped():
    scenario, clock = make_scenario(
        seats=with_armchair(),
        seat_switch={"mode": "move", "people": ["guest1"], "to": "sofa.left"},
    )
    seat(scenario, "sofa.middle")

    step = scenario.todos_changed(todos(("t2", INTRODUCTION, "in_progress")))

    assert of_type(step.events, "seat_switch_skipped") == [
        {
            "type": "seat_switch_skipped",
            "t": clock.t,
            "mode": "move",
            "item": "t2",
            "reason": "no free place for move: sofa.left is taken by host",
        }
    ]
    assert scenario.person("guest1").place == "sofa.middle"


def test_off_never_switches():
    scenario, _ = make_scenario(seat_switch={"mode": "off", "people": []})
    seat(scenario, "sofa.middle")
    seat(scenario, "sofa.right")

    for status in ("in_progress", "completed"):
        assert types(scenario.todos_changed(todos(("t2", INTRODUCTION, status)))) == ["todos"]
    assert scenario.person("guest1").place == "sofa.middle"


def test_the_switch_is_skipped_when_the_introduction_finishes_before_guest2_sits():
    scenario, clock = make_scenario()
    seat(scenario, "sofa.middle")
    scenario.todos_changed(todos(("t2", INTRODUCTION, "in_progress")))

    step = scenario.todos_changed(todos(("t2", INTRODUCTION, "completed")))

    assert of_type(step.events, "seat_switch_skipped") == [
        {
            "type": "seat_switch_skipped",
            "t": clock.t,
            "mode": "swap",
            "item": "t2",
            "reason": ITEM_FINISHED,
        }
    ]
    assert not of_type(seat(scenario, "sofa.right").events, "seat_switch")


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_an_introduction_already_finished_when_first_listed_skips_the_switch(status):
    scenario, _ = make_scenario()

    step = scenario.todos_changed(todos(("t2", INTRODUCTION, status)))

    [skipped] = of_type(step.events, "seat_switch_skipped")
    assert (skipped["item"], skipped["reason"]) == ("t2", ITEM_FINISHED)


def test_the_introduction_leaving_the_list_skips_the_switch():
    scenario, _ = make_scenario()
    scenario.todos_changed(todos(("t2", INTRODUCTION, "in_progress")))

    # A reset publishes an empty list.
    step = scenario.todos_changed([])

    [skipped] = of_type(step.events, "seat_switch_skipped")
    assert (skipped["item"], skipped["reason"]) == ("t2", ITEM_GONE)


def test_a_finished_introduction_skips_nothing_while_another_is_still_to_come():
    scenario, _ = make_scenario()
    first = ("t1", "Introduce yourself to the host", "completed")

    step = scenario.todos_changed(todos(first, ("t4", INTRODUCTION, "pending")))

    assert types(step) == ["todos"]
    scenario.todos_changed(todos(first, ("t4", INTRODUCTION, "in_progress")))
    seat(scenario, "sofa.middle")
    step = seat(scenario, "sofa.right")
    assert of_type(step.events, "seat_switch")[0]["item"] == "t4"


def test_the_early_bag_variant_hangs_the_bag_only_on_the_second_request():
    scenario, clock = make_scenario(EARLY_BAG_FILE)
    seat(scenario, "sofa.middle")

    first = scenario.robot_said("Could you hand me your bag, please?")
    second = scenario.robot_said("Could you hand me your bag again, please?")

    assert first.bag is None and second.bag == HANG
    requests = of_type(first.events, "bag_request") + of_type(second.events, "bag_request")
    assert [(r["count"], r["hang"], r["early_answer"]) for r in requests] == [
        (1, False, True),
        (2, True, True),
    ]
    # Both are answered: the first close then reads empty, and the second holds the bag.
    answers = [a for a in scenario.tick(clock.t + 1.0).answers if a.guest == "guest2"]
    assert [a.text for a in answers] == ["It is on your hand."] * 2


def test_a_reset_drops_the_answers_not_sent_and_restores_the_bag():
    scenario, clock = make_scenario()
    scenario.robot_said("What is your name?")

    step = scenario.reset(clock.t + 0.5)

    assert step.bag == RESTORE
    assert step.events == ({"type": "reset", "t": clock.t + 0.5, "dropped": 1},)
    assert scenario.tick(clock.t + 2.0).answers == ()


def test_a_clock_that_goes_back_moves_the_pending_answers_and_the_pacing_with_it():
    scenario, clock = make_scenario(answers={"mumble_once": None})
    said = clock.t
    scenario.robot_said("What is your name?")
    old_t = said + 0.5
    talking = scenario.talking(old_t)

    # The simulation restarts: its clock goes from 100.5 back to 0.5.
    step = scenario.clock_jumped(old_t, 0.5)

    assert step.events == (
        {"type": "clock_jump", "t": 0.5, "from": old_t, "to": 0.5, "pending": 1},
    )
    # The line was said at 0.0 on the new clock, so its answer is due 1 s later.
    assert scenario.tick(0.99).answers == ()
    [answer] = scenario.tick(1.0).answers
    assert (answer.text, answer.asked_t) == ("My name is Sam.", said)
    # The guest paces on from where they were, in the log and in the scene.
    assert scenario.talking(0.5) == talking
    guest = scenario.people_at(0.5)["guest1"]
    assert (guest.x, guest.y) == pytest.approx((talking["x"], talking["y"]), abs=1e-4)
    # The bystander still sets off 3 s after the first question.
    assert not of_type(scenario.tick(2.99).events, "bystander")
    assert of_type(scenario.tick(3.0).events, "bystander")[0]["state"] == "walking"


def test_the_bystander_walks_its_path_once_from_3_s_after_the_first_question():
    scenario, clock = make_scenario()
    assert scenario.tick(clock.t + 60.0).events == ()
    t0 = clock.advance(60.0)
    scenario.robot_said("What is your name?")
    start, end = t0 + 3.0, t0 + 3.0 + 1.7 / 0.6

    assert not of_type(scenario.tick(start - 0.01).events, "bystander")
    assert BYSTANDER_ID not in scenario.people_at(start - 0.01)
    [walking] = of_type(scenario.tick(start).events, "bystander")
    walker = scenario.people_at(start + 1.0)[BYSTANDER_ID]
    assert walking == {
        "type": "bystander",
        "t": start,
        "state": "walking",
        "track_id": walker.track_id,
        "x": -0.35,
        "y": -0.6,
    }
    # 0.6 m along the path after 1 s, facing along it.
    assert (walker.x, walker.y) == pytest.approx((-0.35, 0.0))
    assert walker.facing == pytest.approx(math.pi / 2)
    assert (walker.appearance, walker.face, walker.height) == ("passerby", None, 1.70)
    assert scenario.people_at(end - 0.01)[BYSTANDER_ID].y == pytest.approx(1.1, abs=0.01)
    assert not of_type(scenario.tick(end - 0.01).events, "bystander")
    [gone] = of_type(scenario.tick(end + 0.01).events, "bystander")
    assert (gone["state"], gone["x"], gone["y"]) == ("gone", -0.35, 1.1)
    assert BYSTANDER_ID not in scenario.people_at(end + 0.01)
    # Once only: a later question brings nobody back.
    scenario.robot_said("What would you like to drink?", end + 1.0)
    assert not of_type(scenario.tick(end + 10.0).events, "bystander")
    assert BYSTANDER_ID not in scenario.people_at(end + 5.0)


def test_the_talking_guest_paces_along_the_hallway():
    scenario, clock = make_scenario()
    assert scenario.talking() is None
    t0 = clock.t
    scenario.robot_said("What is your name?")

    for share, offset in ((0.0, 0.0), (0.25, 0.28), (0.5, 0.0), (0.75, -0.28)):
        guest = scenario.people_at(t0 + share * 6.0)["guest1"]
        # x stays on the hallway's line.
        assert (guest.x, guest.y) == pytest.approx((0.65, 0.03 + offset), abs=1e-9)
        # Turned to the robot all the while.
        assert guest.facing == pytest.approx(math.atan2(DOOR.y - guest.y, DOOR.x - guest.x))

    talking = scenario.talking(t0 + 1.5)
    assert talking == {
        "guest": "guest1",
        "x": 0.65,
        "y": 0.31,
        "bearing_deg": round(bearing_deg(DOOR, 0.65, 0.31), 2),
    }
    # About 9 degrees to each side of the spot, as seen from the door pose.
    spot = bearing_deg(DOOR, 0.65, 0.03)
    assert bearing_deg(DOOR, 0.65, 0.31) - spot == pytest.approx(-9.06, abs=0.01)
    assert bearing_deg(DOOR, 0.65, -0.25) - spot == pytest.approx(9.40, abs=0.01)


def test_without_an_axis_the_guest_paces_across_the_line_to_the_door():
    scenario, clock = make_scenario(pacing={"amplitude": 0.2, "axis": None})
    t0 = clock.t
    scenario.robot_said("What is your name?")
    dx, dy = DOOR.x - 0.65, DOOR.y - 0.03
    across = (-dy / math.hypot(dx, dy), dx / math.hypot(dx, dy))

    for share, offset in ((0.0, 0.0), (0.25, 0.2), (0.5, 0.0), (0.75, -0.2)):
        guest = scenario.people_at(t0 + share * 6.0)["guest1"]
        assert (guest.x, guest.y) == pytest.approx(
            (0.65 + offset * across[0], 0.03 + offset * across[1]), abs=1e-9
        )
        # The distance to the door hardly changes.
        assert math.hypot(DOOR.x - guest.x, DOOR.y - guest.y) == pytest.approx(
            math.hypot(dx, dy), abs=0.012
        )

def test_people_at_a_time_before_a_seat_offer_draws_everyone_as_they_were_then():
    scenario, clock = make_scenario()
    guide(scenario)
    scenario.robot_pose(facing(scenario, "sofa.middle"))
    before = clock.t
    taken = clock.advance(1.0)

    scenario.robot_said("Please sit here.")

    # An image taken before the offer: guest 1 follows, in no view, and
    # guest 2 has not come yet.
    then = scenario.people_at(before + 0.5)
    assert set(then) == {"host"}
    # From the offer on, guest 1 sits and guest 2 waits at the guest spot.
    for t in (taken, taken + 0.5):
        people = scenario.people_at(t)
        assert set(people) == {"host", "guest1", "guest2"}
        guest1, guest2 = people["guest1"], people["guest2"]
        assert (guest1.posture, guest1.x, guest1.y) == ("sitting", 3.115, -1.65)
        assert (guest2.x, guest2.y) == (0.65, 0.03)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"seated": [{"id": "host", "place": "sofa.top"}]},
            "seated.0.place: 'sofa.top' is not a place; the places are sofa.left, sofa.middle, "
            "sofa.right.",
        ),
        (
            {"seat_switch": {"people": ["guest1", "guest3"]}},
            "seat_switch.people: 'guest3' is neither a guest (guest1, guest2) nor a seated "
            "person (host).",
        ),
        (
            {"seats": [{**read_yaml(DEFAULT_FILE)["seats"][0], "places": {"left": [3.1, -1.1]}}]},
            "seats.0: a sofa needs exactly the places left, middle, right, the parts find_seats "
            "splits it into; got left.",
        ),
        (
            {"seat_switch": {"mode": "move", "people": ["guest1"]}},
            "seat_switch.mode 'move' needs 4 places, one for each of the 2 guests and 1 seated "
            "people and one free place to move to, but the seats have 3.",
        ),
        (
            {"guests": [*read_yaml(DEFAULT_FILE)["guests"], {"name": "Kim", "drink": "tea"}]},
            "seat_switch.mode 'swap' needs 4 places, one for each of the 3 guests and 1 seated "
            "people, but the seats have 3.",
        ),
        ({"seat_switch": {"people": ["guest1"]}}, "seat_switch.mode 'swap' needs exactly two"),
        ({"colour": "red"}, "colour: Extra inputs are not permitted"),
        (
            {"seats": [{**read_yaml(DEFAULT_FILE)["seats"][0], "label": "bed"}]},
            "seats.0: label 'bed' is not one of find_seats's seat classes",
        ),
        (
            {"seats": with_armchair(places={"left": [3.05, -0.25]})},
            "seats.1: only a sofa class (sofa, couch, bench, loveseat) has places",
        ),
        ({"answers": {"mumble_once": "guest3.drink"}}, "answers.mumble_once must be <guest>.name"),
        (
            {"bag": {"owner": "guest3"}},
            "bag.owner must be a guest of this file (guest1, guest2), got 'guest3'.",
        ),
        ({"oracle": {"outlier_rate": 2.0}}, "oracle: outlier_rate must lie in [0, 1]."),
        ({"pacing": {"axis": [0.0, 0.0]}}, "pacing: axis must not be (0, 0)."),
    ],
)
def test_a_scenario_file_with_a_mistake_is_refused_with_what_to_fix(tmp_path, change, message):
    path = write_scenario(tmp_path, deep_update(read_yaml(DEFAULT_FILE), change))

    with pytest.raises(ScenarioError) as raised:
        load_scenario(path)

    text = str(raised.value)
    assert text.startswith(f"The scenario {path} is not valid:\n")
    assert f"\n  - {message}" in text
    # A seat that fails is not also reported as a missing list of seats.
    assert "at least 1 item" not in text


def test_every_problem_of_a_file_is_named_at_once(tmp_path):
    change = {
        "seated": [{"id": "host", "place": "sofa.top"}],
        "seat_switch": {"people": ["guest1", "guest3"]},
        "bag": {"owner": "guest3"},
    }
    path = write_scenario(tmp_path, deep_update(read_yaml(DEFAULT_FILE), change))

    with pytest.raises(ScenarioError) as raised:
        load_scenario(path)

    _, *problems = str(raised.value).splitlines()
    assert [line.split(":")[0] for line in problems] == [
        "  - seated.0.place",
        "  - seat_switch.people",
        "  - bag.owner must be a guest of this file (guest1, guest2), got 'guest3'.",
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- a list\n- of things\n", "must be a mapping of options, not list."),
        ("guests: [\n", "is not valid YAML"),
    ],
)
def test_a_file_that_is_not_yaml_or_not_a_mapping_is_refused(tmp_path, text, message):
    path = tmp_path / "scenario.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ScenarioError, match=re.escape(message)):
        load_scenario(path)


def test_a_missing_scenario_file_is_refused(tmp_path):
    path = tmp_path / "missing.yaml"

    with pytest.raises(ScenarioError) as raised:
        load_scenario(path)

    assert str(raised.value) == f"Cannot read the scenario {path}: No such file or directory."


def test_a_locations_file_without_door_is_refused(tmp_path):
    locations = tmp_path / "locations.toml"
    locations.write_text("[locations.living_room]\nx = 1.85\ny = -1.71\n", encoding="utf-8")
    path = write_scenario(
        tmp_path, deep_update(read_yaml(DEFAULT_FILE), {"locations": str(locations)})
    )

    with pytest.raises(ScenarioError) as raised:
        load_scenario(path)

    assert str(raised.value) == (
        f"The scenario {path}: locations: {locations} has no door; the scenario reads door "
        "and living_room from it."
    )
    path = write_scenario(
        tmp_path, deep_update(read_yaml(DEFAULT_FILE), {"locations": str(tmp_path / "none")})
    )
    with pytest.raises(ScenarioError, match="locations: cannot read"):
        load_scenario(path)


def test_the_robot_frame_is_the_navigation_tools():
    assert ROBOT_FRAME == NAVIGATION_ROBOT_FRAME


@pytest.mark.parametrize(
    ("text", "question", "asks", "seat"),
    [
        ("What is your name?", True, (NAME,), False),
        ("What should I call you?", True, (NAME,), False),
        ("Please tell me your name.", True, (NAME,), False),
        ("Let me know what you would like to drink.", True, (DRINK,), False),
        ("Would you like a beverage?", True, (DRINK,), False),
        ("What is your name and your favourite drink?", True, (NAME, DRINK), False),
        ("Could you hand me your bag?", True, (BAG,), False),
        ("Please put the bag in my hand.", False, (BAG,), False),
        ("Put your bag on my hand.", False, (BAG,), False),
        ("Place the bag on my hand.", False, (BAG,), False),
        ("Now, hand me your bag.", False, (BAG,), False),
        ("Go ahead and put your bag on my hand.", False, (BAG,), False),
        ("I will put your bag down later.", False, (), False),
        ("Let me take your bag.", False, (), False),
        ("I will hold out my hand for your bag.", False, (), False),
        ("How was your trip?", True, (OTHER,), False),
        ("Welcome! What is your name? Did you find us easily?", True, (NAME, OTHER), False),
        ("Please have a seat.", False, (), True),
        ("Would you like to sit here?", True, (OTHER,), True),
        ("Nice to meet you, Sam.", False, (), False),
    ],
)
def test_classify_line_reads_each_sentence(text, question, asks, seat):
    assert classify_line(text) == Line(question=question, asks=asks, seat=seat)


@pytest.mark.parametrize(
    ("x", "y", "bearing"),
    [(1.0, 0.0, 0.0), (0.0, 1.0, 90.0), (0.0, -1.0, -90.0), (-1.0, 0.0, 180.0)],
)
def test_a_bearing_is_positive_to_the_left_and_180_behind(x, y, bearing):
    assert bearing_deg(Pose(0.0, 0.0, 0.0), x, y) == pytest.approx(bearing)
    # Turned by a full circle, the base sees the same.
    assert bearing_deg(Pose(0.0, 0.0, 2 * math.pi), x, y) == pytest.approx(bearing)


def test_the_event_log_writes_json_lines_and_nothing_after_its_end(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    log = EventLog(path)

    log.write(
        {
            "type": "a",
            "t": 1.23456,
            "missing": float("nan"),
            "pair": (1, 2),
            "file": Path("config/x.yaml"),
            "nested": {"far": float("inf"), "small": np.float32(0.5)},
        }
    )
    log.write({"type": "end", "t": 2.0})
    log.write({"type": "late", "t": 3.0})
    log.close()
    log.write({"type": "after_close", "t": 4.0})

    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [
        {
            "type": "a",
            "t": 1.235,
            "missing": None,
            "pair": [1, 2],
            "file": "config/x.yaml",
            "nested": {"far": None, "small": 0.5},
        },
        {"type": "end", "t": 2.0},
    ]


# -- The default files, seen by the simulation's head camera --------------------------------
#
# The camera is test_projection's, whose constants cite walkie-ros: 640 x 360 px,
# fx = fy = 224.057 px, cx 320, cy 180, and the optical origin at (0.275, 0.060,
# 1.145) m in base_footprint, pitched 9.0 deg down.


def view(pose: Pose, scene: Scene) -> tuple[PersonDetectionArray, ObjectDetectionArray]:
    """The oracle's arrays, without noise, of one image of the head camera on a base at `pose`."""
    return draw_detections(scene, camera_info(), sim_camera(*pose), QUIET, np.random.default_rng(0))


def seen_as(people: PersonDetectionArray, track_id: int) -> PersonDetection:
    [person] = [p for p in people.people if p.track_id == track_id]
    return person


def robot_buffer(pose: Pose) -> Buffer:
    """A TF buffer without a node: the base at `pose` at STAMP, and the head camera on it."""
    buffer = Buffer()
    map_from_base = euler_matrix(0.0, 0.0, pose.yaw)
    map_from_base[:3, 3] = (pose.x, pose.y, 0.0)
    for parent, child, matrix in (
        (MAP_FRAME, ROBOT_FRAME, map_from_base),
        (ROBOT_FRAME, CAMERA_FRAME, base_from_camera()),
    ):
        msg = tf_msg(parent, child, matrix)
        msg.header.stamp = STAMP
        buffer.set_transform(msg, "test")
    return buffer


def test_the_scenarios_door_and_living_room_are_the_locations_files():
    loaded = load_scenario(DEFAULT_FILE)
    scenario = HriScenario(loaded.spec, loaded.locations, FakeClock())
    places = load_locations(LOCATIONS_FILE)

    assert loaded.locations_path.resolve() == LOCATIONS_FILE
    for name, pose in (("door", scenario.door), ("living_room", scenario.living_room)):
        assert pose == Pose(places[name].x, places[name].y, places[name].yaw)
    # The poses this file's geometry was computed at.
    assert (scenario.door, scenario.living_room) == (DOOR, LIVING_ROOM)


def test_the_early_bag_file_differs_from_the_default_only_in_early_answer():
    default, early = read_yaml(DEFAULT_FILE), read_yaml(EARLY_BAG_FILE)

    assert (default["bag"]["early_answer"], early["bag"]["early_answer"]) == (False, True)
    early["bag"]["early_answer"] = False
    assert early == default
    # The comments too: each file reads on its own.
    lines = [p.read_text(encoding="utf-8").splitlines() for p in (DEFAULT_FILE, EARLY_BAG_FILE)]
    assert len(lines[0]) == len(lines[1])
    assert [(a, b) for a, b in zip(*lines) if a != b] == [
        ("bag: {early_answer: false}", "bag: {early_answer: true}")
    ]


def test_from_the_living_room_each_seated_hip_lies_well_inside_its_third_of_the_sofa():
    scenario, _ = make_scenario()
    seat(scenario, "sofa.middle")
    seat(scenario, "sofa.right")

    people, objects = view(LIVING_ROOM, scenario.scene())

    [sofa] = objects.detections
    box = bbox_xyxy(sofa.bbox)
    # One view sees the whole sofa: its box is not cut at either side.
    assert (box[0], box[2]) == pytest.approx((54.8, 585.2), abs=0.5)
    thirds = dict(split_seat_regions(box, has_middle=DEFAULT_SEATING_CONFIG.sofa_has_middle))
    for place, person in (("left", "host"), ("middle", "guest1"), ("right", "guest2")):
        seen = seen_as(people, scenario.person(person).track_id)
        u, _ = hip_anchor(
            seen.keypoints, seen.keypoint_scores, DEFAULT_SEATING_CONFIG.min_hip_score
        )
        left, _, right, _ = thirds[place]
        assert min(u - left, right - u) >= 20.0, (place, u, left, right)
        # The sitters' faces are trusted too, the side ones turned 29 deg away.
        assert face_trusted(seen, people.camera_info), person


@pytest.mark.parametrize(
    ("part", "heading_deg", "off_deg"),
    [("left", 38.6, -12.9), ("middle", 0.0, 2.7), ("right", -38.6, 17.5)],
)
def test_turning_to_a_thirds_box_heading_makes_the_offer_pick_its_place(part, heading_deg, off_deg):
    scenario, _ = make_scenario()
    _, objects = view(LIVING_ROOM, scenario.scene())
    [sofa] = objects.detections
    third = dict(split_seat_regions(bbox_xyxy(sofa.bbox)))[part]
    info = camera_info()

    heading = box_heading(third, info, info.header, robot_buffer(LIVING_ROOM))

    assert math.degrees(heading) == pytest.approx(heading_deg, abs=0.1)
    guide(scenario)
    scenario.robot_pose(Pose(LIVING_ROOM.x, LIVING_ROOM.y, heading))
    offer = offer_of(scenario.robot_said("Please sit here."))
    place = f"sofa.{part}"
    assert offer["place"] == place
    # The ray through a third's centre column is biased outwards, so the
    # place lies off the heading, but within 25 deg less 7.5 for the turn.
    bearings = offer["bearings_deg"]
    assert bearings[place] == pytest.approx(off_deg, abs=0.1)
    others = [abs(b) for pid, b in bearings.items() if pid != place]
    if part == "middle":
        assert min(others) > abs(bearings[place]) + 15.0
    else:
        assert min(others) >= 35.8
    # The host sits on the left, so that offer is the bad one the rule must catch.
    expected = ("bad_offer", TAKEN, "host") if part == "left" else ("seat_taken", None, None)
    assert (offer["type"], offer["reason"], offer["taken_by"]) == expected


@pytest.mark.parametrize("share", [0.0, 0.25, 0.75])
@pytest.mark.parametrize(("guest", "height"), [("guest1", 1.70), ("guest2", 1.80)])
def test_from_the_door_the_pacing_guest_shows_the_whole_face_and_the_hips(guest, height, share):
    scenario, clock = make_scenario()
    if guest == "guest2":
        seat(scenario, "sofa.middle")
    t0 = clock.advance(10.0)
    scenario.robot_said("What is your name?")
    assert scenario.people_at()[guest].height == height

    people, _ = view(DOOR, scenario.scene(t0 + share * 6.0))

    seen = seen_as(people, scenario.person(guest).track_id)
    x1, y1, x2, y2 = bbox_xyxy(seen.face_bbox)
    assert 0.0 < x1 < x2 < WIDTH and 0.0 < y1 < y2 < HEIGHT
    assert seen.face_det_score == pytest.approx(DETECTION_SCORE) and len(seen.face_embedding)
    assert face_trusted(seen, people.camera_info)
    # Seen, not only inside the image: the oracle clips every keypoint into
    # it, and scores one it does not see 0.
    hip_scores = np.asarray(seen.keypoint_scores)[[LEFT_HIP, RIGHT_HIP]]
    assert hip_scores.tolist() == pytest.approx([KEYPOINT_SCORE] * 2)
    assert hip_scores.min() > DEFAULT_SEATING_CONFIG.min_hip_score


def test_the_pacing_ends_lie_in_the_hallways_free_span_outside_the_frame():
    scenario, clock = make_scenario()
    t0 = clock.t
    scenario.robot_said("What is your name?")
    period = make_spec().pacing.period

    ends = [scenario.people_at(t0 + share * period)["guest1"] for share in (0.25, 0.75)]

    assert (ends[0].x, ends[0].y) == pytest.approx((0.65, 0.31), abs=1e-9)
    assert (ends[1].x, ends[1].y) == pytest.approx((0.65, -0.25), abs=1e-9)
    for end in ends:
        assert end.x == pytest.approx(HALLWAY_X)
        assert HALLWAY_FREE_Y[0] < end.y < HALLWAY_FREE_Y[1]
        assert end.x < FRAME_X[0]


def test_the_bystander_crosses_the_doorways_sight_cone_and_shows_no_face():
    scenario, clock = make_scenario()
    path = make_spec().bystander.path
    camera_x, camera_y = (float(v) for v in camera_position(sim_camera(*DOOR))[:2])
    walk_x = path[0][0]

    def cone_y(edge_y: float) -> float:
        # The line from the camera through an edge of the frame's opening, at the walk.
        share = (walk_x - camera_x) / (FRAME_CENTRE_X - camera_x)
        return camera_y + share * (edge_y - camera_y)

    low, high = cone_y(FRAME_OPENING_Y[0]), cone_y(FRAME_OPENING_Y[1])
    assert all(x == walk_x == -0.35 for x, _ in path)
    assert (low, high) == pytest.approx((-0.497, 0.956), abs=2e-3)
    assert min(y for _, y in path) < low and max(y for _, y in path) > high

    t0 = clock.t
    scenario.robot_said("What is your name?")
    start = t0 + 3.0
    in_cone = 0
    for t in np.arange(start, start + 1.7 / 0.6, 0.1):
        walker = scenario.people_at(t)[BYSTANDER_ID]
        assert walker.x == pytest.approx(walk_x)
        off = abs(math.degrees(math.atan2(camera_y - walker.y, camera_x - walker.x)) - 90.0)
        assert 80.0 < off < 119.0 and off > FACE_LIMIT_DEG
        assert not walker.faces_camera((camera_x, camera_y))
        if low <= walker.y <= high:
            in_cone += 1
            seen = seen_as(view(DOOR, scenario.scene(t))[0], walker.track_id)
            assert seen.face_det_score == 0.0 and len(seen.face_embedding) == 0
    assert in_cone >= 20


# -- GraspParameter, on fake service clients ----------------------------------------------

COMMANDER = "/bimanual_commander"


class FakeClient:
    """A service client whose requests the test answers.

    call_async keeps each request with its future. A future without an
    executor runs its done callbacks at once, on the thread that answers it.
    """

    def __init__(self) -> None:
        self.ready = True
        self.calls: list[tuple[Any, Future]] = []
        # Run once, at the start of the next call_async: what another thread
        # may do after a request was decided and before it is sent.
        self.on_call: Callable[[], None] | None = None

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, request: Any) -> Future:
        hook, self.on_call = self.on_call, None
        if hook is not None:
            hook()
        future = Future()
        self.calls.append((request, future))
        return future


class FakeTimer:
    """A timer that fires only when the test fires it."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeNode:
    """What GraspParameter takes from its node: a client for each service, and a timer."""

    def __init__(self) -> None:
        self.clients: dict[type, FakeClient] = {}
        self.timer: FakeTimer | None = None

    def create_client(self, srv_type: type, _name: str, **_kwargs: Any) -> FakeClient:
        self.clients[srv_type] = FakeClient()
        return self.clients[srv_type]

    def create_timer(
        self, _period: float, callback: Callable[[], None], **_kwargs: Any
    ) -> FakeTimer:
        self.timer = FakeTimer(callback)
        return self.timer


class GraspRig:
    """GraspParameter on a FakeNode, with a commander whose answers the test sends.

    `value` is the commander's grasp_detect_min; a set that succeeds changes
    it. `clock` is GraspParameter's monotonic clock, and `records` what it
    logged.
    """

    def __init__(self, value: float = GRASP_DETECT_MIN) -> None:
        self.value = value
        self.clock = FakeClock()
        self.records: list[dict[str, Any]] = []
        self.node = FakeNode()
        self.grasp = GraspParameter(self.node, COMMANDER, self._record, monotonic=self.clock)
        self.get = self.node.clients[GetParameters]
        self.set = self.node.clients[SetParameters]

    @classmethod
    def read(cls) -> "GraspRig":
        """A rig whose commander was read at its value."""
        rig = cls()
        rig.fire()
        rig.answer_read()
        return rig

    def types(self) -> list[str]:
        return [record["type"] for record in self.records]

    def fire(self) -> None:
        """One period of the retry timer."""
        timer = self.node.timer
        assert timer is not None and not timer.cancelled
        timer.callback()

    def sets(self) -> list[float]:
        """The value of every set sent, in the order sent."""
        return [request.parameters[0].value.double_value for request, _ in self.set.calls]

    def answer_read(self, index: int = -1) -> None:
        """Answer a read, the last one sent unless `index` says, with the commander's value."""
        _, future = self.get.calls[index]
        value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=self.value)
        future.set_result(GetParameters.Response(values=[value]))

    def answer_set(self, index: int = -1, *, successful: bool = True, reason: str = "") -> None:
        """Answer a set, the last one sent unless `index` says."""
        request, future = self.set.calls[index]
        if successful:
            self.value = request.parameters[0].value.double_value
        result = SetParametersResult(successful=successful, reason=reason)
        future.set_result(SetParameters.Response(results=[result]))

    def _record(self, type_: str, **fields: Any) -> None:
        self.records.append({"type": type_, **fields})


def test_a_reset_as_the_hang_that_waited_for_the_read_goes_out_ends_at_the_original():
    rig = GraspRig()
    # A bag request before the commander was read: the hang waits for the read.
    hang = rig.grasp.hang()
    assert hang is not None and rig.sets() == []
    # A reset comes just as the read's answer sends the hang.
    rig.set.on_call = lambda: rig.grasp.restore("reset")

    rig.answer_read()

    # The restore waits for the hang's answer, so it goes out last.
    assert rig.sets() == [HUNG_VALUE]
    rig.answer_set()
    assert rig.sets() == [HUNG_VALUE, GRASP_DETECT_MIN]
    rig.answer_set()
    assert rig.value == GRASP_DETECT_MIN and hang.is_set()
    assert rig.records == [
        {"type": "commander", "node": COMMANDER, "grasp_detect_min": GRASP_DETECT_MIN},
        {"type": "bag_hung", "value": HUNG_VALUE},
        {"type": "bag_restored", "value": GRASP_DETECT_MIN, "cause": "reset"},
    ]


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("gone", f"no {COMMANDER}: {COMMANDER}/set_parameters has no server"),
        ("refused", "the arm is moving"),
        ("error", "setting grasp_detect_min failed: the commander went away"),
    ],
)
def test_a_restore_that_fails_is_sent_again_until_the_commander_takes_it(failure, reason):
    rig = GraspRig.read()
    rig.grasp.hang()
    rig.answer_set()
    rig.set.ready = failure != "gone"

    done = rig.grasp.restore("reset")

    if failure == "refused":
        rig.answer_set(successful=False, reason=reason)
    elif failure == "error":
        rig.set.calls[-1][1].set_exception(RuntimeError("the commander went away"))
    assert rig.records[-1] == {"type": "bag_retry", "action": RESTORE, "reason": reason}
    assert done is not None and not done.is_set()
    assert rig.value == HUNG_VALUE
    rig.set.ready = True
    rig.fire()
    rig.answer_set()
    assert done.is_set() and rig.value == GRASP_DETECT_MIN
    assert rig.records[-1] == {"type": "bag_restored", "value": GRASP_DETECT_MIN, "cause": "reset"}
    assert "bag_skipped" not in rig.types()


def test_a_second_hang_while_the_first_is_pending_gets_the_same_event():
    rig = GraspRig.read()
    first = rig.grasp.hang()

    second = rig.grasp.hang()

    assert first is not None and second is first and not first.is_set()
    assert rig.sets() == [HUNG_VALUE]
    rig.answer_set()
    assert first.is_set() and rig.types() == ["commander", "bag_hung"]
    # Once the commander holds the hang, a request gets that hang's event, already set.
    assert rig.grasp.hang() is first
    assert rig.sets() == [HUNG_VALUE]


def test_a_request_without_an_answer_is_sent_again_after_request_timeout_s(monkeypatch):
    monkeypatch.setattr(hri_scenario, "REQUEST_TIMEOUT_S", 0.5)
    rig = GraspRig()
    rig.fire()
    rig.clock.advance(0.25)
    rig.fire()
    assert len(rig.get.calls) == 1
    rig.clock.advance(0.25)

    rig.fire()

    # The read got no answer within 0.5 s, so it went out again.
    assert len(rig.get.calls) == 2
    # The first read's late answer is ignored; the second one's counts.
    rig.answer_read(0)
    assert rig.records == []
    rig.answer_read()
    assert rig.types() == ["commander"]
    hang = rig.grasp.hang()
    assert hang is not None and rig.sets() == [HUNG_VALUE]
    rig.clock.advance(0.5)
    rig.fire()
    assert rig.sets() == [HUNG_VALUE, HUNG_VALUE]
    assert rig.records[-1] == {
        "type": "bag_retry",
        "action": HANG,
        "reason": f"{COMMANDER} did not answer within 0.5 s",
    }
    assert not hang.is_set()
    rig.answer_set()
    assert hang.is_set() and rig.records[-1] == {"type": "bag_hung", "value": HUNG_VALUE}


def test_once_finishing_a_hang_is_skipped_and_only_the_restore_goes_out():
    rig = GraspRig.read()
    rig.grasp.hang()
    rig.answer_set()
    done = rig.grasp.finish()

    # A bag request while the exit's restore is in flight.
    assert rig.grasp.hang() is None

    assert rig.records[-1] == {
        "type": "bag_skipped",
        "action": HANG,
        "reason": "the scenario is closing",
    }
    rig.answer_set()
    rig.fire()
    assert done is not None and done.is_set()
    assert rig.sets() == [HUNG_VALUE, GRASP_DETECT_MIN] and rig.value == GRASP_DETECT_MIN
    assert rig.records[-1] == {"type": "bag_restored", "value": GRASP_DETECT_MIN, "cause": "exit"}


# -- The node over ROS --------------------------------------------------------------------


def near(record_pose: dict[str, float] | None, pose: Pose, tolerance: float = 1e-3) -> bool:
    return record_pose is not None and all(
        abs(record_pose[key] - value) < tolerance for key, value in zip(("x", "y", "yaw"), pose)
    )


def read_records(path: Path) -> list[dict[str, Any]]:
    """An event log's complete lines so far."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n")]


def goal_status(status: int) -> GoalStatusArray:
    return GoalStatusArray(status_list=[GoalStatus(status=status)])


def odometry(pose: Pose) -> Odometry:
    msg = Odometry()
    msg.header.frame_id = MAP_FRAME
    position, orientation = msg.pose.pose.position, msg.pose.pose.orientation
    position.x, position.y = pose.x, pose.y
    orientation.x, orientation.y, orientation.z, orientation.w = (
        float(v) for v in quaternion_from_euler(0.0, 0.0, pose.yaw)
    )
    return msg


class ScenarioRig:
    """HriScenarioNode on topics of its own, with the nodes it talks to.

    `sim` plays the agent and the simulation: it publishes the robot lines,
    the todo list, resets, /cmd_vel, Nav2's goal status, /joint_states, the
    ground truth, the robot's transforms on /tf_static and the head camera's
    camera_info, and keeps the answers and the oracle's people arrays.
    `commander` is a fake bimanual_commander with only its parameters:
    grasp_detect_min at the value given, none for UNDECLARED, and no node
    at all for None. One MultiThreadedExecutor spins them all, as main()
    spins the scenario.
    """

    def __init__(self, log_path: Path, commander: float | str | None, changes: dict) -> None:
        suffix = random.randint(0, 1_000_000)
        prefix = f"/hri_test_{suffix}"
        self.topics = ScenarioTopics(
            to_human=f"{prefix}/to_human",
            from_human=f"{prefix}/from_human",
            todos=f"{prefix}/todos",
            reset=f"{prefix}/reset",
            cmd_vel=f"{prefix}/cmd_vel",
            nav_status=f"{prefix}/navigate_to_pose/_action/status",
            joint_states=f"{prefix}/joint_states",
            ground_truth=f"{prefix}/ground_truth",
            commander=f"{prefix}/bimanual_commander",
        )
        self.oracle_config = dataclasses.replace(
            QUIET,
            camera_info_topic=f"{prefix}/camera_info",
            people_topic=f"{prefix}/people",
            objects_topic=f"{prefix}/objects",
        )
        # Answers 0.1 s after the line and no mumble, so each answer comes soon.
        spec = make_spec(**deep_update({"answers": {"delay": 0.1, "mumble_once": None}}, changes))
        loaded = LoadedScenario(DEFAULT_FILE, spec, load_locations(LOCATIONS_FILE), LOCATIONS_FILE)
        self.log_path = log_path
        self.log = EventLog(log_path)
        self.commander = None
        if commander is not None:
            self.commander = rclpy.create_node("bimanual_commander", namespace=prefix)
            if commander != UNDECLARED:
                self.commander.declare_parameter(GRASP_DETECT_MIN_PARAM, float(commander))
        self.node = HriScenarioNode(
            loaded,
            self.log,
            use_sim_time=False,
            topics=self.topics,
            oracle_config=self.oracle_config,
        )
        self.sim = sim = rclpy.create_node(f"hri_sim_{suffix}")
        self.to_human_pub = sim.create_publisher(HRIMessage, self.topics.to_human, TO_HUMAN_QOS)
        self.todos_pub = sim.create_publisher(String, self.topics.todos, TODOS_QOS)
        self.reset_pub = sim.create_publisher(Empty, self.topics.reset, 10)
        self.cmd_vel_pub = sim.create_publisher(TwistStamped, self.topics.cmd_vel, 10)
        self.nav_status_pub = sim.create_publisher(
            GoalStatusArray, self.topics.nav_status, qos_profile_action_status_default
        )
        self.joint_states_pub = sim.create_publisher(JointState, self.topics.joint_states, 10)
        self.ground_truth_pub = sim.create_publisher(Odometry, self.topics.ground_truth, 10)
        self.camera_info_pub = sim.create_publisher(
            CameraInfo, self.oracle_config.camera_info_topic, 10
        )
        self.tf_static_pub = sim.create_publisher(TFMessage, "/tf_static", STATIC_TF_QOS)
        self.answers: list[HRIMessage] = []
        self.people: dict[int, PersonDetectionArray] = {}
        self.subscriptions = (
            sim.create_subscription(
                HRIMessage, self.topics.from_human, self.answers.append, FROM_HUMAN_QOS
            ),
            sim.create_subscription(
                PersonDetectionArray, self.oracle_config.people_topic, self._on_people, 10
            ),
        )
        self.executor = MultiThreadedExecutor(num_threads=6)
        for node in (self.node, self.sim, self.commander):
            if node is not None:
                self.executor.add_node(node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self._count = 0

    def wait_for_discovery(self) -> None:
        publishers = (
            self.to_human_pub,
            self.todos_pub,
            self.reset_pub,
            self.cmd_vel_pub,
            self.nav_status_pub,
            self.joint_states_pub,
            self.ground_truth_pub,
            self.camera_info_pub,
        )
        assert wait_until(
            lambda: all(p.get_subscription_count() > 0 for p in publishers)
            and all(s.get_publisher_count() > 0 for s in self.subscriptions)
        ), "The scenario node's subscriptions and publishers were not discovered."

    def records(self) -> list[dict[str, Any]]:
        """The log's complete lines so far."""
        return read_records(self.log_path)

    def find(
        self, type_: str, where: Callable[[dict], bool] | None = None, **fields: Any
    ) -> dict[str, Any] | None:
        """The first record of `type_` with `fields` for which `where` holds, or None."""
        return next(
            (
                r
                for r in self.records()
                if r["type"] == type_
                and all(r.get(key) == value for key, value in fields.items())
                and (where is None or where(r))
            ),
            None,
        )

    def wait_for(
        self,
        type_: str,
        where: Callable[[dict], bool] | None = None,
        timeout: float = 3.0,
        **fields: Any,
    ) -> dict[str, Any]:
        assert wait_until(lambda: self.find(type_, where, **fields) is not None, timeout), (
            f"No {type_} record with {fields} was logged."
        )
        record = self.find(type_, where, **fields)
        assert record is not None
        return record

    def publish_until(
        self,
        publish: Callable[[], None],
        type_: str,
        where: Callable[[dict], bool] | None = None,
    ) -> dict[str, Any]:
        """Publish until a record of `type_` for which `where` holds is logged.

        For the topics the node reads BEST_EFFORT, which may drop a message.
        """
        for _ in range(10):
            publish()
            if wait_until(lambda: self.find(type_, where) is not None, timeout=0.3):
                record = self.find(type_, where)
                assert record is not None
                return record
        raise AssertionError(f"No {type_} record came of 10 messages.")

    def publish_line(self, text: str) -> None:
        """Publish a robot line on /to_human once, as the speak tool does."""
        self.to_human_pub.publish(
            HRIMessage(text=text, communication_id=uuid.uuid4().hex, seq_no=0, seq_end=True)
        )

    def say(self, text: str) -> dict[str, Any]:
        """Publish a robot line on /to_human, as the speak tool does, and return its record."""
        for _ in range(3):
            self.publish_line(text)
            if wait_until(lambda: self.find("robot_line", text=text) is not None, timeout=1.0):
                return self.wait_for("robot_line", text=text)
        raise AssertionError(f"The line {text!r} was not logged.")

    def wait_for_answer(self, text: str) -> HRIMessage:
        assert wait_until(lambda: any(a.text == text for a in self.answers)), (
            f"No answer {text!r} came on {self.topics.from_human}."
        )
        return next(a for a in self.answers if a.text == text)

    def place_robot(self, pose: Pose) -> None:
        """Publish the base at `pose` and the head camera on it, and wait for the node to see it."""
        map_from_base = euler_matrix(0.0, 0.0, pose.yaw)
        map_from_base[:3, 3] = (pose.x, pose.y, 0.0)
        self.tf_static_pub.publish(
            TFMessage(
                transforms=[
                    tf_msg(MAP_FRAME, ROBOT_FRAME, map_from_base),
                    tf_msg(ROBOT_FRAME, CAMERA_FRAME, base_from_camera()),
                ]
            )
        )
        self.wait_for("state", where=lambda r: near(r["pose"], pose))

    def capture(self) -> tuple[CameraInfo, PersonDetectionArray]:
        """Publish the camera_info of new images until the oracle answers one."""
        for _ in range(10):
            # The counter keeps two stamps made in the same tick apart.
            self._count += 1
            now_ns = self.sim.get_clock().now().nanoseconds + self._count
            info = camera_info(TimeMsg(sec=now_ns // 1_000_000_000, nanosec=now_ns % 1_000_000_000))
            self.camera_info_pub.publish(info)
            if wait_until(lambda: now_ns in self.people, timeout=0.5):
                return info, self.people[now_ns]
        raise AssertionError("None of 10 camera_info messages was answered.")

    def commander_value(self) -> float:
        assert self.commander is not None
        return self.commander.get_parameter(GRASP_DETECT_MIN_PARAM).value

    def close(self) -> None:
        self.node.close()
        self.executor.shutdown()
        self.thread.join(timeout=2)
        for node in (self.node, self.sim, self.commander):
            if node is not None:
                node.destroy_node()
        self.log.close()

    def _on_people(self, msg: PersonDetectionArray) -> None:
        stamp = msg.header.stamp
        self.people[stamp.sec * 1_000_000_000 + stamp.nanosec] = msg


@pytest.fixture
def make_rig(tmp_path, monkeypatch):
    """A ScenarioRig factory that closes every rig it made.

    `commander` is the fake commander's grasp_detect_min, UNDECLARED or None;
    the keyword arguments change the default scenario.
    """
    # Retry the commander every 0.2 s, so that a test waits for it tenths of a
    # second, not seconds.
    monkeypatch.setattr(hri_scenario, "COMMANDER_RETRY_S", 0.2)
    rigs: list[ScenarioRig] = []

    def make(commander: float | str | None = GRASP_DETECT_MIN, **changes: Any) -> ScenarioRig:
        rig = ScenarioRig(tmp_path / f"events_{len(rigs)}.jsonl", commander, changes)
        rigs.append(rig)
        rig.wait_for_discovery()
        return rig

    yield make
    for rig in rigs:
        rig.close()


@dataclasses.dataclass(frozen=True)
class IdleNode:
    """HriScenarioNode on topics of its own that no executor spins, and its log.

    The test runs the node's callbacks itself, on its own thread.
    """

    node: HriScenarioNode
    topics: ScenarioTopics
    log_path: Path

    def records(self) -> list[dict[str, Any]]:
        return read_records(self.log_path)


@pytest.fixture
def make_idle_node(tmp_path):
    """An IdleNode factory that closes every node it made.

    The keyword arguments change the default scenario.
    """
    made: list[tuple[IdleNode, EventLog]] = []

    def make(**changes: Any) -> IdleNode:
        prefix = f"/hri_idle_{random.randint(0, 1_000_000)}"
        topics = ScenarioTopics(
            **{name: prefix + topic for name, topic in dataclasses.asdict(ScenarioTopics()).items()}
        )
        oracle_config = dataclasses.replace(
            QUIET,
            camera_info_topic=f"{prefix}/camera_info",
            people_topic=f"{prefix}/people",
            objects_topic=f"{prefix}/objects",
        )
        log_path = tmp_path / f"events_{len(made)}.jsonl"
        log = EventLog(log_path)
        spec = make_spec(**changes)
        loaded = LoadedScenario(DEFAULT_FILE, spec, load_locations(LOCATIONS_FILE), LOCATIONS_FILE)
        node = HriScenarioNode(
            loaded, log, use_sim_time=False, topics=topics, oracle_config=oracle_config
        )
        idle = IdleNode(node, topics, log_path)
        made.append((idle, log))
        return idle

    yield make
    for idle, log in made:
        idle.node.close()
        idle.node.destroy_node()
        log.close()


@pytest.fixture
def idle_node(make_idle_node):
    return make_idle_node()


class StubGrasp:
    """The node's GraspParameter, whose hang() calls return the events given, in order."""

    def __init__(self, *events: threading.Event | None) -> None:
        self.events = list(events)

    def hang(self) -> threading.Event | None:
        return self.events.pop(0)

    def restore(self, _cause: str) -> None:
        return None

    def finish(self) -> None:
        return None

    def close(self) -> None:
        pass


def test_robot_lines_on_to_human_get_answers_on_from_human(make_rig):
    rig = make_rig()
    rig.place_robot(DOOR)

    line = rig.say("Hello, I am Walkie. What is your name?")

    answer = rig.wait_for_answer("My name is Sam.")
    assert answer.seq_end and answer.communication_id
    record = rig.wait_for("answer")
    assert record == {
        "type": "answer",
        "t": record["t"],
        "guest": "guest1",
        "text": "My name is Sam.",
        "to": line["text"],
        "asked_t": line["t"],
        "asks": [NAME],
        "mumbled": False,
    }
    # Sent by the first tick after the delay of 0.1 s.
    assert 0.1 <= record["t"] - line["t"] < 0.5
    assert (line["guest"], line["question"], line["asks"], line["offer"]) == (
        "guest1",
        True,
        [NAME],
        False,
    )
    assert near(line["pose"], DOOR)
    # The guest spot is 0.83 deg left of the door pose's heading.
    assert line["bearings_deg"]["guest1"] == pytest.approx(0.83, abs=0.05)

    records = rig.records()
    start = records[0]
    assert [r["type"] for r in records[:3]] == ["start", "person", "person"]
    assert (start["world"], start["file"], start["seed"], start["locations"]) == (
        "chula_pat14_6",
        str(DEFAULT_FILE),
        7,
        str(LOCATIONS_FILE),
    )
    assert start["door"] == {"x": 2.35, "y": -0.17, "yaw": 3.01}
    assert start["living_room"] == {"x": 1.85, "y": -1.71, "yaw": 0.0}
    assert start["places"] == {
        "sofa.left": [3.115, -1.1],
        "sofa.middle": [3.115, -1.65],
        "sofa.right": [3.115, -2.2],
    }
    assert start["scenario"]["answers"] == {"delay": 0.1, "mumble_once": None}
    assert start["oracle"]["camera_info_topic"] == rig.oracle_config.camera_info_topic
    assert datetime.fromisoformat(start["wall_time"]).tzinfo is not None
    state = rig.wait_for("state", where=lambda r: r["t"] > line["t"])
    assert state["conversation"] is True and state["talking"]["guest"] == "guest1"
    assert state["talking"]["bearing_deg"] == pytest.approx(0.83, abs=8.0)


def test_a_seat_offer_over_ros_judges_the_heading_by_tf(make_rig):
    rig = make_rig()
    rig.place_robot(DOOR)
    rig.say("What is your name?")
    rig.wait_for("person", person="guest1", state=TALKING)
    middle = (3.115, -1.65)
    pose = Pose(
        LIVING_ROOM.x,
        LIVING_ROOM.y,
        math.atan2(middle[1] - LIVING_ROOM.y, middle[0] - LIVING_ROOM.x),
    )
    rig.place_robot(pose)
    rig.wait_for("person", person="guest1", state=FOLLOWING)

    rig.say("Please sit here.")

    taken = rig.wait_for("seat_taken")
    assert (taken["guest"], taken["place"]) == ("guest1", "sofa.middle")
    assert near(taken["pose"], pose)
    rig.wait_for("person", person="guest2", state=AT_DOOR)


def test_the_state_and_cmd_vel_records_carry_nav_arms_grippers_and_ground_truth(make_rig):
    rig = make_rig()
    tables = load_arm_poses().poses["home"].tables
    model = load_robot_model()
    fingers = {arm: model.groups[GRIPPER_GROUPS[arm]][0] for arm in ARMS}
    joints = {**tables["left"], **tables["right"], fingers["left"]: 0.044, fingers["right"]: 0.012}
    truth = Pose(1.9, -1.6, 0.5)
    rig.nav_status_pub.publish(goal_status(GoalStatus.STATUS_EXECUTING))

    def publish_state() -> None:
        rig.joint_states_pub.publish(JointState(name=list(joints), position=list(joints.values())))
        rig.ground_truth_pub.publish(odometry(truth))

    state = rig.publish_until(
        publish_state,
        "state",
        where=lambda r: r["nav_active"] and r["arms_travel_safe"] and r["ground_truth"],
    )

    assert near(state["ground_truth"], truth)
    assert state["arms"] == {
        "left": {"pose": "home", "travel_safe": True},
        "right": {"pose": "home", "travel_safe": True},
    }
    assert state["grippers"] == {"left": 0.044, "right": 0.012}
    assert (state["talking"], state["conversation"]) == (None, False)

    twist = TwistStamped()
    twist.header.stamp = TimeMsg(sec=12, nanosec=500_000_000)
    twist.header.frame_id = "base_link"
    twist.twist.linear.x = 0.25
    twist.twist.angular.z = -0.4
    record = rig.publish_until(lambda: rig.cmd_vel_pub.publish(twist), "cmd_vel")
    assert record == {
        "type": "cmd_vel",
        "t": record["t"],
        "stamp": 12.5,
        "frame_id": "base_link",
        "linear": [0.25, 0.0, 0.0],
        "angular": [0.0, 0.0, -0.4],
        "nav_active": True,
        "arms_travel_safe": True,
    }

    rig.nav_status_pub.publish(goal_status(GoalStatus.STATUS_SUCCEEDED))
    rig.wait_for("state", where=lambda r: r["t"] > record["t"] and r["nav_active"] is False)
    # A ground truth older than 1 s is unknown.
    rig.wait_for("state", where=lambda r: r["t"] > record["t"] and r["ground_truth"] is None)


def test_each_state_stream_has_a_mutually_exclusive_group_of_its_own(idle_node):
    groups = {sub.topic_name: sub.callback_group for sub in idle_node.node.subscriptions}
    topics = idle_node.topics

    streams = [
        groups[topic]
        for topic in (topics.cmd_vel, topics.nav_status, topics.joint_states, topics.ground_truth)
    ]

    assert all(isinstance(group, MutuallyExclusiveCallbackGroup) for group in streams)
    assert len({id(group) for group in streams}) == 4
    # The scenario's inputs share a group of their own.
    inputs = groups[topics.to_human]
    assert groups[topics.todos] is inputs and groups[topics.reset] is inputs
    assert all(group is not inputs for group in streams)


def test_the_node_logs_a_clock_that_goes_back_and_drops_the_ground_truth_of_before(
    idle_node, monkeypatch
):
    node = idle_node.node
    clock = FakeClock(node._now())
    monkeypatch.setattr(node, "_now", clock)
    truth = Pose(1.9, -1.6, 0.5)
    node._on_ground_truth(odometry(truth))
    node._on_tick()
    assert near(idle_node.records()[-1]["ground_truth"], truth)
    old_t = clock.t

    # The simulation restarts, and its clock goes back 100 s.
    new_t = clock.advance(-100.0)
    node._on_tick()

    jump, state = idle_node.records()[-2:]
    # The log keeps times to the millisecond.
    assert jump == {
        "type": "clock_jump",
        "t": round(new_t, 3),
        "from": round(old_t, 3),
        "to": round(new_t, 3),
        "pending": 0,
    }
    assert (state["type"], state["t"], state["ground_truth"]) == ("state", round(new_t, 3), None)
    # A ground truth stamped after the node's time was taken before the jump.
    clock.advance(0.5)
    node._on_ground_truth(odometry(truth))
    clock.advance(-0.4)
    node._on_tick()
    state = idle_node.records()[-1]
    assert (state["type"], state["ground_truth"]) == ("state", None)
    assert len(of_type(idle_node.records(), "clock_jump")) == 1


def test_after_close_no_callback_calls_into_the_scenario(idle_node, monkeypatch):
    node = idle_node.node
    node.close()
    called: list[str] = []
    for name in ("robot_pose", "robot_said", "todos_changed", "reset", "tick", "scene"):
        monkeypatch.setattr(
            node.scenario, name, lambda *_args, _name=name, **_kwargs: called.append(_name)
        )

    node._on_to_human(HRIMessage(text="Could you hand me your bag, please?"))
    node._on_todos(String(data=json.dumps(todos(("t2", INTRODUCTION, "in_progress")))))
    node._on_reset(Empty())
    node._on_tick()
    scene = node._scene_at(node.get_clock().now())

    assert called == []
    assert scene == Scene()


def test_a_bag_request_skipped_while_a_hang_is_pending_keeps_the_answers_waiting(
    make_idle_node, monkeypatch
):
    # Far beyond the test's run, so the wait cannot end at its deadline.
    monkeypatch.setattr(hri_scenario, "HANG_WAIT_S", 60.0)
    idle = make_idle_node(bag={"owner": "guest1"}, answers={"delay": 0.1, "mumble_once": None})
    node = idle.node
    clock = FakeClock(node._now())
    monkeypatch.setattr(node, "_now", clock)
    # The second request's hang is skipped at once, as when the commander
    # went away while the first one's was still pending.
    pending = threading.Event()
    monkeypatch.setattr(node, "_grasp", StubGrasp(pending, None))
    node._on_to_human(HRIMessage(text="Could you hand me your bag, please?"))
    node._on_to_human(HRIMessage(text="Please give me your bag."))
    clock.advance(0.5)

    node._on_tick()

    assert of_type(idle.records(), "answer") == []
    pending.set()
    node._on_tick()
    answers = of_type(idle.records(), "answer")
    assert [a["text"] for a in answers] == ["It is on your hand."] * 2


def test_until_tf_has_the_robot_the_node_logs_once_that_it_waits_for_amcl(make_rig, caplog):
    caplog.set_level(logging.INFO, logger=hri_scenario.__name__)
    rig = make_rig()

    assert wait_until(lambda: len(of_type(rig.records(), "state")) >= 3)

    assert all(r["pose"] is None for r in of_type(rig.records(), "state"))
    messages = [r.getMessage() for r in caplog.records if r.name == hri_scenario.__name__]
    assert len([m for m in messages if m.startswith("Waiting for AMCL")]) == 1
    rig.place_robot(DOOR)
    messages = [r.getMessage() for r in caplog.records if r.name == hri_scenario.__name__]
    assert len([m for m in messages if m.startswith("Waiting for AMCL")]) == 1
    assert len([m for m in messages if m.endswith("the robot pose is known.")]) == 1


def test_the_nodes_oracle_draws_the_guest_at_the_door_with_a_trusted_face(make_rig):
    rig = make_rig()
    rig.place_robot(DOOR)
    track = rig.wait_for("person", person="guest1")["track_id"]

    info, people = rig.capture()

    assert people.header == info.header and people.camera_info == info
    # The host sits behind the robot.
    [guest] = people.people
    assert guest.track_id == track
    x1, y1, x2, y2 = bbox_xyxy(guest.face_bbox)
    assert 0.0 < x1 < x2 < WIDTH and 0.0 < y1 < y2 < HEIGHT
    assert face_trusted(guest, people.camera_info)
    x1, _, x2, _ = bbox_xyxy(guest.bbox)
    assert abs((x1 + x2) / 2 - CX) < 10.0


def test_the_latched_todo_list_is_logged(make_rig):
    rig = make_rig()
    items = todos(("t2", INTRODUCTION, "in_progress"))

    rig.todos_pub.publish(String(data=json.dumps(items)))

    assert rig.wait_for("todos")["items"] == items


def test_a_bag_request_hangs_the_bag_before_its_answer_and_a_reset_restores_it(make_rig):
    rig = make_rig(bag={"owner": "guest1"})
    commander = rig.wait_for("commander")
    assert commander["node"] == rig.topics.commander
    assert commander["grasp_detect_min"] == GRASP_DETECT_MIN

    rig.say("Could you hand me your bag, please?")

    rig.wait_for_answer("It is on your hand.")
    assert rig.commander_value() == HUNG_VALUE
    found = [r["type"] for r in rig.records()]
    assert found.index("bag_request") < found.index("bag_hung") < found.index("answer")
    assert rig.find("bag_hung")["value"] == HUNG_VALUE

    rig.reset_pub.publish(Empty())

    assert rig.wait_for("reset")["dropped"] == 0
    restored = rig.wait_for("bag_restored")
    assert (restored["value"], restored["cause"]) == (GRASP_DETECT_MIN, "reset")
    assert rig.commander_value() == GRASP_DETECT_MIN


def test_the_bag_answer_waits_for_a_commander_slower_than_the_answer_delay(make_rig):
    rig = make_rig(bag={"owner": "guest1"})
    rig.wait_for("commander")
    # The commander takes the hang only once the test lets it, long after
    # the 0.1 s answer delay.
    gate = threading.Event()

    def held(_parameters) -> SetParametersResult:
        gate.wait(2.0)
        return SetParametersResult(successful=True)

    rig.commander.add_on_set_parameters_callback(held)

    said = rig.say("Could you hand me your bag, please?")["t"]

    assert not wait_until(lambda: rig.answers, timeout=0.5)
    # The robot's state is still logged at 10 Hz while the answer waits.
    assert len([r for r in of_type(rig.records(), "state") if r["t"] > said]) >= 3
    gate.set()
    rig.wait_for_answer("It is on your hand.")
    found = [r["type"] for r in rig.records()]
    assert found.index("bag_hung") < found.index("answer")


def test_closing_restores_the_hung_bag_and_ends_the_log(make_rig):
    rig = make_rig(bag={"owner": "guest1"})
    rig.wait_for("commander")
    rig.say("Could you hand me your bag, please?")
    rig.wait_for_answer("It is on your hand.")

    rig.node.close()

    records = rig.records()
    found = [r["type"] for r in records]
    assert found[-1] == "end"
    restored = records[found.index("bag_restored")]
    assert (restored["value"], restored["cause"]) == (GRASP_DETECT_MIN, "exit")
    assert rig.commander_value() == GRASP_DETECT_MIN
    # The node still spins, but a bag request after the end hangs nothing
    # and gets no answer, and nothing comes after the end.
    rig.publish_line("Could you hand me your bag, please?")
    assert not wait_until(
        lambda: len(rig.answers) > 1 or rig.commander_value() != GRASP_DETECT_MIN, timeout=0.5
    )
    assert rig.records() == records


def test_after_close_no_input_reaches_the_commander_or_the_agent(make_rig):
    rig = make_rig(bag={"owner": "guest1"}, answers={"delay": 0.5})
    rig.wait_for("commander")
    sets: list[Any] = []

    def taken(parameters) -> SetParametersResult:
        sets.extend(parameter.value for parameter in parameters)
        return SetParametersResult(successful=True)

    rig.commander.add_on_set_parameters_callback(taken)
    # Its answer is due 0.5 s after the line, after the close.
    line = rig.say("What is your name?")

    rig.node.close()
    end = rig.records()[-1]
    assert end["type"] == "end"
    assert end["t"] < line["t"] + 0.5, "The node closed after the answer was due."
    rig.publish_line("Could you hand me your bag, please?")
    rig.reset_pub.publish(Empty())

    # Until well after the answer was due.
    assert not wait_until(lambda: rig.answers or sets, timeout=0.8)
    assert rig.commander_value() == GRASP_DETECT_MIN
    assert rig.records()[-1]["type"] == "end"


def test_an_exit_restore_the_commander_refuses_once_is_sent_again_within_the_wait(make_rig):
    rig = make_rig(bag={"owner": "guest1"})
    rig.wait_for("commander")
    rig.say("Could you hand me your bag, please?")
    rig.wait_for("bag_hung")
    refused: list[Any] = []

    def refuse_once(parameters) -> SetParametersResult:
        if refused:
            return SetParametersResult(successful=True)
        refused.extend(parameter.value for parameter in parameters)
        return SetParametersResult(successful=False, reason="the arm is moving")

    rig.commander.add_on_set_parameters_callback(refuse_once)

    rig.node.close()

    records = rig.records()
    found = [r["type"] for r in records]
    assert refused == [GRASP_DETECT_MIN]
    retry = records[found.index("bag_retry")]
    assert (retry["action"], retry["reason"]) == (RESTORE, "the arm is moving")
    restored = records[found.index("bag_restored")]
    assert (restored["value"], restored["cause"]) == (GRASP_DETECT_MIN, "exit")
    assert found.index("bag_retry") < found.index("bag_restored") < len(found) - 1
    assert found[-1] == "end" and "bag_skipped" not in found
    assert rig.commander_value() == GRASP_DETECT_MIN


def test_a_second_bag_request_while_the_hang_is_pending_waits_for_the_same_hang(
    make_rig, monkeypatch
):
    # Beyond the test's run, so the wait cannot end at its deadline on a slow machine.
    monkeypatch.setattr(hri_scenario, "HANG_WAIT_S", 10.0)
    rig = make_rig(bag={"owner": "guest1"})
    rig.wait_for("commander")
    # The commander takes the hang only once the test lets it.
    gate = threading.Event()

    def held(_parameters) -> SetParametersResult:
        gate.wait(5.0)
        return SetParametersResult(successful=True)

    rig.commander.add_on_set_parameters_callback(held)
    rig.say("Could you hand me your bag, please?")

    rig.say("Please give me your bag.")

    assert not wait_until(lambda: rig.answers, timeout=0.5)
    gate.set()
    assert wait_until(lambda: len(rig.answers) == 2)
    found = [r["type"] for r in rig.records()]
    assert found.count("bag_hung") == 1 and found.count("answer") == 2
    assert found.index("bag_hung") < found.index("answer")
    assert "bag_skipped" not in found and "bag_unconfirmed" not in found


def test_a_hang_not_taken_by_the_deadline_is_unconfirmed_and_still_lands(make_rig, monkeypatch):
    monkeypatch.setattr(hri_scenario, "HANG_WAIT_S", 0.3)
    rig = make_rig(bag={"owner": "guest1"})
    rig.wait_for("commander")
    gate = threading.Event()

    def held(_parameters) -> SetParametersResult:
        gate.wait(5.0)
        return SetParametersResult(successful=True)

    rig.commander.add_on_set_parameters_callback(held)

    rig.say("Could you hand me your bag, please?")

    unconfirmed = rig.wait_for("bag_unconfirmed")
    assert unconfirmed["waited_s"] >= 0.3
    assert unconfirmed["reason"] == (
        f"{rig.topics.commander} had not taken the hang within 0.3 s, so the answer went out first"
    )
    rig.wait_for_answer("It is on your hand.")
    # The hang is still wanted, and is logged once the commander takes it.
    gate.set()
    rig.wait_for("bag_hung")
    found = [r["type"] for r in rig.records()]
    assert found.index("bag_unconfirmed") < found.index("answer") < found.index("bag_hung")
    assert rig.commander_value() == HUNG_VALUE


def test_without_a_commander_the_bag_is_skipped_and_still_answered(make_rig):
    rig = make_rig(commander=None, bag={"owner": "guest1"})

    rig.say("Could you hand me your bag, please?")

    rig.wait_for_answer("It is on your hand.")
    skipped = rig.wait_for("bag_skipped")
    commander = rig.topics.commander
    assert (skipped["action"], skipped["reason"]) == (
        "hang",
        f"no {commander}: {commander}/set_parameters has no server",
    )
    rig.node.close()
    assert not rig.find("bag_restored") and rig.records()[-1]["type"] == "end"


def test_a_commander_without_grasp_detect_min_skips_the_bag(make_rig):
    rig = make_rig(commander=UNDECLARED, bag={"owner": "guest1"})
    commander = rig.wait_for("commander")
    assert commander["grasp_detect_min"] is None

    rig.say("Could you hand me your bag, please?")

    skipped = rig.wait_for("bag_skipped")
    assert (skipped["action"], skipped["reason"]) == (
        "hang",
        f"{rig.topics.commander} has no parameter grasp_detect_min",
    )
    rig.wait_for_answer("It is on your hand.")


def test_a_hung_value_left_from_another_run_is_restored_at_start(make_rig):
    rig = make_rig(commander=HUNG_VALUE)

    assert rig.wait_for("commander")["grasp_detect_min"] == HUNG_VALUE

    restored = rig.wait_for("bag_restored")
    assert (restored["value"], restored["cause"]) == (GRASP_DETECT_MIN, "start")
    assert rig.commander_value() == GRASP_DETECT_MIN


# -- The entry point ----------------------------------------------------------------------


def fail_if_ros_starts(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("ROS started for a scenario that cannot run.")


def test_main_refuses_a_bad_scenario_before_ros_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(hri_scenario.rclpy, "init", fail_if_ros_starts)
    missing = tmp_path / "missing.yaml"
    log = tmp_path / "events.jsonl"

    with pytest.raises(SystemExit) as raised:
        hri_scenario.main(["hri_scenario", str(missing), "--log", str(log)])

    assert raised.value.code == f"Cannot read the scenario {missing}: No such file or directory."
    path = write_scenario(tmp_path, deep_update(read_yaml(DEFAULT_FILE), {"bag": {"owner": "x"}}))
    with pytest.raises(SystemExit) as raised:
        hri_scenario.main(["hri_scenario", str(path), "--log", str(log)])
    assert "bag.owner must be a guest of this file" in raised.value.code
    assert not log.exists()


def test_main_refuses_unreadable_arm_poses_before_ros_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(hri_scenario.rclpy, "init", fail_if_ros_starts)

    def broken() -> None:
        raise ArmPoseError("config/arm_poses.toml", ["poses.home: no left table"])

    monkeypatch.setattr(hri_scenario, "load_arm_poses", broken)
    log = tmp_path / "events.jsonl"

    with pytest.raises(SystemExit) as raised:
        hri_scenario.main(["hri_scenario", str(DEFAULT_FILE), "--log", str(log)])

    assert raised.value.code.startswith(
        "Cannot read the arm poses for the state records: config/arm_poses.toml: 1 problem"
    )
    assert not log.exists()
