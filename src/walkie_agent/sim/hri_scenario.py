"""The HRI scenario: virtual guests, seats and answers in walkie-ros's MuJoCo world.

    python -m walkie_agent.sim.hri_scenario config/scenarios/hri.yaml [--log path.jsonl]

walkie-ros's MuJoCo worlds have no person bodies, so the people of the HRI
task exist only here (hri.md, "In simulation"). A scenario file places them
in the map frame of the world it names, chula_pat14_6 by default, and the
node plays them against the agent:

  - OraclePerception draws the people and the seats as walkie-vision's
    arrays, for each image of the robot's head camera. An image older than
    the scene's history of HISTORY_S gets none.
  - The current guest answers each robot line on /to_human that asks
    something, on /from_human after a delay. The guest at the door paces
    while talking.
  - The guest who talked follows the robot once its base is more than
    0.5 m from the door pose, and is then out of view. A line that offers a
    seat seats the following guest, if the robot faces a free place.
  - The bag owner's hand-over request sets the commander's grasp_detect_min
    to -1, so that the next empty close reads as holding the bag.
  - Every event, and the robot's state at 10 Hz, goes to the event log.

HriScenario is the scenario's logic, with no ROS and an injected clock.
HriScenarioNode runs it on ROS with use_sim_time, so its timer and the log
follow the simulation's /clock. The node only reads /cmd_vel; it is a
simulation tool, not an agent tool, so the agent's forbidden topic names
do not apply to it.

Rules
-----
- Lines. classify_line() reads each robot line sentence by sentence: a
  question asks for the name, the drink, or something "other". A sentence
  that mentions a bag asks for the bag when it is a question, holds
  "please", or opens with a request verb such as "put" or "hand", so "Put
  your bag on my hand." asks for it and "I will put your bag down later."
  does not.
- Answers. The current guest is the one at the door, talking or following.
  The name gets "My name is <name>.", the drink "I would like <drink>.",
  the bag owner's bag "It is on your hand.", and anything else "Okay.",
  logged as an extra question. answers.mumble_once names one question,
  such as guest1.drink, whose first ask is answered with "Hmm." instead.
  The answer goes out answers.delay seconds after the line.
- Guests. Guest 1 waits at the guest spot from the start, facing the
  robot. A guest starts talking at the first question, and guest k+1
  arrives once guest k is seated. The bystander walks its path once,
  bystander.delay seconds after the run's first question.
- Onlookers. People at the door who are not the guest, such as one far
  behind them in the hallway and one beside the robot, for
  find_people(in_front=True) to leave out. They arrive with each guest,
  each on a new track after the guest's, stand at their place facing the
  robot, and leave when the guest starts to follow. They never sit,
  follow, answer, take a seat offer or join the seat switch.
- Seat offers. A line that mentions sitting or a seat, while a guest
  follows, is an offer. It seats the guest at the place with the smallest
  bearing from the robot's base, by TF, if that bearing is within
  offer_window_deg and the place is free; on a tie, the place first in the
  file wins. Otherwise it is a bad offer and the guest keeps following.
- Seat switch. When the todo item whose text contains "introduce" becomes
  in_progress or waiting, the switch fires as soon as every person in
  seat_switch.people is seated. swap exchanges two people's places; move
  takes each person to seat_switch.to, or to the first free place in file
  order; off does nothing. It fires at most once, and is skipped when the
  item finishes or leaves the list first.
- The bag. Each request hangs the bag, except the first one with
  bag.early_answer, so that the first close reads empty. Its answer waits
  up to HANG_WAIT_S for the commander to take the hang. On a reset
  (/walkie/reset) and on exit, grasp_detect_min goes back to the value
  read at start. The node sends the commander one request at a time. One
  that fails, or gets no answer within REQUEST_TIMEOUT_S, is sent again
  within COMMANDER_RETRY_S, until the commander holds the value wanted. A
  hang the commander refuses is given up. On exit the restore gets
  RESTORE_WAIT_S from when it can first go out, after the request in
  flight is answered or dropped, so the node waits at most
  REQUEST_TIMEOUT_S + RESTORE_WAIT_S for it.
- The clock. A clock that goes back, as when the simulation restarts,
  moves every time the scenario keeps with it and is logged as
  clock_jump. An image drawn while the clock went back gets no arrays. A
  run whose log has a clock_jump is not a valid run, and the pass check
  fails it.

Event log
---------
One JSON object per line, flushed as it is written. Every record has
"type" and "t", the node's clock in seconds (the simulation's time under
use_sim_time) to the millisecond. Records written before the first /clock
message have t = 0. Poses are {"x", "y", "yaw"} in the map frame, in meters
and radians, or null when unknown. Fields ending in _deg are degrees. A
bearing is the direction of a point from the robot's base against the
robot's heading, positive to the left, in (-180, 180]. People are named by
id: guest1, guest2, ... and onlooker1, onlooker2, ... in file order, the
seated people's own ids, and "bystander". A place is "<seat>.<part>" on a
sofa, such as "sofa.left", and the seat's id on any other seat. A number
that is not finite is null.

  start          Always the first line. world, file, seed; door and
                 living_room (poses from the locations file); locations
                 (that file); places ({place: [x, y]}); scenario (the
                 validated file with its defaults); guests ({guest:
                 {name, drink}} in id order); bag_owner (the guest who
                 brings the bag, or null); onlookers (their ids); oracle
                 (the OracleConfig in use); wall_time (ISO 8601).
  person         A person's state changed. person; name (null for the
                 seated people); state: at_door, talking, following or
                 seated; track_id (null while following); place (null
                 unless seated); x, y (the guest spot at the door, the
                 place when seated, null while following).
  robot_line     A line on /to_human. text; guest (the current guest or
                 null); question; asks (name, drink, bag, other, in order);
                 offer (whether it is a seat offer); pose; bearings_deg
                 ({person: bearing} of everyone the scene shows, null
                 without a pose).
  answer         An answer sent on /from_human. guest, text, to (the line
                 it answers), asked_t (that line's t), asks, mumbled.
  extra_question A line asked something other than the current guest's
                 name or drink, or the bag owner's bag. guest, text.
  seat_taken     An offer seated the following guest. guest, place,
                 reason (null), taken_by (null), pose, heading_deg (the
                 robot's yaw), bearings_deg ({place: bearing} of every
                 place), window_deg.
  bad_offer      An offer that seated nobody, with the fields of
                 seat_taken. reason "taken" (place is the nearest place,
                 taken_by who sits there) or "no place within the window"
                 (place and taken_by null).
  seat_switch    mode, item (the todo item's id), old and new ({person:
                 place}).
  seat_switch_skipped   mode, item, reason.
  bag_request    The bag owner was asked for the bag. guest, count (the
                 requests so far), hang (whether this one hangs the bag),
                 early_answer.
  bag_hung       The commander took grasp_detect_min = value (-1.0).
                 already_held: true when it held -1 still, as after a
                 restore it refused, so that no set was sent.
  bag_restored   The commander took back value. cause: reset, exit, or
                 start (a negative value found at start, left by a run
                 that did not restore it).
  bag_retry      A hang or restore that could not be sent, failed, or got
                 no answer within REQUEST_TIMEOUT_S. action (hang or
                 restore), reason. The node tries again within
                 COMMANDER_RETRY_S, until the commander holds the value
                 wanted.
  bag_unconfirmed  The commander had not taken a hang when the answer's
                 wait ended, so the answer went out first. waited_s,
                 reason. The hang is still wanted: bag_hung follows once
                 the commander takes it.
  bag_skipped    A bag action given up. action (hang or restore), reason:
                 no commander at the request, no such parameter, a
                 refused hang, a reset or exit before the commander took
                 the hang, a request while the node closes, or a restore
                 on exit the commander had not taken RESTORE_WAIT_S after
                 it could first go out.
  commander      The commander answered the read of grasp_detect_min.
                 The node sends the read once the commander is found,
                 looking every COMMANDER_RETRY_S, and again when it fails
                 or gets no answer within REQUEST_TIMEOUT_S. node,
                 grasp_detect_min (as read; null when the commander has no
                 such parameter).
  clock_jump     The node's clock went back, as when the simulation
                 restarts. from and to (the two times), pending (the
                 answers still queued, each due as long after the jump as
                 it was before it). The run is not valid.
  todos          A message on /walkie/todos. items, the list as published.
  reset          A message on /walkie/reset. dropped: the answers not sent.
  bystander      state (walking or gone), track_id, x, y.
  onlooker       An onlooker arrived or left. onlooker; state (standing or
                 gone); track_id (the track that ended when gone); x, y
                 (its place).
  state          At 10 Hz. pose (TF map -> base_footprint); ground_truth
                 (/mujoco_ground_truth/odom, null when none arrived in the
                 last 1 s, or since the clock went back); talking ({guest,
                 x, y, bearing_deg} of the guest talking at the door, or
                 null); conversation (whether a guest is talking);
                 nav_active (a navigate_to_pose goal is accepted,
                 executing or canceling; false before any status);
                 arms_travel_safe (true, false, or null when unknown, as
                 arm/state.py judges /joint_states); arms ({left, right:
                 {pose, travel_safe} or null}); grippers ({left, right: the
                 finger joint's travel in meters, or null}).
  cmd_vel        Every /cmd_vel message. stamp (its header's, in seconds),
                 frame_id, linear and angular ([x, y, z], unrounded),
                 nav_active, arms_travel_safe.
  end            Always the last line, written on exit.
"""

import argparse
import dataclasses
import json
import logging
import math
import re
import signal
import sys
import threading
import time
import tomllib
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, get_args

import rclpy
import yaml
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rai_interfaces.msg import HRIMessage
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_action_status_default,
    qos_profile_sensor_data,
)
from rclpy.signals import SignalHandlerOptions
from rclpy.task import Future
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformException, TransformListener
from tf_transformations import euler_from_quaternion

from walkie_agent.arm.poses import (
    ARMS,
    GRASP_DETECT_MIN,
    GRIPPER_GROUPS,
    ArmPoseError,
    ArmPoses,
    RobotModel,
    load_arm_poses,
    load_robot_model,
)
from walkie_agent.arm.state import ArmsState, ArmState
from walkie_agent.jsonl import EventLog
from walkie_agent.locations import Location, load_locations
from walkie_agent.perception.geometry import MAP_FRAME
from walkie_agent.perception.seating import DEFAULT_SEATING_CONFIG
from walkie_agent.sim.oracle_perception import OracleConfig, OraclePerception
from walkie_agent.sim.scene import (
    DEFAULT_SEAT_TOP_M,
    STANDARD_HEIGHT_M,
    MapBox,
    Scene,
    SceneObject,
    ScenePerson,
)
from walkie_agent.todo import OPEN_STATUSES, TodoStatus

logger = logging.getLogger(__name__)

# Nav2's robot frame on walkie-ros, as walkie_agent.tools.navigation has it.
# Repeated here so that a simulation tool does not import the agent's tools.
ROBOT_FRAME = "base_footprint"


@dataclass(frozen=True)
class ScenarioTopics:
    """The node's topics, and the commander node whose parameter hangs the bag."""

    to_human: str = "/to_human"
    from_human: str = "/from_human"
    todos: str = "/walkie/todos"
    reset: str = "/walkie/reset"
    cmd_vel: str = "/cmd_vel"
    nav_status: str = "/navigate_to_pose/_action/status"
    joint_states: str = "/joint_states"
    ground_truth: str = "/mujoco_ground_truth/odom"
    commander: str = "/bimanual_commander"


DEFAULT_TOPICS = ScenarioTopics()

# The commander's parameter, and the value that makes an empty close read
# as holding something: no settled finger gap lies below it.
GRASP_DETECT_MIN_PARAM = "grasp_detect_min"
HUNG_VALUE = -1.0

# How often the node reads the robot's pose, sends the answers due and logs
# the robot's state, in seconds of the node's clock.
STATE_PERIOD_S = 0.1
# How often the node sends the commander a request that failed or got no
# answer, until the commander holds the value wanted, in seconds of the
# computer's clock.
COMMANDER_RETRY_S = 1.0
# How long a request to the commander may go unanswered before the node
# drops it and may send another, in seconds of the computer's clock. A
# service call has no timeout of its own, so a lost answer would otherwise
# block every later request.
REQUEST_TIMEOUT_S = 2.0
# How long a bag request waits for the commander before its answer may go
# out, in seconds.
HANG_WAIT_S = 2.0
# How long close() waits for the commander to take the restore, in seconds
# of the computer's clock. Counted from when the restore can first go out,
# once the request in flight is answered or dropped, so close() waits at
# most REQUEST_TIMEOUT_S + RESTORE_WAIT_S. At least REQUEST_TIMEOUT_S, so a
# restore sent gets the same time to answer as any other request.
RESTORE_WAIT_S = 2.0
# How far back the scene can be drawn as of a camera image's stamp, in
# seconds. TF's wait does not bound an image's age, since an image may
# reach the oracle late, so an older image is not drawn.
HISTORY_S = 5.0
# The guest who talked follows the robot once its base is this far from the
# door pose, in meters: further than Nav2's goal tolerance and AMCL's error.
GUIDE_DISTANCE_M = 0.5
# A ground-truth pose older than this is logged as unknown, in seconds.
GROUND_TRUTH_MAX_AGE_S = 1.0

# The places of a sofa: the parts find_seats splits a sofa's box into, with
# the seating config's default middle place.
SOFA_PLACES: tuple[str, ...] = ("left", "middle", "right")
# The locations the scenario reads from the locations file.
REQUIRED_LOCATIONS: tuple[str, ...] = ("door", "living_room")
# A todo item whose text contains this introduces the guests.
INTRODUCTION_KEYWORD = "introduce"
# The todo statuses of an item being worked on, and of a finished one.
ACTIVE_STATUSES: tuple[str, ...] = tuple(s for s in OPEN_STATUSES if s != "pending")
FINISHED_STATUSES: tuple[str, ...] = tuple(
    s for s in get_args(TodoStatus) if s not in OPEN_STATUSES
)
# The goal states in which Nav2 drives, or may still drive, the base, as
# the gaze loop reads them.
NAV_ACTIVE_STATUSES = frozenset(
    (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING)
)

# What a guest says.
NAME_ANSWER = "My name is {name}."
DRINK_ANSWER = "I would like {drink}."
BAG_ANSWER = "It is on your hand."
OTHER_ANSWER = "Okay."
# A mumble with no answer in it. Not empty: the agent puts the text into an
# LLM message, and a blank message is rejected.
MUMBLE = "Hmm."

# What a robot line asks.
NAME = "name"
DRINK = "drink"
BAG = "bag"
OTHER = "other"

# A person's states. ABSENT is a guest who has not arrived yet, or an
# onlooker while no guest is at the door.
ABSENT = "absent"
AT_DOOR = "at_door"
TALKING = "talking"
FOLLOWING = "following"
SEATED = "seated"
STANDING = "standing"
BYSTANDER_ID = "bystander"
# The onlooker record's state when an onlooker leaves.
GONE = "gone"

# Why an offer seated nobody.
TAKEN = "taken"
NO_PLACE = "no place within the window"

# Why the seat switch was skipped.
ITEM_FINISHED = "the introduction item finished first"
ITEM_GONE = "the introduction item left the todo list"
NO_FREE_PLACE = "no free place for move"

# A Step's bag actions.
HANG = "hang"
RESTORE = "restore"

# As the speak tool publishes /to_human and RAI subscribes /from_human.
TO_HUMAN_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
FROM_HUMAN_QOS = TO_HUMAN_QOS
# As the agent publishes the list: latched, so a scenario started after the
# agent still gets the current list.
TODOS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
# BEST_EFFORT, which matches every publisher, reliable or not. The depth
# keeps a burst while the executor is busy, since the log must hold every
# message.
CMD_VEL_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

_ID_PATTERN = r"^[A-Za-z0-9_-]+$"
_MUMBLE_KEY = re.compile(r"^(guest\d+)\.(name|drink)$")

_Positive = Annotated[float, Field(gt=0)]


# -- The scenario file ------------------------------------------------------------------


class ScenarioError(ValueError):
    """The scenario file, or the locations file it names, is not usable."""


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GuestSpotSpec(_Spec):
    """Where each guest waits, in the map frame."""

    x: float
    y: float


class PacingSpec(_Spec):
    """The talking guest's pacing: a sinusoid through the spot, along `axis`."""

    # Meters to each side of the spot; 0 stands still.
    amplitude: float = Field(default=0.2, ge=0)
    # Seconds for one swing to each side and back.
    period: float = Field(default=6.0, gt=0)
    # The direction of the pacing in the map frame, as (x, y); its length
    # does not matter. None paces across the line from the spot to the door,
    # so the guest's bearing from the door pose changes while the distance
    # hardly does. A narrow hallway that is not square to that line needs
    # its own direction, so that the guest stays in its free span.
    axis: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check_axis(self) -> "PacingSpec":
        if self.axis is not None and math.hypot(*self.axis) == 0:
            raise ValueError("axis must not be (0, 0).")
        return self


class GuestSpec(_Spec):
    """A guest, guest1, guest2, ... in file order."""

    name: str = Field(min_length=1)
    drink: str = Field(min_length=1)
    # The keys of the face and appearance vectors (ScenePerson); None shows none.
    face: str | None = None
    appearance: str | None = None
    height: float = Field(default=STANDARD_HEIGHT_M, gt=0)


class SeatBoxSpec(_Spec):
    """A seat's box in the map frame: centre, size and yaw, as find_seats sees it."""

    center: tuple[float, float, float]
    size: tuple[_Positive, _Positive, _Positive]
    yaw: float = 0.0

    def map_box(self) -> MapBox:
        return MapBox(center=self.center, size=self.size, yaw=self.yaw)


class SeatSpec(_Spec):
    """A seat: its box, the way its sitters face, and its places."""

    id: str = Field(pattern=_ID_PATTERN)
    # The detector class, one of find_seats's seat classes.
    label: str
    box: SeatBoxSpec
    # The map-frame yaw that a sitter faces, in radians.
    facing: float
    # A sofa's places, left, middle and right as find_seats names them from
    # the robot's view, at (x, y). Any other seat is one place, at its box's
    # centre, and lists none.
    places: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # The top of the seat under a sitter, in meters; None for the scene's default.
    seat_top: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_places(self) -> "SeatSpec":
        seating = DEFAULT_SEATING_CONFIG
        # Exact, as the oracle compares the labels with its classes.
        if self.label not in seating.seat_classes:
            raise ValueError(
                f"label {self.label!r} is not one of find_seats's seat classes "
                f"({', '.join(seating.seat_classes)})."
            )
        if seating.is_sofa(self.label):
            if sorted(self.places) != sorted(SOFA_PLACES):
                raise ValueError(
                    f"a {self.label} needs exactly the places {', '.join(SOFA_PLACES)}, the "
                    f"parts find_seats splits it into; got {', '.join(self.places) or 'none'}."
                )
        elif self.places:
            raise ValueError(
                f"only a sofa class ({', '.join(seating.sofa_classes)}) has places; "
                f"{self.label!r} is one place, at its box's centre, so remove its places."
            )
        return self


class SeatedSpec(_Spec):
    """A person who sits from the start, such as the host."""

    id: str = Field(pattern=_ID_PATTERN)
    place: str
    face: str | None = None
    appearance: str | None = None
    height: float = Field(default=STANDARD_HEIGHT_M, gt=0)


class SeatSwitchSpec(_Spec):
    """The change of places before the introductions (hri.md, "Seats")."""

    when: Literal["introduction"] = "introduction"
    mode: Literal["swap", "move", "off"] = "off"
    people: tuple[str, ...] = ()
    # move only: the place to move the one listed person to.
    to: str | None = None


class BystanderSpec(_Spec):
    """A person who walks a path once, behind the guest at the door, facing along it."""

    appearance: str | None = None
    path: tuple[tuple[float, float], ...] = Field(min_length=2)
    # Meters per second.
    speed: float = Field(default=0.6, gt=0)
    height: float = Field(default=STANDARD_HEIGHT_M, gt=0)
    # Seconds after the run's first question.
    delay: float = Field(default=3.0, ge=0)

    @model_validator(mode="after")
    def _check_length(self) -> "BystanderSpec":
        if not self.length() > 0:
            raise ValueError("path needs two different points to walk between.")
        return self

    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.path, self.path[1:]))


class OnlookerSpec(_Spec):
    """A person at the door who is not the guest, onlooker1, onlooker2, ... in file order.

    An onlooker stands at (x, y) while a guest is at the door, facing the robot.
    """

    x: float
    y: float
    # Required, since find_people(in_front=True) judges people by their face.
    face: str = Field(min_length=1)
    appearance: str | None = None
    height: float = Field(default=STANDARD_HEIGHT_M, gt=0)


class AnswersSpec(_Spec):
    """How the guests answer."""

    # Seconds from the robot's line to the answer.
    delay: float = Field(default=1.0, ge=0)
    # <guest>.name or <guest>.drink: the question whose first ask gets the mumble.
    mumble_once: str | None = "guest1.drink"


class BagSpec(_Spec):
    """The bag hand-over."""

    # The first request is answered without hanging the bag.
    early_answer: bool = False
    # The guest who brings the bag; None for guest2, or nobody with one guest.
    owner: str | None = None


class OracleSpec(_Spec):
    """Overrides of OracleConfig's noise; a field left out keeps the oracle's default."""

    pixel_noise_px: float | None = None
    vector_noise: float | None = None
    outlier_rate: float | None = None
    dropout_rate: float | None = None

    @model_validator(mode="after")
    def _check_config(self) -> "OracleSpec":
        try:
            OracleConfig(**self.overrides())
        except ValueError as e:
            raise ValueError(str(e).replace("OracleConfig.", "")) from None
        return self

    def overrides(self) -> dict[str, float]:
        return {name: value for name, value in self.model_dump().items() if value is not None}


@dataclass(frozen=True)
class Place:
    """A place a person can sit at: a sofa's part, or a whole seat."""

    # "<seat>.<part>" on a sofa, else the seat's id.
    id: str
    seat: str
    x: float
    y: float
    # The map-frame yaw that its sitter faces.
    facing: float
    seat_top: float


class ScenarioSpec(_Spec):
    """A scenario file, validated: config/scenarios/hri.yaml is the default."""

    # The seed of the oracle's noise.
    seed: int = Field(default=0, ge=0)
    # The world the geometry belongs to; written to the log.
    world: str = Field(min_length=1)
    # The locations file with door and living_room, relative to the working directory.
    locations: str = Field(min_length=1)
    guest_spot: GuestSpotSpec
    pacing: PacingSpec = PacingSpec()
    guests: tuple[GuestSpec, ...] = Field(min_length=1)
    seats: tuple[SeatSpec, ...] = Field(min_length=1)
    seated: tuple[SeatedSpec, ...] = ()
    seat_switch: SeatSwitchSpec = SeatSwitchSpec()
    # A seat offer seats the guest only at a place this close to the robot's heading.
    offer_window_deg: float = Field(default=25.0, gt=0, le=180)
    bystander: BystanderSpec | None = None
    onlookers: tuple[OnlookerSpec, ...] = ()
    answers: AnswersSpec = AnswersSpec()
    bag: BagSpec = BagSpec()
    oracle: OracleSpec = OracleSpec()

    @property
    def guest_ids(self) -> tuple[str, ...]:
        return tuple(f"guest{i}" for i in range(1, len(self.guests) + 1))

    @property
    def onlooker_ids(self) -> tuple[str, ...]:
        return tuple(f"onlooker{i}" for i in range(1, len(self.onlookers) + 1))

    def bag_owner(self) -> str | None:
        """The guest who brings the bag: bag.owner, else guest2 when there is one."""
        if self.bag.owner is not None:
            return self.bag.owner
        return "guest2" if len(self.guests) >= 2 else None

    def places(self) -> dict[str, Place]:
        """Every place, by id, in file order."""
        places: dict[str, Place] = {}
        for seat in self.seats:
            top = DEFAULT_SEAT_TOP_M if seat.seat_top is None else seat.seat_top
            if seat.places:
                for part, (x, y) in seat.places.items():
                    place_id = f"{seat.id}.{part}"
                    places[place_id] = Place(place_id, seat.id, x, y, seat.facing, top)
            else:
                x, y, _ = seat.box.center
                places[seat.id] = Place(seat.id, seat.id, x, y, seat.facing, top)
        return places

    @model_validator(mode="after")
    def _check_references(self) -> "ScenarioSpec":
        # Every problem is collected, so one round of fixes is enough.
        errors: list[str] = []
        guests = self.guest_ids
        places = self.places()
        known_places = ", ".join(places)

        seat_ids = [seat.id for seat in self.seats]
        for seat_id in sorted({s for s in seat_ids if seat_ids.count(s) > 1}):
            errors.append(f"seats: the id {seat_id!r} is used twice; each seat needs its own.")

        onlookers = self.onlooker_ids
        seated_ids = [person.id for person in self.seated]
        sitters: dict[str, str] = {}
        for i, person in enumerate(self.seated):
            if seated_ids.count(person.id) > 1 and seated_ids.index(person.id) == i:
                errors.append(f"seated: the id {person.id!r} is used twice.")
            if person.id in guests or person.id in onlookers or person.id == BYSTANDER_ID:
                listed = f" and onlookers {', '.join(onlookers)}" if onlookers else ""
                errors.append(
                    f"seated.{i}.id: {person.id!r} is taken: guests are {', '.join(guests)}"
                    f"{listed} in file order, and {BYSTANDER_ID!r} is the bystander."
                )
            if person.place not in places:
                errors.append(
                    f"seated.{i}.place: {person.place!r} is not a place; the places are "
                    f"{known_places}."
                )
            elif person.place in sitters:
                errors.append(
                    f"seated.{i}.place: {sitters[person.place]} already sits at {person.place}; "
                    "a place holds one person."
                )
            else:
                sitters[person.place] = person.id

        # An onlooker's vectors must be its own, or recognize_person could
        # name it as someone else. Faces and appearances hash apart (scene.py),
        # so each kind is checked against its own kind only.
        owners: dict[str, dict[str, str]] = {"face": {}, "appearance": {}}
        others: list[tuple[str, str | None, str | None]] = [
            *((gid, g.face, g.appearance) for gid, g in zip(guests, self.guests)),
            *((p.id, p.face, p.appearance) for p in self.seated),
        ]
        if self.bystander is not None:
            others.append((BYSTANDER_ID, None, self.bystander.appearance))
        for person_id, face, appearance in others:
            for kind, key in (("face", face), ("appearance", appearance)):
                if key is not None:
                    owners[kind].setdefault(key, person_id)
        for i, (onlooker_id, onlooker) in enumerate(zip(onlookers, self.onlookers)):
            for kind, key in (("face", onlooker.face), ("appearance", onlooker.appearance)):
                if key is None:
                    continue
                owner = owners[kind].setdefault(key, onlooker_id)
                if owner != onlooker_id:
                    errors.append(
                        f"onlookers.{i}.{kind}: {key!r} is {owner}'s {kind} key too; an "
                        f"onlooker's keys must be its own, or recognize_person could name it "
                        f"{owner}."
                    )

        switch = self.seat_switch
        people = set(guests) | set(seated_ids)
        for person in switch.people:
            if person not in people:
                errors.append(
                    f"seat_switch.people: {person!r} is neither a guest ({', '.join(guests)}) "
                    f"nor a seated person ({', '.join(seated_ids) or 'none'})."
                )
        if len(set(switch.people)) != len(switch.people):
            errors.append("seat_switch.people: a person is listed twice.")
        if switch.mode == "swap" and len(switch.people) != 2:
            errors.append(
                f"seat_switch.mode 'swap' needs exactly two people in seat_switch.people, "
                f"got {len(switch.people)}."
            )
        if switch.mode == "move" and not switch.people:
            errors.append(
                "seat_switch.mode 'move' needs at least one person in seat_switch.people."
            )
        if switch.to is not None:
            if switch.mode != "move":
                errors.append(f"seat_switch.to applies to mode 'move' only, not {switch.mode!r}.")
            elif len(switch.people) != 1:
                errors.append(
                    f"seat_switch.to moves one person, but seat_switch.people lists "
                    f"{len(switch.people)}."
                )
            if switch.to not in places:
                errors.append(
                    f"seat_switch.to: {switch.to!r} is not a place; the places are {known_places}."
                )

        needed = len(self.guests) + len(self.seated)
        if switch.mode == "move" and len(places) < needed + 1:
            errors.append(
                f"seat_switch.mode 'move' needs {needed + 1} places, one for each of the "
                f"{len(self.guests)} guests and {len(self.seated)} seated people and one free "
                f"place to move to, but the seats have {len(places)}."
            )
        elif len(places) < needed:
            errors.append(
                f"seat_switch.mode {switch.mode!r} needs {needed} places, one for each of the "
                f"{len(self.guests)} guests and {len(self.seated)} seated people, but the "
                f"seats have {len(places)}."
            )

        mumble = self.answers.mumble_once
        if mumble is not None:
            match = _MUMBLE_KEY.match(mumble)
            if match is None or match.group(1) not in guests:
                errors.append(
                    f"answers.mumble_once must be <guest>.name or <guest>.drink for a guest of "
                    f"this file ({', '.join(guests)}), got {mumble!r}."
                )
        if self.bag.owner is not None and self.bag.owner not in guests:
            errors.append(
                f"bag.owner must be a guest of this file ({', '.join(guests)}), "
                f"got {self.bag.owner!r}."
            )
        if errors:
            raise ValueError("\n".join(errors))
        return self


@dataclass(frozen=True)
class LoadedScenario:
    """A scenario file, validated, with the locations it names."""

    path: Path
    spec: ScenarioSpec
    locations: dict[str, Location]
    locations_path: Path


def load_scenario(path: str | Path) -> LoadedScenario:
    """Read and validate a scenario file. Raises ScenarioError naming every problem found."""
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError as e:
        raise ScenarioError(f"Cannot read the scenario {path}: {e.strerror or e}.") from e
    except yaml.YAMLError as e:
        raise ScenarioError(f"The scenario {path} is not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ScenarioError(
            f"The scenario {path} must be a mapping of options, not {type(data).__name__}."
        )
    try:
        spec = ScenarioSpec.model_validate(data)
    except ValidationError as e:
        lines = "\n".join(f"  - {line}" for line in _validation_lines(e))
        raise ScenarioError(f"The scenario {path} is not valid:\n{lines}") from None

    locations_path = Path(spec.locations)
    try:
        locations = load_locations(locations_path)
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError, ValueError) as e:
        raise ScenarioError(
            f"The scenario {path}: locations: cannot read {locations_path}: {e}"
        ) from e
    missing = [name for name in REQUIRED_LOCATIONS if name not in locations]
    if missing:
        raise ScenarioError(
            f"The scenario {path}: locations: {locations_path} has no "
            f"{' and no '.join(missing)}; the scenario reads door and living_room from it."
        )
    return LoadedScenario(path, spec, locations, locations_path)


def _validation_lines(error: ValidationError) -> list[str]:
    """One line per problem, each prefixed with the option it is about."""
    items = error.errors()
    lines: list[str] = []
    for item in items:
        loc = item["loc"]
        # pydantic drops the items that failed and then finds the list too
        # short, which would read as if the file listed no seats at all. The
        # items' own errors say what is wrong.
        if item["type"] == "too_short" and any(
            len(other["loc"]) > len(loc) and other["loc"][: len(loc)] == loc for other in items
        ):
            continue
        where = ".".join(str(part) for part in loc)
        message = item["msg"].removeprefix("Value error, ")
        for text in message.splitlines():
            lines.append(f"{where}: {text}" if where else text)
    return lines


# -- The scenario's logic -----------------------------------------------------------------


class Pose(NamedTuple):
    """The robot's base in the map frame: meters, and the yaw in radians."""

    x: float
    y: float
    yaw: float


class Line(NamedTuple):
    """What a robot line asks, by classify_line()."""

    # A sentence of it is a question.
    question: bool
    # NAME, DRINK, BAG and OTHER, in the order asked, each once.
    asks: tuple[str, ...]
    # It mentions sitting or a seat.
    seat: bool


class Answer(NamedTuple):
    """A guest's answer to send on /from_human."""

    text: str
    # The guest who answers; None when no guest is current.
    guest: str | None
    # The robot line it answers, and that line's time.
    to: str
    asked_t: float
    asks: tuple[str, ...]
    mumbled: bool


class Step(NamedTuple):
    """What one call of HriScenario returns."""

    # The answers to send now.
    answers: tuple[Answer, ...] = ()
    # The records to log, in order, each with "type" and "t".
    events: tuple[dict[str, Any], ...] = ()
    # HANG to set grasp_detect_min to -1, RESTORE to set it back, or None.
    bag: str | None = None


class PersonState(NamedTuple):
    """One person as the scenario has them now."""

    id: str
    # The guest's name; None for a seated person.
    name: str | None
    state: str
    # 0 while not in view.
    track_id: int
    place: str | None


@dataclass
class _Person:
    id: str
    name: str | None
    drink: str | None
    face: str | None
    appearance: str | None
    height: float
    guest: bool
    state: str = ABSENT
    track_id: int = 0
    place: str | None = None
    # When the guest started talking, for the pacing's phase.
    talk_start: float | None = None
    # Where an onlooker stands; None for everyone else.
    spot: tuple[float, float] | None = None


class _Entry(NamedTuple):
    """A person's state from `t` on, as people_at() draws it."""

    t: float
    state: str
    track_id: int
    place: str | None
    talk_start: float | None


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_QUESTION_PHRASES = ("tell me", "let me know")
_NAME_WORDS = re.compile(r"\b(name|call you)\b")
_DRINK_WORDS = re.compile(r"\b(drinks?|beverages?)\b")
_BAG_WORDS = re.compile(r"\bbags?\b")
_PLEASE = re.compile(r"\bplease\b")
# A sentence that opens with a request verb, after any opening words, each
# with an optional comma: "Now, hand me your bag."
_REQUEST_OPENING = re.compile(
    r"(?:(?:okay|ok|alright|now|so|then|just|go\s+ahead\s+and)\b,?\s*)*"
    r"(?:put|place|hand|give|pass|set|hang|rest|leave|drop)\b"
)
_SEAT_WORDS = re.compile(r"\b(sit|sits|sitting|seat|seats|seated)\b")


def classify_line(text: str) -> Line:
    """What a robot line asks, sentence by sentence.

    The rules are plain on purpose, since the lines come from an LLM:
      - a sentence is a question when it ends with "?" or holds "tell me"
        or "let me know";
      - a sentence that mentions a bag asks for the bag when it is a
        question, holds "please", or opens with put, place, hand, give,
        pass, set, hang, rest, leave or drop, after any of okay, ok,
        alright, now, so, then, just and "go ahead and". So "Put your bag
        on my hand." asks for the bag, and "I will hold out my hand for
        your bag." asks nothing;
      - a question that mentions a name asks for the name, one that
        mentions a drink or a beverage asks for the drink, and a question
        that asks for none of the three asks OTHER;
      - the line mentions a seat when a sentence holds sit, sits, sitting,
        seat, seats or seated.
    """
    question = False
    seat = False
    asks: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        lower = sentence.lower()
        is_question = lower.rstrip().endswith("?") or any(p in lower for p in _QUESTION_PHRASES)
        question = question or is_question
        seat = seat or _SEAT_WORDS.search(lower) is not None
        topics: list[str] = []
        if _BAG_WORDS.search(lower) and (
            is_question or _PLEASE.search(lower) or _REQUEST_OPENING.match(lower.lstrip())
        ):
            topics.append(BAG)
        if is_question:
            if _NAME_WORDS.search(lower):
                topics.append(NAME)
            if _DRINK_WORDS.search(lower):
                topics.append(DRINK)
            if not topics:
                topics.append(OTHER)
        asks += [topic for topic in topics if topic not in asks]
    return Line(question=question, asks=tuple(asks), seat=seat)


def bearing_deg(pose: Pose, x: float, y: float) -> float:
    """The bearing of the point (x, y) from the base at `pose`, in degrees, positive to the left."""
    return _wrap_deg(math.degrees(math.atan2(y - pose.y, x - pose.x) - pose.yaw))


def _wrap_deg(deg: float) -> float:
    """`deg` in (-180, 180]."""
    rad = math.radians(deg)
    wrapped = math.degrees(math.atan2(math.sin(rad), math.cos(rad)))
    return 180.0 if wrapped <= -180.0 else wrapped


def _pose_json(pose: Pose | None) -> dict[str, float] | None:
    if pose is None:
        return None
    return {"x": round(pose.x, 4), "y": round(pose.y, 4), "yaw": round(pose.yaw, 4)}


def _location_pose(location: Location) -> Pose:
    return Pose(location.x, location.y, location.yaw)


class HriScenario:
    """The scenario's logic: who is where, what they answer, and what to log.

    robot_said(), todos_changed(), robot_pose(), reset(), tick() and
    clock_jumped() each return a Step: the answers due now, the events to
    log, and a bag action. scene() gives the Scene for the oracle, or None
    for a time it does not know. Times
    are seconds on `clock`, which each method reads unless given `t`. The
    events that happen at construction, the seated people, and guest 1 and
    the onlookers at the door, come with the first Step. Not thread-safe: the node calls it
    under a lock.
    """

    def __init__(
        self,
        spec: ScenarioSpec,
        locations: Mapping[str, Location],
        clock: Callable[[], float],
    ) -> None:
        missing = [name for name in REQUIRED_LOCATIONS if name not in locations]
        if missing:
            raise ScenarioError(
                f"The locations have no {' and no '.join(missing)}; the scenario needs door "
                "and living_room."
            )
        self._spec = spec
        self._clock = clock
        self._door = _location_pose(locations["door"])
        self._living_room = _location_pose(locations["living_room"])
        self._places = spec.places()
        self._spot = (spec.guest_spot.x, spec.guest_spot.y)
        # The pacing's direction (PacingSpec.axis): by default across the
        # line from the spot to the door, so the guest's bearing from the
        # door pose changes while the distance hardly does.
        if spec.pacing.axis is not None:
            ax, ay = spec.pacing.axis
            length = math.hypot(ax, ay)
            self._pacing_axis = (ax / length, ay / length)
        else:
            dx, dy = self._door.x - self._spot[0], self._door.y - self._spot[1]
            length = math.hypot(dx, dy)
            self._pacing_axis = (-dy / length, dx / length) if length > 0 else (0.0, 1.0)
        self._bag_owner = spec.bag_owner()

        # Track ids: the seats first, then each appearance of a person.
        self._seat_tracks = {seat.id: i for i, seat in enumerate(spec.seats, start=1)}
        self._last_track = len(spec.seats)
        self._objects = tuple(
            SceneObject(self._seat_tracks[seat.id], seat.label, seat.box.map_box())
            for seat in spec.seats
        )

        self._events: list[dict[str, Any]] = []
        self._pending: list[tuple[float, int, Answer]] = []
        self._answer_count = 0
        self._pose: Pose | None = None
        self._first_question_t: float | None = None
        self._mumbled = False
        self._bag_requests = 0
        self._occupants: dict[str, str] = {}
        self._switch_item: str | None = None
        self._switch_done = False
        self._bystander_state = "waiting"
        # Each person's states of the last HISTORY_S, oldest first, so that
        # a camera image's stamp draws the people as they were then.
        self._timeline: dict[str, list[_Entry]] = {}

        t = clock()
        self._people: dict[str, _Person] = {}
        for seated in spec.seated:
            person = _Person(
                id=seated.id,
                name=None,
                drink=None,
                face=seated.face,
                appearance=seated.appearance,
                height=seated.height,
                guest=False,
            )
            self._people[seated.id] = person
            self._seat(person, seated.place, t)
        self._bystander_track = self._new_track() if spec.bystander is not None else 0
        for guest_id, guest in zip(spec.guest_ids, spec.guests):
            self._people[guest_id] = _Person(
                id=guest_id,
                name=guest.name,
                drink=guest.drink,
                face=guest.face,
                appearance=guest.appearance,
                height=guest.height,
                guest=True,
            )
            self._remember(self._people[guest_id], t)
        for onlooker_id, onlooker in zip(spec.onlooker_ids, spec.onlookers):
            self._people[onlooker_id] = _Person(
                id=onlooker_id,
                name=None,
                drink=None,
                face=onlooker.face,
                appearance=onlooker.appearance,
                height=onlooker.height,
                guest=False,
                spot=(onlooker.x, onlooker.y),
            )
            self._remember(self._people[onlooker_id], t)
        self._arrive_next(t)

    # -- Read-only views --

    @property
    def door(self) -> Pose:
        return self._door

    @property
    def living_room(self) -> Pose:
        return self._living_room

    @property
    def places(self) -> Mapping[str, Place]:
        return dict(self._places)

    @property
    def pose(self) -> Pose | None:
        """The robot's last pose, or None before the first."""
        return self._pose

    def person(self, person_id: str) -> PersonState:
        p = self._people[person_id]
        return PersonState(p.id, p.name, p.state, p.track_id, p.place)

    # -- Inputs --

    def robot_said(self, text: str, t: float | None = None) -> Step:
        """A line the robot said: log it, queue the answer, and act on an offer or a bag request."""
        t = self._now(t)
        line = classify_line(text)
        guest = self._current_guest()
        guest_id = guest.id if guest is not None else None
        offer = line.seat and guest is not None and guest.state == FOLLOWING
        self._emit(
            "robot_line",
            t,
            text=text,
            guest=guest_id,
            question=line.question,
            asks=list(line.asks),
            offer=offer,
            pose=_pose_json(self._pose),
            bearings_deg=self._bearings(t),
        )
        if line.question:
            if self._first_question_t is None:
                self._first_question_t = t
            if guest is not None and guest.state == AT_DOOR:
                guest.state = TALKING
                guest.talk_start = t
                self._emit_person(guest, t)

        parts: list[str] = []
        answered: list[str] = []
        mumbled = False
        extra = False
        bag = None
        for topic in line.asks:
            if topic in (NAME, DRINK) and guest is not None:
                if f"{guest.id}.{topic}" == self._spec.answers.mumble_once and not self._mumbled:
                    self._mumbled = mumbled = True
                    parts.append(MUMBLE)
                elif topic == NAME:
                    parts.append(NAME_ANSWER.format(name=guest.name))
                else:
                    parts.append(DRINK_ANSWER.format(drink=guest.drink))
                answered.append(topic)
            elif topic == BAG and guest is not None and guest.id == self._bag_owner:
                self._bag_requests += 1
                early = self._spec.bag.early_answer
                hang = not (early and self._bag_requests == 1)
                self._emit(
                    "bag_request",
                    t,
                    guest=guest.id,
                    count=self._bag_requests,
                    hang=hang,
                    early_answer=early,
                )
                if hang:
                    bag = HANG
                parts.append(BAG_ANSWER)
                answered.append(BAG)
            else:
                extra = True
        if extra:
            self._emit("extra_question", t, guest=guest_id, text=text)
            answered.append(OTHER)
            if not parts:
                parts.append(OTHER_ANSWER)
        if parts:
            answer = Answer(" ".join(parts), guest_id, text, t, tuple(answered), mumbled)
            self._pending.append((t + self._spec.answers.delay, self._answer_count, answer))
            self._answer_count += 1
        if offer:
            # A guest follows only once a pose has come, so the pose is known here.
            assert guest is not None and self._pose is not None
            self._offer(guest, self._pose, t)
        return self._step(bag=bag)

    def robot_pose(self, pose: Pose, t: float | None = None) -> Step:
        """The robot's pose from TF. The guest who talked follows once it leaves the door."""
        t = self._now(t)
        self._pose = pose
        guest = self._current_guest()
        away = math.hypot(pose.x - self._door.x, pose.y - self._door.y)
        if guest is not None and guest.state == TALKING and away > GUIDE_DISTANCE_M:
            guest.state = FOLLOWING
            guest.track_id = 0
            guest.talk_start = None
            self._emit_person(guest, t)
            for onlooker in self._onlookers():
                if onlooker.state == STANDING:
                    ended, onlooker.state, onlooker.track_id = onlooker.track_id, ABSENT, 0
                    self._emit_onlooker(onlooker, GONE, ended, t)
        return self._step()

    def todos_changed(self, items: Sequence[Any], t: float | None = None) -> Step:
        """The agent's todo list, as published on /walkie/todos: arms or skips the seat switch."""
        t = self._now(t)
        self._emit("todos", t, items=list(items))
        if self._spec.seat_switch.mode == "off" or self._switch_done:
            return self._step()
        entries = [item for item in items if isinstance(item, Mapping)]
        if self._switch_item is None:
            matches = [
                e for e in entries if INTRODUCTION_KEYWORD in str(e.get("content", "")).lower()
            ]
            active = next((e for e in matches if e.get("status") in ACTIVE_STATUSES), None)
            if active is not None:
                self._switch_item = str(active.get("id", ""))
                self._try_switch(t)
            elif matches and all(e.get("status") in FINISHED_STATUSES for e in matches):
                # Skipped only once no introduction is left to come.
                self._skip(ITEM_FINISHED, t, str(matches[0].get("id", "")))
        else:
            item = next((e for e in entries if str(e.get("id", "")) == self._switch_item), None)
            if item is None:
                self._skip(ITEM_GONE, t)
            elif item.get("status") in FINISHED_STATUSES:
                self._skip(ITEM_FINISHED, t)
        return self._step()

    def reset(self, t: float | None = None) -> Step:
        """A reset of the agent: the answers not sent yet are dropped, and the bag restored."""
        t = self._now(t)
        dropped = len(self._pending)
        self._pending.clear()
        self._emit("reset", t, dropped=dropped)
        return self._step(bag=RESTORE)

    def tick(self, t: float | None = None) -> Step:
        """The answers due by `t`, and the bystander's start and end."""
        t = self._now(t)
        due = sorted((e for e in self._pending if e[0] <= t), key=lambda e: (e[0], e[1]))
        self._pending = [e for e in self._pending if e[0] > t]
        for _, _, answer in due:
            self._emit(
                "answer",
                t,
                guest=answer.guest,
                text=answer.text,
                to=answer.to,
                asked_t=round(answer.asked_t, 3),
                asks=list(answer.asks),
                mumbled=answer.mumbled,
            )
        window = self._bystander_window()
        if window is not None:
            start, end = window
            path = self._spec.bystander.path  # type: ignore[union-attr]
            if self._bystander_state == "waiting" and t >= start:
                self._bystander_state = "walking"
                self._emit_bystander("walking", path[0], t)
            if self._bystander_state == "walking" and t > end:
                self._bystander_state = "gone"
                self._emit_bystander("gone", path[-1], t)
        return self._step(answers=tuple(answer for _, _, answer in due))

    def clock_jumped(self, old_t: float, new_t: float) -> Step:
        """The clock went from `old_t` to `new_t`: every time the scenario keeps moves with it.

        A pending answer is then due as long after `new_t` as it was after
        `old_t`, and the pacing, the bystander and the scene's history keep
        their place. Answer.asked_t stays the time logged with its line. A
        run with a jump is not valid (module docstring).
        """
        shift = new_t - old_t
        self._pending = [(due + shift, n, answer) for due, n, answer in self._pending]
        if self._first_question_t is not None:
            self._first_question_t += shift
        for person in self._people.values():
            if person.talk_start is not None:
                person.talk_start += shift
        for entries in self._timeline.values():
            entries[:] = [
                entry._replace(
                    t=entry.t + shift,
                    talk_start=None if entry.talk_start is None else entry.talk_start + shift,
                )
                for entry in entries
            ]
        self._emit(
            "clock_jump",
            new_t,
            **{"from": round(old_t, 3), "to": round(new_t, 3), "pending": len(self._pending)},
        )
        return self._step()

    # -- Outputs --

    def scene(self, t: float | None = None) -> Scene | None:
        """Everyone and every seat as of `t`, for the oracle, or None (people_at())."""
        people = self.people_at(t)
        if people is None:
            return None
        return Scene(people=tuple(people.values()), objects=self._objects)

    def people_at(self, t: float | None = None) -> dict[str, ScenePerson] | None:
        """Everyone the scene holds as of `t`, by id; None when `t` is unknown.

        `t` selects each person's state as of `t`, so a camera image's stamp
        draws the people as they were when it was taken. A person's states
        older than HISTORY_S before their latest change are dropped, so the
        scene is known for at least the last HISTORY_S. A `t` older than the
        states kept, or before the scenario's start, is unknown, so that an
        old image is never drawn with a later scene. A following guest is in
        no view. The guests at the door and the onlookers face the robot's
        last pose.
        """
        t = self._now(t)
        drawn: dict[str, ScenePerson] = {}
        for person in self._people.values():
            entry = self._entry_at(person.id, t)
            if entry is None:
                return None
            if entry.state in (AT_DOOR, TALKING, STANDING):
                if person.spot is not None:
                    x, y = person.spot
                else:
                    x, y = self._guest_xy(entry.state, entry.talk_start, t)
                # Turned to the robot, or to where it will stand before a pose comes.
                target = self._pose if self._pose is not None else self._door
                drawn[person.id] = ScenePerson(
                    track_id=entry.track_id,
                    x=x,
                    y=y,
                    facing=math.atan2(target.y - y, target.x - x),
                    height=person.height,
                    face=person.face,
                    appearance=person.appearance,
                )
            elif entry.state == SEATED and entry.place is not None:
                place = self._places[entry.place]
                drawn[person.id] = ScenePerson(
                    track_id=entry.track_id,
                    x=place.x,
                    y=place.y,
                    facing=place.facing,
                    height=person.height,
                    posture="sitting",
                    face=person.face,
                    appearance=person.appearance,
                    seat_top=place.seat_top,
                )
        walker = self._bystander_at(t)
        if walker is not None:
            drawn[BYSTANDER_ID] = walker
        return drawn

    def talking(self, t: float | None = None) -> dict[str, Any] | None:
        """The guest talking at the door, where they are and their bearing; None if nobody talks."""
        t = self._now(t)
        guest = self._current_guest()
        if guest is None or guest.state != TALKING:
            return None
        x, y = self._guest_xy(guest.state, guest.talk_start, t)
        bearing = None if self._pose is None else round(bearing_deg(self._pose, x, y), 2)
        return {"guest": guest.id, "x": round(x, 4), "y": round(y, 4), "bearing_deg": bearing}

    # -- Internals --

    def _now(self, t: float | None) -> float:
        return self._clock() if t is None else t

    def _emit(self, type_: str, t: float, **fields: Any) -> None:
        self._events.append({"type": type_, "t": t, **fields})

    def _step(self, answers: tuple[Answer, ...] = (), bag: str | None = None) -> Step:
        events, self._events = tuple(self._events), []
        return Step(answers=answers, events=events, bag=bag)

    def _new_track(self) -> int:
        self._last_track += 1
        return self._last_track

    def _current_guest(self) -> _Person | None:
        return next(
            (
                p
                for p in self._people.values()
                if p.guest and p.state in (AT_DOOR, TALKING, FOLLOWING)
            ),
            None,
        )

    def _onlookers(self) -> list[_Person]:
        return [p for p in self._people.values() if p.spot is not None]

    def _arrive_next(self, t: float) -> None:
        """Bring the next guest to the door, and the onlookers with them, each on a new track."""
        guest = next((p for p in self._people.values() if p.guest and p.state == ABSENT), None)
        if guest is not None:
            guest.state = AT_DOOR
            guest.track_id = self._new_track()
            self._emit_person(guest, t)
            for onlooker in self._onlookers():
                onlooker.state = STANDING
                onlooker.track_id = self._new_track()
                self._emit_onlooker(onlooker, STANDING, onlooker.track_id, t)

    def _seat(self, person: _Person, place: str, t: float) -> None:
        # A new track: the person was out of view, or changed places at once.
        person.state = SEATED
        person.place = place
        person.track_id = self._new_track()
        person.talk_start = None
        self._occupants[place] = person.id
        self._emit_person(person, t)

    def _emit_person(self, person: _Person, t: float) -> None:
        x: float | None = None
        y: float | None = None
        if person.state in (AT_DOOR, TALKING):
            x, y = self._spot
        elif person.state == SEATED and person.place is not None:
            x, y = self._places[person.place].x, self._places[person.place].y
        self._emit(
            "person",
            t,
            person=person.id,
            name=person.name,
            state=person.state,
            track_id=person.track_id or None,
            place=person.place if person.state == SEATED else None,
            x=x,
            y=y,
        )
        self._remember(person, t)

    def _emit_onlooker(self, onlooker: _Person, state: str, track_id: int, t: float) -> None:
        # GONE carries the track that ended, as the bystander's record does.
        assert onlooker.spot is not None
        x, y = onlooker.spot
        self._emit("onlooker", t, onlooker=onlooker.id, state=state, track_id=track_id, x=x, y=y)
        self._remember(onlooker, t)

    def _remember(self, person: _Person, t: float) -> None:
        """Add the person's state from `t` to their timeline, and drop what is too old."""
        entries = self._timeline.setdefault(person.id, [])
        entries.append(_Entry(t, person.state, person.track_id, person.place, person.talk_start))
        # The last entry at or before the horizon still holds at the horizon.
        horizon = t - HISTORY_S
        while len(entries) > 1 and entries[1].t <= horizon:
            del entries[0]

    def _entry_at(self, person_id: str, t: float) -> _Entry | None:
        """The person's last entry at or before `t`; None before their oldest one kept.

        A timeline is complete from its oldest entry on (_remember()).
        """
        entries = self._timeline[person_id]
        if t < entries[0].t:
            return None
        found = entries[0]
        for entry in entries:
            if entry.t > t:
                break
            found = entry
        return found

    def _offer(self, guest: _Person, pose: Pose, t: float) -> None:
        bearings = {pid: bearing_deg(pose, p.x, p.y) for pid, p in self._places.items()}
        # min() keeps the first of equals, so a tie goes to the place first in the file.
        nearest = min(bearings, key=lambda pid: abs(bearings[pid]))
        if abs(bearings[nearest]) > self._spec.offer_window_deg:
            self._emit_offer("bad_offer", t, guest, pose, bearings, None, NO_PLACE)
        elif nearest in self._occupants:
            taken_by = self._occupants[nearest]
            self._emit_offer("bad_offer", t, guest, pose, bearings, nearest, TAKEN, taken_by)
        else:
            self._emit_offer("seat_taken", t, guest, pose, bearings, nearest)
            self._seat(guest, nearest, t)
            self._arrive_next(t)
            self._try_switch(t)

    def _emit_offer(
        self,
        type_: str,
        t: float,
        guest: _Person,
        pose: Pose,
        bearings: dict[str, float],
        place: str | None,
        reason: str | None = None,
        taken_by: str | None = None,
    ) -> None:
        self._emit(
            type_,
            t,
            guest=guest.id,
            place=place,
            reason=reason,
            taken_by=taken_by,
            pose=_pose_json(pose),
            heading_deg=round(_wrap_deg(math.degrees(pose.yaw)), 2),
            bearings_deg={pid: round(b, 2) for pid, b in bearings.items()},
            window_deg=self._spec.offer_window_deg,
        )

    def _try_switch(self, t: float) -> None:
        """Fire the armed seat switch once everyone it lists is seated."""
        switch = self._spec.seat_switch
        if self._switch_item is None or self._switch_done:
            return
        people = [self._people[person_id] for person_id in switch.people]
        if any(person.state != SEATED for person in people):
            return
        old = {person.id: person.place for person in people}
        if switch.mode == "swap":
            first, second = people
            new = {first.id: second.place, second.id: first.place}
        else:
            targets, reason = self._move_targets(people)
            if targets is None:
                self._skip(reason or NO_FREE_PLACE, t)
                return
            new = targets
        self._switch_done = True
        self._emit("seat_switch", t, mode=switch.mode, item=self._switch_item, old=old, new=new)
        for person in people:
            if person.place is not None and self._occupants.get(person.place) == person.id:
                del self._occupants[person.place]
        for person in people:
            place = new[person.id]
            assert place is not None
            self._seat(person, place, t)

    def _move_targets(self, people: list[_Person]) -> tuple[dict[str, str] | None, str | None]:
        """Where `move` takes each person, one after the other, or None and why not."""
        to = self._spec.seat_switch.to
        occupants = dict(self._occupants)
        targets: dict[str, str] = {}
        for person in people:
            assert person.place is not None
            if to is not None:
                holder = occupants.get(to)
                if holder == person.id:
                    return None, f"{NO_FREE_PLACE}: {person.id} already sits at {to}"
                if holder is not None:
                    return None, f"{NO_FREE_PLACE}: {to} is taken by {holder}"
                target = to
            else:
                free = next((pid for pid in self._places if pid not in occupants), None)
                if free is None:
                    return None, NO_FREE_PLACE
                target = free
            del occupants[person.place]
            occupants[target] = person.id
            targets[person.id] = target
        return targets, None

    def _skip(self, reason: str, t: float, item: str | None = None) -> None:
        self._switch_done = True
        self._emit(
            "seat_switch_skipped",
            t,
            mode=self._spec.seat_switch.mode,
            item=item if item is not None else self._switch_item,
            reason=reason,
        )

    def _guest_xy(self, state: str, talk_start: float | None, t: float) -> tuple[float, float]:
        x, y = self._spot
        pacing = self._spec.pacing
        if state == TALKING and talk_start is not None and pacing.amplitude > 0:
            phase = 2 * math.pi * max(0.0, t - talk_start) / pacing.period
            offset = pacing.amplitude * math.sin(phase)
            x, y = x + offset * self._pacing_axis[0], y + offset * self._pacing_axis[1]
        return x, y

    def _bystander_window(self) -> tuple[float, float] | None:
        spec = self._spec.bystander
        if spec is None or self._first_question_t is None:
            return None
        start = self._first_question_t + spec.delay
        return start, start + spec.length() / spec.speed

    def _bystander_at(self, t: float) -> ScenePerson | None:
        spec = self._spec.bystander
        window = self._bystander_window()
        if spec is None or window is None or not window[0] <= t <= window[1]:
            return None
        walked = (t - window[0]) * spec.speed
        point, heading = spec.path[-1], 0.0
        for a, b in zip(spec.path, spec.path[1:]):
            segment = math.dist(a, b)
            if segment == 0:
                continue
            heading = math.atan2(b[1] - a[1], b[0] - a[0])
            if walked <= segment:
                share = walked / segment
                point = (a[0] + share * (b[0] - a[0]), a[1] + share * (b[1] - a[1]))
                break
            walked -= segment
        return ScenePerson(
            track_id=self._bystander_track,
            x=point[0],
            y=point[1],
            facing=heading,
            height=spec.height,
            appearance=spec.appearance,
        )

    def _emit_bystander(self, state: str, point: tuple[float, float], t: float) -> None:
        self._emit(
            "bystander", t, state=state, track_id=self._bystander_track, x=point[0], y=point[1]
        )

    def _bearings(self, t: float) -> dict[str, float] | None:
        people = self.people_at(t)
        if self._pose is None or people is None:
            return None
        pose = self._pose
        return {pid: round(bearing_deg(pose, p.x, p.y), 2) for pid, p in people.items()}


# -- The commander's parameter --------------------------------------------------------------


def _double(value: float) -> ParameterValue:
    return ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value))


def _parameter_json(value: ParameterValue | None) -> Any:
    if value is None or value.type == ParameterType.PARAMETER_NOT_SET:
        return None
    if value.type == ParameterType.PARAMETER_DOUBLE:
        return value.double_value
    if value.type == ParameterType.PARAMETER_INTEGER:
        return value.integer_value
    return str(value)


def _negative(value: ParameterValue) -> bool:
    if value.type == ParameterType.PARAMETER_DOUBLE:
        return value.double_value < 0
    if value.type == ParameterType.PARAMETER_INTEGER:
        return value.integer_value < 0
    return False


# What the commander holds because of this node, as GraspParameter knows
# it. ORIGINAL also before the first read, since nothing was changed yet.
# UNKNOWN after a negative value found at start, or a set whose answer was
# lost or failed.
_HELD_ORIGINAL = "original"
_HELD_HUNG = "hung"
_HELD_UNKNOWN = "unknown"


@dataclass(frozen=True)
class _Request:
    """The one request to the commander in flight."""

    seq: int
    # HANG or RESTORE for a set; None for the read.
    action: str | None
    # The value a set sends.
    value: ParameterValue | None
    # The cause a restore is logged with.
    cause: str | None
    # When it was sent, on the computer's monotonic clock.
    sent_at: float


class GraspParameter:
    """The commander's grasp_detect_min: read once, set to -1 to hang the bag, and set back.

    Through the commander's standard parameter services, as walkie-ros's
    bimanual_commander offers them. The value read at start is the one
    restored. A negative value at start is a hang left by a run that did not
    restore it, so the commander's default is set back at once.

    hang() and restore() set the value wanted, and the commander is driven
    to it by one request at a time. A request that cannot be sent, fails,
    or gets no answer within REQUEST_TIMEOUT_S is tried again within
    COMMANDER_RETRY_S, until the commander holds the value wanted; each such
    hang or restore is logged as bag_retry. A hang is given up, as
    bag_skipped, only when there is no commander at the request, it has no
    such parameter, it refuses the hang, or a restore comes before the
    commander took it. Each value the commander takes is logged once it has
    answered. A hang wanted while the commander still holds -1, as after a
    restore it refused, is logged as bag_hung with already_held at once.

    Thread-safe. Each decision is made and logged under its own lock, so
    the log's order is the order of the decisions. A send lock is held from
    deciding a request until it is sent. So no other thread drops it as
    unanswered before it went out, and close() waits for a send under way,
    so nothing goes out after it. The send lock is re-entrant, since rclpy
    may run an answer's callback at once on the sending thread, and that
    callback may send the next request. The decision lock is released
    before each send, so such a callback can take it. Neither lock is held
    while waiting for an answer, and neither takes the node's lock, so the
    node may call it under its own (node -> send -> grasp -> log).
    """

    def __init__(
        self,
        node: Node,
        commander: str,
        record: Callable[..., None],
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._commander = commander
        self._record = record
        self._monotonic = monotonic
        group = ReentrantCallbackGroup()
        self._get = node.create_client(
            GetParameters, f"{commander}/get_parameters", callback_group=group
        )
        self._set = node.create_client(
            SetParameters, f"{commander}/set_parameters", callback_group=group
        )
        # Held from deciding a request until it is sent (class docstring).
        self._send_lock = threading.RLock()
        self._lock = threading.Lock()
        self._original: ParameterValue | None = None
        self._missing = False
        self._want_hung = False
        self._held = _HELD_ORIGINAL
        self._inflight: _Request | None = None
        self._seq = 0
        # Set once the current hang is resolved: taken, given up, or
        # superseded by a restore before the commander took it.
        self._hang_event: threading.Event | None = None
        # Set once the commander holds the original again.
        self._restore_event: threading.Event | None = None
        # The cause the wanted restore is logged with.
        self._cause = "start"
        self._finishing = False
        # When the restore on exit could first go out, on the monotonic
        # clock; set by finish().
        self._exit_free_at: float | None = None
        self._closed = False
        # On the computer's clock, so the commander is found before the
        # simulation's /clock runs too. Never cancelled before close(): it
        # sends again what failed or got no answer.
        self._timer = node.create_timer(
            COMMANDER_RETRY_S,
            self._on_timer,
            callback_group=group,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    @property
    def original(self) -> ParameterValue | None:
        """The value to restore, once read."""
        with self._lock:
            return self._original

    def hang(self) -> threading.Event | None:
        """Want grasp_detect_min at -1. The event is set once this hang is resolved.

        Resolved: the commander took it (bag_hung), it held -1 still with
        nothing in flight, as after a restore it refused (bag_hung with
        already_held), it was given up (bag_skipped), or a restore came
        before the commander took it. A restore in flight at the request is
        answered first: once taken, the hang is sent again; once refused,
        the commander holds -1 still. A request while a hang is wanted gets
        that hang's event, which may be set already. None when the hang was
        logged as skipped at once.
        """
        with self._lock:
            reason = None
            if self._finishing:
                reason = "the scenario is closing"
            elif self._missing:
                reason = self._no_parameter()
            elif not self._set.service_is_ready():
                # No commander at the request: the run has none.
                reason = self._no_server()
            if reason is not None:
                self._record("bag_skipped", action=HANG, reason=reason)
                return None
            if self._want_hung:
                return self._hang_event
            self._want_hung = True
            # An event not set yet belongs to a hang still in flight from
            # before a restore. Its answer resolves this hang too.
            if self._hang_event is None or self._hang_event.is_set():
                self._hang_event = threading.Event()
            event = self._hang_event
            self._settle()
        self._reconcile()
        return event

    def restore(self, cause: str) -> threading.Event | None:
        """Want grasp_detect_min back at the value read at start, logged with `cause`.

        The event is set once the commander holds that value again; None
        when it holds it already and nothing is in flight. A wanted hang
        that is not in flight is logged as skipped. A hang in flight is
        logged as its answer says, and the restore follows it at once.
        """
        with self._lock:
            self._want_hung = False
            self._cause = cause
            hang = self._hang_event
            if hang is not None and not hang.is_set() and not self._hang_in_flight():
                self._record(
                    "bag_skipped",
                    action=HANG,
                    reason=f"{cause} before {self._commander} took the hang",
                )
                hang.set()
            if self._held == _HELD_ORIGINAL and self._inflight is None:
                return None
            if self._restore_event is None:
                self._restore_event = threading.Event()
            event = self._restore_event
        self._reconcile()
        return event

    def finish(self) -> threading.Event | None:
        """Refuse every later hang, then restore("exit"). For the node's close().

        wait_restored() then waits for the event, if any.
        """
        with self._lock:
            self._finishing = True
            # When the restore can first go out: now, or once the request in
            # flight is answered or, at the latest, dropped as unanswered.
            now = self._monotonic()
            due = self._due()
            self._exit_free_at = now if due is None else max(now, due)
        return self.restore("exit")

    def wait_restored(self, done: threading.Event, wait_s: float) -> bool:
        """After finish(), wait for its event `done`; False when it was not set in time.

        A request in flight at finish() holds the restore back until it is
        answered or its REQUEST_TIMEOUT_S ends. Then the wait drops it and
        sends the restore at once, rather than at the timer's next tick. The
        wait ends `wait_s` after the restore could first go out, so it lasts
        at most REQUEST_TIMEOUT_S + `wait_s`, and a restore sent gets
        `wait_s` for its answer. One that fails meanwhile is sent again
        within COMMANDER_RETRY_S, as always.
        """
        with self._lock:
            free_at = self._exit_free_at
        deadline = (self._monotonic() if free_at is None else free_at) + wait_s
        while not done.is_set():
            now = self._monotonic()
            if now >= deadline:
                return False
            with self._lock:
                due = self._due()
            # Awake at least every COMMANDER_RETRY_S, for a request the timer
            # sends meanwhile.
            wake = min(deadline, now + COMMANDER_RETRY_S, deadline if due is None else due)
            if done.wait(max(0.0, wake - now)):
                return True
            if due is not None and self._monotonic() >= due:
                # Drops the unanswered request and sends the restore.
                self._reconcile()
        return True

    def close(self) -> None:
        """Send nothing more, once a send under way has gone out, and stop the timer."""
        with self._send_lock:
            with self._lock:
                self._closed = True
        self._timer.cancel()

    # -- Internals --

    def _on_timer(self) -> None:
        try:
            self._reconcile()
        except Exception:
            # Never raised into the executor, which would end the thread.
            logger.exception("A request to %s failed.", self._commander)

    def _reconcile(self) -> None:
        """Send the request the commander needs next, unless one is in flight.

        Under the send lock from the decision until the request is sent.
        """
        with self._send_lock:
            with self._lock:
                request = self._next_request()
            if request is not None:
                self._send(request)

    def _next_request(self) -> _Request | None:
        """Decide what to send next, and mark it in flight. Called under the lock."""
        if self._closed:
            return None
        lost = self._inflight
        if lost is not None:
            if self._monotonic() - lost.sent_at < REQUEST_TIMEOUT_S:
                return None
            self._inflight = None
            if lost.action is None:
                logger.warning(
                    "%s did not answer the read of %s within %g s; reading again.",
                    self._commander,
                    GRASP_DETECT_MIN_PARAM,
                    REQUEST_TIMEOUT_S,
                )
            else:
                self._held = _HELD_UNKNOWN
                self._record(
                    "bag_retry",
                    action=lost.action,
                    reason=f"{self._commander} did not answer within {REQUEST_TIMEOUT_S:g} s",
                )
                self._settle()
        # A hang waits for the read, so it never overwrites the value the
        # read is meant to keep.
        if self._original is None and not self._missing:
            if not self._get.service_is_ready():
                return None
            return self._mark(None, None, None)
        if self._want_hung and self._held != _HELD_HUNG and not self._missing:
            return self._mark_set(HANG, _double(HUNG_VALUE), None)
        if not self._want_hung and self._held != _HELD_ORIGINAL and self._original is not None:
            return self._mark_set(RESTORE, self._original, self._cause)
        return None

    def _mark_set(self, action: str, value: ParameterValue, cause: str | None) -> _Request | None:
        """Mark a set in flight, or log bag_retry while the commander is gone. Under the lock."""
        if not self._set.service_is_ready():
            self._record("bag_retry", action=action, reason=self._no_server())
            return None
        return self._mark(action, value, cause)

    def _mark(
        self, action: str | None, value: ParameterValue | None, cause: str | None
    ) -> _Request:
        self._seq += 1
        self._inflight = _Request(self._seq, action, value, cause, self._monotonic())
        return self._inflight

    def _send(self, request: _Request) -> None:
        """Send the request marked in flight. Called under the send lock, without the lock."""
        with self._lock:
            # Only _reconcile() and close(), under the send lock, drop or
            # replace the request in flight before its answer; checked all
            # the same, so nothing goes out that is no longer decided.
            if self._closed or self._inflight is not request:
                return
        try:
            if request.action is None:
                future = self._get.call_async(
                    GetParameters.Request(names=[GRASP_DETECT_MIN_PARAM])
                )
            else:
                parameter = ParameterMsg(name=GRASP_DETECT_MIN_PARAM, value=request.value)
                future = self._set.call_async(SetParameters.Request(parameters=[parameter]))
            future.add_done_callback(lambda done: self._on_answer(request, done))
        except Exception as e:
            logger.exception("Sending a request to %s failed.", self._commander)
            self._answered(request, None, e)

    def _on_answer(self, request: _Request, future: Future) -> None:
        try:
            try:
                response, error = future.result(), None
            except Exception as e:
                response, error = None, e
            self._answered(request, response, error)
        except Exception:
            # Never raised into the executor, which would end the thread.
            logger.exception("An answer from %s could not be handled.", self._commander)

    def _answered(self, request: _Request, response: Any, error: Exception | None) -> None:
        """Apply the answer to `request`, unless a later request took its place."""
        with self._lock:
            if self._inflight is None or self._inflight.seq != request.seq:
                logger.debug("A late answer from %s was ignored.", self._commander)
                return
            self._inflight = None
            if request.action is None:
                again = self._on_read(response, error)
            else:
                again = self._on_set(request, response, error)
            self._settle()
        if again:
            # The next value goes out at once, such as a restore wanted while
            # a hang was in flight. A failure waits for the timer.
            self._reconcile()

    def _on_read(self, response: Any, error: Exception | None) -> bool:
        """Apply the read's answer; whether the next request may go now. Under the lock."""
        if response is None:
            logger.warning(
                "Reading %s from %s failed (%s); the node reads again.",
                GRASP_DETECT_MIN_PARAM,
                self._commander,
                error if error is not None else "no response",
            )
            return False
        # A node answers no value for a parameter it has not declared; a
        # value of type NOT_SET says the same.
        read = response.values[0] if response.values else ParameterValue()
        self._record("commander", node=self._commander, grasp_detect_min=_parameter_json(read))
        if read.type == ParameterType.PARAMETER_NOT_SET:
            self._missing = True
            if self._want_hung:
                self._give_up_hang(self._no_parameter())
        elif _negative(read):
            # A hang left by a run that did not restore it: the commander's
            # default goes back, unless a hang is wanted meanwhile.
            self._original = _double(GRASP_DETECT_MIN)
            self._held = _HELD_UNKNOWN
            if not self._want_hung:
                self._cause = "start"
        else:
            self._original = read
        return True

    def _on_set(self, request: _Request, response: Any, error: Exception | None) -> bool:
        """Apply a set's answer; whether the next request may go now. Under the lock."""
        result = response.results[0] if response is not None and response.results else None
        if result is None:
            # The commander may hold either value now.
            self._held = _HELD_UNKNOWN
            if error is not None:
                reason = f"setting {GRASP_DETECT_MIN_PARAM} failed: {error}"
            else:
                reason = f"{self._commander} answered with no result"
            self._record("bag_retry", action=request.action, reason=reason)
            return False
        if result.successful:
            if request.action == HANG:
                self._held = _HELD_HUNG
                self._record("bag_hung", value=HUNG_VALUE, already_held=False)
                if self._hang_event is not None:
                    self._hang_event.set()
            else:
                self._held = _HELD_ORIGINAL
                self._record(
                    "bag_restored", value=_parameter_json(request.value), cause=request.cause
                )
                if self._restore_event is not None:
                    self._restore_event.set()
                    self._restore_event = None
            return True
        reason = result.reason or f"{self._commander} did not take the value"
        if request.action == HANG:
            # The commander would refuse it again.
            self._give_up_hang(reason)
            return True
        self._record("bag_retry", action=RESTORE, reason=reason)
        return False

    def _give_up_hang(self, reason: str) -> None:
        """Log the wanted hang as skipped and stop wanting it. Called under the lock."""
        self._record("bag_skipped", action=HANG, reason=reason)
        self._want_hung = False
        if self._hang_event is not None:
            self._hang_event.set()

    def _settle(self) -> None:
        """Set the events that nothing would set later. Called under the lock."""
        if self._want_hung:
            hang = self._hang_event
            if (
                hang is not None
                and not hang.is_set()
                and self._held == _HELD_HUNG
                and self._inflight is None
            ):
                # The commander holds -1 still, as after a restore it
                # refused, so no set is needed for this hang.
                self._record("bag_hung", value=HUNG_VALUE, already_held=True)
                hang.set()
            return
        if self._hang_event is not None and not self._hang_in_flight():
            # A hang that is no longer wanted, and not in flight, is over.
            self._hang_event.set()
        if (
            self._restore_event is not None
            and self._held == _HELD_ORIGINAL
            and self._inflight is None
            and (self._original is not None or self._missing)
        ):
            # Read as the original, or never changed.
            self._restore_event.set()
            self._restore_event = None

    def _hang_in_flight(self) -> bool:
        return self._inflight is not None and self._inflight.action == HANG

    def _due(self) -> float | None:
        """When the request in flight is dropped if unanswered; None if none. Under the lock."""
        if self._inflight is None:
            return None
        return self._inflight.sent_at + REQUEST_TIMEOUT_S

    def _no_parameter(self) -> str:
        return f"{self._commander} has no parameter {GRASP_DETECT_MIN_PARAM}"

    def _no_server(self) -> str:
        return f"no {self._commander}: {self._commander}/set_parameters has no server"


# -- The node -------------------------------------------------------------------------------


def _arm_json(state: ArmState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {"pose": state.pose, "travel_safe": state.travel_safe}


class HriScenarioNode(Node):
    """The scenario on ROS: the oracle, the answers, the bag and the event log.

    Spin it with a MultiThreadedExecutor: the oracle waits for TF on its own
    callback group, and close() waits for the commander's answer, which
    arrives on an executor thread. Call close() before shutting the
    executor down, so the restore can still be answered. After close(), the
    scenario's inputs, the tick and the oracle's scenes are ignored.

    Every call into HriScenario holds the node's lock: the scenario's inputs
    and the tick run in one mutually exclusive group, but the oracle asks
    for the scene from its own group. Each input reads the node's clock
    once (_enter()), and the whole input uses that time.
    """

    def __init__(
        self,
        loaded: LoadedScenario,
        log: EventLog,
        *,
        use_sim_time: bool = True,
        topics: ScenarioTopics = DEFAULT_TOPICS,
        oracle_config: OracleConfig | None = None,
        arm_poses: ArmPoses | None = None,
        robot_model: RobotModel | None = None,
    ) -> None:
        super().__init__(
            "hri_scenario",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)],
        )
        spec = loaded.spec
        self._log = log
        self._topics = topics
        self._lock = threading.Lock()
        self._closed = False
        self._arm_poses = arm_poses if arm_poses is not None else load_arm_poses()
        model = robot_model if robot_model is not None else load_robot_model()
        self._fingers = {arm: model.groups[GRIPPER_GROUPS[arm]][0] for arm in ARMS}
        self._joints: dict[str, float] = {}
        self._nav_active = False
        self._ground_truth: tuple[Pose, float] | None = None
        self._waiting_logged = False
        self._pose_found = False
        # A hang the commander has not taken yet, when the wait for it began,
        # and until when the tick holds the answers back for it, on the
        # computer's monotonic clock.
        self._hang_done: threading.Event | None = None
        self._hang_since = 0.0
        self._hang_deadline = 0.0
        config = oracle_config
        if config is None:
            config = OracleConfig(seed=spec.seed, **spec.oracle.overrides())

        self._scenario = HriScenario(spec, loaded.locations, self._now)
        scenario = self._scenario
        self._record(
            "start",
            world=spec.world,
            file=str(loaded.path),
            seed=spec.seed,
            door=_pose_json(scenario.door),
            living_room=_pose_json(scenario.living_room),
            locations=str(loaded.locations_path),
            places={pid: [place.x, place.y] for pid, place in scenario.places.items()},
            scenario=spec.model_dump(mode="json"),
            guests={
                guest_id: {"name": guest.name, "drink": guest.drink}
                for guest_id, guest in zip(spec.guest_ids, spec.guests)
            },
            bag_owner=spec.bag_owner(),
            onlookers=list(spec.onlooker_ids),
            oracle=dataclasses.asdict(config),
            wall_time=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        # The time of the last input, to find a clock that went back.
        self._last_t = self._now()
        # How often the clock went back, so the oracle drops an image whose
        # drawing spans a jump (OraclePerception's epoch).
        self._epoch = 0
        self._write_events(scenario.tick(self._last_t).events)

        # One buffer for the robot's pose and the oracle's camera transforms,
        # on the node's clock, so a simulation restart that moves sim time
        # back clears it.
        self._tf_buffer = Buffer(node=self)
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._oracle = OraclePerception(
            self, self._scene_at, config, tf_buffer=self._tf_buffer, epoch=self._clock_epoch
        )

        # The scenario's inputs run one at a time, in the order they came.
        # Each stream of the robot's state has a mutually exclusive group of
        # its own: its messages are handled one at a time in the order
        # taken, and none waits behind another stream or a slow input.
        scenario_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            HRIMessage,
            topics.to_human,
            self._on_to_human,
            TO_HUMAN_QOS,
            callback_group=scenario_group,
        )
        self.create_subscription(
            String, topics.todos, self._on_todos, TODOS_QOS, callback_group=scenario_group
        )
        # As the agent subscribes it, so every reset the agent gets reaches here too.
        self.create_subscription(
            Empty, topics.reset, self._on_reset, 10, callback_group=scenario_group
        )
        self.create_subscription(
            TwistStamped,
            topics.cmd_vel,
            self._on_cmd_vel,
            CMD_VEL_QOS,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_subscription(
            GoalStatusArray,
            topics.nav_status,
            self._on_nav_status,
            qos_profile_action_status_default,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        # BEST_EFFORT, which matches either publisher QoS.
        self.create_subscription(
            JointState,
            topics.joint_states,
            self._on_joint_states,
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_subscription(
            Odometry,
            topics.ground_truth,
            self._on_ground_truth,
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._from_human = self.create_publisher(HRIMessage, topics.from_human, FROM_HUMAN_QOS)
        self._grasp = GraspParameter(self, topics.commander, self._record)
        self.create_timer(STATE_PERIOD_S, self._on_tick, callback_group=scenario_group)

    @property
    def scenario(self) -> HriScenario:
        """The scenario's logic. Its methods are not thread-safe while the node spins."""
        return self._scenario

    def close(self, timeout_s: float = RESTORE_WAIT_S) -> None:
        """Restore grasp_detect_min and end the log, while the node still spins.

        From here on no input reaches the scenario, so no answer goes out
        and nothing hangs the bag again. The restore gets `timeout_s` from
        when it can first go out, after a request in flight is answered or
        dropped at its REQUEST_TIMEOUT_S, so close() waits at most
        REQUEST_TIMEOUT_S + `timeout_s` (GraspParameter.wait_restored()).
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._hang_done = None
        done = self._grasp.finish()
        if done is not None and not self._grasp.wait_restored(done, timeout_s):
            self._record(
                "bag_skipped",
                action=RESTORE,
                reason=(
                    f"{self._topics.commander} did not take the value back within "
                    f"{timeout_s:g} s"
                ),
            )
        self._grasp.close()
        self._record("end")

    # -- Callbacks --

    def _on_to_human(self, msg: HRIMessage) -> None:
        try:
            # A fresh pose, so a seat offer judges the heading the line was said at.
            pose = self._lookup_pose()
            with self._lock:
                now = self._enter()
                if now is None:
                    return
                if pose is not None:
                    self._write_events(self._scenario.robot_pose(pose, now).events)
                step = self._scenario.robot_said(msg.text, now)
                self._write_events(step.events)
                if step.bag == HANG:
                    # Under the lock, so no tick can send "It is on your
                    # hand." before the tick knows to hold it (_hang_pending).
                    # The callback does not wait itself: the tick keeps
                    # logging the robot's state at 10 Hz meanwhile. A
                    # request while a hang is pending gets the same event,
                    # and keeps the first deadline.
                    event = self._grasp.hang()
                    if event is not None and event is not self._hang_done:
                        self._hang_done = event
                        self._hang_since = time.monotonic()
                        self._hang_deadline = self._hang_since + HANG_WAIT_S
        except Exception:
            # Never raised into the executor, which would end the thread.
            logger.exception("The scenario could not handle a robot line.")

    def _on_todos(self, msg: String) -> None:
        try:
            items = json.loads(msg.data)
        except json.JSONDecodeError:
            logger.warning("A message on %s is not JSON; ignored.", self._topics.todos)
            return
        if not isinstance(items, list):
            logger.warning("A message on %s is not a JSON list; ignored.", self._topics.todos)
            return
        try:
            with self._lock:
                now = self._enter()
                if now is None:
                    return
                self._write_events(self._scenario.todos_changed(items, now).events)
        except Exception:
            logger.exception("The scenario could not handle a todo list.")

    def _on_reset(self, _msg: Empty) -> None:
        try:
            with self._lock:
                now = self._enter()
                if now is None:
                    return
                step = self._scenario.reset(now)
                self._write_events(step.events)
                # The reset dropped the answer that waited for the hang.
                self._hang_done = None
            if step.bag == RESTORE:
                self._grasp.restore("reset")
        except Exception:
            logger.exception("The scenario could not handle a reset.")

    def _on_tick(self) -> None:
        try:
            pose = self._lookup_pose()
            with self._lock:
                now = self._enter()
                if now is None:
                    return
                if pose is not None:
                    self._write_events(self._scenario.robot_pose(pose, now).events)
                if not self._hang_pending():
                    step = self._scenario.tick(now)
                    self._write_events(step.events)
                    for answer in step.answers:
                        self._publish_answer(answer)
                self._write_state(pose, now)
        except Exception:
            logger.exception("The scenario's tick failed.")

    def _hang_pending(self) -> bool:
        """Whether the answers wait for the commander to take a hang. Called under the lock.

        The scenario's tick waits as a whole, so the answer is logged and
        sent only after bag_hung, and the answers due meanwhile keep their
        order. After HANG_WAIT_S the answers go out anyway, logged as
        bag_unconfirmed; the hang stays wanted, and bag_hung follows once
        the commander takes it.
        """
        done = self._hang_done
        if done is None:
            return False
        if done.is_set():
            self._hang_done = None
            return False
        now = time.monotonic()
        if now < self._hang_deadline:
            return True
        reason = (
            f"{self._topics.commander} had not taken the hang within {HANG_WAIT_S:g} s, "
            "so the answer went out first"
        )
        logger.warning("%s.", reason)
        self._record("bag_unconfirmed", waited_s=round(now - self._hang_since, 3), reason=reason)
        self._hang_done = None
        return False

    def _on_cmd_vel(self, msg: TwistStamped) -> None:
        try:
            with self._lock:
                nav_active = self._nav_active
                arms = ArmsState.from_joints(self._joints, self._arm_poses.poses)
            linear, angular = msg.twist.linear, msg.twist.angular
            self._record(
                "cmd_vel",
                stamp=Time.from_msg(msg.header.stamp).nanoseconds / 1e9,
                frame_id=msg.header.frame_id,
                linear=[linear.x, linear.y, linear.z],
                angular=[angular.x, angular.y, angular.z],
                nav_active=nav_active,
                arms_travel_safe=arms.travel_safe,
            )
        except Exception:
            logger.exception("The scenario could not log a /cmd_vel message.")

    def _on_nav_status(self, msg: GoalStatusArray) -> None:
        # Every message lists all the goals the server keeps, so the newest
        # one alone says whether a goal is active.
        active = any(status.status in NAV_ACTIVE_STATUSES for status in msg.status_list)
        with self._lock:
            self._nav_active = active

    def _on_joint_states(self, msg: JointState) -> None:
        # Merged by name: a publisher may split the joints over messages.
        with self._lock:
            for name, position in zip(msg.name, msg.position):
                self._joints[name] = float(position)

    def _on_ground_truth(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        pose = Pose(p.x, p.y, euler_from_quaternion([q.x, q.y, q.z, q.w])[2])
        now = self._now()
        with self._lock:
            self._ground_truth = (pose, now)

    # -- Helpers --

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _record(self, type_: str, **fields: Any) -> None:
        self._log.write({"type": type_, "t": self._now(), **fields})

    def _write_events(self, events: Sequence[Mapping[str, Any]]) -> None:
        for event in events:
            self._log.write(event)

    def _enter(self) -> float | None:
        """The time of one input into the scenario, or None once closed. Called under the lock.

        A clock that went back, as when the simulation restarts, first moves
        the scenario's times with it (HriScenario.clock_jumped()), drops the
        ground truth taken on the old clock, and counts a new epoch.
        """
        if self._closed:
            return None
        now = self._now()
        if now < self._last_t:
            self._write_events(self._scenario.clock_jumped(self._last_t, now).events)
            self._ground_truth = None
            self._epoch += 1
        self._last_t = now
        return now

    def _clock_epoch(self) -> int:
        """How often the clock went back so far, as the inputs found it; for the oracle."""
        with self._lock:
            return self._epoch

    def _scene_at(self, stamp: Time) -> Scene | None:
        """The scene as of a camera image's stamp, for the oracle.

        None for a stamp the scenario does not know (HriScenario.people_at()),
        and empty once closed.
        """
        with self._lock:
            if self._enter() is None:
                return Scene()
            return self._scenario.scene(stamp.nanoseconds / 1e9)

    def _lookup_pose(self) -> Pose | None:
        """The robot's pose from TF now, or None while AMCL does not run."""
        try:
            transform = self._tf_buffer.lookup_transform(MAP_FRAME, ROBOT_FRAME, Time())
        except TransformException:
            if not self._pose_found and not self._waiting_logged:
                self._waiting_logged = True
                logger.info(
                    "Waiting for AMCL: TF has no %s -> %s yet. Nav2 activates it about 10 s "
                    "after the simulation starts; until then the robot pose is unknown and the "
                    "oracle publishes nothing.",
                    MAP_FRAME,
                    ROBOT_FRAME,
                )
            return None
        if not self._pose_found:
            self._pose_found = True
            logger.info("TF has %s -> %s: the robot pose is known.", MAP_FRAME, ROBOT_FRAME)
        t, q = transform.transform.translation, transform.transform.rotation
        return Pose(t.x, t.y, euler_from_quaternion([q.x, q.y, q.z, q.w])[2])

    def _publish_answer(self, answer: Answer) -> None:
        msg = HRIMessage(
            text=answer.text, communication_id=uuid.uuid4().hex, seq_no=0, seq_end=True
        )
        msg.header.stamp = self.get_clock().now().to_msg()
        self._from_human.publish(msg)

    def _write_state(self, pose: Pose | None, now: float) -> None:
        """The 10 Hz state record at `now`. Called under the lock."""
        talking = self._scenario.talking(now)
        truth = None
        if self._ground_truth is not None:
            seen, at = self._ground_truth
            # One taken after `now` was taken before the clock went back.
            if 0 <= now - at <= GROUND_TRUTH_MAX_AGE_S:
                truth = seen
        arms = ArmsState.from_joints(self._joints, self._arm_poses.poses)
        self._log.write(
            {
                "type": "state",
                "t": now,
                "pose": _pose_json(pose),
                "ground_truth": _pose_json(truth),
                "talking": talking,
                "conversation": talking is not None,
                "nav_active": self._nav_active,
                "arms_travel_safe": arms.travel_safe,
                "arms": {arm: _arm_json(arms.arm(arm)) for arm in ARMS},
                "grippers": {arm: self._joints.get(self._fingers[arm]) for arm in ARMS},
            }
        )


# -- Entry point ----------------------------------------------------------------------------


def _raise_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m walkie_agent.sim.hri_scenario",
        description="Play the HRI scenario's guests against the agent in walkie-ros's simulation.",
    )
    parser.add_argument("file", type=Path, help="The scenario, such as config/scenarios/hri.yaml.")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="The event log to write (JSON lines). Default: logs/hri_scenario_<time>.jsonl.",
    )
    args = parser.parse_args(remove_ros_args(argv)[1:])
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(name)s]: %(message)s")

    # Checked before ROS starts, so a mistake in a file fails at once.
    try:
        loaded = load_scenario(args.file)
    except ScenarioError as e:
        raise SystemExit(str(e)) from None
    try:
        arm_poses = load_arm_poses()
        robot_model = load_robot_model()
    except (OSError, tomllib.TOMLDecodeError, ArmPoseError) as e:
        raise SystemExit(f"Cannot read the arm poses for the state records: {e}") from None
    log_path = args.log
    if log_path is None:
        log_path = Path("logs") / f"hri_scenario_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    log = EventLog(log_path)

    # rclpy's own handler would shut the context down on Ctrl-C, before the
    # node could restore grasp_detect_min. Without it, Ctrl-C and SIGTERM
    # raise KeyboardInterrupt in this thread while the executor still spins.
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    node: HriScenarioNode | None = None
    executor: MultiThreadedExecutor | None = None
    try:
        node = HriScenarioNode(loaded, log, arm_poses=arm_poses, robot_model=robot_model)
        executor = MultiThreadedExecutor(num_threads=6)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        logger.info("Playing %s; the event log is %s.", loaded.path, log_path)
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        log.close()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
