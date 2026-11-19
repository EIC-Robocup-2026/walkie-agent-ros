"""Where pick_object stands and where it moves the hand: its settings and its geometry.

pick_object drives to a stand-off pose that faces a remembered object, reaches the
hand to a pre-grasp pose through the commander's whole-body group, moves it straight
in to grasp the object at its centre height, closes the gripper and lifts the hand
straight up (decision PK-4). This module holds the numbers, read from
config/pick.toml, and the plain geometry that turns an object's centre and an
approach heading into those poses. None of it reaches the LLM (ADR-0022), and
nothing here imports ROS.

Every position is in the settings' frame, the map frame of the object memory. A
heading is the direction the robot faces at the stand-off pose, in radians from the
frame's +x. The hand approaches along it, so the object lies straight ahead of the
hand, and ahead of the robot a little to its right, on the right hand's side.
"""

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from walkie_agent.arm.poses import ARMS

# Relative to the working directory, like main.py's other config files.
DEFAULT_PICK_PATH = Path("config/pick.toml")

_FLOAT_KEYS = (
    "standoff_m",
    "standoff_left_m",
    "grasp_back_m",
    "pregrasp_back_m",
    "lift_m",
    "hand_roll_deg",
    "hand_pitch_deg",
    "hand_yaw_offset_deg",
    "reach_timeout_s",
    "cartesian_timeout_s",
)
# The keys that must be above zero; every float key must be finite.
_POSITIVE_KEYS = frozenset(
    ("standoff_m", "pregrasp_back_m", "lift_m", "reach_timeout_s", "cartesian_timeout_s")
)
_KEYS = frozenset(("arm", "frame_id", "heading_offsets_deg", *_FLOAT_KEYS))


class PickSettingsError(ValueError):
    """config/pick.toml is wrong. `errors` holds every problem found."""

    def __init__(self, source: str, errors: list[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        count = "1 problem" if len(errors) == 1 else f"{len(errors)} problems"
        super().__init__(f"{source}: {count}:\n" + "\n".join(f"- {e}" for e in errors))


@dataclass(frozen=True)
class HandPose:
    """A target for one hand's link: a position in meters and roll, pitch, yaw in radians.

    In `frame_id`, such as "map". The commander turns the angles into a quaternion
    with tf2's setRPY. Built only by plan_pick, and never shown to the LLM (ADR-0022).
    """

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    frame_id: str


@dataclass(frozen=True)
class PickSettings:
    """The numbers of config/pick.toml, which that file describes key by key."""

    arm: str
    frame_id: str
    standoff_m: float
    standoff_left_m: float
    heading_offsets_deg: tuple[float, ...]
    grasp_back_m: float
    pregrasp_back_m: float
    lift_m: float
    hand_roll_deg: float
    hand_pitch_deg: float
    hand_yaw_offset_deg: float
    reach_timeout_s: float
    cartesian_timeout_s: float


@dataclass(frozen=True)
class PickPlan:
    """One approach to an object: where the base stands, and the hand's three targets.

    The stand-off pose is base_footprint's, (standoff_x, standoff_y) facing
    `heading`. pregrasp, grasp and lift are the hand link's, in that order.
    """

    standoff_x: float
    standoff_y: float
    heading: float
    pregrasp: HandPose
    grasp: HandPose
    lift: HandPose


def load_pick_settings(path: str | Path = DEFAULT_PICK_PATH) -> PickSettings:
    """Read and check config/pick.toml. Raises PickSettingsError listing every problem.

    A missing file raises OSError, as the other config files do.
    """
    source = str(path)
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise PickSettingsError(source, [f"Not valid TOML: {e}."]) from e
    return check_pick_settings(data, source=source)


def check_pick_settings(data: Mapping[str, Any], source: str = "pick.toml") -> PickSettings:
    """Check the parsed pick.toml. Raises one PickSettingsError naming `source`."""
    errors = [f"Unknown key {key!r}." for key in data if key not in _KEYS]
    errors += [f"Missing key {key!r}." for key in sorted(_KEYS) if key not in data]
    arm = data.get("arm")
    if "arm" in data and arm not in ARMS:
        errors.append(f"arm must be one of {', '.join(ARMS)}, not {arm!r}.")
    frame_id = data.get("frame_id")
    if "frame_id" in data and not (isinstance(frame_id, str) and frame_id.strip()):
        errors.append("frame_id must be a frame's name.")
    numbers: dict[str, float] = {}
    for key in _FLOAT_KEYS:
        if key not in data:
            continue
        value = data[key]
        if not _finite(value):
            errors.append(f"{key} must be a finite number, not {value!r}.")
        elif key in _POSITIVE_KEYS and value <= 0:
            errors.append(f"{key} must be above zero, not {value!r}.")
        else:
            numbers[key] = float(value)
    offsets = data.get("heading_offsets_deg")
    if "heading_offsets_deg" in data and not (
        isinstance(offsets, list) and offsets and all(_finite(value) for value in offsets)
    ):
        errors.append("heading_offsets_deg must be a non-empty list of finite numbers.")
    if errors:
        raise PickSettingsError(source, errors)
    return PickSettings(
        arm=arm,
        frame_id=frame_id.strip(),
        heading_offsets_deg=tuple(float(value) for value in offsets),
        **numbers,
    )


def approach_headings(
    settings: PickSettings, robot_x: float, robot_y: float, object_x: float, object_y: float
) -> tuple[float, ...]:
    """The headings to try, in radians wrapped to [-pi, pi], in the settings' order.

    Each is heading_from() the robot's position plus one of the settings' heading
    offsets, so the first one approaches the object from the side the robot is on,
    along the line the robot already stands on.
    """
    start = heading_from(settings, robot_x, robot_y, object_x, object_y)
    return tuple(_wrap(start + math.radians(offset)) for offset in settings.heading_offsets_deg)


def heading_from(
    settings: PickSettings, robot_x: float, robot_y: float, object_x: float, object_y: float
) -> float:
    """The approach heading whose line passes standoff_left_m right of (robot_x, robot_y).

    So a robot standing at (robot_x, robot_y), at any distance from the object, has
    the object on the same line ahead of its right hand as at the stand-off pose of
    the heading. A robot closer to the object than standoff_left_m gets the bearing.
    In radians, wrapped to [-pi, pi].
    """
    dx, dy = object_x - robot_x, object_y - robot_y
    bearing = math.atan2(dy, dx)
    distance = math.hypot(dx, dy)
    if distance <= abs(settings.standoff_left_m):
        return bearing
    return _wrap(bearing + math.asin(settings.standoff_left_m / distance))


def plan_pick(
    settings: PickSettings, x: float, y: float, z: float, heading: float
) -> PickPlan:
    """The approach along `heading` to the object whose centre is (x, y, z).

    The stand-off pose lies standoff_m back from the centre along the heading and
    standoff_left_m to its left, and faces along the heading. The grasp pose puts
    the hand link grasp_back_m back from the centre along the heading; the
    pre-grasp pose stands pregrasp_back_m further back, and the lift pose lift_m
    above the grasp pose. All three hold the settings' roll and pitch, and the
    heading plus hand_yaw_offset_deg as their yaw.
    """
    cos, sin = math.cos(heading), math.sin(heading)
    yaw = _wrap(heading + math.radians(settings.hand_yaw_offset_deg))

    def hand(back_m: float, up_m: float = 0.0) -> HandPose:
        return HandPose(
            x=x - back_m * cos,
            y=y - back_m * sin,
            z=z + up_m,
            roll=math.radians(settings.hand_roll_deg),
            pitch=math.radians(settings.hand_pitch_deg),
            yaw=yaw,
            frame_id=settings.frame_id,
        )

    grasp_back = settings.grasp_back_m
    return PickPlan(
        standoff_x=x - settings.standoff_m * cos - settings.standoff_left_m * sin,
        standoff_y=y - settings.standoff_m * sin + settings.standoff_left_m * cos,
        heading=heading,
        pregrasp=hand(grasp_back + settings.pregrasp_back_m),
        grasp=hand(grasp_back),
        lift=hand(grasp_back, settings.lift_m),
    )


def relative_move(
    start_x: float,
    start_y: float,
    start_yaw: float,
    goal_x: float,
    goal_y: float,
    goal_yaw: float,
) -> tuple[float, float, float]:
    """The relative move from the start pose to the goal pose, both in one frame.

    Returns (forward_m, left_m, turn_deg) as ctx.base.move_relative takes them:
    the goal's offset in the start pose's own axes, and the turn from the start
    yaw to the goal yaw, wrapped to [-180, 180].
    """
    dx, dy = goal_x - start_x, goal_y - start_y
    cos, sin = math.cos(start_yaw), math.sin(start_yaw)
    forward = dx * cos + dy * sin
    left = -dx * sin + dy * cos
    return forward, left, math.degrees(_wrap(goal_yaw - start_yaw))


def _wrap(angle: float) -> float:
    """`angle` in radians, wrapped to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _finite(value: object) -> bool:
    """Whether `value` is an int or a float with a finite value. A bool is not a number here."""
    return (
        not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
    )
