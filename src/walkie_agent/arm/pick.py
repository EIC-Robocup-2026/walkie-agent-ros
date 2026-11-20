"""Where pick_object stands and where it moves the hand: its settings and its geometry.

pick_object drives to a stand-off pose that faces a remembered object, reaches the
hand to a pre-grasp pose through the commander's whole-body group, moves it straight
in to grasp the object at its centre height, closes the gripper, lifts the hand
straight up and moves it beside the base, clear of the surface (decision PK-4). This
module holds the numbers, read from config/pick.toml, the plain geometry that
turns an object's centre and an approach heading into those poses, and the check
that the object and every pose lie in a sane envelope around the base before any
arm goal is sent. None of it reaches the LLM (ADR-0022), and nothing here imports
ROS.

Every position is in the settings' frame, the map frame of the object memory. A
heading is the direction the robot faces at the stand-off pose, in radians from the
frame's +x. The object lies ahead of the robot and a little to its right, on the
right hand's side. The hand approaches along the line from the base to the object,
as walkie-ros's scripted bottle pick (sim_bottle_pick.py) does: in the Chula
simulation, a hand approaching along the heading itself, 17 degrees off that line,
could not follow the straight move in to the bottle.
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
    "clear_forward_m",
    "clear_side_m",
    "home_lift_m",
    "hand_roll_deg",
    "hand_pitch_deg",
    "hand_yaw_offset_deg",
    "reach_timeout_s",
    "cartesian_timeout_s",
    "max_observation_age_s",
    "observation_wait_s",
    "max_target_shift_m",
    "min_target_z_m",
    "max_target_z_m",
    "min_reach_m",
    "max_reach_m",
    "min_reach_z_m",
    "max_reach_z_m",
)
# The keys that must be above zero; every float key must be finite.
_POSITIVE_KEYS = frozenset(
    (
        "standoff_m",
        "pregrasp_back_m",
        "lift_m",
        "home_lift_m",
        "reach_timeout_s",
        "cartesian_timeout_s",
        "max_observation_age_s",
        "observation_wait_s",
        "max_target_shift_m",
        "min_reach_m",
        "max_reach_m",
    )
)
# Each range's lower and upper key: the lower must be below the upper.
_RANGES = (
    ("min_target_z_m", "max_target_z_m"),
    ("min_reach_m", "max_reach_m"),
    ("min_reach_z_m", "max_reach_z_m"),
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
    clear_forward_m: float
    clear_side_m: float
    home_lift_m: float
    hand_roll_deg: float
    hand_pitch_deg: float
    hand_yaw_offset_deg: float
    reach_timeout_s: float
    cartesian_timeout_s: float
    max_observation_age_s: float
    observation_wait_s: float
    max_target_shift_m: float
    min_target_z_m: float
    max_target_z_m: float
    min_reach_m: float
    max_reach_m: float
    min_reach_z_m: float
    max_reach_z_m: float


@dataclass(frozen=True)
class PickPlan:
    """One approach to an object: where the base stands, and the hand's four targets.

    The stand-off pose is base_footprint's, (standoff_x, standoff_y) facing
    `heading`. pregrasp, grasp, lift and clear are the hand link's, in that order,
    and the hand moves in along `approach`.
    """

    standoff_x: float
    standoff_y: float
    heading: float
    approach: float
    pregrasp: HandPose
    grasp: HandPose
    lift: HandPose
    clear: HandPose


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
    for low, high in _RANGES:
        if low in numbers and high in numbers and numbers[low] >= numbers[high]:
            errors.append(f"{low} must be below {high}, not {numbers[low]!r}.")
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
    settings: PickSettings,
    x: float,
    y: float,
    z: float,
    heading: float,
    approach: float | None = None,
    base: tuple[float, float, float] | None = None,
) -> PickPlan:
    """The approach at `heading` to the object whose centre is (x, y, z).

    The stand-off pose lies standoff_m back from the centre along the heading and
    standoff_left_m to its left, and faces along the heading. The hand moves in
    along `approach`, by default the direction from the stand-off pose to the
    centre; pass the direction from where the base stands instead once it has
    stopped. The grasp pose puts the hand link grasp_back_m back from the centre
    along the approach; the pre-grasp pose stands pregrasp_back_m further back,
    and the lift pose lift_m above the grasp pose. The clear pose is at the lift
    pose's height, clear_forward_m ahead of the base and clear_side_m out to the
    picking arm's side; `base` is base_footprint's (x, y, yaw) once the base has
    stopped, by default the stand-off pose. All four hold the settings' roll and
    pitch, and the approach plus hand_yaw_offset_deg as their yaw.
    """
    cos, sin = math.cos(heading), math.sin(heading)
    standoff_x = x - settings.standoff_m * cos - settings.standoff_left_m * sin
    standoff_y = y - settings.standoff_m * sin + settings.standoff_left_m * cos
    if approach is None:
        approach = math.atan2(y - standoff_y, x - standoff_x)
    approach = _wrap(approach)
    along_x, along_y = math.cos(approach), math.sin(approach)
    yaw = _wrap(approach + math.radians(settings.hand_yaw_offset_deg))

    def hand(at_x: float, at_y: float, at_z: float) -> HandPose:
        return HandPose(
            x=at_x,
            y=at_y,
            z=at_z,
            roll=math.radians(settings.hand_roll_deg),
            pitch=math.radians(settings.hand_pitch_deg),
            yaw=yaw,
            frame_id=settings.frame_id,
        )

    def back(back_m: float, up_m: float = 0.0) -> HandPose:
        return hand(x - back_m * along_x, y - back_m * along_y, z + up_m)

    base_x, base_y, base_yaw = (standoff_x, standoff_y, heading) if base is None else base
    # Positive is to the base's left, so the right arm's side is negative.
    left = settings.clear_side_m if settings.arm == "left" else -settings.clear_side_m
    forward = settings.clear_forward_m
    base_cos, base_sin = math.cos(base_yaw), math.sin(base_yaw)
    grasp_back = settings.grasp_back_m
    return PickPlan(
        standoff_x=standoff_x,
        standoff_y=standoff_y,
        heading=heading,
        approach=approach,
        pregrasp=back(grasp_back + settings.pregrasp_back_m),
        grasp=back(grasp_back),
        lift=back(grasp_back, settings.lift_m),
        clear=hand(
            base_x + forward * base_cos - left * base_sin,
            base_y + forward * base_sin + left * base_cos,
            z + settings.lift_m,
        ),
    )


def reach_problems(
    settings: PickSettings,
    plan: PickPlan,
    target: tuple[float, float, float],
    base: tuple[float, float, float],
) -> list[str]:
    """What puts the object or one of the plan's hand poses outside the settings' envelope.

    `target` is the object's centre (x, y, z) and `base` is base_footprint's (x, y,
    yaw) where the base stopped, both in the settings' frame. The object's height
    must lie in [min_target_z_m, max_target_z_m]. The object and each hand pose
    (pregrasp, grasp, lift, clear) must lie [min_reach_m, max_reach_m] from the base
    across the floor, and the hand poses [min_reach_z_m, max_reach_z_m] above the
    frame's floor, where base_footprint stands. Returns one line per problem, with
    its numbers, for the log only (ADR-0022); empty when everything is inside.
    """
    problems = []
    x, y, z = target
    if not settings.min_target_z_m <= z <= settings.max_target_z_m:
        problems.append(
            f"the object's centre is {z:.3f} m high, outside"
            f" [{settings.min_target_z_m:g}, {settings.max_target_z_m:g}]"
        )
    base_x, base_y = base[0], base[1]
    points = [("the object", x, y, None)] + [
        (f"the {name} pose", pose.x, pose.y, pose.z)
        for name, pose in (
            ("pregrasp", plan.pregrasp),
            ("grasp", plan.grasp),
            ("lift", plan.lift),
            ("clear", plan.clear),
        )
    ]
    for name, at_x, at_y, at_z in points:
        distance = math.hypot(at_x - base_x, at_y - base_y)
        if not settings.min_reach_m <= distance <= settings.max_reach_m:
            problems.append(
                f"{name} is {distance:.3f} m from the base, outside"
                f" [{settings.min_reach_m:g}, {settings.max_reach_m:g}]"
            )
        if at_z is not None and not settings.min_reach_z_m <= at_z <= settings.max_reach_z_m:
            problems.append(
                f"{name} is {at_z:.3f} m high, outside"
                f" [{settings.min_reach_z_m:g}, {settings.max_reach_z_m:g}]"
            )
    return problems


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
