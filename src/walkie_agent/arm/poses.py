"""Named arm poses from config/arm_poses.toml, checked against the robot model (ADR-0022).

The LLM picks a pose by name and never sees joint values. load_arm_poses()
reads the poses and config/robot_model.toml, the snapshot of the pinned
walkie-ros commit's SRDF groups, presets and joint limits, checks every pose
against the snapshot, and raises one ArmPoseError that lists every problem
it found. So a typo or an out-of-limit value stops the agent at startup, not
on the robot, and fixing the file takes one round, not one per error.
"""

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

# Relative to the working directory, like main.py's other config files.
DEFAULT_POSES_PATH = Path("config/arm_poses.toml")
DEFAULT_MODEL_PATH = Path("config/robot_model.toml")

# The values of the atomics' `arm` argument that name one arm, in the order
# answers list them.
ARMS = ("left", "right")
BOTH = "both"
# The commander's groups. The *_lift groups also move the lift, which the
# agent never commands (ADR-0026).
ARM_GROUPS = {"left": "left_arm", "right": "right_arm"}
BOTH_ARMS_GROUP = "both_arms"
GRIPPER_GROUPS = {"left": "left_gripper", "right": "right_gripper"}
LIFT_GROUP_SUFFIX = "_lift"

# walkie-ros launched with right_joint2_fixed makes this a fixed joint, and
# /get_joint_states then lists no such joint. Every SRDF preset has it at 0.
RIGHT_JOINT2 = "openarm_right_joint2"

# The commander's grasp_close_threshold parameter, in metres: a gripper goal
# at or below it is a close, and only a close reports `grasped` and the gap.
GRASP_CLOSE_THRESHOLD = 0.03

# How long one move may take when a pose sets no timeout_s. A move takes
# about 5 s in the simulation, plus planning.
DEFAULT_TIMEOUT_S = 30.0

GRIPPER_KEYS = ("open_position", "half_position", "closed_position")
# The gripper position of each value of set_gripper's `state` argument.
GRIPPER_STATES = {"open": "open_position", "half": "half_position", "close": "closed_position"}
_POSE_BOOLS = ("travel_safe", "offered", "both_arms", "tried_on_robot")
_POSE_KEYS = frozenset(("description", "srdf_state", "timeout_s", *_POSE_BOOLS, *ARMS))


class ArmPoseError(ValueError):
    """arm_poses.toml or robot_model.toml is wrong. `errors` holds every problem found."""

    def __init__(self, source: str, errors: list[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        count = "1 problem" if len(errors) == 1 else f"{len(errors)} problems"
        super().__init__(f"{source}: {count}:\n" + "\n".join(f"- {e}" for e in errors))


@dataclass(frozen=True)
class RobotModel:
    """config/robot_model.toml: the agent's groups, the SRDF presets and the joint limits."""

    # Group name -> its joints, in the SRDF's order.
    groups: Mapping[str, tuple[str, ...]]
    # The joints only the *_lift groups move.
    lift_joints: frozenset[str]
    # Group -> SRDF state name -> joint -> value, for left_arm, right_arm
    # and both_arms.
    states: Mapping[str, Mapping[str, Mapping[str, float]]]
    # Joint -> (min_position, max_position), from joint_limits.yaml.
    limits: Mapping[str, tuple[float, float]]

    def arm_joints(self, arm: str) -> tuple[str, ...]:
        return self.groups[ARM_GROUPS[arm]]

    def srdf_table(self, arm: str, state: str) -> dict[str, float] | None:
        """The SRDF state `state` for one arm, or None when the SRDF has none.

        The arm's own group state when there is one, else the arm's joints
        cut from the both_arms state, as tray exists only for both arms.
        """
        own = self.states.get(ARM_GROUPS[arm], {}).get(state)
        if own is not None:
            return dict(own)
        both = self.states.get(BOTH_ARMS_GROUP, {}).get(state)
        if both is None:
            return None
        joints = self.arm_joints(arm)
        return {joint: value for joint, value in both.items() if joint in joints}


@dataclass(frozen=True)
class GripperPositions:
    """The finger travel, in metres, that set_gripper sends for each state."""

    open_position: float
    half_position: float
    closed_position: float

    def position(self, state: str) -> float:
        """The position for set_gripper's `state`: "open", "half" or "close"."""
        return getattr(self, GRIPPER_STATES[state])


@dataclass(frozen=True)
class ArmPose:
    """One named pose: a joint table for each arm it can move."""

    name: str
    description: str
    # The robot may drive with an arm in this pose (ADR-0025).
    travel_safe: bool
    # "left" and/or "right" -> joint name -> radians.
    tables: Mapping[str, Mapping[str, float]]
    # The SRDF state the tables copy, as named in the SRDF.
    srdf_state: str | None = None
    # False keeps the pose from the LLM.
    offered: bool = True
    # False when the two tables cannot be sent together as one both_arms
    # goal, as with both arms in carry, where the hands meet.
    both_arms: bool = True
    # False until the pose has run on the robot with collision checking on;
    # until then only the simulation offers it.
    tried_on_robot: bool = False
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def arms(self) -> tuple[str, ...]:
        """The arms the pose has a table for, left first."""
        return tuple(arm for arm in ARMS if arm in self.tables)

    @property
    def arm_options(self) -> tuple[str, ...]:
        """The values of move_arm_to_pose's `arm` this pose accepts."""
        if len(self.arms) == len(ARMS) and self.both_arms:
            return (*self.arms, BOTH)
        return self.arms


@dataclass(frozen=True)
class ArmPoses:
    """config/arm_poses.toml, checked: the poses in file order, and the gripper positions."""

    poses: Mapping[str, ArmPose]
    grippers: GripperPositions

    def offered_poses(self, sim: bool) -> dict[str, ArmPose]:
        """The poses the LLM may use, in file order.

        `offered = false` hides a pose everywhere. A pose not yet tried on
        the robot is offered only in the simulation (`sim`), where a bad
        pose costs nothing.
        """
        return {
            name: pose
            for name, pose in self.poses.items()
            if pose.offered and (pose.tried_on_robot or sim)
        }


def load_robot_model(path: str | Path = DEFAULT_MODEL_PATH) -> RobotModel:
    """Read config/robot_model.toml. Raises ArmPoseError when it lacks what the loader needs."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    errors: list[str] = []
    groups: dict[str, tuple[str, ...]] = {}
    for name, joints in _table(data.get("groups")).items():
        if isinstance(joints, list) and all(isinstance(joint, str) for joint in joints):
            groups[name] = tuple(joints)
        else:
            errors.append(f"groups.{name} is not a list of joint names.")
    needed = [*ARM_GROUPS.values(), BOTH_ARMS_GROUP, *GRIPPER_GROUPS.values()]
    errors += [f"No group {name}." for name in needed if name not in groups]
    limits: dict[str, tuple[float, float]] = {}
    for joint, entry in _table(data.get("limits")).items():
        entry = _table(entry)
        low, high = entry.get("min_position"), entry.get("max_position")
        if _is_number(low) and _is_number(high):
            limits[joint] = (float(low), float(high))
        else:
            errors.append(f"limits.{joint} needs a min_position and a max_position.")
    for name in needed:
        errors += [
            f"No limits for {joint} of {name}."
            for joint in groups.get(name, ())
            if joint not in limits
        ]
    states = {
        group: MappingProxyType(
            {
                state: MappingProxyType({joint: float(value) for joint, value in table.items()})
                for state, table in _table(by_name).items()
            }
        )
        for group, by_name in _table(data.get("states")).items()
    }
    lift = _table(data.get("lift")).get("joints", [])
    if errors:
        raise ArmPoseError(f"{path} (regenerate it with scripts/snapshot_robot_model.py)", errors)
    return RobotModel(
        groups=MappingProxyType(groups),
        lift_joints=frozenset(lift),
        states=MappingProxyType(states),
        limits=MappingProxyType(limits),
    )


def load_arm_poses(
    path: str | Path = DEFAULT_POSES_PATH, model_path: str | Path = DEFAULT_MODEL_PATH
) -> ArmPoses:
    """Read and check config/arm_poses.toml. Raises ArmPoseError listing every problem."""
    model = load_robot_model(model_path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return check_arm_poses(data, model, source=str(path))


def check_arm_poses(
    data: Mapping[str, Any], model: RobotModel, source: str = "arm_poses.toml"
) -> ArmPoses:
    """Check the parsed arm_poses.toml against the robot model.

    Raises one ArmPoseError naming `source` and listing every problem.
    """
    errors: list[str] = []
    errors += [
        f"Unknown table [{key}]; the file has [grippers] and [poses.<name>]."
        for key in data
        if key not in ("grippers", "poses")
    ]
    grippers = _check_grippers(data.get("grippers"), model, errors)
    poses: dict[str, ArmPose] = {}
    raw_poses = data.get("poses")
    if not isinstance(raw_poses, Mapping) or not raw_poses:
        errors.append("No poses; add a [poses.<name>] table.")
    else:
        for name, entry in raw_poses.items():
            pose = _check_pose(name, entry, model, errors)
            if pose is not None:
                poses[name] = pose
    if errors or grippers is None:
        raise ArmPoseError(source, errors)
    return ArmPoses(poses=MappingProxyType(poses), grippers=grippers)


def _check_grippers(raw: Any, model: RobotModel, errors: list[str]) -> GripperPositions | None:
    if not isinstance(raw, Mapping):
        errors.append("No [grippers] table; it needs " + ", ".join(GRIPPER_KEYS) + ".")
        return None
    errors += [
        f"grippers: unknown key {key!r}; it has " + ", ".join(GRIPPER_KEYS) + "."
        for key in raw
        if key not in GRIPPER_KEYS
    ]
    # Both fingers' limits, so a value is valid for either gripper.
    joints = [joint for group in GRIPPER_GROUPS.values() for joint in model.groups[group]]
    low = max(model.limits[joint][0] for joint in joints)
    high = min(model.limits[joint][1] for joint in joints)
    values: dict[str, float] = {}
    for key in GRIPPER_KEYS:
        if key not in raw:
            errors.append(f"grippers.{key} is missing.")
        elif not _is_number(raw[key]):
            errors.append(f"grippers.{key} = {raw[key]!r} is not a number.")
        elif not low <= float(raw[key]) <= high:
            errors.append(
                f"grippers.{key} = {float(raw[key])} is outside the finger travel "
                f"[{low}, {high}] m."
            )
        else:
            values[key] = float(raw[key])
    closed, half = values.get("closed_position"), values.get("half_position")
    if closed is not None and closed > GRASP_CLOSE_THRESHOLD:
        errors.append(
            f"grippers.closed_position = {closed} is above the commander's "
            f"grasp_close_threshold, {GRASP_CLOSE_THRESHOLD} m, so a close would "
            "not detect a grasp."
        )
    if half is not None and half <= GRASP_CLOSE_THRESHOLD:
        errors.append(
            f"grippers.half_position = {half} is not above the commander's "
            f"grasp_close_threshold, {GRASP_CLOSE_THRESHOLD} m, so the commander "
            "would treat it as a close and may report a grasp."
        )
    if len(values) < len(GRIPPER_KEYS):
        return None
    return GripperPositions(**values)


def _check_pose(name: str, raw: Any, model: RobotModel, errors: list[str]) -> ArmPose | None:
    where = f"poses.{name}"
    if not isinstance(raw, Mapping):
        errors.append(f"{where} is not a table.")
        return None
    count = len(errors)
    for key in raw:
        if key in _POSE_KEYS:
            continue
        if key.endswith(LIFT_GROUP_SUFFIX):
            errors.append(
                f"{where}.{key}: poses of the *_lift groups are rejected, since the "
                "agent never moves the lift (ADR-0026)."
            )
        else:
            errors.append(
                f"{where}: unknown key {key!r}; a pose has " + ", ".join(sorted(_POSE_KEYS)) + "."
            )
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"{where}.description is missing; the LLM chooses poses by it.")
    flags: dict[str, bool] = {}
    for key in _POSE_BOOLS:
        if key in raw and not isinstance(raw[key], bool):
            errors.append(f"{where}.{key} = {raw[key]!r} is not true or false.")
        elif key in raw:
            flags[key] = raw[key]
    if "travel_safe" not in raw:
        errors.append(f"{where}.travel_safe is missing; say whether the robot may drive in it.")
    timeout_s = raw.get("timeout_s", DEFAULT_TIMEOUT_S)
    if not _is_number(timeout_s) or not float(timeout_s) > 0:
        errors.append(f"{where}.timeout_s = {timeout_s!r} is not a positive number of seconds.")
    srdf_state = raw.get("srdf_state")
    if srdf_state is not None and not isinstance(srdf_state, str):
        errors.append(f"{where}.srdf_state = {srdf_state!r} is not a state name.")
        srdf_state = None
    tables: dict[str, MappingProxyType[str, float]] = {}
    for arm in ARMS:
        if arm in raw:
            table = _check_table(f"{where}.{arm}", arm, raw[arm], model, errors)
            if table is not None:
                if srdf_state is not None:
                    _check_srdf(f"{where}.{arm}", arm, table, srdf_state, model, errors)
                tables[arm] = MappingProxyType(table)
    if not any(arm in raw for arm in ARMS):
        errors.append(f"{where} has no joint table; add [{where}.left] or [{where}.right].")
    if len(errors) > count:
        return None
    return ArmPose(
        name=name,
        description=description.strip(),
        tables=MappingProxyType(tables),
        srdf_state=srdf_state,
        timeout_s=float(timeout_s),
        **flags,
    )


def _check_table(
    where: str, arm: str, raw: Any, model: RobotModel, errors: list[str]
) -> dict[str, float] | None:
    """The table's values by joint, with every problem added to `errors`."""
    if not isinstance(raw, Mapping):
        errors.append(f"{where} is not a table of joint values.")
        return None
    joints = model.arm_joints(arm)
    group = ARM_GROUPS[arm]
    table: dict[str, float] = {}
    for joint, value in raw.items():
        if joint in model.lift_joints:
            errors.append(f"{where}: {joint} is the lift, which poses never set (ADR-0026).")
        elif joint not in joints:
            errors.append(f"{where}: {joint} is not a joint of the {arm} arm ({group}).")
        elif not _is_number(value):
            errors.append(f"{where}.{joint} = {value!r} is not a number of radians.")
        else:
            table[joint] = float(value)
            low, high = model.limits[joint]
            if not low <= table[joint] <= high:
                errors.append(
                    f"{where}.{joint} = {table[joint]} is outside its limits [{low}, {high}]."
                )
    missing = [joint for joint in joints if joint not in raw]
    if missing:
        errors.append(
            f"{where} is missing {', '.join(missing)}; a table sets every joint of {group}."
        )
    return table


def _check_srdf(
    where: str,
    arm: str,
    table: Mapping[str, float],
    state: str,
    model: RobotModel,
    errors: list[str],
) -> None:
    expected = model.srdf_table(arm, state)
    if expected is None:
        errors.append(
            f"{where}: robot_model.toml has no SRDF state {state!r} for the {arm} arm; "
            "fix srdf_state, or drop it for a pose that is not an SRDF preset."
        )
        return
    differences = [
        f"{joint} is {table[joint]}, the SRDF has {value}"
        for joint, value in expected.items()
        if joint in table and table[joint] != value
    ]
    if differences:
        errors.append(
            f"{where} differs from the SRDF state {state!r}: {'; '.join(differences)}. "
            "Copy the state from robot_model.toml, or drop srdf_state for a pose "
            "retuned here."
        )


def _table(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_number(value: Any) -> bool:
    # TOML booleans are Python bools, which are ints; a pose value of `true`
    # is a mistake, not 1 rad.
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
