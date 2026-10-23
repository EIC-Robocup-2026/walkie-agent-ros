"""Snapshot walkie-ros's arm groups, presets and joint limits into config/robot_model.toml.

    pixi run python scripts/snapshot_robot_model.py ros2_ws/src/walkie-ros

Reads a walkie-ros checkout, never changing it, and writes the file named by
--output (by default this repository's config/robot_model.toml):

- From manipulate/openarm_bimanual_moveit_config/config/openarm_bimanual.srdf,
  the joints of the groups the agent commands and the group states of
  left_arm, right_arm and both_arms. The SRDF is a xacro file: its
  `<xacro:unless value="$(arg right_joint2_fixed)">` wrappers are removed and
  their content kept, which is the SRDF with right_joint2_fixed at its default,
  false, as real_omnibot.launch.py launches it. A group or a group state
  inside any other xacro element, at any depth, stops the script.
- From .../config/joint_limits.yaml, min_position and max_position of every
  joint of those groups. MoveIt enforces these, not the URDF's: they widen
  joints 4 and 7 by 0.02 rad, and some presets need that margin.

The agent checks config/arm_poses.toml against the snapshot at startup, and a
test regenerates it from the pinned clone and compares, so a retune in
walkie-ros reaches the agent only when someone copies it on purpose
(ADR-0022). Values are written as the shortest decimal that reads back as the
same float, so nothing is rounded.
"""

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "config" / "robot_model.toml"

CONFIG_DIR = Path("manipulate/openarm_bimanual_moveit_config/config")
SRDF_PATH = CONFIG_DIR / "openarm_bimanual.srdf"
LIMITS_PATH = CONFIG_DIR / "joint_limits.yaml"

# The groups the agent commands, in the order the snapshot lists them
# (manipulation.md). The *_lift groups also move the lift, which the agent
# never commands (ADR-0026).
AGENT_GROUPS = ("left_arm", "right_arm", "both_arms", "left_gripper", "right_gripper")
# The groups whose presets the arm poses copy.
STATE_GROUPS = ("left_arm", "right_arm", "both_arms")
LIFT_SUFFIX = "_lift"

XACRO = "{http://www.ros.org/wiki/xacro}"
# The one xacro condition the SRDF uses, around a joint of a group state. Its
# content is kept, as when the arg is false. Any other condition stops the
# script: keeping or dropping its content would be a guess.
KEPT_CONDITION = ("unless", "$(arg right_joint2_fixed)")
CONDITION_ARG = "right_joint2_fixed"
# The SRDF elements that hold what the script reads: the groups, with their
# joints, and the group states.
READ_TAGS = ("group", "group_state")

BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


class SnapshotError(Exception):
    """The checkout does not have the shape this script expects."""


def parse_srdf(path: Path) -> ET.Element:
    root = ET.parse(path).getroot()
    for arg in root.iter(f"{XACRO}arg"):
        if arg.get("name") == CONDITION_ARG and arg.get("default") != "false":
            raise SnapshotError(
                f"{path}: {CONDITION_ARG} defaults to {arg.get('default')!r}, not 'false'; "
                "the snapshot keeps the joints the condition removes only when it is false."
            )
    check_enclosing(root, path)
    return root


def check_enclosing(root: ET.Element, path: Path) -> None:
    """Fail on a group or a group state that a xacro element encloses, at any depth.

    xacro keeps or drops such an element by a value the script does not
    evaluate, so reading it or skipping it would be a guess. The elements
    inside a group or a group state are checked where they are read.
    ElementTree keeps no link from an element to its parent, so the links are
    mapped first.
    """
    parents = {child: parent for parent in root.iter() for child in parent}
    # In document order, so an element is named before the elements inside it.
    for element in root.iter():
        if element.tag not in READ_TAGS:
            continue
        parent = parents.get(element)
        while parent is not None:
            if parent.tag.startswith(XACRO):
                raise SnapshotError(
                    f"{path}: <{tag(element)}> {element.attrib} is inside <{tag(parent)}> "
                    f"{parent.attrib}. The snapshot resolves only the {CONDITION_ARG} "
                    "wrappers inside group states, so it cannot tell whether xacro keeps it."
                )
            parent = parents.get(parent)


def tag(element: ET.Element) -> str:
    """The element's tag as the SRDF writes it, such as xacro:unless."""
    return element.tag.replace(XACRO, "xacro:")


def state_joints(state: ET.Element, where: str) -> list[ET.Element]:
    """The <joint> elements of a group state, with the right_joint2_fixed wrappers removed."""
    joints: list[ET.Element] = []
    for child in state:
        if child.tag == "joint":
            joints.append(child)
        elif (child.tag.removeprefix(XACRO), child.get("value")) == KEPT_CONDITION:
            for inner in child:
                if inner.tag != "joint":
                    raise SnapshotError(
                        f"{where}: unexpected <{tag(inner)}> in a xacro condition."
                    )
                joints.append(inner)
        else:
            raise SnapshotError(f"{where}: unexpected <{tag(child)}> {child.attrib}.")
    return joints


def read_value(text: str | None, where: str) -> float:
    try:
        value = float(text or "")
    except ValueError:
        raise SnapshotError(f"{where}: {text!r} is not a number.") from None
    if not math.isfinite(value):
        raise SnapshotError(f"{where}: {text!r} is not a finite number.")
    return value


def read_states(root: ET.Element) -> dict[str, dict[str, dict[str, float]]]:
    """Every group state, as group -> state -> joint -> value, in the SRDF's order.

    Only the children of <robot>, the group states MoveIt reads.
    """
    states: dict[str, dict[str, dict[str, float]]] = {}
    for state in root.findall("group_state"):
        group, name = state.get("group", ""), state.get("name", "")
        where = f"group_state {name!r} of {group!r}"
        if name in states.get(group, {}):
            raise SnapshotError(f"{where} appears twice.")
        values: dict[str, float] = {}
        for joint in state_joints(state, where):
            joint_name = joint.get("name", "")
            if joint_name in values:
                raise SnapshotError(f"{where}: {joint_name} appears twice.")
            values[joint_name] = read_value(joint.get("value"), f"{where}, {joint_name}")
        states.setdefault(group, {})[name] = values
    return states


def group_joints(
    root: ET.Element,
    states: Mapping[str, Mapping[str, Mapping[str, float]]],
    names: Iterable[str],
) -> dict[str, list[str]]:
    """The joints of each group in `names`, in order.

    A group lists its joints, lists subgroups, or names a chain. The joints of
    a chain are in the URDF, not the SRDF, so they are taken from the group's
    states, which must all name the same joints.
    """
    elements = {group.get("name", ""): group for group in root.findall("group")}

    def joints_of(name: str, seen: tuple[str, ...]) -> list[str]:
        if name in seen:
            raise SnapshotError(f"Group {name!r} contains itself.")
        if name not in elements:
            raise SnapshotError(f"Group {name!r} is not defined in the SRDF.")
        joints: list[str] = []
        for child in elements[name]:
            if child.tag == "joint":
                joints.append(child.get("name", ""))
            elif child.tag == "group":
                joints.extend(joints_of(child.get("name", ""), (*seen, name)))
            elif child.tag == "chain":
                joints.extend(chain_joints(name, states.get(name, {})))
            else:
                raise SnapshotError(f"Group {name!r}: unexpected <{tag(child)}>.")
        # Subgroups may share a joint, as both_arms_lift's two halves share
        # the lift; MoveIt counts it once.
        return list(dict.fromkeys(joints))

    return {name: joints_of(name, ()) for name in names}


def chain_joints(group: str, states: Mapping[str, Mapping[str, float]]) -> list[str]:
    """The joints of a chain group, in the order of its first state."""
    if not states:
        raise SnapshotError(
            f"Group {group!r} is a chain with no group state, so its joints are unknown."
        )
    first = list(next(iter(states.values())))
    for name, values in states.items():
        if set(values) != set(first):
            raise SnapshotError(
                f"The states of chain group {group!r} name different joints: "
                f"{name!r} names {list(values)}, another names {first}."
            )
    return first


def read_limits(path: Path, joints: Iterable[str]) -> dict[str, tuple[float, float]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    table = data.get("joint_limits", {}) if isinstance(data, dict) else {}
    limits: dict[str, tuple[float, float]] = {}
    for joint in joints:
        entry = table.get(joint)
        if not isinstance(entry, dict) or not {"min_position", "max_position"} <= set(entry):
            raise SnapshotError(f"{path}: no min_position and max_position for {joint}.")
        low = read_value(str(entry["min_position"]), f"{path}, {joint} min_position")
        high = read_value(str(entry["max_position"]), f"{path}, {joint} max_position")
        if low > high:
            raise SnapshotError(f"{path}: {joint} has min_position {low} above {high}.")
        limits[joint] = (low, high)
    return limits


def check_states(
    groups: Mapping[str, list[str]], states: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> None:
    """Fail on a state that sets a joint outside its group, or leaves one of its joints out."""
    for group in STATE_GROUPS:
        for name, values in states.get(group, {}).items():
            if set(values) != set(groups[group]):
                raise SnapshotError(
                    f"group_state {name!r} of {group!r} names {list(values)}, "
                    f"not the group's joints {groups[group]}."
                )


def toml_key(key: str) -> str:
    if BARE_KEY.match(key):
        return key
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def toml_float(value: float) -> str:
    # repr() is the shortest decimal that reads back as the same float, and
    # always has a "." or an exponent, so TOML reads it as a float.
    return repr(float(value))


def toml_names(names: list[str]) -> str:
    return "[\n" + "".join(f'    "{name}",\n' for name in names) + "]"


def render(
    groups: Mapping[str, list[str]],
    lift_joints: list[str],
    states: Mapping[str, Mapping[str, Mapping[str, float]]],
    limits: Mapping[str, tuple[float, float]],
) -> str:
    lines = [
        "# Generated by scripts/snapshot_robot_model.py. Do not edit by hand.",
        "#",
        "# A snapshot of the walkie-ros commit pinned in ros2_ws.repos:",
        "# - the joints of the groups the agent commands, and the group states of",
        "#   left_arm, right_arm and both_arms, from",
        f"#   {SRDF_PATH}",
        "#   with right_joint2_fixed false, its default. A chain group's joints are",
        "#   the ones its group states name.",
        "# - min_position and max_position of every joint of those groups, from",
        f"#   {LIMITS_PATH}.",
        "# Values are copied exactly.",
        "#",
        "# config/arm_poses.toml is checked against this file at startup, and",
        "# tests/test_arm_poses.py regenerates it from ros2_ws/src/walkie-ros and",
        "# compares (ADR-0022). After moving the pin, regenerate it with",
        "#   pixi run python scripts/snapshot_robot_model.py ros2_ws/src/walkie-ros",
        "# and copy a retuned preset into config/arm_poses.toml on purpose.",
        "",
        "# Joint names in the SRDF's order. The order the commander expects comes",
        "# from /get_joint_states at run time.",
        "[groups]",
    ]
    lines += [f"{toml_key(group)} = {toml_names(groups[group])}" for group in AGENT_GROUPS]
    lines += [
        "",
        "# The joints that only the *_lift groups move (ADR-0026).",
        "[lift]",
        f"joints = {toml_names(lift_joints)}",
    ]
    for group in STATE_GROUPS:
        for name, values in states.get(group, {}).items():
            lines += ["", f"[states.{toml_key(group)}.{toml_key(name)}]"]
            lines += [f"{toml_key(joint)} = {toml_float(value)}" for joint, value in values.items()]
    lines += [
        "",
        "# Radians for the arm joints, metres of finger travel for the grippers.",
        "[limits]",
    ]
    lines += [
        f"{toml_key(joint)} = {{ min_position = {toml_float(low)}, "
        f"max_position = {toml_float(high)} }}"
        for joint, (low, high) in limits.items()
    ]
    return "\n".join(lines) + "\n"


def snapshot(checkout: Path) -> str:
    """The text of robot_model.toml for the walkie-ros checkout at `checkout`."""
    root = parse_srdf(checkout / SRDF_PATH)
    states = read_states(root)
    defined = [group.get("name", "") for group in root.findall("group")]
    missing = [group for group in AGENT_GROUPS if group not in defined]
    if missing:
        raise SnapshotError(f"{checkout / SRDF_PATH}: no group {', '.join(missing)}.")
    lift_groups = [group for group in defined if group.endswith(LIFT_SUFFIX)]
    groups = group_joints(root, states, [*AGENT_GROUPS, *lift_groups])
    check_states(groups, states)
    agent_joints = list(dict.fromkeys(j for group in AGENT_GROUPS for j in groups[group]))
    lift_joints = list(
        dict.fromkeys(
            joint
            for group, joints in groups.items()
            if group.endswith(LIFT_SUFFIX)
            for joint in joints
            if joint not in agent_joints
        )
    )
    limits = read_limits(checkout / LIMITS_PATH, agent_joints)
    return render(groups, lift_joints, states, limits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkout", type=Path, help="a walkie-ros checkout, read only")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="the file to write (%(default)s)"
    )
    args = parser.parse_args()
    try:
        text = snapshot(args.checkout)
    except (OSError, ET.ParseError, yaml.YAMLError, SnapshotError) as err:
        print(f"snapshot_robot_model: {err}", file=sys.stderr)
        return 1
    args.output.write_text(text)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
