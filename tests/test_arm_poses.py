"""The arm pose files, their loader and checks, pose matching and the arm state sentences.

No ROS graph: the loader and the matcher are plain logic. The robot-model
snapshot is regenerated from the pinned walkie-ros clone that
`pixi run build-ros2` makes, so a pin move that retunes a preset fails here
until someone copies it on purpose (ADR-0022).
"""

import difflib
import importlib.util
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from conftest import PROJECT_ROOT
from walkie_agent.arm.poses import (
    ArmPose,
    ArmPoseError,
    GripperPositions,
    check_arm_poses,
    load_arm_poses,
    load_robot_model,
)
from walkie_agent.arm.state import (
    MATCH_TOLERANCE_RAD,
    NOT_SAFE_SENTENCE,
    SAFE_SENTENCE,
    UNKNOWN_SENTENCE,
    ArmsState,
    ArmState,
    match_pose,
    worst_joint,
)

CONFIG = PROJECT_ROOT / "config"
POSES_PATH = CONFIG / "arm_poses.toml"
MODEL_PATH = CONFIG / "robot_model.toml"
SCRIPT = PROJECT_ROOT / "scripts" / "snapshot_robot_model.py"
WALKIE_ROS = PROJECT_ROOT / "ros2_ws" / "src" / "walkie-ros"
WALKIE_ROS_KEY = "ros2_ws/src/walkie-ros"
SRDF = Path("manipulate/openarm_bimanual_moveit_config/config/openarm_bimanual.srdf")
LIMITS = Path("manipulate/openarm_bimanual_moveit_config/config/joint_limits.yaml")

LEFT = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT = [f"openarm_right_joint{i}" for i in range(1, 8)]

# Recorded in the MuJoCo simulation on 2026-10-23 (arm_poses.toml), joint 1
# to joint 7.
RECEIVE_LEFT = [
    -0.0658863214414044,
    -0.07971820689149219,
    0.015474061909107776,
    1.8928851211307536,
    1.5010163287531721,
    0.3812924696543403,
    0.027460132968568748,
]
CARRY_LEFT = [
    0.502319611786006,
    -0.49939831397427176,
    0.8376006267167734,
    1.5558171542209278,
    1.1509266972060037,
    0.743132906887623,
    -0.7650041056352623,
]
CARRY_RIGHT = [
    -0.502319611786006,
    0.49939831397427176,
    -0.8376006267167734,
    1.5558171542209278,
    -1.1509266972060037,
    -0.743132906887623,
    0.7650041056352623,
]


@pytest.fixture(scope="module")
def model():
    return load_robot_model(MODEL_PATH)


@pytest.fixture(scope="module")
def arm_poses():
    return load_arm_poses(POSES_PATH, MODEL_PATH)


@pytest.fixture
def raw_poses():
    with open(POSES_PATH, "rb") as f:
        return tomllib.load(f)


def table(arm: str, values: list[float]) -> dict[str, float]:
    return dict(zip(LEFT if arm == "left" else RIGHT, values, strict=True))


def reading(poses, left: str | None, right: str | None) -> dict[str, float]:
    """A joint reading with each arm exactly in the named pose, or absent for None."""
    joints: dict[str, float] = {}
    if left is not None:
        joints.update(poses[left].tables["left"])
    if right is not None:
        joints.update(poses[right].tables["right"])
    return joints


def load_snapshot_script():
    spec = importlib.util.spec_from_file_location("snapshot_robot_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The committed files.


def test_committed_files_load(arm_poses):
    poses = arm_poses.poses
    assert list(poses) == [
        "home",
        "hands_up",
        "standby",
        "tray",
        "pre_place",
        "basket",
        "basket_hold",
        "receive",
        "carry",
    ]
    assert {name for name, pose in poses.items() if pose.travel_safe} == {"home", "carry"}
    assert {name for name, pose in poses.items() if not pose.offered} == {
        "pre_place",
        "basket",
        "basket_hold",
    }
    assert {name for name, pose in poses.items() if not pose.tried_on_robot} == {
        "tray",
        "receive",
        "carry",
    }
    assert poses["home"].arm_options == ("left", "right", "both")
    assert poses["pre_place"].arm_options == ("left",)
    assert poses["receive"].arm_options == ("left",)
    # Both arms in carry put the hands together in front of the body, and
    # MoveIt finds no plan.
    assert not poses["carry"].both_arms
    assert poses["carry"].arm_options == ("left", "right")
    assert all(pose.timeout_s == 30.0 and pose.description for pose in poses.values())
    assert arm_poses.grippers == GripperPositions(
        open_position=0.044, half_position=0.036, closed_position=0.0
    )
    assert arm_poses.grippers.position("half") == 0.036


def test_srdf_presets_are_the_snapshot_states(arm_poses, model):
    copied = {name: pose for name, pose in arm_poses.poses.items() if pose.srdf_state}
    assert set(copied) == {
        "home",
        "hands_up",
        "standby",
        "tray",
        "pre_place",
        "basket",
        "basket_hold",
    }
    for pose in copied.values():
        for arm, values in pose.tables.items():
            assert dict(values) == model.srdf_table(arm, pose.srdf_state), (pose.name, arm)
    # tray exists only for both arms, so its tables are cut from that state.
    assert "tray" not in model.states["left_arm"]
    assert dict(arm_poses.poses["tray"].tables["right"])["openarm_right_joint7"] == -1.5882


def test_receive_and_carry_are_the_recorded_values(arm_poses):
    poses = arm_poses.poses
    assert dict(poses["receive"].tables["left"]) == table("left", RECEIVE_LEFT)
    assert poses["receive"].arms == ("left",)
    assert dict(poses["carry"].tables["left"]) == table("left", CARRY_LEFT)
    assert dict(poses["carry"].tables["right"]) == table("right", CARRY_RIGHT)
    # The right arm mirrors the left: every joint negated except joint 4.
    mirrored = [value if i == 3 else -value for i, value in enumerate(CARRY_LEFT)]
    assert CARRY_RIGHT == mirrored


# The snapshot.


def test_robot_model_matches_the_pinned_walkie_ros(tmp_path):
    if not (WALKIE_ROS / SRDF).exists() or not (WALKIE_ROS / LIMITS).exists():
        pytest.skip(
            f"{WALKIE_ROS_KEY} is not cloned, so config/robot_model.toml cannot be "
            "compared with the pinned walkie-ros. Run `pixi run build-ros2` to clone it."
        )
    repos = yaml.safe_load((PROJECT_ROOT / "ros2_ws.repos").read_text())
    pinned = repos["repositories"][WALKIE_ROS_KEY]["version"]
    head = subprocess.run(
        ["git", "-C", str(WALKIE_ROS), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != pinned:
        pytest.fail(
            f"{WALKIE_ROS_KEY} is at {head}, but ros2_ws.repos pins {pinned}. "
            "Run `pixi run build-ros2` to check out the pin."
        )
    output = tmp_path / "robot_model.toml"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(WALKIE_ROS), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    expected, committed = output.read_text(), MODEL_PATH.read_text()
    diff = "".join(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            "config/robot_model.toml",
            f"generated from {WALKIE_ROS_KEY}",
        )
    )
    assert committed == expected, (
        "config/robot_model.toml differs from the pinned walkie-ros. Regenerate it with "
        "`pixi run python scripts/snapshot_robot_model.py ros2_ws/src/walkie-ros`, and copy "
        "a retuned preset into config/arm_poses.toml on purpose.\n" + diff
    )


SMALL_SRDF = """<?xml version="1.0"?>
<robot name="r" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="right_joint2_fixed" default="{default}"/>
  <group name="left_arm_lift"><chain base_link="b" tip_link="l"/></group>
  <group name="left_arm"><chain base_link="m" tip_link="l"/></group>
  <group name="right_arm"><chain base_link="m" tip_link="r"/></group>
  <group name="both_arms"><group name="left_arm"/><group name="right_arm"/></group>
  <group name="left_gripper"><joint name="lf"/></group>
  <group name="right_gripper"><joint name="rf"/></group>
  <group_state name="up" group="left_arm"><joint name="l1" value="0.1745"/></group_state>
  <group_state name="up" group="right_arm">
    <joint name="r1" value="-0.5760"/>
    <xacro:{condition} value="$(arg right_joint2_fixed)">
      <joint name="r2" value="0"/>
    </xacro:{condition}>
  </group_state>
  <group_state name="up" group="left_arm_lift">
    <joint name="lift" value="0.7"/><joint name="l1" value="0.1745"/>
  </group_state>
</robot>
"""
SMALL_LIMITS = {
    "joint_limits": {
        name: {"min_position": -1, "max_position": 0.174533}
        for name in ("l1", "r1", "r2", "lf", "rf", "lift")
    }
}


def small_checkout(tmp_path: Path, default: str = "false", condition: str = "unless") -> Path:
    (tmp_path / SRDF).parent.mkdir(parents=True)
    (tmp_path / SRDF).write_text(SMALL_SRDF.format(default=default, condition=condition))
    (tmp_path / LIMITS).write_text(yaml.safe_dump(SMALL_LIMITS))
    return tmp_path


def test_snapshot_keeps_the_right_joint2_wrappers_and_copies_values_exactly(tmp_path):
    text = load_snapshot_script().snapshot(small_checkout(tmp_path))
    data = tomllib.loads(text)
    assert data["groups"]["right_arm"] == ["r1", "r2"]
    assert data["groups"]["both_arms"] == ["l1", "r1", "r2"]
    assert data["lift"]["joints"] == ["lift"]
    assert data["states"]["right_arm"]["up"] == {"r1": -0.576, "r2": 0.0}
    assert data["states"]["left_arm"]["up"] == {"l1": 0.1745}
    assert data["limits"]["l1"] == {"min_position": -1.0, "max_position": 0.174533}
    assert "lift" not in data["limits"]
    assert "left_arm_lift" not in data["states"]


@pytest.mark.parametrize(
    ("default", "condition", "message"),
    [
        ("true", "unless", "right_joint2_fixed defaults to 'true'"),
        ("false", "if", "group_state 'up' of 'right_arm': unexpected <xacro:if>"),
    ],
)
def test_snapshot_refuses_what_it_cannot_resolve(tmp_path, default, condition, message):
    module = load_snapshot_script()
    with pytest.raises(module.SnapshotError) as err:
        module.snapshot(small_checkout(tmp_path, default, condition))
    assert message in str(err.value)


# The loader's checks. Each case breaks the committed file in one place.


def _set(*path_and_value):
    *path, key, value = path_and_value

    def mutate(data):
        target = data
        for part in path:
            target = target[part]
        target[key] = value

    return mutate


def _delete(*path):
    *path, key = path

    def mutate(data):
        target = data
        for part in path:
            target = target[part]
        del target[key]

    return mutate


CHECKS: list[tuple[str, Callable[[dict], None], str]] = [
    (
        "unknown joint",
        _set("poses", "home", "left", "openarm_right_joint1", 0.0),
        "poses.home.left: openarm_right_joint1 is not a joint of the left arm (left_arm).",
    ),
    (
        "missing joint",
        _delete("poses", "home", "left", "openarm_left_joint7"),
        "poses.home.left is missing openarm_left_joint7; a table sets every joint of left_arm.",
    ),
    (
        "outside the limits",
        _set("poses", "receive", "left", "openarm_left_joint4", 2.5),
        "poses.receive.left.openarm_left_joint4 = 2.5 is outside its limits [-0.04, 2.463461].",
    ),
    (
        "lift joint",
        _set("poses", "receive", "left", "lift_joint", 0.7435),
        "poses.receive.left: lift_joint is the lift, which poses never set (ADR-0026).",
    ),
    (
        "lift group",
        _set("poses", "receive", "left_arm_lift", {"lift_joint": 0.7435}),
        "poses.receive.left_arm_lift: poses of the *_lift groups are rejected",
    ),
    (
        "differs from the SRDF state",
        _set("poses", "home", "left", "openarm_left_joint1", 0.26),
        "poses.home.left differs from the SRDF state 'home': openarm_left_joint1 is 0.26, "
        "the SRDF has 0.2618.",
    ),
    (
        "no such SRDF state",
        _set("poses", "pre_place", "srdf_state", "pre_place"),
        "poses.pre_place.left: robot_model.toml has no SRDF state 'pre_place' for the left arm",
    ),
    (
        "gripper beyond the finger travel",
        _set("grippers", "open_position", 0.05),
        "grippers.open_position = 0.05 is outside the finger travel [0.0, 0.044] m.",
    ),
    (
        "gripper below the finger travel",
        _set("grippers", "closed_position", -0.001),
        "grippers.closed_position = -0.001 is outside the finger travel [0.0, 0.044] m.",
    ),
    (
        "closed above the grasp threshold",
        _set("grippers", "closed_position", 0.031),
        "grippers.closed_position = 0.031 is above the commander's grasp_close_threshold, "
        "0.03 m, so a close would not detect a grasp.",
    ),
    (
        "half at the grasp threshold",
        _set("grippers", "half_position", 0.03),
        "grippers.half_position = 0.03 is not above the commander's grasp_close_threshold",
    ),
    (
        "missing gripper position",
        _delete("grippers", "half_position"),
        "grippers.half_position is missing.",
    ),
    (
        "missing travel_safe",
        _delete("poses", "carry", "travel_safe"),
        "poses.carry.travel_safe is missing",
    ),
    (
        "unknown pose key",
        _set("poses", "home", "travel_saf", True),
        "poses.home: unknown key 'travel_saf'",
    ),
    (
        "no joint table",
        _delete("poses", "receive", "left"),
        "poses.receive has no joint table; add [poses.receive.left] or [poses.receive.right].",
    ),
    (
        "joint value not a number",
        _set("poses", "receive", "left", "openarm_left_joint1", True),
        "poses.receive.left.openarm_left_joint1 = True is not a number of radians.",
    ),
    (
        "flag not a bool",
        _set("poses", "home", "offered", "yes"),
        "poses.home.offered = 'yes' is not true or false.",
    ),
    (
        "timeout not positive",
        _set("poses", "home", "timeout_s", 0),
        "poses.home.timeout_s = 0 is not a positive number of seconds.",
    ),
    (
        "no description",
        _set("poses", "home", "description", " "),
        "poses.home.description is missing",
    ),
    (
        "unknown table",
        _set("pose", {}),
        "Unknown table [pose]",
    ),
]


@pytest.mark.parametrize(
    ("mutate", "message"), [case[1:] for case in CHECKS], ids=[case[0] for case in CHECKS]
)
def test_each_check_fails_with_a_clear_message(raw_poses, model, mutate, message):
    mutate(raw_poses)
    with pytest.raises(ArmPoseError) as err:
        check_arm_poses(raw_poses, model, source="config/arm_poses.toml")
    assert len(err.value.errors) == 1, err.value.errors
    assert message in err.value.errors[0]
    assert str(err.value).startswith("config/arm_poses.toml: 1 problem:\n- ")


def test_several_errors_are_listed_together(raw_poses, model):
    for _, mutate, _ in (CHECKS[0], CHECKS[2], CHECKS[9]):
        mutate(raw_poses)
    with pytest.raises(ArmPoseError) as err:
        check_arm_poses(raw_poses, model, source="config/arm_poses.toml")
    assert len(err.value.errors) == 3
    text = str(err.value)
    assert text.startswith("config/arm_poses.toml: 3 problems:\n")
    for _, _, message in (CHECKS[0], CHECKS[2], CHECKS[9]):
        assert f"- {message}" in text


def test_load_arm_poses_names_the_file(tmp_path):
    path = tmp_path / "arm_poses.toml"
    path.write_text("[grippers]\nopen_position = 0.044\nhalf_position = 0.036\n")
    with pytest.raises(ArmPoseError) as err:
        load_arm_poses(path, MODEL_PATH)
    assert str(err.value).startswith(f"{path}: 2 problems:")
    assert "grippers.closed_position is missing." in err.value.errors
    assert "No poses; add a [poses.<name>] table." in err.value.errors


def test_offered_poses_hide_untried_poses_outside_the_simulation(arm_poses):
    assert list(arm_poses.offered_poses(sim=True)) == [
        "home",
        "hands_up",
        "standby",
        "tray",
        "receive",
        "carry",
    ]
    assert list(arm_poses.offered_poses(sim=False)) == ["home", "hands_up", "standby"]


# Matching a reading to a pose.


def test_match_pose_finds_each_arm_in_its_pose(arm_poses):
    poses = arm_poses.poses
    joints = reading(poses, "receive", "home")
    assert match_pose("left", joints, poses) == "receive"
    assert match_pose("right", joints, poses) == "home"


def test_match_pose_allows_joints_within_the_tolerance(arm_poses):
    poses = arm_poses.poses
    assert MATCH_TOLERANCE_RAD == 0.05
    joints = reading(poses, "home", None)
    joints["openarm_left_joint4"] += 0.049
    assert match_pose("left", joints, poses) == "home"
    joints["openarm_left_joint4"] += 0.002
    assert match_pose("left", joints, poses) is None
    joint, error = worst_joint(poses["home"].tables["left"], joints)
    assert (joint, error) == ("openarm_left_joint4", pytest.approx(0.051))


def test_match_pose_picks_the_nearest_by_largest_joint_error(arm_poses):
    zeros = dict.fromkeys(LEFT, 0.0)
    # Both poses match the reading. It is 0.03 rad from "a" on two joints and
    # 0.04 rad from "b" on one: "b" is nearer in sum, "a" by the largest error.
    a = {**zeros, "openarm_left_joint1": 0.03, "openarm_left_joint2": 0.03}
    b = {**zeros, "openarm_left_joint2": 0.1}
    poses = {
        "b": ArmPose(name="b", description="b", travel_safe=False, tables={"left": b}),
        "a": ArmPose(name="a", description="a", travel_safe=True, tables={"left": a}),
    }
    joints = {**zeros, "openarm_left_joint2": 0.06}
    joint, error = worst_joint(a, joints)
    assert (joint, error) == ("openarm_left_joint1", pytest.approx(0.03))
    joint, error = worst_joint(b, joints)
    assert (joint, error) == ("openarm_left_joint2", pytest.approx(0.04))
    assert match_pose("left", joints, poses) == "a"
    # On a tie the first pose wins: basket and basket_hold share the left arm.
    committed = arm_poses.poses
    basket = reading(committed, "basket", None)
    assert match_pose("left", basket, committed) == "basket"


def test_match_pose_skips_only_a_missing_right_joint2_at_zero(arm_poses):
    poses = arm_poses.poses
    # walkie-ros launched with right_joint2_fixed lists no openarm_right_joint2.
    home = reading(poses, None, "home")
    del home["openarm_right_joint2"]
    assert match_pose("right", home, poses) == "home"
    # carry has the right joint 2 at 0.499 rad, which a fixed joint cannot reach.
    carry = reading(poses, None, "carry")
    del carry["openarm_right_joint2"]
    assert match_pose("right", carry, poses) is None
    # Any other joint missing from the reading leaves the arm unmatched.
    left = reading(poses, "home", None)
    del left["openarm_left_joint3"]
    assert match_pose("left", left, poses) is None


def test_match_pose_ignores_poses_without_a_table_for_the_arm(arm_poses):
    poses = arm_poses.poses
    # Only receive has this left-arm table; no pose has a right table like it.
    joints = {
        name.replace("left", "right"): value
        for name, value in poses["receive"].tables["left"].items()
    }
    assert match_pose("right", joints, poses) is None


# The arm state sentence.


def test_both_arms_travel_safe_is_safe_for_driving(arm_poses):
    poses = arm_poses.poses
    state = ArmsState.from_joints(reading(poses, "carry", "home"), poses)
    assert state.left == ArmState(pose="carry", travel_safe=True)
    assert state.right == ArmState(pose="home", travel_safe=True)
    assert state.travel_safe is True
    assert state.unsafe_arms == ()
    assert state.sentence() == SAFE_SENTENCE == "Arms: safe for driving."


def test_an_arm_out_of_a_travel_safe_pose_is_not_safe(arm_poses):
    poses = arm_poses.poses
    state = ArmsState.from_joints(reading(poses, "receive", "home"), poses)
    assert state.left == ArmState(pose="receive", travel_safe=False)
    assert state.unsafe_arms == ("left",)
    assert state.travel_safe is False
    assert state.sentence() == NOT_SAFE_SENTENCE == "Arms: not safe for driving."


def test_an_arm_in_no_named_pose_is_not_safe(arm_poses):
    poses = arm_poses.poses
    joints = reading(poses, "home", "home")
    joints["openarm_right_joint4"] = 1.0
    state = ArmsState.from_joints(joints, poses)
    assert state.right == ArmState(pose=None, travel_safe=False)
    assert state.unsafe_arms == ("right",)
    assert state.sentence() == NOT_SAFE_SENTENCE


def test_an_unread_arm_is_unknown(arm_poses):
    poses = arm_poses.poses
    assert ArmsState().sentence() == UNKNOWN_SENTENCE == "Arms: state unknown."
    assert ArmsState().travel_safe is None
    # A reading of the left arm's group says nothing about the right arm.
    left_only = ArmsState.from_joints(reading(poses, "home", None), poses)
    assert left_only.right is None
    assert left_only.travel_safe is None
    assert left_only.sentence() == UNKNOWN_SENTENCE
    # One arm known to be out is enough to refuse a drive.
    out = ArmsState.from_joints(reading(poses, "receive", None), poses)
    assert out.unsafe_arms == ("left",)
    assert out.sentence() == NOT_SAFE_SENTENCE
