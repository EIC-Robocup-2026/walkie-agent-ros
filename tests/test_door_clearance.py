"""The receive pose keeps the arms clear of the open door leaf at `door`.

In walkie-ros's world chula_pat14_6 the entrance door starts open, 85 deg into
the room, and the arms collide with it. At `door` the agent holds the left arm
out in receive to take a guest's bag, so no arm collision shape may come within
0.10 m of the leaf wherever Nav2 can end the goal. The joint values come from
config/arm_poses.toml, so a retuned receive pose that comes nearer fails here.

The mujoco package is not in the environment, so the check is numpy: forward
kinematics of the description's walkie.xml, and the convex hulls of the arms'
collision meshes in their body frames from tests/data/arm_collision_hulls.npz
(scripts/extract_arm_collision_hulls.py). MuJoCo collides a mesh as its convex
hull, so the hulls are the collision shapes. A box per link would be too coarse
for a margin of a few millimeters. One test compares the model with what
MuJoCo 3.15's mj_geomDistance measured.

The scene's world frame is the map frame. The base, the lift and the head
mount are not modeled: no arm pose moves them, and at `door` they are 0.46 m
or more from the leaf.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import PROJECT_ROOT
from walkie_agent.arm.poses import load_arm_poses
from walkie_agent.locations import load_locations

CONFIG = PROJECT_ROOT / "config"
HULLS_PATH = Path(__file__).resolve().parent / "data" / "arm_collision_hulls.npz"

# The goal: x and y in the map frame (m), and yaw (rad).
DOOR = (2.35, -0.17, 3.01)

# Nav2 ends a goal anywhere within these of it: general_goal_checker in
# walkie-navigation/config/nav2/nav2_mppi_walkie_sim_params.yaml (lines 128 to
# 132) of walkie-ros navigation 359719f, the params file that walkie-ros's
# nav_mujoco_omnibot.launch.py loads by default.
XY_GOAL_TOLERANCE = 0.1
YAW_GOAL_TOLERANCE = 0.01
# The least distance any arm collision shape may keep from the leaf.
MIN_CLEARANCE = 0.10
# The rim of the xy tolerance is sampled every RIM_STEP_DEG. At a fixed yaw the
# arm only translates, and the distance between a translated convex hull and
# the box has no minimum inside the disc while they are apart, so the least
# clearance is on the rim. Samples 5 deg apart are 0.009 m apart.
RIM_STEP_DEG = 5

# The door leaf, entrance_door_00, as the world starts it: scene.xml of
# simulation/mujoco/worlds/chula_pat14_6/ on walkie-ros's
# feat/chula-pat14-6-simulation (adb9725). The door is the body entrance_door
# (line 147), on a hinge at its origin. The hinge's ref is the body's -85 deg,
# so the door starts in this pose, and nothing drives it. The leaf is a box in
# the body (line 150).
HINGE_POS = np.array([0.9, -0.335, 0.0])
HINGE_QUAT = (0.737277337, 0.0, 0.0, -0.675590208)
LEAF_POS = np.array([0.0, 0.34, 1.05])
# Half sizes of the 0.035 x 0.680 x 2.040 m box.
LEAF_HALF = np.array([0.0175, 0.34, 1.02])

# walkie.xml of walkie-ros's description at 526c976, the commit that
# feat/chula-pat14-6-simulation pins. A body is (name, parent, pos, quat (w, x,
# y, z), joint), and a joint is (type, name, axis) at its body's origin with
# ref 0: a hinge turns the body about the axis by the joint's value, and a
# slide moves it along the axis. The model has no default classes, so these
# are all the attributes that place the bodies.
X, Y, Z = np.eye(3)
IDENTITY = (1, 0, 0, 0)
# base_footprint (line 93) carries the free joint, which starts it at its pos.
BASE_HEIGHT = -0.0062
# Lines 95 and 583.
BASE = (
    ("base_link", "base_footprint", (0, 0, 0.0312), IDENTITY, None),
    ("lift_link", "base_link", (-0.07, 0, 0.3985), (0.707107, 0, 0, 0.707107),
     ("slide", "lift_joint", Z)),
)
# Each arm in the file's order, its names without the openarm_left_ or
# openarm_right_ prefix: lines 614 to 672 for the left arm and 685 to 743 for
# the right. The two differ in link0's pos and quat, link2's quat and joint7's
# axis.
LEFT_ARM = (
    ("link0", "lift_link", (0.03, -0.23, 0.1085), (0.499998, -0.5, 0.500002, -0.5), None),
    ("link1", "link0", (0, 0, 0.0625), IDENTITY, ("hinge", "joint1", Z)),
    ("link2", "link1", (-0.0301, 0, 0.06), (0.707107, -0.707107, 0, 0), ("hinge", "joint2", -X)),
    ("link3", "link2", (0.0301, 0, 0.06625), IDENTITY, ("hinge", "joint3", Z)),
    ("link4", "link3", (0, 0.0315, 0.20375), IDENTITY, ("hinge", "joint4", Y)),
    ("link5", "link4", (0, -0.0315, 0.0955), IDENTITY, ("hinge", "joint5", Z)),
    ("link6", "link5", (0.0375, 0, 0.1205), IDENTITY, ("hinge", "joint6", X)),
    ("link7", "link6", (-0.0375, 0, 0), IDENTITY, ("hinge", "joint7", -Y)),
    ("hand", "link7", (0, 0, 0.1001), IDENTITY, None),
    ("right_finger", "hand", (0, -0.006, 0.015), IDENTITY, ("slide", "finger_joint1", -Y)),
    ("left_finger", "hand", (0, 0.006, 0.015), IDENTITY, ("slide", "finger_joint2", Y)),
    ("wristcam_bracket", "hand", (0, 0, 0), IDENTITY, None),
)
RIGHT_ARM = (
    ("link0", "lift_link", (-0.03, -0.23, 0.1085), (0.499998, 0.5, -0.500002, -0.5), None),
    ("link1", "link0", (0, 0, 0.0625), IDENTITY, ("hinge", "joint1", Z)),
    ("link2", "link1", (-0.0301, 0, 0.06), (0.707107, 0.707107, 0, 0), ("hinge", "joint2", -X)),
    ("link3", "link2", (0.0301, 0, 0.06625), IDENTITY, ("hinge", "joint3", Z)),
    ("link4", "link3", (0, 0.0315, 0.20375), IDENTITY, ("hinge", "joint4", Y)),
    ("link5", "link4", (0, -0.0315, 0.0955), IDENTITY, ("hinge", "joint5", Z)),
    ("link6", "link5", (0.0375, 0, 0.1205), IDENTITY, ("hinge", "joint6", X)),
    ("link7", "link6", (-0.0375, 0, 0), IDENTITY, ("hinge", "joint7", Y)),
    ("hand", "link7", (0, 0, 0.1001), IDENTITY, None),
    ("right_finger", "hand", (0, -0.006, 0.015), IDENTITY, ("slide", "finger_joint1", -Y)),
    ("left_finger", "hand", (0, 0.006, 0.015), IDENTITY, ("slide", "finger_joint2", Y)),
    ("wristcam_bracket", "hand", (0, 0, 0), IDENTITY, None),
)
# The simulation's lift starts at 0.507 m, and the agent never moves it
# (docs/getting-started.md, "Lift height").
LIFT = 0.507

# The MuJoCo check ran with receive and home as config/arm_poses.toml had them
# then, and both grippers open. Kept here, so a retuned pose changes only the
# clearance test.
MUJOCO_LEFT = {
    "openarm_left_joint1": -0.0658863214414044,
    "openarm_left_joint2": -0.07971820689149219,
    "openarm_left_joint3": 0.015474061909107776,
    "openarm_left_joint4": 1.8928851211307536,
    "openarm_left_joint5": 1.5010163287531721,
    "openarm_left_joint6": 0.3812924696543403,
    "openarm_left_joint7": 0.027460132968568748,
}
MUJOCO_RIGHT = {
    "openarm_right_joint1": -0.2618,
    "openarm_right_joint2": 0.0,
    "openarm_right_joint3": 0.0,
    "openarm_right_joint4": 0.2618,
    "openarm_right_joint5": 0.0,
    "openarm_right_joint6": 0.0,
    "openarm_right_joint7": 0.0,
}
MUJOCO_FINGER = 0.044
# What it measured: the clearance of the three collision geoms nearest the
# leaf at `door`, by body and nearest first, and the least clearance over the
# stops, with its body and its stop.
MUJOCO_AT_DOOR = {
    "openarm_left_left_finger": 0.205,
    "openarm_left_wristcam_bracket": 0.233,
    "openarm_left_right_finger": 0.266,
}
MUJOCO_WORST = (0.104, "openarm_left_left_finger", (XY_GOAL_TOLERANCE, 185, -YAW_GOAL_TOLERANCE))
MODEL_TOLERANCE = 0.002


def quat_rotation(quat):
    """The rotation matrix of a quaternion (w, x, y, z), normalized first as MuJoCo does."""
    w, x, y, z = np.asarray(quat, float) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def axis_rotation(axis, angle):
    """The rotation by `angle` about the unit vector `axis` (Rodrigues' formula)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


LEAF_ROTATION = quat_rotation(HINGE_QUAT)
LEAF_CENTRE = HINGE_POS + LEAF_ROTATION @ LEAF_POS


def named(side, arm):
    """The rows of `arm` with the openarm_<side>_ prefix on its bodies and joints."""
    prefix = f"openarm_{side}_"
    rows = []
    for name, parent, pos, quat, joint in arm:
        if parent != "lift_link":
            parent = prefix + parent
        if joint is not None:
            kind, joint_name, axis = joint
            joint = (kind, prefix + joint_name, axis)
        rows.append((prefix + name, parent, pos, quat, joint))
    return tuple(rows)


ARM_BODIES = named("left", LEFT_ARM) + named("right", RIGHT_ARM)
BODIES = BASE + ARM_BODIES


def joint_values(left, right, finger):
    """The joints body_poses reads: the lift, both arms' tables and four fingers at `finger`."""
    fingers = {
        f"openarm_{side}_finger_joint{i}": finger for side in ("left", "right") for i in (1, 2)
    }
    return {"lift_joint": LIFT, **left, **right, **fingers}


def body_poses(base, joints):
    """The world rotation and origin of every body, with base_footprint at `base`."""
    x, y, yaw = base
    poses = {"base_footprint": (axis_rotation(Z, yaw), np.array([x, y, BASE_HEIGHT]))}
    for name, parent, pos, quat, joint in BODIES:
        rotation, origin = poses[parent]
        origin = origin + rotation @ np.asarray(pos, float)
        rotation = rotation @ quat_rotation(quat)
        if joint is not None:
            kind, joint_name, axis = joint
            if kind == "hinge":
                rotation = rotation @ axis_rotation(axis, joints[joint_name])
            else:
                origin = origin + rotation @ (axis * joints[joint_name])
        poses[name] = (rotation, origin)
    return poses


def leaf_clearance(points):
    """A lower bound of the distance from the convex hull of `points` to the leaf, in m.

    Negative when a point is inside the leaf. The nearest point's distance is
    an upper bound: an edge or a face of the hull can come nearer. Along the
    direction from the leaf to that point, the hull and the leaf are at least
    as far apart as their projections, which is the lower bound. The two are
    equal when that point is the hull's nearest, as for the finger at `door`.
    """
    local = (points - LEAF_CENTRE) @ LEAF_ROTATION
    gaps = local - np.clip(local, -LEAF_HALF, LEAF_HALF)
    distances = np.linalg.norm(gaps, axis=1)
    nearest = distances.argmin()
    if distances[nearest] == 0.0:
        # The depth of the deepest point under the leaf face nearest to it.
        return float((np.abs(local) - LEAF_HALF).max(axis=1).min())
    direction = gaps[nearest] / distances[nearest]
    return float((local @ direction).min() - np.abs(direction) @ LEAF_HALF)


def clearances(base, joints, hulls):
    """The clearance of each body's hull from the leaf, with base_footprint at `base`."""
    poses = body_poses(base, joints)
    result = {}
    for body, vertices in hulls.items():
        rotation, origin = poses[body]
        result[body] = leaf_clearance(origin + vertices @ rotation.T)
    return result


def stops(goal):
    """Where Nav2 can end `goal`, keyed by (offset in m, map heading in deg, yaw offset in rad).

    The goal itself and every RIM_STEP_DEG on the rim of the xy tolerance, each
    at the goal's yaw and at both ends of the yaw tolerance.
    """
    x, y, yaw = goal
    offsets = [(0.0, 0)] + [(XY_GOAL_TOLERANCE, heading) for heading in range(0, 360, RIM_STEP_DEG)]
    return {
        (offset, heading, turn): (
            x + offset * math.cos(math.radians(heading)),
            y + offset * math.sin(math.radians(heading)),
            yaw + turn,
        )
        for offset, heading in offsets
        for turn in (-YAW_GOAL_TOLERANCE, 0.0, YAW_GOAL_TOLERANCE)
    }


def worst_stop(goal, joints, hulls):
    """The least clearance over the stops of `goal`, as (clearance, body, stop)."""
    return min(
        (clearance, body, stop)
        for stop, base in stops(goal).items()
        for body, clearance in clearances(base, joints, hulls).items()
    )


@pytest.fixture(scope="module")
def hulls():
    with np.load(HULLS_PATH) as data:
        loaded = {body: data[body].astype(float) for body in data.files}
    # Every arm body has a collision geom, and the chain places each one.
    assert set(loaded) == {name for name, *_ in ARM_BODIES}
    return loaded


def test_door_is_the_locations_files_door():
    door = load_locations(CONFIG / "locations.chula_pat14_6.toml")["door"]
    assert (door.x, door.y, door.yaw) == DOOR


def test_the_model_matches_mujoco(hulls):
    joints = joint_values(MUJOCO_LEFT, MUJOCO_RIGHT, MUJOCO_FINGER)
    at_door = clearances(DOOR, joints, hulls)
    assert sorted(at_door, key=at_door.get)[: len(MUJOCO_AT_DOOR)] == list(MUJOCO_AT_DOOR)
    for body, measured in MUJOCO_AT_DOOR.items():
        assert at_door[body] == pytest.approx(measured, abs=MODEL_TOLERANCE), body
    clearance, body, stop = worst_stop(DOOR, joints, hulls)
    assert (body, stop) == MUJOCO_WORST[1:]
    assert clearance == pytest.approx(MUJOCO_WORST[0], abs=MODEL_TOLERANCE)


def test_receive_clears_the_door_leaf_within_the_goal_tolerance(hulls):
    arm_poses = load_arm_poses(CONFIG / "arm_poses.toml", CONFIG / "robot_model.toml")
    # The right arm stays in home and both grippers are open, as in the MuJoCo
    # check.
    joints = joint_values(
        arm_poses.poses["receive"].tables["left"],
        arm_poses.poses["home"].tables["right"],
        arm_poses.grippers.open_position,
    )
    clearance, body, (offset, heading, turn) = worst_stop(DOOR, joints, hulls)
    assert clearance >= MIN_CLEARANCE, (
        f"In receive, {body} comes {clearance:.3f} m from the door leaf when Nav2 stops "
        f"{offset} m from door towards map heading {heading} deg, with yaw {turn:+} rad."
    )
