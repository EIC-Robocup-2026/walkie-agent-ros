"""Which named pose each arm is in, and whether the robot may drive (ADR-0024, ADR-0025).

Each arm is matched on its own, since a pose can move one arm alone
(ADR-0022). An arm is in a pose when every joint of the pose's table for that
arm is within MATCH_TOLERANCE_RAD of the reading: the tolerance
move_arm_to_pose checks after a move, so an arm that a move left within it
counts as arrived. A joint the reading lacks, or gives as a value that is not
a finite number, matches nothing. The robot may drive only while both arms
are in travel-safe poses.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from walkie_agent.arm.poses import ARMS, RIGHT_JOINT2, ArmPose

# The largest joint error, in radians, at which an arm still counts as in a
# pose. The commander's moves end within about 0.02 rad in the simulation.
MATCH_TOLERANCE_RAD = 0.05

SAFE_SENTENCE = "Arms: safe for driving."
NOT_SAFE_SENTENCE = "Arms: not safe for driving."
UNKNOWN_SENTENCE = "Arms: state unknown."


def unread_joints(table: Mapping[str, float], joints: Mapping[str, float]) -> list[str]:
    """The joints of `table` that the reading `joints` gives no usable position for, by name.

    A joint is unread when the reading lacks it, or gives a value that is not
    a finite number. A NaN would otherwise pass every tolerance check, since
    every comparison with it is false. The one exception is
    openarm_right_joint2 at 0, when the reading lacks it: walkie-ros launched
    with right_joint2_fixed holds that joint at 0, and /get_joint_states does
    not list it.
    """
    return sorted(
        joint
        for joint, target in table.items()
        if not (joint in joints and math.isfinite(joints[joint]))
        and not (joint == RIGHT_JOINT2 and target == 0.0 and joint not in joints)
    )


def worst_joint(
    table: Mapping[str, float], joints: Mapping[str, float]
) -> tuple[str, float] | None:
    """The joint of `table` furthest from the reading `joints`, and how far, in radians.

    None when the reading gives no usable position for a joint of the table
    (unread_joints), since the reading then cannot tell where that joint is.
    None as well when nothing is left to compare.
    """
    if unread_joints(table, joints):
        return None
    worst: tuple[str, float] | None = None
    for joint, target in table.items():
        if joint not in joints:
            # openarm_right_joint2 at 0, which right_joint2_fixed holds there.
            continue
        error = abs(float(joints[joint]) - target)
        if worst is None or error > worst[1]:
            worst = (joint, error)
    return worst


def match_pose(arm: str, joints: Mapping[str, float], poses: Mapping[str, ArmPose]) -> str | None:
    """The name of the pose `arm` is in, or None when it is in none.

    `arm` is "left" or "right", and `joints` a reading by joint name, which
    may hold the other arm's joints too. A pose matches only when the reading
    gives every joint of its table as a finite number (worst_joint). When
    several poses match, the one with the smallest largest joint error wins;
    on a tie, the first in `poses`.
    """
    best: tuple[float, str] | None = None
    for name, pose in poses.items():
        table = pose.tables.get(arm)
        if table is None:
            continue
        worst = worst_joint(table, joints)
        if worst is None or worst[1] > MATCH_TOLERANCE_RAD:
            continue
        if best is None or worst[1] < best[0]:
            best = (worst[1], name)
    return None if best is None else best[1]


@dataclass(frozen=True)
class ArmState:
    """One arm as read: the named pose it is in, and whether the robot may drive so."""

    # None when the arm is in no named pose.
    pose: str | None
    # False for an arm in no named pose: nobody has judged that shape safe.
    travel_safe: bool


def arm_state(
    arm: str, joints: Mapping[str, float], poses: Mapping[str, ArmPose]
) -> ArmState | None:
    """`arm`'s state in the reading `joints`, or None when the reading holds none of its joints.

    A reading of one arm's group says nothing about the other arm, which is
    then unknown, not out of every pose.
    """
    arm_joints = {joint for pose in poses.values() for joint in pose.tables.get(arm, {})}
    if not arm_joints & joints.keys():
        return None
    pose = match_pose(arm, joints, poses)
    return ArmState(pose=pose, travel_safe=pose is not None and poses[pose].travel_safe)


@dataclass(frozen=True)
class ArmsState:
    """Both arms as last read. An arm that is None is unknown; ArmsState() knows neither."""

    left: ArmState | None = None
    right: ArmState | None = None

    @classmethod
    def from_joints(cls, joints: Mapping[str, float], poses: Mapping[str, ArmPose]) -> "ArmsState":
        """Match each arm of the reading `joints` against `poses` on its own."""
        return cls(left=arm_state("left", joints, poses), right=arm_state("right", joints, poses))

    def arm(self, arm: str) -> ArmState | None:
        """The state of "left" or "right"."""
        return self.left if arm == "left" else self.right

    @property
    def unsafe_arms(self) -> tuple[str, ...]:
        """The arms known to be out of every travel-safe pose, left first."""
        return tuple(
            arm for arm in ARMS if (state := self.arm(arm)) is not None and not state.travel_safe
        )

    @property
    def travel_safe(self) -> bool | None:
        """True when both arms are in travel-safe poses, False when either is not, else None.

        One arm known to be out is enough for False, whatever the other arm
        is: the robot may not drive either way.
        """
        if self.unsafe_arms:
            return False
        if self.left is None or self.right is None:
            return None
        return True

    def sentence(self) -> str:
        """The sentence every arm tool ends its answer with (ADR-0024)."""
        safe = self.travel_safe
        if safe is None:
            return UNKNOWN_SENTENCE
        return SAFE_SENTENCE if safe else NOT_SAFE_SENTENCE
