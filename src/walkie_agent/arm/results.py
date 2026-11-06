"""The records the commander client returns (walkie_agent.ros.commander).

Plain data, with no ROS import, so the capabilities, the atomics and their tests
can use them on an installation without ROS.
"""

from dataclasses import dataclass

from walkie_agent.arm.state import ArmsState

# Why check_for_driving() could not read the arms (DriveCheck.unread): the
# /get_joint_states service is not discovered, its read did not answer in
# time, or the commander answered that it could not read the joints.
NOT_FOUND = "not found"
NO_ANSWER = "no answer"
NO_READING = "no reading"


@dataclass(frozen=True)
class GripperResult:
    """The commander's answer to one gripper goal."""

    # True when the controller reached the commanded position. A real grasp
    # usually ends false, since the fingers stall on the object; on a close,
    # `grasped` and the gap are the answer.
    success: bool
    status: str
    # Only for a close, a position at or below GRASP_CLOSE_THRESHOLD. On an
    # open or "half" the commander reports false and 0.0 whatever the
    # fingers did, so they are None here and cannot be misread. A close the
    # commander did not measure, with grasp_detect_enable off, superseded,
    # or with no finger reading, reports false and 0.0 too, which looks like
    # a measured empty close.
    grasped: bool | None
    # The settled travel of one finger, in metres.
    gripper_gap: float | None
    # The number of this command among the client's commands for its group,
    # for joint_states_since().
    command: int


@dataclass(frozen=True)
class Recovery:
    """What the recovery that last cleared the unverified state checked (last_recovery)."""

    # unverified_count when it cleared the state, which is also the count
    # when it began: a change in between keeps the state.
    count: int
    # (what, how) for each goal resolved since the arms became unverified,
    # such as ("the move of left_arm", "unknown"). `how` is "result" (its
    # result came), "rejected", "ended" (it answered a cancel that the goal
    # had already ended) or "unknown" (it does not know the goal, as after
    # a restart).
    goals: tuple[tuple[str, str], ...]
    # The gripper groups whose finger the settled read checked, besides
    # both arms.
    grippers: tuple[str, ...]


@dataclass(frozen=True)
class DriveCheck:
    """What check_for_driving() found: both arms, or why they were not read."""

    # From a complete reading of both arms; None when they were not read.
    state: ArmsState | None
    # NOT_FOUND, NO_ANSWER or NO_READING when `state` is None, else None.
    unread: str | None = None


@dataclass(frozen=True)
class ClientSnapshot:
    """What the client knows at one moment (snapshot())."""

    # last_state: the arms as of the last read.
    state: ArmsState
    # last_recovery: what the recovery that last cleared the unverified
    # state checked, else None.
    recovery: Recovery | None
    # unverified_count: how many times the arms became unverified so far.
    count: int
    # unverified: why the arms are unverified, or None.
    unverified: str | None
