"""Shared by the arm atomics: the backend name, config, sentences and helpers.

list_arm_poses, move_arm_to_pose, set_gripper and get_arm_state reach
walkie-ros's bimanual_commander only through the `arm` backend, a
CommanderClient (ADR-0021). The LLM picks poses by name and never sees joint
values (ADR-0022). Every answer, success or failure, ends with the arm state
sentence, so the LLM always knows whether it may drive (ADR-0024). The
module's name starts with "_", so the registry does not load it as a tool.
"""

import threading
from dataclasses import dataclass

from walkie_agent.arm.commander import CommanderClient
from walkie_agent.arm.poses import ARMS, BOTH, GRIPPER_GROUPS, ArmPose
from walkie_agent.arm.state import UNKNOWN_SENTENCE
from walkie_agent.toolbox import StepError, ToolContext

# The backend name in META.requires and ctx.backend().
ARM = "arm"

# The answer when a stop or a reset interrupts the read after a move. The
# move had ended, but the arms were not checked against the pose. A goal that
# a stop or a reset interrupts is answered by interrupted() instead.
STOPPED = f"Stopped by reset. {UNKNOWN_SENTENCE}"

# What to do after a stop or a reset interrupted a goal whose cancel the
# commander accepted (StepError.data["cancel"], CommanderClient): for an arm
# goal, then for a gripper goal. An accepted joint cancel stops both arms,
# but a gripper finishes the motion it is in.
_AFTER_ACCEPTED = (
    "They may take a moment to stop, so call get_arm_state before relying on where they are.",
    "Wait a few seconds for it to finish.",
)
# After a cancel the commander refused, did not know or did not answer, and
# after a goal it had not accepted yet ("pending"): an arm or the gripper
# may still be moving.
_AFTER_UNSURE = "Wait a few seconds, then call get_arm_state."

# Each gripper's finger joint: the one joint of its group in
# config/robot_model.toml, and the joint whose position the commander's grasp
# detection reads. Its position is the travel of one finger, in metres.
FINGER_JOINTS = {"left": "openarm_left_finger_joint1", "right": "openarm_right_finger_joint1"}


@dataclass(frozen=True)
class ArmConfig:
    """What the arm atomics use. A move's own timeout is its pose's timeout_s."""

    # Every read of the joints, including the one after each move, is
    # bounded at 3 s (ADR-0024).
    read_timeout_s: float = 3.0
    # One gripper goal. A close that stalls on an object takes the
    # commander's 5 s controller wait to end (CommanderClient).
    gripper_timeout_s: float = 10.0
    # A close holds something only when the settled gap is at least this far
    # below the open position. A gap closer to open means the fingers barely
    # moved, which no object explains: the gripper is stuck or blocked.
    holding_margin_m: float = 0.004
    # After a close, the commander's `grasped` is used only when its gap is
    # within this of set_gripper's own read of the fingers. A close that the
    # commander did not measure reports a gap of 0.0, and every read that the
    # default detection band counts as holding, GRASP_DETECT_MIN or more, is
    # further than this from it.
    gap_agreement_m: float = 0.002


# Read by the atomics on every call, as `_arm.CONFIG`, so a test can replace it.
CONFIG = ArmConfig()


def commander(ctx: ToolContext) -> CommanderClient:
    return ctx.backend(ARM)


def in_simulation(ctx: ToolContext) -> bool:
    """True when the agent runs on sim time, where the poses not yet tried on the robot are offered.

    main.py's --use-sim-time sets the node's use_sim_time parameter through
    the connector, so the flag that puts the agent on the simulation's clock
    also decides which poses it offers, and nothing else needs wiring.
    """
    return bool(ctx.node.get_parameter("use_sim_time").value)


def offered_poses(ctx: ToolContext, client: CommanderClient) -> dict[str, ArmPose]:
    """The poses the LLM may use here, in file order."""
    return client.poses.offered_poses(sim=in_simulation(ctx))


def pose_name(text: str) -> str:
    """A pose name as arm_poses.toml spells it: the LLM may write "Hands up" or "hands-up"."""
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def arms_text(pose: ArmPose) -> str:
    """Which values of move_arm_to_pose's `arm` the pose accepts, in words."""
    options = pose.arm_options
    if BOTH in options:
        return "left, right or both"
    if len(options) == len(ARMS):
        return "left or right, one arm at a time"
    return f"{options[0]} arm only"


def arm_sentence(client: CommanderClient, cancel: threading.Event, read: bool = False) -> str:
    """The arm state sentence an answer ends with.

    From the client's memory, which every read and every move keeps up to
    date, so most answers cost no call. The arms are read once when that
    memory cannot say whether the robot may drive, as before the first read
    or after a move that did not end with a read, and always with `read`,
    as after a failed move, which may have stopped the arms part way. A read
    that fails leaves the memory's sentence: the answer is about the call,
    and the read is only for its last sentence.
    """
    state = client.last_state
    if (read or state.travel_safe is None) and client.available():
        try:
            state = client.read_arms(cancel, CONFIG.read_timeout_s)
        except StepError:
            state = client.last_state
    return state.sentence()


def interrupted(error: StepError, gripper: bool) -> str | None:
    """What a goal that a stop or a reset interrupted did, and what to do next.

    `error` is the CANCELLED that set_joint_position() or control_gripper()
    raised once the call's cancel was set; `gripper` is True for
    control_gripper(). The client's message says how the commander answered
    the cancel, and what to do about that answer follows it (manipulation.md,
    Safety). Nothing follows when the goal had already ended, or when the
    commander itself reported the goal cancelled. None when no goal was sent,
    as when the reset came while the call waited for the commander.
    """
    message = full_stop(error.message)
    outcome = error.data.get("cancel")
    if outcome is None:
        # Only the commander's own report of a cancelled goal has a status.
        return message if "status" in error.data else None
    if outcome == "ended":
        return message
    if outcome == "accepted":
        return f"{message} {_AFTER_ACCEPTED[1 if gripper else 0]}"
    return f"{message} {_AFTER_UNSURE}"


def read_finger(
    client: CommanderClient, arm: str, cancel: threading.Event, command: int | None = None
) -> float:
    """How far `arm`'s gripper is open now: one finger's travel, in metres, read once.

    With `command`, a GripperResult.command, the read is made only while
    that command is still the gripper's latest
    (CommanderClient.joint_states_since()). Raises StepError as the client's
    read does, and INTERNAL when the reading lacks the finger joint.
    """
    group = GRIPPER_GROUPS[arm]
    if command is None:
        reading = client.joint_states(group, cancel, CONFIG.read_timeout_s)
    else:
        reading = client.joint_states_since(group, command, cancel, CONFIG.read_timeout_s)
    joint = FINGER_JOINTS[arm]
    if joint not in reading:
        raise StepError(
            "INTERNAL",
            f"The commander's reading of {group} lacks {joint}.",
            {"group": group, "joints": list(reading)},
        )
    return reading[joint]


def full_stop(text: str) -> str:
    """`text` ending with a full stop, unless it already ends a sentence.

    Some commander messages end with its own status text, which has none.
    """
    return text if text.endswith((".", "?", "!")) else f"{text}."


def with_sentence(error: StepError, prefix: str, sentence: str) -> StepError:
    """`error` with `prefix` before its message and `sentence` after it, same code and data."""
    return StepError(error.code, f"{prefix}{full_stop(error.message)} {sentence}", error.data)
