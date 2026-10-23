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
from walkie_agent.arm.poses import ARMS, BOTH, ArmPose
from walkie_agent.arm.state import UNKNOWN_SENTENCE
from walkie_agent.toolbox import StepError, ToolContext

# The backend name in META.requires and ctx.backend().
ARM = "arm"

# The answer to a move that a stop or a reset interrupted (manipulation.md,
# Tools). The arms stopped wherever the cancel found them.
STOPPED = f"Stopped by reset. {UNKNOWN_SENTENCE}"


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


def with_sentence(error: StepError, prefix: str, sentence: str) -> StepError:
    """`error` with `prefix` before its message and `sentence` after it, same code and data."""
    return StepError(error.code, f"{prefix}{error.message} {sentence}", error.data)
