"""Shared by the arm atomics: the backend name, config, sentences, helpers and ctx.arms.

list_arm_poses, move_arm_to_pose, set_gripper and get_arm_state reach
walkie-ros's bimanual_commander only through the `arm` backend, a
CommanderClient (ADR-0021). The LLM picks poses by name and never sees joint
values (ADR-0022). Every answer, success or failure, ends with the arm state
sentence, so the LLM always knows whether it may drive (ADR-0024). RosArms is
ctx.arms on that client: the steps of move_arm_to_pose and set_gripper
behind one interface (walkie_agent.capabilities.Arms, ADR-0052). The
module's name starts with "_", so the registry does not load it as a tool.
"""

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from walkie_agent.arm.commander import ClientSnapshot, CommanderClient
from walkie_agent.arm.poses import (
    ARM_GROUPS,
    ARMS,
    BOTH,
    BOTH_ARMS_GROUP,
    GRASP_DETECT_MAX,
    GRASP_DETECT_MIN,
    GRIPPER_GROUPS,
    RIGHT_JOINT2,
    ArmPose,
    GripperPositions,
)
from walkie_agent.arm.state import (
    MATCH_TOLERANCE_RAD,
    UNKNOWN_SENTENCE,
    unread_joints,
    worst_joint,
)
from walkie_agent.capabilities import (
    Arm,
    ArmsMoved,
    ArmsReport,
    Gripper,
    GripperMoved,
    GripperResult,
    GripperState,
)
from walkie_agent.toolbox import StepError, ToolContext

logger = logging.getLogger(__name__)

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

# The commander's status for a success in plan-only mode, where nothing moved.
PLAN_ONLY = "Plan Only"
PLAN_ONLY_TEXT = "The commander runs in plan-only mode, so it moved nothing"
# What to do about it: the commander reads plan_only once, at startup.
PLAN_ONLY_NEXT = (
    "No arm can move until the commander restarts without plan_only, so tell the person "
    "instead of trying again."
)


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
    # further than this from it. A close that meets nothing can be measured
    # before the fingers settle, up to 7.7 mm from the read in a loaded
    # simulation. Either way the read is then judged by the commander's band,
    # as set_gripper reads it.
    gap_agreement_m: float = 0.002


# Read by the atomics on every call, as `_arm.CONFIG`, so a test can replace it.
CONFIG = ArmConfig()


def commander(ctx: ToolContext) -> CommanderClient:
    return ctx.backend(ARM)


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


def arm_refusal(pose: ArmPose) -> str:
    """Why the pose does not take the asked `arm`, such as "'receive' is for the left arm only."."""
    if len(pose.arms) == 1:
        return f"{pose.name!r} is for the {pose.arms[0]} arm only."
    # Both tables exist, but not as one both_arms goal (both_arms = false).
    return f"{pose.name!r} is for one arm at a time: left or right."


def offered_pose(
    offered: Mapping[str, ArmPose], pose: str, arm: str, sentence: Callable[[], str]
) -> ArmPose:
    """The offered pose that `pose` names, if it takes `arm`.

    Raises StepError NOT_FOUND for a pose not in `offered`, since a hidden
    pose gets the same answer as an unknown one, so the LLM is never told
    about a pose it may not use; REFUSED for an `arm` the pose does not
    take. Both messages end with `sentence()`, which runs only then.
    """
    name = pose_name(pose)
    found = offered.get(name)
    if found is None:
        known = ", ".join(offered) or "none"
        raise StepError(
            "NOT_FOUND",
            f"No pose named {pose!r}. The poses are: {known}. {sentence()}",
            {"pose": pose, "poses": list(offered)},
        )
    if arm not in found.arm_options:
        raise StepError(
            "REFUSED",
            f"{arm_refusal(found)} {sentence()}",
            {"pose": name, "arm": arm, "arms": list(found.arm_options)},
        )
    return found


def arms_report(snapshot: ClientSnapshot) -> ArmsReport:
    """The ArmsReport of one CommanderClient.snapshot()."""
    return ArmsReport(
        arms=snapshot.state,
        unverified=snapshot.unverified,
        unverified_count=snapshot.count,
        recovery=snapshot.recovery,
    )


def state_sentence(report: ArmsReport) -> str:
    """The arm state sentence an answer ends with, from one report.

    Never that the robot may drive while the arms are unverified
    (CommanderClient.unverified): the drive check then recovers first, and
    refuses when it cannot. last_state keeps the arms known while only a
    gripper goal made them unverified, so it can say that they are safe for
    driving then. The arms and whether they are unverified come from the
    one snapshot, so that they agree. RosArms.state() says when the arms
    are read for it.
    """
    if travel_safe(report) is None:
        return UNKNOWN_SENTENCE
    return report.arms.sentence()


def travel_safe(report: ArmsReport) -> bool | None:
    """Whether the robot may drive, by `report`, as state_sentence() says it.

    None while the arms are unverified and the arms would say that they
    are safe, as while only a gripper goal made them unverified.
    """
    if report.arms.travel_safe and report.unverified is not None:
        return None
    return report.arms.travel_safe


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


class RosArms:
    """ctx.arms on the agent's CommanderClient (walkie_agent.capabilities.Arms).

    The steps are move_arm_to_pose's and set_gripper's as they were before
    ctx.arms, with the same messages: the pose's joint tables go to the
    commander's /set_joint_position in the order its /get_joint_states
    gives for the group, so a commander that lists its joints in another
    order, or without openarm_right_joint2, still gets the right values. A
    reported success is not trusted: the joints are read after every move
    and compared with the pose. In the commander's plan-only mode it
    reports success with "Plan Only" and moves nothing, so the check passes
    only when the arms were already in the pose.

    Every attribute of the client is looked up on each call, so a test that
    replaces one on the client applies here too.
    """

    def __init__(self, ctx: ToolContext, client: CommanderClient) -> None:
        self._ctx = ctx
        self._client = client

    def poses(self) -> Mapping[str, ArmPose]:
        """The poses the LLM may use here, in file order: outside sim, only those tried."""
        return self._client.poses.offered_poses(sim=self._ctx.in_simulation)

    def state(self, *, read: bool | None = None) -> ArmsReport:
        """The client's snapshot, after a read of the arms when `read` asks for one.

        From the client's memory, which every read and every move keeps up
        to date, so most answers cost no call. With `read` None the arms are
        read once when that memory cannot say whether the robot may drive,
        as before the first read or after a move that did not end with a
        read; with True always, as after a failed move, which may have
        stopped the arms part way; with False never, as when the commander
        is not answering or the base may still be turning. A read that fails
        leaves the memory: the answer is about the call, and the read is
        only for its last sentence. The report comes from one snapshot,
        taken after any read.
        """
        client = self._client
        if (
            read is not False
            and (read or client.last_state.travel_safe is None)
            and client.available()
        ):
            try:
                client.read_arms(self._ctx.cancel, CONFIG.read_timeout_s)
            except StepError:
                pass
        return arms_report(client.snapshot())

    def move_to(self, pose: str, arm: Arm = "both") -> ArmsMoved:
        """Move `arm` to the offered pose `pose`, wait, and check the joints against it."""
        # Here, not at the top: _gaze imports this module.
        from walkie_agent.tools.atomic import _gaze

        ctx, client = self._ctx, self._client
        offered = self.poses()
        name = pose_name(pose)
        found = offered_pose(offered, pose, arm, lambda: state_sentence(self.state()))

        arms = ARMS if arm == BOTH else (arm,)
        group = BOTH_ARMS_GROUP if arm == BOTH else ARM_GROUPS[arm]
        targets = {joint: value for side in arms for joint, value in found.tables[side].items()}
        # Before the first call to the commander on the way to a move, so the
        # base is still before the arm leaves its pose (ADR-0051). A request
        # refused above moves nothing and leaves the gaze loop running.
        try:
            _gaze.stop_before_arm_motion(ctx)
        except StepError as e:
            # The base may still be turning, so nothing goes to the commander,
            # not even a read for the sentence. state(read=False) reads only
            # the client's memory, and never says that the robot may drive
            # while the arms are unverified.
            raise StepError(
                e.code,
                f"Could not move to {name!r}: {full_stop(e.message)} {_gaze.NO_ARM_GOAL} "
                f"{self._memory_sentence()}",
                {"pose": name, **e.data},
            ) from None
        try:
            order = client.joint_names(group, ctx.cancel, CONFIG.read_timeout_s)
        except StepError as e:
            if e.code == "CANCELLED" and ctx.cancel.is_set():
                raise StepError(
                    "CANCELLED",
                    f"Stopped by reset before the arms moved. {self._memory_sentence()}",
                ) from None
            # The read failed, so a second read for the sentence would too.
            raise with_sentence(
                e, f"Could not move to {name!r}: ", self._memory_sentence()
            ) from None
        positions = _positions(self, found, arm, targets, order)

        try:
            status = client.set_joint_position(group, positions, ctx.cancel, found.timeout_s)
        except StepError as e:
            raise _move_error(self, name, e) from None

        try:
            reading = client.joint_states(BOTH_ARMS_GROUP, ctx.cancel, CONFIG.read_timeout_s)
        except StepError as e:
            if e.code == "CANCELLED" and ctx.cancel.is_set():
                raise StepError("CANCELLED", STOPPED) from None
            # A move that cannot be checked counts as failed (ADR-0024).
            raise unchecked(
                status,
                f"the arms could not be checked against {name!r}: {full_stop(e.message)}",
                self._memory_sentence(),
                {"pose": name, "status": status, "read_error": e.code},
            ) from None

        # The report from one snapshot, so that a goal from an earlier call
        # that the commander accepts late cannot come between the sentence
        # and travel_safe.
        report = arms_report(client.snapshot())
        sentence = state_sentence(report)
        worst = worst_joint(targets, reading)
        if worst is None:
            missing = unread_joints(targets, reading)
            raise unchecked(
                status,
                f"its reading lacks a usable position for {', '.join(missing)}, so the arms "
                f"cannot be checked against {name!r}.",
                sentence,
                {"pose": name, "status": status, "missing": missing},
            )
        joint, error = worst
        if error > MATCH_TOLERANCE_RAD:
            raise unchecked(
                status,
                f"{joint} is {error:.2f} rad from {name!r}.",
                sentence,
                {"pose": name, "status": status, "joint": joint, "error_rad": error},
            )
        return ArmsMoved(pose=name, arm=arm, status=status, max_error_rad=error, report=report)

    def gripper(self, arm: Gripper, state: GripperState) -> GripperMoved:
        """Move one gripper to `state`, and after a close judge whether it holds something.

        An open or "half" must succeed: the commander reports a failure when
        the gripper controller did not reach the position, "not_reached". A
        close is judged otherwise, since a real grasp usually ends with
        success false, because the fingers stall on the object (set_gripper
        says how). Only a goal that did not end as the commander allows
        raises; an outcome that leaves whether it holds something unknown
        is returned.
        """
        ctx, client = self._ctx, self._client
        grippers = client.poses.grippers
        position = grippers.position(state)
        try:
            result = client.control_gripper(
                GRIPPER_GROUPS[arm], position, ctx.cancel, CONFIG.gripper_timeout_s
            )
        except StepError as e:
            if e.code == "CANCELLED" and ctx.cancel.is_set():
                stopped = (
                    interrupted(e, gripper=True)
                    or f"Stopped by reset before the {arm} gripper moved."
                )
                raise StepError(
                    "CANCELLED", f"{stopped} {self._memory_sentence()}", e.data
                ) from None
            raise with_sentence(
                e, f"Could not move the {arm} gripper: ", self._memory_sentence()
            ) from None
        if state != "close":
            outcome = "moved" if result.success else "not_reached"
            return GripperMoved(arm=arm, state=state, outcome=outcome, result=result)
        # In each case where whether the close holds something is unknown,
        # the close itself ended as the commander allows, so the step goes
        # on; only the answer about the object is missing.
        if _unmeasured_failure(result):
            return GripperMoved(arm=arm, state=state, outcome="unmeasured", result=result)
        try:
            # The module's read_finger, looked up on each call, so a test can
            # replace it.
            finger = read_finger(client, arm, ctx.cancel, result.command)
            # Raises CANCELLED only, when it reads the commander's band.
            held = _holding(result, finger, grippers, client, ctx.cancel)
        except StepError as e:
            if e.code == "CANCELLED" and ctx.cancel.is_set():
                raise StepError(
                    "CANCELLED",
                    f"Stopped by reset before checking whether the {arm} gripper holds "
                    f"something. {self._memory_sentence()}",
                    e.data,
                ) from None
            if e.code == "PRECONDITION_FAILED" and "command" in e.data:
                # joint_states_since(): a newer command for the gripper came.
                return GripperMoved(arm=arm, state=state, outcome="superseded", result=result)
            return GripperMoved(
                arm=arm, state=state, outcome="unread", result=result, read_error=e.message
            )
        if held is None:
            # The fingers barely moved, whatever `grasped` says.
            return GripperMoved(
                arm=arm, state=state, outcome="not_closed", result=result, finger_m=finger
            )
        return GripperMoved(
            arm=arm,
            state=state,
            outcome="holding" if held else "empty",
            result=result,
            finger_m=finger,
        )

    def finger(self, arm: Gripper) -> float:
        """How far `arm`'s gripper is open now, in metres, read once; no goal is sent."""
        # The module's read_finger, looked up on each call, so a test can
        # replace it.
        return read_finger(self._client, arm, self._ctx.cancel)

    def _memory_sentence(self) -> str:
        """The arm sentence from the client's memory, with no call to the commander."""
        return state_sentence(self.state(read=False))


def unchecked(status: str, problem: str, sentence: str, data: dict[str, Any]) -> StepError:
    """The POSTCONDITION for a move whose check after it failed; `problem` says how."""
    if status == PLAN_ONLY:
        text = f"{PLAN_ONLY_TEXT}, and {problem} Treat this as a failure. {PLAN_ONLY_NEXT}"
    else:
        text = f"The commander reported success, but {problem} Treat this as a failure."
    return StepError("POSTCONDITION", f"{text} {sentence}", data)


def _positions(
    arms: RosArms,
    pose: ArmPose,
    arm: str,
    targets: dict[str, float],
    order: tuple[str, ...],
) -> list[float]:
    """The pose's values in the commander's joint order for the group.

    walkie-ros launched with right_joint2_fixed lists no openarm_right_joint2
    and holds it at 0, so the joint is dropped when the pose has it at 0. A
    pose that needs it elsewhere cannot be reached that way. Any other
    difference means config/robot_model.toml no longer matches the running
    walkie-ros, which the snapshot test should have caught.
    """
    missing = [joint for joint in targets if joint not in order]
    extra = [joint for joint in order if joint not in targets]
    if missing == [RIGHT_JOINT2] and not extra:
        value = targets[RIGHT_JOINT2]
        if value != 0.0:
            raise StepError(
                "REFUSED",
                f"{pose.name!r} cannot move the right arm while walkie-ros runs with "
                f"right_joint2_fixed: the pose needs {RIGHT_JOINT2} at {value:.2f} rad, and "
                "that joint is held at 0. Use the left arm or another pose. "
                f"{state_sentence(arms.state())}",
                {"pose": pose.name, "arm": arm, "joint": RIGHT_JOINT2},
            )
    elif missing or extra:
        raise StepError(
            "INTERNAL",
            f"The commander's joints for {arm} do not match config/robot_model.toml "
            f"(missing: {', '.join(missing) or 'none'}; unknown: {', '.join(extra) or 'none'}). "
            "Regenerate the snapshot for the running walkie-ros. "
            f"{state_sentence(arms.state())}",
            {"pose": pose.name, "missing": missing, "unknown": extra},
        )
    return [targets[joint] for joint in order]


def _move_error(arms: RosArms, name: str, error: StepError) -> StepError:
    """The answer to a move the commander did not finish as asked."""
    if error.code == "CANCELLED" and arms._ctx.cancel.is_set():
        # A stop or a reset: the call ends at once, so the arms are not read.
        stopped = interrupted(error, gripper=False) or "Stopped by reset before the arms moved."
        return StepError("CANCELLED", f"{stopped} {arms._memory_sentence()}", error.data)
    if error.code in ("SKILL_FAILED", "INTERNAL", "CANCELLED"):
        # The commander answered, so it can be read: a failed plan left the
        # arms where they were, and a failed execution may have stopped them
        # part way.
        sentence = state_sentence(arms.state(read=True))
    else:
        # TIMEOUT, UNAVAILABLE or PRECONDITION_FAILED: the commander is not
        # answering or is busy, so a read would fail or wait out its timeout.
        sentence = arms._memory_sentence()
    return with_sentence(error, f"Could not move to {name!r}: ", sentence)


def _unmeasured_failure(result: GripperResult) -> bool:
    """True for a close that the commander reports failed with a gap of 0.0 and grasped false.

    The commander reports that for a close that a newer command for the
    group superseded, also one from another node, and for a failed close
    that it did not measure, as with grasp_detect_enable off. Which of these
    happened cannot be told, and after a supersession the fingers show the
    newer command, so such a close is not judged from a read.
    """
    return not result.success and result.gripper_gap == 0.0 and result.grasped is False


def _holding(
    result: GripperResult,
    finger: float,
    grippers: GripperPositions,
    client: CommanderClient,
    cancel: threading.Event,
) -> bool | None:
    """Whether a close holds something, by `finger`, the travel read after it.

    Only for a close that _unmeasured_failure() does not exclude, read while
    it was still the gripper's latest command. None when the gripper did not
    close: fingers near the open position barely moved, which no object
    explains, so the gripper is stuck or blocked, even when the travel lies
    inside the detection band. Otherwise the commander's `grasped` is the
    answer, by its own detection parameters, when its gap agrees with the
    read. A gap that does not agree is one the commander did not measure on
    a close that reached its position, one it measured before the fingers
    settled, as on a close that meets nothing, or the fingers moved since,
    as another node can make them. The read is then judged by the same
    detection parameters, read from the commander (_band). Raises StepError
    CANCELLED only.
    """
    if finger >= grippers.open_position - CONFIG.holding_margin_m:
        return None
    gap = result.gripper_gap
    if gap is not None and abs(gap - finger) <= CONFIG.gap_agreement_m:
        return bool(result.grasped)
    low, high = _band(client, cancel)
    return low <= finger <= high


def _band(client: CommanderClient, cancel: threading.Event) -> tuple[float, float]:
    """The commander's grasp detection band now, or its default when that cannot be read.

    The default is the one the commander starts with, so a read that fails
    judges as the commander would unless its band was changed. Raises
    StepError CANCELLED only.
    """
    try:
        return client.grasp_band(cancel, CONFIG.read_timeout_s)
    except StepError as e:
        if e.code == "CANCELLED":
            raise
        logger.warning(
            f"Judging a close by the commander's default grasp detection band: {e.message}"
        )
        return GRASP_DETECT_MIN, GRASP_DETECT_MAX
