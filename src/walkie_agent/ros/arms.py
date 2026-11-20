"""RosArms: ctx.arms over the agent's CommanderClient (ADR-0021, ADR-0052).

The verbs of walkie_agent.capabilities.Arms on walkie-ros's bimanual_commander: named
poses, the grippers, a read of both, pick_object's hand poses, and the arm check before
every drive (ADR-0025). The LLM picks poses by name and never sees joint values
(ADR-0022), nor the hand poses, which pick_object computes from the object memory
(decision PK-4). Every message of move_to(), gripper() and reach() ends with the arm
sentence, ArmsReport.sentence(), so the LLM always knows whether the robot may drive
(ADR-0024).

The client holds the waits, the cancels, the unverified state and recovery. This module
only turns the client's answers into the Arms contract. The client's StepError carries
no data, so what it says about a cancel is read from the end of its message, against
the constants that built it (CANCEL_OUTCOMES, PENDING_OUTCOMES).
"""

import logging
import math
from collections.abc import Mapping

from walkie_agent.arm import answers
from walkie_agent.arm.pick import HandPose
from walkie_agent.arm.poses import (
    ARM_GROUPS,
    ARMS,
    BOTH,
    BOTH_ARMS_GROUP,
    GRASP_DETECT_MAX,
    GRASP_DETECT_MIN,
    GRIPPER_GROUPS,
    LIFT_ARM_GROUPS,
    LIFT_JOINT,
    RIGHT_JOINT2,
    WHOLE_BODY_GROUPS,
    ArmPose,
    GripperPositions,
)
from walkie_agent.arm.results import ClientSnapshot, GripperResult
from walkie_agent.arm.state import MATCH_TOLERANCE_RAD, unread_joints, worst_joint
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.capabilities.records import (
    Arm,
    ArmsMoved,
    ArmsReading,
    ArmsReport,
    Gripper,
    GripperMoved,
    GripperReading,
    GripperState,
)
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.ros.commander import CANCEL_OUTCOMES, PENDING_OUTCOMES, CommanderClient

logger = logging.getLogger(__name__)

# Each gripper's finger joint: the one joint of its group in config/robot_model.toml,
# and the joint whose position the commander's grasp detection reads. Its position is
# the travel of one finger, in metres.
FINGER_JOINTS = {"left": "openarm_left_finger_joint1", "right": "openarm_right_finger_joint1"}

# A close holds something only when the settled finger travel is at least this far
# below the open position. A travel closer to open means the fingers barely moved,
# which no object explains: the gripper is stuck or blocked.
HOLDING_MARGIN_M = 0.004
# After a close, the commander's `grasped` is used only when its gap is within this of
# the read of the finger after the close. A close that the commander did not measure
# reports a gap of 0.0, and every travel that the default detection band counts as
# holding, GRASP_DETECT_MIN or more, is further than this from it. A close that meets
# nothing can be measured before the fingers settle, up to 7.7 mm from the read in a
# loaded simulation. Either way the read is then judged by the commander's band.
GAP_AGREEMENT_M = 0.002

# Before a straight reach, the arm is read every STILL_POLL_S seconds of real time
# until two reads in a row agree within STILL_TOLERANCE_RAD, for at most STILL_WAIT_S.
# The simulation's arm controllers report a trajectory done without checking where it
# ended, so the wrist can still be closing in on the end of the previous reach. A
# straight path planned from such a reading starts more than 0.1 rad from where the
# arm is when it runs, and MoveIt refuses to execute it ("Execution Failed").
STILL_POLL_S = 0.5
STILL_TOLERANCE_RAD = 0.01
STILL_WAIT_S = 10.0

# The answer to pick, place_on and hand_over: the robot has no grasp skill yet.
NO_GRASP_SKILL = (
    "Walkie has no grasp skill yet, so it cannot pick, place or hand over objects. "
    "move_arm_to_pose and set_gripper still work."
)

# What to do after a stop or a reset interrupted a goal whose cancel the commander
# accepted: for an arm goal, then for a gripper goal. An accepted joint cancel stops
# both arms, but a gripper finishes the motion it is in.
_AFTER_ACCEPTED = (
    "They may take a moment to stop, so call get_arm_state before relying on where they are.",
    "Wait a few seconds for it to finish.",
)
# After a cancel the commander refused, did not know or did not answer, and after a
# goal it had not accepted yet: an arm or the gripper may still be moving.
_AFTER_UNSURE = "Wait a few seconds, then call get_arm_state."
# How the client's message starts when the commander itself reported a goal
# cancelled (_move_failed, control_gripper). The goal was sent and has ended.
_COMMANDER_CANCELLED = "The commander cancelled"
# Why a step ended before it sent anything or before it read, for the answers below.
_STOP_IN_EFFECT = "a stop or a reset is in effect"


def arms_report(snapshot: ClientSnapshot) -> ArmsReport:
    """The ArmsReport of one CommanderClient.snapshot()."""
    return ArmsReport(
        arms=snapshot.state,
        unverified=snapshot.unverified,
        unverified_count=snapshot.count,
        recovery=snapshot.recovery,
    )


def interrupted(error: StepError, gripper: bool) -> str | None:
    """What a goal that a stop or a reset interrupted did, and what to do next.

    `error` is the CANCELLED that set_joint_position() or control_gripper() raised
    once the token was set; `gripper` is True for control_gripper(). The client's
    message ends with how the commander answered the cancel ("...; {outcome}."), and
    what to do about that answer follows it. Nothing follows when the goal had already
    ended, or when the commander itself reported the goal cancelled. None when no
    goal was sent, as when the stop came before the send or while the call waited for
    the commander's server.
    """
    text = error.message
    message = answers.full_stop(text)
    if text.startswith(_COMMANDER_CANCELLED):
        return message
    side = 1 if gripper else 0
    for outcome, words in CANCEL_OUTCOMES.items():
        if text.endswith(f"; {words[side]}."):
            if outcome == "ended":
                return message
            if outcome == "accepted":
                return f"{message} {_AFTER_ACCEPTED[side]}"
            return f"{message} {_AFTER_UNSURE}"
    if text.endswith(f"; {PENDING_OUTCOMES[side]}."):
        return f"{message} {_AFTER_UNSURE}"
    return None


class RosArms:
    """ctx.arms on the agent's CommanderClient (walkie_agent.capabilities.Arms).

    `cancel` is the runtime's token, the one the client was made with: a stop is a
    CANCELLED that comes while it is set. `sim` offers the poses not yet tried on the
    robot. A move waits for the longer of its pose's timeout_s and
    `min_move_timeout_s`, so a slow simulation can raise the bound of every move.

    A pose's joint tables go to the commander's /set_joint_position in the order its
    /get_joint_states gives for the group, so a commander that lists its joints in
    another order, or without openarm_right_joint2, still gets the right values. A
    reported success is not trusted: the joints are read after every move and compared
    with the pose. In the commander's plan-only mode it reports success with "Plan
    Only" and moves nothing, so the check passes only when the arms were already in
    the pose.

    Every attribute of the client is looked up on each call, so a test that replaces
    one on the client applies here too.
    """

    def __init__(
        self,
        client: CommanderClient,
        cancel: CancelToken,
        *,
        sim: bool,
        min_move_timeout_s: float = 0.0,
    ) -> None:
        number = isinstance(min_move_timeout_s, int | float) and not isinstance(
            min_move_timeout_s, bool
        )
        if not (number and math.isfinite(min_move_timeout_s) and min_move_timeout_s >= 0):
            raise ValueError(
                "min_move_timeout_s must be zero or a positive number of seconds, not "
                f"{min_move_timeout_s!r}."
            )
        self._client = client
        self._cancel = cancel
        self._sim = sim
        self._min_move_timeout_s = min_move_timeout_s
        # The commander's allow_gripper_vs_octomap as this object last left it: False,
        # the commander's default, at the start, and None after a set that failed.
        self._touch: bool | None = False

    # -- The Arms verbs ----------------------------------------------------------

    def poses(self) -> Mapping[str, ArmPose]:
        """The poses the LLM may use here, in file order: outside sim, only those tried."""
        return self._client.poses.offered_poses(sim=self._sim)

    def state(self, *, read: bool | None = None) -> ArmsReport:
        """The client's snapshot, after a read of the arms when `read` asks for one. Never raises.

        From the client's memory, which every read and every move keeps up to date, so
        most answers cost no call. With `read` None the arms are read once when that
        memory cannot say whether the robot may drive, as before the first read or
        after a move that did not end with a read; with True always, as after a failed
        move, which may have stopped the arms part way; with False never, as when the
        commander is not answering. A read is made only while the commander is
        discovered. A read that fails leaves the memory: the answer is about the call,
        and the read is only for its last sentence. The report comes from one
        snapshot, taken after any read.
        """
        client = self._client
        if (
            read is not False
            and (read or client.last_state.travel_safe is None)
            and client.available()
        ):
            try:
                client.read_arms()
            except StepError:
                pass
        return arms_report(client.snapshot())

    def read(self) -> ArmsReading:
        """Read both arms now, then each gripper's finger; see capabilities.Arms.

        The arms are read by the client's read_arms(), which recovers first while they
        are unverified, and its StepError is raised as it is. A gripper that cannot be
        read, also one whose reading lacks its finger joint, is a GripperReading with
        `read_error` and the state of the last position sent to it, unless a stop
        ended the read: that CANCELLED is raised as it is.
        """
        client = self._client
        before = arms_report(client.snapshot())
        client.read_arms()
        grippers = tuple(self._read_gripper(arm) for arm in ARMS)
        return ArmsReading(before=before, after=arms_report(client.snapshot()), grippers=grippers)

    def move_to(
        self,
        pose: str,
        arm: Arm = "both",
        *,
        lift_m: float | None = None,
        touch: bool = False,
    ) -> ArmsMoved:
        """Move `arm` to the offered pose `pose`, wait, and check the joints against it.

        Raises StepError as capabilities.Arms says. Every message ends with the arm
        sentence: from the client's memory when the commander did not answer or a stop
        came, and from a fresh read after a failure that the commander reported, which
        may have stopped the arms part way. With `lift_m`, the goal goes to the arm's
        *_lift group, which also moves the lift to `lift_m`, and the check reads that
        group; one arm only. Before the goal, the commander's finger allowance is set
        to `touch`.
        """
        if lift_m is not None and arm == BOTH:
            raise ValueError("A move with the lift takes one arm.")
        client = self._client
        found = answers.offered_pose(self.poses(), pose, arm, lambda: self.state().sentence())
        name = found.name
        arms = ARMS if arm == BOTH else (arm,)
        targets = {joint: value for side in arms for joint, value in found.tables[side].items()}
        if lift_m is None:
            group = BOTH_ARMS_GROUP if arm == BOTH else ARM_GROUPS[arm]
        else:
            group = LIFT_ARM_GROUPS[arm]
            targets[LIFT_JOINT] = lift_m

        try:
            order = client.joint_names(group)
        except StepError as e:
            if self._stopped(e):
                raise StepError(
                    ErrorCode.CANCELLED,
                    f"Stopped before the arms moved: {_STOP_IN_EFFECT}. "
                    f"{self._memory_sentence()}",
                ) from None
            # The read failed, so a second read for the sentence would too.
            raise answers.with_sentence(
                e, f"Could not move to {name!r}: ", self._memory_sentence()
            ) from None
        positions = self._positions(found, arm, targets, order)

        timeout_s = max(found.timeout_s, self._min_move_timeout_s)
        try:
            # A joint move keeps the fingers off the octomap unless asked. With the
            # allowance a pick left on, MoveIt routed them through the table in the
            # Chula simulation. Right after a pick's lift, the octomap still has voxels
            # at the hand, so the move home that ends the pick keeps it.
            self._set_touch(touch)
            status = client.set_joint_position(group, positions, timeout_s)
        except StepError as e:
            raise self._move_error(f"Could not move to {name!r}: ", e) from None

        try:
            reading = client.joint_states(BOTH_ARMS_GROUP if lift_m is None else group)
        except StepError as e:
            if self._stopped(e):
                raise StepError(
                    ErrorCode.CANCELLED,
                    f"Stopped before the arms were checked against {name!r}: "
                    f"{_STOP_IN_EFFECT}. {self._memory_sentence()}",
                ) from None
            # A move that cannot be checked counts as failed (ADR-0024).
            raise answers.unchecked(
                status,
                f"the arms could not be checked against {name!r}: "
                f"{answers.full_stop(e.message)}",
                self._memory_sentence(),
            ) from None

        # The report from one snapshot, so that a goal from an earlier call that the
        # commander accepts late cannot come between the sentence and travel_safe.
        report = arms_report(client.snapshot())
        sentence = report.sentence()
        worst = worst_joint(targets, reading)
        if worst is None:
            missing = unread_joints(targets, reading)
            raise answers.unchecked(
                status,
                f"its reading lacks a usable position for {', '.join(missing)}, so the arms "
                f"cannot be checked against {name!r}.",
                sentence,
            )
        joint, error = worst
        if error > MATCH_TOLERANCE_RAD:
            raise answers.unchecked(status, f"{joint} is {error:.2f} rad from {name!r}.", sentence)
        return ArmsMoved(pose=name, arm=arm, status=status, max_error_rad=error, report=report)

    def gripper(self, arm: Gripper, state: GripperState) -> GripperMoved:
        """Move one gripper to `state`, and after a close judge whether it holds something.

        An open or "half" is "moved" when the commander reports success, else
        "not_reached". A close is judged otherwise, since a real grasp usually ends with
        success false, because the fingers stall on the object. Only a goal that did
        not end as the commander allows raises, as capabilities.Arms says, with the arm
        sentence from the client's memory; an outcome that leaves whether the gripper
        holds something unknown is returned.
        """
        client = self._client
        grippers = client.poses.grippers
        group = GRIPPER_GROUPS[arm]
        try:
            result = client.control_gripper(group, grippers.position(state))
        except StepError as e:
            if self._stopped(e):
                stopped = (
                    interrupted(e, gripper=True)
                    or f"Stopped before the {arm} gripper moved: {_STOP_IN_EFFECT}."
                )
                raise StepError(
                    ErrorCode.CANCELLED, f"{stopped} {self._memory_sentence()}"
                ) from None
            raise answers.with_sentence(
                e, f"Could not move the {arm} gripper: ", self._memory_sentence()
            ) from None
        if state != "close":
            if state == "open":
                self._release_touch()
            outcome = "moved" if result.success else "not_reached"
            return GripperMoved(arm=arm, state=state, outcome=outcome, result=result)
        # In each case where whether the close holds something is unknown, the close
        # itself ended as the commander allows, so only the answer about the object is
        # missing.
        if _unmeasured_failure(result):
            return GripperMoved(arm=arm, state=state, outcome="unmeasured", result=result)
        try:
            finger = _finger(client.joint_states_since(group, result.command), arm)
            # Raises CANCELLED only, when it reads the commander's band.
            held = self._holding(result, finger, grippers)
        except StepError as e:
            if self._stopped(e):
                raise StepError(
                    ErrorCode.CANCELLED,
                    f"Stopped before checking whether the {arm} gripper holds something: "
                    f"{_STOP_IN_EFFECT}. {self._memory_sentence()}",
                ) from None
            if _superseded(e, group):
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

    def reach(
        self,
        arm: Gripper,
        pose: HandPose,
        *,
        whole_body: bool = False,
        cartesian: bool = False,
        touch: bool = False,
        timeout_s: float = 60.0,
    ) -> str:
        """Move `arm`'s hand link to `pose` through the commander's /go_to_pose.

        See capabilities.Arms. The goal goes to the arm's whole-body group with
        `whole_body`, to the arm with the lift with `cartesian`, else to its arm
        group, and waits for the longer of `timeout_s` and min_move_timeout_s.
        Before the goal, the commander's finger allowance is set to `touch` when it
        may differ (CommanderClient.set_gripper_touch), and a straight reach waits
        for the arm to come to rest (_wait_still). A reported success is the
        commander's own check: a pose goal has no joint table to compare a reading
        with. Its plan-only status means nothing moved, so it is a failure here.
        Every message ends with the arm sentence, from a fresh read after a failure
        the commander reported, as move_to()'s does.
        """
        answers.check_reach(pose, whole_body=whole_body, cartesian=cartesian, timeout_s=timeout_s)
        if whole_body:
            group = WHOLE_BODY_GROUPS[arm]
        else:
            group = (LIFT_ARM_GROUPS if cartesian else ARM_GROUPS)[arm]
        try:
            self._set_touch(touch)
            if cartesian:
                self._wait_still(arm)
            status = self._client.go_to_pose(
                group, pose, max(timeout_s, self._min_move_timeout_s), cartesian=cartesian
            )
        except StepError as e:
            raise self._move_error(f"Could not move the {arm} hand: ", e) from None
        if status.startswith(answers.PLAN_ONLY):
            raise StepError(
                ErrorCode.SKILL_FAILED,
                f"{answers.PLAN_ONLY_TEXT}. {answers.PLAN_ONLY_NEXT} {self._memory_sentence()}",
            )
        return status

    def pick(self, object_id: str, *, timeout_s: float = 60.0) -> str:
        """Raises StepError UNAVAILABLE: the robot has no grasp skill yet."""
        raise StepError(ErrorCode.UNAVAILABLE, NO_GRASP_SKILL)

    def place_on(self, place: str, *, timeout_s: float = 60.0) -> str:
        """Raises StepError UNAVAILABLE: the robot has no grasp skill yet."""
        raise StepError(ErrorCode.UNAVAILABLE, NO_GRASP_SKILL)

    def hand_over(self, person_id: str, *, timeout_s: float = 60.0) -> str:
        """Raises StepError UNAVAILABLE: the robot has no grasp skill yet."""
        raise StepError(ErrorCode.UNAVAILABLE, NO_GRASP_SKILL)

    # -- The drive check ---------------------------------------------------------

    def check_for_driving(self, what: str, timeout_s: float) -> None:
        """The arm check before `what`, a base motion (ADR-0025): returns to let it go.

        For RosBase's arms_check. The read is bounded by the client's read_timeout_s,
        and by `timeout_s`, the time the base verb has left. Raises StepError to refuse,
        and fails open, as answers.drive_verdict() says.
        """
        client = self._client
        answers.drive_verdict(
            lambda: client.check_for_driving(min(client.read_timeout_s, timeout_s)), what
        )

    # -- Helpers -----------------------------------------------------------------

    def _stopped(self, error: StepError) -> bool:
        """True when `error` is the CANCELLED of a stop or a reset, not the commander's own."""
        return error.code is ErrorCode.CANCELLED and self._cancel.cancelled

    def _memory_sentence(self) -> str:
        """The arm sentence from the client's memory, with no call to the commander."""
        return self.state(read=False).sentence()

    def _read_gripper(self, arm: Gripper) -> GripperReading:
        """One gripper for read(): its finger read now, or why not and its last command."""
        client = self._client
        group = GRIPPER_GROUPS[arm]
        positions = client.poses.grippers
        try:
            finger = _finger(client.joint_states(group), arm)
        except StepError as e:
            if self._stopped(e):
                raise
            commanded = answers.commanded_state(positions, client.last_gripper_position(group))
            return GripperReading(arm, None, None, read_error=e.message, commanded=commanded)
        return GripperReading(arm, finger, answers.reads_as(positions, finger))

    def _positions(
        self, pose: ArmPose, arm: Arm, targets: dict[str, float], order: tuple[str, ...]
    ) -> list[float]:
        """The pose's values in the commander's joint order for the group.

        walkie-ros launched with right_joint2_fixed lists no openarm_right_joint2 and
        holds it at 0, so the joint is dropped when the pose has it at 0; a pose that
        needs it elsewhere is REFUSED. Any other difference means
        config/robot_model.toml no longer matches the running walkie-ros, which the
        snapshot test should have caught: INTERNAL.
        """
        missing = [joint for joint in targets if joint not in order]
        extra = [joint for joint in order if joint not in targets]
        if missing == [RIGHT_JOINT2] and not extra:
            value = targets[RIGHT_JOINT2]
            if value != 0.0:
                raise StepError(
                    ErrorCode.REFUSED,
                    f"{pose.name!r} cannot move the right arm while walkie-ros runs with "
                    f"right_joint2_fixed: the pose needs {RIGHT_JOINT2} at {value:.2f} rad, "
                    "and that joint is held at 0. Use the left arm or another pose. "
                    f"{self.state().sentence()}",
                )
        elif missing or extra:
            raise StepError(
                ErrorCode.INTERNAL,
                f"The commander's joints for {arm} do not match config/robot_model.toml "
                f"(missing: {', '.join(missing) or 'none'}; unknown: "
                f"{', '.join(extra) or 'none'}). Regenerate the snapshot for the running "
                f"walkie-ros. {self.state().sentence()}",
            )
        return [targets[joint] for joint in order]

    def _set_touch(self, touch: bool) -> None:
        """Set the commander's finger allowance to `touch`, unless it was last set so.

        Raises the client's StepError, after which the allowance is unknown.
        """
        if self._touch is touch:
            return
        self._touch = None
        self._client.set_gripper_touch(touch)
        self._touch = touch

    def _wait_still(self, arm: Gripper) -> None:
        """Wait until two reads of `arm`'s joints STILL_POLL_S apart agree.

        After STILL_WAIT_S the reach goes ahead anyway, with a warning: MoveIt still
        refuses a path whose start is off. Raises StepError CANCELLED on a stop, and
        the client's StepError when a read fails.
        """
        group = ARM_GROUPS[arm]
        readings: list[Mapping[str, float]] = []
        failed: list[StepError] = []

        def still() -> bool:
            try:
                readings.append(self._client.joint_states(group))
            except StepError as e:
                failed.append(e)
                return True
            if len(readings) < 2:
                return False
            before, now = readings[-2], readings[-1]
            moved = max(
                (abs(value - before[name]) for name, value in now.items() if name in before),
                default=0.0,
            )
            return moved <= STILL_TOLERANCE_RAD

        try:
            self._cancel.wait_for(
                still, STILL_WAIT_S, f"the rest of the {arm} arm", poll_s=STILL_POLL_S
            )
        except StepError as e:
            if e.code is not ErrorCode.TIMEOUT or failed:
                raise
            logger.warning(
                "The %s arm still moved after %g s; the straight reach goes ahead.",
                arm,
                STILL_WAIT_S,
            )
        if failed:
            raise failed[0]
        if len(readings) > 2:
            logger.info(
                "The %s arm came to rest after %d reads %g s apart.",
                arm,
                len(readings),
                STILL_POLL_S,
            )

    def _release_touch(self) -> None:
        """After an open: the fingers hold nothing, so they may no longer touch the octomap.

        Only when this object may have set it: the commander starts with it False, so
        an open costs no call otherwise. The open has already happened, so a failed
        reset is logged, not raised, and the next reach or joint move sets the
        allowance it needs.
        """
        if self._touch is False:
            return
        try:
            self._set_touch(False)
        except StepError as e:
            logger.warning("Could not reset the gripper octomap allowance: %s", e.message)

    def _move_error(self, prefix: str, error: StepError) -> StepError:
        """The answer to a move the commander did not finish as asked, `prefix` first."""
        if self._stopped(error):
            # The call ends at once, so the arms are not read.
            stopped = (
                interrupted(error, gripper=False)
                or f"Stopped before the arms moved: {_STOP_IN_EFFECT}."
            )
            return StepError(ErrorCode.CANCELLED, f"{stopped} {self._memory_sentence()}")
        if error.code in (ErrorCode.SKILL_FAILED, ErrorCode.INTERNAL, ErrorCode.CANCELLED):
            # The commander answered, so it can be read: a failed plan left the arms
            # where they were, and a failed execution may have stopped them part way.
            sentence = self.state(read=True).sentence()
        else:
            # TIMEOUT, UNAVAILABLE or PRECONDITION_FAILED: the commander is not
            # answering or is busy, so a read would fail or wait out its timeout.
            sentence = self._memory_sentence()
        return answers.with_sentence(error, prefix, sentence)

    def _holding(
        self, result: GripperResult, finger: float, grippers: GripperPositions
    ) -> bool | None:
        """Whether a close holds something, by `finger`, the travel read after it.

        Only for a close that _unmeasured_failure() does not exclude, read while it was
        still the gripper's latest command. None when the gripper did not close: fingers
        within HOLDING_MARGIN_M of the open position barely moved, which no object
        explains, even when the travel lies inside the detection band. Otherwise the
        commander's `grasped` is the answer when its gap agrees with the read. A gap
        that does not agree is one the commander did not measure on a close that
        reached its position, one it measured before the fingers settled, or the
        fingers moved since; the read is then judged by the commander's detection band,
        read from it (_band). Raises StepError CANCELLED only.
        """
        if finger >= grippers.open_position - HOLDING_MARGIN_M:
            return None
        gap = result.gripper_gap
        if gap is not None and abs(gap - finger) <= GAP_AGREEMENT_M:
            return bool(result.grasped)
        low, high = self._band()
        return low <= finger <= high

    def _band(self) -> tuple[float, float]:
        """The commander's grasp detection band now, or its default when that cannot be read.

        The default is the one the commander starts with, so a read that fails judges
        as the commander would unless its band was changed. Raises StepError CANCELLED
        only.
        """
        try:
            return self._client.grasp_band()
        except StepError as e:
            if e.code is ErrorCode.CANCELLED:
                raise
            logger.warning(
                "Judging a close by the commander's default grasp detection band: %s",
                e.message,
            )
            return GRASP_DETECT_MIN, GRASP_DETECT_MAX


def _finger(reading: Mapping[str, float], arm: Gripper) -> float:
    """The finger joint of `arm`'s gripper in `reading`, or StepError INTERNAL when it lacks it."""
    joint = FINGER_JOINTS[arm]
    if joint not in reading:
        raise StepError(
            ErrorCode.INTERNAL,
            f"The commander's reading of {GRIPPER_GROUPS[arm]} lacks {joint}.",
        )
    return reading[joint]


def _superseded(error: StepError, group: str) -> bool:
    """True for the client's refusal of a read after a newer command for `group` came.

    The PRECONDITION_FAILED of CommanderClient.joint_states_since(), by the start of
    the message its _check_command() writes.
    """
    return error.code is ErrorCode.PRECONDITION_FAILED and error.message.startswith(
        f"Another {group} command came after command "
    )


def _unmeasured_failure(result: GripperResult) -> bool:
    """True for a close that the commander reports failed with a gap of 0.0 and grasped false.

    The commander reports that for a close that a newer command for the group
    superseded, also one from another node, and for a failed close that it did not
    measure, as with grasp_detect_enable off. Which of these happened cannot be told,
    and after a supersession the fingers show the newer command, so such a close is
    not judged from a read.
    """
    return not result.success and result.gripper_gap == 0.0 and result.grasped is False
