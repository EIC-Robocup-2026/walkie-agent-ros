"""pick_object: pick up a remembered object with one hand, by its name (decision PK-4).

The atomic finds the object in the object memory, drives to a stand-off pose facing
it, raises the other arm out of the way, and reaches through ctx.arms: a whole-body
reach to a pre-grasp pose, a straight approach, a close of the gripper judged by the
commander's grasp band, a straight lift, a straight move to beside the base, and a move
of both arms home, the picking arm with the lift. The gripper is then read again and
judged by the same band, so an object that fell out of the hand after the close is
reported lost, not picked. The geometry and its numbers are in walkie_agent.arm.pick
and config/pick.toml.
The LLM gives only the object's name and reads only the outcome: no position, pose or
joint value reaches it (ADR-0022). Every answer after the drive ends with the arm
sentence (ADR-0024).

The object is read from the memory again at three checkpoints, and each refusal sends
no further goal: after the drive, before the first arm goal; before the whole-body
reach, once the other arm is out of the way and the hand is open (checkpoint A); and
right before the straight approach's goal, with the hand at the pre-grasp pose, the
allowance on and the wait for the arm to rest over (checkpoint B).
Each time the memory must hold the object as present, with a full position, seen at
most max_observation_age_s ago, and the atomic waits up to observation_wait_s for such
a sighting. That position must lie within max_target_shift_m of the previous
checkpoint's, or of the one the drive aimed at. After the drive and at A, the object
and every hand pose must also lie in the reach envelope around where the base stands
(arm.pick.reach_problems); A plans the hand poses afresh from its own reading. A
motion is never cut short for a newer reading: the checks run between motions. Once
the first arm goal may go, the fingers' octomap allowance (allow_gripper_vs_octomap)
is turned off again however the pick ends, a stop included (_grasp_then_touch_off).
That reset sends no motion, so a stop still ends every motion.

The stand-off pose is reached with ctx.base.move_relative, a Nav2 goal like any other
drive, after a go_to to the named place nearest the object when the object is further
than one relative move allows. walkie-ros's /navigate_to_object computes a stand-off
of its own, but it is in walkie-ros's navigation submodule, whose interfaces the agent's
workspace does not build, and its pose would not match the hand's approach here.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

from walkie_agent.arm import answers
from walkie_agent.arm.pick import (
    HandPose,
    PickPlan,
    PickSettings,
    approach_headings,
    heading_from,
    load_pick_settings,
    plan_pick,
    reach_problems,
    relative_move,
)
from walkie_agent.capabilities import Ctx, Pose2D, RecalledObject
from walkie_agent.capabilities.guards import MAX_MOVE_M
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.tools.atomic.navigation import APPROACH_PLACE_MAX_M, approach_place

logger = logging.getLogger(__name__)

# How close the base must already be to a stand-off pose to skip the drive to it, in
# metres and degrees: a relative move of nothing is refused, and Nav2 would not
# settle closer than its own goal tolerances.
AT_STANDOFF_M = 0.05
AT_STANDOFF_DEG = 3.0
# How close a drive that Nav2 failed must have left the base to its stand-off pose for
# the pick to go on from there, in metres and degrees. Nav2 aborts a goal it cannot
# settle on ("Failed to make progress") although the base stopped a few centimetres
# off, as it did beside the Chula bar. The hand then moves in along the line from
# where the base stopped (_plan_from_here), and the whole-body reach covers the rest.
NEAR_STANDOFF_M = 0.10
NEAR_STANDOFF_DEG = 10.0
# How many objects of the name the memory is asked for. The first with a position is
# picked, in the memory's order, which puts the closest matches of the name first.
SEARCH_LIMIT = 10
# Where the arm that does not pick goes once the base stands at the stand-off pose,
# before the reach. The straight moves of the arm with the lift can lower the lift by
# over 0.1 m, and the other arm hangs from the same column: at home its fingers then
# touch the base, and MoveIt cuts the approach short (walkie-ros sim_bottle_pick.py
# tucks the left arm the same way).
OTHER_ARM_POSE = "standby"
_OTHER_ARM = {"left": "right", "right": "left"}
# Where both arms end a pick, held object or not: the one offered pose that is safe
# for driving with an object in the hand (walkie-agent-v2 carried objects in it).
HOME_POSE = "home"
# How often the memory is read while the atomic waits for a new sighting of the
# object at a checkpoint, in seconds. The memory saves a sighting about every 2 s.
SIGHTING_POLL_S = 0.5


@dataclass(frozen=True)
class _Checkpoint:
    """One moment at which pick_object reads the object again, and how its refusals say it.

    `name` goes to the log. A refusal says that, for its reason, `refused`; that the
    object moved `moving`; and that the memory could not be read `unreadable`, in
    which "{label}" stands for the object. With `arms_out`, the arms have left their
    driving pose by then, so every refusal first asks for them to come home.
    """

    name: str
    refused: str
    moving: str
    unreadable: str
    arms_out: bool

    def advice(self, step: str | None) -> str:
        """The next step the LLM is asked to take, `step` starting in lower case."""
        if self.arms_out:
            home = "Bring both arms home with move_arm_to_pose"
            return f"{home}, then {step}." if step else f"{home}."
        return f"{step[0].upper()}{step[1:]}." if step else ""


_AFTER_DRIVE = _Checkpoint(
    name="after the drive",
    refused="it did not reach for it",
    moving="while Walkie drove up to it",
    unreadable="after driving up to {label}",
    arms_out=False,
)
_BEFORE_REACH = _Checkpoint(
    name="before the reach",
    refused="it did not reach for it",
    moving="while Walkie readied its arms",
    unreadable="before reaching for {label}",
    arms_out=True,
)
_BEFORE_APPROACH = _Checkpoint(
    name="before the approach",
    refused="it did not move the hand in to grasp it",
    moving="while Walkie reached toward it",
    unreadable="before moving the hand in to {label}",
    arms_out=True,
)

# What the answer says when a close did not show that the hand holds the object, by
# GripperMoved.outcome.
_NOT_HELD = {
    "empty": "the {arm} hand closed on nothing",
    "not_closed": "the {arm} gripper did not close",
    "unmeasured": "the commander could not tell whether the {arm} hand holds it",
    "superseded": "another command for the {arm} gripper came after the close",
    "unread": "the {arm} gripper could not be read after the close",
}


def pick_object(ctx: Ctx, name: str) -> str:
    """Pick up an object you remember seeing, by its name, and lift it a little.

    Walkie finds the object in its memory, drives up to it, raises the left arm out
    of the way, reaches for it with the right hand, closes the hand, lifts it, and
    brings both arms home, where it may drive with the object. It takes a minute or
    more. Tell the people near you before you call it. Returns a sentence that says
    whether the right hand holds the object, which ends with whether the arms are
    safe for driving.

    Args:
        name: What the object is called, such as "water bottle" or "cup".
    """
    try:
        settings = load_pick_settings()
    except (OSError, ValueError) as error:
        logger.error("Cannot read the pick settings: %s", error)
        raise StepError(
            ErrorCode.INTERNAL, "Walkie's pick settings cannot be read, so it cannot pick."
        ) from None
    found = _remembered(ctx, name)
    label = f"the {found.name}"
    _drive_to_standoff(ctx, settings, found, label)
    found = _seen_again(ctx, settings, found, label, _AFTER_DRIVE)
    _plan_in_reach(ctx, settings, found, label, _AFTER_DRIVE)
    return _grasp_then_touch_off(ctx, settings, found, label)


def _remembered(ctx: Ctx, name: str) -> RecalledObject:
    """The first object of `name` that the memory holds as present with a full position."""
    asked = name.strip()
    if not asked:
        raise StepError(ErrorCode.REFUSED, "Refused: name the object to pick up.")
    matches = ctx.object_memory.search(asked, limit=SEARCH_LIMIT)
    if not matches:
        raise StepError(
            ErrorCode.NOT_FOUND,
            f"Walkie remembers no {asked}. Call recall_object to see what it remembers.",
        )
    for found in matches:
        if found.x is not None and found.y is not None and found.z is not None:
            return found
    raise StepError(
        ErrorCode.PRECONDITION_FAILED,
        f"Walkie remembers a {asked}, but not exactly where it is, so it cannot reach for "
        "it. Go to it and call find_objects to look again.",
    )


def _seen_again(
    ctx: Ctx,
    settings: PickSettings,
    found: RecalledObject,
    label: str,
    checkpoint: _Checkpoint,
) -> RecalledObject:
    """`found` as the memory holds it now, freshly seen, at `checkpoint`.

    `found` is the reading of the checkpoint before, or the one the drive aimed at.
    The memory keeps updating an object's position while the camera sees it, and
    from the stand-off pose it sees the object from close by. The position read
    before the drive can be old: in the Chula simulation it was where the hand had
    lifted the bottle in an earlier pick, 6 cm above where it stood again, and the
    hand closed on nothing. A motion of the arms can take a minute or more, and the
    object can be taken or moved meanwhile. So each next goal goes only to a sighting
    at most max_observation_age_s old, and the memory is read every SIGHTING_POLL_S
    for up to observation_wait_s until it holds one. Walkie has no head to turn
    toward the object, so it only waits. Every refusal comes before the
    checkpoint's next goal:

    - NOT_FOUND: the memory no longer holds the object.
    - UNAVAILABLE: the memory could not be read (any other memory error). That never
      means the object is gone, and the atomic may be called again.
    - PRECONDITION_FAILED: the memory holds it in a state other than present, such as
      held or missing, or without a full position; no fresh sighting came in time;
      or the sighting lies more than max_target_shift_m from `found`, when a new
      stand-off pose is needed.

    A stop is raised as CANCELLED.
    """
    latest: list[RecalledObject] = []

    def fresh() -> bool:
        now = _read_again(ctx, found, label, checkpoint)
        latest[:] = [now]
        return now.seen_s_ago is not None and now.seen_s_ago <= settings.max_observation_age_s

    try:
        ctx.cancel.wait_for(
            fresh, settings.observation_wait_s, f"a new sighting of {label}",
            poll_s=SIGHTING_POLL_S,
        )
    except StepError as error:
        if error.code is not ErrorCode.TIMEOUT:
            raise
        logger.info(
            "No sighting of %s at most %g s old within %g s %s; the last was %s s old.",
            found.id,
            settings.max_observation_age_s,
            settings.observation_wait_s,
            checkpoint.name,
            latest[0].seen_s_ago if latest else None,
        )
        raise StepError(
            ErrorCode.PRECONDITION_FAILED,
            f"Walkie stands by {label} but has not seen it in the last few seconds, so "
            f"{checkpoint.refused}: it may have moved or be out of view. "
            f"{checkpoint.advice('call find_objects to look again')} "
            f"{ctx.arms.state().sentence()}",
        ) from None
    now = latest[0]
    shift = math.dist((found.x, found.y, found.z), (now.x, now.y, now.z))
    logger.info("Object %s moved %.3f m in the memory %s.", found.id, shift, checkpoint.name)
    if shift > settings.max_target_shift_m:
        raise StepError(
            ErrorCode.PRECONDITION_FAILED,
            f"{label.capitalize()} moved {checkpoint.moving}, so {checkpoint.refused}. "
            f"{checkpoint.advice('call pick_object again to approach it where it is now')} "
            f"{ctx.arms.state().sentence()}",
        )
    return now


def _read_again(
    ctx: Ctx, found: RecalledObject, label: str, checkpoint: _Checkpoint
) -> RecalledObject:
    """One read of `found` from the memory, as _seen_again's refusals name it."""
    refused = checkpoint.refused
    try:
        now = ctx.object_memory.get(found.id)
    except StepError as error:
        if error.code is ErrorCode.CANCELLED:
            raise
        sentence = ctx.arms.state().sentence()
        if error.code is ErrorCode.NOT_FOUND:
            logger.info("The memory no longer holds %s %s: %s", found.id, checkpoint.name, error)
            raise StepError(
                ErrorCode.NOT_FOUND,
                f"Walkie's memory no longer holds {label}, so {refused}. "
                f"{checkpoint.advice('call find_objects to look for it again')} {sentence}",
            ) from None
        logger.warning("Could not read %s again %s: %s", found.id, checkpoint.name, error)
        raise StepError(
            ErrorCode.UNAVAILABLE,
            f"Walkie could not read its object memory {checkpoint.unreadable.format(label=label)}"
            f", so {refused}. {checkpoint.advice('call pick_object again')} {sentence}",
        ) from None
    if now.state != "present":
        logger.info("The memory holds %s as %s %s.", found.id, now.state, checkpoint.name)
        raise StepError(
            ErrorCode.PRECONDITION_FAILED,
            f"Walkie's memory now holds {label} as {now.state}, so {refused}. "
            f"{checkpoint.advice('call find_objects to look again')} "
            f"{ctx.arms.state().sentence()}",
        )
    if now.x is None or now.y is None or now.z is None:
        logger.info("The memory holds no full position for %s %s.", found.id, checkpoint.name)
        raise StepError(
            ErrorCode.PRECONDITION_FAILED,
            f"Walkie's memory no longer holds exactly where {label} is, so {refused}. "
            f"{checkpoint.advice('call find_objects to look again')} "
            f"{ctx.arms.state().sentence()}",
        )
    return now


def _drive_to_standoff(
    ctx: Ctx, settings: PickSettings, found: RecalledObject, label: str
) -> None:
    """Drive to a stand-off pose facing `found`, trying each approach heading in turn.

    A first go_to to the named place nearest the object comes when the first
    stand-off is further than one relative move allows. The headings are taken
    from where the robot stands then. A heading whose drive Nav2 fails
    (SKILL_FAILED) gives way to the next, unless the base stopped near its
    stand-off pose (_near); any other error is raised as it is.
    """
    start = ctx.base.pose()
    plans = _plans(settings, found, start)
    if _distance(start, plans[0]) > MAX_MOVE_M:
        place = approach_place(ctx, found)
        if place is None:
            raise StepError(
                ErrorCode.PRECONDITION_FAILED,
                f"{label.capitalize()} is far away, and Walkie knows no named place within "
                f"{APPROACH_PLACE_MAX_M:g} m of it to drive to first.",
            )
        try:
            ctx.base.go_to(place.name)
        except StepError as error:
            # The drive to the place only has to bring the stand-off within one
            # relative move. Nav2 aborted the drive to the Chula bar 0.1 m from its goal.
            if error.code is not ErrorCode.SKILL_FAILED:
                raise
            start = ctx.base.pose()
            if _distance(start, _plans(settings, found, start)[0]) > MAX_MOVE_M:
                raise
            logger.info("The drive to %s failed close enough to go on: %s", place.name, error)
        start = ctx.base.pose()
        plans = _plans(settings, found, start)
    failures: list[str] = []
    for plan in plans:
        here = ctx.base.pose()
        if _distance(here, plan) > MAX_MOVE_M:
            failures.append("a spot too far for one move")
            continue
        forward, left, turn = relative_move(
            here.x, here.y, here.yaw, plan.standoff_x, plan.standoff_y, plan.heading
        )
        if math.hypot(forward, left) <= AT_STANDOFF_M and abs(turn) <= AT_STANDOFF_DEG:
            return
        try:
            ctx.base.move_relative(forward, left, turn)
        except StepError as error:
            if error.code is not ErrorCode.SKILL_FAILED:
                raise
            if _near(ctx.base.pose(), plan):
                logger.info(
                    "The drive to %s at heading %.2f failed close to its goal; the pick "
                    "goes on from there: %s",
                    label,
                    plan.heading,
                    error,
                )
                return
            logger.info("The approach to %s at heading %.2f failed: %s", label, plan.heading, error)
            failures.append(error.message.rstrip("."))
            continue
        return
    raise StepError(
        ErrorCode.SKILL_FAILED,
        f"Walkie could not reach a spot to pick up {label} from, after {len(plans)} tries. "
        f"The last one: {failures[-1]}.",
    )


def _plans(settings: PickSettings, found: RecalledObject, start: Pose2D) -> list[PickPlan]:
    """One plan per approach heading, from the robot's pose `start`."""
    headings = approach_headings(settings, start.x, start.y, found.x, found.y)
    return [plan_pick(settings, found.x, found.y, found.z, heading) for heading in headings]


def _plan_from_here(settings: PickSettings, found: RecalledObject, here: Pose2D) -> PickPlan:
    """The plan for the heading the robot now faces the object along (arm.pick.heading_from).

    The hand approaches along the line from where the base stopped, `here`, to the
    object, so a drive that ended a little off its goal still reaches straight at it.
    """
    heading = heading_from(settings, here.x, here.y, found.x, found.y)
    approach = math.atan2(found.y - here.y, found.x - here.x)
    return plan_pick(
        settings, found.x, found.y, found.z, heading, approach, (here.x, here.y, here.yaw)
    )


def _plan_in_reach(
    ctx: Ctx,
    settings: PickSettings,
    found: RecalledObject,
    label: str,
    checkpoint: _Checkpoint,
) -> PickPlan:
    """The plan from where the base stands now, refused outside the reach envelope.

    Before the checkpoint's next arm goal (arm.pick.reach_problems). The numbers go
    to the log only. An object too low or too high stays so from any stand-off pose;
    any other problem may pass from a new one.
    """
    here = ctx.base.pose()
    plan = _plan_from_here(settings, found, here)
    problems = reach_problems(
        settings, plan, (found.x, found.y, found.z), (here.x, here.y, here.yaw)
    )
    if not problems:
        return plan
    logger.warning(
        "Refused to reach for %s (object %s) %s: %s.",
        label,
        found.id,
        checkpoint.name,
        "; ".join(problems),
    )
    sentence = ctx.arms.state().sentence()
    if not settings.min_target_z_m <= found.z <= settings.max_target_z_m:
        side = "low" if found.z < settings.min_target_z_m else "high"
        advice = checkpoint.advice(None)
        raise StepError(
            ErrorCode.PRECONDITION_FAILED,
            f"{label.capitalize()} is too {side} for the {settings.arm} hand to pick up, so "
            f"Walkie did not reach for it. {f'{advice} ' if advice else ''}{sentence}",
        )
    raise StepError(
        ErrorCode.PRECONDITION_FAILED,
        f"{label.capitalize()} is out of the {settings.arm} hand's reach from where Walkie "
        f"stopped, so it did not reach for it. "
        f"{checkpoint.advice('call pick_object again to approach it afresh')} {sentence}",
    )


def _grasp_then_touch_off(
    ctx: Ctx, settings: PickSettings, found: RecalledObject, label: str
) -> str:
    """_grasp, then the fingers' octomap allowance turned off, however _grasp ended.

    The grasp turns the allowance on for the approach, the close, the lift and the
    move clear, and its last move home turns it off. But a failed step, a stop, or
    any other error ends the pick with the allowance on, and the commander keeps it
    for every motion after, of any tool. So ctx.arms.restore_touch() runs on every
    path out: after an answer, a StepError (CANCELLED from a stop included) or any
    other exception. It sends no motion and goes through a stop.

    If the reset fails, the pick fails too: the reset's error leads the message,
    which still ends with the arm sentence, under the original error's code and
    class, or, after an answer, under the reset's own code. Another exception
    keeps its type and gets the failure as a note. ctx.arms then knows the
    allowance's state as unknown, so its next motion first turns the allowance to
    what that motion needs and is refused if the commander does not confirm it.
    """
    error: BaseException | None = None
    answer = ""
    try:
        answer = _grasp(ctx, settings, found, label)
    except BaseException as raised:
        error = raised
        raise
    finally:
        _touch_off(ctx, label, error, answer)
    return answer


def _touch_off(ctx: Ctx, label: str, error: BaseException | None, answer: str) -> None:
    """Turn the allowance off after the pick; raise if that fails (_grasp_then_touch_off)."""
    try:
        ctx.arms.restore_touch()
        return
    except Exception as failure:  # Any failure: the pick's own outcome must survive it.
        logger.error(
            "Could not turn off allow_gripper_vs_octomap after the pick of %s: %r", label, failure
        )
        if isinstance(failure, StepError):
            code, why = failure.code, failure.message.rstrip(".")
        else:
            code, why = ErrorCode.INTERNAL, "the reset failed"
    note = (
        "Walkie could not turn off its fingers' allowance to touch what its cameras sense: "
        f"{why}. Its next motion of an arm or a gripper sets the allowance again first, and "
        "moves only if that works."
    )
    if error is None:
        raise StepError(code, f"{note} {answer}")
    if isinstance(error, StepError):
        raise type(error)(error.code, f"{note} {error.message}") from None
    error.add_note(note)


def _grasp(ctx: Ctx, settings: PickSettings, found: RecalledObject, label: str) -> str:
    """Tuck the other arm, open the hand, reach, approach, close, lift and go home.

    `found` is the object as the memory held it after the drive. It is read again
    before the reach (checkpoint A), which plans the hand poses from that reading,
    and right before the approach's goal (checkpoint B), once the allowance is on and
    the wait for the arm to rest is over, which keeps A's plan: a straight move from
    the pre-grasp pose must
    keep the hand's orientation. A refusal at either
    sends no further goal, and the arms stay where they are. After a close that holds
    nothing, both arms still go home. After a close that holds the object, the hand
    is read again once both arms are home (_still_holds).

    Returns the answer to the LLM.
    """
    arm = settings.arm
    other = _OTHER_ARM[arm]
    try:
        ctx.arms.move_to(OTHER_ARM_POSE, other)
    except StepError as error:
        raise type(error)(
            error.code,
            f"Walkie did not reach for {label}: it could not move the {other} arm out of the "
            f"way first. {error.message}",
        ) from None
    opened = ctx.arms.gripper(arm, "open")
    if opened.outcome != "moved":
        raise StepError(
            ErrorCode.SKILL_FAILED,
            f"The {arm} gripper did not open, so Walkie did not reach for {label}. "
            f"{ctx.arms.state().sentence()}",
        )
    # The other arm's move can take half a minute, and the object can be taken or
    # moved meanwhile.
    seen = _seen_again(ctx, settings, found, label, _BEFORE_REACH)
    plan = _plan_in_reach(ctx, settings, seen, label, _BEFORE_REACH)
    _reach(ctx, settings, plan.pregrasp, f"The reach toward {label} failed: ", whole_body=True)
    # The fingers close on the object's octomap voxels, so from here on they may touch
    # the octomap. The allowance stays until the move of the other arm home (_go_home).
    # The whole-body reach can take a minute or more, and the arm then gets up to 10 s
    # to come to rest. The memory kept saving sightings of the Chula bottle
    # throughout, so a fresh one is required after that wait, right before the
    # approach's goal.
    _reach(
        ctx, settings, plan.grasp, f"The approach to {label} failed: ", cartesian=True,
        touch=True, ready=lambda: _seen_again(ctx, settings, seen, label, _BEFORE_APPROACH),
    )
    closed = ctx.arms.gripper(arm, "close")
    if closed.outcome != "holding":
        why = _NOT_HELD.get(closed.outcome, "the close did not hold it").format(arm=arm)
        missed = f"Walkie reached {label}, but {why}, so it did not pick it up"
        _go_home(ctx, settings, f"{missed}, and the move of the arms home failed: ")
        raise StepError(
            ErrorCode.SKILL_FAILED,
            f"{missed}. It brought both arms home. The object may have moved: call "
            f"find_objects to look again. {ctx.arms.state().sentence()}",
        )
    _reach(
        ctx, settings, plan.lift, f"The {arm} hand holds {label}, but the lift failed: ",
        cartesian=True,
        touch=True,
    )
    # The move home swings the hand down toward the base. Started over the surface,
    # it dragged the Chula bottle across the table's corner and out of the fingers,
    # so the hand first moves straight to beside the base at the lifted height.
    _reach(
        ctx, settings, plan.clear,
        f"The {arm} hand holds {label}, but the move clear of the surface failed: ",
        cartesian=True,
        touch=True,
    )
    _go_home(ctx, settings, f"The {arm} hand holds {label}, but the move of the arms home failed: ")
    _still_holds(ctx, arm, label)
    return (
        f"Picked up {label} with the {arm} hand, lifted it and brought both arms home. "
        f"{ctx.arms.state().sentence()}"
    )


def _still_holds(ctx: Ctx, arm: str, label: str) -> None:
    """Raise unless the `arm` hand still holds the object once both arms are home.

    The close was judged before the lift, and the lift, the move clear and the move
    home can each shake the object out of the fingers. So the gripper is read again
    through ctx.arms.read(), whose GripperReading.in_grasp_band applies the band the
    close was judged by. A hand found empty is SKILL_FAILED: the object was lost
    after the lift. A read that fails keeps its code and class, and a gripper that
    could not be read is SKILL_FAILED; both say that the object may be lost. A stop
    during the read is raised as it is.
    """
    unsure = (
        f"Walkie lifted {label} with the {arm} hand and brought both arms home, but could not "
        f"read the {arm} gripper after, so it may have lost {label} after the lift: "
    )
    check = f"Call get_arm_state to see whether the {arm} hand still holds it."
    try:
        reading = ctx.arms.read()
    except StepError as error:
        if error.code is ErrorCode.CANCELLED:
            raise
        logger.warning("Could not read the %s gripper after the pick of %s: %s", arm, label, error)
        now = ctx.arms.state(read=False)
        raise answers.with_sentence(error, unsure, f"{check} {now.sentence()}") from None
    gripper = next((found for found in reading.grippers if found.arm == arm), None)
    if gripper is None or gripper.in_grasp_band is None:
        why = gripper.read_error if gripper is not None and gripper.read_error else (
            f"The reading lacks the {arm} gripper."
        )
        logger.warning("Could not read the %s gripper after the pick of %s: %s", arm, label, why)
        raise StepError(
            ErrorCode.SKILL_FAILED,
            f"{unsure}{answers.full_stop(why)} {check} {ctx.arms.state().sentence()}",
        )
    if not gripper.in_grasp_band:
        logger.warning(
            "The %s hand was empty after the pick of %s: finger at %.4f m.",
            arm,
            label,
            gripper.finger_m,
        )
        raise StepError(
            ErrorCode.SKILL_FAILED,
            f"Walkie lifted {label} with the {arm} hand, but the hand was empty once both arms "
            f"came home: it lost {label} after the lift. It may have fallen near where Walkie "
            f"stood to pick it up: call find_objects to look for it. "
            f"{ctx.arms.state().sentence()}",
        )


def _go_home(ctx: Ctx, settings: PickSettings, prefix: str) -> None:
    """Move the picking arm home with the lift at home_lift_m, then the other arm home.

    The straight moves may have lowered the lift so far that the arm's home pose
    touches the base, so the first move also raises the lift. It keeps the fingers'
    octomap allowance: right after the lift the octomap still has voxels at the
    hand, and MoveIt finds the start state in collision without it. The second
    move ends the allowance. An error keeps its code and class, with `prefix` before
    its message.
    """
    arm = settings.arm
    try:
        ctx.arms.move_to(HOME_POSE, arm, lift_m=settings.home_lift_m, touch=True)
        ctx.arms.move_to(HOME_POSE, _OTHER_ARM[arm])
    except StepError as error:
        raise type(error)(error.code, f"{prefix}{error.message}") from None


def _reach(
    ctx: Ctx,
    settings: PickSettings,
    pose: HandPose,
    prefix: str,
    *,
    whole_body: bool = False,
    cartesian: bool = False,
    touch: bool = False,
    ready: Callable[[], object] | None = None,
) -> None:
    """One reach of the settings' arm, with `prefix` before its error's message.

    The error keeps its code and its class, and its message still ends with the arm
    sentence. `ready` runs right before the goal (Arms.reach), and its error passes
    without `prefix`: no goal was sent, so the reach did not fail.
    """
    timeout_s = settings.reach_timeout_s if whole_body else settings.cartesian_timeout_s
    refused: list[StepError] = []
    check = None
    if ready is not None:
        call = ready

        def check() -> None:
            try:
                call()
            except StepError as error:
                refused.append(error)
                raise

    try:
        ctx.arms.reach(
            settings.arm,
            pose,
            whole_body=whole_body,
            cartesian=cartesian,
            touch=touch,
            timeout_s=timeout_s,
            ready=check,
        )
    except StepError as error:
        if refused and error is refused[0]:
            raise
        raise type(error)(error.code, f"{prefix}{error.message}") from None


def _distance(pose: Pose2D, plan: PickPlan) -> float:
    return math.hypot(plan.standoff_x - pose.x, plan.standoff_y - pose.y)


def _near(pose: Pose2D, plan: PickPlan) -> bool:
    """Whether `pose` is within NEAR_STANDOFF_M and NEAR_STANDOFF_DEG of the plan's stand-off."""
    turn = math.degrees(math.remainder(plan.heading - pose.yaw, math.tau))
    return _distance(pose, plan) <= NEAR_STANDOFF_M and abs(turn) <= NEAR_STANDOFF_DEG
