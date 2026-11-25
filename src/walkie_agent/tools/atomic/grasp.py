"""pick_object: pick up a remembered object with one hand, by its name (decision PK-4).

The atomic finds the object in the object memory, one the camera sees now first,
refuses it before anything moves when the memory's fresh box for it is wider than the
open gripper (max_grasp_width_m) on both sides, drives to a stand-off pose facing it,
along its longer side when that one
is wider than the gripper, raises the other arm out of the way, and
reaches through ctx.arms: a whole-body
reach to a pre-grasp pose, a straight approach, a close of the gripper judged by the
commander's grasp band, a straight lift, a straight move to beside the base, and a move
of both arms home, the picking arm with the lift. The gripper is then read again and
judged by the same band, so an object that fell out of the hand after the close is
reported lost, not picked. The hand takes the object the way config/pick.toml's grasp
table gives for the memory's detector label (arm.pick.PickSettings.grasp_for): level
from the side, or straight down from above. The geometry and its numbers are in
walkie_agent.arm.pick and config/pick.toml.
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
(arm.pick.reach_problems), and the object must be no wider than the open gripper
across the fingers of the planned hand (arm.pick.width_across_fingers); A plans the
hand poses afresh from its own reading. A
motion is never cut short for a newer reading: the checks run between motions. Once
the first arm goal may go, the fingers' octomap allowance (allow_gripper_vs_octomap)
is turned off again however the pick ends, a stop included (_grasp_then_touch_off).
That reset sends no motion, so a stop still ends every motion.

The stand-off pose is reached with ctx.base.move_relative, a Nav2 goal like any other
drive. When the first stand-off is further than one relative move allows, a drive
comes first: back to where Walkie stood when it last saw the object
(navigation.seen_from, decision PK-79) when the base kept that pose and a stand-off
lies within one move of it, then, should that drive fail far off, to the named place
nearest the object. The approach headings are taken from that viewpoint when it
is known, else from where the robot stands. The viewpoint is a spot the robot stood on
with a line of sight to the object, so a stand-off on its side faces the object across
floor the robot is known to reach. A heading taken from elsewhere, such as from the
far side of a table, can put the stand-off where the global costmap, built from the 2D
lidar alone, sees free floor under a tabletop it cannot see. walkie-ros's
/navigate_to_object computes a stand-off of its own, but it is in walkie-ros's
navigation submodule, whose interfaces the agent's workspace does not build, and its
pose would not match the hand's approach here.

The object memory also records what the hand holds (tools.atomic.held, decision
PK-70). Once the close holds the object, and before the lift, it is recorded held in
the picking hand, so the memory stops matching the hand's sightings of it to the
table; a hand found empty after the move home records it lost. Before anything
moves, a pick while the memory holds an object in a hand reads the picking gripper:
an empty one records the object lost and the pick goes on, and one that holds
something, or cannot be read, refuses the pick with PRECONDITION_FAILED, since
opening the hand would drop the object. Every record is best-effort and never ends
a pick.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

from walkie_agent.arm import answers
from walkie_agent.arm.pick import (
    AT_STANDOFF_DEG,
    AT_STANDOFF_M,
    Grasp,
    HandPose,
    PickPlan,
    PickSettings,
    approach_headings,
    heading_from,
    height_problem,
    load_pick_settings,
    plan_pick,
    reach_problems,
    relative_move,
    too_wide_to_grasp,
    width_across_fingers,
)
from walkie_agent.capabilities import Ctx, Pose2D, RecalledObject
from walkie_agent.capabilities.guards import MAX_MOVE_M
from walkie_agent.errors import ErrorCode, GoalFailed, StepError
from walkie_agent.perception.object_words import matches
from walkie_agent.tools.atomic.held import held_objects, record
from walkie_agent.tools.atomic.navigation import APPROACH_PLACE_MAX_M, approach_place, seen_from

logger = logging.getLogger(__name__)

# How close a drive that Nav2 failed must have left the base to its stand-off pose for
# the pick to go on from there, in metres and degrees. Nav2 aborts a goal it cannot
# settle on ("Failed to make progress") although the base stopped a few centimetres
# off, as it did beside the Chula bar. The hand then moves in along the line from
# where the base stopped (_plan_from_here), and the whole-body reach covers the rest.
NEAR_STANDOFF_M = 0.10
NEAR_STANDOFF_DEG = 10.0
# How many objects of the name the memory is asked for. The first with a position is
# picked, those seen at most max_observation_age_s ago first, then in the memory's
# order, which puts the closest matches of the name first and, among equal matches,
# the one nearest the robot: with two bottles in the memory, the one in front of
# Walkie, not one it saw in another room. The memory can hold stale tracks of an
# object it no longer sees there, such as one that followed a bottle into the hand
# during a pick, so a track the camera sees now goes before them (decision PK-67).
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
    asked = name.strip()
    if not asked:
        raise StepError(ErrorCode.REFUSED, "Refused: name the object to pick up.")
    try:
        settings = load_pick_settings()
    except (OSError, ValueError) as error:
        logger.error("Cannot read the pick settings: %s", error)
        raise StepError(
            ErrorCode.INTERNAL, "Walkie's pick settings cannot be read, so it cannot pick."
        ) from None
    _hand_is_free(ctx, settings, asked)
    found = _remembered(ctx, settings, asked)
    label = f"the {found.name}"
    grasp = settings.grasp_for(found.label)
    _fits_hand(settings, grasp, found, label)
    _drive_to_standoff(ctx, settings, grasp, found, label)
    found = _seen_again(ctx, settings, found, label, _AFTER_DRIVE)
    _plan_in_reach(ctx, settings, grasp, found, label, _AFTER_DRIVE)
    return _grasp_then_touch_off(ctx, settings, grasp, found, label)


def _hand_is_free(ctx: Ctx, settings: PickSettings, name: str) -> None:
    """Refuse when the memory holds an object in a hand and the picking hand holds one.

    The pick opens the hand first, which would drop what it holds. The memory's word
    is checked against the gripper (_hand_holds): a hand that the read finds empty
    lost the object unnoticed, so the object is recorded lost and the pick goes on.
    A gripper that cannot be read is taken to hold what the memory says.

    The memory does not say which hand holds an object, so every object it holds is
    taken to be in the picking hand. The record for an object in the other hand is
    refused by the memory, which leaves it held.
    """
    held = held_objects(ctx)
    if not held:
        return
    arm = settings.arm
    if _hand_holds(ctx, arm) is False:
        for found in held:
            record(ctx, found.id, "grip_lost", arm)
        return
    asked = name.lower()
    wanted = next(
        (found for found in held if matches(asked, found.name) or matches(asked, found.label)),
        None,
    )
    handover = (
        f"To give it to a person, move the {arm} arm to the hands_up pose and open the "
        "gripper once they hold it."
    )
    if wanted is not None:
        raise StepError(
            ErrorCode.PRECONDITION_FAILED,
            f"Walkie is already holding the {wanted.name} in its {arm} hand. {handover}",
        )
    names = " and the ".join(found.name for found in held)
    raise StepError(
        ErrorCode.PRECONDITION_FAILED,
        f"Walkie's {arm} hand already holds the {names}, and opening it to pick up the "
        f"{asked} would drop it. {handover}",
    )


def _hand_holds(ctx: Ctx, arm: str) -> bool | None:
    """Whether the `arm` gripper holds something by its grasp band; None when unread."""
    try:
        reading = ctx.arms.read()
    except StepError as error:
        if error.code is ErrorCode.CANCELLED:
            raise
        logger.warning("Could not read the %s gripper before the pick: %s", arm, error.message)
        return None
    gripper = next((found for found in reading.grippers if found.arm == arm), None)
    return None if gripper is None else gripper.in_grasp_band


def _remembered(ctx: Ctx, settings: PickSettings, name: str) -> RecalledObject:
    """The first object of `name` that the memory holds as present with a full position.

    One seen at most max_observation_age_s ago comes first (_is_fresh); otherwise, and
    among equal matches of the name, the one nearest where the robot stands.
    """
    here = ctx.base.pose()
    matches = ctx.object_memory.search(name, limit=SEARCH_LIMIT, near=(here.x, here.y))
    if not matches:
        raise StepError(
            ErrorCode.NOT_FOUND,
            f"Walkie remembers no {name}. Call recall_object to see what it remembers.",
        )
    placed = [
        found
        for found in matches
        if found.x is not None and found.y is not None and found.z is not None
    ]
    if placed:
        # sorted() is stable, so the memory's order holds inside each group.
        return sorted(placed, key=lambda found: not _is_fresh(settings, found))[0]
    raise StepError(
        ErrorCode.PRECONDITION_FAILED,
        f"Walkie remembers a {name}, but not exactly where it is, so it cannot reach for "
        "it. Go to it and call find_objects to look again.",
    )


def _fits_hand(
    settings: PickSettings, grasp: Grasp, found: RecalledObject, label: str
) -> None:
    """Refuse an object wider than the open gripper, before anything moves.

    By the memory's box (arm.pick.too_wide_to_grasp), and only when the camera saw the
    object at most max_observation_age_s ago (_is_fresh). An older box can belong to a
    stale track or an earlier view, and the robot then drives up, and the check after
    the drive measures the width on a fresh sighting (_fits_fingers). An object with
    no box passes. The width goes to the LLM: a size, not a position (ADR-0022). The
    box and the grasp are logged for every pick, since nothing yet checks the memory's
    boxes against the objects.
    """
    logger.info(
        "Picking %s (object %s) with the %s grasp for label %r: the memory's box is %s m, "
        "yaw %s rad.",
        label,
        found.id,
        grasp.name,
        found.label,
        "none" if found.size is None else "x".join(f"{side:.3f}" for side in found.size),
        "none" if found.yaw is None else f"{found.yaw:.2f}",
    )
    width = too_wide_to_grasp(settings, found.size)
    if width is None:
        return
    if not _is_fresh(settings, found):
        logger.info(
            "%s (object %s) reads %.3f m wide, above max_grasp_width_m %g, but was last "
            "seen %s s ago; the width is checked again after the drive.",
            label.capitalize(),
            found.id,
            width,
            settings.max_grasp_width_m,
            found.seen_s_ago,
        )
        return
    logger.info(
        "Refused to pick %s (object %s): %.3f m wide, above max_grasp_width_m %g.",
        label,
        found.id,
        width,
        settings.max_grasp_width_m,
    )
    raise StepError(
        ErrorCode.PRECONDITION_FAILED,
        f"{label.capitalize()} is {width:.3f} m wide, and the {settings.arm} hand cannot open "
        "that wide, so Walkie did not try to pick it up. Tell the person it cannot pick it up.",
    )


def _is_fresh(settings: PickSettings, found: RecalledObject) -> bool:
    """True when the memory saw `found` at most max_observation_age_s ago."""
    return found.seen_s_ago is not None and found.seen_s_ago <= settings.max_observation_age_s


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
        return _is_fresh(settings, now)

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
    ctx: Ctx, settings: PickSettings, grasp: Grasp, found: RecalledObject, label: str
) -> None:
    """Drive to a stand-off pose of `grasp` facing `found`, trying each approach heading in turn.

    The headings are taken from where Walkie stood when it last saw the object
    (seen_from), when the base kept that pose, else from where the robot stands. That
    spot had a line of sight to the object and the robot stood on it, so a stand-off
    on its side is one the robot can reach facing the object; from the far side of a
    table, the global costmap, built from the 2D lidar alone, cannot see the tabletop
    a stand-off there would sit under. When the first stand-off is further than one
    relative move allows, _drive_closer comes first, and the headings are then taken
    again. A heading whose drive Nav2 ended without success (GoalFailed) gives way to
    the next, unless the base stopped near its stand-off pose (_near). Any other
    error is raised as it is: after a goal whose state is unknown the base may still
    be driving, so neither its pose nor the next goal is any use (S2-9). Each
    heading's plan goes to ctx.display before its drive.
    """
    viewpoint = seen_from(ctx, found)
    start = ctx.base.pose()
    plans = _plans(settings, grasp, found, viewpoint or start)
    if _distance(start, plans[0]) > MAX_MOVE_M:
        _drive_closer(ctx, settings, grasp, found, label, viewpoint)
        plans = _plans(settings, grasp, found, viewpoint or ctx.base.pose())
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
        ctx.display.show_pick(plan, found, "driving to the stand-off")
        try:
            ctx.base.move_relative(forward, left, turn)
        except GoalFailed as error:
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


def _drive_closer(
    ctx: Ctx,
    settings: PickSettings,
    grasp: Grasp,
    found: RecalledObject,
    label: str,
    viewpoint: Pose2D | None,
) -> None:
    """The first drive of _drive_to_standoff, to where a stand-off is one relative move away.

    It tries go_to_pose back to `viewpoint`, where Walkie last saw `found`, when the
    first stand-off from there lies within one relative move of it, then go_to the
    named place nearest the object (approach_place). A drive that Nav2 ended without
    success (GoalFailed) still counts when the base stopped within one move of the
    first stand-off, since Nav2 aborted the drive to the Chula bar 0.1 m from its
    goal; otherwise the next drive is tried, and the last one's failure is raised.
    Any other error is raised as it is. PRECONDITION_FAILED when neither drive applies.
    """
    def ended_near(where: str, error: GoalFailed) -> bool:
        here = ctx.base.pose()
        near = _distance(here, _plans(settings, grasp, found, viewpoint or here)[0]) <= MAX_MOVE_M
        logger.info(
            "The drive to %s failed %s: %s", where, "close enough to go on" if near else "far off",
            error,
        )
        return near

    failed: GoalFailed | None = None
    if viewpoint is not None:
        first = _plans(settings, grasp, found, viewpoint)[0]
        if _distance(viewpoint, first) <= MAX_MOVE_M:
            try:
                ctx.base.go_to_pose(viewpoint)
                return
            except GoalFailed as error:
                if ended_near("where Walkie last saw it", error):
                    return
                failed = error
    place = approach_place(ctx, found)
    if place is None and failed is not None:
        raise failed
    if place is None:
        raise StepError(
            ErrorCode.PRECONDITION_FAILED,
            f"{label.capitalize()} is far away, and Walkie knows no named place "
            f"within {APPROACH_PLACE_MAX_M:g} m of it to drive to first.",
        )
    try:
        ctx.base.go_to(place.name)
    except GoalFailed as error:
        if not ended_near(place.name, error):
            raise


def _plans(
    settings: PickSettings, grasp: Grasp, found: RecalledObject, origin: Pose2D
) -> list[PickPlan]:
    """One plan per approach heading, as approach_headings orders them from `origin`'s position."""
    headings = approach_headings(
        settings, grasp, origin.x, origin.y, found.x, found.y, found.size, found.yaw
    )
    return [
        plan_pick(settings, grasp, found.x, found.y, found.z, heading) for heading in headings
    ]


def _plan_from_here(
    settings: PickSettings, grasp: Grasp, found: RecalledObject, here: Pose2D
) -> PickPlan:
    """The plan for the heading the robot now faces the object along (arm.pick.heading_from).

    The hand approaches along the line from where the base stopped, `here`, to the
    object, so a drive that ended a little off its goal still reaches straight at it.
    """
    heading = heading_from(grasp, here.x, here.y, found.x, found.y)
    approach = math.atan2(found.y - here.y, found.x - here.x)
    return plan_pick(
        settings, grasp, found.x, found.y, found.z, heading, approach, (here.x, here.y, here.yaw)
    )


def _plan_in_reach(
    ctx: Ctx,
    settings: PickSettings,
    grasp: Grasp,
    found: RecalledObject,
    label: str,
    checkpoint: _Checkpoint,
) -> PickPlan:
    """The plan from where the base stands now, refused outside the reach envelope.

    Before the checkpoint's next arm goal (arm.pick.reach_problems). The numbers go
    to the log only. An object too low or too high, or one the grasp would take with
    the hand too low or too high, stays so from any stand-off pose
    (arm.pick.height_problem); any other problem may pass from a new one. A plan in
    reach whose hand would close across more of the object than the open gripper is
    refused too (_fits_fingers). The plan goes to ctx.display first, refused or not.
    """
    here = ctx.base.pose()
    plan = _plan_from_here(settings, grasp, found, here)
    ctx.display.show_pick(plan, found, f"planned {checkpoint.name}")
    problems = reach_problems(
        settings, grasp, plan, (found.x, found.y, found.z), (here.x, here.y, here.yaw)
    )
    if not problems:
        _fits_fingers(ctx, settings, found, label, checkpoint, plan)
        return plan
    logger.warning(
        "Refused to reach for %s (object %s) %s: %s.",
        label,
        found.id,
        checkpoint.name,
        "; ".join(problems),
    )
    sentence = ctx.arms.state().sentence()
    side = height_problem(settings, plan, found.z)
    if side is not None:
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


def _fits_fingers(
    ctx: Ctx,
    settings: PickSettings,
    found: RecalledObject,
    label: str,
    checkpoint: _Checkpoint,
    plan: PickPlan,
) -> None:
    """Refuse a plan whose hand would close across more of the object than it opens.

    First by the narrower side of the memory's box (arm.pick.too_wide_to_grasp), which
    needs no yaw: an object wider than the open gripper both ways fits from no
    heading, so the answer does not ask for another try. A box seen long ago is let
    through the gate before the drive and stops here instead (_fits_hand). Then by
    the box and the grasp pose's yaw (arm.pick.width_across_fingers). The drive aims
    the hand across the narrower side of a long object, but the hand moves in along
    the line from where the base stopped, which can turn it. An object with no box
    passes, and one with no yaw passes the second check. The width goes to the LLM: a
    size, not a position (ADR-0022).
    """
    narrower = too_wide_to_grasp(settings, found.size)
    if narrower is not None:
        logger.warning(
            "Refused to reach for %s (object %s) %s: %.3f m wide, above max_grasp_width_m %g.",
            label,
            found.id,
            checkpoint.name,
            narrower,
            settings.max_grasp_width_m,
        )
        advice = checkpoint.advice(None)
        raise StepError(
            ErrorCode.PRECONDITION_FAILED,
            f"{label.capitalize()} is {narrower:.3f} m wide, and the {settings.arm} hand "
            "cannot open that wide, so Walkie did not reach for it. "
            f"{f'{advice} ' if advice else ''}{ctx.arms.state().sentence()}",
        )
    width = width_across_fingers(found.size, found.yaw, plan.grasp.yaw)
    if width is None or width <= settings.max_grasp_width_m:
        return
    logger.warning(
        "Refused to reach for %s (object %s) %s: %.3f m across the fingers, above "
        "max_grasp_width_m %g.",
        label,
        found.id,
        checkpoint.name,
        width,
        settings.max_grasp_width_m,
    )
    raise StepError(
        ErrorCode.PRECONDITION_FAILED,
        f"From where Walkie stopped, the {settings.arm} hand would close across "
        f"{label} where it is {width:.3f} m wide, wider than the hand opens, so Walkie "
        f"did not reach for it. "
        f"{checkpoint.advice('call pick_object again to approach it afresh')} "
        f"{ctx.arms.state().sentence()}",
    )


def _grasp_then_touch_off(
    ctx: Ctx, settings: PickSettings, grasp: Grasp, found: RecalledObject, label: str
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
        answer = _grasp(ctx, settings, grasp, found, label)
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


def _grasp(
    ctx: Ctx, settings: PickSettings, grasp: Grasp, found: RecalledObject, label: str
) -> str:
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
    plan = _plan_in_reach(ctx, settings, grasp, seen, label, _BEFORE_REACH)
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
    # Before the lift: from here on the camera sees the object move with the hand, and
    # the memory must stop matching those sightings to the place it was picked from.
    record(ctx, seen.id, "grasp_succeeded", arm)
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
    _still_holds(ctx, arm, label, seen.id)
    return (
        f"Picked up {label} with the {arm} hand, lifted it and brought both arms home. "
        f"{ctx.arms.state().sentence()}"
    )


def _still_holds(ctx: Ctx, arm: str, label: str, object_id: str) -> None:
    """Raise unless the `arm` hand still holds the object once both arms are home.

    The close was judged before the lift, and the lift, the move clear and the move
    home can each shake the object out of the fingers. So the gripper is read again
    through ctx.arms.read(), whose GripperReading.in_grasp_band applies the band the
    close was judged by. A hand found empty is SKILL_FAILED: the object was lost
    after the lift, and the memory records `object_id` lost, so it places the object
    nowhere until the camera sees it again. A read that fails keeps its code and
    class, and a gripper that could not be read is SKILL_FAILED; both say that the
    object may be lost. A stop during the read is raised as it is.
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
        record(ctx, object_id, "grip_lost", arm)
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
