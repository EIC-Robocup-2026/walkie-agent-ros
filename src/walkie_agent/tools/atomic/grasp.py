"""pick_object: pick up a remembered object with one hand, by its name (decision PK-4).

The atomic finds the object in the object memory, drives to a stand-off pose facing
it, and reaches through ctx.arms: a whole-body reach to a pre-grasp pose, a straight
approach, a close of the gripper judged by the commander's grasp band, and a straight
lift. The geometry and its numbers are in walkie_agent.arm.pick and config/pick.toml.
The LLM gives only the object's name and reads only the outcome: no position, pose or
joint value reaches it (ADR-0022). Every answer ends with the arm sentence (ADR-0024).

The stand-off pose is reached with ctx.base.move_relative, a Nav2 goal like any other
drive, after a go_to to the named place nearest the object when the object is further
than one relative move allows. walkie-ros's /navigate_to_object computes a stand-off
of its own, but it is in walkie-ros's navigation submodule, whose interfaces the agent's
workspace does not build, and its pose would not match the hand's approach here.
"""

import logging
import math

from walkie_agent.arm.pick import (
    HandPose,
    PickPlan,
    PickSettings,
    approach_headings,
    load_pick_settings,
    plan_pick,
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
# How many objects of the name the memory is asked for. The first with a position is
# picked, in the memory's order, which puts the closest matches of the name first.
SEARCH_LIMIT = 10

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

    Walkie finds the object in its memory, drives up to it, reaches for it with the
    right hand, closes the hand and lifts it. It takes a minute or more. Tell the
    people near you before you call it. Returns a sentence that says whether the
    right hand holds the object, which ends with whether the arms are safe for
    driving. After a pick the right arm is not safe for driving: before you drive,
    move it to a pose that list_arm_poses says is safe for driving.

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
    return _grasp(ctx, settings, _plan_from_here(ctx, settings, found), label)


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


def _drive_to_standoff(
    ctx: Ctx, settings: PickSettings, found: RecalledObject, label: str
) -> None:
    """Drive to a stand-off pose facing `found`, trying each approach heading in turn.

    A first go_to to the named place nearest the object comes when the first
    stand-off is further than one relative move allows. The headings are taken
    from where the robot stands then. A heading whose drive Nav2 fails
    (SKILL_FAILED) gives way to the next; any other error is raised as it is.
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
        ctx.base.go_to(place.name)
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


def _plan_from_here(ctx: Ctx, settings: PickSettings, found: RecalledObject) -> PickPlan:
    """The plan for the heading the robot now faces the object along.

    The hand approaches along the line from where the base stopped, so a drive that
    ended a little off its goal still reaches straight at the object.
    """
    here = ctx.base.pose()
    heading = math.atan2(found.y - here.y, found.x - here.x)
    return plan_pick(settings, found.x, found.y, found.z, heading)


def _grasp(ctx: Ctx, settings: PickSettings, plan: PickPlan, label: str) -> str:
    """Open the hand, reach, approach, close and lift. Returns the answer to the LLM."""
    arm = settings.arm
    opened = ctx.arms.gripper(arm, "open")
    if opened.outcome != "moved":
        raise StepError(
            ErrorCode.SKILL_FAILED,
            f"The {arm} gripper did not open, so Walkie did not reach for {label}. "
            f"{ctx.arms.state().sentence()}",
        )
    _reach(ctx, settings, plan.pregrasp, f"The reach toward {label} failed: ", whole_body=True)
    _reach(ctx, settings, plan.grasp, f"The approach to {label} failed: ", cartesian=True)
    closed = ctx.arms.gripper(arm, "close")
    if closed.outcome != "holding":
        why = _NOT_HELD.get(closed.outcome, "the close did not hold it").format(arm=arm)
        raise StepError(
            ErrorCode.SKILL_FAILED,
            f"Walkie reached {label}, but {why}, so it did not pick it up. The object may "
            f"have moved: call find_objects to look again. {ctx.arms.state().sentence()}",
        )
    _reach(
        ctx, settings, plan.lift, f"The {arm} hand holds {label}, but the lift failed: ",
        cartesian=True,
    )
    return (
        f"Picked up {label} with the {arm} hand and lifted it. Before you drive, move the "
        f"{arm} arm to a pose that is safe for driving. {ctx.arms.state().sentence()}"
    )


def _reach(
    ctx: Ctx,
    settings: PickSettings,
    pose: HandPose,
    prefix: str,
    *,
    whole_body: bool = False,
    cartesian: bool = False,
) -> None:
    """One reach of the settings' arm, with `prefix` before its error's message.

    The error keeps its code and its class, and its message still ends with the arm
    sentence.
    """
    timeout_s = settings.reach_timeout_s if whole_body else settings.cartesian_timeout_s
    try:
        ctx.arms.reach(
            settings.arm, pose, whole_body=whole_body, cartesian=cartesian, timeout_s=timeout_s
        )
    except StepError as error:
        raise type(error)(error.code, f"{prefix}{error.message}") from None


def _distance(pose: Pose2D, plan: PickPlan) -> float:
    return math.hypot(plan.standoff_x - pose.x, plan.standoff_y - pose.y)
