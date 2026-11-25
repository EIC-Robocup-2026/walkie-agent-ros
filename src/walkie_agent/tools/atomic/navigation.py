"""Atomics that move the base and name the places it can go to."""

import logging
import math

from walkie_agent.capabilities import Ctx, Pose2D, RecalledObject
from walkie_agent.errors import ErrorCode, GoalFailed, StepError
from walkie_agent.locations import Location, nearest_place
from walkie_agent.perception.geometry import wrap_deg
from walkie_agent.perception.object_words import matches

logger = logging.getLogger(__name__)

# How far from straight ahead a person may be and still count as faced. A person
# within it gets no turn, and a turn on the spot larger than it that Nav2 reports as
# ended without success still counts as done if its start pose is known and it ended
# within it of its goal: Nav2's progress checker can abort a turn on the spot that is
# still settling on its 0.6 degree yaw tolerance (decisions S2-4, S2-9 and S2-11).
FACE_TOLERANCE_DEG = 10.0
# How far a named place may be from a remembered object for go_to_object to drive
# there, and for recall_object to name it as the object's near_place (decision SG-18).
# In the Chula simulation every object the memory kept was 0.47 to 1.41 m from its
# nearest place.
APPROACH_PLACE_MAX_M = 2.0
# A pose this close to an object, seen from above, is no viewpoint on it: no heading
# faces the object from there (seen_from).
VIEWPOINT_MIN_M = 0.05
# A pose further than this from an object is no viewpoint on it either (seen_from):
# go_to_object would stop further from the object than a named place within
# APPROACH_PLACE_MAX_M, and pick_object's stand-off would lie beyond one relative
# move (MAX_MOVE_M, 3 m) from it. A glance from across a room, or a stale sighting
# time from an earlier run, gives such a pose.
VIEWPOINT_MAX_M = 3.0
# The states in which a remembered object is no longer where Walkie saw it.
_MOVED_STATES = frozenset({"grasping", "held", "handed_over", "discarded"})


def list_places(ctx: Ctx) -> list[dict[str, str]]:
    """List the named places Walkie can go to.

    Returns a list of objects with the keys name and description.
    """
    return [
        {"name": place.name, "description": place.description}
        for place in ctx.base.places().values()
    ]


def go_to(ctx: Ctx, place: str) -> str:
    """Drive to a named place and wait until Walkie arrives.

    Returns a sentence that says where Walkie arrived.

    Args:
        place: The name of a room or a location, such as "kitchen room" or "shelf".
    """
    ctx.base.go_to(place)
    return f"Arrived at {place.strip()}."


def move_relative(
    ctx: Ctx, forward_m: float, left_m: float = 0.0, turn_deg: float = 0.0
) -> str:
    """Move a short distance relative to where Walkie stands and faces now.

    One call moves at most 3 m. Use go_to for anything with a name. Returns a
    sentence with the move and the pose Walkie ended at.

    Args:
        forward_m: Meters to move forward. Negative moves backward.
        left_m: Meters to move to the left. Negative moves right.
        turn_deg: Degrees to turn once there. Positive turns left, negative turns right.
    """
    # Wrapped, since a 355 degree goal is 5 degrees from where a refused turn leaves
    # the robot.
    if not forward_m and not left_m and abs(wrap_deg(turn_deg)) > FACE_TOLERANCE_DEG:
        what = f"the turn of {turn_deg:g} degrees"
        pose = _turn(ctx, _start_pose(ctx, what), turn_deg, what)
    else:
        pose = ctx.base.move_relative(forward_m, left_m, turn_deg)
    return (
        f"Moved {_describe_move(forward_m, left_m, turn_deg)}. "
        f"Now at x={_round(pose.x, 2):.2f} m, y={_round(pose.y, 2):.2f} m, "
        f"heading {_round(math.degrees(pose.yaw), 0):.0f} degrees."
    )


def face_person(ctx: Ctx, person_id: str) -> str:
    """Turn on the spot, once, to face a person in view.

    Returns a sentence that says how far Walkie turned. A person already within
    10 degrees of straight ahead gets no turn. Walkie does not keep following
    the person afterwards.

    Args:
        person_id: The id of a person, as find_people returns it.
    """
    seen = next((p for p in ctx.vision.people() if p.id == person_id), None)
    if seen is None:
        raise StepError(
            ErrorCode.NOT_FOUND, f"{person_id} is not in view here. Call find_people first."
        )
    if seen.bearing_deg is None:
        raise StepError(ErrorCode.UNAVAILABLE, f"Walkie cannot tell which way {person_id} is.")
    bearing = seen.bearing_deg
    if abs(bearing) <= FACE_TOLERANCE_DEG:
        return f"Already facing {person_id}, {_offset(bearing)}."
    what = f"the turn to {person_id}"
    start = _start_pose(ctx, what)
    end = _turn(ctx, start, bearing, what)
    # Without a start, a turn the base reports as done is taken to have turned the
    # bearing it was asked to.
    turned = bearing if start is None else wrap_deg(math.degrees(end.yaw - start.yaw))
    return f"Turned {_round(abs(turned), 0):.0f} degrees {_side(turned)} to face {person_id}."


def go_to_object(ctx: Ctx, object_id: str) -> str:
    """Drive to an object Walkie remembers, and turn to face it.

    Use it to fetch or show an object that recall_object or list_room_objects
    found. It drives to where Walkie last saw the object when it knows that spot,
    otherwise to the named place nearest it. Walkie has not looked for the object
    when it arrives, and it may have moved since, so call find_objects next to see
    whether it is still there. Returns a sentence that says where Walkie arrived
    and whether it faces the object.

    Args:
        object_id: The id of a remembered object, as recall_object or
            list_room_objects return it, such as "18". Not an id from find_objects.
    """
    found = ctx.object_memory.get(object_id)
    asked = object_id.strip()
    # The memory follows a merge to the object that took the asked one over.
    renamed = f"Object {asked} is now remembered as {found.id}. " if found.id != asked else ""
    label = f"the {found.name} {found.id}"
    if found.state in _MOVED_STATES:
        state = found.state.replace("_", " ")
        raise StepError(
            ErrorCode.PRECONDITION_FAILED,
            f"{renamed}Walkie remembers {label} as {state}, so it is not where Walkie saw it.",
        )
    word = _camera_word(found)
    room = f" Go to the {found.room} and call find_objects there.{word}" if found.room else ""
    if found.x is None or found.y is None:
        raise StepError(
            ErrorCode.PRECONDITION_FAILED, f"{renamed}Walkie does not know where {label} is.{room}"
        )
    state = "" if found.state == "present" else f" Walkie remembers it as {found.state}."
    after = (
        f"{state} Walkie has not looked for it yet: call find_objects to see whether it "
        f"is still there.{word}"
    )
    # Where Walkie stood when it last saw the object had a line of sight to it, which a
    # named place near it need not have (decision PK-79).
    viewpoint = seen_from(ctx, found)
    failed = ""
    if viewpoint is not None:
        try:
            ctx.base.go_to_pose(viewpoint)
        except GoalFailed as error:
            # Only a drive known to have ended: with its state unknown the base may
            # still be driving, and the next goal would be refused (S2-9).
            logger.warning(
                "Could not drive back to where Walkie last saw %s, so driving to the named "
                "place nearest it: %s",
                label,
                error,
            )
            failed = error.message.rstrip(".")
        else:
            facing = _face_point(ctx, found.x, found.y, label)
            return f"{renamed}Drove back to where Walkie last saw {label}, {facing}.{after}"
    place = approach_place(ctx, found)
    if place is None and failed:
        raise StepError(
            ErrorCode.SKILL_FAILED,
            f"{renamed}Walkie could not drive back to where it last saw {label} ({failed}), "
            f"and knows no named place within {APPROACH_PLACE_MAX_M:g} m of it to try "
            f"instead.{room}",
        )
    if place is None:
        raise StepError(
            ErrorCode.PRECONDITION_FAILED,
            f"{renamed}Walkie knows no named place within {APPROACH_PLACE_MAX_M:g} m of "
            f"{label}.{room}",
        )
    ctx.base.go_to(place.name)
    facing = _face_point(ctx, found.x, found.y, label)
    return f"{renamed}Arrived at {place.name}, the named place nearest {label}, {facing}.{after}"


def _camera_word(found: RecalledObject) -> str:
    """The sentence naming the detector's word for `found`, or "" when its name is that word.

    An official name, such as "stool", is not a word the detector looks for, so
    find_objects would refuse it; the stored label, such as "chair", is (SG-27).
    """
    if not found.label or matches(found.name.strip().lower(), found.label):
        return ""
    return f' The camera calls it "{found.label}", so call find_objects("{found.label}").'


def approach_place(ctx: Ctx, found: RecalledObject) -> Location | None:
    """The named place that go_to_object drives to for `found`, or None without one.

    It is the place nearest the object within APPROACH_PLACE_MAX_M, one in the
    object's room first (locations.nearest_place), and None for an object whose
    position is unknown.
    """
    if found.x is None or found.y is None:
        return None
    return nearest_place(
        ctx.base.places().values(),
        found.x,
        found.y,
        max_m=APPROACH_PLACE_MAX_M,
        room=found.room,
    )


def seen_from(ctx: Ctx, found: RecalledObject) -> Pose2D | None:
    """Where Walkie stood when it last saw `found`, facing it, or None when unknown.

    The position is ctx.base.pose_at at the memory's last sighting, and the yaw faces
    the object's centre from there. That pose had a line of sight to the object, so
    go_to_object and pick_object drive back to it before a named place. None when the
    sighting's time or the object's position is unknown, when the base kept no pose
    for that time, or when the pose lies within VIEWPOINT_MIN_M or further than
    VIEWPOINT_MAX_M from the object. A failed lookup is logged and counts as
    unknown; a stop is raised, since nothing more may run under it.
    """
    if found.seen_at is None or found.x is None or found.y is None:
        return None
    try:
        pose = ctx.base.pose_at(found.seen_at)
    except StepError as error:
        if error.code is ErrorCode.CANCELLED:
            raise
        logger.warning(
            "Could not look up where Walkie stood when it last saw object %s: %s", found.id, error
        )
        return None
    if pose is None:
        return None
    if not VIEWPOINT_MIN_M < math.hypot(found.x - pose.x, found.y - pose.y) <= VIEWPOINT_MAX_M:
        return None
    return Pose2D(pose.x, pose.y, math.atan2(found.y - pose.y, found.x - pose.x))


def _face_point(ctx: Ctx, x: float, y: float, label: str) -> str:
    """Turn on the spot to face the map point (x, y); words for how that ended.

    The drive has already arrived, so a failed turn goes into the reply instead of
    failing the step. A stop is raised, since nothing more may run under it.
    """
    what = f"the turn to {label}"
    try:
        start = ctx.base.pose()
    except StepError as error:
        if error.code is ErrorCode.CANCELLED:
            raise
        return f"but could not turn to face it: {error.message.rstrip('.')}"
    bearing = wrap_deg(math.degrees(math.atan2(y - start.y, x - start.x) - start.yaw))
    if abs(bearing) <= FACE_TOLERANCE_DEG:
        return f"and it is {_offset(bearing)}"
    try:
        _turn(ctx, start, bearing, what)
    except StepError as error:
        if error.code is ErrorCode.CANCELLED:
            raise
        return f"but could not turn to face it: {error.message.rstrip('.')}"
    return f"and turned {_round(abs(bearing), 0):.0f} degrees {_side(bearing)} to face it"


def _start_pose(ctx: Ctx, what: str) -> Pose2D | None:
    """The pose before the turn `what`, or None when it cannot be read.

    The start serves only to judge a turn that the base reports as ended without
    success, and face_person's reply, so a failed read is logged and the turn goes
    ahead. ctx.base.move_relative then raises its own refusal if it has one, such as
    the arm check before a drive (ADR-0025), a goal still live or its own failed pose
    read, and that refusal says what to fix where the failed read would not. A stop
    is raised at once, since nothing more may run under it.
    """
    try:
        return ctx.base.pose()
    except StepError as error:
        if error.code is ErrorCode.CANCELLED:
            raise
        logger.info("Starting %s without its start pose: %s", what, error)
        return None


def _turn(ctx: Ctx, start: Pose2D | None, turn_deg: float, what: str) -> Pose2D:
    """Turn on the spot by turn_deg from start, and return the pose it ended at.

    A turn that the base reports as ended without success still counts as done if it
    ended within FACE_TOLERANCE_DEG of its goal. The callers send only turns whose
    goal, wrapped to [-180, 180), is further than that tolerance from start, so a
    refused goal, which leaves the robot at start, still raises. With start None the
    goal is unknown, and when the end pose cannot be read the end is, so in either case
    such a turn raises the base's failure as it is.
    """
    try:
        return ctx.base.move_relative(0.0, 0.0, turn_deg)
    except GoalFailed as error:
        if start is None:
            raise
        # Only a goal known to have ended: with its state unknown the base may still
        # be turning, so a pose read now says nothing about where it stops (S2-9).
        try:
            end = ctx.base.pose()
        except StepError as read_error:
            if read_error.code is ErrorCode.CANCELLED:
                raise
            # Without its end the turn cannot be judged, so the base's failure stands.
            logger.info("Could not judge %s by its end pose: %s", what, read_error)
            raise error from None
        missed = wrap_deg(math.degrees(start.yaw) + turn_deg - math.degrees(end.yaw))
        if abs(missed) > FACE_TOLERANCE_DEG:
            raise
        logger.info("Counting %s as done, %.1f degrees off: %s", what, abs(missed), error)
        return end


def _offset(bearing_deg: float) -> str:
    """Plain words for a small bearing, for example "3 degrees to the left"."""
    shown = _round(abs(bearing_deg), 0)
    if not shown:
        return "straight ahead"
    return f"{shown:.0f} degrees to the {_side(bearing_deg)}"


def _side(angle_deg: float) -> str:
    return "left" if angle_deg > 0 else "right"


def _describe_move(forward_m: float, left_m: float, turn_deg: float) -> str:
    """Plain words for a relative move, for example "1 m forward, 0.5 m right"."""
    parts = []
    if forward_m:
        parts.append(f"{abs(forward_m):g} m {'forward' if forward_m > 0 else 'backward'}")
    if left_m:
        parts.append(f"{abs(left_m):g} m {'left' if left_m > 0 else 'right'}")
    if turn_deg:
        parts.append(f"turning {abs(turn_deg):g} degrees {'left' if turn_deg > 0 else 'right'}")
    return ", ".join(parts)


def _round(value: float, digits: int) -> float:
    """Round for display, turning -0.0 into 0.0 so it never prints as "-0"."""
    return round(value, digits) + 0.0
