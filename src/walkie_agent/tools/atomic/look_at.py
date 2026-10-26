"""Turn the base toward a person and follow them, or toward a seat and hold (ADR-0051).

The head only tilts, so the robot looks at someone by turning its base, and
the gaze loop (walkie_agent.gaze) does the turning on a thread of its own.
A person handle starts the loop on the person's track: it turns toward each
new sighting until stop_looking, a stop or a reset, a new task, a drive, an
arm motion, or 1 s after the last sighting. The answer comes once the base
first faces them, so a speak that follows is said to them, not on the way to
them, or after PERSON_TURN_S if it does not, as for someone who keeps
walking, and then says that the base is still turning; either way the loop
goes on following them for as long as the conversation lasts. Both waits
are on the node clock, so they scale with a slow simulation. A person
already lost when the loop takes them, or lost before the base faces them,
last seen more than 1 s before, is refused with STALE rather than answered
with "Looking at".

A seat handle turns the base to the seat's heading, fixed in the map when
find_seats saw it, so it is never looked up again after the TF buffer has
dropped that moment, and holds it. That answer comes once the base faces
the seat, so a find_people or a speak that follows sees, or points at, the
right place.

The base turns only with both arms in travel-safe poses: the check is
navigation.check_arms, the one the drives make, and it fails open the same
way when the commander is not found, does not answer or gives no reading,
except while the arms are unverified. It refuses while a commander goal may
still move an arm or a gripper, and on a reading that cannot show the arms
are safe.
"""

import time

from pydantic import BaseModel, Field
from rclpy.time import Time
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion

from walkie_agent.gaze import (
    ARM_MOTION,
    ARM_OUT,
    DEFAULT_CONFIG,
    ERROR,
    INTERRUPT,
    LOST,
    NAVIGATION_ACTIVE,
    SHUTDOWN,
    GazeLoop,
    GazeStatus,
    wrap_angle,
)
from walkie_agent.toolbox import Effect, Handle, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _gaze as shared
from walkie_agent.tools.atomic._arm import ARM
from walkie_agent.tools.atomic._perception import VISION, SeatPayload
from walkie_agent.tools.navigation import MAP_FRAME, ROBOT_FRAME, check_arms

# How long a turn to a seat may take. A half turn at the loop's 0.6 rad/s,
# with the ramp up and the slow approach to the deadband, takes about 6 s.
# Read on every call, so a test can shorten it.
TURN_TIMEOUT_S = 10.0
# How long look_at waits for the base to first face a person before it
# answers anyway, with the loop still following them. Someone at the edge of
# the image is about 55 degrees off, a turn of about 2.2 s at the loop's
# settings. Read on every call, so a test can change it.
PERSON_TURN_S = 4.0
# Both limits are on the node clock, which follows /clock under sim time, so
# they scale with a simulation that runs slower than real time. A wait also
# ends this many times its limit after it began on the monotonic clock, for
# a clock that stalls or moves back, as at a simulation restart.
_BACKSTOP_FACTOR = 3.0
# How often the wait for the base to face a seat or a person reads TF and
# the loop's status.
_POLL_S = 0.05

META = ToolMeta(
    name="look_at",
    description=(
        "Turn to face a person or a seat. target is one person handle from find_people, "
        "such as per_51c0.1, or one seat handle from find_seats. For a person, the robot "
        "keeps turning to follow them until you call stop_looking, someone says "
        '"stop" or "never mind", they are out of view for '
        f"{DEFAULT_CONFIG.lost_after_s:g} s, or you drive or move an arm; the answer comes once "
        f"it faces them, or after {PERSON_TURN_S:g} s while it still turns, and then says so. "
        "For a seat, the robot turns to it and stays facing it; the answer comes once it "
        "faces the seat. Look at the person you are talking to while you talk to them. A "
        "drive or an arm move stops it by itself. Call stop_looking only when someone asks "
        "you to stop turning, or when you stay where you are after your last line to them."
    ),
    effects=frozenset({Effect.N}),
    questions=(
        "Look at me.",
        "Look at the guest you are talking to.",
        "Turn to the person on the left.",
        "Face the free seat.",
        "Show the guest where to sit.",
    ),
    examples=({"target": "per_51c0.1"}, {"target": "seat_77aa.2"}),
    # Every handle look_at takes comes from a vision atomic, and a person
    # is followed through the vision feed.
    requires=(shared.GAZE, VISION),
)

# The error for a target that the loop stopped before look_at answered, a
# seat before the base faced it or a person before the base first faced
# them, by its stop reason. Any other reason, such as an interrupt or a
# newer target, is CANCELLED. A person lost before look_at answers is STALE
# (_lost_at_start).
_ENDED_CODES = {
    NAVIGATION_ACTIVE: "REFUSED",
    ARM_OUT: "PRECONDITION_FAILED",
    ARM_MOTION: "PRECONDITION_FAILED",
    SHUTDOWN: "UNAVAILABLE",
    ERROR: "INTERNAL",
}


class Args(BaseModel):
    target: str = Field(
        description=(
            "One person handle from find_people, such as per_51c0.1, or one seat handle "
            "from find_seats, such as seat_77aa.2."
        )
    )


class Out(ToolOutput):
    target: str
    # "person", followed until lost or stopped, or "seat", a heading held.
    kind: str
    # The map-frame yaw the base turns toward now, in radians.
    heading: float | None
    # True when the base faced the target, within the loop's deadband,
    # before the answer; False for a person still turned toward after
    # PERSON_TURN_S.
    facing: bool
    # True when the arm backend is present but the arms were not checked:
    # the commander was not found, did not answer or gave no reading. The
    # base turns with the arms unchecked.
    arms_unknown: bool


def run(ctx: ToolContext, args: Args) -> Out:
    gaze = shared.gaze_loop(ctx)
    # First, before the arm check, which may wait up to 3 s for the
    # commander: a stop anywhere in the call, stop_looking or "stop" heard
    # included, then makes the turn below raise CANCELLED rather than turn
    # the base after the stop's zeros.
    stop_calls = gaze.stop_calls()
    handle = _handle(ctx, args.target)
    payload = handle.payload
    # The handle is checked before the arms, so a handle that cannot be
    # looked at is refused without a call to the commander.
    if handle.kind == "person" and payload.track is None:
        raise StepError(
            "REFUSED",
            f"{handle.id} is not tracked, cannot follow. Call find_people again for a handle "
            "the robot can follow.",
            {"handle": handle.id},
        )
    if handle.kind == "seat" and payload.heading_map is None:
        raise StepError(
            "UNAVAILABLE",
            f"No heading toward {handle.id}: TF had no camera position when it was seen. "
            "Call find_seats again.",
            {"handle": handle.id},
        )

    arm = ctx.backend(ARM) if ctx.has_backend(ARM) else None
    # The action is this call's turn: a refusal reads "Not starting a turn
    # toward per_51c0.1", which stays true while the loop still follows a
    # target from an earlier call, this one included, as it does during a
    # gripper command with both arms known. It says nothing of that target,
    # which the loop may stop at its next tick, on the reading this check
    # just made.
    note = check_arms(
        arm, f"starting a turn toward {handle.id}", "'carry' or 'home'", ctx.cancel
    )

    if handle.kind == "person":
        gaze.follow_person(
            handle.id,
            payload.track,
            payload.heading_map,
            payload.stamp_ns,
            cancel=ctx.cancel,
            stop_calls=stop_calls,
        )
        _check_following(gaze, handle.id)
        # Past the limit the loop goes on turning toward them, as the answer says.
        facing = _face(ctx, gaze, handle.id, PERSON_TURN_S, person=True)
        late = "" if facing else " The base does not face them yet and is still turning."
        summary = (
            f"Looking at {handle.id}.{late} The robot turns to keep them in view, and stops "
            f"once it has not seen them for {gaze.config.lost_after_s:g} s.{note}"
        )
    else:
        gaze.hold_heading(
            handle.id, payload.heading_map, cancel=ctx.cancel, stop_calls=stop_calls
        )
        timeout_s = TURN_TIMEOUT_S
        if not _face(ctx, gaze, handle.id, timeout_s, person=False):
            # A base that cannot turn, or a TF that stopped, must not keep
            # the loop commanding.
            gaze.stop(shared.TURN_TIMEOUT)
            raise StepError(
                "TIMEOUT",
                f"The robot did not face {handle.id} within {timeout_s:g} s, so it stopped "
                "turning.",
                {"handle": handle.id},
            )
        facing = True
        summary = f"Turned toward {handle.id} ({_seat_words(payload)}).{note}"
    return Out(
        summary=summary,
        target=handle.id,
        kind=handle.kind,
        heading=gaze.status().setpoint,
        facing=facing,
        arms_unknown=bool(note),
    )


def _handle(ctx: ToolContext, target: str) -> Handle:
    """The person or seat handle `target`, or the store's NOT_FOUND, STALE or UNAVAILABLE.

    Its payload is a PersonPayload or a SeatPayload, by its kind.
    """
    hint = (
        " look_at takes one person handle from find_people, such as per_51c0.1, or one seat "
        "handle from find_seats."
    )
    try:
        handle = ctx.handles.get(target)
    except StepError as e:
        if e.code == "NOT_FOUND":
            raise StepError("NOT_FOUND", e.message + hint, e.data) from None
        raise
    if handle.kind not in ("person", "seat"):
        raise StepError(
            "NOT_FOUND",
            f"{handle.id} is a handle of kind {handle.kind}, not a person or a seat.{hint}",
            {"handle": handle.id},
        )
    return handle


def _seat_words(seat: SeatPayload) -> str:
    """The seat and its place, such as "sofa, left", or "chair" without a place."""
    return seat.label if seat.part is None else f"{seat.label}, {seat.part}"


def _check_following(gaze: GazeLoop, name: str) -> GazeStatus:
    """The loop's status; raise unless it follows `name` from a sighting it has not already lost.

    follow_person checks the age of the sighting it starts from before it
    sets the target, and the newest array's before its wait for TF, so by
    the time the target is set, that sighting may be more than
    lost_after_s old, and the loop's first tick would lose the person. This
    applies the loop's rule (GazeControl._lost) once, at the start, when
    follow_person has just cleared the arrays the loop has not sighted yet:
    a target lost then is stopped for LOST, as the loop's first tick would
    stop it, and refused with STALE. A target that the loop already stopped
    for another reason, or that another target replaced, gets the error
    _turn_ended gives.
    """
    status = gaze.status()
    lost_after_s = gaze.config.lost_after_s
    if status.running and status.target == name and status.last_seen is not None:
        if abs(status.now - status.last_seen) <= lost_after_s:
            return status
        gaze.stop(LOST)
    elif status.target != name or status.stop_reason != LOST or status.last_seen is None:
        raise _turn_ended(name, status)
    raise _lost_at_start(name, status.now - status.last_seen, lost_after_s)


def _still_following(gaze: GazeLoop, name: str) -> GazeStatus:
    """The loop's status while _face waits; raise once the loop no longer follows `name`.

    The loss rule is left to the loop's own tick, which sights the arrays
    that came since its last tick before it applies the rule, so a poll
    between two ticks does not lose a person the next tick still sees. A
    person the loop lost is STALE, as at the start (_check_following); any
    other end gets the error _turn_ended gives.
    """
    status = gaze.status()
    if status.running and status.target == name:
        return status
    if status.target == name and status.stop_reason == LOST and status.last_seen is not None:
        raise _lost_at_start(name, status.now - status.last_seen, gaze.config.lost_after_s)
    raise _turn_ended(name, status)


def _lost_at_start(name: str, age: float, lost_after_s: float) -> StepError:
    """STALE for a person last seen `age` seconds ago, too long ago for the loop to follow."""
    if age < 0:
        # As GazeControl._lost: a clock that moved back loses the person too.
        why = (
            f"their last sighting is stamped {-age:.1f} s ahead of the robot's clock, which "
            "has moved back, as at a simulation restart."
        )
    else:
        why = (
            f"they were last seen {age:.1f} s ago. The robot stops following a person "
            f"{lost_after_s:g} s after the last sighting."
        )
    return StepError(
        "STALE",
        f"Not turning toward {name}: {why} Call find_people for a current handle.",
        {"handle": name},
    )


def _face(ctx: ToolContext, gaze: GazeLoop, name: str, timeout_s: float, *, person: bool) -> bool:
    """Wait until the base faces the loop's setpoint for `name`, within the loop's deadband.

    True once it does, False when it does not within `timeout_s` on the
    node clock, or within _BACKSTOP_FACTOR times that on the monotonic
    clock; the loop goes on either way, and the caller decides what a late
    turn means. For a person the setpoint moves with each sighting, and a
    person the loop loses during the wait is STALE (_still_following).
    Raises CANCELLED on a stop or a reset, and the loop's own reason when
    it ended the turn.
    """
    start = ctx.now().nanoseconds
    backstop = time.monotonic() + _BACKSTOP_FACTOR * timeout_s
    while True:
        if ctx.cancel.is_set():
            # The interrupt listener stops the loop too; whichever comes
            # first, the base stops.
            gaze.stop(INTERRUPT)
            ctx.check_cancelled()
        if person:
            status = _still_following(gaze, name)
        else:
            status = gaze.status()
            if not status.running or status.target != name:
                raise _turn_ended(name, status)
        yaw = _base_yaw(ctx, gaze.config.yaw_max_age_s)
        if (
            yaw is not None
            and status.setpoint is not None
            and abs(wrap_angle(status.setpoint - yaw)) <= gaze.config.deadband_rad
        ):
            return True
        # A clock that moved back gives a negative time, which leaves the
        # end to the backstop.
        waited_s = (ctx.now().nanoseconds - start) / 1e9
        if waited_s >= timeout_s or time.monotonic() >= backstop:
            return False
        ctx.cancel.wait(_POLL_S)


def _turn_ended(name: str, status: GazeStatus) -> StepError:
    """The error for a target `name` that the loop ended before look_at answered."""
    reason = status.stop_reason if status.target == name else None
    if reason is None:
        return StepError(
            "CANCELLED",
            f"Stopped turning toward {name}: another target replaced it.",
            {"handle": name},
        )
    return StepError(
        _ENDED_CODES.get(reason, "CANCELLED"),
        f"Stopped turning toward {name} {shared.stop_cause(reason)}.",
        {"handle": name, "reason": reason},
    )


def _base_yaw(ctx: ToolContext, max_age_s: float) -> float | None:
    """The base's yaw in the map from the newest TF, read as the gaze loop reads it.

    None when TF has no map -> base_footprint, or when the newest is more
    than `max_age_s` from the node clock: the loop commands nothing then,
    and an old yaw would end the wait while the base is elsewhere.
    """
    try:
        stamped = ctx.tf_buffer.lookup_transform(MAP_FRAME, ROBOT_FRAME, Time())
    except TransformException:
        return None
    stamp_ns = Time.from_msg(stamped.header.stamp).nanoseconds
    # Stamp 0 is a chain of static transforms, as in a test, which never
    # goes stale.
    if stamp_ns and abs(ctx.now().nanoseconds - stamp_ns) / 1e9 > max_age_s:
        return None
    q = stamped.transform.rotation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw
