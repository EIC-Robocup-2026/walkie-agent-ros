"""Shared by the gaze atomics: the backend name, the arm-motion stop, stop causes and ctx.gaze.

look_at and stop_looking reach the gaze loop, walkie_agent.gaze.GazeLoop,
only through the `gaze` backend (ADR-0051), and move_arm_to_pose stops it
through the same backend before an arm moves. RosGaze is ctx.gaze on that
loop: look_at's steps behind one interface (walkie_agent.capabilities.Gaze,
ADR-0052). The module's name starts with "_", so the registry does not load
it as a tool.
"""

import time

from rclpy.time import Time
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion

from walkie_agent.capabilities import Looking
from walkie_agent.gaze import (
    ARM_MOTION,
    ARM_OUT,
    ERROR,
    INTERRUPT,
    LOST,
    NAVIGATION,
    NAVIGATION_ACTIVE,
    NEW_TASK,
    SHUTDOWN,
    STOP_HEARD,
    STOP_LOOKING,
    STOP_UNCONFIRMED_NEXT,
    STOP_WAIT_S,
    GazeLoop,
    GazeStatus,
    wrap_angle,
)
from walkie_agent.toolbox import Handle, StepError, ToolContext
from walkie_agent.tools.atomic._arm import ARM
from walkie_agent.tools.atomic._perception import PersonPayload, SeatPayload
from walkie_agent.tools.navigation import MAP_FRAME, ROBOT_FRAME, check_arms

# The backend name in META.requires and ctx.backend().
GAZE = "gaze"

# look_at's stop reason when the base does not face a seat in time.
TURN_TIMEOUT = "turn timeout"

# What follows the loop's message in an arm atomic's refusal when
# stop_before_arm_motion raises.
NO_ARM_GOAL = f"No arm goal was sent. {STOP_UNCONFIRMED_NEXT}"

# Why the loop stopped, as the end of "The last target, per_51c0.1, was
# stopped 2 s ago ..." and of "Stopped turning toward seat_77aa.2 ...".
# LOST is not here: stop_looking says "was lost" instead.
_CAUSES = {
    STOP_LOOKING: "with stop_looking",
    INTERRUPT: "by a stop or a reset",
    STOP_HEARD: 'because someone said "stop" or "never mind"',
    NEW_TASK: "when a new task started",
    # Before the drive's arm check, so also for a drive that never ran.
    NAVIGATION: "when a drive was asked for",
    NAVIGATION_ACTIVE: "because a navigation goal became active",
    # Before an arm atomic's first commander call, and from the loop when an
    # arm became unknown, when a commander goal ran with an arm unknown,
    # which may be a gripper's (busy()), and while the arms were unverified,
    # after a goal that may still move an arm or a gripper was abandoned.
    ARM_MOTION: "because an arm or a gripper was about to move, or may have been moving",
    # From the commander's last reading, which does not say when the arm went out.
    ARM_OUT: "because an arm was out of every pose that is safe for driving",
    SHUTDOWN: "at shutdown",
    ERROR: "after an error in the gaze loop",
    TURN_TIMEOUT: "because the turn took too long",
}

# The error for a target that the loop stopped before look_at answered, a
# seat before the base faced it or a person before the base first faced
# them, by its stop reason. Any other reason, such as an interrupt or a
# newer target, is CANCELLED. A person lost before look_at answers is STALE
# (lost_at_start).
_ENDED_CODES = {
    NAVIGATION_ACTIVE: "REFUSED",
    ARM_OUT: "PRECONDITION_FAILED",
    ARM_MOTION: "PRECONDITION_FAILED",
    SHUTDOWN: "UNAVAILABLE",
    ERROR: "INTERNAL",
}

# Both of look_at's limits are on the node clock, which follows /clock under
# sim time, so they scale with a simulation that runs slower than real time.
# A wait also ends this many times its limit after it began on the monotonic
# clock, for a clock that stalls or moves back, as at a simulation restart.
_BACKSTOP_FACTOR = 3.0
# How often the wait for the base to face a seat or a person reads TF and
# the loop's status.
_POLL_S = 0.05


def gaze_loop(ctx: ToolContext) -> GazeLoop:
    return ctx.backend(GAZE)


def stop_before_arm_motion(ctx: ToolContext) -> None:
    """Stop the gaze loop, when the agent has one, before an arm atomic moves an arm.

    The loop turns the base only with both arms in travel-safe poses
    (ADR-0051). Stopping before the goal is sent, rather than when the
    commander's next reading shows the arm moving, keeps the base still for
    the whole motion. The gaze thread publishes the stop's zeros, and this
    waits up to STOP_WAIT_S for them, so the base has been told to stop
    before the arm leaves its pose. When they are not published by then,
    it raises the loop's StepError TIMEOUT: the base may still be turning,
    so the caller refuses and sends nothing to the commander; its answer
    follows the loop's message with NO_ARM_GOAL. The loop is stopped either
    way. The grippers do not move the arm and do not stop it.
    """
    if ctx.has_backend(GAZE):
        gaze_loop(ctx).stop(ARM_MOTION, wait_s=STOP_WAIT_S)


def stop_cause(reason: str) -> str:
    """Why the loop stopped for `reason`, as the words that end a sentence about the stop."""
    return _CAUSES.get(reason, f"({reason})")


def look_target(ctx: ToolContext, target: str) -> Handle:
    """The person or seat handle `target`, if the loop can turn toward it.

    Its payload is a PersonPayload or a SeatPayload, by its kind. Raises
    the store's NOT_FOUND, STALE or UNAVAILABLE, NOT_FOUND for a handle of
    another kind, REFUSED for a person who is not tracked, and UNAVAILABLE
    for a seat seen without a heading.
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
    payload = handle.payload
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
    return handle


class RosGaze:
    """ctx.gaze on the agent's GazeLoop (walkie_agent.capabilities.Gaze).

    A person handle starts the loop on the person's track: it turns toward
    each new sighting until a stop, a new task, a drive, an arm motion, or
    lost_after_s after the last sighting. A seat handle turns the base to
    the seat's heading, fixed in the map when find_seats saw it, and holds
    it. The base turns only with both arms in travel-safe poses: the check
    is navigation.check_arms, the one the drives make. Every method of the
    loop is looked up on each call, so a test that replaces one on the loop
    applies here too.
    """

    def __init__(self, ctx: ToolContext, loop: GazeLoop) -> None:
        self._ctx = ctx
        self._loop = loop

    def look_at(self, handle: str) -> Looking:
        """Start the loop on `handle`; return once it has the target, before the base turns."""
        ctx, loop = self._ctx, self._loop
        # First, before the arm check, which may wait up to 3 s for the
        # commander: a stop anywhere in the call, stop_looking or "stop" heard
        # included, then makes the turn below raise CANCELLED rather than turn
        # the base after the stop's zeros.
        stop_calls = loop.stop_calls()
        # The handle is checked before the arms, so a handle that cannot be
        # looked at is refused without a call to the commander.
        target = look_target(ctx, handle)
        payload = target.payload
        arm = ctx.backend(ARM) if ctx.has_backend(ARM) else None
        # The action is this call's turn: a refusal reads "Not starting a turn
        # toward per_51c0.1", which stays true while the loop still follows a
        # target from an earlier call, this one included, as it does during a
        # gripper command with both arms known. It says nothing of that target,
        # which the loop may stop at its next tick, on the reading this check
        # just made.
        note = check_arms(
            arm, f"starting a turn toward {target.id}", "'carry' or 'home'", ctx.cancel
        )
        notes = (note.removeprefix(" "),) if note else ()
        if target.kind == "person":
            person: PersonPayload = payload
            loop.follow_person(
                target.id,
                person.track,
                person.heading_map,
                person.stamp_ns,
                cancel=ctx.cancel,
                stop_calls=stop_calls,
            )
            _check_following(loop, target.id)
            return Looking(
                target=target.id,
                kind="person",
                notes=notes,
                lost_after_s=loop.config.lost_after_s,
                seat_label=None,
                seat_part=None,
            )
        seat: SeatPayload = payload
        loop.hold_heading(target.id, seat.heading_map, cancel=ctx.cancel, stop_calls=stop_calls)
        return Looking(
            target=target.id,
            kind="seat",
            notes=notes,
            lost_after_s=None,
            seat_label=seat.label,
            seat_part=seat.part,
        )

    def face(self, looking: Looking, timeout_s: float) -> bool:
        """Wait until the base faces the loop's setpoint for `looking`, within the loop's deadband.

        True once it does, False when it does not within `timeout_s` on the
        node clock, or within _BACKSTOP_FACTOR times that on the monotonic
        clock; the loop goes on either way, and the caller decides what a
        late turn means. For a person the setpoint moves with each sighting,
        and a person the loop loses during the wait is STALE
        (_still_following). Raises CANCELLED on a stop or a reset, and the
        loop's own reason when it ended the turn.
        """
        ctx, loop = self._ctx, self._loop
        name = looking.target
        person = looking.kind == "person"
        start = ctx.now().nanoseconds
        backstop = time.monotonic() + _BACKSTOP_FACTOR * timeout_s
        while True:
            if ctx.cancel.is_set():
                # The interrupt listener stops the loop too; whichever comes
                # first, the base stops.
                loop.stop(INTERRUPT)
                ctx.check_cancelled()
            if person:
                status = _still_following(loop, name)
            else:
                status = loop.status()
                if not status.running or status.target != name:
                    raise turn_ended(name, status)
            yaw = _base_yaw(ctx, loop.config.yaw_max_age_s)
            if (
                yaw is not None
                and status.setpoint is not None
                and abs(wrap_angle(status.setpoint - yaw)) <= loop.config.deadband_rad
            ):
                return True
            # A clock that moved back gives a negative time, which leaves the
            # end to the backstop.
            waited_s = (ctx.now().nanoseconds - start) / 1e9
            if waited_s >= timeout_s or time.monotonic() >= backstop:
                return False
            ctx.cancel.wait(_POLL_S)

    def stop(self, reason: str | None = None) -> str | None:
        """Stop the loop, for STOP_LOOKING by default; the target it stopped, or None."""
        return self._loop.stop(reason or STOP_LOOKING)

    def status(self) -> GazeStatus:
        return self._loop.status()


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
    turn_ended gives.
    """
    status = gaze.status()
    lost_after_s = gaze.config.lost_after_s
    if status.running and status.target == name and status.last_seen is not None:
        if abs(status.now - status.last_seen) <= lost_after_s:
            return status
        gaze.stop(LOST)
    elif status.target != name or status.stop_reason != LOST or status.last_seen is None:
        raise turn_ended(name, status)
    raise lost_at_start(name, status.now - status.last_seen, lost_after_s)


def _still_following(gaze: GazeLoop, name: str) -> GazeStatus:
    """The loop's status while face() waits; raise once the loop no longer follows `name`.

    The loss rule is left to the loop's own tick, which sights the arrays
    that came since its last tick before it applies the rule, so a poll
    between two ticks does not lose a person the next tick still sees. A
    person the loop lost is STALE, as at the start (_check_following); any
    other end gets the error turn_ended gives.
    """
    status = gaze.status()
    if status.running and status.target == name:
        return status
    if status.target == name and status.stop_reason == LOST and status.last_seen is not None:
        raise lost_at_start(name, status.now - status.last_seen, gaze.config.lost_after_s)
    raise turn_ended(name, status)


def lost_at_start(name: str, age: float, lost_after_s: float) -> StepError:
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


def turn_ended(name: str, status: GazeStatus) -> StepError:
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
        f"Stopped turning toward {name} {stop_cause(reason)}.",
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
