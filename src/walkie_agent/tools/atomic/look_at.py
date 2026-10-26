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

ctx.gaze runs these steps (_gaze.RosGaze): look_at() takes the target, and
face() waits for the base to face it.
"""

from pydantic import BaseModel, Field

from walkie_agent.capabilities import Looking
from walkie_agent.gaze import DEFAULT_CONFIG
from walkie_agent.toolbox import Effect, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _gaze as shared
from walkie_agent.tools.atomic._perception import VISION

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
# they scale with a simulation that runs slower than real time; ctx.gaze's
# face() also ends a wait at a backstop on the monotonic clock.

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
    looking = ctx.gaze.look_at(args.target)
    target = looking.target
    note = "".join(f" {line}" for line in looking.notes)
    if looking.kind == "person":
        # Past the limit the loop goes on turning toward them, as the answer says.
        facing = _face(ctx, looking, PERSON_TURN_S)
        late = "" if facing else " The base does not face them yet and is still turning."
        summary = (
            f"Looking at {target}.{late} The robot turns to keep them in view, and stops "
            f"once it has not seen them for {looking.lost_after_s:g} s.{note}"
        )
    else:
        timeout_s = TURN_TIMEOUT_S
        if not _face(ctx, looking, timeout_s):
            # A base that cannot turn, or a TF that stopped, must not keep
            # the loop commanding.
            ctx.gaze.stop(shared.TURN_TIMEOUT)
            raise StepError(
                "TIMEOUT",
                f"The robot did not face {target} within {timeout_s:g} s, so it stopped "
                "turning.",
                {"handle": target},
            )
        facing = True
        summary = f"Turned toward {target} ({_seat_words(looking)}).{note}"
    return Out(
        summary=summary,
        target=target,
        kind=looking.kind,
        heading=ctx.gaze.status().setpoint,
        facing=facing,
        arms_unknown=bool(looking.notes),
    )


def _face(ctx: ToolContext, looking: Looking, timeout_s: float) -> bool:
    """Wait until the base faces `looking`'s target (Gaze.face); looked up on each call."""
    return ctx.gaze.face(looking, timeout_s)


def _seat_words(seat: Looking) -> str:
    """The seat and its place, such as "sofa, left", or "chair" without a place."""
    return seat.seat_label if seat.seat_part is None else f"{seat.seat_label}, {seat.seat_part}"
