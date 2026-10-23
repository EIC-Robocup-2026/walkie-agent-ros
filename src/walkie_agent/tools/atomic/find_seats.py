"""Find the places to sit in front of the camera, each free or taken.

Pairs the newest /walkie_vision/objects array with the /walkie_vision/people
array of the same image, so who sits where is judged in one image (hri.md,
Seats). Seat classes, sofa places and occupancy are walkie-agent-v2's rules
(walkie_agent.perception.seating).
"""

from pydantic import BaseModel
from walkie_vision_msgs.msg import ObjectDetectionArray

from walkie_agent.perception.seating import SeatingConfig, find_seat_places
from walkie_agent.toolbox import Effect, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _perception as shared


def _seat_list(seating: SeatingConfig) -> str:
    return ", ".join(seating.seat_classes)


# The description reads the config once, at import, since the registry
# fixes META when it loads the tool; run() reads it on every call.
_SEATING = shared.CONFIG.seating
_SEATS = _seat_list(_SEATING)
_SOFAS = ", ".join(c for c in _SEATING.seat_classes if _SEATING.is_sofa(c))
_SOFA_PLACES = "left, middle and right" if _SEATING.sofa_has_middle else "left and right"

META = ToolMeta(
    name="find_seats",
    description=(
        "Look at the seats in front of the camera now and see which are free. Returns a "
        "seat handle for each place to sit, such as seat_51c0.0, left to right as you see "
        f"them. Seats are: {_SEATS}. A {_SOFAS} has a {_SOFA_PLACES} place, each with its "
        "own handle. Each line gives the handle, the seat, the place on a sofa, free or "
        "taken, and where the place is in the image (left, centre or right). The last "
        "line counts the free places. A handle stays valid for 10 minutes while you do "
        "not drive; turning keeps it."
    ),
    effects=frozenset({Effect.P}),
    questions=(
        "Where can I sit?",
        "Is there a free seat?",
        "Which seats are taken?",
        "Offer the guest a free seat.",
    ),
    examples=({},),
    requires=(shared.VISION,),
)


class Args(BaseModel):
    pass


class Out(ToolOutput):
    set_id: str
    handles: list[str]
    # The handles of the free places, left to right.
    free: list[str]


def run(ctx: ToolContext, args: Args) -> Out:
    config = shared.CONFIG
    feed = shared.vision_feed(ctx)
    objects = feed.latest_objects(config.max_age_s, config.wait_s, ctx.cancel)
    # walkie-vision gives both arrays of one image that image's header, so
    # the stamp finds the people of the same image. A newer people array
    # would compare boxes from two images, after the camera or the guests
    # may have moved.
    people = feed.people_at(objects.header.stamp, config.pair_wait_s, ctx.cancel).msg
    places = find_seat_places(objects.detections, people.people, config.seating)
    if not places:
        raise StepError("NOT_FOUND", _nothing_to_sit_on(objects, config.seating), {"seats": 0})

    width = objects.camera_info.width
    heading_maps = shared.headings(
        [place.box for place in places], objects.camera_info, objects.header, ctx.tf_buffer
    )
    payloads = [
        shared.SeatPayload(
            position=shared.image_position((place.box[0] + place.box[2]) / 2, width),
            box=place.box,
            label=place.label,
            part=place.part,
            occupied=place.occupied,
            header=objects.header,
            image_width=int(width),
            image_height=int(objects.camera_info.height),
            heading_map=heading,
        )
        for place, heading in zip(places, heading_maps, strict=True)
    ]
    handle_set = shared.put_handles(ctx, "seat", payloads, objects.header)

    lines = []
    free = []
    for handle_id, payload in zip(handle_set.member_ids, payloads, strict=True):
        parts = [payload.label]
        if payload.part is not None:
            parts.append(f"{payload.part} place")
        parts.append("taken" if payload.occupied else "free")
        parts.append(payload.position)
        lines.append(f"{handle_id}: {', '.join(parts)}")
        if not payload.occupied:
            free.append(handle_id)
    lines.append(f"Free places: {len(free)} of {len(payloads)}.")
    return Out(
        summary="\n".join(lines),
        set_id=handle_set.id,
        handles=list(handle_set.member_ids),
        free=free,
    )


def _nothing_to_sit_on(objects: ObjectDetectionArray, seating: SeatingConfig) -> str:
    """The NOT_FOUND message: the seat classes, and what is in view instead.

    It takes the config run() used, so the message names the classes that
    were actually looked for.
    """
    labels = list(dict.fromkeys(d.label for d in objects.detections if d.label))
    seen = ", ".join(labels) if labels else "no objects"
    message = f"No seat in view. Seats are {_seat_list(seating)}; in view: {seen}."
    # walkie-vision detects only the classes it was prompted with (hri.md,
    # question 1), so a list without a seat class explains an empty answer.
    if objects.classes and not any(seating.is_seat(c) for c in objects.classes):
        message += " The detector is not looking for any seat class."
    return message
