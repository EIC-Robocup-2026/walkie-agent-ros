"""Find the people in front of the camera, with their gestures and posture.

Reads the newest /walkie_vision/people array through the `vision` backend
and keeps one person handle per person, left to right (ADR-0050). Gestures
and postures come from the keypoints, by walkie-agent-v2's rules
(walkie_agent.perception.gestures).
"""

from typing import NamedTuple

from pydantic import BaseModel, Field
from rclpy.time import Time
from walkie_vision_msgs.msg import PersonDetection, PersonDetectionArray

from walkie_agent.perception.gestures import GESTURES, POSTURES, classify_gestures
from walkie_agent.perception.seating import BBox, bbox_xyxy
from walkie_agent.perception.vision_feed import (
    TrackKey,
    VisionFeed,
    face_trusted,
    model_vector,
)
from walkie_agent.toolbox import Effect, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _perception as shared

_GESTURE_LIST = ", ".join(GESTURES)

# What each person's line says of their face. Only a seen face can name the
# person; one too small or unclear is detected but never matched.
FACE_SEEN = "face seen"
FACE_UNCLEAR = "face too small or unclear to recognize"
NO_FACE = "no face seen"

META = ToolMeta(
    name="find_people",
    description=(
        "Look at the people in front of the camera now. Returns a person handle for "
        "each, such as per_51c0.0, left to right as you see them. Each line gives the "
        "handle, where the person is in the image (left, centre or right), their height "
        "in the image as a share of its height (larger is usually nearer), their posture "
        "and gestures, and whether their face is seen well enough to recognize them: "
        f"{FACE_SEEN}, {FACE_UNCLEAR}, or {NO_FACE}. "
        "The first line names the set, such as per_51c0, which recognize_person takes "
        "to recognize everyone at once. With gesture, only the people who show it: one "
        f"of {_GESTURE_LIST}. raising_left_arm and raising_right_arm mean the person's "
        "own arm; pointing_left and pointing_right are directions as you see them. A "
        "handle stays valid for 10 minutes while you do not drive; turning keeps it."
    ),
    effects=frozenset({Effect.P}),
    questions=(
        "Who is in front of you?",
        "Is anyone waving at you?",
        "Find the person raising their left arm.",
        "How many people are sitting down?",
        "Is a guest standing at the door?",
    ),
    examples=({}, {"gesture": "waving"}),
    requires=(shared.VISION,),
)


class Args(BaseModel):
    gesture: str | None = Field(
        default=None,
        description=(
            f"Only people who show this gesture or posture, one of {_GESTURE_LIST}. "
            "Leave it out to find everyone."
        ),
    )


class Out(ToolOutput):
    set_id: str
    handles: list[str]


class _Seen(NamedTuple):
    """One person in the array, as find_people reads them."""

    person: PersonDetection
    # The key of the person's track collection, from the PeopleFrame.
    track: TrackKey | None
    position: str
    gestures: frozenset[str]


def run(ctx: ToolContext, args: Args) -> Out:
    config = shared.CONFIG
    # Checked first, so a misspelt gesture fails at once and not after a wait.
    gesture = _gesture_name(args.gesture)
    feed = shared.vision_feed(ctx)
    frame = feed.latest_people(config.max_age_s, config.wait_s, ctx.cancel)
    msg = frame.msg
    if not msg.people:
        raise StepError("NOT_FOUND", "Nobody is in view.")

    # Left to right by the box's centre, as the LLM reads the lines. The
    # index breaks ties, so the order never depends on the sort.
    order = sorted(
        range(len(msg.people)), key=lambda i: (msg.people[i].bbox.center.position.x, i)
    )
    width = msg.camera_info.width
    seen = [
        _Seen(
            person=msg.people[i],
            track=frame.tracks[i],
            position=shared.image_position(msg.people[i].bbox.center.position.x, width),
            gestures=frozenset(
                classify_gestures(
                    msg.people[i].keypoints, msg.people[i].keypoint_scores, config.gestures
                )
            ),
        )
        for i in order
    ]
    chosen = [entry for entry in seen if gesture is None or gesture in entry.gestures]
    if not chosen:
        in_view = "; ".join(_describe(entry.position, entry.gestures) for entry in seen)
        raise StepError(
            "NOT_FOUND",
            f"Nobody in view shows {gesture}. In view, left to right: {in_view}.",
            {"gesture": gesture, "people": len(seen)},
        )

    face_boxes = [_face_box(entry.person) for entry in chosen]
    # Toward the face when there is one, since the gaze loop looks at it.
    heading_maps = shared.headings(
        [
            face or bbox_xyxy(entry.person.bbox)
            for entry, face in zip(chosen, face_boxes, strict=True)
        ],
        msg.camera_info,
        msg.header,
        ctx.tf_buffer,
    )
    stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
    payloads = [
        _payload(feed, msg, stamp_ns, entry, face_box, heading)
        for entry, face_box, heading in zip(chosen, face_boxes, heading_maps, strict=True)
    ]
    handle_set = shared.put_handles(ctx, "person", payloads, msg.header)

    if gesture is None:
        lines = [f"{handle_set.id}: {_count(len(seen))} in view, left to right."]
    else:
        verb = "shows" if len(chosen) == 1 else "show"
        lines = [
            f"{handle_set.id}: {len(chosen)} of {_count(len(seen))} in view {verb} "
            f"{gesture}, left to right."
        ]
    image_height = msg.camera_info.height
    for handle_id, payload, entry in zip(handle_set.member_ids, payloads, chosen, strict=True):
        share = (payload.bbox[3] - payload.bbox[1]) / image_height if image_height else None
        description = _describe(payload.position, payload.gestures, share)
        lines.append(f"{handle_id}: {description}, {_face_text(payload)}")
    return Out(summary="\n".join(lines), set_id=handle_set.id, handles=list(handle_set.member_ids))


def _gesture_name(gesture: str | None) -> str | None:
    """The gesture as GESTURES spells it, or None for everyone; REFUSED when unknown."""
    if gesture is None or not gesture.strip():
        return None
    # The LLM may write "Waving" or "raising left arm".
    name = gesture.strip().lower().replace(" ", "_").replace("-", "_")
    if name not in GESTURES:
        raise StepError(
            "REFUSED",
            f"{gesture!r} is not a gesture find_people knows. Use one of {_GESTURE_LIST}.",
            {"gesture": gesture},
        )
    return name


def _describe(position: str, gestures: frozenset[str], share: float | None = None) -> str:
    """Such as "left, height 0.62, standing, waving": the posture first, then the arms."""
    parts = [position]
    if share is not None:
        # Two decimals, so the LLM can tell the nearest of several people.
        parts.append(f"height {share:.2f}")
    parts.extend(name for name in GESTURES if name in gestures and name in POSTURES)
    parts.extend(name for name in GESTURES if name in gestures and name not in POSTURES)
    return ", ".join(parts)


def _face_box(person: PersonDetection) -> BBox | None:
    # face_det_score is 0 when no face was found in the box (PersonDetection.msg).
    return bbox_xyxy(person.face_bbox) if person.face_det_score > 0 else None


def _face_text(payload: shared.PersonPayload) -> str:
    """FACE_SEEN, FACE_UNCLEAR or NO_FACE, by what the handle can be recognized by."""
    if payload.face_bbox is None:
        return NO_FACE
    # The payload keeps a face vector only when the face passed face_trusted,
    # the gate of the track's collection too, and came with a vector: a face
    # without one cannot be recognized however clear it is.
    return FACE_SEEN if payload.face is not None else FACE_UNCLEAR


def _payload(
    feed: VisionFeed,
    msg: PersonDetectionArray,
    stamp_ns: int,
    entry: _Seen,
    face_box: BBox | None,
    heading: float | None,
) -> shared.PersonPayload:
    person = entry.person
    face = None
    if face_trusted(person, msg.camera_info, feed.config):
        face = model_vector(person.face_model, person.face_embedding)
    return shared.PersonPayload(
        track=entry.track,
        position=entry.position,
        bbox=bbox_xyxy(person.bbox),
        face_bbox=face_box,
        keypoints=tuple(float(v) for v in person.keypoints),
        keypoint_scores=tuple(float(v) for v in person.keypoint_scores),
        gestures=entry.gestures,
        face=face,
        appearance=model_vector(person.appearance_model, person.appearance_embedding),
        header=msg.header,
        stamp_ns=stamp_ns,
        image_width=int(msg.camera_info.width),
        image_height=int(msg.camera_info.height),
        heading_map=heading,
    )


def _count(n: int) -> str:
    return "1 person" if n == 1 else f"{n} people"
