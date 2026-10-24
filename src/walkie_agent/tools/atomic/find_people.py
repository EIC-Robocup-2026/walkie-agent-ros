"""Find the people in front of the camera, with their gestures and posture.

Reads the newest /walkie_vision/people array through the `vision` backend
and keeps one person handle per person, left to right (ADR-0050). Gestures
and postures come from the keypoints, by walkie-agent-v2's rules
(walkie_agent.perception.gestures). With in_front, only the people close in
front of the robot get handles, by walkie_agent.perception.front's rule, so
that at the door a passer-by or someone beside the robot is not taken for
the guest.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from pydantic import BaseModel, Field
from rclpy.time import Time
from walkie_vision_msgs.msg import PersonDetection, PersonDetectionArray

from walkie_agent.perception.front import REASONS, face_size_rad, front_reason
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
from walkie_agent.tools.navigation import MAP_FRAME

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
        "own arm; pointing_left and pointing_right are directions as you see them. "
        "With in_front, only the people close in front of the robot, and a last line "
        "counts those left out and why: use it at the door to find the guest, not at "
        "the seats, where seated people sit to the side. A handle stays valid for 10 "
        "minutes while you do not drive; turning keeps it."
    ),
    effects=frozenset({Effect.P}),
    questions=(
        "Who is in front of you?",
        "Is anyone waving at you?",
        "Find the person raising their left arm.",
        "How many people are sitting down?",
        "Is a guest standing at the door?",
    ),
    examples=({}, {"gesture": "waving"}, {"in_front": True}),
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
    in_front: bool = Field(
        default=False,
        description=(
            "Only people close in front of the robot, with a clear, large face near its "
            "heading, such as a guest at the door. Passers-by behind them and people "
            "beside the robot are left out."
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
    image_height = msg.camera_info.height
    if not args.in_front:
        handle_set = shared.put_handles(ctx, "person", payloads, msg.header)
        if gesture is None:
            lines = [f"{handle_set.id}: {_count(len(seen))} in view, left to right."]
        else:
            verb = "shows" if len(chosen) == 1 else "show"
            lines = [
                f"{handle_set.id}: {len(chosen)} of {_count(len(seen))} in view {verb} "
                f"{gesture}, left to right."
            ]
        lines.extend(_person_lines(handle_set.member_ids, payloads, image_height))
        return Out(
            summary="\n".join(lines), set_id=handle_set.id, handles=list(handle_set.member_ids)
        )

    # The base's pose at the image's stamp, from the same TF as the
    # headings, so the localization error cancels out of the angle between
    # them. It is also the handles' capture pose: one lookup serves both.
    try:
        pose = shared.capture_pose(ctx, msg.header)
    except StepError:
        ctx.check_cancelled()
        raise
    if all(heading is None for heading in heading_maps):
        ctx.check_cancelled()
        raise StepError(
            "UNAVAILABLE",
            "Which way each person is from the robot is unknown, so nothing was stored. "
            f"It needs the camera's transform ({MAP_FRAME} -> {msg.header.frame_id}) at the "
            "image's stamp and a camera_info with a focal length.",
        )
    face_sizes = [
        None if face_box is None else face_size_rad(face_box, msg.camera_info)
        for face_box in face_boxes
    ]
    reasons = [
        front_reason(
            entry.person.face_det_score,
            face_size,
            heading,
            pose.yaw,
            # The feed's limit, so that a person kept here can be recognized.
            feed.config.min_face_det_score,
            config.front,
        )
        for entry, face_size, heading in zip(chosen, face_sizes, heading_maps, strict=True)
    ]
    kept = [i for i, reason in enumerate(reasons) if reason is None]
    left_out = _left_out(gesture, len(seen) - len(chosen), reasons)
    if not kept:
        # A stop during the lookups is why nobody was kept, if one came.
        ctx.check_cancelled()
        raise _nobody_in_front(gesture, seen, chosen, reasons, left_out, image_height)

    # Only the people kept get handles, so no handle names someone left out.
    kept_payloads = [payloads[i] for i in kept]
    handle_set = shared.put_handles(ctx, "person", kept_payloads, msg.header, pose=pose)
    verb, shows = ("is", "shows") if len(kept) == 1 else ("are", "show")
    first = (
        f"{handle_set.id}: {len(kept)} of {_count(len(seen))} in view {verb} close in front"
        + ("" if gesture is None else f" and {shows} {gesture}")
        + ", left to right"
    )
    if len(kept) >= 2:
        # Marked rather than put first, since the handles' numbers keep the
        # left-to-right order. max() takes the first of a tie, the leftmost.
        largest = max(range(len(kept)), key=lambda j: face_sizes[kept[j]])
        first += f"; {handle_set.member_ids[largest]} has the largest face"
    lines = [first + "."]
    lines.extend(_person_lines(handle_set.member_ids, kept_payloads, image_height))
    lines.append(f"Left out: {_counts_text(left_out) or 'nobody'}.")
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


def _person_lines(
    handle_ids: Sequence[str], payloads: Sequence[shared.PersonPayload], image_height: int
) -> list[str]:
    """One line per handle, such as "per_51c0.0: centre, height 0.74, standing, face seen"."""
    lines = []
    for handle_id, payload in zip(handle_ids, payloads, strict=True):
        share = (payload.bbox[3] - payload.bbox[1]) / image_height if image_height else None
        description = _describe(payload.position, payload.gestures, share)
        lines.append(f"{handle_id}: {description}, {_face_text(payload)}")
    return lines


def _left_out(
    gesture: str | None, without_gesture: int, reasons: Sequence[str | None]
) -> dict[str, int]:
    """How many were left out for each reason, those without the gesture first; none at 0."""
    left_out = {}
    if gesture is not None and without_gesture:
        left_out[f"not showing {gesture}"] = without_gesture
    for reason in REASONS:
        if count := reasons.count(reason):
            left_out[reason] = count
    return left_out


def _counts_text(left_out: Mapping[str, int]) -> str:
    """Such as "1 too far, 1 to the side"; empty for nobody."""
    return ", ".join(f"{count} {reason}" for reason, count in left_out.items())


def _nobody_in_front(
    gesture: str | None,
    seen: Sequence[_Seen],
    chosen: Sequence[_Seen],
    reasons: Sequence[str | None],
    left_out: dict[str, int],
    image_height: int,
) -> StepError:
    """NOT_FOUND for in_front when nobody was kept, saying why each person in view was not.

    It names no way to come closer: the robot does not direct people
    (hri.md), and the LLM can call find_people without in_front instead.
    """
    reason_of = {id(entry): reason for entry, reason in zip(chosen, reasons, strict=True)}
    parts = []
    for entry in seen:
        box = bbox_xyxy(entry.person.bbox)
        share = (box[3] - box[1]) / image_height if image_height else None
        why = reason_of.get(id(entry), f"not showing {gesture}")
        parts.append(f"{_describe(entry.position, entry.gestures, share)}, {why}")
    who = "" if gesture is None else f" who shows {gesture}"
    return StepError(
        "NOT_FOUND",
        f"Nobody in view{who} is close in front of the robot. In view, left to right: "
        f"{'; '.join(parts)}. find_people without in_front gives each a handle.",
        {"gesture": gesture, "people": len(seen), "left_out": left_out},
    )
