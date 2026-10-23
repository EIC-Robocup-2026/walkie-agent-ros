"""Seats, sofa places and which of them are taken, ported from walkie-agent-v2.

find_seats pairs walkie-vision's objects array with the people array of the
same image (hri.md, Seats). A place is taken when a person's seating anchor
lies in its box: the hips, or else the lower centre of the person's box. A
sofa splits into left, middle and right places, each taken on its own, so a
sofa with one guest on it still offers its free places. The rules are
walkie-agent-v2's tasks/skills/seating.py and tasks/skills/geometry.py, on
message boxes instead of v2's PersonPose and detections.

As in v2, everyone in view is taken to be seated: while the robot looks at
the seats, the guest it brought stands beside it, outside the seating area.
A person whose anchor lies on no seat sits on something else and takes none
of them.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from vision_msgs.msg import BoundingBox2D
from walkie_vision_msgs.msg import ObjectDetection, PersonDetection

# A box as x1, y1, x2, y2 in pixels. x grows to the right of the image, so
# the smallest x is the robot's left.
BBox = tuple[float, float, float, float]
Point = tuple[float, float]

# COCO's hips, the part of the body that rests on the seat.
_LEFT_HIP, _RIGHT_HIP = 11, 12


@dataclass(frozen=True)
class SeatingConfig:
    """The seat classes and the anchor rule.

    The defaults are walkie-agent-v2's, tuned on the robot; the name in each
    comment is the v2 environment variable that held the value.
    """

    # The detector classes that are seats. Detections of other classes are
    # left out. HRI_SEAT_CLASSES, tuned on the robot.
    seat_classes: tuple[str, ...] = ("chair", "sofa", "armchair", "stool")
    # The classes that seat several people and split into places. Only a
    # seat class is a seat, so a sofa class splits only when it is a seat
    # class too. HRI_SOFA_CLASSES, tuned on the robot.
    sofa_classes: tuple[str, ...] = ("sofa", "couch", "bench", "loveseat")
    # Whether a sofa has a middle place: left, middle and right, else left
    # and right. HRI_SOFA_HAS_MIDDLE, tuned on the robot.
    sofa_has_middle: bool = True
    # A hip keypoint must score above this to anchor the person; below it
    # the position is a guess. HRI_POSE_KP_CONF, tuned on the robot.
    min_hip_score: float = 0.3
    # Without a hip, the anchor is this share of the box height below the
    # box centre: roughly where the seat is under a seated person. v2's
    # person_seat_anchor, tuned on the robot.
    box_anchor_drop: float = 0.25

    def is_seat(self, label: str) -> bool:
        return label.lower() in {c.lower() for c in self.seat_classes}

    def is_sofa(self, label: str | None) -> bool:
        return (label or "").lower() in {c.lower() for c in self.sofa_classes}


DEFAULT_SEATING_CONFIG = SeatingConfig()


@dataclass(frozen=True)
class SeatPlace:
    """One place to sit: a whole seat, or one place of a sofa."""

    # The place's box: the seat's own, or one part of a sofa's.
    box: BBox
    # The detector's class, such as "chair" or "sofa".
    label: str
    # "left", "middle" or "right" for a place on a sofa, else None. Sides
    # are in the image, the robot's view.
    part: str | None
    occupied: bool
    # The detector's confidence in the seat.
    score: float
    # The seat's index in ObjectDetectionArray.detections.
    detection_index: int


def bbox_xyxy(bbox: BoundingBox2D) -> BBox:
    """A message's box, a centre and a size, as x1, y1, x2, y2."""
    cx, cy = bbox.center.position.x, bbox.center.position.y
    half_w, half_h = bbox.size_x / 2, bbox.size_y / 2
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def point_in_box(point: Point, box: BBox) -> bool:
    """True when the point lies in the box, edges included."""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def hip_anchor(
    keypoints: Sequence[float], keypoint_scores: Sequence[float], min_score: float
) -> Point | None:
    """The midpoint of the hips, or the one hip that scores above `min_score`.

    This is where a person rests on a seat. It tells which seat or sofa place
    they take far better than their box, which spreads sideways over the
    seats next to them. None when neither hip is seen.
    """
    hips = [
        (float(keypoints[2 * i]), float(keypoints[2 * i + 1]))
        for i in (_LEFT_HIP, _RIGHT_HIP)
        if keypoint_scores[i] > min_score
    ]
    if not hips:
        return None
    return (sum(x for x, _ in hips) / len(hips), sum(y for _, y in hips) / len(hips))


def seat_anchor(person: PersonDetection, config: SeatingConfig = DEFAULT_SEATING_CONFIG) -> Point:
    """Where the person sits: the hips when seen, else the lower centre of their box."""
    anchor = hip_anchor(person.keypoints, person.keypoint_scores, config.min_hip_score)
    if anchor is not None:
        return anchor
    centre = person.bbox.center.position
    return (centre.x, centre.y + person.bbox.size_y * config.box_anchor_drop)


def split_seat_regions(box: BBox, *, has_middle: bool = True) -> list[tuple[str, BBox]]:
    """A sofa's box split into equal places, left to right in the image."""
    x1, y1, x2, y2 = box
    parts = ("left", "middle", "right") if has_middle else ("left", "right")
    step = (x2 - x1) / len(parts)
    return [
        (part, (x1 + i * step, y1, x1 + (i + 1) * step, y2)) for i, part in enumerate(parts)
    ]


def occupied_region_indices(anchors: Sequence[Point], region_boxes: Sequence[BBox]) -> set[int]:
    """The indices of the boxes that a person's anchor lies in.

    Each person takes the first box that holds their anchor, so a person on
    the edge between two places of a sofa takes one of them, not both.
    """
    occupied: set[int] = set()
    for anchor in anchors:
        for i, box in enumerate(region_boxes):
            if point_in_box(anchor, box):
                occupied.add(i)
                break
    return occupied


def find_seat_places(
    detections: Sequence[ObjectDetection],
    people: Sequence[PersonDetection],
    config: SeatingConfig = DEFAULT_SEATING_CONFIG,
) -> list[SeatPlace]:
    """The places to sit among `detections`, each free or taken by one of `people`.

    `detections` and `people` must come from the same image, since their
    boxes are compared in pixels. The places are ordered left to right by
    their own box centre, after each sofa is split into its places, so a
    chair in front of a sofa's middle comes between the sofa's left and
    right places. Places with the same centre keep the detections' order.
    """
    anchors = [seat_anchor(person, config) for person in people]
    places: list[SeatPlace] = []
    for index, detection in enumerate(detections):
        if not config.is_seat(detection.label):
            continue
        box = bbox_xyxy(detection.bbox)
        regions: list[tuple[str | None, BBox]] = (
            list(split_seat_regions(box, has_middle=config.sofa_has_middle))
            if config.is_sofa(detection.label)
            else [(None, box)]
        )
        occupied = occupied_region_indices(anchors, [region for _, region in regions])
        places.extend(
            SeatPlace(
                box=region,
                label=detection.label,
                part=part,
                occupied=i in occupied,
                score=float(detection.score),
                detection_index=index,
            )
            for i, (part, region) in enumerate(regions)
        )
    # Sorted by place rather than by seat: the LLM reads the handles as left
    # to right and takes the first free one as the leftmost, and a sofa's
    # centre says nothing about where its left place is. The sort is stable,
    # so a tie keeps the detections' order and a sofa's parts in order.
    places.sort(key=lambda place: (place.box[0] + place.box[2]) / 2)
    return places
