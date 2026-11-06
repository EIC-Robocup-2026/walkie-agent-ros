"""A fixed simulated scene, read from a TOML file. No ROS.

oracle_scene draws the scene this returns for every image of the robot's
camera (walkie_agent.sim.oracle_perception). Nobody in it moves: a person
stands or sits where the file puts them for the whole run. The file looks
like this, in the map frame, in meters and degrees:

    world = "chula_pat14_6"
    classes = ["sofa", "table", "mug"]

    [[people]]
    track_id = 1
    x = 3.05
    y = -0.45
    facing_deg = -133.0
    gesture = "waving"
    face = "guest1"
    appearance = "guest1"

    [[objects]]
    track_id = 11
    label = "sofa"
    center = [3.1675, -1.65, 0.4175]
    size = [0.575, 1.88, 0.835]

A person has track_id, x, y and facing_deg, and may give height, gesture,
posture, face, appearance and seat_top (walkie_agent.sim.scene.ScenePerson).
An object has track_id, label, center and size, and may give yaw_deg,
view_quality and color (walkie_agent.sim.scene.SceneObject). `world` is
written to the oracle's log only. `classes` is the detector's vocabulary,
ObjectDetectionArray.classes; without it, it is the objects' labels in file
order, or the oracle's default seat classes for a file with no objects.

A file with an unknown key, a value of the wrong type, a value out of range,
a label missing from `classes` or two people or two objects that share a
track id other than 0 raises ValueError with a one-line reason.
"""

import math
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)

from walkie_agent.sim.scene import (
    DEFAULT_SEAT_TOP_M,
    STANDARD_HEIGHT_M,
    MapBox,
    Scene,
    SceneObject,
    ScenePerson,
)


def _finite(value: float) -> float:
    try:
        number = float(value)
    except OverflowError as e:  # A TOML integer has no size limit.
        raise ValueError("is too large") from e
    if not math.isfinite(number):  # A TOML float can be nan or inf.
        raise ValueError("is not a finite number")
    return number


# Strict, so a boolean or a quoted number is refused instead of being converted.
_Number = Annotated[StrictInt | StrictFloat, AfterValidator(_finite)]


class _Person(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: StrictInt
    x: _Number
    y: _Number
    facing_deg: _Number
    height: _Number = STANDARD_HEIGHT_M
    gesture: StrictStr | None = None
    posture: StrictStr = "standing"
    face: StrictStr | None = None
    appearance: StrictStr | None = None
    seat_top: _Number = DEFAULT_SEAT_TOP_M


class _Object(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: StrictInt
    label: StrictStr
    center: tuple[_Number, _Number, _Number]
    size: tuple[_Number, _Number, _Number]
    yaw_deg: _Number = 0.0
    view_quality: _Number = 0.0
    color: StrictStr | None = None


class _File(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world: StrictStr = ""
    classes: list[StrictStr] | None = None
    people: list[_Person] = Field(default_factory=list)
    objects: list[_Object] = Field(default_factory=list)


@dataclass(frozen=True)
class SceneFile:
    """A scene file's contents: the scene, the detector's classes and the world's name."""

    scene: Scene
    # None leaves the oracle's default classes.
    classes: tuple[str, ...] | None
    world: str


def load_scene(path: str | Path) -> SceneFile:
    """Read a scene file. Raises ValueError with a one-line reason for a file that breaks the rules."""
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path}: {e}.") from e
    try:
        parsed = _File.model_validate(data)
    except ValidationError as e:
        reason = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in e.errors()
        )
        raise ValueError(f"{path}: {reason}.") from e

    people = tuple(
        _entry(path, f"people.{i}", _person, entry) for i, entry in enumerate(parsed.people)
    )
    objects = tuple(
        _entry(path, f"objects.{i}", _object, entry) for i, entry in enumerate(parsed.objects)
    )
    _check_unique_tracks(path, "people", [p.track_id for p in people])
    _check_unique_tracks(path, "objects", [o.track_id for o in objects])

    classes = parsed.classes
    if classes is None and objects:
        classes = list(dict.fromkeys(o.label for o in objects))
    if classes is not None:
        if not classes or not all(classes) or len(set(classes)) != len(classes):
            raise ValueError(
                f"{path}: classes must be unique non-empty names, and at least one, got {classes}."
            )
        missing = [o.label for o in objects if o.label not in classes]
        if missing:
            raise ValueError(
                f"{path}: the labels {sorted(set(missing))} are not in classes {classes}, "
                "so those objects would never be detected."
            )
    return SceneFile(
        scene=Scene(people=people, objects=objects),
        classes=None if classes is None else tuple(classes),
        world=parsed.world,
    )


def _entry[E, T](path: str | Path, where: str, build: Callable[[E], T], entry: E) -> T:
    """`build(entry)`, with the file and the entry named in the ValueError the scene types raise."""
    try:
        return build(entry)
    except ValueError as e:
        raise ValueError(f"{path}: {where}: {e}") from e


def _person(entry: _Person) -> ScenePerson:
    return ScenePerson(
        track_id=entry.track_id,
        x=entry.x,
        y=entry.y,
        facing=math.radians(entry.facing_deg),
        height=entry.height,
        gesture=entry.gesture,
        posture=entry.posture,
        face=entry.face,
        appearance=entry.appearance,
        seat_top=entry.seat_top,
    )


def _object(entry: _Object) -> SceneObject:
    return SceneObject(
        track_id=entry.track_id,
        label=entry.label,
        box=MapBox(center=entry.center, size=entry.size, yaw=math.radians(entry.yaw_deg)),
        view_quality=entry.view_quality,
        color=entry.color,
    )


def _check_unique_tracks(path: str | Path, kind: str, track_ids: list[int]) -> None:
    # 0 is untracked, so several may share it.
    tracked = [t for t in track_ids if t != 0]
    if len(set(tracked)) != len(tracked):
        repeated = sorted({t for t in tracked if tracked.count(t) > 1})
        raise ValueError(f"{path}: {kind} share the track ids {repeated}.")
