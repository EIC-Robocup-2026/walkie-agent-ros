"""The projection and the person figure that oracle_perception draws, with no ROS.

walkie_agent.sim.projection turns map-frame points into pixels and draws a
person as COCO keypoints; walkie_agent.sim.scene holds the people and
objects. The camera is the simulation's head camera, so the checks hold for
the images the oracle draws in walkie-ros's MuJoCo world. The gesture tests
run the figure through walkie_agent.perception.gestures, the rules
find_people applies, so the figure and the rules cannot drift apart.
"""

import math

import numpy as np
import pytest

from walkie_agent.perception.gestures import COCO_KEYPOINTS, classify_gestures
from walkie_agent.sim.projection import (
    BOX_ABOVE_HEAD_M,
    BOX_SIDE_PAD,
    FACE_SIDE_M,
    MIN_DEPTH_M,
    camera_from_map_at,
    camera_position,
    clip_box,
    in_image,
    person_figure,
    project,
    project_box,
    project_person,
)
from walkie_agent.sim.scene import ARM_GESTURES, MapBox, ScenePerson

# The head camera of walkie-ros's MuJoCo simulation (branch
# feat/chula-pat14-6-simulation at adb9725): 640 x 360 px, and MuJoCo's fovy
# of 77.554 deg gives fx = fy = 180 / tan(fovy / 2) = 224.057 px.
WIDTH, HEIGHT = 640, 360
FX = FY = 224.057
CX, CY = 320.0, 180.0
K = (FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0)
# The optical frame in base_footprint at the lift's 0.507 m, from the
# forward kinematics of the description's walkie.xml at 526c976: the origin
# at (0.275, 0.060, 1.145) m, pitched 9.0 deg down, the image's right along
# the base's -y.
CAMERA_XYZ = (0.275, 0.060, 1.145)
CAMERA_PITCH_DEG = 9.0

_LEFT_SHOULDER = COCO_KEYPOINTS.index("left_shoulder")
_RIGHT_SHOULDER = COCO_KEYPOINTS.index("right_shoulder")
_LEFT_HIP = COCO_KEYPOINTS.index("left_hip")
_LEFT_ANKLE = COCO_KEYPOINTS.index("left_ankle")


def base_from_camera(pitch_deg: float = CAMERA_PITCH_DEG) -> np.ndarray:
    """The optical frame's pose in base_footprint, looking ahead and pitched down."""
    p = math.radians(pitch_deg)
    transform = np.eye(4)
    # Columns: the optical x (image right), y (image down) and z (out of the
    # lens) axes in the base's frame.
    transform[:3, :3] = [
        [0.0, -math.sin(p), math.cos(p)],
        [-1.0, 0.0, 0.0],
        [0.0, -math.cos(p), -math.sin(p)],
    ]
    transform[:3, 3] = CAMERA_XYZ
    return transform


def sim_camera(
    x: float = 0.0, y: float = 0.0, yaw: float = 0.0, pitch_deg: float = CAMERA_PITCH_DEG
) -> np.ndarray:
    """camera <- map for the head camera on a base at (x, y, yaw)."""
    return camera_from_map_at(x, y, yaw, base_from_camera(pitch_deg))


def facing_person(distance: float, **fields) -> ScenePerson:
    """A person `distance` m ahead of the camera on a base at the origin, facing it."""
    camera = camera_position(sim_camera())
    defaults = {"track_id": 1, "x": camera[0] + distance, "y": camera[1], "facing": math.pi}
    return ScenePerson(**(defaults | fields))


def gestures_of(person: ScenePerson, camera: np.ndarray | None = None) -> set[str]:
    view = project_person(person, sim_camera() if camera is None else camera, K)
    assert view is not None
    # The whole figure is in the image, so every rule sees every point.
    assert in_image(view.keypoints, WIDTH, HEIGHT).all()
    return classify_gestures(view.keypoints.reshape(-1), [0.9] * len(COCO_KEYPOINTS))


# -- Projection ---------------------------------------------------------------------------


def test_a_point_on_the_optical_axis_lands_on_the_principal_point():
    camera = sim_camera(1.0, -2.0, 0.7)
    axis = np.linalg.inv(camera)[:3, 2]

    uv, depth = project([camera_position(camera) + 2.5 * axis], camera, K)

    assert uv[0] == pytest.approx((CX, CY), abs=1e-6)
    assert depth[0] == pytest.approx(2.5)


def test_a_point_behind_the_camera_is_not_seen():
    camera = sim_camera()
    origin, axis = camera_position(camera), np.linalg.inv(camera)[:3, 2]

    uv, depth = project([origin - axis, origin + 0.5 * MIN_DEPTH_M * axis], camera, K)

    assert np.isnan(uv).all()
    assert depth == pytest.approx([-1.0, 0.5 * MIN_DEPTH_M])
    behind = ScenePerson(track_id=1, x=origin[0] - 2.0, y=origin[1], facing=0.0)
    assert project_person(behind, camera, K) is None
    assert project_box(MapBox((origin[0] - 2.0, 0.0, 0.5), (0.5, 0.5, 1.0)), camera, K) is None
    # A box wholly nearer than MIN_DEPTH_M counts as behind as well: its
    # pixels would lie far outside the image.
    near = MapBox(tuple(origin + 0.5 * MIN_DEPTH_M * axis), (0.05, 0.05, 0.05))
    assert project_box(near, camera, K) is None


def test_a_box_partly_in_the_image_is_clipped_and_one_outside_is_not_seen():
    assert clip_box((-10.0, 20.0, 50.0, 400.0), WIDTH, HEIGHT) == (0.0, 20.0, 50.0, 360.0)
    assert clip_box((650.0, 0.0, 700.0, 10.0), WIDTH, HEIGHT) is None
    # A box that only touches the image's edge has nothing inside it.
    assert clip_box((-20.0, 0.0, 0.0, 10.0), WIDTH, HEIGHT) is None

    camera = sim_camera()
    # 2 m ahead and 2.5 m to the left: the camera's 55 deg half view ends at
    # 2.86 m to the side, so the box's right part is in the image.
    box = project_box(MapBox((2.275, 2.56, 0.5), (0.6, 1.0, 1.0)), camera, K)
    assert box is not None and box[0] < 0 < box[2]
    clipped = clip_box(box, WIDTH, HEIGHT)
    assert clipped is not None and clipped[0] == 0.0 and clipped[2] == box[2]


def test_a_box_that_reaches_behind_the_camera_keeps_its_visible_part():
    camera = sim_camera()
    # A bench from 1 m behind the camera to 2 m ahead of it, on the floor.
    bench = MapBox((0.775, 0.06, 0.25), (3.0, 0.5, 0.5))

    box = project_box(bench, camera, K)

    assert box is not None and np.isfinite(box).all()
    clipped = clip_box(box, WIDTH, HEIGHT)
    # Its far end is ahead in the image's lower half, and it runs out of the
    # image's bottom towards the camera.
    assert clipped is not None
    assert clipped[1] > CY and clipped[3] == HEIGHT
    assert clipped[0] < CX < clipped[2]


def test_turning_the_camera_moves_the_boxes_the_right_way():
    thing = MapBox((3.275, 0.06, 0.6), (0.4, 0.4, 0.4))
    person = facing_person(3.0)
    ahead = project_box(thing, sim_camera(), K)
    seen = project_person(person, sim_camera(), K).box

    # Turning left puts what was ahead to the image's right, by about
    # fx * tan(0.3) = 69 px; pitching further down raises it in the image.
    turned_left = project_box(thing, sim_camera(yaw=0.3), K)
    turned_right = project_box(thing, sim_camera(yaw=-0.3), K)
    pitched_down = project_box(thing, sim_camera(pitch_deg=20.0), K)
    person_left = project_person(person, sim_camera(yaw=0.3), K).box

    centre = (ahead[0] + ahead[2]) / 2
    assert centre == pytest.approx(CX, abs=1.0)
    assert (turned_left[0] + turned_left[2]) / 2 - centre == pytest.approx(
        FX * math.tan(0.3), rel=0.1
    )
    assert (turned_right[0] + turned_right[2]) / 2 < centre - 50
    assert pitched_down[1] < ahead[1] and pitched_down[3] < ahead[3]
    assert (person_left[0] + person_left[2]) / 2 > (seen[0] + seen[2]) / 2 + 50


# -- The person figure ----------------------------------------------------------------------


def test_the_figure_has_the_heights_the_sim_poses_were_chosen_with():
    standing = person_figure(ScenePerson(track_id=1, x=0.0, y=0.0, facing=0.0))
    tall = person_figure(ScenePerson(track_id=1, x=0.0, y=0.0, facing=0.0, height=1.80))
    seated = person_figure(
        ScenePerson(track_id=1, x=0.0, y=0.0, facing=0.0, posture="sitting", seat_top=0.46)
    )

    # A 1.70 m person: the face's centre at 1.60 m, the head's top at 1.70 m
    # and the hips at 0.90 m; a taller one scales.
    assert (standing.face_height, standing.head_top) == pytest.approx((1.60, 1.70))
    assert standing.points[_LEFT_HIP, 1] == pytest.approx(0.90)
    assert (tall.face_height, tall.head_top) == pytest.approx((1.60 * 1.80 / 1.70, 1.80))
    # Sitting: the hips 0.10 m above the seat, the face's centre at 1.15 m.
    assert seated.points[_LEFT_HIP, 1] == pytest.approx(0.56)
    assert seated.face_height == pytest.approx(1.15)
    assert seated.points[_LEFT_ANKLE, 1] >= 0.0


def test_the_person_box_pads_the_keypoints_and_reaches_above_the_head():
    camera = sim_camera()
    person = facing_person(2.0)

    view = project_person(person, camera, K)

    (u1, v1), (u2, v2) = view.keypoints.min(axis=0), view.keypoints.max(axis=0)
    pad = BOX_SIDE_PAD * (u2 - u1)
    above_head, _ = project(
        [(person.x, person.y, person_figure(person).head_top + BOX_ABOVE_HEAD_M)], camera, K
    )
    assert view.box == pytest.approx((u1 - pad, above_head[0][1], u2 + pad, v2))
    assert above_head[0][1] < v1


def test_the_face_box_is_the_projected_face_square():
    camera = sim_camera()
    person = facing_person(2.0)

    view = project_person(person, camera, K)

    centre, depth = project(
        [(person.x, person.y, person_figure(person).face_height)], camera, K
    )
    x1, y1, x2, y2 = view.face_box
    assert ((x1 + x2) / 2, (y1 + y2) / 2) == pytest.approx(tuple(centre[0]), abs=0.5)
    assert x2 - x1 == pytest.approx(FACE_SIDE_M * FX / depth[0], rel=0.03)
    # The square stands upright, 22 deg above the optical axis of a camera
    # that looks down, so it is drawn a little taller than wide.
    assert y2 - y1 == pytest.approx(FACE_SIDE_M * FY / depth[0], rel=0.06)
    assert view.box[0] < x1 < x2 < view.box[2] and view.box[1] < y1


def test_a_person_facing_the_camera_has_their_left_on_the_image_right():
    camera = sim_camera()
    towards = project_person(facing_person(2.0), camera, K).keypoints
    away = project_person(facing_person(2.0, facing=0.0), camera, K).keypoints

    assert towards[_LEFT_SHOULDER, 0] > towards[_RIGHT_SHOULDER, 0]
    assert away[_LEFT_SHOULDER, 0] < away[_RIGHT_SHOULDER, 0]


# The gestures each variant shows to find_people. One image cannot show a
# wave's motion, so the rules take any raised hand for a wave, and the
# figure waves with its right hand.
VARIANTS = {
    ("standing", None): {"standing"},
    ("sitting", None): {"sitting"},
    ("standing", "waving"): {"standing", "waving", "raising_right_arm"},
    ("standing", "raising_left_arm"): {"standing", "raising_left_arm", "waving"},
    ("standing", "raising_right_arm"): {"standing", "raising_right_arm", "waving"},
    ("standing", "pointing_left"): {"standing", "pointing_left"},
    ("standing", "pointing_right"): {"standing", "pointing_right"},
}


def test_the_variants_cover_every_drawn_gesture():
    assert {gesture for _, gesture in VARIANTS} - {None} == set(ARM_GESTURES)


@pytest.mark.parametrize("distance", [1.2, 3.0])
@pytest.mark.parametrize(("posture", "gesture"), list(VARIANTS))
def test_each_gesture_variant_classifies_as_itself(posture, gesture, distance):
    person = facing_person(distance, posture=posture, gesture=gesture)

    assert gestures_of(person) == VARIANTS[(posture, gesture)]


@pytest.mark.parametrize("distance", [1.2, 3.0])
@pytest.mark.parametrize("gesture", ARM_GESTURES)
def test_arm_gestures_hold_while_sitting(gesture, distance):
    person = facing_person(distance, posture="sitting", gesture=gesture, seat_top=0.4575)

    shown = VARIANTS[("standing", gesture)] - {"standing"} | {"sitting"}
    assert gestures_of(person) == shown


@pytest.mark.parametrize("gesture", ["pointing_left", "pointing_right"])
def test_pointing_is_towards_the_image_side_also_when_turned_away(gesture):
    person = facing_person(2.0, gesture=gesture, facing=0.0)

    assert gestures_of(person) == {"standing", gesture}


# -- The scene ----------------------------------------------------------------------------


def test_a_face_is_towards_the_camera_within_the_limit():
    def person(facing: float) -> ScenePerson:
        return ScenePerson(track_id=1, x=2.0, y=0.0, facing=facing)

    assert person(math.pi).faces_camera((0.0, 0.0))
    assert person(math.pi - math.radians(59)).faces_camera((0.0, 0.0))
    assert person(-math.pi + math.radians(59)).faces_camera((0.0, 0.0))
    assert not person(math.pi - math.radians(61)).faces_camera((0.0, 0.0))
    assert not person(0.0).faces_camera((0.0, 0.0))
    assert person(math.pi - math.radians(61)).faces_camera((0.0, 0.0), limit_deg=70.0)


@pytest.mark.parametrize(
    "fields",
    [
        {"gesture": "lying_down"},
        {"posture": "lying_down"},
        {"track_id": -1},
        {"height": 0.0},
    ],
)
def test_a_person_the_figure_cannot_draw_is_refused(fields):
    with pytest.raises(ValueError):
        ScenePerson(**({"track_id": 1, "x": 0.0, "y": 0.0, "facing": 0.0} | fields))
