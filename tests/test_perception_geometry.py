"""perception/geometry.py: the map-frame yaw of a pixel's ray, the floor's distance, and the angle wrap."""

import math

import numpy as np
import pytest
from test_projection import CX, CY, FX, HEIGHT, K, WIDTH, base_from_camera
from tf_transformations import quaternion_from_matrix

from walkie_agent.perception.geometry import (
    ANKLE_HEIGHT_M,
    MAX_FLOOR_RANGE_M,
    ankle_pixels,
    floor_range,
    mean_yaw,
    pixel_yaw,
    rotate,
    torso_pixels,
    wrap_deg,
    yaw_from,
)
from walkie_agent.perception.gestures import COCO_KEYPOINTS
from walkie_agent.sim.projection import camera_from_map_at, camera_position, project

# A level camera that looks along the map's x axis: the optical z is the map's x,
# the optical x its -y and the optical y its -z.
LEVEL = (-0.5, 0.5, -0.5, 0.5)


def map_from_camera(x: float, y: float, yaw: float, pitch_deg: float) -> np.ndarray:
    return np.linalg.inv(camera_from_map_at(x, y, yaw, base_from_camera(pitch_deg)))


def rotation_of(matrix: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in quaternion_from_matrix(matrix))


def test_the_centre_pixel_of_a_level_camera_is_straight_ahead():
    assert pixel_yaw(CX, CY, K, LEVEL) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("u", [0.0, 100.0, CX, 500.0, WIDTH])
def test_a_column_left_of_centre_is_to_the_left(u):
    expected = math.atan2(CX - u, FX)
    assert pixel_yaw(u, CY, K, LEVEL) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("pitch_deg", [0.0, 9.0, 30.0])
@pytest.mark.parametrize("yaw_deg", [0.0, 35.0, -120.0, 179.0])
def test_the_yaw_of_a_projected_point_is_its_direction_from_the_camera(yaw_deg, pitch_deg):
    """Round trip through the oracle's own projection, corners included."""
    x, y, yaw = 1.76, -1.48, math.radians(yaw_deg)
    camera_from_map = camera_from_map_at(x, y, yaw, base_from_camera(pitch_deg))
    rotation = rotation_of(np.linalg.inv(camera_from_map))
    camera = camera_position(camera_from_map)
    checked = 0
    for ahead in (0.8, 2.0, 4.0):
        for side in (-2.5, -1.0, 0.0, 1.0, 2.5):
            for height in (0.0, 0.9, 1.7):
                point = np.array(
                    [
                        camera[0] + ahead * math.cos(yaw) - side * math.sin(yaw),
                        camera[1] + ahead * math.sin(yaw) + side * math.cos(yaw),
                        height,
                    ]
                )
                uv, depth = project([point], camera_from_map, K)
                u, v = uv[0]
                if depth[0] <= 0 or not (0 <= u <= WIDTH and 0 <= v <= HEIGHT):
                    continue
                truth = math.atan2(point[1] - camera[1], point[0] - camera[0])
                found = pixel_yaw(u, v, K, rotation)
                assert math.isclose(
                    math.remainder(found - truth, math.tau), 0.0, abs_tol=1e-9
                ), (ahead, side, height)
                checked += 1
    assert checked > 20


def test_the_row_changes_the_yaw_of_a_pitched_camera():
    """Why the ray goes through the box's row: the column alone is off at the corners."""
    rotation = rotation_of(map_from_camera(0.0, 0.0, 0.0, 9.0))
    top, middle, bottom = (pixel_yaw(WIDTH, v, K, rotation) for v in (0.0, CY, HEIGHT))
    assert math.degrees(top - middle) > 3.0
    assert math.degrees(middle - bottom) > 3.0


def test_a_quaternion_that_is_not_unit_is_normalised_first():
    scaled = tuple(2.0 * v for v in LEVEL)
    assert pixel_yaw(100.0, 50.0, K, scaled) == pytest.approx(pixel_yaw(100.0, 50.0, K, LEVEL))
    assert rotate((0.0, 0.0, 0.0, 3.0), (1.0, 2.0, 3.0)) == pytest.approx((1.0, 2.0, 3.0))


@pytest.mark.parametrize(
    ("u", "v", "k", "rotation"),
    [
        pytest.param(CX, CY, (0.0,) * 9, LEVEL, id="a camera_info never filled in"),
        pytest.param(CX, CY, K[:5], LEVEL, id="a short k"),
        pytest.param(CX, CY, (math.nan, *K[1:]), LEVEL, id="a focal length that is NaN"),
        pytest.param(CX, CY, (FX, 0.0, CX, 0.0, -FX, CY, 0.0, 0.0, 1.0), LEVEL, id="a negative fy"),
        pytest.param(CX, CY, K, (0.0, 0.0, 0.0, 0.0), id="a zero quaternion"),
        pytest.param(math.nan, CY, K, LEVEL, id="a column that is NaN"),
        pytest.param(CX, math.inf, K, LEVEL, id="a row that is infinite"),
    ],
)
def test_no_yaw_without_the_numbers_for_one(u, v, k, rotation):
    assert pixel_yaw(u, v, k, rotation) is None


def test_the_central_ray_of_a_camera_that_looks_straight_down_has_no_yaw():
    down = rotation_of(map_from_camera(0.0, 0.0, 0.0, 90.0))
    assert pixel_yaw(CX, CY, K, down) is None
    # Any other pixel of that image still has a direction.
    assert pixel_yaw(CX, 0.0, K, down) == pytest.approx(0.0, abs=1e-9)


def body(**seen: tuple[float, float, float]) -> tuple[list[float], list[float]]:
    """Flat keypoints and scores with `name=(u, v, score)` set and the rest unseen at (0, 0)."""
    keypoints, scores = [0.0] * (2 * len(COCO_KEYPOINTS)), [0.0] * len(COCO_KEYPOINTS)
    for name, (u, v, score) in seen.items():
        i = COCO_KEYPOINTS.index(name)
        keypoints[2 * i : 2 * i + 2], scores[i] = [u, v], score
    return keypoints, scores


def test_the_torso_pixels_are_the_shoulders_and_hips_seen_well_enough():
    keypoints, scores = body(
        left_shoulder=(300.0, 100.0, 0.9),
        right_shoulder=(340.0, 100.0, 0.9),
        left_hip=(305.0, 200.0, 0.2),
        right_wrist=(600.0, 90.0, 0.9),
    )
    assert torso_pixels(keypoints, scores, 0.3) == [(300.0, 100.0), (340.0, 100.0)]
    assert len(torso_pixels(keypoints, scores, 0.1)) == 3


@pytest.mark.parametrize(
    ("keypoints", "scores"),
    [
        pytest.param(*body(left_shoulder=(300.0, 100.0, 0.9)), id="one seen"),
        pytest.param(*body(), id="none seen"),
        pytest.param(
            *body(left_shoulder=(math.nan, 100.0, 0.9), right_shoulder=(340.0, 100.0, 0.9)),
            id="one not finite",
        ),
        pytest.param([], [], id="no keypoints at all"),
        pytest.param([0.0] * 34, [1.0] * 16, id="a score short"),
    ],
)
def test_too_few_torso_pixels_give_none(keypoints, scores):
    assert torso_pixels(keypoints, scores, 0.3) == []


def test_the_mean_yaw_is_the_rays_middle_not_the_middle_pixels():
    # Two shoulders 6 degrees either side of 40 degrees to the left, level camera.
    left, right = (CX - FX * math.tan(math.radians(a)) for a in (46.0, 34.0))
    found = mean_yaw([(left, CY), (right, CY)], K, LEVEL)
    assert math.degrees(found) == pytest.approx(40.0, abs=1e-9)
    middle = pixel_yaw((left + right) / 2, CY, K, LEVEL)
    assert math.degrees(middle) > 40.4


def test_the_mean_yaw_wraps_across_behind():
    # Rays at 179 and -179 degrees average to 180, not to 0.
    camera = rotation_of(map_from_camera(0.0, 0.0, math.pi, 0.0))
    pixels = [(CX - FX * math.tan(math.radians(a)), CY) for a in (-1.0, 1.0)]
    assert abs(math.degrees(mean_yaw(pixels, K, camera))) == pytest.approx(180.0)


def test_no_mean_yaw_when_no_pixel_has_one():
    assert mean_yaw([], K, LEVEL) is None
    assert mean_yaw([(CX, CY)], (0.0,) * 9, LEVEL) is None


def test_the_ankle_pixels_are_the_ankles_seen_well_enough():
    keypoints, scores = body(
        left_ankle=(300.0, 330.0, 0.9),
        right_ankle=(340.0, 332.0, 0.2),
        left_shoulder=(300.0, 100.0, 0.9),
    )
    assert ankle_pixels(keypoints, scores, 0.3) == [(300.0, 330.0)]
    assert ankle_pixels(keypoints, scores, 0.1) == [(300.0, 330.0), (340.0, 332.0)]
    assert ankle_pixels(*body(left_shoulder=(300.0, 100.0, 0.9)), 0.3) == []
    assert ankle_pixels([0.0] * 34, [1.0] * 16, 0.3) == []


def sim_view(x: float, y: float, yaw_deg: float, pitch_deg: float):
    """The sim head camera on a base at (x, y, yaw): camera <- map, its rotation and its origin."""
    camera_from_map = camera_from_map_at(x, y, math.radians(yaw_deg), base_from_camera(pitch_deg))
    rotation = rotation_of(np.linalg.inv(camera_from_map))
    return camera_from_map, rotation, tuple(float(v) for v in camera_position(camera_from_map))


def ahead_of(origin, yaw: float, ahead: float, side: float, height: float) -> np.ndarray:
    return np.array(
        [
            origin[0] + ahead * math.cos(yaw) - side * math.sin(yaw),
            origin[1] + ahead * math.sin(yaw) + side * math.cos(yaw),
            height,
        ]
    )


@pytest.mark.parametrize("pitch_deg", [0.0, 9.0, 30.0])
@pytest.mark.parametrize("yaw_deg", [0.0, 35.0, -120.0])
def test_the_floor_range_of_a_projected_ankle_is_its_distance_from_the_camera(yaw_deg, pitch_deg):
    """Round trip through the oracle's own projection, for points an ankle above the floor."""
    camera_from_map, rotation, origin = sim_view(1.76, -1.48, yaw_deg, pitch_deg)
    checked = 0
    for ahead in (1.0, 2.0, 4.0, 7.0):
        for side in (-2.5, -1.0, 0.0, 1.0, 2.5):
            point = ahead_of(origin, math.radians(yaw_deg), ahead, side, ANKLE_HEIGHT_M)
            uv, depth = project([point], camera_from_map, K)
            u, v = uv[0]
            if depth[0] <= 0 or not (0 <= u <= WIDTH and 0 <= v <= HEIGHT):
                continue
            truth = math.hypot(point[0] - origin[0], point[1] - origin[1])
            assert floor_range([(u, v)], K, rotation, origin) == pytest.approx(truth, abs=1e-9)
            checked += 1
    assert checked > 10


def test_the_floor_range_is_the_mean_over_the_ankles_in_range():
    camera_from_map, rotation, origin = sim_view(0.0, 0.0, 0.0, 9.0)
    near, far, beyond = (
        project([ahead_of(origin, 0.0, ahead, side, ANKLE_HEIGHT_M)], camera_from_map, K)[0][0]
        for ahead, side in ((2.0, 0.1), (3.0, -0.1), (MAX_FLOOR_RANGE_M + 1.0, 0.0))
    )
    expected = (math.hypot(2.0, 0.1) + math.hypot(3.0, 0.1)) / 2
    assert floor_range([tuple(near), tuple(far), tuple(beyond)], K, rotation, origin) == (
        pytest.approx(expected, abs=1e-9)
    )


def test_no_floor_range_from_rays_that_do_not_meet_the_floor_near_enough():
    camera_from_map, rotation, origin = sim_view(0.0, 0.0, 0.0, 9.0)
    beyond = project(
        [ahead_of(origin, 0.0, MAX_FLOOR_RANGE_M + 0.5, 0.0, ANKLE_HEIGHT_M)], camera_from_map, K
    )[0][0]
    above_the_camera = project([ahead_of(origin, 0.0, 3.0, 0.0, origin[2] + 0.5)], camera_from_map, K)[0][0]
    for pixels in ([tuple(beyond)], [tuple(above_the_camera)], []):
        assert floor_range(pixels, K, rotation, origin) is None
    # A level ray through the centre of a level camera never comes down.
    level = rotation_of(map_from_camera(0.0, 0.0, 0.0, 0.0))
    assert floor_range([(CX, CY)], K, level, origin) is None


@pytest.mark.parametrize("z", [ANKLE_HEIGHT_M, 0.0, -1.0, math.nan])
def test_no_floor_range_for_a_camera_at_or_below_the_ankles_plane(z):
    camera_from_map, rotation, origin = sim_view(0.0, 0.0, 0.0, 9.0)
    ankle = project([ahead_of(origin, 0.0, 2.0, 0.0, ANKLE_HEIGHT_M)], camera_from_map, K)[0][0]
    assert floor_range([tuple(ankle)], K, rotation, (origin[0], origin[1], z)) is None


def test_no_floor_range_without_the_numbers_for_a_ray():
    camera_from_map, rotation, origin = sim_view(0.0, 0.0, 0.0, 9.0)
    ankle = tuple(project([ahead_of(origin, 0.0, 2.0, 0.0, ANKLE_HEIGHT_M)], camera_from_map, K)[0][0])
    assert floor_range([ankle], (0.0,) * 9, rotation, origin) is None
    assert floor_range([ankle], K, (0.0, 0.0, 0.0, 0.0), origin) is None


def test_the_yaw_from_the_centre_of_a_point_seen_from_the_camera():
    # A camera 0.275 m ahead of the centre, a person 2 m from it, 40 degrees left.
    yaw = math.radians(40.0)
    found = yaw_from((0.0, 0.0), (0.275, 0.0), yaw, 2.0)
    expected = math.atan2(2.0 * math.sin(yaw), 0.275 + 2.0 * math.cos(yaw))
    assert found == pytest.approx(expected, abs=1e-12)
    assert 35.0 < math.degrees(found) < 36.0
    assert yaw_from((1.0, 2.0), (1.0, 1.0), math.pi / 2, 1.0) is None


@pytest.mark.parametrize(
    ("angle", "wrapped"),
    [(0.0, 0.0), (179.5, 179.5), (180.0, -180.0), (-180.0, -180.0), (190.0, -170.0), (-190.0, 170.0), (720.0, 0.0)],
)
def test_wrap_deg_lands_in_minus_180_to_180(angle, wrapped):
    assert wrap_deg(angle) == pytest.approx(wrapped)
