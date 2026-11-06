"""Scene files: the loader's rules, oracle_scene's start-up, and the Chula scene.

The Chula checks draw config/scenes/chula_living_room.toml through the
oracle's draw_detections, without noise, from the `living room` place of
config/locations.chula_pat14_6.toml, with test_projection's simulated head
camera: what the agent would see there in the simulation. The furniture and
walls come from chula_pat14_6's scene.xml on walkie-ros
feat/chula-pat14-6-simulation at adb9725. No ROS graph.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

from test_projection import HEIGHT, K, WIDTH, base_from_camera
from walkie_agent.locations import load_locations
from walkie_agent.perception.gestures import classify_gestures
from walkie_agent.sim import oracle_scene
from walkie_agent.sim.oracle_perception import (
    DETECTION_SCORE,
    SEAT_CLASSES,
    OracleConfig,
    draw_detections,
)
from walkie_agent.sim.projection import camera_from_map_at, camera_position
from walkie_agent.sim.scene import COLOR_NAMES, MapBox
from walkie_agent.sim.scene_file import load_scene

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHULA_SCENE = PROJECT_ROOT / "config" / "scenes" / "chula_living_room.toml"
CHULA_LOCATIONS = PROJECT_ROOT / "config" / "locations.chula_pat14_6.toml"
STREAM_ID = "6f1c2a4e-8b3d-4f7a-9c2e-1d5b7a9e3f60"

# chula_pat14_6's walls and furniture that the scene's objects leave out, as
# x1, y1, x2, y2 footprints in the map frame: chula_tv_desk with its
# partition, and chula_room_divider's low wall. The east wall's inner face is
# at x = 3.515.
TV_DESK = (0.8775, -2.49, 1.25, -0.81)
DIVIDER = (1.73, -3.32, 3.52, -3.24)
EAST_WALL_X = 3.515
# The room a standing person needs between their centre and any furniture.
CLEARANCE_M = 0.2


def camera_info() -> CameraInfo:
    header = Header(frame_id="zed_head_left_camera_frame_optical")
    return CameraInfo(header=header, width=WIDTH, height=HEIGHT, k=list(K))


def living_room_camera() -> np.ndarray:
    place = load_locations(CHULA_LOCATIONS)["living room"]
    return camera_from_map_at(place.x, place.y, place.yaw, base_from_camera())


def footprint(box: MapBox) -> tuple[float, float, float, float]:
    assert box.yaw == 0.0, "The footprint is axis-aligned only."
    (cx, cy, _), (sx, sy, _) = box.center, box.size
    return (cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2)


def gap(x: float, y: float, rect: tuple[float, float, float, float]) -> float:
    """The distance from a point to a rectangle; 0 inside it."""
    x1, y1, x2, y2 = rect
    return math.hypot(max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2))


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "scene.toml"
    path.write_text(text)
    return path


# -- The Chula scene ----------------------------------------------------------------------


def test_the_chula_scene_loads():
    loaded = load_scene(CHULA_SCENE)

    assert loaded.world == "chula_pat14_6"
    assert [p.track_id for p in loaded.scene.people] == [1, 2]
    assert [(o.label, o.color) for o in loaded.scene.objects] == [
        ("sofa", "grey"),
        ("table", "grey"),
        ("mug", "blue"),
    ]
    # The oracle's seat classes stay, so seat detection sees what it would by default.
    assert set(SEAT_CLASSES) <= set(loaded.classes)


def test_from_the_living_room_both_people_are_whole_in_view_with_their_gestures():
    loaded = load_scene(CHULA_SCENE)
    config = OracleConfig(
        pixel_noise_px=0.0, vector_noise=0.0, outlier_rate=0.0, classes=loaded.classes
    )

    people, objects = draw_detections(
        loaded.scene,
        camera_info(),
        living_room_camera(),
        config,
        np.random.default_rng(0),
        stream_id=STREAM_ID,
    )

    seen = {p.track_id: p for p in people.people}
    assert sorted(seen) == [1, 2]
    for person in seen.values():
        # Every keypoint in the image, and the face found.
        assert np.all(np.asarray(person.keypoint_scores) > 0.0)
        assert person.face_det_score == pytest.approx(DETECTION_SCORE)
    guest, host = seen[1], seen[2]
    # The guest raises the left arm; the rules read any raised hand as a wave.
    assert classify_gestures(guest.keypoints, guest.keypoint_scores) == {
        "raising_left_arm",
        "standing",
        "waving",
    }
    assert classify_gestures(host.keypoints, host.keypoint_scores) == {"sitting"}
    assert sorted(d.label for d in objects.detections) == ["mug", "sofa", "table"]
    [mug] = [d for d in objects.detections if d.label == "mug"]
    assert int(np.argmax(mug.color_shares)) == COLOR_NAMES.index("blue")


def test_the_chula_people_face_the_living_room_camera():
    loaded = load_scene(CHULA_SCENE)
    camera_xy = tuple(camera_position(living_room_camera())[:2])

    assert all(person.faces_camera(camera_xy) for person in loaded.scene.people)


def test_a_standing_person_keeps_clear_of_the_furniture_and_walls():
    loaded = load_scene(CHULA_SCENE)
    furniture = [footprint(o.box) for o in loaded.scene.objects] + [TV_DESK, DIVIDER]

    standing = [p for p in loaded.scene.people if p.posture == "standing"]
    assert standing
    for person in standing:
        assert min(gap(person.x, person.y, rect) for rect in furniture) >= CLEARANCE_M
        assert EAST_WALL_X - person.x >= CLEARANCE_M


def test_the_host_sits_on_the_sofas_cushions():
    loaded = load_scene(CHULA_SCENE)
    [sofa] = [o for o in loaded.scene.objects if o.label == "sofa"]
    [host] = [p for p in loaded.scene.people if p.posture == "sitting"]

    assert gap(host.x, host.y, footprint(sofa.box)) == 0.0
    # chula_sofa's cushions top out at 0.4575 m.
    assert host.seat_top == pytest.approx(0.4575)


def test_the_mug_stands_on_the_coffee_table():
    objects = {o.label: o for o in load_scene(CHULA_SCENE).scene.objects}
    mug, table = objects["mug"].box, objects["table"].box

    x1, y1, x2, y2 = footprint(table)
    mx1, my1, mx2, my2 = footprint(mug)
    assert x1 <= mx1 and mx2 <= x2 and y1 <= my1 and my2 <= y2
    table_top = table.center[2] + table.size[2] / 2
    assert mug.center[2] - mug.size[2] / 2 == pytest.approx(table_top, abs=0.005)


# -- The loader's rules -------------------------------------------------------------------


def test_classes_default_to_the_objects_labels_in_file_order(tmp_path):
    path = write(
        tmp_path,
        """
[[objects]]
track_id = 1
label = "table"
center = [0, 0, 0.4]
size = [1, 1, 0.8]

[[objects]]
track_id = 2
label = "mug"
center = [0, 0, 0.85]
size = [0.1, 0.1, 0.1]

[[objects]]
track_id = 3
label = "table"
center = [2, 0, 0.4]
size = [1, 1, 0.8]
""",
    )

    assert load_scene(path).classes == ("table", "mug")


def test_a_file_with_no_objects_leaves_the_oracles_classes(tmp_path):
    path = write(tmp_path, '[[people]]\ntrack_id = 1\nx = 1\ny = 2\nfacing_deg = 90\n')

    loaded = load_scene(path)

    assert loaded.classes is None and loaded.world == ""
    [person] = loaded.scene.people
    assert (person.x, person.y, person.posture) == (1.0, 2.0, "standing")
    assert person.facing == pytest.approx(math.pi / 2)


PERSON = "[[people]]\ntrack_id = 1\nx = 1\ny = 2\nfacing_deg = 0\n"
MUG = '[[objects]]\ntrack_id = 5\nlabel = "mug"\ncenter = [0, 0, 1]\nsize = [0.1, 0.1, 0.1]\n'


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (PERSON + "hat = 'red'\n", r"people\.0\.hat"),
        ("colour = 1\n", "colour"),
        (PERSON.replace("x = 1", "x = '1'"), r"people\.0\.x"),
        (PERSON.replace("x = 1", "x = true"), r"people\.0\.x"),
        (PERSON.replace("x = 1", "x = nan"), "finite"),
        (PERSON.replace("x = 1", "x = 1e999"), "finite"),
        (PERSON + 'gesture = "dancing"\n', r"people\.0: .*gesture"),
        (PERSON + 'posture = "kneeling"\n', r"people\.0: .*posture"),
        (PERSON + "height = 0\n", r"people\.0: .*height"),
        (MUG + 'color = "teal"\n', r"objects\.0: .*color"),
        (MUG + "view_quality = 1.5\n", r"objects\.0: .*view_quality"),
        (MUG.replace('"mug"', '""'), r"objects\.0: .*label"),
        (MUG.replace("size = [0.1, 0.1, 0.1]", "size = [0.1, 0.1]"), r"objects\.0\.size"),
        (PERSON + PERSON, r"people share the track ids \[1\]"),
        (MUG + MUG, r"objects share the track ids \[5\]"),
        ('classes = ["sofa"]\n' + MUG, r"labels \['mug'\] are not in classes"),
        ('classes = ["mug", "mug"]\n' + MUG, "unique"),
        ("classes = []\n", "at least one"),
        ("people = [", "scene.toml"),
    ],
)
def test_a_file_that_breaks_the_rules_is_refused_with_its_reason(tmp_path, text, match):
    path = write(tmp_path, text)

    with pytest.raises(ValueError, match=match) as error:
        load_scene(path)

    # One line that names the file.
    assert str(error.value).startswith(f"{path}: ")
    assert "\n" not in str(error.value)


def test_untracked_entries_may_share_track_id_zero(tmp_path):
    untracked = PERSON.replace("track_id = 1", "track_id = 0")

    loaded = load_scene(write(tmp_path, untracked + untracked))

    assert [p.track_id for p in loaded.scene.people] == [0, 0]


# -- oracle_scene's start-up --------------------------------------------------------------


def test_the_command_line_defaults_to_the_head_camera_and_wall_time():
    args = oracle_scene.parse_args(["scene.toml"])

    assert (args.scene, args.use_sim_time, args.seed) == ("scene.toml", False, 0)
    assert args.camera_info_topic == "/zed_head/zed_node/depth/camera_info"


def test_the_oracle_config_takes_the_files_classes_and_the_command_line():
    args = oracle_scene.parse_args(["--seed", "4", "--camera-info-topic", "/cam", "x.toml"])

    config = oracle_scene.oracle_config(load_scene(CHULA_SCENE), args)

    assert (config.seed, config.camera_info_topic) == (4, "/cam")
    assert config.classes == load_scene(CHULA_SCENE).classes


def test_a_scene_without_classes_keeps_the_oracles_seat_classes(tmp_path):
    args = oracle_scene.parse_args(["x.toml"])

    config = oracle_scene.oracle_config(load_scene(write(tmp_path, PERSON)), args)

    assert config.classes == SEAT_CLASSES


@pytest.mark.parametrize("text", [None, PERSON + PERSON])
def test_a_missing_or_bad_scene_file_stops_the_runner_with_exit_code_2(tmp_path, capsys, text):
    path = tmp_path / "scene.toml" if text is None else write(tmp_path, text)

    code = oracle_scene.main([str(path)])

    assert code == oracle_scene.STARTUP_ERROR_EXIT_CODE == 2
    err = capsys.readouterr().err
    assert err.startswith("oracle_scene: ") and str(path) in err
