import math
import tomllib

import pytest
from conftest import PROJECT_ROOT
from walkie_agent.locations import load_locations
from walkie_agent.tools import ListLocationsTool


def test_load_locations_reads_all_entries(locations):
    assert {"start", "kitchen", "living_room", "bedroom"} <= set(locations)
    kitchen = locations["kitchen"]
    assert (kitchen.x, kitchen.y, kitchen.yaw) == (3.5, 1.2, 1.57)
    assert "counter" in kitchen.description


def test_load_locations_defaults_yaw_and_description(tmp_path):
    path = tmp_path / "locations.toml"
    path.write_text("[locations.door]\nx = 1\ny = 2\n")
    door = load_locations(path)["door"]
    assert (door.x, door.y, door.yaw, door.description) == (1.0, 2.0, 0.0, "")


def test_load_locations_refuses_places_without_x_and_y(tmp_path):
    path = tmp_path / "locations.toml"
    path.write_text(
        "[locations.door]\ndescription = 'At the door.'\n"
        "[locations.sofa]\nx = 1.0\ny = 2.0\n"
        "[locations.kitchen]\nx = 1.0\n"
        "[locations.hall]\nx = 'here'\ny = 2.0\n"
        "[locations.porch]\nx = nan\ny = 1.0\n"
        "[locations.yard]\nx = 1.0\ny = inf\n"
    )
    with pytest.raises(ValueError) as e:
        load_locations(path)
    message = str(e.value)
    assert str(path) in message
    assert "door, kitchen, hall, porch, yard have no measured x and y" in message
    assert "sofa" not in message


@pytest.mark.parametrize("yaw", ["nan", "inf", "true", "'north'"])
def test_load_locations_refuses_a_yaw_that_is_not_a_finite_number(tmp_path, yaw):
    path = tmp_path / "locations.toml"
    path.write_text(
        f"[locations.door]\nx = 1.0\ny = 2.0\nyaw = {yaw}\n[locations.sofa]\nx = 1.0\ny = 2.0\n"
    )
    with pytest.raises(ValueError) as e:
        load_locations(path)
    message = str(e.value)
    assert "door has a yaw that is not a finite number" in message
    assert "sofa" not in message


def _measured(entry: dict) -> bool:
    return all(
        isinstance(entry.get(key), int | float)
        and not isinstance(entry.get(key), bool)
        and math.isfinite(entry[key])
        for key in ("x", "y")
    )


def test_robot_locations_load_only_once_measured():
    # The file lists the places the HRI template drives to. It ships without
    # coordinates, so the robot run cannot start on made-up poses, and it
    # loads once the operator has measured them on the arena map.
    path = PROJECT_ROOT / "config" / "locations.robot.toml"
    with open(path, "rb") as f:
        places = tomllib.load(f)["locations"]
    assert {"door", "living_room"} <= set(places)
    for name, entry in places.items():
        assert entry["description"], name
    unmeasured = [name for name, entry in places.items() if not _measured(entry)]
    if unmeasured:
        with pytest.raises(ValueError) as e:
            load_locations(path)
        verb = "has" if len(unmeasured) == 1 else "have"
        assert f"{', '.join(unmeasured)} {verb} no measured x and y" in str(e.value)
    else:
        assert {"door", "living_room"} <= set(load_locations(path))


def test_sim_locations_fit_the_blank_map():
    # walkie-ros's MuJoCo map is 20 x 20 m centered on the start pose.
    sim = load_locations(PROJECT_ROOT / "config" / "locations.sim.toml")
    assert "start" in sim
    for loc in sim.values():
        assert abs(loc.x) < 10 and abs(loc.y) < 10, loc.name
        assert loc.description, loc.name


# The rooms of walkie-ros's world chula_pat14_6 as x and y bounds in its map
# frame, in meters: the room_polygons of simulation/mujoco/worlds/chula_pat14_6/
# authoring/source/reconstruction_geometry.json on feat/chula-pat14-6-simulation
# (adb9725), which are axis-aligned rectangles, rounded to the millimeter.
CHULA_ROOMS = {
    "living room": ((0.872, 3.572), (-3.363, 0.391)),
    "kitchen": ((-0.362, 3.546), (-7.610, -3.395)),
}


def test_chula_locations_lie_in_the_rooms():
    places = load_locations(PROJECT_ROOT / "config" / "locations.chula_pat14_6.toml")
    # The places the HRI template drives to.
    assert {"door", "living_room"} <= set(places)
    for loc in places.values():
        assert loc.description, loc.name
        assert any(
            x_min <= loc.x <= x_max and y_min <= loc.y <= y_max
            for (x_min, x_max), (y_min, y_max) in CHULA_ROOMS.values()
        ), loc.name


def test_list_locations_tool_lists_names_and_descriptions(locations):
    output = ListLocationsTool(locations=locations).invoke({})
    assert "- kitchen: Kitchen counter." in output
    assert output.count("\n") == len(locations) - 1
