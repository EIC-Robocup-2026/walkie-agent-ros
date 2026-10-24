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
