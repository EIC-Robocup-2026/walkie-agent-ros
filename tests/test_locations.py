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


def test_list_locations_tool_lists_names_and_descriptions(locations):
    output = ListLocationsTool(locations=locations).invoke({})
    assert "- kitchen: Kitchen counter." in output
    assert output.count("\n") == len(locations) - 1
