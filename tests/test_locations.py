import pytest

from conftest import PROJECT_ROOT
from walkie_agent.atomics import list_places
from walkie_agent.capabilities.fake import CHULA_PAT14_6_PLACES, FakeRobot
from walkie_agent.locations import Location, load_locations


def write(tmp_path, text):
    path = tmp_path / "locations.toml"
    path.write_text(text)
    return path


def test_load_locations_reads_all_entries(locations):
    assert {"start", "kitchen", "living_room", "bedroom"} <= set(locations)
    kitchen = locations["kitchen"]
    assert (kitchen.x, kitchen.y, kitchen.yaw) == (3.5, 1.2, 1.57)
    assert "counter" in kitchen.description


def test_load_locations_defaults_yaw_and_description(tmp_path):
    door = load_locations(write(tmp_path, "[locations.door]\nx = 1\ny = 2\n"))["door"]
    assert (door.x, door.y, door.yaw, door.description) == (1.0, 2.0, 0.0, "")
    assert isinstance(door.x, float)


def test_location_stays_a_positional_dataclass():
    assert Location("door", 1.0, 2.0, 0.5, "The door.").description == "The door."


def test_sim_locations_fit_the_blank_map():
    # walkie-ros's MuJoCo map is 20 x 20 m centered on the start pose.
    sim = load_locations(PROJECT_ROOT / "config" / "locations.sim.toml")
    assert "start" in sim
    for loc in sim.values():
        assert abs(loc.x) < 10 and abs(loc.y) < 10, loc.name
        assert loc.description, loc.name


def test_chula_locations_match_fake_robot_s_world():
    # FakeRobot and the simulation must agree, or a command that works in the gym
    # drives somewhere else in MuJoCo.
    chula = load_locations(PROJECT_ROOT / "config" / "locations.chula_pat14_6.toml")
    assert {
        name: {"x": place.x, "y": place.y, "yaw": place.yaw, "room": place.room}
        for name, place in chula.items()
    } == CHULA_PAT14_6_PLACES
    assert all(place.description for place in chula.values())


def test_a_place_names_its_room(tmp_path):
    places = load_locations(
        write(
            tmp_path,
            '[locations.den]\nx = 0\ny = 0\nroom = "den"\n'
            '[locations.desk]\nx = 1\ny = 0\nroom = "den"\n'
            "[locations.hall]\nx = 2\ny = 0\n",
        )
    )
    assert [places[name].room for name in ("den", "desk", "hall")] == ["den", "den", ""]


def test_fake_robot_places_carry_their_room():
    places = FakeRobot().places()
    assert places["shelf"].room == "kitchen room"
    assert places["living room"].room == "living room"


def test_production_locations_ship_empty():
    # A place written here is where the robot really drives, so none are invented.
    assert load_locations(PROJECT_ROOT / "config" / "locations.toml") == {}


def test_a_file_with_no_locations_table_means_no_places(tmp_path):
    assert load_locations(write(tmp_path, "")) == {}


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('title = "x"\n', "title: Extra inputs"),
        ("locations = 3\n", "locations: Input should be a valid dictionary"),
        ("[locations]\ndoor = 3\n", "locations.door: Input should be a valid dictionary"),
        ("[locations.door]\nx = 1\ny = 2\nz = 3\n", "locations.door.z: Extra inputs"),
        ("[locations.door]\ny = 2\n", "locations.door.x: Field required"),
        ("[locations.door]\nx = 1\n", "locations.door.y: Field required"),
        ("[locations.door]\nx = true\ny = 2\n", "locations.door.x"),
        ('[locations.door]\nx = "1"\ny = 2\n', "locations.door.x"),
        ("[locations.door]\nx = 1\ny = false\n", "locations.door.y"),
        ('[locations.door]\nx = 1\ny = 2\nyaw = "0"\n', "locations.door.yaw"),
        ("[locations.door]\nx = nan\ny = 2\n", "not a finite number"),
        ("[locations.door]\nx = 1\ny = inf\n", "not a finite number"),
        ("[locations.door]\nx = 1\ny = 2\nyaw = -inf\n", "not a finite number"),
        ("[locations.door]\nx = 1e999\ny = 2\n", "not a finite number"),
        (f"[locations.door]\nx = 1{'0' * 400}\ny = 2\n", "too large"),
        ("[locations.door]\nx = 1\ny = 2\ndescription = 3\n", "locations.door.description"),
        ('[locations.""]\nx = 1\ny = 2\n', "blank"),
        ('[locations."  "]\nx = 1\ny = 2\n', "blank"),
        ("[locations.Kitchen]\nx = 1\ny = 2\n", "must be written 'kitchen'"),
        ('[locations." kitchen"]\nx = 1\ny = 2\n', "must be written 'kitchen'"),
        ('[locations."kitchen "]\nx = 1\ny = 2\n', "must be written 'kitchen'"),
        # The first of two spellings of one place is the one that fails.
        (
            "[locations.Kitchen]\nx = 1\ny = 2\n[locations.kitchen]\nx = 3\ny = 4\n",
            "must be written 'kitchen'",
        ),
        ("[locations.door\n", "locations.toml"),
        ("[locations.door]\nx = 1\ny = 2\nroom = 3\n", "locations.door.room"),
        ('[locations.door]\nx = 1\ny = 2\nroom = "hall"\n', "door.room 'hall' is not a place"),
        (
            '[locations.door]\nx = 1\ny = 2\nroom = "Hall"\n[locations.hall]\nx = 0\ny = 0\n',
            "door.room 'Hall' must be written 'hall'",
        ),
        # A room must name itself, so a place cannot be put in another plain place.
        (
            '[locations.door]\nx = 1\ny = 2\nroom = "hall"\n[locations.hall]\nx = 0\ny = 0\n',
            "door.room 'hall' is not a room",
        ),
    ],
)
def test_a_broken_file_is_refused_with_its_path_and_reason(tmp_path, text, error):
    path = write(tmp_path, text)
    with pytest.raises(ValueError, match=error) as caught:
        load_locations(path)
    message = str(caught.value)
    assert str(path) in message
    assert "\n" not in message


def test_a_name_may_hold_inner_spaces(tmp_path):
    places = load_locations(write(tmp_path, '[locations."dining table"]\nx = 1\ny = 2\n'))
    assert list(places) == ["dining table"]


def test_list_places_atomic_lists_names_and_descriptions():
    robot = FakeRobot()
    listed = list_places(robot.ctx)
    assert {entry["name"] for entry in listed} == set(robot.places())
    assert all(set(entry) == {"name", "description"} for entry in listed)
    assert all(entry["description"] for entry in listed)
