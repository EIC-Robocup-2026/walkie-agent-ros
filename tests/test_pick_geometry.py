"""config/pick.toml and the geometry of a pick (decision PK-4), in walkie_agent.arm.pick."""

import dataclasses
import math

import pytest

from conftest import PROJECT_ROOT
from walkie_agent.arm.pick import (
    DEFAULT_PICK_PATH,
    HandPose,
    PickSettings,
    PickSettingsError,
    approach_headings,
    check_pick_settings,
    load_pick_settings,
    plan_pick,
    relative_move,
)

VALID = {
    "arm": "right",
    "frame_id": "map",
    "standoff_m": 0.55,
    "heading_offsets_deg": [0.0, 30.0, -30.0],
    "tcp_offset_m": 0.08,
    "pregrasp_back_m": 0.10,
    "lift_m": 0.10,
    "hand_roll_deg": 0.0,
    "hand_pitch_deg": 90.0,
    "reach_timeout_s": 90.0,
    "cartesian_timeout_s": 30.0,
}
SETTINGS = check_pick_settings(VALID)
OFFSETS_PROBLEM = "heading_offsets_deg must be a non-empty list of finite numbers."


def with_(**changes) -> dict:
    return {**VALID, **changes}


def problems(data) -> tuple[str, ...]:
    with pytest.raises(PickSettingsError) as raised:
        check_pick_settings(data, source="pick.toml")
    return raised.value.errors


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-9)


# Settings


def test_the_shipped_file_loads_with_the_provisional_numbers():
    settings = load_pick_settings(PROJECT_ROOT / DEFAULT_PICK_PATH)

    assert settings == PickSettings(
        arm="right",
        frame_id="map",
        standoff_m=0.55,
        heading_offsets_deg=(0.0, 30.0, -30.0, 60.0, -60.0),
        tcp_offset_m=0.08,
        pregrasp_back_m=0.10,
        lift_m=0.10,
        hand_roll_deg=0.0,
        hand_pitch_deg=90.0,
        reach_timeout_s=90.0,
        cartesian_timeout_s=30.0,
    )


def test_integers_are_taken_as_floats_and_the_frame_is_trimmed():
    settings = check_pick_settings(with_(standoff_m=1, heading_offsets_deg=[0], frame_id=" map "))

    assert settings.standoff_m == 1.0 and isinstance(settings.standoff_m, float)
    assert settings.heading_offsets_deg == (0.0,)
    assert settings.frame_id == "map"


def test_every_problem_is_listed_at_once_with_the_source():
    data = with_(arm="middle", lift_m=0.0, extra=1)
    del data["frame_id"]

    with pytest.raises(PickSettingsError) as raised:
        check_pick_settings(data, source="config/pick.toml")

    assert raised.value.errors == (
        "Unknown key 'extra'.",
        "Missing key 'frame_id'.",
        "arm must be one of left, right, not 'middle'.",
        "lift_m must be above zero, not 0.0.",
    )
    assert str(raised.value).startswith("config/pick.toml: 4 problems:\n- Unknown key 'extra'.")


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"frame_id": "  "}, "frame_id must be a frame's name."),
        ({"frame_id": 3}, "frame_id must be a frame's name."),
        ({"standoff_m": math.nan}, "standoff_m must be a finite number, not nan."),
        ({"hand_pitch_deg": math.inf}, "hand_pitch_deg must be a finite number, not inf."),
        ({"tcp_offset_m": "0.08"}, "tcp_offset_m must be a finite number, not '0.08'."),
        ({"reach_timeout_s": True}, "reach_timeout_s must be a finite number, not True."),
        ({"cartesian_timeout_s": -1.0}, "cartesian_timeout_s must be above zero, not -1.0."),
        ({"heading_offsets_deg": []}, OFFSETS_PROBLEM),
        ({"heading_offsets_deg": 0.0}, OFFSETS_PROBLEM),
        ({"heading_offsets_deg": [0.0, math.nan]}, OFFSETS_PROBLEM),
    ],
)
def test_a_wrong_value_is_named(changes, problem):
    assert problems(with_(**changes)) == (problem,)


def test_a_zero_tcp_offset_and_negative_angles_are_allowed():
    settings = check_pick_settings(with_(tcp_offset_m=0.0, hand_roll_deg=-10.0))

    assert (settings.tcp_offset_m, settings.hand_roll_deg) == (0.0, -10.0)


def test_a_file_that_is_not_toml_is_one_problem(tmp_path):
    path = tmp_path / "pick.toml"
    path.write_text("arm = \n")

    with pytest.raises(PickSettingsError) as raised:
        load_pick_settings(path)

    assert raised.value.source == str(path)
    assert len(raised.value.errors) == 1
    assert raised.value.errors[0].startswith("Not valid TOML: ")


def test_a_missing_file_raises_os_error(tmp_path):
    with pytest.raises(OSError):
        load_pick_settings(tmp_path / "pick.toml")


# Approach headings


def test_the_headings_start_at_the_bearing_and_keep_the_settings_order():
    headings = approach_headings(SETTINGS, 0.0, 0.0, 1.0, 1.0)

    assert [round(math.degrees(h), 6) for h in headings] == [45.0, 75.0, 15.0]


def test_a_heading_past_half_a_turn_wraps():
    settings = dataclasses.replace(SETTINGS, heading_offsets_deg=(0.0, 30.0, -30.0))

    headings = approach_headings(settings, 1.0, 0.0, 0.0, 0.0)

    assert [round(math.degrees(h), 6) for h in headings] == [180.0, -150.0, 150.0]


# The plan


def test_the_plan_stands_back_along_the_heading_and_faces_the_object():
    plan = plan_pick(SETTINGS, 2.0, 1.0, 0.75, 0.0)

    assert (plan.standoff_x, plan.standoff_y, plan.heading) == (2.0 - 0.55, 1.0, 0.0)
    assert plan.grasp == HandPose(2.0 - 0.08, 1.0, 0.75, 0.0, math.pi / 2, 0.0, "map")
    assert close(plan.pregrasp.x, 2.0 - 0.18)
    assert (plan.pregrasp.y, plan.pregrasp.z) == (1.0, 0.75)
    assert plan.lift == dataclasses.replace(plan.grasp, z=0.75 + 0.10)


def test_the_plan_turns_with_the_heading():
    heading = math.radians(120.0)
    plan = plan_pick(SETTINGS, 0.0, 0.0, 0.5, heading)
    back = (-math.cos(heading), -math.sin(heading))

    assert close(plan.standoff_x, 0.55 * back[0]) and close(plan.standoff_y, 0.55 * back[1])
    for pose, back_m in ((plan.pregrasp, 0.18), (plan.grasp, 0.08), (plan.lift, 0.08)):
        assert close(pose.x, back_m * back[0]) and close(pose.y, back_m * back[1])
        assert (pose.roll, pose.pitch, pose.yaw, pose.frame_id) == (
            0.0,
            math.pi / 2,
            heading,
            "map",
        )
    assert close(plan.lift.z, 0.6)


def test_the_hand_s_tool_centre_point_lands_on_the_object():
    # With roll 0 and pitch 90 degrees the hand link's +z lies along the heading, and
    # the tool centre point is tcp_offset_m along it.
    heading = math.radians(-35.0)
    grasp = plan_pick(SETTINGS, 1.5, -2.0, 0.8, heading).grasp

    tip_x = grasp.x + SETTINGS.tcp_offset_m * math.cos(grasp.yaw) * math.sin(grasp.pitch)
    tip_y = grasp.y + SETTINGS.tcp_offset_m * math.sin(grasp.yaw) * math.sin(grasp.pitch)

    assert close(tip_x, 1.5) and close(tip_y, -2.0)


# Relative moves


def test_a_relative_move_is_in_the_start_pose_s_axes():
    assert relative_move(0.0, 0.0, 0.0, 1.0, 2.0, 0.0) == (1.0, 2.0, 0.0)
    forward, left, turn = relative_move(1.0, 1.0, math.pi / 2, 1.0, 3.0, 0.0)

    assert close(forward, 2.0) and close(left, 0.0) and close(turn, -90.0)


def test_a_relative_turn_takes_the_short_way_round():
    _, _, turn = relative_move(0.0, 0.0, math.radians(170.0), 0.0, 0.0, math.radians(-170.0))

    assert close(turn, 20.0)
