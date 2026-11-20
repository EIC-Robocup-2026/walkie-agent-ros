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
    heading_from,
    load_pick_settings,
    plan_pick,
    reach_problems,
    relative_move,
    too_wide_to_grasp,
    width_across_fingers,
)

VALID = {
    "arm": "right",
    "frame_id": "map",
    "standoff_m": 0.65,
    "standoff_left_m": 0.20,
    "heading_offsets_deg": [0.0, 30.0, -30.0],
    "grasp_back_m": 0.06,
    "pregrasp_back_m": 0.10,
    "lift_m": 0.10,
    "clear_forward_m": 0.30,
    "clear_side_m": 0.35,
    "home_lift_m": 0.7435,
    "hand_roll_deg": 0.0,
    "hand_pitch_deg": -90.0,
    "hand_yaw_offset_deg": 180.0,
    "reach_timeout_s": 90.0,
    "cartesian_timeout_s": 30.0,
    "max_observation_age_s": 5.0,
    "observation_wait_s": 6.0,
    "max_target_shift_m": 0.15,
    "min_target_z_m": 0.55,
    "max_target_z_m": 1.15,
    "min_reach_m": 0.35,
    "max_reach_m": 0.90,
    "min_reach_z_m": 0.50,
    "max_reach_z_m": 1.30,
    "max_grasp_width_m": 0.078,
}
SETTINGS = check_pick_settings(VALID)
OFFSETS_PROBLEM = "heading_offsets_deg must be a non-empty list of finite numbers."
# openarm_right_hand_tcp's offset along the hand link's +z, in walkie-ros's URDF.
TCP_OFFSET_M = 0.08


def with_(**changes) -> dict:
    return {**VALID, **changes}


def problems(data) -> tuple[str, ...]:
    with pytest.raises(PickSettingsError) as raised:
        check_pick_settings(data, source="pick.toml")
    return raised.value.errors


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-9)


def fingers(pose: HandPose) -> tuple[float, float, float]:
    """The hand link's +z in the frame, for tf2's setRPY(roll, pitch, yaw)."""
    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    return (cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr)


# Settings


def test_the_shipped_file_loads_with_the_numbers_the_walkie_ros_pick_proved():
    settings = load_pick_settings(PROJECT_ROOT / DEFAULT_PICK_PATH)

    assert settings == PickSettings(
        arm="right",
        frame_id="map",
        standoff_m=0.65,
        standoff_left_m=0.20,
        heading_offsets_deg=(0.0, 30.0, -30.0, 60.0, -60.0, 120.0, -120.0, 180.0),
        grasp_back_m=0.06,
        pregrasp_back_m=0.10,
        lift_m=0.10,
        clear_forward_m=0.30,
        clear_side_m=0.35,
        home_lift_m=0.7435,
        hand_roll_deg=0.0,
        hand_pitch_deg=-90.0,
        hand_yaw_offset_deg=180.0,
        reach_timeout_s=90.0,
        cartesian_timeout_s=30.0,
        max_observation_age_s=5.0,
        observation_wait_s=6.0,
        max_target_shift_m=0.15,
        min_target_z_m=0.55,
        max_target_z_m=1.15,
        min_reach_m=0.35,
        max_reach_m=0.82,
        min_reach_z_m=0.50,
        max_reach_z_m=1.30,
        max_grasp_width_m=0.078,
    )


def test_the_shipped_envelope_holds_the_proven_chula_pick_with_the_base_a_little_off():
    settings = load_pick_settings(PROJECT_ROOT / DEFAULT_PICK_PATH)
    # The Chula water bottle's centre, which walkie-ros's scripted pick grasped.
    bottle = (0.94, -5.57, 0.862)
    plan = plan_pick(settings, *bottle, 0.7)

    for forward in (-0.10, 0.0, 0.10):
        base_x = plan.standoff_x + forward * math.cos(plan.heading)
        base_y = plan.standoff_y + forward * math.sin(plan.heading)
        approach = math.atan2(bottle[1] - base_y, bottle[0] - base_x)
        stopped = plan_pick(settings, *bottle, 0.7, approach, (base_x, base_y, plan.heading))

        assert reach_problems(settings, stopped, bottle, (base_x, base_y, plan.heading)) == []


# Where base_footprint stood (x, y, yaw) for scripted picks in the Chula simulation on
# 2026-11-21, and the object's centre: the straight approach held the bottle from the
# first two and could not be planned from the last two (the bottle, then a can).
HELD_FROM = [
    ((0.7249, -6.3377, 1.5796), (0.94, -5.57, 0.862)),
    ((0.6855, -6.3436, 1.6465), (0.94, -5.57, 0.862)),
]
FAILED_FROM = [
    ((0.7212, -6.3872, 1.5805), (0.94, -5.57, 0.862)),
    ((1.3453, -4.3317, -1.1093), (1.65, -5.12, 0.867)),
]


def plan_from(settings, base, target):
    heading = heading_from(settings, base[0], base[1], target[0], target[1])
    approach = math.atan2(target[1] - base[1], target[0] - base[0])
    return plan_pick(settings, *target, heading, approach, base)


@pytest.mark.parametrize(("base", "target"), HELD_FROM)
def test_the_shipped_envelope_keeps_the_furthest_picks_that_held_in_simulation(base, target):
    settings = load_pick_settings(PROJECT_ROOT / DEFAULT_PICK_PATH)

    assert reach_problems(settings, plan_from(settings, base, target), target, base) == []


@pytest.mark.parametrize(("base", "target"), FAILED_FROM)
def test_the_shipped_envelope_refuses_the_picks_whose_straight_approach_failed(base, target):
    settings = load_pick_settings(PROJECT_ROOT / DEFAULT_PICK_PATH)

    problems = reach_problems(settings, plan_from(settings, base, target), target, base)

    assert [line.split(" is ")[0] for line in problems] == ["the object"]


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
        ({"grasp_back_m": "0.06"}, "grasp_back_m must be a finite number, not '0.06'."),
        ({"standoff_left_m": None}, "standoff_left_m must be a finite number, not None."),
        ({"reach_timeout_s": True}, "reach_timeout_s must be a finite number, not True."),
        ({"cartesian_timeout_s": -1.0}, "cartesian_timeout_s must be above zero, not -1.0."),
        ({"heading_offsets_deg": []}, OFFSETS_PROBLEM),
        ({"heading_offsets_deg": 0.0}, OFFSETS_PROBLEM),
        ({"heading_offsets_deg": [0.0, math.nan]}, OFFSETS_PROBLEM),
        ({"max_observation_age_s": 0.0}, "max_observation_age_s must be above zero, not 0.0."),
        ({"observation_wait_s": -1.0}, "observation_wait_s must be above zero, not -1.0."),
        ({"max_target_shift_m": 0.0}, "max_target_shift_m must be above zero, not 0.0."),
        ({"min_reach_m": 0.0}, "min_reach_m must be above zero, not 0.0."),
        ({"min_target_z_m": 1.15}, "min_target_z_m must be below max_target_z_m, not 1.15."),
        ({"max_reach_m": 0.30}, "min_reach_m must be below max_reach_m, not 0.35."),
        ({"min_reach_z_m": 1.4}, "min_reach_z_m must be below max_reach_z_m, not 1.4."),
        ({"max_grasp_width_m": 0.0}, "max_grasp_width_m must be above zero, not 0.0."),
    ],
)
def test_a_wrong_value_is_named(changes, problem):
    assert problems(with_(**changes)) == (problem,)


def test_a_range_with_a_wrong_end_reports_only_that_end():
    assert problems(with_(max_target_z_m=math.nan)) == (
        "max_target_z_m must be a finite number, not nan.",
    )


def test_zero_and_negative_offsets_and_angles_are_allowed():
    settings = check_pick_settings(
        with_(grasp_back_m=0.0, standoff_left_m=-0.2, hand_roll_deg=-10.0, hand_yaw_offset_deg=0)
    )

    assert (
        settings.grasp_back_m,
        settings.standoff_left_m,
        settings.hand_roll_deg,
        settings.hand_yaw_offset_deg,
    ) == (0.0, -0.2, -10.0, 0.0)


def test_the_old_tool_centre_point_key_is_unknown():
    data = {**VALID, "tcp_offset_m": 0.08}
    del data["grasp_back_m"]

    assert problems(data) == ("Unknown key 'tcp_offset_m'.", "Missing key 'grasp_back_m'.")


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


def test_without_a_side_offset_the_headings_start_at_the_bearing_in_the_settings_order():
    settings = dataclasses.replace(SETTINGS, standoff_left_m=0.0)

    headings = approach_headings(settings, 0.0, 0.0, 1.0, 1.0)

    assert [round(math.degrees(h), 6) for h in headings] == [45.0, 75.0, 15.0]


def test_the_first_heading_is_the_one_whose_line_the_robot_stands_on():
    headings = approach_headings(SETTINGS, 0.0, 0.0, 1.0, 0.0)

    assert close(headings[0], heading_from(SETTINGS, 0.0, 0.0, 1.0, 0.0))
    assert close(headings[0], math.asin(0.20))
    assert close(headings[1] - headings[0], math.radians(30.0))


def test_a_heading_past_half_a_turn_wraps():
    settings = dataclasses.replace(
        SETTINGS, standoff_left_m=0.0, heading_offsets_deg=(0.0, 30.0, -30.0)
    )

    headings = approach_headings(settings, 1.0, 0.0, 0.0, 0.0)

    assert [round(math.degrees(h), 6) for h in headings] == [180.0, -150.0, 150.0]


def test_the_heading_from_a_stand_off_pose_is_the_heading_it_was_planned_for():
    for degrees in (0.0, 90.0, -135.0, 170.0):
        heading = math.radians(degrees)
        plan = plan_pick(SETTINGS, 1.0, -2.0, 0.8, heading)

        found = heading_from(SETTINGS, plan.standoff_x, plan.standoff_y, 1.0, -2.0)

        assert close(found, math.atan2(math.sin(heading), math.cos(heading)))


def test_the_heading_from_further_back_keeps_the_object_ahead_of_the_right_hand():
    # The object 0.20 m to the right of the robot's line, at any distance ahead.
    heading = heading_from(SETTINGS, 0.0, 0.2, 2.0, 0.0)

    # The line from the object back along the heading passes 0.20 m right of the robot.
    assert close(heading, 0.0)
    lateral = -(0.0 - 2.0) * math.sin(heading) + (0.2 - 0.0) * math.cos(heading)
    assert close(lateral, 0.20)


def test_the_heading_from_closer_than_the_side_offset_is_the_bearing():
    assert close(heading_from(SETTINGS, 0.0, 0.0, 0.1, 0.1), math.radians(45.0))


# The plan


# The line from the stand-off pose to the object turns this far right of the heading.
STANDOFF_TURN = math.atan2(0.20, 0.65)


def test_the_plan_stands_back_and_to_the_left_and_faces_along_the_heading():
    plan = plan_pick(SETTINGS, 2.0, 1.0, 0.75, 0.0)

    assert (plan.standoff_x, plan.standoff_y, plan.heading) == (2.0 - 0.65, 1.0 + 0.20, 0.0)
    # The hand moves in along the line from the stand-off pose to the object.
    assert close(plan.approach, -STANDOFF_TURN)
    along = (math.cos(plan.approach), math.sin(plan.approach))
    assert close(plan.grasp.x, 2.0 - 0.06 * along[0])
    assert close(plan.grasp.y, 1.0 - 0.06 * along[1])
    assert (plan.grasp.z, plan.grasp.roll, plan.grasp.pitch) == (0.75, 0.0, -math.pi / 2)
    assert close(plan.grasp.yaw, math.pi - STANDOFF_TURN)
    assert close(plan.pregrasp.x, 2.0 - 0.16 * along[0])
    assert close(plan.pregrasp.y, 1.0 - 0.16 * along[1])
    assert plan.pregrasp.z == 0.75
    assert plan.lift == dataclasses.replace(plan.grasp, z=0.75 + 0.10)


def test_the_plan_turns_with_the_heading():
    heading = math.radians(120.0)
    plan = plan_pick(SETTINGS, 0.0, 0.0, 0.5, heading)
    back = (-math.cos(heading), -math.sin(heading))
    left = (-math.sin(heading), math.cos(heading))
    approach = heading - STANDOFF_TURN

    assert close(plan.standoff_x, 0.65 * back[0] + 0.20 * left[0])
    assert close(plan.standoff_y, 0.65 * back[1] + 0.20 * left[1])
    assert close(plan.approach, approach)
    for pose, back_m in ((plan.pregrasp, 0.16), (plan.grasp, 0.06), (plan.lift, 0.06)):
        assert close(pose.x, -back_m * math.cos(approach))
        assert close(pose.y, -back_m * math.sin(approach))
        assert (pose.roll, pose.pitch, pose.frame_id) == (0.0, -math.pi / 2, "map")
        assert close(pose.yaw, approach - math.pi)
    assert close(plan.lift.z, 0.6)


def test_a_given_approach_moves_the_hand_and_leaves_the_stand_off():
    default = plan_pick(SETTINGS, 1.0, -2.0, 0.8, math.radians(30.0))
    plan = plan_pick(SETTINGS, 1.0, -2.0, 0.8, math.radians(30.0), math.radians(10.0))

    assert (plan.standoff_x, plan.standoff_y, plan.heading) == (
        default.standoff_x,
        default.standoff_y,
        default.heading,
    )
    assert close(plan.approach, math.radians(10.0))
    assert close(plan.grasp.x, 1.0 - 0.06 * math.cos(math.radians(10.0)))
    assert close(plan.grasp.y, -2.0 - 0.06 * math.sin(math.radians(10.0)))
    assert close(plan.grasp.yaw, math.radians(-170.0))


def test_the_clear_pose_is_beside_the_stand_off_pose_at_the_lifted_height():
    plan = plan_pick(SETTINGS, 2.0, 1.0, 0.75, 0.0)

    # The right arm's side is the base's right.
    assert close(plan.clear.x, plan.standoff_x + 0.30)
    assert close(plan.clear.y, plan.standoff_y - 0.35)
    assert plan.clear == dataclasses.replace(plan.lift, x=plan.clear.x, y=plan.clear.y)


def test_the_clear_pose_follows_where_the_base_stopped_and_the_picking_arm():
    plan = plan_pick(SETTINGS, 2.0, 1.0, 0.75, 0.0, base=(1.0, 1.0, math.pi / 2))
    left = plan_pick(
        dataclasses.replace(SETTINGS, arm="left"), 2.0, 1.0, 0.75, 0.0, base=(1.0, 1.0, math.pi / 2)
    )

    # Facing +y, ahead is +y and the right side is +x.
    assert close(plan.clear.x, 1.0 + 0.35) and close(plan.clear.y, 1.0 + 0.30)
    assert close(left.clear.x, 1.0 - 0.35) and close(left.clear.y, 1.0 + 0.30)
    assert plan.clear.z == plan.lift.z


def test_the_fingers_point_along_the_approach_and_the_tool_centre_point_passes_the_centre():
    # Pitch -90 with the yaw turned half a turn points the hand link's +z along the
    # approach, and the tool centre point lies TCP_OFFSET_M along it.
    approach = math.radians(-35.0)
    grasp = plan_pick(SETTINGS, 1.5, -2.0, 0.8, 0.0, approach).grasp
    zx, zy, zz = fingers(grasp)

    assert close(zx, math.cos(approach)) and close(zy, math.sin(approach)) and close(zz, 0.0)
    past = (grasp.x + TCP_OFFSET_M * zx - 1.5) * zx + (grasp.y + TCP_OFFSET_M * zy + 2.0) * zy
    assert close(past, TCP_OFFSET_M - SETTINGS.grasp_back_m)


# The reach envelope


def stopped_at_the_stand_off(x: float, y: float, z: float, heading: float = 0.0):
    plan = plan_pick(SETTINGS, x, y, z, heading)
    return plan, (plan.standoff_x, plan.standoff_y, plan.heading)


def test_a_plan_from_its_own_stand_off_pose_is_inside_the_envelope():
    plan, base = stopped_at_the_stand_off(2.0, 1.0, 0.86)

    assert reach_problems(SETTINGS, plan, (2.0, 1.0, 0.86), base) == []


@pytest.mark.parametrize(
    ("z", "problems_"),
    [
        (
            0.40,
            [
                "the object's centre is 0.400 m high, outside [0.55, 1.15]",
                "the pregrasp pose is 0.400 m high, outside [0.5, 1.3]",
                "the grasp pose is 0.400 m high, outside [0.5, 1.3]",
            ],
        ),
        (
            1.25,
            [
                "the object's centre is 1.250 m high, outside [0.55, 1.15]",
                "the lift pose is 1.350 m high, outside [0.5, 1.3]",
                "the clear pose is 1.350 m high, outside [0.5, 1.3]",
            ],
        ),
    ],
)
def test_an_object_too_low_or_too_high_is_named_with_the_poses_it_moves_out(z, problems_):
    plan, base = stopped_at_the_stand_off(2.0, 1.0, z)

    assert reach_problems(SETTINGS, plan, (2.0, 1.0, z), base) == problems_


def test_an_object_the_memory_moved_past_the_reach_is_named_pose_by_pose():
    # The plan was made for (2.0, 1.0); the object is now 0.30 m further along +x.
    plan, base = stopped_at_the_stand_off(2.0, 1.0, 0.86)
    moved = plan_pick(SETTINGS, 2.3, 1.0, 0.86, 0.0, None, base)

    found = reach_problems(SETTINGS, moved, (2.3, 1.0, 0.86), base)

    # The pre-grasp pose, 0.16 m short of the object, still lies inside.
    assert [line.split(" is ")[0] for line in found] == [
        "the object",
        "the grasp pose",
        "the lift pose",
    ]
    assert found[0] == "the object is 0.971 m from the base, outside [0.35, 0.9]"


def test_an_object_beside_the_base_is_too_close():
    base = (2.0, 1.0, 0.0)
    plan = plan_pick(SETTINGS, 2.1, 1.0, 0.86, 0.0, 0.0, base)

    found = reach_problems(SETTINGS, plan, (2.1, 1.0, 0.86), base)

    assert found[0] == "the object is 0.100 m from the base, outside [0.35, 0.9]"
    assert "the clear pose" not in " ".join(found)


# The open gripper's width


@pytest.mark.parametrize(
    "size",
    [
        # A water bottle and a soda can, as the object memory's upright box: the side
        # along the box's yaw, the side across it, and the height.
        (0.068, 0.068, 0.22),
        (0.076, 0.076, 0.12),
        # A side exactly at the limit still fits.
        (0.078, 0.078, 0.10),
        # A box long one way: the approach turns the fingers across its narrower side.
        (0.14, 0.068, 0.22),
    ],
    ids=["bottle", "can", "at-limit", "long-box"],
)
def test_an_object_as_narrow_as_the_open_gripper_fits(size):
    assert too_wide_to_grasp(SETTINGS, size) is None


@pytest.mark.parametrize("size", [(0.12, 0.082, 0.10), (0.082, 0.12, 0.10)], ids=["x", "y"])
def test_an_object_wider_than_the_open_gripper_gives_its_narrower_side(size):
    assert too_wide_to_grasp(SETTINGS, size) == 0.082


@pytest.mark.parametrize(
    "size",
    [None, (0.0, 0.0, 0.0), (0.12, 0.0, 0.10), (math.nan, 0.12, 0.10), (math.inf, 0.12, 0.10)],
    ids=["none", "zero", "zero-side", "nan", "inf"],
)
def test_an_object_with_no_usable_size_is_not_refused(size):
    assert too_wide_to_grasp(SETTINGS, size) is None


# The width across the fingers

# A box 0.14 m long along its yaw and 0.068 m across it: wider than the open gripper
# end to end, narrow enough across.
LONG_BOX = (0.14, 0.068, 0.22)


def closes_along(pose: HandPose) -> tuple[float, float, float]:
    """The hand link's +y in the frame, along which the fingers close (openarm_hand.xacro)."""
    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    return (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr)


@pytest.mark.parametrize("box_deg", [0.0, 25.0, -110.0])
@pytest.mark.parametrize("hand_deg", [0.0, 30.0, 45.0, 90.0, 160.0])
def test_a_round_object_s_square_box_is_as_wide_across_the_fingers_at_any_yaw(box_deg, hand_deg):
    width = width_across_fingers(
        (0.068, 0.068, 0.22), math.radians(box_deg), math.radians(hand_deg)
    )

    assert width == pytest.approx(0.068)


@pytest.mark.parametrize(
    ("hand_deg", "width"),
    [
        # Fingers along the box's yaw meet its 0.14 m side; square to it, the 0.068 m.
        (-90.0, 0.14),
        (90.0, 0.14),
        (0.0, 0.068),
        (180.0, 0.068),
        # Between the two: the ellipse inside the box, not its corners.
        (-45.0, math.hypot(0.14, 0.068) / math.sqrt(2.0)),
    ],
)
def test_a_long_box_is_as_wide_across_the_fingers_as_the_ellipse_inside_it(hand_deg, width):
    assert width_across_fingers(LONG_BOX, 0.0, math.radians(hand_deg)) == pytest.approx(width)


@pytest.mark.parametrize(
    ("size", "yaw"),
    [
        (None, 0.0),
        (LONG_BOX, None),
        (LONG_BOX, math.nan),
        (LONG_BOX, math.inf),
        ((0.14, 0.0, 0.22), 0.0),
        ((math.nan, 0.068, 0.22), 0.0),
    ],
    ids=["no-size", "no-yaw", "nan-yaw", "inf-yaw", "zero-side", "nan-side"],
)
def test_a_box_with_no_usable_side_or_yaw_has_no_width_across_the_fingers(size, yaw):
    assert width_across_fingers(size, yaw, 0.0) is None


def test_a_round_object_as_wide_as_the_open_hand_fits_it_at_every_hand_yaw():
    size = (SETTINGS.max_grasp_width_m, SETTINGS.max_grasp_width_m, 0.1)

    widths = [width_across_fingers(size, 0.0, math.radians(tenth / 10)) for tenth in range(3600)]

    assert max(widths) <= SETTINGS.max_grasp_width_m


def test_a_long_box_lying_along_the_fingers_of_the_robot_s_own_line_gets_turned_headings():
    # Codex's case: from the robot's own line to the box, the hand would close across
    # its 0.14 m length, although its 0.068 m side passes the gate before the drive.
    first = approach_headings(SETTINGS, 0.0, 0.0, 1.0, 0.0)[0]
    yaw = plan_pick(SETTINGS, 1.0, 0.0, 0.8, first).grasp.yaw + math.pi / 2
    assert too_wide_to_grasp(SETTINGS, LONG_BOX) is None
    assert width_across_fingers(
        LONG_BOX, yaw, plan_pick(SETTINGS, 1.0, 0.0, 0.8, first).grasp.yaw
    ) == pytest.approx(0.14)

    headings = approach_headings(SETTINGS, 0.0, 0.0, 1.0, 0.0, LONG_BOX, yaw)

    assert len(headings) == 2
    for heading in headings:
        grasp = plan_pick(SETTINGS, 1.0, 0.0, 0.8, heading).grasp
        assert width_across_fingers(LONG_BOX, yaw, grasp.yaw) == pytest.approx(0.068)


@pytest.mark.parametrize("size", [LONG_BOX, (0.068, 0.14, 0.22)], ids=["along-yaw", "across-yaw"])
@pytest.mark.parametrize("box_deg", [0.0, 37.0, 90.0, -120.0, 180.0])
def test_a_long_box_is_approached_so_the_fingers_close_across_its_narrow_side(size, box_deg):
    yaw = math.radians(box_deg)
    # The narrow side's direction in the frame: along the yaw when it is the first side.
    narrow = yaw if size[0] < size[1] else yaw + math.pi / 2
    start = heading_from(SETTINGS, 0.0, 0.0, 1.0, 0.0)

    headings = approach_headings(SETTINGS, 0.0, 0.0, 1.0, 0.0, size, yaw)

    assert len(headings) == 2
    assert abs(math.remainder(headings[0] - start, math.tau)) <= abs(
        math.remainder(headings[1] - start, math.tau)
    )
    for heading in headings:
        x, y, z = closes_along(plan_pick(SETTINGS, 1.0, 0.0, 0.8, heading).grasp)
        assert close(abs(x * math.cos(narrow) + y * math.sin(narrow)), 1.0) and close(z, 0.0)


@pytest.mark.parametrize(
    "size", [(0.068, 0.068, 0.22), (0.078, 0.06, 0.1), None], ids=["round", "fits", "no-size"]
)
def test_an_object_that_fits_the_fingers_either_way_keeps_the_settings_headings(size):
    assert approach_headings(SETTINGS, 0.0, 0.0, 1.0, 0.0, size, 0.7) == approach_headings(
        SETTINGS, 0.0, 0.0, 1.0, 0.0
    )


@pytest.mark.parametrize("yaw", [None, math.nan, math.inf], ids=["none", "nan", "inf"])
def test_a_long_box_with_no_finite_yaw_keeps_the_settings_headings(yaw):
    assert approach_headings(SETTINGS, 0.0, 0.0, 1.0, 0.0, LONG_BOX, yaw) == approach_headings(
        SETTINGS, 0.0, 0.0, 1.0, 0.0
    )


# Relative moves


def test_a_relative_move_is_in_the_start_pose_s_axes():
    assert relative_move(0.0, 0.0, 0.0, 1.0, 2.0, 0.0) == (1.0, 2.0, 0.0)
    forward, left, turn = relative_move(1.0, 1.0, math.pi / 2, 1.0, 3.0, 0.0)

    assert close(forward, 2.0) and close(left, 0.0) and close(turn, -90.0)


def test_a_relative_turn_takes_the_short_way_round():
    _, _, turn = relative_move(0.0, 0.0, math.radians(170.0), 0.0, 0.0, math.radians(-170.0))

    assert close(turn, 20.0)
