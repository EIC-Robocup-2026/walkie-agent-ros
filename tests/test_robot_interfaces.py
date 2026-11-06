"""The my_robot_interfaces fields the arm commander's client reads (no ROS graph).

The client comes with step S6 of the see, remember and act roadmap. It sends
SetJointPosition and ControlGripper goals and calls GetJointStates on
walkie-ros's bimanual commander, and reads every field below. A pin move in
ros2_ws.repos that renames or retypes one of them fails here, before the
client builds a goal the commander cannot read. Under rmw_zenoh_cpp a client
built from another version of these types finds the commander but is never
answered (ADR-0035), so the failure would otherwise show only as a timeout.
"""

import pytest
from my_robot_interfaces.action import ControlGripper, SetJointPosition
from my_robot_interfaces.srv import GetJointStates


@pytest.mark.parametrize(
    ("message", "fields"),
    [
        (
            SetJointPosition.Goal,
            {"group_name": "string", "joint_positions": "sequence<double>"},
        ),
        (SetJointPosition.Result, {"success": "boolean", "status": "string"}),
        (ControlGripper.Goal, {"group_name": "string", "position": "double"}),
        (
            ControlGripper.Result,
            {
                "success": "boolean",
                "status": "string",
                # True when the settled finger gap is in the commander's detect band.
                "grasped": "boolean",
                "gripper_gap": "double",
            },
        ),
        (GetJointStates.Request, {"group_name": "string"}),
        (
            GetJointStates.Response,
            {
                "joint_names": "sequence<string>",
                "joint_positions": "sequence<double>",
                "success": "boolean",
                "status": "string",
            },
        ),
    ],
)
def test_fields_the_commander_client_reads(message, fields):
    types = message.get_fields_and_field_types()
    assert {name: types.get(name) for name in fields} == fields
