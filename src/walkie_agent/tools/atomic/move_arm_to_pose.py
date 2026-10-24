"""Move one arm or both to a named pose, wait, and check the joints (ADR-0022, ADR-0024).

The pose's joint tables go to the commander's /set_joint_position in the
order its /get_joint_states gives for the group, so a commander that lists
its joints in another order, or without openarm_right_joint2, still gets the
right values. A reported success is not trusted: the joints are read after
every move and compared with the pose. In the commander's plan-only mode it
reports success with "Plan Only" and moves nothing, so the answer never says
the arms moved: it says they were already in the pose, or fails.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from walkie_agent.arm.commander import CommanderClient
from walkie_agent.arm.poses import ARM_GROUPS, ARMS, BOTH, BOTH_ARMS_GROUP, RIGHT_JOINT2, ArmPose
from walkie_agent.arm.state import MATCH_TOLERANCE_RAD, unread_joints, worst_joint
from walkie_agent.toolbox import Effect, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _arm as shared

# The commander's status for a success in plan-only mode, where nothing moved.
PLAN_ONLY = "Plan Only"
PLAN_ONLY_TEXT = "The commander runs in plan-only mode, so it moved nothing"
# What to do about it: the commander reads plan_only once, at startup.
PLAN_ONLY_NEXT = (
    "No arm can move until the commander restarts without plan_only, so tell the person "
    "instead of trying again."
)

META = ToolMeta(
    name="move_arm_to_pose",
    description=(
        "Move one arm or both to a named pose and wait until they arrive, usually 5 to 10 "
        "seconds. Use a pose name from list_arm_poses. arm is left, right or both (the "
        "default); list_arm_poses says which arms each pose can move. Tell the people near "
        "you before the arms move. The answer ends with whether the arms are safe for "
        "driving: before you drive, move both arms to a pose that is, such as 'home'."
    ),
    effects=frozenset({Effect.M}),
    questions=(
        "Put your arms down.",
        "Raise your hands.",
        "Hold out your left hand to take the bag.",
        "Tuck your arms in before you drive.",
        "Move your right arm to standby.",
    ),
    examples=({"pose": "home"}, {"pose": "receive", "arm": "left"}),
    requires=(shared.ARM,),
)


class Args(BaseModel):
    pose: str = Field(description="The pose's name, as list_arm_poses gives it, such as 'home'.")
    arm: Literal["left", "right", "both"] = Field(
        default="both", description="The arm to move: left, right or both."
    )


class Out(ToolOutput):
    pose: str
    arm: str
    # The commander's status: "Success", or "Plan Only" when the commander runs
    # with plan_only and planned without moving; the joint check then fails
    # unless the arms were already in the pose, and the answer says so.
    status: str
    # The largest joint error after the move, in radians.
    max_error_rad: float
    travel_safe: bool | None


def run(ctx: ToolContext, args: Args) -> Out:
    client = shared.commander(ctx)
    offered = shared.offered_poses(ctx, client)
    name = shared.pose_name(args.pose)
    pose = offered.get(name)
    if pose is None:
        # A hidden pose gets the same answer as an unknown one, so the LLM is
        # never told about a pose it may not use.
        known = ", ".join(offered) or "none"
        raise StepError(
            "NOT_FOUND",
            f"No pose named {args.pose!r}. The poses are: {known}. "
            f"{shared.arm_sentence(client, ctx.cancel)}",
            {"pose": args.pose, "poses": list(offered)},
        )
    if args.arm not in pose.arm_options:
        raise StepError(
            "REFUSED",
            f"{_arm_refusal(pose)} {shared.arm_sentence(client, ctx.cancel)}",
            {"pose": name, "arm": args.arm, "arms": list(pose.arm_options)},
        )

    arms = ARMS if args.arm == BOTH else (args.arm,)
    group = BOTH_ARMS_GROUP if args.arm == BOTH else ARM_GROUPS[args.arm]
    targets = {joint: value for arm in arms for joint, value in pose.tables[arm].items()}
    try:
        order = client.joint_names(group, ctx.cancel, shared.CONFIG.read_timeout_s)
    except StepError as e:
        if e.code == "CANCELLED" and ctx.cancel.is_set():
            raise StepError(
                "CANCELLED",
                f"Stopped by reset before the arms moved. {shared.state_sentence(client)}",
            ) from None
        # The read failed, so a second read for the sentence would too.
        sentence = shared.state_sentence(client)
        raise shared.with_sentence(e, f"Could not move to {name!r}: ", sentence) from None
    positions = _positions(ctx, client, pose, args.arm, targets, order)

    try:
        status = client.set_joint_position(group, positions, ctx.cancel, pose.timeout_s)
    except StepError as e:
        raise _move_error(ctx, client, name, e) from None

    try:
        reading = client.joint_states(BOTH_ARMS_GROUP, ctx.cancel, shared.CONFIG.read_timeout_s)
    except StepError as e:
        if e.code == "CANCELLED" and ctx.cancel.is_set():
            raise StepError("CANCELLED", shared.STOPPED) from None
        # A move that cannot be checked counts as failed (ADR-0024).
        raise _unchecked(
            status,
            f"the arms could not be checked against {name!r}: {shared.full_stop(e.message)}",
            shared.state_sentence(client),
            {"pose": name, "status": status, "read_error": e.code},
        ) from None

    sentence = shared.state_sentence(client)
    worst = worst_joint(targets, reading)
    if worst is None:
        missing = unread_joints(targets, reading)
        raise _unchecked(
            status,
            f"its reading lacks a usable position for {', '.join(missing)}, so the arms "
            f"cannot be checked against {name!r}.",
            sentence,
            {"pose": name, "status": status, "missing": missing},
        )
    joint, error = worst
    if error > MATCH_TOLERANCE_RAD:
        raise _unchecked(
            status,
            f"{joint} is {error:.2f} rad from {name!r}.",
            sentence,
            {"pose": name, "status": status, "joint": joint, "error_rad": error},
        )

    if status == PLAN_ONLY:
        # Nothing moved, so the check passed because the arms were already there.
        where = (
            f"Already at {name!r}."
            if args.arm == BOTH
            else f"The {args.arm} arm is already at {name!r}."
        )
        summary = f"{where} {PLAN_ONLY_TEXT}. {sentence}"
    else:
        moved = (
            f"Moved to {name!r}." if args.arm == BOTH else f"Moved the {args.arm} arm to {name!r}."
        )
        summary = f"{moved} {sentence}"
    return Out(
        summary=summary,
        pose=name,
        arm=args.arm,
        status=status,
        max_error_rad=error,
        travel_safe=client.last_state.travel_safe,
    )


def _unchecked(status: str, problem: str, sentence: str, data: dict[str, Any]) -> StepError:
    """The POSTCONDITION for a move whose check after it failed; `problem` says how."""
    if status == PLAN_ONLY:
        text = f"{PLAN_ONLY_TEXT}, and {problem} Treat this as a failure. {PLAN_ONLY_NEXT}"
    else:
        text = f"The commander reported success, but {problem} Treat this as a failure."
    return StepError("POSTCONDITION", f"{text} {sentence}", data)


def _arm_refusal(pose: ArmPose) -> str:
    """Why the pose does not take the asked `arm`, such as "'receive' is for the left arm only."."""
    if len(pose.arms) == 1:
        return f"{pose.name!r} is for the {pose.arms[0]} arm only."
    # Both tables exist, but not as one both_arms goal (both_arms = false).
    return f"{pose.name!r} is for one arm at a time: left or right."


def _positions(
    ctx: ToolContext,
    client: CommanderClient,
    pose: ArmPose,
    arm: str,
    targets: dict[str, float],
    order: tuple[str, ...],
) -> list[float]:
    """The pose's values in the commander's joint order for the group.

    walkie-ros launched with right_joint2_fixed lists no openarm_right_joint2
    and holds it at 0, so the joint is dropped when the pose has it at 0. A
    pose that needs it elsewhere cannot be reached that way. Any other
    difference means config/robot_model.toml no longer matches the running
    walkie-ros, which the snapshot test should have caught.
    """
    missing = [joint for joint in targets if joint not in order]
    extra = [joint for joint in order if joint not in targets]
    if missing == [RIGHT_JOINT2] and not extra:
        value = targets[RIGHT_JOINT2]
        if value != 0.0:
            raise StepError(
                "REFUSED",
                f"{pose.name!r} cannot move the right arm while walkie-ros runs with "
                f"right_joint2_fixed: the pose needs {RIGHT_JOINT2} at {value:.2f} rad, and "
                "that joint is held at 0. Use the left arm or another pose. "
                f"{shared.arm_sentence(client, ctx.cancel)}",
                {"pose": pose.name, "arm": arm, "joint": RIGHT_JOINT2},
            )
    elif missing or extra:
        raise StepError(
            "INTERNAL",
            f"The commander's joints for {arm} do not match config/robot_model.toml "
            f"(missing: {', '.join(missing) or 'none'}; unknown: {', '.join(extra) or 'none'}). "
            "Regenerate the snapshot for the running walkie-ros. "
            f"{shared.arm_sentence(client, ctx.cancel)}",
            {"pose": pose.name, "missing": missing, "unknown": extra},
        )
    return [targets[joint] for joint in order]


def _move_error(
    ctx: ToolContext, client: CommanderClient, name: str, error: StepError
) -> StepError:
    """The answer to a move the commander did not finish as asked."""
    if error.code == "CANCELLED" and ctx.cancel.is_set():
        # A stop or a reset: the call ends at once, so the arms are not read.
        stopped = (
            shared.interrupted(error, gripper=False) or "Stopped by reset before the arms moved."
        )
        return StepError("CANCELLED", f"{stopped} {shared.state_sentence(client)}", error.data)
    if error.code in ("SKILL_FAILED", "INTERNAL", "CANCELLED"):
        # The commander answered, so it can be read: a failed plan left the
        # arms where they were, and a failed execution may have stopped them
        # part way.
        sentence = shared.arm_sentence(client, ctx.cancel, read=True)
    else:
        # TIMEOUT, UNAVAILABLE or PRECONDITION_FAILED: the commander is not
        # answering or is busy, so a read would fail or wait out its timeout.
        sentence = shared.state_sentence(client)
    return shared.with_sentence(error, f"Could not move to {name!r}: ", sentence)
