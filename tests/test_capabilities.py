"""The shared guards, the cancel token, the Unavailable stand-ins and the Ctx container."""

import math
import threading
import time

import pytest

from walkie_agent.arm.state import ArmsState
from walkie_agent.capabilities import (
    NO_ARMS,
    Ctx,
    HandPose,
    ObjectSeen,
    Pose2D,
    UnavailableArms,
    UnavailableFacts,
    UnavailableVision,
)
from walkie_agent.capabilities.guards import (
    MAX_MOVE_M,
    MAX_QUESTION_CHARS,
    CancelToken,
    begin_crops,
    begin_lookup,
    check_relative_move,
    is_finite_number,
    relative_goal,
    require_finite,
    require_question,
    require_text,
    require_timeout,
)
from walkie_agent.errors import ErrorCode, StepError


def code_of(call, *args, **kwargs) -> ErrorCode:
    with pytest.raises(StepError) as caught:
        call(*args, **kwargs)
    return caught.value.code


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_require_finite_refuses_and_names_the_value(bad):
    with pytest.raises(StepError) as caught:
        require_finite(ok=1.0, left_m=bad)
    assert caught.value.code is ErrorCode.REFUSED
    assert "left_m" in caught.value.message
    assert "ok" not in caught.value.message


def test_require_finite_passes_finite_values():
    require_finite(a=0.0, b=-3.5, c=2)


@pytest.mark.parametrize("bad", [True, False, "1", None, [], 10**400])
def test_require_finite_refuses_what_is_not_a_number(bad):
    assert not is_finite_number(bad)
    assert code_of(require_finite, value=bad) is ErrorCode.REFUSED


@pytest.mark.parametrize("bad", [None, 1, [], True])
def test_require_text_refuses_what_is_not_a_string_and_names_it(bad):
    with pytest.raises(StepError) as caught:
        require_text(ok="fine", place=bad)
    assert caught.value.code is ErrorCode.REFUSED
    assert "place" in caught.value.message
    assert "ok" not in caught.value.message


def test_require_text_leaves_a_blank_string_to_the_verb():
    require_text(query="")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True])
def test_require_timeout_refuses_a_wait_with_no_bound(bad):
    assert code_of(require_timeout, bad) is ErrorCode.REFUSED


def test_begin_lookup_checks_the_bound_then_the_query_then_the_latch():
    token = CancelToken()
    begin_lookup(token, "", 1.0, "looking for people")
    token.cancel()
    with pytest.raises(StepError) as bound:
        begin_lookup(token, 3, 0, "looking for people")
    assert (bound.value.code, bound.value.message) == (
        ErrorCode.REFUSED,
        "Refused: timeout_s must be positive, not 0.",
    )
    with pytest.raises(StepError) as query:
        begin_lookup(token, 3, 1.0, "looking for people")
    assert (query.value.code, query.value.message) == (
        ErrorCode.REFUSED,
        "Refused: query must be text.",
    )
    with pytest.raises(StepError) as latch:
        begin_lookup(token, "", 1.0, "looking for people")
    assert (latch.value.code, latch.value.message) == (
        ErrorCode.CANCELLED,
        "Cancelled looking for people: a stop is in effect.",
    )


def test_begin_crops_checks_the_bound_then_the_ids_then_the_latch():
    token = CancelToken()
    assert begin_crops(token, ("cup_2", "bottle_3"), 1.0, "looking at objects") == [
        "cup_2",
        "bottle_3",
    ]
    token.cancel()
    with pytest.raises(StepError) as bound:
        begin_crops(token, "cup_2", 0, "looking at objects")
    assert bound.value.message == "Refused: timeout_s must be positive, not 0."
    for bad in ("cup_2", ["cup_2", 3], None, {"cup_2": 1}):
        with pytest.raises(StepError) as ids:
            begin_crops(token, bad, 1.0, "looking at objects")
        assert (ids.value.code, ids.value.message) == (
            ErrorCode.REFUSED,
            "Refused: object_ids must be a list of ids.",
        )
    with pytest.raises(StepError) as latch:
        begin_crops(token, [], 1.0, "looking at objects")
    assert latch.value.message == "Cancelled looking at objects: a stop is in effect."


def test_require_question_collapses_whitespace_and_refuses_what_is_long_or_not_text():
    assert require_question("") == ""
    assert require_question("  What does\n the label say? ") == "What does the label say?"
    assert require_question("x" * MAX_QUESTION_CHARS) == "x" * MAX_QUESTION_CHARS
    with pytest.raises(StepError) as long:
        require_question("x" * (MAX_QUESTION_CHARS + 1))
    assert long.value.message == "Refused: a question must be at most 300 characters long."
    assert code_of(require_question, 3) is ErrorCode.REFUSED


def test_unavailable_vision_checks_the_crops_arguments_then_raises():
    token = CancelToken()
    vision = UnavailableVision(token)
    with pytest.raises(StepError) as raised:
        vision.crops(["cup_2"])
    assert (raised.value.code, raised.value.message) == (
        ErrorCode.UNAVAILABLE,
        "Live object perception is not connected.",
    )
    assert code_of(vision.crops, "cup_2") is ErrorCode.REFUSED
    token.cancel()
    assert code_of(vision.crops, ["cup_2"]) is ErrorCode.CANCELLED


@pytest.mark.parametrize(
    "move",
    [
        (float("nan"), 0.0, 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 0.0, float("-inf")),
        (0.0, 0.0, 0.0),
        (MAX_MOVE_M + 0.01, 0.0, 0.0),
        (2.5, 2.5, 0.0),
        (True, 0.0, 0.0),
        (0.0, True, 0.0),
        (0.0, 0.0, True),
        ("far", 0.0, 0.0),
        (10**400, 0.0, 0.0),
    ],
)
def test_check_relative_move_refuses(move):
    assert code_of(check_relative_move, *move) is ErrorCode.REFUSED


@pytest.mark.parametrize("move", [(MAX_MOVE_M, 0.0, 0.0), (-1.0, 1.0, 0.0), (0.0, 0.0, 90.0)])
def test_check_relative_move_accepts(move):
    check_relative_move(*move)


def test_the_cap_is_main_s_three_meters():
    assert MAX_MOVE_M == 3.0


def test_relative_goal_moves_in_the_robot_s_own_frame():
    goal = relative_goal(Pose2D(1.0, 2.0, math.pi / 2), 1.0, 0.5, 90.0)
    assert goal.x == pytest.approx(0.5)
    assert goal.y == pytest.approx(3.0)
    assert goal.yaw == pytest.approx(math.pi)


def test_relative_goal_wraps_the_yaw():
    goal = relative_goal(Pose2D(0.0, 0.0, 3.0), 0.0, 0.0, 180.0)
    assert -math.pi <= goal.yaw <= math.pi
    assert goal.yaw == pytest.approx(3.0 - math.pi)


def test_token_check_passes_until_cancelled_and_after_a_reset():
    token = CancelToken()
    token.check("the drive")
    assert not token.cancelled
    token.cancel()
    assert token.cancelled
    assert code_of(token.check, "the drive") is ErrorCode.CANCELLED
    token.reset()
    token.check("the drive")


def test_wait_for_returns_once_done():
    flags = iter([False, False, True])
    CancelToken().wait_for(lambda: next(flags), 2.0, "the drive", poll_s=0.001)


def test_wait_for_times_out_and_names_the_step():
    with pytest.raises(StepError) as caught:
        CancelToken().wait_for(lambda: False, 0.03, "the drive", poll_s=0.001)
    assert caught.value.code is ErrorCode.TIMEOUT
    assert "the drive" in caught.value.message


def test_wait_for_refuses_a_timeout_that_bounds_nothing():
    assert code_of(CancelToken().wait_for, lambda: True, 0.0, "the drive") is ErrorCode.REFUSED


def test_wait_for_refuses_when_already_cancelled_even_if_done():
    token = CancelToken()
    token.cancel()
    assert code_of(token.wait_for, lambda: True, 1.0, "the drive") is ErrorCode.CANCELLED


def test_a_cancel_during_the_completion_check_wins_over_the_completion():
    token = CancelToken()

    def done() -> bool:
        token.cancel()
        return True

    assert code_of(token.wait_for, done, 1.0, "the drive") is ErrorCode.CANCELLED


def test_unless_cancelled_skips_the_block_once_cancelled():
    token = CancelToken()
    token.cancel()
    ran = []
    with pytest.raises(StepError) as caught:
        with token.unless_cancelled("the drive"):
            ran.append(True)
    assert caught.value.code is ErrorCode.CANCELLED
    assert not ran


def test_a_cancel_waits_for_a_commit_in_progress():
    token = CancelToken()
    in_block = threading.Event()
    release = threading.Event()

    def commit() -> None:
        with token.unless_cancelled("the drive"):
            in_block.set()
            release.wait(5.0)

    committer = threading.Thread(target=commit)
    committer.start()
    assert in_block.wait(5.0)
    canceller = threading.Thread(target=token.cancel)
    canceller.start()
    canceller.join(0.05)
    assert canceller.is_alive()
    assert not token.cancelled
    release.set()
    committer.join(5.0)
    canceller.join(5.0)
    assert token.cancelled


def test_a_cancel_inside_the_commit_block_does_not_deadlock():
    token = CancelToken()
    with token.unless_cancelled("the drive"):
        token.cancel()
    assert token.cancelled


def test_a_cancel_from_another_thread_ends_the_wait_quickly():
    token = CancelToken()
    timer = threading.Timer(0.02, token.cancel)
    timer.start()
    started = time.monotonic()
    assert code_of(token.wait_for, lambda: False, 30.0, "the drive") is ErrorCode.CANCELLED
    timer.join()
    assert time.monotonic() - started < 5.0


@pytest.mark.parametrize(
    "call",
    [
        lambda arms: arms.poses(),
        lambda arms: arms.read(),
        lambda arms: arms.move_to("home"),
        lambda arms: arms.gripper("left", "open"),
        lambda arms: arms.reach("right", HandPose(0.6, -0.2, 0.8, 0.0, 1.57, 0.0, "map")),
        lambda arms: arms.pick("apple_1"),
        lambda arms: arms.place_on("table"),
        lambda arms: arms.hand_over("operator"),
    ],
)
def test_unavailable_arms_refuse_every_verb_but_state(call):
    with pytest.raises(StepError) as caught:
        call(UnavailableArms())
    assert str(caught.value) == f"UNAVAILABLE: {NO_ARMS}"


def test_unavailable_arms_have_no_allowance_to_restore():
    # No commander, so nothing can have turned the fingers' allowance on.
    assert UnavailableArms().restore_touch() is None


@pytest.mark.parametrize("read", [None, True, False])
def test_unavailable_arms_say_the_arms_are_unknown(read):
    report = UnavailableArms().state(read=read)
    assert report.arms == ArmsState()
    assert report.travel_safe is None
    assert report.sentence() == "Arms: state unknown."


def test_ctx_defaults_to_unavailable_arms():
    ctx = Ctx(cancel=CancelToken(), base=None, speech=None, vision=None)
    assert isinstance(ctx.arms, UnavailableArms)


@pytest.mark.parametrize("call", [lambda facts: facts.now(), lambda facts: facts.team()])
def test_unavailable_facts_refuse_every_verb(call):
    with pytest.raises(StepError) as caught:
        call(UnavailableFacts())
    assert caught.value.code is ErrorCode.UNAVAILABLE
    assert "no fact source" in str(caught.value)


def test_ctx_defaults_to_unavailable_facts():
    ctx = Ctx(cancel=CancelToken(), base=None, speech=None, vision=None)
    assert isinstance(ctx.facts, UnavailableFacts)


def test_records_are_frozen():
    with pytest.raises(AttributeError):
        ObjectSeen("a_1", "apple", "fruit", "shelf").place = "bar"
