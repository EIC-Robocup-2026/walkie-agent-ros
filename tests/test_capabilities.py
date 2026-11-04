"""The shared guards, the cancel token, UnavailableArms and the Ctx container."""

import math
import threading
import time

import pytest

from walkie_agent.capabilities import Ctx, ObjectSeen, Pose2D, UnavailableArms
from walkie_agent.capabilities.guards import (
    MAX_MOVE_M,
    CancelToken,
    check_relative_move,
    relative_goal,
    require_finite,
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
    require_finite(a=0.0, b=-3.5)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_require_timeout_refuses_a_wait_with_no_bound(bad):
    assert code_of(require_timeout, bad) is ErrorCode.REFUSED


@pytest.mark.parametrize(
    "move",
    [
        (float("nan"), 0.0, 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 0.0, float("-inf")),
        (0.0, 0.0, 0.0),
        (MAX_MOVE_M + 0.01, 0.0, 0.0),
        (2.5, 2.5, 0.0),
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
        lambda arms: arms.pick("apple_1"),
        lambda arms: arms.place_on("table"),
        lambda arms: arms.hand_over("operator"),
    ],
)
def test_unavailable_arms_refuse_every_verb(call):
    assert code_of(call, UnavailableArms()) is ErrorCode.UNAVAILABLE


def test_ctx_defaults_to_unavailable_arms():
    ctx = Ctx(cancel=CancelToken(), base=None, speech=None, vision=None)
    assert isinstance(ctx.arms, UnavailableArms)


def test_records_are_frozen():
    with pytest.raises(AttributeError):
        ObjectSeen("a_1", "apple", "fruit", "shelf").place = "bar"
