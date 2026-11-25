"""The checks every implementation of a verb calls, fake or ROS-backed.

Guards live in the verbs rather than in the atomics, so a bundle or a script
that calls a verb cannot skip them. This module holds them once, so that the
fake refuses exactly what the robot refuses.
"""

import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager

from walkie_agent.capabilities.records import Pose2D
from walkie_agent.errors import ErrorCode, StepError

# The longest single relative move, from main's tools/navigation.py. It bounds
# the damage of a misread unit, such as centimeters taken as meters.
MAX_MOVE_M = 3.0

# The limits of a remembered person, from experiment/hri-agent's remember_person. The
# name is short enough to say back to a guest and long enough for any popular name.
# All four lie inside walkie-database's own limits: 100, 32, 64 and 256.
MAX_NAME_CHARS = 40
MAX_FACTS = 8
MAX_FACT_KEY_CHARS = 30
MAX_FACT_VALUE_CHARS = 80
# The longest question ctx.describer takes about a picture (decision PK-76): a
# sentence or two, enough for "what does the label say?" and far from a prompt.
MAX_QUESTION_CHARS = 300


class CancelToken:
    """Set by a stop or a reset. Once set, it ends every wait and refuses new motion.

    It stays set until the runtime calls reset(), which is what latches motion
    after a stop. It is safe to use from any thread.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._commit = threading.RLock()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        with self._commit:
            self._event.set()

    def reset(self) -> None:
        self._event.clear()

    def check(self, what: str) -> None:
        """Raise StepError CANCELLED if the token is set. `what` names the step."""
        if self._event.is_set():
            raise StepError(ErrorCode.CANCELLED, f"Cancelled {what}: a stop is in effect.")

    @contextmanager
    def unless_cancelled(self, what: str) -> Iterator[None]:
        """Run the block that commits a verb's result, unless the token is set.

        This orders a commit against cancel(): a cancel that comes first raises
        StepError CANCELLED and the block never runs, and a later one waits for
        the block to end. So no state changes after cancel() has returned. Keep
        the block short, since a stop waits for it.
        """
        with self._commit:
            self.check(what)
            yield

    def wait_for(
        self,
        done: Callable[[], bool],
        timeout_s: float,
        what: str,
        *,
        poll_s: float = 0.02,
    ) -> None:
        """Wait until done() is true, for at most `timeout_s`.

        Raises StepError CANCELLED once the token is set, which ends the wait
        within `poll_s`, and TIMEOUT once the time is up. A caller that holds a
        goal cancels it on either error.
        """
        require_timeout(timeout_s)
        deadline = time.monotonic() + timeout_s
        while True:
            self.check(what)
            if done():
                # A cancel that landed during done() wins over the completion.
                self.check(what)
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StepError(
                    ErrorCode.TIMEOUT, f"Timed out: {what} did not finish within {timeout_s:g} s."
                )
            self._event.wait(min(poll_s, remaining))


def is_finite_number(value: object) -> bool:
    """Whether `value` is an int or a float with a finite value. A bool is not a number here."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # An int too large for a float.
        return False


def require_finite(**values: float) -> None:
    """Raise StepError REFUSED for anything but a finite number.

    That covers NaN and infinity, which json.loads and pydantic both accept, and
    a bool, which Python would otherwise count as the number 1 and move a meter.
    """
    bad = [name for name, value in values.items() if not is_finite_number(value)]
    if bad:
        raise StepError(
            ErrorCode.REFUSED, f"Refused: {', '.join(bad)} must be a finite number."
        )


def require_text(**values: str) -> None:
    """Raise StepError REFUSED for an argument that is not a string."""
    bad = [name for name, value in values.items() if not isinstance(value, str)]
    if bad:
        raise StepError(ErrorCode.REFUSED, f"Refused: {', '.join(bad)} must be text.")


def require_timeout(timeout_s: float) -> None:
    """Raise StepError REFUSED unless `timeout_s` bounds a wait."""
    require_finite(timeout_s=timeout_s)
    if timeout_s <= 0:
        raise StepError(ErrorCode.REFUSED, f"Refused: timeout_s must be positive, not {timeout_s:g}.")


def begin_lookup(cancel: CancelToken, query: str, timeout_s: float, what: str) -> None:
    """The guards that open a vision lookup: the bound, the query, then the latch.

    Raises StepError REFUSED for a `timeout_s` that does not bound a wait or a `query`
    that is not text, and CANCELLED while `cancel` is set. `what` names the step in
    the CANCELLED message.
    """
    require_timeout(timeout_s)
    require_text(query=query)
    cancel.check(what)


def begin_crops(
    cancel: CancelToken, object_ids: Sequence[str], timeout_s: float, what: str
) -> list[str]:
    """The guards that open ctx.vision.crops: the bound, the ids, then the latch.

    Returns the ids as a list. Raises StepError REFUSED for a `timeout_s` that does
    not bound a wait or `object_ids` that is not a list of text, a lone string
    included, and CANCELLED while `cancel` is set. `what` names the step in the
    CANCELLED message.
    """
    require_timeout(timeout_s)
    if (
        isinstance(object_ids, str)
        or not isinstance(object_ids, Sequence)
        or not all(isinstance(object_id, str) for object_id in object_ids)
    ):
        raise StepError(ErrorCode.REFUSED, "Refused: object_ids must be a list of ids.")
    cancel.check(what)
    return list(object_ids)


def require_question(question: str) -> str:
    """The question for ctx.describer with its whitespace collapsed; "" for none.

    Raises StepError REFUSED for a question that is not text or is longer than
    MAX_QUESTION_CHARS.
    """
    require_text(question=question)
    text = " ".join(question.split())
    if len(text) > MAX_QUESTION_CHARS:
        raise StepError(
            ErrorCode.REFUSED,
            f"Refused: a question must be at most {MAX_QUESTION_CHARS} characters long.",
        )
    return text


def require_person_details(
    name: str, facts: Mapping[str, str] | None
) -> tuple[str, dict[str, str]]:
    """The name and facts of a person to remember, stripped, or StepError REFUSED.

    The name must hold 1 to MAX_NAME_CHARS characters once stripped. `facts` is None
    or a mapping of at most MAX_FACTS text keys to text values, each 1 to
    MAX_FACT_KEY_CHARS or MAX_FACT_VALUE_CHARS characters once stripped, with no two
    keys equal once stripped.
    """
    require_text(name=name)
    stripped = name.strip()
    if not 1 <= len(stripped) <= MAX_NAME_CHARS:
        raise StepError(
            ErrorCode.REFUSED,
            f"Refused: name must hold 1 to {MAX_NAME_CHARS} characters, not {len(stripped)}.",
        )
    if facts is None:
        return stripped, {}
    if not isinstance(facts, Mapping):
        raise StepError(ErrorCode.REFUSED, "Refused: facts must map text keys to text values.")
    if len(facts) > MAX_FACTS:
        raise StepError(
            ErrorCode.REFUSED, f"Refused: at most {MAX_FACTS} facts, not {len(facts)}."
        )
    kept: dict[str, str] = {}
    for key, value in facts.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise StepError(
                ErrorCode.REFUSED, "Refused: facts must map text keys to text values."
            )
        key, value = key.strip(), value.strip()
        if not 1 <= len(key) <= MAX_FACT_KEY_CHARS:
            raise StepError(
                ErrorCode.REFUSED,
                f"Refused: a fact's key must hold 1 to {MAX_FACT_KEY_CHARS} characters.",
            )
        if not 1 <= len(value) <= MAX_FACT_VALUE_CHARS:
            raise StepError(
                ErrorCode.REFUSED,
                f"Refused: the fact {key!r} must hold 1 to {MAX_FACT_VALUE_CHARS} characters.",
            )
        if key in kept:
            raise StepError(ErrorCode.REFUSED, f"Refused: the fact {key!r} is given twice.")
        kept[key] = value
    return stripped, kept


def check_relative_move(forward_m: float, left_m: float, turn_deg: float) -> None:
    """Refuse a relative move that is not finite, does nothing, or is over the cap."""
    require_finite(forward_m=forward_m, left_m=left_m, turn_deg=turn_deg)
    # A zero move fails too: the robot did not move, so the call must not count
    # as evidence that it did.
    if not (forward_m or left_m or turn_deg):
        raise StepError(ErrorCode.REFUSED, "Nothing to do: the move is zero.")
    distance = math.hypot(forward_m, left_m)
    if distance > MAX_MOVE_M:
        raise StepError(
            ErrorCode.REFUSED,
            f"Refused: a move of {distance:.2f} m is more than the {MAX_MOVE_M:g} m allowed "
            "in one move.",
        )


def relative_goal(start: Pose2D, forward_m: float, left_m: float, turn_deg: float) -> Pose2D:
    """The pose a relative move ends at, with yaw wrapped to [-pi, pi]."""
    cos, sin = math.cos(start.yaw), math.sin(start.yaw)
    yaw = start.yaw + math.radians(turn_deg)
    return Pose2D(
        start.x + forward_m * cos - left_m * sin,
        start.y + forward_m * sin + left_m * cos,
        math.atan2(math.sin(yaw), math.cos(yaw)),
    )
