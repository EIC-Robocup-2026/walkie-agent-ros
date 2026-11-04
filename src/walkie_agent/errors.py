"""The one failure type of capability verbs and atomics.

A failed step raises StepError with a code from a small fixed set, so a caller
can act on the code instead of parsing a sentence. Its text is what the LLM
reads: the adapter's error formatter passes str(error) to the ToolMessage.
"""

from enum import StrEnum


class ErrorCode(StrEnum):
    """Why a step failed."""

    # A backend, server or topic is missing or not answering.
    UNAVAILABLE = "UNAVAILABLE"
    # The step started but did not finish in time.
    TIMEOUT = "TIMEOUT"
    # Nothing matches: an unknown place, object or person.
    NOT_FOUND = "NOT_FOUND"
    # A rule forbids the request as asked, such as a move over the cap.
    REFUSED = "REFUSED"
    # The robot is not in the state the step needs, such as an empty hand.
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    # The robot-side motion or skill ran and failed, such as an aborted goal.
    SKILL_FAILED = "SKILL_FAILED"
    # A stop or a reset interrupted the step, or latched it before it began.
    CANCELLED = "CANCELLED"
    # A bug in a tool.
    INTERNAL = "INTERNAL"


class StepError(Exception):
    """A failed step: a code, and one plain sentence for the LLM."""

    def __init__(self, code: ErrorCode | str, message: str) -> None:
        self.code = ErrorCode(code)
        self.message = message
        super().__init__(self.code, message)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
