"""The shared types of typed atomic tools (ADR-0037).

An atomic answers with a typed `Out` whose `summary` is all the LLM sees on
success, and fails with a `StepError` whose code comes from a fixed set, so
a caller can act on the code instead of parsing a sentence.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ErrorCode = Literal[
    "UNAVAILABLE",
    "TIMEOUT",
    "NOT_FOUND",
    "AMBIGUOUS",
    "OUT_OF_REACH",
    "STALE",
    "REFUSED",
    "PRECONDITION_FAILED",
    "SKILL_FAILED",
    "POSTCONDITION",
    "SAFETY_STOP",
    "CANCELLED",
    "INTERNAL",
]

# One line per code. StepError accepts only these codes, so a new code needs
# a meaning here before any tool can raise it, and a test checks that the
# table in docs/architecture/agent.md lists the same codes in this order.
ERROR_CODES: dict[ErrorCode, str] = {
    "UNAVAILABLE": "a backend, server or topic is missing or not answering",
    "TIMEOUT": "the operation started but did not finish in time",
    "NOT_FOUND": "nothing matches: no one in view, an unknown handle or pose name",
    "AMBIGUOUS": "several things match and the tool will not guess",
    "OUT_OF_REACH": "the target is beyond what the arm or base may reach",
    "STALE": "a handle is too old, or the robot has moved since it was captured",
    "REFUSED": "a rule forbids the request as asked, such as a value out of range",
    "PRECONDITION_FAILED": (
        "the robot is not in the state the call needs, such as an arm out before "
        "driving (ADR-0025)"
    ),
    "SKILL_FAILED": "the robot-side motion or skill ran and failed, such as no MoveIt plan",
    "POSTCONDITION": (
        "the backend reported success but the check after it failed, such as a joint "
        "off its target or an empty gripper"
    ),
    "SAFETY_STOP": "an emergency stop or safety controller ended the motion",
    "CANCELLED": "a reset, a stop or a newer command interrupted it",
    "INTERNAL": "a bug in the tool",
}


class StepError(Exception):
    """A tool failure with a code from ERROR_CODES.

    `message` is a plain sentence for the LLM, and `data` holds details for
    code, such as the candidates of an AMBIGUOUS match. str(err) is
    "CODE: message", which is what the LLM reads.
    """

    def __init__(
        self, code: ErrorCode, message: str, data: Mapping[str, Any] | None = None
    ) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"Unknown StepError code {code!r}; use one of ERROR_CODES.")
        self.code: ErrorCode = code
        self.message = message
        self.data: dict[str, Any] = dict(data or {})
        # All three in args, so a pickled error comes back with its data.
        super().__init__(code, message, self.data)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class Effect(StrEnum):
    """What a tool does to the world. A bundle's effects are the union of its steps."""

    R = "R"  # reads stored or robot state
    P = "P"  # perceives or computes
    S = "S"  # speaks or plays a sound
    N = "N"  # moves the base
    M = "M"  # moves an arm or gripper
    # Writes to the robot's memory. Provisional: Perception question 3
    # (docs/architecture/perception.md) says no effect covers a memory write
    # yet. Until it is decided, a W tool is a reviewed atomic that generated
    # tools may never call.
    W = "W"


_IDEMPOTENT_EFFECTS = frozenset({Effect.R, Effect.P})


class ToolOutput(BaseModel):
    """The base of every atomic's `Out`. Subclasses add typed fields.

    `summary` is all the LLM sees on success. It may span several lines,
    such as one line per handle.
    """

    summary: str

    @field_validator("summary")
    @classmethod
    def _summary_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must not be empty: it is all the LLM sees.")
        return value


class ToolMeta(BaseModel):
    """What an atomic declares about itself in its module's `META`.

    Unknown fields are rejected, so a misspelt field fails at load instead of
    being ignored. `examples` are checked against the atomic's `Args` by the
    registry, which also checks that `name` equals the module's name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,47}$")
    # The LLM reads it as the tool's description.
    description: str
    effects: frozenset[Effect]
    # Spoken-style requests the tool answers, for the future tool search.
    questions: tuple[str, ...] = Field(min_length=3, max_length=5)
    examples: tuple[dict[str, Any], ...] = Field(min_length=1)
    status: Literal["verified", "draft"] = "verified"
    created_by: str = "human"
    # Sandbox workers come later; every atomic runs in the agent process.
    runtime: Literal["inprocess"] = "inprocess"
    # Backend names that must be present for the tool to load.
    requires: tuple[str, ...] = ()

    @field_validator("description")
    @classmethod
    def _description_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be empty: the LLM reads it.")
        return value

    @field_validator("questions")
    @classmethod
    def _questions_not_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not question.strip() for question in value):
            raise ValueError("every question must be a non-empty request.")
        return value

    @property
    def idempotent(self) -> bool:
        """True when the tool only reads or perceives, so a retry is safe."""
        return self.effects <= _IDEMPOTENT_EFFECTS
