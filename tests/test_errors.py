"""StepError: its codes, its text and how it travels."""

import pickle

import pytest

from walkie_agent.errors import ErrorCode, StepError


def test_str_is_the_code_and_the_message():
    assert str(StepError(ErrorCode.REFUSED, "Too far.")) == "REFUSED: Too far."


def test_code_is_normalised_from_a_plain_string():
    error = StepError("NOT_FOUND", "No such place.")
    assert error.code is ErrorCode.NOT_FOUND
    assert error.message == "No such place."


def test_unknown_code_is_refused():
    with pytest.raises(ValueError):
        StepError("OOPS", "Not a code.")


def test_error_survives_a_pickle():
    # An error that crosses a process boundary keeps its code and message. The
    # bytes are the ones this test just made, so unpickling them is safe.
    error = pickle.loads(pickle.dumps(StepError(ErrorCode.TIMEOUT, "Too slow.")))
    assert (error.code, error.message) == (ErrorCode.TIMEOUT, "Too slow.")
    assert str(error) == "TIMEOUT: Too slow."
