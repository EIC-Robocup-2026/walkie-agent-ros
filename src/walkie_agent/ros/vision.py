"""ctx.vision on the robot until a live perception backend exists.

Both verbs raise UNAVAILABLE. An empty list would say that nothing is in view, and a
remembered database row is not a current view, so neither stands in for a backend.
"""

from walkie_agent.capabilities import ObjectSeen, PersonSeen
from walkie_agent.capabilities.guards import CancelToken, require_text, require_timeout
from walkie_agent.errors import ErrorCode, StepError


class UnavailableVision:
    """Checks a verb's arguments and the latch like a real backend, then raises UNAVAILABLE."""

    def __init__(self, cancel: CancelToken) -> None:
        self._cancel = cancel

    def objects(self, query: str = "", *, timeout_s: float = 5.0) -> list[ObjectSeen]:
        self._begin(query, timeout_s, "looking for objects")
        raise StepError(ErrorCode.UNAVAILABLE, "Live object perception is not connected.")

    def people(self, query: str = "", *, timeout_s: float = 5.0) -> list[PersonSeen]:
        self._begin(query, timeout_s, "looking for people")
        raise StepError(ErrorCode.UNAVAILABLE, "Live people perception is not connected.")

    def _begin(self, query: str, timeout_s: float, what: str) -> None:
        require_timeout(timeout_s)
        require_text(query=query)
        self._cancel.check(what)
