"""The toolbox core: typed atomic tools, their ctx and handles (ADR-0037).

Toolbox finds the atomics in walkie_agent.tools.atomic, builds one ctx for
them and turns each into a LangChain tool. WalkieAgent tells it when a stop
or a reset interrupts a task and when a new task replaces the todo list.
Bundles, their linker and runner, access levels and tool search come later
(self-evolving-agent.md).
"""

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.tools import BaseTool
from rai.communication.ros2 import ROS2Connector

from walkie_agent.toolbox.atomic import Atomic, AtomicTool
from walkie_agent.toolbox.context import ToolContext
from walkie_agent.toolbox.handles import (
    Handle,
    HandleSet,
    HandleStore,
    KindPolicy,
    Pose2D,
    register_kind,
)
from walkie_agent.toolbox.registry import (
    DEFAULT_PACKAGE,
    AtomicLoadError,
    build_atomic_tools,
    discover,
)
from walkie_agent.toolbox.types import (
    ERROR_CODES,
    Effect,
    ErrorCode,
    StepError,
    ToolMeta,
    ToolOutput,
)

__all__ = [
    "ERROR_CODES",
    "Atomic",
    "AtomicLoadError",
    "AtomicTool",
    "Effect",
    "ErrorCode",
    "Handle",
    "HandleSet",
    "HandleStore",
    "KindPolicy",
    "Pose2D",
    "StepError",
    "ToolContext",
    "ToolMeta",
    "ToolOutput",
    "Toolbox",
    "build_atomic_tools",
    "discover",
    "register_kind",
]

logger = logging.getLogger(__name__)

Listener = Callable[[], None]


class Toolbox:
    """The atomic tools and their shared ctx.

    Discovering the atomics raises AtomicLoadError for a broken one, so the
    agent does not start. The tools are built once, here.

    Listeners let later parts react to the task's lifecycle, such as the gaze
    loop stopping on an interrupt. A listener that raises is logged and does
    not stop the others.
    """

    def __init__(
        self,
        connector: ROS2Connector,
        backends: Mapping[str, Any] | None = None,
        package: str = DEFAULT_PACKAGE,
    ) -> None:
        self.ctx = ToolContext(connector, backends)
        self.atomics = discover(package)
        self._tools: list[BaseTool] = list(build_atomic_tools(self.ctx, self.atomics))
        self._lock = threading.Lock()
        self._interrupt_listeners: list[Listener] = []
        self._new_task_listeners: list[Listener] = []
        # Interrupts whose cancel is set but whose listeners have not all
        # run yet. Guarded by _lock.
        self._pending_interrupts = 0

    def tools(self) -> list[BaseTool]:
        """The atomic tools whose backends are present."""
        return list(self._tools)

    def on_interrupt(self, callback: Listener) -> None:
        """Call `callback()` on every interrupt, after `ctx.cancel` is set."""
        with self._lock:
            self._interrupt_listeners.append(callback)

    def on_new_task(self, callback: Listener) -> None:
        """Call `callback()` whenever a new task replaces the todo list."""
        with self._lock:
            self._new_task_listeners.append(callback)

    def interrupt(self) -> None:
        """A stop or a reset: set `ctx.cancel`, then call the interrupt listeners.

        Safe from any thread. Equal to cancel() followed by notify_interrupt().
        """
        self.cancel()
        self.notify_interrupt()

    def cancel(self) -> None:
        """Set `ctx.cancel`, and count an interrupt as pending until notify_interrupt().

        It never waits on a listener, so a caller may hold a lock. Each
        cancel() is followed by exactly one notify_interrupt().
        """
        with self._lock:
            self._pending_interrupts += 1
        self.ctx.cancel.set()

    def notify_interrupt(self) -> None:
        """Call the interrupt listeners. They may block, so call it without locks held."""
        try:
            self._notify(self._interrupt_listeners, "interrupt")
        finally:
            with self._lock:
                self._pending_interrupts = max(0, self._pending_interrupts - 1)

    def interrupt_pending(self) -> bool:
        """True while an interrupt's listeners are still running or not yet called.

        A run must not start then: a listener that runs late, such as the
        gaze loop stopping, would act on the new run instead of the old one.
        """
        with self._lock:
            return self._pending_interrupts > 0

    def begin_task(self) -> None:
        """A run starts: clear `ctx.cancel`, so it does not inherit an old cancel.

        The caller checks interrupt_pending() first and starts no run while
        it is True.
        """
        self.ctx.cancel.clear()

    def new_task(self) -> None:
        """A new task replaced the todo list: forget the handles, call the listeners."""
        self.ctx.handles.clear()
        self._notify(self._new_task_listeners, "new task")

    def _notify(self, listeners: list[Listener], event: str) -> None:
        with self._lock:
            snapshot = list(listeners)
        for callback in snapshot:
            try:
                callback()
            except Exception:
                logger.exception(f"A toolbox {event} listener failed.")
