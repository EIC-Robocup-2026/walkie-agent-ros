"""Atomics and AtomicTool, the LangChain tool that runs one of them.

An atomic is any object, normally a module, that defines four names:

- `META`, a ToolMeta: the name, description and effects the agent sees;
- `Args`, a pydantic model of the arguments;
- `Out`, a ToolOutput subclass, whose `summary` is all the LLM sees;
- `run(ctx, args) -> Out`, which raises StepError on failure.
"""

import logging
from typing import Any, Protocol

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel

from walkie_agent.toolbox.types import StepError, ToolMeta, ToolOutput

logger = logging.getLogger(__name__)


class Atomic(Protocol):
    """The shape of an atomic, for type hints. Modules match it by their names."""

    META: ToolMeta
    Args: type[BaseModel]
    Out: type[ToolOutput]

    def run(self, ctx: Any, args: Any) -> ToolOutput: ...


class AtomicTool(BaseTool):
    """Runs one atomic with a ctx, as a tool the LLM can call.

    The LLM sees the atomic's summary on success. A StepError becomes a
    ToolException whose message is "CODE: message", so with handle_tool_error
    on, the LLM reads it in a ToolMessage of status "error", and
    WalkieToolRunner's ledger counts the call as failed. Any other exception
    is a bug: it is logged with its traceback and reaches the LLM as INTERNAL.

    Arguments are validated against `Args` by BaseTool before _run, so a
    pydantic ValidationError from the LLM's arguments reaches RAI's
    ToolRunner unchanged and gets its usual message. The tool returns plain
    text and never an artifact: RAI pickles every artifact to disk, and the
    full Out stays with Python callers of call().
    """

    handle_tool_error: bool = True
    atomic: Any
    ctx: Any

    def __init__(self, atomic: Atomic, ctx: Any, **kwargs: Any) -> None:
        super().__init__(
            name=atomic.META.name,
            description=atomic.META.description,
            args_schema=atomic.Args,
            atomic=atomic,
            ctx=ctx,
            **kwargs,
        )

    @property
    def meta(self) -> ToolMeta:
        return self.atomic.META

    def call(self, **kwargs: Any) -> ToolOutput:
        """Run the atomic from Python and return its Out.

        Raises pydantic's ValidationError for arguments that do not fit
        `Args`, and StepError for every failure of the atomic itself,
        INTERNAL included.
        """
        args = self.atomic.Args.model_validate(kwargs)
        try:
            out = self.atomic.run(self.ctx, args)
        except StepError:
            raise
        except Exception as e:
            logger.exception(f"Atomic tool {self.name} failed with an unexpected error.")
            raise StepError("INTERNAL", f"{type(e).__name__}: {e}") from e
        if not isinstance(out, self.atomic.Out):
            logger.error(
                f"Atomic tool {self.name} returned {type(out).__name__}, "
                f"not its Out, {self.atomic.Out.__name__}."
            )
            raise StepError(
                "INTERNAL",
                f"{self.name} returned {type(out).__name__} instead of "
                f"{self.atomic.Out.__name__}.",
            )
        return out

    def _run(self, **kwargs: Any) -> str:
        try:
            return self.call(**kwargs).summary
        except StepError as e:
            raise ToolException(str(e)) from e
