"""The one adapter from an atomic to what calls it: a LangChain tool or a plain callable.

Both shapes hide ctx, so neither the LLM nor a Monty script can see or pass it.
All tool errors reach the model through ToolErrorMiddleware and
format_tool_error. Argument validation errors do not: ToolNode answers those
before the middleware runs.
"""

import functools
import inspect
import json
import logging
from collections.abc import Callable, Iterable
from typing import Any, Concatenate, ParamSpec, TypeVar

from langchain.agents.middleware import ToolCallRequest, ToolErrorMiddleware
from langchain_core.tools import StructuredTool

from walkie_agent.capabilities import Ctx
from walkie_agent.errors import ErrorCode, StepError

P = ParamSpec("P")
R = TypeVar("R")

_logger = logging.getLogger(__name__)


def bind(fn: Callable[Concatenate[Ctx, P], R], ctx: Ctx) -> Callable[P, R]:
    """Return `fn` as a plain callable with ctx bound, for the Monty driver.

    It keeps the name, the docstring and the signature without ctx, and returns
    what `fn` returns. It keeps `__wrapped__`, so typing.get_type_hints still
    resolves string annotations in the module of `fn`.
    """
    return _without_ctx(fn, ctx, lambda result: result)


def as_tool(fn: Callable[..., Any], ctx: Ctx) -> StructuredTool:
    """Build the tool for atomic `fn`, whose schema comes from its signature and docstring.

    Raises ValueError for a parameter the docstring does not describe, since
    the LLM would have to guess what it means. A malformed docstring raises too.

    The schema validates strictly. In its default mode pydantic turns a JSON true
    into 1.0 before any guard sees it, which would drive the base a meter. So a
    model that sends a bool or a numeric string gets a validation error back.
    """
    tool = StructuredTool.from_function(
        _without_ctx(fn, ctx, _as_text),
        parse_docstring=True,
        error_on_invalid_docstring=True,
    )
    undescribed = [name for name, field in tool.args.items() if not field.get("description")]
    if undescribed:
        raise ValueError(f"{fn.__name__}: no description for {', '.join(undescribed)}.")
    tool.args_schema.model_config["strict"] = True
    tool.args_schema.model_rebuild(force=True)
    return tool


def as_tools(fns: Iterable[Callable[..., Any]], ctx: Ctx) -> list[StructuredTool]:
    """Build the tools of one tool set. Raises ValueError for a name used twice."""
    tools = [as_tool(fn, ctx) for fn in fns]
    names = [tool.name for tool in tools]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"Tool names used twice: {', '.join(repeated)}.")
    return tools


def format_tool_error(exc: Exception, request: ToolCallRequest) -> str:
    """ToolErrorMiddleware's on_error: "CODE: message" for the model.

    Anything but a StepError is a bug. The model gets only its tool and type, so
    internal detail stays out of the conversation, and the log gets the traceback.
    """
    if isinstance(exc, StepError):
        return str(exc)
    name = request.tool_call["name"]
    _logger.error("Tool %s failed.", name, exc_info=exc)
    return f"{ErrorCode.INTERNAL}: {name} failed with {type(exc).__name__}."


def tool_error_middleware() -> ToolErrorMiddleware:
    """The single error path for tool calls, on both the sync and the async route."""
    return ToolErrorMiddleware(on_error=format_tool_error)


def _without_ctx(
    fn: Callable[..., Any], ctx: Ctx, finish: Callable[[Any], Any]
) -> Callable[..., Any]:
    signature = inspect.signature(fn)
    parameters = list(signature.parameters.values())
    if not parameters or parameters[0].name != "ctx":
        raise ValueError(f"{fn.__name__} must take ctx as its first parameter.")

    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        return finish(fn(ctx, *args, **kwargs))

    # inspect.signature stops at __signature__, and langchain_core reads
    # __annotations__ to match the docstring's Args section against parameters.
    call.__signature__ = signature.replace(parameters=parameters[1:])
    call.__annotations__ = {
        name: value for name, value in fn.__annotations__.items() if name != "ctx"
    }
    return call


def _as_text(result: Any) -> str:
    """JSON text for a non-string result.

    Left alone, langchain_core would send an empty list as an empty message and
    take a list of strings for content blocks.
    """
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
