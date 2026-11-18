"""build_agent: create_agent over a Ctx's tools, with the runtime's middleware stack."""

import logging
from collections.abc import Callable, Iterable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from walkie_agent.capabilities import Ctx
from walkie_agent.prompts import SYSTEM_PROMPT
from walkie_agent.runtime.middleware import (
    AnswerEmptyReply,
    Deadline,
    EndAfterQuestion,
    RepairHistory,
    SerialGuard,
    StopLatch,
    ToolLog,
)
from walkie_agent.runtime.todos import TODO_TOOL_COUNT, WalkieTodos
from walkie_agent.tools.adapter import as_tools, tool_error_middleware
from walkie_agent.tools.sets import DEFAULT

logger = logging.getLogger(__name__)

# More tools than this and a small model may pick the wrong one.
MAX_BOUND_TOOLS = 20


def build_middleware(
    ctx: Ctx,
    *,
    run_limit: int,
    deadline_s: float,
    on_todos: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    fallback: BaseChatModel | None = None,
    tool_log: ToolLog | None = None,
) -> list[AgentMiddleware]:
    """The stack, in the one order that works. tests/test_runtime.py pins it.

    create_agent runs before_agent and before_model hooks in list order, after_model
    hooks in reverse, and makes the first wrapper the outermost. So:

    - The limit counts in after_model and comes after everything that can jump from
      there (AnswerEmptyReply, the WalkieTodos gate, RepairHistory), so it sees every
      call first. A jump skips the after_model hooks earlier in the list, so
      AnswerEmptyReply goes before WalkieTodos, and the todo gate wins when both apply.
    - Around a tool call: the log sees every answer. The stop and order refusals come
      next, outside the ledger, so a call that never ran cannot poison completion.
      ToolErrorMiddleware is innermost, so the ledger sees errors as results.
    - Around a model call: the reminder goes on once, and each attempt of a fallback
      chain gets its own Deadline. ModelFallbackMiddleware's constructor needs a model,
      so with none configured it is left out, and a TimeoutError reaches the Runner.
    """
    return [
        *([tool_log] if tool_log else []),
        StopLatch(ctx.cancel),
        SerialGuard(),
        AnswerEmptyReply(ctx.cancel),
        WalkieTodos(ctx.cancel, ctx.speech.say, on_todos),
        EndAfterQuestion(),
        RepairHistory(),
        ModelCallLimitMiddleware(run_limit=run_limit, exit_behavior="end"),
        *([ModelFallbackMiddleware(fallback)] if fallback else []),
        Deadline(deadline_s),
        tool_error_middleware(),
    ]


def build_agent(
    model: BaseChatModel,
    ctx: Ctx,
    *,
    tools: Iterable[Callable[..., Any]] = DEFAULT,
    extra_tools: Iterable[BaseTool] = (),
    run_limit: int = 25,
    deadline_s: float = 60.0,
    on_todos: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    fallback: BaseChatModel | None = None,
    tool_log: ToolLog | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> CompiledStateGraph:
    """Build the agent. Run it with ainvoke and `Session.config()`, which is async only.

    `run_limit` and `deadline_s` default to provisional values, which `[agent]` in
    config.toml overrides through `agent_settings`. No fallback is configured by
    default, since that choice is the team's (decision 19). `extra_tools` are tools
    already built over `ctx`, such as run_script with --scripts, bound after the atomics.
    Raises ValueError for a tool name used twice.
    """
    bound = as_tools(tools, ctx)
    for tool in extra_tools:
        if tool.name in {other.name for other in bound}:
            raise ValueError(f"Tool names used twice: {tool.name}.")
        bound.append(tool)
    if len(bound) + TODO_TOOL_COUNT > MAX_BOUND_TOOLS:
        logger.warning(
            "%d tools are bound, more than %d; a small model may pick the wrong one.",
            len(bound) + TODO_TOOL_COUNT,
            MAX_BOUND_TOOLS,
        )
    middleware = build_middleware(
        ctx,
        run_limit=run_limit,
        deadline_s=deadline_s,
        on_todos=on_todos,
        fallback=fallback,
        tool_log=tool_log,
    )
    # With two tools of one name, the graph would silently run only the last one. The
    # attribute is optional on a middleware, so this reads it as create_agent does.
    taken = {tool.name for layer in middleware for tool in getattr(layer, "tools", [])}
    collisions = sorted(taken & {tool.name for tool in bound})
    if collisions:
        raise ValueError(f"Tool names reserved by the middleware: {', '.join(collisions)}.")
    return create_agent(
        model,
        bound,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=InMemorySaver() if checkpointer is None else checkpointer,
    )
