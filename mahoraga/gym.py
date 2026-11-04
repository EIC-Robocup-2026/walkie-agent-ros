"""The gym: one Inspect task per arm, over seeded fake worlds, behind the oracle gate.

The agent is a black-box solver: LangChain's create_agent, as the competition
agent uses, reaches the eval's model through Inspect's agent_bridge. So the model,
temperature, seed and retries come from the eval's config, and each arm shares them.
"""

import contextlib
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.agent import agent_bridge
from inspect_ai.log import EvalLog
from inspect_ai.model import Model, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from pydantic_monty import Monty

from mahoraga import home
from mahoraga.arms import SCRIPT_ARMS, ArmSetup, require_arm
from mahoraga.commands import commands as generated_commands
from mahoraga.gate import require_gate
from mahoraga.sandbox import open_pool
from mahoraga.tasks import dataset, episode_robot, ground_truth
from walkie_agent.adapter import tool_error_middleware

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash:nitro"
OPENROUTER = "https://openrouter.ai/api/v1"
_SERVICE = "openrouter"
MAX_ROUNDS = 30
# Guards against a runaway episode. MAX_ROUNDS is what normally ends one.
_TOKEN_LIMIT = 200_000
_TIME_LIMIT_S = 300


def gpsr_task(
    arm: str,
    records: Sequence[Mapping[str, Any]],
    *,
    epochs: int = 1,
    pool: Monty | None = None,
    shelf_root: str | None = None,
) -> Task:
    """One arm over `records`. Raises GateError unless the oracle gate covers them."""
    require_gate(records, epochs)
    setup = ArmSetup(arm, pool=pool, shelf_root=shelf_root)
    return Task(
        dataset=dataset(records),
        solver=_agent_solver(setup),
        scorer=ground_truth(setup.shelf),
        epochs=epochs,
        name=f"gpsr_{arm}",
        token_limit=_TOKEN_LIMIT,
        time_limit=_TIME_LIMIT_S,
        metadata={"arm": arm, "shelf_digest": setup.shelf.digest() if setup.shelf else None},
    )


def _agent_solver(setup: ArmSetup) -> Solver:
    # The bridge sends model "inspect" to the eval's model and ignores this key.
    model = ChatOpenAI(model="inspect", api_key="unused", max_retries=0)

    @solver(name="agent")
    def factory() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            with episode_robot(state) as robot:
                episode = setup.episode(robot.ctx)
                agent = create_agent(
                    model,
                    episode.tools,
                    system_prompt=episode.prompt,
                    middleware=[tool_error_middleware()],
                )
                try:
                    async with agent_bridge():
                        # Each round is one model step and one tool step.
                        result = await agent.ainvoke(
                            {"messages": [{"role": "user", "content": state.input_text}]},
                            {"recursion_limit": 2 * MAX_ROUNDS + 1},
                        )
                    state.output.completion = result["messages"][-1].text
                except GraphRecursionError:
                    state.store.set("out_of_rounds", True)
                finally:
                    state.store.set("scripts", episode.scripts)
            return state

        return solve

    return factory()


def resolve_model(model: str | Model, base_url: str | None = None) -> str | Model:
    """A model id becomes Inspect's generic OpenAI-compatible provider at `base_url`.

    Inspect's "openai" provider takes a name it does not know for a new GPT model
    and drops the temperature, so every arm would run at the provider's default.
    The key stays in OPENAI_API_KEY: only the variable's name is passed, and a
    missing key raises ValueError. An Inspect model, or an id of its mock provider,
    passes through.
    """
    if not isinstance(model, str) or model.startswith("mockllm/"):
        return model
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(f"Set OPENAI_API_KEY in the environment to run {model}.")
    return get_model(
        f"openai-api/{_SERVICE}/{model}",
        base_url=base_url,
        api_key_var="OPENAI_API_KEY",
        responses_api=False,
    )


def run_episodes(
    arms: Sequence[str],
    commands: int,
    *,
    epochs: int = 1,
    split: str = "trial",
    category: str = "mixed",
    model: str | Model = DEFAULT_MODEL,
    base_url: str | None = OPENROUTER,
    temperature: float = 0.0,
    seed: int = 0,
    max_samples: int = 8,
    log_dir: str | Path | None = None,
    shelf_root: str | None = None,
    sandbox: bool = True,
) -> list[EvalLog]:
    """Run `commands` commands for each of `arms`, `epochs` times each, and return the logs.

    Raises GateError before any arm starts unless the gate covers the data, and
    ValueError for an unknown arm or a count below 1. A sample that errors, such as
    on a provider failure, stays unscored in its log, and the run goes on.
    """
    if commands < 1:
        raise ValueError(f"The number of commands must be at least 1, not {commands}.")
    for arm in arms:
        require_arm(arm)
    records = generated_commands(split, commands, category)
    require_gate(records, epochs)
    resolved = resolve_model(model, base_url)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = Path(log_dir) if log_dir else home() / "runs" / stamp
    logs: list[EvalLog] = []
    needs_pool = any(arm in SCRIPT_ARMS for arm in arms)
    opened = open_pool(sandbox=sandbox, max_processes=max_samples) if needs_pool else contextlib.nullcontext()
    with opened as pool:
        for arm in arms:
            logs += inspect_eval(
                gpsr_task(arm, records, epochs=epochs, pool=pool, shelf_root=shelf_root),
                model=resolved,
                log_dir=str(target),
                log_format="eval",
                display="none",
                temperature=temperature,
                seed=seed,
                # A parallel call could run base moves concurrently on one robot.
                parallel_tool_calls=False,
                max_samples=max_samples,
                fail_on_error=False,
                retry_on_error=2,
                max_retries=3,
                timeout=120,
                score_display=False,
            )
    return logs
