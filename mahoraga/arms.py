"""The arms of the comparison, which differ only in what the model is given.

All of them share the prompt, the atomics and whatever the eval's config sets.

    A       per-step tools, as on main
    B       run_script, no shelf
    C       run_script with a frozen, verified, callable shelf
    C2      run_script, with the same verified scripts as prompt examples only
    D       per-step tools plus bundles
    induct  run_script with a writable shelf and save=[names], for training only
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic_monty import Monty

from mahoraga.bundles import BUNDLES
from mahoraga.driver import Budgets, ScriptRunner, allow_list, describe_functions, script_summary
from mahoraga.shelf import Shelf
from mahoraga.tasks import script_record
from walkie_agent.adapter import as_tool, as_tools
from walkie_agent.atomics import GPSR
from walkie_agent.capabilities import Ctx
from walkie_agent.capabilities.fake import CHULA_PAT14_6_PLACES

ARMS = ("A", "B", "C", "C2", "D")
INDUCT = "induct"
SCRIPT_ARMS = frozenset({"B", "C", "C2", INDUCT})

_PLACES = "\n".join(f"- {name} ({place['room']})" for name, place in CHULA_PAT14_6_PLACES.items())
PROMPT = f"""\
You are Walkie, a service robot in a home. Your operator gives you one command,
and you carry it out with your tools.

Where things are:
{_PLACES}

You and the operator start at the instruction point. The operator stays there,
and their id is "operator". "Me" in a command means the operator, who is never
one of the people a command talks about, so leave them out of any count.

Only what your tools do counts. A text reply reaches nobody, so say every answer
aloud with say, and look before you answer. You see only from where you stand: at
a location, the objects on it; at a room, the objects on the locations in that
room; and everybody in the room you are in.

To greet, talk to or tell something to a person, go to their room, look for them,
then say it there. Read the time, the date and your team with get_facts, and never
guess a fact. To follow or guide a person, find them, then go to the
destination yourself. Objects go down on a location, or to a person by hand.

The command is complete, and nobody can answer a question. When it is done, stop
calling tools and reply with one short sentence."""

_SCRIPT_NOTES = f"""\
A script is Python, and the functions below are its only way to act. Nothing
carries over from one script to the next. Call the functions in plain loops, not
inside a builtin key= lambda. They return plain lists and dicts, with the keys
their docstrings give.

A failed call raises RuntimeError("CODE: message"), or TypeError for a bad
argument, and your script may catch either. A script may make at most
{Budgets().max_calls} calls, run for {Budgets().feed_s} seconds between calls, and
print {Budgets().print_bytes} bytes. The script's last expression and what it
printed come back to you.

Functions:"""
_SHELF_CALLABLE = """\
These functions are also on your shelf. Scripts may call them directly:

{}"""
_SHELF_EXAMPLES = """\
These scripts worked on earlier commands. They are examples only, and calling
them will fail, so copy what you need into your script:

{}"""
_SAVE_NOTES = """\
You are also building a library of functions for later commands, which will name
other places, objects and people. Once you know what this command needs, write its
steps as a function that takes those names as parameters, call it in the same
script, and name it in save. Write no place, object or person into the function.
When a function on your shelf fits, call it instead of writing a new one. A function
is kept only if the script succeeds and the function stands alone."""


@dataclass
class Episode:
    """What one episode hands the model, and the record of its scripts for the harness."""

    tools: list[StructuredTool]
    prompt: str
    scripts: list[dict[str, Any]] = field(default_factory=list)


def require_arm(arm: str) -> None:
    """Raise ValueError for a name that is not an arm."""
    if arm not in (*ARMS, INDUCT):
        raise ValueError(f"Unknown arm: {arm}. Use one of {', '.join((*ARMS, INDUCT))}.")


class ArmSetup:
    """One arm of one eval. Its shelf view is fixed here, so every episode sees the same one."""

    def __init__(self, arm: str, *, pool: Monty | None = None, shelf_root: str | None = None) -> None:
        require_arm(arm)
        if arm in SCRIPT_ARMS and pool is None:
            raise ValueError(f"Arm {arm} runs scripts, so it needs a Monty pool.")
        self.arm = arm
        self.shelf: Shelf | None = None
        if arm in ("C", "C2"):
            self.shelf = Shelf(shelf_root, frozen=True)
        elif arm == INDUCT:
            self.shelf = Shelf(shelf_root)
        self._runner = None
        if arm in SCRIPT_ARMS:
            # C2 sees the same scripts as C, but its runner has no shelf to call them from.
            callable_shelf = None if arm == "C2" else self.shelf
            self._runner = ScriptRunner(pool, shelf=callable_shelf)

    def episode(self, ctx: Ctx) -> Episode:
        """The tools and prompt of one episode, with `ctx` bound."""
        scripts: list[dict[str, Any]] = []
        if self.arm == "A":
            return Episode(as_tools(GPSR, ctx), PROMPT, scripts)
        if self.arm == "D":
            return Episode(as_tools([*GPSR, *BUNDLES], ctx), PROMPT, scripts)
        return Episode([self._script_tool(ctx, scripts)], self._script_prompt(), scripts)

    def _script_prompt(self) -> str:
        """The shared prompt, plus what this arm's shelf adds. An empty shelf adds nothing."""
        parts = [PROMPT]
        if self.arm == INDUCT:
            parts.append(_SAVE_NOTES)
        if self.arm in ("C", INDUCT) and self.shelf.describe():
            parts.append(_SHELF_CALLABLE.format(self.shelf.describe()))
        if self.arm == "C2" and self.shelf.examples():
            parts.append(_SHELF_EXAMPLES.format(self.shelf.examples()))
        return "\n\n".join(parts)

    def _script_tool(self, ctx: Ctx, scripts: list[dict[str, Any]]) -> StructuredTool:
        factory = _saving_run_script if self.arm == INDUCT else _plain_run_script
        tool = as_tool(factory(self._runner, scripts), ctx)
        tool.description = (
            f"{tool.description}\n\n{_SCRIPT_NOTES}\n\n{describe_functions(allow_list(GPSR, ctx))}"
        )
        return tool


def _plain_run_script(runner: ScriptRunner, scripts: list[dict[str, Any]]) -> Callable[..., Any]:
    def run_script(ctx: Ctx, code: str) -> Mapping[str, Any]:
        """Run one Python script that calls the robot's functions.

        Returns whether it worked, its output, what it printed and any error.

        Args:
            code: The script's source.
        """
        result = runner.run(code, allow_list(GPSR, ctx))
        scripts.append(script_record(code, result))
        return script_summary(result)

    return run_script


def _saving_run_script(runner: ScriptRunner, scripts: list[dict[str, Any]]) -> Callable[..., Any]:
    def run_script(ctx: Ctx, code: str, save: list[str] | None = None) -> Mapping[str, Any]:
        """Run one Python script that calls the robot's functions, and keep what it names.

        Returns whether it worked, its output, what it printed, any error, and
        which functions it saved.

        Args:
            code: The script's source.
            save: Names of top-level functions in the script to keep for later commands.
        """
        result = runner.run(code, allow_list(GPSR, ctx), save=save or ())
        scripts.append(script_record(code, result))
        return script_summary(result)

    return run_script
