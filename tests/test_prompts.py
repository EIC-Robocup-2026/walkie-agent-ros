"""The system prompts of walkie_agent.prompts: which rules a tool set and the run's flags
put in a prompt, and the robot description the prompt ends with.

What main assembles, the prompt a run's model receives included, is tested in
tests/test_main.py. build_agent's default prompt is tested in tests/test_runtime.py.
"""

import re
from pathlib import Path

import pytest

from walkie_agent.config import load_config
from walkie_agent.locations import Location
from walkie_agent.prompts import _SCRIPTS_RULE, build_system_prompt
from walkie_agent.tools.sets import (
    ARM_TOOLS,
    ARMS,
    DEFAULT,
    GPSR,
    HRI,
    OBJECT_MEMORY_TOOLS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLACES = {
    "kitchen": Location("kitchen", 3.5, 1.2, 1.57, "Kitchen counter."),
    "door": Location("door", 1.0, 0.0, 0.0),
}
ROBOT = "Walkie is a test robot with a unique marker 7f3a."
MIDDLEWARE_TOOLS = {"update_todos", "wait_for_person"}
TOOL_SET_LISTS = {"default": DEFAULT, "gpsr": GPSR, "hri": HRI, "arms": ARMS}


def tool_names(tool_set, arms=False, object_memory=False):
    """What a run binds: the set's atomics, less the arm and object memory atomics without their flags."""
    atomics = {fn.__name__ for fn in TOOL_SET_LISTS[tool_set]}
    dropped = (set() if arms else ARM_TOOLS) | (set() if object_memory else OBJECT_MEMORY_TOOLS)
    return (atomics - dropped) | MIDDLEWARE_TOOLS


@pytest.mark.parametrize("tool_set", ["default", "gpsr", "hri"])
def test_only_a_set_with_the_memory_tools_gets_a_memory_rule(tool_set):
    prompts = {
        memory: build_system_prompt(tool_set, PLACES, ROBOT, vision=True, memory=memory)
        for memory in (False, True)
    }
    for prompt in prompts.values():
        assert ("`remember_person`" in prompt) == (tool_set == "hri")
        assert set(re.findall(r"`(\w+)`", prompt)) <= tool_names(tool_set)
    # DEFAULT and GPSR, and their prompts, stay as they were.
    assert (prompts[False] == prompts[True]) == (tool_set != "hri")


@pytest.mark.parametrize("tool_set", ["default", "gpsr", "hri", "arms"])
def test_only_scripts_add_the_run_script_rule_to_a_prompt(tool_set):
    arms = tool_set == "arms"
    without = build_system_prompt(tool_set, PLACES, ROBOT, arms=arms)
    with_scripts = build_system_prompt(tool_set, PLACES, ROBOT, arms=arms, scripts=True)

    assert "run_script" not in without
    assert build_system_prompt(tool_set, PLACES, ROBOT, arms=arms, scripts=False) == without
    assert with_scripts.count(_SCRIPTS_RULE) == 1
    assert with_scripts.replace(_SCRIPTS_RULE, "") == without
    # The rule sits in the rules, before the general ones that close them.
    assert with_scripts.index("`run_script`") < with_scripts.index("If a tool reports a failure")


def test_the_shipped_robot_description_claims_only_what_is_connected():
    text = (PROJECT_ROOT / "robot.md").read_text(encoding="utf-8")
    assert "Chulalongkorn" in text and "polite, patient and brief" in text
    assert "Two robot arms" in text
    assert "not connected" in text
    # Seeing people and moving the arms depend on the run, so it defers to the prompt.
    flat = " ".join(text.split())
    assert "when the agent runs with people perception" in flat
    assert "when the agent runs with the arm tools" in flat
    assert "has no grasp skill yet" in flat
    # It describes the robot. How to use a tool is the prompt's job, set by set. Some tool
    # names are plain English ("pick up"), so only the unmistakable ones are looked for.
    assert "`" not in text
    every = {name for tool_set in TOOL_SET_LISTS for name in tool_names(tool_set, arms=True)}
    for name in every:
        assert "_" not in name or name not in text, name
    # It claims the facts get_facts reads, and names the team as config.toml does.
    assert "the time and the date from the computer's clock" in text
    team = load_config(PROJECT_ROOT / "config.toml")["team"]
    for key in ("name", "country", "affiliation"):
        assert team[key] in text, key
