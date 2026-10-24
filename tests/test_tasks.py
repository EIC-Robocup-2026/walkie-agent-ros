"""Task templates and /walkie/task messages (walkie_agent.tasks, ADR-0041). No ROS."""

import json
import re
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT
from walkie_agent.tasks import (
    TaskTemplate,
    load_task_templates,
    match_template,
    parse_task_message,
)
from walkie_agent.toolbox.registry import discover
from walkie_agent.tools import (
    FindObjectsTool,
    ListLocationsTool,
    MoveRelativeTool,
    NavigateToLocationTool,
    SpeakTool,
    UpdateTodosTool,
    WaitForPersonTool,
)

TEMPLATES_FILE = PROJECT_ROOT / "config" / "task_templates.toml"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "task_templates.toml"
    path.write_text(text)
    return path


def template(name: str = "tour", keywords: tuple[str, ...] = ("house tour",)) -> TaskTemplate:
    return TaskTemplate(
        name=name, keywords=keywords, task="Show the person the house.", todos=["Go to the kitchen"]
    )


# -- Loading ---------------------------------------------------------------------


def test_load_task_templates_reads_every_field_in_file_order(tmp_path):
    path = write(
        tmp_path,
        '''
[templates.receptionist]
keywords = ["guests", "  front   door "]
task = """
Receive the guests one at a time.
"""
todos = [
  "Meet the guest",
  { content = "Take the bag", active_form = "Taking the bag", requires_tool = "speak" },
]

[templates.tour]
task = "Show the person the house."
todos = ["Go to the kitchen"]
''',
    )
    templates = load_task_templates(path)

    assert list(templates) == ["receptionist", "tour"]
    receptionist = templates["receptionist"]
    assert receptionist.name == "receptionist"
    assert receptionist.keywords == ("guests", "front door")
    assert receptionist.task == "Receive the guests one at a time."
    assert receptionist.todos == [
        "Meet the guest",
        {"content": "Take the bag", "active_form": "Taking the bag", "requires_tool": "speak"},
    ]
    # Without keywords, only the operator can pick the template.
    assert templates["tour"].keywords == ()


def test_a_missing_templates_file_means_no_templates(tmp_path):
    assert load_task_templates(tmp_path / "task_templates.toml") == {}


def shipped_example(tmp_path: Path) -> dict[str, TaskTemplate]:
    """The shipped file's commented example, loaded once its "# " prefixes are removed."""
    lines = TEMPLATES_FILE.read_text().splitlines()
    start = next(n for n, line in enumerate(lines) if line.startswith("# [templates."))
    example = "\n".join(line.removeprefix("#").removeprefix(" ") for line in lines[start:])
    return load_task_templates(write(tmp_path, example))


def test_the_shipped_file_has_the_hri_template_and_a_valid_example(tmp_path):
    assert list(load_task_templates(TEMPLATES_FILE)) == ["hri"]

    assert list(shipped_example(tmp_path)) == ["tour"]


def test_every_requires_tool_in_the_shipped_file_names_a_real_tool(tmp_path):
    # Nothing at load checks the name, and a step whose tool does not exist
    # can never be completed.
    tools = {
        cls.model_fields["name"].default
        for cls in (
            FindObjectsTool,
            ListLocationsTool,
            MoveRelativeTool,
            NavigateToLocationTool,
            SpeakTool,
            UpdateTodosTool,
            WaitForPersonTool,
        )
    }
    tools |= {atomic.META.name for atomic in discover()}
    templates = [*load_task_templates(TEMPLATES_FILE).values(), *shipped_example(tmp_path).values()]

    required = {
        (template.name, step["requires_tool"])
        for template in templates
        for step in template.todos
        if isinstance(step, dict) and "requires_tool" in step
    }

    assert {name for name, _ in required} == {"hri", "tour"}
    assert [(name, tool) for name, tool in sorted(required) if tool not in tools] == []


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('[tour]\ntask = "x"\ntodos = ["a"]\n', "unknown top-level keys tour"),
        ("templates = 3\n", "templates must be a table"),
        ("[templates]\ntour = 3\n", "template 'tour' must be a table"),
        ('[templates.tour]\ntask = "x"\nitems = ["a"]\n', "template 'tour' has unknown keys: items"),
        ('[templates.tour]\nkeywords = "tour"\ntask = "x"\ntodos = ["a"]\n', "keywords must be"),
        ('[templates.tour]\nkeywords = [" "]\ntask = "x"\ntodos = ["a"]\n', "keywords must be"),
        ('[templates.tour]\ntodos = ["a"]\n', "template 'tour' has no task text"),
        ('[templates.tour]\ntask = "x"\ntodos = []\n', "template 'tour' needs todos"),
        ('[templates.tour]\ntask = "x"\ntodos = "a"\n', "template 'tour' needs todos"),
        ('[templates.tour]\ntask = "x"\ntodos = [{ content = "" }]\n', "Seed item t1 has no content"),
        (
            '[templates.tour]\ntask = "x"\ntodos = [{ content = "a", require_tool = "speak" }]\n',
            "Seed item t1 has unknown keys: require_tool",
        ),
    ],
)
def test_a_bad_template_fails_at_load(tmp_path, text, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        load_task_templates(write(tmp_path, text))


# -- Matching --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["Give me a house tour.", "HOUSE TOUR!", "house tour", "a House\n  Tour, please"]
)
def test_a_keyword_matches_as_whole_words_in_any_case(text):
    tour = template()
    assert match_template(text, {"tour": tour}) is tour


@pytest.mark.parametrize(
    "text", ["a lighthouse tour", "house tours", "the house", "a tour of the house", ""]
)
def test_no_match_without_the_whole_keyword(text):
    assert match_template(text, {"tour": template()}) is None


def test_the_first_matching_template_in_file_order_wins():
    mug = template("mug", keywords=("mug",))
    kitchen = template("kitchen", keywords=("kitchen",))
    text = "Bring the mug from the kitchen."

    assert match_template(text, {"mug": mug, "kitchen": kitchen}) is mug
    assert match_template(text, {"kitchen": kitchen, "mug": mug}) is kitchen


# -- /walkie/task messages -------------------------------------------------------


def test_a_task_message_with_todos():
    data = json.dumps({"task": " Bring me the mug. ", "todos": ["Go to the kitchen"]})
    assert parse_task_message(data, {}) == ("Bring me the mug.", ["Go to the kitchen"])


def test_a_task_message_with_a_template():
    data = json.dumps({"task": "Show me around.", "template": "tour"})
    assert parse_task_message(data, {"tour": template()}) == (
        "Show me around.\n\nShow the person the house.",
        ["Go to the kitchen"],
    )


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ("Go to the kitchen", "not valid JSON"),
        ('["Go to the kitchen"]', "must be a JSON object"),
        ('{"todos": ["Go"]}', "no task text"),
        ('{"task": " ", "todos": ["Go"]}', "no task text"),
        ('{"task": 3, "todos": ["Go"]}', "no task text"),
        ('{"task": "Go."}', "exactly one of todos and template"),
        ('{"task": "Go.", "todos": ["Go"], "template": "tour"}', "exactly one of todos and template"),
        ('{"task": "Go.", "template": "garage"}', "no task template 'garage'. Known templates: tour"),
        ('{"task": "Go.", "todos": {"content": "Go"}}', "todos must be a list"),
        ('{"task": "Go.", "todo": ["Go"]}', "unknown keys: todo"),
    ],
)
def test_a_malformed_task_message_raises(data, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        parse_task_message(data, {"tour": template()})
