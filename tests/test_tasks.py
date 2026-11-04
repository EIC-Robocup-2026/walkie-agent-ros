"""Task templates and /walkie/task messages (walkie_agent.tasks, ADR-0041). No ROS."""

import json
import re
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT
from walkie_agent.atomics import DEFAULT, GPSR, HRI
from walkie_agent.runtime.todos import parse_seed
from walkie_agent.tasks import (
    TaskTemplate,
    check_required_tools,
    load_task_templates,
    parse_task_message,
)

TEMPLATES_FILE = PROJECT_ROOT / "config" / "task_templates.toml"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "task_templates.toml"
    path.write_text(text)
    return path


def template(name: str = "tour") -> TaskTemplate:
    return TaskTemplate(name=name, task="Show the person the house.", todos=["Go to the kitchen"])


# -- Loading ---------------------------------------------------------------------


def test_load_task_templates_reads_every_field_in_file_order(tmp_path):
    path = write(
        tmp_path,
        '''
[templates.receptionist]
task = """
Receive the guests one at a time.
"""
todos = [
  "Meet the guest",
  { content = "Take the bag", active_form = "Taking the bag", requires_tool = "say" },
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
    assert receptionist.task == "Receive the guests one at a time."
    assert receptionist.todos == [
        "Meet the guest",
        {"content": "Take the bag", "active_form": "Taking the bag", "requires_tool": "say"},
    ]


def test_a_missing_templates_file_means_no_templates(tmp_path):
    assert load_task_templates(tmp_path / "task_templates.toml") == {}


def test_the_shipped_file_has_no_templates_and_a_valid_example(tmp_path):
    assert load_task_templates(TEMPLATES_FILE) == {}

    # The commented example still loads once its "# " prefixes are removed.
    lines = TEMPLATES_FILE.read_text().splitlines()
    start = next(n for n, line in enumerate(lines) if line.startswith("# [templates."))
    example = "\n".join(line.removeprefix("#").removeprefix(" ") for line in lines[start:])
    templates = load_task_templates(write(tmp_path, example))
    assert list(templates) == ["tour"]
    # It names only tools that every selectable tool set really binds.
    for tool_set in (DEFAULT, GPSR, HRI):
        check_required_tools(templates, {fn.__name__ for fn in tool_set})


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('[tour]\ntask = "x"\ntodos = ["a"]\n', "tour: Extra inputs"),
        ("templates = 3\n", "templates: Input should be a valid dictionary"),
        ("[templates]\ntour = 3\n", "templates.tour: Input should be a valid dictionary"),
        ('[templates.tour]\ntask = "x"\nitems = ["a"]\n', "templates.tour.items: Extra inputs"),
        (
            '[templates.tour]\nkeywords = ["tour"]\ntask = "x"\ntodos = ["a"]\n',
            "templates.tour.keywords: Extra inputs",
        ),
        # The name is the table's key, so a name inside the table is an unknown key.
        (
            '[templates.tour]\nname = "other"\ntask = "x"\ntodos = ["a"]\n',
            "templates.tour.name: Extra inputs",
        ),
        ('[templates.tour]\ntodos = ["a"]\n', "templates.tour.task: Field required"),
        ('[templates.tour]\ntask = " "\ntodos = ["a"]\n', "templates.tour.task: String should"),
        ('[templates.tour]\ntask = "x"\ntodos = []\n', "templates.tour.todos: List should"),
        ('[templates.tour]\ntask = "x"\ntodos = "a"\n', "templates.tour.todos: Input should"),
        ('[templates.tour]\ntask = "x"\ntodos = [{ content = "" }]\n', "Seed item t1 has no content"),
        (
            '[templates.tour]\ntask = "x"\ntodos = [{ content = "a", require_tool = "say" }]\n',
            "Seed item t1 has unknown keys: require_tool",
        ),
    ],
)
def test_a_bad_template_fails_at_load(tmp_path, text, error):
    path = write(tmp_path, text)
    with pytest.raises(ValueError, match=re.escape(error)) as caught:
        load_task_templates(path)
    message = str(caught.value)
    assert "\n" not in message
    assert str(path) in message
    assert "[templates.<name>]" in message


def test_a_template_task_is_stripped():
    assert TaskTemplate(name="tour", task="  Show.\n", todos=["a"]).task == "Show."


def test_template_validation_keeps_raw_seed_for_runner():
    raw = ["Meet the guest", {"content": "Greet", "requires_tool": "say"}]
    tour = TaskTemplate(name="tour", task="Show.", todos=raw)
    assert tour.todos == raw
    # The Runner parses the seed again. Parsed items would be refused here.
    assert [item["content"] for item in parse_seed(tour.todos)] == ["Meet the guest", "Greet"]
    with pytest.raises(ValueError, match="unknown keys"):
        parse_seed(parse_seed(raw))


def test_required_tools_must_be_available():
    tour = TaskTemplate(
        name="tour",
        task="Show.",
        todos=["Meet", {"content": "Go", "requires_tool": "go_to"}, {"content": "Say"}],
    )
    check_required_tools({"tour": tour}, {"go_to"})
    with pytest.raises(ValueError, match=r"tour\.t2 requires go_to.*Available: say\."):
        check_required_tools({"tour": tour}, {"say"})


def test_a_template_with_no_requirement_needs_no_tool():
    check_required_tools({"tour": template()}, ())


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
        ("Go to the kitchen", "Invalid JSON"),
        ('["Go to the kitchen"]', "Input should be an object"),
        ('{"todos": ["Go"]}', "task: Field required"),
        ('{"task": " ", "todos": ["Go"]}', "task: String should have at least 1 character"),
        ('{"task": 3, "todos": ["Go"]}', "task: Input should be a valid string"),
        ('{"task": "Go."}', "exactly one of todos and template"),
        ('{"task": "Go.", "todos": ["Go"], "template": "tour"}', "exactly one of todos and template"),
        # Both keys are present, so a null on either side does not make it one seed.
        ('{"task": "Go.", "todos": null, "template": "tour"}', "todos: Input should be a valid array"),
        ('{"task": "Go.", "todos": [], "template": null}', "template: Input should be a valid string"),
        ('{"task": "Go.", "todos": null}', "todos: Input should be a valid array"),
        ('{"task": "Go.", "template": null}', "template: Input should be a valid string"),
        ('{"task": "Go.", "template": "garage"}', "no task template 'garage'. Known templates: tour"),
        ('{"task": "Go.", "template": " tour "}', "no task template ' tour '. Known templates: tour"),
        ('{"task": "Go.", "todos": {"content": "Go"}}', "todos: Input should be a valid array"),
        ('{"task": "Go.", "todo": ["Go"]}', "todo: Extra inputs"),
    ],
)
def test_a_malformed_task_message_raises(data, error):
    with pytest.raises(ValueError, match=re.escape(error)) as caught:
        parse_task_message(data, {"tour": template()})
    assert "\n" not in str(caught.value)


def test_an_empty_seed_is_a_seed():
    assert parse_task_message('{"task": "Go.", "todos": []}', {}) == ("Go.", [])
