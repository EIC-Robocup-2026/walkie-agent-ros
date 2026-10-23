"""Task templates and operator task messages: two ways to seed a task (ADR-0041).

A fixed task, such as Receptionist, can start with its steps already on the
todo list. This module reads where those steps come from, without ROS:

- Templates, loaded from `config/task_templates.toml`. An utterance that
  contains a template's keyword picks the template, but only while no todo
  item is open. The operator can also pick a template by name.
- Messages on `/walkie/task`, a std_msgs/String holding JSON: either
  `{"task": "...", "todos": [...]}` or `{"task": "...", "template": "name"}`.

The agent (walkie_agent.agent) subscribes to the topic and submits the result.
"""

import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from walkie_agent.todo import TodoList

# Unknown keys are most likely typos, such as "keyword" or "items", that would
# silently drop a template or a seed, so both formats reject them.
_TEMPLATE_KEYS = frozenset({"keywords", "task", "todos"})
_TASK_KEYS = frozenset({"task", "todos", "template"})


@dataclass(frozen=True)
class TaskTemplate:
    """A fixed task whose steps are written in advance."""

    name: str
    keywords: tuple[str, ...]  # words or phrases that pick the template, in any case
    task: str  # instructions for the LLM, sent after the person's request
    todos: list[str | dict[str, Any]]  # the seed, in the shape TodoList.seed takes

    def prompt(self, text: str) -> str:
        """The message to run: the request, a blank line, then the template's task."""
        return f"{text}\n\n{self.task}"

    def matches(self, text: str) -> bool:
        """True when `text` contains one of the keywords as whole words."""
        return any(_keyword_pattern(keyword).search(text) for keyword in self.keywords)


def load_task_templates(path: str | Path) -> dict[str, TaskTemplate]:
    """Load `[templates.<name>]` tables from a TOML file, keyed by name, in file order.

    A missing file means no templates. A template that could not seed a task,
    such as one with a malformed todo item, raises ValueError here, so a bad
    file stops the agent at startup rather than in the middle of a task.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        data = tomllib.load(f)
    unknown = sorted(set(data) - {"templates"})
    if unknown:
        raise ValueError(
            f"{path}: unknown top-level keys {', '.join(unknown)}; "
            "put each template in a [templates.<name>] table."
        )
    entries = data.get("templates", {})
    if not isinstance(entries, dict):
        raise ValueError(f"{path}: templates must be a table of [templates.<name>] tables.")
    return {name: _load_template(name, entry, path) for name, entry in entries.items()}


def match_template(text: str, templates: Mapping[str, TaskTemplate]) -> TaskTemplate | None:
    """The first template, in file order, with a keyword in `text`, or None."""
    return next((template for template in templates.values() if template.matches(text)), None)


def parse_task_message(
    data: str, templates: Mapping[str, TaskTemplate]
) -> tuple[str, list[str | dict[str, Any]]]:
    """Read a `/walkie/task` message into the text to run and its seed.

    The message is a JSON object with the task's text and exactly one of
    `todos`, the seed itself, or `template`, the name of a template whose
    task and todos to use. Raises ValueError, with a one-line reason, for a
    malformed message. The seed's items are checked by WalkieAgent.submit.
    """
    try:
        message = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"The message is not valid JSON: {e}.") from e
    if not isinstance(message, dict):
        raise ValueError("The message must be a JSON object with a task.")
    unknown = sorted(set(message) - _TASK_KEYS)
    if unknown:
        raise ValueError(f"The message has unknown keys: {', '.join(unknown)}.")
    text = message.get("task")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The message has no task text.")
    text = text.strip()
    if ("todos" in message) == ("template" in message):
        raise ValueError("The message needs exactly one of todos and template.")
    if "template" in message:
        name = message["template"]
        template = templates.get(name) if isinstance(name, str) else None
        if template is None:
            known = ", ".join(templates) or "none"
            raise ValueError(f"There is no task template {name!r}. Known templates: {known}.")
        return template.prompt(text), template.todos
    todos = message["todos"]
    # A JSON object would iterate as its keys and seed one item per key.
    if not isinstance(todos, list):
        raise ValueError("The message's todos must be a list.")
    return text, todos


def _load_template(name: str, entry: Any, path: Path) -> TaskTemplate:
    where = f"{path}: template '{name}'"
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a table.")
    unknown = sorted(set(entry) - _TEMPLATE_KEYS)
    if unknown:
        raise ValueError(f"{where} has unknown keys: {', '.join(unknown)}.")
    # A template without keywords can still be picked by name on /walkie/task.
    keywords = entry.get("keywords", [])
    if not isinstance(keywords, list) or not all(
        isinstance(keyword, str) and keyword.strip() for keyword in keywords
    ):
        raise ValueError(f"{where}: keywords must be a list of words or phrases.")
    task = entry.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"{where} has no task text.")
    todos = entry.get("todos")
    if not isinstance(todos, list) or not todos:
        raise ValueError(f"{where} needs todos, a list of at least one step.")
    try:
        # A scratch list runs the same checks as the seed will.
        TodoList().seed(todos)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from e
    return TaskTemplate(
        name=name,
        keywords=tuple(" ".join(keyword.split()) for keyword in keywords),
        task=task.strip(),
        todos=todos,
    )


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    # Whole words only, so "hri" does not match "shrimp". The words of a
    # phrase may be separated by any whitespace. re caches compiled patterns.
    words = r"\s+".join(re.escape(word) for word in keyword.split())
    return re.compile(rf"(?<!\w){words}(?!\w)", re.IGNORECASE)
