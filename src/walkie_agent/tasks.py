"""Task templates and operator task messages: two ways to seed a task (ADR-0041).

A fixed task, such as Receptionist, can start with its steps already on the
todo list. This module reads where those steps come from, without ROS:

- Templates, loaded from `config/task_templates.toml`. Only the operator picks
  one, by name, so a stray word in speech cannot start the wrong plan.
- Messages on `/walkie/task`, a std_msgs/String holding JSON: either
  `{"task": "...", "todos": [...]}` or `{"task": "...", "template": "name"}`.

Pydantic checks both formats. `extra="forbid"` matters, because an unknown key
is most likely a typo, such as "items", that would silently drop a seed.
The agent (walkie_agent.agent) subscribes to the topic and submits the result.
"""

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from walkie_agent.todo import TodoList

# Only task text is stripped: a template name must match exactly.
_TaskText = Annotated[StrictStr, StringConstraints(strip_whitespace=True, min_length=1)]


class _TemplateEntry(BaseModel):
    """One `[templates.<name>]` table. The name is the table's key, not a field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: _TaskText  # instructions for the LLM, sent after the person's request
    todos: list[Any] = Field(min_length=1)  # the seed, in the shape TodoList.seed takes

    @field_validator("todos")
    @classmethod
    def _can_seed(cls, todos: list[Any]) -> list[Any]:
        # A scratch list runs the same checks as the seed will.
        TodoList().seed(todos)
        return todos


class TaskTemplate(_TemplateEntry):
    """A fixed task whose steps are written in advance."""

    name: str

    def prompt(self, text: str) -> str:
        """The message to run: the request, a blank line, then the template's task."""
        return f"{text}\n\n{self.task}"


class _TemplateFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    templates: dict[str, _TemplateEntry] = Field(default_factory=dict)


class TaskMessage(BaseModel):
    """A `/walkie/task` message. Its seed's items are checked by WalkieAgent.submit."""

    model_config = ConfigDict(extra="forbid")

    task: _TaskText
    # The defaults are never read: a null is refused by the type, and the
    # validator below requires exactly one of the two keys to be present.
    todos: list[Any] = Field(default_factory=list)
    template: StrictStr = ""

    @model_validator(mode="after")
    def _exactly_one_seed(self) -> "TaskMessage":
        if ("todos" in self.model_fields_set) == ("template" in self.model_fields_set):
            raise ValueError("needs exactly one of todos and template")
        return self


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
    try:
        entries = _TemplateFile.model_validate(data).templates
    except ValidationError as e:
        raise ValueError(
            f"{path}: {_reason(e)}. Put each template in a [templates.<name>] table "
            "with task and todos."
        ) from e
    return {
        name: TaskTemplate(name=name, task=entry.task, todos=entry.todos)
        for name, entry in entries.items()
    }


def parse_task_message(
    data: str, templates: Mapping[str, TaskTemplate]
) -> tuple[str, list[str | dict[str, Any]]]:
    """Read a `/walkie/task` message into the text to run and its seed.

    The message is a JSON object with the task's text and exactly one of
    `todos`, the seed itself, or `template`, the name of a template whose
    task and todos to use. Raises ValueError, with a one-line reason, for a
    malformed message.
    """
    try:
        message = TaskMessage.model_validate_json(data)
    except ValidationError as e:
        raise ValueError(f"The task message is malformed: {_reason(e)}.") from e
    if "todos" in message.model_fields_set:
        return message.task, message.todos
    template = templates.get(message.template)
    if template is None:
        known = ", ".join(templates) or "none"
        raise ValueError(
            f"There is no task template {message.template!r}. Known templates: {known}."
        )
    return template.prompt(message.task), template.todos


def _reason(error: ValidationError) -> str:
    """Pydantic's multi-line error as one line: `<location>: <message>`, joined by `; `."""
    return "; ".join(
        ": ".join(filter(None, (".".join(map(str, e["loc"])), e["msg"]))) for e in error.errors()
    )
