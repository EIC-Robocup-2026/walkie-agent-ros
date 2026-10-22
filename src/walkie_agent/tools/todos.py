"""Todo list tools: edit the task's todo list, and stop to ask the person a question.

Both tools act on the TodoList of the current task (ADR-0041). Neither touches
ROS itself: wait_for_person speaks through the speak tool it is given.

A refusal raises ToolException rather than returning text, and
handle_tool_error stays off. RAI's ToolRunner then answers the call with a
ToolMessage of status "error" and the content "Failed to run tool. Error:
<message>", so the call counts as failed, not as a result to act on.
"""

from typing import Any, Type

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from walkie_agent.todo import TodoList, TodoOp, TodoRejected


class UpdateTodosToolInput(BaseModel):
    ops: list[TodoOp] = Field(
        description="The changes to make, applied in order. Either all of them apply, or none."
    )


class UpdateTodosTool(BaseTool):
    """Applies the LLM's edits to the todo list and returns a one-sentence summary.

    The rules live in TodoList.apply. This tool only adapts its result to a
    tool call, and never returns a LangGraph Command: RAI's ToolRunner reads
    the content of each result and cannot handle one.
    """

    name: str = "update_todos"
    description: str = (
        "Edit your todo list for the current task, referring to items by id. "
        "Ops: add (content, optional status and after), "
        "update (id plus any of content, active_form, status, note, after), "
        "remove (id). "
        "Statuses: pending, in_progress, completed, failed, cancelled. "
        "failed and cancelled need a note. Only one item can be in progress at a time. "
        "Seeded steps cannot be reworded or removed. "
        "Put all changes in one call."
    )
    args_schema: Type[UpdateTodosToolInput] = UpdateTodosToolInput
    todos: TodoList

    def _run(self, ops: list[TodoOp | dict[str, Any]]) -> str:
        # langchain-core passes validated TodoOp instances, but a direct call
        # may pass dicts; TodoList.apply accepts both.
        try:
            return self.todos.apply(ops)
        except TodoRejected as e:
            # apply is all or nothing; say so, or the LLM may resend only the
            # op that was refused.
            raise ToolException(f"{e} None of the changes were applied.") from e


class WaitForPersonToolInput(BaseModel):
    question: str = Field(
        description=(
            "The question to ask out loud, in plain words, for example "
            "'Do you mean the red mug or the blue one?'"
        )
    )


class WaitForPersonTool(BaseTool):
    """Asks the person a question out loud and marks the current item waiting.

    The answer arrives as the next message while the agent is idle, so it
    never interrupts a run. Ending the turn after this tool is the agent
    graph's job, not the tool's.
    """

    name: str = "wait_for_person"
    description: str = (
        "Ask the person a question out loud and wait for the answer. "
        "Use this instead of speak when you cannot go on without an answer, "
        "for example to learn which object they mean. "
        "Your turn ends, and the answer arrives as the next message."
    )
    args_schema: Type[WaitForPersonToolInput] = WaitForPersonToolInput
    todos: TodoList
    speak_tool: BaseTool

    def _run(self, question: str) -> str:
        result = str(self.speak_tool.invoke({"text": question}))
        # SpeakTool returns "Said: <text>" on success and a sentence about the
        # failure otherwise. An unheard question must not leave the item waiting.
        if not result.startswith("Said:"):
            raise ToolException(f"Could not ask the question: {result}")
        # With no item in progress there is nothing to mark, but the question
        # was still asked.
        self.todos.set_waiting(question)
        return f"Asked: {question}"
