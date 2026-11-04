"""as_tool, bind and the single error path: schemas from docstrings, ctx hidden."""

import asyncio
import inspect
import json
import logging
import typing

import pydantic
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from scripted_llm import ScriptedChatModel, tool_call

from walkie_agent.adapter import (
    as_tool,
    as_tools,
    bind,
    format_tool_error,
    tool_error_middleware,
)
from walkie_agent.atomics import (
    DEFAULT,
    GPSR,
    HRI,
    find_objects,
    go_to,
    list_places,
    move_relative,
    say,
)
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.errors import ErrorCode, StepError


@pytest.fixture
def robot() -> FakeRobot:
    return FakeRobot(seed=1)


def run_agent(robot: FakeRobot, tools, call) -> ToolMessage:
    """One scripted tool call through a real create_agent, with the real middleware."""
    agent = _agent(tools, call)
    messages = agent.invoke({"messages": [("user", "Go.")]})["messages"]
    return next(message for message in messages if isinstance(message, ToolMessage))


def _agent(tools, call):
    model = ScriptedChatModel(responses=[call, AIMessage(content="Done.")])
    return create_agent(model, tools=tools, middleware=[tool_error_middleware()])


# Schemas


def test_the_schema_comes_from_the_signature_and_the_docstring(robot):
    tool = as_tool(move_relative, robot.ctx)
    assert tool.name == "move_relative"
    assert tool.description.startswith("Move a short distance")
    assert "at most 3 m" in tool.description
    schema = tool.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"forward_m", "left_m", "turn_deg"}
    assert schema["required"] == ["forward_m"]
    assert schema["properties"]["left_m"]["default"] == 0.0
    assert schema["properties"]["forward_m"]["type"] == "number"
    assert schema["properties"]["forward_m"]["description"].startswith("Meters to move forward")


def test_the_description_carries_what_comes_back_and_not_the_args_section(robot):
    description = as_tool(find_objects, robot.ctx).description
    assert "keys id, name, category and place" in " ".join(description.split())
    assert "Args:" not in description


def test_a_function_with_no_arguments_has_an_empty_schema(robot):
    assert as_tool(list_places, robot.ctx).args == {}


@pytest.mark.parametrize(
    "args",
    [{"forward_m": True}, {"forward_m": "1"}, {"forward_m": 0.5, "left_m": True}],
)
def test_a_tool_refuses_a_bool_or_a_numeric_string_for_a_number(robot, args):
    before = robot.pose()
    with pytest.raises(pydantic.ValidationError):
        as_tool(move_relative, robot.ctx).invoke(args)
    assert robot.pose() == before


def test_a_tool_still_takes_an_int_for_a_number(robot):
    before = robot.pose()
    as_tool(move_relative, robot.ctx).invoke({"forward_m": 1})
    assert robot.pose() != before


def test_a_bool_from_the_model_comes_back_as_an_error_and_moves_nothing(robot):
    before = robot.pose()
    call = tool_call("move_relative", {"forward_m": True}, "c1")
    message = run_agent(robot, [as_tool(move_relative, robot.ctx)], call)
    assert message.status == "error"
    assert robot.pose() == before


def test_bind_refuses_a_bool_through_the_verb_s_guard(robot):
    before = robot.pose()
    with pytest.raises(StepError) as caught:
        bind(move_relative, robot.ctx)(True)
    assert caught.value.code is ErrorCode.REFUSED
    assert robot.pose() == before


@pytest.mark.parametrize("tool_set", [GPSR, HRI, DEFAULT], ids=["GPSR", "HRI", "DEFAULT"])
def test_every_tool_set_builds_with_ctx_hidden(robot, tool_set):
    tools = as_tools(tool_set, robot.ctx)
    assert [tool.name for tool in tools] == [fn.__name__ for fn in tool_set]
    for tool in tools:
        assert "ctx" not in tool.args
        assert "ctx" not in tool.description


def test_a_parameter_with_no_description_is_refused(robot):
    def shrug(ctx, how_much: int) -> str:
        """Shrug.

        Args:
            how_much:
        """
        return "ok"

    with pytest.raises(ValueError, match="how_much"):
        as_tool(shrug, robot.ctx)


def test_a_docstring_with_no_args_section_is_refused(robot):
    def shrug(ctx, how_much: int) -> str:
        """Shrug."""
        return "ok"

    with pytest.raises(ValueError):
        as_tool(shrug, robot.ctx)


def test_a_function_without_ctx_first_is_refused(robot):
    def shrug(how_much: int) -> str:
        """Shrug.

        Args:
            how_much: How much.
        """
        return "ok"

    with pytest.raises(ValueError, match="ctx"):
        as_tool(shrug, robot.ctx)
    with pytest.raises(ValueError, match="ctx"):
        bind(shrug, robot.ctx)


def test_duplicate_names_are_refused(robot):
    with pytest.raises(ValueError, match="say"):
        as_tools([say, go_to, say], robot.ctx)


# Results


def test_a_tool_returns_a_string_as_it_is_and_runs_against_ctx(robot):
    assert as_tool(go_to, robot.ctx).invoke({"place": "shelf"}) == "Arrived at shelf."
    assert robot.snapshot()["at"] == "shelf"


def test_a_tool_returns_a_list_as_json_text(robot):
    robot.go_to("living room")
    reply = as_tool(list_places, robot.ctx).invoke({})
    assert isinstance(reply, str)
    assert {"name": "shelf", "description": "In the kitchen room."} in json.loads(reply)


def test_an_empty_result_reaches_the_model_as_brackets_not_an_empty_message(robot):
    tools = as_tools([find_objects], robot.ctx)
    message = run_agent(robot, tools, tool_call("find_objects", {"what": "no such thing"}, "c1"))
    assert message.status != "error"
    assert message.content == "[]"


def test_non_ascii_text_stays_readable(robot):
    def names(ctx) -> list[str]:
        """List names.

        Returns a list of names.
        """
        return ["ธวัช"]

    assert as_tool(names, robot.ctx).invoke({}) == '["ธวัช"]'


# The error path


def test_a_step_error_reaches_the_model_as_code_and_message(robot):
    tools = as_tools([move_relative], robot.ctx)
    message = run_agent(robot, tools, tool_call("move_relative", {"forward_m": 9.0}, "c1"))
    assert message.status == "error"
    assert message.content.startswith("REFUSED: Refused: a move of 9.00 m")
    assert robot.calls[-1]["error"] == "REFUSED"


def test_a_step_error_takes_the_same_path_on_the_async_route(robot):
    agent = _agent(as_tools([go_to], robot.ctx), tool_call("go_to", {"place": "moon"}, "c1"))
    messages = asyncio.run(agent.ainvoke({"messages": [("user", "Go.")]}))["messages"]
    message = next(message for message in messages if isinstance(message, ToolMessage))
    assert message.status == "error"
    assert message.content.startswith("NOT_FOUND: Unknown place 'moon'.")


def test_a_bug_reaches_the_model_as_internal_with_no_detail_and_is_logged(robot, caplog):
    def broken(ctx) -> str:
        """Fail like a bug would.

        Returns nothing, ever.
        """
        raise KeyError("secret detail")

    with caplog.at_level(logging.ERROR, logger="walkie_agent.adapter"):
        message = run_agent(
            robot, as_tools([broken], robot.ctx), tool_call("broken", {}, "c1")
        )
    assert message.status == "error"
    assert message.content == "INTERNAL: broken failed with KeyError."
    assert "secret detail" in caplog.text


def test_a_wrong_argument_is_answered_before_the_middleware(robot):
    tools = as_tools([go_to], robot.ctx)
    message = run_agent(robot, tools, tool_call("go_to", {"where": "shelf"}, "c1"))
    assert message.status == "error"
    assert not message.content.startswith(("INTERNAL", "NOT_FOUND"))
    assert robot.calls == []


def test_format_tool_error_needs_no_request_for_a_step_error():
    error = StepError(ErrorCode.TIMEOUT, "Too slow.")
    assert format_tool_error(error, None) == "TIMEOUT: Too slow."


# bind


def test_bind_returns_a_plain_callable_with_ctx_hidden(robot):
    call = bind(find_objects, robot.ctx)
    assert call.__name__ == "find_objects"
    assert call.__doc__ == find_objects.__doc__
    assert list(typing.get_type_hints(call)) == ["what", "return"]
    assert list(inspect.signature(call).parameters) == ["what"]


def test_bind_returns_the_atomic_s_raw_value(robot):
    robot.go_to("living room")
    places = bind(list_places, robot.ctx)()
    assert isinstance(places, list)
    assert places[0].keys() == {"name", "description"}


def test_bind_takes_positional_and_keyword_arguments(robot):
    call = bind(move_relative, robot.ctx)
    assert call(0.5, turn_deg=10.0).startswith("Moved 0.5 m forward, turning 10 degrees left.")


def test_bind_lets_a_step_error_through(robot):
    with pytest.raises(StepError) as caught:
        bind(go_to, robot.ctx)("moon")
    assert caught.value.code is ErrorCode.NOT_FOUND


# A stop


@pytest.mark.parametrize(
    "start",
    [
        lambda robot: bind(say, robot.ctx)("Too late."),
        lambda robot: as_tool(say, robot.ctx).invoke({"text": "Too late."}),
        lambda robot: bind(list_places, robot.ctx)(),
    ],
    ids=["bind", "tool", "read"],
)
def test_nothing_starts_once_the_token_is_set(robot, start):
    robot.cancel.cancel()
    with pytest.raises(StepError) as caught:
        start(robot)
    assert caught.value.code is ErrorCode.CANCELLED
    assert robot.said == []
    assert robot.calls == []


def test_bind_works_with_validate_call_as_the_monty_driver_will_use_it(robot):
    call = pydantic.validate_call(bind(move_relative, robot.ctx))
    assert call(forward_m="0.5").startswith("Moved 0.5 m forward")
    with pytest.raises(pydantic.ValidationError):
        call(forward_m="far")
    with pytest.raises(pydantic.ValidationError):
        call(forward_m=0.5, ctx=robot.ctx)
