"""The agent's tool-call record (tool_log.py), written by WalkieToolRunner.

The graph runs here as in test_graph.py, on a scripted LLM, with a real
HandleStore and no ROS.
"""

import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException, tool
from pydantic import BaseModel, ValidationError

from scripted_llm import ScriptedChatModel, tool_call
from walkie_agent.agent import WalkieAgent
from walkie_agent.arm.state import ARMS_OUT
from walkie_agent.graph import AFTER_QUESTION, ToolSet, WalkieToolRunner
from walkie_agent.jsonl import read_jsonl
from walkie_agent.perception.vision_feed import TrackKey
from walkie_agent.todo import TodoList
from walkie_agent.tool_log import CURRENT_CALL, ToolLog
from walkie_agent.toolbox import (
    AtomicTool,
    Effect,
    HandleStore,
    Pose2D,
    StepError,
    Toolbox,
    ToolMeta,
    ToolOutput,
)
from walkie_agent.tools.atomic import _perception  # noqa: F401  registers person and seat
from walkie_agent.tools.navigation import _arms_not_safe
from walkie_agent.tools.todos import UpdateTodosTool, WaitForPersonTool

ANA = SimpleNamespace(track=TrackKey(7, 1))
BEN = SimpleNamespace(track=TrackKey(9, 2))
# A person find_people saw without a track.
UNTRACKED = SimpleNamespace(track=None)


class FakeClock:
    """The node clock: starts at 12.5 s and moves 0.25 s on every read."""

    def __init__(self) -> None:
        self.t = 12.5

    def __call__(self) -> float:
        now = self.t
        self.t += 0.25
        return now


class World:
    """The robot's pose for the handle store, which a test can move."""

    def __init__(self) -> None:
        self.pose = Pose2D(0.0, 0.0, 0.0)


class FakeGaze:
    def __init__(self, target: str | None, running: bool) -> None:
        self.target = target
        self.running = running

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(target=self.target, running=self.running, kind="person")


class TargetArgs(BaseModel):
    target: str


class Looked(ToolOutput):
    target: str
    track_id: int | None


def look(ctx, args: TargetArgs) -> Looked:
    handle = ctx.handles.get(args.target, kind="person")
    track = handle.payload.track
    return Looked(
        summary=f"Looking at {handle.id}.",
        target=handle.id,
        track_id=None if track is None else track.track_id,
    )


def atomic_tool(toolbox: Toolbox, name: str, run, args: type[BaseModel] = TargetArgs) -> AtomicTool:
    meta = ToolMeta(
        name=name,
        description=f"The {name} atomic.",
        effects=frozenset({Effect.M}),
        questions=("Do it.", "Go on.", "Please do it now."),
        examples=({"target": "per_51c0.0"},),
    )
    return AtomicTool(SimpleNamespace(META=meta, Args=args, Out=Looked, run=run), toolbox.ctx)


def fake_speak() -> BaseTool:
    @tool
    def speak(text: str) -> str:
        """Say something out loud."""
        return f"Said: {text}"

    return speak


def drive_refused() -> BaseTool:
    def navigate_to_location(location: str) -> str:
        """Drive to a named place."""
        raise ToolException(_arms_not_safe(("left",), "driving", "'carry_left'"))

    return StructuredTool.from_function(navigate_to_location, handle_tool_error=True)


class Rig:
    """An agent with a tool log, a toolbox and a person set per_51c0 of Ana and Ben."""

    def __init__(
        self, tmp_path: Path, gaze: FakeGaze | None = None, log: bool = True, look_run=look
    ) -> None:
        self.world = World()
        self.look_run = look_run
        self.toolbox = Toolbox(connector=None)
        ids = iter([0x51C0, 0x5EA7, 0x0A11])
        self.toolbox.ctx.handles = HandleStore(lambda: self.world.pose, id_source=ids.__next__)
        self.people = self.toolbox.ctx.handles.put_set("person", [ANA, BEN])
        self.seats = self.toolbox.ctx.handles.put_set("seat", [SimpleNamespace(free=True)])
        self.clock = FakeClock()
        self.path = tmp_path / "logs" / "agent_tools.jsonl"
        self.log = (
            ToolLog(self.path, self.clock, self.toolbox.ctx.handles, gaze) if log else None
        )
        if self.log is not None:
            self.toolbox.on_new_task(self.log.new_task)
        self.todos = TodoList()

    def tools(self) -> list[BaseTool]:
        speak = fake_speak()
        return [
            atomic_tool(self.toolbox, "look_at", self.look_run),
            speak,
            UpdateTodosTool(todos=self.todos),
            WaitForPersonTool(todos=self.todos, speak_tool=speak),
            drive_refused(),
        ]

    def run(self, *responses: AIMessage) -> list[BaseMessage]:
        llm = ScriptedChatModel(responses=[*responses, AIMessage("Done.")])
        agent = WalkieAgent(
            target_connectors={},
            llm=llm,
            tools=self.tools(),
            stream_response=False,
            todos=self.todos,
            toolbox=self.toolbox,
            tool_log=self.log,
        )
        state = {"messages": [HumanMessage(content="Do it.")]}
        agent.agent.invoke(state)
        return state["messages"]

    def records(self) -> list[dict]:
        return read_jsonl(self.path)

    def calls(self) -> list[dict]:
        return [record for record in self.records() if record["type"] == "call"]

    def call(self, call_id: str) -> dict:
        return next(record for record in self.calls() if record["call_id"] == call_id)


def person(handle: str, where: str, track: TrackKey | None) -> dict:
    return {
        "handle": handle,
        "in": where,
        "kind": "person",
        "track_id": None if track is None else track.track_id,
        "generation": None if track is None else track.generation,
    }


def result_of(messages: list[BaseMessage], call_id: str) -> ToolMessage:
    return next(m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == call_id)


def test_the_log_opens_with_a_start_record(tmp_path):
    rig = Rig(tmp_path)

    (start,) = rig.records()

    assert start["type"] == "start"
    assert start["t"] == 12.5
    assert isinstance(start["wall_time"], str) and "T" in start["wall_time"]


def test_a_successful_atomic_records_its_out_and_the_track_it_named(tmp_path):
    rig = Rig(tmp_path)

    rig.run(tool_call("look_at", {"target": "per_51c0.0"}, "call_look"))

    record = rig.call("call_look")
    assert record["tool"] == "look_at"
    assert record["args"] == {"target": "per_51c0.0"}
    assert (record["outcome"], record["code"]) == ("ok", None)
    assert record["result"] == "Looking at per_51c0.0."
    assert record["out"] == {"summary": "Looking at per_51c0.0.", "target": "per_51c0.0",
                             "track_id": 7}
    assert record["people"] == [
        person("per_51c0.0", "args", ANA.track),
        person("per_51c0.0", "result", ANA.track),
    ]
    assert record["t"] < record["t_end"]
    assert record["wall_s"] >= 0.0
    assert record["gaze"] is None


def test_a_step_error_records_its_code_and_no_out(tmp_path):
    def refuse(ctx, args):
        raise StepError("REFUSED", f"{args.target} is not a guest.")

    rig = Rig(tmp_path, look_run=refuse)

    rig.run(tool_call("look_at", {"target": "per_51c0.1"}, "call_look"))

    record = rig.call("call_look")
    assert (record["outcome"], record["code"]) == ("error", "REFUSED")
    assert record["result"] == "REFUSED: per_51c0.1 is not a guest."
    assert record["out"] is None
    assert record["people"][0] == person("per_51c0.1", "args", BEN.track)


def test_a_drive_refused_for_an_arm_out_keeps_the_text_without_a_code(tmp_path):
    rig = Rig(tmp_path)

    rig.run(tool_call("navigate_to_location", {"location": "door"}, "call_drive"))

    record = rig.call("call_drive")
    assert (record["outcome"], record["code"], record["out"]) == ("error", None, None)
    assert ARMS_OUT in record["result"]
    assert record["result"].startswith("Not driving: the left arm is")


def test_bad_arguments_are_invalid(tmp_path):
    rig = Rig(tmp_path)

    messages = rig.run(tool_call("look_at", {"target": ["per_51c0.0"]}, "call_bad"))

    record = rig.call("call_bad")
    assert record["outcome"] == "invalid"
    assert record["result"].startswith("ValidationError: ")
    assert record["out"] is None
    # The LLM still reads RAI's own message.
    assert "Validation error in tool look_at" in result_of(messages, "call_bad").content


def test_an_unknown_tool_is_not_run(tmp_path):
    rig = Rig(tmp_path)

    rig.run(tool_call("fly", {"to": "seat_5ea7.0"}, "call_fly"))

    record = rig.call("call_fly")
    assert (record["outcome"], record["result"]) == ("not_run", "Unknown tool fly.")
    # A seat has no track.
    assert record["people"] == [
        {"handle": "seat_5ea7.0", "in": "args", "kind": "seat", "track_id": None,
         "generation": None}
    ]


def test_a_call_after_a_question_is_not_run_and_the_question_records_the_gaze(tmp_path):
    rig = Rig(tmp_path, gaze=FakeGaze("per_51c0.1", running=True))
    question = AIMessage(
        content="",
        tool_calls=[
            {"name": "wait_for_person", "args": {"question": "What is your name?"},
             "id": "call_ask", "type": "tool_call"},
            {"name": "look_at", "args": {"target": "per_51c0.0"}, "id": "call_look",
             "type": "tool_call"},
        ],
    )

    rig.run(question)

    asked = rig.call("call_ask")
    assert asked["outcome"] == "ok"
    assert asked["gaze"] == {"target": "per_51c0.1", "running": True, "track_id": 9}
    skipped = rig.call("call_look")
    assert (skipped["outcome"], skipped["result"], skipped["out"]) == (
        "not_run", AFTER_QUESTION, None
    )
    # Only speech records the gaze.
    assert skipped["gaze"] is None


def test_speak_records_the_gaze_at_its_start(tmp_path):
    gaze = FakeGaze(None, running=False)
    rig = Rig(tmp_path, gaze=gaze)

    rig.run(tool_call("speak", {"text": "Hello."}, "call_hi"))
    gaze.target, gaze.running = "per_51c0.0", True
    rig.run(tool_call("speak", {"text": "Hello, Ana."}, "call_ana"))

    assert rig.call("call_hi")["gaze"] == {"target": None, "running": False, "track_id": None}
    assert rig.call("call_ana")["gaze"] == {"target": "per_51c0.0", "running": True,
                                            "track_id": 7}


def test_a_set_id_names_every_member(tmp_path):
    rig = Rig(tmp_path)

    rig.run(tool_call("speak", {"text": "per_51c0 and per_ffff.0, hello."}, "call_set"))

    record = rig.call("call_set")
    args = [entry for entry in record["people"] if entry["in"] == "args"]
    assert args == [
        person("per_51c0.0", "args", ANA.track),
        person("per_51c0.1", "args", BEN.track),
        # An id the store does not know is still named.
        {"handle": "per_ffff.0", "in": "args", "kind": None, "track_id": None,
         "generation": None},
    ]


def test_a_stale_handle_still_names_its_track(tmp_path):
    rig = Rig(tmp_path)
    rig.world.pose = Pose2D(1.0, 0.0, 0.0)

    rig.run(tool_call("look_at", {"target": "per_51c0.1"}, "call_stale"))

    record = rig.call("call_stale")
    assert (record["outcome"], record["code"]) == ("error", "STALE")
    assert record["people"][0] == person("per_51c0.1", "args", BEN.track)


def test_an_untracked_person_has_no_track(tmp_path):
    rig = Rig(tmp_path)
    alone = rig.toolbox.ctx.handles.put_set("person", [UNTRACKED], pose=Pose2D(0.0, 0.0, 0.0))

    rig.run(tool_call("look_at", {"target": alone.member_ids[0]}, "call_look"))

    assert rig.call("call_look")["people"][0] == person(alone.member_ids[0], "args", None)


def test_a_new_task_is_recorded(tmp_path):
    rig = Rig(tmp_path)

    rig.toolbox.new_task()

    assert [record["type"] for record in rig.records()] == ["start", "new_task"]
    assert rig.records()[1]["t"] == 12.75


def test_a_write_failure_is_logged_once_and_the_call_goes_on(tmp_path, caplog, monkeypatch):
    rig = Rig(tmp_path)

    def fail(record):
        raise OSError("No space left on device")

    monkeypatch.setattr(rig.log._log, "write", fail)
    with caplog.at_level(logging.ERROR, logger="walkie_agent.tool_log"):
        messages = rig.run(
            tool_call("look_at", {"target": "per_51c0.0"}, "call_1"),
            tool_call("look_at", {"target": "per_51c0.1"}, "call_2"),
        )
        rig.toolbox.new_task()

    assert result_of(messages, "call_1").content == "Looking at per_51c0.0."
    assert result_of(messages, "call_2").content == "Looking at per_51c0.1."
    failures = [r for r in caplog.records if r.name == "walkie_agent.tool_log"]
    assert len(failures) == 1
    assert failures[0].exc_info is not None


def test_a_clock_that_fails_leaves_t_null(tmp_path):
    rig = Rig(tmp_path)

    def broken() -> float:
        raise RuntimeError("The node is shut down.")

    rig.log._clock = broken
    messages = rig.run(tool_call("look_at", {"target": "per_51c0.0"}, "call_look"))

    assert result_of(messages, "call_look").status == "success"
    record = rig.call("call_look")
    assert (record["t"], record["t_end"], record["outcome"]) == (None, None, "ok")


def test_every_line_is_json_with_a_finite_t(tmp_path):
    rig = Rig(tmp_path, gaze=FakeGaze("per_51c0.0", running=True))

    rig.run(
        tool_call("look_at", {"target": "per_51c0.0"}, "call_look"),
        tool_call("speak", {"text": "Hello."}, "call_hi"),
        tool_call("fly", {}, "call_fly"),
        tool_call("look_at", {"target": 3}, "call_bad"),
    )
    rig.toolbox.new_task()

    lines = rig.path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["type"] for r in records] == ["start", *["call"] * 4, "new_task"]
    assert all(isinstance(r["t"], float) and math.isfinite(r["t"]) for r in records)


@pytest.mark.parametrize(
    ("args", "error"),
    [({"target": "per_51c0.0"}, None), ({"target": ["per_51c0.0"]}, ValidationError)],
)
def test_a_recorded_call_leaves_no_current_call_behind(tmp_path, args, error):
    # Called here, not through the graph, which runs each call in a copy of
    # the context. A call left current would take the Out of a later
    # AtomicTool.call into a record already written.
    rig = Rig(tmp_path)
    runner = WalkieToolRunner(ToolSet(rig.tools()), rig.todos, tool_log=rig.log)
    call = {"name": "look_at", "args": args, "id": "call_look", "type": "tool_call"}

    if error is None:
        runner._run_call("look_at", call, None)
    else:
        with pytest.raises(error):
            runner._run_call("look_at", call, None)

    assert CURRENT_CALL.get() is None
    assert rig.call("call_look")["outcome"] == ("ok" if error is None else "invalid")


@pytest.mark.parametrize("with_log", [False, True])
def test_the_log_changes_neither_the_results_nor_the_ledger(tmp_path, with_log):
    # The same script with and without a log: the same messages and the same
    # todo list, the failed look_at still blocking "completed".
    rig = Rig(tmp_path, log=with_log)
    rig.todos.seed([{"content": "Look at Ana", "requires_tool": "look_at"}])
    rig.world.pose = Pose2D(1.0, 0.0, 0.0)

    messages = rig.run(
        tool_call("update_todos", {"ops": [{"op": "update", "id": "t1",
                                            "status": "in_progress"}]}, "call_start"),
        tool_call("look_at", {"target": "per_51c0.0"}, "call_look"),
        tool_call("update_todos", {"ops": [{"op": "update", "id": "t1",
                                            "status": "completed"}]}, "call_done"),
        tool_call("fly", {}, "call_fly"),
        tool_call("update_todos", {"ops": [{"op": "update", "id": "t1", "status": "failed",
                                            "note": "Ana moved away."}]}, "call_fail"),
    )

    contents = [(type(m).__name__, m.content) for m in messages]
    assert contents == [
        ("HumanMessage", "Do it."),
        ("AIMessage", ""),
        ("ToolMessage", result_of(messages, "call_start").content),
        ("AIMessage", ""),
        ("ToolMessage", result_of(messages, "call_look").content),
        ("AIMessage", ""),
        ("ToolMessage", result_of(messages, "call_done").content),
        ("AIMessage", ""),
        ("ToolMessage", "Unknown tool fly."),
        ("AIMessage", ""),
        ("ToolMessage", result_of(messages, "call_fail").content),
        ("AIMessage", "Done."),
    ]
    assert result_of(messages, "call_look").content.startswith("STALE: per_51c0.0 is stale")
    assert result_of(messages, "call_done").status == "error"
    item = rig.todos.snapshot()[0]
    assert (item.status, item.evidence) == ("failed", [])
    assert rig.path.exists() is with_log
