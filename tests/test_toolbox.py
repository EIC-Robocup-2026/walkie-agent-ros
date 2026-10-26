"""The toolbox core (ADR-0037): shared types, AtomicTool, handles, the registry and ctx.

Most tests need no ROS: a ToolContext built with a stand-in connector never
touches it. The ctx tests that create entities or read TF use the
`connector` fixture, on Fast DDS and a random domain (ADR-0019).
"""

import importlib
import logging
import math
import pickle
import re
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
from geometry_msgs.msg import TransformStamped
from langchain_core.messages import ToolMessage
from nav2_msgs.action import NavigateToPose
from pydantic import BaseModel, Field, ValidationError
from rclpy.task import Future
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler

from walkie_agent.testing import FakeClock, FakeObject, FakePerson, fake_ctx
from walkie_agent.toolbox import (
    ERROR_CODES,
    AtomicLoadError,
    AtomicTool,
    Effect,
    ErrorCode,
    HandleStore,
    KindPolicy,
    Pose2D,
    StepError,
    ToolContext,
    Toolbox,
    ToolMeta,
    ToolOutput,
    build_atomic_tools,
    discover,
    register_kind,
)
from walkie_agent.toolbox import handles as handles_module
from walkie_agent.tools import build_walkie_tools
from walkie_agent.tools.waiters import wait_for_future

PERSON = KindPolicy(
    prefix="per", max_translation_m=0.5, max_rotation_deg=20.0, max_age_s=60.0,
    source="find_people",
)
SEAT = KindPolicy(prefix="seat", max_translation_m=0.05, max_rotation_deg=3.0, max_age_s=600.0)
# No limit at all, so a lookup never asks for the robot's pose.
NOTE = KindPolicy(prefix="note", max_translation_m=None, max_rotation_deg=None, max_age_s=None)


@pytest.fixture(autouse=True)
def kinds(monkeypatch):
    """A private kind registry, with only the built-in kind and the kinds these tests use.

    The shipped atomics register their own person and seat kinds when first
    imported. They are imported here, before the swap, so that a test which
    discovers them registers nothing into this registry, and the real kinds
    are left out of it, so these tests can use their own policies.
    """
    importlib.import_module("walkie_agent.tools.atomic._perception")
    monkeypatch.setattr(handles_module, "_kinds", {"object": handles_module.OBJECT_POLICY})
    register_kind("person", PERSON)
    register_kind("seat", SEAT)
    register_kind("note", NOTE)


# -- Shared types ----------------------------------------------------------------


def meta(name: str = "echo_text", **fields) -> ToolMeta:
    values = {
        "name": name,
        "description": "Say the text back.",
        "effects": frozenset({Effect.P}),
        "questions": ("Echo this.", "Say it back.", "Repeat after me."),
        "examples": ({"text": "hello"},),
        **fields,
    }
    return ToolMeta(**values)


def test_error_codes_are_the_thirteen_codes_with_a_meaning():
    assert list(ERROR_CODES) == list(get_args(ErrorCode))
    assert len(ERROR_CODES) == 13
    assert all(meaning.strip() for meaning in ERROR_CODES.values())


def test_the_docs_table_lists_every_error_code_in_order():
    # The table in agent.md is a hand copy; this keeps it from drifting.
    text = (Path(__file__).parents[1] / "docs/architecture/agent.md").read_text()
    section = text.split("### Error codes", 1)[1].split("\n### ", 1)[0]
    codes = re.findall(r"^\| `([A-Z_]+)` \|", section, flags=re.MULTILINE)
    assert codes == list(ERROR_CODES)


def test_step_error_reads_as_code_and_message():
    err = StepError("NOT_FOUND", "No one is in view.")
    assert str(err) == "NOT_FOUND: No one is in view."
    assert (err.code, err.message, err.data) == ("NOT_FOUND", "No one is in view.", {})

    # RAI pickles tool results to disk, so an error must survive a round trip
    # of its own bytes; nothing untrusted is unpickled here.
    ambiguous = StepError("AMBIGUOUS", "Two mugs match.", {"candidates": ["obj_1a2b.0"]})
    copy = pickle.loads(pickle.dumps(ambiguous))
    assert (copy.code, copy.message, copy.data) == (
        "AMBIGUOUS", "Two mugs match.", {"candidates": ["obj_1a2b.0"]}
    )


def test_step_error_rejects_an_unknown_code():
    with pytest.raises(ValueError, match="Unknown StepError code 'BROKEN'"):
        StepError("BROKEN", "Something broke.")  # type: ignore[arg-type]


def test_tool_meta_accepts_a_sound_declaration():
    m = meta(effects=["R", "P"], requires=["vision"])
    assert m.effects == frozenset({Effect.R, Effect.P})
    assert (m.status, m.created_by, m.runtime, m.requires) == (
        "verified", "human", "inprocess", ("vision",)
    )
    with pytest.raises(ValidationError):
        m.name = "other_name"  # frozen


@pytest.mark.parametrize(
    "fields",
    [
        {"name": "Echo"},
        {"name": "ab"},
        {"name": "1echo"},
        {"name": "echo-text"},
        {"name": "e" * 49},
        {"description": "  "},
        {"effects": ["X"]},
        {"questions": ("One?", "Two?")},
        {"questions": ("One?", "Two?", "Three?", "Four?", "Five?", "Six?")},
        {"questions": ("One?", "Two?", " ")},
        {"examples": ()},
        {"status": "experimental"},
        {"runtime": "sandbox"},
        {"effect": ["P"]},  # a misspelt field is rejected, not ignored
    ],
    ids=lambda fields: next(iter(fields)) + "=" + repr(next(iter(fields.values())))[:12],
)
def test_tool_meta_rejects_a_bad_declaration(fields):
    with pytest.raises(ValidationError):
        meta(**fields)


@pytest.mark.parametrize(
    ("effects", "idempotent"),
    [((), True), (("R",), True), (("R", "P"), True), (("P", "S"), False), (("W",), False)],
)
def test_only_reading_and_perceiving_are_idempotent(effects, idempotent):
    assert meta(effects=effects).idempotent is idempotent


def test_tool_output_needs_a_summary():
    assert ToolOutput(summary="per_51c0.0 person\nper_51c0.1 person").summary.count("\n") == 1
    with pytest.raises(ValidationError):
        ToolOutput(summary=" \n ")


# -- AtomicTool ------------------------------------------------------------------


class EchoArgs(BaseModel):
    text: str = Field(description="What to say back.")


class EchoOut(ToolOutput):
    length: int


def atomic(run, name: str = "echo_text", requires: tuple[str, ...] = ()) -> SimpleNamespace:
    """An atomic as a namespace, which has the same four names a module has."""
    return SimpleNamespace(META=meta(name, requires=requires), Args=EchoArgs, Out=EchoOut, run=run)


def echo(ctx, args: EchoArgs) -> EchoOut:
    return EchoOut(summary=f"Echo: {args.text}", length=len(args.text))


def call(args: dict, name: str = "echo_text", call_id: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_atomic_tool_shows_the_summary_and_keeps_the_out_off_the_message():
    seen = []
    ctx = object()

    def run(ctx_, args):
        seen.append(ctx_)
        return echo(ctx_, args)

    tool = AtomicTool(atomic(run), ctx)
    assert (tool.name, tool.description, tool.args_schema) == (
        "echo_text", "Say the text back.", EchoArgs
    )

    result = tool.invoke(call({"text": "hello"}))

    assert isinstance(result, ToolMessage)
    assert (result.content, result.status, result.artifact) == ("Echo: hello", "success", None)
    assert seen == [ctx]


def test_atomic_tool_call_returns_the_typed_out():
    out = AtomicTool(atomic(echo), None).call(text="hello")
    assert isinstance(out, EchoOut)
    assert (out.summary, out.length) == ("Echo: hello", 5)


def test_a_step_error_reaches_the_llm_as_code_and_message():
    def run(ctx, args):
        raise StepError("NOT_FOUND", "No one is in view.")

    tool = AtomicTool(atomic(run), None)

    result = tool.invoke(call({"text": "hello"}))
    assert (result.content, result.status) == ("NOT_FOUND: No one is in view.", "error")
    # Python callers get the error itself.
    with pytest.raises(StepError) as raised:
        tool.call(text="hello")
    assert raised.value.code == "NOT_FOUND"


def test_an_unexpected_exception_becomes_internal_and_is_logged(caplog):
    def run(ctx, args):
        raise RuntimeError("boom")

    tool = AtomicTool(atomic(run), None)

    with caplog.at_level(logging.ERROR, logger="walkie_agent.toolbox.atomic"):
        result = tool.invoke(call({"text": "hello"}))

    assert (result.content, result.status) == ("INTERNAL: RuntimeError: boom", "error")
    assert any(record.exc_info for record in caplog.records)
    with pytest.raises(StepError) as raised:
        tool.call(text="hello")
    assert raised.value.code == "INTERNAL"


@pytest.mark.parametrize(
    "returned", ["Echo: hello", ToolOutput(summary="Echo: hello")], ids=["str", "base_output"]
)
def test_a_wrong_return_type_is_internal(returned):
    tool = AtomicTool(atomic(lambda ctx, args: returned), None)

    result = tool.invoke(call({"text": "hello"}))

    assert result.status == "error"
    assert result.content == (
        f"INTERNAL: echo_text returned {type(returned).__name__} instead of EchoOut."
    )


def test_bad_arguments_raise_a_validation_error_like_other_tools():
    tool = AtomicTool(atomic(echo), None)
    with pytest.raises(ValidationError):
        tool.invoke(call({"txt": "hello"}))
    with pytest.raises(ValidationError):
        tool.call(text=3)


# -- HandleStore -----------------------------------------------------------------


class World:
    """A fake robot pose and clock for a HandleStore."""

    def __init__(self) -> None:
        self.pose: Pose2D | None = Pose2D(1.0, 2.0, 0.0)
        self.time = 1000.0
        self.pose_calls = 0

    def pose_provider(self) -> Pose2D | None:
        self.pose_calls += 1
        return self.pose

    def clock(self) -> float:
        return self.time

    def store(self, ids=(0x51C0, 0x09AB, 0x1234, 0x5678), max_sets: int = 64) -> HandleStore:
        source = iter(ids)
        return HandleStore(self.pose_provider, self.clock, max_sets=max_sets,
                           id_source=lambda: next(source))


def error_of(fn, *args, **kwargs) -> StepError:
    with pytest.raises(StepError) as raised:
        fn(*args, **kwargs)
    return raised.value


def test_put_set_names_the_set_and_its_members():
    world = World()
    store = world.store()

    people = store.put_set("person", ["Ana", "Ben", "Cy"])

    assert (people.id, people.kind) == ("per_51c0", "person")
    assert people.member_ids == ("per_51c0.0", "per_51c0.1", "per_51c0.2")
    ben = store.get("per_51c0.1", kind="person")
    assert (ben.id, ben.kind, ben.payload) == ("per_51c0.1", "person", "Ben")
    assert (ben.pose, ben.captured_at) == (Pose2D(1.0, 2.0, 0.0), 1000.0)
    assert [h.payload for h in store.get_set("per_51c0", kind="person")] == ["Ana", "Ben", "Cy"]
    # Without a kind, any live handle is returned.
    assert store.get(" per_51c0.2 ").payload == "Cy"


def test_set_ids_are_unique_among_live_sets_of_every_kind():
    world = World()
    store = world.store(ids=(0x51C0, 0x51C0, 0x09AB))

    store.put_set("person", ["Ana"])
    seats = store.put_set("seat", ["sofa"], pose=Pose2D(1.0, 2.0, 0.0))

    assert seats.id == "seat_09ab"


def test_an_explicit_pose_is_used_instead_of_the_provider():
    world = World()
    store = world.store()

    store.put_set("seat", ["sofa"], pose=Pose2D(1.0, 2.0, 0.0))

    assert world.pose_calls == 0


def test_a_handle_of_another_kind_is_not_found():
    world = World()
    store = world.store(ids=(0x09AB,))
    store.put_set("seat", ["left", "middle", "right"])

    err = error_of(store.get, "seat_09ab.2", kind="person")

    assert err.code == "NOT_FOUND"
    assert err.message == (
        "seat_09ab.2 is a seat, not a person. Person handles come from find_people."
    )
    assert error_of(store.get_set, "seat_09ab", kind="person").code == "NOT_FOUND"


def test_a_set_holds_at_most_max_set_members():
    world = World()
    store = world.store()
    limit = handles_module.MAX_SET_MEMBERS

    # The last member of a full set can still be looked up.
    full = store.put_set("note", ["milk"] * limit)
    assert full.member_ids[-1] == f"{full.id}.{limit - 1}"
    assert store.get(full.member_ids[-1]).payload == "milk"
    with pytest.raises(ValueError, match=f"at most {limit} members"):
        store.put_set("note", ["milk"] * (limit + 1))


def test_a_set_is_the_payloads_as_put_not_as_they_change_later():
    world = World()
    payloads = ["milk"] * handles_module.MAX_SET_MEMBERS

    # The caller's list grows while the pose provider runs, as it may while
    # it waits for TF. The set keeps what was passed, so every id resolves.
    def growing_pose() -> Pose2D:
        payloads.append("tea")
        return world.pose_provider()

    store = HandleStore(growing_pose, world.clock, id_source=lambda: 0x51C0)
    full = store.put_set("note", payloads)

    assert len(full.member_ids) == handles_module.MAX_SET_MEMBERS
    assert all(store.get(member).payload == "milk" for member in full.member_ids)


@pytest.mark.parametrize(
    "handle_id",
    # The last is longer than int() converts: still NOT_FOUND, not a ValueError.
    ["per_ffff.0", "per_51c0.3", "Ana", "per_51c0", "", "per_51c0." + "9" * 4301],
)
def test_an_unknown_or_malformed_handle_is_not_found(handle_id):
    world = World()
    store = world.store()
    store.put_set("person", ["Ana", "Ben", "Cy"])

    err = error_of(store.get, handle_id, kind="person")

    assert err.code == "NOT_FOUND"
    assert err.message == (
        f"{handle_id} is not a known person handle. Person handles come from find_people."
    )
    assert error_of(store.get, handle_id).message == f"{handle_id} is not a known handle."


def test_a_handle_goes_stale_once_the_base_moves_too_far():
    world = World()
    store = world.store()
    store.put_set("person", ["Ana"])

    world.pose = Pose2D(1.0, 2.49, 0.0)  # within the 0.5 m allowed
    assert store.get("per_51c0.0").payload == "Ana"

    world.pose = Pose2D(1.0, 2.51, 0.0)
    err = error_of(store.get, "per_51c0.0", kind="person")
    assert err.code == "STALE"
    assert err.message == (
        "per_51c0.0 is stale: the robot has moved 0.51 m since it was captured, more than "
        "the 0.5 m allowed. Person handles come from find_people."
    )
    assert err.data["handle"] == "per_51c0.0"


def test_rotation_is_compared_with_wrap_around():
    world = World()
    world.pose = Pose2D(0.0, 0.0, math.radians(179.0))
    store = world.store(ids=(0x09AB,))
    store.put_set("seat", ["sofa"])

    world.pose = Pose2D(0.0, 0.0, math.radians(-179.0))  # 2 degrees on
    assert store.get("seat_09ab.0", kind="seat").payload == "sofa"

    world.pose = Pose2D(0.0, 0.0, math.radians(-176.0))  # 5 degrees on
    err = error_of(store.get, "seat_09ab.0", kind="seat")
    assert err.code == "STALE"
    assert "turned 5 degrees since it was captured, more than the 3 degrees allowed." in (
        err.message
    )


def test_a_handle_goes_stale_with_age():
    world = World()
    store = world.store()
    store.put_set("person", ["Ana"])

    world.time += 60.0
    assert store.get("per_51c0.0").payload == "Ana"
    world.time += 1.0
    err = error_of(store.get_set, "per_51c0")
    assert err.code == "STALE"
    assert err.message == (
        "per_51c0 is stale: it was captured 61 s ago, and a person handle lasts 60 s. "
        "Person handles come from find_people."
    )


def test_an_unknown_pose_cannot_check_motion():
    world = World()
    store = world.store()

    world.pose = None
    store.put_set("person", ["Ana"])
    # Unknown at capture: the answer is UNAVAILABLE without asking TF again.
    world.pose = Pose2D(0.0, 0.0, 0.0)
    calls = world.pose_calls
    err = error_of(store.get, "per_51c0.0")
    assert (err.code, err.message) == (
        "UNAVAILABLE", "The robot's position is unknown, so the handle cannot be checked."
    )
    assert world.pose_calls == calls

    world.pose = Pose2D(0.0, 0.0, 0.0)
    store.put_set("seat", ["sofa"])
    world.pose = None
    assert error_of(store.get, "seat_09ab.0").code == "UNAVAILABLE"

    # A kind without motion limits never needs the pose.
    store.put_set("note", ["milk"])
    calls = world.pose_calls
    assert store.get("note_1234.0").payload == "milk"
    assert world.pose_calls == calls


def test_the_pose_provider_is_called_without_the_lock():
    held: list[bool] = []
    store = HandleStore(lambda: held.append(store._lock.locked()) or Pose2D(0.0, 0.0, 0.0))

    store.put_set("person", ["Ana"])
    store.get_set(store.put_set("person", ["Ben"]).id)

    assert held == [False, False, False]


def test_the_oldest_set_is_dropped_past_max_sets():
    world = World()
    store = world.store(max_sets=2)
    first = store.put_set("person", ["Ana"])
    second = store.put_set("person", ["Ben"])
    third = store.put_set("person", ["Cy"])

    assert error_of(store.get_set, first.id).code == "NOT_FOUND"
    assert [store.get_set(s.id)[0].payload for s in (second, third)] == ["Ben", "Cy"]


def test_clear_forgets_every_set():
    world = World()
    store = world.store()
    people = store.put_set("person", ["Ana"])

    store.clear()

    assert error_of(store.get_set, people.id).code == "NOT_FOUND"


def test_peek_resolves_handles_and_sets_without_any_check():
    world = World()
    store = world.store()
    people = store.put_set("person", ["Ana", "Ben"])

    # Stale by age and by motion: get() refuses, peek() still answers.
    world.time += 3600.0
    world.pose = Pose2D(9.0, 9.0, 0.0)
    assert error_of(store.get, "per_51c0.1").code == "STALE"
    calls = world.pose_calls
    assert [(h.id, h.kind, h.payload) for h in store.peek(" per_51c0.1 ")] == [
        ("per_51c0.1", "person", "Ben")
    ]
    assert [h.payload for h in store.peek(people.id)] == ["Ana", "Ben"]
    # It never asks for the pose, so it never waits on TF.
    assert world.pose_calls == calls
    world.pose = None
    assert [h.payload for h in store.peek("per_51c0.0")] == ["Ana"]
    for unknown in ("per_51c0.2", "per_ffff", "per_ffff.0", "Ana", "", "per_51c0." + "9" * 4301):
        assert store.peek(unknown) == []


def test_kinds_are_registered_once_with_their_own_prefix():
    register_kind("person", PERSON)  # the same policy again does nothing
    with pytest.raises(ValueError, match="already registered"):
        register_kind("person", SEAT)
    with pytest.raises(ValueError, match="already used"):
        register_kind("chair", KindPolicy("seat", None, None, None))
    with pytest.raises(ValueError, match="prefix"):
        register_kind("chair", KindPolicy("Chair_1", None, None, None))
    with pytest.raises(ValueError, match="Unknown handle kind"):
        World().store().put_set("chair", ["stool"])
    # Objects are registered by default, with 5 cm, 3 degrees and 10 minutes.
    assert handles_module.kind_policy("object") == KindPolicy("obj", 0.05, 3.0, 600.0)


# -- The registry ----------------------------------------------------------------


def atomic_source(
    name: str,
    *,
    meta_name: str | None = None,
    requires: tuple[str, ...] = (),
    args_base: str = "BaseModel",
    out_base: str = "ToolOutput",
    run_params: str = "ctx, args",
    example: str = '{"room": "kitchen"}',
) -> str:
    return textwrap.dedent(
        f'''\
        """Count the chairs in view."""
        from pydantic import BaseModel
        from walkie_agent.toolbox import Effect, ToolMeta, ToolOutput

        META = ToolMeta(
            name="{meta_name or name}",
            description="Count the chairs in view.",
            effects=frozenset({{Effect.P}}),
            questions=("How many chairs are there?", "Count the chairs.", "Is there a seat?"),
            examples=({example},),
            requires={requires!r},
        )


        class Args({args_base}):
            room: str = "here"


        class Out({out_base}):
            count: int = 3


        def run({run_params}):
            return Out(summary="3 chairs.")
        '''
    )


@pytest.fixture
def make_package(tmp_path, monkeypatch):
    """Writes a package of atomics under tmp_path and returns its name."""
    names: list[str] = []
    monkeypatch.syspath_prepend(str(tmp_path))

    def _make(files: dict[str, str]) -> str:
        package = f"atomics_{uuid.uuid4().hex[:8]}"
        root = tmp_path / package
        root.mkdir()
        (root / "__init__.py").write_text('"""Test atomics."""\n')
        for module, source in files.items():
            (root / f"{module}.py").write_text(source)
        # The import system caches directory listings; a package written
        # after the path was added might otherwise not be found.
        importlib.invalidate_caches()
        names.append(package)
        return package

    yield _make
    for module in [m for m in sys.modules if m.split(".")[0] in names]:
        del sys.modules[module]


def test_discover_loads_sound_atomics_and_skips_private_modules(make_package):
    package = make_package(
        {
            "count_chairs": atomic_source("count_chairs"),
            "lift_box": atomic_source("lift_box", requires=("arm",)),
            "_helper": "raise RuntimeError('a helper is never imported as a tool')\n",
        }
    )

    atomics = discover(package)

    assert [a.META.name for a in atomics] == ["count_chairs", "lift_box"]


def test_discover_reports_every_broken_atomic_at_once(make_package):
    package = make_package(
        {
            "count_chairs": atomic_source("count_chairs"),
            "no_meta": atomic_source("no_meta") + "\ndel META\n",
            "wrong_name": atomic_source("wrong_name", meta_name="other_name"),
            "plain_args": atomic_source("plain_args", args_base="object"),
            "plain_out": atomic_source("plain_out", out_base="BaseModel"),
            "one_param": atomic_source("one_param", run_params="ctx"),
            "bad_example": atomic_source("bad_example", example='{"room": 3}'),
            "two_problems": atomic_source("two_problems", out_base="object", run_params=""),
            "broken_import": "raise ImportError('no module named walkie_magic')\n",
            "async_run": atomic_source("async_run").replace("def run(", "async def run("),
            # Pydantic turns only ValueError and AssertionError into a
            # ValidationError; a TypeError from a validator passes through.
            "odd_validator": atomic_source("odd_validator") + textwrap.dedent(
                """
                from pydantic import field_validator


                class Args(BaseModel):
                    room: str = "here"

                    @field_validator("room")
                    @classmethod
                    def _odd(cls, value):
                        raise TypeError("rooms are odd")
                """
            ),
        }
    )

    with pytest.raises(AtomicLoadError) as raised:
        discover(package)

    problems = [p.removeprefix(f"{package}.") for p in raised.value.problems]
    assert problems == [
        "async_run: run is async; AtomicTool calls it synchronously.",
        "bad_example: example 1 does not fit Args: room: Input should be a valid string",
        "broken_import: import failed: ImportError: no module named walkie_magic",
        "no_meta: META is missing or not a ToolMeta.",
        "odd_validator: example 1 does not fit Args: TypeError: rooms are odd",
        "one_param: run cannot be called as run(ctx, args).",
        "plain_args: Args is missing or not a pydantic BaseModel subclass.",
        "plain_out: Out is missing or not a ToolOutput subclass.",
        "two_problems: Out is missing or not a ToolOutput subclass.",
        "two_problems: run cannot be called as run(ctx, args).",
        "wrong_name: META.name is 'other_name', but the module is named 'wrong_name'.",
    ]
    assert str(raised.value).startswith("11 problems with the atomic tools:\n- ")


def test_discover_reports_duplicate_names(make_package):
    package = make_package(
        {"alpha": atomic_source("alpha", meta_name="beta"), "beta": atomic_source("beta")}
    )

    with pytest.raises(AtomicLoadError) as raised:
        discover(package)

    assert raised.value.problems == [
        f"{package}.alpha: META.name is 'beta', but the module is named 'alpha'.",
        f"The tool name 'beta' is used by {package}.alpha, {package}.beta.",
    ]


def test_atomics_load_only_with_their_backends(make_package, caplog):
    atomics = discover(
        make_package(
            {
                "count_chairs": atomic_source("count_chairs"),
                "lift_box": atomic_source("lift_box", requires=("arm", "gripper")),
            }
        )
    )

    with caplog.at_level(logging.INFO, logger="walkie_agent.toolbox.registry"):
        tools = build_atomic_tools(ToolContext(None, backends={"gripper": object()}), atomics)
    assert [t.name for t in tools] == ["count_chairs"]
    assert "Skipped the atomic tool lift_box: missing backends arm." in caplog.messages

    ctx = ToolContext(None, backends={"arm": object(), "gripper": object()})
    tools = build_atomic_tools(ctx, atomics)
    assert [t.name for t in tools] == ["count_chairs", "lift_box"]
    assert all(isinstance(t, AtomicTool) and t.ctx is ctx for t in tools)
    assert tools[1].invoke(call({}, name="lift_box")).content == "3 chairs."


def test_the_shipped_package_has_the_people_seat_arm_and_gaze_atomics():
    """vision alone loads find_people and find_seats; people_memory adds two; arm adds four;
    gaze adds stop_looking, and look_at with vision."""
    atomics = discover()
    vision_only = ToolContext(None, backends={"vision": object()})
    both = ToolContext(None, backends={"vision": object(), "people_memory": object()})
    arm_only = ToolContext(None, backends={"arm": object()})
    gaze_only = ToolContext(None, backends={"gaze": object()})
    vision_and_gaze = ToolContext(None, backends={"vision": object(), "gaze": object()})
    everything = ToolContext(
        None,
        backends={
            "vision": object(),
            "people_memory": object(),
            "arm": object(),
            "gaze": object(),
        },
    )

    assert [atomic.META.name for atomic in atomics] == [
        "find_people",
        "find_seats",
        "get_arm_state",
        "list_arm_poses",
        "look_at",
        "move_arm_to_pose",
        "recognize_person",
        "remember_person",
        "set_gripper",
        "stop_looking",
    ]
    assert [tool.name for tool in build_atomic_tools(vision_only, atomics)] == [
        "find_people",
        "find_seats",
    ]
    assert [tool.name for tool in build_atomic_tools(both, atomics)] == [
        "find_people",
        "find_seats",
        "recognize_person",
        "remember_person",
    ]
    assert [tool.name for tool in build_atomic_tools(arm_only, atomics)] == [
        "get_arm_state",
        "list_arm_poses",
        "move_arm_to_pose",
        "set_gripper",
    ]
    # look_at also needs vision: every handle it takes comes from a vision atomic.
    assert [tool.name for tool in build_atomic_tools(gaze_only, atomics)] == ["stop_looking"]
    assert [tool.name for tool in build_atomic_tools(vision_and_gaze, atomics)] == [
        "find_people",
        "find_seats",
        "look_at",
        "stop_looking",
    ]
    assert len(build_atomic_tools(everything, atomics)) == len(atomics)


# -- ToolContext -----------------------------------------------------------------


def test_backend_is_unavailable_when_missing():
    arm = object()
    ctx = ToolContext(None, backends={"arm": arm})

    assert ctx.backend("arm") is arm
    assert (ctx.has_backend("arm"), ctx.has_backend("vision")) == (True, False)
    err = error_of(ctx.backend, "vision")
    assert (err.code, err.message) == ("UNAVAILABLE", "The vision backend is not available.")


class IdleExecutor:
    """Takes the tasks rclpy schedules on an executor, and never runs them."""

    def __init__(self) -> None:
        self.scheduled = 0

    def create_task(self, _callback, *_args) -> None:
        self.scheduled += 1


def test_wait_returns_the_result_or_times_out():
    ctx = ToolContext(None)
    done: Future = Future()
    done.set_result("arrived")
    assert ctx.wait(done, 1.0) == "arrived"

    later: Future = Future()
    threading.Timer(0.1, later.set_result, ["arrived late"]).start()
    assert ctx.wait(later, 5.0) == "arrived late"

    # A future of an executor that is not spinning: its done callbacks are
    # only scheduled, so wait() has to look at the future itself.
    idle = IdleExecutor()
    owned: Future = Future(executor=idle)
    owned.set_result("arrived")
    ctx.cancel.set()
    assert ctx.wait(owned, 0.1) == "arrived"
    assert idle.scheduled == 1
    ctx.cancel.clear()

    err = error_of(ctx.wait, Future(), 0.1)
    assert (err.code, err.message) == ("TIMEOUT", "The call did not finish within 0.1 s.")
    err = error_of(ctx.wait, Future(), 0.1, what="The arm")
    assert err.message == "The arm did not finish within 0.1 s."


def test_the_tools_wait_for_future_sees_a_future_of_an_idle_executor():
    # tools/waiters.py, which the navigation and memory tools use, has the
    # same reason as ctx.wait() to look at the future itself.
    # The future holds its executor by weak reference, so the test keeps it
    # alive; without one, rclpy runs the callbacks at once.
    idle = IdleExecutor()
    owned: Future = Future(executor=idle)
    owned.set_result("arrived")

    started = time.monotonic()
    assert wait_for_future(owned, 2.0) == "arrived"
    assert time.monotonic() - started < 1.0
    assert idle.scheduled == 1
    assert wait_for_future(Future(), 0.1) is None


def test_wait_ends_once_cancel_is_set():
    ctx = ToolContext(None)
    threading.Timer(0.1, ctx.cancel.set).start()

    started = time.monotonic()
    err = error_of(ctx.wait, Future(), 5.0)

    assert (err.code, err.message) == ("CANCELLED", "A stop or a reset interrupted the call.")
    assert time.monotonic() - started < 1.0
    assert error_of(ctx.check_cancelled).code == "CANCELLED"
    ctx.cancel.clear()
    ctx.check_cancelled()


def _ignore(_msg) -> None:
    pass


ENTITIES = {
    "subscription": lambda ctx, name, **kw: ctx.subscription(String, name, _ignore, 10, **kw),
    "publisher": lambda ctx, name, **kw: ctx.publisher(String, name, 10, **kw),
    "service_client": lambda ctx, name, **kw: ctx.service_client(Trigger, name, **kw),
    "action_client": lambda ctx, name, **kw: ctx.action_client(NavigateToPose, name, **kw),
}


@pytest.mark.parametrize("kind", ENTITIES)
def test_forbidden_names_need_an_explicit_exception(connector, kind):
    ctx = ToolContext(connector)
    make = ENTITIES[kind]

    with pytest.raises(ValueError, match="/cmd_vel is in FORBIDDEN_NAMES"):
        make(ctx, "/cmd_vel")
    # Allowing one name does not allow another.
    with pytest.raises(ValueError, match="/walkie/reset is in FORBIDDEN_NAMES"):
        make(ctx, "/walkie/reset", allow_forbidden=("/cmd_vel",))
    # A relative name is checked as the node resolves it, in the root namespace.
    with pytest.raises(ValueError, match="/cmd_vel is in FORBIDDEN_NAMES"):
        make(ctx, "cmd_vel")
    allowed = make(ctx, "/cmd_vel", allow_forbidden=("/cmd_vel",))
    assert allowed is not None
    assert make(ctx, "cmd_vel", allow_forbidden=("/cmd_vel",)) is allowed


@pytest.mark.parametrize("kind", ENTITIES)
def test_entities_are_created_once_per_name(connector, kind):
    ctx = ToolContext(connector)
    make = ENTITIES[kind]

    first = make(ctx, "/walkie_test/entity")

    assert make(ctx, "/walkie_test/entity") is first
    assert make(ctx, "walkie_test/entity") is first  # the same name, resolved
    assert make(ctx, "/walkie_test/other") is not first


def test_an_entity_asked_for_with_other_arguments_is_an_error(connector):
    ctx = ToolContext(connector)
    ctx.subscription(String, "/walkie_test/words", _ignore, 10)

    ctx.publisher(String, "/walkie_test/count", 10)

    with pytest.raises(ValueError, match="already exists with other arguments"):
        ctx.subscription(String, "/walkie_test/words", lambda _msg: None, 10)
    with pytest.raises(ValueError, match="already exists with other arguments"):
        ctx.publisher(TransformStamped, "/walkie_test/count", 10)
    assert connector.node.count_subscribers("/walkie_test/words") == 1


def test_robot_pose_without_tf_is_unavailable(connector):
    ctx = ToolContext(connector)

    err = error_of(ctx.robot_pose, timeout_s=0.3)

    assert err.code == "UNAVAILABLE"
    assert err.message.startswith("The robot's position (map -> base_footprint) is unknown.")
    assert err.message.endswith(
        "No node publishes /tf: either the robot is not running, or the "
        "agent uses a different ROS_DOMAIN_ID or Zenoh router than the robot."
    )


def test_robot_pose_reads_tf_and_feeds_the_handle_store(connector):
    static = StaticTransformBroadcaster(connector.node)
    t = TransformStamped()
    t.header.frame_id = "map"
    t.child_frame_id = "base_footprint"
    t.transform.translation.x = 1.5
    t.transform.translation.y = -0.5
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, math.radians(90.0))
    t.transform.rotation.x, t.transform.rotation.y = qx, qy
    t.transform.rotation.z, t.transform.rotation.w = qz, qw
    static.sendTransform(t)
    ctx = ToolContext(connector)

    pose = ctx.robot_pose(timeout_s=3.0)

    assert (pose.x, pose.y, pose.yaw) == pytest.approx((1.5, -0.5, math.radians(90.0)))
    mugs = ctx.handles.put_set("object", ["mug"])
    assert ctx.handles.get(mugs.member_ids[0], kind="object").pose == pose
    assert isinstance(ctx.now(), Time)


def unavailable(name: str) -> tuple[str, str]:
    return ("UNAVAILABLE", f"The {name} capability is not available.")


def test_base_and_speech_have_no_default_whatever_the_backends():
    # Their ROS implementations come in PR B (capabilities.md).
    ctx = ToolContext(None, backends={"arm": object(), "gaze": object(), "vision": object()})

    for name in ("base", "speech"):
        err = error_of(getattr, ctx, name)
        assert (err.code, err.message) == unavailable(name)


def test_a_capability_without_its_backend_or_from_a_none_factory_is_unavailable():
    bare = ToolContext(None)
    for name in ("arms", "gaze", "vision"):
        err = error_of(getattr, bare, name)
        assert (err.code, err.message) == unavailable(name)

    ctx = ToolContext(None, capabilities={"pose": lambda _ctx: None})
    err = error_of(getattr, ctx, "pose")
    assert (err.code, err.message) == unavailable("pose")

    ctx, robot = fake_ctx(without=("gaze", "speech"))
    for name in ("gaze", "speech"):
        err = error_of(getattr, ctx, name)
        assert (err.code, err.message) == unavailable(name)
    assert (ctx.base, ctx.arms) == (robot.base, robot.arms)


def test_a_capability_factory_runs_once_with_the_ctx():
    built: list[tuple[str, ToolContext]] = []
    pose = object()

    def factory(name: str, capability: object):
        def build(ctx: ToolContext) -> object:
            built.append((name, ctx))
            return capability

        return build

    ctx = ToolContext(
        None, capabilities={"pose": factory("pose", pose), "base": factory("base", None)}
    )

    assert ctx.pose is pose
    assert ctx.pose is pose
    # None is kept too: the factory is not asked again.
    assert error_of(getattr, ctx, "base").code == "UNAVAILABLE"
    assert error_of(getattr, ctx, "base").code == "UNAVAILABLE"
    assert built == [("pose", ctx), ("base", ctx)]


def test_a_capability_factory_may_read_another_but_not_its_own():
    pose = object()
    ctx = ToolContext(None, capabilities={"pose": lambda _ctx: pose, "base": lambda c: c.pose})

    # The base factory reads ctx.pose while the ctx builds ctx.base.
    assert ctx.base is pose

    looped = ToolContext(
        None, capabilities={"base": lambda c: c.speech, "speech": lambda c: c.base}
    )
    with pytest.raises(
        RuntimeError, match=re.escape("The base capability's factory reads ctx.base.")
    ):
        _ = looped.base
    # The failed build is not kept, so the next read raises the same way.
    with pytest.raises(RuntimeError, match="reads ctx.base"):
        _ = looped.base


def test_an_unknown_capability_name_is_an_error():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unknown capabilities ['legs']; a ctx has base, speech, pose, arms, gaze, vision."
        ),
    ):
        ToolContext(None, capabilities={"legs": lambda _ctx: None})


def test_every_verb_of_fake_ctx_is_cancelled_before_it_sends_anything():
    ctx, robot = fake_ctx(locations={"kitchen": (2.0, 1.0, 0.0)})
    robot.vision.show(people=[FakePerson()], objects=[FakeObject("cup")])
    seat = robot.vision.seat(0.5)
    ctx.cancel.set()
    verbs = {
        "base.go_to": lambda: ctx.base.go_to("kitchen"),
        "base.move_relative": lambda: ctx.base.move_relative(forward_m=1.0),
        "base.turn_by": lambda: ctx.base.turn_by(90.0),
        "base.turn_to": lambda: ctx.base.turn_to(1.0),
        "speech.say": lambda: ctx.speech.say("Hello."),
        "pose.fresh": lambda: ctx.pose.fresh(),
        "arms.move_to": lambda: ctx.arms.move_to("hands_up"),
        "arms.gripper": lambda: ctx.arms.gripper("left", "close"),
        "gaze.look_at": lambda: ctx.gaze.look_at(seat),
        "vision.people": lambda: ctx.vision.people(max_age_s=1.0),
        "vision.objects": lambda: ctx.vision.objects(max_age_s=1.0),
    }

    for name, verb in verbs.items():
        assert (name, error_of(verb).code) == (name, "CANCELLED")

    assert robot.calls == list(verbs)
    assert robot.base.goals == [] and robot.speech.said == []
    assert robot.arms.goals == [] and robot.arms.gripper_goals == []
    steps = robot.base.steps + robot.arms.steps + robot.gaze.steps
    assert not {"base.goal", "arms.goal", "gripper.goal", "gaze.start"} & set(steps)
    assert ctx.gaze.status().target is None


def test_the_fake_clock_runs_each_wait_to_its_end_at_once():
    clock = FakeClock()
    ctx = ToolContext(None, clock=clock)

    started = time.monotonic()
    err = error_of(ctx.wait, Future(), 30.0)
    assert (err.code, err.message) == ("TIMEOUT", "The call did not finish within 30 s.")
    assert clock.monotonic() == pytest.approx(1030.0)
    assert time.monotonic() - started < 1.0

    started = time.monotonic()
    ctx.sleep(60.0)
    assert clock.monotonic() == pytest.approx(1090.0)
    assert time.monotonic() - started < 1.0

    started = time.monotonic()
    err = error_of(ctx.wait_until, lambda: False, 10.0, "The turn")
    assert (err.code, err.message) == ("TIMEOUT", "The turn did not finish within 10 s.")
    assert clock.monotonic() == pytest.approx(1100.0)
    assert time.monotonic() - started < 1.0


def test_a_stop_ends_sleep_and_wait_until_on_the_real_clock():
    ctx = ToolContext(None)
    waits = {
        "sleep": lambda: ctx.sleep(5.0),
        "wait_until": lambda: ctx.wait_until(lambda: False, 5.0),
    }

    for name, wait in waits.items():
        ctx.cancel.clear()
        timer = threading.Timer(0.1, ctx.cancel.set)
        timer.start()
        started = time.monotonic()
        err = error_of(wait)
        assert (name, err.code, err.message) == (
            name,
            "CANCELLED",
            "A stop or a reset interrupted the call.",
        )
        assert time.monotonic() - started < 1.0
        timer.join()


@pytest.mark.parametrize("timeout_s", [math.nan, -1.0, math.inf])
def test_a_wait_refuses_a_timeout_that_is_not_a_finite_duration(timeout_s):
    ctx = ToolContext(None)
    # Each error names the argument it refuses.
    waits = (
        ("timeout_s", lambda: ctx.wait(Future(), timeout_s)),
        ("seconds", lambda: ctx.sleep(timeout_s)),
        ("timeout_s", lambda: ctx.wait_until(lambda: True, timeout_s)),
    )

    for name, wait in waits:
        with pytest.raises(ValueError, match=f"^{name} must be a finite number of seconds"):
            wait()


def test_now_ns_follows_the_clock_and_in_simulation_can_be_injected():
    ctx, robot = fake_ctx(in_simulation=True)

    assert ctx.now_ns() == robot.clock.now_ns() == 1_000_000_000_000
    robot.clock.advance(1.5)
    assert ctx.now_ns() == ctx.now().nanoseconds == 1_001_500_000_000
    assert ctx.in_simulation is True
    assert ToolContext(None, in_simulation=False).in_simulation is False


def test_now_ns_and_in_simulation_read_the_node(connector):
    ctx = ToolContext(connector)
    clock = connector.node.get_clock()

    before = clock.now().nanoseconds
    assert before <= ctx.now_ns() <= clock.now().nanoseconds
    # The node's use_sim_time parameter, False in the tests.
    assert ctx.in_simulation is False
    assert ToolContext(connector, in_simulation=True).in_simulation is True


# -- Toolbox ---------------------------------------------------------------------


def test_interrupt_sets_cancel_before_the_listeners_and_survives_a_failing_one(caplog):
    toolbox = Toolbox(None)
    heard: list[str] = []

    def failing() -> None:
        heard.append("failing")
        raise RuntimeError("listener broke")

    toolbox.on_interrupt(lambda: heard.append(f"first, cancel {toolbox.ctx.cancel.is_set()}"))
    toolbox.on_interrupt(failing)
    toolbox.on_interrupt(lambda: heard.append("last"))

    with caplog.at_level(logging.ERROR, logger="walkie_agent.toolbox"):
        toolbox.interrupt()

    assert heard == ["first, cancel True", "failing", "last"]
    assert "A toolbox interrupt listener failed." in caplog.messages

    toolbox.begin_task()
    assert not toolbox.ctx.cancel.is_set()


def test_an_interrupt_is_pending_until_its_listeners_have_run():
    toolbox = Toolbox(None)
    pending: list[bool] = []

    def failing() -> None:
        pending.append(toolbox.interrupt_pending())
        raise RuntimeError("listener broke")

    toolbox.on_interrupt(failing)
    assert not toolbox.interrupt_pending()

    toolbox.cancel()
    assert toolbox.interrupt_pending()
    # No run starts while it is pending, and the cancel stays set.
    assert toolbox.begin_task() is False
    assert toolbox.ctx.cancel.is_set()
    toolbox.cancel()  # a second interrupt before the first is delivered
    toolbox.notify_interrupt()
    assert toolbox.interrupt_pending()
    assert toolbox.begin_task() is False
    toolbox.notify_interrupt()

    # Pending while the listeners ran, and no longer once they are done,
    # even though the listener failed.
    assert pending == [True, True]
    assert not toolbox.interrupt_pending()
    assert toolbox.begin_task() is True
    assert not toolbox.ctx.cancel.is_set()


def test_new_task_clears_the_handles_and_tells_the_listeners():
    toolbox = Toolbox(None)
    told: list[int] = []
    toolbox.on_new_task(lambda: told.append(1))
    notes = toolbox.ctx.handles.put_set("note", ["milk"], pose=Pose2D(0.0, 0.0, 0.0))
    toolbox.cancel()

    toolbox.new_task()

    assert error_of(toolbox.ctx.handles.get_set, notes.id).code == "NOT_FOUND"
    assert told == [1]
    # A new task does not end a cancel; only begin_task does.
    assert toolbox.ctx.cancel.is_set()


def test_stop_heard_tells_its_listeners_and_cancels_nothing(caplog):
    toolbox = Toolbox(None)
    heard: list[str] = []

    def failing() -> None:
        heard.append("failing")
        raise RuntimeError("listener broke")

    toolbox.on_stop_heard(lambda: heard.append("first"))
    toolbox.on_stop_heard(failing)
    toolbox.on_stop_heard(lambda: heard.append("last"))
    toolbox.on_interrupt(lambda: heard.append("interrupt"))
    toolbox.on_new_task(lambda: heard.append("new task"))

    with caplog.at_level(logging.ERROR, logger="walkie_agent.toolbox"):
        toolbox.stop_heard()

    assert heard == ["first", "failing", "last"]
    assert "A toolbox stop-heard listener failed." in caplog.messages
    # Only an interrupt cancels the run and holds the next one back.
    assert not toolbox.ctx.cancel.is_set()
    assert not toolbox.interrupt_pending()
    assert toolbox.begin_task() is True


def test_toolbox_builds_its_tools_once(make_package):
    toolbox = Toolbox(None, package=make_package({"count_chairs": atomic_source("count_chairs")}))

    tools = toolbox.tools()

    assert [t.name for t in tools] == ["count_chairs"]
    assert toolbox.tools()[0] is tools[0]
    assert tools[0].ctx is toolbox.ctx


def test_build_walkie_tools_adds_the_atomics_after_the_other_tools(
    connector, locations, make_package
):
    package = make_package({"count_chairs": atomic_source("count_chairs")})
    toolbox = Toolbox(connector, package=package)

    names =[t.name for t in build_walkie_tools(connector, locations, toolbox=toolbox)]

    assert names[-1] == "count_chairs"
    assert names[:-1] == [t.name for t in build_walkie_tools(connector, locations)]


def test_build_walkie_tools_refuses_an_atomic_named_like_another_tool(
    connector, locations, make_package
):
    toolbox = Toolbox(connector, package=make_package({"speak": atomic_source("speak")}))

    with pytest.raises(ValueError, match="clash with existing tools: speak"):
        build_walkie_tools(connector, locations, toolbox=toolbox)
