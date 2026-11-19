"""The assembly and the process lifecycle of walkie_agent.main.

The ROS lifecycle and the model factories are patched and their calls inspected, so
nothing here needs a live graph, credentials or the network. Signals are real, sent to
this process, which is why `main` runs on the test's main thread. One test runs the real
entry point in a fresh interpreter, to see what the operator's terminal gets.
"""

import inspect
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from rai_interfaces.msg import HRIMessage
from rclpy.signals import SignalHandlerOptions
from ros_fakes import WATCHDOG_S, FakeActionClient, FakeConnector, FakeNode
from runtime_harness import ScriptedModel, ai, call, wait_until
from std_msgs.msg import Bool, Empty
from test_ros_vision import CAMERA_FRAME, person, with_vectors
from test_scene_file import camera_info
from walkie_vision_msgs.msg import PersonDetectionArray

import walkie_agent.main as main
import walkie_agent.prompts as prompts
from walkie_agent.arm.poses import ArmPoses, load_arm_poses
from walkie_agent.arm.results import ClientSnapshot, DriveCheck
from walkie_agent.arm.state import ArmsState, ArmState
from walkie_agent.capabilities import (
    Team,
    UnavailableArms,
    UnavailableFacts,
    UnavailableMemory,
    UnavailableObjectMemory,
    UnavailableVision,
    UnavailableWebSearch,
)
from walkie_agent.capabilities.facts import SystemFacts
from walkie_agent.capabilities.web_search import TavilySearch
from walkie_agent.chat_models import build_model
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import Location, load_locations
from walkie_agent.people_memory.store import InMemoryPeopleMemory
from walkie_agent.people_memory.track_memory import TrackMemory
from walkie_agent.perception import OBJECTS_TOPIC, PEOPLE_TOPIC
from walkie_agent.ros.arms import RosArms
from walkie_agent.ros.base import RosBase
from walkie_agent.ros.commander import READ_TIMEOUT_S, CommanderClient
from walkie_agent.ros.speech import RosSpeech
from walkie_agent.ros.vision import RosVision
from walkie_agent.runtime import Runner, ToolLog
from walkie_agent.runtime.task_templates import TaskTemplate
from walkie_agent.scripting import agent_tool
from walkie_agent.scripting.agent_tool import AgentScripts
from walkie_agent.scripting.sandbox import SandboxUnavailable
from walkie_agent.scripting.shelf import Shelf
from walkie_agent.tools.atomic import get_facts, list_places
from walkie_agent.tools.sets import (
    ARM_TOOLS,
    ARMS,
    DEFAULT,
    GPSR,
    HRI,
    OBJECT_MEMORY_TOOLS,
    PICK_TOOLS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DONE = AIMessage("Done.")
PLACES = {
    "kitchen": Location("kitchen", 3.5, 1.2, 1.57, "Kitchen counter."),
    "door": Location("door", 1.0, 0.0, 0.0),
}
ROBOT = "Walkie is a test robot with a unique marker 7f3a."
MIDDLEWARE_TOOLS = {"update_todos", "wait_for_person"}
TOOL_SET_LISTS = {"default": DEFAULT, "gpsr": GPSR, "hri": HRI, "arms": ARMS}
# The sets a run without --arms may use, and those a run with --arms may use.
SETS_WITHOUT_ARMS = ("default", "gpsr", "hri")
SETS_WITH_ARMS = ("hri", "arms")
FIXTURE_LOCATIONS = PROJECT_ROOT / "tests" / "fixtures" / "locations.toml"
MODEL_CONFIG = '[vendor]\ncomplex_model = "openai"\n[openai]\ncomplex_model = "m"\n'
TEAM_CONFIG = (
    '[team]\nname = "EIC"\ncountry = "Thailand"\naffiliation = "Chulalongkorn University"\n'
    'timezone = "Asia/Bangkok"\n'
)
POSES = load_arm_poses(
    PROJECT_ROOT / "config" / "arm_poses.toml", PROJECT_ROOT / "config" / "robot_model.toml"
)
HOME = ArmState(pose="home", travel_safe=True)
HANDS_UP = ArmState(pose="hands_up", travel_safe=False)


def tool_names(tool_set, arms=False, object_memory=False):
    """What a run binds: the set's atomics, less the atomics of a missing flag.

    The arm atomics need --arms, the object memory atomics --object-memory, and
    pick_object both.
    """
    atomics = {fn.__name__ for fn in TOOL_SET_LISTS[tool_set]}
    dropped = (set() if arms else ARM_TOOLS) | (set() if object_memory else OBJECT_MEMORY_TOOLS)
    if not (arms and object_memory):
        dropped |= PICK_TOOLS
    return (atomics - dropped) | MIDDLEWARE_TOOLS


def template(*todos, name="tour"):
    return TaskTemplate(name=name, task="Do it.", todos=list(todos))


@pytest.fixture
def nav(monkeypatch):
    """Gives main's RosBase a FakeActionClient. Holds the clients it made."""
    clients = []

    def factory(*args, **kwargs):
        clients.append(FakeActionClient(*args, **kwargs))
        return clients[-1]

    monkeypatch.setattr(main, "RosBase", partial(RosBase, client_factory=factory))
    return clients


class StandInCommander:
    """CommanderClient's constructor, and the part of the client that main and RosArms use.

    FakeNode makes no service clients, so the real client cannot be made over it.
    `state` is what every read of the arms finds, and `log` records the drive checks'
    bounds and the calls of stop, close and destroy, in order. close() calls `on_close`,
    which may change `closes` for the next close(), and returns what `closes` was when
    it was called. unresolved() gives `left` after a close() that returned False.
    """

    def __init__(self, connector, cancel, poses=None, **bounds):
        self.connector = connector
        self.cancel = cancel
        self.poses = poses
        self.bounds = bounds
        self.state = ArmsState(left=HOME, right=HOME)
        self.log = []
        self.closes = True
        self.left = ["The move of left_arm has not ended: a cancel was sent, but no result has come."]
        self.on_close = None
        # What the last close() returned; None before the first.
        self._closed = None

    @property
    def read_timeout_s(self):
        return self.bounds.get("read_timeout_s", READ_TIMEOUT_S)

    @property
    def last_state(self):
        return self.state

    def available(self):
        return True

    def read_arms(self):
        return self.state

    def snapshot(self):
        return ClientSnapshot(self.state, None, 0, None)

    def check_for_driving(self, timeout_s=None):
        self.log.append(("check", timeout_s))
        return DriveCheck(self.state)

    def stop(self):
        self.log.append("stop")
        self.cancel.cancel()

    def close(self, timeout_s):
        self.log.append("close")
        self._closed = self.closes
        if self.on_close is not None:
            self.on_close()
        return self._closed

    def unresolved(self):
        return list(self.left) if self._closed is False else []

    def destroy(self):
        self.log.append("destroy")


@pytest.fixture
def commanders(monkeypatch):
    """Gives main a StandInCommander in place of CommanderClient. Holds the ones it made."""
    made = []

    class Recording(StandInCommander):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(main, "CommanderClient", Recording)
    return made


@pytest.fixture
def build(nav):
    """assemble over a fake connector and a scripted model. Closes what it built."""
    apps = []

    def make(*script, then=DONE, connector=None, **overrides):
        model = ScriptedModel(script=list(script), then=then)
        arguments = {
            "connector": connector or FakeConnector(),
            "model": model,
            "config": {},
            "locations": PLACES,
            "templates": {},
            "robot_description": ROBOT,
            **overrides,
        }
        app = main.assemble(**arguments)
        apps.append(app)
        return SimpleNamespace(
            app=app, connector=arguments["connector"], model=arguments["model"], node=arguments["connector"].node
        )

    yield make
    for app in apps:
        assert app.close(WATCHDOG_S), f"could not close: {app.unresolved()}"


def run_one(built, text="Hello."):
    built.app.runner.submit(text)
    assert built.app.runner.wait_idle(WATCHDOG_S)


def topics(entities):
    return {entity.topic for entity in entities}


# -- assemble ------------------------------------------------------------------------


def test_assemble_uses_runtime_and_one_shared_token(build, monkeypatch):
    built_with = {}
    real_build_agent = main.build_agent

    def recording_build_agent(model, ctx, **kwargs):
        built_with.update(kwargs, ctx=ctx)
        return real_build_agent(model, ctx, **kwargs)

    monkeypatch.setattr(main, "build_agent", recording_build_agent)
    callback, tool_log, fallback = BaseCallbackHandler(), ToolLog(os.devnull), ScriptedModel(then=DONE)
    built = build(
        config={"agent": {"run_limit": 3, "deadline_s": 7.0}},
        callbacks=[callback],
        tool_log=tool_log,
        fallback=fallback,
    )
    app, node = built.app, built.node

    # The runtime's own parts, over the ROS wrappers, and no RAI agent host.
    assert not hasattr(main, "AgentRunner")
    assert isinstance(app.runner, Runner)
    assert app.runner.ctx is app.ctx
    assert isinstance(app.ctx.base, RosBase)
    assert isinstance(app.ctx.speech, RosSpeech)
    assert isinstance(app.ctx.vision, UnavailableVision)
    assert isinstance(app.ctx.arms, UnavailableArms)
    assert isinstance(app.ctx.facts, UnavailableFacts)

    # What config.toml and the caller chose reaches the runtime.
    assert built_with["ctx"] is app.ctx
    assert (built_with["run_limit"], built_with["deadline_s"]) == (3, 7.0)
    assert built_with["fallback"] is fallback
    assert built_with["tool_log"] is tool_log
    assert built_with["on_todos"] == app.bridge.on_todos
    assert app.runner._callbacks == [callback]
    assert app.runner._tool_log is tool_log

    # The publishers exist before the Runner starts, and the input opens only after it.
    assert topics(node.publishers) == {"/to_human", "/walkie/console", "/walkie/todos"}
    assert topics(node.subscriptions) == {"/voice_commands"}
    app.start()
    assert topics(node.subscriptions) == {
        "/voice_commands",
        "/from_human",
        "/walkie/task",
        "/walkie/reset",
    }

    # One token: every verb and the Runner see the same latch.
    app.ctx.cancel.cancel()
    with pytest.raises(StepError) as raised:
        app.ctx.base.places()
    assert raised.value.code is ErrorCode.CANCELLED
    with pytest.raises(StepError) as raised:
        app.ctx.vision.objects()
    assert raised.value.code is ErrorCode.CANCELLED
    app.ctx.speech.say("Stopping.")  # speech ignores the latch on purpose
    run_one(built)  # the next task releases the latch that the verbs read
    assert not app.ctx.cancel.cancelled
    assert set(app.ctx.base.places()) == set(PLACES)

    # The reply reaches the console, and list_places runs over the RosBase.
    console = next(p for p in node.publishers if p.topic == "/walkie/console")
    assert [message.text for message in console.published] == ["Done."]
    assert list_places(app.ctx) == [
        {"name": "kitchen", "description": "Kitchen counter."},
        {"name": "door", "description": ""},
    ]


def test_assemble_puts_the_facts_it_is_given_on_ctx(build):
    facts = SystemFacts(Team("EIC", "Thailand", "Chulalongkorn University"), "Asia/Bangkok", clock=lambda: 0.0)
    built = build(facts=facts)
    assert built.app.ctx.facts is facts
    assert get_facts(built.app.ctx)["time"] == "07:00"


@pytest.mark.parametrize("tool_set", ["default", "gpsr", "hri"])
def test_with_vision_find_people_and_find_objects_read_live_and_the_prompt_says_so(build, tool_set):
    built = build(tool_set=tool_set, vision=True)
    app, node = built.app, built.node
    assert isinstance(app.ctx.vision, RosVision)
    [people] = [s for s in node.subscriptions if s.topic == PEOPLE_TOPIC]
    [objects] = [s for s in node.subscriptions if s.topic == OBJECTS_TOPIC]
    app.start()
    run_one(built)

    system = built.model.received[0][0].content
    assert "`find_people` reports the people your camera sees now" in system
    assert "live people perception, which is not connected" not in system
    assert "`find_objects` reports the objects your camera sees now" in system
    assert "live object perception, which is not" not in system
    assert system == prompts.build_system_prompt(tool_set, PLACES, ROBOT, vision=True)
    named = set(re.findall(r"`(\w+)`", system))
    assert named <= set(built.model.bound[0])
    # object_info is named only where the set binds it.
    assert ("`object_info`" in system) == (tool_set != "hri")

    # Closed with the rest, after the Runner, and once.
    assert app.close(WATCHDOG_S)
    assert app.close(WATCHDOG_S)
    assert node.destroyed.count(people) == 1
    assert node.destroyed.count(objects) == 1
    app.ctx.cancel.reset()  # close set the latch, which a verb checks first
    with pytest.raises(StepError) as raised:
        app.ctx.vision.people()
    assert str(raised.value) == "UNAVAILABLE: Live people perception has shut down."


def test_with_vision_the_objects_topic_is_the_one_given(build):
    built = build(tool_set="hri", vision=True, objects_topic="/walkie_vision/sim/objects")
    topics = {s.topic for s in built.node.subscriptions}
    assert {PEOPLE_TOPIC, "/walkie_vision/sim/objects"} <= topics
    assert OBJECTS_TOPIC not in topics


def guest_array(node, track_id=1):
    """A people array stamped at `node`'s clock, with one guest whose appearance is in view."""
    guest = with_vectors(person(320, track_id=track_id), appearance="guest")
    array = PersonDetectionArray(people=[guest], camera_info=camera_info())
    array.header.frame_id = CAMERA_FRAME
    array.header.stamp = node.clock.now().to_msg()
    return array


def test_without_a_people_memory_the_memory_tools_are_unavailable_and_the_prompt_says_so(build):
    built = build(tool_set="hri", vision=True)
    app = built.app
    assert isinstance(app.ctx.memory, UnavailableMemory)
    assert app.runner._on_reset is None
    assert app.runner._on_stop is None  # no commander without --arms
    assert app.ctx.vision._namer is None
    app.start()
    run_one(built)

    system = built.model.received[0][0].content
    assert system == prompts.build_system_prompt("hri", PLACES, ROBOT, vision=True)
    assert "`remember_person` and `recognize_person` need a people memory" in system
    assert "When someone tells you their name" not in system
    with pytest.raises(StepError) as raised:
        app.ctx.memory.recognize("person_1")
    assert raised.value.code is ErrorCode.UNAVAILABLE


def test_a_mock_people_memory_remembers_from_the_vision_and_forgets_at_a_reset(build):
    built = build(tool_set="hri", vision=True, people_memory="mock")
    app, node = built.app, built.node
    memory = app.ctx.memory
    assert isinstance(memory, TrackMemory)
    assert memory._source is app.ctx.vision
    assert memory._cancel is app.ctx.cancel
    store = memory._store
    assert isinstance(store, InMemoryPeopleMemory)
    # The mock writes through the runtime's fence, so a stop never stores anyone.
    assert store._cancel is app.ctx.cancel
    assert app.ctx.vision._namer == memory.names
    assert app.runner._on_reset == memory.clear
    app.start()
    run_one(built)

    system = built.model.received[0][0].content
    assert system == prompts.build_system_prompt("hri", PLACES, ROBOT, vision=True, memory=True)
    assert "When someone tells you their name, call `remember_person`" in system
    assert "need a people memory" not in system
    named = set(re.findall(r"`(\w+)`", system))
    assert named <= set(built.model.bound[0])

    # The guest's samples come from the people subscription, and find_people names them.
    [people] = [s for s in node.subscriptions if s.topic == PEOPLE_TOPIC]
    people.callback(guest_array(node))
    assert app.ctx.memory.remember_person("person_1", "Sam").name == "Sam"
    assert [p.name for p in app.ctx.vision.people()] == ["Sam"]

    # A reset ends the task, and the Runner forgets the guest.
    next(s for s in node.subscriptions if s.topic == "/walkie/reset").callback(Empty())
    assert app.runner.wait_idle(WATCHDOG_S)
    assert len(store) == 0
    app.ctx.cancel.reset()  # the reset set the latch, which a verb checks first
    assert [p.name for p in app.ctx.vision.people()] == [None]


@pytest.mark.parametrize("resume", [False, True])
def test_a_ros_people_memory_is_made_on_the_connector_with_the_agents_token(build, monkeypatch, resume):
    made = []

    class StandIn(InMemoryPeopleMemory):
        """RosPeopleMemory's constructor over the mock: FakeNode makes no service clients."""

        def __init__(self, connector, cancel, *, resume_session=False):
            super().__init__()
            made.append((connector, cancel, resume_session))

    monkeypatch.setattr(main, "RosPeopleMemory", StandIn)
    built = build(
        tool_set="hri", vision=True, people_memory="ros", resume_people_session=resume
    )
    [(connector, cancel, resume_session)] = made
    assert connector is built.connector
    assert cancel is built.app.ctx.cancel
    assert resume_session is resume
    assert isinstance(built.app.ctx.memory._store, StandIn)
    assert built.app.runner._on_reset == built.app.ctx.memory.clear


@pytest.mark.parametrize(
    ("options", "words"),
    [
        ({"people_memory": "mock"}, "--people-memory mock needs --vision"),
        ({"people_memory": "ros"}, "--people-memory ros needs --vision"),
        ({"vision": True, "resume_people_session": True}, "needs --people-memory ros"),
        (
            {"vision": True, "people_memory": "mock", "resume_people_session": True},
            "needs --people-memory ros",
        ),
        ({"vision": True, "people_memory": "sqlite"}, "no people memory 'sqlite'"),
    ],
)
def test_assembly_refuses_a_people_memory_it_cannot_use_before_building_anything(
    build, options, words
):
    connector = FakeConnector()
    with pytest.raises(ValueError, match=re.escape(words)):
        build(connector=connector, **options)
    assert connector.node.publishers == []
    assert connector.node.subscriptions == []


def test_assemble_never_spins_or_destroys_the_connector(build):
    # FakeConnector has no spin or shutdown, so a call to either would raise here.
    built = build()
    built.app.start()
    run_one(built)
    assert built.app.close(WATCHDOG_S)
    assert set(built.node.publishers) <= set(built.node.destroyed)


def test_an_unknown_tool_set_is_refused(build):
    with pytest.raises(ValueError, match="no tool set 'garage'"):
        build(tool_set="garage")


def test_a_reset_is_not_a_shutdown(build):
    built = build()
    built.app.start()
    next(s for s in built.node.subscriptions if s.topic == "/walkie/reset").callback(Empty())
    assert built.app.runner.wait_idle(WATCHDOG_S)
    assert built.node.destroyed == []
    run_one(built)
    assert len(built.model.received) == 1


@pytest.mark.parametrize("again", ["stop", "reset"])
def test_a_later_stop_or_reset_asks_again_after_a_refused_cancel(build, nav, again):
    built = build(ai(call("go_to", {"place": "kitchen"})))
    built.app.start()
    callbacks = {s.topic: s.callback for s in built.node.subscriptions}
    built.app.runner.submit("Drive to the kitchen.")
    nav[0].wait_for_goals()
    handle = nav[0].accept(
        on_cancel=lambda h, _future: h.acknowledge(CancelGoal.Response.ERROR_REJECTED)
    )
    assert handle.result_asked.wait(WATCHDOG_S)

    callbacks["/from_human"](HRIMessage(text="stop"))
    assert built.app.runner.wait_idle(WATCHDOG_S)
    # One ask, or two when the verb's refused ask lands before the stop's own.
    refused = handle.cancel_asks
    assert refused >= 1
    assert built.app.ctx.base.unresolved(), "a refused cancel leaves the goal live"

    # No verb waits on this goal any more, so only the Runner can ask again.
    handle.on_cancel = lambda h, _future: h.finish(GoalStatus.STATUS_CANCELED)
    if again == "stop":
        callbacks["/from_human"](HRIMessage(text="stop"))
    else:
        callbacks["/walkie/reset"](Empty())
    assert built.app.runner.wait_idle(WATCHDOG_S)
    assert handle.cancel_asks == refused + 1
    assert built.app.ctx.base.unresolved() == []


@pytest.mark.parametrize("tool_set", ["default", "gpsr", "hri"])
def test_each_tool_set_gets_a_matching_prompt(build, tool_set):
    built = build(tool_set=tool_set)
    built.app.start()
    run_one(built)

    bound = set(built.model.bound[0])
    assert bound == tool_names(tool_set)
    assert not bound & ARM_TOOLS, "without --arms no set binds the arm tools"
    system = built.model.received[0][0]
    assert isinstance(system, SystemMessage)
    named = set(re.findall(r"`(\w+)`", system.content))
    assert named <= bound, f"The prompt names tools it lacks: {named - bound}"
    assert {"say", "go_to", "move_relative", "update_todos", "wait_for_person"} <= named
    assert "You also have two arms" not in system.content

    # A prompt says plainly that what its set offers is not connected yet.
    assert "UNAVAILABLE" in system.content
    assert "live object perception, which is not" in system.content
    assert ("`pick`" in system.content) == (tool_set == "gpsr")
    assert "`find_people`" in system.content
    assert "live people perception, which is not connected" in system.content
    assert ("`get_facts`" in system.content) == (tool_set != "hri")

    # GPSR cannot list places, so its prompt holds them as data. The others look them up.
    assert ("- kitchen: Kitchen counter." in system.content) == (tool_set == "gpsr")
    assert ("- door\n" in system.content) == (tool_set == "gpsr")
    assert ("`list_places`" in system.content) == (tool_set != "gpsr")


def test_with_scripts_the_agent_binds_run_script_and_takes_the_verdict(build, pool):
    built = build(tool_set="hri", scripts=AgentScripts(pool, Shelf()))
    node = built.node
    built.app.start()
    run_one(built)

    bound = set(built.model.bound[0])
    assert bound == tool_names("hri") | {"run_script"}
    system = built.model.received[0][0].content
    assert set(re.findall(r"`(\w+)`", system)) <= bound
    assert "`run_script`" in system
    assert "/walkie/verdict" in topics(node.subscriptions)

    # The task just ended, so its verdict is taken, and the answer goes to the console.
    callbacks = {s.topic: s.callback for s in node.subscriptions}
    callbacks["/walkie/verdict"](Bool(data=True))
    callbacks["/walkie/verdict"](Bool(data=True))
    console = next(p for p in node.publishers if p.topic == "/walkie/console")
    assert [message.text for message in console.published] == [
        "Done.",
        "Recorded a passed verdict on the last task. It proved no saved function, so nothing was verified.",
        "Ignored the verdict: no task has ended since the last one.",
    ]


def test_gpsr_says_so_when_no_place_is_configured(build):
    built = build(tool_set="gpsr", locations={})
    built.app.start()
    run_one(built)
    assert "No named places are configured" in built.model.received[0][0].content


def test_robot_description_appended_once_without_whoami(build):
    built = build()
    built.app.start()
    run_one(built)
    system = built.model.received[0][0].content
    assert system.count(ROBOT) == 1
    assert system.endswith(ROBOT + "\n")

    # Empty content is valid and adds nothing, and there is no generated fallback.
    empty = prompts.build_system_prompt("default", PLACES, " \n")
    assert empty == prompts.build_system_prompt("default", PLACES, "")
    assert system.startswith(empty)
    assert "whoami" not in Path(main.__file__).read_text(encoding="utf-8")
    assert "whoami" not in Path(prompts.__file__).read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        main._parse_args(["--whoami-dir", "whoami"])


def test_template_requirements_must_be_bound(build, commanders):
    bound = template({"content": "Pick", "requires_tool": "pick"})
    unbound = {"tour": bound}
    build(tool_set="gpsr", templates=unbound)

    connector = FakeConnector()
    with pytest.raises(ValueError, match=r"tour\.t1 requires pick"):
        build(tool_set="hri", templates=unbound, connector=connector)
    assert connector.node.publishers == [], "nothing is built for a template that cannot run"

    # A stale tool name from the old host, and the one tool the ledger never records.
    for name in ("speak", "update_todos"):
        stale = {"tour": template({"content": "Go", "requires_tool": name})}
        with pytest.raises(ValueError, match=f"requires {name}"):
            build(templates=stale)

    # A person can be asked in every set, and a step with no requirement is always valid.
    runs = [*((s, None) for s in SETS_WITHOUT_ARMS), *((s, POSES) for s in SETS_WITH_ARMS)]
    assert {tool_set for tool_set, _ in runs} == set(TOOL_SET_LISTS)
    for tool_set, arm_poses in runs:
        asks = {"tour": template("Plain", {"content": "Ask", "requires_tool": "wait_for_person"})}
        build(tool_set=tool_set, templates=asks, arm_poses=arm_poses)


def test_a_template_step_on_an_arm_tool_needs_the_arms(build, commanders):
    steps = {"tour": template({"content": "Open the hand", "requires_tool": "set_gripper"})}
    connector = FakeConnector()
    with pytest.raises(ValueError, match=r"tour\.t1 requires set_gripper"):
        build(tool_set="hri", templates=steps, connector=connector)
    assert connector.node.publishers == []
    assert commanders == []

    build(tool_set="hri", templates=steps, arm_poses=POSES)


@pytest.mark.parametrize("wanted", [True, False])
def test_assembly_refuses_a_connector_on_the_other_clock(build, wanted):
    connector = FakeConnector()
    connector.node.use_sim_time = not wanted
    with pytest.raises(ValueError, match="use_sim_time"):
        build(connector=connector, use_sim_time=wanted)
    assert connector.node.publishers == []


def test_injected_model_never_builds_network_clients(build, monkeypatch):
    def never(*_args, **_kwargs):
        raise AssertionError("assemble must not build a model or a tracer.")

    for name in ("build_model", "build_fallback", "build_callbacks"):
        monkeypatch.setattr(main, name, never)
        monkeypatch.setattr(f"walkie_agent.chat_models.{name}", never)
    monkeypatch.setattr("walkie_agent.chat_models._chat_model", never)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", never)
    monkeypatch.setattr("langchain_ollama.ChatOllama", never)
    monkeypatch.setattr("langfuse.Langfuse", never)

    built = build(config={"tracing": {"langfuse": {"use_langfuse": True}}})
    built.app.start()
    run_one(built)
    assert len(built.model.received) == 1


# -- Application.close -------------------------------------------------------------


def test_close_before_start_cleans_up_and_repeats_safely(build, nav):
    built = build()
    assert built.app.close(WATCHDOG_S)
    assert nav[0].destroyed == 1
    assert set(built.node.publishers) <= set(built.node.destroyed)
    count = len(built.node.destroyed)
    assert built.app.close(WATCHDOG_S)
    assert nav[0].destroyed == 1
    assert len(built.node.destroyed) == count


def test_close_stops_at_a_run_that_has_not_ended(build, nav, monkeypatch):
    built = build()
    built.app.start()
    monkeypatch.setattr(built.app.runner, "shutdown", lambda timeout=None: False)

    assert built.app.close(0.0) is False

    assert built.app.ctx.cancel.cancelled
    assert built.node.destroyed == [], "a destroy can end the spin thread that a goal still needs"
    assert nav[0].destroyed == 0
    assert built.app.unresolved() == ["The agent's current run has not ended."]

    monkeypatch.undo()
    assert built.app.close(WATCHDOG_S)
    assert built.app.unresolved() == []


def test_a_script_sandbox_that_has_not_closed_keeps_close_false(build, pool):
    release = threading.Event()
    scripts = AgentScripts(pool, Shelf(), closer=lambda: release.wait(WATCHDOG_S))
    built = build(scripts=scripts)
    built.app.start()

    assert built.app.close(0.05) is False
    # The sandbox closes last, so everything else has closed already.
    assert set(built.node.publishers) <= set(built.node.destroyed)
    assert built.app.unresolved() == ["The script sandbox's workers have not closed."]

    release.set()
    assert built.app.close(WATCHDOG_S)
    assert built.app.unresolved() == []


# -- The arms ------------------------------------------------------------------------


def test_with_arms_the_commander_shares_the_token_and_the_base_checks_the_arms(
    build, commanders, monkeypatch
):
    made_before_the_base = []
    base = main.RosBase

    def recording_base(*args, **kwargs):
        made_before_the_base.append(len(commanders))
        return base(*args, **kwargs)

    monkeypatch.setattr(main, "RosBase", recording_base)
    built = build(tool_set="hri", arm_poses=POSES)
    app = built.app

    [commander] = commanders
    assert made_before_the_base == [1]
    assert commander.connector is built.connector
    assert commander.cancel is app.ctx.cancel
    assert commander.poses is POSES
    assert commander.bounds == {}, "off the simulation the client keeps its robot bounds"

    arms = app.ctx.arms
    assert isinstance(arms, RosArms)
    assert arms._client is commander
    assert arms._cancel is app.ctx.cancel
    assert (arms._sim, arms._min_move_timeout_s) == (False, 0.0)
    assert app.ctx.base._arms_check == arms.check_for_driving
    assert list(arms.poses()) == list(POSES.offered_poses(sim=False))
    # A runtime stop or reset also publishes the commander's stop.
    assert app.runner._on_stop == commander.stop
    app.runner.stop()
    assert commander.log == ["stop"]


def test_under_sim_time_the_commander_gets_the_sim_bounds(build, commanders):
    connector = FakeConnector()
    connector.node.use_sim_time = True
    built = build(connector=connector, use_sim_time=True, tool_set="arms", arm_poses=POSES)

    [commander] = commanders
    bounds = dict(main.SIM_ARM_BOUNDS)
    min_move_timeout_s = bounds.pop("min_move_timeout_s")
    assert commander.bounds == bounds
    # The stand-in takes any keyword, so the real constructor must take these.
    inspect.signature(CommanderClient).bind(connector, built.app.ctx.cancel, POSES, **bounds)
    assert bounds["read_timeout_s"] > READ_TIMEOUT_S
    arms = built.app.ctx.arms
    assert (arms._sim, arms._min_move_timeout_s) == (True, min_move_timeout_s)
    assert min_move_timeout_s > 0
    # The simulation offers the poses not yet tried on the robot.
    assert list(arms.poses()) == list(POSES.offered_poses(sim=True))
    assert len(arms.poses()) > len(POSES.offered_poses(sim=False))


@pytest.mark.parametrize("tool_set", SETS_WITH_ARMS)
def test_with_arms_the_agent_gets_the_arm_tools_and_the_arm_rule(build, commanders, tool_set):
    built = build(ai(call("list_arm_poses")), tool_set=tool_set, arm_poses=POSES)
    built.app.start()
    run_one(built, "Which arm poses do you know?")

    bound = set(built.model.bound[0])
    assert bound == tool_names(tool_set, arms=True)
    assert ARM_TOOLS <= bound
    system = built.model.received[0][0].content
    assert system == prompts.build_system_prompt(tool_set, PLACES, ROBOT, arms=True)
    named = set(re.findall(r"`(\w+)`", system))
    assert named <= bound, f"The prompt names tools it lacks: {named - bound}"
    assert "You also have two arms with grippers." in system
    assert "Never pick up an object by combining poses" in system
    # The drive check refuses an arm out with its own words, so no rule repeats them.
    assert "Before you drive" not in system
    assert "need an arm commander client" not in system

    # list_arm_poses reads ctx.arms, so it answers with the poses offered on the robot.
    answer = built.model.received[1][-1]
    assert isinstance(answer, ToolMessage)
    for name in POSES.offered_poses(sim=False):
        assert f"{name}: " in answer.content
    assert "receive: " not in answer.content, "a pose not tried on the robot stays out"


def test_with_an_arm_out_the_agents_drive_is_refused_before_a_goal(build, commanders, nav):
    built = build(ai(call("go_to", {"place": "kitchen"})), tool_set="hri", arm_poses=POSES)
    [commander] = commanders
    commander.state = ArmsState(left=HANDS_UP, right=HOME)
    built.app.start()
    run_one(built, "Drive to the kitchen.")

    answer = built.model.received[1][-1]
    assert isinstance(answer, ToolMessage)
    assert answer.content.startswith(
        "PRECONDITION_FAILED: Not starting the drive to kitchen: the left arm is"
    )
    assert "move_arm_to_pose" in answer.content
    assert nav[0].goals == []
    # One read, bounded by the client's read timeout.
    assert commander.log == [("check", READ_TIMEOUT_S)]


@pytest.mark.parametrize(
    ("tool_set", "arm_poses", "words"),
    [
        ("arms", None, "--tool-set arms needs --arms"),
        ("default", POSES, "the default set has no move_arm_to_pose"),
        ("gpsr", POSES, "the gpsr set has no move_arm_to_pose"),
    ],
)
def test_assembly_refuses_a_set_that_does_not_go_with_the_arms_before_building_anything(
    build, commanders, tool_set, arm_poses, words
):
    connector = FakeConnector()
    with pytest.raises(ValueError, match=re.escape(words)):
        build(connector=connector, tool_set=tool_set, arm_poses=arm_poses)
    assert connector.node.publishers == []
    assert connector.node.subscriptions == []
    assert commanders == []


@pytest.fixture
def object_memories(monkeypatch):
    """What main makes RosObjectMemory with: FakeNode makes no service clients."""
    made = []

    class StandIn(UnavailableObjectMemory):
        def __init__(self, connector, cancel):
            made.append((connector, cancel))

    monkeypatch.setattr(main, "RosObjectMemory", StandIn)
    return made


@pytest.mark.parametrize("tool_set", ["default", "hri"])
def test_with_object_memory_the_agent_gets_the_object_tools_and_their_rule(
    build, object_memories, tool_set
):
    built = build(tool_set=tool_set, vision=True, object_memory=True)
    built.app.start()
    run_one(built, "Where is the cup?")

    [(connector, cancel)] = object_memories
    assert connector is built.connector
    assert cancel is built.app.ctx.cancel
    assert isinstance(built.app.ctx.object_memory, main.RosObjectMemory)
    bound = set(built.model.bound[0])
    assert bound == tool_names(tool_set, object_memory=True)
    assert OBJECT_MEMORY_TOOLS <= bound
    system = built.model.received[0][0].content
    assert system == prompts.build_system_prompt(
        tool_set, PLACES, ROBOT, vision=True, object_memory=True
    )
    assert "`recall_object` tells where you saw objects before" in system
    named = set(re.findall(r"`(\w+)`", system))
    assert named <= bound, f"The prompt names tools it lacks: {named - bound}"


def test_with_arms_and_object_memory_hri_gets_pick_object_and_its_rule(
    build, object_memories, commanders
):
    built = build(tool_set="hri", vision=True, object_memory=True, arm_poses=POSES)
    built.app.start()
    run_one(built, "Bring me the water bottle.")

    bound = set(built.model.bound[0])
    assert bound == tool_names("hri", arms=True, object_memory=True)
    assert PICK_TOOLS <= bound
    system = built.model.received[0][0].content
    assert system == prompts.build_system_prompt(
        "hri", PLACES, ROBOT, vision=True, arms=True, object_memory=True
    )
    assert "call\n  `pick_object` with its name" in system
    named = set(re.findall(r"`(\w+)`", system))
    assert named <= bound, f"The prompt names tools it lacks: {named - bound}"


@pytest.mark.parametrize(
    ("arms", "object_memory"), [(True, False), (False, True)], ids=["arms", "object-memory"]
)
def test_pick_object_needs_both_the_arms_and_the_object_memory(
    build, object_memories, commanders, arms, object_memory
):
    built = build(
        tool_set="hri",
        vision=True,
        object_memory=object_memory,
        arm_poses=POSES if arms else None,
    )
    built.app.start()
    run_one(built)

    assert PICK_TOOLS.isdisjoint(built.model.bound[0])
    assert "pick_object" not in built.model.received[0][0].content


@pytest.mark.parametrize("tool_set", ["default", "hri"])
def test_without_object_memory_the_run_keeps_the_tools_and_prompt_it_had(build, tool_set):
    built = build(tool_set=tool_set, vision=True)
    built.app.start()
    run_one(built)

    assert isinstance(built.app.ctx.object_memory, UnavailableObjectMemory)
    assert OBJECT_MEMORY_TOOLS.isdisjoint(built.model.bound[0])
    assert "recall_object" not in built.model.received[0][0].content


@pytest.mark.parametrize(
    ("tool_set", "vision", "words"),
    [
        ("default", False, "--object-memory needs --vision"),
        ("hri", False, "--object-memory needs --vision"),
        ("gpsr", True, "the gpsr set has no recall_object"),
        ("arms", True, "the arms set has no recall_object"),
    ],
)
def test_assembly_refuses_an_object_memory_it_cannot_use_before_building_anything(
    build, object_memories, tool_set, vision, words
):
    connector = FakeConnector()
    with pytest.raises(ValueError, match=re.escape(words)):
        build(
            connector=connector,
            tool_set=tool_set,
            vision=vision,
            object_memory=True,
            arm_poses=POSES if tool_set == "arms" else None,
        )
    assert connector.node.publishers == []
    assert connector.node.subscriptions == []
    assert object_memories == []


def test_close_stops_the_arms_at_once_and_closes_the_commander_after_the_base(
    build, commanders, nav, monkeypatch
):
    built = build(tool_set="hri", arm_poses=POSES)
    app = built.app
    [commander] = commanders
    real_shutdown = app.runner.shutdown

    def shutdown(timeout=None):
        commander.log.append("runner")
        return real_shutdown(timeout)

    monkeypatch.setattr(app.runner, "shutdown", shutdown)
    # What is closed as the commander closes: the base, and nothing of speech,
    # vision or the bridge.
    commander.on_close = lambda: commander.log.append(
        ("base destroyed", nav[0].destroyed, "node destroyed", list(built.node.destroyed))
    )
    app.start()

    assert app.close(WATCHDOG_S)
    assert commander.log == ["stop", "runner", "close", ("base destroyed", 1, "node destroyed", [])]
    assert set(built.node.publishers) <= set(built.node.destroyed)

    # A repeated close sends "stop" again, which never raises, and closes the client
    # again, which then does nothing. Only an assembly's rollback calls destroy().
    commander.on_close = None
    assert app.close(WATCHDOG_S)
    assert commander.log[4:] == ["stop", "runner", "close"]
    assert "destroy" not in commander.log


@pytest.mark.parametrize("stage", ["runner", "base"])
def test_a_close_that_has_not_finished_keeps_the_commander(build, commanders, nav, monkeypatch, stage):
    built = build(tool_set="hri", arm_poses=POSES)
    app = built.app
    [commander] = commanders
    app.start()
    if stage == "runner":
        monkeypatch.setattr(app.runner, "shutdown", lambda timeout=None: False)
    else:
        monkeypatch.setattr(app._base, "close", lambda timeout_s=None: False)

    assert app.close(WATCHDOG_S) is False
    assert commander.log == ["stop"], "a goal still live may need the commander's clients"

    monkeypatch.undo()
    assert app.close(WATCHDOG_S)
    assert commander.log == ["stop", "stop", "close"]


def test_a_commander_close_that_has_not_finished_keeps_speech_vision_and_the_bridge(
    build, commanders, nav
):
    built = build(tool_set="hri", arm_poses=POSES, vision=True)
    app = built.app
    [commander] = commanders
    commander.closes = False
    app.start()

    assert app.close(WATCHDOG_S) is False

    assert commander.log == ["stop", "close"]
    assert nav[0].destroyed == 1, "the base closes before the commander"
    assert built.node.destroyed == [], "speech, vision and the bridge stay open"
    assert app.unresolved() == commander.left

    # The goal ends, and the retried close finishes.
    commander.closes = True
    assert app.close(WATCHDOG_S)
    assert commander.log == ["stop", "close", "stop", "close"]
    assert set(built.node.publishers) <= set(built.node.destroyed)
    assert set(built.node.subscriptions) <= set(built.node.destroyed)
    assert app.unresolved() == []


@pytest.mark.parametrize("stage", ["RosBase", "RosSpeech", "build_agent"])
def test_a_failure_after_the_commander_destroys_it_once(build, commanders, nav, monkeypatch, stage):
    raise_in(monkeypatch, stage)
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        build(tool_set="hri", arm_poses=POSES)
    [commander] = commanders
    assert commander.log == ["destroy"]
    assert all(client.destroyed == 1 for client in nav)


# -- The command line --------------------------------------------------------------


class _Exited(Exception):
    """os._exit, as a test can see it. A real one would end the test run."""


class RecordingConnector(FakeConnector):
    """What ROS2Connector's constructor and shutdown were asked to do."""

    def __init__(self, cli, **kwargs):
        super().__init__(FakeNode())
        cli.connectors.append(self)
        self.cli = cli
        self.kwargs = kwargs
        self.node.use_sim_time = kwargs["use_sim_time"]
        self.on_shutdown = None
        self.node.destroy_subscription = partial(self._destroyed, self.node.destroy_subscription)
        self.node.destroy_publisher = partial(self._destroyed, self.node.destroy_publisher)
        cli.events.append("connector")

    def _destroyed(self, destroy, entity):
        self.cli.events.append(f"destroy {entity.topic}")
        return destroy(entity)

    def shutdown(self):
        self.cli.events.append("connector.shutdown")
        if self.on_shutdown is not None:
            self.on_shutdown()


class FakeApp:
    """An Application whose close() gives scripted answers."""

    def __init__(self, cli, closes=(True,), unresolved=("the drive to kitchen: no answer.",)):
        self.cli = cli
        self.closes = list(closes)
        self._unresolved = list(unresolved)
        self.on_start = None
        self.on_close = None

    def start(self):
        self.cli.events.append("start")
        if self.on_start is not None:
            self.on_start()

    def close(self, timeout_s=None):
        self.cli.events.append("close")
        if self.on_close is not None:
            self.on_close()
        return self.closes.pop(0) if len(self.closes) > 1 else self.closes[0]

    def unresolved(self):
        return self._unresolved


def send(number=signal.SIGINT):
    os.kill(os.getpid(), number)


@pytest.fixture
def cli(monkeypatch, tmp_path, nav):
    """Runs main() over patched ROS lifecycle and model factories. Records the order."""
    state = SimpleNamespace(
        events=[],
        init_kwargs=[],
        connectors=[],
        model=ScriptedModel(then=DONE),
        on_started=None,
        nav=nav,
    )
    events = state.events

    def init(**kwargs):
        events.append("init")
        state.init_kwargs.append(kwargs)

    monkeypatch.setattr(
        main,
        "rclpy",
        SimpleNamespace(init=init, try_shutdown=lambda: events.append("try_shutdown")),
    )

    def connector(**kwargs):
        return RecordingConnector(state, **kwargs)

    monkeypatch.setattr(main, "ROS2Connector", connector)

    def recording(name, real):
        def wrapper(*args, **kwargs):
            events.append(name)
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(main, "load_dotenv", lambda: events.append("dotenv"))
    monkeypatch.setattr(main, "build_model", recording("build_model", lambda _config: state.model))
    monkeypatch.setattr(main, "build_fallback", recording("build_fallback", main.build_fallback))
    monkeypatch.setattr(main, "build_callbacks", recording("build_callbacks", main.build_callbacks))

    def exited(code):
        raise _Exited(code)

    monkeypatch.setattr(os, "_exit", exited)

    real_start = main.Application.start

    def start(self):
        real_start(self)
        events.append("started")
        if state.on_started is not None:
            state.on_started(self)

    monkeypatch.setattr(main.Application, "start", start)

    config = tmp_path / "config.toml"
    config.write_text(MODEL_CONFIG + TEAM_CONFIG)
    robot = tmp_path / "robot.md"
    robot.write_text(ROBOT)
    missing_templates = tmp_path / "no_templates.toml"
    state.paths = {
        "--config": str(config),
        "--locations": str(FIXTURE_LOCATIONS),
        "--task-templates": str(missing_templates),
        "--robot-description": str(robot),
    }
    state.tmp_path = tmp_path

    def run(*flags, **paths):
        chosen = {**state.paths, **{f"--{k.replace('_', '-')}": v for k, v in paths.items()}}
        argv = [item for pair in chosen.items() for item in pair]
        return main.main([*argv, *flags])

    state.run = run
    state.index = events.index

    def use_fake_app(app):
        monkeypatch.setattr(main, "assemble", lambda **_kwargs: app)
        return app

    state.use_fake_app = use_fake_app

    # Every signal a test sends counts, however close together; the tests of the repeat
    # window set it themselves.
    monkeypatch.setattr(main, "SIGNAL_REPEAT_S", 0.0)

    # A stuck main thread would hang the whole run. Two signals force it out as a failure,
    # the second after the repeat window, which a test may have set. The teardown ends the
    # wait between them, so a first signal that let the run finish sends no second one
    # to the handlers restored after it.
    torn_down = threading.Event()

    def force_out():
        send(signal.SIGTERM)
        if not torn_down.wait(main.SIGNAL_REPEAT_S + 0.1):
            send(signal.SIGTERM)

    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    watchdog = threading.Timer(3 * WATCHDOG_S, force_out)
    watchdog.daemon = True
    watchdog.start()
    yield state
    watchdog.cancel()
    torn_down.set()
    watchdog.join(WATCHDOG_S)
    for number, handler in handlers.items():
        assert signal.getsignal(number) == handler, "main did not restore a signal handler"


def stop_when_started(cli):
    """The usual run: start, a first signal, then the shutdown."""
    cli.on_started = lambda _app: send()


def test_cli_passes_files_and_flags_into_the_assembly(cli, monkeypatch):
    received = {}

    def assemble(**kwargs):
        received.update(kwargs)
        app = FakeApp(cli)
        app.on_start = send
        return app

    monkeypatch.setattr(main, "assemble", assemble)
    tool_log = cli.tmp_path / "tools.jsonl"
    code = cli.run(
        "--tool-set",
        "gpsr",
        "--tool-log",
        str(tool_log),
        "--input-topic",
        "/i",
        "--output-topic",
        "/o",
        "--reset-topic",
        "/r",
        "--task-topic",
        "/k",
        "--todos-topic",
        "/t",
        "--vision",
    )

    assert code == 0
    assert received["vision"] is True
    assert received["objects_topic"] == OBJECTS_TOPIC
    assert received["connector"] is cli.connectors[0]
    assert received["model"] is cli.model
    assert received["fallback"] is None
    assert received["callbacks"] == []
    assert received["config"]["vendor"]["complex_model"] == "openai"
    assert received["locations"] == load_locations(FIXTURE_LOCATIONS)
    assert received["templates"] == {}
    assert received["robot_description"] == ROBOT
    assert received["tool_set"] == "gpsr"
    assert isinstance(received["tool_log"], ToolLog)
    assert received["tool_log"].path == tool_log
    assert isinstance(received["facts"], SystemFacts)
    assert received["facts"].team() == Team("EIC", "Thailand", "Chulalongkorn University")
    assert received["facts"].timezone == "Asia/Bangkok"
    assert received["arm_poses"] is None, "without --arms no pose file is read"
    assert (
        received["input_topic"],
        received["output_topic"],
        received["reset_topic"],
        received["task_topic"],
        received["todos_topic"],
        received["use_sim_time"],
    ) == ("/i", "/o", "/r", "/k", "/t", False)


def test_cli_defaults_keep_the_existing_paths_and_topics():
    args = main._parse_args([])
    assert args.config == "config.toml"
    assert args.locations == "config/locations.toml"
    assert args.task_templates == "config/task_templates.toml"
    assert args.robot_description == "robot.md"
    assert args.tool_set == "default"
    assert args.tool_log is None
    assert (
        args.input_topic,
        args.output_topic,
        args.reset_topic,
        args.task_topic,
        args.todos_topic,
    ) == ("/from_human", "/walkie/console", "/walkie/reset", "/walkie/task", "/walkie/todos")
    assert args.use_sim_time is False
    assert args.vision is False
    assert args.people_memory == "unavailable"
    assert args.resume_people_session is False
    assert args.arms is False
    assert args.scripts is False
    assert main._parse_args(["--scripts"]).scripts is True
    assert main._parse_args(["--tool-set", "hri"]).tool_set == "hri"
    with pytest.raises(SystemExit):
        main._parse_args(["--tool-set", "garage"])
    with pytest.raises(SystemExit):
        main._parse_args(["--fake-model"])


@pytest.mark.parametrize(
    ("flags", "memory", "resume"),
    [
        (["--vision", "--people-memory", "mock"], "mock", False),
        (["--vision", "--people-memory", "ros"], "ros", False),
        (["--vision", "--people-memory", "ros", "--resume-people-session"], "ros", True),
        (["--vision", "--people-memory", "unavailable"], "unavailable", False),
    ],
)
def test_a_people_memory_with_what_it_needs_is_accepted(flags, memory, resume):
    args = main._parse_args(flags)
    assert (args.people_memory, args.resume_people_session) == (memory, resume)


def test_the_objects_topic_is_read_with_vision():
    args = main._parse_args(["--vision", "--objects-topic", "/walkie_vision/sim/objects"])
    assert args.objects_topic == "/walkie_vision/sim/objects"
    assert main._parse_args([]).objects_topic == OBJECTS_TOPIC


@pytest.mark.parametrize(
    ("flags", "words"),
    [
        (["--people-memory", "mock"], "--people-memory mock needs --vision"),
        (["--people-memory", "ros", "--resume-people-session"], "--people-memory ros needs --vision"),
        (["--vision", "--resume-people-session"], "--resume-people-session needs --people-memory ros"),
        (
            ["--vision", "--people-memory", "mock", "--resume-people-session"],
            "--resume-people-session needs --people-memory ros",
        ),
        (["--vision", "--people-memory", "sqlite"], "invalid choice: 'sqlite'"),
        (["--objects-topic", "/walkie_vision/sim/objects"], "--objects-topic needs --vision"),
    ],
)
def test_a_people_memory_without_what_it_needs_is_a_usage_error(capsys, flags, words):
    with pytest.raises(SystemExit) as exited:
        main._parse_args(flags)
    assert exited.value.code == 2
    error = capsys.readouterr().err
    assert error.startswith("usage: ")
    assert words in error


def test_cli_passes_the_people_memory_into_the_assembly(cli, monkeypatch):
    received = {}

    def assemble(**kwargs):
        received.update(kwargs)
        app = FakeApp(cli)
        app.on_start = send
        return app

    monkeypatch.setattr(main, "assemble", assemble)
    assert cli.run("--vision", "--people-memory", "ros", "--resume-people-session") == 0
    assert (received["vision"], received["people_memory"], received["resume_people_session"]) == (
        True,
        "ros",
        True,
    )


def test_a_people_memory_without_vision_stops_before_ros_init(cli, capsys):
    with pytest.raises(SystemExit) as exited:
        cli.run("--people-memory", "mock")
    assert exited.value.code == 2
    assert "--people-memory mock needs --vision" in capsys.readouterr().err
    # Before .env, the model and ROS: nothing has started.
    assert cli.events == []
    assert cli.connectors == []


@pytest.mark.parametrize("tool_set", SETS_WITH_ARMS)
def test_a_set_with_the_arm_tools_is_accepted_with_arms(tool_set):
    args = main._parse_args(["--arms", "--tool-set", tool_set])
    assert (args.arms, args.tool_set) == (True, tool_set)


@pytest.mark.parametrize(
    ("flags", "words"),
    [
        (["--tool-set", "arms"], "--tool-set arms needs --arms"),
        (
            ["--arms"],
            (
                "--arms needs a tool set with the arm tools, such as hri or arms: the default "
                "set has no move_arm_to_pose"
            ),
        ),
        (["--arms", "--tool-set", "gpsr"], "the gpsr set has no move_arm_to_pose"),
    ],
)
def test_a_set_that_does_not_go_with_the_arms_is_a_usage_error(capsys, flags, words):
    with pytest.raises(SystemExit) as exited:
        main._parse_args(flags)
    assert exited.value.code == 2
    error = capsys.readouterr().err
    assert error.startswith("usage: ")
    assert words in error


@pytest.mark.parametrize("tool_set", ["default", "hri"])
def test_object_memory_is_accepted_with_vision_and_a_set_that_has_its_tools(tool_set):
    args = main._parse_args(["--object-memory", "--vision", "--tool-set", tool_set])
    assert (args.object_memory, args.tool_set) == (True, tool_set)
    assert main._parse_args([]).object_memory is False


@pytest.mark.parametrize(
    ("flags", "words"),
    [
        (
            ["--object-memory"],
            (
                "--object-memory needs --vision: go_to_object leaves the LLM to check with "
                "find_objects"
            ),
        ),
        (
            ["--object-memory", "--vision", "--tool-set", "gpsr"],
            (
                "--object-memory needs a tool set with the object memory tools, such as "
                "default or hri: the gpsr set has no recall_object."
            ),
        ),
    ],
)
def test_an_object_memory_without_what_it_needs_is_a_usage_error(capsys, flags, words):
    with pytest.raises(SystemExit) as exited:
        main._parse_args(flags)
    assert exited.value.code == 2
    error = capsys.readouterr().err
    assert error.startswith("usage: ")
    assert words in error


def test_the_arms_set_without_arms_stops_before_ros_init(cli, capsys):
    with pytest.raises(SystemExit) as exited:
        cli.run("--tool-set", "arms")
    assert exited.value.code == 2
    assert "--tool-set arms needs --arms" in capsys.readouterr().err
    assert cli.events == []
    assert cli.connectors == []


def test_cli_with_arms_loads_the_checked_poses_into_the_assembly(cli, monkeypatch):
    received = {}

    def assemble(**kwargs):
        received.update(kwargs)
        app = FakeApp(cli)
        app.on_start = send
        return app

    monkeypatch.setattr(main, "assemble", assemble)
    # The pose files are read from the working directory, as on the robot.
    monkeypatch.chdir(PROJECT_ROOT)
    assert cli.run("--arms", "--tool-set", "hri", "--use-sim-time") == 0
    assert isinstance(received["arm_poses"], ArmPoses)
    assert received["arm_poses"] == POSES
    assert (received["tool_set"], received["use_sim_time"]) == ("hri", True)


def test_a_template_on_an_object_memory_tool_starts_only_with_object_memory(
    cli, capsys, monkeypatch
):
    steps = broken_file(
        cli,
        '[templates.tour]\ntask = "x"\ntodos = [{ content = "a", requires_tool = "recall_object" }]\n',
    )
    flags = ("--vision", "--tool-set", "hri")
    error = startup_error(cli, capsys, *flags, task_templates=steps)
    assert "tour.t1 requires recall_object" in error

    received = {}

    def assemble(**kwargs):
        received.update(kwargs)
        app = FakeApp(cli)
        app.on_start = send
        return app

    monkeypatch.setattr(main, "assemble", assemble)
    assert cli.run("--object-memory", *flags, task_templates=steps) == 0
    assert received["object_memory"] is True


@pytest.mark.parametrize(
    ("text", "words"),
    [
        (None, "No such file or directory: 'config/arm_poses.toml'"),
        ("not toml = [", "config/arm_poses.toml: 1 problem: - Not valid TOML: "),
        ('[poses.home]\ndescription = "x"\n', "config/arm_poses.toml: "),
    ],
    ids=["no-pose-file", "bad-toml", "bad-pose"],
)
def test_a_broken_arm_pose_file_stops_with_one_line_before_ros_init(cli, capsys, monkeypatch, text, words):
    (cli.tmp_path / "config").mkdir()
    model = PROJECT_ROOT / "config" / "robot_model.toml"
    (cli.tmp_path / "config" / "robot_model.toml").write_bytes(model.read_bytes())
    if text is not None:
        (cli.tmp_path / "config" / "arm_poses.toml").write_text(text)
    monkeypatch.chdir(cli.tmp_path)
    assert words in startup_error(cli, capsys, "--arms", "--tool-set", "hri")
    assert cli.connectors == []


@pytest.mark.parametrize(
    ("text", "words"),
    [
        (None, "No such file or directory: 'config/pick.toml'"),
        ("standoff_m = [", "config/pick.toml: 1 problem: - Not valid TOML: "),
        ('arm = "right"\n', "config/pick.toml: "),
    ],
    ids=["no-pick-file", "bad-toml", "missing-keys"],
)
def test_a_broken_pick_file_stops_with_one_line_when_pick_object_is_bound(
    cli, capsys, monkeypatch, text, words
):
    (cli.tmp_path / "config").mkdir()
    for name in ("robot_model.toml", "arm_poses.toml"):
        source = PROJECT_ROOT / "config" / name
        (cli.tmp_path / "config" / name).write_bytes(source.read_bytes())
    if text is not None:
        (cli.tmp_path / "config" / "pick.toml").write_text(text)
    monkeypatch.chdir(cli.tmp_path)
    flags = ("--arms", "--object-memory", "--vision", "--tool-set", "hri")
    assert words in startup_error(cli, capsys, *flags)
    assert cli.connectors == []


def test_a_run_without_pick_object_never_reads_the_pick_file(cli, monkeypatch):
    (cli.tmp_path / "config").mkdir()
    for name in ("robot_model.toml", "arm_poses.toml"):
        source = PROJECT_ROOT / "config" / name
        (cli.tmp_path / "config" / name).write_bytes(source.read_bytes())
    (cli.tmp_path / "config" / "pick.toml").write_text("not toml = [")

    def assemble(**kwargs):
        app = FakeApp(cli)
        app.on_start = send
        return app

    monkeypatch.setattr(main, "assemble", assemble)
    monkeypatch.chdir(cli.tmp_path)
    assert cli.run("--arms", "--tool-set", "hri") == 0


def test_with_arms_the_shutdown_waits_for_the_commander_to_close_before_the_connector(
    cli, commanders, monkeypatch, caplog
):
    events = cli.events

    def started(app):
        [commander] = commanders
        assert app.ctx.arms._client is commander
        # The first close finds an arm goal that has not ended; it ends before the retry.
        commander.closes = False

        def close():
            events.append(("commander closes", commander.closes, cli.nav[0].destroyed))
            commander.closes = True

        commander.on_close = close
        send()

    cli.on_started = started
    monkeypatch.chdir(PROJECT_ROOT)
    with caplog.at_level(logging.WARNING, logger="walkie_agent.main"):
        assert cli.run("--arms", "--tool-set", "hri") == 0

    [commander] = commanders
    assert commander.bounds == {}
    assert commander.log == ["stop", "close", "stop", "close"]
    assert "Shutdown is waiting" in caplog.text
    assert commander.left[0] in caplog.text
    timeline = events[events.index("started") :]
    expected = [
        ("commander closes", False, 1),
        ("commander closes", True, 1),
        "destroy /to_human",
        "connector.shutdown",
        "try_shutdown",
    ]
    positions = [timeline.index(event) for event in expected]
    assert positions == sorted(positions), timeline


def test_cli_with_scripts_opens_the_sandbox_before_ros_and_closes_it_at_shutdown(
    cli, monkeypatch, isolated_tools_home
):
    opened = []
    real_open = AgentScripts.open.__func__

    def recording_open(cls, shelf, **kwargs):
        cli.events.append("sandbox")
        opened.append(real_open(cls, shelf, **kwargs))
        return opened[-1]

    monkeypatch.setattr(main.AgentScripts, "open", classmethod(recording_open))
    apps = []

    def started(app):
        apps.append(app)
        send()

    cli.on_started = started
    assert cli.run("--scripts") == 0

    [scripts] = opened
    assert apps[0]._scripts is scripts
    assert cli.index("sandbox") < cli.index("init")
    assert scripts.shelf.root == isolated_tools_home.resolve()
    assert (isolated_tools_home / "temporary").is_dir()
    # Application.close closed the pool, and main's own close found nothing left.
    assert scripts.unresolved() == []
    with pytest.raises(RuntimeError, match="not active"), scripts._runner.pool.checkout():
        pass


def test_without_scripts_no_sandbox_or_shelf_is_made(cli, monkeypatch, isolated_tools_home):
    def never(*_args, **_kwargs):
        raise AssertionError("no sandbox without --scripts")

    monkeypatch.setattr(main.AgentScripts, "open", never)
    received = {}

    def assemble(**kwargs):
        received.update(kwargs)
        app = FakeApp(cli)
        app.on_start = send
        return app

    monkeypatch.setattr(main, "assemble", assemble)
    assert cli.run() == 0
    assert received["scripts"] is None
    assert not isolated_tools_home.exists()


def test_scripts_without_the_sandbox_stop_with_one_line_before_ros_init(cli, capsys, monkeypatch):
    def no_bwrap(**_kwargs):
        raise SandboxUnavailable("bwrap is not installed. Install it, or pass --no-bwrap on purpose.")

    monkeypatch.setattr(agent_tool, "open_pool", no_bwrap)

    assert cli.run("--scripts") == main.STARTUP_ERROR_EXIT_CODE
    assert capsys.readouterr().err == (
        "walkie-agent: --scripts runs scripts only inside bwrap, and bwrap is not installed. "
        "Install bwrap, or start without --scripts.\n"
    )
    assert "init" not in cli.events
    assert cli.connectors == []


def test_scripts_with_a_store_git_does_not_ignore_stop_with_one_line(cli, capsys, monkeypatch, tmp_path):
    store = tmp_path / "repo" / "tools"
    subprocess.run(["git", "init", "-q", str(store.parent)], check=True)
    monkeypatch.setenv("WALKIE_TOOLS_HOME", str(store))

    assert cli.run("--scripts") == main.STARTUP_ERROR_EXIT_CODE
    error = capsys.readouterr().err
    assert error.startswith("walkie-agent: ")
    assert error.count("\n") == 1
    assert "init" not in cli.events


def test_models_and_files_are_ready_before_ros_starts(cli):
    stop_when_started(cli)
    assert cli.run() == 0
    assert cli.events[:5] == ["dotenv", "build_model", "build_fallback", "build_callbacks", "init"]


def broken_file(cli, text, name="broken.toml"):
    path = cli.tmp_path / name
    path.write_text(text)
    return str(path)


def startup_error(cli, capsys, *flags, **paths):
    """Run main() into an error found before ROS starts. Returns its message.

    Every such error ends the same way: exit code 2, no rclpy.init, and one line on
    stderr that starts with "walkie-agent: ".
    """
    code = cli.run(*flags, **paths)
    stderr = capsys.readouterr().err
    assert code == 2, stderr
    assert len(stderr.splitlines()) == 1, stderr
    assert stderr.startswith("walkie-agent: ") and stderr.endswith("\n"), stderr
    assert "init" not in cli.events
    return stderr.removeprefix("walkie-agent: ").removesuffix("\n")


@pytest.mark.parametrize(
    ("flag", "text", "extra"),
    [
        ("config", None, ()),
        ("config", "not toml = [", ()),
        ("config", '[agent]\nfallback_vendor = "openai"\n', ()),
        ("config", "[tracing.langfuse]\nuse_langfuse = true\n", ()),
        ("config", MODEL_CONFIG, ()),
        ("config", MODEL_CONFIG + TEAM_CONFIG.replace('"Thailand"', '"  "'), ()),
        ("config", MODEL_CONFIG + TEAM_CONFIG.replace("Asia/Bangkok", "Asia/Atlantis"), ()),
        ("locations", None, ()),
        ("locations", "[locations.Kitchen]\nx = 1\ny = 2\n", ()),
        ("locations", "[locations.door]\nx = nan\ny = 2\n", ()),
        ("task_templates", '[templates.tour]\ntask = "x"\ntodos = []\n', ()),
        (
            "task_templates",
            '[templates.tour]\ntask = "x"\ntodos = [{ content = "a", requires_tool = "pick" }]\n',
            ("--tool-set", "hri"),
        ),
        (
            "task_templates",
            '[templates.tour]\ntask = "x"\ntodos = [{ content = "a", requires_tool = "set_gripper" }]\n',
            ("--tool-set", "hri"),
        ),
        ("robot_description", None, ()),
    ],
    ids=[
        "no-config",
        "bad-toml",
        "half-a-fallback",
        "tracing-without-keys",
        "no-team",
        "blank-team-value",
        "unknown-timezone",
        "no-locations",
        "noncanonical-place",
        "nan-place",
        "empty-template",
        "unbound-requirement",
        "arm-requirement-without-arms",
        "no-robot-description",
    ],
)
def test_config_errors_happen_before_ros_init(cli, capsys, monkeypatch, flag, text, extra):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    path = str(cli.tmp_path / "missing") if text is None else broken_file(cli, text)
    assert startup_error(cli, capsys, *extra, **{flag: path})
    assert cli.connectors == []


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (MODEL_CONFIG, r"needs a \[team\] table"),
        (MODEL_CONFIG + '[team]\nname = "EIC"\n', r"for: country, affiliation, timezone\.$"),
        (MODEL_CONFIG + TEAM_CONFIG.replace('"EIC"', "7"), r"for: name\.$"),
        (MODEL_CONFIG + TEAM_CONFIG.replace("Asia/Bangkok", "asia/bangkok"), "Unknown timezone 'asia/bangkok'"),
        (MODEL_CONFIG + TEAM_CONFIG.replace("Asia/Bangkok", "../etc/passwd"), "Unknown timezone"),
    ],
    ids=["no-team", "missing-values", "not-text", "wrong-case-timezone", "path-timezone"],
)
def test_a_bad_team_table_stops_with_one_line_before_ros_init(cli, capsys, text, match):
    assert re.search(match, startup_error(cli, capsys, config=broken_file(cli, text)))


def test_a_model_that_cannot_be_built_stops_before_ros_init(cli, capsys, monkeypatch):
    def no_key(_config):
        raise ValueError("no API key")

    monkeypatch.setattr(main, "build_model", no_key)
    assert startup_error(cli, capsys) == "no API key"


def test_a_bad_base_url_stops_with_one_line_before_ros_init(cli, capsys, monkeypatch):
    # The real builder, since the client's own URL error is what would escape the catch.
    monkeypatch.setattr(main, "build_model", build_model)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    text = MODEL_CONFIG + 'base_url = "http://127.0.0.1:bad/v1"\n' + TEAM_CONFIG
    message = startup_error(cli, capsys, config=broken_file(cli, text))
    assert message.startswith("[openai].base_url has a bad port in 'http://127.0.0.1:bad/v1'")


def test_a_message_of_several_lines_is_printed_as_one(cli, capsys, monkeypatch):
    # pydantic's ValidationError, from a model client given a bad setting, has this shape.
    def invalid(_config):
        raise ValueError(
            "1 validation error for ChatOllama\nmodel\n  Input should be a valid string\n"
        )

    monkeypatch.setattr(main, "build_model", invalid)
    assert startup_error(cli, capsys) == (
        "1 validation error for ChatOllama model Input should be a valid string"
    )


@pytest.mark.parametrize(
    ("text", "key"),
    [
        (TEAM_CONFIG, "'vendor'"),
        ('[vendor]\ncomplex_model = "openai"\n' + TEAM_CONFIG, "'openai'"),
        ('[vendor]\ncomplex_model = "openai"\n[openai]\n' + TEAM_CONFIG, "'complex_model'"),
    ],
    ids=["no-vendor-table", "no-vendor-model-table", "no-model-name"],
)
def test_a_missing_model_setting_names_the_file_and_key(cli, capsys, monkeypatch, text, key):
    # The real builder, which reads these keys before it builds any client.
    monkeypatch.setattr(main, "build_model", build_model)
    config = broken_file(cli, text)
    assert startup_error(cli, capsys, config=config) == f"{config}: missing key {key}."


def test_a_missing_openai_key_stops_with_one_line_before_ros_init(cli, capsys, monkeypatch):
    # The real builder. The client refuses to exist without a key, so nothing is sent.
    monkeypatch.setattr(main, "build_model", build_model)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    assert "OPENAI_API_KEY" in startup_error(cli, capsys)


def test_web_search_without_a_tavily_key_stops_with_one_line_naming_it(cli, capsys, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    message = startup_error(cli, capsys, "--web-search")
    assert message.startswith("--web-search needs TAVILY_API_KEY")
    # Checked before either model is built.
    assert "build_model" not in cli.events


def test_cli_passes_a_tavily_search_only_with_web_search(cli, monkeypatch):
    received = []

    def assemble(**kwargs):
        received.append(kwargs)
        app = FakeApp(cli)
        app.on_start = send
        return app

    monkeypatch.setattr(main, "assemble", assemble)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key-not-real")
    assert cli.run() == 0
    assert received[-1]["web_search"] is None
    assert cli.run("--web-search") == 0
    assert isinstance(received[-1]["web_search"], TavilySearch)
    # The agent's model and the summarizer: build_model ran once for each.
    assert cli.events.count("build_model") == 3
    assert main._parse_args([]).web_search is False


def test_assemble_with_a_web_search_binds_search_web_and_its_rule(build, monkeypatch):
    built_with = {}
    real_build_agent = main.build_agent

    def recording_build_agent(model, ctx, **kwargs):
        built_with.update(kwargs)
        return real_build_agent(model, ctx, **kwargs)

    monkeypatch.setattr(main, "build_agent", recording_build_agent)
    web = UnavailableWebSearch()
    app = build(tool_set="gpsr", web_search=web).app
    assert app.ctx.web is web
    assert [fn.__name__ for fn in built_with["tools"]][-1] == "search_web"
    assert "`search_web`" in built_with["system_prompt"]

    plain = build(tool_set="gpsr").app
    assert isinstance(plain.ctx.web, UnavailableWebSearch)
    assert "search_web" not in [fn.__name__ for fn in built_with["tools"]]
    assert "search_web" not in built_with["system_prompt"]


def test_an_error_raised_by_ros_init_is_not_a_startup_error(cli, capsys, monkeypatch):
    # Only the reading before ROS starts ends in one line. From rclpy.init on, the traceback
    # stays. The partial start failure tests cover errors after a successful init.
    def fails(**_kwargs):
        raise ValueError("rcl could not start")

    monkeypatch.setattr(main.rclpy, "init", fails)
    with pytest.raises(ValueError, match="rcl could not start"):
        cli.run()
    assert "walkie-agent:" not in capsys.readouterr().err


def test_the_entry_point_ends_a_bad_team_table_with_one_line_and_code_2(tmp_path):
    """`python -m walkie_agent.main` in a fresh interpreter, as the operator runs it.

    The config is the openai vendor that the shipped config.toml uses, with a fake key and
    a local port, and no [team] table. The client is built and never sends anything.
    """
    config = tmp_path / "config.toml"
    config.write_text(MODEL_CONFIG + 'base_url = "http://127.0.0.1:1/v1"\n')
    robot = tmp_path / "robot.md"
    robot.write_text(ROBOT)
    env = {
        **os.environ,
        # This tree's src first, since the environment's editable install can point at
        # another checkout.
        "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
        "PYTHON_DOTENV_DISABLED": "1",
        "OPENAI_API_KEY": "test-key-not-real",
        # Off any domain a robot uses, should a regression reach rclpy.init.
        "ROS_DOMAIN_ID": "77",
    }
    command = [
        sys.executable,
        "-m",
        "walkie_agent.main",
        "--config",
        str(config),
        "--locations",
        str(FIXTURE_LOCATIONS),
        "--task-templates",
        str(tmp_path / "no_templates.toml"),
        "--robot-description",
        str(robot),
    ]
    done = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 2, done.stderr
    assert "Traceback" not in done.stderr
    assert done.stderr.splitlines() == [
        "walkie-agent: The config needs a [team] table with name, country, affiliation and timezone."
    ]
    assert done.stdout == ""


def test_sim_time_reaches_connector(cli):
    stop_when_started(cli)
    assert cli.run("--use-sim-time") == 0
    assert cli.connectors[0].kwargs == {"node_name": "walkie_agent", "use_sim_time": True}

    assert cli.run() == 0
    assert cli.connectors[1].kwargs == {"node_name": "walkie_agent", "use_sim_time": False}


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_ctrl_c_requests_cleanup_without_early_context_shutdown(cli, number):
    app = cli.use_fake_app(FakeApp(cli))
    before = signal.getsignal(number)

    def signalled():
        handler = signal.getsignal(number)
        assert handler != before, "main's handler is installed while the agent runs"
        send(number)
        wait_until(lambda: handler.__self__.count == 1)
        # The handler ran, and it did nothing but count.
        assert "close" not in cli.events
        assert "connector.shutdown" not in cli.events
        assert "try_shutdown" not in cli.events

    app.on_start = signalled
    assert cli.run() == 0

    assert cli.init_kwargs == [{"signal_handler_options": SignalHandlerOptions.NO}]
    assert cli.events[-5:] == ["connector", "start", "close", "connector.shutdown", "try_shutdown"]
    assert signal.getsignal(number) == before


def test_shutdown_orders_runner_goal_drain_connector_context(cli, monkeypatch):
    events = cli.events

    class RecordingRunner(Runner):
        def shutdown(self, timeout=None):
            ended = super().shutdown(timeout)
            if ended and "runner joined" not in events:
                events.append("runner joined")
                # Nav2 answers only now, so the drain must outlast the Runner's join.
                cli.nav[0].accept(
                    on_cancel=lambda handle, _future: handle.finish(GoalStatus.STATUS_CANCELED)
                )
            return ended

    monkeypatch.setattr(main, "Runner", RecordingRunner)
    cli.model.script = [ai(call("go_to", {"place": "kitchen"}, "drive"))]

    def started(app):
        client = cli.nav[0]
        client.log = events  # one timeline for the client's events and everything else
        client.on_send = send  # the signal arrives while the goal waits for its answer
        close_ingress = app.bridge.close_ingress
        app.bridge.close_ingress = lambda: (events.append("ingress closed"), close_ingress())
        cli.connectors[0].node.subscriptions[1].callback(HRIMessage(text="Go to the kitchen."))
        cli.app = app

    cli.on_started = started
    assert cli.run() == 0

    timeline = events[events.index("started") :]
    expected = [
        "send",
        "ingress closed",
        "runner joined",
        "cancel",
        "terminal",
        "destroy",
        "destroy /to_human",
        "destroy /from_human",
        "destroy /walkie/console",
        "destroy /walkie/todos",
        "connector.shutdown",
        "try_shutdown",
    ]
    positions = [timeline.index(event) for event in expected]
    assert positions == sorted(positions), timeline
    assert timeline[-2:] == ["connector.shutdown", "try_shutdown"]
    assert cli.app.ctx.cancel.cancelled, "the latch stays set for good"
    assert cli.nav[0].destroyed == 1


def test_partial_start_failure_with_no_connector_still_shuts_the_context(cli, monkeypatch, caplog):
    def fails(**_kwargs):
        raise RuntimeError("no middleware")

    monkeypatch.setattr(main, "ROS2Connector", fails)
    assert cli.run() == 1
    assert cli.events[-2:] == ["init", "try_shutdown"]
    assert "no middleware" in caplog.text


def raise_in(monkeypatch, name):
    def fails(*_args, **_kwargs):
        raise RuntimeError(f"{name} failed")

    monkeypatch.setattr(main, name, fails)


@pytest.mark.parametrize(
    ("stage", "built_before_it"),
    [
        ("RosBase", set()),
        ("RosSpeech", set()),
        ("RosBridge", {"/to_human", "/voice_commands"}),
        ("build_agent", {"/to_human", "/voice_commands", "/walkie/console", "/walkie/todos"}),
        ("Runner", {"/to_human", "/voice_commands", "/walkie/console", "/walkie/todos"}),
    ],
)
def test_partial_start_failure_cleans_up(cli, monkeypatch, stage, built_before_it):
    raise_in(monkeypatch, stage)
    assert cli.run() == 1

    node = cli.connectors[0].node
    assert topics(node.destroyed) == built_before_it
    assert topics(node.publishers + node.subscriptions) == built_before_it
    assert len(cli.nav) == (0 if stage == "RosBase" else 1)
    assert all(client.destroyed == 1 for client in cli.nav)
    assert cli.events[-2:] == ["connector.shutdown", "try_shutdown"]
    assert cli.events.count("connector.shutdown") == 1
    assert "started" not in cli.events


def test_a_failure_while_opening_the_input_closes_the_started_runner(cli, monkeypatch):
    real = FakeNode.create_subscription

    def fails_on_task_topic(self, msg_type, topic, callback, qos, **kwargs):
        if topic == "/walkie/task":
            raise RuntimeError("cannot subscribe")
        return real(self, msg_type, topic, callback, qos, **kwargs)

    monkeypatch.setattr(FakeNode, "create_subscription", fails_on_task_topic)
    assert cli.run() == 1

    node = cli.connectors[0].node
    assert "/from_human" in topics(node.destroyed)
    assert {"/walkie/console", "/walkie/todos", "/to_human"} <= topics(node.destroyed)
    assert cli.nav[0].destroyed == 1
    assert cli.events[-2:] == ["connector.shutdown", "try_shutdown"]


def test_unresolved_shutdown_keeps_executor_alive(cli, monkeypatch, caplog):
    """A goal with no answer holds the whole shutdown, after its run has ended. An answer frees it."""
    monkeypatch.setattr(main, "CLOSE_ATTEMPT_S", 0.0)
    events = cli.events
    cli.model.script = [ai(call("go_to", {"place": "kitchen"}, "drive"))]
    held = []
    real_close = main.Application.close

    def close(self, timeout_s):
        client = cli.nav[0]
        if held and not client.handles:
            # The operator has seen the report, and Nav2 answers at last.
            client.accept(
                on_cancel=lambda handle, _future: handle.finish(GoalStatus.STATUS_CANCELED)
            )
        finished = real_close(self, timeout_s)
        run_over = "The agent's current run has not ended." not in self.unresolved()
        if run_over and not client.handles:
            held.append(finished)
            # Only the goal is left: the latch holds, and ROS is all still there.
            assert self.ctx.cancel.cancelled
            assert client.destroyed == 0
            assert cli.connectors[0].node.destroyed == []
            assert "connector.shutdown" not in events and "try_shutdown" not in events
        return finished

    monkeypatch.setattr(main.Application, "close", close)

    def started(_app):
        cli.nav[0].on_send = send
        cli.connectors[0].node.subscriptions[1].callback(HRIMessage(text="Go to the kitchen."))

    cli.on_started = started
    with caplog.at_level(logging.WARNING, logger="walkie_agent.main"):
        assert cli.run() == 0

    assert held and not any(held), "the unanswered goal must have held the shutdown"
    assert "Shutdown is waiting" in caplog.text
    assert "Nav2 has not answered the goal" in caplog.text
    assert cli.nav[0].destroyed == 1
    assert events[-2:] == ["connector.shutdown", "try_shutdown"]


def test_a_changed_unresolved_list_is_reported_once_each(cli, monkeypatch, caplog):
    monkeypatch.setattr(main, "POLL_S", 0.0)
    app = cli.use_fake_app(FakeApp(cli, closes=(False, False, False, True)))
    app.on_start = send
    reports = iter((["The run."], ["The run."], ["The goal."]))
    app.unresolved = lambda: next(reports)
    with caplog.at_level(logging.WARNING, logger="walkie_agent.main"):
        assert cli.run() == 0
    assert caplog.text.count("Shutdown is waiting") == 2


def test_a_close_that_raises_keeps_ros_up_and_retries(cli, monkeypatch, caplog):
    monkeypatch.setattr(main, "POLL_S", 0.0)
    app = cli.use_fake_app(FakeApp(cli, closes=(True,)))
    app.on_start = send
    failures = [RuntimeError("drain failed")]

    def close_once_failing():
        if failures:
            assert "connector.shutdown" not in cli.events
            raise failures.pop()

    app.on_close = close_once_failing
    assert cli.run() == 0
    assert "drain failed" in caplog.text
    assert cli.events[-2:] == ["connector.shutdown", "try_shutdown"]


@pytest.mark.parametrize(
    "first",
    [signal.SIGINT, signal.SIGTERM],
    ids=["first-sigint", "first-sigterm"],
)
def test_second_signal_forces_exit_and_logs_what_is_unresolved(cli, caplog, first):
    app = cli.use_fake_app(FakeApp(cli, closes=(False,)))
    app.on_start = lambda: send(first)
    closes = []

    def second_on_the_first_attempt():
        closes.append(1)
        if len(closes) == 1:
            send(signal.SIGTERM if first == signal.SIGINT else signal.SIGINT)

    app.on_close = second_on_the_first_attempt
    with caplog.at_level(logging.ERROR, logger="walkie_agent.main"):
        with pytest.raises(_Exited) as forced:
            cli.run()

    assert forced.value.args == (main.FORCED_EXIT_CODE,)
    assert main.FORCED_EXIT_CODE != 0
    assert len(closes) == 1, "the fixture's watchdog forced this, not the second signal"
    assert "A second signal forced the exit" in caplog.text
    assert "the drive to kitchen: no answer." in caplog.text
    # A forced exit does not pretend: nothing of ROS was shut down under the live worker.
    assert "connector.shutdown" not in cli.events
    assert "try_shutdown" not in cli.events


def test_a_second_signal_is_not_a_failure_if_the_shutdown_finishes(cli):
    app = cli.use_fake_app(FakeApp(cli, closes=(True,)))

    def both():
        send()
        send()

    app.on_start = both
    assert cli.run() == 0
    assert cli.events[-2:] == ["connector.shutdown", "try_shutdown"]


def test_a_signal_within_the_repeat_window_of_the_last_counted_one_is_not_counted():
    now = [100.0]
    signals = main._Signals(clock=lambda: now[0])
    for at, count in [(100.0, 1), (100.01, 1), (100.49, 1), (100.5, 2), (100.9, 2), (101.0, 3)]:
        now[0] = at
        signals._handle(signal.SIGINT, None)
        assert signals.count == count, at


def test_a_signal_forwarded_by_pixi_does_not_force_the_exit(cli, monkeypatch, caplog):
    monkeypatch.setattr(main, "SIGNAL_REPEAT_S", 0.5)
    # The shutdown needs a second attempt, as when an arm goal has not ended yet.
    app = cli.use_fake_app(FakeApp(cli, closes=(False, True)))

    def as_pixi_delivers_it():
        # From the process group, and again from `pixi run`, which forwards it.
        send()
        send()

    app.on_start = as_pixi_delivers_it
    with caplog.at_level(logging.WARNING, logger="walkie_agent.main"):
        assert cli.run() == 0
    assert "Shutdown is waiting" in caplog.text
    assert "forced the exit" not in caplog.text
    assert cli.events[-2:] == ["connector.shutdown", "try_shutdown"]


def test_a_second_signal_after_the_repeat_window_forces_the_exit(cli, monkeypatch, caplog):
    monkeypatch.setattr(main, "SIGNAL_REPEAT_S", 0.05)
    app = cli.use_fake_app(FakeApp(cli, closes=(False,)))

    def twice_then_again():
        send()
        send()

    closes = []

    def again_after_the_window():
        closes.append(1)
        if len(closes) == 1:
            time.sleep(0.1)
            send()

    app.on_start = twice_then_again
    app.on_close = again_after_the_window
    with (
        caplog.at_level(logging.ERROR, logger="walkie_agent.main"),
        pytest.raises(_Exited) as forced,
    ):
        cli.run()
    assert forced.value.args == (main.FORCED_EXIT_CODE,)
    assert len(closes) == 1, "the fixture's watchdog forced this, not the second signal"
    assert "A second signal forced the exit" in caplog.text


def test_a_second_signal_ends_a_hung_connector_shutdown(cli, monkeypatch, caplog):
    app = cli.use_fake_app(FakeApp(cli))
    app.on_start = send
    release = threading.Event()

    def hung():
        send()
        release.wait(3 * WATCHDOG_S)

    def connector_that_hangs(**kwargs):
        connector = RecordingConnector(cli, **kwargs)
        connector.on_shutdown = hung
        return connector

    monkeypatch.setattr(main, "ROS2Connector", connector_that_hangs)
    with caplog.at_level(logging.ERROR, logger="walkie_agent.main"):
        with pytest.raises(_Exited) as forced:
            cli.run()
    release.set()
    # The shutdown thread outlives the forced exit, and ends with rclpy.try_shutdown().
    # It must finish while the fixture's stand-in for rclpy is in place: after the
    # fixture's teardown it would shut down the real context that later tests share.
    for thread in threading.enumerate():
        if thread.name == "walkie-ros-shutdown":
            thread.join(WATCHDOG_S)
            assert not thread.is_alive()
    assert "try_shutdown" in cli.events
    assert forced.value.args == (main.FORCED_EXIT_CODE,)
    assert "The ROS connector has not shut down." in caplog.text


def test_after_a_failed_start_one_signal_forces_the_exit(cli, caplog):
    app = cli.use_fake_app(FakeApp(cli, closes=(False,)))

    def start_fails():
        raise RuntimeError("no subscription")

    app.on_start = start_fails
    closes = []

    def signal_on_the_first_attempt():
        closes.append(1)
        if len(closes) == 1:
            send()

    app.on_close = signal_on_the_first_attempt
    with pytest.raises(_Exited):
        cli.run()
    assert len(closes) == 1, "the fixture's watchdog forced this, not the one signal"
    assert "connector.shutdown" not in cli.events
