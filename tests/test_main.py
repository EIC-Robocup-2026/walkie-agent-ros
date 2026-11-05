"""The assembly and the process lifecycle of walkie_agent.main.

The ROS lifecycle and the model factories are patched and their calls inspected, so
nothing here needs a live graph, credentials or the network. Signals are real, sent to
this process, which is why `main` runs on the test's main thread.
"""

import logging
import os
import re
import signal
import threading
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, SystemMessage
from rai_interfaces.msg import HRIMessage
from rclpy.signals import SignalHandlerOptions
from ros_fakes import WATCHDOG_S, FakeActionClient, FakeConnector, FakeNode
from runtime_harness import ScriptedModel, ai, call, wait_until
from std_msgs.msg import Empty

import walkie_agent.main as main
from walkie_agent.atomics import DEFAULT, GPSR, HRI, get_facts, list_places
from walkie_agent.capabilities import Team, UnavailableArms, UnavailableFacts
from walkie_agent.capabilities.facts import SystemFacts
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import Location, load_locations
from walkie_agent.ros import RosBase, RosSpeech, UnavailableVision
from walkie_agent.runtime import Runner, ToolLog
from walkie_agent.tasks import TaskTemplate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DONE = AIMessage("Done.")
PLACES = {
    "kitchen": Location("kitchen", 3.5, 1.2, 1.57, "Kitchen counter."),
    "door": Location("door", 1.0, 0.0, 0.0),
}
ROBOT = "Walkie is a test robot with a unique marker 7f3a."
MIDDLEWARE_TOOLS = {"update_todos", "wait_for_person"}
TOOL_SET_LISTS = {"default": DEFAULT, "gpsr": GPSR, "hri": HRI}
FIXTURE_LOCATIONS = PROJECT_ROOT / "tests" / "fixtures" / "locations.toml"
MODEL_CONFIG = '[vendor]\ncomplex_model = "openai"\n[openai]\ncomplex_model = "m"\n'
TEAM_CONFIG = (
    '[team]\nname = "EIC"\ncountry = "Thailand"\naffiliation = "Chulalongkorn University"\n'
    'timezone = "Asia/Bangkok"\n'
)


def tool_names(tool_set):
    return {fn.__name__ for fn in TOOL_SET_LISTS[tool_set]} | MIDDLEWARE_TOOLS


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
    system = built.model.received[0][0]
    assert isinstance(system, SystemMessage)
    named = set(re.findall(r"`(\w+)`", system.content))
    assert named <= bound, f"The prompt names tools it lacks: {named - bound}"
    assert {"say", "go_to", "move_relative", "update_todos", "wait_for_person"} <= named

    # A prompt says plainly that what its set offers is not connected yet.
    assert "UNAVAILABLE" in system.content
    assert ("`find_objects`" in system.content) == (tool_set != "hri")
    assert ("`pick`" in system.content) == (tool_set == "gpsr")
    assert "`find_people`" in system.content
    assert ("`get_facts`" in system.content) == (tool_set != "hri")

    # GPSR cannot list places, so its prompt holds them as data. The others look them up.
    assert ("- kitchen: Kitchen counter." in system.content) == (tool_set == "gpsr")
    assert ("- door\n" in system.content) == (tool_set == "gpsr")
    assert ("`list_places`" in system.content) == (tool_set != "gpsr")


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
    empty = main.build_system_prompt("default", PLACES, " \n")
    assert empty == main.build_system_prompt("default", PLACES, "")
    assert system.startswith(empty)
    assert "whoami" not in Path(main.__file__).read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        main._parse_args(["--whoami-dir", "whoami"])


def test_the_shipped_robot_description_claims_only_what_is_connected():
    text = (PROJECT_ROOT / "robot.md").read_text(encoding="utf-8")
    assert "Chulalongkorn" in text and "polite, patient and brief" in text
    assert "Two robot arms" in text
    assert "not connected" in text
    # It describes the robot. How to use a tool is the prompt's job, set by set. Some tool
    # names are plain English ("pick up"), so only the unmistakable ones are looked for.
    assert "`" not in text
    for name in tool_names("default") | tool_names("gpsr") | tool_names("hri"):
        assert "_" not in name or name not in text, name
    # It claims the facts get_facts reads, and names the team as config.toml does.
    assert "the time and the date from the computer's clock" in text
    team = main.load_config(PROJECT_ROOT / "config.toml")["team"]
    for key in ("name", "country", "affiliation"):
        assert team[key] in text, key


def test_template_requirements_must_be_bound(build):
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
    for tool_set in TOOL_SET_LISTS:
        asks = {"tour": template("Plain", {"content": "Ask", "requires_tool": "wait_for_person"})}
        build(tool_set=tool_set, templates=asks)


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
        monkeypatch.setattr(f"walkie_agent.runtime.model.{name}", never)
    monkeypatch.setattr("walkie_agent.runtime.model._chat_model", never)
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

    # A stuck main thread would hang the whole run. Two signals force it out as a failure.
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    watchdog = threading.Timer(3 * WATCHDOG_S, lambda: (send(signal.SIGTERM), send(signal.SIGTERM)))
    watchdog.daemon = True
    watchdog.start()
    yield state
    watchdog.cancel()
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
    )

    assert code == 0
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
    assert main._parse_args(["--tool-set", "hri"]).tool_set == "hri"
    with pytest.raises(SystemExit):
        main._parse_args(["--tool-set", "garage"])
    with pytest.raises(SystemExit):
        main._parse_args(["--fake-model"])


def test_models_and_files_are_ready_before_ros_starts(cli):
    stop_when_started(cli)
    assert cli.run() == 0
    assert cli.events[:5] == ["dotenv", "build_model", "build_fallback", "build_callbacks", "init"]


def broken_file(cli, text, name="broken.toml"):
    path = cli.tmp_path / name
    path.write_text(text)
    return str(path)


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
        "no-robot-description",
    ],
)
def test_config_errors_happen_before_ros_init(cli, monkeypatch, flag, text, extra):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    path = str(cli.tmp_path / "missing") if text is None else broken_file(cli, text)
    with pytest.raises((OSError, ValueError)):
        cli.run(*extra, **{flag: path})
    assert "init" not in cli.events
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
def test_a_bad_team_table_stops_with_one_line_before_ros_init(cli, text, match):
    with pytest.raises(ValueError, match=match) as raised:
        cli.run(config=broken_file(cli, text))
    assert "\n" not in str(raised.value)
    assert "init" not in cli.events


def test_a_model_that_cannot_be_built_stops_before_ros_init(cli, monkeypatch):
    def no_key(_config):
        raise ValueError("no API key")

    monkeypatch.setattr(main, "build_model", no_key)
    with pytest.raises(ValueError, match="no API key"):
        cli.run()
    assert "init" not in cli.events


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
