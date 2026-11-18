"""Entry point: runs the Walkie agent as a ROS 2 node.

The agent reads tasks from /from_human (rai_interfaces/msg/HRIMessage, text only). What
it says out loud goes to /to_human through the `say` tool, where RAI's text-to-speech
agent picks it up. Its plain-text replies and one line per todo status change go to the
operator console topic, /walkie/console. Any message on /walkie/reset (std_msgs/msg/Empty)
makes it forget the conversation so far.

The operator can also send a task with its todo list seeded on /walkie/task
(std_msgs/msg/String holding JSON). The current todo list is published as JSON on
/walkie/todos (std_msgs/msg/String, latched). Task templates are read from
config/task_templates.toml when it exists.

The LLM is selected by config.toml, which is read from the current working directory by
default. API keys are read from the environment, or from a .env file in the working
directory (see .env.example). PYTHON_DOTENV_DISABLED=1 turns the .env file off. Its
[team] table names the team and the timezone that `get_facts` reads, with the time from
the computer's wall clock.

With --vision, `find_people` reads walkie-vision's people on /walkie_vision/people, and
`find_objects` its objects on /walkie_vision/objects, which the simulation's
oracle_perception publishes too. --objects-topic reads the objects elsewhere, such as
walkie-vision's own detector on the simulation's images. Without --vision, both report
UNAVAILABLE and the prompt says so.

--people-memory chooses where `remember_person` and `recognize_person` keep the people
the robot meets, from the faces and appearances that --vision collects (ADR-0070):
`unavailable`, the default, where both report UNAVAILABLE and the prompt says so;
`mock`, a store inside this process, for the simulation; or `ros`, walkie-database's
people services. With a memory, `find_people` also shows the names of the people in
view. A message on /walkie/reset ends the task, and forgets the people remembered in it.
--resume-people-session, with `ros` only, keeps the database's open session at the
first task instead, so an agent restarted mid-task still knows its guests. `mock` and
`ros` need --vision: a bad combination stops the agent at once, with exit code 2.

--object-memory binds `recall_object`, `list_room_objects` and `go_to_object`, which read
where walkie-database's object memory last saw objects (decisions SG-16 to SG-21). Every
set leaves them out without it. It needs --vision and a set with those atomics, default
or hri: a bad combination stops the agent at once, with exit code 2.

--arms connects the arms through walkie-ros's bimanual_commander (ADR-0021). It binds the
arm atomics of the chosen tool set, which every set leaves out without it, and go_to and
move_relative then refuse to drive with an arm out (ADR-0025). The poses come from
config/arm_poses.toml, checked against config/robot_model.toml. With --use-sim-time, the
poses not yet tried on the robot are offered too, and the commander's bounds are the
simulation's. --arms needs a set with the arm atomics, and the arms set needs --arms: a
bad combination stops the agent at once, with exit code 2.

--scripts adds the run_script tool to any set: the LLM writes a short Python script over
the bound tools, which runs in a Monty worker inside bwrap, and may save functions from
it for later tasks. The workers start before ROS does, and if bwrap cannot start them the
agent stops at once, with exit code 2. The saved functions are kept in
src/walkie_agent/tools, or in WALKIE_TOOLS_HOME when it is set. After a task, the operator
judges it on /walkie/verdict (std_msgs/msg/Bool): true verifies the saved functions that
the task proved, and false verifies nothing.

A missing or broken file, a missing setting or API key, or a bad [team] table stops the
agent before ROS starts, with one line on stderr and exit code 2.

A first SIGINT or SIGTERM starts an orderly shutdown. If that cannot finish, for example
because Nav2 has not answered a cancel, the agent keeps ROS running and keeps trying. A
second signal then ends the process at once, with exit code 3, and the goal's state unknown.
A signal within SIGNAL_REPEAT_S of the first is treated as the same one, since `pixi run`
forwards it.
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import FrameType, MappingProxyType
from typing import Any

import rclpy
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from rai.communication.ros2 import ROS2Connector
from rclpy.signals import SignalHandlerOptions

from walkie_agent.arm.poses import ArmPoses, load_arm_poses
from walkie_agent.capabilities import (
    Arms,
    CancelToken,
    Ctx,
    Facts,
    Memory,
    ObjectMemory,
    UnavailableArms,
    UnavailableFacts,
    UnavailableMemory,
    UnavailableObjectMemory,
)
from walkie_agent.capabilities.facts import facts_from_config
from walkie_agent.locations import Location, load_locations
from walkie_agent.perception import OBJECTS_TOPIC
from walkie_agent.perception.memory import TrackMemory
from walkie_agent.perception.people_memory import InMemoryPeopleMemory, PeopleMemory
from walkie_agent.ros.arms import RosArms
from walkie_agent.ros.base import RosBase
from walkie_agent.ros.commander import CommanderClient
from walkie_agent.ros.io import (
    CONSOLE_TOPIC,
    INPUT_TOPIC,
    RESET_TOPIC,
    TASK_TOPIC,
    TODOS_TOPIC,
    RosBridge,
)
from walkie_agent.ros.object_memory import RosObjectMemory
from walkie_agent.ros.people_memory import RosPeopleMemory
from walkie_agent.ros.speech import RosSpeech
from walkie_agent.ros.vision import RosVision, UnavailableVision
from walkie_agent.runtime import Runner, ToolLog, build_agent
from walkie_agent.runtime.model import (
    agent_settings,
    build_callbacks,
    build_fallback,
    build_model,
    load_config,
)
from walkie_agent.scripting.agent_tool import AgentScripts
from walkie_agent.scripting.sandbox import SandboxUnavailable
from walkie_agent.scripting.shelf import Shelf
from walkie_agent.tasks import TaskTemplate, check_required_tools, load_task_templates
from walkie_agent.tools.atomic import ARM_TOOLS, ARMS, DEFAULT, GPSR, HRI, OBJECT_MEMORY_TOOLS

logger = logging.getLogger(__name__)

NODE_NAME = "walkie_agent"
TOOL_SETS = {"default": DEFAULT, "gpsr": GPSR, "hri": HRI, "arms": ARMS}
# The set made for the arms. Without --arms its arm atomics are dropped, as from every
# set, and nothing it exists for is left, so a run without --arms refuses it.
ARMS_TOOL_SET = "arms"
# With --arms, the set must bind this one. The drive check's refusals and the client's
# unverified refusals tell the LLM to call move_arm_to_pose or get_arm_state, so a set
# without the arm atomics could not act on them.
ARM_TOOL = "move_arm_to_pose"
# The commander's bounds under --use-sim-time, in seconds of wall-clock time: the
# CommanderClient's keyword bounds, and RosArms' floor on every move's timeout. The
# simulation runs slower than real time, and every bound is wall-clock time, so each is
# raised over its robot default (decision S6-4). Provisional: the sim run measures the
# real-time factor and may change them.
SIM_ARM_BOUNDS: Mapping[str, float] = MappingProxyType(
    {
        "read_timeout_s": 6.0,
        "gripper_timeout_s": 15.0,
        "cancel_answer_s": 3.0,
        "abandon_after_s": 120.0,
        "absent_abandon_s": 10.0,
        "ready_wait_s": 3.0,
        "min_move_timeout_s": 60.0,
    }
)
# --people-memory's choices: none, a store in this process, or walkie-database.
PEOPLE_MEMORIES = ("unavailable", "mock", "ros")
# With --object-memory, the set must bind this one, so the flag never connects a
# memory that no tool reads.
OBJECT_MEMORY_TOOL = "recall_object"
# A todo step can require this one too. It cannot require update_todos, because the
# ledger that proves a step's work leaves that tool out.
WAIT_TOOL = "wait_for_person"

CLOSE_TIMEOUT_S = 5.0
# How long one shutdown attempt waits at each of its two stages. It also bounds how long
# a second signal goes unnoticed, so it stays short.
CLOSE_ATTEMPT_S = 1.0
POLL_S = 0.05
SIGNALS_TO_FORCE = 2
# A signal this soon after the last one counted is treated as the same signal. `pixi run`
# forwards the SIGINT or SIGTERM it gets to the agent, which also gets it from its process
# group, so one Ctrl+C or one `chula-sim.sh down` arrives twice within milliseconds.
SIGNAL_REPEAT_S = 0.5
FORCED_EXIT_CODE = 3

# What bad input raises while main() reads everything before ROS starts. ValueError
# includes tomllib.TOMLDecodeError, UnicodeDecodeError and pydantic's ValidationError.
# OSError is a missing or unreadable file, and KeyError a key that build_model looks up in
# config.toml. A single value where a table belongs, such as `vendor = "openai"`, raises
# TypeError or AttributeError. Those stay out, because they also mean a bug, which needs
# its traceback.
STARTUP_ERRORS = (OSError, ValueError, KeyError)
# The code argparse exits with for a bad command line.
STARTUP_ERROR_EXIT_CODE = 2

_RULES_START = """\
You are Walkie, a home service robot. You have a mobile base and can drive between named places.

Rules:
- To talk to a person in the room, use the `say` tool. Your plain-text replies
  go to the operator console only, never to the speaker.
- For short moves from where you are, such as "move forward one meter" or
  "turn around", use `move_relative`.
"""

# Kept only for a tool set that binds the key, so a prompt never names a missing tool.
_TOOL_RULES = {
    "list_places": "- Use `list_places` to see where you can go, then `go_to`.\n",
    "find_objects": (
        "- `find_objects` and `object_info` need live object perception, which is not\n"
        "  connected yet, so they report UNAVAILABLE. Say that you cannot see objects,\n"
        "  and never guess what is there.\n"
    ),
    "find_people": (
        "- `find_people` needs live people perception, which is not connected yet, so\n"
        "  it reports UNAVAILABLE. Say that you cannot see people, and never guess who\n"
        "  is there.\n"
    ),
    "face_person": (
        "- To look at someone, as in \"look at me\", find them with `find_people`, then\n"
        "  call `face_person` with their id. It turns you once and does not follow them.\n"
    ),
    "remember_person": (
        "- `remember_person` and `recognize_person` need a people memory, which is not\n"
        "  connected in this run, so they report UNAVAILABLE. Say that you cannot\n"
        "  remember people, and never pretend that you remembered or know someone.\n"
    ),
    "pick": (
        "- `pick`, `place_on` and `hand_over` need a grasp skill, which Walkie does not\n"
        "  have yet, so they report UNAVAILABLE. Say that you cannot do it, and never\n"
        "  claim you moved an object.\n"
    ),
    # A set binds the arm atomics only with --arms, so this rule is always live.
    # It does not tell the LLM to move the arms home before it drives: the drive
    # check refuses an arm out with words that say so (ADR-0025).
    ARM_TOOL: (
        "- You also have two arms with grippers. `list_arm_poses` lists the poses\n"
        "  `move_arm_to_pose` takes, and `set_gripper` opens and closes a hand.\n"
        "- Before you move an arm or a gripper, tell the people near you with `say`.\n"
        "- If an arm or gripper tool fails, call `get_arm_state`, then try again at most\n"
        "  once.\n"
        "- Never pick up an object by combining poses and gripper commands yourself.\n"
        "  Taking an object a person hands you is fine.\n"
    ),
    "get_facts": (
        "- For the time, the date, the day or your team's name, country or affiliation,\n"
        "  call `get_facts` and say what it returns. Never guess a fact.\n"
    ),
    # A set binds the object memory atomics only with --object-memory, so this rule
    # is always live too.
    OBJECT_MEMORY_TOOL: (
        "- `recall_object` tells where you saw objects before, and `list_room_objects`\n"
        "  what you saw in a room. They come from your memory, not from what is in view\n"
        "  now, so say it is where you last saw it. Their ids are memory ids, never ids\n"
        "  from `find_objects`. To fetch or show a remembered object, call `go_to_object`\n"
        "  with its id, then `find_objects` to check that it is still there. If they\n"
        "  report UNAVAILABLE, say that you cannot recall objects right now, and never\n"
        "  guess where something is.\n"
    ),
}

# The find_people rule of a run with --vision, in place of the one above.
_LIVE_PEOPLE_RULE = (
    "- `find_people` reports the people your camera sees now, left to right, each with\n"
    "  a pose and a gesture. It sees only what is in front of you, so turn to look\n"
    "  elsewhere. A raised hand also counts as waving. In a pointing gesture, left and\n"
    "  right are as you see them. An id names the same person only while they stay in\n"
    "  view. If it reports UNAVAILABLE, say that you cannot see people right now, and\n"
    "  never guess who is there.\n"
)

# The find_objects rule of a set without object_info, such as HRI, in place of the one above.
_FIND_OBJECTS_ONLY_RULE = (
    "- `find_objects` needs live object perception, which is not connected yet, so it\n"
    "  reports UNAVAILABLE. Say that you cannot see objects, and never guess what is\n"
    "  there.\n"
)

# The find_objects rule of a run with --vision, in place of both above, and the line it
# gains when the set binds object_info.
_LIVE_OBJECTS_RULE = (
    "- `find_objects` reports the objects your camera sees now, left to right, each\n"
    "  with a name, a category and the place you stand at. It sees only what is in\n"
    "  front of you, so turn or go to a place to look elsewhere. It finds only the\n"
    "  kinds of object your camera looks for, and says so when you ask for another\n"
    "  kind: then say that you cannot look for that, not that it is absent. An id\n"
    "  names the same object only while it stays in view. Never guess what is there.\n"
)
_LIVE_OBJECT_INFO_LINE = "  `object_info` cannot measure objects yet, so its measures are null.\n"

# The remember_person rule of a run with a people memory, in place of the one above.
_LIVE_MEMORY_RULE = (
    "- When someone tells you their name, call `remember_person` with their id from\n"
    "  `find_people`, their name and any facts they told you, while they are still in\n"
    "  view. `find_people` and `recognize_person` give the names of the people you\n"
    "  remembered. A null name is someone you do not know, so never guess it.\n"
)

# The run_script rule of a run with --scripts. Without the flag the prompt is unchanged.
_SCRIPTS_RULE = (
    "- `run_script` runs one short Python script that calls your other tools as\n"
    "  functions. Use it for several steps that need no decision from you in\n"
    "  between, such as visiting places in turn. When you must see a result before\n"
    "  you choose the next step, call the tools one at a time instead.\n"
    "- To keep a step you will need again, write it as a function whose changing\n"
    "  parts are parameters, and list its name in the save argument. Later scripts\n"
    "  may call the saved functions that the tool lists. One marked unverified may\n"
    "  be wrong, so check what it returns.\n"
    "- A todo step that names a tool is proven only by your own call of that tool,\n"
    "  never by a call inside a script.\n"
)

_RULES_END = """\
- If a tool reports a failure, tell the person what went wrong with `say`
  and do not pretend the action succeeded. When one call of a message fails,
  the later calls of that message do not run, so decide again.
- For a task with more than one step, write the steps with `update_todos`
  before you start. Mark a step in_progress when you start it, and completed
  only after its tool succeeded.
- If a step cannot be done, mark it failed with a note and tell the person
  with `say`.
- The <todos> block at the end of the messages is your current list. The
  system writes it, not the person.
- When you need an answer from the person to go on, for example which object
  they mean, ask with `wait_for_person` instead of `say`. Your turn ends,
  and the answer arrives as the next message.
- The system handles a bare "stop" before it reaches you. If a message asks
  you to stop and do something else, such as "stop and come here", drop what
  you were doing and do only the new request.
- Keep spoken sentences short and friendly. They are read out by a
  text-to-speech voice, so never use markdown, lists or code in them.
"""


def build_system_prompt(
    tool_set: str,
    locations: Mapping[str, Location],
    robot_description: str,
    *,
    vision: bool = False,
    memory: bool = False,
    arms: bool = False,
    object_memory: bool = False,
    scripts: bool = False,
) -> str:
    """The rules for the tools a run binds, then the static robot description, once.

    GPSR lacks `list_places`, so its prompt carries the configured places as data.
    `vision` says that find_people and find_objects read live perception, and `memory` that
    remember_person and recognize_person have a people memory. `arms` keeps the arm
    atomics in the set (bound_tools), and with them the arm rule; `object_memory` does
    the same for the object memory atomics and their rule. `scripts` adds the
    run_script rule after the tools' own.
    """
    names = _atomic_names(tool_set, arms=arms, object_memory=object_memory)
    rules = dict(_TOOL_RULES)
    measures = "object_info" in names
    if not measures:
        rules["find_objects"] = _FIND_OBJECTS_ONLY_RULE
    if vision:
        rules["find_people"] = _LIVE_PEOPLE_RULE
        rules["find_objects"] = _LIVE_OBJECTS_RULE + (_LIVE_OBJECT_INFO_LINE if measures else "")
    if memory:
        rules["remember_person"] = _LIVE_MEMORY_RULE
    prompt = _RULES_START
    prompt += "".join(rule for tool, rule in rules.items() if tool in names)
    if scripts:
        prompt += _SCRIPTS_RULE
    prompt += _RULES_END
    if "list_places" not in names:
        prompt += "\n" + _places_section(locations)
    if robot_description.strip():
        prompt += "\n" + robot_description.strip() + "\n"
    return prompt


def _places_section(locations: Mapping[str, Location]) -> str:
    if not locations:
        return "No named places are configured, so `go_to` has nowhere to go.\n"
    lines = [
        f"- {place.name}: {place.description}" if place.description else f"- {place.name}"
        for place in locations.values()
    ]
    return "The named places you can go to with `go_to`:\n" + "\n".join(lines) + "\n"


def bound_tools(
    tool_set: str, *, arms: bool = False, object_memory: bool = False
) -> list[Callable[..., Any]]:
    """The atomics a run binds: the set's, in its order, less the atomics of a missing flag.

    Without --arms ctx.arms has no commander, and without --object-memory there is no
    object memory, so their atomics would only report UNAVAILABLE: the arm atomics are
    dropped unless `arms`, and the object memory atomics unless `object_memory`.
    Raises ValueError for an unknown set.
    """
    if tool_set not in TOOL_SETS:
        known = ", ".join(TOOL_SETS)
        raise ValueError(f"There is no tool set {tool_set!r}. Known tool sets: {known}.")
    dropped = (set() if arms else ARM_TOOLS) | (set() if object_memory else OBJECT_MEMORY_TOOLS)
    return [fn for fn in TOOL_SETS[tool_set] if fn.__name__ not in dropped]


def _atomic_names(tool_set: str, *, arms: bool = False, object_memory: bool = False) -> set[str]:
    return {fn.__name__ for fn in bound_tools(tool_set, arms=arms, object_memory=object_memory)}


def check_templates(
    templates: Mapping[str, TaskTemplate],
    tool_set: str,
    *,
    arms: bool = False,
    object_memory: bool = False,
) -> None:
    """Raise ValueError for a template step that this run's tools could never complete."""
    names = _atomic_names(tool_set, arms=arms, object_memory=object_memory)
    check_required_tools(templates, names | {WAIT_TOOL})


def check_arms(tool_set: str, *, arms: bool) -> None:
    """Raise ValueError for a tool set that does not go with this run's arms.

    With `arms` the set must bind move_arm_to_pose (ARM_TOOL), and without them it must
    not be the arms set (ARMS_TOOL_SET). Raises ValueError for an unknown set as well.
    """
    names = _atomic_names(tool_set, arms=True)
    if arms and ARM_TOOL not in names:
        sets = " or ".join(
            name for name in TOOL_SETS if ARM_TOOL in _atomic_names(name, arms=True)
        )
        raise ValueError(
            f"--arms needs a tool set with the arm tools, such as {sets}: the {tool_set} "
            f"set has no {ARM_TOOL}, which the drive check's refusals ask for."
        )
    if not arms and tool_set == ARMS_TOOL_SET:
        raise ValueError(
            f"--tool-set {ARMS_TOOL_SET} needs --arms, which connects the arms its tools move."
        )


def check_people_memory(people_memory: str, *, vision: bool, resume: bool) -> None:
    """Raise ValueError for a people memory that this run could never use.

    A memory stores the samples that RosVision collects, so mock and ros need vision.
    Only ros has sessions, so only ros can resume one.
    """
    if people_memory not in PEOPLE_MEMORIES:
        known = ", ".join(PEOPLE_MEMORIES)
        raise ValueError(
            f"There is no people memory {people_memory!r}. Known people memories: {known}."
        )
    if people_memory != "unavailable" and not vision:
        raise ValueError(
            f"--people-memory {people_memory} needs --vision, which collects the faces "
            "and appearances it remembers people by."
        )
    if resume and people_memory != "ros":
        raise ValueError(
            "--resume-people-session needs --people-memory ros, the only memory with a "
            "session to resume."
        )


def check_object_memory(tool_set: str, *, object_memory: bool, vision: bool) -> None:
    """Raise ValueError for --object-memory in a run that could never use it.

    The set must bind recall_object (OBJECT_MEMORY_TOOL), and the run needs vision,
    since go_to_object's reply and the prompt send the LLM to find_objects to check
    that a remembered object is still there. Raises ValueError for an unknown set too.
    """
    if not object_memory:
        return
    if OBJECT_MEMORY_TOOL not in _atomic_names(tool_set, object_memory=True):
        sets = " or ".join(
            name
            for name in TOOL_SETS
            if OBJECT_MEMORY_TOOL in _atomic_names(name, object_memory=True)
        )
        raise ValueError(
            f"--object-memory needs a tool set with the object memory tools, such as "
            f"{sets}: the {tool_set} set has no {OBJECT_MEMORY_TOOL}."
        )
    if not vision:
        raise ValueError(
            "--object-memory needs --vision: go_to_object leaves the LLM to check with "
            "find_objects that a remembered object is still there."
        )


class Application:
    """One assembled agent. `start` opens it and `close` drains it, in that order only.

    The connector and the ROS context belong to the caller, and outlive `close`.
    `commander` is the arm commander client of a run with --arms, else None.
    `scripts` is run_script's sandbox of a run with --scripts, else None.
    """

    def __init__(
        self,
        ctx: Ctx,
        runner: Runner,
        bridge: RosBridge,
        base: RosBase,
        speech: RosSpeech,
        vision: RosVision | None = None,
        commander: CommanderClient | None = None,
        scripts: AgentScripts | None = None,
    ) -> None:
        self.ctx = ctx
        self.runner = runner
        self.bridge = bridge
        self._base = base
        self._speech = speech
        self._vision = vision
        self._commander = commander
        self._scripts = scripts
        self._run_ended = True

    def start(self) -> None:
        """Start the Runner, which publishes the empty list, then take input."""
        self.runner.start()
        self.bridge.open(self.runner)

    def close(self, timeout_s: float = CLOSE_TIMEOUT_S) -> bool:
        """Drain everything this application owns. True once nothing of it is left running.

        False means that a run or a goal has not ended within `timeout_s` at its stage.
        The endpoints it may still need are then alive, input stays closed and motion
        stays latched. Keep the executor spinning and call this again, because destroying
        ROS under a live worker could lose the cancel of a goal that is still driving.
        A repeated call is safe. The latch stays set for good.

        With arms, the commander's "stop" goes out at once, beside the base's. Once the
        base has closed, the commander client closes too: it waits for an arm goal that
        has not ended, such as one the commander accepts only after the stop, so that
        the goal's cancel and result still reach it. Until it has closed, close() returns
        False with speech, vision and the bridge still open.

        With --scripts, the script sandbox's workers close last, once the Runner's run,
        and with it any script, has ended. A worker that does not exit within
        `timeout_s` makes close() return False, and a later call waits for it again.
        """
        self.bridge.close_ingress()
        self.ctx.cancel.cancel()
        self._base.stop()
        if self._commander is not None:
            # The latch already ends a running arm call, which cancels its goal. "stop"
            # also halts a move whose call gave up on it earlier.
            self._commander.stop()
        self._run_ended = self.runner.shutdown(timeout_s)
        if not self._run_ended:
            return False
        if not self._base.close(timeout_s):
            return False
        # After the base, whose drive check reads the arms. A repeated close() of the
        # client after it finished does nothing.
        if self._commander is not None and not self._commander.close(timeout_s):
            return False
        # After the Runner's join, so a stop line in flight can still be published.
        self._speech.close()
        if self._vision is not None:
            self._vision.close()
        self.bridge.close()
        return self._scripts is None or self._scripts.close(timeout_s)

    def unresolved(self) -> list[str]:
        """What kept the last close() from finishing, as sentences for the operator's log."""
        worker = [] if self._run_ended else ["The agent's current run has not ended."]
        arms = [] if self._commander is None else self._commander.unresolved()
        scripts = [] if self._scripts is None else self._scripts.unresolved()
        return [*worker, *self._base.unresolved(), *arms, *scripts]


def assemble(
    *,
    connector: ROS2Connector,
    model: BaseChatModel,
    config: Mapping[str, Any],
    locations: Mapping[str, Location],
    templates: Mapping[str, TaskTemplate],
    robot_description: str,
    tool_set: str = "default",
    callbacks: Iterable[BaseCallbackHandler] = (),
    tool_log: ToolLog | None = None,
    input_topic: str = INPUT_TOPIC,
    output_topic: str = CONSOLE_TOPIC,
    reset_topic: str = RESET_TOPIC,
    task_topic: str = TASK_TOPIC,
    todos_topic: str = TODOS_TOPIC,
    use_sim_time: bool = False,
    fallback: BaseChatModel | None = None,
    facts: Facts = UnavailableFacts(),
    vision: bool = False,
    objects_topic: str = OBJECTS_TOPIC,
    people_memory: str = "unavailable",
    resume_people_session: bool = False,
    object_memory: bool = False,
    arm_poses: ArmPoses | None = None,
    scripts: AgentScripts | None = None,
) -> Application:
    """Wire the runtime and the ROS wrappers over a connector the caller already made.

    Nothing here builds a model or a tracer, initializes or spins ROS, or destroys the
    connector. A failure part of the way through closes what was built before it raises.
    `facts` is ctx.facts. main() passes the SystemFacts of config.toml's [team] table.
    `vision` makes ctx.vision a RosVision, which reads people and objects from
    walkie-vision, the objects on `objects_topic`.

    `people_memory` is one of PEOPLE_MEMORIES. With mock or ros, ctx.memory is a
    TrackMemory over the RosVision's samples and an InMemoryPeopleMemory or a
    RosPeopleMemory, which `resume_people_session` tells to keep the open session. The
    RosVision then names the people find_people reports, and the Runner forgets the
    people remembered at each reset. Raises ValueError for an unknown memory, a memory
    without `vision`, or a resume without ros, before anything is built.

    `object_memory` makes ctx.object_memory a RosObjectMemory on walkie-database's
    object services and binds the object memory atomics, which are dropped from the
    set without it (bound_tools). Raises ValueError for a set without them or a run
    without `vision` (check_object_memory) before anything is built.

    `arm_poses`, the checked poses that main() loads for --arms, gives the run its arms:
    a CommanderClient on the connector with the agent's token, made before the RosBase,
    and ctx.arms a RosArms over it, whose check_for_driving the RosBase runs before
    every drive. With `use_sim_time` the client takes SIM_ARM_BOUNDS, and RosArms
    offers the poses not yet tried on the robot. Without `arm_poses`, ctx.arms is
    UnavailableArms, the RosBase drives with no arm check, and the arm atomics are
    dropped from the set (bound_tools). Raises ValueError for a set that does not go
    with the arms (check_arms) before anything is built.

    `scripts`, the AgentScripts that main() opens for --scripts, adds run_script over
    the bound atomics and the run_script rule. The Runner marks each task's start and
    end on it, and the bridge subscribes to /walkie/verdict for its verdict. The
    Application closes it last. A failure here leaves it open for the caller.
    """
    arms = arm_poses is not None
    check_templates(templates, tool_set, arms=arms, object_memory=object_memory)
    check_arms(tool_set, arms=arms)
    check_people_memory(people_memory, vision=vision, resume=resume_people_session)
    check_object_memory(tool_set, object_memory=object_memory, vision=vision)
    remembers = people_memory != "unavailable"
    prompt = build_system_prompt(
        tool_set,
        locations,
        robot_description,
        vision=vision,
        memory=remembers,
        arms=arms,
        object_memory=object_memory,
        scripts=scripts is not None,
    )
    clock = connector.node.get_parameter("use_sim_time").value
    if bool(clock) != use_sim_time:
        raise ValueError(
            f"The connector has use_sim_time={clock}, but this run asks for {use_sim_time}. "
            "Build the connector with the clock you want: a clock changed later would "
            "invalidate the stamps already in use."
        )

    cancel = CancelToken()
    built: list[Callable[[], Any]] = []
    try:
        commander: CommanderClient | None = None
        ctx_arms: Arms = UnavailableArms()
        arms_check: Callable[[str, float], None] | None = None
        if arm_poses is not None:
            bounds = dict(SIM_ARM_BOUNDS) if use_sim_time else {}
            min_move_timeout_s = bounds.pop("min_move_timeout_s", 0.0)
            # The same token as the RosBase and the Runner, so a stop or a reset ends
            # an arm call and its drive check too. Made first, since the RosBase checks
            # the arms through it.
            commander = CommanderClient(connector, cancel, arm_poses, **bounds)
            built.append(commander.destroy)
            ros_arms = RosArms(
                commander, cancel, sim=use_sim_time, min_move_timeout_s=min_move_timeout_s
            )
            ctx_arms, arms_check = ros_arms, ros_arms.check_for_driving
        base = RosBase(connector, locations, cancel, arms_check=arms_check)
        built.append(base.close)
        speech = RosSpeech(connector)
        built.append(speech.close)
        live = RosVision(connector, base, cancel, objects_topic=objects_topic) if vision else None
        if live is not None:
            built.append(live.close)
        memory: Memory = UnavailableMemory()
        on_reset = None
        if live is not None and remembers:
            store: PeopleMemory = (
                InMemoryPeopleMemory(cancel)
                if people_memory == "mock"
                else RosPeopleMemory(connector, cancel, resume_session=resume_people_session)
            )
            tracks = TrackMemory(live, store, cancel)
            live.set_namer(tracks.names)
            memory, on_reset = tracks, tracks.clear
        objects: ObjectMemory = (
            RosObjectMemory(connector, cancel) if object_memory else UnavailableObjectMemory()
        )
        ctx = Ctx(
            cancel=cancel,
            base=base,
            speech=speech,
            vision=live or UnavailableVision(cancel),
            facts=facts,
            memory=memory,
            arms=ctx_arms,
            object_memory=objects,
        )
        bridge = RosBridge(
            connector,
            templates,
            input_topic=input_topic,
            output_topic=output_topic,
            reset_topic=reset_topic,
            task_topic=task_topic,
            todos_topic=todos_topic,
            on_verdict=None if scripts is None else scripts.verdict,
        )
        built.append(bridge.close)
        atomics = bound_tools(tool_set, arms=arms, object_memory=object_memory)
        agent = build_agent(
            model,
            ctx,
            tools=atomics,
            extra_tools=() if scripts is None else (scripts.tool(atomics, ctx),),
            on_todos=bridge.on_todos,
            fallback=fallback,
            tool_log=tool_log,
            system_prompt=prompt,
            **agent_settings(config),
        )
        runner = Runner(
            agent,
            ctx,
            on_todos=bridge.on_todos,
            on_reply=bridge.on_reply,
            callbacks=callbacks,
            tool_log=tool_log,
            on_reset=on_reset,
            # A stop also halts an arm goal whose call already gave up (ADR-0021).
            on_stop=None if commander is None else commander.stop,
            on_task_start=None if scripts is None else scripts.begin,
            on_task_end=None if scripts is None else scripts.end,
        )
    except BaseException:
        for close in reversed(built):
            close()
        raise
    return Application(ctx, runner, bridge, base, speech, live, commander, scripts)


class _Signals:
    """Counts SIGINT and SIGTERM. The main thread polls the count.

    The handler only reads `clock` and adds one, and a signal within SIGNAL_REPEAT_S of
    the last one counted adds nothing. It runs on the main thread between two of its
    bytecodes, so anything that takes a lock, threading.Event.set included, could
    deadlock against the code it interrupted.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.count = 0
        self._clock = clock
        self._counted_at: float | None = None
        self._previous: dict[int, Any] = {}

    def install(self) -> None:
        for number in (signal.SIGINT, signal.SIGTERM):
            self._previous[number] = signal.signal(number, self._handle)

    def restore(self) -> None:
        for number, handler in self._previous.items():
            signal.signal(number, handler)

    def _handle(self, _number: int, _frame: FrameType | None) -> None:
        now = self._clock()
        if self._counted_at is not None and now - self._counted_at < SIGNAL_REPEAT_S:
            return
        self._counted_at = now
        self.count += 1


def _force_exit(unresolved: Iterable[str]) -> None:
    """End the process now. Only os._exit can: the connector's spin thread is not a daemon."""
    logger.error(
        "A second signal forced the exit with the shutdown unfinished, so the robot's "
        "state is unknown. Unresolved: %s",
        " ".join(unresolved) or "nothing was reported.",
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(FORCED_EXIT_CODE)


def _close_application(app: Application, signals: _Signals) -> None:
    """Retry close() until it finishes. Only a second signal ends this early."""
    reported: list[str] = []
    while True:
        try:
            if app.close(CLOSE_ATTEMPT_S):
                return
        except Exception:
            logger.exception("Shutting down the agent failed. ROS stays up to retry.")
            time.sleep(POLL_S)
        unresolved = app.unresolved()
        if unresolved != reported:
            logger.warning(
                "Shutdown is waiting, with ROS still running and input closed. "
                "A second signal forces the exit. Unresolved: %s",
                " ".join(unresolved) or "see the error above.",
            )
            reported = unresolved
        if signals.count >= SIGNALS_TO_FORCE:
            _force_exit(unresolved)


def _shut_down_ros(connector: ROS2Connector | None, signals: _Signals) -> None:
    """Destroy the node and the context, on a thread so a second signal can still end it.

    The connector's shutdown joins its spin thread with no time limit.
    """

    def finish() -> None:
        try:
            if connector is not None:
                connector.shutdown()
        except Exception:
            logger.exception("Could not shut down the ROS connector.")
        finally:
            rclpy.try_shutdown()

    thread = threading.Thread(target=finish, name="walkie-ros-shutdown", daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(POLL_S)
        if thread.is_alive() and signals.count >= SIGNALS_TO_FORCE:
            _force_exit(["The ROS connector has not shut down."])


def _run(args: argparse.Namespace, **assembly: Any) -> int:
    """Own the ROS context and the signals for one run. Returns the exit code."""
    signals = _Signals()
    # NO keeps rclpy's own SIGINT handler out, since it would shut the context down
    # before the cancel of a running Nav2 goal could be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signals.install()
    connector, app, code = None, None, 0
    try:
        connector = ROS2Connector(node_name=NODE_NAME, use_sim_time=args.use_sim_time)
        app = assemble(
            connector=connector,
            tool_set=args.tool_set,
            input_topic=args.input_topic,
            output_topic=args.output_topic,
            reset_topic=args.reset_topic,
            task_topic=args.task_topic,
            todos_topic=args.todos_topic,
            use_sim_time=args.use_sim_time,
            vision=args.vision,
            objects_topic=args.objects_topic,
            people_memory=args.people_memory,
            resume_people_session=args.resume_people_session,
            object_memory=args.object_memory,
            **assembly,
        )
        app.start()
        logger.info("Walkie's agent is running, with the %s tool set.", args.tool_set)
        while not signals.count:
            time.sleep(POLL_S)
        logger.info("Shutting down.")
    except Exception:
        logger.exception("The agent failed, so it is shutting down.")
        code = 1
    # A failure counts as the first request to shut down, so the next signal forces the exit.
    signals.count = max(signals.count, 1)
    try:
        if app is not None:
            _close_application(app, signals)
        _shut_down_ros(connector, signals)
    finally:
        signals.restore()
    return code


def _startup_message(error: Exception, config_path: str) -> str:
    """The one line that main() prints for one of STARTUP_ERRORS."""
    if isinstance(error, KeyError):
        # Its str() is the quoted key alone. Here only config.toml lookups raise it.
        text = f"{config_path}: missing key {error}."
    else:
        text = str(error)
    # A message can span several lines, as pydantic's ValidationError does. Its lines are
    # joined with spaces, so the operator still gets one line.
    line = " ".join(part.strip() for part in text.splitlines() if part.strip())
    return line or type(error).__name__


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.toml", help="The model and tracing settings.")
    parser.add_argument("--locations", default="config/locations.toml")
    parser.add_argument(
        "--task-templates",
        default="config/task_templates.toml",
        help="Task templates; a missing file means none.",
    )
    parser.add_argument(
        "--robot-description",
        default="robot.md",
        help="The robot's static description, appended to the system prompt.",
    )
    parser.add_argument(
        "--tool-set", choices=sorted(TOOL_SETS), default="default", help="The tools the LLM gets."
    )
    parser.add_argument("--tool-log", help="Append one JSON line per tool call to this file.")
    parser.add_argument("--input-topic", default=INPUT_TOPIC)
    parser.add_argument("--output-topic", default=CONSOLE_TOPIC)
    parser.add_argument("--reset-topic", default=RESET_TOPIC)
    parser.add_argument("--task-topic", default=TASK_TOPIC)
    parser.add_argument("--todos-topic", default=TODOS_TOPIC)
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        help="Use the simulator's /clock, so goal timestamps match the simulated robot.",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Let find_people and find_objects read walkie-vision's people and objects, or "
        "the simulation's oracle.",
    )
    parser.add_argument(
        "--objects-topic",
        default=OBJECTS_TOPIC,
        help="Where --vision reads the objects, such as walkie-vision's "
        "/walkie_vision/sim/objects in the simulation. Needs --vision.",
    )
    parser.add_argument(
        "--people-memory",
        choices=PEOPLE_MEMORIES,
        default="unavailable",
        help="Where remember_person keeps people: nowhere, in this process, or "
        "walkie-database. mock and ros need --vision.",
    )
    parser.add_argument(
        "--resume-people-session",
        action="store_true",
        help="With --people-memory ros, keep the database's open session at the first "
        "task instead of forgetting it.",
    )
    parser.add_argument(
        "--object-memory",
        action="store_true",
        help="Bind recall_object, list_room_objects and go_to_object, which read the "
        "objects walkie-database's object memory keeps. Needs --vision and a tool set "
        "with those tools, such as default or hri.",
    )
    parser.add_argument(
        "--arms",
        action="store_true",
        help="Drive the arms through walkie-ros's bimanual_commander: binds the arm tools, "
        "and go_to and move_relative then refuse to drive with an arm out. With "
        "--use-sim-time, poses not yet tried on the robot are offered too. Needs a tool "
        "set with the arm tools, such as hri or arms.",
    )
    parser.add_argument(
        "--scripts",
        action="store_true",
        help="Add run_script, which runs a short Python script over the bound tools inside "
        "bwrap and keeps the functions it saves. Needs bwrap. The operator judges each "
        "task on /walkie/verdict.",
    )
    args = parser.parse_args(argv)
    try:
        check_arms(args.tool_set, arms=args.arms)
        check_people_memory(
            args.people_memory, vision=args.vision, resume=args.resume_people_session
        )
        check_object_memory(args.tool_set, object_memory=args.object_memory, vision=args.vision)
        if args.objects_topic != OBJECTS_TOPIC and not args.vision:
            raise ValueError("--objects-topic needs --vision, which reads the objects.")
    except ValueError as error:
        # Exit code 2 and the usage line, as for any other bad command line.
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    # Variables already set in the shell take precedence over .env.
    load_dotenv()

    # All of this runs before ROS starts, so a bad file or credential fails fast, in one
    # line. A failure once ROS has started, in _run, keeps its traceback.
    try:
        config = load_config(args.config)
        locations = load_locations(args.locations)
        templates = load_task_templates(args.task_templates)
        check_templates(
            templates, args.tool_set, arms=args.arms, object_memory=args.object_memory
        )
        # config/arm_poses.toml and config/robot_model.toml, from the working directory.
        # ArmPoseError is a ValueError, so a bad pose stops the agent here, in one line.
        arm_poses = load_arm_poses() if args.arms else None
        robot_description = Path(args.robot_description).read_text(encoding="utf-8")
        model = build_model(config)
        fallback = build_fallback(config)
        callbacks = build_callbacks(config)
        facts = facts_from_config(config)
        tool_log = ToolLog(args.tool_log) if args.tool_log else None
        # WALKIE_TOOLS_HOME or the tools package's folder. StoreRefused is a ValueError,
        # so a store that git does not ignore stops the agent here, in one line.
        shelf = Shelf() if args.scripts else None
    except STARTUP_ERRORS as error:
        print(f"walkie-agent: {_startup_message(error, args.config)}", file=sys.stderr)
        return STARTUP_ERROR_EXIT_CODE

    # From the main thread, which outlives every run: a worker dies with the thread
    # that started it. Without bwrap no script may run, so the agent does not start.
    scripts = None
    if shelf is not None:
        try:
            scripts = AgentScripts.open(shelf, tool_log=tool_log)
        except SandboxUnavailable as error:
            print(f"walkie-agent: {_startup_message(error, args.config)}", file=sys.stderr)
            return STARTUP_ERROR_EXIT_CODE
    try:
        return _run(
            args,
            model=model,
            config=config,
            locations=locations,
            templates=templates,
            robot_description=robot_description,
            callbacks=callbacks,
            tool_log=tool_log,
            fallback=fallback,
            facts=facts,
            arm_poses=arm_poses,
            scripts=scripts,
        )
    finally:
        # Application.close closed it already, unless the run failed before assembling.
        if scripts is not None and not scripts.close(CLOSE_TIMEOUT_S):
            logger.warning("The script sandbox's workers did not close; the exit ends them.")


if __name__ == "__main__":
    sys.exit(main())
