"""Atomics: plain typed functions over ctx, the catalogue of every tool the LLM can get.

An atomic takes ctx first, returns a JSON-able value, and raises StepError on a
failure. Its Google-style docstring is what the LLM reads, so the return value
is described in the summary: LangChain drops a Returns section from a tool's
description. Tool messages and Monty scripts call the same functions, through
walkie_agent.tools.adapter. Which atomics each run binds is in walkie_agent.tools.sets.
"""

from walkie_agent.tools.atomic.arms import get_arm_state, list_arm_poses, move_arm_to_pose, set_gripper
from walkie_agent.tools.atomic.facts import get_facts
from walkie_agent.tools.atomic.grasp import pick_object
from walkie_agent.tools.atomic.manipulation import hand_over, pick, place_on
from walkie_agent.tools.atomic.memory import (
    list_room_objects,
    recall_object,
    recognize_person,
    remember_person,
)
from walkie_agent.tools.atomic.navigation import (
    face_person,
    go_to,
    go_to_object,
    list_places,
    move_relative,
)
from walkie_agent.tools.atomic.perception import (
    describe_objects,
    find_objects,
    find_people,
    object_info,
)
from walkie_agent.tools.atomic.speech import say
from walkie_agent.tools.atomic.web import search_web

__all__ = [
    "describe_objects",
    "face_person",
    "find_objects",
    "find_people",
    "get_arm_state",
    "get_facts",
    "go_to",
    "go_to_object",
    "hand_over",
    "list_arm_poses",
    "list_places",
    "list_room_objects",
    "move_arm_to_pose",
    "move_relative",
    "object_info",
    "pick",
    "pick_object",
    "place_on",
    "recall_object",
    "recognize_person",
    "remember_person",
    "say",
    "search_web",
    "set_gripper",
]
