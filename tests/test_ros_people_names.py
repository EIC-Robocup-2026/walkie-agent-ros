"""find_people's names: RosVision, TrackMemory and the memory atomics together.

RosVision collects the samples of each track and TrackMemory stores and matches them
in an InMemoryPeopleMemory, as `--people-memory mock` runs them. The first test feeds
RosVision over tests/ros_fakes.py's FakeConnector. The second publishes the oracle's
arrays of config/scenes/chula_living_room.toml on Fast DDS, as the simulation does,
and reads them back through a real connector.
"""

import random
import time

import numpy as np
import pytest
import rclpy
from ros_fakes import WATCHDOG_S, FakeTfBuffer
from test_ros_vision import QUIET, Rig, StubBase, person, with_vectors
from test_scene_file import CHULA_LOCATIONS, CHULA_SCENE, STREAM_ID, camera_info, living_room_camera
from walkie_vision_msgs.msg import PersonDetectionArray

from walkie_agent.atomics import find_people, recognize_person, remember_person
from walkie_agent.capabilities import CancelToken, Ctx
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.locations import load_locations
from walkie_agent.perception import PEOPLE_TOPIC
from walkie_agent.perception.memory import TrackMemory
from walkie_agent.perception.people_memory import InMemoryPeopleMemory
from walkie_agent.ros import RosVision
from walkie_agent.sim.oracle_perception import ORACLE_QOS, draw_detections
from walkie_agent.sim.scene_file import load_scene


class Silent:
    """ctx.speech, which no memory tool uses."""

    def say(self, text: str, *, timeout_s: float = 10.0) -> None:
        raise AssertionError(f"Nothing here speaks, but {text!r} was said.")


def remembering(vision: RosVision, base, cancel) -> tuple[Ctx, TrackMemory, InMemoryPeopleMemory]:
    """A Ctx whose memory is a TrackMemory over `vision`, wired as main wires --people-memory mock."""
    store = InMemoryPeopleMemory()
    memory = TrackMemory(vision, store, cancel)
    vision.set_namer(memory.names)
    ctx = Ctx(cancel=cancel, base=base, speech=Silent(), vision=vision, memory=memory)
    return ctx, memory, store


def names(found: list[dict]) -> dict[str, str | None]:
    return {person["id"]: person["name"] for person in found}


def test_a_guest_who_turns_away_and_comes_back_on_a_new_track_is_found_by_name():
    made = Rig()
    try:
        ctx, memory, store = remembering(made.vision, made.base, made.cancel)
        made.deliver(
            with_vectors(person(100, track_id=1), face="sam", appearance="sam"),
            with_vectors(person(300, track_id=2), face="kim", appearance="kim"),
        )
        assert names(find_people(ctx)) == {"person_1": None, "person_2": None}
        assert remember_person(ctx, "person_1", "Sam", {"drink": "tea"}) == {
            "id": "person_1",
            "name": "Sam",
            "facts": {"drink": "tea"},
            "faces": 1,
            "appearances": 1,
        }

        # Sam turns away: the newest array has no face of theirs, the track's samples still do.
        made.advance(0.1)
        made.deliver(
            with_vectors(person(100, track_id=1), appearance="sam"),
            with_vectors(person(300, track_id=2), face="kim", appearance="kim"),
        )
        assert [p["id"] for p in find_people(ctx, "sam")] == ["person_1"]

        # Sam leaves and comes back facing away, on a new track: only the appearance names them.
        made.advance(0.1)
        made.deliver(
            with_vectors(person(150, track_id=5), appearance="sam"),
            with_vectors(person(300, track_id=2), face="kim", appearance="kim"),
        )
        assert names(find_people(ctx)) == {"person_5": "Sam", "person_2": None}
        assert recognize_person(ctx, "person_5") == {
            "id": "person_5",
            "name": "Sam",
            "facts": {"drink": "tea"},
            "matched_by": "appearance",
        }
        assert recognize_person(ctx, "person_2")["name"] is None

        # A reset, the end of the task, forgets Sam.
        memory.clear()
        assert len(store) == 0
        assert find_people(ctx, "sam") == []
    finally:
        made.vision.close()


def test_a_guest_remembered_over_dds_is_found_by_name_in_a_fresh_array(connector):
    """The Chula guest is remembered as Sam and found by that name; the host stays unknown."""
    loaded = load_scene(CHULA_SCENE)
    places = load_locations(CHULA_LOCATIONS)
    place = places["living room"]
    base = StubBase(place.x, place.y, place.yaw)
    base.places_by_name = places
    cancel = CancelToken()
    vision = RosVision(connector, base, cancel, tf_buffer=FakeTfBuffer())
    ctx, _, _ = remembering(vision, base, cancel)
    [guest] = [who for who in loaded.scene.people if who.face == "guest1"]
    [host] = [who for who in loaded.scene.people if who.face == "host"]
    guest_id, host_id = f"person_{guest.track_id}", f"person_{host.track_id}"
    publisher_node = rclpy.create_node(f"oracle_stand_in_{random.randint(0, 1_000_000)}")
    publisher = publisher_node.create_publisher(PersonDetectionArray, PEOPLE_TOPIC, ORACLE_QOS)
    rng = np.random.default_rng(0)

    def publish_until(check):
        """Publish one oracle array at a time until `check()` is true, as discovery takes a while."""
        deadline = time.monotonic() + WATCHDOG_S
        while time.monotonic() < deadline:
            info = camera_info()
            info.header.stamp = connector.node.get_clock().now().to_msg()
            # No pixel noise, so the gestures do not depend on how many images discovery took.
            people, _ = draw_detections(
                loaded.scene, info, living_room_camera(), QUIET, rng, stream_id=STREAM_ID
            )
            publisher.publish(people)
            try:
                if check():
                    return
            except StepError as error:
                assert error.code is ErrorCode.UNAVAILABLE, error
        pytest.fail("The oracle's arrays did not arrive in time.")

    try:
        publish_until(lambda: len(vision.people(timeout_s=0.2)) == 2)
        assert names(find_people(ctx)) == {guest_id: None, host_id: None}

        remembered = remember_person(ctx, guest_id, "Sam", {"drink": "tea"})
        assert (remembered["id"], remembered["name"], remembered["facts"]) == (
            guest_id,
            "Sam",
            {"drink": "tea"},
        )
        assert remembered["faces"] >= 1 and remembered["appearances"] >= 1
        assert recognize_person(ctx, guest_id) == {
            "id": guest_id,
            "name": "Sam",
            "facts": {"drink": "tea"},
            "matched_by": "face",
        }
        assert recognize_person(ctx, host_id) == {
            "id": host_id,
            "name": None,
            "facts": {},
            "matched_by": None,
        }

        # A fresh array, whose face sample of the guest was not stored. The newest sample
        # is the buffer's own object, so another object there is a newer array's.
        stored = vision.collected(guest_id).faces[-1]
        publish_until(lambda: vision.collected(guest_id).faces[-1] is not stored)
        [found] = find_people(ctx, "sam")
        assert (found["id"], found["name"]) == (guest_id, "Sam")
        assert names(find_people(ctx)) == {guest_id: "Sam", host_id: None}
    finally:
        vision.close()
        publisher_node.destroy_node()
