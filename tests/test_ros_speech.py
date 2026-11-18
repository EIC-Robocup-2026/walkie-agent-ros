"""RosSpeech with fake endpoints: subscriber counts, barriers and a publish log. No TTS, no audio."""

import math
import threading

import pytest
from rai_interfaces.msg import HRIMessage
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from ros_fakes import WATCHDOG_S, Background, FakeConnector, bounded
from std_msgs.msg import String

from walkie_agent.capabilities import Ctx
from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.capabilities.guards import CancelToken
from walkie_agent.errors import ErrorCode, StepError
from walkie_agent.ros.speech import RosSpeech


def speech(**kwargs) -> tuple[RosSpeech, FakeConnector]:
    connector = FakeConnector()
    return RosSpeech(connector, **kwargs), connector


def endpoints(connector: FakeConnector):
    return connector.node.publishers[0], connector.node.subscriptions[0]


def test_say_publishes_first_message_and_returns_none():
    wrapper, connector = speech()
    publisher, subscription = endpoints(connector)
    assert (publisher.msg_type, publisher.topic, publisher.qos) == (HRIMessage, "/to_human", 10)
    assert (subscription.msg_type, subscription.topic, subscription.qos) == (
        String,
        "/voice_commands",
        10,
    )

    # The listener shows up only after the first count, and the line still goes out.
    publisher.subscribers = 0

    def listener_arrives():
        publisher.subscribers = 1

    publisher.on_count = listener_arrives
    assert wrapper.say("hello") is None

    assert [message.text for message in publisher.published] == ["hello"]
    message = publisher.published[0]
    assert message.seq_end is True
    assert message.communication_id
    assert message.header.stamp == connector.node.clock.now().to_msg()
    assert len(connector.node.publishers) == 1


def test_say_missing_listener_is_unavailable():
    wrapper, connector = speech(discovery_s=0.05)
    publisher, _ = endpoints(connector)
    publisher.subscribers = 0
    with pytest.raises(StepError) as raised:
        wrapper.say("hello")
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert "/to_human" in raised.value.message
    assert publisher.published == []

    # Discovery yields to a shorter timeout of the call. Its own budget would outlast the watchdog.
    wrapper, connector = speech(discovery_s=10 * WATCHDOG_S)
    endpoints(connector)[0].subscribers = 0
    assert Background(wrapper.say, "hello", timeout_s=0.05).step_error().code is ErrorCode.UNAVAILABLE


@pytest.mark.parametrize("text", ["", "   \n", None, 5, ["hello"]])
def test_say_rejects_blank_and_bad_timeout(text):
    wrapper, connector = speech()
    publisher, _ = endpoints(connector)
    counted = []
    publisher.on_count = lambda: counted.append(1)

    with pytest.raises(StepError) as raised:
        wrapper.say(text)
    assert raised.value.code is ErrorCode.REFUSED
    for timeout_s in (0, -1, math.nan, math.inf, True, "1"):
        with pytest.raises(StepError) as raised:
            wrapper.say("hello", timeout_s=timeout_s)
        assert raised.value.code is ErrorCode.REFUSED

    assert publisher.published == []
    assert counted == []


def test_speech_id_changes_only_on_voice_stop():
    wrapper, connector = speech()
    publisher, subscription = endpoints(connector)

    def id_after(*words: str) -> str:
        for word in words:
            subscription.callback(String(data=word))
        wrapper.say("line")
        return publisher.published[-1].communication_id

    first = id_after()
    assert id_after() == first
    assert id_after("STOP", " stop", "stop ", "stopped", "") == first
    second = id_after("stop")
    assert second != first
    assert id_after() == second


def test_stop_line_can_overlap_a_blocked_say():
    wrapper, connector = speech(discovery_s=WATCHDOG_S)
    publisher, subscription = endpoints(connector)
    blocked = threading.Event()
    release = threading.Event()
    calls = []

    def hold_the_first_caller():
        calls.append(threading.current_thread())
        if len(calls) == 1:
            blocked.set()
            assert release.wait(WATCHDOG_S)

    publisher.on_count = hold_the_first_caller
    ordinary = Background(wrapper.say, "ordinary line")
    assert blocked.wait(WATCHDOG_S)

    # An ordinary say is stuck in discovery. The stop line and a voice stop go through.
    bounded(wrapper.say, "stop line")
    assert [message.text for message in publisher.published] == ["stop line"]
    before = publisher.published[0].communication_id
    bounded(subscription.callback, String(data="stop"))
    assert not ordinary.done()

    release.set()
    assert ordinary.result() is None
    assert [message.text for message in publisher.published] == ["stop line", "ordinary line"]
    assert publisher.published[1].communication_id != before


def test_say_ignores_motion_latch():
    wrapper, connector = speech()
    publisher, _ = endpoints(connector)
    robot = FakeRobot()
    token = CancelToken()
    ctx = Ctx(cancel=token, base=robot, speech=wrapper, vision=robot)
    ctx.cancel.cancel()
    ctx.speech.say("Stopping.")

    # Holding the commit fence does not hold up a line either, since say never takes it.
    token.reset()
    entered = threading.Event()
    release = threading.Event()

    def hold_the_fence():
        with token.unless_cancelled("the test"):
            entered.set()
            assert release.wait(WATCHDOG_S)

    holder = Background(hold_the_fence)
    assert entered.wait(WATCHDOG_S)
    bounded(ctx.speech.say, "Still talking.")
    release.set()
    holder.result()
    assert [message.text for message in publisher.published] == ["Stopping.", "Still talking."]


def test_close_and_publish_failures_are_bounded(caplog):
    # A handle or context that is gone means ROS is unavailable, not a bug.
    for gone in (_rclpy.RCLError("context is invalid"), _rclpy.InvalidHandle("destroyed")):
        wrapper, connector = speech()
        publisher, _ = endpoints(connector)

        def fail(_message, error=gone):
            raise error

        publisher.on_publish = fail
        with pytest.raises(StepError) as raised:
            wrapper.say("hello")
        assert raised.value.code is ErrorCode.UNAVAILABLE

        publisher.on_publish = None

        def fail_count(error=gone):
            raise error

        publisher.on_count = fail_count
        with pytest.raises(StepError) as raised:
            wrapper.say("hello")
        assert raised.value.code is ErrorCode.UNAVAILABLE

    # Any other failure is a bug: logged in full, and the model gets only the type.
    wrapper, connector = speech()
    publisher, _ = endpoints(connector)

    def bug(_message):
        raise ValueError("secret detail")

    publisher.on_publish = bug
    with caplog.at_level("ERROR"), pytest.raises(StepError) as raised:
        wrapper.say("hello")
    assert raised.value.code is ErrorCode.INTERNAL
    assert "secret detail" not in raised.value.message
    assert "secret detail" in caplog.text

    # Close releases a say that is waiting in discovery, long before its deadline.
    wrapper, connector = speech(discovery_s=10 * WATCHDOG_S)
    publisher, subscription = endpoints(connector)
    publisher.subscribers = 0
    waiting = threading.Event()
    publisher.on_count = waiting.set
    say = Background(wrapper.say, "hello", timeout_s=10 * WATCHDOG_S)
    assert waiting.wait(WATCHDOG_S)
    bounded(wrapper.close)
    assert say.step_error().code is ErrorCode.UNAVAILABLE

    # Closing twice destroys once. A later say is refused without touching the publisher.
    bounded(wrapper.close)
    assert connector.node.destroyed == [publisher, subscription]
    publisher.subscribers = 1
    with pytest.raises(StepError) as raised:
        wrapper.say("hello")
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert publisher.published == []

    # A close that loses the race to a say's discovery still stops the publish.
    wrapper, connector = speech()
    publisher, _ = endpoints(connector)
    publisher.on_count = wrapper.close
    assert Background(wrapper.say, "hello").step_error().code is ErrorCode.UNAVAILABLE
    assert publisher.published == []

    # A failed destroy is logged, and does not stop the rest of the close.
    wrapper, connector = speech()
    connector.node.destroy_error = _rclpy.InvalidHandle("destroyed")
    bounded(wrapper.close)
    assert len(connector.node.destroyed) == 2
