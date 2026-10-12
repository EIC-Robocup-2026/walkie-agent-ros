"""Half-duplex voice: speech recognition ignores the microphone while Walkie speaks.

RAI's SpeechRecognitionAgent treats any voice on the microphone as a person
talking: it sends "pause" to TextToSpeechAgent, and once the voice ends it sends
"stop" and transcribes what it heard. When the microphone picks up Walkie's own
speaker, Walkie interrupts itself mid-sentence and the agent receives its own
words as a new request.

So the text-to-speech agent publishes std_msgs/Bool on /walkie/speaking: True,
repeated while audio is playing, and False once when playback finishes, is
paused or is stopped. The speech recognition agent skips voice detection while
True holds and for ECHO_TAIL seconds after False. The cost: a person cannot
interrupt Walkie by voice mid-sentence.

The other direction needs a fix too. RAI's TextToSpeechAgent resumes playback
on every /to_human message, even while speech recognition has paused it for a
person who is talking. Walkie would then talk over the person, and the
microphone, now ignored, would lose their request. So the text-to-speech agent
holds new speech after a "pause" until "play" or "stop" arrives.

Importing this module loads rai_s2s and all its speech backends, which takes
several seconds, so walkie_agent.voice imports it only when an agent is built.
"""

from threading import Event, Thread
from time import monotonic
from typing import Any

import numpy as np
from rai_s2s.asr import SpeechRecognitionAgent
from rai_s2s.tts import TextToSpeechAgent
from std_msgs.msg import Bool

SPEAKING_TOPIC = "/walkie/speaking"
# Seconds between repeated True messages while audio is playing.
SPEAKING_REPEAT = 0.25
# Seconds a True message keeps the microphone ignored. Longer than
# SPEAKING_REPEAT, so a text-to-speech agent that dies mid-sentence leaves
# speech recognition deaf for at most this long.
SPEAKING_TIMEOUT = 1.0
# Seconds the microphone stays ignored after playback stops, for audio still in
# the output buffers and the room's echo.
ECHO_TAIL = 0.5
# Seconds without a speaker callback after which playback counts as stopped.
# The callbacks stop when the audio stream dies, for example when the sound
# device disappears or the callback raises.
SPEAKER_STALL_TIMEOUT = 0.5
# Seconds a "pause" holds new speech. Speech recognition repeats "pause" while
# it hears a voice and sends "stop" or "play" within its silence_grace_period
# after the voice ends, so this only matters if it dies in between. Must be
# longer than silence_grace_period in config.toml.
PAUSE_TIMEOUT = 5.0


class HalfDuplexTextToSpeechAgent(TextToSpeechAgent):
    """TextToSpeechAgent that publishes on /walkie/speaking whether it is playing,
    and that does not start speaking while speech recognition has paused it."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.speaking = False
        self.speaking_changed = Event()
        self.last_callback = 0.0
        self.paused_until = 0.0
        self.speaking_publisher = self.hri_ros2_connector.node.create_publisher(
            Bool, SPEAKING_TOPIC, 10
        )
        self.speaking_thread: Thread | None = None

    def _setup_ros2_connector(self):
        # The base method stores the /voice_commands connector in
        # self.ros2_connector and returns None, which __init__ then assigns to
        # self.ros2_connector. Return it so it can still be shut down.
        super()._setup_ros2_connector()
        return self.ros2_connector

    def run(self) -> None:
        super().run()
        self.speaking_thread = Thread(target=self._publish_speaking, daemon=True)
        self.speaking_thread.start()

    def stop(self) -> None:
        # Silence the speaker first, so the last message on /walkie/speaking is
        # False. The base class never closes the audio stream.
        self.playback_data.playing = False
        super().stop()
        self.speaking_changed.set()
        if self.speaking_thread is not None:
            self.speaking_thread.join()

    def _on_command_message(self, message) -> None:
        command = message.payload.data
        if command == "pause":
            self.paused_until = monotonic() + PAUSE_TIMEOUT
        super()._on_command_message(message)
        if command != "pause":
            # After the base method, so "stop" has dropped the held speech
            # before the speaker callback can play it.
            self.paused_until = 0.0

    def _speaker_callback(self, outdata: np.ndarray, frames: int, time, status_dict) -> None:
        # Runs on the audio thread, so it only records the state; publishing
        # happens on speaking_thread.
        now = monotonic()
        self.last_callback = now
        if now < self.paused_until:
            # Leave the queued audio where it is until "play" or "stop".
            outdata.fill(0)
            speaking = False
        else:
            super()._speaker_callback(outdata, frames, time, status_dict)
            speaking = self.playback_data.playing and self.playback_data.data is not None
        if speaking != self.speaking:
            self.speaking = speaking
            self.speaking_changed.set()

    def _publish_speaking(self) -> None:
        published = False
        while not self.terminate_agent.is_set():
            self.speaking_changed.wait(timeout=SPEAKING_REPEAT)
            self.speaking_changed.clear()
            speaking = (
                self.speaking and monotonic() - self.last_callback < SPEAKER_STALL_TIMEOUT
            )
            if speaking or published:
                self.speaking_publisher.publish(Bool(data=speaking))
                published = speaking
        if published:
            self.speaking_publisher.publish(Bool(data=False))


class HalfDuplexSpeechRecognitionAgent(SpeechRecognitionAgent):
    """SpeechRecognitionAgent that ignores the microphone while Walkie speaks.

    Ignored samples skip voice detection, so they neither start a recording nor
    send "pause" or "stop" to the text-to-speech agent.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.ignore_until = 0.0
        self.ignoring = False
        self.ros2_connector.node.create_subscription(
            Bool, SPEAKING_TOPIC, self._on_speaking, 10
        )

    def _on_speaking(self, msg: Bool) -> None:
        self.ignore_until = monotonic() + (SPEAKING_TIMEOUT if msg.data else ECHO_TAIL)

    def _on_new_sample(self, indata: np.ndarray, status_flags: dict[str, Any]) -> None:
        if monotonic() >= self.ignore_until:
            if self.ignoring:
                self.ignoring = False
                self.logger.info("Walkie stopped speaking, listening again")
            super()._on_new_sample(indata, status_flags)
            return

        if not self.ignoring:
            self.ignoring = True
            self.logger.info("Walkie is speaking, ignoring the microphone")
        if self.recording_started:
            self._discard_recording()
        with self.sample_buffer_lock:
            self.sample_buffer = []

    def _discard_recording(self) -> None:
        """Drop the recording in progress without transcribing it.

        A recording sends "pause", which holds the text-to-speech agent, so
        Walkie starts speaking during one only when both start at almost the
        same moment. The recording may then be the start of Walkie's own voice,
        heard before the /walkie/speaking message arrived, and transcribing it
        would bring back the problem this module solves. A person who started
        at that moment has to repeat themselves.
        """
        self.logger.info("Discarding the recording in progress")
        self.recording_started = False
        self.grace_period_start = 0
        # The transcription thread for this recording was created but never
        # started; stop() waits for every thread in this dict to finish.
        self.transcription_threads.pop(self.active_thread, None)
        self.active_thread = ""
