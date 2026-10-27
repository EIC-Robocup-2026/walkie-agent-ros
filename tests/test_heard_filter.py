"""The heard filter's text rules (walkie_agent.heard_filter). No ROS."""

import pytest

from walkie_agent.heard_filter import BLANK, HALLUCINATION, NON_SPEECH, text_verdict


@pytest.mark.parametrize(
    "text, reason",
    [
        ("", BLANK),
        (" ", BLANK),
        ("\n", BLANK),
        (" \t\r\n ", BLANK),
        ("[BLANK_AUDIO]", NON_SPEECH),
        ("  [BLANK_AUDIO]  ", NON_SPEECH),
        ("[BLANK_AUDIO] [BLANK_AUDIO]", NON_SPEECH),
        ("[Music]", NON_SPEECH),
        ("[ Silence ]", NON_SPEECH),
        ("(music)", NON_SPEECH),
        ("(upbeat music)", NON_SPEECH),
        ("*coughs*", NON_SPEECH),
        ("♪", NON_SPEECH),
        ("♪♪♪", NON_SPEECH),
        ("[Music] (applause)", NON_SPEECH),
        ("(laughs).", NON_SPEECH),
        ("[]", NON_SPEECH),
        ("Thanks for watching!", HALLUCINATION),
        ("thanks for watching", HALLUCINATION),
        ("Thank you for watching.", HALLUCINATION),
        ("THANK YOU FOR WATCHING", HALLUCINATION),
        ("Thank you so much for watching!", HALLUCINATION),
        ("Thanks very much for watching.", HALLUCINATION),
        ("  Thanks   for watching!  ", HALLUCINATION),
        ("[Music] Thanks for watching!", HALLUCINATION),
        ("you", HALLUCINATION),
        ("You.", HALLUCINATION),
        ("you you", HALLUCINATION),
        ("Subtitles by the Amara.org community", HALLUCINATION),
        ("Subtitles made by the community of Amara.org", HALLUCINATION),
        ("Thanks for watching! Subtitles by the Amara.org community.", HALLUCINATION),
    ],
)
def test_noise_is_dropped_with_its_reason(text, reason):
    assert text_verdict(text) == reason


@pytest.mark.parametrize(
    "text",
    [
        # No answer in them, but the person is still talking to the robot.
        "Hmm.",
        "Okay.",
        "okay",
        # Guests say these, so they are not dropped for their text alone.
        "Thank you.",
        "Thank you",
        "Bye.",
        "Yes.",
        "No.",
        "My name is Sam.",
        "My name is Amara.",
        "I would like milk.",
        "stop",
        "Never mind!",
        # The words around a phrase or a tag make the line speech.
        "Thanks for watching the bag.",
        "Thank you, you are very kind.",
        "Are you Walkie?",
        "[Music] Hello there.",
        "Hello (quietly).",
        "♪ Happy birthday ♪",
        "2 * 3 * 4",
        "My name is Sam. Thanks for watching!",
        "My name is Sam. Subtitles by the Amara.org community",
        "I would like milk. Amara.org",
        "My name is Amara. Org",
        # Punctuation alone is not a tag.
        "...",
        "?",
    ],
)
def test_speech_passes(text):
    assert text_verdict(text) is None
