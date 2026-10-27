"""The heard filter: which incoming lines are noise rather than speech.

Speech recognition turns every sound it records into text, and some of that
text is no speech at all. On silence, a cough or a clap, Whisper writes an
empty line, a non-speech tag such as "[BLANK_AUDIO]" or "(music)", or a
phrase it learned from subtitled videos, such as "Thanks for watching!". The
agent drops such a line before it is queued (WalkieAgent._enqueue), so it
never reaches the LLM. A blank line would do more than waste a turn: the LLM
endpoint rejects a request with a blank message, and the message stays in
the history, so every later task would fail too.

These rules judge a line by its text alone, so they apply to every source,
typed lines included. A line that only carries no answer, such as "Hmm." or
"Okay.", passes: the person is still talking to the robot. So do "Thank you"
and "Bye", which guests say too. No ROS imports.
"""

import re

# Why a line is dropped, as the console line names it.
BLANK = "blank"
NON_SPEECH = "non_speech"
HALLUCINATION = "hallucination"

# A tag that Whisper writes for a sound that is not speech, such as
# "[BLANK_AUDIO]", "(music)" or "*coughs*", or one music note. Each
# alternative ends at its first closing mark, so matching stays linear.
_TAG = re.compile(r"\[[^\[\]]*\]|\([^()]*\)|\*[^*]*\*|[♪♫♬♩]")
_WORD = re.compile(r"[\w']+")

# Phrases that Whisper writes for silence or noise, because they end so many
# of the subtitled videos it was trained on: "you", "Thanks for watching!"
# and the Amara subtitle credit, in its two known wordings ("Subtitles by the
# Amara.org community", "Subtitles made by the community of Amara.org").
# Matched against the line's whole text in lowercase words, so "Thanks for
# watching!" matches and "Thanks for watching the bag" does not, and a line
# with speech before the credit passes. A phrase may repeat, as in "you you".
# Each alternative opens with its own word, so matching stays linear.
_CREDIT = r"subtitles (?:by the amara org community|made by the community of amara org)"
_PHRASE = rf"you|(?:thanks|thank you)(?: so much| very much)? for watching|{_CREDIT}"
_HALLUCINATION = re.compile(rf"(?:{_PHRASE})(?: (?:{_PHRASE}))*")


def text_verdict(text: str) -> str | None:
    """Why `text` is not speech, or None when it may reach the LLM.

    The reason is BLANK for an empty or whitespace-only line, NON_SPEECH
    for a line of non-speech tags with nothing but spaces and punctuation
    between them, and HALLUCINATION for a known Whisper phrase as the whole
    line, after any tags. Case and punctuation do not matter.
    """
    if not text.strip():
        return BLANK
    words = " ".join(_WORD.findall(_TAG.sub(" ", text).lower()))
    if not words and _TAG.search(text):
        return NON_SPEECH
    if _HALLUCINATION.fullmatch(words):
        return HALLUCINATION
    return None
