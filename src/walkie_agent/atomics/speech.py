"""The atomic that speaks."""

from walkie_agent.capabilities import Ctx


def say(ctx: Ctx, text: str) -> str:
    """Say one line out loud to the people around Walkie.

    Returns a sentence that repeats what was said.

    Args:
        text: The line to say, in plain spoken words.
    """
    ctx.speech.say(text)
    return f"Said: {text}"
