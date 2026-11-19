"""The atomic that looks something up on the web (decision PK-6)."""

from typing import Any

from walkie_agent.capabilities import Ctx


def search_web(ctx: Ctx, query: str) -> dict[str, Any]:
    """Search the web for a current fact Walkie cannot know, such as news, the weather or a recipe.

    Returns an object with answer, a short summary of what the web pages say, to
    say in your own words, and sources, the addresses of those pages, which you
    never read aloud.

    Args:
        query: What to look up, as a short search query in plain words, such as
            "Bangkok weather tomorrow". Never put what you saw or heard about the
            people in the room into it.
    """
    found = ctx.web.search(query)
    return {"answer": found.summary, "sources": list(found.sources)}
