"""Walkie's identity from RAI whoami: a robot description and a documentation search.

`pixi run whoami-build` reads the source files in whoami/ (documentation/ and
images/) and writes whoami/generated/:
  - info.json: description, rules, capabilities and behaviors that an LLM
    derived from the sources. It is added to the agent's system prompt.
  - a FAISS index of the documentation, searched by the search_robot_docs tool.

Both are optional: without whoami/generated the agent runs on SYSTEM_PROMPT
alone and has no documentation tool.
"""

from pathlib import Path

from langchain_core.tools import BaseTool

DOCS_TOOL_DESCRIPTION = (
    "Search Walkie's own documentation (hardware, specifications, manuals) with a "
    "natural language query. Use it for questions about the robot itself, such as "
    "its size, payload or sensors. Returns the most relevant passages."
)


def build_system_prompt(base_prompt: str, whoami_dir: Path | None) -> str:
    """Append the whoami description to `base_prompt`, if one has been built.

    The photos stored with the description are not attached: the build already
    turned them into the text description, and attached images would be sent
    (and billed) with every LLM call.
    """
    if whoami_dir is None or not (whoami_dir / "generated" / "info.json").exists():
        return base_prompt

    from rai_whoami import EmbodimentInfo

    info = EmbodimentInfo.from_directory(whoami_dir)
    # The message content is a list of text and image blocks; keep the text.
    content = info.to_langchain(style="xml_tags").content
    if isinstance(content, str):
        embodiment = content
    else:
        embodiment = "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return f"{base_prompt}\n<embodiment>\n{embodiment}</embodiment>\n"


def build_docs_tool(whoami_dir: Path) -> BaseTool | None:
    """A search tool over the documentation index, if one has been built."""
    if not (whoami_dir / "generated" / "index.faiss").exists():
        return None

    from rai_whoami.tools import QueryDatabaseTool

    # QueryDatabaseTool appends "generated" to root_dir itself.
    return QueryDatabaseTool(
        root_dir=str(whoami_dir),
        name="search_robot_docs",
        description=DOCS_TOOL_DESCRIPTION,
    )
