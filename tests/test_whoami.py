"""Loading RAI whoami output into the agent's system prompt and tools."""

from rai_whoami import EmbodimentInfo

from walkie_agent.whoami import build_docs_tool, build_system_prompt


def test_without_whoami_output_the_base_prompt_is_used(tmp_path):
    assert build_system_prompt("Base prompt.", None) == "Base prompt."
    assert build_system_prompt("Base prompt.", tmp_path) == "Base prompt."
    assert build_docs_tool(tmp_path) is None


def test_whoami_description_is_appended_without_images(tmp_path):
    info = EmbodimentInfo(
        description="An omnidirectional service robot.",
        rules=["Never block a doorway."],
        images=["aGVsbG8="],
    )
    info.to_directory(tmp_path / "generated")

    prompt = build_system_prompt("Base prompt.", tmp_path)

    assert prompt.startswith("Base prompt.\n<embodiment>\n")
    assert "An omnidirectional service robot." in prompt
    assert "Never block a doorway." in prompt
    assert "aGVsbG8=" not in prompt
