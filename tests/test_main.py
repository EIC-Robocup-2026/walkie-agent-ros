"""The agent's command line and LLM (main.py), checked without starting ROS."""

import sys
from datetime import datetime
from pathlib import Path

import httpx2
import pytest
from langchain_openai import ChatOpenAI

import walkie_agent.main as agent_main
from walkie_agent.llm_deadline import AgentModelConfig, DeadlineTransport
from walkie_agent.main import build_llm, build_parser, parse_args, tool_log_path
from walkie_agent.tuning import Tuning

NOW = datetime(2026, 10, 24, 9, 5, 7)


@pytest.fixture
def rai_config(tmp_path, monkeypatch):
    """Write RAI's config.toml, with `vendor` for every model, in a new working directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def write(vendor: str) -> None:
        (tmp_path / "config.toml").write_text(
            "[vendor]\n"
            f'simple_model = "{vendor}"\n'
            f'complex_model = "{vendor}"\n'
            f'embeddings_model = "{vendor}"\n'
            "\n"
            "[openai]\n"
            'simple_model = "test-simple"\n'
            'complex_model = "test-complex"\n'
            'embeddings_model = "test-embeddings"\n'
            'base_url = "http://127.0.0.1:9/v1"\n'
        )

    return write


def test_no_tool_log_without_the_flag():
    args = build_parser().parse_args([])

    assert tool_log_path(args.tool_log, NOW) is None


def test_the_bare_tool_log_flag_names_the_log_by_time():
    args = build_parser().parse_args(["--tool-log", "--arms"])

    assert args.arms
    assert tool_log_path(args.tool_log, NOW) == Path("logs/agent_tools_20261024-090507.jsonl")


def test_the_tool_log_flag_takes_a_path():
    args = build_parser().parse_args(["--tool-log", "/tmp/run1/tools.jsonl"])

    assert tool_log_path(args.tool_log, NOW) == Path("/tmp/run1/tools.jsonl")


def test_the_people_session_is_forgotten_unless_resumed():
    assert not parse_args([]).resume_people_session
    assert parse_args(["--resume-people-session"]).resume_people_session


def test_resuming_the_people_session_needs_the_database(capsys):
    with pytest.raises(SystemExit) as exit_:
        parse_args(["--resume-people-session", "--mock-people-memory"])

    assert exit_.value.code == 2
    assert "--resume-people-session needs walkie-database" in capsys.readouterr().err


def test_the_openai_vendor_gets_the_bounds_on_each_call(rai_config):
    rai_config("openai")

    llm = build_llm(AgentModelConfig(timeout_s=20.0, read_timeout_s=10.0))

    try:
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "test-complex"
        assert llm.streaming
        # No retries, in ChatOpenAI and in the OpenAI SDK client it builds.
        assert llm.max_retries == 0
        assert llm.root_client.max_retries == 0
        assert llm.root_client.timeout == httpx2.Timeout(10.0, connect=5.0)
        # The SDK client sends through the client with the deadline.
        assert llm.root_client._client is llm.http_client
        transport = llm.http_client._transport
        assert isinstance(transport, DeadlineTransport)
        assert transport.deadline_s == 20.0
    finally:
        llm.http_client.close()


@pytest.mark.parametrize("vendor", ["ollama", "aws", "google"])
def test_no_other_vendor_gets_the_bounds(rai_config, monkeypatch, vendor):
    rai_config(vendor)
    calls = []
    model = object()

    def get_llm_model(model_type, **kwargs):
        calls.append((model_type, kwargs))
        return model

    monkeypatch.setattr(agent_main, "get_llm_model", get_llm_model)

    assert build_llm(AgentModelConfig()) is model
    assert calls == [("complex_model", {"vendor": vendor, "streaming": True})]


def test_the_tuning_is_loaded_before_the_llm_is_built(monkeypatch):
    class Stop(Exception):
        pass

    order = []
    built = {}

    def load_tuning():
        order.append("tuning")
        return Tuning(agent_model=AgentModelConfig(timeout_s=30.0, read_timeout_s=7.0))

    def get_llm_model_config_and_vendor(model_type):
        order.append("vendor")
        return None, "openai"

    def get_llm_model(model_type, **kwargs):
        order.append("llm")
        built.update(kwargs)
        return object()

    def load_task_templates(path):
        # Past the LLM, so main() stops before ROS starts.
        raise Stop

    monkeypatch.setattr(sys, "argv", ["walkie-agent"])
    monkeypatch.setattr(agent_main, "load_dotenv", lambda: None)
    monkeypatch.setattr(agent_main, "load_tuning", load_tuning)
    monkeypatch.setattr(
        agent_main, "get_llm_model_config_and_vendor", get_llm_model_config_and_vendor
    )
    monkeypatch.setattr(agent_main, "get_llm_model", get_llm_model)
    monkeypatch.setattr(agent_main, "load_task_templates", load_task_templates)

    with pytest.raises(Stop):
        agent_main.main()

    try:
        assert order == ["tuning", "vendor", "llm"]
        # The LLM is built with the tuning's bounds, not the defaults.
        assert built["max_retries"] == 0
        assert built["timeout"] == httpx2.Timeout(7.0, connect=5.0)
        assert built["http_client"]._transport.deadline_s == 30.0
    finally:
        built["http_client"].close()
