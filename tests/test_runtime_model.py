"""The model builder and the deadline, with a local fake provider and no real LLM.

D1 to D4 are spike_04's claims, here against the real client code on 127.0.0.1.
"""

import asyncio
import json
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from runtime_harness import ScriptedModel

from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.runtime import (
    agent_settings,
    build_agent,
    build_callbacks,
    build_fallback,
    build_model,
    load_config,
)
from walkie_agent.runtime.model import DEFAULT_DEADLINE_S, DEFAULT_RUN_LIMIT

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"


def config_for(vendor: str = "openai", **agent) -> dict:
    return {
        "vendor": {"complex_model": vendor, "simple_model": vendor},
        "openai": {
            "complex_model": "big-openai",
            "simple_model": "small-openai",
            "base_url": "http://127.0.0.1:1/v1",
        },
        "ollama": {
            "complex_model": "big-ollama",
            "simple_model": "small-ollama",
            "base_url": "http://127.0.0.1:2",
        },
        "agent": agent,
    }


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")


# -- Settings ---------------------------------------------------------------------


def test_the_repo_config_builds_its_model_with_no_agent_section(api_key):
    config = load_config(REPO_CONFIG)
    assert "agent" not in config
    assert agent_settings(config) == {
        "run_limit": DEFAULT_RUN_LIMIT,
        "deadline_s": DEFAULT_DEADLINE_S,
    }
    assert build_model(config).model_name == config["openai"]["complex_model"]
    assert build_fallback(config) is None
    assert build_callbacks(config) == []


def test_load_config_reads_the_named_file(tmp_path):
    path = tmp_path / "other.toml"
    path.write_text('[agent]\nrun_limit = 7\ndeadline_s = 1.5\n')
    assert agent_settings(load_config(path)) == {"run_limit": 7, "deadline_s": 1.5}


def test_a_config_that_is_not_toml_is_refused_with_its_path(tmp_path):
    # main() prints this one line alone, so the line has to say which file is broken.
    path = tmp_path / "config.toml"
    path.write_text("not toml = [")
    with pytest.raises(ValueError, match=r"line 1, column 5") as caught:
        load_config(path)
    assert str(caught.value).startswith(f"{path}: ")


# -- Provider options -------------------------------------------------------------


def test_openai_gets_the_deadline_as_timeout_and_no_retries(api_key):
    model = build_model(config_for(deadline_s=12.0))
    assert model.model_name == "big-openai"
    assert model.openai_api_base == "http://127.0.0.1:1/v1"
    assert model.request_timeout == 12.0
    assert model.max_retries == 0


def test_ollama_gets_the_timeout_through_client_kwargs():
    # ChatOllama ignores timeout= and max_retries= silently, so read what reached the client.
    model = build_model(config_for("ollama", deadline_s=12.0))
    assert model.model == "big-ollama"
    assert model.base_url == "http://127.0.0.1:2"
    assert model._async_client._client.timeout == httpx.Timeout(12.0)
    assert model._client._client.timeout == httpx.Timeout(12.0)


def test_build_model_picks_the_role(api_key):
    assert build_model(config_for(), "simple_model").model_name == "small-openai"


def test_an_unknown_vendor_is_refused():
    config = config_for("mistery")
    config["mistery"] = {"complex_model": "x"}
    with pytest.raises(ValueError, match="Unknown model vendor 'mistery'"):
        build_model(config)


def test_a_missing_openai_key_is_a_value_error_that_names_it(monkeypatch):
    # main() reports a ValueError in one line, and the client's own OpenAIError is not one.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY") as caught:
        build_model(config_for())
    assert "'big-openai'" in str(caught.value)
    assert isinstance(caught.value.__cause__, openai.OpenAIError)


# -- The fallback -----------------------------------------------------------------


@pytest.mark.parametrize(
    "agent", [{"fallback_vendor": "ollama"}, {"fallback_model": "small-ollama"}]
)
def test_a_fallback_must_name_both_vendor_and_model(agent):
    with pytest.raises(ValueError, match="needs both"):
        build_fallback(config_for(**agent))


def test_a_fallback_uses_the_named_model_not_the_vendors_default():
    fallback = build_fallback(config_for(fallback_vendor="ollama", fallback_model="tiny"))
    assert fallback.model == "tiny"
    assert fallback.base_url == "http://127.0.0.1:2"


def test_no_fallback_by_default():
    assert build_fallback(config_for()) is None


# -- Tracing ----------------------------------------------------------------------


def tracing_config(enabled: bool = True) -> dict:
    settings = {"use_langfuse": enabled, "host": "http://127.0.0.1:3"}
    return {"tracing": {"langfuse": settings}}


@pytest.fixture
def fake_langfuse(monkeypatch):
    """Replaces the langfuse package, so nothing starts a client or touches a network."""
    clients: list[dict] = []
    package = types.ModuleType("langfuse")
    package.Langfuse = lambda **kwargs: clients.append(kwargs)
    integration = types.ModuleType("langfuse.langchain")
    integration.CallbackHandler = type("CallbackHandler", (), {})
    monkeypatch.setitem(sys.modules, "langfuse", package)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", integration)
    return types.SimpleNamespace(clients=clients, handler=integration.CallbackHandler)


def test_tracing_builds_a_callback_handler_from_the_environment(monkeypatch, fake_langfuse):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    handlers = build_callbacks(tracing_config())
    assert [type(h) for h in handlers] == [fake_langfuse.handler]
    assert fake_langfuse.clients == [
        {"public_key": "pk-test", "secret_key": "sk-test", "host": "http://127.0.0.1:3"}
    ]


def test_tracing_needs_its_keys(monkeypatch, fake_langfuse):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        build_callbacks(tracing_config())
    assert fake_langfuse.clients == []


def test_tracing_off_builds_nothing(fake_langfuse):
    assert build_callbacks(tracing_config(enabled=False)) == []
    assert build_callbacks({}) == []
    assert fake_langfuse.clients == []


def test_the_installed_langfuse_has_the_handler_the_builder_imports():
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    assert Langfuse and CallbackHandler


# -- A local fake provider --------------------------------------------------------


class _Provider:
    """A fake OpenAI endpoint with one behaviour per base URL: silent, drip or ok."""

    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []
        self.closing = threading.Event()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                kind = self.path.split("/")[1]
                provider.events.append((f"{kind} request", time.monotonic()))
                if kind == "silent":
                    provider.closing.wait(30)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    if kind == "drip":
                        # What OpenRouter sends while it waits for the model, which
                        # keeps a read timeout from ever firing.
                        while not provider.closing.is_set():
                            self.wfile.write(b": OPENROUTER PROCESSING\n\n")
                            self.wfile.flush()
                            time.sleep(0.1)
                    else:
                        self.wfile.write(json.dumps(_completion("Local answer.")).encode())
                except (BrokenPipeError, ConnectionResetError):
                    provider.events.append((f"{kind} client gone", time.monotonic()))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def url(self, kind: str) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/{kind}"

    def times_of(self, event: str) -> list[float]:
        return [at for name, at in self.events if name == event]

    def close(self) -> None:
        self.closing.set()
        self.server.shutdown()
        self.server.server_close()


def _completion(text: str) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
    }


@pytest.fixture
def provider(monkeypatch, api_key):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    fake = _Provider()
    yield fake
    fake.close()


def provider_model(provider: _Provider, kind: str, deadline_s: float) -> BaseChatModel:
    config = config_for(deadline_s=deadline_s)
    config["openai"]["base_url"] = provider.url(kind)
    return build_model(config)


def test_the_built_model_answers_through_the_real_stack(provider):
    robot = FakeRobot()
    agent = build_agent(provider_model(provider, "ok", 5.0), robot.ctx, deadline_s=5.0)
    config = {"configurable": {"thread_id": "t"}, "max_concurrency": 1}
    payload = {"messages": [HumanMessage("Hi.")], "todo_seed": None}
    result = asyncio.run(agent.ainvoke(payload, config))
    assert result["messages"][-1].content == "Local answer."


def test_d1_the_sdk_timeout_ends_a_silent_call(provider):
    model = provider_model(provider, "silent", 0.5)
    started = time.monotonic()
    with pytest.raises(openai.APITimeoutError):
        asyncio.run(model.ainvoke("Hi."))
    assert 0.3 < time.monotonic() - started < 2


def test_d2_the_sdk_timeout_does_not_end_a_call_that_gets_sse_comments(provider):
    # The reason Deadline exists. The read timeout would fire at 0.5 s, and it does not.
    model = provider_model(provider, "drip", 0.5)

    async def bounded():
        async with asyncio.timeout(1.2):
            await model.ainvoke("Hi.")

    with pytest.raises(TimeoutError):
        asyncio.run(bounded())


def test_d3_d4_deadline_ends_a_dripping_call_and_the_fallback_answers(provider):
    robot = FakeRobot()
    agent = build_agent(
        provider_model(provider, "drip", 30.0),
        robot.ctx,
        deadline_s=0.5,
        fallback=ScriptedModel(then=AIMessage("From the fallback.")),
    )
    config = {"configurable": {"thread_id": "t"}, "max_concurrency": 1}
    payload = {"messages": [HumanMessage("Hi.")], "todo_seed": None}

    started = time.monotonic()
    result = asyncio.run(agent.ainvoke(payload, config))
    elapsed = time.monotonic() - started

    assert result["messages"][-1].content == "From the fallback."
    assert 0.4 < elapsed < 1.5  # about the deadline, which is not a hard wall-clock bound

    # The client let go. This says nothing about whether the provider stopped computing.
    deadline = time.monotonic() + 3
    while not provider.times_of("drip client gone") and time.monotonic() < deadline:
        time.sleep(0.02)
    assert provider.times_of("drip client gone")
    assert provider.times_of("drip client gone")[0] - provider.times_of("drip request")[0] < 2.5


def test_a_timeout_with_no_fallback_reaches_the_caller(provider):
    robot = FakeRobot()
    agent = build_agent(provider_model(provider, "drip", 30.0), robot.ctx, deadline_s=0.3)
    config = {"configurable": {"thread_id": "t"}, "max_concurrency": 1}
    payload = {"messages": [HumanMessage("Hi.")], "todo_seed": None}
    with pytest.raises(TimeoutError):
        asyncio.run(agent.ainvoke(payload, config))
