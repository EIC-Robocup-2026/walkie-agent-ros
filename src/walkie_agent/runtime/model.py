"""Builds the chat model, the optional fallback and the tracing callbacks from config.toml.

It reads the same file as RAI but not through get_llm_model, because the provider
options differ: ChatOllama silently ignores `timeout` and `max_retries`, and only
honours `client_kwargs`. Provider imports are lazy, so a vendor's package loads
only when used. API keys stay in the environment.
"""

import os
import tomllib
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

# Provisional until real runs measure them; `[agent]` in config.toml overrides them.
DEFAULT_RUN_LIMIT = 25
DEFAULT_DEADLINE_S = 60.0


def load_config(path: str | Path = "config.toml") -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def agent_settings(config: dict[str, Any]) -> dict[str, Any]:
    """The `run_limit` and `deadline_s` keyword arguments of build_agent."""
    agent = config.get("agent", {})
    return {
        "run_limit": agent.get("run_limit", DEFAULT_RUN_LIMIT),
        "deadline_s": agent.get("deadline_s", DEFAULT_DEADLINE_S),
    }


def build_model(config: dict[str, Any], role: str = "complex_model") -> BaseChatModel:
    """The model that `[vendor]` names for `role`, such as complex_model."""
    vendor = config["vendor"][role]
    return _chat_model(config, vendor, config[vendor][role])


def build_fallback(config: dict[str, Any]) -> BaseChatModel | None:
    """The fallback named by `fallback_vendor` and `fallback_model` in `[agent]`, or None.

    It must name both. Choosing the fallback is the team's call (decision 19), so
    a vendor alone must not silently enable that vendor's complex_model.
    """
    agent = config.get("agent", {})
    vendor, name = agent.get("fallback_vendor"), agent.get("fallback_model")
    if not vendor and not name:
        return None
    if not (vendor and name):
        raise ValueError("A fallback needs both fallback_vendor and fallback_model in [agent].")
    return _chat_model(config, vendor, name)


def build_callbacks(config: dict[str, Any]) -> list[BaseCallbackHandler]:
    """A Langfuse CallbackHandler when `[tracing.langfuse]` enables it, otherwise nothing.

    The handler groups a session's runs by the `langfuse_session_id` metadata that
    Session.config sends. It uses the client that Langfuse() configures here.
    """
    settings = config.get("tracing", {}).get("langfuse", {})
    if not settings.get("use_langfuse"):
        return []
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        raise ValueError("Tracing needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in the env.")
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(public_key=public_key, secret_key=secret_key, host=settings.get("host"))
    return [CallbackHandler()]


def _chat_model(config: dict[str, Any], vendor: str, name: str) -> BaseChatModel:
    # The SDK's timeout backs up Deadline, and its retries would only spend Deadline's
    # budget, so they are off where the provider has any.
    timeout_s = agent_settings(config)["deadline_s"]
    base_url = config.get(vendor, {}).get("base_url")
    if vendor == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, base_url=base_url, timeout=timeout_s, max_retries=0)
    if vendor == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=name, base_url=base_url, client_kwargs={"timeout": timeout_s})
    raise ValueError(f"Unknown model vendor {vendor!r}; use openai or ollama.")
