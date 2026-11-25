"""Builds the chat models, the optional fallback and the tracing callbacks from config.toml.

It reads the same file as RAI but not through get_llm_model, because the provider
options differ: ChatOllama silently ignores `timeout` and `max_retries`, and only
honours `client_kwargs`. Provider imports are lazy, so a vendor's package loads
only when used. API keys stay in the environment.
"""

import os
from typing import Any
from urllib.parse import urlsplit

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

from walkie_agent.config import agent_settings

# The role of the model that describes the camera's pictures of objects (decision
# PK-76), and the role it falls back to in a config.toml that names no such model.
DESCRIBE_ROLE = "describe_model"
DESCRIBE_FALLBACK_ROLE = "simple_model"


def build_model(config: dict[str, Any], role: str = "complex_model") -> BaseChatModel:
    """The model that `[vendor]` names for `role`, such as complex_model."""
    vendor = config["vendor"][role]
    return _chat_model(config, vendor, config[vendor][role])


def build_describe_model(config: dict[str, Any]) -> BaseChatModel:
    """The model that describes pictures: describe_model, or simple_model without it.

    The fallback applies only when `[vendor]` has no describe_model. The chosen model
    must take image parts. A missing key is a KeyError, as in build_model.
    """
    has_role = DESCRIBE_ROLE in config.get("vendor", {})
    return build_model(config, DESCRIBE_ROLE if has_role else DESCRIBE_FALLBACK_ROLE)


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
    if base_url is not None:
        _check_base_url(vendor, base_url)
    if vendor == "openai":
        from langchain_openai import ChatOpenAI
        from openai import OpenAIError

        try:
            return ChatOpenAI(model=name, base_url=base_url, timeout=timeout_s, max_retries=0)
        except OpenAIError as e:
            # Such as a missing OPENAI_API_KEY. main() reports a ValueError in one line.
            raise ValueError(f"The openai model {name!r} cannot be built: {e}") from e
    if vendor == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=name, base_url=base_url, client_kwargs={"timeout": timeout_s})
    raise ValueError(f"Unknown model vendor {vendor!r}; use openai or ollama.")


def _check_base_url(vendor: str, base_url: object) -> None:
    """Refuse a base_url the HTTP client would refuse, as a ValueError that main() reports in one line.

    The client raises its own URL error class from inside the SDK, which is neither a
    ValueError nor the SDK's error, so the check runs here, before the client is built.
    It covers the scheme, the host and the port, not every rule of the client's parser.
    """
    where = f"[{vendor}].base_url"
    if not isinstance(base_url, str):
        raise ValueError(f"{where} must be a string, not {base_url!r}.")
    try:
        # urlsplit itself refuses some hosts, such as an unclosed IPv6 bracket.
        parts = urlsplit(base_url)
    except ValueError as e:
        raise ValueError(f"{where} is not a URL the client can parse, {base_url!r}: {e}.") from e
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{where} must be an http or https URL with a host, not {base_url!r}.")
    try:
        parts.port
    except ValueError as e:
        raise ValueError(f"{where} has a bad port in {base_url!r}: {e}.") from e
