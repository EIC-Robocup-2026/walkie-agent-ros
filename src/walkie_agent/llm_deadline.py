"""The bounds on each call of the agent's LLM: a whole-call deadline, a read timeout, no retries.

httpx applies its timeouts to each connect, read and write, not to the
whole request. A server that keeps sending bytes keeps a call alive for as
long as it sends them, and OpenRouter sends an SSE comment, ": OPENROUTER
PROCESSING", while the model has not answered yet. So a read timeout alone
does not bound a slow streamed call. DeadlineTransport adds the whole-call
deadline: it stamps each request's start on the monotonic clock and checks
the deadline on every chunk of the response, comments included.

    timeout_s       the whole call, from the request's start to the reply's
                    last chunk, checked on each chunk
    read_timeout_s  the longest silence: no byte from the server for this
                    long ends the call, before or after the headers
    connect         CONNECT_TIMEOUT_S, fixed
    retries         none: the SDK's max_retries is 0, so a 429 or a 5xx
                    raises at once, whatever its Retry-After asks

A call that keeps sending bytes ends at timeout_s, and a silent one
read_timeout_s after its last byte, so the worst case is about timeout_s
plus read_timeout_s. The check runs in the thread that reads the response:
a watchdog thread that closed the socket from outside would not wake a
recv() blocked in the reading thread on Linux, and the SSL socket is not
safe to tear down from another thread.

The values are config.toml's [walkie.models.agent] (walkie_agent.tuning),
and main.py hands chat_openai_kwargs() to RAI's openai vendor only
(ADR-0042). openai 3 is built on httpx2, so the transport and the timeout
are httpx2's.
"""

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx2
import openai

# How long opening the connection may take, TLS included. The OpenAI SDK's
# own default.
CONNECT_TIMEOUT_S = 5.0
# The message of the ReadTimeout that the deadline raises, so a log tells it
# from a read timeout.
DEADLINE_MESSAGE = "call deadline"


@dataclass(frozen=True)
class AgentModelConfig:
    """The bounds on each call of the agent's LLM. config.toml sets both, in [walkie.models.agent].

    The rest of ADR-0042's role table, the primary model, the fallbacks and
    api_key_env, comes with ADR-0042's own change; until then the model is
    RAI's complex_model and the key OPENAI_API_KEY.
    """

    # The whole-call deadline, ADR-0042's agent timeout. A call that keeps
    # streaming past it raises on its next chunk.
    timeout_s: float = 20.0
    # The longest silence. OpenRouter does not document how often it sends
    # its keep-alive comment. In 8 streamed calls of the HRI task's first
    # turns on 2026-10-27 (deepseek/deepseek-v4.1-flash:nitro, requests of
    # 19 to 21 KB with the 20 tools bound), the longest silence was 1.52 s,
    # before the headers, and the longest between two chunks 0.29 s; no
    # comment was sent. Measure again before lowering it, or after a
    # change of model or provider.
    read_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        for name in ("timeout_s", "read_timeout_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"AgentModelConfig.{name} must be a positive number, not {value}."
                )
        if self.read_timeout_s > self.timeout_s:
            # A server silent from the start would then outlast the deadline.
            raise ValueError(
                f"AgentModelConfig.read_timeout_s {self.read_timeout_s:g} must not be above "
                f"timeout_s {self.timeout_s:g}."
            )


class DeadlineTransport(httpx2.HTTPTransport):
    """An HTTP transport that ends each request `deadline_s` after it starts.

    Each request gets its own deadline, so one client serves every call.
    Once the deadline has passed, the next chunk of the response raises
    httpx2.ReadTimeout("call deadline") instead of arriving. The OpenAI SDK
    raises that as APITimeoutError, like any other timeout. A server that
    sends nothing sends no chunk to check, so the client's read timeout
    ends that call.
    """

    def __init__(self, deadline_s: float, **kwargs: Any) -> None:
        if not (math.isfinite(deadline_s) and deadline_s > 0):
            raise ValueError(f"deadline_s must be a positive number of seconds, not {deadline_s}.")
        super().__init__(**kwargs)
        self.deadline_s = deadline_s

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        deadline = time.monotonic() + self.deadline_s
        response = super().handle_request(request)
        response.stream = _DeadlineStream(response.stream, deadline, request)
        return response


class _DeadlineStream(httpx2.SyncByteStream):
    """A response's byte stream that raises ReadTimeout on the first chunk past `deadline`."""

    def __init__(self, stream: Any, deadline: float, request: httpx2.Request) -> None:
        self._stream = stream
        self._deadline = deadline
        self._request = request

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._stream:
            if time.monotonic() >= self._deadline:
                raise httpx2.ReadTimeout(DEADLINE_MESSAGE, request=self._request)
            yield chunk

    def close(self) -> None:
        # httpcore closes the connection of a reply that was cut short, so
        # the pool never reuses one that is still mid-reply.
        self._stream.close()


def chat_openai_kwargs(config: AgentModelConfig) -> dict[str, Any]:
    """ChatOpenAI's keyword arguments that bound each call by `config`, with no retries.

    ChatOpenAI passes `timeout` and `max_retries` to the OpenAI SDK's
    client, and uses `http_client` for its sync calls, which are the ones
    the agent makes. The client carries its own transport, so
    langchain_openai's TCP keepalive options, which it sets only on the
    clients it builds, do not apply; the read timeout bounds a silent
    connection much sooner. httpx also reads no proxy from the environment
    for a client with its own transport.
    """
    return {
        "timeout": httpx2.Timeout(config.read_timeout_s, connect=CONNECT_TIMEOUT_S),
        "max_retries": 0,
        "http_client": openai.DefaultHttpxClient(transport=DeadlineTransport(config.timeout_s)),
    }
