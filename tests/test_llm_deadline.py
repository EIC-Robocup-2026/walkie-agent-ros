"""The bounds on each LLM call (walkie_agent.llm_deadline), against a fake server.

The server runs on 127.0.0.1, in a thread, and answers each chat completion
request with a scripted reply. The client is the one main.py builds for
RAI's openai vendor: a streaming ChatOpenAI with chat_openai_kwargs(). The
bounds a test reaches are 1.5 s or less, so the tests run fast; the upper
limits allow for a loaded test machine.
"""

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx2
import openai
import pytest
from langchain_openai import ChatOpenAI

from walkie_agent.llm_deadline import (
    DEADLINE_MESSAGE,
    AgentModelConfig,
    DeadlineTransport,
    chat_openai_kwargs,
)

# How much later than its bound a call may end on a loaded test machine.
SLACK_S = 1.0
# OpenRouter's keep-alive comment, sent while the model has not answered.
COMMENT = b": OPENROUTER PROCESSING\n\n"
# The longest a scripted reply lasts, so a call that is never cut still ends.
REPLY_CAP_S = 10.0

_Reply = Callable[[BaseHTTPRequestHandler], None]


class FakeServer:
    """A chat completions endpoint on 127.0.0.1 that answers each request with `reply`."""

    def __init__(self) -> None:
        self.reply: _Reply | None = None
        self.requests = 0
        # Set at teardown, so a reply still waiting or streaming ends.
        self.done = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            # HTTP/1.1, for chunked replies, as OpenRouter streams them.
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                server.requests += 1
                assert server.reply is not None
                try:
                    server.reply(self)
                except OSError:
                    # The client cut the call and closed the connection.
                    self.close_connection = True

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_port}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.done.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)


@pytest.fixture
def server():
    fake = FakeServer()
    yield fake
    fake.close()


@pytest.fixture
def chat_model(server):
    """Build the streaming ChatOpenAI that main.py builds, on the fake server."""
    models: list[ChatOpenAI] = []

    def build(config: AgentModelConfig) -> ChatOpenAI:
        llm = ChatOpenAI(
            model="test-model",
            base_url=server.base_url,
            api_key="test-key",
            streaming=True,
            **chat_openai_kwargs(config),
        )
        models.append(llm)
        return llm

    yield build
    for llm in models:
        llm.http_client.close()


def start_events(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()


def send_chunk(handler: BaseHTTPRequestHandler, data: bytes) -> None:
    handler.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
    handler.wfile.flush()


def end_chunks(handler: BaseHTTPRequestHandler) -> None:
    handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()


def completion_event(content: str, finish_reason: str | None = None) -> bytes:
    """One SSE event of a streamed chat completion."""
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}
        ],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def reply_parts(parts: list[str]) -> list[bytes]:
    """The SSE events of a reply that streams `parts`, then ends."""
    return [
        *(completion_event(part) for part in parts),
        completion_event("", finish_reason="stop"),
        b"data: [DONE]\n\n",
    ]


def streams(events: list[bytes], gap_s: float) -> _Reply:
    """A reply that sends each of `events` as a chunk, `gap_s` apart."""

    def reply(handler: BaseHTTPRequestHandler) -> None:
        start_events(handler)
        for event in events:
            send_chunk(handler, event)
            time.sleep(gap_s)
        end_chunks(handler)

    return reply


def timed_call(llm: ChatOpenAI) -> tuple[BaseException, float]:
    """The error a call raises, and how long it took."""
    start = time.monotonic()
    with pytest.raises(Exception) as caught:
        llm.invoke("Who is at the door?")
    return caught.value, time.monotonic() - start


def read_timeout_in(error: BaseException) -> httpx2.ReadTimeout | None:
    """The httpx2.ReadTimeout behind `error`, through the SDK's and LangChain's wrappers."""
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, httpx2.ReadTimeout):
            return cause
        cause = cause.__cause__ or cause.__context__
    return None


def test_a_stream_of_comments_only_ends_by_the_deadline_and_one_comment_interval(
    server, chat_model
):
    # The deadline falls halfway between two comments, so the call ends on
    # the comment at 2 s. The tolerance is under half an interval, so a call
    # that ends one comment later, at 3 s, fails the bound.
    interval_s = 1.0
    deadline_s = 1.5
    tolerance_s = 0.4

    def comments(handler: BaseHTTPRequestHandler) -> None:
        # Each comment is inside the read timeout, so only the deadline can
        # end the call.
        start_events(handler)
        start = time.monotonic()
        while not server.done.is_set() and time.monotonic() - start < REPLY_CAP_S:
            send_chunk(handler, COMMENT)
            server.done.wait(interval_s)

    server.reply = comments
    llm = chat_model(AgentModelConfig(timeout_s=deadline_s, read_timeout_s=deadline_s))

    error, elapsed = timed_call(llm)

    assert isinstance(error, openai.APITimeoutError)
    timeout = read_timeout_in(error)
    assert timeout is not None and str(timeout) == DEADLINE_MESSAGE
    assert deadline_s <= elapsed < deadline_s + interval_s + tolerance_s


def test_headers_then_silence_end_after_the_read_timeout(server, chat_model):
    def silence(handler: BaseHTTPRequestHandler) -> None:
        start_events(handler)
        server.done.wait(REPLY_CAP_S)

    server.reply = silence
    llm = chat_model(AgentModelConfig(timeout_s=5.0, read_timeout_s=0.5))

    error, elapsed = timed_call(llm)

    assert isinstance(error, openai.APITimeoutError)
    # The client's read timeout, not the deadline: no chunk came to check it on.
    timeout = read_timeout_in(error)
    assert timeout is not None and str(timeout) != DEADLINE_MESSAGE
    assert 0.5 <= elapsed < 0.5 + SLACK_S


def test_a_normal_stream_passes_through_untouched(server):
    events = reply_parts(["Hello", " there", "."])
    server.reply = streams(events, gap_s=0.05)

    with httpx2.Client(transport=DeadlineTransport(5.0)) as client:
        with client.stream("POST", f"{server.base_url}/chat/completions", json={}) as response:
            body = b"".join(response.iter_raw())

    assert response.status_code == 200
    assert body == b"".join(events)


def test_a_normal_reply_reaches_the_chat_model(server, chat_model):
    server.reply = streams(reply_parts(["Hello", " there", "."]), gap_s=0.05)
    llm = chat_model(AgentModelConfig(timeout_s=5.0, read_timeout_s=1.0))

    assert llm.invoke("Who is at the door?").content == "Hello there."
    assert server.requests == 1


def test_each_call_gets_its_own_deadline(server, chat_model):
    # Two calls of about 0.6 s each, on one client: together they run past
    # the 1 s deadline, and neither alone does.
    server.reply = streams(reply_parts(["Hello", " there", "."]), gap_s=0.12)
    llm = chat_model(AgentModelConfig(timeout_s=1.0, read_timeout_s=1.0))

    replies = [llm.invoke("Who is at the door?").content for _ in range(2)]

    assert replies == ["Hello there.", "Hello there."]
    assert server.requests == 2


def test_a_rate_limit_with_retry_after_raises_at_once(server, chat_model):
    def rate_limited(handler: BaseHTTPRequestHandler) -> None:
        body = json.dumps({"error": {"message": "Rate limited.", "code": 429}}).encode()
        handler.send_response(429)
        # The SDK would wait out a Retry-After of up to 120 s before a retry.
        handler.send_header("Retry-After", "60")
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    server.reply = rate_limited
    llm = chat_model(AgentModelConfig())

    error, elapsed = timed_call(llm)

    assert isinstance(error, openai.RateLimitError)
    assert elapsed < SLACK_S
    assert server.requests == 1


def test_the_deadline_must_be_a_positive_number():
    with pytest.raises(ValueError, match="deadline_s must be a positive number of seconds"):
        DeadlineTransport(0.0)
