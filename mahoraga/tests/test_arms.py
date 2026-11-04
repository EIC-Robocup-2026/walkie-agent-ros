"""Each arm runs once on Inspect's mock model, through the same gate, driver and scorer."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest
from inspect_ai.log import resolve_sample_attachments
from inspect_ai.model import ModelOutput, get_model

from mahoraga.arms import ARMS, INDUCT, PROMPT, ArmSetup
from mahoraga.commands import commands
from mahoraga.gym import run_episodes
from mahoraga.oracle import oracle_script
from mahoraga.shelf import Shelf
from walkie_agent.capabilities.fake import FakeRobot

MOCK = "mockllm/model"
SPEAK = 'def speak_up():\n    """Say hello."""\n    say("Hello.")\n'


def _first(kind, **fields):
    for record in commands("trial", 200):
        spec = record["spec"]
        if spec["kind"] == kind and all(spec[key] == value for key, value in fields.items()):
            return record
    raise AssertionError(f"No {kind} in the first 200 commands.")


def _index(record):
    """The n of commands("trial", n) that ends with `record`."""
    return [r["seed"] for r in commands("trial", 200)].index(record["seed"]) + 1


def _done():
    return ModelOutput.from_content(MOCK, "Done.")


def _mock(record, *calls):
    """A model that makes these tool calls in order for `record`, and replies with text otherwise."""
    outputs = iter([ModelOutput.for_tool_call(MOCK, name, args) for name, args in calls])

    def reply(messages, tools, tool_choice, config):
        asked = next(message.text for message in messages if message.role == "user")
        return next(outputs, _done()) if asked == record["text"] else _done()

    return get_model(MOCK, custom_outputs=reply, memoize=False)


def _run(gated, arm, record, *calls, **kwargs):
    n = _index(record)
    gated(commands("trial", n))
    log = run_episodes([arm], n, model=_mock(record, *calls), max_samples=1, **kwargs)[0]
    assert log.status == "success", log.error
    return log, next(sample for sample in log.samples if sample.id == record["seed"])


def _verified_shelf():
    shelf = Shelf()
    assert shelf.save(SPEAK, ["speak_up"], check=lambda source: None)[0] == ["speak_up"]
    shelf.mark_verified(["speak_up"])


def test_arm_a_gets_per_step_tools(gated):
    record = _first("talk", greet=False)
    spec = record["spec"]
    who = spec["person"].get("name") or spec["person"]["description"]

    log, sample = _run(
        gated,
        "A",
        record,
        ("go_to", {"place": spec["place"]}),
        ("find_people", {"description": who}),
        ("say", {"text": "It is noon."}),
    )

    assert sample.scores["ground_truth"].value == 1.0
    assert sample.store["scripts"] == []
    assert log.eval.metadata["arm"] == "A"


def test_arm_b_runs_the_oracles_script_as_one_call(gated):
    record = _first("move_object")

    _, sample = _run(gated, "B", record, ("run_script", {"code": oracle_script(record["spec"])}))

    assert sample.scores["ground_truth"].value == 1.0
    [script] = sample.store["scripts"]
    assert script["ok"], script["error"]
    assert script["shelf_used"] == []


def test_arm_b_reports_a_failed_script_back_to_the_model(gated):
    record = _first("move_object")

    _, sample = _run(gated, "B", record, ("run_script", {"code": "open('/etc/passwd')"}))

    assert sample.scores["ground_truth"].value == 0.0
    [script] = sample.store["scripts"]
    assert not script["ok"]
    tool_messages = [m for m in _last_input(sample) if m.role == "tool"]
    assert "open" in tool_messages[0].text


def _last_input(sample):
    resolved = resolve_sample_attachments(sample)
    return [event for event in resolved.events if event.event == "model"][-1].input


def test_arm_c_calls_a_verified_function_from_the_frozen_shelf(gated):
    _verified_shelf()
    record = _first("talk", greet=False)
    spec = record["spec"]
    code = (
        f"go_to({json.dumps(spec['place'])})\n"
        "find_people()\n"
        "speak_up()"
    )

    log, sample = _run(gated, "C", record, ("run_script", {"code": code}))

    assert sample.scores["ground_truth"].value == 1.0
    [script] = sample.store["scripts"]
    assert script["shelf_used"] == ["speak_up"]
    assert log.eval.metadata["shelf_digest"] == Shelf(frozen=True).digest()
    assert "def speak_up" in _last_input(sample)[0].text
    assert Shelf().counts()["speak_up"]["run"] == 1
    assert Shelf().counts()["speak_up"]["verified"] == 2


def test_only_the_induction_arm_can_save(pool):
    _verified_shelf()
    ctx = FakeRobot(seed=1).ctx

    for arm in ("B", "C", "C2"):
        assert "save" not in ArmSetup(arm, pool=pool).episode(ctx).tools[0].args
    assert "save" in ArmSetup(INDUCT, pool=pool).episode(ctx).tools[0].args


def test_arm_c2_sees_the_scripts_but_cannot_call_them(gated):
    _verified_shelf()
    record = _first("talk", greet=False)

    _, sample = _run(
        gated,
        "C2",
        record,
        ("run_script", {"code": "speak_up()"}),
        ("run_script", {"code": oracle_script(record["spec"])}),
    )

    first, second = sample.store["scripts"]
    assert first["phase"] == "type_check"
    assert first["shelf_used"] == []
    assert second["ok"], second["error"]
    assert sample.scores["ground_truth"].value == 1.0
    assert 'say("Hello.")' in _last_input(sample)[0].text


def test_c_and_c2_have_the_same_scripts_and_differ_in_callability(pool):
    _verified_shelf()
    ctx = FakeRobot(seed=1).ctx

    callable_arm = ArmSetup("C", pool=pool).episode(ctx)
    examples_arm = ArmSetup("C2", pool=pool).episode(ctx)

    assert "Scripts may call them directly" in callable_arm.prompt
    assert "calling\nthem will fail" in examples_arm.prompt
    assert callable_arm.tools[0].args == examples_arm.tools[0].args


def test_arm_d_gets_atomics_and_bundles(gated):
    record = _first("move_object", dest={"to": "operator"})
    spec = record["spec"]

    _, sample = _run(
        gated,
        "D",
        record,
        ("fetch_object", {"what": spec["want"], "where": spec["source"]}),
        ("give_to_operator", {}),
    )

    assert sample.scores["ground_truth"].value == 1.0


def test_the_induction_arm_saves_and_the_harness_verifies(gated):
    record = _first("talk", greet=False)
    spec = record["spec"]
    code = f"{SPEAK}\ngo_to({json.dumps(spec['place'])})\nfind_people()\nspeak_up()"

    _, sample = _run(gated, INDUCT, record, ("run_script", {"code": code, "save": ["speak_up"]}))

    assert sample.scores["ground_truth"].value == 1.0
    assert sample.store["scripts"][0]["saved"] == ["speak_up"]
    assert "speak_up" in Shelf(frozen=True).names()


def test_a_failed_episode_verifies_nothing(gated):
    record = _first("move_object")

    _, sample = _run(gated, INDUCT, record, ("run_script", {"code": SPEAK + "speak_up()", "save": ["speak_up"]}))

    assert sample.scores["ground_truth"].value == 0.0
    assert sample.store["scripts"][0]["saved"] == ["speak_up"]
    assert "speak_up" not in Shelf(frozen=True).names()


def test_only_the_tool_set_and_the_shelf_text_differ_between_arms(pool):
    _verified_shelf()
    ctx = FakeRobot(seed=1).ctx

    prompts = {arm: ArmSetup(arm, pool=pool).episode(ctx).prompt for arm in ARMS}

    assert prompts["A"] == prompts["B"] == prompts["D"] == PROMPT
    assert prompts["C"].startswith(PROMPT)
    assert prompts["C2"].startswith(PROMPT)


def test_a_script_arm_describes_its_functions(pool):
    tool = ArmSetup("B", pool=pool).episode(FakeRobot(seed=1).ctx).tools[0]

    assert "go_to(place: str" in tool.description
    assert "key= lambda" in tool.description
    assert "ctx" not in tool.args


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="Unknown arm"):
        ArmSetup("Z")


def test_a_script_arm_needs_a_pool():
    with pytest.raises(ValueError, match="pool"):
        ArmSetup("B")


def test_arms_meet_the_same_world_per_command_and_epoch(gated):
    record = commands("trial", 1)[0]
    gated(commands("trial", 1), 2)
    worlds = {}
    for arm in ("A", "B"):
        log = run_episodes([arm], 1, epochs=2, model=_mock(record), max_samples=1)[0]
        assert log.status == "success", log.error
        worlds[arm] = {(s.id, s.epoch): s.store["start"] for s in log.samples}

    assert worlds["A"] == worlds["B"]
    assert worlds["A"][(record["seed"], 1)] != worlds["A"][(record["seed"], 2)]


def test_the_eval_config_reaches_every_model_call(gated):
    record = commands("trial", 1)[0]
    gated([record])

    log = run_episodes(["A"], 1, model=_mock(record), temperature=0.3, seed=11, max_samples=1)[0]

    sample = log.samples[0]
    [event] = [event for event in sample.events if event.event == "model"]
    assert event.config.temperature == 0.3
    assert event.config.seed == 11
    assert event.config.parallel_tool_calls is False


def test_an_episode_that_never_stops_is_ended_and_scored(gated, monkeypatch):
    record = commands("trial", 1)[0]
    gated([record])
    loop = lambda *_: ModelOutput.for_tool_call(MOCK, "move_relative", {"turn_deg": 1.0})  # noqa: E731
    monkeypatch.setattr("mahoraga.gym.MAX_ROUNDS", 3)

    log = run_episodes(["A"], 1, model=get_model(MOCK, custom_outputs=loop, memoize=False), max_samples=1)[0]

    sample = log.samples[0]
    assert sample.store["out_of_rounds"] is True
    assert sample.scores["ground_truth"].value == 0.0


def test_a_provider_error_ends_the_sample_after_one_attempt(gated):
    """A second attempt would replay the script's ledger events, with its usage dropped from the log."""
    _verified_shelf()
    record = _first("talk", greet=False)
    n = _index(record)
    gated(commands("trial", n))

    def reply(messages, *_):
        if any(message.role == "tool" for message in messages):
            raise ValueError("The provider fell over.")
        asked = next(message.text for message in messages if message.role == "user")
        return ModelOutput.for_tool_call(MOCK, "run_script", {"code": "speak_up()"}) if asked == record["text"] else _done()

    log = run_episodes(["C"], n, model=get_model(MOCK, custom_outputs=reply, memoize=False), max_samples=1)[0]

    sample = next(sample for sample in log.samples if sample.id == record["seed"])
    assert "The provider fell over." in sample.error.message
    assert not sample.error_retries
    assert Shelf().counts()["speak_up"]["run"] == 1


class _FakeProvider(BaseHTTPRequestHandler):
    seen: list[dict] = []
    failures: int = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(
            {"path": self.path, "auth": self.headers["Authorization"], "body": body}
        )
        if type(self).failures > 0:
            type(self).failures -= 1
            self.send_error(500)
            return
        answer = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Done."},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(answer)))
        self.end_headers()
        self.wfile.write(answer)

    def log_message(self, *args):
        pass


def _run_on_the_fake_provider(gated, monkeypatch, *, failures=0):
    server = HTTPServer(("127.0.0.1", 0), _FakeProvider)
    Thread(target=server.serve_forever, daemon=True).start()
    _FakeProvider.seen.clear()
    _FakeProvider.failures = failures
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    gated(commands("trial", 1))
    try:
        return run_episodes(
            ["A"],
            1,
            model="acme/some-model:nitro",
            base_url=f"http://127.0.0.1:{server.server_port}/api/v1",
            max_samples=1,
        )[0]
    finally:
        server.shutdown()
        server.server_close()


def test_a_transient_provider_error_is_retried_inside_the_model_call(gated, monkeypatch):
    """With no sample retry, the model's own retries are what a 500 meets. Inspect waits about 3 s."""
    log = _run_on_the_fake_provider(gated, monkeypatch, failures=1)

    sample = log.samples[0]
    assert len(_FakeProvider.seen) == 2
    assert sample.error is None
    assert not sample.error_retries
    assert "ground_truth" in sample.scores


def test_a_model_id_goes_to_the_base_url_with_the_key_from_the_environment(gated, monkeypatch):
    """The whole route of a real run, against a local stand-in for OpenRouter."""
    log = _run_on_the_fake_provider(gated, monkeypatch)

    assert log.status == "success", log.error
    assert log.samples[0].error is None
    [call] = _FakeProvider.seen
    assert call["path"] == "/api/v1/chat/completions"
    assert call["auth"] == "Bearer fake-key-for-test"
    assert call["body"]["model"] == "acme/some-model:nitro"
    assert call["body"]["temperature"] == 0.0
    assert call["body"]["seed"] == 0
    assert call["body"]["parallel_tool_calls"] is False
    assert {tool["function"]["name"] for tool in call["body"]["tools"]} >= {"go_to", "pick"}
    assert "fake-key-for-test" not in json.dumps(log.model_dump(mode="json"))
