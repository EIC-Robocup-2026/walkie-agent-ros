"""Each arm runs once on Inspect's mock model, through the same gate, driver and scorer."""

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest
from inspect_ai.log import resolve_sample_attachments
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from mahoraga.arms import ARMS, INDUCT, PROMPT, ArmSetup
from mahoraga.commands import commands
from mahoraga.gym import gpsr_task, run_episodes
from mahoraga.oracle import oracle_script
from mahoraga.report import arm_metrics
from mahoraga.shelf import Shelf
from walkie_agent.capabilities.fake import FakeRobot

MOCK = "mockllm/model"
SPEAK = 'def speak_up():\n    """Say hello."""\n    say("Hello.")\n'


def _first(kind, split="trial", **fields):
    for record in commands(split, 200):
        spec = record["spec"]
        if spec["kind"] == kind and all(spec[key] == value for key, value in fields.items()):
            return record
    raise AssertionError(f"No {kind} in the first 200 commands.")


def _index(record, split="trial"):
    """The n of commands(split, n) that ends with `record`."""
    return [r["seed"] for r in commands(split, 200)].index(record["seed"]) + 1


def _done():
    return ModelOutput.from_content(MOCK, "Done.")


def _mock(record, *calls):
    """A model that makes these tool calls in order for `record`, and replies with text otherwise."""
    outputs = iter([ModelOutput.for_tool_call(MOCK, name, args) for name, args in calls])

    def reply(messages, tools, tool_choice, config):
        asked = next(message.text for message in messages if message.role == "user")
        return next(outputs, _done()) if asked == record["text"] else _done()

    return get_model(MOCK, custom_outputs=reply, memoize=False)


def _run(gated, arm, record, *calls, split="trial", **kwargs):
    n = _index(record, split)
    gated(commands(split, n))
    log = run_episodes([arm], n, split=split, model=_mock(record, *calls), max_samples=1, **kwargs)[0]
    assert log.status == "success", log.error
    return log, next(sample for sample in log.samples if sample.id == record["seed"])


def _induct(gated, record, *calls):
    return _run(gated, INDUCT, record, *calls, split="train")


def _verified_shelf():
    shelf = Shelf()
    assert shelf.save(SPEAK, ["speak_up"], check=lambda source: None)[0] == ["speak_up"]
    shelf.mark_verified(shelf.versions())


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
    record = _first("talk", "train", greet=False)
    code = SPEAK + _talk_to(record["spec"], "speak_up()")

    _, sample = _induct(gated, record, ("run_script", {"code": code, "save": ["speak_up"]}))

    assert sample.scores["ground_truth"].value == 1.0
    assert sample.store["scripts"][0]["saved"] == ["speak_up"]
    assert "speak_up" in Shelf(frozen=True).names()


def test_a_failed_episode_verifies_nothing(gated):
    record = _first("move_object", "train")

    _, sample = _induct(gated, record, ("run_script", {"code": SPEAK + "speak_up()", "save": ["speak_up"]}))

    assert sample.scores["ground_truth"].value == 0.0
    assert sample.store["scripts"][0]["saved"] == ["speak_up"]
    assert "speak_up" not in Shelf(frozen=True).names()


def _talk_to(spec, then):
    return f"go_to({json.dumps(spec['place'])})\nfind_people()\n{then}\n"


@pytest.mark.parametrize("split", ["trial", "heldout"])
def test_induction_is_refused_off_the_train_split(gated, monkeypatch, split):
    gated(commands(split, 1))
    monkeypatch.setattr("mahoraga.gym.inspect_eval", lambda *a, **k: pytest.fail("It ran."))

    with pytest.raises(ValueError, match="only on the train split"):
        run_episodes([INDUCT], 1, split=split, model=MOCK)


@pytest.mark.parametrize("arms", [[INDUCT, "C"], ["C", INDUCT], ["A", INDUCT]])
def test_induction_is_refused_beside_another_arm(gated, monkeypatch, arms):
    gated(commands("train", 1))
    monkeypatch.setattr("mahoraga.gym.inspect_eval", lambda *a, **k: pytest.fail("It ran."))

    with pytest.raises(ValueError, match="alone"):
        run_episodes(arms, 1, split="train", model=MOCK)


@pytest.mark.parametrize("split", ["trial", "heldout"])
def test_a_task_for_induction_takes_only_training_commands(gated, pool, split):
    records = commands(split, 1)
    gated(records)

    with pytest.raises(ValueError, match="train split"):
        gpsr_task(INDUCT, records, pool=pool)


def test_the_plans_flow_trains_first_then_evaluates_held_out_on_the_frozen_shelf(gated):
    trained = _first("talk", "train", greet=False)
    _induct(gated, trained, ("run_script", {"code": SPEAK + _talk_to(trained["spec"], "speak_up()"), "save": ["speak_up"]}))
    assert Shelf(frozen=True).names() == ["speak_up"]

    target = _first("talk", "heldout", greet=False)
    log, sample = _run(
        gated,
        "C",
        target,
        ("run_script", {"code": _talk_to(target["spec"], "speak_up()")}),
        split="heldout",
    )

    assert sample.scores["ground_truth"].value == 1.0
    assert sample.store["scripts"][0]["shelf_used"] == ["speak_up"]
    assert log.eval.metadata["shelf_digest"] == Shelf(frozen=True).digest()


def test_a_function_saved_but_never_run_is_not_verified(gated):
    record = _first("move_object", "train")
    code = f'def untested_bad() -> str:\n    return go_to("does not exist")\n\n{oracle_script(record["spec"])}'

    _, sample = _induct(gated, record, ("run_script", {"code": code, "save": ["untested_bad"]}))

    assert sample.scores["ground_truth"].value == 1.0
    assert sample.store["scripts"][0]["saved"] == ["untested_bad"]
    assert Shelf().counts()["untested_bad"] == {"saved": 1, "run": 0, "fail": 0, "verified": 0}
    assert Shelf(frozen=True).names() == []


def test_a_function_saved_in_one_script_and_run_in_the_next_is_verified(gated):
    record = _first("talk", "train", greet=False)

    _, sample = _induct(
        gated,
        record,
        ("run_script", {"code": SPEAK, "save": ["speak_up"]}),
        ("run_script", {"code": _talk_to(record["spec"], "speak_up()")}),
    )

    assert sample.scores["ground_truth"].value == 1.0
    assert sample.store["scripts"][1]["shelf_used"] == ["speak_up"]
    assert Shelf(frozen=True).names() == ["speak_up"]


def test_a_rewrite_that_never_ran_earns_nothing_from_the_version_that_did(gated):
    record = _first("talk", "train", greet=False)
    rewrite = 'def speak_up():\n    """Say goodbye."""\n    say("Goodbye.")\n'

    _, sample = _induct(
        gated,
        record,
        ("run_script", {"code": SPEAK + _talk_to(record["spec"], "speak_up()"), "save": ["speak_up"]}),
        ("run_script", {"code": rewrite, "save": ["speak_up"]}),
    )

    assert sample.scores["ground_truth"].value == 1.0
    assert "Goodbye" in (Shelf().root / "speak_up.py").read_text()
    assert Shelf().counts()["speak_up"]["verified"] == 0
    assert Shelf(frozen=True).names() == []


def test_a_function_that_raised_once_is_not_verified_by_another_call_that_returned(gated):
    record = _first("talk", "train", greet=False)
    spec = record["spec"]
    code = (
        "def reach(place: str) -> str:\n    return go_to(place)\n\n"
        'try:\n    reach("does not exist")\nexcept RuntimeError:\n    pass\n'
        f"reach({json.dumps(spec['place'])})\nfind_people()\nsay(\"Hello.\")\n"
    )

    _, sample = _induct(gated, record, ("run_script", {"code": code, "save": ["reach"]}))

    assert sample.scores["ground_truth"].value == 1.0
    assert sample.store["scripts"][0]["saved"] == ["reach"]
    assert Shelf().counts()["reach"]["verified"] == 0
    assert Shelf(frozen=True).names() == []


def test_a_shelf_name_in_dead_code_is_no_hit_and_earns_nothing(gated):
    _verified_shelf()
    record = _first("move_object")
    code = f"if False:\n    speak_up()\n{oracle_script(record['spec'])}"

    log, sample = _run(gated, "C", record, ("run_script", {"code": code}))

    assert sample.scores["ground_truth"].value == 1.0
    assert sample.store["scripts"][0]["shelf_used"] == []
    assert Shelf().counts()["speak_up"] == {"saved": 1, "run": 0, "fail": 0, "verified": 1}
    assert arm_metrics(log)["M7_shelf_hit"] == 0.0


def test_a_shelf_function_that_raised_under_a_catch_is_harmful_and_earns_nothing(gated):
    shelf = Shelf()
    bad = 'def bad_move():\n    """Go nowhere."""\n    go_to("does not exist")\n'
    shelf.save(bad, ["bad_move"], check=lambda source: None)
    shelf.mark_verified(shelf.versions())
    record = _first("move_object")
    code = f"try:\n    bad_move()\nexcept RuntimeError:\n    pass\n{oracle_script(record['spec'])}"

    log, sample = _run(gated, "C", record, ("run_script", {"code": code}))

    assert sample.scores["ground_truth"].value == 1.0
    assert sample.store["scripts"][0]["shelf_raised"] == ["bad_move"]
    assert Shelf().counts()["bad_move"]["verified"] == 1
    metrics = arm_metrics(log)
    assert metrics["M7_shelf_hit"] == 1.0
    assert metrics["M8_harmful_reuse"] == 1.0


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


@pytest.mark.parametrize(("ending", "phase"), [("while True:\n    pass", "budget"), ("1 / 0", "runtime")])
def test_a_last_script_cut_off_after_the_work_scores_zero(gated, ending, phase):
    record = _first("move_object")
    code = f"{oracle_script(record['spec'])}\n{ending}"

    _, sample = _run(gated, "B", record, ("run_script", {"code": code}))

    assert sample.store["scripts"][-1]["phase"] == phase
    assert sample.scores["ground_truth"].value == 0.0
    assert "did not run to completion" in sample.scores["ground_truth"].explanation


def test_a_failed_script_the_agent_recovered_from_still_counts(gated):
    record = _first("move_object")

    _, sample = _run(
        gated,
        "B",
        record,
        ("run_script", {"code": "1 / 0"}),
        ("run_script", {"code": oracle_script(record["spec"])}),
    )

    assert [script["ok"] for script in sample.store["scripts"]] == [False, True]
    assert sample.scores["ground_truth"].value == 1.0


def test_a_failed_script_after_the_work_was_done_scores_zero(gated):
    record = _first("move_object")

    _, sample = _run(
        gated,
        "B",
        record,
        ("run_script", {"code": oracle_script(record["spec"])}),
        ("run_script", {"code": "1 / 0"}),
    )

    assert [script["ok"] for script in sample.store["scripts"]] == [True, False]
    assert sample.scores["ground_truth"].value == 0.0


def _after_the_work(record, then):
    """A model that does the whole job in one script, then answers each later call with `then()`."""

    def reply(messages, *_):
        asked = next(message.text for message in messages if message.role == "user")
        if asked != record["text"]:
            return _done()
        if any(message.role == "tool" for message in messages):
            return then()
        return ModelOutput.for_tool_call(MOCK, "run_script", {"code": oracle_script(record["spec"])})

    return get_model(MOCK, custom_outputs=reply, memoize=False)


def _sample_of(gated, record, model):
    n = _index(record)
    gated(commands("trial", n))
    log = run_episodes(["B"], n, model=model, max_samples=1)[0]
    return next(sample for sample in log.samples if sample.id == record["seed"])


def test_running_out_of_rounds_after_the_work_scores_zero(gated, monkeypatch):
    record = _first("move_object")
    monkeypatch.setattr("mahoraga.gym.MAX_ROUNDS", 3)

    sample = _sample_of(
        gated,
        record,
        _after_the_work(record, lambda: ModelOutput.for_tool_call(MOCK, "run_script", {"code": 'say("still going")'})),
    )

    assert sample.store["out_of_rounds"] is True
    assert sample.store["scripts"][-1]["ok"]
    assert sample.scores["ground_truth"].value == 0.0
    assert "out of rounds" in sample.scores["ground_truth"].explanation


def test_the_token_limit_after_the_work_scores_zero(gated):
    record = _first("move_object")

    def over_the_limit():
        output = _done()
        output.usage = ModelUsage(input_tokens=250_000, output_tokens=1, total_tokens=250_001)
        return output

    sample = _sample_of(gated, record, _after_the_work(record, over_the_limit))

    assert sample.limit.type == "token"
    assert sample.store["scripts"][-1]["ok"]
    assert sample.scores["ground_truth"].value == 0.0
    assert "A limit ended the episode" in sample.scores["ground_truth"].explanation


def test_the_time_limit_after_the_work_scores_zero(gated, monkeypatch):
    record = _first("move_object")
    monkeypatch.setattr("mahoraga.gym._TIME_LIMIT_S", 1)

    def too_slow():
        time.sleep(1.1)
        return _done()

    sample = _sample_of(gated, record, _after_the_work(record, too_slow))

    assert sample.limit.type == "time"
    assert sample.scores["ground_truth"].value == 0.0
    assert "A limit ended the episode" in sample.scores["ground_truth"].explanation


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
