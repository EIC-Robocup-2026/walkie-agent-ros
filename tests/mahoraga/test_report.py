"""The metrics read what the logs hold, and the paired statistics hold on known inputs."""

import subprocess
from types import SimpleNamespace

import pytest
from inspect_ai.log import write_eval_log
from inspect_ai.model import ModelCall, ModelOutput, ModelUsage, get_model

from walkie_agent.mahoraga.commands import commands
from walkie_agent.mahoraga.gym import run_episodes
from walkie_agent.mahoraga.oracle import oracle_script
from walkie_agent.mahoraga.report import (
    arm_metrics,
    by_command,
    compare,
    mcnemar_p,
    read_runs,
    recorded_shelf_root,
    report,
    shelf_metrics,
)
from walkie_agent.mahoraga.tasks import SCORER
from walkie_agent.scripting.shelf import Shelf, StoreRefused

MOCK = "mockllm/model"


def _mock(*calls):
    outputs = [ModelOutput.for_tool_call(MOCK, name, args) for name, args in calls]
    outputs.append(ModelOutput.from_content(MOCK, "Done."))
    return get_model(MOCK, custom_outputs=outputs, memoize=False)


def test_mcnemar_on_known_counts():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) == 1.0
    assert mcnemar_p(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_p(0, 10) == pytest.approx(2 / 1024)


def test_compare_pairs_by_command_and_drops_the_unpaired():
    baseline = {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0}
    other = {1: 1.0, 2: 1.0, 3: 1.0, 5: 1.0}

    result = compare(baseline, other, resamples=2000)

    assert result["commands"] == 3
    assert result["diff"] == pytest.approx(2 / 3)
    assert result["ci95"][0] <= result["diff"] <= result["ci95"][1]
    assert result["mcnemar_p"] == pytest.approx(0.5)


def test_compare_repeats_for_its_seed():
    baseline = {n: float(n % 2) for n in range(30)}
    other = {n: float(n % 3 == 0) for n in range(30)}

    assert compare(baseline, other) == compare(baseline, other)


def test_compare_with_nothing_in_common():
    assert compare({1: 1.0}, {2: 1.0})["commands"] == 0


def test_metrics_read_rounds_tokens_scripts_and_success(gated, tmp_path):
    record = commands("trial", 1)[0]
    gated([record])

    log = run_episodes(
        ["B"],
        1,
        model=_mock(
            ("run_script", {"code": "not_a_function()"}),
            ("run_script", {"code": oracle_script(record["spec"])}),
        ),
        max_samples=1,
        log_dir=tmp_path,
    )[0]

    metrics = arm_metrics(log)
    assert metrics["episodes"] == 1
    assert metrics["errored"] == 0
    assert metrics["M1_success"] == 1.0
    assert metrics["M2_rounds"] == 3
    assert metrics["M5_first_try_type_check"] == 0.0
    assert metrics["M6_ok_after_one_repair"] == 1.0
    assert metrics["M7_shelf_hit"] == 0.0
    assert metrics["M8_harmful_reuse"] is None
    assert metrics["M9_refused_calls"] >= 0
    assert read_runs(tmp_path)["B"].eval.task == "gpsr_B"


def test_a_call_off_the_list_counts_as_refused(gated):
    record = commands("trial", 1)[0]
    gated([record])
    code = 'open("/etc/passwd")  # type: ignore'

    log = run_episodes(["B"], 1, model=_mock(("run_script", {"code": code})), max_samples=1)[0]

    assert arm_metrics(log)["M9_refused_calls"] == 1


def test_the_report_compares_arms_by_command(gated, tmp_path):
    record = commands("trial", 1)[0]
    gated([record])
    code = oracle_script(record["spec"])
    run_episodes(["A"], 1, model=_mock(), max_samples=1, log_dir=tmp_path)
    run_episodes(["B"], 1, model=_mock(("run_script", {"code": code})), max_samples=1, log_dir=tmp_path)

    result = report(tmp_path)

    assert set(result["arms"]) == {"A", "B"}
    assert result["arms"]["A"]["M1_success"] == 0.0
    assert result["arms"]["B"]["M1_success"] == 1.0
    assert result["against_baseline"]["B"]["diff"] == 1.0
    assert result["against_baseline"]["B"]["commands"] == 1
    assert result["warning"] is None
    assert result["against_baseline"]["B"]["complete"] is True


def test_the_report_reads_the_shelf_its_logs_record_unless_told_another(gated, tmp_path):
    record = commands("trial", 1)[0]
    gated([record])
    store = tmp_path / "store"
    Shelf(store).save('def speak():\n    """One."""\n    say("one")\n', ["speak"])
    run_episodes(["C"], 1, model=_mock(), max_samples=1, log_dir=tmp_path / "logs", shelf_root=store)

    assert report(tmp_path / "logs")["shelf"] == {"root": str(store), "M10_shelf_size": 1, "M10_shelf_churn": 0}
    other = report(tmp_path / "logs", shelf_root=tmp_path / "other")["shelf"]
    assert other == {"root": str(tmp_path / "other"), "M10_shelf_size": 0, "M10_shelf_churn": 0}


def test_logs_that_record_no_shelf_report_the_default_store_by_name(isolated_tools_home, tmp_path):
    (tmp_path / "logs").mkdir()

    assert report(tmp_path / "logs")["shelf"]["root"] == str(isolated_tools_home)


def test_a_shelf_root_a_log_records_is_only_read(monkeypatch, tmp_path):
    def no_git(*args, **kwargs):
        raise AssertionError("the report ran a subprocess")

    monkeypatch.setattr(subprocess, "run", no_git)
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    named = tmp_path / "repo" / "store"

    assert shelf_metrics(named) == {"M10_shelf_size": 0, "M10_shelf_churn": 0}
    assert not named.exists()


def test_logs_that_ran_on_different_shelves_name_no_one_shelf():
    def logged(root):
        return SimpleNamespace(eval=SimpleNamespace(metadata={"arm": "-", "shelf_root": root}))

    assert recorded_shelf_root({"A": logged(None), "C": logged("/a")}) == "/a"
    assert recorded_shelf_root({"A": logged(None)}) is None
    with pytest.raises(StoreRefused, match="different shelves"):
        recorded_shelf_root({"C": logged("/a"), "C2": logged("/b")})


def _used(output):
    output.usage = ModelUsage(input_tokens=100, output_tokens=1, total_tokens=101)
    return output


def _errors_after_its_script(code):
    """A model that runs `code`, spending tokens, and then fails on every later call."""

    def reply(messages, *_):
        if any(message.role == "tool" for message in messages):
            raise ValueError("A harmless injected provider failure.")
        return _used(ModelOutput.for_tool_call(MOCK, "run_script", {"code": code}))

    return get_model(MOCK, custom_outputs=reply, memoize=False)


def _one_success_two_errors(gated):
    """One command's three epochs. The log is built from real runs, since a mock cannot fail on one epoch only."""
    record = commands("trial", 1)[0]
    gated([record])
    code = oracle_script(record["spec"])
    good = run_episodes(["B"], 1, model=_mock(("run_script", {"code": code})), max_samples=1)[0]
    bad = run_episodes(["B"], 1, model=_errors_after_its_script(code), max_samples=1)[0]
    assert bad.samples[0].error is not None
    assert bad.samples[0].model_usage
    log = good.model_copy(deep=True)
    log.samples = [good.samples[0], bad.samples[0], bad.samples[0].model_copy(deep=True)]
    for epoch, sample in enumerate(log.samples, 1):
        sample.epoch = epoch
    log.eval.config.epochs = 3
    return record, log


def _one_cached_success(gated, tmp_path):
    """A passing episode of two model calls, each with cached input in its usage."""
    record = commands("trial", 1)[0]
    gated([record])
    outputs = [
        ModelOutput.for_tool_call(MOCK, "run_script", {"code": oracle_script(record["spec"])}),
        ModelOutput.from_content(MOCK, "Done."),
    ]
    for output in outputs:
        output.usage = ModelUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=115,
            input_tokens_cache_read=90,
            input_tokens_cache_write=10,
        )
    model = get_model(MOCK, custom_outputs=outputs, memoize=False)
    return run_episodes(["B"], 1, model=model, max_samples=1, log_dir=tmp_path)[0]


def _model_events(log):
    return [event for sample in log.samples for event in sample.events if event.event == "model"]


def test_tokens_count_the_input_that_was_read_from_or_written_to_the_cache(gated, tmp_path):
    metrics = arm_metrics(_one_cached_success(gated, tmp_path))

    assert metrics["M3_tokens_in"] == 2 * (10 + 90 + 10)
    assert metrics["M3_tokens_in_cached"] == 2 * 90
    assert metrics["M3_tokens_out"] == 2 * 5
    assert metrics["M3_tokens_per_success"] == 2 * 115


def test_cost_is_what_the_provider_recorded_and_a_call_without_a_record_is_counted(gated, tmp_path):
    log = _one_cached_success(gated, tmp_path)
    first, second = _model_events(log)
    first.call = ModelCall(request={}, response={"usage": {"cost": 0.25}})
    second.call = None

    metrics = arm_metrics(log)

    assert metrics["M3_cost_usd"] == 0.25
    assert metrics["M3_cost_per_success_usd"] == 0.25
    assert metrics["M3_cost_unrecorded_calls"] == 1


def test_cost_is_unknown_when_no_call_has_a_record(gated, tmp_path):
    log = _one_cached_success(gated, tmp_path)
    for event in _model_events(log):
        event.call = ModelCall(request={}, response={"usage": {"total_tokens": 1}})

    metrics = arm_metrics(log)

    assert metrics["M3_cost_usd"] is None
    assert metrics["M3_cost_per_success_usd"] is None
    assert metrics["M3_cost_unrecorded_calls"] == 2


def test_an_errored_epoch_stays_in_the_denominator_and_its_usage_counts(gated):
    record, log = _one_success_two_errors(gated)
    tokens = sum(u.input_tokens + u.output_tokens for s in log.samples for u in s.model_usage.values())
    good_tokens = sum(u.input_tokens + u.output_tokens for u in log.samples[0].model_usage.values())

    metrics = arm_metrics(log)

    assert tokens >= good_tokens + 2 * 101
    assert by_command(log) == {record["seed"]: pytest.approx(1 / 3)}
    assert (metrics["episodes"], metrics["errored"], metrics["never_ran"]) == (3, 2, 0)
    assert metrics["M1_success"] == pytest.approx(1 / 3)
    assert metrics["M3_tokens_in"] + metrics["M3_tokens_out"] == tokens
    assert metrics["M3_tokens_per_success"] == tokens
    assert metrics["complete"] is False


def test_an_errored_episode_still_counts_in_the_script_metrics_and_rounds(gated):
    gated(commands("trial", 1))
    code = 'open("/etc/passwd")  # type: ignore'

    log = run_episodes(["B"], 1, model=_errors_after_its_script(code), max_samples=1)[0]

    assert log.samples[0].error is not None
    metrics = arm_metrics(log)
    assert metrics["M2_rounds"] == 2
    assert metrics["M6_ok_after_one_repair"] == 0.0
    assert metrics["M9_refused_calls"] == 1


def test_an_episode_that_never_ran_counts_as_a_failure(gated):
    record = commands("trial", 1)[0]
    gated([record], 2)
    code = oracle_script(record["spec"])
    log = run_episodes(
        ["B"], 1, epochs=2, model=_always_runs(code), max_samples=1
    )[0]
    assert by_command(log)[record["seed"]] == 1.0
    log.samples = log.samples[:1]
    log.status = "cancelled"

    metrics = arm_metrics(log)

    assert by_command(log)[record["seed"]] == pytest.approx(0.5)
    assert (metrics["episodes"], metrics["errored"], metrics["never_ran"]) == (2, 0, 1)
    assert metrics["complete"] is False


def _always_runs(code):
    def reply(messages, *_):
        if any(message.role == "tool" for message in messages):
            return ModelOutput.from_content(MOCK, "Done.")
        return ModelOutput.for_tool_call(MOCK, "run_script", {"code": code})

    return get_model(MOCK, custom_outputs=reply, memoize=False)


def test_a_log_that_did_not_end_cleanly_is_not_complete(gated):
    record = commands("trial", 1)[0]
    gated([record])
    log = run_episodes(["B"], 1, model=_always_runs(oracle_script(record["spec"])), max_samples=1)[0]
    assert arm_metrics(log)["complete"] is True

    log.status = "cancelled"

    assert arm_metrics(log)["complete"] is False


def test_the_report_says_plainly_when_a_comparison_is_incomplete(gated, tmp_path):
    record = commands("trial", 1)[0]
    gated([record])
    code = oracle_script(record["spec"])
    run_episodes(["A"], 1, model=_mock(), max_samples=1, log_dir=tmp_path)
    run_episodes(["B"], 1, model=_errors_after_its_script(code), max_samples=1, log_dir=tmp_path)

    result = report(tmp_path)

    assert "Incomplete" in result["warning"]
    assert "B: 1 of 1 errored" in result["warning"]
    assert result["arms"]["A"]["complete"] is True
    assert result["arms"]["B"]["complete"] is False
    assert result["arms"]["B"]["M1_success"] == 0.0
    assert result["arms"]["B"]["M3_tokens_in"] > 0
    assert result["against_baseline"]["B"]["complete"] is False
    assert result["against_baseline"]["B"]["commands"] == 1


def test_a_comparison_over_different_commands_is_not_complete(gated, tmp_path):
    gated(commands("trial", 2))
    done = get_model(MOCK, custom_outputs=lambda *_: ModelOutput.from_content(MOCK, "Done."), memoize=False)
    run_episodes(["A"], 2, model=done, max_samples=1, log_dir=tmp_path)
    run_episodes(["B"], 1, model=done, max_samples=1, log_dir=tmp_path)

    result = report(tmp_path)

    assert result["arms"]["A"]["complete"] is True
    assert result["arms"]["B"]["complete"] is True
    assert result["against_baseline"]["B"]["commands"] == 1
    assert result["against_baseline"]["B"]["complete"] is False


def _says_done():
    return get_model(MOCK, custom_outputs=lambda *_: ModelOutput.from_content(MOCK, "Done."), memoize=False)


def test_a_comparison_over_different_epochs_is_not_complete(gated, tmp_path):
    gated(commands("trial", 1), 3)
    run_episodes(["A"], 1, model=_says_done(), max_samples=1, log_dir=tmp_path)
    run_episodes(["B"], 1, epochs=3, model=_says_done(), max_samples=1, log_dir=tmp_path)

    result = report(tmp_path)

    assert result["arms"]["A"]["complete"] is True
    assert result["arms"]["B"]["complete"] is True
    assert result["against_baseline"]["B"]["commands"] == 1
    assert result["against_baseline"]["B"]["complete"] is False


def test_a_comparison_over_the_same_epochs_is_complete(gated, tmp_path):
    gated(commands("trial", 1), 2)
    run_episodes(["A"], 1, epochs=2, model=_says_done(), max_samples=1, log_dir=tmp_path)
    run_episodes(["B"], 1, epochs=2, model=_says_done(), max_samples=1, log_dir=tmp_path)

    assert report(tmp_path)["against_baseline"]["B"]["complete"] is True


@pytest.mark.parametrize("changed", [0, 1], ids=["first_command", "last_command"])
def test_a_comparison_of_one_id_with_different_text_is_not_complete(gated, tmp_path, changed):
    gated(commands("trial", 2))
    run_episodes(["A"], 2, model=_says_done(), max_samples=1, log_dir=tmp_path)
    other = run_episodes(["B"], 2, model=_says_done(), max_samples=1, log_dir=tmp_path)[0]
    assert report(tmp_path)["against_baseline"]["B"]["complete"] is True

    other.samples[changed].input = "An entirely different command."
    write_eval_log(other)

    result = report(tmp_path)

    assert result["arms"]["A"]["complete"] is True
    assert result["arms"]["B"]["complete"] is True
    assert result["against_baseline"]["B"]["commands"] == 2
    assert result["against_baseline"]["B"]["complete"] is False


def _one_of_six_logged_without_ids(gated):
    """A real run of two commands over three epochs, cut down to one passing episode and no ids."""
    records = commands("trial", 2)
    gated(records, 3)
    log = run_episodes(
        ["B"], 2, epochs=3, model=_always_runs(oracle_script(records[0]["spec"])), max_samples=1
    )[0]
    seed = records[0]["seed"]
    log.samples = [sample for sample in log.samples if (sample.id, sample.epoch) == (seed, 1)]
    assert float(log.samples[0].scores[SCORER].value) == 1.0
    log.eval.dataset.sample_ids = None
    return seed, log


def test_a_log_without_ids_still_schedules_every_episode(gated):
    seed, log = _one_of_six_logged_without_ids(gated)

    metrics = arm_metrics(log)

    assert log.status == "success"
    assert (metrics["episodes"], metrics["errored"], metrics["never_ran"]) == (6, 0, 5)
    assert metrics["complete"] is False
    assert by_command(log)[seed] == pytest.approx(1 / 3)
    assert metrics["M1_success"] == pytest.approx(1 / 6)


def test_a_log_without_ids_that_lost_nothing_is_complete(gated):
    records = commands("trial", 2)
    gated(records, 2)
    log = run_episodes(["A"], 2, epochs=2, model=_says_done(), max_samples=1)[0]
    log.eval.dataset.sample_ids = None

    metrics = arm_metrics(log)

    assert (metrics["episodes"], metrics["never_ran"]) == (4, 0)
    assert metrics["complete"] is True


def test_commands_lost_from_two_logs_without_ids_pair_with_nothing(gated):
    _, log = _one_of_six_logged_without_ids(gated)
    other = log.model_copy(deep=True)
    other.eval.eval_id = "another-eval"

    assert compare(by_command(log), by_command(other))["commands"] == 1


def test_by_command_means_the_epochs(gated):
    record = commands("trial", 1)[0]
    gated([record], 2)
    code = oracle_script(record["spec"])
    # The first epoch's episode runs the script, and every later call replies with text.
    outputs = iter([ModelOutput.for_tool_call(MOCK, "run_script", {"code": code})])

    def reply(*_):
        return next(outputs, ModelOutput.from_content(MOCK, "Done."))

    log = run_episodes(
        ["B"], 1, epochs=2, model=get_model(MOCK, custom_outputs=reply, memoize=False), max_samples=1
    )[0]

    assert by_command(log)[record["seed"]] == pytest.approx(0.5)


def test_shelf_metrics_count_size_and_churn():
    shelf = Shelf()
    one = 'def speak():\n    """One."""\n    say("one")\n'
    two = 'def speak():\n    """Two."""\n    say("two")\n'
    shelf.save(one, ["speak"], check=lambda source: None)
    shelf.save(two, ["speak"], check=lambda source: None)

    assert shelf_metrics() == {"M10_shelf_size": 1, "M10_shelf_churn": 1}
