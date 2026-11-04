"""The metrics read what the logs hold, and the paired statistics hold on known inputs."""

import pytest
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from mahoraga.commands import commands
from mahoraga.gym import run_episodes
from mahoraga.oracle import oracle_script
from mahoraga.report import arm_metrics, by_command, compare, mcnemar_p, read_runs, report, shelf_metrics
from mahoraga.shelf import Shelf

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


def test_an_errored_episode_still_counts_in_the_script_metrics(gated):
    _, log = _one_success_two_errors(gated)

    metrics = arm_metrics(log)

    assert metrics["M5_first_try_type_check"] == 1.0
    assert metrics["M7_shelf_hit"] == 0.0
    assert metrics["M2_rounds"] == pytest.approx(2.0)


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
    records = commands("trial", 2)
    gated(records)
    run_episodes(["A"], 2, model=_mock(), max_samples=1, log_dir=tmp_path)
    run_episodes(["B"], 1, model=_mock(), max_samples=1, log_dir=tmp_path)

    result = report(tmp_path)

    assert result["arms"]["B"]["complete"] is True
    assert result["against_baseline"]["B"]["commands"] == 1
    assert result["against_baseline"]["B"]["complete"] is False


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
