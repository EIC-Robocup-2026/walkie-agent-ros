"""The metrics read what the logs hold, and the paired statistics hold on known inputs."""

import pytest
from inspect_ai.model import ModelOutput, get_model

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
