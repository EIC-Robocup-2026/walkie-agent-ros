"""The oracle gate passes only on two identical, complete runs, and it holds the LLM arms back."""

import json
from pathlib import Path

import pytest
from inspect_ai.log import read_eval_log

import mahoraga.gate as gate
import mahoraga.gym as gym
import mahoraga.tasks as tasks
from mahoraga import home
from mahoraga.commands import commands
from mahoraga.gate import (
    GateError,
    canonical,
    code_digest,
    first_difference,
    require_gate,
    run_gate,
)
from mahoraga.gym import gpsr_task, run_episodes

MOCK = "mockllm/model"


def test_a_pass_scores_everything_twice_with_identical_logs():
    result = run_gate(split="trial", n=4, epochs=2, max_samples=2)

    assert result.passed, result.failures
    assert result.episodes_per_pass == 8
    assert result.digests[0] == result.digests[1]
    one, two = (read_eval_log(path) for path in result.logs)
    assert one.location != two.location
    assert len(one.samples) == len(two.samples) == 8
    assert all(sample.scores["ground_truth"].value == 1.0 for sample in one.samples)
    assert canonical(one) == canonical(two)
    assert json.loads(Path(result.path).read_text())["passed"] is True


def test_the_canonical_log_keeps_what_the_episode_did():
    result = run_gate(split="trial", n=2, epochs=1, max_samples=2)

    data = canonical(read_eval_log(result.logs[0]))

    assert [sample["id"] for sample in data["samples"]] == sorted(
        (record["seed"] for record in commands("trial", 2)), key=str
    )
    sample = data["samples"][0]
    assert sample["store"]["end"]["calls"]
    assert sample["store"]["scripts"][0]["ok"] is True
    assert sample["events"]


def test_a_checker_that_fails_fails_the_gate(monkeypatch):
    monkeypatch.setattr(tasks, "check", lambda spec, start, end: (False, "Wrong."))

    result = run_gate(split="trial", n=2, epochs=1, max_samples=2)

    assert not result.passed
    assert any("Wrong." in failure for failure in result.failures)
    records = commands("trial", 2)
    with pytest.raises(GateError):
        require_gate(records, 1)


def test_two_runs_that_differ_fail_the_gate(monkeypatch):
    runs = []
    real = gate.oracle_script

    def changes_each_call(spec):
        runs.append(spec)
        return f"# call {len(runs)}\n{real(spec)}"

    monkeypatch.setattr(gate, "oracle_script", changes_each_call)

    result = run_gate(split="trial", n=2, epochs=1, max_samples=1)

    assert not result.passed
    assert result.digests[0] != result.digests[1]
    assert any("The logs differ at $.samples" in failure for failure in result.failures)


def test_first_difference_names_the_path():
    assert first_difference({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]}) == "$.a[1].b: 2 against 3"
    assert first_difference({"a": 1}, {"a": 1}) is None
    assert first_difference({"a": 1}, {"b": 1}) == "$.a"
    assert first_difference([1], [1, 2]) == "$: length 1 against 2"


def test_require_gate_refuses_with_no_gate():
    with pytest.raises(GateError, match="python -m mahoraga gate"):
        require_gate(commands("trial", 1), 1)


def test_an_llm_arm_refuses_to_start_without_a_gate():
    with pytest.raises(GateError):
        gpsr_task("A", commands("trial", 1))


def test_run_episodes_refuses_before_any_arm_starts(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("An arm started.")

    monkeypatch.setattr(gym, "inspect_eval", never)
    monkeypatch.setattr(gym, "open_pool", never)

    with pytest.raises(GateError):
        run_episodes(["A", "B"], 1, model=MOCK)


def test_a_passing_gate_lets_the_same_data_through():
    assert run_gate(split="trial", n=2, epochs=2, max_samples=2).passed

    require_gate(commands("trial", 2), 2)
    require_gate(commands("trial", 1), 1)


@pytest.mark.parametrize(
    ("split", "n", "epochs", "category"),
    [
        ("trial", 3, 2, "mixed"),
        ("trial", 2, 3, "mixed"),
        ("train", 2, 2, "mixed"),
        ("trial", 2, 2, "people"),
    ],
)
def test_a_gate_does_not_cover_other_data(gated, split, n, epochs, category):
    gated(commands("trial", 2), 2)

    with pytest.raises(GateError):
        require_gate(commands(split, n, category), epochs)


def test_a_world_the_gate_did_not_see_is_refused(gated, monkeypatch):
    records = commands("trial", 2)
    gated(records)
    real = gate.build_world

    def other_world(spec, seed):
        world = real(spec, seed)
        world["places"]["table"]["x"] += 1.0
        return world

    monkeypatch.setattr(gate, "build_world", other_world)

    with pytest.raises(GateError):
        require_gate(records, 1)


def test_a_command_the_gate_did_not_see_is_refused(gated):
    records = commands("trial", 2)
    gated(records)

    with pytest.raises(GateError):
        require_gate([{**records[0], "text": "Something else."}, records[1]], 1)


def test_a_code_change_stales_the_gate(gated, monkeypatch):
    records = commands("trial", 2)
    gated(records)
    require_gate(records, 1)

    monkeypatch.setattr(gate, "code_digest", lambda: "another")

    with pytest.raises(GateError):
        require_gate(records, 1)


def test_the_digest_covers_the_files_an_oracle_pass_depends_on():
    names = {path.name for path in gate._digested_files()}

    assert {"checkers.py", "worlds.py", "oracle.py", "driver.py", "fake.py", "navigation.py"} <= names
    assert {"objects.md", "names.md"} <= names
    assert len(code_digest()) == 16


def test_a_failed_result_on_disk_does_not_count(gated):
    records = commands("trial", 1)
    path = home() / "gate" / f"gate-{code_digest()}-failed.json"
    path.parent.mkdir(parents=True)
    pairs = sorted(gate.pair_digests(records, 1))
    path.write_text(json.dumps({"passed": False, "code_digest": code_digest(), "pairs": pairs}))

    with pytest.raises(GateError):
        require_gate(records, 1)


def test_the_gate_uses_no_model_call():
    result = run_gate(split="trial", n=2, epochs=1, max_samples=2)

    log = read_eval_log(result.logs[0])
    assert log.eval.model == MOCK
    assert not [event for sample in log.samples for event in sample.events if event.event == "model"]
