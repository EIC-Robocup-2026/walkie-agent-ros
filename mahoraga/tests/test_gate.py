"""The oracle gate passes only on two identical, complete runs, and it holds the LLM arms back."""

import json
from pathlib import Path

import pytest
from inspect_ai.event import SampleLimitEvent
from inspect_ai.log import EvalSampleLimit, read_eval_log

import mahoraga.gate as gate
import mahoraga.gym as gym
import mahoraga.tasks as tasks
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


def test_an_oracle_that_ends_in_an_error_fails_the_gate(monkeypatch):
    real = gate.oracle_script
    monkeypatch.setattr(gate, "oracle_script", lambda spec: f"{real(spec)}\n1 / 0")

    result = run_gate(split="trial", n=2, epochs=1, max_samples=2)

    assert not result.passed
    assert any("did not run to completion" in failure for failure in result.failures)
    with pytest.raises(GateError):
        require_gate(commands("trial", 2), 1)


def test_the_gate_checks_each_script_and_each_limit_itself():
    """The oracle plays one script, so the scorer alone would pass an earlier failure it recovered from."""
    result = run_gate(split="trial", n=1, epochs=1, max_samples=1)
    good = read_eval_log(result.logs[0])
    assert gate._failures_of(good, 1) == []

    failed_first = good.model_copy(deep=True)
    scripts = failed_first.samples[0].store["scripts"]
    scripts.insert(0, {**scripts[0], "ok": False})
    assert any("did not run to completion" in failure for failure in gate._failures_of(failed_first, 1))

    no_script = good.model_copy(deep=True)
    no_script.samples[0].store["scripts"] = []
    assert any("did not run to completion" in failure for failure in gate._failures_of(no_script, 1))

    limited = good.model_copy(deep=True)
    limited.samples[0].limit = EvalSampleLimit(type="time", limit=1)
    assert any("a limit ended it" in failure for failure in gate._failures_of(limited, 1))

    limit_event = good.model_copy(deep=True)
    limit_event.samples[0].events.append(SampleLimitEvent(type="token", limit=1, message="Over."))
    assert any("a limit ended it" in failure for failure in gate._failures_of(limit_event, 1))


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
    paths = {path.relative_to(gate._REPO).as_posix() for path in gate._digested_files()}

    assert {
        "mahoraga/arms.py",
        "mahoraga/bundles.py",
        "mahoraga/checkers.py",
        "mahoraga/driver.py",
        "mahoraga/gym.py",
        "mahoraga/oracle.py",
        "mahoraga/shelf.py",
        "mahoraga/worlds.py",
        "mahoraga/data/chula_pat14_6/objects/objects.md",
        "mahoraga/data/chula_pat14_6/names/names.md",
        "src/walkie_agent/adapter.py",
        "src/walkie_agent/errors.py",
        "src/walkie_agent/locations.py",
        "src/walkie_agent/capabilities/__init__.py",
        "src/walkie_agent/capabilities/fake.py",
        "src/walkie_agent/capabilities/guards.py",
        "src/walkie_agent/capabilities/records.py",
        "src/walkie_agent/atomics/__init__.py",
        "src/walkie_agent/atomics/navigation.py",
        "pixi.lock",
    } <= paths
    assert not {"mahoraga/report.py", "mahoraga/__main__.py"} & paths
    assert len(code_digest()) == 16


def test_an_edit_to_a_guard_stales_the_gate(gated, tmp_path, monkeypatch):
    records = commands("trial", 1)
    gated(records)
    require_gate(records, 1)
    edited = tmp_path / "guards.py"
    real = gate._digested_files()
    guards = next(path for path in real if path.name == "guards.py")
    edited.write_text(guards.read_text().replace("MAX_MOVE_M", "MAX_MOVE_METRES"))
    real_bytes = Path.read_bytes

    def read_bytes(path):
        return real_bytes(edited if path == guards else path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    with pytest.raises(GateError):
        require_gate(records, 1)


def test_two_files_with_one_name_in_different_places_digest_apart(tmp_path, monkeypatch):
    for place in ("one", "two"):
        (tmp_path / place).mkdir()
        (tmp_path / place / "same.py").write_text("x = 1\n")
    monkeypatch.setattr(gate, "_REPO", tmp_path)
    digests = []
    for place in ("one", "two"):
        monkeypatch.setattr(gate, "_digested_files", lambda place=place: [tmp_path / place / "same.py"])
        digests.append(code_digest())

    assert digests[0] != digests[1]


@pytest.mark.parametrize(
    "change",
    [
        {"passed": False},
        {"passed": 1},
        {"failures": ["Sample 1 epoch 1: wrong."]},
        {"digests": ["one", "two"]},
        {"digests": ["same"]},
        {"digests": []},
        {"logs": ["/does/not/exist", "/does/not/exist"]},
        {"log_hashes": ["0" * 64, "0" * 64]},
        {"log_hashes": []},
    ],
    ids=str,
)
def test_a_result_that_is_not_a_whole_pass_does_not_count(gated, change):
    records = commands("trial", 1)
    path = gated(records)
    require_gate(records, 1)
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))

    with pytest.raises(GateError):
        require_gate(records, 1)


@pytest.mark.parametrize("key", ["passed", "failures", "digests", "logs", "log_hashes", "pairs"])
def test_a_result_missing_a_field_does_not_count(gated, key):
    records = commands("trial", 1)
    path = gated(records)
    kept = json.loads(path.read_text())
    del kept[key]
    path.write_text(json.dumps(kept))

    with pytest.raises(GateError):
        require_gate(records, 1)


def test_a_result_that_is_not_json_does_not_count(gated):
    records = commands("trial", 1)
    gated(records).write_text("{not json")

    with pytest.raises(GateError):
        require_gate(records, 1)


def test_a_log_edited_or_deleted_after_the_gate_stales_it():
    records = commands("trial", 1)
    result = run_gate(split="trial", n=1, epochs=1, max_samples=1)
    assert result.passed, result.failures
    assert len(result.log_hashes) == 2
    require_gate(records, 1)

    edited = Path(result.logs[0])
    kept = edited.read_bytes()
    edited.write_bytes(kept + b"\0")
    with pytest.raises(GateError):
        require_gate(records, 1)

    edited.write_bytes(kept)
    require_gate(records, 1)

    Path(result.logs[1]).unlink()
    with pytest.raises(GateError):
        require_gate(records, 1)


def test_the_gate_uses_no_model_call():
    result = run_gate(split="trial", n=2, epochs=1, max_samples=2)

    log = read_eval_log(result.logs[0])
    assert log.eval.model == MOCK
    assert not [event for sample in log.samples for event in sample.events if event.event == "model"]
