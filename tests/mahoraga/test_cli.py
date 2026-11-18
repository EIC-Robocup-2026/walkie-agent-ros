"""The command line has no default that starts a large run."""

import json

import pytest

import walkie_agent.evals.mahoraga.__main__ as cli
from walkie_agent.evals.mahoraga import home
from walkie_agent.evals.mahoraga.commands import commands
from walkie_agent.scripting.shelf import StoreRefused


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--arms", "A"],
        ["run", "--commands", "1"],
        ["run"],
        ["gate", "--split", "trial"],
    ],
)
def test_a_required_argument_is_required(argv, monkeypatch):
    monkeypatch.setattr(cli, "run_gate", lambda **kwargs: pytest.fail("It ran."))
    monkeypatch.setattr(cli, "run_episodes", lambda *a, **k: pytest.fail("It ran."))

    with pytest.raises(SystemExit) as stopped:
        cli.main(argv)

    assert stopped.value.code == 2


def test_run_is_refused_without_a_gate_and_says_what_to_run(capsys):
    assert cli.main(["run", "--arms", "A,B", "--commands", "2"]) == 2

    assert "python -m walkie_agent.evals.mahoraga gate" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arms", "commands", "message"),
    [("A,Z", "1", "Unknown arm: Z"), ("A", "0", "at least 1")],
)
def test_bad_arguments_are_refused_before_an_arm_starts(gated, capsys, monkeypatch, arms, commands, message):
    monkeypatch.setattr("walkie_agent.evals.mahoraga.gym.inspect_eval", lambda *a, **k: pytest.fail("It ran."))

    assert cli.main(["run", "--arms", arms, "--commands", commands]) == 2

    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arms", "split", "message"),
    [
        ("induct", "trial", "only on the train split"),
        ("induct", "heldout", "only on the train split"),
        ("induct,C", "train", "alone"),
        ("C,induct", "heldout", "alone"),
    ],
)
def test_induction_off_the_train_split_or_beside_an_arm_is_refused(gated, capsys, monkeypatch, arms, split, message):
    gated(commands(split, 1))
    monkeypatch.setattr("walkie_agent.evals.mahoraga.gym.inspect_eval", lambda *a, **k: pytest.fail("It ran."))

    assert cli.main(["run", "--arms", arms, "--commands", "1", "--split", split]) == 2

    assert message in capsys.readouterr().err


def test_a_missing_key_is_refused_before_an_arm_starts(gated, capsys, monkeypatch):
    gated(commands("trial", 1))
    monkeypatch.setattr("walkie_agent.evals.mahoraga.gym.inspect_eval", lambda *a, **k: pytest.fail("It ran."))

    assert cli.main(["run", "--arms", "A", "--commands", "1"]) == 2

    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_run_passes_its_arguments_on(monkeypatch):
    seen = {}

    def run_episodes(arms, commands, **kwargs):
        seen.update(arms=arms, commands=commands, **kwargs)
        return []

    monkeypatch.setattr(cli, "run_episodes", run_episodes)

    assert cli.main(["run", "--arms", "A, B", "--commands", "5", "--epochs", "3", "--no-bwrap"]) == 0

    assert seen["arms"] == ["A", "B"]
    assert seen["commands"] == 5
    assert seen["epochs"] == 3
    assert seen["sandbox"] is False
    assert seen["model"] == "deepseek/deepseek-v4.1-flash:nitro"
    assert seen["base_url"] == "https://openrouter.ai/api/v1"


def test_gate_passes_and_writes_under_the_home(capsys, tmp_path):
    target = tmp_path / "elsewhere"

    assert cli.main(["--home", str(target), "gate", "--split", "trial", "--commands", "2", "--epochs", "1"]) == 0

    assert "PASS trial mixed: 2 episodes, twice." in capsys.readouterr().out
    assert home() == target
    assert list((target / "gate").glob("gate-*.json"))


def test_gate_defaults_to_the_plans_three_data_sets(monkeypatch):
    seen = []

    def standard(**kwargs):
        seen.append(kwargs)
        return []

    monkeypatch.setattr(cli, "run_standard_gate", standard)

    assert cli.main(["gate"]) == 0
    assert seen == [{"category": "mixed", "max_samples": 8, "sandbox": True}]


def test_report_prints_json(capsys, monkeypatch):
    def report(log_dir, baseline, shelf_root):
        return {"log_dir": log_dir, "baseline": baseline, "shelf_root": shelf_root}

    monkeypatch.setattr(cli, "report", report)

    assert cli.main(["report", "--log-dir", "/tmp/x"]) == 0
    assert json.loads(capsys.readouterr().out) == {"log_dir": "/tmp/x", "baseline": "A", "shelf_root": None}
    assert cli.main(["report", "--log-dir", "/tmp/x", "--shelf", "/tmp/store"]) == 0
    assert json.loads(capsys.readouterr().out)["shelf_root"] == "/tmp/store"


def test_run_points_the_shelf_at_the_store_given(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_episodes", lambda arms, commands, **kwargs: seen.update(kwargs) or [])

    assert cli.main(["run", "--arms", "C", "--commands", "1", "--shelf", "/tmp/store"]) == 0
    assert seen["shelf_root"] == "/tmp/store"
    assert cli.main(["run", "--arms", "C", "--commands", "1"]) == 0
    assert seen["shelf_root"] is None


def test_a_report_on_a_shelf_in_an_unignored_work_tree_only_reads_it(capsys, tmp_path):
    # The report writes nothing to the store, so it needs no refusal there (T34).
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "logs").mkdir()

    assert cli.main(["report", "--log-dir", str(tmp_path / "logs"), "--shelf", str(tmp_path / "repo" / "store")]) == 0

    assert "Refused:" not in capsys.readouterr().err
    assert not (tmp_path / "repo" / "store").exists()


def test_a_store_refusal_in_a_report_is_printed_as_one(capsys, monkeypatch):
    def report(log_dir, baseline, shelf_root):
        raise StoreRefused("The logs ran on different shelves: /a, /b. Name one with shelf_root.")

    monkeypatch.setattr(cli, "report", report)

    assert cli.main(["report", "--log-dir", "/tmp/x"]) == 2
    assert "Refused: The logs ran on different shelves" in capsys.readouterr().err


def test_a_report_that_fails_for_another_reason_is_not_called_a_refusal(monkeypatch):
    def report(log_dir, baseline, shelf_root):
        raise ValueError("A log is broken.")

    monkeypatch.setattr(cli, "report", report)

    with pytest.raises(ValueError, match="A log is broken."):
        cli.main(["report", "--log-dir", "/tmp/x"])
