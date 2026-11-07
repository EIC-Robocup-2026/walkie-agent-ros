"""The shelf keeps what succeeded in git-ignored tier folders, and only the harness verifies."""

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import walkie_agent.scripting.shelf as shelf_module
import walkie_agent.tools
from walkie_agent.scripting.driver import ScriptRunner, script_summary
from walkie_agent.scripting.shelf import (
    PACKAGE_STORE,
    Shelf,
    StoreRefused,
    definition_source,
    require_ignored,
    shape,
    source_sha,
    store_root,
    top_level_defs,
)

FETCH = 'def to_shelf() -> str:\n    """Drive to the shelf."""\n    return go_to("shelf")\n\nto_shelf()'
VALUE_1 = "def value() -> int:\n    return 1\n"
VALUE_2 = "def value() -> int:\n    return 2\n"
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture
def shelf(tmp_path):
    return Shelf(tmp_path / "shelf")


@pytest.fixture
def shelf_runner(pool, shelf):
    return ScriptRunner(pool, shelf=shelf)


@pytest.fixture
def repo(tmp_path):
    """An empty git repository."""
    root = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def events(shelf):
    return [json.loads(line) for line in (shelf.root / "ledger.jsonl").read_text().splitlines()]


def test_the_default_store_is_the_tools_package_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("WALKIE_TOOLS_HOME")

    assert store_root() == PACKAGE_STORE == Path(walkie_agent.tools.__file__).resolve().parent
    monkeypatch.setenv("WALKIE_TOOLS_HOME", str(tmp_path / "from-env"))
    assert store_root() == tmp_path / "from-env"
    assert store_root(tmp_path / "given") == tmp_path / "given"
    assert Shelf(tmp_path / "given").temporary == tmp_path / "given" / "temporary"


@needs_git
def test_git_ignores_what_the_shelf_writes_in_the_package_folder():
    if not any((parent / ".git").exists() for parent in (PACKAGE_STORE, *PACKAGE_STORE.parents)):
        pytest.skip("The package is not in a git work tree.")
    probes = [
        "temporary/probe_tool.py",
        "temporary/tmpk3j9x2.tmp",
        "bundled/probe_tool.py",
        "bundled/tmpk3j9x2.tmp",
        "ledger.jsonl",
    ]
    done = subprocess.run(
        ["git", "-C", str(PACKAGE_STORE), "check-ignore", "-z", "--stdin"],
        input="\0".join(probes) + "\0",
        capture_output=True,
        text=True,
        check=False,
    )
    readmes = subprocess.run(
        ["git", "-C", str(PACKAGE_STORE), "check-ignore", "temporary/README.md", "bundled/README.md"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert sorted(filter(None, done.stdout.split("\0"))) == sorted(probes), done.stderr
    assert readmes.returncode == 1, readmes.stdout
    require_ignored(PACKAGE_STORE)


@needs_git
def test_a_store_in_a_git_work_tree_that_does_not_ignore_it_is_refused_before_it_is_created(repo):
    with pytest.raises(StoreRefused, match="does not ignore") as refused:
        Shelf(repo / "store")

    assert "WALKIE_TOOLS_HOME" in str(refused.value)
    assert not (repo / "store").exists()
    with pytest.raises(ValueError, match="does not ignore"):
        Shelf(Path(__file__).parent / "shelf")


@needs_git
def test_a_store_that_git_ignores_only_in_part_is_refused_and_told_what_is_missing(repo):
    (repo / ".gitignore").write_text("/store/temporary/\n")

    with pytest.raises(ValueError, match="does not ignore") as refused:
        Shelf(repo / "store")

    assert "store/bundled/ignore_probe.py" in str(refused.value)
    assert "store/ledger.jsonl" in str(refused.value)
    assert "store/temporary/" not in str(refused.value)


@needs_git
def test_a_store_that_git_ignores_is_accepted(repo):
    (repo / ".gitignore").write_text("/store/\n")

    shelf = Shelf(repo / "store")

    assert shelf.save(VALUE_1, ["value"]) == (["value"], {})
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True)
    assert status.stdout == "?? .gitignore\n"


@needs_git
def test_a_folder_git_cannot_read_is_refused(tmp_path):
    (tmp_path / "broken" / ".git").mkdir(parents=True)

    with pytest.raises(ValueError, match="git check-ignore failed"):
        Shelf(tmp_path / "broken" / "store")


def test_without_git_only_the_package_folder_passes_inside_a_work_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(shelf_module.shutil, "which", lambda name: None)
    (tmp_path / "repo" / ".git").mkdir(parents=True)

    with pytest.raises(ValueError, match="git is not installed"):
        Shelf(tmp_path / "repo" / "store")
    require_ignored(PACKAGE_STORE)
    assert Shelf(tmp_path / "plain").root == tmp_path / "plain"


def test_saving_writes_the_latest_version_to_temporary_and_nothing_to_bundled(shelf):
    shelf.save(VALUE_1, ["value"])
    shelf.save(VALUE_2, ["value"])

    assert (shelf.temporary / "value.py").read_text() == VALUE_2
    assert list(shelf.bundled.iterdir()) == []


def test_verifying_writes_that_exact_version_to_bundled(shelf):
    shelf.save(REPORT, ["report"])

    shelf.mark_verified(shelf.versions())

    verified = events(shelf)[-1]
    assert (verified["event"], verified["name"]) == ("verified", "report")
    assert (shelf.bundled / "report.py").read_bytes() == REPORT.encode()
    assert source_sha((shelf.bundled / "report.py").read_text()) == verified["sha"]


def test_a_version_verified_after_a_rewrite_goes_to_bundled_as_it_ran(shelf):
    shelf.save(VALUE_1, ["value"])
    first = shelf.versions()
    shelf.save(VALUE_2, ["value"])

    shelf.mark_verified(first)

    assert (shelf.bundled / "value.py").read_text() == VALUE_1
    assert (shelf.temporary / "value.py").read_text() == VALUE_2
    assert Shelf(shelf.root, frozen=True).names() == []
    shelf.mark_verified(shelf.versions())
    assert (shelf.bundled / "value.py").read_text() == VALUE_2
    assert Shelf(shelf.root, frozen=True).examples() == VALUE_2


def test_an_older_version_scored_after_the_latest_was_verified_leaves_the_latest_in_bundled(shelf):
    shelf.save(VALUE_1, ["value"])
    first = shelf.versions()
    shelf.save(VALUE_2, ["value"])
    shelf.mark_verified(shelf.versions())

    shelf.mark_verified(first)

    verified = [event["sha"] for event in events(shelf) if event["event"] == "verified"]
    assert verified == [source_sha(VALUE_2), source_sha(VALUE_1)]
    assert (shelf.bundled / "value.py").read_text() == VALUE_2
    assert Shelf(shelf.root, frozen=True).examples() == VALUE_2


def test_saving_a_verified_version_again_puts_it_back_in_bundled(shelf):
    shelf.save(VALUE_1, ["value"])
    shelf.mark_verified(shelf.versions())
    shelf.save(VALUE_2, ["value"])
    shelf.mark_verified(shelf.versions())

    shelf.save(VALUE_1, ["value"])

    assert events(shelf)[-1]["event"] == "saved"
    assert (shelf.bundled / "value.py").read_text() == VALUE_1
    assert Shelf(shelf.root, frozen=True).examples() == VALUE_1


@pytest.mark.parametrize(
    "steps",
    [
        [("save", VALUE_1), ("verify", VALUE_1)],
        [("save", VALUE_1), ("verify", VALUE_1), ("save", VALUE_2)],
        [("save", VALUE_1), ("save", VALUE_2), ("verify", VALUE_1), ("verify", VALUE_2)],
        [("save", VALUE_1), ("save", VALUE_2), ("verify", VALUE_2), ("verify", VALUE_1)],
        [("save", VALUE_1), ("verify", VALUE_1), ("save", VALUE_2), ("verify", VALUE_2), ("save", VALUE_1)],
        [("save", VALUE_1), ("verify", VALUE_1), ("save", VALUE_2), ("save", VALUE_1), ("verify", VALUE_2)],
    ],
)
def test_a_frozen_shelf_offers_the_latest_versions_counts_report_verified_in_any_scoring_order(shelf, steps):
    for step, source in steps:
        if step == "save":
            shelf.save(source, ["value"])
        else:
            shelf.mark_verified({"value": source_sha(source)})

    verified = [name for name, entry in shelf.counts().items() if entry["verified"]]
    assert Shelf(shelf.root, frozen=True).names() == verified
    for name in verified:
        assert (shelf.bundled / f"{name}.py").read_text() == (shelf.temporary / f"{name}.py").read_text()


def test_a_frozen_shelf_loads_only_verified_files_in_bundled(shelf):
    shelf.save(VALUE_1, ["value"])
    shelf.mark_verified(shelf.versions())
    shelf.save("def other() -> int:\n    return 3\n", ["other"])
    (shelf.bundled / "other.py").write_text((shelf.temporary / "other.py").read_text())

    assert Shelf(shelf.root, frozen=True).names() == ["value"]
    (shelf.bundled / "value.py").unlink()
    assert Shelf(shelf.root, frozen=True).names() == []


def test_verifying_a_version_the_shelf_never_held_is_refused_and_changes_nothing(shelf):
    shelf.save(VALUE_1, ["value"])

    with pytest.raises(ValueError, match="never held"):
        shelf.mark_verified({"value": source_sha(VALUE_2)})
    with pytest.raises(ValueError, match="never held"):
        shelf.mark_verified({"value": source_sha(VALUE_1), "missing": source_sha(VALUE_1)})

    assert [event["event"] for event in events(shelf)] == ["saved"]
    assert list(shelf.bundled.iterdir()) == []


def test_a_success_saves_one_file_per_definition_and_a_ledger_line(shelf_runner, shelf, fns):
    result = shelf_runner.run(FETCH, fns, save=["to_shelf"])

    assert result.saved == ["to_shelf"]
    assert (shelf.temporary / "to_shelf.py").read_text().startswith("def to_shelf() -> str:")
    assert "to_shelf()\n" not in (shelf.temporary / "to_shelf.py").read_text().split("return")[1]
    event = events(shelf)[0]
    assert (event["event"], event["name"]) == ("saved", "to_shelf")
    assert event["sha"] and event["ts"]


# Each later line has leading text the shelf must keep: a body, a continuation at column 0,
# a blank line of spaces, and the lines of a string.
REPORT = (
    "def report(rooms: list[str]) -> str:\n"
    '    """Say which rooms were checked."""\n'
    "    names = [\n"
    "room.upper()\n"
    "        for room in rooms]\n"
    "    \n"
    '    return """Checked:\n'
    '  """ + ", ".join(names)\n'
)


def test_a_top_level_definition_is_saved_with_its_text_exactly_as_written(shelf):
    """A definition's text is its source as written, so each sha on a shelf stays valid."""
    code = REPORT + '\nreport(["kitchen"])'

    assert definition_source(code, top_level_defs(code)["report"]) == REPORT
    assert shelf.save(code, ["report"])[0] == ["report"]
    assert (shelf.temporary / "report.py").read_text() == REPORT
    assert events(shelf)[0]["sha"] == source_sha(REPORT)


@pytest.mark.parametrize("ending", ["\r", "\r\n"], ids=["lone-cr", "crlf"])
def test_a_definition_is_saved_with_newline_endings_so_its_ledger_sha_is_the_one_the_shelf_reports(shelf, ending):
    code = REPORT.replace("\n", ending) + 'report(["kitchen"])'

    assert definition_source(code, top_level_defs(code)["report"]) == REPORT
    assert shelf.save(code, ["report"])[0] == ["report"]
    assert (shelf.temporary / "report.py").read_bytes() == REPORT.encode()
    assert events(shelf)[0]["sha"] == shelf.versions()["report"] == source_sha(REPORT)


def test_a_form_feed_before_a_top_level_definition_takes_nothing_off_its_lines(shelf):
    """The def's column offset is 1 and its text starts at the def, so the text and sha stay as written."""
    note = 'def note():\n    return """first\n    second"""\n'
    code = "\f" + note + "note()"

    assert definition_source(code, top_level_defs(code)["note"]) == note
    assert shelf.save(code, ["note"])[0] == ["note"]
    assert (shelf.temporary / "note.py").read_text() == note
    assert events(shelf)[0]["sha"] == source_sha(note)


def test_a_shape_leaves_out_layout_comments_and_decorators_but_not_what_the_code_does():
    plain = top_level_defs("def f(x):\n    return x + 1\n")["f"]
    laid_out = top_level_defs("def f(x):  # Add one.\n\n    return (x\n            + 1)\n")["f"]
    decorated = top_level_defs("@wrap\ndef f(x):\n    return x + 1\n")["f"]
    other = top_level_defs("def f(x):\n    return x + 2\n")["f"]

    assert shape(plain) == shape(laid_out) == shape(decorated)
    assert shape(other) != shape(plain)
    # The caller's tree keeps its decorators.
    assert [ast.unparse(decorator) for decorator in decorated.decorator_list] == ["wrap"]


def test_a_failed_run_saves_nothing(shelf_runner, shelf, fns):
    result = shelf_runner.run(FETCH + '\ngo_to("garage")', fns, save=["to_shelf"])

    assert result.saved == []
    assert "to_shelf" in result.refused
    assert shelf.names() == []


@pytest.mark.parametrize(
    ("code", "name", "reason"),
    [
        (FETCH, "missing", "No top-level function"),
        (FETCH, "../../evil", "No top-level function"),
        ("def go_to(place: str) -> str:\n    return place\n", "go_to", "built-in"),
        ("def outer() -> int:\n    def inner() -> int:\n        return 1\n    return inner()\n", "inner", "No top-level"),
        ("def helper() -> int:\n    return 1\n\ndef uses() -> int:\n    return helper()\n", "uses", "helper"),
        ("LIMIT = 3\n\ndef over() -> int:\n    return LIMIT\n", "over", "LIMIT"),
    ],
)
def test_save_refuses_what_could_not_stand_on_the_shelf(shelf_runner, shelf, fns, code, name, reason):
    result = shelf_runner.run(code, fns, save=[name])

    assert result.ok, result.error
    assert result.saved == []
    assert reason in result.refused[name]
    assert shelf.names() == []


@pytest.mark.parametrize(
    "parameters",
    [
        "x=helper",
        "x=eval",
        "x=len('a')",
        "x=set()",
        "x=(1, set())",
        "x=lambda: 1",
        "x=str.upper",
        "x=[eval]",
        "x=(1, eval)",
        "x={'key': eval}",
        "x={[]: 1}",
        "x=1 + 1",
        "x, y=eval",
        "x=1, y=eval",
        "x=eval, /",
        "x=1, /, y=eval",
        "*, x=eval",
        "*, x, y=eval",
        "*, x=eval, y",
        "x=1, *, y=2, z=eval",
    ],
)
def test_save_refuses_a_parameter_default_that_is_not_a_literal(shelf, parameters):
    saved, refused = shelf.save(f"def kept({parameters}):\n    return 1\n", ["kept"])

    assert saved == []
    assert "literal" in refused["kept"]
    assert shelf.names() == []


@pytest.mark.parametrize(
    "parameters",
    [
        "x=3",
        "x=-1",
        "x=1.5",
        "x='x'",
        "x=None",
        "x=True",
        "x=(1, 2)",
        "x=[]",
        "x={}",
        "x=[1, ('a', None)]",
        "x, y=1",
        "x=1, /, y=2",
        "*, x",
        "*, x, y=1",
        "*, x=1, y",
        "x=1, *, y, z=2",
    ],
)
def test_save_keeps_a_function_whose_defaults_are_all_literal(shelf, parameters):
    assert shelf.save(f"def kept({parameters}):\n    return 1\n", ["kept"]) == (["kept"], {})


def test_a_refused_default_runs_as_written_and_tells_the_model_why(shelf_runner, shelf, fns):
    code = "def helper() -> int:\n    return 1\n\ndef uses(fn=helper) -> int:\n    return fn()\n\nuses()"

    result = shelf_runner.run(code, fns, save=["helper", "uses"])

    assert (result.ok, result.output) == (True, 1), result.error
    assert result.saved == ["helper"]
    assert script_summary(result)["not_saved"] == {"uses": "A saved function's parameter defaults must be literal values."}
    assert shelf.names() == ["helper"]


def test_a_literal_default_still_applies_when_the_function_runs_from_the_shelf(shelf_runner, fns):
    shelf_runner.run("def offset(n: int = -1, *, by: int = 2) -> int:\n    return n + by\n", fns, save=["offset"])

    result = shelf_runner.run("offset() + offset(3, by=4)", fns)

    assert (result.output, result.shelf_used) == (8, ["offset"]), result.error


@pytest.mark.parametrize("name", ["locals", "eval", "exec"])
def test_save_refuses_a_name_the_loader_defines(shelf, name):
    saved, refused = shelf.save(f"def {name}() -> int:\n    return 73\n", [name])

    assert saved == []
    assert "built-in" in refused[name]
    assert shelf.names() == []


def test_definitions_saved_together_may_call_each_other(shelf_runner, shelf, fns):
    code = "def helper() -> int:\n    return 1\n\ndef uses() -> int:\n    return helper()\n"

    assert shelf_runner.run(code, fns, save=["helper", "uses"]).saved == ["helper", "uses"]


def test_a_saved_function_runs_later_through_the_same_driver(shelf_runner, shelf, fns, robot):
    shelf_runner.run(FETCH, fns, save=["to_shelf"])
    robot.calls.clear()

    result = shelf_runner.run('to_shelf()\ngo_to("bar")\nto_shelf()', fns)

    assert result.ok, result.error
    assert result.shelf_used == ["to_shelf"]
    assert [call["tool"] for call in result.calls] == ["go_to", "go_to", "go_to"]
    assert [call["verb"] for call in robot.calls] == ["base.go_to"] * 3


def test_calls_inside_a_shelf_function_count_against_the_scripts_budget(shelf_runner, fns, robot):
    shelf_runner.run('def hello() -> str:\n    return say("hi")\n\nhello()', fns, save=["hello"])
    robot.said.clear()

    result = shelf_runner.run("for n in range(100):\n    hello()", fns)

    assert result.phase == "budget"
    assert len(robot.said) == 40


def test_a_shelf_function_that_calls_another_pulls_it_in(shelf_runner, shelf, fns):
    code = "def helper() -> int:\n    return 1\n\ndef uses() -> int:\n    return helper()\n"
    shelf_runner.run(code, fns, save=["helper", "uses"])

    result = shelf_runner.run("uses() + 1", fns)

    assert (result.output, result.shelf_used) == (2, ["helper", "uses"])


def test_a_scripts_own_definition_wins_over_the_shelf(shelf_runner, fns):
    shelf_runner.run("def value() -> int:\n    return 1\n\nvalue()", fns, save=["value"])

    result = shelf_runner.run("def value() -> int:\n    return 2\n\nvalue()", fns)

    assert (result.output, result.shelf_used) == (2, [])


def test_a_shelf_function_that_raises_is_told_apart_from_a_script_that_raises(shelf_runner, fns):
    shelf_runner.run('def bad() -> str:\n    return go_to("garage")\n\ntry:\n    bad()\nexcept RuntimeError:\n    pass', fns, save=["bad"])
    shelf_runner.run("def good() -> int:\n    return 1\n\ngood()", fns, save=["good"])

    raised = shelf_runner.run("bad()", fns)
    own = shelf_runner.run('good()\ngo_to("garage")', fns)

    assert (raised.shelf_used, raised.shelf_raised) == (["bad"], ["bad"])
    assert (own.shelf_used, own.shelf_raised) == (["good"], [])


def test_counts_follow_the_ledger(shelf_runner, shelf, fns):
    shelf_runner.run("def value() -> int:\n    return 1\n\nvalue()", fns, save=["value"])
    shelf_runner.run("value()", fns)
    shelf_runner.run('value()\ngo_to("garage")', fns)
    shelf.mark_verified(shelf.versions())

    assert shelf.counts() == {"value": {"saved": 1, "run": 1, "fail": 1, "verified": 1}}


def test_a_rewrite_starts_its_counts_again_but_keeps_the_churn(shelf_runner, shelf, fns):
    shelf_runner.run("def value() -> int:\n    return 1\n\nvalue()", fns, save=["value"])
    shelf.mark_verified(shelf.versions())
    shelf_runner.run("def value() -> int:\n    return 2\n\nvalue()", fns, save=["value"])

    assert shelf.counts() == {"value": {"saved": 2, "run": 0, "fail": 0, "verified": 0}}


def test_a_frozen_shelf_offers_only_what_was_verified_when_it_froze(tmp_path, shelf_runner, shelf, fns):
    code = "def kept() -> int:\n    return 1\n\ndef unproven() -> int:\n    return 2\n"
    shelf_runner.run(code, fns, save=["kept", "unproven"])
    versions = shelf.versions()
    shelf.mark_verified({"kept": versions["kept"]})

    frozen = Shelf(shelf.root, frozen=True)
    shelf.mark_verified({"unproven": versions["unproven"]})

    assert frozen.names() == ["kept"]
    assert "unproven" not in frozen.examples()
    assert ScriptRunner(shelf_runner.pool, shelf=frozen).run("kept()", fns).output == 1
    assert not ScriptRunner(shelf_runner.pool, shelf=frozen).run("unproven()", fns).ok


def test_a_frozen_shelf_takes_no_new_definition(shelf_runner, shelf, fns):
    frozen_runner = ScriptRunner(shelf_runner.pool, shelf=Shelf(shelf.root, frozen=True))

    result = frozen_runner.run("def new() -> int:\n    return 1\n\nnew()", fns, save=["new"])

    assert result.saved == []
    assert "frozen" in result.refused["new"]
    assert shelf.names() == []


def test_describe_shows_the_interface_and_examples_the_source(shelf_runner, shelf, fns):
    shelf_runner.run(FETCH, fns, save=["to_shelf"])

    assert "Drive to the shelf." in shelf.describe()
    assert "go_to" not in shelf.describe()
    assert 'return go_to("shelf")' in shelf.examples()


def test_a_run_without_a_shelf_says_it_cannot_save(runner, fns):
    result = runner.run("def value() -> int:\n    return 1\n", fns, save=["value"])

    assert result.saved == []
    assert "no shelf" in result.refused["value"]
