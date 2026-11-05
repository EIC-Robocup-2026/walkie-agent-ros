"""The shelf keeps what succeeded, outside git, and only the harness verifies."""

import ast
import json
from pathlib import Path

import pytest

import mahoraga
from mahoraga.driver import ScriptRunner, script_summary
from mahoraga.shelf import Shelf, definition_source, shape, source_sha, top_level_defs

FETCH = 'def to_shelf() -> str:\n    """Drive to the shelf."""\n    return go_to("shelf")\n\nto_shelf()'


@pytest.fixture
def shelf(tmp_path):
    return Shelf(tmp_path / "shelf")


@pytest.fixture
def shelf_runner(pool, shelf):
    return ScriptRunner(pool, shelf=shelf)


def events(shelf):
    return [json.loads(line) for line in (shelf.root / "ledger.jsonl").read_text().splitlines()]


def test_home_comes_from_the_argument_then_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MAHORAGA_HOME", str(tmp_path / "from-env"))

    assert mahoraga.home() == tmp_path / "from-env"
    assert mahoraga.home(tmp_path / "given") == tmp_path / "given"
    assert Shelf().root == tmp_path / "from-env" / "shelf"


def test_home_defaults_outside_the_repo(monkeypatch, tmp_path):
    monkeypatch.delenv("MAHORAGA_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert mahoraga.home() == tmp_path / ".local/share/mahoraga"


def test_a_directory_in_a_git_work_tree_is_refused(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)

    with pytest.raises(ValueError, match="git work tree"):
        Shelf(tmp_path / "repo" / "shelf")
    with pytest.raises(ValueError, match="git work tree"):
        mahoraga.home(Path(mahoraga.__file__).parent / "shelf")


def test_a_success_saves_one_file_per_definition_and_a_ledger_line(shelf_runner, shelf, fns):
    result = shelf_runner.run(FETCH, fns, save=["to_shelf"])

    assert result.saved == ["to_shelf"]
    assert (shelf.root / "to_shelf.py").read_text().startswith("def to_shelf() -> str:")
    assert "to_shelf()\n" not in (shelf.root / "to_shelf.py").read_text().split("return")[1]
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
    assert (shelf.root / "report.py").read_text() == REPORT
    assert events(shelf)[0]["sha"] == source_sha(REPORT)


@pytest.mark.parametrize("ending", ["\r", "\r\n"], ids=["lone-cr", "crlf"])
def test_a_definition_is_saved_with_newline_endings_so_its_ledger_sha_is_the_one_the_shelf_reports(shelf, ending):
    code = REPORT.replace("\n", ending) + 'report(["kitchen"])'

    assert definition_source(code, top_level_defs(code)["report"]) == REPORT
    assert shelf.save(code, ["report"])[0] == ["report"]
    assert (shelf.root / "report.py").read_bytes() == REPORT.encode()
    assert events(shelf)[0]["sha"] == shelf.versions()["report"] == source_sha(REPORT)


def test_a_form_feed_before_a_top_level_definition_takes_nothing_off_its_lines(shelf):
    """The def's column offset is 1 and its text starts at the def, so the text and sha stay as written."""
    note = 'def note():\n    return """first\n    second"""\n'
    code = "\f" + note + "note()"

    assert definition_source(code, top_level_defs(code)["note"]) == note
    assert shelf.save(code, ["note"])[0] == ["note"]
    assert (shelf.root / "note.py").read_text() == note
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
