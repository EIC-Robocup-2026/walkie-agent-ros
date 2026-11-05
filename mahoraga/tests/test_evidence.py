"""A function counts as used, returned or raised from what it did at run time, not from its name."""

import ast
import functools
import re
from types import SimpleNamespace

import pytest

import mahoraga.driver as driver
from mahoraga.driver import Budgets, ScriptRunner
from mahoraga.evidence import Watch
from mahoraga.shelf import Shelf, definition_source, source_sha, top_level_defs
from mahoraga.tasks import _proven, script_record

VALUE ="def value() -> int:\n    return 1\n"
BAD = 'def bad() -> str:\n    return go_to("garage")\n'
OUTER = "def outer() -> str:\n    return bad()\n"
CATCHES = 'def catches() -> str:\n    try:\n        return bad()\n    except RuntimeError:\n        return "caught"\n'
HIDDEN = re.compile(r"_h[0-9a-f]{16}")


@pytest.fixture
def shelf(tmp_path):
    shelf = Shelf(tmp_path / "shelf")
    code = VALUE + BAD + OUTER + CATCHES
    assert shelf.save(code, ["value", "bad", "outer", "catches"])[0] == ["bad", "catches", "outer", "value"]
    return shelf


@pytest.fixture
def run(pool, shelf, fns):
    runner = ScriptRunner(pool, shelf=shelf)
    return lambda code, **kwargs: runner.run(code, fns, **kwargs)


def _events(shelf, name):
    return {key: shelf.counts()[name][key] for key in ("run", "fail", "verified")}


def test_a_raise_in_a_failed_script_counts_against_the_version_it_ran(pool, tmp_path, fns):
    """The script saved nothing, but the same text is on the shelf, so its verification must not stand."""
    divide = "def divide(divisor=1):\n    return 1 / divisor\n"
    runner = ScriptRunner(pool, shelf=Shelf(tmp_path / "shelf"))

    first = runner.run(divide + "divide()", fns, save=["divide"])
    second = runner.run(divide + "divide(0)", fns, save=["divide"])

    assert first.saved == ["divide"]
    assert second.ok is False
    assert second.saved == []
    assert second.returned == {}
    assert second.raised == first.returned == {"divide": source_sha(divide)}
    assert _proven([{"returned": r.returned, "raised": r.raised} for r in (first, second)]) == {}


DIVIDE = "def divide(divisor=1): return 1 / divisor\n"
# Indented, a definition over two lines keeps the scope's indentation on its second line.
DIVIDE_ON_TWO_LINES = "def divide(divisor=1):\n    return 1 / divisor\n"


def _verify_after(pool, tmp_path, fns, second_code, saved=DIVIDE):
    """Save and return `saved` in one script, run `second_code` unsaved, score both, freeze."""
    shelf = Shelf(tmp_path / "shelf")
    runner = ScriptRunner(pool, shelf=shelf)
    first = runner.run(saved + "divide()", fns, save=["divide"])
    second = runner.run(second_code, fns)
    assert first.returned == {"divide": source_sha(saved)}
    assert second.saved == []
    proven = _proven([script_record("", first), script_record(second_code, second)])
    shelf.mark_verified(proven)
    return proven, Shelf(shelf.root, frozen=True).names()


@pytest.mark.parametrize(
    ("saved", "second_code"),
    [
        # Watched: the wrapper sees the raise.
        (DIVIDE, DIVIDE + "divide(0)"),
        # The others go unwatched, so their versions count as raised.
        (DIVIDE, DIVIDE + "divide = divide\ndivide(0)"),
        (
            DIVIDE,
            DIVIDE + "try:\n    divide(0)\nexcept ZeroDivisionError:\n    pass\n"
            "def divide(divisor=1): return 2 / divisor\ndivide()",
        ),
        (DIVIDE, "if True:\n    def divide(divisor=1): return 1 / divisor\ndivide(0)"),
        (DIVIDE, "def outer():\n    def divide(divisor=1): return 1 / divisor\n    return divide(0)\nouter()"),
        (DIVIDE_ON_TWO_LINES, "if True:\n    def divide(divisor=1):\n        return 1 / divisor\ndivide(0)"),
        (
            DIVIDE_ON_TWO_LINES,
            "def outer():\n    def divide(divisor=1):\n        return 1 / divisor\n    return divide(0)\nouter()",
        ),
    ],
    ids=[
        "watched",
        "bound-twice",
        "redefined-later",
        "conditional",
        "nested",
        "conditional-on-two-lines",
        "nested-on-two-lines",
    ],
)
def test_a_raise_through_an_unsaved_copy_keeps_the_version_off_the_frozen_shelf(
    pool, tmp_path, fns, saved, second_code
):
    """The second script asks to save nothing, so only its raise can stop the verification."""
    proven, frozen = _verify_after(pool, tmp_path, fns, second_code, saved)

    assert "divide" not in proven
    assert frozen == []


def _definition(code, name):
    """The one definition of `name` in `code`, at any depth."""
    (node,) = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.FunctionDef) and node.name == name]
    return node


NOTE = 'def note() -> str:\n    return """first\nsecond"""\n'


@pytest.mark.parametrize(
    ("top_level", "code", "name"),
    [
        (DIVIDE_ON_TWO_LINES, "if True:\n    def divide(divisor=1):\n        return 1 / divisor\n", "divide"),
        (
            DIVIDE_ON_TWO_LINES,
            "def outer():\n    def divide(divisor=1):\n        return 1 / divisor\n    return divide(0)\n",
            "divide",
        ),
        (
            DIVIDE_ON_TWO_LINES,
            "def outer():\n    if True:\n        def divide(divisor=1):\n            return 1 / divisor\n    return divide(0)\n",
            "divide",
        ),
        (DIVIDE_ON_TWO_LINES, "if True:\n\tdef divide(divisor=1):\n\t    return 1 / divisor\n", "divide"),
        # A line with less than the scope's indentation, here inside a string, stays as it is.
        (NOTE, 'if True:\n    def note() -> str:\n        return """first\nsecond"""\n', "note"),
        (NOTE, 'if True:\n    def note() -> str:\n        return """first\n    second"""\n', "note"),
    ],
    ids=[
        "conditional",
        "nested",
        "two-scopes-deep",
        "tab-indented",
        "string-line-at-column-0",
        "string-line-indented-with-the-rest",
    ],
)
def test_an_indented_copy_has_the_text_and_sha_of_the_same_definition_at_top_level(top_level, code, name):
    assert definition_source(code, _definition(code, name)) == top_level
    assert (name, source_sha(top_level)) in Watch(code, {}, watch_own=True).unwatched


def test_an_unsaved_copy_that_returns_does_not_block_the_verification(pool, tmp_path, fns):
    proven, frozen = _verify_after(pool, tmp_path, fns, DIVIDE + "divide(2)")

    assert proven == {"divide": source_sha(DIVIDE)}
    assert frozen == ["divide"]


def test_each_unwatched_version_of_a_name_is_kept(run):
    code = "def mine() -> int:\n    return 1\n\nfirst = mine()\n\ndef mine() -> int:\n    return 2\n\nfirst + mine()"

    result = run(code)

    assert result.output == 3
    assert result.raised == {}
    assert [name for name, _ in result.unwatched] == ["mine", "mine"]
    assert len({sha for _, sha in result.unwatched}) == 2


def test_a_function_that_ran_is_used_and_returned(run, shelf):
    result = run("value()")

    assert result.shelf_used == ["value"]
    assert result.shelf_raised == []
    assert result.returned == {"value": shelf.versions()["value"]}
    assert result.raised == {}
    assert _events(shelf, "value") == {"run": 1, "fail": 0, "verified": 0}


def test_a_name_in_dead_code_is_not_a_use(run, shelf):
    result = run("if False:\n    value()\n1")

    assert result.ok
    assert result.shelf_used == []
    assert result.returned == {}
    assert _events(shelf, "value") == {"run": 0, "fail": 0, "verified": 0}


def test_a_name_only_read_is_not_a_use(run):
    assert run("kept = value\n1").shelf_used == []


def test_a_script_that_never_starts_uses_nothing(run, shelf):
    result = run('value()\nvalue("too many")')

    assert result.phase == "type_check"
    assert result.shelf_used == []
    assert _events(shelf, "value") == {"run": 0, "fail": 0, "verified": 0}


def test_a_raise_its_caller_caught_still_counts_as_raised(run, shelf):
    result = run("try:\n    bad()\nexcept RuntimeError:\n    pass\n1")

    assert result.ok
    assert (result.shelf_used, result.shelf_raised) == (["bad"], ["bad"])
    assert result.returned == {}
    assert result.raised == {"bad": shelf.versions()["bad"]}


def test_a_raise_through_two_functions_counts_for_both(run):
    result = run("try:\n    outer()\nexcept RuntimeError:\n    pass\n1")

    assert result.shelf_used == ["bad", "outer"]
    assert result.shelf_raised == ["bad", "outer"]


def test_a_function_that_handled_a_raise_returned_and_the_inner_one_raised(run):
    result = run("catches()")

    assert result.output == "caught"
    assert result.shelf_used == ["bad", "catches"]
    assert result.shelf_raised == ["bad"]
    assert list(result.returned) == ["catches"]


def test_a_raise_on_one_call_counts_even_if_another_call_returned(pool, tmp_path, fns):
    shelf = Shelf(tmp_path / "other")
    shelf.save('def maybe(place: str) -> str:\n    return go_to(place)\n', ["maybe"])

    result = ScriptRunner(pool, shelf=shelf).run(
        'maybe("shelf")\ntry:\n    maybe("garage")\nexcept RuntimeError:\n    pass', fns
    )

    assert result.shelf_raised == ["maybe"]
    assert result.raised == result.returned == shelf.versions()


def test_a_script_that_fails_after_a_function_returned_keeps_its_evidence(run, shelf):
    result = run("value()\n1 / 0")

    assert not result.ok
    assert (result.shelf_used, result.shelf_raised) == (["value"], [])
    assert _events(shelf, "value") == {"run": 0, "fail": 1, "verified": 0}


def test_the_ledger_keys_a_run_to_the_source_that_ran(run, shelf):
    ran = run("value()").returned
    shelf.save("def value() -> int:\n    return 2\n", ["value"])

    assert ran != shelf.versions()
    shelf.mark_verified(ran)
    assert _events(shelf, "value") == {"run": 0, "fail": 0, "verified": 0}
    assert "value" not in Shelf(shelf.root, frozen=True).names()


def test_counting_spends_no_call_and_leaves_the_ledger_alone(run):
    result = run('catches()\nvalue()\nsay("hi")')

    assert [call["tool"] for call in result.calls] == ["go_to", "say"]


def test_a_shelf_function_still_works_as_a_key_function(pool, tmp_path, fns):
    shelf = Shelf(tmp_path / "other")
    shelf.save("def backwards(n: int) -> int:\n    return -n\n", ["backwards"])

    result = ScriptRunner(pool, shelf=shelf).run("sorted([1, 3, 2], key=backwards)", fns)

    assert result.output == [3, 2, 1]
    assert result.shelf_used == ["backwards"]


def test_a_shelf_function_keeps_its_signature_in_the_type_check(run):
    result = run("value(1)")

    assert result.phase == "type_check"
    assert not HIDDEN.search(result.error)


def test_wrong_arguments_at_run_time_read_as_the_functions_own(run):
    result = run("value(1)  # type: ignore")

    assert result.phase == "runtime"
    assert "value()" in result.error
    assert not HIDDEN.search(result.error)


def test_a_runtime_error_shows_only_the_scripts_own_frames(run):
    result = run("bad()")

    assert result.phase == "runtime"
    assert 'return go_to("garage")' in result.error
    assert "value = " not in result.error
    assert "kwargs" not in result.error
    assert not HIDDEN.search(result.error)


def test_a_call_budget_inside_a_watched_function_is_spent_as_before(pool, tmp_path, fns):
    shelf = Shelf(tmp_path / "other")
    shelf.save('def loud() -> None:\n    while True:\n        say("x")\n', ["loud"])

    result = ScriptRunner(pool, shelf=shelf).run("loud()", fns)

    assert result.phase == "budget"
    assert (result.shelf_used, result.shelf_raised) == (["loud"], ["loud"])
    assert len(result.calls) == 40


@pytest.fixture
def clocked(monkeypatch, fns):
    """fns with a go_to that takes 31 s on a clock only it moves."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: clock.now))
    go_to_on_robot = fns["go_to"]

    @functools.wraps(go_to_on_robot)
    def slow(*args, **kwargs):
        clock.now += 31.0
        return go_to_on_robot(*args, **kwargs)

    return {**fns, "go_to": slow}


@pytest.fixture
def to_shelf(pool, tmp_path):
    shelf = Shelf(tmp_path / "other")
    shelf.save('def to_shelf() -> str:\n    return go_to("shelf")\n', ["to_shelf"])
    return ScriptRunner(pool, shelf=shelf)


def test_a_wall_budget_abort_keeps_the_evidence_of_what_ran(to_shelf, clocked, robot):
    result = to_shelf.run("to_shelf()\nto_shelf()", clocked)

    assert (result.ok, result.phase) == (False, "budget")
    assert "wall clock" in result.error
    assert (result.shelf_used, result.shelf_raised) == (["to_shelf"], ["to_shelf"])
    assert len(result.calls) == 1
    assert len([call for call in robot.calls if call["verb"] == "base.go_to"]) == 1


def test_a_script_that_catches_the_refusals_cannot_outlast_the_budget(to_shelf, clocked, robot):
    code = "to_shelf()\nwhile True:\n    try:\n        to_shelf()\n    except TimeoutError:\n        pass"

    result = to_shelf.run(code, clocked)

    assert (result.ok, result.phase) == (False, "budget")
    assert "wall clock" in result.error
    assert result.shelf_used == ["to_shelf"]
    assert len([call for call in robot.calls if call["verb"] == "base.go_to"]) == 1


def test_a_script_that_ends_after_the_refusal_still_fails_the_budget(to_shelf, clocked):
    code = "to_shelf()\ntry:\n    to_shelf()\nexcept TimeoutError:\n    pass\n1"

    assert to_shelf.run(code, clocked).phase == "budget"


def test_a_script_that_spins_past_its_feed_is_a_budget_and_keeps_its_evidence(pool, shelf, fns):
    runner = ScriptRunner(pool, shelf=shelf, budgets=Budgets(feed_s=0.2))

    result = runner.run("value()\nwhile True:\n    pass", fns)

    assert result.phase == "budget", result.error
    assert result.returned == {"value": shelf.versions()["value"]}
    assert ScriptRunner(pool, shelf=shelf).run("value()", fns).ok


def test_counters_that_cannot_be_read_credit_nothing_and_count_as_raised(run, shelf, monkeypatch):
    def refuses(*args, **kwargs):
        raise RuntimeError("monty worker protocol error")

    real = Watch.read
    monkeypatch.setattr(Watch, "read", lambda self, session: real(self, SimpleNamespace(feed_run=refuses)))

    result = run("value()")

    assert result.ok
    assert result.shelf_used == []
    assert result.returned == {}
    # A raise may be what was lost, so another script's return cannot verify the version.
    assert result.raised == {"value": shelf.versions()["value"]}


@pytest.mark.parametrize(
    "code",
    [
        "globals()  # type: ignore",
        "vars()  # type: ignore",
        "dir()  # type: ignore",
        "eval('1')",
        "exec('x = 1')",
        "[eval][0]('1')",
        "compile('1', 'f', 'eval')  # type: ignore",
        "__import__('gc')  # type: ignore",
        "import gc  # type: ignore",
        "import builtins",
        "import inspect",
        "value.__globals__",
        "value.__closure__",
        "value.__code__",
        "value.__wrapped__  # type: ignore",
        "value.__dict__",
        "getattr(value, '__globals__')  # type: ignore",
        "__builtins__",
    ],
)
def test_a_script_cannot_reach_the_counters(run, code):
    result = run(f"{code}\nvalue()\n1")

    assert result.phase == "runtime", result.error
    assert not HIDDEN.search(str(result.error))
    assert result.returned == {}


def test_a_name_guessed_for_a_counter_reaches_nothing(run):
    code = (
        '_h0000000000000000_in["value"] = 5  # type: ignore\n'
        '_h0000000000000000_out["value"] = 5  # type: ignore\n'
        "if False:\n    value()\n1"
    )

    result = run(code)

    assert result.phase == "runtime", result.error
    assert result.returned == {}
    assert result.shelf_used == []


# The token does reach the script in a caught error. What it must not do is turn into a name.
_READ_THE_TOKEN = "try:\n    mine()\nexcept Exception as error:\n    tag = str(error).split('_mine')[0]\n"
_NEVER_RUNS = "\nif False:\n    value()\n\ndef mine() -> int:\n    return 3\n\n1"


@pytest.mark.parametrize(
    "forge",
    [
        "exec(tag + '_in[\"{name}\"] = 1')\nexec(tag + '_out[\"{name}\"] = 1')",
        "eval(tag + '_in')['{name}'] = 1\neval(tag + '_out')['{name}'] = 1",
        "run_it = [exec][0]\nrun_it(tag + '_in[\"{name}\"] = 1')\nrun_it(tag + '_out[\"{name}\"] = 1')",
        "locals()[tag + '_in']['{name}'] = 1\nlocals()[tag + '_out']['{name}'] = 1",
    ],
)
@pytest.mark.parametrize("name", ["mine", "value"])
def test_a_token_read_from_an_error_forges_no_evidence(run, forge, name):
    attempt = "".join(f"    {line}\n" for line in forge.format(name=name).split("\n"))
    code = f"{_READ_THE_TOKEN}try:\n{attempt}except Exception:\n    pass{_NEVER_RUNS}"

    result = run(code, save=["mine"])

    assert result.ok, result.error
    assert result.saved == ["mine"]
    assert result.shelf_used == []
    assert result.returned == {}


# A default is evaluated in the shelf's feed, before the loader's shadows exist. That feed
# defines functions in sorted order, so `take` comes after a saved `set`, which literal_eval
# would let a default call.
_EVAL_IN_A_DEFAULT = [
    "def take(code: str, evaluate=eval):\n    return evaluate(code)\n",
    "def take(code: str, *, evaluate=eval):\n    return evaluate(code)\n",
    "def take(code: str, box=[eval]):\n    return box[0](code)\n",
    "def set():\n    return eval\n\ndef take(code: str, evaluate=set()):\n    return evaluate(code)\n",
    "def set():\n    return eval\n\ndef take(code: str, box=(set(),)):\n    return box[0](code)\n",
]


def _attempt(line):
    """A line under a catch, so the run goes on if `take` never reached the shelf.

    `nothing` keeps the ignore comment in use whether or not `take` exists, because the
    type check refuses an unused one.
    """
    return f"try:\n    {line}; nothing  # type: ignore\nexcept Exception:\n    pass\n"


@pytest.mark.parametrize("definition", _EVAL_IN_A_DEFAULT)
def test_a_default_keeps_no_real_eval_to_forge_evidence_with(run, shelf, definition):
    run(definition, save=list(top_level_defs(definition)))
    forge = _attempt("take(tag + '_in')['value'] = 1") + _attempt("take(tag + '_out')['value'] = 1")

    result = run(_READ_THE_TOKEN + forge + _NEVER_RUNS, save=["mine"])

    assert result.ok, result.error
    assert result.returned == {}
    assert "take" not in shelf.names()


@pytest.mark.parametrize("definition", _EVAL_IN_A_DEFAULT)
def test_a_default_keeps_no_real_eval_to_erase_a_raise_with(run, shelf, definition):
    run(definition, save=list(top_level_defs(definition)))
    caught = "try:\n    bad()\nexcept RuntimeError:\n    pass\n"
    erase = _attempt("take(tag + '_in').pop('bad')") + _attempt("take(tag + '_out').pop('bad', None)")

    result = run(_READ_THE_TOKEN + caught + erase + "\ndef mine() -> int:\n    return 3\n\n1", save=["mine"])

    assert result.ok, result.error
    assert result.shelf_raised == ["bad"]
    assert result.raised["bad"] == shelf.versions()["bad"]


def test_a_helper_in_a_default_is_refused_and_the_rewrite_around_it_is_counted(pool, tmp_path, fns):
    shelf = Shelf(tmp_path / "other")
    runner = ScriptRunner(pool, shelf=shelf)
    helper = "def helper(place: str) -> str:\n    return go_to(place)\n\n"
    guarded = "def uses(place: str{}) -> int:\n    try:\n        {}(place)\n    except RuntimeError:\n        pass\n    return 7\n\n"

    in_a_default = runner.run(helper + guarded.format(", fn=helper", "fn") + 'uses("shelf")', fns, save=["helper", "uses"])

    assert in_a_default.ok, in_a_default.error
    assert in_a_default.saved == ["helper"]
    assert "literal" in in_a_default.refused["uses"]

    assert runner.run(guarded.format("", "helper"), fns, save=["uses"]).saved == ["uses"]

    result = runner.run('uses("does not exist")', fns)

    assert (result.ok, result.output) == (True, 7), result.error
    assert (result.shelf_used, result.shelf_raised) == (["helper", "uses"], ["helper"])
    assert result.raised == {"helper": shelf.versions()["helper"]}


_SHADOWED = ["locals", "eval", "exec"]


@pytest.mark.parametrize("name", _SHADOWED)
def test_a_name_the_loader_defines_is_never_watched(name):
    code = f"def {name}() -> int:\n    return 73\n\n{name}()"

    watch = Watch(code, {"value": "sha"}, watch_own=True)

    assert watch.own_versions == {}
    assert watch.code == code
    assert watch.stubs == ""


@pytest.mark.parametrize("name", _SHADOWED)
def test_a_name_the_loader_defines_runs_as_written_and_is_neither_saved_nor_credited(run, shelf, name):
    result = run(f"def {name}() -> int:\n    return 73\n\n{name}()", save=[name])

    assert (result.ok, result.output) == (True, 73), result.error
    assert result.saved == []
    assert "built-in" in result.refused[name]
    assert result.returned == {}
    assert name not in shelf.names()


@pytest.mark.parametrize("name", _SHADOWED)
def test_a_script_may_define_a_name_the_loader_defines_for_its_own_use(run, shelf, name):
    result = run(f"def {name}() -> int:\n    return 73\n\nvalue()\n{name}()")

    assert (result.ok, result.output) == (True, 73), result.error
    assert result.returned == {"value": shelf.versions()["value"]}


def test_locals_shows_nothing_hidden(run):
    result = run("value()\nlocals()")

    assert result.ok, result.error
    assert result.output == {}


def test_nothing_hidden_reaches_the_ledger_or_the_print(run):
    code = (
        "try:\n    mine()\nexcept Exception as error:\n    print(error)\n"
        "print(value)\nvalue()\nprint(locals())\n\n"
        "def mine() -> int:\n    return 3\n"
    )

    result = run(code, save=["mine"])

    assert result.ok, result.error
    assert "mine is not allowed" in result.printed
    assert not HIDDEN.search(result.printed)
    assert result.calls == [{"tool": "mine", "status": "denied"}]


def test_a_run_with_nothing_to_watch_has_no_loader(pool, fns, tmp_path):
    empty = Shelf(tmp_path / "empty")

    assert Watch("say('hi')", {}).loader == ""
    assert ScriptRunner(pool, shelf=empty).run("len(locals())", fns).ok


def test_a_definition_saved_and_called_in_its_own_script_has_evidence(run, shelf, fns):
    code = "def mine() -> int:\n    return 3\n\nmine()"

    result = run(code, save=["mine"])

    assert (result.ok, result.output, result.saved) == (True, 3, ["mine"])
    assert result.returned == {"mine": shelf.versions()["mine"]}
    # It is the script's own, so it is no reuse of the shelf.
    assert result.shelf_used == []
    assert shelf.counts()["mine"]["run"] == 0


def test_a_definition_saved_but_never_called_has_no_evidence(run):
    result = run("def mine() -> int:\n    return 3\n", save=["mine"])

    assert result.saved == ["mine"]
    assert result.returned == {}


def test_a_definition_in_dead_code_is_not_called(run):
    result = run("def mine() -> int:\n    return 3\n\nif False:\n    mine()\n1", save=["mine"])

    assert result.saved == ["mine"]
    assert result.returned == {}


def test_a_definition_that_raised_under_a_catch_has_evidence_of_that(run, shelf):
    code = 'def mine() -> str:\n    return go_to("garage")\n\ntry:\n    mine()\nexcept RuntimeError:\n    pass\n1'

    result = run(code, save=["mine"])

    assert result.ok
    assert result.saved == ["mine"]
    assert result.returned == {}
    assert result.raised == {"mine": shelf.versions()["mine"]}


def test_a_definition_not_saved_earns_no_evidence(run):
    code = "def mine() -> int:\n    return 3\n\ndef other() -> int:\n    return 4\n\nmine() + other()"

    result = run(code, save=["other"])

    assert result.output == 7
    assert list(result.returned) == ["other"]


def test_a_refused_save_earns_no_evidence(run):
    code = "LIMIT = 3\n\ndef over() -> int:\n    return LIMIT\n\nover()"

    result = run(code, save=["over"])

    assert result.ok, result.error
    assert result.saved == []
    assert result.returned == {}


@pytest.mark.parametrize(
    ("code", "output"),
    [
        ("def mine() -> int:\n    return 1\n\ndef mine() -> int:\n    return 2\n\nmine()", 2),
        ("def mine() -> int:\n    return 1\n\nfirst = mine()\nmine = 5  # type: ignore\nfirst", 1),
    ],
)
def test_a_definition_bound_twice_runs_as_written_and_earns_no_evidence(run, code, output):
    result = run(code, save=["mine"])

    assert (result.ok, result.output) == (True, output), result.error
    assert result.returned == {}
    # Unwatched, so a raise through it would go unseen: the scorer counts it as raised.
    assert result.raised == {}
    assert {name for name, _ in result.unwatched} == {"mine"}


def test_a_saved_definition_keeps_its_line_numbers_and_name_in_errors(run):
    code = "# Line one.\ndef mine() -> float:\n    return 1 / 0\n\nmine()"

    result = run(code, save=["mine"])

    assert result.phase == "runtime"
    assert "line 3, in mine" in result.error
    assert "line 5" in result.error
    assert not HIDDEN.search(result.error)


def test_a_saved_definition_keeps_its_signature_in_the_type_check(run):
    code = 'def mine(n: int) -> int:\n    return n\n\nmine("one")'

    result = run(code, save=["mine"])

    assert result.phase == "type_check"
    assert not HIDDEN.search(result.error)


def test_a_saved_definition_with_a_body_type_error_is_caught(run):
    code = 'def mine() -> int:\n    return say(1)\n\nmine()'

    result = run(code, save=["mine"])

    assert result.phase == "type_check"
    assert not HIDDEN.search(result.error)


def test_calling_a_saved_definition_before_it_exists_fails_under_its_own_name(run):
    code = "mine()\n\ndef mine() -> int:\n    return 3\n"

    result = run(code, save=["mine"])

    assert not result.ok
    assert result.calls == [{"tool": "mine", "status": "denied"}]
    assert "mine" in result.error
    assert not HIDDEN.search(result.error)


def test_a_frozen_shelf_watches_what_it_loads_and_no_definition(run, pool, shelf, fns):
    shelf.mark_verified(shelf.versions())
    frozen = ScriptRunner(pool, shelf=Shelf(shelf.root, frozen=True))

    result = frozen.run("def mine() -> int:\n    return value()\n\nmine()", fns, save=["mine"])

    assert result.output == 1
    assert result.shelf_used == ["value"]
    assert list(result.returned) == ["value"]


def test_a_verified_function_naming_one_that_never_ran_stays_off_the_frozen_shelf(pool, tmp_path, fns):
    shelf = Shelf(tmp_path / "other")
    code = (
        "def helper() -> int:\n    return 1\n\n"
        "def uses(flag: bool) -> int:\n    return helper() if flag else 2\n\n"
        "def leaf() -> int:\n    return 3\n"
    )
    assert shelf.save(code, ["helper", "uses", "leaf"])[0] == ["helper", "leaf", "uses"]
    runner = ScriptRunner(pool, shelf=shelf)

    shelf.mark_verified(runner.run("uses(False) + leaf()", fns).returned)

    frozen = Shelf(shelf.root, frozen=True)
    assert frozen.names() == ["leaf"]
    assert ScriptRunner(pool, shelf=frozen).run("leaf()", fns).output == 3
