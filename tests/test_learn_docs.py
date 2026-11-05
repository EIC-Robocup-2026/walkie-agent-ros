"""The runnable examples on the docs/learn/ pages, which are doctests in pycon fences."""

import doctest
import threading

import pytest

from conftest import PROJECT_ROOT

LEARN_DIR = PROJECT_ROOT / "docs" / "learn"
LEARN_PAGES = sorted(LEARN_DIR.glob("*.md"))
# These pages must keep real examples, so that stripping every pycon block cannot pass.
CORE_PAGES = [
    "asking-questions.md",
    "one-task.md",
    "queue-stop-reset.md",
    "rai-and-our-graph.md",
    "read-the-code.md",
    "seeding.md",
    "the-graph.md",
    "todo-items.md",
    "todo-rules.md",
    "watching-progress.md",
]
# The section's landing page only links to the others.
PAGES_WITHOUT_EXAMPLES = {"index.md"}


def run_examples(page):
    return doctest.testfile(
        str(page),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )


def test_the_learn_section_has_pages():
    assert LEARN_PAGES


def test_every_learn_page_is_designated():
    # A new page has to choose a side, and a removed core page fails here.
    assert {page.name for page in LEARN_PAGES} == set(CORE_PAGES) | PAGES_WITHOUT_EXAMPLES


@pytest.mark.parametrize("page", LEARN_PAGES, ids=lambda page: page.name)
def test_learn_page_examples_match_the_code(page):
    # A failing example prints the expected and actual output to stdout,
    # which pytest shows with the failure.
    before = set(threading.enumerate())
    result = run_examples(page)
    assert result.failed == 0, f"{result.failed} example(s) on {page.name} no longer match the code"
    # A Runner left running would fail a later test's thread check, far from its cause.
    assert [thread.name for thread in set(threading.enumerate()) - before] == []


@pytest.mark.parametrize("name", CORE_PAGES)
def test_core_learning_pages_execute_examples(name):
    assert run_examples(LEARN_DIR / name).attempted > 0, f"{name} has no examples left to run"


def test_a_page_with_its_examples_stripped_attempts_nothing(tmp_path):
    # The control for the check above: a stripped page passes on failures alone.
    page = tmp_path / "stripped.md"
    page.write_text("# A page\n\nThe prose stayed, and the pycon blocks did not.\n")
    result = run_examples(page)
    assert (result.failed, result.attempted) == (0, 0)
