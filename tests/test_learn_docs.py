"""The runnable examples on the docs/learn/ pages, which are doctests in pycon fences."""

import doctest

import pytest

from conftest import PROJECT_ROOT

# The current architecture replaces this comparison, so the docs stage deletes the page.
DELETED_PAGES = {"old-vs-new.md"}
LEARN_PAGES = [
    page
    for page in sorted((PROJECT_ROOT / "docs" / "learn").glob("*.md"))
    if page.name not in DELETED_PAGES
]
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
# The legacy cut deleted the classes these examples use. A strict mark fails as soon as a
# page's examples pass, so the rewrite has to remove the mark along with the old examples.
AWAITING_REWRITE = pytest.mark.xfail(
    strict=True,
    reason="The examples use modules the legacy cut deleted. The docs stage rewrites them.",
)


def run_examples(page):
    return doctest.testfile(
        str(page),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )


def test_the_learn_section_has_pages():
    assert LEARN_PAGES


@pytest.mark.parametrize(
    "page",
    [
        pytest.param(page, id=page.name, marks=[AWAITING_REWRITE] if page.name in CORE_PAGES else [])
        for page in LEARN_PAGES
    ],
)
def test_learn_page_examples_match_the_code(page):
    # A failing example prints the expected and actual output to stdout,
    # which pytest shows with the failure.
    result = run_examples(page)
    assert result.failed == 0, f"{result.failed} example(s) on {page.name} no longer match the code"


@pytest.mark.parametrize("name", CORE_PAGES)
def test_core_learning_pages_execute_examples(name):
    assert run_examples(PROJECT_ROOT / "docs" / "learn" / name).attempted > 0
