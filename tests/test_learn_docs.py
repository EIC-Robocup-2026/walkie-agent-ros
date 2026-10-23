"""The runnable examples on the docs/learn/ pages, which are doctests in pycon fences."""

import doctest

import pytest

from conftest import PROJECT_ROOT

LEARN_PAGES = sorted((PROJECT_ROOT / "docs" / "learn").glob("*.md"))


def test_the_learn_section_has_pages():
    assert LEARN_PAGES


@pytest.mark.parametrize("page", LEARN_PAGES, ids=lambda page: page.name)
def test_learn_page_examples_match_the_code(page):
    # A failing example prints the expected and actual output to stdout,
    # which pytest shows with the failure.
    result = doctest.testfile(
        str(page),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert result.failed == 0, f"{result.failed} example(s) on {page.name} no longer match the code"
