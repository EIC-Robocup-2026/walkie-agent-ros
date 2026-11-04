"""The research dependencies stay in Mahoraga's own pixi environment, out of default."""

import importlib.util

import pytest


@pytest.mark.parametrize("module", ["pydantic_monty", "inspect_ai", "robocupathome_generator"])
def test_a_mahoraga_dependency_is_not_in_the_default_environment(module):
    assert importlib.util.find_spec(module) is None
