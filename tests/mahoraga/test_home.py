"""The gym's home: where its shelf, gate results and logs live, always outside git."""

from pathlib import Path

import pytest

import walkie_agent.mahoraga
from walkie_agent.mahoraga import home, shelf_dir
from walkie_agent.scripting.shelf import Shelf


def test_home_comes_from_the_argument_then_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MAHORAGA_HOME", str(tmp_path / "from-env"))

    assert home() == tmp_path / "from-env"
    assert home(tmp_path / "given") == tmp_path / "given"
    assert Shelf(shelf_dir()).root == tmp_path / "from-env" / "shelf"


def test_home_defaults_outside_the_repo(monkeypatch, tmp_path):
    monkeypatch.delenv("MAHORAGA_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert home() == tmp_path / ".local/share/mahoraga"


def test_a_home_in_a_git_work_tree_is_refused():
    with pytest.raises(ValueError, match="git work tree"):
        home(Path(walkie_agent.mahoraga.__file__).parent / "shelf")
