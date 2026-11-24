"""Reading config.toml and the agent's settings from it."""

import tomllib

import pytest

from walkie_agent.config import (
    DEFAULT_COMPACT_AFTER_TOKENS,
    DEFAULT_COMPACT_KEEP_TOKENS,
    agent_settings,
    compaction_settings,
    load_config,
)


def test_load_config_reads_the_named_file(tmp_path):
    path = tmp_path / "other.toml"
    path.write_text('[agent]\nrun_limit = 7\ndeadline_s = 1.5\n')
    assert agent_settings(load_config(path)) == {"run_limit": 7, "deadline_s": 1.5}


def test_compaction_has_defaults_that_the_agent_table_overrides(tmp_path):
    assert compaction_settings({}) == {
        "after_tokens": DEFAULT_COMPACT_AFTER_TOKENS,
        "keep_tokens": DEFAULT_COMPACT_KEEP_TOKENS,
    }
    path = tmp_path / "config.toml"
    path.write_text("[agent]\ncompact_after_tokens = 0\ncompact_keep_tokens = 500\n")
    assert compaction_settings(load_config(path)) == {"after_tokens": 0, "keep_tokens": 500}


@pytest.mark.parametrize(
    "line",
    [
        "compact_keep_tokens = 0",  # would leave no turn to cut at
        "compact_after_tokens = -1",  # would compact after every task
        "compact_after_tokens = 1.5",
        "compact_keep_tokens = true",
    ],
)
def test_a_compaction_threshold_out_of_range_is_refused_with_its_key(line):
    with pytest.raises(ValueError, match=rf"\[agent\] {line.split()[0]} must be a whole number"):
        compaction_settings({"agent": tomllib.loads(line)})


def test_a_config_that_is_not_toml_is_refused_with_its_path(tmp_path):
    # main() prints this one line alone, so the line has to say which file is broken.
    path = tmp_path / "config.toml"
    path.write_text("not toml = [")
    with pytest.raises(ValueError, match=r"line 1, column 5") as caught:
        load_config(path)
    assert str(caught.value).startswith(f"{path}: ")
