"""The typed readers of config.toml's tables (walkie_agent.config_tables).

The readers are plain logic over a parsed TOML file. They are tried on a
sample table with a key of each type they read. The skip of RAI's top-level
tables belongs to walkie_agent.tuning, and is tried on config.toml.
"""

import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT
from walkie_agent import config_tables, tuning

CONFIG = PROJECT_ROOT / "config.toml"
SRC = PROJECT_ROOT / "src"


@dataclass(frozen=True)
class Sample:
    """A table with a field of each type the readers take."""

    rate_hz: float = 15.0
    retries: int = 1
    enabled: bool = True
    voice: str = "af_heart"
    words: tuple[str, ...] = ("yes", "no")
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError(f"Sample.retries must be at least 0, not {self.retries}.")


@dataclass(frozen=True)
class Limits:
    """A table with a field that no key can set."""

    limits: Mapping[str, float] = field(default_factory=dict)


SAMPLE_KEYS = ("rate_hz", "retries", "enabled", "voice", "words", "deadline_s")


def read(toml: str) -> tuple[Sample, tuple[str, ...]]:
    """The Sample that [walkie.sample] in `toml` sets, and every problem found."""
    errors: list[str] = []
    walkie = config_tables.table(tomllib.loads(toml), "walkie", errors)
    raw = config_tables.table(walkie, "walkie.sample", errors)
    config_tables.refuse_unknown(raw, SAMPLE_KEYS, "walkie.sample", errors)
    changes = config_tables.read_keys(Sample, raw, SAMPLE_KEYS, "walkie.sample", errors)
    return config_tables.replace(Sample(), changes, "walkie.sample", errors), tuple(errors)


def problems(toml: str) -> tuple[str, ...]:
    sample, errors = read(toml)
    # A refused value leaves its field on the default.
    assert sample == Sample()
    return errors


# -- Reading each type --------------------------------------------------------------------------


def test_without_the_table_every_field_keeps_its_default():
    assert read('[vendor]\ncomplex_model = "openai"\n') == (Sample(), ())


def test_each_key_is_read_as_its_field_is_typed():
    sample, errors = read(
        """
[walkie.sample]
rate_hz = 10
retries = 3
enabled = false
voice = "am_adam"
words = ["coke", "orange juice"]
deadline_s = 2.5
"""
    )

    assert errors == ()
    assert sample == Sample(
        rate_hz=10.0,
        retries=3,
        enabled=False,
        voice="am_adam",
        words=("coke", "orange juice"),
        deadline_s=2.5,
    )
    # A float field holds a float, and a tuple field a tuple, not TOML's list.
    assert isinstance(sample.rate_hz, float)
    assert isinstance(sample.words, tuple)


def test_a_float_or_none_left_out_keeps_none():
    sample, errors = read("[walkie.sample]\nrate_hz = 12.0\n")

    assert errors == ()
    assert sample.deadline_s is None


def test_a_float_or_none_that_is_set_is_read_as_a_number():
    sample, errors = read("[walkie.sample]\ndeadline_s = 3\n")

    assert errors == ()
    assert sample.deadline_s == 3.0
    assert isinstance(sample.deadline_s, float)


def test_a_tuple_of_strings_may_be_empty():
    sample, errors = read("[walkie.sample]\nwords = []\n")

    assert errors == ()
    assert sample.words == ()


# -- Refusing bad values ------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "shown"), [("1.5", "1.5"), ("2.0", "2.0"), ("true", "True")])
def test_an_int_refuses_a_float_and_a_boolean(value, shown):
    assert problems(f"[walkie.sample]\nretries = {value}\n") == (
        f"walkie.sample.retries = {shown} is not an integer.",
    )


@pytest.mark.parametrize(("value", "shown"), [("1", "1"), ("0", "0"), ('"true"', "'true'")])
def test_a_bool_takes_only_true_or_false(value, shown):
    assert problems(f"[walkie.sample]\nenabled = {value}\n") == (
        f"walkie.sample.enabled = {shown} is not true or false.",
    )


def test_a_str_refuses_a_number():
    assert problems("[walkie.sample]\nvoice = 1\n") == ("walkie.sample.voice = 1 is not a string.",)


@pytest.mark.parametrize(("value", "shown"), [('"coke"', "'coke'"), ('["coke", 1]', "['coke', 1]")])
def test_a_tuple_takes_only_a_list_of_strings(value, shown):
    assert problems(f"[walkie.sample]\nwords = {value}\n") == (
        f"walkie.sample.words = {shown} is not a list of strings.",
    )


def test_a_float_or_none_refuses_what_a_float_refuses():
    assert problems("[walkie.sample]\ndeadline_s = true\nrate_hz = nan\n") == (
        "walkie.sample.rate_hz = nan is not a finite number.",
        "walkie.sample.deadline_s = True is not a number.",
    )


def test_an_unknown_key_is_refused_with_what_the_table_takes():
    assert problems("[walkie.sample]\ndeadline = 2.5\n") == (
        "[walkie.sample]: unknown key 'deadline'; it takes rate_hz, retries, enabled, voice, "
        "words, deadline_s.",
    )


def test_a_value_the_dataclass_refuses_is_refused_with_its_reason():
    # The type is right; __post_init__ refuses the value.
    assert problems('[walkie.sample]\nretries = -1\nvoice = "am_adam"\n') == (
        "[walkie.sample]: Sample.retries must be at least 0, not -1.",
    )


# -- Mistakes in the code -----------------------------------------------------------------------


def test_a_field_no_key_can_set_raises_even_when_the_table_leaves_it_out():
    # A mistake in the code, so it fails in the tests rather than on the day a
    # config.toml first sets the key.
    with pytest.raises(TypeError, match=r"^Limits\.limits is a .*Mapping\[str, float\]"):
        config_tables.read_keys(Limits, {}, ("limits",), "walkie.limits", [])
    with pytest.raises(TypeError, match="^Sample has no field 'rate'"):
        config_tables.read_keys(Sample(), {}, ("rate",), "walkie.sample", [])


# -- RAI's tables -------------------------------------------------------------------------------


def test_rais_top_level_tables_are_the_ones_config_toml_has():
    data = tomllib.loads(CONFIG.read_text())

    assert set(data) - {"walkie"} == set(tuning.RAI_TABLES)


@pytest.mark.parametrize("name", tuning.RAI_TABLES)
def test_rais_top_level_tables_are_never_refused_even_named_as_a_walkie_table(
    monkeypatch, name
):
    # As a [walkie.asr] table would name rai_s2s's [asr]: without the skip,
    # [asr] would be refused as a walkie table without its [walkie.] prefix.
    expected = tuning.load_tuning(CONFIG)
    monkeypatch.setattr(tuning, "_TABLE_NAMES", (*tuning._TABLE_NAMES, name))
    data = tomllib.loads(CONFIG.read_text())

    assert name in data
    assert tuning.parse_tuning(data) == expected


# -- The module ---------------------------------------------------------------------------------


def test_the_readers_import_nothing_of_the_agent():
    # In a fresh interpreter, since this one has the agent and rclpy loaded
    # already: a process other than the agent reads its tables with them.
    code = (
        "import json, sys\n"
        "import walkie_agent.config_tables as module\n"
        "roots = ('walkie_agent.tuning', 'walkie_agent.agent', 'rclpy')\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if any(name == root or name.startswith(root + '.') for root in roots)\n"
        ")\n"
        "walkie = sorted(name for name in sys.modules if name.split('.')[0] == 'walkie_agent')\n"
        "print(json.dumps({'file': module.__file__, 'loaded': loaded, 'walkie': walkie}))\n"
    )
    path = os.pathsep.join(part for part in (str(SRC), os.environ.get("PYTHONPATH")) if part)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    found = json.loads(result.stdout.strip().splitlines()[-1])

    assert found["loaded"] == []
    assert found["walkie"] == ["walkie_agent", "walkie_agent.config_tables"]
    assert Path(found["file"]).resolve().is_relative_to(SRC.resolve())


def test_tuning_error_keeps_its_old_import_path():
    assert tuning.TuningError is config_tables.TuningError
