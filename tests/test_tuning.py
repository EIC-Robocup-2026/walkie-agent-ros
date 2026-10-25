"""The perception and gaze values that config.toml sets (walkie_agent.tuning).

The loader is plain logic over a parsed TOML file. One test builds the agent,
to check that the configs reach the vision feed, the gaze loop and the
perception atomics.
"""

import math
import re
import tomllib

import pytest

from conftest import PROJECT_ROOT
from scripted_llm import ScriptedChatModel
from walkie_agent.agent import create_walkie_agent
from walkie_agent.gaze import DEFAULT_CONFIG as DEFAULT_GAZE_CONFIG
from walkie_agent.gaze import GazeConfig
from walkie_agent.perception.front import FrontConfig
from walkie_agent.perception.identity import IdentityConfig
from walkie_agent.perception.vision_feed import DEFAULT_CONFIG as DEFAULT_VISION_FEED_CONFIG
from walkie_agent.tools.atomic import _perception
from walkie_agent.tools.atomic._perception import PerceptionConfig
from walkie_agent.tuning import (
    FRONT_KEYS,
    GAZE_KEYS,
    PERCEPTION_KEYS,
    VISION_FEED_KEYS,
    Tuning,
    TuningError,
    load_tuning,
    parse_tuning,
)

CONFIG = PROJECT_ROOT / "config.toml"

# Every key the loader reads, as config.toml's example writes it.
EVERY_KEY = """
[walkie.perception]
tf_timeout_s = 0.35

[walkie.perception.front]
min_face_rad = 0.12
half_angle_deg = 20.0

[walkie.perception.vision_feed]
min_face_size_rad = 0.05

[walkie.perception.identity.face_distance_limits]
arcface_w600k_r50 = 0.45

[walkie.perception.identity.appearance_distance_limits]
osnet_x1_0_msmt17 = 0.3

[walkie.gaze]
tf_wait_s = 0.4
"""

# Every (table, key) that config.toml's example shows: the keys the loader
# reads, and the real models' distance limits.
EXAMPLE_KEYS = (
    {("walkie.perception", key) for key in PERCEPTION_KEYS}
    | {("walkie.perception.front", key) for key in FRONT_KEYS}
    | {("walkie.perception.vision_feed", key) for key in VISION_FEED_KEYS}
    | {
        ("walkie.perception.identity.face_distance_limits", "arcface_w600k_r50"),
        ("walkie.perception.identity.appearance_distance_limits", "osnet_x1_0_msmt17"),
    }
    | {("walkie.gaze", key) for key in GAZE_KEYS}
)

# A table or key line of config.toml, commented or not: "[table]" or
# "key = value", after "# " when commented. Prose comments hold no " = ".
_TABLE_LINE = re.compile(r"^(# )?\[([a-z0-9_.]+)\]$")
_KEY_LINE = re.compile(r"^(# )?([a-z0-9_]+) = (\S+)$")


def problems(toml: str) -> tuple[str, ...]:
    with pytest.raises(TuningError) as caught:
        parse_tuning(tomllib.loads(toml))
    return caught.value.errors


def walkie_key_lines(text: str) -> list[tuple[str, str, str, bool]]:
    """(table, key, value, commented) for each key line of a [walkie...] table in `text`.

    Tables and keys count whether they are commented or not, so a key
    belongs to the table above it after a tuning edit too: config.toml says
    to uncomment a table and the keys that change.
    """
    lines = []
    table = None
    for line in text.splitlines():
        if match := _TABLE_LINE.match(line):
            table = match[2]
        elif table is not None and table.startswith("walkie."):
            if match := _KEY_LINE.match(line):
                lines.append((table, match[2], match[3], match[1] is not None))
    return lines


# -- Defaults -----------------------------------------------------------------------------------


def test_without_walkie_tables_every_config_keeps_its_default():
    tuning = parse_tuning(tomllib.loads('[vendor]\ncomplex_model = "openai"\n'))

    assert tuning == Tuning()
    assert tuning.perception == PerceptionConfig() == _perception.CONFIG
    assert tuning.vision_feed is DEFAULT_VISION_FEED_CONFIG
    assert tuning.gaze is DEFAULT_GAZE_CONFIG


def test_the_repository_config_loads():
    # Whatever has been tuned in it: the agent must start on it.
    assert isinstance(load_tuning(CONFIG), Tuning)


def test_config_toml_shows_every_key():
    # Commented or tuned, every key the loader reads is in the file, in its table.
    keys = {(table, key) for table, key, _, _ in walkie_key_lines(CONFIG.read_text())}

    assert keys == EXAMPLE_KEYS


def test_the_commented_keys_in_config_toml_show_their_defaults():
    tables: dict[str, list[str]] = {}
    for table, key, value, commented in walkie_key_lines(CONFIG.read_text()):
        if commented:
            tables.setdefault(table, []).append(f"{key} = {value}")
    example = "\n".join(f"[{table}]\n" + "\n".join(keys) for table, keys in tables.items())

    # Uncommented, the keys still commented change nothing.
    assert parse_tuning(tomllib.loads(example)) == Tuning()


def test_a_tuning_edit_keeps_the_config_toml_checks_passing(tmp_path):
    # As config.toml says: uncomment a table and the keys you change.
    text = CONFIG.read_text()
    assert "# [walkie.perception.front]\n# min_face_rad = 0.10\n" in text
    tuned = text.replace(
        "# [walkie.perception.front]\n# min_face_rad = 0.10\n",
        "[walkie.perception.front]\nmin_face_rad = 0.12\n",
    )
    path = tmp_path / "config.toml"
    path.write_text(tuned)

    assert load_tuning(path).perception.front == FrontConfig(min_face_rad=0.12)
    lines = walkie_key_lines(tuned)
    assert {(table, key) for table, key, _, _ in lines} == EXAMPLE_KEYS
    # The key left commented still belongs to its table.
    assert ("walkie.perception.front", "half_angle_deg", "25.0", True) in lines


def test_the_defaults_are_the_values_the_code_had():
    # The values hri.md, perception.md and the code comments give. A
    # change here changes what the robot does without a config section.
    perception = PerceptionConfig()
    assert perception.tf_timeout_s == 0.2
    assert (perception.front.min_face_rad, perception.front.half_angle_deg) == (0.10, 25.0)
    assert DEFAULT_VISION_FEED_CONFIG.min_face_size_rad == 0.0469
    assert perception.identity.face_distance_limits["arcface_w600k_r50"] == 0.4
    assert perception.identity.appearance_distance_limits["osnet_x1_0_msmt17"] == 0.5
    gaze = GazeConfig()
    assert (gaze.gain, gaze.min_speed, gaze.max_speed) == (1.2, 0.15, 0.6)
    assert (gaze.deadband_deg, gaze.tf_wait_s) == (6.0, 0.2)


# -- Setting values -----------------------------------------------------------------------------


def test_the_tables_set_their_values_and_leave_the_rest_alone():
    tuning = parse_tuning(tomllib.loads(EVERY_KEY))

    perception = tuning.perception
    assert perception.tf_timeout_s == 0.35
    assert perception.front == FrontConfig(min_face_rad=0.12, half_angle_deg=20.0)
    assert tuning.vision_feed.min_face_size_rad == 0.05
    assert tuning.gaze == GazeConfig(tf_wait_s=0.4)
    faces = perception.identity.face_distance_limits
    appearances = perception.identity.appearance_distance_limits
    assert faces["arcface_w600k_r50"] == 0.45
    assert appearances["osnet_x1_0_msmt17"] == 0.3
    # The models the tables leave out keep their limits, and the limits
    # stay read-only.
    assert faces["fake_face"] == 0.2
    assert appearances["fake_reid"] == 0.2
    with pytest.raises(TypeError):
        faces["arcface_w600k_r50"] = 1.0  # type: ignore[index]
    # Values that config.toml cannot set keep the code's.
    defaults = PerceptionConfig()
    assert (perception.max_age_s, perception.wait_s, perception.pair_wait_s) == (
        defaults.max_age_s,
        defaults.wait_s,
        defaults.pair_wait_s,
    )
    assert perception.identity.outlier_mad_k == defaults.identity.outlier_mad_k
    assert perception.seating == defaults.seating
    assert perception.gestures == defaults.gestures
    assert tuning.vision_feed.min_face_det_score == DEFAULT_VISION_FEED_CONFIG.min_face_det_score
    # The gaze loop's control values are v2's, tuned on the robot.
    gaze, default = tuning.gaze, DEFAULT_GAZE_CONFIG
    assert (gaze.gain, gaze.min_speed, gaze.max_speed, gaze.deadband_deg) == (
        default.gain,
        default.min_speed,
        default.max_speed,
        default.deadband_deg,
    )
    assert (gaze.rate_hz, gaze.lost_after_s, gaze.yaw_max_age_s) == (
        default.rate_hz,
        default.lost_after_s,
        default.yaw_max_age_s,
    )
    assert gaze.cmd_vel_topic == "/cmd_vel"


def test_one_key_changes_only_its_own_value():
    tuning = parse_tuning(tomllib.loads("[walkie.perception.front]\nhalf_angle_deg = 30.0\n"))

    assert tuning.perception.front == FrontConfig(half_angle_deg=30.0)
    assert tuning.perception.identity == IdentityConfig()
    assert tuning.vision_feed is DEFAULT_VISION_FEED_CONFIG
    assert tuning.gaze is DEFAULT_GAZE_CONFIG


def test_an_integer_is_a_number():
    tuning = parse_tuning(tomllib.loads("[walkie.perception.front]\nhalf_angle_deg = 20\n"))

    assert tuning.perception.front.half_angle_deg == 20.0
    assert isinstance(tuning.perception.front.half_angle_deg, float)


def test_load_tuning_reads_a_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(EVERY_KEY)

    assert load_tuning(path) == parse_tuning(tomllib.loads(EVERY_KEY))


# -- Refusing bad values ------------------------------------------------------------------------


def test_unknown_tables_and_keys_are_refused_with_what_the_table_takes():
    errors = problems(
        """
[walkie.gase]
gain = 1.0

[walkie.perception]
min_face_rad = 0.12

[walkie.perception.front]
min_face_px = 107

[walkie.perception.identity]
outlier_mad_k = 2.0

[walkie.gaze]
gain = 1.0
cmd_vel_topic = "/cmd_vel_nav"
lost_after_s = 2.0
"""
    )

    assert errors == (
        "[walkie]: unknown key 'gase'; it takes perception, gaze.",
        "[walkie.perception]: unknown key 'min_face_rad'; it takes tf_timeout_s, front, "
        "vision_feed, identity.",
        "[walkie.perception.front]: unknown key 'min_face_px'; it takes min_face_rad, "
        "half_angle_deg.",
        "[walkie.perception.identity]: unknown key 'outlier_mad_k'; it takes "
        "face_distance_limits, appearance_distance_limits.",
        # The control values are v2's, tuned on the robot, and stay in the code.
        "[walkie.gaze]: unknown key 'gain'; it takes tf_wait_s.",
        "[walkie.gaze]: unknown key 'cmd_vel_topic'; it takes tf_wait_s.",
        "[walkie.gaze]: unknown key 'lost_after_s'; it takes tf_wait_s.",
    )


def test_a_top_level_table_meant_for_walkie_is_refused():
    # Each would be read by nothing, and the defaults would stay in place.
    errors = problems(
        """
[walky.gaze]
tf_wait_s = 0.4

[gaze]
tf_wait_s = 0.4

[walkie_perception]
tf_timeout_s = 0.3

[front]
min_face_rad = 0.12
"""
    )

    assert errors == tuple(
        f"Unknown table [{name}]; the agent's tables go under [walkie], which takes "
        "perception, gaze."
        for name in ("walky", "gaze", "walkie_perception", "front")
    )


def test_rais_own_tables_are_left_alone():
    data = tomllib.loads(CONFIG.read_text())
    data.update(tomllib.loads('[aws]\nregion_name = "x"\n[google]\nsimple_model = "x"\n'))

    assert set(data) - {"walkie"} >= {"vendor", "openai", "ollama", "tracing", "asr", "tts"}
    assert parse_tuning(data) == load_tuning(CONFIG)


def test_a_limit_for_a_model_without_one_is_refused():
    # A misspelt model would leave the model meant on its old limit.
    errors = problems(
        "[walkie.perception.identity.face_distance_limits]\narcface_w600k_r5 = 0.45\n"
        "[walkie.perception.identity.appearance_distance_limits]\nosnet = 0.3\n"
    )

    assert errors == (
        "[walkie.perception.identity.face_distance_limits]: unknown model "
        "'arcface_w600k_r5'; the models with a limit are arcface_w600k_r50, fake_face.",
        "[walkie.perception.identity.appearance_distance_limits]: unknown model 'osnet'; "
        "the models with a limit are osnet_x1_0_msmt17, fake_reid.",
    )


def test_values_that_are_not_numbers_are_refused():
    errors = problems(
        """
[walkie.perception]
tf_timeout_s = "0.3"

[walkie.perception.front]
min_face_rad = true
half_angle_deg = inf

[walkie.perception.identity.face_distance_limits]
arcface_w600k_r50 = nan

[walkie.gaze]
tf_wait_s = [0.2]
"""
    )

    assert errors == (
        "walkie.perception.front.min_face_rad = True is not a number.",
        "walkie.perception.front.half_angle_deg = inf is not a finite number.",
        "walkie.perception.identity.face_distance_limits.arcface_w600k_r50 = nan is not a "
        "finite number.",
        "walkie.perception.tf_timeout_s = '0.3' is not a number.",
        "walkie.gaze.tf_wait_s = [0.2] is not a number.",
    )


def test_a_table_written_as_a_value_is_refused():
    assert problems("walkie = 1\n") == ("walkie = 1 is not a table; write it as [walkie].",)
    assert problems("[walkie]\ngaze = 0.5\n") == (
        "walkie.gaze = 0.5 is not a table; write it as [walkie.gaze].",
    )
    assert problems("[walkie.perception.identity]\nface_distance_limits = 0.4\n") == (
        "walkie.perception.identity.face_distance_limits = 0.4 is not a table; write it as "
        "[walkie.perception.identity.face_distance_limits].",
    )


def test_values_a_config_refuses_are_refused_with_its_reason():
    errors = problems(
        """
[walkie.perception]
tf_timeout_s = 0.0

[walkie.perception.front]
half_angle_deg = 180.0

[walkie.perception.vision_feed]
min_face_size_rad = -0.01

[walkie.perception.identity.appearance_distance_limits]
osnet_x1_0_msmt17 = 2.0

[walkie.gaze]
tf_wait_s = 0.0
"""
    )

    assert errors == (
        "[walkie.perception.front]: FrontConfig.half_angle_deg must lie in (0, 180).",
        "[walkie.perception.vision_feed]: VisionFeedConfig.min_face_size_rad must be positive.",
        "[walkie.perception.identity]: IdentityConfig.appearance_distance_limits"
        "['osnet_x1_0_msmt17'] must lie in (0, 2), not 2.0.",
        "[walkie.perception]: PerceptionConfig.tf_timeout_s must be a positive number, not 0.0.",
        "[walkie.gaze]: GazeConfig.tf_wait_s must be a positive number, not 0.0.",
    )


@pytest.mark.parametrize(
    ("toml", "front", "gate"),
    [
        # Fewer noisy embeddings: the face gate rises above the front rule's limit.
        ("[walkie.perception.vision_feed]\nmin_face_size_rad = 0.12\n", "0.1", "0.12"),
        # A deep doorway: the front rule drops below the face gate.
        ("[walkie.perception.front]\nmin_face_rad = 0.04\n", "0.04", "0.0469"),
    ],
)
def test_a_front_rule_that_keeps_faces_too_small_to_recognize_is_refused(toml, front, gate):
    # Each value alone is one its config takes.
    assert problems(toml) == (
        f"[walkie.perception.front]: min_face_rad {front} is below "
        f"[walkie.perception.vision_feed] min_face_size_rad {gate}; a person kept in front "
        "must have a face that can be recognized.",
    )


def test_a_front_rule_on_the_face_gate_is_kept():
    tuning = parse_tuning(tomllib.loads("[walkie.perception.front]\nmin_face_rad = 0.0469\n"))

    assert tuning.perception.front.min_face_rad == tuning.vision_feed.min_face_size_rad


def test_the_error_names_the_file_and_lists_every_problem(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[walkie.gaze]\ntf_wait_s = 0.0\ntf_wait = 0.4\n")

    with pytest.raises(TuningError) as caught:
        load_tuning(path)

    assert str(caught.value) == (
        f"{path}: 2 problems:\n"
        "- [walkie.gaze]: unknown key 'tf_wait'; it takes tf_wait_s.\n"
        "- [walkie.gaze]: GazeConfig.tf_wait_s must be a positive number, not 0.0."
    )
    # A ValueError, as the other config loaders raise, so start-up stops on it.
    assert isinstance(caught.value, ValueError)


def test_identity_and_perception_configs_check_their_values_themselves():
    with pytest.raises(ValueError, match=r"face_distance_limits\['m'\] must lie in \(0, 2\)"):
        IdentityConfig(face_distance_limits={"m": 0.0})
    with pytest.raises(ValueError, match="PerceptionConfig.wait_s must be a positive number"):
        PerceptionConfig(wait_s=-1.0)
    with pytest.raises(ValueError, match="PerceptionConfig.max_age_s"):
        PerceptionConfig(max_age_s=math.nan)


# -- The agent ----------------------------------------------------------------------------------


def test_the_agent_is_built_with_the_tuning(connector, locations, monkeypatch):
    # create_walkie_agent replaces the module's config; monkeypatch puts
    # the default back after the test.
    monkeypatch.setattr(_perception, "CONFIG", _perception.CONFIG)
    tuning = parse_tuning(tomllib.loads(EVERY_KEY))

    agent = create_walkie_agent(
        ScriptedChatModel(responses=[]), connector, locations, target_connectors={}, tuning=tuning
    )

    assert agent.toolbox.ctx.backend("vision").config is tuning.vision_feed
    assert agent.toolbox.ctx.backend("gaze").config is tuning.gaze
    assert _perception.CONFIG is tuning.perception


def test_without_a_tuning_the_agent_keeps_the_code_defaults(connector, locations):
    config = _perception.CONFIG

    agent = create_walkie_agent(
        ScriptedChatModel(responses=[]), connector, locations, target_connectors={}
    )

    assert agent.toolbox.ctx.backend("vision").config is DEFAULT_VISION_FEED_CONFIG
    assert agent.toolbox.ctx.backend("gaze").config is DEFAULT_GAZE_CONFIG
    assert _perception.CONFIG is config
