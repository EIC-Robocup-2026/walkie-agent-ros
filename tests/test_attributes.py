"""The attribute rules on synthetic colour shares and cloud answers, with no images.

The arbitrate tests start from a person in a blue top and black bottoms,
seen in three agreeing views, with a cloud answer that agrees and is sure
of every field. Each test changes one thing and checks the field the rule
decides, so a test names the rule it covers. The sentence tests build a
line from every set of up to 4 fields and read it back through the HRI
scenario's classify_line, the HRI check's pronoun rule and parse_mentions.
"""

import dataclasses
import itertools
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from walkie_agent.perception import attributes
from walkie_agent.perception.attributes import (
    ATTRIBUTES,
    CLOUD,
    CLOUD_AND_LOCAL,
    CLOUD_DISAGREES,
    CLOUD_ONLY_ATTRIBUTES,
    CLOUD_UNKNOWN,
    CLOUD_UNSURE,
    COLOR_NAMES,
    DEFAULT_SAY,
    FIELD_VALUES,
    GATED,
    IN_A_DRESS,
    LOCAL,
    MAYBE_HAT,
    NEGATIVE,
    NO_CLOUD,
    NO_LOCAL_AGREEMENT,
    NOT_ALLOWED,
    NOT_IN_SAY,
    OVER_MAX_ATTRIBUTES,
    PRINTED_TEXT,
    REASONS,
    SAYABLE_VALUES,
    SOURCES,
    TOO_FEW_VIEWS,
    UNKNOWN,
    AttributeThresholds,
    CloudAnswer,
    Mention,
    View,
    arbitrate,
    parse_mentions,
    region_color,
    sentence,
    vote,
)
from walkie_agent.sim.hri_check import PRONOUNS
from walkie_agent.sim.hri_scenario import Line, classify_line

_THRESHOLDS = AttributeThresholds()
# Every field other than the colours may be said on the cloud's word.
_ALLOW_ALL = AttributeThresholds(allow_cloud_only=CLOUD_ONLY_ATTRIBUTES)


def _shares(colors: Mapping[str, float] | None) -> list[float]:
    """Colour shares in COLOR_NAMES order, 0 for the colours not named; None for a gated region."""
    if colors is None:
        return []
    assert set(colors) <= set(COLOR_NAMES)
    return [colors.get(name, 0.0) for name in COLOR_NAMES]


def _view(
    top: Mapping[str, float] | None = None,
    bottom: Mapping[str, float] | None = None,
    *,
    top_left: float = 1.0,
    bottom_left: float = 1.0,
    clipped_at_bottom: bool = False,
) -> View:
    return View(
        color_names=COLOR_NAMES,
        top_shares=_shares(top),
        bottom_shares=_shares(bottom),
        top_left=top_left,
        bottom_left=bottom_left,
        clipped_at_bottom=clipped_at_bottom,
    )


# A blue top and black bottoms, each well over min_share, with a small
# runner-up.
_SEEN = _view(top={"blue": 0.8, "white": 0.1}, bottom={"black": 0.8, "grey": 0.1})
_VIEWS = (_SEEN, _SEEN, _SEEN)
_CLOUD = CloudAnswer(
    top_color="blue",
    bottom_color="black",
    top_type="t-shirt",
    bottom_type="jeans",
    glasses="yes",
    hat="no",
    beard="yes",
    hair_color="brown",
    hair_length="short",
    printed_text="no",
    sure=frozenset(FIELD_VALUES),
)


def _cloud(**changes: object) -> CloudAnswer:
    return dataclasses.replace(_CLOUD, **changes)


def _unsure(*names: str) -> CloudAnswer:
    return _cloud(sure=_CLOUD.sure - set(names))


# -- region_color -------------------------------------------------------------


def test_region_color_holds_from_min_share():
    assert region_color(_shares({"blue": 0.55, "white": 0.2}), 1.0) == "blue"
    assert region_color(_shares({"blue": 0.54, "white": 0.2}), 1.0) == UNKNOWN


def test_region_color_needs_the_runner_up_below_its_limit():
    # A two-colour pattern, such as stripes, is no colour.
    assert region_color(_shares({"blue": 0.6, "white": 0.24}), 1.0) == "blue"
    assert region_color(_shares({"blue": 0.6, "white": 0.25}), 1.0) == UNKNOWN


def test_region_color_needs_min_region_left():
    shares = _shares({"blue": 0.8})
    assert region_color(shares, 0.4) == "blue"
    assert region_color(shares, 0.39) == UNKNOWN


def test_region_color_local_only_needs_the_stricter_share():
    assert region_color(_shares({"blue": 0.6}), 1.0) == "blue"
    assert region_color(_shares({"blue": 0.6}), 1.0, local_only=True) == UNKNOWN
    assert region_color(_shares({"blue": 0.65}), 1.0, local_only=True) == "blue"


def test_region_color_of_a_gated_region_is_unknown():
    assert region_color([], 1.0) == UNKNOWN


def test_region_color_follows_the_thresholds():
    strict = AttributeThresholds(
        min_share=0.7, max_runner_up=0.1, local_only_min_share=0.8, min_region_left=0.9
    )
    shares = _shares({"blue": 0.65, "white": 0.05})
    assert region_color(shares, 1.0) == "blue"
    assert region_color(shares, 1.0, strict) == UNKNOWN
    assert region_color(_shares({"blue": 0.8, "white": 0.1}), 1.0, strict) == UNKNOWN
    assert region_color(_shares({"blue": 0.8}), 0.85, strict) == UNKNOWN


def test_region_color_refuses_shares_of_the_wrong_length():
    with pytest.raises(ValueError, match="11 colour shares"):
        region_color([0.9] + [0.0] * 10, 1.0)


# -- The colour names ---------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [
        # Two colours swapped: every share would name the wrong colour.
        ("white", "black", *COLOR_NAMES[2:]),
        COLOR_NAMES[:-1],
        (*COLOR_NAMES, "unknown"),
        tuple(name.upper() for name in COLOR_NAMES),
    ],
)
def test_a_view_with_other_colour_names_is_refused(names):
    with pytest.raises(ValueError, match="color_names"):
        View(
            color_names=names,
            top_shares=[],
            bottom_shares=[],
            top_left=1.0,
            bottom_left=1.0,
            clipped_at_bottom=False,
        )


def test_a_view_reads_the_names_as_any_sequence():
    view = dataclasses.replace(_SEEN, color_names=list(COLOR_NAMES))
    assert view.sees("top")


def test_a_view_refuses_shares_of_the_wrong_length():
    with pytest.raises(ValueError, match="11 bottom shares"):
        dataclasses.replace(_SEEN, bottom_shares=[0.9] + [0.0] * 10)


def test_a_view_sees_a_region_with_shares_and_a_bottom_not_clipped():
    assert _SEEN.sees("top") and _SEEN.sees("bottom")
    assert not _view(bottom={"black": 0.8}).sees("top")
    assert not _view(top={"blue": 0.8}, bottom={"black": 0.8}, clipped_at_bottom=True).sees(
        "bottom"
    )


# -- vote ---------------------------------------------------------------------


def test_vote_takes_a_colour_two_views_give_and_none_contradicts():
    gated = _view()
    unsure = _view(top={"blue": 0.5, "white": 0.4})
    assert vote([_SEEN, _SEEN, gated], "top") == "blue"
    assert vote([_SEEN, unsure, _SEEN], "top") == "blue"
    assert vote([_SEEN, _SEEN], "bottom") == "black"


def test_vote_needs_two_views():
    assert vote([_SEEN], "top") == UNKNOWN
    assert vote([_SEEN, _view()], "top") == UNKNOWN
    assert vote([], "top") == UNKNOWN


def test_vote_drops_a_colour_another_view_contradicts():
    red = _view(top={"red": 0.8})
    assert vote([_SEEN, _SEEN, red], "top") == UNKNOWN
    assert vote([_SEEN, red], "top") == UNKNOWN


def test_vote_skips_a_bottom_clipped_at_the_image_edge():
    clipped = dataclasses.replace(_SEEN, clipped_at_bottom=True)
    assert vote([_SEEN, clipped], "bottom") == UNKNOWN
    assert vote([_SEEN, clipped, _SEEN], "bottom") == "black"
    # The top is still seen.
    assert vote([clipped, clipped], "top") == "blue"


def test_vote_local_only_needs_every_view_at_the_stricter_share():
    faint = _view(top={"blue": 0.6})
    assert vote(_VIEWS, "top", local_only=True) == "blue"
    assert vote([_SEEN, _SEEN, faint], "top", local_only=True) == UNKNOWN
    assert vote([_SEEN, _SEEN, faint], "top") == "blue"
    # A view that does not see the region withholds the colour too.
    assert vote([_SEEN, _SEEN, _view()], "top", local_only=True) == UNKNOWN
    assert vote([_SEEN], "top", local_only=True) == UNKNOWN


# -- arbitrate ----------------------------------------------------------------


def test_arbitrate_keeps_what_the_views_and_a_sure_cloud_agree_on():
    result = arbitrate(_CLOUD, _VIEWS, face_seen=True)

    # The default allow-list is top_type, glasses and hat, and the hat is
    # "no", so the cloud alone gives two fields.
    assert list(result.kept) == ["top_color", "top_type", "glasses", "bottom_color"]
    assert result.kept == {
        "top_color": "blue",
        "top_type": "t-shirt",
        "glasses": "yes",
        "bottom_color": "black",
    }
    assert result.sources == {
        "top_color": CLOUD_AND_LOCAL,
        "top_type": CLOUD,
        "glasses": CLOUD,
        "bottom_color": CLOUD_AND_LOCAL,
    }
    assert result.dropped == {
        "bottom_type": NOT_ALLOWED,
        "hat": NEGATIVE,
        "beard": NOT_ALLOWED,
        "hair_color": NOT_ALLOWED,
        "hair_length": NOT_ALLOWED,
    }


@pytest.mark.parametrize(
    ("local", "said"), [("black", "grey"), ("red", "pink"), ("brown", "beige")]
)
def test_a_colour_the_cloud_disagrees_with_is_dropped(local, said):
    views = [_view(top={local: 0.8})] * 3

    result = arbitrate(_cloud(top_color=said), views, face_seen=True)

    assert result.dropped["top_color"] == CLOUD_DISAGREES
    assert "top_color" not in result.kept


def test_a_colour_the_cloud_is_unsure_of_is_dropped():
    result = arbitrate(_unsure("top_color"), _VIEWS, face_seen=True)

    assert result.dropped["top_color"] == CLOUD_UNSURE
    assert result.kept["bottom_color"] == "black"


def test_a_colour_the_cloud_calls_unknown_is_dropped():
    result = arbitrate(_cloud(top_color=UNKNOWN), _VIEWS, face_seen=True)

    assert result.dropped["top_color"] == CLOUD_UNKNOWN


def test_a_colour_whose_region_no_view_sees_is_dropped():
    no_legs = [_view(top={"blue": 0.8})] * 3
    clipped = [dataclasses.replace(_SEEN, clipped_at_bottom=True)] * 3

    for views in (no_legs, clipped):
        result = arbitrate(_CLOUD, views, face_seen=True)
        assert result.dropped["bottom_color"] == GATED
        assert result.kept["top_color"] == "blue"


def test_a_colour_needs_two_usable_views():
    result = arbitrate(_CLOUD, [_SEEN], face_seen=True)

    assert result.dropped["top_color"] == TOO_FEW_VIEWS
    assert result.dropped["bottom_color"] == TOO_FEW_VIEWS
    # A field from the cloud alone needs only its region seen.
    assert result.kept == {"top_type": "t-shirt", "glasses": "yes"}


def test_a_colour_is_never_said_on_the_clouds_word_alone():
    # The cloud is sure the top is blue, but the views see no one colour.
    striped = [_view(top={"blue": 0.5, "white": 0.45}, bottom={"black": 0.8})] * 3

    result = arbitrate(_CLOUD, striped, _ALLOW_ALL, face_seen=True)

    assert result.dropped["top_color"] == NO_LOCAL_AGREEMENT
    with pytest.raises(ValueError, match="never said on the cloud's word alone"):
        AttributeThresholds(allow_cloud_only=("top_type", "top_color"))


def test_without_the_cloud_a_colour_needs_the_stricter_share_in_every_view():
    faint = [_view(top={"blue": 0.6}, bottom={"black": 0.7})] * 3

    offline = arbitrate(None, faint, face_seen=True)
    online = arbitrate(_CLOUD, faint, face_seen=True)

    # 0.6 is enough when the cloud confirms the colour, and not without it.
    assert offline.dropped["top_color"] == NO_LOCAL_AGREEMENT
    assert online.kept["top_color"] == "blue"
    assert offline.kept == {"bottom_color": "black"}
    assert offline.sources == {"bottom_color": LOCAL}


def test_without_the_cloud_one_view_without_the_colour_withholds_it():
    views = [_SEEN, _SEEN, _view(top={"blue": 0.8})]

    result = arbitrate(None, views, face_seen=True)

    assert result.kept == {"top_color": "blue"}
    assert result.dropped["bottom_color"] == NO_LOCAL_AGREEMENT


def test_without_the_cloud_only_colours_are_said():
    result = arbitrate(None, _VIEWS, _ALLOW_ALL, face_seen=True)

    assert result.kept == {"top_color": "blue", "bottom_color": "black"}
    assert result.sources == {"top_color": LOCAL, "bottom_color": LOCAL}
    assert {name: result.dropped[name] for name in CLOUD_ONLY_ATTRIBUTES} == dict.fromkeys(
        CLOUD_ONLY_ATTRIBUTES, NO_CLOUD
    )


@pytest.mark.parametrize("printed_text", ["yes", UNKNOWN])
def test_printed_text_blocks_every_field_from_the_cloud_alone(printed_text):
    result = arbitrate(_cloud(printed_text=printed_text), _VIEWS, _ALLOW_ALL, face_seen=True)

    # The colours are checked against the pixels, so they stay.
    assert result.kept == {"top_color": "blue", "bottom_color": "black"}
    for name in ("top_type", "bottom_type", "glasses", "beard", "hair_color", "hair_length"):
        assert result.dropped[name] == PRINTED_TEXT


def test_only_allow_listed_fields_are_said_on_the_clouds_word():
    default = arbitrate(_CLOUD, _VIEWS, face_seen=True)
    wider = arbitrate(
        _CLOUD,
        _VIEWS,
        AttributeThresholds(allow_cloud_only=("bottom_type",), max_attributes=9),
        face_seen=True,
    )

    assert default.dropped["bottom_type"] == NOT_ALLOWED
    assert wider.kept["bottom_type"] == "jeans"
    assert wider.dropped["top_type"] == NOT_ALLOWED
    assert wider.dropped["glasses"] == NOT_ALLOWED


def test_a_field_the_cloud_is_unsure_of_is_dropped():
    result = arbitrate(_unsure("top_type", "glasses"), _VIEWS, face_seen=True)

    assert result.dropped["top_type"] == CLOUD_UNSURE
    assert result.dropped["glasses"] == CLOUD_UNSURE


def test_a_field_the_cloud_calls_unknown_is_dropped():
    result = arbitrate(_cloud(top_type=UNKNOWN), _VIEWS, face_seen=True)

    assert result.dropped["top_type"] == CLOUD_UNKNOWN


def test_a_garment_type_needs_its_region_seen():
    no_torso = [_view(bottom={"black": 0.8})] * 3
    clipped = [dataclasses.replace(_SEEN, clipped_at_bottom=True)] * 3

    assert arbitrate(_CLOUD, no_torso, face_seen=True).dropped["top_type"] == GATED
    assert arbitrate(_CLOUD, clipped, _ALLOW_ALL, face_seen=True).dropped["bottom_type"] == GATED


def test_the_head_fields_need_a_face_box():
    result = arbitrate(_cloud(hat="yes"), _VIEWS, _ALLOW_ALL, face_seen=False)

    for name in ("glasses", "hat", "beard", "hair_color", "hair_length"):
        assert result.dropped[name] == GATED
    assert result.kept["top_type"] == "t-shirt"


def test_no_view_keeps_nothing():
    result = arbitrate(_CLOUD, [], _ALLOW_ALL, face_seen=True)

    assert result.kept == {}
    assert set(result.dropped) == set(ATTRIBUTES)


@pytest.mark.parametrize("name", ["glasses", "hat", "beard"])
def test_glasses_a_hat_and_a_beard_are_never_said_as_a_negative(name):
    result = arbitrate(_cloud(**{name: "no"}), _VIEWS, _ALLOW_ALL, face_seen=True)

    assert result.dropped[name] == NEGATIVE


@pytest.mark.parametrize(("hat", "kept"), [("no", True), ("yes", False), (UNKNOWN, False)])
def test_hair_is_said_only_when_the_cloud_sees_no_hat(hat, kept):
    cfg = AttributeThresholds(allow_cloud_only=("hair_color", "hair_length"))

    result = arbitrate(_cloud(hat=hat), _VIEWS, cfg, face_seen=True)

    for name in ("hair_color", "hair_length"):
        if kept:
            assert name in result.kept
        else:
            assert result.dropped[name] == MAYBE_HAT


# A blue dress: the thighs show the dress, so the views see blue there too.
_DRESS_VIEWS = [_view(top={"blue": 0.8}, bottom={"blue": 0.8})] * 3


def test_a_dress_is_said_once_as_the_top_garment():
    cloud = _cloud(
        bottom_color="blue", top_type="dress", bottom_type="dress", glasses="no", beard="no"
    )

    result = arbitrate(cloud, _DRESS_VIEWS, face_seen=True)
    wider = arbitrate(cloud, _DRESS_VIEWS, _ALLOW_ALL, face_seen=True)

    assert result.kept == {"top_color": "blue", "top_type": "dress"}
    # "Blue bottoms" would name a garment the person is not wearing.
    assert result.dropped["bottom_color"] == IN_A_DRESS
    assert wider.dropped["bottom_color"] == IN_A_DRESS
    assert wider.dropped["bottom_type"] == IN_A_DRESS
    assert sentence("Ana", result.kept) == "Ana wears a blue dress."


@pytest.mark.parametrize(
    "dress",
    [
        {"top_type": "dress", "bottom_type": UNKNOWN},
        {"top_type": UNKNOWN, "bottom_type": "dress"},
        # A dress under a jacket.
        {"top_type": "jacket", "bottom_type": "dress"},
        # The cloud need not be sure of the dress to withhold the bottoms.
        {
            "top_type": "dress",
            "bottom_type": "dress",
            "sure": _CLOUD.sure - {"top_type", "bottom_type"},
        },
    ],
)
@pytest.mark.parametrize("cfg", [_THRESHOLDS, _ALLOW_ALL])
def test_any_dress_the_cloud_names_withholds_the_bottom_colour(dress, cfg):
    cloud = _cloud(bottom_color="blue", **dress)

    result = arbitrate(cloud, _DRESS_VIEWS, cfg, face_seen=True)
    text = sentence("Ana", result.kept)

    assert result.dropped["bottom_color"] == IN_A_DRESS
    assert "bottom_type" not in result.kept
    assert "bottoms" not in text
    assert not {"bottom_color", "bottom_type"} & {m.field for m in parse_mentions(text)}, text


def test_arbitrate_keeps_max_attributes_in_the_order_of_say():
    result = arbitrate(_CLOUD, _VIEWS, _ALLOW_ALL, face_seen=True)

    assert list(result.kept) == ["top_color", "top_type", "glasses", "hair_length"]
    assert result.dropped == {
        "bottom_color": OVER_MAX_ATTRIBUTES,
        "bottom_type": OVER_MAX_ATTRIBUTES,
        "hat": NEGATIVE,
        "beard": OVER_MAX_ATTRIBUTES,
        "hair_color": OVER_MAX_ATTRIBUTES,
    }


def test_arbitrate_follows_max_attributes_and_say():
    cfg = AttributeThresholds(
        allow_cloud_only=CLOUD_ONLY_ATTRIBUTES, max_attributes=2, say=("beard", "bottom_color")
    )

    result = arbitrate(_CLOUD, _VIEWS, cfg, face_seen=True)

    assert list(result.kept) == ["beard", "bottom_color"]
    for name in ("top_color", "top_type", "bottom_type", "glasses", "hair_color", "hair_length"):
        assert result.dropped[name] == NOT_IN_SAY


@pytest.mark.parametrize(
    "cloud",
    [
        None,
        _CLOUD,
        _cloud(printed_text="yes"),
        _unsure(*FIELD_VALUES),
        _cloud(hat="yes"),
        _cloud(top_type="dress", bottom_type="dress"),
    ],
)
@pytest.mark.parametrize("views", [(), (_SEEN,), _VIEWS, (_SEEN, _view())])
@pytest.mark.parametrize("cfg", [_THRESHOLDS, _ALLOW_ALL])
@pytest.mark.parametrize("face_seen", [True, False])
def test_arbitrate_gives_every_attribute_one_verdict(cloud, views, cfg, face_seen):
    result = arbitrate(cloud, views, cfg, face_seen=face_seen)

    assert not set(result.kept) & set(result.dropped)
    assert set(result.kept) | set(result.dropped) == set(ATTRIBUTES)
    assert list(result.dropped) == [name for name in ATTRIBUTES if name in result.dropped]
    assert set(result.dropped.values()) <= set(REASONS)
    assert set(result.sources) == set(result.kept)
    assert set(result.sources.values()) <= set(SOURCES)
    assert len(result.kept) <= cfg.max_attributes
    for name, value in result.kept.items():
        assert value in SAYABLE_VALUES[name]
    if result.kept:
        assert sentence("Sam", result.kept, cfg)


# -- The thresholds and the cloud answer --------------------------------------


def test_attribute_thresholds_defaults():
    assert (
        _THRESHOLDS.min_share,
        _THRESHOLDS.max_runner_up,
        _THRESHOLDS.local_only_min_share,
        _THRESHOLDS.min_region_left,
        _THRESHOLDS.max_attributes,
        _THRESHOLDS.allow_cloud_only,
    ) == (0.55, 0.25, 0.65, 0.4, 4, ("top_type", "glasses", "hat"))
    assert _THRESHOLDS.say == DEFAULT_SAY
    assert DEFAULT_SAY == (
        "top_color",
        "top_type",
        "glasses",
        "hat",
        "hair_length",
        "bottom_color",
        "bottom_type",
        "hair_color",
        "beard",
    )


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"min_share": 0.0}, "min_share"),
        ({"min_share": 1.1}, "min_share"),
        ({"max_runner_up": 0.0}, "max_runner_up"),
        ({"max_runner_up": 0.6}, "max_runner_up"),
        ({"local_only_min_share": 0.5}, "at least as strict"),
        ({"local_only_min_share": 1.1}, "local_only_min_share"),
        ({"min_region_left": -0.1}, "min_region_left"),
        ({"min_region_left": 1.1}, "min_region_left"),
        ({"max_attributes": 0}, "max_attributes"),
        ({"say": ("top_color", "shoes")}, r"say names \['shoes'\]"),
        ({"say": ("top_color", "top_color")}, "say names a field twice"),
        ({"allow_cloud_only": ("printed_text",)}, "not attributes"),
        ({"allow_cloud_only": ("bottom_color",)}, "never said on the cloud's word alone"),
    ],
)
def test_attribute_thresholds_refuse_bad_values(fields, match):
    with pytest.raises(ValueError, match=match):
        AttributeThresholds(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"top_color": "navy"},
        {"top_color": "blonde"},
        {"top_type": "jeans"},
        {"glasses": "maybe"},
        {"printed_text": "true"},
    ],
)
def test_a_cloud_answer_off_its_lists_is_refused(fields):
    name = next(iter(fields))
    with pytest.raises(ValueError, match=f"CloudAnswer.{name}"):
        CloudAnswer(**fields)


def test_a_cloud_answer_sure_of_an_unknown_field_is_refused():
    with pytest.raises(ValueError, match=r"sure names \['shoes'\]"):
        CloudAnswer(sure=frozenset({"top_color", "shoes"}))


def test_the_vocabulary():
    assert len(COLOR_NAMES) == 12
    assert UNKNOWN not in COLOR_NAMES
    assert FIELD_VALUES["top_color"] == (*COLOR_NAMES, UNKNOWN)
    assert list(FIELD_VALUES) == [*ATTRIBUTES, "printed_text"]
    assert SAYABLE_VALUES["glasses"] == ("yes",)
    assert "dress" in SAYABLE_VALUES["top_type"]
    assert "dress" not in SAYABLE_VALUES["bottom_type"]


# -- sentence -----------------------------------------------------------------


def _every_kept() -> Iterator[dict[str, str]]:
    """Every set of 1 to 4 fields, with every sayable value of each field in one of them.

    A bottom colour beside a dress is left out: arbitrate never keeps it,
    and sentence refuses it.
    """
    for size in range(1, 5):
        for names in itertools.combinations(ATTRIBUTES, size):
            longest = max(len(SAYABLE_VALUES[name]) for name in names)
            for i in range(longest):
                kept = {
                    name: SAYABLE_VALUES[name][i % len(SAYABLE_VALUES[name])] for name in names
                }
                if kept.get("top_type") != "dress" or "bottom_color" not in kept:
                    yield kept


def test_the_sentence_names_the_person_and_each_field():
    kept = {"top_color": "blue", "top_type": "t-shirt", "glasses": "yes", "hair_length": "short"}

    assert sentence("Sam", kept) == "Sam wears a blue t-shirt and glasses, and has short hair."


@pytest.mark.parametrize(
    ("kept", "text"),
    [
        ({"top_color": "blue"}, "Sam wears a blue top."),
        ({"top_color": "orange"}, "Sam wears an orange top."),
        ({"top_type": "hoodie"}, "Sam wears a hoodie."),
        ({"bottom_color": "black"}, "Sam wears black bottoms."),
        ({"bottom_color": "black", "bottom_type": "skirt"}, "Sam wears a black skirt."),
        ({"top_type": "dress", "bottom_type": "jeans"}, "Sam wears a dress and jeans."),
        ({"hat": "yes", "beard": "yes"}, "Sam wears a hat, and has a beard."),
        ({"hair_color": "blonde", "hair_length": "long"}, "Sam has long blonde hair."),
        (
            {"top_color": "white", "bottom_color": "blue", "glasses": "yes", "hat": "yes"},
            "Sam wears a white top, blue bottoms, glasses and a hat.",
        ),
    ],
)
def test_the_sentence_phrases(kept, text):
    assert sentence("Sam", kept) == text


def test_the_sentence_starts_a_placeholder_name_with_a_capital():
    assert sentence("the first guest", {"top_color": "blue"}) == "The first guest wears a blue top."


def test_no_sentence_asks_offers_a_seat_or_uses_a_pronoun():
    for kept in _every_kept():
        for name in ("Sam", "the first guest"):
            text = sentence(name, kept)
            assert classify_line(text) == Line(question=False, asks=(), seat=False), text
            assert PRONOUNS.search(text.casefold()) is None, text


def test_every_sentence_says_its_fields_and_no_more():
    # Read back per field, as a check scores it: "a blue t-shirt" is two.
    for kept in _every_kept():
        text = sentence("Sam", kept)
        mentions = parse_mentions(text)
        assert dict(mentions) == kept, text
        assert len(mentions) == len(kept) <= _THRESHOLDS.max_attributes, text


def test_the_sentence_says_at_most_max_attributes_fields():
    kept = {"top_color": "blue", "top_type": "t-shirt", "glasses": "yes"}

    assert sentence("Sam", kept, AttributeThresholds(max_attributes=3))
    with pytest.raises(ValueError, match="over max_attributes 2"):
        sentence("Sam", kept, AttributeThresholds(max_attributes=2))
    with pytest.raises(ValueError, match="over max_attributes 4"):
        sentence("Sam", {**kept, "hat": "yes", "beard": "yes"})


@pytest.mark.parametrize(
    "kept",
    [
        {"top_color": UNKNOWN},
        {"glasses": "no"},
        {"hat": UNKNOWN},
        {"bottom_type": "dress"},
        {"top_type": "dress", "bottom_color": "blue"},
        {"top_color": "blue", "top_type": "dress", "bottom_color": "blue", "bottom_type": "jeans"},
        {"printed_text": "no"},
        {"shoes": "red"},
    ],
)
def test_the_sentence_refuses_a_field_arbitrate_never_keeps(kept):
    with pytest.raises(ValueError, match="cannot say"):
        sentence("Sam", kept)


def test_the_sentence_needs_a_name_and_a_field():
    with pytest.raises(ValueError, match="name"):
        sentence("  ", {"top_color": "blue"})
    with pytest.raises(ValueError, match="at least one"):
        sentence("Sam", {})


def test_arbitrate_and_sentence_together():
    result = arbitrate(_CLOUD, _VIEWS, face_seen=True)

    assert sentence("Sam", result.kept) == (
        "Sam wears a blue t-shirt, black bottoms and glasses."
    )


# -- parse_mentions -----------------------------------------------------------


def test_parse_mentions_scores_per_field():
    truth = {"top_color": "blue", "top_type": "shirt"}

    mentions = parse_mentions("Sam wears a blue t-shirt.")
    correct = [m for m in mentions if truth.get(m.field) == m.value]
    wrong = [m for m in mentions if truth.get(m.field) != m.value]

    assert mentions == [Mention("top_color", "blue"), Mention("top_type", "t-shirt")]
    assert (len(correct), len(wrong)) == (1, 1)


def test_parse_mentions_reads_a_reworded_line():
    text = "Sam is wearing a light blue T-shirt, black jeans and glasses, and has long blonde hair."

    assert parse_mentions(text) == [
        Mention("top_color", "blue"),
        Mention("top_type", "t-shirt"),
        Mention("bottom_color", "black"),
        Mention("bottom_type", "jeans"),
        Mention("glasses", "yes"),
        Mention("hair_length", "long"),
        Mention("hair_color", "blonde"),
    ]


@pytest.mark.parametrize(
    ("text", "mentions"),
    [
        ("Sam has no glasses and wears a hat.", [("glasses", "no"), ("hat", "yes")]),
        ("Sam doesn't wear glasses or a hat.", [("glasses", "no"), ("hat", "no")]),
        ("Sam is without a beard, with glasses.", [("beard", "no"), ("glasses", "yes")]),
        # A negated garment says nothing about the person.
        ("Sam isn't wearing a red jacket.", []),
        # An "and" between two colours of one noun stays inside the negation.
        ("Sam isn't wearing a red and white jacket.", []),
        ("Sam doesn't have black and grey hair.", []),
        ("Sam has no red and white hat.", [("hat", "no")]),
        # An "and" after a colour that reaches no noun still ends it.
        ("Sam isn't in blue and wears a white top.", [("top_color", "white")]),
    ],
)
def test_parse_mentions_reads_negatives(text, mentions):
    assert parse_mentions(text) == [Mention(*m) for m in mentions]


@pytest.mark.parametrize(
    ("text", "mentions"),
    [
        (
            "Sam wears a gray tee and grey pants.",
            [
                ("top_color", "grey"),
                ("top_type", "t-shirt"),
                ("bottom_color", "grey"),
                ("bottom_type", "trousers"),
            ],
        ),
        ("Sam wears a tee shirt and a cap.", [("top_type", "t-shirt"), ("hat", "yes")]),
        ("Sam has blond hair.", [("hair_color", "blonde")]),
        ("Sam’s jumper is warm.", [("top_type", "sweater")]),
        # A colour the hair vocabulary lacks is still read, so a check can
        # count it as wrong.
        ("Sam has red hair.", [("hair_color", "red")]),
    ],
)
def test_parse_mentions_reads_other_spellings(text, mentions):
    assert parse_mentions(text) == [Mention(*m) for m in mentions]


def test_a_colour_belongs_only_to_the_garment_right_after_it():
    assert parse_mentions("Sam's favourite drink is orange juice.") == []
    assert parse_mentions("Sam likes orange juice, and wears a blue top.") == [
        Mention("top_color", "blue")
    ]
    assert parse_mentions("Sam wears a blue and white shirt.") == [
        Mention("top_color", "blue"),
        Mention("top_color", "white"),
        Mention("top_type", "shirt"),
    ]


def test_parse_mentions_gives_each_field_and_value_once():
    assert parse_mentions("Sam wears a blue top. Yes, a blue top.") == [
        Mention("top_color", "blue")
    ]


def test_a_line_with_no_looks_mentions_nothing():
    assert parse_mentions("Sam is waiting in the living room.") == []
    assert parse_mentions("Nice to meet you, Ana. Please follow me.") == []


# -- The module ---------------------------------------------------------------


def test_the_module_imports_no_ros():
    # A fresh interpreter, since this suite has long imported rclpy, so that
    # the HRI check, which imports no ROS, can read lines with it. -I drops
    # PYTHONPATH, where the ROS workspaces' packages are.
    src = Path(attributes.__file__).parents[2]
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(src)!r}); "
        "import walkie_agent.perception.attributes; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'rclpy', 'rai', "
        "'walkie_vision_msgs', 'langchain_core'}))"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
