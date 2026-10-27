"""Which of a person's looks to say, from pixel colour shares and a cloud VLM's answers.

describe_person, not built yet (perception.md, question 4), is to tell the
second guest of the HRI task how the first looks. The score sheet gives +20
per correct attribute and -20 per incorrect one, up to 4 of them and never
below zero (hri.md, The task). An attribute left unsaid costs nothing, so
every rule here withholds when the evidence does not agree:
  - a colour is said only when the pixel shares of at least MIN_VIEWS views
    of the person agree on it and the cloud gives the same colour and is
    sure of it. Without a cloud answer, every view must agree at the
    stricter local_only_min_share. A colour is never said on the cloud's
    word alone;
  - a garment type, glasses, a hat, a beard or hair comes from the cloud
    alone, so it is said only when the cloud is sure, printed_text is "no",
    the field is on allow_cloud_only and its region was seen. Letters or a
    logo on the person or behind them could steer the cloud's answer, which
    is why printed_text "no" is required;
  - glasses, a hat and a beard are said only as "yes", never as a
    negative, and hair only when the cloud says there is no hat;
  - a dress is said once, as the top garment. When the cloud names a
    dress as either garment, sure of it or not, the bottom's colour is not
    said either: the thighs show the dress, and "blue bottoms" would name
    a garment the person is not wearing.

The unit is the field: "a blue t-shirt" says top_color and top_type, two
fields, and sentence says at most max_attributes of them. The rulebook does
not say whether that is one attribute or two, so counting fields is the
safe reading: past the 4th, a field can only lose points. sentence builds
the line from the kept fields and names the person, with no pronoun, no
question and no word for a seat. parse_mentions reads the fields back from
a spoken line, so a check can score each field against the truth.

walkie-vision computes the colour shares on the full image, for the torso
(top) and the thighs (bottom), and gates a region whose keypoints it did not
see. View holds one crop's shares as plain values, so this module needs no
walkie_vision_msgs, and CloudAnswer holds the describer's enums. Pure logic:
no ROS and no image decoding. The thresholds are code defaults, not yet
tuned on the robot.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

UNKNOWN = "unknown"
YES = "yes"
NO = "no"

# The colour vocabulary, in the order of walkie-vision's colour shares. A
# crop whose color_names differ is refused, since its shares would name the
# wrong colours.
COLOR_NAMES: tuple[str, ...] = (
    "black",
    "white",
    "grey",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "beige",
)
TOP_TYPES: tuple[str, ...] = ("t-shirt", "shirt", "sweater", "hoodie", "jacket", "coat", "dress")
BOTTOM_TYPES: tuple[str, ...] = ("trousers", "jeans", "shorts", "skirt", "dress")
HAIR_COLORS: tuple[str, ...] = ("black", "brown", "blonde", "grey")
HAIR_LENGTHS: tuple[str, ...] = ("short", "long")
_YES_NO = (YES, NO, UNKNOWN)

# Each field of the describer's answer and the values it may take, in the
# order of its schema.
FIELD_VALUES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "top_color": (*COLOR_NAMES, UNKNOWN),
        "bottom_color": (*COLOR_NAMES, UNKNOWN),
        "top_type": (*TOP_TYPES, UNKNOWN),
        "bottom_type": (*BOTTOM_TYPES, UNKNOWN),
        "glasses": _YES_NO,
        "hat": _YES_NO,
        "beard": _YES_NO,
        "hair_color": (*HAIR_COLORS, UNKNOWN),
        "hair_length": (*HAIR_LENGTHS, UNKNOWN),
        "printed_text": _YES_NO,
    }
)
# The fields that can be said: all but printed_text, which only gates.
ATTRIBUTES: tuple[str, ...] = tuple(name for name in FIELD_VALUES if name != "printed_text")
# The colour fields and the region whose shares each one reads.
COLOR_FIELDS: Mapping[str, str] = MappingProxyType({"top_color": "top", "bottom_color": "bottom"})
# The fields that only the cloud can give.
CLOUD_ONLY_ATTRIBUTES: tuple[str, ...] = tuple(
    name for name in ATTRIBUTES if name not in COLOR_FIELDS
)
REGIONS: tuple[str, ...] = ("top", "bottom")
# The values each field may be said with: no unknown and no negative. A
# dress is said once, as the top garment, so bottom_type never says it.
SAYABLE_VALUES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        name: tuple(
            value
            for value in FIELD_VALUES[name]
            if value not in (UNKNOWN, NO) and (name, value) != ("bottom_type", "dress")
        )
        for name in ATTRIBUTES
    }
)

# The order sentence takes the kept fields in, and the fields the cloud
# alone may give until a labelled photo set measures its precision per field.
DEFAULT_SAY: tuple[str, ...] = (
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
DEFAULT_ALLOW_CLOUD_ONLY: tuple[str, ...] = ("top_type", "glasses", "hat")

# A colour needs at least this many usable views, and below it there is no
# local result at all.
MIN_VIEWS = 2

# Where a kept field's value came from.
CLOUD_AND_LOCAL = "cloud+local"
LOCAL = "local"
CLOUD = "cloud"
SOURCES = (CLOUD_AND_LOCAL, LOCAL, CLOUD)

# Why arbitrate dropped a field. The first check that fails names the
# field's reason; arbitrate's docstring gives the order.
TOO_FEW_VIEWS = "too_few_views"
GATED = "gated"
NO_LOCAL_AGREEMENT = "no_local_agreement"
NO_CLOUD = "no_cloud"
CLOUD_UNKNOWN = "cloud_unknown"
CLOUD_DISAGREES = "cloud_disagrees"
CLOUD_UNSURE = "cloud_unsure"
NEGATIVE = "negative"
NOT_ALLOWED = "not_allowed"
PRINTED_TEXT = "printed_text"
MAYBE_HAT = "maybe_hat"
IN_A_DRESS = "in_a_dress"
NOT_IN_SAY = "not_in_say"
OVER_MAX_ATTRIBUTES = "over_max_attributes"
REASONS = (
    TOO_FEW_VIEWS,
    GATED,
    NO_LOCAL_AGREEMENT,
    NO_CLOUD,
    CLOUD_UNKNOWN,
    CLOUD_DISAGREES,
    CLOUD_UNSURE,
    NEGATIVE,
    NOT_ALLOWED,
    PRINTED_TEXT,
    MAYBE_HAT,
    IN_A_DRESS,
    NOT_IN_SAY,
    OVER_MAX_ATTRIBUTES,
)

# The garment types and the region each one is seen in. The other
# cloud-only fields are on the head, which the person's face box shows.
_TYPE_REGIONS = MappingProxyType({"top_type": "top", "bottom_type": "bottom"})
# Said only as "yes".
_YES_ONLY = frozenset({"glasses", "hat", "beard"})
# Hidden or changed under a hat.
_HAIR = frozenset({"hair_color", "hair_length"})


@dataclass(frozen=True)
class AttributeThresholds:
    """What a colour share must reach, and which fields may be said and how many.

    New, not from walkie-agent-v2. The defaults were chosen before any photo
    of a real guest was labelled, and are to tune on the robot. They are
    code defaults: config.toml cannot set them yet.
    """

    # A region's colour holds when its share of the region's pixels is at
    # least this.
    min_share: float = 0.55
    # ... and the next colour's share is below this, so a two-colour
    # pattern, such as stripes, is no colour.
    max_runner_up: float = 0.25
    # Without a cloud answer, every view must reach this share instead:
    # nothing else checks the pixels then.
    local_only_min_share: float = 0.65
    # A region with less than this share of its pixels left after clipping
    # to the person's box and masking other people gives no colour.
    min_region_left: float = 0.4
    # The most fields one sentence says. The score sheet counts up to 4
    # correct and up to 4 incorrect attributes, so a 5th can only lose.
    max_attributes: int = 4
    # The fields that may be said, in the order they are taken: the first
    # max_attributes kept ones are said. A field left out is never said.
    say: tuple[str, ...] = DEFAULT_SAY
    # The fields other than colours that may be said on the cloud's word
    # alone. A colour is never allowed here.
    allow_cloud_only: tuple[str, ...] = DEFAULT_ALLOW_CLOUD_ONLY

    def __post_init__(self) -> None:
        if not 0 < self.min_share <= 1:
            raise ValueError("AttributeThresholds.min_share must lie in (0, 1].")
        # A runner-up limit above min_share would let a tie between two
        # colours pass.
        if not 0 < self.max_runner_up <= self.min_share:
            raise ValueError("AttributeThresholds.max_runner_up must lie in (0, min_share].")
        if not self.min_share <= self.local_only_min_share <= 1:
            raise ValueError(
                "AttributeThresholds.local_only_min_share must lie in [min_share, 1]: "
                "without the cloud, the share must be at least as strict."
            )
        if not 0 <= self.min_region_left <= 1:
            raise ValueError("AttributeThresholds.min_region_left must lie in [0, 1].")
        if self.max_attributes < 1:
            raise ValueError("AttributeThresholds.max_attributes must be at least 1.")
        for name in ("say", "allow_cloud_only"):
            fields = getattr(self, name)
            unknown = [f for f in fields if f not in ATTRIBUTES]
            if unknown:
                raise ValueError(
                    f"AttributeThresholds.{name} names {unknown}, which are not attributes; "
                    f"it takes {', '.join(ATTRIBUTES)}."
                )
            if len(set(fields)) != len(fields):
                raise ValueError(f"AttributeThresholds.{name} names a field twice.")
        colors = [f for f in self.allow_cloud_only if f in COLOR_FIELDS]
        if colors:
            raise ValueError(
                f"AttributeThresholds.allow_cloud_only names {colors}: a colour is never said "
                "on the cloud's word alone."
            )


DEFAULT_ATTRIBUTE_THRESHOLDS = AttributeThresholds()


@dataclass(frozen=True)
class View:
    """One usable crop of the person: the colour shares of its two regions.

    The values come from one walkie-vision crop response, copied out as
    plain values. A view is usable when walkie-vision served it and it is
    still the person being described; the caller passes only those.
    """

    # The colours the shares follow, which must be COLOR_NAMES in order.
    color_names: Sequence[str]
    # The torso's share of each colour, in color_names order. Empty when
    # walkie-vision gated the region: one of its four keypoints scored too
    # low to place it.
    top_shares: Sequence[float]
    # The thighs' shares, in the same way.
    bottom_shares: Sequence[float]
    # The share of each region's pixels left after clipping to the person's
    # box and masking other people.
    top_left: float
    bottom_left: float
    # True when the person's box touches the image's bottom edge. The legs
    # may then be cut off, so the bottom region does not count as seen.
    clipped_at_bottom: bool

    def __post_init__(self) -> None:
        if tuple(self.color_names) != COLOR_NAMES:
            raise ValueError(
                f"The crop's color_names {list(self.color_names)} differ from the agent's "
                f"{list(COLOR_NAMES)}; walkie-vision and walkie_agent.perception.attributes "
                "must list the same colours in the same order."
            )
        for region in REGIONS:
            count = len(self.shares(region))
            if count not in (0, len(COLOR_NAMES)):
                raise ValueError(
                    f"The crop has {count} {region} shares; expected {len(COLOR_NAMES)}, "
                    "or none for a gated region."
                )

    def shares(self, region: str) -> Sequence[float]:
        return {"top": self.top_shares, "bottom": self.bottom_shares}[region]

    def left(self, region: str) -> float:
        return {"top": self.top_left, "bottom": self.bottom_left}[region]

    def sees(self, region: str) -> bool:
        """True when the region passed its gate: it has shares, and the bottom is not clipped."""
        if region == "bottom" and self.clipped_at_bottom:
            return False
        return len(self.shares(region)) > 0


@dataclass(frozen=True)
class CloudAnswer:
    """The cloud describer's answer: one value per field, and the fields it is sure of.

    Each value must be one of its field's FIELD_VALUES, so a caller that
    builds a CloudAnswer from the describer's JSON can treat a ValueError
    as invalid JSON.
    """

    top_color: str = UNKNOWN
    bottom_color: str = UNKNOWN
    top_type: str = UNKNOWN
    bottom_type: str = UNKNOWN
    glasses: str = UNKNOWN
    hat: str = UNKNOWN
    beard: str = UNKNOWN
    hair_color: str = UNKNOWN
    hair_length: str = UNKNOWN
    printed_text: str = UNKNOWN
    # The fields the cloud says it is certain of.
    sure: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name, values in FIELD_VALUES.items():
            value = getattr(self, name)
            if value not in values:
                raise ValueError(
                    f"CloudAnswer.{name} must be one of {', '.join(values)}, not {value!r}."
                )
        unknown = sorted(set(self.sure) - set(FIELD_VALUES))
        if unknown:
            raise ValueError(f"CloudAnswer.sure names {unknown}, which are not fields.")


@dataclass(frozen=True)
class Arbitration:
    """What arbitrate keeps and drops, per field."""

    # The fields to say and their values, in the order of `say`, at most
    # max_attributes of them.
    kept: Mapping[str, str]
    # Where each kept field's value came from, one of SOURCES.
    sources: Mapping[str, str]
    # Every other attribute and why it is not said, one of REASONS, in the
    # order of ATTRIBUTES.
    dropped: Mapping[str, str]


def region_color(
    shares: Sequence[float],
    left: float,
    cfg: AttributeThresholds = DEFAULT_ATTRIBUTE_THRESHOLDS,
    *,
    local_only: bool = False,
) -> str:
    """The region's colour, one of COLOR_NAMES, or UNKNOWN.

    `shares` are one region's colour shares in COLOR_NAMES order, and `left`
    the share of its pixels left after clipping and masking. A colour holds
    when its share is at least min_share, or local_only_min_share with
    `local_only`, the next colour's share is below max_runner_up, and at
    least min_region_left of the region is left. Empty shares, a gated
    region, give UNKNOWN.
    """
    if not shares:
        return UNKNOWN
    if len(shares) != len(COLOR_NAMES):
        raise ValueError(f"Got {len(shares)} colour shares; expected {len(COLOR_NAMES)}.")
    if left < cfg.min_region_left:
        return UNKNOWN
    order = sorted(range(len(shares)), key=lambda i: shares[i], reverse=True)
    best, runner_up = shares[order[0]], shares[order[1]]
    floor = cfg.local_only_min_share if local_only else cfg.min_share
    if best >= floor and runner_up < cfg.max_runner_up:
        return COLOR_NAMES[order[0]]
    return UNKNOWN


def vote(
    views: Sequence[View],
    region: str,
    cfg: AttributeThresholds = DEFAULT_ATTRIBUTE_THRESHOLDS,
    *,
    local_only: bool = False,
) -> str:
    """The colour the views agree on for the region, or UNKNOWN.

    Each view gives region_color's colour, or UNKNOWN where it does not see
    the region. At least MIN_VIEWS views must give the colour and none may
    give another one. With `local_only`, for a colour no cloud answer
    confirms, every view must give it, at local_only_min_share, so one view
    that saw no colour or did not see the region is enough to withhold it.
    """
    colors = [
        region_color(view.shares(region), view.left(region), cfg, local_only=local_only)
        if view.sees(region)
        else UNKNOWN
        for view in views
    ]
    named = set(colors) - {UNKNOWN}
    if len(named) != 1:
        return UNKNOWN
    (color,) = named
    if local_only:
        agree = len(colors) >= MIN_VIEWS and all(c == color for c in colors)
    else:
        agree = colors.count(color) >= MIN_VIEWS
    return color if agree else UNKNOWN


def arbitrate(
    cloud: CloudAnswer | None,
    views: Sequence[View],
    cfg: AttributeThresholds = DEFAULT_ATTRIBUTE_THRESHOLDS,
    *,
    face_seen: bool,
) -> Arbitration:
    """Which attributes to say, from the cloud's answer and the usable views.

    `cloud` is None when there is no cloud answer: no describer, a timeout,
    an error or invalid JSON. `views` are the usable views only. `face_seen`
    is True when the person's handle has a face box, which shows the head.

    A colour is checked in this order: at least MIN_VIEWS views
    (TOO_FEW_VIEWS), a view that sees its region (GATED), the views' vote
    (NO_LOCAL_AGREEMENT), local_only without a cloud answer; then, with a
    cloud answer, the cloud's colour is known (CLOUD_UNKNOWN), the same
    (CLOUD_DISAGREES) and sure (CLOUD_UNSURE). Black against grey or red
    against pink is a disagreement like any other. bottom_color then needs
    the cloud's top_type and bottom_type both to be other than dress
    (IN_A_DRESS), whether or not the cloud is sure of them.

    Any other field is checked in this order: a cloud answer (NO_CLOUD), a
    known value (CLOUD_UNKNOWN), "yes" for glasses, a hat or a beard
    (NEGATIVE), on allow_cloud_only (NOT_ALLOWED), sure (CLOUD_UNSURE),
    printed_text "no" (PRINTED_TEXT), and its region seen (GATED): for a
    garment type, a view that sees its region; on the head, a face box and
    at least one view. Hair then needs the cloud's hat to be "no"
    (MAYBE_HAT), and a bottom_type of dress is never said (IN_A_DRESS),
    since only top_type says a dress.

    The fields that pass are kept in the order of `say` (NOT_IN_SAY for
    one not on it) until max_attributes are kept (OVER_MAX_ATTRIBUTES).
    """
    passed: dict[str, tuple[str, str]] = {}
    reasons: dict[str, str] = {}
    for name in ATTRIBUTES:
        if name in COLOR_FIELDS:
            value, note = _color(name, cloud, views, cfg)
        else:
            value, note = _cloud_only(name, cloud, views, cfg, face_seen=face_seen)
        if value is None:
            reasons[name] = note
        else:
            passed[name] = (value, note)
    kept: dict[str, str] = {}
    sources: dict[str, str] = {}
    for name in cfg.say:
        if name not in passed:
            continue
        if len(kept) < cfg.max_attributes:
            kept[name], sources[name] = passed[name]
        else:
            reasons[name] = OVER_MAX_ATTRIBUTES
    for name in passed:
        if name not in cfg.say:
            reasons[name] = NOT_IN_SAY
    dropped = {name: reasons[name] for name in ATTRIBUTES if name in reasons}
    return Arbitration(kept=kept, sources=sources, dropped=dropped)


def _color(
    name: str, cloud: CloudAnswer | None, views: Sequence[View], cfg: AttributeThresholds
) -> tuple[str | None, str]:
    """The colour to say and its source, or None and why it is dropped."""
    region = COLOR_FIELDS[name]
    if len(views) < MIN_VIEWS:
        return None, TOO_FEW_VIEWS
    if not any(view.sees(region) for view in views):
        return None, GATED
    local = vote(views, region, cfg, local_only=cloud is None)
    if local == UNKNOWN:
        return None, NO_LOCAL_AGREEMENT
    if cloud is None:
        return local, LOCAL
    said = getattr(cloud, name)
    if said == UNKNOWN:
        return None, CLOUD_UNKNOWN
    if said != local:
        return None, CLOUD_DISAGREES
    if name not in cloud.sure:
        return None, CLOUD_UNSURE
    if name == "bottom_color" and "dress" in (cloud.top_type, cloud.bottom_type):
        return None, IN_A_DRESS
    return local, CLOUD_AND_LOCAL


def _cloud_only(
    name: str,
    cloud: CloudAnswer | None,
    views: Sequence[View],
    cfg: AttributeThresholds,
    *,
    face_seen: bool,
) -> tuple[str | None, str]:
    """The cloud's value to say and its source, or None and why it is dropped."""
    if cloud is None:
        return None, NO_CLOUD
    value = getattr(cloud, name)
    if value == UNKNOWN:
        return None, CLOUD_UNKNOWN
    if name in _YES_ONLY and value != YES:
        return None, NEGATIVE
    if name not in cfg.allow_cloud_only:
        return None, NOT_ALLOWED
    if name not in cloud.sure:
        return None, CLOUD_UNSURE
    if cloud.printed_text != NO:
        return None, PRINTED_TEXT
    region = _TYPE_REGIONS.get(name)
    if region is not None:
        seen = any(view.sees(region) for view in views)
    else:
        # The cloud's crop is one of the views, so with none there is no
        # head to vouch for.
        seen = face_seen and len(views) > 0
    if not seen:
        return None, GATED
    if name in _HAIR and cloud.hat != NO:
        return None, MAYBE_HAT
    if name == "bottom_type" and value == "dress":
        return None, IN_A_DRESS
    return value, CLOUD


# Garment nouns that take no article: "black jeans".
_PLURAL_NOUNS = frozenset({"trousers", "jeans", "shorts", "bottoms"})
# The nouns for a colour whose garment type is not kept: "a blue top" and
# "black bottoms". Neither is a garment type, so neither says one.
_TOP_NOUN = "top"
_BOTTOM_NOUN = "bottoms"


def sentence(
    name: str,
    kept: Mapping[str, str],
    cfg: AttributeThresholds = DEFAULT_ATTRIBUTE_THRESHOLDS,
) -> str:
    """The line that says how the person looks, built from the kept fields.

    For example "Sam wears a blue t-shirt and glasses, and has short hair."
    `name` is the person's name, or a placeholder such as "the first guest",
    and `kept` is arbitrate's kept fields. Each field is said once. The line
    names the person and uses no pronoun. It holds no question, no request
    and no word for a seat, so it is never taken for a question or a seat
    offer. A field arbitrate never keeps, such as an unknown value, a
    negative, a bottom colour beside a dress, or more than max_attributes
    fields, is refused.
    """
    subject = name.strip()
    if not subject:
        raise ValueError("sentence needs the person's name.")
    if not kept:
        raise ValueError("sentence needs at least one kept field.")
    if len(kept) > cfg.max_attributes:
        raise ValueError(
            f"sentence got {len(kept)} fields, over max_attributes {cfg.max_attributes}."
        )
    for field_name, value in kept.items():
        if value not in SAYABLE_VALUES.get(field_name, ()):
            raise ValueError(f"sentence cannot say {field_name} {value!r}.")
    if kept.get("top_type") == "dress" and "bottom_color" in kept:
        raise ValueError(
            "sentence cannot say bottom_color beside a dress: the thighs show the dress."
        )
    wear = [
        phrase
        for phrase in (
            _garment(kept.get("top_color"), kept.get("top_type"), _TOP_NOUN),
            _garment(kept.get("bottom_color"), kept.get("bottom_type"), _BOTTOM_NOUN),
        )
        if phrase is not None
    ]
    if "glasses" in kept:
        wear.append("glasses")
    if "hat" in kept:
        wear.append("a hat")
    has = []
    hair = [kept[field_name] for field_name in ("hair_length", "hair_color") if field_name in kept]
    if hair:
        has.append(" ".join([*hair, "hair"]))
    if "beard" in kept:
        has.append("a beard")
    verbs = []
    if wear:
        verbs.append(f"wears {_join(wear)}")
    if has:
        verbs.append(f"has {_join(has)}")
    return f"{subject[0].upper()}{subject[1:]} {', and '.join(verbs)}."


def _garment(color: str | None, kind: str | None, noun: str) -> str | None:
    """A garment's phrase, "a blue t-shirt", "a blue top" or "a t-shirt"; None for neither field."""
    if color is None and kind is None:
        return None
    words = " ".join(word for word in (color, kind or noun) if word is not None)
    if (kind or noun) in _PLURAL_NOUNS:
        return words
    return f"{'an' if words[0] in 'aeiou' else 'a'} {words}"


def _join(phrases: Sequence[str]) -> str:
    """The phrases as a list in a sentence: "a", "a and b" or "a, b and c"."""
    if len(phrases) == 1:
        return phrases[0]
    return f"{', '.join(phrases[:-1])} and {phrases[-1]}"


class Mention(NamedTuple):
    """One attribute a line says: a field and the value it gives."""

    field: str
    value: str


# "T-shirt", "t shirt", "tshirt", "tee shirt" and "tee", as one token.
_T_SHIRT = re.compile(r"\b(?:t|tee)[\s-]?shirts?\b|\btees?\b")
# Words, with an apostrophe inside ("isn't", "Sam's"), and punctuation.
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,;:!?]")
# The colour words, with their other spellings.
_COLOR_WORDS = MappingProxyType(
    {
        **{color: color for color in COLOR_NAMES},
        "gray": "grey",
        "blond": "blonde",
        "blonde": "blonde",
    }
)
_LENGTH_WORDS = frozenset(HAIR_LENGTHS)
# Each garment word's region, and its type, or None for a word that names a
# region only, such as "top".
_GARMENT_WORDS = MappingProxyType(
    {
        "tshirt": ("top", "t-shirt"),
        "shirt": ("top", "shirt"),
        "sweater": ("top", "sweater"),
        "jumper": ("top", "sweater"),
        "pullover": ("top", "sweater"),
        "hoodie": ("top", "hoodie"),
        "hoody": ("top", "hoodie"),
        "jacket": ("top", "jacket"),
        "coat": ("top", "coat"),
        "dress": ("top", "dress"),
        "top": ("top", None),
        "tops": ("top", None),
        "trousers": ("bottom", "trousers"),
        "pants": ("bottom", "trousers"),
        "jeans": ("bottom", "jeans"),
        "shorts": ("bottom", "shorts"),
        "skirt": ("bottom", "skirt"),
        "bottoms": ("bottom", None),
    }
)
_ACCESSORY_WORDS = MappingProxyType(
    {
        "glasses": "glasses",
        "spectacles": "glasses",
        "sunglasses": "glasses",
        "hat": "hat",
        "cap": "hat",
        "beanie": "hat",
        "beard": "beard",
    }
)
# Words that may stand between a colour and its noun, as in "a blue and
# white shirt" or "dark blue jeans". Any other word drops the colour, so
# "orange juice" names no garment's colour.
_BRIDGE_WORDS = frozenset({"and", "or", "light", "dark", "bright", "pale"})
_NEGATIONS = frozenset({"no", "not", "without", "never", "neither", "nor"})
# Words that end a negation: "no glasses and a hat" says the hat.
_NEGATION_ENDS = frozenset({"and", "but", "with", ".", ",", ";", ":", "!", "?"})


def parse_mentions(text: str) -> list[Mention]:
    """The attributes a line says, per field, in the order they come.

    "a blue t-shirt" gives top_color blue and top_type t-shirt, two fields.
    A colour belongs to the garment or the hair right after it, and a
    length to the hair. Glasses, a hat and a beard give "yes", or "no"
    after a negation such as "no", "not", "without" or "doesn't", which
    lasts until "and", "but", "with" or punctuation. An "and" that joins
    colours or lengths before one noun, as in "isn't wearing a red and
    white jacket", stays inside the negation. A negated garment or hair
    gives nothing. Each field and value is given once. sentence's lines
    read back to the fields they were built from.
    """
    lower = _T_SHIRT.sub(" tshirt ", text.casefold().replace("’", "'"))
    mentions: list[Mention] = []
    colors: list[str] = []
    lengths: list[str] = []
    negated = False
    # Set by an "and" after a colour or a length, as in "red and white": it
    # ends the negation only if the colours and lengths reach no noun.
    ending = False

    def add(field_name: str, value: str) -> None:
        mention = Mention(field_name, value)
        if mention not in mentions:
            mentions.append(mention)

    for token in _TOKEN.findall(lower):
        if token in _COLOR_WORDS:
            colors.append(_COLOR_WORDS[token])
            continue
        if token in _LENGTH_WORDS:
            lengths.append(token)
            continue
        if token in _NEGATION_ENDS:
            if token in _BRIDGE_WORDS and (colors or lengths):
                ending = True
            else:
                negated = False
        if token in _BRIDGE_WORDS:
            continue
        noun = token in _GARMENT_WORDS or token == "hair" or token in _ACCESSORY_WORDS
        if ending and not noun:
            negated = False
        ending = False
        if token in _GARMENT_WORDS:
            region, kind = _GARMENT_WORDS[token]
            if not negated:
                for color in colors:
                    add(f"{region}_color", color)
                if kind is not None:
                    add(f"{region}_type", kind)
        elif token == "hair":
            if not negated:
                for length in lengths:
                    add("hair_length", length)
                for color in colors:
                    add("hair_color", color)
        elif token in _ACCESSORY_WORDS:
            add(_ACCESSORY_WORDS[token], NO if negated else YES)
        elif token in _NEGATIONS or token.endswith("n't"):
            negated = True
        colors, lengths = [], []
    return mentions
