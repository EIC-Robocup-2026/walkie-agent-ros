"""The words find_objects uses for walkie-vision's object labels. No ROS.

A detector names an object by its label, one of the text prompts it was given
(ObjectDetectionArray.classes). find_objects reports that label as the object's
name, and a category from CATEGORIES: the arena's categories for the GPSR objects,
the same as FakeRobot's, and "furniture" for the places people sit and things stand
on. A label the table lacks has the category OTHER.

A query matches an object's name, its category or the category's plural, ignoring
case and spaces at either end, as the Vision protocol says. A detector looks only
for its own classes, so `looks_for` tells a query it could never match from one
that matched nothing in view.
"""

from collections.abc import Iterable

# The category of a label the table lacks. It has no plural, so no query names it.
OTHER = "other"

CATEGORIES: dict[str, str] = {
    # The arena's objects, with FakeRobot's categories.
    "apple": "fruit",
    "banana": "fruit",
    "orange": "fruit",
    "mug": "dish",
    "cup": "dish",
    "red plate": "dish",
    "plate": "dish",
    "bowl": "dish",
    "chips bag": "snack",
    "pringles can": "snack",
    "cookies box": "snack",
    "water bottle": "drink",
    "bottle": "drink",
    "cola can": "drink",
    "milk carton": "drink",
    # The seats and surfaces of walkie-vision's hri and memory tasks, and their kin.
    "chair": "furniture",
    "sofa": "furniture",
    "armchair": "furniture",
    "stool": "furniture",
    "bench": "furniture",
    "table": "furniture",
    "desk": "furniture",
    "shelf": "furniture",
    "cabinet": "furniture",
    "bed": "furniture",
    # walkie-vision's memory task looks for backpacks, the bags guests hand over.
    "backpack": "bag",
    "bag": "bag",
}

# A category's plural, as a command names it: "fruits", "dishes". FakeRobot's for the
# arena's four; furniture has no plural of its own.
PLURALS: dict[str, str] = {
    "fruit": "fruits",
    "dish": "dishes",
    "snack": "snacks",
    "drink": "drinks",
    "furniture": "furniture",
    "bag": "bags",
}


def category_of(label: str) -> str:
    """The category of the label `label`, ignoring case; OTHER for one the table lacks."""
    return CATEGORIES.get(label.strip().lower(), OTHER)


def words_for(label: str) -> frozenset[str]:
    """What a query may say to match `label`: the label, its category and the plural, in lower case."""
    name = label.strip().lower()
    category = category_of(name)
    words = {name}
    if category != OTHER:
        words.add(category)
        words.add(PLURALS.get(category, f"{category}s"))
    return frozenset(words)


def matches(wanted: str, label: str) -> bool:
    """True when the query `wanted`, already stripped and in lower case, matches `label`.

    An empty query matches every label.
    """
    return not wanted or wanted in words_for(label)


def looks_for(wanted: str, classes: Iterable[str]) -> bool:
    """True when some label of `classes` could match `wanted`, or `wanted` is empty.

    `classes` is the detector's vocabulary. Without one, every query counts, since
    nothing says what the detector leaves out.
    """
    labels = list(classes)
    return not wanted or not labels or any(matches(wanted, label) for label in labels)
