"""The words find_objects uses for walkie-vision's object labels. No ROS.

A detector names an object by its label, one of the text prompts it was given
(ObjectDetectionArray.classes). find_objects reports that label as the object's
name, and a category from CATEGORIES: the arena's categories for the GPSR objects,
the same as FakeRobot's, and "furniture" for the places people sit and things stand
on. A label the table lacks has the category OTHER.

A query matches an object's name, its category or the category's plural, ignoring
case and spaces at either end, as the Vision protocol says. An object of the table
whose name ends with the label as a word, such as "water bottle" for "bottle", names a
kind of it and matches too. A label in BRANDS names one object of the table by its
brand, and takes that object's name and category. A detector looks only
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


# Labels that name one object of the table by its brand. walkie-vision's memory task
# finds the Chula simulation's Pringles cans with the prompt "pringles"; with "can"
# its detector found neither (decision PK-58).
BRANDS: dict[str, str] = {
    "pringles": "pringles can",
}


def category_of(label: str) -> str:
    """The category of the label `label`, ignoring case; OTHER for one the table lacks.

    A brand's label has the category of the object it names.
    """
    name = label.strip().lower()
    return CATEGORIES.get(BRANDS.get(name, name), OTHER)


def words_for(label: str) -> frozenset[str]:
    """What a query may say to match `label`, in lower case.

    The label, the name of the object a brand's label names, the category and its
    plural.
    """
    name = label.strip().lower()
    category = category_of(name)
    words = {name}
    if name in BRANDS:
        words.add(BRANDS[name])
    if category != OTHER:
        words.add(category)
        words.add(PLURALS.get(category, f"{category}s"))
    return frozenset(words)


def matches(wanted: str, label: str) -> bool:
    """True when the query `wanted`, already stripped and in lower case, matches `label`.

    An empty query matches every label. An object of the table (CATEGORIES) whose
    name ends with the label as a word also matches it: "water bottle" matches the
    label "bottle", since a detector given the prompt "bottle" names a water bottle so.
    Any other query matches only as words_for() says, so "coffee table" does not match
    a plain "table".
    """
    if not wanted:
        return True
    name = label.strip().lower()
    if wanted in words_for(name):
        return True
    return bool(name) and wanted in CATEGORIES and wanted.endswith(f" {name}")


def looks_for(wanted: str, classes: Iterable[str]) -> bool:
    """True when some label of `classes` could match `wanted`, or `wanted` is empty.

    `classes` is the detector's vocabulary. Without one, every query counts, since
    nothing says what the detector leaves out.
    """
    labels = list(classes)
    return not wanted or not labels or any(matches(wanted, label) for label in labels)
