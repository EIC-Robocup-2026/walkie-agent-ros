"""Remember a person with their name and facts, so recognize_person can name them.

Sends the handle's own face and appearance vectors and every one VisionFeed
collected for the person's track, without the outliers of each model, to
the `people_memory` backend (hri.md, Faces collected while talking). Its effect is the
provisional W: no decided effect covers a memory write yet (perception.md,
question 3).
"""

from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from walkie_agent.perception.identity import reject_outliers
from walkie_agent.perception.people_memory import ModelVector
from walkie_agent.toolbox import Effect, StepError, ToolContext, ToolMeta, ToolOutput
from walkie_agent.tools.atomic import _perception as shared

META = ToolMeta(
    name="remember_person",
    description=(
        "Remember a person you are talking to, so recognize_person can name them later. "
        "Give their person handle from find_people, such as per_51c0.0, their name as "
        "they said it, and the facts they told you, such as {\"drink\": \"milk\"}. It "
        "stores every face and appearance of that person seen while they were in view, so "
        "call it while they are still in front of you. Fails with NOT_FOUND when neither "
        "their face nor their appearance was seen."
    ),
    effects=frozenset({Effect.W}),
    questions=(
        "My name is Sam, please remember me.",
        "I am Ana and my favourite drink is tea.",
        "Remember this guest's name and drink.",
    ),
    examples=(
        {"person": "per_51c0.0", "name": "Sam", "facts": {"drink": "milk"}},
        {"person": "per_51c0.1", "name": "Ana"},
    ),
    requires=(shared.VISION, shared.PEOPLE_MEMORY),
)

# Short enough to say back to a guest, long enough for any popular name.
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
FactKey = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=30)]
FactValue = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class Args(BaseModel):
    person: str = Field(description="The person's handle from find_people, such as per_51c0.0.")
    name: Name = Field(description="The person's name, as they said it.")
    facts: dict[FactKey, FactValue] = Field(
        default_factory=dict,
        max_length=8,
        description=(
            'What the person told you about themselves, such as {"drink": "milk"}: '
            "at most 8 facts, all text."
        ),
    )


class Out(ToolOutput):
    person_id: str
    # The vectors stored.
    faces: int
    appearances: int
    # The vectors left out as outliers, such as a bystander's face in one image.
    dropped_faces: int
    dropped_appearances: int
    # The models the people memory has no table for, whose vectors were left
    # out, so they can never recognize the person.
    unstored_models: list[str]


def run(ctx: ToolContext, args: Args) -> Out:
    person = args.person.strip()
    if not shared.is_handle_id(person):
        # A set id: one name cannot go to several people.
        raise StepError(
            "REFUSED",
            f"{person} is not one person's handle. Give one handle, such as per_51c0.0.",
            {"handle": person},
        )
    handle = ctx.handles.get(person, kind="person")
    memory = ctx.backend(shared.PEOPLE_MEMORY)
    faces, appearances = shared.person_vectors(shared.vision_feed(ctx), handle.payload)
    kept_faces = _without_outliers(faces)
    kept_appearances = _without_outliers(appearances)
    if not kept_faces and not kept_appearances:
        raise shared.no_vectors(handle.id)

    # The guest's answer may come long after the question; a stop since
    # then must not store them.
    ctx.check_cancelled()
    stored = memory.remember(args.name, args.facts, kept_faces, kept_appearances)

    facts = f" ({shared.facts_text(args.facts)})" if args.facts else ""
    vectors = "vector" if stored.appearances == 1 else "vectors"
    summary = (
        f"Remembered {args.name}{facts} from {stored.faces} face and "
        f"{stored.appearances} appearance {vectors}."
    )
    if stored.unstored_models:
        # Such a model can never name them, so the LLM must not count on it.
        summary += (
            f" Left out the vectors of {', '.join(stored.unstored_models)}: the people "
            "memory has no table for them."
        )
    return Out(
        summary=summary,
        person_id=stored.person_id,
        faces=stored.faces,
        appearances=stored.appearances,
        dropped_faces=len(faces) - len(kept_faces),
        dropped_appearances=len(appearances) - len(kept_appearances),
        unstored_models=list(stored.unstored_models),
    )


def _without_outliers(vectors: Sequence[ModelVector]) -> list[ModelVector]:
    """The vectors without each model's outliers, model by model in order of first sight.

    Outliers are judged within one model, since the vectors of two models
    are never compared.
    """
    kept: list[ModelVector] = []
    for model, group in shared.group_by_model(vectors).items():
        kept.extend(
            ModelVector(model, vector)
            for vector in reject_outliers(group, shared.CONFIG.identity)
        )
    return kept
