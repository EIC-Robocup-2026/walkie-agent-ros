"""What the oracle and the LLM arms share: the dataset, the episode's robot and the scorer."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.event import SampleLimitEvent
from inspect_ai.log import transcript
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from mahoraga.checkers import check
from mahoraga.driver import ScriptResult
from mahoraga.shelf import Shelf
from mahoraga.worlds import build_world, world_seed
from walkie_agent.capabilities.fake import FakeRobot

SCORER = "ground_truth"


def dataset(records: Sequence[Mapping[str, Any]]) -> MemoryDataset:
    """One sample per command. Its id is the generator's seed, which is stable across runs."""
    return MemoryDataset(
        [
            Sample(id=record["seed"], input=record["text"], metadata={"record": dict(record)})
            for record in records
        ],
        name="gpsr",
    )


def spec_of(state: TaskState) -> Mapping[str, Any]:
    return state.metadata["record"]["spec"]


@contextmanager
def episode_robot(state: TaskState) -> Iterator[FakeRobot]:
    """The seeded robot of one (command, epoch), with its first and last snapshots kept."""
    robot = FakeRobot(build_world(spec_of(state), world_seed(state.sample_id, state.epoch)))
    state.store.set("start", robot.snapshot())
    try:
        yield robot
    finally:
        state.store.set("end", robot.snapshot())


def script_record(code: str, result: ScriptResult) -> dict[str, Any]:
    """One script's entry in the sample's store, which the model never sees."""
    return {
        "code": code,
        "ok": result.ok,
        "phase": result.phase,
        "error": result.error,
        "calls": result.calls,
        "shelf_used": result.shelf_used,
        "shelf_raised": result.shelf_raised,
        "returned": result.returned,
        "raised": result.raised,
        "unwatched": [list(version) for version in result.unwatched],
        "saved": result.saved,
        "refused": result.refused,
    }


def _proven(scripts: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Per name, the last version that returned in an episode and that nothing raised through in it.

    A version that ran where no wrapper watched it counts as raised. A record from
    before `unwatched` existed has none.
    """
    raised = {version for script in scripts for version in script["raised"].items()}
    raised |= {tuple(version) for script in scripts for version in script.get("unwatched", ())}
    proven: dict[str, str] = {}
    for script in scripts:
        for name, sha in script["returned"].items():
            if (name, sha) not in raised:
                proven[name] = sha
    return proven


def _how_it_ended_badly(state: TaskState) -> str | None:
    """Why an episode cannot count, whatever its final world shows.

    A limit or a script cut off means the episode did not finish the work, even if
    the world already looks right. A script that failed earlier and was recovered from
    does not count against it, so only the last script's end matters.
    """
    for event in transcript().events:
        if isinstance(event, SampleLimitEvent):
            return f"A limit ended the episode: {event.message}"
    if state.store.get("out_of_rounds"):
        return "The episode ran out of rounds."
    scripts = state.store.get("scripts", [])
    if scripts and not scripts[-1]["ok"]:
        return f"The last script did not run to completion: {scripts[-1]['error']}"
    return None


def ground_truth(shelf: Shelf | None = None) -> Scorer:
    """Score from the snapshots and how the episode ended.

    A success marks, per function name, the last version that ran and returned in it
    as verified. A function it saved or loaded and never ran earns nothing, and neither
    does a version that raised, even if its caller caught that, or one whose text also
    ran in the episode where no wrapper watched it.
    """

    @scorer(metrics=[mean()], name=SCORER)
    def factory() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            start, end = state.store.get("start"), state.store.get("end")
            if start is None or end is None:
                return Score(value=0.0, explanation="The episode never started.")
            ok, why = check(spec_of(state), start, end)
            ended_badly = _how_it_ended_badly(state) if ok else None
            if ended_badly:
                ok, why = False, ended_badly
            if ok and shelf is not None:
                shelf.mark_verified(_proven(state.store.get("scripts", [])))
            return Score(
                value=1.0 if ok else 0.0,
                explanation=why,
                metadata={"kind": spec_of(state)["kind"]},
            )

        return score

    return factory()
