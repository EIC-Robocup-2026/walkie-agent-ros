"""What the oracle and the LLM arms share: the dataset, the episode's robot and the scorer."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
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
        "saved": result.saved,
        "refused": result.refused,
    }


def ground_truth(shelf: Shelf | None = None) -> Scorer:
    """Score from the snapshots. A success marks the shelf functions it used or saved as verified."""

    @scorer(metrics=[mean()], name=SCORER)
    def factory() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            start, end = state.store.get("start"), state.store.get("end")
            if start is None or end is None:
                return Score(value=0.0, explanation="The episode never started.")
            ok, why = check(spec_of(state), start, end)
            if ok and shelf is not None:
                scripts = state.store.get("scripts", [])
                shelf.mark_verified(
                    sorted({name for s in scripts for name in (*s["shelf_used"], *s["saved"])})
                )
            return Score(
                value=1.0 if ok else 0.0,
                explanation=why,
                metadata={"kind": spec_of(state)["kind"]},
            )

        return score

    return factory()
