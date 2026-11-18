"""The plan's metrics, per arm, from `.eval` logs, and each arm's paired comparison.

The command is the unit, so an arm's success on a command is the mean of its
scheduled epochs. An episode that errored or never ran stays in that mean as a
failure, and the usage an errored one logged counts towards tokens and cost. Rounds,
wall time and the script metrics cover every episode logged. Such an episode marks
its arm and each comparison with that arm as incomplete, and the report says so. A
command missing from either arm drops out of their comparison, which is then
incomplete too. So is a comparison whose arms were scheduled different epochs of a
command, or hold different text under one sample id.
"""

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, EvalSample, read_eval_log

from walkie_agent.evals.mahoraga.tasks import SCORER
from walkie_agent.scripting.shelf import StoreRefused, read_counts, store_root

_WARNING ="Incomplete, so these comparisons do not measure the whole evaluation. {}. They count as failures."


def read_runs(log_dir: str | Path) -> dict[str, EvalLog]:
    """The latest log of each arm under `log_dir`."""
    logs = [read_eval_log(str(path)) for path in sorted(Path(log_dir).rglob("*.eval"))]
    latest: dict[str, EvalLog] = {}
    for log in sorted(logs, key=lambda log: log.eval.created):
        latest[log.eval.metadata["arm"]] = log
    return latest


def _was_scored(sample: EvalSample) -> bool:
    return sample.error is None and SCORER in (sample.scores or {})


def _scheduled(log: EvalLog) -> dict[tuple[Any, int], float]:
    """Each (command, epoch) the run was given, with its score. One with no score has 0.

    A log without sample ids still says how many commands it was given. Each one it never
    logged gets a stand-in id unique to the log, so it stays a failure and pairs with nothing.
    """
    logged = log.samples or []
    commands = log.eval.dataset.sample_ids
    if not commands:
        commands = list(dict.fromkeys(sample.id for sample in logged))
        unlogged = (log.eval.dataset.samples or 0) - len(commands)
        commands += [f"{log.eval.eval_id}:unlogged:{n}" for n in range(unlogged)]
    epochs = range(1, (log.eval.config.epochs or 1) + 1)
    values = {(command, epoch): 0.0 for command in commands for epoch in epochs}
    for sample in logged:
        values[sample.id, sample.epoch] = float(sample.scores[SCORER].value) if _was_scored(sample) else 0.0
    return values


def by_command(log: EvalLog) -> dict[Any, float]:
    """Mean success of each command over its scheduled epochs."""
    values = defaultdict(list)
    for (command, _), value in _scheduled(log).items():
        values[command].append(value)
    return {command: statistics.fmean(value) for command, value in values.items()}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _recorded_cost(event: ModelEvent) -> float | None:
    """The USD a provider's raw response says a model call cost, if the log holds one."""
    response = event.call.response if event.call else None
    usage = response.get("usage") if isinstance(response, dict) else None
    return usage.get("cost") if isinstance(usage, dict) else None


def arm_metrics(log: EvalLog) -> dict[str, Any]:
    """M1 to M9 for one arm. M9 counts refused calls: an escape would not show in a log.

    M3's input holds everything the model was sent, with the part read from the cache
    also given alone, because Inspect's `input_tokens` leaves out cache reads and writes.
    Its cost is what the provider recorded. Any call with no record, such as an abandoned
    attempt, is counted beside it, which makes the cost a lower bound, and no records at
    all make it None.
    """
    logged = log.samples or []
    scheduled = _scheduled(log)
    errored = sum(not _was_scored(sample) for sample in logged)
    never_ran = len(set(scheduled) - {(sample.id, sample.epoch) for sample in logged})
    successes = sum(scheduled.values())
    usages = [usage for sample in logged for usage in (sample.model_usage or {}).values()]
    tokens_cached = sum(usage.input_tokens_cache_read or 0 for usage in usages)
    tokens_in = tokens_cached + sum(usage.input_tokens + (usage.input_tokens_cache_write or 0) for usage in usages)
    tokens_out = sum(usage.output_tokens for usage in usages)
    costs = [_recorded_cost(event) for sample in logged for event in sample.events if event.event == "model"]
    recorded = [cost for cost in costs if cost is not None]
    cost = sum(recorded) if recorded else None
    per_episode = [sample.store.get("scripts") or [] for sample in logged]
    with_scripts = [scripts for scripts in per_episode if scripts]
    failed_first = [scripts for scripts in with_scripts if not scripts[0]["ok"]]
    shelf_hits = [s for s in with_scripts if any(script["shelf_used"] for script in s)]
    return {
        "complete": log.status == "success" and not errored and not never_ran,
        "episodes": len(scheduled),
        "errored": errored,
        "never_ran": never_ran,
        "M1_success": _mean(by_command(log).values()),
        "M2_rounds": _mean(sum(e.event == "model" for e in s.events) for s in logged),
        "M3_tokens_in": tokens_in,
        "M3_tokens_in_cached": tokens_cached,
        "M3_tokens_out": tokens_out,
        "M3_tokens_per_success": _rate(tokens_in + tokens_out, successes),
        "M3_cost_usd": cost,
        "M3_cost_per_success_usd": None if cost is None else _rate(cost, successes),
        "M3_cost_unrecorded_calls": len(costs) - len(recorded),
        "M4_wall_s": _mean(s.total_time for s in logged if s.total_time is not None),
        "M5_first_try_type_check": _rate(
            sum(scripts[0]["phase"] not in ("type_check", "syntax") for scripts in with_scripts),
            len(with_scripts),
        ),
        "M6_ok_after_one_repair": _rate(
            sum(len(scripts) > 1 and scripts[1]["ok"] for scripts in failed_first),
            len(failed_first),
        ),
        "M7_shelf_hit": _rate(len(shelf_hits), len(with_scripts)),
        "M8_harmful_reuse": _rate(
            sum(any(script["shelf_raised"] for script in s) for s in shelf_hits),
            len(shelf_hits),
        ),
        "M9_refused_calls": sum(
            call["status"] in ("denied", "unknown_name")
            for scripts in per_episode
            for script in scripts
            for call in script["calls"]
        ),
    }


def shelf_metrics(root: str | Path | None = None) -> dict[str, int]:
    """M10: the size of the shelf in the store at `root`, and how many saved versions it has replaced.

    It only reads the store (read_counts), because `root` may come from a log's metadata:
    opening a Shelf there would create folders and run git in whatever folder the log names.
    """
    counts = read_counts(root)
    return {
        "M10_shelf_size": len(counts),
        "M10_shelf_churn": sum(entry["saved"] for entry in counts.values()) - len(counts),
    }


def mcnemar_p(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p from the two discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def compare(
    baseline: Mapping[Any, float],
    other: Mapping[Any, float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Other minus baseline, by command: the mean, a paired bootstrap 95% CI, and McNemar's check.

    For McNemar's check a command counts as a success when at least half its epochs were.
    """
    paired = sorted(set(baseline) & set(other), key=str)
    if not paired:
        return {"commands": 0, "diff": None, "ci95": None, "mcnemar_p": None}
    diffs = [other[command] - baseline[command] for command in paired]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    only_baseline = sum(baseline[c] >= 0.5 > other[c] for c in paired)
    only_other = sum(other[c] >= 0.5 > baseline[c] for c in paired)
    return {
        "commands": len(paired),
        "diff": statistics.fmean(diffs),
        "ci95": [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]],
        "mcnemar_p": mcnemar_p(only_baseline, only_other),
    }


def _same_commands(baseline: EvalLog, other: EvalLog) -> bool:
    """Whether two arms were scheduled the same (command, epoch) pairs under the same text.

    A world's seed follows from (command, epoch), so the same pairs mean the same worlds.
    """
    if set(_scheduled(baseline)) != set(_scheduled(other)):
        return False
    text = {sample.id: sample.input for sample in baseline.samples or []}
    return all(text.get(sample.id, sample.input) == sample.input for sample in other.samples or [])


def recorded_shelf_root(runs: Mapping[str, EvalLog]) -> str | None:
    """The shelf's store root that the arms' logs record, or None if none records one.

    Raises StoreRefused if they record different roots, as no one shelf then describes them.
    """
    roots = sorted({log.eval.metadata.get("shelf_root") for log in runs.values()} - {None})
    if len(roots) > 1:
        raise StoreRefused(f"The logs ran on different shelves: {', '.join(roots)}. Name one with shelf_root.")
    return roots[0] if roots else None


def report(log_dir: str | Path, *, baseline: str = "A", shelf_root: str | Path | None = None) -> dict[str, Any]:
    """Every arm's metrics under `log_dir`, and each arm against the baseline.

    `warning` is None only if every arm is complete. Each comparison has its own `complete`.
    M10 reads the shelf at `shelf_root`, else the one the logs record, else store_root's
    default, as logs of arms A and B alone, or from before the tier folders, record none.
    `shelf.root` names the store M10 read.
    """
    runs = read_runs(log_dir)
    shelf = store_root(shelf_root or recorded_shelf_root(runs))
    scores = {arm: by_command(log) for arm, log in runs.items()}
    arms = {arm: arm_metrics(log) for arm, log in runs.items()}
    incomplete = [
        f"{arm}: {metrics['errored']} of {metrics['episodes']} errored, {metrics['never_ran']} never ran"
        for arm, metrics in arms.items()
        if not metrics["complete"]
    ]
    return {
        "warning": _WARNING.format("; ".join(incomplete)) if incomplete else None,
        "arms": arms,
        "against_baseline": {
            arm: {
                **compare(scores[baseline], score),
                "complete": (
                    arms[baseline]["complete"] and arms[arm]["complete"] and _same_commands(runs[baseline], runs[arm])
                ),
            }
            for arm, score in scores.items()
            if baseline in scores and arm != baseline
        },
        "shelf": {"root": str(shelf), **shelf_metrics(shelf)},
    }
