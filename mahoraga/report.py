"""The plan's metrics, per arm, from `.eval` logs, and each arm's paired comparison.

The command is the unit, so an arm's success on a command is the mean of its
epochs. A sample that errored has no score, and a command missing from either arm
drops out of their comparison.
"""

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, EvalSample, read_eval_log

from mahoraga.shelf import Shelf
from mahoraga.tasks import SCORER

# OpenRouter's USD per million tokens for deepseek-v4.1-flash, from the plan.
PRICE_IN = 0.045
PRICE_OUT = 0.50


def read_runs(log_dir: str | Path) -> dict[str, EvalLog]:
    """The latest log of each arm under `log_dir`."""
    logs = [read_eval_log(str(path)) for path in sorted(Path(log_dir).rglob("*.eval"))]
    latest: dict[str, EvalLog] = {}
    for log in sorted(logs, key=lambda log: log.eval.created):
        latest[log.eval.metadata["arm"]] = log
    return latest


def _scored(log: EvalLog) -> list[EvalSample]:
    return [sample for sample in log.samples or [] if SCORER in (sample.scores or {})]


def by_command(log: EvalLog) -> dict[Any, float]:
    """Mean success of each command over its scored epochs."""
    values = defaultdict(list)
    for sample in _scored(log):
        values[sample.id].append(float(sample.scores[SCORER].value))
    return {command: statistics.fmean(value) for command, value in values.items()}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def arm_metrics(log: EvalLog, *, price_in: float = PRICE_IN, price_out: float = PRICE_OUT) -> dict[str, Any]:
    """M1 to M9 for one arm. M9 counts refused calls: an escape would not show in a log."""
    scored = _scored(log)
    successes = sum(float(sample.scores[SCORER].value) for sample in scored)
    tokens_in = sum(u.input_tokens for s in scored for u in s.model_usage.values())
    tokens_out = sum(u.output_tokens for s in scored for u in s.model_usage.values())
    cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
    per_episode = [sample.store.get("scripts") or [] for sample in scored]
    with_scripts = [scripts for scripts in per_episode if scripts]
    failed_first = [scripts for scripts in with_scripts if not scripts[0]["ok"]]
    shelf_hits = [s for s in with_scripts if any(script["shelf_used"] for script in s)]
    return {
        "episodes": len(log.samples or []),
        "errored": len(log.samples or []) - len(scored),
        "M1_success": _mean(by_command(log).values()),
        "M2_rounds": _mean(sum(e.event == "model" for e in s.events) for s in scored),
        "M3_tokens_in": tokens_in,
        "M3_tokens_out": tokens_out,
        "M3_tokens_per_success": _rate(tokens_in + tokens_out, successes),
        "M3_cost_usd": cost,
        "M3_cost_per_success_usd": _rate(cost, successes),
        "M4_wall_s": _mean(s.total_time for s in scored if s.total_time is not None),
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
    """M10: the shelf's size, and how many saved versions it has replaced."""
    counts = Shelf(root).counts()
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


def report(log_dir: str | Path, *, baseline: str = "A", shelf_root: str | Path | None = None) -> dict[str, Any]:
    """Every arm's metrics under `log_dir`, and each arm against the baseline."""
    runs = read_runs(log_dir)
    scores = {arm: by_command(log) for arm, log in runs.items()}
    return {
        "arms": {arm: arm_metrics(log) for arm, log in runs.items()},
        "against_baseline": {
            arm: compare(scores[baseline], score)
            for arm, score in scores.items()
            if baseline in scores and arm != baseline
        },
        "shelf": shelf_metrics(shelf_root),
    }
