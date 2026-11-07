"""The oracle gate: no LLM arm runs on data the oracle has not passed twice, identically.

A gate run plays the oracle's scripts through the same driver, world builder and
scorer an arm uses, twice, into `.eval` logs. It passes when both logs read back
complete, every sample scored 1.0 with every script run to completion and no limit
hit, and the two are equal once the fields that differ between any two runs are
removed (VOLATILE). The result goes under the Mahoraga home, with a digest of each
(command, epoch, world) it covered, a digest of the code it covered and a hash of
each log. require_gate() refuses anything else.
"""

import asyncio
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.solver import Generate, Solver, TaskState, solver
from pydantic_monty import Monty

import walkie_agent.tools.atomic as atomics
from walkie_agent.mahoraga import home
from walkie_agent.mahoraga.commands import DATA_DIR, TRAIN_COMMANDS, commands
from walkie_agent.mahoraga.oracle import oracle_script
from walkie_agent.mahoraga.tasks import (
    SCORER,
    dataset,
    episode_robot,
    ground_truth,
    spec_of,
)
from walkie_agent.mahoraga.worlds import build_world, world_seed
from walkie_agent.scripting.driver import ScriptRunner, allow_list, script_record
from walkie_agent.scripting.sandbox import open_pool

# Fields that differ between any two runs of the same thing: clocks, random ids
# and where the log landed. Nothing else is excluded. "id" is random on an event
# or a message, so canonical() puts each sample's back.
VOLATILE = frozenset(
    {
        "completed_at",
        "created",
        "eval_id",
        "id",
        "location",
        "parent_id",
        "run_id",
        "span_id",
        "started_at",
        "task_id",
        "timestamp",
        "total_time",
        "uuid",
        "working_start",
        "working_time",
    }
)
# (split, commands, epochs): the sizes of the plan's R1 trial, induction and held-out set.
STANDARD_DATA = (("trial", 100, 3), ("train", TRAIN_COMMANDS, 1), ("heldout", 160, 3))
_MODEL = "mockllm/model"
_REPO = Path(__file__).resolve().parents[3]
# These run after every episode has ended, so an edit to them cannot change an oracle pass.
_NOT_DIGESTED = frozenset({"report.py", "__main__.py"})


class GateError(RuntimeError):
    """An LLM arm asked to run on data or code the oracle gate has not passed."""


@dataclass
class GateResult:
    """One gate run. `digests` holds each pass's canonical log digest."""

    passed: bool
    split: str
    category: str
    commands: int
    epochs: int
    episodes_per_pass: int
    code_digest: str
    digests: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    log_hashes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    pairs: list[str] = field(default_factory=list)
    path: str = ""


def _digested_files() -> list[Path]:
    package = Path(__file__).resolve().parent
    walkie = package.parent
    return sorted(
        [
            *(path for path in package.glob("*.py") if path.name not in _NOT_DIGESTED),
            *(path.resolve() for path in DATA_DIR.rglob("*") if path.is_file()),
            *(walkie / name for name in ("__init__.py", "errors.py", "adapter.py", "locations.py")),
            *(walkie / "scripting").glob("*.py"),
            *(walkie / "capabilities").glob("*.py"),
            # The package docstring, the atomics with their tool sets, and arm D's stand-ins.
            walkie / "tools" / "__init__.py",
            *(walkie / "tools" / "atomic").glob("*.py"),
            *(walkie / "tools" / "bundled_preset").glob("*.py"),
            # The arm records, states and answers that capabilities and the atomics
            # import. Its loaders read config/*.toml only when called, and no oracle
            # pass calls them: FakeRobot has no arm poses there.
            *(walkie / "arm").glob("*.py"),
            # FakeRobot and face_person import the angle wrap from geometry.py, which
            # imports gestures.py. people.py serves only RosVision.
            *(walkie / "perception" / name for name in ("__init__.py", "geometry.py", "gestures.py")),
            _REPO / "pixi.lock",
        ]
    )


def code_digest() -> str:
    """A digest of everything an oracle pass depends on, so an edit to any of it stales the gate.

    Each file counts by its repository path and its content, so two files with one
    name stay apart. A broad list costs a rerun of about a minute after an unrelated edit.
    """
    digest = hashlib.sha256()
    for path in _digested_files():
        digest.update(path.relative_to(_REPO).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def _file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_digests(records: Iterable[Mapping[str, Any]], epochs: int) -> set[str]:
    """One digest per (command, epoch), over its text, spec and world."""
    pairs = set()
    for record in records:
        for epoch in range(1, epochs + 1):
            text = json.dumps(
                {
                    "id": record["seed"],
                    "epoch": epoch,
                    "text": record["text"],
                    "spec": record["spec"],
                    "world": build_world(record["spec"], world_seed(record["seed"], epoch)),
                },
                sort_keys=True,
            )
            pairs.add(hashlib.sha256(text.encode()).hexdigest()[:24])
    return pairs


def oracle_task(records: Sequence[Mapping[str, Any]], epochs: int, pool: Monty) -> Task:
    """The oracle as a task: no model call, only its scripts on the driver."""
    return Task(
        dataset=dataset(records),
        solver=_oracle_solver(ScriptRunner(pool)),
        scorer=ground_truth(),
        epochs=epochs,
        name="gpsr_oracle",
        metadata={"arm": "oracle"},
    )


def _oracle_solver(runner: ScriptRunner) -> Solver:
    @solver(name="oracle")
    def factory() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            code = oracle_script(spec_of(state))
            with episode_robot(state) as robot:
                # The driver blocks on its worker, so it leaves the event loop free.
                result = await asyncio.to_thread(runner.run, code, allow_list(atomics.GPSR, robot.ctx))
            state.store.set("scripts", [script_record(code, result)])
            return state

        return solve

    return factory()


def canonical(log: EvalLog) -> dict[str, Any]:
    """A log as JSON data with the volatile fields gone and the samples in a fixed order."""
    data = log.model_dump(mode="json", exclude={"samples"})
    samples = sorted(log.samples or [], key=lambda sample: (str(sample.id), sample.epoch))
    data["samples"] = [sample.model_dump(mode="json") for sample in samples]
    clean = _without_volatile(data)
    for sample, kept in zip(samples, clean["samples"], strict=True):
        kept["id"] = sample.id
    return clean


def _without_volatile(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _without_volatile(v) for k, v in data.items() if k not in VOLATILE}
    if isinstance(data, list):
        return [_without_volatile(item) for item in data]
    return data


def first_difference(one: Any, two: Any, path: str = "$") -> str | None:
    """Where two JSON values first differ, as a path, for a failed gate's report."""
    if isinstance(one, dict) and isinstance(two, dict):
        for key in sorted({*one, *two}):
            if key not in one or key not in two:
                return f"{path}.{key}"
            found = first_difference(one[key], two[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(one, list) and isinstance(two, list):
        if len(one) != len(two):
            return f"{path}: length {len(one)} against {len(two)}"
        for index, (left, right) in enumerate(zip(one, two, strict=True)):
            found = first_difference(left, right, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if one == two else f"{path}: {one!r} against {two!r}"


def _digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _failures_of(log: EvalLog, expected: int) -> list[str]:
    """Why one pass fell short of complete and 100%."""
    failures = []
    if log.status != "success":
        failures.append(f"The log's status is {log.status}.")
    samples = log.samples or []
    if len(samples) != expected:
        failures.append(f"The log has {len(samples)} samples, not {expected}.")
    for sample in samples:
        name = f"Sample {sample.id} epoch {sample.epoch}"
        score = (sample.scores or {}).get(SCORER)
        if sample.error is not None or score is None or score.value != 1.0:
            why = sample.error.message if sample.error else (score.explanation if score else "unscored")
            failures.append(f"{name}: {why}")
        if sample.limit is not None or any(event.event == "sample_limit" for event in sample.events):
            failures.append(f"{name}: a limit ended it.")
        scripts = sample.store.get("scripts") or []
        if not scripts or not all(script["ok"] for script in scripts):
            failures.append(f"{name}: an oracle script did not run to completion.")
    return failures


def run_gate(
    *,
    split: str = "trial",
    n: int = 100,
    epochs: int = 3,
    category: str = "mixed",
    max_samples: int = 8,
    sandbox: bool = True,
) -> GateResult:
    """Run the oracle twice over one split's data, and record the result under the home."""
    records = commands(split, n, category)
    expected = len(records) * epochs
    root = home() / "gate"
    code = code_digest()
    result = GateResult(False, split, category, len(records), epochs, expected, code)
    data = []
    with open_pool(sandbox=sandbox, max_processes=max_samples) as pool:
        for attempt in (1, 2):
            log = inspect_eval(
                oracle_task(records, epochs, pool),
                model=_MODEL,
                log_dir=str(root / "logs" / f"{code}-{split}-{category}-{attempt}"),
                log_format="eval",
                display="none",
                fail_on_error=True,
                max_samples=max_samples,
                score_display=False,
            )[0]
            back = read_eval_log(log.location)
            result.logs.append(str(log.location))
            result.log_hashes.append(_file_hash(log.location))
            result.failures += _failures_of(back, expected)
            data.append(canonical(back))
            result.digests.append(_digest(data[-1]))
    if result.digests[0] != result.digests[1]:
        result.failures.append(f"The logs differ at {first_difference(*data)}.")
    result.passed = not result.failures
    if result.passed:
        result.pairs = sorted(pair_digests(records, epochs))
    path = root / f"gate-{code}-{split}-{category}-{len(records)}x{epochs}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.path = str(path)
    path.write_text(json.dumps(asdict(result), indent=1) + "\n")
    return result


def run_standard_gate(*, category: str = "mixed", max_samples: int = 8, sandbox: bool = True) -> list[GateResult]:
    """Gate the data of the plan's trial, induction and held-out runs."""
    return [
        run_gate(split=split, n=n, epochs=epochs, category=category, max_samples=max_samples, sandbox=sandbox)
        for split, n, epochs in STANDARD_DATA
    ]


def _pairs_of(path: Path, code: str) -> set[str]:
    """The pairs a result on disk covers, which are none unless it is a whole, passing gate for `code`.

    Its two passes must agree and its two logs must still hash as recorded. This
    catches a stale, partial or hand-edited file made by mistake. It is no security
    boundary: whoever can write the home can rewrite a log and its hash together.
    """
    try:
        result = json.loads(path.read_text())
        first, second = result["digests"]
        logs = result["logs"]
        hashes = result["log_hashes"]
        whole = (
            result["passed"] is True
            and result["code_digest"] == code
            and result["failures"] == []
            and first == second
            and len(logs) == len(hashes) == 2
            and all(_file_hash(log) == kept for log, kept in zip(logs, hashes, strict=True))
        )
        return set(result["pairs"]) if whole else set()
    except (KeyError, TypeError, ValueError, OSError):
        return set()


def require_gate(records: Iterable[Mapping[str, Any]], epochs: int) -> None:
    """Raise GateError unless whole, passing gate results, for this code, cover this data."""
    code = code_digest()
    covered: set[str] = set()
    for path in (home() / "gate").glob(f"gate-{code}-*.json"):
        covered |= _pairs_of(path, code)
    missing = len(pair_digests(records, epochs) - covered)
    if missing:
        raise GateError(
            f"The oracle gate has not passed {missing} of these (command, epoch) pairs for code "
            f"{code}. Run: python -m walkie_agent.mahoraga gate --split SPLIT --commands N --epochs K"
        )
