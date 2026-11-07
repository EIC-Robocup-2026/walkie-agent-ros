"""The command line: python -m walkie_agent.mahoraga gate | run | report."""

import argparse
import json
import os
import sys
from collections.abc import Sequence

from walkie_agent.mahoraga.arms import ARMS, INDUCT
from walkie_agent.mahoraga.commands import CATEGORIES, SPLITS
from walkie_agent.mahoraga.gate import GateError, run_gate, run_standard_gate
from walkie_agent.mahoraga.gym import DEFAULT_MODEL, OPENROUTER, run_episodes
from walkie_agent.mahoraga.report import report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m walkie_agent.mahoraga")
    parser.add_argument("--home", help="Where the shelf, gate results and logs live. Default: MAHORAGA_HOME.")
    commands = parser.add_subparsers(dest="command", required=True)

    gate = commands.add_parser("gate", help="Run the oracle gate, with no LLM. Default: the plan's three data sets.")
    gate.add_argument("--split", choices=SPLITS)
    gate.add_argument("--commands", type=int, help="Needed with --split.")
    gate.add_argument("--epochs", type=int, default=3)
    gate.add_argument("--category", choices=CATEGORIES, default="mixed")
    gate.add_argument("--max-samples", type=int, default=8)
    gate.add_argument("--no-bwrap", action="store_true", help="Run workers outside bwrap, on a machine without it.")

    run = commands.add_parser("run", help="Run N commands of the chosen arms. This calls a real LLM.")
    run.add_argument(
        "--arms",
        required=True,
        help=f"Comma-separated, from {', '.join(ARMS)}. Or {INDUCT} alone, with --split train.",
    )
    run.add_argument("--commands", type=int, required=True, help="How many commands. There is no default on purpose.")
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--split", choices=SPLITS, default="trial")
    run.add_argument("--category", choices=CATEGORIES, default="mixed")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--base-url", default=OPENROUTER)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-samples", type=int, default=8)
    run.add_argument("--log-dir", help="Default: a new directory under the home's runs/.")
    run.add_argument("--no-bwrap", action="store_true", help="Run workers outside bwrap, on a machine without it.")

    summary = commands.add_parser("report", help="Print the metrics of a run's logs as JSON.")
    summary.add_argument("--log-dir", required=True)
    summary.add_argument("--baseline", default="A")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.home:
        os.environ["MAHORAGA_HOME"] = args.home
    if args.command == "gate":
        return _gate(parser, args)
    if args.command == "run":
        return _run(args)
    print(json.dumps(report(args.log_dir, baseline=args.baseline), indent=1))
    return 0


def _gate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    sandbox = not args.no_bwrap
    if args.split:
        if args.commands is None:
            parser.error("gate --split needs --commands.")
        results = [
            run_gate(
                split=args.split,
                n=args.commands,
                epochs=args.epochs,
                category=args.category,
                max_samples=args.max_samples,
                sandbox=sandbox,
            )
        ]
    else:
        results = run_standard_gate(category=args.category, max_samples=args.max_samples, sandbox=sandbox)
    for result in results:
        print(
            f"{'PASS' if result.passed else 'FAIL'} {result.split} {result.category}: "
            f"{result.episodes_per_pass} episodes, twice. Digests {result.digests[0][:16]} "
            f"{result.digests[1][:16]}. Result: {result.path}"
        )
        for failure in result.failures[:20]:
            print(f"  {failure}")
    return 0 if all(result.passed for result in results) else 1


def _run(args: argparse.Namespace) -> int:
    try:
        logs = run_episodes(
            [arm.strip() for arm in args.arms.split(",")],
            args.commands,
            epochs=args.epochs,
            split=args.split,
            category=args.category,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            seed=args.seed,
            max_samples=args.max_samples,
            log_dir=args.log_dir,
            sandbox=not args.no_bwrap,
        )
    except (GateError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    for log in logs:
        print(f"{log.eval.metadata['arm']}: {log.status}, log at {log.location}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
