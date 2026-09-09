"""Prepare, execute, and summarize thesis experiment E1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.experiments.e1_specification import (
    prepare_e1,
    run_e1_phase,
    summarize_e1,
    verify_e1_freeze,
)


DEFAULT_CONFIG = ROOT / "project/benchmarks/experiments/e1_specification_evaluation_coverage_20260907_v1/config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "verify", "run-development", "run-heldout", "summarize"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_e1(config_path=args.config, repository_root=ROOT))
    elif args.command == "verify":
        verify_e1_freeze(args.config.parent / "freeze_manifest.json", ROOT)
        print("E1 freeze verified")
    elif args.command == "run-development":
        print(f"Completed {len(run_e1_phase(config_path=args.config, repository_root=ROOT, phase='development'))} development runs")
    elif args.command == "run-heldout":
        print(f"Completed {len(run_e1_phase(config_path=args.config, repository_root=ROOT, phase='held_out'))} held-out runs")
    else:
        summary = summarize_e1(config_path=args.config, repository_root=ROOT)
        print(f"E1: {summary['successful_runs']}/{summary['runs']} successful runs")


if __name__ == "__main__":
    main()
