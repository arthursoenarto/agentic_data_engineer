"""Summarize a complete matched matrix of pipeline condition trials."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.etl_pipeline import (  # noqa: E402
    PipelineConditionTrialResult,
    summarize_matched_trials,
    write_experiment_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--trial", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trials = [
        PipelineConditionTrialResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for path in args.trial
    ]
    summary = summarize_matched_trials(
        experiment_id=args.experiment_id,
        repetitions_per_condition=args.repetitions,
        trials=trials,
    )
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    json_path, markdown_path = write_experiment_summary(output.resolve(), summary)
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
