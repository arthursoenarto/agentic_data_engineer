"""Run a constrained station-time-series to Parquet evaluation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.env import load_env  # noqa: E402
from backend.evaluation.station_parquet import run_station_parquet_evaluation  # noqa: E402
from backend.evaluation.station_parquet_schemas import (  # noqa: E402
    StationParquetEvaluationConfig,
)
from backend.llm import LLMClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-engineering-judge", action="store_true")
    arguments = parser.parse_args()
    load_env()
    config = StationParquetEvaluationConfig.model_validate_json(
        arguments.config.read_text(encoding="utf-8")
    )
    if arguments.skip_engineering_judge:
        config.engineering_quality.enabled = False
    run, path = run_station_parquet_evaluation(
        config,
        config_path=arguments.config.resolve(),
        output_dir=arguments.output_dir.resolve(),
        repository_root=ROOT,
        client=None if arguments.skip_engineering_judge else LLMClient(),
    )
    print(f"Feasible: {run.feasible}")
    print(f"Optimization ready: {run.optimization_ready}")
    print(f"Result: {path}")
    return 0 if run.feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
