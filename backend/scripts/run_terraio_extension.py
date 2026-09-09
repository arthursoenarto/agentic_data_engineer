"""Apply and verify a generated TerraIO extension in an independent clone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.etl_pipeline import execute_repository_extension  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a generated extension to a disposable clone and run its "
            "frozen baseline, verification, and workflow commands."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--extension-id", required=True)
    parser.add_argument(
        "--source-repository",
        type=Path,
        help="Optional source override; defaults to extension_contract.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _args()
    run = execute_repository_extension(
        dataset_dir=arguments.dataset_dir,
        extension_id=arguments.extension_id,
        source_repository=arguments.source_repository,
    )
    print(f"Run ID: {run.receipt.run_id}")
    print(f"Status: {run.receipt.final_status}")
    print(f"Receipt: {run.run_dir / 'pipeline_run.json'}")
    raise SystemExit(0 if run.succeeded else 1)
