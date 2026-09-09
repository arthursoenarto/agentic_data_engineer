"""Execute one immutable contract lock through a reusable generated pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.etl_pipeline import execute_family_pipeline  # noqa: E402
from backend.env import ENV_FILE  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dataset-family pipeline and write immutable run evidence.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--contract-lock", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional external cache root; defaults to DATASET_DIR/data/cache.",
    )
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    parser.add_argument(
        "--repair-run-reference",
        help="Optional repository-relative repair log linked from the run receipt.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _args()
    run = execute_family_pipeline(
        dataset_dir=arguments.dataset_dir,
        pipeline_id=arguments.pipeline_id,
        contract_lock_path=arguments.contract_lock,
        inventory_path=arguments.inventory,
        cache_dir=arguments.cache_dir,
        run_id=arguments.run_id,
        timeout_seconds=arguments.timeout_seconds,
        env_path=arguments.env_file,
        repair_run_reference=arguments.repair_run_reference,
        python_executable=arguments.python_executable,
    )
    print(f"Run ID: {run.paths.run_id}")
    print(f"Status: {run.receipt.final_status}")
    print(f"Receipt: {run.paths.receipt}")
    print(f"Output: {run.paths.output_dir}")
    if not run.succeeded and run.paths.stderr.exists():
        diagnostics = run.paths.stderr.read_text(encoding="utf-8").strip()
        if diagnostics:
            print(diagnostics, file=sys.stderr)
    raise SystemExit(0 if run.succeeded else 1)
