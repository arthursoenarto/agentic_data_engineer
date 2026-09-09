"""Run generation-owned acceptance for one immutable family pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.etl_pipeline.acceptance import accept_family_pipeline  # noqa: E402
from backend.env import ENV_FILE  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify generated tests, real Zarr publication, and two credential-free "
            "read-only-cache executions without modifying candidate source."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--contract-lock", type=Path, required=True)
    parser.add_argument("--prepared-cache", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--acceptance-id")
    parser.add_argument("--candidate-origin", choices=["generated", "repaired"], default="generated")
    parser.add_argument("--repair-reference")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    parser.add_argument("--skip-generated-tests", action="store_true")
    parser.add_argument("--python-executable", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _args()
    report, report_path = accept_family_pipeline(
        dataset_dir=arguments.dataset_dir,
        pipeline_id=arguments.pipeline_id,
        contract_lock_path=arguments.contract_lock,
        prepared_cache_dir=arguments.prepared_cache,
        inventory_path=arguments.inventory,
        acceptance_id=arguments.acceptance_id,
        candidate_origin=arguments.candidate_origin,
        repair_reference=arguments.repair_reference,
        timeout_seconds=arguments.timeout_seconds,
        env_path=arguments.env_file,
        run_generated_tests=not arguments.skip_generated_tests,
        python_executable=arguments.python_executable,
    )
    print(f"Acceptance: {report.acceptance_id}")
    print(f"Status: {report.final_status}")
    print(f"Report: {report_path}")
    for check in report.checks:
        role = "required" if check.required else "advisory"
        print(f"{check.status.upper()} [{role}] {check.check_id}: {check.feedback_code}")
    raise SystemExit(0 if report.final_status == "passed" else 1)
