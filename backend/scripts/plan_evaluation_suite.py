"""Create an immutable hybrid LLM/deterministic evaluation plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.evaluation_planning import EvaluationPlanningAgent  # noqa: E402
from backend.evaluation.planning import (  # noqa: E402
    create_evaluation_planning_artifacts,
)
from backend.llm import LLMClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--contract-lock", type=Path, required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-id")
    parser.add_argument("--model")
    parser.add_argument("--llm-timeout-seconds", type=int, default=600)
    parser.add_argument("--disable-engineering", action="store_true")
    parser.add_argument("--extensibility-probe", action="store_true")
    args = parser.parse_args()

    run, artifacts = create_evaluation_planning_artifacts(
        agent=EvaluationPlanningAgent(
            LLMClient(model=args.model, timeout_seconds=args.llm_timeout_seconds)
        ),
        plan_id=args.plan_id,
        dataset_dir=_resolve(args.dataset_dir),
        contract_lock_path=_resolve(args.contract_lock),
        repository_root=ROOT,
        output_dir=None if args.output_dir is None else _resolve(args.output_dir),
        profile_id=args.profile_id,
        engineering_enabled=not args.disable_engineering,
        extensibility_probe_enabled=args.extensibility_probe,
    )
    print(f"Evaluation plan: {artifacts.planning_run}")
    print(f"Resolved checks: {len(run.check_plan.checks)}")
    print(f"Suite ready: {run.suite_ready}")
    print(f"Quarantined proposals: {len(run.quarantined_check_proposals)}")


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
