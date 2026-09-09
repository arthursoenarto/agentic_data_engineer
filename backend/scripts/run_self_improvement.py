"""Run a bounded Pareto-greedy or adaptive pipeline improvement search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.orchestration import (  # noqa: E402
    SelfImprovementConfig,
    run_self_improvement,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = SelfImprovementConfig.model_validate_json(
        arguments.config.read_text(encoding="utf-8")
    )
    report, path = run_self_improvement(config)
    print(f"Search status: {report.status}")
    print(f"Strategy: {report.strategy}")
    print(f"Nodes: {len(report.node_records)}")
    print(f"Pareto archive: {', '.join(report.pareto_archive) or 'none'}")
    print(f"Continuation node: {report.continuation_node_id or 'none'}")
    print(f"Stop reason: {report.stop_reason}")
    print(f"Report: {path}")
    return 0 if report.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
