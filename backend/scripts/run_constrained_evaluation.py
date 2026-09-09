"""Run a constrained regular-grid-to-Zarr evaluation suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.evaluation.constrained_workflow import (  # noqa: E402
    run_constrained_evaluation_file,
)
from backend.evaluation.objective_v3_schemas import (  # noqa: E402
    ConstrainedEvaluationRunV3,
    EvaluationFailureV3,
)
from backend.evaluation.objective_v3_workflow import (  # noqa: E402
    run_evaluation_v3_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a versioned constrained evaluation and emit feasible-only "
            "operational and engineering objectives."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--llm-timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    config_for_dispatch = (
        args.config.resolve()
        if args.config.is_absolute()
        else (ROOT / args.config).resolve()
    )
    payload = json.loads(config_for_dispatch.read_text(encoding="utf-8"))
    if payload.get("schema_version") in {
        "evaluation_constrained.regular_grid_zarr.v3",
        "evaluation_constrained.terraio_extension.v3",
    }:
        run, result_path = run_evaluation_v3_file(
            args.config,
            repository_root=ROOT,
            output_dir=args.output_dir,
            llm_timeout_seconds=args.llm_timeout_seconds,
        )
    else:
        run, result_path = run_constrained_evaluation_file(
            args.config,
            repository_root=ROOT,
            output_dir=args.output_dir,
        )
    try:
        displayed = result_path.relative_to(ROOT)
    except ValueError:
        displayed = result_path
    print(f"Saved constrained evaluation to {displayed}")
    if isinstance(run, EvaluationFailureV3):
        print(f"status={run.status} phase={run.phase} error={run.error}")
        return 2
    if isinstance(run, ConstrainedEvaluationRunV3):
        print(v3_summary(run))
        return 0 if run.summary.objective_vector_complete else 1
    print(json_summary(run))
    return 0 if run.summary.feasible else 1


def json_summary(run: object) -> str:
    """Keep CLI output compact while the JSON report retains full evidence."""

    summary = getattr(run, "summary")
    return summary.model_dump_json(indent=2)


def v3_summary(run: ConstrainedEvaluationRunV3) -> str:
    """Print the complete human signal required by the v3 protocol."""

    lines = ["Hard constraints:"]
    lines.extend(
        f"  {name}: {status.value}" for name, status in run.summary.constraints.items()
    )
    lines.append("Diagnostic operational measurements:")
    for name in (
        "materialization_seconds",
        "consumer_samples_per_second",
        "output_bytes",
    ):
        direction = run.summary.objective_directions[name]
        value = run.summary.diagnostic_operational_metrics.get(name, "N/A")
        lines.append(f"  {name} ({direction}): {value}")
    storage = run.deterministic.summary.diagnostics.get("output", {})
    open_latency = storage.get("dataset_open_latency_seconds") or {}
    lines.extend(
        [
            "Storage diagnostics:",
            f"  chunk_bytes: {storage.get('chunk_bytes', 'N/A')}",
            f"  metadata_bytes: {storage.get('metadata_bytes', 'N/A')}",
            f"  object_count: {storage.get('object_count', 'N/A')}",
            "  dataset_open_latency_seconds_median: "
            + str(open_latency.get("median", "N/A")),
        ]
    )
    lines.append(
        "Diagnostic engineering quality: q_engineering (maximize): "
        + str(
            run.summary.diagnostic_engineering_quality.get(
                "q_engineering", "N/A"
            )
        )
    )
    lines.append("MERODA components:")
    for component in run.engineering_quality.components:
        value = (
            "N/A"
            if component.applicability == "not_applicable"
            else str(component.median_score)
        )
        lines.append(f"  {component.dimension}: {value}")
    if not run.engineering_quality.components:
        lines.extend(f"  {dimension}: N/A" for dimension in "MERODA")
    lines.extend(
        [
            f"feasible: {run.summary.feasible}",
            f"engineering_assessed: {run.summary.engineering_assessed}",
            f"objective_vector_complete: {run.summary.objective_vector_complete}",
            f"optimization_ready: {run.summary.optimization_ready}",
            f"thesis_evidence_ready: {run.summary.thesis_evidence_ready}",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
