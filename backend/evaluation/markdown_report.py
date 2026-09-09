"""Human- and agent-readable summaries for immutable evaluation runs."""

from __future__ import annotations

from backend.evaluation.constrained_schemas import ConstrainedEvaluationRun
from backend.evaluation.objective_v3_schemas import (
    ConstrainedEvaluationRunV3,
    EvaluationFailureV3,
    TerraioExtensionEvaluationRunV3,
)
from backend.evaluation.schemas import CheckStatus


def render_evaluation_report(
    run: ConstrainedEvaluationRun
    | ConstrainedEvaluationRunV3
    | TerraioExtensionEvaluationRunV3
    | EvaluationFailureV3,
) -> str:
    """Render the canonical Markdown companion to machine-readable JSON."""

    if isinstance(run, EvaluationFailureV3):
        return _failure_report(run)
    if isinstance(run, TerraioExtensionEvaluationRunV3):
        return _extension_report(run)
    if isinstance(run, ConstrainedEvaluationRunV3):
        return _regular_grid_report(run.deterministic, run)
    return _regular_grid_report(run, None)


def _regular_grid_report(
    deterministic: ConstrainedEvaluationRun,
    v3: ConstrainedEvaluationRunV3 | None,
) -> str:
    summary = deterministic.summary
    executable = any(
        scenario.execution.exit_code == 0 and not scenario.execution.timed_out
        for scenario in deterministic.scenarios
    )
    lines = [
        "# Evaluation Report",
        "",
        "## Decision",
        "",
        f"- Candidate: `{_cell(summary.target)}`",
        f"- Run: `{_cell(deterministic.run_id)}`",
        f"- Executable under evaluator control: `{str(executable).lower()}`",
        f"- Hard-gate feasible: `{str(summary.feasible).lower()}`",
        "- Official objective vector eligible: "
        f"`{str(v3.summary.objective_vector_complete if v3 else summary.feasible).lower()}`",
    ]
    if v3 is not None:
        lines.extend(
            [
                f"- Engineering quality assessed: `{str(v3.summary.engineering_assessed).lower()}`",
                f"- Optimization ready: `{str(v3.summary.optimization_ready).lower()}`",
                f"- Thesis evidence ready: `{str(v3.summary.thesis_evidence_ready).lower()}`",
            ]
        )

    lines.extend(["", "## Hard Constraints", "", "| Constraint | Status |", "|---|---|"])
    lines.extend(
        f"| `{name}` | **{status.value}** |"
        for name, status in summary.constraints.items()
    )

    diagnostic = (
        v3.summary.diagnostic_operational_metrics
        if v3 is not None
        else summary.diagnostic_metrics
    )
    lines.extend(
        [
            "",
            "## Diagnostic Measurements",
            "",
            "These observations are diagnostic. They are not optimizer-facing unless all hard gates pass.",
            "",
            "| Measurement | Direction | Observed |",
            "|---|---|---:|",
        ]
    )
    for name in (
        "materialization_seconds",
        "consumer_samples_per_second",
        "output_bytes",
    ):
        lines.append(
            f"| `{name}` | {summary.objective_directions[name]} | "
            f"{_number(diagnostic.get(name))} |"
        )
    artifact = summary.diagnostics.get("candidate_artifact", {})
    if artifact:
        lines.extend(
            [
                "",
                "Artifact diagnostics: "
                f"mapped Zarr readable=`{str(artifact.get('mapped_zarr_readable')).lower()}`, "
                f"directory bytes=`{artifact.get('output_directory_bytes', 'N/A')}`, "
                f"files=`{artifact.get('output_directory_files', 'N/A')}`.",
            ]
        )
    storage = summary.diagnostics.get("output", {})
    if storage:
        open_latency = storage.get("dataset_open_latency_seconds") or {}
        lines.extend(
            [
                "",
                "Native Zarr storage diagnostics: "
                f"chunk bytes=`{storage.get('chunk_bytes', 'N/A')}`, "
                f"metadata bytes=`{storage.get('metadata_bytes', 'N/A')}`, "
                f"objects=`{storage.get('object_count', 'N/A')}`, "
                "median local open latency seconds="
                f"`{_number(open_latency.get('median'))}`.",
            ]
        )

    if v3 is not None:
        lines.extend(["", "## Engineering Quality", ""])
        q_value = v3.summary.diagnostic_engineering_quality.get("q_engineering")
        lines.append(f"Diagnostic `q_engineering`: **{_number(q_value)} / 4**")
        lines.extend(
            [
                "",
                "| MERODA | Applicability | Median | Range | Improvement |",
                "|---|---|---:|---:|---|",
            ]
        )
        for component in v3.engineering_quality.components:
            score_range = (
                "N/A"
                if component.score_min is None
                else f"{component.score_min}-{component.score_max}"
            )
            improvement = (
                component.improvements[0]
                if component.improvements
                else "No assessed improvement."
            )
            lines.append(
                f"| {component.dimension} | {component.applicability} | "
                f"{_number(component.median_score)} | {score_range} | "
                f"{_cell(improvement)} |"
            )
        if not v3.engineering_quality.components:
            lines.append("| M/E/R/O/D/A | not assessed | N/A | N/A | "
                         f"{_cell(v3.engineering_quality.error or 'No assessment.')} |")

    eligible_operational = (
        v3.summary.operational_objectives if v3 is not None else summary.objectives
    )
    eligible_engineering = (
        v3.summary.engineering_objective if v3 is not None else {}
    )
    lines.extend(["", "## Official Objective Vector", ""])
    if eligible_operational and (v3 is None or eligible_engineering):
        vector = [
            f"{name}={_number(eligible_operational[name])}"
            for name in (
                "materialization_seconds",
                "consumer_samples_per_second",
                "output_bytes",
            )
        ]
        if v3 is not None:
            vector.append(
                "q_engineering="
                f"{_number(eligible_engineering.get('q_engineering'))}"
            )
        lines.append("`F(p) = (" + ", ".join(vector) + ")`")
    else:
        lines.append(
            "Not eligible. Repair failed hard gates before comparing or optimizing `F(p)`."
        )

    lines.extend(
        [
            "",
            "## Executions",
            "",
            "| Scenario | Exit | Time (s) | Output | Receipt |",
            "|---|---:|---:|---|---|",
        ]
    )
    for scenario in deterministic.scenarios:
        lines.append(
            f"| {scenario.scenario} | {scenario.execution.exit_code} | "
            f"{scenario.execution.duration_seconds:.6f} | "
            f"`{_cell(scenario.output_path)}` | `{_cell(scenario.receipt_path)}` |"
        )
    if not deterministic.scenarios:
        lines.append("| not run | N/A | N/A | N/A | N/A |")

    failed_checks = [
        (constraint.constraint, check)
        for constraint in deterministic.constraints
        for check in constraint.checks
        if check.status != CheckStatus.PASS
    ]
    lines.extend(
        [
            "",
            "## Repair Signals",
            "",
            "| Constraint | Code | Status | What failed |",
            "|---|---|---|---|",
        ]
    )
    for constraint, check in failed_checks:
        lines.append(
            f"| `{constraint}` | `{_cell(check.feedback_code)}` | "
            f"{check.status.value} | {_cell(check.summary)} |"
        )
    if not failed_checks:
        lines.append("| all | `NONE` | pass | All hard-gate checks passed. |")

    lines.extend(["", "## Next Iteration", ""])
    if not executable:
        lines.append("1. Make the candidate execute successfully under evaluator control.")
    elif not summary.feasible:
        lines.append(
            "1. Preserve executability and address the failed hard-gate feedback codes above."
        )
        lines.append("2. Do not optimize diagnostic measurements yet.")
    elif v3 is not None and not v3.summary.engineering_assessed:
        lines.append("1. Complete the MERODA assessment before optimizer comparison.")
    else:
        lines.append(
            "1. Candidate is eligible for Pareto or lexicographic comparison using the official vector."
        )

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Suite: `{_cell(deterministic.config_file)}`",
            "- Machine-readable result: `evaluation.json`",
            f"- Evaluator source bundle: `{deterministic.provenance.get('evaluator_source_bundle', 'N/A')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _extension_report(run: TerraioExtensionEvaluationRunV3) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "## Decision",
        "",
        f"- Candidate: `{_cell(run.summary.target)}`",
        f"- Hard-gate feasible: `{str(run.summary.feasible).lower()}`",
        f"- Engineering quality assessed: `{str(run.summary.engineering_assessed).lower()}`",
        f"- Official objective eligible: `{str(run.summary.objective_vector_complete).lower()}`",
        f"- Optimization ready: `{str(run.summary.optimization_ready).lower()}`",
        f"- Thesis evidence ready: `{str(run.summary.thesis_evidence_ready).lower()}`",
        "",
        "## Extension Gates",
        "",
        "| Check | Status | Code | Summary |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| `{_cell(check.check_id)}` | {check.status.value} | "
        f"`{_cell(check.feedback_code)}` | {_cell(check.summary)} |"
        for check in run.deterministic_checks
    )
    lines.extend(["", "## Engineering Quality", ""])
    lines.append(f"Diagnostic `q_engineering`: **{_number(run.summary.q_engineering)} / 4**")
    lines.extend(["", "| MERODA | Median | Improvements |", "|---|---:|---|"])
    for component in run.engineering_quality.components:
        lines.append(
            f"| {component.dimension} | {_number(component.median_score)} | "
            f"{_cell('; '.join(component.improvements) or 'None recorded.')} |"
        )
    lines.extend(
        [
            "",
            "## Next Iteration",
            "",
            (
                "1. Compare the eligible architecture-quality objective."
                if run.summary.objective_vector_complete
                else "1. Address failed extension gates before optimizer comparison."
            ),
            "",
            "## Provenance",
            "",
            f"- Suite: `{_cell(run.config_file)}`",
            "- Machine-readable result: `evaluation.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _failure_report(run: EvaluationFailureV3) -> str:
    return "\n".join(
        [
            "# Evaluation Report",
            "",
            "## Terminal Failure",
            "",
            f"- Run: `{_cell(run.run_id)}`",
            f"- Phase: `{run.phase}`",
            f"- Status: `{run.status}`",
            f"- Error type: `{_cell(run.error_type)}`",
            f"- Error: {_cell(run.error)}",
            "",
            "No objective values are eligible. Resolve this evaluator or setup failure before candidate optimization.",
            "",
            "Machine-readable result: `evaluation_failure.json`",
            "",
        ]
    )


def _number(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()
