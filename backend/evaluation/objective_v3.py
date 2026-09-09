"""Constrained v3 orchestration without generation-strategy knowledge."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from backend.evaluation.atomic_report import write_new_atomic
from backend.evaluation.check_library import validate_emitted_check_ids
from backend.evaluation.constrained import run_constrained_evaluation
from backend.evaluation.engineering_quality_v3 import (
    run_engineering_quality_v3,
)
from backend.evaluation.evidence import display_path, secret_values, verify_frozen_file
from backend.evaluation.extensibility_probe import run_extensibility_probe
from backend.evaluation.objective_v3_schemas import (
    ConstrainedEvaluationConfigV3,
    ConstrainedEvaluationRunV3,
    ExtensibilityProbeResult,
    V3EvaluationSummary,
)
from backend.evaluation.planning import load_evaluation_planning_run
from backend.evaluation.schemas import CheckStatus
from backend.llm import LLMClient


class V3EvaluationPhaseError(RuntimeError):
    """Internal phase marker used only to persist an accurate failure report."""

    def __init__(
        self, phase: str, error: Exception, partial: dict[str, Any] | None = None
    ):
        super().__init__(f"{type(error).__name__}: {error}")
        self.phase = phase
        self.original = error
        self.partial = partial


def run_constrained_evaluation_v3(
    config: ConstrainedEvaluationConfigV3,
    *,
    repository_root: Path,
    config_file: Path,
    run_dir: Path,
    llm_client: LLMClient | None,
) -> ConstrainedEvaluationRunV3:
    """Run unchanged v2 constraints/objectives, then secondary MERODA assessment."""

    started = time.perf_counter()
    root = repository_root.resolve()
    try:
        _verify_evaluation_planning(config, root)
    except Exception as error:
        raise V3EvaluationPhaseError("planning", error) from error
    try:
        deterministic = run_constrained_evaluation(
            config.as_v2(),
            repository_root=root,
            config_file=config_file,
            run_dir=run_dir,
            execution_policy=config.execution_policy,
        )
    except Exception as error:
        raise V3EvaluationPhaseError("deterministic", error) from error
    if config.check_plan is None:
        raise V3EvaluationPhaseError(
            "deterministic",
            ValueError("Validated v3 suites require a resolved evaluation check plan."),
            deterministic.model_dump(mode="json"),
        )
    try:
        validate_emitted_check_ids(
            config.check_plan,
            {
                check.check_id
                for constraint in deterministic.constraints
                for check in constraint.checks
            },
        )
    except ValueError as error:
        raise V3EvaluationPhaseError(
            "deterministic",
            error,
            deterministic.model_dump(mode="json"),
        ) from error
    if deterministic.summary.feasible:
        probe = run_extensibility_probe(
            config,
            repository_root=root,
            run_dir=run_dir,
        )
    else:
        probe = ExtensibilityProbeResult(
            status=CheckStatus.NOT_ASSESSED,
            required=(
                True
                if config.extensibility_probe is None
                else config.extensibility_probe.required
            ),
            suite_path=(
                None
                if config.extensibility_probe is None
                else config.extensibility_probe.suite.path
            ),
            feedback_code="EXTENSIBILITY_BLOCKED_BY_CONSTRAINTS",
            error="The alternate-contract probe requires a feasible primary candidate.",
        )

    deterministic_evidence = _judge_evidence(deterministic, probe)
    evidence_path = run_dir / "deterministic_evidence.v3.json"
    write_new_atomic(
        evidence_path,
        json.dumps(deterministic_evidence, indent=2, sort_keys=True) + "\n",
    )
    review_paths = [*config.candidate_source.paths, display_path(evidence_path, root)]
    secrets = _secret_values(config)
    try:
        engineering = run_engineering_quality_v3(
            client=llm_client,
            settings=config.engineering_quality,
            target_name=config.candidate_id,
            review_paths=review_paths,
            deterministic_evidence=deterministic_evidence,
            repository_root=root,
            redact_values=secrets,
            blind_values=[config.candidate_source.cwd],
            extensibility_score_cap=probe.score_cap,
        )
    except Exception as error:
        partial = {
            "deterministic": deterministic.model_dump(mode="json"),
            "extensibility_probe": probe.model_dump(mode="json"),
        }
        raise V3EvaluationPhaseError("judge", error, partial) from error

    operational = (
        dict(deterministic.summary.objectives) if deterministic.summary.feasible else {}
    )
    engineering_objective = (
        {"q_engineering": engineering.q_engineering}
        if deterministic.summary.feasible
        and engineering.engineering_assessed
        and engineering.q_engineering is not None
        else {}
    )
    diagnostic_engineering_quality = (
        {"q_engineering": engineering.q_engineering}
        if engineering.engineering_assessed
        and engineering.q_engineering is not None
        else {}
    )
    vector_complete = (
        deterministic.summary.feasible and engineering.engineering_assessed
    )
    optimization_ready = bool(vector_complete)
    thesis_evidence_ready = bool(
        vector_complete
        and config.engineering_quality.mode == "thesis"
        and config.engineering_quality.repetitions >= 3
        and engineering.valid_judgments == config.engineering_quality.repetitions
        and not engineering.advisory
        and _full_isolation(deterministic)
        and (not probe.required or probe.status == CheckStatus.PASS)
    )
    feedback_codes = list(
        dict.fromkeys(
            [
                *deterministic.summary.feedback_codes,
                probe.feedback_code,
                *(
                    code
                    for component in engineering.components
                    for code in component.feedback_codes
                ),
            ]
        )
    )
    summary = V3EvaluationSummary(
        target=config.candidate_id,
        constraints={
            name: status for name, status in deterministic.summary.constraints.items()
        },
        feasible=deterministic.summary.feasible,
        engineering_assessed=engineering.engineering_assessed,
        objective_vector_complete=vector_complete,
        optimization_ready=optimization_ready,
        thesis_evidence_ready=thesis_evidence_ready,
        diagnostic_operational_metrics=dict(
            deterministic.summary.diagnostic_metrics
        ),
        diagnostic_engineering_quality=diagnostic_engineering_quality,
        operational_objectives=operational,
        engineering_objective=engineering_objective,
        engineering_profile=config.engineering_quality.profile,
        feedback_codes=feedback_codes,
    )
    return ConstrainedEvaluationRunV3(
        suite_id=config.suite_id,
        suite_version=config.suite_version,
        run_id=run_dir.name,
        config_file=display_path(config_file, root),
        check_plan=config.check_plan,
        deterministic=deterministic,
        extensibility_probe=probe,
        engineering_quality=engineering,
        summary=summary,
        duration_seconds=time.perf_counter() - started,
    )


def _judge_evidence(
    deterministic: Any, probe: ExtensibilityProbeResult
) -> dict[str, Any]:
    """Bounded neutral facts; candidate identity and generation strategy are omitted."""

    return {
        "hard_constraints": {
            item.constraint: {
                "status": item.status.value,
                "checks": [
                    {
                        "check_id": check.check_id,
                        "status": check.status.value,
                        "feedback_code": check.feedback_code,
                        "summary": check.summary,
                    }
                    for check in item.checks
                ],
            }
            for item in deterministic.constraints
        },
        "diagnostic_operational_metrics": dict(
            deterministic.summary.diagnostic_metrics
        ),
        "eligible_operational_objectives": dict(deterministic.summary.objectives),
        "rerun_timings": deterministic.summary.diagnostics.get("timings", {}),
        "resource_diagnostics": deterministic.summary.diagnostics.get("resources", {}),
        "extensibility_probe": probe.model_dump(mode="json"),
    }


def _full_isolation(deterministic: Any) -> bool:
    return bool(deterministic.scenarios) and all(
        scenario.execution.isolation is not None and scenario.execution.isolation.full
        for scenario in deterministic.scenarios
    )


def _secret_values(config: ConstrainedEvaluationConfigV3) -> list[str]:
    return secret_values(config.security.secret_environment_variables)


def _verify_evaluation_planning(
    config: ConstrainedEvaluationConfigV3,
    repository_root: Path,
) -> None:
    if config.evaluation_planning is None:
        return
    path = verify_frozen_file(config.evaluation_planning, repository_root)
    planning = load_evaluation_planning_run(path)
    if not planning.suite_ready:
        raise ValueError("Referenced evaluation planning run is not suite-ready")
    if config.evaluation_profile != planning.profile:
        raise ValueError("Suite profile differs from its evaluation planning run")
    if planning.engineering_enabled != config.engineering_quality.enabled:
        raise ValueError("Suite engineering setting differs from its planning run")
    probe_enabled = bool(
        config.extensibility_probe is not None
        and config.extensibility_probe.enabled
    )
    if probe_enabled and not planning.extensibility_probe_enabled:
        raise ValueError("Suite enables an extensibility probe absent from its plan")
