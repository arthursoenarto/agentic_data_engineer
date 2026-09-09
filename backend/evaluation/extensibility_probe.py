"""Evaluator-owned alternate-contract behavioral probe for MERODA E."""

from __future__ import annotations

import json
import time
from pathlib import Path

from backend.evaluation.atomic_report import write_new_atomic
from backend.evaluation.constrained import run_constrained_evaluation
from backend.evaluation.constrained_schemas import ConstrainedEvaluationConfig
from backend.evaluation.evidence import (
    canonical_json_file_hash,
    display_path,
    source_bundle_hash,
)
from backend.evaluation.objective_v3_schemas import (
    ConstrainedEvaluationConfigV3,
    ExtensibilityProbeResult,
)
from backend.evaluation.schemas import CheckStatus


def run_extensibility_probe(
    primary: ConstrainedEvaluationConfigV3,
    *,
    repository_root: Path,
    run_dir: Path,
) -> ExtensibilityProbeResult:
    """Run the same candidate against one frozen alternate valid lock and oracle."""

    spec = primary.extensibility_probe
    if spec is None:
        return ExtensibilityProbeResult(
            status=CheckStatus.NOT_ASSESSED,
            required=True,
            feedback_code="EXTENSIBILITY_PROBE_NOT_CONFIGURED",
            error="No alternate-contract behavioral probe is configured.",
        )
    if not spec.enabled:
        return ExtensibilityProbeResult(
            status=CheckStatus.NOT_ASSESSED,
            required=spec.required,
            suite_path=spec.suite.path,
            feedback_code="EXTENSIBILITY_PROBE_DISABLED",
            error="The frozen suite disables its extensibility probe.",
        )

    probe_dir = run_dir / "extensibility_probe"
    probe_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    same_source: bool | None = None
    same_inventory: bool | None = None
    source_unchanged: bool | None = None
    local_inputs: bool | None = None
    fresh_success: bool | None = None
    try:
        suite_path = _repo_path(spec.suite.path, repository_root)
        if canonical_json_file_hash(suite_path) != spec.suite.sha256:
            raise ValueError("Alternate probe suite hash mismatch")
        alternate = ConstrainedEvaluationConfig.model_validate_json(
            suite_path.read_text(encoding="utf-8")
        )
        same_source = alternate.candidate_source.model_dump(
            mode="json"
        ) == primary.candidate_source.model_dump(mode="json")
        same_inventory = alternate.inventory.sha256 == primary.inventory.sha256
        if not same_source:
            raise ValueError("Probe must use the exact same candidate source bundle")
        if not same_inventory:
            raise ValueError("Probe must use the same frozen dataset inventory")
        if alternate.contract_lock.sha256 == primary.contract_lock.sha256:
            raise ValueError("Probe contract lock must differ from the primary lock")

        before = source_bundle_hash(primary.candidate_source, repository_root)
        run = run_constrained_evaluation(
            alternate,
            repository_root=repository_root,
            config_file=suite_path,
            run_dir=probe_dir,
            execution_policy=primary.execution_policy,
        )
        after = source_bundle_hash(primary.candidate_source, repository_root)
        source_unchanged = before == after == primary.candidate_source.sha256
        grounding = next(
            (
                check
                for constraint in run.constraints
                for check in constraint.checks
                if check.check_id
                in {"contract.grounding", "contract.reference_integrity"}
                and check.status != CheckStatus.PASS
            ),
            None,
        )
        local_inputs = grounding is None
        fresh_success = run.summary.feasible
        write_new_atomic(
            probe_dir / "evaluation.json", run.model_dump_json(indent=2) + "\n"
        )
        write_new_atomic(
            probe_dir / "suite.json",
            json.dumps(alternate.model_dump(mode="json"), indent=2) + "\n",
        )
        passed = bool(
            run.summary.feasible
            and source_unchanged
            and same_source
            and same_inventory
            and local_inputs
        )
        return ExtensibilityProbeResult(
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            required=spec.required,
            suite_path=display_path(suite_path, repository_root),
            same_candidate_source=same_source,
            same_inventory=same_inventory,
            source_unchanged=source_unchanged,
            local_cache_and_oracle_valid=local_inputs,
            fresh_output_succeeded=fresh_success,
            deterministic_run_path=display_path(
                probe_dir / "evaluation.json", repository_root
            ),
            feedback_code=(
                "EXTENSIBILITY_ALTERNATE_CONTRACT_PASSED"
                if passed
                else "EXTENSIBILITY_ALTERNATE_CONTRACT_FAILED"
            ),
            error=None
            if passed
            else "The alternate-contract deterministic run was not feasible.",
        )
    except Exception as error:
        return ExtensibilityProbeResult(
            status=CheckStatus.ERROR,
            required=spec.required,
            suite_path=spec.suite.path,
            same_candidate_source=same_source,
            same_inventory=same_inventory,
            source_unchanged=source_unchanged,
            local_cache_and_oracle_valid=local_inputs,
            fresh_output_succeeded=fresh_success,
            feedback_code="EXTENSIBILITY_PROBE_ERROR",
            error=f"{type(error).__name__}: {error}; elapsed_seconds={time.perf_counter() - started:.6f}",
        )


def _repo_path(raw: str, repository_root: Path) -> Path:
    path = Path(raw)
    resolved = (
        path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    )
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"Probe suite escapes repository: {resolved}") from error
    return resolved
