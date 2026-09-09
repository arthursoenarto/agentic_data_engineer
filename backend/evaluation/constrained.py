"""Constrained multiobjective evaluation for regular-grid Zarr pipelines."""

from __future__ import annotations

import statistics
import time
import os
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError

from backend.evaluation.core_schemas import CommandExecutionPolicy
from backend.evaluation.constrained_schemas import (
    CompactEvaluationSummary,
    ConstrainedEvaluationConfig,
    ConstrainedEnvironment,
    ConstrainedEvaluationRun,
    ConstraintId,
    ConstraintResult,
    EvaluationCheck,
    FeedbackSignal,
    NeutralPipelineReceipt,
)
from backend.evaluation.controlled_materialization import (
    Grounding as _Grounding,
    Scenario as _Scenario,
    candidate_claim_mismatches as _candidate_claim_mismatches,
    run_scenario as _run_scenario,
    validate_grounding as _validate_grounding,
    validate_receipt as _validate_receipt,
)
from backend.evaluation.evidence import (
    canonical_json_file_hash,
    display_path as _display,
    frozen_path_hash,
    path_size as _path_size,
    redact as _redact,
    repo_path as _repo_path,
    require_within as _require_within,
    resolve_evidence_path as _resolve_evidence_path,
    scan_secret_files as _scan_secrets,
    secret_values,
    source_bundle_hash,
    source_files_hash,
)
from backend.evaluation.gridded_dataset import GriddedTensorDataset
from backend.evaluation.regular_grid_zarr import (
    ResolvedGrid,
    UnsupportedGridError,
    benchmark_target,
    compare_coordinates,
    compare_values,
    logical_fingerprint,
    logical_uncompressed_bytes,
    resolve_regular_grid,
    inspect_zarr_storage,
    validate_metadata,
    validate_zarr_integrity,
)
from backend.evaluation.schemas import CheckStatus
from backend.evaluation.tensor_loading import (
    TensorLoadingProtocol,
    measure_tensor_loading,
)


def run_constrained_evaluation(
    config: ConstrainedEvaluationConfig,
    *,
    repository_root: Path,
    config_file: Path,
    run_dir: Path,
    execution_policy: CommandExecutionPolicy | None = None,
) -> ConstrainedEvaluationRun:
    """Run four hard constraints, then measure exactly three objectives."""

    started = time.perf_counter()
    root = repository_root.resolve()
    secrets = _secret_values(config)
    provenance: dict[str, str] = {
        "suite": canonical_json_file_hash(config_file),
        "evaluator_source_bundle": _evaluator_hash(),
        "contract_lock": config.contract_lock.sha256,
        "inventory": config.inventory.sha256,
        "pipeline_contract": config.execution.pipeline_contract.sha256,
        "manifest": config.execution.manifest.sha256,
        "candidate_source_bundle": config.candidate_source.sha256,
        "reference_artifact": config.reference.artifact.sha256,
        "reference_content_id": config.reference.content_id,
        "frozen_cache": config.execution.frozen_cache.sha256,
    }
    contract_checks: list[EvaluationCheck] = []
    semantic_checks: list[EvaluationCheck] = []
    rerun_checks: list[EvaluationCheck] = []
    provenance_checks: list[EvaluationCheck] = []
    diagnostics: dict[str, Any] = {}
    scenarios: list[_Scenario] = []
    reference: ResolvedGrid | None = None

    grounding: _Grounding | None = None
    grounding_started = time.perf_counter()
    try:
        grounding = _validate_grounding(config, root)
        contract_checks.append(
            _check(
                "contract.grounding",
                CheckStatus.PASS,
                "Contract lock, inventory, mappings, suite inputs, and controlled command are grounded.",
                "CONTRACT_GROUNDING_VALID",
                observed={
                    "field_ids": grounding.contract_ids,
                    "scope": grounding.lock_payload["contract"].get("scope", {}),
                },
                evidence=[
                    config.contract_lock.path,
                    config.inventory.path,
                    config.execution.pipeline_contract.path,
                    config.execution.manifest.path,
                ],
                duration_seconds=time.perf_counter() - grounding_started,
            )
        )
    except Exception as error:
        contract_checks.append(
            _check(
                "contract.grounding",
                _status_for(error),
                "Trusted-suite grounding failed.",
                "CONTRACT_GROUNDING_INVALID",
                observed={"error": _redact(str(error), secrets)},
                duration_seconds=time.perf_counter() - grounding_started,
            )
        )

    if grounding is not None:
        reference_started = time.perf_counter()
        try:
            reference = resolve_regular_grid(
                config.grid.reference_grid,
                config.grid.channels,
                candidate=False,
                repository_root=root,
                output_dir=None,
                expected_shape=config.grid.expected_shape,
            )
            _require_reference_paths(config, reference, root)
            arrays = validate_zarr_integrity(
                reference.store_paths,
                max_block_bytes=config.comparison.max_block_bytes,
            )
            contract_checks.append(
                _check(
                    "contract.reference_integrity",
                    CheckStatus.PASS,
                    "The frozen independent Zarr oracle is complete and block-readable.",
                    "REFERENCE_VALID",
                    observed={"arrays_read": arrays, "shape": list(reference.shape)},
                    evidence=[config.reference.artifact.path],
                    duration_seconds=time.perf_counter() - reference_started,
                )
            )
        except Exception as error:
            contract_checks.append(
                _check(
                    "contract.reference_integrity",
                    _status_for(error),
                    "The frozen independent oracle is unusable.",
                    "REFERENCE_INVALID",
                    observed={"error": _redact(str(error), secrets)},
                    duration_seconds=time.perf_counter() - reference_started,
                )
            )

    if grounding is not None and reference is not None:
        for scenario_name in ("initial", "rerun"):
            scenarios.append(
                _run_scenario(
                    scenario_name,
                    config,
                    grounding,
                    repository_root=root,
                    run_dir=run_dir,
                    secrets=secrets,
                    execution_policy=execution_policy,
                )
            )

    initial = scenarios[0] if len(scenarios) == 2 else None
    rerun = scenarios[1] if len(scenarios) == 2 else None
    if execution_policy is not None:
        provenance_checks.append(_isolation_check(scenarios))
    if initial is None:
        contract_checks.append(
            _check(
                "contract.initial_materialization",
                CheckStatus.NOT_ASSESSED,
                "Initial materialization was blocked by trusted-suite validation.",
                "MATERIALIZATION_BLOCKED",
            )
        )
    else:
        initial_checks, storage_diagnostics = _initial_contract_checks(config, initial)
        contract_checks.extend(initial_checks)
        if storage_diagnostics is not None:
            diagnostics["output_storage"] = storage_diagnostics

    if initial is not None and initial.grid is not None and reference is not None:
        coordinate_started = time.perf_counter()
        coordinate_result = compare_coordinates(
            initial.grid,
            reference,
            atol=config.comparison.coordinates_atol,
        )
        if coordinate_result["matched"]:
            contract_checks.append(
                _check(
                    "contract.scope_coordinates",
                    CheckStatus.PASS,
                    "Sample and spatial coordinates match the frozen oracle.",
                    "SCOPE_COORDINATES_VALID",
                    observed=coordinate_result,
                    duration_seconds=time.perf_counter() - coordinate_started,
                )
            )
            semantic_started = time.perf_counter()
            try:
                value_stats = compare_values(
                    initial.grid,
                    reference,
                    config.grid.channels,
                    default_policy=config.comparison.value_policy,
                    max_block_bytes=config.comparison.max_block_bytes,
                    mismatch_limit=config.comparison.mismatch_examples,
                )
                diagnostics["semantic_comparison"] = value_stats
                semantic_checks.append(
                    _check(
                        "semantic.native_values_and_missingness",
                        CheckStatus.PASS
                        if value_stats["matched"]
                        else CheckStatus.FAIL,
                        (
                            "All native values and missingness markers match the oracle."
                            if value_stats["matched"]
                            else "Native values or missingness markers differ from the oracle."
                        ),
                        "SEMANTIC_EQUIVALENT"
                        if value_stats["matched"]
                        else "SEMANTIC_MISMATCH",
                        observed=value_stats,
                        delta={
                            "mismatch_count": value_stats["mismatch_count"],
                            "missingness_mismatch_count": value_stats[
                                "missingness_mismatch_count"
                            ],
                            "max_finite_absolute_error": value_stats[
                                "max_finite_absolute_error"
                            ],
                            "max_finite_relative_error": value_stats[
                                "max_finite_relative_error"
                            ],
                        },
                        affected_slice=(
                            {
                                "field_id": value_stats["examples"][0]["field_id"],
                                **value_stats["examples"][0]["index"],
                            }
                            if value_stats["examples"]
                            else None
                        ),
                        evidence=[
                            config.reference.artifact.path,
                            initial.result.output_path,
                        ],
                        duration_seconds=time.perf_counter() - semantic_started,
                    )
                )
            except Exception as error:
                semantic_checks.append(
                    _check(
                        "semantic.native_values_and_missingness",
                        _status_for(error),
                        "Native semantic comparison could not be completed.",
                        "SEMANTIC_COMPARISON_FAILED",
                        observed={"error": _redact(str(error), secrets)},
                        duration_seconds=time.perf_counter() - semantic_started,
                    )
                )
        else:
            contract_checks.append(
                _check(
                    "contract.scope_coordinates",
                    CheckStatus.FAIL,
                    "Sample or spatial coordinates differ from the frozen oracle.",
                    "SCOPE_COORDINATE_MISMATCH",
                    observed=coordinate_result,
                    affected_slice=(
                        coordinate_result["mismatches"][0]
                        if coordinate_result["mismatches"]
                        else None
                    ),
                    evidence=[
                        config.reference.artifact.path,
                        initial.result.output_path,
                    ],
                    duration_seconds=time.perf_counter() - coordinate_started,
                )
            )
            semantic_checks.append(
                _check(
                    "semantic.native_values_and_missingness",
                    CheckStatus.NOT_ASSESSED,
                    "Semantic comparison requires matching logical coordinates.",
                    "SEMANTIC_BLOCKED_BY_SCOPE",
                )
            )
    else:
        semantic_checks.append(
            _check(
                "semantic.native_values_and_missingness",
                CheckStatus.NOT_ASSESSED,
                "Semantic comparison requires valid candidate and reference grids.",
                "SEMANTIC_BLOCKED_BY_STRUCTURE",
            )
        )

    rerun_result_checks, logical_hashes = _rerun_checks(
        config, initial, rerun, secrets=secrets
    )
    rerun_checks.extend(rerun_result_checks)
    provenance.update(logical_hashes)
    provenance_checks.append(
        _provenance_identity_check(
            grounding=grounding,
            provenance=provenance,
        )
    )
    if grounding is not None:
        provenance_checks.append(
            _trusted_inputs_unchanged_check(
                config,
                config_file=config_file,
                repository_root=root,
                expected=provenance,
                secrets=secrets,
            )
        )
        provenance_checks.append(_candidate_claims_check(config, grounding))
        receipt_checks, cache_diagnostics = _provenance_checks(
            config,
            grounding,
            scenarios,
            secrets=secrets,
        )
        provenance_checks.extend(receipt_checks)
        if cache_diagnostics:
            diagnostics["cache_evidence"] = cache_diagnostics
    else:
        provenance_checks.append(
            _check(
                "provenance.receipts",
                CheckStatus.NOT_ASSESSED,
                "Receipt provenance requires valid trusted-suite grounding.",
                "PROVENANCE_BLOCKED_BY_GROUNDING",
            )
        )

    secret_scan_started = time.perf_counter()
    scan_paths = _secret_scan_paths(config, scenarios, root)
    findings = _scan_secrets(
        scan_paths,
        known_values=secrets,
        max_bytes=config.security.max_text_file_bytes,
    )
    for scenario in scenarios:
        if scenario.result.execution.redactions_applied:
            findings.append(
                {
                    "path": scenario.result.execution.stdout_log,
                    "reason": "known_secret_redacted_from_command_log",
                }
            )
    provenance_checks.append(
        _check(
            "security.secret_scan",
            CheckStatus.PASS if not findings else CheckStatus.FAIL,
            (
                "Candidate sources, output metadata, receipts, and logs contain no detected secrets."
                if not findings
                else "Potential secret material was detected in evaluation evidence."
            ),
            "SECRET_SCAN_CLEAN" if not findings else "SECRET_MATERIAL_DETECTED",
            observed={"files_scanned": len(scan_paths), "findings": findings},
            evidence=[item["path"] for item in findings],
            duration_seconds=time.perf_counter() - secret_scan_started,
        )
    )

    constraint_results = [
        _constraint("contract_correctness", contract_checks),
        _constraint("semantic_equivalence", semantic_checks),
        _constraint("rerun_safety", rerun_checks),
        _constraint("provenance_security", provenance_checks),
    ]

    workload: dict[str, Any] | None = None
    if initial is not None and initial.grid is not None:
        workload_started = time.perf_counter()
        try:
            workload = _measure_consumer(config, initial.grid, initial.output_dir, root)
            diagnostics["consumer_workload"] = workload
            contract_checks.append(
                _check(
                    "contract.pytorch_consumer",
                    CheckStatus.PASS,
                    "The output is consumable as warm shuffled float32 [C,H,W] tensors.",
                    "PYTORCH_CONSUMER_VALID",
                    observed={
                        "shape": workload["sample_shape"],
                        "dtype": workload["dtype"],
                    },
                    duration_seconds=time.perf_counter() - workload_started,
                )
            )
        except Exception as error:
            contract_checks.append(
                _check(
                    "contract.pytorch_consumer",
                    _status_for(error),
                    "The primary PyTorch consumer workload failed.",
                    "PYTORCH_CONSUMER_FAILED",
                    observed={"error": _redact(str(error), secrets)},
                    duration_seconds=time.perf_counter() - workload_started,
                )
            )
        constraint_results[0] = _constraint("contract_correctness", contract_checks)

    diagnostic_metrics: dict[str, int | float] = {}
    if initial is not None:
        artifact_bytes, artifact_files = (
            _path_size(initial.output_dir)
            if initial.output_dir.exists()
            else (0, 0)
        )
        diagnostics["candidate_artifact"] = {
            "execution_succeeded": (
                initial.result.execution.exit_code == 0
                and not initial.result.execution.timed_out
            ),
            "mapped_zarr_readable": initial.grid is not None,
            "output_directory_bytes": artifact_bytes,
            "output_directory_files": artifact_files,
        }
    if initial is not None and initial.grid is not None:
        output_bytes, output_files = _unique_path_size(initial.grid.store_paths)
        storage_diagnostics = diagnostics.get("output_storage")
        if isinstance(storage_diagnostics, dict):
            diagnosed_bytes = int(storage_diagnostics["output_bytes"])
            if diagnosed_bytes != output_bytes:
                raise ValueError(
                    "Storage diagnostics and native output footprint disagree."
                )
        diagnostic_metrics = {
            "materialization_seconds": initial.result.execution.duration_seconds,
            "output_bytes": output_bytes,
        }
        if workload is not None:
            diagnostic_metrics["consumer_samples_per_second"] = workload[
                "median_samples_per_second"
            ]
        diagnostics["output"] = {
            "file_count": output_files,
            "chunk_bytes": (
                storage_diagnostics.get("chunk_bytes")
                if isinstance(storage_diagnostics, dict)
                else None
            ),
            "metadata_bytes": (
                storage_diagnostics.get("metadata_bytes")
                if isinstance(storage_diagnostics, dict)
                else None
            ),
            "object_count": (
                storage_diagnostics.get("object_count")
                if isinstance(storage_diagnostics, dict)
                else output_files
            ),
            "dataset_open_latency_seconds": (
                storage_diagnostics.get("dataset_open_latency_seconds")
                if isinstance(storage_diagnostics, dict)
                else None
            ),
            "logical_uncompressed_bytes": logical_uncompressed_bytes(
                initial.grid, config.grid.channels
            ),
        }

    feasible = all(item.passed for item in constraint_results) and workload is not None
    objectives: dict[str, int | float] = (
        dict(diagnostic_metrics) if feasible else {}
    )

    diagnostics["timings"] = {
        scenario.result.scenario: {
            "setup_seconds": scenario.result.setup_seconds,
            "execution_seconds": scenario.result.execution.duration_seconds,
            "validation_seconds": scenario.result.validation_seconds,
            "total_seconds": scenario.result.total_seconds,
        }
        for scenario in scenarios
    }
    diagnostics["resources"] = {
        scenario.result.scenario: scenario.result.execution.resources.model_dump(
            mode="json"
        )
        for scenario in scenarios
    }
    if reference is not None:
        reference.close()
    for scenario in scenarios:
        if scenario.grid is not None:
            scenario.grid.close()

    feedback = _feedback(constraint_results)
    summary = CompactEvaluationSummary(
        target=config.candidate_id,
        constraints={item.constraint: item.status for item in constraint_results},
        feasible=feasible,
        diagnostic_metrics=diagnostic_metrics,
        objectives=objectives,
        diagnostics=diagnostics,
        feedback_codes=list(dict.fromkeys(item.code for item in feedback)),
        feedback=feedback,
    )
    run = ConstrainedEvaluationRun(
        suite_id=config.suite_id,
        suite_version=config.suite_version,
        run_id=run_dir.name,
        config_file=_display(config_file, root),
        environment=_environment(),
        constraints=constraint_results,
        scenarios=[scenario.result for scenario in scenarios],
        provenance=provenance,
        summary=summary,
        duration_seconds=time.perf_counter() - started,
    )
    return _redacted_run(run, secrets)


def _initial_contract_checks(
    config: ConstrainedEvaluationConfig,
    initial: _Scenario,
) -> tuple[list[EvaluationCheck], dict[str, Any] | None]:
    checks = [
        _check(
            "contract.initial_execution",
            CheckStatus.PASS
            if initial.result.execution.exit_code == 0
            and not initial.result.execution.timed_out
            else CheckStatus.FAIL,
            (
                "The evaluator-controlled initial materialization succeeded."
                if initial.result.execution.exit_code == 0
                and not initial.result.execution.timed_out
                else "The evaluator-controlled initial materialization failed."
            ),
            "INITIAL_EXECUTION_VALID"
            if initial.result.execution.exit_code == 0
            and not initial.result.execution.timed_out
            else "INITIAL_EXECUTION_FAILED",
            observed={
                "exit_code": initial.result.execution.exit_code,
                "timed_out": initial.result.execution.timed_out,
            },
            evidence=[
                initial.result.execution.stdout_log,
                initial.result.execution.stderr_log,
            ],
            duration_seconds=initial.result.execution.duration_seconds,
        )
    ]
    if initial.grid is None:
        policy_failure = initial.policy_failure
        checks.append(
            _check(
                (
                    "contract.public_output_policy"
                    if policy_failure
                    else "contract.zarr_integrity"
                ),
                (
                    CheckStatus.NOT_ASSESSED
                    if initial.unsupported
                    else CheckStatus.FAIL
                ),
                (
                    "Candidate output violates the generation-visible output policy."
                    if policy_failure
                    else "Candidate output is outside the supported class or failed Zarr validation."
                ),
                (
                    "PUBLIC_OUTPUT_POLICY_INVALID"
                    if policy_failure
                    else (
                        "UNSUPPORTED_GRID_CLASS"
                        if initial.unsupported
                        else "ZARR_INTEGRITY_FAILED"
                    )
                ),
                observed={"error": initial.validation_error},
                evidence=[initial.result.output_path],
                duration_seconds=initial.result.validation_seconds,
            )
        )
        if config.output_policy is not None:
            checks.append(
                _check(
                    "contract.zarr_output_policy",
                    CheckStatus.NOT_ASSESSED,
                    "The Zarr output policy requires a valid mapped candidate store.",
                    "ZARR_OUTPUT_POLICY_BLOCKED",
                )
            )
        return checks, None
    checks.append(
        _check(
            "contract.zarr_integrity",
            CheckStatus.PASS,
            "All mapped candidate Zarr stores are complete and block-readable.",
            "ZARR_INTEGRITY_VALID",
            observed={"shape": list(initial.grid.shape)},
            evidence=[initial.result.output_path],
            duration_seconds=initial.result.validation_seconds,
        )
    )
    storage_diagnostics: dict[str, Any] | None = None
    if config.output_policy is not None:
        storage_started = time.perf_counter()
        try:
            storage_diagnostics = inspect_zarr_storage(
                initial.grid.store_paths,
                expected_format=config.output_policy.format_version,
                require_consolidated=config.output_policy.consolidated_metadata,
                open_latency_repetitions=(
                    config.output_policy.open_latency_repetitions
                ),
            )
            checks.append(
                _check(
                    "contract.zarr_output_policy",
                    CheckStatus.PASS,
                    "Mapped outputs use Zarr format 3 with complete consolidated metadata.",
                    "ZARR_V3_CONSOLIDATED_VALID",
                    observed=storage_diagnostics,
                    evidence=[initial.result.output_path],
                    duration_seconds=time.perf_counter() - storage_started,
                )
            )
        except Exception as error:
            checks.append(
                _check(
                    "contract.zarr_output_policy",
                    _status_for(error),
                    "Mapped output violates the fixed Zarr v3 consolidated policy.",
                    "ZARR_V3_CONSOLIDATED_INVALID",
                    observed={"error": str(error)},
                    evidence=[initial.result.output_path],
                    duration_seconds=time.perf_counter() - storage_started,
                )
            )
    metadata_started = time.perf_counter()
    metadata = validate_metadata(initial.grid, config.grid.channels)
    checks.append(
        _check(
            "contract.declared_metadata",
            CheckStatus.PASS if not metadata else CheckStatus.FAIL,
            "All declared dtype and metadata assertions pass."
            if not metadata
            else "A declared dtype or metadata assertion failed.",
            "DECLARED_METADATA_VALID" if not metadata else "DECLARED_METADATA_MISMATCH",
            observed={"mismatches": metadata},
            affected_slice=(
                {"field_id": metadata[0]["field_id"]} if metadata else None
            ),
            evidence=[initial.result.output_path],
            duration_seconds=time.perf_counter() - metadata_started,
        )
    )
    return checks, storage_diagnostics


def _rerun_checks(
    config: ConstrainedEvaluationConfig,
    initial: _Scenario | None,
    rerun: _Scenario | None,
    *,
    secrets: list[str],
) -> tuple[list[EvaluationCheck], dict[str, str]]:
    if initial is None or rerun is None:
        return (
            [
                _check(
                    "rerun.logical_equivalence",
                    CheckStatus.NOT_ASSESSED,
                    "Controlled rerun requires two completed materialization scenarios.",
                    "RERUN_BLOCKED",
                )
            ],
            {},
        )
    checks = [
        _check(
            "rerun.execution",
            CheckStatus.PASS
            if rerun.result.execution.exit_code == 0
            and not rerun.result.execution.timed_out
            else CheckStatus.FAIL,
            "The exact controlled rerun succeeded."
            if rerun.result.execution.exit_code == 0
            and not rerun.result.execution.timed_out
            else "The exact controlled rerun failed.",
            "RERUN_EXECUTION_VALID"
            if rerun.result.execution.exit_code == 0
            and not rerun.result.execution.timed_out
            else "RERUN_EXECUTION_FAILED",
            observed={
                "exit_code": rerun.result.execution.exit_code,
                "timed_out": rerun.result.execution.timed_out,
                "duration_seconds": rerun.result.execution.duration_seconds,
            },
            evidence=[
                rerun.result.execution.stdout_log,
                rerun.result.execution.stderr_log,
            ],
            duration_seconds=rerun.result.execution.duration_seconds,
        )
    ]
    if initial.grid is None or rerun.grid is None:
        policy_failure = initial.policy_failure or rerun.policy_failure
        checks.append(
            _check(
                (
                    "rerun.public_output_policy"
                    if policy_failure
                    else "rerun.logical_equivalence"
                ),
                CheckStatus.NOT_ASSESSED if rerun.unsupported else CheckStatus.FAIL,
                (
                    "A controlled output violates the generation-visible output policy."
                    if policy_failure
                    else "Both outputs must be valid Zarr grids before fingerprint comparison."
                ),
                (
                    "RERUN_PUBLIC_OUTPUT_POLICY_INVALID"
                    if policy_failure
                    else (
                        "RERUN_UNSUPPORTED_GRID"
                        if rerun.unsupported
                        else "RERUN_OUTPUT_INVALID"
                    )
                ),
                observed={"error": rerun.validation_error},
                evidence=[initial.result.output_path, rerun.result.output_path],
            )
        )
        return checks, {}
    if config.output_policy is not None:
        storage_started = time.perf_counter()
        try:
            observed_storage = inspect_zarr_storage(
                rerun.grid.store_paths,
                expected_format=config.output_policy.format_version,
                require_consolidated=config.output_policy.consolidated_metadata,
                open_latency_repetitions=1,
            )
            checks.append(
                _check(
                    "rerun.zarr_output_policy",
                    CheckStatus.PASS,
                    "The rerun preserves the Zarr v3 consolidated output policy.",
                    "RERUN_ZARR_V3_CONSOLIDATED_VALID",
                    observed={
                        "zarr_format": observed_storage["zarr_format"],
                        "consolidated_metadata": observed_storage[
                            "consolidated_metadata"
                        ],
                    },
                    evidence=[rerun.result.output_path],
                    duration_seconds=time.perf_counter() - storage_started,
                )
            )
        except Exception as error:
            checks.append(
                _check(
                    "rerun.zarr_output_policy",
                    _status_for(error),
                    "The rerun violates the fixed Zarr v3 consolidated policy.",
                    "RERUN_ZARR_V3_CONSOLIDATED_INVALID",
                    observed={"error": _redact(str(error), secrets)},
                    evidence=[rerun.result.output_path],
                    duration_seconds=time.perf_counter() - storage_started,
                )
            )
    metadata_started = time.perf_counter()
    metadata = validate_metadata(rerun.grid, config.grid.channels)
    checks.append(
        _check(
            "rerun.declared_metadata",
            CheckStatus.PASS if not metadata else CheckStatus.FAIL,
            (
                "The rerun preserves all declared dtype and metadata assertions."
                if not metadata
                else "The rerun changed a declared dtype or metadata assertion."
            ),
            ("RERUN_METADATA_VALID" if not metadata else "RERUN_METADATA_MISMATCH"),
            observed={"mismatches": metadata},
            affected_slice=(
                {"field_id": metadata[0]["field_id"]} if metadata else None
            ),
            evidence=[rerun.result.output_path],
            duration_seconds=time.perf_counter() - metadata_started,
        )
    )
    fingerprint_started = time.perf_counter()
    try:
        first = logical_fingerprint(
            initial.grid,
            config.grid.channels,
            default_policy=config.comparison.value_policy,
            max_block_bytes=config.comparison.max_block_bytes,
        )
        second = logical_fingerprint(
            rerun.grid,
            config.grid.channels,
            default_policy=config.comparison.value_policy,
            max_block_bytes=config.comparison.max_block_bytes,
        )
        checks.append(
            _check(
                "rerun.logical_equivalence",
                CheckStatus.PASS if first == second else CheckStatus.FAIL,
                "Fresh outputs have identical evaluator-owned logical fingerprints."
                if first == second
                else "Fresh outputs have different evaluator-owned logical fingerprints.",
                "RERUN_EQUIVALENT" if first == second else "RERUN_FINGERPRINT_MISMATCH",
                expected={"logical_fingerprint": first},
                observed={"logical_fingerprint": second},
                evidence=[initial.result.output_path, rerun.result.output_path],
                duration_seconds=time.perf_counter() - fingerprint_started,
            )
        )
        fingerprints = {
            "logical_output_initial": first,
            "logical_output_rerun": second,
        }
    except Exception as error:
        checks.append(
            _check(
                "rerun.logical_equivalence",
                _status_for(error),
                "Logical rerun fingerprints could not be computed.",
                "RERUN_FINGERPRINT_FAILED",
                observed={"error": _redact(str(error), secrets)},
                duration_seconds=time.perf_counter() - fingerprint_started,
            )
        )
        fingerprints = {}
    return checks, fingerprints


def _provenance_identity_check(
    *,
    grounding: _Grounding | None,
    provenance: dict[str, str],
) -> EvaluationCheck:
    required = {
        "suite",
        "evaluator_source_bundle",
        "contract_lock",
        "inventory",
        "pipeline_contract",
        "manifest",
        "candidate_source_bundle",
        "reference_artifact",
        "reference_content_id",
        "frozen_cache",
        "logical_output_initial",
        "logical_output_rerun",
    }
    missing = sorted(required - set(provenance))
    passed = grounding is not None and not missing
    return _check(
        "provenance.identities",
        CheckStatus.PASS if passed else CheckStatus.NOT_ASSESSED,
        (
            "All required trusted-input and logical-output identities are recorded."
            if passed
            else "Required provenance identities could not all be established."
        ),
        "PROVENANCE_IDENTITIES_VALID" if passed else "PROVENANCE_IDENTITIES_MISSING",
        expected={"required": sorted(required)},
        observed={"recorded": sorted(provenance), "missing": missing},
    )


def _isolation_check(scenarios: list[_Scenario]) -> EvaluationCheck:
    """Require every v3 candidate invocation to have full local isolation."""

    evidence = [scenario.result.execution.isolation for scenario in scenarios]
    complete = bool(evidence) and all(item is not None and item.full for item in evidence)
    unsupported = [
        {
            "scenario": scenario.result.scenario,
            "isolation": (
                None
                if scenario.result.execution.isolation is None
                else scenario.result.execution.isolation.model_dump(mode="json")
            ),
        }
        for scenario in scenarios
        if scenario.result.execution.isolation is None
        or not scenario.result.execution.isolation.full
    ]
    return _check(
        "security.execution_isolation",
        CheckStatus.PASS if complete else CheckStatus.NOT_ASSESSED,
        (
            "All candidate executions were network-disabled, read-only outside evaluator-owned writable roots, and secret-free by construction."
            if complete
            else "The host could not establish every required execution-isolation property."
        ),
        "EXECUTION_ISOLATION_VALID" if complete else "EXECUTION_ISOLATION_UNSUPPORTED",
        observed={"scenarios": unsupported if unsupported else len(evidence)},
    )


def _provenance_checks(
    config: ConstrainedEvaluationConfig,
    grounding: _Grounding,
    scenarios: list[_Scenario],
    *,
    secrets: list[str],
) -> tuple[list[EvaluationCheck], dict[str, Any]]:
    if not scenarios:
        return (
            [
                _check(
                    "provenance.receipts",
                    CheckStatus.NOT_ASSESSED,
                    "No controlled executions were available for receipt validation.",
                    "RECEIPTS_BLOCKED",
                )
            ],
            {},
        )
    checks: list[EvaluationCheck] = []
    cache_diagnostics: dict[str, Any] = {}
    for scenario in scenarios:
        receipt_started = time.perf_counter()
        try:
            receipt = NeutralPipelineReceipt.model_validate_json(
                scenario.receipt_path.read_text(encoding="utf-8")
            )
            _validate_receipt(
                receipt,
                scenario,
                grounding,
                config,
            )
            cache_diagnostics[scenario.result.scenario] = {
                key: receipt.cache.get(key)
                for key in (
                    "hits",
                    "misses",
                    "acquired",
                    "reused_keys",
                    "acquired_keys",
                )
            }
            checks.append(
                _check(
                    f"provenance.receipt.{scenario.result.scenario}",
                    CheckStatus.PASS,
                    f"The {scenario.result.scenario} receipt has valid linked identities.",
                    "RECEIPT_VALID",
                    observed={"cache": cache_diagnostics[scenario.result.scenario]},
                    evidence=[scenario.result.receipt_path],
                    duration_seconds=time.perf_counter() - receipt_started,
                )
            )
        except (OSError, ValueError, ValidationError) as error:
            checks.append(
                _check(
                    f"provenance.receipt.{scenario.result.scenario}",
                    CheckStatus.FAIL,
                    f"The {scenario.result.scenario} receipt is missing or invalid.",
                    "RECEIPT_INVALID",
                    observed={"error": _redact(str(error), secrets)},
                    evidence=[scenario.result.receipt_path],
                    duration_seconds=time.perf_counter() - receipt_started,
                )
            )
    return checks, cache_diagnostics


def _candidate_claims_check(
    config: ConstrainedEvaluationConfig,
    grounding: _Grounding,
) -> EvaluationCheck:
    mismatches = _candidate_claim_mismatches(config, grounding)
    return _check(
        "provenance.candidate_claims",
        CheckStatus.PASS if not mismatches else CheckStatus.FAIL,
        (
            "Candidate-authored manifest and pipeline-contract claims are linked and consistent."
            if not mismatches
            else "Candidate-authored provenance claims conflict with trusted observations."
        ),
        "CANDIDATE_CLAIMS_VALID" if not mismatches else "CANDIDATE_CLAIMS_INVALID",
        expected={
            "candidate_id": config.candidate_id,
            "output_format": config.grid.output_format,
        },
        observed={"mismatched_claims": mismatches},
        evidence=[
            config.execution.pipeline_contract.path,
            config.execution.manifest.path,
        ],
    )


def _trusted_inputs_unchanged_check(
    config: ConstrainedEvaluationConfig,
    *,
    config_file: Path,
    repository_root: Path,
    expected: dict[str, str],
    secrets: list[str],
) -> EvaluationCheck:
    started = time.perf_counter()
    try:
        observed = {
            "suite": canonical_json_file_hash(config_file),
            "evaluator_source_bundle": _evaluator_hash(),
            "contract_lock": canonical_json_file_hash(
                _repo_path(config.contract_lock.path, repository_root)
            ),
            "inventory": canonical_json_file_hash(
                _repo_path(config.inventory.path, repository_root)
            ),
            "pipeline_contract": canonical_json_file_hash(
                _repo_path(config.execution.pipeline_contract.path, repository_root)
            ),
            "manifest": canonical_json_file_hash(
                _repo_path(config.execution.manifest.path, repository_root)
            ),
            "candidate_source_bundle": source_bundle_hash(
                config.candidate_source, repository_root
            ),
            "reference_artifact": frozen_path_hash(
                _repo_path(config.reference.artifact.path, repository_root)
            ),
            "frozen_cache": frozen_path_hash(
                _repo_path(config.execution.frozen_cache.path, repository_root)
            ),
        }
        mismatches = sorted(
            name for name, value in observed.items() if expected.get(name) != value
        )
        return _check(
            "provenance.trusted_inputs_unchanged",
            CheckStatus.PASS if not mismatches else CheckStatus.FAIL,
            (
                "Frozen suite inputs and evaluator source remained unchanged during execution."
                if not mismatches
                else "A frozen suite input or evaluator source changed during execution."
            ),
            (
                "TRUSTED_INPUTS_UNCHANGED"
                if not mismatches
                else "TRUSTED_INPUT_TAMPERED"
            ),
            expected={"hashes": {name: expected[name] for name in observed}},
            observed={"hashes": observed, "mismatches": mismatches},
            duration_seconds=time.perf_counter() - started,
        )
    except Exception as error:
        return _check(
            "provenance.trusted_inputs_unchanged",
            CheckStatus.FAIL,
            "Frozen input identities could not be revalidated after execution.",
            "TRUSTED_INPUT_REVALIDATION_FAILED",
            observed={"error": _redact(str(error), secrets)},
            duration_seconds=time.perf_counter() - started,
        )


def _measure_consumer(
    config: ConstrainedEvaluationConfig,
    grid: ResolvedGrid,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    target = benchmark_target(
        grid,
        config.grid.channels,
        output_dir=output_dir,
        repository_root=repository_root,
    )
    dataset = GriddedTensorDataset(
        target,
        sample_limit=config.workload.max_samples,
    )
    try:
        count = len(dataset)
        expected_shape = (len(config.grid.channels), grid.shape[1], grid.shape[2])
        if config.workload.protocol == "tensor_loading.v1":
            class _DatasetReader:
                def sample(self, index: int) -> np.ndarray:
                    return dataset.read_numpy(index)

            measurement = measure_tensor_loading(
                _DatasetReader(),
                TensorLoadingProtocol(
                    sample_count=count,
                    sample_shape=expected_shape,
                    seed=config.workload.seed,
                    warmup_passes=config.workload.warmup_passes,
                    timed_passes=config.workload.repetitions,
                ),
            )
            summary = measurement["summary"]
            return {
                "kind": config.workload.kind,
                "protocol": measurement["protocol"],
                "access_pattern": config.workload.access_pattern,
                "effective_cache_mode": config.workload.cache_mode,
                "seed": config.workload.seed,
                "repetitions": config.workload.repetitions,
                "sample_shape": list(expected_shape),
                "dtype": "float32",
                "samples": count,
                "sample_order": measurement["sample_order"],
                "warmup": measurement["warmup"],
                "timed_passes": measurement["timed_passes"],
                "durations_seconds": [
                    item["seconds"] for item in measurement["timed_passes"]
                ],
                "samples_per_second": [
                    item["samples_per_second"]
                    for item in measurement["timed_passes"]
                ],
                "median_samples_per_second": summary[
                    "median_samples_per_second"
                ],
                "min_samples_per_second": summary[
                    "minimum_samples_per_second"
                ],
                "max_samples_per_second": summary[
                    "maximum_samples_per_second"
                ],
                "q1_samples_per_second": summary["q1_samples_per_second"],
                "q3_samples_per_second": summary["q3_samples_per_second"],
                "iqr_samples_per_second": summary["iqr_samples_per_second"],
            }
        order = np.random.default_rng(config.workload.seed).permutation(count).tolist()
        sample_shape: list[int] | None = None
        sample_dtype: str | None = None
        for _ in range(config.workload.warmup_passes):
            for index in order:
                tensor = dataset[index]
                sample_shape = list(tensor.shape)
                sample_dtype = str(tensor.dtype).replace("torch.", "")
                float(tensor.sum())
        expected_shape_list = list(expected_shape)
        if sample_shape != expected_shape_list or sample_dtype != "float32":
            raise ValueError(
                f"Consumer tensor contract differs: shape={sample_shape}, dtype={sample_dtype}"
            )
        rates: list[float] = []
        durations: list[float] = []
        for _ in range(config.workload.repetitions):
            measured_started = time.perf_counter()
            for index in order:
                float(dataset[index].sum())
            duration = time.perf_counter() - measured_started
            if duration <= 0:
                raise ValueError("Consumer workload duration was not measurable")
            durations.append(duration)
            rates.append(count / duration)
        return {
            "kind": config.workload.kind,
            "access_pattern": config.workload.access_pattern,
            "effective_cache_mode": config.workload.cache_mode,
            "seed": config.workload.seed,
            "repetitions": config.workload.repetitions,
            "sample_shape": sample_shape,
            "dtype": sample_dtype,
            "samples": count,
            "durations_seconds": durations,
            "samples_per_second": rates,
            "median_samples_per_second": statistics.median(rates),
            "min_samples_per_second": min(rates),
            "max_samples_per_second": max(rates),
        }
    finally:
        dataset.close()


def _require_reference_paths(
    config: ConstrainedEvaluationConfig,
    reference: ResolvedGrid,
    repository_root: Path,
) -> None:
    artifact = _repo_path(config.reference.artifact.path, repository_root)
    for store in reference.store_paths:
        _require_within(store, artifact, "Reference store")


def _unique_path_size(paths: set[Path]) -> tuple[int, int]:
    """Count each mapped native store once, excluding unrelated sidecars."""

    totals = [_path_size(path) for path in sorted({path.resolve() for path in paths})]
    return sum(size for size, _ in totals), sum(count for _, count in totals)


def _secret_values(config: ConstrainedEvaluationConfig) -> list[str]:
    return secret_values(config.security.secret_environment_variables)


def _secret_scan_paths(
    config: ConstrainedEvaluationConfig,
    scenarios: list[_Scenario],
    repository_root: Path,
) -> list[Path]:
    roots = [
        _repo_path(path, repository_root) for path in config.candidate_source.paths
    ]
    roots.extend(
        [
            _repo_path(config.execution.pipeline_contract.path, repository_root),
            _repo_path(config.execution.manifest.path, repository_root),
        ]
    )
    for scenario in scenarios:
        roots.extend([scenario.output_dir, scenario.receipt_path])
        roots.extend(
            [
                _resolve_evidence_path(
                    scenario.result.execution.stdout_log, repository_root
                ),
                _resolve_evidence_path(
                    scenario.result.execution.stderr_log, repository_root
                ),
            ]
        )
    files: set[Path] = set()
    for root in roots:
        if root.is_file():
            files.add(root)
        elif root.is_dir():
            files.update(item for item in root.rglob("*") if item.is_file())
    return sorted(files)


def _feedback(results: list[ConstraintResult]) -> list[FeedbackSignal]:
    signals: list[FeedbackSignal] = []
    for result in results:
        for check in result.checks:
            if check.status == CheckStatus.PASS:
                continue
            affected: list[str] = []
            if isinstance(check.observed, dict):
                raw = check.observed.get("affected_channels", [])
                if isinstance(raw, list):
                    affected = [str(value) for value in raw]
            signals.append(
                FeedbackSignal(
                    constraint=result.constraint,
                    code=check.feedback_code,
                    message=check.summary,
                    affected_channels=affected,
                )
            )
    return signals


def _constraint(
    constraint: ConstraintId, checks: list[EvaluationCheck]
) -> ConstraintResult:
    statuses = {check.status for check in checks}
    if CheckStatus.ERROR in statuses:
        status = CheckStatus.ERROR
    elif CheckStatus.FAIL in statuses:
        status = CheckStatus.FAIL
    elif CheckStatus.NOT_ASSESSED in statuses or not checks:
        status = CheckStatus.NOT_ASSESSED
    else:
        status = CheckStatus.PASS
    return ConstraintResult(
        constraint=constraint,
        status=status,
        passed=status == CheckStatus.PASS,
        checks=checks,
    )


def _check(
    check_id: str,
    status: CheckStatus,
    summary: str,
    feedback_code: str,
    *,
    expected: Any | None = None,
    observed: Any | None = None,
    delta: Any | None = None,
    affected_slice: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    duration_seconds: float | None = None,
) -> EvaluationCheck:
    severity: Literal["error", "warning", "info"] = (
        "info"
        if status == CheckStatus.PASS
        else "warning"
        if status == CheckStatus.NOT_ASSESSED
        else "error"
    )
    return EvaluationCheck(
        check_id=check_id,
        status=status,
        severity=severity,
        summary=summary,
        feedback_code=feedback_code,
        expected=expected,
        observed=observed,
        delta=delta,
        affected_slice=affected_slice,
        evidence=evidence or [],
        duration_seconds=duration_seconds,
    )


def _status_for(error: Exception) -> CheckStatus:
    return (
        CheckStatus.NOT_ASSESSED
        if isinstance(error, UnsupportedGridError)
        else CheckStatus.FAIL
    )


def _evaluator_hash() -> str:
    directory = Path(__file__).resolve().parent
    return source_files_hash(
        [
            directory / "constrained.py",
            directory / "constrained_schemas.py",
            directory / "constrained_workflow.py",
            directory / "core_schemas.py",
            directory / "controlled_materialization.py",
            directory / "evidence.py",
            directory / "regular_grid_zarr.py",
            directory / "command_runner.py",
            directory / "gridded_dataset.py",
            directory / "gridded_store.py",
        ]
    )


def _environment() -> ConstrainedEnvironment:
    packages = ("numpy", "pydantic", "psutil", "torch", "zarr")
    return ConstrainedEnvironment(
        platform=platform.system(),
        platform_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count(),
        software={name: version(name) for name in packages},
    )


def _redacted_run(
    run: ConstrainedEvaluationRun, secrets: list[str]
) -> ConstrainedEvaluationRun:
    payload = run.model_dump_json()
    redacted = _redact(payload, secrets)
    return ConstrainedEvaluationRun.model_validate_json(redacted)
