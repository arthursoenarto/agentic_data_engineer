"""Independent deterministic and MERODA evaluation for Terraio extensions."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from backend.evaluation.atomic_report import write_new_atomic
from backend.evaluation.command_runner import run_observed_command
from backend.evaluation.engineering_quality_v3 import run_engineering_quality_v3
from backend.evaluation.evidence import (
    display_path,
    scan_secret_files,
    secret_values,
    source_bundle_hash,
    verify_frozen_file,
    verify_frozen_path,
)
from backend.evaluation.isolation import prepare_command
from backend.evaluation.objective_v3_schemas import (
    ExtensionDeterministicCheck,
    NeutralExtensionCommand,
    NeutralExtensionContract,
    NeutralExtensionReceipt,
    TerraioExtensionEvaluationConfigV3,
    TerraioExtensionEvaluationRunV3,
    TerraioExtensionSummaryV3,
)
from backend.evaluation.schemas import CheckStatus
from backend.llm import LLMClient


def run_terraio_extension_evaluation_v3(
    config: TerraioExtensionEvaluationConfigV3,
    *,
    repository_root: Path,
    config_file: Path,
    run_dir: Path,
    llm_client: LLMClient | None,
) -> TerraioExtensionEvaluationRunV3:
    """Validate frozen extension evidence, then assess all six MERODA dimensions."""

    started = time.perf_counter()
    root = repository_root.resolve()
    checks: list[ExtensionDeterministicCheck] = []
    contract: NeutralExtensionContract | None = None
    receipt: NeutralExtensionReceipt | None = None
    paths: dict[str, Path] = {}
    try:
        paths = {
            "contract": verify_frozen_file(config.extension_contract, root),
            "receipt": verify_frozen_file(config.extension_receipt, root),
            "patched": verify_frozen_path(config.patched_checkout, root),
            "reference": verify_frozen_path(config.reference_architecture, root),
        }
        if (
            source_bundle_hash(config.candidate_source, root)
            != config.candidate_source.sha256
        ):
            raise ValueError("Candidate extension source bundle hash mismatch")
        contract = NeutralExtensionContract.model_validate_json(
            paths["contract"].read_text(encoding="utf-8")
        )
        receipt = NeutralExtensionReceipt.model_validate_json(
            paths["receipt"].read_text(encoding="utf-8")
        )
        checks.append(
            _check(
                "extension.frozen_inputs",
                True,
                "Contract, receipt, proposal, patched checkout, and reference architecture are hash-pinned.",
                "EXTENSION_INPUTS_FROZEN",
            )
        )
    except Exception as error:
        checks.append(
            _check(
                "extension.frozen_inputs",
                False,
                "Frozen extension inputs failed strict validation.",
                "EXTENSION_INPUTS_INVALID",
                error=f"{type(error).__name__}: {error}",
            )
        )

    if contract is not None and paths:
        try:
            checks.append(
                _run_independent_acceptance(
                    config,
                    contract,
                    paths,
                    repository_root=root,
                    run_dir=run_dir,
                    secrets=secret_values([]),
                )
            )
        except Exception as error:
            checks.append(
                _check(
                    "extension.independent_acceptance",
                    False,
                    "Evaluator-isolated extension acceptance could not complete.",
                    "EXTENSION_INDEPENDENT_ACCEPTANCE_ERROR",
                    error=f"{type(error).__name__}: {error}",
                )
            )

    if contract is not None and receipt is not None:
        identity_ok = (
            contract.extension_id == receipt.extension_id == config.candidate_id
            and contract.base_commit == receipt.base_commit
        )
        checks.append(
            _check(
                "extension.identities",
                identity_ok,
                "Extension, candidate, and base-commit identities agree."
                if identity_ok
                else "Extension identity or base commit differs across frozen evidence.",
                "EXTENSION_IDENTITIES_VALID"
                if identity_ok
                else "EXTENSION_IDENTITIES_INVALID",
                observed={
                    "candidate_id": config.candidate_id,
                    "contract_extension_id": contract.extension_id,
                    "receipt_extension_id": receipt.extension_id,
                    "contract_base_commit": contract.base_commit,
                    "receipt_base_commit": receipt.base_commit,
                },
            )
        )
        changed_paths_ok, path_errors = _changed_paths_valid(
            contract, receipt.changed_paths
        )
        checks.append(
            _check(
                "extension.change_scope",
                changed_paths_ok,
                "All changed paths are safe, allowed, and outside protected paths."
                if changed_paths_ok
                else "A changed path violates the frozen extension contract.",
                "EXTENSION_CHANGE_SCOPE_VALID"
                if changed_paths_ok
                else "EXTENSION_CHANGE_SCOPE_INVALID",
                observed={
                    "changed_paths": receipt.changed_paths,
                    "errors": path_errors,
                },
            )
        )
        receipt_ok = (
            receipt.final_status == "succeeded"
            and receipt.patch_applied
            and receipt.source_tree_sha256_before == receipt.source_tree_sha256_after
        )
        checks.append(
            _check(
                "extension.execution_receipt",
                receipt_ok,
                "Patch application succeeded and left the authoritative source tree unchanged."
                if receipt_ok
                else "The extension receipt does not prove a safe successful application.",
                "EXTENSION_EXECUTION_VALID"
                if receipt_ok
                else "EXTENSION_EXECUTION_INVALID",
                observed={
                    "final_status": receipt.final_status,
                    "patch_applied": receipt.patch_applied,
                    "source_unchanged": receipt.source_tree_sha256_before
                    == receipt.source_tree_sha256_after,
                },
            )
        )
        commands_ok, command_errors = _commands_valid(contract, receipt)
        checks.append(
            _check(
                "extension.acceptance_commands",
                commands_ok,
                "Baseline, public-interface, verification, and representative workflow gates passed."
                if commands_ok
                else "A required extension acceptance command is missing or failed.",
                "EXTENSION_ACCEPTANCE_VALID"
                if commands_ok
                else "EXTENSION_ACCEPTANCE_INVALID",
                observed={
                    "errors": command_errors,
                    "command_count": len(receipt.commands),
                },
            )
        )
        patch_ok, patch_error = _patch_identity(config, receipt, root)
        checks.append(
            _check(
                "extension.patch_identity",
                patch_ok,
                "Receipt patch identity matches the frozen candidate patch."
                if patch_ok
                else "Receipt patch identity cannot be linked to the frozen candidate patch.",
                "EXTENSION_PATCH_IDENTITY_VALID"
                if patch_ok
                else "EXTENSION_PATCH_IDENTITY_INVALID",
                error=patch_error,
            )
        )
    else:
        checks.append(
            ExtensionDeterministicCheck(
                check_id="extension.acceptance_blocked",
                status=CheckStatus.NOT_ASSESSED,
                summary="Extension acceptance gates require valid frozen inputs.",
                feedback_code="EXTENSION_ACCEPTANCE_BLOCKED",
            )
        )

    scan_paths = _candidate_paths(config, root)
    if paths:
        scan_paths.extend([paths["contract"], paths["receipt"]])
        scan_paths.extend(
            item for item in paths["patched"].rglob("*") if item.is_file()
        )
    secrets = secret_values([])
    findings = scan_secret_files(scan_paths, known_values=secrets, max_bytes=1_000_000)
    checks.append(
        _check(
            "extension.secret_scan",
            not findings,
            "No credential-like material was detected in extension evidence."
            if not findings
            else "Potential secret material was detected in extension evidence.",
            "EXTENSION_SECRET_SCAN_CLEAN"
            if not findings
            else "EXTENSION_SECRET_MATERIAL_DETECTED",
            observed={"findings": findings},
        )
    )

    feasible = all(check.status == CheckStatus.PASS for check in checks)
    evidence = {
        "deterministic_checks": [check.model_dump(mode="json") for check in checks],
        "operational_objectives": "not_applicable_for_terraio_extension",
        "architecture_contract": (
            None
            if contract is None
            else {
                "required_public_interfaces": contract.required_public_interfaces,
                "architectural_invariants": contract.architectural_invariants,
                "dependency_policy": contract.dependency_policy,
                "expected_dataset_workflow": contract.expected_dataset_workflow,
                "expected_artifact_contract": contract.expected_artifact_contract,
            }
        ),
    }
    evidence_path = run_dir / "deterministic_evidence.extension.v3.json"
    write_new_atomic(
        evidence_path, json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    candidate_review = [
        *config.candidate_source.paths,
        config.patched_checkout.path,
        config.extension_contract.path,
        display_path(evidence_path, root),
    ]
    engineering = run_engineering_quality_v3(
        client=llm_client if feasible else None,
        settings=(
            config.engineering_quality
            if feasible
            else config.engineering_quality.model_copy(update={"enabled": False})
        ),
        target_name=config.candidate_id,
        review_paths=candidate_review,
        deterministic_evidence=evidence,
        repository_root=root,
        redact_values=secrets,
        blind_values=[config.candidate_source.cwd, config.candidate_id],
        reference_paths=[config.reference_architecture.path],
    )
    q = (
        engineering.q_engineering
        if feasible and engineering.engineering_assessed
        else None
    )
    vector_complete = feasible and q is not None
    optimization_ready = bool(vector_complete)
    thesis_evidence_ready = bool(
        vector_complete
        and config.engineering_quality.mode == "thesis"
        and config.engineering_quality.repetitions >= 3
        and engineering.valid_judgments == config.engineering_quality.repetitions
        and not engineering.advisory
    )
    feedback_codes = list(
        dict.fromkeys(
            [
                *(check.feedback_code for check in checks),
                *(
                    code
                    for component in engineering.components
                    for code in component.feedback_codes
                ),
            ]
        )
    )
    return TerraioExtensionEvaluationRunV3(
        suite_id=config.suite_id,
        suite_version=config.suite_version,
        run_id=run_dir.name,
        config_file=display_path(config_file, root),
        deterministic_checks=checks,
        engineering_quality=engineering,
        summary=TerraioExtensionSummaryV3(
            target=config.candidate_id,
            feasible=feasible,
            engineering_assessed=engineering.engineering_assessed,
            objective_vector_complete=vector_complete,
            optimization_ready=optimization_ready,
            thesis_evidence_ready=thesis_evidence_ready,
            q_engineering=q,
            feedback_codes=feedback_codes,
        ),
        duration_seconds=time.perf_counter() - started,
    )


def _check(
    check_id: str,
    passed: bool,
    summary: str,
    code: str,
    *,
    observed: dict[str, Any] | None = None,
    error: str | None = None,
) -> ExtensionDeterministicCheck:
    values = dict(observed or {})
    if error is not None:
        values["error"] = error
    return ExtensionDeterministicCheck(
        check_id=check_id,
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        summary=summary,
        feedback_code=code,
        observed=values,
    )


def _changed_paths_valid(
    contract: NeutralExtensionContract, changed_paths: list[str]
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not changed_paths:
        errors.append("No changed paths were recorded")
    for raw in changed_paths:
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"Unsafe path: {raw}")
            continue
        allowed = any(
            fnmatch.fnmatch(raw, pattern) for pattern in contract.allowed_paths
        )
        protected = any(
            fnmatch.fnmatch(raw, pattern) for pattern in contract.protected_paths
        )
        if not allowed or protected:
            errors.append(f"Out-of-scope path: {raw}")
        if contract.dependency_policy == "existing_dependencies_only" and (
            path.name.startswith("requirements")
            or path.name in {"pyproject.toml", "poetry.lock", "uv.lock"}
        ):
            errors.append(f"Dependency declaration changed: {raw}")
    return not errors, errors


def _commands_valid(
    contract: NeutralExtensionContract, receipt: NeutralExtensionReceipt
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    by_key = {(item.phase, item.name): item for item in receipt.commands}
    expected: list[tuple[str, str]] = [
        *(("baseline", item.name) for item in contract.baseline_commands),
        *(("verification", item.name) for item in contract.verification_commands),
        *(
            ("workflow", item.name)
            for item in contract.representative_workflow_commands
        ),
        *(
            (
                "verification",
                "interface_" + target.replace(":", "_").replace(".", "_"),
            )
            for target in contract.required_public_interfaces
        ),
    ]
    for key in expected:
        result = by_key.get(key)
        if result is None:
            errors.append(f"Missing command: {key[0]}/{key[1]}")
        elif not result.succeeded:
            errors.append(f"Failed command: {key[0]}/{key[1]}")
    return not errors, errors


def _run_independent_acceptance(
    config: TerraioExtensionEvaluationConfigV3,
    contract: NeutralExtensionContract,
    paths: dict[str, Path],
    *,
    repository_root: Path,
    run_dir: Path,
    secrets: list[str],
) -> ExtensionDeterministicCheck:
    """Rerun frozen acceptance gates without trusting the generation receipt."""

    execution_root = run_dir / "extension_execution"
    execution_root.mkdir(parents=True, exist_ok=False)
    baseline = execution_root / "baseline_checkout"
    patched = execution_root / "patched_checkout"
    shutil.copytree(paths["reference"], baseline)
    shutil.copytree(paths["patched"], patched)
    planned = [
        *(("baseline", item, baseline) for item in contract.baseline_commands),
        *(("verification", item, patched) for item in contract.verification_commands),
        *(
            ("verification", _interface_command(target, contract), patched)
            for target in contract.required_public_interfaces
        ),
        *(
            ("workflow", item, patched)
            for item in contract.representative_workflow_commands
        ),
    ]
    observations: list[dict[str, Any]] = []
    passed = True
    for index, (phase, spec, cwd) in enumerate(planned, 1):
        command = list(spec.argv)
        if command[0] in {"python", "python3"}:
            command[0] = sys.executable
        prepared = prepare_command(
            command,
            cwd=cwd,
            writable_root=execution_root,
            policy=config.execution_policy,
        )
        observation = run_observed_command(
            command,
            execution_command=prepared.command,
            environment=prepared.environment,
            isolation=prepared.evidence,
            cwd=cwd,
            log_dir=execution_root / "logs",
            label=f"{index:02d}_{phase}_{_safe_label(spec.name)}",
            repository_root=repository_root,
            redact_values=secrets,
            timeout_seconds=spec.timeout_seconds,
        )
        diagnostic_count = _diagnostic_count(
            observation.stdout_tail + "\n" + observation.stderr_tail
        )
        succeeded = (
            observation.exit_code in spec.expected_return_codes
            and not observation.timed_out
            and observation.isolation is not None
            and observation.isolation.full
            and (
                spec.expected_diagnostic_count is None
                or diagnostic_count == spec.expected_diagnostic_count
            )
        )
        passed = passed and succeeded
        observations.append(
            {
                "phase": phase,
                "name": spec.name,
                "succeeded": succeeded,
                "expected_return_codes": spec.expected_return_codes,
                "expected_diagnostic_count": spec.expected_diagnostic_count,
                "diagnostic_count": diagnostic_count,
                "observation": observation.model_dump(mode="json"),
            }
        )
    originals_unchanged = True
    try:
        verify_frozen_path(config.reference_architecture, repository_root)
        verify_frozen_path(config.patched_checkout, repository_root)
    except Exception:
        originals_unchanged = False
    passed = passed and originals_unchanged
    return _check(
        "extension.independent_acceptance",
        passed,
        (
            "Evaluator-isolated baseline, interface, verification, and workflow commands passed."
            if passed
            else "An evaluator-isolated extension acceptance command or post-run source hash failed."
        ),
        "EXTENSION_INDEPENDENT_ACCEPTANCE_VALID"
        if passed
        else "EXTENSION_INDEPENDENT_ACCEPTANCE_INVALID",
        observed={
            "trusted_originals_unchanged": originals_unchanged,
            "commands": observations,
        },
    )


def _interface_command(
    target: str, contract: NeutralExtensionContract
) -> NeutralExtensionCommand:
    module, attribute = target.split(":", maxsplit=1)
    roots = repr(contract.python_source_roots)
    script = (
        "import importlib, sys; "
        f"[sys.path.insert(0, root) for root in {roots}]; "
        f"module = importlib.import_module({module!r}); "
        f"assert getattr(module, {attribute!r}) is not None"
    )
    return NeutralExtensionCommand(
        name="interface_" + target.replace(":", "_").replace(".", "_"),
        kind="typecheck",
        argv=[sys.executable, "-c", script],
        timeout_seconds=120,
    )


def _diagnostic_count(output: str) -> int | None:
    match = re.search(r"Found\s+(\d+)\s+diagnostics?", output)
    return int(match.group(1)) if match else None


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)[:80] or "command"


def _patch_identity(
    config: TerraioExtensionEvaluationConfigV3,
    receipt: NeutralExtensionReceipt,
    root: Path,
) -> tuple[bool, str | None]:
    candidates = [
        path
        for path in _candidate_paths(config, root)
        if path.is_file() and path.name == "patch.diff"
    ]
    if len(candidates) != 1:
        return False, "Candidate source must include exactly one patch.diff"
    digest = hashlib.sha256(candidates[0].read_bytes()).hexdigest()
    return (
        digest == receipt.patch_sha256,
        None
        if digest == receipt.patch_sha256
        else "patch.diff hash differs from receipt",
    )


def _candidate_paths(
    config: TerraioExtensionEvaluationConfigV3, root: Path
) -> list[Path]:
    files: list[Path] = []
    for raw in config.candidate_source.paths:
        path = (
            Path(raw).resolve() if Path(raw).is_absolute() else (root / raw).resolve()
        )
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
    return sorted(set(files))
