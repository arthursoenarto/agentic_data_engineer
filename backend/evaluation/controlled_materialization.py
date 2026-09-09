"""Evaluator-controlled family-pipeline materialization and receipt validation."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from backend.file_copy import copytree_isolated
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.contracts import (
    RegularGridZarrOutputPolicy,
    content_hash,
    read_contract_lock,
    validate_contract_against_inventory,
)
from backend.evaluation.command_runner import run_observed_command
from backend.evaluation.constrained_schemas import (
    ConstrainedEvaluationConfig,
    NeutralPipelineReceipt,
    ScenarioResult,
)
from backend.evaluation.evidence import (
    display_path,
    frozen_path_hash,
    logical_field_id,
    path_size,
    pipeline_artifact_hash,
    redact,
    repo_path,
    require_within,
    source_bundle_hash,
    verify_frozen_file,
    verify_frozen_path,
    write_atomic_text,
)
from backend.evaluation.core_schemas import (
    CommandExecutionPolicy,
    CommandIsolationEvidence,
    CommandObservation,
    CommandResourceUsage,
)
from backend.evaluation.isolation import prepare_command
from backend.evaluation.regular_grid_zarr import (
    OutputPolicyError,
    ResolvedGrid,
    UnsupportedGridError,
    resolve_regular_grid,
    validate_zarr_integrity,
)


class _InventoryView(BaseModel):
    """Minimal neutral inventory adapter needed by backend.contracts validation."""

    schema_version: str
    dataset_slug: str
    dataset_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    availability: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class Grounding:
    """Validated identities and source paths from one trusted suite."""

    lock_payload: dict[str, Any]
    inventory_payload: dict[str, Any]
    pipeline_contract: dict[str, Any]
    manifest: dict[str, Any]
    contract_ids: list[str]
    paths: dict[str, Path]


@dataclass
class Scenario:
    """One controlled command plus its independently resolved output."""

    result: ScenarioResult
    output_dir: Path
    receipt_path: Path
    command: list[str]
    grid: ResolvedGrid | None
    validation_error: str | None
    unsupported: bool = False
    policy_failure: bool = False


def validate_grounding(
    config: ConstrainedEvaluationConfig, repository_root: Path
) -> Grounding:
    """Validate every frozen identity before candidate execution."""

    paths = {
        "contract": verify_frozen_file(config.contract_lock, repository_root),
        "inventory": verify_frozen_file(config.inventory, repository_root),
        "pipeline_contract": verify_frozen_file(
            config.execution.pipeline_contract, repository_root
        ),
        "manifest": verify_frozen_file(config.execution.manifest, repository_root),
        "cache": verify_frozen_path(config.execution.frozen_cache, repository_root),
        "reference": verify_frozen_path(config.reference.artifact, repository_root),
    }
    if not paths["cache"].is_dir():
        raise ValueError("frozen_cache must be a directory")
    if config.execution.python_executable is not None:
        runtime = Path(config.execution.python_executable).expanduser().absolute()
        if not runtime.is_absolute():
            raise ValueError("Candidate python_executable must be absolute")
        try:
            runtime.relative_to(repository_root.absolute())
        except ValueError as error:
            raise ValueError("Candidate runtime path escapes the repository") from error
        if not runtime.is_file():
            raise FileNotFoundError(f"Candidate runtime is missing: {runtime}")
    if config.reference.content_id != f"sha256:{config.reference.artifact.sha256}":
        raise ValueError("reference content_id must equal sha256:<artifact hash>")
    if (
        source_bundle_hash(config.candidate_source, repository_root)
        != config.candidate_source.sha256
    ):
        raise ValueError("Candidate source bundle hash differs from the trusted suite")

    lock_payload = json.loads(paths["contract"].read_text(encoding="utf-8"))
    inventory_payload = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    pipeline_contract = json.loads(
        paths["pipeline_contract"].read_text(encoding="utf-8")
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    lock = read_contract_lock(paths["contract"])
    inventory = _InventoryView.model_validate(inventory_payload)
    if lock.inventory_sha256 != content_hash(inventory_payload):
        raise ValueError(
            "Contract lock inventory hash does not match the frozen inventory"
        )
    if lock.inventory_schema_version != inventory.schema_version:
        raise ValueError("Contract lock and inventory schema versions differ")
    validate_contract_against_inventory(lock.contract, inventory)  # type: ignore[arg-type]
    if config.output_policy is not None:
        try:
            generated_policy = RegularGridZarrOutputPolicy.model_validate(
                pipeline_contract["policy"]["zarr"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Pipeline contract has no valid public regular-grid Zarr policy"
            ) from error
        if generated_policy != config.output_policy.public_policy():
            raise ValueError(
                "Evaluation output policy differs from the generation-visible pipeline policy"
            )

    contract_ids = [
        logical_field_id(
            field.name,
            [selector.model_dump(mode="json") for selector in field.selectors],
        )
        for field in lock.contract.fields
    ]
    mapped_ids = [channel.field_id for channel in config.grid.channels]
    if len(contract_ids) != len(set(contract_ids)) or mapped_ids != contract_ids:
        raise ValueError(
            f"Candidate/reference mappings must exactly match contract order: {contract_ids}"
        )
    cwd = repo_path(config.candidate_source.cwd, repository_root)
    if not cwd.is_dir():
        raise FileNotFoundError(f"Candidate cwd does not exist: {cwd}")
    return Grounding(
        lock_payload=lock_payload,
        inventory_payload=inventory_payload,
        pipeline_contract=pipeline_contract,
        manifest=manifest,
        contract_ids=contract_ids,
        paths=paths,
    )


def run_scenario(
    scenario: Literal["initial", "rerun"],
    config: ConstrainedEvaluationConfig,
    grounding: Grounding,
    *,
    repository_root: Path,
    run_dir: Path,
    secrets: list[str],
    execution_policy: CommandExecutionPolicy | None = None,
) -> Scenario:
    """Materialize into a fresh output from a private frozen-cache copy."""

    scenario_started = time.perf_counter()
    scenario_dir = run_dir / "materializations" / scenario
    scenario_dir.mkdir(parents=True, exist_ok=False)
    setup_started = time.perf_counter()
    inputs = scenario_dir / "inputs"
    inputs.mkdir()
    contract_path = inputs / "contract.lock.json"
    inventory_path = inputs / "inventory.json"
    shutil.copy2(grounding.paths["contract"], contract_path)
    shutil.copy2(grounding.paths["inventory"], inventory_path)
    writable_root = scenario_dir
    if execution_policy is not None:
        writable_root = scenario_dir / f"{scenario}_writable"
        writable_root.mkdir()
    # The complete frozen cache is a trusted benchmark input. Keep its private
    # copy outside the candidate's writable root so sandbox evidence matches
    # the actual filesystem policy.
    cache_dir = inputs / "cache"
    copytree_isolated(grounding.paths["cache"], cache_dir)
    if frozen_path_hash(cache_dir) != config.execution.frozen_cache.sha256:
        raise ValueError("Controlled cache copy differs from the frozen source")
    output_dir = writable_root / "output"
    receipt_path = writable_root / "pipeline_run.json"
    setup_seconds = time.perf_counter() - setup_started
    command = _resolve_command(
        config.execution.command,
        contract=contract_path,
        inventory=inventory_path,
        cache=cache_dir,
        output=output_dir,
        receipt=receipt_path,
        python_executable=config.execution.python_executable,
    )
    with tempfile.TemporaryDirectory(prefix="evaluation-candidate-") as temporary:
        candidate_cwd = _stage_candidate_source(
            config.candidate_source,
            repository_root=repository_root,
            destination=Path(temporary) / "candidate",
            supplemental_paths=[
                grounding.paths["pipeline_contract"],
                grounding.paths["manifest"],
            ],
        )
        prepared = (
            prepare_command(
                command,
                cwd=candidate_cwd,
                writable_root=writable_root,
                policy=execution_policy,
            )
            if execution_policy is not None
            else None
        )
        observation = _observe_command(
            command,
            cwd=candidate_cwd,
            log_dir=scenario_dir / "logs",
            repository_root=repository_root,
            secrets=secrets,
            timeout_seconds=config.execution.timeout_seconds,
            environment=None if prepared is None else prepared.environment,
            execution_command=None if prepared is None else prepared.command,
            isolation=None if prepared is None else prepared.evidence,
        )
    validation_started = time.perf_counter()
    grid: ResolvedGrid | None = None
    validation_error: str | None = None
    unsupported = False
    policy_failure = False
    if observation.exit_code == 0 and not observation.timed_out:
        try:
            if not output_dir.is_dir():
                raise FileNotFoundError(
                    "Pipeline did not publish its controlled output directory"
                )
            grid = resolve_regular_grid(
                config.grid.candidate_grid,
                config.grid.channels,
                candidate=True,
                repository_root=repository_root,
                output_dir=output_dir,
                expected_shape=config.grid.expected_shape,
                required_coordinate_metadata=(
                    config.output_policy.coordinate_metadata
                    if config.output_policy is not None
                    else None
                ),
            )
            validate_zarr_integrity(
                grid.store_paths,
                max_block_bytes=config.comparison.max_block_bytes,
            )
        except Exception as error:
            if grid is not None:
                grid.close()
                grid = None
            validation_error = redact(str(error), secrets)
            unsupported = isinstance(error, UnsupportedGridError)
            policy_failure = isinstance(error, OutputPolicyError)
    else:
        validation_error = (
            "Pipeline execution timed out."
            if observation.timed_out
            else f"Pipeline exited with code {observation.exit_code}."
        )
    validation_seconds = time.perf_counter() - validation_started
    result = ScenarioResult(
        scenario=scenario,
        setup_seconds=setup_seconds,
        execution=observation,
        validation_seconds=validation_seconds,
        total_seconds=time.perf_counter() - scenario_started,
        output_path=display_path(output_dir, repository_root),
        receipt_path=display_path(receipt_path, repository_root),
    )
    return Scenario(
        result=result,
        output_dir=output_dir,
        receipt_path=receipt_path,
        command=command,
        grid=grid,
        validation_error=validation_error,
        unsupported=unsupported,
        policy_failure=policy_failure,
    )


def candidate_claim_mismatches(
    config: ConstrainedEvaluationConfig,
    grounding: Grounding,
) -> list[str]:
    """Compare candidate-authored claims without treating them as grading inputs."""

    contract = grounding.pipeline_contract
    manifest = grounding.manifest
    manifest_core = dict(manifest)
    manifest_core.pop("pipeline_contract_hash", None)
    expected = {
        "pipeline_id": config.candidate_id,
        "dataset_slug": grounding.lock_payload["contract"].get("dataset_slug"),
        "command_template": config.execution.command,
        "storage_format": config.grid.output_format,
        "generation_manifest_hash": content_hash(manifest_core),
        "pipeline_contract_hash": content_hash(contract),
    }
    observed = {
        "pipeline_id": contract.get("pipeline_id"),
        "dataset_slug": contract.get("dataset_slug"),
        "command_template": contract.get("command_template"),
        "storage_format": contract.get("output_artifact", {}).get("storage_format"),
        "generation_manifest_hash": contract.get("generation_manifest_hash"),
        "pipeline_contract_hash": manifest.get("pipeline_contract_hash"),
    }
    return [name for name, value in expected.items() if observed[name] != value]


def validate_receipt(
    receipt: NeutralPipelineReceipt,
    scenario: Scenario,
    grounding: Grounding,
    config: ConstrainedEvaluationConfig,
) -> None:
    """Validate receipt identities against evaluator-observed inputs and bytes."""

    contract = grounding.pipeline_contract
    manifest_core = dict(grounding.manifest)
    manifest_core.pop("pipeline_contract_hash", None)
    expected = {
        "run_id": scenario.receipt_path.parent.name,
        "pipeline_id": config.candidate_id,
        "manifest_hash": content_hash(manifest_core),
        "pipeline_contract_hash": content_hash(contract),
        "contract_lock_hash": content_hash(grounding.lock_payload),
        "inventory_hash": content_hash(grounding.inventory_payload),
        "command": scenario.command,
        "exit_code": scenario.result.execution.exit_code,
    }
    observed = {name: getattr(receipt, name) for name in expected}
    if observed != expected:
        mismatched = [
            name for name, value in expected.items() if observed[name] != value
        ]
        raise ValueError(
            f"Receipt identities differ from evaluator-controlled inputs: {mismatched}"
        )
    if receipt.final_status != "succeeded" or receipt.exit_code != 0:
        raise ValueError("Receipt does not record a successful materialization")
    if len(receipt.outputs) != 1:
        raise ValueError("Receipt must identify exactly one output artifact")
    output = receipt.outputs[0]
    expected_artifact = contract.get("output_artifact", {})
    output_path = Path(str(output.get("path", ""))).resolve()
    if output_path != scenario.output_dir.resolve():
        raise ValueError("Receipt output path differs from the controlled output")
    if output.get("artifact_id") != expected_artifact.get("artifact_id"):
        raise ValueError(
            "Receipt output artifact identity differs from the pipeline contract"
        )
    if output.get("storage_format") != "zarr":
        raise ValueError("Receipt output format is not Zarr")
    size, count = path_size(output_path)
    digest = pipeline_artifact_hash(output_path)
    output_mismatches = [
        name
        for name, expected_value in {
            "size_bytes": size,
            "file_count": count,
            "sha256": digest,
        }.items()
        if output.get(name) != expected_value
    ]
    if output_mismatches:
        raise ValueError(
            "Receipt output fingerprint differs from observed bytes: "
            f"{output_mismatches}"
        )


def _observe_command(
    command: list[str],
    *,
    cwd: Path,
    log_dir: Path,
    repository_root: Path,
    secrets: list[str],
    timeout_seconds: float | None,
    environment: dict[str, str] | None = None,
    execution_command: list[str] | None = None,
    isolation: CommandIsolationEvidence | None = None,
) -> CommandObservation:
    try:
        return run_observed_command(
            command,
            cwd=cwd,
            log_dir=log_dir,
            label="pipeline",
            repository_root=repository_root,
            redact_values=secrets,
            timeout_seconds=timeout_seconds,
            environment=environment,
            execution_command=execution_command,
            isolation=isolation,
        )
    except Exception as error:
        now = datetime.now(UTC).isoformat()
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "pipeline.stdout.log"
        stderr_path = log_dir / "pipeline.stderr.log"
        message = redact(f"{type(error).__name__}: {error}", secrets)
        write_atomic_text(stdout_path, "")
        write_atomic_text(stderr_path, message + "\n")
        return CommandObservation(
            command=command,
            executed_command=execution_command or command,
            cwd=display_path(cwd, repository_root),
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
            exit_code=-1,
            stdout_log=display_path(stdout_path, repository_root),
            stderr_log=display_path(stderr_path, repository_root),
            stderr_tail=message,
            environment_variable_names=sorted(environment or {}),
            isolation=isolation,
            resources=CommandResourceUsage(),
        )


def _stage_candidate_source(
    spec: CandidateSourceSpec,
    *,
    repository_root: Path,
    destination: Path,
    supplemental_paths: list[Path],
) -> Path:
    """Copy declared source and separately frozen runner inputs for execution."""

    source_cwd = repo_path(spec.cwd, repository_root)
    destination.mkdir(parents=True, exist_ok=False)
    for configured_path in spec.paths:
        source = repo_path(configured_path, repository_root)
        relative = source.relative_to(source_cwd)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    for source in supplemental_paths:
        relative = source.resolve().relative_to(source_cwd.resolve())
        target = destination / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def _resolve_command(
    template: list[str],
    *,
    contract: Path,
    inventory: Path,
    cache: Path,
    output: Path,
    receipt: Path,
    python_executable: str | None = None,
) -> list[str]:
    replacements = {
        "{contract_lock_json}": str(contract.resolve()),
        "{dataset_inventory_json}": str(inventory.resolve()),
        "{cache_dir}": str(cache.resolve()),
        "{output_dir}": str(output.resolve()),
        "{pipeline_run_json}": str(receipt.resolve()),
    }
    command = [replacements.get(token, token) for token in template]
    if command[0] == "python":
        command[0] = str(
            Path(python_executable).absolute()
            if python_executable is not None
            else Path(sys.executable).absolute()
        )
    if any("\0" in token or "{" in token or "}" in token for token in command):
        raise ValueError("Command contains an unsafe or unresolved token")
    return command
