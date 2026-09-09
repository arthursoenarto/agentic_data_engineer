"""Generation-owned acceptance checks for immutable dataset-family pipelines."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from backend.file_copy import copytree_isolated
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.agents.etl_pipeline.execution import (
    PipelineExecutionRun,
    execute_family_pipeline,
    validate_pipeline_bundle,
)
from backend.agents.etl_pipeline.paths import pipeline_run_id
from backend.agents.etl_pipeline.schemas import (
    DatasetArtifactLayout,
    PipelineManifest,
    TabularDatasetArtifactLayout,
)
from backend.contracts.output_policy import CoordinateMetadataRequirements
from backend.env import ENV_FILE

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class FamilyAcceptanceCheck(BaseModel):
    """One deterministic generation-side acceptance result."""

    check_id: str
    required: bool = True
    status: Literal["pass", "fail"]
    feedback_code: str
    summary: str
    evidence: dict[str, object] = Field(default_factory=dict)


class FamilyPipelineAcceptanceReport(BaseModel):
    """Immutable proof that generated code satisfies the public family interface."""

    schema_version: str = "family_pipeline_acceptance.v1"
    acceptance_id: str
    pipeline_id: str
    candidate_origin: Literal["generated", "repaired"]
    repair_reference: str | None = None
    generated_source_sha256_before: str
    generated_source_sha256_after: str
    contract_lock_path: str
    inventory_path: str
    frozen_cache_path: str
    frozen_cache_sha256_before: str
    frozen_cache_sha256_after: str
    initial_run_receipt: str | None = None
    rerun_receipt: str | None = None
    dataset_artifact: DatasetArtifactLayout | TabularDatasetArtifactLayout | None = None
    checks: list[FamilyAcceptanceCheck]
    final_status: Literal["passed", "failed"]
    generated_at: str

    @model_validator(mode="after")
    def status_must_match_checks(self) -> FamilyPipelineAcceptanceReport:
        required_checks = [check for check in self.checks if check.required]
        passed = bool(required_checks) and all(
            check.status == "pass" for check in required_checks
        )
        if passed != (self.final_status == "passed"):
            raise ValueError("Acceptance final_status must match all check statuses.")
        if self.candidate_origin == "repaired" and not self.repair_reference:
            raise ValueError("Repaired candidates require repair_reference provenance.")
        if self.candidate_origin == "generated" and self.repair_reference:
            raise ValueError(
                "Unrepaired generated candidates cannot declare repair_reference."
            )
        return self


def accept_family_pipeline(
    *,
    dataset_dir: Path,
    pipeline_id: str,
    contract_lock_path: Path,
    prepared_cache_dir: Path,
    inventory_path: Path | None = None,
    acceptance_id: str | None = None,
    candidate_origin: Literal["generated", "repaired"] = "generated",
    repair_reference: str | None = None,
    timeout_seconds: int = 300,
    env_path: Path = ENV_FILE,
    run_generated_tests: bool = True,
    python_executable: Path | None = None,
) -> tuple[FamilyPipelineAcceptanceReport, Path]:
    """Verify output integrity and two credential-free read-only-cache runs."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    dataset_dir = dataset_dir.resolve()
    pipeline_dir = dataset_dir / "pipelines" / pipeline_id
    source_cache = prepared_cache_dir.resolve()
    if not source_cache.is_dir() or not any(source_cache.rglob("*")):
        raise ValueError("prepared_cache_dir must contain a candidate-native cache.")
    inventory = (inventory_path or dataset_dir / "dataset_inventory.json").resolve()
    contract_lock = contract_lock_path.resolve()
    manifest, pipeline_contract = validate_pipeline_bundle(pipeline_dir)
    if (
        pipeline_contract.implementation_interface_version
        not in {
            "family_pipeline_interface.v2",
            "family_pipeline_interface.v3",
            "family_pipeline_interface.v4",
        }
    ):
        raise ValueError("Generation acceptance requires a typed family interface.")
    zarr_policy = pipeline_contract.policy.zarr
    requires_zarr_v3 = (
        pipeline_contract.implementation_interface_version
        == "family_pipeline_interface.v3"
    )
    requires_parquet = (
        pipeline_contract.implementation_interface_version
        == "family_pipeline_interface.v4"
    )
    if requires_zarr_v3 and zarr_policy is None:
        raise ValueError("family_pipeline_interface.v3 requires a Zarr output policy.")

    resolved_id = acceptance_id or _acceptance_id()
    acceptance_dir = pipeline_dir / "acceptance" / resolved_id
    acceptance_dir.mkdir(parents=True, exist_ok=False)
    frozen_cache = acceptance_dir / "frozen_cache"
    copytree_isolated(source_cache, frozen_cache)
    _make_read_only(frozen_cache)

    source_before = _generated_source_hash(dataset_dir, pipeline_dir, manifest)
    cache_before = _path_hash(frozen_cache)
    checks: list[FamilyAcceptanceCheck] = []
    initial: PipelineExecutionRun | None = None
    rerun: PipelineExecutionRun | None = None
    artifact: DatasetArtifactLayout | TabularDatasetArtifactLayout | None = None
    credentials = set(pipeline_contract.credential_environment_variables)

    tests_ok, test_evidence = _run_generated_tests(
        pipeline_dir=pipeline_dir,
        acceptance_dir=acceptance_dir,
        credentials=credentials,
        timeout_seconds=timeout_seconds,
        enabled=run_generated_tests,
        python_executable=python_executable,
    )
    checks.append(
        _check(
            "generation.generated_tests",
            tests_ok,
            "GENERATED_TESTS_PASSED",
            "GENERATED_TESTS_FAILED",
            "Generated diagnostic tests passed."
            if tests_ok
            else "Generated diagnostic tests failed.",
            test_evidence,
            required=False,
        )
    )

    execution_error: str | None = None
    try:
        initial = execute_family_pipeline(
            dataset_dir=dataset_dir,
            pipeline_id=pipeline_id,
            contract_lock_path=contract_lock,
            inventory_path=inventory,
            cache_dir=frozen_cache,
            run_id=pipeline_run_id(),
            timeout_seconds=timeout_seconds,
            env_path=env_path,
            unset_environment_variables=credentials,
            repair_run_reference=repair_reference,
            python_executable=python_executable,
        )
        rerun = execute_family_pipeline(
            dataset_dir=dataset_dir,
            pipeline_id=pipeline_id,
            contract_lock_path=contract_lock,
            inventory_path=inventory,
            cache_dir=frozen_cache,
            run_id=pipeline_run_id(),
            timeout_seconds=timeout_seconds,
            env_path=env_path,
            unset_environment_variables=credentials,
            repair_run_reference=repair_reference,
            python_executable=python_executable,
        )
    except Exception as error:  # noqa: BLE001 - preserve typed failure evidence.
        execution_error = f"{type(error).__name__}: {error}"

    runs_succeeded = bool(initial and rerun and initial.succeeded and rerun.succeeded)
    checks.append(
        _check(
            "generation.interface_execution",
            runs_succeeded,
            "INTERFACE_EXECUTION_PASSED",
            "INTERFACE_EXECUTION_FAILED",
            (
                "Both framework-controlled materializations succeeded."
                if runs_succeeded
                else "A framework-controlled materialization failed."
            ),
            {
                "initial_status": initial.receipt.final_status if initial else None,
                "rerun_status": rerun.receipt.final_status if rerun else None,
                "error": execution_error,
            },
        )
    )

    cache_only = False
    cache_evidence: dict[str, object] = {}
    if initial and rerun:
        cache_evidence = {
            "initial": initial.receipt.cache.model_dump(mode="json"),
            "rerun": rerun.receipt.cache.model_dump(mode="json"),
        }
        cache_only = all(
            run.receipt.cache.hits > 0
            and run.receipt.cache.misses == 0
            and run.receipt.cache.acquired == 0
            and Path(run.receipt.cache.cache_dir).resolve() == frozen_cache.resolve()
            for run in (initial, rerun)
        )
    checks.append(
        _check(
            "generation.credential_free_read_only_cache",
            cache_only,
            "READ_ONLY_CACHE_RERUN_PASSED",
            "READ_ONLY_CACHE_RERUN_FAILED",
            (
                "Both credential-free runs used only the supplied read-only cache."
                if cache_only
                else "A run missed, acquired, or did not use the supplied read-only cache."
            ),
            cache_evidence,
        )
    )

    zarr_valid = False
    zarr_evidence: dict[str, object] = {}
    if (
        initial
        and rerun
        and initial.receipt.dataset_artifact
        and rerun.receipt.dataset_artifact
        and initial.receipt.dataset_artifact == rerun.receipt.dataset_artifact
    ):
        artifact = initial.receipt.dataset_artifact
        try:
            first = _validate_zarr_layout(
                initial.paths.output_dir,
                artifact,
                require_v3=requires_zarr_v3,
                require_consolidated=requires_zarr_v3,
                required_coordinate_metadata=(
                    zarr_policy.coordinate_metadata if zarr_policy is not None else None
                ),
            )
            second = _validate_zarr_layout(
                rerun.paths.output_dir,
                artifact,
                require_v3=requires_zarr_v3,
                require_consolidated=requires_zarr_v3,
                required_coordinate_metadata=(
                    zarr_policy.coordinate_metadata if zarr_policy is not None else None
                ),
            )
            zarr_evidence = {"initial": first, "rerun": second}
            zarr_valid = first == second
        except Exception as error:  # noqa: BLE001 - convert to acceptance evidence.
            zarr_evidence = {"error": f"{type(error).__name__}: {error}"}
    checks.append(
        _check(
            "generation.real_zarr_layout",
            zarr_valid,
            "REAL_ZARR_LAYOUT_PASSED",
            "REAL_ZARR_LAYOUT_FAILED",
            (
                "Declared Zarr arrays are readable and stable across exact reruns."
                if zarr_valid
                else "Declared Zarr layout is missing, unreadable, or unstable."
            ),
            zarr_evidence,
            required=not requires_parquet,
        )
    )
    zarr_policy_valid = bool(
        zarr_valid
        and zarr_evidence.get("initial", {}).get("zarr_format") == 3
        and zarr_evidence.get("initial", {}).get("consolidated_metadata") is True
        and zarr_evidence.get("rerun", {}).get("zarr_format") == 3
        and zarr_evidence.get("rerun", {}).get("consolidated_metadata") is True
    )
    checks.append(
        _check(
            "generation.zarr_v3_consolidated",
            zarr_policy_valid,
            "ZARR_V3_CONSOLIDATED_PASSED",
            "ZARR_V3_CONSOLIDATED_FAILED",
            (
                "Both outputs use Zarr format 3 with consolidated root metadata."
                if zarr_policy_valid
                else "An output violates the required Zarr v3 consolidated policy."
            ),
            zarr_evidence,
            required=requires_zarr_v3,
        )
    )

    parquet_valid = not requires_parquet
    parquet_evidence: dict[str, object] = (
        {} if requires_parquet else {"status": "not_applicable"}
    )
    if (
        requires_parquet
        and initial
        and rerun
        and isinstance(initial.receipt.dataset_artifact, TabularDatasetArtifactLayout)
        and initial.receipt.dataset_artifact == rerun.receipt.dataset_artifact
    ):
        artifact = initial.receipt.dataset_artifact
        try:
            first = _validate_parquet_layout(initial.paths.output_dir, artifact)
            second = _validate_parquet_layout(rerun.paths.output_dir, artifact)
            parquet_evidence = {"initial": first, "rerun": second}
            parquet_valid = first == second
        except Exception as error:  # noqa: BLE001 - acceptance evidence.
            parquet_evidence = {"error": f"{type(error).__name__}: {error}"}
    checks.append(
        _check(
            "generation.real_parquet_layout",
            parquet_valid,
            "REAL_PARQUET_LAYOUT_PASSED",
            "REAL_PARQUET_LAYOUT_FAILED",
            (
                "Declared Parquet table is readable and stable across exact reruns."
                if parquet_valid
                else "Declared Parquet table is missing, invalid, or unstable."
            ),
            parquet_evidence,
            required=requires_parquet,
        )
    )

    outputs_equal = bool(
        initial
        and rerun
        and initial.receipt.outputs
        and rerun.receipt.outputs
        and initial.receipt.outputs[0].sha256 == rerun.receipt.outputs[0].sha256
    )
    checks.append(
        _check(
            "generation.exact_output_rerun",
            outputs_equal,
            "EXACT_OUTPUT_RERUN_PASSED",
            "EXACT_OUTPUT_RERUN_FAILED",
            "Exact rerun output fingerprints match."
            if outputs_equal
            else "Exact rerun output fingerprints differ.",
            {
                "initial_sha256": (
                    initial.receipt.outputs[0].sha256
                    if initial and initial.receipt.outputs
                    else None
                ),
                "rerun_sha256": (
                    rerun.receipt.outputs[0].sha256
                    if rerun and rerun.receipt.outputs
                    else None
                ),
            },
            required=False,
        )
    )

    source_after = _generated_source_hash(dataset_dir, pipeline_dir, manifest)
    cache_after = _path_hash(frozen_cache)
    immutable = source_before == source_after and cache_before == cache_after
    checks.append(
        _check(
            "generation.frozen_inputs_unchanged",
            immutable,
            "FROZEN_INPUTS_UNCHANGED",
            "FROZEN_INPUTS_CHANGED",
            "Generated source and frozen cache remained unchanged."
            if immutable
            else "Generated source or frozen cache changed during acceptance.",
            {
                "source_before": source_before,
                "source_after": source_after,
                "cache_before": cache_before,
                "cache_after": cache_after,
            },
        )
    )

    final_status: Literal["passed", "failed"] = (
        "passed"
        if all(check.status == "pass" for check in checks if check.required)
        else "failed"
    )
    report = FamilyPipelineAcceptanceReport(
        acceptance_id=resolved_id,
        pipeline_id=pipeline_id,
        candidate_origin=candidate_origin,
        repair_reference=repair_reference,
        generated_source_sha256_before=source_before,
        generated_source_sha256_after=source_after,
        contract_lock_path=_display_path(contract_lock),
        inventory_path=_display_path(inventory),
        frozen_cache_path=_display_path(frozen_cache),
        frozen_cache_sha256_before=cache_before,
        frozen_cache_sha256_after=cache_after,
        initial_run_receipt=_display_path(initial.paths.receipt) if initial else None,
        rerun_receipt=_display_path(rerun.paths.receipt) if rerun else None,
        dataset_artifact=artifact,
        checks=checks,
        final_status=final_status,
        generated_at=datetime.now(UTC).isoformat(),
    )
    report_path = acceptance_dir / "acceptance.json"
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report, report_path


def _run_generated_tests(
    *,
    pipeline_dir: Path,
    acceptance_dir: Path,
    credentials: set[str],
    timeout_seconds: int,
    enabled: bool,
    python_executable: Path | None,
) -> tuple[bool, dict[str, object]]:
    tests = pipeline_dir / "tests"
    if not enabled:
        return True, {"status": "disabled"}
    if not tests.is_dir():
        return False, {"status": "missing", "path": str(tests)}
    environment = dict(os.environ)
    for name in credentials:
        environment.pop(name, None)
    completed = subprocess.run(
        [str((python_executable or Path(sys.executable)).absolute()), "-m", "pytest", "-q"],
        cwd=pipeline_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    stdout = acceptance_dir / "generated_tests.stdout.log"
    stderr = acceptance_dir / "generated_tests.stderr.log"
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    return completed.returncode == 0, {
        "exit_code": completed.returncode,
        "stdout_log": _display_path(stdout),
        "stderr_log": _display_path(stderr),
    }


def _validate_zarr_layout(
    output_dir: Path,
    artifact: DatasetArtifactLayout,
    *,
    require_v3: bool,
    require_consolidated: bool,
    required_coordinate_metadata: CoordinateMetadataRequirements | None,
) -> dict[str, object]:
    try:
        import zarr
    except ImportError as error:
        raise RuntimeError("Generation acceptance requires zarr.") from error

    store = output_dir / artifact.store_path
    root_metadata_path = store / "zarr.json"
    zarr_format = 3 if root_metadata_path.is_file() else 2
    consolidated_metadata = False
    if zarr_format == 3:
        root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
        consolidated = root_metadata.get("consolidated_metadata")
        consolidated_metadata = bool(
            isinstance(consolidated, dict)
            and isinstance(consolidated.get("metadata"), dict)
        )
    else:
        consolidated_metadata = (store / ".zmetadata").is_file()
    if require_v3 and zarr_format != 3:
        raise ValueError("Published output is not Zarr format 3.")
    if require_consolidated and not consolidated_metadata:
        raise ValueError("Published output lacks consolidated metadata.")
    group = zarr.open_group(
        str(store),
        mode="r",
        use_consolidated=True if require_consolidated else None,
    )
    paths = {
        *artifact.coordinates.values(),
        *(channel.array_path for channel in artifact.channels),
        *(
            path
            for channel in artifact.channels
            for path in channel.selector_coordinate_paths.values()
        ),
    }
    required_non_scalar = {
        *artifact.coordinates.values(),
        *(channel.array_path for channel in artifact.channels),
    }
    arrays: dict[str, object] = {}
    array_dimensions: dict[str, tuple[str, ...]] = {}
    for path in sorted(paths):
        array = group[path]
        shape = [int(value) for value in array.shape]
        if (path in required_non_scalar and not shape) or any(
            value <= 0 for value in shape
        ):
            raise ValueError(f"Declared Zarr array is empty: {path}")
        dimensions = _zarr_array_dimensions(array)
        if len(dimensions) != len(shape):
            raise ValueError(
                f"Declared Zarr array {path!r} has no complete dimension metadata."
            )
        array_dimensions[path] = dimensions
        arrays[path] = {
            "shape": shape,
            "dtype": str(array.dtype),
            "dimensions": list(dimensions),
        }

    for role, path in artifact.coordinates.items():
        expected = (artifact.dimensions[role],)
        if array_dimensions[path] != expected:
            raise ValueError(
                f"Declared {role} coordinate {path!r} is on "
                f"{array_dimensions[path]!r}, expected {expected!r}."
            )
    required_grid_dimensions = set(artifact.dimensions.values())
    for channel in artifact.channels:
        channel_dimensions = set(array_dimensions[channel.array_path])
        missing = sorted(required_grid_dimensions - channel_dimensions)
        if missing:
            raise ValueError(
                f"Channel {channel.field_id!r} is missing declared grid dimensions: "
                f"{missing}"
            )
        for selector, path in channel.selector_coordinate_paths.items():
            selector_dimensions = array_dimensions[path]
            if len(selector_dimensions) != 1:
                raise ValueError(
                    f"Selector coordinate {selector!r} must be one-dimensional."
                )
            if selector_dimensions[0] not in channel_dimensions:
                raise ValueError(
                    f"Channel {channel.field_id!r} has no dimension for selector "
                    f"{selector!r}."
                )
    required_attributes: dict[str, dict[str, object]] = {}
    if required_coordinate_metadata is not None:
        for role, path in artifact.coordinates.items():
            requirements = required_coordinate_metadata.for_role(role)  # type: ignore[arg-type]
            if not requirements:
                continue
            attributes = dict(group[path].attrs)
            mismatched = {
                name: {"expected": expected, "observed": attributes.get(name)}
                for name, expected in requirements.items()
                if attributes.get(name) != expected
            }
            if mismatched:
                raise ValueError(
                    f"Coordinate {role!r} violates required public metadata: {mismatched}"
                )
            required_attributes[role] = requirements
    return {
        "store_path": artifact.store_path,
        "zarr_format": zarr_format,
        "consolidated_metadata": consolidated_metadata,
        "arrays": arrays,
        "field_ids": sorted(channel.field_id for channel in artifact.channels),
        "required_coordinate_metadata": required_attributes,
    }


def _zarr_array_dimensions(array: object) -> tuple[str, ...]:
    metadata = getattr(array, "metadata", None)
    dimensions = getattr(metadata, "dimension_names", None)
    if dimensions is None:
        attributes = dict(getattr(array, "attrs", {}))
        dimensions = attributes.get("_ARRAY_DIMENSIONS")
    if not isinstance(dimensions, (list, tuple)):
        return ()
    return tuple(str(value) for value in dimensions)


def _validate_parquet_layout(
    output_dir: Path,
    artifact: TabularDatasetArtifactLayout,
) -> dict[str, object]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Generation acceptance requires pyarrow.") from error
    path = (output_dir / artifact.file_path).resolve()
    path.relative_to(output_dir.resolve())
    metadata = pq.read_metadata(path)
    schema = metadata.schema.to_arrow_schema()
    if metadata.num_rows <= 0:
        raise ValueError("Parquet output contains no rows.")
    if schema.names != artifact.columns:
        raise ValueError("Parquet schema differs from declared columns.")
    keys = pq.read_table(path, columns=artifact.primary_key).to_pandas()
    if keys.isnull().any(axis=None):
        raise ValueError("Parquet primary key contains nulls.")
    distinct = int(keys.drop_duplicates().shape[0])
    if distinct != metadata.num_rows:
        raise ValueError("Parquet primary key contains duplicates.")
    return {
        "row_count": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "columns": schema.names,
        "primary_key_distinct": distinct,
        "size_bytes": path.stat().st_size,
    }


def _generated_source_hash(
    dataset_dir: Path,
    pipeline_dir: Path,
    manifest: PipelineManifest,
) -> str:
    paths = [dataset_dir / path for path in manifest.generated_files]
    paths.extend(
        pipeline_dir / name
        for name in ("manifest.json", "pipeline_contract.json", "run_pipeline.py")
    )
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            raise FileNotFoundError(f"Generated source file is missing: {path}")
        try:
            relative = path.relative_to(pipeline_dir.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(
                f"Generated source escapes pipeline directory: {path}"
            ) from error
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _path_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _make_read_only(path: Path) -> None:
    for item in sorted(path.rglob("*"), reverse=True):
        mode = item.stat().st_mode
        item.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _check(
    check_id: str,
    passed: bool,
    pass_code: str,
    fail_code: str,
    summary: str,
    evidence: dict[str, object],
    *,
    required: bool = True,
) -> FamilyAcceptanceCheck:
    return FamilyAcceptanceCheck(
        check_id=check_id,
        required=required,
        status="pass" if passed else "fail",
        feedback_code=pass_code if passed else fail_code,
        summary=summary,
        evidence=evidence,
    )


def _acceptance_id() -> str:
    return "acceptance_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())
