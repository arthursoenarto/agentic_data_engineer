"""Build trusted regular-grid suites from generation acceptance evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.contracts.output_policy import (
    RegularGridZarrOutputPolicy,
    StationTimeSeriesParquetOutputPolicy,
)
from backend.evaluation.check_library import EvaluationProfile
from backend.evaluation.constrained_schemas import (
    ArrayLocation,
    CandidateSourceSpec,
    ComparisonSettings,
    ConsumerWorkloadSpec,
    CoordinateLocation,
    ExpectedGridShape,
    GridSideSpec,
    SecuritySettings,
)
from backend.evaluation.evidence import (
    canonical_json_file_hash,
    frozen_path_hash,
    source_bundle_hash,
)
from backend.evaluation.objective_v3_schemas import (
    ConstrainedEvaluationConfigV3,
    EngineeringJudgeSettingsV3,
)
from backend.evaluation.station_parquet_schemas import (
    StationParquetEvaluationConfig,
    StationTableSpec,
    StationWorkloadSpec,
)
from backend.evaluation.target_mapping import resolve_cf_datetime_coordinate


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateAxisPolicy(_StrictModel):
    """Dataset-owned assertions applied to an accepted physical coordinate."""

    normalization: Literal[
        "none",
        "ascending",
        "longitude_modulo_360",
        "cf_datetime",
        "cf_datetime_ascending",
    ] = "none"
    expected_dtype: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_fill_value: str | int | float | bool | None = None
    monotonic: Literal["increasing", "decreasing"] | None = None
    expected_step: float | None = Field(default=None, gt=0.0)
    expected_step_seconds: float | None = Field(default=None, gt=0.0)

    def bind(self, *, store_path: str, array_path: str) -> CoordinateLocation:
        return CoordinateLocation(
            store_path=store_path,
            array_path=array_path,
            **self.model_dump(mode="json"),
        )


class CandidateGridPolicy(_StrictModel):
    """Logical-axis policies independent of candidate physical names."""

    sample: CandidateAxisPolicy
    y: CandidateAxisPolicy
    x: CandidateAxisPolicy


class ReferenceChannelSpec(_StrictModel):
    """Evaluator-owned location for one exact contract field-selector identity."""

    field_id: str
    location: ArrayLocation
    expected_dtype: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_fill_value: str | int | float | bool | None = None


class AcceptedRegularGridSuiteSpec(_StrictModel):
    """Dataset-local facts needed to instantiate reusable deterministic checks."""

    schema_version: Literal["accepted_regular_grid_suite_spec.v1"] = (
        "accepted_regular_grid_suite_spec.v1"
    )
    suite_id_prefix: str
    suite_version: str
    description: str
    dataset_dir: Path
    contract_lock: Path
    inventory: Path
    reference_artifact: Path
    reference_provenance: str
    expected_shape: ExpectedGridShape
    candidate_grid_policy: CandidateGridPolicy
    reference_grid: GridSideSpec
    reference_channels: list[ReferenceChannelSpec] = Field(min_length=1)
    comparison: ComparisonSettings = Field(default_factory=ComparisonSettings)
    workload: ConsumerWorkloadSpec = Field(default_factory=ConsumerWorkloadSpec)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    execution_timeout_seconds: float = Field(default=3600.0, gt=0.0)
    output_open_latency_repetitions: int = Field(default=5, ge=1, le=20)
    engineering_quality: EngineeringJudgeSettingsV3 = Field(
        default_factory=EngineeringJudgeSettingsV3
    )
    evaluation_profile: EvaluationProfile

    @model_validator(mode="after")
    def reference_channels_are_unique(self) -> "AcceptedRegularGridSuiteSpec":
        field_ids = [channel.field_id for channel in self.reference_channels]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("Reference channel field IDs must be unique")
        return self


class AcceptedStationParquetSuiteSpec(_StrictModel):
    """Dataset-local facts for a station-time-series Parquet suite."""

    schema_version: Literal["accepted_station_parquet_suite_spec.v1"] = (
        "accepted_station_parquet_suite_spec.v1"
    )
    suite_id_prefix: str
    suite_version: str
    dataset_dir: Path
    contract_lock: Path
    inventory: Path
    reference_artifact: Path
    reference_provenance: str
    table: StationTableSpec
    workload: StationWorkloadSpec = Field(default_factory=StationWorkloadSpec)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    execution_timeout_seconds: float = Field(default=3600.0, gt=0.0)
    engineering_quality: EngineeringJudgeSettingsV3 = Field(
        default_factory=EngineeringJudgeSettingsV3
    )
    evaluation_profile: EvaluationProfile


def load_accepted_regular_grid_suite_spec(
    path: Path,
) -> AcceptedRegularGridSuiteSpec:
    """Read and validate one dataset-local YAML suite specification."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Suite specification must contain a mapping: {path}")
    return AcceptedRegularGridSuiteSpec.model_validate(payload)


def load_accepted_station_parquet_suite_spec(
    path: Path,
) -> AcceptedStationParquetSuiteSpec:
    """Read and validate one dataset-local station suite specification."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Suite specification must contain a mapping: {path}")
    return AcceptedStationParquetSuiteSpec.model_validate(payload)


def build_accepted_station_parquet_suite(
    *,
    spec: AcceptedStationParquetSuiteSpec,
    candidate_id: str,
    repository_root: Path,
    output_dir: Path,
    python_executable: str | None = None,
) -> Path:
    """Bind accepted Parquet paths to a frozen dataset-owned suite."""

    root = repository_root.resolve()
    dataset_dir = _resolve(spec.dataset_dir, root)
    contract_lock = _resolve(spec.contract_lock, root)
    inventory = _resolve(spec.inventory, root)
    reference_artifact = _resolve(spec.reference_artifact, root)
    candidate_dir = dataset_dir / "pipelines" / candidate_id
    manifest_path = candidate_dir / "manifest.json"
    pipeline_contract_path = candidate_dir / "pipeline_contract.json"
    acceptance_path, acceptance = _latest_passed_acceptance_for(
        candidate_dir,
        policy_check_id="generation.real_parquet_layout",
    )
    artifact = acceptance.get("dataset_artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"Acceptance has no dataset artifact: {acceptance_path}")
    if artifact.get("storage_format") != "parquet":
        raise ValueError("Acceptance artifact is not Parquet")
    if artifact.get("primary_key") != ["station_id", "timestamp", "field_id"]:
        raise ValueError("Acceptance artifact has a non-canonical primary key")

    manifest = _read_json(manifest_path)
    pipeline_contract = _read_json(pipeline_contract_path)
    output_policy = StationTimeSeriesParquetOutputPolicy.model_validate(
        pipeline_contract["policy"]["parquet"]
    )
    frozen_cache = _resolve(Path(str(acceptance["frozen_cache_path"])), root)

    generated_paths = [
        str((dataset_dir / relative).relative_to(root))
        for relative in manifest["generated_files"]
    ]
    source_spec = CandidateSourceSpec(
        cwd=str(candidate_dir.relative_to(root)),
        paths=generated_paths,
        sha256="0" * 64,
    )
    source_spec.sha256 = source_bundle_hash(source_spec, root)

    payload = {
        "schema_version": "evaluation_constrained.station_parquet.v1",
        "suite_id": f"{spec.suite_id_prefix}_{candidate_id}",
        "suite_version": spec.suite_version,
        "candidate_id": candidate_id,
        "contract_lock": _frozen_file(contract_lock, root),
        "inventory": _frozen_file(inventory, root),
        "candidate_source": source_spec.model_dump(mode="json"),
        "execution": {
            "command": pipeline_contract["command_template"],
            "python_executable": python_executable,
            "timeout_seconds": spec.execution_timeout_seconds,
            "pipeline_contract": _frozen_file(pipeline_contract_path, root),
            "manifest": _frozen_file(manifest_path, root),
            "frozen_cache": _frozen_path(frozen_cache, root),
        },
        "reference": {
            "artifact": _frozen_path(reference_artifact, root),
            "content_id": f"sha256:{frozen_path_hash(reference_artifact)}",
            "provenance": spec.reference_provenance,
        },
        "table": spec.table.model_dump(mode="json"),
        "output_policy": output_policy.model_dump(mode="json"),
        "workload": spec.workload.model_dump(mode="json"),
        "security": spec.security.model_dump(mode="json"),
        "execution_policy": {
            "backend": "auto",
            "environment_overrides": {
                "PIPELINE_ACCEPTANCE_REFERENCE": str(
                    acceptance_path.relative_to(root)
                )
            },
        },
        "engineering_quality": spec.engineering_quality.model_dump(mode="json"),
        "evaluation_profile": spec.evaluation_profile.model_dump(mode="json"),
    }
    config = StationParquetEvaluationConfig.model_validate(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{candidate_id}.json"
    if destination.exists():
        existing = StationParquetEvaluationConfig.model_validate_json(
            destination.read_text(encoding="utf-8")
        )
        if existing != config:
            raise FileExistsError(f"Existing suite differs: {destination}")
        return destination
    destination.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return destination


def build_accepted_regular_grid_suite(
    *,
    spec: AcceptedRegularGridSuiteSpec,
    candidate_id: str,
    repository_root: Path,
    output_dir: Path,
    python_executable: str | None = None,
) -> Path:
    """Bind accepted physical paths to a frozen dataset-owned suite."""

    root = repository_root.resolve()
    dataset_dir = _resolve(spec.dataset_dir, root)
    contract_lock = _resolve(spec.contract_lock, root)
    inventory = _resolve(spec.inventory, root)
    reference_artifact = _resolve(spec.reference_artifact, root)
    candidate_dir = dataset_dir / "pipelines" / candidate_id
    manifest_path = candidate_dir / "manifest.json"
    pipeline_contract_path = candidate_dir / "pipeline_contract.json"
    acceptance_path, acceptance = _latest_passed_acceptance(candidate_dir)
    artifact = acceptance.get("dataset_artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"Acceptance has no dataset artifact: {acceptance_path}")

    accepted_channels = artifact.get("channels")
    if not isinstance(accepted_channels, list):
        raise ValueError("Acceptance dataset artifact channels must be a list")
    accepted_by_id = {
        str(channel["field_id"]): channel
        for channel in accepted_channels
        if isinstance(channel, dict) and "field_id" in channel
    }
    reference_by_id = {
        channel.field_id: channel for channel in spec.reference_channels
    }
    if set(accepted_by_id) != set(reference_by_id):
        raise ValueError(
            "Accepted channels must match suite reference channels exactly: "
            f"accepted={sorted(accepted_by_id)}, "
            f"reference={sorted(reference_by_id)}"
        )

    manifest = _read_json(manifest_path)
    pipeline_contract = _read_json(pipeline_contract_path)
    output_policy = RegularGridZarrOutputPolicy.model_validate(
        pipeline_contract["policy"]["zarr"]
    )
    frozen_cache = _resolve(Path(str(acceptance["frozen_cache_path"])), root)
    initial_receipt = _resolve(Path(str(acceptance["initial_run_receipt"])), root)
    accepted_store = (
        initial_receipt.parent / "output" / str(artifact["store_path"])
    )

    dimensions = artifact.get("dimensions")
    coordinates = artifact.get("coordinates")
    if not isinstance(dimensions, dict) or not isinstance(coordinates, dict):
        raise ValueError("Acceptance artifact requires dimensions and coordinates")
    required_roles = {"sample", "y", "x"}
    if set(dimensions) != required_roles or set(coordinates) != required_roles:
        raise ValueError("Acceptance dimensions/coordinates must define sample/y/x")

    candidate_store = f"{{output_dir}}/{artifact['store_path']}"
    sample_coordinate = resolve_cf_datetime_coordinate(
        accepted_store,
        declared_path=str(coordinates["sample"]),
        sample_dimension=str(dimensions["sample"]),
    )
    candidate_grid = {
        "dimensions": dimensions,
        "sample_coordinate": spec.candidate_grid_policy.sample.bind(
            store_path=candidate_store,
            array_path=sample_coordinate,
        ).model_dump(mode="json"),
        "y_coordinate": spec.candidate_grid_policy.y.bind(
            store_path=candidate_store,
            array_path=str(coordinates["y"]),
        ).model_dump(mode="json"),
        "x_coordinate": spec.candidate_grid_policy.x.bind(
            store_path=candidate_store,
            array_path=str(coordinates["x"]),
        ).model_dump(mode="json"),
    }
    channels = []
    for reference in spec.reference_channels:
        accepted = accepted_by_id[reference.field_id]
        channels.append(
            {
                "field_id": reference.field_id,
                "candidate": {
                    "store_path": candidate_store,
                    "array_path": accepted["array_path"],
                    "selectors": accepted.get("selectors", {}),
                    "selector_coordinate_paths": accepted.get(
                        "selector_coordinate_paths", {}
                    ),
                },
                "reference": reference.location.model_dump(mode="json"),
                "expected_dtype": reference.expected_dtype,
                "metadata": reference.metadata,
                "expected_fill_value": reference.expected_fill_value,
            }
        )

    generated_paths = [
        str((dataset_dir / relative).relative_to(root))
        for relative in manifest["generated_files"]
    ]
    source_spec = CandidateSourceSpec(
        cwd=str(candidate_dir.relative_to(root)),
        paths=generated_paths,
        sha256="0" * 64,
    )
    source_spec.sha256 = source_bundle_hash(source_spec, root)

    payload = {
        "schema_version": "evaluation_constrained.regular_grid_zarr.v3",
        "suite_id": f"{spec.suite_id_prefix}_{candidate_id}",
        "suite_version": spec.suite_version,
        "candidate_id": candidate_id,
        "description": spec.description,
        "contract_lock": _frozen_file(contract_lock, root),
        "inventory": _frozen_file(inventory, root),
        "candidate_source": source_spec.model_dump(mode="json"),
        "execution": {
            "command": pipeline_contract["command_template"],
            "python_executable": python_executable,
            "timeout_seconds": spec.execution_timeout_seconds,
            "pipeline_contract": _frozen_file(pipeline_contract_path, root),
            "manifest": _frozen_file(manifest_path, root),
            "frozen_cache": _frozen_path(frozen_cache, root),
        },
        "reference": {
            "artifact": _frozen_path(reference_artifact, root),
            "content_id": f"sha256:{frozen_path_hash(reference_artifact)}",
            "provenance": spec.reference_provenance,
        },
        "grid": {
            "expected_shape": spec.expected_shape.model_dump(mode="json"),
            "candidate_grid": candidate_grid,
            "reference_grid": spec.reference_grid.model_dump(mode="json"),
            "channels": channels,
        },
        "comparison": spec.comparison.model_dump(mode="json"),
        "workload": spec.workload.model_dump(mode="json"),
        "security": spec.security.model_dump(mode="json"),
        "output_policy": {
            **output_policy.model_dump(mode="json"),
            "open_latency_repetitions": spec.output_open_latency_repetitions,
        },
        "execution_policy": {
            "backend": "auto",
            "environment_overrides": {
                "PIPELINE_ACCEPTANCE_REFERENCE": str(
                    acceptance_path.relative_to(root)
                )
            },
        },
        "engineering_quality": spec.engineering_quality.model_dump(mode="json"),
        "evaluation_profile": spec.evaluation_profile.model_dump(mode="json"),
    }
    config = ConstrainedEvaluationConfigV3.model_validate(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{candidate_id}.json"
    if destination.exists():
        existing = ConstrainedEvaluationConfigV3.model_validate_json(
            destination.read_text(encoding="utf-8")
        )
        if existing != config:
            raise FileExistsError(f"Existing suite differs: {destination}")
        return destination
    destination.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return destination


def _latest_passed_acceptance(candidate_dir: Path) -> tuple[Path, dict[str, Any]]:
    return _latest_passed_acceptance_for(
        candidate_dir,
        policy_check_id="generation.zarr_v3_consolidated",
    )


def _latest_passed_acceptance_for(
    candidate_dir: Path,
    *,
    policy_check_id: str,
) -> tuple[Path, dict[str, Any]]:
    reports = sorted((candidate_dir / "acceptance").glob("*/acceptance.json"))
    if not reports:
        raise FileNotFoundError(f"No acceptance report found for {candidate_dir.name}")
    passed: list[tuple[Path, dict[str, Any]]] = []
    for path in reports:
        payload = _read_json(path)
        if payload.get("schema_version") != "family_pipeline_acceptance.v1":
            raise ValueError(f"Unsupported acceptance report: {path}")
        if payload.get("final_status") == "passed":
            passed.append((path, payload))
    if not passed:
        raise ValueError(f"Candidate has no passed acceptance: {candidate_dir.name}")
    path, payload = passed[-1]
    policy = next(
        (
            check
            for check in payload.get("checks", [])
            if check.get("check_id") == policy_check_id
        ),
        None,
    )
    if not isinstance(policy, dict) or policy.get("status") != "pass":
        raise ValueError(f"Candidate lacks accepted evidence for {policy_check_id}")
    return path, payload


def _resolve(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _frozen_file(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": canonical_json_file_hash(path),
    }


def _frozen_path(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": frozen_path_hash(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload
