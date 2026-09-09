"""Schemas and stable identifiers for generated ETL pipelines."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.agents.contract_drafting.schemas import DatasetContract
from backend.contracts.output_policy import (
    RegularGridZarrOutputPolicy,
    StationTimeSeriesParquetOutputPolicy,
)
from backend.llm import LLMUsageSummary


PIPELINE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,127}")
SHA256_PATTERN = re.compile(r"[a-f0-9]{64}")


class PipelineVariant(StrEnum):
    """Swappable ETL pipeline generation strategies for product and research use."""

    DIRECT_LLM = "direct_llm"
    STAGED_LLM = "staged_llm"
    TERRAIO_DIRECT = "terraio_direct"
    TERRAIO_STAGED = "terraio_staged"
    TEMPLATE_HYBRID = "template_hybrid"


class PipelineConditionName(StrEnum):
    """Controlled experimental treatments independent of orchestration strategy."""

    NAIVE_LLM = "naive_llm"
    EXPERT_DIRECT_LLM = "expert_direct_llm"
    TERRAIO_REFERENCED = "terraio_referenced"
    EXPERT_DIRECT_LLM_CONCISE = "expert_direct_llm_concise"
    TERRAIO_REFERENCED_CONCISE = "terraio_referenced_concise"
    EXPERT_DIRECT_LLM_CONCISE_V2 = "expert_direct_llm_concise_v2"
    TERRAIO_REFERENCED_CONCISE_V2 = "terraio_referenced_concise_v2"
    EXPERT_DIRECT_LLM_CONCISE_V3 = "expert_direct_llm_concise_v3"
    TERRAIO_REFERENCED_CONCISE_V3 = "terraio_referenced_concise_v3"


class PromptExpertise(StrEnum):
    """Level of data-engineering guidance intentionally supplied to the model."""

    MINIMAL = "minimal"
    EXPERT = "expert"


class ReferenceContextMode(StrEnum):
    """Source-code context supplied to a generation condition."""

    NONE = "none"
    CURATED_TERRAIO = "curated_terraio"
    FROZEN_TERRAIO_REPOSITORY = "frozen_terraio_repository"


class ArtifactTargetKind(StrEnum):
    """Kind of artifact a generation condition is expected to produce."""

    STANDALONE_FAMILY_ADAPTER = "standalone_family_adapter"
    TERRAIO_REPOSITORY_EXTENSION = "terraio_repository_extension"


class ResearchRole(StrEnum):
    """How a strategy or condition participates in the thesis experiments."""

    PRIMARY_BASELINE = "primary_baseline"
    PRIMARY_CONDITION = "primary_condition"
    PRIMARY_CONTEXT_CONDITION = "primary_context_condition"
    LEGACY_ABLATION = "legacy_ablation"
    OPTIONAL_ABLATION = "optional_ablation"
    CASE_STUDY = "case_study"


class PipelineConditionMetadata(BaseModel):
    """Frozen manipulated variables for one controlled generation condition."""

    name: PipelineConditionName
    orchestration_variant: PipelineVariant
    prompt_name: str
    prompt_expertise: PromptExpertise
    reference_context_mode: ReferenceContextMode
    artifact_target_kind: ArtifactTargetKind = ArtifactTargetKind.STANDALONE_FAMILY_ADAPTER
    research_role: ResearchRole


class StrategyStatusMetadata(BaseModel):
    """Research status of an orchestration mechanism, separate from its prompt."""

    variant: PipelineVariant
    primary_matrix: bool
    research_role: ResearchRole
    note: str


class PromptSourceProvenance(BaseModel):
    """Hash of one frozen prompt template used to render a model call."""

    path: str
    sha256: str

    @field_validator("sha256")
    @classmethod
    def hash_must_be_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("Prompt source hash must be a lowercase SHA-256 digest.")
        return value


class RenderedPromptProvenance(BaseModel):
    """Exact rendered request content and configuration for one LLM call."""

    call_name: str
    model: str
    reasoning_effort: str | None = None
    max_output_tokens: int = Field(gt=0)
    tools: list[str] = Field(default_factory=list)
    system_prompt: str
    user_prompt: str
    system_sha256: str
    user_sha256: str

    @model_validator(mode="after")
    def hashes_must_match_content(self) -> "RenderedPromptProvenance":
        expected_system = hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()
        expected_user = hashlib.sha256(self.user_prompt.encode("utf-8")).hexdigest()
        if self.system_sha256 != expected_system or self.user_sha256 != expected_user:
            raise ValueError("Rendered prompt hashes do not match their exact content.")
        return self


class PromptProvenance(BaseModel):
    """Exact prompts and source hashes required to reproduce an agent run."""

    sources: list[PromptSourceProvenance] = Field(default_factory=list)
    calls: list[RenderedPromptProvenance] = Field(min_length=1)
    rendered_bundle_sha256: str

    @model_validator(mode="after")
    def bundle_hash_must_match_calls(self) -> "PromptProvenance":
        payload = []
        for call in self.calls:
            item = {
                "call_name": call.call_name,
                "system_sha256": call.system_sha256,
                "user_sha256": call.user_sha256,
                "model": call.model,
                "max_output_tokens": call.max_output_tokens,
                "tools": call.tools,
            }
            if call.reasoning_effort is not None:
                item["reasoning_effort"] = call.reasoning_effort
            payload.append(item)
        expected = stable_json_hash(payload)
        if self.rendered_bundle_sha256 != expected:
            raise ValueError("Rendered prompt bundle hash does not match its calls.")
        return self


class GenerationPromptOverride(BaseModel):
    """Frozen experiment-local prompt templates for one generation candidate."""

    schema_version: Literal["generation_prompt_override.v1"] = (
        "generation_prompt_override.v1"
    )
    prompt_id: str = Field(pattern=r"[a-z0-9][a-z0-9_-]{2,127}")
    system_prompt: Path
    user_prompt: Path
    system_sha256: str
    user_sha256: str

    @field_validator("system_sha256", "user_sha256")
    @classmethod
    def hashes_must_be_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("Prompt override hashes must be lowercase SHA-256 digests.")
        return value

    def validated_sources(self) -> tuple[Path, Path]:
        """Resolve prompt files and verify them against the frozen hashes."""

        system = self.system_prompt.resolve()
        user = self.user_prompt.resolve()
        for path, expected in (
            (system, self.system_sha256),
            (user, self.user_sha256),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Prompt override hash mismatch: {path}")
        return system, user


class ReferenceContextProvenance(BaseModel):
    """Frozen source context supplied to a context-bearing strategy."""

    repository_url: str
    base_commit: str
    selected_file_paths: list[str]
    rendered_context_size: int = Field(ge=0)
    rendered_context_sha256: str

    @field_validator("base_commit")
    @classmethod
    def commit_must_be_sha1(cls, value: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{40}", value):
            raise ValueError("Reference base_commit must be a full lowercase Git SHA-1.")
        return value

    @field_validator("rendered_context_sha256")
    @classmethod
    def context_hash_must_be_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("Reference context hash must be a lowercase SHA-256 digest.")
        return value


class GenerationMode(StrEnum):
    """Whether generated code serves one contract or a frozen dataset inventory."""

    CONTRACT_SPECIALIZED = "contract_specialized"
    DATASET_FAMILY = "dataset_family"


class AcquisitionFormat(StrEnum):
    """Provider payload formats currently supported by generation policy."""

    PROVIDER_NATIVE = "provider_native"
    GRIB = "grib"
    NETCDF = "netcdf"
    JSON = "json"
    CSV = "csv"


class PublicationFormat(StrEnum):
    """Materialized output formats currently supported by generation policy."""

    ZARR = "zarr"
    JSON = "json"
    PARQUET = "parquet"


class PipelineDataModel(StrEnum):
    """Logical data models currently supported by generated family pipelines."""

    XARRAY_DATASET = "xarray_dataset"
    RECORDS = "records"


class ZarrPublicationPolicy(RegularGridZarrOutputPolicy):
    """Fixed on-disk policy for generated Zarr artifacts."""


class ParquetPublicationPolicy(StationTimeSeriesParquetOutputPolicy):
    """Fixed long-form policy for generated station time-series artifacts."""


class PipelinePolicy(BaseModel):
    """Fixed implementation choices that do not belong in a user contract."""

    provider: str | None = None
    dataset_id: str | None = None
    acquisition_format: AcquisitionFormat = AcquisitionFormat.PROVIDER_NATIVE
    publication_format: PublicationFormat = PublicationFormat.ZARR
    data_model: PipelineDataModel = PipelineDataModel.XARRAY_DATASET
    zarr: ZarrPublicationPolicy | None = None
    parquet: ParquetPublicationPolicy | None = None
    cache_partitioning: list[str] = Field(
        default_factory=lambda: ["field", "selectors", "date"]
    )
    atomic_cache_publication: bool = True
    atomic_output_publication: bool = True
    implementation_constraints: dict[str, Any] = Field(default_factory=dict)


class PipelinePathPlaceholders(BaseModel):
    """Named path arguments exposed by the common generated-pipeline CLI."""

    contract: str = "{contract_lock_json}"
    inventory: str = "{dataset_inventory_json}"
    cache_dir: str = "{cache_dir}"
    output_dir: str = "{output_dir}"
    run_receipt: str = "{pipeline_run_json}"


class OutputArtifactContract(BaseModel):
    """Declared output produced by one successful pipeline execution."""

    artifact_id: str
    data_model: str
    storage_format: str
    path_template: str = "{output_dir}"


class OutputChannelLocation(BaseModel):
    """Neutral location of one requested field-selector channel in an output store."""

    field_id: str = Field(min_length=1)
    array_path: str = Field(min_length=1)
    selectors: dict[str, str] = Field(default_factory=dict)
    selector_coordinate_paths: dict[str, str] = Field(default_factory=dict)

    @field_validator("array_path")
    @classmethod
    def array_path_must_be_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Output array_path must be relative and must not contain '..'.")
        return value


class DatasetArtifactLayout(BaseModel):
    """Generated-code declaration of the physical layout of one published dataset."""

    schema_version: str = "dataset_artifact_layout.v1"
    storage_format: Literal["zarr"] = "zarr"
    store_path: str = Field(min_length=1)
    dimensions: dict[str, str]
    coordinates: dict[str, str]
    channels: list[OutputChannelLocation] = Field(min_length=1)

    @field_validator("store_path")
    @classmethod
    def store_path_must_be_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("Output store_path must be a non-root relative path.")
        return value

    @model_validator(mode="after")
    def layout_must_expose_regular_grid_axes(self) -> "DatasetArtifactLayout":
        required = {"sample", "y", "x"}
        if set(self.dimensions) != required or set(self.coordinates) != required:
            raise ValueError(
                "Output layout dimensions and coordinates must contain exactly sample, y, and x."
            )
        if any(not value for value in [*self.dimensions.values(), *self.coordinates.values()]):
            raise ValueError("Output layout dimension and coordinate paths must be non-empty.")
        field_ids = [channel.field_id for channel in self.channels]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("Output layout channel field_id values must be unique.")
        return self


class TabularDatasetArtifactLayout(BaseModel):
    """Generated-code declaration of one canonical Parquet table."""

    schema_version: Literal["dataset_artifact_layout.tabular.v1"] = (
        "dataset_artifact_layout.tabular.v1"
    )
    storage_format: Literal["parquet"] = "parquet"
    file_path: str = Field(min_length=1)
    columns: list[str] = Field(min_length=1)
    primary_key: list[str] = Field(min_length=1)

    @field_validator("file_path")
    @classmethod
    def file_path_must_be_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".parquet":
            raise ValueError("Parquet file_path must be a relative .parquet path.")
        return value

    @model_validator(mode="after")
    def columns_and_primary_key_are_valid(self) -> "TabularDatasetArtifactLayout":
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Parquet artifact columns must be unique.")
        if len(self.primary_key) != len(set(self.primary_key)):
            raise ValueError("Parquet artifact primary_key must be unique.")
        if not set(self.primary_key).issubset(self.columns):
            raise ValueError("Parquet primary_key must be present in columns.")
        return self


class PipelineContract(BaseModel):
    """Framework-owned execution and artifact interface for one pipeline version."""

    schema_version: str = "etl_pipeline_contract.v1"
    implementation_interface_version: Literal[
        "family_pipeline_interface.v1",
        "family_pipeline_interface.v2",
        "family_pipeline_interface.v3",
        "family_pipeline_interface.v4",
    ] = "family_pipeline_interface.v1"
    pipeline_id: str
    dataset_slug: str
    generation_mode: GenerationMode
    command_template: list[str] = Field(
        default_factory=lambda: [
            "python",
            "run_pipeline.py",
            "--contract",
            "{contract_lock_json}",
            "--inventory",
            "{dataset_inventory_json}",
            "--cache-dir",
            "{cache_dir}",
            "--output-dir",
            "{output_dir}",
            "--run-receipt",
            "{pipeline_run_json}",
        ]
    )
    success_exit_code: int = 0
    accepted_lock_schema_versions: list[str] = Field(
        default_factory=lambda: ["dataset_contract_lock.v1"]
    )
    accepted_contract_schema_versions: list[str] = Field(
        default_factory=lambda: ["dataset_contract.v1", "dataset_contract.v2"]
    )
    accepted_inventory_schema_versions: list[str] = Field(
        default_factory=lambda: ["dataset_inventory.v1"]
    )
    credential_environment_variables: list[str] = Field(default_factory=list)
    placeholders: PipelinePathPlaceholders = Field(default_factory=PipelinePathPlaceholders)
    output_artifact: OutputArtifactContract
    policy: PipelinePolicy
    generation_manifest_reference: str = "manifest.json"
    generation_manifest_hash: str

    @field_validator("pipeline_id")
    @classmethod
    def pipeline_id_must_be_safe(cls, value: str) -> str:
        if not PIPELINE_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "pipeline_id must contain 3-128 lowercase letters, numbers, underscores, or hyphens."
            )
        return value

    @field_validator("generation_manifest_hash")
    @classmethod
    def manifest_hash_must_be_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("generation_manifest_hash must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def command_must_expose_all_placeholders(self) -> "PipelineContract":
        command = set(self.command_template)
        required = set(self.placeholders.model_dump().values())
        missing = sorted(required - command)
        if missing:
            raise ValueError(
                "Pipeline command_template is missing required placeholders: "
                + ", ".join(missing)
            )
        if self.command_template[:2] != ["python", "run_pipeline.py"]:
            raise ValueError("Pipeline command_template must invoke python run_pipeline.py.")
        return self


class CacheEvidence(BaseModel):
    """Observable cache behavior reported by deterministic pipeline code."""

    cache_dir: str
    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    acquired: int = Field(default=0, ge=0)
    reused_keys: list[str] = Field(default_factory=list)
    acquired_keys: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OutputArtifactEvidence(BaseModel):
    """Fingerprint and location for one published output artifact."""

    artifact_id: str
    path: str
    storage_format: str
    size_bytes: int = Field(ge=0)
    file_count: int = Field(ge=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def output_hash_must_be_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("Output sha256 must be a lowercase SHA-256 digest.")
        return value


class PipelineRunReceipt(BaseModel):
    """Framework-validated provenance for one generated-pipeline execution."""

    schema_version: str = "etl_pipeline_run.v1"
    run_id: str
    pipeline_id: str
    manifest_hash: str
    pipeline_contract_hash: str
    contract_lock_hash: str
    inventory_hash: str
    command: list[str] = Field(min_length=1)
    started_at: str
    completed_at: str
    duration_seconds: float = Field(ge=0)
    exit_code: int
    final_status: Literal["succeeded", "failed"]
    cache: CacheEvidence
    outputs: list[OutputArtifactEvidence] = Field(default_factory=list)
    dataset_artifact: DatasetArtifactLayout | TabularDatasetArtifactLayout | None = None
    repair_run_reference: str | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)

    @field_validator(
        "manifest_hash",
        "pipeline_contract_hash",
        "contract_lock_hash",
        "inventory_hash",
    )
    @classmethod
    def hashes_must_be_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("Receipt hashes must be lowercase SHA-256 digests.")
        return value

    @model_validator(mode="after")
    def status_must_match_exit_code(self) -> "PipelineRunReceipt":
        if (self.final_status == "succeeded") != (self.exit_code == 0):
            raise ValueError("Pipeline run status and exit_code disagree.")
        if self.final_status == "succeeded" and not self.outputs:
            raise ValueError("Successful pipeline runs must declare at least one output.")
        return self


class GeneratedFile(BaseModel):
    """A file proposed by a pipeline generation strategy."""

    relative_path: str = Field(
        ...,
        description=(
            "Path relative to the dataset directory. It must stay under a historical "
            "pipeline output or pipelines/{pipeline_id}/."
        ),
    )
    content: str

    @field_validator("relative_path")
    @classmethod
    def path_must_stay_under_pipeline(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Generated file paths must be relative and must not contain '..'.")
        if not path.parts or not is_pipeline_artifact_path(path):
            raise ValueError(
                "Generated pipeline files must be under pipeline/, pipeline_*/, "
                "or pipelines/{pipeline_id}/."
            )
        if path.name == "":
            raise ValueError("Generated file path must include a filename.")
        return path.as_posix()


class PipelineManifest(BaseModel):
    """Immutable provenance for how one ETL pipeline version was generated."""

    schema_version: str = "etl_pipeline_manifest.v2"
    dataset_slug: str = ""
    variant: PipelineVariant = PipelineVariant.DIRECT_LLM
    prompt_name: str = ""
    model: str = ""
    contract_hash: str = ""
    output_dir: str = "pipeline"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    generated_files: list[str] = Field(default_factory=list)
    reference_context_files: list[str] = Field(default_factory=list)
    reference_context_size: int | None = None
    reference_context: ReferenceContextProvenance | None = None
    condition: PipelineConditionMetadata | None = None
    strategy_status: StrategyStatusMetadata | None = None
    prompt_provenance: PromptProvenance | None = None
    llm_usage: LLMUsageSummary | None = None
    generation_mode: GenerationMode = GenerationMode.CONTRACT_SPECIALIZED
    pipeline_id: str | None = None
    inventory_hash: str | None = None
    inventory_schema_version: str | None = None
    seed_contract_hash: str | None = None
    pipeline_contract_hash: str | None = None
    fixed_policy: PipelinePolicy | None = None
    notes: list[str] = Field(default_factory=list)


class PipelineGenerationResult(BaseModel):
    """Structured output from a pipeline generation strategy."""

    manifest: PipelineManifest
    files: list[GeneratedFile] = Field(default_factory=list)
    tests: list[GeneratedFile] = Field(default_factory=list)


def stable_json_hash(value: Any) -> str:
    """Hash JSON-compatible content using sorted compact serialization."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contract_hash(contract: DatasetContract) -> str:
    """Return a stable hash for the contract content used to generate a pipeline."""

    return stable_json_hash(contract)


def pipeline_manifest_hash(manifest: PipelineManifest) -> str:
    """Hash immutable generation provenance without its reciprocal contract link."""

    payload = manifest.model_dump(mode="json", exclude={"pipeline_contract_hash"})
    if manifest.schema_version in {"etl_pipeline_manifest.v2", "etl_pipeline_manifest.v3"}:
        for v4_field in (
            "reference_context",
            "condition",
            "strategy_status",
            "prompt_provenance",
        ):
            payload.pop(v4_field, None)
    if (
        manifest.fixed_policy is not None
        and "zarr" not in manifest.fixed_policy.model_fields_set
        and isinstance(payload.get("fixed_policy"), dict)
    ):
        payload["fixed_policy"].pop("zarr", None)
    if (
        manifest.fixed_policy is not None
        and "parquet" not in manifest.fixed_policy.model_fields_set
        and isinstance(payload.get("fixed_policy"), dict)
    ):
        payload["fixed_policy"].pop("parquet", None)
    return stable_json_hash(payload)


def pipeline_contract_hash(contract: PipelineContract) -> str:
    """Hash the complete framework-owned pipeline execution contract."""

    payload = contract.model_dump(mode="json")
    if contract.schema_version == "etl_pipeline_contract.v1":
        payload.pop("implementation_interface_version", None)
    if (
        "zarr" not in contract.policy.model_fields_set
        and isinstance(payload.get("policy"), dict)
    ):
        payload["policy"].pop("zarr", None)
    if (
        "parquet" not in contract.policy.model_fields_set
        and isinstance(payload.get("policy"), dict)
    ):
        payload["policy"].pop("parquet", None)
    return stable_json_hash(payload)


def is_pipeline_artifact_path(path: PurePosixPath) -> bool:
    """Return whether a generated path is under an allowed pipeline root."""

    if not path.parts:
        return False
    if is_pipeline_output_dir(path.parts[0]):
        return len(path.parts) >= 2
    return (
        len(path.parts) >= 3
        and path.parts[0] == "pipelines"
        and PIPELINE_ID_PATTERN.fullmatch(path.parts[1]) is not None
    )


def is_pipeline_output_dir(value: str) -> bool:
    """Return whether a historical pipeline output directory name is allowed."""

    return value == "pipeline" or re.fullmatch(
        r"pipeline_[a-z0-9_]+_\d{8}T\d{6}Z",
        value,
    ) is not None


def pipeline_experiment_slug(
    variant: PipelineVariant,
    generated_at: datetime | None = None,
) -> str:
    """Create a stable experiment output directory name for a pipeline variant."""

    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"pipeline_{variant.value}_{timestamp}"


def pipeline_version_id(
    dataset_slug: str,
    variant: PipelineVariant,
    inventory_hash: str,
    generated_at: datetime | None = None,
    *,
    label: str | None = None,
) -> str:
    """Create a unique, inspectable identifier for a reusable pipeline version."""

    normalized_slug = re.sub(r"[^a-z0-9_-]+", "_", dataset_slug.lower()).strip("_")
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    normalized_label = re.sub(
        r"[^a-z0-9_-]+",
        "_",
        (label or variant.value).lower(),
    ).strip("_")
    return f"{normalized_slug}_{normalized_label}_{inventory_hash[:8]}_{timestamp}"[:128]
