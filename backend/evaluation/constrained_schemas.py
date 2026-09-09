"""Typed contracts for constrained regular-grid-to-Zarr evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.contracts.output_policy import RegularGridZarrOutputPolicy
from backend.evaluation.core_schemas import CommandObservation
from backend.evaluation.schemas import CheckConfidence, CheckStatus


ConstraintId = Literal[
    "contract_correctness",
    "semantic_equivalence",
    "rerun_safety",
    "provenance_security",
]
ObjectiveName = Literal[
    "materialization_seconds",
    "consumer_samples_per_second",
    "output_bytes",
]
ObjectiveValue = int | float
Scalar = str | int | float | bool


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenFile(_StrictModel):
    """A suite-owned file pinned by canonical JSON SHA-256."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FrozenPath(_StrictModel):
    """A suite-owned file or directory pinned by deterministic byte hash."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CandidateSourceSpec(_StrictModel):
    """Explicit source bundle; dependencies remain in the candidate workspace."""

    cwd: str
    paths: list[str] = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DimensionRoles(_StrictModel):
    """Physical dimension names assigned to the three supported logical roles."""

    sample: str
    y: str
    x: str

    @model_validator(mode="after")
    def names_must_be_distinct(self) -> "DimensionRoles":
        if len({self.sample, self.y, self.x}) != 3:
            raise ValueError("sample, y, and x dimension names must be distinct")
        return self


class ExpectedGridShape(_StrictModel):
    """Frozen logical dimensions for the reference benchmark."""

    sample: int = Field(ge=1)
    y: int = Field(ge=1)
    x: int = Field(ge=1)


class ArrayLocation(_StrictModel):
    """One explicitly located Zarr array and any declared extra-dimension fixes."""

    store_path: str
    array_path: str
    selectors: dict[str, Scalar] = Field(default_factory=dict)
    indices: dict[str, int] = Field(default_factory=dict)
    selector_coordinate_paths: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fixed_dimensions_must_be_unambiguous(self) -> "ArrayLocation":
        overlap = set(self.selectors) & set(self.indices)
        if overlap:
            raise ValueError(
                f"Dimensions cannot use both selectors and indices: {sorted(overlap)}"
            )
        unknown = set(self.selector_coordinate_paths) - set(self.selectors)
        if unknown:
            raise ValueError(
                "selector_coordinate_paths has no matching selector for "
                f"{sorted(unknown)}"
            )
        if any(index < 0 for index in self.indices.values()):
            raise ValueError("Array indices must be non-negative")
        return self


class CoordinateLocation(_StrictModel):
    """Coordinate-array location and explicitly authorized normalization."""

    store_path: str
    array_path: str
    normalization: Literal[
        "none",
        "ascending",
        "longitude_modulo_360",
        "cf_datetime",
        "cf_datetime_ascending",
    ] = "none"
    expected_dtype: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_fill_value: Scalar | None = None
    monotonic: Literal["increasing", "decreasing"] | None = None
    expected_step: float | None = Field(default=None, gt=0.0)
    expected_step_seconds: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def step_assertion_must_be_unambiguous(self) -> "CoordinateLocation":
        if self.expected_step is not None and self.expected_step_seconds is not None:
            raise ValueError(
                "Use expected_step for numeric coordinates or "
                "expected_step_seconds for datetime coordinates, not both"
            )
        return self


class GridSideSpec(_StrictModel):
    """Physical mapping for one side of a logical regular grid."""

    dimensions: DimensionRoles
    sample_coordinate: CoordinateLocation
    y_coordinate: CoordinateLocation
    x_coordinate: CoordinateLocation


class ValuePolicy(_StrictModel):
    """Native-value and missingness comparison policy."""

    atol: float = Field(default=0.0, ge=0.0)
    rtol: float = Field(default=0.0, ge=0.0)
    equal_nan: bool = True
    missing_values: list[float] = Field(default_factory=list)


class LogicalChannelSpec(_StrictModel):
    """One exact contract field-selector combination on both grid sides."""

    field_id: str
    candidate: ArrayLocation
    reference: ArrayLocation
    expected_dtype: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_fill_value: Scalar | None = None
    value_policy: ValuePolicy | None = None


class RegularGridZarrSpec(_StrictModel):
    """Dataset-class policy for regular rectilinear arrays published as Zarr."""

    dataset_class: Literal["regular_rectilinear_grid"] = "regular_rectilinear_grid"
    output_format: Literal["zarr"] = "zarr"
    expected_shape: ExpectedGridShape
    candidate_grid: GridSideSpec
    reference_grid: GridSideSpec
    channels: list[LogicalChannelSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def channel_ids_must_be_unique(self) -> "RegularGridZarrSpec":
        ids = [channel.field_id for channel in self.channels]
        if len(ids) != len(set(ids)):
            raise ValueError("Logical channel field_id values must be unique")
        return self


class CommandTemplateSpec(_StrictModel):
    """Black-box command and frozen inputs under evaluator control."""

    command: list[str] = Field(min_length=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    pipeline_contract: FrozenFile
    manifest: FrozenFile
    frozen_cache: FrozenPath
    python_executable: str | None = None

    @model_validator(mode="after")
    def required_placeholders_must_be_exact_tokens(self) -> "CommandTemplateSpec":
        required = {
            "{contract_lock_json}",
            "{dataset_inventory_json}",
            "{cache_dir}",
            "{output_dir}",
            "{pipeline_run_json}",
        }
        observed = {token for token in self.command if token in required}
        if observed != required:
            raise ValueError(
                "command must expose each controlled path placeholder exactly once"
            )
        if any(self.command.count(token) != 1 for token in required):
            raise ValueError("Each controlled path placeholder must occur exactly once")
        embedded = [
            token for token in self.command if "{" in token and token not in required
        ]
        if embedded:
            raise ValueError(
                f"Unsupported or embedded command placeholders: {embedded}"
            )
        return self


class ReferenceSpec(_StrictModel):
    """Frozen independent oracle and its provenance note."""

    artifact: FrozenPath
    content_id: str = Field(min_length=1)
    provenance: str = Field(min_length=1)


class ComparisonSettings(_StrictModel):
    """Bounded deterministic validation settings."""

    coordinates_atol: float = Field(default=0.0, ge=0.0)
    value_policy: ValuePolicy = Field(default_factory=ValuePolicy)
    max_block_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    mismatch_examples: int = Field(default=8, ge=1, le=100)


class ConsumerWorkloadSpec(_StrictModel):
    """Primary ML-facing workload; optional alternatives belong in ablations."""

    kind: Literal["full_field_tensor"] = "full_field_tensor"
    protocol: Literal["legacy_tensor_sum.v1", "tensor_loading.v1"] = (
        "legacy_tensor_sum.v1"
    )
    tensor_layout: Literal["C,H,W"] = "C,H,W"
    dtype: Literal["float32"] = "float32"
    access_pattern: Literal["shuffled"] = "shuffled"
    cache_mode: Literal["warm"] = "warm"
    warmup_passes: int = Field(default=1, ge=1)
    repetitions: int = Field(default=3, ge=1)
    max_samples: int | None = Field(default=None, ge=1)
    seed: int = 20260803


class SecuritySettings(_StrictModel):
    """Secret sources and scan bounds frozen into the suite."""

    secret_environment_variables: list[str] = Field(default_factory=list)
    max_text_file_bytes: int = Field(default=1_000_000, ge=1024)


class ZarrOutputPolicy(RegularGridZarrOutputPolicy):
    """Required candidate storage policy and bounded local diagnostics."""

    open_latency_repetitions: int = Field(default=5, ge=1, le=20)

    def public_policy(self) -> RegularGridZarrOutputPolicy:
        """Return the generation-visible part of this evaluation policy."""

        return RegularGridZarrOutputPolicy.model_validate(
            self.model_dump(mode="json", exclude={"open_latency_repetitions"})
        )


class ConstrainedEvaluationConfig(_StrictModel):
    """Trusted suite for constrained regular-grid-to-Zarr comparison."""

    schema_version: Literal["evaluation_constrained.regular_grid_zarr.v2"] = (
        "evaluation_constrained.regular_grid_zarr.v2"
    )
    suite_id: str
    suite_version: str
    candidate_id: str
    description: str | None = None
    contract_lock: FrozenFile
    inventory: FrozenFile
    candidate_source: CandidateSourceSpec
    execution: CommandTemplateSpec
    reference: ReferenceSpec
    grid: RegularGridZarrSpec
    comparison: ComparisonSettings = Field(default_factory=ComparisonSettings)
    workload: ConsumerWorkloadSpec = Field(default_factory=ConsumerWorkloadSpec)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    output_policy: ZarrOutputPolicy | None = None

    @model_validator(mode="after")
    def candidate_outputs_must_be_controlled(self) -> "ConstrainedEvaluationConfig":
        candidate_paths = {
            channel.candidate.store_path for channel in self.grid.channels
        }
        candidate_paths.update(
            {
                self.grid.candidate_grid.sample_coordinate.store_path,
                self.grid.candidate_grid.y_coordinate.store_path,
                self.grid.candidate_grid.x_coordinate.store_path,
            }
        )
        if any("{output_dir}" not in path for path in candidate_paths):
            raise ValueError(
                "Every candidate Zarr store must be rooted at {output_dir}"
            )
        return self


class EvaluationCheck(_StrictModel):
    """Stable actionable evidence for one deterministic check."""

    check_id: str
    status: CheckStatus
    severity: Literal["error", "warning", "info"]
    summary: str
    feedback_code: str
    expected: Any | None = None
    observed: Any | None = None
    delta: Any | None = None
    affected_slice: dict[str, Any] | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: CheckConfidence = CheckConfidence.HIGH
    duration_seconds: float | None = Field(default=None, ge=0.0)


class ConstraintResult(_StrictModel):
    """One hard constraint; only PASS counts toward feasibility."""

    constraint: ConstraintId
    status: CheckStatus
    passed: bool
    checks: list[EvaluationCheck]

    @model_validator(mode="after")
    def pass_flag_must_match_status(self) -> "ConstraintResult":
        if self.passed != (self.status == CheckStatus.PASS):
            raise ValueError("Constraint passed flag must match PASS status")
        return self


class FeedbackSignal(_StrictModel):
    """Compact repair/search signal without leaking sensitive evidence."""

    constraint: ConstraintId
    code: str
    message: str
    affected_channels: list[str] = Field(default_factory=list)


class CompactEvaluationSummary(_StrictModel):
    """The only high-level signal consumed by a future search/meta agent."""

    target: str
    constraints: dict[ConstraintId, CheckStatus]
    feasible: bool
    diagnostic_metrics: dict[ObjectiveName, ObjectiveValue] = Field(
        default_factory=dict
    )
    objectives: dict[ObjectiveName, ObjectiveValue] = Field(default_factory=dict)
    objective_directions: dict[ObjectiveName, Literal["minimize", "maximize"]] = Field(
        default_factory=lambda: {
            "materialization_seconds": "minimize",
            "consumer_samples_per_second": "maximize",
            "output_bytes": "minimize",
        }
    )
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    feedback_codes: list[str] = Field(default_factory=list)
    feedback: list[FeedbackSignal] = Field(default_factory=list)

    @model_validator(mode="after")
    def objectives_exist_only_for_feasible_candidates(
        self,
    ) -> "CompactEvaluationSummary":
        required = {
            "materialization_seconds",
            "consumer_samples_per_second",
            "output_bytes",
        }
        required_constraints = {
            "contract_correctness",
            "semantic_equivalence",
            "rerun_safety",
            "provenance_security",
        }
        if set(self.constraints) != required_constraints:
            raise ValueError("Exactly the four frozen constraints are required")
        if set(self.objective_directions) != required:
            raise ValueError(
                "Exactly the three frozen objective directions are required"
            )
        if self.feasible != all(
            status == CheckStatus.PASS for status in self.constraints.values()
        ):
            raise ValueError("feasible must equal the conjunction of all constraints")
        if self.feasible and set(self.objectives) != required:
            raise ValueError("A feasible result must contain exactly three objectives")
        if not self.feasible and self.objectives:
            raise ValueError("An infeasible result must not contain objectives")
        if not set(self.diagnostic_metrics).issubset(required):
            raise ValueError("Diagnostic metrics must be a subset of the objectives")
        if self.feasible and self.diagnostic_metrics != self.objectives:
            raise ValueError(
                "A feasible result's diagnostic metrics must match its objectives"
            )
        return self


class ScenarioResult(_StrictModel):
    """Evaluator-controlled materialization evidence for one fresh scenario."""

    scenario: Literal["initial", "rerun"]
    setup_seconds: float = Field(ge=0.0)
    execution: CommandObservation
    validation_seconds: float = Field(ge=0.0)
    total_seconds: float = Field(ge=0.0)
    output_path: str
    receipt_path: str


class ConstrainedEnvironment(_StrictModel):
    """Host and dependency identity for reproducible objective measurements."""

    platform: str
    platform_release: str
    machine: str
    processor: str
    python_version: str
    cpu_count: int | None
    software: dict[str, str]


class ConstrainedEvaluationRun(_StrictModel):
    """Immutable detailed and compact constrained-evaluation report."""

    schema_version: Literal["evaluation_constrained_run.v2"] = (
        "evaluation_constrained_run.v2"
    )
    suite_id: str
    suite_version: str
    run_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    config_file: str
    environment: ConstrainedEnvironment
    constraints: list[ConstraintResult]
    scenarios: list[ScenarioResult] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)
    summary: CompactEvaluationSummary
    duration_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def all_constraints_must_be_present_once(self) -> "ConstrainedEvaluationRun":
        expected = {
            "contract_correctness",
            "semantic_equivalence",
            "rerun_safety",
            "provenance_security",
        }
        observed = [item.constraint for item in self.constraints]
        if set(observed) != expected or len(observed) != len(expected):
            raise ValueError("Each hard constraint must occur exactly once")
        return self


class NeutralOutputChannelLocation(_StrictModel):
    """Evaluator-owned view of one generated output channel location."""

    field_id: str = Field(min_length=1)
    array_path: str = Field(min_length=1)
    selectors: dict[str, str] = Field(default_factory=dict)
    selector_coordinate_paths: dict[str, str] = Field(default_factory=dict)


class NeutralDatasetArtifactLayout(_StrictModel):
    """Evaluator-owned view of additive physical-layout receipt evidence."""

    schema_version: Literal["dataset_artifact_layout.v1"]
    storage_format: Literal["zarr"]
    store_path: str = Field(min_length=1)
    dimensions: dict[str, str]
    coordinates: dict[str, str]
    channels: list[NeutralOutputChannelLocation] = Field(min_length=1)

    @model_validator(mode="after")
    def axes_must_be_complete(self) -> "NeutralDatasetArtifactLayout":
        axes = {"sample", "y", "x"}
        if set(self.dimensions) != axes or set(self.coordinates) != axes:
            raise ValueError("Dataset artifact layout must declare sample, y, and x axes")
        return self


class NeutralPipelineReceipt(_StrictModel):
    """Evaluator-owned adapter for the generated-pipeline receipt protocol."""

    schema_version: Literal["etl_pipeline_run.v1"] = "etl_pipeline_run.v1"
    run_id: str
    pipeline_id: str
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pipeline_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    command: list[str] = Field(min_length=1)
    started_at: str
    completed_at: str
    duration_seconds: float = Field(ge=0.0)
    exit_code: int
    final_status: Literal["succeeded", "failed"]
    cache: dict[str, Any]
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    dataset_artifact: NeutralDatasetArtifactLayout | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    repair_run_reference: str | None = None
