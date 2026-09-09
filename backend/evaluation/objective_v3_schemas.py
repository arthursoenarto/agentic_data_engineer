"""Strict contracts for constrained v3 and MERODA engineering assessment."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.evaluation.check_library import (
    EvaluationProfile,
    ResolvedEvaluationCheckPlan,
    compile_evaluation_check_plan,
)
from backend.evaluation.constrained_schemas import (
    CandidateSourceSpec,
    CompactEvaluationSummary,
    ConstrainedEvaluationConfig,
    ConstrainedEvaluationRun,
    FrozenFile,
    FrozenPath,
)
from backend.evaluation.core_schemas import CommandExecutionPolicy
from backend.evaluation.schemas import CheckStatus
from backend.llm import LLMUsageSummary


EngineeringProfile = Literal["general_pipeline", "terraio_extension"]
EngineeringMode = Literal["development", "thesis"]
MerodaDimension = Literal["M", "E", "R", "O", "D", "A"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EngineeringJudgeSettingsV3(_StrictModel):
    """Frozen repeated-judge settings for one MERODA profile."""

    enabled: bool = True
    profile: EngineeringProfile = "general_pipeline"
    mode: EngineeringMode = "development"
    provider: Literal["openai"] = "openai"
    model: str | None = None
    prompt_version: Literal["engineering_quality_meroda_v1"] = (
        "engineering_quality_meroda_v1"
    )
    rubric_version: Literal["meroda_v1"] = "meroda_v1"
    repetitions: int = Field(default=1, ge=1, le=9)
    max_retries_per_repetition: int = Field(default=1, ge=0, le=3)
    max_output_tokens: int = Field(default=8192, ge=512)
    max_review_characters: int = Field(default=180_000, ge=1_000)
    max_file_characters: int = Field(default=50_000, ge=1_000)

    @model_validator(mode="after")
    def thesis_mode_requires_repeated_judgments(self) -> "EngineeringJudgeSettingsV3":
        if self.mode == "thesis" and self.repetitions < 3:
            raise ValueError(
                "Thesis mode requires at least three independent judgments"
            )
        return self


class ExtensibilityProbeSpec(_StrictModel):
    """Alternate valid suite for the same candidate and dataset inventory."""

    enabled: bool = True
    required: bool = True
    suite: FrozenFile


class ConstrainedEvaluationConfigV3(ConstrainedEvaluationConfig):
    """V3 regular-grid suite: v2 gates/objectives plus MERODA quality."""

    schema_version: Literal["evaluation_constrained.regular_grid_zarr.v3"] = (
        "evaluation_constrained.regular_grid_zarr.v3"
    )
    execution_policy: CommandExecutionPolicy = Field(
        default_factory=CommandExecutionPolicy
    )
    engineering_quality: EngineeringJudgeSettingsV3 = Field(
        default_factory=EngineeringJudgeSettingsV3
    )
    extensibility_probe: ExtensibilityProbeSpec | None = None
    evaluation_profile: EvaluationProfile | None = None
    check_plan: ResolvedEvaluationCheckPlan | None = None
    evaluation_planning: FrozenFile | None = None

    @model_validator(mode="after")
    def general_profile_is_required(self) -> "ConstrainedEvaluationConfigV3":
        if self.engineering_quality.profile != "general_pipeline":
            raise ValueError("Regular-grid candidates use the general_pipeline profile")
        expected_profile = EvaluationProfile(
            profile_id=(
                f"{self.grid.dataset_class}-{self.grid.output_format}-"
                f"v{self.output_policy.format_version}"
                if self.output_policy is not None
                else f"{self.grid.dataset_class}-{self.grid.output_format}-unspecified"
            ),
            data_class=self.grid.dataset_class,
            output_format=self.grid.output_format,
            output_format_version=(
                self.output_policy.format_version
                if self.output_policy is not None
                else None
            ),
            workload_kind=self.workload.kind,
            engineering_profile=self.engineering_quality.profile,
        )
        if self.evaluation_profile is None:
            self.evaluation_profile = expected_profile
        else:
            structural_fields = (
                "data_class",
                "output_format",
                "output_format_version",
                "workload_kind",
                "engineering_profile",
            )
            mismatches = {
                field: {
                    "profile": getattr(self.evaluation_profile, field),
                    "suite": getattr(expected_profile, field),
                }
                for field in structural_fields
                if getattr(self.evaluation_profile, field)
                != getattr(expected_profile, field)
            }
            if mismatches:
                raise ValueError(
                    "evaluation_profile conflicts with the typed suite: "
                    f"{mismatches}"
                )
        expected_plan = compile_evaluation_check_plan(
            self.evaluation_profile,
            engineering_enabled=self.engineering_quality.enabled,
            extensibility_probe_enabled=bool(
                self.extensibility_probe is not None
                and self.extensibility_probe.enabled
            ),
        )
        if self.check_plan is None:
            self.check_plan = expected_plan
        elif self.check_plan != expected_plan:
            raise ValueError(
                "check_plan must exactly match deterministic profile resolution"
            )
        return self

    def as_v2(self) -> ConstrainedEvaluationConfig:
        """Return the exact v2 deterministic contract without v3-only fields."""

        payload = self.model_dump(
            mode="json",
            exclude={
                "execution_policy",
                "engineering_quality",
                "extensibility_probe",
                "evaluation_profile",
                "check_plan",
                "evaluation_planning",
            },
        )
        payload["schema_version"] = "evaluation_constrained.regular_grid_zarr.v2"
        return ConstrainedEvaluationConfig.model_validate(payload)


class ReviewBundleItem(_StrictModel):
    """One candidate/reviewer artifact and its bounded-review coverage."""

    alias: str
    source_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    category: Literal[
        "source",
        "test",
        "documentation",
        "configuration",
        "deterministic_evidence",
        "reference",
    ]
    total_characters: int = Field(ge=0)
    included_characters: int = Field(ge=0)
    total_lines: int = Field(ge=0)
    included_lines: int = Field(ge=0)
    truncated: bool


class ReviewBundleCoverage(_StrictModel):
    """Inventory and truncation evidence for a blinded review bundle."""

    discovered_files: int = Field(ge=0)
    included_files: int = Field(ge=0)
    excluded_files: int = Field(ge=0)
    included_characters: int = Field(ge=0)
    maximum_characters: int = Field(ge=1)
    truncated_files: int = Field(ge=0)
    items: list[ReviewBundleItem] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)


class MerodaCitation(_StrictModel):
    """Line-bounded evidence from the exact blinded review bundle."""

    path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def line_range_is_ordered(self) -> "MerodaCitation":
        if self.line_end < self.line_start:
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


class MerodaComponentJudgment(_StrictModel):
    """One anchored component in an individual judge response."""

    dimension: MerodaDimension
    name: str
    applicability: Literal["applicable", "not_applicable"]
    score: int | None = Field(default=None, ge=0, le=4)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    evidence: list[MerodaCitation] = Field(default_factory=list)
    improvement: str = Field(min_length=1)
    feedback_code: str = Field(min_length=1)

    @model_validator(mode="after")
    def applicability_controls_score_and_evidence(self) -> "MerodaComponentJudgment":
        if self.applicability == "not_applicable":
            if self.score is not None or self.evidence:
                raise ValueError(
                    "A not-applicable component has null score and no citations"
                )
        elif self.score is None or not self.evidence:
            raise ValueError("An applicable component requires a score and citation")
        return self


class MerodaJudgment(_StrictModel):
    """One complete M/E/R/O/D/A judgment."""

    profile: EngineeringProfile
    components: list[MerodaComponentJudgment] = Field(min_length=6, max_length=6)
    overall_rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def dimensions_and_profile_are_consistent(self) -> "MerodaJudgment":
        expected = {"M", "E", "R", "O", "D", "A"}
        observed = [item.dimension for item in self.components]
        if set(observed) != expected or len(observed) != len(expected):
            raise ValueError("A judgment must contain M/E/R/O/D/A exactly once")
        architecture = next(item for item in self.components if item.dimension == "A")
        expected_applicability = (
            "not_applicable" if self.profile == "general_pipeline" else "applicable"
        )
        if architecture.applicability != expected_applicability:
            raise ValueError("Architecture applicability does not match the profile")
        if any(
            item.applicability != "applicable"
            for item in self.components
            if item.dimension != "A"
        ):
            raise ValueError("M/E/R/O/D are applicable in both profiles")
        return self


class JudgeCallRecordV3(_StrictModel):
    """One provider call, including retries, latency, usage, and validation."""

    repetition: int = Field(ge=1)
    attempt: int = Field(ge=1)
    provider: str
    model: str
    prompt_version: str
    rubric_version: str
    settings: dict[str, Any]
    duration_seconds: float = Field(ge=0.0)
    succeeded: bool
    usage: LLMUsageSummary | None = None
    judgment: MerodaJudgment | None = None
    error: str | None = None

    @model_validator(mode="after")
    def outcome_fields_are_consistent(self) -> "JudgeCallRecordV3":
        if self.succeeded and (self.judgment is None or self.error is not None):
            raise ValueError("A successful judge call requires judgment and no error")
        if not self.succeeded and self.error is None:
            raise ValueError("A failed judge call requires an error")
        return self


class MerodaAggregateComponent(_StrictModel):
    """Median and dispersion across independent valid judgments."""

    dimension: MerodaDimension
    name: str
    applicability: Literal["applicable", "not_applicable"]
    median_score: float | None = Field(default=None, ge=0.0, le=4.0)
    scores: list[int] = Field(default_factory=list)
    score_min: int | None = Field(default=None, ge=0, le=4)
    score_max: int | None = Field(default=None, ge=0, le=4)
    median_confidence: float = Field(ge=0.0, le=1.0)
    feedback_codes: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def aggregation_is_consistent(self) -> "MerodaAggregateComponent":
        if self.applicability == "not_applicable":
            if self.median_score is not None or self.scores:
                raise ValueError("A not-applicable aggregate has no scores")
            if self.score_min is not None or self.score_max is not None:
                raise ValueError("A not-applicable aggregate has no range")
            return self
        if self.scores:
            ordered = sorted(self.scores)
            middle = len(ordered) // 2
            median = (
                float(ordered[middle])
                if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2
            )
            if self.median_score != median:
                raise ValueError("median_score must equal the score median")
            if self.score_min != min(self.scores) or self.score_max != max(self.scores):
                raise ValueError("score range must match component scores")
        elif any(
            value is not None
            for value in (self.median_score, self.score_min, self.score_max)
        ):
            raise ValueError("An empty score list cannot expose aggregate values")
        return self


class EngineeringQualityRunV3(_StrictModel):
    """Detailed repeated MERODA result with review and cost provenance."""

    schema_version: Literal["engineering_quality_meroda.v2"] = (
        "engineering_quality_meroda.v2"
    )
    status: CheckStatus
    profile: EngineeringProfile
    mode: EngineeringMode
    engineering_assessed: bool
    prompt_version: str
    rubric_version: str
    provider: str
    model: str
    repetitions_requested: int = Field(ge=1)
    valid_judgments: int = Field(ge=0)
    review_coverage: ReviewBundleCoverage
    prompt_sha256: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    calls: list[JudgeCallRecordV3] = Field(default_factory=list)
    components: list[MerodaAggregateComponent] = Field(default_factory=list)
    q_engineering: float | None = Field(default=None, ge=0.0, le=4.0)
    usage: LLMUsageSummary | None = None
    duration_seconds: float = Field(ge=0.0)
    error: str | None = None
    advisory: bool = True

    @model_validator(mode="after")
    def score_requires_complete_assessment(self) -> "EngineeringQualityRunV3":
        if self.valid_judgments > self.repetitions_requested:
            raise ValueError("valid_judgments cannot exceed repetitions_requested")
        if self.components:
            dimensions = [item.dimension for item in self.components]
            if (
                set(dimensions) != {"M", "E", "R", "O", "D", "A"}
                or len(dimensions) != 6
            ):
                raise ValueError("Aggregate components must contain M/E/R/O/D/A once")
            architecture = next(
                item for item in self.components if item.dimension == "A"
            )
            expected_architecture = (
                "not_applicable" if self.profile == "general_pipeline" else "applicable"
            )
            if architecture.applicability != expected_architecture:
                raise ValueError(
                    "Aggregate architecture applicability is profile-bound"
                )
        applicable = [
            item for item in self.components if item.applicability == "applicable"
        ]
        expected_count = 5 if self.profile == "general_pipeline" else 6
        complete = len(applicable) == expected_count and all(
            item.median_score is not None for item in applicable
        )
        if self.engineering_assessed != (complete and self.q_engineering is not None):
            raise ValueError(
                "engineering_assessed must match complete component aggregation"
            )
        if not self.engineering_assessed and self.q_engineering is not None:
            raise ValueError(
                "Incomplete engineering assessment cannot expose q_engineering"
            )
        if self.engineering_assessed:
            if self.valid_judgments != self.repetitions_requested:
                raise ValueError("A complete assessment requires every repetition")
            expected_q = sum(
                item.median_score
                for item in applicable
                if item.median_score is not None
            ) / len(applicable)
            if abs(self.q_engineering - expected_q) > 1e-12:  # type: ignore[operator]
                raise ValueError("q_engineering must be the equal applicable mean")
            if self.status != CheckStatus.PASS:
                raise ValueError("A complete assessment must have PASS status")
        elif self.status == CheckStatus.PASS:
            raise ValueError("An incomplete assessment cannot have PASS status")
        return self


class ExtensibilityProbeResult(_StrictModel):
    """Behavioral alternate-contract probe used by the E component."""

    status: CheckStatus
    required: bool
    suite_path: str | None = None
    same_candidate_source: bool | None = None
    same_inventory: bool | None = None
    source_unchanged: bool | None = None
    local_cache_and_oracle_valid: bool | None = None
    fresh_output_succeeded: bool | None = None
    deterministic_run_path: str | None = None
    feedback_code: str
    error: str | None = None

    @property
    def score_cap(self) -> int | None:
        """A failed required behavioral probe caps extensibility at weak (1/4)."""

        return (
            1
            if self.required
            and self.status != CheckStatus.PASS
            and self.feedback_code != "EXTENSIBILITY_BLOCKED_BY_CONSTRAINTS"
            else None
        )


class ObjectiveMeasurementV3(_StrictModel):
    """One optimizer-facing value with its direction and physical unit."""

    value: int | float
    direction: Literal["minimize", "maximize"]
    unit: Literal["seconds", "samples_per_second", "bytes", "score_0_to_4"]


class OperationalObjectiveGroupV3(_StrictModel):
    """Primary operational objective group."""

    priority: Literal["primary"] = "primary"
    objectives: dict[
        Literal[
            "materialization_seconds",
            "consumer_samples_per_second",
            "output_bytes",
        ],
        ObjectiveMeasurementV3,
    ] = Field(default_factory=dict)


class EngineeringObjectiveGroupV3(_StrictModel):
    """Secondary engineering-quality objective group."""

    priority: Literal["secondary"] = "secondary"
    profile: EngineeringProfile
    objectives: dict[Literal["q_engineering"], ObjectiveMeasurementV3] = Field(
        default_factory=dict
    )


class ObjectiveGroupsV3(_StrictModel):
    """Explicit priority grouping for the constrained objective vector."""

    operational: OperationalObjectiveGroupV3
    engineering: EngineeringObjectiveGroupV3


class V3EvaluationSummary(_StrictModel):
    """Compact meta-agent signal with explicit operational and evidence states."""

    target: str
    constraints: dict[str, CheckStatus]
    feasible: bool
    engineering_assessed: bool
    objective_vector_complete: bool
    optimization_ready: bool
    thesis_evidence_ready: bool
    diagnostic_operational_metrics: dict[str, int | float] = Field(
        default_factory=dict
    )
    diagnostic_engineering_quality: dict[Literal["q_engineering"], float] = Field(
        default_factory=dict
    )
    operational_objectives: dict[str, int | float] = Field(default_factory=dict)
    engineering_objective: dict[Literal["q_engineering"], float] = Field(
        default_factory=dict
    )
    objective_directions: dict[str, Literal["minimize", "maximize"]] = Field(
        default_factory=lambda: {
            "materialization_seconds": "minimize",
            "consumer_samples_per_second": "maximize",
            "output_bytes": "minimize",
            "q_engineering": "maximize",
        }
    )
    engineering_profile: EngineeringProfile
    objective_groups: ObjectiveGroupsV3 | None = None
    feedback_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def state_and_objective_rules_hold(self) -> "V3EvaluationSummary":
        expected_constraints = {
            "contract_correctness",
            "semantic_equivalence",
            "rerun_safety",
            "provenance_security",
        }
        if set(self.constraints) != expected_constraints:
            raise ValueError("Exactly the four frozen constraints are required")
        if self.feasible != all(
            status == CheckStatus.PASS for status in self.constraints.values()
        ):
            raise ValueError("feasible must equal the conjunction of hard constraints")
        expected_directions = {
            "materialization_seconds",
            "consumer_samples_per_second",
            "output_bytes",
            "q_engineering",
        }
        if set(self.objective_directions) != expected_directions:
            raise ValueError("Exactly four v3 objective directions are required")
        operational = {
            "materialization_seconds",
            "consumer_samples_per_second",
            "output_bytes",
        }
        if self.feasible and set(self.operational_objectives) != operational:
            raise ValueError(
                "A feasible candidate exposes exactly three operational objectives"
            )
        if not self.feasible and (
            self.operational_objectives or self.engineering_objective
        ):
            raise ValueError("An infeasible candidate exposes no objectives")
        if not set(self.diagnostic_operational_metrics).issubset(operational):
            raise ValueError("Diagnostic operational metrics contain unknown values")
        if self.feasible and (
            self.diagnostic_operational_metrics != self.operational_objectives
        ):
            raise ValueError(
                "Feasible diagnostic metrics must match official objectives"
            )
        if self.engineering_assessed != (
            set(self.diagnostic_engineering_quality) == {"q_engineering"}
        ):
            raise ValueError(
                "engineering_assessed must match diagnostic q_engineering availability"
            )
        expected_engineering_objective = (
            self.diagnostic_engineering_quality if self.feasible else {}
        )
        if self.engineering_objective != expected_engineering_objective:
            raise ValueError(
                "Official q_engineering requires both feasibility and assessment"
            )
        complete = self.feasible and self.engineering_assessed
        if self.objective_vector_complete != complete:
            raise ValueError("objective_vector_complete must match complete objectives")
        if self.optimization_ready != self.objective_vector_complete:
            raise ValueError(
                "optimization_ready must match objective-vector completeness"
            )
        if self.thesis_evidence_ready and not self.optimization_ready:
            raise ValueError("thesis_evidence_ready requires optimization readiness")
        expected_groups = _objective_groups(
            operational=self.operational_objectives,
            engineering=self.engineering_objective,
            profile=self.engineering_profile,
        )
        if self.objective_groups is None:
            self.objective_groups = expected_groups
        elif self.objective_groups != expected_groups:
            raise ValueError(
                "objective_groups must exactly represent the flat objective fields"
            )
        return self


def _objective_groups(
    *,
    operational: dict[str, int | float],
    engineering: dict[str, float],
    profile: EngineeringProfile,
) -> ObjectiveGroupsV3:
    units = {
        "materialization_seconds": "seconds",
        "consumer_samples_per_second": "samples_per_second",
        "output_bytes": "bytes",
    }
    directions = {
        "materialization_seconds": "minimize",
        "consumer_samples_per_second": "maximize",
        "output_bytes": "minimize",
    }
    return ObjectiveGroupsV3(
        operational=OperationalObjectiveGroupV3(
            objectives={
                name: ObjectiveMeasurementV3(
                    value=value,
                    direction=directions[name],
                    unit=units[name],
                )
                for name, value in operational.items()
            }
        ),
        engineering=EngineeringObjectiveGroupV3(
            profile=profile,
            objectives={
                "q_engineering": ObjectiveMeasurementV3(
                    value=engineering["q_engineering"],
                    direction="maximize",
                    unit="score_0_to_4",
                )
            }
            if "q_engineering" in engineering
            else {},
        ),
    )


class ConstrainedEvaluationRunV3(_StrictModel):
    """Immutable v3 report preserving deterministic and qualitative evidence."""

    schema_version: Literal[
        "evaluation_constrained_run.v3.1",
        "evaluation_constrained_run.v3.2",
    ] = (
        "evaluation_constrained_run.v3.2"
    )
    suite_id: str
    suite_version: str
    run_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    config_file: str
    check_plan: ResolvedEvaluationCheckPlan | None = None
    deterministic: ConstrainedEvaluationRun
    extensibility_probe: ExtensibilityProbeResult
    engineering_quality: EngineeringQualityRunV3
    summary: V3EvaluationSummary
    duration_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def nested_reports_match_compact_summary(self) -> "ConstrainedEvaluationRunV3":
        if (
            self.schema_version == "evaluation_constrained_run.v3.2"
            and self.check_plan is None
        ):
            raise ValueError("V3.2 evaluation reports require a frozen check plan")
        if (
            self.check_plan is not None
            and self.check_plan.profile.engineering_profile
            != self.summary.engineering_profile
        ):
            raise ValueError(
                "Check-plan engineering profile must match the compact summary"
            )
        deterministic = self.deterministic.summary
        if self.summary.target != deterministic.target:
            raise ValueError("V3 target must match deterministic target")
        if self.summary.feasible != deterministic.feasible:
            raise ValueError("V3 feasibility must match deterministic feasibility")
        if self.summary.constraints != deterministic.constraints:
            raise ValueError("V3 constraints must match deterministic constraints")
        if self.summary.operational_objectives != deterministic.objectives:
            raise ValueError(
                "V3 operational objectives must match deterministic objectives"
            )
        if (
            self.summary.diagnostic_operational_metrics
            != deterministic.diagnostic_metrics
        ):
            raise ValueError(
                "V3 diagnostic metrics must match deterministic observations"
            )
        engineering = self.engineering_quality
        if self.summary.engineering_assessed != engineering.engineering_assessed:
            raise ValueError("V3 assessment state must match MERODA")
        expected_diagnostic_quality = (
            {"q_engineering": engineering.q_engineering}
            if engineering.q_engineering is not None
            else {}
        )
        if self.summary.diagnostic_engineering_quality != expected_diagnostic_quality:
            raise ValueError("V3 diagnostic engineering quality must match MERODA")
        expected_quality = (
            expected_diagnostic_quality if self.summary.feasible else {}
        )
        if self.summary.engineering_objective != expected_quality:
            raise ValueError("V3 engineering objective must match MERODA")
        if self.summary.engineering_profile != engineering.profile:
            raise ValueError("V3 engineering profile must match MERODA")
        fully_isolated = bool(self.deterministic.scenarios) and all(
            scenario.execution.isolation is not None
            and scenario.execution.isolation.full
            for scenario in self.deterministic.scenarios
        )
        probe_ready = (
            not self.extensibility_probe.required
            or self.extensibility_probe.status == CheckStatus.PASS
        )
        expected_thesis_ready = bool(
            self.summary.optimization_ready
            and engineering.mode == "thesis"
            and engineering.valid_judgments >= 3
            and engineering.valid_judgments == engineering.repetitions_requested
            and not engineering.advisory
            and fully_isolated
            and probe_ready
        )
        if self.summary.thesis_evidence_ready != expected_thesis_ready:
            raise ValueError(
                "thesis_evidence_ready must match the frozen evidence requirements"
            )
        return self


class EvaluationFailureV3(_StrictModel):
    """Atomically persisted typed terminal report for every failed invocation."""

    schema_version: Literal["evaluation_failure.v3"] = "evaluation_failure.v3"
    run_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    phase: Literal[
        "setup",
        "planning",
        "deterministic",
        "probe",
        "judge",
        "validation",
        "unsupported",
    ]
    status: Literal["error", "timeout", "unsupported"]
    config_file: str
    error_type: str
    error: str
    partial_report: dict[str, Any] | None = None
    duration_seconds: float = Field(ge=0.0)


# Kept as an explicit alias for consumers migrating from v2 compact summaries.
DeterministicSummaryV2 = CompactEvaluationSummary


class NeutralExtensionCommand(_StrictModel):
    """Evaluator-owned view of an extension acceptance command."""

    name: str
    kind: Literal["test", "lint", "typecheck", "documentation", "workflow"]
    argv: list[str] = Field(min_length=1)
    timeout_seconds: int = Field(default=600, gt=0)
    expected_return_codes: list[int] = Field(default_factory=lambda: [0], min_length=1)
    expected_diagnostic_count: int | None = Field(default=None, ge=0)


class NeutralExtensionContract(_StrictModel):
    """Strict neutral adapter for the frozen architecture/acceptance contract."""

    schema_version: Literal["terraio_extension_contract.v1"] = (
        "terraio_extension_contract.v1"
    )
    extension_id: str
    dataset_slug: str
    target_kind: Literal["terraio_repository_extension"]
    repository_url: str
    repository_path: str
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    context_paths: list[str] = Field(min_length=1)
    allowed_paths: list[str] = Field(min_length=1)
    protected_paths: list[str] = Field(default_factory=list)
    required_public_interfaces: list[str] = Field(min_length=1)
    python_source_roots: list[str] = Field(default_factory=lambda: ["."])
    architectural_invariants: list[str] = Field(min_length=1)
    dependency_policy: Literal[
        "existing_dependencies_only", "allow_optional_dependency_changes"
    ]
    baseline_commands: list[NeutralExtensionCommand] = Field(min_length=1)
    verification_commands: list[NeutralExtensionCommand] = Field(min_length=1)
    representative_workflow_commands: list[NeutralExtensionCommand] = Field(
        min_length=1
    )
    expected_dataset_workflow: str
    expected_artifact_contract: str
    prompt_name: str
    prompt_expertise: str
    reference_context_mode: str
    research_role: str


class NeutralExtensionCommandResult(_StrictModel):
    """Strict neutral adapter for one observed extension command."""

    phase: Literal["baseline", "verification", "workflow"]
    name: str
    kind: Literal["test", "lint", "typecheck", "documentation", "workflow"]
    argv: list[str]
    started_at: str
    completed_at: str
    duration_seconds: float = Field(ge=0.0)
    return_code: int
    timed_out: bool = False
    stdout_log: str
    stderr_log: str
    tests_collected: int | None = Field(default=None, ge=0)
    tests_passed: int | None = Field(default=None, ge=0)
    tests_failed: int | None = Field(default=None, ge=0)
    expected_return_codes: list[int] = Field(default_factory=lambda: [0])
    diagnostic_count: int | None = Field(default=None, ge=0)
    expected_diagnostic_count: int | None = Field(default=None, ge=0)

    @property
    def succeeded(self) -> bool:
        return (
            not self.timed_out
            and self.return_code in self.expected_return_codes
            and (
                self.expected_diagnostic_count is None
                or self.diagnostic_count == self.expected_diagnostic_count
            )
        )


class NeutralExtensionReceipt(_StrictModel):
    """Evaluator-owned view of an extension execution receipt."""

    schema_version: Literal["repository_extension_run.v1"] = (
        "repository_extension_run.v1"
    )
    run_id: str
    extension_id: str
    target_kind: Literal["terraio_repository_extension"]
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_tree_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_tree_sha256_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: str
    completed_at: str
    duration_seconds: float = Field(ge=0.0)
    final_status: Literal[
        "succeeded",
        "baseline_failed",
        "patch_failed",
        "verification_failed",
        "workflow_failed",
        "source_modified",
    ]
    patch_applied: bool
    changed_paths: list[str]
    environment: dict[str, str]
    commands: list[NeutralExtensionCommandResult] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class TerraioExtensionEvaluationConfigV3(_StrictModel):
    """Separate v3 suite for frozen Terraio repository extensions."""

    schema_version: Literal["evaluation_constrained.terraio_extension.v3"] = (
        "evaluation_constrained.terraio_extension.v3"
    )
    suite_id: str
    suite_version: str
    candidate_id: str
    description: str | None = None
    candidate_source: CandidateSourceSpec
    extension_contract: FrozenFile
    extension_receipt: FrozenFile
    patched_checkout: FrozenPath
    reference_architecture: FrozenPath
    execution_policy: CommandExecutionPolicy = Field(
        default_factory=CommandExecutionPolicy
    )
    engineering_quality: EngineeringJudgeSettingsV3

    @model_validator(mode="after")
    def extension_profile_is_required(self) -> "TerraioExtensionEvaluationConfigV3":
        if self.engineering_quality.profile != "terraio_extension":
            raise ValueError("Terraio extensions require the terraio_extension profile")
        return self


class ExtensionDeterministicCheck(_StrictModel):
    """One evaluator-owned frozen-extension gate."""

    check_id: str
    status: CheckStatus
    summary: str
    feedback_code: str
    observed: dict[str, Any] = Field(default_factory=dict)


class TerraioExtensionSummaryV3(_StrictModel):
    """Profile-specific signal; operational pipeline objectives are N/A."""

    target: str
    profile: Literal["terraio_extension"] = "terraio_extension"
    feasible: bool
    engineering_assessed: bool
    objective_vector_complete: bool
    optimization_ready: bool
    thesis_evidence_ready: bool
    operational_objectives_applicable: Literal[False] = False
    q_engineering: float | None = Field(default=None, ge=0.0, le=4.0)
    objective_direction: Literal["maximize"] = "maximize"
    feedback_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def extension_state_is_consistent(self) -> "TerraioExtensionSummaryV3":
        if self.engineering_assessed != (self.q_engineering is not None):
            raise ValueError("engineering_assessed must match q_engineering")
        if self.objective_vector_complete != (
            self.feasible and self.engineering_assessed
        ):
            raise ValueError(
                "Extension vector completeness requires gates and q_engineering"
            )
        if self.optimization_ready != self.objective_vector_complete:
            raise ValueError(
                "optimization_ready must match extension-vector completeness"
            )
        if self.thesis_evidence_ready and not self.optimization_ready:
            raise ValueError("thesis_evidence_ready requires optimization readiness")
        return self


class TerraioExtensionEvaluationRunV3(_StrictModel):
    """Detailed extension report, intentionally incomparable to pipeline objectives."""

    schema_version: Literal["evaluation_terraio_extension_run.v3.1"] = (
        "evaluation_terraio_extension_run.v3.1"
    )
    suite_id: str
    suite_version: str
    run_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    config_file: str
    deterministic_checks: list[ExtensionDeterministicCheck]
    engineering_quality: EngineeringQualityRunV3
    summary: TerraioExtensionSummaryV3
    duration_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def readiness_matches_evidence(self) -> "TerraioExtensionEvaluationRunV3":
        engineering = self.engineering_quality
        expected_thesis_ready = bool(
            self.summary.optimization_ready
            and engineering.mode == "thesis"
            and engineering.valid_judgments >= 3
            and engineering.valid_judgments == engineering.repetitions_requested
            and not engineering.advisory
        )
        if self.summary.thesis_evidence_ready != expected_thesis_ready:
            raise ValueError(
                "thesis_evidence_ready must match extension evidence requirements"
            )
        return self
