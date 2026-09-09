"""Typed LLM output for dataset evaluation planning."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.agents.contract_drafting.schemas import PipelineOptimizationRequirements


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegularGridTargetDraft(_StrictModel):
    """Supported regular-grid evaluation target."""

    data_class: Literal["regular_rectilinear_grid"]
    output_format: Literal["zarr"]
    output_format_version: Literal[3]
    workload_kind: Literal["full_field_tensor"]
    engineering_profile: Literal["general_pipeline"]


class StationTargetDraft(_StrictModel):
    """Supported station-time-series evaluation target."""

    data_class: Literal["station_time_series"]
    output_format: Literal["parquet"]
    output_format_version: Literal[1]
    workload_kind: Literal["filtered_station_scan"]
    engineering_profile: Literal["general_pipeline"]


EvaluationTargetDraft = Annotated[
    RegularGridTargetDraft | StationTargetDraft,
    Field(discriminator="data_class"),
]


class AxisMappingDraft(_StrictModel):
    """Logical axis requirement; physical paths are bound after generation."""

    role: Literal["sample", "y", "x"]
    semantic: str = Field(min_length=3, max_length=500)
    candidate_name_hints: list[str] = Field(min_length=1, max_length=8)
    evidence: str = Field(min_length=3, max_length=1_000)

    @field_validator("candidate_name_hints")
    @classmethod
    def hints_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Axis name hints must be unique")
        return value


class FieldMappingDraft(_StrictModel):
    """Logical contract channel and non-authoritative physical-name hints."""

    field_id: str = Field(min_length=1, max_length=300)
    candidate_array_hints: list[str] = Field(default_factory=list, max_length=8)
    selector_dimensions: list[str] = Field(default_factory=list, max_length=8)
    comparison_mode: Literal["exact_native", "declared_tolerance"]
    rationale: str = Field(min_length=3, max_length=1_000)


class StationMappingDraft(_StrictModel):
    """Canonical semantics required by a station-table evaluator adapter."""

    station_key_semantic: str = Field(min_length=3, max_length=500)
    timestamp_semantic: str = Field(min_length=3, max_length=500)
    field_semantic: str = Field(min_length=3, max_length=500)
    value_semantic: str = Field(min_length=3, max_length=500)
    candidate_column_hints: dict[
        Literal["station_id", "timestamp_utc", "field_id", "value"],
        list[str],
    ]
    evidence: str = Field(min_length=3, max_length=1_000)

    @model_validator(mode="after")
    def all_canonical_columns_are_mapped(self) -> "StationMappingDraft":
        required = {"station_id", "timestamp_utc", "field_id", "value"}
        if set(self.candidate_column_hints) != required:
            raise ValueError(
                "Station mappings must contain station_id/timestamp_utc/field_id/value"
            )
        for role, hints in self.candidate_column_hints.items():
            if not hints or len(hints) != len(set(hints)):
                raise ValueError(f"Station column hints for {role} must be non-empty and unique")
        return self


class OracleStrategyDraft(_StrictModel):
    """Required independent-reference strategy, not an oracle implementation."""

    strategy: Literal[
        "independent_provider_materialization",
        "independent_trusted_reference",
    ]
    description: str = Field(min_length=10, max_length=1_500)
    independence_requirements: list[str] = Field(min_length=1, max_length=8)


class ProposedEvaluationCheckDraft(_StrictModel):
    """Untrusted check-gap proposal that cannot enter the catalog directly."""

    proposal_id: str = Field(pattern=r"^proposal\.[a-z][a-z0-9_.-]+$")
    title: str = Field(min_length=3, max_length=120)
    layer: Literal[
        "core",
        "data_class",
        "output_format",
        "workload",
        "objective",
        "engineering",
        "robustness",
    ]
    uncovered_requirement: str = Field(min_length=10, max_length=1_500)
    why_existing_checks_are_insufficient: str = Field(min_length=10, max_length=1_500)
    required_independent_evidence: list[str] = Field(min_length=1, max_length=8)
    mutation_cases: list[str] = Field(min_length=1, max_length=8)


class EvaluationPlanningDraft(_StrictModel):
    """One constrained LLM classification and mapping proposal."""

    compatibility: Literal["compatible", "uncertain", "incompatible"]
    target: EvaluationTargetDraft
    target_rationale: str = Field(min_length=10, max_length=2_000)
    domain_tags: list[str] = Field(default_factory=list, max_length=8)
    axes: list[AxisMappingDraft] = Field(default_factory=list, max_length=3)
    station_mapping: StationMappingDraft | None = None
    fields: list[FieldMappingDraft] = Field(min_length=1)
    oracle: OracleStrategyDraft
    objective_policy: PipelineOptimizationRequirements | None = None
    proposed_checks: list[ProposedEvaluationCheckDraft] = Field(
        default_factory=list,
        max_length=8,
    )

    @field_validator("domain_tags")
    @classmethod
    def domains_are_unique_nonempty_labels(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Domain tags must be unique")
        invalid = [tag for tag in value if not tag.strip()]
        if invalid:
            raise ValueError("Domain tags must be non-empty labels")
        return sorted(value)

    @model_validator(mode="after")
    def logical_mappings_are_unique(self) -> "EvaluationPlanningDraft":
        roles = [axis.role for axis in self.axes]
        if self.target.data_class == "regular_rectilinear_grid":
            if set(roles) != {"sample", "y", "x"} or len(roles) != 3:
                raise ValueError("Grid mappings must contain sample/y/x exactly once")
            if self.station_mapping is not None:
                raise ValueError("Grid plans cannot contain a station mapping")
        else:
            if roles:
                raise ValueError("Station plans cannot contain grid axis mappings")
            if self.station_mapping is None:
                raise ValueError("Station plans require canonical station mappings")
        field_ids = [field.field_id for field in self.fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("Field mappings must be unique")
        proposal_ids = [item.proposal_id for item in self.proposed_checks]
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("Proposed check IDs must be unique")
        return self
