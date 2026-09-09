"""Schemas for dataset candidate intake and contract drafting."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class DatasetCandidateInput(BaseModel):
    """Flexible human-provided dataset candidate input."""

    name: str | None = Field(default=None, min_length=1)
    url: HttpUrl | None = None
    slug: str | None = Field(
        default=None,
        description="Optional stable folder name for this dataset under project/datasets/.",
    )
    description: str | None = None
    user_goal: str | None = Field(
        default=None,
        description="Optional hint about why the user is considering this dataset.",
    )

    @field_validator("slug")
    @classmethod
    def slug_must_be_folder_safe(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            return None
        cleaned = value.strip()
        if not cleaned.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Slug may contain only letters, numbers, hyphens, and underscores.")
        return cleaned.lower().replace("-", "_")


class DatasetCandidate(BaseModel):
    """Minimal user-provided dataset information."""

    name: str = Field(..., min_length=1)
    url: HttpUrl
    slug: str | None = Field(
        default=None,
        description="Optional stable folder name for this dataset under project/datasets/.",
    )
    description: str | None = None
    user_goal: str | None = Field(
        default=None,
        description="Optional hint about why the user is considering this dataset.",
    )

    @field_validator("slug")
    @classmethod
    def slug_must_be_folder_safe(cls, value: str | None) -> str | None:
        return DatasetCandidateInput.slug_must_be_folder_safe(value)


class SourceEvidence(BaseModel):
    """Auditable source used to draft the contract."""

    title: str | None = None
    url: HttpUrl
    note: str | None = None


class ContractSelectorSpec(BaseModel):
    """Dataset-native selector that narrows a field into an exact channel."""

    dimension: str = Field(
        ...,
        description="Provider-native selector dimension, such as pressure_level, band, depth, station_id, lead_time, or ensemble_member.",
    )
    value: str | int | float | bool
    unit: str | None = None
    label: str | None = None


class ContractFieldSpec(BaseModel):
    """Exact selected dataset field/channel, independent of provider request syntax."""

    name: str
    display_name: str | None = None
    selectors: list[ContractSelectorSpec] = Field(
        default_factory=list,
        description="Dataset-native selector values that make this field exact, such as pressure_level=500 hPa or band=B04.",
    )
    units: str | None = None
    description: str | None = None


PipelineObjectiveName = Literal[
    "consumer_samples_per_second",
    "output_bytes",
    "materialization_seconds",
    "q_engineering",
]


class PipelineObjectivePreference(BaseModel):
    """One ordered optimization objective in the pipeline specification."""

    objective: PipelineObjectiveName
    direction: Literal["maximize", "minimize"]
    priority: int = Field(ge=1)

    @model_validator(mode="after")
    def direction_matches_objective(self) -> "PipelineObjectivePreference":
        expected = {
            "consumer_samples_per_second": "maximize",
            "output_bytes": "minimize",
            "materialization_seconds": "minimize",
            "q_engineering": "maximize",
        }[self.objective]
        if self.direction != expected:
            raise ValueError(
                f"{self.objective} must use direction={expected!r}, "
                f"not {self.direction!r}"
            )
        return self


class PipelineOptimizationRequirements(BaseModel):
    """Ordered optimization intent plus measurements retained descriptively."""

    ordered_objectives: list[PipelineObjectivePreference] = Field(min_length=1)
    descriptive_measurements: list[PipelineObjectiveName] = Field(default_factory=list)

    @model_validator(mode="after")
    def objectives_are_unique_and_contiguous(self) -> "PipelineOptimizationRequirements":
        names = [item.objective for item in self.ordered_objectives]
        priorities = [item.priority for item in self.ordered_objectives]
        if len(names) != len(set(names)):
            raise ValueError("Optimization objectives must be unique")
        if sorted(priorities) != list(range(1, len(priorities) + 1)):
            raise ValueError("Optimization priorities must be unique and contiguous from 1")
        if len(self.descriptive_measurements) != len(set(self.descriptive_measurements)):
            raise ValueError("Descriptive measurements must be unique")
        overlap = sorted(set(names) & set(self.descriptive_measurements))
        if overlap:
            raise ValueError(
                "Measurements cannot be both ordered and descriptive: "
                f"{overlap}"
            )
        return self


class PipelineDownstreamUse(BaseModel):
    """Declared consumer context used by generation and evaluation planning."""

    kind: Literal["ml_training", "scientific_analysis"]
    workload_kind: Literal["full_field_tensor", "filtered_station_scan"]
    description: str | None = None


class PipelineRequirements(BaseModel):
    """Machine-readable pipeline intent beyond the selected source data."""

    downstream_use: PipelineDownstreamUse
    optimization: PipelineOptimizationRequirements


class DatasetContract(BaseModel):
    """Minimal human-editable data request drafted from a dataset candidate."""

    schema_version: str = "dataset_contract.v1"
    dataset_slug: str
    title: str
    source_url: HttpUrl
    intent: str
    summary: str
    provider: str | None = None
    dataset_family: str | None = Field(
        default=None,
        description="Dataset-native family such as reanalysis, air_quality_api, tabular_file, paper_supplement, imagery, or generic_api.",
    )
    access_methods: list[str] = Field(default_factory=list)
    credential_requirements: list[str] = Field(default_factory=list)
    fields: list[ContractFieldSpec] = Field(
        default_factory=list,
        description="Exact selected fields/channels. Each unique field plus selector combination is one field.",
    )
    scope: dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset-native minimal scope fields. Keep this small and human-editable.",
    )
    defaults_used: list[str] = Field(default_factory=list)
    human_editable_fields: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks_or_unknowns: list[str] = Field(default_factory=list)
    evidence: list[SourceEvidence] = Field(default_factory=list)
    advanced_options: dict[str, Any] = Field(default_factory=dict)
    pipeline_requirements: PipelineRequirements | None = Field(
        default=None,
        description=(
            "Optional typed downstream workload and optimization intent. "
            "Historical contracts without this field remain valid."
        ),
    )
    human_confirmed: bool = False
    recommended_next_step: str | None = None
