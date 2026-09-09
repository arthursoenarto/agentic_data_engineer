"""Typed outputs for evaluation-guided pipeline improvement proposals."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.llm import LLMUsageSummary


ObjectiveName = Literal[
    "materialization_seconds",
    "consumer_samples_per_second",
    "output_bytes",
    "q_engineering",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImprovementEdit(_StrictModel):
    """One exact, reviewable replacement inside candidate-owned source."""

    relative_path: str
    old_text: str
    new_text: str
    rationale: str

    @field_validator("relative_path")
    @classmethod
    def path_is_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("Improvement paths must be confined relative paths.")
        return path.as_posix()

    @model_validator(mode="after")
    def replacement_changes_content(self) -> "ImprovementEdit":
        if not self.old_text:
            raise ValueError("old_text must be non-empty for an exact replacement.")
        if self.old_text == self.new_text:
            raise ValueError("An improvement edit must change its matched text.")
        return self


class ImprovementProposal(_StrictModel):
    """One atomic experimental hypothesis and its exact source edits."""

    title: str = Field(min_length=3, max_length=120)
    hypothesis: str = Field(min_length=10, max_length=1_500)
    target_objectives: list[ObjectiveName] = Field(min_length=1, max_length=2)
    expected_tradeoffs: str = Field(min_length=3, max_length=1_000)
    edits: list[ImprovementEdit] = Field(min_length=1, max_length=4)
    validation_plan: str = Field(min_length=3, max_length=1_000)

    @model_validator(mode="after")
    def proposal_is_focused(self) -> "ImprovementProposal":
        if len(set(self.target_objectives)) != len(self.target_objectives):
            raise ValueError("target_objectives must be unique.")
        return self


class ImprovementProposalRecord(_StrictModel):
    """Research provenance for one model-generated proposal."""

    schema_version: Literal["pipeline_improvement_proposal.v1"] = (
        "pipeline_improvement_proposal.v1"
    )
    model: str
    prompt_version: Literal["pipeline_improvement_v1"] = "pipeline_improvement_v1"
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal: ImprovementProposal
    llm_usage: LLMUsageSummary | None = None


class ImprovementProposalFailureRecord(_StrictModel):
    """Research provenance for a billed proposal call that did not validate."""

    schema_version: Literal["pipeline_improvement_proposal_failure.v1"] = (
        "pipeline_improvement_proposal_failure.v1"
    )
    model: str
    prompt_version: Literal["pipeline_improvement_v1"] = "pipeline_improvement_v1"
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    error_type: str
    error: str
    llm_usage: LLMUsageSummary | None = None
