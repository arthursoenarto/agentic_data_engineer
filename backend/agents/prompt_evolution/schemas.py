"""Typed artifacts for reflective prompt mutation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.llm import LLMUsageSummary


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromptMutation(_StrictModel):
    """One complete replacement prompt and its causal hypothesis."""

    reflection: str = Field(min_length=20, max_length=4_000)
    hypothesis: str = Field(min_length=20, max_length=1_500)
    expected_effect: str = Field(min_length=10, max_length=1_000)
    risk: str = Field(min_length=3, max_length=1_000)
    system_prompt: str = Field(min_length=500, max_length=20_000)

    @field_validator("system_prompt")
    @classmethod
    def prompt_preserves_generation_boundary(cls, value: str) -> str:
        concept_terms = {
            "structured output": ("structured", "json"),
            "contract": ("contract",),
            "offline fixture": ("fixture", "offline"),
            "Zarr publication": ("zarr",),
        }
        lowered = value.lower()
        missing = [
            concept
            for concept, alternatives in concept_terms.items()
            if not any(term in lowered for term in alternatives)
        ]
        if missing:
            raise ValueError(f"Evolved prompt omitted required concepts: {missing}")
        return value.strip() + "\n"


class PromptMutationCall(_StrictModel):
    """Immutable provenance for one prompt-evolution model call."""

    schema_version: Literal["gepa_prompt_mutation.v1"] = "gepa_prompt_mutation.v1"
    experiment_id: str
    iteration: int = Field(ge=1)
    candidate_id: str
    parent_candidate_id: str
    model: str
    system_prompt_sha256: str = Field(pattern=r"[a-f0-9]{64}")
    user_prompt_sha256: str = Field(pattern=r"[a-f0-9]{64}")
    parent_generation_prompt_sha256: str = Field(pattern=r"[a-f0-9]{64}")
    evaluation_feedback_sha256: str = Field(pattern=r"[a-f0-9]{64}")
    mutation: PromptMutation
    llm_usage: LLMUsageSummary
