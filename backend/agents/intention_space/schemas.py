"""Typed outputs and provenance for user-intention specification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.agents.contract_drafting.schemas import DatasetContract
from backend.llm import LLMUsageSummary


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationTurn(_StrictModel):
    """One immutable turn in a replayable user-system interaction."""

    role: Literal["user", "system"]
    content: str = Field(min_length=1, max_length=20_000)


class CredentialReference(_StrictModel):
    """Secret-free reference to an externally supplied credential."""

    name: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,127}$")
    purpose: str = Field(min_length=3, max_length=500)
    required: bool = True


class IntentionSpaceDraft(_StrictModel):
    """Single authoritative structured proposal for all intention artifacts."""

    goal: str = Field(min_length=10, max_length=2_000)
    selected_data_summary: str = Field(min_length=10, max_length=2_000)
    decisions: list[str] = Field(default_factory=list, max_length=20)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    clarification_questions: list[str] = Field(default_factory=list, max_length=8)
    credential_references: list[CredentialReference] = Field(default_factory=list)
    contract: DatasetContract

    @field_validator("decisions", "assumptions", "clarification_questions")
    @classmethod
    def list_items_are_nonempty_unique(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("Intention-space list items must be non-empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Intention-space list items must be unique")
        return normalized

    @model_validator(mode="after")
    def contract_is_current_and_machine_readable(self) -> "IntentionSpaceDraft":
        if self.contract.schema_version != "dataset_contract.v2":
            raise ValueError("Intention-space contracts must use dataset_contract.v2")
        if self.contract.pipeline_requirements is None:
            raise ValueError("Intention-space contracts require typed pipeline requirements")
        names = [item.name for item in self.credential_references]
        if len(names) != len(set(names)):
            raise ValueError("Credential references must be unique")
        return self


class IntentionInputFile(_StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class IntentionAgentCallRecord(_StrictModel):
    """Exact prompt, output, and usage for one initial or clarification call."""

    round_index: int = Field(ge=0)
    system_prompt: str
    user_prompt: str
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    latency_seconds: float = Field(ge=0)
    usage: LLMUsageSummary | None = None
    draft: IntentionSpaceDraft


class IntentionSpaceRun(_StrictModel):
    """Immutable provenance linking a conversation to materialized specification."""

    schema_version: Literal["intention_space_run.v1"] = "intention_space_run.v1"
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,119}$")
    created_at: str
    dataset_slug: str
    model: str
    prompt_version: str
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory: IntentionInputFile
    interaction: IntentionInputFile
    intention: IntentionInputFile
    editable_contract: IntentionInputFile
    contract_lock: IntentionInputFile
    credentials: IntentionInputFile
    draft: IntentionSpaceDraft
    agent_calls: list[IntentionAgentCallRecord] = Field(default_factory=list)
    clarification_rounds: int = Field(ge=0)
    clarification_turns: int = Field(ge=0)
    llm_latency_seconds: float = Field(ge=0)
    llm_usage: LLMUsageSummary | None = None
