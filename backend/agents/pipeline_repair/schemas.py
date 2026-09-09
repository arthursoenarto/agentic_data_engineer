"""Schemas for bounded, execution-guided pipeline repair."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.llm import LLMUsageSummary


FailureOrigin = Literal["candidate", "harness", "environment", "provider", "policy"]


class PipelineRepairEdit(BaseModel):
    """One exact text replacement proposed for an existing pipeline file."""

    relative_path: str
    old_text: str
    new_text: str
    rationale: str

    @field_validator("relative_path")
    @classmethod
    def path_must_be_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Repair paths must be relative and must not contain '..'.")
        return value

    @model_validator(mode="after")
    def replacement_must_change_text(self) -> "PipelineRepairEdit":
        if not self.old_text:
            raise ValueError("old_text must not be empty.")
        if self.old_text == self.new_text:
            raise ValueError("A repair edit must change the matched text.")
        return self


class PipelineRepairProposal(BaseModel):
    """Structured minimal patch returned by the repair model."""

    diagnosis: str
    edits: list[PipelineRepairEdit] = Field(min_length=1)
    expected_outcome: str


class PipelineExecutionResult(BaseModel):
    """Secret-redacted result from one deterministic pipeline command."""

    command: list[str] = Field(min_length=1)
    execution_label: str | None = None
    execution_dir: str | None = None
    started_at: str
    completed_at: str
    duration_seconds: float = Field(ge=0)
    return_code: int | None = None
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""
    ok: bool = False
    failure_origin: FailureOrigin | None = None
    failure_code: str | None = None

    @model_validator(mode="after")
    def outcome_controls_failure_classification(self) -> "PipelineExecutionResult":
        if self.ok and (self.failure_origin is not None or self.failure_code is not None):
            raise ValueError("Successful execution cannot have a failure classification.")
        if (self.failure_origin is None) != (self.failure_code is None):
            raise ValueError("Failure origin and code must either both be set or both be absent.")
        return self


class PipelineRepairAttempt(BaseModel):
    """One LLM repair call and its deterministic verification result."""

    attempt_number: int = Field(ge=1, le=3)
    started_at: str
    completed_at: str
    status: Literal["succeeded", "execution_failed", "patch_failed", "llm_failed"]
    failure_before: PipelineExecutionResult
    context_files: list[str] = Field(default_factory=list)
    context_size_chars: int = Field(ge=0)
    context_truncated: bool = False
    proposal: PipelineRepairProposal | None = None
    repair_error: str | None = None
    execution_after: PipelineExecutionResult | None = None
    llm_usage: LLMUsageSummary | None = None


class PipelineRepairLog(BaseModel):
    """Immutable record of one bounded repair run for a generated pipeline."""

    schema_version: str = "pipeline_repair_log.v3"
    pipeline_dir: str
    command: list[str] = Field(min_length=1)
    model: str
    reasoning_effort: str | None = None
    prompt_name: str
    max_attempts: int = Field(ge=0, le=3)
    started_at: str
    completed_at: str
    final_status: Literal[
        "already_succeeded",
        "repaired",
        "exhausted",
        "non_candidate_failure",
    ]
    initial_execution: PipelineExecutionResult
    attempts: list[PipelineRepairAttempt] = Field(default_factory=list, max_length=3)
    aggregate_llm_usage: LLMUsageSummary | None = None
