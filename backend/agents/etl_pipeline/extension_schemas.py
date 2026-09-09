"""Typed contracts and receipts for isolated repository-extension experiments."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.agents.etl_pipeline.schemas import (
    ArtifactTargetKind,
    PIPELINE_ID_PATTERN,
    PromptExpertise,
    PromptProvenance,
    ReferenceContextMode,
    ReferenceContextProvenance,
    ResearchRole,
    SHA256_PATTERN,
)
from backend.llm import LLMUsageSummary


GIT_COMMIT_PATTERN = re.compile(r"[a-f0-9]{40}")


class RepositoryChangeOperation(StrEnum):
    """Allowed model-proposed repository mutations."""

    ADD = "add"
    MODIFY = "modify"


class ExtensionFileChange(BaseModel):
    """One complete file addition or replacement proposed by the model."""

    relative_path: str
    operation: RepositoryChangeOperation
    content: str
    rationale: str

    @field_validator("relative_path")
    @classmethod
    def path_must_be_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Extension file paths must be safe repository-relative paths.")
        if path.as_posix().startswith(".git/") or path.as_posix() == ".git":
            raise ValueError("Extension changes must never target Git metadata.")
        return path.as_posix()


class RepositoryExtensionResult(BaseModel):
    """Canonical LLM output for a repository-extension proposal."""

    summary: str
    files: list[ExtensionFileChange] = Field(default_factory=list)
    tests: list[ExtensionFileChange] = Field(min_length=1)
    readme: str

    @model_validator(mode="after")
    def changes_must_be_unique_and_tests_scoped(self) -> "RepositoryExtensionResult":
        paths = [change.relative_path for change in [*self.files, *self.tests]]
        if len(paths) != len(set(paths)):
            raise ValueError("Repository extension contains duplicate change paths.")
        if not self.files:
            raise ValueError("Repository extension must contain at least one source change.")
        if any(not change.relative_path.startswith("tests/") for change in self.tests):
            raise ValueError("Proposed extension tests must live under tests/.")
        return self


class ExtensionCommandKind(StrEnum):
    """Purpose of one deterministic repository command."""

    TEST = "test"
    LINT = "lint"
    TYPECHECK = "typecheck"
    DOCUMENTATION = "documentation"
    WORKFLOW = "workflow"


class ExtensionCommand(BaseModel):
    """One shell-free command executed inside the isolated repository clone."""

    name: str
    kind: ExtensionCommandKind
    argv: list[str] = Field(min_length=1)
    timeout_seconds: int = Field(default=600, gt=0, le=7200)
    expected_return_codes: list[int] = Field(default_factory=lambda: [0], min_length=1)
    expected_diagnostic_count: int | None = Field(default=None, ge=0)

    @field_validator("argv")
    @classmethod
    def command_must_not_use_shell_control(cls, values: list[str]) -> list[str]:
        if any("\x00" in value for value in values):
            raise ValueError("Extension command arguments must not contain NUL bytes.")
        return values


class ExtensionContract(BaseModel):
    """Frozen task definition and deterministic acceptance gates for one extension."""

    schema_version: str = "terraio_extension_contract.v1"
    extension_id: str
    dataset_slug: str
    target_kind: Literal[
        ArtifactTargetKind.TERRAIO_REPOSITORY_EXTENSION
    ] = ArtifactTargetKind.TERRAIO_REPOSITORY_EXTENSION
    repository_url: str
    repository_path: str
    base_commit: str
    context_paths: list[str] = Field(min_length=1)
    allowed_paths: list[str] = Field(min_length=1)
    protected_paths: list[str] = Field(default_factory=lambda: [".git/**"])
    required_public_interfaces: list[str] = Field(min_length=1)
    python_source_roots: list[str] = Field(default_factory=lambda: ["."])
    architectural_invariants: list[str] = Field(min_length=1)
    dependency_policy: Literal[
        "existing_dependencies_only",
        "allow_optional_dependency_changes",
    ] = "existing_dependencies_only"
    baseline_commands: list[ExtensionCommand] = Field(min_length=1)
    verification_commands: list[ExtensionCommand] = Field(min_length=1)
    representative_workflow_commands: list[ExtensionCommand] = Field(min_length=1)
    expected_dataset_workflow: str
    expected_artifact_contract: str
    prompt_name: str = "pr_extension_v1"
    prompt_expertise: PromptExpertise = PromptExpertise.EXPERT
    reference_context_mode: Literal[
        ReferenceContextMode.FROZEN_TERRAIO_REPOSITORY
    ] = ReferenceContextMode.FROZEN_TERRAIO_REPOSITORY
    research_role: Literal[ResearchRole.CASE_STUDY] = ResearchRole.CASE_STUDY

    @field_validator("extension_id")
    @classmethod
    def extension_id_must_be_safe(cls, value: str) -> str:
        if not PIPELINE_ID_PATTERN.fullmatch(value):
            raise ValueError("extension_id must be a safe 3-128 character identifier.")
        return value

    @field_validator("base_commit")
    @classmethod
    def base_commit_must_be_full_sha(cls, value: str) -> str:
        if not GIT_COMMIT_PATTERN.fullmatch(value):
            raise ValueError("base_commit must be a full lowercase Git SHA-1.")
        return value

    @field_validator("context_paths", "allowed_paths", "protected_paths")
    @classmethod
    def repository_patterns_must_be_safe(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("Repository paths and patterns must be safe and relative.")
            cleaned.append(path.as_posix())
        return cleaned

    @field_validator("python_source_roots")
    @classmethod
    def source_roots_must_be_safe(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Python source roots must stay inside the repository.")
            cleaned.append(path.as_posix())
        return cleaned

    @field_validator("required_public_interfaces")
    @classmethod
    def interfaces_must_be_import_targets(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*",
                value,
            ):
                raise ValueError(
                    "Required interfaces must use importable 'module:attribute' notation."
                )
        return values


class RepositoryExtensionManifest(BaseModel):
    """Immutable generation provenance for one repository-extension proposal."""

    schema_version: str = "repository_extension_manifest.v1"
    extension_id: str
    dataset_slug: str
    target_kind: Literal[
        ArtifactTargetKind.TERRAIO_REPOSITORY_EXTENSION
    ] = ArtifactTargetKind.TERRAIO_REPOSITORY_EXTENSION
    model: str
    prompt_name: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    base_commit: str
    extension_contract_sha256: str
    proposal_sha256: str
    patch_sha256: str
    changed_paths: list[str]
    generated_artifacts: list[str]
    prompt_provenance: PromptProvenance
    reference_context: ReferenceContextProvenance
    llm_usage: LLMUsageSummary
    research_role: Literal[ResearchRole.CASE_STUDY] = ResearchRole.CASE_STUDY

    @field_validator(
        "extension_contract_sha256",
        "proposal_sha256",
        "patch_sha256",
    )
    @classmethod
    def hashes_must_be_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("Extension manifest hashes must be lowercase SHA-256 digests.")
        return value

    @field_validator("base_commit")
    @classmethod
    def manifest_commit_must_be_full_sha(cls, value: str) -> str:
        if not GIT_COMMIT_PATTERN.fullmatch(value):
            raise ValueError("Extension manifest base_commit must be a full Git SHA-1.")
        return value


class ExtensionCommandResult(BaseModel):
    """Structured result for one command in an isolated extension run."""

    phase: Literal["baseline", "verification", "workflow"]
    name: str
    kind: ExtensionCommandKind
    argv: list[str]
    started_at: str
    completed_at: str
    duration_seconds: float = Field(ge=0)
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
            self.return_code in self.expected_return_codes
            and not self.timed_out
            and (
                self.expected_diagnostic_count is None
                or self.diagnostic_count == self.expected_diagnostic_count
            )
        )


class RepositoryExtensionRunReceipt(BaseModel):
    """Immutable evidence from applying and checking an extension in a clone."""

    schema_version: str = "repository_extension_run.v1"
    run_id: str
    extension_id: str
    target_kind: Literal[
        ArtifactTargetKind.TERRAIO_REPOSITORY_EXTENSION
    ] = ArtifactTargetKind.TERRAIO_REPOSITORY_EXTENSION
    base_commit: str
    patch_sha256: str
    source_tree_sha256_before: str
    source_tree_sha256_after: str
    started_at: str
    completed_at: str
    duration_seconds: float = Field(ge=0)
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
    commands: list[ExtensionCommandResult] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)

    @field_validator(
        "patch_sha256",
        "source_tree_sha256_before",
        "source_tree_sha256_after",
    )
    @classmethod
    def receipt_hashes_must_be_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("Extension receipt hashes must be lowercase SHA-256 digests.")
        return value
