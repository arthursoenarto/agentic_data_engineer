"""Internal structured outputs used by ETL pipeline generation strategies."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.agents.etl_pipeline.schemas import GeneratedFile


class FamilyPipelineCompletion(BaseModel):
    """Model-owned source payload for matched standalone family conditions."""

    files: list[GeneratedFile] = Field(min_length=1)
    tests: list[GeneratedFile] = Field(default_factory=list)

    @model_validator(mode="after")
    def coalesce_identical_duplicate_paths(self) -> "FamilyPipelineCompletion":
        """Tolerate redundant categorization without hiding conflicting source."""

        seen: dict[str, str] = {}
        files: list[GeneratedFile] = []
        tests: list[GeneratedFile] = []
        for destination, artifacts in ((files, self.files), (tests, self.tests)):
            for artifact in artifacts:
                previous = seen.get(artifact.relative_path)
                if previous is None:
                    seen[artifact.relative_path] = artifact.content
                    destination.append(artifact)
                    continue
                if previous != artifact.content:
                    raise ValueError(
                        "Family completion contains conflicting content for "
                        f"duplicate path: {artifact.relative_path}"
                    )
        self.files = files
        self.tests = tests
        return self


class PipelineDesignFile(BaseModel):
    """One file proposed during the staged design call."""

    relative_path: str
    purpose: str


class PipelineDesign(BaseModel):
    """Provider-aware implementation design produced before code generation."""

    architecture_summary: str
    contract_requirements: list[str] = Field(min_length=1)
    provider_requests: list[str] = Field(default_factory=list)
    data_flow: list[str] = Field(min_length=1)
    file_layout: list[PipelineDesignFile] = Field(min_length=1)
    runtime_dependencies: list[str] = Field(default_factory=list)
    validation_plan: list[str] = Field(min_length=1)
    reliability_controls: list[str] = Field(min_length=1)
    reference_invariants: list[str] = Field(default_factory=list)
    execution_interface: Literal["python run_pipeline.py"] = "python run_pipeline.py"


class PipelineReviewIssue(BaseModel):
    """One concrete defect found by the staged review call."""

    severity: Literal["critical", "high", "medium", "low"]
    category: str
    relative_path: str | None = None
    description: str
    required_change: str


class PipelineReview(BaseModel):
    """Review verdict over a complete generated pipeline draft."""

    verdict: Literal["accept", "revise"]
    summary: str
    strengths: list[str] = Field(default_factory=list)
    issues: list[PipelineReviewIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def serious_issues_require_revision(self) -> "PipelineReview":
        has_serious_issue = any(
            issue.severity in {"critical", "high"} for issue in self.issues
        )
        if has_serious_issue and self.verdict != "revise":
            raise ValueError("Reviews with critical or high issues must request revision.")
        return self


PROTECTED_TEMPLATE_PATHS = {
    ".gitignore",
    "README.md",
    "manifest.json",
    "pipeline_contract.json",
    "pipeline_run.json",
    "requirements.txt",
    "run_pipeline.py",
}


class TemplateFile(BaseModel):
    """One model-filled file inside the deterministic pipeline scaffold."""

    relative_path: str
    content: str

    @field_validator("relative_path")
    @classmethod
    def path_must_be_safe_and_unprotected(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Template file paths must be safe relative paths.")
        if path.as_posix() in PROTECTED_TEMPLATE_PATHS:
            raise ValueError(f"Template file is framework-owned: {value}")
        return path.as_posix()


class TemplatePipelineCompletion(BaseModel):
    """Provider-specific slots filled inside the deterministic scaffold."""

    runtime_dependencies: list[str] = Field(min_length=1)
    files: list[TemplateFile] = Field(min_length=1)
    tests: list[TemplateFile] = Field(default_factory=list)
    readme: str

    @field_validator("runtime_dependencies")
    @classmethod
    def dependencies_must_be_plain_requirements(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            requirement = value.strip()
            if (
                not requirement
                or "\n" in requirement
                or "\r" in requirement
                or requirement.startswith("-")
                or "://" in requirement
                or any(character in requirement for character in (";", "&", "|", "`", "$"))
            ):
                raise ValueError(
                    "Runtime dependencies must be plain package requirement lines."
                )
            cleaned.append(requirement)
        return cleaned

    @model_validator(mode="after")
    def completion_must_fill_required_slots(self) -> "TemplatePipelineCompletion":
        file_paths = [file.relative_path for file in self.files]
        test_paths = [file.relative_path for file in self.tests]
        all_paths = [*file_paths, *test_paths]
        if "pipeline_impl.py" not in file_paths:
            raise ValueError("Template completion must provide pipeline_impl.py.")
        if any(not path.startswith("tests/") for path in test_paths):
            raise ValueError("Template test files must live under tests/.")
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("Template completion contains duplicate file paths.")
        return self
