"""Schemas for ETL pipeline generation experiments."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, field_validator

from backend.agents.contract_drafting.schemas import DatasetContract


class PipelineVariant(StrEnum):
    """Swappable ETL pipeline generation strategies for product and research use."""

    DIRECT_LLM = "direct_llm"
    STAGED_LLM = "staged_llm"
    TERRAIO_DIRECT = "terraio_direct"
    TERRAIO_STAGED = "terraio_staged"
    TEMPLATE_HYBRID = "template_hybrid"


class GeneratedFile(BaseModel):
    """A file proposed by a pipeline generation strategy."""

    relative_path: str = Field(
        ...,
        description="Path relative to the dataset directory. Must stay under pipeline/ or a pipeline_* experiment directory.",
    )
    content: str

    @field_validator("relative_path")
    @classmethod
    def path_must_stay_under_pipeline(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Generated file paths must be relative and must not contain '..'.")
        if not path.parts or not is_pipeline_output_dir(path.parts[0]):
            raise ValueError("Generated pipeline files must be written under pipeline/ or pipeline_*/.")
        if path.name == "":
            raise ValueError("Generated file path must include a filename.")
        return value


class PipelineManifest(BaseModel):
    """Metadata for a generated ETL pipeline run."""

    schema_version: str = "etl_pipeline_manifest.v1"
    dataset_slug: str = ""
    variant: PipelineVariant = PipelineVariant.DIRECT_LLM
    prompt_name: str = ""
    model: str = ""
    contract_hash: str = ""
    output_dir: str = "pipeline"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    generated_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PipelineGenerationResult(BaseModel):
    """Structured output from a pipeline generation strategy."""

    manifest: PipelineManifest
    files: list[GeneratedFile] = Field(default_factory=list)
    tests: list[GeneratedFile] = Field(default_factory=list)


def contract_hash(contract: DatasetContract) -> str:
    """Return a stable hash for the contract content used to generate a pipeline."""

    payload = json.dumps(contract.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_pipeline_output_dir(value: str) -> bool:
    """Return whether a directory name is allowed for generated pipeline artifacts."""

    return value == "pipeline" or re.fullmatch(r"pipeline_[a-z0-9_]+_\d{8}T\d{6}Z", value) is not None


def pipeline_experiment_slug(variant: PipelineVariant, generated_at: datetime | None = None) -> str:
    """Create a stable experiment output directory name for a pipeline variant."""

    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"pipeline_{variant.value}_{timestamp}"
