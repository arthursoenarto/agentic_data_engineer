"""Frozen prompt-template locks for matched pipeline-generation experiments."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.agents.etl_pipeline.conditions import pipeline_condition
from backend.agents.etl_pipeline.schemas import (
    PipelineConditionMetadata,
    PipelineConditionName,
    PipelineManifest,
    PromptSourceProvenance,
)


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PRIMARY_CONDITIONS = (
    PipelineConditionName.NAIVE_LLM,
    PipelineConditionName.EXPERT_DIRECT_LLM,
    PipelineConditionName.TERRAIO_REFERENCED,
)


class ConditionPromptLock(BaseModel):
    """Frozen condition metadata and prompt sources for one treatment."""

    condition: PipelineConditionMetadata
    sources: list[PromptSourceProvenance] = Field(min_length=2)


class PipelinePromptLock(BaseModel):
    """Exact source-template lock shared by every matched repetition."""

    schema_version: Literal["pipeline_prompt_lock.v1"] = "pipeline_prompt_lock.v1"
    lock_id: str = Field(min_length=1)
    created_at: str
    conditions: list[ConditionPromptLock] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def contains_primary_conditions_once(self) -> "PipelinePromptLock":
        observed = [item.condition.name for item in self.conditions]
        if len(set(observed)) != len(observed) or set(observed) != set(PRIMARY_CONDITIONS):
            raise ValueError("Prompt lock must contain each primary condition once")
        return self

    def for_condition(self, name: PipelineConditionName) -> ConditionPromptLock:
        return next(item for item in self.conditions if item.condition.name == name)


def create_pipeline_prompt_lock(*, lock_id: str) -> PipelinePromptLock:
    """Hash the current primary prompt templates into one immutable lock."""

    conditions = []
    for name in PRIMARY_CONDITIONS:
        metadata = pipeline_condition(name)
        sources = [
            PromptSourceProvenance(
                path=_display(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in condition_prompt_paths(name)
        ]
        conditions.append(ConditionPromptLock(condition=metadata, sources=sources))
    return PipelinePromptLock(
        lock_id=lock_id,
        created_at=datetime.now(UTC).isoformat(),
        conditions=conditions,
    )


def write_pipeline_prompt_lock(path: Path, lock: PipelinePromptLock) -> None:
    """Write a prompt lock once; experiment inputs are never overwritten."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(lock.model_dump_json(indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_and_validate_pipeline_prompt_lock(path: Path) -> PipelinePromptLock:
    """Load a lock and fail before generation if any source has drifted."""

    lock = PipelinePromptLock.model_validate_json(path.read_text(encoding="utf-8"))
    validate_pipeline_prompt_lock(lock)
    return lock


def validate_pipeline_prompt_lock(lock: PipelinePromptLock) -> None:
    """Verify condition metadata, source paths, and bytes against the repository."""

    for item in lock.conditions:
        expected_metadata = pipeline_condition(item.condition.name)
        if item.condition != expected_metadata:
            raise ValueError(
                f"Condition metadata drifted for {item.condition.name.value}"
            )
        expected_paths = condition_prompt_paths(item.condition.name)
        observed = {source.path: source.sha256 for source in item.sources}
        expected_names = {_display(path) for path in expected_paths}
        if set(observed) != expected_names:
            raise ValueError(
                f"Prompt source set drifted for {item.condition.name.value}"
            )
        for path in expected_paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed[_display(path)] != digest:
                raise ValueError(f"Frozen prompt changed: {_display(path)}")


def validate_manifest_prompt_lock(
    manifest: PipelineManifest,
    lock: PipelinePromptLock,
) -> None:
    """Prove a generated result used the locked condition templates."""

    if manifest.condition is None or manifest.prompt_provenance is None:
        raise ValueError("Matched generation requires condition and prompt provenance")
    locked = lock.for_condition(manifest.condition.name)
    if manifest.condition != locked.condition:
        raise ValueError("Generated condition metadata differs from the prompt lock")
    expected = {(source.path, source.sha256) for source in locked.sources}
    observed = {
        (source.path, source.sha256)
        for source in manifest.prompt_provenance.sources
    }
    if observed != expected:
        raise ValueError("Generated prompt provenance differs from the prompt lock")


def condition_prompt_paths(name: PipelineConditionName) -> tuple[Path, Path]:
    """Return the exact source templates used by one primary condition."""

    condition = pipeline_condition(name)
    if condition.prompt_name.startswith(("expert_family_", "expert_reference_family_")):
        version = condition.prompt_name.rsplit("_", 1)[-1]
        stem = PROMPTS_DIR / "shared" / f"expert_family_{version}"
    else:
        stem = PROMPTS_DIR / condition.orchestration_variant.value / condition.prompt_name
    paths = (Path(f"{stem}_system.md"), Path(f"{stem}_user.md"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prompt source is missing: {missing[0]}")
    return paths


def _display(path: Path) -> str:
    root = Path(__file__).resolve().parents[3]
    return path.resolve().relative_to(root).as_posix()
