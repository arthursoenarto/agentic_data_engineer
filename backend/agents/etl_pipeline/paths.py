"""Deterministic helpers for the dataset-local pipeline workspace layout."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from backend.agents.etl_pipeline.schemas import PIPELINE_ID_PATTERN


RUN_ID_PATTERN = re.compile(r"pipeline_run_\d{8}T\d{12}Z_[a-f0-9]{8}")


@dataclass(frozen=True)
class DatasetPipelinePaths:
    """Canonical generation and execution paths for one dataset workspace."""

    dataset_dir: Path

    @property
    def inventory(self) -> Path:
        return self.dataset_dir / "dataset_inventory.json"

    @property
    def contracts(self) -> Path:
        return self.dataset_dir / "contracts"

    @property
    def editable_contract(self) -> Path:
        return self.contracts / "contract.yaml"

    @property
    def pipelines(self) -> Path:
        return self.dataset_dir / "pipelines"

    @property
    def cache(self) -> Path:
        return self.dataset_dir / "data" / "cache"

    def pipeline(self, pipeline_id: str) -> Path:
        if not PIPELINE_ID_PATTERN.fullmatch(pipeline_id):
            raise ValueError(f"Invalid pipeline_id: {pipeline_id!r}")
        return self.pipelines / pipeline_id

    def run(self, pipeline_id: str, run_id: str) -> "PipelineRunPaths":
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"Invalid pipeline run_id: {run_id!r}")
        pipeline_dir = self.pipeline(pipeline_id)
        run_dir = pipeline_dir / "runs" / run_id
        return PipelineRunPaths(
            pipeline_id=pipeline_id,
            run_id=run_id,
            run_dir=run_dir,
            output_dir=run_dir / "output",
        )


@dataclass(frozen=True)
class PipelineRunPaths:
    """One immutable execution bundle owned by a generated pipeline."""

    pipeline_id: str
    run_id: str
    run_dir: Path
    output_dir: Path

    @property
    def receipt(self) -> Path:
        return self.run_dir / "pipeline_run.json"

    @property
    def contract_lock(self) -> Path:
        return self.run_dir / "contract.lock.json"

    @property
    def logs(self) -> Path:
        return self.run_dir / "logs"

    @property
    def stdout(self) -> Path:
        return self.logs / "stdout.log"

    @property
    def stderr(self) -> Path:
        return self.logs / "stderr.log"

    @property
    def repairs(self) -> Path:
        return self.run_dir / "repairs"


def pipeline_run_id(now: datetime | None = None) -> str:
    """Return a collision-resistant, chronologically sortable execution ID."""

    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"pipeline_run_{timestamp}_{secrets.token_hex(4)}"
