"""Matched-condition success accounting for pipeline-generation experiments."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.agents.etl_pipeline.prompt_lock import PRIMARY_CONDITIONS
from backend.agents.etl_pipeline.schemas import PipelineConditionName


class PipelineConditionTrialResult(BaseModel):
    """Authoritative lifecycle result for one condition and repetition."""

    schema_version: Literal["pipeline_condition_trial.v1"] = (
        "pipeline_condition_trial.v1"
    )
    experiment_id: str = Field(min_length=1)
    condition: PipelineConditionName
    repetition: int = Field(ge=1)
    prompt_lock_id: str = Field(min_length=1)
    generation_succeeded: bool
    pipeline_id: str | None = None
    prompt_sources_verified: bool = False
    initial_acceptance_passed: bool = False
    repairs_attempted: int = Field(default=0, ge=0, le=3)
    repair_final_status: Literal[
        "already_succeeded", "repaired", "exhausted"
    ] | None = None
    final_acceptance_passed: bool = False
    manifest_path: str | None = None
    initial_acceptance_report: str | None = None
    repair_log_paths: list[str] = Field(default_factory=list)
    final_acceptance_report: str | None = None
    generation_error: str | None = None
    success_at_1: bool | None = None
    success_after_le_3_repairs: bool | None = None

    @model_validator(mode="after")
    def lifecycle_is_consistent(self) -> "PipelineConditionTrialResult":
        if self.generation_succeeded:
            if not self.pipeline_id or not self.manifest_path:
                raise ValueError(
                    "Successful generation requires pipeline_id and manifest_path"
                )
            if not self.prompt_sources_verified:
                raise ValueError(
                    "Successful matched generation requires verified prompt sources"
                )
            if self.generation_error:
                raise ValueError("Successful generation cannot include generation_error")
        elif any(
            [
                self.pipeline_id,
                self.manifest_path,
                self.initial_acceptance_passed,
                self.final_acceptance_passed,
                self.repairs_attempted,
            ]
        ):
            raise ValueError("Generation failure cannot expose later lifecycle success")
        elif not self.generation_error:
            raise ValueError("Generation failure requires generation_error")

        if self.initial_acceptance_passed:
            if self.repairs_attempted != 0 or not self.final_acceptance_passed:
                raise ValueError(
                    "Initial acceptance success is final and uses no repairs"
                )
            if not self.initial_acceptance_report:
                raise ValueError("Initial acceptance success requires its report")
        if self.repairs_attempted:
            if self.initial_acceptance_passed or not self.repair_log_paths:
                raise ValueError(
                    "Repair attempts require an initial failure and repair logs"
                )
            if self.repair_final_status is None:
                raise ValueError("Repair attempts require repair_final_status")
        elif self.repair_final_status not in {None, "already_succeeded"}:
            raise ValueError("A repair status requires at least one repair attempt")
        elif self.repair_log_paths:
            raise ValueError("Repair logs require at least one repair attempt")
        if self.final_acceptance_passed and not (
            self.initial_acceptance_passed
            or self.repair_final_status == "repaired"
        ):
            raise ValueError("Final acceptance must follow initial or repaired success")
        if self.final_acceptance_passed and not self.final_acceptance_report:
            raise ValueError("Final acceptance success requires its report")

        expected_at_1 = self.generation_succeeded and self.initial_acceptance_passed
        expected_with_repairs = self.generation_succeeded and (
            self.initial_acceptance_passed
            or (
                1 <= self.repairs_attempted <= 3
                and self.repair_final_status == "repaired"
                and self.final_acceptance_passed
            )
        )
        if self.success_at_1 is None:
            self.success_at_1 = expected_at_1
        elif self.success_at_1 != expected_at_1:
            raise ValueError("success_at_1 does not match lifecycle evidence")
        if self.success_after_le_3_repairs is None:
            self.success_after_le_3_repairs = expected_with_repairs
        elif self.success_after_le_3_repairs != expected_with_repairs:
            raise ValueError(
                "success_after_le_3_repairs does not match lifecycle evidence"
            )
        return self


class ConditionSuccessSummary(BaseModel):
    """Success counts and rates for one frozen condition."""

    condition: PipelineConditionName
    repetitions: int = Field(ge=1)
    success_at_1_count: int = Field(ge=0)
    success_at_1_rate: float = Field(ge=0.0, le=1.0)
    success_after_le_3_repairs_count: int = Field(ge=0)
    success_after_le_3_repairs_rate: float = Field(ge=0.0, le=1.0)
    generation_failure_count: int = Field(ge=0)
    repair_attempt_distribution: dict[int, int]


class MatchedGenerationExperimentSummary(BaseModel):
    """Balanced success summary across all primary conditions."""

    schema_version: Literal["pipeline_generation_experiment_summary.v1"] = (
        "pipeline_generation_experiment_summary.v1"
    )
    experiment_id: str
    generated_at: str
    repetitions_per_condition: int = Field(ge=1)
    matched: Literal[True] = True
    trials: list[PipelineConditionTrialResult]
    conditions: list[ConditionSuccessSummary] = Field(min_length=3, max_length=3)


def summarize_matched_trials(
    *,
    experiment_id: str,
    repetitions_per_condition: int,
    trials: list[PipelineConditionTrialResult],
) -> MatchedGenerationExperimentSummary:
    """Validate a balanced matrix and compute success@1 and repaired success."""

    expected = {
        (condition, repetition)
        for condition in PRIMARY_CONDITIONS
        for repetition in range(1, repetitions_per_condition + 1)
    }
    observed = {(trial.condition, trial.repetition) for trial in trials}
    if len(observed) != len(trials):
        raise ValueError("Condition/repetition trial keys must be unique")
    if observed != expected:
        missing = sorted(
            f"{condition.value}:{repetition}"
            for condition, repetition in expected - observed
        )
        extra = sorted(
            f"{condition.value}:{repetition}"
            for condition, repetition in observed - expected
        )
        raise ValueError(f"Experiment matrix is not matched; missing={missing}, extra={extra}")
    if any(trial.experiment_id != experiment_id for trial in trials):
        raise ValueError("All trial experiment_id values must match")

    conditions = []
    for condition in PRIMARY_CONDITIONS:
        selected = sorted(
            (trial for trial in trials if trial.condition == condition),
            key=lambda trial: trial.repetition,
        )
        success_at_1 = sum(bool(trial.success_at_1) for trial in selected)
        success_repaired = sum(
            bool(trial.success_after_le_3_repairs) for trial in selected
        )
        distribution = {
            attempts: sum(trial.repairs_attempted == attempts for trial in selected)
            for attempts in range(4)
        }
        conditions.append(
            ConditionSuccessSummary(
                condition=condition,
                repetitions=repetitions_per_condition,
                success_at_1_count=success_at_1,
                success_at_1_rate=success_at_1 / repetitions_per_condition,
                success_after_le_3_repairs_count=success_repaired,
                success_after_le_3_repairs_rate=(
                    success_repaired / repetitions_per_condition
                ),
                generation_failure_count=sum(
                    not trial.generation_succeeded for trial in selected
                ),
                repair_attempt_distribution=distribution,
            )
        )
    return MatchedGenerationExperimentSummary(
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC).isoformat(),
        repetitions_per_condition=repetitions_per_condition,
        trials=sorted(
            trials, key=lambda trial: (trial.condition.value, trial.repetition)
        ),
        conditions=conditions,
    )


def write_trial_result(path: Path, trial: PipelineConditionTrialResult) -> None:
    _write_new(path, trial.model_dump_json(indent=2) + "\n")


def write_experiment_summary(
    output_dir: Path,
    summary: MatchedGenerationExperimentSummary,
) -> tuple[Path, Path]:
    """Persist machine-readable and concise human-readable matched results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "report.md"
    _write_new(json_path, summary.model_dump_json(indent=2) + "\n")
    lines = [
        "# Pipeline Generation Experiment",
        "",
        f"Experiment: `{summary.experiment_id}`",
        f"Matched repetitions per condition: `{summary.repetitions_per_condition}`",
        "",
        "| Condition | success@1 | success after <=3 repairs | Generation failures |",
        "|---|---:|---:|---:|",
    ]
    for condition in summary.conditions:
        lines.append(
            f"| `{condition.condition.value}` | "
            f"{condition.success_at_1_count}/{condition.repetitions} "
            f"({condition.success_at_1_rate:.1%}) | "
            f"{condition.success_after_le_3_repairs_count}/{condition.repetitions} "
            f"({condition.success_after_le_3_repairs_rate:.1%}) | "
            f"{condition.generation_failure_count} |"
        )
    _write_new(markdown_path, "\n".join([*lines, ""]))
    return json_path, markdown_path


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
