"""Typed suite and result contracts for station-time-series to Parquet."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.contracts.output_policy import StationTimeSeriesParquetOutputPolicy
from backend.evaluation.check_library import (
    EvaluationProfile,
    ResolvedEvaluationCheckPlan,
    compile_evaluation_check_plan,
)
from backend.evaluation.constrained_schemas import (
    CandidateSourceSpec,
    CommandTemplateSpec,
    FrozenFile,
    FrozenPath,
    ReferenceSpec,
    SecuritySettings,
)
from backend.evaluation.core_schemas import CommandExecutionPolicy, CommandObservation
from backend.evaluation.objective_v3_schemas import (
    EngineeringJudgeSettingsV3,
    EngineeringQualityRunV3,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StationTableSpec(_StrictModel):
    dataset_class: Literal["station_time_series"] = "station_time_series"
    output_format: Literal["parquet"] = "parquet"
    expected_field_ids: list[str] = Field(min_length=1)
    expected_station_ids: list[str] = Field(min_length=1)
    value_atol: float = Field(default=0.0, ge=0.0)
    value_rtol: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> "StationTableSpec":
        if len(self.expected_field_ids) != len(set(self.expected_field_ids)):
            raise ValueError("expected_field_ids must be unique")
        if len(self.expected_station_ids) != len(set(self.expected_station_ids)):
            raise ValueError("expected_station_ids must be unique")
        return self


class StationWorkloadSpec(_StrictModel):
    kind: Literal["filtered_station_scan"] = "filtered_station_scan"
    protocol: Literal[
        "filtered_station_scan.v1",
        "filtered_station_scan.v2",
    ] = "filtered_station_scan.v1"
    repetitions: int = Field(default=5, ge=1, le=30)
    warmup_passes: int = Field(default=1, ge=0, le=5)


class StationParquetEvaluationConfig(_StrictModel):
    schema_version: Literal["evaluation_constrained.station_parquet.v1"] = (
        "evaluation_constrained.station_parquet.v1"
    )
    suite_id: str
    suite_version: str
    candidate_id: str
    contract_lock: FrozenFile
    inventory: FrozenFile
    candidate_source: CandidateSourceSpec
    execution: CommandTemplateSpec
    reference: ReferenceSpec
    table: StationTableSpec
    output_policy: StationTimeSeriesParquetOutputPolicy = Field(
        default_factory=StationTimeSeriesParquetOutputPolicy
    )
    workload: StationWorkloadSpec = Field(default_factory=StationWorkloadSpec)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    execution_policy: CommandExecutionPolicy = Field(
        default_factory=CommandExecutionPolicy
    )
    engineering_quality: EngineeringJudgeSettingsV3 = Field(
        default_factory=EngineeringJudgeSettingsV3
    )
    evaluation_profile: EvaluationProfile | None = None
    check_plan: ResolvedEvaluationCheckPlan | None = None

    @model_validator(mode="after")
    def profile_and_plan_match_adapter(self) -> "StationParquetEvaluationConfig":
        expected = EvaluationProfile(
            profile_id="station-time-series-parquet-v1",
            data_class="station_time_series",
            output_format="parquet",
            output_format_version=1,
            workload_kind=self.workload.kind,
            engineering_profile=self.engineering_quality.profile,
        )
        if self.evaluation_profile is None:
            self.evaluation_profile = expected
        else:
            for name in (
                "data_class",
                "output_format",
                "output_format_version",
                "workload_kind",
                "engineering_profile",
            ):
                if getattr(self.evaluation_profile, name) != getattr(expected, name):
                    raise ValueError(f"evaluation_profile conflicts on {name}")
        plan = compile_evaluation_check_plan(
            self.evaluation_profile,
            engineering_enabled=self.engineering_quality.enabled,
            extensibility_probe_enabled=False,
        )
        if self.check_plan is None:
            self.check_plan = plan
        elif self.check_plan != plan:
            raise ValueError("check_plan differs from deterministic profile resolution")
        return self


class StationEvaluationCheck(_StrictModel):
    check_id: str
    constraint: Literal[
        "contract_correctness",
        "semantic_equivalence",
        "rerun_safety",
        "provenance_security",
    ]
    status: Literal["pass", "fail", "error"]
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class StationScenarioResult(_StrictModel):
    scenario: Literal["initial", "rerun"]
    execution: CommandObservation
    output_path: str
    receipt_path: str
    artifact_path: str | None = None
    validation_error: str | None = None


class StationParquetEvaluationRun(_StrictModel):
    schema_version: Literal["evaluation_result.station_parquet.v1"] = (
        "evaluation_result.station_parquet.v1"
    )
    suite_id: str
    suite_version: str
    candidate_id: str
    run_id: str
    scenarios: list[StationScenarioResult]
    checks: list[StationEvaluationCheck]
    constraints: dict[str, Literal["pass", "fail", "error"]]
    feasible: bool
    operational_objectives: dict[str, float | int] = Field(default_factory=dict)
    engineering_quality: EngineeringQualityRunV3
    objective_vector: dict[str, float | int] = Field(default_factory=dict)
    optimization_ready: bool
    check_plan: ResolvedEvaluationCheckPlan
    provenance: dict[str, Any]
