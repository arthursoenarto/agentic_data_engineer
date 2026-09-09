"""Versioned schemas for immutable pipeline self-improvement searches."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.evaluation.objective_v3_schemas import V3EvaluationSummary


SearchStrategy = Literal["pareto_greedy", "pareto_archive", "adaptive_branching"]
ProposalHistoryMode = Literal["full", "none"]
SearchPhase = Literal["root", "greedy", "pareto_archive", "branching"]
ProposalObjective = Literal[
    "materialization_seconds",
    "consumer_samples_per_second",
    "output_bytes",
    "q_engineering",
]
ProposalFocus = Literal["throughput", "footprint", "joint"]
NodeStatus = Literal[
    "optimization_ready",
    "infeasible",
    "incomplete",
    "proposal_failed",
    "invalid_patch",
    "evaluation_failed",
]
StopReason = Literal[
    "iteration_budget",
    "wall_time_budget",
    "root_not_optimization_ready",
    "workflow_error",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectiveVector(_StrictModel):
    """Measured objectives, with engineering quality optional by protocol."""

    materialization_seconds: float = Field(gt=0)
    consumer_samples_per_second: float = Field(gt=0)
    output_bytes: int = Field(ge=1)
    q_engineering: float | None = Field(default=None, ge=0, le=4)

    @model_validator(mode="after")
    def values_are_finite(self) -> "ObjectiveVector":
        values = (
            self.materialization_seconds,
            self.consumer_samples_per_second,
            float(self.output_bytes),
            *(value for value in (self.q_engineering,) if value is not None),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Objective values must be finite.")
        return self

    @classmethod
    def from_summary(
        cls,
        summary: V3EvaluationSummary,
        *,
        required_objectives: list[ProposalObjective] | None = None,
    ) -> "ObjectiveVector":
        required = required_objectives or [
            "materialization_seconds",
            "consumer_samples_per_second",
            "output_bytes",
            "q_engineering",
        ]
        if not summary.feasible:
            raise ValueError("Only correctness-feasible summaries expose search objectives.")
        operational = summary.operational_objectives
        engineering = summary.engineering_objective
        available = set(operational) | set(engineering)
        missing = set(required) - available
        if missing:
            raise ValueError(f"Evaluation is missing required objectives: {sorted(missing)}")
        return cls(
            materialization_seconds=float(operational["materialization_seconds"]),
            consumer_samples_per_second=float(
                operational["consumer_samples_per_second"]
            ),
            output_bytes=int(operational["output_bytes"]),
            q_engineering=(
                float(engineering["q_engineering"])
                if "q_engineering" in engineering
                else None
            ),
        )

    def values(self) -> dict[str, float]:
        values = {
            "materialization_seconds": self.materialization_seconds,
            "consumer_samples_per_second": self.consumer_samples_per_second,
            "output_bytes": float(self.output_bytes),
        }
        if self.q_engineering is not None:
            values["q_engineering"] = self.q_engineering
        return values


class ProposalFocusAllocation(_StrictModel):
    """Exact attempted-child budget assigned to each bi-objective proposal focus."""

    throughput: int = Field(ge=0)
    footprint: int = Field(ge=0)
    joint: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.throughput + self.footprint + self.joint


class ObjectiveTolerance(_StrictModel):
    """Noise floor used when deciding whether one measurement is meaningful."""

    relative: float = Field(default=0.0, ge=0, lt=1)
    absolute: float = Field(default=0.0, ge=0)


class ParetoTolerances(_StrictModel):
    """Conservative defaults for short development-suite measurements."""

    materialization_seconds: ObjectiveTolerance = Field(
        default_factory=lambda: ObjectiveTolerance(relative=0.02, absolute=0.02)
    )
    consumer_samples_per_second: ObjectiveTolerance = Field(
        default_factory=lambda: ObjectiveTolerance(relative=0.03, absolute=20.0)
    )
    output_bytes: ObjectiveTolerance = Field(
        default_factory=lambda: ObjectiveTolerance(relative=0.005, absolute=64.0)
    )
    q_engineering: ObjectiveTolerance = Field(
        default_factory=lambda: ObjectiveTolerance(relative=0.0, absolute=0.1)
    )

    def for_name(self, name: str) -> ObjectiveTolerance:
        value = getattr(self, name, None)
        if not isinstance(value, ObjectiveTolerance):
            raise KeyError(name)
        return value


class SelfImprovementConfig(_StrictModel):
    """Frozen protocol and budgets for one development-suite search."""

    schema_version: Literal["pipeline_self_improvement_config.v1"] = (
        "pipeline_self_improvement_config.v1"
    )
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,119}$")
    dataset_dir: Path
    root_candidate_dir: Path
    base_suite: Path
    strategy: SearchStrategy
    max_iterations: int = Field(default=4, ge=1, le=50)
    max_wall_time_seconds: int = Field(default=1_800, ge=30, le=86_400)
    operational_repetitions: int = Field(default=3, ge=1, le=9)
    stagnation_patience: int = Field(default=2, ge=1, le=10)
    branch_width: int = Field(default=2, ge=2, le=8)
    max_edits_per_proposal: int = Field(default=3, ge=1, le=4)
    pareto_objectives: list[ProposalObjective] = Field(
        default_factory=lambda: [
            "materialization_seconds",
            "consumer_samples_per_second",
            "output_bytes",
            "q_engineering",
        ],
        min_length=2,
        max_length=4,
    )
    proposal_focus_allocation: ProposalFocusAllocation | None = None
    parent_selection_seed: int = 20260907
    proposal_objective_cycle: list[list[ProposalObjective]] = Field(
        default_factory=lambda: [
            ["materialization_seconds", "consumer_samples_per_second"],
            ["output_bytes"],
            ["materialization_seconds"],
            ["q_engineering"],
            ["consumer_samples_per_second"],
        ],
        min_length=1,
        max_length=20,
    )
    max_documentation_proposals: int = Field(default=1, ge=0, le=10)
    proposal_novelty_threshold: float = Field(default=0.72, ge=0.5, le=1.0)
    proposal_history_mode: ProposalHistoryMode = "full"
    llm_model: str = "gpt-5.5"
    llm_timeout_seconds: int = Field(default=600, ge=1)
    require_extensibility_probe: bool = True
    tolerances: ParetoTolerances = Field(default_factory=ParetoTolerances)

    @field_validator("dataset_dir", "root_candidate_dir", "base_suite")
    @classmethod
    def paths_are_explicit(cls, value: Path) -> Path:
        if not value.parts:
            raise ValueError("Search paths must be non-empty.")
        return value

    @field_validator("proposal_objective_cycle")
    @classmethod
    def objective_cycle_is_focused(
        cls, value: list[list[ProposalObjective]]
    ) -> list[list[ProposalObjective]]:
        for targets in value:
            if not 1 <= len(targets) <= 2:
                raise ValueError("Each proposal objective slot requires one or two targets.")
            if len(set(targets)) != len(targets):
                raise ValueError("Proposal objective targets must be unique per slot.")
        return value

    @model_validator(mode="after")
    def search_protocol_is_coherent(self) -> "SelfImprovementConfig":
        if len(set(self.pareto_objectives)) != len(self.pareto_objectives):
            raise ValueError("pareto_objectives must be unique")
        if self.proposal_focus_allocation is not None:
            if self.proposal_focus_allocation.total != self.max_iterations:
                raise ValueError(
                    "proposal_focus_allocation must exactly equal max_iterations"
                )
            expected = {"consumer_samples_per_second", "output_bytes"}
            if set(self.pareto_objectives) != expected:
                raise ValueError(
                    "proposal focus allocation requires throughput/output Pareto objectives"
                )
        return self


class SearchNodeRecord(_StrictModel):
    """Terminal, immutable record for one complete candidate snapshot."""

    schema_version: Literal["pipeline_search_node.v1"] = "pipeline_search_node.v1"
    node_id: str
    parent_id: str | None
    iteration: int = Field(ge=0)
    depth: int = Field(ge=0)
    phase: SearchPhase
    branch_id: str | None = None
    status: NodeStatus
    candidate_dir: str
    proposal_record: str | None = None
    evaluation_report: str | None = None
    evaluation_seconds: float | None = Field(default=None, ge=0)
    constraints: dict[str, str] = Field(default_factory=dict)
    optimization_ready: bool = False
    objective: ObjectiveVector | None = None
    parent_relation: Literal[
        "dominates", "dominated", "tradeoff", "equivalent", "not_comparable"
    ] = "not_comparable"
    objective_changes: dict[str, float] = Field(default_factory=dict)
    admitted_to_archive: bool = False
    policy_reason: str | None = None
    failure: str | None = None
    created_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def objective_matches_readiness(self) -> "SearchNodeRecord":
        if self.optimization_ready != (self.objective is not None):
            raise ValueError("optimization_ready must match objective availability.")
        if self.status == "optimization_ready" and not self.optimization_ready:
            raise ValueError("Optimization-ready status requires a complete objective.")
        return self


class OperationalRepetition(_StrictModel):
    """One isolated deterministic measurement used in the search median."""

    repetition: int = Field(ge=1)
    evaluation_report: str
    objectives: dict[str, int | float]


class SearchMeasurementReport(_StrictModel):
    """Aggregate F(p) evidence without repeating the qualitative judge."""

    schema_version: Literal["pipeline_search_measurement.v1"] = (
        "pipeline_search_measurement.v1"
    )
    node_id: str
    full_evaluation_report: str
    operational_repetitions: list[OperationalRepetition] = Field(min_length=1)
    aggregate: Literal["median"] = "median"
    objective: ObjectiveVector


class SearchPolicyState(_StrictModel):
    """Serializable policy state for analysis and deterministic resumption work."""

    phase: Literal["greedy", "pareto_archive", "branching"]
    incumbent_id: str
    archive_ids: list[str]
    consecutive_valid_nonimprovements: int = Field(ge=0)
    branches: dict[str, str] = Field(default_factory=dict)


class SelfImprovementRunReport(_StrictModel):
    """Mutable run index whose node records and source snapshots are immutable."""

    schema_version: Literal["pipeline_self_improvement_run.v2"] = (
        "pipeline_self_improvement_run.v2"
    )
    run_id: str
    strategy: SearchStrategy
    status: Literal["running", "completed", "failed"] = "running"
    started_at: datetime
    completed_at: datetime | None = None
    config_path: str
    root_node_id: str | None = None
    node_records: list[str] = Field(default_factory=list)
    policy_state: SearchPolicyState | None = None
    pareto_archive: list[str] = Field(default_factory=list)
    continuation_node_id: str | None = None
    stop_reason: StopReason | None = None
    error: str | None = None


def compact_history_entry(node: SearchNodeRecord) -> Mapping[str, object]:
    """Bounded proposal context derived from durable node evidence."""

    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "phase": node.phase,
        "branch_id": node.branch_id,
        "status": node.status,
        "objective": (
            node.objective.model_dump(mode="json") if node.objective else None
        ),
        "parent_relation": node.parent_relation,
        "objective_changes": node.objective_changes,
        "admitted_to_archive": node.admitted_to_archive,
        "policy_reason": node.policy_reason,
        "failure": node.failure,
    }
