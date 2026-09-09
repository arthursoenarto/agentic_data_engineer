"""Deterministic quarantine gates for proposed evaluator check logic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.agents.evaluation_planning.schemas import ProposedEvaluationCheckDraft


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratedCheckCandidate(_StrictModel):
    """Evidence bundle for generated logic that remains outside the catalog."""

    schema_version: Literal["generated_evaluation_check_candidate.v1"] = (
        "generated_evaluation_check_candidate.v1"
    )
    proposal: ProposedEvaluationCheckDraft
    implementation_path: str
    interface_version: Literal["evaluation_check_candidate.v1"]
    independent_oracle_path: str | None = None
    positive_test_ids: list[str] = Field(default_factory=list)
    negative_test_ids: list[str] = Field(default_factory=list)
    mutation_test_ids: list[str] = Field(default_factory=list)
    sandbox_verified: bool = False
    provider_network_disabled: bool = False
    secret_access_disabled: bool = False
    replacement_check_ids: list[str] = Field(default_factory=list)
    proposed_catalog_version: str

    @model_validator(mode="after")
    def cannot_replace_trusted_checks(self) -> "GeneratedCheckCandidate":
        if self.replacement_check_ids:
            raise ValueError("Generated checks cannot replace or weaken trusted checks")
        return self


class CheckPromotionAssessment(_StrictModel):
    """Promotion-gate result; it never mutates the trusted catalog."""

    status: Literal["quarantined", "promotion_gates_passed"]
    failures: list[str]
    catalog_mutated: Literal[False] = False


def assess_generated_check_candidate(
    candidate: GeneratedCheckCandidate,
    *,
    current_catalog_version: str = "evaluation_check_catalog.v1",
) -> CheckPromotionAssessment:
    """Evaluate evidence without executing or registering generated code."""

    failures: list[str] = []
    if not candidate.implementation_path.startswith(
        "project/benchmarks/evaluation_check_candidates/"
    ):
        failures.append("implementation must remain in the quarantined candidate area")
    if not candidate.independent_oracle_path:
        failures.append("independent evaluator-owned oracle is required")
    if not candidate.positive_test_ids:
        failures.append("positive tests are required")
    if not candidate.negative_test_ids:
        failures.append("negative tests are required")
    if not candidate.mutation_test_ids:
        failures.append("mutation tests are required")
    if not candidate.sandbox_verified:
        failures.append("sandbox verification is required")
    if not candidate.provider_network_disabled:
        failures.append("provider network must be disabled")
    if not candidate.secret_access_disabled:
        failures.append("secret access must be disabled")
    if candidate.proposed_catalog_version == current_catalog_version:
        failures.append("promotion requires a new catalog version")
    return CheckPromotionAssessment(
        status="quarantined" if failures else "promotion_gates_passed",
        failures=failures,
    )
