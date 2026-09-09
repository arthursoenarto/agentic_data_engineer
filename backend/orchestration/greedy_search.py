"""Pareto-aware greedy hill climbing over immutable pipeline candidates."""

from __future__ import annotations

from backend.orchestration.pareto import ParetoArchive
from backend.orchestration.search_policy import (
    ParentSelection,
    PolicyObservation,
)
from backend.orchestration.self_improvement_schemas import (
    ObjectiveVector,
    ParetoTolerances,
    SearchPolicyState,
)


class ParetoGreedyPolicy:
    """Advance to every newly non-dominated child and preserve the full frontier."""

    def __init__(self, tolerances: ParetoTolerances, objectives=None) -> None:
        self.objectives = tuple(objectives or ()) or None
        self.archive = ParetoArchive(tolerances, self.objectives)
        self.all_feasible: dict[str, ObjectiveVector] = {}
        self.incumbent_id: str | None = None
        self.consecutive_valid_nonimprovements = 0

    def initialize(self, node_id: str, objective: ObjectiveVector) -> None:
        if self.incumbent_id is not None:
            raise RuntimeError("Greedy policy is already initialized.")
        self.archive.add_root(node_id, objective)
        self.all_feasible[node_id] = objective
        self.incumbent_id = node_id

    def select_parent(self) -> ParentSelection:
        if self.incumbent_id is None:
            raise RuntimeError("Greedy policy is not initialized.")
        return ParentSelection(node_id=self.incumbent_id, phase="greedy")

    def observe(
        self,
        selection: ParentSelection,
        node_id: str,
        objective: ObjectiveVector | None,
    ) -> PolicyObservation:
        if selection.phase != "greedy":
            raise ValueError("Greedy policy received a non-greedy selection.")
        if objective is None:
            return PolicyObservation(
                admitted_to_archive=False,
                reason="Candidate was not optimization-ready; incumbent retained.",
            )
        self.all_feasible[node_id] = objective
        decision = self.archive.consider(node_id, objective)
        if decision.admitted:
            self.incumbent_id = node_id
            self.consecutive_valid_nonimprovements = 0
        else:
            self.consecutive_valid_nonimprovements += 1
        return PolicyObservation(
            admitted_to_archive=decision.admitted,
            reason=decision.reason,
            removed_ids=decision.removed_ids,
        )

    def state(self) -> SearchPolicyState:
        if self.incumbent_id is None:
            raise RuntimeError("Greedy policy is not initialized.")
        return SearchPolicyState(
            phase="greedy",
            incumbent_id=self.incumbent_id,
            archive_ids=self.archive.ids,
            consecutive_valid_nonimprovements=(
                self.consecutive_valid_nonimprovements
            ),
        )
