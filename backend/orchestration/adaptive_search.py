"""Adaptive search that branches after Pareto-aware greedy stagnation."""

from __future__ import annotations

from dataclasses import dataclass

from backend.orchestration.greedy_search import ParetoGreedyPolicy
from backend.orchestration.pareto import select_branch_seeds
from backend.orchestration.search_policy import ParentSelection, PolicyObservation
from backend.orchestration.self_improvement_schemas import (
    ObjectiveVector,
    ParetoTolerances,
    SearchPolicyState,
)


@dataclass
class _Branch:
    branch_id: str
    incumbent_id: str


class AdaptiveBranchingPolicy:
    """Start greedy, then explore diverse historical candidates round-robin."""

    def __init__(
        self,
        tolerances: ParetoTolerances,
        *,
        stagnation_patience: int = 2,
        branch_width: int = 2,
        objectives=None,
    ) -> None:
        if stagnation_patience < 1:
            raise ValueError("stagnation_patience must be positive.")
        if branch_width < 2:
            raise ValueError("branch_width must be at least two.")
        self._tolerances = tolerances
        self._patience = stagnation_patience
        self._branch_width = branch_width
        self._objectives = tuple(objectives or ()) or None
        self._greedy = ParetoGreedyPolicy(tolerances, self._objectives)
        self._branches: list[_Branch] = []
        self._next_branch = 0

    @property
    def phase(self) -> str:
        return "branching" if self._branches else "greedy"

    def initialize(self, node_id: str, objective: ObjectiveVector) -> None:
        self._greedy.initialize(node_id, objective)

    def select_parent(self) -> ParentSelection:
        if not self._branches:
            return self._greedy.select_parent()
        branch = self._branches[self._next_branch]
        self._next_branch = (self._next_branch + 1) % len(self._branches)
        return ParentSelection(
            node_id=branch.incumbent_id,
            phase="branching",
            branch_id=branch.branch_id,
        )

    def observe(
        self,
        selection: ParentSelection,
        node_id: str,
        objective: ObjectiveVector | None,
    ) -> PolicyObservation:
        if not self._branches:
            observation = self._greedy.observe(selection, node_id, objective)
            if (
                self._greedy.consecutive_valid_nonimprovements >= self._patience
                and len(self._greedy.all_feasible) >= 2
            ):
                self._activate_branches()
            return observation

        if selection.phase != "branching" or selection.branch_id is None:
            raise ValueError("Adaptive branching received an invalid selection.")
        if objective is None:
            return PolicyObservation(
                admitted_to_archive=False,
                reason="Candidate was not optimization-ready; branch retained.",
            )
        self._greedy.all_feasible[node_id] = objective
        decision = self._greedy.archive.consider(node_id, objective)
        branch = next(
            item for item in self._branches if item.branch_id == selection.branch_id
        )
        if decision.admitted:
            branch.incumbent_id = node_id
            self._greedy.incumbent_id = node_id
        return PolicyObservation(
            admitted_to_archive=decision.admitted,
            reason=decision.reason,
            removed_ids=decision.removed_ids,
        )

    def state(self) -> SearchPolicyState:
        greedy_state = self._greedy.state()
        return SearchPolicyState(
            phase="branching" if self._branches else "greedy",
            incumbent_id=greedy_state.incumbent_id,
            archive_ids=greedy_state.archive_ids,
            consecutive_valid_nonimprovements=(
                greedy_state.consecutive_valid_nonimprovements
            ),
            branches={
                branch.branch_id: branch.incumbent_id for branch in self._branches
            },
        )

    def _activate_branches(self) -> None:
        assert self._greedy.incumbent_id is not None
        seeds = select_branch_seeds(
            self._greedy.all_feasible,
            width=self._branch_width,
            preferred_id=self._greedy.incumbent_id,
            tolerances=self._tolerances,
            objectives=self._objectives,
        )
        self._branches = [
            _Branch(branch_id=f"branch_{index:02d}", incumbent_id=node_id)
            for index, node_id in enumerate(seeds, 1)
        ]
