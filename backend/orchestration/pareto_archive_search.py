"""Uniform-parent search over a noise-aware Pareto archive."""

from __future__ import annotations

import random

from backend.orchestration.pareto import ParetoArchive
from backend.orchestration.search_policy import ParentSelection, PolicyObservation
from backend.orchestration.self_improvement_schemas import (
    ObjectiveVector,
    ParetoTolerances,
    SearchPolicyState,
)


class ParetoArchivePolicy:
    """Select parents uniformly from the current non-dominated archive."""

    def __init__(
        self,
        tolerances: ParetoTolerances,
        *,
        objectives=None,
        seed: int = 20260907,
    ) -> None:
        self.archive = ParetoArchive(tolerances, objectives)
        self._random = random.Random(seed)
        self._incumbent_id: str | None = None
        self._nonimprovements = 0

    def initialize(self, node_id: str, objective: ObjectiveVector) -> None:
        if self._incumbent_id is not None:
            raise RuntimeError("Pareto archive policy is already initialized.")
        self.archive.add_root(node_id, objective)
        self._incumbent_id = node_id

    def select_parent(self) -> ParentSelection:
        if self._incumbent_id is None:
            raise RuntimeError("Pareto archive policy is not initialized.")
        node_id = self._random.choice(sorted(self.archive.ids))
        return ParentSelection(node_id=node_id, phase="pareto_archive")

    def observe(
        self,
        selection: ParentSelection,
        node_id: str,
        objective: ObjectiveVector | None,
    ) -> PolicyObservation:
        if selection.phase != "pareto_archive":
            raise ValueError("Pareto archive policy received an invalid selection.")
        if objective is None:
            self._nonimprovements += 1
            return PolicyObservation(
                admitted_to_archive=False,
                reason="Candidate failed search eligibility; archive retained.",
            )
        decision = self.archive.consider(node_id, objective)
        if decision.admitted:
            self._incumbent_id = node_id
            self._nonimprovements = 0
        else:
            self._nonimprovements += 1
        return PolicyObservation(
            admitted_to_archive=decision.admitted,
            reason=decision.reason,
            removed_ids=decision.removed_ids,
        )

    def state(self) -> SearchPolicyState:
        if self._incumbent_id is None:
            raise RuntimeError("Pareto archive policy is not initialized.")
        return SearchPolicyState(
            phase="pareto_archive",
            incumbent_id=self._incumbent_id,
            archive_ids=self.archive.ids,
            consecutive_valid_nonimprovements=self._nonimprovements,
        )
