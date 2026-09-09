"""Shared policy protocol for pipeline candidate search strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.orchestration.self_improvement_schemas import (
    ObjectiveVector,
    SearchPolicyState,
)


@dataclass(frozen=True)
class ParentSelection:
    node_id: str
    phase: str
    branch_id: str | None = None


@dataclass(frozen=True)
class PolicyObservation:
    admitted_to_archive: bool
    reason: str
    removed_ids: tuple[str, ...] = ()


class SearchPolicy(Protocol):
    def initialize(self, node_id: str, objective: ObjectiveVector) -> None: ...

    def select_parent(self) -> ParentSelection: ...

    def observe(
        self,
        selection: ParentSelection,
        node_id: str,
        objective: ObjectiveVector | None,
    ) -> PolicyObservation: ...

    def state(self) -> SearchPolicyState: ...
