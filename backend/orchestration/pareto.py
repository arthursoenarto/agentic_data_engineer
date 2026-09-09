"""Noise-aware Pareto comparison and deterministic frontier maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

from backend.orchestration.self_improvement_schemas import (
    ObjectiveTolerance,
    ObjectiveVector,
    ParetoTolerances,
)


Relation = Literal["dominates", "dominated", "tradeoff", "equivalent"]
DIRECTIONS = {
    "materialization_seconds": "minimize",
    "consumer_samples_per_second": "maximize",
    "output_bytes": "minimize",
    "q_engineering": "maximize",
}


@dataclass(frozen=True)
class ArchiveDecision:
    admitted: bool
    removed_ids: tuple[str, ...]
    reason: str


def pareto_relation(
    candidate: ObjectiveVector,
    reference: ObjectiveVector,
    tolerances: ParetoTolerances,
    objectives: Iterable[str] | None = None,
) -> Relation:
    """Compare candidate with reference after applying per-objective noise floors."""

    signs = _comparison_signs(candidate, reference, tolerances, objectives)
    better = any(value > 0 for value in signs.values())
    worse = any(value < 0 for value in signs.values())
    if better and not worse:
        return "dominates"
    if worse and not better:
        return "dominated"
    if better and worse:
        return "tradeoff"
    return "equivalent"


def objective_changes(
    candidate: ObjectiveVector,
    reference: ObjectiveVector,
    objectives: Iterable[str] | None = None,
) -> dict[str, float]:
    """Return direction-normalized relative changes; positive always means better."""

    changes: dict[str, float] = {}
    candidate_values = candidate.values()
    reference_values = reference.values()
    names = tuple(objectives or DIRECTIONS)
    for name in names:
        direction = DIRECTIONS[name]
        old = reference_values[name]
        new = candidate_values[name]
        denominator = max(abs(old), 1e-12)
        raw = (new - old) / denominator
        changes[name] = -raw if direction == "minimize" else raw
    return changes


class ParetoArchive:
    """Keep the non-dominated, non-equivalent objective measurements."""

    def __init__(
        self,
        tolerances: ParetoTolerances,
        objectives: Iterable[str] | None = None,
    ) -> None:
        self._tolerances = tolerances
        self._objectives = tuple(objectives or DIRECTIONS)
        if len(set(self._objectives)) != len(self._objectives):
            raise ValueError("Pareto objectives must be unique.")
        unknown = set(self._objectives) - set(DIRECTIONS)
        if unknown:
            raise ValueError(f"Unknown Pareto objectives: {sorted(unknown)}")
        self._points: dict[str, ObjectiveVector] = {}

    @property
    def ids(self) -> list[str]:
        return list(self._points)

    @property
    def points(self) -> Mapping[str, ObjectiveVector]:
        return dict(self._points)

    def add_root(self, node_id: str, objective: ObjectiveVector) -> None:
        if self._points:
            raise ValueError("Pareto archive root may only be initialized once.")
        self._points[node_id] = objective

    def consider(self, node_id: str, objective: ObjectiveVector) -> ArchiveDecision:
        if node_id in self._points:
            raise ValueError(f"Archive already contains node {node_id!r}.")
        dominated_by: list[str] = []
        equivalent_to: list[str] = []
        removed: list[str] = []
        for existing_id, existing in self._points.items():
            relation = pareto_relation(
                objective,
                existing,
                self._tolerances,
                self._objectives,
            )
            if relation == "dominated":
                dominated_by.append(existing_id)
            elif relation == "equivalent":
                equivalent_to.append(existing_id)
            elif relation == "dominates":
                removed.append(existing_id)
        if dominated_by:
            return ArchiveDecision(
                admitted=False,
                removed_ids=(),
                reason=f"Pareto-dominated by {sorted(dominated_by)}.",
            )
        if equivalent_to:
            return ArchiveDecision(
                admitted=False,
                removed_ids=(),
                reason=f"Within measurement tolerance of {sorted(equivalent_to)}.",
            )
        for existing_id in removed:
            del self._points[existing_id]
        self._points[node_id] = objective
        return ArchiveDecision(
            admitted=True,
            removed_ids=tuple(removed),
            reason=(
                f"Admitted and dominated {sorted(removed)}."
                if removed
                else "Admitted as a non-dominated tradeoff."
            ),
        )


def non_dominated_fronts(
    points: Mapping[str, ObjectiveVector],
    tolerances: ParetoTolerances,
    objectives: Iterable[str] | None = None,
) -> list[list[str]]:
    """Return deterministic Pareto layers for adaptive branch seeding."""

    remaining = dict(points)
    fronts: list[list[str]] = []
    while remaining:
        front = [
            candidate_id
            for candidate_id, candidate in sorted(remaining.items())
            if not any(
                other_id != candidate_id
                and pareto_relation(other, candidate, tolerances, objectives)
                == "dominates"
                for other_id, other in remaining.items()
            )
        ]
        if not front:
            raise RuntimeError("Pareto layering made no progress.")
        fronts.append(front)
        for candidate_id in front:
            del remaining[candidate_id]
    return fronts


def select_branch_seeds(
    points: Mapping[str, ObjectiveVector],
    *,
    width: int,
    preferred_id: str,
    tolerances: ParetoTolerances,
    objectives: Iterable[str] | None = None,
) -> list[str]:
    """Choose strong then objective-diverse candidates without scalarizing F(p)."""

    if preferred_id not in points:
        raise KeyError(preferred_id)
    names = tuple(objectives or DIRECTIONS)
    ranked = [
        item
        for front in non_dominated_fronts(points, tolerances, names)
        for item in front
    ]
    selected = [preferred_id]
    candidates = [item for item in ranked if item != preferred_id]
    while candidates and len(selected) < width:
        scales = _objective_scales(points, names)
        best = max(
            candidates,
            key=lambda item: (
                min(
                    _normalized_distance(points[item], points[chosen], scales, names)
                    for chosen in selected
                ),
                -ranked.index(item),
                item,
            ),
        )
        selected.append(best)
        candidates.remove(best)
    return selected


def _comparison_signs(
    candidate: ObjectiveVector,
    reference: ObjectiveVector,
    tolerances: ParetoTolerances,
    objectives: Iterable[str] | None = None,
) -> dict[str, int]:
    signs: dict[str, int] = {}
    candidate_values = candidate.values()
    reference_values = reference.values()
    names = tuple(objectives or DIRECTIONS)
    for name in names:
        direction = DIRECTIONS[name]
        new = candidate_values[name]
        old = reference_values[name]
        threshold = _threshold(old, tolerances.for_name(name))
        directional_delta = old - new if direction == "minimize" else new - old
        signs[name] = (
            1
            if directional_delta > threshold
            else (-1 if directional_delta < -threshold else 0)
        )
    return signs


def _threshold(reference: float, tolerance: ObjectiveTolerance) -> float:
    return max(tolerance.absolute, abs(reference) * tolerance.relative)


def _objective_scales(
    points: Mapping[str, ObjectiveVector], objectives: Iterable[str] | None = None
) -> dict[str, float]:
    values = [point.values() for point in points.values()]
    names = tuple(objectives or DIRECTIONS)
    return {
        name: max(max(item[name] for item in values) - min(item[name] for item in values), 1e-12)
        for name in names
    }


def _normalized_distance(
    left: ObjectiveVector,
    right: ObjectiveVector,
    scales: Mapping[str, float],
    objectives: Iterable[str] | None = None,
) -> float:
    left_values = left.values()
    right_values = right.values()
    names = tuple(objectives or DIRECTIONS)
    return sum(
        abs(left_values[name] - right_values[name]) / scales[name]
        for name in names
    )
