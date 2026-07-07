"""Extractor protocol for dataset inventory generation."""

from __future__ import annotations

from typing import Literal, Protocol

from backend.agents.contract_drafting.schemas import DatasetCandidate
from backend.agents.dataset_inventory.schemas import DatasetInventory


class InventoryExtractor(Protocol):
    """Reusable provider/catalog-specific inventory extractor."""

    name: str
    extraction_method: Literal["deterministic", "llm", "manual"]

    def can_handle(self, candidate: DatasetCandidate) -> bool:
        """Return whether this extractor can inspect the candidate dataset."""

    def extract(self, candidate: DatasetCandidate, *, dataset_slug: str) -> DatasetInventory:
        """Return a provider option-space inventory for the candidate."""
