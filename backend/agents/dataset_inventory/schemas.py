"""Schemas for dataset inventory artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class InventoryEvidence(BaseModel):
    """Auditable source used to build an inventory."""

    title: str | None = None
    url: HttpUrl
    note: str | None = None


class DatasetInventory(BaseModel):
    """Provider option space discovered for one candidate dataset."""

    schema_version: str = "dataset_inventory.v1"
    dataset_slug: str
    title: str | None = None
    description: str | None = None
    source_url: HttpUrl
    source_metadata_url: HttpUrl | None = None
    generated_at: str
    provider: str | None = None
    dataset_id: str | None = None
    process_version: str | None = None
    extractor_name: str
    extraction_method: Literal["deterministic", "llm", "manual"]
    note: str | None = None
    catalogue_metadata: dict[str, Any] = Field(default_factory=dict)
    request_fields: list[str] = Field(default_factory=list)
    input_fields: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    option_metadata: dict[str, Any] = Field(default_factory=dict)
    option_units: dict[str, str] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    availability: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    links: list[dict[str, Any]] = Field(default_factory=list)
    job_control_options: list[str] = Field(default_factory=list)
    output_transmission: list[str] = Field(default_factory=list)
    evidence: list[InventoryEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
