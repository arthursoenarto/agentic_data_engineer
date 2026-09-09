"""Schemas for immutable dataset contract locks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.agents.contract_drafting.schemas import DatasetContract


class DatasetContractLock(BaseModel):
    """Canonical execution snapshot produced from an editable contract."""

    schema_version: str = "dataset_contract_lock.v1"
    lock_version: int = Field(ge=1)
    created_at: str
    source_yaml_sha256: str
    inventory_sha256: str
    contract_schema_version: str
    inventory_schema_version: str
    contract: DatasetContract
    intention_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    interaction_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
