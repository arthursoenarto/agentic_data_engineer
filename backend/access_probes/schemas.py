"""Schemas for deterministic dataset access probes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AccessProbeResult(BaseModel):
    """Result of a lightweight deterministic dataset access check."""

    schema_version: str = "access_probe.v1"
    dataset_slug: str
    provider: str
    ok: bool
    status: str
    checked_at: str
    credential_names: list[str]
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
