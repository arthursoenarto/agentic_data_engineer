"""Schemas for LLM-generated access probe scripts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AccessProbeScript(BaseModel):
    """A standalone Python probe script generated for one dataset."""

    schema_version: str = "access_probe_script.v1"
    filename: str = "access_probe.py"
    python_code: str = Field(
        ...,
        description="Standalone Python 3 code. It must print only access_probe.v1 JSON to stdout.",
    )
    notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
