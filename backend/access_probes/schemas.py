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


class CredentialEnvVar(BaseModel):
    """Secret-free pointer to an environment variable used for dataset access."""

    purpose: str
    env_var: str
    aliases: list[str] = Field(default_factory=list)
    required: bool = True


class AccessContext(BaseModel):
    """Machine-facing access instructions for pipeline generation, without secrets."""

    schema_version: str = "dataset_access_context.v1"
    dataset_slug: str
    provider: str
    verified: bool = False
    verified_at: str | None = None
    credential_env_vars: list[CredentialEnvVar] = Field(default_factory=list)
    env_loading: dict[str, Any] = Field(default_factory=dict)
    python_client: dict[str, Any] = Field(default_factory=dict)
    source_probe_file: str | None = None
    summary: str | None = None
    notes: list[str] = Field(default_factory=list)
