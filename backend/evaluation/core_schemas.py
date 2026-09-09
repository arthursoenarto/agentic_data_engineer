"""Format-neutral execution evidence shared by evaluation suites."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CommandExecutionPolicy(BaseModel):
    """Evaluator-owned child-process environment and isolation policy."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["auto", "macos_sandbox", "bubblewrap", "process_only"] = "auto"
    environment_allowlist: list[str] = Field(
        default_factory=lambda: [
            "LANG",
            "LC_ALL",
            "PATH",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TZ",
        ]
    )
    environment_overrides: dict[str, str] = Field(default_factory=dict)


class CommandIsolationEvidence(BaseModel):
    """Observed isolation guarantees for one black-box invocation."""

    model_config = ConfigDict(extra="forbid")

    requested_backend: str
    applied_backend: str
    source_read_only: bool
    trusted_inputs_read_only: bool
    writes_confined: bool
    network_disabled: bool
    supported: bool
    limitation: str | None = None

    @property
    def full(self) -> bool:
        """Whether every required v3 security property was enforced."""

        return (
            self.supported
            and self.source_read_only
            and self.trusted_inputs_read_only
            and self.writes_confined
            and self.network_disabled
        )


class CommandResourceUsage(BaseModel):
    """Best-effort process-tree resource diagnostics for a black-box command."""

    peak_rss_bytes: int | None = Field(default=None, ge=0)
    user_cpu_seconds: float | None = Field(default=None, ge=0.0)
    system_cpu_seconds: float | None = Field(default=None, ge=0.0)
    read_bytes: int | None = Field(default=None, ge=0)
    write_bytes: int | None = Field(default=None, ge=0)
    sample_interval_seconds: float = Field(default=0.05, gt=0.0)


class CommandObservation(BaseModel):
    """Neutral evidence from one evaluator-observed pipeline execution."""

    command: list[str]
    executed_command: list[str] | None = None
    cwd: str
    started_at: str
    completed_at: str
    duration_seconds: float = Field(ge=0.0)
    exit_code: int
    timed_out: bool = False
    stdout_log: str
    stderr_log: str
    stdout_tail: str = ""
    stderr_tail: str = ""
    redactions_applied: int = Field(default=0, ge=0)
    environment_variable_names: list[str] = Field(default_factory=list)
    isolation: CommandIsolationEvidence | None = None
    resources: CommandResourceUsage
