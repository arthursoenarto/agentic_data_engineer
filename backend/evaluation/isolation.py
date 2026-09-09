"""Portable, explicitly reported isolation for evaluator-launched commands."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from backend.evaluation.core_schemas import (
    CommandExecutionPolicy,
    CommandIsolationEvidence,
)


@dataclass(frozen=True)
class PreparedCommand:
    """Executable command, bounded environment, and honest isolation evidence."""

    command: list[str]
    environment: dict[str, str]
    evidence: CommandIsolationEvidence


def prepare_command(
    command: list[str],
    *,
    cwd: Path,
    writable_root: Path,
    policy: CommandExecutionPolicy,
) -> PreparedCommand:
    """Prepare a local, network-disabled command when the host supports it."""

    resolved_cwd = cwd.resolve()
    resolved_writable = writable_root.resolve()
    environment = _bounded_environment(policy, resolved_writable)
    requested = policy.backend
    backend = requested
    if backend == "auto":
        if platform.system() == "Darwin" and Path("/usr/bin/sandbox-exec").is_file():
            backend = "macos_sandbox"
        elif shutil.which("bwrap"):
            backend = "bubblewrap"
        else:
            backend = "process_only"

    if backend == "macos_sandbox":
        sandbox = Path("/usr/bin/sandbox-exec")
        if platform.system() != "Darwin" or not sandbox.is_file():
            return _unsupported(
                command, environment, requested, "macOS sandbox unavailable"
            )
        profile = " ".join(
            [
                "(version 1)",
                "(deny default)",
                "(allow process*)",
                "(allow file-read*)",
                "(allow sysctl-read)",
                "(allow mach-lookup)",
                "(allow signal)",
                "(allow ipc-posix*)",
                f'(allow file-write* (subpath "{_escape_profile(resolved_writable)}"))',
            ]
        )
        return PreparedCommand(
            command=[str(sandbox), "-p", profile, *command],
            environment=environment,
            evidence=_full_evidence(requested, "macos_sandbox"),
        )

    if backend == "bubblewrap":
        bwrap = shutil.which("bwrap")
        if not bwrap:
            return _unsupported(
                command, environment, requested, "bubblewrap unavailable"
            )
        return PreparedCommand(
            command=[
                bwrap,
                "--die-with-parent",
                "--unshare-net",
                "--ro-bind",
                "/",
                "/",
                "--bind",
                str(resolved_writable),
                str(resolved_writable),
                "--chdir",
                str(resolved_cwd),
                "--",
                *command,
            ],
            environment=environment,
            evidence=_full_evidence(requested, "bubblewrap"),
        )

    return _unsupported(
        command,
        environment,
        requested,
        "Process-group isolation cannot enforce read-only inputs or disable network access.",
    )


def _bounded_environment(
    policy: CommandExecutionPolicy, writable_root: Path
) -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in policy.environment_allowlist
        if name in os.environ and not _looks_sensitive(name)
    }
    for name, value in policy.environment_overrides.items():
        if not _looks_sensitive(name):
            environment[name] = value
    environment["PYTHONFAULTHANDLER"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TMPDIR"] = str(writable_root)
    return environment


def _looks_sensitive(name: str) -> bool:
    upper = name.upper()
    markers = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "CREDENTIAL")
    return any(marker in upper for marker in markers)


def _full_evidence(requested: str, applied: str) -> CommandIsolationEvidence:
    return CommandIsolationEvidence(
        requested_backend=requested,
        applied_backend=applied,
        source_read_only=True,
        trusted_inputs_read_only=True,
        writes_confined=True,
        network_disabled=True,
        supported=True,
    )


def _unsupported(
    command: list[str],
    environment: dict[str, str],
    requested: str,
    limitation: str,
) -> PreparedCommand:
    return PreparedCommand(
        command=command,
        environment=environment,
        evidence=CommandIsolationEvidence(
            requested_backend=requested,
            applied_backend="process_only",
            source_read_only=False,
            trusted_inputs_read_only=False,
            writes_confined=False,
            network_disabled=False,
            supported=False,
            limitation=limitation,
        ),
    )


def _escape_profile(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')
